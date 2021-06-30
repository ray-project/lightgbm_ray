import os
import shutil
import tempfile
import time
from unittest.mock import patch, DEFAULT
import lightgbm

import numpy as np
import unittest
from lightgbm import LGBMModel
from sklearn.utils import shuffle

import ray

from lightgbm_ray import train, RayDMatrix, RayParams
from xgboost_ray.session import get_actor_rank, put_queue

from xgboost_ray.tests.utils import flatten_obj


def get_num_trees(model_or_booster):
    if isinstance(model_or_booster, LGBMModel):
        return model_or_booster.booster_.current_iteration()
    return model_or_booster.current_iteration()


def _kill_callback(die_lock_file: str,
                   actor_rank: int = 0,
                   fail_iteration: int = 6):
    """Returns a callback to kill an actor process.

    Args:
        die_lock_file (str): A file lock used to prevent race conditions
            when killing the actor.
        actor_rank (int): The rank of the actor to kill.
        fail_iteration (int): The iteration after which the actor is killed.

    """

    def _callback(env):
        if get_actor_rank() == actor_rank:
            put_queue((env.iteration, time.time()))
        if get_actor_rank() == actor_rank and \
                env.iteration == fail_iteration and \
                not os.path.exists(die_lock_file):

            # Get PID
            pid = os.getpid()
            print(f"Killing process: {pid}")
            with open(die_lock_file, "wt") as fp:
                fp.write("")

            time.sleep(2)
            print(f"Testing: Rank {get_actor_rank()} will now die.")
            os.kill(pid, 9)

    _callback.order = 10  # type: ignore
    return _callback


def _checkpoint_callback(frequency: int = 1, before_iteration_=False):
    """Returns a callback to checkpoint a model.

    Args:
        frequency (int): The interval at which checkpointing occurs. If
            frequency is set to n, checkpointing occurs every n epochs.
        before_iteration_ (bool): If True, checkpoint before the iteration
            begins. Else, checkpoint after the iteration ends.

    """

    def _callback(env):
        if env.iteration % frequency == 0:
            put_queue(env.model.model_to_string())

    _callback.before_iteration = before_iteration_
    return _callback


def _fail_callback(die_lock_file: str,
                   actor_rank: int = 0,
                   fail_iteration: int = 6):
    """Returns a callback to cause an Xgboost actor to fail training.

    Args:
        die_lock_file (str): A file lock used to prevent race conditions
            when causing the actor to fail.
        actor_rank (int): The rank of the actor to fail.
        fail_iteration (int): The iteration after which the training for
            the specified actor fails.

    """

    def _callback(env):
        if get_actor_rank() == actor_rank:
            put_queue((env.iteration, time.time()))
        if get_actor_rank(
        ) == actor_rank and env.iteration == fail_iteration \
                and not os.path.exists(die_lock_file):

            with open(die_lock_file, "wt") as fp:
                fp.write("")
            time.sleep(2)
            import sys
            print(f"Testing: Rank {get_actor_rank()} will now fail.")
            sys.exit(1)

    return _callback


class LightGBMRayFaultToleranceTest(unittest.TestCase):
    """In this test suite we validate fault tolerance when a Ray actor dies.

    For this, we set up a callback that makes one worker die exactly once.
    """

    def setUp(self):
        repeat = 64  # Repeat data a couple of times for stability
        self.x = np.array([
            [1, 0, 0, 0],  # Feature 0 -> Label 0
            [0, 1, 0, 0],  # Feature 1 -> Label 1
            [0, 0, 1, 1],  # Feature 2+3 -> Label 2
            [0, 0, 1, 0],  # Feature 2+!3 -> Label 3
        ] * repeat)
        self.y = np.array([0, 1, 2, 3] * repeat)

        self.x, self.y = shuffle(self.x, self.y, random_state=1)

        self.params = {
            "nthread": 2,
            "max_depth": 2,
            "num_leaves": 2,
            "tree_learner": "data",
            "objective": "multiclass",
            "num_class": 4,
            "random_state": 1,
            "deterministic": True,
            "time_out": 1,
        }

        self.tmpdir = str(tempfile.mkdtemp())

        self.die_lock_file = "/tmp/died_worker.lock"
        if os.path.exists(self.die_lock_file):
            os.remove(self.die_lock_file)

        self.die_lock_file_2 = "/tmp/died_worker_2.lock"
        if os.path.exists(self.die_lock_file_2):
            os.remove(self.die_lock_file_2)

    def tearDown(self) -> None:
        if os.path.exists(self.tmpdir):
            shutil.rmtree(self.tmpdir)
        ray.shutdown()

        if os.path.exists(self.die_lock_file):
            os.remove(self.die_lock_file)

        if os.path.exists(self.die_lock_file_2):
            os.remove(self.die_lock_file_2)

    def testTrainingContinuationKilled(self):
        """This should continue after one actor died."""
        ray.init(num_cpus=4, num_gpus=0, log_to_driver=True)
        additional_results = {}
        keep_actors = {}

        def keep(actors, *args, **kwargs):
            keep_actors["actors"] = actors.copy()
            return DEFAULT

        with patch("lightgbm_ray.main._shutdown") as mocked:
            mocked.side_effect = keep
            bst = train(
                self.params,
                RayDMatrix(self.x, self.y),
                callbacks=[_kill_callback(self.die_lock_file)],
                num_boost_round=50,
                ray_params=RayParams(
                    max_actor_restarts=1, num_actors=2, cpus_per_actor=2),
                additional_results=additional_results)

        self.assertEqual(50, get_num_trees(bst))

        pred_y = bst.predict(self.x)
        pred_y = np.argmax(pred_y, axis=1)
        self.assertSequenceEqual(list(self.y), list(pred_y))
        print(f"Got correct predictions: {pred_y}")

        actors = keep_actors["actors"]
        # End with two working actors
        self.assertTrue(actors[0])
        self.assertTrue(actors[1])

        # Two workers finished, so N=64*4
        self.assertEqual(additional_results["total_n"], 64 * 4)

    def testTrainingStop(self):
        """This should now stop training after one actor died."""
        # The `train()` function raises a RuntimeError
        ray.init(num_cpus=4, num_gpus=0, log_to_driver=True)
        with self.assertRaises(RuntimeError):
            train(
                self.params,
                RayDMatrix(self.x, self.y),
                callbacks=[_kill_callback(self.die_lock_file)],
                num_boost_round=20,
                ray_params=RayParams(max_actor_restarts=0, num_actors=2))

    def testCheckpointContinuationValidity(self):
        """Test that checkpoints are stored and loaded correctly"""

        ray.init(num_cpus=4, num_gpus=0, log_to_driver=True)
        # Train once, get checkpoint via callback returns
        res_1 = {}
        train(
            self.params,
            RayDMatrix(self.x, self.y),
            callbacks=[
                _checkpoint_callback(frequency=1, before_iteration_=False)
            ],
            num_boost_round=2,
            ray_params=RayParams(num_actors=2, cpus_per_actor=2),
            additional_results=res_1)
        last_checkpoint_1 = res_1["callback_returns"][0][-1]

        lc1 = lightgbm.Booster(model_str=last_checkpoint_1)

        # Start new training run, starting from existing model
        res_2 = {}
        train(
            self.params,
            RayDMatrix(self.x, self.y),
            callbacks=[
                _checkpoint_callback(frequency=1, before_iteration_=True),
                _checkpoint_callback(frequency=1, before_iteration_=False)
            ],
            num_boost_round=4,
            ray_params=RayParams(num_actors=2, cpus_per_actor=2),
            additional_results=res_2,
            init_model=lc1)
        first_checkpoint_2 = res_2["callback_returns"][0][0]
        last_checkpoint_2 = res_2["callback_returns"][0][-1]

        fcp_bst = lightgbm.Booster(model_str=first_checkpoint_2)

        lcp_bst = lightgbm.Booster(model_str=last_checkpoint_2)

        # Training should not have proceeded for the first checkpoint,
        # so trees should be equal
        self.assertEqual(lc1.current_iteration(), fcp_bst.current_iteration())

        # Training should have proceeded for the last checkpoint,
        # so trees should not be equal
        self.assertNotEqual(fcp_bst.model_to_string(),
                            lcp_bst.model_to_string())

    def testSameResultWithAndWithoutError(self):
        """Get the same model with and without errors during training."""

        ray.init(num_cpus=5, num_gpus=0, log_to_driver=True)
        # Run training
        print("test no error")
        bst_noerror = train(
            self.params,
            RayDMatrix(self.x, self.y),
            num_boost_round=10,
            ray_params=RayParams(
                max_actor_restarts=0, num_actors=2, cpus_per_actor=2))

        print("test part 1")
        bst_2part_1 = train(
            self.params,
            RayDMatrix(self.x, self.y),
            num_boost_round=5,
            ray_params=RayParams(
                max_actor_restarts=0, num_actors=2, cpus_per_actor=2))

        print("test part 2")
        bst_2part_2 = train(
            self.params,
            RayDMatrix(self.x, self.y),
            num_boost_round=5,
            ray_params=RayParams(
                max_actor_restarts=0, num_actors=2, cpus_per_actor=2),
            init_model=bst_2part_1)

        print("test error")
        res_error = {}
        bst_error = train(
            self.params,
            RayDMatrix(self.x, self.y),
            callbacks=[_fail_callback(self.die_lock_file, fail_iteration=7)],
            num_boost_round=10,
            ray_params=RayParams(
                max_actor_restarts=1,
                num_actors=2,
                checkpoint_frequency=5,
                cpus_per_actor=2),
            additional_results=res_error)

        self.assertEqual(bst_error.booster_.current_iteration(),
                         bst_noerror.booster_.current_iteration())
        self.assertEqual(bst_2part_2.booster_.current_iteration(),
                         bst_noerror.booster_.current_iteration())

        flat_noerror = flatten_obj({"tree": bst_noerror.booster_.dump_model()})
        flat_error = flatten_obj({"tree": bst_error.booster_.dump_model()})
        flat_2part = flatten_obj({"tree": bst_2part_2.booster_.dump_model()})

        for key in flat_noerror:
            self.assertAlmostEqual(
                flat_noerror[key], flat_error[key], places=4)
            self.assertAlmostEqual(
                flat_noerror[key], flat_2part[key], places=4)

        # We fail at iteration 7, but checkpoints are saved at iteration 5
        # Thus we have two additional returns here.
        print("Callback returns:", res_error["callback_returns"][0])
        self.assertEqual(len(res_error["callback_returns"][0]), 10 + 2)


if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main(["-v", __file__]))
