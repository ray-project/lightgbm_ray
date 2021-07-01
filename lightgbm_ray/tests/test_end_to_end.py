import os
import shutil
import tempfile

import numpy as np
import unittest

import logging

import lightgbm as lgbm

import ray
from ray.exceptions import RayActorError, RayTaskError

from lightgbm_ray import RayParams, train, RayDMatrix, predict, RayShardingMode
from lightgbm_ray.main import RayXGBoostTrainingError
from xgboost_ray.callback import DistributedCallback

# from sklearn.utils import shuffle

logging.getLogger("lightgbm_ray.main").setLevel(logging.DEBUG)


def _make_callback(tmpdir: str) -> DistributedCallback:
    class TestDistributedCallback(DistributedCallback):
        logdir = tmpdir

        def on_init(self, actor, *args, **kwargs):
            log_file = os.path.join(self.logdir, f"rank_{actor.rank}.log")
            actor.log_fp = open(log_file, "at")
            actor.log_fp.write(f"Actor {actor.rank}: Init\n")
            actor.log_fp.flush()

        def before_data_loading(self, actor, data, *args, **kwargs):
            actor.log_fp.write(f"Actor {actor.rank}: Before loading\n")
            actor.log_fp.flush()

        def after_data_loading(self, actor, data, *args, **kwargs):
            actor.log_fp.write(f"Actor {actor.rank}: After loading\n")
            actor.log_fp.flush()

        def before_train(self, actor, *args, **kwargs):
            actor.log_fp.write(f"Actor {actor.rank}: Before train\n")
            actor.log_fp.flush()

        def after_train(self, actor, result_dict, *args, **kwargs):
            actor.log_fp.write(f"Actor {actor.rank}: After train\n")
            actor.log_fp.flush()

        def before_predict(self, actor, *args, **kwargs):
            actor.log_fp.write(f"Actor {actor.rank}: Before predict\n")
            actor.log_fp.flush()

        def after_predict(self, actor, predictions, *args, **kwargs):
            actor.log_fp.write(f"Actor {actor.rank}: After predict\n")
            actor.log_fp.flush()

    return TestDistributedCallback()


class LGBMRayEndToEndTest(unittest.TestCase):
    """In this test suite we validate Ray-XGBoost multi class prediction.

    First, we validate that XGBoost is able to achieve 100% accuracy on
    a simple training task.

    Then we split the dataset into two halves. These halves don't have access
    to all relevant data, so overfit on their respective data. I.e. the first
    half always predicts feature 2 -> label 2, while the second half always
    predicts feature 2 -> label 3.

    We then train using Ray XGBoost. Again both halves will be trained
    separately, but because of Rabit's allreduce, they should end up being
    able to achieve 100% accuracy, again."""

    def setUp(self):
        repeat = 64  # Repeat data a couple of times for stability
        self.x = np.array([
            [1, 0, 0, 0],  # Feature 0 -> Label 0
            [0, 1, 0, 0],  # Feature 1 -> Label 1
            [0, 0, 1, 1],  # Feature 2+3 -> Label 2
            [0, 0, 1, 0],  # Feature 2+!3 -> Label 3
        ] * repeat)
        self.y = np.array([0, 1, 2, 3] * repeat)

        # self.x, self.y = shuffle(self.x, self.y, random_state=1)

        self.params = {
            "boosting": "gbdt",
            "objective": "multiclass",
            "num_class": 4,
            "random_state": 1,
            "tree_learner": "data",
            "deterministic": True,
        }

    def tearDown(self):
        ray.shutdown()

    def testSingleTraining(self):
        """Test that XGBoost learns to predict full matrix"""
        dtrain = lgbm.Dataset(self.x, self.y)
        bst = lgbm.train(self.params, dtrain, num_boost_round=2)

        pred_y = np.argmax(bst.predict(self.x), axis=1)
        self.assertSequenceEqual(list(self.y), list(pred_y))

    def testHalfTraining(self):
        """Test that XGBoost learns to predict half matrices individually"""
        x_first = self.x[::2]
        y_first = self.y[::2]

        x_second = self.x[1::2]
        y_second = self.y[1::2]

        # Test case: The first model only sees feature 2 --> label 2
        # and the second model only sees feature 2 --> label 3
        test_X = np.array([[0, 0, 1, 1], [0, 0, 1, 0]])
        test_y_first = [2, 2]
        test_y_second = [3, 3]

        # First half
        dtrain = lgbm.Dataset(x_first, y_first)
        bst = lgbm.train(self.params, dtrain, num_boost_round=2)

        pred_y = np.argmax(bst.predict(x_first), axis=1)
        self.assertSequenceEqual(list(y_first), list(pred_y))

        pred_test = np.argmax(bst.predict(test_X), axis=1)
        self.assertSequenceEqual(test_y_first, list(pred_test))

        # Second half
        dtrain = lgbm.Dataset(x_second, y_second)
        bst = lgbm.train(self.params, dtrain, num_boost_round=2)

        pred_y = np.argmax(bst.predict(x_second), axis=1)
        self.assertSequenceEqual(list(y_second), list(pred_y))

        pred_test = np.argmax(bst.predict(test_X), axis=1)
        self.assertSequenceEqual(test_y_second, list(pred_test))

    def _testJointTraining(self, cpus_per_actor):
        ray.init(num_cpus=4, num_gpus=0)

        bst = train(
            self.params,
            RayDMatrix(self.x, self.y, sharding=RayShardingMode.BATCH),
            num_boost_round=50,
            ray_params=RayParams(num_actors=2, cpus_per_actor=cpus_per_actor))

        self.assertEqual(bst.booster_.current_iteration(), 50)

        pred_y = bst.predict(self.x)
        pred_y = np.argmax(pred_y, axis=1)
        self.assertSequenceEqual(list(self.y), list(pred_y))

        pred_y = predict(
            bst,
            RayDMatrix(self.x),
            ray_params=RayParams(num_actors=2, cpus_per_actor=cpus_per_actor))
        pred_y = np.argmax(pred_y, axis=1)
        self.assertSequenceEqual(list(self.y), list(pred_y))

        pred_y = predict(
            bst.booster_,
            RayDMatrix(self.x),
            ray_params=RayParams(num_actors=2, cpus_per_actor=cpus_per_actor))
        pred_y = np.argmax(pred_y, axis=1)
        self.assertSequenceEqual(list(self.y), list(pred_y))

    def testJointTraining(self):
        """Train with Ray. The data will be split, but the trees
        should be combined together and find the true model."""
        return self._testJointTraining(cpus_per_actor=2)

    def testJointTrainingDefaultRayParams(self):
        """Train with Ray. The data will be split, but the trees
        should be combined together and find the true model."""
        return self._testJointTraining(cpus_per_actor=0)

    def testCpusPerActorEqualTo1RaisesException(self):
        ray.init(num_cpus=4, num_gpus=0)
        with self.assertRaisesRegex(ValueError,
                                    "cpus_per_actor is set to less than 2"):
            train(
                self.params,
                RayDMatrix(self.x, self.y),
                num_boost_round=50,
                ray_params=RayParams(num_actors=2, cpus_per_actor=1))

    def testBothEvalsAndValidSetsRaisesException(self):
        ray.init(num_cpus=4, num_gpus=0)
        with self.assertRaisesRegex(
                ValueError,
                "Specifying both `evals` and `valid_sets` is ambiguous"):
            data = RayDMatrix(self.x, self.y),
            train(
                self.params,
                data,
                num_boost_round=50,
                ray_params=RayParams(num_actors=2, cpus_per_actor=1),
                evals=[(data, "eval")],
                valid_sets=[data])

    def testTrainPredict(self, init=True, remote=None, **ray_param_dict):
        """Train with evaluation and predict"""
        if init:
            ray.init(num_cpus=4, num_gpus=0)

        dtrain = RayDMatrix(self.x, self.y, sharding=RayShardingMode.BATCH)

        params = self.params

        evals_result = {}
        bst = train(
            params,
            dtrain,
            num_boost_round=38,
            ray_params=RayParams(
                num_actors=2, cpus_per_actor=2, **ray_param_dict),
            evals=[(dtrain, "dtrain")],
            evals_result=evals_result,
            _remote=remote)

        self.assertTrue("dtrain" in evals_result)

        evals_result = {}
        bst = train(
            params,
            dtrain,
            num_boost_round=38,
            ray_params=RayParams(
                num_actors=2, cpus_per_actor=2, **ray_param_dict),
            valid_sets=[dtrain],
            valid_names=["dtrain"],
            evals_result=evals_result,
            _remote=remote)

        self.assertTrue("dtrain" in evals_result)

        x_mat = RayDMatrix(self.x)
        pred_y = predict(
            bst,
            x_mat,
            ray_params=RayParams(num_actors=2, **ray_param_dict),
            _remote=remote)

        self.assertEqual(pred_y.shape[1], len(np.unique(self.y)))
        pred_y = np.argmax(pred_y, axis=1)

        self.assertSequenceEqual(list(self.y), list(pred_y))

    def testTrainPredictRemote(self):
        """Train with evaluation and predict in a remote call"""
        self.testTrainPredict(init=True, remote=True)

    def testTrainPredictClient(self):
        """Train with evaluation and predict in a client session"""
        if ray.__version__ <= "1.2.0":
            self.skipTest("Ray client mocks do not work in Ray <= 1.2.0")
        from ray.util.client.ray_client_helpers import ray_start_client_server

        # (yard1) this hangs when num_cpus=2
        ray.init(num_cpus=5, num_gpus=0)
        self.assertFalse(ray.util.client.ray.is_connected())
        with ray_start_client_server():
            self.assertTrue(ray.util.client.ray.is_connected())

            self.testTrainPredict(init=False, remote=None)

    def testDistributedCallbacksTrainPredict(self, init=True, remote=False):
        """Test distributed callbacks for train/predict"""
        tmpdir = tempfile.mkdtemp()
        test_callback = _make_callback(tmpdir)

        self.testTrainPredict(
            init=init, remote=remote, distributed_callbacks=[test_callback])
        rank_0_log_file = os.path.join(tmpdir, "rank_0.log")
        rank_1_log_file = os.path.join(tmpdir, "rank_1.log")
        self.assertTrue(os.path.exists(rank_1_log_file))

        rank_0_log = open(rank_0_log_file, "rt").read()
        self.assertEqual(
            rank_0_log, "Actor 0: Init\n"
            "Actor 0: Before loading\n"
            "Actor 0: After loading\n"
            "Actor 0: Before train\n"
            "Actor 0: After train\n"
            "Actor 0: Init\n"
            "Actor 0: Before loading\n"
            "Actor 0: After loading\n"
            "Actor 0: Before train\n"
            "Actor 0: After train\n"
            "Actor 0: Init\n"
            "Actor 0: Before loading\n"
            "Actor 0: After loading\n"
            "Actor 0: Before predict\n"
            "Actor 0: After predict\n")
        shutil.rmtree(tmpdir)

    def testDistributedCallbacksTrainPredictClient(self):
        """Test distributed callbacks for train/predict via Ray client"""

        if ray.__version__ <= "1.2.0":
            self.skipTest("Ray client mocks do not work in Ray <= 1.2.0")
        from ray.util.client.ray_client_helpers import ray_start_client_server

        ray.init(num_cpus=5, num_gpus=0)
        self.assertFalse(ray.util.client.ray.is_connected())
        with ray_start_client_server():
            self.assertTrue(ray.util.client.ray.is_connected())

            self.testDistributedCallbacksTrainPredict(init=False, remote=None)

    def testFailPrintErrors(self):
        """Test that XGBoost training errors are propagated"""
        x = np.random.uniform(0, 1, size=(100, 4))
        y = np.random.randint(0, 2, size=100)

        train_set = RayDMatrix(x, y)

        try:
            train(
                {
                    **self.params,
                    **{
                        "num_class": 2,
                        "metric": ["multi_logloss", "multi_error"]
                    }
                },  # This will error
                train_set,
                evals=[(train_set, "train")],
                ray_params=RayParams(
                    num_actors=1, cpus_per_actor=2, max_actor_restarts=0))
        except RuntimeError as exc:
            self.assertTrue(exc.__cause__)
            self.assertTrue(isinstance(exc.__cause__, RayActorError))

            self.assertTrue(exc.__cause__.__cause__)
            self.assertTrue(isinstance(exc.__cause__.__cause__, RayTaskError))

            self.assertTrue(exc.__cause__.__cause__.cause)
            self.assertTrue(
                isinstance(exc.__cause__.__cause__.cause,
                           RayXGBoostTrainingError))

            self.assertIn("label and prediction size not match",
                          str(exc.__cause__.__cause__))


class LGBMRayEndToEndTestVoting(LGBMRayEndToEndTest):
    def setUp(self):
        super().setUp()
        self.params["tree_learner"] = "voting"


if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main(["-v", __file__]))
