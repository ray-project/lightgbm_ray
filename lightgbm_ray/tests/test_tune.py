import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import ray
from ray import tune
try:
    from ray.tune.integration.lightgbm import \
        TuneReportCallback as OrigTuneReportCallback, \
        TuneReportCheckpointCallback as OrigTuneReportCheckpointCallback
except ImportError:
    OrigTuneReportCallback = OrigTuneReportCheckpointCallback = \
        None

from lightgbm_ray import RayDMatrix, train, RayParams, RayShardingMode
from lightgbm_ray.tune import TuneReportCallback,\
    TuneReportCheckpointCallback, _try_add_tune_callback


class LightGBMRayTuneTest(unittest.TestCase):
    def setUp(self):
        repeat = 64  # Repeat data a couple of times for stability
        x = np.array([
            [1, 0, 0, 0],  # Feature 0 -> Label 0
            [0, 1, 0, 0],  # Feature 1 -> Label 1
            [0, 0, 1, 1],  # Feature 2+3 -> Label 2
            [0, 0, 1, 0],  # Feature 2+!3 -> Label 3
        ] * repeat)
        y = np.array([0, 1, 2, 3] * repeat)

        self.params = {
            "lgbm": {
                "boosting": "gbdt",
                "objective": "multiclass",
                "num_class": 4,
                "random_state": 1,
                "tree_learner": "data",
                "metrics": ["multi_logloss", "multi_error"]
            },
            "num_boost_round": tune.choice([1, 3])
        }

        def train_func(ray_params, callbacks=None, **kwargs):
            def _inner_train(config, checkpoint_dir):
                train_set = RayDMatrix(x, y, sharding=RayShardingMode.BATCH)
                train(
                    config["lgbm"],
                    dtrain=train_set,
                    ray_params=ray_params,
                    num_boost_round=config["num_boost_round"],
                    evals=[(train_set, "train")],
                    callbacks=callbacks,
                    **kwargs)

            return _inner_train

        self.train_func = train_func
        self.experiment_dir = tempfile.mkdtemp()

    def tearDown(self):
        ray.shutdown()
        shutil.rmtree(self.experiment_dir)

    # noinspection PyTypeChecker
    def testNumIters(self, init=True):
        """Test that the number of reported tune results is correct"""
        if init:
            ray.init(num_cpus=8)
        ray_params = RayParams(cpus_per_actor=2, num_actors=2)
        analysis = tune.run(
            self.train_func(ray_params),
            config=self.params,
            resources_per_trial=ray_params.get_tune_resources(),
            num_samples=2)

        self.assertSequenceEqual(
            list(analysis.results_df["training_iteration"]),
            list(analysis.results_df["config.num_boost_round"]))

    def testNumItersClient(self):
        """Test ray client mode"""
        ray.init(num_cpus=8)
        if ray.__version__ <= "1.2.0":
            self.skipTest("Ray client mocks do not work in Ray <= 1.2.0")

        from ray.util.client.ray_client_helpers import ray_start_client_server

        self.assertFalse(ray.util.client.ray.is_connected())
        with ray_start_client_server():
            self.assertTrue(ray.util.client.ray.is_connected())
            self.testNumIters(init=False)

    @unittest.skipIf(OrigTuneReportCallback is None,
                     "integration.lightgbmnot yet in ray.tune")
    def testReplaceTuneCheckpoints(self):
        """Test if ray.tune.integration.lightgbm callbacks are replaced"""
        ray.init(num_cpus=4)
        # Report callback
        in_cp = [OrigTuneReportCallback(metrics="met")]
        in_dict = {"callbacks": in_cp}

        with patch("lightgbm_ray.tune.is_session_enabled") as mocked:
            mocked.return_value = True
            _try_add_tune_callback(in_dict)

        replaced = in_dict["callbacks"][0]
        self.assertTrue(isinstance(replaced, TuneReportCallback))
        self.assertSequenceEqual(replaced._metrics, ["met"])

        # Report and checkpointing callback
        in_cp = [
            OrigTuneReportCheckpointCallback(metrics="met", filename="test")
        ]
        in_dict = {"callbacks": in_cp}

        with patch("lightgbm_ray.tune.is_session_enabled") as mocked:
            mocked.return_value = True
            _try_add_tune_callback(in_dict)

        replaced = in_dict["callbacks"][0]
        self.assertTrue(isinstance(replaced, TuneReportCheckpointCallback))
        self.assertSequenceEqual(replaced._report._metrics, ["met"])
        self.assertEqual(replaced._checkpoint._filename, "test")

    def testEndToEndCheckpointing(self):
        ray.init(num_cpus=4)
        ray_params = RayParams(cpus_per_actor=2, num_actors=1)
        analysis = tune.run(
            self.train_func(
                ray_params,
                callbacks=[TuneReportCheckpointCallback(frequency=1)]),
            config=self.params,
            resources_per_trial=ray_params.get_tune_resources(),
            num_samples=1,
            metric="train-multi_logloss",
            mode="min",
            log_to_file=True,
            local_dir=self.experiment_dir)

        self.assertTrue(os.path.exists(analysis.best_checkpoint))

    @unittest.skipIf(OrigTuneReportCallback is None,
                     "integration.lightgbmnot yet in ray.tune")
    def testEndToEndCheckpointingOrigTune(self):
        ray.init(num_cpus=4)
        ray_params = RayParams(cpus_per_actor=2, num_actors=1)
        analysis = tune.run(
            self.train_func(
                ray_params, callbacks=[OrigTuneReportCheckpointCallback()]),
            config=self.params,
            resources_per_trial=ray_params.get_tune_resources(),
            num_samples=1,
            metric="train-multi_logloss",
            mode="min",
            log_to_file=True,
            local_dir=self.experiment_dir)

        self.assertTrue(os.path.exists(analysis.best_checkpoint))


if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main(["-v", __file__]))
