from typing import Tuple

import unittest

import numpy as np

import lightgbm
from lightgbm.callback import CallbackEnv

import ray

from lightgbm_ray import RayDMatrix, train, RayParams, RayShardingMode
from lightgbm_ray.tune import _TuneLGBMRank0Mixin

from xgboost_ray.session import put_queue


def gradient(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return (np.log1p(y_pred) - np.log1p(y_true)) / (y_pred + 1)


def hessian(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return (
        (-np.log1p(y_pred) + np.log1p(y_true) + 1) / np.power(y_pred + 1, 2))


def squared_log(y_true: np.ndarray,
                y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y_pred[y_pred < -1] = -1 + 1e-6
    grad = gradient(y_pred, y_true)
    hess = hessian(y_pred, y_true)
    return grad, hess


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[str, float]:
    y_pred[y_pred < -1] = -1 + 1e-6
    elements = np.power(np.log1p(y_true) - np.log1p(y_pred), 2)
    return "PyRMSLE", float(np.sqrt(np.sum(elements) / len(y_true))), False


class LightGBMAPITest(unittest.TestCase):
    """This test suite validates core LightGBM API functionality."""

    def setUp(self):
        repeat = 128  # Repeat data a couple of times for stability
        self.x = np.array([
            [1, 0, 0, 0],  # Feature 0 -> Label 0
            [0, 1, 0, 0],  # Feature 1 -> Label 1
            [0, 0, 1, 1],  # Feature 2+3 -> Label 0
            [0, 0, 1, 0],  # Feature 2+!3 -> Label 1
        ] * repeat)
        self.y = np.array([0, 1, 0, 1] * repeat)

        self.params = {
            "nthread": 2,
            "objective": "binary",
            "random_state": 1000,
            "deterministic": True,
        }

        self.kwargs = {}

    def tearDown(self) -> None:
        ray.shutdown()

    def _init_ray(self):
        ray.init(num_cpus=4, num_gpus=0)

    def testCustomObjectiveFunction(self):
        """Ensure that custom objective functions work.

        Runs a custom objective function with pure LightGBM and
        LightGBM on Ray and compares the prediction outputs."""
        self._init_ray()

        params = self.params.copy()
        params["objective"] = squared_log

        model_lgbm = lightgbm.LGBMModel(**params).fit(self.x, self.y)

        model_ray = train(
            params,
            RayDMatrix(self.x, self.y, sharding=RayShardingMode.BATCH),
            ray_params=RayParams(num_actors=2),
            num_boost_round=100,
            **self.kwargs)

        pred_y_lgbm = np.round(model_lgbm.predict(self.x))
        pred_y_ray = np.round(model_ray.predict(self.x))

        self.assertSequenceEqual(list(pred_y_lgbm), list(pred_y_ray))
        self.assertSequenceEqual(
            list(self.y.astype(float)), list(pred_y_ray * -1))

    def testCustomMetricFunction(self):
        """Ensure that custom objective functions work.

        Runs a custom objective function with pure LightGBM and
        LightGBM on Ray and compares the prediction outputs."""
        self._init_ray()

        params = self.params.copy()
        params["objective"] = squared_log

        model_lgbm = lightgbm.LGBMModel(**params).fit(
            self.x,
            self.y,
            eval_metric=[rmsle],
            eval_set=[(self.x, self.y)],
            eval_names=["dtrain"])
        evals_result_lgbm = model_lgbm.evals_result_

        dtrain_ray = RayDMatrix(self.x, self.y, sharding=RayShardingMode.BATCH)
        evals_result_ray = {}
        train(
            params,
            dtrain_ray,
            ray_params=RayParams(num_actors=2),
            eval_metric=[rmsle],
            evals=[(dtrain_ray, "dtrain")],
            evals_result=evals_result_ray,
            num_boost_round=100,
            **self.kwargs)

        print(evals_result_ray["dtrain"]["PyRMSLE"])
        print(evals_result_lgbm["dtrain"]["PyRMSLE"])

        self.assertTrue(
            np.allclose(
                evals_result_lgbm["dtrain"]["PyRMSLE"],
                evals_result_ray["dtrain"]["PyRMSLE"],
                atol=0.1))

    def testCallbacks(self):
        self._init_ray()

        class _Callback(_TuneLGBMRank0Mixin):
            def __call__(self, env: CallbackEnv) -> None:
                print(f"My rank: {self.is_rank_0}")
                put_queue(("rank", self.is_rank_0))

        callback = _Callback()

        additional_results = {}
        train(
            self.params,
            RayDMatrix(self.x, self.y),
            ray_params=RayParams(num_actors=2),
            callbacks=[callback],
            additional_results=additional_results,
            **self.kwargs)

        self.assertEqual(len(additional_results["callback_returns"]), 2)
        self.assertTrue(
            all(
                rank is True
                for (_, rank) in additional_results["callback_returns"][0]))
        self.assertTrue(
            all(
                rank is False
                for (_, rank) in additional_results["callback_returns"][1]))


if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main(["-v", __file__]))
