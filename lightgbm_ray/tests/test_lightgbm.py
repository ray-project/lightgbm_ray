"""Tests for lightgbm-ray, based om lightgbm.dask tests"""

# The MIT License (MIT)

# Copyright (c) Microsoft Corporation

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File based on:
# https://github.com/microsoft/LightGBM/blob/c3b9363d02564625332583e166e3ab3135f436e3/tests/python_package_test/test_dask.py

# License:
# https://github.com/microsoft/LightGBM/blob/c3b9363d02564625332583e166e3ab3135f436e3/LICENSE

import ray
from lightgbm_ray import RayDMatrix, RayParams, RayShardingMode
from lightgbm_ray.sklearn import RayLGBMClassifier, RayLGBMRegressor

import unittest
from parameterized import parameterized
import itertools

import lightgbm as lgb

import numpy as np
import pandas as pd
import sklearn.utils.estimator_checks as sklearn_checks
from sklearn.datasets import make_blobs, make_regression
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import _safe_indexing

data_output = [
    "array",
    "dataframe",
    "dataframe-with-categorical",
    "raydmatrix-interleaved",  # "raydmatrix-batch"
]
boosting_types = ["gbdt"]  # "dart", "goss", "rf"]
distributed_training_algorithms = ["data", "voting"]


def sklearn_checks_to_run():
    check_names = [
        "check_estimator_get_tags_default_keys", "check_get_params_invariance",
        "check_set_params"
    ]
    checks = []
    for check_name in check_names:
        check_func = getattr(sklearn_checks, check_name, None)
        if check_func:
            checks.append(check_func)
    return checks


estimators_to_test = [RayLGBMClassifier, RayLGBMRegressor]


def _create_data(objective, n_samples=2000, output="array", **kwargs):
    if objective.endswith("classification"):
        if objective == "binary-classification":
            centers = [[-4, -4], [4, 4]]
        elif objective == "multiclass-classification":
            centers = [[-4, -4], [4, 4], [-4, 4]]
        else:
            raise ValueError(f"Unknown classification task '{objective}'")
        X, y = make_blobs(
            n_samples=n_samples, centers=centers, random_state=42)
    elif objective == "regression":
        X, y = make_regression(
            n_samples=n_samples,
            n_features=4,
            n_informative=2,
            random_state=42)
    # elif objective == "ranking":
    #     return _create_ranking_data(
    #         n_samples=n_samples,
    #         output=output,
    #         chunk_size=chunk_size,
    #         **kwargs
    #     )
    else:
        raise ValueError(f"Unknown objective '{objective}'")
    rnd = np.random.RandomState(42)
    weights = rnd.random(X.shape[0]) * 0.01

    def convert_data(X, y, weights):
        if output == "array":
            dX = X
            dy = y
            dw = weights
        elif output.startswith("dataframe"):
            X_df = pd.DataFrame(
                X, columns=[f"feature_{i}" for i in range(X.shape[1])])
            if output == "dataframe-with-categorical":
                num_cat_cols = 2
                for i in range(num_cat_cols):
                    col_name = f"cat_col{i}"
                    cat_values = rnd.choice(["a", "b"], X.shape[0])
                    cat_series = pd.Series(cat_values, dtype="category")
                    X_df[col_name] = cat_series
                    X = np.hstack((X, cat_series.cat.codes.values.reshape(
                        -1, 1)))

                # make one categorical feature relevant to the target
                cat_col_is_a = X_df["cat_col0"] == "a"
                if objective == "regression":
                    y = np.where(cat_col_is_a, y, 2 * y)
                elif objective == "binary-classification":
                    y = np.where(cat_col_is_a, y, 1 - y)
                elif objective == "multiclass-classification":
                    n_classes = 3
                    y = np.where(cat_col_is_a, y, (1 + y) % n_classes)
            y_df = pd.Series(y, name="target")
            dX = X_df
            dy = y_df
            dw = pd.Series(weights)
        elif output.startswith("raydmatrix"):
            sharding = {
                "raydmatrix-interleaved": RayShardingMode.INTERLEAVED,
                "raydmatrix-batch": RayShardingMode.BATCH
            }
            dX = RayDMatrix(X, y, weights, sharding=sharding[output])
            dy = None
            dw = None
        else:
            raise ValueError(f"Unknown output type '{output}'")
        return dX, dy, dw

    train_idx, test_idx = (train_test_split(
        np.arange(0, len(X)),
        test_size=0.5,
        stratify=y if objective.endswith("classification") else None,
        random_state=42,
        shuffle=True))

    if output.startswith("raydmatrix"):
        dX, dy, dw = convert_data(X[train_idx], y[train_idx],
                                  weights[train_idx])
        dX_test, dy_test, dw_test = convert_data(X[test_idx], y[test_idx],
                                                 weights[test_idx])
    else:
        dX, dy, dw = convert_data(X, y, weights)
        dX_test = _safe_indexing(dX, test_idx)
        dy_test = _safe_indexing(dy, test_idx)
        dw_test = _safe_indexing(dw, test_idx)
        dX = _safe_indexing(dX, train_idx)
        dy = _safe_indexing(dy, train_idx)
        dw = _safe_indexing(dw, train_idx)

    return (X[train_idx], y[train_idx], weights[train_idx], None, dX, dy, dw,
            None, dX_test, dy_test, dw_test)


class LGBMRayTest(unittest.TestCase):
    def setUp(self):
        self.ray_params = RayParams(num_actors=2, cpus_per_actor=2)

    def tearDown(self):
        ray.shutdown()

    @parameterized.expand(
        list(
            itertools.product(
                data_output,
                ["binary-classification", "multiclass-classification"],
                boosting_types,
                distributed_training_algorithms,
            )))
    def testClassifier(self, output, task, boosting_type, tree_learner):
        ray.init(num_cpus=4, num_gpus=0)

        print(output, task, boosting_type, tree_learner)

        X, y, w, _, dX, dy, dw, _, dX_test, dy_test, dw_test = _create_data(
            objective=task, output=output)

        eval_weights = [dw_test]
        if dy_test is None:
            dy_test = "test"
            eval_weights = None
        eval_set = [(dX_test, dy_test)]

        if "raydmatrix" in output:
            lX = X
            ly = y
            lw = w
        else:
            lX = dX
            ly = dy
            lw = dw

        params = {
            "boosting_type": boosting_type,
            "tree_learner": tree_learner,
            "n_estimators": 50,
            "num_leaves": 31,
            "random_state": 1,
            "deterministic": True,
        }
        if boosting_type == "rf":
            params.update({
                "bagging_freq": 1,
                "bagging_fraction": 0.9,
            })
        elif boosting_type == "goss":
            params["top_rate"] = 0.5

        ray_classifier = RayLGBMClassifier(**params)
        ray_classifier = ray_classifier.fit(
            dX,
            dy,
            sample_weight=dw,
            ray_params=self.ray_params,
            eval_set=eval_set,
            eval_sample_weight=eval_weights)
        ray_classifier = ray_classifier.fit(
            dX, dy, sample_weight=dw, ray_params=self.ray_params)
        p1 = ray_classifier.predict(dX, ray_params=self.ray_params)
        p1_proba = ray_classifier.predict_proba(dX, ray_params=self.ray_params)
        p1_pred_leaf = ray_classifier.predict(
            dX, pred_leaf=True, ray_params=self.ray_params)
        p1_local = ray_classifier.to_local().predict(lX)
        s1 = accuracy_score(ly, p1)

        local_classifier = lgb.LGBMClassifier(**params)
        local_classifier.fit(lX, ly, sample_weight=lw)
        p2 = local_classifier.predict(lX)
        p2_proba = local_classifier.predict_proba(lX)
        s2 = local_classifier.score(lX, ly)

        if boosting_type == "rf":
            # https://github.com/microsoft/LightGBM/issues/4118
            self.assertTrue(np.allclose(s1, s2, atol=0.01))
            self.assertTrue(np.allclose(p1_proba, p2_proba, atol=0.8))
        else:
            self.assertTrue(np.allclose(s1, s2))
            self.assertTrue(np.allclose(p1, p2))
            self.assertTrue(np.allclose(p1, ly))
            self.assertTrue(np.allclose(p2, ly))
            self.assertTrue(np.allclose(p1_proba, p2_proba, atol=0.1))
            self.assertTrue(np.allclose(p1_local, p2))
            self.assertTrue(np.allclose(p1_local, ly))

        # pref_leaf values should have the right shape
        # and values that look like valid tree nodes
        pred_leaf_vals = p1_pred_leaf
        assert pred_leaf_vals.shape == (lX.shape[0],
                                        ray_classifier.booster_.num_trees())
        assert np.max(pred_leaf_vals) <= params["num_leaves"]
        assert np.min(pred_leaf_vals) >= 0
        assert len(np.unique(pred_leaf_vals)) <= params["num_leaves"]

        # be sure LightGBM actually used at least one categorical column,
        # and that it was correctly treated as a categorical feature
        if output == "dataframe-with-categorical":
            cat_cols = [
                col for col in dX.columns if dX.dtypes[col].name == "category"
            ]
            tree_df = ray_classifier.booster_.trees_to_dataframe()
            node_uses_cat_col = tree_df["split_feature"].isin(cat_cols)
            assert node_uses_cat_col.sum() > 0
            assert tree_df.loc[node_uses_cat_col, "decision_type"].unique()[
                0] == "=="

    @parameterized.expand(
        list(
            itertools.product(
                data_output,
                ["binary-classification", "multiclass-classification"],
            )))
    def testClassifierPredContrib(self, output, task):
        ray.init(num_cpus=4, num_gpus=0)

        X, y, w, _, dX, dy, dw, _, dX_test, dy_test, dw_test = _create_data(
            objective=task, output=output)

        params = {
            "n_estimators": 10,
            "num_leaves": 10,
            "random_state": 1,
            "deterministic": True,
        }

        ray_classifier = RayLGBMClassifier(tree_learner="data", **params)
        ray_classifier = ray_classifier.fit(
            dX, dy, sample_weight=dw, ray_params=self.ray_params)
        preds_with_contrib = ray_classifier.predict(
            dX, pred_contrib=True, ray_params=self.ray_params)

        local_classifier = lgb.LGBMClassifier(**params)
        if "raydmatrix" in output:
            lX = X
            ly = y
            lw = w
        else:
            lX = dX
            ly = dy
            lw = dw
        local_classifier.fit(lX, ly, sample_weight=lw)
        local_preds_with_contrib = local_classifier.predict(
            lX, pred_contrib=True)

        # be sure LightGBM actually used at least one categorical column,
        # and that it was correctly treated as a categorical feature
        if output == "dataframe-with-categorical":
            cat_cols = [
                col for col in dX.columns if dX.dtypes[col].name == "category"
            ]
            tree_df = ray_classifier.booster_.trees_to_dataframe()
            node_uses_cat_col = tree_df["split_feature"].isin(cat_cols)
            assert node_uses_cat_col.sum() > 0
            assert tree_df.loc[node_uses_cat_col, "decision_type"].unique()[
                0] == "=="

        # shape depends on whether it is binary or multiclass classification
        num_features = ray_classifier.n_features_
        num_classes = ray_classifier.n_classes_
        if num_classes == 2:
            expected_num_cols = num_features + 1
        else:
            expected_num_cols = (num_features + 1) * num_classes

        # * shape depends on whether it is binary or multiclass classification
        # * matrix for binary classification is of the form [feature_contrib,
        #   base_value],
        #   for multi-class it"s [feat_contrib_class1, base_value_class1,
        #   feat_contrib_class2, base_value_class2, etc.]
        # * contrib outputs for distributed training are different than from
        #   local training, so we can just test
        #   that the output has the right shape and base values are in the
        #   right position
        assert preds_with_contrib.shape[1] == expected_num_cols
        assert preds_with_contrib.shape == local_preds_with_contrib.shape

        if num_classes == 2:
            assert len(np.unique(preds_with_contrib[:, num_features]) == 1)
        else:
            for i in range(num_classes):
                base_value_col = num_features * (i + 1) + i
                assert len(
                    np.unique(preds_with_contrib[:, base_value_col]) == 1)

    @parameterized.expand(
        list(
            itertools.product(
                data_output,
                boosting_types,
                distributed_training_algorithms,
            )))
    def testRegressor(self, output, boosting_type, tree_learner):
        ray.init(num_cpus=4, num_gpus=0)

        X, y, w, _, dX, dy, dw, _, dX_test, dy_test, dw_test = _create_data(
            objective="regression", output=output)

        eval_weights = [dw_test]
        if dy_test is None:
            dy_test = "test"
            eval_weights = None
        eval_set = [(dX_test, dy_test)]

        if "raydmatrix" in output:
            lX = X
            ly = y
            lw = w
        else:
            lX = dX
            ly = dy
            lw = dw

        params = {
            "boosting_type": boosting_type,
            "random_state": 42,
            "num_leaves": 31,
            "n_estimators": 20,
            "deterministic": True,
        }
        if boosting_type == "rf":
            params.update({
                "bagging_freq": 1,
                "bagging_fraction": 0.9,
            })

        ray_regressor = RayLGBMRegressor(tree=tree_learner, **params)
        ray_regressor = ray_regressor.fit(
            dX,
            dy,
            sample_weight=dw,
            ray_params=self.ray_params,
            eval_set=eval_set,
            eval_sample_weight=eval_weights)
        ray_regressor = ray_regressor.fit(
            dX, dy, sample_weight=dw, ray_params=self.ray_params)
        p1 = ray_regressor.predict(dX, ray_params=self.ray_params)
        p1_pred_leaf = ray_regressor.predict(
            dX, pred_leaf=True, ray_params=self.ray_params)

        s1 = r2_score(ly, p1)
        p1_local = ray_regressor.to_local().predict(lX)
        s1_local = ray_regressor.to_local().score(lX, ly)

        local_regressor = lgb.LGBMRegressor(**params)
        local_regressor.fit(lX, ly, sample_weight=lw)
        s2 = local_regressor.score(lX, ly)
        p2 = local_regressor.predict(lX)

        # Scores should be the same
        self.assertTrue(np.allclose(s1, s2, atol=0.01))
        self.assertTrue(np.allclose(s1, s1_local))

        # Predictions should be roughly the same.
        self.assertTrue(np.allclose(p1, p1_local))

        # pref_leaf values should have the right shape
        # and values that look like valid tree nodes
        pred_leaf_vals = p1_pred_leaf
        assert pred_leaf_vals.shape == (lX.shape[0],
                                        ray_regressor.booster_.num_trees())
        assert np.max(pred_leaf_vals) <= params["num_leaves"]
        assert np.min(pred_leaf_vals) >= 0
        assert len(np.unique(pred_leaf_vals)) <= params["num_leaves"]

        self.assertTrue(np.allclose(p2, ly, rtol=0.5, atol=50.))
        self.assertTrue(np.allclose(p1, ly, rtol=0.5, atol=50.))

        # be sure LightGBM actually used at least one categorical column,
        # and that it was correctly treated as a categorical feature
        if output == "dataframe-with-categorical":
            cat_cols = [
                col for col in dX.columns if dX.dtypes[col].name == "category"
            ]
            tree_df = ray_regressor.booster_.trees_to_dataframe()
            node_uses_cat_col = tree_df["split_feature"].isin(cat_cols)
            assert node_uses_cat_col.sum() > 0
            assert tree_df.loc[node_uses_cat_col, "decision_type"].unique()[
                0] == "=="

    @parameterized.expand(data_output)
    def testRegressorPredContrib(self, output):
        ray.init(num_cpus=4, num_gpus=0)

        X, y, w, _, dX, dy, dw, _, dX_test, dy_test, dw_test = _create_data(
            objective="regression", output=output)

        if "raydmatrix" in output:
            lX = X
            ly = y
            lw = w
        else:
            lX = dX
            ly = dy
            lw = dw

        params = {
            "n_estimators": 10,
            "num_leaves": 10,
            "random_state": 1,
            "deterministic": True,
        }

        ray_regressor = RayLGBMRegressor(tree_learner="data", **params)
        ray_regressor = ray_regressor.fit(
            dX, dy, sample_weight=dw, ray_params=self.ray_params)
        preds_with_contrib = ray_regressor.predict(
            dX, pred_contrib=True, ray_params=self.ray_params)

        local_regressor = lgb.LGBMRegressor(**params)
        local_regressor.fit(lX, ly, sample_weight=lw)
        local_preds_with_contrib = local_regressor.predict(
            lX, pred_contrib=True)

        # contrib outputs for distributed training are different than
        # from local training, so we can just test
        # that the output has the right shape and base values are in
        # the right position
        num_features = lX.shape[1]
        assert preds_with_contrib.shape[1] == num_features + 1
        assert preds_with_contrib.shape == local_preds_with_contrib.shape

        # be sure LightGBM actually used at least one categorical column,
        # and that it was correctly treated as a categorical feature
        if output == "dataframe-with-categorical":
            cat_cols = [
                col for col in dX.columns if dX.dtypes[col].name == "category"
            ]
            tree_df = ray_regressor.booster_.trees_to_dataframe()
            node_uses_cat_col = tree_df["split_feature"].isin(cat_cols)
            assert node_uses_cat_col.sum() > 0
            assert tree_df.loc[node_uses_cat_col, "decision_type"].unique()[
                0] == "=="

    @parameterized.expand(list(itertools.product(data_output, [.1, .5, .9])))
    def testRegressorQuantile(self, output, alpha):
        ray.init(num_cpus=4, num_gpus=0)

        X, y, w, _, dX, dy, dw, _, dX_test, dy_test, dw_test = _create_data(
            objective="regression", output=output)

        params = {
            "objective": "quantile",
            "alpha": alpha,
            "random_state": 42,
            "n_estimators": 10,
            "num_leaves": 10,
            "deterministic": True,
        }

        if "raydmatrix" in output:
            lX = X
            ly = y
            lw = w
        else:
            lX = dX
            ly = dy
            lw = dw

        ray_regressor = RayLGBMRegressor(
            tree_learner_type="data_parallel", **params)
        ray_regressor = ray_regressor.fit(
            dX, dy, sample_weight=dw, ray_params=self.ray_params)
        p1 = ray_regressor.predict(dX, ray_params=self.ray_params)
        q1 = np.count_nonzero(ly < p1) / ly.shape[0]

        local_regressor = lgb.LGBMRegressor(**params)
        local_regressor.fit(lX, ly, sample_weight=lw)
        p2 = local_regressor.predict(lX)
        q2 = np.count_nonzero(ly < p2) / ly.shape[0]

        # Quantiles should be right
        np.testing.assert_allclose(q1, alpha, atol=0.2)
        np.testing.assert_allclose(q2, alpha, atol=0.2)

        # be sure LightGBM actually used at least one categorical column,
        # and that it was correctly treated as a categorical feature
        if output == "dataframe-with-categorical":
            cat_cols = [
                col for col in dX.columns if dX.dtypes[col].name == "category"
            ]
            tree_df = ray_regressor.booster_.trees_to_dataframe()
            node_uses_cat_col = tree_df["split_feature"].isin(cat_cols)
            assert node_uses_cat_col.sum() > 0
            assert tree_df.loc[node_uses_cat_col, "decision_type"].unique()[
                0] == "=="

    @parameterized.expand(
        list(itertools.product(
            estimators_to_test,
            sklearn_checks_to_run(),
        )))
    def testSklearnIntegration(self, estimator, check):
        estimator = estimator()
        estimator.set_params(local_listen_port=18000, time_out=5)
        name = type(estimator).__name__
        check(name, estimator)


if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main(["-v", __file__]))
