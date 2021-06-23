"""Tests for lightgbm.dask module"""

from numpy.core.numeric import allclose
import ray
from lightgbm_ray import train, predict, RayDMatrix, RayParams, RayShardingMode
from lightgbm_ray.sklearn import RayLGBMClassifier, RayLGBMRegressor

import inspect
import pickle
import random
import socket
from itertools import groupby
from os import getenv
from sys import platform

import unittest
from parameterized import parameterized, param
import itertools

import lightgbm as lgb

import cloudpickle
#import dask.array as da
#import dask.dataframe as dd
import joblib
import numpy as np
import pandas as pd
import sklearn.utils.estimator_checks as sklearn_checks
#from dask.array.utils import assert_eq
#from dask.distributed import Client, LocalCluster, default_client, wait
from pkg_resources import parse_version
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr
from sklearn import __version__ as sk_version
from sklearn.datasets import make_blobs, make_regression
from sklearn.metrics import r2_score, accuracy_score

data_output = [
    #"array", "dataframe", "dataframe-with-categorical",
    "raydmatrix-interleaved",
    "raydmatrix-batch"
]
boosting_types = ['gbdt']  #'dart', 'goss', 'rf']
distributed_training_algorithms = ['data', 'voting']


def _create_data(objective, n_samples=1_000, output='array', **kwargs):
    if objective.endswith('classification'):
        if objective == 'binary-classification':
            centers = [[-4, -4], [4, 4]]
        elif objective == 'multiclass-classification':
            centers = [[-4, -4], [4, 4], [-4, 4]]
        else:
            raise ValueError(f"Unknown classification task '{objective}'")
        X, y = make_blobs(
            n_samples=n_samples, centers=centers, random_state=42)
    elif objective == 'regression':
        X, y = make_regression(
            n_samples=n_samples,
            n_features=4,
            n_informative=2,
            random_state=42)
    # elif objective == 'ranking':
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

    if output == 'array':
        dX = X
        dy = y
        dw = weights
    elif output.startswith('dataframe'):
        X_df = pd.DataFrame(
            X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        if output == 'dataframe-with-categorical':
            num_cat_cols = 2
            for i in range(num_cat_cols):
                col_name = f"cat_col{i}"
                cat_values = rnd.choice(['a', 'b'], X.shape[0])
                cat_series = pd.Series(cat_values, dtype='category')
                X_df[col_name] = cat_series
                X = np.hstack((X, cat_series.cat.codes.values.reshape(-1, 1)))

            # make one categorical feature relevant to the target
            cat_col_is_a = X_df['cat_col0'] == 'a'
            if objective == 'regression':
                y = np.where(cat_col_is_a, y, 2 * y)
            elif objective == 'binary-classification':
                y = np.where(cat_col_is_a, y, 1 - y)
            elif objective == 'multiclass-classification':
                n_classes = 3
                y = np.where(cat_col_is_a, y, (1 + y) % n_classes)
        y_df = pd.Series(y, name='target')
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

    return X, y, weights, None, dX, dy, dw, None


class LGBMRayTest(unittest.TestCase):
    def setUp(self):
        self.ray_params = RayParams(num_actors=2, cpus_per_actor=2)

    def tearDown(self):
        if ray.is_initialized:
            ray.shutdown()

    @parameterized.expand(
        list(
            itertools.product(
                data_output,
                ['binary-classification', 'multiclass-classification'],
                boosting_types,
                distributed_training_algorithms,
            )))
    def testClassifier(self, output, task, boosting_type, tree_learner):
        ray.init(num_cpus=4, num_gpus=0)

        X, y, w, _, dX, dy, dw, _ = _create_data(objective=task, output=output)

        params = {
            "boosting_type": boosting_type,
            "tree_learner": tree_learner,
            "n_estimators": 50,
            "num_leaves": 31
        }
        if boosting_type == 'rf':
            params.update({
                'bagging_freq': 1,
                'bagging_fraction': 0.9,
            })
        elif boosting_type == 'goss':
            params['top_rate'] = 0.5

        ray_classifier = RayLGBMClassifier(**params)
        ray_classifier = ray_classifier.fit(
            dX, dy, sample_weight=dw, ray_params=self.ray_params)
        p1 = ray_classifier.predict(dX, ray_params=self.ray_params)
        p1_proba = ray_classifier.predict_proba(dX, ray_params=self.ray_params)
        p1_pred_leaf = ray_classifier.predict(
            dX, pred_leaf=True, ray_params=self.ray_params)
        p1_local = ray_classifier.to_local().predict(X)
        s1 = accuracy_score(y, p1)

        local_classifier = lgb.LGBMClassifier(**params)
        local_classifier.fit(X, y, sample_weight=w)
        p2 = local_classifier.predict(X)
        p2_proba = local_classifier.predict_proba(X)
        s2 = local_classifier.score(X, y)

        if boosting_type == 'rf':
            # https://github.com/microsoft/LightGBM/issues/4118
            self.assertTrue(np.allclose(s1, s2, atol=0.01))
            self.assertTrue(np.allclose(p1_proba, p2_proba, atol=0.8))
        else:
            self.assertTrue(np.allclose(s1, s2))
            self.assertTrue(np.allclose(p1, p2))
            self.assertTrue(np.allclose(p1, y))
            self.assertTrue(np.allclose(p2, y))
            self.assertTrue(np.allclose(p1_proba, p2_proba, atol=0.03))
            self.assertTrue(np.allclose(p1_local, p2))
            self.assertTrue(np.allclose(p1_local, y))

        # pref_leaf values should have the right shape
        # and values that look like valid tree nodes
        pred_leaf_vals = p1_pred_leaf
        assert pred_leaf_vals.shape == (X.shape[0],
                                        ray_classifier.booster_.num_trees())
        assert np.max(pred_leaf_vals) <= params['num_leaves']
        assert np.min(pred_leaf_vals) >= 0
        assert len(np.unique(pred_leaf_vals)) <= params['num_leaves']

        # be sure LightGBM actually used at least one categorical column,
        # and that it was correctly treated as a categorical feature
        if output == 'dataframe-with-categorical':
            cat_cols = [
                col for col in dX.columns if dX.dtypes[col].name == 'category'
            ]
            tree_df = ray_classifier.booster_.trees_to_dataframe()
            node_uses_cat_col = tree_df['split_feature'].isin(cat_cols)
            assert node_uses_cat_col.sum() > 0
            assert tree_df.loc[node_uses_cat_col, "decision_type"].unique()[
                0] == '=='

    @parameterized.expand(
        list(
            itertools.product(
                data_output,
                ['binary-classification', 'multiclass-classification'],
            )))
    def testClassifierPredContrib(self, output, task):
        ray.init(num_cpus=4, num_gpus=0)

        X, y, w, _, dX, dy, dw, _ = _create_data(objective=task, output=output)

        params = {"n_estimators": 10, "num_leaves": 10}

        ray_classifier = RayLGBMClassifier(tree_learner='data', **params)
        ray_classifier = ray_classifier.fit(
            dX, dy, sample_weight=dw, ray_params=self.ray_params)
        preds_with_contrib = ray_classifier.predict(
            dX, pred_contrib=True, ray_params=self.ray_params)

        local_classifier = lgb.LGBMClassifier(**params)
        local_classifier.fit(X, y, sample_weight=w)
        local_preds_with_contrib = local_classifier.predict(
            X, pred_contrib=True)

        # be sure LightGBM actually used at least one categorical column,
        # and that it was correctly treated as a categorical feature
        if output == 'dataframe-with-categorical':
            cat_cols = [
                col for col in dX.columns if dX.dtypes[col].name == 'category'
            ]
            tree_df = ray_classifier.booster_.trees_to_dataframe()
            node_uses_cat_col = tree_df['split_feature'].isin(cat_cols)
            assert node_uses_cat_col.sum() > 0
            assert tree_df.loc[node_uses_cat_col, "decision_type"].unique()[
                0] == '=='

        # shape depends on whether it is binary or multiclass classification
        num_features = ray_classifier.n_features_
        num_classes = ray_classifier.n_classes_
        if num_classes == 2:
            expected_num_cols = num_features + 1
        else:
            expected_num_cols = (num_features + 1) * num_classes

        # * shape depends on whether it is binary or multiclass classification
        # * matrix for binary classification is of the form [feature_contrib, base_value],
        #   for multi-class it's [feat_contrib_class1, base_value_class1, feat_contrib_class2, base_value_class2, etc.]
        # * contrib outputs for distributed training are different than from local training, so we can just test
        #   that the output has the right shape and base values are in the right position
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

        X, y, w, _, dX, dy, dw, _ = _create_data(
            objective='regression', output=output)

        params = {
            "boosting_type": boosting_type,
            "random_state": 42,
            "num_leaves": 31,
            "n_estimators": 20,
        }
        if boosting_type == 'rf':
            params.update({
                'bagging_freq': 1,
                'bagging_fraction': 0.9,
            })

        ray_regressor = RayLGBMRegressor(tree=tree_learner, **params)
        ray_regressor = ray_regressor.fit(
            dX, dy, sample_weight=dw, ray_params=self.ray_params)
        p1 = ray_regressor.predict(dX, ray_params=self.ray_params)
        p1_pred_leaf = ray_regressor.predict(
            dX, pred_leaf=True, ray_params=self.ray_params)

        s1 = r2_score(y, p1)
        p1_local = ray_regressor.to_local().predict(X)
        s1_local = ray_regressor.to_local().score(X, y)

        local_regressor = lgb.LGBMRegressor(**params)
        local_regressor.fit(X, y, sample_weight=w)
        s2 = local_regressor.score(X, y)
        p2 = local_regressor.predict(X)

        # Scores should be the same
        self.assertTrue(np.allclose(s1, s2, atol=0.01))
        self.assertTrue(np.allclose(s1, s1_local))

        # Predictions should be roughly the same.
        self.assertTrue(np.allclose(p1, p1_local))

        # pref_leaf values should have the right shape
        # and values that look like valid tree nodes
        pred_leaf_vals = p1_pred_leaf
        assert pred_leaf_vals.shape == (X.shape[0],
                                        ray_regressor.booster_.num_trees())
        assert np.max(pred_leaf_vals) <= params['num_leaves']
        assert np.min(pred_leaf_vals) >= 0
        assert len(np.unique(pred_leaf_vals)) <= params['num_leaves']

        self.assertTrue(np.allclose(p1, y, rtol=0.5, atol=50.))
        self.assertTrue(np.allclose(p2, y, rtol=0.5, atol=50.))

        # be sure LightGBM actually used at least one categorical column,
        # and that it was correctly treated as a categorical feature
        if output == 'dataframe-with-categorical':
            cat_cols = [
                col for col in dX.columns if dX.dtypes[col].name == 'category'
            ]
            tree_df = ray_regressor.booster_.trees_to_dataframe()
            node_uses_cat_col = tree_df['split_feature'].isin(cat_cols)
            assert node_uses_cat_col.sum() > 0
            assert tree_df.loc[node_uses_cat_col, "decision_type"].unique()[
                0] == '=='

    @parameterized.expand(data_output)
    def testRegressorPredContrib(self, output):
        ray.init(num_cpus=4, num_gpus=0)

        X, y, w, _, dX, dy, dw, _ = _create_data(
            objective='regression', output=output)

        params = {"n_estimators": 10, "num_leaves": 10}

        ray_regressor = RayLGBMRegressor(tree_learner='data', **params)
        ray_regressor = ray_regressor.fit(
            dX, dy, sample_weight=dw, ray_params=self.ray_params)
        preds_with_contrib = ray_regressor.predict(
            dX, pred_contrib=True, ray_params=self.ray_params)

        local_regressor = lgb.LGBMRegressor(**params)
        local_regressor.fit(X, y, sample_weight=w)
        local_preds_with_contrib = local_regressor.predict(
            X, pred_contrib=True)

        # contrib outputs for distributed training are different than from local training, so we can just test
        # that the output has the right shape and base values are in the right position
        num_features = X.shape[1]
        assert preds_with_contrib.shape[1] == num_features + 1
        assert preds_with_contrib.shape == local_preds_with_contrib.shape

        # be sure LightGBM actually used at least one categorical column,
        # and that it was correctly treated as a categorical feature
        if output == 'dataframe-with-categorical':
            cat_cols = [
                col for col in dX.columns if dX.dtypes[col].name == 'category'
            ]
            tree_df = ray_regressor.booster_.trees_to_dataframe()
            node_uses_cat_col = tree_df['split_feature'].isin(cat_cols)
            assert node_uses_cat_col.sum() > 0
            assert tree_df.loc[node_uses_cat_col, "decision_type"].unique()[
                0] == '=='

    @parameterized.expand(list(itertools.product(data_output, [.1, .5, .9])))
    def testRegressorQuantile(self, output, alpha):
        X, y, w, _, dX, dy, dw, _ = _create_data(
            objective='regression', output=output)

        params = {
            "objective": "quantile",
            "alpha": alpha,
            "random_state": 42,
            "n_estimators": 10,
            "num_leaves": 10
        }

        ray_regressor = RayLGBMRegressor(
            tree_learner_type='data_parallel', **params)
        ray_regressor = ray_regressor.fit(
            dX, dy, sample_weight=dw, ray_params=self.ray_params)
        p1 = ray_regressor.predict(dX, ray_params=self.ray_params)
        q1 = np.count_nonzero(y < p1) / y.shape[0]

        local_regressor = lgb.LGBMRegressor(**params)
        local_regressor.fit(X, y, sample_weight=w)
        p2 = local_regressor.predict(X)
        q2 = np.count_nonzero(y < p2) / y.shape[0]

        # Quantiles should be right
        np.testing.assert_allclose(q1, alpha, atol=0.2)
        np.testing.assert_allclose(q2, alpha, atol=0.2)

        # be sure LightGBM actually used at least one categorical column,
        # and that it was correctly treated as a categorical feature
        if output == 'dataframe-with-categorical':
            cat_cols = [
                col for col in dX.columns if dX.dtypes[col].name == 'category'
            ]
            tree_df = ray_regressor.booster_.trees_to_dataframe()
            node_uses_cat_col = tree_df['split_feature'].isin(cat_cols)
            assert node_uses_cat_col.sum() > 0
            assert tree_df.loc[node_uses_cat_col, "decision_type"].unique()[
                0] == '=='


if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main(["-v", __file__]))