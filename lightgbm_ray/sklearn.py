from typing import Optional, Dict, Union, Type, Any, List, Callable, Iterable, Tuple
import numpy as np
import pandas as pd
from copy import deepcopy

from lightgbm import LGBMModel, LGBMClassifier, LGBMRegressor, LGBMRanker
from lightgbm.basic import LightGBMError, _choose_param_value
from lightgbm.compat import (
    SKLEARN_INSTALLED, LGBMNotFittedError, _LGBMAssertAllFinite,
    _LGBMCheckArray, _LGBMCheckClassificationTargets, _LGBMCheckSampleWeight,
    _LGBMCheckXY, _LGBMClassifierBase, _LGBMComputeSampleWeight,
    _LGBMLabelEncoder, _LGBMModelBase, _LGBMRegressorBase, dt_DataTable,
    pd_DataFrame)

from xgboost_ray.sklearn import _wrap_evaluation_matrices, _check_if_params_are_ray_dmatrix
from lightgbm_ray.main import train, predict, RayDMatrix, RayParams

import logging

logger = logging.getLogger(__name__)


class _RayLGBMModel:
    def _ray_fit(self,
                 model_factory: Type[LGBMModel],
                 X,
                 y,
                 sample_weight=None,
                 init_score=None,
                 group=None,
                 eval_set=None,
                 eval_names: Optional[List[str]] = None,
                 eval_sample_weight=None,
                 eval_init_score=None,
                 eval_group=None,
                 eval_metric: Optional[Union[Callable, str, List[Union[
                     Callable, str]]]] = None,
                 early_stopping_rounds: Optional[int] = None,
                 ray_params: Union[None, RayParams, Dict] = None,
                 _remote: Optional[bool] = None,
                 ray_dmatrix_params: Optional[Dict] = None,
                 **kwargs: Any) -> "_RayLGBMModel":
        # if not all((DASK_INSTALLED, PANDAS_INSTALLED, SKLEARN_INSTALLED)):
        #     raise LightGBMError('dask, pandas and scikit-learn are required for lightgbm.dask')

        if early_stopping_rounds is not None:
            raise RuntimeError(
                'early_stopping_rounds is not currently supported in lightgbm-ray'
            )

        if eval_names:
            if len(eval_names) != len(eval_set):
                raise ValueError(
                    f"Length of `eval_names` ({len(eval_names)}) doesn't match the length of `eval_set` ({len(eval_set)})"
                )

        params = self.get_params(True)

        params = _choose_param_value(
            main_param_name="n_estimators", params=params, default_value=100)

        num_boosting_round = params.pop("n_estimators")
        ray_dmatrix_params = ray_dmatrix_params or {}

        train_dmatrix, evals = _check_if_params_are_ray_dmatrix(
            X, sample_weight, init_score, eval_set, eval_sample_weight,
            eval_init_score)

        if train_dmatrix is None:
            train_dmatrix, evals = _wrap_evaluation_matrices(
                missing=None,
                X=X,
                y=y,
                group=group,
                qid=None,
                sample_weight=sample_weight,
                base_margin=init_score,
                feature_weights=None,
                eval_set=eval_set,
                sample_weight_eval_set=eval_sample_weight,
                base_margin_eval_set=eval_init_score,
                eval_group=eval_group,
                eval_qid=None,
                # changed in xgboost-ray:
                create_dmatrix=lambda **kwargs: RayDMatrix(**{
                    **kwargs,
                    **ray_dmatrix_params
                }))

        if eval_names:
            evals = [(eval_tuple[0], eval_names[i])
                     for i, eval_tuple in enumerate(evals)]

        model = train(
            dtrain=train_dmatrix,
            num_boost_round=num_boosting_round,
            params=params,
            model_factory=model_factory,
            evals=evals,
            eval_metric=eval_metric,
            ray_params=ray_params,
            _remote=_remote,
            **kwargs)

        self.set_params(**model.get_params())
        self._lgb_ray_copy_extra_params(model, self)

        return self

    def _ray_predict(self,
                     X,
                     model_factory: Type[LGBMModel],
                     *,
                     method: str = "predict",
                     ray_params: Union[None, RayParams, Dict] = None,
                     _remote: Optional[bool] = None,
                     ray_dmatrix_params: Optional[Dict],
                     **kwargs):
        ray_dmatrix_params = ray_dmatrix_params or {}
        if not isinstance(X, RayDMatrix):
            test = RayDMatrix(X, **ray_dmatrix_params)
        else:
            test = X
        return predict(
            self._lgb_ray_to_local(model_factory),
            data=test,
            method=method,
            ray_params=ray_params,
            _remote=_remote,
            **kwargs,
        )

    def _lgb_ray_to_local(self, model_factory: Type[LGBMModel]) -> LGBMModel:
        params = self.get_params()
        model = model_factory(**params)
        self._lgb_ray_copy_extra_params(self, model)
        return model

    @staticmethod
    def _lgb_ray_copy_extra_params(
            source: Union["_RayLGBMModel", LGBMModel],
            dest: Union["_RayLGBMModel", LGBMModel]) -> None:
        params = source.get_params()
        attributes = source.__dict__
        extra_param_names = set(attributes.keys()).difference(params.keys())
        for name in extra_param_names:
            setattr(dest, name, attributes[name])


class RayLGBMClassifier(LGBMClassifier, _RayLGBMModel):
    def fit(self,
            X,
            y,
            sample_weight=None,
            init_score=None,
            eval_set=None,
            eval_names: Optional[List[str]] = None,
            eval_sample_weight=None,
            eval_class_weight: Optional[List[Union[dict, str]]] = None,
            eval_init_score=None,
            eval_metric: Optional[Union[Callable, str, List[Union[
                Callable, str]]]] = None,
            early_stopping_rounds: Optional[int] = None,
            ray_params: Union[None, RayParams, Dict] = None,
            _remote: Optional[bool] = None,
            ray_dmatrix_params: Optional[Dict] = None,
            **kwargs: Any) -> "RayLGBMClassifier":
        return self._ray_fit(
            model_factory=LGBMClassifier,
            X=X,
            y=y,
            sample_weight=sample_weight,
            init_score=init_score,
            eval_set=eval_set,
            eval_names=eval_names,
            eval_sample_weight=eval_sample_weight,
            eval_class_weight=eval_class_weight,
            eval_init_score=eval_init_score,
            eval_metric=eval_metric,
            early_stopping_rounds=early_stopping_rounds,
            ray_params=ray_params,
            _remote=_remote,
            ray_dmatrix_params=ray_dmatrix_params,
            **kwargs)

    def predict_proba(self,
                      X,
                      *,
                      ray_params: Union[None, RayParams, Dict] = None,
                      _remote: Optional[bool] = None,
                      ray_dmatrix_params: Optional[Dict] = None,
                      **kwargs):
        return self._ray_predict(
            X,
            model_factory=LGBMClassifier,
            method="predict_proba",
            ray_params=ray_params,
            _remote=_remote,
            ray_dmatrix_params=ray_dmatrix_params,
            **kwargs)

    def predict(self,
                X,
                *,
                ray_params: Union[None, RayParams, Dict] = None,
                _remote: Optional[bool] = None,
                ray_dmatrix_params: Optional[Dict] = None,
                **kwargs):
        return self._ray_predict(
            X,
            model_factory=LGBMClassifier,
            method="predict",
            ray_params=ray_params,
            _remote=_remote,
            ray_dmatrix_params=ray_dmatrix_params,
            **kwargs)

    def to_local(self) -> LGBMClassifier:
        """Create regular version of lightgbm.LGBMClassifier from the distributed version.
        Returns
        -------
        model : lightgbm.LGBMClassifier
            Local underlying model.
        """
        return self._lgb_ray_to_local(LGBMClassifier)


class RayLGBMRegressor(LGBMRegressor, _RayLGBMModel):
    def fit(self,
            X,
            y,
            sample_weight=None,
            init_score=None,
            eval_set=None,
            eval_names: Optional[List[str]] = None,
            eval_sample_weight=None,
            eval_init_score=None,
            eval_metric: Optional[Union[Callable, str, List[Union[
                Callable, str]]]] = None,
            early_stopping_rounds: Optional[int] = None,
            ray_params: Union[None, RayParams, Dict] = None,
            _remote: Optional[bool] = None,
            ray_dmatrix_params: Optional[Dict] = None,
            **kwargs: Any) -> "RayLGBMRegressor":
        return self._ray_fit(
            model_factory=LGBMRegressor,
            X=X,
            y=y,
            sample_weight=sample_weight,
            init_score=init_score,
            eval_set=eval_set,
            eval_names=eval_names,
            eval_sample_weight=eval_sample_weight,
            eval_init_score=eval_init_score,
            eval_metric=eval_metric,
            early_stopping_rounds=early_stopping_rounds,
            ray_params=ray_params,
            _remote=_remote,
            ray_dmatrix_params=ray_dmatrix_params,
            **kwargs)

    def predict(self,
                X,
                *,
                ray_params: Union[None, RayParams, Dict] = None,
                _remote: Optional[bool] = None,
                ray_dmatrix_params: Optional[Dict] = None,
                **kwargs):
        return self._ray_predict(
            X,
            model_factory=LGBMRegressor,
            method="predict",
            ray_params=ray_params,
            _remote=_remote,
            ray_dmatrix_params=ray_dmatrix_params,
            **kwargs)

    def to_local(self) -> LGBMRegressor:
        """Create regular version of lightgbm.LGBMRegressor from the distributed version.
        Returns
        -------
        model : lightgbm.LGBMRegressor
            Local underlying model.
        """
        return self._lgb_ray_to_local(LGBMRegressor)