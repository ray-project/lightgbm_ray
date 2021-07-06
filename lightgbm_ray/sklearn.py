"""scikit-learn wrapper for lightgbm-ray. Based on lightgbm.dask."""

# Portions of code used in this file and implementation logic are based
# on lightgbm.dask.
# https://github.com/microsoft/LightGBM/blob/b5502d19b2b462f665e3d1edbaa70c0d6472bca4/python-package/lightgbm/dask.py

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

# License:
# https://github.com/microsoft/LightGBM/blob/c3b9363d02564625332583e166e3ab3135f436e3/LICENSE

from typing import (Optional, Dict, Union, Type, Any, List, Callable)

from lightgbm import LGBMModel, LGBMClassifier, LGBMRegressor  # LGBMRanker
from lightgbm.basic import _choose_param_value, _ConfigAliases
from xgboost_ray.sklearn import (_wrap_evaluation_matrices,
                                 _check_if_params_are_ray_dmatrix)
from lightgbm_ray.main import train, predict, RayDMatrix, RayParams

import warnings
import logging

logger = logging.getLogger(__name__)

_RAY_PARAMS_DOC = """
    ray_params : RayParams or dict, optional (default=None)
        Parameters to configure Ray-specific behavior.
        See :class:`RayParams` for a list of valid configuration parameters.
        Will override ``n_jobs`` attribute with own ``num_actors`` parameter.
    _remote : bool, optional (default=False)
        Whether to run the driver process in a remote function.
        This is enabled by default in Ray client mode.
    ray_dmatrix_params : dict, optional (default=None)
        Dict of parameters (such as sharding mode) passed to the internal
        RayDMatrix initialization."""

_N_JOBS_DOC_REPLACE = (
    """        n_jobs : int, optional (default=-1)
            Number of parallel threads.""",  # noqa: E501, W291
    """        n_jobs : int, optional (default=1)
            Number of Ray actors used to run LightGBM in parallel.
            In order to set number of threads per actor, pass a :class:`RayParams`
            object to the relevant method as a ``ray_params`` argument. Will be
            overriden by the ``num_actors`` parameter of ``ray_params`` argument
            should it be passed to a method.""",  # noqa: E501, W291
)


def _treat_estimator_doc(doc: str) -> str:
    """Helper function to make nececssary changes in estimator docstrings"""
    doc = doc.replace(*_N_JOBS_DOC_REPLACE).replace(
        "Construct a gradient boosting model.",
        "Construct a gradient boosting model distributed on Ray.")
    return doc


def _treat_method_doc(doc: str, insert_before: str) -> str:
    """Helper function to make changes in estimator method docstrings"""
    doc = doc[:doc.find(insert_before)] + _RAY_PARAMS_DOC + doc[doc.find(
        insert_before):]
    return doc


class _RayLGBMModel:
    def _ray_set_ray_params_n_jobs(
            self, ray_params: Optional[Union[RayParams, dict]],
            n_jobs: Optional[int]) -> RayParams:
        """Helper function to set num_actors in ray_params if not
        set by the user"""
        if ray_params is None:
            if not n_jobs or n_jobs < 1:
                n_jobs = 1
            ray_params = RayParams(num_actors=n_jobs)
        elif n_jobs is not None:
            warnings.warn("`ray_params` is not `None` and will override "
                          "the `n_jobs` attribute.")
        return ray_params

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

        if early_stopping_rounds is not None:
            raise RuntimeError(
                "early_stopping_rounds is not currently supported in "
                "lightgbm-ray")

        params = self.get_params(True)

        ray_params = self._ray_set_ray_params_n_jobs(ray_params,
                                                     params["n_jobs"])

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

        eval_names = eval_names or []

        for i, _ in enumerate(evals):
            if len(eval_names) > i:
                evals[i] = (evals[i][0], eval_names[i])
            else:
                # _wrap_evaluation_matrices sets default names to
                # `validation_`, but lgbm uses `valid_`, so
                # we fix that here
                evals[i] = (evals[i][0], f"valid_{i}")

        for param in _ConfigAliases.get("n_jobs"):
            params.pop(param, None)

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
        params = self.get_params(True)
        ray_params = self._ray_set_ray_params_n_jobs(ray_params,
                                                     params["n_jobs"])

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

    fit.__doc__ = _treat_method_doc(LGBMClassifier.fit.__doc__,
                                    "\n\n    Returns")

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

    predict_proba.__doc__ = _treat_method_doc(
        LGBMClassifier.predict_proba.__doc__, "\n    **kwargs")

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

    predict.__doc__ = _treat_method_doc(LGBMClassifier.predict.__doc__,
                                        "\n    **kwargs")

    def to_local(self) -> LGBMClassifier:
        """Create regular version of lightgbm.LGBMClassifier from the
        distributed version.

        Returns
        -------
        model : lightgbm.LGBMClassifier
            Local underlying model.
        """
        return self._lgb_ray_to_local(LGBMClassifier)


RayLGBMClassifier.__init__.__doc__ = _treat_estimator_doc(
    LGBMClassifier.__init__.__doc__)


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

    fit.__doc__ = _treat_method_doc(LGBMRegressor.fit.__doc__,
                                    "\n\n    Returns")

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

    predict.__doc__ = _treat_method_doc(LGBMRegressor.predict.__doc__,
                                        "\n    **kwargs")

    def to_local(self) -> LGBMRegressor:
        """Create regular version of lightgbm.LGBMRegressor from the
        distributed version.

        Returns
        -------
        model : lightgbm.LGBMRegressor
            Local underlying model.
        """
        return self._lgb_ray_to_local(LGBMRegressor)


RayLGBMRegressor.__init__.__doc__ = _treat_estimator_doc(
    RayLGBMRegressor.__init__.__doc__)
