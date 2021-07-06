# Tune imports.
import os
from typing import Dict, Union, List

import ray
import logging

from lightgbm.basic import Booster
from lightgbm.callback import CallbackEnv

from xgboost_ray.session import put_queue
from xgboost_ray.util import Unavailable, force_on_current_node

try:
    from ray import tune
    from ray.tune import is_session_enabled
    from ray.tune.utils import flatten_dict

    TUNE_INSTALLED = True
except ImportError:
    tune = None

    def is_session_enabled():
        return False

    flatten_dict = is_session_enabled
    TUNE_INSTALLED = False

try:
    from ray.tune.integration.lightgbm import \
        TuneReportCallback as OrigTuneReportCallback, \
        _TuneCheckpointCallback as _OrigTuneCheckpointCallback, \
        TuneReportCheckpointCallback as OrigTuneReportCheckpointCallback
except ImportError:
    TuneReportCallback = _TuneCheckpointCallback = \
        TuneReportCheckpointCallback = Unavailable
    OrigTuneReportCallback = _OrigTuneCheckpointCallback = \
        OrigTuneReportCheckpointCallback = object

if not hasattr(OrigTuneReportCallback, "_get_report_dict"):
    TUNE_LEGACY = True
else:
    TUNE_LEGACY = False

try:
    from ray.tune import PlacementGroupFactory

    TUNE_USING_PG = True
except ImportError:
    TUNE_USING_PG = False
    PlacementGroupFactory = Unavailable


class _TuneLGBMRank0Mixin:
    """Mixin to allow for dynamic setting of rank so that only
    one actor actually fires the callback"""

    @property
    def is_rank_0(self) -> bool:
        try:
            return self._is_rank_0
        except AttributeError:
            return True

    @is_rank_0.setter
    def is_rank_0(self, val: bool):
        self._is_rank_0 = val


if TUNE_LEGACY and TUNE_INSTALLED:

    class TuneReportCallback(_TuneLGBMRank0Mixin, OrigTuneReportCallback):
        """Create a callback that reports metrics to Ray Tune."""
        order = 20

        def __init__(
                self,
                metrics: Union[None, str, List[str], Dict[str, str]] = None):
            if isinstance(metrics, str):
                metrics = [metrics]
            self._metrics = metrics

        def _get_report_dict(self,
                             evals_log: Dict[str, Dict[str, list]]) -> dict:
            result_dict = flatten_dict(evals_log, delimiter="-")
            if not self._metrics:
                report_dict = result_dict
            else:
                report_dict = {}
                for key in self._metrics:
                    if isinstance(self._metrics, dict):
                        metric = self._metrics[key]
                    else:
                        metric = key
                    report_dict[key] = result_dict[metric]
            return report_dict

        def _get_eval_result(self, env: CallbackEnv) -> dict:
            eval_result = {}
            for data_name, eval_name, result, _ in env.evaluation_result_list:
                if data_name not in eval_result:
                    eval_result[data_name] = {}
                eval_result[data_name][eval_name] = result
            return eval_result

        def __call__(self, env: CallbackEnv) -> None:
            if not self.is_rank_0:
                return
            eval_result = self._get_eval_result(env)
            report_dict = self._get_report_dict(eval_result)
            put_queue(lambda: tune.report(**report_dict))

    class _TuneCheckpointCallback(_TuneLGBMRank0Mixin,
                                  _OrigTuneCheckpointCallback):
        """LightGBM checkpoint callback"""
        order = 19

        def __init__(self,
                     filename: str = "checkpoint",
                     frequency: int = 5,
                     *,
                     is_rank_0: bool = False):
            self._filename = filename
            self._frequency = frequency
            self.is_rank_0 = is_rank_0

        @staticmethod
        def _create_checkpoint(model: Booster, epoch: int, filename: str,
                               frequency: int):
            if epoch % frequency > 0:
                return
            with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
                model.save_model(os.path.join(checkpoint_dir, filename))

        def __call__(self, env: CallbackEnv) -> None:
            if not self.is_rank_0:
                return
            put_queue(lambda: self._create_checkpoint(
                env.model, env.iteration, self._filename, self._frequency))

    class TuneReportCheckpointCallback(_TuneLGBMRank0Mixin,
                                       OrigTuneReportCheckpointCallback):
        """Creates a callback that reports metrics and checkpoints model."""
        order = 21

        _checkpoint_callback_cls = _TuneCheckpointCallback
        _report_callback_cls = TuneReportCallback

        def __init__(
                self,
                metrics: Union[None, str, List[str], Dict[str, str]] = None,
                filename: str = "checkpoint",
                frequency: int = 5):
            self._checkpoint = self._checkpoint_callback_cls(
                filename, frequency)
            self._report = self._report_callback_cls(metrics)

        @property
        def is_rank_0(self) -> bool:
            try:
                return self._is_rank_0
            except AttributeError:
                return False

        @is_rank_0.setter
        def is_rank_0(self, val: bool):
            self._is_rank_0 = val
            if hasattr(self, "_checkpoint"):
                self._checkpoint.is_rank_0 = val
            if hasattr(self, "_report"):
                self._report.is_rank_0 = val

        def __call__(self, env: CallbackEnv) -> None:
            self._checkpoint(env)
            self._report(env)

elif TUNE_INSTALLED:
    # New style callbacks.
    class TuneReportCallback(_TuneLGBMRank0Mixin, OrigTuneReportCallback):
        def __call__(self, env: CallbackEnv) -> None:
            if not self.is_rank_0:
                return
            eval_result = self._get_eval_result(env)
            report_dict = self._get_report_dict(eval_result)
            put_queue(lambda: tune.report(**report_dict))

    class _TuneCheckpointCallback(_TuneLGBMRank0Mixin,
                                  _OrigTuneCheckpointCallback):
        def __call__(self, env: CallbackEnv) -> None:
            if not self.is_rank_0:
                return
            put_queue(lambda: self._create_checkpoint(
                env.model, env.iteration, self._filename, self._frequency))

    class TuneReportCheckpointCallback(_TuneLGBMRank0Mixin,
                                       OrigTuneReportCheckpointCallback):
        _checkpoint_callback_cls = _TuneCheckpointCallback
        _report_callback_cls = TuneReportCallback

        @property
        def is_rank_0(self) -> bool:
            try:
                return self._is_rank_0
            except AttributeError:
                return False

        @is_rank_0.setter
        def is_rank_0(self, val: bool):
            self._is_rank_0 = val
            if hasattr(self, "_checkpoint"):
                self._checkpoint.is_rank_0 = val
            if hasattr(self, "_report"):
                self._report.is_rank_0 = val


def _try_add_tune_callback(kwargs: Dict):
    if TUNE_INSTALLED and is_session_enabled():
        callbacks = kwargs.get("callbacks", []) or []
        new_callbacks = []
        has_tune_callback = False

        REPLACE_MSG = "Replaced `{orig}` with `{target}`. If you want to " \
                      "avoid this warning, pass `{target}` as a callback " \
                      "directly in your calls to `lightgbm_ray.train()`."

        for cb in callbacks:
            if isinstance(cb,
                          (TuneReportCallback, TuneReportCheckpointCallback)):
                has_tune_callback = True
                new_callbacks.append(cb)
            elif isinstance(cb, OrigTuneReportCallback):
                replace_cb = TuneReportCallback(metrics=cb._metrics)
                new_callbacks.append(replace_cb)
                logging.warning(
                    REPLACE_MSG.format(
                        orig=(
                            "ray.tune.integration.lightgbm.TuneReportCallback"
                        ),
                        target="lightgbm_ray.tune.TuneReportCallback"))
                has_tune_callback = True
            elif isinstance(cb, OrigTuneReportCheckpointCallback):
                if TUNE_LEGACY:
                    replace_cb = TuneReportCheckpointCallback(
                        metrics=cb._report._metrics,
                        filename=cb._checkpoint._filename)
                else:
                    replace_cb = TuneReportCheckpointCallback(
                        metrics=cb._report._metrics,
                        filename=cb._checkpoint._filename,
                        frequency=cb._checkpoint._frequency)
                new_callbacks.append(replace_cb)
                logging.warning(
                    REPLACE_MSG.format(
                        orig="ray.tune.integration.lightgbm."
                        "TuneReportCheckpointCallback",
                        target="lightgbm_ray.tune.TuneReportCheckpointCallback"
                    ))
                has_tune_callback = True
            else:
                new_callbacks.append(cb)

        if not has_tune_callback:
            # Todo: Maybe add checkpointing callback
            new_callbacks.append(TuneReportCallback())

        kwargs["callbacks"] = new_callbacks
        return True
    else:
        return False


def load_model(model_path):
    """Loads the model stored in the provided model_path.

    If using Ray Client, this will automatically handle loading the path on
    the server by using a Ray task.

    Returns:
        lightgbm.Booster object of the model stored in the provided model_path

    """

    def load_model_fn(model_path):
        best_bst = Booster(model_file=model_path)
        return best_bst

    # Load the model checkpoint.
    if ray.util.client.ray.is_connected():
        # If using Ray Client, the best model is saved on the server.
        # So we have to wrap the model loading in a ray task.
        remote_load = ray.remote(load_model_fn)
        remote_load = force_on_current_node(remote_load)
        bst = ray.get(remote_load.remote(model_path))
    else:
        bst = load_model_fn(model_path)

    return bst
