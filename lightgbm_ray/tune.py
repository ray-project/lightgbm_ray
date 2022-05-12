# Tune imports.
from typing import Dict

import ray
import logging

from ray.util.annotations import PublicAPI

from lightgbm.basic import Booster
from lightgbm.callback import CallbackEnv

from xgboost_ray.session import put_queue
from xgboost_ray.util import force_on_current_node

try:
    from ray import tune
    from ray.tune import is_session_enabled
    from ray.tune.integration.lightgbm import (
        TuneReportCallback as OrigTuneReportCallback, _TuneCheckpointCallback
        as _OrigTuneCheckpointCallback, TuneReportCheckpointCallback as
        OrigTuneReportCheckpointCallback)

    TUNE_INSTALLED = True
except ImportError:
    tune = None

    def is_session_enabled():
        return False

    TUNE_INSTALLED = False


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


if TUNE_INSTALLED:

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


@PublicAPI(stability="beta")
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
