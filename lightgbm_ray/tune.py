import logging
from typing import Dict

from lightgbm.basic import Booster
from lightgbm.callback import CallbackEnv
from xgboost_ray.session import put_queue
from xgboost_ray.util import force_on_current_node

import ray
from ray.util.annotations import PublicAPI

try:
    import ray.train
    import ray.tune
except (ImportError, ModuleNotFoundError) as e:
    raise RuntimeError(
        "Ray Train and Ray Tune are required dependencies of `lightgbm_ray.tune` "
        'Please install with: `pip install "ray[train]"`'
    ) from e


from ray.tune.integration.lightgbm import (
    TuneReportCallback as OrigTuneReportCallback,
)
from ray.tune.integration.lightgbm import (
    TuneReportCheckpointCallback as OrigTuneReportCheckpointCallback,
)


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


class TuneReportCheckpointCallback(
    _TuneLGBMRank0Mixin, OrigTuneReportCheckpointCallback
):
    def __call__(self, env: CallbackEnv):
        if self.is_rank_0:
            put_queue(
                lambda: super(TuneReportCheckpointCallback, self).__call__(env=env)
            )


class TuneReportCallback(_TuneLGBMRank0Mixin, OrigTuneReportCallback):
    def __new__(cls: type, *args, **kwargs):
        # TODO(justinvyu): [code_removal] Remove in Ray 2.11.
        raise DeprecationWarning(
            "`TuneReportCallback` is deprecated. "
            "Use `xgboost_ray.tune.TuneReportCheckpointCallback` instead."
        )


def _try_add_tune_callback(kwargs: Dict):
    ray_train_context_initialized = ray.train.get_context()
    if ray_train_context_initialized:
        callbacks = kwargs.get("callbacks", []) or []
        new_callbacks = []
        has_tune_callback = False

        REPLACE_MSG = (
            "Replaced `{orig}` with `{target}`. If you want to "
            "avoid this warning, pass `{target}` as a callback "
            "directly in your calls to `lightgbm_ray.train()`."
        )

        for cb in callbacks:
            if isinstance(cb, TuneReportCheckpointCallback):
                has_tune_callback = True
                new_callbacks.append(cb)
            elif isinstance(cb, OrigTuneReportCheckpointCallback):
                orig_metrics = cb._metrics
                orig_frequency = cb._frequency

                replace_cb = TuneReportCheckpointCallback(
                    metrics=orig_metrics,
                    frequency=orig_frequency,
                )
                new_callbacks.append(replace_cb)
                logging.warning(
                    REPLACE_MSG.format(
                        orig="ray.tune.integration.lightgbm."
                        "TuneReportCheckpointCallback",
                        target="lightgbm_ray.tune.TuneReportCheckpointCallback",
                    )
                )
                has_tune_callback = True
            else:
                new_callbacks.append(cb)

        if not has_tune_callback:
            new_callbacks.append(TuneReportCheckpointCallback(frequency=0))

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
