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

from typing import (Tuple, Dict, Any, List, Optional, Type, Union, Sequence,
                    Callable)
from copy import deepcopy
from dataclasses import dataclass

import time
import logging
import os
import threading
import warnings

import numpy as np
import pandas as pd

from lightgbm import LGBMModel, LGBMRanker, Booster
from lightgbm.basic import (_choose_param_value, _ConfigAliases, LightGBMError,
                            _safe_call)
from lightgbm.callback import CallbackEnv

import ray

from xgboost_ray.main import (
    _handle_queue, RayXGBoostActor, LEGACY_MATRIX, RayDeviceQuantileDMatrix,
    concat_dataframes, _set_omp_num_threads, Queue, Event, DistributedCallback,
    STATUS_FREQUENCY_S, RayActorError, pickle, _PrepareActorTask, RayParams as
    RayXGBParams, _TrainingState, _is_client_connected, is_session_enabled,
    force_on_current_node, _assert_ray_support, _maybe_print_legacy_warning,
    _Checkpoint, _create_communication_processes, TUNE_USING_PG,
    _USE_SPREAD_STRATEGY, RayTaskError, RayXGBoostActorAvailable,
    RayXGBoostTrainingError, _create_placement_group, _shutdown,
    PlacementGroup, ActorHandle, RayXGBoostTrainingStopped, combine_data,
    _trigger_data_load, DEFAULT_PG, _autodetect_resources as
    _autodetect_resources_base)
from xgboost_ray.session import put_queue
from xgboost_ray import RayDMatrix

from lightgbm_ray.util import find_free_port, is_port_free, lgbm_network_free
from lightgbm_ray.tune import _try_add_tune_callback, _TuneLGBMRank0Mixin

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

ELASTIC_RESTART_DISABLED = True


class StopException(Exception):
    pass


def _check_cpus_per_actor_at_least_2(cpus_per_actor: int,
                                     suppress_exception: bool):
    """Raise an exception or a warning if cpus_per_actor < 2"""
    if cpus_per_actor < 2:
        if suppress_exception:
            warnings.warn("cpus_per_actor is set to less than 2. Distributed"
                          " LightGBM needs at least 2 CPUs per actor to "
                          "train efficiently. This may lead to a "
                          "degradation of performance during training.")
        else:
            raise ValueError(
                "cpus_per_actor is set to less than 2. Distributed"
                " LightGBM needs at least 2 CPUs per actor to "
                "train efficiently. You can suppress this "
                "exception by setting allow_less_than_two_cpus "
                "to True.")


def _get_data_dict(data: RayDMatrix, param: Dict) -> Dict:
    if not LEGACY_MATRIX and isinstance(data, RayDeviceQuantileDMatrix):
        # If we only got a single data shard, create a list so we can
        # iterate over it
        if not isinstance(param["data"], list):
            param["data"] = [param["data"]]

            if not isinstance(param["label"], list):
                param["label"] = [param["label"]]
            if not isinstance(param["weight"], list):
                param["weight"] = [param["weight"]]
            if not isinstance(param["data"], list):
                param["base_margin"] = [param["base_margin"]]

        param["label_lower_bound"] = [None]
        param["label_upper_bound"] = [None]

        dm_param = {
            "feature_names": data.feature_names,
            "feature_types": data.feature_types,
            "missing": data.missing,
        }
        param.update(dm_param)
    else:
        if isinstance(param["data"], list):
            dm_param = {
                "data": concat_dataframes(param["data"]),
                "label": concat_dataframes(param["label"]),
                "weight": concat_dataframes(param["weight"]),
                "base_margin": concat_dataframes(param["base_margin"]),
                "label_lower_bound": concat_dataframes(
                    param["label_lower_bound"]),
                "label_upper_bound": concat_dataframes(
                    param["label_upper_bound"]),
            }
            param.update(dm_param)

    return param

    # data.update_matrix_properties(matrix)
    # return matrix


@dataclass
class RayParams(RayXGBParams):
    # The RayParams from XGBoost-Ray can also be used, in which
    # case allow_less_than_two_cpus will just default to False
    allow_less_than_two_cpus: bool = False

    __doc__ = RayXGBParams.__doc__.replace(
        """        elastic_training (bool): If True, training will continue with
            fewer actors if an actor fails. Default False.""",
        """        allow_less_than_two_cpus (bool): If True, an exception will not
            be raised if `cpus_per_actor`. Default False."""
    ).replace(
        """cpus_per_actor (int): Number of CPUs to be used per Ray actor.""",
        """cpus_per_actor (int): Number of CPUs to be used per Ray actor.
            If smaller than 2, training might be substantially slower
            because communication work and training work will block
            each other. This will raise an exception unless
            `allow_less_than_two_cpus` is True.""")

    def get_tune_resources(self):
        _check_cpus_per_actor_at_least_2(
            self.cpus_per_actor,
            getattr(self, "allow_less_than_two_cpus", False))
        return super().get_tune_resources()


def _validate_ray_params(ray_params: Union[None, RayParams, dict]) \
        -> RayParams:
    if ray_params is None:
        ray_params = RayParams()
    elif isinstance(ray_params, dict):
        ray_params = RayParams(**ray_params)
    elif not isinstance(ray_params, RayParams):
        raise ValueError(
            f"`ray_params` must be a `RayParams` instance, a dict, or None, "
            f"but it was {type(ray_params)}."
            f"\nFIX THIS preferably by passing a `RayParams` instance as "
            f"the `ray_params` parameter.")
    if ray_params.num_actors < 2:
        warnings.warn(
            f"`num_actors` in `ray_params` is smaller than 2 "
            f"({ray_params.num_actors}). LightGBM will NOT be distributed!")
    return ray_params


class RayLightGBMActor(RayXGBoostActor):
    def __init__(
            self,
            rank: int,
            num_actors: int,
            model_factory: Optional[Type[LGBMModel]] = None,
            queue: Optional[Queue] = None,
            stop_event: Optional[Event] = None,
            checkpoint_frequency: int = 5,
            distributed_callbacks: Optional[List[DistributedCallback]] = None,
            network_params: Optional[dict] = None,
    ):
        self.network_params = {} if not network_params else \
            network_params.copy()
        if "time_out" not in self.network_params:
            self.network_params["time_out"] = 120
        self.model_factory = model_factory
        super().__init__(
            rank=rank,
            num_actors=num_actors,
            queue=queue,
            stop_event=stop_event,
            checkpoint_frequency=checkpoint_frequency,
            distributed_callbacks=distributed_callbacks)

    def _save_checkpoint_callback(self, is_rank_0: bool) -> Callable:
        this = self

        def _save_internal_checkpoint_callback() -> Callable:
            def _callback(env: CallbackEnv) -> None:
                if not is_rank_0:
                    return
                if env.iteration % this.checkpoint_frequency == 0:
                    put_queue(
                        _Checkpoint(env.iteration, pickle.dumps(env.model)))
                if env.iteration == env.end_iteration - 1:
                    put_queue(_Checkpoint(-1, pickle.dumps(env.model)))

            _callback.order = 25  # type: ignore
            return _callback

        return _save_internal_checkpoint_callback()

    def _stop_callback(self, is_rank_0: bool) -> Callable:
        this = self
        # Keep track of initial stop event. Since we're training in a thread,
        # the stop event might be overwritten, which should he handled
        # as if the previous stop event was set.
        initial_stop_event = self._stop_event

        def _stop_callback() -> Callable:
            def _callback(env: CallbackEnv) -> None:
                try:
                    if this._stop_event.is_set() or \
                            this._get_stop_event() is not initial_stop_event:
                        raise StopException()
                except RayActorError:
                    raise StopException()

            _callback.order = 26  # type: ignore
            return _callback

        return _stop_callback()

    def find_free_address(self) -> Tuple[str, int]:
        port = self.port()
        if not port:
            port = find_free_port()
        else:
            assert self.is_port_free(port)
        return (self.ip(), port)

    def port(self) -> Optional[int]:
        return self.network_params.get("local_listen_port", None)

    def is_port_free(self, port: int) -> bool:
        return is_port_free(port)

    def set_network_params(
            self,
            machines: str,
            local_listen_port: int,
            num_machines: int,
            time_out: Optional[int] = None,
    ):
        """Set LightGBM params responsible for networking"""
        self.network_params["machines"] = machines
        self.network_params["local_listen_port"] = local_listen_port
        self.network_params["num_machines"] = num_machines
        if time_out is not None:
            self.network_params["time_out"] = time_out

    def load_data(self, data: RayDMatrix):
        # LightGBM specific - Main difference between this and XGBoost:
        # XGBoost needs a local DMatrix, while this runs off Pandas
        # objects returned by the RayDMatrix directly.
        if data in self._data:
            return

        self._distributed_callbacks.before_data_loading(self, data)

        param = data.get_data(self.rank, self.num_actors)
        if isinstance(param["data"], list):
            self._local_n[data] = sum(len(a) for a in param["data"])
        else:
            self._local_n[data] = len(param["data"])
        data.unload_data()  # Free object store

        d = _get_data_dict(data, param).copy()
        self._data[data] = d

        self._distributed_callbacks.after_data_loading(self, data)

    def train(self, return_bst: bool, params: Dict[str, Any],
              dtrain: RayDMatrix, evals: Tuple[RayDMatrix, str],
              boost_rounds_left: int, *args, **kwargs) -> Dict[str, Any]:
        if self.model_factory is None:
            raise ValueError("model_factory cannot be None for training")

        # LightGBM specific - import the CDLL pointer
        # so that _LIB.LGBM_NetworkFree() can be called
        # in lgbm_network_free context later
        from lightgbm.basic import _LIB

        self._distributed_callbacks.before_train(self)

        num_threads = _set_omp_num_threads()

        local_params = _choose_param_value(
            main_param_name="num_threads",
            params=params,
            default_value=num_threads if num_threads > 0 else
            sum(num
                for _, num in ray.worker.get_resource_ids().get("CPU", [])))

        if "init_model" in kwargs:
            if isinstance(kwargs["init_model"], bytes):
                # bytearray type gets lost in remote actor call
                kwargs["init_model"] = bytearray(kwargs["init_model"])

        if dtrain not in self._data:
            self.load_data(dtrain)

        local_dtrain = self._data[dtrain]

        # if not local_dtrain.get_label().size:
        #     raise RuntimeError(
        #         "Training data has no label set. Please make sure to set "
        #         "the `label` argument when initializing `RayDMatrix()` "
        #         "for data you would like to train on.")

        local_evals = []
        local_eval_names = []
        local_eval_sample_weights = []
        local_eval_init_scores = []
        for deval, name in evals:
            if deval not in self._data:
                self.load_data(deval)
            local_evals.append((self._data[deval]["data"],
                                self._data[deval]["label"]))
            local_eval_names.append(name)
            local_eval_sample_weights.append(self._data[deval]["weight"])
            local_eval_init_scores.append(self._data[deval]["base_margin"])

        if "callbacks" in kwargs:
            callbacks = kwargs["callbacks"] or []
        else:
            callbacks = []
        callbacks.append(self._save_checkpoint_callback(is_rank_0=return_bst))
        callbacks.append(self._stop_callback(is_rank_0=return_bst))
        for callback in callbacks:
            if isinstance(callback, _TuneLGBMRank0Mixin):
                callback.is_rank_0 = return_bst
        kwargs["callbacks"] = callbacks

        result_dict = {}
        error_dict = {}

        network_params = self.network_params
        local_params.update(network_params)

        local_params["n_estimators"] = boost_rounds_left

        is_ranker = issubclass(self.model_factory, LGBMRanker)

        # We run fit in a thread to be able to react to the stop event.

        def _train():
            logger.debug(f"starting LightGBM training, rank {self.rank}, "
                         f"{self.network_params}, {local_params}, {kwargs}")
            try:
                model = self.model_factory(**local_params)
                # LightGBM specific - this context calls
                # _LIB.LGBM_NetworkFree(), which is
                # supposed to clean up the network and
                # free up ports should the training fail
                # this is also called separately for good measure
                with lgbm_network_free(model, _LIB):
                    if is_ranker:
                        # missing group arg, update later
                        model.fit(
                            local_dtrain["data"],
                            local_dtrain["label"],
                            sample_weight=local_dtrain["weight"],
                            init_score=local_dtrain["base_margin"],
                            eval_set=local_evals,
                            eval_names=local_eval_names,
                            eval_sample_weight=local_eval_sample_weights,
                            eval_init_score=local_eval_init_scores,
                            **kwargs)
                    else:
                        model.fit(
                            local_dtrain["data"],
                            local_dtrain["label"],
                            sample_weight=local_dtrain["weight"],
                            init_score=local_dtrain["base_margin"],
                            eval_set=local_evals,
                            eval_names=local_eval_names,
                            eval_sample_weight=local_eval_sample_weights,
                            eval_init_score=local_eval_init_scores,
                            **kwargs)
                result_dict.update({
                    "bst": model,
                    "evals_result": model.evals_result_,
                    "train_n": self._local_n[dtrain]
                })
            except StopException:
                # Usually this should be caught by XGBoost core.
                # Silent fail, will be raised as RayXGBoostTrainingStopped.

                # LightGBM specific - clean up network and open ports
                _safe_call(_LIB.LGBM_NetworkFree())
                return
            except LightGBMError as e:
                # LightGBM specific - clean up network and open ports
                _safe_call(_LIB.LGBM_NetworkFree())
                error_dict.update({"exception": e})
                return
            finally:
                # LightGBM specific - clean up network and open ports
                _safe_call(_LIB.LGBM_NetworkFree())

        thread = threading.Thread(target=_train)
        thread.daemon = True
        thread.start()
        while thread.is_alive():
            thread.join(timeout=0)
            if self._stop_event.is_set():
                # LightGBM specific - clean up network and open ports
                _safe_call(_LIB.LGBM_NetworkFree())
                raise RayXGBoostTrainingStopped("Training was interrupted.")
            time.sleep(0.1)

        if not result_dict:
            raise_from = error_dict.get("exception", None)
            # LightGBM specific - clean up network and open ports
            _safe_call(_LIB.LGBM_NetworkFree())
            raise RayXGBoostTrainingError("Training failed.") from raise_from

        thread.join()
        # LightGBM specific - clean up network and open ports
        _safe_call(_LIB.LGBM_NetworkFree())
        self._distributed_callbacks.after_train(self, result_dict)

        if not return_bst:
            result_dict.pop("bst", None)

        return result_dict

    def predict(self,
                model: Union[LGBMModel, Booster],
                data: RayDMatrix,
                method="predict",
                **kwargs):
        self._distributed_callbacks.before_predict(self)

        _set_omp_num_threads()

        if data not in self._data:
            self.load_data(data)
        local_data = self._data[data]["data"]

        predictions = getattr(model, method)(local_data, **kwargs)

        if predictions.ndim == 1:
            callback_predictions = pd.Series(predictions)
        else:
            callback_predictions = pd.DataFrame(predictions)
        self._distributed_callbacks.after_predict(self, callback_predictions)
        return predictions


@ray.remote
class _RemoteRayLightGBMActor(RayLightGBMActor):
    pass


def _autodetect_resources(ray_params: RayParams,
                          use_tree_method: bool = False) -> Tuple[int, int]:
    cpus_per_actor, gpus_per_actor = _autodetect_resources_base(
        ray_params, use_tree_method)
    if ray_params.cpus_per_actor <= 0:
        cpus_per_actor = max(2, cpus_per_actor)
    return cpus_per_actor, gpus_per_actor


def _create_actor(
        rank: int,
        num_actors: int,
        model_factory: Type[LGBMModel],
        num_cpus_per_actor: int,
        num_gpus_per_actor: int,
        resources_per_actor: Optional[Dict] = None,
        placement_group: Optional[PlacementGroup] = None,
        queue: Optional[Queue] = None,
        checkpoint_frequency: int = 5,
        distributed_callbacks: Optional[Sequence[DistributedCallback]] = None,
        ip: Optional[str] = None,
        port: Optional[int] = None,
) -> ActorHandle:
    # If we have an IP passed, force the actor to be spawned on a node
    # with that IP
    if ip:
        if resources_per_actor is not None:
            resources_per_actor[f"node:{ip}"] = 0.01
        else:
            resources_per_actor = {f"node:{ip}": 0.01}
    # Send DEFAULT_PG here, which changed in Ray > 1.4.0
    # If we send `None`, this will ignore the parent placement group and
    # lead to errors e.g. when used within Ray Tune
    return _RemoteRayLightGBMActor.options(
        num_cpus=num_cpus_per_actor,
        num_gpus=num_gpus_per_actor,
        resources=resources_per_actor,
        placement_group=placement_group or DEFAULT_PG).remote(
            rank=rank,
            num_actors=num_actors,
            model_factory=model_factory,
            queue=queue,
            checkpoint_frequency=checkpoint_frequency,
            distributed_callbacks=distributed_callbacks,
            network_params={"local_listen_port": port} if port else None)


def _train(params: Dict,
           dtrain: RayDMatrix,
           model_factory: Type[LGBMModel],
           boost_rounds_left: int,
           *args,
           evals=(),
           ray_params: RayParams,
           cpus_per_actor: int,
           gpus_per_actor: int,
           _training_state: _TrainingState,
           machine_addresses: Optional[List[Tuple[str, str]]] = None,
           listen_port: Optional[int] = None,
           **kwargs) -> Tuple[LGBMModel, Dict, Dict]:
    """This is the local train function wrapped by :func:`train() <train>`.

    This function can be thought of one invocation of a multi-actor lightgbm
    training run. It starts the required number of actors, triggers data
    loading, collects the results, and handles (i.e. registers) actor failures
    - but it does not handle fault tolerance or general training setup.

    Generally, this function is called one or multiple times by the
    :func:`train() <train>` function. It is called exactly once if no
    errors occur. It is called more than once if errors occurred (e.g. an
    actor died) and failure handling is enabled.
    """
    from xgboost_ray.elastic import _maybe_schedule_new_actors, \
        _update_scheduled_actor_states, _get_actor_alive_status

    # Un-schedule possible scheduled restarts
    _training_state.restart_training_at = None

    params = deepcopy(params)

    if "n_jobs" in params:
        if params["n_jobs"] > cpus_per_actor:
            raise ValueError(
                "Specified number of threads greater than number of CPUs. "
                "\nFIX THIS by passing a lower value for the `n_jobs` "
                "parameter or a higher number for `cpus_per_actor`.")
    else:
        params["n_jobs"] = cpus_per_actor

    _check_cpus_per_actor_at_least_2(
        params["n_jobs"], getattr(ray_params, "allow_less_than_two_cpus",
                                  False))

    # This is a callback that handles actor failures.
    # We identify the rank of the failed actor, add this to a set of
    # failed actors (which we might want to restart later), and set its
    # entry in the actor list to None.
    def handle_actor_failure(actor_id):
        rank = _training_state.actors.index(actor_id)
        _training_state.failed_actor_ranks.add(rank)
        _training_state.actors[rank] = None

    # Here we create new actors. In the first invocation of _train(), this
    # will be all actors. In future invocations, this may be less than
    # the num_actors setting, depending on the failure mode.
    newly_created = 0

    for i in list(_training_state.failed_actor_ranks):
        if _training_state.actors[i] is not None:
            raise RuntimeError(
                f"Trying to create actor with rank {i}, but it already "
                f"exists.")
        ip = None
        port = None
        if machine_addresses:
            ip = machine_addresses[i][0]
            port = machine_addresses[i][1]
        elif listen_port:
            port = listen_port
        actor = _create_actor(
            rank=i,
            num_actors=ray_params.num_actors,
            model_factory=model_factory,
            num_cpus_per_actor=cpus_per_actor,
            num_gpus_per_actor=gpus_per_actor,
            resources_per_actor=ray_params.resources_per_actor,
            placement_group=_training_state.placement_group,
            queue=_training_state.queue,
            checkpoint_frequency=ray_params.checkpoint_frequency,
            distributed_callbacks=ray_params.distributed_callbacks,
            ip=ip,
            port=port)
        # Set actor entry in our list
        _training_state.actors[i] = actor
        # Remove from this set so it is not created again
        _training_state.failed_actor_ranks.remove(i)
        newly_created += 1

    alive_actors = sum(1 for a in _training_state.actors if a is not None)
    logger.info(f"[RayLightGBM] Created {newly_created} new actors "
                f"({alive_actors} total actors). Waiting until actors "
                f"are ready for training.")

    # For distributed datasets (e.g. Modin), this will initialize
    # (and fix) the assignment of data shards to actor ranks
    dtrain.assert_enough_shards_for_actors(num_actors=ray_params.num_actors)
    dtrain.assign_shards_to_actors(_training_state.actors)
    for deval, _ in evals:
        deval.assert_enough_shards_for_actors(num_actors=ray_params.num_actors)
        deval.assign_shards_to_actors(_training_state.actors)

    load_data = [dtrain] + [eval[0] for eval in evals]

    prepare_actor_tasks = [
        _PrepareActorTask(
            actor,
            # Maybe we got a new Queue actor, so send it to all actors.
            queue=_training_state.queue,
            # Maybe we got a new Event actor, so send it to all actors.
            stop_event=_training_state.stop_event,
            # Trigger data loading
            load_data=load_data) for actor in _training_state.actors
        if actor is not None
    ]

    start_wait = time.time()
    last_status = start_wait
    try:
        # Construct list before calling any() to force evaluation
        ready_states = [task.is_ready() for task in prepare_actor_tasks]
        while not all(ready_states):
            if time.time() >= last_status + STATUS_FREQUENCY_S:
                wait_time = time.time() - start_wait
                logger.info(f"Waiting until actors are ready "
                            f"({wait_time:.0f} seconds passed).")
                last_status = time.time()
            time.sleep(0.1)
            ready_states = [task.is_ready() for task in prepare_actor_tasks]

    except Exception as exc:
        _training_state.stop_event.set()
        _get_actor_alive_status(_training_state.actors, handle_actor_failure)
        raise RayActorError from exc

    logger.info("[RayLightGBM] Starting LightGBM training.")

    # # Start Rabit tracker for gradient sharing
    # rabit_process, env = _start_rabit_tracker(alive_actors)
    # rabit_args = [("%s=%s" % item).encode() for item in env.items()]

    # Load checkpoint if we have one. In that case we need to adjust the
    # number of training rounds.
    if _training_state.checkpoint.value:
        kwargs["init_model"] = pickle.loads(_training_state.checkpoint.value)
        if _training_state.checkpoint.iteration == -1:
            # -1 means training already finished.
            logger.error(
                "Trying to load continue from checkpoint, but the checkpoint"
                "indicates training already finished. Returning last"
                "checkpointed model instead.")
            return kwargs["init_model"], {}, _training_state.additional_results

    # The callback_returns dict contains actor-rank indexed lists of
    # results obtained through the `put_queue` function, usually
    # sent via callbacks.
    callback_returns = _training_state.additional_results.get(
        "callback_returns")
    if callback_returns is None:
        callback_returns = [list() for _ in range(len(_training_state.actors))]
        _training_state.additional_results[
            "callback_returns"] = callback_returns

    _training_state.training_started_at = time.time()

    # Trigger the train function
    live_actors = [
        actor for actor in _training_state.actors if actor is not None
    ]

    # LightGBM specific: handle actor addresses
    # if neither local_listening_port nor machines are set
    # get the ips and a random port from the actors, and then
    # assign them back so the lgbm params are updated.
    # do this in a loop to ensure that if there is a port
    # confilict, it can try and choose a new one. Most of the times
    # it will complete in one iteration
    machines = None
    for i in range(5):
        addresses = ray.get(
            [actor.find_free_address.remote() for actor in live_actors])
        if addresses:
            _, ports = zip(*addresses)
            ports = list(ports)
            machine_addresses_new = [f"{ip}:{port}" for ip, port in addresses]
            if len(machine_addresses_new) == len(set(machine_addresses_new)):
                machines = ",".join(machine_addresses_new)
                break
            if machine_addresses:
                raise ValueError(
                    "Machine addresses contains non-unique entries.")
            else:
                logger.debug("Couldn't obtain unique addresses, trying again.")
    if machines:
        logger.debug(f"Obtained unique addresses in {i} attempts.")
    else:
        raise ValueError(
            f"Couldn't obtain enough unique addresses for {len(live_actors)}."
            " Try reducing the number of actors.")
    for i, actor in enumerate(live_actors):
        actor.set_network_params.remote(machines, ports[i], len(live_actors),
                                        params.get("time_out", 120))

    training_futures = [
        actor.train.remote(
            i == 0,  # return_bst
            params,
            dtrain,
            evals,
            boost_rounds_left,
            *args,
            **kwargs) for i, actor in enumerate(live_actors)
    ]

    # Failure handling loop. Here we wait until all training tasks finished.
    # If a training task fails, we stop training on the remaining actors,
    # check which ones are still alive, and raise the error.
    # The train() wrapper function will then handle the error.
    start_wait = time.time()
    last_status = start_wait
    try:
        not_ready = training_futures
        while not_ready:
            if _training_state.queue:
                _handle_queue(
                    queue=_training_state.queue,
                    checkpoint=_training_state.checkpoint,
                    callback_returns=callback_returns)

            if ray_params.elastic_training \
                    and not ELASTIC_RESTART_DISABLED:
                _maybe_schedule_new_actors(
                    training_state=_training_state,
                    num_cpus_per_actor=cpus_per_actor,
                    num_gpus_per_actor=gpus_per_actor,
                    resources_per_actor=ray_params.resources_per_actor,
                    ray_params=ray_params,
                    load_data=load_data)

                # This may raise RayXGBoostActorAvailable
                _update_scheduled_actor_states(_training_state)

            if time.time() >= last_status + STATUS_FREQUENCY_S:
                wait_time = time.time() - start_wait
                logger.info(f"Training in progress "
                            f"({wait_time:.0f} seconds since last restart).")
                last_status = time.time()

            ready, not_ready = ray.wait(
                not_ready, num_returns=len(not_ready), timeout=1)
            ray.get(ready)

        # Get items from queue one last time
        if _training_state.queue:
            _handle_queue(
                queue=_training_state.queue,
                checkpoint=_training_state.checkpoint,
                callback_returns=callback_returns)

    # The inner loop should catch all exceptions
    except Exception as exc:
        logger.debug(f"Caught exception in training loop: {exc}")

        # Stop all other actors from training
        _training_state.stop_event.set()

        # Check which actors are still alive
        _get_actor_alive_status(_training_state.actors, handle_actor_failure)

        raise RayActorError from exc

    # Training is now complete.
    # # Stop Rabit tracking process
    # _stop_rabit_tracker(rabit_process)

    # Get all results from all actors.
    all_results: List[Dict[str, Any]] = ray.get(training_futures)

    # All results should be the same. But only
    # the first one actually returns its bst object.
    bst: LGBMModel = all_results[0]["bst"]
    evals_result = all_results[0]["evals_result"]

    if not listen_port:
        for param in _ConfigAliases.get("local_listen_port"):
            bst._other_params.pop(param, None)

    if not machine_addresses:
        for param in _ConfigAliases.get("machines"):
            bst._other_params.pop(param, None)

    for param in _ConfigAliases.get("num_machines", "time_out"):
        bst._other_params.pop(param, None)

    if callback_returns:
        _training_state.additional_results[
            "callback_returns"] = callback_returns

    total_n = sum(res["train_n"] or 0 for res in all_results)

    _training_state.additional_results["total_n"] = total_n

    return bst, evals_result, _training_state.additional_results


def train(
        params: Dict,
        dtrain: RayDMatrix,
        model_factory: Type[LGBMModel] = LGBMModel,
        num_boost_round: int = 10,
        *args,
        valid_sets: Optional[List[RayDMatrix]] = None,
        valid_names: Optional[List[str]] = None,
        verbose_eval: Union[bool, int] = True,
        evals: Union[List[Tuple[RayDMatrix, str]], Tuple[RayDMatrix, str]] = (
        ),
        evals_result: Optional[Dict] = None,
        additional_results: Optional[Dict] = None,
        ray_params: Union[None, RayParams, Dict] = None,
        _remote: Optional[bool] = None,
        **kwargs) -> LGBMModel:
    """Distributed LightGBM training via Ray.

    This function will connect to a Ray cluster, create ``num_actors``
    remote actors, send data shards to them, and have them train an
    LightGBM model using LightGBM's built-in distributed mode.

    This method handles setting up the following network parameters:
    - ``local_listen_port``: port that each LightGBM worker opens a
        listening socket on, to accept connections from other workers.
        This can differ from LightGBM worker to LightGBM worker, but
        does not have to.
    - ``machines``: a comma-delimited list of all workers in the cluster,
        in the form ``ip:port,ip:port``. If running multiple workers
        on the same Ray Node, use different ports for each worker. For
        example, for ``ray_params.num_actors=3``, you might pass
        ``"127.0.0.1:12400,127.0.0.1:12401,127.0.0.1:12402"``.

    The default behavior of this function is to generate ``machines`` based
    on Ray workers, and to search for an open port on each worker to be
    used as ``local_listen_port``.

    If ``machines`` is provided explicitly in ``params``, this function uses
    the hosts and ports in that list directly, and will try to start Ray
    workers on the nodes with the given ips. If that is not possible, or any
    of those ports are not free when training starts, training will fail.

    If ``local_listen_port`` is provided in ``params`` and ``machines`` is not,
    this function constructs ``machines`` automatically from auto-assigned Ray
    workers, assuming that each one will use the same ``local_listen_port``.

    Failure handling:

    LightGBM on Ray supports automatic failure handling that can be configured
    with the :class:`ray_params <RayParams>` argument. If an actor or local
    training task dies, the Ray actor is marked as dead and
    the number of restarts is below ``ray_params.max_actor_restarts``,
    Ray will try to schedule the dead actor again, load the data shard
    on this actor, and then continue training from the latest checkpoint.

    Otherwise, training is aborted.

    Args:
        params (Dict): parameter dict passed to ``LGBMModel``
        dtrain (RayDMatrix): Data object containing the training data.
        model_factory (Type[LGBMModel]) Model class to use for training.
        valid_sets (Optional[List[RayDMatrix]]):
            List of data to be evaluated on during training.
            Mutually exclusive with ``evals``.
        valid_names Optional[List[str]]:
            Names of ``valid_sets``.
        evals (Union[List[Tuple[RayDMatrix, str]], Tuple[RayDMatrix, str]]):
            ``evals`` tuple passed to ``LGBMModel.fit()``.
            Mutually exclusive with ``valid_sets``.
        evals_result (Optional[Dict]): Dict to store evaluation results in.
        verbose_eval (Union[bool, int]):
            Requires at least one validation data.
            If True, the eval metric on the valid set is printed at each
            boosting stage.
            If int, the eval metric on the valid set is printed at every
            ``verbose_eval`` boosting stage.
            The last boosting stage or the boosting stage found by using
            ``early_stopping_rounds`` is also printed.
            With ``verbose_eval`` = 4 and at least one item in ``valid_sets``,
            an evaluation metric is printed every 4 (instead of 1) boosting
            stages.
        additional_results (Optional[Dict]): Dict to store additional results.
        ray_params (Union[None, RayParams, Dict]): Parameters to configure
            Ray-specific behavior. See :class:`RayParams` for a list of valid
            configuration parameters.
        _remote (bool): Whether to run the driver process in a remote
            function. This is enabled by default in Ray client mode.
        **kwargs: Keyword arguments will be passed to the local
            `model_factory.fit()` calls.

    Returns: An ``LGBMModel`` object.
    """
    os.environ.setdefault("RAY_IGNORE_UNHANDLED_ERRORS", "1")

    if _remote is None:
        _remote = _is_client_connected() and \
                  not is_session_enabled()

    if not ray.is_initialized():
        ray.init()

    if _remote:
        # Run this function as a remote function to support Ray client mode.
        @ray.remote(num_cpus=0)
        def _wrapped(*args, **kwargs):
            _evals_result = {}
            _additional_results = {}
            bst = train(
                *args,
                model_factory=model_factory,
                num_boost_round=num_boost_round,
                evals_result=_evals_result,
                additional_results=_additional_results,
                verbose_eval=verbose_eval,
                **kwargs)
            return bst, _evals_result, _additional_results

        # Make sure that train is called on the server node.
        _wrapped = force_on_current_node(_wrapped)

        bst, train_evals_result, train_additional_results = ray.get(
            _wrapped.remote(
                params,
                dtrain,
                *args,
                valid_sets=valid_sets,
                valid_names=valid_names,
                evals=evals,
                ray_params=ray_params,
                _remote=False,
                **kwargs,
            ))
        if isinstance(evals_result, dict):
            evals_result.update(train_evals_result)
        if isinstance(additional_results, dict):
            additional_results.update(train_additional_results)
        return bst

    start_time = time.time()

    ray_params = _validate_ray_params(ray_params)

    params = params.copy()

    if evals and valid_sets:
        raise ValueError(
            "Specifying both `evals` and `valid_sets` is ambiguous.")

    # LightGBM specific - capture whether local_listen_port or its aliases
    # were provided
    listen_port_in_params = any(
        alias in params for alias in _ConfigAliases.get("local_listen_port"))

    # LightGBM specific - capture whether machines or its aliases
    # were provided
    machines_in_params = any(
        alias in params for alias in _ConfigAliases.get("machines"))

    # LightGBM specific - validate machines and local_listening_port
    machine_addresses = None
    listen_port = None
    if machines_in_params:
        params = _choose_param_value(
            main_param_name="machines", params=params, default_value=None)
        machines = params["machines"]
        machine_addresses = machines.split(",")
        if len(set(machine_addresses)) != len(machine_addresses):
            raise ValueError(
                f"Found duplicates in `machines` ({machines}). Each entry in "
                "`machines` must be a unique IP-port combination.")
        if len(machine_addresses) != ray_params.num_actors:
            raise ValueError(
                f"`num_actors` in `ray_params` ({ray_params.num_actors}) must "
                "match the number of IP-port combinations in `machines` "
                f"({len(machine_addresses)}).")
        logger.info(f"Using user passed machines {machine_addresses}")
    if listen_port_in_params:
        params = _choose_param_value(
            main_param_name="local_listen_port",
            params=params,
            default_value=None)
        listen_port = params["local_listen_port"]
        logger.info(f"Using user passed local_listen_port {listen_port}")

    max_actor_restarts = ray_params.max_actor_restarts \
        if ray_params.max_actor_restarts >= 0 else float("inf")
    _assert_ray_support()

    if not isinstance(dtrain, RayDMatrix):
        raise ValueError(
            "The `dtrain` argument passed to `train()` is not a RayDMatrix, "
            "but of type {}. "
            "\nFIX THIS by instantiating a RayDMatrix first: "
            "`dtrain = RayDMatrix(data=data, label=label)`.".format(
                type(dtrain)))

    added_tune_callback = _try_add_tune_callback(kwargs)
    # LightGBM currently does not support elastic training.
    if ray_params.elastic_training:
        raise ValueError("Elastic Training cannot be used with LightGBM. "
                         "Please disable elastic_training in `ray_params` "
                         "in order to use LightGBM-Ray.")

    params = _choose_param_value(
        main_param_name="tree_learner", params=params, default_value="data")

    params = _choose_param_value(
        main_param_name="device_type", params=params, default_value="cpu")

    if added_tune_callback:
        # Don't autodetect resources when used with Tune.
        cpus_per_actor = ray_params.cpus_per_actor
        gpus_per_actor = max(0, ray_params.gpus_per_actor)
    else:
        cpus_per_actor, gpus_per_actor = _autodetect_resources(
            ray_params=ray_params,
            use_tree_method="device_type" in params
            and params["device_type"] is not None
            and params["device_type"] != "cpu")

    allowed_tree_learners = {
        "data", "data_parallel", "voting", "voting_parallel"
        # not yet supported in LightGBM python API
        # (as of ver 3.2.1)
        # "feature", "feature_parallel",
    }
    if params["tree_learner"] not in allowed_tree_learners:
        warnings.warn(
            f"Parameter tree_learner set to {params['tree_learner']},"
            " which is not allowed. Using 'data' as default")
        params["tree_learner"] = "data"

    for param_alias in _ConfigAliases.get("num_machines", "num_threads",
                                          "num_iterations", "n_estimators"):
        if param_alias in params:
            warnings.warn(f"Parameter {param_alias} will be ignored.")
            params.pop(param_alias)

    if not ("verbose" in kwargs and verbose_eval is True):
        kwargs["verbose"] = verbose_eval

    if gpus_per_actor > 0 and params["device_type"] == "cpu":
        warnings.warn(
            f"GPUs have been assigned to the actors, but the current LightGBM "
            f"device type is set to 'cpu'. Thus, GPUs will "
            f"currently not be used. To enable GPUs usage, please set the "
            f"`device_type` to a GPU-compatible option, "
            f"e.g. `gpu`.")

    if gpus_per_actor == 0 and cpus_per_actor == 0:
        raise ValueError("cpus_per_actor and gpus_per_actor both cannot be "
                         "0. Are you sure your cluster has CPUs available?")

    if ray_params.elastic_training and ray_params.max_failed_actors == 0:
        raise ValueError(
            "Elastic training enabled but the maximum number of failed "
            "actors is set to 0. This means that elastic training is "
            "effectively disabled. Please set `RayParams.max_failed_actors` "
            "to something larger than 0 to enable elastic training.")

    if ray_params.elastic_training and ray_params.max_actor_restarts == 0:
        raise ValueError(
            "Elastic training enabled but the maximum number of actor "
            "restarts is set to 0. This means that elastic training is "
            "effectively disabled. Please set `RayParams.max_actor_restarts` "
            "to something larger than 0 to enable elastic training.")

    if not dtrain.has_label:
        raise ValueError(
            "Training data has no label set. Please make sure to set "
            "the `label` argument when initializing `RayDMatrix()` "
            "for data you would like to train on.")

    if not dtrain.loaded and not dtrain.distributed:
        dtrain.load_data(ray_params.num_actors)

    if valid_sets is not None:
        evals = []
        if isinstance(valid_sets, RayDMatrix):
            valid_sets = [valid_sets]
        if isinstance(valid_names, str):
            valid_names = [valid_names]
        for i, valid_data in enumerate(valid_sets):
            if valid_names is not None and len(valid_names) > i:
                evals.append((valid_data, valid_names[i]))
            else:
                evals.append((valid_data, f"valid_{i}"))

    if evals:
        for (deval, name) in evals:
            if not isinstance(deval, RayDMatrix):
                raise ValueError("Evaluation data must be a `RayDMatrix`, got "
                                 f"{type(deval)}.")
            if not deval.has_label:
                raise ValueError(
                    "Evaluation data has no label set. Please make sure to set"
                    " the `label` argument when initializing `RayDMatrix()` "
                    "for data you would like to evaluate on.")
            if not deval.loaded and not deval.distributed:
                deval.load_data(ray_params.num_actors)

    bst = None
    train_evals_result = {}
    train_additional_results = {}

    tries = 0
    checkpoint = _Checkpoint()  # Keep track of latest checkpoint
    current_results = {}  # Keep track of additional results
    actors = [None] * ray_params.num_actors  # All active actors
    pending_actors = {}

    # Create the Queue and Event actors.
    queue, stop_event = _create_communication_processes(added_tune_callback)

    placement_strategy = None
    if not ray_params.elastic_training:
        if added_tune_callback:
            if TUNE_USING_PG:
                # If Tune is using placement groups, then strategy has already
                # been set. Don't create an additional placement_group here.
                placement_strategy = None
            else:
                placement_strategy = "PACK"
        elif bool(_USE_SPREAD_STRATEGY):
            placement_strategy = "SPREAD"

    if placement_strategy is not None:
        pg = _create_placement_group(cpus_per_actor, gpus_per_actor,
                                     ray_params.resources_per_actor,
                                     ray_params.num_actors, placement_strategy)
    else:
        pg = None

    start_actor_ranks = set(range(ray_params.num_actors))  # Start these

    total_training_time = 0.
    boost_rounds_left = num_boost_round
    last_checkpoint_value = checkpoint.value
    while tries <= max_actor_restarts:
        # Only update number of iterations if the checkpoint changed
        # If it didn't change, we already subtracted the iterations.
        if checkpoint.iteration >= 0 and \
                checkpoint.value != last_checkpoint_value:
            boost_rounds_left -= checkpoint.iteration + 1

        last_checkpoint_value = checkpoint.value

        logger.debug(f"Boost rounds left: {boost_rounds_left}")

        training_state = _TrainingState(
            actors=actors,
            queue=queue,
            stop_event=stop_event,
            checkpoint=checkpoint,
            additional_results=current_results,
            training_started_at=0.,
            placement_group=pg,
            failed_actor_ranks=start_actor_ranks,
            pending_actors=pending_actors)

        try:
            bst, train_evals_result, train_additional_results = _train(
                params,
                dtrain,
                model_factory,
                boost_rounds_left,
                *args,
                evals=evals,
                ray_params=ray_params,
                cpus_per_actor=cpus_per_actor,
                gpus_per_actor=gpus_per_actor,
                _training_state=training_state,
                machine_addresses=machine_addresses,
                listen_port=listen_port,
                **kwargs)
            if training_state.training_started_at > 0.:
                total_training_time += time.time(
                ) - training_state.training_started_at
            break
        except (RayActorError, RayTaskError) as exc:
            if training_state.training_started_at > 0.:
                total_training_time += time.time(
                ) - training_state.training_started_at
            alive_actors = sum(1 for a in actors if a is not None)
            start_again = False
            if ray_params.elastic_training:
                if alive_actors < ray_params.num_actors - \
                        ray_params.max_failed_actors:
                    raise RuntimeError(
                        "A Ray actor died during training and the maximum "
                        "number of dead actors in elastic training was "
                        "reached. Shutting down training.") from exc

                # Do not start new actors before resuming training
                # (this might still restart actors during training)
                start_actor_ranks.clear()

                if exc.__cause__ and isinstance(exc.__cause__,
                                                RayXGBoostActorAvailable):
                    # New actor available, integrate into training loop
                    logger.info(
                        f"A new actor became available. Re-starting training "
                        f"from latest checkpoint with new actor. "
                        f"This will use {alive_actors} existing actors and "
                        f"start {len(start_actor_ranks)} new actors. "
                        f"Sleeping for 10 seconds for cleanup.")
                    tries -= 1  # This is deliberate so shouldn't count
                    start_again = True

                elif tries + 1 <= max_actor_restarts:
                    if exc.__cause__ and isinstance(exc.__cause__,
                                                    RayXGBoostTrainingError):
                        logger.warning(f"Caught exception: {exc.__cause__}")
                    logger.warning(
                        f"A Ray actor died during training. Trying to "
                        f"continue training on the remaining actors. "
                        f"This will use {alive_actors} existing actors and "
                        f"start {len(start_actor_ranks)} new actors. "
                        f"Sleeping for 10 seconds for cleanup.")
                    start_again = True

            elif tries + 1 <= max_actor_restarts:
                if exc.__cause__ and isinstance(exc.__cause__,
                                                RayXGBoostTrainingError):
                    logger.warning(f"Caught exception: {exc.__cause__}")
                logger.warning(
                    f"A Ray actor died during training. Trying to restart "
                    f"and continue training from last checkpoint "
                    f"(restart {tries + 1} of {max_actor_restarts}). "
                    f"This will use {alive_actors} existing actors and start "
                    f"{len(start_actor_ranks)} new actors. "
                    f"Sleeping for 10 seconds for cleanup.")
                start_again = True

            if start_again:
                time.sleep(5)
                queue.shutdown()
                stop_event.shutdown()
                time.sleep(5)
                queue, stop_event = _create_communication_processes()
            else:
                raise RuntimeError(
                    f"A Ray actor died during training and the maximum number "
                    f"of retries ({max_actor_restarts}) is exhausted."
                ) from exc
            tries += 1

    total_time = time.time() - start_time

    train_additional_results["training_time_s"] = total_training_time
    train_additional_results["total_time_s"] = total_time

    logger.info("[RayLightGBM] Finished LightGBM training on training data "
                "with total N={total_n:,} in {total_time_s:.2f} seconds "
                "({training_time_s:.2f} pure LightGBM training time).".format(
                    **train_additional_results))

    _shutdown(
        actors=actors,
        pending_actors=pending_actors,
        queue=queue,
        event=stop_event,
        placement_group=pg,
        force=False)

    if isinstance(evals_result, dict):
        evals_result.update(train_evals_result)
    if isinstance(additional_results, dict):
        additional_results.update(train_additional_results)

    return bst


def _predict(model: LGBMModel, data: RayDMatrix, method: str,
             ray_params: RayParams, **kwargs):
    _assert_ray_support()

    if not ray.is_initialized():
        ray.init()

    # Create remote actors
    actors = [
        _create_actor(
            rank=i,
            num_actors=ray_params.num_actors,
            model_factory=None,
            num_cpus_per_actor=ray_params.cpus_per_actor,
            num_gpus_per_actor=ray_params.gpus_per_actor
            if ray_params.gpus_per_actor >= 0 else 0,
            resources_per_actor=ray_params.resources_per_actor,
            distributed_callbacks=ray_params.distributed_callbacks)
        for i in range(ray_params.num_actors)
    ]
    logger.info(f"[RayLightGBM] Created {len(actors)} remote actors.")

    # Split data across workers
    wait_load = []
    for actor in actors:
        wait_load.extend(_trigger_data_load(actor, data, []))

    try:
        ray.get(wait_load)
    except Exception as exc:
        logger.warning(f"Caught an error during prediction: {str(exc)}")
        _shutdown(actors, force=True)
        raise

    # Put model into object store
    model_ref = ray.put(model)

    logger.info("[RayLightGBM] Starting LightGBM prediction.")

    # Train
    fut = [
        actor.predict.remote(model_ref, data, method, **kwargs)
        for actor in actors
    ]

    try:
        actor_results = ray.get(fut)
    except Exception as exc:
        logger.warning(f"Caught an error during prediction: {str(exc)}")
        _shutdown(actors=actors, force=True)
        raise

    _shutdown(actors=actors, force=False)

    return combine_data(data.sharding, actor_results)


def predict(model: Union[LGBMModel, Booster],
            data: RayDMatrix,
            method: str = "predict",
            ray_params: Union[None, RayParams, Dict] = None,
            _remote: Optional[bool] = None,
            **kwargs) -> Optional[np.ndarray]:
    """Distributed LightGBM predict via Ray.

    This function will connect to a Ray cluster, create ``num_actors``
    remote actors, send data shards to them, and have them predict labels
    using an LightGBM model. The results are then combined and
    returned.

    Args:
        model (Union[LGBMModel, Booster]): Model or booster object to
            call for prediction.
        data (RayDMatrix): Data object containing the prediction data.
        method (str): Name of estimator method to use for prediction.
        ray_params (Union[None, RayParams, Dict]): Parameters to configure
            Ray-specific behavior. See :class:`RayParams` for a list of valid
            configuration parameters.
        _remote (bool): Whether to run the driver process in a remote
            function. This is enabled by default in Ray client mode.
        **kwargs: Keyword arguments will be passed to the local
            `xgb.predict()` calls.

    Returns: ``np.ndarray`` containing the predicted labels.

    """
    os.environ.setdefault("RAY_IGNORE_UNHANDLED_ERRORS", "1")

    if _remote is None:
        _remote = _is_client_connected() and \
                  not is_session_enabled()

    if not ray.is_initialized():
        ray.init()

    if _remote:
        return ray.get(
            ray.remote(num_cpus=0)(predict).remote(
                model, data, method, ray_params, _remote=False, **kwargs))

    _maybe_print_legacy_warning()

    ray_params = _validate_ray_params(ray_params)

    max_actor_restarts = ray_params.max_actor_restarts \
        if ray_params.max_actor_restarts >= 0 else float("inf")
    _assert_ray_support()

    if not isinstance(data, RayDMatrix):
        raise ValueError(
            "The `data` argument passed to `predict()` is not a RayDMatrix, "
            "but of type {}. "
            "\nFIX THIS by instantiating a RayDMatrix first: "
            "`data = RayDMatrix(data=data)`.".format(type(data)))

    tries = 0
    while tries <= max_actor_restarts:
        try:
            return _predict(
                model, data, method=method, ray_params=ray_params, **kwargs)
        except RayActorError:
            if tries + 1 <= max_actor_restarts:
                logger.warning(
                    "A Ray actor died during prediction. Trying to restart "
                    "prediction from scratch. "
                    "Sleeping for 10 seconds for cleanup.")
                time.sleep(10)
            else:
                raise RuntimeError(
                    "A Ray actor died during prediction and the maximum "
                    "number of retries ({}) is exhausted.".format(
                        max_actor_restarts))
            tries += 1
    return None
