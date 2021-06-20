from typing import Tuple, Dict, Any, List, Optional, Type, Union, Sequence

from copy import deepcopy
import time
import logging
import os
import threading

import numpy as np
import pandas as pd

from lightgbm import LGBMModel, LGBMRanker
from lightgbm.basic import _choose_param_value, _ConfigAliases
from lightgbm.basic import LightGBMError
from lightgbm.callback import EarlyStopException

import ray

import xgboost as xgb
from xgboost_ray.main import RayXGBoostActor, LEGACY_MATRIX, RayDeviceQuantileDMatrix, concat_dataframes, _set_omp_num_threads, Queue, Event, DistributedCallback, _handle_queue, STATUS_FREQUENCY_S, RayActorError, ELASTIC_RESTART_DISABLED, pickle, _PrepareActorTask, RayParams, _TrainingState, _is_client_connected, is_session_enabled, force_on_current_node, _assert_ray_support, _validate_ray_params, _maybe_print_legacy_warning, _try_add_tune_callback, _autodetect_resources, _Checkpoint, _create_communication_processes, TUNE_USING_PG, _USE_SPREAD_STRATEGY, RayTaskError, RayXGBoostActorAvailable, RayXGBoostTrainingError, _create_placement_group, _shutdown, PlacementGroup, ActorHandle, RayXGBoostTrainingStopped, combine_data, _trigger_data_load
from xgboost_ray import RayDMatrix

from lightgbm_ray.util import find_free_port, lgbm_network_free

logger = logging.getLogger(__name__)




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

    #data.update_matrix_properties(matrix)
    #return matrix


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
        self.network_params = {} if not network_params else network_params.copy(
        )
        if "time_out" not in self.network_params:
            self.network_params["time_out"] = 120
        self.model_factory = model_factory
        return super().__init__(rank=rank,
                                num_actors=num_actors,
                                queue=queue,
                                stop_event=stop_event,
                                checkpoint_frequency=checkpoint_frequency,
                                distributed_callbacks=distributed_callbacks)

    def _save_checkpoint_callback(self):
        return

    def _stop_callback(self):
        return

    def find_free_address(self):
        return (self.ip(), find_free_port())

    def port(self) -> Optional[int]:
        return self.network_params.get("local_listen_port", None)

    def set_network_params(
        self,
        machines: str,
        local_listen_port: int,
        num_machines: int,
        time_out: Optional[int] = None,
    ):
        self.network_params["machines"] = machines
        self.network_params["local_listen_port"] = local_listen_port
        self.network_params["num_machines"] = num_machines
        if time_out is not None:
            self.network_params["time_out"] = time_out

    def load_data(self, data: RayDMatrix):
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
              dtrain: RayDMatrix, evals: Tuple[RayDMatrix, str], *args,
              **kwargs) -> Dict[str, Any]:
        if self.model_factory is None:
            raise ValueError("model_factory cannot be None for training")

        from lightgbm.basic import _LIB

        self._distributed_callbacks.before_train(self)

        num_threads = _set_omp_num_threads()

        local_params = _choose_param_value(
            main_param_name="num_threads",
            params=params,
            default_value=num_threads if num_threads > 0 else sum(
                num
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
            local_evals.append((self._data[deval]["data"], self._data[deval]["label"]))
            local_eval_names.append(name)
            local_eval_sample_weights.append(self._data[deval]["weight"])
            local_eval_init_scores.append(self._data[deval]["base_margin"])

        # if "callbacks" in kwargs:
        #     callbacks = kwargs["callbacks"] or []
        # else:
        #     callbacks = []
        # callbacks.append(self._save_checkpoint_callback())
        # callbacks.append(self._stop_callback())
        # kwargs["callbacks"] = callbacks

        result_dict = {}
        error_dict = {}

        network_params = self.network_params
        local_params.update(network_params)

        is_ranker = issubclass(self.model_factory, LGBMRanker)

        print(local_params)

        # We run xgb.train in a thread to be able to react to the stop event.
        def _train():
            try:
                model = self.model_factory(**local_params)
                with lgbm_network_free(_LIB):
                    if is_ranker:
                        # missing group arg
                        model.fit(local_dtrain["data"],
                                local_dtrain["label"],
                                sample_weight=local_dtrain["weight"],
                                init_score=local_dtrain["base_margin"],
                                eval_set=local_evals,
                                eval_names=local_eval_names,
                                eval_sample_weight=local_eval_sample_weights,
                                eval_init_score=local_eval_init_scores,
                                **kwargs)
                    else:
                        model.fit(local_dtrain["data"],
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
            except EarlyStopException:
                # Usually this should be caught by XGBoost core.
                # Silent fail, will be raised as RayXGBoostTrainingStopped.
                return
            except LightGBMError as e:
                print(e)
                error_dict.update({"exception": e})
                return

        thread = threading.Thread(target=_train)
        thread.daemon = True
        thread.start()
        while thread.is_alive():
            thread.join(timeout=0)
            if self._stop_event.is_set():
                raise RayXGBoostTrainingStopped("Training was interrupted.")
            time.sleep(0.1)

        if not result_dict:
            raise_from = error_dict.get("exception", None)
            raise RayXGBoostTrainingError("Training failed.") from raise_from

        thread.join()
        self._distributed_callbacks.after_train(self, result_dict)

        if not return_bst:
            result_dict.pop("bst", None)

        return result_dict

    def predict(self, model: LGBMModel, data: RayDMatrix, **kwargs):
        self._distributed_callbacks.before_predict(self)

        _set_omp_num_threads()

        if data not in self._data:
            self.load_data(data)
        local_data = self._data[data]["data"]

        if hasattr(model, "predict_proba"):
            predictions = model.predict_proba(local_data, **kwargs)
        else:
            predictions = model.predict(local_data, **kwargs)

        if predictions.ndim == 1:
            callback_predictions = pd.Series(predictions)
        else:
            callback_predictions = pd.DataFrame(predictions)
        self._distributed_callbacks.after_predict(self, callback_predictions)
        return predictions


@ray.remote
class _RemoteRayLightGBMActor(RayLightGBMActor):
    pass

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
    distributed_callbacks: Optional[Sequence[DistributedCallback]] = None
) -> ActorHandle:
    return _RemoteRayLightGBMActor.options(
        num_cpus=num_cpus_per_actor,
        num_gpus=num_gpus_per_actor,
        resources=resources_per_actor,
        placement_group=placement_group).remote(
            rank=rank,
            num_actors=num_actors,
            model_factory=model_factory,
            queue=queue,
            checkpoint_frequency=checkpoint_frequency,
            distributed_callbacks=distributed_callbacks)


def _train(params: Dict,
           dtrain: RayDMatrix,
           model_factory: Type[LGBMModel],
           *args,
           evals=(),
           ray_params: RayParams,
           cpus_per_actor: int,
           gpus_per_actor: int,
           _training_state: _TrainingState,
           **kwargs) -> Tuple[xgb.Booster, Dict, Dict]:
    """This is the local train function wrapped by :func:`train() <train>`.

    This function can be thought of one invocation of a multi-actor xgboost
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

    # capture whether local_listen_port or its aliases were provided
    listen_port_in_params = any(
        alias in params for alias in _ConfigAliases.get("local_listen_port"))

    # capture whether machines or its aliases were provided
    machines_in_params = any(alias in params
                             for alias in _ConfigAliases.get("machines"))

    if "n_jobs" in params:
        if params["n_jobs"] > cpus_per_actor:
            raise ValueError(
                "Specified number of threads greater than number of CPUs. "
                "\nFIX THIS by passing a lower value for the `n_jobs` "
                "parameter or a higher number for `cpus_per_actor`.")
    else:
        params["n_jobs"] = cpus_per_actor

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
            distributed_callbacks=ray_params.distributed_callbacks)
        # Set actor entry in our list
        _training_state.actors[i] = actor
        # Remove from this set so it is not created again
        _training_state.failed_actor_ranks.remove(i)
        newly_created += 1

    alive_actors = sum(1 for a in _training_state.actors if a is not None)
    logger.info(f"[RayXGBoost] Created {newly_created} new actors "
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

    logger.info("[RayXGBoost] Starting XGBoost training.")

    # Start Rabit tracker for gradient sharing
    #rabit_process, env = _start_rabit_tracker(alive_actors)
    #rabit_args = [("%s=%s" % item).encode() for item in env.items()]

    # Load checkpoint if we have one. In that case we need to adjust the
    # number of training rounds.
    if _training_state.checkpoint.value:
        kwargs["xgb_model"] = pickle.loads(_training_state.checkpoint.value)
        if _training_state.checkpoint.iteration == -1:
            # -1 means training already finished.
            logger.error(
                "Trying to load continue from checkpoint, but the checkpoint"
                "indicates training already finished. Returning last"
                "checkpointed model instead.")
            return kwargs["xgb_model"], {}, _training_state.additional_results

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
    addresses = ray.get(
        [actor.find_free_address.remote() for actor in live_actors])
    ips, ports = zip(*addresses)
    ips = list(ips)
    ports = list(ports)
    machines = ','.join([f"{ip}:{port}" for ip, port in addresses])

    for i, actor in enumerate(live_actors):
        actor.set_network_params.remote(machines, ports[i], len(live_actors),
                                        params.get("time_out", 120))

    training_futures = [
        actor.train.remote(
            i == 0,  # return_bst
            params,
            dtrain,
            evals,
            *args,
            **kwargs) for i, actor in enumerate(live_actors)
    ]

    # # Failure handling loop. Here we wait until all training tasks finished.
    # # If a training task fails, we stop training on the remaining actors,
    # # check which ones are still alive, and raise the error.
    # # The train() wrapper function will then handle the error.
    # start_wait = time.time()
    # last_status = start_wait
    # try:
    #     not_ready = training_futures
    #     while not_ready:
    #         if _training_state.queue:
    #             _handle_queue(
    #                 queue=_training_state.queue,
    #                 checkpoint=_training_state.checkpoint,
    #                 callback_returns=callback_returns)

    #         if ray_params.elastic_training \
    #                 and not ELASTIC_RESTART_DISABLED:
    #             _maybe_schedule_new_actors(
    #                 training_state=_training_state,
    #                 num_cpus_per_actor=cpus_per_actor,
    #                 num_gpus_per_actor=gpus_per_actor,
    #                 resources_per_actor=ray_params.resources_per_actor,
    #                 ray_params=ray_params,
    #                 load_data=load_data)

    #             # This may raise RayXGBoostActorAvailable
    #             _update_scheduled_actor_states(_training_state)

    #         if time.time() >= last_status + STATUS_FREQUENCY_S:
    #             wait_time = time.time() - start_wait
    #             logger.info(f"Training in progress "
    #                         f"({wait_time:.0f} seconds since last restart).")
    #             last_status = time.time()

    #         ready, not_ready = ray.wait(
    #             not_ready, num_returns=len(not_ready), timeout=1)
    #         ray.get(ready)

    #     # Get items from queue one last time
    #     if _training_state.queue:
    #         _handle_queue(
    #             queue=_training_state.queue,
    #             checkpoint=_training_state.checkpoint,
    #             callback_returns=callback_returns)

    # # The inner loop should catch all exceptions
    # except Exception as exc:
    #     logger.debug(f"Caught exception in training loop: {exc}")

    #     # Stop all other actors from training
    #     _training_state.stop_event.set()

    #     # Check which actors are still alive
    #     _get_actor_alive_status(_training_state.actors, handle_actor_failure)

    #     # Todo: Try to fetch newer checkpoint, store in `_checkpoint`
    #     # Shut down rabit
    #     _stop_rabit_tracker(rabit_process)

    #     raise RayActorError from exc

    # # Training is now complete.
    # # Stop Rabit tracking process
    # _stop_rabit_tracker(rabit_process)

    # Get all results from all actors.
    all_results: List[Dict[str, Any]] = ray.get(training_futures)

    print(all_results)

    # All results should be the same because of Rabit tracking. But only
    # the first one actually returns its bst object.
    bst = all_results[0]["bst"]
    evals_result = all_results[0]["evals_result"]

    if not listen_port_in_params:
        for param in _ConfigAliases.get('local_listen_port'):
            bst._other_params.pop(param, None)

    if not machines_in_params:
        for param in _ConfigAliases.get('machines'):
            bst._other_params.pop(param, None)

    for param in _ConfigAliases.get('num_machines', 'timeout'):
        bst._other_params.pop(param, None)

    if callback_returns:
        _training_state.additional_results[
            "callback_returns"] = callback_returns

    total_n = sum(res["train_n"] or 0 for res in all_results)

    _training_state.additional_results["total_n"] = total_n

    return bst, evals_result, _training_state.additional_results


def train(params: Dict,
          dtrain: RayDMatrix,
          model_factory: Type[LGBMModel] = LGBMModel,
          num_boost_round: int = 10,
          *args,
          evals: Union[List[Tuple[RayDMatrix, str]], Tuple[RayDMatrix,
                                                           str]] = (),
          evals_result: Optional[Dict] = None,
          additional_results: Optional[Dict] = None,
          ray_params: Union[None, RayParams, Dict] = None,
          _remote: Optional[bool] = None,
          **kwargs) -> LGBMModel:
    """Distributed XGBoost training via Ray.

    This function will connect to a Ray cluster, create ``num_actors``
    remote actors, send data shards to them, and have them train an
    XGBoost classifier. The XGBoost parameters will be shared and combined
    via Rabit's all-reduce protocol.

    If running inside a Ray Tune session, this function will automatically
    handle results to tune for hyperparameter search.

    Failure handling:

    XGBoost on Ray supports automatic failure handling that can be configured
    with the :class:`ray_params <RayParams>` argument. If an actor or local
    training task dies, the Ray actor is marked as dead, and there are
    three options on how to proceed.

    First, if ``ray_params.elastic_training`` is ``True`` and
    the number of dead actors is below ``ray_params.max_failed_actors``,
    training will continue right away with fewer actors. No data will be
    loaded again and the latest available checkpoint will be used.
    A maximum of ``ray_params.max_actor_restarts`` restarts will be tried
    before exiting.

    Second, if ``ray_params.elastic_training`` is ``False`` and
    the number of restarts is below ``ray_params.max_actor_restarts``,
    Ray will try to schedule the dead actor again, load the data shard
    on this actor, and then continue training from the latest checkpoint.

    Third, if none of the above is the case, training is aborted.

    Args:
        params (Dict): parameter dict passed to ``xgboost.train()``
        dtrain (RayDMatrix): Data object containing the training data.
        evals (Union[List[Tuple[RayDMatrix, str]], Tuple[RayDMatrix, str]]):
            ``evals`` tuple passed to ``xgboost.train()``.
        evals_result (Optional[Dict]): Dict to store evaluation results in.
        additional_results (Optional[Dict]): Dict to store additional results.
        ray_params (Union[None, RayParams, Dict]): Parameters to configure
            Ray-specific behavior. See :class:`RayParams` for a list of valid
            configuration parameters.
        _remote (bool): Whether to run the driver process in a remote
            function. This is enabled by default in Ray client mode.
        **kwargs: Keyword arguments will be passed to the local
            `xgb.train()` calls.

    Returns: An ``xgboost.Booster`` object.
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
            bst = train(*args,
                        num_boost_round=num_boost_round,
                        evals_result=_evals_result,
                        additional_results=_additional_results,
                        **kwargs)
            return bst, _evals_result, _additional_results

        # Make sure that train is called on the server node.
        _wrapped = force_on_current_node(_wrapped)

        bst, train_evals_result, train_additional_results = ray.get(
            _wrapped.remote(
                params,
                dtrain,
                *args,
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

    _maybe_print_legacy_warning()

    start_time = time.time()

    ray_params = _validate_ray_params(ray_params)

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
    # Tune currently does not support elastic training.
    if added_tune_callback and ray_params.elastic_training and not bool(
            os.getenv("RXGB_ALLOW_ELASTIC_TUNE", "0")):
        raise ValueError("Elastic Training cannot be used with Ray Tune. "
                         "Please disable elastic_training in RayParams in "
                         "order to use xgboost_ray with Tune.")

    if added_tune_callback:
        # Don't autodetect resources when used with Tune.
        cpus_per_actor = ray_params.cpus_per_actor
        gpus_per_actor = max(0, ray_params.gpus_per_actor)
    else:
        cpus_per_actor, gpus_per_actor = _autodetect_resources(
            ray_params=ray_params,
            use_tree_method="tree_method" in params
            and params["tree_method"] is not None
            and params["tree_method"].startswith("gpu"))

    params = _choose_param_value(main_param_name="tree_learner",
                                 params=params,
                                 default_value="data")

    params = _choose_param_value(main_param_name="device_type",
                                 params=params,
                                 default_value="cpu")

    allowed_tree_learners = {
        'data', 'data_parallel', 
        'voting','voting_parallel'
        # not yet supported in LightGBM python API
        #'feature', 'feature_parallel', 
    }
    if params["tree_learner"] not in allowed_tree_learners:
        logger.warning(
            'Parameter tree_learner set to %s, which is not allowed. Using "data" as default'
            % params['tree_learner'])
        params['tree_learner'] = 'data'

    for param_alias in _ConfigAliases.get('num_machines', 'num_threads'):
        if param_alias in params:
            logger.warning(f"Parameter {param_alias} will be ignored.")
            params.pop(param_alias)

    if gpus_per_actor > 0 and params["device_type"] == "cpu":
        logger.warning(
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

    for (deval, name) in evals:
        if not deval.has_label:
            raise ValueError(
                "Evaluation data has no label set. Please make sure to set "
                "the `label` argument when initializing `RayDMatrix()` "
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

        training_state = _TrainingState(actors=actors,
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

    logger.info("[RayXGBoost] Finished XGBoost training on training data "
                "with total N={total_n:,} in {total_time_s:.2f} seconds "
                "({training_time_s:.2f} pure XGBoost training time).".format(
                    **train_additional_results))

    _shutdown(actors=actors,
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


def _predict(model: LGBMModel, data: RayDMatrix, ray_params: RayParams,
             **kwargs):
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
    logger.info(f"[RayXGBoost] Created {len(actors)} remote actors.")

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

    logger.info("[RayXGBoost] Starting XGBoost prediction.")

    # Train
    fut = [actor.predict.remote(model_ref, data, **kwargs) for actor in actors]

    try:
        actor_results = ray.get(fut)
    except Exception as exc:
        logger.warning(f"Caught an error during prediction: {str(exc)}")
        _shutdown(actors=actors, force=True)
        raise

    _shutdown(actors=actors, force=False)

    return combine_data(data.sharding, actor_results)

def predict(model: LGBMModel,
            data: RayDMatrix,
            ray_params: Union[None, RayParams, Dict] = None,
            _remote: Optional[bool] = None,
            **kwargs) -> Optional[np.ndarray]:
    """Distributed XGBoost predict via Ray.

    This function will connect to a Ray cluster, create ``num_actors``
    remote actors, send data shards to them, and have them predict labels
    using an XGBoost booster model. The results are then combined and
    returned.

    Args:
        model (xgb.Booster): Booster object to call for prediction.
        data (RayDMatrix): Data object containing the prediction data.
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
                model, data, ray_params, _remote=False, **kwargs))

    _maybe_print_legacy_warning()

    ray_params = _validate_ray_params(ray_params)

    max_actor_restarts = ray_params.max_actor_restarts \
        if ray_params.max_actor_restarts >= 0 else float("inf")
    _assert_ray_support()

    if not isinstance(data, RayDMatrix):
        raise ValueError(
            "The `data` argument passed to `train()` is not a RayDMatrix, "
            "but of type {}. "
            "\nFIX THIS by instantiating a RayDMatrix first: "
            "`data = RayDMatrix(data=data)`.".format(type(data)))

    tries = 0
    while tries <= max_actor_restarts:
        try:
            return _predict(model, data, ray_params=ray_params, **kwargs)
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
