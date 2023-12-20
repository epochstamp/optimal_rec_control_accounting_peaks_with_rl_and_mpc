from ray.rllib.evaluation.worker_set import WorkerSet
from typing import Optional, Callable, Any
from ray.rllib.utils.typing import AgentID, EnvType, ModelGradients, MultiAgentPolicyConfigDict, PartialAlgorithmConfigDict, PolicyID, PolicyState, SampleBatchType
import gymnasium as gym
import inspect
from ray.rllib.evaluation.sampler import SyncSampler
import torch
from typing import (
    Callable,
    Container,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)
import torch as tc
from ray.rllib.policy.sample_batch import (
    concat_samples,
    DEFAULT_POLICY_ID
)
from multiprocess import Pool, Manager
from ray.actor import ActorHandle
from ray.rllib.core.learner import LearnerGroup
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.policy.policy import Policy, PolicyState
from ray.rllib.utils.annotations import DeveloperAPI
from copy import deepcopy
from ray.rllib.utils.deprecation import (
    Deprecated,
    DEPRECATED_VALUE,
)
import random
import numpy as np
from ray.rllib.utils.typing import (
    AgentID,
    EnvCreator,
    EnvType,
    EpisodeID,
    PartialAlgorithmConfigDict,
    PolicyID,
    SampleBatchType,
    TensorType,
)

GLOBAL_WORKERS = dict()

class ParallelRolloutWorkerWrapper(RolloutWorker):
    def __init__(self, idx_worker, idx_worker_set):
        self._idx_worker = idx_worker
        self._idx_worker_set = idx_worker_set

    def sample(self, **kwargs) -> SampleBatchType:
        global GLOBAL_WORKERS
        rollout_worker = GLOBAL_WORKERS[self._idx_worker_set][self._idx_worker]
        #rollout_worker.set_weights(rollout_worker.get_weights())
        #print(rollout_worker.get_state())
        #rollout_worker.set_state(rollout_worker.get_state())
        #idx_workerset_number = GLOBAL_VALUES[self._idx_worker_set]
        #idx_unroll = self._idx_worker + idx_workerset_number
        #SampleBatchBuilder._next_unroll_id = idx_unroll
        #print(idx_unroll)
        batch = self._sample(rollout_worker, **kwargs)
        return batch

        # Always do writes prior to compression for consistency and to allow
        # for better compression inside the writer.
        #rollout_worker.output_writer.write(batch)

        #if rollout_worker.config.compress_observations:
        #    batch.compress(bulk=rollout_worker.config.compress_observations == "bulk")

        #if rollout_worker.config.fake_sampler:
        #    rollout_worker.last_batch = batch
    
    def _sample(self, rollout_worker:RolloutWorker, **kwargs) -> SampleBatchType:
        """Returns a batch of experience sampled from this worker.

        This method must be implemented by subclasses.

        Returns:
            A columnar batch of experiences (e.g., tensors) or a MultiAgentBatch.

        Examples:
            >>> import gymnasium as gym
            >>> from ray.rllib.evaluation.rollout_worker import RolloutWorker
            >>> from ray.rllib.algorithms.pg.pg_tf_policy import PGTF1Policy
            >>> worker = RolloutWorker( # doctest: +SKIP
            ...   env_creator=lambda _: gym.make("CartPole-v1"), # doctest: +SKIP
            ...   default_policy_class=PGTF1Policy, # doctest: +SKIP
            ...   config=AlgorithmConfig(), # doctest: +SKIP
            ... )
            >>> print(worker.sample()) # doctest: +SKIP
            SampleBatch({"obs": [...], "action": [...], ...})
        """
        #print(rollout_worker.total_rollout_fragment_length)
        if self._idx_worker != 0:
            random_seed = rollout_worker.random_seed
            random.seed(random_seed)
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
        
        if rollout_worker.config.fake_sampler and rollout_worker.last_batch is not None:
            return rollout_worker.last_batch
        elif rollout_worker.input_reader is None:
            raise ValueError(
                "RolloutWorker has no `input_reader` object! "
                "Cannot call `sample()`. You can try setting "
                "`create_env_on_driver` to True."
            )
        rollout_worker.worker_index = self._idx_worker
        batches = [rollout_worker.input_reader.next()]
        steps_so_far = (
            batches[0].count
            if rollout_worker.config.count_steps_by == "env_steps"
            else batches[0].agent_steps()
        )
        # In truncate_episodes mode, never pull more than 1 batch per env.
        # This avoids over-running the target batch size.
        if (
            rollout_worker.config.batch_mode == "truncate_episodes"
            and not rollout_worker.config.offline_sampling
        ):
            max_batches = rollout_worker.config.num_envs_per_worker
        else:
            max_batches = float("inf")
        while steps_so_far < rollout_worker.total_rollout_fragment_length and (
            len(batches) < max_batches
        ):
            batch = rollout_worker.input_reader.next()
            steps_so_far += (
                batch.count
                if rollout_worker.config.count_steps_by == "env_steps"
                else batch.agent_steps()
            )
            batches.append(batch)
        batch = concat_samples(batches)
        
        #for batch in batches:
        #    print(batch)
        #print()
        #print(len(batches))
        #print(len(batches))
        #print(batch["default_policy"]["unroll_id"])
        #print()
        #print(rollout_worker.total_rollout_fragment_length)
        
        #batch.count *= self._num_envs

        #rollout_worker.callbacks.on_sample_end(worker=rollout_worker, samples=batch)

        # Always do writes prior to compression for consistency and to allow
        # for better compression inside the writer.
        #rollout_worker.output_writer.write(batch)

        #if rollout_worker.config.compress_observations:
        #    batch.compress(bulk=rollout_worker.config.compress_observations == "bulk")

        #if rollout_worker.config.fake_sampler:
        #    rollout_worker.last_batch = batch
        return batch
    
    def sample_and_learn(self, expected_batch_size: int, num_sgd_iter: int, sgd_minibatch_size: str, standardize_fields: List[str]) -> Tuple[dict, int]:
        global GLOBAL_WORKER
        rollout_worker = GLOBAL_WORKER
        return rollout_worker.sample_and_learn(
            expected_batch_size=expected_batch_size,
            num_sgd_iter=num_sgd_iter,
            sgd_minibatch_size=sgd_minibatch_size,
            standardize_fields=standardize_fields
        )
    
    def sample_with_count(self) -> Tuple[SampleBatchType, int]:
        global GLOBAL_WORKER
        rollout_worker = GLOBAL_WORKER
        return rollout_worker.sample_with_count()

class ParallelWorkerSetWrapper(WorkerSet):

    def __init__(self, worker_set:WorkerSet, number_of_processes:int = 1, num_env_per_worker:int =1, id_worker_set=None):
        global GLOBAL_WORKERS
        self._id_worker_set = id_worker_set
        if self._id_worker_set is None:
            self._id_worker_set = id(self)
        rollout_worker=worker_set.local_worker()
        self._local_config = deepcopy(worker_set._local_config)
        self._local_config["rollout_fragment_length"] = rollout_worker.total_rollout_fragment_length//(number_of_processes*num_env_per_worker)
        rollout_worker._is_eval = rollout_worker.env.eval_env
        #print("***")
        GLOBAL_WORKERS[self._id_worker_set] = [
            RolloutWorker(
                env_creator=rollout_worker.env_creator,
                validate_env=None,
                default_policy_class=rollout_worker.default_policy_class,
                config=self._local_config,
                worker_index=i,
                num_workers=number_of_processes,
                recreated_worker=rollout_worker.recreated_worker,
                log_dir=None,
                spaces=rollout_worker.spaces,
                dataset_shards=rollout_worker._ds_shards,
            )
            for i in range(number_of_processes)
        ]
        for i, worker in enumerate(GLOBAL_WORKERS[self._id_worker_set]):
            worker._is_eval = rollout_worker.env.eval_env
        self._worker_set_parallel_local_worker = rollout_worker
        self._parallel_rollout_workers = [
            ParallelRolloutWorkerWrapper(
                idx, self._id_worker_set
            ) for idx in range(number_of_processes)
        ]
        self._worker_set = worker_set
        self._number_of_processes=number_of_processes

    def _setup(
        self,
        *,
        validate_env: Optional[Callable[[EnvType], None]] = None,
        config: Optional[Any] = None,
        num_workers: int = 0,
        local_worker: bool = True
    ):
        return self._worker_set._setup(
            validate_env=validate_env,
            config=config,
            num_workers=num_workers,
            local_worker=local_worker
        )

    def _get_spaces_from_remote_worker(self):
        return self._worker_set._get_spaces_from_remote_worker()
    
    def _target_inspected_funcs(self, fct):
        return "synchronous_parallel_sample" in fct or "evaluate" in fct

    @DeveloperAPI
    def local_worker(self) -> RolloutWorker:
        fct = inspect.stack()[1].function
        if self._target_inspected_funcs(fct):
            return self._worker_set_parallel_local_worker
        else:
            """Returns the local rollout worker."""
            return self._worker_set._local_worker

    @DeveloperAPI
    def healthy_worker_ids(self) -> List[int]:
        """Returns the list of remote worker IDs."""
        fct = inspect.stack()[1].function
        if self._target_inspected_funcs(fct):
            return list(range(self._number_of_processes))
        else:
            return self._worker_set.healthy_worker_ids()
    
    def _num_remote_workers(self, origin:str) -> int:
        """Returns the number of remote rollout workers."""
        if self._target_inspected_funcs(origin):
            return self._number_of_processes
        else:
            return self._worker_set.num_remote_workers()
    @DeveloperAPI
    def num_remote_workers(self) -> int:
        """Returns the number of remote rollout workers."""
        fct = inspect.stack()[1].function
        return self._num_remote_workers(origin=fct)


    @DeveloperAPI
    def num_healthy_remote_workers(self) -> int:
        """Returns the number of healthy remote workers."""
        fct = inspect.stack()[1].function
        return self._num_remote_workers(origin=fct)
    @DeveloperAPI
    def num_healthy_workers(self) -> int:
        """Returns the number of all healthy workers, including the local worker."""
        fct = inspect.stack()[1].function
        return self._num_remote_workers(origin=fct)

    @DeveloperAPI
    def num_in_flight_async_reqs(self) -> int:
        """Returns the number of in-flight async requests."""
        return self._worker_set.num_in_flight_async_reqs()

    @DeveloperAPI
    def num_remote_worker_restarts(self) -> int:
        """Total number of times managed remote workers have been restarted."""
        return self._worker_set.num_remote_worker_restarts()

    @DeveloperAPI
    def sync_weights(
        self,
        policies: Optional[List[PolicyID]] = None,
        from_worker_or_learner_group: Optional[
            Union[RolloutWorker, LearnerGroup]
        ] = None,
        to_worker_indices: Optional[List[int]] = None,
        global_vars: Optional[Dict[str, TensorType]] = None,
        timeout_seconds: Optional[int] = 0,
    ) -> None:
        self._worker_set.sync_weights(
            policies=policies,
            from_worker_or_learner_group=from_worker_or_learner_group,
            to_worker_indices=to_worker_indices,
            global_vars=global_vars,
            timeout_seconds=timeout_seconds
        )

    @DeveloperAPI
    def add_policy(
        self,
        policy_id: PolicyID,
        policy_cls: Optional[Type[Policy]] = None,
        policy: Optional[Policy] = None,
        *,
        observation_space: Optional[gym.spaces.Space] = None,
        action_space: Optional[gym.spaces.Space] = None,
        config: Optional[Union["AlgorithmConfig", PartialAlgorithmConfigDict]] = None,
        policy_state: Optional[PolicyState] = None,
        policy_mapping_fn: Optional[Callable[[AgentID, EpisodeID], PolicyID]] = None,
        policies_to_train: Optional[
            Union[
                Container[PolicyID],
                Callable[[PolicyID, Optional[SampleBatchType]], bool],
            ]
        ] = None,
        module_spec: Optional[SingleAgentRLModuleSpec] = None,
        # Deprecated.
        workers: Optional[List[Union[RolloutWorker, ActorHandle]]] = DEPRECATED_VALUE,
    ) -> None:
        self._worker_set.add_policy(
            policy_id,
            policy_cls=policy_cls,
            policy=policy,
            observation_space=observation_space,
            action_space=action_space,
            config=config,
            policy_state=policy_state,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=policies_to_train,
            module_spec=module_spec,
            workers=workers
        )

    @DeveloperAPI
    def add_workers(self, num_workers: int, validate: bool = False) -> None:
        self._worker_set.add_workers(
            num_workers,
            validate=validate
        )

    @DeveloperAPI
    def reset(self, new_remote_workers: List[ActorHandle]) -> None:
        self._worker_set.reset(new_remote_workers=new_remote_workers)

    @DeveloperAPI
    def stop(self) -> None:
        self._worker_set.stop()

    @DeveloperAPI
    def is_policy_to_train(
        self, policy_id: PolicyID, batch: Optional[SampleBatchType] = None
    ) -> bool:
        return self._worker_set.is_policy_to_train(policy_id, batch=batch)

    @DeveloperAPI
    def foreach_worker(
        self,
        func: Callable,
        *,
        local_worker=True,
        # TODO(jungong) : switch to True once Algorithm is migrated.
        healthy_only=False,
        remote_worker_ids: List[int] = None,
        timeout_seconds: Optional[int] = None,
        return_obj_refs: bool = False,
        mark_healthy: bool = False,
    ) -> List:
        global GLOBAL_WORKERS
        funcLambda = str(inspect.getsourcelines(func)[0][0])
        if "sample" in str(func) or "w.sample()" in str(funcLambda):
            true_local_worker = self._worker_set.local_worker()
            if self._number_of_processes > 1:
                for i in range(self._number_of_processes):
                    GLOBAL_WORKERS[self._id_worker_set][i].set_state(self._worker_set._local_worker.get_state())
                    GLOBAL_WORKERS[self._id_worker_set][i].set_weights(self._worker_set._local_worker.get_weights())
                    GLOBAL_WORKERS[self._id_worker_set][i].random_seed = np.random.randint(0, 10000000)
                tc.set_num_threads(1)
                with Pool(self._number_of_processes) as p:
                    lst_batches = list(p.map(func, [self._parallel_rollout_workers[i] for i in range(self._number_of_processes)]))
                tc.set_num_threads(self._number_of_processes)
                    #self._worker_set._local_worker.callbacks.on_sample_end(worker=self._worker_set._local_worker, samples=batch)

                    # Always do writes prior to compression for consistency and to allow
                    # for better compression inside the writer.
                    

                    # Always do writes prior to compression for consistency and to allow
                    # for better compression inside the writer.
                    #true_local_worker.output_writer.write(batch)

                    #if rollout_worker.config.compress_observations:
                    #    batch.compress(bulk=rollout_worker.config.compress_observations == "bulk")

                    #if true_local_worker.config.fake_sampler:
                    #    true_local_worker.last_batch = batch
                    
                    
            else:
                lst_batches = func(self._worker_set_parallel_local_worker)
            batch = concat_samples(lst_batches)
            #true_local_worker.output_writer.write(batch)
            true_local_worker.callbacks.on_sample_end(worker=true_local_worker, samples=batch)
            return [batch]
        else:
            """Returns the local rollout worker."""
            return self._worker_set.foreach_worker(
                func,
                local_worker=local_worker,
                healthy_only=healthy_only,
                remote_worker_ids=remote_worker_ids,
                timeout_seconds=timeout_seconds,
                return_obj_refs=return_obj_refs,
                mark_healthy=mark_healthy
            )

    @DeveloperAPI
    def foreach_worker_with_id(
        self,
        func: Callable,
        *,
        local_worker=True,
        # TODO(jungong) : switch to True once Algorithm is migrated.
        healthy_only=False,
        remote_worker_ids: List[int] = None,
        timeout_seconds: Optional[int] = None,
    ) -> List:
        return self._worker_set.foreach_worker_with_id(
            func,
            local_worker=local_worker,
            healthy_only=healthy_only,
            remote_worker_ids=remote_worker_ids,
            timeout_seconds=timeout_seconds
        )

    @DeveloperAPI
    def foreach_worker_async(
        self,
        func: Callable,
        *,
        # TODO(jungong) : switch to True once Algorithm is migrated.
        healthy_only=False,
        remote_worker_ids: List[int] = None,
    ) -> int:
        return self._worker_set.foreach_worker_async(
            func,
            healthy_only=healthy_only,
            remote_worker_ids=remote_worker_ids
        )

    @DeveloperAPI
    def fetch_ready_async_reqs(
        self,
        *,
        timeout_seconds: Optional[int] = 0,
        return_obj_refs: bool = False,
        mark_healthy: bool = False,
    ) -> List:
        return self._worker_set.fetch_ready_async_reqs(
            timeout_seconds = timeout_seconds,
            return_obj_refs = return_obj_refs,
            mark_healthy = mark_healthy,
        )

    @DeveloperAPI
    def foreach_policy(self, func: Callable):
        """Calls `func` with each worker's (policy, PolicyID) tuple.

        Note that in the multi-agent case, each worker may have more than one
        policy.

        Args:
            func: A function - taking a Policy and its ID - that is
                called on all workers' Policies.

        Returns:
            The list of return values of func over all workers' policies. The
                length of this list is:
                (num_workers + 1 (local-worker)) *
                [num policies in the multi-agent config dict].
                The local workers' results are first, followed by all remote
                workers' results
        """
        return self._worker_set.foreach_policy(func)

    @DeveloperAPI
    def foreach_policy_to_train(self, func: Callable):
        """Apply `func` to all workers' Policies iff in `policies_to_train`.

        Args:
            func: A function - taking a Policy and its ID - that is
                called on all workers' Policies, for which
                `worker.is_policy_to_train()` returns True.

        Returns:
            List[any]: The list of n return values of all
                `func([trainable policy], [ID])`-calls.
        """
        return self._worker_set.foreach_policy_to_train(
            func
        )

    @DeveloperAPI
    def foreach_env(self, func: Callable):
        """Calls `func` with all workers' sub-environments as args.

        An "underlying sub environment" is a single clone of an env within
        a vectorized environment.
        `func` takes a single underlying sub environment as arg, e.g. a
        gym.Env object.

        Args:
            func: A function - taking an EnvType (normally a gym.Env object)
                as arg and returning a list of lists of return values, one
                value per underlying sub-environment per each worker.

        Returns:
            The list (workers) of lists (sub environments) of results.
        """
        return self._worker_set.foreach_env(func)

    @DeveloperAPI
    def foreach_env_with_context(
        self, func: Callable
    ) -> List:
        return self._worker_set.foreach_env_with_context(func)

    @DeveloperAPI
    def probe_unhealthy_workers(self) -> List[int]:
        return self._worker_set.probe_unhealthy_workers()

    @staticmethod
    def _from_existing(
        local_worker: RolloutWorker, remote_workers: List[ActorHandle] = None
    ):
        workers = WorkerSet(
            env_creator=None, default_policy_class=None, config=None, _setup=False
        )
        workers.reset(remote_workers or [])
        workers._local_worker = local_worker
        workers = ParallelWorkerSetWrapper(workers)
        return workers

    def _make_worker(
        self,
        *,
        cls: Callable,
        env_creator: EnvCreator,
        validate_env: Optional[Callable[[EnvType], None]],
        worker_index: int,
        num_workers: int,
        recreated_worker: bool = False,
        config=None,
        spaces: Optional[
            Dict[PolicyID, Tuple[gym.spaces.Space, gym.spaces.Space]]
        ] = None,
    ) -> Union[RolloutWorker, ActorHandle]:
        return self._worker_set._make_worker(
            cls,
            env_creator,
            validate_env,
            worker_index,
            num_workers,
            recreated_worker=recreated_worker,
            config=config,
            spaces=spaces
        )

    @property
    @Deprecated(
        old="_remote_workers",
        new="Use either the `foreach_worker()`, `foreach_worker_with_id()`, or "
        "`foreach_worker_async()` APIs of `WorkerSet`, which all handle fault "
        "tolerance.",
        error=False,
    )
    def _remote_workers(self) -> List[ActorHandle]:
        return self._worker_set._remote_workers()

    @Deprecated(
        old="remote_workers()",
        new="Use either the `foreach_worker()`, `foreach_worker_with_id()`, or "
        "`foreach_worker_async()` APIs of `WorkerSet`, which all handle fault "
        "tolerance.",
        error=False,
    )
    def remote_workers(self) -> List[ActorHandle]:
        return self._worker_set.remote_workers() 