import copy
import numpy as np
import tree
import time

from collections import defaultdict
from gymnasium.spaces import Space
from Models.action_mask_catalog import ActionMaskCatalog
from Models.action_mask_rlm import TorchActionMaskRLM
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.connectors.util import (
    create_connectors_for_policy,
    maybe_get_filters_for_syncing,
)
from ray.rllib.core.models.base import STATE_IN, STATE_OUT
from ray.rllib.core.rl_module.rl_module import RLModuleConfig
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner
from ray.rllib.evaluation.metrics import RolloutMetrics
from ray.rllib.models import ModelCatalog
from ray.rllib.models.preprocessors import Preprocessor
from ray.rllib.policy.policy import Policy, PolicyID, PolicyState
from ray.rllib.policy.policy_map import PolicyMap
from ray.rllib.policy.sample_batch import SampleBatch, DEFAULT_POLICY_ID
from ray.rllib.utils.annotations import override
from ray.rllib.utils.filter import get_filter
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.policy import create_policy_for_framework
from ray.rllib.utils.tf_utils import get_tf_eager_cls_if_necessary
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.utils.typing import TensorStructType, TensorType, MultiAgentPolicyConfigDict
from typing import Dict, List, Optional, Tuple, Type


torch, nn = try_import_torch()

"""
Description - The majority of the customization that had to be done to get ray working is in this file.
                This is a combination between the singleAgentEnvRunner and the RemoteWorker.
                It handles both sampling as well as policy fetching. 
                There are a bunch of assumptions I made to make this simpler than the ray code
                like assuming I am not using an LSTM so all of the State code is removed.
                I assumed I am not using a multi-agent-environment.
                Majority is copied from Ray.
                Act randomly is never used in this project.
"""
class ActionMaskEnvRunner(EnvRunner):
    def __init__(
            self,
            config: AlgorithmConfig,
            default_policy_class: Optional[Type[Policy]] = None,
            spaces: Optional[Dict[PolicyID, Tuple[Space, Space]]] = None,
            **kwargs,
    ):
        """

        Args:
            config: The config to use to setup this EnvRunner.
        """
        super().__init__(config=config)

        # Get the worker index on which this instance is running.
        self.worker_index: int = kwargs.get("worker_index")

        self.policy_mapping_fn = (
            lambda agent_id, episode, worker, **kw: DEFAULT_POLICY_ID
        )

        # Create the vectorized gymnasium env.
        self.env = self.config.env(self.config.env_config["data_generator"], self.config.num_envs_per_worker)

        self.num_envs: int = self.env.num_envs
        assert self.num_envs == self.config.num_envs_per_worker

        # Create our own instance of the (single-agent) `RLModule` (which
        # the needs to be weight-synched) each iteration.
        try:
            rl_config = RLModuleConfig(
                observation_space=self.env.observation_space,
                action_space=self.env.action_space,
                model_config_dict=self.config.model,
                catalog_class=ActionMaskCatalog,
            )
            self.module = TorchActionMaskRLM(rl_config)

        except NotImplementedError:
            self.module = None

        self.spaces = spaces

        self.default_policy_class = default_policy_class
        self.policy_dict, self.is_policy_to_train = self.config.get_multi_agent_setup(
            env=self.env,
            spaces=self.spaces,
            default_policy_class=self.default_policy_class,
        )

        self.policy_map: Optional[PolicyMap] = None

        self.preprocessing_enabled: bool = not config._disable_preprocessor_api
        self.preprocessors: Dict[PolicyID, Preprocessor] = None

        self.seed = (
            None
            if self.config.seed is None
            else self.config.seed + self.worker_index + self.config.in_evaluation * 10000
        )

        self.callbacks: DefaultCallbacks = self.config.callbacks_class()

        self.global_vars: dict = {
            "timestep": 0,
            # Counter for performed gradient updates per policy in `self.policy_map`.
            # Allows for compiling metrics on the off-policy'ness of an update given
            # that the number of gradient updates of the sampling policies are known
            # to the learner (and can be compared to the learner version of the same
            # policy).
            "num_grad_updates_per_policy": defaultdict(int),
        }

        self.obs_formatted = False

        self._update_policy_map(policy_dict=self.policy_dict)

        # This should be the default.
        self._needs_initial_reset: bool = True
        self._episodes: List[Optional["SingleAgentEpisode"]] = [
            None for _ in range(self.num_envs)
        ]

        self._done_episodes_for_metrics: List["SingleAgentEpisode"] = []
        self._ongoing_episodes_for_metrics: Dict[List] = defaultdict(list)
        self._ts_since_last_metrics: int = 0
        self._weights_seq_no: int = 0

        self._states = [None for _ in range(self.num_envs)]

        self._needs_initial_reset = True

    @override(SingleAgentEnvRunner)
    def _sample_timesteps(
            self,
            num_timesteps: int,
            explore: bool = True,
            random_actions: bool = False,
            force_reset: bool = False,
    ) -> List["SingleAgentEpisode"]:
        """Helper method to sample n timesteps."""

        # TODO (sven): This gives a tricky circular import that goes
        # deep into the library. We have to see, where to dissolve it.
        from ray.rllib.env.single_agent_episode import SingleAgentEpisode

        done_episodes_to_return: List["SingleAgentEpisode"] = []

        # Have to reset the env (on all vector sub_envs).
        if force_reset or self._needs_initial_reset:
            obs, infos = self.env.vector_reset()

            # We just reset the env. Don't have to force this again in the next
            # call to `self._sample_timesteps()`.
            self._needs_initial_reset = False

            self._episodes = [SingleAgentEpisode() for _ in range(self.num_envs)]

            # Set initial obs and states in the episodes.
            for i in range(self.num_envs):
                self._episodes[i].add_env_reset(
                    observation=obs[i],
                    infos=infos[i],
                )
                self._states[i] = {}
        # Do not reset envs, but instead continue in already started episodes.
        else:
            # Pick up stored observations and states from previous timesteps.
            obs = np.stack([eps.observations[-1] for eps in self._episodes])
            obs = self.list_to_dict_two(obs)
            self.obs_formatted = True

        # Loop through env in enumerate.(self._episodes):
        ts = 0

        while ts < num_timesteps:
            # Act randomly.
            if random_actions:
                actions = self.env.action_space.sample()
                action_logp = np.zeros(shape=(actions.shape[0],))
                fwd_out = {}
            # Compute an action using the RLModule.
            else:
                # Note, RLModule `forward()` methods expect `NestedDict`s.
                batch = {
                    STATE_IN: tree.map_structure(lambda s: self._convert_from_numpy(s), {},),
                    SampleBatch.OBS: tree.map_structure(lambda s: self._convert_from_numpy(s), obs,)
                }

                if self.obs_formatted:
                    self.obs_formatted = False
                else:
                    batch = self.list_to_dict(batch)

                # Explore or not.
                if explore:
                    fwd_out = self.module.forward_exploration(batch)
                else:
                    fwd_out = self.module.forward_inference(batch)

                actions, action_logp = self._sample_actions_if_necessary(
                    fwd_out, explore
                )

                fwd_out = convert_to_numpy(fwd_out)

                if STATE_OUT in fwd_out:
                    states = fwd_out[STATE_OUT]

            obs, rewards, terminateds, truncateds, infos = self.env.vector_step(actions)

            ts += self.num_envs

            for i in range(self.num_envs):
                # The last entry in self.observations[i] is already the reset
                # obs of the new episode.
                # certain env parameter during different episodes (for example for
                # benchmarking).
                extra_model_output = {}

                for k, v in fwd_out.items():
                    if SampleBatch.ACTIONS != k:
                        # This line breaks if using one env per worker
                        extra_model_output[k] = v[i]
                extra_model_output[SampleBatch.ACTION_LOGP] = action_logp[i]

                # In inference we have only the action logits.
                if terminateds[i] or truncateds[i]:
                    # Finish the episode with the actual terminal observation stored in
                    # the info dict.
                    self._episodes[i].add_env_step(
                        obs[i],
                        actions[i],
                        rewards[i],
                        infos=infos[i],
                        terminated=terminateds[i],
                        truncated=truncateds[i],
                        extra_model_outputs=extra_model_output,
                    )

                    done_episodes_to_return.append(self._episodes[i].finalize())

                    new_obs, new_info = self.env.reset_at(i)
                    obs[i] = new_obs
                    infos[i] = new_info

                    # Create a new episode object with already the reset data in it.
                    self._episodes[i] = SingleAgentEpisode(
                        observations=[obs[i]], infos=[infos[i]]
                    )
                else:
                    self._episodes[i].add_env_step(
                        obs[i],
                        actions[i],
                        rewards[i],
                        infos=infos[i],
                        extra_model_outputs=extra_model_output,
                    )

        # Return done episodes ...
        self._done_episodes_for_metrics.extend(done_episodes_to_return)
        # Also, make sure, we return a copy and start new chunks so that callers
        # of this function do not alter the ongoing and returned Episode objects.
        new_episodes = [eps.cut() for eps in self._episodes]

        # ... and all ongoing episode chunks.
        # Initialized episodes do not have recorded any step and lack
        # `extra_model_outputs`.
        ongoing_episodes_to_return = [
            episode.finalize() for episode in self._episodes if episode.t > 0
        ]
        for eps in ongoing_episodes_to_return:
            self._ongoing_episodes_for_metrics[eps.id_].append(eps)

        # Record last metrics collection.
        self._ts_since_last_metrics += ts

        self._episodes = new_episodes

        return done_episodes_to_return + ongoing_episodes_to_return

    @override(SingleAgentEnvRunner)
    def _sample_episodes(
            self,
            num_episodes: int,
            explore: bool = True,
            random_actions: bool = False,
            with_render_data: bool = False,
    ) -> List["SingleAgentEpisode"]:
        """Helper method to run n episodes.

        See docstring of `self.sample()` for more details.
        """

        # TODO (sven): This gives a tricky circular import that goes
        # deep into the library. We have to see, where to dissolve it.
        from ray.rllib.env.single_agent_episode import SingleAgentEpisode

        # If user calls sample(num_timesteps=..) after this, we must reset again
        # at the beginning.
        self._needs_initial_reset = True

        done_episodes_to_return: List["SingleAgentEpisode"] = []

        obs, infos = self.env.vector_reset()
        episodes = [SingleAgentEpisode() for _ in range(self.num_envs)]

        # Get initial states for all 'batch_size_B` rows in the forward batch,
        # i.e. for all vector sub_envs.
        if hasattr(self.module, "get_initial_state"):
            states = tree.map_structure(
                lambda s: np.repeat(s, self.num_envs, axis=0),
                self.module.get_initial_state(),
            )
        else:
            states = {}

        render_images = [None] * self.num_envs
        if with_render_data:
            render_images = [e.render() for e in self.env.envs]

        for i in range(self.num_envs):
            episodes[i].add_env_reset(
                observation=obs[i],
                infos=infos[i],
                render_image=render_images[i],
            )

        eps = 0
        while eps < num_episodes:
            if random_actions:
                actions = self.env.action_space.sample()
                action_logp = np.zeros(shape=(actions.shape[0]))
                fwd_out = {}
            else:
                batch = {
                    STATE_IN: tree.map_structure(lambda s: self._convert_from_numpy(s), states),
                    SampleBatch.OBS: tree.map_structure(lambda s: self._convert_from_numpy(s), obs,)
                }

                batch = self.list_to_dict(batch)

                # Explore or not.
                if explore:
                    fwd_out = self.module.forward_exploration(batch)
                else:
                    fwd_out = self.module.forward_inference(batch)

                actions, action_logp = self._sample_actions_if_necessary(fwd_out, explore)

                fwd_out = convert_to_numpy(fwd_out)

                if STATE_OUT in fwd_out:
                    states = convert_to_numpy(fwd_out[STATE_OUT])

            obs, rewards, terminateds, truncateds, infos = self.env.vector_step(actions)
            if with_render_data:
                render_images = [e.render() for e in self.env.envs]

            for i in range(self.num_envs):
                # Extract info and state for vector sub_env.
                # info = {k: v[i] for k, v in infos.items()}
                # The last entry in self.observations[i] is already the reset
                # obs of the new episode.
                extra_model_output = {}
                for k, v in fwd_out.items():
                    if SampleBatch.ACTIONS not in k:
                        extra_model_output[k] = v[i]

                extra_model_output[SampleBatch.ACTION_LOGP] = action_logp[i]

                if terminateds[i] or truncateds[i]:
                    eps += 1

                    episodes[i].add_env_step(
                        obs[i],
                        actions[i],
                        rewards[i],
                        infos=infos[i],
                        terminated=terminateds[i],
                        truncated=truncateds[i],
                        extra_model_outputs=extra_model_output,
                    )

                    done_episodes_to_return.append(episodes[i])

                    # Also early-out if we reach the number of episodes within this
                    # for-loop.
                    if eps == num_episodes:
                        break

                    new_obs, new_info = self.env.reset_at(i)
                    obs[i] = new_obs
                    infos[i] = new_info

                    # Create a new episode object.
                    episodes[i] = SingleAgentEpisode(
                        observations=[obs[i]],
                        infos=[infos[i]],
                        render_images=None
                        if render_images[i] is None
                        else [render_images[i]],
                    )
                else:
                    episodes[i].add_env_step(
                        obs[i],
                        actions[i],
                        rewards[i],
                        infos=infos[i],
                        render_image=render_images[i],
                        extra_model_outputs=extra_model_output,
                    )

        self._done_episodes_for_metrics.extend(done_episodes_to_return)
        self._ts_since_last_metrics += sum(len(eps) for eps in done_episodes_to_return)

        # Initialized episodes have to be removed as they lack `extra_model_outputs`.
        return [episode for episode in done_episodes_to_return if episode.t > 0]

    @override(EnvRunner)
    def sample(
            self,
            *,
            num_timesteps: int = None,
            num_episodes: int = None,
            explore: bool = True,
            random_actions: bool = False,
            with_render_data: bool = False,
    ) -> List["SingleAgentEpisode"]:
        """Runs and returns a sample (n timesteps or m episodes) on the env(s)."""
        assert not (num_timesteps is not None and num_episodes is not None)

        # If not execution details are provided, use the config.
        if num_timesteps is None and num_episodes is None:
            if self.config.batch_mode == "truncate_episodes":
                num_timesteps = (
                        self.config.get_rollout_fragment_length(
                            worker_index=self.worker_index
                        )
                        * self.num_envs
                )
            else:
                num_episodes = self.num_envs

        # Sample n timesteps.
        if num_timesteps is not None:
            return self._sample_timesteps(
                num_timesteps=num_timesteps,
                explore=explore,
                random_actions=random_actions,
                force_reset=False,
            )
        # Sample m episodes.
        else:
            return self._sample_episodes(
                num_episodes=num_episodes,
                explore=explore,
                random_actions=random_actions,
                with_render_data=with_render_data,
            )

    def get_metrics(self) -> List[RolloutMetrics]:
        # Compute per-episode metrics (only on already completed episodes).
        metrics = []
        for eps in self._done_episodes_for_metrics:
            assert eps.is_done
            episode_length = len(eps)
            episode_reward = eps.get_return()
            # Don't forget about the already returned chunks of this episode.
            if eps.id_ in self._ongoing_episodes_for_metrics:
                for eps2 in self._ongoing_episodes_for_metrics[eps.id_]:
                    episode_length += len(eps2)
                    episode_reward += eps2.get_return()
                del self._ongoing_episodes_for_metrics[eps.id_]

            metrics.append(
                RolloutMetrics(
                    episode_length=episode_length,
                    episode_reward=episode_reward,
                )
            )

        self._done_episodes_for_metrics.clear()
        self._ts_since_last_metrics = 0

        return metrics

    def set_weights(self, weights, global_vars=None, weights_seq_no: int = 0):
        """Writes the weights of our (single-agent) RLModule."""
        if isinstance(weights, dict) and DEFAULT_POLICY_ID in weights:
            weights = weights[DEFAULT_POLICY_ID]
        weights = self._convert_to_tensor(weights)
        self.module.set_state(weights)

    def get_weights(self, modules=None):
        """Returns the weights of our (single-agent) RLModule."""

        return self.module.get_state()

    @override(EnvRunner)
    def assert_healthy(self):
        # Make sure, we have built our gym.vector.Env and RLModule properly.
        assert self.env and self.module

    @override(EnvRunner)
    def stop(self):
        # Close our env object via gymnasium's API.
        self.env.close()

    def _sample_actions_if_necessary(
            self, fwd_out: TensorStructType, explore: bool = True
    ) -> Tuple[np.array, np.array]:
        """Samples actions from action distribution if necessary."""

        if SampleBatch.ACTIONS in fwd_out.keys():
            actions = convert_to_numpy(fwd_out[SampleBatch.ACTIONS])
            action_logp = convert_to_numpy(fwd_out[SampleBatch.ACTION_LOGP])
        # If no actions are provided we need to sample them.
        else:
            # Explore or not.
            if explore:
                action_dist_cls = self.module.get_exploration_action_dist_cls()
            else:
                action_dist_cls = self.module.get_inference_action_dist_cls()
            # Generate action distribution and sample actions.
            action_dist = action_dist_cls.from_logits(
                fwd_out[SampleBatch.ACTION_DIST_INPUTS]
            )
            actions = action_dist.sample()
            # We need numpy actions for gym environments.
            action_logp = convert_to_numpy(action_dist.logp(actions))
            actions = convert_to_numpy(actions)

        return actions, action_logp

    def _convert_from_numpy(self, array: np.array) -> TensorType:
        return torch.from_numpy(array).to('cuda').to(torch.float32)

    def _convert_to_tensor(self, struct) -> TensorType:
        return convert_to_torch_tensor(struct, 'cuda')

    def _update_policy_map(
            self,
            *,
            policy_dict: MultiAgentPolicyConfigDict,
            policy: Optional[Policy] = None,
            policy_states: Optional[Dict[PolicyID, PolicyState]] = None,
    ) -> None:
        """Updates the policy map (and other stuff) on this worker.

        It performs the following:
            1. It updates the observation preprocessors and updates the policy_specs
                with the postprocessed observation_spaces.
            2. It updates the policy_specs with the complete algorithm_config (merged
                with the policy_spec's config).
            3. If needed it will update the self.marl_module_spec on this worker
            3. It updates the policy map with the new policies
            4. It updates the filter dict
            5. It calls the on_create_policy() hook of the callbacks on the newly added
                policies.

        Args:
            policy_dict: The policy dict to update the policy map with.
            policy: The policy to update the policy map with.
            policy_states: The policy states to update the policy map with.
        """

        # Update the input policy dict with the postprocessed observation spaces and
        # merge configs. Also updates the preprocessor dict.
        updated_policy_dict = self._get_complete_policy_specs_dict(policy_dict)

        # Deleted code here to related to API Stack

        # Builds the self.policy_map dict
        self._build_policy_map(
            policy_dict=updated_policy_dict,
            policy=policy,
            policy_states=policy_states,
        )

        # Initialize the filter dict
        self._update_filter_dict(updated_policy_dict)

        # Call callback policy init hooks (only if the added policy did not exist
        # before).
        if policy is None:
            self._call_callbacks_on_create_policy()

    def _get_complete_policy_specs_dict(
            self, policy_dict: MultiAgentPolicyConfigDict
    ) -> MultiAgentPolicyConfigDict:
        """Processes the policy dict and creates a new copy with the processed attrs.

        This processes the observation_space and prepares them for passing to rl module
        construction. It also merges the policy configs with the algorithm config.
        During this processing, we will also construct the preprocessors dict.
        """
        from ray.rllib.algorithms.algorithm_config import AlgorithmConfig

        updated_policy_dict = copy.deepcopy(policy_dict)
        # If our preprocessors dict does not exist yet, create it here.
        self.preprocessors = self.preprocessors or {}
        # Loop through given policy-dict and add each entry to our map.
        for name, policy_spec in sorted(updated_policy_dict.items()):
            # Policy brings its own complete AlgorithmConfig -> Use it for this policy.
            if isinstance(policy_spec.config, AlgorithmConfig):
                merged_conf = policy_spec.config
            else:
                # Update the general config with the specific config
                # for this particular policy.
                merged_conf: "AlgorithmConfig" = self.config.copy(copy_frozen=False)
                merged_conf.update_from_dict(policy_spec.config or {})

            # Update num_workers and worker_index.
            merged_conf.worker_index = self.worker_index

            # Preprocessors.
            obs_space = policy_spec.observation_space
            # Initialize preprocessor for this policy to None.
            self.preprocessors[name] = None
            if self.preprocessing_enabled:
                # Policies should deal with preprocessed (automatically flattened)
                # observations if preprocessing is enabled.
                preprocessor = ModelCatalog.get_preprocessor_for_space(
                    obs_space,
                    merged_conf.model,
                    include_multi_binary=self.config.get(
                        "_enable_new_api_stack", False
                    ),
                )
                # Original observation space should be accessible at
                # obs_space.original_space after this step.
                if preprocessor is not None:
                    obs_space = preprocessor.observation_space

                if not merged_conf.enable_connectors:
                    # If connectors are not enabled, rollout worker will handle
                    # the running of these preprocessors.
                    self.preprocessors[name] = preprocessor

            policy_spec.config = merged_conf
            policy_spec.observation_space = obs_space

        return updated_policy_dict

    def _build_policy_map(
            self,
            *,
            policy_dict: MultiAgentPolicyConfigDict,
            policy: Optional[Policy] = None,
            policy_states: Optional[Dict[PolicyID, PolicyState]] = None,
    ) -> None:
        """Adds the given policy_dict to `self.policy_map`.

        Args:
            policy_dict: The MultiAgentPolicyConfigDict to be added to this
                worker's PolicyMap.
            policy: If the policy to add already exists, user can provide it here.
            policy_states: Optional dict from PolicyIDs to PolicyStates to
                restore the states of the policies being built.
        """

        # If our policy_map does not exist yet, create it here.
        self.policy_map = self.policy_map or PolicyMap(
            capacity=self.config.policy_map_capacity,
            policy_states_are_swappable=self.config.policy_states_are_swappable,
        )

        # Loop through given policy-dict and add each entry to our map.
        for name, policy_spec in sorted(policy_dict.items()):
            # Create the actual policy object.
            if policy is None:
                new_policy = create_policy_for_framework(
                    policy_id=name,
                    policy_class=get_tf_eager_cls_if_necessary(
                        policy_spec.policy_class, policy_spec.config
                    ),
                    merged_config=policy_spec.config,
                    observation_space=policy_spec.observation_space,
                    action_space=policy_spec.action_space,
                    worker_index=self.worker_index,
                    seed=self.seed,
                )
            else:
                new_policy = policy

            # Maybe torch compile an RLModule.
            if self.config.get("_enable_new_api_stack", False) and self.config.get(
                    "torch_compile_worker"
            ):
                if self.config.framework_str != "torch":
                    raise ValueError("Attempting to compile a non-torch RLModule.")
                rl_module = getattr(new_policy, "model", None)
                if rl_module is not None:
                    compile_config = self.config.get_torch_compile_worker_config()
                    rl_module.compile(compile_config)

            self.policy_map[name] = new_policy

            restore_states = (policy_states or {}).get(name, None)
            # Set the state of the newly created policy before syncing filters, etc.
            if restore_states:
                new_policy.set_state(restore_states)

    def _update_filter_dict(self, policy_dict: MultiAgentPolicyConfigDict) -> None:
        """Updates the filter dict for the given policy_dict."""

        for name, policy_spec in sorted(policy_dict.items()):
            new_policy = self.policy_map[name]
            if policy_spec.config.enable_connectors:
                # Note(jungong) : We should only create new connectors for the
                # policy iff we are creating a new policy from scratch. i.e,
                # we should NOT create new connectors when we already have the
                # policy object created before this function call or have the
                # restoring states from the caller.
                # Also note that we cannot just check the existence of connectors
                # to decide whether we should create connectors because we may be
                # restoring a policy that has 0 connectors configured.
                if (
                        new_policy.agent_connectors is None
                        or new_policy.action_connectors is None
                ):
                    create_connectors_for_policy(new_policy, policy_spec.config)
                maybe_get_filters_for_syncing(self, name)
            else:
                filter_shape = tree.map_structure(
                    lambda s: (
                        None
                        if isinstance(s, (Discrete, MultiDiscrete))  # noqa
                        else np.array(s.shape)
                    ),
                    new_policy.observation_space_struct,
                )

                self.filters[name] = get_filter(
                    policy_spec.config.observation_filter,
                    filter_shape,
                )

    def _call_callbacks_on_create_policy(self):
        """Calls the on_create_policy callback for each policy in the policy map."""
        for name, policy in self.policy_map.items():
            self.callbacks.on_create_policy(policy_id=name, policy=policy)

    # Switching environment over to vectorized made this method not necessary
    def fetch_obs_index(self, observation, idx):
        # Shape for opponent [number of players, {"board", "scalars", "traits"}, num_envs, len_of_section]
        return {
            "observations": {
                "player": {
                    "bench": observation["observations"]["player"]["bench"][idx],
                    "board": observation["observations"]["player"]["board"][idx],
                    "items": observation["observations"]["player"]["items"][idx],
                    "scalars": observation["observations"]["player"]["scalars"][idx],
                    "shop": observation["observations"]["player"]["shop"][idx],
                    "traits": observation["observations"]["player"]["traits"][idx],
                },
                "opponents": observation["observations"]["opponents"][idx]
            },
            "action_mask": observation["action_mask"][idx]
        }

    def list_to_dict(self, observation):
        # Extract common sub-dictionaries outside the loop
        player_data = observation['obs'][0]["observations"]['player']
        action_mask = [obs_item['action_mask'] for obs_item in observation['obs']]

        return {
            'state_in': {},
            'obs': {
                'observations': {
                    'player': {key: torch.swapaxes(torch.stack([obs_item['observations']['player'][key]
                                                                for obs_item in observation['obs']], dim=1), 0, 1)
                               for key in player_data},
                    'opponents': torch.swapaxes(torch.stack(
                        [obs_item['observations']['opponents'] for obs_item in observation['obs']], dim=1), 0, 1)
                },
                'action_mask': torch.swapaxes(torch.stack(action_mask, dim=1), 0, 1)
            }
        }

    def list_to_dict_two(self, observation):
        # Extract common sub-dictionaries outside the loop
        player_data = observation[0]["observations"]['player']
        action_mask = [obs_item['action_mask'] for obs_item in observation]

        return {
            'observations': {
                'player': {key: np.asarray([obs_item['observations']['player'][key] for obs_item in observation])
                           for key in player_data},
                'opponents': np.asarray([obs_item['observations']['opponents'] for obs_item in observation])
            },
            'action_mask': np.asarray(action_mask)
        }

    def get_policy(self, policy_id: PolicyID = DEFAULT_POLICY_ID) -> Optional[Policy]:
        """Return policy for the specified id, or None.

        Args:
            policy_id: ID of the policy to return. None for DEFAULT_POLICY_ID
                (in the single agent case).

        Returns:
            The policy under the given ID (or None if not found).
        """
        return self.policy_map.get(policy_id)

    def get_global_vars(self) -> dict:
        """Returns the current `self.global_vars` dict of this RolloutWorker.

        Returns:
            The current `self.global_vars` dict of this RolloutWorker.
        """
        return self.global_vars

    def set_global_vars(
        self,
        global_vars: dict,
        policy_ids: Optional[List[PolicyID]] = None,
    ) -> None:
        """Updates this worker's and all its policies' global vars.

        Updates are done using the dict's update method.

        Args:
            global_vars: The global_vars dict to update the `self.global_vars` dict
                from.
            policy_ids: Optional list of Policy IDs to update. If None, will update all
                policies on the to-be-updated workers.

        .. testcode::
            :skipif: True

            worker = ...
            global_vars = worker.set_global_vars(
            ...     {"timestep": 4242})
        """
        # Handle per-policy values.
        if global_vars:
            global_vars_copy = global_vars.copy()
            gradient_updates_per_policy = global_vars_copy.pop(
                "num_grad_updates_per_policy", {}
            )
            self.global_vars["num_grad_updates_per_policy"].update(
                gradient_updates_per_policy
            )
            # Only update explicitly provided policies or those that that are being
            # trained, in order to avoid superfluous access of policies, which might have
            # been offloaded to the object store.
            # Important b/c global vars are constantly being updated.
            for pid in policy_ids if policy_ids is not None else self.policy_map.keys():
                if self.is_policy_to_train is None or self.is_policy_to_train(pid, None):
                    self.policy_map[pid].on_global_var_update(
                        dict(
                            global_vars_copy,
                            # If count is None, Policy won't update the counter.
                            **{"num_grad_updates": gradient_updates_per_policy.get(pid)},
                        )
                    )

            # Update all other global vars.
            self.global_vars.update(global_vars_copy)
        else:
            self.global_vars["timestep"] += self.num_envs
            self.global_vars["num_grad_updates_per_policy"] = {
                    pid: self.policy_map[pid].num_grad_updates
                    for pid in list(self.policy_map.keys())}
