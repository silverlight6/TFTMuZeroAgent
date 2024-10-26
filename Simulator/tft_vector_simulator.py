import logging
import time
import numpy as np

import config
from Simulator.tft_position_simulator import TFT_Position_Simulator
from Simulator.tft_single_player_simulator import TFT_Single_Player_Simulator
from Simulator.tft_simulator import TFTConfig
from typing import List, Optional

logger = logging.getLogger(__name__)


if config.MUZERO_POSITION:
    ray_enabled = False
else:
    import ray
    ray_enabled = True

def optional_ray_remote(cls):
    """Applies `ray.remote` to a class if Ray is enabled."""
    if ray_enabled:
        return ray.remote(cls)
    return cls

@optional_ray_remote
class TFT_Vector_Pos_Simulator:
    """Internal wrapper to translate the position simulator into a VectorEnv object."""

    def __init__(self, data_generator=None, num_envs=1):
        """Initializes a TFT_Vector_Pos_Simulator object.

        Args:
            data_generator: Queue object that data from the simulated games get stored to replay battles from
            num_envs: Total number of sub environments in this VectorEnv.
        """
        self.envs = [TFT_Position_Simulator(data_generator, 0)]
        self.num_envs = num_envs
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.data_generator = data_generator

        # Fill up missing envs (so we have exactly num_envs sub-envs in this
        index = 1
        while len(self.envs) < num_envs:
            self.envs.append(TFT_Position_Simulator(data_generator, index))
            index += 1

        self.restart_failed_sub_environments = True
        self.rounds_improvement = 0
        self.current_level = 0
        self.average_rewards = []

    # A fair amount of this code came strait from Ray source files. Not changing unless broken.
    def vector_reset(
            self, *, seeds: Optional[List[int]] = None, options: Optional[List[dict]] = None
    ):
        seeds = seeds or [None] * self.num_envs
        options = options or [None] * self.num_envs
        # Use reset_at(index) to restart and retry until
        # we successfully create a new env.
        resetted_obs = []
        resetted_infos = []
        for i in range(len(self.envs)):
            while True:
                obs, infos = self.reset_at(i, seed=seeds[i], options=options[i])
                if not isinstance(obs, Exception):
                    break
            resetted_obs.append(obs)
            resetted_infos.append(infos)
        return resetted_obs, resetted_infos

    def reset_at(
            self,
            index: Optional[int] = None,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        if index is None:
            index = 0
        try:
            obs_and_infos = self.envs[index].reset(seed=seed, options=options)

        except Exception as e:
            if self.restart_failed_sub_environments:
                logger.exception(e.args[0])
                self.restart_at(index)
                obs_and_infos = e, {}
            else:
                raise e

        return obs_and_infos

    def restart_at(self, index: Optional[int] = None) -> None:
        logger.warning("Finding myself in an reset or step error.")
        if index is None:
            index = 0

        # Try closing down the old (possibly faulty) sub-env, but ignore errors.
        self.envs[index].close()

        # Re-create the sub-env at the new index.
        logger.warning(f"Trying to restart sub-environment at index {index}.")
        self.envs[index] = self.make_env(index)
        logger.warning(f"Sub-environment at index {index} restarted successfully.")

    def vector_step(self, actions: List):
        obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch = (
            [],
            [],
            [],
            [],
            [],
        )
        for i in range(self.num_envs):
            try:
                results = self.envs[i].step(actions[i])
            except Exception as e:
                logger.warning(f"exception found {e}")
                if self.restart_failed_sub_environments:
                    logger.exception(e.args[0])
                    self.restart_at(i)
                    results = e, 0.0, True, True, {}
                else:
                    raise e

            obs, reward, terminated, truncated, info = results

            if not isinstance(info, dict):
                raise ValueError(
                    "Info should be a dict, got {} ({})".format(info, type(info))
                )
            obs_batch.append(obs)
            reward_batch.append(reward)
            terminated_batch.append(terminated)
            truncated_batch.append(truncated)
            info_batch.append(info)

        return obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch

    def get_sub_environments(self):
        return self.envs

    def try_render_at(self, index: Optional[int] = None):
        if index is None:
            index = 0
        return self.envs[index].render()

    def make_env(self, index):
        return TFT_Position_Simulator(self.data_generator, index)

    def close(self):
        for env in self.envs:
            env.close()

    def vector_reset_step(self, ray_actions: List, ray_seeds: Optional[List[int]] = None,
                          ray_options: Optional[List[dict]] = None):
        obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch = (
            [],
            [],
            [],
            [],
            [],
        )
        for i in range(self.num_envs):
            try:
                results = self.envs[i].step(ray_actions[i])
            except Exception as e:
                logger.warning(f"exception found {e}")
                if self.restart_failed_sub_environments:
                    logger.exception(e.args[0])
                    self.restart_at(i)
                    results = e, 0.0, True, True, {}
                else:
                    raise e

            obs, reward, terminated, truncated, info = results

            if not isinstance(info, dict):
                raise ValueError(
                    "Info should be a dict, got {} ({})".format(info, type(info))
                )
            obs_batch.append(obs)
            reward_batch.append(reward)
            terminated_batch.append(terminated)
            truncated_batch.append(truncated)
            info_batch.append(info)
        seeds = ray_seeds or [None] * self.num_envs
        options = ray_options or [None] * self.num_envs
        # Use reset_at(index) to restart and retry until
        # we successfully create a new env.
        resetted_obs = []
        resetted_infos = []
        for i in range(len(self.envs)):
            while True:
                obs, infos = self.reset_at(i, seed=seeds[i], options=options[i])
                if not isinstance(obs, Exception):
                    break
            resetted_obs.append(obs)
            resetted_infos.append(infos)
        self.average_rewards.append(sum(reward_batch) / len(reward_batch))
        # try to keep the size down
        if len(self.average_rewards) > 100:
            self.average_rewards = self.average_rewards[-95:]
        if self.check_greater_than_last_five():
            self.average_rewards = []
            for env in self.envs:
                env.level_up()

        return resetted_obs, reward_batch, terminated_batch, truncated_batch, resetted_infos

    def check_greater_than_last_five(self):
        """
        Checks if the current value in a list is greater than or equal to the last 5 values.

        Returns:
          True if the current value is greater than or equal to the last 5 values, False otherwise.
        """
        if len(self.average_rewards) < 10:
            return False  # Not enough previous values to compare

        overall_average = sum(self.average_rewards) / len(self.average_rewards)

        for i in range(5):
            if self.average_rewards[i - 5] < overall_average:
                return False

        self.current_level += 1
        print(f"LEVELING UP WITH AVERAGE REWARD OF {[self.average_rewards[-5:]]} "
              f"with overall average {overall_average} to level {self.current_level}")
        return True

    @staticmethod
    def list_to_dict(observation):
        # Extract common sub-dictionaries outside the loop
        player_data = observation[0][0]["observations"]
        action_mask = [obs_item['action_mask'] for obs_step in observation for obs_item in obs_step]

        return {
            'observations': {key: np.array([obs_item['observations'][key] for obs_step in observation
                                            for obs_item in obs_step])
                             for key in player_data},
            'action_mask': np.array(action_mask)
        }

class TFT_Single_Player_Vector_Simulator:
    """Internal wrapper to translate the position simulator into a VectorEnv object."""

    def __init__(self, tft_config: TFTConfig, num_envs: int):
        """Initializes a TFT_Vector_Pos_Simulator object.

        Args:
            tft_config: config to play the game of TFT
            num_envs: Total number of sub environments in this VectorEnv.
        """
        self.envs = [TFT_Single_Player_Simulator(tft_config)]
        self.num_envs = num_envs
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.tft_config = tft_config

        # Fill up missing envs (so we have exactly num_envs sub-envs in this
        while len(self.envs) < num_envs:
            self.envs.append(TFT_Single_Player_Simulator(tft_config))

        self.restart_failed_sub_environments = True

    # A fair amount of this code came strait from Ray source files. Not changing unless broken.
    def vector_reset(
            self, *, seeds: Optional[List[int]] = None, options: Optional[List[dict]] = None
    ):
        seeds = seeds or [None] * self.num_envs
        options = options or [None] * self.num_envs
        # Use reset_at(index) to restart and retry until
        # we successfully create a new env.
        resetted_obs = []
        resetted_infos = []
        for i in range(len(self.envs)):
            while True:
                obs, infos = self.reset_at(i, seed=seeds[i], options=options[i])
                if not isinstance(obs, Exception):
                    break
            resetted_obs.append(obs)
            resetted_infos.append(infos)
        return resetted_obs, resetted_infos

    def reset_at(
            self,
            index: Optional[int] = None,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        if index is None:
            index = 0
        try:
            obs_and_infos = self.envs[index].reset(seed=seed, options=options)

        except Exception as e:
            if self.restart_failed_sub_environments:
                logger.exception(e.args[0])
                self.restart_at(index)
                obs_and_infos = e, {}
            else:
                raise e

        return obs_and_infos

    def restart_at(self, index: Optional[int] = None) -> None:
        logger.warning("Finding myself in an reset or step error.")
        if index is None:
            index = 0

        # Try closing down the old (possibly faulty) sub-env, but ignore errors.
        self.envs[index].close()

        # Re-create the sub-env at the new index.
        logger.warning(f"Trying to restart sub-environment at index {index}.")
        self.envs[index] = self.make_env()
        logger.warning(f"Sub-environment at index {index} restarted successfully.")

    def vector_step(self, actions: List, prev_terminated: List):
        obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch = (
            [],
            [],
            [],
            [],
            [],
        )
        alive_i = 0
        for i in range(self.num_envs):
            if not prev_terminated[i]:
                try:
                    results = self.envs[i].step(actions[alive_i])
                except Exception as e:
                    print(f"Exception {e} found")
                    time.sleep(2)
                    logger.warning(f"exception found {e}")
                    if self.restart_failed_sub_environments:
                        logger.exception(e.args[0])
                        self.restart_at(i)
                        results = e, 0.0, True, True, {"state_empty": True}
                    else:
                        raise e

                obs, reward, terminated, truncated, info = results

                if not isinstance(info, dict):
                    raise ValueError(
                        "Info should be a dict, got {} ({})".format(info, type(info))
                    )
                obs_batch.append(obs)
                reward_batch.append(reward)
                terminated_batch.append(terminated)
                truncated_batch.append(truncated)
                info_batch.append(info)
                alive_i += 1
            else:
                terminated_batch.append(True)

        return obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch

    def get_sub_environments(self):
        return self.envs

    def try_render_at(self, index: Optional[int] = None):
        if index is None:
            index = 0
        return self.envs[index].render()

    def make_env(self):
        return TFT_Single_Player_Simulator(self.tft_config)

    def close(self):
        for env in self.envs:
            env.close()

    def vector_reset_step(self, ray_actions: List, ray_seeds: Optional[List[int]] = None,
                          ray_options: Optional[List[dict]] = None):
        obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch = (
            [],
            [],
            [],
            [],
            [],
        )
        for i in range(self.num_envs):
            try:
                results = self.envs[i].step(ray_actions[i])
            except Exception as e:
                logger.warning(f"exception found {e}")
                if self.restart_failed_sub_environments:
                    logger.exception(e.args[0])
                    self.restart_at(i)
                    results = e, 0.0, True, True, {}
                else:
                    raise e

            obs, reward, terminated, truncated, info = results

            if not isinstance(info, dict):
                raise ValueError(
                    "Info should be a dict, got {} ({})".format(info, type(info))
                )
            obs_batch.append(obs)
            reward_batch.append(reward)
            terminated_batch.append(terminated)
            truncated_batch.append(truncated)
            info_batch.append(info)
        seeds = ray_seeds or [None] * self.num_envs
        options = ray_options or [None] * self.num_envs
        # Use reset_at(index) to restart and retry until
        # we successfully create a new env.
        resetted_obs = []
        resetted_infos = []
        for i in range(len(self.envs)):
            while True:
                obs, infos = self.reset_at(i, seed=seeds[i], options=options[i])
                if not isinstance(obs, Exception):
                    break
            resetted_obs.append(obs)
            resetted_infos.append(infos)

        return resetted_obs, reward_batch, terminated_batch, truncated_batch, resetted_infos
