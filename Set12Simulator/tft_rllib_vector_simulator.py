import logging
import time

from ray.rllib.env.vector_env import VectorEnv
from ray.rllib.env.base_env import convert_to_base_env
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import (
    EnvActionType,
    EnvInfoDict,
    EnvObsType,
    EnvType,
)
from ray.util import log_once

from Set12Simulator.tft_position_simulator import TFT_Position_Simulator
from typing import List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class TFT_RLLIB_Vector_Pos_Simulator(VectorEnv):
    """Internal wrapper to translate the position simulator into a VectorEnv object."""

    def __init__(self, data_generator, num_envs):
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

        super().__init__(
            observation_space=self.envs[0].observation_space,
            action_space=self.envs[0].action_space,
            num_envs=num_envs,
        )

        self.restart_failed_sub_environments = True

    # A fair amount of this code came strait from Ray source files. Not changing unless broken.
    @override(VectorEnv)
    def vector_reset(
        self, *, seeds: Optional[List[int]] = None, options: Optional[List[dict]] = None
    ) -> Tuple[List[EnvObsType], List[EnvInfoDict]]:
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

    @override(VectorEnv)
    def reset_at(
        self,
        index: Optional[int] = None,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Union[EnvObsType, Exception], Union[EnvInfoDict, Exception]]:
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

    @override(VectorEnv)
    def restart_at(self, index: Optional[int] = None) -> None:
        logger.warning("Finding myself in an reset or step error.")
        if index is None:
            index = 0

        # Try closing down the old (possibly faulty) sub-env, but ignore errors.
        try:
            self.envs[index].close()
        except Exception as e:
            if log_once("close_sub_env"):
                logger.warning(
                    "Trying to close old and replaced sub-environment (at vector "
                    f"index={index}), but closing resulted in error:\n{e}"
                )

        # Re-create the sub-env at the new index.
        logger.warning(f"Trying to restart sub-environment at index {index}.")
        self.envs[index] = self.make_env(index)
        logger.warning(f"Sub-environment at index {index} restarted successfully.")

    @override(VectorEnv)
    def vector_step(
        self, actions: List[EnvActionType]
    ) -> Tuple[
        List[EnvObsType], List[float], List[bool], List[bool], List[EnvInfoDict]
    ]:
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

    @override(VectorEnv)
    def get_sub_environments(self) -> List[EnvType]:
        return self.envs

    @override(VectorEnv)
    def try_render_at(self, index: Optional[int] = None):
        if index is None:
            index = 0
        return self.envs[index].render()

    def make_env(self, index):
        return TFT_Position_Simulator(self.data_generator, index)

    def close(self):
        for env in self.envs:
            env.close()
