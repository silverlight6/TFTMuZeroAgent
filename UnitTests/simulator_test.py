import pytest
from pettingzoo.test import parallel_api_test, api_test

from Simulator import config
from Simulator.simulators.tft_simulator import parallel_env, env as tft_env, TFTConfig
from Simulator.observation.token.basic_observation import ObservationToken
from Simulator.observation.vector.observation import ObservationVector

def aec(env):
    return tft_env(env)

def parallel(env):
    return parallel_env(env)


def test_Env():
    """
    PettingZoo's api tests for the simulator.
    """
    tftConfig = TFTConfig(observation_class=ObservationToken, num_players=config.NUM_PLAYERS)
    raw_env = aec(tftConfig)
    api_test(raw_env, num_cycles=50)
    local_env = parallel(TFTConfig(observation_class=ObservationVector, num_players=config.NUM_PLAYERS))
    parallel_api_test(local_env, num_cycles=50)
