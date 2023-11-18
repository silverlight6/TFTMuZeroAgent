import pytest
from pettingzoo.test import parallel_api_test, api_test
from Simulator.tft_simulator import parallel_env, env as tft_env

@pytest.fixture
def aec():
    return tft_env()

@pytest.fixture
def parallel():
    return parallel_env()


def test_Env():
    """
    PettingZoo's api tests for the simulator.
    """
    raw_env = aec()
    api_test(raw_env, num_cycles=100000)
    local_env = parallel()
    parallel_api_test(local_env, num_cycles=100000)
