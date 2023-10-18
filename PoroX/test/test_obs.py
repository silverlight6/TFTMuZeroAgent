import pytest

from Simulator.tft_simulator import parallel_env

@pytest.fixture
def env():
    return parallel_env()

@pytest.fixture
def obs(env):
    return env.