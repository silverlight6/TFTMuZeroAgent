import pytest

from pettingzoo.test import parallel_api_test

from Simulator.porox.tft_simulator import parallel_env

@pytest.fixture
def env():
    return parallel_env()

def test_parallel_api(env):
    parallel_api_test(env)
