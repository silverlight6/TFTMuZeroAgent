import pytest

from Simulator.tft_simulator import parallel_env


@pytest.fixture
def env():
    return parallel_env()

def test_env_completes():
    pass

def test_env(env):
    # A comprehensive test of the environment

    # We need to test the following functionality of the game:
    # 1. Board/Bench movement
    # 2. Item movement
    # 3. Shop buying
    # 4. Round progression
    # 5. Player health
    # 6. Player gold
    # 7. Player experience
    # 8. Player level
    pass
