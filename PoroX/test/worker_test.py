import pytest

from Simulator.porox.tft_simulator import parallel_env, TFTConfig
from PoroX.modules.worker import collect_gameplay_experience
from PoroX.architectures.mctx_agent import MCTSAgent

@pytest.fixture
def env():
    return parallel_env()

def test_worker(env):
    collect_gameplay_experience(env, MCTSAgent())