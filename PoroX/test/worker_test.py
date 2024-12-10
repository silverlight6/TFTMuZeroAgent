from PoroX.modules.worker import collect_gameplay_experience
from PoroX.models.mctx_agent import MCTSAgent

def test_worker(env):
    collect_gameplay_experience(env, MCTSAgent())
