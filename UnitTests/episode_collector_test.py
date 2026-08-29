from Simulator.generators.episode_collector import collect_episode
from Simulator.simulators.tft_simulator import TFTConfig, parallel_env


def test_collect_episode_saves_placements(tmp_path):
    env = parallel_env(TFTConfig(num_players=8))
    agents = list(env.possible_agents)
    recorder = collect_episode(env, max_steps=8)
    env.close()

    path = recorder.save(tmp_path / "episode_00000.npz")
    assert path.exists()
    assert set(recorder.placements.keys()) == set(agents)
    assert all(place is not None for place in recorder.placements.values())
    assert any(recorder.transitions[agent] for agent in recorder.agents)
