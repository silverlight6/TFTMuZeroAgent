import json
from pathlib import Path

import numpy as np

from Simulator.game.player import Player
from Simulator.game.pool import pool
from Simulator.observation.token.action import ActionToken
from Simulator.simulators.tft_item_simulator import TFT_Item_Simulator
from Simulator.simulators.tft_position_simulator import TFT_Position_Simulator
from Simulator.simulators.tft_simulator import TFTConfig
from Simulator.simulators.tft_single_player_simulator import TFT_Single_Player_Simulator
from Simulator.simulators.ui import GameState, is_porosight_render


def test_is_porosight_render_only_accepts_porosight():
    assert is_porosight_render("porosight")
    assert not is_porosight_render(None)
    assert not is_porosight_render(False)
    assert not is_porosight_render("json")
    assert not is_porosight_render(True)


def test_game_state_writes_kind_and_traits(tmp_path):
    player = Player(pool(), 0)
    path = GameState.for_position(player, None, None, str(tmp_path)).write_json()
    data = json.loads(Path(path).read_text())
    assert data["kind"] == "position"
    assert "player_0" in data["players"]
    state = data["players"]["player_0"]["state"]
    for key in ("health", "gold", "board", "bench", "shop", "items", "traits"):
        assert key in state
    assert isinstance(state["traits"], list)


def test_decode_action_accepts_vector_and_index():
    player = Player(pool(), 0)
    gs = GameState.for_single_player({"player_0": player}, None, "/tmp", ActionToken)
    assert gs.decode_action([5, 3, 10]) == [5, 3, 10]
    assert gs.decode_action(np.asarray([3, 0, 0])) == [3, 0, 0]
    decoded = gs.decode_action(np.int64(52 * 38))
    assert decoded[0] == 0


def test_position_writes_only_when_porosight(tmp_path):
    off_dir = tmp_path / "off"
    on_dir = tmp_path / "on"
    off_dir.mkdir()
    on_dir.mkdir()

    env = TFT_Position_Simulator(render_mode=None, render_path=str(off_dir))
    observation, info = env.reset()
    action = np.zeros(env.max_action_count, dtype=np.int64)
    legal = np.argwhere(observation["action_mask"] > 0)
    if len(legal):
        action[0] = legal[0][1]
    env.step(action)
    env.close()
    assert list(off_dir.glob("*.json")) == []

    env = TFT_Position_Simulator(render_mode="porosight", render_path=str(on_dir))
    observation, info = env.reset()
    action = np.zeros(env.max_action_count, dtype=np.int64)
    legal = np.argwhere(observation["action_mask"] > 0)
    if len(legal):
        action[0] = legal[0][1]
    env.step(action)
    env.close()
    files = list(on_dir.glob("*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text())
    assert data["kind"] == "position"
    assert data["players"]


def test_item_writes_only_when_porosight(tmp_path):
    off_dir = tmp_path / "off"
    on_dir = tmp_path / "on"
    off_dir.mkdir()
    on_dir.mkdir()

    env = TFT_Item_Simulator(render_mode=None, render_path=str(off_dir))
    observation, info = env.reset()
    action = np.zeros(10, dtype=np.int64)
    env.step(action)
    env.close()
    assert list(off_dir.glob("*.json")) == []

    env = TFT_Item_Simulator(render_mode="porosight", render_path=str(on_dir))
    observation, info = env.reset()
    action = np.zeros(10, dtype=np.int64)
    env.step(action)
    env.close()
    files = list(on_dir.glob("*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text())
    assert data["kind"] == "item"
    hero = next(iter(data["players"].values()))
    assert len(hero["battles"]) >= 2


def test_single_player_reset_records_only_when_porosight(tmp_path):
    off_dir = tmp_path / "off"
    on_dir = tmp_path / "on"
    off_dir.mkdir()
    on_dir.mkdir()

    env = TFT_Single_Player_Simulator(TFTConfig(num_players=1, render_mode=None, render_path=str(off_dir)))
    env.reset()
    assert not hasattr(env, "game_state")
    env.close()
    assert list(off_dir.glob("*.json")) == []

    env = TFT_Single_Player_Simulator(
        TFTConfig(num_players=1, render_mode="porosight", render_path=str(on_dir))
    )
    env.reset()
    path = env.game_state.write_json()
    env.close()
    data = json.loads(Path(path).read_text())
    assert data["kind"] == "single_player"
    assert "player_0" in data["players"]
