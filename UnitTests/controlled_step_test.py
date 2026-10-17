"""Controlled steps use the mask and let Default_Agent play the other seats."""

from __future__ import annotations

import contextlib
import io

from Simulator.battle.champion import champion
from Simulator.battle.combat_context import CombatContext
from Simulator.encoding.token.action import ActionToken
from Simulator.game.player import Player
from Simulator.game.pool import pool
from Simulator.rng import EnvRNG
from Simulator.simulators.tft_simulator import TFT_Simulator, TFTConfig


def _env(seed=1):
    env = TFT_Simulator(TFTConfig())
    with contextlib.redirect_stdout(io.StringIO()):
        env.reset(seed=seed)
    return env


def _quiet(env, method, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return getattr(env, method)(*args, **kwargs)


def test_one_controlled_action_then_one_default_action_each():
    env = _env()
    result = _quiet(env, "step_controlled", [[0, 0, 0]], ("player_0",))
    assert result["accepted"] is True
    assert result["start_turn"] is False
    assert result["terminal"] is False
    assert env.actions_taken["player_0"] == 1
    assert all(env.actions_taken[agent] == 1 for agent in env.possible_agents)
    assert env.agent_selection == "player_0"
    assert env.game_round.current_round == 1


def test_several_controlled_ids_fill_the_other_seats():
    env = _env()
    result = _quiet(env, "step_controlled", [[0, 0, 0], [0, 0, 0]], ("player_0", "player_1"))
    assert result["accepted"] is True
    assert env.actions_taken["player_0"] == 1
    assert env.actions_taken["player_1"] == 1
    assert all(env.actions_taken[agent] == 1 for agent in env.possible_agents)
    assert env.agent_selection == "player_0"


def test_masked_action_does_not_change_the_game():
    env = _env()
    player = env.player_manager.player_states["player_0"]
    shop = list(player.shop)
    gold = player.gold
    taken = dict(env.actions_taken)
    result = _quiet(env, "step_controlled", [[4, 28, 0]], ("player_0",))
    assert result["accepted"] is False
    assert result["action"] == [4, 28, 0]
    assert [0, 0, 0] in result["mask"]
    assert [4, 28, 0] not in result["mask"]
    assert env.actions_taken == taken
    assert player.shop == shop
    assert player.gold == gold


def test_truncating_action_resolves_the_round():
    env = _env()
    for agent in env.agents:
        env.actions_taken[agent] = 14
    result = _quiet(env, "step_controlled", [[0, 0, 0]], ("player_0",))
    assert result["accepted"] is True
    assert result["start_turn"] is True
    assert env.game_round.current_round == 2


def test_llm_observe_text_and_mask_on_every_call():
    env = _env()
    first = env.llm_observe("player_0")
    assert first["text"].startswith("round ")
    assert "player player_0" in first["text"]
    assert "max_units " in first["text"]
    assert "\n    traits\n" in first["text"] or first["text"].rstrip().endswith("traits")
    assert [0, 0, 0] in first["mask"]
    assert env.llm_observe("player_0")["text"] == first["text"]
    _quiet(env, "step_controlled", [[0, 0, 0]], ("player_0",))
    second = env.llm_observe("player_0")
    headers = [line for line in second["text"].splitlines() if line and not line.startswith(" ")]
    assert headers[0] == "action succeeded"
    assert "gold" not in headers
    assert "board" not in headers
    assert "actions 1 of 15" not in second["text"]
    assert second["mask"]
    assert env.llm_observe("player_0")["text"] == second["text"]


def test_llm_observe_move_returns_only_the_board():
    env = _env()
    from Simulator.encoding.llm_observation import legal_commands
    env.llm_observe("player_0")
    move = next(command for command in legal_commands(env.player_manager.action_handlers["player_0"])
                if command[0] == 5 and command[1] < 28 and command[2] < 28 and command[1] != command[2])
    assert _quiet(env, "step_controlled", [move], ("player_0",))["accepted"] is True
    text = env.llm_observe("player_0")["text"]
    headers = [line for line in text.splitlines() if line and not line.startswith(" ")]
    assert headers == ["action succeeded", "board"]


def test_llm_observe_failure_does_not_repeat_the_position():
    env = _env()
    opening = env.llm_observe("player_0")["text"]
    refused = _quiet(env, "step_controlled", [[4, 28, 0]], ("player_0",))
    assert refused["accepted"] is False and refused["text"] == "action failed"
    assert env.llm_observe("player_0")["text"] == "action failed"
    assert env.llm_observe("player_0")["text"] == opening


def test_llm_observe_new_turn_returns_changed_sections_and_opponents():
    env = _env()
    env.llm_observe("player_0")
    for agent in env.agents:
        env.actions_taken[agent] = 14
    result = _quiet(env, "step_controlled", [[0, 0, 0]], ("player_0",))
    assert result["start_turn"] is True
    text = env.llm_observe("player_0")["text"]
    assert text.startswith("action succeeded\n")
    assert "\nround 2\n" in text
    assert "\nopponents\n" in text


def test_refresh_mask_uses_refresh_cost():
    with CombatContext(rng=EnvRNG.from_episode_seed(4)).bind():
        player = Player(pool(), 0)
        player.refresh_cost = 3
        player.gold = 2
        assert ActionToken(player).refresh_mask == 0
        player.gold = 3
        assert ActionToken(player).refresh_mask == 1


def test_full_bench_blocks_buys_unless_the_unit_upgrades():
    with CombatContext(rng=EnvRNG.from_episode_seed(4)).bind():
        player = Player(pool(), 0)
        player.gold = 10
        player.bench = [champion("elise") for _ in range(9)]
        player.shop = ["diana", None, None, None, None]
        player.shop_champions = player.create_shop_champions()
        assert list(ActionToken(player).buy_mask) == [0, 0, 0, 0, 0]

        player.shop = ["elise", None, None, None, None]
        player.shop_champions = player.create_shop_champions()
        player.triple_catalog = [{"name": "elise", "level": 1, "num": 2}]
        assert list(ActionToken(player).buy_mask) == [1, 0, 0, 0, 0]
