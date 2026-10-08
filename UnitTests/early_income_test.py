"""Passive gold and XP arrive from 1-2 on, as in the real game.

Real game: passive income is 2 / 2 / 3 / 4 gold at the start of 1-2 / 1-3 / 1-4 / 2-1
and 5 gold (plus interest and streak gold) from 2-2 on, and every round grants 2 XP.
In the simulator, round index 0 is the 1-1 carousel plus the 1-2 fight with no planning
step in between, and the env never calls start_round() before the 1-3 planning phase,
so players reached 1-3 with no passive gold or XP and the table [0, 2, 2, 3, 4] was
then paid at 1-4 / 2-1 / 2-2 instead: income ran two rounds late.
"""

from __future__ import annotations

import numpy as np

from Simulator.battle import minion
from Simulator.battle.combat_context import CombatContext
from Simulator.game import single_player_game_round
from Simulator.game.game_round import Game_Round
from Simulator.game.player import Player
from Simulator.game.pool import pool
from Simulator.rng import EnvRNG
from Simulator.simulators.tft_simulator import TFTConfig, env as aec_env, parallel_env
from Simulator.simulators.tft_single_player_simulator import TFT_Single_Player_Simulator
from Simulator.simulators.tft_vector_simulator import TFT_Single_Player_Vector_Simulator

PASS = np.array([0, 0, 0])
# (gold, level, exp) at the start of the 1-3, 1-4 and 2-1 planning phases with no spending:
# 1-2 and 1-3 pay 2 + 2 before 1-3, then 1-4 pays 3 and 2-1 pays 4; 2 XP each time.
FIRST_THREE_PLANNING_PHASES = [(4, 3, 0), (7, 3, 2), (11, 3, 4)]


def _skip_pve(monkeypatch):
    # PvE loot can contain gold; keep the test to passive income only.
    monkeypatch.setattr(minion, "minion_round", lambda *args, **kwargs: True)


def test_first_planning_phase_has_income_of_1_2_and_1_3(monkeypatch):
    _skip_pve(monkeypatch)
    with CombatContext(rng=EnvRNG.from_episode_seed(2)).bind():
        base_pool = pool()
        players = {f"player_{i}": Player(base_pool, i) for i in range(8)}
        game_round = Game_Round(players, base_pool, None)
        game_round.play_game_round()  # 1-1 carousel and the 1-2 fight; next comes the 1-3 planning phase
    assert game_round.current_round == 1
    for player in players.values():
        assert (player.gold, player.level, player.exp) == (4, 3, 0)


def test_passive_income_schedule_through_2_2(monkeypatch):
    _skip_pve(monkeypatch)
    with CombatContext(rng=EnvRNG.from_episode_seed(2)).bind():
        base_pool = pool()
        player = Player(base_pool, 0)
        game_round = Game_Round({"player_0": player}, base_pool, None)
        game_round.play_game_round()
        seen = [(player.gold, player.level, player.exp)]
        # The env calls start_round(r) at the start of each later planning phase r.
        for round_index in (2, 3, 4):
            player.start_round(round_index)
            seen.append((player.gold, player.level, player.exp))
    # 1-3: 2+2, 1-4: +3, 2-1: +4, 2-2: +5 and 1 interest on 11 gold; level 4 at 2-2.
    assert seen == [(4, 3, 0), (7, 3, 2), (11, 3, 4), (17, 4, 0)]


def test_single_player_first_planning_phase_has_same_income(monkeypatch):
    _skip_pve(monkeypatch)
    with CombatContext(rng=EnvRNG.from_episode_seed(2)).bind():
        player = Player(pool(), 0)
        game_round = single_player_game_round.Game_Round(player, player.pool_obj, None)
        game_round.play_game_round()
    assert (player.gold, player.level, player.exp) == (4, 3, 0)


def test_gold_income_table():
    with CombatContext(rng=EnvRNG.from_episode_seed(2)).bind():
        base_pool = pool()
        paid = []
        for round_index in range(6):
            player = Player(base_pool, 0)
            player.gold_income(round_index)
            paid.append(player.gold)
    # rounds 0..3 are the planning phases of 1-2, 1-3, 1-4 and 2-1
    assert paid == [2, 2, 3, 4, 5, 5]


# --- Every simulator that plays rounds: the parallel and AEC full-game envs (TFT_Simulator,
# Game_Round) and the single-player and vector single-player envs (single_player_game_round).
# The position and item envs play one fight from generated boards and pay no round income.

def _state(player):
    return player.gold, player.level, player.exp


def _quiet(monkeypatch):
    from Simulator import config
    monkeypatch.setattr(config, "PRINTMESSAGES", False)
    monkeypatch.setattr(config, "LOGMESSAGES", False)


def _full_game_states(env):
    players = env.unwrapped.player_manager.player_states
    return {seat: _state(player) for seat, player in players.items()}


def test_parallel_env_income(monkeypatch, tmp_path):
    _skip_pve(monkeypatch)
    _quiet(monkeypatch)
    monkeypatch.chdir(tmp_path)
    env = parallel_env(TFTConfig())
    env.reset(seed=4)
    seen = [_full_game_states(env)]
    while len(seen) < 4:
        current = env.unwrapped.game_round.current_round
        while env.unwrapped.game_round.current_round == current:
            env.step({agent: PASS for agent in env.agents})
        seen.append(_full_game_states(env))
    seats = [f"player_{i}" for i in range(8)]
    for phase, expected in enumerate(FIRST_THREE_PLANNING_PHASES):
        assert seen[phase] == {seat: expected for seat in seats}
    # 2-2: 5 + 1 interest on 11 gold, plus 1 gold for winning the 2-1 fight; level 4.
    players = env.unwrapped.player_manager.player_states
    for seat in seats:
        won = players[seat].win_streak
        assert seen[3][seat] == (17 + won, 4, 0)


def test_aec_env_income(monkeypatch, tmp_path):
    _skip_pve(monkeypatch)
    _quiet(monkeypatch)
    monkeypatch.chdir(tmp_path)
    env = aec_env(TFTConfig())
    env.reset(seed=4)
    seen = [_full_game_states(env)]
    current = env.unwrapped.game_round.current_round
    for agent in env.agent_iter():
        _, _, termination, truncation, _ = env.last()
        env.step(None if termination or truncation else PASS)
        if env.unwrapped.game_round.current_round != current:
            current = env.unwrapped.game_round.current_round
            seen.append(_full_game_states(env))
            if len(seen) == 3:
                break
    seats = [f"player_{i}" for i in range(8)]
    for phase, expected in enumerate(FIRST_THREE_PLANNING_PHASES):
        assert seen[phase] == {seat: expected for seat in seats}


def _win_every_single_player_fight(monkeypatch):
    monkeypatch.setattr(single_player_game_round.Game_Round, "single_combat_phase",
                        lambda self, player, enemy: True)


def test_single_player_env_income(monkeypatch, tmp_path):
    _skip_pve(monkeypatch)
    _win_every_single_player_fight(monkeypatch)
    _quiet(monkeypatch)
    monkeypatch.chdir(tmp_path)
    env = TFT_Single_Player_Simulator(TFTConfig(num_players=1))
    env.reset(seed=4)
    seen = [_state(env.PLAYER)]
    for _ in range(3):
        for _ in range(env.max_actions_per_round):
            _, _, terminated, _, _ = env.step(PASS)
            assert not terminated
        seen.append(_state(env.PLAYER))
    # 2-2: 5 + 1 interest on 11 gold (single-player fights pay no win gold); level 4.
    assert seen == FIRST_THREE_PLANNING_PHASES + [(17, 4, 0)]


def test_vector_single_player_env_income(monkeypatch, tmp_path):
    _skip_pve(monkeypatch)
    _win_every_single_player_fight(monkeypatch)
    _quiet(monkeypatch)
    monkeypatch.chdir(tmp_path)
    vec = TFT_Single_Player_Vector_Simulator(TFTConfig(num_players=1), num_envs=2)
    vec.restart_failed_sub_environments = False
    vec.vector_reset(seeds=[4, 5])
    seen = [[_state(sub.PLAYER) for sub in vec.envs]]
    terminated = [False, False]
    for _ in range(3):
        for _ in range(vec.envs[0].max_actions_per_round):
            _, _, terminated, _, _ = vec.vector_step([PASS, PASS], terminated)
            assert not any(terminated)
        seen.append([_state(sub.PLAYER) for sub in vec.envs])
    assert seen == [[expected] * 2 for expected in FIRST_THREE_PLANNING_PHASES + [(17, 4, 0)]]
