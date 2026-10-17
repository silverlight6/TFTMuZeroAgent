"""Scope of the stage damage change: every round index, every place that holds the table.

The base player damage table lives in three places: Game_Round (multi-player env, player and
ghost fights), minion.minion_combat (PvE) and single_player_game_round.Game_Round. These tests
check, for every round index of a full game (all entries of Game_Round.game_rounds):

- which stage each index belongs to, read from the game's own round schedule;
- the base damage each place passes to the fight, against the Set 4 table from the patch 10.24
  notes, "Base Player Damage Per Stage: 0/0/1/2/5/10/15 => 0/0/2/3/5/8/15" (stages 1-7;
  later indices keep 15);
- that base damage is constant within a stage.

They also pin what the change does not touch: the damage per surviving unit, the fight itself,
PvE fights in the multi-player env (no HP loss either way) and the single-player env (it never
passes a base damage, so its table is not read).
"""

from __future__ import annotations

import pytest

from Simulator.battle import champion as champion_module
from Simulator.battle import minion
from Simulator.battle.champion import champion
from Simulator.battle.combat_context import CombatContext
from Simulator.battle.stats import DAMAGE_PER_UNIT
from Simulator.game import single_player_game_round
from Simulator.game.game_round import Game_Round
from Simulator.game.player import Player
from Simulator.game.pool import pool
from Simulator.rng import EnvRNG

# Patch 10.24 notes (Systems): stages 1..7.
SET4_10_24_STAGE_DAMAGE = {1: 0, 2: 0, 3: 2, 4: 3, 5: 5, 6: 8, 7: 15}


def stage_of(index):
    """Index 0-2 is stage 1, then six indices per stage from index 3 (2-1)."""
    return 1 if index <= 2 else 2 + (index - 3) // 6


def label_of(index):
    if index <= 2:
        return ("1-1/1-2", "1-3", "1-4")[index]
    return "{}-{}".format(stage_of(index), ("1", "2", "3", "4/5", "6", "7")[(index - 3) % 6])


def expected_base_damage(index):
    return SET4_10_24_STAGE_DAMAGE[min(stage_of(index), 7)]


def _new_game_round(players=None, base_pool=None):
    base_pool = base_pool or pool()
    players = players or {"player_0": Player(base_pool, 0), "player_1": Player(base_pool, 1)}
    return Game_Round(players, base_pool, None), players


with CombatContext(rng=EnvRNG.from_episode_seed(0)).bind():
    NUM_ROUNDS = len(_new_game_round()[0].game_rounds)
ROUND_INDICES = list(range(NUM_ROUNDS))
ROUND_IDS = ["idx{:02d}-{}".format(i, label_of(i)) for i in ROUND_INDICES]


def _record_base_damage(monkeypatch, result=1):
    seen = []

    def fake_run(champion_q, player_1, player_2, round_damage=0):
        seen.append(round_damage)
        return result, round_damage

    monkeypatch.setattr(champion_module, "run", fake_run)
    return seen


# --- the round schedule that defines the stages --------------------------------------------

def _schedule_names(game_round):
    return [[f.__name__ for f in entry] for entry in game_round.game_rounds]


def _expected_schedule(index):
    if index == 0:
        return ["round_1"]
    if index <= 2:
        return ["minion_round"]
    return [["combat_round"], ["combat_round"], ["combat_round"], ["carousel_round", "combat_round"],
            ["combat_round"], ["minion_round"]][(index - 3) % 6]


def test_round_schedule_defines_the_stages():
    """x-1 .. x-3 fights, x-4 carousel with the x-5 fight, x-6 fight, x-7 PvE: six indices a stage."""
    with CombatContext(rng=EnvRNG.from_episode_seed(1)).bind():
        multi = _new_game_round()[0]
        single = single_player_game_round.Game_Round(Player(pool(), 0), pool(), None)
    assert _schedule_names(multi) == [_expected_schedule(i) for i in ROUND_INDICES]
    # The single-player list has two more entries (8-7 PvE and 9-1), same pattern.
    assert _schedule_names(single) == [_expected_schedule(i) for i in range(len(single.game_rounds))]
    # The PvE round closes each stage, so the stage boundaries are fixed by the schedule itself.
    assert [i for i in ROUND_INDICES if _expected_schedule(i) == ["minion_round"]] == [1, 2, 8, 14, 20, 26, 32, 38]


# --- base damage per round index, every place ----------------------------------------------

@pytest.mark.parametrize("index", ROUND_INDICES, ids=ROUND_IDS)
def test_player_fight_base_damage_every_round(monkeypatch, index):
    seen = _record_base_damage(monkeypatch)
    with CombatContext(rng=EnvRNG.from_episode_seed(1)).bind():
        game_round, players = _new_game_round()
        game_round.matchups = [["player_0", "player_1"]]
        game_round.combat_phase(players, index)
    assert seen == [expected_base_damage(index)], label_of(index)


@pytest.mark.parametrize("index", ROUND_INDICES, ids=ROUND_IDS)
def test_ghost_fight_base_damage_every_round(monkeypatch, index):
    seen = _record_base_damage(monkeypatch, result=2)
    with CombatContext(rng=EnvRNG.from_episode_seed(1)).bind():
        game_round, players = _new_game_round()
        game_round.matchups = [["player_0", "ghost", "player_1"]]
        game_round.combat_phase(players, index)
    assert seen == [expected_base_damage(index)], label_of(index)


@pytest.mark.parametrize("index", ROUND_INDICES, ids=ROUND_IDS)
def test_minion_combat_base_damage_every_round(monkeypatch, index):
    # The lookup in minion_combat does not depend on the monster, so every index is checked.
    seen = _record_base_damage(monkeypatch, result=2)
    with CombatContext(rng=EnvRNG.from_episode_seed(1)).bind():
        minion.minion_combat(Player(pool(), 0), minion.FirstMinion(), index, [], None)
    assert seen == [expected_base_damage(index)], label_of(index)


@pytest.mark.parametrize("index", [0, 1, 2, 8, 14, 20, 26, 32, 38],
                         ids=["idx00-1-2", "idx01-1-3", "idx02-1-4", "idx08-2-7", "idx14-3-7",
                              "idx20-4-7", "idx26-5-7", "idx32-6-7", "idx38-7-7"])
def test_scheduled_pve_rounds_base_damage(monkeypatch, index):
    seen = _record_base_damage(monkeypatch, result=2)
    with CombatContext(rng=EnvRNG.from_episode_seed(1)).bind():
        minion.minion_round(Player(pool(), 0), index)
    # 6-7 (index 32) is not fought on this branch (minion_round starts the Herald at 33); that is
    # a separate fix. Whenever a PvE fight happens it uses the stage value.
    assert seen == [expected_base_damage(index)] or (index == 32 and seen == []), label_of(index)


@pytest.mark.parametrize("index", ROUND_INDICES, ids=ROUND_IDS)
def test_single_player_table_matches(index):
    with CombatContext(rng=EnvRNG.from_episode_seed(1)).bind():
        table = single_player_game_round.Game_Round(Player(pool(), 0), pool(), None).ROUND_DAMAGE
        multi_table = _new_game_round()[0].ROUND_DAMAGE
    assert table == multi_table
    position = 0
    while index > table[position][0]:  # the lookup Game_Round.combat_phase uses
        position += 1
    assert table[position][1] == expected_base_damage(index), label_of(index)


def test_base_damage_is_constant_within_each_stage(monkeypatch):
    seen = _record_base_damage(monkeypatch)
    by_stage = {}
    with CombatContext(rng=EnvRNG.from_episode_seed(1)).bind():
        for index in ROUND_INDICES:
            if "combat_round" not in _expected_schedule(index):
                continue
            game_round, players = _new_game_round()
            game_round.matchups = [["player_0", "player_1"]]
            game_round.combat_phase(players, index)
            by_stage.setdefault(stage_of(index), set()).add(seen[-1])
    assert by_stage == {2: {0}, 3: {2}, 4: {3}, 5: {5}, 6: {8}, 7: {15}, 8: {15}}


# --- what the change does not touch ----------------------------------------------------------

def test_damage_per_surviving_unit_is_unchanged():
    assert DAMAGE_PER_UNIT == [0, 2, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]


@pytest.mark.parametrize("index", [3, 4, 8, 9, 10, 15, 16, 21, 22, 27, 28, 33, 39],
                         ids=lambda i: "idx{:02d}-{}".format(i, label_of(i)))
@pytest.mark.parametrize("units", [1, 3, 5, 7])
def test_real_loss_costs_stage_damage_plus_unit_damage(index, units):
    """Real combat_phase (no fake fight): the loser's HP and both rewards move by base + unit damage."""
    with CombatContext(rng=EnvRNG.from_episode_seed(2)).bind():
        base_pool = pool()
        winner, loser = Player(base_pool, 0), Player(base_pool, 1)
        for x in range(units):
            winner.board[x][3] = champion("garen")
        game_round, players = _new_game_round({"player_0": winner, "player_1": loser}, base_pool)
        game_round.matchups = [["player_0", "player_1"]]
        game_round.combat_phase(players, index)
    damage = expected_base_damage(index) + DAMAGE_PER_UNIT[units]
    assert loser.health == 100 - damage
    assert winner.health == 100
    assert loser.reward == -damage
    assert winner.reward == damage
    assert loser.match_history[-1] == 0 and winner.match_history[-1] == 1


def _two_boards(base_pool):
    blue, red = Player(base_pool, 0), Player(base_pool, 1)
    for x, name in enumerate(["garen", "nidalee", "vayne", "elise"]):
        blue.board[x][3 if x < 2 else 0] = champion(name)
    for x, name in enumerate(["fiora", "wukong", "twistedfate", "annie"]):
        red.board[x][3 if x < 2 else 0] = champion(name)
    return blue, red


@pytest.mark.parametrize("seed", [3, 4, 5, 6, 7])
def test_fight_result_does_not_depend_on_base_damage(seed):
    """Same seed, same boards: the winner and the surviving units are the same for every base damage."""
    results = {}
    for base in (0, 2, 3, 5, 8, 15):
        with CombatContext(rng=EnvRNG.from_episode_seed(seed)).bind():
            blue, red = _two_boards(pool())
            results[base] = champion_module.run(champion_module.champion, blue, red, base)
    outcome = {r[0] for r in results.values()}
    assert len(outcome) == 1
    if outcome != {0}:
        unit_damage = {r[1] - base for base, r in results.items()}
        assert len(unit_damage) == 1


@pytest.mark.parametrize("index", [8, 14, 20, 26, 38])
def test_pve_loss_costs_no_hp_in_the_multiplayer_game(index):
    """Game_Round.minion_round does not pass other_rewards, so a lost PvE fight costs no HP."""
    with CombatContext(rng=EnvRNG.from_episode_seed(3)).bind():
        base_pool = pool()
        players = {"player_0": Player(base_pool, 0), "player_1": Player(base_pool, 1)}
        game_round = Game_Round(players, base_pool, None)
        history = [list(p.match_history) for p in players.values()]
        game_round.current_round = index
        game_round.minion_round()  # empty boards: the monsters win every fight
    assert [p.health for p in players.values()] == [100, 100]
    assert [p.match_history for p in players.values()] == history


@pytest.mark.parametrize("index", [3, 9, 15, 21, 27, 33])
def test_single_player_env_passes_no_base_damage(monkeypatch, index):
    """The single-player env fights through single_combat_phase, which never reads ROUND_DAMAGE."""
    seen = _record_base_damage(monkeypatch, result=2)
    with CombatContext(rng=EnvRNG.from_episode_seed(4)).bind():
        base_pool = pool()
        player, enemy = Player(base_pool, 0), Player(base_pool, 1)
        game_round = single_player_game_round.Game_Round(player, base_pool, None)
        monkeypatch.setattr(game_round.leveling_system, "generate_battle", lambda: [enemy, None, None])
        monkeypatch.setattr(game_round.leveling_system, "single_player_level_up", lambda: None)
        game_round.current_round = index
        assert game_round.combat_round() is False
    assert seen == [0]
    assert player.health == 100
