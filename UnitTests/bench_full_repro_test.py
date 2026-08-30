"""Carousel onto a full bench must autosell quietly.

Shop buys are already masked when the bench is full. A carousel grant is an
environment action: if the unit cannot upgrade, sell it for gold and do not
print ``add_to_bench but full``.
"""

from __future__ import annotations

import numpy as np

from Simulator import config
from Simulator.battle.combat_context import CombatContext
from Simulator.encoding.token.action import ActionToken
from Simulator.game.carousel import carousel
from Simulator.game import player as player_mod
from Simulator.game.player import Player
from Simulator.game.pool import pool
from Simulator.rng import EnvRNG
from Simulator.simulators.tft_simulator import TFTConfig, parallel_env


N_TO = 38
SHOP_FROM = 47
PASS_FROM = 52
REFRESH_FROM = 54
PASS_INDEX = PASS_FROM * N_TO
REFRESH_INDEX = REFRESH_FROM * N_TO
SHOP_START = SHOP_FROM * N_TO
SHOP_STOP = PASS_FROM * N_TO

CAROUSEL_ROUNDS = (6, 12, 18, 24, 30)


def _quiet(monkeypatch):
    monkeypatch.setattr(config, "DEBUG", False)
    monkeypatch.setattr(config, "PRINTMESSAGES", False)
    monkeypatch.setattr(config, "LOGMESSAGES", False)
    monkeypatch.setattr(config, "LOG_COMBAT", False)
    monkeypatch.setattr(config, "AUTO_BATTLER_PERCENTAGE", 1)
    monkeypatch.setattr(player_mod, "DEBUG", False)


def _legal_indices(mask):
    return np.flatnonzero(np.asarray(mask).reshape(-1) > 0)


def _shop_slot(index):
    from_idx, dest = divmod(int(index), N_TO)
    if dest != 0 or from_idx < SHOP_FROM or from_idx >= PASS_FROM:
        return None
    return from_idx - SHOP_FROM


def _owned_copies(player, name, stars=1):
    count = 0
    for champ in player.bench:
        if champ and champ.name == name and champ.stars == stars:
            count += 1
    for row in player.board:
        for champ in row:
            if champ and champ.name == name and champ.stars == stars:
                count += 1
    return count


def _shop_name(player, slot):
    champ = player.shop_champions[slot] if player.shop_champions else None
    return champ.name if champ else None


def pick_visible_fill_action(mask, player):
    """Buy a unit that will occupy a new bench slot; otherwise refresh or pass.

    Never returns an index the mask marks illegal. Avoids a 3rd copy so a
    triple does not free bench space before carousel.
    """
    legal = set(_legal_indices(mask).tolist())
    assert legal, "action mask has no visible actions"

    if player is not None and not player.bench_full():
        shop_choices = []
        for index in sorted(i for i in legal if SHOP_START <= i < SHOP_STOP):
            slot = _shop_slot(index)
            if slot is None:
                continue
            name = _shop_name(player, slot)
            if name and _owned_copies(player, name) < 2:
                shop_choices.append(index)
        if shop_choices:
            return shop_choices[0]
        if REFRESH_INDEX in legal:
            return REFRESH_INDEX

    if PASS_INDEX in legal:
        return PASS_INDEX
    return min(legal)


def record_full_bench_adds(monkeypatch):
    """Split full-bench grants into the old terminal warning vs quiet carousel sells."""
    warnings = []
    autosells = []
    original = Player.add_to_bench

    def wrapped(self, a_champion, from_carousel=False):
        vacancy = self.bench_vacancy()
        result = original(self, a_champion, from_carousel)
        if vacancy < 0:
            payload = {
                "champion": a_champion.name,
                "round": self.round,
                "units_in_play": self.num_units_in_play,
                "max_units": self.max_units,
                "from_carousel": from_carousel,
                "result": result,
            }
            if not result and not from_carousel:
                warnings.append(payload)
            elif from_carousel and result:
                autosells.append(payload)
        return result

    monkeypatch.setattr(Player, "add_to_bench", wrapped)
    return warnings, autosells


def _apply_command(player, handler, command):
    action_type, x1, _x2 = command
    if action_type == 0:
        player.pass_action()
    elif action_type == 1:
        player.buy_exp_action()
    elif action_type == 2:
        player.refresh_shop_action()
    elif action_type == 3:
        player.buy_shop_action(x1)
    else:
        raise AssertionError(f"fill policy must stay on shop/refresh/pass, got {command}")
    handler.update_action_mask(command)


def fill_bench_with_visible_actions(player, handler, max_steps=80):
    """Drive shop / refresh / pass from the live ActionToken mask until full."""
    taken = []
    for _ in range(max_steps):
        if player.bench_full():
            return taken
        mask = np.asarray(handler.fetch_action_mask()).reshape(-1)
        index = pick_visible_fill_action(mask, player)
        assert mask[index] > 0, f"policy chose masked index {index}"
        command = ActionToken.action_space_to_action(index)
        _apply_command(player, handler, command)
        taken.append((index, command))
    return taken


def test_legal_shop_sequence_then_carousel_autosells_quietly(monkeypatch, capsys):
    """Buy onto a 9-unit bench using only visible actions, then run carousel."""
    _quiet(monkeypatch)
    warnings, autosells = record_full_bench_adds(monkeypatch)

    ctx = CombatContext(rng=EnvRNG.from_episode_seed(7))
    with ctx.bind():
        base_pool = pool()
        player = Player(base_pool, 0)
        player.gold = 200
        player.round = 6
        player.max_units = 4
        player.refresh_shop()
        handler = ActionToken(player)

        taken = fill_bench_with_visible_actions(player, handler)
        assert player.bench_full(), (
            f"could not fill bench with visible shop actions after {len(taken)} steps; "
            f"occupied={sum(c is not None for c in player.bench)}"
        )
        assert taken, "expected at least one visible buy or refresh"
        assert all(cmd[0] in (0, 2, 3) for _idx, cmd in taken)

        buy_mask = np.asarray(handler.create_buy_action_mask(player))
        assert not np.any(buy_mask), "shop buys must stay masked on a full bench"

        gold_before = player.gold
        bench_before = [champ.name if champ else None for champ in player.bench]
        monkeypatch.setattr(config, "DEBUG", True)
        monkeypatch.setattr(player_mod, "DEBUG", True)
        carousel([player], player.round, base_pool)
        captured = capsys.readouterr()
        assert "add_to_bench but full" not in captured.out
        assert "add_to_bench but full" not in captured.err

        assert player.bench_full()
        assert [champ.name if champ else None for champ in player.bench] == bench_before
        assert player.gold > gold_before

    assert not warnings
    assert autosells
    assert autosells[0]["round"] == 6
    assert autosells[0]["from_carousel"] is True


def test_training_env_carousel_full_bench_does_not_warn(monkeypatch):
    """Same quiet autosell through the 8-player ActionToken env used by training."""
    _quiet(monkeypatch)
    warnings, autosells = record_full_bench_adds(monkeypatch)

    env = parallel_env(
        TFTConfig(action_class=ActionToken, num_players=8, max_actions_per_round=15)
    )
    observations, infos = env.reset(seed=7)
    illegal_buys_on_full_bench = 0

    try:
        for _ in range(8 * 15 * 12):
            if autosells or not env.agents:
                break
            actions = {}
            for agent in env.agents:
                mask = observations[agent]["action_mask"]
                player = infos.get(agent, {}).get("player")
                index = pick_visible_fill_action(mask, player)
                assert mask[index] > 0, f"{agent} chose masked index {index}"
                if player is not None and player.bench_full() and _shop_slot(index) is not None:
                    illegal_buys_on_full_bench += 1
                actions[agent] = index
            observations, _rewards, terms, _truncs, infos = env.step(actions)
            if all(terms.get(agent, False) for agent in env.possible_agents):
                break
    finally:
        env.close()

    assert illegal_buys_on_full_bench == 0
    assert not warnings
    assert autosells
    assert any(event["round"] in CAROUSEL_ROUNDS for event in autosells)
