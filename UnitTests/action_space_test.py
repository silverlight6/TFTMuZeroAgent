"""Unit tests for the compact vector and 7-D MultiDiscrete action spaces."""

import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete

from Simulator.battle.champion import champion
from Simulator.encoding.token.action import ActionToken
from Simulator.encoding.token.action_multi import (
    ActionMultiDiscrete,
    BENCH_SLICE,
    BOARD_SLICE,
    ITEM_SLICE,
    LEVEL_SLICE,
    MASK_DIM,
    NVEC,
    PASS_SLICE,
    REFRESH_SLICE,
    SHOP_SLICE,
)
from Simulator.encoding.token.action_vector import (
    ACTION_DIM,
    ActionVector,
    BOARD_TO_BOARD,
    BENCH_TO_BOARD,
    ITEM_TO_BOARD,
    LEVEL_INDEX,
    PASS_INDEX,
    REFRESH_INDEX,
    SHOP_START,
)
from Simulator.game.player import Player
from Simulator.game.pool import pool
from Simulator.simulators.tft_simulator import TFTConfig, parallel_env
from Simulator.simulators.tft_single_player_simulator import TFT_Single_Player_Simulator


def _player_with_board_bench_item():
    player = Player(pool(), 0)
    player.gold = 50
    player.max_units = 4
    player.buy_champion(champion("leesin"))
    player.move_bench_to_board(0, 0, 0)
    player.buy_champion(champion("nami"))
    player.add_to_item_bench("bf_sword")
    player.refresh_shop()
    return player


def _token_grid(player):
    return np.asarray(ActionToken(player).fetch_action_mask())


def test_action_vector_space_and_mask_shapes():
    space = ActionVector.action_space()
    mask_space = ActionVector.action_mask_space()
    assert isinstance(space, Discrete)
    assert space.n == ACTION_DIM == 9 * 28 + 28 * 27 + 10 * 28 + 5 + 3
    assert mask_space.shape == (ACTION_DIM,)

    player = _player_with_board_bench_item()
    mask = ActionVector(player).fetch_action_mask()
    assert mask.shape == (ACTION_DIM,)
    assert mask.dtype == np.int8
    assert mask_space.contains(mask)
    assert mask[PASS_INDEX] == 1


def test_action_vector_decodes_each_region():
    assert ActionVector.action_space_to_action(ActionVector.bench_to_board_index(0, 0)) == [5, 28, 0]
    assert ActionVector.action_space_to_action(ActionVector.bench_to_board_index(2, 7)) == [5, 30, 7]
    assert ActionVector.action_space_to_action(ActionVector.board_to_board_index(0, 1)) == [5, 0, 1]
    assert ActionVector.action_space_to_action(ActionVector.board_to_board_index(5, 3)) == [5, 5, 3]
    assert ActionVector.action_space_to_action(ActionVector.board_to_board_index(27, 0)) == [5, 27, 0]
    assert ActionVector.action_space_to_action(ActionVector.item_to_board_index(0, 4)) == [6, 4, 0]
    assert ActionVector.action_space_to_action(ActionVector.shop_index(3)) == [3, 3, 0]
    assert ActionVector.action_space_to_action(PASS_INDEX) == [0, 0, 0]
    assert ActionVector.action_space_to_action(LEVEL_INDEX) == [1, 0, 0]
    assert ActionVector.action_space_to_action(REFRESH_INDEX) == [2, 0, 0]


def test_action_vector_index_helpers_round_trip():
    for bench in (0, 4, 8):
        for board in (0, 13, 27):
            idx = ActionVector.bench_to_board_index(bench, board)
            assert ActionVector.action_space_to_action(idx) == [5, 28 + bench, board]

    for from_board in (0, 9, 27):
        for to_board in (0, 9, 27):
            if from_board == to_board:
                continue
            idx = ActionVector.board_to_board_index(from_board, to_board)
            assert ActionVector.action_space_to_action(idx) == [5, from_board, to_board]

    assert BENCH_TO_BOARD + BOARD_TO_BOARD + ITEM_TO_BOARD + 5 + 3 == ACTION_DIM


def test_action_vector_mask_matches_token_overlap():
    player = _player_with_board_bench_item()
    token = _token_grid(player)
    vector = ActionVector(player).fetch_action_mask()

    for bench in range(9):
        for board in range(28):
            idx = ActionVector.bench_to_board_index(bench, board)
            assert vector[idx] == int(token[28 + bench, board] > 0)

    for from_board in range(28):
        for to_board in range(28):
            if from_board == to_board:
                continue
            idx = ActionVector.board_to_board_index(from_board, to_board)
            assert vector[idx] == int(token[from_board, to_board] > 0)

    for item in range(10):
        for board in range(28):
            idx = ActionVector.item_to_board_index(item, board)
            assert vector[idx] == int(token[37 + item, board] > 0)

    for shop in range(5):
        assert vector[ActionVector.shop_index(shop)] == int(token[47 + shop, 0] > 0)

    assert vector[PASS_INDEX] == int(token[52, 0] > 0)
    assert vector[LEVEL_INDEX] == int(token[53, 0] > 0)
    assert vector[REFRESH_INDEX] == int(token[54, 0] > 0)


def test_action_vector_masks_empty_and_occupied_slots():
    player = _player_with_board_bench_item()
    mask = ActionVector(player).fetch_action_mask()

    assert mask[ActionVector.bench_to_board_index(0, 3)] == 1
    assert mask[ActionVector.bench_to_board_index(1, 3)] == 0
    assert mask[ActionVector.board_to_board_index(0, 5)] == 1
    assert mask[ActionVector.item_to_board_index(0, 0)] == 1
    assert mask[ActionVector.item_to_board_index(1, 0)] == 0


def test_action_vector_low_gold_masks_econ():
    player = _player_with_board_bench_item()
    player.gold = 0
    mask = ActionVector(player).fetch_action_mask()
    assert mask[LEVEL_INDEX] == 0
    assert mask[REFRESH_INDEX] == 0
    assert not np.any(mask[SHOP_START:PASS_INDEX])
    assert mask[PASS_INDEX] == 1


def test_action_vector_update_mask_after_buy():
    player = _player_with_board_bench_item()
    handler = ActionVector(player)
    before = handler.fetch_action_mask().copy()
    shop = int(np.argmax(before[SHOP_START:PASS_INDEX]))
    assert before[ActionVector.shop_index(shop)] == 1
    command = ActionVector.action_space_to_action(ActionVector.shop_index(shop))
    player.buy_shop_action(command[1])
    handler.update_action_mask(command)
    after = handler.fetch_action_mask()
    assert after[ActionVector.shop_index(shop)] == 0


def test_action_multi_space_and_mask_shapes():
    space = ActionMultiDiscrete.action_space()
    mask_space = ActionMultiDiscrete.action_mask_space()
    assert isinstance(space, MultiDiscrete)
    assert list(space.nvec) == list(NVEC) == [2, 2, 2, 6, 29, 10, 11]
    assert mask_space.shape == (MASK_DIM,) == (62,)

    player = _player_with_board_bench_item()
    mask = ActionMultiDiscrete(player).fetch_action_mask()
    assert mask.shape == (MASK_DIM,)
    assert mask.dtype == np.int8
    assert mask_space.contains(mask)

    sample_mask = ActionMultiDiscrete.mask_to_sample_mask(mask)
    assert len(sample_mask) == 7
    for part, size in zip(sample_mask, NVEC):
        assert part.shape == (size,)
        assert part.dtype == np.int8
    assert space.contains(space.sample(mask=sample_mask))


def test_action_multi_cascade_examples():
    assert ActionMultiDiscrete.action_space_to_action([0, 0, 0, 0, 1, 1, 9]) == [5, 28, 0]
    assert ActionMultiDiscrete.action_space_to_action([1, 1, 1, 4, 8, 3, 2]) == [0, 0, 0]
    assert ActionMultiDiscrete.action_space_to_action([0, 1, 1, 4, 8, 3, 2]) == [1, 0, 0]
    assert ActionMultiDiscrete.action_space_to_action([0, 0, 1, 4, 8, 3, 2]) == [2, 0, 0]
    assert ActionMultiDiscrete.action_space_to_action([0, 0, 0, 3, 8, 3, 2]) == [3, 2, 0]
    assert ActionMultiDiscrete.action_space_to_action([0, 0, 0, 0, 5, 0, 1]) == [6, 4, 0]
    assert ActionMultiDiscrete.action_space_to_action([0, 0, 0, 0, 5, 0, 0]) == [0, 0, 0]
    assert ActionMultiDiscrete.action_space_to_action(np.zeros(7, dtype=np.int64)) == [0, 0, 0]


def test_action_multi_helpers_match_decode():
    assert ActionMultiDiscrete.action_space_to_action(ActionMultiDiscrete.pass_vector()) == [0, 0, 0]
    assert ActionMultiDiscrete.action_space_to_action(ActionMultiDiscrete.level_vector()) == [1, 0, 0]
    assert ActionMultiDiscrete.action_space_to_action(ActionMultiDiscrete.refresh_vector()) == [2, 0, 0]
    assert ActionMultiDiscrete.action_space_to_action(ActionMultiDiscrete.shop_vector(4)) == [3, 4, 0]
    assert ActionMultiDiscrete.action_space_to_action(
        ActionMultiDiscrete.move_bench_to_board_vector(2, 6)
    ) == [5, 30, 6]
    assert ActionMultiDiscrete.action_space_to_action(
        ActionMultiDiscrete.item_to_board_vector(3, 11)
    ) == [6, 11, 3]


def test_action_multi_zero_slots_always_legal():
    player = Player(pool(), 0)
    player.gold = 0
    mask = ActionMultiDiscrete(player).fetch_action_mask()
    assert mask[PASS_SLICE.start] == 1
    assert mask[LEVEL_SLICE.start] == 1
    assert mask[REFRESH_SLICE.start] == 1
    assert mask[SHOP_SLICE.start] == 1
    assert mask[BOARD_SLICE.start] == 1
    assert mask[BENCH_SLICE.start] == 1
    assert mask[ITEM_SLICE.start] == 1
    assert mask[PASS_SLICE.start + 1] == 1
    assert mask[LEVEL_SLICE.start + 1] == 0
    assert mask[REFRESH_SLICE.start + 1] == 0


def test_action_multi_mask_matches_token_components():
    player = _player_with_board_bench_item()
    token = _token_grid(player)
    handler = ActionMultiDiscrete(player)
    mask = handler.fetch_action_mask()

    assert mask[LEVEL_SLICE.start + 1] == int(token[53, 0] > 0)
    assert mask[REFRESH_SLICE.start + 1] == int(token[54, 0] > 0)
    for shop in range(5):
        assert mask[SHOP_SLICE.start + 1 + shop] == int(token[47 + shop, 0] > 0)

    bench_to_board = token[28:37, :28] > 0
    item_to_board = token[37:47, :28] > 0
    assert np.array_equal(mask[BENCH_SLICE.start + 1:BENCH_SLICE.stop], bench_to_board.any(axis=1))
    assert np.array_equal(mask[ITEM_SLICE.start + 1:ITEM_SLICE.stop], item_to_board.any(axis=1))
    assert np.array_equal(
        mask[BOARD_SLICE.start + 1:BOARD_SLICE.stop],
        bench_to_board.any(axis=0) | item_to_board.any(axis=0),
    )
    assert mask[BENCH_SLICE.start + 1] == 1
    assert mask[BENCH_SLICE.start + 2] == 0
    assert mask[ITEM_SLICE.start + 1] == 1
    assert mask[ITEM_SLICE.start + 2] == 0
    assert mask[BOARD_SLICE.start + 1] == 1


def test_action_multi_item_ignored_when_move_selected():
    command = ActionMultiDiscrete.action_space_to_action([0, 0, 0, 0, 1, 1, 4])
    assert command == [5, 28, 0]


def test_decode_env_action_accepts_commands_for_all_classes():
    command = [3, 2, 0]
    for cls in (ActionToken, ActionVector, ActionMultiDiscrete):
        assert cls.decode_env_action(command) == command
        assert cls.decode_env_action(np.asarray(command)) == command


def test_action_token_pair_still_decodes():
    assert ActionToken.decode_env_action(np.array([52, 0])) == [0, 0, 0]
    assert ActionToken.decode_env_action(np.array([47, 0])) == [3, 0, 0]


def _step_legal_actions(env, observations, action_cls, steps=4):
    for _ in range(steps):
        if not env.agents:
            break
        actions = {}
        for agent in env.agents:
            mask = observations[agent]["action_mask"]
            space = env.action_space(agent)
            obs_space = env.observation_space(agent)
            assert obs_space["action_mask"].contains(mask)
            if hasattr(space, "nvec"):
                sample = space.sample(mask=action_cls.mask_to_sample_mask(mask))
            else:
                legal = np.flatnonzero(mask > 0)
                sample = int(legal[np.random.randint(len(legal))])
            assert space.contains(sample)
            actions[agent] = sample
        observations, *_ = env.step(actions)
    return observations


def test_action_vector_env_steps():
    env = parallel_env(TFTConfig(action_class=ActionVector, num_players=8, max_actions_per_round=3))
    observations, _ = env.reset()
    assert env.action_space("player_0").n == ACTION_DIM
    assert observations["player_0"]["action_mask"].shape == (ACTION_DIM,)
    _step_legal_actions(env, observations, ActionVector, steps=2)
    env.close()


def test_action_multi_env_steps():
    env = parallel_env(
        TFTConfig(action_class=ActionMultiDiscrete, num_players=8, max_actions_per_round=3)
    )
    observations, _ = env.reset()
    space = env.action_space("player_0")
    assert list(space.nvec) == [2, 2, 2, 6, 29, 10, 11]
    assert observations["player_0"]["action_mask"].shape == (MASK_DIM,)
    _step_legal_actions(env, observations, ActionMultiDiscrete, steps=2)
    env.close()


def test_action_vector_env_accepts_length_three_command():
    env = parallel_env(TFTConfig(action_class=ActionVector, num_players=8, max_actions_per_round=3))
    env.reset()
    observations, *_ = env.step({agent: [0, 0, 0] for agent in env.agents})
    assert observations["player_0"]["action_mask"].shape == (ACTION_DIM,)
    env.close()


def test_action_vector_single_player_steps():
    env = TFT_Single_Player_Simulator(
        TFTConfig(action_class=ActionVector, num_players=1, max_actions_per_round=3)
    )
    observation, _ = env.reset()
    assert observation["action_mask"].shape == (ACTION_DIM,)
    for _ in range(3):
        legal = np.flatnonzero(observation["action_mask"] > 0)
        observation, _, terminated, _, _ = env.step(int(legal[0]))
        if terminated:
            break
    env.close()


def test_action_multi_single_player_steps():
    env = TFT_Single_Player_Simulator(
        TFTConfig(action_class=ActionMultiDiscrete, num_players=1, max_actions_per_round=3)
    )
    observation, _ = env.reset()
    space = env.action_space
    assert observation["action_mask"].shape == (MASK_DIM,)
    for _ in range(3):
        sample = space.sample(mask=ActionMultiDiscrete.mask_to_sample_mask(observation["action_mask"]))
        observation, _, terminated, _, _ = env.step(sample)
        if terminated:
            break
    env.close()
