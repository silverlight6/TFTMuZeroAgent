"""Hierarchical 7-D MultiDiscrete action space.

Dimensions, each with a skip/0 slot:
    [pass, level, refresh, shop, board, bench, item]

The first non-zero dimension wins. Later dimensions are ignored.
    [1, *, *, *, *, *, *]                 pass
    [0, 1, *, *, *, *, *]                 level
    [0, 0, 1, *, *, *, *]                 refresh
    [0, 0, 0, s, *, *, *]                 buy shop slot s-1
    [0, 0, 0, 0, b, n, *]                 move bench n-1 to board b-1
    [0, 0, 0, 0, b, 0, i]                 put item i-1 on board b-1
    [0, 0, 0, 0, 0, 0, 0]                 pass (nothing selected)

Board-to-board is not representable here; use ActionVector or ActionToken.
"""

import numpy as np
from gymnasium.spaces import Box, MultiDiscrete

from Simulator.encoding.token.action import ActionToken
from Simulator.encoding.token.action_vector import N_BENCH, N_BOARD, N_ITEM, N_SHOP


# Inclusive of the 0 / skip slot.
NVEC = (
    2,              # pass
    2,              # level
    2,              # refresh
    1 + N_SHOP,     # shop
    1 + N_BOARD,    # board
    1 + N_BENCH,    # bench
    1 + N_ITEM,     # item
)
MASK_DIM = int(sum(NVEC))

PASS_SLICE = slice(0, 2)
LEVEL_SLICE = slice(2, 4)
REFRESH_SLICE = slice(4, 6)
SHOP_SLICE = slice(6, 12)
BOARD_SLICE = slice(12, 41)
BENCH_SLICE = slice(41, 51)
ITEM_SLICE = slice(51, 62)


class ActionMultiDiscrete(ActionToken):
    """7-D MultiDiscrete action space with a concatenated per-dimension mask."""

    @staticmethod
    def action_space():
        return MultiDiscrete(list(NVEC))

    @staticmethod
    def action_mask_space():
        return Box(0, 1, shape=(MASK_DIM,), dtype=np.int8)

    @staticmethod
    def mask_to_sample_mask(mask):
        """Split the concatenated mask into the tuple MultiDiscrete.sample expects."""
        mask = np.asarray(mask, dtype=np.int8).reshape(-1)
        if mask.size != MASK_DIM:
            raise ValueError(f"Expected mask of length {MASK_DIM}, got {mask.size}")
        parts = []
        offset = 0
        for size in NVEC:
            parts.append(np.asarray(mask[offset:offset + size], dtype=np.int8))
            offset += size
        return tuple(parts)

    @staticmethod
    def pass_vector():
        return np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.int64)

    @staticmethod
    def level_vector():
        return np.array([0, 1, 0, 0, 0, 0, 0], dtype=np.int64)

    @staticmethod
    def refresh_vector():
        return np.array([0, 0, 1, 0, 0, 0, 0], dtype=np.int64)

    @staticmethod
    def shop_vector(shop):
        return np.array([0, 0, 0, int(shop) + 1, 0, 0, 0], dtype=np.int64)

    @staticmethod
    def move_bench_to_board_vector(bench, board):
        return np.array([0, 0, 0, 0, int(board) + 1, int(bench) + 1, 0], dtype=np.int64)

    @staticmethod
    def item_to_board_vector(item, board):
        return np.array([0, 0, 0, 0, int(board) + 1, 0, int(item) + 1], dtype=np.int64)

    @staticmethod
    def action_space_to_action(action):
        action = np.asarray(action)
        if action.shape == (3,):
            return [int(action[0]), int(action[1]), int(action[2])]

        vec = np.asarray(action, dtype=np.int64).reshape(-1)
        if vec.size != 7:
            raise ValueError(f"Expected a 7-D action vector, got shape {np.asarray(action).shape}")

        passed, level, refresh, shop, board, bench, item = (int(v) for v in vec)

        if passed:
            return [0, 0, 0]
        if level:
            return [1, 0, 0]
        if refresh:
            return [2, 0, 0]
        if shop:
            return [3, shop - 1, 0]
        if board and bench:
            return [5, N_BOARD + (bench - 1), board - 1]
        if board and item:
            return [6, board - 1, item - 1]
        return [0, 0, 0]

    def fetch_action_mask(self):
        mask = np.zeros(MASK_DIM, dtype=np.int8)
        # Skip/0 is always legal so the cascade can fall through.
        mask[PASS_SLICE.start] = 1
        mask[LEVEL_SLICE.start] = 1
        mask[REFRESH_SLICE.start] = 1
        mask[SHOP_SLICE.start] = 1
        mask[BOARD_SLICE.start] = 1
        mask[BENCH_SLICE.start] = 1
        mask[ITEM_SLICE.start] = 1

        mask[PASS_SLICE.start + 1] = 1
        mask[LEVEL_SLICE.start + 1] = int(self.exp_mask)
        mask[REFRESH_SLICE.start + 1] = int(self.refresh_mask)
        mask[SHOP_SLICE.start + 1:SHOP_SLICE.stop] = np.asarray(self.buy_mask) > 0

        bench_to_board = np.asarray(self.move_sell_bench_mask)[:, :N_BOARD] > 0
        item_to_board = np.asarray(self.item_mask)[:, :N_BOARD] > 0
        mask[BOARD_SLICE.start + 1:BOARD_SLICE.stop] = bench_to_board.any(axis=0) | item_to_board.any(axis=0)
        mask[BENCH_SLICE.start + 1:BENCH_SLICE.stop] = bench_to_board.any(axis=1)
        mask[ITEM_SLICE.start + 1:ITEM_SLICE.stop] = item_to_board.any(axis=1)
        return mask
