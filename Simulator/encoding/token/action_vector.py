"""Compact single-vector action space.

Layout (source-major, 1296 ints):
    9 * 28   bench to board
    28 * 27  board to board (destination skips the source hex)
    10 * 28  item to board
    5        shop
    3        pass, level, refresh
"""

import numpy as np
from gymnasium.spaces import Box, Discrete

from Simulator.encoding.token.action import ActionToken


N_BOARD = 28
N_BENCH = 9
N_ITEM = 10
N_SHOP = 5

BENCH_TO_BOARD = N_BENCH * N_BOARD
BOARD_TO_BOARD = N_BOARD * (N_BOARD - 1)
ITEM_TO_BOARD = N_ITEM * N_BOARD

BENCH_TO_BOARD_START = 0
BOARD_TO_BOARD_START = BENCH_TO_BOARD_START + BENCH_TO_BOARD
ITEM_TO_BOARD_START = BOARD_TO_BOARD_START + BOARD_TO_BOARD
SHOP_START = ITEM_TO_BOARD_START + ITEM_TO_BOARD
PASS_INDEX = SHOP_START + N_SHOP
LEVEL_INDEX = PASS_INDEX + 1
REFRESH_INDEX = LEVEL_INDEX + 1
ACTION_DIM = REFRESH_INDEX + 1


class ActionVector(ActionToken):
    """Discrete(1296) action space with a matching 1-D legality mask."""

    @staticmethod
    def action_space():
        return Discrete(ACTION_DIM)

    @staticmethod
    def action_mask_space():
        return Box(0, 1, shape=(ACTION_DIM,), dtype=np.int8)

    @staticmethod
    def bench_to_board_index(bench, board):
        return BENCH_TO_BOARD_START + int(bench) * N_BOARD + int(board)

    @staticmethod
    def board_to_board_index(from_board, to_board):
        from_board = int(from_board)
        to_board = int(to_board)
        if from_board == to_board:
            raise ValueError("board-to-board moves cannot use the same hex")
        compact = to_board if to_board < from_board else to_board - 1
        return BOARD_TO_BOARD_START + from_board * (N_BOARD - 1) + compact

    @staticmethod
    def item_to_board_index(item, board):
        return ITEM_TO_BOARD_START + int(item) * N_BOARD + int(board)

    @staticmethod
    def shop_index(shop):
        return SHOP_START + int(shop)

    @staticmethod
    def action_space_to_action(action):
        action = np.asarray(action)
        if action.shape == (3,):
            return [int(action[0]), int(action[1]), int(action[2])]
        action = int(action)

        if action < BOARD_TO_BOARD_START:
            local = action - BENCH_TO_BOARD_START
            bench, board = divmod(local, N_BOARD)
            return [5, N_BOARD + bench, board]

        if action < ITEM_TO_BOARD_START:
            local = action - BOARD_TO_BOARD_START
            from_board, compact = divmod(local, N_BOARD - 1)
            to_board = compact if compact < from_board else compact + 1
            return [5, from_board, to_board]

        if action < SHOP_START:
            local = action - ITEM_TO_BOARD_START
            item, board = divmod(local, N_BOARD)
            return [6, board, item]

        if action < PASS_INDEX:
            return [3, action - SHOP_START, 0]

        if action == PASS_INDEX:
            return [0, 0, 0]

        if action == LEVEL_INDEX:
            return [1, 0, 0]

        if action == REFRESH_INDEX:
            return [2, 0, 0]

        raise ValueError(f"Action index is outside the vector space: {action}")

    def fetch_action_mask(self):
        mask = np.zeros(ACTION_DIM, dtype=np.int8)

        mask[BENCH_TO_BOARD_START:BOARD_TO_BOARD_START] = (
            np.asarray(self.move_sell_bench_mask)[:, :N_BOARD].reshape(-1) > 0
        )

        board_dests = np.reshape(self.move_sell_board_mask, (N_BOARD, -1))[:, :N_BOARD]
        mask[BOARD_TO_BOARD_START:ITEM_TO_BOARD_START] = (
            board_dests[~np.eye(N_BOARD, dtype=bool)] > 0
        )

        mask[ITEM_TO_BOARD_START:SHOP_START] = (
            np.asarray(self.item_mask)[:, :N_BOARD].reshape(-1) > 0
        )
        mask[SHOP_START:PASS_INDEX] = np.asarray(self.buy_mask) > 0
        mask[PASS_INDEX] = 1
        mask[LEVEL_INDEX] = int(self.exp_mask)
        mask[REFRESH_INDEX] = int(self.refresh_mask)
        return mask
