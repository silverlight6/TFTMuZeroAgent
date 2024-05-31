import numpy as np
import config
import random
from Simulator.stats import BASE_CHAMPION_LIST
from Simulator.pool import pool
from Simulator.player import Player
from Simulator.utils import coord_to_x_y, x_y_to_1d_coord
from Simulator.item_stats import item_builds
from Simulator.champion import champion
from Simulator.observation.token.action import ActionToken


"""
Description - This is the Battle Generator class
              Generates random battles with random items and random number of players with random positioning
"""
class BattleGenerator:
    def __init__(self):
        ...

    def generate_battle(self, epoch=0):
        level = np.random.randint(3, 5 + min(int(epoch / 1000), 4))
        base_pool = pool()
        player_list = [Player(base_pool, player_num) for player_num in range(config.NUM_PLAYERS)]
        for player in player_list:
            player.level = level
            player.max_units = level
            action_mask = ActionToken(player)
            self.add_champions(player, action_mask, base_pool)
            # Add these back in later after I see proof of learning
            # self.add_items_to_champions(player, action_mask)
            # self.add_items_to_item_bench(player)
        player_returns = random.sample(range(0, 8), 2)
        return [player_list[player_returns[0]], player_list[player_returns[1]],
                {f"player_{player.player_num}": player for player in player_list}]

    def add_champions(self, player, action_mask, base_pool):
        move_failure = 0
        for _ in range(player.max_units):
            random_champ = base_pool.sample(player, 1, allow_chosen=False)
            player.add_to_bench(champion(random_champ[0]))
            _, bench_mask = action_mask.create_move_and_sell_action_mask(player)
            coord = np.random.randint(0, 28)
            coord_x, coord_y = coord_to_x_y(coord)
            if bench_mask[move_failure][coord]:
                player.move_bench_to_board(move_failure, coord_x, coord_y)
            else:
                move_failure += 1

    def add_items_to_champions(self, player, action_mask):
        move_failures = 0
        for x in range(len(player.board)):
            for y in range(len(player.board[0])):
                if player.board[x][y]:
                    random_item = random.sample(list(item_builds.keys()), 1)
                    player.add_to_item_bench(random_item[0])
                    item_mask = action_mask.create_item_action_mask(player)
                    if item_mask[move_failures][x_y_to_1d_coord(x, y)]:
                        player.move_item(move_failures, x, y)
                    else:
                        move_failures += 1

    def add_items_to_item_bench(self, player):
        for _ in range(8):
            if not player.item_bench_full(1):
                random_item = random.sample(list(item_builds.keys()), 1)
                player.add_to_item_bench(random_item[0])
