import numpy as np
import config
from Simulator import utils

class RandomAgent:
    def __init__(self, action_size):
        self.action_size = action_size

    def select_action(self, observation, mask):
        return np.random.randint([6, 37, 28], size=(observation.shape[0], self.action_size))
    
class BuyingAgent:

    def __init__(self, action_size, units_to_buy):
        self.action_size = action_size
        self.units_to_buy = units_to_buy

    def select_action(self, observation, mask):
        gold = utils.gold_from_obs(observation[0])
        # print(gold)
        units_in_shop, chosen = utils.units_in_shop_from_obs(observation[0])
        print("SHOP", units_in_shop, chosen)
        for champ in units_in_shop:
            if champ in self.units_to_buy:
                action = np.array([[2, utils.champ_id_from_name(champ), 0]])
                return action
        board = utils.board_from_obs(observation[0])
        # print(board)
        for champ in board:
            if champ["name"] not in self.units_to_buy:
                action = np.array([[3, utils.x_y_to_1d_coord(champ["pos_x"], champ["pos_y"]), 0]])
                return action
        level = utils.level_from_obs(observation[0])
        if gold > 54.0 and level < 8.0:
            action = np.array([[5, 0, 0]])
            return action
        if level >= 8.0 and gold > 50.0:
            action = np.array([[4, 0, 0]])
            return action
        # print(board)
        bench = utils.bench_from_obs(observation[0])
        # print("COMMON BENCH", bench)
        # return np.random.randint([6, 37, 28], size=(observation.shape[0], self.action_size))
        return np.array([[0,0,0]])
