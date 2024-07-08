import numpy as np
import config
from Simulator import utils

class RandomAgent:
    def __init__(self, action_size, action_limits, global_buffer):
        self.action_size = action_size
        self.action_limits = action_limits
        self.global_buffer = global_buffer

    def select_action(self, observation, mask, reward, terminated):
        return np.random.randint(self.action_limits, size=(observation.shape[0], self.action_size))
class BuyingAgent:

    def __init__(self, action_size, action_limits, units_to_buy, global_buffer):
        self.action_size = action_size
        self.units_to_buy = units_to_buy
        self.action_limits = action_limits
        self.global_buffer = global_buffer

    def select_action(self, observation, mask, reward, terminated):
        actions = []
        for i in range(observation.shape[0]):
            actions.append(self.decide_action(observation[i]))
        return actions
        
    def decide_action(self, observation):
        gold = utils.gold_from_obs(observation)
        # print(gold)
        units_in_shop, chosen = utils.units_in_shop_from_obs(observation)
        # print("SHOP", units_in_shop, chosen)
        for champ in units_in_shop:
            if champ in self.units_to_buy:
                action = np.array([2, utils.champ_id_from_name(champ), 0])
                # print(action)
                return action
        board = utils.board_from_obs(observation)
        # print(board)
        for champ in board:
            if champ["name"] not in self.units_to_buy:
                action = np.array([3, utils.x_y_to_1d_coord(champ["pos_x"], champ["pos_y"]), 0])
                # print(action)
                return action
        level = utils.level_from_obs(observation)
        if (level >= 8.0 or len(board) < level) and gold > 52.0:
            action = np.array([4, 0, 0])
            # print("REFRESHING")
            return action
        if gold > 54.0 and level < 8.0:
            action = np.array([5, 0, 0])
            # print("Level up")
            return action
        bench = utils.bench_from_obs(observation)
        # sell units on bench not on directive
        for i, champ in enumerate(bench):
            if champ not in self.units_to_buy:
                return np.array([3,28+i,0])
        return np.array([0,0,0])

class CultistAgent(BuyingAgent):
    pass

class DivineAgent(BuyingAgent):
    pass
