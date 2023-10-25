import numpy as np

import Simulator.config as config
import Simulator.champion as champion
from Simulator.porox.observation.observation_helper import ObservationManager

"""
Description - Object used for the simulation to interact with the environment. The agent passes in actions and those 
              actions take effect in this object.
Inputs      - pool_obj: Object pointer to the pool
                Pool object pointer used for refreshing shops and generating shop vectors.
              observation_obj: Object pointer to a Game_observation object
                Used for generating shop_vector and other player vectors on pass options.
"""


class Step_Function:
    def __init__(self, pool_obj, observation_objs):
        self.pool_obj = pool_obj
        self.observation_objs = observation_objs

    def perform_action(self, action, player):
        """Performs the action given by the agent.

        7 Types of actions:
        [0, 0, 0] - Pass action
        [1, 0, 0] - Level up action
        [2, 0, 0] - Refresh action
        [3, X1, 0] - Buy action; X1 is an index from 0 to 4 for the shop locations
        [4, X1, 0] - Sell Action; X1 is the index of the champion to sell (0 to 36)
        [5, X1, X2] - Move Action; X1 is the index of the champion to move (0 to 36), X2 is the index of the location to move to (0 to 36)
        [6, X1, X2] - Item Action; X1 is the index of the item to move (0 to 9), X2 is the index of the champion to move to (0 to 36)

        Args:
            action (list): Action to perform. Must be of length 3.

        Returns:
            bool: True if action was performed successfully, False otherwise.
        """
        if len(action) != 3:
            print(f"Action is not of length 3: {action}")
            return
        action_type, x1, x2 = action

        # Pass Action
        if action_type == 0:
            player.pass_action()

        # Level Action
        elif action_type == 1:
            player.buy_exp_action()

        # Refresh Action
        elif action_type == 2:
            player.refresh_shop_action()

        # Buy Action
        elif action_type == 3:
            player.buy_shop_action(x1)

        # Sell Action
        elif action_type == 4:
            player.sell_action(x1)

        # Move/Sell Action
        elif action_type == 5:
            player.move_champ_action(x1, x2)

        # Item action
        elif action_type == 6:
            player.move_item_action(x1, x2)

        else:
            player.print(f"Action Type is invalid: {action}")
            player.pass_action()
