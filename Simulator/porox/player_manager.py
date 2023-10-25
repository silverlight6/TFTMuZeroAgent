from Simulator import pool
from Simulator.game_round import Game_Round

from Simulator.porox.player import Player as player_class
from Simulator.porox.observation import Observation


class PlayerManager:
    """Manages player states, observations, and actions."""

    def __init__(self, num_players, pool_obj):
        self.pool_obj = pool_obj

        self.players = {
            "player_" + str(player_id) for player_id in range(num_players)
        }

        self.player_states = {
            player: player_class(self.pool_obj, player_id)
            for player_id, player in enumerate(self.players)
        }

        self.observation_states = {
            player: Observation(self.player_states[player])
            for player in self.players
        }

        self.opponent_observations = {
            player: self.fetch_opponent_observations(player)
            for player in self.players
        }

    def fetch_opponent_observations(self, player_id):
        """Fetches the opponent observations for the given player.

        Args:
            player_id (int): Player id to fetch opponent observations for.

        Returns:
            list: List of observations for the given player.
        """
        observations = [
            self.observations_states[player].fetch_public_observation()
            for player in self.players
            if player != player_id
        ]

        return observations

    def fetch_observation(self, player_id):
        """Creates the observation for the given player.

        Format:
        {
            "player": PlayerObservation
            "mask": (5, 11, 38)  # Same as action space
            "opponents": [PlayerPublicObservation, ...]
        }
        """

        return {
            "player": self.observation_states[player_id].fetch_player_observation(),
            "mask": self.observation_states[player_id].fetch_action_mask(),
            "opponents": self.opponent_observations[player_id],
        }

    def perform_action(self, player_id, action):
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

        player = self.player_states[player_id]
        observation = self.observation_states[player_id]

        # Pass Action
        if action_type == 0:
            # Update opponent observations on pass action
            self.opponent_observations[player_id] = self.fetch_opponent_observations(
                player_id
            )

        # Level Action
        elif action_type == 1:
            player.buy_exp_action()
            observation.update_exp_action(action)

        # Refresh Action
        elif action_type == 2:
            player.refresh_shop_action()
            observation.update_refresh_action(action)

        # Buy Action
        elif action_type == 3:
            player.buy_shop_action(x1)
            observation.update_buy_action(action)

        # Sell Action
        elif action_type == 4:
            player.sell_action(x1)
            observation.update_sell_action(action)

        # Move/Sell Action
        elif action_type == 5:
            player.move_champ_action(x1, x2)
            observation.update_move_action(action)

        # Item action
        elif action_type == 6:
            player.move_item_action(x1, x2)
            observation.update_item_action(action)

        else:
            player.print(f"Action Type is invalid: {action}")
