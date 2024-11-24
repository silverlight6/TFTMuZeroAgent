import config
from Simulator.player import Player as player_class

class PlayerManager:
    def __init__(self, 
                 num_players,
                 pool_obj,
                 tft_config
                 ):
        
        self.pool_obj = pool_obj
        self.config = tft_config
        
        self.players = {
            "player_" + str(player_id) for player_id in range(num_players)
        }
        
        # Ensure that the opponent obs are always in the same order
        self.player_ids = sorted(list(self.players))

        self.terminations = {player: False for player in self.players}

        self.player_states = {
            player: player_class(self.pool_obj, player_id)
            for player_id, player in enumerate(self.players)
        }

        self.observation_states = {
            player: tft_config.observation_class(self.player_states[player])
            for player in self.players
        }
        
        self.opponent_observations = {
            player: self.fetch_opponent_observations(player)
            for player in self.players
        }
        
        self.action_handlers = {
            player: tft_config.action_class(self.player_states[player])
            for player in self.players
        }

    def kill_player(self, player_id):
        """TODO: Change how this works... it is like this to be compatible with the old sim"""
        self.terminations[player_id] = True
        
        player = self.player_states[player_id]
        self.pool_obj.return_hero(player)

        self.player_states[player_id] = None

    def fetch_opponent_observations(self, player_id):
        """Fetches the opponent observations for the given player.

        Args:
            player_id (int): Player id to fetch opponent observations for.

        Returns:
            list: List of observations for the given player.
        """
        observations = [
            self.observation_states[player].fetch_public_observation()
            if not self.terminations[player]
            else self.observation_states[player].fetch_dead_observation()
            for player in self.player_ids if player != player_id
        ]

        return observations

    def fetch_opponent_position_observations(self, player_id):
        """Fetches the opponent observations for the given player.

        Args:
            player_id (int): Player id to fetch opponent observations for.

        Returns:
            list: List of observations for the given player.
        """
        observations = [
            self.observation_states[player].fetch_public_position_observation()
            if not self.terminations[player]
            else self.observation_states[player].fetch_dead_position_observation()
            for player in self.player_ids if player != player_id
        ]

        return observations

    def fetch_observation(self, player_id):
        """Creates the observation for the given player.

        Format:
        {
            "player": PlayerObservation
            "action_mask": (5, 11, 38)  # Same as action space
            "opponents": [PlayerPublicObservation, ...]
        }
        """

        return {
            "player": self.observation_states[player_id].fetch_player_observation(),
            "action_mask": self.action_handlers[player_id].fetch_action_mask(),
            "opponents": self.opponent_observations[player_id],
        }

    def fetch_position_observation(self, player_id):
        """Creates the observation for the given player.

        Format:
        {
            "player": PlayerObservation
            "action_mask": (5, 11, 38)  # Same as action space
            "opponents": [PlayerPublicObservation, ...]
        }
        """

        return {
            "player": self.observation_states[player_id].fetch_player_position_observation(),
            "opponents": self.fetch_opponent_position_observations(player_id),
            "action_mask": self.action_handlers[player_id].fetch_action_mask(),
        }

    def fetch_observations(self):
        """Creates the observation for every player."""
        return {player_id: self.fetch_observation(player_id) for player_id, alive in self.terminations.items() if alive}
    
    def update_game_round(self):
        for player in self.players:
            if not self.terminations[player]:
                self.player_states[player].actions_remaining = self.config.max_actions_per_round
                self.observation_states[player].update_game_round()
                self.action_handlers[player].update_game_round()
                
                # This is only here so that the Encoder doesn't have to do 64 operations, but instead 16
                # This ensures that the masked player states are the same for all players
                # 8 full players and 8 masked players
                # If we had separate masked players for each player, then we would have to do 64 operations
                self.opponent_observations[player] = self.fetch_opponent_observations(player)

    def refresh_all_shops(self):
        for player in self.players:
            if not self.terminations[player]:
                self.player_states[player].refresh_shop()
                self.observation_states[player].update_observation([2, 0, 0])
                self.action_handlers[player].update_action_mask([2, 0, 0])

    def refresh_player_shop(self, player):
        self.player_states[player].refresh_shop()
        self.observation_states[player].update_observation([2, 0, 0])
        self.action_handlers[player].update_action_mask([2, 0, 0])

    # - Used so I don't have to change the Game_Round class -
    def generate_shops(self, players):
        self.refresh_all_shops()

    def generate_shop_vectors(self, players):
        pass

    def reinit_player_set(self, new_player_set):
        self.players = {
            "player_" + str(player_id) for player_id in range(config.NUM_PLAYERS)
        }

        # Ensure that the opponent obs are always in the same order
        self.player_ids = sorted(list(self.players))

        self.terminations = {player: False for player in self.players}

        self.player_states = {
            "player_" + str(player.player_num): player
            for player in new_player_set
        }

        for x in range(config.NUM_PLAYERS):
            if f"player_{x}" not in self.player_states.keys():
                self.player_states[f"player_{x}"] = player_class(self.pool_obj, x)

        self.observation_states = {
            player: self.config.observation_class(self.player_states[player])
            for player in self.players
        }

        self.opponent_observations = {
            player: self.fetch_opponent_observations(player)
            for player in self.players
        }

        self.action_handlers = {
            player: self.config.action_class(self.player_states[player])
            for player in self.players
        }
    # -----
