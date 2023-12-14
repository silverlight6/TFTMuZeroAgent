from Simulator.porox.player import Player as player_class

class PlayerManager:
    def __init__(self, 
                 num_players,
                 pool_obj,
                 config
                 ):
        
        self.pool_obj = pool_obj
        self.config = config
        
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
            player: config.observation_class(self.player_states[player])
            for player in self.players
        }
        
        self.opponent_observations = {
            player: self.fetch_opponent_observations(player)
            for player in self.players
        }
        
        self.action_handlers = {
            player: config.action_class(self.player_states[player])
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
            for player in self.player_ids
            
            # TODO: make this an option
            # if player != player_id
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

    # - Used so I don't have to change the Game_Round class -
    # TODO: Refactor Game_Round class to use refresh_all_shops
    def generate_shops(self, players):
        self.refresh_all_shops()

    def generate_shop_vectors(self, players):
        pass
    # -----
        
    # --- Main Action Function ---
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
        action = self.action_handlers[player_id].action_space_to_action(action)

        if type(action) is not list and len(action) != 3:
            print(f"Action is not a list of length 3: {action}")
            return

        action_type, x1, x2 = action

        player = self.player_states[player_id]

        # Pass Action
        if action_type == 0:
            player.pass_action()
            # Update opponent observations on pass action
            # self.opponent_observations[player_id] = self.fetch_opponent_observations(
            #     player_id
            # )
            # Lets only update opponent obs at the beginning of the round
            # This way I can use an optimized encoder that does 16 operations instead of 64

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
            
        player.actions_remaining -= 1
        self.observation_states[player_id].update_observation(action)
        self.action_handlers[player_id].update_action_mask(action)