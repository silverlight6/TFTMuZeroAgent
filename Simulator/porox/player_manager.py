from Simulator import pool
from Simulator.porox.player import Player as player_class
from Simulator.porox.observation import Observation


class PlayerManager:
    """Manages player states, observations, and actions."""

    def __init__(self, num_players, pool_obj):
        self.pool_obj = pool_obj

        self.players = {
            "player_" + str(player_id) for player_id in range(num_players)
        }
        self.terminations = {player: False for player in self.players}

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
        
    def kill_player(self, player_id):
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
            for player in self.players
            if player != player_id
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
            "action_mask": self.observation_states[player_id].fetch_action_mask_v2(),
            "opponents": self.opponent_observations[player_id],
        }
        
    def fetch_observations(self):
        """Creates the observation for every player.
        """
        return {player_id: self.fetch_observation(player_id) for player_id, alive in self.terminations.items() if alive}
    
    def update_game_round(self):
        """Update observations after a battle
        
        After a battle, your hp, gold, exp, and shop changes
        `refresh_all_shops` takes care of the shop change
        """
        for player in self.players:
            if not self.terminations[player]:
                self.observation_states[player].update_game_round()


    def refresh_all_shops(self):
        for player in self.players:
            if not self.terminations[player]:
                self.player_states[player].refresh_shop()
                self.observation_states[player].update_refresh_action([2, 0, 0])

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
        if type(action) is not list:
            action = action_space_to_action_v2(action)

        if len(action) != 3:
            print(f"Action is not of length 3: {action}")
            return

        action_type, x1, x2 = action

        player = self.player_states[player_id]
        observation = self.observation_states[player_id]

        # Pass Action
        if action_type == 0:
            player.pass_action()
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
            observation.update_move_sell_action(action)

        # Move/Sell Action
        elif action_type == 5:
            player.move_champ_action(x1, x2)
            observation.update_move_sell_action(action)

        # Item action
        elif action_type == 6:
            player.move_item_action(x1, x2)
            observation.update_item_action(action)

        else:
            player.print(f"Action Type is invalid: {action}")

def action_space_to_action_v1(action):
    """Converts an action sampled from the action space to an action that can be performed.
    
    Action Space: (5, 11, 38)

                11
        |P|L|R|B|B|B|B|B|B|B|S|
        |b|b|b|B|B|B|B|B|B|B|S|
    5  |b|b|b|B|B|B|B|B|B|B|S| x 38
        |b|b|b|B|B|B|B|B|B|B|S|
        |I|I|I|I|I|I|I|I|I|I|S|

    TODO: Make this make sense LMAO

                11
        |P |L |R |3 |7 |11|15|19|23|27|0|
        |28|29|30|2 |6 |10|14|18|22|26|1|
    5  |31|32|33|1 |5 |9 |13|17|21|25|2| x 38
        |34|35|36|0 |4 |8 |12|16|20|24|3|
        |0 |1 |2 |3 |4 |5 |6 |7 |8 |9 |4|
        
    0-27 -> Board Slots
    28-36 -> Bench Slots
    37 -> Sell Slot

    Args:
        action int: Action index from the action space.
        
    Returns:
        list: Action that can be performed. [action_type, x1, x2]
    """
    row, col, index = action // 38 // 11, action // 38 % 11, action % 38
    
    action = []
    
    # Pass Action
    if row == 0 and col == 0:
        action = [0, 0, 0]
    # Level Action
    elif row == 0 and col == 1:
        action = [1, 0, 0]
    # Refresh Action
    elif row == 0 and col == 2:
        action = [2, 0, 0]
    
    # Board Slot
    elif row < 4 and (col >= 3 and col <= 9):
        pos = (col-3, 3-row)
        from_loc = pos[0] * 4 + pos[1]
        
        if index == 37:
            action = [4, from_loc, 0]
        else:
            to_loc = index
            action = [5, from_loc, to_loc]
            
    # Bench Slot
    elif (row >= 1 and row <= 3) and (col >= 0 and col <= 2):
        pos = (col, row-1)
        from_loc = pos[0] * 3 + pos[1] + 28
        
        if index == 37:
            action = [4, from_loc, 0]
        else:
            to_loc = index
            action = [5, from_loc, to_loc]
            
    # Buy Slot
    elif col == 10:
        action = [3, row, 0]
    
    # Item Slot
    elif row == 4 and col < 10:
        from_loc = col
        to_loc = index
        action = [6, from_loc, to_loc]

    return action
    
def action_space_to_action_v2(action):
    """Converts an action sampled from the action space to an action that can be performed.
    
    Action Space: (55, 38)
    55:
    |0|1|2|3|4|5|6|7|8|9|10|11|12|13|...|27| (Board Slots)
    |28|29|30|31|32|33|34|35|36| (Bench Slots)
    |37|38|39|40|41|42|43|44|45|46| (Item Bench Slots)
    |47|48|49|50|51| (Shop Slots)
    |52| (Pass)
    |53| (Level)
    |54| (Refresh)
    
    38:
    0-27 -> Board Slots
    28-36 -> Bench Slots
    37 -> Sell Slot
    
    """
    
    col, index = action // 38, action % 38
    
    action = []

    # Board and Bench slots
    if col < 37:
        from_loc = col

        if index == 37:
            action = [4, from_loc, 0]
        else:
            to_loc = index
            action = [5, from_loc, to_loc]
            
    # Item Bench Slots
    elif col < 47:
        from_loc = col - 37
        to_loc = index
        action = [6, from_loc, to_loc]
        
    # Shop Slots
    elif col < 52:
        action = [3, col - 47, 0]
        
    # Pass Action
    elif col == 52:
        action = [0, 0, 0]
    
    # Level Action
    elif col == 53:
        action = [1, 0, 0]
    
    # Refresh Action
    elif col == 54:
        action = [2, 0, 0]
        
    return action