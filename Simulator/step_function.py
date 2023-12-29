class Step_Function:
    def __init__(self, player_manager):
        self.player_manager = player_manager

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
            player_id : string id for the player
            action (list): Action to perform. Must be of length 3.

        Returns:
            bool: True if action was performed successfully, False otherwise.
        """

        if type(action) is not list and len(action) != 3:
            print(f"Action is not a list of length 3: {action}")
            return

        action_type, x1, x2 = action

        player = self.player_manager.player_states[player_id]

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
        self.player_manager.observation_states[player_id].update_observation(action)
        self.player_manager.action_handlers[player_id].update_action_mask(action)

    def perform_1d_action(self,  player_id, action):
        action = self.player_manager.action_handlers[player_id].action_space_to_action(action)
        self.perform_action(player_id, action)
