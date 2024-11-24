import time

from Simulator.utils import coord_to_x_y, x_y_to_1d_coord

class Step_Function:
    def __init__(self, player_manager):
        self.player_manager = player_manager
        self.position_list = []

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
            player.move_item_action(x2, x1)

        else:
            player.print(f"Action Type is invalid: {action}")

        player.actions_remaining -= 1
        self.player_manager.observation_states[player_id].update_observation(action)
        self.player_manager.action_handlers[player_id].update_action_mask(action)

    def perform_1d_action(self,  player_id, action):
        action = self.player_manager.action_handlers[player_id].action_space_to_action(action)
        self.perform_action(player_id, action)

    def position_controller(self, action, player):
        """
        Takes an action which is 12 by 29. If 28, the action is considered a pass.
        All final positions are calculated prior to moving. If multiple movements to the same square are requested,
        only the first one will be recognized.
        If two units are swapped by an early movement, then the movement for the first character will still be the one
        to move by the future command.
        All actions where the model requests the unit to move to the square where it already is will be treated
        the same as 28 or pass. This means if there is a swap to the square early, the unit will be moved.
                                        Top
                | (0, 3) (1, 3) (2, 3) (3, 3) (4, 3) (5, 3) (6, 3) |
          Left  | (0, 2) (1, 2) (2, 2) (3, 2) (4, 2) (5, 2) (6, 2) |
                | (0, 1) (1, 1) (2, 1) (3, 1) (4, 1) (5, 1) (6, 1) |  Right
                | (0, 0) (1, 0) (2, 0) (3, 0) (4, 0) (5, 0) (6, 0) |
                                        Bottom

                                        Top
                | (0)  (1)  (2)  (3)  (4)  (5)  (6)  |
          Left  | (7)  (8)  (9)  (10) (11) (12) (13) |
                | (14) (15) (16) (17) (18) (19) (20) |  Right
                | (21) (22) (23) (24) (25) (26) (27) |
                                        Bottom
        """
        # print(f"action {action} for player {player.player_num}")
        # Remove all duplicate actions, keep the early ones.
        destination = []
        for x in action:
            if x not in destination:
                destination.append(x)
            else:
                destination.append(28)

        destination_coords = []
        for x in destination:
            x1, y1 = coord_to_x_y(int(x))
            destination_coords.append([x1, y1])

        for i, coord in enumerate(destination_coords):
            # 28 pass rule.
            if coord[0] != 7:
                temp_square = self.position_list[i]
                player.move_board_to_board(temp_square[0], temp_square[1], coord[0], coord[1])
                # Make note that this square was already used
                self.position_list[i] = [-1, -1]
                # Adjust starting points to keep track of where units are moving.
                for j, square in enumerate(self.position_list):
                    if coord == square:
                        self.position_list[j] = temp_square

    def multi_step_position_controller(self, action, step):
        if action != 28 and step < len(self.position_list):
            temp_square = self.position_list[step]
            temp_coord = x_y_to_1d_coord(temp_square[0], temp_square[1])
            self.perform_action("player_0", [5, temp_coord, action])

    def fake_multi_step_position_controller(self, action, player, step):
        x, y = coord_to_x_y(action)
        coord = [x, y]
        if coord[0] != 7 and step < len(self.position_list):
            temp_square = self.position_list[step]
            player.move_board_to_board(temp_square[0], temp_square[1], coord[0], coord[1])

    def item_controller(self, action, player, item_guide):
        """
        Takes an action which is 10 by 28. If 28, the action is considered a pass.
        All items will be placed according to the commands. If the item command is not possible, nothing will happen.
                                        Top
                | (0, 3) (1, 3) (2, 3) (3, 3) (4, 3) (5, 3) (6, 3) |
          Left  | (0, 2) (1, 2) (2, 2) (3, 2) (4, 2) (5, 2) (6, 2) |
                | (0, 1) (1, 1) (2, 1) (3, 1) (4, 1) (5, 1) (6, 1) |  Right
                | (0, 0) (1, 0) (2, 0) (3, 0) (4, 0) (5, 0) (6, 0) |
                                        Bottom

                                        Top
                | (0)  (1)  (2)  (3)  (4)  (5)  (6)  |
          Left  | (7)  (8)  (9)  (10) (11) (12) (13) |
                | (14) (15) (16) (17) (18) (19) (20) |  Right
                | (21) (22) (23) (24) (25) (26) (27) |
                                        Bottom
        """
        destination_coords = []
        for x in action:
            x1, y1 = coord_to_x_y(int(x))
            destination_coords.append([x1, y1])

        for i, use_item in enumerate(item_guide):
            if not use_item[0] or player.item_bench[i] is None:
                destination_coords[i] = [7, 0]

        for i, coord in enumerate(destination_coords):
            # 28 pass rule.
            if coord[0] != 7:
                if coord in self.position_list:
                    player.move_item(i, coord[0], coord[1])

    def create_unit_list(self, player):
        self.position_list = []
        for coord in range(len(player.board) * len(player.board[0])):
            x, y = coord_to_x_y(coord)
            if player.board[x][y]:
                self.position_list.append([x, y])
