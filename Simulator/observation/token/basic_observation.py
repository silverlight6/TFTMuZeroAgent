import numpy as np
import config
import time

from Simulator.observation.util import Util
from Simulator.observation.normalization import Normalizer, safe_normalize
from Simulator.observation.interface import ObservationBase, ObservationUpdateBase
from Simulator.item_stats import items
from Simulator.stats import COST
from Simulator.origin_class_stats import tiers, origin_class
from Simulator.utils import x_y_to_1d_coord


class ObservationToken(ObservationBase, ObservationUpdateBase):
    """Observation object that stores the observation for a player."""

    def __init__(self, player):
        super().__init__()

        self.player = player

        self.util = Util()
        self.normalizer = Normalizer()

        # -- Items -- #
        self.item_vector_length = 1

        # -- Traits -- #
        self.trait_champion_vector_length = 7
        self.trait_player_vector_length = len(self.util.trait_ids)

        # -- Champions -- #
        # Create ids for champions
        self.stat_vector_length = 12
        # 2 for chosen, 4 for stars, 1 for cost
        self.other_stat_vector_length = 2 + 4 + 1

        # championID, items, origins, stats
        self.champion_vector_length = 5

        # --- Game Values --- #
        self.game_scalars = self.create_game_scalars(player)

        # --- Public Scalars --- #
        self.public_scalars = self.create_public_scalars(player)

        # --- Private Scalars --- #
        self.private_scalars = self.create_private_scalars(player)

        # --- Embedding Scalars --- #
        self.embedding_scalars = self.create_embedding_scalars(player)

        # --- Public Vectors --- #
        self.board_vector = self.create_board_vector(player)
        self.bench_vector = self.create_bench_vector(player)
        self.item_bench_vector = self.create_item_bench_vector(player)
        self.trait_vector = self.create_trait_vector(player)

        # --- Private Vectors --- #
        self.shop_vector = self.create_shop_vector(player)

    def fetch_player_observation(self):
        """Fetch the PlayerObservation for a player.

        PlayerObservation:
            scalars: [Game Values, Public Scalars, Private Scalars]
            board: board token
            bench: bench token
            shop: shop token
            items: item token
            traits: trait token
        """

        return {
            "scalars": np.concatenate([
                self.game_scalars,
                self.public_scalars,
                self.private_scalars,
            ]),
            "emb_scalars": self.embedding_scalars,
            "board": self.board_vector,
            "bench": self.bench_vector,
            "shop": self.shop_vector,
            "items": self.item_bench_vector,
            "traits": self.trait_vector,
        }

    def fetch_player_position_observation(self):
        """Fetch the PlayerObservation for a player.

        PlayerObservation:
            board: board token
            traits: trait token
        """
        return {
            "board": self.board_vector,
            "traits": self.trait_vector,
        }

    def fetch_public_observation(self):
        """Fetch the PlayerPublicObservation for a player.

        PlayerPublicObservation:
            scalars: [Public Scalars]
            board: board token
            bench: bench token
            items: item token
            traits: trait token
        """
        return {
            "scalars": self.public_scalars,
            "board": self.board_vector,
            "bench": self.bench_vector,
            "items": self.item_bench_vector,
            "traits": self.trait_vector,
        }

    def fetch_public_position_observation(self):
        """Fetch the PlayerPublicObservation for a player.

                PlayerPublicObservation:
                    board: board token
                    traits: trait token

                Commenting out the bench and items to save on storage space. Can comment back in a future iteration
        """
        return {
            "board": self.board_vector,
            "traits": self.trait_vector,
        }

    def fetch_dead_observation(self):
        """Zero out public observations for all dead players"""
        return {
            "scalars": np.zeros(self.public_scalars.shape, dtype=np.float32),
            "board": np.zeros(self.board_vector.shape, dtype=np.float32),
            "bench": np.zeros(self.bench_vector.shape, dtype=np.float32),
            "shop": np.zeros(self.shop_vector.shape, dtype=np.float32),
            "items": np.zeros(self.item_bench_vector.shape, dtype=np.float32),
            "traits": np.zeros(self.trait_vector.shape, dtype=np.float32),
        }

    def fetch_dead_position_observation(self):
        """Zero out public observations for all dead players"""
        return {
            "board": np.zeros(self.board_vector.shape, dtype=np.float32),
            "traits": np.zeros(self.trait_vector.shape, dtype=np.float32),
        }

    def update_observation(self, action):
        action_type, x1, x2 = action

        # Pass Action
        if action_type == 0:
            ...

        # Level Action
        elif action_type == 1:
            self.update_exp_action(action)

        # Refresh Action
        elif action_type == 2:
            self.update_refresh_action(action)

        # Buy Action
        elif action_type == 3:
            self.update_buy_action(action)

        # Sell Action
        elif action_type == 4:
            self.update_move_sell_action(action)

        # Move/Sell Action
        elif action_type == 5:
            self.update_move_sell_action(action)

        # Item action
        elif action_type == 6:
            self.update_item_action(action)

        else:
            print(f"Action Type is invalid: {action}")

    def update_game_round(self):
        """Update vectors and masks after game round.

        After a battle, your hp, gold, exp, and shop changes
        `refresh_all_shops` calls `update_refresh_action`, so no use calling those twice
        Round might be a carousel/minion round, so just update everything just in case
        """
        self.game_scalars = self.create_game_scalars(self.player)
        self.public_scalars = self.create_public_scalars(self.player)
        self.private_scalars = self.create_private_scalars(self.player)
        self.embedding_scalars = self.create_embedding_scalars(self.player)

        self.board_vector = self.create_board_vector(self.player)
        self.bench_vector = self.create_bench_vector(self.player)
        self.item_bench_vector = self.create_item_bench_vector(self.player)
        self.trait_vector = self.create_trait_vector(self.player)

    def update_exp_action(self, action):
        """Update vectors and masks related to the exp action.

        Updated Values:
            - Gold, Exp, Level
            - If level up, then bench masks need to be updated

        """
        self.game_scalars = self.create_game_scalars(self.player)
        self.public_scalars = self.create_public_scalars(self.player)
        self.private_scalars = self.create_private_scalars(self.player)
        self.embedding_scalars = self.create_embedding_scalars(self.player)

    def update_refresh_action(self, action):
        """Update vectors and masks related to the refresh action.

        Updated Values:
            - Shop
        """
        self.game_scalars = self.create_game_scalars(self.player)
        self.public_scalars = self.create_public_scalars(self.player)
        self.private_scalars = self.create_private_scalars(self.player)
        self.embedding_scalars = self.create_embedding_scalars(self.player)

        self.shop_vector = self.create_shop_vector(self.player)

    def update_buy_action(self, action):
        """Update vectors and masks related to the buy action.

        Updated Values:
            - Shop, Bench, Gold
        """
        self.game_scalars = self.create_game_scalars(self.player)
        self.public_scalars = self.create_public_scalars(self.player)
        self.private_scalars = self.create_private_scalars(self.player)
        self.embedding_scalars = self.create_embedding_scalars(self.player)

        self.board_vector = self.create_board_vector(self.player)
        self.bench_vector = self.create_bench_vector(self.player)
        self.shop_vector = self.create_shop_vector(self.player)
        self.trait_vector = self.create_trait_vector(self.player)

    def update_move_sell_action(self, action):
        """Update vectors and masks related to the move sell board/bench action.

        Updated Values:
            - Board, Bench, Gold
        """
        self.game_scalars = self.create_game_scalars(self.player)
        self.private_scalars = self.create_private_scalars(self.player)
        self.public_scalars = self.create_public_scalars(self.player)
        self.embedding_scalars = self.create_embedding_scalars(self.player)

        self.board_vector = self.create_board_vector(self.player)
        self.bench_vector = self.create_bench_vector(self.player)
        self.trait_vector = self.create_trait_vector(self.player)
        self.item_bench_vector = self.create_item_bench_vector(self.player)

    def update_item_action(self, action):
        """Update vectors and masks related to the item action.

        Updated Values:
            - Bench, Item Bench, Gold
        """
        self.game_scalars = self.create_game_scalars(self.player)
        self.private_scalars = self.create_private_scalars(self.player)
        self.public_scalars = self.create_public_scalars(self.player)
        self.embedding_scalars = self.create_embedding_scalars(self.player)

        self.board_vector = self.create_board_vector(self.player)
        self.bench_vector = self.create_bench_vector(self.player)
        self.trait_vector = self.create_trait_vector(self.player)
        self.item_bench_vector = self.create_item_bench_vector(self.player)

    # --- Implementation of ObservationVectorBase --- #
    def create_game_scalars(self, player):
        """Create game scalars for a player

        Game Scalars:
            - Round: int
            - Actions remaining: int
            - opponents: list of ints
            - Action History?: [[int, int, int], [int, int, int], ...]  # TODO
        """

        opponentIDs = np.zeros(3)
        count = 0
        for playerID, will_fight in player.opponent_options.items():
            if will_fight == 1 and count < 3:
                playerID = int(playerID[-1])
                opponentIDs[count] = playerID
                count += 1

        return np.array([
            player.round / 30,
            player.actions_remaining / 30,
            *opponentIDs,
            # TODO: Action History
        ])

    # TODO: Turn health and gold into embeddable numbers.
    def create_public_scalars(self, player):
        """Create public scalars for a player

        Public Scalars:
            - health: float
            - level: float
            - win streak: float
            - loss streak: float
            - max units: float
            - available units: float
            - economy: float
        """

        streak_lvl = 0
        if player.win_streak == 4:
            streak_lvl = 0.5
        elif player.win_streak >= 5:
            streak_lvl = 1

        return np.array([
            player.health / 100,
            player.level / 10,
            player.max_units / 10,
            (player.max_units - player.num_units_in_play) / 10,
            player.win_streak / 20,
            player.loss_streak / 20,
            streak_lvl,
            min(player.gold // 10, 5) / 5
        ])

    def create_private_scalars(self, player):
        """Create private scalars for a player

        Private Scalars:
            - exp: int
            - exp to next level: int
            - gold: int
        """
        match_history = np.zeros(3)
        if len(player.match_history) > 2:
            match_history = player.match_history[-3::]
        # Who we can play against in the next round. / 20 to keep numbers between 0 and 1.

        return_array = np.array([
            player.exp / 100,
            player.round / 30,
            max(player.win_streak, player.loss_streak) / 30,
            player.actions_remaining / config.ACTIONS_PER_TURN
        ])

        return np.concatenate([return_array, match_history], axis=-1)

    def create_embedding_scalars(self, player):
        # 60 on this patch, probably higher on a future patch
        gold = player.gold if player.gold < 60 else 60
        # 101 on this patch, probably higher on a future patch, 0 is never used but this is more human friendly
        health = player.health
        # lets be safe and go with 100
        exp_to_level = 0
        if player.level < player.max_level:
            exp_to_level = (player.level_costs[player.level] - player.exp)
        # no more than 40
        game_round = player.round
        opponent_options = np.zeros(8)
        for x in range(8):
            if ("player_" + str(x)) in player.opponent_options:
                opponent_options[x] = 1 if player.opponent_options["player_" + str(x)] > 0 else 0
            else:
                opponent_options[x] = 0
        # 64 values
        next_fight_scalar = 0
        for i, x in enumerate(opponent_options):
            if i != player.player_num:
                next_fight_scalar *= 2
                next_fight_scalar += x
        # 10 values
        level = player.level
        assert gold <= 60
        assert health <= 100
        assert exp_to_level <= 100
        assert game_round < 40
        assert next_fight_scalar < 256
        assert level < 10
        return np.array([gold, health, exp_to_level, game_round, next_fight_scalar, level], dtype=np.int16)

    # -- Champion -- #
    def create_champion_vector(self, champion):
        """Create a champion token for a champion

        Champion Vector:

        championID: (shape: MAX_CHAMPION_IN_SET)

        items: (shape: MAX_ITEMS_IN_SET * 3)

        other stats: [
            chosen, stars, cost
        ]
        """
        champion_vector = np.zeros(5, dtype=np.int16)
        if champion is None:
            return champion_vector

        # -2 because 1 and 2 are not used if not there.
        champion_vector[0] = list(COST.keys()).index(champion.name) * 3 + champion.stars - 2
        assert champion_vector[0] < 221

        for i, item_id in enumerate(champion.items):
            if item_id is not None:
                champion_vector[i + 1] = list(items.keys()).index(item_id)
                assert champion_vector[i + 1] < 58

        is_chosen = True if champion.chosen else False

        trait_number = list(origin_class.keys()).index(champion.name)
        champion_vector[-1] = trait_number + len(origin_class.keys()) if is_chosen else trait_number
        assert champion_vector[-1] < 145

        return champion_vector

    def create_board_vector(self, player):
        """Create a board token for a player

        Board Vector: (28, champion_vector_length)

        Array board layout
                        Left
            | (0,0) (0,1) (0,2) (0,3) |
            | (1,0) (1,1) (1,2) (1,3) |
            | (2,0) (2,1) (2,2) (2,3) |
    Bottom  | (3,0) (3,1) (3,2) (3,3) |  Top
            | (4,0) (4,1) (4,2) (4,3) |
            | (5,0) (5,1) (5,2) (5,3) |
            | (6,0) (6,1) (6,2) (6,3) |
                        Right

        Rotated to match the board in game
                                    Top
            | (0, 3) (1, 3) (2, 3) (3, 3) (4, 3) (5, 3) (6, 3) |
      Left  | (0, 2) (1, 2) (2, 2) (3, 2) (4, 2) (5, 2) (6, 2) |
            | (0, 1) (1, 1) (2, 1) (3, 1) (4, 1) (5, 1) (6, 1) |  Right
            | (0, 0) (1, 0) (2, 0) (3, 0) (4, 0) (5, 0) (6, 0) |
                                    Bottom

        """

        board_vector = np.zeros((28, self.champion_vector_length), dtype=np.float32)

        for x in range(len(player.board)):
            for y in range(len(player.board[x])):
                if champion := player.board[x][y]:
                    champion_vector = self.create_champion_vector(champion)

                    loc = x_y_to_1d_coord(x, y)
                    board_vector[loc] = champion_vector

        return board_vector

    # -- Bench Vector -- #
    def create_bench_vector(self, player):
        """Create a bench token for a player

        Bench Vector: (9, champion_vector_length)

        Array bench layout
            | (0) (1) (2) (3) (4) (5) (6) (7) (8) |

        """
        bench_vector = np.zeros((9, self.champion_vector_length), dtype=np.int16)

        for idx, champion in enumerate(player.bench):
            if champion:
                champion_vector = self.create_champion_vector(champion)
                bench_vector[idx] = champion_vector

        return bench_vector

    # -- Shop Vector -- #
    def create_shop_vector(self, player):
        """Create shop token for a player

        Shop Vector: (5, champion_vector_length)

        Array Shop layout
            | (0) (1) (2) (3) (4) |

        """

        shop_vector = np.zeros((5, self.champion_vector_length), dtype=np.float32)

        for idx, champion in enumerate(player.shop_champions):
            if champion:
                champion_vector = self.create_champion_vector(champion)
                shop_vector[idx] = champion_vector

        return shop_vector

    # -- Item Bench Vector -- #
    def create_item_bench_vector(self, player):
        """Create an item bench token for a player

        Item Bench Vector: (10,)

        Array bench layout
            | (0) (1) (2) (3) (4) (5) (6) (7) (8) (9) |

        """
        item_bench_vector = np.zeros(10, dtype=np.int16)

        for i, item in enumerate(player.item_bench):
            if item:
                item_bench_vector[i] = self.util.get_item_id(item)
                assert(item_bench_vector[i]) < 58

        return item_bench_vector

    def create_trait_vector(self, player):
        current_position = 0
        tiers_vector = np.zeros(config.TIERS_FLATTEN_LENGTH, dtype=np.float32)
        base_tier_values = list(tiers.values())
        player_tier_values = list(player.team_tiers.values())
        for i in range(len(base_tier_values)):
            try:
                tiers_vector[current_position + player_tier_values[i]] = 1
                player.team_tier_labels[i] = np.zeros(config.TEAM_TIERS_VECTOR[i], dtype=np.float32)
                player.team_tier_labels[i][player_tier_values[i]] = 1
                current_position += len(base_tier_values[i]) + 1
            except IndexError:
                print("index i {} with player_tier_values[i] {}".format(i, player_tier_values[i]))
                print("team_tier_labels {}".format(player.team_tier_labels))

        chosen_vector = np.zeros(5, dtype=np.float32)
        if player.chosen:
            i_index = list(player.team_composition.keys()).index(player.chosen)
            # This should update the item name section of the token
            for z in range(5, 0, -1):
                if i_index > 2 * z:
                    chosen_vector[5 - z] = 1
                    i_index -= 2 * z
        tiers_vector = np.concatenate([tiers_vector, chosen_vector], axis=-1)
        return safe_normalize(tiers_vector)

    @staticmethod
    def observation_to_position_input(observation, action_count):
        other_players = {
            "board":
                np.array(
                    [observation["opponents"][x]["board"] for x in range(config.NUM_PLAYERS - 1)], dtype=np.int16
                )
            ,
            "traits":
                np.array(
                    [observation["opponents"][x]["traits"] for x in range(config.NUM_PLAYERS - 1)],
                )
            ,
        }
        a = np.array([action_count])
        action_count_one_hot = np.zeros((a.size, 12), dtype=np.float32)
        action_count_one_hot[np.arange(a.size), a] = 1

        return {
            "board": np.concatenate([np.expand_dims(observation["player"]["board"], axis=0),
                                     other_players["board"]], axis=0).astype(np.int16),
            "traits": np.concatenate([np.expand_dims(observation["player"]["traits"], axis=0),
                                      other_players["traits"]], axis=0, dtype=np.float32),
            "action_count": action_count_one_hot
        }

    def observation_to_input(self, observation):
        other_players = {
            "board":
                np.array(
                    [observation["opponents"][x]["board"] for x in range(config.NUM_PLAYERS - 1)], dtype=np.int16
                )
            ,
            "traits":
                np.array(
                    [observation["opponents"][x]["traits"] for x in range(config.NUM_PLAYERS - 1)],
                )
            ,
            "scalars":
                np.reshape(np.array(
                    [observation["opponents"][x]["scalars"] for x in range(config.NUM_PLAYERS - 1)]
                ), (1, -1)),
        }
        return {
            "board": np.concatenate([np.expand_dims(observation["player"]["board"], axis=0),
                                     other_players["board"]], axis=0).astype(np.int16),
            "traits": np.concatenate([np.expand_dims(observation["player"]["traits"], axis=0),
                                      other_players["traits"]], axis=0, dtype=np.float32),
            "bench": np.expand_dims(observation["player"]["bench"], axis=0),
            "items": np.expand_dims(observation["player"]["items"], axis=0),
            "shop": observation["player"]["shop"],
            "scalars": np.concatenate([np.expand_dims(observation["player"]["scalars"], axis=0),
                                      other_players["scalars"]], axis=1, dtype=np.float32),
            "emb_scalars": observation["player"]["emb_scalars"],
        }

    @staticmethod
    def observation_to_dictionary(observation):
        """Converts a list of observations to a batched dictionary."""

        return {
            key: np.stack([obs[key] for obs in observation])
            for key in observation[0].keys()
        }
