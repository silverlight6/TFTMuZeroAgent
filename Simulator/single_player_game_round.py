import Simulator.config as config
import time
from Simulator import champion, minion
from Simulator.carousel import carousel
from Simulator.game_round import log_to_file, log_to_file_combat, log_to_file_start, log_end_turn
from Simulator.position_leveling_system import PositionLevelingSystem
from copy import deepcopy


class Game_Round:
    def __init__(self, game_player, pool_obj, step_func_obj):
        # Amount of damage taken as a base per round. First number is max round, second is damage
        self.ROUND_DAMAGE = [
            [3, 0],
            [9, 2],
            [15, 3],
            [21, 5],
            [27, 8],
            [10000, 15]
        ]
        self.PLAYER = game_player
        self.pool_obj = pool_obj
        self.step_func_obj = step_func_obj

        self.current_round = 0

        self.save_current_battle = {"player_" + str(player_id): False for player_id in range(config.NUM_PLAYERS)}
        self.leveling_system = PositionLevelingSystem()

        log_to_file_start()

        # TODO: Verify that the carousel rounds are in the correct locations
        self.game_rounds = [
            [self.round_1],
            [self.minion_round],
            [self.minion_round],
            [self.combat_round],
            [self.combat_round],
            [self.combat_round],
            [self.carousel_round, self.combat_round],
            [self.combat_round],
            [self.minion_round],
            [self.combat_round],
            [self.combat_round],
            [self.combat_round],
            [self.carousel_round, self.combat_round],
            [self.combat_round],
            [self.minion_round],
            [self.combat_round],
            [self.combat_round],
            [self.combat_round],
            [self.carousel_round, self.combat_round],
            [self.combat_round],
            [self.minion_round],
            [self.combat_round],
            [self.combat_round],
            [self.combat_round],
            [self.carousel_round, self.combat_round],
            [self.combat_round],
            [self.minion_round],
            [self.combat_round],
            [self.combat_round],
            [self.combat_round],
            [self.carousel_round, self.combat_round],
            [self.combat_round],
            [self.minion_round],
            [self.combat_round],
            [self.combat_round],
            [self.combat_round],
            [self.carousel_round, self.combat_round],
            [self.combat_round],
            [self.minion_round],
            [self.combat_round],
            [self.combat_round],
            [self.combat_round],
            [self.carousel_round, self.combat_round],
            [self.combat_round],
            [self.minion_round],
            [self.combat_round]
        ]

    def single_combat_phase(self, player, enemy):
        """
        Plays a single round of combat between 2 players. Used by the positioning and item models for an environment.
        Does not alter health of players, only tracks reward

        args:
            players: List[player 0, player 1]
        """
        # Fixing the time signature to see how long battles take.
        player.start_time = time.time_ns()
        enemy.start_time = time.time_ns()
        config.WARLORD_WINS['blue'] = player.win_streak
        config.WARLORD_WINS['red'] = 0

        player_0 = deepcopy(player)
        index_won, damage = champion.run(champion.champion, player_0, enemy)
        log_to_file_combat()

        # Draw
        if index_won == 0 or index_won == 2:
            return False

        # Blue side won
        else:
            return True

    def play_game_round(self):
        result = False
        for i in range(len(self.game_rounds[self.current_round])):
            result = self.game_rounds[self.current_round][i]()
        self.current_round += 1
        self.PLAYER.printComp()
        return result

    def start_round(self):
        self.PLAYER.refresh_shop()
        self.PLAYER.start_round(self.current_round)

    def round_1(self):
        carousel([self.PLAYER], self.current_round, self.pool_obj)
        log_to_file(self.PLAYER)

        result = minion.minion_round(self.PLAYER, 0, self.PLAYER)
        self.PLAYER.refresh_shop()
        # False stands for no one died
        return result

    # r for minion round
    def minion_round(self):
        self.PLAYER.end_turn_actions()
        self.PLAYER.combat = False
        log_to_file(self.PLAYER)
        log_end_turn(self.current_round)

        combat_result = minion.minion_round(self.PLAYER, self.current_round)
        return combat_result

    def combat_round(self):
        self.PLAYER.end_turn_actions()
        self.PLAYER.combat = False
        log_to_file(self.PLAYER)
        log_end_turn(self.current_round)

        # Insert Generated battle here
        [enemy, _, _] = self.leveling_system.generate_battle()

        level_result = self.single_combat_phase(self.PLAYER, enemy)
        self.leveling_system.level_up()

        log_to_file_combat()
        return level_result

    # executes carousel round for all players
    def carousel_round(self):
        carousel([self.PLAYER], self.current_round, self.pool_obj)
        log_to_file(self.PLAYER)
        self.PLAYER.refill_item_pool()

        return True
