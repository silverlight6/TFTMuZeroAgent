import Simulator.config as config
import config as global_config
import time
import random
import numpy as np
from Simulator import champion, minion
from Simulator.champion_functions import MILLIS
from Simulator.carousel import carousel
from Simulator.alt_autobattler import alt_auto_battle


class Game_Round:
    def __init__(self, game_players, pool_obj, step_func_obj):
        # Amount of damage taken as a base per round. First number is max round, second is damage
        self.ROUND_DAMAGE = [
            [3, 0],
            [9, 2],
            [15, 3],
            [21, 5],
            [27, 8],
            [10000, 15]
        ]
        self.PLAYERS = game_players
        self.pool_obj = pool_obj
        self.step_func_obj = step_func_obj

        self.NUM_DEAD = 0
        self.current_round = 0
        self.matchups = []

        log_to_file_start()

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
        ]

    # Archived way to calculate reward
    # Now done in a simpler way by giving a positive reward to the winner and a negative of equal amount to the loser
    # of each fight

    def combat_phase(self, players, player_round):
        round_index = 0
        while player_round > self.ROUND_DAMAGE[round_index][0]:
            round_index += 1
        for match in self.matchups:
            if not match[1] == "ghost":
                # Assigning a battle
                players[match[0]].opponent = players[match[1]]
                players[match[1]].opponent = players[match[0]]
                # Fixing the time signature to see how long battles take.
                players[match[0]].start_time = time.time_ns()
                players[match[1]].start_time = time.time_ns()
                config.WARLORD_WINS['blue'] = players[match[0]].win_streak
                config.WARLORD_WINS['red'] = players[match[1]].win_streak

                if global_config.AUTO_BATTLER_PERCENTAGE < np.random.rand():
                    # Main simulation call
                    index_won, damage = champion.run(champion.champion, players[match[0]], players[match[1]],
                                                     self.ROUND_DAMAGE[round_index][1])
                else:
                    index_won, damage = alt_auto_battle(players[match[0]], players[match[1]],
                                                        self.ROUND_DAMAGE[round_index][1])

                # Draw
                if index_won == 0:
                    players[match[0]].loss_round(damage)
                    players[match[0]].health -= damage
                    players[match[1]].loss_round(damage)
                    players[match[1]].health -= damage
                    for player in players.values():
                        if player != players[match[0]] and player != players[match[1]]:
                            if player:  # Not sure if there can be a dead player here.
                                player.spill_reward(damage / (len(players) - 2))
                    if len(players) == 2:
                        players[match[0]].spill_reward(damage)
                        players[match[1]].spill_reward(damage)

                # Blue side won
                if index_won == 1:
                    players[match[0]].won_round(damage)
                    players[match[1]].loss_round(damage)
                    players[match[1]].health -= damage

                # Red side won
                if index_won == 2:
                    players[match[0]].loss_round(damage)
                    players[match[0]].health -= damage
                    players[match[1]].won_round(damage)
                players[match[0]].combat = True
                players[match[1]].combat = True

            else:
                players[match[0]].start_time = time.time_ns()
                config.WARLORD_WINS['blue'] = players[match[0]].win_streak
                config.WARLORD_WINS['red'] = players[match[2]].win_streak
                index_won, damage = champion.run(champion.champion, players[match[0]], players[match[2]],
                                                 self.ROUND_DAMAGE[round_index][1])
                if index_won == 2 or index_won == 0:
                    players[match[0]].health -= damage
                    players[match[0]].loss_round(damage)
                    players[match[0]].combat = True
                    # if the alive player loses to a dead player, the dead player's reward is
                    # given out to all other alive players
                    alive = []
                    for other in players.values():
                        if other:
                            if other.health > 0 and other is not players[match[0]]:
                                alive.append(other)
                    for other in alive:
                        other.spill_reward(damage / len(alive))
        log_to_file_combat()
        return True

    def decide_player_combat(self):
        player_list = []
        self.matchups = []
        for key, player in self.PLAYERS.items():
            if player:
                player_list.append(key)
        random.shuffle(player_list)
        ghost = player_list[0]      # if there's a ghost combat, always use first player after shuffling
        while len(player_list) >= 2:
            index = 1   # this is place in player_list that gets chosen as the opponent, should never be 0
            weights = 0
            # Specifying the player as its own variable due to number of usages.
            player = self.PLAYERS[player_list[0]]
            player.opponent_options = {"player_" + str(player_id): 0 for player_id in self.PLAYERS.keys()}
            for key in player_list:
                if not key == player_list[0]:
                    # if any possible opponents have a high enough possible opponents value consider them for combat
                    if player.possible_opponents[key] >= config.MATCHMAKING_WEIGHTS:
                        weights += player.possible_opponents[key]
            if weights == 0:
                # if no opponents have a high enough possible opponents value, take whichever one is highest
                opponent = 0
                for i, key in enumerate(player_list):
                    if i < 0:
                        if player.possible_opponents[key] >= opponent:
                            opponent = player.possible_opponents[key]
                            index = i
            else:
                # if there are opponents with a high enough value, use weights to determine who to fight
                r = random.randint(0, weights)
                while r >= player.possible_opponents[player_list[index]]:
                    r -= player.possible_opponents[player_list[index]]
                    index += 1
                    if index == len(player_list):
                        index = 1
            self.matchups.append([player_list[0], player_list[index]])
            opposition = self.PLAYERS[player_list[index]]
            opposition.opponent_options = {"player_" + str(player_id): 0 for player_id in self.PLAYERS.keys()}
            # giving the players the possible opponents
            for key, player_check in self.PLAYERS.items():
                if player_check:
                    if player.possible_opponents[key] >= config.MATCHMAKING_WEIGHTS:
                        player.opponent_options[key] = 1
                    if opposition.possible_opponents[key] >= config.MATCHMAKING_WEIGHTS:
                        opposition.opponent_options[key] = 1
                    if not player.possible_opponents[key] == -1:
                        player.possible_opponents[key] += config.WEIGHTS_INCREMENT
                    if not opposition.possible_opponents[key] == -1:
                        opposition.possible_opponents[key] += config.WEIGHTS_INCREMENT
                else:
                    player.possible_opponents[key] = 0
                    opposition.possible_opponents[key] = 0
            if 1 not in player.opponent_options:
                player.opponent_options[player_list[index]] = 1
            if 1 not in opposition.opponent_options:
                opposition.opponent_options[player_list[0]] = 1
            # resetting the weights for the player and opponent
            player.possible_opponents[player_list[index]] = 0
            opposition.possible_opponents[player_list[0]] = 0
            player_list.remove(player_list[index])
            player_list.remove(player_list[0])
        if len(player_list) == 1:   # if there is only one player left to match, have it fight a ghost
            self.matchups.append([player_list[0], "ghost", ghost])
            self.PLAYERS[player_list[0]].opponent_options = \
                {"player_" + str(player_id): 0 for player_id in range(config.NUM_PLAYERS)}
            self.PLAYERS[player_list[0]].opponent_options[ghost] = 1

    def play_game_round(self):
        for i in range(len(self.game_rounds[self.current_round])):
            self.game_rounds[self.current_round][i]()
        self.current_round += 1

    def start_round(self):
        self.step_func_obj.generate_shops(self.PLAYERS)
        self.step_func_obj.generate_shop_vectors(self.PLAYERS)
        self.decide_player_combat()
        for player in self.PLAYERS.values():
            if player:
                player.start_round(self.current_round)

    def round_1(self):
        carousel(list(self.PLAYERS.values()), self.current_round, self.pool_obj)
        for player in self.PLAYERS.values():
                log_to_file(player)

        for player in self.PLAYERS.values():
            minion.minion_round(player, 0, self.PLAYERS.values())
        # False stands for no one died
        return False

    # r for minion round
    def minion_round(self):
        for player in self.PLAYERS.values():
            if player:
                log_to_file(player)
        log_end_turn(self.current_round)

        for player in self.PLAYERS.values():
            if player:
                player.end_turn_actions()
                player.combat = False

        for player in self.PLAYERS.values():
            if player:
                minion.minion_round(player, self.current_round, self.PLAYERS.values())
        return False

    # r stands for round or game_round but round is a keyword so using r instead
    def combat_round(self):
        for player in self.PLAYERS.values():
            if player:
                player.end_turn_actions()
                player.combat = False
                log_to_file(player)
        log_end_turn(self.current_round)

        self.combat_phase(self.PLAYERS, self.current_round)
        # Will implement check dead later
        # if self.check_dead(agent, buffer, game_episode):
        #     return True
        log_to_file_combat()
        return False

    # executes carousel round for all players
    def carousel_round(self):
        carousel(list(self.PLAYERS.values()), self.current_round, self.pool_obj)
        for player in self.PLAYERS.values():
                if player:
                    log_to_file(player)
                    player.refill_item_pool()

    def terminate_game(self):
        print("Game has gone on way too long. There has to be a bug somewhere")
        for player in self.PLAYERS.values():
            if player:
                print(player.health)
        return False

    def update_players(self, players):
        self.PLAYERS = players


def log_to_file_start():
    if config.LOGMESSAGES:
        with open('log.txt', "w") as out:
            out.write("Start of a new run")
            out.write('\n')


def log_to_file(player):
    if config.LOGMESSAGES:
        with open('log.txt', "a") as out:
            for line in player.log:
                out.write(str(line))
                out.write('\n')
    player.log = []


def log_end_turn(game_round):
    if config.LOGMESSAGES:
        with open('log.txt', "a") as out:
            out.write("END OF ROUND " + str(game_round) + " : " + time.strftime("%H:%M:%S", time.localtime()))
            out.write('\n')


# This one is for the champion and logging the battles.
def log_to_file_combat():
    if config.LOGMESSAGES and config.LOG_COMBAT:
        with open('log.txt', "a") as out:
            if len(champion.log) > 0:
                if MILLIS() < 75000:
                    if champion.log[-1] == 'BLUE TEAM WON':
                        champion.test_multiple['blue'] += 1
                    if champion.log[-1] == 'RED TEAM WON':
                        champion.test_multiple['red'] += 1
                elif MILLIS() < 200000:
                    champion.test_multiple['draw'] += 1
                for line in champion.log:
                    out.write(str(line))
                    out.write('\n')
    champion.log = []
