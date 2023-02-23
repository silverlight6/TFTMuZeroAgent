import time
import config
import random
import numpy as np
# import multiprocessing
from Simulator import champion, pool_stats, minion
from Simulator.item_stats import item_builds as full_items, starting_items
from Simulator.player import player as player_class
from Simulator.champion_functions import MILLIS


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
            self.round_1,
            self.minion_round,
            self.minion_round,
            self.combat_round,
            self.combat_round,
            self.combat_round,
            self.carousel2_4,
            self.combat_round,
            self.combat_round,
            self.minion_round,
            self.combat_round,
            self.combat_round,
            self.combat_round,
            self.carousel3_4,
            self.combat_round,
            self.combat_round,
            self.minion_round,
            self.combat_round,
            self.combat_round,
            self.combat_round,
            self.carousel4_4,
            self.combat_round,
            self.combat_round,
            self.minion_round,
            self.combat_round,
            self.combat_round,
            self.combat_round,
            self.carousel5_4,
            self.combat_round,
            self.combat_round,
            self.minion_round,
            self.combat_round,
            self.combat_round,
            self.combat_round,
            self.carousel6_4,
            self.combat_round,
            self.combat_round,
            self.minion_round,
            self.combat_round,
            self.combat_round,
            self.combat_round,
            self.carousel7_4,
            self.combat_round,
            self.combat_round,
            self.minion_round,
            self.combat_round,
            self.combat_round,
            self.combat_round,
            self.carousel8_4,
            self.combat_round,
            self.combat_round,
        ]

    # Archived way to calculate reward
    # Now done in a simpler way by giving a positive reward to the winner and a negative of equal amount to the loser
    # of each fight

    def combat_phase(self, players, player_round):
        round_index = 0
        while player_round > self.ROUND_DAMAGE[round_index][0]:
            round_index += 1
        for player in players:
            if player:
                player.end_turn_actions()
                player.combat = False
        for match in self.matchups:
            if not match[1] == "ghost":
                # Assigning a battle
                if not players[match[0]] or not players[match[1]]:
                    print(match[0], match[1])
                players[match[0]].opponent = players[match[1]]
                players[match[1]].opponent = players[match[0]]
                # Fixing the time signature to see how long battles take.
                players[match[0]].start_time = time.time_ns()
                players[match[1]].start_time = time.time_ns()
                config.WARLORD_WINS['blue'] = players[match[0]].win_streak
                config.WARLORD_WINS['red'] = players[match[1]].win_streak

                # Main simulation call
                index_won, damage = champion.run(champion.champion, players[match[0]], players[match[1]],
                                                 self.ROUND_DAMAGE[round_index][1])

                # Draw
                if index_won == 0:
                    players[match[0]].loss_round(damage)
                    players[match[0]].health -= damage
                    players[match[1]].loss_round(damage)
                    players[match[1]].health -= damage

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
                # if the alive player loses to a dead player, the dead player's reward is
                # given out to all other alive players
                alive = []
                for other in players:
                    if other:
                        if other.health > 0 and other is not players[match[0]]:
                            alive.append(other)
                if index_won == 2 or index_won == 0:
                    players[match[0]].health -= damage
                    players[match[0]].loss_round(player_round)
                    if len(alive) > 0:
                        for other in alive:
                            other.won_ghost(damage/len(alive))
                players[match[0]].combat = True
        log_to_file_combat()
        return True

    def play_game_round(self):
        self.game_rounds[self.current_round]()
        self.current_round += 1

    def start_round(self):
        self.step_func_obj.generate_shops(self.PLAYERS)
        self.step_func_obj.generate_shop_vectors(self.PLAYERS)
        for player in self.PLAYERS.values():
            if player:
                player.start_round(self.current_round)
        self.decide_player_combat(self.PLAYERS)

        # TO DO
        # Change this so it isn't tied to a player and can log as time proceeds

    def round_1(self):
        for player in self.PLAYERS.values():
            if player:
                # first carousel
                ran_cost_1 = list(pool_stats.COST_1.items())[random.randint(0, len(pool_stats.COST_1) - 1)][0]
                ran_cost_1 = champion.champion(ran_cost_1,
                                               itemlist=[starting_items[random.randint(0, len(starting_items) - 1)]])
                self.pool_obj.update_pool(ran_cost_1, -1)
                player.add_to_bench(ran_cost_1)
                log_to_file(player)

        for player in self.PLAYERS.values():
            minion.minion_round(player, 0, self.PLAYERS.values())
        for player in self.PLAYERS.values():
            if player:
                player.start_round(1)
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
        self.start_round()
        return False

    # r stands for round or game_round but round is a keyword so using r instead
    def combat_round(self):
        for player in self.PLAYERS.values():
            if player:
                log_to_file(player)
        log_end_turn(self.current_round)

        self.combat_phase(list(self.PLAYERS.values()), self.current_round)
        # Will implement check dead later
        # if self.check_dead(agent, buffer, game_episode):
        #     return True
        log_to_file_combat()
        return False

    def carousel2_4(self):
        for player in self.PLAYERS.values():
            ran_cost_3 = list(pool_stats.COST_3.items())[random.randint(0, len(pool_stats.COST_3) - 1)][0]
            ran_cost_3 = champion.champion(ran_cost_3,
                                           itemlist=[starting_items[random.randint(0, len(starting_items) - 1)]])
            self.pool_obj.update_pool(ran_cost_3, -1)
            player.add_to_bench(ran_cost_3)
            player.refill_item_pool()

    def carousel3_4(self):
        for player in self.PLAYERS.values():
            if player:
                ran_cost_3 = list(pool_stats.COST_3.items())[random.randint(0, len(pool_stats.COST_3) - 1)][0]
                ran_cost_3 = champion.champion(ran_cost_3,
                                               itemlist=[starting_items[random.randint(0, len(starting_items) - 1)]])
                self.pool_obj.update_pool(ran_cost_3, -1)
                player.add_to_bench(ran_cost_3)
                player.refill_item_pool()

    def carousel4_4(self):
        for player in self.PLAYERS.values():
            if player:
                ran_cost_4 = list(pool_stats.COST_4.items())[random.randint(0, len(pool_stats.COST_4) - 1)][0]
                ran_cost_4 = champion.champion(ran_cost_4,
                                               itemlist=[starting_items[random.randint(0, len(starting_items) - 1)]])
                self.pool_obj.update_pool(ran_cost_4, -1)
                player.add_to_bench(ran_cost_4)
                player.refill_item_pool()

    def carousel5_4(self):
        for player in self.PLAYERS.values():
            if player:
                ran_cost_5 = list(pool_stats.COST_5.items())[random.randint(0, len(pool_stats.COST_5) - 1)][0]
                item_list = list(full_items.keys())
                ran_cost_5 = champion.champion(ran_cost_5, itemlist=[item_list[random.randint(0, len(item_list) - 1)]])
                self.pool_obj.update_pool(ran_cost_5, -1)
                player.add_to_bench(ran_cost_5)
                player.refill_item_pool()

    def carousel6_4(self):
        for player in self.PLAYERS.values():
            if player:
                ran_cost_5 = list(pool_stats.COST_5.items())[random.randint(0, len(pool_stats.COST_5) - 1)][0]
                item_list = list(full_items.keys())
                ran_cost_5 = champion.champion(ran_cost_5, itemlist=[item_list[random.randint(0, len(item_list) - 1)]])
                self.pool_obj.update_pool(ran_cost_5, -1)
                player.add_to_bench(ran_cost_5)
                player.refill_item_pool()

    def carousel7_4(self):
        for player in self.PLAYERS.values():
            if player:
                ran_cost_5 = list(pool_stats.COST_5.items())[random.randint(0, len(pool_stats.COST_5) - 1)][0]
                item_list = list(full_items.keys())
                ran_cost_5 = champion.champion(ran_cost_5, itemlist=[item_list[random.randint(0, len(item_list) - 1)]])
                self.pool_obj.update_pool(ran_cost_5, -1)
                player.add_to_bench(ran_cost_5)
                player.refill_item_pool()

    def carousel8_4(self):
        for player in self.PLAYERS.values():
            if player:
                ran_cost_5 = list(pool_stats.COST_5.items())[random.randint(0, len(pool_stats.COST_5) - 1)][0]
                item_list = list(full_items.keys())
                ran_cost_5 = champion.champion(ran_cost_5, itemlist=[item_list[random.randint(0, len(item_list) - 1)]])
                self.pool_obj.update_pool(ran_cost_5, -1)
                player.add_to_bench(ran_cost_5)
                player.refill_item_pool()

    def decide_player_combat(self, players):
        player_list = []
        self.matchups = []
        for player in players.values():
            if player:
                player_list.append(player.player_num)
        random.shuffle(player_list)
        ghost = player_list[0]      # if there's a ghost combat, always use first player after shuffling
        while len(player_list) >= 2:
            index = 1   # this is place in player_list that gets chosen as the opponent, should never be 0
            weights = 0
            player = list(players.values())[player_list[0]]
            player.opponent_options = np.zeros(config.NUM_PLAYERS)
            for num in player_list:
                if not num == player_list[0]:
                    # if any possible opponents have a high enough possible opponents value consider them for combat
                    if player.possible_opponents[num] >= config.MATCHMAKING_WEIGHTS:
                        weights += player.possible_opponents[num]
            if weights == 0:
                # if no opponents have a high enough possible opponents value, take whichever one is highest
                opponent = 0
                for i, num in enumerate(player_list):
                    if i < 0:
                        if player.possible_opponents[num] >= opponent:
                            opponent = player.possible_opponents[num]
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
            opposition = list(players.values())[player_list[index]]
            opposition.opponent_options = np.zeros(config.NUM_PLAYERS)
            for x in range(config.NUM_PLAYERS):
                if player.possible_opponents[x] >= config.MATCHMAKING_WEIGHTS:
                    player.opponent_options[x] = 1
                if opposition.possible_opponents[x] >= config.MATCHMAKING_WEIGHTS:
                    opposition.opponent_options[x] = 1
                if not player.possible_opponents[x] == -1:
                    player.possible_opponents[x] += config.WEIGHTS_INCREMENT
                if not opposition.possible_opponents[x] == -1:
                    opposition.possible_opponents[x] += config.WEIGHTS_INCREMENT
            if 1 not in player.opponent_options:
                player.opponent_options[player_list[index]] = 1
            if 1 not in opposition.opponent_options:
                opposition.opponent_options[player_list[0]] = 1
            player.possible_opponents[player_list[index]] = 0
            opposition.possible_opponents[player_list[0]] = 0
            player_list.remove(player_list[index])
            player_list.remove(player_list[0])
        if len(player_list) == 1:   # if there is only one player left to match, have it fight a ghost
            self.matchups.append([player_list[0], "ghost", ghost])
            player = list(players.values())[player_list[0]]
            player.opponent_options = np.zeros(config.NUM_PLAYERS)
            player.opponent_options[ghost] = 1

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
            out.write("END OF ROUND " + str(game_round))
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
