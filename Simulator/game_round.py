import time
import config
import random
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
        random.shuffle(players)
        player_nums = list(range(0, len(players)))
        players_matched = 0
        round_index = 0
        while player_round > self.ROUND_DAMAGE[round_index][0]:
            round_index += 1
        for player in players:
            if player:
                player.end_turn_actions()
                player.combat = False
        for num in player_nums:
            # make sure I am dealing with one of the players who has yet to fight.
            if players[num] and not players[num].combat:
                # If there is more than one player left ot be matched.
                if players_matched < config.NUM_PLAYERS - 1 - self.NUM_DEAD:
                    # The player to match is always going to be a higher number than the first player.
                    player_index = random.randint(0, len(players) - 1)
                    # if the index is not in the player_nums, then it shouldn't check the second part.
                    # Although the index is always going to be the index of some.
                    # Make sure the player is alive as well.
                    while ((not players[player_index]) or players[num].opponent == players[player_index] or players[
                            player_index].combat or num == player_index):
                        player_index = random.randint(0, len(players) - 1)
                        if (players[player_index] and (players_matched == config.NUM_PLAYERS - 2 - self.NUM_DEAD)
                                and players[num].opponent == players[player_index]):
                            break
                    # Assigning a battle
                    players[num].opponent = players[player_index]
                    players[player_index].opponent = players[num]

                    # Fixing the time signature to see how long battles take.
                    players[num].start_time = time.time_ns()
                    players[player_index].start_time = time.time_ns()
                    # Increment to see how many battles have happened
                    players_matched += 2
                    config.WARLORD_WINS['blue'] = players[num].win_streak
                    config.WARLORD_WINS['red'] = players[player_index].win_streak

                    # Main simulation call
                    index_won, damage = champion.run(champion.champion, players[num], players[player_index],
                                                     self.ROUND_DAMAGE[round_index][1])

                    # Draw
                    if index_won == 0:
                        players[num].loss_round(damage)
                        players[num].health -= damage
                        players[player_index].loss_round(damage)
                        players[player_index].health -= damage

                    # Blue side won
                    if index_won == 1:
                        players[num].won_round(damage)
                        players[player_index].loss_round(damage)
                        players[player_index].health -= damage

                    # Red side won
                    if index_won == 2:
                        players[num].loss_round(damage)
                        players[num].health -= damage
                        players[player_index].won_round(damage)
                    players[player_index].combat = True
                    players[num].combat = True

                # This is here when there is an odd number of players
                # Behavior is to fight a random player.
                elif len(player_nums) == 1 or players_matched == config.NUM_PLAYERS - 1 - self.NUM_DEAD:
                    player_index = random.randint(0, len(players) - 1)
                    while ((not players[player_index]) or players[num].opponent == players[player_index]
                            or num == player_index):
                        player_index = random.randint(0, len(players) - 1)
                    player_copy = player_class(self.pool_obj, players[player_index].player_num)
                    player_copy.board = players[player_index].board
                    player_copy.chosen = players[player_index].chosen
                    players[num].start_time = time.time_ns()
                    config.WARLORD_WINS['blue'] = players[num].win_streak
                    config.WARLORD_WINS['red'] = player_copy.win_streak
                    index_won, damage = champion.run(champion.champion, players[num], player_copy,
                                                     self.ROUND_DAMAGE[round_index][1])
                    # if the alive player loses to a dead player, the dead player's reward is
                    # given out to all other alive players
                    alive = []
                    for other in players:
                        if other:
                            if other.health > 0 and other is not players[num]:
                                alive.append(other)
                    if index_won == 2 or index_won == 0:
                        players[num].health -= damage
                        players[num].loss_round(player_round)
                        if len(alive) > 0:
                            for other in alive:
                                other.won_ghost(damage/len(alive))
                    players[num].combat = True
                    players_matched += 1
                else:
                    return False
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
        self.start_round()

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
