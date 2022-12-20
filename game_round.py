import time
import config
import AI_interface
import random
import multiprocessing
import numpy as np
from Simulator import champion, pool, pool_stats, minion
from Simulator.item_stats import item_builds as full_items, starting_items
from Simulator.player import player as player_class
from Simulator.champion_functions import MILLIS


class TFT_Simulation:
    def __init__(self):
        # Amount of damage taken as a base per round. First number is max round, second is damage
        self.ROUND_DAMAGE = [
            [3, 0],
            [9, 2],
            [15, 3],
            [21, 5],
            [27, 8],
            [10000, 15]
        ]
        self.pool_obj = pool.pool()
        self.num_players = config.NUM_PLAYERS
        self.PLAYERS = [player_class(self.pool_obj, i) for i in range(self.num_players)]
        self.NUM_DEAD = 0
        self.player_rewards = [0 for _ in range(self.num_players)]
        self.last_observation = [[] for _ in range(self.num_players)]
        self.last_action = [[] for _ in range(self.num_players)]
        self.last_value = [[] for _ in range(self.num_players)]
        self.last_policy = [[] for _ in range(self.num_players)]
        self.previous_reward = [0 for _ in range(self.num_players)]
        log_to_file_start()

    # Archived way to calculate reward
    # Now done in a simpler way by giving a positive reward to the winner and a negative of equal amount to the loser
    # of each fight
    def calculate_reward(self, player, previous_reward):
        self.player_rewards[player.player_num] = player.reward - previous_reward
        average = 0
        for i in range(self.num_players):
            if i != player.player_num and self.PLAYERS[i]:
                average += self.player_rewards[i]
        if self.NUM_DEAD < self.num_players - 1:
            average = average / (self.num_players - self.NUM_DEAD - 1)
        return player.reward - previous_reward - average

    def combat_phase(self, players, player_round):
        battle_start_time = time.time_ns()
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
                if players_matched < self.num_players - 1 - self.NUM_DEAD:
                    # The player to match is always going to be a higher number than the first player.
                    player_index = random.randint(0, len(players) - 1)
                    # if the index is not in the player_nums, then it shouldn't check the second part.
                    # Although the index is always going to be the index of some.
                    # Make sure the player is alive as well.
                    while ((not players[player_index]) or players[num].opponent == players[player_index] or players[
                            player_index].combat or num == player_index):
                        player_index = random.randint(0, len(players) - 1)
                        if (players[player_index] and (players_matched == self.num_players - 2 - self.NUM_DEAD)
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
                elif len(player_nums) == 1 or players_matched == self.num_players - 1 - self.NUM_DEAD:
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
                                other.won_round(damage/len(alive))
                    players[num].combat = True
                    players_matched += 1
                else:
                    return False
        log_to_file_combat()
        return True

    def check_dead(self, agents, buffers, game_episode):
        num_alive = 0
        game_observation = AI_interface.Observation()
        for i, player in enumerate(self.PLAYERS):
            if player:
                if player.health <= 0:

                    # This won't take into account how much health the most recent
                    # dead had if multiple players die at once
                    # Take an observation
                    observation, _ = game_observation.observation(player, buffers[player.player_num],
                                                                  [1, 0, 0, 0, 0, 0, 0, 0])
                    action, logits = agents[player.player_num].policy(observation, player.player_num)

                    # Get reward of -0.5 for losing the game
                    reward = -0.5
                    # Store experience to buffer
                    buffers[player.player_num].store_replay_buffer(observation, action, reward, logits)
                    print("Player " + str(player.player_num) + " achieved individual reward = " + str(player.reward))
                    self.NUM_DEAD += 1
                    self.pool_obj.return_hero(player)

                    self.PLAYERS[i] = None
                else:
                    num_alive += 1
        if num_alive == 1:
            for i, player in enumerate(self.PLAYERS):
                if player:
                    player.won_game()
                    self.pool_obj.return_hero(player)
                    shop = self.pool_obj.sample(player, 5)
                    # Take an observation
                    observation, _ = game_observation.observation(player, buffers[player.player_num],
                                                                  [1, 0, 0, 0, 0, 0, 0, 0])
                    action, logits = agents[player.player_num].policy(observation, player.player_num)

                    # Get reward 1 for winning the game
                    reward = 1
                    # Store experience to buffer
                    buffers[player.player_num].store_replay_buffer(observation, action, reward, logits)
                    print("Player " + str(player.player_num) + " achieved individual reward = " + str(player.reward))
                    print("PLAYER {} WON".format(player.player_num))
                    # with agents[player.player_num].file_writer.as_default():
                    #     summary.scalar('player {} reward'.format(player.player_num), player.reward, game_episode)
                    return True
        return False

    def ai_buy_phase(self, player, agent, buffer, game_episode=0):
        # First store the end turn and reward for the previous battle together into the buffer
        # If statement is to make sure I don't put a record in for the first round.
        # Player reward starts at 0 but has to change when units go to the board at the end of the first turn
        
        shop = self.pool_obj.sample(player, 5)
        step_done = False
        actions_taken = 0
        game_observation = AI_interface.Observation()
        game_observation.generate_game_comps_vector()
        game_observation.generate_shop_vector(shop)
        # step_counter = 0
        while not step_done:
            action_vector = np.array([1, 0, 0, 0, 0, 0, 0, 0])
            # Take an observation
            observation, game_state_vector = game_observation.observation(player, buffer, action_vector)
            # Get action from the policy network
            action, policy = agent.policy(observation, player.player_num)
            # Take a step
            shop, step_done, success, time_taken = AI_interface.multi_step(action, player, shop, self.pool_obj,
                                                                           game_observation, agent, buffer)

            # Get reward - This is always 0 with our current reward structure
            # Leaving it here in case we change our reward structure
            # reward = self.calculate_reward(player, previous_reward)
            reward = player.reward - self.previous_reward[player.player_num]

            # Store experience to buffer
            if step_done:
                self.last_observation[player.player_num] = observation
                self.last_action[player.player_num] = action
                self.last_policy[player.player_num] = policy
            else:
                buffer.store_replay_buffer(observation, action, reward, policy)
            if success and ~step_done:
                buffer.store_observation(game_state_vector)

            self.previous_reward[player.player_num] = player.reward
            actions_taken += time_taken
            if actions_taken >= 30:
                step_done = True
        player.print(str(actions_taken) + " actions taken this turn")
        # step_counter += 1

    def player_round(self, game_round, players, agent, buffers, game_episode):
        for i in range(config.NUM_PLAYERS):
            players[i].start_round(game_round)
        AI_interface.batch_step(players, agent, buffers, self.pool_obj)

        # TO DO
        # Change this so it isn't tied to a player and can log as time proceeds
        for i in range(config.NUM_PLAYERS):
            log_to_file(players[i])

    # This is going to take a list of agents, 1 for each player
    # This is also taking in a list of buffers, 1 for each player.
    # TO DO: Add a set of buffers at the end for the ending state of each player
    def episode(self, agent, buffer, game_episode=0):
        # Currently Carousel rounds are compressed with the combat round after it in the round counter.
        # ROUND 0 AKA 1-1/1-2, Carousel + First Minion Combat
        for player in self.PLAYERS:
            if player:
                # first carousel
                ran_cost_1 = list(pool_stats.COST_1.items())[random.randint(0, len(pool_stats.COST_1) - 1)][0]
                ran_cost_1 = champion.champion(ran_cost_1,
                                               itemlist=[starting_items[random.randint(0, len(starting_items) - 1)]])
                self.pool_obj.update(ran_cost_1, -1)
                player.add_to_bench(ran_cost_1)
                log_to_file(player)

        # local_time = time.time_ns()
        processes = []
        for player in self.PLAYERS:
            p = multiprocessing.Process(target=minion.minion_round, args=(player, 0, self.PLAYERS))
            p.start()
            processes.append(p)

        for process in processes:
            process.join()
        # print("First minion round took {} time".format(time.time_ns() - local_time))

        # ROUND 1-3 - Buy phase + Give 1 item component and 1 random 3 cost champion
        self.player_round(1, self.PLAYERS, agent, buffer, game_episode)

        log_end_turn(1)

        for player in self.PLAYERS:
            if player:
                minion.minion_round(player, 1, self.PLAYERS)

        self.player_round(2, self.PLAYERS, agent, buffer, game_episode)

        log_end_turn(2)
        for player in self.PLAYERS:
            if player:
                minion.minion_round(player, 2, self.PLAYERS)

        # STAGE 2 BEGINS HERE
        # Round 2-1 to 2-3: 3 Player Combats
        for r in range(3, 6):
            self.player_round(r, self.PLAYERS, agent, buffer, game_episode)
            log_end_turn(r)

            self.combat_phase(self.PLAYERS, r)
            if self.check_dead(agent, buffer, game_episode):
                return True
            log_to_file_combat()

        # random 3 drop with item
        # (Carousel round)
        for player in self.PLAYERS:
            # Technically round 2-4 here (carousel)
            ran_cost_3 = list(pool_stats.COST_3.items())[random.randint(0, len(pool_stats.COST_3) - 1)][0]
            ran_cost_3 = champion.champion(ran_cost_3,
                                           itemlist=[starting_items[random.randint(0, len(starting_items) - 1)]])
            self.pool_obj.update(ran_cost_3, -1)
            player.add_to_bench(ran_cost_3)

        # Round 2-5 to 2-6: 2 Player Combats
        for r in range(6, 8):
            self.player_round(r, self.PLAYERS, agent, buffer, game_episode)
            log_end_turn(r)

            self.combat_phase(self.PLAYERS, r)
            if self.check_dead(agent, buffer, game_episode):
                return True
            log_to_file_combat()

        # Round 2-7 Krugs Round - 3 gold plus 3 item components
        for player in self.PLAYERS:
            if player:
                minion.minion_round(player, 8, self.PLAYERS)

        # STAGE 3 BEGINS HERE
        # Round 3-1 to 3-3: 3 Player Combats
        for r in range(9, 12):
            self.player_round(r, self.PLAYERS, agent, buffer, game_episode)
            log_end_turn(r)

            self.combat_phase(self.PLAYERS, r)
            if self.check_dead(agent, buffer, game_episode):
                return True
            log_to_file_combat()

        # Another carousel (technically round 3-4)
        for player in self.PLAYERS:
            if player:
                ran_cost_3 = list(pool_stats.COST_3.items())[random.randint(0, len(pool_stats.COST_3) - 1)][0]
                ran_cost_3 = champion.champion(ran_cost_3,
                                               itemlist=[starting_items[random.randint(0, len(starting_items) - 1)]])
                self.pool_obj.update(ran_cost_3, -1)
                player.add_to_bench(ran_cost_3)

        # Round 3-4 to 3-5: 2 Player Combats
        for r in range(12, 14):
            self.player_round(r, self.PLAYERS, agent, buffer, game_episode)
            log_end_turn(r)

            self.combat_phase(self.PLAYERS, r)
            if self.check_dead(agent, buffer, game_episode):
                return True
            log_to_file_combat()
        
        # Round 3-7: Wolves round
        for player in self.PLAYERS:
            if player:
                minion.minion_round(player, 14, self.PLAYERS)

        # STAGE 4 BEGINS HERE
        # Round 4-1 to 4-3: 3 Player Combats
        for r in range(15, 18):
            self.player_round(r, self.PLAYERS, agent, buffer, game_episode)
            log_end_turn(r)

            self.combat_phase(self.PLAYERS, r)
            if self.check_dead(agent, buffer, game_episode):
                return True
            log_to_file_combat()

        # Another carousel (Technically round 4-4)
        for player in self.PLAYERS:
            if player:
                ran_cost_4 = list(pool_stats.COST_4.items())[random.randint(0, len(pool_stats.COST_4) - 1)][0]
                ran_cost_4 = champion.champion(ran_cost_4,
                                               itemlist=[starting_items[random.randint(0, len(starting_items) - 1)]])
                self.pool_obj.update(ran_cost_4, -1)
                player.add_to_bench(ran_cost_4)

        # Round 4-5 to 4-6: 2 Player Combats
        for r in range(18, 20):
            self.player_round(r, self.PLAYERS, agent, buffer, game_episode)
            log_end_turn(r)

            self.combat_phase(self.PLAYERS, r)
            if self.check_dead(agent, buffer, game_episode):
                return True
            log_to_file_combat()

        # Round 4-7: Raptors round
        for player in self.PLAYERS:
            if player:
                minion.minion_round(player, 20, self.PLAYERS)

        # STAGE 5 BEGINS HERE
        # Round 5-1 to 5-3: 3 Player Combats
        for r in range(21, 24):
            self.player_round(r, self.PLAYERS, agent, buffer, game_episode)
            log_end_turn(r)

            self.combat_phase(self.PLAYERS, r)
            if self.check_dead(agent, buffer, game_episode):
                return True
            log_to_file_combat()

        # Another carousel (Technically round 5-4)
        for player in self.PLAYERS:
            if player:
                ran_cost_5 = list(pool_stats.COST_5.items())[random.randint(0, len(pool_stats.COST_5) - 1)][0]
                item_list = list(full_items.keys())
                ran_cost_5 = champion.champion(ran_cost_5, itemlist=[item_list[random.randint(0, len(item_list) - 1)]])
                self.pool_obj.update(ran_cost_5, -1)
                player.add_to_bench(ran_cost_5)

        # Round 5-5 to 5-6: 2 Player Combats
        for r in range(24, 26):
            self.player_round(r, self.PLAYERS, agent, buffer, game_episode)
            log_end_turn(r)

            self.combat_phase(self.PLAYERS, r)
            if self.check_dead(agent, buffer, game_episode):
                return True
            log_to_file_combat()

        # Round 5-7: Dragon Round - 6 gold and a full item
        # here be dragons
        for player in self.PLAYERS:
            if player:
                minion.minion_round(player, 26, self.PLAYERS)

        # STAGE 6 BEGINS HERE
        # Round 6-1 to 6-3: 3 Player Combats
        for r in range(27, 30):
            self.player_round(r, self.PLAYERS, agent, buffer, game_episode)
            log_end_turn(r)

            self.combat_phase(self.PLAYERS, r)
            if self.check_dead(agent, buffer, game_episode):
                return True
            log_to_file_combat()

        # Another carousel (Technically round 6-4)
        for player in self.PLAYERS:
            if player:
                ran_cost_5 = list(pool_stats.COST_5.items())[random.randint(0, len(pool_stats.COST_5) - 1)][0]
                item_list = list(full_items.keys())
                ran_cost_5 = champion.champion(ran_cost_5, itemlist=[item_list[random.randint(0, len(item_list) - 1)]])
                self.pool_obj.update(ran_cost_5, -1)
                player.add_to_bench(ran_cost_5)

        # Round 6-5 to 6-6: 2 Player Combats
        for r in range(30, 32):
            self.player_round(r, self.PLAYERS, agent, buffer, game_episode)
            log_end_turn(r)

            self.combat_phase(self.PLAYERS, r)
            if self.check_dead(agent, buffer, game_episode):
                return True
            log_to_file_combat()

        # Round 6-7: Rift Herald - 6 gold and a full item
        for player in self.PLAYERS:
            if player:
                minion.minion_round(player, 32, self.PLAYERS)

        # STAGE 7 BEGINS HERE
        # Round 7-1 to 7-3: 3 Player Combats
        for r in range(32, 35):
            self.player_round(r, self.PLAYERS, agent, buffer, game_episode)
            log_end_turn(r)

            self.combat_phase(self.PLAYERS, r)
            if self.check_dead(agent, buffer, game_episode):
                return True

        # Another carousel (Technically round 7-4)
        for player in self.PLAYERS:
            if player:
                ran_cost_5 = list(pool_stats.COST_5.items())[random.randint(0, len(pool_stats.COST_5) - 1)][0]
                item_list = list(full_items.keys())
                ran_cost_5 = champion.champion(ran_cost_5, itemlist=[item_list[random.randint(0, len(item_list) - 1)]])
                self.pool_obj.update(ran_cost_5, -1)
                player.add_to_bench(ran_cost_5)

        # Round 7-5 to 7-6: 2 Player Combats
        for r in range(35, 37):
            self.player_round(r, self.PLAYERS, agent, buffer, game_episode)
            log_end_turn(r)

            self.combat_phase(self.PLAYERS, r)
            if self.check_dead(agent, buffer, game_episode):
                return True
            log_to_file_combat()

        # Round 7-7: Another Rift Herald (long game)
        for player in self.PLAYERS:
            if player:
                minion.minion_round(player, 37, self.PLAYERS)

        # STAGE 8 BEGINS HERE
        # this should rarely/never happen, but just in case
        # Round 8-1 to 8-3: 3 Player Combats
        for r in range(38, 41):
            self.player_round(r, self.PLAYERS, agent, buffer, game_episode)
            log_end_turn(r)

            self.combat_phase(self.PLAYERS, r)
            if self.check_dead(agent, buffer, game_episode):
                return True

        # Another carousel (Technically round 8-4)
        for player in self.PLAYERS:
            if player:
                ran_cost_5 = list(pool_stats.COST_5.items())[random.randint(0, len(pool_stats.COST_5) - 1)][0]
                item_list = list(full_items.keys())
                ran_cost_5 = champion.champion(ran_cost_5, itemlist=[item_list[random.randint(0, len(item_list) - 1)]])
                self.pool_obj.update(ran_cost_5, -1)
                player.add_to_bench(ran_cost_5)

        # Round 8-5 to 8-6: 2 Player Combats
        for r in range(41, 43):
            self.player_round(r, self.PLAYERS, agent, buffer, game_episode)
            log_end_turn(r)

            self.combat_phase(self.PLAYERS, r)
            if self.check_dead(agent, buffer, game_episode):
                return True
            log_to_file_combat()

        # Round 8-7: The final Rift Herald
        for player in self.PLAYERS:
            if player:
                minion.minion_round(player, 43, self.PLAYERS)

        print("Game has gone on way too long. There has to be a bug somewhere")
        for player in self.PLAYERS:
            if player:
                print(player.health)
        return False


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
