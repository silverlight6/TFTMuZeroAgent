import config
import AI_interface
import random
from Simulator import champion, pool, pool_stats
from Simulator.item_stats import item_builds as full_items, basic_items, starting_items
from Simulator.player import player as player_class
from interface import interface
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
                    players[num].opponent = players[player_index]
                    players[player_index].opponent = players[num]
                    players_matched += 2
                    config.WARLORD_WINS['blue'] = players[num].win_streak
                    config.WARLORD_WINS['red'] = players[player_index].win_streak
                    players[player_index].start_round()
                    players[num].start_round()
                    index_won, damage = champion.run(champion.champion, players[num], players[player_index],
                                                     self.ROUND_DAMAGE[round_index][1])
                    if index_won == 0:
                        players[num].loss_round(player_round)
                        players[num].health -= damage
                        players[player_index].loss_round(player_round)
                        players[player_index].health -= damage
                    if index_won == 1:
                        players[num].won_round(player_round)
                        players[player_index].loss_round(player_round)
                        players[player_index].health -= damage
                    if index_won == 2:
                        players[num].loss_round(player_round)
                        players[num].health -= damage
                        players[player_index].won_round(player_round)
                    players[player_index].combat = True
                    players[num].combat = True

                elif len(player_nums) == 1 or players_matched == self.num_players - 1 - self.NUM_DEAD:
                    player_index = random.randint(0, len(players) - 1)
                    while ((not players[player_index]) or players[num].opponent == players[player_index]
                            or num == player_index):
                        player_index = random.randint(0, len(players) - 1)
                    player_copy = player_class(self.pool_obj, players[player_index].player_num)
                    player_copy.board = players[player_index].board
                    player_copy.chosen = players[player_index].chosen
                    config.WARLORD_WINS['blue'] = players[num].win_streak
                    config.WARLORD_WINS['red'] = player_copy.win_streak
                    index_won, damage = champion.run(champion.champion, players[num], player_copy,
                                                     self.ROUND_DAMAGE[round_index][1])
                    if index_won == 1:
                        players[num].won_round(player_round)
                    elif index_won == 2 or index_won == 0:
                        players[num].health -= damage
                        players[num].loss_round(player_round)
                    players[num].start_round()
                    players[num].combat = True
                    players_matched += 1
                else:
                    return False
        log_to_file_combat()
        return True

    def check_dead(self, agents, buffers, game_episode):
        num_alive = 0
        for i, player in enumerate(self.PLAYERS):
            if player:
                if player.health <= 0:

                    # This won't take into account how much health the most recent
                    # dead had if multiple players die at once
                    # But this should be good enough for now.
                    # Get action from the policy network
                    shop = self.pool_obj.sample(player, 5)
                    # Take an observation
                    observation, _ = AI_interface.observation(shop, player, buffers[player.player_num])
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
                    observation, _ = AI_interface.observation(shop, player, buffers[player.player_num])
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
        if player.reward != 0:
            buffer.store_replay_buffer(self.last_observation[player.player_num], self.last_action[player.player_num],
                                       player.reward - self.previous_reward[player.player_num],
                                       self.last_policy[player.player_num])
        # Generate a shop for the observation to use
        shop = self.pool_obj.sample(player, 5)
        step_done = False
        actions_taken = 0
        # step_counter = 0
        while not step_done:
            # Take an observation
            observation, game_state_vector = AI_interface.observation(shop, player, buffer)
            # Get action from the policy network
            action, policy = agent.policy(observation, player.player_num)
            # Take a step
            shop, step_done, success = AI_interface.step(action, player, shop, self.pool_obj)

            # Get reward
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
            actions_taken += 1
        player.print(str(actions_taken) + " actions taken this turn")
        # step_counter += 1

    def human_game_logic(self):
        interface_obj = interface()
        # ROUND 0 - Give a random 1 cost unit with item. Add random item at end of round
        # items here is a method of the list and not related to the ingame items.
        # TO DO - Give a different 1 cost unit to each players instead of a random one to each player
        # TO DO MUCH LATER - Add randomness to the drops,
        # 3 gold one round vs 3 1 star units vs 3 2 star units and so on.
        for player in self.PLAYERS:
            ran_cost_1 = list(pool_stats.COST_1.items())[random.randint(0, len(pool_stats.COST_1) - 1)][0]
            ran_cost_1 = champion.champion(ran_cost_1)
            ran_cost_1.add_item(starting_items[random.randint(0, len(starting_items) - 1)])
            player.add_to_bench(ran_cost_1)
            log_to_file(player)
            player.add_to_item_bench(starting_items[random.randint(0, len(starting_items) - 1)])

        # ROUND 1 - Buy phase + Give 1 item component and 1 random 3 cost champion
        for player in self.PLAYERS:
            interface_obj.outer_loop(self.pool_obj, player, 1)
            log_to_file(player)
            player.add_to_item_bench(starting_items[random.randint(0, len(starting_items) - 1)])
            ran_cost_3 = list(pool_stats.COST_3.items())[random.randint(0, len(pool_stats.COST_3) - 1)][0]
            ran_cost_3 = champion.champion(ran_cost_3)
            player.add_to_bench(ran_cost_3)

        # Round 2 -  Buy phase + Give 3 gold and 1 random item component
        for player in self.PLAYERS:
            interface_obj.outer_loop(self.pool_obj, player, 2)
            log_to_file(player)
            player.add_to_item_bench(starting_items[random.randint(0, len(starting_items) - 1)])
            player.gold += 3

        for r in range(3, 6):
            # Round 3 to 5 - Buy phase + Combat phase
            for player in self.PLAYERS:
                interface_obj.outer_loop(self.pool_obj, player, r)
                log_to_file(player)

            self.combat_phase(self.PLAYERS, r)
            if self.check_dead(): return True
            log_to_file_combat()

        # Round 6 - random 3 drop with item + Combat phase
        # (Carosell round)
        for player in self.PLAYERS:
            ran_cost_3 = list(pool_stats.COST_3.items())[random.randint(0, len(pool_stats.COST_3) - 1)][0]
            ran_cost_3 = champion.champion(ran_cost_3)
            ran_cost_3.add_item(basic_items[random.randint(0, len(basic_items) - 1)])
            player.add_to_bench(ran_cost_3)

        for r in range(6, 9):
            for player in self.PLAYERS:
                interface_obj.outer_loop(self.pool_obj, player, r)
                log_to_file(player)

            self.combat_phase(self.PLAYERS, r)
            if self.check_dead():
                return True
            log_to_file_combat()

        # Golum Round - 3 gold plus 3 item components
        for player in self.PLAYERS:
            player.gold += 3
            for _ in range(0, 3):
                player.add_to_item_bench(starting_items[random.randint(0, len(starting_items) - 1)])

        for r in range(9, 12):
            for player in self.PLAYERS:
                interface_obj.outer_loop(self.pool_obj, player, r)
                log_to_file(player)

            self.combat_phase(self.PLAYERS, r)
            if self.check_dead():
                return True
            log_to_file_combat()

        # Another carousel
        for player in self.PLAYERS:
            ran_cost_3 = list(pool_stats.COST_3.items())[random.randint(0, len(pool_stats.COST_3) - 1)][0]
            ran_cost_3 = champion.champion(ran_cost_3)
            ran_cost_3.add_item(basic_items[random.randint(0, len(basic_items) - 1)])
            player.add_to_bench(ran_cost_3)

        for r in range(9, 12):
            for player in self.PLAYERS:
                interface_obj.outer_loop(self.pool_obj, player, r)
                log_to_file(player)

            self.combat_phase(self.PLAYERS, r)
            if self.check_dead():
                return True
            log_to_file_combat()

        for player in self.PLAYERS:
            player.gold += 3
            for _ in range(0, 3):
                player.add_to_item_bench(starting_items[random.randint(0, len(starting_items) - 1)])

        for r in range(12, 15):
            for player in self.PLAYERS:
                interface_obj.outer_loop(self.pool_obj, player, r)
                log_to_file(player)

            self.combat_phase(self.PLAYERS, r)
            if self.check_dead():
                return True
            log_to_file_combat()

        # Another carosell
        for player in self.PLAYERS:
            ran_cost_4 = list(pool_stats.COST_4.items())[random.randint(0, len(pool_stats.COST_4) - 1)][0]
            ran_cost_4 = champion.champion(ran_cost_4)
            ran_cost_4.add_item(basic_items[random.randint(0, len(basic_items) - 1)])
            player.add_to_bench(ran_cost_4)

        for r in range(15, 18):
            for player in self.PLAYERS:
                interface_obj.outer_loop(self.pool_obj, player, r)
                log_to_file(player)

            self.combat_phase(self.PLAYERS, r)
            if self.check_dead():
                return True
            log_to_file_combat()

        # Wolves Round - 3 gold plus 3 item components
        for player in self.PLAYERS:
            player.gold += 6
            for _ in range(0, 4):
                player.add_to_item_bench(starting_items[random.randint(0, len(starting_items) - 1)])

        for r in range(18, 21):
            # Round 3 to 5 - Buy phase + Combat phase
            for player in self.PLAYERS:
                interface_obj.outer_loop(self.pool_obj, player, r)
                log_to_file(player)

            self.combat_phase(self.PLAYERS, r)
            if self.check_dead():
                return True
            log_to_file_combat()

        # Another carousel
        for player in self.PLAYERS:
            ran_cost_5 = list(pool_stats.COST_5.items())[random.randint(0, len(pool_stats.COST_4) - 1)][0]
            ran_cost_5 = champion.champion(ran_cost_5)
            item_list = list(full_items.keys())
            ran_cost_5.add_item(item_list[random.randint(0, len(item_list) - 1)])
            player.add_to_bench(ran_cost_5)

        for r in range(21, 24):
            # Round 3 to 5 - Buy phase + Combat phase
            for player in self.PLAYERS:
                interface_obj.outer_loop(self.pool_obj, player, r)
                log_to_file(player)

            self.combat_phase(self.PLAYERS, r)
            if self.check_dead():
                return True
            log_to_file_combat()

        # Dragon Round - 3 gold plus 3 item components
        for player in self.PLAYERS:
            player.gold += 6
            item_list = list(full_items.keys())
            player.add_to_item_bench(item_list[random.randint(0, len(item_list) - 1)])

        for r in range(24, 27):
            # Round 3 to 5 - Buy phase + Combat phase
            for player in self.PLAYERS:
                interface_obj.outer_loop(self.pool_obj, player, r)
                log_to_file(player)

            self.combat_phase(self.PLAYERS, r)
            if self.check_dead():
                return True
            log_to_file_combat()

        # Another carosell
        for player in self.PLAYERS:
            ran_cost_5 = list(pool_stats.COST_5.items())[random.randint(0, len(pool_stats.COST_4) - 1)][0]
            ran_cost_5 = champion.champion(ran_cost_5)
            item_list = list(full_items.keys())
            ran_cost_5.add_item(item_list[random.randint(0, len(item_list) - 1)])
            player.add_to_bench(ran_cost_5)

        for r in range(27, 30):
            for player in self.PLAYERS:
                interface_obj.outer_loop(self.pool_obj, player, r)
                log_to_file(player)

            self.combat_phase(self.PLAYERS, r)
            if self.check_dead():
                return True
            log_to_file_combat()

        # Rift Herald - 3 gold plus 3 item components
        for player in self.PLAYERS:
            player.gold += 6
            item_list = list(full_items.keys())
            player.add_to_item_bench(item_list[random.randint(0, len(item_list) - 1)])

        for r in range(30, 33):
            # Round 3 to 5 - Buy phase + Combat phase
            for player in self.PLAYERS:
                interface_obj.outer_loop(self.pool_obj, player, r)
                log_to_file(player)

            self.combat_phase(self.PLAYERS, r)
            if self.check_dead():
                return True

        print("Game has gone on way too long. There has to be a bug somewhere")
        for player in self.PLAYERS:
            print(player.health)
        return False

    # This is going to take a list of agents, 1 for each player
    # This is also taking in a list of buffers, 1 for each player.
    # TO DO: Add an additional set of buffers at the end for the ending state of each player
    def episode(self, agents, buffer, game_episode=0):

        for player in self.PLAYERS:
            if player:
                ran_cost_1 = list(pool_stats.COST_1.items())[random.randint(0, len(pool_stats.COST_1) - 1)][0]
                ran_cost_1 = champion.champion(ran_cost_1,
                                               itemlist=[starting_items[random.randint(0, len(starting_items) - 1)]])
                self.pool_obj.update(ran_cost_1, -1)
                player.add_to_bench(ran_cost_1)
                log_to_file(player)
                player.add_to_item_bench(starting_items[random.randint(0, len(starting_items) - 1)])

        # ROUND 1 - Buy phase + Give 1 item component and 1 random 3 cost champion
        for player in self.PLAYERS:
            if player:
                player.gold_income(1)
                self.ai_buy_phase(player, agents[player.player_num], buffer[player.player_num], game_episode)
                log_to_file(player)

        log_end_turn(1)

        for player in self.PLAYERS:
            if player:
                player.add_to_item_bench(starting_items[random.randint(0, len(starting_items) - 1)])
                ran_cost_3 = list(pool_stats.COST_3.items())[random.randint(0, len(pool_stats.COST_3) - 1)][0]
                ran_cost_3 = champion.champion(ran_cost_3)
                self.pool_obj.update(ran_cost_3, -1)
                player.add_to_bench(ran_cost_3)

        # Round 2 -  Buy phase + Give 3 gold and 1 random item component
        for player in self.PLAYERS:
            if player:
                player.gold_income(2)
                self.ai_buy_phase(player, agents[player.player_num], buffer[player.player_num], game_episode)
                log_to_file(player)

        log_end_turn(2)
        for player in self.PLAYERS:
            if player:
                player.add_to_item_bench(starting_items[random.randint(0, len(starting_items) - 1)])
                player.gold += 3

        for r in range(3, 6):
            # Round 3 to 5 - Buy phase + Combat phase
            for player in self.PLAYERS:
                if player:
                    player.gold_income(r)
                    self.ai_buy_phase(player, agents[player.player_num], buffer[player.player_num], game_episode)
                    log_to_file(player)
            log_end_turn(r)

            self.combat_phase(self.PLAYERS, r)
            if self.check_dead(agents, buffer, game_episode):
                return True
            log_to_file_combat()

        # Round 6 - random 3 drop with item + Combat phase
        # (Carousel round)
        for player in self.PLAYERS:
            ran_cost_3 = list(pool_stats.COST_3.items())[random.randint(0, len(pool_stats.COST_3) - 1)][0]
            ran_cost_3 = champion.champion(ran_cost_3,
                                           itemlist=[starting_items[random.randint(0, len(starting_items) - 1)]])
            self.pool_obj.update(ran_cost_3, -1)
            player.add_to_bench(ran_cost_3)

        for r in range(6, 9):
            for player in self.PLAYERS:
                if player:
                    player.gold_income(r)
                    self.ai_buy_phase(player, agents[player.player_num], buffer[player.player_num], game_episode)
                    log_to_file(player)
            log_end_turn(r)

            self.combat_phase(self.PLAYERS, r)
            if self.check_dead(agents, buffer, game_episode):
                return True
            log_to_file_combat()

        # Game Round - 3 gold plus 3 item components
        for player in self.PLAYERS:
            if player:
                player.gold += 3
                for _ in range(0, 3):
                    player.add_to_item_bench(starting_items[random.randint(0, len(starting_items) - 1)])

        for r in range(9, 12):
            for player in self.PLAYERS:
                if player:
                    player.gold_income(r)
                    self.ai_buy_phase(player, agents[player.player_num], buffer[player.player_num], game_episode)
                    log_to_file(player)
            log_end_turn(r)

            self.combat_phase(self.PLAYERS, r)
            if self.check_dead(agents, buffer, game_episode):
                return True
            log_to_file_combat()

        # Another carousel
        for player in self.PLAYERS:
            if player:
                ran_cost_3 = list(pool_stats.COST_3.items())[random.randint(0, len(pool_stats.COST_3) - 1)][0]
                ran_cost_3 = champion.champion(ran_cost_3,
                                               itemlist=[starting_items[random.randint(0, len(starting_items) - 1)]])
                self.pool_obj.update(ran_cost_3, -1)
                player.add_to_bench(ran_cost_3)

        for r in range(12, 15):
            for player in self.PLAYERS:
                if player:
                    player.gold_income(r)
                    self.ai_buy_phase(player, agents[player.player_num], buffer[player.player_num], game_episode)
                    log_to_file(player)
            log_end_turn(r)

            self.combat_phase(self.PLAYERS, r)
            if self.check_dead(agents, buffer, game_episode):
                return True
            log_to_file_combat()

        for player in self.PLAYERS:
            if player:
                player.gold += 3
                for _ in range(0, 3):
                    player.add_to_item_bench(starting_items[random.randint(0, len(starting_items) - 1)])

        for r in range(15, 18):
            for player in self.PLAYERS:
                if player:
                    player.gold_income(r)
                    self.ai_buy_phase(player, agents[player.player_num], buffer[player.player_num], game_episode)
                    log_to_file(player)
            log_end_turn(r)

            self.combat_phase(self.PLAYERS, r)
            if self.check_dead(agents, buffer, game_episode):
                return True
            log_to_file_combat()

        # Another carousel
        for player in self.PLAYERS:
            if player:
                ran_cost_4 = list(pool_stats.COST_4.items())[random.randint(0, len(pool_stats.COST_4) - 1)][0]
                ran_cost_4 = champion.champion(ran_cost_4,
                                               itemlist=[starting_items[random.randint(0, len(starting_items) - 1)]])
                self.pool_obj.update(ran_cost_4, -1)
                player.add_to_bench(ran_cost_4)

        for r in range(18, 21):
            for player in self.PLAYERS:
                if player:
                    player.gold_income(r)
                    self.ai_buy_phase(player, agents[player.player_num], buffer[player.player_num], game_episode)
                    log_to_file(player)
            log_end_turn(r)

            self.combat_phase(self.PLAYERS, r)
            if self.check_dead(agents, buffer, game_episode):
                return True
            log_to_file_combat()

        # Wolves Round - 3 gold plus 3 item components
        for player in self.PLAYERS:
            if player:
                player.gold += 6
                for _ in range(0, 4):
                    player.add_to_item_bench(starting_items[random.randint(0, len(starting_items) - 1)])

        for r in range(21, 24):
            for player in self.PLAYERS:
                if player:
                    player.gold_income(r)
                    self.ai_buy_phase(player, agents[player.player_num], buffer[player.player_num], game_episode)
                    log_to_file(player)
            log_end_turn(r)

            self.combat_phase(self.PLAYERS, r)
            if self.check_dead(agents, buffer, game_episode):
                return True
            log_to_file_combat()

        # Another carousel
        for player in self.PLAYERS:
            if player:
                ran_cost_5 = list(pool_stats.COST_5.items())[random.randint(0, len(pool_stats.COST_5) - 1)][0]
                item_list = list(full_items.keys())
                ran_cost_5 = champion.champion(ran_cost_5, itemlist=[item_list[random.randint(0, len(item_list) - 1)]])
                self.pool_obj.update(ran_cost_5, -1)
                player.add_to_bench(ran_cost_5)

        for r in range(24, 27):
            # Round 3 to 5 - Buy phase + Combat phase
            for player in self.PLAYERS:
                if player:
                    player.gold_income(r)
                    self.ai_buy_phase(player, agents[player.player_num], buffer[player.player_num], game_episode)
                    log_to_file(player)
            log_end_turn(r)

            self.combat_phase(self.PLAYERS, r)
            if self.check_dead(agents, buffer, game_episode):
                return True
            log_to_file_combat()

        # Dragon Round - 3 gold plus 3 item components
        for player in self.PLAYERS:
            if player:
                player.gold += 6
                item_list = list(full_items.keys())
                player.add_to_item_bench(item_list[random.randint(0, len(item_list) - 1)])

        for r in range(27, 30):
            for player in self.PLAYERS:
                if player:
                    player.gold_income(r)
                    self.ai_buy_phase(player, agents[player.player_num], buffer[player.player_num], game_episode)
                    log_to_file(player)
            log_end_turn(r)

            self.combat_phase(self.PLAYERS, r)
            if self.check_dead(agents, buffer, game_episode):
                return True
            log_to_file_combat()

        # Another carousel
        for player in self.PLAYERS:
            if player:
                ran_cost_5 = list(pool_stats.COST_5.items())[random.randint(0, len(pool_stats.COST_5) - 1)][0]
                item_list = list(full_items.keys())
                ran_cost_5 = champion.champion(ran_cost_5, itemlist=[item_list[random.randint(0, len(item_list) - 1)]])
                self.pool_obj.update(ran_cost_5, -1)
                player.add_to_bench(ran_cost_5)

        for r in range(30, 33):
            for player in self.PLAYERS:
                if player:
                    player.gold_income(r)
                    self.ai_buy_phase(player, agents[player.player_num], buffer[player.player_num], game_episode)
                    log_to_file(player)
            log_end_turn(r)

            self.combat_phase(self.PLAYERS, r)
            if self.check_dead(agents, buffer, game_episode):
                return True
            log_to_file_combat()

        # Rift Herald - 3 gold plus 3 item components
        for player in self.PLAYERS:
            if player:
                player.gold += 6
                item_list = list(full_items.keys())
                player.add_to_item_bench(item_list[random.randint(0, len(item_list) - 1)])

        for r in range(33, 36):
            for player in self.PLAYERS:
                if player:
                    player.gold_income(r)
                    self.ai_buy_phase(player, agents[player.player_num], buffer[player.player_num], game_episode)
                    log_to_file(player)
            log_end_turn(r)

            self.combat_phase(self.PLAYERS, r)
            if self.check_dead(agents, buffer, game_episode):
                return True

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
    if config.LOGMESSAGES:
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
