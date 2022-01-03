import pool
from item_stats import item_builds as full_items, basic_items, starting_items
import pool_stats
import replay_buffer
import AI_interface
import dqn_agent
import random
import champion
from player import player as player_class
from interface import interface
from champion_functions import MILLIS

# Amount of damage taken as a base per round. First number is max round, second is damage
ROUND_DAMAGE = [
	[3, 0], 
	[9, 2], 
	[15, 3], 
	[21, 5],
	[27, 8], 
	[10000, 15]
]


def log_to_file_start():
	with open('log.txt', "w") as out:
		out.write("Start of a new run")


pool_obj = pool.pool()
num_players = 8
PLAYERS = [ player_class(pool_obj, i) for i in range( num_players ) ]
NUM_DEAD = 0
log_to_file_start()



# Log the info from the players to a log file.
def log_to_file(player):
	with open('log.txt', "a") as out:
		for line in player.log:
			out.write(str(line))
			out.write('\n')
		player.log = []


# This one is for the champion and logging the battles.
def log_to_file_combat():
	with open('log.txt', "a") as out:
		if len(champion.log) > 0:
			if(MILLIS() < 75000):
				if(champion.log[-1] == 'BLUE TEAM WON'): champion.test_multiple['blue'] += 1
				if(champion.log[-1] == 'RED TEAM WON'): champion.test_multiple['red'] += 1
			elif(MILLIS() < 200000): champion.test_multiple['draw'] += 1
			for line in champion.log:
				out.write(str(line))
				out.write('\n')
			champion.log = []


most_recent_dead = None
def combat_phase(players, player_round):
	# Pseudocode --> 
	# This feels like it should be easy but there are many unknown steps.
	# I don't want to copy the players array because it may be large and don't want to cause that memory problem.
	# I can start with the first player and pick a random number from 1 to 7. 
	# If that number is not the player that you played last round, play that player. 
	# This is a little unideal because you can play the same 2 players over and over again but this is unlikely and be corrected later.
	# Do this for the length of the array, any players that has to play that last a random player but that player loses no health.
	# I can assume that all players have more 0 health at the start of the combat if I check the health at the end of combat.
	# The order of the players shouldn't matter to any other part of the game.

	# TO DO LATER: implement a fix for when 6 players are matched and the 7th needs to go against their last opponent.
	random.shuffle(players)
	player_nums = list(range(0, len(players)))
	players_matched = 0
	round_index = 0
	while(player_round > ROUND_DAMAGE[round_index][0]): round_index += 1
	# print("Round_damage = " + str(ROUND_DAMAGE[round_index][1]) + " with round_index = " + str(round_index))
	for player in players:
		if player:
			player.combat = False
	for num in player_nums:
		# make sure I am dealing with one of the players who has yet to fight.
		if players[num] and players[num].combat == False:
			# If there is more than one player left ot be matched.
			if players_matched < num_players - 1 - NUM_DEAD:
				# The player to match is always going to be a higher number than the first player.
				player_index = random.randint(0, len(players) - 1)
				# if the index is not in the player_nums, then it shouldn't check the second part. 
				# Although the index is always going to be the index of some.
				# Make sure the player is alive as well.
				# print("initial num = " + str(num) + ", and player_index = " + str(player_index))
				while((not players[player_index]) or players[num].opponite == players[player_index] or players[player_index].combat or num == player_index):
					# if not players[player_index]:
					# 	print("option 0 - num = " + str(num) + ", and player_index = " + str(player_index) + " and players_matched = " + str(players_matched))
					# elif players[num].opponite == players[player_index]:
					# 	print("option 1 - num = " + str(num) + ", and player_index = " + str(player_index) + " and players_matched = " + str(players_matched))
					# elif players[player_index].combat:
					# 	print("option 2 - num = " + str(num) + ", and player_index = " + str(player_index) + " and players_matched = " + str(players_matched))
					# elif num == player_index:
					# 	print("option 3 - num = " + str(num) + ", and player_index = " + str(player_index) + " and players_matched = " + str(players_matched))
					player_index = random.randint(0, len(players) - 1)
					# print("num = " + str(num) + ", and player_index = " + str(player_index) + " and players_matched = " + str(players_matched))
					if (players[player_index] and (players_matched == num_players - 2 - NUM_DEAD) and players[num].opponite == players[player_index]):
						# print("broke out")
						break
				players[num].opponite = players[player_index]
				players[player_index].opponite = players[num]
				index_won, damage = champion.run(champion.champion, players[num], players[player_index], ROUND_DAMAGE[round_index][1])
				# print("final match for round " + str(player_round) + " is num = " + str(num) + ", and player_index = " + str(player_index))
				if index_won == 0:
					players[player_index].health -= damage
					players[num].health -= damage
				if index_won == 1:
					players[player_index].health -= damage
				if index_won == 2:
					players[num].health -= damage
				players[player_index].combat = True
				players[num].combat = True
				players_matched += 2
				# print("players " + str(player_index) + ", and " + str(num) + " matched")
			elif len(player_nums) == 1 or players_matched == num_players - 1 - NUM_DEAD:
				# print("Playing the random dude")
				player_index = random.randint(0, len(players) - 1)
				while((not players[player_index]) or players[num].opponite == players[player_index] or num == player_index):
					player_index = random.randint(0, len(players) - 1)
				index_won, damage = champion.run(champion.champion, players[num], players[player_index], ROUND_DAMAGE[round_index][1])
				if index_won == 2 or index_won == 0:
					players[num].health -= damage
				players[num].combat = True
				players_matched += 1
			else:
				return False
	log_to_file_combat()
	return True


def check_dead():
	global NUM_DEAD
	num_alive = 0
	for i, player in enumerate(PLAYERS):
		if player:
			# print("player " + str(i) + " has health = " + str(player.health))
			if player.health <= 0:
				# This won't take into account how much health the most recent dead had if multiple players die at once
				# But this should be good enough for now.
				print("Player achieved reward this game equal to " + str(player.reward))
				NUM_DEAD += 1
				PLAYERS[i] = None
			else:
				num_alive += 1
	if num_alive == 1:
		for i, player in enumerate(PLAYERS):
			if player:
				print("Player achieved reward this game equal to " + str(player.reward))
				player.won_game()
				print("PLAYER {} WON".format(player.player_num))
				return True
	return False


def game_end_buffer(buffers):
	for buffer in buffers:
		last_item = list(buffer.gameplay_experiences.pop())
		last_item[4] = True
		# last_item[3] = None
		buffer.gameplay_experiences.append(last_item)
	return True


def ai_buy_phase(player, agent, buffer):
	# Generate a shop for the opservation to use
	shop = pool_obj.sample(player, 5)
	# Take an observation
	observation = AI_interface.observation(shop, player)
	# print("NEXT PLAYERS TURN " + str(player.player_num))
	step_done = False
	while not step_done:
		# Get action from the policy network
		action = agent.collect_policy(observation)
		# Take a step
		shop, step_done = AI_interface.step(action, player, shop, pool_obj)
		# Get reward
		reward = AI_interface.reward(player)
		# print("reward is equal to " + str(AI_interface.reward(player)))
		# Get following state
		next_observation = AI_interface.observation(shop, player)
		# Store experience to buffer
		buffer.store_replay_buffer(observation, action, reward, next_observation, False)
		observation = next_observation


def human_game_logic():
	interface_obj = interface()
	# ROUND 0 - Give a random 1 cost unit with item. Add random item at end of round
	# items here is a method of the list and not related to the ingame items.
	# TO DO - Give a different 1 cost unit to each players instead of a random one to each player
	# TO DO MUCH LATER - Add randomness to the drops, 3 gold one round vs 3 1 star units vs 3 2 star units and so on.
	for player in PLAYERS:
		# print(random.randint(0, len(pool_stats.COST_1)))
		ran_cost_1 = list(pool_stats.COST_1.items())[random.randint(0, len(pool_stats.COST_1) - 1)][0]
		ran_cost_1 = champion.champion(ran_cost_1)
		ran_cost_1.add_item(starting_items[random.randint(0, len(starting_items) - 1)])
		player.add_to_bench(ran_cost_1)
		log_to_file(player)
		player.add_to_item_bench(starting_items[random.randint(0, len(starting_items) - 1)])

	# ROUND 1 - Buy phase + Give 1 item component and 1 random 3 cost champion 
	for player in PLAYERS:
		interface_obj.outer_loop(pool_obj, player, 1)
		log_to_file(player)
		player.add_to_item_bench(starting_items[random.randint(0, len(starting_items) - 1)])
		ran_cost_3 = list(pool_stats.COST_3.items())[random.randint(0, len(pool_stats.COST_3) - 1)][0]
		ran_cost_3 = champion.champion(ran_cost_3)
		player.add_to_bench(ran_cost_3)


	# Round 2 -  Buy phase + Give 3 gold and 1 random item component
	for player in PLAYERS:
		interface_obj.outer_loop(pool_obj, player, 2)
		log_to_file(player)
		player.add_to_item_bench(starting_items[random.randint(0, len(starting_items) - 1)])
		player.gold += 3


	for r in range(3, 6):
	# Round 3 to 5 - Buy phase + Combat phase
		for player in PLAYERS:
			interface_obj.outer_loop(pool_obj, player, r)
			log_to_file(player)

		combat_phase(PLAYERS, r)
		if check_dead(): return True
		log_to_file_combat()

	# Round 6 - random 3 drop with item + Combat phase
	# (Carosell round)
	for player in PLAYERS:
		ran_cost_3 = list(pool_stats.COST_3.items())[random.randint(0, len(pool_stats.COST_3) - 1)][0]
		ran_cost_3 = champion.champion(ran_cost_3)
		ran_cost_3.add_item(basic_items[random.randint(0, len(basic_items) - 1)])
		player.add_to_bench(ran_cost_3)

	for r in range(6, 9):
		for player in PLAYERS:
			interface_obj.outer_loop(pool_obj, player, r)
			log_to_file(player)

		combat_phase(PLAYERS, r)
		if check_dead(): return True
		log_to_file_combat()

	# Golum Round - 3 gold plus 3 item components
	for player in PLAYERS:
		player.gold += 3
		for _ in range(0, 3):
			player.add_to_item_bench(starting_items[random.randint(0, len(starting_items) - 1)])

	for r in range(9, 12):
		for player in PLAYERS:
			interface_obj.outer_loop(pool_obj, player, r)
			log_to_file(player)

		combat_phase(PLAYERS, r)
		if check_dead(): return True
		log_to_file_combat()

	# Another carosell
	for player in PLAYERS:
		ran_cost_3 = list(pool_stats.COST_3.items())[random.randint(0, len(pool_stats.COST_3) - 1)][0]
		ran_cost_3 = champion.champion(ran_cost_3)
		ran_cost_3.add_item(basic_items[random.randint(0, len(basic_items) - 1)])
		player.add_to_bench(ran_cost_3)

	for r in range(9, 12):
		for player in PLAYERS:
			interface_obj.outer_loop(pool_obj, player, r)
			log_to_file(player)

		combat_phase(PLAYERS, r)
		if check_dead(): return True
		log_to_file_combat()

	for player in PLAYERS:
		player.gold += 3
		for _ in range(0, 3):
			player.add_to_item_bench(starting_items[random.randint(0, len(starting_items) - 1)])

	for r in range(12, 15):
		for player in PLAYERS:
			interface_obj.outer_loop(pool_obj, player, r)
			log_to_file(player)

		combat_phase(PLAYERS, r)
		if check_dead(): return True
		log_to_file_combat()

	# Another carosell
	for player in PLAYERS:
		ran_cost_4 = list(pool_stats.COST_4.items())[random.randint(0, len(pool_stats.COST_4) - 1)][0]
		ran_cost_4 = champion.champion(ran_cost_4)
		ran_cost_4.add_item(basic_items[random.randint(0, len(basic_items) - 1)])
		player.add_to_bench(ran_cost_4)

	for r in range(15, 18):
		for player in PLAYERS:
			interface_obj.outer_loop(pool_obj, player, r)
			log_to_file(player)

		combat_phase(PLAYERS, r)
		if check_dead(): return True
		log_to_file_combat()

	# Wolves Round - 3 gold plus 3 item components
	for player in PLAYERS:
		player.gold += 6
		for _ in range(0, 4):
			player.add_to_item_bench(starting_items[random.randint(0, len(starting_items) - 1)])

	for r in range(18, 21):
	# Round 3 to 5 - Buy phase + Combat phase
		for player in PLAYERS:
			interface_obj.outer_loop(pool_obj, player, r)
			log_to_file(player)

		combat_phase(PLAYERS, r)
		if check_dead(): return True
		log_to_file_combat()

	# Another carosell
	for player in PLAYERS:
		ran_cost_5 = list(pool_stats.COST_5.items())[random.randint(0, len(pool_stats.COST_4) - 1)][0]
		ran_cost_5 = champion.champion(ran_cost_5)
		item_list = list(full_items.keys())
		ran_cost_5.add_item(item_list[random.randint(0, len(item_list) - 1)])
		player.add_to_bench(ran_cost_5)

	for r in range(21, 24):
	# Round 3 to 5 - Buy phase + Combat phase
		for player in PLAYERS:
			interface_obj.outer_loop(pool_obj, player, r)
			log_to_file(player)

		combat_phase(PLAYERS, r)
		if check_dead(): return True
		log_to_file_combat()

	# Dragon Round - 3 gold plus 3 item components
	for player in PLAYERS:
		player.gold += 6
		item_list = list(full_items.keys())
		player.add_to_item_bench(item_list[random.randint(0, len(item_list) - 1)])

	for r in range(24, 27):
	# Round 3 to 5 - Buy phase + Combat phase
		for player in PLAYERS:
			interface_obj.outer_loop(pool_obj, player, r)
			log_to_file(player)

		combat_phase(PLAYERS, r)
		if check_dead(): return True
		log_to_file_combat()

	# Another carosell
	for player in PLAYERS:
		ran_cost_5 = list(pool_stats.COST_5.items())[random.randint(0, len(pool_stats.COST_4) - 1)][0]
		ran_cost_5 = champion.champion(ran_cost_5)
		item_list = list(full_items.keys())
		ran_cost_5.add_item(item_list[random.randint(0, len(item_list) - 1)])
		player.add_to_bench(ran_cost_5)

	for r in range(27, 30):
		for player in PLAYERS:
			interface_obj.outer_loop(pool_obj, player, r)
			log_to_file(player)

		combat_phase(PLAYERS, r)
		if check_dead(): return True
		log_to_file_combat()

	# Rift Herald - 3 gold plus 3 item components
	for player in PLAYERS:
		player.gold += 6
		item_list = list(full_items.keys())
		player.add_to_item_bench(item_list[random.randint(0, len(item_list) - 1)])

	for r in range(30, 33):
	# Round 3 to 5 - Buy phase + Combat phase
		for player in PLAYERS:
			interface_obj.outer_loop(pool_obj, player, r)
			log_to_file(player)

		combat_phase(PLAYERS, r)
		if check_dead(): return True

	print("Game has gone on way too long. There has to be a bug somewhere")
	for player in PLAYERS:
		print(player.health)
	return False


# This is going to take a list of agents, 1 for each player
# This is also taking in a list of buffers, 1 for each player.
# TO DO: Add an additional set of buffers at the end for the ending state of each player
def episode(agent, buffer):
	# ROUND 0 - Give a random 1 cost unit with item. Add random item at end of round
	# items here is a method of the list and not related to the ingame items.
	# TO DO - Give a different 1 cost unit to each players instead of a random one to each player
	# TO DO MUCH LATER - Add randomness to the drops, 3 gold one round vs 3 1 star units vs 3 2 star units and so on.
	# Update this whenever a player dies with their player class.
	
	for player in PLAYERS:
		if player:
			# print(random.randint(0, len(pool_stats.COST_1)))
			ran_cost_1 = list(pool_stats.COST_1.items())[random.randint(0, len(pool_stats.COST_1) - 1)][0]
			ran_cost_1 = champion.champion(ran_cost_1)
			ran_cost_1.add_item(starting_items[random.randint(0, len(starting_items) - 1)])
			player.add_to_bench(ran_cost_1)
			log_to_file(player)
			player.add_to_item_bench(starting_items[random.randint(0, len(starting_items) - 1)])

	# ROUND 1 - Buy phase + Give 1 item component and 1 random 3 cost champion 
	for player in PLAYERS:
		if player:
			ai_buy_phase(player, agent, buffer[player.player_num])
			log_to_file(player)
			player.add_to_item_bench(starting_items[random.randint(0, len(starting_items) - 1)])
			ran_cost_3 = list(pool_stats.COST_3.items())[random.randint(0, len(pool_stats.COST_3) - 1)][0]
			ran_cost_3 = champion.champion(ran_cost_3)
			player.add_to_bench(ran_cost_3)

	# Round 2 -  Buy phase + Give 3 gold and 1 random item component
	for player in PLAYERS:
		if player:
			ai_buy_phase(player, agent, buffer[player.player_num])
			log_to_file(player)
			player.add_to_item_bench(starting_items[random.randint(0, len(starting_items) - 1)])
			player.gold += 3

	for r in range(3, 6):
		# Round 3 to 5 - Buy phase + Combat phase
		for player in PLAYERS:
			if player: 
				ai_buy_phase(player, agent, buffer[player.player_num])
				log_to_file(player)

		combat_phase(PLAYERS, r)
		if check_dead():
			return game_end_buffer(buffer)
		log_to_file_combat()

	# Round 6 - random 3 drop with item + Combat phase
	# (Carosell round)
	for player in PLAYERS:
		ran_cost_3 = list(pool_stats.COST_3.items())[random.randint(0, len(pool_stats.COST_3) - 1)][0]
		ran_cost_3 = champion.champion(ran_cost_3)
		ran_cost_3.add_item(basic_items[random.randint(0, len(basic_items) - 1)])
		player.add_to_bench(ran_cost_3)

	for r in range(6, 9):
		for player in PLAYERS:
			if player:
				ai_buy_phase(player, agent, buffer[player.player_num])
				log_to_file(player)

		combat_phase(PLAYERS, r)
		if check_dead():
			return game_end_buffer(buffer)
		log_to_file_combat()

	# Golum Round - 3 gold plus 3 item components
	for player in PLAYERS:
		if player:
			player.gold += 3
			for _ in range(0, 3):
				player.add_to_item_bench(starting_items[random.randint(0, len(starting_items) - 1)])

	for r in range(9, 12):
		for player in PLAYERS:
			if player:
				ai_buy_phase(player, agent, buffer[player.player_num])
				log_to_file(player)

		combat_phase(PLAYERS, r)
		if check_dead():
			return game_end_buffer(buffer)
		log_to_file_combat()

	# Another carousel
	for player in PLAYERS:
		if player:
			ran_cost_3 = list(pool_stats.COST_3.items())[random.randint(0, len(pool_stats.COST_3) - 1)][0]
			ran_cost_3 = champion.champion(ran_cost_3)
			ran_cost_3.add_item(basic_items[random.randint(0, len(basic_items) - 1)])
			player.add_to_bench(ran_cost_3)

	for r in range(12, 15):
		for player in PLAYERS:
			if player:
				ai_buy_phase(player, agent, buffer[player.player_num])
				log_to_file(player)

		combat_phase(PLAYERS, r)
		if check_dead():
			return game_end_buffer(buffer)
		log_to_file_combat()

	for player in PLAYERS:
		if player:
			player.gold += 3
			for _ in range(0, 3):
				player.add_to_item_bench(starting_items[random.randint(0, len(starting_items) - 1)])

	for r in range(15, 18):
		for player in PLAYERS:
			if player:
				ai_buy_phase(player, agent, buffer[player.player_num])
				log_to_file(player)

		combat_phase(PLAYERS, r)
		if check_dead():
			return game_end_buffer(buffer)
		log_to_file_combat()

	# Another carousel
	for player in PLAYERS:
		if player:
			ran_cost_4 = list(pool_stats.COST_4.items())[random.randint(0, len(pool_stats.COST_4) - 1)][0]
			ran_cost_4 = champion.champion(ran_cost_4)
			ran_cost_4.add_item(basic_items[random.randint(0, len(basic_items) - 1)])
			player.add_to_bench(ran_cost_4)

	for r in range(18, 21):
		for player in PLAYERS:
			if player:
				ai_buy_phase(player, agent, buffer[player.player_num])
				log_to_file(player)

		combat_phase(PLAYERS, r)
		if check_dead():
			return game_end_buffer(buffer)
		log_to_file_combat()

	# Wolves Round - 3 gold plus 3 item components
	for player in PLAYERS:
		if player:
			player.gold += 6
			for _ in range(0, 4):
				player.add_to_item_bench(starting_items[random.randint(0, len(starting_items) - 1)])

	for r in range(21, 24):
		for player in PLAYERS:
			if player:
				ai_buy_phase(player, agent, buffer[player.player_num])
				log_to_file(player)

		combat_phase(PLAYERS, r)
		if check_dead():
			return game_end_buffer(buffer)
		log_to_file_combat()

	# Another carousel
	for player in PLAYERS:
		if player:
			ran_cost_5 = list(pool_stats.COST_5.items())[random.randint(0, len(pool_stats.COST_5) - 1)][0]
			ran_cost_5 = champion.champion(ran_cost_5)
			item_list = list(full_items.keys())
			ran_cost_5.add_item(item_list[random.randint(0, len(item_list) - 1)])
			player.add_to_bench(ran_cost_5)

	for r in range(24, 27):
		# Round 3 to 5 - Buy phase + Combat phase
		for player in PLAYERS:
			if player:
				ai_buy_phase(player, agent, buffer[player.player_num])
				log_to_file(player)

		combat_phase(PLAYERS, r)
		if check_dead():
			return game_end_buffer(buffer)
		log_to_file_combat()

	# Dragon Round - 3 gold plus 3 item components
	for player in PLAYERS:
		if player:
			player.gold += 6
			item_list = list(full_items.keys())
			player.add_to_item_bench(item_list[random.randint(0, len(item_list) - 1)])

	for r in range(27, 30):
		for player in PLAYERS:
			if player:
				ai_buy_phase(player, agent, buffer[player.player_num])
				log_to_file(player)

		combat_phase(PLAYERS, r)
		if check_dead():
			return game_end_buffer(buffer)
		log_to_file_combat()

	# Another carousel
	for player in PLAYERS:
		if player:
			ran_cost_5 = list(pool_stats.COST_5.items())[random.randint(0, len(pool_stats.COST_5) - 1)][0]
			ran_cost_5 = champion.champion(ran_cost_5)
			item_list = list(full_items.keys())
			ran_cost_5.add_item(item_list[random.randint(0, len(item_list) - 1)])
			player.add_to_bench(ran_cost_5)

	for r in range(30, 33):
		for player in PLAYERS:
			if player:
				ai_buy_phase(player, agent, buffer[player.player_num])
				log_to_file(player)

		combat_phase(PLAYERS, r)
		if check_dead():
			return game_end_buffer(buffer)
		log_to_file_combat()

	# Rift Herald - 3 gold plus 3 item components
	for player in PLAYERS:
		if player:
			player.gold += 6
			item_list = list(full_items.keys())
			player.add_to_item_bench(item_list[random.randint(0, len(item_list) - 1)])

	for r in range(33, 36):
		for player in PLAYERS:
			if player:
				ai_buy_phase(player, agent, buffer[player.player_num])
				log_to_file(player)

		combat_phase(PLAYERS, r)
		if check_dead():
			return game_end_buffer(buffer)

	print("Game has gone on way too long. There has to be a bug somewhere")
	for player in PLAYERS:
		if player:
			print(player.health)
	return False
