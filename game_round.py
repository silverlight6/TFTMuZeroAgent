import pool
from item_stats import item_builds as full_items, basic_items, starting_items
import pool_stats
import random
from champion import champion
from player import player as player_class
from interface import interface

# Amount of damage taken as a base per round. First number is max round, second is damage
ROUND_DAMAGE = [
	[3, 0], 
	[9, 2], 
	[15, 3], 
	[21, 5],
	[27, 8], 
	[10000, 7]
]


num_players = 2
PLAYERS = [ player_class() for x in range( num_players ) ]

# Log the info from the players to a log file.
def log_to_file(player):
	with open('log.txt', "w") as out:
        for line in player.log:
            out.write(str(line))
            out.write('\n')
        player.log = []


# This one is for the champion and logging the battles.
def log_to_file():
	with open('log.txt', "w") as out:
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
	random.shuffle(players)
	player_nums = list(range(0, len(players)))
	for num in player_nums:
		# make sure I am dealing with one of the players.
		if players[num]:
			if len(player_nums) != 1:
				player_index = random.randint(num, len(players))
				# Make sure the number exist and the player is not the last opponite
				# Check that there is a unit at that location.
				# if the index is not in the player_nums, then it shouldn't check the second part. 
				# Although the index is always going to be the index of some 
				while((not players[num]) or (not player_index in player_nums) or (players[num].opponite != players[player_index])):
					player_index = random.randint(num, len(players))
				round_index = 0
				while(player_round < ROUND_DAMAGE[round_index[0]]): round_index += 1
				index_won, damage = champion.run(champion.champion, players[num], players[player_index], ROUND_DAMAGE[round_index[1]])
				if index_won == 1:
					players[player_index].health -= damage
				if index_won == 2:
					players[num].health -= damage
				player_nums.delete(player_index)
				player_nums.delete(num)
			elif len(player_nums) == 1:
				round_index = 0
				while(player_round < ROUND_DAMAGE[round_index[0]]): round_index += 1
				index_won, damage = champion.run(champion.champion, players[num], most_recent_dead, ROUND_DAMAGE[round_index[1]])
				if index_won == 2:
					players[num].health -= damage
				player_nums.delete(num)
			else:
				return False
	log_to_file()
	return True


def check_dead():
	num_alive = 0
	for player in PLAYERS:
		if player.health <= 0:
			# This won't take into account how much health the most recent dead had if multiple players die at once
			# But this should be good enough for now.
			most_recent_dead = player
			PLAYERS.delete(player)
		else:
			num_alive += 1
	if(num_alive == 1):
		for player in PLAYERS:
			if player:
				print("PLAYER {} WON".format(player.name))
				return True
	return False

def game_logic():
	interface_obj = interface()
	# ROUND 0 - Give a random 1 cost unit with item. Add random item at end of round
	# items here is a method of the list and not related to the ingame items.
	# TO DO - Give a different 1 cost unit to each players instead of a random one to each player
	# TO DO MUCH LATER - Add randomness to the drops, 3 gold one round vs 3 1 star units vs 3 2 star units and so on.
	for player in PLAYERS:
		# print(random.randint(0, len(pool_stats.COST_1)))
		ran_cost_1 = list(pool_stats.COST_1.items())[random.randint(0, len(pool_stats.COST_1) - 1)][0]
		ran_cost_1 = champion(ran_cost_1)
		ran_cost_1.add_item(starting_items[random.randint(0, len(starting_items) - 1)])
		player.add_to_bench(ran_cost_1)
		log_to_file(player)
		player.add_to_item_bench(starting_items[random.randint(0, len(starting_items) - 1)])

	# ROUND 1 - Buy phase + Give 1 item component and 1 random 3 cost champion 
	for player in PLAYERS:
		interface_obj.outer_loop(player, 1)
		log_to_file(player)
		player.add_to_item_bench(starting_items[random.randint(0, len(starting_items) - 1)])
		ran_cost_3 = list(pool_stats.COST_3.items())[random.randint(0, len(pool_stats.COST_3) - 1)][0]
		ran_cost_3 = champion(ran_cost_3)
		player.add_to_bench(ran_cost_3)


	# Round 2 -  Buy phase + Give 3 gold and 1 random item component
	for player in PLAYERS:
		interface_obj.outer_loop(player, 1)
		log_to_file(player)
		player.add_to_item_bench(starting_items[random.randint(0, len(starting_items) - 1)])
		player.gold += 3


	for r in range(3, 6):
	# Round 3 to 5 - Buy phase + Combat phase
		for player in PLAYERS:
			interface_obj.outer_loop(player, 1)
			log_to_file(player)

		combat_phase(PLAYERS, r)
		if check_dead(): return True

	# Round 6 - random 3 drop with item + Combat phase
	# (Carosell round)
	for player in PLAYERS:
		# print(random.randint(0, len(pool_stats.COST_1)))
		ran_cost_3 = list(pool_stats.COST_3.items())[random.randint(0, len(pool_stats.COST_3) - 1)][0]
		ran_cost_3 = champion(ran_cost_3)
		ran_cost_3.add_item(basic_items[random.randint(0, len(basic_items) - 1)])
		player.add_to_bench(ran_cost_3)

	for r in range(6, 9):
	# Round 3 to 5 - Buy phase + Combat phase
		for player in PLAYERS:
			interface_obj.outer_loop(player, 1)
			log_to_file(player)

		combat_phase(PLAYERS, r)
		if check_dead(): return True

	# Golum Round - 3 gold plus 3 item components
	for player in PLAYERS:
		# print(random.randint(0, len(pool_stats.COST_1)))
		player.gold += 3
		for _ in range(0, 3):
			player.add_to_item_bench(starting_items[random.randint(0, len(starting_items) - 1)])

	for r in range(9, 12):
	# Round 3 to 5 - Buy phase + Combat phase
		for player in PLAYERS:
			interface_obj.outer_loop(player, 1)
			log_to_file(player)

		combat_phase(PLAYERS, r)
		if check_dead(): return True

	# Another carosell
	for player in PLAYERS:
		ran_cost_3 = list(pool_stats.COST_3.items())[random.randint(0, len(pool_stats.COST_3) - 1)][0]
		ran_cost_3 = champion(ran_cost_3)
		ran_cost_3.add_item(basic_items[random.randint(0, len(basic_items) - 1)])
		player.add_to_bench(ran_cost_3)

	for r in range(9, 12):
	# Round 3 to 5 - Buy phase + Combat phase
		for player in PLAYERS:
			interface_obj.outer_loop(player, 1)
			log_to_file(player)

		combat_phase(PLAYERS, r)
		if check_dead(): return True

	# raptors Round - 3 gold plus 3 item components
	for player in PLAYERS:
		# print(random.randint(0, len(pool_stats.COST_1)))
		player.gold += 3
		for _ in range(0, 3):
			player.add_to_item_bench(starting_items[random.randint(0, len(starting_items) - 1)])

	for r in range(12, 15):
	# Round 3 to 5 - Buy phase + Combat phase
		for player in PLAYERS:
			interface_obj.outer_loop(player, 1)
			log_to_file(player)

		combat_phase(PLAYERS, r)
		if check_dead(): return True

	# Another carosell
	for player in PLAYERS:
		ran_cost_4 = list(pool_stats.COST_4.items())[random.randint(0, len(pool_stats.COST_4) - 1)][0]
		ran_cost_4 = champion(ran_cost_4)
		ran_cost_4.add_item(basic_items[random.randint(0, len(basic_items) - 1)])
		player.add_to_bench(ran_cost_4)

	for r in range(15, 18):
	# Round 3 to 5 - Buy phase + Combat phase
		for player in PLAYERS:
			interface_obj.outer_loop(player, 1)
			log_to_file(player)

		combat_phase(PLAYERS, r)
		if check_dead(): return True

	# Wolves Round - 3 gold plus 3 item components
	for player in PLAYERS:
		# print(random.randint(0, len(pool_stats.COST_1)))
		player.gold += 6
		for _ in range(0, 4):
			player.add_to_item_bench(starting_items[random.randint(0, len(starting_items) - 1)])

	for r in range(18, 21):
	# Round 3 to 5 - Buy phase + Combat phase
		for player in PLAYERS:
			interface_obj.outer_loop(player, 1)
			log_to_file(player)

		combat_phase(PLAYERS, r)
		if check_dead(): return True

	# Another carosell
	for player in PLAYERS:
		ran_cost_5 = list(pool_stats.COST_5.items())[random.randint(0, len(pool_stats.COST_4) - 1)][0]
		ran_cost_5 = champion(ran_cost_5)
		item_list = list(full_items.keys())
		ran_cost_5.add_item(item_list[random.randint(0, len(item_list) - 1)])
		player.add_to_bench(ran_cost_5)

	for r in range(21, 24):
	# Round 3 to 5 - Buy phase + Combat phase
		for player in PLAYERS:
			interface_obj.outer_loop(player, 1)
			log_to_file(player)

		combat_phase(PLAYERS, r)
		if check_dead(): return True

	# Dragon Round - 3 gold plus 3 item components
	for player in PLAYERS:
		# print(random.randint(0, len(pool_stats.COST_1)))
		player.gold += 6
		item_list = list(full_items.keys())
		player.add_to_item_bench(item_list[random.randint(0, len(item_list) - 1)])

	for r in range(24, 27):
	# Round 3 to 5 - Buy phase + Combat phase
		for player in PLAYERS:
			interface_obj.outer_loop(player, 1)
			log_to_file(player)

		combat_phase(PLAYERS, r)
		if check_dead(): return True

	# Another carosell
	for player in PLAYERS:
		ran_cost_5 = list(pool_stats.COST_5.items())[random.randint(0, len(pool_stats.COST_4) - 1)][0]
		ran_cost_5 = champion(ran_cost_5)
		item_list = list(full_items.keys())
		ran_cost_5.add_item(item_list[random.randint(0, len(item_list) - 1)])
		player.add_to_bench(ran_cost_5)

	for r in range(27, 30):
	# Round 3 to 5 - Buy phase + Combat phase
		for player in PLAYERS:
			interface_obj.outer_loop(player, 1)
			log_to_file(player)

		combat_phase(PLAYERS, r)
		if check_dead(): return True

	# Rift Herald - 3 gold plus 3 item components
	for player in PLAYERS:
		# print(random.randint(0, len(pool_stats.COST_1)))
		player.gold += 6
		item_list = list(full_items.keys())
		player.add_to_item_bench(item_list[random.randint(0, len(item_list) - 1)])

	for r in range(30, 33):
	# Round 3 to 5 - Buy phase + Combat phase
		for player in PLAYERS:
			interface_obj.outer_loop(player, 1)
			log_to_file(player)

		combat_phase(PLAYERS, r)
		if check_dead(): return True

	print("Game has gone on way too long. There has to be a bug somewhere")
	for player in PLAYERS:
		print(player.health)
	return False