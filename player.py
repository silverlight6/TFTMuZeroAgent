import champion
import pool
import config
import numpy
from item_stats import items as item_list, basic_items, item_builds
from stats import COST
from math import floor
from champion_functions import MILLIS

# Let me create a bit of a TODO list at the top here
# 
class player:
	def __init__(self, player_num=0):

		self.gold = 0
		self.level = 0
		self.exp = 0
		self.health = 100
		self.player_num = player_num

		self.win_streak = 0			# For purposes of gold generation at start of turn
		self.loss_streak = 0		# For purposes of gold generation at start of turn
		
		# array of champions, since order does not matter, can be unordered list
		self.bench = [ None for x in range( 9 ) ] 	
		# Champion array, this is a 8 by 4 array.
		self.board = [ [ None for y in range( 4 ) ] for x in range( 8 ) ]			  
		# List of items, there is no object for this so this is a string array
		self.item_bench = [ None for x in range( 10 ) ] 		

		self.opponite = None		# Other player, player object
		self.opponite_board = None	# Other player's board for combat, not sure if I will use this.
		self.chosen = False			# Does this player have a chosen unit already
		self.log = []

		# I need to comment how this works.
		self.triple_catelog = []
		self.num_units_in_play = 0
		self.max_units = 0
		self.exp_cost = 4
		
		self.level_costs = [0, 2, 6, 10, 20, 36, 56, 80]

		# We have 28 board slots. Each slot has a champion info. 
		# 6 spots for champion. 2 spots for the level. 1 spot for chosen.
		# 6 spots for items. 3 item slots.
		# (6 * 3 + 6 + 2 + 1) * 28 = 756
		self.board_array = np.zeros(756)

		# We have 9 bench slots. Same rules as above
		self.bench_array = np.zeros(243)

		# This time we only need 6 bits per slot with 10 slots
		self.item_array = np.zeros(60)
	

	# Return value for use of pool. 
	# Also I want to treat buying a unit with a full bench as the same as buying and immediately selling it
	def add_to_bench(self, a_champion):	# add champion to reduce confusion over champion from import
		# try to triple second
		self.update_triple_catelog(a_champion)
		if self.bench_full(): 
			sell_champion(a_champion)
			return False
		bench_loc = self.bench_vacency()
		self.bench[bench_loc] = a_champion
		a_champion.bench_loc = bench_loc
		return True


	def add_to_item_bench(self, item):
		if self.item_bench_full(1):
			return False
		bench_loc = self.item_bench_vacency()
		self.item_bench[bench_loc] = item
		

	def bench_full(self):
		for u in self.bench:
			if(not u):
				return False
		return True


	def bench_vacency(self):
		for free_slot, u in enumerate(self.bench):
			if(not u):
				return free_slot
		return False


	def buy_champion(self, a_champion):
		if a_champion.cost > self.gold:
			return False
		self.gold -= a_champion.cost
		self.add_to_bench(a_champion)


	def buy_exp():
		if self.gold < self.exp_cost:
			return False
		self.gold -= 4
		self.exp += 4
		level_up()


	def findItem(self, name):
		for c, i in enumerate(self.item_bench):
			if i == name:
				return c
		return False


	def generate_ai_vector(self, x_i, y_i):
		output_array = np.zeros(27 * (x_i + y_i))
		for x in range(0, x_i):
			for y in range(0, y_i):
				input_array = np.zeros(27)
				if self.board[x][y]:
					# start with champion name
					c_index = list(COST.keys()).index(self.board[x][y].name)
					# This should update the champion name section of the vector
					for z in range(0, 6, -1):
						if c_index > 2 * z:
							input_array[z] = 1
							c_index -= 2 * z
					if self.board[x][y].level == 1:
						input_array(6:7) = [0, 0]
					if self.board[x][y].level == 2:
						input_array(6:7) = [0, 1]
					if self.board[x][y].level == 3:
						input_array(6:7) = [1, 0]
					if self.board[x][y].chosen == True:
						input_array(8) = 1
					else:
						input_array(8) = 0
					input_array(9:27) = np.zeros(18)
					if champion.items:
						for i in range(0, 3):
							if self.board[x][y].items[i]:
								i_index = list(item_list.keys()).index(self.board[x][y].items[i])
								# This should update the item name section of the vector
								for z in range(0, 6, -1):
									if i_index > 2 * z:
										input_array[9 + 6 * (i + 1) - z] = 1
										i_index -= 2 * z 
				output_array(27 * (x + 7 * y): 27 * (x + 7 * y + 1)) = input_array
		return output_array


	# This should update the generate_board_array every time it is called.
	# I should check the time usage of this method later. 
	# I may implement a method that only updates the part of the list that is changed.
	# If time becomes an issue
	def generate_board_array(self):
		output_array = self.generate_ai_vector(7, 4)
		self.board_array = output_array
		return self.board_array


	def generate_bench_array(self):
		output_array = self.generate_ai_vector(9, 1)
		self.bench_array = output_array
		return self.bench_array

	def generate_item_array(self):
		for x in range(0, len(self.item_bench)):
			input_array = np.zeros(6)
			if self.item_bench[x]:
				i_index = list(item_list.keys()).index(self.item_bench[x])
				# This should update the item name section of the vector
				for z in range(0, 6, -1):
					if i_index > 2 * z:
						input_array[6 - z] = 1
						i_index -= 2 * z 
			self.item_array(6 * x: 6* x + 1) = input_array
		return self.item_array


	# This takes every occurance of a champion at a given level and returns 1 of a higher level.
	# Transfers items over. The way I have it would mean it would require bench space.
	def golden(champion):
		x = -1
		y = -1
		chosen = False
		for i in range(0, len(self.bench)):
			if (self.bench[i].name == champion.name and self.bench[i].stars == champion.stars):
				x = i
				if self.bench[i].chosen: chosen = self.bench[i].chosen
				sell_from_bench(i, True)
		for i in range(0, 8):
			for j in range(0, 4):
				if(self.board[i][k].name == champion.name and self.board[i][k].stars == champion.stars):
					x = i
					y = j
					if self.board[i][k].chosen: chosen = self.board[i][k].chosen
					sell_champion(self.board[i][k], True)
		champion.chosen = chosen
		if chosen: champion.new_chosen()
		champion.golden
		add_to_bench(champion)
		if y != -1:
			move_bench_to_board(champion.bench_loc, x, y)
		self.printt("champion {} was made golden".format(champion.name))


	# Including base_exp income here
	def gold_income(self, t_round): # time of round since round is a keyword
		self.exp += 2
		self.level_up()
		if (t_round <= 3):
			self.gold += 3
			self.gold += floor(self.gold / 10)
			return
		self.gold += 5
		self.gold += floor(self.gold / 10)
		if (self.win_streak == 2 or self.win_streak == 3 or self.loss_streak == 2 or self.loss_streak == 3):
			self.gold += 1
		elif (self.win_streak == 3 or self.loss_streak == 3):
			self.gold += 2
		elif (self. win_streak >= 4 or self.loss_streak >= 4):
			self.gold += 3
		return
	

	# num of items to be added to bench, set 0 if not adding.
	# This is going to crash because the item_bench is set to initialize all to NULL but len to 10.
	# I need to redo this to see how many slots within the length of the array are currently full.
	def item_bench_full(self, num_of_items):
		for i in self.item_bench:
			if not i:
				return False
		return True


	def item_bench_vacency(self):
		for free_slot, u in enumerate(self.item_bench):
			if not u:
				return free_slot
		return False
		

	def level_up(self):
		if self.exp > self.level_costs[self.level - 1]:
			self.exp -= self.level_costs[self.level - 1]
			self.level += 1


	# location to pick which unit from bench goes to board.
	def move_bench_to_board(self, location, x, y):
		if self.bench[location] and x < 8 and x >= 0 and y < 4 and y >= 0:
			if self.num_units_in_play < self.max_units:
				m_champion = self.bench.pop(location)
				m_champion.x = x
				m_champion.y = y
				if(self.board[x][y]): move_board_to_bench(x, y)
				self.board[x][y] = m_champion
				return True
		return False


	# automatically put the champion at the end of the open bench
	# Will likely have to deal with azir and other edge cases here.
	# Kinda of the attitude that I will let those issues sting me first and deal with them
	# When they come up and appear.
	def move_board_to_bench(self, x, y):
		if bench_full():
			if self.board[x][y]:
				sell_champion(self.board[x][y])
			return False
		else:
			if self.board[x][y]:
				bench_loc = bench_vacency
				self.bench[bench_loc] = self.board[x][y]
				self.board[x][y] = None
				return True
			else:
				return False

	# TO DO : Item combinations.
	# Move item from item_bench to champion_bench
	def move_item_to_bench(self, xBench, x):
		if self.item_bench[xBench]:
			if self.bench[x]:
				# theives glove exception
				self.print("moving {} to unit {}".format(self.item_bench[xBench], self.bench[x].name))
				if (self.bench[x].num_of_items < 3 and self.item_bench[xBench] != "thiefs_gloves") or 
					(self.bench[x].items[-1] in basic_items and self.item_bench[x] in basic_items and self.bench[x].num_of_items == 3):
					# implement the item combinations here. Make exception with theives gloves
					if self.bench[x].items[-1] in basic_items and self.item_bench[x] in basic_items:
						item_build_values = item_builds.values()
						item_index = 0
						for index, items = enumerate(item_builds_values):
							if((self.bench[x].items[-1] == items[0] and self.item_bench[x] == items[1]) or (self.bench[x].items[-1] == items[1] and self.item_bench[x] == items[0])):
								item_index = index
								break
						if item_builds.keys()[item_index] == "theifs_gloves":
							if self.bench[x].num_of_items != 1:
								return False
							else:
								self.bench[x].num_of_items += 2
						self.item_bench.pop[xBench]
						self.bench[x].append(item_builds.keys()[item_index])
					else:
						self.bench[x].items.append(self.item_bench.pop[xBench])
						self.bench[x].num_of_items += 1
					return True
				elif self.bench[x].num_of_items < 1 and self.item_bench[xBench] == "thiefs_gloves":
					self.bench[x].items.append(self.item_bench.pop[xBench])
					self.bench[x].num_of_items += 3
					return True
				# last case where 3 items but the last item is a basic item and the item to input is also a basic item
		return False


	def move_item_to_board(self, xBench, x, y):
		if self.item_bench[xBench]:
			if self.board[x][y]:
				# theives glove exception
				self.print("moving {} to unit {}".format(self.item_bench[xBench], self.board[x][y].name))
				if (self.board[x][y].num_of_items < 3 and self.item_bench[xBench] != "thiefs_gloves") or 
					(self.board[x][y].items[-1] in basic_items and self.item_bench[x] in basic_items and self.board[x][y].num_of_items == 3):
					# implement the item combinations here. Make exception with theives gloves
					if self.board[x][y].items[-1] in basic_items and self.item_bench[x] in basic_items:
						item_build_values = item_builds.values()
						item_index = 0
						for index, items = enumerate(item_builds_values):
							if((self.board[x][y].items[-1] == items[0] and self.item_bench[x] == items[1]) or (self.board[x][y].items[-1] == items[1] and self.item_bench[x] == items[0])):
								item_index = index
								break
						if item_builds.keys()[item_index] == "theifs_gloves":
							if self.board[x][y].num_of_items != 1:
								return False
							else:
								self.board[x][y].num_of_items += 2
						self.item_bench.pop[xBench]
						self.board[x][y].append(item_builds.keys()[item_index])
					else:
						self.board[x][y].items.append(self.item_bench.pop[xBench])
						self.board[x][y]].num_of_items += 1
					return True
				elif self.board[x][y].num_of_items < 1 and self.item_bench[xBench] == "thiefs_gloves":
					self.board[x][y].items.append(self.item_bench.pop[xBench])
					self.board[x][y].num_of_items += 3
					return True
		return False


	def print(self, msg):
		self.printt('{:<120}'.format('{:<8}'.format(self.player_num) + msg) + str(MILLIS()))


	def printBench(self, log=True):
		for i in bench:
			if i:
				if log:
					i.print()
				else:
					print(i.name + ", ")


	def printItemBench(self, log=True):
		for i in self.item_bench:
			if i:
				if log:
					self.printt('{:<120}'.format('{:<8}',format(self.player_num) + self.item_bench))
				else:
					print('{:<120}'.format('{:<8}',format(self.player_num) + self.item_bench))


	def printt(self, msg):
		if(config.PRINTMESSAGES): self.log.append(msg)
		# if(config.PRINTMESSAGES): print(msg)


	# This is always going to be from the bench
	def return_item_from_bench(self, x):
		# if the unit exists
		if self.bench[x]:
			# skip if there are no items, trying to save a little processing time.
			if self.bench[x].items:
				# if I have enough space on the item bench for the number of items needed
				if (not item_bench_full(self.bench[x].num_of_items)):
					# Each item in posesstion
					for i in self.bench[x][y].items:
						# theives glove exception
						self.item_bench[item_bench_vacency()] = i
				# if there is only one or two spots left on the item_bench and thiefs_gloves is removed
				elif (not item_bench_full(1) and self.bench[x].items[0] == "thiefs_gloves"):
					self.item_bench[item_bench_vacency()] = i
				self.bench[x].items = []
				self.bench[x].num_of_items = 0
			return True
		return False


	def return_item_from_board(self, x, y):
		# if the unit exists
		if self.board[x][y]:
			# skip if there are no items, trying to save a little processing time.
			if self.board[x][y].items:
				# if I have enough space on the item bench for the number of items needed
				if (not item_bench_full(self.board[x][y].num_of_items)):
					# Each item in posesstion
					for i in self.board[x][y].items:
						# theives glove exception
						self.item_bench[item_bench_vacency()] = i
				# if there is only one or two spots left on the item_bench and thiefs_gloves is removed
				elif (not item_bench_full(1) and self.board[x][y].items[0] == "thiefs_gloves"):
					self.item_bench[item_bench_vacency()] = i
				self.board[x][y].items = []
				self.board[x][y].num_of_items = 0
			return True
		return False


	# called when selling a unit
	def remove_triple_catelog(self, champion):
		for c, i in enumerate(self.triple_catelog):
			if (i.name == champion.name and i.level == champion.level):
				i.num -= 1
				if i.num == 0:
					self.triple_catelog.pop(c)
					return True
		return False


	# This should only be called when trying to sell a champion from the field and the bench is full
	# This can occur after a carosell round where you get a free champion and it can enter the field
	# Even if you already have too many units in play. The default behavior will be sell that champion.
	def sell_champion(self, s_champion, golden=False):	# sell champion to reduce confusion over champion from import
		# Need to add the behavior that on carosell when bench is full, add to board.
		if (not remove_triple_catelog(s_champion) or not return_item_from_board(s_champion.x, s_champion.y)):
			return False
		if (not golden):
			self.gold += s_champion.cost
			if s_champion.chosen: self.chosen = False
		self.board[s_champion.x][s_champion.y] = None
		self.num_units_in_play -= 1
		pool.update(s_champion, 1)
		return True


	def sell_from_bench(self, location, golden=False):
		# Check if champion has items
		# Are there any champions with special abilities on sell.
		if self.bench[location]:
			if (not remove_triple_catelog(bench[location]) or not return_item_from_bench(location)):
				return False
			if (not golden):
				self.gold += self.bench[location].cost
				if self.bench[location].chosen: self.chosen = False
			# Update the pool if necessary
			pool.update(self.bench[location], 1)
			return self.bench.pop(location)
		return False


	def update_triple_catelog(self, champion):
		for entry in self.triple_catelog:
			print(entry["name"])
			if (entry["name"] == champion.name and entry["level"] == champion.level):
				entry["num"] += 1
				if entry["num"] == 3:
					champion = golden(champion)
					update_triple_catelog(champion)
					return
		self.triple_catelog.append({"name": champion.name, "level": champion.stars, "num": 1})
		print("adding " + champion.name + " to triple_catelog")
			