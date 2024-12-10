import random
import Simulator.stats as stats

from Simulator.config import LOGMESSAGES
from Simulator.origin_class_stats import origin_class, chosen_exclude
from Simulator.pool_stats import *


class pool:
	def __init__(self):
		self.num_cost_1 = 0
		self.num_cost_2 = 0
		self.num_cost_3 = 0
		self.num_cost_4 = 0
		self.num_cost_5 = 0
		self.reset()
		self.update_stats(allV=True)

	def log_to_file_pool(self):
		if LOGMESSAGES:
			with open('log.txt', "a") as out:
				out.write("COST 1 \n")
				for key, value in COST_1.items():
					out.write('%s:%s\n' % (key, value))
				out.write("COST 2 \n")
				for key, value in COST_2.items():
					out.write('%s:%s\n' % (key, value))
				out.write("COST 3 \n")
				for key, value in COST_3.items():
					out.write('%s:%s\n' % (key, value))
				out.write("COST 4 \n")
				for key, value in COST_4.items():
					out.write('%s:%s\n' % (key, value))
				out.write("COST 5 \n")
				for key, value in COST_5.items():
					out.write('%s:%s\n' % (key, value))

	def pick_chosen(self, champion_name):
		chosen_type = random.choice(origin_class[champion_name])
		counter = 0
		while chosen_type in chosen_exclude:
			counter += 1
			if counter >= 50:
				chosen_type = None
				print("I should never be here with champion {}".format(champion_name))
				break
			# every champion has at least one type that is acceptable to pick so it isn't going to infinite loop here
			chosen_type = random.choice(origin_class[champion_name])
			# Putting this here just in case something like that happens.
			# print("could not pick " + chosen_type + " type")
		return chosen_type

	def reset(self):
		self.COST_1 = COST_1.copy()
		self.COST_2 = COST_2.copy()
		self.COST_3 = COST_3.copy()
		self.COST_4 = COST_4.copy()
		self.COST_5 = COST_5.copy()

		# for key in COST_1:
		# 	COST_1[key] = base_pool_values[0]
		# for key in COST_2:
		# 	COST_2[key] = base_pool_values[1]
		# for key in COST_3:
		# 	COST_3[key] = base_pool_values[2]
		# for key in COST_4:
		# 	COST_4[key] = base_pool_values[3]
		# for key in COST_5:
		# 	COST_5[key] = base_pool_values[4]

	# Used when a player dies.
	def return_hero(self, player):
		# return the board
		for i in range(len(player.board)):
			for k in range(len(player.board[0])):
				if player.board[i][k]:
					self.update_pool(player.board[i][k], 1)
		for i in range(len(player.bench)):
			if player.bench[i]:
				self.update_pool(player.bench[i], 1)

	# Player is the player that the sample is being picked for
	# Num is the number of samples to be returned
	# Index is the level of the champion you want to be sampled, -1 for random or to follow level.
	# Chosen is implemented as a string with the class being the possible one.
	def sample(self, player, num, idx=-1, allow_chosen=True):
		# If player is None, for example they died, return an empty shop
		if player is None:
			return [" " for _ in range(num)]
		ranInt = [0 for _ in range(num)]
		championOptions = [None for _ in range(num)]
		chosen = player.chosen or not allow_chosen
		chosen_index = -1
		if not chosen:
			if random.random() < .5:
				chosen_index = random.randint(0, 4)
		index = idx
		for i in range(0, num):
			if chosen_index != i:
				percents = level_percentage[player.level]
			else:
				percents = chosen_stats[player.level]
			ranInt[i] = random.random()
			if index == -1:
				index = 0
				while ranInt[i] > percents[index]:
					index += 1
					if index > 4:
						print("ERROR with ranInt[i] = " + str(ranInt[i]) + " and player level" + str(player.level))
						break
			counter = 0
			counterIndex = 0

			# cost 1
			if index == 0:
				# Get a list of all the champions in the pool
				cost_1 = list(self.COST_1.values())
				# Pick a random number to look for the position of that champion
				ranPoolInt = random.randint(0, self.num_cost_1 - 1)
				# Until the counter is greater than the number of champions checked
				while counter < ranPoolInt:
					# Increment counter to the next champion
					counter += cost_1[counterIndex]
					# Increment the champion index
					counterIndex += 1
					# If we sample the last champion in the list
					if counterIndex == len(cost_1):
						break
				# Get a list of the champion names
				keys_list = list(self.COST_1)
				# Set the option to be the champion name of choice
				championOptions[i] = keys_list[counterIndex - 1]

			# cost 2
			elif index == 1:
				cost_2 = list(self.COST_2.values())
				ranPoolInt = random.randint(0, self.num_cost_2 - 1)
				while counter < ranPoolInt:
					counter += cost_2[counterIndex]
					counterIndex += 1
					if counterIndex == len(cost_2):
						break
				keys_list = list(self.COST_2)
				championOptions[i] = keys_list[counterIndex - 1]

			# cost 3
			elif index == 2:
				cost_3 = list(self.COST_3.values())
				ranPoolInt = random.randint(0, self.num_cost_3 - 1)
				while counter < ranPoolInt:
					counter += cost_3[counterIndex]
					counterIndex += 1
					if counterIndex == len(cost_3):
						break
				keys_list = list(self.COST_3)
				championOptions[i] = keys_list[counterIndex - 1]

			# cost 4
			elif index == 3:
				cost_4 = list(self.COST_4.values())
				ranPoolInt = random.randint(0, self.num_cost_4 - 1)
				while counter < ranPoolInt:
					counter += cost_4[counterIndex]
					counterIndex += 1
					if counterIndex == len(cost_4):
						break
				keys_list = list(self.COST_4)
				championOptions[i] = keys_list[counterIndex - 1]

			# cost 5
			else:
				cost_5 = list(self.COST_5.values())
				ranPoolInt = random.randint(0, self.num_cost_5 - 1)
				while counter < ranPoolInt:
					if counterIndex == len(cost_5):
						break
					counter += cost_5[counterIndex]
					counterIndex += 1
				keys_list = list(self.COST_5)
				championOptions[i] = keys_list[counterIndex - 1]
			# This adds the chosen aspect to the champion.
			if chosen_index == i:
				chosen_type = self.pick_chosen(championOptions[i])
				championOptions[i] = str(championOptions[i]) + "_" + chosen_type + "_c"
				# player.print("Offering chosen unit {} at index {}".format(championOptions[i], index))
			index = idx
		return championOptions

	def trait_sample(self, player, num, trait, idx=-1, allow_chosen=True):
		if num > 1 or random.random() < .5:
			return self.sample(player, num, idx, allow_chosen)
		else:
			# TODO: Add chosen capabilities
			unit_list = [key for key, traits in origin_class.items() if trait in traits]
			return [random.choice(unit_list)]

	def update_pool(self, u_champion, direction):
		"""
		:param u_champion: everytime a champion is taken from the pool, update the statistics
		:param direction: positive for adding and negative for selling from pool.
		i.e. If a player sells a unit, it's a positive direction for the pool
		:return: NA
		"""
		# This line is actually a little faster than combining this line and cost = stats.COST[u_champion.name]
		# because it does not require loading the array for champion costs to search for the value.
		cost = u_champion.cost
		quantity = 3 ** (u_champion.stars - 1) * direction
		if u_champion.stars != 1:
			cost = stats.COST[u_champion.name]
		if cost == 1:
			self.COST_1[u_champion.name] += quantity
			if self.COST_1[u_champion.name] < 0:
				self.COST_1[u_champion.name] = 0
			elif self.COST_1[u_champion.name] > base_pool_values[0]:
				self.COST_1[u_champion.name] = base_pool_values[0]
			self.update_stats(one=True)
		elif cost == 2:
			self.COST_2[u_champion.name] += quantity
			if self.COST_2[u_champion.name] < 0:
				self.COST_2[u_champion.name] = 0
			elif self.COST_2[u_champion.name] > base_pool_values[1]:
				self.COST_2[u_champion.name] = base_pool_values[1]
			self.update_stats(two=True)
		elif cost == 3:
			self.COST_3[u_champion.name] += quantity
			if self.COST_3[u_champion.name] < 0:
				self.COST_3[u_champion.name] = 0
			elif self.COST_3[u_champion.name] > base_pool_values[2]:
				self.COST_3[u_champion.name] = base_pool_values[2]
			self.update_stats(three=True)
		elif cost == 4:
			self.COST_4[u_champion.name] += quantity
			if self.COST_4[u_champion.name] < 0:
				self.COST_4[u_champion.name] = 0
			elif self.COST_4[u_champion.name] > base_pool_values[3]:
				self.COST_4[u_champion.name] = base_pool_values[3]
			self.update_stats(four=True)
		elif cost == 5:
			self.COST_5[u_champion.name] += quantity
			if self.COST_5[u_champion.name] < 0:
				self.COST_5[u_champion.name] = 0
			elif self.COST_5[u_champion.name] > base_pool_values[4]:
				self.COST_5[u_champion.name] = base_pool_values[4]
			self.update_stats(five=True)

	def update_stats(self, allV=False, one=False, two=False, three=False, four=False, five=False):
		if allV or one:
			cost_1 = self.COST_1.values()
			self.num_cost_1 = sum(cost_1)

		if allV or two:
			cost_2 = self.COST_2.values()
			self.num_cost_2 = sum(cost_2)

		if allV or three:
			cost_3 = self.COST_3.values()
			self.num_cost_3 = sum(cost_3)
		
		if allV or four:
			cost_4 = self.COST_4.values()
			self.num_cost_4 = sum(cost_4)

		if allV or five:
			cost_5 = self.COST_5.values()
			self.num_cost_5 = sum(cost_5)
