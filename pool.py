import champion
import player
import random
import stats
from pool_stats import *
from origin_class_stats import origin_class, chosen_exclude

class pool:
	def __init__(self):
		self.num_cost_1 = 0
		self.num_cost_2 = 0
		self.num_cost_3 = 0
		self.num_cost_4 = 0
		self.num_cost_5 = 0
		self.update_stats(allV=True)


	# Player is the player that the sample is being picked for
	# Num is the number of samples to be returned
	# Index is the level of the champion you want to be sampled, -1 for random or to follow level.
	# TO DO: Turn the cost arrays into a dictionary of dictionaries.
	# TO DO: Implement the chosen mechanic and ensure the doubling of the right stat
	# Chosen is implemented as a string with the class being the possible one.
	def sample(self, player, num, idx=-1):
		ranInt = [0 for x in range(num)]
		championOptions = [None for x in range(num)]
		chosen = player.chosen
		chosen_index = -1
		if chosen:
			if random.random() < .5:
				chosen_index = random.randint(0, 5)
		index = idx
		for i in range(num):
			if chosen_index != i:
				percents = level_percentage[player.level]
			else:
				percents = chosen_stats[player.level]
			ranInt[i] = random.random()
			if index == -1:
				index = 0
				while ranInt[i] > percents[index]:
					index += 1
			# cost 1
			counter = 0
			counterIndex = 0
			# So I need to implement chosen units here. If the index = i, then pick a chosen champion
			if index == 0:
				cost_1 = list(COST_1.values())
				ranPoolInt = random.randint(0, self.num_cost_1 - 1)
				while counter < ranPoolInt:
					counter += cost_1[counterIndex]
					counterIndex += 1
				keys_list = list(COST_1)
				championOptions[i] = keys_list[counterIndex - 1]

			# cost 2
			elif index == 1:
				cost_2 = list(COST_2.values())
				ranPoolInt = random.randint(0, self.num_cost_2 - 1)
				while counter < ranPoolInt:
					counter += cost_2[counterIndex]
					counterIndex += 1
				keys_list = list(COST_2)
				championOptions[i] = keys_list[counterIndex - 1]

			# cost 3
			elif index == 2:
				cost_3 = list(COST_3.values())
				ranPoolInt = random.randint(0, self.num_cost_3 - 1)
				while counter < ranPoolInt:
					counter += cost_3[counterIndex]
					counterIndex += 1
				keys_list = list(COST_3)
				championOptions[i] = keys_list[counterIndex - 1]

			# cost 4
			elif index == 3:
				cost_4 = list(COST_4.values())
				ranPoolInt = random.randint(0, self.num_cost_4 - 1)
				while counter < ranPoolInt:
					counter += cost_1[counterIndex]
					counterIndex += 1
				keys_list = list(COST_4)
				championOptions[i] = keys_list[counterIndex - 1]

			# cost 5
			else:
				cost_5 = list(COST_5.values())
				ranPoolInt = random.randint(0, self.num_cost_5 - 1)
				while counter < ranPoolInt:
					counter += cost_1[counterIndex]
					counterIndex += 1
				keys_list = list(COST_5)
				championOptions[i] = keys_list[counterIndex - 1]
			# This adds the chosen aspect to the champion.
			if chosen_index == i:
				chosen_type = self.pick_chosen(championOptions[i])
				championOptions[i] = championOptions[i] + "_" + chosen_type + "_c"
			index = idx
		return championOptions


	# everytime a minion is taken from the pool, update the statsics 
	# direction is positive for adding and negative for selling from pool.
	# i.e. If a player sells a unit, it's a positive direction for the pool
	def update(self, u_champion, direction):
		# This line is actually a little faster than combining this line and cost = stats.COST[u_champion.name]
		# because it does not require loading the array for champion costs to search for the value.
		cost = u_champion.cost
		quantity = 3 ** (u_champion.stars - 1) * direction
		if u_champion.stars != 1:
			cost = stats.COST[u_champion.name]
		if cost == 1:
			COST_1[u_champion.name] += quantity
			self.update_stats(one=True)
		elif cost == 2:
			COST_2[u_champion.name] += quantity
			self.update_stats(two=True)
		elif cost == 3:
			COST_3[u_champion.name] += quantity
			self.update_stats(three=True)
		elif cost == 4:
			COST_4[u_champion.name] += quantity
			self.update_stats(four=True)
		elif cost == 5:
			COST_5[u_champion.name] += quantity
			self.update_stats(five=True)


	def update_stats(self, allV=False, one=False, two=False, three=False, four=False, five=False):
		if(allV or one):
			cost_1 = COST_1.values()
			self.num_cost_1 = sum(cost_1)

		if(allV or two):
			cost_2 = COST_2.values()
			self.num_cost_2 = sum(cost_2)

		if(allV or three):
			cost_3 = COST_3.values()
			self.num_cost_3 = sum(cost_3)
		
		if(allV or four):
			cost_4 = COST_4.values()
			self.num_cost_4 = sum(cost_4)

		if(allV or five):
			cost_5 = COST_5.values()
			self.num_cost_5 = sum(cost_5)


	def pick_chosen(self, champion_name):
		chosen_type = random.choose(origin_class[champion_name])
		if chosen_type in chosen_exclude:
			# every champion has at least one type that is acceptable to pick so it isn't going to infinite loop here
			self.pick_chosen(champion_name)
			# Putting this here just in case something like that happens.
			print("could not pick " + chosen_type + " type")
		return chosen_type