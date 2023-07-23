from Simulator.pool_stats import cost_star_values
import numpy as np

def alt_auto_battle(player_1, player_2, round_damage=0):
    blue_board = player_1.board
    red_board = player_2.board
    blue_score = 0
    red_score = 0

    for x in range(7):
        for y in range(4):
            if blue_board[x][y]:
                blue_score += cost_star_values[blue_board[x][y].cost - 1][blue_board[x][y].stars - 1]
            if red_board[x][y]:
                red_score += cost_star_values[red_board[x][y].cost - 1][red_board[x][y].stars - 1]

    blue_values = list(player_1.team_composition.values())
    blue_tier_values = list(player_1.team_tiers.values())

    red_values = list(player_2.team_composition.values())
    red_tier_values = list(player_2.team_tiers.values())

    for i in range(len(player_1.team_composition)):
        if blue_values[i] != 0:
            blue_score += blue_values[i] * blue_tier_values[i]

    for i in range(len(player_2.team_composition)):
        if red_values[i] != 0:
            red_score += red_values[i] * red_tier_values[i]
    # print("blue_score: {}, red_score: {}, damage: {}".format(blue_score, red_score,
    #                                                          np.absolute(blue_score - red_score) + round_damage))

    if red_score == blue_score:
        return 0, round_damage
    elif blue_score > red_score:
        return 1, blue_score - red_score + round_damage
    elif red_score > blue_score:
        return 2, red_score - blue_score + round_damage
