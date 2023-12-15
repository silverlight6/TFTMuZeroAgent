from typing import List, Dict, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import os
import numpy as np

from Simulator.porox.player import Player
from pprint import pprint

@dataclass
class Champion:
    name: str
    cost: int
    stars: int
    chosen: str
    
    items: List["Item"]
    
    location: int # 0-27: board, 0-8: bench, 0-4: shop
    
    def __init__(self, champion, location):
        self.name = champion.name
        self.cost = champion.cost
        self.stars = champion.stars
        self.chosen = champion.chosen
        self.items = [Item(item) for item in champion.items]
        self.location = location
    
@dataclass
class Item:
    name: str
    
    def __init__(self, item):
        self.name = item

@dataclass
class Trait:
    name: str
    count: int
    total: int
        
class GameState:
    """Stores the game state as a list of changes to the game state.
    
    Game JSON Format:

    {
        "players": {
            "player_{player_id}": {
                "state" { 
                    "health": int,
                    "exp": int,
                    "level": int,
                    "gold": int,
                    "win_streak": int,
                    "loss_streak": int,
                    "board": [Champion, ...],
                    "bench": [Champion, ...],
                    "shop": [Champion, ...],
                    "items": [Item, ...],
                    "traits": [Trait, ...],
                },
                "actions": [Change, ...]
                "battles": [Battle, ...]
            }
        }
    }
    
    Action can be either a player action or an environment action.

    """
    def __init__(self, players, game_round, render_path, action_class):
        self.players = players
        self.action_class = action_class
        self.player_states = {player_id: self.create_player_state(player) for player_id, player in players.items()}
        self.player_actions = {player_id: [] for player_id in players}
        self.player_battles = {player_id: [] for player_id in players}
        
        self.game_round = game_round
        self.render_path = render_path
        
        # Will be in order of deaths
        self.summaries = []
        
        # --- Utility Variables ---
        self.minion_rounds = {0, 1, 2, 8, 14, 20, 26, 32, 38}
        self.alive_players = [player_id for player_id in players]
        
        # Store First Round Results
        self.store_first_battle()
        
    def write_json(self):
        for alive in self.alive_players:
            self.store_summary(alive)

        time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"game_{time}.json"
        file_path = os.path.join(self.render_path, file_name)
        
        if not os.path.exists(self.render_path):
            os.makedirs(self.render_path)
            
        data = {player_id: {} for player_id in self.players}
        for player in self.players:
            data[player]["state"] = self.player_states[player]
            data[player]["actions"] = self.player_actions[player]
            data[player]["battles"] = self.player_battles[player]
            
        data = {
            "players": data,
            "summaries": self.summaries,
        }
            
        with open(file_path, "w") as f:
            json.dump(data,f)
        
    def create_player_state(self, player):
        return {
            "health": player.health,
            "exp": player.exp,
            "level": player.level,
            "gold": player.gold,
            "win_streak": player.win_streak,
            "loss_streak": player.loss_streak,
            "board": self.update_board(player),
            "bench": self.update_bench(player),
            "shop": self.update_shop(player),
            "items": self.update_items(player),
            "traits": [], # TODO
        }
        
    def update_board(self, player):
        board = []
        for x in range(len(player.board)):
            for y in range(len(player.board[x])):
                champion = player.board[x][y]
                if champion is not None:
                    champion = asdict(Champion(champion, x * 4 + y))
                    board.append(champion)
        return board
        
    def update_bench(self, player):
        bench = []
        for i, champion in enumerate(player.bench):
            if champion is not None:
                champion = asdict(Champion(champion, i))
                bench.append(champion)
        return bench
        
    def update_shop(self, player):
        shop = []
        for i, champion in enumerate(player.shop_champions):
            if champion is not None:
                champion = asdict(Champion(champion, i))
                shop.append(champion)
        return shop
    
    def update_items(self, player):
        items = []
        for item in player.item_bench:
            if item is not None:
                item = asdict(Item(item))
                items.append(item)
        return items
    
    def store_game_round(self):
        for alive in self.alive_players:
            print(f"----- STORE {alive} -----")
            print(self.players[alive])
            print(self.player_battles[alive][-1])
            change = {
                "action": [-1, -1, -1],
                "gold": self.players[alive].gold,
                "exp": self.players[alive].exp,
                "level": self.players[alive].level,
                "shop": self.update_shop(self.players[alive]),
            }
            print(change)
            
            self.player_actions[alive].append(change)
            
            print(f"----- END {alive} -----")
    
    def store_summary(self, player_id):
        player = self.players[player_id]
        
        placement = len(self.players) - len(self.summaries)

        summary = {
            "player": player_id,
            "placement": placement,
            "health": player.health,
            "gold": player.gold,
            "level": player.level,
            "board": self.update_board(player),
        }
        

        print(f"-----START {player_id}------")
        print(self.game_round.matchups)
        print(self.alive_players)

        self.summaries.append(summary)
        self.alive_players.remove(player_id)
        
        print(self.alive_players)
        
        print(f"------END {player_id}--------")
    
    def store_first_battle(self):
        for player in self.players:
            battle = {
                "round": 0,
                "health": self.players[player].health,
                "opponent": "minion",
                "damage": 0,
                "opponentDamage": 0,
                "board": self.update_board(self.players[player]),
                "opponentBoard": None,
                "result": "win"
            }
            self.player_battles[player].append(battle)
    
    def store_battles(self):
        round = self.game_round.current_round - 1
        matchups = self.game_round.matchups
        is_minion_round = round in self.minion_rounds
        is_carousel_round = len(self.game_round.game_rounds[round]) == 2
        
        if is_minion_round:
            print("----- MINION ROUND -----")
            for player in self.alive_players:
                health = self.players[player].health
                prev_health = self.player_battles[player][-1]["health"]
                
                if health < prev_health:
                    result = "loss"
                else:
                    result = "win"

                battle = {
                    "round": round,
                    "health": health,
                    "opponent": "minion",
                    "damage": prev_health - health,
                    "opponentDamage": 0,
                    "board": self.update_board(self.players[player]),
                    "opponentBoard": None,
                    "result": result,
                }
                
                self.player_battles[player].append(battle)

                change = {
                    "action": [-2, -2, -2],
                    "health": health,
                    "gold": self.players[player].gold,
                    "bench": self.update_bench(self.players[player]),
                    "items": self.update_items(self.players[player]),
                }
                
                self.player_actions[player].append(change)
                
                if health <= 0:
                    # TODO: Generate summary
                    self.store_summary(player)
                print(f"----- {player} ---- minion -----")
            print("-----END-----")
                
                
            return # MAKE SURE THIS IS IN THE RIGHT PLACE (TOOK ME AN HOUR TO FIND THIS BUG)

        for matchup in matchups:
            is_ghost = len(matchup) == 3
            player_a = matchup[0]
            player_b = matchup[-1]
                
            health_a = self.players[player_a].health
            health_b = self.players[player_b].health
            
            prev_health_a = self.player_battles[player_a][-1]["health"]
            prev_health_b = self.player_battles[player_b][-1]["health"]
            
            if (health_a < prev_health_a) and (health_b < prev_health_b):
                result_a = "tie"
                result_b = "tie"
            elif health_a < prev_health_a:
                result_a = "loss"
                result_b = "win"
            else:
                result_a = "win"
                result_b = "loss"
                
            if is_ghost:
                result_a = f"{result_a}_ghost"
                result_b = f"{result_b}_ghost"
                
            damage_a = prev_health_a - health_a
            damage_b = prev_health_b - health_b
            
            board_a = self.update_board(self.players[player_a])
            board_b = self.update_board(self.players[player_b])
            
            battle_a = {
                "round": round,
                "opponent": player_b,
                "health": health_a,
                "board": board_a,
                "opponentBoard": board_b,
                "damage": damage_a,
                "opponentDamage": damage_b,
                "result": result_a,
            }
            
            battle_b = {
                "round": round,
                "opponent": player_a,
                "health": health_b,
                "board": board_b,
                "opponentBoard": board_a,
                "damage": damage_b,
                "opponentDamage": damage_a,
                "result": result_b,
            }
            
            self.player_battles[player_a].append(battle_a)
            self.player_battles[player_b].append(battle_b)
            
            change_a = {
                "action": [-2, -2, -2],
                "health": health_a,
                "win_streak": self.players[player_a].win_streak,
                "loss_streak": self.players[player_a].loss_streak,
                "board": board_a,
                "bench": self.update_bench(self.players[player_a]),
            }

            change_b = {
                "action": [-2, -2, -2],
                "health": health_b,
                "win_streak": self.players[player_b].win_streak,
                "loss_streak": self.players[player_b].loss_streak,
                "board": board_b,
                "bench": self.update_bench(self.players[player_b]),
            }
            
            # if is_carousel_round:
            #     change_a["bench"] = self.update_bench(self.players[player_a])
            #     change_b["bench"] = self.update_bench(self.players[player_b])
            
            self.player_actions[player_a].append(change_a)
            self.player_actions[player_b].append(change_b)
            
            print(f"----- {player_a} ---- {player_b} -----")
            if health_a <= 0:
                self.store_summary(player_a)
            if health_b <= 0 and not is_ghost:
                self.store_summary(player_b)
            print("-----END-----")
        
    def store_action(self, agent, action):
        """Called after the action is taken."""
        
        player = self.players[agent]
        action = self.action_class.action_space_to_action(action)
        
        action_type, x1, x2 = action
        
        change = {
            "action": [int(action_type), int(x1), int(x2)],
        }
        
        # Pass Action
        if action_type == 0:
            pass
        # Level Action
        elif action_type == 1:
            change["gold"] = player.gold
            change["exp"] = player.exp
            change["level"] = player.level
        # Refresh Action
        elif action_type == 2:
            change["gold"] = player.gold
            change["shop"] = self.update_shop(player)
        # Buy Action
        elif action_type == 3:
            change["gold"] = player.gold
            change["board"] = self.update_board(player)
            change["bench"] = self.update_bench(player)
            change["shop"] = self.update_shop(player)
        # Sell Action
        elif action_type == 4:
            change["gold"] = player.gold
            change["items"] = self.update_items(player)
            if x1 < 28: # Board champion was sold
                change["board"] = self.update_board(player)
            else: # Bench champion was sold
                change["bench"] = self.update_bench(player)
        # Move Action
        elif action_type == 5:
            if x1 < 28 and x2 < 28: # Board to Board
                change["board"] = self.update_board(player)
            elif x1 < 28 and x2 >= 28: # Board to Bench
                change["board"] = self.update_board(player)
                change["bench"] = self.update_bench(player)
            elif x1 >= 28 and x2 < 28: # Bench to Board
                change["bench"] = self.update_bench(player)
                change["board"] = self.update_board(player)
            else: # Bench to Bench
                change["bench"] = self.update_bench(player)
        # Item action
        elif action_type == 6:
            change["items"] = self.update_items(player)
            if x2 < 28: # Item was given to champion on board
                change["board"] = self.update_board(player)
            else: # Item was given to champion on bench
                change["bench"] = self.update_bench(player)
        
        self.player_actions[agent].append(change)