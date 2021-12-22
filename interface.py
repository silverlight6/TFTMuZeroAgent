import champion
import player as player_class
import AI_interface
from pool import pool
from champion_functions import MILLIS
from stats import COST



class interface:
    def _init_(self):
        self.shop = [ None for x in range( 5 ) ]


    def inputIsInt(self, user_input):
        """
        cast user input to int
        return true if sucess; false, otherwise
        """
        try:
            int(user_input)
        except ValueError:
            return False
        return True
    
    # This is for the human interface.
    # player should be a player object
    def outer_loop(self, player, game_round=-1): 
        # Refresh the shop at the start of the turn
        # This is where the turn starts.
        # print(game_round)
        pool_obj = pool()
        player.gold_income(game_round)
        self.shop = pool_obj.sample(player, 5)
        while(True): 
            self.print_start(player)
            option = input()
            # buy champion from pool
            if (option == '0'):
                if self.shop[0] == " ": pass
                if shop[0].endswith("_c"):
                    c_shop = shop[0].split('_')
                    a_champion = champion.champion(c_shop[0], chosen = c_shop[1])
                else:
                    a_champion = champion.champion(self.shop[0])
                player.buy_champion(a_champion)
                self.shop[0] = " "
                print("buy option 0")
                player.printt("buy option 0")
            elif (option == '1'): 
                if self.shop[1] == " ": pass
                if shop[1].endswith("_c"):
                    c_shop = shop[1].split('_')
                    a_champion = champion.champion(c_shop[0], chosen = c_shop[1])
                else:
                    a_champion = champion.champion(self.shop[1])
                player.buy_champion(a_champion)
                self.shop[1] = " "
                print(" buy option 1")
                player.printt("buy option 1")
            elif (option == '2'): 
                if self.shop[2] == " ": pass
                if shop[2].endswith("_c"):
                    c_shop = shop[2].split('_')
                    a_champion = champion.champion(c_shop[0], chosen = c_shop[1])
                else:
                    a_champion = champion.champion(self.shop[2])
                player.buy_champion(a_champion)
                self.shop[2] = " "
                print("buy option 2")
                player.printt("buy option 2")
            elif (option == '3'): 
                if self.shop[3] == " ": pass
                if shop[3].endswith("_c"):
                    c_shop = shop[3].split('_')
                    a_champion = champion.champion(c_shop[0], chosen = c_shop[1])
                else:
                    a_champion = champion.champion(self.shop[3])
                player.buy_champion(a_champion)
                self.shop[3] = " "
                print("buy option 3")
                player.printt("buy option 3")
            elif (option == '4'): 
                if self.shop[4] == " ": pass
                aif shop[4].endswith("_c"):
                    c_shop = shop[4].split('_')
                    a_champion = champion.champion(c_shop[0], chosen = c_shop[1])
                else:
                    a_champion = champion.champion(self.shop[4])
                player.buy_champion(a_champion)
                self.shop[4] = " "
                print("buy option 4")
                player.printt("buy option 4")
            # Refresh
            elif (option == '5'):
                self.shop = pool_obj.sample(player, 5)
                print("Refresh")
                player.printt("Refresh")
            # buy Exp
            elif (option == '6'): 
                player.buy_exp()
                print("set up exp")
                player.printt("exp")
            # end turn 
            elif (option == '6'): 
                break
            # move Item
            elif (option == '7'): 
                print("call move item method")
                player.printt("move item method")
            # sell Unit
            elif (option == '8'): 
                print("call sell unit method")
                player.printt("sell unit method")
            # move bench to Board
            elif (option == '9'): 
                print("call move bench to Board mthod")
                player.printt("move bench to board")

            # move board to bench
            elif (option == '10'): 
                print("call move board to bench method")
                player.printt("move board to bench")
            # 
            else:
                print("wrong call")
        
     
    def read_integer_commmand_input(self, promptMsg = "please type in an integer", errorMsg = "not an integer!"): 
        print(promptMsg)
        result = input()
        if(inputIsInt(result)): 
            result = int(result) # cast result
            return result
        else: 
            print(errorMsg)
            read_integer_commmand_input(promptMsg, errorMsg)


    def print_start(self, player):
        print("0: " + self.shop[0] + ", 1: " + self.shop[1] + ", 2 : " + self.shop[2] + ", 3: " + self.shop[3] + ", 4: " + self.shop[4])
        print("player gold = " + str(player.gold) + ", player exp = " + str(player.exp) + "player level = " + str(player.level))
        print("5: Refresh, 6: Exp, 7: End Turn")
        print("8: Move Unit, 9: Sell Unit, 10: Move bench to board, 11: Move board to bench")

    
    def move_bench_to_board_api(self, player): 
        print("move bench to board api")
        # get x location on board
        xLocBo = read_integer_commmand_input("enter x Loc Board")
        # get y location on board
        yLocBo = read_integer_commmand_input("enter y Loc Board")

        # get y location on bench
        xLocBe = read_integer_commmand_input("enter x Loc bench")
        # execution
        if(not player.MoveBenchtoBoard(xLocBe, xLocBo, yLocBo)):
            print("Move command failed")

    def move_board_to_bench_api(self, player): 
        print("move bench to board api")
        # get x location on board
        xLocBo = read_integer_commmand_input("enter x Loc Board")
        # get y location on board
        yLocBo = read_integer_commmand_input("enter y Loc Board")
        # execution
        if(not player.MoveBoardtoBench(xLocBo, yLocBo)):
            print("Move command failed")
    
    def sell_unit_api(self, player): 
        print("sell uint api")
        xLocBo = read_integer_commmand_input("enter x Loc bench")
        if(not player.sell_from_bench(xLocBo)):
            print("Unit sale failed")

    def move_item_api(self, player): 
        print("move intem api")
        itemEntry = read_integer_commmand_input("enter item entry")
        # get x location on board
        xLocBo = read_integer_commmand_input("enter x Loc Board")
        # get y location on board
        yLocBo = read_integer_commmand_input("enter y Loc Board")

        index = player.findItem(itemEnt)
        if not player.move_item_to_board(index, xLocBo, yLocBo):
            print("Could not put item on {}, {}".format(x, y))

    