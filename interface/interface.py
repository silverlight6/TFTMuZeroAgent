from Simulator import champion


class interface:
    def _init_(self):
        self.shop = [ None for _ in range( 5 ) ]


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
    def outer_loop(self, pool, player, game_round=-1): 
        # Refresh the shop at the start of the turn
        # This is where the turn starts.
        # print(game_round)
        player.gold_income(game_round)
        self.shop = pool.sample(player, 5)
        while(True): 
            self.print_start(player)
            option = input()

            # buy champion from pool
            if (option == '0'):
                if self.shop[0] == " ": pass
                if self.shop[0].endswith("_c"):
                    c_shop = self.shop[0].split('_')
                    a_champion = champion.champion(c_shop[0], chosen = c_shop[1])
                else:
                    a_champion = champion.champion(self.shop[0])
                success = player.buy_champion(a_champion)
                if (success): 
                    self.shop[0] = " "
                print("buy option 0")
                player.printt("buy option 0")

            elif (option == '1'): 
                if self.shop[1] == " ": pass
                if self.shop[1].endswith("_c"):
                    c_shop = self.shop[1].split('_')
                    a_champion = champion.champion(c_shop[0], chosen = c_shop[1])
                else:
                    a_champion = champion.champion(self.shop[1])
                success = player.buy_champion(a_champion)
                if (success): 
                    self.shop[1] = " "
                print(" buy option 1")
                player.printt("buy option 1")

            elif (option == '2'): 
                if self.shop[2] == " ": pass
                if self.shop[2].endswith("_c"):
                    c_shop = self.shop[2].split('_')
                    a_champion = champion.champion(c_shop[0], chosen = c_shop[1])
                else:
                    a_champion = champion.champion(self.shop[2]) 
                success = player.buy_champion(a_champion)
                if (success): 
                    self.shop[2] = " "
                print("buy option 2")
                player.printt("buy option 2")

            elif (option == '3'): 
                if self.shop[3] == " ": pass
                if self.shop[3].endswith("_c"):
                    c_shop = self.shop[3].split('_')
                    a_champion = champion.champion(c_shop[0], chosen = c_shop[1])
                else:
                    a_champion = champion.champion(self.shop[3])
                
                success = player.buy_champion(a_champion)
                if (success): 
                    self.shop[3] = " "
                print("buy option 3")
                player.printt("buy option 3")

            elif (option == '4'): 
                if self.shop[4] == " ": pass
                if self.shop[4].endswith("_c"):
                    c_shop = self.shop[4].split('_')
                    a_champion = champion.champion(c_shop[0], chosen = c_shop[1])
                else:
                    a_champion = champion.champion(self.shop[4])
                
                success = player.buy_champion(a_champion)
                if (success): 
                    self.shop[4] = " "

                print("buy option 4")
                player.printt("buy option 4")

            # Refresh
            elif (option == '5'):
                if (player.refresh()): 
                    self.shop = pool.sample(player, 5)
                    print("Refresh")
                    player.printt("Refresh")
                else:       
                    print('no gold, failt to refresh')

            # buy Exp
            elif (option == '6'):                
                if player.buy_exp(): 
                    print("set up exp")
                    player.printt("exp")
                else: 
                    print('no enough gold, buy exp failed')

            # end turn 
            elif (option == '7'): 
                break

            # move Item
            elif (option == '8'):  # FIXME: SHOUD NOT HAVE MULTIPLE ITEMS
                print("call move item method")
                # before
                self.print_bench(player)
                #execute
                self.move_item_api(player)
                # after
                self.print_bench(player)
                player.printt("move item method")

            # sell Unit
            elif (option == '9'): 
                print("call sell unit method")
                self.sell_unit_api(player)
                player.printt("sell unit method")

            # move bench to Board
            elif (option == '10'): 
                #before
                print("call move bench to Board mthod\nBefore:")
                self.print_bench(player)
                self.print_board(player)
                # execute
                self.move_bench_to_board_api(player)
                # after
                player.printt("move bench to board")
                print("\nAfter: ")
                self.print_bench(player)
                self.print_board(player)
                print()

            # move board to bench
            elif (option == '11'): 
                # before
                print("call move board to bench method")
                self.print_bench(player)
                self.print_board(player)

                #execute
                self.move_board_to_bench_api(player)

                # after
                player.printt("move board to bench")
                print("\nAfter: ")
                self.print_bench(player)
                self.print_board(player)
            # 
            else:
                print("wrong call")
        
     
    def read_integer_commmand_input(self, promptMsg = "please type in an integer", errorMsg = "not an integer!"): 
        print(promptMsg)
        result = input()
        if(self.inputIsInt(result)): 
            result = int(result) # cast result
            return result
        else: 
            print(errorMsg)
            self.read_integer_commmand_input(promptMsg, errorMsg)


    def print_start(self, player):
        print("0: " + self.shop[0] + ", 1: " + self.shop[1] + ", 2 : " + self.shop[2] + ", 3: " + self.shop[3] + ", 4: " + self.shop[4])
        print("player gold = " + str(player.gold) + ", player exp = " + str(player.exp) + "player level = " + str(player.level))
        print("5: Refresh, 6: Exp, 7: End Turn")
        print("8: Move Unit, 9: Sell Unit, 10: Move bench to board, 11: Move board to bench")

    
    def sell_unit_api(self, player): 
        print("sell uint api")
        xLocBo = self.read_integer_commmand_input("enter x Loc bench")
        if(not player.sell_from_bench(xLocBo)):
            print("Unit sale failed")

    def move_item_api(self, player): 
        print("move intem api")
        itemEntry = self.read_integer_commmand_input("enter item entry")
        # get x location on board
        xLocBo = self.read_integer_commmand_input("enter x Loc Board")
        # get y location on board
        yLocBo = self.read_integer_commmand_input("enter y Loc Board")

        index = player.findItem(itemEntry)
        if not player.move_item_to_board(index, xLocBo, yLocBo):
            print("Could not put item on {}, {}".format(xLocBo, yLocBo))

    def move_bench_to_board_api(self, player): 
        print("move bench to board api")
        # get x location on board
        xLocBo = self.read_integer_commmand_input("enter x Loc Board")
        # get y location on board
        yLocBo = self.read_integer_commmand_input("enter y Loc Board")

        # get y location on bench
        xLocBe = self.read_integer_commmand_input("enter x Loc bench")
        # execution
        if(not player.move_bench_to_board(xLocBe, xLocBo, yLocBo)):
            print("Move command failed")

    def move_board_to_bench_api(self, player): 
        print("move bench to board api")
        # get x location on board
        xLocBo = self.read_integer_commmand_input("enter x Loc Board")
        # get y location on board
        yLocBo = self.read_integer_commmand_input("enter y Loc Board")
        # execution
        if(not player.move_board_to_bench(xLocBo, yLocBo)):
            print("Move command failed")
    
    # print 1-d bench
    def print_bench(self, player):
        print("champ list: ")
        for index, champ in enumerate(player.bench): 
            if(champ): 
                print("inex: " + str(index) + ": " + champ.name)
                for item in champ.items:
                    if(item): 
                        print(str(item))

    
    # print 2-d board
    def print_board(self, player): 
        print("player's board")
        for x in range(7): 
            for y in range(4):
                if player.board[x][y]:
                    print("x: " + str(x) + " y: " + str(y) + ": " + player.board[x][y].name)
