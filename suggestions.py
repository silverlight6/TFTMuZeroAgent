#Mock method for player class to swap 2 units on board
def swapBoardBoard(self, x1:int,y1:int,x2:int,y2:int):
    """Swaps the content of 2 board slots

    Args:
        x1 (int): x coordinate of the first board slot
        y1 (int): y coordinate of the first board slot
        x2 (int): x coordinate of the second board slot
        y2 (int): y coordinate of the second board slot
    """
    self.board[x1][y1], self.board[x2][y2] = self.board[x1][y1],self.board[x1][y1] 
    if self.board[x1][y1] == None and  self.board[x2][y2] == None:
        someNegativeReward = 1

#Mock method for player class swapping board and bench slot
def swapBenchBoard(self, benchX:int, boardX:int, boardY:int):
    """Swaps a bench slot with a board slot

    Args:
        benchX (int): the bench coordinate
        boardX (int): the board x coordinate
        boardY (int): the board y coordinate
    """
    
    #check to prevent illegal moves
    #Is field Azir Soldier
    #Is field full

    #azir soldier related stuff
    self.board[boardX][boardY], self.bench[benchX] = self.bench[benchX],self.board[boardX][boardY]
    #negative reward if both slots are empty
