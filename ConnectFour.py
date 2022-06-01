# code inspired by
# https://www.codecademy.com/courses/learn-the-basics-of-artificial-intelligence-with-minimax/lessons/advanced-minimax
import random
import numpy as np
from copy import deepcopy
from dataclasses import dataclass
from scipy.signal import convolve2d
random.seed(402)

def generateKernels():
    hori_k = np.array([[ 1, 1, 1, 1]])
    vert_k = np.transpose(hori_k)
    diag_k1 = np.eye(4, dtype=int)
    diag_k2 = np.fliplr(diag_k1)
    return [hori_k, vert_k, diag_k1, diag_k2]

def printBoard(board, end= "\n"):
    for row in board:
        print("|", end="")
        for tile in row:
            print(f" {str(tile)} |", end="")
        print()
    print(end=end)

PLAYER_X = True
PLAYER_O = False
BOARD_WIDTH = 7
BOARD_HEIGHT = 6
KERNELS = generateKernels() #for win checks with convolute


@dataclass
class Tile:
    __move: bool = None

    def __repr__(self) -> str:
        return f"Move: {self.__move}"

    def __str__(self) -> str:
        if self.isEmpty():
            return " "
        else:
            return "x" if self.__move else "o"

    def isEmpty(self):
        return self.__move == None

    def getMove(self):
        return self.__move # returns none if no move played anyway
    
    def isMove(self, move):
        return self.__move == move

    def clear(self):
        self.__move = None

    def play(self, player: bool):
        if self.isEmpty():
            self.__move = player

def column(matrix, i):
    return [row[i] for row in matrix]

class Connect4():
    def __init__(self):
        self.__board = np.array([[Tile() for row in range(BOARD_WIDTH)] for col in range(BOARD_HEIGHT)])
        self.__activePlayer = PLAYER_X
    
    def __repr__(self) -> str:
        string= f"\nwidth: {BOARD_WIDTH}\nheight: {BOARD_HEIGHT}\n"
        for row in (self.__board):
            string+= f"{row}\n"
        return string
    
    def __str__(self) -> str:
        string = ""
        for row in reversed(self.__board):
            string += "|"
            for tile in row:
                string += f" {str(tile)} |"
            string += "\n"
        return string

    def __eq__(self, other):
        return (self.__board == other.__board).all() and self.__activePlayer == other.__activePlayer

    def isFull(self):
        for row in self.__board:
            for tile in row:
                if not tile.isEmpty():
                    return False
        return True

    def isMaximizing(self):
        return self.__activePlayer

    def print(self):
        print(self)

    def view(self, currMove = False):
        print(self)
        if currMove:
            print(f"Current player: {'x' if self.__activePlayer else 'o'}")

    def clear(self):
        self.__maxDepth = BOARD_HEIGHT*BOARD_WIDTH
        self.__board = np.array([[Tile() for col in range(BOARD_WIDTH)] for row in range(BOARD_HEIGHT)])
        self.__activePlayer = PLAYER_X

    def validMove(self, col:int):
        if col<0 or col>= BOARD_WIDTH or not self.__board[BOARD_HEIGHT-1][col].isEmpty():
            return False
        return True
    
    def openMoves(self):
        return [move for move in range(BOARD_WIDTH) if self.validMove(move)]

    def play(self, col: int):
        if not self.validMove(col):
            return False
        for tile in column(self.__board, col):
            if tile.isEmpty():
                tile.play(self.__activePlayer)
                self.__activePlayer = (PLAYER_O if self.__activePlayer else PLAYER_X)
                return True
        return False

    def seeAsPlayer(self, player:bool):
        return [[1 if self.__board[col][row].isMove(player) else 0 for row in range(BOARD_WIDTH)]
                for col in range(BOARD_HEIGHT)]

    def conEval(self, player):
        value = 0
        for kernel in KERNELS:
            value = np.amax(convolve2d(self.seeAsPlayer(player), kernel, mode="valid"))
        # this function is for temporary board eval, it also count broken streak
        # we mainly will be looking for complete wins at high depths
        return value

    def hasWin(self, player: bool):
        for kernel in KERNELS:
            if (convolve2d(self.seeAsPlayer(player), kernel, mode="valid") == 4).any():
                return True
        return False

    def isOver(self):
        return self.hasWin(PLAYER_X) or self.hasWin(PLAYER_O) or len(self.openMoves())==0

    def eval(self):
        if self.hasWin(PLAYER_X):
            return float("Inf")
        elif self.hasWin(PLAYER_O):
            return -float("Inf")
        elif len(self.openMoves())==0:
            return 0
        elif self.__activePlayer:
            return self.conEval(self.__activePlayer)
        else:
            return -self.conEval(self.__activePlayer)

def minimax(game: Connect4, depth = 0, alpha = -float("Inf"), beta = float("Inf")):
    if game.isOver() or depth == 0:
        return [game.eval(),]
    if game.isMaximizing():
        bestEval = -float("Inf")
        moves = game.openMoves()
        random.shuffle(moves)
        bestMove = moves[0]
        for move in moves:
            newGame = deepcopy(game)
            newGame.play(move)
            newEval = minimax(newGame, depth-1,alpha,beta)[0]
            if newEval > bestEval:
                bestEval = newEval
                bestMove = move
            alpha = max(alpha,bestEval)
            if alpha>=beta: break
        return [bestEval, bestMove]
    else:
        bestEval = float("Inf")
        moves = game.openMoves()
        random.shuffle(moves)
        bestMove = moves[0]
        for move in moves:
            newGame = deepcopy(game)
            newGame.play(move)
            newEval = minimax(newGame, depth-1,alpha,beta)[0]
            if newEval < bestEval:
                bestEval = newEval
                bestMove = move
            beta = min(beta,bestEval)
            if alpha>=beta: break
        return [bestEval, bestMove]