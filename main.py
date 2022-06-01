from ConnectFour import *

def twoAI():
    board = Connect4()
    while not board.isOver():
        minMax = minimax(board,depth=7)
        print(f"Selected move: {minMax[1]}")
        board.play(minMax[1])
        board.view(True)

    if board.hasWin(PLAYER_X):
        print("X won!")
    elif board.hasWin(PLAYER_O):
        print("O won!")
    else:
        print("Players tied.")

def main():
    twoAI()

if __name__ == "__main__":
    main()