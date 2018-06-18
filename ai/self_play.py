from agent import Agent
import chess
from chess import Board

SESSIONS = 10
a = Agent()
a.build_nn()

board = Board()

for i in range(SESSIONS):
    a.play_move(board)
    print("===========================")
    print(board)
    a.play_move(board)
    print("===========================")
    print(board)

a.train("white")

