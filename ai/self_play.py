from agent import Agent
import chess
from chess import Board

SESSIONS = 10
a = Agent()
a.build_nn()

board = Board()


for i in range(1):
    b = Board()
    while not b.is_game_over(claim_draw=True):
        a.play_move(board)
        print("===========================")
        print(board)
        a.play_move(board)
        print("===========================")
        print(board)

