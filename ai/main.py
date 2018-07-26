# human play crap
import chess
from chess import Board
from agent import Agent
a = Agent()
a.load_nn("model//model3000.h5")

b = Board()

move = a.play_move()
print(move)