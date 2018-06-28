# human play crap
import chess
from chess import Board
from agent import Agent
a = Agent()
a.load_nn("model//model.h5")

b = Board()

while not b.is_game_over(claim_draw=True):
    print("=======================================")
    move = a.play_move(b)
    b.push(move)

    print(b)
    uci = input("Your move: ")
    move = chess.Move.from_uci(uci)
    b.push(move)
    print("=======================================")
    print(b)


#print(a.play_move(b))