# human play crap
import chess
from chess import Board
from agent import Agent
a = Agent()
a.build_nn()

b = Board()

#while not b.is_game_over(claim_draw=True):
#    print("=======================================")
move = a.play_move()
print(move)


#print(a.play_move(b))
"""

from chess import Board
import chess.pgn

result_dict = {
    "1-0": 0,
    "1/2-1/2": 0,
    "*": 0,
    "0-1": 0
}

num_of_games = 50000
starting_game = 17600000

for game_number in range(num_of_games):
    try:
        pgn = open("D://data//qchess//games//" + str(starting_game + game_number) + ".pgn")
        game = chess.pgn.read_game(pgn)
        board = game.board()

        for move in game.main_line():
            board.push(move)

        result_dict[board.result(claim_draw=True)] += 1

    except AttributeError:
        print("AttributeError at " + str(starting_game + game_number) + ".pgn: cannot read data from board")

    except UnicodeDecodeError:
        print("UnicodeDecodeError at " + str(starting_game + game_number) + ".pgn: symbol does not match any recognized")


for key, value in result_dict.items():
    print(key + ":" + str(value))
"""
