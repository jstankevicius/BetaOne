import traceback
import chess.pgn

START = 18000000
GAMES = 2000000

RESULT_DICT = {
    "1-0": 1,
    "1/2-1/2": 0,
    "*": 0,
    "0-1": -1
}

fen = open("D://data//FEN.txt", "w")

for i in range(GAMES):
    if i%1000 == 0:
        print(str(i) + " games have been processed.")

    with open("D://data//games//" + str(START + i) + ".pgn") as pgn:

        try:
            game = chess.pgn.read_game(pgn)
            board = game.board()
            positions = []

            # All positions are initially added from the perspective of white.
            for move in game.main_line():
                board.push(move)
                positions.append(board.copy().fen())

            result = str(RESULT_DICT[board.result(claim_draw=True)])

            for pos in positions:
                lp = len(pos)
                lr = len(result)
                fen.write(pos + " " * (80-lp-lr) + result + "\n")

        except AttributeError:
            print("AttributeError at " + str(START + i))
            print(traceback.format_exc())

        except UnicodeDecodeError:
            print("UnicodeDecodeError at " + str(START + i))
            print(traceback.format_exc())

fen.close()