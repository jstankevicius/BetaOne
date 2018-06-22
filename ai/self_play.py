from agent import Agent
from chess import Board
from keras.models import load_model
import chess.pgn
import translator as tr
import numpy as np

SESSIONS = 10
a = Agent()
a.build_nn()

pgn = open("D://data//qchess//chess_games.pgn")

for i in range(1000):
    game = chess.pgn.read_game(pgn)
    print("GAME " + str(i) + ": " + game.headers["White"] + " vs " + game.headers["Black"])
    board = game.board()

    board_states = []
    moves = []

    # Iterate through moves and add them to a list:
    for move in game.main_line():
        board_states.append(tr.board_tensor(board))
        moves.append(move)
        board.push(move)

    # Now sample 10 moves per game:
    batch_size = 10
    indices = np.random.randint(low=0, high=len(moves), size=batch_size)

    inputs = np.zeros(shape=(batch_size, 8, 8, 12))
    outputs = np.zeros(shape=(batch_size, 8, 8, 64))

    for k in range(len(indices)):
        index = indices[k]

        # If the index is even, we know white played the move.
        if index % 2 == 0:
            inputs[k] = board_states[index]
            outputs[k] = tr.move_to_matrix(moves[index])
        else:
            inputs[k] = tr.mirror_board(board_states[index])
            outputs[k] = tr.move_to_matrix(moves[index], multiplier=-1)

    #a.train(inputs, outputs)

a.get_nn().save("model.h5")