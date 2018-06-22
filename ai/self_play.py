from agent import Agent
from chess import Board
from keras.models import load_model
import chess.pgn
import translator as tr
import numpy as np
import time

BATCHES = 10
BATCH_SIZE = 2048
GAMES = 256
MOVE_SAMPLES = 8


agent = Agent()
agent.build_nn()

pgn = open("D://data//qchess//chess_games.pgn")
game_offsets = chess.pgn.scan_offsets(pgn)

for i in range(BATCHES):

    inputs = np.zeros(shape=(BATCH_SIZE, 8, 8, 12))
    outputs = np.zeros(shape=(BATCH_SIZE, 8, 8, 64))
    a = 0

    for k in range(GAMES):
        offset = next(game_offsets)
        pgn.seek(offset)
        game = chess.pgn.read_game(pgn)
        board = game.board()

        board_states = []
        moves = []

        # Iterate through moves and add them to a list:
        for move in game.main_line():
            board_states.append(tr.board_tensor(board))
            moves.append(move)
            board.push(move)

        # Now sample 8 random moves per game:
        indices = np.random.randint(low=0, high=len(moves), size=MOVE_SAMPLES)

        for k in range(len(indices)):
            index = indices[k]

            # If the index is even, we know white played the move.
            if index % 2 == 0:
                inputs[a] = board_states[index]
                outputs[a] = tr.move_to_matrix(moves[index])
            else:
                inputs[a] = tr.mirror_board(board_states[index])
                outputs[a] = tr.move_to_matrix(moves[index], multiplier=-1)

            a += 1
    
    loss = agent.train(inputs, outputs)
    print("Batch " + str(i) + " results: loss: " + str(loss))


network = agent.get_nn()
network.save("model.h5")