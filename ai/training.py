from agent import Agent
from chess import Board
import chess.pgn
import translator as tr
import numpy as np

BATCHES = 512                       # how many batches are fed into the neural network
GAMES = 16                          # how many games we sample per batch
MOVE_SAMPLES = 2                    # how many board positions are sampled per game
BATCH_SIZE = GAMES * MOVE_SAMPLES   # total number of samples fed into the batch

RESULT_DICT = {
    "1-0": 1,
    "1/2-1/2": 0,
    "*": 0,
    "0-1": -1
}

# The number of the first game
START = 17600000

agent = Agent()
agent.build_nn()

for batch in range(BATCHES):

    inputs = np.zeros(shape=(BATCH_SIZE, 8, 8, 12))
    outputs = np.zeros(shape=(BATCH_SIZE, 1))
    index_in_batch = 0
    print(batch)

    for game_number in range(GAMES):
        pgn = open("D://data//qchess//games//" + str(START + game_number) + ".pgn")
        game = chess.pgn.read_game(pgn)
        board = game.board()

        positions = []

        for move in game.main_line():
            positions.append(tr.board_tensor(board))
            board.push(move)

        sampled_indices = np.random.randint(low=0, high=len(positions), size=MOVE_SAMPLES, dtype=np.uint32)

        for i in range(MOVE_SAMPLES):
            sample_index = sampled_indices[i]

            inputs[index_in_batch] = positions[sample_index]
            outputs[index_in_batch] = RESULT_DICT[board.result(claim_draw=True)]

            index_in_batch += 1

            # TODO: handle errors

    agent.train(inputs, outputs)