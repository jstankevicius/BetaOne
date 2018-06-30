from agent import Agent
from chess import Board
import chess.pgn
import translator as tr
import numpy as np

BATCHES = 15625                     # how many batches are fed into the neural network
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

base_board = Board()
base_tensor = tr.board_tensor(base_board)

agent = Agent()
agent.build_nn()

for batch in range(BATCHES):

    inputs = np.zeros(shape=(BATCH_SIZE, 8, 8, 12))
    outputs = np.zeros(shape=(BATCH_SIZE, 1))
    index_in_batch = 0

    for game_number in range(GAMES):
        try:
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

        except AttributeError:
            print("AttributeError at " + str(START + game_number) + ".pgn: cannot read data from board")

        except UnicodeDecodeError:
            print("UnicodeDecodeError at " + str(START + game_number) + ".pgn: symbol does not match any recognized")

    loss = agent.train(inputs, outputs)
    print(str(batch) + ".\tloss: " + str(loss))

    if (batch + 1) % 625 == 0:
        # perform base evaluation:
        base_eval = agent.eval(base_tensor).astype(np.str_)
        agent.get_nn().save("models//model" + str(batch) + "_BE_" + base_eval + ".h5")