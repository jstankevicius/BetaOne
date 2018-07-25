from agent import Agent
from chess import Board
import chess.pgn
import position as pos
import numpy as np

GAMES = 4000000
BATCH_SIZE = 256
BATCHES = int(GAMES/BATCH_SIZE)

RESULT_DICT = {
    "1-0": 1,
    "1/2-1/2": 0,
    "*": 0,
    "0-1": -1
}

# The number of the first game
START = 17600000

base_board = Board()
base_tensor = pos.board_tensor(base_board)
processed_games = 0

agent = Agent()
agent.build_nn()


for batch in range(BATCHES):

    inputs = np.zeros(shape=(BATCH_SIZE, 8, 8, 12))
    outputs = np.zeros(shape=(BATCH_SIZE, 1))
    index_in_batch = 0

    for game_number in range(GAMES):
        try:
            pgn = open("D://data//qchess//games//" + str(START + processed_games) + ".pgn")
            game = chess.pgn.read_game(pgn)
            board = game.board()

            positions = []

            ply = 0
            for move in game.main_line():
                board.push(move)
                board_tensor = tr.board_tensor(board)

                if ply % 2 != 0:
                    board_tensor = tr.mirror_tensor(board_tensor)

                positions.append(board_tensor)
                ply += 1

            sampled_indices = np.random.randint(low=0, high=len(positions), size=MOVE_SAMPLES, dtype=np.uint32)

            for i in range(MOVE_SAMPLES):
                sample_index = sampled_indices[i]

                inputs[index_in_batch] = positions[sample_index]
                outputs[index_in_batch] = RESULT_DICT[board.result(claim_draw=True)]

                index_in_batch += 1

        except AttributeError:
            print("AttributeError at " + str(START + processed_games) + ".pgn: cannot read data from board")

        except UnicodeDecodeError:
            print("UnicodeDecodeError at " + str(START + processed_games) + ".pgn: symbol does not match any recognized")

        processed_games += 1

    loss = agent.train(inputs, outputs)
    print(str(batch) + ".\tloss: " + str(loss))

    if (batch + 1) % 625 == 0:
        # perform base evaluation:
        base_eval = agent.eval(base_tensor).astype(np.str_)
        agent.get_nn().save("model//model" + str(batch) + "_BE_" + base_eval + ".h5")