from agent import Agent
from chess import Board
import chess.pgn
import translator as tr
import numpy as np


BATCHES = 256                       # how many batches are fed into the neural network
GAMES = 8                           # how many games we sample per batch
MOVE_SAMPLES = 2                    # how many board positions are sampled per game
BATCH_SIZE = GAMES * MOVE_SAMPLES   # total number of samples fed into the batch

RESULT_DICT = {
    "1-0": 1,
    "1/2-1/2": 0,
    "*": 0,
    "0-1": -1
}

base_board = Board()
base_tensor = tr.board_tensor(base_board)


pgn = open("D://data//qchess//chess_games.pgn")
game_offsets = chess.pgn.scan_offsets(pgn)


# Print header
print("Batch #\tGame #\tloss")

for sess in range(16):
    agent = Agent()
    agent.load_nn("model//model.h5")
    print("Base evaluation: " + str(agent.get_nn().predict(np.array([base_tensor]))))

    for i in range(BATCHES):
        inputs = np.zeros(shape=(BATCH_SIZE, 8, 8, 6))
        outputs = np.zeros(shape=(BATCH_SIZE, 1))

        a = 0

        for j in range(GAMES):
            offset = next(game_offsets)
            pgn.seek(offset)
            game = chess.pgn.read_game(pgn)
            board = game.board()

            board_states = []

            # Iterate through moves and add them to a list:
            for move in game.main_line():
                board_states.append(tr.board_tensor(board))
                board.push(move)

            result = game.headers["Result"]

            # Now sample 8 random moves per game:
            indices = np.random.randint(low=0, high=len(board_states), size=MOVE_SAMPLES)

            for k in range(MOVE_SAMPLES):
                index = indices[k]

                # If the index is even, we know white played the move.
                if index % 2 == 0:
                    inputs[a] = board_states[index]
                    outputs[a] = RESULT_DICT[result]
                else:
                    inputs[a] = tr.mirror_board(board_states[index])
                    outputs[a] = -RESULT_DICT[result]

                a += 1

        loss = agent.train(inputs, outputs)
        print(str(i) + "\tloss: " + str(loss))

    agent.get_nn().save("model//model.h5")