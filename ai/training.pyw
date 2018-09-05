from agent import Agent
from chess import Board
from datetime import datetime
import chess.pgn
import position as pos
import numpy as np
import random

# The "absolute" result of the game, from the perspective of an observer.
RESULT_DICT = {
    "1-0": 1,
    "1/2-1/2": 0,
    "*": 0,
    "0-1": -1
}


pgn = open("D://data//FEN.txt")


def get_batch(batch_size=256):

    # Input and output tensors.
    inputs = np.zeros(shape=(batch_size, 8, 8, 30))
    outputs = np.zeros(shape=(batch_size, 1))

    # "Successful" is simply how many lines were successfully parsed.
    successful = 0
    i = 0

    pgn.seek(0, 2)
    size = pgn.tell()
    random_set = sorted(random.sample(range(size), batch_size + 10))

    # We repeat random sampling of moves until we completely fill the inputs and outputs.
    while successful < batch_size:
        try:
            pgn.seek(random_set[i])

            pgn.readline()
            line = pgn.readline()

            board_fen = line[:line.find("  ")]
            board = Board(fen=board_fen)

            tensor = pos.board_tensor(board)
            result = int(line[-3:-1])

            if board.turn == chess.BLACK:
                result *= -1

            inputs[successful] = tensor
            outputs[successful] = result
            successful += 1
            i += 1

        except ValueError:
            i += 1

    return inputs, outputs


def get_latest():
    with open("latest.txt", "r") as file:
        return file.readline().strip()


def training_loop():

    # load last agent
    agent = Agent()
    agent.load_nn(get_latest() + ".h5")

    for batch in range(100):

        inputs, outputs = get_batch(batch_size=2048)
        agent.train(inputs, outputs)

    # Saving procedure:
    t = datetime.now()
    t_string = t.strftime("%m-%d_%H:%N")

    agent.get_nn().save("C://Users//Neda//Documents//GitHub//Shah//ai//models//model" + t_string + ".h5")

    with open("latest.txt", "w") as file:
        file.write(t_string)


while True:
    training_loop()