import pygsheets
from agent import Agent
from chess import Board
import chess.pgn
import position as pos
import numpy as np
import time
import random
import traceback
import datetime


gc = pygsheets.authorize(service_file="C://Users//Neda//Downloads//creds.json")
sh = gc.open('Value Network Performance')

# Select sheet:
wks = sh[0]

dataset_path = "D://data//qchess//games//"

RESULT_DICT = {
    "1-0": 1,
    "1/2-1/2": 0,
    "*": 0,
    "0-1": -1
}

# The number of the first game
START = 17600000
GAMES = 50000
TRAINING = False

pgn = open("D://data//FEN.txt")
d = 0


def get_batch(batch_size=256):
    inputs = np.zeros(shape=(batch_size, 8, 8, 30))
    outputs = np.zeros(shape=(batch_size, 1))
    successful = 0

    pgn.seek(0, 2)
    size = pgn.tell()
    random_set = sorted(random.sample(range(size), batch_size))

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

        except ValueError:
            pass

    return inputs, outputs


def training_loop():

    # optimizer = wks.cell("B2").value
    batch_size = int(wks.cell("B3").value)
    batches = int(wks.cell("B4").value)
    load_existing = wks.cell("B5").value
    load_path = wks.cell("B6").value
    update_frequency = int(wks.cell("B7").value)
    save_frequency = int(wks.cell("B8").value)

    base_board = Board()
    agent = Agent()

    writes = 0

    if load_existing == "yes":
        agent.load_nn(load_path)
    else:
        agent.build_nn()

    for batch in range(batches):

        inputs, outputs = get_batch(batch_size=batch_size)
        loss = agent.train(inputs, outputs)

        # Because we don't want to spam lots of information, UPDATE_FREQUENCY
        # is used to determine when we should send the data to the sheet.
        if (batch + 1) % update_frequency == 0:
            base_eval = agent.eval(base_board).astype(np.str_)
            wks.update_cell("A" + str(13 + writes), batch)
            wks.update_cell("B" + str(13 + writes), str(loss))
            wks.update_cell("C" + str(13 + writes), str(base_eval))
            writes += 1

        if (batch + 1) % save_frequency == 0:
            # perform base evaluation:
            agent.get_nn().save("C://Users//Neda//Documents//GitHub//Shah//ai//models//model" + str(batch + 1) + ".h5")


while True:

    # Do nothing while we wait for scheduled launch.
    while not TRAINING:
        START_TIME = wks.cell("B9").value
        current_time = datetime.datetime.now().strftime("%H:%M")
        if current_time == START_TIME:
            TRAINING = True
        time.sleep(1)

    training_loop()
    TRAINING = False
