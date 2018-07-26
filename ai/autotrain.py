import pygsheets
from agent import Agent
from chess import Board
import chess.pgn
import position as pos
import numpy as np
import traceback
import time
import datetime


gc = pygsheets.authorize(service_file="C://Users//Justas//Desktop//creds.json")
sh = gc.open('Value Network Performance')

# Select sheet:
wks = sh[0]

OPTIMIZER = wks.cell("B2").value
BATCH_SIZE = int(wks.cell("B3").value)
BATCHES = int(wks.cell("B4").value)
LOAD_EXISTING = wks.cell("B5").value
LOAD_PATH = wks.cell("B6").value
UPDATE_FREQUENCY = int(wks.cell("B7").value)
SAVE_FREQUENCY = int(wks.cell("B8").value)

RESULT_DICT = {
    "1-0": 1,
    "1/2-1/2": 0,
    "*": 0,
    "0-1": -1
}

# The number of the first game
START = 17600000
GAMES = 500000
TRAINING = False




# While loop seems iffy.
def get_batch(batch_size=256):
    successful = 0

    inputs = np.zeros(shape=(batch_size, 8, 8, 12))
    outputs = np.zeros(shape=(batch_size, 1))

    # We repeat random sampling of moves until we completely fill the inputs and outputs.
    while successful < batch_size:
        file_number = np.random.randint(low=START, high=START + GAMES)

        with open("D://data//qchess//games//" + str(file_number) + ".pgn") as pgn:

            try:
                game = chess.pgn.read_game(pgn)
                board = game.board()
                positions = []

                ply = 0

                # All positions must be from the perspective of white, with black to play next.
                for move in game.main_line():
                    board.push(move)
                    board_tensor = pos.board_tensor(board)

                    # If the ply is odd (meaning it is black's turn to play), we flip the tensor.
                    if ply % 2 != 0:
                        board_tensor = pos.mirror_tensor(board_tensor)

                    positions.append(board_tensor)
                    ply += 1

                # Assuming we have iterated through the entire game, we check for the result.
                result = RESULT_DICT[board.result(claim_draw=True)]
                move_index = np.random.randint(low=0, high=len(positions))

                if np.mod(move_index, 2) != 0:
                    # If black is to play from this move, we negate the result, as the neural network
                    # always evaluates from the perspective of white.
                    result *= -1

                inputs[successful] = positions[move_index]
                outputs[successful] = result

                # Some diagnostics
                # pos.tensor_to_board(inputs[successful])
                # print("Winner from player's perspective:" + str(outputs[successful]))
                # print("Absolute winner: " + board.result(claim_draw=True))
                # print("Agent evaluation: " + str(agent.get_nn().predict(np.array([inputs[successful]]))[0][0]))
                # print()
                successful += 1
                pgn.close()

            except AttributeError:
                print("AttributeError at " + str(file_number) + ".pgn: cannot read data from board")
                print(str(traceback.format_exc()))
                pgn.close()

            except UnicodeDecodeError:
                print("UnicodeDecodeError at " + str(file_number) + ".pgn: symbol does not match any recognized")
                print(str(traceback.format_exc()))
                pgn.close()

    return inputs, outputs


def training_loop():

    base_board = Board()
    agent = Agent()

    writes = 0

    if LOAD_EXISTING == "yes":
        agent.load_nn(LOAD_PATH)
    else:
        agent.build_nn()

    for batch in range(BATCHES):

        inputs, outputs = get_batch(batch_size=BATCH_SIZE)
        time.sleep(2)
        loss = agent.train(inputs, outputs)

        # Because we don't want to spam lots of information, UPDATE_FREQUENCY
        # is used to determine when we should send the data to the sheet.
        if (batch + 1) % UPDATE_FREQUENCY == 0:
            base_eval = agent.eval(base_board).astype(np.str_)
            wks.update_cell("A" + str(13 + writes), batch)
            wks.update_cell("B" + str(13 + writes), loss)
            wks.update_cell("C" + str(13 + writes), base_eval)
            writes += 1

        if (batch + 1) % SAVE_FREQUENCY == 0:
            # perform base evaluation:
            agent.get_nn().save("model//model" + str(batch + 1) + ".h5")


while True:
    print("Idling...")

    # Do nothing while we wait for scheduled launch.
    while not TRAINING:
        START_TIME = wks.cell("B9").value
        current_time = datetime.datetime.now().strftime("%H:%M")
        if current_time == START_TIME:
            TRAINING = True
        time.sleep(1)

    print("Starting training loop...")
    training_loop()