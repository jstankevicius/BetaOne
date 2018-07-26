from agent import Agent
from chess import Board
import chess.pgn
import position as pos
import numpy as np
import traceback
import random

GAMES = 500000
BATCHES = 1

RESULT_DICT = {
    "1-0": 1,
    "1/2-1/2": 0,
    "*": 0,
    "0-1": -1
}

# The number of the first game
START = 17600000
base_board = Board()
agent = Agent()
agent.build_nn()


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


def main():
    for batch in range(BATCHES):

        inputs, outputs = get_batch(batch_size=10)


        #loss = agent.train(inputs, outputs)
        #print(str(batch) + ".\tloss: " + str(loss))

        if (batch + 1) % 300 == 0:
            # perform base evaluation:
            base_eval = agent.eval(base_board).astype(np.str_)
            print("BASE EVAL: " + base_eval)
            agent.get_nn().save("model//model" + str(batch + 1) + ".h5")


main()