# This defines the Agent object. An Agent should be able to take a board state as an input
# and produce optimal moves in its policy head and the probability of winning in its value
# head. The agent architecture is based on one of the architectures discussed in DeepMind's
# paper on AlphaGo Zero, namely the "dual-conv" variation, with some adjustments.

import chess
import math
import random
import numpy as np
import translator as tr
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, LeakyReLU

# Settings:
CONV_BLOCKS = 10

# We define the functions used to create separate parts of the neural network here. The
# input is processed using a series of convolutional blocks before being passed to the
# value and policy heads. Activation functions and auxillary layers follow those described
# in the paper.
def create_conv_block(x):
    # Convolution block:
    x = Conv2D(filters=64,
               kernel_size=3,
               strides=1,
               padding="same",
               activation="relu",
               data_format="channels_last")(x)

    x = BatchNormalization(axis=1)(x)
    x = LeakyReLU()(x)

    return x


def create_value_head(x):
    x = Conv2D(filters=1,
               kernel_size=1,
               strides=1,
               data_format="channels_last",
               padding="same",
               activation="linear")(x)

    x = BatchNormalization(axis=1)(x)
    x = LeakyReLU()(x)
    x = Flatten()(x)
    x = Dense(20, activation="linear")(x)
    x = LeakyReLU()(x)
    x = Dense(1, activation="tanh", name="value_head")(x)

    return x


def create_policy_head(x):
    x = Conv2D(filters=12,
               kernel_size=1,
               strides=1,
               data_format="channels_last",
               padding="same",
               activation="linear")(x)

    x = BatchNormalization(axis=1)(x)
    x = LeakyReLU()(x)
    x = Dense(64, activation="linear", name="policy_head")(x)

    return x


# Here we define the Agent class.
class Agent:

    def __init__(self,  *, epsilon=0.1, gamma=0.9):
        self.nn = None
        self.memory = []
        self.epsilon = epsilon
        self.gamma = gamma

    def remember(self, state_matrix, action_matrix):
        self.memory.append((state_matrix, action_matrix))

    def play_move(self, board_state):
        board_tensor = tr.board_tensor(board_state)
        action_tensor = self.nn.predict(np.array([board_tensor]))[0]
        move_dict = tr.matrix_to_move(action_tensor, board_state)

        # Select a random move:
        try:
            key = random.choice(list(move_dict))

            best_move = {
                "uci": key,
                "row": move_dict[key][0],
                "col": move_dict[key][1],
                "index": move_dict[key][2],
                "rating": move_dict[key][1]
            }

            if np.random.rand() > self.epsilon:
                for uci in move_dict.keys():
                    move = move_dict[uci]
                    rating = move[3]
                    if rating > best_move["rating"]:
                        best_move["uci"] = uci
                        best_move["row"] = move[0]
                        best_move["col"] = move[1]
                        best_move["index"] = move[2]
                        best_move["rating"] = move[3]

            # Play the actual move:
            board_state.push(chess.Move.from_uci(best_move["uci"]))

            # Reformat matrix and remember:
            perfect_output = np.zeros(shape=(8, 8, 64))
            perfect_output[best_move["row"], best_move["col"], best_move["index"]] = 1

            self.remember(board_tensor, action_tensor)

        except IndexError:
            print("IndexError encountered with this configuration:")
            print(board_state)
            print("Stalemate: " + str(board_state.is_stalemate()))
            print("Result: " + board_state.result())



    def build_nn(self):
        main_input = Input(shape=(8, 8, 12), name="main_input")
        x = create_conv_block(main_input)

        for i in range(CONV_BLOCKS):
            x = create_conv_block(x)

        # value_head = create_value_head(x)
        policy_head = create_policy_head(x)

        model = Model(inputs=[main_input], outputs=[policy_head])
        model.compile(loss={"policy_head": "categorical_crossentropy"},
                      optimizer="sgd",
                      loss_weights={"policy_head": 0.5})

        self.nn = model

    def get_nn(self):
        return self.nn

    def reset(self):
        pass

    def train(self, offset, *, batch_size=10):
        memsize = len(self.memory)

        inputs = np.zeros(shape=(batch_size, 8, 8, 12))
        outputs = np.zeros(shape=(batch_size, 8, 8, 64))

        # The inputs and outputs of the winning agent.
        winning_sample = []

        # The offset parameter defines where to begin sampling the winning positions
        # from. Because each player in chess HAS to make a move (we are assuming that
        # passes are not allowed), when black wins every other move from the second
        # move contains the desired input and output. If white wins, there is no offset.
        sample_index = offset
        for i in range(int(math.ceil(memsize/2)) - 1):
            winning_sample.append(self.memory[sample_index])
            sample_index += 2

        indices = np.random.randint(low=0, high=len(winning_sample), size=batch_size)

        for i in range(len(indices)):
            index = indices[i]

            inputs[i] = winning_sample[index][0]
            outputs[i] = winning_sample[index][1]

        self.nn.train_on_batch({"main_input": inputs},
                    {"policy_head": outputs})


