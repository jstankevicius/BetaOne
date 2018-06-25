# This defines the Agent object. An Agent should be able to take a board state as an input
# and produce optimal moves in its policy head and the probability of winning in its value
# head. The agent architecture is based on one of the architectures discussed in DeepMind's
# paper on AlphaGo Zero, namely the "dual-conv" variation, with some adjustments.

import chess
import math
import random
import time
import numpy as np
from chess import Board
import translator as tr
from keras.models import Model, load_model
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, LeakyReLU

# Settings:
CONV_BLOCKS = 15


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


def get_children(board_state):
    """Returns all possible legal moves from board_state and the resulting
    board configurations."""
    children = []

    for move in board_state.legal_moves:
        board_state.push(move)
        children.append((move, board_state.copy()))
        board_state.pop()

    return children

# Here we define the Agent class.
class Agent:

    def __init__(self,  *, epsilon=0.1, gamma=0.9):
        self.nn = None
        self.epsilon = epsilon
        self.gamma = gamma
        self.evaluations = 0

    def eval(self, tensor):
        return self.nn.predict(np.array([tensor]))[0][0]

    def alphabeta_search(self, state, d=3):
        start = time.time()

        def max_value(state, alpha, beta, depth):
            state_tensor = tr.board_tensor(state)

            if cutoff_test(depth):
                self.evaluations += 1
                return self.eval(state_tensor)

            v = -1

            for (a, s) in get_children(state):
                v = max(v, min_value(s, alpha, beta, depth + 1))

                if v >= beta:
                    return v

                alpha = max(alpha, v)

            return v

        def min_value(state, alpha, beta, depth):
            state_tensor = tr.board_tensor(state)
            if cutoff_test(depth):
                self.evaluations += 1
                return self.eval(state_tensor)

            v = 1

            for (a, s) in get_children(state):
                v = min(v, max_value(s, alpha, beta, depth + 1))

                if v <= alpha:
                    return v

                beta = min(beta, v)

            return v

        # Body of alphabeta_search starts here:
        # The default test cuts off at depth d or at a terminal state
        cutoff_test = lambda depth: depth > d

        # We now have a list of tuple pairs of actions and their resulting board
        # configurations.
        children = get_children(state)

        best_pair = children[0]

        # We search the first branch available, since no other information is given.
        best_score = min_value(best_pair[1], -1, 1, 0)

        # We now iterate through all legal moves and get their scores:
        for x in children:
            x_score = min_value(x[1], -1, 1, 0)
            if x_score > best_score:
                print("Found a better move than " + best_pair[0].uci() + ": " + x[0].uci())
                best_pair, best_score = x, x_score

        end = time.time()
        elapsed = end - start
        print("Evaluated " + str(self.evaluations) + " positions in " + str(round(elapsed, 3)) + " seconds")

        return best_pair[0]

    def build_nn(self):
        main_input = Input(shape=(8, 8, 6), name="main_input")
        x = create_conv_block(main_input)

        for i in range(CONV_BLOCKS):
            x = create_conv_block(x)

        value_head = create_value_head(x)
        model = Model(inputs=[main_input], outputs=[value_head])
        model.compile(loss="mse", optimizer="sgd")

        self.nn = model

    def get_nn(self):
        return self.nn

    def play_move(self, state):
        return self.alphabeta_search(state)

    def load_nn(self, path):
        self.nn = load_model(path)

    def train(self, inputs, outputs):
        loss = self.nn.train_on_batch(inputs, outputs)
        return loss


a = Agent()
a.load_nn("model//model.h5")
b = Board()
print(a.play_move(b))