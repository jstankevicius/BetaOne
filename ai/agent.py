# This defines the Agent object. An Agent should be able to take a board state as an input
# and produce optimal moves in its policy head and the probability of winning in its value
# head. The agent architecture is based on one of the architectures discussed in DeepMind's
# paper on AlphaGo Zero, namely the "dual-conv" variation, with some adjustments.

import chess
import numpy as np
from keras.models import load_model
from keras.models import Sequential, load_model, Model
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, LeakyReLU


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

    def __init__(self,  *, epsilon=0.1):
        self.nn = None
        self.epsilon = epsilon

    def play_move(self, board_state):
        # This should utilize some sort of tree search
        pass

    def build_nn(self):
        main_input = Input(shape=(8, 8, 12), name="main_input")
        x = create_conv_block(main_input)

        for i in range(10):
            x = create_conv_block(x)

        value_head = create_value_head(x)
        policy_head = create_policy_head(x)

        model = Model(inputs=[main_input], outputs=[value_head, policy_head])
        model.compile(loss={"value_head": "mean_squared_error",
                            "policy_head": "categorical_crossentropy"},
                      optimizer="sgd",
                      loss_weights={"value_head": 0.5, "policy_head": 0.5})

        self.nn = model

    def load_nn(self):
        pass

    def get_nn(self):
        return self.nn

    def reset(self):
        pass

    def train(self):
        pass
