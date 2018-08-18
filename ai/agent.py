# This defines the Agent object. An Agent should be able to take a board state as an input
# and produce optimal moves in its policy head and the probability of winning in its value
# head. The agent architecture is based on one of the architectures discussed in DeepMind's
# paper on AlphaGo Zero, namely the "dual-conv" variation, with some adjustments.

import position as pos
import numpy as np
import chess
from node import Node
from keras.models import Model, load_model
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, LeakyReLU, add

# Settings:
RES_BLOCKS = 19

RESULT_DICT = {
    "1-0": 1,
    "1/2-1/2": 0,
    "*": 0,
    "0-1": -1
}


# We define the functions used to create separate parts of the neural network here. The
# input is processed using a series of convolutional blocks before being passed to the
# value and policy heads. Activation functions and auxillary layers follow those described
# in the paper.
def conv_block(x):
    """Adds a convolution block to an existing tower x."""
    x = Conv2D(filters=64,
               kernel_size=3,
               strides=1,
               padding="same",
               activation="linear",

               # Should I be doing this as channels_first? Is there a difference?
               data_format="channels_last")(x)

    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU()(x)

    return x


def res_layer(x):

    input_layer = x

    x = Conv2D(filters=64,
               kernel_size=3,
               strides=1,
               padding="same",
               activation="linear",

               # Should I be doing this as channels_first? Is there a difference?
               data_format="channels_last")(input_layer)

    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU()(x)

    x = Conv2D(filters=64,
               kernel_size=3,
               strides=1,
               padding="same",
               activation="linear",

               # Should I be doing this as channels_first? Is there a difference?
               data_format="channels_last")(x)

    x = BatchNormalization(axis=3)(x)
    x = add([input_layer, x])
    x = LeakyReLU()(x)

    return x


def val_head(x):
    """Adds a value head to an existing tower x. Outputs are activated via tanh, meaning
    they are between -1 and 1."""
    x = Conv2D(filters=1,
               kernel_size=1,
               strides=1,
               data_format="channels_last",
               padding="same",
               activation="linear")(x)

    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU()(x)
    x = Flatten()(x)
    x = Dense(20, activation="linear")(x)
    x = LeakyReLU()(x)
    x = Dense(1, activation="tanh", name="value_head")(x)

    return x


class Agent:

    def __init__(self):
        self.nn = None

        # True for white, False for black
        self.color = True
        self.evaluations = 0
        self.tree = Node("root")
        self.tree.expand()

    def eval(self, state):
        """Returns a 'rating' of the given position as evaluated from the perspective
        of the current player (assumed the be white), with 1 as a win, 0 as draw, and
        -1 as a loss."""

        # Check if we are evaluating a state where it is our opponent's turn to move. In
        # that case, we'd want to return high values for moves that are good for THE OPPONENT,
        # not for us.
        eval_opponent = state.turn != self.color

        tensor = pos.board_tensor(state)

        # TODO: make this less disgusitng
        if state.is_game_over(claim_draw=True):
            if eval_opponent:
                return -RESULT_DICT[state.result(claim_draw=True)]
            else:
                return RESULT_DICT[state.result(claim_draw=True)]
        else:
            if eval_opponent:
                return self.nn.predict(np.array([pos.mirror_tensor(tensor)]))[0][0]
            else:
                return self.nn.predict(np.array([tensor]))[0][0]

    def playout(self):
        current = self.tree
        root = self.tree

        # Selection:
        # We traverse the tree to some leaf node L by selecting nodes that maximize the
        # UCB1 value.
        depth = 0
        while len(current.get_children()) > 0:

            children = current.get_children()

            best_score = -10000
            best_node = None

            for node in children:
                total_simulations = root.get_visits()
                if node.get_visits() > 0 and total_simulations > 0:
                    average_score = node.get_total() / node.get_visits()
                    visits = node.get_visits()

                    # The 2 here is the exploration factor.
                    ucb1 = average_score + 2 * np.sqrt(np.log(total_simulations) / visits)

                    if ucb1 > best_score:
                        #print("Better node: " + node.get_name() + "with score of " + str(ucb1))
                        best_score = ucb1
                        best_node = node
                else:
                    best_score = 10
                    best_node = node

            depth += 1
            current = best_node
            print(current.get_name())
            print(depth)

        # Expansion:
        # If the node is a leaf node, add a bunch of children to it.

        # NOTE: if we are using an opening tree, the initial tree might not contain
        # ALL legal moves for a particular board state, simply because they've never
        # been played. This is probably okay for us (because we trust that LCZ selects
        # good moves) but might have some consequences that I'm not seeing yet.
        current.expand()

        # TODO: maybe evaluate all the children and then backprop? I could multithread this
        # TODO: by assigning a different set of children to different threads.

        # Evaluation:
        score = self.eval(current.get_state())

        # Backprop:
        # Iterate backwards to root node and update stats as we go along.
        at_root = False
        while not at_root:
            current.update_total(score)
            current.update_visits()

            if current.get_parent() is None:
                at_root = True
            else:
                current = current.parent

    def build_nn(self):

        # TODO: fix input shape
        main_input = Input(shape=(8, 8, 30), name="main_input")
        x = conv_block(main_input)

        for i in range(RES_BLOCKS):
            x = res_layer(x)

        value_head = val_head(x)
        model = Model(inputs=[main_input], outputs=[value_head])
        model.compile(loss="mse", optimizer="adadelta")

        self.nn = model

    def play_move(self):
        for i in range(800):
            self.playout()

        best_move = ""
        most_visits = 0

        for node in self.tree.get_children():
            if node.get_visits() > most_visits:
                best_move = node.get_name()
                most_visits = node.get_visits()

        print(len(self.tree.select(best_move).get_children()))
        return chess.Move.from_uci(best_move)

    def get_tree(self):
        return self.tree

    def get_nn(self):
        return self.nn

    def load_nn(self, path):
        self.nn = load_model(path)

    def train(self, inputs, outputs):
        loss = self.nn.train_on_batch(inputs, outputs)
        return loss
