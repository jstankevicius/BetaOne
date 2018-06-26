# This defines the Agent object. An Agent should be able to take a board state as an input
# and produce optimal moves in its policy head and the probability of winning in its value
# head. The agent architecture is based on one of the architectures discussed in DeepMind's
# paper on AlphaGo Zero, namely the "dual-conv" variation, with some adjustments.

import chess
import chess.pgn
from chess import Board
import time
from queue import PriorityQueue
import numpy as np
import translator as tr
from keras.models import Model, load_model
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, LeakyReLU

# Settings:
CONV_BLOCKS = 15
PIECE_VALUES = {
    "P": 100,
    "N": 350,
    "B": 350,
    "R": 525,
    "Q": 1000,
    "K": 2000
}


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

def old(board):

    children = []

    for move in board.legal_moves:
        copy = board.copy()
        copy.push(move)
        children.append((move, copy))

    return children


def get_children(board):
    """Returns all possible legal moves from board_state and the resulting
    board configurations. Pairs are ranked in the following order:
    1. Does the move result in checkmate? If yes, we delete all other moves
       and return only this pair, with a flag to skip evaluation - if we are
       playing, the agent need not evaluate a move that immediately results in
       a win. If our opponent is playing we assume that they will not miss checkmate,
       for logically there are no other moves of a higher value.

    2. Captures, ranked by the difference between capturing and captured pieces.

    3. Everything else? The next order is still to be determined.
    """
    children = PriorityQueue()
    move_number = 0

    for move in board.legal_moves:
        copy = board.copy()
        copy.push(move)
        move_value = 10000

        # Check for checkmate:
        if board.is_checkmate():

            # Immediately construct PQ and return it:
            children = PriorityQueue()
            children.put((move_value, move_number, (move, copy)))
            return children

        # Check for capture:
        from_file, from_rank, to_file, to_rank = tr.get_destinations(move.uci())
        
        try:
            capturing_piece = board.piece_at(chess.square(from_file, from_rank))
            capturing_piece_value = PIECE_VALUES[capturing_piece.symbol().upper()]

            captured_piece = board.piece_at(chess.square(to_file, to_rank))
            captured_piece_value = PIECE_VALUES[captured_piece.symbol().upper()]

            move_value = capturing_piece_value - captured_piece_value

        except AttributeError:
            pass

        children.put((move_value, move_number, (move, copy)))
        move_number += 1

    return children


class Node:

    def __init__(self, name):
        self.name = name
        self.won = 0
        self.total = 0
        self.moves = []

    def get_name(self):
        return self.name

    def get_moves(self):
        return self.moves

    def get_won(self):
        return self.won

    def get_total(self):
        return self.total

    def update_won(self):
        self.won += 1

    def update_total(self):
        self.total += 1

    def add_node(self, node):
        self.moves.append(node)

    def select(self, uci):
        for node in self.moves:
            if node.get_name() == uci:
                return node
        return None

    def contains(self, uci):
        for node in self.moves:
            if node.get_name() == uci:
                return True
        return False


def create_opening_tree(games=20000, d=8, pruning_threshold=0.05):
    file_number = 17600000
    total_nodes = 0

    base_node = Node("base_node")

    for i in range(games):
        try:
            pgn = open("D://data//qchess//games//" + str(file_number + i) + ".pgn")
            game = chess.pgn.read_game(pgn)

            # The current node we are at.
            current_node = base_node
            depth = 0

            for move in game.main_line():
                if depth < d:
                    uci = move.uci()

                    if current_node.contains(uci):
                        current_node = current_node.select(uci)
                    else:
                        new_node = Node(uci)
                        total_nodes += 1
                        current_node.add_node(new_node)
                        current_node = new_node

                    current_node.update_total()
                    depth += 1

        except AttributeError:
            print("AttributeError at game #" + str(i) + ": cannot read data from board")

        except UnicodeDecodeError:
            print("UnicodeDecodeError at game #" + str(i) + ": symbol does not match any recognized")

    print("Total nodes: " + str(total_nodes))
    return base_node


# Here we define the Agent class.
class Agent:

    def __init__(self,  *, epsilon=0.1, gamma=0.9):
        self.nn = None
        self.epsilon = epsilon
        self.gamma = gamma
        self.evaluations = 0

    def eval(self, tensor):
        return self.nn.predict(np.array([tensor]))[0][0]

    # TODO: clean up/document code
    def search(self, state, d=3):
        start = time.time()

        def max_value(state, alpha, beta, depth):
            state_tensor = tr.board_tensor(state)

            if cutoff_test(depth):
                self.evaluations += 1
                return self.eval(state_tensor)

            v = -1

            possible_moves = get_children(state)
            while not possible_moves.empty():
                a, s = possible_moves.get()[2]
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

            possible_moves = get_children(state)
            while not possible_moves.empty():
                a, s = possible_moves.get()[2]
                v = min(v, max_value(s, alpha, beta, depth + 1))

                if v <= alpha:
                    return v

                beta = min(beta, v)

            return v

        # Body of alphabeta_search starts here:
        # The default test cuts off at depth d or at a terminal state
        cutoff_test = lambda depth: depth > d or state.is_checkmate()

        # We now have a list of tuple pairs of actions and their resulting board
        # configurations.
        children = get_children(state)

        first_pair = children.get()[2]
        best_pair = first_pair

        # We search the first branch available, since no other information is given.
        best_score = min_value(best_pair[1], -1, 1, 0)

        # We now iterate through all legal moves and get their scores:
        while not children.empty():
            x = children.get()
            print("Evaluating " + x[2][0].uci())

            x_score = min_value(x[2][1], -1, 1, 0)
            if x_score > best_score:
                print("Found a better move than " + best_pair[0].uci() + ": " + x[2][0].uci())
                best_pair, best_score = x[2], x_score

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
        return self.search(state)

    def load_nn(self, path):
        self.nn = load_model(path)

    def train(self, inputs, outputs):
        loss = self.nn.train_on_batch(inputs, outputs)
        return loss


a = Agent()
b = Board()
a.load_nn("model//model.h5")
a.play_move(b)
