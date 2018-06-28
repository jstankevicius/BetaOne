# This defines the Agent object. An Agent should be able to take a board state as an input
# and produce optimal moves in its policy head and the probability of winning in its value
# head. The agent architecture is based on one of the architectures discussed in DeepMind's
# paper on AlphaGo Zero, namely the "dual-conv" variation, with some adjustments.

import chess
import chess.pgn
import time
from chess import Board
from queue import PriorityQueue
import numpy as np
import translator as tr
from keras.models import Model, load_model
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, LeakyReLU

# Settings:
CONV_BLOCKS = 15

# ZOBRIST_TABLE = np.full((2**32), -10, dtype=np.float64)
COMPUTED_HASHES = []

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
    """Adds a convolution block to an existing tower x."""
    x = Conv2D(filters=64,
               kernel_size=3,
               strides=1,
               padding="same",
               activation="relu",

               # Should I be doing this as channels_first? Is there a difference?
               data_format="channels_last")(x)

    x = BatchNormalization(axis=1)(x)
    x = LeakyReLU()(x)

    return x


def create_value_head(x):
    """Adds a value head to an existing tower x. Outputs are activated via tanh, meaning
    they are between -1 and 1."""
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


def get_legal_moves(board):
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

    # Because we rank each move, a PriorityQueue object is necessary for quick ordering
    # of each move by its utility without having to compute it elsewhere.
    legal_moves = PriorityQueue()

    # This is simply the index of the move in board.legal_moves. However, this is useful as
    # an arbitrary rank in a PriorityQueue, as moves that do not capture technically have
    # the same basic utility. Captures sometimes end up with the same utility as well.
    move_number = 0

    for move in board.legal_moves:
        copy = board.copy()

        # copy contains the resulting board configuration after the move is played.
        copy.push(move)

        # We set the default move value to something very high; this is because items
        # in a PriorityQueue get ranked by the "lowest" priority first. Thus, moves with
        # a high value will actually be ranked last.
        move_value = 10000

        # Check for checkmate:
        if copy.is_checkmate():

            # If the resulting action is a checkmate, we need only return one
            # element, as all others are pointless to evaluate. Checkmate is
            # automatically the highest-valued board configuration. The agent
            # should later check if the initial size of the PriorityQueue
            # returned is 1, as this will indicate a checkmating move.
            legal_moves = PriorityQueue()
            legal_moves.put((move_value, move_number, (move, copy)))
            return legal_moves

        # Check for capture:
        if board.is_capture(move):
            uci = move.uci()

            # Some locations:
            from_file = chess.FILE_NAMES.index(uci[0])
            from_rank = int(uci[1]) - 1
            to_file = chess.FILE_NAMES.index(uci[2])
            to_rank = int(uci[3]) - 1

            from_square = chess.square(from_file, from_rank)
            to_square = chess.square(to_file, to_rank)

            capturing_piece = board.piece_at(from_square)
            captured_piece = board.piece_at(to_square)

            # Check for en passant capture:
            if board.ep_square == to_square:
                to_rank -= 1
                to_square = chess.square(to_file, to_rank)
                captured_piece = board.piece_at(to_square)

            # TODO: if pawn captures into promotion, this gives a bad value.
            capturing_piece_value = PIECE_VALUES[capturing_piece.symbol().upper()]
            captured_piece_value = PIECE_VALUES[captured_piece.symbol().upper()]

            # TODO: We have a bit of a problem here. What happens if this is the endgame
            # TODO: and the king is suppsed to be highly active on the board? Should his
            # TODO: move priority initially be high, and then slowly decrease as we enter
            # TODO: the endgame? This is the most likely solution.
            move_value = capturing_piece_value - captured_piece_value

        legal_moves.put((move_value, move_number, (move, copy)))
        move_number += 1

    return legal_moves


# This is a move node that we use to construct an opening tree. Each node contains some
# stats and its child nodes, which indicate the typical responses to a particular move.
class Node:

    def __init__(self, name):
        self.name = name
        self.visits = 0
        self.won = 0
        self.moves = []

    def get_name(self):
        """Retrieves the node's name (I.E. the current move)."""
        return self.name

    def get_moves(self):
        """Returns the child nodes of the current node."""
        return self.moves

    def get_won(self):
        """Returns the number of times this node has resulted in a win.
        If we are using LeelaChessZero's PGN data as building material
        for the tree, this method is redundant, as Leela never finishes games
        in the dataset. However, this might be useful if we decide to go back
        to the old GM dataset."""
        return self.won

    def update_won(self):
        """Increments the number of times this node has resulted in a win."""
        self.won += 1

    def get_visits(self):
        """Returns the number of times the node has been visited (I.E. how many
        times it has been played)."""
        return self.visits

    def update_visits(self):
        """Increments the total number of times the node has been visited."""
        self.visits += 1

    def add_node(self, node):
        """Adds another node to the current one."""
        self.moves.append(node)

    def select(self, uci):
        """Returns a node from the node's children based on name, or None if
        no such UCI exists."""
        for node in self.moves:
            if node.get_name() == uci:
                return node
        return None

    def contains(self, uci):
        """Checks whether or not a given UCI string exists as a name of one
        of the node's children."""
        for node in self.moves:
            if node.get_name() == uci:
                return True
        return False


def create_opening_tree(games=20000, d=8):
    """Creates a tree of nodes of a given depth out of a given number of games."""

    # TODO: implement pruning_threshold parameter that determines whether or not we
    # TODO: should delete a certain node once its play frequency is lower than
    # TODO: pruning_threshold. This would require another iteration over the tree
    # TODO: at the end of the function.

    # Depending on what data we use as building material, the body of this function
    # could change drastically. LCZ's data is separated into 500,000 files, whereas
    # the GM games are contained within one.

    # Denotes the first file's number in the directory. Only usable with LCZ data.
    file_number = 17600000

    # The total number of nodes in the tree.
    total_nodes = 0

    # The root node.
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
                    current_node.update_visits()
                    if current_node.contains(uci):
                        current_node = current_node.select(uci)
                    else:
                        new_node = Node(uci)
                        total_nodes += 1
                        current_node.add_node(new_node)
                        current_node = new_node

                    depth += 1

        except AttributeError:
            print("AttributeError at game #" + str(i) + ": cannot read data from board")

        except UnicodeDecodeError:
            print("UnicodeDecodeError at game #" + str(i) + ": symbol does not match any recognized")

    print("Total nodes: " + str(total_nodes))
    return base_node


# The agent is essentially an instantiable AI instance. Given a board, it can play a move
# after performing going through search and evaluation steps.
class Agent:

    def __init__(self):
        self.nn = None
        self.evaluations = 0

    def eval(self, tensor):
        """Returns a 'rating' of the given position, with 1 indicating a white
        win and -1 indicating a black win, with 0 as draw or stalemate."""
        return self.nn.predict(np.array([tensor]))[0][0]

    """
    def search(self, state, d=3):

        # We monitor how long it actually takes for the agent to select an optimal move.
        start = time.time()

        # The following is a fairly standard implementation of minimizer-maximizer agents.
        # This is the "maximizer" function.
        def max_value(board, alpha, beta, depth):

            board_tensor = tr.board_tensor(board)

            if depth > d or board.is_checkmate():
                self.evaluations += 1
                return self.eval(board_tensor)

            # Because this is the maximizer (I.E., the value of an action can only go
            # up), we set the value to the lowest possible score, which in tanh is -1.
            value = -1
            possible_moves = get_legal_moves(board)

            while not possible_moves.empty():
                a, s = possible_moves.get()[2]

                ##################################
                # NEW ADDITION
                zobrist = tr.get_board_zobrist(s)
                if ZOBRIST_TABLE[zobrist] == -10:
                    value = max(value, min_value(s, alpha, beta, depth + 1))
                    ZOBRIST_TABLE[zobrist] = value

                else:
                    value = ZOBRIST_TABLE[zobrist]

                # value = max(value, min_value(s, alpha, beta, depth + 1))

                # CUTOFF
                if value >= beta:
                    return value

                alpha = max(alpha, value)

            return value

        # And this is the minimizer function.
        def min_value(board, alpha, beta, depth):

            board_tensor = tr.board_tensor(board)

            # TODO: implement instant return on checkmate
            if depth > d or board.is_checkmate():
                self.evaluations += 1
                return self.eval(board_tensor)

            value = 1
            possible_moves = get_legal_moves(board)

            while not possible_moves.empty():
                action, state = possible_moves.get()[2]

                ##################################
                # NEW ADDITION
                zobrist = tr.get_board_zobrist(state)
                if ZOBRIST_TABLE[zobrist] == -10:
                    value = min(value, max_value(state, alpha, beta, depth + 1))
                    ZOBRIST_TABLE[zobrist] = value
                else:
                    value = ZOBRIST_TABLE[zobrist]

                # value = min(value, max_value(state, alpha, beta, depth + 1))

                # CUTOFF
                if value <= alpha:
                    return value

                beta = min(beta, value)

            return value

        # Body of alphabeta_search starts here:
        # The default test cuts off at depth d or at a terminal state

        # We now have a list of tuple pairs of actions and their resulting board
        # configurations.
        children = get_legal_moves(state)
        best_pair = children.get()[2]

        # We search the first branch available, since no other information is given.
        best_score = min_value(best_pair[1], -1, 1, 0)

        # We now iterate through all legal moves and get their scores:
        while not children.empty():
            x = children.get()
            x_score = min_value(x[2][1], -1, 1, 0)

            print("Evaluating " + x[2][0].uci() + "; value: " + str(x_score))

            if x_score > best_score:
                print("Found a better move than " + best_pair[0].uci() + ": " + x[2][0].uci())
                best_pair, best_score = x[2], x_score

        end = time.time()
        elapsed = end - start
        print("Evaluated " + str(self.evaluations) + " positions in " + str(round(elapsed, 3)) + " seconds")

        return best_pair[0]
    """

    def build_nn(self):
        main_input = Input(shape=(8, 8, 12), name="main_input")
        x = create_conv_block(main_input)

        for i in range(CONV_BLOCKS):
            x = create_conv_block(x)

        value_head = create_value_head(x)
        model = Model(inputs=[main_input], outputs=[value_head])
        model.compile(loss="mse", optimizer="sgd")

        self.nn = model

    def get_nn(self):
        return self.nn

    #def play_move(self, state):
    #    return self.search(state)

    def load_nn(self, path):
        self.nn = load_model(path)

    def train(self, inputs, outputs):
        loss = self.nn.train_on_batch(inputs, outputs)
        return loss
