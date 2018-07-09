# This defines the Agent object. An Agent should be able to take a board state as an input
# and produce optimal moves in its policy head and the probability of winning in its value
# head. The agent architecture is based on one of the architectures discussed in DeepMind's
# paper on AlphaGo Zero, namely the "dual-conv" variation, with some adjustments.

import chess
import chess.pgn
from chess import Board
from queue import PriorityQueue
import numpy as np
import random
import translator as tr
from keras.models import Model, load_model
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, LeakyReLU

# Settings:
CONV_BLOCKS = 15

# ZOBRIST_TABLE = np.full((2**32), -10, dtype=np.float16)
COMPUTED_HASHES = []

RESULT_DICT = {
    "1-0": 1,
    "1/2-1/2": 0,
    "*": 0,
    "0-1": -1
}

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


def get_unordered_legal_moves(board):
    """Simple variant of getting move/result pairs from a board. Does not order by capture
    or any other heuristics."""
    moves = []

    for move in board.legal_moves:
        result = board.copy()
        result.push(move)

        if result.is_checkmate():
            return [(move, result)]

        moves.append((move, result))

    return moves


def get_ordered_legal_moves(board):
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
        result = board.copy()

        # copy contains the resulting board configuration after the move is played.
        result.push(move)

        # We set the default move value to something very high; this is because items
        # in a PriorityQueue get ranked by the "lowest" priority first. Thus, moves with
        # a high value will actually be ranked last.
        move_value = 10000

        # Check for checkmate:
        if result.is_checkmate():

            # If the resulting action is a checkmate, we need only return one
            # element, as all others are pointless to evaluate. Checkmate is
            # automatically the highest-valued board configuration. The agent
            # should later check if the initial size of the PriorityQueue
            # returned is 1, as this will indicate a checkmating move.
            legal_moves = PriorityQueue()
            legal_moves.put((move_value, move_number, (move, result)))
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
            if board.is_en_passant(move):

                # The difference in ranks in EP capture is always 1 or -1. We can
                # use this to shift the location of where we think the captured piece
                # is.
                rank_diff = to_rank - from_rank
                ep_capture_square = chess.square(to_file, to_rank - rank_diff)
                captured_piece = board.piece_at(ep_capture_square)

            # TODO: if pawn captures into promotion, this gives a bad value.
            capturing_piece_value = PIECE_VALUES[capturing_piece.symbol().upper()]
            captured_piece_value = PIECE_VALUES[captured_piece.symbol().upper()]

            # TODO: We have a bit of a problem here. What happens if this is the endgame
            # TODO: and the king is suppsed to be highly active on the board? Should his
            # TODO: move priority initially be high, and then slowly decrease as we enter
            # TODO: the endgame? This is the most likely solution.
            move_value = capturing_piece_value - captured_piece_value

        legal_moves.put((move_value, move_number, (move, result)))
        move_number += 1

    return legal_moves


# This is a move node that we use to construct an opening tree. Each node contains some
# stats and its child nodes, which indicate the typical responses to a particular move.
class Node:

    def __init__(self, name):
        self.name = name
        self.visits = 0
        self.total_score = 0
        self.children = []
        self.state = Board()
        self.parent = None

    def get_name(self):
        """Retrieves the node's name (I.E. the current move)."""
        return self.name

    def get_children(self):
        """Returns the child nodes of the current node."""
        return self.children

    def expand(self, legal_moves):
        for move, result in legal_moves:
            child_node = Node(move.uci())
            child_node.set_state(result)
            child_node.set_parent(self)
            self.add_node(child_node)

    def get_state(self):
        return self.state

    def set_state(self, board):
        self.state = board

    def get_total(self):
        return self.total_score

    def get_parent(self):
        """Returns the parent node."""
        return self.parent

    def update_total(self, n):
        self.total_score += n

    def get_visits(self):
        """Returns the number of times the node has been visited (I.E. how many
        times it has been played)."""
        return self.visits

    def update_visits(self):
        """Increments the total number of times the node has been visited."""
        self.visits += 1

    def set_parent(self, parent):
        self.parent = parent

    def add_node(self, node):
        """Adds another node to the current one."""
        self.children.append(node)

    def remove_node(self, uci):
        if self.contains(uci):
            node = self.select(uci)
            del node

    def select(self, uci):
        """Returns a node from the node's children based on name, or None if
        no such UCI exists."""
        for node in self.children:
            if node.get_name() == uci:
                return node
        return None

    def contains(self, uci):
        """Checks whether or not a given UCI string exists as a name of one
        of the node's children."""
        for node in self.children:
            if node.get_name() == uci:
                return True
        return False


def create_opening_tree(games=10000, d=8):
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


class Agent:

    def __init__(self):
        self.nn = None
        self.evaluations = 0
        self.tree = Node("root")

    def eval(self, tensor):
        """Returns a 'rating' of the given position, with 1 indicating a white
        win and -1 indicating a black win, with 0 as draw or stalemate."""
        return self.nn.predict(np.array([tensor]))[0][0]

    def search(self, iterations=2000):
        """MCTS search from the root node (self.tree) until a given depth."""
        current = self.tree

        for i in range(iterations):

            # Selection:
            # We pick nodes with the maximum UCB1 value.
            children = current.get_children()

            if len(current.get_children()) > 0:

                best_score = -10000
                best_node = None

                for node in children:
                    if node.get_visits() > 0:
                        average_score = node.get_total() / node.get_visits()
                        total_simulations = i
                        visits = node.get_visits()

                        # The 2 here is the exploration factor.
                        ucb1 = average_score + 2 * np.sqrt(np.log(total_simulations)/visits)

                        if ucb1 > best_score:
                            best_score = ucb1
                            best_node = node
                    else:
                        best_score = 10
                        best_node = node

                current = best_node

            else:
                # Expansion
                # If the node is a leaf node, add a bunch of children to it:
                legal_moves = get_unordered_legal_moves(current.get_state())
                current.expand(legal_moves)

                # Simulation
                current_depth = 0

                while not current.get_state().is_game_over(claim_draw=True) and current_depth < 8:
                    action, result = random.choice(legal_moves)

                    child_node = Node(action.uci())
                    child_node.set_state(result)
                    child_node.set_parent(current)
                    current = child_node

                    legal_moves = get_unordered_legal_moves(current.get_state())
                    current_depth += 1

                # Backprop
                score = self.eval(tr.board_tensor(current.get_state()))
                if current.get_state().is_game_over(claim_draw=True):
                    score = RESULT_DICT[current.get_state().result(claim_draw=True)]

                while current.parent is not None:
                    current.update_total(score)
                    current.update_visits()
                    current = current.parent

        children = self.tree.get_children()
        best_node = children[0]
        best_score = best_node.get_total() / best_node.get_visits()

        for node in children:
            value = node.get_total() / node.get_visits()
            print(node.get_name() + ": value: " + str(value))

            if value > best_score:
                best_score = value
                best_node = node

        return chess.Move.from_uci(best_node.get_name())

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

    def play_move(self):
        return self.search()

    def load_nn(self, path):
        self.nn = load_model(path)

    def train(self, inputs, outputs):
        loss = self.nn.train_on_batch(inputs, outputs)
        return loss
