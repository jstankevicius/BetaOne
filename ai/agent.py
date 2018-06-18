# This defines the Agent object. An Agent should be able to take a board state as an input
# and produce optimal moves in its policy head and the probability of winning in its value
# head. The agent architecture is based on one of the architectures discussed in DeepMind's
# paper on AlphaGo Zero, namely the "dual-conv" variation, with some adjustments.

import chess
import numpy as np
from keras.models import Sequential, load_model, Model
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, LeakyReLU

# Global variables are denoted with capital letters for readability. These are constant (after
# being created, of course) and are not modified by any function other than the function that
# populates each list.

# ENCODINGS contains a symbol/one-hot pair, with the one-hot encoding
# denoting the type of piece occupying the square.
PIECES = ("r", "n", "b", "q", "k", "p", "R", "N", "B", "Q", "K", "P")
ENCODINGS = dict()

# This array is populated with x and y coordinate changes corresponding
# to each index of an action matrix.
INDEX_TO_MOVE = np.zeros(shape=(64, 2), dtype=np.int64)

# Settings:
ROWS = 8
COLS = 8
PIECE_TYPES = 6
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


# The functions below deal with creating/converting matrices.
# TODO: Fix this atrocity
def create_index_to_move_encodings():
    linear_index = 0

    # Rook movement up:
    for dy in range(1, 8):
        INDEX_TO_MOVE[linear_index] = np.array([0, dy])
        linear_index += 1

    # Rook movement down:
    for dy in range(1, 8):
        INDEX_TO_MOVE[linear_index] = np.array([0, -dy])
        linear_index += 1

    # Rook movement left:
    for dx in range(1, 8):
        INDEX_TO_MOVE[linear_index] = np.array([-dx, 0])
        linear_index += 1

    # Rook movement right:
    for dy in range(1, 8):
        INDEX_TO_MOVE[linear_index] = np.array([dy, 0])
        linear_index += 1

    # Bishop movement up diagonally right:
    for dxy in range(1, 8):
        INDEX_TO_MOVE[linear_index] = np.array([dxy, dxy])
        linear_index += 1

    # Bishop movement up diagonally left:
    for dxy in range(1, 8):
        INDEX_TO_MOVE[linear_index] = np.array([-dxy, dxy])
        linear_index += 1

    # Bishop movement down diagonally right:
    for dxy in range(1, 8):
        INDEX_TO_MOVE[linear_index] = np.array([dxy, -dxy])
        linear_index += 1

    # Bishop movement down diagonally left:
    for dxy in range(1, 8):
        INDEX_TO_MOVE[linear_index] = np.array([-dxy, -dxy])
        linear_index += 1

    # Knight movement:
    INDEX_TO_MOVE[56] = np.array([1, 2])
    INDEX_TO_MOVE[57] = np.array([2, 1])
    INDEX_TO_MOVE[58] = np.array([2, -1])
    INDEX_TO_MOVE[59] = np.array([1, -2])
    INDEX_TO_MOVE[60] = np.array([-1, -2])
    INDEX_TO_MOVE[61] = np.array([-2, -1])
    INDEX_TO_MOVE[62] = np.array([-2, 1])
    INDEX_TO_MOVE[63] = np.array([-1, 2])


def create_piece_encodings():
    for i in range(len(PIECES)):
        symbol = PIECES[i]

        encoding = np.zeros(12)
        encoding[i] = 1
        ENCODINGS[symbol] = encoding


def get_board_tensor(board_state):

    # The first two dimensions of the tensor represent the rank and file location
    # of a square. The third dimension is a one-hot encoding of the piece occupying the
    # square (all elements set to 0 for an empty square), with 6 for each piece type for
    # white, and 6 for black.

    # Another implementation would remove the multiplication by 2 and treat opposing pieces
    # of the same type as negative values. I'm not sure if that would work, but it might be
    # worth a try.
    tensor = np.zeros((ROWS, COLS, PIECE_TYPES * 2))

    # The way chess denotes board locations is weird for most programming languages. We
    # have to start at A8 first, since that is the top left item, and thus the first element
    # in a 2D array. However, the "first" square in chess is A1, at the bottom left. Thus,
    # we start at the "last" row and move down, and start at the "first" column and move right.
    for row in range(ROWS):
        for col in range(COLS):

            # This is a simple conversion from computer-readable "rows" and "columns" to ones
            # that can be understood with python-chess. During the very first iteration,
            # chess_row should denote the top row of the board, and chess_col the leftmost column.
            file_index = col
            rank_index = ROWS - row - 1

            try:
                # We now allocate one-hot encodings based on those defined in ENCODINGS, corresponding
                # to their respective pieces.
                symbol = board_state.piece_at(chess.square(file_index, rank_index)).symbol()
                tensor[row, col] = ENCODINGS[symbol]

            # python-chess throws an AttributeError if we call piece_at on an empty square.
            # However, if it IS an empty square, we can simply set the corresponding
            # one-hot encoding as well.
            except AttributeError:
                tensor[row, col] = np.zeros(12)

    return tensor


# We now proceed to return a list of chess move objects:
def get_moves_from_matrix(actions, board_state):
    move_dict = dict()

    for row in range(ROWS):
        for col in range(COLS):

            # Same method as before. As we iterate through a computer-readable grid, each grid
            # coordinate in row-column format has to be converted into rank-file format.
            file_index = col
            rank_index = ROWS - row - 1
            from_square = chess.square(file_index, rank_index)

            # Pieces are always white for the bot, so the symbol should always be a capital.
            # INDEX_TO_MOVE and the action matrix are the same length. Thus, as we iterate
            # through INDEX_TO_MOVE, we can set corresponding ratings to 0 if they are illegal.
            # We know they are illegal if a) the move would put a piece outside the board,
            # or b) the move is not within the chess.Board.legal_moves list.
            for i in range(len(INDEX_TO_MOVE)):
                coords = INDEX_TO_MOVE[i]

                # These two numbers denote the direction in which the piece can move, according
                # to the list created with create_index_to_move_encodings.
                dx = coords[0]
                dy = coords[1]

                # We now get the new column and the new row by simply adding the differences
                # to the current row and column. This has a bit of a strange effect of moving
                # pieces "up" with negative dy values - this is because increasing the row
                # number would move DOWN the grid, whereas increasing the rank number would
                # move UP.
                new_col = col + dx
                new_row = row + dy

                # As above, it is necessary to convert from a computer-readable array to a square
                # numbering system that corresponds to the chess board (since python-chess only
                # works based on the rank and file of a square). The file index does not need
                # to be modified to be chess-readable, but it's assigned a value for human readability.
                new_file_index = new_col
                new_rank_index = ROWS - new_row - 1

                # We know what row and column we are at, and which directions of movement are
                # available to us. We now test this against the engine's rules.
                if (0 <= new_rank_index <= 7) and (0 <= new_file_index <= 7):

                    to_square = chess.square(new_file_index, new_rank_index)
                    move = chess.Move(from_square, to_square)

                    if move in board_state.legal_moves:
                        # Row, column, index, and rating
                        move_dict[move.uci()] = [row, col, i, actions[row, col][i]]
                    else:
                        # If the move is illegal, we set the rating to 0. We renormalize the values later.
                        actions[row, col][i] = 0

                # As discussed above, if the move takes the piece off the board, the rating is set to 0.
                else:
                    actions[row, col][i] = 0

    # Renormalization:
    total = np.sum(actions)
    for key in move_dict.keys():
        data = move_dict[key]
        data[3] = data[3]/total

    return move_dict



# Here we define the Agent class.
class Agent:

    def __init__(self,  *, epsilon=0.1, gamma=0.9):
        self.nn = None
        self.memory = {"white": [],
                       "black": []}
        self.epsilon = epsilon
        self.gamma = gamma

    def remember(self, state_matrix, action_matrix, white):
        if white:
            self.memory["white"].append((state_matrix, action_matrix))
        else:
            self.memory["black"].append((state_matrix, action_matrix))

    def play_move(self, board_state):
        board_tensor = get_board_tensor(board_state)
        action_tensor = self.nn.predict(np.array([board_tensor]))[1][0]
        move_dict = get_moves_from_matrix(action_tensor, board_state)

        best_result = ""

        # TODO: fix, can't be bothered right now
        best_rating = -100

        # Select move with the highest rating:
        for uci in move_dict.keys():
            rating = move_dict[uci][3]
            if rating > best_rating:
                best_rating = rating

                best_result = uci

        # Play the actual move:
        board_state.push(chess.Move.from_uci(best_result))

        # Reformat matrix and remember:
        perfect_output = np.zeros(shape=(8, 8, 64))
        row = move_dict[best_result][0]
        col = move_dict[best_result][1]
        index = move_dict[best_result][2]
        perfect_output[row, col, index] = 1

        self.remember(board_tensor, action_tensor, board_state.turn)

    def build_nn(self):
        main_input = Input(shape=(ROWS, COLS, PIECE_TYPES*2), name="main_input")
        x = create_conv_block(main_input)

        for i in range(CONV_BLOCKS):
            x = create_conv_block(x)

        value_head = create_value_head(x)
        policy_head = create_policy_head(x)

        model = Model(inputs=[main_input], outputs=[value_head, policy_head])
        model.compile(loss={"value_head": "mean_squared_error",
                            "policy_head": "categorical_crossentropy"},
                      optimizer="sgd",
                      loss_weights={"value_head": 0.5, "policy_head": 0.5})

        self.nn = model

    def get_nn(self):
        return self.nn

    def reset(self):
        pass

    def train(self, winner):
        memsize = len(self.memory[winner])
        batch_size = round(memsize / 4)
        board_inputs = np.zeros(shape=(batch_size, ROWS, COLS, PIECE_TYPES*2))

        policy_outputs = np.zeros(shape=(batch_size, 8, 8, 64))
        value_outputs = np.ones(shape=(batch_size))

        indices = np.random.randint(low=0, high=memsize, size=batch_size)
        for i in range(len(indices)):
            index = indices[i]
            board_inputs[i] = self.memory[winner][index][0]
            policy_outputs[i] = self.memory[winner][index][1]

        self.nn.fit({"main_input": board_inputs},
                    {"policy_head": policy_outputs,
                     "value_head": value_outputs},
                    epochs=1)


create_index_to_move_encodings()
create_piece_encodings()
