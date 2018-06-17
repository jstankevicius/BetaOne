import chess
import numpy as np
from agent import Agent
# GLOBALS:

# ENCODINGS contains a symbol/one-hot pair, with the one-hot encoding
# denoting the type of piece occupying the square.
PIECES = ["r", "n", "b", "q", "k", "p", "R", "N", "B", "Q", "K", "P"]
ENCODINGS = dict()

# This array is populated with x and y coordinate changes corresponding
# to each index of an action matrix.
INDEX_TO_MOVE = np.zeros(shape=(64, 2))

# Settings (don't know why you would ever need to change these, but this is to keep stuff consistent):
ROWS = 8
COLS = 8
PIECE_TYPES = 6


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
        INDEX_TO_MOVE[linear_index] = np.array([dx, 0])
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
    INDEX_TO_MOVE.astype(np.int64)

def create_piece_encodings():
    for i in range(len(PIECES)):
        symbol = PIECES[i]

        encoding = np.zeros(12)
        encoding[i] = 1
        ENCODINGS[symbol] = encoding


def get_board_tensor(b):

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
                symbol = b.piece_at(chess.square(file_index, rank_index)).symbol()

                tensor[row, col] = ENCODINGS[symbol]

            # python-chess throws an AttributeError if we call piece_at on an empty square.
            # However, if it IS an empty square, we can simply set the corresponding
            # one-hot encoding as well.
            except AttributeError:
                tensor[row, col] = np.zeros(12)

    return tensor



# We now proceed to return a list of chess move objects:
def get_moves_from_matrix(actions):
    # Just as above, the "first" element in the matrix corresponds to the top left
    # square on the board.
    moves = []

    # While passing, we can immediately discard any moves that would move an opponent's
    # pieces.
    for row in range(ROWS):
        for col in range(COLS):

            # Same method as before.
            file_index = col
            rank_index = ROWS - row - 1

            try:
                from_square = chess.square(file_index, rank_index)
                symbol = b.piece_at(from_square).symbol()

                # Pieces are always white for the bot, so the symbol should always be a capital.
                # INDEX_TO_MOVE and the action matrix are the same length. Thus, as we iterate
                # through INDEX_TO_MOVE, we can set corresponding ratings to 0 if they are illegal.
                # We know they are illegal if a) the engine raises a ValueError because the move is
                # out of bounds, or b) the move is not within chess.Board.legal_moves list.
                for i in range(64):
                    coords = INDEX_TO_MOVE[i]
                    dx = coords[0]
                    dy = coords[1]

                    new_col = col + dx
                    new_row = row + dy
                    new_file_index = new_col
                    new_rank_index = ROWS - new_row - 1

                    # We know what row and column we are at, and which directions of movement are
                    # available to us. We now test this against the engine's rules.
                    if (0 <= new_rank_index <= 7) and (0 <= new_file_index <= 7):
                        to_square = chess.square(new_file_index.astype(np.int64), new_rank_index.astype(np.int64))
                        move = chess.Move(from_square, to_square)
                        if move not in b.legal_moves:
                            # If the move is illegal, we set the rating to 0. We renormalize the values later.
                            actions[0, row, col][i] = 0
                        else:
                            print(move)

            except AttributeError:
                pass

    return moves


create_index_to_move_encodings()
create_piece_encodings()
b = chess.Board()
tensor = get_board_tensor(b)


# Testing:
a = Agent()
a.build_nn()


# Sub this out for play_move later on
action_matrix = a.get_nn().predict(np.array([tensor]))[1]
lmao = get_moves_from_matrix(action_matrix)
