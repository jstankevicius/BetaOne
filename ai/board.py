import chess
import numpy as np
# GLOBALS:

# ENCODINGS contains a symbol/one-hot pair, with the one-hot encoding
# denoting the type of piece occupying the square.
PIECES = ["r", "n", "b", "q", "k", "p", "R", "N", "B", "Q", "K", "P"]
ENCODINGS = dict()

# Settings (don't know why you would ever need to change these, but this is to keep stuff consistent):
ROWS = 8
COLS = 8
PIECE_TYPES = 6


def create_encodings():
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

# Testing:
create_encodings()
b = chess.Board()
tensor = get_board_tensor(b)

# dumb
#print(model.predict(np.array([tensor])))