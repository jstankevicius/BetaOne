import numpy as np
import chess

# Utility module that deals with encoding chess boards and pieces as tensors, as well as
# manipulating said tensors.

# Global variables are denoted with capital letters for readability. These are constant (after
# being created, of course) and are not modified by any function other than the function that
# populates each list.
PIECES = ("R", "N", "B", "Q", "K", "P", "r", "n", "b", "q", "k", "p")

ZOBRIST_TABLE = np.empty(shape=(64, 12), dtype=np.uint32)

PIECE_ENCODINGS = {
    "R": np.array([1, 0, 0, 0, 0, 0]),
    "N": np.array([0, 1, 0, 0, 0, 0]),
    "B": np.array([0, 0, 1, 0, 0, 0]),
    "Q": np.array([0, 0, 0, 1, 0, 0]),
    "K": np.array([0, 0, 0, 0, 1, 0]),
    "P": np.array([0, 0, 0, 0, 0, 1]),
    "r": np.array([-1, 0, 0, 0, 0, 0]),
    "n": np.array([0, -1, 0, 0, 0, 0]),
    "b": np.array([0, 0, -1, 0, 0, 0]),
    "q": np.array([0, 0, 0, -1, 0, 0]),
    "k": np.array([0, 0, 0, 0, -1, 0]),
    "p": np.array([0, 0, 0, 0, 0, -1]),

}


def get_board_zobrist(board):
    """Returns a Zobrist hash of the board."""
    h = np.uint64(0)

    for i in range(64):
        square = chess.SQUARES[i]

        # Once again, we check if there is a piece on the square by looking
        # for an AttributeError.
        try:
            piece = board.piece_at(square).symbol()
            piece_index = PIECES.index(piece)
            h = np.bitwise_xor(h, ZOBRIST_TABLE[i, piece_index])

        # Ugly, but there are no alternatives that are written prettier.
        except AttributeError:
            pass

    return h


def board_tensor(board):
    """Returns a board state represented as a tensor of shape (8, 8, 6)."""

    # The first two dimensions of the tensor represent the rank and file location
    # of a square. The third dimension is a one-hot encoding of the piece occupying the
    # square (all elements set to 0 for an empty square), with 6 for each piece type for
    # white, and 6 for black.
    tensor = np.zeros((8, 8, 6))

    # The way chess denotes board locations is weird for most programming languages. We
    # have to start at A8 first, since that is the top left item, and thus the first element
    # in a 2D array. However, the "first" square in chess is A1, at the bottom left. Thus,
    # we start at the "last" row and move down, and start at the "first" column and move right.
    for row in range(8):
        for col in range(8):

            # This is a simple conversion from computer-readable "rows" and "columns" to ones
            # that can be understood with python-chess. During the very first iteration,
            # chess_row should denote the top row of the board, and chess_col the leftmost column.
            file_index = col
            rank_index = 7 - row

            try:
                # We now allocate one-hot encodings based on those defined in ENCODINGS, corresponding
                # to their respective pieces.
                symbol = board.piece_at(chess.square(file_index, rank_index)).symbol()
                tensor[row, col] = PIECE_ENCODINGS[symbol]

            # python-chess throws an AttributeError if we call piece_at on an empty square.
            # However, if it IS an empty square, we can simply set the corresponding
            # one-hot encoding as well.
            except AttributeError:
                tensor[row, col] = np.zeros(6)

    return tensor


def mirror_tensor(tensor):
    """Mirrors a board tensor vertically and horizontally and flips the color of each piece."""
    mirrored_tensor = np.flip(np.copy(tensor), 0)
    mirrored_tensor = np.flip(mirrored_tensor, 1)
    mirrored_tensor *= -1

    return mirrored_tensor
