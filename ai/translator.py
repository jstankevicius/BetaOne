import numpy as np
import chess
from chess import Board

# Global variables are denoted with capital letters for readability. These are constant (after
# being created, of course) and are not modified by any function other than the function that
# populates each list.

# ENCODINGS contains a symbol/one-hot pair, with the one-hot encoding
# denoting the type of piece occupying the square.
PIECES = ("R", "N", "B", "Q", "K", "P", "r", "n", "b", "q", "k", "p")


# TRANSLATION_TABLE contains a chess file (column letter) and the corresponding matrix column
# number.
TRANSLATION_TABLE = {
    "a": 0,
    "b": 1,
    "c": 2,
    "d": 3,
    "e": 4,
    "f": 5,
    "g": 6,
    "h": 7
}
ENCODINGS = dict()

# This array is populated with x and y coordinate changes corresponding
# to each index of an action matrix.
INDEX_TO_MOVE = np.zeros(shape=(64, 2), dtype=np.int64)


def create_piece_encodings():
    for i in range(len(PIECES)):
        symbol = PIECES[i]

        encoding = np.zeros(6)
        index = i
        sign = 1

        if i > 5:
            sign = -1
            index -= 6
            symbol = symbol.lower()

        encoding[index] = sign
        ENCODINGS[symbol] = encoding


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


def board_tensor(board_state):

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
                symbol = board_state.piece_at(chess.square(file_index, rank_index)).symbol()
                tensor[row, col] = ENCODINGS[symbol]

            # python-chess throws an AttributeError if we call piece_at on an empty square.
            # However, if it IS an empty square, we can simply set the corresponding
            # one-hot encoding as well.
            except AttributeError:
                tensor[row, col] = np.zeros(6)

    return tensor


# We now proceed to return a list of chess move objects:
def matrix_to_move(actions, board_state):
    move_dict = dict()

    for row in range(8):
        for col in range(8):

            # Same method as before. As we iterate through a computer-readable grid, each grid
            # coordinate in row-column format has to be converted into rank-file format.
            file_index = col
            rank_index = 7 - row
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
                new_rank_index = 7 - new_row

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


# TODO: remove dependency on multiplier, switch to bool
def move_to_matrix(move, *, multiplier=1):
    move_uci = move.uci()

    origin_file = move_uci[0]
    origin_rank = int(move_uci[1])

    destination_file = move_uci[2]
    destination_rank = int(move_uci[3])

    # We now convert the files into computer-readable columns, and ranks into rows.
    origin_col = TRANSLATION_TABLE[origin_file]
    destination_col = TRANSLATION_TABLE[destination_file]

    origin_row = 8 - origin_rank
    destination_row = 8 - destination_rank

    # To figure out what index belongs to the move, we need to know the location of the piece
    # changed. To do this, we simply calculate the difference in row and the difference in
    # column.

    # NOTE: This is change in ROW and change in COLUMN, not change in RANK and change in FILE.
    dx = destination_col - origin_col
    dy = destination_row - origin_row

    dy *= multiplier
    dx *= multiplier

    for i in range(len(INDEX_TO_MOVE)):
        if dx == INDEX_TO_MOVE[i][0] and dy == INDEX_TO_MOVE[i][1]:
            matrix = np.zeros(shape=(8, 8, 64))
            matrix[origin_row, origin_col, i] = 1
            return matrix


def mirror_board(tensor):
    # Because python-chess has a broken mirror method (or I simply don't understand how it works),
    # I decided to make my own.
    mirrored_tensor = np.flip(np.copy(tensor), 0)
    mirrored_tensor = np.flip(mirrored_tensor, 1)
    mirrored_tensor *= -1

    return mirrored_tensor


create_index_to_move_encodings()
create_piece_encodings()