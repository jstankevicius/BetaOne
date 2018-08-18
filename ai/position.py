import numpy as np
import chess

# Utility module that deals with encoding chess boards and pieces as tensors, as well as
# manipulating said tensors.

# Global variables are denoted with capital letters for readability. These are constant (after
# being created, of course) and are not modified by any function other than the function that
# populates each list.
PIECES = ("R", "N", "B", "Q", "K", "P", "p", "k", "q", "b", "n", "r")


PIECE_ENCODINGS = {
    "R": np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    "N": np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    "B": np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    "Q": np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
    "K": np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
    "P": np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
    "p": np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
    "k": np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
    "q": np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
    "b": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
    "n": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
    "r": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
}


def tensor_to_board(tensor):
    for i in range(8):
        row = ""
        for j in range(8):
            if np.sum(tensor[i, j]) != 0:
                row += PIECES[int(np.argmax(tensor[i, j]))] + " "
            else:
                row += ". "
        print(row)


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


def board_tensor(board):
    tensor = np.zeros(shape=(8, 8, 12))
    attackers = np.zeros(shape=(8, 8, 12))

    pieces = board.piece_map()

    # Add pieces:
    for square in pieces.keys():
        file = chess.square_file(square)
        rank = chess.square_rank(square)

        piece = pieces[square].symbol()

        row = 7 - rank
        col = file

        tensor[row, col] = PIECE_ENCODINGS[piece]

        attacked_squares = board.attacks(square)

        for attacked_square in attacked_squares:

            attacked_file = chess.square_file(attacked_square)
            attacked_rank = chess.square_rank(attacked_square)

            attacked_row = 7 - attacked_rank
            attacked_col = attacked_file

            attackers[attacked_row, attacked_col, np.argmax(PIECE_ENCODINGS[piece])] += 1

    if not board.turn:
        tensor = mirror_tensor(tensor)
        attackers = mirror_tensor(attackers)

    tensor = np.concatenate((tensor, attackers), 2)

    # The feature vector is a concatenation of board tensors from
    # the previous 8 states. We have several constant features that represent
    # data in this particular position:
    # 1. Current player's kingside castling rights
    # 2. Current player's queenside castling rights
    # 3. Opponent's kingside castling rights
    # 3. Opponent's queenside castling rights
    # 4. No-progress count
    # 5. Number of repeated board state pairs (6 plies repeated is instant draw)
    # 6. Total number of moves

    # We initialize the feature vector as a list of counters.
    # "player" is simply the current side's color.

    player = board.turn

    player_has_kingside_castling_rights = float(board.has_kingside_castling_rights(player))
    player_has_queenside_castling_rights = float(board.has_queenside_castling_rights(player))

    # "not player" is my favorite piece of python I've ever written
    opponent_has_kingside_castling_rights = float(board.has_kingside_castling_rights(not player))
    opponent_has_queenside_castling_rights = float(board.has_queenside_castling_rights(not player))
    no_progress_count = board.halfmove_clock
    total_moves = board.fullmove_number

    counter_layers = (
        player_has_kingside_castling_rights,
        player_has_queenside_castling_rights,
        opponent_has_kingside_castling_rights,
        opponent_has_queenside_castling_rights,
        no_progress_count,
        total_moves
    )

    # We add the counters to the feature vector:
    for i in range(len(counter_layers)):
        tensor = np.concatenate((tensor, np.full((8, 8, 1), counter_layers[i])), 2)

    return tensor


def mirror_tensor(tensor):
    """Mirrors a board tensor vertically and horizontally and flips the color of each piece."""
    mirrored_tensor = np.flip(np.copy(tensor), 0)
    mirrored_tensor = np.flip(mirrored_tensor, 1)
    mirrored_tensor = np.flip(mirrored_tensor, 2)

    return mirrored_tensor
