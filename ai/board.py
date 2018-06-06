class Board:

    def __init__(self):
        """Initializes the default board state."""

        # 2D numerical representation of the current board state.
        self.state = [
            [-4, -2, -3, -5, -6, -3, -2, -4],
            [-1, -1, -1, -1, -1, -1, -1, -1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [4, 2, 3, 5, 6, 3, 2, 4]
        ]

        # Multiply by -1 after each turn to switch players.
        self.turn = 1

        # White castling rights:
        self.wqcastle = 1
        self.wkcastle = 1

        # Black castling rights:
        self.bqcastle = 1
        self.bkcastle = 1

        # Move repetition count. Threefold repetition concludes in a draw.
        self.repeats = 0

    def reorient(self):
        """Flips the color of every piece and the board position."""
        pass

    def gen_legal(self):
        """Generates a set of possible valid board states based on player."""

        # This might get pretty expensive. Ew.
        legal_moves = []

        # We first account for what player can actually make a move here. Then,
        # depending on the sign, we may iterate through the board and generate
        # possible moves for each piece.
        for row in range(len(self.state)):

            for col in range(len(self.state[row])):

                tile = self.state[row][col]

                # Hacky.
                piece_type = tile * self.turn

                # Check for piece availability:
                if piece_type > 0:

                    # Now iterate through possible piece types.

                    # Generate all possible pawn moves:
                    if tile == 1:

                        # Check if pawn can move one tile forward:
                        if self.state[row - self.turn][col] == 0:
                            state_copy = self.state
                            state_copy[row - self.turn][col] = 1 * self.turn
                            legal_moves.append(state_copy)

                        # Check if pawn can move two tiles forward:
                        if (self.state[row - 2*self.turn][col] == 0) and (self.state[row - self.turn][col] == 0):
                            state_copy = self.state
                            state_copy[row - 2*self.turn][col] = 1 * self.turn
                            legal_moves.append(state_copy)

                        # TODO: En passant

                    # Generate all possible knight moves:

                    # Generate all possible bishop moves:

                    # Generate all possible rook moves:

                    # Generate all possible queen moves:

                    # Generate all possible king moves:

