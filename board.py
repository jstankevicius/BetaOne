# Pretty simple board class that allows two players to interact across a game.
# This does not represent the numerical abstraction that the neural network
# will be seeing. Rather, this is more of a mini chess engine to allow
# a human developer to understand what exactly is going on. It also lets us
# not worry about too much overhead when creating a multiplayer mode.

class board:
    state = []

    # Which color's turn it is.
    color = 1

    # Pretty dumb, but these are reference dictionaries for unpacking simple notation.
    file_ref = {"a": 0,
                "b": 1,
                "c": 2,
                "d": 3,
                "e": 4,
                "f": 5,
                "g": 6,
                "h": 7}

    rank_ref = {"1": 7,
                "2": 6,
                "3": 5,
                "4": 4,
                "5": 3,
                "6": 2,
                "7": 1,
                "8": 0}

    def __init__(self):

        # Initialize all piece positions. This will be a little counter-intuitive
        # to a chess player, because a board is viewed in a file-rank fashion
        # (column-row), whereas arrays are row-column.
        
        # |p| = piece type
        self.state = [[-4, -2, -3, -5, -6, -3, -2, -4],
                 [-1, -1, -1, -1, -1, -1, -1, -1],
                 [ 0,  0,  0,  0,  0,  0,  0,  0],
                 [ 0,  0,  0,  0,  0,  0,  0,  0],
                 [ 0,  0,  0,  0,  0,  0,  0,  0],
                 [ 0,  0,  0,  0,  0,  0,  0,  0],
                 [ 1,  1,  1,  1,  1,  1,  1,  1],
                 [ 4,  2,  3,  5,  6,  3,  2,  4]]

    def move(self, s):

        # Empty notation string. We add more information onto it as we process the move.
        # For reference = (piece_type + file? + takes? + to_square + check/checkmate?)
        notation = ""

        # For reference, [rank][file]
        from_square = s[:2]
        to_square = s[2:]

        from_row = self.rank_ref[from_square[1]]
        from_col = self.file_ref[from_square[0]]

        to_row = self.rank_ref[to_square[1]]
        to_col = self.file_ref[to_square[0]]

        piece = self.state[from_row][from_col]
        self.state[from_row][from_col] = 0
        self.state[to_row][to_col] = piece
        

    def get_state(self):
        return self.state

    
    def __str__(self):
        s = ""
        for i in range(8):
            for j in range(8):
                s += " " + str(self.state[i][j])
            s += "\n"

        return s






















    
