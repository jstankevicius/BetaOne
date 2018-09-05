import chess
import position as pos
from chess import Board


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

    def expand(self):
        legal_moves = pos.get_unordered_legal_moves(self.state)

        for move, result in legal_moves:

            # If we are ONLY expanding if children == 0, is this if
            # even necessary? I'll leave it in just in case I change
            # the criteria for expansion.
            if not self.contains(move.uci()):

                child_node = Node(move.uci())
                child_node.set_state(result)

                # Add reference from child to parent
                child_node.set_parent(self)

                # Add child to parent's list of children
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

    # Denotes the first file's number in the directory.
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