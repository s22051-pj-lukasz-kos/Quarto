import random
from Pawn import Pawn


class Pieces:
    """
            Represents the pool of pieces, it is populating the pool with unique pieces and shuffling them

            Attributes:
            pawns (list): A list of Quarto pieces in the pool.
    """
    def __init__(self):
        self.pawns = []
        self.populate_pawns()
        self.shuffle_pawns()

    def populate_pawns(self):
        """
             Populates the pool with unique Quarto pieces. Each piece is defined by its size, color,
             shape, and hollow attributes.
        """
        sizes = ["Small", "Large"]
        colors = ["Red", "Blue"]
        shapes = ["Square", "Circular"]
        hollows = ["Hollow", "Solid"]

        for size in sizes:
            for color in colors:
                for shape in shapes:
                    for hollow in hollows:
                        self.pawns.append(Pawn(size, color, shape, hollow))

    def shuffle_pawns(self):
        """
            Randomly shuffles the pieces in the pile
        """
        random.shuffle(self.pawns)
        self.pawns = self.pawns[:16]

    def __str__(self):
        """
            Returns a string representation of the pieces in the pool.
        """
        return "\n".join(map(str, self.pawns))
