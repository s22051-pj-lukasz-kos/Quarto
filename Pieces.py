import random
from Pawn import Pawn

class Pieces:
    def __init__(self):
        self.pawns = []
        self.populate_pawns()
        self.shuffle_pawns()

    def populate_pawns(self):
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
        random.shuffle(self.pawns)
        self.pawns = self.pawns[:16]

    def __str__(self):
        return "\n".join(map(str, self.pawns))
