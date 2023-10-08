""" For prettier representation of pawn's move """

class PawnMove:
    def __init__(self, position, index, pawn):
        self.position = position
        self.index = index
        self.pawn = pawn

    def __str__(self):
        return f"({self.position}, {self.pawn})"