""" For prettier representation of pawn's move """


class PawnMove:
    def __init__(self, position, index, pawn):
        self.position = position
        self.index = index
        self.pawn = pawn

    def __str__(self):
        position_str = f"({self.position[0] + 1}, {self.position[1] + 1})"
        return f"({position_str}, {self.pawn})"
