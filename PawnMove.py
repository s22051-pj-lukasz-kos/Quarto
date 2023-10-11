

class PawnMove:
    """
    Represents a move in Quarto game, by pointing the position on the board and selected pawn
    """
    def __init__(self, position, index, pawn):
        """
        Initialize the move instance

        Parameters:
        position (tuple): A tuple representing the position on the board as (row, column).
        index (int): An index indicating the position of the selected pawn.
        pawn (Pawn): The selected Quarto pawn for the move.
        """
        self.position = position
        self.index = index
        self.pawn = pawn

    def __str__(self):
        """
        Returns a string representation of the move.

        Returns:
        str: A string representing of the move in the format "(row, column), PawnPieceInfo".
        """
        position_str = f"({self.position[0] + 1}, {self.position[1] + 1})"
        return f"({position_str}, {self.pawn})"
