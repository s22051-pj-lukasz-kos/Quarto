from easyAI import TwoPlayerGame
from easyAI.Player import Human_Player

from Pieces import Pieces
from PawnMove import PawnMove


class Quarto(TwoPlayerGame):
    """
    Quarto is a two-player board game where the goal is to form a line of four pieces
    with a common attribute. The game is played on a 4x4 board, and players take turns
    placing pieces on the board. Each pawn has four distinct attributes (red/blue,
    large/small, square/round, hollow/solid) to win you need to form a line of four pawns with one common attribute.
    To make a move is to place one pawn, chosen from 16 pieces common for both players

    Rules and Gameplay:
    - Game is played in turns
    - Each turn player can place only one pawn
    - Winning lines could be horizontal, vertical and diagonal
    - To show possible moves type: 'show moves'
    - To make move type: 'move #{move_number}' e.g. move #11
    - WARNING! Game significantly differs from the original rules, in our variant you choose and place a piece
      in one turn, in original rules you choose the piece to be placed by your opponent

    Setup:
    - To play Quarto, you need Python installed on your system
    - You need easyAI(https://pypi.org/project/easyAI/) library installed, if you don't, install it using pip:

         pip install easyAI

    Authors: Łukasz Kos, Emilian Murawski

    ENJOY:)
    """

    def __init__(self, players):
        """
        Initialize a new Quarto game.

        Parameters:
        players (list): A list of two players participating in the game.

        Attributes:
        pawnsPile (list): The pool of 16 unique game pieces.
        board (list): The 4x4 game board to place pieces.
        current_player (int): An integer (1 or 2) representing the current player.
        """
        self.players = players
        self.pawnsPile = Pieces().pawns
        self.board = [[None for _ in range(4)] for _ in range(4)]
        self.current_player = 1

    def possible_moves(self):
        """
        Calculate and return a list of available moves.

        Returns:
        list: A list of possible moves, where each move represents a piece and its position on the board.
        """
        moves = []
        for row in range(4):
            for col in range(4):
                if self.board[row][col] is None:
                    empty_position = (row, col)
                    for index, pawn in enumerate(self.pawnsPile):
                        # moves.append((empty_position, (index, pawn)))
                        move = PawnMove(empty_position, index, pawn)
                        moves.append(move)
        return moves

    def make_move(self, move):
        """
        Make a move by placing a selected piece on the board at the specified position.

        Parameters:
        move (PawnMove): An instance representing the move to be played.
        """
        row_pos = move.position[0]
        col_pos = move.position[1]
        index = move.index
        pawn = move.pawn
        self.board[row_pos][col_pos] = pawn
        self.pawnsPile.pop(index)

    def unmake_move(self, move):
        """
        Revert a move by removing a piece from the board.

        Parameters:
        move (PawnMove): An instance representing the move to be undone.
        """
        row_pos = move.position[0]
        col_pos = move.position[1]
        index = move.index
        pawn = move.pawn
        self.board[row_pos][col_pos] = None
        self.pawnsPile.insert(index, pawn)

    def lose(self):
        """
        Check if the current player has lost by forming a winning line.

        Returns:
        bool: True if the current player has lost, False otherwise.
        """
        # check for vertical lines
        for col in range(4):
            if (
                    self.board[0][col] is not None
                    and self.board[1][col] is not None
                    and self.board[2][col] is not None
                    and self.board[3][col] is not None
            ):
                if self.has_common_attribute_in_vertical_line(col, "size"):
                    return True
                if self.has_common_attribute_in_vertical_line(col, "color"):
                    return True
                if self.has_common_attribute_in_vertical_line(col, "shape"):
                    return True
                if self.has_common_attribute_in_vertical_line(col, "hollow"):
                    return True

        # check for horizontal lines
        for row in range(4):
            if (
                    self.board[row][0] is not None
                    and self.board[row][1] is not None
                    and self.board[row][2] is not None
                    and self.board[row][3] is not None
            ):
                if self.has_common_attribute_in_horizontal_line(row, "size"):
                    return True
                if self.has_common_attribute_in_horizontal_line(row, "color"):
                    return True
                if self.has_common_attribute_in_horizontal_line(row, "shape"):
                    return True
                if self.has_common_attribute_in_horizontal_line(row, "hollow"):
                    return True

        # check for first diagonal line
        if (
                self.board[0][0] is not None
                and self.board[1][1] is not None
                and self.board[2][2] is not None
                and self.board[3][3] is not None
        ):
            if self.has_common_attribute_in_diagonal_line_1("size"):
                return True
            if self.has_common_attribute_in_diagonal_line_1("color"):
                return True
            if self.has_common_attribute_in_diagonal_line_1("shape"):
                return True
            if self.has_common_attribute_in_diagonal_line_1("hollow"):
                return True

        # check for second diagonal line
        if (
                self.board[3][0] is not None
                and self.board[2][1] is not None
                and self.board[1][2] is not None
                and self.board[0][3] is not None
        ):
            if self.has_common_attribute_in_diagonal_line_2("size"):
                return True
            if self.has_common_attribute_in_diagonal_line_2("color"):
                return True
            if self.has_common_attribute_in_diagonal_line_2("shape"):
                return True
            if self.has_common_attribute_in_diagonal_line_2("hollow"):
                return True

    def has_common_attribute_in_vertical_line(self, col, attribute):
        """
        Check if there is a common attribute in a vertical line on the game board.

        Parameters:
        col (int): The column index (0-based) of the line to check.
        attribute (str): The attribute to check for commonality (e.g., 'size', 'color', 'shape', 'hollow').

        Returns:
        bool: True if there is a common attribute in the specified vertical line, False otherwise.
        """
        common_value = self.board[0][col].__dict__[attribute]
        return all(
            self.board[0 + i][col].__dict__[attribute] == common_value
            for i in range(4)
        )

    def has_common_attribute_in_horizontal_line(self, row, attribute):
        """
        Check if there is a common attribute in a horizontal line on the game board.

        Parameters:
        row (int): The row index (0-based) of the line to check.
        attribute (str): The attribute to check for commonality (e.g., 'size', 'color', 'shape', 'hollow').

        Returns:
        bool: True if there is a common attribute in the specified horizontal line, False otherwise.
        """
        common_value = self.board[row][0].__dict__[attribute]
        return all(
            self.board[row][0 + i].__dict__[attribute] == common_value
            for i in range(4)
        )

    def has_common_attribute_in_diagonal_line_1(self, attribute):
        """
        Check if there is a common attribute in the diagonal line (from top-left to bottom-right) on the board.

        Parameters:
        attribute (str): The attribute to check for commonality (e.g., 'size', 'color', 'shape', 'hollow').

        Returns:
        bool: True if there is a common attribute in the first diagonal line, False otherwise.
        """
        common_value = self.board[0][0].__dict__[attribute]
        return all(
            self.board[0 + i][0 + i].__dict__[attribute] == common_value
            for i in range(4)
        )

    def has_common_attribute_in_diagonal_line_2(self, attribute):
        """
        Check if there is a common attribute in the diagonal line (from top-right to bottom-left) on the board.

        Parameters:
        attribute (str): The attribute to check for commonality (e.g., 'size', 'color', 'shape', 'hollow').

        Returns:
        bool: True if there is a common attribute in the second diagonal line, False otherwise.
        """

        common_value = self.board[3][0].__dict__[attribute]
        return all(
            self.board[3 - i][0 + i].__dict__[attribute] == common_value
            for i in range(4)
        )

    def is_over(self):
        """
        Determine if the game is over (either a player has won or it's a draw).

        Returns:
        bool: True if the game is over, False otherwise.
        """
        return self.lose() or self.is_draw()

    def is_draw(self):
        """
        Check if the game has ended in a draw.

        Returns:
        bool: True if the game is a draw, False otherwise.
        """
        return self.possible_moves() == [] and not self.lose()

    def scoring(self):
        """
        Assign a score to the current game state.

        Returns:
        int: A score of -100 if the current player has lost, 0 otherwise.
        """
        return -100 if self.lose() else 0

    def show(self):
        """
        Display the game state, including the remaining pieces, the board, and the game result.
        """

        print("Pawns in the pool:")
        for pawn in self.pawnsPile:
            print(pawn, end=" ")
        print()
        print("Board")
        for row in self.board:
            for pawn in row:
                if pawn is None:
                    print(" .", end="")
                else:
                    print(pawn, end="")
            print()
        if self.is_draw():
            print("The game is a draw!")
        elif self.lose():
            winning_player = self.current_player
            print(f"Player {winning_player} wins!")
        else:
            print("Type 'show moves' to see all moves")
            print("To move a pawn type e.g. 'move #11'")


if __name__ == "__main__":
    from easyAI import AI_Player, Negamax

    scoring = lambda game: -100 if game.lose() else 0
    """
    The AI player makes its move with a search depth of 4 in the game tree. If you experience delays
    in AI decision-making, consider reducing the depth parameter in the 'Negamax' function below.
    """
    ai_algo = Negamax(4, scoring)
    game = Quarto([Human_Player(), AI_Player(ai_algo)])
    history = game.play()
