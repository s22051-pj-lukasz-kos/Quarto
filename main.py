from easyAI import TwoPlayerGame
from easyAI.Player import Human_Player

from Pieces import Pieces
from PawnMove import PawnMove


class Quarto(TwoPlayerGame):
    """ explain game rules here. Probably """

    def __init__(self, players):
        self.players = players
        self.pawnsPile = Pieces().pawns
        self.board = [[None for _ in range(4)] for _ in range(4)]
        self.current_player = 1

    def possible_moves(self):
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
        row_pos = move.position[0]
        col_pos = move.position[1]
        index = move.index
        pawn = move.pawn
        self.board[row_pos][col_pos] = pawn
        self.pawnsPile.pop(index)

    def unmake_move(self, move):
        row_pos = move.position[0]
        col_pos = move.position[1]
        index = move.index
        pawn = move.pawn
        self.board[row_pos][col_pos] = None
        self.pawnsPile.insert(index, pawn)

    def lose(self):
        """ When player lose """
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
        common_value = self.board[0][col].__dict__[attribute]
        return all(
            self.board[0 + i][col].__dict__[attribute] == common_value
            for i in range(4)
        )

    def has_common_attribute_in_horizontal_line(self, row, attribute):
        common_value = self.board[row][0].__dict__[attribute]
        return all(
            self.board[row][0 + i].__dict__[attribute] == common_value
            for i in range(4)
        )

    def has_common_attribute_in_diagonal_line_1(self, attribute):
        common_value = self.board[0][0].__dict__[attribute]
        return all(
            self.board[0 + i][0 + i].__dict__[attribute] == common_value
            for i in range(4)
        )

    def has_common_attribute_in_diagonal_line_2(self, attribute):
        common_value = self.board[3][0].__dict__[attribute]
        return all(
            self.board[3 - i][0 + i].__dict__[attribute] == common_value
            for i in range(4)
        )

    def is_over(self):
        return self.lose() or self.is_draw()

    def is_draw(self):
        return self.possible_moves() == [] and not self.lose()

    def scoring(self):
        return -100 if self.lose() else 0

    def show(self):
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
    from easyAI import AI_Player, Negamax, solve_with_depth_first_search

    scoring = lambda game: -100 if game.lose() else 0
    ai_algo = Negamax(4, scoring)
    game = Quarto([Human_Player(), AI_Player(ai_algo)])
    history = game.play()
