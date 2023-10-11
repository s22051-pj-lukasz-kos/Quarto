class Pawn:
    """
    Represents a Quarto game piece with attributes like size, color, shape, and hollow.

    Attributes:
    size (str): The size of the pawn, either "Small" or "Large."
    color (str): The color of the pawn, either "Red" or "Blue."
    shape (str): The shape of the pawn, either "Square" or "Circular."
    hollow (str): The type of the pawn, either "Hollow" or "Solid."
    """
    def __init__(self, size, color, shape, hollow):
        """
        Initialize a Quarto game piece (pawn) with the specified attributes.

        Parameters:
        size (str): The size of the pawn, either "Small" or "Large."
        color (str): The color of the pawn, either "Red" or "Blue."
        shape (str): The shape of the pawn, either "Square" or "Circular."
        hollow (str): The type of the pawn, either "Hollow" or "Solid."
        """
        self.size = size
        self.color = color
        self.shape = shape
        self.hollow = hollow

    def __str__(self):
        """
        Returns a string representation of the pawn using colored ASCII characters.

        Returns:
        str: A string representing the pawn, color-coding for Red and Blue, and a character coding for other
        """
        output = ""
        blue = "\u001b[34m"
        red = "\u001b[31m"
        reset_color = "\x1b[0m"
        if self.color == "Red":
            color_str = red
        else:
            color_str = blue
        if self.shape == "Square":
            output += "x"
        else:
            output += "o"
        if self.size == "Small":
            output = output.capitalize()
        if self.hollow == "Hollow":
            output += "h"
        else:
            output += "s"
        return f"{color_str}{output}{reset_color}"
