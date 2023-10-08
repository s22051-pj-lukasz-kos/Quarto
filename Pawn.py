class Pawn:
    def __init__(self, size, color, shape, hollow):
        self.size = size  # Small or Large
        self.color = color  # Red or Blue
        self.shape = shape  # Square or Circular
        self.hollow = hollow  # Hollow or Solid

    def __str__(self):
        output = ""
        blue = "\u001b[34m"
        red = "\u001b[31m"
        reset_color = "\x1b[0m"
        if self.color == 0:
            color_str = red
        else:
            color_str = blue
        if self.shape == 0:
            output += "x"
        else:
            output += "o"
        if self.size == 1:
            output = output.capitalize()
        if self.hollow == 1:
            output += "h"
        else:
            output += "s"
        return f"{color_str}{output}{reset_color}"