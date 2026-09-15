from kalfa.std.common.figure import Figure


class Figures(Figure):
    def __init__(self, format="png", width=None, height=None, dpi=150, style="kalfa"):
        super().__init__(format, width, height, dpi, style)
