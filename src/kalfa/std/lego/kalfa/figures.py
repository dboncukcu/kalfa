from kalfa.registration import lego
from kalfa.std.common.figure import Figure


@lego("/lego/kalfa/figures",
      description="The look of every plot of a run: the file format, the size of one panel in inches, the dpi and "
                  "the style (kalfa, or none for matplotlib's own); the figures section is its params and the built "
                  "object reaches every plot that names figures")
class Figures(Figure):
    def __init__(self, format="png", width=None, height=None, dpi=150, style="kalfa"):
        super().__init__(format, width, height, dpi, style)
