from kalfa.registration import lego
from kalfa.std.pre.base import Preprocessor


class TwoViews(Preprocessor):
    def __init__(self, transform):
        self.transform = transform

    def apply(self, value):
        return (self.transform.apply(value), self.transform.apply(value))


@lego("/pre/kalfa/two_views", alias="two_views", refs={"transform": "preprocessor"},
      description="Two independent applications of a transform to one image, as a pair")
def two_views(transform):
    return TwoViews(transform)
