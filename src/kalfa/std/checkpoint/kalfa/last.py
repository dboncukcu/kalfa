from kalfa.registration import lego
from kalfa.std.checkpoint.base import Policy


class Last(Policy):
    def tags(self, metrics):
        return ["last"]

    def state(self):
        return None

    def restore(self, state):
        pass


@lego("/checkpoint/kalfa/last", alias="last", description="Write last.pt every turn")
def last():
    return Last()
