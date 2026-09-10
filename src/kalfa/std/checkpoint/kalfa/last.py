from kalfa.registration import lego
from kalfa.std.checkpoint.base import Policy


@lego("/checkpoint/kalfa/last", alias="last", writes=["last"], description="Write last.pt every turn")
class Last(Policy):
    def tags(self, metrics):
        return ["last"]
