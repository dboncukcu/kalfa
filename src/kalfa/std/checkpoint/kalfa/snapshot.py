from kalfa.registration import lego
from kalfa.std.checkpoint.base import Policy


@lego("/checkpoint/kalfa/snapshot", alias="snapshot",
      description="Write snapshot_<n>.pt every n turns and last.pt every turn")
class Snapshot(Policy):
    def __init__(self, every):
        self.every = int(every)
        self.seen = 0

    def tags(self, metrics):
        self.seen += 1
        tags = ["last"]
        if self.seen % self.every == 0:
            tags.append(f"snapshot_{self.seen}")
        return tags

    def state(self):
        return {"seen": self.seen}

    def restore(self, state):
        if state and "seen" in state:
            self.seen = int(state["seen"])
