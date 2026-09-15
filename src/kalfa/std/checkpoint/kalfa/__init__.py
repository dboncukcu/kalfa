from kalfa.registration import pack

lego = pack(__name__)


lego("/checkpoint/kalfa/best", "policies:Best", alias="best", writes=["best", "last"],
     description="Write best.pt when the monitored value improves and last.pt every turn; last: false writes best.pt "
                 "alone, for a run that is never resumed")
lego("/checkpoint/kalfa/last", "policies:Last", alias="last", writes=["last"], description="Write last.pt every turn")
lego("/checkpoint/kalfa/snapshot", "policies:Snapshot", alias="snapshot", writes=["last", "snapshot"],
     description="Write snapshot_<n>.pt every n turns and last.pt every turn")
