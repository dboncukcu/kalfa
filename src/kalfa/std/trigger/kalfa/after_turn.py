from kalfa.registration import lego


@lego("/trigger/kalfa/after_turn", partial=True, alias=["after_turn", "after_epoch"], describe="turn ≥ {at}",
      description="Fires once the given number of turns has ended, counted across resumes")
def after_turn(metrics, turn_index, state, at):
    seen = int((state or {}).get("seen", 0)) + 1
    return seen >= int(at), {"seen": seen}
