from kalfa.registration import pack


lego = pack(__name__)


lego("/trigger/kalfa/after_turn", "clock:after_turn", partial=True, alias=["after_turn", "after_epoch"],
     describe="turn ≥ {at}", description="Fires once the given number of turns has ended, counted across resumes")
lego("/trigger/kalfa/time_budget", "clock:time_budget", partial=True, alias="time_budget",
     describe="after {minutes} minutes",
     description="Fires once the given number of minutes has passed since the first turn it saw")

lego("/trigger/kalfa/metric_above", "metrics:metric_above", partial=True, alias="metric_above",
     describe="{monitor} > {value}",
     description="Fires when the monitored value rises above value; a missing value is not seen")
lego("/trigger/kalfa/metric_below", "metrics:metric_below", partial=True, alias="metric_below",
     describe="{monitor} < {value}",
     description="Fires when the monitored value drops below value; a missing value is not seen")
lego("/trigger/kalfa/plateau", "metrics:plateau", partial=True, alias="plateau",
     describe="{monitor} plateau {patience}",
     description="Fires after patience turns without improvement of the monitored value; turns without the value are "
                 "not counted")
