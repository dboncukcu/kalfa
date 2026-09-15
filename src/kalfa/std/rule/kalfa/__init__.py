from kalfa.registration import pack


lego = pack(__name__)


lego("/rule/kalfa/open", "chain:open_rules", description="Open the rule chain of a turn")
lego("/rule/kalfa/rule", "chain:rule", returns="rules", bus=["metrics", "turn_index"],
     description="Evaluate one rule: skipped until its after rule fired in an earlier turn; a sticky rule keeps its "
                 "effects once fired and is not asked again; with sticky false it is asked every turn, its relative "
                 "effects (times, plus) apply once per firing and its trigger starts over; later rules win the same "
                 "key")
lego("/rule/kalfa/effects", "chain:effects", returns=["effects", "losses", "models", "optimizers"],
     mutates=["models", "optimizers"],
     description="The effects the fired rules left for this turn, applied once before it: a model's trainable flag "
                 "and an optimizer's params in place, the models and the optimizers returned as they are, and the "
                 "loss table as a copy with every loss param the rules set, which the turn and the evaluation of "
                 "every set read alike")
lego("/rule/kalfa/stop", "chain:stop", returns=["rules", "stop"], bus=["metrics", "record"],
     description="Close the chain: the stop triggers are or'ed, their states kept under rules.stop; a stop.json in "
                 "the record (kalfa stop, the board, ctrl-c) ends the loop after this turn as well")
