from kalfa.registration import pack


lego = pack(__name__)


lego("/strategy/kalfa/grid", "deterministic:Grid", alias="grid", enumerates=True,
     description="Every combination of the space's choices (a range needs steps); deterministic by id")
lego("/strategy/kalfa/random", "deterministic:RandomSearch", alias="random",
     description="count points drawn uniformly from the space with a seed; deterministic by id")
lego("/strategy/kalfa/sobol", "deterministic:SobolSearch", alias="sobol",
     description="count points of a scrambled Sobol sequence with a seed; deterministic by id")

lego("/strategy/kalfa/optuna", "optuna:OptunaSearch", alias="optuna",
     description="trials points proposed by optuna (tpe or random sampler) from the objectives fed back; local loop "
                 "only, no --id")
