from kalfa.registration import pack

lego = pack(__name__)


lego("/adapter/kalfa/criterion", "criterion:CriterionAdapter", uses=["predicts"],
     description="Feed a criterion the predicts model's output wire and the target field named by the definition's "
                 "keys")

lego("/adapter/kalfa/metric", "metric:MetricAdapter", uses=["predicts"],
     description="Feed a metric the predicts model's output wire and the target field named by the definition's keys")

lego("/adapter/kalfa/objective", "objective:ObjectiveAdapter",
     description="Call an objective with every model of the run and the batch, plus the step, epoch, rng, scaler and "
                 "losses view its signature names")
