from kalfa.registration import pack


lego = pack(__name__)


lego("/criterion/kalfa/cross_entropy", "classification:cross_entropy", partial=True, alias="cross_entropy",
     description="Cross entropy over class logits; weight may be a run time component")
lego("/criterion/kalfa/bce_logits", "classification:bce_logits", partial=True, alias="bce_logits",
     description="Binary cross entropy on logits")

lego("/criterion/kalfa/mse", "regression:mse", partial=True, alias="mse", description="Mean squared error")
lego("/criterion/kalfa/weighted_mse", "regression:weighted_mse", partial=True, alias="weighted_mse",
     refs={"weights": "data"},
     description="Mean squared error with a weight per target column; weights is a list in column order or {uri: "
                 "target_weights, params: {weights: {column: 3.0, 'glob*': 1.5, default: 1.0}}}, named against the "
                 "target columns the dataset carries")
lego("/criterion/kalfa/mae", "regression:mae", partial=True, alias="mae", description="Mean absolute error")
lego("/criterion/kalfa/huber", "regression:huber", partial=True, alias="huber",
     description="Huber loss with threshold delta")
lego("/criterion/kalfa/log_cosh", "regression:log_cosh", partial=True, alias="log_cosh",
     description="log(cosh(error)) loss")
