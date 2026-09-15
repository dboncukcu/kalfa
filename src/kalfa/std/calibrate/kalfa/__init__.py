from kalfa.registration import pack


lego = pack(__name__)


lego("/calibrate/kalfa/threshold", "threshold:Threshold", alias="threshold",
     description="A decision threshold read off a held out set at the end of training: the quantile of the raw output "
                 "of the predicts model on that set; at predict time flag_<output> marks the rows above it")
