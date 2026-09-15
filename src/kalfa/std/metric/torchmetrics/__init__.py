from kalfa.registration import pack


lego = pack(__name__)


lego("/metric/torchmetrics/accuracy", "classification:accuracy", state=True, alias="accuracy",
     description="Accuracy of class logits (argmax) against integer labels")
lego("/metric/torchmetrics/f1", "classification:f1", state=True, alias="f1",
     description="Macro F1 of class logits (argmax) against integer labels")
lego("/metric/torchmetrics/binary_auroc", "classification:binary_auroc", state=True, alias="auroc",
     description="Area under the ROC curve of binary scores (torchmetrics)")
lego("/metric/torchmetrics/binary_average_precision", "classification:binary_average_precision", state=True,
     alias="average_precision", description="Average precision of binary scores (torchmetrics)")
