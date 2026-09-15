from kalfa.registration import pack


lego = pack(__name__)


lego("/plot/torchmetrics/binary_roc", "curves:binary_roc", partial=True, alias="binary_roc", refs={"target": "field"},
     description="ROC curve of the raw test scores against the binary target; output names the wire and target the "
                 "field when the table holds several")
lego("/plot/torchmetrics/binary_precision_recall_curve", "curves:binary_precision_recall_curve", partial=True,
     alias="binary_precision_recall_curve", refs={"target": "field"},
     description="Precision recall curve of the raw test scores against the binary target; output names the wire and "
                 "target the field when the table holds several")
