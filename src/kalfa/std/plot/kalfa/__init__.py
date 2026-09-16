from kalfa.registration import pack


lego = pack(__name__)


lego("/plot/kalfa/architecture", "architecture:architecture", partial=True, alias="architecture",
     description="kalfa's own drawing of every report model under plots/<name>_<model>.png: one box per graph node "
                 "with the name from the config, what it is (a torch layer, a lego, another model), the shapes one "
                 "batch traced through it and the layers inside it, the wires as arrows with their width, the input "
                 "wire with its feature columns, the boundary wires as boxes, and the losses and the optimizers "
                 "beside the outputs they read; matplotlib only, any device")
lego("/plot/kalfa/architecture_text", "architecture:architecture_text", partial=True, alias="architecture_text",
     description="The report models printed as text under plots/<name>.txt, the module repr of each")

lego("/plot/kalfa/class_histogram", "classification:class_histogram", partial=True, alias="class_histogram",
     refs={"target": "field"},
     description="Histogram of the raw scores of the test set, one series per target class; output names the wire and "
                 "target the field when the table holds several")
lego("/plot/kalfa/confusion_matrix", "classification:confusion_matrix", partial=True, alias="confusion_matrix",
     description="Confusion matrix of the decoded test predictions against the target labels, counts and row shares "
                 "in every cell")

lego("/plot/kalfa/target_vs_features", "data:target_vs_features", partial=True, alias="target_vs_features",
     refs={"target": "field"}, needs=["train_loader"],
     description="One panel per feature: the target against it as a hexbin density with the median profile over equal "
                 "count bins; it reads the set the definition names (train without one) and draws in the original "
                 "units")
lego("/plot/kalfa/target_correlation", "data:target_correlation", partial=True, alias="target_correlation",
     refs={"target": "field"}, needs=["train_loader"],
     description="The rank correlation of every column with the target, the strongest first; groups maps a column to "
                 "a group name and colours the bars by it")
lego("/plot/kalfa/correlation_heatmap", "data:correlation_heatmap", partial=True, needs=["train_loader"],
     alias="correlation_heatmap",
     description="The rank correlation of every column of a set against every other, features and targets together; "
                 "it reads the set the definition names (train without one)")
lego("/plot/kalfa/feature_distributions", "data:feature_distributions", partial=True, needs=["train_loader"],
     alias="feature_distributions",
     description="A histogram per feature column of a set, in the original units; log names the columns to draw on a "
                 "log10 axis")

lego("/plot/kalfa/data_pipeline", "data_pipeline:data_pipeline", partial=True, alias="data_pipeline",
     needs=["data_report"],
     description="The data block as one picture: every stage with its rows and columns, the split, the fitted frame "
                 "transforms and preprocessors, the features and targets, the loaders; under the fit, before and "
                 "after histograms of the columns with the longest chains (columns names others) when the source is a "
                 "table")

lego("/plot/kalfa/image_grid", "images:image_grid", partial=True, alias="image_grid",
     description="n outputs of the predicts model on the report set as an image grid")
lego("/plot/kalfa/image_pairs", "images:image_pairs", partial=True, alias="image_pairs",
     description="n inputs of the report set next to the predicts model's outputs (reconstructions)")

lego("/plot/kalfa/loss_curve", "loss_curve:loss_curve", partial=True, alias="loss_curve",
     description="Every history series over the turns, or the named ones; the axis says epoch when the manifest says "
                 "a turn is one, turn otherwise; x: step draws the per update series of steps.jsonl (the loss, the "
                 "gradient norm of every optimizer) over the steps instead; rates: true adds a panel of the learning "
                 "rates below, and a series may name lr/<optimizer>")

lego("/plot/kalfa/permutation_importance", "permutation_importance:permutation_importance", partial=True,
     alias="permutation_importance", refs={"target": "field"},
     description="The drop in R2 when one feature column is shuffled, the largest first; the model runs again for "
                 "every feature and every repeat, so sample bounds the cost; output names the wire and target the "
                 "field it is scored against when the model has several")

lego("/plot/kalfa/pred_vs_true", "predictions:pred_vs_true", partial=True, alias="pred_vs_true",
     description="Predicted against true values of the test set, one panel per predicted field with its R2, as a "
                 "hexbin density over many points and a scatter over few; the panel is titled with the field name, "
                 "plus the output wire when two outputs predict the same field")
lego("/plot/kalfa/pred_histogram", "predictions:pred_histogram", partial=True, alias="pred_histogram",
     description="The distribution of every predicted field of the test set next to the distribution of its truth "
                 "over the same bins, one column per field, with the ratio of predicted to true counts per bin "
                 "below (a dashed line at one, a Poisson error bar per bin, a bin without true points stays empty); "
                 "log draws the counts on a log axis")
lego("/plot/kalfa/residuals", "predictions:residuals", partial=True, alias="residuals", refs={"target": "field"},
     description="Three panels of one prediction's residual: the distribution with its bias and sigma, the residual "
                 "against the truth as a density, and the mean and median error over equal count bins of the target "
                 "range")
lego("/plot/kalfa/error_map", "predictions:error_map", partial=True, alias="error_map",
     refs={"x": "column", "y": "column", "target": "field"},
     description="The error of one prediction over a 2d grid of two columns: with statistic residual blue is a "
                 "prediction below the truth and red above it, with abs the mean absolute error; bins holding fewer "
                 "than min_count points stay empty")
lego("/plot/kalfa/forecast_samples", "predictions:forecast_samples", partial=True, alias="forecast_samples",
     description="n sample windows of the test set: the true horizon against the predicted one")

lego("/plot/kalfa/samples_gif", "samples:samples_gif", partial=True, alias="samples_gif",
     description="The per turn sample grids of samples/turn_*.png as an animation; skipped with a warning when there "
                 "are none")
lego("/plot/kalfa/samples_matrix", "samples:samples_matrix", partial=True, alias="samples_matrix",
     description="A matrix of the per turn samples of samples/turn_*.pt: one row per turn, n columns; skipped with a "
                 "warning when there are none")
