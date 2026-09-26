from kalfa.registration import pack


lego = pack(__name__)


lego("/lego/kalfa/architecture_note", "architecture_note:architecture_note", returns=None, bus=["record", "device"],
     description="The graph of every report model written to the record as architecture.json: the boxes of the "
                 "architecture drawing with their column and row, the layers inside every node as a tree with the "
                 "shapes one batch traced and their parameter counts, the wires as arrows; the board draws it and "
                 "opens a box into its layers")

lego("/lego/kalfa/calibrate", "calibrate:calibrate", returns="calibrations", bus=["record", "device"],
     description="Fit every calibration of the calibrate section on the report models and the sets, in order, and "
                 "keep them in the record under fitted/calibrate; predict applies them to its table")

lego("/lego/kalfa/clone", "clone:Ema", state=True,
     description="An exponential moving average copy of a model with the given decay")

lego("/lego/kalfa/data_report", "data_report:data_report", returns="data_report", bus=["record"],
     description="The shape of the data at every stage of the data block, read from the bus keys the stages wrote: "
                 "the rows and columns of the source and after every transform with the transform's call, the sets "
                 "after the split and after their transforms, the fitted frame transforms and preprocessors, the "
                 "features and targets, the loaders; written to the record as data.json")

lego("/lego/kalfa/transform_set", "data_steps:transform_set",
     description="Apply the transforms that name this set, in order; the frame passes untouched without any")
lego("/lego/kalfa/fit_frames", "data_steps:fit_frames", returns="frames", state=True, bus=["record"],
     description="Fit the frame transforms on the train set, each on what the ones before it produced, and keep them "
                 "in the record under fitted/frames")
lego("/lego/kalfa/apply_frames", "data_steps:apply_frames",
     description="Apply the fitted frame transforms to one set, in the order they were fitted")
lego("/lego/kalfa/read_frames", "data_steps:frames_of", returns="frames",
     description="The fitted frame transforms of a record, read from fitted/frames")

lego("/lego/kalfa/evaluate", "evaluate:evaluate", returns="metrics", bus=["device", "prep", "record"],
     description="Losses (model scale) and metrics (original scale, through prep) of one set under no_grad; an empty "
                 "set gives an empty mapping; record reaches metrics that write files")

lego("/lego/kalfa/figures", "figures:Figures",
     description="The look of every plot of a run: the file format, the size of one panel in inches, the dpi and the "
                 "style (kalfa, or none for matplotlib's own); the figures section is its params and the built object "
                 "reaches every plot that names figures")

lego("/lego/kalfa/generate", "generate:generate", returns=None, bus=["record"],
     description="Run the generate lego with the report models; nothing without a generate section")

lego("/lego/kalfa/parquet_header", "headers:parquet_header",
     description="The columns, their arrow types and the row count of a parquet file, from its metadata; the listed "
                 "columns only when the source names them")
lego("/lego/kalfa/csv_header", "headers:csv_header",
     description="The columns, the dtypes of the first rows and the line count of a CSV file; the listed columns only "
                 "when the source names them")
lego("/lego/kalfa/image_folder_header", "headers:image_folder_header",
     description="The fields, the dtypes, the image count and the classes of an image folder")
lego("/lego/kalfa/text_lines_header", "headers:text_lines_header",
     description="The text field and the line count of a text file")
lego("/lego/kalfa/prepared_header", "headers:prepared_header",
     description="The header a prepared directory recorded in its manifest: the columns, the dtypes and the rows")

lego("/lego/kalfa/history", "history:history", returns=None,
     bus=["monitor", "metrics", "turn_index", "counters_next", "optimizers_next", "rules_next", "effects", "record"],
     description="Append the turn's line to history.jsonl: the metrics, the learning rate and the loss every "
                 "optimizer minimized this turn (as the rules set it), every other effect of the rules in force as "
                 "effect/<target>, the duration as seconds, the rules that fired; and hand it to the monitor")

lego("/lego/kalfa/pixel_features", "pixel_features:pixel_features",
     description="A cheap FID feature extractor for demos and tests: images pooled to size by size and flattened; "
                 "pass it as fid's extractor param")

lego("/lego/kalfa/predict", "predict:predict", returns="predictions", bus=["record", "device"],
     description="Predict a set with the report model, invert the target chain, apply the fitted calibrations, write "
                 "predictions.parquet for the test set and predictions_<set>.parquet for another")

lego("/lego/kalfa/fit", "prep:fit", returns="prep", state=True, bus=["record"],
     description="Resolve the field globs and fit every preprocessor chain on the train set; keys carry the sets a "
                 "preprocessor is limited to")
lego("/lego/kalfa/read_prep", "prep:prep_of", returns="prep",
     description="The fitted preprocessing plan of a record, read from its preprocessors directory")
lego("/lego/kalfa/apply", "prep:apply",
     description="Apply the fitted chains to one set and type its columns; keys carry the sets a preprocessor is "
                 "limited to; the rows the mask query selects stay in the frame and are not scored, the loader leaves "
                 "them out and the plots see them as masked")

lego("/lego/kalfa/run_all", "run_all:run_all", returns=None, bus=["record", "figures"],
     description="Run every plot of the plots table with the predictions, the history and the models; keys carry the "
                 "definition level keys (inputs, sets, width, height) and the lego of the plot, whose refs type the "
                 "inputs and whose needs name the bus keys it cannot work without (skipped with a log line when one "
                 "is missing); bus carries everything else the run has and a plot receives whatever its signature "
                 "names, plus loaders, predicts, sets, name (with the suffix of the predictions it draws) and "
                 "figures, the look of the run's plots sized for the definition")

lego("/lego/kalfa/ratio_sizes", "sizes:ratio_sizes",
     description="The set sizes a split by ratios produces from rows rows; without rows, which sets it produces")
lego("/lego/kalfa/kfold_sizes", "sizes:kfold_sizes",
     description="The set sizes a k fold split produces from rows rows; without rows, which sets it produces")
lego("/lego/kalfa/given_sizes", "sizes:given_sizes",
     description="The set sizes of a given split: the source rows for train, the header of every given file for the "
                 "other sets (header reads a path like the source)")
lego("/lego/kalfa/prepared_sizes", "sizes:prepared_sizes",
     description="The set sizes a prepared directory recorded in its manifest")

lego("/lego/kalfa/init_state", "state:init_state", returns="epochs_left", mutates=["state"],
     bus=["resume", "device", "skip_training"],
     description="Move the state to the device, load a checkpoint when resuming and restore the checkpoint policy "
                 "from it, count the turns left (none when a repair skips training)")
lego("/lego/kalfa/checkpoint", "state:checkpoint", returns=None, bus=["metrics", "record"],
     description="Write the checkpoint files the policy asks for; nothing without a policy")
lego("/lego/kalfa/save_final", "state:save_final", returns=None, bus=["record"],
     description="Write final/state.pt with the full state and the checkpoint policy's state once training ends")
lego("/lego/kalfa/select", "state:select", returns="selected", bus=["record"],
     description="The report models: copies loaded from best.pt, or the final state for last")

lego("/lego/kalfa/const", "values:const", description="A fresh copy of a constant value")
lego("/lego/kalfa/identity", "values:identity", aliases="value", description="The value itself")
lego("/lego/kalfa/pack", "values:pack", aliases="items", description="A mapping of the given items")
lego("/lego/kalfa/merge", "values:merge",
     description="Merge the per set metrics under the prefixes of the sets (train/, val/, test/)")
