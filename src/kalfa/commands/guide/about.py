from kalfa.commands.guide.words import and_list
from kalfa.core.build import TOP_LEVEL


RUN = "run"
RUN_KEYS = ("seed", "device", "record", "include", "plugins")
UNKNOWN_COLOR = "grey35"

COLORS = {"run": "blue_violet", "params": "dark_magenta", "data": "dark_cyan", "models": "dodger_blue3",
          "losses": "red3", "metrics": "dark_orange3", "optimizers": "green4", "train": "dark_goldenrod",
          "rules": "deep_pink4", "plots": "turquoise4", "generate": "medium_purple4", "sweep": "chartreuse4"}

ABOUT = {
    "run": ("Top level values that set up the run as a whole. seed seeds every random stream and the splits (0 when "
            "it is not written). device is auto (cuda, then mps, then cpu), cpu, mps, cuda or cuda:<index>. record "
            "is the directory the run writes, new or empty; without it a run writes runs/<config name>_<datetime>. "
            "include merges other files below this one, and plugins are modules imported before the build so that "
            "the short names they register can be used."),
    "params": ("Values the rest of the config reads as $name$. They are substituted after every layer is merged: a "
               "string that is only $name$ takes the value with its type, and inside a longer string the value is "
               "written as text. The flag -p name=value overrides a param from the command line, a sweep varies them "
               "point by point, and a param that nothing reads is a warning."),
    "data": ("One call: the data class. It reads the files, cuts them into the train, valid and test sets, fits what "
             "it learns (preprocessors, a vocabulary) on the train set only and yields the batches. Its batch fields "
             "are what a model takes as inputs and what a loss or a metric names as target, and the values it "
             "exposes once fitted are read elsewhere as @data.<name>. A relative path is looked for in the working "
             "directory, then next to the config files."),
    "models": ("Each name is a call of an nn.Module. kalfa calls it with the batch fields of inputs (default the "
               "inputs of the data class, x for a table) and names what it returns after outputs (default the model "
               "name); these names are what pred refers to. The slot keys inputs, outputs, init, weights, freeze and "
               "ema are read by kalfa and never reach the class; init, weights, freeze and ema apply in that order. "
               "With one model, that model is the predicts model whose outputs fill the predictions table."),
    "losses": ("Each name is a call of a loss. A criterion compares one prediction with one target, (pred, target), "
               "and kalfa picks the two tensors; an objective reads the outputs, the batch and the run context "
               "itself. Every loss writes the series <set>/<name> on the sets it runs on, such as train/mse and "
               "val/mse. A loss that an optimizer minimizes gives the gradients; the others are computed without "
               "gradients and only observe. sample_weight names a row weight of the data (data.weights) that "
               "multiplies the loss of every row of a criterion."),
    "metrics": ("Each name is a call of a metric, accumulated over a whole set and written as <set>/<name>, such as "
                "val/rmse. A metric takes the same pred, target, on, every and rescale keys as a loss. rescale is on "
                "by default when the target has an invertible preprocessor, so the metric speaks the original units. "
                "A loss and a metric of the same name write the same series."),
    "optimizers": ("Each name is a call of a torch optimizer (adam, adamw, sgd or an import path). over names the "
                   "parameters it steps, minimizes names the loss it steps on, and schedule multiplies the base lr "
                   "after every update. The history writes lr/<name> and minimizes/<name>. A rule may patch the "
                   "torch params and minimizes while the run goes on; over and schedule stay fixed."),
    "train": ("The budget and the shape of the training. epochs (turns that each pass over the train set) or updates "
              "({total, turn}) is required, never both. step is the algorithm that trains a turn (supervised when it "
              "is not written), predicts the model whose outputs fill the predictions table, evaluate the sets "
              "evaluated after every turn (the existing valid and test sets when it is not written), checkpoint and "
              "report the weights kept and used at the end, track the values written as series and log the loggers."),
    "rules": ("Each name is a rule, asked after every turn in the order of the config, after the evaluation and "
              "before the checkpoints. A rule fires when its trigger (when) fires, and with after only from the turn "
              "after the named rule fired; without when it fires as soon as it is ready. It fires once unless once "
              "is false. Its effects are set (patches of plan paths, applied to the live objects), stop (the run "
              "ends after the turn) and run (an action)."),
    "plots": ("Each name is a call of a plot. Plots are drawn at the end of a run into <record>/plots from the "
              "record files, and again at any time with kalfa plots RECORD. A std plot saves its image under its own "
              "name, such as loss_curve.png; a plot of your own that returns its figure is saved under its key here."),
    "generate": ("One call: the sampler. At the end of a run it draws n samples into <record>/samples/samples.pt, "
                 "with a png grid for images and a txt for text; kalfa generate RECORD draws more later, and the "
                 "write_samples action uses it from a rule when it names no sampler."),
    "sweep": ("A sweep runs this config once per point, each point with other values of some params. strategy picks "
              "the points, space gives the values of every swept param, objective ranks the points and record is the "
              "sweep root whose numbered directories 0000, 0001 and so on hold the point records. kalfa sweep reads "
              "this section."),
}

UNKNOWN_ABOUT = ("kalfa does not know the top level key {name}, so check reports it as unknown_section. The top "
                 "level keys are {keys}.")

DATA_KEYS = {
    "file": ("the file to read; a relative path is looked for in the working directory, then next to the config "
             "files"),
    "folder": "the folder to read; every subdirectory is a class",
    "split": "how the rows are cut into the train, valid and test sets: a list of ratios or a split call",
    "batch": "the batch size, or a mapping of size, eval_size, workers, drop_last and shuffle",
    "features": "a mapping of column patterns to preprocessors",
    "targets": "a mapping of target columns to preprocessors",
    "transform": "a list of image transforms applied to every set",
    "train_transform": "a list of image transforms that replaces transform for the train set",
}

RUN_NOTES = {
    "seed": "seeds every random stream (noise, samples, loader) and the splits",
    "record": "the record directory the run writes; it must be new or empty, and every tool reads it afterwards",
}


def about_of(name):
    if name in ABOUT:
        return ABOUT[name]
    return UNKNOWN_ABOUT.format(name=name, keys=and_list(TOP_LEVEL))
