# kalfa

A YAML front end for PyTorch training. You write the data, the model, the losses, the metrics, the rules and the
report in one config file; kalfa turns the config into a cirak recipe, cirak compiles the recipe into a tezgah
graph, tezgah runs it. The full definition of the surface is `CONFIG.md`, the class of every key is
`reference.yaml`, the lego reference is `DOCS.md`, the reference configs are the examples under `examples/`.

## Installation

Python 3.13 and above.

```
pip install kalfa            # or: uv add kalfa
kalfa --help
```

The dependencies come with it: cirak and tezgah (the recipe compiler and the pipeline runner), torch, pandas,
pyarrow, scikit-learn, torchmetrics, matplotlib, pillow, scipy, ruamel.yaml, tqdm and optuna (the fed back sweep
strategy); there are no optional extras. The `architecture` plot draws the models with matplotlib from the graph
the config wrote; the `torchview` plot draws them with torchview when it and the graphviz `dot` binary are
installed, and `pairplot`, `violin` and `kde` need `seaborn`; none of the three is a dependency of kalfa, and a plot
that misses one says so in the log and is skipped.

The reference configs, the examples and the test suite live in the repository:

```
git clone https://github.com/dboncukcu/kalfa.git
cd kalfa
uv sync
uv run kalfa --help
uv run python -m pytest -m "not slow and not subprocess"   # the unit and contract tests, seconds
uv run python -m pytest                                     # everything, the run tests included
```

## First run

The folder `examples/01_mlp_regression` holds the config, the data generator and the commands:

```
cd examples/01_mlp_regression
uv run python make_data.py                          # housing.parquet (x0..x7, price) and new.parquet
uv run kalfa check config.yaml                      # without running: unknown keys, unresolvable names, the set table
uv run kalfa run config.yaml --set record=runs/01   # trains; writes under runs/01/
uv run kalfa predict runs/01 --data new.parquet
```

The record directory holds `history.jsonl` (per turn `train/loss_mse`, `val/rmse`, `test/rmse`, `lr/model`),
`checkpoints/best.pt`, `predictions.parquet` (the test set, target and prediction in the original scale), `plots/`
and `resolved.yaml` (the config that runs again on its own). Param and path overrides: `-p epochs=5`,
`--set training.stop=[]`.

## Examples

Under `examples/` every reference config is runnable: each folder owns its `config.yaml` and a `make_data.py`
that writes its data, and its commands are in `examples/README.md`. Highlights:

* Tables: `01_mlp_regression` (rules, stopping, checkpoints), `11_kfold_cv` (an `include` layer, `kfold`,
  `kalfa collect`), `12_resume`, `14_sweep_grid` (the `sweep` section, `kalfa sweep`, queue usage).
* Images: `04_cnn_images` (`image_folder`, a per item chain, a frozen backbone and `unfreeze`), `05_autoencoder`,
  `06_vae_loss_dynamics`, `07_wgan_gp` (EMA, `fid`, `kalfa generate`), `08_ddpm`, `09_simclr`, `13_distillation`.
* Text: `10_char_lm` (`char_tokenizer`, `vocab_size`, the steps mode, `lm_sampler`).
* Two hand written ones: `alad` (a plugin, five models, two optimizers) and `minimal` (the smallest table
  regression).

For large tables `include: [/alias/kalfa/lazy]` binds the same names to sources that read in chunks (the limits are
in `CONFIG.md` section 3).

## Commands

```
kalfa run cfg.yaml [--set path=value ...] [-p name=value ...]     # check, compile, train; opens the record directory
                    [--executor thread --workers N]               # serial by default; under thread an aliasing warning is an error
                    [--log info|debug] [--no-progress]            # print what the run is doing; drop the progress bar
                    [--progress turns] [--log-every N] [--tensorboard]   # only the turn bar; a line every N steps; event files
kalfa check cfg.yaml [--set ...] [-p ...] [--layers] [--dump] [--recipe] [--measure]
                                                                  # only the problems; --measure runs the data block
kalfa describe cfg.yaml [--measure] [--section data|model|...] [--wiring] [--save report.txt]
                                                                  # the config as an analysis, after the same checks
kalfa predict runs/x [--model name] [--which best|last|final] [--data new.parquet] [--device cuda] [--plots [a,b]]
kalfa export runs/x [--format onnx|pt2|state_dict] [--model name] [--out DIR]   # a model in another format
kalfa generate runs/x [--which best|last] [--device cuda]         # writes samples/ with the sampler of the generate section
kalfa plots runs/x [--only a,b] [--set figures.format=pdf]        # redraw the plots section from the record; nothing trains
kalfa resume runs/x [--set training.epochs=N]                     # continues from last.pt or final/ into a new directory
kalfa sweep cfg.yaml [--record root] [--count | --show N | --id N]  # the sweep section: the local loop or one point
kalfa sweep cfg.yaml --plan [--prepare-data] [--record root]      # write the root once: the manifest, sweep.plan, sweep.sub, sweep.sh
kalfa prepare cfg.yaml --out DIR                                  # the data block once; kalfa run cfg.yaml --prepared DIR starts from it
kalfa collect runs/cv_* | kalfa collect <sweep root>              # fold summaries (cv.json, cv.md) or the sweep table and the best point
kalfa stop runs/x | kalfa stop <sweep root>                       # writes stop.json: the record ends after its current turn
kalfa board <root> [--port 8080]                                  # a page over the records under a root: follows them as they change, stops one
kalfa ls [/alias/kalfa/tabular | /criterion | ... | word]         # packs and legos with their kinds and facts; a word searches
kalfa docs [--write DOCS.md]                                      # the lego reference generated from the registry
kalfa contract [--write contract.yaml]                            # the wiring and the flow blocks kalfa runs a config by
kalfa ls|docs [--plugin module ...] [--config cfg.yaml ...]       # the same listings with your own legos imported first
```

`--set path=value`: the path is dotted from the root of the document (`--set training.epochs=5`,
`--set training.rules=[]`, `--set device=cuda`); a single segment can only be a top level key, any other bare name
is an error with a hint. `-p name=value` (`--param`) is the shortcut for `params.name` (`-p lr=1e-4`). The value
is read as YAML; lists are replaced wholesale, a rule list is rewritten with `--set training.rules=[...]`.

`--contract PATH` (on `run`, `check`, `describe`, `predict`, `generate` and `resume`) runs a config against an
edited copy of the contract: the `wiring` (the adapters that wrap the losses and metrics entries, the builder, the
default split and device, the prep steps, the run inputs, the history prefixes, the loader the `batch` section
parametrises, the figures lego the `figures` section parametrises, the plot bus) and the five flow `blocks` the
driver opens. `kalfa contract --write contract.yaml` exports the built in one; every record keeps the copy it ran
with as `contract.yaml`, and a command over a record reads that copy without the option. A step of your own is a
node in a block of the copy (`after: {flow: {notify: {uri: /lego/acme/slack, inputs: {history: history}}}}`).

`--log` prints to stderr what the run is doing at the moment it is doing it, so a long step is not a silent one.
`--log info` (a bare `--log` means `info`) is the narrative: the config files and the plugins, the device, the
record directory, the file as it is read, the set sizes, the preprocessors as they fit, the batch counts, the
parameter counts, the optimizers, one line per turn with its values and how long it took, the checkpoints written,
the rules that fired, why training stopped, the predictions and the plots. `--log debug` adds every node of the
pipeline with its time and the decisions inside the legos (which preprocessor fitted how many columns, how many
steps each optimizer took, why a turn was not the best, which rules did not fire). A line is `time level stage
message`, the stage naming where it comes from (`data.source`, `models`, `training.turn`, `after.predict`); at
`debug` a node line is tagged with its path in the flow (`training.epochs[3].turn`), the same path `events.jsonl`
uses. `resume`, `predict` and `generate` take the same option.

`--no-progress` (on `run` and `resume`) leaves the tqdm bar out: it is never created and tqdm is never imported.
`--log info --no-progress` is then the plain form, one line per turn carrying every loss and metric of that turn
and the learning rates, nothing redrawing itself; `--no-progress` on its own is a silent run that says only how it
ended. A turn with more than one step shows an inner bar over its steps as well (`--progress turns` keeps only the turn bar) and `--log-every N` prints a line every N steps under
`--log`, with the loss, the learning rate and the gradient norm of every optimizer. The bar is not disabled on its own when stderr is not a terminal, because a notebook is exactly such a
stream and that is where a bar is worth the most. The bar itself comes from `tqdm.auto`, so a notebook draws the
ipywidgets one and a terminal the plain one.

```
12:03:41.204  INFO   data.source     reading housing.parquet
12:03:41.412  INFO   data.source     10000 rows, 12 columns (0.21s)
12:03:41.418  INFO   data.split      random: train 7000, valid 1500, test 1500
12:03:42.130  INFO   data.loader     train 55 batches of 128
12:03:42.310  INFO   models          model: 41,217 parameters, all trainable
12:03:42.480  INFO   optimizers      model: adam lr 0.001 over model, loss loss_mse
12:03:44.402  INFO   training.turn   turn 1  train/loss_mse 1.204  val/rmse 1.03  (2.1s)
12:03:44.410  INFO   training.ckpt   wrote best.pt, last.pt
```

Nothing else changes: without the flag the output is the progress bar and the warning summary, as before, and the
record directory gets no new file. The lines do land in `stderr.txt` of the record, because tezgah tees stderr
while the run is live, and the node timings are in `events.jsonl` either way.

Sweeps: the config gets `sweep: {strategy, space, objective, record}` (`grid`, `random`, `sobol` are deterministic
by id; `optuna` is fed back). `kalfa sweep cfg.yaml --plan --record root` writes the root once before any point
starts: `manifest.json` (the strategy, the space, the objective, the total), `sweep.plan` (`N`, the config, the
root, regenerated on every plan) and the two site files it never overwrites, `sweep.sub` for HTCondor (`include :
sweep.plan`, `queue $(N)`) and `sweep.sh` (the environment of the site, then `kalfa sweep cfg.yaml --id $1`);
`--prepare-data` runs the data block once into `<root>/data/` and every point starts from it. `kalfa board <root>`
watches all of it: a reader of the records under a root, served on `127.0.0.1:8080` with no dependency beyond
Python (the page is `src/kalfa/board/static/`, a Vue application shipped with its own copies of Vue and
ApexCharts, so it works without a network, and the address bar carries the record, the tab, the open plot and the view options, so a
link shares exactly one view and a reload keeps it), whose home page lists what is running with its progress
and what finished lately, that lists the runs, points and sweeps by their manifests, tabulates every run with
its params and best value, compares two records (curves overlaid, config diff), follows a running record on
a monitor tab (progress, the monitored metric, the minimized loss, the step loss, the log tail; by default the
page refreshes only when a file of the record changes, through a server sent event stream that stats the
record's files once a second, or every 3, 5 or 10 seconds when the footer says so; a stop button asks the
record to end after its current turn), explores the test predictions, the architecture (a box clicked opens
into its layers block by block, with their shapes and parameter counts), the data pipeline (a stage clicked
lists the columns it added or removed), the fitted preprocessors, the files
of the record and the timeline of its nodes, and
shows per record the latest metrics, one chart per metric with the sets as lines, the step curves, the plots and
the samples, the config, the notes, the node timings, the events and the log tails; per sweep the live table of
points with the best one, the curves of ticked points overlaid and the difference between two of them; on a
batch system it runs on the login node and the browser reaches it through an ssh tunnel. Then `kalfa sweep`
runs every point as an ordinary run under `<root>/<id>/` in a
subprocess, `kalfa collect <root>` writes the table and the best point; on a queue system one job per point with
`kalfa sweep cfg.yaml --id N --record <shared root>` (`examples/14_sweep_grid`).

Devices: `device` is a lego slot, not a plain string. The short form is an alias (`auto` takes the first available
of cuda, mps and cpu; `cpu`; `cuda`; `mps`), the long form is `{uri: cuda, params: {index: 1}}`; without the key the
run is on the cpu. Every device lego checks its own availability and fails with a clear message instead of falling
back silently, so `device: cuda` on a machine without cuda is an error, not a slow cpu run. `rng` is the same
kind of slot for the model seeds: `derived` (the default) initializes every model in a substream of its own named
after it, so adding or renaming another model changes no weights; `indexed` seeds by the definition position, the
rule of the runs made before the key; `global` forks nothing and every model draws from the one stream in build
order; a lego of your own answers `(seed, name, index)`. The chosen device is
written to `device.json` in the record and printed by `kalfa run`. `kalfa predict` and `kalfa generate` take the
same choice as `--device` (a short name, or a lego call in quotes), so a record can be replayed on a GPU box; they
stay on the cpu without it. Your own device lego (`/device/acme/tpu`) plugs into the same slot.

The plots split in two: the ones that read `predictions.parquet` and the models (`pred_vs_true`, `residuals`,
`error_map`, `permutation_importance`, `confusion_matrix`, the curves) and the ones that read the data itself
(`target_vs_features`, `correlation_heatmap`, `feature_distributions`, `target_correlation`, and the seaborn
wrappers `pairplot`, `violin`, `kde`). A data plot draws the set its definition names, `train` without one
(`plots: {corr: {uri: correlation_heatmap, sets: [test]}}`), in the original units. seaborn is not a dependency of
kalfa: its three plots say so in the log and are skipped when it is not installed.

`figures` sets the look of every plot in one place: `{format: png | pdf | svg, width, height, dpi, style}`, where
`width` and `height` are the size of one panel in inches. A single plot overrides them with the definition level
keys `width` and `height`, and `--set figures.format=pdf` switches the whole run from the command line. The plots
share one palette and one grid, so a record's figures look like one document. The section is the params of the
figures lego the contract names (`/lego/kalfa/figures`), so `check` validates it by building the lego.

`check --dump` prints the graph that will run (`flow.yaml`), `--recipe` the document the driver hands to cirak;
`--layers` shows the layer tree and the overridden leaves. The set table is from the file header, before the
transforms; `--measure` runs the data block and prints the real sizes.

## Python API

Every command is a function and the command line is a thin shell over them: `kalfa.api` holds `check`, `run`,
`resume`, `predict`, `generate` and `plots`, `kalfa.collect` the collectors, `kalfa.sweep` the sweep plan. `paths`
is a list of config files (the same layering as on the command line); an entry may also be a config mapping (one
layer without line provenance) or a record directory (its `resolved.yaml` under its own `contract.yaml`). `sets`
is `--set` written as `(dotted path, value)` pairs, and `-p lr=1e-4` is `("params.lr", 1e-4)`.

```python
from kalfa.api import check, generate, predict, resume, run, stop

prepared = check(["config.yaml"], measure=True)           # nothing trains; measure=True counts the sets after the transforms
result = run(["config.yaml"], sets=[("training.epochs", 3), ("record", "runs/nb_$datetime$")])
longer = resume(result.record, sets=[("training.epochs", 10)])
prediction = predict(result.record, data="new.parquet")
samples = generate(result.record, which="best")
```

| Function | Returns | What it carries |
|---|---|---|
| `check(paths, sets=None, load=False, contract=None)` | `Prepared` | `problems`, `errors`, `warnings`, `sizes` (the set table), `loaded` (real sizes), `header` (columns, dtypes, rows), `implicit` (the implicit bindings), `document`, `analysis`, `pipeline`, `dump()` (the `flow.yaml` document) |
| `probe(document)` | `Probe` | the data and model blocks run on their own: `sizes`, `prep` (the fitted plan), `features` (the width of the feature tensor), `parameters` per model, `shapes` of one batch, `notes`; `kalfa.describe.render(prepared, style, sections, probe)` turns the two into the text `kalfa describe` prints |
| `run(paths, sets=None, executor="serial", workers=None, contract=None, monitor=None)` | `RunResult` | `record` (the directory it opened), `report` (tezgah's, `report.outputs["history"]` is the per turn table), `device`; `monitor` is a `kalfa.std.common.log.Monitor` (`Monitor(logging.INFO, progress=False)` prints the turn lines instead of the bar), the default one draws the bar |
| `resume(run_dir, sets=None, executor="serial", workers=None, contract=None, monitor=None)` | `RunResult` | the same, in a new record directory |
| `stop(record, by="cli")` | list | the directories it wrote `stop.json` into (a sweep root and its running points); the run ends after the turn that sees the file, a record that has ended is a `KalfaError` |
| `predict(run_dir, model=None, which=None, data=None, device=None, contract=None, plots=None)` | `Prediction` | `path`, `table` (a DataFrame), `model`, `plots` (the definitions drawn); `data` is a file or a DataFrame (`predictions_frame.parquet`); `plots="all"` or a list of definitions draws them on the table |
| `generate(run_dir, which=None, device=None, contract=None)` | `Generated` | `path`, `samples` |
| `plots(run_dir, only=None, device=None, contract=None)` | `Plots` | `record`, `names`; the plots section redrawn from the record |
| `open_record(run_dir, which=None, sets=None, contract=None)` | `Opened` | what every command over a record starts from: `contract`, `surface`, `document`, `analysis`, `store`, `prep`, and `rebuild()` for the models with the `which` weights |
| `collect_root(root, out=None)` in `kalfa.collect` | mapping | the sweep or fold table and the best point; `load_runs`, `fold_summary`, `sweep_table` are the pieces |
| `plan(paths, sets=None, record=None)` in `kalfa.sweep` | `Plan` | the points of a sweep without running any of them; `write_plan(plan, prepare=False)` writes the root |
| `prepare_data(paths, sets=None, out=None, contract=None)` | `PreparedData` | the data block run once into `out`: the sets as parquet, `fitted/`, `data.json` and a manifest; `run(..., prepared=out)` starts from it |

`check` never raises, it returns what it found; `run` raises `ConfigError` on an error and warns (`CirakWarning`)
for the rest. `print(render_problems(prepared.problems))` from `cirak.errors` prints them the way the command line
does. Writing a lego needs no plugin file here: `@kalfa.lego` in the session registers it like any other.

**In a notebook.** The kernel has to be an environment kalfa is installed in:

```
pip install jupyterlab ipykernel
jupyter lab                           # paths resolve from the notebook's folder
```

A run blocks the cell and draws its usual progress bar. Four things differ from the command line:

```python
import json
import pandas as pd
from IPython.display import Image
from kalfa.api import run
from tezgah import load_run

result = run(["config.yaml"], sets=[("record", "runs/nb_$datetime$")])   # 1. a fresh directory every time

history = pd.DataFrame(result.report.outputs["history"])                 # 2. the metric table, one row per turn
history.plot(y=["train/loss_mse", "val/rmse"])
lines = [json.loads(line) for line in open(f"{result.record}/history.jsonl")]
pd.DataFrame(lines)[["turn", "global_step", "lr/model", "rules"]].tail()  # what the record adds

record = load_run(result.record)                                         # 3. tezgah's record: status, timings, events
Image(f"{result.record}/plots/loss_curve.png")
```

1. A non empty record directory is an error, so a cell you run twice needs `$datetime$` in `record` or a different
   path each time.
2. `result.report.outputs` holds `history` and `predictions`. The in memory `history` is the metric table, one row
   per turn in order; `history.jsonl` carries the same values plus the bookkeeping (`turn`, `global_step`,
   `lr/<optimizer>`, the rules that fired), and `resolved.yaml` and `flow.yaml` say what ran.
3. `load_run` and `RunCatalog` from tezgah read finished runs, so a notebook can compare several record
   directories without rerunning anything.
4. Registering the same URI twice with a different function is an error, which is what a re-run cell does. Restart
   the kernel after editing a lego, or drop the entry first:

```python
from cirak.registry import registry

for uri in ("/criterion/acme/asymmetric",):
    registry._entries.pop(uri, None)
    registry._resolved.pop(uri, None)
```

Interrupting a cell stops the run where it is; `stop(record)` from another cell or shell asks it to end after
the current turn instead (it writes `stop.json`, the run finishes as an early stop would), and `resume(record)`
continues from the last checkpoint into a new directory. On a GPU box `sets=[("device", "cuda")]` picks the
device, and `predict` and `generate` take `device` the same way.

## The record directory

The `record` key gives the directory, `$datetime$` is filled once at the start of the run. A non empty directory is
never written into (an error).

| File | Contents |
|---|---|
| `manifest.json` | the identity, written once at the start: `kind` (`run`, `point`, `sweep`, `data`), the name, the config paths, the params, the kalfa version, a hash of the contract, `turn` (`epoch` when a turn is one pass over the train loader, `steps` when `training.steps` cuts the turns), for a point its values and its root, for a prepared run the directory it started from |
| `host.json` | the hostname, the pid and the working directory of the process that wrote the record |
| `stop.json` | a stop request (who asked and when): `kalfa stop <record>`, the board's stop button or ctrl-c during training writes it, `touch` does too; the loop ends after the turn that sees it and the run finishes as an early stop would |
| `resolved.yaml` | the config with its aliases and `$param$`s resolved; the source of every overridden value in a comment; runs again on its own |
| `contract.yaml` | the contract the run was compiled by (the wiring and the flow blocks); `predict`, `generate` and `resume` read it back |
| `flow.yaml` | the tezgah graph that ran: the component tables, the model blocks, the expanded flow and tezgah's resolution comments |
| `history.jsonl` | per turn the `train/`, `val/`, `test/` values (`val/` steers the checkpoint, the stop and the rules, `test/` is only reported), `global_step`, `lr/<optimizer>`, `minimizes/<optimizer>` (the loss it stepped on that turn), `seconds`, the rules that fired |
| `architecture.json` | the graph of every report model: the boxes of the architecture drawing with their column and row, the traced shapes and the wires, the layers inside every node as a tree with their shapes and parameter counts; the board draws it and opens a box into its layers |
| `steps.jsonl` | per update `step`, `turn`, `loss/<optimizer>`, `lr/<optimizer>` and, under `grad_clip`, `grad_norm/<optimizer>`; `loss_curve` with `x: step` draws it, `rates: true` adds the learning rates below either curve |
| `data.json` | the shape of the data at every stage of the data block: rows and columns per transform with the transform's call, the sets, the fitted objects, the features and targets, the loaders; `data_pipeline` draws it |
| `events.jsonl`, `run.json`, `stdout.txt`, `stderr.txt` | tezgah's event stream and summary |
| `checkpoints/` | `best.pt`, `last.pt` (by policy); models, optimizers, EMAs, counters, rule states, RNG |
| `final/state.pt` | always, once the run ends, with the same scope |
| `fitted/` | the fitted state: `preprocessors/` (one file per name and `plan.json`, the column chains), `frames/` (the frame transforms) and `calibrate/` (the calibrations and their notes); `predict`, `plots` and `resume` read from here |
| `predictions.parquet` | the test set: `row`, the targets (inverted), `pred_<output>` (inverted), `raw_<output>`, the calibration flags; with `training.targets` one column per predicted field, `pred_<output>_<field>`; `predictions_<set>.parquet` for a set beyond the three the std splits return |
| `plots/` | the outputs of the plot legos, named after the definition (`plots.roc` → `roc.png`, or the format `figures` asks for; `architecture` writes text); `kalfa plots` redraws them from the record and `predict --plots` writes them with the suffix of its predictions file (`roc_new.png`) |
| `samples/` | the output of `generate`: `samples.pt` (for images `grid.png` too), `samples.txt` for text; `turn_<n>.*` from `sample_writer` |
| `plugins/` | copies of the plugin modules the run imported, so predict, generate and resume work from the record |
| `device.json` | the chosen device and the device lego that picked it |
| `git.json` | the commit of the config's repository and whether it was dirty |
| `export/` | what `kalfa export` wrote, `<model>.onnx`, `.pt` |
| `tensorboard/` | under `--tensorboard`, the event files of every history and step value and the params as hparams |
| `resume.json` | in a resumed run the source run and the checkpoint |

`kalfa predict` rewrites `predictions.parquet` on the run's own test set; with `--data`
`predictions_<file name>[_<model>].parquet`.

## Several targets at once

A table can carry more than one target column, and a model more than one output wire. `training.targets` says
which wire predicts which target fields:

```yaml
data:
  fields:
    "y_*":  {target: true, preprocessors: [y_scaler]}   # three columns, three fitted scalers
    z:      {target: true, preprocessors: [y_scaler]}
    "x*":   {preprocessors: [x_scaler]}

training:
  predicts: full
  targets:
    y_hat: "y_*"        # a field name, a list of names or a glob
    z_hat: z

losses:
  l_y: {uri: mse, output: y_hat}     # the target comes from the table
  l_z: {uri: mse, output: z_hat}
```

The fields a selector names become one tensor, so `mse` is called once with two `(batch, 3)` sides; every column
keeps its own fitted preprocessor, so the metrics and `predictions.parquet` invert each one with its own scale
(`pred_y_hat_y_a`, `pred_y_hat_y_b`, ...) and `pred_vs_true` draws one titled panel per field; two outputs
that predict the same field (a baseline wire and a corrected one) carry the wire in the title, `combined_z
(combined_hat)`. `kalfa describe`
prints the table and the place every column takes in its wire (`y_a → y_hat[0]`). `examples/15_multi_target` is
the runnable version.

## Looking at a config

`kalfa check` answers one question: are there problems. `kalfa describe` answers the rest. It runs the same checks
and then prints the config as an analysis: where the data comes from and what every field goes through, the model
graphs as wire diagrams, the optimizers with the loss each one carries, the losses and metrics with the sets they
are reported on, the rule chain with its conditions, what the run writes at the end, and a column by column table
of the source (dtype, field, preprocessor chain, role, the place it takes in the feature tensor).

In a terminal the colors carry the grammar: a lego name is cyan, a parameter name is dim, the values stay plain.
So `parquet  housing.parquet` reads as the source lego and its file, `grouped_table prefix=y_ name=y` as the feed
lego and its parameters, and `random  0.8 / 0.1 / 0.1  seed=7` as the split lego, its ratios and its seed.

Statically it reads the file header and the compiled recipe, so it needs no data beyond the source header.
`--measure` runs the data and model blocks for real (nothing is written): the set sizes after the transforms, the widths
a fitted `one_hot` produces, the tensor slots of every column, and the parameter counts of models whose layers are
lazy until the first batch. `--section data|model|training|after|columns|wiring` narrows the output, `--wiring`
adds the implicit bindings of the compiled pipeline to the default sections. Tables are fitted to the width the
terminal reports (`COLUMNS=140 kalfa describe ...` overrides it, a narrow terminal clips the widest column with an
ellipsis); with `--save report.txt`, or whenever the output is not a terminal, nothing is clipped and the section
rules are cut to the longest line instead.

```
kalfa describe config.yaml
kalfa describe config.yaml --measure
kalfa describe config.yaml --section columns
```

## Alias packs and plugins

`include: [/alias/kalfa/tabular]` brings the short names (`parquet`, `standard_scaler`, `linear_relu`, `mse`,
`rmse`, `adam`, `supervised`, `plateau`, `best`, `loss_curve` ...). There are four packs, `tabular`, `vision`,
`text` and `lazy` (large tables read in chunks), and every one of them includes `/alias/kalfa/base`, the names
they all share; a pack adds only its own names, so a config that wants tables and images includes two packs. The
table is flat, a full URI is valid everywhere; `kalfa ls /alias/kalfa/tabular` prints the resolved contents,
`kalfa ls tokenizer` searches names and descriptions. Your own lego is registered with `@kalfa.lego` in a Python module next to the config and comes in
with `plugins: [module]`; `examples/alad/myexample.py` (the ALAD objectives of that example) and the plugin
modules of the other examples show how. `kalfa ls alad --plugin myexample` and `kalfa docs --config cfg.yaml` import those modules before the
listing, so your own legos come with their signature and facts (`docs` prints them in a separate `Plugin legos`
section, `DOCS.md` stays the reference of what kalfa ships). Details in the Development section.

## Development

**Writing a lego.** A lego is a function or a class registered with `kalfa.lego`; its kind is the first segment
of the URI (`/criterion/`, `/objective/`, `/metric/`, `/layer/`, `/pre/`, `/source/`, `/split/`, `/feed/`,
`/init/`, `/optimizer/`, `/schedule/`, `/turn/`, `/trigger/`, `/checkpoint/`, `/generate/`, `/plot/`,
`/strategy/`, `/data/`, `/lego/`; the list is `kalfa.kinds.KINDS`), and `check` decides from it where the lego may be
written. A lego that builds one object of one class is that class, decorated directly (`@kalfa.lego(...)` above
`class Rmse(Metric)`); cirak reads the constructor's signature. A lego that returns a plain value, chooses between
classes or defers a build is a function. The facts a lego declares: `alias` (short names), `partial` (built with
its params at compile time, called later), `state` (stateful, goes into the record), `returns` (the outputs of a
flow step), `bus` (bus keys bound to defaulted parameters), `mutates` (inputs changed in place and returned under
the same name), `aliases` (the output holds the inputs), `refs` (which params are references and of what type:
`model`, `loss`, `criterion`, `schedule`, `preprocessor`, `generate`, `field`, `column`), `uses` (`predicts` when
the lego needs the prediction model), `needs_grad` (an objective that needs gradients in the evaluation pass),
`extras` (the training keys a turn accepts), `grouped` (a preprocessor fitted once over every column that names
it), `requires` (a library that is not a dependency of kalfa, a name or a list; `check` warns when it is not
installed and the lego loads it with `kalfa.std.common.optional.load(name)`, which returns `None` with a warning
when it is absent). The last six are kalfa's own vocabulary, declared to cirak at import (`kalfa.kinds.FACTS`
through `cirak.declare_facts`); cirak stores them and reads none of them, and a fact kalfa never declared stays a
`RegistryError`, so a misspelled one is still caught. A `/data/` lego is a run time component: `{uri: name}` as a
param value, built once the data exists with the parameters its signature names (`loader`, `prep`, `target`).
The turn contract is the signature of `/turn/kalfa/alternating` (`models, optimizers, emas, counters,
composites, effects, loader, params, extra, losses, metrics, losses_keys, metrics_keys, predicts, steps, stream`
plus the bus keys `device`, `prep`, `record`), returning `{models, optimizers, emas, counters, stream, metrics}`;
`stream` is the batch cursor a `steps` run carries from one turn to the next (`None` at the first turn and in an
epoch run); a custom turn
builds a `Pass` (`kalfa.std.common.runtime`) and calls `update` from `kalfa.std.turn.base` for one optimizer
update over a list of batches, so it writes only its own loop. The strategy contract is the `Strategy` base class
of `kalfa.std.strategy.base`: `deterministic`, `total(space)` and `point(space, index)`, or `ask(space, mode)` and
`tell(trial, value)` for a fed back strategy.

```python
import kalfa
import torch


@kalfa.lego("/criterion/acme/asymmetric", alias="asymmetric", partial=True,
            description="Squared error, over prediction weighted by factor")
def asymmetric(predictions, targets, factor=2.0):
    error = predictions.reshape(-1) - targets.reshape(-1)
    return torch.where(error > 0, factor * error ** 2, error ** 2).mean()


@kalfa.lego("/trigger/acme/after_minutes", partial=True, description="Fires once the run is older than minutes")
def after_minutes(metrics, turn_index, state, minutes=30.0):
    import time

    state = dict(state or {})
    started = state.setdefault("started", time.time())
    return time.time() - started > 60.0 * minutes, state
```

A trigger takes the turn's `metrics`, the `turn_index` and its own `state` mapping and returns `(fired, state)`; a
criterion takes `(predictions, targets, **params)`; an objective `(models, batch, **params)` and, when its
signature names them, `step`, `epoch`, `rng`, `scaler` and `losses` (the other losses of the table on the same
batch, by name). Every other contract is a base class a plugin subclasses; `check` and `describe` read its class
attributes without building the object:

| contract | base class | what a subclass writes |
|---|---|---|
| preprocessor | `kalfa.std.pre.base.Preprocessor`, with `Scaler` (invertible, `rescales`), `Encoder` (fitted, `fits`) and `Tokenizer` (`encode`, `decode`, `size`) below it | `apply(values)`; `fit(values)` when `fits`; `partial_fit(values)` when `incremental`; `inverse(values)`; `columns(name)` for a widening preprocessor; `decode(scores)` when `decodes`; `dtype` for a Dataset source field; `grouped = True` for one object over every column that names it (`fit(matrix)`, `apply(matrix, columns=None)`, `inverse(matrix, columns=None)`), the sklearn way |
| metric | `kalfa.std.metric.base.Metric` | `reset()`, `update(...)` naming what it wants among `predictions`, `targets`, `models`, `batch`, `rng`, `predicts`, `record`, `turn`, `prep`, `set`, and `compute()` (`None` reports nothing) |
| model | `kalfa.std.builder.base.Model` (an `nn.Module`) | `inputs`, `outputs`, `initialized`, `trainable`; the std builder's `Module` and the EMA copy are the two implementations |
| dataset | `kalfa.std.feed.base.Dataset` and `IterableDataset` (a stream) | `inputs`, `targets`, `frame`, `rows()`, `labels(name)`, `size()` (`None` for a stream), `count()` |
| checkpoint policy | `kalfa.std.checkpoint.base.Policy` | `monitor`, `tags(metrics)`, `state()`, `restore(state)` |
| frame transform | `kalfa.std.frame.base.FrameTransform` | `fit(df)` on the train set, `apply(df)` per set |
| calibration | `kalfa.std.calibrate.base.Calibration` | `fit(models, loaders, prep, device, predicts)` at the end of a run, `apply(table)` on the predictions, `note()` for `calibrate.json` |
| optimizer | `kalfa.std.optimizer.base.Optimizer` | `torch_class`; the base creates the torch optimizer on first use, applies the schedule and answers `lr()` |
| strategy | `kalfa.std.strategy.base.Strategy` | `deterministic`, `total(space)`, `point(space, index)`, or `ask` and `tell` |
| losses and metrics entries | `kalfa.std.common.runtime.Loss` | the three std adapters (`/adapter/kalfa/criterion`, `metric`, `objective`) wrap every entry, so the tables are uniform: `loss(context, keys)`, `tracker(name, keys, rescale)`, `with_param(name, value)`, `resolve(**available)` |

A lego that receives `device` gets a `kalfa.std.common.device.Device`: `torch` (the `torch.device`), `move(batch)`,
`place(modules)`, `generator()`, `autocast(enabled)`, `scaler(enabled)`. A device lego (`/device/acme/tpu`) takes
no inputs, returns a `torch.device` and raises when the device is not available (`device: {uri: /device/acme/tpu}`
then selects it); kalfa wraps it. A source lego returns a DataFrame, a `Stream` (`kalfa.std.common.stream`) or
`Samples(source)` (`kalfa.std.common.samples`) over an object with `fields`, `dtypes`, `column(name)` for the
light fields and one mapping per item. A plot that reads the history wraps it in
`kalfa.std.common.history.History` (`series`, `last`, `best`).

**Writing a plot.** A plot lego takes `predictions, history, models, record` and, by naming them in its
signature, anything else the run holds: `prep`, `train_loader`, `valid_loader`, `test_loader`, `loaders` (the
three in one mapping), `predicts`, `sets`, `name`, `device`, `counters`, `optimizers`, `emas`, `rules`, and
`figures`; the bus keys are the `plot_bus` mapping of the contract's wiring. A key the plot cannot work without
goes into its `needs` fact (`needs=["train_loader"]`), and `kalfa plots` or `predict --plots` skip the plot with a
log line when the key is not there instead of calling it. Draw through `figures`, a `kalfa.std.common.figure.Figure` (`single`, `grid`, `label`, `density`,
`profile`, `binned`, `colorbar`, `save` and the palette) carrying the format, the panel size and the style of the
run's `figures` section, already sized for the definition's `width` and `height`, and the figure comes out in the
run's own look. `kalfa.std.plot.base.set_frame(loaders, prep, set)` gives
the columns of one set as a DataFrame in the original units, features and targets together, which is what
`target_vs_features` and `correlation_heatmap` draw.

**Logging from a lego.** `logging.getLogger("kalfa.<stage>")` is the whole contract: nothing is declared, no fact,
no parameter, and the line only appears when the user asked for it with `--log`. The stage is what the third column
of a log line shows, so name it after the place in the run (`data.source`, `models`, `training.turn`,
`after.plots`), not after the module. `kalfa.std.common.log` has `logger_for(stage)` for that, plus `clock()` and
`since(started)` for the durations and `number(value)` for the metric formatting. INFO is what a user wants to see
without asking for detail, DEBUG is the decision behind it; nothing per batch at any level, and a log line never
computes a value the lego does not already have (guard it with `log.isEnabledFor(logging.DEBUG)` when it would).

**Plugin layout.** A module next to the config (or in a `plugins/` folder next to it) comes in with
`plugins: [module]`; the run copies the modules it imported into `<record>/plugins/`, so `predict`, `generate` and
`resume` work from the record alone. A pip package registers its legos on import and declares itself with the
`cirak.plugins` entry point (`[project.entry-points."cirak.plugins"] acme = "acme_legos"` in its `pyproject.toml`),
then `plugins: [acme_legos]` works anywhere the package is installed. Aliases declared by plugin legos are usable
without a pack.

**The std tree.** The URI of a std lego is `/<kind>/<pack>/<name>` and its module sits under
`src/kalfa/std/<kind>/<pack>/`: a file holds one lego with a body of its own (`std/pre/kalfa/char_tokenizer.py`) or
a family of small legos of one kind and one pack (`std/criterion/kalfa/regression.py` holds `mse`, `mae`, `huber`,
`log_cosh` and `weighted_mse`; `std/layer/torch/activations.py` the activations), the helpers of the family above
them in the same file and the registered things last. `src/kalfa/std/__init__.py` imports every
`std/<kind>/<pack>/*.py`, so adding a lego is adding it to its family or adding a file; a contract test checks that
every std lego is registered from a module of its kind and pack and that every module registers one, and `kalfa ls`
prints the module after the facts. Code two families of a kind share, and the base classes, sit in the kind's
`base.py` (`std/pre/base.py`: the field plan, `Prep`, the frames, the base classes `Preprocessor`, `Scaler`,
`Encoder`, `Tokenizer`), code two legos of one pack share in the pack's `base.py` (`std/pre/sklearn/base.py`), and
code that crosses kinds in `std/common/` (`runtime`, `figure`, `log`, `samples`, `stream`, `deferred`, `prediction`,
`generation`, `diffusion`). A kind whose legos share nothing but a family has no `base.py`. `common/` never imports
a lego module, a `base.py` imports `common/`, a lego module imports `common/`, its bases and the family file that
owns a helper it borrows (`lego/kalfa/headers.py` takes `ImageFolder` from `source/kalfa/samples.py`). The base
classes the legos build on are `Preprocessor` (`pre/base.py`), `Metric` (`metric/base.py`), `Loss`
(`common/runtime.py`), `Model` (`builder/base.py`), `Dataset` (`feed/base.py`), `Policy` (`checkpoint/base.py`),
`Optimizer` (`optimizer/base.py`), `Strategy` (`strategy/base.py`), `FrameTransform` (`frame/base.py`) and
`Calibration` (`calibrate/base.py`); a plugin subclasses them.

**Adding an example.** Write `examples/<nn>_<name>/config.yaml` with a comment header (aliases from a pack, paths
relative to the folder), `make_data.py` with the generator of its data inside it (an example folder stands on its own; the tests borrow the generators through `tests/helpers.py`), the plugin module next to them when the
config names one, and the README entry (data, run, predict or generate); regenerate the golden files
(`uv run python tools/regenerate.py --all`) and read the dump by eye; write the missing legos with unit tests under
`tests/unit/std/`; write `tests/runs/test_<name>.py` with the `dataset` and `trained` fixtures and the small
overrides of `tests/runs/conftest.py`, so that it runs on a CPU in seconds and checks the record directory.

**Golden files.** Everything generated lives under `tests/golden/` and is written by one tool,
`tools/regenerate.py`: `dumps` (the expanded graph of `kalfa check --dump` per example), `recipes` (the driver
document per example), `registry` (every std URI with its kind, aliases, signature, facts and description, the
packs and the plot bus keys), `describe` (the analysis of every example) and `docs` (`DOCS.md` at the root).
`--all` writes every one, `--check` writes nothing and reports what would change. The contract tests under
`tests/contract/` compare the same texts, so the suite fails on a stale golden file and says which tool to run.

**The test tree.** `tests/unit/` holds the unit tests of the core and of the std legos, `tests/contract/` the
tests that the repository agrees with itself (the golden files, the packs, the examples), `tests/runs/` one end
to end test per example on its own data, `tests/cli/` the command line. Three markers select what to run:
`slow` (a test that trains), `subprocess` (the sweep loop) and `optional` (seaborn, torchview, cuda or mps). A
test that registers a lego does it inside the registry scope every test runs in, so nothing leaks into the next.

**Test rules.** `uv run python -m pytest` must stay green; every lego has a unit test; every example has a run
test on its own synthetic data, CPU only, no network; the reinstalled environment (`uv sync --reinstall`) runs
the suite before a release.

**Versions and releases.** kalfa pins `tezgah>=` and `cirak>=` in `pyproject.toml`; bump the version, regenerate
the golden files (`uv run python tools/regenerate.py --all`, the headers carry the version), run the suite
against the reinstalled environment (`uv sync --reinstall`), then build (`uv build`). kalfa is 0.2.4 and needs
tezgah 0.2.0 or later and cirak 0.2.3 or later.
