# kalfa

A YAML front end for PyTorch training. You write the data, the model, the losses, the metrics, the rules and the
report in one config file; kalfa turns the config into a cirak recipe, cirak compiles the recipe into a tezgah
graph, tezgah runs it. The full definition of the surface is `CONFIG.md`, the class of every key is
`configs/reference.yaml`, the lego reference is `DOCS.md`, the fourteen reference configs are under `configs/`.

## Installation

Python 3.13 and above.

```
pip install kalfa            # or: uv add kalfa
kalfa --help
```

The dependencies come with it: cirak and tezgah (the recipe compiler and the pipeline runner), torch, pandas,
pyarrow, scikit-learn, torchmetrics, matplotlib, pillow, scipy, ruamel.yaml, tqdm. `pip install "kalfa[sweep]"`
adds optuna for the fed back sweep strategy. The `architecture` plot draws the models when `torchview` and the
graphviz `dot` binary are installed, and says so in the log when they are not; neither is a dependency of kalfa.

The reference configs, the examples and the test suite live in the repository:

```
git clone https://github.com/dboncukcu/kalfa.git
cd kalfa
uv sync
uv run kalfa --help
uv run pytest          # about 260 tests, half a minute
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

Under `examples/` every reference config is runnable: each folder has `config.yaml` (a byte for byte copy of the
one under `configs/`), `make_data.py`, and its commands are in `examples/README.md`. Highlights:

* Tables: `01_mlp_regression` (rules, stopping, checkpoints), `11_kfold_cv` (an `include` layer, `kfold`,
  `kalfa collect`), `12_resume`, `14_sweep_grid` (the `sweep` section, `kalfa sweep`, queue usage).
* Images: `04_cnn_images` (`image_folder`, a per item chain, a frozen backbone and `unfreeze`), `05_autoencoder`,
  `06_vae_loss_dynamics`, `07_wgan_gp` (EMA, `fid`, `kalfa generate`), `08_ddpm`, `09_simclr`, `13_distillation`.
* Text: `10_char_lm` (`char_tokenizer`, `vocab_size`, the steps mode, `lm_sampler`).
* Two examples own their config instead of copying one from `configs/`: `alad` (a plugin, five models, two
  optimizers) and `minimal` (the smallest table regression).

For large tables `include: [/alias/kalfa/lazy]` binds the same names to sources that read in chunks (the limits are
in `CONFIG.md` section 3).

## Commands

```
kalfa run cfg.yaml [--set path=value ...] [-p name=value ...]     # check, compile, train; opens the record directory
                    [--executor thread --workers N]               # serial by default; under thread an aliasing warning is an error
                    [--log info|debug] [--no-progress]            # print what the run is doing; drop the progress bar
kalfa check cfg.yaml [--set ...] [-p ...] [--layers] [--dump] [--recipe] [--load]
                                                                  # only the problems; --load runs the data block
kalfa describe cfg.yaml [--load] [--section data|model|...] [--wiring] [--save report.txt]
                                                                  # the config as an analysis, after the same checks
kalfa predict runs/x [--model name] [--which best|last] [--data new.parquet] [--device cuda]
kalfa generate runs/x [--which best|last] [--device cuda]         # writes samples/ with the sampler of the generate section
kalfa resume runs/x [--set training.epochs=N]                     # continues from last.pt or final/ into a new directory
kalfa sweep cfg.yaml [--record root] [--count | --show N | --id N]  # the sweep section: the local loop or one point
kalfa collect runs/cv_* | kalfa collect <sweep root>              # fold summaries (cv.json, cv.md) or the sweep table and the best point
kalfa ls [/alias/kalfa/tabular | /criterion | ... | word]         # packs and legos with their kinds and facts; a word searches
kalfa docs [--write DOCS.md]                                      # the lego reference generated from the registry
kalfa ls|docs [--plugin module ...] [--config cfg.yaml ...]       # the same listings with your own legos imported first
```

`--set path=value`: the path is dotted from the root of the document (`--set training.epochs=5`,
`--set training.rules=[]`, `--set device=cuda`); a single segment can only be a top level key, any other bare name
is an error with a hint. `-p name=value` (`--param`) is the shortcut for `params.name` (`-p lr=1e-4`). The value
is read as YAML; lists are replaced wholesale, a rule list is rewritten with `--set training.rules=[...]`.

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
ended. The bar is not disabled on its own when stderr is not a terminal, because a notebook is exactly such a
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
by id; `optuna` is fed back), `kalfa sweep` runs every point as an ordinary run under `<root>/<id>/` in a
subprocess, `kalfa collect <root>` writes the table and the best point; on a queue system one job per point with
`kalfa sweep cfg.yaml --id N --record <shared root>` (`examples/14_sweep_grid`).

Devices: `device` is a lego slot, not a plain string. The short form is an alias (`auto` takes the first available
of cuda, mps and cpu; `cpu`; `cuda`; `mps`), the long form is `{uri: cuda, params: {index: 1}}`; without the key the
run is on the cpu. Every device lego checks its own availability and fails with a clear message instead of falling
back silently, so `device: cuda` on a machine without cuda is an error, not a slow cpu run. The chosen device is
written to `device.json` in the record and printed by `kalfa run`. `kalfa predict` and `kalfa generate` take the
same choice as `--device` (a short name, or a lego call in quotes), so a record can be replayed on a GPU box; they
stay on the cpu without it. Your own device lego (`/device/acme/tpu`) plugs into the same slot.

`check --dump` prints the graph that will run (`flow.yaml`), `--recipe` the document the driver hands to cirak;
`--layers` shows the layer tree and the overridden leaves. The set table is from the file header, before the
filters; `--load` runs the data block and prints the real sizes.

## Python API

Every command is a function and the command line is a thin shell over them: `kalfa.api` holds `check`, `run`,
`resume`, `predict` and `generate`, `kalfa.collect` the collectors, `kalfa.sweep` the sweep plan. `paths` is a list
of config files (the same layering as on the command line), `sets` is `--set` written as `(dotted path, value)`
pairs, and `-p lr=1e-4` is `("params.lr", 1e-4)`.

```python
from kalfa.api import check, generate, predict, resume, run

prepared = check(["config.yaml"], load=True)              # nothing trains; load=True measures the real set sizes
result = run(["config.yaml"], sets=[("training.epochs", 3), ("record", "runs/nb_$datetime$")])
longer = resume(result.record, sets=[("training.epochs", 10)])
prediction = predict(result.record, data="new.parquet")
samples = generate(result.record, which="best")
```

| Function | Returns | What it carries |
|---|---|---|
| `check(paths, sets=None, load=False)` | `Prepared` | `problems`, `errors`, `warnings`, `sizes` (the set table), `loaded` (real sizes), `header` (columns, dtypes, rows), `implicit` (the implicit bindings), `document`, `analysis`, `pipeline`, `dump()` (the `flow.yaml` document) |
| `probe(document)` | `Probe` | the data and model blocks run on their own: `sizes`, `prep` (the fitted plan), `features` (the width of the feature tensor), `parameters` per model, `shapes` of one batch, `notes`; `kalfa.describe.render(prepared, style, sections, probe)` turns the two into the text `kalfa describe` prints |
| `run(paths, sets=None, executor="serial", workers=None)` | `RunResult` | `record` (the directory it opened), `report` (tezgah's, `report.outputs["history"]` is the per turn table), `device` |
| `resume(run_dir, sets=None, executor="serial", workers=None)` | `RunResult` | the same, in a new record directory |
| `predict(run_dir, model=None, which=None, data=None, device=None)` | `Prediction` | `path`, `table` (a DataFrame), `model` |
| `generate(run_dir, which=None, device=None)` | `Generated` | `path`, `samples` |
| `collect_root(root, out=None)` in `kalfa.collect` | mapping | the sweep or fold table and the best point; `load_runs`, `fold_summary`, `sweep_table` are the pieces |
| `plan(paths, sets=None, record=None)` in `kalfa.sweep` | `Plan` | the points of a sweep without running any of them |

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

Interrupting a cell stops the run where it is; `resume(record)` continues from the last checkpoint into a new
directory. On a GPU box `sets=[("device", "cuda")]` picks the device, and `predict` and `generate` take `device`
the same way.

## The record directory

The `record` key gives the directory, `$datetime$` is filled once at the start of the run. A non empty directory is
never written into (an error).

| File | Contents |
|---|---|
| `resolved.yaml` | the config with its aliases and `$param$`s resolved; the source of every overridden value in a comment; runs again on its own |
| `flow.yaml` | the tezgah graph that ran: the component tables, the model blocks, the expanded flow and tezgah's resolution comments |
| `history.jsonl` | per turn the `train/`, `val/`, `test/` values, `global_step`, `lr/<optimizer>`, the rules that fired |
| `events.jsonl`, `run.json`, `stdout.txt`, `stderr.txt` | tezgah's event stream and summary |
| `checkpoints/` | `best.pt`, `last.pt` (by policy); models, optimizers, EMAs, counters, rule states, RNG |
| `final/state.pt` | always, once the run ends, with the same scope |
| `preprocessors/` | the fitted preprocessors (one file per name) and `plan.json`; `kalfa predict` reads from here |
| `predictions.parquet` | the test set: `row`, the targets (inverted), `pred_<output>` (inverted), `raw_<output>`; with `training.targets` one column per predicted field, `pred_<output>_<field>` |
| `plots/` | the outputs of the plot legos, named after the definition (`plots.roc` → `roc.png`; `architecture` writes text) |
| `samples/` | the output of `generate`: `samples.pt` (for images `grid.png` too), `samples.txt` for text; `turn_<n>.*` from `sample_writer` |
| `plugins/` | copies of the plugin modules the run imported, so predict, generate and resume work from the record |
| `device.json` | the chosen device and the device lego that picked it |
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
`--load` runs the data and model blocks for real (nothing is written): the set sizes after the filters, the widths
a fitted `one_hot` produces, the tensor slots of every column, and the parameter counts of models whose layers are
lazy until the first batch. `--section data|model|training|after|columns|wiring` narrows the output, `--wiring`
adds the implicit bindings of the compiled pipeline to the default sections. Tables are fitted to the width the
terminal reports (`COLUMNS=140 kalfa describe ...` overrides it, a narrow terminal clips the widest column with an
ellipsis); with `--save report.txt`, or whenever the output is not a terminal, nothing is clipped and the section
rules are cut to the longest line instead.

```
kalfa describe config.yaml
kalfa describe config.yaml --load
kalfa describe config.yaml --section columns
```

## Alias packs and plugins

`include: [/alias/kalfa/tabular]` (or `/alias/kalfa/vision` and `/alias/kalfa/text`, which contain tabular;
`/alias/kalfa/lazy` for large tables) brings the short names (`parquet`, `standard_scaler`, `linear_relu`, `mse`,
`rmse`, `adam`, `supervised`, `plateau`, `best`, `loss_curve` ...). The table is flat, a full URI is valid
everywhere; `kalfa ls /alias/kalfa/tabular` prints the contents, `kalfa ls tokenizer` searches names and
descriptions. Your own lego is registered with `@kalfa.lego` in a Python module next to the config and comes in
with `plugins: [module]`; `examples/alad/myexample.py` (the ALAD objectives of that example) and `tests/plugins/`
are examples. `kalfa ls alad --plugin myexample` and `kalfa docs --config cfg.yaml` import those modules before the
listing, so your own legos come with their signature and facts (`docs` prints them in a separate `Plugin legos`
section, `DOCS.md` stays the reference of what kalfa ships). Details in the Development section.

## Development

**Writing a lego.** A lego is a plain Python callable registered with `kalfa.lego`; its kind is the first segment
of the URI (`/criterion/`, `/objective/`, `/metric/`, `/layer/`, `/pre/`, `/source/`, `/split/`, `/feed/`,
`/init/`, `/optimizer/`, `/schedule/`, `/turn/`, `/trigger/`, `/checkpoint/`, `/generate/`, `/plot/`,
`/strategy/`, `/data/`, `/lego/`; the list is `kalfa.kinds.KINDS`), and `check` decides from it where the lego may be
written. The facts a lego declares: `alias` (short names), `partial` (built with its params at compile time, called
later), `state` (stateful, goes into the record), `returns` (the outputs of a flow step), `bus` (bus keys bound to
defaulted parameters), `mutates` (inputs changed in place and returned under the same name), `aliases` (the output
holds the inputs), `refs` (which params are references and of what type: `model`, `loss`, `criterion`,
`schedule`, `preprocessor`, `generate`, `field`, `column`), `uses` (`predicts` when the lego needs the prediction
model), `needs_grad` (an objective that needs gradients in the evaluation pass), `extras` (the training keys a turn
accepts), `grouped` (a preprocessor fitted once over every column that names it). The last five are kalfa's own
vocabulary, declared to cirak at import (`kalfa.kinds.FACTS` through `cirak.declare_facts`); cirak stores them and
reads none of them, and a fact kalfa never declared stays a `RegistryError`, so a misspelled one is still caught. A `/data/` lego is a run time component: `{uri: name}` as a param value, built once the data exists with
the parameters its signature names (`loader`, `prep`, `target`). The turn contract is the signature of
`/turn/kalfa/alternating` (`models, optimizers, emas, counters, composites, effects, loader, params, extra, losses,
metrics, losses_keys, metrics_keys, predicts, steps` plus the bus keys `device`, `prep`, `record`), returning
`{models, optimizers, emas, counters, metrics}`. The strategy contract: `deterministic`, `total(space)` and
`point(space, index)`, or `ask(space, mode)` and `tell(trial, value)` for a fed back strategy.

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
criterion takes `(predictions, targets, **params)`; an objective `(models, batch, **params)`; a metric is an
object with `update(...)` and `compute()`; a preprocessor an object with `apply` and optionally `fit`, `inverse`,
`partial_fit`, `rescales = True` (the metrics undo it to report in the original scale) and `dtype`; with
`grouped = True` it is fitted once over the matrix of every column that names it (`fit(matrix)`,
`apply(matrix, columns=None)`, `inverse(matrix, columns=None)`, `columns` naming the positions a slice holds), the
sklearn way, and it keeps its statistics per column; a device lego (`/device/acme/tpu`) takes no inputs, returns a `torch.device` and raises when the device is
not available (`device: {uri: /device/acme/tpu}` then selects it).

**Logging from a lego.** `logging.getLogger("kalfa.<stage>")` is the whole contract: nothing is declared, no fact,
no parameter, and the line only appears when the user asked for it with `--log`. The stage is what the third column
of a log line shows, so name it after the place in the run (`data.source`, `models`, `training.turn`,
`after.plots`), not after the module. `kalfa.std.log` has `logger_for(stage)` for that, plus `clock()` and
`since(started)` for the durations and `number(value)` for the metric formatting. INFO is what a user wants to see
without asking for detail, DEBUG is the decision behind it; nothing per batch at any level, and a log line never
computes a value the lego does not already have (guard it with `log.isEnabledFor(logging.DEBUG)` when it would).

**Plugin layout.** A module next to the config (or in a `plugins/` folder next to it) comes in with
`plugins: [module]`; the run copies the modules it imported into `<record>/plugins/`, so `predict`, `generate` and
`resume` work from the record alone. A pip package registers its legos on import and declares itself with the
`cirak.plugins` entry point (`[project.entry-points."cirak.plugins"] acme = "acme_legos"` in its `pyproject.toml`),
then `plugins: [acme_legos]` works anywhere the package is installed. Aliases declared by plugin legos are usable
without a pack.

**Adding a config and an example.** Write `configs/<nn>_<name>.yaml` with a comment header (aliases from a pack,
paths relative to the folder it runs in), add it to `TARGETS` in `tests/fixtures/regenerate_recipes.py` and
`regenerate_dumps.py`, run both scripts and read the dump (`configs/dumps/<nn>.flow.yaml`) by eye; write the
missing legos with unit tests; add the folder to `TABLE` in `tests/fixtures/sync_examples.py` (with the included
configs and the plugins it needs) and run it; write `examples/<nn>_<name>/make_data.py` on top of `kalfa.synthetic`
and the README entry (data, run, predict or generate); write `tests/test_run_<nn>.py` that runs the config on the
synthetic data with `--set` overrides that keep it small on a CPU and checks the record directory.

**Dumps and fixtures.** `configs/dumps/<name>.recipe.yaml` is the driver document (the recipe cirak opens over
`src/kalfa/templates/kalfa.yaml`), `<name>.flow.yaml` the expanded graph of `kalfa check --dump`; both are
generated, never edited; `tests/test_dump.py` checks that they are current and structurally equal to a fresh dump,
`tests/test_examples.py` that the example copies match, `tests/test_docs.py` that `DOCS.md` matches the registry
(`uv run kalfa docs --write DOCS.md` after touching a lego).

**Test rules.** `uv run pytest` must stay green; every lego has a unit test under `tests/test_std_*.py`; every
reference config has an end to end run test on synthetic data, CPU only, no network; test plugins live under
`tests/plugins/`; the reinstalled environment (`uv sync --reinstall`) runs the suite before a release.

**Versions and releases.** kalfa pins `tezgah>=` and `cirak>=` in `pyproject.toml`; bump the version, regenerate
the dumps (`tests/fixtures/regenerate_recipes.py` and `regenerate_dumps.py`, whose headers carry the version), run
the suite against the reinstalled environment (`uv sync --reinstall`), then build (`uv build`). kalfa is 0.2.3 and
needs tezgah 0.2.0 or later and cirak 0.2.2 or later.
