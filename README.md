# kalfa

kalfa trains PyTorch models from a YAML config or from a few lines of Python. A config names the data, the models, the losses, the metrics, the optimizers and the training. kalfa checks it before anything runs, trains it, and writes a record directory that the command line, the Python API, a browser board and a terminal interface all read.

## Why kalfa

* **Check before you run.** `kalfa check` reads the config against the signatures of the classes it names and lists every problem at once: unknown keys, missing params, references that point nowhere, rule paths that cannot be patched. Nothing is built until the config is right.
* **Rules patch a running config.** A rule watches the series of every turn and changes the plan while the run goes on: lower a learning rate on a plateau, swap the loss an optimizer minimizes, unfreeze a backbone, write samples, stop. Each patch is written to the record, and a resumed run replays it.
* **Components refer to each other.** `"@models.net"`, `"@losses.mse"` and `"@data.class_weights"` wire the pieces together, so a GAN, a VAE, a diffusion model or a distillation run is a config and not a fork of a training script.
* **Models are plain torch modules.** Any importable class works through its import path. There is no layer language to learn and no base class to inherit.
* **One record directory for every tool.** A run writes its plan, history, checkpoints, predictions and plots into one directory, and every tool reads that directory and nothing else, so a run from a cluster reads the same as a run from a laptop.

## Install

```sh
uv sync
```

or `pip install .` from a clone. kalfa needs Python 3.11 or newer. One install brings everything: the board, the terminal interface, sweeps with optuna, ONNX export and the TensorBoard logger; there are no optional extras. A run picks its device itself (CUDA, then MPS, then the CPU) unless the config names one.

## A first run

```yaml
data:
  use: table
  file: housing.parquet
  split: [0.7, 0.15, 0.15]
  features: {"x*": standard_scaler}
  targets: {price: standard_scaler}
  batch: 128
models:
  net: {use: mlp, hidden: [64], out: 1}
losses:
  mse: mse
metrics:
  rmse: rmse
optimizers:
  main: {use: adam, lr: 1.0e-3, over: "@models.net", minimizes: "@losses.mse"}
train:
  epochs: 20
  checkpoint: {watch: val/rmse}
rules:
  slower:
    when: {use: plateau, watch: val/rmse, patience: 3}
    set: {optimizers.main.lr: {times: 0.1}}
record: runs/minimal_$datetime$
```

The table is split into train, valid and test sets. Every column matching `x*` is a feature scaled on the train rows, and `price` is the target. Every `{use: ...}` mapping is a call: a short name from the catalog or an import path such as `torch.nn.HuberLoss`. The metric `rmse` is computed in the original units of `price`, because the scaler of the target can be inverted. When `val/rmse` stops improving for three turns, the rule multiplies the learning rate by 0.1, once.

```sh
uv run kalfa check config.yaml
uv run kalfa run config.yaml
uv run kalfa board runs --open
```

`check` prints `ok` or every problem of the config. `run` trains, prints one line per turn and ends with the record directory. `board` opens the records under `runs` in the browser and follows the running ones live.

## What a run leaves behind

```text
runs/minimal_20261110_143000/
  manifest.json          status, times, params, device, git commit, why the run stopped
  resolved.yaml          the merged config the run used
  plan.json              every node with its class, params and slot keys, as check saw it
  history.jsonl          one line per turn: every series, the learning rates, the rules that fired
  updates.jsonl          one line per update: loss, learning rate, gradient norm
  events.jsonl           starts, patches, checkpoints, stops and errors
  model_summary.json     parameter counts, module trees and a diagram of every model
  data.json              what the data found and a flowchart from the file to the batch
  fitted/                preprocessors and classes, so the record opens without fitting again
  checkpoints/           last.pt, best.pt and final.pt
  predictions.parquet    the test set with the predictions in original units
  plots/                 the images of the plots section
  samples/               samples from a generate section
```

`kalfa.open(path)` reads it from Python, and `kalfa run --from` continues it from its last checkpoint into a new directory.

## Commands

| command | what it does |
|---|---|
| `kalfa check CONFIG` | every problem of a config; nothing runs |
| `kalfa run CONFIG` | trains into a record directory; `--from RECORD` continues a record |
| `kalfa predict RECORD` | predicts the test set, or a file given with `--data`, in original units |
| `kalfa generate RECORD` | draws samples from the `generate` section of a GAN, a VAE, a DDPM or a language model |
| `kalfa export RECORD` | writes a model as a state_dict or an ONNX file |
| `kalfa plots RECORD` | draws the plots of a record again |
| `kalfa sweep CONFIG` | runs the points of a sweep (grid, random search or optuna) in parallel, or writes one job per point |
| `kalfa summarize PATHS` | ranks records and sweeps; `--html` writes one self contained report |
| `kalfa stop RECORD` | asks a running record, or every point of a sweep, to stop after its turn |
| `kalfa legos [NAME]` | lists the short names by category, or describes one |
| `kalfa board ROOT` | the browser board over the records under a root |
| `kalfa tui ROOT` | the terminal interface |

Every command that reads a config takes `--set PATH=VALUE` to override a leaf and `-p NAME=VALUE` to set a `$name$` param of the config, so `uv run kalfa run config.yaml --set train.epochs=3` is a short trial without editing the file.

## Python

```python
import kalfa

run = kalfa.Run(
    data=kalfa.call("table", file="housing.parquet", split=[0.7, 0.15, 0.15],
                    features={"x*": "standard_scaler"}, targets={"price": "standard_scaler"}),
    models={"net": kalfa.call("mlp", hidden=[64], out=1)},
    losses={"mse": "mse"},
    metrics={"rmse": "rmse"},
    optimizers={"main": kalfa.call("adam", lr=1.0e-3, over="@models.net", minimizes="@losses.mse")},
    train=kalfa.Train(epochs=20, checkpoint={"watch": "val/rmse"}),
    rules={"slower": kalfa.Rule(when=kalfa.call("plateau", watch="val/rmse", patience=3),
                                set={"optimizers.main.lr": {"times": 0.1}})},
)
print(run.check().text())
result = run.start(record="runs/from_python")
record = kalfa.open(result.record)
print(record.summary["best"])
print(record.history[["turn", "val/rmse"]])
```

`kalfa.open` reads any record directory, finished or still running, `record.stats()` scores its predictions the way the board does, and `kalfa.summarize` ranks many records.

## The board

`uv run kalfa board runs --open` serves the records under `runs` at http://127.0.0.1:8765/ and follows the running ones live. Besides the list of runs, a compare page draws the curves of several records over each other with the params and the config lines that differ, and a sweep page ranks the points. Every record opens in nine tabs:

* **overview**: progress, best and last values, the rules that fired, why the run stopped and a stop button.
* **curves**: every series. Any chart opens large with its own settings (scales, ranges, line style, smoothing, size) and downloads as PNG or SVG.
* **model**: a diagram of every model from one forward pass, with groups that fold, the module tree and the output shapes.
* **data**: a flowchart from the file to the batch, with the rows of every set.
* **predictions**: every prediction column read as its task (regression, binary, multiclass or embedding) with the measures of that task, an overview when there are several targets, a threshold slider for binary tasks and an image gallery when the rows are images.
* **plots**, **config**, **files** and **log**.

## The terminal interface

```sh
uv run kalfa tui runs --config config.yaml
```

Four tabs:

* **records**: the records under `runs` with their summaries and sparklines, and keys to predict, plot, stop or open the board.
* **legos**: the catalog as a tree of categories. Each page says where the name goes in a config and shows its params with notes, YAML examples and the example configs that use it.
* **config**: checks, plans, runs or sweeps a config.
* **guide**: the same config as a document, section by section in color, with every entry explained.

## Examples

`examples/` holds 18 configs, each with a small data maker and a recipe:

| folder | shows |
|---|---|
| `minimal` | the smallest config that trains |
| `01_mlp_regression` | a rule chain that moves the loss from mse to huber, l1 and a plain function |
| `02_mlp_classification` | numeric and categorical columns, class weights from the data and macro f1 |
| `03_timeseries_window` | windows over the hourly load of two sites, split in time within each site |
| `04_cnn_images` | a frozen backbone that a rule unfreezes, with two optimizer groups |
| `05_autoencoder` | reconstructions, latent codes and samples |
| `06_vae` | a VAE whose KL weight warms up while rules change the reconstruction term |
| `07_wgan_gp` | a conditional WGAN GP with five critic steps per generator step, an EMA generator and FID |
| `08_ddpm` | a DDPM with a small UNet under a cosine beta schedule |
| `09_simclr` | self supervised SimCLR with two augmented views per image |
| `10_char_lm` | a character language model with a small causal transformer |
| `11_kfold` | cross validation with one run per fold and the fold summary |
| `12_resume` | continuing a record from its checkpoint |
| `13_distillation` | a teacher trained first, then a student that learns from it |
| `14_sweep` | one sweep three ways: grid, random search and optuna |
| `15_multi_target` | one network with three heads and a weighted sum of three losses |
| `16_mdmm` | MDMM holding the mean absolute error at a target while the mse falls |
| `alad` | ALAD anomaly detection with five models and two adversarial objectives |

```sh
uv run python examples/demo.py
uv run kalfa board runs/demo --open
```

The first command makes the data of every example and plays every recipe in a few turns under `runs/demo`: runs, a resumed run, sweeps, predictions, samples and exports. The second opens the board on them. [examples/README.md](examples/README.md) says more about each one.

## Your own pieces

Any importable class or function can be called by its import path, and a module next to the config is found without installing it. `kalfa.lego` gives your piece a short name with notes, so `kalfa legos` and the terminal interface describe it like the built in ones:

```python
import kalfa
from torch import nn


@kalfa.lego("two_heads", slot="model", doc="a shared body with a mean head and a spread head",
            params={"width": "the width of the shared body"})
class TwoHeads(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.body = nn.Sequential(nn.LazyLinear(width), nn.ReLU())
        self.mean = nn.Linear(width, 1)
        self.spread = nn.Linear(width, 1)

    def forward(self, x):
        hidden = self.body(x)
        return self.mean(hidden), self.spread(hidden)
```

```yaml
plugins: pieces.py
models:
  net: {use: two_heads, width: 64, outputs: [mean, spread]}
```

Objectives, metrics, steps, triggers, actions, loggers and data classes follow small protocols; [DOCS.md](DOCS.md#writing-your-own-pieces) shows each one.

## Learn more

* [DOCS.md](DOCS.md) is the full reference: the grammar, every section, the runtime, the record directory, every command, the Python API, the board, the terminal interface and how to write your own pieces.
* [examples/README.md](examples/README.md) lists the examples and what each one shows.
* `uv run kalfa legos NAME` describes one short name, and `uv run kalfa legos --docs` prints the whole catalog as markdown.

## Development

```sh
uv run pytest
```

runs about 1350 tests in about a minute on a laptop; the tests that train use small synthetic data and a few turns on the CPU. The package lives in `src/kalfa`: `core` (config, plan, build, loop, rules and the record writer), `std` (the built in pieces), `records` (the readers every tool shares), `commands` (the command line), `board` and `tui`. [PLAN.md](PLAN.md) holds the design contract and the notes of every phase, in Turkish.

## License

MIT
