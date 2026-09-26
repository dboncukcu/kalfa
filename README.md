# kalfa

**A YAML front end for PyTorch training.** One config file says where the rows come from, how columns become
tensors, how the model is wired, what is minimized, when to stop and what to draw. kalfa compiles that file into a
pipeline and runs it; everything the run produced lands in one record directory that replays, resumes, sweeps and
browses.

```
┌─ config.yaml ───────────────────────┐      ┌─ runs/housing_20260915_120341/ ──────┐
│ data      rows in, tensors out      │      │ history.jsonl        one line a turn │
│ model     wires and layers          │      │ checkpoints/         best.pt last.pt │
│ losses    what is minimized         │ ───▶ │ predictions.parquet  original scale  │
│ metrics   what is reported          │      │ plots/               the figures     │
│ training  how long, when to stop    │      │ fitted/              the scalers     │
│ plots     what to draw              │      │ resolved.yaml        runs again      │
│ record    where it all goes         │      │ flow.yaml          what actually ran │
└─────────────────────────────────────┘      └──────────────────────────────────────┘
```

No Python is needed to train a model. Python is how you add a piece kalfa does not already have.

| I want to | Go to |
|---|---|
| see it work | [60 seconds](#60-seconds) |
| understand the machine | [How it works](#how-it-works) |
| write a config | [The config](#the-config) · [`CONFIG.md`](CONFIG.md) |
| copy a working setup | [Recipes](#recipes) · [`examples/`](examples/) |
| watch a run | [The board](#the-board) |
| find a lego | [`kalfa ls`](#commands) · [`DOCS.md`](DOCS.md) |
| drive it from Python | [Python API](#python-api) |
| add my own piece | [Your own legos](#your-own-legos) |

## Install

Python 3.13 or newer.

```bash
pip install kalfa          # or: uv add kalfa
kalfa --version            # kalfa, cirak, tezgah, torch and python, the line a bug report needs
```

torch, pandas, pyarrow, scikit-learn, torchmetrics, matplotlib, pillow, scipy, ruamel.yaml, tqdm, optuna and the
two layers below kalfa (cirak, the recipe compiler, and tezgah, the pipeline runner) come with it. There are no
optional extras. Three things are looked up at run time and skipped with a log line when absent: `seaborn` (the
`pairplot`, `violin` and `kde` plots), `torchview` with the graphviz `dot` binary (the `torchview` plot) and
`onnx` (`kalfa export --format onnx`).

To get the reference configs and the test suite:

```bash
git clone https://github.com/dboncukcu/kalfa.git && cd kalfa
uv sync
uv run kalfa --help
uv run python -m pytest -m "not slow and not subprocess"   # the surface and lego layers, seconds
```

## 60 seconds

```bash
cd examples/01_mlp_regression
uv run python make_data.py       # writes housing.parquet (x0..x7, price) and new.parquet
```

A complete config for that data, nothing left out:

```yaml
include: [/alias/kalfa/tabular]           # short names: parquet, standard_scaler, adam, mse, rmse ...

params:                                    # $name$ anywhere below; -p lr=1e-4 overrides from the shell
  epochs: 100
  lr: 1.0e-3
  seed: 7

seed: $seed$

data:
  source: {uri: parquet, params: {path: housing.parquet}}
  split:  {ratios: [0.7, 0.15, 0.15], seed: $seed$}
  batch:  128
  preprocessors:                           # named definitions, fitted on the train set only
    std_scaler: {uri: standard_scaler}
    target_std: {uri: standard_scaler}
  fields:                                  # glob -> role and chain; a column not written never reaches the model
    "x*":  {preprocessors: [std_scaler]}
    price: {target: true, preprocessors: [target_std]}
  feed: table                              # one sample per row

model:                                     # the single model shorthand: no models: block, no optimizers: section
  optimizer: {uri: adam, params: {lr: $lr$}}
  inputs: [x]
  outputs: [y]
  nodes:
    - {uri: linear_relu, params: {out_features: 128}}
    - {uri: linear_relu, params: {out_features: 128}}
    - {uri: linear, params: {out_features: 1}}

losses:                                    # defined here, used by name below
  loss_mse: {uri: mse}

metrics:
  rmse: {uri: rmse}
  mae:  {uri: mae}

training:
  turn: supervised
  loss: loss_mse
  epochs: $epochs$
  stop:       [{uri: plateau, params: {monitor: val/rmse, patience: 6}}]
  checkpoint: {uri: best, params: {monitor: val/rmse}}
  report: best

plots:
  loss_curve:   {uri: loss_curve}
  pred_vs_true: {uri: pred_vs_true}

record: runs/housing_$datetime$
```

`nodes` as a list is a chain. Written as a mapping it is a graph, keyed by the wire each node writes, and that is
where a second input, a skip connection or a second output goes; the rest of the config stays as it is:

```yaml
model:
  optimizer: {uri: adam, params: {lr: $lr$}}
  inputs: [x]
  outputs: [y]
  nodes:
    h1:  {uri: linear_relu, params: {out_features: 128}, inputs: [x]}
    h2:  {uri: linear_relu, params: {out_features: 128}, inputs: [h1]}
    sum: {uri: add, inputs: [h1, h2]}                       # a residual connection: two wires into one node
    y:   {uri: linear, params: {out_features: 1}, inputs: [sum]}
```

```bash
uv run kalfa check config.yaml                      # unknown keys, unresolvable names, the set table; nothing runs
uv run kalfa describe config.yaml                   # the config as an analysis, see below
uv run kalfa run config.yaml --set record=runs/01   # trains
uv run kalfa predict runs/01 --data new.parquet     # the same pipeline over new rows
uv run kalfa board runs                             # a page over every record under runs/
```

What `runs/01/` holds afterwards:

```
runs/01/
├── manifest.json          kind, name, config paths, params, version, contract hash
├── resolved.yaml          aliases and $param$s resolved, with the source of every override in a comment
├── contract.yaml          the wiring this run was compiled by
├── flow.yaml              the tezgah graph that ran
├── history.jsonl          {"turn": 3, "train/loss_mse": 0.41, "val/rmse": 0.9, "lr/model": 0.001, "rules": []}
├── steps.jsonl            one line per update: loss, lr, grad_norm
├── checkpoints/           best.pt, last.pt  (models, optimizers, EMAs, counters, rule states, RNG)
├── final/state.pt         always, once the run ends
├── fitted/                preprocessors/, frames/, calibrate/  (predict and resume read from here)
├── predictions.parquet    the test set: row, price, pred_y, raw_y, in the original units
├── plots/                 loss_curve.png, pred_vs_true.png
├── data.json              rows and columns at every stage of the data block
├── architecture.json      the model graph with traced shapes, for the board
├── device.json, git.json, host.json, events.jsonl, run.json, stdout.txt, stderr.txt
└── plugins/               copies of the modules the run imported
```

`resolved.yaml` runs again on its own, so a record is a reproducible config as well as a result.

## How it works

### Three layers

```
  your config              kalfa                    cirak                   tezgah
  ───────────              ─────                    ─────                   ──────
  config.yaml   ┐          merge the layers         build every lego        run the nodes
  include:      │   ───▶   resolve aliases   ───▶   by its URI       ───▶   in bus order      ───▶  record/
  --set / -p    ┘          check everything         expand the blocks       trace and time them
                           fill the contract
```

kalfa owns the surface (the sections and their keys) and the **contract**, a fixed mapping from those sections
onto five flow blocks. It has no defaults of its own: a key is required, or means "none" when unwritten, or is
derived from the shape, or falls back to the wrapped library's default. `kalfa contract` prints the mapping,
`kalfa check --dump` prints the graph it produced.

### Five blocks

```
  data                 models            optimizers       training (a loop)      after
  ────                 ──────            ──────────       ─────────────────      ─────
  source               builder per       torch optimizer  ┌ effects (rules)      final/state.pt
  transform            model             schedule         │ turn (the update)    report model choice
  split                init, weights     param groups     │ evaluate per set     calibrate
  transform per set    trainable         the bundle       │ merge                predictions.parquet
  frame (fit, apply)   ema clones                         │ rules                figures + plots
  prep (fit, apply)    composites                         │ stop                 generate ─→ samples/
  feed                                                    │ checkpoint
  loader per set                                          └ history line
```

Each of those becomes a node in the tezgah graph, and every node calls one **lego**.

### Legos

A lego is a stateless callable registered under a URI. The first segment is its **kind**, and the kind alone
decides which section it may be written in, so `mae` is one lego used as a loss under `losses` and as a metric
under `metrics`, and a lego in the wrong section is a `check` error.

```
/criterion/kalfa/mse        /pre/sklearn/standard_scaler        /trigger/kalfa/plateau
 ^kind     ^pack ^name
```

| Kind | Written in | Ships | For example |
|---|---|---|---|
| `source` | `data.source` | 7 | `parquet`, `csv`, `image_folder`, `text_lines` |
| `transform` | `data.transform` | 5 | `filter`, `derive`, `rename`, `astype`, `drop` |
| `split` | `data.split` | 5 | `random_split`, `sequential`, `given`, `kfold` |
| `frame` | `data.frame` | 2 | `target_encoding`, `group_statistic` |
| `pre` | `data.preprocessors` | 29 | `standard_scaler`, `one_hot`, `logit`, `resize`, `char_tokenizer` |
| `feed` | `data.feed` | 3 | `table`, `window`, `next_token` |
| `layer` | model `nodes` | 125 | `linear_relu`, `conv2d`, `embedding`, `gru`, `select`, `add`, `multipliers` |
| `init` | model `init` | 5 | `normal`, `xavier`, `kaiming`, `zeros` |
| `criterion` | `losses`, `metrics` | 7 | `mse`, `huber`, `cross_entropy`, `bce_logits` |
| `objective` | `losses` | 8 | `vae`, `wgan_gp_d`, `ddpm`, `ntxent`, `distill`, `mdmm` |
| `metric` | `metrics` | 9 | `rmse`, `f1`, `auroc`, `fid`, `perplexity` |
| `optimizer` | `optimizers` | 3 | `adam`, `adamw`, `sgd` |
| `schedule` | optimizer `schedule` | 4 | `warmup_cosine`, `step_decay`, `linear_betas` |
| `turn` | `training.turn` | 1 | `alternating` (`supervised` is its single optimizer alias) |
| `trigger` | `training.stop`, rule `when` | 5 | `plateau`, `metric_below`, `after_epoch`, `time_budget` |
| `checkpoint` | `training.checkpoint` | 3 | `best`, `last`, `snapshot` |
| `calibrate` | `calibrate` | 1 | `threshold` |
| `generate` | `generate` | 3 | `gan_sampler`, `ddpm_sampler`, `lm_sampler` |
| `plot` | `plots` | 26 | `loss_curve`, `pred_histogram`, `confusion_matrix`, `correlation_heatmap` |
| `strategy` | `sweep.strategy` | 4 | `grid`, `random`, `sobol`, `optuna` |
| `device` | `device` | 4 | `auto`, `cpu`, `cuda`, `mps` |
| `rng` | `rng` | 3 | `derived`, `indexed`, `global` |
| `export` | `kalfa export --format` | 3 | `onnx`, `pt2`, `state_dict` |
| `data` | any param value | 5 | `class_weights`, `vocab_size`, `feature_width`, `feature_index` |

`kalfa ls tokenizer` searches names and descriptions, `kalfa ls /alias/kalfa/tabular` prints a pack,
[`DOCS.md`](DOCS.md) is the generated reference with every signature and fact.

### Four short forms, and only four

```yaml
turn: supervised                                  # a plain string is a call without params
split: {ratios: [0.8, 0.1, 0.1]}                  # a split without uri is the random split
batch: 128                                        # a scalar batch is {size: 128}
model: {inputs: ..., outputs: ..., nodes: ...}    # one model, no models: block
```

Everything else is `{uri: name, params: {...}}`. A full URI is valid wherever a short name is.

### Layers

A config is the union of layers, bottom to top, with leaf overriding between them and strict merging inside one
(the same key written twice in one file is an error):

```
  4. --set path=value   -p name=value        the command line
  3. the config given on the command line
  2. the includes, in list order             each with its own includes below it
  1. the alias packs                         /alias/kalfa/base, tabular, vision, text, lazy_tabular
```

That is the whole mechanism behind variants: a k fold, a sweep and the third idea you had this afternoon are the
same config with a leaf or two replaced.

```yaml
# 11_kfold_cv/config.yaml, in full
include: [../01_mlp_regression/config.yaml]
params: {fold: 0}
data:   {split: {uri: kfold, params: {k: 5, fold: $fold$, val: 0.15, seed: $seed$}}}
record: runs/cv_housing_$fold$
```

```bash
for i in 0 1 2 3 4; do kalfa run config.yaml -p fold=$i; done
kalfa collect runs/cv_housing_*        # mean and deviation over the folds, runs/reports/cv.json and cv.md
```

`kalfa check --layers` prints the layer tree and which file overrode which leaf; `resolved.yaml` carries the
source of every overridden value in a comment (`epochs: 5  # --set; overrides config.yaml:11`).

## The config

The whole surface, in one block. Everything is defined at the top and used by name below; there are no roles, a
loss is bound to an optimizer and the optimizer minimizes it over the parameters of the models that picked it.

```yaml
include: []      # alias packs, YAML fragments, other configs
plugins: []      # Python modules next to the config; @kalfa.lego registrations
params: {}       # $name$ placeholders, overridden with -p name=value
seed: 7          # python, numpy and torch
device: auto     # a device lego: auto | cpu | cuda | mps | {uri: cuda, params: {index: 1}}
rng: derived     # how model seeds derive from seed: derived | indexed | global

data: {...}      # source, transform, frame, split, batch, mask, preprocessors, drop, fields, feed
model: {...}     # templates, models  (or the single model shorthand)
metrics: {}      # reported, original scale
losses: {}       # minimized, model scale         (at least one)
optimizers: {}   # who minimizes what over which parameters
training: {...}  # turn, epochs or steps, loss, stop, rules, checkpoint, report
calibrate: {}    # fitted after training: thresholds, priors
generate: {}     # a sampler, run at the end and by kalfa generate
plots: {}        # drawn at the end
figures: {}      # format, width, height, dpi, style for all of them
sweep: {}        # strategy, space, objective, record
record: runs/x   # required
```

An unknown key is an error, not a warning. `reference.yaml` marks every key as required, empty, derived or
library default.

### `data`

The order is fixed and independent of the writing order:

```
  source                     a parquet, a csv, a folder of images, a text file
    ↓
  transform                  filter, derive, rename: the entries without sets
    ↓
  split                      train  ·  valid  ·  test
    ↓
  transform per set          the entries carrying sets: [train]
    ↓
  frame                      fitted on train, mapped onto every set
    ↓
  drop + fields + preprocessors    fitted on train, applied per set; this is what fitted/ holds
    ↓
  feed                       rows become tensors: table, window, next_token
    ↓
  loader                     one per set
```

```yaml
data:
  source: {uri: parquet, params: {path: tree.parquet}}   # columns: [...] leaves the rest of a wide file on disk

  transform:                                        # frame to frame, in list order
    - "(y > -4) & (y < 4)"                          # a bare string is a filter query
    - {uri: derive, params: {column: ratio, expr: "m1 / m2"}}
    - {uri: filter, params: {query: "ratio > 0"}, sets: [train]}    # with sets: after the split

  frame:                                            # fitted on train, mapped onto every set
    - {uri: target_encoding, params: {column: cat_a, target: y, smoothing: 1.0}}

  split: {ratios: [0.7, 0.15, 0.15], seed: 7}       # always three, 0.0 means "no set"
  batch: {size: 128, eval_size: 512, drop_last: auto, workers: 4, balanced: false}
  mask:  "m1 < 0"                                   # kept in the frame, left out of the batch

  preprocessors:                                    # a definition may carry sets: [train] too (augmentation)
    x_scaler: {uri: standard_scaler}                # one object over every column that names it
    onehot:   {uri: one_hot}
    y_scaler: {uri: standard_scaler}                # a second definition keeps a second set of statistics

  drop: [event_id]                                  # never reach a glob

  fields:                                           # globs: ? one character, * zero or more
    "m*":     {preprocessors: [x_scaler]}
    "cat_*":  {preprocessors: [onehot]}
    y:        {target: true, preprocessors: [y_scaler]}

  spectators: [run_id, "weight_*"]                  # carried raw to the plots and predictions.parquet, never to the model

  feed: table                                       # table | window | next_token
```

| Key | Choices |
|---|---|
| `source` | `parquet`, `csv` (a table), `image_folder`, `text_lines` (a Dataset), `prepared` |
| `split` | the bare `{ratios, seed}` is `random_split`; also `sequential` (with `group`), `given`, `kfold` |
| `feed` | `table` (features join into one tensor `x`, targets keep their names), `window` (`{size, horizon, context, group}`), `next_token` (`{seq_len}`) |
| `fields` | a column that matches no glob never reaches the model; the most specific glob wins, a tie is an error |
| `spectators` | columns that ride along unpreprocessed; anything neither field, spectator nor lego reference is read and discarded, and `check` says so |
| `batch` | unwritten means no batching at all: one batch per set, one step per turn |

A scaler is fitted on the **train set only**, lands in `fitted/preprocessors/`, and is inverted in the metrics
and in `predictions.parquet`, so every reported number is in the original units. `kalfa predict` reads the same
fitted objects back from the record.

Big tables: `include: [/alias/kalfa/lazy_tabular]` is the tabular pack with `parquet` and `csv` rebound to chunked
stream sources, so a config turns lazy by changing its `include` line and nothing else. Filters run per chunk, the
split becomes `sequential` or `given`, shuffling goes through `batch.buffer`, and what a stream cannot do
(`kfold`, `window`, `mask`, a random split) is a `check` error, not a surprise.

### `model`

`nodes` as a **list** is a chain, each node taking the output of the one before it. `nodes` as a **mapping** is a
graph keyed by the wire each node writes.

```yaml
model:                                              # a catalogue; these five would not share one config
  templates:                                        # a piece with variables, copied at every use
    cond_embed:
      variables: {n_classes: {required: true}}
      nodes: [{uri: embedding, params: {num: $n_classes$, dim: 64}}]

  models:
    backbone:
      optimizer: main                               # a name in optimizers, or an inline {uri, params}
      trainable: false                              # requires_grad off AND eval mode; a rule can open it
      inputs: [image]
      outputs: [feature]
      nodes: [{uri: timm_backbone, params: {name: resnet50, pretrained: true}}]

    head:
      optimizer: main
      init: {weights: {uri: kaiming}, bias: {uri: zeros}}   # roles: weights, bias, scale, each {uri, params}; or by pattern
      inputs: [feature]
      outputs: [logits]
      nodes: [{uri: linear, params: {out_features: 37}}]

    teacher:
      trainable: false
      weights: {run: runs/teacher_01, model: net, which: best}   # loaded wholesale from another record
      inputs: [image]
      outputs: [logits]
      nodes: [{uri: timm_backbone, params: {name: resnet50}}, {uri: linear, params: {out_features: 10}}]

    generator:
      optimizer: g
      ema: {decay: 0.999}                           # selected later as generator.ema
      inputs: [z, label]
      outputs: [image]
      nodes:                                        # a mapping is a graph; the key is the wire
        y:     {template: cond_embed, params: {n_classes: 10}, inputs: [label]}
        zy:    {uri: concat, params: {dim: 1}, inputs: [z, y]}
        image: {uri: dcgan_generator, params: {channels: 64}, inputs: [zy]}

    corrector:                                      # the output is a correction of two input columns
      optimizer: main
      inputs: [x]
      outputs: [y]
      nodes:
        base:  {uri: select, params: {index: {uri: feature_index, params: {columns: [m1, m2]}}}, inputs: [x]}
        delta: {uri: mlp, params: {widths: [128, 128], out_features: 2}, inputs: [x]}
        y:     {uri: add, inputs: [base, delta]}

    lambdas:                                        # the multipliers of the mdmm loss below, one per constraint
      optimizer: main
      inputs: [x]
      outputs: [lmbda]
      nodes: [{uri: multipliers, params: {names: $constraints$}}]

    classifier:                                     # a composite: real models as nodes, no parameters of its own
      inputs: [image]
      outputs: [logits]
      nodes:
        feature: {model: backbone, inputs: [image]}
        logits:  {model: head, inputs: [feature]}
```

| Key | Means |
|---|---|
| `optimizer` | which optimizer trains it; models sharing a name share one optimizer object |
| `init` | per role or by parameter name pattern, each a `{uri, params}` call; unwritten roles keep the torch default |
| `weights` | load another record's weights after the build (teacher, fine tune, warm start) |
| `ema` | a moving average copy, usable everywhere as `<name>.ema` |
| `trainable: false` | frozen and in eval mode, so BatchNorm statistics freeze too |
| `{model: name}` node | a composite; it has no parameters, no checkpoint entry, and its parts keep their own modes |

A layer param can be a run time value: `{uri: vocab_size}` for an `embedding`'s `num`, `{uri: feature_width}` for
a `layer_norm`, `{uri: feature_index, params: {columns: [m1, m2]}}` for a `select`. It is built once the fitted
preprocessors and the train loader exist.

Wires meet in the arithmetic nodes: `add` and `multiply` take any number of wires, `subtract` and `divide` two,
`negate` one, and shapes broadcast. `select` takes positions out of a wire, by index or by column name through
`feature_index`, which follows the fitted plan (a `one_hot` that widened a column is already counted); with `add`
it is a long range skip connection, the output as a correction of part of the input, as `corrector` above.
`multipliers` is a model of one node that carries the Lagrange multipliers of an `mdmm` loss, so the optimizer
updates them and the checkpoint keeps them (the `losses` section below).

`timm_backbone` and `dcgan_generator` above are not std legos: they come from the plugin modules of
`examples/04_cnn_images` and `examples/07_wgan_gp`. A plugin lego is written exactly like a std one, which is the
point. The 125 layer legos kalfa ships mirror `torch.nn` under `/layer/torch/` and add its own under
`/layer/kalfa/`.

### `losses`, `metrics`, `optimizers`

```yaml
metrics:
  rmse:     {uri: rmse}
  f1:       {uri: f1}
  fid:      {uri: fid, params: {model: generator.ema, latent: 128}, every: 5, sets: [valid]}

losses:
  ce:       {uri: cross_entropy, params: {weight: {uri: class_weights}}}   # a run time component as a param
  recon:    {uri: mse, output: x_hat, target: input}      # a named wire, and the model's own input as target
  total:    {uri: weighted_sum, params: {terms: {ce: 1.0, recon: 0.1}}}    # the keys are loss definition names
  held:     {uri: mdmm, params: {primary: ce, multipliers: lambdas,     # ce under the constraint recon = 0.05
                                 constraints: $constraints$}}       # {recon: {epsilon: 0.05, lmbda_init: -1.0}}
  vae_loss:
    uri: vae
    params:
      encoder: encoder                                    # refs of type model, resolved by name
      decoder: decoder
      recon: mse                                          # a ref of type criterion
      kl_schedule: {uri: linear_warmup, params: {start: 0.0, end: 1.0, steps: 2000}}

optimizers:
  main:
    uri: adamw
    params: {lr: 1.0e-3, groups: [{name: backbone, match: "backbone.*", lr: 1.0e-5},
                                  {name: lambdas, match: "lambdas.*", lr: -2.0e-4}]}   # the multipliers climb
    loss: ce                                              # the name of what it minimizes
    schedule: {uri: warmup_cosine, params: {warmup: 500, total: 20000}}
```

Three kinds go in these sections:

| Kind | Signature | Written in |
|---|---|---|
| `criterion` | `(predictions, targets, **params)` | `losses` **and** `metrics` |
| `objective` | `(models, batch, **params)`, plus `step`, `epoch`, `rng`, `scaler`, `losses` if named | `losses` only |
| `metric` | stateful `update(...)` / `compute()`, arguments bound by name | `metrics` only |

Definition level keys sit beside `uri` and `params`, never inside them: `sets: [train]`, `every: 5`,
`output: x_hat` (which wire of a multi output model), `target: input` (a field, a list, a glob, or the model's own
input).

**The scale rule follows the section, not the kind.** `losses` are computed in the model scale (the transformed
target), `metrics` in the original scale: the adapter runs prediction and target back through the rescaling
preprocessors. So `mae` written twice, once under each section, reports two different and both correct numbers.

**The set rule.** Every loss and metric is computed on every existing set every turn. `valid` steers the run
(checkpoint, stop, rules, `report: best`), `test` decides nothing and is only written, so the `test/` column of
every turn is an honest read of a model that `test` never chose.

### `training`

```yaml
training:
  turn: supervised                 # alias for alternating with one optimizer
  # turn: {uri: alternating, params: {order: [d, g], steps: {d: 5, g: 1}, fresh_batch: true}}

  epochs: 100                      # or: steps: {total: 20000, turn: 200}
  loss: loss_mse                   # with a single optimizer
  predicts: classifier             # derived when there is one trained model
  targets: {y_hat: "y_*", z_hat: z}   # which wire predicts which target fields

  amp: true                        # the turn's extra keys: amp, grad_clip, accumulate
  grad_clip: 1.0
  accumulate: 2

  stop: [{uri: plateau, params: {monitor: val/rmse, patience: 6}}]
  checkpoint: {uri: best, params: {monitor: val/rmse}}
  report: best                     # best | last: which weights the predictions and plots use

  rules:                           # evaluated at the end of a turn, effective in the next
    - name: unfreeze
      when: {uri: after_epoch, params: {at: 5}}
      set:  {backbone.trainable: true}
    - name: to_huber
      when: {uri: plateau, params: {monitor: val/rmse, patience: 3}}
      set:  {loss: loss_huber}
    - name: to_mae
      after: to_huber                                            # not evaluated until to_huber fires
      when: {uri: metric_below, params: {monitor: train/loss_huber, value: 0.01}}
      set:  {loss: loss_mae}
```

Rules are the piece that usually lives in a hand written training loop. A rule that fired stays fired; `after:`
chains them; among rules writing the same key the last one wins; `set:` can retarget the loss, open a frozen
model, change a loss parameter or an optimizer learning rate: `main.lr` is the rate of the default group,
`main.backbone.lr` that of a named group, `main.*.lr` every group, and a relative value (`{times: 0.5}`) applies to
each group's own rate. Every effect in force is written to `history.jsonl` as `effect/<target>`, so a curve can
be read against what the run was doing at the time.

`alternating` walks the optimizers in `order` every step; each computes its own loss `steps` times, backpropagates
and updates only its own models, zeroing only its own gradients. That one turn lego covers supervised training,
autoencoders and VAEs, WGAN GP, ALAD, DDPM, SimCLR, language models, distillation and multi loss regimes.
Reinforcement learning, MAML's inner loop and regimes drawing from two loaders in one turn need a turn lego of
their own (`CONFIG.md` section 9).

### After training

```yaml
calibrate:
  flag: {uri: threshold, params: {set: valid, quantile: 0.95, output: score}}

generate:
  uri: lm_sampler
  params: {model: gpt, prompt: "ROMEO:", max_new_tokens: 500, temperature: 0.9}

figures: {format: pdf, width: 6, height: 4, dpi: 200, style: kalfa}   # one look for every plot

plots:
  loss_curve:   {uri: loss_curve, params: {series: [train/lm, val/perplexity], x: step, rates: true}}
  confusion:    {uri: confusion_matrix}
  corr:         {uri: correlation_heatmap, sets: [test], width: 11, height: 11}
  architecture: {uri: architecture}

record: runs/charlm_$datetime$      # $datetime$ is filled once, at the start
```

The plots split in two. Some read `predictions.parquet` and the models (`pred_vs_true`, `residuals`, `binary_roc`,
`confusion_matrix`, `permutation_importance`); some read the data itself in its original units
(`target_vs_features`, `correlation_heatmap`, `feature_distributions`, and the seaborn wrappers `pairplot`,
`violin`, `kde`). `kalfa plots runs/x --set figures.format=pdf` redraws them all from the record without training
anything.

## Recipes

Every folder under [`examples/`](examples/) is a runnable version of one of these: a `config.yaml`, a
`make_data.py` that writes small synthetic data next to it, and the commands in
[`examples/README.md`](examples/README.md). The fragments below are the part that matters.

<details open>
<summary><b>Classification with class weights</b> (<code>02_mlp_classification</code>)</summary>

```yaml
data:
  preprocessors:
    scaler: {uri: standard_scaler}
    onehot: {uri: one_hot}
    label:  {uri: label_encoder}
  fields:
    "num_*": {preprocessors: [scaler]}
    "cat_*": {preprocessors: [onehot]}
    churned: {target: true, preprocessors: [label]}

losses:
  ce: {uri: cross_entropy, params: {weight: {uri: class_weights}}}   # computed once the train loader exists

training:
  stop:       [{uri: plateau, params: {monitor: val/f1, mode: max, patience: 5}}]
  checkpoint: {uri: best, params: {monitor: val/f1, mode: max}}
```

The prediction is decoded back to the original label in `predictions.parquet`.
</details>

<details>
<summary><b>Pretrained backbone, frozen then opened</b> (<code>04_cnn_images</code>)</summary>

```yaml
data:
  source: {uri: image_folder, params: {path: data/pets}}
  batch:  {size: 64, eval_size: 256, balanced: true}      # class balancing sampler
  preprocessors:
    resize:    {uri: resize, params: {size: 224}}
    augment:   {uri: random_crop_flip, params: {size: 224}, sets: [train]}   # train only
    to_tensor: {uri: to_tensor}
    normalize: {uri: normalize, params: {mean: imagenet, std: imagenet}}
  fields:
    image: {preprocessors: [resize, augment, to_tensor, normalize]}
    label: {target: true}

optimizers:
  main: {uri: adamw, params: {lr: 1.0e-3, groups: [{match: "backbone.*", lr: 1.0e-5}]}, loss: ce}

training:
  predicts: classifier                                     # the composite of backbone + head
  rules:
    - {name: unfreeze, when: {uri: after_epoch, params: {at: 5}}, set: {backbone.trainable: true}}
```
</details>

<details>
<summary><b>Two optimizers, WGAN GP</b> (<code>07_wgan_gp</code>)</summary>

```yaml
losses:
  wgan_d: {uri: wgan_gp_d, params: {generator: generator, critic: critic, latent: 128, gp_weight: 10}, sets: [train]}
  wgan_g: {uri: wgan_g,    params: {generator: generator, critic: critic, latent: 128}}

optimizers:
  d: {uri: adam, params: {lr: 1.0e-4, betas: [0.0, 0.9]}, loss: wgan_d}
  g: {uri: adam, params: {lr: 1.0e-4, betas: [0.0, 0.9]}, loss: wgan_g}

metrics:
  fid: {uri: fid, params: {model: generator.ema, latent: 128, n: 2000}, every: 5, sets: [valid]}

training:
  turn: {uri: alternating, params: {order: [d, g], steps: {d: 5, g: 1}, fresh_batch: true}}
  checkpoint: {uri: best, params: {monitor: val/fid}}

generate: {uri: gan_sampler, params: {model: generator.ema, latent: 128, n: 64}}
```

`fresh_batch: true` draws a new real batch for every critic step; the gradient penalty lives inside the loss.
`kalfa generate runs/x` samples again later from the same record.
</details>

<details>
<summary><b>Language model, counted in steps</b> (<code>10_char_lm</code>)</summary>

```yaml
plugins: [gpt_legos]                   # the gpt block lives next to the config

data:
  source: {uri: text_lines, params: {path: data/shakespeare.txt}}
  preprocessors: {tokenizer: {uri: char_tokenizer}}
  fields: {text: {preprocessors: [tokenizer]}}
  feed: {uri: next_token, params: {seq_len: 128}}

model:
  models:
    gpt:
      optimizer: main
      inputs: [input_ids]
      outputs: [logits]
      nodes:
        - {uri: embedding, params: {num: {uri: vocab_size}, dim: 256}}   # from the fitted tokenizer
        - {uri: gpt, params: {d_model: 256, layers: 4, heads: 4, seq_len: 128}}
        - {uri: linear, params: {out_features: {uri: vocab_size}}}

training:
  steps: {total: 20000, turn: 200}     # a turn is 200 updates, not a pass over the data
  accumulate: 2
  amp: true
  grad_clip: 1.0

generate: {uri: lm_sampler, params: {model: gpt, prompt: "ROMEO:", max_new_tokens: 500}}
```
</details>

<details>
<summary><b>Several targets at once</b> (<code>15_multi_target</code>)</summary>

```yaml
data:
  preprocessors:
    x_scaler: {uri: standard_scaler}
    y_scaler: {uri: standard_scaler}
  fields:
    "y_*": {target: true, preprocessors: [y_scaler]}   # three columns, three fitted scales
    z:     {target: true, preprocessors: [y_scaler]}
    "x*":  {preprocessors: [x_scaler]}

training:
  targets:                        # which output wire predicts which fields
    y_hat: "y_*"
    z_hat: z

losses:
  l_y: {uri: mse, output: y_hat}  # the target comes from the table above
  l_z: {uri: mse, output: z_hat}
```

The fields a selector names become one tensor, so `mse` is called once with two `(batch, 3)` sides, while every
column keeps its own scale for the metrics and for `pred_y_hat_y_a`, `pred_y_hat_y_b` in `predictions.parquet`.
</details>

<details>
<summary><b>A sweep, locally or on a batch system</b> (<code>14_sweep_grid</code>)</summary>

```yaml
include: [../01_mlp_regression/config.yaml]
params:  {epochs: 20, width: 128}
model:
  nodes:
    - {uri: linear_relu, params: {out_features: $width$}}
    - {uri: linear_relu, params: {out_features: $width$}}
    - {uri: linear, params: {out_features: 1}}

sweep:
  strategy: grid                                   # grid | random | sobol (deterministic by id) | optuna (fed back)
  space:
    lr:    [1.0e-2, 1.0e-3, 1.0e-4]
    width: [64, 128]
  objective: {monitor: val/rmse, mode: min, at: best}
  record: runs/sweep_housing
```

```bash
kalfa sweep config.yaml --count                 # 6
kalfa sweep config.yaml --show 4                # the params of point 4
kalfa sweep config.yaml                         # the local loop, one subprocess per point
kalfa collect runs/sweep_housing                # runs/sweep_housing/reports/sweep.md, the report with its figures

# on a queue system: plan the root once, then one job per point
kalfa sweep config.yaml --plan --prepare-data --record /shared/sweeps/housing
condor_submit /shared/sweeps/housing/sweep.sub  # sweep.sub and sweep.sh are yours to edit, never overwritten
kalfa collect /shared/sweeps/housing            # works on a half finished sweep too
```

A k fold inside a sweep is one more swept param: `fold: [0, 1, 2, 3, 4]` in the space and `fold: $fold$` in the
`kfold` split. `kalfa collect <root> --mean-over fold` then ranks the settings of the other params by their mean
objective over the folds, so the best setting is not the one that met the easiest fold (`groups.csv`).

A point is an ordinary record under `<root>/<id>/`. `grid`, `random` and `sobol` are deterministic by id, so
`--id N` reproduces one point anywhere and looks at no other point.
</details>

## Commands

```
kalfa run      cfg.yaml [--set path=value] [-p name=value] [--log info|debug] [--no-progress]
kalfa check    cfg.yaml [--layers] [--dump] [--recipe] [--measure]
kalfa describe cfg.yaml [--measure] [--section data|model|training|after|columns|wiring] [--save report.txt]
kalfa predict  runs/x   [--model name] [--which best|last|final] [--data new.parquet] [--device cuda] [--plots]
kalfa generate runs/x   [--which best|last] [--device cuda]
kalfa export   runs/x   [--format onnx|pt2|state_dict] [--format-param opset=18] [--model name] [--out DIR]
kalfa plots    runs/x   [--only a,b] [--set figures.format=pdf]
kalfa resume   runs/x   [--set training.epochs=N]
kalfa repair   cfg.yaml [--record runs/x | --record root [--id N]]   # continue a failed run or sweep points in place
kalfa prepare  cfg.yaml --out DIR                        # run the data block once; run --prepared DIR reuses it
kalfa sweep    cfg.yaml [--count | --show N | --id N | --plan [--prepare-data]] [--record root]
kalfa collect  <sweep root> [--mean-over fold] | runs/cv_*   # the sweep report, or the k fold summary
kalfa stop     runs/x                                    # the run ends after its current turn
kalfa board    <root> [--port 8080]                      # a page over every record under a root
kalfa ls       [/alias/kalfa/tabular | /criterion | word] [--plugin mod] [--config cfg.yaml]
kalfa docs     [--write DOCS.md]
kalfa contract [--write contract.yaml]
kalfa --version
```

Two override forms, on every command that reads a config:

```bash
--set training.epochs=5          # a dotted path from the root of the document; the value is parsed as YAML
--set training.stop=[]           # a list is replaced wholesale
--set device=cuda
-p lr=1e-4                       # short for --set params.lr=1e-4
```

`run`, `check` and `describe` also take a record directory in place of a config, reading that record's
`resolved.yaml` under its own `contract.yaml`.

### Before you run: `check` and `describe`

`kalfa check` answers one question, are there problems: unknown keys, unresolvable names, a lego in the wrong
section, an unresolved reference, a column that matches no glob or two, a `test/` monitor, an unused optimizer,
a rule writing a target that does not exist, two unordered nodes touching the same object. It never raises, it
reports. `--measure` runs the data block for the real set sizes, `--dump` prints the graph that would run.

`kalfa describe` answers the rest, printing the config as an analysis. This is `examples/01_mlp_regression`,
which is the config above plus a chain of rules that switches the loss three times, abridged in the middle:

```
config.yaml                                                                          kalfa 0.4.0
  layers      config.yaml, /alias/kalfa/base, /alias/kalfa/tabular
  seed        7
  device      cpu (no device key)
  rng         derived (no rng key)
  record      runs/housing_$datetime$

── DATA ────────────────────────────────────────────────────────────────────────────────────────
  source      parquet  housing.parquet                    2 000 rows · 9 columns
  split       random  0.7 / 0.15 / 0.15  seed=7           train 1 400  ·  valid 300  ·  test 300
  batch       128                                         feed  table

  field  columns       preprocessors  role
  ───────────────────────────────────────────
  x*     x0 … x7  (8)  std_scaler     feature
  price  price         target_std     target

  housing.parquet ─→ split random
    ├─ train    1 400  ─→ fit std_scaler, target_std ─→ table
    ├─ valid      300  ─→ apply                      ─→ table
    └─ test       300  ─→ apply                      ─→ table

── MODEL ───────────────────────────────────────────────────────────────────────────────────────
  model   trained  ·  optimizer model
    x ─→ linear_relu 128 ─→ linear_relu 128 ─→ linear 1 ─→ y

── TRAINING ────────────────────────────────────────────────────────────────────────────────────
  turn        alternating         100 epochs      predicts model

  optimizer  lego           trains    loss      schedule
  ─────────────────────────────────────────────────────────
  model      adam lr=0.001  ─→ model  loss_mse  no schedule

  loss          lego           reported on
  ───────────────────────────────────────────────────────────────────
  loss_mse      mse            train, valid, test  active at turn 1
  loss_huber    huber delta=1  train, valid, test  held for the rules

  checkpoint  best monitor=val/rmse                       report best
  stop        val/rmse plateau 6

  rules (evaluated at the end of a turn, effective in the next)
    turn ≥ 50                ─→  to_huber    loss := loss_huber
    train/loss_huber < 0.01  ─→  to_mae      loss := loss_mae      after to_huber

── COLUMNS ─────────────────────────────────────────────────────────────────────────────────────
  column  dtype   field  preprocessors  role     tensor
  ─────────────────────────────────────────────────────
  x0      double  x*     std_scaler     feature  x
  price   double  price  target_std     target   price
```

Statically it reads only the file header and the compiled recipe. `--measure` runs the data and model blocks for
real, without writing anything, and fills in the sizes after the filters, the widths a fitted `one_hot` produces
and the parameter counts of lazy layers. In a terminal the colors carry the grammar: a lego name is cyan, a
parameter name is dim, the values stay plain, so `parquet  housing.parquet` reads as the lego and its file.
Tables are fitted to the terminal width; with `--save report.txt`, or any time the output is not a terminal,
nothing is clipped.

### While it runs: `--log`

```bash
kalfa run config.yaml --log info --no-progress
```

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

`--log debug` adds every node with its timing and the decisions inside the legos. Without the flag the output is
the tqdm progress bar; `--no-progress` drops the bar entirely (tqdm is never even imported); `--log-every N`
prints a line every N updates with the loss, the learning rate and the gradient norm.

```bash
kalfa stop runs/x        # writes stop.json; the run ends after its current turn, as an early stop would
```

### When it fails: `repair`

A run that fails writes `failure.json` with the traceback of every failed node. Fix the lego or the plot, then
continue the same run in its own directory, with the config you started it with:

```bash
kalfa repair cfg.yaml --record runs/x_20260916_115459          # --record when the config names $datetime$
kalfa repair sweep.yaml --id 12 --record sweeps/lr             # one sweep point, for a queue job
kalfa repair sweep.yaml --record sweeps/lr                     # every failed point, the others skipped
```

Where it continues from is read from the record: with `final/state.pt` training was over, so only the after block
runs again (calibration, predictions, plots); with `checkpoints/last.pt` training goes on from that turn and
`history.jsonl` and `steps.jsonl` are cut back to it. Without either there is nothing to continue from and repair
says so; a run is repairable when it writes a checkpoint every turn (`training.checkpoint: last`, or `best`). The
plugins next to the config are the code that runs, and the config may differ from the recorded one only under
`plots`, `figures`, `device`, `calibrate`, `generate`, `params` and `training.report`; anything else is a new run
(`kalfa run`) or more turns (`kalfa resume`). Only a failed run is repaired: a finished, running, pending or lost
one is refused, and the sweep loop skips it.

## The board

```bash
kalfa board runs --port 8080        # http.server plus one page; Vue and Plotly come from jsDelivr, pinned by version and hash
```

The board reads records; it never writes one except to ask a run to stop. It finds every record under the root,
follows the live ones over a server sent event stream, and gives every record one page of tabs. The address bar
carries the record, the tab, the open chart and the view options, so a link is one exact view and a reload keeps
it. Every chart zooms with a drag and resets with a double click, and a log axis is a real log axis. On a batch
system it runs on the login node and the browser reaches it through an ssh tunnel.

<p align="center"><img src="https://raw.githubusercontent.com/dboncukcu/kalfa/main/docs/images/board_monitor.png" width="920" alt="the monitor tab of a finished run"></p>

**Monitor.** The turn and batch bars, the checkpoint monitor across the sets, the loss every optimizer minimizes,
the batch loss per step, the latest values and the log tail, refreshed as the record changes. The sidebar lists
the runs and the sweeps with their points, a dot for the state of each; the tab title of the browser carries the
progress of everything running, so a background tab says how far the training is.

<p align="center"><img src="https://raw.githubusercontent.com/dboncukcu/kalfa/main/docs/images/board_curves.png" width="920" alt="one chart per history series, the sets as lines"></p>

**Loss and metrics, optimizer steps.** One chart per history series with the sets as lines and the turns a rule
fired marked; one chart per step series with the turns labelled along the top. Every chart expands into a large
view beside a settings panel (log scale on either axis, the axis ranges and titles, grid, legend, font, line
width, points, height, the bins of a histogram, and a JSON box for any other Plotly layout key) whose download
writes it as drawn, as png or svg; the axis says epochs when a turn is one.

<p align="center"><img src="https://raw.githubusercontent.com/dboncukcu/kalfa/main/docs/images/board_model.png" width="920" alt="the architecture drawn as a schematic"></p>

**Model.** The architecture from `architecture.json` as a schematic: a block per node with its pins, every wire
with its width, the losses and the optimizers beside the outputs they read. A block opens into its layers with the
tensor shapes one batch traced; a model inside a model opens into its own nodes. Drag a block, bend a wire, zoom,
save the png as arranged.

<p align="center"><img src="https://raw.githubusercontent.com/dboncukcu/kalfa/main/docs/images/board_data.png" width="920" alt="the data pipeline stage by stage"></p>

**Data.** The data block as the same kind of schematic, stage by stage with the rows and columns, and the tables
of what every transform added or removed, what the fit learned on train and what each loader yields.

<p align="center"><img src="https://raw.githubusercontent.com/dboncukcu/kalfa/main/docs/images/board_predictions.png" width="920" alt="the predictions tab with a query"></p>

**Predictions.** The prediction against the truth with the y = x line, the residual histogram, the distribution of
the prediction outlined over the distribution of the truth on the same bins with the ratio of their counts and its
error below, and the largest errors, in the original units. A switch per target turns it into a classification:
the ROC of a score column with its AUC, the background efficiency and rejection at 50, 80, 90 and 95 % signal
efficiency with the cut that gives each, the score by class, and the confusion of the predicted classes. Nothing is
drawn until you pick the score column and the signal class; the switch and the two picks are kept in the browser
per target name, so every run and point with that target opens the same way. A filter over the
columns of the file, the targets, the `pred_` and `raw_` wires, the calibration flags and the `data.spectators`
columns narrows every number on the page to the rows it keeps, and rides in the address bar with the rest of the
view:

```
site == "b" and pred_y > 0 and price < 4
```

It takes column names, numbers, strings, comparisons, `and`, `or`, `not`, arithmetic and lists; anything else (an
attribute, a call, a variable) is refused before pandas sees it.

**Failures.** A failed run shows its failed node and the error in the header, and the overview the traceback from
`failure.json` with every repair since; a sweep groups its failed points by their first error. A run whose
heartbeat stopped for more than ten minutes without an end is shown as lost. A value that is not finite in the
history (NaN, inf) is named in the header with the turn it appeared, and its charts break there instead of drawing
over it.

<p align="center"><img src="https://raw.githubusercontent.com/dboncukcu/kalfa/main/docs/images/board_sweep_space.png" width="920" alt="every point of a sweep across the space"></p>

**Sweeps.** A sweep is one record with its points below it. Its page draws every point across the whole space,
the failed ones dashed, coloured by the rank of the objective; a drag on any axis narrows the charts and the table
together.

<p align="center"><img src="https://raw.githubusercontent.com/dboncukcu/kalfa/main/docs/images/board_sweep_explorer.png" width="920" alt="the explorer and the table of points"></p>

The explorer picks its own axes and colour, the table of points sorts by any column, and the best point so far
stands above it with its params.

<p align="center"><img src="https://raw.githubusercontent.com/dboncukcu/kalfa/main/docs/images/board_sweep_params.png" width="920" alt="every param against the objective, and where the points died"></p>

Every param against the objective, a strip per level for a choice, and where the failed points died: a level that
fails about as often as every other level points at the node or the environment, not at the params.

<p align="center"><img src="https://raw.githubusercontent.com/dboncukcu/kalfa/main/docs/images/board_sweep_progress.png" width="920" alt="the objective over the sweep order"></p>

The objective over the sweep order says whether the search is still finding anything; below it any series of the
ticked points overlaid (the objective unless you pick another, the best 5, 10, 20 or 50 ticked in one step) and the
`resolved.yaml` difference between two of them.

**Runs table.** Every run and sweep point under the root with its params, its best value and its last evaluation
values. The filter takes words, each one a comparison of a param, a last value, `best`, `turns` or `seconds`, or a
text found in the name, the path or a `param=value`; a row stays when every word holds:

```
lr<0.01 val/rmse<=0.3 width=64 fatjet
```

Chips keep the states you pick (running, failed, lost, ...) and a menu the runs started in the last day, week or
month; the sidebar has the same chips over the whole tree. A columns menu picks the params and values the table
carries (every param and the first 8 values unless you pick, kept in the browser per root), and the table, or the
points of a sweep, saves as CSV as it is shown: filtered, sorted and with the columns picked.

**Compare.** Tick any number of rows in the runs table (or the first 50 shown) or points of a sweep, and every
history series of them is drawn together, one chart per series with a line per record, a colour and a dash per
record so that 32 stay apart. Chips keep one set, a filter the series whose name holds a word; the params that
differ are tabled below, and two records also get their `resolved.yaml` difference. A record that is running
redraws as it goes. The page is a link (`#/compare?runs=a,b,c`), so a comparison can be shared as it is.

**On a shared filesystem.** One background loop follows the records for every open tab, every two seconds, and
stops half a minute after the last tab closes, so nothing polls the filesystem while nobody looks. The history,
the steps and the events are read from where they were left, a log from its end, and a small file only when its
size, time or inode changed; the last seen time of a record comes from the few files a run keeps writing. The
last four prediction files read stay in memory while they are unchanged, so moving the bins or picking another
score reads nothing again.

<p align="center"><img src="https://raw.githubusercontent.com/dboncukcu/kalfa/main/docs/images/board_plots.png" width="920" alt="the plots of a record as a gallery"></p>

**Plots, files and the rest.** The figures under `plots/` as a gallery, the sample images as they are written,
every file of the record with a viewer for text, images and PDFs, `resolved.yaml`, the fitted preprocessors and
what each learned, the identity notes (`manifest.json`, `host.json`, `device.json`, `git.json`), the event tail
and `describe` on demand. The node timeline draws a loop as one bar and tables every node inside it summed over
the iterations (runs, total, mean, longest and share of the loop), with every iteration as its own row a click
away. The logs show the last 300, 2000 or 10000 lines of `stdout.txt` and `stderr.txt` with a progress bar as its
last state, follow the end unless you scroll up, and link the whole file. A running record shows a stop button in
its header; `kalfa stop` from the shell does the same.

## Python API

Every command is a function; the command line is a thin shell over them. `paths` is the list of config layers,
`sets` is `--set` as `(dotted path, value)` pairs.

```python
from kalfa.api import check, generate, predict, resume, run, stop

prepared = check(["config.yaml"], measure=True)       # never raises; it returns what it found
print(prepared.errors, prepared.warnings, prepared.sizes)

result = run(["config.yaml"], sets=[("training.epochs", 3), ("record", "runs/nb_$datetime$")])
result.record                                         # the directory it opened
result.report.outputs["history"]                      # the per turn metric table

longer     = resume(result.record, sets=[("training.epochs", 10)])
prediction = predict(result.record, data="new.parquet")   # .table is a DataFrame
samples    = generate(result.record, which="best")
```

| Function | Returns | Carries |
|---|---|---|
| `check(paths, sets, measure, contract, prepared)` | `Prepared` | `problems`, `errors`, `warnings`, `sizes`, `header`, `document`, `analysis`, `dump()` |
| `probe(document, contract)` | `Probe` | the data and model blocks run alone: `sizes`, `prep`, `features`, `parameters`, `shapes` |
| `run(paths, sets, executor, workers, contract, monitor, prepared)` | `RunResult` | `record`, `report`, `device` |
| `resume(run_dir, sets, ...)` | `RunResult` | the same, in a new record directory |
| `repair(paths, sets, record, prepared, executor, workers, contract, monitor)` | `Repaired` | `result`, `stage` (`after` or `training`), `turn`, `attempt`; the failed run goes on in its own directory |
| `predict(run_dir, model, which, data, sets, device, contract, plots)` | `Prediction` | `path`, `table`, `model`, `plots` |
| `generate(run_dir, which, sets, device, contract)` | `Generated` | `path`, `samples` |
| `plots(run_dir, only, sets, device, contract)` | `Plots` | `record`, `names` |
| `open_record(run_dir, which, sets, contract)` | `Opened` | `contract`, `document`, `store`, `prep`, `rebuild()` |
| `stop(record, by="cli")` | list | the directories it wrote `stop.json` into |
| `prepare_data(paths, sets, out)` | `PreparedData` | the data block run once into `out` |
| `kalfa.collect.collect(run_dirs, out, markdown, style, top, figures, mean_over)` | tuple | `kind` (`sweep` or `cv`), the terminal text, the directory written and the files written in it |
| `kalfa.sweep.plan(paths, sets, record)` | `Plan` | the points, without running any of them |

In a notebook, four things differ from the shell:

```python
import json, pandas as pd
from IPython.display import Image
from kalfa.api import run
from tezgah import load_run

result = run(["config.yaml"], sets=[("record", "runs/nb_$datetime$")])   # 1. a non empty record dir is an error

history = pd.DataFrame(result.report.outputs["history"])                 # 2. the metric table, one row per turn
history.plot(y=["train/loss_mse", "val/rmse"])
pd.DataFrame([json.loads(l) for l in open(f"{result.record}/history.jsonl")])[["turn", "lr/model", "rules"]]

record = load_run(result.record)                                         # 3. tezgah reads finished records
Image(f"{result.record}/plots/loss_curve.png")
```

4. Registering the same URI twice is an error, which is what a re-run cell does. Restart the kernel after editing
a lego, or drop the entry first:

```python
from cirak.registry import registry
registry._entries.pop("/criterion/acme/asymmetric", None)
registry._resolved.pop("/criterion/acme/asymmetric", None)
```

Interrupting a cell stops the run where it is; `stop(record)` from another cell asks it to end after the current
turn instead, and `resume(record)` picks it up from the last checkpoint.

## The record directory

One writer per file, whole files written to a temporary name and renamed, growing files appended one JSON line at
a time. A thousand sweep points can write into one shared filesystem while somebody watches.

| File | Contents |
|---|---|
| `manifest.json` | the identity, written once: `kind` (`run`, `point`, `sweep`, `data`), name, config paths, params, version, contract hash, whether a turn is an epoch or a step count |
| `resolved.yaml` | the config with aliases and `$param$`s resolved, the source of every override in a comment; runs again on its own |
| `contract.yaml` | the contract this run was compiled by; `predict`, `generate` and `resume` read it back |
| `flow.yaml` | the tezgah graph that ran, with tezgah's resolution comments |
| `history.jsonl` | per turn: the `train/`, `val/`, `test/` values, `global_step`, `lr/<optimizer>`, `minimizes/<optimizer>`, `effect/<target>`, `seconds`, the rules that fired |
| `steps.jsonl` | per update: `step`, `turn`, `loss/<optimizer>`, `lr/<optimizer>`, `grad_norm/<optimizer>` under `grad_clip` |
| `checkpoints/` | `best.pt`, `last.pt`, `snapshot_<n>.pt` by policy: models, optimizers, EMAs, counters, rule states, RNG |
| `final/state.pt` | always, once the run ends |
| `fitted/` | `preprocessors/` (one file per name plus `plan.json`), `frames/`, `calibrate/` |
| `predictions.parquet` | the test set: `row`, the targets inverted, `pred_<output>`, `raw_<output>`, the calibration flags, and the `data.spectators` columns as they were read |
| `plots/`, `samples/` | named after the definition (`plots.confusion` → `confusion.png`); `samples/` is what `generate` wrote |
| `data.json` | rows and columns at every stage of the data block, the sets, the fitted objects, the loaders |
| `architecture.json` | every report model as boxes, wires and traced shapes, with the layers inside each node |
| `device.json`, `git.json`, `host.json` | the chosen device, the commit of the config's repository and whether it was dirty, the host and pid |
| `stop.json` | a stop request, the one file written from outside; the loop ends after the turn that sees it |
| `events.jsonl`, `run.json`, `stdout.txt`, `stderr.txt` | tezgah's event stream and summary |
| `failure.json` | when the run fails: every failed node with its path, the exception and the full traceback |
| `heartbeat` | touched every minute while the run runs, removed when it ends; one left behind means the process was killed |
| `repair.json`, `attempts/<n>/` | the repairs of a failed run and the files of every failed attempt (`kalfa repair`) |
| `plugins/` | copies of the modules the run imported, so `predict`, `generate` and `resume` work from the record alone |
| `export/`, `resume.json` | what `kalfa export` wrote, the source run of a resume |

Status is derived, never written by a coordinator: a manifest alone is pending, a growing `history.jsonl` is
running, `run.json` is finished or failed, a `failure.json` without `run.json` is failed, and a heartbeat older than
ten minutes without either is lost (a job the batch system killed, or a machine that went away).

## Your own legos

A lego is a function or a class registered with `@kalfa.lego`. Its kind is the first segment of the URI, and the
kind alone decides where it may be written. Nothing else is needed: no base class for the simple kinds, no
registration file, no framework import inside your training code.

```python
# mylegos.py, next to the config
import kalfa
import torch


@kalfa.lego("/criterion/acme/asymmetric", alias="asymmetric", partial=True,
            description="Squared error, over prediction weighted by factor")
def asymmetric(predictions, targets, factor=2.0):
    error = predictions.reshape(-1) - targets.reshape(-1)
    return torch.where(error > 0, factor * error ** 2, error ** 2).mean()


@kalfa.lego("/trigger/acme/after_minutes", partial=True,
            description="Fires once the run is older than minutes")
def after_minutes(metrics, turn_index, state, minutes=30.0):
    import time
    state = dict(state or {})
    started = state.setdefault("started", time.time())
    return time.time() - started > 60.0 * minutes, state
```

```yaml
plugins: [mylegos]

losses:
  asym: {uri: asymmetric, params: {factor: 3.0}}

training:
  stop: [{uri: /trigger/acme/after_minutes, params: {minutes: 90}}]
```

```bash
kalfa ls asymmetric --plugin mylegos     # your lego with its signature and facts
kalfa docs --config config.yaml          # the full reference with a "Plugin legos" section
```

The run copies every plugin module into `<record>/plugins/`, so `predict`, `generate` and `resume` keep working
from the record alone. A pip package declares itself with the `cirak.plugins` entry point instead and then works
anywhere it is installed.

### Facts

Facts are what a lego tells the framework about itself, so that `check` and `describe` can reason about it
without building it:

| Fact | Says |
|---|---|
| `alias` | short names for the alias packs |
| `partial` | built with its params at compile time, called later |
| `state` | stateful, goes into the record |
| `returns` | the outputs of a flow step |
| `refs` | which params are references and of what type (`model`, `loss`, `criterion`, `schedule`, `preprocessor`, `field`, `column`) |
| `bus` | bus keys bound to defaulted parameters |
| `mutates`, `aliases` | inputs changed in place; outputs holding their inputs (this is what the aliasing check reads) |
| `uses`, `needs_grad`, `extras` | needs the prediction model; needs gradients in the evaluation pass; the training keys a turn accepts |
| `grouped` | a preprocessor fitted once over every column that names it |
| `requires` | a library that is not a kalfa dependency; `check` warns when it is not installed, and the lego says so and is skipped |

An undeclared fact is a `RegistryError`, so a misspelled one is caught at import.

### The contracts

Simple kinds are plain functions. The rest subclass a base, and `check` and `describe` read the class attributes
without building the object:

| Contract | Base class | A subclass writes |
|---|---|---|
| criterion | none, a function | `(predictions, targets, **params)` |
| objective | none, a function | `(models, batch, **params)`, plus `step`, `epoch`, `rng`, `scaler`, `losses` if named |
| trigger | none, a function | `(metrics, turn_index, state)` returning `(fired, state)` |
| preprocessor | `std.pre.base.Preprocessor` (`Scaler`, `Affine`, `Encoder`, `Tokenizer`) | `apply`, and `fit`, `inverse`, `columns`, `decode` as its facts declare; `affine()` under `Affine`, or an `inverse_torch` of its own, keeps the metrics' inversion on the device |
| metric | `std.metric.base.Metric` | `reset()`, `update(...)` naming what it wants, `compute()` |
| model | `std.builder.base.Model` | `inputs`, `outputs`, `initialized`, `trainable` |
| dataset | `std.feed.base.Dataset` / `IterableDataset` | `inputs`, `targets`, `frame`, `rows()`, `labels(name)`, `size()` |
| checkpoint policy | `std.checkpoint.base.Policy` | `monitor`, `tags(metrics)`, `state()`, `restore(state)` |
| frame transform | `std.frame.base.FrameTransform` | `fit(df)` on train, `apply(df)` per set |
| calibration | `std.calibrate.base.Calibration` | `fit(models, loaders, prep, device, predicts)`, `apply(table)`, `note()` |
| optimizer | `std.optimizer.base.Optimizer` | `torch_class` |
| strategy | `std.strategy.base.Strategy` | `deterministic`, `total(space)`, `point(space, index)`, or `ask` / `tell` |
| turn | none, a function | the signature of `/turn/kalfa/alternating`, returning the same mapping |

A plot lego takes whatever it names in its signature out of what the run holds at that moment: `predictions`,
`history`, `models`, `record`, `prep`, `train_loader`, `valid_loader`, `test_loader`, `loaders`, `predicts`,
`sets`, `name`, `device`, `counters`, `optimizers`, `emas`, `rules`, `figures`. Draw through `figures` and the
figure comes out in the run's own format, panel size, palette and grid.

```python
from kalfa.std.common.figure import Figure


@kalfa.lego("/plot/acme/spread", description="Prediction spread against the truth")
def spread(predictions, record, name=None, figures=None):
    figures = figures or Figure()
    drawing, axis = figures.single()
    axis.plot(predictions["price"], predictions["pred_y"], ".")
    figures.label(axis, title="spread", xlabel="price", ylabel="prediction")
    figures.save(drawing, record, name or "spread")
```

A key the plot cannot work without goes into its `needs` fact (`needs=["train_loader"]`); `kalfa plots` and
`predict --plots` then skip the plot with a log line instead of calling it with nothing.

Logging from a lego needs no declaration at all: `logging.getLogger("kalfa.<stage>")`, where the stage is the
place in the run (`data.source`, `training.turn`, `after.plots`). The line appears only when the user asked for
it with `--log`.

### The contract, when a lego is not enough

Some things are not a lego but a step in the flow. `kalfa contract --write contract.yaml` exports the mapping
kalfa runs configs by, and `--contract contract.yaml` runs against your edited copy:

```yaml
after:
  flow:
    notify: {uri: /lego/acme/slack, inputs: {history: history}}
```

Add a node to a block, change where the default split comes from, or evaluate `train/` in a clean pass
(`sets: {default: [train, valid, test]}`) without touching kalfa. Every record keeps the copy it ran with, so a
run stays reproducible when the default changes.

## Where everything is documented

| Document | What it is |
|---|---|
| [`CONFIG.md`](CONFIG.md) | the complete definition of the config surface, key by key, plus the appendix on the mapping onto the lower layer |
| [`DOCS.md`](DOCS.md) | the lego reference, generated from the registry: every URI with its aliases, signature, facts and description |
| [`reference.yaml`](reference.yaml) | every key with its class: required, empty, derived or library default |
| [`examples/`](examples/) | seventeen runnable configs with their data generators and commands |
| `kalfa contract` | the wiring and the five flow blocks kalfa runs a config by |
| [`tests/configs/reference.yaml`](tests/configs/reference.yaml) | the reference config of the test suite: every mechanism in one file, trained once and inspected facet by facet |

## Development

```bash
uv sync
uv run python -m pytest -m "not slow and not subprocess"    # the surface and lego layers, seconds
uv run python -m pytest                                      # everything, the runs and the command line included
uv run kalfa docs --write DOCS.md                            # the lego reference, from the registry
```

The tests sit in four layers that follow a config on its way to a run: `tests/surface/` (loading, checking and
shaping a config, one file per module), `tests/legos/` (every std lego built by URI on hand written inputs, one
file per kind; a contract test fails while a std URI is exercised nowhere), `tests/runs/` (one reference config,
`tests/configs/reference.yaml`, trained once per session and inspected facet by facet, plus one small run per
mechanism a tabular model cannot express) and `tests/cli/` (every command, on the reference record). Nothing is
compared to a golden file: a test asserts the exact files, keys and shapes a run writes, and the seeded reference
run is repeated to the bit. A `UserWarning` or a `ResourceWarning` fails the suite, so a matplotlib layout that
does not fit or a file left open is a failure, never a line of noise. Markers: `slow` (trains a model),
`subprocess` (spawns kalfa in a child process).

**The std tree.** A std lego's URI is `/<kind>/<pack>/<name>` and its module is `src/kalfa/std/<kind>/<pack>/`. A
file holds one lego with a body of its own, or a family of small ones (`std/criterion/kalfa/regression.py` holds
`mse`, `mae`, `huber`, `log_cosh` and `weighted_mse`). Code two families of a kind share sits in the kind's
`base.py`, code two legos of a pack share in the pack's `base.py`, code crossing kinds in `std/common/`. A
contract test checks that every std lego is registered from a module of its own kind and pack, that every
module registers one, and that every std URI is exercised by a test of its kind under `tests/legos/`.

**Adding an example.** `examples/<nn>_<name>/` with a `config.yaml` (a comment header, paths relative to the
folder), a `make_data.py` carrying its own generator, the plugin module when the config names one, and an entry in
`examples/README.md`. Every example is checked, never trained, by `tests/surface/test_examples.py`. A new lego
gets its test in `tests/legos/test_<kind>.py`; a mechanism a tabular model cannot express gets a small config under
`tests/configs/` and a `tests/runs/test_mechanism_<name>.py` that finishes on a laptop CPU in seconds.

**Releasing.** `bash tools/release.sh 0.2.7 "a one line summary"` writes the version into `pyproject.toml` and
`uv.lock`, writes `DOCS.md` again (its header carries the version), and asks before the
commit, the annotated tag and the push. Run the suite against a reinstalled environment (`uv sync --reinstall`)
first, then `uv build`. kalfa needs tezgah 0.2.0 or later and cirak 0.2.3 or later.

## License

MIT.
