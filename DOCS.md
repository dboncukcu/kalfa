# kalfa reference

This is the full reference of kalfa: the words it uses, the config grammar, every section and key, what happens while a run trains, the files of a record directory, every command, the Python API, the board and the terminal interface, and how to write your own pieces. The catalog of short names is not copied here, because it changes with the code: `uv run kalfa legos --docs` prints it as markdown, `uv run kalfa legos` lists it in the terminal and `uv run kalfa legos NAME` shows one page with the params, their defaults and notes, YAML examples and the example configs that use the name.

Contents:

1. [Vocabulary](#vocabulary)
2. [The config grammar](#the-config-grammar)
3. [Sections](#sections)
4. [The runtime](#the-runtime)
5. [The record directory](#the-record-directory)
6. [Commands](#commands)
7. [Python API](#python-api)
8. [The board](#the-board)
9. [The terminal interface](#the-terminal-interface)
10. [Writing your own pieces](#writing-your-own-pieces)
11. [Reference config](#reference-config)

## Vocabulary

* **config**: one or more YAML files (or Python mappings) that describe a run. Later files override earlier ones.
* **section**: a top level key of the config, such as `data`, `models` or `rules`.
* **call**: a mapping with a `use` key, `{use: mlp, hidden: [64], out: 1}`. kalfa builds the object `use` names with the other keys as params. A mapping without `use` is plain data.
* **short name**: a name from the catalog (`mlp`, `adam`, `plateau`). The catalog is also called the legos; `kalfa legos` lists it.
* **import path**: a dotted path to any importable class or function, `torch.nn.HuberLoss` or `mylegos.Classifier`.
* **slot**: the role of a call, given by where it stands: data, model, loss, metric, optimizer, step, trigger, action, logger, schedule, plot, sampler, strategy or value.
* **category**: the group a short name is listed under by `kalfa legos` and the terminal interface, such as `losses`, `lr schedules` or `table preprocessing`, with the place in a config where its names go (see [kalfa legos](#kalfa-legos)).
* **slot key**: a key kalfa reads itself instead of passing it to the class, such as `outputs` on a model or `minimizes` on an optimizer.
* **reference**: a string that starts with `@` and names another built object, `"@models.net"`.
* **node**: one built object of the plan, addressed by its path (`models.net`, `losses.total`, `rules.stop.when`).
* **plan**: the checked, resolved form of a config, written as `plan.json`.
* **record**: the directory a run writes. Every tool reads records and nothing else.
* **turn**: one unit of training. With `train.epochs` a turn is an epoch; with `train.updates` it is a fixed number of updates.
* **update**: one optimizer step. With `accumulate: k` an update takes k batches.
* **step**: the algorithm object that trains a turn (`supervised`, `alternating` or your own class). It is not a counter.
* **set**: a named part of the data: `train`, `valid` and `test` for the std data classes. Series of the valid set use the prefix `val`.
* **series**: a named number per turn, written to `history.jsonl`: `train/mse`, `val/rmse`, `lr/main`, `track/optimizers.main.lr`.
* **objective**: a loss that reads everything (`forward(outputs, batch, ctx)`), a subclass of `kalfa.Objective`.
* **criterion**: a loss that compares one prediction with one target (`function(pred, target)`); kalfa wraps it and picks the two tensors.
* **predicts model**: the model whose outputs the step computes for each batch and whose outputs go into the predictions table.
* **rule**: a condition (`when`) and its effects (`set`, `stop`, `run`), checked after every turn.
* **trigger**: the object behind `when`; it answers `fires(ctx)`.
* **action**: the object behind `run`; it is called with the run context.
* **patch**: one change a rule makes to a running config, such as `optimizers.main.lr` times 0.5.
* **sweep root**: a directory with `sweep.json` whose numbered subdirectories (`0000`, `0001`, ...) are the records of the sweep points.

## The config grammar

The top level keys are `include`, `params`, `plugins`, `seed`, `device`, `record`, `data`, `models`, `losses`, `metrics`, `optimizers`, `train`, `rules`, `plots`, `generate` and `sweep`. Any other key is the problem `unknown_section`.

### Values

A value is one of these:

* a scalar: a number, a string, a boolean or null;
* a param, `$name$`: replaced by the value of `params.name`. A string that is only `$name$` takes the value with its type (a list stays a list); inside a longer string it is formatted as text (`runs/lr_$lr$`). `$name.field$` reads a field of a mapping param. `$datetime$` is built in and gives the start time as `20261108_143000`. `$$` writes a literal dollar sign;
* a call, `{use: ..., ...}`;
* a reference, `"@models.net"` (quote it, since YAML reserves a leading `@`);
* a list or a mapping of these.

Substitution happens after every layer is merged. An unknown `$name$` is the error `unknown_param`, a param that refers to itself is `param_cycle`, and a param that nothing reads is a warning (an error when it came from `-p`).

### Calls

`{use: <name>, <param>: <value>, ...}` builds the target of `use` with the params. Where a section expects a call, a bare name is a call without params: `mse: mse` is the same as `mse: {use: mse}`.

The params are checked against the signature of the target before anything is built: a key the target does not take is `unknown_key`, a required param that is missing is `missing_key`. When the signature cannot be read, or the target takes `**kwargs`, the params pass and the plan writes `signature: open` for that node. A call nested in a param is built first and passed as the object (`schedule: {use: cosine, steps: 1000}`).

A plain function can be a loss. When `use` under `losses` names a function whose first two positional params are required (`log_cosh(pred, target)`), the function itself is the criterion and the other params are bound to it. A function without required params is a factory: kalfa calls it and uses what it returns.

### Short names and import paths

A name without a dot is looked up in the catalog; a name with a dot is imported. The directories of the config files and the working directory are put on `sys.path`, so a module next to the config (`mylegos.py`) imports as `mylegos.Thing`. The catalog is one name space across slots: `mae` is the metric and the loss is `l1` (or `torch.nn.L1Loss`); `sequential` is the split and the layer stack model is `stack`. A misspelt name gets a suggestion (`did you mean to_tensor`).

### References

A reference is a string that starts with `@`. It is resolved to the built object when the node that holds it is built (and, in a rule patch, when the rule fires).

| reference | resolves to |
|---|---|
| `@models.<name>` | the model itself, the plain `nn.Module` |
| `@models.<name>.ema` | the EMA copy of the model (needs the `ema` key) |
| `@models.<name>.<child>.<child>` | a submodule, such as `@models.ae.decoder` |
| `@losses.<name>` | the objective, or the wrapped criterion |
| `@losses.<name>.<attribute>` | an attribute of a loss, such as `@losses.ddpm.schedule` |
| `@metrics.<name>` | the metric entry |
| `@optimizers.<name>` | the optimizer entry (the torch optimizer is its `real` attribute) |
| `@data.<attribute>` | a value the data class exposes after it is fitted, such as `@data.n_classes` |
| `@generate` | the sampler of the `generate` section |
| `@train.step` | the step object |

`@data.<attribute>` accepts only the names in the `exposes` list of the data class. Inside one section the order of the entries does not matter (`@losses.mse` may be used above `mse`); across sections the build order holds, so a loss can name a model but a model cannot name a loss. A reference that names nothing is `reference_unknown`.

### Tensor names are plain text

Outputs and batch fields are named with plain strings, never with references. `pred: x_hat` names an output of the step (the names come from the model's `outputs`), and `target: image` names a field of the batch (the names come from the data class: `x` and the targets of a table, `image` and `label` of an image folder, `input_ids` and `targets` of a text file).

### Layers: include, params, --set and -p

A config is merged from layers, lowest first:

1. the files named by `include` (a path or a list of paths, relative to the file that includes them); an included file may include others;
2. the config files in the order given (and Python mappings after them);
3. `--set path=value` overrides, one leaf each (`--set train.epochs=3`, `--set data.batch=32`). The path is dotted and takes no list index. The value is parsed as YAML, and a value that starts with `@` stays a reference;
4. `-p name=value` params (`-p lr=3e-4`).

The same key twice inside one layer is an error (`duplicate_key`). Across layers a later leaf wins. A mapping with `use` and a list are replaced as a whole, never merged item by item; other mappings merge key by key. `params` mappings merge, the later layer winning. `resolved.yaml` in the record shows the merged config with a comment beside every leaf that a layer overrode.

`plugins` is a module name or a `.py` file, or a list of them, imported by `check` before anything is built. A `.py` file is looked for in the working directory and the config directories. Plugins register short names with `kalfa.lego`; a plain import path needs no plugin entry.

`seed` (default 0) seeds every random stream and the splits. `device` is `auto` (the default: cuda, then mps, then cpu), `cpu`, `mps`, `cuda` or `cuda:<index>`. `record` is the record directory; without it the run writes `runs/<config name>_<datetime>`. A record directory must be new or empty.

### Slot keys

Slot keys are read by kalfa and never passed to the target. A slot key that is also a param of the target is the error `slot_conflict`, since the class could never receive it from a config. The step slot is the exception: a step constructor receives the slot keys it takes (`supervised` takes `order`, `amp`, `grad_clip` and `accumulate`).

| slot | slot keys | fixed during a run |
|---|---|---|
| model | `inputs` (batch fields fed in order, default the data class `inputs`), `outputs` (names of the results, default the model name), `ema` (a decay, `true`, or `{decay, every}`), `freeze` (`true`, `false` or a list of parameter name patterns), `weights` (`{run, model, which}`), `init` (a call applied to the module) | `inputs`, `outputs`, `ema`, `weights`, `init` and every constructor param |
| loss, metric | `pred`, `target`, `on` (the sets it runs on), `every` (every k turns), `rescale`; a loss also takes `sample_weight` (a row weight field of the data) | `pred`, `target`, `rescale`, `sample_weight` and the `fixed` list of the class |
| optimizer | `over` (what it optimizes), `minimizes` (the loss), `schedule` (an lr factor per update) | `over`, `schedule` |
| step | `order`, `amp`, `grad_clip`, `accumulate` (and the params of the class, such as `repeat` and `fresh_batch` of `alternating`) | the `fixed` list of the class |

The last column lists what a rule cannot patch; everything else of these slots can change while the run goes on (see [rules](#rules)).

### The five defaults

kalfa fills in exactly these, and nothing else:

1. With one model, `train.predicts` is that model. With several, nothing is predicted unless `train.predicts` names one.
2. The `pred` of a criterion or metric is the first output of the predicts model, and its `target` is the only target of the data. Otherwise `check` asks for them (`pred_required`, `target_required`).
3. `rescale` is on for a metric when the target has an invertible preprocessor (a scaler), so metrics speak the original units; it is off for a loss.
4. `train.step` is `supervised`, and `train.evaluate` is the existing `valid` and `test` sets.
5. The seed of a split is the top level `seed`.

The filled `pred`, `target` and metric `rescale` and the default step are written into `plan.json`; `train.predicts`, `train.evaluate` and `seed` stay there as the config wrote them.

## Sections

| key | what it holds |
|---|---|
| `include` | files merged below this one |
| `params` | values read as `$name$` |
| `plugins` | modules imported before the build |
| `seed` | the seed of every random stream (default 0) |
| `device` | `auto`, `cpu`, `mps`, `cuda`, `cuda:<index>` |
| `record` | the record directory |
| `data` | one call: the data class |
| `models` | name: call of an `nn.Module` |
| `losses` | name: call of an objective or a criterion |
| `metrics` | name: call of a metric |
| `optimizers` | name: call of a torch optimizer |
| `train` | the budget, the step and what to keep |
| `rules` | name: rule |
| `plots` | name: call of a plot |
| `generate` | one call: the sampler |
| `sweep` | the strategy, the space and the objective of a sweep |

### data

| data class | short name | batch fields | exposes |
|---|---|---|---|
| a parquet or csv table | `table` | `x` (float32 features), every target by name and every weight of `weights` by name | `n_features`, `feature_names`, `target_names`, `n_classes`, `class_weights`, `classes` |
| a folder of class folders of images | `image_folder` | `image` (float `(C, H, W)`), `label` (int64) | `n_classes`, `classes`, `class_weights`, `image_shape` |
| a text file of lines | `text_lines` | `input_ids`, `targets` (the next tokens) | `vocab_size`, `tokenizer`, `seq_len` |
| your own class | its import path | what its loaders yield | its `exposes` list |

A relative data path is looked for in the working directory first, then in the directories of the config files, so `file: housing.parquet` next to the config works from anywhere.

#### table

```yaml
data:
  use: table
  file: housing.parquet
  split: [0.7, 0.15, 0.15]
  features: {"x*": standard_scaler, size: [log, minmax_scaler]}
  targets: {price: standard_scaler}
  filter: {all: "price > 0", train: "size < 500"}
  drop: [notes]
  spectators: [id]
  batch: {size: 128, eval_size: 512}
```

* `file`: a `.parquet`, `.pq`, `.csv` or `.tsv` table.
* `split`: required. A list of ratios is a shuffled split into train, valid and test (`[0.8, 0.2]` makes no test set). A call picks another split: `{use: random, ratios: [...], seed}`, `{use: sequential, ratios: [...], group: <column>}` (rows in their order, within each group when `group` names a column), `{use: kfold, folds: 5, fold: 0, seed, valid_ratio: 0.1}` (fold `fold` is the test set, `valid_ratio` carves a valid share from the rest) or `{use: given, column: <column>}` (the column says `train`, `valid` or `test` for every row; the three labels can be renamed with the `train`, `valid` and `test` params).
* `features`: a mapping of column name patterns (fnmatch, `x*`) to a preprocessor, a list of preprocessors applied in order, or null for the raw column. A literal column name wins over a pattern, and a more specific pattern wins over a looser one. Columns that are targets, spectators, weights or the feed group are never features; a column no pattern matches is dropped with a warning.
* `targets`: a mapping of target columns to a preprocessor, a list, or null. A `label` preprocessor makes a classification target: its batch field is int64, its classes are saved, and `class_weights` (inverse frequency, mean one) is exposed.
* Preprocessors follow the scikit learn protocol (`fit`, `transform`, `inverse_transform`) and are fitted on the train rows only. The std ones are `standard_scaler`, `minmax_scaler`, `robust_scaler`, `one_hot` (one column per category, named `<column>_<category>`), `label`, `abs`, `log {base, norm}` and `cast {dtype}`; any import path works too.
* `filter`: a pandas query kept for every row before the split, or a mapping of `all` (before the split) and `train`, `valid`, `test` (each set after the split) to queries. The filters are saved, so `kalfa predict --data` applies the `all` filter to new rows too.
* `drop`: columns removed after the split and the filters, before the features are chosen.
* `weights`: row weights for a loss, a mapping of a batch field name to a weight column of the table (`w: event_weight`) or to a mapping of conditions to weights (`w: {"price < 5": 5, "5 <= price < 15": 12}`). A condition is a pandas expression over the raw columns, as in `filter`, evaluated before the preprocessors and before `drop`, so it speaks the original units and may read any column. The first condition a row meets wins, and a row that meets none weighs 1. The weights are divided by their mean over the train set (a streamed table takes the mean of its fit rows), so they average one there and the size of the loss stays comparable; the mean must be positive. The field is a float32 tensor with one value per row (with a window feed, the rows of the horizon), and a loss reads it with `sample_weight`. A weight field is never a feature, cannot be named `x` or like a target, and `kalfa predict --data` does not compute it for new rows.
* `spectators`: columns (or patterns) that are not fed to the model but are copied into the predictions table, such as an id.
* `batch`: a size (default 128), or a mapping of `size`, `eval_size` (the batch of the valid and test sets, default `size`), `workers` (DataLoader workers, default 0), `drop_last` and `shuffle` (the train set, default true).
* `feed`: turns the rows into samples another way, see the window feed below.
* `stream`: reads a large table in pieces, see below.

Predictions of a table come back in the original units of every target, with the class names of a label target.

#### The window feed

```yaml
data:
  use: table
  file: energy.parquet
  split: {use: sequential, ratios: [0.7, 0.15, 0.15], group: site}
  features: {"x*": standard_scaler}
  targets: {load: standard_scaler}
  feed: {use: window, size: 24, horizon: 6, stride: 1, context: true, group: site}
```

A window takes the features of `size` consecutive rows as `x` (a `(size, features)` tensor) and every target over the next `horizon` rows (`(horizon,)` for one column). `stride` skips start rows. `group` names a column whose value every window keeps to (one series per site); that column is not a feature. With `context: true` the first windows of the valid and test sets reach back into the rows of the previous set, so no rows are lost at the borders. The context reaches only into the set just before (a test window looks back into the valid rows, never into train), so a group gives test windows only when its test rows number at least `horizon` and its valid and test rows together at least `size + horizon`. The predictions table writes one column per horizon step: `load_0`, `load_1`, ... and `pred_<output>_0`, `pred_<output>_1`, ..., on the test set and on new rows given to `kalfa predict --data` alike (the truth of step i is the target at row `row + i` of the same group); the `forecast_samples` plot draws them, and the board pairs `pred_<output>_<i>` with `load_<i>`, one pair per step.

#### Streaming a large table

`stream: true` or `stream: {chunk: 65536, fit_rows: 100000}` reads the train set chunk by chunk instead of loading the file. It needs a `sequential` or `given` split (a stream cuts the file in order); the preprocessors are fitted on the first `fit_rows` train rows; rows are shuffled within each chunk; the valid and test sets are loaded whole. A stream does not combine with a feed or a grouped sequential split.

#### image_folder

```yaml
data:
  use: image_folder
  folder: data/pets
  split: [0.7, 0.15, 0.15]
  transform: [{use: resize, size: 32}, to_tensor, {use: normalize, mean: imagenet}]
  train_transform: [{use: random_crop, size: 32, padding: 2}, random_flip, to_tensor, {use: normalize, mean: imagenet}]
  batch: 64
```

Every subdirectory of `folder` is a class and its images are the samples. `transform` applies to every set and `train_transform` replaces it for the train set. A transform list holds short names, calls or import paths, each a function from image to image (PIL image or tensor): `resize {size}`, `center_crop {size}`, `random_crop {size, padding}`, `random_flip {p}`, `grayscale`, `to_tensor` (float `(C, H, W)` in [0, 1]), `normalize {mean, std}` (numbers, lists or the presets `imagenet`, `cifar10`, `mnist`) and `two_views {each}` (applies the `each` list twice with fresh randomness and stacks the results as `(2, C, H, W)`, for contrastive learning). `labels: false` keeps the class folders as a spectator column but gives no target (autoencoders); a folder without class folders has no target either. `extensions` lists the image suffixes. The batch default is 64.

#### text_lines

```yaml
data:
  use: text_lines
  file: data/lines.txt
  tokenizer: char
  seq_len: 64
  split: [0.9, 0.05, 0.05]
  batch: 32
```

The file is read line by line (empty lines are skipped) and the lines are split into sets. `tokenizer: char` builds a character vocabulary from the train lines (id 0 is unknown); any object with `encode`, `decode`, `size` and `fit` works too. Each set is cut into windows of `seq_len + 1` tokens with `stride` (default `seq_len`): `input_ids` are the first `seq_len` and `targets` the next token at each place. `lowercase: true` lowercases the text. A text file writes no predictions table; `lm_sampler` writes text instead.

### models

```yaml
models:
  net:
    use: mlp
    hidden: [64, 64]
    out: 1
    init: {use: xavier_init}
    ema: {decay: 0.999, every: 1}
  teacher:
    use: mylegos.Classifier
    n_classes: "@data.n_classes"
    weights: {run: runs/teacher, model: net, which: best}
    freeze: true
```

A model is any `nn.Module`. kalfa calls it with the batch fields listed in `inputs`, in that order (default: the `inputs` of the data class, `x` for a table), and names what it returns: a tensor takes the first name of `outputs` (default the model name), a tuple is matched with `outputs` in order, and a dict is used as it is. These names are what `pred` refers to.

* `init`: a call that receives the module, such as `{use: normal_init, std: 0.02}`; a bare name is the call without params, so `init: xavier_init` is `init: {use: xavier_init}` (the same holds for the `schedule` of an optimizer). Lazy layers are filled on their first forward pass.
* `weights`: loads a model from an earlier record: `run` is the record directory (or a bare string), `model` the model name there (default this name), `which` the checkpoint (`best`, `last` or `final`; default `last`).
* `freeze`: `true` freezes every parameter and keeps the module in eval mode; a list of patterns (`["backbone.*"]`) freezes the matching parameter names and keeps fully frozen submodules in eval mode; `false` unfreezes. A rule may patch `freeze`.
* `ema`: keeps an exponential moving average copy that follows every optimizer step whose `over` covers the model. A number is the decay; `{decay, every}` updates it every k steps. Name the copy with `@models.<name>.ema`, for a sampler or a metric.

The build order is `init`, `weights`, `freeze`, `ema`. The std models are `mlp {hidden, out, activation, dropout, in_features}` and `stack {layers}`, a list of layer calls and short names (`linear`, `conv2d`, `batch_norm`, `layer_norm`, `embedding`, `dropout`, `flatten`, `relu`, `gelu`, `tanh`, `sigmoid`, `silu`, `last_step`) or import paths:

```yaml
models:
  net:
    use: stack
    layers:
      - {use: torch.nn.GRU, input_size: 2, hidden_size: 32, batch_first: true}
      - last_step
      - {use: linear, out_features: 6}
```

### losses

A loss is either a criterion or an objective.

* A **criterion** compares one prediction with one target: any callable `(pred, target) -> tensor`, such as `mse`, `l1`, `huber`, `bce_logits`, `cross_entropy`, `poisson_nll`, `torch.nn.SmoothL1Loss` or a plain function. kalfa picks the output named by `pred` and the batch field named by `target` (see the defaults); a trailing dimension of one is squeezed to match. With `rescale: true` both are first turned back into original units.
* An **objective** reads everything: a `kalfa.Objective` with `forward(outputs, batch, ctx)` that returns a tensor or a dict with a `loss` key and more terms. The std objectives are `weighted_sum`, `mdmm`, `vae_elbo`, `wgan_gp_d`, `wgan_g`, `ddpm`, `distill` and `ntxent`.

```yaml
losses:
  mse: mse
  l1: l1
  rec: {use: mse, pred: x_hat, target: image}
  ce: {use: cross_entropy, weight: "@data.class_weights", label_smoothing: 0.1}
  total:
    use: weighted_sum
    terms:
      - {of: "@losses.mse", weight: 1.0}
      - {of: "@losses.l1", weight: 0.1}
```

Every loss is a series on every set it runs on: `train/<loss>`, `val/<loss>`, `test/<loss>`, and one more series per extra term of a dict (`train/total/mse`, `train/elbo/kl`). A loss that an optimizer minimizes is averaged over the batches of the turn; the others observe without gradients. `on: [train]` limits a loss to some sets and `every: 2` computes it every second turn. `weighted_sum` sums `{of, weight, term}` entries (`term` picks a key of a dict result). `mdmm` adds constraints to an objective with one Lagrange multiplier each: `{use: mdmm, objective: "@losses.mse", constraints: [{of: "@losses.l1", epsilon: 0.05}]}`; its `lmbda` parameter goes into an optimizer group with a negative lr (`{of: "@losses.total", lr: -0.05}`).

A criterion weighs its rows with `sample_weight`, the name of a weight field that `data.weights` puts into the batch:

```yaml
data:
  use: table
  targets: {price: standard_scaler}
  weights:
    w: {"price < 5": 5, "5 <= price < 15": 12}
losses:
  mse: {use: mse, sample_weight: w}
```

The criterion then computes one loss per element, multiplies the loss of every row by its weight and takes the mean (the sum with `reduction: sum`), on every set it runs on. It needs a loss with a `reduction` param, which the torch losses and the std ones have; a plain function of `(pred, target)` that returns one value is refused. The weight has one value per row and spreads over the other dimensions of the loss. A weighted criterion works inside `weighted_sum`, `mdmm` and `vae_elbo` like any other. Metrics take no `sample_weight` and stay unweighted, and an objective reads the weight field from the batch itself.

### metrics

A metric is any object with `update(pred, target)`, `compute()` and `reset()` (a torchmetrics metric works), or a `kalfa.ContextMetric` with `update(outputs, batch, ctx)` that reads everything. The std metrics are `rmse`, `mae`, `accuracy`, `f1`, `auroc`, `average_precision`, `perplexity` and `fid`.

```yaml
metrics:
  rmse: rmse
  f1: {use: f1, num_classes: "@data.n_classes", average: macro}
  fid: {use: fid, model: "@models.generator.ema", latent: 64, real: image, n: 256, every: 5}
```

A metric is accumulated over the whole set and written as `<set>/<metric>`. It takes the same `pred`, `target`, `on`, `every` and `rescale` keys as a loss; `rescale` is on by default when the target can be inverted. A loss and a metric with the same name write the same series, and `check` warns (`series_clash`).

### optimizers

```yaml
optimizers:
  main:
    use: adamw
    lr: 1.0e-3
    weight_decay: 1.0e-2
    over:
      - {of: "@models.net", match: "backbone.*", lr: 1.0e-4}
      - "@models.net"
    minimizes: "@losses.ce"
    schedule: {use: cosine, steps: 5000, warmup: 200}
```

`use` is a torch optimizer class (`adam`, `adamw`, `sgd` or any import path) and the other params go to it. `over` names what it optimizes: a reference, a list of references, or groups `{of: <reference>, match: <pattern>, <torch params>}`. A group takes the parameters of `of` whose names (`named_parameters`) match the fnmatch pattern and that no earlier group took, with its own torch params; plain references take whatever is left. `over` may name objectives too (the multipliers of `mdmm`). `minimizes` is the loss the optimizer steps on; std steps require it (`minimizes_required`). `schedule` is a callable from the update count to a factor of the base lr: `warmup {steps}`, `cosine {steps, warmup, floor}`, `linear_warmup {start, end, steps}`, `step_decay {every, gamma}` or your own.

Every update runs `zero_grad`, `backward` (through the grad scaler under AMP on cuda), gradient clipping, `step`, the schedule and the EMA copies. The history writes `lr/<optimizer>` (the first torch param group), `lr/<optimizer>/<group>` for every group of `over` (named `<model>[<match>]`, such as `lr/main/net[backbone.*]`), and `minimizes/<optimizer>` (the name of the loss it minimizes now); `updates.jsonl` writes `loss/<optimizer>`, `lr/<optimizer>` and `grad_norm/<optimizer>` per update.

### train

| key | value |
|---|---|
| `epochs` | the number of turns, each a pass over the train set |
| `updates` | `{total, turn}`: the training counts updates, `turn` updates per turn (instead of `epochs`) |
| `step` | the step call, default `supervised` |
| `predicts` | `"@models.<name>"`, or `false` for a run whose model makes no predictions (a DDPM denoiser) |
| `evaluate` | the sets evaluated after every turn, default the existing `valid` and `test` |
| `checkpoint` | `{watch, mode, last, every}`: `best.pt` when the `watch` series improves (`mode` min or max), `last.pt` every turn unless `last: false`, `turn_<k>.pt` every `every` turns; `final.pt` is always written at the end |
| `report` | `best` or `last` (default): the weights the final predictions, plots and samples use; `best` needs a `watch` |
| `track` | plan paths whose values become series every turn: `[optimizers.main.lr, "losses.total.constraints[0].epsilon"]` writes `track/<path>` (quote a path with an index inside brackets) |
| `log` | loggers: `[tensorboard]` writes TensorBoard event files to `<record>/tensorboard`, `[{use: echo, series: [val/rmse]}]` prints series per turn |

Either `epochs` or `updates` is required, never both.

### rules

```yaml
rules:
  warm:
    when: {use: after_epoch, at: 5}
    set: {optimizers.main.lr: {times: 0.5}}
  to_huber:
    after: warm
    when: {use: below, watch: train/mse, value: 0.05}
    set: {optimizers.main.minimizes: "@losses.huber"}
  samples:
    when: {use: every, epochs: 2}
    once: false
    run: {use: write_samples, n: 16}
  stop:
    when: {use: plateau, watch: val/rmse, patience: 8}
    stop: true
```

Rules are checked after every turn, after the evaluation and before the checkpoints, in the order of the config. A rule fires when it is ready and its trigger fires; without `when` it fires as soon as it is ready.

* `when`: a trigger call.
* `after`: the name of another rule; this rule is asked only from the turn after that rule fired.
* `once`: true by default, so a rule fires once. With `once: false` it may fire every turn, and its trigger is reset after each firing.
* `set`: a mapping of plan paths to patch values.
* `stop: true`: the run ends after this turn, with its final checkpoint, predictions and plots.
* `run`: an action call, such as `write_samples`.

The turns a rule fired are in the `rules` list of every history line and on the board.

#### Triggers

| trigger | params | fires |
|---|---|---|
| `after_epoch` | `at` | from the turn whose number reaches `at` (absolute, so a resumed run counts on) |
| `every` | `epochs` | every `epochs` turns |
| `plateau` | `watch`, `patience`, `mode` (min), `min_delta` (0) | after `patience` turns without the series improving by `min_delta`; turns without a value are not counted |
| `above` | `watch`, `value` | when the series is above `value` |
| `below` | `watch`, `value` | when the series is below `value` |
| `time_budget` | `minutes` | once `minutes` have passed since the first turn that asked it |

`watch` names a series of the turn, such as `val/rmse`, `train/total/kl` or `track/optimizers.main.lr`.

#### Patches

A path is dotted with optional list indexes: `optimizers.main.lr`, `losses.total.terms[1].weight`, `losses.total.constraints[0].epsilon`, `rules.stop.when.patience`. Its root is `models`, `losses`, `metrics`, `optimizers`, `train` or `rules`. A value is one of:

* a plain value: a number, a string, a boolean or a list (`models.net.freeze: false`);
* `{times: x}`: the current value times x;
* `{plus: x}`: the current value plus x;
* a reference, `"@losses.huber"`, resolved when the rule fires.

What may be patched: the torch params of an optimizer (lr, weight decay, betas and the rest) and its `minimizes`; the `freeze` of a model; the constructor params of losses, metrics, the step and triggers (except their `fixed` lists) and the `on` and `every` of losses and metrics; `train.epochs` (or `train.updates.total`). `check` tests every rule path against this list (`not_mutable`, `unknown_path`). A patch of the optimizer lr sets the base value the schedule multiplies. A patch of `train.epochs` changes the length of the running training.

Every patch is applied to the live object and to `plan.json`, and written to `events.jsonl` as `{"kind": "patch", "turn", "by", "path", "value", "asked"}` with the resolved value; a resumed run replays these events (see [resume](#checkpoints-resume-and-patch-replay)).

### plots

```yaml
plots:
  loss_curve: {use: loss_curve, series: [train/total, val/rmse]}
  pred_vs_true: pred_vs_true
  confusion_matrix: confusion_matrix
```

Plots are drawn at the end of a run into `<record>/plots`, from the record files (`history.jsonl`, `predictions.parquet`, `samples/`), and again at any time with `kalfa plots RECORD`. A std plot saves its image under its own name (`loss_curve.png`, `pred_vs_true.png`), so two entries of the same std plot write the same file; a plot of your own that returns its figure is saved under its key in the section. The std plots are `loss_curve {series}`, `pred_vs_true {pairs}`, `confusion_matrix {pred, target}`, `class_histogram {pred, target, bins}`, `binary_roc`, `binary_precision_recall_curve`, `forecast_samples {n, pred, target}`, `image_pairs {n, set}` (images above the reconstructions of the predicts model; it needs the run, so it draws during a run or through `kalfa plots`) and `samples_grid {n, file}`.

### generate

```yaml
generate: {use: latent_sampler, model: "@models.ae.decoder", latent: 8, n: 16}
```

One sampler call. At the end of a run kalfa draws `n` samples into `<record>/samples/samples.pt`, with a PNG grid for images and a `.txt` for text, and `kalfa generate RECORD` draws more later. The std samplers are `gan_sampler {model, latent, n_classes, n}`, `latent_sampler {model, latent, n}`, `ddpm_sampler {model, schedule, shape, n, steps}` and `lm_sampler {model, tokenizer, prompt, max_new_tokens, temperature, top_k, n}`. The action `write_samples {sampler, n}` writes `samples/turn_<k>` from a rule, with the generate sampler when it names none.

### sweep

```yaml
params:
  lr: 1.0e-3
  width: 64
sweep:
  strategy: grid
  space:
    lr: [1.0e-2, 1.0e-3]
    width: [32, 64, 128]
  objective: {watch: val/rmse, mode: min, at: best}
  record: runs/sweep_lr_width
```

A sweep runs the config once per point, each point with other values of some params.

* `strategy`: `grid {steps}` (every combination; a range needs `steps`), `random {trials, seed}` (the catalog calls it `random_search`; the seed defaults to the config seed), `optuna {trials, sampler: tpe | random, seed}` (asks the points one after another from an optuna study), or the import path of your own strategy (`points(space, seed)` for a fixed list, or `ask(space, objective)` and `tell(trial, value)` for an adaptive one).
* `space`: param names of the `params` section, each a list of values, `{float: [a, b]}`, `{log: [a, b]}` (log uniform, or geometric steps in a grid) or `{int: [a, b]}`.
* `objective`: `watch` (a series), `mode` (min or max), `at` (`best` over the turns of a point, or its `last` value).
* `record`: the sweep root.

The root holds `sweep.json`, `points.jsonl`, `sweep.sh` (one `kalfa sweep ... --point <i>` line per point), `sweep.sub` (an HTCondor submit file), `logs/` for points run with `--jobs`, and the point records `0000`, `0001` and so on. `kalfa summarize ROOT`, the sweep page of the board and `kalfa summarize ROOT --html` rank the points.

## The runtime

### Build order

`data` is built and fitted (preprocessors and vocabularies learn from the train set), then `models` (the `@data.*` references resolve here; then `init`, `weights`, `freeze`, `ema`), `losses`, `metrics`, `optimizers`, `train.step`, `train.log`, `plots`, `generate`, and last the rules with their triggers and actions. An object with a `bind(ctx)` method gets it called once built. Each object then receives `name` and `path` attributes, unless its constructor takes params of those names. The models, the objectives and the metrics are moved to the device.

### The loop

Before the first turn the record already holds `fitted/` and `data.json`, and the loop writes `model_summary.json`, so the board shows the data flowchart and the model diagrams of a running record.

Every turn:

1. `turn` grows by one;
2. the step trains the turn (`train_turn`);
3. every set of `train.evaluate` is evaluated: losses and metrics that run on it (`on`, `every`) are computed without gradients, the losses as batch weighted means;
4. the series of the turn are gathered: `train/<loss>`, `train/<loss>/<term>`, `val/<metric>`, `test/...`, `track/<path>`;
5. the rules are asked in order;
6. the checkpoints are written;
7. the history line is written (with `lr/*`, `minimizes/*`, `seconds` and the rules that fired), the loggers and the progress line follow;
8. a `stop.json` in the record ends the run after this turn.

While the run is alive a background thread renews `updated` in the manifest every ten seconds, whatever the loop is doing (a long turn, the evaluation, the predictions, the plots), so a record reads as `stale` only when its process is gone. With it the manifest carries `within`, where the turn is: the loop notes the phase (`train`, the set being evaluated, then `after`, `predictions`, `samples` and `plots` at the end) and, while it trains or evaluates, how many batches of how many are done (updates of the turn when the unit is `updates`). The readers turn that into the fraction of the run that is done and the time left: the mean of the last five turns times the turns left, or the pace of the first turn while no turn has ended.

The loop stops when the planned turns are done or a stop was asked. Then it writes `checkpoints/final.pt` and `model_summary.json`, loads the reported weights (`train.report`), writes `predictions.parquet` for the test set and every extra set of the data class (in original units: `row`, the targets, `pred_<output>`, `raw_<output>`, `prob_<output>_<class>` for classes, the spectators), draws the samples and the plots, and marks the manifest `done` (or `stopped`). An output that pairs with no target becomes `pred_<output>` (one value per row) or `pred_<output>_<k>` columns when it has at most 32 values per row; a wider one (an image, a long embedding) stays out of the table.

### Steps

A step object trains a turn. The protocol (`kalfa.Step`):

* `train_turn(ctx)`: trains one turn and returns the train series. The default walks the train loader, calls `train_batch` for each batch, writes the update lines and computes the observer losses and train metrics;
* `train_batch(batch, ctx) -> {optimizer name: loss value}`;
* `eval_batch(batch, ctx) -> outputs` for evaluation;
* `predict(batch, ctx) -> outputs` for the predictions table;
* attributes `order`, `amp`, `grad_clip`, `accumulate` (set from the slot keys), `fixed` and `persist` (attribute names saved in checkpoints).

The two std steps:

* `supervised {order, amp, grad_clip, accumulate}`: for every batch, each optimizer in `order` (default the config order) computes the outputs of the predicts model (when there is one), evaluates the loss it minimizes and steps. `accumulate: k` sums the gradients of k batches into one update.
* `alternating {order, repeat, fresh_batch, amp, grad_clip, accumulate}`: per round each optimizer in `order` takes `repeat[name]` steps (`repeat: {critic: 5, generator: 1}`), on the shared batch of the round or, with `fresh_batch: true`, on a new batch per step. It is the step of GANs.

### The context

Every hook receives the run context `ctx`. Its attributes: `config`, `tree`, `plan`, `settings` (the raw `train` mapping), `train` (the parsed train settings), `data`, `models` (name to model entry, whose `module` is the `nn.Module`), `losses`, `metrics`, `optimizers` (name to optimizer entry), `step`, `rules`, `plots`, `generate`, `loggers`, `device`, `rng`, `seed`, `record` (the record writer, or None), `history` (the history lines so far), `turn`, `updates`, `batches`, `set_name` (the set being walked), `latest` (the series of this turn), `fired` (the rules that fired this turn), `stopping`, `monitor`, `logger` (a `logging` logger), `best`, `patches`, `cursor`, `last_norms`, `rules_by_name`, `fitted_from` and `want_sets`. Its methods: `predicts()` (the predicts model entry or None), `loaders()` (set name to DataLoader), `set_modes(train)`, `stop(reason)` and `elapsed()` (seconds since the start).

A model entry calls its module on a batch with `entry.call(batch)` and returns the named outputs. An optimizer entry has `minimize(loss, clip)`, `zero_grad()`, `backward(loss)`, `clip(limit)`, `step()`, `rate()` and `set_param(name, value)`; its torch optimizer is `entry.real`. The record writer has `path(name)`, `write_json(name, value)`, `write_text(name, text)`, `append(name, line)` (one JSON line) and `event(kind, turn, **fields)`.

### Named random streams

Randomness comes from named streams so that turning a feature on or off does not shift the randomness of another: `ctx.rng.normal("noise", shape, like=x)`, `ctx.rng.uniform(name, shape, like=x)`, `ctx.rng.integers(name, low, high, shape, device=...)`, `ctx.rng.permutation(name, count)`, and `ctx.rng.stream(name, device)` for a `torch.Generator`. Each stream is seeded with the sha256 of `<seed>:<name>`. The std pieces use `noise` (training objectives), `samples` (samplers), `fid` (the fid metric) and `loader` (the shuffling of the data). At the start of a run the global python, numpy and torch generators are seeded too, and the stream states go into every checkpoint.

### Devices and mixed precision

`ctx.device.torch` is the `torch.device`; batches are moved to it before the step sees them. `amp: true` on the step runs the forward passes under `torch.autocast` on cuda and mps (it does nothing on cpu), with a gradient scaler on cuda.

### Checkpoints, resume and patch replay

A checkpoint holds `turn`, `update`, the state of every stateful node (`models.<name>`, `models.<name>.ema`, `losses.<name>`, `metrics.<name>`, `optimizers.<name>` with the torch state, the update count and the grad scaler, `train.step`, `rules.<name>` with its trigger), the random stream states, the number of patches and the best watched value.

`kalfa run --from RECORD` continues a record from its `last.pt` (or `final.pt`) into a new directory (`<record>_<datetime>`, or `--record DIR`, which must be empty). It loads the config of the source (`resolved.yaml`, or the config files given), applies `--set` and `-p` on top (`--set train.epochs=40` trains longer), restores the checkpoint, replays the patch events of the source up to the checkpoint turn, copies the history lines, `best.pt` and the `plugins/` folder, and goes on from the next turn. The new manifest names the source in `resumed_from` and lists the config files of the source after its own, so relative data paths still resolve when the resumed record is opened later.

Opening a record (`predict`, `generate`, `export`, `plots`, `kalfa.open(...).predict()`) never fits the data again: the fitted preprocessors and vocabularies are loaded from `fitted/`, the sets are rebuilt with the same seed, and the weights come from the reported checkpoint.

### Stopping a run

A run ends early in three ways: a rule with `stop: true`; `kalfa stop RECORD` (or the stop button of the board), which writes `stop.json` that the loop reads after the turn; ctrl-c, which ends the run after the turn (a second ctrl-c aborts at once). Each way still writes the final checkpoint, the predictions and the plots, and the manifest says `stopped`.

## The record directory

```text
runs/housing_20261108_143000/
  manifest.json
  resolved.yaml
  plan.json
  history.jsonl
  updates.jsonl
  events.jsonl
  log.txt
  model_summary.json
  data.json
  fitted/
  checkpoints/
  predictions.parquet
  plots/
  samples/
  exports/
  plugins/
```

* `manifest.json`: `format`, `kalfa` (the version), `name`, `started`, `updated` (renewed every ten seconds while the process lives, see [the loop](#the-loop)), `finished`, `status` (`running`, `done`, `failed` or `stopped`; a running record whose `updated` is older than two minutes reads as `stale`), `within` (only while it runs: `{turn, phase, done, total, unit, started}`, the turn in progress and how far it is), `config_files` (the files given and every included layer; opening the record looks for relative data paths and plugins in their directories), `params`, `unit` (`epochs` or `updates`), `turns_planned`, `turn`, `host`, `device`, `git` (`commit` and `dirty`, or null), `sweep` (`root`, `id` and `values` for a sweep point), `resumed_from`, `failure` (`type`, `message`, `traceback`).
* `resolved.yaml`: the merged and substituted config, with a comment beside every leaf a layer overrode. `kalfa check RECORD` and `kalfa run --from RECORD` read it.
* `plan.json`: every node with its slot, import path, params, slot keys, fixed params and signature state; the data inputs, exposed names and fields; the train settings; the rules; the series; the mutable paths. References are written as `{"ref": "models.net"}` or `{"ref": "models.ae", "attr": "decoder"}`. Patches update it.
* `history.jsonl`: one line per turn with `turn`, `update`, the series (`<set>/<name>`, `<set>/<name>/<term>`, `lr/<opt>`, `lr/<opt>/<group>`, `minimizes/<opt>`, `track/<path>`), `seconds` and `rules`.
* `updates.jsonl`: one line per update with `update`, `turn`, `loss/<opt>`, `lr/<opt>` and `grad_norm/<opt>`.
* `events.jsonl`: one line per event with `at`, `turn` and `kind`: `run_start`, `turn`, `patch`, `checkpoint`, `stop`, `resume`, `error` and `run_end`.
* `log.txt`: what the run said, every line with its time: the host, the process id and the device, one line per turn with its first values and the rules that fired, every warning kalfa logged (and its info lines when the command line shows them), a stop that was asked, the end, and for a failed run the error with its whole traceback. The board shows its tail on the log tab and the terminal interface with `l`.
* `model_summary.json`: per model its `class`, `parameters`, `trainable`, the module `tree` with parameter counts, the `outputs` shapes on the first batch and `graph`, the diagram of one forward pass. The run writes it before its first turn and again at the end. The first batch comes from the valid set, else the test set, else the train set, and taking it leaves the randomness of the run untouched: the torch, numpy and python generators and the named streams, the `loader` stream of the train set among them, are restored. `graph` is `{"method": "eager", "truncated", "nodes", "edges", "groups"}`: kalfa calls the model once without gradients on the batch fields it reads, with a hook on every module and a `TorchFunctionMode` for the torch calls between modules. A leaf module (one without submodules) is a `module` node: `label` is the class name, `path` and `detail` the module path, `class` the full class path, and `parameters` its count; a module called twice is two nodes. A module with submodules, such as a `Sequential` or a block of yours, is a group around the nodes it ran (the root model is not a group). A torch function called outside the leaf modules on a traced tensor (`cat`, `add`, `matmul`, `softmax`, `scaled_dot_product_attention`, functional activations and the like) is an `op` node, while reshaping and moving calls (`view`, `reshape`, `permute`, `flatten`, `squeeze`, `to`, indexing and the like) only pass their input on. `input` nodes carry the field names and `output` nodes the output names, the edges follow the tensors, and every `shape` is per sample, the batch size left out. Past 300 module and op nodes the rest is left out and `truncated` is true. A model that a loss calls with inputs of its own, such as the denoiser of `ddpm` (called with the noised image and the step `t`) or a GAN generator (called with noise `z`), cannot run on the batch fields. kalfa keeps a copy of the inputs of each model's first call in training and traces the model on them instead: the summary written after the first turn and the one at the end carry its diagram, the `input` nodes are named after the parameters of the model's `forward`, and `graph_from` is `"first training call"` (it is `"first batch"` otherwise). When neither works, `graph` is null and `graph_note` says why.
* `data.json`: `{"format": 1, "kind", "report", "pipeline"}`. `kind` is the short name of the data class (or its import path), `report` what the data reports about itself (the rows of each set, the columns with their chains, windows, classes, the image shape, lines and tokens) and `pipeline` the flowchart of the data tab: `nodes` (`id`, `kind`, `label`, `detail`, `count`, `group`), `edges` (`from`, `to`, `label`) and `groups` (`id`, `label`, `detail`, `parent`). A table has two lanes that meet at the batch, or at the window feed: the rows run from the source through the filter and the split into the sets, and the columns run from the source past the dropped columns through each feature pattern and each target and its fitted chain into the `x` and target tensors. A streamed table has a stream node and cuts the file by position before it filters, so its filter comes after the split. An image folder shows its source, classes, split, sets, the `transform` and `train_transform` chains and the image and label tensors; text lines show the source, the split, the sets (counted in windows, with their lines and tokens in the detail), the tokenizer, the windows and the tensors. A set node counts the rows the report gives for that set. The run writes the file after `fitted/` and again when it resumes; when the flowchart cannot be drawn, the run logs a warning and writes `pipeline: null`. A data class of yours without a `pipeline()` method gets a generic flowchart (see [a data class](#a-data-class)).
* `fitted/`: what the data learned, so a record opens without fitting again: `preprocessors.joblib` and `fields.json` for a table (features, targets, classes, filters, spectators, the scales of the weights), `fields.json` for an image folder, `tokenizer.json` for text.
* `checkpoints/`: `last.pt`, `best.pt`, `turn_<k>.pt` and `final.pt`.
* `predictions.parquet`: the test set in original units; `kalfa predict` adds `predictions_<best|last|final>.parquet`, `predictions_<data file>.parquet` and a `_<model>` suffix for another model.
* `plots/`: one image per plot.
* `samples/`: `samples.pt` (with `.png` or `.txt`), `turn_<k>.*` from `write_samples`, `generate_<which>.*` from `kalfa generate`.
* `exports/`: `<model>_<which>.pt` or `.onnx` from `kalfa export`.
* `plugins/`: copies of the modules under the config directories that the run imported, so the record opens on another machine.
* `tensorboard/`: event files of the tensorboard logger.
* `stop.json`: written by `kalfa stop`, `{"by", "at"}`.

## Commands

Commands print their results (problems, tables, the paths they wrote) on stdout, and progress lines and errors on stderr. The exit code is 0 on success, 1 when the config has problems, 2 for any other error (and for a sweep with a failed point), 130 when aborted with ctrl-c. `kalfa --version` prints the version.

### kalfa check

`kalfa check CONFIG... [--set PATH=VALUE] [-p NAME=VALUE] [--plan]`

Loads and merges the configs, imports the plugins, builds the static plan and prints every problem as `error <code>: <path>: <message> (<hint>)`, or `ok`. Nothing is built or trained. A record directory instead of a file checks its `resolved.yaml`, with its `plugins/` folder and the directories of its config files searched for plugins and data (`kalfa run` and `kalfa sweep` treat a record directory the same way). `resolved.yaml` keeps the `record` key of the config with its `$datetime$` already filled in, so `kalfa run RECORD` on a record written at its config's own `record` path aims at that record again and stops because it is not empty: name a new directory with `--record DIR` (in the config tab of the terminal interface, `record=DIR` in the `--set` field). `--plan` prints `plan.json` after the problems. Warnings do not fail the check.

The problem codes: `yaml_error`, `duplicate_key`, `file_not_found`, `not_a_mapping`, `include_cycle`, `bad_include`, `bad_params`, `bad_plugins`, `plugin_import`, `unknown_param`, `param_cycle`, `unused_param`, `bad_set`, `bad_param`, `unknown_section`, `bad_section`, `not_a_call`, `bad_key`, `unknown_use`, `import_failed`, `unknown_key`, `missing_key`, `slot_conflict`, `reference_unknown`, `reference_expected`, `minimizes_required`, `minimizes_ref`, `predicts_ref`, `pred_required`, `target_required`, `unknown_output`, `unknown_weight`, `weight_objective`, `series_clash`, `split_required`, `data_file_missing`, `data_value`, `device_value`, `train_key`, `train_unit`, `train_report`, `report_best`, `train_track`, `train_log`, `track_path`, `checkpoint_key`, `checkpoint_value`, `checkpoint_watch`, `watch_unknown`, `unknown_series`, `evaluate_set`, `rules_type`, `rule_type`, `rule_key`, `rule_empty`, `rule_after`, `rule_ref`, `rule_set`, `rule_path`, `rule_value`, `unknown_path`, `not_mutable`, `generate_required`, `sweep_key`, `sweep_space`, `sweep_strategy`, `sweep_objective`, `sweep_param`, `sweep_record`.

### kalfa run

`kalfa run [CONFIG...] [--set PATH=VALUE] [-p NAME=VALUE] [--record DIR] [--from RECORD] [--quiet]`

Checks the config, trains into the record directory and prints it. The record directory is announced first and repeated on failure. One progress line per turn goes to stderr, with the build and data notes; `--quiet` hides them. `--record DIR` overrides the `record` of the config. A record directory in place of CONFIG trains its `resolved.yaml` again. When the record directory a run would write into already holds a run (a record given in place of CONFIG, or a config whose `record` has no `$datetime$`, such as the fold records of `11_kfold` or the teacher of `13_distillation`), kalfa writes into `<record>_<datetime>` instead and says so on stderr; nothing is overwritten. `--from RECORD` continues that record from its last checkpoint into a new directory (see resume); configs given with it replace its `resolved.yaml`.

### kalfa predict

`kalfa predict RECORD [--data FILE] [--model NAME] [--which best|last|final] [--device D]`

Opens the record and predicts. Without `--data` it predicts the test set again into `predictions_<which>.parquet`. `--data` takes a `.parquet`, `.pq`, `.csv` or `.tsv` table with the raw columns, or whatever the data class reads (a folder of images, an image, a text file), and writes `predictions_<name>.parquet`. `--model` predicts with another model than `train.predicts` and adds `_<model>` to the file name. `--which` picks the weights (default: `best` when `train.report` is best, else `last`).

### kalfa summarize

`kalfa summarize RECORD... [--by SERIES] [--mode min|max] [--top K] [--out DIR] [--markdown] [--columns NAMES] [--html] [--points IDS]`

Prints one row per record, best first, and the best row. A path may be a record, a sweep root (its points) or a directory with records under it. The rows rank by `--by` (default the checkpoint `watch` of the first record) in `--mode` (default its mode, or min). The table holds `name`, `status`, `turn`, `total`, `seconds`, every param, `best`, `best_turn` and the last value of every series as `last_<series>`; the printed text table hides the train and lr series unless `--columns all` or a list of names, while `--markdown` and the `summary.md` of `--out` show every column. Records with a `fold` param are grouped by their other params, and after the best row every group prints the mean and the standard deviation of each last value over its folds. `--markdown` prints a markdown table, `--out DIR` writes `summary.csv` and `summary.md` (and `folds.csv`, one row per group and series, when there are fold groups), and `--html` writes a single file `report.html` (summary, curves, fold groups, sweep points and param scatters, and per record a section for each prediction pair drawn for its task; no external resources) into the first record or sweep root, or to `--out` (a file, or a directory for it); `--points` limits the details of a sweep report to some point ids.

### kalfa generate

`kalfa generate RECORD [--n N] [--which best|last|final] [--device D] [--out DIR]`

Opens the record with its `generate` sampler and writes `n` samples (default the sampler's `n`) to `<record>/samples/generate_<which>.pt`, with a PNG grid for images or a `.txt` for text. `--out` writes elsewhere.

### kalfa plots

`kalfa plots RECORD [--only NAMES] [--which best|last|final] [--device D]`

Draws the plots of the record's `plots` section again into `<record>/plots` and prints the files. `--only` takes comma separated plot names; `--which` picks the weights for plots that need the run, such as `image_pairs`.

### kalfa export

`kalfa export RECORD [--format state_dict|onnx] [--model NAME] [--which best|last|final] [--out PATH] [--device D] [--opset N] [--static]`

Writes a model of the record: `state_dict` (default) saves the parameters with `torch.save`, `onnx` traces the model on a test batch (opset 17 by default, a dynamic batch axis unless `--static`). The default file is `<record>/exports/<model>_<which>.pt` or `.onnx`; `--out` names a file or a directory.

### kalfa sweep

`kalfa sweep CONFIG... [--set PATH=VALUE] [-p NAME=VALUE] [--record ROOT] [--count] [--show N] [--plan] [--point N] [--repair] [--jobs K] [--quiet]`

Runs the points of the `sweep` section one after another and prints the root. `--count` prints the number of points, `--show N` the first N, `--plan` writes the plan files (`sweep.json`, `points.jsonl`, `sweep.sh`, `sweep.sub`) and exits, for a cluster. `--point N` runs one point (what each line of `sweep.sh` and each job of `sweep.sub` does). `--repair` runs only the failed, stopped or missing points. `--jobs K` runs K points at once in child processes, their output in `<root>/logs/<id>.log`. A failed point does not stop the others. `--record ROOT` overrides `sweep.record`.

### kalfa stop

`kalfa stop RECORD...`

Writes `stop.json` into each running record; the run stops after its current turn and still writes its final files. A sweep root stops its running points. A finished record is left alone with a warning.

### kalfa legos

`kalfa legos [NAME|WORD] [--slot SLOT] [--docs]`

Without arguments lists every short name with its one line description under its category; the categories follow the order of the config sections, and a name that a plugin registered is marked `(plugin)`. An exact name prints its page: target, slot, category, `goes in` (where the name sits in a config, for a category that has a place) and description; the params with their defaults (`hidden (required)`, `activation = 'relu'`), each with its note indented below it; `examples:`, YAML pieces that show the name in its place in a config (an optimizer under `optimizers:`, a preprocessor under `data.features`, a trigger under `rules.<name>.when`); and `used in:`, the configs under `examples/` that use the name as a lego (in a `use:`, in a short form such as `losses: {mse: mse}`, as a feature, target, split or transform, or in a lego param such as `activation`, `schedule` or `when`; tensor names do not count). On a terminal the page is in color, with the YAML highlighted; piped or redirected it is plain text. Another word searches the names and descriptions and lists the hits under their categories. `--slot` keeps one slot (`data`, `model`, `loss`, `metric`, `optimizer`, `schedule`, `step`, `trigger`, `action`, `logger`, `sampler`, `plot`, `strategy`, `value`), still listed by category. `--docs` prints the whole catalog as markdown: a `##` section per category with a line on where its names go, and under it a `###` entry per name with its target, description, params with their notes, examples in `yaml` blocks and the configs that use it.

The categories, in their order, and where their names go:

| category | members | goes in |
|---|---|---|
| data | slot `data` | `data` |
| splits | target under `kalfa.std.data.splits` | `data.split` |
| table preprocessing | target under `kalfa.std.data.pre` or `sklearn.preprocessing` | `data.features` and `data.targets` |
| feeds | target under `kalfa.std.data.window` | `data.feed` |
| image transforms | target under `kalfa.std.data.transforms` | `data.transform` and `data.train_transform` |
| tokenizers | `char_tokenizer` | `data.tokenizer` |
| models | slot `model` | `models.<name>` |
| layers | `linear`, `conv2d`, `batch_norm`, `layer_norm`, `dropout`, `embedding`, `flatten`, `last_step` | layers of a `stack` model |
| activations | `relu`, `gelu`, `silu`, `sigmoid`, `tanh` | the activation of `mlp` or layers of `stack` |
| init | target under `kalfa.std.init` | `models.<name>.init` |
| losses | slot `loss`, apart from the objectives | `losses.<name>` |
| objectives | `vae_elbo`, `wgan_gp_d`, `wgan_g`, `ddpm`, `distill`, `ntxent`, `mdmm`, `weighted_sum` | `losses.<name>` |
| noise schedules | `cosine_beta`, `linear_beta` | the schedule of `ddpm` |
| metrics | slot `metric` | `metrics.<name>` |
| optimizers | slot `optimizer` | `optimizers.<name>` |
| lr schedules | slot `schedule` | `optimizers.<name>.schedule` |
| steps | slot `step` | `train.step` |
| triggers | slot `trigger` | `rules.<name>.when` |
| actions | slot `action` | `rules.<name>.run` |
| samplers | slot `sampler` | `generate` |
| plots | slot `plot` | `plots.<name>` |
| loggers | slot `logger` | `train.log` |
| sweep strategies | slot `strategy` | `sweep.strategy` |
| other | every other name | no fixed place |

A std name is placed by its name first (layers, activations, objectives, noise schedules), then by the module of its target, then by its slot. A plugin name is placed by its slot alone and falls into `other` when its slot has no category. Empty categories are not shown.

### kalfa board

`kalfa board [ROOT] [--host HOST] [--port PORT] [--open]`

Serves the board over the records under `ROOT` (default the working directory) at http://127.0.0.1:8765/; `--open` opens the browser. See [the board](#the-board).

### kalfa tui

`kalfa tui [ROOT] [--config CONFIG]`

Opens the terminal interface over the records under `ROOT`; `--config` loads a config into its config tab, which the guide tab then explains. See [the terminal interface](#the-terminal-interface).

## Python API

```python
import kalfa

checked = kalfa.check("config.yaml", sets=["train.epochs=3"], params=["lr=3e-4"])
print(checked.ok, checked.text())

result = kalfa.run("config.yaml", sets=["train.epochs=3"], record="runs/try", monitor=True)
record = kalfa.open(result.record)
print(record.state, record.best)
print(record.series(["train/mse", "val/rmse"]))
table = record.predict(data="new.parquet")
```

* `kalfa.call(use, **params)` returns the call mapping `{"use": use, **params}`.
* `kalfa.Run(**sections)` holds a config as Python objects, one keyword per section. `run.to_mapping()` gives the plain mapping, `run.check(sets=(), params=())` returns the check result, `run.start(record=None, sets=(), params=(), monitor=None)` trains.
* `kalfa.Rule(when=None, after=None, once=True, set={}, stop=False, run=None)` and `kalfa.Train(epochs, updates, step, predicts, evaluate, checkpoint, report, track, log)` are dataclasses for the `rules` entries and the `train` section of a `Run`.
* `kalfa.check(source, sets=(), params=())`: `source` is a path, a list of paths or a mapping. It returns a check result with `ok`, `problems`, `errors`, `warnings`, `text()`, `plan` and `config`.
* `kalfa.run(source, sets=(), params=(), record=None, monitor=None)` trains and returns a result with `record` (the directory), `history` (the lines) and `context`. `sets` and `params` are lists of `"path=value"` strings or `(path, value)` pairs, or one mapping (`sets={"train.epochs": 3}`). `monitor=True` prints the progress lines, `None` or `False` is silent, and a `kalfa.Monitor(stream=None, quiet=False)` writes them to a stream.
* `kalfa.open(directory)` returns a `kalfa.Record`, which reads the files of a record, live or finished:
  * properties: `manifest`, `plan`, `summary` (name, status, unit, turn, total, times, seconds, params, best, last values, rules that fired, device, host, failure, sweep, resumed_from, and `within`, `fraction`, `eta` and `nonfinite`: the turn in progress, the share of the run that is done, the seconds left, and the first turn with a value that is not finite as `{turn, series, value}`), `state` (`running`, `stale`, `done`, `failed`, `stopped`, `unknown`), `progress`, `best` (`{series, mode, value, turn}`), `history` and `updates` (DataFrames of the raw files), `events`, `files`, `model_summary`, `data_report` (`data.json`: the data report and its flowchart, or None for an older record), `config` (the `resolved.yaml` text), `fields` (`fitted/fields.json`), `predictions` (a DataFrame or None);
  * methods: `series(keys=None, limit=None)` and `steps(keys=None, limit=None)` (numeric DataFrames of the history and the updates, thinned to `limit` rows), `patches()`, `stats(where=None, **options)` (the prediction statistics the board shows: `rows`, one entry in `pairs` per prediction with its `task` and the numbers of that task, and the `overview`; `name`, `bins`, `sample`, `worst` and `overrides` pass through, see [prediction tasks](#prediction-tasks)), `compare(*others, keys=None)`, `predict(data=None, model=None, which=None, device=None)` (a DataFrame), `generate(n=None, which=None, device=None, out=None)`, `plots(only=None, which=None, device=None)` (the plot names drawn), `export(format="state_dict", model=None, which=None, out=None, **options)`, `stop(by="api")` and `resume(sets=(), params=())`.
* `kalfa.api.records(root)` returns a `Record` for every record under a root (the points of a sweep included).
* `kalfa.summarize(paths, by=None, mode=None, top=None)` returns a summary with `table` (a DataFrame), `best` (its first row), `by`, `mode`, `records`, `folds`, `sweep` and `points`; `kalfa.summary.text_of(summary)`, `kalfa.summary.markdown_of(summary)` and `kalfa.summary.write(summary, out)` print or save it.
* `kalfa.predict(directory, data=None, model=None, which=None, device=None)` writes the predictions file and returns its `path` and `table`.
* `kalfa.api.generate(directory, n=None, which=None, device=None, out=None)` returns the written `paths` and the `samples`.
* `kalfa.api.export(directory, format="state_dict", model=None, which=None, out=None, device=None, opset=17, dynamic=True)` returns the written `path`.
* `kalfa.api.stop(directories, by="api")` writes `stop.json` files and returns their paths.
* `kalfa.api.sweep(source, sets=(), params=(), record=None, jobs=1, repair=False, monitor=None)` runs a sweep locally and returns one result per point (None for a failed point).
* `kalfa.api.report(paths, out=None, by=None, mode=None, top=None, points=None)` writes `report.html` and returns its path.
* `kalfa.lego(name, *, slot=None, doc="")` registers a short name (see [plugins](#kalfalego-and-plugins)); `kalfa.register(name, target, *, slot=None, doc="")` does the same without a decorator (`slot` and `doc` are keyword only) and `kalfa.resolve(name)` returns what the name was registered with: an import path string for a std name, the object itself for a `kalfa.lego` name.
* Errors: `kalfa.KalfaError` is the base; `ConfigError`, `BuildError` and `RunError` derive from it; a config with errors raises `kalfa.core.errors.ProblemsError`, whose `problems` are `kalfa.Problem(code, message, path, hint, severity)` values.
* `kalfa.records` is the package of record readers (`summary_of`, `series_of`, `events_of`, `records_under`, `sweep_points` and more) that every tool shares.

## The board

`uv run kalfa board runs --open` serves a page over every record and sweep under `runs` and follows them while they run. The page is a Vue and Plotly application served from the kalfa package; the two libraries load from cdn.jsdelivr.net, so the browser needs internet access. Every number the page shows is computed in Python from the record files; the browser only draws. Every chart opens large with a settings panel (see [charts](#charts)).

Pages:

* **home**: the running records first, then the recent ones, with progress and the best value. A running record moves within its turn too: the bar counts the batches done, and the card says where the turn is (`batch 120 of 400`, `valid batch 3 of 9`) and about how long is left. A record whose history holds a value that is not finite says which series and from which turn.
* **runs**: the summary table of every record and sweep, sortable and filterable, with failed and stale counts; select records to compare them.
* **compare**: the series of the selected records drawn over each other, the params that differ, and the `resolved.yaml` diff.
* **sweep**: the points ranked by the objective, the best point, and a button that writes `report.html`. The points can be ticked (or the best 5, 10 or 20 taken at once) and compared like records. Three kinds of chart follow: the objective by point in the order the sweep made them with the best value so far, which says whether the search still finds anything; every point as one line across the params and the objective (a parallel coordinates chart, the color is the objective with the best bright, a drag along an axis keeps a range; points without an objective are left out and counted); and one scatter of the objective per param.
* **record**: one record in nine tabs:
  1. **overview**: progress with the batches of the turn in progress and the time left, best value, last values, params, rules that fired, identity (host, device, git), a stop button, and for a failed record the error with its traceback;
  2. **curves**: every series against the turn, and the update curves (`loss/*`, `lr/*`, `grad_norm/*`) below, with a log axis switch. A dotted line with the names of the rules marks every turn a rule fired, on the turn charts and at the same update on the update charts (up to twelve marks carry their names; the rule marks button hides them);
  3. **model**: every model with its class, parameter and trainable counts and three views: the diagram of one forward pass (see [the model diagram](#the-model-diagram)), the module tree with the share of the parameters each module holds, and the output shapes on the first batch and per sample. A record written before kalfa drew diagrams, or a model it could not trace, opens on the tree and says why;
  4. **data**: the flowchart of `data.json` (see [the data flowchart](#the-data-flowchart)) with the details of the box clicked, the sets with their rows (and windows, lines or tokens when the data reports them), the other report fields and the columns that go into the model; below them the data section of the plan and the fitted fields (columns, preprocessor chains, classes). A record without `data.json` shows the plan and the fitted fields only;
  5. **predictions**: the prediction tables of the record scored on the server, every `pred_` column read as a task with the panels of that task (see [prediction tasks](#prediction-tasks));
  6. **plots**: the images under `plots/` and `samples/`;
  7. **config**: `resolved.yaml`;
  8. **files**: every file of the record and a text viewer that follows the tail of a growing file;
  9. **log**: the last 300 lines of `log.txt`, followed while the record runs, then the events as a timeline and the tail of `events.jsonl`.

Live updates: the server looks at the records (the manifest, the history, the events and the updates of each) every 2 seconds (1, 2, 5, 10 or 30, chosen in the top bar and remembered by the browser) and pushes the changed paths over server sent events; only the page in view reloads its data. The top bar also switches the theme (auto follows the system). The server keeps the summary of a record until one of those files changes, so a page over hundreds of records reads again only the ones that moved.

Security: the board binds 127.0.0.1 by default and has no login, so only the local machine reaches it. Requests whose `Host` header is not localhost, an IP address, the `--host` value or the machine name are refused (against DNS rebinding); requests that change something (stop, report) or carry a `where` filter must come from the board page itself (`Sec-Fetch-Site` and `Origin` checks). A `where` filter accepts only column names, constants, comparisons, boolean and arithmetic operators and `in` over lists. File reads cannot leave the root, and the image route serves only image files that the `path` column of a prediction table names. `--host 0.0.0.0` exposes every record under the root, and the stop button, to the network: use it only on a network you trust, or reach a remote board through an SSH tunnel instead (`ssh -L 8765:127.0.0.1:8765 host`).

The JSON behind the page is plain HTTP and can be scripted: `GET /api/records`, `GET /api/record/<path>` (and `/series?keys=a,b&limit=n`, `/updates`, `/events?kinds=a,b`, `/files`, `/file?name=<file>&tail=n`, `/predictions?where=&sample=&bins=&name=&pair=&task=&score=&positive=&threshold=&histograms=`, `/image?index=n&name=<table>`), `GET /api/compare?records=a,b&keys=`, `GET /api/sweep/<path>`, `POST /api/record/<path>/stop` (the record answer carries `failure` with the traceback and `marks`, the turns a rule fired; the sweep answer carries `order`, the objective and the best so far by point, and `space`, the axes of the parallel chart), `POST /api/report` with `{"records": [...], "out": null}` and `GET /api/report/<job>`, and `GET /api/live` as an event stream.

### Charts

Every chart has an expand button that opens it large with a settings panel beside it (below it on a narrow window); the settings button hides the panel, and escape or a click outside closes the view. The panel sets:

* axes: the x and y scale (auto, linear or log) and range (an empty end follows the data), the axis titles and the grid;
* lines and markers: the line width, the line style (solid, dashed, dotted), lines, markers or both, the marker size, the opacity and, for line charts, smoothing from 0 to 0.99 (an exponential moving average; the raw line stays faint and the hover shows both values); for a histogram the number of bins, which the server counts again (each histogram of the predictions tab keeps its own count, see [prediction tasks](#prediction-tasks));
* look: a title drawn on the chart, the legend (auto, on or off) and its place (top, bottom, right, inside) and the palette (theme, colorblind, classic, viridis);
* the width and height of the large view, and PNG or SVG downloads at a chosen width, height and scale;
* advanced: Plotly layout JSON, merged over the layout of the chart when it parses (the box shows the parse error otherwise).

Controls that do not fit a chart are hidden: a heatmap has no scales or ranges, a bar chart none on its category axis. The settings apply to the small chart and the large view alike and stay in the browser under `kalfa.chart.<key>`. The key is the chart kind and title, so a `train/mse` chart keeps its settings from record to record, while the predictions tab keys its charts by record and pair. Only the values that differ from the defaults are stored, a dot on expand marks a chart with saved settings, and reset clears them. The log y button of a card writes the same y scale setting. Zoom and hidden legend entries survive the live redraws.

### The model diagram

The model tab draws the `graph` of `model_summary.json` from left to right (TB turns it from top to bottom): a box for every group, and on each node its label in bold, its detail faint, its per sample shape as `[16]` and its parameter count abbreviated in the corner. Inputs, modules, operations and outputs have their own colors, and a folded group has a dashed border. A click on a group header folds the group into one node that gathers its edges, and a click on the folded node opens it again; with more than 60 nodes the top groups start folded. A click on a node shows its path, class, detail, shape and parameters, the group it sits in, and the nodes it takes from and feeds with the shapes on those edges. The toolbar fits the diagram into view, switches between LR and TB, expands or collapses every group and downloads the diagram as an SVG file; the wheel zooms and a drag pans. A model traced on its first call in training, such as a diffusion denoiser or a GAN generator, carries the chip "drawn from its first call in training"; while such a run is in its first turn, its tab still shows the tree.

### The data flowchart

The data tab draws the `pipeline` of `data.json` with the same diagram: one color per node kind (source, stream, filter, drop, split, set, classes, features, targets, weights, chain, transform, tokenizer, windows, feed, tensor, batch), the count in the corner (rows read, rows per set, columns per feature pattern, images, classes, windows, the vocabulary size) and the split shares on the edges into the sets. Chains of preprocessors and transforms are groups; with more than 40 nodes they start folded. A click on a box lists its kind, detail, count and group and what arrives on its edges.

### Prediction tasks

The predictions tab scores one prediction table at a time (a selector picks among the `predictions*.parquet` files of the record) on the server; a `where` filter, a pandas query over the columns such as `price > 200 and town == 'ank'`, narrows the rows that are scored, but never changes the task or the threshold, which are found on the whole table.

Pairing: a `pred_<output>` column pairs with the `target` its loss or metric names, else with the only declared target of the data; a `pred_<output>_<i>` column pairs only with a `<target>_<i>` column, so an embedding is never compared with a spectator such as `path`. A column without a target is kept: the spread columns `pred_z_0`, `pred_z_1`, ... form one embedding pair, and a single column is a values pair.

The task of a pair with a target is detected in this order:

1. when its classes are known (from the plan or from `fitted/`), when `prob_<output>_<class>` columns exist or when the target is not numeric, it is a classification: binary with two classes, multiclass with more;
2. a numeric target with two distinct values is binary, and the larger value is the positive class;
3. a numeric target of whole numbers with 3 to 20 distinct values, whose predictions all fall in that set, is multiclass;
4. anything else is a regression.

Labels are always text, and whole numbers are written without decimals (`"0"`, `"1"`). The task selector offers what a pair can be read as, with auto naming the detected task: a label target with two classes binary or multiclass, a numeric target with two values binary or regression, a numeric multiclass target multiclass or regression; every other pair has one task.

* **binary**: the score is the probability column of the positive class (`prob_<output>_<positive>`) when there is one, else the first candidate: one minus the probability of the other class (the selector names it after `prob_<output>_<other>`), any other `prob_` column as it is, then `raw_<output>` and `pred_<output>` when they are numeric with more than two distinct values. Such a number is read as growing with the default positive class (the larger value, or the second class); with the other class picked as positive it is turned around too, as one minus `pred_<output>` when it lies in [0, 1] and as its negative (`-pred_<output>`) otherwise, so the curves keep their meaning. A row is predicted positive when its score reaches the threshold. The threshold is 0.5 when every score lies in [0, 1], else the threshold with the best f1 on the table (an anomaly score, such as the one of `alad`), and the slider sets it by hand. Without a score the predicted labels are counted and there are no curves. The panels: accuracy, precision, recall, f1, specificity, balanced accuracy, ROC AUC and average precision; the confusion matrix as counts or as the share of each true class; the ROC and precision recall curves; the score histogram of each true class with the threshold line; and a sweep of precision, recall and f1 over at most 101 thresholds. Two selectors pick the score column and the positive class.
* **multiclass**: accuracy, macro and weighted f1, the confusion matrix as counts or shares, precision, recall and f1 per class as bars and as a table, and an ROC curve per class against the rest for the classes with a `prob_` column.
* **regression**: rmse, mae, r2 and bias, prediction against truth with the y = x line, the residual histogram, the histograms of truth and prediction, the residual quantiles and the worst rows.
* **embedding**: a PCA map of the rows (at most 2000 of them), colored by the `label` column, else by a target, else by a text column with fewer than 20 values, and the histogram of the vector lengths.
* **values**: the histogram and the quantiles, or the count of each label for a text column.

When a table has several pairs (the targets of a multi target model, or several outputs), an overview table on top lists them with their task and one headline number: r2 (or rmse) for a regression, ROC AUC (or f1 without a score) for a binary pair, accuracy for a multiclass pair, the variance share of the two PCA components for an embedding and the row count for values. A click on a row, or on a button of the pair bar below it, opens that pair.

When the table has a `path` column of image files (png, jpg, jpeg, gif, bmp, webp), as the predictions of an image folder do, every pair carries a gallery of at most 24 rows with the image, the truth, the prediction and the score or the error: the wrong predictions first for a classification, the largest errors first for a regression, the first rows otherwise. The images come from `/api/record/<path>/image?index=n&name=<table>`, which reads the `path` value of that row and serves it only when it names an image file.

The choices of a pair (task, score, positive class, threshold) stay in the browser for each record, table and pair under `kalfa.predictions.<record>|<table>|<pair>`, and back to auto clears them. From Python, `record.stats(overrides={"pred_net": {"task": "multiclass"}})` or `{"pred_score": {"threshold": 3.5}}` applies the same choices (the `threshold_rule` of a given threshold is `given`); a task the pair cannot take, a score that is not a candidate or a positive label that is not one of the labels is an error.

Bins: every histogram of the tab keeps its own number of bins in its chart settings, and the page asks the server to count again only the histograms of the selected pair whose count was set; the others use 40 bins. The histograms are `residual_hist` and `truth_hist` for a regression (truth and prediction share their edges, so the count of `truth_hist` sets both), `score_hist` for a binary pair (its score chart is drawn as lines and still has the bins box), `norm_hist` for an embedding and `hist` for values; a multiclass pair has none. From Python, `record.stats(overrides={"pred_net": {"bins": {"residual_hist": 60}}})` sets one count, and `bins` (default 40) sets every histogram the override does not name. A count below 1 is an error, a count above 1000 is cut to 1000, and a name that the task of the pair does not draw is an error that lists the names it draws. Over HTTP the page sends `histograms=residual_hist:60,truth_hist:30` together with `pair`.

## The terminal interface

`uv run kalfa tui runs` opens a Textual interface with four tabs; `1`, `2`, `3` and `4` switch between them, escape leaves an input field and `q` quits.

* **records**: a table of the records under the root (name, status, turn of total, best), with a filter field and a line that says how many rows are shown and by what they are sorted. Selecting one shows its summary, params, last values and a sparkline per series; a running record also says where its turn is and about how long is left, a record with a value that is not finite names the series and the turn, and a failed one ends with the last lines of its traceback. Keys: `/` goes to the filter (words, each found in the name, the status or a `param=value`; enter or escape goes back to the table), `t` sorts by the next of started, name, status, turn and best (a click on a column header sorts by it, a second click turns it around), enter on a sweep lists its points and backspace comes back, `p` predict, `g` generate, `o` plots, `u` resume (`kalfa run --from` the record, into a new directory beside it; a running record is refused), `l` prints the last 40 lines of `log.txt`, `s` stop, `b` open the board on the root, `r` refresh. The table also refreshes every ten seconds, reading the records in a thread of its own and keeping the summary of every record that did not change, so the keys stay responsive over many records.
* **legos**: a search field above a tree of the categories of [kalfa legos](#kalfa-legos) with their counts (`losses (6)`), and the page of the highlighted entry beside it. Every category starts closed; enter opens it, and its leaves show each name with its one line description. A search keeps the names whose name or description holds the word, opens the categories that have hits and puts the cursor on the first hit; enter or escape in the search field goes to the tree. A highlighted name shows its page: its category and where it goes in a config, the notes of its params, its YAML examples highlighted and the example configs that use it. A highlighted category shows where its names go and lists them.
* **config**: a config path, `--set` and `-p` fields and an output pane. Keys: `c` check, `p` check and print the plan, `x` run, `w` sweep. Runs and sweeps start as child processes whose output streams into the pane; they keep running when the interface closes, while a board started with `b` closes with it.
* **guide**: the config of the config tab (its path, `--set` and `-p` fields, read when the tab opens and again with `r`; a record directory reads its `resolved.yaml` as in the config tab) shown as a document. On the left is the list of its sections in the order of the config, with the top level values (`seed`, `device`, `record`, `include`, `plugins`) gathered into one `run` section; choosing a section scrolls to it. On the right every section has a title bar in its own color, a paragraph on what kalfa does with it and its entries: every param, every top level value, every key of `data`, `train`, `generate` and `sweep`, and every name of `models`, `losses`, `metrics`, `optimizers`, `plots` and `rules`, each with its YAML highlighted and dim notes below it. The notes say what a name is (its one line description, or the class or function an import path names), what the params and slot keys written there do, what each reference and each `$param$` stands for, what kalfa fills in when a key is not written (the output and the target a loss or a metric compares, whether a metric is rescaled), which optimizer minimizes a loss and which rule switches to it, who reads a model or a loss, the series an entry writes, what a rule does in one sentence, and the layer a value came from (another file, `--set` or `-p`). Each problem of `kalfa check` stands under its entry, yellow for a warning and red for an error; one that matches no entry stands under its section, and one without a path at the top. Nothing is built and no data is read. From Python, `kalfa.commands.guide.document(paths, sets=(), params=())` returns the same document as a mapping (title, ok, problems and the sections with their entries), and `guide.text(doc)` writes it as plain text.

## Writing your own pieces

Every piece below is a plain Python class or function in a module next to the config, named by its import path (`pieces.TwoHeads`). The samples form one module, `pieces.py`: the config under [Putting the pieces together](#putting-the-pieces-together) uses them on the housing table, the data class comes with its own config, and the tests of kalfa train both.

### A model

Any `nn.Module`. The constructor params are the keys of the call; `forward` receives the batch fields named by `inputs`.

```python
from torch import nn


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
models:
  net: {use: pieces.TwoHeads, width: 64, outputs: [mean, spread]}
```

### A criterion function

A function whose first two params are the prediction and the target; the other params are bound from the config.

```python
def scaled_l1(pred, target, scale=1.0):
    return ((pred - target).abs() / scale).mean()
```

```yaml
losses:
  l1: {use: pieces.scaled_l1, scale: 2.0}
```

### An Objective

A `kalfa.Objective` reads the outputs, the batch and the context, and returns a tensor or a dict with `loss` and more terms; each extra term becomes a series (`train/nll/sigma`). Constructor params named like the loss slot keys (`pred`, `target`, `on`, `every`, `rescale`, `sample_weight`) never reach the class from a config (`slot_conflict`), so give them other names.

```python
import torch

import kalfa


class GaussianNll(kalfa.Objective):
    def __init__(self, field="price", floor=1.0e-3):
        super().__init__()
        self.field = field
        self.floor = float(floor)

    def forward(self, outputs, batch, ctx):
        mean = outputs["mean"]
        sigma = torch.nn.functional.softplus(outputs["spread"]) + self.floor
        truth = batch[self.field].view_as(mean)
        nll = (torch.log(sigma) + 0.5 * ((truth - mean) / sigma) ** 2).mean()
        return {"loss": nll, "sigma": sigma.mean()}
```

An objective that holds models gets them as references (`generator: "@models.generator"`), and its own parameters (such as the multipliers of `mdmm`) can be optimized by naming it in `over`.

### A metric

An object with `update(pred, target)`, `compute()` and `reset()`; kalfa picks `pred` and `target` as for a criterion, in original units when the target can be inverted. A `kalfa.ContextMetric` reads everything instead.

```python
import torch

import kalfa


class MaxError:
    def __init__(self):
        self.worst = 0.0

    def reset(self):
        self.worst = 0.0

    def update(self, pred, target):
        self.worst = max(self.worst, float((pred - target).abs().max()))

    def compute(self):
        return self.worst


class Coverage(kalfa.ContextMetric):
    on = ("valid", "test")

    def __init__(self, field="price", width=1.0):
        self.field = field
        self.width = float(width)
        self.inside = 0
        self.count = 0

    def reset(self):
        self.inside = 0
        self.count = 0

    def update(self, outputs, batch, ctx):
        mean = outputs["mean"]
        sigma = torch.nn.functional.softplus(outputs["spread"])
        truth = batch[self.field].view_as(mean)
        self.inside += int(((truth - mean).abs() <= self.width * sigma).sum())
        self.count += truth.numel()

    def compute(self):
        return self.inside / max(self.count, 1)
```

### A Step

A `kalfa.Step` decides what a batch does. Overriding `train_batch`, `eval_batch` and `predict` keeps the default turn (batches, update lines, observers); overriding `train_turn` takes the whole turn. The slot keys `order`, `amp`, `grad_clip` and `accumulate` arrive as attributes.

```python
import kalfa


class NoisyInputs(kalfa.Step):
    def __init__(self, sigma=0.1):
        self.sigma = float(sigma)

    def train_batch(self, batch, ctx):
        noise = ctx.rng.normal("noise", None, like=batch["x"]) * self.sigma
        noisy = {**batch, "x": batch["x"] + noise}
        model = ctx.predicts()
        losses = {}
        for name, optimizer in ctx.optimizers.items():
            value = optimizer.minimizes(model.call(noisy), noisy, ctx)
            loss = value["loss"] if isinstance(value, dict) else value
            ctx.last_norms[name] = optimizer.minimize(loss, clip=self.grad_clip)
            losses[name] = value
        return losses

    def eval_batch(self, batch, ctx):
        return ctx.predicts().call(batch)

    def predict(self, batch, ctx):
        return ctx.predicts().call(batch)
```

```yaml
train:
  epochs: 20
  step: {use: pieces.NoisyInputs, sigma: 0.05, grad_clip: 1.0}
```

### A Trigger

A `kalfa.Trigger` answers `fires(ctx)` after every turn. `describe()` names it in the logs, `reset()` runs after each firing of a rule with `once: false`, and the attributes named in `persist` are saved in checkpoints.

```python
import math

import kalfa


class NotFinite(kalfa.Trigger):
    def __init__(self, watch):
        self.watch = watch

    def fires(self, ctx):
        value = ctx.latest.get(self.watch)
        return value is not None and not math.isfinite(value)

    def describe(self):
        return f"{self.watch} is not finite"
```

### An Action

A `kalfa.Action` is called with the context when its rule fires.

```python
import kalfa


class Snapshot(kalfa.Action):
    def __init__(self, series=("val/max_error",)):
        self.series = list(series)

    def __call__(self, ctx):
        if ctx.record is not None:
            values = {name: ctx.latest.get(name) for name in self.series}
            ctx.record.append("snapshots.jsonl", {"turn": ctx.turn, **values})
```

### A Logger

A `kalfa.Logger` listed in `train.log` gets `bind(ctx)` once, `update(ctx, line)` after every update, `turn(ctx, line)` after every turn and `close(ctx)` when training ends, also on failure.

```python
import kalfa


class BestSeen(kalfa.Logger):
    def __init__(self, watch="val/max_error"):
        self.watch = watch
        self.best = None

    def turn(self, ctx, line):
        value = line.get(self.watch)
        if value is not None and (self.best is None or value < self.best):
            self.best = value

    def close(self, ctx):
        if ctx.record is not None and self.best is not None:
            ctx.record.write_json("best_seen.json", {self.watch: self.best})
```

### A sampler

A sampler has `sample(ctx, n)` and returns a tensor (images as `(N, C, H, W)` get a PNG grid) or a list of strings. Draw randomness from the `samples` stream.

```python
import torch


class Probe:
    def __init__(self, model, width, n=8):
        self.model = model
        self.width = int(width)
        self.n = int(n)

    def sample(self, ctx, n=None):
        x = ctx.rng.normal("samples", (n or self.n, self.width), device=ctx.device.torch)
        with torch.no_grad():
            mean, _ = self.model(x)
        return mean
```

### A plot

A function `(record, figures, **params)` that reads the record files and returns a matplotlib figure, saved as `plots/<key>.png` (or saves its own with `figures.save(figure, name)` and returns None). `figures.new()` makes a figure in the house style and `figures.colors` is the palette. A plot that takes a `ctx` param receives the run context.

```python
from pathlib import Path

import pandas


def residuals(record, figures, pred="pred_mean", target="price", bins=30):
    table = pandas.read_parquet(Path(record) / "predictions.parquet")
    figure = figures.new()
    axis = figure.subplots()
    axis.hist(table[pred] - table[target], bins=bins, color=figures.colors[0])
    axis.set_xlabel(f"{pred} minus {target}")
    return figure
```

### kalfa.lego and plugins

`kalfa.lego` gives a class or function a short name. The module must be imported before the config is built, so list it under `plugins` (a module name or a `.py` file next to the config). Registered names join the catalog of the Python process that imported the module; the `kalfa legos` command imports no plugins, so it lists the std names only. The record copies the module into its `plugins/` directory.

```python
import torch

import kalfa


@kalfa.lego("logcosh", slot="loss", doc="log of cosh of the error: quadratic near zero, linear far away")
def logcosh(pred, target):
    return torch.log(torch.cosh(pred - target)).mean()
```

```yaml
plugins: [pieces.py]
losses:
  smooth: logcosh
```

### Putting the pieces together

```yaml
plugins: [pieces.py]
seed: 3
data:
  use: table
  file: housing.parquet
  split: [0.7, 0.15, 0.15]
  features: {"x*": standard_scaler}
  targets: {price: standard_scaler}
  batch: 32
models:
  net: {use: pieces.TwoHeads, width: 32, outputs: [mean, spread]}
losses:
  nll: {use: pieces.GaussianNll, field: price}
  l1: {use: pieces.scaled_l1, scale: 2.0}
  smooth: logcosh
metrics:
  max_error: pieces.MaxError
  coverage: {use: pieces.Coverage, width: 1.0}
optimizers:
  main: {use: adam, lr: 1.0e-2, over: "@models.net", minimizes: "@losses.nll"}
train:
  epochs: 20
  step: {use: pieces.NoisyInputs, sigma: 0.05, grad_clip: 1.0}
  log: [{use: pieces.BestSeen, watch: val/max_error}]
rules:
  guard:
    when: {use: pieces.NotFinite, watch: train/nll}
    stop: true
  snapshot:
    when: {use: every, epochs: 1}
    once: false
    run: {use: pieces.Snapshot, series: [val/max_error, val/coverage]}
generate: {use: pieces.Probe, model: "@models.net", width: "@data.n_features", n: 8}
plots:
  residuals: {use: pieces.residuals, bins: 20}
record: runs/pieces_$datetime$
```

### A data class

A data class declares `inputs` (the batch fields a model gets by default) and `exposes` (the names `@data.<name>` may read), and gives `fit(ctx)` (learn from the train set and make the sets) and `loaders()` (set name to a DataLoader of dict batches). `target_names` lists the targets for the defaults. `save(directory)` writes what was learned under `<directory>/fitted`, `load(directory)` reads it back and `rebuild(ctx)` makes the sets again when a record is opened; a class that learns nothing may leave out `save` and `load`. `predictions = False` skips the predictions table; a class that writes one also gives `rows(set)`, `spectators(set)`, `inverse(name, values)`, `decode(name, values)`, `is_label(name)`, `rescales(name)` and `loader_for(data)`. An optional class method `check(params, problems, directories)` adds problems to `kalfa check`. An optional `report()` returns a mapping of facts for `data.json`, with the rows of each set under `rows`, and an optional `pipeline()` returns its flowchart as `{"nodes", "edges", "groups"}` in the format of [data.json](#the-record-directory); `kalfa.std.data.flow.Flow` builds one (`node`, `edge`, `join`, `step`, `group`, `chain` and `result`). Without `pipeline()` the run writes a generic flowchart: a source, a set node for each entry of `report()["rows"]`, a tensor for each input and target, and the batch.

```python
import torch
from torch.utils.data import DataLoader


class Sine:
    inputs = ("x",)
    exposes = ("n_features",)
    predictions = False
    target_names = ["y"]

    def __init__(self, rows=512, batch=64):
        self.rows = int(rows)
        self.batch = int(batch)
        self.n_features = 1
        self.made = {}

    def fit(self, ctx):
        generator = torch.Generator().manual_seed(ctx.seed)
        x = torch.rand(self.rows, 1, generator=generator) * 6.0
        samples = [{"x": point, "y": torch.sin(point)} for point in x]
        cut = int(self.rows * 0.8)
        self.made = {"train": DataLoader(samples[:cut], batch_size=self.batch, shuffle=True,
                                         generator=ctx.rng.stream("loader")),
                     "valid": DataLoader(samples[cut:], batch_size=self.batch)}

    def loaders(self):
        return self.made

    def rebuild(self, ctx):
        self.fit(ctx)

    def save(self, directory):
        pass

    def load(self, directory):
        pass
```

```yaml
plugins: [pieces.py]
data: {use: pieces.Sine, rows: 256, batch: 32}
models:
  net: {use: mlp, hidden: [16, 16], out: 1, activation: tanh, in_features: "@data.n_features"}
losses:
  mse: mse
metrics:
  rmse: rmse
optimizers:
  main: {use: adam, lr: 1.0e-2, over: "@models.net", minimizes: "@losses.mse"}
train:
  epochs: 30
  checkpoint: {watch: val/rmse}
record: runs/sine_$datetime$
```

## Reference config

This config uses only std names and most of the grammar; the tests of kalfa check it and train it for two turns on the housing table that `examples/minimal/make_data.py` writes. Running it for two turns: `uv run kalfa run reference.yaml -p epochs=2`.

```yaml
params:
  epochs: 40
  lr: 3.0e-3
  seed: 11
seed: $seed$
record: runs/reference_$datetime$
data:
  use: table
  file: housing.parquet
  split: {use: random, ratios: [0.7, 0.15, 0.15]}
  features:
    "x*": standard_scaler
    x3: [{use: cast, dtype: float64}, robust_scaler]
  targets: {price: standard_scaler}
  filter: {train: "price > 0"}
  batch: {size: 32, eval_size: 128}
models:
  net:
    use: mlp
    hidden: [64, 64]
    out: 1
    activation: gelu
    dropout: 0.1
    in_features: "@data.n_features"
    init: {use: xavier_init}
    ema: {decay: 0.98, every: 1}
losses:
  mse: mse
  l1: l1
  huber: {use: huber, delta: 1.0}
  total:
    use: weighted_sum
    terms:
      - {of: "@losses.mse", weight: 1.0}
      - {of: "@losses.l1", weight: 0.1}
metrics:
  rmse: rmse
  mae: {use: mae, on: [valid, test]}
  rmse_scaled: {use: rmse, rescale: false, every: 2}
optimizers:
  main:
    use: adamw
    lr: $lr$
    weight_decay: 1.0e-4
    over:
      - {of: "@models.net", match: "*.bias", weight_decay: 0.0}
      - "@models.net"
    minimizes: "@losses.total"
    schedule: {use: cosine, steps: 400, warmup: 20}
train:
  epochs: $epochs$
  step: {use: supervised, grad_clip: 5.0}
  evaluate: [valid, test]
  checkpoint: {watch: val/rmse, mode: min, last: true, every: 10}
  report: best
  track: [optimizers.main.lr, "losses.total.terms[1].weight"]
  log: [{use: echo, series: [val/rmse]}]
rules:
  halve:
    when: {use: after_epoch, at: 1}
    set: {optimizers.main.lr: {times: 0.5}}
  more_l1:
    when: {use: every, epochs: 1}
    once: false
    set:
      losses.total.terms[1].weight: {plus: 0.05}
  to_huber:
    after: halve
    when: {use: below, watch: train/total, value: 100.0}
    set: {optimizers.main.minimizes: "@losses.huber"}
  calm:
    when: {use: plateau, watch: val/rmse, patience: 5, min_delta: 0.01}
    stop: true
  far_off:
    when: {use: above, watch: val/rmse, value: 1.0e+6}
    stop: true
  budget:
    when: {use: time_budget, minutes: 30}
    stop: true
plots:
  loss_curve: {use: loss_curve, series: [train/total, val/rmse, val/mae]}
  pred_vs_true: pred_vs_true
sweep:
  strategy: grid
  space:
    lr: [1.0e-3, 3.0e-3]
    seed: [11, 12]
  objective: {watch: val/rmse, mode: min, at: best}
  record: runs/reference_sweep
```

`kalfa sweep reference.yaml` runs the four points of its sweep section under `runs/reference_sweep`; `kalfa run` ignores that section.
