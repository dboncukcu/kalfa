# kalfa config reference (v4)

This document defines the config surface: every section, every key, what it means and what `check` says about
it. The configs under `examples/` (the hand written `examples/alad` and `examples/minimal` among them) and
`reference.yaml` are written against it. The mapping of the surface onto the tezgah graph is the contract,
`src/kalfa/contract.yaml` (`kalfa contract` prints it), and `tests/golden/dumps/*.flow.yaml` shows it expanded.

## 0. How to read this document

Sections 1 to 7 are for the user: the principles, the sections, then the keys of every section, one heading per
key. Sections 8 to 11 are the appendix for designers and plugin authors: the mapping onto the lower layer, what
fits the standard turn and what does not, what v1 leaves out, the alias packs.

Under a key: what it means, a YAML example where the writing is not obvious, the choices as a table, and what
`check` says. The `check` codes are written in code font (`kind_mismatch`), so a line from the terminal can be
searched here.

### Vocabulary

| Term | Meaning |
|---|---|
| lego | a stateless Python callable registered under a URI; `@lego` declares facts (`refs`, `uses`, `needs_grad`, `bus`, `mutates`, `grouped`; kalfa's own facts are `kalfa.kinds.FACTS`, declared to cirak at import) |
| kind | the first segment of a lego's URI: `criterion`, `objective`, `metric`, `layer`, `pre`, `source`, ...; `check` knows from it whether a lego fits the section it is written in |
| criterion | a differentiable lego taking `(predictions, targets)`; used in `losses` and `metrics` (`mae`, `mse`, `huber`) |
| objective | a loss lego taking `(models, batch)`; `losses` only (`vae`, `wgan_gp_d`, `mdmm`) |
| adapter | the std wrapper that hands the `predicts` output and the target to criterion and metric legos; its keys `output` and `target` sit on the definition |
| field | a column of the source or a Dataset field; `fields` gives it a role and a chain |
| batch | the mapping of fields one loader step yields (`x`, `price`, `image`, `label`) |
| wire | the name of the tensor a node of a model graph produces |
| node | a layer, a template use or a `{model: name}` reference in a model graph |
| template | a model piece parameterized with `variables`, copied at every use |
| composite model | a graph that uses trained models through `{model: name}` nodes and has no parameters of its own |
| role | a parameter class for `init`: `weights`, `bias`, `scale` |
| turn | one pass of the training loop: an epoch in the `epochs` mode, `steps.turn` steps in the `steps` mode |
| step | one update pass of the turn lego (`order` walked once); k batches with `accumulate: k` |
| epoch | one full pass over the train loader |
| set | `train`, `valid`, `test`; the history prefixes are `train/`, `val/`, `test/` |
| history | the metric and loss values per turn, `history.jsonl` |
| rule | an item of `rules`: `name`, `when`, `set`, `after`; evaluated at the end of a turn, effective in the next |
| trigger | the lego of `when` and `stop`; it carries state (a patience counter) |
| checkpoint | an intermediate record during training (`best.pt`, `last.pt`, `snapshot_<n>.pt`); `final/` is separate from it |
| run time component | a `{uri: ..., kind: data}` param value; built once the data exists, seeing the train set (`class_weights`) |

## 1. Principles

1. **What is not written does not exist.** The section keys are fixed and an unknown key is an error. kalfa has
   no defaults of its own. Every key is one of four things, and `reference.yaml` marks each with its class:
   * required;
   * "none" when it is not written (an empty list, `null`);
   * derived by one rule that looks at the shape (`predicts` is the single trained model);
   * the default of the wrapped library (torch for init, sklearn for scaling, a lego's own param defaults).

   The only things the framework adds on its own are observation: the record directory, the dump, the progress
   bar, the event stream, the history and the console log of `--log`.
2. **Everything is defined at the top, `training` only uses it.** `preprocessors`, `templates`, `models`,
   `metrics`, `losses` and `optimizers` are named definition sections; the other sections use them by name. There
   are no roles: a loss is bound to an optimizer, and the optimizer minimizes it over the parameters of the models
   that pick it.
3. **A lego call is `{uri, params}`, with four short forms.** A plain string is a call without params
   (`turn: supervised`, `checkpoint: last`, `feed: table`); a `split` without `uri` is the random split; a scalar
   `batch` is `size`; `model` is the single model shorthand. There are no other short forms. Short names come from
   the alias packs an `include` brings; the table is flat (name to URI) and a full URI is valid everywhere. Which
   section a lego may be written in is decided by its kind: `mae` is one lego (`/criterion/kalfa/mae`), a loss
   under `losses` and a metric under `metrics`, and `check` catches a lego in the wrong section (section 5).
4. **Param values.** A constant, a `$param$`, an inline lego `{uri, params}` (a nested schedule, say), a run time
   component (`{uri: class_weights}`), or a **reference**. A lego declares which of its params are references with
   the `refs` fact (`encoder: model`, `recon: criterion`, `transform: preprocessor`, `group: column`,
   `terms: loss`). Only declared params are resolved by name and validated by `check`; an undeclared string is
   plain text (`name: resnet50`).
5. **The mapping is fixed.** kalfa turns the sections into a tezgah graph with a fixed template (section 8). The
   template only looks at whether a section exists, how many items a list has and the shape of a key; the regime
   is chosen explicitly with `training.turn`.
6. **The dump.** `resolved.yaml` in the record directory is the config with its aliases and `$param$`s resolved,
   the source of every overridden value in a comment; it runs again on its own. `flow.yaml` is the generated
   tezgah graph, written nested (one transparent Pipeline per section, the same structure as the event paths),
   readable and checkable.

## 2. The sections

```yaml
include: []          # [empty] alias packs, YAML fragments, other config files; the lower layer
plugins: []          # [empty] Python modules next to the config, or pip packages; @lego registrations
params: {}           # [empty] $name$ placeholders; overridden with -p name=value (--param); no reserved names
seed: null           # [library] the global seed; without it torch runs unseeded, check warns
device: null         # [library] a device lego: auto | cpu | cuda | mps or {uri: cuda, params: {index: 1}}; cpu when absent
rng: null            # [wiring] the model seed rule, an rng lego: derived | indexed | global or {uri, params}; derived when absent

data:                # section 3
  source: ...        # [required]
  transform: []      # [empty]
  frame: []          # [empty]
  split: ...         # [required]
  batch: ...         # [empty]
  mask: null         # [empty]
  preprocessors: {}  # [empty]
  drop: []           # [empty]
  fields: {}         # [required]
  spectators: []     # [empty]
  feed: ...          # [required]

model:               # section 4; [required]
  templates: {}      # [empty]
  models: {}
# or the single model shorthand: model: {inputs, outputs, nodes, optimizer, init, ema, trainable, weights}

metrics: {}          # section 5; [empty]
losses: {}           # [required, at least one]
optimizers: {}       # [empty, with an inline optimizer]

training:            # section 6
  turn: ...          # [required]
  predicts: ...      # [derived] the single trained model when there is one
  targets: {}        # [empty] output wire -> the target fields it predicts
  epochs: ...        # [required, one of the two] or steps: {total, turn}
  loss: ...          # the name of the loss with a single optimizer
  checkpoint: null   # [empty] the intermediate record policy
  report: ...        # [required] best | last
  stop: []           # [empty]
  rules: []          # [empty]
  # every other key goes to the turn's params (amp, grad_clip, accumulate ...)

calibrate: {}        # section 7; [empty]
generate: null       # [empty]
plots: {}            # [empty]
figures: {}          # [empty]
sweep: null          # [empty]
record: ...          # [required]
```

### `seed`, `device`, `rng`

`seed` is the global seed. Python, numpy and torch are seeded with it before the run; the split, the loader
shuffle and the generators of the passes draw from it. Without it torch runs unseeded and `check` warns.

`device` is a lego slot. The short form is an alias (`auto` tries cuda, mps and cpu in that order; `cpu`; `cuda`;
`mps`), the long form a call (`{uri: cuda, params: {index: 1}}`). Every device lego returns a `torch.device` and
checks the availability itself: an unavailable device is an explicit error, there is no silent fallback. The value
on the bus is the `torch.device` object and the chosen device is written to `device.json` in the record. A user
lego under `/device/` (a TPU, say) works the same way.

`rng` says how a model's initialization derives from `seed`; it is written like `device`.

| Value | Rule |
|---|---|
| `derived` (the default, `wiring.default_rng`) | every model is initialized in a forked stream seeded with `sha256(seed:name)`, a substream per model by name, so inserting or renaming another model changes no weights |
| `indexed` | by the definition position, `sha256(seed:index)`; the rule of the runs made before the key, which reproduce with it |
| `global` | no stream is forked; every model draws from the global one in build order, the way a plain script does |
| `{uri: /rng/acme/mine, params: {...}}` | a lego of the `rng` kind: one function `(seed, name, index)` that answers the seed of the forked stream, or `None` for the global one |

Without `seed` the std rules answer `None` and torch runs unseeded. The workers of a loader (`batch.workers`) are
seeded from the torch seed one by one, numpy included, and the image augmentations draw from a generator of their
own, so two workers never repeat each other's crops.

### Layers and `include`

The config of a run is the union of layers, bottom to top:

1. The alias packs (`/alias/kalfa/...`, from the registry).
2. The `include`d files, in the order of the list. Each with its own `include`s forms one layer, recursively: an
   included file is below the file that includes it.
3. The including file, the config given on the command line.
4. `--set path=value` and `-p name=value`, the top.

The `--set` path is dotted from the root of the document; a bare name is a top level key (`--set record=runs/x`,
`--set device=cuda`, `--set training.epochs=5`, `--set training.stop=[]`, `--set params.lr=1e-4`). The value is
parsed as YAML. `-p name=value` (`--param`) is the shortcut for `params.name` (`-p lr=1e-4`). Both go to cirak as
they are.

The merge rules:

* inside a layer the merge is strict: the same key defined twice is an error, even with an equal value;
* between layers the leaf overrides: the upper value wins and mappings merge recursively;
* an upper value that is a lego call (a mapping carrying `uri` or `block`) replaces the lower one wholesale; the
  `params` do not mix;
* an upper value that writes a path inside a call (a path override, `--set training.checkpoint.params.monitor=val/mae`)
  merges;
* the `include` and `plugins` lists are pooled across layers, and a file included by two paths is loaded once.

`resolved.yaml` writes the source of every overridden value in a comment
(`epochs: 5  # --set; overrides 01_mlp_regression.yaml:11`), and `kalfa check --layers` prints the layer tree and
the leaves every file overrides. `11_kfold_cv` and `12_resume` work by this rule: they take `01` as the lower
layer, override the `split` and `record` leaves, and everything else comes as it is.

## 3. `data`

The order of the data block is fixed and independent of the writing order:

```
  source
    ↓
  transform            the entries without sets, before the split
    ↓
  split                train · valid · test
    ↓
  transform per set    the entries with sets: [...], after the split
    ↓
  frame                fitted on train, mapped onto every set
    ↓
  drop, fields, spectators, preprocessors    fitted on train, applied per set
    ↓
  feed                 rows become tensors
    ↓
  loader               one per set
```

Where a data operation belongs:

| The operation | Goes under |
|---|---|
| not fitted, one column (`cast`) | a preprocessor without a fit |
| not fitted, the whole frame | `transform` |
| fitted on train, one column | `preprocessors` |
| fitted on train, the whole frame | `frame` |

### `source`

The lego that brings the data. It returns a DataFrame (`parquet`, `csv`) or a Dataset (`image_folder`,
`text_lines`). Every column or field of the source is a field: `image`, `label`, `text`, the table columns.

`check` reads the data header (column names and types, set sizes); it does not load the data. A table source takes
`columns`, the list of columns to read (`{uri: parquet, params: {path: tree.parquet, columns: [M1, M2]}}`): the
rest of a wide file stays on disk, the header lists those columns only, and `check` refuses a name the file lacks.

### `transform`

An ordered list of frame to frame legos of the `transform` kind, the place for anything that touches the frame as
a whole.

| Lego | Params | Does |
|---|---|---|
| `filter` | `query` | a pandas `query`; a bare string in the list is this call, the shorthand of the common case |
| `derive` | `column`, `expr` | a named column from a pandas `eval` expression |
| `rename` | `pattern`, `to` | a regular expression over the column names |
| `astype` | `columns: {name: dtype}` | a type conversion |
| `drop` | `columns` | columns removed |
| a lego of your own | | takes a frame, returns a frame |

Two stages, one list, by the presence of `sets`: an entry without `sets` runs before the split, in the order
written; an entry with `sets` runs after the split on those sets only. A derived column can be filtered on and
derived from again, in the order the list says.

The rule a transform must obey: row by row and deterministic, never fitted. It runs before the split, so a
transform that learns anything from the data (a mean, a vocabulary, a quantile) leaks the test set into training;
everything that learns belongs in `preprocessors`, which are fitted on train only. `check` cannot enforce this, so
it is written here and in the kind's description.

`predict --data` replays the list: every entry without `sets` and the entries that name `test`, exactly what the
test set went through, filters included, and the dropped count goes to the log. On a Dataset source a filter is a
field equality (`label == 3`); on a stream only `filter` runs (the others need the table, a `lazy_transform`
error).

### `frame`

An ordered list of fitted frame transforms: fitted on train and reading the whole frame, key columns included.

| Lego | Params | Learns |
|---|---|---|
| `group_statistic` | `by`, `column`, `statistic`, `name` | a statistic of a column per group on the train set, mapped onto every set as a new column |
| `target_encoding` | `column`, `target`, `smoothing`, `name` | the train mean of the target per category, smoothed toward the overall mean |
| a lego of your own | | subclasses `kalfa.std.frame.base.FrameTransform` with `fit(df)` and `apply(df)` |

They run after the split, each fitted on what the ones before it produced, are applied to every set before the
column chains, are kept under `fitted/frames/` and are replayed by `predict --data`. A stream source cannot fit
one (`lazy_frame`) and a Dataset source has no frame (`frame_needs_table`).

`check` foresees the columns the transforms and the frames add by running them on an empty table with the header's
columns, so a field may name a derived column and the column table of `describe` shows it.

### `split`

The split lego.

| Writing | Means |
|---|---|
| `{ratios: [0.7, 0.15, 0.15], seed: $seed$}` | the short form of the random split, `{uri: random_split, params: {ratios, seed}}` written out (the short name is `random_split` because `random` is the sweep strategy) |
| `{uri: sequential, params: {ratios, group}}` | sequential and grouped |
| `{uri: given, params: {valid: b.parquet, test: c.parquet}}` | the sets given as files |
| `{uri: kfold, params: {k: 5, fold: $fold$, val: 0.15, seed: $seed$}}` | one fold of k, see below |

`ratios` always has three entries, in the order `train`, `valid`, `test`; `0.0` means "no set". Metrics and
losses are computed on the sets that exist, the key of a missing set is not in the history, and a `monitor`
watching a missing set is a `check` error. `check` prints the set table (`train 45000, valid 5000, test none`).

The sets are what the split lego returns (its `returns` fact). The std splits return `train`, `valid`, `test`,
with the history prefixes `train/`, `val/`, `test/` from the contract's wiring. A split of your own may return
more, a `calib` set for a threshold, a conformal or a probability calibration or an ensemble blend, and everything
follows the list: a loader per set on the plot bus, `sets: [calib]` on a definition, `training` evaluates every set
beyond the wiring's defaults, the history prefix of a set the wiring does not name is its own name (`calib/rmse`),
`predictions_<set>.parquet` beside `predictions.parquet`, and a `threshold` reads it with `set: calib`. A split
must return `train`.

**K fold.** Every fold is a separate run (`-p fold=i`) and `kalfa collect` gathers them (section 7); inside a
sweep, `fold` is a swept param and `kalfa collect <root> --mean-over fold` ranks the settings by their mean. The
held out fold is the **test** set; the CV estimate is the `test/` metrics and exists in every fold. With `val` a
`valid` set of that ratio is carved from the training part; without it there is no `valid`, `stop` and
`checkpoint` cannot watch `val/` (a `check` error), and the fold runs for `epochs`. The `kfold` lego has no param
that makes the held out fold the `valid` set: early stopping on the test fold makes the estimate optimistic, and
kalfa does not let it be written.

### `batch`

The batch size, or the long form. The mapping is the params of the loader lego of the contract's wiring
(`loader: /loader/kalfa/torch`), so its signature is the key list, `check` reads the keys from it, and a contract
with a loader of your own accepts what that loader's signature says.

| Key | The std loader |
|---|---|
| `size` | the train batch size; a scalar `batch` is this key |
| `eval_size` | the evaluation batch size; `size` when not written |
| `shuffle` | on for the train loader, off for the evaluation loaders |
| `drop_last` | `false`; `auto` drops the last train batch only when it would hold one row, the line every hand written loop has for a BatchNorm model |
| `workers`, `eval_workers` | `0`; `eval_workers` is `workers` when not written, so an evaluation pipeline lighter than the training one is written `eval_workers: 0` |
| `persistent` | keeps the workers alive between turns; on whenever there are workers, because torch otherwise tears the pool down and respawns it once per loader per turn |
| `prefetch` | torch's `prefetch_factor`; left to torch when not written |
| `pin_memory` | follows the device when not written: on for cuda, off for cpu and mps, where pinning costs memory and gives nothing back |
| `collate` | `null`, the torch default |
| `balanced` | `true` puts a class balancing sampler over the target field on the train loader |
| `buffer` | the shuffle buffer of the lazy set, default 4096 |

Not written means no batching: the whole set is one batch (the loader's own `size: None`), one step per turn, the
evaluation sets one batch each. A stream source needs a size. `device` reaches the loader from the bus and cannot
be written in the section.

### `mask`

A pandas query. The rows it selects are a third state between "in the batch" and "gone": they stay in the frame (a
plot reads them through `set_frame`, where the `masked` column marks them) and are not scored (the loader leaves
them out, so they are neither trained on nor predicted). The mask applies to every set and to the data of
`predict --data`. A table only key: a stream cannot carry one (`lazy_mask`), a Dataset source has no frame
(`mask_needs_table`).

### `preprocessors`

Named transform definitions. Those that need a fit are fitted on the filtered train set only and go into the record
(`fitted/preprocessors/`); `kalfa predict` reads them from there. A type conversion is a lego too (`cast`).

* A definition may carry `sets: [train]`: it applies to those sets only (data augmentation). The data of
  `kalfa predict` is treated like the `test` set: among the preprocessors that write `sets`, only those including
  `test` apply.
* A definition is either used in a `fields` chain or named by a param whose `refs` type is `preprocessor`
  (`two_views {transform: simclr_aug}`); when neither, `check` warns.
* On a Dataset source a fitting preprocessor fits on the column when the field can be read as one (text lines); on
  an image field it is an error.

The catalogue, independent of the modality:

| Family | Legos | Notes |
|---|---|---|
| fitted scalers | `standard_scaler`, `median_std_scaler`, `minmax_scaler`, `max_abs_scaler`, `robust_scaler`, `quantile_transformer`, `power_transformer` | `robust_scaler` is the median and a percentile distance, outliers do not move it; `power_transformer` is Yeo-Johnson |
| widening | `one_hot`, `kbins_discretizer`, `spline_transformer` | bins as `<field>_bin<n>` (or `encode: ordinal`), a B-spline basis as `<field>_spline<n>` |
| scale transforms without a fit | `log`, `abs`, `logit`, and `asinh`, `sinh`, `tanh`, `atanh` on `x / scale` | `asinh` is the signed log of a heavy tailed column, defined at zero; `tanh` bounds into `(-1, 1)` and its inverse saturates beyond about 19 scale; `atanh` needs `\|x\| < scale` and says so; `logit` takes a probability onto the real line and is inverted by the sigmoid |
| missing values | `simple_imputer`, `fill` | `simple_imputer` (`{strategy: mean, median, most_frequent or constant, fill_value, indicator}`) fills with the train statistic and, with `indicator: true`, adds `<field>_missing` as a feature of its own, computed before the fill; `fill` (`{value}` or `{method: ffill or bfill}`) fills without a fit, and `value: missing` makes the gap its own category before `one_hot` |
| types and labels | `cast`, `label_encoder` | |
| images | `resize`, `random_crop_flip`, `to_tensor`, `to_tensor_signed`, `normalize`, `two_views`, `simclr_aug` | the step from PIL to a tensor is always an explicit lego; `normalize` expects a tensor |
| text | `char_tokenizer` | fitted; its vocabulary goes into the record and `generate` reads it from there |

A transform that reads the whole feature vector is a layer, not a preprocessor: the field plan is one column in,
that field's columns out. Feature interactions (`polynomial`) and per sample normalization (`l2_normalize`,
sklearn's `Normalizer`) sit in the model over the `x` tensor, where they cost nothing in the plan and run per
batch.

**Side outputs.** A preprocessor may hand back extras beside the value that continues down the chain
(`extras(values)` on the object, `{key: column}`). They leave the chain, become features named `<field>_<key>`
after the field's own columns, are recorded in the plan and are never inverted.

**One definition, one object.** A preprocessor that declares `grouped = True` (the scalers) is fitted once over
the matrix of every column that names it, in the plan's field order, and keeps its statistics per column, so each
column is still transformed and inverted with its own scale. A definition that two chains name is one shared
object; write a second definition under another name to keep two sets of statistics apart. The rest (`one_hot`,
`label_encoder`, `cast`, `log`, the image and text transforms) are fitted per column, one copy each. The
declaration is the `grouped` fact of the lego (`kalfa ls` and `DOCS.md` show it) and the same attribute on the
object it builds, which is what the fit reads. The limits of a grouped preprocessor:

* every column reaching it must be one column wide (an error otherwise, so it cannot follow `one_hot`);
* all its columns must reach it at the same point of their chains (two grouped preprocessors written in different
  orders are an error, `grouped_order`);
* the lazy set fits it column by column, which is identical for a per column transform, so only a transform that
  mixes its columns is affected.

**From the frame type to the torch type.**

| Frame dtype | Torch dtype |
|---|---|
| `float16`, `float32`, `float64` | the default torch float type (`float32`) |
| `int8` to `int64`, `uint8` | `int64` |
| `bool` | `bool` |
| `category`, `object`, `string`, `datetime` | a `check` error: a preprocessor is needed (`one_hot`, `label_encoder`, `cast`, the tokenizer) |

### `spectators`

Globs naming the columns that ride along without entering the model. They go through no preprocessor, keep their
original values, and reach the data plots (`set_frame`) and `predictions.parquet`, so an id, a weight, a run number
or the true label of a set the model never saw can be filtered and grouped on afterwards.

A column a lego names through a `column` typed reference (`group` on the `sequential` split and on the `window`
feed, `by` on `group_statistic`) is carried the same way without being written here, because it is already
written in that lego's params; the dump shows the whole list on the `fit` node.

What `check` says: a column that is neither a field, nor a spectator, nor referenced is read and discarded
(`column_unused`), which is what a mistyped glob looks like; a spectator that a `fields` glob also claims is an
error (`spectator_in_fields`), as is one that only matches dropped columns (`dropped_spectator`). A table only
key, like `mask`: a Dataset source has no frame (`spectators_need_table`) and a stream carries only the fields it
reads (`lazy_spectators`).

### `drop`

Columns dropped before the field mapping; they reach no pattern. Columns named by a `column` typed reference
(`group`) cannot be dropped (a `check` error); they are not written in `fields` either, they do not enter the
model, they reach the lego.

### `fields`

Field to role and chain. The keys are globs (`?` one character, `*` zero or more); `{preprocessors: [...]}` is the
chain and `target: true` the target. A field that is not written does not enter the model.

When a column matches several patterns:

1. a name without wildcards wins;
2. the one with more fixed characters wins;
3. at a tie the one with more `?` wins;
4. still tied is a `check` error (`glob_ambiguous`).

`input` and `x` are reserved names: a field named `input` and a non target column named `x` are `check` errors.
The target transform is inverted in the report and in the predictions.

### `feed`

The shape of the batch.

| Lego | The batch |
|---|---|
| `table` | the non target fields join into one tensor named `x`; the target fields keep their names; Dataset fields (`image`, `label`) keep their names |
| `{uri: window, params: {size, horizon, context: true, group}}` | window and horizon pairs; `context` takes context from the previous set at the split boundary; `group` is the series column (`column` typed, a window does not cross a series) |
| `{uri: next_token, params: {seq_len}}` | `input_ids` and `targets` from the token stream |
| a lego of your own | |

The batch is a mapping; losses and metrics read the fields by name.

### The lazy set

`include: [/alias/kalfa/lazy_tabular]` is the tabular pack with `parquet` and `csv` bound to stream sources that
read in chunks. What changes:

* the `filter` transforms run on the stream; the other transforms need the table (a `check` error);
* the split is `sequential` or `given`;
* `fit` makes one pass per chain position: an `incremental` preprocessor (the sklearn scalers) fits through
  `partial_fit`, one that is not collects its column in memory (a `lazy_fit` warning), a `grouped` preprocessor is
  fitted column by column;
* the `table` feed produces an IterableDataset and the train pass shuffles through the `batch.buffer` buffer.

The limits, every one a `check` error: no random `split`, no `kfold`, `balanced`, `window` or `class_weights`;
`workers: 0`; an empty train stream is an error at its first pass.

## 4. `model`

### `templates` and `models`

`templates` are pieces parameterized with `variables`, used with `{template: name, params: {...}}`; every use is a
separate copy. `repeat: n` stacks an item n times (n separate copies), valid in a chain and in a graph.

`models` are the named models. Every model and template is a mapping whose `nodes` is either a list or a mapping:

* a **list** is a chain: every item takes the output of the previous one;
* a **mapping** is a graph: the key is the node name and, without `outputs`, the single wire the node writes;
  with `outputs` the wires are those names and the node name is not a wire.

```yaml
models:
  tower:                                       # a chain
    inputs: [x]
    outputs: [y]
    nodes:
      - {uri: linear_relu, params: {out_features: 128}}
      - {uri: linear, params: {out_features: 1}}
  corrector:                                   # a graph: the output is a correction of two input columns
    inputs: [x]
    outputs: [y]
    nodes:
      base:  {uri: select, params: {index: {uri: feature_index, params: {columns: [m1, m2]}}}, inputs: [x]}
      delta: {uri: mlp, params: {widths: [128, 128], out_features: 2}, inputs: [x]}
      y:     {uri: add, inputs: [base, delta]}
```

The boundary wires are `inputs` and `outputs`, written in a chain too. When the boundary of a chain has more than
one wire (`inputs: [image, t]`) the first node takes every input, the following ones the output of the previous
node, and the last node writes the `outputs` wires. A node may produce several wires (`outputs: [feature, logit]`,
picked by name from the template's `outputs`), and the model's `outputs` picks among them. A wire produced and not
used is a `check` warning.

A layer param may be a `kind: data` component (`{uri: vocab_size}` for an `embedding`'s `num`,
`{uri: feature_width}` for a `layer_norm`, `{uri: feature_index, params: {columns: ...}}` for a `select`); the
builder builds it from the fitted preprocessors and the train loader.

Wiring between models (the encoder's `z` into the generator) is the job of the loss lego; it is not written in
the config. A `.` in a model name is forbidden.

### Composite models

A `{model: name}` node puts a trained model itself (the same object) into the graph, which makes the graph a
composite model.

* It has no trainable parameters of its own; parameterless layers (`reparam`, `concat`, `l1_distance`, `add`) are
  allowed.
* It cannot write `optimizer`, `init`, `ema`, `trainable`, `weights`.
* It does not enter the checkpoint; it is rebuilt from the YAML.
* It has no mode; its parts are in their own mode.
* When the same model is called twice, the wires of every call are named separately with `outputs`.

`/layer/kalfa/reparam` samples in train mode and returns `mu` in eval mode.

### `optimizer`

A name in the `optimizers` section (a string) or an inline definition (a mapping; its loss through
`training.loss`). Models writing the same name share one optimizer object. A model that writes none is not trained
(`check` warns; composites and `trainable: false` models excepted).

### `init`

A role mapping: `weights` (two and more dimensional), `bias` (one dimensional names ending in bias), `scale` (the
normalization gamma). Every role is a `{uri, params}` call; a short name (`bias: zeros`) is a `check` error, because
only the mapping is built. Two levels:

| Level | Writing |
|---|---|
| per model | `init: {weights: {uri: kaiming}, bias: {uri: zeros}}` |
| by pattern | `init: {patterns: [{match: "head.h.*", weights: {...}}]}`, a parameter name pattern, after the roles in list order |

`init` on a `nodes` item is not applied, and `check` refuses it; a node's parameters are reached by a pattern on the
node's name (`match: "delta.*"`). A pattern matches the name as the config writes it, so a node of a `template`
under `repeat: 2` is reached by `match: "h_0.f.*"` or `match: "h_*.f.*"`. There is no global default: a role that is
not written stays at the torch
initialization. A lazy layer builds its weights on the first batch; the build happens under the model seed and `init`
applies at that moment.

### `weights`

`weights: {run: runs/x, model: name, which: best | last | final}`: after the build, the weights of the named model
of that run's record are loaded wholesale (teacher, fine tuning, warm start). `which` is required. Together with
`init` on the same model it is a `check` error. `check` compares the model block of the source run's
`resolved.yaml` with this one, and the `state_dict` fit is checked again at run time. It can be written together
with `ema` (the copy is cloned from the loaded weights).

### `ema`

`ema: {decay: 0.999}`: the exponential moving average copy of the model. It is shifted after every real optimizer
update (once per k batches with `accumulate: k`), kept in fp32 under AMP, moved and saved, and selected under the
name `<name>.ema` in `predicts`, `generate`, `refs` params of type `model` and in `kalfa predict --model`.

### `trainable`

`trainable: false`: the parameters' `requires_grad` is off **and** the module is in eval mode (BatchNorm statistics
freeze, dropout turns off). The rule `set: {<name>.trainable: true}` turns both on. When an optimizer is written the
parameters stay in the optimizer and start updating once opened. There is no separate mode key.

### The single model shorthand

`model: {inputs, outputs, nodes, optimizer: {uri: adam, ...}}`. Its name is `model`, `predicts` is not written, the
loss goes through `training.loss`. `model` and `model.models` cannot be written together.

### The layer catalogue

The `layer` kind mirrors `torch.nn` under `/layer/torch/` and adds kalfa's own under `/layer/kalfa/`. There is no
`width` param anywhere.

| Family | Legos |
|---|---|
| activations | `relu`, `relu6`, `leaky_relu` (`negative_slope`), `prelu`, `rrelu`, `elu`, `celu`, `selu`, `gelu`, `silu`, `mish`, `hardswish`, `hardsigmoid`, `hardtanh`, `hardshrink`, `softshrink`, `tanhshrink`, `softsign`, `softplus`, `sigmoid`, `log_sigmoid`, `tanh_layer`, `threshold_layer`, `glu`, `softmax`, `softmin`, `log_softmax`, `softmax2d` |
| normalisation | `batch_norm` (`dims` 1, 2 or 3, lazy in the number of features), `instance_norm` (the same `dims`, lazy), `layer_norm` and `rms_norm` (`normalized_shape`, a number or `{uri: feature_width}`), `group_norm` (`num_groups`, `num_channels`), `local_response_norm` |
| dropout | `dropout`, `dropout1d`, `dropout2d`, `dropout3d`, `alpha_dropout`, `feature_alpha_dropout` (`p`) |
| pooling | `maxpool`, `avgpool`, `lppool`, `adaptive_maxpool`, `adaptive_avgpool`, `fractional_maxpool`, `max_unpool` over images, with the `1d` and `3d` suffixed twins over sequences and volumes |
| padding | `zero_pad`, `constant_pad`, `reflection_pad`, `replication_pad`, `circular_pad`, with the same twins |
| convolution | `conv1d`, `conv2d`, `conv3d`, `conv_transpose1d`, `conv_transpose2d`, `conv_transpose3d` (`out_channels`, `kernel`, `stride`, `padding`, `dilation`, `groups`, `bias`; lazy without `in_channels`) |
| shape | `flatten`, `/layer/torch/unflatten` (`dim`, `size`), `concat`, `last_step`, `fold`, `unfold`, `pixel_shuffle`, `pixel_unshuffle`, `channel_shuffle`, `upsample` |
| recurrent | `gru`, `lstm`, `rnn` (`hidden`, `layers`, `dropout`, `bidirectional`; the input width from the first batch) and the cells `gru_cell`, `lstm_cell`, `rnn_cell` |
| attention | `multihead_attention`, `transformer_encoder_layer`, `transformer_encoder`, `transformer_decoder_layer`, `transformer_decoder`, `transformer` |
| embedding, distance, linear | `embedding` and `embedding_bag` (`num` may be `{uri: vocab_size}`), `cosine_similarity`, `pairwise_distance`, `/layer/torch/linear` (`in_features`, `out_features`), `bilinear`, `identity` |
| kalfa's blocks | `linear` (`out_features` required, `in_features` optional; lazy without it), `linear_relu` (the same params, `Linear` plus `ReLU`), `mlp` (`widths`, the hidden widths; `activation`, `dropout`, an optional `out_features` for the plain last layer; one node for a whole perceptron) |
| kalfa's features | `polynomial` (the feature interactions), `l2_normalize`, `select` (the positions of the feature axis `index` names, a list or `{uri: feature_index}`; `dim` picks the axis), `unflatten` (a per sample reshape) |
| kalfa's wires | `add`, `multiply` (every input wire, broadcast), `subtract`, `divide` (the first minus or over the second), `negate`, `reparam`, `l1_distance` |
| kalfa's parameters | `multipliers` (the Lagrange multipliers of an `mdmm` loss as a model of their own; `names` takes the constraints mapping, `init` the start of a name without `lmbda_init`) |
| plugins | `timm_backbone` in the examples, or any `nn.Module` registered with `@lego` |

## 5. `losses`, `metrics`, `optimizers`

### The three kinds

Which section a lego may be written in follows from its kind:

| Kind | Signature | Where | Std |
|---|---|---|---|
| `criterion` | `(predictions, targets, **params)` to a scalar; differentiable | `losses` and `metrics` | `mse`, `weighted_mse`, `huber`, `mae`, `log_cosh`, `cross_entropy`, `bce_logits` |
| `objective` | `(models, batch, **params)` to a scalar or a mapping of terms | `losses` only | `vae`, `wgan_gp_d`, `wgan_g`, `ddpm`, `ntxent`, `distill`, `weighted_sum`, `mdmm` |
| `metric` | stateful: `update(...)`, `compute()`; the `update` arguments are bound by name (`predictions`, `targets`, `models`, `batch`, `rng`, `predicts`, `record`, `turn`, `prep`, `set`); when `compute()` returns None nothing is written to the history | `metrics` only | `rmse`, `accuracy`, `f1`, `auroc`, `average_precision`, `perplexity`, `recon_error`, `fid`, `sample_writer` |

`weighted_mse` takes a weight per target column: `{uri: target_weights, params: {weights: {column: 3.0,
"glob*": 1.5, default: 1.0}}}`, resolved against the target columns the dataset carries once the train loader
exists.

`check`: a `metric` kind under `losses` and an `objective` kind under `metrics` are errors (`kind_mismatch`).
Replacing a criterion with `set` (`vae_loss.recon: huber`) is validated with the `refs` type `criterion`. The
alias table is flat; there is no `/loss/...` and `/metric/...` pair. Metric and loss definition names share one
namespace, and the same name in both is a `check` error.

### The adapter: `output` and `target`

Legos that want `predictions` and `targets` (every criterion, most metrics) go through the adapter the contract's
wiring names for their kind (`criterion_adapter`, `metric_adapter`, `objective_adapter`; the std ones under
`/adapter/kalfa/`). Per batch the adapter computes the output of the `predicts` model once, shared by every loss
and metric, and picks the target field. Metrics that want `models` and `batch` (`fid`) are called directly.

The adapter keys sit on the definition beside `every` and `sets`; `params` go to the lego only:

| Key | Means |
|---|---|
| `output` | the wire of a multi output model; the first output when not written |
| `target` | a field name, a list of names, a glob, or `input` (the model's own input). When not written: the selector of the wire in `training.targets`, else the single field marked `target: true`; a `check` error when there are several and nothing names one. Several fields are stacked into one `(batch, n)` tensor in the plan's field order and every column is rescaled with its own chain |

```yaml
losses:
  rec: {uri: mse, target: input}
  ce:  {uri: cross_entropy, params: {weight: {uri: class_weights}}}
```

Under `losses` the criterion value is backpropagated when it is the active loss and enters the history as a
running mean in every case. Under `metrics` the adapter detaches the prediction and the target before the lego sees
them (a metric observes, it never backpropagates) and keeps a sample weighted mean under `no_grad`; the six std
criteria decompose into sample means, so the mean is exact.

### The scale rule

The scale follows the section, not the kind of the lego:

* everything under `losses` is computed in the model scale, the transformed target, the unit the model sees;
* everything under `metrics`, criteria included, is computed in the original scale: the adapter runs the prediction
  and the target back through the rescaling preprocessors (the scalers and `log`; encoders and `cast` are not
  undone, a target that is not a field stays as it is).

So the same criterion is seen in two scales in two sections. In the history `*/<loss>` is in the model scale and
`*/<metric>` in the original scale; `predictions.parquet` is in the original scale and `raw_<output>` raw.

The inversion runs where the batch already is when every preprocessor of the chain has an `inverse_torch` method
(the affine scalers, `log`, `sinh`, `atanh`, `logit`), and falls back to numpy on the CPU when one of them does not
(`quantile_transformer` and `power_transformer` have no closed form; `asinh` and `tanh` need to read their own
values or more precision than float32 gives). There is nothing to declare: a preprocessor of your own joins the
first group by subclassing `kalfa.std.pre.base.Affine` and answering `affine()` with its `(shift, scale)`, or by
writing `inverse_torch(tensor, columns=None)` itself.

### The objective lego

`(models, batch, **params)`: `models` is the named mapping (`models["encoder"]`, the EMA copy
`models["unet.ema"]`), `batch` the field mapping. The wiring between models (the encoder's `z` into the generator)
is written here. It returns a scalar or a mapping of terms (`{loss: ..., recon: ..., kl: ...}`; `loss` is required
in the mapping and the turn backpropagates it). When the signature names `step`, `epoch`, `rng` or `scaler` kalfa
passes them (step schedules, a seeded `torch.Generator`, the AMP scale; the name `generator` is free for a model
reference); when it names `losses` it gets the other definitions of the section, evaluated on the same batch by
name.

The terms enter the history: a scalar as `train/<name>`, a mapping as `train/<name>/<term>` with its `loss` term
as `train/<name>`.

Two std objectives combine other definitions:

* `weighted_sum` (`{terms: {rec: 1.0, kl: 0.1}}`, the keys are `losses` definition names) binds several losses to
  one optimizer.
* `mdmm` minimizes `primary` under equality constraints on other definitions, each held at its `epsilon` by a
  Lagrange multiplier: the loss is `primary + sum(scale * (lambda * inf + damping * inf^2 / 2))` with
  `inf = epsilon - value`, so the config writes the target of a term in its own unit instead of a guessed weight.
  A constraint names a definition, or a term of a mapping objective as `name.term`.

```yaml
params:
  constraints:
    recon_mean: {epsilon: 1.0, lmbda_init: -1.0}   # epsilon required; lmbda_init, scale, damping optional
    mmd.all:    {epsilon: 0.0}

model:
  models:
    lambdas:                                        # the multipliers, a model of one node
      optimizer: main
      inputs: [x]
      outputs: [lmbda]
      nodes: [{uri: multipliers, params: {names: $constraints$}}]

losses:
  total: {uri: mdmm, params: {primary: mse, multipliers: lambdas, constraints: $constraints$}}

optimizers:
  main: {uri: adam, params: {lr: 1.0e-5, groups: [{match: "lambdas.*", lr: -2.0e-4}]}, loss: total}
```

The multipliers are a model, so the optimizer updates them, a rule can reach them and the checkpoint carries them;
they climb through a group with a negative learning rate, the gradient ascent of the saddle point. `names:
$constraints$` writes the mapping once, so the model and the loss stay in step (a mismatch is an error at the first
batch). The history gets `lambda/<name>` and `inf/<name>` per constraint, and `set: {main.loss: other}` freezes
the multipliers along with the phase. A rule on `main.lr` changes the default group only, so the multipliers keep
their rate; `main.lambdas.lr` reaches them by name, `main.*.lr: {times: 0.5}` halves every group's own rate and
keeps the sign, and an absolute `main.*.lr` would write a positive rate into their group, which `check` warns
about.

### The metric lego

It accumulates at the set level (`update` and `compute`; the std ones wrap torchmetrics). In an undefined case
(a single class set) the std wrapper returns NaN and warns once; triggers treat NaN as "absent".

### Every set, every turn

Every metric and loss is computed on every existing set every turn and written to the history. The sets have
different jobs: `valid` steers the run (`checkpoint` monitors a `val/` key, `stop` and the rules watch it,
`report: best` picks the turn by it); `test` is only measured and written, no decision of the run reads it, so the
`test/` line of every turn shows how the model chosen on `valid` fares on data that chose nothing;
`predictions.parquet` is the chosen model on `test`.

The train set values are accumulated in a streaming way during the training pass (the turn lego takes `metrics`
and updates them every step), so the `train/` values are the training pass itself (augmented batches, train
mode), not a clean pass. The evaluation pass runs under `no_grad` and in eval mode.

The exception keys sit on the definition, for losses and metrics alike: `every: k` for the frequency,
`sets: [valid, test]` for silencing. An objective that needs gradients (the `needs_grad` fact) or only makes sense
on the train batch writes `sets: [train]`; without it a `check` error. A `needs_grad` objective and `amp: true`
can be written together only when the signature has `scaler`.

### The optimizer

`{uri, params, loss, schedule}`:

* `loss` is the name of the loss it minimizes; the parameters are the union of the models that pick it. With a
  single optimizer the loss is written through `training.loss`.
* `params.groups: [{name: backbone, match: "backbone.*", lr: 1.0e-5}]` is a parameter group; the pattern is
  matched against the `<model>.<parameter>` name, a parameter joins the first group that matches and the rest
  form the default group, whose values are the other keys of `params`. A key a group does not write follows the
  default group. `name` is optional and plain (no `.`, `*`, `?` or `[`), unique within the optimizer, and is how a
  rule and the history reach the group: `set: {main.backbone.lr: ...}`, `lr/main/backbone`.
* `schedule: {uri: warmup_cosine, params: {warmup, total}}` is a step schedule; it takes the learning rate from the
  optimizer and `total` counts the updates of its own optimizer.
* An unused definition is an error. The writing order of the `optimizers` mapping is meaningful: it is the default
  of the turn's `order`.

## 6. `training`

The fixed keys: `turn`, `predicts`, `targets`, `epochs` or `steps`, `loss`, `checkpoint`, `report`, `stop`,
`rules`. Every other key goes to the turn's `extra` mapping; when the turn lego's `extras` fact does not list it,
it is an error (a turn without the fact takes no extra key). The std turn takes `amp`, `grad_clip` and
`accumulate`. `amp` is float16 autocast on cuda and bfloat16 autocast on the CPU; bfloat16 on the CPU can be very
slow depending on the hardware, so the CPU demonstrations write `--set training.amp=false` (`examples/10`).

### `turn`

The std turn is `alternating`: `{uri: alternating, params: {order: [d, g], steps: {d: 5, g: 1},
fresh_batch: false}}`; `supervised` is its short name for a single optimizer (a `check` error with several).

Every step it walks the optimizers in `order`. Each computes its own loss `steps` times (the trained models in
train mode, the `trainable: false` ones in eval mode), backpropagates and updates its own models, zeroing only the
gradients of its own parameters; stopping a gradient flowing into another model is the `detach` job of the loss.
Models with an EMA are shifted after a real update. `accumulate` and `grad_clip` apply per optimizer, `amp` to all.
`fresh_batch: true` draws a new batch from the loader for every optimizer step (the canonical WGAN GP), `false`
reuses the same batch. With `steps: {d: 5}` `train/<loss>` is the mean of the five steps. The lego defaults:
`order` is the writing order of `optimizers`, `steps` 1, `fresh_batch` false.

The turn lego takes `models`, `optimizers`, `emas`, `losses`, `metrics`, `predicts`, `counters`, `composites`,
`effects` and the loader, and returns the `train/` metrics; when a custom turn's signature has no `metrics`,
`check` warns that the `train/` metrics are not computed. A custom turn is only for regimes that do not fit this
pattern (section 9).

### `predicts`

The output of which model counts as "the prediction": the criteria and metrics that go through the adapter, the
plots, the default of `kalfa predict`. Not written when there is a single trained model (it is that model).
Optional otherwise; when a lego that needs it (the `uses: predicts` fact) exists and it is absent, a `check`
error. It may be a composite model. In generative models the derived `predicts` can be meaningless; `kalfa
generate` is used.

### `targets`

Which target fields every output wire of the `predicts` model predicts, `{wire: selector}`. A selector is a field
name, a list of names or a glob (`"y_*"`); a glob is expanded in the order the plan builds the fields (pattern by
pattern, column by column), a list in the order written. Several fields become one tensor `(batch, n)`: the loss
compares the whole block in one call, and the report inverts every column with its own chain.

The definitions inherit from this table: a loss or a metric that names `output` and no `target` takes the
selector of that wire (with a single entry in the table a definition that names neither takes it too); an explicit
`target` wins. Not needed with one target field or one output wire. With several of both, writing it is a `check`
error away (`targets_missing`), because nothing else pairs the outputs with the targets and the report would write
no predictions.

### `epochs` or `steps`

`epochs: N`, or `steps: {total: T, turn: K}`. A step is one update pass; k batches with `accumulate: k`. In the
`steps` mode a turn is K steps and the total is T; the batch iterator lives across turns and does not enter the
checkpoint (resuming restarts the data stream). `global_step` counts steps.

### `loss`

The name of the loss with a single optimizer; the only place to write it when the optimizer is inline.

### `checkpoint`

The intermediate records during training: a lego or `null`.

| Writing | Writes |
|---|---|
| `{uri: best, params: {monitor: val/rmse, mode: min}}` | `best.pt` on improvement and `last.pt` every turn; `mode` defaults to `min`; `last: false` leaves `last.pt` out |
| `last` | `last.pt` every turn |
| `{uri: snapshot, params: {every: 10}}` | `snapshot_<n>.pt` |
| `null` | no intermediate record |

`last: false` on the `best` policy writes only the improving turns; the price is that a run interrupted mid
training cannot be resumed (`kalfa resume` falls back to `final/`, which exists once the run has ended), while
`report: best` is unaffected because `best.pt` is still written. The watched value is `train/` or `val/`; watching
`test/` is a `check` error. When the watched value is absent in a turn (`every: k`, NaN) the checkpoint does not
see that turn.

### `report`

`best | last`: the identity of the model after training. When the run ends `final/` is always written (every
model, the EMA copies, the optimizers, the counters, the rule states, the RNG). `plots`, `predictions.parquet`,
`generate`, `kalfa predict` and `kalfa generate` work with the `report` choice. `best` can only be written with
`checkpoint: {uri: best}` (a `check` error otherwise); `last` is always valid.

### `stop`

A list of triggers; the loop ends when one fires ("or"), and runs for `epochs` turns when the list is empty. The
triggers are shared with the rules: `plateau`, `metric_below`, `metric_above`, `after_turn` (alias
`after_epoch`), `time_budget`, or a user lego. The patience counts only the turns in which the watched value
exists.

A `stop.json` in the record ends the loop after the current turn as well: `kalfa stop <record>` writes it (into a
sweep root and its running points), the board's stop button and ctrl-c during training do the same, and
`touch <record>/stop.json` works from a cluster shell. The run then ends like an early stop, with its final state,
predictions and plots, and `run.json` says ok. A point that starts under a stopped root (its `stop.json`) exits
with an error instead of running.

### `rules`

A list. An item is `name`, `when` (a trigger), `set` (dotted path targets), `after` (this rule is not evaluated
until the named rule fires, and it is evaluated from the turn after the one it fired in, so it does not look before
the effect of the previous rule is seen) and `sticky`.

```yaml
rules:
  - name: cut_lr
    when: {uri: plateau, params: {monitor: val/rmse, patience: 5}}
    set: {main.lr: {times: 0.5}}
    sticky: false                      # asked again every turn; halves the rate at every plateau
  - name: to_huber
    when: {uri: after_epoch, params: {at: 50}}
    set: {loss: loss_huber}
```

* Rules are evaluated at the end of every turn in list order, with `metrics` and `losses` computed, and are
  effective in the next turn.
* A rule that fired stays sticky and is not asked again; its absolute effects are re-applied every turn.
  `sticky: false` asks it again every turn: its trigger starts over after a firing (the cooldown of a
  ReduceLROnPlateau) and its effects apply once per firing.
* A value may be relative for an optimizer param, `{times: 0.5}` or `{plus: -0.1}` (`describe` prints
  `main.lr := ×0.5`).
* Among rules writing the same key the last one wins.

The `set` targets:

| Target | Changes |
|---|---|
| `<opt>.loss`, or `loss` with a single optimizer | the optimizer's loss |
| `<loss>.<param>` | a loss param |
| `<loss>.<param>.<key>` | one key of a loss param that is a mapping (`loss_total.terms.loss_combined: 1.0` for a `weighted_sum`); the key is written in the definition with the value it starts with, `0.0` for a weight that begins silent, and the new value is of the same kind |
| `<loss>.<param>: {...}` | the whole mapping |
| `<opt>.<param>` | an optimizer param of the default group (`main.lr`); a group that does not write the param follows it |
| `<opt>.<group>.<param>` | the param of the group of that name, or of every group a glob matches (`main.lambdas.lr`, `main.lam*.lr`) |
| `<opt>.*.<param>` | the param of every group, the default one included; a relative value (`{times: 0.5}`) applies to each group's own value, an absolute one is written into each |
| `<model>.trainable` | opens or freezes a trained model |

The effects are applied once, at the start of the turn, to the models, the optimizers and a copy of the loss table
that the turn and the evaluation of every set read alike, so a loss a rule changed is the same loss on `train/`,
`val/` and `test/`. The rule state (the slot, the trigger counters) is carried and saved.

`check`: when the value the `when` trigger watches is on a set that does not exist, an error (`set_missing`, the
same rule as `stop` and `checkpoint`); when the value is absent in a turn on an existing set the trigger does not
see that turn. It verifies that the target exists and that the value fits the signature and the `refs` type: for
`vae_loss.recon: huber`, `recon` is of type `criterion`, `huber` is resolved from the flat table, its kind is
checked and it is built at compile time.

### The history

Per turn: the `train/`, `val/`, `test/` metric and loss values (losses in the model scale, metrics in the original
scale), `global_step`, the rule firings, the learning rates (`lr/<opt>`), the loss every optimizer minimized
(`minimizes/<opt>`), every other effect in force (`effect/<target>`) and the seconds of the turn. Section 8 has
the line in full.

## 7. After training, the record and the commands

### `generate`

`generate: {uri: sampler_lego, params: {...}}`: a lego that takes the models by name (`unet.ema`, `generator`;
`refs` type `model`), runs at the end of training with the `report` model and again with `kalfa generate runs/x`,
and writes its output under `samples/`.

To see samples during training a `sample_writer` is written in the `metrics` table:
`{uri: sample_writer, params: {sampler: generate, n: 16}, every: 5, sets: [valid]}`. `sampler: generate` is the
call of this section (without it, the output of the `predicts` model); it writes `samples/turn_<n>.png` and `.pt`
and no value to the history. The `samples_gif` and `samples_matrix` plots produce an animation and a turn by
sample matrix from those files, and skip with a warning when there are none.

In the language model the tokenizer is read from the fitted preprocessor; the window of `lm_sampler` is the
`context` param, the model's `seq_len` when not written, and sampling is limited to the tokenizer's vocabulary.

### `calibrate`

What the run learns at the end that is neither weights nor a column transform: a mapping of named legos of the
`calibrate` kind, fitted on the report models and the sets once training has ended (after the `report` choice,
before the predictions), kept under `fitted/calibrate/` beside the preprocessors and the frame transforms with
their notes in `calibrate.json`, and applied to the prediction table by the run and by `kalfa predict`.

`threshold` (`{set: valid, quantile: 0.95, output}`) reads the quantile of the raw output of the `predicts` model
off a held out set and marks the rows above it as `flag_<output>`. A lego of your own subclasses
`kalfa.std.calibrate.base.Calibration` with `fit(models, loaders, prep, device, predicts)`, `apply(table)` and
`note()`; the same slot holds a class prior, a calibration curve or an ensemble weight.

### `plots`

Plot legos that run once training has ended. The file is named after the definition (`plots.roc` writes
`plots/roc.png`): `run_all` passes the definition name to a lego whose signature has `name`; a lego without it
writes under its own name and cannot be used in two definitions (`plot_name_clash`).

What a lego may take by name in its signature: `predictions` (the test predictions, original scale), `history`,
`models`, `loaders` (one per set the split returns), `predicts` (the name of the prediction model), `sets` (the
sets the definition names), `name`, `figures`, and everything the run has on the bus at that point: `prep`,
`train_loader`, `valid_loader`, `test_loader`, `composites` (inside `models` too), `device`, `counters`,
`optimizers`, `emas`, `rules`, `losses`, `losses_keys`, `data_report`. The list is one mapping in the contract
(`wiring.plot_bus`), so a plot never needs a framework change to reach a value the run already holds. With `prep`
and a loader a plot reads the data of a set in its original units (`kalfa.std.plot.set_frame`), which is what the
data plots draw. An extra input comes through `inputs: {param: name}`, and the lego declares the type of the name
with `refs` (`history` a history key, `field` a field, `model` a model).

A plot declares the bus keys it cannot work without with the `needs` fact (`needs=["train_loader"]` on the data
plots whose default set is `train`); `run_all` skips such a plot when a key is missing and says why in the log
(`target_vs_features skipped: the bus has no train_loader`), so nothing is silent and no lego is defensive about a
key that is simply not there.

| Plot | Draws |
|---|---|
| `loss_curve` | every history series over the turns, or the named ones; `x: step` the per update series; `rates: true` the learning rates below |
| `pred_vs_true` | predicted against true, one panel per predicted field with its R2, hexbin over many points |
| `pred_histogram` | the distribution of every predicted field over its truth on the same bins, the truth filled and the prediction outlined, the ratio of the counts with its Poisson error below |
| `residuals` | the residual distribution, the residual against the truth, the error over the target range |
| `error_map` | the error over a 2d grid of two columns |
| `permutation_importance` | the drop in R2 when a feature is shuffled |
| `target_vs_features` | one hexbin panel per feature with the median profile |
| `correlation_heatmap`, `target_correlation`, `feature_distributions` | the data of a set in its original units |
| `pairplot`, `violin`, `kde` | the seaborn wrappers; skipped with a warning when seaborn is not installed |
| `confusion_matrix`, `class_histogram`, `binary_roc`, `binary_precision_recall_curve` | classification |
| `forecast_samples` | true and predicted horizons of sample windows |
| `image_pairs`, `image_grid` | input and reconstruction pairs; the outputs of the `predicts` model on the report set (test, else valid) |
| `architecture` | kalfa's drawing of every report model: one box per graph node with its name, what it is and the shapes one batch traced through it, the wires as labelled arrows, the losses and the optimizers beside the outputs they read |
| `architecture_text` | the module repr of every model as text |
| `data_pipeline` | the data block as one picture: every stage with its rows and columns, the split, the fitted objects, the features and targets, the loaders, and under the fit before and after histograms of the columns with the longest chains (`columns` names others); it reads the `data_report` the data block ends with, so a stream or a Dataset source keeps the stage lines and drops the histograms |
| `torchview` | torchview's drawing per model when torchview and the graphviz `dot` binary are installed |
| `samples_gif`, `samples_matrix` | from the `samples/turn_*` files of `sample_writer` |

Two commands draw outside the run. `kalfa plots runs/x` redraws the section from the record: the predictions and
the history from their files, the models from the report weights, the loaders from the data block replayed without
a fit, the optimizers, the counters and the rules from `final/state.pt`; nothing is fitted or trained again.
`kalfa predict runs/x --plots` draws right after predicting, on the predictions it just made, with the test loader
it built (`predict --data` builds no other set). The files follow the predictions: `predict --data new.parquet
--plots` writes `plots/pred_vs_true_new.png` beside the run's own plots, and predicting the run's own test set
with the report model overwrites them, which is the refresh case.

### `figures`

The look of every plot, in one place.

| Key | Means |
|---|---|
| `format` | `png` (the default), `pdf` or `svg` |
| `width`, `height` | the size of **one panel** in inches; a plot that lays out a grid multiplies them, a plot whose height follows its content (a bar chart) takes the width only |
| `dpi` | the raster resolution |
| `style` | `kalfa` (the built in palette and grid) or `none` (matplotlib as the environment configured it) |

Nothing written means every plot keeps the size it was written with and writes `png`. A single plot overrides the
size with the definition level keys `width` and `height` (`plots: {corr: {uri: correlation_heatmap, width: 11,
height: 11}}`), and `--set figures.format=pdf` overrides the section from the command line. The per turn sample
files of `sample_writer` stay `png` whatever the format is, because `samples_gif` and `samples_matrix` read them
back. The section is the params of the figures lego of the contract's wiring (`figures: /lego/kalfa/figures`),
built once after training and handed to every plot that names `figures`; `check` validates the section by
building the lego, so a figures lego of your own carries its own keys and palette.

### `sweep`

`sweep: {strategy, space, objective, record}`.

| Key | Means |
|---|---|
| `strategy` | a strategy lego: `grid`; `random(count, seed)`, `sobol(count, seed)`; `optuna(trials, seed)`, fed back and local loop only |
| `space` | a param name to a list of choices or to `{low, high, log, int, steps}`; the grid needs `steps` on a range |
| `objective` | `{monitor, mode: min or max, at: best or last}`, read from the point's history |
| `record` | the root directory; `--record` overrides |

Every point is an ordinary record under `<root>/<id>/` plus `sweep.json`. The commands are below; the table comes
from `kalfa collect <root>`.

### `record`

The record directory; `$datetime$` is built in (`YYYYmmdd_HHMMSS`, filled once at the start of the run).

| File | Holds |
|---|---|
| `manifest.json`, `host.json` | the identity (section 8) |
| `resolved.yaml`, `contract.yaml`, `flow.yaml` | the config as run, the contract it was compiled by, the graph that ran |
| `history.jsonl` | one line per turn: `{"turn": 3, "global_step": 1200, "train/loss_mse": 0.41, "val/rmse": 0.9, "lr/model": 0.001, "rules": ["to_huber"]}` |
| `steps.jsonl` | one line per update: `step`, `turn`, `loss/<optimizer>`, `lr/<optimizer>`, `grad_norm/<optimizer>` under `grad_clip` |
| `events.jsonl`, `run.json`, `stdout.txt`, `stderr.txt` | tezgah's event stream and summary, the console |
| `checkpoints/`, `final/` | `best.pt`, `last.pt`, `snapshot_<n>.pt`; the final state |
| `fitted/preprocessors/`, `fitted/frames/`, `fitted/calibrate/` | one file per preprocessor and `plan.json`; the fitted frame transforms; the fitted calibrations and their notes |
| `predictions.parquet` | the source row id `row`, the target fields inverted, `pred_<output>` inverted, `raw_<output>` raw, the calibration flags, the spectators; with `training.targets` one column per predicted field, `pred_<output>_<field>`; on the test set with the `report` model, not written without a test set; `predictions_<set>.parquet` for every set beyond the wiring's three |
| `plots/`, `samples/` | in the format `figures` asks for; what `generate` and `sample_writer` wrote |
| `data.json` | the shape of the data at every stage of the data block: the rows and columns of the source and after every transform, the sets after the split, the fitted objects, the features and targets, the loaders; greppable, and a diff between two runs shows what changed in the data before anything else is compared |
| `architecture.json` | the graph of every report model with the traced shapes, laid out by column and row; the board draws it |
| `device.json`, `git.json` | the chosen device and its lego; the commit of the config's repository and whether it was dirty |
| `export/`, `resume.json` | what `kalfa export` wrote; the source run of a resume |
| `stop.json` | a stop request, who asked and when; the loop ends after the turn that sees it |
| `failure.json` | when the run fails: every failed node with its path, the exception and the full traceback; a config that fails before training writes it too |
| `heartbeat` | touched every minute while the run runs and removed when it ends; one older than ten minutes without `run.json` makes the record lost |
| `repair.json`, `attempts/<n>/` | every `kalfa repair` of the record (its stage, its source, its turn, its host, how it ended) and the files of the attempt it replaced |

### The commands

#### `kalfa --version`

The versions of kalfa, cirak, tezgah, torch and python on one line, what to quote when something does not work; a
`pip install kalfa` that silently fell back to an older release shows up here first.

#### `kalfa run`

`kalfa run cfg.yaml [--set path=value ...] [-p name=value ...] [--executor serial|thread --workers N]
[--log info|debug] [--no-progress] [--progress turns] [--log-every N] [--contract PATH]`.

An error when `record` points at a non empty directory; nothing is overwritten. The thread executor treats the
aliasing warning as an error. Ctrl-c while the training loop runs asks the run to stop: it writes `stop.json`
into the record, the loop ends after the current turn and the after block runs (the final state, the predictions,
the plots); a second ctrl-c, or one before or after the loop, aborts at once.

`--log` prints to stderr what the run is doing while it does it, and changes nothing else: no key in the config,
no file in the record directory (the lines land in `stderr.txt` all the same, because tezgah tees stderr while the
run is live).

| Flag | Shows |
|---|---|
| `--log info` (a bare `--log`) | the narrative: the config files and the plugins, the device, the record directory, the rows read, the set sizes, the preprocessors as they fit, the batch counts, the parameter counts, the optimizers, one line per turn, the checkpoints written, the rules that fired, why training stopped, the predictions and the plots |
| `--log debug` | every node of the pipeline with its time (the same nodes `events.jsonl` records) and the decisions inside the legos: which preprocessor fitted how many columns, how many steps each optimizer took, why a turn was not the best, which rules did not fire |
| `--no-progress` | no progress bar; `--log info --no-progress` is the plain form, one line per turn and nothing that redraws itself |
| `--progress steps` | an inner bar over the steps of a turn |
| `--log-every N` | a line every N steps under `--log`: the loss, the learning rate and the gradient norm of every optimizer |

Without the flag the output is the progress bar and the warning summary; with it a warning also appears at the
moment it is raised. A line is `time level stage message`, the stage being where it comes from (`data.source`,
`models`, `training.turn`, `after.predict`); at `debug` the stage of a node line is its path in the flow
(`training.epochs[3].turn`). `resume`, `sweep`, `predict` and `generate` take the same options. None of the flags
is a config key: how a run displays itself is not part of the recipe.

#### `kalfa check`

`kalfa check cfg.yaml [--layers] [--measure] [--dump] [--recipe]`. It answers one question, are there problems,
and reports instead of raising. The format is cirak's `Problem`: `severity kind message file:line hint`.

Errors: unknown and missing keys, an unresolvable name, a kind mismatch (`kind_mismatch`), the `refs` resolution,
a signature mismatch, the `set` target and value, a column matching no glob or an ambiguous one, a reserved field
name, a reference to a dropped column, a missing set and watching `test/`, an unused optimizer, the need for
`predicts`, `report` against `checkpoint`, `needs_grad` against `sets`, `amp` against `scaler`, the `weights`
structure, a dot in a model name, an `order` that does not fit the turn, what the lazy set cannot do
(`lazy_split`, `lazy_batch`, `lazy_feed`, `lazy_data`), two grouped preprocessors written in opposite orders in two
chains (`grouped_order`), two definitions of a nameless plot lego (`plot_name_clash`), the `sweep` section
(`sweep_space`, a missing param, the objective), `sampler: generate` asking for the `generate` section
(`generate_missing`), the target table (`targets_missing` when several wires meet several target fields and
nothing pairs them, `targets_not_a_wire`, `target_not_a_field` for a glob that matches nothing, `target_missing`
for a definition whose target cannot be resolved).

Warnings: a target name the data does not carry as a field (`target_not_a_field`; only a feed that writes it puts
it in the batch), an unused preprocessor, a model without an optimizer, an unused wire, an unseeded run, a turn
that takes no `metrics`, aliasing (`aliasing`: of two unordered siblings one mutates in place the object the other
reads; an error under the thread executor), a preprocessor collecting its column on the lazy set (`lazy_fit`).

It prints nothing else: the set table, the implicit bindings and the rest of the analysis belong to `kalfa
describe`. `--measure` runs the data block and reports what it computed (source, filters, split, the preprocessors
fitted on train, the feed and the loaders) with the real set sizes after the filters; `--dump` prints the graph
that would run, `--recipe` the driver document.

#### `kalfa describe`

`kalfa describe cfg.yaml [--measure] [--section data|model|training|after|columns|wiring] [--wiring] [--save
report.txt] [--set ...] [-p ...]`: the same checks, then the config as an analysis. A record directory in place of
the config (`kalfa describe runs/x`, the same for `check` and `run`) reads its `resolved.yaml` under its
`contract.yaml`.

| Section | Shows |
|---|---|
| DATA | source, split, batch, feed, filters, the field table, the path of every set through the preprocessors |
| MODEL | one line per trained model, the spec as a chain and a graph block as a node table, composites with their `{model:}` nodes, ema, init, weights, frozen |
| TRAINING | turn and horizon, the optimizer table, the losses and metrics with the sets they are reported on, checkpoint, stop, the rule chain with its trigger conditions |
| AFTER | report, predict, plots, generate, record |
| COLUMNS | every source column with its dtype, field, preprocessor chain, role and its place in the feature tensor |

Without `--measure` it reads the file header and the recipe only; `--measure` runs the data and model blocks
(nothing is written) and fills in the sizes after the filters, the fitted column widths and the parameter counts.
A source lego of your own has no header to read (`kalfa.std.source.header` knows the std sources), so the column
table and the expansion of a target glob come from the fitted plan under `--measure`. `--section` prints one
section, `--wiring` adds the implicit bindings. The tables are fitted to the width the terminal reports (`COLUMNS`
overrides it); `--save report.txt` writes the analysis to a file and, like any output that is not a terminal, clips
nothing. The exit code is 1 when there are errors; sections that need a shaped config say so instead.

#### `kalfa predict`

`kalfa predict runs/x [--model encoder] [--which best|last|final] [--data new.parquet] [--device cuda]
[--plots [a,b]]`: predictions from a record. On the run's own test set `<run>/predictions.parquet` is rewritten;
with `--data` the file is `<run>/predictions_<file name>[_<model>].parquet` (`--set` only overrides config paths,
new data is given with `--data`). `--model` is any model, composites and `.ema` included; the defaults are
`predicts` and `report`. New data goes through the `fields` mapping and the `feed`, and the model's `inputs` wires
are bound from the batch by name; a wire missing from the batch is an error. `--device` takes a device lego value
(a short name or a `{uri, params}` call) and moves the models onto it; without it the run stays on the cpu.

#### `kalfa generate`

`kalfa generate runs/x [--which best|last|final] [--device cuda] [--set ...]`: the `generate` section over the
record's models, into `samples/`.

#### `kalfa export`

`kalfa export runs/x [--format onnx|pt2|state_dict] [--model name] [--which best|last|final] [--out DIR]
[--device cuda]`: a model of a record in another format, written under `<run>/export/<model>.<suffix>` by a lego
of the `export` kind. `onnx` needs the `onnx` package and the wires are the input and output names; `pt2` exports
one traced batch with torch.export, the batch dimension dynamic; `state_dict` the plain weights. A plugin writes
its own format with `(model, inputs, directory, stem)`.

#### `kalfa plots`

`kalfa plots runs/x [--only a,b] [--set figures.format=pdf] [--device cuda]`: redraws the plots section of a
record from its files and its data; `--only` a comma separated list of definitions. The answer to "I changed the
plot" and to "I want them as pdf now". `--set` and `-p` apply to the resolved config of the record, so `figures`
and the plot params can change; the data is read again, nothing is fitted or trained.

#### `kalfa resume`

`kalfa resume runs/x [--set training.epochs=N]`: from `last.pt` when it exists, else from `final/` when the run
has ended; an error when neither exists. It opens a new directory; there is no `training.resume` key. Both files
carry the state of the checkpoint policy (the best value so far, the snapshot count), so a resumed `best` policy
overwrites `best.pt` only on a real improvement.

#### `kalfa repair`

`kalfa repair CONFIG... [--set path=value] [-p name=value] [--contract PATH] [--record DIR] [--id N]` continues a
failed run in its own directory, with the config files it was started with (never the record directory; `--record`
when the config writes a new directory per run through `$datetime$`). Where it continues from is read from the
record: with `final/state.pt` training was over and only the after block runs again (`init_state` loads the final
state and skips training); with `checkpoints/last.pt` training goes on from that turn, and `history.jsonl` and
`steps.jsonl` are cut back to it; with neither it is refused, with the way to make a run repairable
(`training.checkpoint: last`, or `best`). The files of the failed attempt (`run.json`, `failure.json`,
`events.jsonl`, the console, the notes, the heartbeat) move to `attempts/<n>/` first, and `repair.json` lists every
attempt. The plugins next to the config are the code that runs; the config may differ from `resolved.yaml` only
under `plots`, `figures`, `device`, `calibrate`, `generate`, `params`, `record`, `include`, `plugins`, `sweep` and
`training.report`, and a change anywhere else (a model, the data, an optimizer, a loss, a metric) is refused with
the keys that changed, since the checkpoint and the history so far were made under the recorded config. Only a
failed record is repaired: a finished, running, pending or lost one is refused. With a sweep config it takes the
arguments of `kalfa sweep`: `--id N` repairs point N (a queue job), and without it every failed point is repaired
one after the other, the others skipped and counted in the first line.

#### `kalfa prepare`

`kalfa prepare cfg.yaml --out DIR` runs the data block once and writes it: `manifest.json` (kind `data`, a hash of
the resolved data section, the sets and their sizes, the header), one parquet per set (the frame after the
transforms, the split and the fitted frame transforms, with the row ids), `fitted/` (the preprocessors and the
frame transforms) and `data.json`. A Dataset source keeps its items where they are and the directory holds the
split as position lists; a stream source is lazy already and is not prepared.

`kalfa run cfg.yaml --prepared DIR` starts from it: the data block reads the directory (`wiring.prepared_source`,
`wiring.prepared_split`, `read_prep`, no transforms), the feed and the loaders are built as always, the run copies
`fitted/` into its own record, and a data section that differs from the one prepared is an error
(`prepared_mismatch`). `check` and `describe` take `--prepared` too and read the sizes from the manifest instead
of running the block.

#### `kalfa sweep`

`kalfa sweep cfg.yaml [--record root] [--set ...] [-p ...]` is the local loop: every point in a subprocess
(`python -m kalfa.cli sweep --id N`), finished points kept. `--count` prints the number of points, `--show N` the
point, `--id N` runs that point only (the strategies deterministic by id: grid, random, sobol; for queue systems,
it looks at no other point; an error with optuna). `--set` and `-p` apply to every point.

`kalfa sweep cfg.yaml --plan [--prepare-data] [--record root]` writes the root once, before any point starts:

* `manifest.json` (kind `sweep`: the strategy, the space, the objective, the total, the config, whether the data is
  prepared);
* `sweep.plan` (`N`, `CONFIG`, `ROOT`; regenerated on every plan);
* the two site files it never overwrites: `sweep.sub` (an HTCondor submit file: `include : sweep.plan`,
  `executable = sweep.sh`, `arguments = $(Process)`, `queue $(N)`) and `sweep.sh` (the environment of the site,
  commented, then `kalfa sweep "$CONFIG" --id "$1" --record "$ROOT" --no-progress --log info`, so a job log is a
  line per turn and not a redrawn bar).

`--prepare-data` also runs `prepare` into `<root>/data/` and says so in the manifest; a point reads the manifest
and starts from the prepared data when it is there and from the source when it is not, so a dataset too large to
prepare on the submit host is left to the nodes. A swept name the data section reads (`$width$` under `data`)
makes the data differ per point, and `--prepare-data` refuses then and says why. The local loop plans the same way
and runs the points itself; a point's record is a run with `kind: point`, its values and its root in the manifest.

#### `kalfa collect`

`kalfa collect <root>` walks the point records of a sweep root (it reads the root manifest for the objective), puts
those with a `sweep.json` into the table (`sweep.csv`, `sweep.json`) and writes the report `sweep.md`: the counts,
the best point with its params, its `val/` and `test/` values at its objective turn and a copy of its plots
(`best/`), the top points with their gap to the best (`--top K`, 5 without it), every param against the objective
(`plots/param_<name>.png`, a table per level for a choice), the objective curves of every point with the top ones in
colour (`plots/curves.png`), the failed, lost and unfinished points with the first error of each, the
`resolved.yaml` difference of the two best, and every finished point in a closed section. It works on a half done
sweep; `--no-figures` skips the drawing. The terminal carries the swept params, the objective and the turn of every
point with the best one marked, and `--markdown` prints the report instead.

The parameters section opens with one number per param, also in `sweep.json` as `importance`: the Spearman rank
correlation of a numeric param with the objective (three points or more) with the way it is better, and for a choice
the share η² of the objective's variance its levels explain with the level of the best mean (when a level holds more
than one point, since one point per level explains everything). Both look at one param at a time. The point records
are read eight at a time, which is what a sweep on a network filesystem waits for.

`--mean-over PARAM` names a swept param whose values the space lists, `fold` for a k fold inside the sweep. The
points that share every other param are one setting; a setting is ranked by its mean objective over the values of
`PARAM` once every value has a finished point with a finite objective. The report opens with the best setting (its
mean, its deviation, its point per value and the mean of its `val/` and `test/` values at the objective turn), the
top settings and the settings still missing a value; the param figures and level tables take the setting means,
the curves colour the points of the best setting, and the config difference pairs the two best settings at the
same value of `PARAM`. `groups.csv` holds every setting.

`kalfa collect runs/cv_*` is the k fold summary when there is a `fold` param (the mean and deviation of every
fold's `test/` metrics at its last turn, the values per fold, the same at the reported turn when
`training.report` is `best`, the state of every fold and their curves, `cv.json` and `cv.md`); otherwise the sweep
table of the list of runs, the `params` differences as columns. The files go to `reports/` inside the directory
given, or inside the parent of the runs when there are several (`runs/reports/cv.md`); `--out DIR` puts them
elsewhere.

#### `kalfa ls`, `kalfa docs`, `kalfa contract`

`kalfa ls [/alias/kalfa/tabular | /criterion | word] [--kind kind]` lists the packs and the legos with their
facts; a word without a leading slash (`kalfa ls tokenizer`) searches the URIs, aliases and descriptions and shows
the aliases with their packs. `kalfa docs --write DOCS.md` writes the lego reference generated from the registry.

Both take `--plugin module` (a module name, or a path to a `.py` file) and `--config cfg.yaml` (the modules of the
config's `plugins` section), repeatable: the modules are imported before the listing, so your own legos are listed
with their signature and facts. The config is read, not validated, so a half written config still brings its legos
in; an import that fails is printed and the exit code is 1. `docs` prints them in a separate `Plugin legos`
section, so `DOCS.md` (written without the options) stays the reference of what kalfa ships.

`kalfa contract [--write PATH]` prints the contract kalfa runs configs by: the `wiring` (the adapters, the
builder, the default split and device, the prep steps `fit` and `read_prep`, the run inputs, the history prefixes,
the loader, the figures lego, the plot bus) and the five flow `blocks`; `--write` exports it for editing.
`--contract PATH` on `run`, `check`, `describe`, `predict`, `generate` and `resume` runs against the edited copy;
a command over a record takes the record's own `contract.yaml` without the option.

#### `kalfa stop`

`kalfa stop <record> [<record> ...]` asks a running record to stop after its current turn by writing `stop.json`
into it (who asked and when); a sweep root gets the file with every running point, so no new point starts and the
running ones end after their turn. The run ends like an early stop: the final state, the report model, the
predictions and the plots are written and `run.json` says ok. A record that has ended is an error.

#### `kalfa board`

`kalfa board <root> [--host 127.0.0.1] [--port 8080]`: a reader of records. A `http.server` with JSON endpoints
and one page, the Vue application under `src/kalfa/board/static/` served at `/static/`; Vue and Plotly come from
jsDelivr at pinned versions with their hashes, so the browser needs the internet and the server does not. It
finds the manifests under the root (a record made before the manifest by its `resolved.yaml`),
shows the tree, and follows the records as they change: one background loop stats the files of the open records
every two seconds for every open tab, stops half a minute after the last tab closes, and pushes the names of the
changed ones over a server sent event stream (`/api/watch`); the page reloads only those (the growing files by
offset), or polls every 3, 5 or 10 seconds when the footer says so. A small file is parsed again only when its
size, time or inode changed, a growing one is read from where it was left, a log from its end, and the last four
prediction files stay in memory while unchanged. A record that cannot be read is listed as unreadable instead of
taking the page down, and a page that loses the server, or the single sign on in front of it, says so in a band. The page
shows a spinner until the tree, the live list and the record it opens on have answered. The address bar carries
the record, the tab, the open chart and the view options, so a link shares one view and a reload keeps it.

| Page | Shows |
|---|---|
| home | what is running with its progress and what finished lately; the tab title of the browser carries the finished turns over the planned turns of every live run as a percentage (`Kalfa Board (59%)`), and the icon's colour says the state on every page: grey when nothing runs, amber while something runs, red when the latest record failed or the open record did |
| table | every run and point with its params, best value and last values, sortable; a filter of words, each a comparison (`lr<0.01`, `val/rmse<=0.3`) or a text, chips for the states and a menu for when it started (the sidebar has the same over the tree), a columns menu kept per root, and the rows as shown saved as CSV |
| compare | any number of ticked runs or points (`#/compare?runs=a,b,c`): one chart per history series with a line per record, a colour and a dash each, the params that differ, and the `resolved.yaml` difference of two |
| a run | the tabs below |
| a sweep | the live table of points (finished ones from `sweep.json`, running ones with the best value so far from the history tail), every point drawn across the whole space (drag on an axis to narrow the charts and the table together), an explorer with its own axes, every param against the objective, where the failed points died, the objective over the sweep order, the curves of the ticked points overlaid (any series, the best K ticked in one step) and the `resolved.yaml` difference between two of them, the failed points grouped by their first error, the points saved as CSV; the queued points count, so a half started sweep is not finished; the table and the axes say epochs when the points count epochs |

The tabs of a run:

| Tab | Shows |
|---|---|
| monitor | the turn and batch bars, the checkpoint monitor across the sets, the loss every optimizer minimizes, the batch loss per step, the latest values, the log tail |
| overview | the latest turn per series with the lego behind it, the weights and params the rules left and their course over the turns (from `effect/`), the learning rates, the best turn, the checkpoints and the calibrations |
| loss and metrics | the turn curves, one chart per series with the sets as lines, the rules that fired named on their marks |
| optimizer steps | the step curves with the turns marked and labelled along the top; the tooltip names the turn of a step |
| model | the architecture from `architecture.json` as a schematic: one block per node with its input pins on the left and its output pins on the right, every wire with its width, the losses and the optimizers beside the outputs. A block clicked opens in place (the layers of a node chained left to right with the tensor shapes on the wires between them, the nodes of another model, the features of the input); an inner block opens the same way. A block drags, the upright part of a wire drags to bend it, the wheel zooms and the background pans; the arrangement is kept in the browser per record, with fit and reset to undo it, save writes the png as arranged and expand opens it large |
| data | the data pipeline drawn by the same schematic and the stage tables of `data.json`; a stage clicked lists the columns it added or removed |
| predictions | the prediction against the truth with the y = x line on a sample of 2,000 rows or on every row, the residual histogram, the distributions of the prediction and the truth over the same bins with their ratio below, the largest errors, all over the rows a filter keeps (`/api/predictions` takes it and applies it before every number on the page is computed); a switch per target, kept in the browser, turns it into a classification (`/api/classify`): the ROC of the picked score column with its AUC, the background efficiency, the rejection and the cut at 50, 80, 90 and 95 % signal efficiency, the score by class and the confusion, nothing drawn until the score and the signal class are picked |
| prep | the fitted preprocessors and what each learned |
| plots, samples | the figures under `plots/` as a gallery and the sample images as they are written |
| config, notes, files | `resolved.yaml`; `manifest.json`, `host.json`, `device.json`, `git.json` and the node timings of `run.json`; every file of the record with a viewer for text, images and PDFs and a download for every file |
| timeline, events, logs, describe | the timeline of the nodes with a loop as one bar and every node inside it summed over the iterations (every iteration as a row on request), the event tail, the last 300, 2000 or 10000 lines of `stdout.txt` and `stderr.txt` with a progress bar as its last state and a link to the whole file, `describe` on demand |

Every chart zooms with a drag (a box) or the wheel and resets with a double click; a log axis shows its decades as
10^n with the digits between them as small ticks, and labels the digits when less than a decade is in view. Every
chart expands into a large view beside a settings panel: log scale on either axis, the range of every axis (the
ratio panel of the distribution too), the axis titles, grid, legend, font size, line width, points, the point
size, the curve, the height, the bins of a histogram, and a JSON box merged into the Plotly layout for anything
else; the download writes the chart as drawn, as png or svg, and reset restores the defaults. An axis of integers
(turns, steps, points) gets integer ticks.

The endpoints: `/api/tree`, `/api/live`, `/api/watch`, `/api/table`, `/api/record`, `/api/predictions`,
`/api/classify`, `/api/series`, `/api/prep`, `/api/files`, `/api/text`, `/api/events`, `/api/history`,
`/api/steps`, `/api/tail`, `/api/sweep`, `/api/diff`, `/api/describe`, `/file` (sent in pieces, with an ETag), and
the one POST, `/api/stop`, which writes `stop.json` into a record, or
into a sweep root and its running points. The board writes nothing else; the header of a running record and of a
live sweep shows the stop button, with a confirmation, anyone who reaches the page can press it, and a stopped run
continues from its last checkpoint with `kalfa resume`. On a batch system it runs where the files are readable and
the browser reaches it through an ssh tunnel; the nodes never talk to it. The filter of the predictions tab is
evaluated server side after a check that lets through column names, numbers, strings, comparisons, `and`, `or`,
`not`, arithmetic and lists only (no attribute, call or variable); still, `--host` beyond `127.0.0.1` hands the
stop button and the files to whoever can reach the port.

## Appendix (designer)

## 8. The mapping onto the lower layer

### The contract

The mapping is the contract, `src/kalfa/contract.yaml`, in two parts:

* `wiring`: the choices of the framework that belong to no lego. The adapters that wrap the losses and metrics
  entries, the builder, the default split and device, the prep step of a run (`fit`) and of a command over a record
  (`read_prep`), the run inputs `device`, `record` and `monitor`, the history prefixes, the loader whose params the
  `batch` section is, the figures lego whose params the `figures` section is, the plot bus.
* `blocks`: the five cirak flow blocks, `data`, `models`, `optimizers`, `training`, `after`.

What is a property of one lego is a fact on that lego instead, read from the registry, so a plugin lego gets the
same treatment as a std one: `sizes` and `header` name the helper that gives the set sizes of a split and the
header of a source, `stream` and `samples` say what a source yields, `needs_table` and `counts` what a feed or a
`/data/` component needs from the data, `writes` the files a checkpoint policy writes, `enumerates` that a strategy
can count its points, `describe` how a trigger is worded, `roles` the init roles of the builder, `requires` the
optional library a lego loads.

kalfa's Python side is a driver: it fills the block variables from the section keys with a fixed table and
performs the reshapings, all of which look at the shape and none at a config value: the single model shorthand,
the inline optimizer, trained against composite models, the inverse optimizer mapping, the EMA and `weights`
lists, the `split` short form, the `batch` mapping into one loader call per set, the `figures` section into the
figures call, the filter kinds, `model.templates` and the models into `blocks`, criteria, metrics and objectives
wrapped in their adapter by kind, the definition level keys (`sets`, `every`, `inputs`, `output`, `target`)
separated from the call and handed to the lego through the parallel `<section>_keys` table, the target of a
definition that names only its output wire filled in from `training.targets`. The driver document is
`tests/golden/recipes/<name>.recipe.yaml`, the expanded graph `tests/golden/dumps/<name>.flow.yaml`; both are
generated by `tools/regenerate.py` for every example.

`kalfa contract --write contract.yaml` exports the document, `--contract contract.yaml` runs a config against
the edited copy, and every record keeps the copy it ran with as `contract.yaml`, so a run stays reproducible when
the default changes. The contract is the extension point of the flow: a step kalfa does not have is a node in a
block of the copy (`after: {flow: {notify: {uri: /lego/acme/slack, inputs: {history: history}}}}`), and the same
way `training.sets` in the copy takes `train` (`sets: {default: [train, valid, test]}`) for a clean evaluation
pass over the train set, where by default `train/` is the running mean of the training pass. Neither needs a
change to kalfa.

| Section | Contract block | Generated nodes |
|---|---|---|
| `data` | `data` | the source; the transform chain before the split; the split; per set the transforms that name it; the frame transforms fitted on the train set (`fit_frames` in a run, `read_frames` over a record) and applied per set; the prep step (`fit` on the train set in a run, `read_prep` from the record under `predict`, `generate` and `resume`); per set the apply, the `feed`, the loader of the wiring with the `batch` mapping as its params; the data report reading what every stage wrote through pattern inputs (`df_*`, `*_df_0`, `*_frame`, `*_loader`) into `data_report` and `data.json` |
| `model` | `models` | per model a `{block, builder}` constructor with the builder of the wiring (the rng rule, the seed, the name and the index, `init`, `trainable`, `weights`); the EMA clones; the `models`, `emas`, `composites` bundles; per composite a constructor (with the `models` input) |
| `optimizers` | `optimizers` | per optimizer a constructor (the model bundle, `groups`, `schedule`); the `optimizers` bundle |
| `losses`, `metrics` | component tables | no node; callables built at compile time, passed as params to the turn and the evaluation; the definition level keys (`every`, `sets`, `output`, `target`) are separated from the call and go to the lego through the parallel table; the run time components once the data exists |
| `training` | `training` | the `counters`, `rules` and `stream` constants; `init` (the device, resuming with the checkpoint policy restored, `epochs_left`); the Loop (`carry: [models, optimizers, emas, counters, rules, stream]`, `until: stop`, `trace`): `effects` (the rules' effects applied once: to the models and the optimizers, returned as `models_ruled` and `optimizers_ruled` for the turn, and to a copy of the loss table put on the bus as `turn_losses`), the turn, `evaluate` per set (both reading `turn_losses`), `merge`, the rule chain (`rules_0..n`), `stop`, `checkpoint` (the `@checkpoint` policy of the driver document), `log` (the history line to the record and to the monitor) |
| `generate`, `plots`, `figures`, `calibrate`, `record` | `after` | `final/`, the `report` choice, the calibrations fitted on the report models and the sets (`@calibrate`), the test predictions with the calibrations applied, the figures lego, the architecture note, the plots, the generation |

### The monitor

The command builds one `Monitor` (`kalfa.std.common.log`) from `--log`, `--no-progress`, `--progress` and
`--log-every` and hands it to the run as the `monitor` run input beside `device` and `record`.
The history step gives it every turn line (`monitor.turn(line)`), the turn every update (`monitor.step(line)`),
and tezgah's events reach it through `monitor.sink`, so the bars, the turn and step lines and the node
timings are the monitor's business, never the turn's. The Python API builds the default one, a bar
without a log, when `run` is called without `monitor=`.

### The adapters

Every entry of the `losses` and `metrics` tables reaches the turn and the evaluation as one object of the `Loss`
contract (`kalfa.std.common.runtime`), built by the adapter of its kind: `/adapter/kalfa/criterion` feeds a
criterion the `predicts` model's output wire and the target field the definition keys name,
`/adapter/kalfa/metric` feeds a metric what its `update` signature names, `/adapter/kalfa/objective` calls an
objective with every model of the run and the batch, plus `step`, `epoch`, `rng`, `scaler` and `losses` (the other
entries evaluated on the same batch, by name) when its signature names them. The adapter answers
`loss(context, keys)` for the optimizer, `tracker(name, keys, rescale)` for the running mean of a pass (the model
scale under `losses`, the original scale under `metrics`, through the preprocessors), `with_param(name, value)`
for a rule effect and `resolve(**available)` for a `/data/` component among its params, built once the train
loader exists.

### The pass and the batch

What is fixed for one pass over a loader is a `Pass` (`kalfa.std.common.runtime`): the models, the composites and
the EMA copies, `predicts`, the target fields of the dataset, the epoch, the `Device`, the pass's own random
generator, the preprocessors, the set name, the loss scaler of mixed precision, the `losses` table with its keys,
the record. What changes per batch is a `Context` over it: the batch on the device, the global step, and the
outputs of the `predicts` model computed once and shared by every entry that reads `predictions`;
`target(name, output)` resolves a definition's target selector against the batch (`input` is the model's first
input wire), `rescaled(output, target)` gives both in the original units.

### The frames

`apply` turns one set into a `Frame`: a table (`data` holds the typed columns, `features` the feature order,
`targets` the target layout, `extra` the `spectators` and the columns a `column` typed reference names, which reach
legos by name, the data plots and `predictions.parquet`, and never the model), a Dataset source (`dataset` is the
sample store, `fields` the field names, `chains` the fitted preprocessors per field that apply to this set,
applied per item by the feed) or a stream (`stream` yields chunks of typed columns laid out like the table). The
`feed` builds a `Dataset` of `kalfa.std.feed.base` from it: `table` gives the feature columns as one tensor `x`
and every target field under its own name (a stream yields them chunk by chunk, the train pass shuffled through a
buffer), `window` sliding windows, `next_token` token windows.

### The record protocol

A thousand points on a batch system write into one shared filesystem while a person watches, so a record keeps
five rules, the rules of `kalfa.record.Record`:

1. one writer per file: a point writes only inside its own directory, the root of a sweep is written by
   `kalfa sweep --plan` before any point starts, and the one file written from outside is `stop.json`, the stop
   request, which the run only reads;
2. a file written whole is written to a temporary name in the same directory and renamed (`manifest.json`,
   `host.json`, `resolved.yaml`, `contract.yaml`, the checkpoints and the final state, the predictions, the plots
   and the sample images, `data.json`, `sweep.json`, `stop.json`);
3. a file that grows is appended one JSON line at a time, opened and closed per write, and a reader ignores a
   partial last line (`history.jsonl`, `steps.jsonl`);
4. identity is written once at the start: `manifest.json` (`kind` run, point, sweep or data, the name, the config
   paths, the params, the started time, the kalfa version, a hash of the contract, whether a turn is an epoch or a
   step count) and `host.json` (the hostname, the pid, the working directory);
5. status is derived, never written by a coordinator: a manifest alone is pending, a growing `history.jsonl` is
   running, `run.json` is finished or failed, a `failure.json` without `run.json` is failed, a `heartbeat` older
   than ten minutes without `run.json` is lost (the process was killed), and `Record.status()` says so with the
   time the record was last seen and the stop request when there is one.

kalfa recognises a record by its manifest, and one made before the manifest by its `resolved.yaml`, so the layout
above the records is the user's.

### The history

The `history` step appends one line per turn to `history.jsonl`: `turn`, `global_step`, every metric of the turn
under its set prefix (`train/`, `val/`, `test/`), the learning rate of every optimizer as `lr/<name>`, the loss it
minimized that turn as `minimizes/<name>` (the configured one, or what a rule set), every other effect of the
rules in force that turn as `effect/<target>` (the value, or the name of a lego; the board draws the weights and
params a turn ran under from it), the duration of the turn as `seconds` and the rules that fired. A series is a
numeric key with a `/` in it, `lr/` and `effect/` aside. `kalfa.std.common.history.History` reads the file and
the trace on the bus and answers `series(names)`, `last(prefixes)` and `best(monitor, mode, at)`; `collect`,
`sweep` and the plots read it through that class only.

The turn appends one line per update to `steps.jsonl`: `step`, `turn`, `loss/<optimizer>`, `lr/<optimizer>` and,
under `grad_clip`, `grad_norm/<optimizer>`; `History.read_steps` reads it and `loss_curve` with `x: step` draws
it.

### Two siblings and one object (the aliasing check)

A serial run orders nodes by their bus dependencies alone, so two siblings without a dependency run in an order
that happens to be right; the thread executor runs them side by side and the outcome depends on timing. `check`
warns for every pair of unordered siblings where one mutates an object the other touches; the alias table comes
from the legos' facts (`mutates`, `aliases`), the loop carry and the frame renames. `kalfa run --executor thread`
refuses a config with such a pair.

## 9. What fits the `alternating` turn and what does not

What fits, shown in the example configs: supervised (a single optimizer), the autoencoder and the VAE (the
reparameterization and the KL schedule inside the loss), WGAN GP (`order: [d, g]`, `steps: {d: 5, g: 1}`,
`fresh_batch`, the gradient penalty in the loss, `sets: [train]`), ALAD (two optimizers, five models), DDPM (the
time step and the noise inside the loss), SimCLR (the loss takes the two views, `sets: [train]`), the language
model (`steps`), distillation (the teacher's `weights` and `trainable: false`, the loss reads two models), loss
dynamics (rules change the optimizer's loss and the loss params), several losses on one optimizer
(`weighted_sum`), constrained training (`mdmm`, the multipliers as a model with their own optimizer group), a
regularizer every k steps (`step` from the signature, the condition inside the loss).

What does not fit needs a custom turn: reinforcement learning (interaction with an environment, not a batch
loop), the inner loop of meta learning (MAML: a temporary parameter copy inside a step), regimes taking batches
from different loaders in the same turn (unpaired image translation with two sources), regimes choosing the
optimizer by the batch contents, losses that need the optimizer state. They are written with
`training.turn: {uri: /turn/proj/x}`; the turn lego takes the same inputs as the std turn and writes the same
history.

## 10. Not in v1

An in process Map (k fold out of process; the sweep with `kalfa sweep`, one subprocess per point, section 7); a
`training.resume` key; a `generate` node during training (only at the end; samples during training through the
`sample_writer` metric); a separate key for the model mode (`trainable` decides); conditional expansion;
reloading `weights` during training; `bpe_tokenizer`; a preprocessor fitting by walking the items of a Dataset
source (fitting preprocessors work on fields readable as a column: table columns and the `text` field of
`text_lines`); inequality constraints and adaptive epsilons in `mdmm` (equality constraints only).

## 11. Alias packs

The table is flat: short name to URI, one name, one URI. The kinds are grouped for readability only; which section
a lego may be written in comes from its kind. Three packs include `/alias/kalfa/base`, the names they all share,
and add their own; `lazy_tabular` includes `tabular` and rebinds two names, so a config turns lazy by changing its
`include` line and nothing else. `kalfa ls /alias/kalfa/tabular` prints the resolved contents with the kinds. A
name two packs would share lives in base; a config that wants tables and images includes two packs (`include` is
pooled across layers, base is loaded once).

| Pack | Contents |
|---|---|
| `/alias/kalfa/base` | split: `sequential`, `given`; pre: `standard_scaler`, `minmax_scaler`, `cast`, `abs`, `log`, `one_hot`, `label_encoder`; feed: `table`; layer: `linear`, `linear_relu`, `concat`, `flatten`, `relu`, `leaky_relu`, `dropout`, `reparam`, `select`, `add`, `subtract`, `multiply`, `divide`, `negate`, `multipliers`, `l1_distance`, `gru`, `last_step`; init: `normal`, `xavier`, `kaiming`, `zeros`; criterion: `mse`, `huber`, `mae`, `log_cosh`, `cross_entropy`, `bce_logits`; objective: `weighted_sum`, `mdmm`; metric: `rmse`, `accuracy`, `f1`, `auroc`, `average_precision`; optimizer: `adam`, `adamw`, `sgd`; schedule: `warmup_cosine`, `linear_warmup`, `step_decay`; turn: `supervised`, `alternating`; trigger: `plateau`, `metric_below`, `metric_above`, `after_turn`, `after_epoch`, `time_budget`; checkpoint: `best`, `last`, `snapshot`; plot: `loss_curve`, `pred_vs_true`, `pred_histogram`, `class_histogram`, `confusion_matrix`, `forecast_samples`, `architecture`; strategy: `grid`, `random`, `sobol`, `optuna`; device: `auto`, `cpu`, `cuda`, `mps` |
| `/alias/kalfa/tabular` | base plus source: `parquet`, `csv`; split: `random_split`, `kfold`; pre: `max_abs_scaler`, `robust_scaler`, `quantile_transformer`, `power_transformer`, `asinh`, `sinh`, `tanh`, `atanh`, `logit`, `kbins_discretizer`, `spline_transformer`; feed: `window`; layer: `polynomial`, `l2_normalize`; plot: `target_vs_features`, `correlation_heatmap`, `residuals`, `error_map`, `permutation_importance`, `feature_distributions`, `target_correlation`, `pairplot`, `violin`, `kde`; data: `class_weights`, `target_weights`, `feature_width`, `feature_index` |
| `/alias/kalfa/vision` | base plus source: `parquet`, `csv`, `image_folder`; split: `random_split`, `kfold`; pre: `to_tensor`, `to_tensor_signed`, `resize`, `random_crop_flip`, `normalize`, `two_views`, `simclr_aug`; feed: `window`; layer: `unflatten`, `embedding`, `conv2d`, `maxpool`; objective: `vae`, `distill`, `wgan_gp_d`, `wgan_g`, `ddpm`, `ntxent`; metric: `fid`, `recon_error`, `sample_writer`; schedule: `linear_betas`; generate: `gan_sampler`, `ddpm_sampler`; plot: the data plots of tabular, `image_pairs`, `image_grid`, `samples_gif`, `samples_matrix`; data: `class_weights` |
| `/alias/kalfa/text` | base plus source: `parquet`, `csv`, `text_lines`; split: `random_split`, `kfold`; pre: `char_tokenizer`; feed: `window`, `next_token`; layer: `embedding`; metric: `perplexity`, `sample_writer`; generate: `lm_sampler`; plot: the data plots of tabular; data: `class_weights`, `vocab_size` |
| `/alias/kalfa/lazy_tabular` | tabular over a table read in chunks: `parquet` and `csv` bound to the stream sources (the `chunk` param), the `sequential` or `given` split, filters on the stream, buffered shuffling with `batch.buffer`; no `kfold`, `window` or `class_weights`, `balanced` and the random `split` are `check` errors (section 3) |
