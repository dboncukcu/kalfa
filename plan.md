# Refactor plan

A working document for one revision pass. It is deleted when the pass ends.

## Why

Three complaints, measured today:

| | now |
|---|---|
| std modules | 27 files, 7,039 lines; `pre.py` 1,275, `plot.py` 1,026 |
| core modules | 11 files, 4,833 lines; `check.py` 1,206, `describe.py` 766 |
| std legos | 161, spread over 29 (kind, pack) pairs, with no rule that ties a URI to a file |
| docstrings | 282 (there are no `#` comments) |
| URIs written into the core | 5 in `driver.py`, 1 in `api.py`, 2 in `recipe.py`, 12 in `check.py`, 5 in `describe.py` |

A reader who has a URI cannot find the code, a reader who has the code cannot tell what the framework
decided for them, and the two long files hide everything inside them.

## What must not change

The pass is a move, not a rewrite. Everything below stays true at every commit:

* The 161 URIs, their signatures, their facts and their descriptions. `DOCS.md` regenerates byte identical
  until a stage deliberately changes a lego.
* The 16 documents under `configs/dumps/`. Stage 3 changes them once, on purpose.
* The 354 tests, the 16 example folders, `kalfa ls`, `kalfa docs`, the aliasing check.
* The alias packs under `src/kalfa/packs/`. They are the short names; the tree is the addresses.

**Guardrail first (stage 0).** Before anything moves, a test writes the registry to a fixture: every URI with
its kind, alias list, signature, facts and description. Every later commit compares against it. A mechanical
move that changes one signature fails immediately instead of three stages later.

## 1. The file tree

The rule, with no exceptions: **the URI is the path.**

```
/plot/kalfa/pred_vs_true        ->  src/kalfa/std/plot/kalfa/pred_vs_true.py
/pre/sklearn/standard_scaler    ->  src/kalfa/std/pre/sklearn/standard_scaler.py
/optimizer/torch/adam           ->  src/kalfa/std/optimizer/torch/adam.py
/plot/torchmetrics/binary_roc   ->  src/kalfa/std/plot/torchmetrics/binary_roc.py
```

One lego per file. 161 files, many of them ten lines long; that is the price of being able to follow a URI
with `ls`. Shared code sits at the nearest directory that needs it:

```
src/kalfa/std/
  common/            code that is not a lego and crosses kinds
    runtime.py       Context, trackers, model calling
    figure.py        the palette, the figure settings, the drawing helpers
    samples.py  stream.py  deferred.py
  plot/
    base.py          shared by every plot pack: prediction_pairs, set_frame, columns_of, report_loader
    kalfa/
      pred_vs_true.py  residuals.py  error_map.py  loss_curve.py  ...
    seaborn/
      base.py        the optional import and the sampling helper
      pairplot.py  violin.py  kde.py
    torchmetrics/
      binary_roc.py  binary_precision_recall_curve.py
  pre/
    base.py          Prep, Frame, the field plan, the chain runner
    kalfa/           asinh.py  cast.py  one_hot.py  ...
    sklearn/
      base.py        the scaler wrapper the eight sklearn preprocessors share
      standard_scaler.py  minmax_scaler.py  ...
```

**Registration.** `std/__init__.py` walks `std/<kind>/<pack>/*.py` and imports every module. Four lines, and
the tree becomes the list, so adding a lego is adding a file. The invariant is then machine checked:

* every registered std URI has a file at the matching path,
* every file registers exactly one lego, and its URI matches its path,
* nothing else registers anything.

That test is what makes the promise real rather than a convention people forget.

**Kinds with one pack** (`turn`, `loader`, `builder`, ...) get a directory of one file. It looks deep for a
moment and then it never surprises anyone, which is worth more than saving a level.

## 2. Code style

* Delete all 282 docstrings and any comment inside `src/kalfa`. The code says what it does or it gets
  rewritten until it does.
* Keep `description=` in `@lego`. It is data: `DOCS.md`, `kalfa ls` and `kalfa describe` read it.
* Keep the comments in `templates/kalfa.yaml`. That file becomes a document the user exports and edits.
* No helper that exists only to hold a block. Two smells found today and fixed in stage 2:
  `_draw_all` (ten parameters, one call, exists only to put a loop inside a `try`) and `_feature_batch`
  (returns an imported function as a value so the next helper does not have to import it).
* No function that takes a callable in order to avoid an import. No generality without a second caller.
* Early returns over nesting. A function that needs a paragraph to explain itself is two functions.

## 3. Nothing fixed in the core

Today the driver, the checker and the describer name std URIs directly, so a user who wants a different
adapter, a different builder or a different default split has to edit kalfa. Everything below moves into one
reference document.

`src/kalfa/templates/kalfa.yaml` grows from "the five flow blocks" into the whole contract:

```yaml
blocks:      # as today: data, models, optimizers, training, after
wiring:      # what the driver inserts
  criterion_adapter: /adapter/kalfa/criterion
  metric_adapter:    /adapter/kalfa/metric
  builder:           /builder/kalfa/module
  progress:          /lego/kalfa/progress
  default_split:     /split/kalfa/random
  default_device:    /device/kalfa/cpu
  run_inputs:        [device, record]
checks:      # what the checker compares against
  ratio_splits:  [/split/kalfa/random, /split/kalfa/sequential]
  fold_split:    /split/kalfa/kfold
  given_split:   /split/kalfa/given
  window_feed:   /feed/kalfa/window
  class_weights: /data/kalfa/class_weights
  grid_strategy: /strategy/kalfa/grid
  image_source:  /source/kalfa/image_folder
  best_policy:   /checkpoint/kalfa/best
describe:    # how a trigger is worded in the analysis
  /trigger/kalfa/plateau: "{monitor} plateau {patience}"
  ...
```

Commands:

* `kalfa template --write kalfa.yaml` exports the document that is compiled in, so it can be read and edited.
* `kalfa run cfg.yaml --template my.yaml` runs against a custom one. `check`, `describe` and `resume` take
  the same option.
* The record keeps a copy as `<record>/template.yaml`, so a run stays reproducible when the default changes.

Threading: `api.prepare` takes the path, `recipe.analyze` opens it, the driver and the checker read the
tables from the loaded document instead of module constants.

Second axis, same idea: `kinds.py` holds the vocabulary (`KINDS`, `DEFINITION_KEYS`, `TRAINING_FIXED`,
`SETS`, `HISTORY_PREFIX`). Whether that moves into the document too is an open decision below; it is a bigger
change than the URIs because `check` and the driver branch on it.

## 4. The architecture trio

One lego becomes three, because they answer different questions:

| URI | alias | writes | needs |
|---|---|---|---|
| `/plot/kalfa/architecture` | `architecture` | our own drawing (the new default) | nothing outside kalfa |
| `/plot/kalfa/architecture_text` | `architecture_text` | `architecture.txt`, the module `repr` | nothing |
| `/plot/torchview/architecture` | `torchview` | torchview's drawing per model | torchview, graphviz `dot` |

`reports/` does not exist in kalfa, so the text one stays a plot.

**Our drawing.** torchview labels every box with `type(module).__name__` (`recorder_tensor.py:124`), so it
can never show `z_hat`, `combined_base` or `f1`. kalfa has what it lacks: the declarative graph, with the
names from the config. The drawing reads the graph for the structure and one traced batch for the truth:

* one box per graph node, carrying the node name from the YAML, what it is (a torch layer, a lego, another
  model) and the real input and output shapes, taken from forward hooks on one batch, not from the config,
* arrows labelled with the wire name,
* the boundary wires of the model as their own boxes,
* colour by category: input, layer, lego, model reference, output, loss, optimizer,
* the losses and the optimizers drawn beside the outputs they read, with their names and their targets.

For that the plot needs `losses` on the plot bus, which is one line in the template now that the bus is a
declared group. Layout is a layered walk over the graph (depth from the inputs) and matplotlib boxes and
arrows; no new dependency, works on any device.

## 5. The core modules

`check.py` (1,206 lines) and `describe.py` (766) are the same problem as `pre.py` was. Same treatment, by
responsibility rather than by URI, because they hold no legos:

```
check/__init__.py    the Checker and the problem list
check/sections.py    data, model, training, plots, figures, sweep
check/data.py        the header, the sizes, the columns
check/refs.py        the reference and kind rules
describe/...         one module per printed section
```

This stage is separable and can wait until the std tree is done.

## Order of work

| Stage | What | Risk |
|---|---|---|
| 0 | The registry fixture and the URI to path test | none, only additions |
| 1 | Move std into the tree, one lego per file | mechanical, large diff, the fixture guards it |
| 2 | Delete the docstrings, fix the two smells, tidy what the move exposes | medium, touches everything |
| 3 | The reference document, `template --write`, `run --template` | dumps change once, the driver and the checker are rewired |
| 4 | The architecture trio | new code, no risk to what exists |
| 5 | Split `check` and `describe` | optional |

Each stage is one commit and leaves the suite green.

## Open decisions

1. **The pack level.** The example in the request was `plot/true_vs_pred.py`, without the pack. The plan uses
   `plot/kalfa/pred_vs_true.py`, the full URI. The full mirror is the only version where a URI can be found
   without knowing anything else. Confirm.
2. **Auto discovery or explicit imports** in `std/__init__.py`. The plan walks the tree, so a new file is a
   new lego. The alternative is 161 import lines that can drift from the tree.
3. **The vocabulary in the document** (section 3, second axis) or left in `kinds.py` for now.
4. **The name of the export command**: `kalfa template --write`, or `kalfa reference --write`.
5. **`architecture` changing meaning.** Today it writes text; after stage 4 it draws. The configs and the
   examples that use it move to `architecture_text` where the text was the point. Confirm that the alias
   `architecture` should belong to the drawing.
