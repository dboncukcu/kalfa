# kalfa

A YAML front end for PyTorch training. You write the data, the model, the losses, the metrics, the rules and the
report in one config file; kalfa turns the config into a cirak recipe, cirak compiles the recipe into a tezgah
graph, tezgah runs it. The full definition of the surface is `CONFIG.md`, the class of every key is
`configs/reference.yaml`, the lego reference is `DOCS.md`, the fourteen reference configs are under `configs/`.

## Installation

Python 3.13 and above. `../tezgah` and `../cirak` are path dependencies (editable installs); the three repositories
sit side by side.

```
uv sync
uv run kalfa --help
uv run pytest          # about 260 tests, half a minute
```

Dependencies: torch, pandas, pyarrow, scikit-learn, torchmetrics, matplotlib, pillow, scipy, ruamel.yaml, tqdm;
`uv sync --extra sweep` adds optuna for the fed back sweep strategy.

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
kalfa check cfg.yaml [--set ...] [-p ...] [--layers] [--dump] [--recipe] [--load]
kalfa predict runs/x [--model name] [--which best|last] [--data new.parquet] [--device cuda]
kalfa generate runs/x [--which best|last] [--device cuda]         # writes samples/ with the sampler of the generate section
kalfa resume runs/x [--set training.epochs=N]                     # continues from last.pt or final/ into a new directory
kalfa sweep cfg.yaml [--record root] [--count | --show N | --id N]  # the sweep section: the local loop or one point
kalfa collect runs/cv_* | kalfa collect <sweep root>              # fold summaries (cv.json, cv.md) or the sweep table and the best point
kalfa ls [/alias/kalfa/tabular | /criterion | ... | word]         # packs and legos with their kinds and facts; a word searches
kalfa docs [--write DOCS.md]                                      # the lego reference generated from the registry
```

`--set path=value`: the path is dotted from the root of the document (`--set training.epochs=5`,
`--set training.rules=[]`, `--set device=cuda`); a single segment can only be a top level key, any other bare name
is an error with a hint. `-p name=value` (`--param`) is the shortcut for `params.name` (`-p lr=1e-4`). The value
is read as YAML; lists are replaced wholesale, a rule list is rewritten with `--set training.rules=[...]`.

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
| `predictions.parquet` | the test set: `row`, the targets (inverted), `pred_<output>` (inverted), `raw_<output>` |
| `plots/` | the outputs of the plot legos, named after the definition (`plots.roc` → `roc.png`; `architecture` writes text) |
| `samples/` | the output of `generate`: `samples.pt` (for images `grid.png` too), `samples.txt` for text; `turn_<n>.*` from `sample_writer` |
| `plugins/` | copies of the plugin modules the run imported, so predict, generate and resume work from the record |
| `device.json` | the chosen device and the device lego that picked it |
| `resume.json` | in a resumed run the source run and the checkpoint |

`kalfa predict` rewrites `predictions.parquet` on the run's own test set; with `--data`
`predictions_<file name>[_<model>].parquet`.

## Alias packs and plugins

`include: [/alias/kalfa/tabular]` (or `/alias/kalfa/vision` and `/alias/kalfa/text`, which contain tabular;
`/alias/kalfa/lazy` for large tables) brings the short names (`parquet`, `standard_scaler`, `linear_relu`, `mse`,
`rmse`, `adam`, `supervised`, `plateau`, `best`, `loss_curve` ...). The table is flat, a full URI is valid
everywhere; `kalfa ls /alias/kalfa/tabular` prints the contents, `kalfa ls tokenizer` searches names and
descriptions. Your own lego is registered with `@kalfa.lego` in a Python module next to the config and comes in
with `plugins: [module]`; `examples/alad/myexample.py` (the ALAD objectives of that example) and `tests/plugins/`
are examples. Details in the Development section.

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
accepts). A `/data/` lego is a run time component: `{uri: name}` as a param value, built once the data exists with
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
`partial_fit`; a device lego (`/device/acme/tpu`) takes no inputs, returns a `torch.device` and raises when the device is
not available (`device: {uri: /device/acme/tpu}` then selects it).

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
(`uv run kalfa docs --write DOCS.md` after touching a lego). cirak copies the recipes and dumps as its own fixtures.

**Test rules.** `uv run pytest` must stay green; every lego has a unit test under `tests/test_std_*.py`; every
reference config has an end to end run test on synthetic data, CPU only, no network; test plugins live under
`tests/plugins/`; the reinstalled environment (`uv sync --reinstall`) runs the suite before a release.

**Versions and releases.** The three packages are released in dependency order: tezgah first, then cirak (which
depends on tezgah), then kalfa (which pins `tezgah>=` and `cirak>=` in `pyproject.toml`); bump the version, run the
suite of each repository against the reinstalled environment, then build (`uv build`). kalfa is 0.2.0 with tezgah
and cirak 0.2.0.
