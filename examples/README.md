# Examples

Every example is a folder with an `example.json` recipe, a `config.yaml` (and sometimes more YAML files; `14_sweep` has `grid.yaml`, `random.yaml` and `optuna.yaml` instead), a `make_data.py` that writes small synthetic data next to the config (`11_kfold`, `12_resume` and `14_sweep` read the data of `01_mlp_regression` instead) and a plugin module when the example brings its own model. No example needs torchvision, timm or a download: images are 16x16 PNG files drawn with PIL, tables are parquet files and text is a `.txt` file, all made in a few seconds.

| folder | what it shows | plugin |
|---|---|---|
| `minimal` | the smallest config that runs: a table, one model, one loss, one metric, one optimizer | none |
| `01_mlp_regression` | a rule chain (`after_epoch`, `below`, `plateau`), a plain function as a criterion, predictions in the original units, onnx export | `mylegos.py` |
| `02_mlp_classification` | `one_hot` features, a `label` target, `@data.class_weights`, `f1`, a confusion matrix | none |
| `03_timeseries_window` | a `sequential` split, the `window` feed with context, a `stack` model, a forecast over a horizon | none |
| `04_cnn_images` | `image_folder` and transforms, a frozen backbone and the rule that unfreezes it, optimizer groups, AMP | `cnn_legos.py` |
| `05_autoencoder` | a model with two outputs, `latent_sampler`, `image_pairs` | `ae_legos.py` |
| `06_vae` | `vae_elbo`, a `linear_warmup` weight, rules that change a weight and a reference | `vae_legos.py` |
| `07_wgan_gp` | `alternating` with `repeat`, an EMA copy, `fid`, a `write_samples` rule, `gan_sampler` | `gan_legos.py` |
| `08_ddpm` | `ddpm`, `cosine_beta`, `predicts: false`, `ddpm_sampler`, an lr schedule | `unet_legos.py` |
| `09_simclr` | `two_views`, `ntxent`, embeddings as predictions | `simclr_legos.py` |
| `10_char_lm` | `text_lines`, `train.updates`, `perplexity`, `lm_sampler` | `gpt_legos.py` |
| `11_kfold` | the config of `01_mlp_regression` as a layer, `kfold`, the fold groups of a summary | none |
| `12_resume` | `kalfa run --from` continues a record into a new directory | none |
| `13_distillation` | a shared data layer, a teacher loaded from its own record with `weights`, `freeze`, `distill` | `distill_legos.py` |
| `14_sweep` | grid, `random_search` and optuna sweeps, a sweep report | none |
| `15_multi_target` | three targets, `weighted_sum`, a rule that patches a param inside a list | `multihead_legos.py` |
| `16_mdmm` | an `mdmm` constraint, a negative lr group, a rule that tightens epsilon, `train.track` | none |
| `alad` | five models, two custom objectives, a filter per set, a chain of preprocessors, anomaly scores | `alad_legos.py` |

On the board, the model tab draws the diagram of every model the first batch can drive (a model whose inputs the batch does not hold, such as the generator of `07_wgan_gp`, which reads the noise `z`, or the denoiser of `08_ddpm`, shows its module tree instead), the data tab draws the flowchart of a table, an image folder or text lines, and the predictions tab reads each prediction as a task: regression for `01_mlp_regression`, binary for `02_mlp_classification` (scored by `prob_net_yes`) and for `alad` (scored by its anomaly score, with the threshold of the best f1), multiclass with an image gallery for `04_cnn_images`, embeddings for `05_autoencoder`, `06_vae` and `09_simclr`, and three regression pairs under one overview for `15_multi_target`.

## Running one

From the root of the repository:

```sh
uv run python examples/01_mlp_regression/make_data.py
uv run kalfa run examples/01_mlp_regression/config.yaml
```

`make_data.py` writes the data into the example's own folder. The data paths of a config are relative to the config (`file: housing.parquet`, `folder: data/pets`): kalfa looks for a relative path in the working directory first and in the directories of the config files next, so the command works from any directory. The plugin module beside a config is found the same way. Records go to `runs/<name>_<datetime>` under the working directory, and `kalfa run` prints the directory it writes. Two configs write a fixed directory instead: `11_kfold` writes `runs/cv_housing_fold<k>` for fold k and `13_distillation/teacher.yaml` writes `runs/teacher`. A second run of the same fold or of the teacher finds that directory taken and writes into `<record>_<datetime>` beside it, saying so on stderr; `--record DIR` names another directory, and `-p teacher_run=DIR` points the student at a newer teacher.

The budgets of a config are the ones of a full training. A first try is shorter with an override: `--set train.epochs=3` for a config that counts epochs, or `-p epochs=3` when the config reads the budget from a param (the `params` block at its top says which). `10_char_lm` counts updates, and its cosine schedule reads the same params as its budget, so its first try is `-p total_steps=300 -p turn_updates=100 -p warmup=20`. Some full configs need more than a shorter budget, and the `sets` and `params` of the `run` steps in a folder's `example.json` are a short try that the tests run. `07_wgan_gp` is one: its `fid` metric scores every fifth epoch, so in a three epoch run the `val/fid` series that its checkpoint watches never appears and `report: best` stops the run after the last turn. A short try of it takes `--set metrics.fid.every=1` on top of `-p epochs=3`, and its recipe also passes `--set metrics.fid.n=16 --set data.batch.size=8`. `kalfa check examples/<folder>/config.yaml` (or `examples/14_sweep/grid.yaml` and its two siblings) lists the problems of a config without running it. The recipe of a folder shows the commands it is meant for, such as `kalfa predict`, `kalfa generate`, `kalfa export` or `kalfa summarize`.

An example whose recipe lists `needs` builds on another folder: `11_kfold`, `12_resume` and `14_sweep` include the config of `01_mlp_regression` as a layer and read its table, so run that folder's `make_data.py` first. `13_distillation` trains its teacher from `teacher.yaml` first into `runs/teacher`, the record the student config reads by default through its `teacher_run` param; `-p teacher_run=DIR` points the student at another teacher record. `14_sweep` is run with `kalfa sweep` over `grid.yaml`, `random.yaml` or `optuna.yaml`.

## The demo

```sh
uv run python examples/demo.py
uv run kalfa board runs/demo --open
```

`demo.py` runs the `make_data.py` of every folder and plays its recipe with short budgets, each example under `runs/demo/<folder>`, then prints how to open the board. `--only 05_autoencoder 07_wgan_gp` plays some folders, `--out DIR` picks another root and `--device cpu` overrides the device of every config. An example directory the demo made before is removed and made again; an existing directory it did not make stops the demo with a message and is left alone.

## Recipes

`example.json` holds the title of an example, the features it shows, the folders whose data it needs and its steps:

```json
{
  "title": "...",
  "shows": ["after_epoch", "plateau"],
  "needs": [],
  "steps": [
    {"run": "config.yaml", "record": "housing", "params": {"epochs": 3}, "sets": {"data.batch": 32}},
    {"predict": "housing", "data": "new.parquet"},
    {"export": "housing", "format": "onnx"}
  ]
}
```

Each step names one action: `run` (a config with `sets` and `params`), `resume` (a record of an earlier step into a new `record`), `sweep` (a sweep config into a sweep root), `predict` (with an optional `data` file or folder and `which`), `generate` (with `n`), `export` (with `format`) or `summarize` (records or a sweep root, with `out` for the csv and markdown tables, plus `folds.csv` when the records carry a `fold` param, and `html` for a report). In text values `{records}` becomes the record root and `{here}` the example folder, so the student step of `13_distillation` passes `teacher_run: {records}/teacher`. Every run is short: at most three turns, a resumed run four in total, one turn per sweep point. These recipes are what `tests/examples/test_examples.py` plays on a copy of the folder and what `demo.py` plays into the demo root, so every example in this table is run by the tests.
