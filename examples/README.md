# Examples

Every folder is the runnable form of one reference config: `config.yaml` is a byte for byte copy of the one under
`configs/` (a test catches drift; the plugins are copies of `tests/plugins/`), `make_data.py` writes small synthetic
data next to it (`kalfa.synthetic`), the commands run from inside the folder (the paths are relative to it). The
sizes finish in a minute or two on a laptop CPU; where a config asks for a GPU, `--set device=cpu` and a shortened
`epochs` are in the command, the config stays untouched (`cpu` is a device lego like `auto`, `cuda` and `mps`; a
device that is not available is an error, never a silent fallback). The `uv run` prefix uses the project's virtual
environment; drop it inside an activated one.

The general pattern:

```
cd examples/<folder>
uv run python make_data.py
uv run kalfa run config.yaml --set record=runs/x [...]
uv run kalfa predict runs/x [--device cuda] [...]     # or kalfa generate runs/x [--device cuda]
```

`kalfa predict` and `kalfa generate` take `--device` with the same values as the config key, so a record made here
can be replayed on a GPU box; without it they stay on the cpu.

## 01_mlp_regression

Table regression: `standard_scaler`, a two layer MLP, four loss definitions and a loss switch through rules
evaluated at the end of a turn (`rules`), stopping with `plateau`, the `best` checkpoint, `predictions.parquet` in
the original scale.

```
uv run python make_data.py
uv run kalfa run config.yaml --set record=runs/01
uv run kalfa predict runs/01 --data new.parquet
```

## 02_mlp_classification

Binary classification: `one_hot` and `label_encoder`, `cross_entropy` with class weights (the `class_weights` run
time component), `accuracy` and `f1`, `best` with `mode: max`, the prediction decoded back to the label and the
`confusion` plot.

```
uv run python make_data.py
uv run kalfa run config.yaml --set record=runs/02
uv run kalfa predict runs/02 --data new.parquet
```

## 03_timeseries_window

Time series: the `sequential` split (grouped by `site_id`), the `window` feed (window, horizon, context at the split
boundary), `gru` and `last_step`, a multi step target (`pred_y_0..`), the `forecast` plot.

```
uv run python make_data.py
uv run kalfa run config.yaml --set record=runs/03
uv run kalfa predict runs/03
```

## 04_cnn_images

Image classification: the `image_folder` source (`root/<class>/*.png`), a per item chain (`resize`,
`random_crop_flip` on the train set only, `to_tensor`, `normalize`), the `balanced` sampler, a backbone from a
plugin (`timm_legos.py` is a small convolutional network here), the `unfreeze` rule opening the backbone that
starts frozen, group learning rates.

```
uv run python make_data.py
uv run kalfa run config.yaml --set device=cpu -p epochs=3 --set record=runs/04
uv run kalfa predict runs/04 --data data/pets_new
```

## 05_autoencoder

An autoencoder: `flatten` and `unflatten`, a composite model (`{model: name}` nodes), `target: input`, the
`reconstructions` plot (`image_pairs`), inference with the encoder (`--model encoder`).

```
uv run python make_data.py
uv run kalfa run config.yaml --set record=runs/05
uv run kalfa predict runs/05 --model encoder --data data/mnist_new
```

## 06_vae_loss_dynamics

A VAE and loss dynamics: the `vae` objective with a mapping of terms (`train/vae_loss/recon`, `kl`, `w_kl`),
`reparam`, the `linear_warmup` KL schedule, rules targeting objective params (`vae_loss.w_rec`, `vae_loss.recon`),
`plateau`.

```
uv run python make_data.py
uv run kalfa run config.yaml --set record=runs/06
uv run kalfa predict runs/06 --data data/mnist_new
```

## 07_wgan_gp

WGAN GP: two optimizers with `order` and `steps` (n_critic 5), `fresh_batch`, the gradient penalty inside the loss
(`needs_grad`), the EMA copy of the generator, `fid` and `sample_writer` every five turns (`samples/turn_<n>.png`;
the `samples_gif` and `samples_matrix` plots at the end), `best` by `val/fid`, generation with `gan_sampler`. The
`pixel_features` extractor in the command computes the FID from pixel statistics instead of the Inception network
(a fast demonstration); for the real FID `extractor` is not written (torchmetrics downloads the Inception weights).

```
uv run python make_data.py
uv run kalfa run config.yaml --set device=cpu -p epochs=10 --set data.batch.size=16 --set 'metrics.fid.params.extractor={uri: /lego/kalfa/pixel_features}' --set metrics.fid.params.n=64 --set record=runs/07
uv run kalfa generate runs/07
```

On a machine with a GPU the same record generates there: `uv run kalfa generate runs/07 --device cuda`.

## 08_ddpm

DDPM: the `ddpm` objective (the time step and the noise from the turn's `rng`), the `linear_betas` schedule,
`warmup_cosine`, `amp` and `grad_clip`, `sample_writer` every five turns, generation from the EMA copy with
`ddpm_sampler` (`samples/grid.png`).

```
uv run python make_data.py
uv run kalfa run config.yaml --set device=cpu -p epochs=5 -p diffusion_steps=100 --set generate.params.n=16 --set record=runs/08
uv run kalfa generate runs/08
```

## 09_simclr

SimCLR: `two_views` (a preprocessor referencing another preprocessor, train set only), `simclr_aug`, `ntxent`,
training without targets and metrics, inference with the backbone (`--model backbone`, a new folder).

```
uv run python make_data.py
uv run kalfa run config.yaml --set device=cpu -p epochs=3 --set data.batch.size=64 --set record=runs/09
uv run kalfa predict runs/09 --model backbone --data data/stl10/test
```

## 10_char_lm

A character level language model: the `text_lines` source, `char_tokenizer` (the vocabulary goes into the record),
the vocabulary size reaching the model through the `vocab_size` component, the `next_token` feed, the steps mode
(`steps: {total, turn}`), `accumulate`, `amp`, `grad_clip`, `perplexity`, generation with `lm_sampler`
(`samples/samples.txt`). The step count is reduced for a short demonstration; `amp` is for the GPU, on the CPU
bfloat16 mixed precision can be very slow depending on the hardware, the command turns it off.

```
uv run python make_data.py
uv run kalfa run config.yaml --set device=cpu --set training.amp=false -p total_steps=100 -p turn_steps=25 --set record=runs/10
uv run kalfa generate runs/10
```

## 11_kfold_cv

k fold cross validation: config 01 as the lower layer (`include`), the `kfold` split (the held out fold is the
test set), five runs with `-p fold=i`, the fold summary with `kalfa collect` (`cv.json`, `cv.md`).

```
uv run python make_data.py
for i in 0 1 2 3 4; do uv run kalfa run config.yaml -p fold=$i -p epochs=10; done
uv run kalfa collect runs/cv_housing_*
```

## 12_resume

Resuming: config 01 as the lower layer, the run ends (`final/`), `kalfa resume` continues to more epochs in a new
directory (`resume.json` shows the source).

```
uv run python make_data.py
uv run kalfa run config.yaml -p epochs=5 --set record=runs/12
uv run kalfa resume runs/12 --set training.epochs=10 --set record=runs/12_more
```

## 13_distillation

Knowledge distillation: first `teacher.yaml` (a small config specific to this folder) produces the run
`runs/cifar_resnet50_x`; then `config.yaml` loads the teacher with `weights: {run, model, which}`, keeps it frozen
and in eval mode with `trainable: false`, trains the student with the `distill` objective, and the `cool_down` rule
lowers the learning rate.

```
uv run python make_data.py
uv run kalfa run teacher.yaml
uv run kalfa run config.yaml --set device=cpu -p epochs=3 --set record=runs/13
uv run kalfa predict runs/13 --data data/cifar10_new
```

## 14_sweep_grid

A sweep: config 01 as the lower layer, a `width` param and the `sweep` section (lr x width, a 3x2 grid, the
objective is the best turn of `val/rmse`). Every point is an ordinary run under `runs/sweep_housing/<id>/` plus
`sweep.json`; `kalfa collect` on the root writes the table (`sweep.csv`, `sweep.json`, `sweep.md`) and the best
point, and works on a half done sweep too.

```
uv run python make_data.py
uv run kalfa sweep config.yaml -p epochs=5
uv run kalfa collect runs/sweep_housing
```

Queue usage (a SLURM array job, say): one job per point, a shared root; the points are deterministic by id, so the
jobs do not look at each other and `collect` gathers the finished ones.

```
uv run kalfa sweep config.yaml --count                                    # 6
uv run kalfa sweep config.yaml --id $SLURM_ARRAY_TASK_ID --record /shared/sweep_housing
uv run kalfa collect /shared/sweep_housing
```

A fed back strategy runs in the local loop only: `--set 'sweep.strategy={uri: optuna, params: {trials: 8, seed: 1}}'
--set 'sweep.space.lr={low: 1.0e-4, high: 1.0e-2, log: true}'` (`uv sync --extra sweep`).

## 15_multi_target

Multi target regression: three analysis scores come out of one three wide output wire, their combination out of a
second and a tail label out of a third. `training.targets` binds every wire to the target fields it predicts, so
one loss call compares the whole `(batch, 3)` block, every column is reported and inverted with its own scaler,
and `pred_vs_true` draws one titled panel per field. The losses and the metrics name only their wire and inherit
the target from that table.

```
uv run python make_data.py
uv run kalfa describe config.yaml --load
uv run kalfa run config.yaml -p epochs=10 --set record=runs/15
```

## alad

Adversarial anomaly detection with a plugin: five trained models and a composite score model, two optimizers over
two objectives from `myexample.py` (the plugin next to the config), filters before and after the split, `one_hot`
free feature chains (`abs`, `log`, `standard_scaler`), no checkpoint (`checkpoint: null`, `report: last`) and four
plots. The config is the target surface written by hand; the dumps `configs/dumps/tidy.*` are generated from it.

```
uv run python make_data.py
uv run kalfa run config.yaml -p epochs=10 --set record=runs/alad
uv run kalfa predict runs/alad
```

## minimal

The smallest table regression: one model named `net` with an inline optimizer (no `optimizers` section), four loss
definitions with the rule chain, `plateau` stopping and the `best` checkpoint. The twin of `01_mlp_regression` in
the user's own writing.

```
uv run python make_data.py
uv run kalfa run config.yaml -p epochs=10 --set record=runs/minimal
uv run kalfa predict runs/minimal --data new.parquet
```
