# kalfa lego reference

Generated from the registry by `kalfa docs --write DOCS.md`; do not edit by hand, the test `tests/test_docs.py`
compares this file with the registry. One table per kind: the URI, the alias names, the signature with the param
defaults, the facts a lego declares and its description. The kind of a lego is the first segment of its URI; which
config section it may be written in follows from the kind (`CONFIG.md` section 5). Every URI of the catalog is
valid in a config; the alias packs at the end give the short names. The skeleton steps come after the catalog.

## Catalog

The legos a config writes, by kind.

### Kinds

| Kind | Where it is written | Count |
|---|---|---|
| `source` | data.source | 6 |
| `split` | data.split | 4 |
| `pre` | data.preprocessors | 25 |
| `feed` | data.feed | 3 |
| `layer` | model nodes | 18 |
| `init` | model init | 4 |
| `criterion` | losses, metrics | 6 |
| `objective` | losses | 7 |
| `metric` | metrics | 9 |
| `adapter` | the driver | 2 |
| `optimizer` | optimizers | 3 |
| `schedule` | optimizer schedule | 4 |
| `turn` | training.turn | 1 |
| `trigger` | training.stop, rules when | 5 |
| `checkpoint` | training.checkpoint | 3 |
| `generate` | generate | 3 |
| `plot` | plots | 12 |
| `strategy` | sweep.strategy | 4 |
| `device` | device, predict --device, generate --device | 4 |
| `lego` | a param value, or the driver | 2 |
| `data` | a param value ({uri: name}) | 2 |

### source

| URI | Alias | Signature | Facts | Description |
|---|---|---|---|---|
| `/source/kalfa/csv` | `csv` | `(path)` | returns: df | Read a CSV file into a DataFrame |
| `/source/kalfa/csv_stream` |  | `(path, chunk=65536)` | returns: df | Read a CSV file in chunks (the lazy set) |
| `/source/kalfa/image_folder` | `image_folder` | `(path)` | returns: df | Images under root/<class>/ as a Dataset with fields image and label |
| `/source/kalfa/parquet` | `parquet` | `(path)` | returns: df | Read a parquet file into a DataFrame |
| `/source/kalfa/parquet_stream` |  | `(path, chunk=65536)` | returns: df | Read a parquet file in chunks (the lazy set): a stream the data legos filter, cut and fit without loading the table |
| `/source/kalfa/text_lines` | `text_lines` | `(path)` | returns: df | The lines of a text file as a Dataset with the field text |

### split

| URI | Alias | Signature | Facts | Description |
|---|---|---|---|---|
| `/split/kalfa/given` | `given` | `(df, valid=None, test=None)` | returns: train, valid, test | The source is the train set; valid and test come from the given paths, read like the source (a missing path means no set) |
| `/split/kalfa/kfold` | `kfold` | `(df, k, fold, val=None, seed=None)` | returns: train, valid, test | k folds of a seeded permutation: the held out fold is the test set, val carves the valid set from the rest; without val there is no valid set |
| `/split/kalfa/random` | `random_split` | `(df, ratios, seed=None)` | returns: train, valid, test | Shuffle the rows with a seed and cut them by ratios into train, valid and test; the short form of a split without a uri |
| `/split/kalfa/sequential` | `sequential` | `(df, ratios, group=None)` | returns: train, valid, test; refs: group=column | Cut the rows in their order by ratios; with a group column every group is cut on its own |

### pre

| URI | Alias | Signature | Facts | Description |
|---|---|---|---|---|
| `/pre/kalfa/abs` | `abs` | `()` |  | Absolute value of a column |
| `/pre/kalfa/asinh` | `asinh` | `(scale=1.0, overflow=700.0)` |  | Signed log scale of a heavy tailed column: arcsinh(x / scale), inverted by scale sinh(y); keeps the sign, linear near zero, logarithmic in the tails, defined at zero; the inverse refuses values past overflow, where sinh leaves float64 |
| `/pre/kalfa/atanh` | `atanh` | `(scale=1.0)` |  | artanh(x / scale) of a bounded column, inverted by scale tanh(y); a value outside (-scale, scale) is an error that names how many and how large |
| `/pre/kalfa/cast` | `cast` | `(dtype)` |  | Cast a column to a numpy dtype |
| `/pre/kalfa/char_tokenizer` | `char_tokenizer` | `()` | state: True | Character level tokenizer fitted on the train text; the vocabulary goes into the record |
| `/pre/kalfa/label_encoder` | `label_encoder` | `()` | state: True | Integer codes of a label column, sorted by label; inverted in reports and predictions, class scores decode to labels |
| `/pre/kalfa/log` | `log` | `(base=10.0, norm=1.0)` |  | log1p of a column divided by norm, in the given base |
| `/pre/kalfa/normalize` | `normalize` | `(mean, std)` |  | Normalize an image tensor per channel; mean and std are numbers, lists or the presets imagenet and cifar10 |
| `/pre/kalfa/one_hot` | `one_hot` | `()` | state: True | One hot columns <field>_<category> of a categorical column; unknown categories give zeros |
| `/pre/kalfa/random_crop_flip` | `random_crop_flip` | `(size)` |  | Random crop of size after padding and a random horizontal flip |
| `/pre/kalfa/resize` | `resize` | `(size)` |  | Resize an image to size (int or [h, w]) |
| `/pre/kalfa/simclr_aug` | `simclr_aug` | `(size, scale=(0.5, 1.0))` |  | SimCLR augmentation: random resized crop to size, horizontal flip, brightness jitter |
| `/pre/kalfa/sinh` | `sinh` | `(scale=1.0, overflow=700.0)` |  | sinh(x / scale), the direction opposite to asinh: it stretches the tails instead of compressing them; a value past overflow is an error, where sinh leaves float64 |
| `/pre/kalfa/tanh` | `tanh` | `(scale=1.0, eps=1e-15)` |  | tanh(x / scale) into (-1, 1); the inverse clips at 1 - eps, so a value that saturated in float64 (past about 19 scale) comes back at the clip instead of infinity |
| `/pre/kalfa/to_tensor` | `to_tensor` | `()` |  | Image to a float tensor in [0, 1], channels first |
| `/pre/kalfa/to_tensor_signed` | `to_tensor_signed` | `()` |  | Image to a float tensor in [-1, 1], channels first |
| `/pre/kalfa/two_views` | `two_views` | `(transform)` | refs: transform=preprocessor | Two independent applications of a transform to one image, as a pair |
| `/pre/sklearn/kbins_discretizer` | `kbins_discretizer` | `(bins=5, strategy='quantile', encode='onehot')` | state: True | Cut a column into bins and write them as one hot columns <field>_bin<n> (encode: ordinal for one integer column); strategy quantile, uniform or kmeans (sklearn KBinsDiscretizer) |
| `/pre/sklearn/max_abs_scaler` | `max_abs_scaler` | `()` | state: True; grouped: True | Scale a column by its largest absolute value, into [-1, 1] with the sign and the zeros kept (sklearn MaxAbsScaler); one object over every column that names it |
| `/pre/sklearn/minmax_scaler` | `minmax_scaler` | `(low=0.0, high=1.0)` | state: True; grouped: True | Scale a column into [low, high] (sklearn MinMaxScaler); one object over every column that names it, its statistics per column |
| `/pre/sklearn/power_transformer` | `power_transformer` | `(method='yeo-johnson', standardize=True)` | state: True | Yeo-Johnson (or Box-Cox for positive columns) with the exponent fitted per column, then standardized (sklearn PowerTransformer); the invertible way to a near normal column |
| `/pre/sklearn/quantile_transformer` | `quantile_transformer` | `(quantiles=1000, output='uniform', seed=None)` | state: True | Map a column onto its own quantiles, uniform or normal (sklearn QuantileTransformer); flattens any shape, the inverse interpolates between the stored quantiles |
| `/pre/sklearn/robust_scaler` | `robust_scaler` | `(low=25.0, high=75.0)` | state: True; grouped: True | Center a column on its median and scale it by the distance between the low and high percentiles (sklearn RobustScaler); outliers do not move the statistics |
| `/pre/sklearn/spline_transformer` | `spline_transformer` | `(knots=5, degree=3, extrapolation='constant')` | state: True | A B-spline basis of a column, <field>_spline<n>: a smooth non linear expansion of one feature that a linear head can use (sklearn SplineTransformer) |
| `/pre/sklearn/standard_scaler` | `standard_scaler` | `()` | state: True; grouped: True | Standardize a column to zero mean and unit variance (sklearn StandardScaler); one object over every column that names it, its statistics per column |

### feed

| URI | Alias | Signature | Facts | Description |
|---|---|---|---|---|
| `/feed/kalfa/next_token` | `next_token` | `(frame, frames, seq_len)` |  | input_ids and targets windows of seq_len tokens over the set's token stream |
| `/feed/kalfa/table` | `table` | `(frame, frames=None)` |  | Feature columns as one tensor x and target fields by name; Dataset fields by name |
| `/feed/kalfa/window` | `window` | `(frame, frames, size, horizon, context=False, group=None)` | refs: group=column | Windows of size steps and the next horizon steps of the targets; context takes the tail of the previous set at the split boundary, group keeps series apart |

### layer

| URI | Alias | Signature | Facts | Description |
|---|---|---|---|---|
| `/layer/kalfa/l1_distance` | `l1_distance` | `()` |  | Mean absolute difference of two wires per sample |
| `/layer/kalfa/l2_normalize` | `l2_normalize` | `(eps=1e-12)` |  | Divide every sample by the L2 norm of its own feature vector (sklearn's Normalizer as a layer: it reads the whole vector, so it belongs to the model, not to a column chain) |
| `/layer/kalfa/linear` | `linear` | `(out_features, in_features=None)` |  | Linear layer; without in_features the input width is taken from the first batch |
| `/layer/kalfa/linear_relu` | `linear_relu` | `(out_features, in_features=None)` |  | Linear layer followed by ReLU; lazy without in_features |
| `/layer/kalfa/polynomial` | `polynomial` | `(degree=2, interaction_only=False, bias=False, keep=True)` |  | Polynomial expansion of the feature vector: the features and every product of degree of them (interaction_only drops the squares, bias adds a constant column, keep: false returns the products alone); the place for feature interactions, computed per batch |
| `/layer/kalfa/reparam` | `reparam` | `()` |  | Sample z from mu and logvar in train mode, return mu in eval mode |
| `/layer/kalfa/unflatten` | `unflatten` | `(shape)` |  | Reshape the features of every sample to shape |
| `/layer/torch/concat` | `concat` | `(dim=1)` |  | Concatenate wires along a dimension |
| `/layer/torch/conv2d` | `conv2d` | `(out_channels, kernel, stride=1, padding=0, in_channels=None)` |  | 2d convolution; without in_channels the input channels are taken from the first batch |
| `/layer/torch/dropout` | `dropout` | `(p=0.5)` |  | Dropout |
| `/layer/torch/embedding` | `embedding` | `(num, dim)` |  | torch.nn.Embedding(num, dim); num may be a kind data component such as vocab_size |
| `/layer/torch/flatten` | `flatten` | `()` |  | Flatten every dimension but the batch |
| `/layer/torch/gru` | `gru` | `(hidden, layers=1)` |  | GRU over (batch, steps, features) returning every step; the input width comes from the first batch |
| `/layer/torch/last_step` | `last_step` | `()` |  | The last step of a sequence |
| `/layer/torch/leaky_relu` | `leaky_relu` | `(negative_slope=0.01)` |  | LeakyReLU activation |
| `/layer/torch/linear` |  | `(in_features, out_features)` |  | torch.nn.Linear |
| `/layer/torch/maxpool` | `maxpool` | `(kernel, stride=None)` |  | 2d max pooling |
| `/layer/torch/relu` | `relu` | `()` |  | ReLU activation |

### init

| URI | Alias | Signature | Facts | Description |
|---|---|---|---|---|
| `/init/torch/kaiming` | `kaiming` | `(nonlinearity='relu')` |  | Kaiming normal initialization |
| `/init/torch/normal` | `normal` | `(std, mean=0.0)` |  | Normal initialization with std and mean |
| `/init/torch/xavier` | `xavier` | `(gain=1.0)` |  | Xavier uniform initialization |
| `/init/torch/zeros` | `zeros` | `()` |  | Zero initialization |

### criterion

| URI | Alias | Signature | Facts | Description |
|---|---|---|---|---|
| `/criterion/kalfa/bce_logits` | `bce_logits` | `(predictions, targets, pos_weight=None)` | partial: True | Binary cross entropy on logits |
| `/criterion/kalfa/cross_entropy` | `cross_entropy` | `(predictions, targets, weight=None, label_smoothing=0.0)` | partial: True | Cross entropy over class logits; weight may be a run time component |
| `/criterion/kalfa/huber` | `huber` | `(predictions, targets, delta=1.0)` | partial: True | Huber loss with threshold delta |
| `/criterion/kalfa/log_cosh` | `log_cosh` | `(predictions, targets)` | partial: True | log(cosh(error)) loss |
| `/criterion/kalfa/mae` | `mae` | `(predictions, targets)` | partial: True | Mean absolute error |
| `/criterion/kalfa/mse` | `mse` | `(predictions, targets)` | partial: True | Mean squared error |

### objective

| URI | Alias | Signature | Facts | Description |
|---|---|---|---|---|
| `/objective/kalfa/ddpm` | `ddpm` | `(models, batch, model, schedule, rng=None)` | partial: True; refs: model=model, schedule=schedule | DDPM noise prediction loss: a random time step and noise per sample (rng), the model predicts the noise of the noised input |
| `/objective/kalfa/distill` | `distill` | `(models, batch, student, teacher, temperature=1.0, alpha=0.5, target=None)` | partial: True; refs: student=model, teacher=model | Knowledge distillation: alpha * KL(teacher \|\| student) at temperature T (times T squared) plus (1 - alpha) * cross entropy of the student; returns loss, ce and kl |
| `/objective/kalfa/ntxent` | `ntxent` | `(models, batch, model, temperature=0.5)` | partial: True; refs: model=model | NT-Xent contrastive loss over the two views of every image in the batch |
| `/objective/kalfa/vae` | `vae` | `(models, batch, encoder, decoder, recon, w_rec=1.0, kl_schedule=None, step=None, rng=None)` | partial: True; refs: encoder=model, decoder=model, recon=criterion, kl_schedule=schedule | VAE loss: w_rec * recon(decoder(z), x) + kl_schedule(step) * KL, z sampled from the encoder's mu and logvar; returns loss, recon, kl and w_kl |
| `/objective/kalfa/weighted_sum` | `weighted_sum` | `(models, batch, terms, losses)` | partial: True; refs: terms=loss | The weighted sum of other losses definitions on the same batch: terms maps a losses name to its weight; returns loss and every term |
| `/objective/kalfa/wgan_g` | `wgan_g` | `(models, batch, generator, critic, latent, conditional=False, rng=None)` | partial: True; refs: generator=model, critic=model | WGAN generator loss: minus the critic's mean score of generated samples |
| `/objective/kalfa/wgan_gp_d` | `wgan_gp_d` | `(models, batch, generator, critic, latent, gp_weight=10.0, conditional=False, rng=None, scaler=None)` | partial: True; refs: generator=model, critic=model; needs_grad: True | WGAN critic loss with a gradient penalty on interpolates; the label conditions both models |

### metric

| URI | Alias | Signature | Facts | Description |
|---|---|---|---|---|
| `/metric/kalfa/fid` | `fid` | `(model, latent, conditional=False, n=1000, extractor=None)` | state: True; refs: model=model; uses: models, batch | Fréchet inception distance of n samples of the model against n real images of the set; conditional samples use the batch labels |
| `/metric/kalfa/perplexity` | `perplexity` | `()` | state: True | exp of the mean token cross entropy of the logits against the targets |
| `/metric/kalfa/recon_error` | `recon_error` | `()` | state: True | Mean per sample squared reconstruction error of the output against the target |
| `/metric/kalfa/rmse` | `rmse` | `()` | state: True | Root mean squared error |
| `/metric/kalfa/sample_writer` | `sample_writer` | `(n=16, sampler=None)` | state: True; refs: sampler=generate; uses: models | A metric that writes n samples per pass under samples/turn_<n> (png and pt, or txt) from the sampler (sampler: generate takes the generate section) or the predicts model; it reports no value, use every and sets to pace it |
| `/metric/torchmetrics/accuracy` | `accuracy` | `()` | state: True | Accuracy of class logits (argmax) against integer labels |
| `/metric/torchmetrics/binary_auroc` | `auroc` | `()` | state: True | Area under the ROC curve of binary scores (torchmetrics) |
| `/metric/torchmetrics/binary_average_precision` | `average_precision` | `()` | state: True | Average precision of binary scores (torchmetrics) |
| `/metric/torchmetrics/f1` | `f1` | `(average='macro')` | state: True | Macro F1 of class logits (argmax) against integer labels |

### adapter

| URI | Alias | Signature | Facts | Description |
|---|---|---|---|---|
| `/adapter/kalfa/criterion` |  | `(criterion)` | uses: predicts | Feed a criterion the predicts model's output wire and the target field named by the definition's keys |
| `/adapter/kalfa/metric` |  | `(metric)` | uses: predicts | Feed a metric the predicts model's output wire and the target field named by the definition's keys |

### optimizer

| URI | Alias | Signature | Facts | Description |
|---|---|---|---|---|
| `/optimizer/torch/adam` | `adam` | `(models, params=None, schedule=None, loss=None)` | aliases: models; state: True; refs: loss=loss, schedule=schedule | torch Adam over the union of its models; params are Adam's keyword arguments |
| `/optimizer/torch/adamw` | `adamw` | `(models, params=None, schedule=None, loss=None)` | aliases: models; state: True; refs: loss=loss, schedule=schedule | torch AdamW over the union of its models |
| `/optimizer/torch/sgd` | `sgd` | `(models, params=None, schedule=None, loss=None)` | aliases: models; state: True; refs: loss=loss, schedule=schedule | torch SGD over the union of its models |

### schedule

| URI | Alias | Signature | Facts | Description |
|---|---|---|---|---|
| `/schedule/kalfa/linear_betas` | `linear_betas` | `(step, steps, start=0.0001, end=0.02)` | partial: True | The diffusion noise schedule: beta rises linearly from start to end over steps |
| `/schedule/kalfa/linear_warmup` | `linear_warmup` | `(step, start, end, steps)` | partial: True | Linear ramp from start to end over steps updates, then end |
| `/schedule/kalfa/step_decay` | `step_decay` | `(step, step_size, gamma)` | partial: True | Multiply by gamma every step_size updates; the value is the factor |
| `/schedule/kalfa/warmup_cosine` | `warmup_cosine` | `(step, warmup, total)` | partial: True | Linear warmup to one over warmup updates, then a cosine decay to zero at total |

### turn

| URI | Alias | Signature | Facts | Description |
|---|---|---|---|---|
| `/turn/kalfa/alternating` | `alternating`, `supervised` | `(models, optimizers, emas, counters, composites, effects, loader, params, extra, losses, metrics, losses_keys, metrics_keys, predicts, steps, device=None, prep=None, record=None)` | returns: models, optimizers, emas, counters, metrics; bus: device=device, prep=prep, record=record; mutates: models, optimizers, emas, counters; extras: amp, grad_clip, accumulate | One turn: every step each optimizer in order minimizes its loss for its steps; a turn is an epoch, or K steps with a stream that lives across turns; losses and metrics are the running means of the pass |

### trigger

| URI | Alias | Signature | Facts | Description |
|---|---|---|---|---|
| `/trigger/kalfa/after_turn` | `after_turn`, `after_epoch` | `(metrics, turn_index, state, at)` | partial: True | Fires once the given number of turns has ended, counted across resumes |
| `/trigger/kalfa/metric_above` | `metric_above` | `(metrics, turn_index, state, monitor, value)` | partial: True | Fires when the monitored value rises above value; a missing value is not seen |
| `/trigger/kalfa/metric_below` | `metric_below` | `(metrics, turn_index, state, monitor, value)` | partial: True | Fires when the monitored value drops below value; a missing value is not seen |
| `/trigger/kalfa/plateau` | `plateau` | `(metrics, turn_index, state, monitor, patience, mode='min', min_delta=0.0)` | partial: True | Fires after patience turns without improvement of the monitored value; turns without the value are not counted |
| `/trigger/kalfa/time_budget` | `time_budget` | `(metrics, turn_index, state, minutes)` | partial: True | Fires once the given number of minutes has passed since the first turn it saw |

### checkpoint

| URI | Alias | Signature | Facts | Description |
|---|---|---|---|---|
| `/checkpoint/kalfa/best` | `best` | `(monitor, mode='min')` |  | Write best.pt when the monitored value improves and last.pt every turn |
| `/checkpoint/kalfa/last` | `last` | `()` |  | Write last.pt every turn |
| `/checkpoint/kalfa/snapshot` | `snapshot` | `(every)` |  | Write snapshot_<n>.pt every n turns and last.pt every turn |

### generate

| URI | Alias | Signature | Facts | Description |
|---|---|---|---|---|
| `/generate/kalfa/ddpm_sampler` | `ddpm_sampler` | `(models, prep, rng, model, schedule, shape, n=64)` | partial: True; refs: model=model, schedule=schedule | n samples by the reverse diffusion of the noise schedule from pure noise |
| `/generate/kalfa/gan_sampler` | `gan_sampler` | `(models, prep, rng, model, latent, n=64, conditional=False, n_classes=None)` | partial: True; refs: model=model | n samples of a generator from latent noise; conditional samples cycle through n_classes |
| `/generate/kalfa/lm_sampler` | `lm_sampler` | `(models, prep, rng, model, prompt, max_new_tokens=100, temperature=1.0, context=None)` | partial: True; refs: model=model | Autoregressive text from a prompt with the record's tokenizer; temperature scales the logits, the window is context or the model's seq_len; the model's last layer has one logit per vocabulary entry (vocab_size) |

### plot

| URI | Alias | Signature | Facts | Description |
|---|---|---|---|---|
| `/plot/kalfa/architecture` | `architecture` | `(predictions, history, models, record, name=None)` | partial: True | The report models printed as text under plots/architecture.txt |
| `/plot/kalfa/class_histogram` | `class_histogram` | `(predictions, history, models, record, bins=40, name=None)` | partial: True | Histogram of the raw scores of the test set, one series per target class |
| `/plot/kalfa/confusion_matrix` | `confusion_matrix` | `(predictions, history, models, record, name=None)` | partial: True | Confusion matrix of the decoded test predictions against the target labels |
| `/plot/kalfa/forecast_samples` | `forecast_samples` | `(predictions, history, models, record, n=6, name=None)` | partial: True | n sample windows of the test set: the true horizon against the predicted one |
| `/plot/kalfa/image_grid` | `image_grid` | `(predictions, history, models, record, loaders=None, predicts=None, n=16, set=None, name=None)` | partial: True | n outputs of the predicts model on the report set as an image grid |
| `/plot/kalfa/image_pairs` | `image_pairs` | `(predictions, history, models, record, loaders=None, predicts=None, n=8, set=None, name=None)` | partial: True | n inputs of the report set next to the predicts model's outputs (reconstructions) |
| `/plot/kalfa/loss_curve` | `loss_curve` | `(predictions, history, models, record, series=None, name=None)` | partial: True | Every history series over the turns, or the named ones |
| `/plot/kalfa/pred_vs_true` | `pred_vs_true` | `(predictions, history, models, record, name=None, columns=4)` | partial: True | Predicted against true values of the test set, one panel per predicted field, laid out in a grid of columns panels per row and titled with the field name |
| `/plot/kalfa/samples_gif` | `samples_gif` | `(predictions, history, models, record, name=None, duration=400)` | partial: True | The per turn sample grids of samples/turn_*.png as an animation; skipped with a warning when there are none |
| `/plot/kalfa/samples_matrix` | `samples_matrix` | `(predictions, history, models, record, name=None, n=8)` | partial: True | A matrix of the per turn samples of samples/turn_*.pt: one row per turn, n columns; skipped with a warning when there are none |
| `/plot/torchmetrics/binary_precision_recall_curve` |  | `(predictions, history, models, record, name=None)` | partial: True | Precision recall curve of the raw test scores against the binary target |
| `/plot/torchmetrics/binary_roc` |  | `(predictions, history, models, record, name=None)` | partial: True | ROC curve of the raw test scores against the binary target |

### strategy

| URI | Alias | Signature | Facts | Description |
|---|---|---|---|---|
| `/strategy/kalfa/grid` | `grid` | `()` |  | Every combination of the space's choices (a range needs steps); deterministic by id |
| `/strategy/kalfa/optuna` | `optuna` | `(trials, seed=0, sampler='tpe')` |  | trials points proposed by optuna (tpe or random sampler) from the objectives fed back; local loop only, no --id |
| `/strategy/kalfa/random` | `random` | `(count, seed=0)` |  | count points drawn uniformly from the space with a seed; deterministic by id |
| `/strategy/kalfa/sobol` | `sobol` | `(count, seed=0, scramble=True)` |  | count points of a scrambled Sobol sequence with a seed; deterministic by id |

### device

| URI | Alias | Signature | Facts | Description |
|---|---|---|---|---|
| `/device/kalfa/auto` | `auto` | `()` |  | The first available device of cuda, mps, cpu |
| `/device/kalfa/cpu` | `cpu` | `()` |  | The CPU |
| `/device/kalfa/cuda` | `cuda` | `(index=0)` |  | The cuda device with the given index; an error when cuda or that index is not available |
| `/device/kalfa/mps` | `mps` | `()` |  | The Apple mps device; an error when it is not available |

### lego

| URI | Alias | Signature | Facts | Description |
|---|---|---|---|---|
| `/lego/kalfa/pixel_features` |  | `(size=4)` |  | A cheap FID feature extractor for demos and tests: images pooled to size by size and flattened; pass it as fid's extractor param |
| `/lego/kalfa/progress` |  | `()` |  | The progress display of a run |

### data

| URI | Alias | Signature | Facts | Description |
|---|---|---|---|---|
| `/data/kalfa/class_weights` | `class_weights` | `(loader, target=None)` |  | Inverse frequency class weights of the train set's target field, mean one; built once the train loader exists |
| `/data/kalfa/vocab_size` | `vocab_size` | `(prep)` |  | The vocabulary size of the fitted tokenizer among the preprocessors; built once prep exists |

## Skeleton steps

These are the skeleton steps `src/kalfa/templates/kalfa.yaml` calls; they are not written in a config, the
template places them and the driver fills their params from the config sections. The list is derived from the URIs
the template mentions, so it cannot drift. Two more legos are inserted by the driver rather than by the template
and stay in the catalog above: the adapters (`/adapter/kalfa/criterion` and `/adapter/kalfa/metric`, which wrap the
criteria and metrics of a config) and the progress component (`/lego/kalfa/progress`).

| URI | Alias | Signature | Facts | Description |
|---|---|---|---|---|
| `/builder/kalfa/module` |  | `(graph, seed=None, index=0, init=None, trainable=True, weights=None, models=None, prep=None, train_loader=None)` | bus: prep=prep, train_loader=train_loader | Build a model graph into an nn.Module under hash(seed, index), apply init roles, trainable and weights; reference nodes take the models dict; layer params that are kind data components are built from prep and the train loader |
| `/lego/kalfa/apply` |  | `(df, prep, set, keys=None)` |  | Apply the fitted chains to one set and type its columns; keys carry the sets a preprocessor is limited to |
| `/lego/kalfa/checkpoint` |  | `(state, policy, metrics=None, record=None)` | returns: None; bus: metrics=metrics, record=record | Write the checkpoint files the policy asks for; nothing without a policy |
| `/lego/kalfa/clone` |  | `(model, decay)` | state: True | An exponential moving average copy of a model with the given decay |
| `/lego/kalfa/const` |  | `(value)` |  | A fresh copy of a constant value |
| `/lego/kalfa/evaluate` |  | `(models, emas, composites, counters, effects, loader, set, losses, metrics, losses_keys, metrics_keys, predicts, device=None, prep=None, record=None)` | returns: metrics; bus: device=device, prep=prep, record=record | Losses (model scale) and metrics (original scale, through prep) of one set under no_grad; an empty set gives an empty mapping; record reaches metrics that write files |
| `/lego/kalfa/filter` |  | `(df, query)` |  | Keep the rows a pandas query selects; a Dataset source takes field equality queries |
| `/lego/kalfa/filter_set` |  | `(df, set, filters)` |  | Apply the {query, sets} filters that name this set; the frame passes untouched otherwise |
| `/lego/kalfa/fit` |  | `(df, fields, preprocessors, drop, keys=None, record=None)` | returns: prep; bus: record=record; state: True | Resolve the field globs and fit every preprocessor chain on the train set; keys carry the sets a preprocessor is limited to |
| `/lego/kalfa/generate` |  | `(models, composites, prep, generate, record=None)` | returns: None; bus: record=record | Run the generate lego with the report models; nothing without a generate section |
| `/lego/kalfa/history` |  | `(progress, metrics=None, turn_index=None, counters_next=None, optimizers_next=None, rules_next=None, record=None)` | returns: None; bus: metrics=metrics, turn_index=turn_index, counters_next=counters_next, optimizers_next=optimizers_next, rules_next=rules_next, record=record | Append the turn's line to history.jsonl and advance the progress display |
| `/lego/kalfa/identity` |  | `(value)` | aliases: value | The value itself |
| `/lego/kalfa/init_state` |  | `(state, epochs, steps, resume=None, device=None)` | returns: epochs_left; bus: resume=resume, device=device; mutates: state | Move the state to the device, load a checkpoint when resuming, count the turns left |
| `/lego/kalfa/merge` |  | `(parts)` |  | Merge the per set metrics under train/, val/ and test/ |
| `/lego/kalfa/pack` |  | `(items)` | aliases: items | A mapping of the given items |
| `/lego/kalfa/predict` |  | `(models, composites, loader, prep, predicts, set, target_map=None, record=None, device=None)` | returns: predictions; bus: record=record, device=device | Predict the test set with the report model, invert the target chain, write predictions.parquet |
| `/lego/kalfa/run_all` |  | `(predictions, history, models, plots, keys=None, predicts=None, composites=None, valid_loader=None, test_loader=None, record=None)` | returns: None; bus: record=record, composites=composites, valid_loader=valid_loader, test_loader=test_loader | Run every plot of the plots table with the predictions, the history and the models; keys carry the extra inputs a plot names; plots that take loaders, predicts or name get them, name being the definition key the file is named after |
| `/lego/kalfa/save_final` |  | `(models, optimizers, emas, counters, rules, record=None)` | returns: None; bus: record=record | Write final/state.pt with the full state once training ends |
| `/lego/kalfa/select` |  | `(models, emas, which, record=None)` | returns: selected; bus: record=record | The report models: copies loaded from best.pt, or the final state for last |
| `/loader/kalfa/torch` |  | `(data, set, batch)` |  | torch DataLoader; shuffles the train set only, eval_size for the other sets; a stream dataset shuffles through its buffer and takes no sampler or workers |
| `/rule/kalfa/effects` |  | `(rules)` | returns: effects | The effects the fired rules left for this turn |
| `/rule/kalfa/open` |  | `(rules)` |  | Open the rule chain of a turn |
| `/rule/kalfa/rule` |  | `(rules, name, when, set, after=None, metrics=None, turn_index=None)` | returns: rules; bus: metrics=metrics, turn_index=turn_index | Evaluate one rule: skipped until its after rule fired in an earlier turn, sticky once fired, later rules win the same key |
| `/rule/kalfa/stop` |  | `(rules, triggers, metrics=None)` | returns: rules, stop; bus: metrics=metrics | Close the chain: the stop triggers are or'ed, their states kept under rules.stop |

## Alias packs

### /alias/kalfa/lazy

| Alias | URI | Kind |
|---|---|---|
| `parquet` | `/source/kalfa/parquet_stream` | source |
| `csv` | `/source/kalfa/csv_stream` | source |
| `sequential` | `/split/kalfa/sequential` | split |
| `standard_scaler` | `/pre/sklearn/standard_scaler` | pre |
| `minmax_scaler` | `/pre/sklearn/minmax_scaler` | pre |
| `cast` | `/pre/kalfa/cast` | pre |
| `abs` | `/pre/kalfa/abs` | pre |
| `log` | `/pre/kalfa/log` | pre |
| `one_hot` | `/pre/kalfa/one_hot` | pre |
| `label_encoder` | `/pre/kalfa/label_encoder` | pre |
| `table` | `/feed/kalfa/table` | feed |
| `linear` | `/layer/kalfa/linear` | layer |
| `linear_relu` | `/layer/kalfa/linear_relu` | layer |
| `concat` | `/layer/torch/concat` | layer |
| `flatten` | `/layer/torch/flatten` | layer |
| `relu` | `/layer/torch/relu` | layer |
| `leaky_relu` | `/layer/torch/leaky_relu` | layer |
| `dropout` | `/layer/torch/dropout` | layer |
| `reparam` | `/layer/kalfa/reparam` | layer |
| `l1_distance` | `/layer/kalfa/l1_distance` | layer |
| `gru` | `/layer/torch/gru` | layer |
| `last_step` | `/layer/torch/last_step` | layer |
| `normal` | `/init/torch/normal` | init |
| `xavier` | `/init/torch/xavier` | init |
| `kaiming` | `/init/torch/kaiming` | init |
| `zeros` | `/init/torch/zeros` | init |
| `mse` | `/criterion/kalfa/mse` | criterion |
| `huber` | `/criterion/kalfa/huber` | criterion |
| `mae` | `/criterion/kalfa/mae` | criterion |
| `log_cosh` | `/criterion/kalfa/log_cosh` | criterion |
| `cross_entropy` | `/criterion/kalfa/cross_entropy` | criterion |
| `bce_logits` | `/criterion/kalfa/bce_logits` | criterion |
| `rmse` | `/metric/kalfa/rmse` | metric |
| `accuracy` | `/metric/torchmetrics/accuracy` | metric |
| `f1` | `/metric/torchmetrics/f1` | metric |
| `auroc` | `/metric/torchmetrics/binary_auroc` | metric |
| `average_precision` | `/metric/torchmetrics/binary_average_precision` | metric |
| `adam` | `/optimizer/torch/adam` | optimizer |
| `adamw` | `/optimizer/torch/adamw` | optimizer |
| `sgd` | `/optimizer/torch/sgd` | optimizer |
| `warmup_cosine` | `/schedule/kalfa/warmup_cosine` | schedule |
| `linear_warmup` | `/schedule/kalfa/linear_warmup` | schedule |
| `step_decay` | `/schedule/kalfa/step_decay` | schedule |
| `supervised` | `/turn/kalfa/alternating` | turn |
| `alternating` | `/turn/kalfa/alternating` | turn |
| `plateau` | `/trigger/kalfa/plateau` | trigger |
| `metric_below` | `/trigger/kalfa/metric_below` | trigger |
| `metric_above` | `/trigger/kalfa/metric_above` | trigger |
| `after_turn` | `/trigger/kalfa/after_turn` | trigger |
| `after_epoch` | `/trigger/kalfa/after_turn` | trigger |
| `time_budget` | `/trigger/kalfa/time_budget` | trigger |
| `best` | `/checkpoint/kalfa/best` | checkpoint |
| `last` | `/checkpoint/kalfa/last` | checkpoint |
| `snapshot` | `/checkpoint/kalfa/snapshot` | checkpoint |
| `loss_curve` | `/plot/kalfa/loss_curve` | plot |
| `pred_vs_true` | `/plot/kalfa/pred_vs_true` | plot |
| `class_histogram` | `/plot/kalfa/class_histogram` | plot |
| `confusion_matrix` | `/plot/kalfa/confusion_matrix` | plot |
| `forecast_samples` | `/plot/kalfa/forecast_samples` | plot |
| `architecture` | `/plot/kalfa/architecture` | plot |
| `given` | `/split/kalfa/given` | split |
| `weighted_sum` | `/objective/kalfa/weighted_sum` | objective |
| `grid` | `/strategy/kalfa/grid` | strategy |
| `random` | `/strategy/kalfa/random` | strategy |
| `sobol` | `/strategy/kalfa/sobol` | strategy |
| `optuna` | `/strategy/kalfa/optuna` | strategy |
| `auto` | `/device/kalfa/auto` | device |
| `cpu` | `/device/kalfa/cpu` | device |
| `cuda` | `/device/kalfa/cuda` | device |
| `mps` | `/device/kalfa/mps` | device |

### /alias/kalfa/tabular

| Alias | URI | Kind |
|---|---|---|
| `parquet` | `/source/kalfa/parquet` | source |
| `csv` | `/source/kalfa/csv` | source |
| `random_split` | `/split/kalfa/random` | split |
| `sequential` | `/split/kalfa/sequential` | split |
| `kfold` | `/split/kalfa/kfold` | split |
| `standard_scaler` | `/pre/sklearn/standard_scaler` | pre |
| `minmax_scaler` | `/pre/sklearn/minmax_scaler` | pre |
| `max_abs_scaler` | `/pre/sklearn/max_abs_scaler` | pre |
| `robust_scaler` | `/pre/sklearn/robust_scaler` | pre |
| `quantile_transformer` | `/pre/sklearn/quantile_transformer` | pre |
| `power_transformer` | `/pre/sklearn/power_transformer` | pre |
| `asinh` | `/pre/kalfa/asinh` | pre |
| `sinh` | `/pre/kalfa/sinh` | pre |
| `tanh` | `/pre/kalfa/tanh` | pre |
| `atanh` | `/pre/kalfa/atanh` | pre |
| `cast` | `/pre/kalfa/cast` | pre |
| `abs` | `/pre/kalfa/abs` | pre |
| `log` | `/pre/kalfa/log` | pre |
| `one_hot` | `/pre/kalfa/one_hot` | pre |
| `kbins_discretizer` | `/pre/sklearn/kbins_discretizer` | pre |
| `spline_transformer` | `/pre/sklearn/spline_transformer` | pre |
| `label_encoder` | `/pre/kalfa/label_encoder` | pre |
| `table` | `/feed/kalfa/table` | feed |
| `window` | `/feed/kalfa/window` | feed |
| `linear` | `/layer/kalfa/linear` | layer |
| `polynomial` | `/layer/kalfa/polynomial` | layer |
| `l2_normalize` | `/layer/kalfa/l2_normalize` | layer |
| `linear_relu` | `/layer/kalfa/linear_relu` | layer |
| `concat` | `/layer/torch/concat` | layer |
| `flatten` | `/layer/torch/flatten` | layer |
| `relu` | `/layer/torch/relu` | layer |
| `leaky_relu` | `/layer/torch/leaky_relu` | layer |
| `dropout` | `/layer/torch/dropout` | layer |
| `reparam` | `/layer/kalfa/reparam` | layer |
| `l1_distance` | `/layer/kalfa/l1_distance` | layer |
| `gru` | `/layer/torch/gru` | layer |
| `last_step` | `/layer/torch/last_step` | layer |
| `normal` | `/init/torch/normal` | init |
| `xavier` | `/init/torch/xavier` | init |
| `kaiming` | `/init/torch/kaiming` | init |
| `zeros` | `/init/torch/zeros` | init |
| `mse` | `/criterion/kalfa/mse` | criterion |
| `huber` | `/criterion/kalfa/huber` | criterion |
| `mae` | `/criterion/kalfa/mae` | criterion |
| `log_cosh` | `/criterion/kalfa/log_cosh` | criterion |
| `cross_entropy` | `/criterion/kalfa/cross_entropy` | criterion |
| `bce_logits` | `/criterion/kalfa/bce_logits` | criterion |
| `rmse` | `/metric/kalfa/rmse` | metric |
| `accuracy` | `/metric/torchmetrics/accuracy` | metric |
| `f1` | `/metric/torchmetrics/f1` | metric |
| `auroc` | `/metric/torchmetrics/binary_auroc` | metric |
| `average_precision` | `/metric/torchmetrics/binary_average_precision` | metric |
| `adam` | `/optimizer/torch/adam` | optimizer |
| `adamw` | `/optimizer/torch/adamw` | optimizer |
| `sgd` | `/optimizer/torch/sgd` | optimizer |
| `warmup_cosine` | `/schedule/kalfa/warmup_cosine` | schedule |
| `linear_warmup` | `/schedule/kalfa/linear_warmup` | schedule |
| `step_decay` | `/schedule/kalfa/step_decay` | schedule |
| `supervised` | `/turn/kalfa/alternating` | turn |
| `alternating` | `/turn/kalfa/alternating` | turn |
| `plateau` | `/trigger/kalfa/plateau` | trigger |
| `metric_below` | `/trigger/kalfa/metric_below` | trigger |
| `metric_above` | `/trigger/kalfa/metric_above` | trigger |
| `after_turn` | `/trigger/kalfa/after_turn` | trigger |
| `after_epoch` | `/trigger/kalfa/after_turn` | trigger |
| `time_budget` | `/trigger/kalfa/time_budget` | trigger |
| `best` | `/checkpoint/kalfa/best` | checkpoint |
| `last` | `/checkpoint/kalfa/last` | checkpoint |
| `snapshot` | `/checkpoint/kalfa/snapshot` | checkpoint |
| `loss_curve` | `/plot/kalfa/loss_curve` | plot |
| `pred_vs_true` | `/plot/kalfa/pred_vs_true` | plot |
| `class_histogram` | `/plot/kalfa/class_histogram` | plot |
| `confusion_matrix` | `/plot/kalfa/confusion_matrix` | plot |
| `forecast_samples` | `/plot/kalfa/forecast_samples` | plot |
| `architecture` | `/plot/kalfa/architecture` | plot |
| `class_weights` | `/data/kalfa/class_weights` | data |
| `given` | `/split/kalfa/given` | split |
| `weighted_sum` | `/objective/kalfa/weighted_sum` | objective |
| `grid` | `/strategy/kalfa/grid` | strategy |
| `random` | `/strategy/kalfa/random` | strategy |
| `sobol` | `/strategy/kalfa/sobol` | strategy |
| `optuna` | `/strategy/kalfa/optuna` | strategy |
| `auto` | `/device/kalfa/auto` | device |
| `cpu` | `/device/kalfa/cpu` | device |
| `cuda` | `/device/kalfa/cuda` | device |
| `mps` | `/device/kalfa/mps` | device |

### /alias/kalfa/text

| Alias | URI | Kind |
|---|---|---|
| `parquet` | `/source/kalfa/parquet` | source |
| `csv` | `/source/kalfa/csv` | source |
| `random_split` | `/split/kalfa/random` | split |
| `sequential` | `/split/kalfa/sequential` | split |
| `kfold` | `/split/kalfa/kfold` | split |
| `standard_scaler` | `/pre/sklearn/standard_scaler` | pre |
| `minmax_scaler` | `/pre/sklearn/minmax_scaler` | pre |
| `cast` | `/pre/kalfa/cast` | pre |
| `abs` | `/pre/kalfa/abs` | pre |
| `log` | `/pre/kalfa/log` | pre |
| `one_hot` | `/pre/kalfa/one_hot` | pre |
| `label_encoder` | `/pre/kalfa/label_encoder` | pre |
| `table` | `/feed/kalfa/table` | feed |
| `window` | `/feed/kalfa/window` | feed |
| `linear` | `/layer/kalfa/linear` | layer |
| `linear_relu` | `/layer/kalfa/linear_relu` | layer |
| `concat` | `/layer/torch/concat` | layer |
| `flatten` | `/layer/torch/flatten` | layer |
| `relu` | `/layer/torch/relu` | layer |
| `leaky_relu` | `/layer/torch/leaky_relu` | layer |
| `dropout` | `/layer/torch/dropout` | layer |
| `reparam` | `/layer/kalfa/reparam` | layer |
| `l1_distance` | `/layer/kalfa/l1_distance` | layer |
| `gru` | `/layer/torch/gru` | layer |
| `last_step` | `/layer/torch/last_step` | layer |
| `normal` | `/init/torch/normal` | init |
| `xavier` | `/init/torch/xavier` | init |
| `kaiming` | `/init/torch/kaiming` | init |
| `zeros` | `/init/torch/zeros` | init |
| `mse` | `/criterion/kalfa/mse` | criterion |
| `huber` | `/criterion/kalfa/huber` | criterion |
| `mae` | `/criterion/kalfa/mae` | criterion |
| `log_cosh` | `/criterion/kalfa/log_cosh` | criterion |
| `cross_entropy` | `/criterion/kalfa/cross_entropy` | criterion |
| `bce_logits` | `/criterion/kalfa/bce_logits` | criterion |
| `rmse` | `/metric/kalfa/rmse` | metric |
| `accuracy` | `/metric/torchmetrics/accuracy` | metric |
| `f1` | `/metric/torchmetrics/f1` | metric |
| `auroc` | `/metric/torchmetrics/binary_auroc` | metric |
| `average_precision` | `/metric/torchmetrics/binary_average_precision` | metric |
| `adam` | `/optimizer/torch/adam` | optimizer |
| `adamw` | `/optimizer/torch/adamw` | optimizer |
| `sgd` | `/optimizer/torch/sgd` | optimizer |
| `warmup_cosine` | `/schedule/kalfa/warmup_cosine` | schedule |
| `linear_warmup` | `/schedule/kalfa/linear_warmup` | schedule |
| `step_decay` | `/schedule/kalfa/step_decay` | schedule |
| `supervised` | `/turn/kalfa/alternating` | turn |
| `alternating` | `/turn/kalfa/alternating` | turn |
| `plateau` | `/trigger/kalfa/plateau` | trigger |
| `metric_below` | `/trigger/kalfa/metric_below` | trigger |
| `metric_above` | `/trigger/kalfa/metric_above` | trigger |
| `after_turn` | `/trigger/kalfa/after_turn` | trigger |
| `after_epoch` | `/trigger/kalfa/after_turn` | trigger |
| `time_budget` | `/trigger/kalfa/time_budget` | trigger |
| `best` | `/checkpoint/kalfa/best` | checkpoint |
| `last` | `/checkpoint/kalfa/last` | checkpoint |
| `snapshot` | `/checkpoint/kalfa/snapshot` | checkpoint |
| `loss_curve` | `/plot/kalfa/loss_curve` | plot |
| `pred_vs_true` | `/plot/kalfa/pred_vs_true` | plot |
| `class_histogram` | `/plot/kalfa/class_histogram` | plot |
| `confusion_matrix` | `/plot/kalfa/confusion_matrix` | plot |
| `forecast_samples` | `/plot/kalfa/forecast_samples` | plot |
| `architecture` | `/plot/kalfa/architecture` | plot |
| `class_weights` | `/data/kalfa/class_weights` | data |
| `text_lines` | `/source/kalfa/text_lines` | source |
| `char_tokenizer` | `/pre/kalfa/char_tokenizer` | pre |
| `next_token` | `/feed/kalfa/next_token` | feed |
| `embedding` | `/layer/torch/embedding` | layer |
| `perplexity` | `/metric/kalfa/perplexity` | metric |
| `lm_sampler` | `/generate/kalfa/lm_sampler` | generate |
| `given` | `/split/kalfa/given` | split |
| `weighted_sum` | `/objective/kalfa/weighted_sum` | objective |
| `vocab_size` | `/data/kalfa/vocab_size` | data |
| `grid` | `/strategy/kalfa/grid` | strategy |
| `random` | `/strategy/kalfa/random` | strategy |
| `sobol` | `/strategy/kalfa/sobol` | strategy |
| `optuna` | `/strategy/kalfa/optuna` | strategy |
| `sample_writer` | `/metric/kalfa/sample_writer` | metric |
| `auto` | `/device/kalfa/auto` | device |
| `cpu` | `/device/kalfa/cpu` | device |
| `cuda` | `/device/kalfa/cuda` | device |
| `mps` | `/device/kalfa/mps` | device |

### /alias/kalfa/vision

| Alias | URI | Kind |
|---|---|---|
| `parquet` | `/source/kalfa/parquet` | source |
| `csv` | `/source/kalfa/csv` | source |
| `random_split` | `/split/kalfa/random` | split |
| `sequential` | `/split/kalfa/sequential` | split |
| `kfold` | `/split/kalfa/kfold` | split |
| `standard_scaler` | `/pre/sklearn/standard_scaler` | pre |
| `minmax_scaler` | `/pre/sklearn/minmax_scaler` | pre |
| `cast` | `/pre/kalfa/cast` | pre |
| `abs` | `/pre/kalfa/abs` | pre |
| `log` | `/pre/kalfa/log` | pre |
| `one_hot` | `/pre/kalfa/one_hot` | pre |
| `label_encoder` | `/pre/kalfa/label_encoder` | pre |
| `table` | `/feed/kalfa/table` | feed |
| `window` | `/feed/kalfa/window` | feed |
| `linear` | `/layer/kalfa/linear` | layer |
| `linear_relu` | `/layer/kalfa/linear_relu` | layer |
| `concat` | `/layer/torch/concat` | layer |
| `flatten` | `/layer/torch/flatten` | layer |
| `relu` | `/layer/torch/relu` | layer |
| `leaky_relu` | `/layer/torch/leaky_relu` | layer |
| `dropout` | `/layer/torch/dropout` | layer |
| `reparam` | `/layer/kalfa/reparam` | layer |
| `l1_distance` | `/layer/kalfa/l1_distance` | layer |
| `gru` | `/layer/torch/gru` | layer |
| `last_step` | `/layer/torch/last_step` | layer |
| `normal` | `/init/torch/normal` | init |
| `xavier` | `/init/torch/xavier` | init |
| `kaiming` | `/init/torch/kaiming` | init |
| `zeros` | `/init/torch/zeros` | init |
| `mse` | `/criterion/kalfa/mse` | criterion |
| `huber` | `/criterion/kalfa/huber` | criterion |
| `mae` | `/criterion/kalfa/mae` | criterion |
| `log_cosh` | `/criterion/kalfa/log_cosh` | criterion |
| `cross_entropy` | `/criterion/kalfa/cross_entropy` | criterion |
| `bce_logits` | `/criterion/kalfa/bce_logits` | criterion |
| `rmse` | `/metric/kalfa/rmse` | metric |
| `accuracy` | `/metric/torchmetrics/accuracy` | metric |
| `f1` | `/metric/torchmetrics/f1` | metric |
| `auroc` | `/metric/torchmetrics/binary_auroc` | metric |
| `average_precision` | `/metric/torchmetrics/binary_average_precision` | metric |
| `adam` | `/optimizer/torch/adam` | optimizer |
| `adamw` | `/optimizer/torch/adamw` | optimizer |
| `sgd` | `/optimizer/torch/sgd` | optimizer |
| `warmup_cosine` | `/schedule/kalfa/warmup_cosine` | schedule |
| `linear_warmup` | `/schedule/kalfa/linear_warmup` | schedule |
| `step_decay` | `/schedule/kalfa/step_decay` | schedule |
| `supervised` | `/turn/kalfa/alternating` | turn |
| `alternating` | `/turn/kalfa/alternating` | turn |
| `plateau` | `/trigger/kalfa/plateau` | trigger |
| `metric_below` | `/trigger/kalfa/metric_below` | trigger |
| `metric_above` | `/trigger/kalfa/metric_above` | trigger |
| `after_turn` | `/trigger/kalfa/after_turn` | trigger |
| `after_epoch` | `/trigger/kalfa/after_turn` | trigger |
| `time_budget` | `/trigger/kalfa/time_budget` | trigger |
| `best` | `/checkpoint/kalfa/best` | checkpoint |
| `last` | `/checkpoint/kalfa/last` | checkpoint |
| `snapshot` | `/checkpoint/kalfa/snapshot` | checkpoint |
| `loss_curve` | `/plot/kalfa/loss_curve` | plot |
| `pred_vs_true` | `/plot/kalfa/pred_vs_true` | plot |
| `class_histogram` | `/plot/kalfa/class_histogram` | plot |
| `confusion_matrix` | `/plot/kalfa/confusion_matrix` | plot |
| `forecast_samples` | `/plot/kalfa/forecast_samples` | plot |
| `architecture` | `/plot/kalfa/architecture` | plot |
| `class_weights` | `/data/kalfa/class_weights` | data |
| `image_folder` | `/source/kalfa/image_folder` | source |
| `to_tensor` | `/pre/kalfa/to_tensor` | pre |
| `to_tensor_signed` | `/pre/kalfa/to_tensor_signed` | pre |
| `resize` | `/pre/kalfa/resize` | pre |
| `random_crop_flip` | `/pre/kalfa/random_crop_flip` | pre |
| `normalize` | `/pre/kalfa/normalize` | pre |
| `unflatten` | `/layer/kalfa/unflatten` | layer |
| `image_pairs` | `/plot/kalfa/image_pairs` | plot |
| `vae` | `/objective/kalfa/vae` | objective |
| `distill` | `/objective/kalfa/distill` | objective |
| `wgan_gp_d` | `/objective/kalfa/wgan_gp_d` | objective |
| `wgan_g` | `/objective/kalfa/wgan_g` | objective |
| `fid` | `/metric/kalfa/fid` | metric |
| `gan_sampler` | `/generate/kalfa/gan_sampler` | generate |
| `ddpm` | `/objective/kalfa/ddpm` | objective |
| `ddpm_sampler` | `/generate/kalfa/ddpm_sampler` | generate |
| `linear_betas` | `/schedule/kalfa/linear_betas` | schedule |
| `two_views` | `/pre/kalfa/two_views` | pre |
| `simclr_aug` | `/pre/kalfa/simclr_aug` | pre |
| `ntxent` | `/objective/kalfa/ntxent` | objective |
| `embedding` | `/layer/torch/embedding` | layer |
| `given` | `/split/kalfa/given` | split |
| `weighted_sum` | `/objective/kalfa/weighted_sum` | objective |
| `conv2d` | `/layer/torch/conv2d` | layer |
| `maxpool` | `/layer/torch/maxpool` | layer |
| `image_grid` | `/plot/kalfa/image_grid` | plot |
| `recon_error` | `/metric/kalfa/recon_error` | metric |
| `grid` | `/strategy/kalfa/grid` | strategy |
| `random` | `/strategy/kalfa/random` | strategy |
| `sobol` | `/strategy/kalfa/sobol` | strategy |
| `optuna` | `/strategy/kalfa/optuna` | strategy |
| `sample_writer` | `/metric/kalfa/sample_writer` | metric |
| `samples_gif` | `/plot/kalfa/samples_gif` | plot |
| `samples_matrix` | `/plot/kalfa/samples_matrix` | plot |
| `auto` | `/device/kalfa/auto` | device |
| `cpu` | `/device/kalfa/cpu` | device |
| `cuda` | `/device/kalfa/cuda` | device |
| `mps` | `/device/kalfa/mps` | device |
