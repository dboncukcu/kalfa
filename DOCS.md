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
| `source` | data.source | 7 |
| `transform` | data.transform | 5 |
| `split` | data.split | 5 |
| `frame` | data.frame | 2 |
| `pre` | data.preprocessors | 28 |
| `feed` | data.feed | 3 |
| `layer` | model nodes | 118 |
| `init` | model init | 5 |
| `criterion` | losses, metrics | 7 |
| `objective` | losses | 7 |
| `metric` | metrics | 9 |
| `adapter` | the driver | 3 |
| `optimizer` | optimizers | 3 |
| `schedule` | optimizer schedule | 4 |
| `turn` | training.turn | 1 |
| `trigger` | training.stop, rules when | 5 |
| `checkpoint` | training.checkpoint | 3 |
| `generate` | generate | 3 |
| `plot` | plots | 25 |
| `strategy` | sweep.strategy | 4 |
| `device` | device, predict --device, generate --device | 4 |
| `rng` | rng | 3 |
| `export` | kalfa export --format | 3 |
| `calibrate` | calibrate | 1 |
| `lego` | a param value, or the contract | 1 |
| `data` | a param value ({uri: name}) | 4 |

### source

| URI | Alias | Signature | Facts | Description |
|---|---|---|---|---|
| `/source/kalfa/csv` | `csv` | `(path, columns=None)` | returns: df; header: /lego/kalfa/csv_header | Read a CSV file into a DataFrame; columns lists the ones to read, the others stay on disk |
| `/source/kalfa/csv_stream` |  | `(path, chunk=65536, columns=None)` | returns: df; header: /lego/kalfa/csv_header; stream: True | Read a CSV file in chunks (the lazy set) |
| `/source/kalfa/image_folder` | `image_folder` | `(path)` | returns: df; header: /lego/kalfa/image_folder_header; samples: True | Images under root/<class>/ as a Dataset with fields image and label |
| `/source/kalfa/parquet` | `parquet` | `(path, columns=None)` | returns: df; header: /lego/kalfa/parquet_header | Read a parquet file into a DataFrame; columns lists the ones to read, the others stay on disk |
| `/source/kalfa/parquet_stream` |  | `(path, chunk=65536, columns=None)` | returns: df; header: /lego/kalfa/parquet_header; stream: True | Read a parquet file in chunks (the lazy set): a stream the data legos filter, cut and fit without loading the table |
| `/source/kalfa/prepared` |  | `(path)` | returns: df; header: /lego/kalfa/prepared_header | The data kalfa prepare wrote: the sets of a table read back into one frame marked by set, or the items of a Dataset source read from where they are with the split kept as positions |
| `/source/kalfa/text_lines` | `text_lines` | `(path)` | returns: df; header: /lego/kalfa/text_lines_header; samples: True | The lines of a text file as a Dataset with the field text |

### transform

| URI | Alias | Signature | Facts | Description |
|---|---|---|---|---|
| `/transform/kalfa/astype` | `astype` | `(df, columns)` | partial: True; needs_table: True | Cast columns to dtypes, {column: dtype} |
| `/transform/kalfa/derive` | `derive` | `(df, column, expr)` | partial: True; needs_table: True | A new column from a pandas eval expression over the frame (log10(x), a > b, a + b) |
| `/transform/kalfa/drop` | `drop` | `(df, columns)` | partial: True; needs_table: True | Drop columns from the frame before anything reads them |
| `/transform/kalfa/filter` | `filter` | `(df, query)` | partial: True | Keep the rows a pandas query selects; a bare string in data.transform is this call; a stream applies it chunk by chunk and a Dataset source takes field equality queries |
| `/transform/kalfa/rename` | `rename` | `(df, pattern, to)` | partial: True; needs_table: True | Rename the columns a regular expression matches, pattern to the replacement, backreferences allowed (cms_(.*)_Z_score to z_\1) |

### split

| URI | Alias | Signature | Facts | Description |
|---|---|---|---|---|
| `/split/kalfa/given` | `given` | `(df, valid=None, test=None)` | returns: train, valid, test; sizes: /lego/kalfa/given_sizes | The source is the train set; valid and test come from the given paths, read like the source (a missing path means no set) |
| `/split/kalfa/kfold` | `kfold` | `(df, k, fold, val=None, seed=None)` | returns: train, valid, test; sizes: /lego/kalfa/kfold_sizes; needs_table: True | k folds of a seeded permutation: the held out fold is the test set, val carves the valid set from the rest; without val there is no valid set |
| `/split/kalfa/prepared` |  | `(df, path)` | returns: train, valid, test; sizes: /lego/kalfa/prepared_sizes | The split kalfa prepare recorded: the sets a prepared frame is marked with, or the positions of a Dataset source's items per set |
| `/split/kalfa/random` | `random_split` | `(df, ratios, seed=None)` | returns: train, valid, test; sizes: /lego/kalfa/ratio_sizes; needs_table: True | Shuffle the rows with a seed and cut them by ratios into train, valid and test; the short form of a split without a uri |
| `/split/kalfa/sequential` | `sequential` | `(df, ratios, group=None)` | returns: train, valid, test; refs: group=column; sizes: /lego/kalfa/ratio_sizes | Cut the rows in their order by ratios; with a group column every group is cut on its own |

### frame

| URI | Alias | Signature | Facts | Description |
|---|---|---|---|---|
| `/frame/kalfa/group_statistic` | `group_statistic` | `(by, column, statistic='mean', name=None)` | refs: by=column; needs_table: True | A statistic of a column per group, learned on the train set and mapped onto every set as a new column (name, or <column>_<statistic>_by_<by>); a group the train set never saw takes the statistic over the whole train set |
| `/frame/kalfa/target_encoding` | `target_encoding` | `(column, target, smoothing=1.0, name=None)` | refs: column=column; needs_table: True | The train mean of the target per category of a column, smoothed toward the overall mean by smoothing pseudo counts, as a new column (name, or <column>_target); a category the train set never saw takes the overall mean |

### pre

| URI | Alias | Signature | Facts | Description |
|---|---|---|---|---|
| `/pre/kalfa/abs` | `abs` | `()` |  | Absolute value of a column |
| `/pre/kalfa/asinh` | `asinh` | `(scale=1.0, overflow=700.0)` |  | Signed log scale of a heavy tailed column: arcsinh(x / scale), inverted by scale sinh(y); keeps the sign, linear near zero, logarithmic in the tails, defined at zero; the inverse refuses values past overflow, where sinh leaves float64 |
| `/pre/kalfa/atanh` | `atanh` | `(scale=1.0)` |  | artanh(x / scale) of a bounded column, inverted by scale tanh(y); a value outside (-scale, scale) is an error that names how many and how large |
| `/pre/kalfa/cast` | `cast` | `(dtype)` |  | Cast a column to a numpy dtype |
| `/pre/kalfa/char_tokenizer` | `char_tokenizer` | `()` | state: True | Character level tokenizer fitted on the train text; the vocabulary goes into the record |
| `/pre/kalfa/fill` | `fill` | `(value=None, method=None)` |  | Fill the missing values of a column without a fit: a constant (a number, or a name such as missing that becomes its own category), or method ffill or bfill along the rows |
| `/pre/kalfa/label_encoder` | `label_encoder` | `()` | state: True | Integer codes of a label column, sorted by label; inverted in reports and predictions, class scores decode to labels |
| `/pre/kalfa/log` | `log` | `(base=10.0, norm=1.0)` |  | log1p of a column divided by norm, in the given base |
| `/pre/kalfa/median_std_scaler` | `median_std_scaler` | `()` | grouped: True | Center a column on its median and scale it by its standard deviation: the center an outlier does not move, the scale of a standard scaler |
| `/pre/kalfa/normalize` | `normalize` | `(mean, std)` |  | Normalize an image tensor per channel; mean and std are numbers, lists or the presets imagenet and cifar10 |
| `/pre/kalfa/one_hot` | `one_hot` | `()` | state: True | One hot columns <field>_<category> of a categorical column; unknown categories give zeros |
| `/pre/kalfa/random_crop_flip` | `random_crop_flip` | `(size)` |  | Random crop of size after padding and a random horizontal flip |
| `/pre/kalfa/resize` | `resize` | `(size)` |  | Resize an image to size (int or [h, w]) |
| `/pre/kalfa/simclr_aug` | `simclr_aug` | `(size, scale=(0.5, 1.0))` |  | SimCLR augmentation: random resized crop to size, horizontal flip, brightness jitter |
| `/pre/kalfa/simple_imputer` | `simple_imputer` | `(strategy='mean', fill_value=None, indicator=False)` |  | Fill the missing values of a column with the train mean, median, most frequent value or a constant; indicator adds <field>_missing, computed before the fill, as a feature of its own |
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
| `/feed/kalfa/window` | `window` | `(frame, frames, size, horizon, context=False, group=None)` | refs: group=column; needs_table: True | Windows of size steps and the next horizon steps of the targets; context takes the tail of the previous set at the split boundary, group keeps series apart |

### layer

| URI | Alias | Signature | Facts | Description |
|---|---|---|---|---|
| `/layer/kalfa/l1_distance` | `l1_distance` | `()` |  | Mean absolute difference of two wires per sample |
| `/layer/kalfa/l2_normalize` | `l2_normalize` | `(eps=1e-12)` |  | Divide every sample by the L2 norm of its own feature vector (sklearn's Normalizer as a layer: it reads the whole vector, so it belongs to the model, not to a column chain) |
| `/layer/kalfa/linear` | `linear` | `(out_features, in_features=None)` |  | Linear layer; without in_features the input width is taken from the first batch |
| `/layer/kalfa/linear_relu` | `linear_relu` | `(out_features, in_features=None)` |  | Linear layer followed by ReLU; lazy without in_features |
| `/layer/kalfa/mlp` | `mlp` | `(widths, activation='relu', dropout=0.0, out_features=None, in_features=None)` |  | A multilayer perceptron in one node: a linear layer, the activation and dropout for every width, then a plain linear layer of out_features when it is written; lazy without in_features |
| `/layer/kalfa/polynomial` | `polynomial` | `(degree=2, interaction_only=False, bias=False, keep=True)` |  | Polynomial expansion of the feature vector: the features and every product of degree of them (interaction_only drops the squares, bias adds a constant column, keep: false returns the products alone); the place for feature interactions, computed per batch |
| `/layer/kalfa/reparam` | `reparam` | `()` |  | Sample z from mu and logvar in train mode, return mu in eval mode |
| `/layer/kalfa/unflatten` | `unflatten` | `(shape)` |  | Reshape the features of every sample to shape |
| `/layer/torch/adaptive_avgpool` | `adaptive_avgpool` | `(output_size=1)` |  | torch.nn.AdaptiveAvgPool2d to output_size, a number or [height, width] |
| `/layer/torch/adaptive_avgpool1d` | `adaptive_avgpool1d` | `(output_size=1)` |  | torch.nn.AdaptiveAvgPool1d |
| `/layer/torch/adaptive_avgpool3d` | `adaptive_avgpool3d` | `(output_size=1)` |  | torch.nn.AdaptiveAvgPool3d |
| `/layer/torch/adaptive_maxpool` | `adaptive_maxpool` | `(output_size=1)` |  | torch.nn.AdaptiveMaxPool2d to output_size, a number or [height, width] |
| `/layer/torch/adaptive_maxpool1d` | `adaptive_maxpool1d` | `(output_size=1)` |  | torch.nn.AdaptiveMaxPool1d |
| `/layer/torch/adaptive_maxpool3d` | `adaptive_maxpool3d` | `(output_size=1)` |  | torch.nn.AdaptiveMaxPool3d |
| `/layer/torch/alpha_dropout` | `alpha_dropout` | `(p=0.5)` |  | torch.nn.AlphaDropout, the dropout of selu |
| `/layer/torch/avgpool` | `avgpool` | `(kernel, stride=None, padding=0)` |  | torch.nn.AvgPool2d |
| `/layer/torch/avgpool1d` | `avgpool1d` | `(kernel, stride=None, padding=0)` |  | torch.nn.AvgPool1d |
| `/layer/torch/avgpool3d` | `avgpool3d` | `(kernel, stride=None, padding=0)` |  | torch.nn.AvgPool3d |
| `/layer/torch/batch_norm` | `batch_norm` | `(dims=1, eps=1e-05, momentum=0.1, affine=True)` |  | torch.nn.BatchNorm over the feature axis, lazy in the number of features: dims 1 for (batch, features) and sequences, 2 for images, 3 for volumes |
| `/layer/torch/bilinear` | `bilinear` | `(in1_features, in2_features, out_features, bias=True)` |  | torch.nn.Bilinear over two inputs of in1_features and in2_features |
| `/layer/torch/celu` | `celu` | `(alpha=1.0)` |  | torch.nn.CELU with alpha |
| `/layer/torch/channel_shuffle` | `channel_shuffle` | `(groups)` |  | torch.nn.ChannelShuffle over groups |
| `/layer/torch/circular_pad` | `circular_pad` | `(padding)` |  | torch.nn.CircularPad2d |
| `/layer/torch/circular_pad1d` | `circular_pad1d` | `(padding)` |  | torch.nn.CircularPad1d |
| `/layer/torch/circular_pad3d` | `circular_pad3d` | `(padding)` |  | torch.nn.CircularPad3d |
| `/layer/torch/concat` | `concat` | `(dim=1)` |  | Concatenate wires along a dimension |
| `/layer/torch/constant_pad` | `constant_pad` | `(padding, value=0.0)` |  | torch.nn.ConstantPad2d with value |
| `/layer/torch/constant_pad1d` | `constant_pad1d` | `(padding, value=0.0)` |  | torch.nn.ConstantPad1d |
| `/layer/torch/constant_pad3d` | `constant_pad3d` | `(padding, value=0.0)` |  | torch.nn.ConstantPad3d |
| `/layer/torch/conv1d` | `conv1d` | `(out_channels, kernel, stride=1, padding=0, dilation=1, groups=1, bias=True, in_channels=None)` |  | torch.nn.Conv1d; without in_channels the input channels are taken from the first batch |
| `/layer/torch/conv2d` | `conv2d` | `(out_channels, kernel, stride=1, padding=0, dilation=1, groups=1, bias=True, in_channels=None)` |  | 2d convolution; without in_channels the input channels are taken from the first batch |
| `/layer/torch/conv3d` | `conv3d` | `(out_channels, kernel, stride=1, padding=0, dilation=1, groups=1, bias=True, in_channels=None)` |  | torch.nn.Conv3d; without in_channels the input channels are taken from the first batch |
| `/layer/torch/conv_transpose1d` | `conv_transpose1d` | `(out_channels, kernel, stride=1, padding=0, output_padding=0, dilation=1, groups=1, bias=True, in_channels=None)` |  | torch.nn.ConvTranspose1d; without in_channels the input channels are taken from the first batch |
| `/layer/torch/conv_transpose2d` | `conv_transpose2d` | `(out_channels, kernel, stride=1, padding=0, output_padding=0, dilation=1, groups=1, bias=True, in_channels=None)` |  | torch.nn.ConvTranspose2d; without in_channels the input channels are taken from the first batch |
| `/layer/torch/conv_transpose3d` | `conv_transpose3d` | `(out_channels, kernel, stride=1, padding=0, output_padding=0, dilation=1, groups=1, bias=True, in_channels=None)` |  | torch.nn.ConvTranspose3d; without in_channels the input channels are taken from the first batch |
| `/layer/torch/cosine_similarity` | `cosine_similarity` | `(dim=1, eps=1e-08)` |  | torch.nn.CosineSimilarity of two inputs along dim |
| `/layer/torch/dropout` | `dropout` | `(p=0.5)` |  | torch.nn.Dropout |
| `/layer/torch/dropout1d` | `dropout1d` | `(p=0.5)` |  | torch.nn.Dropout1d, whole channels of a sequence |
| `/layer/torch/dropout2d` | `dropout2d` | `(p=0.5)` |  | torch.nn.Dropout2d, whole channels of an image |
| `/layer/torch/dropout3d` | `dropout3d` | `(p=0.5)` |  | torch.nn.Dropout3d, whole channels of a volume |
| `/layer/torch/elu` | `elu` | `(alpha=1.0)` |  | torch.nn.ELU with alpha |
| `/layer/torch/embedding` | `embedding` | `(num, dim, padding_idx=None)` |  | torch.nn.Embedding(num, dim); num may be a kind data component such as vocab_size |
| `/layer/torch/embedding_bag` | `embedding_bag` | `(num, dim, mode='mean')` |  | torch.nn.EmbeddingBag(num, dim), the mean, sum or max of a bag of ids; num may be vocab_size |
| `/layer/torch/feature_alpha_dropout` | `feature_alpha_dropout` | `(p=0.5)` |  | torch.nn.FeatureAlphaDropout, alpha dropout of whole channels |
| `/layer/torch/flatten` | `flatten` | `(start_dim=1, end_dim=-1)` |  | torch.nn.Flatten from start_dim to end_dim |
| `/layer/torch/fold` | `fold` | `(output_size, kernel, dilation=1, padding=0, stride=1)` |  | torch.nn.Fold, sliding blocks back into an image of output_size |
| `/layer/torch/fractional_maxpool` | `fractional_maxpool` | `(kernel, output_size=None, output_ratio=None)` |  | torch.nn.FractionalMaxPool2d to output_size or output_ratio |
| `/layer/torch/fractional_maxpool3d` | `fractional_maxpool3d` | `(kernel, output_size=None, output_ratio=None)` |  | torch.nn.FractionalMaxPool3d |
| `/layer/torch/gelu` | `gelu` | `(approximate='none')` |  | torch.nn.GELU; approximate none or tanh |
| `/layer/torch/glu` | `glu` | `(dim=-1)` |  | torch.nn.GLU, the gated linear unit over dim |
| `/layer/torch/group_norm` | `group_norm` | `(num_groups, num_channels, eps=1e-05, affine=True)` |  | torch.nn.GroupNorm: num_groups groups over num_channels channels |
| `/layer/torch/gru` | `gru` | `(hidden, layers=1, dropout=0.0, bidirectional=False)` |  | GRU over (batch, steps, features) returning every step; the input width comes from the first batch |
| `/layer/torch/gru_cell` | `gru_cell` | `(input_size, hidden, bias=True)` |  | torch.nn.GRUCell over one step; two inputs, x and h |
| `/layer/torch/hardshrink` | `hardshrink` | `(lambd=0.5)` |  | torch.nn.Hardshrink, zero inside [-lambd, lambd] |
| `/layer/torch/hardsigmoid` | `hardsigmoid` | `()` |  | torch.nn.Hardsigmoid |
| `/layer/torch/hardswish` | `hardswish` | `()` |  | torch.nn.Hardswish |
| `/layer/torch/hardtanh` | `hardtanh` | `(min_val=-1.0, max_val=1.0)` |  | torch.nn.Hardtanh clipped to [min_val, max_val] |
| `/layer/torch/identity` | `identity` | `()` |  | torch.nn.Identity, the input as it is |
| `/layer/torch/instance_norm` | `instance_norm` | `(dims=2, eps=1e-05, momentum=0.1, affine=False)` |  | torch.nn.InstanceNorm, lazy in the number of features: dims 1 for sequences, 2 for images, 3 for volumes |
| `/layer/torch/last_step` | `last_step` | `()` |  | The last step of a sequence |
| `/layer/torch/layer_norm` | `layer_norm` | `(normalized_shape, eps=1e-05, elementwise_affine=True)` |  | torch.nn.LayerNorm over the last axis; normalized_shape is its width, a number or {uri: feature_width} for the width of the feature tensor |
| `/layer/torch/leaky_relu` | `leaky_relu` | `(negative_slope=0.01)` |  | torch.nn.LeakyReLU with negative_slope |
| `/layer/torch/linear` |  | `(in_features, out_features, bias=True)` |  | torch.nn.Linear with in_features written out |
| `/layer/torch/local_response_norm` | `local_response_norm` | `(size, alpha=0.0001, beta=0.75, k=1.0)` |  | torch.nn.LocalResponseNorm over size neighbouring channels |
| `/layer/torch/log_sigmoid` | `log_sigmoid` | `()` |  | torch.nn.LogSigmoid |
| `/layer/torch/log_softmax` | `log_softmax` | `(dim=-1)` |  | torch.nn.LogSoftmax over dim |
| `/layer/torch/lppool` | `lppool` | `(norm_type, kernel, stride=None)` |  | torch.nn.LPPool2d, the power average pool of norm_type |
| `/layer/torch/lppool1d` | `lppool1d` | `(norm_type, kernel, stride=None)` |  | torch.nn.LPPool1d |
| `/layer/torch/lppool3d` | `lppool3d` | `(norm_type, kernel, stride=None)` |  | torch.nn.LPPool3d |
| `/layer/torch/lstm` | `lstm` | `(hidden, layers=1, dropout=0.0, bidirectional=False)` |  | LSTM over (batch, steps, features) returning every step; the input width comes from the first batch |
| `/layer/torch/lstm_cell` | `lstm_cell` | `(input_size, hidden, bias=True)` |  | torch.nn.LSTMCell over one step; inputs x and (h, c), outputs (h, c) |
| `/layer/torch/max_unpool` | `max_unpool` | `(kernel, stride=None, padding=0)` |  | torch.nn.MaxUnpool2d, the inverse of a max pool that kept its indices; two inputs |
| `/layer/torch/max_unpool1d` | `max_unpool1d` | `(kernel, stride=None, padding=0)` |  | torch.nn.MaxUnpool1d; two inputs |
| `/layer/torch/max_unpool3d` | `max_unpool3d` | `(kernel, stride=None, padding=0)` |  | torch.nn.MaxUnpool3d; two inputs |
| `/layer/torch/maxpool` | `maxpool` | `(kernel, stride=None, padding=0)` |  | torch.nn.MaxPool2d |
| `/layer/torch/maxpool1d` | `maxpool1d` | `(kernel, stride=None, padding=0)` |  | torch.nn.MaxPool1d |
| `/layer/torch/maxpool3d` | `maxpool3d` | `(kernel, stride=None, padding=0)` |  | torch.nn.MaxPool3d |
| `/layer/torch/mish` | `mish` | `()` |  | torch.nn.Mish |
| `/layer/torch/multihead_attention` | `multihead_attention` | `(embed_dim, heads, dropout=0.0, bias=True, batch_first=True)` |  | torch.nn.MultiheadAttention over (batch, steps, embed_dim); inputs query, key and value, outputs the attended values and the weights |
| `/layer/torch/pairwise_distance` | `pairwise_distance` | `(p=2.0, eps=1e-06, keepdim=False)` |  | torch.nn.PairwiseDistance of two inputs, the p norm of their difference |
| `/layer/torch/pixel_shuffle` | `pixel_shuffle` | `(factor)` |  | torch.nn.PixelShuffle by factor |
| `/layer/torch/pixel_unshuffle` | `pixel_unshuffle` | `(factor)` |  | torch.nn.PixelUnshuffle by factor |
| `/layer/torch/prelu` | `prelu` | `(num_parameters=1, init=0.25)` |  | torch.nn.PReLU, a learned slope per channel (num_parameters) starting at init |
| `/layer/torch/reflection_pad` | `reflection_pad` | `(padding)` |  | torch.nn.ReflectionPad2d |
| `/layer/torch/reflection_pad1d` | `reflection_pad1d` | `(padding)` |  | torch.nn.ReflectionPad1d |
| `/layer/torch/reflection_pad3d` | `reflection_pad3d` | `(padding)` |  | torch.nn.ReflectionPad3d |
| `/layer/torch/relu` | `relu` | `()` |  | torch.nn.ReLU |
| `/layer/torch/relu6` | `relu6` | `()` |  | torch.nn.ReLU6, ReLU clipped at 6 |
| `/layer/torch/replication_pad` | `replication_pad` | `(padding)` |  | torch.nn.ReplicationPad2d |
| `/layer/torch/replication_pad1d` | `replication_pad1d` | `(padding)` |  | torch.nn.ReplicationPad1d |
| `/layer/torch/replication_pad3d` | `replication_pad3d` | `(padding)` |  | torch.nn.ReplicationPad3d |
| `/layer/torch/rms_norm` | `rms_norm` | `(normalized_shape, eps=None, elementwise_affine=True)` |  | torch.nn.RMSNorm over the last axis; normalized_shape is its width, a number or {uri: feature_width} |
| `/layer/torch/rnn` | `rnn` | `(hidden, layers=1, dropout=0.0, bidirectional=False, nonlinearity='tanh')` |  | Elman RNN over (batch, steps, features) returning every step, nonlinearity tanh or relu; the input width comes from the first batch |
| `/layer/torch/rnn_cell` | `rnn_cell` | `(input_size, hidden, bias=True, nonlinearity='tanh')` |  | torch.nn.RNNCell over one step; two inputs, x and h |
| `/layer/torch/rrelu` | `rrelu` | `(lower=0.125, upper=0.3333333333333333)` |  | torch.nn.RReLU, a random slope in [lower, upper] in train mode |
| `/layer/torch/selu` | `selu` | `()` |  | torch.nn.SELU, self normalizing; pairs with alpha_dropout |
| `/layer/torch/sigmoid` | `sigmoid` | `()` |  | torch.nn.Sigmoid |
| `/layer/torch/silu` | `silu` | `()` |  | torch.nn.SiLU, x times sigmoid(x) |
| `/layer/torch/softmax` | `softmax` | `(dim=-1)` |  | torch.nn.Softmax over dim |
| `/layer/torch/softmax2d` | `softmax2d` | `()` |  | torch.nn.Softmax2d, the softmax over the channels of an image |
| `/layer/torch/softmin` | `softmin` | `(dim=-1)` |  | torch.nn.Softmin over dim |
| `/layer/torch/softplus` | `softplus` | `(beta=1.0, threshold=20.0)` |  | torch.nn.Softplus with beta and the linear threshold |
| `/layer/torch/softshrink` | `softshrink` | `(lambd=0.5)` |  | torch.nn.Softshrink, shrunk toward zero by lambd |
| `/layer/torch/softsign` | `softsign` | `()` |  | torch.nn.Softsign, x over 1 plus \|x\| |
| `/layer/torch/tanh` | `tanh_layer` | `()` |  | torch.nn.Tanh; the alias is tanh_layer because tanh names the preprocessor |
| `/layer/torch/tanhshrink` | `tanhshrink` | `()` |  | torch.nn.Tanhshrink, x minus tanh(x) |
| `/layer/torch/threshold` | `threshold_layer` | `(threshold, value)` |  | torch.nn.Threshold: value where x is at or below threshold; the alias is threshold_layer because threshold names the calibration |
| `/layer/torch/transformer` | `transformer` | `(d_model=512, heads=8, encoder_layers=6, decoder_layers=6, dim_feedforward=2048, dropout=0.1, activation='relu', norm_first=False, batch_first=True)` |  | torch.nn.Transformer, the encoder and the decoder; inputs the source and the target sequences |
| `/layer/torch/transformer_decoder` | `transformer_decoder` | `(d_model, heads, layers, dim_feedforward=2048, dropout=0.1, activation='relu', norm_first=False, batch_first=True)` |  | torch.nn.TransformerDecoder of layers decoder layers; inputs the target sequence and the memory |
| `/layer/torch/transformer_decoder_layer` | `transformer_decoder_layer` | `(d_model, heads, dim_feedforward=2048, dropout=0.1, activation='relu', norm_first=False, batch_first=True)` |  | torch.nn.TransformerDecoderLayer; inputs the target sequence and the memory |
| `/layer/torch/transformer_encoder` | `transformer_encoder` | `(d_model, heads, layers, dim_feedforward=2048, dropout=0.1, activation='relu', norm_first=False, batch_first=True)` |  | torch.nn.TransformerEncoder of layers encoder layers over (batch, steps, d_model) |
| `/layer/torch/transformer_encoder_layer` | `transformer_encoder_layer` | `(d_model, heads, dim_feedforward=2048, dropout=0.1, activation='relu', norm_first=False, batch_first=True)` |  | torch.nn.TransformerEncoderLayer over (batch, steps, d_model) |
| `/layer/torch/unflatten` |  | `(dim, size)` |  | torch.nn.Unflatten of dim into size; the alias unflatten names kalfa's per sample reshape |
| `/layer/torch/unfold` | `unfold` | `(kernel, dilation=1, padding=0, stride=1)` |  | torch.nn.Unfold, an image into sliding blocks |
| `/layer/torch/upsample` | `upsample` | `(size=None, scale_factor=None, mode='nearest', align_corners=None)` |  | torch.nn.Upsample to size or by scale_factor; mode nearest, linear, bilinear, bicubic or trilinear |
| `/layer/torch/zero_pad` | `zero_pad` | `(padding)` |  | torch.nn.ZeroPad2d; padding is a number or [left, right, top, bottom] |
| `/layer/torch/zero_pad1d` | `zero_pad1d` | `(padding)` |  | torch.nn.ZeroPad1d |
| `/layer/torch/zero_pad3d` | `zero_pad3d` | `(padding)` |  | torch.nn.ZeroPad3d |

### init

| URI | Alias | Signature | Facts | Description |
|---|---|---|---|---|
| `/init/torch/kaiming` | `kaiming` | `(nonlinearity='relu')` |  | Kaiming normal initialization |
| `/init/torch/kaiming_uniform` | `kaiming_uniform` | `(nonlinearity='relu')` |  | Kaiming uniform initialization |
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
| `/criterion/kalfa/weighted_mse` | `weighted_mse` | `(predictions, targets, weights)` | partial: True; refs: weights=data | Mean squared error with a weight per target column; weights is a list in column order or {uri: target_weights, params: {weights: {column: 3.0, 'glob*': 1.5, default: 1.0}}}, named against the target columns the dataset carries |

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
| `/adapter/kalfa/objective` |  | `(objective)` |  | Call an objective with every model of the run and the batch, plus the step, epoch, rng, scaler and losses view its signature names |

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
| `/turn/kalfa/alternating` | `alternating`, `supervised` | `(models, optimizers, emas, counters, composites, effects, loader, params, extra, losses, metrics, losses_keys, metrics_keys, predicts, steps, stream=None, device=None, prep=None, record=None, monitor=None)` | returns: models, optimizers, emas, counters, stream, metrics; bus: device=device, prep=prep, record=record, monitor=monitor; mutates: models, optimizers, emas, counters; extras: amp, grad_clip, accumulate | One turn: every step each optimizer in order minimizes its loss for its steps; a turn is an epoch, or K steps with a stream that lives across turns; losses and metrics are the running means of the pass; every update writes a line to steps.jsonl (the loss, the learning rate and, under grad_clip, the gradient norm of each optimizer) and reaches the monitor |

### trigger

| URI | Alias | Signature | Facts | Description |
|---|---|---|---|---|
| `/trigger/kalfa/after_turn` | `after_turn`, `after_epoch` | `(metrics, turn_index, state, at)` | partial: True; describe: turn ≥ {at} | Fires once the given number of turns has ended, counted across resumes |
| `/trigger/kalfa/metric_above` | `metric_above` | `(metrics, turn_index, state, monitor, value)` | partial: True; describe: {monitor} > {value} | Fires when the monitored value rises above value; a missing value is not seen |
| `/trigger/kalfa/metric_below` | `metric_below` | `(metrics, turn_index, state, monitor, value)` | partial: True; describe: {monitor} < {value} | Fires when the monitored value drops below value; a missing value is not seen |
| `/trigger/kalfa/plateau` | `plateau` | `(metrics, turn_index, state, monitor, patience, mode='min', min_delta=0.0)` | partial: True; describe: {monitor} plateau {patience} | Fires after patience turns without improvement of the monitored value; turns without the value are not counted |
| `/trigger/kalfa/time_budget` | `time_budget` | `(metrics, turn_index, state, minutes)` | partial: True; describe: after {minutes} minutes | Fires once the given number of minutes has passed since the first turn it saw |

### checkpoint

| URI | Alias | Signature | Facts | Description |
|---|---|---|---|---|
| `/checkpoint/kalfa/best` | `best` | `(monitor, mode='min')` | writes: best, last | Write best.pt when the monitored value improves and last.pt every turn |
| `/checkpoint/kalfa/last` | `last` | `()` | writes: last | Write last.pt every turn |
| `/checkpoint/kalfa/snapshot` | `snapshot` | `(every)` | writes: last, snapshot | Write snapshot_<n>.pt every n turns and last.pt every turn |

### generate

| URI | Alias | Signature | Facts | Description |
|---|---|---|---|---|
| `/generate/kalfa/ddpm_sampler` | `ddpm_sampler` | `(models, prep, rng, model, schedule, shape, n=64)` | partial: True; refs: model=model, schedule=schedule | n samples by the reverse diffusion of the noise schedule from pure noise |
| `/generate/kalfa/gan_sampler` | `gan_sampler` | `(models, prep, rng, model, latent, n=64, conditional=False, n_classes=None)` | partial: True; refs: model=model | n samples of a generator from latent noise; conditional samples cycle through n_classes |
| `/generate/kalfa/lm_sampler` | `lm_sampler` | `(models, prep, rng, model, prompt, max_new_tokens=100, temperature=1.0, context=None)` | partial: True; refs: model=model | Autoregressive text from a prompt with the record's tokenizer; temperature scales the logits, the window is context or the model's seq_len; the model's last layer has one logit per vocabulary entry (vocab_size) |

### plot

| URI | Alias | Signature | Facts | Description |
|---|---|---|---|---|
| `/plot/kalfa/architecture` | `architecture` | `(predictions, history, models, record, loaders=None, device=None, predicts=None, losses=None, losses_keys=None, optimizers=None, name=None, figures=None)` | partial: True | kalfa's own drawing of every report model under plots/<name>_<model>.png: one box per graph node with the name from the config, what it is (a torch layer, a lego, another model) and the shapes one batch traced through it, the wires as arrows labelled only where the wire is not the node's name, the boundary wires as boxes, and the losses and the optimizers beside the outputs they read; matplotlib only, any device |
| `/plot/kalfa/architecture_text` | `architecture_text` | `(predictions, history, models, record, name=None, figures=None)` | partial: True | The report models printed as text under plots/<name>.txt, the module repr of each |
| `/plot/kalfa/class_histogram` | `class_histogram` | `(predictions, history, models, record, output=None, target=None, bins=40, name=None, figures=None)` | partial: True; refs: target=field | Histogram of the raw scores of the test set, one series per target class; output names the wire and target the field when the table holds several |
| `/plot/kalfa/confusion_matrix` | `confusion_matrix` | `(predictions, history, models, record, name=None, figures=None)` | partial: True | Confusion matrix of the decoded test predictions against the target labels, counts and row shares in every cell |
| `/plot/kalfa/correlation_heatmap` | `correlation_heatmap` | `(predictions, history, models, record, loaders=None, prep=None, sets=None, method='spearman', columns=None, sample=80000, annotate=False, name=None, figures=None)` | partial: True; needs: train_loader | The rank correlation of every column of a set against every other, features and targets together; it reads the set the definition names (train without one) |
| `/plot/kalfa/data_pipeline` | `data_pipeline` | `(predictions, history, models, record, data_report=None, train_df=None, train_frame=None, prep=None, columns=None, name=None, figures=None)` | partial: True; needs: data_report | The data block as one picture: every stage with its rows and columns, the split, the fitted frame transforms and preprocessors, the features and targets, the loaders; under the fit, before and after histograms of the columns with the longest chains (columns names others) when the source is a table |
| `/plot/kalfa/error_map` | `error_map` | `(predictions, history, models, record, loaders=None, prep=None, sets=None, x=None, y=None, output=None, target=None, statistic='residual', bins=55, min_count=15, name=None, figures=None)` | partial: True; refs: x=column, y=column, target=field | The error of one prediction over a 2d grid of two columns: with statistic residual blue is a prediction below the truth and red above it, with abs the mean absolute error; bins holding fewer than min_count points stay empty |
| `/plot/kalfa/feature_distributions` | `feature_distributions` | `(predictions, history, models, record, loaders=None, prep=None, sets=None, columns=None, log=None, bins=80, limit=24, per_row=4, name=None, figures=None)` | partial: True; needs: train_loader | A histogram per feature column of a set, in the original units; log names the columns to draw on a log10 axis |
| `/plot/kalfa/forecast_samples` | `forecast_samples` | `(predictions, history, models, record, n=6, name=None, figures=None)` | partial: True | n sample windows of the test set: the true horizon against the predicted one |
| `/plot/kalfa/image_grid` | `image_grid` | `(predictions, history, models, record, loaders=None, predicts=None, n=16, set=None, name=None, figures=None)` | partial: True | n outputs of the predicts model on the report set as an image grid |
| `/plot/kalfa/image_pairs` | `image_pairs` | `(predictions, history, models, record, loaders=None, predicts=None, n=8, set=None, name=None, figures=None)` | partial: True | n inputs of the report set next to the predicts model's outputs (reconstructions) |
| `/plot/kalfa/loss_curve` | `loss_curve` | `(predictions, history, models, record, series=None, log=False, x='turn', rates=False, name=None, figures=None)` | partial: True | Every history series over the turns, or the named ones; x: step draws the per update series of steps.jsonl (the loss, the gradient norm of every optimizer) over the steps instead; rates: true adds a panel of the learning rates below, and a series may name lr/<optimizer> |
| `/plot/kalfa/permutation_importance` | `permutation_importance` | `(predictions, history, models, record, loaders=None, prep=None, predicts=None, sets=None, device=None, repeats=3, sample=20000, top=25, output=None, target=None, groups=None, seed=0, name=None, figures=None)` | partial: True; refs: target=field | The drop in R2 when one feature column is shuffled, the largest first; the model runs again for every feature and every repeat, so sample bounds the cost; output names the wire and target the field it is scored against when the model has several |
| `/plot/kalfa/pred_vs_true` | `pred_vs_true` | `(predictions, history, models, record, name=None, columns=4, kind='auto', gridsize=70, figures=None)` | partial: True | Predicted against true values of the test set, one panel per predicted field with its R2, as a hexbin density over many points and a scatter over few; the panel is titled with the field name, plus the output wire when two outputs predict the same field |
| `/plot/kalfa/residuals` | `residuals` | `(predictions, history, models, record, output=None, target=None, bins=20, gridsize=60, name=None, figures=None)` | partial: True; refs: target=field | Three panels of one prediction's residual: the distribution with its bias and sigma, the residual against the truth as a density, and the mean and median error over equal count bins of the target range |
| `/plot/kalfa/samples_gif` | `samples_gif` | `(predictions, history, models, record, name=None, duration=400, figures=None)` | partial: True | The per turn sample grids of samples/turn_*.png as an animation; skipped with a warning when there are none |
| `/plot/kalfa/samples_matrix` | `samples_matrix` | `(predictions, history, models, record, name=None, n=8, figures=None)` | partial: True | A matrix of the per turn samples of samples/turn_*.pt: one row per turn, n columns; skipped with a warning when there are none |
| `/plot/kalfa/target_correlation` | `target_correlation` | `(predictions, history, models, record, loaders=None, prep=None, sets=None, target=None, method='spearman', columns=None, top=25, groups=None, name=None, figures=None)` | partial: True; refs: target=field; needs: train_loader | The rank correlation of every column with the target, the strongest first; groups maps a column to a group name and colours the bars by it |
| `/plot/kalfa/target_vs_features` | `target_vs_features` | `(predictions, history, models, record, loaders=None, prep=None, sets=None, target=None, columns=None, log=None, gridsize=60, bins=60, limit=24, per_row=4, name=None, figures=None)` | partial: True; refs: target=field; needs: train_loader | One panel per feature: the target against it as a hexbin density with the median profile over equal count bins; it reads the set the definition names (train without one) and draws in the original units |
| `/plot/seaborn/kde` | `kde` | `(predictions, history, models, record, loaders=None, prep=None, sets=None, x=None, y=None, hue=None, sample=20000, fill=True, name=None, figures=None)` | partial: True; refs: x=column, y=column, hue=column; needs: train_loader; requires: seaborn | seaborn's kernel density of one column of a set, or of two as contours; skipped with a warning when seaborn is not installed |
| `/plot/seaborn/pairplot` | `pairplot` | `(predictions, history, models, record, loaders=None, prep=None, sets=None, columns=None, hue=None, sample=5000, kind='scatter', diagonal='hist', height=2.2, name=None, figures=None)` | partial: True; needs: train_loader; requires: seaborn | seaborn's pairwise grid of a few columns of a set, hue colouring the points by a column; skipped with a warning when seaborn is not installed |
| `/plot/seaborn/violin` | `violin` | `(predictions, history, models, record, loaders=None, prep=None, sets=None, value=None, group=None, sample=20000, name=None, figures=None)` | partial: True; refs: value=column, group=column; needs: train_loader; requires: seaborn | seaborn's violin of one column of a set, split by a grouping column when one is named; skipped with a warning when seaborn is not installed |
| `/plot/torchmetrics/binary_precision_recall_curve` | `binary_precision_recall_curve` | `(predictions, history, models, record, output=None, target=None, name=None, figures=None)` | partial: True; refs: target=field | Precision recall curve of the raw test scores against the binary target; output names the wire and target the field when the table holds several |
| `/plot/torchmetrics/binary_roc` | `binary_roc` | `(predictions, history, models, record, output=None, target=None, name=None, figures=None)` | partial: True; refs: target=field | ROC curve of the raw test scores against the binary target; output names the wire and target the field when the table holds several |
| `/plot/torchview/architecture` | `torchview` | `(predictions, history, models, record, loaders=None, device=None, name=None, figures=None)` | partial: True; requires: torchview | torchview's drawing of every report model the batch feeds, under plots/<name>_<model>.png; it needs the graphviz dot binary and runs on the device of the run, so a composite keeps its referenced models with it |

### strategy

| URI | Alias | Signature | Facts | Description |
|---|---|---|---|---|
| `/strategy/kalfa/grid` | `grid` | `()` | enumerates: True | Every combination of the space's choices (a range needs steps); deterministic by id |
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

### rng

| URI | Alias | Signature | Facts | Description |
|---|---|---|---|---|
| `/rng/kalfa/derived` | `derived` | `(seed, name, index)` | partial: True | A substream per model by name: the build runs under sha256(seed:name), so inserting or renaming another model changes no weights; None without a seed |
| `/rng/kalfa/global` | `global` | `(seed, name, index)` | partial: True | One global stream: no model touches the RNG, every build draws from the stream the seed started in build order, the way a plain script does |
| `/rng/kalfa/indexed` | `indexed` | `(seed, name, index)` | partial: True | A substream per model by position: the build runs under sha256(seed:index), the rule of the runs made before the rng key, which reproduce with it; None without a seed |

### export

| URI | Alias | Signature | Facts | Description |
|---|---|---|---|---|
| `/export/kalfa/onnx` | `onnx` | `(model, inputs, directory, stem, opset=17)` | requires: onnx | The model exported to <stem>.onnx from one traced batch, the wires as the input and output names, at the opset given |
| `/export/kalfa/pt2` | `pt2` | `(model, inputs, directory, stem)` |  | The model exported with torch.export from one traced batch, the batch dimension left dynamic, and saved as <stem>.pt2, the archive torch.export.load reads back |
| `/export/kalfa/state_dict` | `state_dict` | `(model, inputs, directory, stem)` |  | The model's state_dict as <stem>.pt, the plain torch weights |

### calibrate

| URI | Alias | Signature | Facts | Description |
|---|---|---|---|---|
| `/calibrate/kalfa/threshold` | `threshold` | `(set='valid', quantile=0.95, output=None)` |  | A decision threshold read off a held out set at the end of training: the quantile of the raw output of the predicts model on that set; at predict time flag_<output> marks the rows above it |

### lego

| URI | Alias | Signature | Facts | Description |
|---|---|---|---|---|
| `/lego/kalfa/pixel_features` |  | `(size=4)` |  | A cheap FID feature extractor for demos and tests: images pooled to size by size and flattened; pass it as fid's extractor param |

### data

| URI | Alias | Signature | Facts | Description |
|---|---|---|---|---|
| `/data/kalfa/class_weights` | `class_weights` | `(loader, target=None)` | counts: True | Inverse frequency class weights of the train set's target field, mean one; built once the train loader exists |
| `/data/kalfa/feature_width` | `feature_width` | `(loader)` |  | The width of the feature tensor x, from the fitted plan the train loader carries; for a layer whose shape follows it (layer_norm) |
| `/data/kalfa/target_weights` | `target_weights` | `(loader, weights, target=None)` |  | A weight per target column, from names and globs resolved against the columns of the target field the dataset carries (every target field in order without target), default for the rest; built once the train loader exists |
| `/data/kalfa/vocab_size` | `vocab_size` | `(prep)` |  | The vocabulary size of the fitted tokenizer among the preprocessors; built once prep exists |

## Skeleton steps

These are the skeleton steps `src/kalfa/contract.yaml` calls: the nodes of its blocks, the builder,
the loader, fit, read_prep and figures of its wiring, and the helpers the `sizes` and `header` facts name. They
are not written in a config; the contract places them and the driver fills their params from the config
sections. The list is derived from the URIs the contract mentions, so it cannot drift. The rest of the wiring
stays in the catalog above: the adapters (`/adapter/kalfa/criterion`, `/adapter/kalfa/metric` and
`/adapter/kalfa/objective`, which wrap the losses and metrics entries of a config by kind) and the defaults that
stand in for a config value (`/split/kalfa/random`, `/device/kalfa/cpu`, `/rng/kalfa/derived`).

| URI | Alias | Signature | Facts | Description |
|---|---|---|---|---|
| `/builder/kalfa/module` |  | `(graph, rng=None, seed=None, name=None, index=0, init=None, trainable=True, weights=None, models=None, prep=None, train_loader=None)` | bus: prep=prep, train_loader=train_loader; roles: weights, bias, scale | Build a model graph into an nn.Module in the stream the rng lego derives from the seed, the name and the index, apply init roles, trainable and weights; reference nodes take the models dict; layer params that are kind data components are built from prep and the train loader |
| `/lego/kalfa/apply` |  | `(df, prep, set, keys=None, mask=None)` |  | Apply the fitted chains to one set and type its columns; keys carry the sets a preprocessor is limited to; the rows the mask query selects stay in the frame and are not scored, the loader leaves them out and the plots see them as masked |
| `/lego/kalfa/apply_frames` |  | `(df, frames)` |  | Apply the fitted frame transforms to one set, in the order they were fitted |
| `/lego/kalfa/architecture_note` |  | `(models, loaders, composites=None, predicts=None, losses=None, losses_keys=None, optimizers=None, record=None, device=None)` | returns: None; bus: record=record, device=device | The graph of every report model written to the record as architecture.json: the boxes of the architecture drawing with their column and row, the shapes one batch traced and the wires as arrows; the board draws it |
| `/lego/kalfa/calibrate` |  | `(models, composites, prep, loaders, calibrations, predicts, record=None, device=None)` | returns: calibrations; bus: record=record, device=device | Fit every calibration of the calibrate section on the report models and the sets, in order, and keep them in the record under fitted/calibrate; predict applies them to its table |
| `/lego/kalfa/checkpoint` |  | `(state, policy, metrics=None, record=None)` | returns: None; bus: metrics=metrics, record=record | Write the checkpoint files the policy asks for; nothing without a policy |
| `/lego/kalfa/clone` |  | `(model, decay)` | state: True | An exponential moving average copy of a model with the given decay |
| `/lego/kalfa/const` |  | `(value)` |  | A fresh copy of a constant value |
| `/lego/kalfa/csv_header` |  | `(path, chunk=None, columns=None)` |  | The columns, the dtypes of the first rows and the line count of a CSV file; the listed columns only when the source names them |
| `/lego/kalfa/data_report` |  | `(stages, split, after, fitted, prep, frames, loaders, transforms=None, set_transforms=None, record=None)` | returns: data_report; bus: record=record | The shape of the data at every stage of the data block, read from the bus keys the stages wrote: the rows and columns of the source and after every transform with the transform's call, the sets after the split and after their transforms, the fitted frame transforms and preprocessors, the features and targets, the loaders; written to the record as data.json |
| `/lego/kalfa/evaluate` |  | `(models, emas, composites, counters, effects, loader, set, losses, metrics, losses_keys, metrics_keys, predicts, device=None, prep=None, record=None)` | returns: metrics; bus: device=device, prep=prep, record=record | Losses (model scale) and metrics (original scale, through prep) of one set under no_grad; an empty set gives an empty mapping; record reaches metrics that write files |
| `/lego/kalfa/figures` |  | `(format='png', width=None, height=None, dpi=150, style='kalfa')` |  | The look of every plot of a run: the file format, the size of one panel in inches, the dpi and the style (kalfa, or none for matplotlib's own); the figures section is its params and the built object reaches every plot that names figures |
| `/lego/kalfa/fit` |  | `(df, fields, preprocessors, drop, keys=None, record=None)` | returns: prep; bus: record=record; state: True | Resolve the field globs and fit every preprocessor chain on the train set; keys carry the sets a preprocessor is limited to |
| `/lego/kalfa/fit_frames` |  | `(df, frames, record=None)` | returns: frames; bus: record=record; state: True | Fit the frame transforms on the train set, each on what the ones before it produced, and keep them in the record under fitted/frames |
| `/lego/kalfa/generate` |  | `(models, composites, prep, generate, record=None)` | returns: None; bus: record=record | Run the generate lego with the report models; nothing without a generate section |
| `/lego/kalfa/given_sizes` |  | `(rows, valid=None, test=None, header=None)` |  | The set sizes of a given split: the source rows for train, the header of every given file for the other sets (header reads a path like the source) |
| `/lego/kalfa/history` |  | `(monitor=None, metrics=None, turn_index=None, counters_next=None, optimizers_next=None, rules_next=None, record=None)` | returns: None; bus: monitor=monitor, metrics=metrics, turn_index=turn_index, counters_next=counters_next, optimizers_next=optimizers_next, rules_next=rules_next, record=record | Append the turn's line to history.jsonl, its duration as seconds, and hand it to the monitor |
| `/lego/kalfa/identity` |  | `(value)` | aliases: value | The value itself |
| `/lego/kalfa/image_folder_header` |  | `(path)` |  | The fields, the dtypes, the image count and the classes of an image folder |
| `/lego/kalfa/init_state` |  | `(state, epochs, steps, policy=None, resume=None, device=None)` | returns: epochs_left; bus: resume=resume, device=device; mutates: state | Move the state to the device, load a checkpoint when resuming and restore the checkpoint policy from it, count the turns left |
| `/lego/kalfa/kfold_sizes` |  | `(rows, k, fold, val=None, seed=None)` |  | The set sizes a k fold split produces from rows rows; without rows, which sets it produces |
| `/lego/kalfa/merge` |  | `(parts, prefixes)` |  | Merge the per set metrics under the prefixes of the sets (train/, val/, test/) |
| `/lego/kalfa/pack` |  | `(items)` | aliases: items | A mapping of the given items |
| `/lego/kalfa/parquet_header` |  | `(path, chunk=None, columns=None)` |  | The columns, their arrow types and the row count of a parquet file, from its metadata; the listed columns only when the source names them |
| `/lego/kalfa/predict` |  | `(models, composites, loader, prep, predicts, set, target_map=None, calibrations=None, record=None, device=None)` | returns: predictions; bus: record=record, device=device | Predict a set with the report model, invert the target chain, apply the fitted calibrations, write predictions.parquet for the test set and predictions_<set>.parquet for another |
| `/lego/kalfa/prepared_header` |  | `(path)` |  | The header a prepared directory recorded in its manifest: the columns, the dtypes and the rows |
| `/lego/kalfa/prepared_sizes` |  | `(rows, path)` |  | The set sizes a prepared directory recorded in its manifest |
| `/lego/kalfa/ratio_sizes` |  | `(rows, ratios, seed=None, group=None)` |  | The set sizes a split by ratios produces from rows rows; without rows, which sets it produces |
| `/lego/kalfa/read_frames` |  | `(record)` | returns: frames | The fitted frame transforms of a record, read from fitted/frames |
| `/lego/kalfa/read_prep` |  | `(record)` | returns: prep | The fitted preprocessing plan of a record, read from its preprocessors directory |
| `/lego/kalfa/run_all` |  | `(predictions, history, models, plots, keys=None, predicts=None, bus=None, record=None, figures=None, suffix='')` | returns: None; bus: record=record, figures=figures | Run every plot of the plots table with the predictions, the history and the models; keys carry the definition level keys (inputs, sets, width, height) and the lego of the plot, whose refs type the inputs and whose needs name the bus keys it cannot work without (skipped with a log line when one is missing); bus carries everything else the run has and a plot receives whatever its signature names, plus loaders, predicts, sets, name (with the suffix of the predictions it draws) and figures, the look of the run's plots sized for the definition |
| `/lego/kalfa/save_final` |  | `(models, optimizers, emas, counters, rules, record=None)` | returns: None; bus: record=record | Write final/state.pt with the full state once training ends |
| `/lego/kalfa/select` |  | `(models, emas, which, record=None)` | returns: selected; bus: record=record | The report models: copies loaded from best.pt, or the final state for last |
| `/lego/kalfa/text_lines_header` |  | `(path)` |  | The text field and the line count of a text file |
| `/lego/kalfa/transform_set` |  | `(df, set, transforms)` |  | Apply the transforms that name this set, in order; the frame passes untouched without any |
| `/loader/kalfa/torch` |  | `(data, set, size=None, eval_size=None, shuffle=True, drop_last=False, workers=0, collate=None, balanced=False, buffer=4096)` |  | torch DataLoader over a dataset: size batches shuffled for the train set, eval_size batches in order for the other sets, the whole set as one batch without a size; drop_last auto drops the last train batch only when it would hold one row; balanced puts a class balancing sampler over the single target field; every worker is seeded from the torch seed on its own; a stream dataset shuffles through buffer rows and takes no sampler or workers |
| `/rule/kalfa/effects` |  | `(rules)` | returns: effects | The effects the fired rules left for this turn |
| `/rule/kalfa/open` |  | `(rules)` |  | Open the rule chain of a turn |
| `/rule/kalfa/rule` |  | `(rules, name, when, set, after=None, metrics=None, turn_index=None, sticky=True)` | returns: rules; bus: metrics=metrics, turn_index=turn_index | Evaluate one rule: skipped until its after rule fired in an earlier turn; a sticky rule keeps its effects once fired and is not asked again; with sticky false it is asked every turn, its relative effects (times, plus) apply once per firing and its trigger starts over; later rules win the same key |
| `/rule/kalfa/stop` |  | `(rules, triggers, metrics=None)` | returns: rules, stop; bus: metrics=metrics | Close the chain: the stop triggers are or'ed, their states kept under rules.stop |

## Alias packs

### /alias/kalfa/base

| Alias | URI | Kind |
|---|---|---|
| `filter` | `/transform/kalfa/filter` | transform |
| `derive` | `/transform/kalfa/derive` | transform |
| `rename` | `/transform/kalfa/rename` | transform |
| `astype` | `/transform/kalfa/astype` | transform |
| `drop` | `/transform/kalfa/drop` | transform |
| `sequential` | `/split/kalfa/sequential` | split |
| `given` | `/split/kalfa/given` | split |
| `group_statistic` | `/frame/kalfa/group_statistic` | frame |
| `target_encoding` | `/frame/kalfa/target_encoding` | frame |
| `standard_scaler` | `/pre/sklearn/standard_scaler` | pre |
| `median_std_scaler` | `/pre/kalfa/median_std_scaler` | pre |
| `minmax_scaler` | `/pre/sklearn/minmax_scaler` | pre |
| `cast` | `/pre/kalfa/cast` | pre |
| `abs` | `/pre/kalfa/abs` | pre |
| `log` | `/pre/kalfa/log` | pre |
| `one_hot` | `/pre/kalfa/one_hot` | pre |
| `label_encoder` | `/pre/kalfa/label_encoder` | pre |
| `simple_imputer` | `/pre/kalfa/simple_imputer` | pre |
| `fill` | `/pre/kalfa/fill` | pre |
| `table` | `/feed/kalfa/table` | feed |
| `linear` | `/layer/kalfa/linear` | layer |
| `linear_relu` | `/layer/kalfa/linear_relu` | layer |
| `concat` | `/layer/torch/concat` | layer |
| `flatten` | `/layer/torch/flatten` | layer |
| `relu` | `/layer/torch/relu` | layer |
| `leaky_relu` | `/layer/torch/leaky_relu` | layer |
| `batch_norm` | `/layer/torch/batch_norm` | layer |
| `layer_norm` | `/layer/torch/layer_norm` | layer |
| `group_norm` | `/layer/torch/group_norm` | layer |
| `dropout` | `/layer/torch/dropout` | layer |
| `reparam` | `/layer/kalfa/reparam` | layer |
| `l1_distance` | `/layer/kalfa/l1_distance` | layer |
| `gru` | `/layer/torch/gru` | layer |
| `last_step` | `/layer/torch/last_step` | layer |
| `softsign` | `/layer/torch/softsign` | layer |
| `avgpool3d` | `/layer/torch/avgpool3d` | layer |
| `adaptive_maxpool3d` | `/layer/torch/adaptive_maxpool3d` | layer |
| `constant_pad1d` | `/layer/torch/constant_pad1d` | layer |
| `elu` | `/layer/torch/elu` | layer |
| `pixel_unshuffle` | `/layer/torch/pixel_unshuffle` | layer |
| `zero_pad1d` | `/layer/torch/zero_pad1d` | layer |
| `alpha_dropout` | `/layer/torch/alpha_dropout` | layer |
| `lppool1d` | `/layer/torch/lppool1d` | layer |
| `rrelu` | `/layer/torch/rrelu` | layer |
| `prelu` | `/layer/torch/prelu` | layer |
| `cosine_similarity` | `/layer/torch/cosine_similarity` | layer |
| `replication_pad` | `/layer/torch/replication_pad` | layer |
| `embedding_bag` | `/layer/torch/embedding_bag` | layer |
| `max_unpool3d` | `/layer/torch/max_unpool3d` | layer |
| `hardshrink` | `/layer/torch/hardshrink` | layer |
| `hardswish` | `/layer/torch/hardswish` | layer |
| `rnn` | `/layer/torch/rnn` | layer |
| `conv_transpose2d` | `/layer/torch/conv_transpose2d` | layer |
| `dropout2d` | `/layer/torch/dropout2d` | layer |
| `dropout3d` | `/layer/torch/dropout3d` | layer |
| `avgpool` | `/layer/torch/avgpool` | layer |
| `lppool` | `/layer/torch/lppool` | layer |
| `threshold_layer` | `/layer/torch/threshold` | layer |
| `conv_transpose3d` | `/layer/torch/conv_transpose3d` | layer |
| `tanh_layer` | `/layer/torch/tanh` | layer |
| `identity` | `/layer/torch/identity` | layer |
| `mlp` | `/layer/kalfa/mlp` | layer |
| `feature_alpha_dropout` | `/layer/torch/feature_alpha_dropout` | layer |
| `adaptive_avgpool3d` | `/layer/torch/adaptive_avgpool3d` | layer |
| `softplus` | `/layer/torch/softplus` | layer |
| `softmax` | `/layer/torch/softmax` | layer |
| `silu` | `/layer/torch/silu` | layer |
| `softmin` | `/layer/torch/softmin` | layer |
| `lstm_cell` | `/layer/torch/lstm_cell` | layer |
| `constant_pad3d` | `/layer/torch/constant_pad3d` | layer |
| `transformer_decoder_layer` | `/layer/torch/transformer_decoder_layer` | layer |
| `relu6` | `/layer/torch/relu6` | layer |
| `reflection_pad1d` | `/layer/torch/reflection_pad1d` | layer |
| `maxpool1d` | `/layer/torch/maxpool1d` | layer |
| `softshrink` | `/layer/torch/softshrink` | layer |
| `conv_transpose1d` | `/layer/torch/conv_transpose1d` | layer |
| `conv1d` | `/layer/torch/conv1d` | layer |
| `hardsigmoid` | `/layer/torch/hardsigmoid` | layer |
| `reflection_pad` | `/layer/torch/reflection_pad` | layer |
| `local_response_norm` | `/layer/torch/local_response_norm` | layer |
| `conv3d` | `/layer/torch/conv3d` | layer |
| `lstm` | `/layer/torch/lstm` | layer |
| `max_unpool` | `/layer/torch/max_unpool` | layer |
| `constant_pad` | `/layer/torch/constant_pad` | layer |
| `mish` | `/layer/torch/mish` | layer |
| `rnn_cell` | `/layer/torch/rnn_cell` | layer |
| `zero_pad` | `/layer/torch/zero_pad` | layer |
| `pairwise_distance` | `/layer/torch/pairwise_distance` | layer |
| `celu` | `/layer/torch/celu` | layer |
| `log_sigmoid` | `/layer/torch/log_sigmoid` | layer |
| `fold` | `/layer/torch/fold` | layer |
| `pixel_shuffle` | `/layer/torch/pixel_shuffle` | layer |
| `log_softmax` | `/layer/torch/log_softmax` | layer |
| `transformer_encoder_layer` | `/layer/torch/transformer_encoder_layer` | layer |
| `fractional_maxpool` | `/layer/torch/fractional_maxpool` | layer |
| `adaptive_maxpool` | `/layer/torch/adaptive_maxpool` | layer |
| `adaptive_avgpool1d` | `/layer/torch/adaptive_avgpool1d` | layer |
| `glu` | `/layer/torch/glu` | layer |
| `instance_norm` | `/layer/torch/instance_norm` | layer |
| `circular_pad3d` | `/layer/torch/circular_pad3d` | layer |
| `circular_pad` | `/layer/torch/circular_pad` | layer |
| `upsample` | `/layer/torch/upsample` | layer |
| `circular_pad1d` | `/layer/torch/circular_pad1d` | layer |
| `adaptive_avgpool` | `/layer/torch/adaptive_avgpool` | layer |
| `selu` | `/layer/torch/selu` | layer |
| `transformer_encoder` | `/layer/torch/transformer_encoder` | layer |
| `channel_shuffle` | `/layer/torch/channel_shuffle` | layer |
| `softmax2d` | `/layer/torch/softmax2d` | layer |
| `reflection_pad3d` | `/layer/torch/reflection_pad3d` | layer |
| `transformer` | `/layer/torch/transformer` | layer |
| `zero_pad3d` | `/layer/torch/zero_pad3d` | layer |
| `rms_norm` | `/layer/torch/rms_norm` | layer |
| `max_unpool1d` | `/layer/torch/max_unpool1d` | layer |
| `sigmoid` | `/layer/torch/sigmoid` | layer |
| `multihead_attention` | `/layer/torch/multihead_attention` | layer |
| `adaptive_maxpool1d` | `/layer/torch/adaptive_maxpool1d` | layer |
| `maxpool3d` | `/layer/torch/maxpool3d` | layer |
| `lppool3d` | `/layer/torch/lppool3d` | layer |
| `fractional_maxpool3d` | `/layer/torch/fractional_maxpool3d` | layer |
| `hardtanh` | `/layer/torch/hardtanh` | layer |
| `dropout1d` | `/layer/torch/dropout1d` | layer |
| `bilinear` | `/layer/torch/bilinear` | layer |
| `tanhshrink` | `/layer/torch/tanhshrink` | layer |
| `gelu` | `/layer/torch/gelu` | layer |
| `gru_cell` | `/layer/torch/gru_cell` | layer |
| `replication_pad1d` | `/layer/torch/replication_pad1d` | layer |
| `avgpool1d` | `/layer/torch/avgpool1d` | layer |
| `unfold` | `/layer/torch/unfold` | layer |
| `replication_pad3d` | `/layer/torch/replication_pad3d` | layer |
| `transformer_decoder` | `/layer/torch/transformer_decoder` | layer |
| `normal` | `/init/torch/normal` | init |
| `xavier` | `/init/torch/xavier` | init |
| `kaiming` | `/init/torch/kaiming` | init |
| `kaiming_uniform` | `/init/torch/kaiming_uniform` | init |
| `zeros` | `/init/torch/zeros` | init |
| `mse` | `/criterion/kalfa/mse` | criterion |
| `weighted_mse` | `/criterion/kalfa/weighted_mse` | criterion |
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
| `binary_roc` | `/plot/torchmetrics/binary_roc` | plot |
| `binary_precision_recall_curve` | `/plot/torchmetrics/binary_precision_recall_curve` | plot |
| `confusion_matrix` | `/plot/kalfa/confusion_matrix` | plot |
| `forecast_samples` | `/plot/kalfa/forecast_samples` | plot |
| `architecture` | `/plot/kalfa/architecture` | plot |
| `data_pipeline` | `/plot/kalfa/data_pipeline` | plot |
| `architecture_text` | `/plot/kalfa/architecture_text` | plot |
| `torchview` | `/plot/torchview/architecture` | plot |
| `weighted_sum` | `/objective/kalfa/weighted_sum` | objective |
| `grid` | `/strategy/kalfa/grid` | strategy |
| `random` | `/strategy/kalfa/random` | strategy |
| `sobol` | `/strategy/kalfa/sobol` | strategy |
| `optuna` | `/strategy/kalfa/optuna` | strategy |
| `auto` | `/device/kalfa/auto` | device |
| `cpu` | `/device/kalfa/cpu` | device |
| `cuda` | `/device/kalfa/cuda` | device |
| `mps` | `/device/kalfa/mps` | device |
| `threshold` | `/calibrate/kalfa/threshold` | calibrate |
| `onnx` | `/export/kalfa/onnx` | export |
| `pt2` | `/export/kalfa/pt2` | export |
| `state_dict` | `/export/kalfa/state_dict` | export |
| `derived` | `/rng/kalfa/derived` | rng |
| `indexed` | `/rng/kalfa/indexed` | rng |
| `global` | `/rng/kalfa/global` | rng |

### /alias/kalfa/lazy

| Alias | URI | Kind |
|---|---|---|
| `filter` | `/transform/kalfa/filter` | transform |
| `derive` | `/transform/kalfa/derive` | transform |
| `rename` | `/transform/kalfa/rename` | transform |
| `astype` | `/transform/kalfa/astype` | transform |
| `drop` | `/transform/kalfa/drop` | transform |
| `sequential` | `/split/kalfa/sequential` | split |
| `given` | `/split/kalfa/given` | split |
| `group_statistic` | `/frame/kalfa/group_statistic` | frame |
| `target_encoding` | `/frame/kalfa/target_encoding` | frame |
| `standard_scaler` | `/pre/sklearn/standard_scaler` | pre |
| `median_std_scaler` | `/pre/kalfa/median_std_scaler` | pre |
| `minmax_scaler` | `/pre/sklearn/minmax_scaler` | pre |
| `cast` | `/pre/kalfa/cast` | pre |
| `abs` | `/pre/kalfa/abs` | pre |
| `log` | `/pre/kalfa/log` | pre |
| `one_hot` | `/pre/kalfa/one_hot` | pre |
| `label_encoder` | `/pre/kalfa/label_encoder` | pre |
| `simple_imputer` | `/pre/kalfa/simple_imputer` | pre |
| `fill` | `/pre/kalfa/fill` | pre |
| `table` | `/feed/kalfa/table` | feed |
| `linear` | `/layer/kalfa/linear` | layer |
| `linear_relu` | `/layer/kalfa/linear_relu` | layer |
| `concat` | `/layer/torch/concat` | layer |
| `flatten` | `/layer/torch/flatten` | layer |
| `relu` | `/layer/torch/relu` | layer |
| `leaky_relu` | `/layer/torch/leaky_relu` | layer |
| `batch_norm` | `/layer/torch/batch_norm` | layer |
| `layer_norm` | `/layer/torch/layer_norm` | layer |
| `group_norm` | `/layer/torch/group_norm` | layer |
| `dropout` | `/layer/torch/dropout` | layer |
| `reparam` | `/layer/kalfa/reparam` | layer |
| `l1_distance` | `/layer/kalfa/l1_distance` | layer |
| `gru` | `/layer/torch/gru` | layer |
| `last_step` | `/layer/torch/last_step` | layer |
| `softsign` | `/layer/torch/softsign` | layer |
| `avgpool3d` | `/layer/torch/avgpool3d` | layer |
| `adaptive_maxpool3d` | `/layer/torch/adaptive_maxpool3d` | layer |
| `constant_pad1d` | `/layer/torch/constant_pad1d` | layer |
| `elu` | `/layer/torch/elu` | layer |
| `pixel_unshuffle` | `/layer/torch/pixel_unshuffle` | layer |
| `zero_pad1d` | `/layer/torch/zero_pad1d` | layer |
| `alpha_dropout` | `/layer/torch/alpha_dropout` | layer |
| `lppool1d` | `/layer/torch/lppool1d` | layer |
| `rrelu` | `/layer/torch/rrelu` | layer |
| `prelu` | `/layer/torch/prelu` | layer |
| `cosine_similarity` | `/layer/torch/cosine_similarity` | layer |
| `replication_pad` | `/layer/torch/replication_pad` | layer |
| `embedding_bag` | `/layer/torch/embedding_bag` | layer |
| `max_unpool3d` | `/layer/torch/max_unpool3d` | layer |
| `hardshrink` | `/layer/torch/hardshrink` | layer |
| `hardswish` | `/layer/torch/hardswish` | layer |
| `rnn` | `/layer/torch/rnn` | layer |
| `conv_transpose2d` | `/layer/torch/conv_transpose2d` | layer |
| `dropout2d` | `/layer/torch/dropout2d` | layer |
| `dropout3d` | `/layer/torch/dropout3d` | layer |
| `avgpool` | `/layer/torch/avgpool` | layer |
| `lppool` | `/layer/torch/lppool` | layer |
| `threshold_layer` | `/layer/torch/threshold` | layer |
| `conv_transpose3d` | `/layer/torch/conv_transpose3d` | layer |
| `tanh_layer` | `/layer/torch/tanh` | layer |
| `identity` | `/layer/torch/identity` | layer |
| `mlp` | `/layer/kalfa/mlp` | layer |
| `feature_alpha_dropout` | `/layer/torch/feature_alpha_dropout` | layer |
| `adaptive_avgpool3d` | `/layer/torch/adaptive_avgpool3d` | layer |
| `softplus` | `/layer/torch/softplus` | layer |
| `softmax` | `/layer/torch/softmax` | layer |
| `silu` | `/layer/torch/silu` | layer |
| `softmin` | `/layer/torch/softmin` | layer |
| `lstm_cell` | `/layer/torch/lstm_cell` | layer |
| `constant_pad3d` | `/layer/torch/constant_pad3d` | layer |
| `transformer_decoder_layer` | `/layer/torch/transformer_decoder_layer` | layer |
| `relu6` | `/layer/torch/relu6` | layer |
| `reflection_pad1d` | `/layer/torch/reflection_pad1d` | layer |
| `maxpool1d` | `/layer/torch/maxpool1d` | layer |
| `softshrink` | `/layer/torch/softshrink` | layer |
| `conv_transpose1d` | `/layer/torch/conv_transpose1d` | layer |
| `conv1d` | `/layer/torch/conv1d` | layer |
| `hardsigmoid` | `/layer/torch/hardsigmoid` | layer |
| `reflection_pad` | `/layer/torch/reflection_pad` | layer |
| `local_response_norm` | `/layer/torch/local_response_norm` | layer |
| `conv3d` | `/layer/torch/conv3d` | layer |
| `lstm` | `/layer/torch/lstm` | layer |
| `max_unpool` | `/layer/torch/max_unpool` | layer |
| `constant_pad` | `/layer/torch/constant_pad` | layer |
| `mish` | `/layer/torch/mish` | layer |
| `rnn_cell` | `/layer/torch/rnn_cell` | layer |
| `zero_pad` | `/layer/torch/zero_pad` | layer |
| `pairwise_distance` | `/layer/torch/pairwise_distance` | layer |
| `celu` | `/layer/torch/celu` | layer |
| `log_sigmoid` | `/layer/torch/log_sigmoid` | layer |
| `fold` | `/layer/torch/fold` | layer |
| `pixel_shuffle` | `/layer/torch/pixel_shuffle` | layer |
| `log_softmax` | `/layer/torch/log_softmax` | layer |
| `transformer_encoder_layer` | `/layer/torch/transformer_encoder_layer` | layer |
| `fractional_maxpool` | `/layer/torch/fractional_maxpool` | layer |
| `adaptive_maxpool` | `/layer/torch/adaptive_maxpool` | layer |
| `adaptive_avgpool1d` | `/layer/torch/adaptive_avgpool1d` | layer |
| `glu` | `/layer/torch/glu` | layer |
| `instance_norm` | `/layer/torch/instance_norm` | layer |
| `circular_pad3d` | `/layer/torch/circular_pad3d` | layer |
| `circular_pad` | `/layer/torch/circular_pad` | layer |
| `upsample` | `/layer/torch/upsample` | layer |
| `circular_pad1d` | `/layer/torch/circular_pad1d` | layer |
| `adaptive_avgpool` | `/layer/torch/adaptive_avgpool` | layer |
| `selu` | `/layer/torch/selu` | layer |
| `transformer_encoder` | `/layer/torch/transformer_encoder` | layer |
| `channel_shuffle` | `/layer/torch/channel_shuffle` | layer |
| `softmax2d` | `/layer/torch/softmax2d` | layer |
| `reflection_pad3d` | `/layer/torch/reflection_pad3d` | layer |
| `transformer` | `/layer/torch/transformer` | layer |
| `zero_pad3d` | `/layer/torch/zero_pad3d` | layer |
| `rms_norm` | `/layer/torch/rms_norm` | layer |
| `max_unpool1d` | `/layer/torch/max_unpool1d` | layer |
| `sigmoid` | `/layer/torch/sigmoid` | layer |
| `multihead_attention` | `/layer/torch/multihead_attention` | layer |
| `adaptive_maxpool1d` | `/layer/torch/adaptive_maxpool1d` | layer |
| `maxpool3d` | `/layer/torch/maxpool3d` | layer |
| `lppool3d` | `/layer/torch/lppool3d` | layer |
| `fractional_maxpool3d` | `/layer/torch/fractional_maxpool3d` | layer |
| `hardtanh` | `/layer/torch/hardtanh` | layer |
| `dropout1d` | `/layer/torch/dropout1d` | layer |
| `bilinear` | `/layer/torch/bilinear` | layer |
| `tanhshrink` | `/layer/torch/tanhshrink` | layer |
| `gelu` | `/layer/torch/gelu` | layer |
| `gru_cell` | `/layer/torch/gru_cell` | layer |
| `replication_pad1d` | `/layer/torch/replication_pad1d` | layer |
| `avgpool1d` | `/layer/torch/avgpool1d` | layer |
| `unfold` | `/layer/torch/unfold` | layer |
| `replication_pad3d` | `/layer/torch/replication_pad3d` | layer |
| `transformer_decoder` | `/layer/torch/transformer_decoder` | layer |
| `normal` | `/init/torch/normal` | init |
| `xavier` | `/init/torch/xavier` | init |
| `kaiming` | `/init/torch/kaiming` | init |
| `kaiming_uniform` | `/init/torch/kaiming_uniform` | init |
| `zeros` | `/init/torch/zeros` | init |
| `mse` | `/criterion/kalfa/mse` | criterion |
| `weighted_mse` | `/criterion/kalfa/weighted_mse` | criterion |
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
| `binary_roc` | `/plot/torchmetrics/binary_roc` | plot |
| `binary_precision_recall_curve` | `/plot/torchmetrics/binary_precision_recall_curve` | plot |
| `confusion_matrix` | `/plot/kalfa/confusion_matrix` | plot |
| `forecast_samples` | `/plot/kalfa/forecast_samples` | plot |
| `architecture` | `/plot/kalfa/architecture` | plot |
| `data_pipeline` | `/plot/kalfa/data_pipeline` | plot |
| `architecture_text` | `/plot/kalfa/architecture_text` | plot |
| `torchview` | `/plot/torchview/architecture` | plot |
| `weighted_sum` | `/objective/kalfa/weighted_sum` | objective |
| `grid` | `/strategy/kalfa/grid` | strategy |
| `random` | `/strategy/kalfa/random` | strategy |
| `sobol` | `/strategy/kalfa/sobol` | strategy |
| `optuna` | `/strategy/kalfa/optuna` | strategy |
| `auto` | `/device/kalfa/auto` | device |
| `cpu` | `/device/kalfa/cpu` | device |
| `cuda` | `/device/kalfa/cuda` | device |
| `mps` | `/device/kalfa/mps` | device |
| `threshold` | `/calibrate/kalfa/threshold` | calibrate |
| `onnx` | `/export/kalfa/onnx` | export |
| `pt2` | `/export/kalfa/pt2` | export |
| `state_dict` | `/export/kalfa/state_dict` | export |
| `derived` | `/rng/kalfa/derived` | rng |
| `indexed` | `/rng/kalfa/indexed` | rng |
| `global` | `/rng/kalfa/global` | rng |
| `parquet` | `/source/kalfa/parquet_stream` | source |
| `csv` | `/source/kalfa/csv_stream` | source |

### /alias/kalfa/tabular

| Alias | URI | Kind |
|---|---|---|
| `filter` | `/transform/kalfa/filter` | transform |
| `derive` | `/transform/kalfa/derive` | transform |
| `rename` | `/transform/kalfa/rename` | transform |
| `astype` | `/transform/kalfa/astype` | transform |
| `drop` | `/transform/kalfa/drop` | transform |
| `sequential` | `/split/kalfa/sequential` | split |
| `given` | `/split/kalfa/given` | split |
| `group_statistic` | `/frame/kalfa/group_statistic` | frame |
| `target_encoding` | `/frame/kalfa/target_encoding` | frame |
| `standard_scaler` | `/pre/sklearn/standard_scaler` | pre |
| `median_std_scaler` | `/pre/kalfa/median_std_scaler` | pre |
| `minmax_scaler` | `/pre/sklearn/minmax_scaler` | pre |
| `cast` | `/pre/kalfa/cast` | pre |
| `abs` | `/pre/kalfa/abs` | pre |
| `log` | `/pre/kalfa/log` | pre |
| `one_hot` | `/pre/kalfa/one_hot` | pre |
| `label_encoder` | `/pre/kalfa/label_encoder` | pre |
| `simple_imputer` | `/pre/kalfa/simple_imputer` | pre |
| `fill` | `/pre/kalfa/fill` | pre |
| `table` | `/feed/kalfa/table` | feed |
| `linear` | `/layer/kalfa/linear` | layer |
| `linear_relu` | `/layer/kalfa/linear_relu` | layer |
| `concat` | `/layer/torch/concat` | layer |
| `flatten` | `/layer/torch/flatten` | layer |
| `relu` | `/layer/torch/relu` | layer |
| `leaky_relu` | `/layer/torch/leaky_relu` | layer |
| `batch_norm` | `/layer/torch/batch_norm` | layer |
| `layer_norm` | `/layer/torch/layer_norm` | layer |
| `group_norm` | `/layer/torch/group_norm` | layer |
| `dropout` | `/layer/torch/dropout` | layer |
| `reparam` | `/layer/kalfa/reparam` | layer |
| `l1_distance` | `/layer/kalfa/l1_distance` | layer |
| `gru` | `/layer/torch/gru` | layer |
| `last_step` | `/layer/torch/last_step` | layer |
| `softsign` | `/layer/torch/softsign` | layer |
| `avgpool3d` | `/layer/torch/avgpool3d` | layer |
| `adaptive_maxpool3d` | `/layer/torch/adaptive_maxpool3d` | layer |
| `constant_pad1d` | `/layer/torch/constant_pad1d` | layer |
| `elu` | `/layer/torch/elu` | layer |
| `pixel_unshuffle` | `/layer/torch/pixel_unshuffle` | layer |
| `zero_pad1d` | `/layer/torch/zero_pad1d` | layer |
| `alpha_dropout` | `/layer/torch/alpha_dropout` | layer |
| `lppool1d` | `/layer/torch/lppool1d` | layer |
| `rrelu` | `/layer/torch/rrelu` | layer |
| `prelu` | `/layer/torch/prelu` | layer |
| `cosine_similarity` | `/layer/torch/cosine_similarity` | layer |
| `replication_pad` | `/layer/torch/replication_pad` | layer |
| `embedding_bag` | `/layer/torch/embedding_bag` | layer |
| `max_unpool3d` | `/layer/torch/max_unpool3d` | layer |
| `hardshrink` | `/layer/torch/hardshrink` | layer |
| `hardswish` | `/layer/torch/hardswish` | layer |
| `rnn` | `/layer/torch/rnn` | layer |
| `conv_transpose2d` | `/layer/torch/conv_transpose2d` | layer |
| `dropout2d` | `/layer/torch/dropout2d` | layer |
| `dropout3d` | `/layer/torch/dropout3d` | layer |
| `avgpool` | `/layer/torch/avgpool` | layer |
| `lppool` | `/layer/torch/lppool` | layer |
| `threshold_layer` | `/layer/torch/threshold` | layer |
| `conv_transpose3d` | `/layer/torch/conv_transpose3d` | layer |
| `tanh_layer` | `/layer/torch/tanh` | layer |
| `identity` | `/layer/torch/identity` | layer |
| `mlp` | `/layer/kalfa/mlp` | layer |
| `feature_alpha_dropout` | `/layer/torch/feature_alpha_dropout` | layer |
| `adaptive_avgpool3d` | `/layer/torch/adaptive_avgpool3d` | layer |
| `softplus` | `/layer/torch/softplus` | layer |
| `softmax` | `/layer/torch/softmax` | layer |
| `silu` | `/layer/torch/silu` | layer |
| `softmin` | `/layer/torch/softmin` | layer |
| `lstm_cell` | `/layer/torch/lstm_cell` | layer |
| `constant_pad3d` | `/layer/torch/constant_pad3d` | layer |
| `transformer_decoder_layer` | `/layer/torch/transformer_decoder_layer` | layer |
| `relu6` | `/layer/torch/relu6` | layer |
| `reflection_pad1d` | `/layer/torch/reflection_pad1d` | layer |
| `maxpool1d` | `/layer/torch/maxpool1d` | layer |
| `softshrink` | `/layer/torch/softshrink` | layer |
| `conv_transpose1d` | `/layer/torch/conv_transpose1d` | layer |
| `conv1d` | `/layer/torch/conv1d` | layer |
| `hardsigmoid` | `/layer/torch/hardsigmoid` | layer |
| `reflection_pad` | `/layer/torch/reflection_pad` | layer |
| `local_response_norm` | `/layer/torch/local_response_norm` | layer |
| `conv3d` | `/layer/torch/conv3d` | layer |
| `lstm` | `/layer/torch/lstm` | layer |
| `max_unpool` | `/layer/torch/max_unpool` | layer |
| `constant_pad` | `/layer/torch/constant_pad` | layer |
| `mish` | `/layer/torch/mish` | layer |
| `rnn_cell` | `/layer/torch/rnn_cell` | layer |
| `zero_pad` | `/layer/torch/zero_pad` | layer |
| `pairwise_distance` | `/layer/torch/pairwise_distance` | layer |
| `celu` | `/layer/torch/celu` | layer |
| `log_sigmoid` | `/layer/torch/log_sigmoid` | layer |
| `fold` | `/layer/torch/fold` | layer |
| `pixel_shuffle` | `/layer/torch/pixel_shuffle` | layer |
| `log_softmax` | `/layer/torch/log_softmax` | layer |
| `transformer_encoder_layer` | `/layer/torch/transformer_encoder_layer` | layer |
| `fractional_maxpool` | `/layer/torch/fractional_maxpool` | layer |
| `adaptive_maxpool` | `/layer/torch/adaptive_maxpool` | layer |
| `adaptive_avgpool1d` | `/layer/torch/adaptive_avgpool1d` | layer |
| `glu` | `/layer/torch/glu` | layer |
| `instance_norm` | `/layer/torch/instance_norm` | layer |
| `circular_pad3d` | `/layer/torch/circular_pad3d` | layer |
| `circular_pad` | `/layer/torch/circular_pad` | layer |
| `upsample` | `/layer/torch/upsample` | layer |
| `circular_pad1d` | `/layer/torch/circular_pad1d` | layer |
| `adaptive_avgpool` | `/layer/torch/adaptive_avgpool` | layer |
| `selu` | `/layer/torch/selu` | layer |
| `transformer_encoder` | `/layer/torch/transformer_encoder` | layer |
| `channel_shuffle` | `/layer/torch/channel_shuffle` | layer |
| `softmax2d` | `/layer/torch/softmax2d` | layer |
| `reflection_pad3d` | `/layer/torch/reflection_pad3d` | layer |
| `transformer` | `/layer/torch/transformer` | layer |
| `zero_pad3d` | `/layer/torch/zero_pad3d` | layer |
| `rms_norm` | `/layer/torch/rms_norm` | layer |
| `max_unpool1d` | `/layer/torch/max_unpool1d` | layer |
| `sigmoid` | `/layer/torch/sigmoid` | layer |
| `multihead_attention` | `/layer/torch/multihead_attention` | layer |
| `adaptive_maxpool1d` | `/layer/torch/adaptive_maxpool1d` | layer |
| `maxpool3d` | `/layer/torch/maxpool3d` | layer |
| `lppool3d` | `/layer/torch/lppool3d` | layer |
| `fractional_maxpool3d` | `/layer/torch/fractional_maxpool3d` | layer |
| `hardtanh` | `/layer/torch/hardtanh` | layer |
| `dropout1d` | `/layer/torch/dropout1d` | layer |
| `bilinear` | `/layer/torch/bilinear` | layer |
| `tanhshrink` | `/layer/torch/tanhshrink` | layer |
| `gelu` | `/layer/torch/gelu` | layer |
| `gru_cell` | `/layer/torch/gru_cell` | layer |
| `replication_pad1d` | `/layer/torch/replication_pad1d` | layer |
| `avgpool1d` | `/layer/torch/avgpool1d` | layer |
| `unfold` | `/layer/torch/unfold` | layer |
| `replication_pad3d` | `/layer/torch/replication_pad3d` | layer |
| `transformer_decoder` | `/layer/torch/transformer_decoder` | layer |
| `normal` | `/init/torch/normal` | init |
| `xavier` | `/init/torch/xavier` | init |
| `kaiming` | `/init/torch/kaiming` | init |
| `kaiming_uniform` | `/init/torch/kaiming_uniform` | init |
| `zeros` | `/init/torch/zeros` | init |
| `mse` | `/criterion/kalfa/mse` | criterion |
| `weighted_mse` | `/criterion/kalfa/weighted_mse` | criterion |
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
| `binary_roc` | `/plot/torchmetrics/binary_roc` | plot |
| `binary_precision_recall_curve` | `/plot/torchmetrics/binary_precision_recall_curve` | plot |
| `confusion_matrix` | `/plot/kalfa/confusion_matrix` | plot |
| `forecast_samples` | `/plot/kalfa/forecast_samples` | plot |
| `architecture` | `/plot/kalfa/architecture` | plot |
| `data_pipeline` | `/plot/kalfa/data_pipeline` | plot |
| `architecture_text` | `/plot/kalfa/architecture_text` | plot |
| `torchview` | `/plot/torchview/architecture` | plot |
| `weighted_sum` | `/objective/kalfa/weighted_sum` | objective |
| `grid` | `/strategy/kalfa/grid` | strategy |
| `random` | `/strategy/kalfa/random` | strategy |
| `sobol` | `/strategy/kalfa/sobol` | strategy |
| `optuna` | `/strategy/kalfa/optuna` | strategy |
| `auto` | `/device/kalfa/auto` | device |
| `cpu` | `/device/kalfa/cpu` | device |
| `cuda` | `/device/kalfa/cuda` | device |
| `mps` | `/device/kalfa/mps` | device |
| `threshold` | `/calibrate/kalfa/threshold` | calibrate |
| `onnx` | `/export/kalfa/onnx` | export |
| `pt2` | `/export/kalfa/pt2` | export |
| `state_dict` | `/export/kalfa/state_dict` | export |
| `derived` | `/rng/kalfa/derived` | rng |
| `indexed` | `/rng/kalfa/indexed` | rng |
| `global` | `/rng/kalfa/global` | rng |
| `parquet` | `/source/kalfa/parquet` | source |
| `csv` | `/source/kalfa/csv` | source |
| `random_split` | `/split/kalfa/random` | split |
| `kfold` | `/split/kalfa/kfold` | split |
| `max_abs_scaler` | `/pre/sklearn/max_abs_scaler` | pre |
| `robust_scaler` | `/pre/sklearn/robust_scaler` | pre |
| `quantile_transformer` | `/pre/sklearn/quantile_transformer` | pre |
| `power_transformer` | `/pre/sklearn/power_transformer` | pre |
| `asinh` | `/pre/kalfa/asinh` | pre |
| `sinh` | `/pre/kalfa/sinh` | pre |
| `tanh` | `/pre/kalfa/tanh` | pre |
| `atanh` | `/pre/kalfa/atanh` | pre |
| `kbins_discretizer` | `/pre/sklearn/kbins_discretizer` | pre |
| `spline_transformer` | `/pre/sklearn/spline_transformer` | pre |
| `window` | `/feed/kalfa/window` | feed |
| `polynomial` | `/layer/kalfa/polynomial` | layer |
| `l2_normalize` | `/layer/kalfa/l2_normalize` | layer |
| `target_vs_features` | `/plot/kalfa/target_vs_features` | plot |
| `correlation_heatmap` | `/plot/kalfa/correlation_heatmap` | plot |
| `residuals` | `/plot/kalfa/residuals` | plot |
| `error_map` | `/plot/kalfa/error_map` | plot |
| `permutation_importance` | `/plot/kalfa/permutation_importance` | plot |
| `feature_distributions` | `/plot/kalfa/feature_distributions` | plot |
| `target_correlation` | `/plot/kalfa/target_correlation` | plot |
| `pairplot` | `/plot/seaborn/pairplot` | plot |
| `violin` | `/plot/seaborn/violin` | plot |
| `kde` | `/plot/seaborn/kde` | plot |
| `class_weights` | `/data/kalfa/class_weights` | data |
| `target_weights` | `/data/kalfa/target_weights` | data |
| `feature_width` | `/data/kalfa/feature_width` | data |

### /alias/kalfa/text

| Alias | URI | Kind |
|---|---|---|
| `filter` | `/transform/kalfa/filter` | transform |
| `derive` | `/transform/kalfa/derive` | transform |
| `rename` | `/transform/kalfa/rename` | transform |
| `astype` | `/transform/kalfa/astype` | transform |
| `drop` | `/transform/kalfa/drop` | transform |
| `sequential` | `/split/kalfa/sequential` | split |
| `given` | `/split/kalfa/given` | split |
| `group_statistic` | `/frame/kalfa/group_statistic` | frame |
| `target_encoding` | `/frame/kalfa/target_encoding` | frame |
| `standard_scaler` | `/pre/sklearn/standard_scaler` | pre |
| `median_std_scaler` | `/pre/kalfa/median_std_scaler` | pre |
| `minmax_scaler` | `/pre/sklearn/minmax_scaler` | pre |
| `cast` | `/pre/kalfa/cast` | pre |
| `abs` | `/pre/kalfa/abs` | pre |
| `log` | `/pre/kalfa/log` | pre |
| `one_hot` | `/pre/kalfa/one_hot` | pre |
| `label_encoder` | `/pre/kalfa/label_encoder` | pre |
| `simple_imputer` | `/pre/kalfa/simple_imputer` | pre |
| `fill` | `/pre/kalfa/fill` | pre |
| `table` | `/feed/kalfa/table` | feed |
| `linear` | `/layer/kalfa/linear` | layer |
| `linear_relu` | `/layer/kalfa/linear_relu` | layer |
| `concat` | `/layer/torch/concat` | layer |
| `flatten` | `/layer/torch/flatten` | layer |
| `relu` | `/layer/torch/relu` | layer |
| `leaky_relu` | `/layer/torch/leaky_relu` | layer |
| `batch_norm` | `/layer/torch/batch_norm` | layer |
| `layer_norm` | `/layer/torch/layer_norm` | layer |
| `group_norm` | `/layer/torch/group_norm` | layer |
| `dropout` | `/layer/torch/dropout` | layer |
| `reparam` | `/layer/kalfa/reparam` | layer |
| `l1_distance` | `/layer/kalfa/l1_distance` | layer |
| `gru` | `/layer/torch/gru` | layer |
| `last_step` | `/layer/torch/last_step` | layer |
| `softsign` | `/layer/torch/softsign` | layer |
| `avgpool3d` | `/layer/torch/avgpool3d` | layer |
| `adaptive_maxpool3d` | `/layer/torch/adaptive_maxpool3d` | layer |
| `constant_pad1d` | `/layer/torch/constant_pad1d` | layer |
| `elu` | `/layer/torch/elu` | layer |
| `pixel_unshuffle` | `/layer/torch/pixel_unshuffle` | layer |
| `zero_pad1d` | `/layer/torch/zero_pad1d` | layer |
| `alpha_dropout` | `/layer/torch/alpha_dropout` | layer |
| `lppool1d` | `/layer/torch/lppool1d` | layer |
| `rrelu` | `/layer/torch/rrelu` | layer |
| `prelu` | `/layer/torch/prelu` | layer |
| `cosine_similarity` | `/layer/torch/cosine_similarity` | layer |
| `replication_pad` | `/layer/torch/replication_pad` | layer |
| `embedding_bag` | `/layer/torch/embedding_bag` | layer |
| `max_unpool3d` | `/layer/torch/max_unpool3d` | layer |
| `hardshrink` | `/layer/torch/hardshrink` | layer |
| `hardswish` | `/layer/torch/hardswish` | layer |
| `rnn` | `/layer/torch/rnn` | layer |
| `conv_transpose2d` | `/layer/torch/conv_transpose2d` | layer |
| `dropout2d` | `/layer/torch/dropout2d` | layer |
| `dropout3d` | `/layer/torch/dropout3d` | layer |
| `avgpool` | `/layer/torch/avgpool` | layer |
| `lppool` | `/layer/torch/lppool` | layer |
| `threshold_layer` | `/layer/torch/threshold` | layer |
| `conv_transpose3d` | `/layer/torch/conv_transpose3d` | layer |
| `tanh_layer` | `/layer/torch/tanh` | layer |
| `identity` | `/layer/torch/identity` | layer |
| `mlp` | `/layer/kalfa/mlp` | layer |
| `feature_alpha_dropout` | `/layer/torch/feature_alpha_dropout` | layer |
| `adaptive_avgpool3d` | `/layer/torch/adaptive_avgpool3d` | layer |
| `softplus` | `/layer/torch/softplus` | layer |
| `softmax` | `/layer/torch/softmax` | layer |
| `silu` | `/layer/torch/silu` | layer |
| `softmin` | `/layer/torch/softmin` | layer |
| `lstm_cell` | `/layer/torch/lstm_cell` | layer |
| `constant_pad3d` | `/layer/torch/constant_pad3d` | layer |
| `transformer_decoder_layer` | `/layer/torch/transformer_decoder_layer` | layer |
| `relu6` | `/layer/torch/relu6` | layer |
| `reflection_pad1d` | `/layer/torch/reflection_pad1d` | layer |
| `maxpool1d` | `/layer/torch/maxpool1d` | layer |
| `softshrink` | `/layer/torch/softshrink` | layer |
| `conv_transpose1d` | `/layer/torch/conv_transpose1d` | layer |
| `conv1d` | `/layer/torch/conv1d` | layer |
| `hardsigmoid` | `/layer/torch/hardsigmoid` | layer |
| `reflection_pad` | `/layer/torch/reflection_pad` | layer |
| `local_response_norm` | `/layer/torch/local_response_norm` | layer |
| `conv3d` | `/layer/torch/conv3d` | layer |
| `lstm` | `/layer/torch/lstm` | layer |
| `max_unpool` | `/layer/torch/max_unpool` | layer |
| `constant_pad` | `/layer/torch/constant_pad` | layer |
| `mish` | `/layer/torch/mish` | layer |
| `rnn_cell` | `/layer/torch/rnn_cell` | layer |
| `zero_pad` | `/layer/torch/zero_pad` | layer |
| `pairwise_distance` | `/layer/torch/pairwise_distance` | layer |
| `celu` | `/layer/torch/celu` | layer |
| `log_sigmoid` | `/layer/torch/log_sigmoid` | layer |
| `fold` | `/layer/torch/fold` | layer |
| `pixel_shuffle` | `/layer/torch/pixel_shuffle` | layer |
| `log_softmax` | `/layer/torch/log_softmax` | layer |
| `transformer_encoder_layer` | `/layer/torch/transformer_encoder_layer` | layer |
| `fractional_maxpool` | `/layer/torch/fractional_maxpool` | layer |
| `adaptive_maxpool` | `/layer/torch/adaptive_maxpool` | layer |
| `adaptive_avgpool1d` | `/layer/torch/adaptive_avgpool1d` | layer |
| `glu` | `/layer/torch/glu` | layer |
| `instance_norm` | `/layer/torch/instance_norm` | layer |
| `circular_pad3d` | `/layer/torch/circular_pad3d` | layer |
| `circular_pad` | `/layer/torch/circular_pad` | layer |
| `upsample` | `/layer/torch/upsample` | layer |
| `circular_pad1d` | `/layer/torch/circular_pad1d` | layer |
| `adaptive_avgpool` | `/layer/torch/adaptive_avgpool` | layer |
| `selu` | `/layer/torch/selu` | layer |
| `transformer_encoder` | `/layer/torch/transformer_encoder` | layer |
| `channel_shuffle` | `/layer/torch/channel_shuffle` | layer |
| `softmax2d` | `/layer/torch/softmax2d` | layer |
| `reflection_pad3d` | `/layer/torch/reflection_pad3d` | layer |
| `transformer` | `/layer/torch/transformer` | layer |
| `zero_pad3d` | `/layer/torch/zero_pad3d` | layer |
| `rms_norm` | `/layer/torch/rms_norm` | layer |
| `max_unpool1d` | `/layer/torch/max_unpool1d` | layer |
| `sigmoid` | `/layer/torch/sigmoid` | layer |
| `multihead_attention` | `/layer/torch/multihead_attention` | layer |
| `adaptive_maxpool1d` | `/layer/torch/adaptive_maxpool1d` | layer |
| `maxpool3d` | `/layer/torch/maxpool3d` | layer |
| `lppool3d` | `/layer/torch/lppool3d` | layer |
| `fractional_maxpool3d` | `/layer/torch/fractional_maxpool3d` | layer |
| `hardtanh` | `/layer/torch/hardtanh` | layer |
| `dropout1d` | `/layer/torch/dropout1d` | layer |
| `bilinear` | `/layer/torch/bilinear` | layer |
| `tanhshrink` | `/layer/torch/tanhshrink` | layer |
| `gelu` | `/layer/torch/gelu` | layer |
| `gru_cell` | `/layer/torch/gru_cell` | layer |
| `replication_pad1d` | `/layer/torch/replication_pad1d` | layer |
| `avgpool1d` | `/layer/torch/avgpool1d` | layer |
| `unfold` | `/layer/torch/unfold` | layer |
| `replication_pad3d` | `/layer/torch/replication_pad3d` | layer |
| `transformer_decoder` | `/layer/torch/transformer_decoder` | layer |
| `normal` | `/init/torch/normal` | init |
| `xavier` | `/init/torch/xavier` | init |
| `kaiming` | `/init/torch/kaiming` | init |
| `kaiming_uniform` | `/init/torch/kaiming_uniform` | init |
| `zeros` | `/init/torch/zeros` | init |
| `mse` | `/criterion/kalfa/mse` | criterion |
| `weighted_mse` | `/criterion/kalfa/weighted_mse` | criterion |
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
| `binary_roc` | `/plot/torchmetrics/binary_roc` | plot |
| `binary_precision_recall_curve` | `/plot/torchmetrics/binary_precision_recall_curve` | plot |
| `confusion_matrix` | `/plot/kalfa/confusion_matrix` | plot |
| `forecast_samples` | `/plot/kalfa/forecast_samples` | plot |
| `architecture` | `/plot/kalfa/architecture` | plot |
| `data_pipeline` | `/plot/kalfa/data_pipeline` | plot |
| `architecture_text` | `/plot/kalfa/architecture_text` | plot |
| `torchview` | `/plot/torchview/architecture` | plot |
| `weighted_sum` | `/objective/kalfa/weighted_sum` | objective |
| `grid` | `/strategy/kalfa/grid` | strategy |
| `random` | `/strategy/kalfa/random` | strategy |
| `sobol` | `/strategy/kalfa/sobol` | strategy |
| `optuna` | `/strategy/kalfa/optuna` | strategy |
| `auto` | `/device/kalfa/auto` | device |
| `cpu` | `/device/kalfa/cpu` | device |
| `cuda` | `/device/kalfa/cuda` | device |
| `mps` | `/device/kalfa/mps` | device |
| `threshold` | `/calibrate/kalfa/threshold` | calibrate |
| `onnx` | `/export/kalfa/onnx` | export |
| `pt2` | `/export/kalfa/pt2` | export |
| `state_dict` | `/export/kalfa/state_dict` | export |
| `derived` | `/rng/kalfa/derived` | rng |
| `indexed` | `/rng/kalfa/indexed` | rng |
| `global` | `/rng/kalfa/global` | rng |
| `parquet` | `/source/kalfa/parquet` | source |
| `csv` | `/source/kalfa/csv` | source |
| `text_lines` | `/source/kalfa/text_lines` | source |
| `random_split` | `/split/kalfa/random` | split |
| `kfold` | `/split/kalfa/kfold` | split |
| `window` | `/feed/kalfa/window` | feed |
| `next_token` | `/feed/kalfa/next_token` | feed |
| `target_vs_features` | `/plot/kalfa/target_vs_features` | plot |
| `correlation_heatmap` | `/plot/kalfa/correlation_heatmap` | plot |
| `residuals` | `/plot/kalfa/residuals` | plot |
| `error_map` | `/plot/kalfa/error_map` | plot |
| `permutation_importance` | `/plot/kalfa/permutation_importance` | plot |
| `feature_distributions` | `/plot/kalfa/feature_distributions` | plot |
| `target_correlation` | `/plot/kalfa/target_correlation` | plot |
| `pairplot` | `/plot/seaborn/pairplot` | plot |
| `violin` | `/plot/seaborn/violin` | plot |
| `kde` | `/plot/seaborn/kde` | plot |
| `class_weights` | `/data/kalfa/class_weights` | data |
| `vocab_size` | `/data/kalfa/vocab_size` | data |
| `char_tokenizer` | `/pre/kalfa/char_tokenizer` | pre |
| `embedding` | `/layer/torch/embedding` | layer |
| `perplexity` | `/metric/kalfa/perplexity` | metric |
| `sample_writer` | `/metric/kalfa/sample_writer` | metric |
| `lm_sampler` | `/generate/kalfa/lm_sampler` | generate |

### /alias/kalfa/vision

| Alias | URI | Kind |
|---|---|---|
| `filter` | `/transform/kalfa/filter` | transform |
| `derive` | `/transform/kalfa/derive` | transform |
| `rename` | `/transform/kalfa/rename` | transform |
| `astype` | `/transform/kalfa/astype` | transform |
| `drop` | `/transform/kalfa/drop` | transform |
| `sequential` | `/split/kalfa/sequential` | split |
| `given` | `/split/kalfa/given` | split |
| `group_statistic` | `/frame/kalfa/group_statistic` | frame |
| `target_encoding` | `/frame/kalfa/target_encoding` | frame |
| `standard_scaler` | `/pre/sklearn/standard_scaler` | pre |
| `median_std_scaler` | `/pre/kalfa/median_std_scaler` | pre |
| `minmax_scaler` | `/pre/sklearn/minmax_scaler` | pre |
| `cast` | `/pre/kalfa/cast` | pre |
| `abs` | `/pre/kalfa/abs` | pre |
| `log` | `/pre/kalfa/log` | pre |
| `one_hot` | `/pre/kalfa/one_hot` | pre |
| `label_encoder` | `/pre/kalfa/label_encoder` | pre |
| `simple_imputer` | `/pre/kalfa/simple_imputer` | pre |
| `fill` | `/pre/kalfa/fill` | pre |
| `table` | `/feed/kalfa/table` | feed |
| `linear` | `/layer/kalfa/linear` | layer |
| `linear_relu` | `/layer/kalfa/linear_relu` | layer |
| `concat` | `/layer/torch/concat` | layer |
| `flatten` | `/layer/torch/flatten` | layer |
| `relu` | `/layer/torch/relu` | layer |
| `leaky_relu` | `/layer/torch/leaky_relu` | layer |
| `batch_norm` | `/layer/torch/batch_norm` | layer |
| `layer_norm` | `/layer/torch/layer_norm` | layer |
| `group_norm` | `/layer/torch/group_norm` | layer |
| `dropout` | `/layer/torch/dropout` | layer |
| `reparam` | `/layer/kalfa/reparam` | layer |
| `l1_distance` | `/layer/kalfa/l1_distance` | layer |
| `gru` | `/layer/torch/gru` | layer |
| `last_step` | `/layer/torch/last_step` | layer |
| `softsign` | `/layer/torch/softsign` | layer |
| `avgpool3d` | `/layer/torch/avgpool3d` | layer |
| `adaptive_maxpool3d` | `/layer/torch/adaptive_maxpool3d` | layer |
| `constant_pad1d` | `/layer/torch/constant_pad1d` | layer |
| `elu` | `/layer/torch/elu` | layer |
| `pixel_unshuffle` | `/layer/torch/pixel_unshuffle` | layer |
| `zero_pad1d` | `/layer/torch/zero_pad1d` | layer |
| `alpha_dropout` | `/layer/torch/alpha_dropout` | layer |
| `lppool1d` | `/layer/torch/lppool1d` | layer |
| `rrelu` | `/layer/torch/rrelu` | layer |
| `prelu` | `/layer/torch/prelu` | layer |
| `cosine_similarity` | `/layer/torch/cosine_similarity` | layer |
| `replication_pad` | `/layer/torch/replication_pad` | layer |
| `embedding_bag` | `/layer/torch/embedding_bag` | layer |
| `max_unpool3d` | `/layer/torch/max_unpool3d` | layer |
| `hardshrink` | `/layer/torch/hardshrink` | layer |
| `hardswish` | `/layer/torch/hardswish` | layer |
| `rnn` | `/layer/torch/rnn` | layer |
| `conv_transpose2d` | `/layer/torch/conv_transpose2d` | layer |
| `dropout2d` | `/layer/torch/dropout2d` | layer |
| `dropout3d` | `/layer/torch/dropout3d` | layer |
| `avgpool` | `/layer/torch/avgpool` | layer |
| `lppool` | `/layer/torch/lppool` | layer |
| `threshold_layer` | `/layer/torch/threshold` | layer |
| `conv_transpose3d` | `/layer/torch/conv_transpose3d` | layer |
| `tanh_layer` | `/layer/torch/tanh` | layer |
| `identity` | `/layer/torch/identity` | layer |
| `mlp` | `/layer/kalfa/mlp` | layer |
| `feature_alpha_dropout` | `/layer/torch/feature_alpha_dropout` | layer |
| `adaptive_avgpool3d` | `/layer/torch/adaptive_avgpool3d` | layer |
| `softplus` | `/layer/torch/softplus` | layer |
| `softmax` | `/layer/torch/softmax` | layer |
| `silu` | `/layer/torch/silu` | layer |
| `softmin` | `/layer/torch/softmin` | layer |
| `lstm_cell` | `/layer/torch/lstm_cell` | layer |
| `constant_pad3d` | `/layer/torch/constant_pad3d` | layer |
| `transformer_decoder_layer` | `/layer/torch/transformer_decoder_layer` | layer |
| `relu6` | `/layer/torch/relu6` | layer |
| `reflection_pad1d` | `/layer/torch/reflection_pad1d` | layer |
| `maxpool1d` | `/layer/torch/maxpool1d` | layer |
| `softshrink` | `/layer/torch/softshrink` | layer |
| `conv_transpose1d` | `/layer/torch/conv_transpose1d` | layer |
| `conv1d` | `/layer/torch/conv1d` | layer |
| `hardsigmoid` | `/layer/torch/hardsigmoid` | layer |
| `reflection_pad` | `/layer/torch/reflection_pad` | layer |
| `local_response_norm` | `/layer/torch/local_response_norm` | layer |
| `conv3d` | `/layer/torch/conv3d` | layer |
| `lstm` | `/layer/torch/lstm` | layer |
| `max_unpool` | `/layer/torch/max_unpool` | layer |
| `constant_pad` | `/layer/torch/constant_pad` | layer |
| `mish` | `/layer/torch/mish` | layer |
| `rnn_cell` | `/layer/torch/rnn_cell` | layer |
| `zero_pad` | `/layer/torch/zero_pad` | layer |
| `pairwise_distance` | `/layer/torch/pairwise_distance` | layer |
| `celu` | `/layer/torch/celu` | layer |
| `log_sigmoid` | `/layer/torch/log_sigmoid` | layer |
| `fold` | `/layer/torch/fold` | layer |
| `pixel_shuffle` | `/layer/torch/pixel_shuffle` | layer |
| `log_softmax` | `/layer/torch/log_softmax` | layer |
| `transformer_encoder_layer` | `/layer/torch/transformer_encoder_layer` | layer |
| `fractional_maxpool` | `/layer/torch/fractional_maxpool` | layer |
| `adaptive_maxpool` | `/layer/torch/adaptive_maxpool` | layer |
| `adaptive_avgpool1d` | `/layer/torch/adaptive_avgpool1d` | layer |
| `glu` | `/layer/torch/glu` | layer |
| `instance_norm` | `/layer/torch/instance_norm` | layer |
| `circular_pad3d` | `/layer/torch/circular_pad3d` | layer |
| `circular_pad` | `/layer/torch/circular_pad` | layer |
| `upsample` | `/layer/torch/upsample` | layer |
| `circular_pad1d` | `/layer/torch/circular_pad1d` | layer |
| `adaptive_avgpool` | `/layer/torch/adaptive_avgpool` | layer |
| `selu` | `/layer/torch/selu` | layer |
| `transformer_encoder` | `/layer/torch/transformer_encoder` | layer |
| `channel_shuffle` | `/layer/torch/channel_shuffle` | layer |
| `softmax2d` | `/layer/torch/softmax2d` | layer |
| `reflection_pad3d` | `/layer/torch/reflection_pad3d` | layer |
| `transformer` | `/layer/torch/transformer` | layer |
| `zero_pad3d` | `/layer/torch/zero_pad3d` | layer |
| `rms_norm` | `/layer/torch/rms_norm` | layer |
| `max_unpool1d` | `/layer/torch/max_unpool1d` | layer |
| `sigmoid` | `/layer/torch/sigmoid` | layer |
| `multihead_attention` | `/layer/torch/multihead_attention` | layer |
| `adaptive_maxpool1d` | `/layer/torch/adaptive_maxpool1d` | layer |
| `maxpool3d` | `/layer/torch/maxpool3d` | layer |
| `lppool3d` | `/layer/torch/lppool3d` | layer |
| `fractional_maxpool3d` | `/layer/torch/fractional_maxpool3d` | layer |
| `hardtanh` | `/layer/torch/hardtanh` | layer |
| `dropout1d` | `/layer/torch/dropout1d` | layer |
| `bilinear` | `/layer/torch/bilinear` | layer |
| `tanhshrink` | `/layer/torch/tanhshrink` | layer |
| `gelu` | `/layer/torch/gelu` | layer |
| `gru_cell` | `/layer/torch/gru_cell` | layer |
| `replication_pad1d` | `/layer/torch/replication_pad1d` | layer |
| `avgpool1d` | `/layer/torch/avgpool1d` | layer |
| `unfold` | `/layer/torch/unfold` | layer |
| `replication_pad3d` | `/layer/torch/replication_pad3d` | layer |
| `transformer_decoder` | `/layer/torch/transformer_decoder` | layer |
| `normal` | `/init/torch/normal` | init |
| `xavier` | `/init/torch/xavier` | init |
| `kaiming` | `/init/torch/kaiming` | init |
| `kaiming_uniform` | `/init/torch/kaiming_uniform` | init |
| `zeros` | `/init/torch/zeros` | init |
| `mse` | `/criterion/kalfa/mse` | criterion |
| `weighted_mse` | `/criterion/kalfa/weighted_mse` | criterion |
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
| `binary_roc` | `/plot/torchmetrics/binary_roc` | plot |
| `binary_precision_recall_curve` | `/plot/torchmetrics/binary_precision_recall_curve` | plot |
| `confusion_matrix` | `/plot/kalfa/confusion_matrix` | plot |
| `forecast_samples` | `/plot/kalfa/forecast_samples` | plot |
| `architecture` | `/plot/kalfa/architecture` | plot |
| `data_pipeline` | `/plot/kalfa/data_pipeline` | plot |
| `architecture_text` | `/plot/kalfa/architecture_text` | plot |
| `torchview` | `/plot/torchview/architecture` | plot |
| `weighted_sum` | `/objective/kalfa/weighted_sum` | objective |
| `grid` | `/strategy/kalfa/grid` | strategy |
| `random` | `/strategy/kalfa/random` | strategy |
| `sobol` | `/strategy/kalfa/sobol` | strategy |
| `optuna` | `/strategy/kalfa/optuna` | strategy |
| `auto` | `/device/kalfa/auto` | device |
| `cpu` | `/device/kalfa/cpu` | device |
| `cuda` | `/device/kalfa/cuda` | device |
| `mps` | `/device/kalfa/mps` | device |
| `threshold` | `/calibrate/kalfa/threshold` | calibrate |
| `onnx` | `/export/kalfa/onnx` | export |
| `pt2` | `/export/kalfa/pt2` | export |
| `state_dict` | `/export/kalfa/state_dict` | export |
| `derived` | `/rng/kalfa/derived` | rng |
| `indexed` | `/rng/kalfa/indexed` | rng |
| `global` | `/rng/kalfa/global` | rng |
| `parquet` | `/source/kalfa/parquet` | source |
| `csv` | `/source/kalfa/csv` | source |
| `image_folder` | `/source/kalfa/image_folder` | source |
| `random_split` | `/split/kalfa/random` | split |
| `kfold` | `/split/kalfa/kfold` | split |
| `window` | `/feed/kalfa/window` | feed |
| `target_vs_features` | `/plot/kalfa/target_vs_features` | plot |
| `correlation_heatmap` | `/plot/kalfa/correlation_heatmap` | plot |
| `residuals` | `/plot/kalfa/residuals` | plot |
| `error_map` | `/plot/kalfa/error_map` | plot |
| `permutation_importance` | `/plot/kalfa/permutation_importance` | plot |
| `feature_distributions` | `/plot/kalfa/feature_distributions` | plot |
| `target_correlation` | `/plot/kalfa/target_correlation` | plot |
| `pairplot` | `/plot/seaborn/pairplot` | plot |
| `violin` | `/plot/seaborn/violin` | plot |
| `kde` | `/plot/seaborn/kde` | plot |
| `image_pairs` | `/plot/kalfa/image_pairs` | plot |
| `image_grid` | `/plot/kalfa/image_grid` | plot |
| `samples_gif` | `/plot/kalfa/samples_gif` | plot |
| `samples_matrix` | `/plot/kalfa/samples_matrix` | plot |
| `class_weights` | `/data/kalfa/class_weights` | data |
| `to_tensor` | `/pre/kalfa/to_tensor` | pre |
| `to_tensor_signed` | `/pre/kalfa/to_tensor_signed` | pre |
| `resize` | `/pre/kalfa/resize` | pre |
| `random_crop_flip` | `/pre/kalfa/random_crop_flip` | pre |
| `normalize` | `/pre/kalfa/normalize` | pre |
| `two_views` | `/pre/kalfa/two_views` | pre |
| `simclr_aug` | `/pre/kalfa/simclr_aug` | pre |
| `unflatten` | `/layer/kalfa/unflatten` | layer |
| `embedding` | `/layer/torch/embedding` | layer |
| `conv2d` | `/layer/torch/conv2d` | layer |
| `maxpool` | `/layer/torch/maxpool` | layer |
| `vae` | `/objective/kalfa/vae` | objective |
| `distill` | `/objective/kalfa/distill` | objective |
| `wgan_gp_d` | `/objective/kalfa/wgan_gp_d` | objective |
| `wgan_g` | `/objective/kalfa/wgan_g` | objective |
| `ddpm` | `/objective/kalfa/ddpm` | objective |
| `ntxent` | `/objective/kalfa/ntxent` | objective |
| `fid` | `/metric/kalfa/fid` | metric |
| `recon_error` | `/metric/kalfa/recon_error` | metric |
| `sample_writer` | `/metric/kalfa/sample_writer` | metric |
| `gan_sampler` | `/generate/kalfa/gan_sampler` | generate |
| `ddpm_sampler` | `/generate/kalfa/ddpm_sampler` | generate |
| `linear_betas` | `/schedule/kalfa/linear_betas` | schedule |
