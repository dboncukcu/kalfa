from kalfa.core import aliases


def lines(*rows):
    return "\n".join(rows)


LR = "the base learning rate; a schedule multiplies it, a group of over may set its own and a rule may patch it"
BETAS = "[beta1, beta2]: the decay rates of the running mean of the gradient and of its square"
EPS = "added to the denominator for numerical stability"
MAXIMIZE = "true climbs the objective instead of descending it"
REDUCTION = ("how the element losses combine: mean (the default) or sum; none leaves a tensor that an optimizer cannot "
             "step on")
DEPRECATED = "deprecated by torch; write reduction instead"
INPLACE = "true overwrites the input tensor to save memory"
BATCH = ("the batch size, or a mapping of size, eval_size (the batch of the valid and test sets), workers, drop_last "
         "and shuffle")
WATCH = "the series read every turn, such as val/rmse, train/total/kl or track/optimizers.main.lr"
ORDER = "the optimizer names in the order they step; null takes the order of the optimizers section"
AMP = ("runs the forward passes under torch.autocast on cuda and mps (it does nothing on cpu), with a gradient scaler "
       "on cuda")
GRAD_CLIP = "clips the gradient norm of every update to this value; null does not clip"
ACCUMULATE = "sums the gradients of this many batches into one update"
PLOT_PRED = ("the output to draw, by its name (net) or its column (pred_net); null takes the first output that pairs "
             "with a target")
PLOT_TARGET = "the target column of the predictions table; null takes the target that pairs with pred"
CLASSES_LAST = ("null finds the class axis from the shapes; true means (batch, tokens, vocabulary) and false (batch, "
                "vocabulary, tokens); needed only when the tokens and the vocabulary have the same size")
COUNTED = ("the classes 0 to num_classes - 1 that are counted; null counts the classes seen in the predictions and the "
           "targets")
SCORED = ("kept for symmetry with accuracy and f1 and not read: the classes come from the width of the logits, and one "
          "score column (or (N, 2) logits) is a binary task")
SAMPLES = "the samples drawn at the end of a run; kalfa generate and write_samples may ask for another count"
SIZE = "an int for a square, or [height, width]"
LATENT = "the width of the latent noise z"
SPLIT_SEED = "the seed of the shuffle; null takes the top level seed"
SPLIT_RATIOS = "up to three shares for train, valid and test that sum to 1; [0.8, 0.2] makes no test set"
DEVICE = "the device torch makes the weights on; kalfa moves the models to the run device, so leave it out"
DTYPE = "the dtype of the weights; leave it out to keep float32"
GIVEN = "the value that marks the {} rows; null makes no {} set"
PATH = "a relative path is looked for in the working directory, then next to the config files"
LOCAL = "an object with write, given from Python; null writes to stderr"


DATA = {
    "table": {
        "params": {
            "file": f"the table to read: a .parquet, .pq, .csv or .tsv file; {PATH}",
            "split": ("required: a list of ratios for a shuffled split into train, valid and test ([0.8, 0.2] makes "
                      "no test set), or a split call: random, sequential, kfold or given"),
            "features": ("a mapping of column names or fnmatch patterns (\"x*\") to a preprocessor, a list of "
                         "preprocessors applied in order, or null for the raw column; a literal name wins over a "
                         "pattern, and a column that no pattern matches is dropped with a warning"),
            "targets": ("a mapping of target columns to a preprocessor, a list of them or null; every target is a "
                        "batch field under its column name, and label makes a classification target (an int64 field, "
                        "with classes and class_weights exposed)"),
            "filter": ("a pandas query kept for every row before the split, or a mapping of all (before the split) "
                       "and train, valid, test (each set after the split) to queries"),
            "drop": "columns removed after the split and the filters, before the features are chosen",
            "spectators": ("columns or patterns that are not fed to the model but are copied into the predictions "
                           "table, such as an id"),
            "feed": "a call that turns the rows into samples another way, such as {use: window, size: 24, horizon: 6}",
            "batch": f"{BATCH}; the default size is 128",
            "stream": ("true or {chunk, fit_rows} reads the train set chunk by chunk instead of loading the file; it "
                       "needs a sequential or given split and no feed"),
        },
        "examples": [
            lines(
                "data:",
                "  use: table",
                "  file: housing.parquet",
                "  split: [0.7, 0.15, 0.15]",
                '  features: {"x*": standard_scaler, size: [log, minmax_scaler]}',
                "  targets: {price: standard_scaler}",
                '  filter: {all: "price > 0", train: "size < 500"}',
                "  spectators: [id]",
                "  batch: {size: 128, eval_size: 512}",
            ),
            lines(
                "data:",
                "  use: table",
                "  file: churn.parquet",
                "  split: {use: kfold, folds: 5, fold: 0, valid_ratio: 0.15}",
                '  features: {"num_*": standard_scaler, "cat_*": one_hot}',
                "  targets: {churned: label}",
                "  drop: [notes]",
                "  batch: 64",
            ),
        ],
    },
    "image_folder": {
        "params": {
            "folder": f"the directory whose subdirectories are the classes, each holding its images; {PATH}",
            "split": ("required: a list of ratios for a shuffled split into train, valid and test, or a split call "
                      "such as {use: kfold, folds: 5, fold: 0}"),
            "transform": ("the transforms of every set, in order: short names, calls or import paths, each from image "
                          "to image; the result always becomes a float tensor (C, H, W)"),
            "train_transform": ("replaces transform for the train set, the place for augmentation such as random_crop "
                                "and random_flip; null uses transform"),
            "batch": f"{BATCH}; the default size is 64",
            "labels": ("true gives the label field (int64 codes of the class folders); false gives no target, for "
                       "autoencoders and generative models"),
            "extensions": "the file suffixes read as images",
        },
        "examples": [
            lines(
                "data:",
                "  use: image_folder",
                "  folder: data/pets",
                "  split: [0.7, 0.15, 0.15]",
                "  transform: [{use: resize, size: 32}, to_tensor, {use: normalize, mean: imagenet}]",
                "  train_transform:",
                "    - {use: resize, size: 36}",
                "    - {use: random_crop, size: 32}",
                "    - random_flip",
                "    - to_tensor",
                "    - {use: normalize, mean: imagenet}",
                "  batch: {size: 16, eval_size: 256}",
            ),
            lines(
                "data:",
                "  use: image_folder",
                "  folder: data/digits",
                "  split: [0.8, 0.1, 0.1]",
                "  labels: false",
                "  transform: [to_tensor]",
                "  batch: 16",
            ),
        ],
    },
    "text_lines": {
        "params": {
            "file": f"a utf-8 text file read line by line, empty lines skipped; {PATH}",
            "tokenizer": ("char builds a character vocabulary from the train lines (id 0 is unknown); also a short "
                          "name such as char_tokenizer, a call, an import path or an object with encode, decode, size "
                          "and fit"),
            "seq_len": ("the tokens of a window: input_ids holds seq_len tokens and targets the next token at each "
                        "place"),
            "split": ("required: a list of ratios over the lines, or a split call such as {use: sequential, ratios: "
                      "[0.9, 0.1]}"),
            "batch": f"{BATCH}; the default size is 64",
            "stride": "the tokens between the starts of two windows; null takes seq_len, so the windows do not overlap",
            "lowercase": "lowercases the text before the tokenizer reads it",
        },
        "examples": [
            lines(
                "data:",
                "  use: text_lines",
                "  file: data/corpus.txt",
                "  tokenizer: char",
                "  seq_len: 64",
                "  split: [0.9, 0.05, 0.05]",
                "  batch: 64",
            ),
            lines(
                "data:",
                "  use: text_lines",
                "  file: data/lines.txt",
                "  seq_len: 128",
                "  stride: 64",
                "  lowercase: true",
                "  split: {use: sequential, ratios: [0.9, 0.1]}",
                "  batch: 32",
            ),
        ],
    },
}


MODELS = {
    "mlp": {
        "params": {
            "hidden": "the widths of the hidden layers, a list such as [128, 128] or one int",
            "out": ("the width of the last linear layer: 1 for one regression target, \"@data.n_classes\" for the "
                    "logits of a classification"),
            "activation": ("the activation after every hidden layer: relu, gelu, tanh, silu, sigmoid, elu, leaky_relu, "
                           "softplus or a module class"),
            "dropout": "the dropout probability after every hidden activation; 0 adds no dropout",
            "in_features": ("the input width; null lets the first layer infer it on the first forward pass, a number "
                            "or \"@data.n_features\" builds the weights at once"),
        },
        "examples": [
            lines(
                "models:",
                "  net: {use: mlp, hidden: [128, 128], out: 1}",
            ),
            lines(
                "models:",
                '  net: {use: mlp, hidden: [64, 32], out: "@data.n_classes", activation: gelu, dropout: 0.1}',
            ),
        ],
    },
    "stack": {
        "params": {
            "layers": ("the layers in order: calls such as {use: linear, out_features: 64}, short names of layers "
                       "without params (relu, flatten, last_step) or import paths; the first layer receives the model "
                       "input"),
        },
        "examples": [
            lines(
                "models:",
                "  net:",
                "    use: stack",
                "    layers:",
                '      - {use: torch.nn.GRU, input_size: "@data.n_features", hidden_size: 64, batch_first: true}',
                "      - last_step",
                "      - {use: linear, out_features: 24}",
            ),
        ],
    },
}


LOSSES = {
    "mse": {
        "params": {"reduction": REDUCTION, "size_average": DEPRECATED, "reduce": DEPRECATED},
        "examples": [
            lines(
                "losses:",
                "  mse: mse",
            ),
            lines(
                "losses:",
                "  rec: {use: mse, pred: x_hat, target: image}",
            ),
        ],
    },
    "l1": {
        "params": {"reduction": REDUCTION, "size_average": DEPRECATED, "reduce": DEPRECATED},
        "examples": [
            lines(
                "losses:",
                "  l1: l1",
            ),
            lines(
                "losses:",
                "  mae: {use: l1, pred: y_yield, target: yield}",
            ),
        ],
    },
    "huber": {
        "params": {
            "delta": "the error size where the loss turns from quadratic to linear",
            "reduction": REDUCTION,
        },
        "examples": [
            lines(
                "losses:",
                "  huber: {use: huber, delta: 0.5}",
            ),
        ],
    },
    "bce_logits": {
        "params": {
            "weight": "element weights that broadcast over the outputs; a YAML number or list becomes a tensor",
            "pos_weight": ("weighs the positive class of every output, one number per output (above 1 favors recall); "
                           "a YAML number or list becomes a tensor"),
            "reduction": REDUCTION,
            "size_average": DEPRECATED,
            "reduce": DEPRECATED,
        },
        "examples": [
            lines(
                "data:",
                "  use: table",
                "  file: anomaly_data.parquet",
                '  filter: "is_anomaly in [0, 1]"',
                "  split: [0.7, 0.15, 0.15]",
                '  features: {"x*": [abs, {use: log, base: 10, norm: 1200}, standard_scaler]}',
                "  targets: {is_anomaly: ~}",
                "models:",
                "  net: {use: mlp, hidden: [64], out: 1}",
                "losses:",
                "  bce: {use: bce_logits, pos_weight: 4.0}",
            ),
        ],
    },
    "poisson_nll": {
        "params": {
            "log_input": ("true reads the prediction as the log of the rate (loss exp(pred) - target * pred); false "
                          "reads it as the rate itself"),
            "full": "adds the Stirling approximation term of log(target!)",
            "eps": "added inside the log when log_input is false, so a zero rate does not give infinity",
            "reduction": REDUCTION,
            "size_average": DEPRECATED,
            "reduce": DEPRECATED,
        },
        "examples": [
            lines(
                "losses:",
                "  l_defects: {use: poisson_nll, pred: y_defects, target: defects, log_input: true}",
            ),
        ],
    },
    "cross_entropy": {
        "params": {
            "weight": ("a weight per class, a list or \"@data.class_weights\" (inverse frequency, mean one); null "
                       "weighs every class the same"),
            "label_smoothing": "the share of the target mass spread over all classes, from 0 to 1",
            "classes_last": CLASSES_LAST,
        },
        "examples": [
            lines(
                "losses:",
                '  ce: {use: cross_entropy, weight: "@data.class_weights", label_smoothing: 0.1}',
            ),
            lines(
                "losses:",
                "  ce: {use: cross_entropy, target: targets}",
            ),
        ],
    },
    "weighted_sum": {
        "params": {
            "terms": ("a list of {of, weight, term}: of is a loss reference, weight its factor (default 1.0, a rule "
                      "may patch terms[i].weight) and term a key of a dict result; every term is also a series"),
        },
        "examples": [
            lines(
                "losses:",
                "  l_yield: {use: mse, pred: y_yield, target: yield}",
                "  l_moisture: {use: huber, pred: y_moisture, target: moisture}",
                "  total:",
                "    use: weighted_sum",
                "    terms:",
                '      - {of: "@losses.l_yield", weight: 1.0}',
                '      - {of: "@losses.l_moisture", weight: 0.5}',
            ),
        ],
    },
    "mdmm": {
        "params": {
            "objective": "the loss to minimize, a reference such as \"@losses.mse\"",
            "constraints": ("a list of {of, epsilon, scale, damping, lmbda_init, term}: of is the constrained loss, "
                            "epsilon the value it is held at, scale and damping weigh the multiplier and the quadratic "
                            "damping, lmbda_init starts the multiplier and term picks a key of a dict result; every "
                            "constraint has a lambda, which an optimizer group with a negative lr trains"),
            "term": "a key of the dict result of the objective to minimize; null takes its loss",
        },
        "examples": [
            lines(
                "losses:",
                "  mse: mse",
                "  mae: l1",
                "  total:",
                "    use: mdmm",
                '    objective: "@losses.mse"',
                "    constraints:",
                '      - {of: "@losses.mae", epsilon: 0.05, lmbda_init: 0.5}',
                "optimizers:",
                "  main:",
                "    use: adam",
                "    lr: 1.0e-3",
                '    over: ["@models.net", {of: "@losses.total", lr: -0.05}]',
                '    minimizes: "@losses.total"',
            ),
        ],
    },
    "vae_elbo": {
        "params": {
            "recon": ("the reconstruction criterion, a reference such as \"@losses.mse\" whose pred and target name "
                      "the reconstruction and the input"),
            "w_rec": "the weight of the reconstruction term; a rule may patch it",
            "w_kl": ("the weight of the KL term: a number or a schedule of the update count such as {use: "
                     "linear_warmup, start: 0.0, end: 1.0, steps: 1000}"),
            "mu": "the output name of the latent means",
            "logvar": "the output name of the latent log variances",
        },
        "examples": [
            lines(
                "models:",
                "  vae: {use: vae_legos.Vae, latent: 32, outputs: [x_hat, mu, logvar]}",
                "losses:",
                "  mse: {use: mse, pred: x_hat, target: image}",
                "  elbo:",
                "    use: vae_elbo",
                '    recon: "@losses.mse"',
                "    w_rec: 1.0",
                "    w_kl: {use: linear_warmup, start: 0.0, end: 1.0, steps: 100}",
            ),
        ],
    },
    "wgan_gp_d": {
        "params": {
            "generator": ("the generator, a reference such as \"@models.generator\"; called as generator(z) or "
                          "generator(z, label)"),
            "critic": ("the critic, a reference such as \"@models.critic\"; called as critic(images) or "
                       "critic(images, label)"),
            "latent": f"{LATENT}, drawn from the noise stream",
            "real": "the batch field of the real images",
            "label": "the batch field of the class labels for a conditional pair; null for an unconditional one",
            "gp_weight": "the weight of the gradient penalty on images between the real and the fake ones",
        },
        "examples": [
            lines(
                "losses:",
                '  d: {use: wgan_gp_d, generator: "@models.generator", critic: "@models.critic", latent: 128,',
                "      real: image, label: label, gp_weight: 10}",
                '  g: {use: wgan_g, generator: "@models.generator", critic: "@models.critic", latent: 128,',
                "      label: label}",
            ),
        ],
    },
    "wgan_g": {
        "params": {
            "generator": ("the generator, a reference such as \"@models.generator\"; called as generator(z) or "
                          "generator(z, label)"),
            "critic": "the critic that scores the fresh fakes, a reference such as \"@models.critic\"",
            "latent": f"{LATENT}, drawn from the noise stream",
            "label": "the batch field of the class labels for a conditional pair; null for an unconditional one",
        },
        "examples": [
            lines(
                "losses:",
                '  g: {use: wgan_g, generator: "@models.generator", critic: "@models.critic", latent: 128}',
                "optimizers:",
                '  g: {use: adam, lr: 1.0e-4, betas: [0.0, 0.9], over: "@models.generator", minimizes: "@losses.g"}',
            ),
        ],
    },
    "ddpm": {
        "params": {
            "model": ("the denoiser, a reference such as \"@models.unet\"; it is called as model(x_t, t) and predicts "
                      "the noise"),
            "schedule": ("the beta schedule, {use: cosine_beta, steps: 1000} or {use: linear_beta, steps: 1000}; a "
                         "sampler reads it back as \"@losses.ddpm.schedule\""),
            "x": "the batch field that is noised",
        },
        "examples": [
            lines(
                "losses:",
                '  ddpm: {use: ddpm, model: "@models.unet", x: image, schedule: {use: cosine_beta, steps: 300}}',
            ),
        ],
    },
    "distill": {
        "params": {
            "student": "the model being trained, a reference such as \"@models.student\"",
            "teacher": ("the guiding model, a reference; it runs without gradients, so freeze it and load its weights "
                        "with the weights key"),
            "x": "the batch field fed to both models; null takes the first input of the data",
            "label": "the batch field of the true classes for the cross entropy term; null takes the only target",
            "temperature": "softens both distributions; the KL term is scaled by temperature squared",
            "alpha": "the weight of the KL term; 1 - alpha weighs the cross entropy",
        },
        "examples": [
            lines(
                "models:",
                "  teacher:",
                "    use: distill_legos.Classifier",
                "    width: 32",
                '    n_classes: "@data.n_classes"',
                "    weights: {run: runs/teacher, model: net, which: best}",
                "    freeze: true",
                '  student: {use: distill_legos.Classifier, width: 8, n_classes: "@data.n_classes"}',
                "losses:",
                '  kd: {use: distill, student: "@models.student", teacher: "@models.teacher", temperature: 4.0,',
                "       alpha: 0.7}",
            ),
        ],
    },
    "ntxent": {
        "params": {
            "encoder": "the model that embeds every view, a reference such as \"@models.encoder\"",
            "projector": "a head applied after the encoder, a reference; null uses the encoder output",
            "views": ("the batch field with two views per item, (B, 2, C, H, W), as two_views in the train transform "
                      "makes them"),
            "temperature": "divides the cosine similarities; a lower value sharpens the contrast",
        },
        "examples": [
            lines(
                "losses:",
                '  ntxent: {use: ntxent, encoder: "@models.encoder", projector: "@models.projector", views: image,',
                "           temperature: 0.5, on: [train]}",
            ),
        ],
    },
}


METRICS = {
    "rmse": {
        "examples": [
            lines(
                "metrics:",
                "  rmse: rmse",
            ),
            lines(
                "metrics:",
                "  rmse_yield: {use: rmse, pred: y_yield, target: yield}",
            ),
        ],
    },
    "mae": {
        "examples": [
            lines(
                "metrics:",
                "  mae: mae",
            ),
            lines(
                "metrics:",
                "  recon: {use: mae, pred: x_hat, target: image}",
            ),
        ],
    },
    "accuracy": {
        "params": {
            "num_classes": COUNTED,
            "average": ("micro is the share of rows predicted right; macro is the mean recall over the classes present "
                        "in the targets"),
        },
        "examples": [
            lines(
                "metrics:",
                "  accuracy: accuracy",
                "  balanced: {use: accuracy, average: macro}",
            ),
        ],
    },
    "f1": {
        "params": {
            "num_classes": f"{COUNTED}; with num_classes, a class that never appears scores 0 under macro",
            "average": ("macro is the mean f1 of the classes, micro the share of rows right, weighted the mean "
                        "weighted by support, binary the f1 of class 1"),
        },
        "examples": [
            lines(
                "metrics:",
                '  f1: {use: f1, num_classes: "@data.n_classes", average: macro}',
            ),
        ],
    },
    "auroc": {
        "params": {"num_classes": SCORED},
        "examples": [
            lines(
                "metrics:",
                "  auroc: {use: auroc, pred: score, target: is_anomaly}",
            ),
        ],
    },
    "average_precision": {
        "params": {"num_classes": SCORED},
        "examples": [
            lines(
                "metrics:",
                "  average_precision: {use: average_precision, pred: score, target: is_anomaly}",
            ),
        ],
    },
    "perplexity": {
        "params": {"classes_last": CLASSES_LAST},
        "examples": [
            lines(
                "metrics:",
                "  ppl: {use: perplexity, target: targets}",
            ),
        ],
    },
    "fid": {
        "params": {
            "model": ("the generator, a reference such as \"@models.generator.ema\"; called as model(z) or model(z, "
                      "label)"),
            "latent": f"{LATENT}, drawn from the fid stream",
            "real": "the batch field of the real images",
            "label": "the batch field of the class labels for a conditional generator; null for an unconditional one",
            "n": "the real images, and as many fakes, compared at every evaluation",
            "extractor": ("the import path of a function from images to features; null pools the pixels to 16 by 16, "
                          "which is cheap but cruder than an inception network"),
            "batch": "the generator batch size while the fakes are made",
        },
        "examples": [
            lines(
                "metrics:",
                '  fid: {use: fid, model: "@models.generator.ema", latent: 128, real: image, label: label, n: 256,',
                "        every: 5, on: [valid]}",
            ),
        ],
    },
}


OPTIMIZERS = {
    "adam": {
        "params": {
            "lr": LR,
            "betas": f"{BETAS}; GANs often take [0.0, 0.9] or [0.5, 0.999]",
            "eps": EPS,
            "weight_decay": "an L2 penalty added to the gradient; adamw decouples it instead",
            "amsgrad": "keeps the largest squared gradient average seen, the AMSGrad variant",
            "maximize": MAXIMIZE,
            "decoupled_weight_decay": "true applies weight_decay the AdamW way",
        },
        "examples": [
            lines(
                "optimizers:",
                '  main: {use: adam, lr: 1.0e-3, over: "@models.net", minimizes: "@losses.mse"}',
            ),
            lines(
                "optimizers:",
                '  d: {use: adam, lr: 1.0e-4, betas: [0.0, 0.9], over: "@models.critic", minimizes: "@losses.d"}',
                '  g: {use: adam, lr: 1.0e-4, betas: [0.0, 0.9], over: "@models.generator", minimizes: "@losses.g"}',
            ),
        ],
    },
    "adamw": {
        "params": {
            "lr": LR,
            "betas": f"{BETAS}; language models often take [0.9, 0.95]",
            "eps": EPS,
            "weight_decay": "the decoupled weight decay, applied to the weights apart from the gradient step",
            "amsgrad": "keeps the largest squared gradient average seen, the AMSGrad variant",
            "maximize": MAXIMIZE,
        },
        "examples": [
            lines(
                "optimizers:",
                "  main:",
                "    use: adamw",
                "    lr: 1.0e-3",
                "    weight_decay: 1.0e-2",
                "    over:",
                '      - {of: "@models.net", match: "backbone.*", lr: 1.0e-4}',
                '      - "@models.net"',
                '    minimizes: "@losses.ce"',
                "    schedule: {use: cosine, steps: 5000, warmup: 200}",
            ),
        ],
    },
    "sgd": {
        "params": {
            "lr": LR,
            "momentum": "the momentum factor; 0 is plain gradient descent",
            "dampening": "damps the gradient added to the momentum",
            "weight_decay": "an L2 penalty added to the gradient",
            "nesterov": "Nesterov momentum; needs momentum above 0 and dampening 0",
            "maximize": MAXIMIZE,
        },
        "examples": [
            lines(
                "optimizers:",
                "  main:",
                "    use: sgd",
                "    lr: 0.1",
                "    momentum: 0.9",
                "    nesterov: true",
                "    weight_decay: 5.0e-4",
                '    over: "@models.net"',
                '    minimizes: "@losses.ce"',
                "    schedule: {use: cosine, steps: 360}",
            ),
        ],
    },
}


SCHEDULES = {
    "cosine": {
        "params": {
            "steps": "the update where the decay reaches floor, counted from the start (the warmup included)",
            "warmup": "the updates of the linear rise from 0 to 1 before the decay",
            "floor": "the factor at the end of the decay and after it",
        },
        "examples": [
            lines(
                "optimizers:",
                "  main:",
                "    use: adamw",
                "    lr: 6.0e-4",
                '    over: "@models.lm"',
                '    minimizes: "@losses.ce"',
                "    schedule: {use: cosine, warmup: 50, steps: 1000, floor: 0.1}",
            ),
        ],
    },
    "warmup": {
        "params": {"steps": "the updates of the linear rise from 0 to 1; the factor stays 1 afterwards"},
        "examples": [
            lines(
                "optimizers:",
                "  main:",
                "    use: adamw",
                "    lr: 2.0e-4",
                '    over: "@models.unet"',
                '    minimizes: "@losses.ddpm"',
                "    schedule: {use: warmup, steps: 80}",
            ),
        ],
    },
    "linear_warmup": {
        "params": {
            "start": "the value at update 0",
            "end": "the value reached at steps updates and kept afterwards",
            "steps": "the updates of the linear move from start to end",
        },
        "examples": [
            lines(
                "losses:",
                "  elbo:",
                "    use: vae_elbo",
                '    recon: "@losses.mse"',
                "    w_kl: {use: linear_warmup, start: 0.0, end: 1.0, steps: 1000}",
            ),
            lines(
                "optimizers:",
                "  main:",
                "    use: adam",
                "    lr: 1.0e-3",
                '    over: "@models.net"',
                '    minimizes: "@losses.mse"',
                "    schedule: {use: linear_warmup, start: 1.0, end: 0.1, steps: 5000}",
            ),
        ],
    },
    "step_decay": {
        "params": {
            "every": "the updates between two decays",
            "gamma": "the factor applied at every decay",
        },
        "examples": [
            lines(
                "optimizers:",
                "  main:",
                "    use: sgd",
                "    lr: 0.1",
                "    momentum: 0.9",
                '    over: "@models.net"',
                '    minimizes: "@losses.ce"',
                "    schedule: {use: step_decay, every: 2000, gamma: 0.5}",
            ),
        ],
    },
}


STEPS = {
    "supervised": {
        "params": {"order": ORDER, "amp": AMP, "grad_clip": GRAD_CLIP, "accumulate": ACCUMULATE},
        "examples": [
            lines(
                "train:",
                "  epochs: 20",
                "  step: {use: supervised, amp: true, grad_clip: 1.0}",
            ),
            lines(
                "train:",
                "  updates: {total: 1000, turn: 100}",
                "  step: {use: supervised, accumulate: 4}",
            ),
        ],
    },
    "alternating": {
        "params": {
            "order": ORDER,
            "repeat": ("a mapping of optimizer names to the steps each takes per round, such as {d: 5, g: 1}; a "
                       "missing name takes 1"),
            "fresh_batch": "true gives every step a new batch; false shares one batch per round",
            "amp": AMP,
            "grad_clip": GRAD_CLIP,
            "accumulate": ACCUMULATE,
        },
        "examples": [
            lines(
                "train:",
                "  epochs: 60",
                "  step: {use: alternating, order: [d, g], repeat: {d: 5, g: 1}, fresh_batch: true}",
            ),
        ],
    },
}


TRIGGERS = {
    "after_epoch": {
        "params": {"at": "the turn number from which it fires; absolute, so a resumed run counts on"},
        "examples": [
            lines(
                "rules:",
                "  warm:",
                "    when: {use: after_epoch, at: 5}",
                "    set: {optimizers.main.lr: {times: 0.5}}",
            ),
        ],
    },
    "every": {
        "params": {"epochs": "fires on every turn whose number is a multiple of epochs; pair it with once: false"},
        "examples": [
            lines(
                "rules:",
                "  samples:",
                "    when: {use: every, epochs: 5}",
                "    once: false",
                "    run: {use: write_samples, n: 16}",
            ),
        ],
    },
    "plateau": {
        "params": {
            "watch": WATCH,
            "patience": "the turns without an improvement before it fires",
            "mode": "min when lower is better, max when higher is better",
            "min_delta": "the smallest change that counts as an improvement",
        },
        "examples": [
            lines(
                "rules:",
                "  stop:",
                "    when: {use: plateau, watch: val/rmse, patience: 8}",
                "    stop: true",
            ),
            lines(
                "rules:",
                "  stop:",
                "    when: {use: plateau, watch: val/f1, mode: max, patience: 8, min_delta: 0.001}",
                "    stop: true",
            ),
        ],
    },
    "above": {
        "params": {"watch": WATCH, "value": "the threshold the series must be strictly above"},
        "examples": [
            lines(
                "rules:",
                "  cool_down:",
                "    when: {use: above, watch: val/accuracy, value: 0.9}",
                "    set: {optimizers.main.lr: 1.0e-4}",
            ),
        ],
    },
    "below": {
        "params": {"watch": WATCH, "value": "the threshold the series must be strictly below"},
        "examples": [
            lines(
                "rules:",
                "  to_huber:",
                "    when: {use: below, watch: train/mse, value: 0.05}",
                '    set: {optimizers.main.minimizes: "@losses.huber"}',
            ),
        ],
    },
    "time_budget": {
        "params": {"minutes": "the wall clock minutes since the first turn that asked it"},
        "examples": [
            lines(
                "rules:",
                "  budget:",
                "    when: {use: time_budget, minutes: 90}",
                "    stop: true",
            ),
        ],
    },
}


ACTIONS = {
    "write_samples": {
        "params": {
            "sampler": "a sampler call or \"@generate\"; null uses the sampler of the generate section",
            "n": "the samples drawn at every firing, written to samples/turn_<k>",
        },
        "examples": [
            lines(
                "rules:",
                "  samples:",
                "    when: {use: every, epochs: 5}",
                "    once: false",
                '    run: {use: write_samples, sampler: "@generate", n: 16}',
            ),
        ],
    },
}


LOGGERS = {
    "tensorboard": {
        "params": {
            "dir": ("the directory of the event files; null writes to <record>/tensorboard (a run without a "
                    "record needs it)"),
            "updates": ("true writes loss/, lr/ and grad_norm/ of every update against the update count; false writes "
                        "every series once per turn"),
            "flush_every": "flushes the event file every k turns",
        },
        "examples": [
            lines(
                "train:",
                "  epochs: 20",
                "  log: [tensorboard]",
            ),
            lines(
                "train:",
                "  epochs: 20",
                "  log: [{use: tensorboard, updates: false, flush_every: 5}]",
            ),
        ],
    },
    "echo": {
        "params": {
            "series": "a series name or a list of them; null prints every numeric series of the turn",
            "stream": LOCAL,
        },
        "examples": [
            lines(
                "train:",
                "  epochs: 20",
                "  log: [{use: echo, series: [val/rmse, lr/main]}]",
            ),
        ],
    },
}


SAMPLERS = {
    "gan_sampler": {
        "params": {
            "model": ("the generator, a reference such as \"@models.generator.ema\"; called as model(z) or "
                      "model(z, label)"),
            "latent": f"{LATENT}, drawn from the samples stream",
            "n_classes": ("labels 0 to n_classes - 1 cycle over the samples and go in as model(z, label); null for an "
                          "unconditional generator"),
            "n": SAMPLES,
        },
        "examples": [
            lines(
                'generate: {use: gan_sampler, model: "@models.generator.ema", latent: 128,',
                '           n_classes: "@data.n_classes", n: 64}',
            ),
        ],
    },
    "latent_sampler": {
        "params": {
            "model": "the decoder, a reference such as \"@models.ae.decoder\"",
            "latent": "the width of the standard normal latents fed to the decoder",
            "n": SAMPLES,
        },
        "examples": [
            lines(
                'generate: {use: latent_sampler, model: "@models.ae.decoder", latent: 32, n: 64}',
            ),
        ],
    },
    "ddpm_sampler": {
        "params": {
            "model": "the denoiser, called as model(x_t, t); often the EMA copy, \"@models.unet.ema\"",
            "schedule": "the beta schedule of the training loss, \"@losses.ddpm.schedule\"",
            "shape": "the shape of one sample, [C, H, W]",
            "n": SAMPLES,
            "steps": ("null walks every step of the schedule with fresh noise; a number takes that many evenly spaced "
                      "deterministic steps, which is faster"),
        },
        "examples": [
            lines(
                "generate:",
                "  use: ddpm_sampler",
                '  model: "@models.unet.ema"',
                '  schedule: "@losses.ddpm.schedule"',
                "  shape: [3, 16, 16]",
                "  n: 64",
                "  steps: 50",
            ),
        ],
    },
    "lm_sampler": {
        "params": {
            "model": "the language model, a reference; it returns logits (batch, tokens, vocabulary)",
            "tokenizer": ("the tokenizer that encodes the prompt and decodes the draws, \"@data.tokenizer\"; null "
                          "falls back to the tokenizer of the data"),
            "prompt": "the text to continue; empty starts from token 0",
            "max_new_tokens": "the tokens drawn after the prompt",
            "temperature": "divides the logits before the draw; 0 picks the most likely token every time",
            "top_k": "keeps only the k most likely tokens at every draw; null keeps them all",
            "n": "the texts drawn",
        },
        "examples": [
            lines(
                "generate:",
                "  use: lm_sampler",
                '  model: "@models.lm"',
                '  tokenizer: "@data.tokenizer"',
                '  prompt: "ROMEO:"',
                "  max_new_tokens: 400",
                "  temperature: 0.8",
                "  top_k: 20",
            ),
        ],
    },
}


PLOTS = {
    "loss_curve": {
        "params": {"series": "a series name or a list of them; null draws every train/ and val/ series"},
        "examples": [
            lines(
                "plots:",
                "  loss_curve: loss_curve",
            ),
            lines(
                "plots:",
                "  loss_curve: {use: loss_curve, series: [train/elbo/recon, train/elbo/kl, val/recon_rmse]}",
            ),
        ],
    },
    "pred_vs_true": {
        "params": {
            "pairs": ("a mapping of outputs (or pred_ columns) to target columns; null draws every numeric pair it "
                      "finds"),
        },
        "examples": [
            lines(
                "plots:",
                "  pred_vs_true: pred_vs_true",
            ),
            lines(
                "plots:",
                "  pred_vs_true: {use: pred_vs_true, pairs: {y_yield: yield, y_moisture: moisture}}",
            ),
        ],
    },
    "confusion_matrix": {
        "params": {"pred": PLOT_PRED, "target": PLOT_TARGET},
        "examples": [
            lines(
                "plots:",
                "  confusion: confusion_matrix",
            ),
            lines(
                "plots:",
                "  confusion: {use: confusion_matrix, pred: net, target: churned}",
            ),
        ],
    },
    "class_histogram": {
        "params": {
            "pred": PLOT_PRED,
            "target": PLOT_TARGET,
            "bins": "the number of score bins; probabilities are binned over [0, 1]",
        },
        "examples": [
            lines(
                "plots:",
                "  scores: {use: class_histogram, bins: 30}",
            ),
        ],
    },
    "binary_roc": {
        "params": {"pred": PLOT_PRED, "target": PLOT_TARGET},
        "examples": [
            lines(
                "plots:",
                "  roc: binary_roc",
            ),
        ],
    },
    "binary_precision_recall_curve": {
        "params": {"pred": PLOT_PRED, "target": PLOT_TARGET},
        "examples": [
            lines(
                "plots:",
                "  pr: {use: binary_precision_recall_curve, pred: net, target: churned}",
            ),
        ],
    },
    "forecast_samples": {
        "params": {
            "n": "the windows drawn, spread evenly over the predicted rows",
            "pred": PLOT_PRED,
            "target": PLOT_TARGET,
        },
        "examples": [
            lines(
                "plots:",
                "  forecast: {use: forecast_samples, n: 4}",
            ),
        ],
    },
    "image_pairs": {
        "params": {
            "n": "the images drawn, each above its reconstruction",
            "set": "the set the images come from: test, valid or train",
            "ctx": "the run context, which kalfa passes itself during a run and in kalfa plots; never write it",
        },
        "examples": [
            lines(
                "plots:",
                "  reconstructions: {use: image_pairs, n: 8, set: valid}",
            ),
        ],
    },
    "samples_grid": {
        "params": {
            "n": "the images in the grid",
            "file": ("the name under samples/: samples (written by generate) or turn_<k> (written by write_samples), "
                     "a .pt or a .txt file"),
        },
        "examples": [
            lines(
                "plots:",
                "  samples: {use: samples_grid, n: 64}",
            ),
            lines(
                "plots:",
                "  turn_10: {use: samples_grid, n: 16, file: turn_10}",
            ),
        ],
    },
}


STRATEGIES = {
    "grid": {
        "params": {
            "steps": ("the points of every range entry ({float: [a, b]} evenly, {log: [a, b]} geometric, {int: [a, "
                      "b]}); null allows lists only"),
        },
        "examples": [
            lines(
                "params: {lr: 1.0e-3, width: 128}",
                "sweep:",
                "  strategy: grid",
                "  space:",
                "    lr: [1.0e-2, 1.0e-3, 1.0e-4]",
                "    width: [64, 128, 256]",
                "  objective: {watch: val/rmse, mode: min, at: best}",
                "  record: runs/sweep_housing",
            ),
            lines(
                "params: {lr: 1.0e-3}",
                "sweep:",
                "  strategy: {use: grid, steps: 4}",
                "  space:",
                "    lr: {log: [1.0e-4, 1.0e-1]}",
                "  objective: {watch: val/rmse, mode: min}",
                "  record: runs/sweep_lr",
            ),
        ],
    },
    "random_search": {
        "params": {
            "trials": "the number of points; a point drawn twice is drawn again",
            "seed": "the seed of numpy's default_rng; null takes the top level seed",
        },
        "examples": [
            lines(
                "params: {lr: 1.0e-3, width: 128}",
                "sweep:",
                "  strategy: {use: random_search, trials: 12, seed: 1}",
                "  space:",
                "    lr: {log: [1.0e-4, 1.0e-2]}",
                "    width: {int: [16, 256]}",
                "  objective: {watch: val/rmse, mode: min, at: best}",
                "  record: runs/sweep_housing_random",
            ),
        ],
    },
    "optuna": {
        "params": {
            "trials": "the number of points, asked one after another",
            "sampler": "tpe (adaptive, learns from the finished points) or random",
            "seed": "the seed of the sampler; null takes the top level seed",
        },
        "examples": [
            lines(
                "params: {lr: 1.0e-3, width: 128}",
                "sweep:",
                "  strategy: {use: optuna, trials: 12, sampler: tpe}",
                "  space:",
                "    lr: {log: [1.0e-4, 1.0e-2]}",
                "    width: {int: [16, 256]}",
                "  objective: {watch: val/rmse, mode: min, at: best}",
                "  record: runs/sweep_housing_optuna",
            ),
        ],
    },
}


def stacked(*layers):
    return lines("models:", "  net:", "    use: stack", "    layers:", *[f"      - {layer}" for layer in layers])


LAYERS = {
    "linear": {
        "params": {
            "out_features": "the output width; the input width is inferred on the first forward pass",
            "bias": "adds a learned bias",
            "device": DEVICE,
            "dtype": DTYPE,
        },
        "examples": [stacked("{use: linear, out_features: 64}", "relu", "{use: linear, out_features: 1}")],
    },
    "conv2d": {
        "params": {
            "out_channels": "the output channels; the input channels are inferred on the first forward pass",
            "kernel_size": "the kernel size, an int or [height, width]",
            "stride": "the step of the kernel, an int or [height, width]",
            "padding": "zeros added on every side, an int, [height, width], same or valid",
            "dilation": "the spacing between kernel elements",
            "groups": "splits the channels into groups convolved apart",
            "bias": "adds a learned bias",
            "padding_mode": "zeros, reflect, replicate or circular",
            "device": DEVICE,
            "dtype": DTYPE,
        },
        "examples": [stacked("{use: conv2d, out_channels: 16, kernel_size: 3, padding: 1}", "relu",
                             "{use: torch.nn.MaxPool2d, kernel_size: 2}", "flatten",
                             '{use: linear, out_features: "@data.n_classes"}')],
    },
    "embedding": {
        "params": {
            "num_embeddings": "the vocabulary size, such as \"@data.vocab_size\"",
            "embedding_dim": "the width of every embedding vector",
            "padding_idx": "an id whose vector stays zero and gets no gradient",
            "max_norm": "renormalises the looked up vectors to at most this norm",
            "norm_type": "the p of the norm max_norm uses",
            "scale_grad_by_freq": "scales the gradients by the inverse frequency of the ids in the batch",
            "sparse": "gives sparse gradients, which only some optimizers take",
            "device": DEVICE,
            "dtype": DTYPE,
        },
        "examples": [lines(
            "models:",
            "  lm:",
            "    use: stack",
            "    layers:",
            '      - {use: embedding, num_embeddings: "@data.vocab_size", embedding_dim: 64}',
            '      - {use: linear, out_features: "@data.vocab_size"}',
        )],
    },
    "layer_norm": {
        "params": {
            "normalized_shape": "the trailing shape that is normalised, such as 64 or [64]",
            "eps": EPS,
            "elementwise_affine": "learns a scale and a shift per element",
            "bias": "learns the shift too when elementwise_affine is true",
            "device": DEVICE,
            "dtype": DTYPE,
        },
        "examples": [stacked("{use: linear, out_features: 64}", "{use: layer_norm, normalized_shape: 64}", "gelu",
                             "{use: linear, out_features: 1}")],
    },
    "batch_norm": {
        "params": {
            "eps": EPS,
            "momentum": "the update rate of the running mean and variance",
            "affine": "learns a scale and a shift per feature",
            "track_running_stats": "keeps running statistics for evaluation; false uses the batch statistics always",
            "bias": "learns the shift too when affine is true",
            "device": DEVICE,
            "dtype": DTYPE,
        },
        "examples": [stacked("{use: linear, out_features: 64}", "batch_norm", "relu",
                             "{use: linear, out_features: 1}")],
    },
    "dropout": {
        "params": {"p": "the probability that an element is zeroed in training", "inplace": INPLACE},
        "examples": [stacked("{use: linear, out_features: 128}", "relu", "{use: dropout, p: 0.2}",
                             "{use: linear, out_features: 1}")],
    },
    "flatten": {
        "params": {"start_dim": "the first dimension that is flattened",
                   "end_dim": "the last dimension that is flattened"},
        "examples": [stacked("{use: conv2d, out_channels: 8, kernel_size: 3}", "relu", "flatten",
                             '{use: linear, out_features: "@data.n_classes"}')],
    },
    "relu": {
        "params": {"inplace": INPLACE},
        "examples": [stacked("{use: linear, out_features: 64}", "relu", "{use: linear, out_features: 1}")],
    },
    "gelu": {
        "params": {"approximate": "none for the exact form, tanh for the faster approximation"},
        "examples": [
            stacked("{use: linear, out_features: 64}", "gelu", "{use: linear, out_features: 1}"),
            lines("models:", "  net: {use: mlp, hidden: [64, 64], out: 1, activation: gelu}"),
        ],
    },
    "silu": {
        "params": {"inplace": INPLACE},
        "examples": [stacked("{use: linear, out_features: 64}", "silu", "{use: linear, out_features: 1}")],
    },
    "tanh": {
        "examples": [stacked("{use: linear, out_features: 64}", "tanh", "{use: linear, out_features: 1}")],
    },
    "sigmoid": {
        "examples": [stacked("{use: linear, out_features: 64}", "relu", "{use: linear, out_features: 1}", "sigmoid")],
    },
    "last_step": {
        "examples": [stacked('{use: torch.nn.LSTM, input_size: "@data.n_features", hidden_size: 32, batch_first: true}',
                             "last_step", "{use: linear, out_features: 6}")],
    },
}


def table_with(features=None, targets="{price: standard_scaler}", split="[0.7, 0.15, 0.15]", file="housing.parquet",
               extra=()):
    rows = ["data:", "  use: table", f"  file: {file}", f"  split: {split}"]
    if features:
        rows.append(f"  features: {features}")
    rows.append(f"  targets: {targets}")
    return lines(*rows, *extra)


def images_with(*transform, folder="data/pets", key="transform"):
    rows = ["data:", "  use: image_folder", f"  folder: {folder}", "  split: [0.8, 0.1, 0.1]", f"  {key}:"]
    return lines(*rows, *[f"    - {step}" for step in transform])


PREPROCESSORS = {
    "standard_scaler": {
        "params": {
            "with_mean": "subtracts the mean of every column",
            "with_std": "divides by the standard deviation of every column",
            "copy": "lets scikit learn copy the values; leave it true",
        },
        "examples": [table_with('{"x*": standard_scaler}')],
    },
    "minmax_scaler": {
        "params": {
            "feature_range": ("the range the train values map onto, as [low, high]; kalfa passes the YAML list as the "
                              "tuple scikit learn wants"),
            "clip": "clips the values of later rows into feature_range",
            "copy": "lets scikit learn copy the values; leave it true",
        },
        "examples": [table_with('{"x*": standard_scaler, size: [log, minmax_scaler]}'),
                     table_with('{"x*": {use: minmax_scaler, feature_range: [-1, 1]}}')],
    },
    "robust_scaler": {
        "params": {
            "with_centering": "subtracts the median of every column",
            "with_scaling": "divides by the interquartile range of every column",
            "quantile_range": ("the lower and upper quantiles of the range, as [25.0, 75.0]; kalfa passes the YAML "
                               "list as the tuple scikit learn wants"),
            "unit_variance": "rescales so that normal data ends with variance 1",
            "copy": "lets scikit learn copy the values; leave it true",
        },
        "examples": [table_with('{"x*": robust_scaler}', "{price: robust_scaler}")],
    },
    "one_hot": {
        "examples": [table_with('{"num_*": standard_scaler, "cat_*": one_hot}', "{churned: label}",
                                file="churn.parquet")],
    },
    "label": {
        "examples": [table_with('{"num_*": standard_scaler, "cat_*": one_hot}', "{churned: label}",
                                file="churn.parquet")],
    },
    "abs": {
        "examples": [table_with('{"x*": [abs, {use: log, base: 10, norm: 1200}, standard_scaler]}', "{is_anomaly: ~}",
                                file="anomaly_data.parquet")],
    },
    "log": {
        "params": {
            "base": "the base of the logarithm, positive and not 1",
            "norm": ("divides the values before the log, log(1 + x / norm); positive, and the values must stay above "
                     "minus norm, so abs comes first for signed columns"),
        },
        "examples": [
            table_with('{"x*": standard_scaler, size: [log, minmax_scaler]}'),
            table_with('{"x*": [abs, {use: log, base: 10, norm: 1200}, standard_scaler]}', "{is_anomaly: ~}",
                       file="anomaly_data.parquet"),
        ],
    },
    "cast": {
        "params": {"dtype": "a numpy dtype name such as float32, int64 or bool"},
        "examples": [table_with('{"num_*": standard_scaler, "flag_*": {use: cast, dtype: float32}}',
                                "{sales: standard_scaler}", file="shop.parquet")],
    },
}


SPLITS = {
    "random": {
        "params": {"ratios": SPLIT_RATIOS, "seed": SPLIT_SEED},
        "examples": [table_with('{"x*": standard_scaler}', split="{use: random, ratios: [0.8, 0.1, 0.1], seed: 3}")],
    },
    "sequential": {
        "params": {
            "ratios": f"{SPLIT_RATIOS}; the rows are cut in their order",
            "group": ("a column; the rows of every value are cut in their order on their own, so every series has its "
                      "own train, valid and test part"),
        },
        "examples": [table_with('{"x*": standard_scaler}', "{load: standard_scaler}",
                                "{use: sequential, ratios: [0.7, 0.15, 0.15], group: site_id}", "energy.parquet")],
    },
    "kfold": {
        "params": {
            "folds": "the number of folds, at least 2",
            "fold": "the fold that is the test set, from 0 to folds - 1",
            "seed": SPLIT_SEED,
            "valid_ratio": "the share of the other folds that becomes the valid set, at least 0 and below 1",
        },
        "examples": [lines(
            "include: [../01_mlp_regression/config.yaml]",
            "params: {fold: 0}",
            "data:",
            "  split: {use: kfold, folds: 5, fold: $fold$, valid_ratio: 0.15}",
        )],
    },
    "given": {
        "params": {
            "column": "the column whose values name the set of every row",
            "train": GIVEN.format("train", "train"),
            "valid": GIVEN.format("valid", "valid"),
            "test": GIVEN.format("test", "test"),
        },
        "examples": [table_with('{"x*": standard_scaler}', split="{use: given, column: part, test: holdout}")],
    },
}


TRANSFORMS = {
    "to_tensor": {"examples": [images_with("{use: resize, size: 32}", "to_tensor")]},
    "normalize": {
        "params": {
            "mean": ("a number, a list per channel or a preset name (imagenet, cifar10, mnist); a preset also gives "
                     "std unless std is set"),
            "std": "a number, a list per channel or a preset name; null takes the std of the preset, or 0.5",
        },
        "examples": [
            images_with("to_tensor", "{use: normalize, mean: imagenet}"),
            images_with("to_tensor", "{use: normalize, mean: 0.5, std: 0.5}", folder="data/faces"),
        ],
    },
    "resize": {
        "params": {"size": f"the new size, {SIZE}"},
        "examples": [images_with("{use: resize, size: [32, 32]}", "to_tensor")],
    },
    "center_crop": {
        "params": {"size": f"the crop size, {SIZE}; a crop larger than the image pads with zeros"},
        "examples": [images_with("{use: resize, size: 36}", "{use: center_crop, size: 32}", "to_tensor")],
    },
    "random_crop": {
        "params": {
            "size": f"the crop size, {SIZE}",
            "padding": "zero pixels added on every side before a random window is cut",
        },
        "examples": [images_with("{use: random_crop, size: 32, padding: 4}", "random_flip", "to_tensor",
                                 key="train_transform")],
    },
    "random_flip": {
        "params": {"p": "the probability of a horizontal flip"},
        "examples": [images_with("{use: random_crop, size: 32, padding: 4}", "{use: random_flip, p: 0.5}", "to_tensor",
                                 key="train_transform")],
    },
    "grayscale": {
        "examples": [images_with("grayscale", "to_tensor", "{use: normalize, mean: mnist}", folder="data/digits")],
    },
    "two_views": {
        "params": {"each": ("the transform list applied twice with fresh randomness; each view becomes a float tensor "
                             "(C, H, W)")},
        "examples": [lines(
            "data:",
            "  use: image_folder",
            "  folder: data/unlabeled",
            "  split: [0.8, 0.1, 0.1]",
            "  labels: false",
            "  train_transform:",
            "    - use: two_views",
            "      each: [{use: random_crop, size: 16, padding: 2}, random_flip, to_tensor]",
            "  transform: [to_tensor]",
        )],
    },
}


OTHERS = {
    "window": {
        "params": {
            "size": "the rows of a window: x is a (size, features) tensor",
            "horizon": "the rows after the window that every target covers",
            "stride": "the rows between the starts of two windows",
            "context": ("true lets the first windows of the valid and test sets reach back into the rows of the set "
                        "before"),
            "group": "a column whose value every window keeps to, one series per value; it is not a feature",
        },
        "examples": [table_with('{"x*": standard_scaler}', "{load: standard_scaler}",
                                "{use: sequential, ratios: [0.7, 0.15, 0.15], group: site_id}", "energy.parquet",
                                ["  feed: {use: window, size: 168, horizon: 24, context: true, group: site_id}"])],
    },
    "char_tokenizer": {
        "params": {
            "lowercase": ("lowercases the text before it is encoded; as a bare name the tokenizer takes the lowercase "
                          "of text_lines"),
        },
        "examples": [lines(
            "data:",
            "  use: text_lines",
            "  file: data/corpus.txt",
            "  tokenizer: {use: char_tokenizer, lowercase: true}",
            "  seq_len: 64",
            "  split: [0.9, 0.05, 0.05]",
        )],
    },
    "normal_init": {
        "params": {
            "std": "the standard deviation of the weights",
            "mean": "the mean of the weights",
        },
        "examples": [lines(
            "models:",
            "  generator:",
            "    use: gan_legos.Generator",
            "    latent: 128",
            "    init: {use: normal_init, std: 0.02}",
        )],
    },
    "xavier_init": {
        "params": {"gain": "scales the uniform bound; 1.0 suits linear and tanh layers, about 1.41 suits relu"},
        "examples": [
            lines("models:", "  net: {use: mlp, hidden: [64, 64], out: 1, init: xavier_init}"),
            lines("models:", "  net: {use: mlp, hidden: [64, 64], out: 1, init: {use: xavier_init, gain: 1.41}}"),
        ],
    },
    "cosine_beta": {
        "params": {
            "steps": "the diffusion steps T",
            "s": "the small offset of the cosine curve that keeps the first betas from vanishing",
        },
        "examples": [lines(
            "losses:",
            '  ddpm: {use: ddpm, model: "@models.unet", x: image, schedule: {use: cosine_beta, steps: 1000}}',
        )],
    },
    "linear_beta": {
        "params": {
            "steps": "the diffusion steps T",
            "start": "the beta of the first step, inside (0, 1)",
            "end": "the beta of the last step, inside (0, 1)",
        },
        "examples": [lines(
            "losses:",
            "  ddpm:",
            "    use: ddpm",
            '    model: "@models.unet"',
            "    schedule: {use: linear_beta, steps: 1000, start: 1.0e-4, end: 0.02}",
        )],
    },
}


NOTES = {**DATA, **MODELS, **LOSSES, **METRICS, **OPTIMIZERS, **SCHEDULES, **STEPS, **TRIGGERS, **ACTIONS, **LOGGERS,
         **SAMPLERS, **PLOTS, **STRATEGIES, **LAYERS, **PREPROCESSORS, **SPLITS, **TRANSFORMS, **OTHERS}


def apply():
    for name, note in NOTES.items():
        if name in aliases.ALIASES:
            aliases.annotate(name, params=note.get("params"), examples=note.get("examples"))


apply()
