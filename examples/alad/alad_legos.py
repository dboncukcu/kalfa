from pathlib import Path

import numpy
import pandas
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from torch import nn
from torch.nn import functional

import kalfa


class Mlp(nn.Module):
    def __init__(self, in_features, hidden, out_features, skip=False):
        super().__init__()
        widths = [int(in_features), *(int(width) for width in hidden)]
        self.layers = nn.ModuleList(nn.Linear(before, after) for before, after in zip(widths, widths[1:]))
        self.skips = [bool(skip) and before == after for before, after in zip(widths, widths[1:])]
        self.out = nn.Linear(widths[-1], int(out_features))

    def forward(self, x):
        for layer, skip in zip(self.layers, self.skips):
            hidden = functional.leaky_relu(layer(x), 0.2)
            x = x + hidden if skip else hidden
        return self.out(x)


class Dxz(nn.Module):
    def __init__(self, x_dim, z_dim, width=32):
        super().__init__()
        self.x_branch = nn.Sequential(nn.Linear(x_dim, width), nn.LeakyReLU(0.2))
        self.z_branch = nn.Sequential(nn.Linear(z_dim, width), nn.LeakyReLU(0.2))
        self.joint = nn.Sequential(nn.Linear(2 * width, width), nn.LeakyReLU(0.2), nn.Linear(width, 1))

    def forward(self, x, z):
        return self.joint(torch.cat([self.x_branch(x), self.z_branch(z)], dim=1))


class Dxx(nn.Module):
    def __init__(self, x_dim, width=64):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(2 * x_dim, width), nn.LeakyReLU(0.2))
        self.head = nn.Linear(width, 1)

    def forward(self, x, other):
        feature = self.body(torch.cat([x, other], dim=1))
        return self.head(feature), feature


class Dzz(nn.Module):
    def __init__(self, z_dim, width=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * z_dim, width), nn.LeakyReLU(0.2), nn.Linear(width, 1))

    def forward(self, z, other):
        return self.net(torch.cat([z, other], dim=1))


class AnomalyScore(nn.Module):
    def __init__(self, encoder, generator, dxx):
        super().__init__()
        self.encoder = encoder
        self.generator = generator
        self.dxx = dxx

    def forward(self, x):
        rebuilt = self.generator(self.encoder(x))
        _, real = self.dxx(x, x)
        _, fake = self.dxx(x, rebuilt)
        return (real - fake).abs().sum(dim=1)


def bce(logits, real):
    target = torch.ones_like(logits) if real else torch.zeros_like(logits)
    return functional.binary_cross_entropy_with_logits(logits, target)


class Alad(kalfa.Objective):
    on = ("train",)

    def __init__(self, encoder, generator, dxz, dxx, dzz, latent, x="x"):
        super().__init__()
        self.encoder = encoder
        self.generator = generator
        self.dxz = dxz
        self.dxx = dxx
        self.dzz = dzz
        self.latent = int(latent)
        self.x = x

    def draw(self, batch, ctx):
        x = batch[self.x]
        return x, ctx.rng.normal("noise", (len(x), self.latent), like=x)


class AladD(Alad):
    def forward(self, outputs, batch, ctx):
        x, z = self.draw(batch, ctx)
        with torch.no_grad():
            z_of_x = self.encoder(x)
            x_of_z = self.generator(z)
            x_again = self.generator(z_of_x)
            z_again = self.encoder(x_of_z)
        xz = bce(self.dxz(x, z_of_x), True) + bce(self.dxz(x_of_z, z), False)
        xx = bce(self.dxx(x, x)[0], True) + bce(self.dxx(x, x_again)[0], False)
        zz = bce(self.dzz(z, z), True) + bce(self.dzz(z, z_again), False)
        return {"loss": xz + xx + zz, "xz": xz, "xx": xx, "zz": zz}


class AladG(Alad):
    def forward(self, outputs, batch, ctx):
        x, z = self.draw(batch, ctx)
        z_of_x = self.encoder(x)
        x_of_z = self.generator(z)
        xz = bce(self.dxz(x_of_z, z), True) + bce(self.dxz(x, z_of_x), False)
        xx = bce(self.dxx(x, x)[0], False) + bce(self.dxx(x, self.generator(z_of_x))[0], True)
        zz = bce(self.dzz(z, z), False) + bce(self.dzz(z, self.encoder(x_of_z)), True)
        return {"loss": xz + xx + zz, "xz": xz, "cycle_x": xx, "cycle_z": zz}


def scored(record, pred, target):
    path = Path(record) / "predictions.parquet"
    if not path.is_file():
        return None
    frame = pandas.read_parquet(path)
    if pred not in frame.columns or target not in frame.columns:
        return None
    frame = frame[[pred, target]].dropna()
    scores = frame[pred].to_numpy(dtype="float64")
    truth = frame[target].to_numpy(dtype="int64")
    if len(truth) == 0 or truth.min() == truth.max():
        return None
    return scores, truth


def score_histogram(record, figures, pred="pred_score", target="is_anomaly", bins=30):
    found = scored(record, pred, target)
    if found is None:
        return None
    scores, truth = found
    edges = numpy.histogram_bin_edges(scores, bins=int(bins))
    figure = figures.new(8.0, 5.0)
    axis = figure.subplots()
    for position, (label, name) in enumerate(((0, "normal"), (1, "anomaly"))):
        chosen = scores[truth == label]
        axis.hist(chosen, bins=edges, alpha=0.45, color=figures.colors[position], label=f"{name} (n = {len(chosen)})")
    axis.set_xlabel("anomaly score")
    axis.set_ylabel("rows")
    axis.set_title(f"Anomaly score by true class (n = {len(scores)})", loc="left")
    axis.legend(frameon=False)
    return figure


def roc(record, figures, pred="pred_score", target="is_anomaly"):
    found = scored(record, pred, target)
    if found is None:
        return None
    scores, truth = found
    false_rate, true_rate, _ = roc_curve(truth, scores)
    area = roc_auc_score(truth, scores)
    figure = figures.new(5.6, 5.0)
    axis = figure.subplots()
    axis.plot(false_rate, true_rate, color=figures.colors[0], linewidth=2.0, label=f"score (AUC {area:.3f})")
    axis.plot([0.0, 1.0], [0.0, 1.0], color="#8c8b86", linewidth=1.0, linestyle="--", label="chance")
    axis.set_xlabel("false positive rate")
    axis.set_ylabel("true positive rate")
    axis.set_title(f"ROC of the anomaly score (AUC {area:.3f})", loc="left")
    axis.legend(loc="lower right", frameon=False)
    return figure


def pr_curve(record, figures, pred="pred_score", target="is_anomaly"):
    found = scored(record, pred, target)
    if found is None:
        return None
    scores, truth = found
    precision, recall, _ = precision_recall_curve(truth, scores)
    average = average_precision_score(truth, scores)
    rate = float(truth.mean())
    figure = figures.new(5.6, 5.0)
    axis = figure.subplots()
    axis.step(recall, precision, where="post", color=figures.colors[0], linewidth=2.0,
              label=f"score (AP {average:.3f})")
    axis.axhline(rate, color="#8c8b86", linewidth=1.0, linestyle="--", label=f"base rate {rate:.3f}")
    axis.set_xlabel("recall")
    axis.set_ylabel("precision")
    axis.set_title(f"Precision and recall of the anomaly score (AP {average:.3f})", loc="left")
    axis.legend(loc="upper right", frameon=False)
    return figure
