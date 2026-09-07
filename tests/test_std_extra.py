"""The legos tidy needs beyond 01: torchmetrics metrics, l1_distance, the score plots and the myexample plugin."""

import math
import warnings

import pandas
import pytest
import torch
from cirak.registry import registry
from torch import nn

import kalfa  # noqa: F401
from kalfa.std.adapter import metric as metric_adapter
from kalfa.std.layer import l1_distance
from kalfa.std.metric import binary_auroc, binary_average_precision
from kalfa.std.plot import architecture, binary_precision_recall_curve, binary_roc, class_histogram
from kalfa.std.runtime import Context


def test_torchmetrics_wrappers_and_nan_on_one_class():
    metric = binary_auroc()
    metric.update(torch.tensor([[0.1], [0.9], [0.8], [0.2]]), torch.tensor([0, 1, 1, 0]))
    assert metric.compute() == pytest.approx(1.0)
    metric.reset()
    metric.update(torch.tensor([0.1, 0.9]), torch.tensor([1, 1]))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = metric.compute()
    assert math.isnan(value) and len(caught) == 1
    assert math.isnan(metric.compute())
    precision = binary_average_precision()
    precision.update(torch.tensor([0.1, 0.9]), torch.tensor([0, 1]))
    assert precision.compute() == pytest.approx(1.0)
    tracker = metric_adapter(binary_auroc()).tracker("auroc")
    assert tracker.live is not tracker.adapter.metric


def test_l1_distance_per_sample():
    layer = l1_distance()
    out = layer(torch.tensor([[1.0, 2.0], [0.0, 0.0]]), torch.tensor([[0.0, 0.0], [2.0, 2.0]]))
    assert out.tolist() == [1.5, 2.0]


def test_score_plots_write_files(tmp_path):
    predictions = pandas.DataFrame({"row": [0, 1, 2, 3], "is_anomaly": [0, 1, 0, 1],
                                    "raw_score": [0.1, 0.8, 0.3, 0.9]})
    class_histogram(predictions, [], {}, str(tmp_path))
    binary_roc(predictions, [], {}, str(tmp_path))
    binary_precision_recall_curve(predictions, [], {}, str(tmp_path))
    architecture(predictions, [], {"m": nn.Linear(2, 1)}, str(tmp_path))
    for name in ("class_histogram.png", "binary_roc.png", "binary_precision_recall_curve.png", "architecture.txt"):
        assert (tmp_path / "plots" / name).exists(), name
    assert "Linear" in (tmp_path / "plots" / "architecture.txt").read_text()
    assert binary_roc(pandas.DataFrame({"row": [0], "is_anomaly": [1], "raw_s": [0.5]}), [], {}, str(tmp_path)) is None
    assert class_histogram(pandas.DataFrame(), [], {}, str(tmp_path)) is None


class Two(nn.Module):
    def __init__(self, width, outputs):
        super().__init__()
        self.layer = nn.Linear(width, 3)
        self.logit = nn.Linear(3, 1)
        self.outputs = outputs

    def forward(self, *parts):
        feature = torch.relu(self.layer(torch.cat(parts, dim=1)))
        logit = self.logit(feature)
        return (logit, feature) if len(self.outputs) == 2 else logit


def test_myexample_objectives_register_with_facts_and_run():
    import myexample  # noqa: F401
    from kalfa.std.criterion import bce_logits

    facts = registry.facts("/objective/myexample/alad_discriminator")
    assert facts.kind == "objective" and facts.partial and facts.refs == {"criterion": "criterion"}
    assert set(facts.needs_models) == {"encoder", "generator", "dxz", "dxx", "dzz"}
    torch.manual_seed(0)
    models = {"encoder": nn.Linear(6, 4), "generator": nn.Linear(4, 6), "dxz": Two(10, ["logit"]),
              "dxx": Two(12, ["logit", "feature"]), "dzz": Two(8, ["logit"])}
    batch = {"x": torch.randn(5, 6)}
    context = Context(batch, models, targets=[], rng=torch.Generator().manual_seed(1))
    disc = registry.resolve("/objective/myexample/alad_discriminator")(models, batch, criterion=bce_logits, latent_dim=4,
                                                                    rng=context.rng)
    gen = registry.resolve("/objective/myexample/alad_generator")(models, batch, criterion=bce_logits, latent_dim=4,
                                                               feature_matching=0.5, rng=context.rng)
    assert disc.shape == () and gen.shape == () and disc.requires_grad and gen.requires_grad
    disc.backward()
    assert models["encoder"].weight.grad is None and models["dxz"].layer.weight.grad is not None
