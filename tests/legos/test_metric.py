import copy
import math
import warnings

import pytest
import torch
from cirak.registry import registry
from torch import nn

from helpers import batch, build, tiny_model
from kalfa.std import STD_URIS
from kalfa.std.common.runtime import Context, Pass
from kalfa.std.metric.kalfa.fid import Fid
from kalfa.std.metric.kalfa.perplexity import Perplexity
from kalfa.std.metric.kalfa.recon_error import ReconError
from kalfa.std.metric.kalfa.rmse import Rmse
from kalfa.std.metric.kalfa.sample_writer import SampleWriter
from kalfa.std.metric.torchmetrics.base import ClassMetric, TorchMetric


METRICS = sorted(uri for uri in STD_URIS if uri.startswith("/metric/"))

REAL = torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]]).reshape(6, 1, 2, 2)


class Generator(nn.Module):
    def __init__(self, shift=0.0):
        super().__init__()
        self.shift = shift
        self.inputs = ["z"]
        self.outputs = ["image"]
        self.labels = None

    def forward(self, noise, labels=None):
        self.labels = labels
        return REAL[:len(noise)] + self.shift


def test_metric_scope_is_the_kalfa_and_torchmetrics_metrics():
    assert METRICS == ["/metric/kalfa/fid", "/metric/kalfa/perplexity", "/metric/kalfa/recon_error",
                       "/metric/kalfa/rmse", "/metric/kalfa/sample_writer", "/metric/torchmetrics/accuracy",
                       "/metric/torchmetrics/binary_auroc", "/metric/torchmetrics/binary_average_precision",
                       "/metric/torchmetrics/f1"]


def test_every_metric_carries_state_and_its_alias():
    for uri in METRICS:
        assert registry.facts(uri).state is True
    aliases = registry.aliases()
    assert aliases["rmse"] == "/metric/kalfa/rmse"
    assert aliases["recon_error"] == "/metric/kalfa/recon_error"
    assert aliases["perplexity"] == "/metric/kalfa/perplexity"
    assert aliases["fid"] == "/metric/kalfa/fid"
    assert aliases["sample_writer"] == "/metric/kalfa/sample_writer"
    assert aliases["accuracy"] == "/metric/torchmetrics/accuracy"
    assert aliases["f1"] == "/metric/torchmetrics/f1"
    assert aliases["auroc"] == "/metric/torchmetrics/binary_auroc"
    assert aliases["average_precision"] == "/metric/torchmetrics/binary_average_precision"


def test_rmse_accumulates_squared_errors_over_every_element():
    metric = build("/metric/kalfa/rmse")
    assert isinstance(metric, Rmse)
    assert math.isnan(metric.compute())
    metric.update(torch.tensor([[1.0, 1.0], [0.0, 0.0]]), torch.zeros(2, 2))
    assert metric.compute() == math.sqrt(0.5)
    metric.update(torch.tensor([[2.0]]), torch.tensor([0.0]))
    assert metric.compute() == math.sqrt(6.0 / 5.0)
    metric.reset()
    assert math.isnan(metric.compute())


def test_recon_error_averages_the_per_sample_mean_squared_error():
    metric = build("/metric/kalfa/recon_error")
    assert isinstance(metric, ReconError)
    metric.update(torch.tensor([[1.0, 1.0], [0.0, 0.0]]), torch.zeros(2, 2))
    assert metric.compute() == 0.5
    metric.update(torch.tensor([[2.0, 0.0]]), torch.zeros(1, 2))
    assert metric.compute() == 1.0
    metric.reset()
    assert math.isnan(metric.compute())


def test_perplexity_is_exp_of_the_mean_token_cross_entropy():
    metric = build("/metric/kalfa/perplexity")
    assert isinstance(metric, Perplexity)
    assert math.isnan(metric.compute())
    metric.update(torch.zeros(2, 4), torch.tensor([0, 3]))
    assert metric.compute() == pytest.approx(4.0)
    metric.update(torch.tensor([[math.log(3.0), 0.0, 0.0, 0.0]]), torch.tensor([0]))
    assert metric.compute() == pytest.approx(2.0 ** (5.0 / 3.0))
    metric.reset()
    metric.update(torch.zeros(1, 2, 4), torch.tensor([[1, 2]]))
    assert metric.compute() == pytest.approx(4.0)


def test_fid_of_a_shifted_copy_is_the_squared_shift_over_the_features():
    metric = build("/metric/kalfa/fid", model="g", latent=2, n=6, extractor=build("/lego/kalfa/pixel_features", size=2))
    assert isinstance(metric, Fid)
    assert math.isnan(metric.compute())
    metric.update({"g": Generator(shift=1.0)}, {"image": REAL}, rng=torch.Generator().manual_seed(0))
    assert metric.seen_real == 6 and metric.seen_fake == 6
    assert metric.compute() == pytest.approx(4.0, abs=1e-6)


def test_fid_of_identical_samples_is_zero():
    metric = build("/metric/kalfa/fid", model="g", latent=2, n=6, extractor=build("/lego/kalfa/pixel_features", size=2))
    metric.update({"g": Generator()}, {"image": REAL})
    assert metric.compute() == pytest.approx(0.0, abs=1e-6)
    metric.reset()
    assert metric.real == [] and metric.fake == [] and metric.seen_real == 0
    assert math.isnan(metric.compute())


def test_fid_stops_collecting_at_n_samples():
    metric = build("/metric/kalfa/fid", model="g", latent=2, n=4, extractor=build("/lego/kalfa/pixel_features", size=2))
    metric.update({"g": Generator()}, {"image": REAL})
    metric.update({"g": Generator()}, {"image": REAL})
    assert metric.seen_real == 4 and metric.seen_fake == 4
    assert sum(len(block) for block in metric.real) == 4
    assert sum(len(block) for block in metric.fake) == 4
    assert metric.fake[0].shape == (4, 4)


def test_fid_conditional_samples_use_the_batch_labels():
    generator = Generator()
    metric = build("/metric/kalfa/fid", model="g", latent=2, n=3, conditional=True,
                   extractor=build("/lego/kalfa/pixel_features", size=2))
    labels = torch.tensor([0, 1, 0, 1, 0, 1])
    metric.update({"g": generator}, {"image": REAL, "label": labels})
    assert torch.equal(generator.labels, labels[:3])


def test_fid_through_the_metric_tracker_reads_models_batch_and_rng():
    metric = build("/metric/kalfa/fid", model="g", latent=2, n=6, extractor=build("/lego/kalfa/pixel_features", size=2))
    adapter = build("/adapter/kalfa/metric", metric=metric)
    tracker = adapter.tracker("fid")
    scope = Pass(models={"g": Generator(shift=1.0)}, rng=torch.Generator().manual_seed(1))
    tracker.observe(Context({"image": REAL}, scope))
    assert tracker.result() == {"fid": pytest.approx(4.0, abs=1e-6)}


def test_sample_writer_writes_the_predicts_outputs_of_the_turn(tmp_path):
    metric = build("/metric/kalfa/sample_writer", n=2)
    assert isinstance(metric, SampleWriter)
    model = tiny_model()
    data = batch()
    metric.update({"m": model}, "m", None, str(tmp_path), 3, batch=data)
    saved = torch.load(tmp_path / "samples" / "turn_0003.pt", weights_only=False)
    with torch.no_grad():
        assert torch.equal(saved, model(data["x"])[:2])
    assert sorted(path.name for path in (tmp_path / "samples").iterdir()) == ["turn_0003.pt"]
    assert metric.compute() is None


def test_sample_writer_draws_image_outputs_as_a_grid(tmp_path):
    metric = build("/metric/kalfa/sample_writer", n=4)
    metric.update({"g": Generator()}, "g", None, str(tmp_path), 12, batch={"z": torch.zeros(6, 2)})
    assert sorted(path.name for path in (tmp_path / "samples").iterdir()) == ["turn_0012.png", "turn_0012.pt"]
    assert (tmp_path / "samples" / "turn_0012.png").stat().st_size > 0
    assert torch.load(tmp_path / "samples" / "turn_0012.pt", weights_only=False).shape == (4, 1, 2, 2)


def test_sample_writer_writes_once_per_pass(tmp_path):
    metric = build("/metric/kalfa/sample_writer", n=2)
    model = tiny_model()
    metric.update({"m": model}, "m", None, str(tmp_path), 1, batch=batch())
    metric.update({"m": model}, "m", None, str(tmp_path), 2, batch=batch())
    assert sorted(path.name for path in (tmp_path / "samples").iterdir()) == ["turn_0001.pt"]
    metric.reset()
    metric.update({"m": model}, "m", None, str(tmp_path), 2, batch=batch())
    assert sorted(path.name for path in (tmp_path / "samples").iterdir()) == ["turn_0001.pt", "turn_0002.pt"]


def test_sample_writer_uses_the_sampler_and_writes_text_as_txt(tmp_path):
    seen = {}

    def sampler(models, prep, rng, n):
        seen.update({"models": models, "prep": prep, "rng": rng, "n": n})
        return "a line of text"

    metric = build("/metric/kalfa/sample_writer", n=3, sampler=sampler)
    rng = torch.Generator()
    metric.update({"m": tiny_model()}, None, rng, str(tmp_path), 5, prep="prep")
    assert (tmp_path / "samples" / "turn_0005.txt").read_text() == "a line of text"
    assert seen["n"] == 3 and seen["rng"] is rng and seen["prep"] == "prep" and list(seen["models"]) == ["m"]


def test_sample_writer_without_a_record_or_a_source_of_samples(tmp_path):
    metric = build("/metric/kalfa/sample_writer")
    metric.update({"m": tiny_model()}, "m", None, None, 1, batch=batch())
    assert metric.done is False
    with pytest.raises(ValueError, match=r"sample_writer needs a sampler \(sampler: generate\) or a predicts model "
                                          r"and a batch"):
        metric.update({"m": tiny_model()}, None, None, str(tmp_path), 1)


def test_accuracy_counts_the_argmax_hits_over_the_labels():
    metric = build("/metric/torchmetrics/accuracy")
    assert isinstance(metric, ClassMetric)
    assert math.isnan(metric.compute())
    logits = torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, 2.0]])
    metric.update(logits, torch.tensor([0, 1, 2, 1]))
    assert metric.compute() == 0.75
    metric.update(logits, torch.tensor([0, 1, 2, 2]))
    assert metric.compute() == 0.875
    metric.reset()
    assert math.isnan(metric.compute())


def test_accuracy_thresholds_a_single_score_at_zero():
    metric = build("/metric/torchmetrics/accuracy")
    metric.update(torch.tensor([0.5, -0.5, 2.0]), torch.tensor([1, 0, 0]))
    assert metric.compute() == pytest.approx(2.0 / 3.0)
    column = build("/metric/torchmetrics/accuracy")
    column.update(torch.tensor([[0.5], [-0.5], [2.0]]), torch.tensor([[1], [0], [1]]))
    assert column.compute() == 1.0


def test_f1_is_macro_averaged_over_the_classes():
    metric = build("/metric/torchmetrics/f1")
    logits = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    metric.update(logits, torch.tensor([0, 1, 1, 1]))
    assert metric.compute() == pytest.approx(11.0 / 15.0)
    micro = build("/metric/torchmetrics/f1", average="micro")
    micro.update(logits, torch.tensor([0, 1, 1, 1]))
    assert micro.compute() == 0.75


def test_binary_auroc_is_the_share_of_ordered_positive_negative_pairs():
    metric = build("/metric/torchmetrics/binary_auroc")
    assert isinstance(metric, TorchMetric)
    metric.update(torch.tensor([0.1, 0.4]), torch.tensor([0, 0]))
    metric.update(torch.tensor([[0.35], [0.8]]), torch.tensor([1, 1]))
    assert metric.compute() == 0.75
    metric.reset()
    assert metric.classes == set()


def test_binary_average_precision_sums_precision_over_the_recall_steps():
    metric = build("/metric/torchmetrics/binary_average_precision")
    metric.update(torch.tensor([0.1, 0.4, 0.35, 0.8]), torch.tensor([0, 0, 1, 1]))
    assert metric.compute() == pytest.approx(5.0 / 6.0)


def test_torchmetric_is_nan_with_one_warning_on_a_single_class():
    metric = build("/metric/torchmetrics/binary_auroc")
    metric.update(torch.tensor([0.1, 0.4]), torch.tensor([1, 1]))
    with pytest.warns(UserWarning, match=r"auroc is undefined on a set with fewer than two classes; reported as NaN"):
        assert math.isnan(metric.compute())
    assert metric.warned is True
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert math.isnan(metric.compute())


def test_torchmetric_deepcopy_starts_a_fresh_metric_that_remembers_the_warning():
    metric = build("/metric/torchmetrics/binary_auroc")
    metric.update(torch.tensor([0.1, 0.4, 0.35, 0.8]), torch.tensor([0, 0, 1, 1]))
    metric.warned = True
    fresh = copy.deepcopy(metric)
    assert isinstance(fresh, TorchMetric)
    assert fresh.classes == set() and fresh.warned is True
    assert fresh.metric is not metric.metric
    assert metric.compute() == 0.75
