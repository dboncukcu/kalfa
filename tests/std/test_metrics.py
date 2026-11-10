import logging
import math
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.preprocessing import label_binarize
from torch import nn

from kalfa.core import aliases
from kalfa.core.errors import BuildError, RunError
from kalfa.core.protocols import ContextMetric
from kalfa.core.rng import Rng
from kalfa.std.metrics import (
    Fid,
    accuracy,
    auroc,
    average_precision,
    f1,
    frechet_distance,
    mae,
    perplexity,
    pixels,
    rmse,
)


PRED = torch.tensor([[0.5], [1.5], [3.0], [-1.0], [2.0], [0.0]])
TRUE = torch.tensor([0.0, 2.0, 2.5, -0.5, 2.0, 1.0])

LOGITS = torch.tensor([[2.0, 0.1, 0.0], [0.2, 1.5, 0.3], [0.1, 0.2, 3.0], [1.0, 0.9, 0.0],
                       [0.0, 0.5, 2.0], [0.3, 2.0, 0.1], [1.5, 0.0, 0.2], [0.0, 0.1, 0.2]])
LABELS = torch.tensor([0, 1, 2, 1, 2, 1, 0, 0])
PICKED = LOGITS.argmax(dim=1)

SCORES = torch.tensor([0.1, 0.4, 0.35, 0.8, 0.6, 0.2, 0.55])
BINARY = torch.tensor([0, 0, 1, 1, 1, 0, 0])


def fed(metric, pred, target, parts=1):
    for chunk, truth in zip(torch.chunk(pred, parts), torch.chunk(target, parts)):
        metric.update(chunk, truth)
    return metric.compute()


def test_rmse_and_mae_match_sklearn_across_batches():
    expected_rmse = root_mean_squared_error(TRUE.numpy(), PRED.reshape(-1).numpy())
    expected_mae = mean_absolute_error(TRUE.numpy(), PRED.reshape(-1).numpy())
    assert fed(rmse(), PRED, TRUE) == pytest.approx(expected_rmse)
    assert fed(rmse(), PRED, TRUE, parts=3) == pytest.approx(expected_rmse)
    assert fed(mae(), PRED, TRUE, parts=2) == pytest.approx(expected_mae)


def test_metrics_reset_and_report_nan_when_empty():
    metric = rmse()
    assert math.isnan(metric.compute())
    metric.update(PRED, TRUE)
    metric.reset()
    assert math.isnan(metric.compute())
    assert metric.to("cpu") is metric
    assert math.isnan(auroc().compute())
    assert math.isnan(f1().compute())


def test_rmse_rejects_mismatched_sizes():
    with pytest.raises(ValueError):
        rmse().update(PRED, TRUE[:3])


def test_accuracy_from_logits_labels_and_scores():
    assert fed(accuracy(), LOGITS, LABELS, parts=2) == pytest.approx(accuracy_score(LABELS, PICKED))
    assert fed(accuracy(), PICKED, LABELS) == pytest.approx(accuracy_score(LABELS, PICKED))
    logits = torch.tensor([1.0, -2.0, 0.5, -0.1, 3.0])
    truth = torch.tensor([1, 0, 0, 0, 1])
    assert fed(accuracy(), logits, truth) == pytest.approx(0.8)
    assert fed(accuracy(), logits.reshape(-1, 1), truth.reshape(-1, 1).float()) == pytest.approx(0.8)
    macro = fed(accuracy(average="macro"), LOGITS, LABELS)
    assert macro == pytest.approx(balanced_accuracy_score(LABELS, PICKED))
    with pytest.raises(ValueError):
        accuracy(average="weighted")


def test_f1_matches_sklearn_for_every_average():
    for average in ("macro", "micro", "weighted"):
        expected = f1_score(LABELS, PICKED, average=average)
        assert fed(f1(average=average), LOGITS, LABELS, parts=2) == pytest.approx(expected), average
    logits = SCORES - 0.5
    picked = (logits > 0).long()
    assert fed(f1(average="binary"), logits, BINARY) == pytest.approx(f1_score(BINARY, picked, average="binary"))
    padded = f1_score(LABELS, PICKED, labels=[0, 1, 2, 3], average="macro", zero_division=0)
    assert fed(f1(num_classes=4), LOGITS, LABELS) == pytest.approx(padded)


def test_auroc_and_average_precision_from_binary_scores():
    assert fed(auroc(), SCORES, BINARY, parts=2) == pytest.approx(roc_auc_score(BINARY, SCORES))
    assert fed(auroc(), SCORES.reshape(-1, 1), BINARY) == pytest.approx(roc_auc_score(BINARY, SCORES))
    expected = average_precision_score(BINARY, SCORES)
    assert fed(average_precision(), SCORES, BINARY, parts=3) == pytest.approx(expected)
    logits = torch.stack([-SCORES, SCORES], dim=1)
    positive = torch.softmax(logits, dim=1)[:, 1]
    assert fed(auroc(), logits, BINARY) == pytest.approx(roc_auc_score(BINARY, positive))
    assert fed(average_precision(), logits, BINARY) == pytest.approx(average_precision_score(BINARY, positive))


def test_auroc_and_average_precision_one_versus_rest_over_classes():
    probabilities = torch.softmax(LOGITS, dim=1).numpy()
    expected = roc_auc_score(LABELS.numpy(), probabilities, multi_class="ovr", average="macro")
    assert fed(auroc(), LOGITS, LABELS, parts=2) == pytest.approx(expected)
    binarized = label_binarize(LABELS.numpy(), classes=[0, 1, 2])
    expected = average_precision_score(binarized, probabilities, average="macro")
    assert fed(average_precision(), LOGITS, LABELS) == pytest.approx(expected)


def test_scores_over_one_class_are_nan():
    assert math.isnan(fed(auroc(), SCORES, torch.ones_like(BINARY)))
    assert math.isnan(fed(average_precision(), SCORES, torch.zeros_like(BINARY)))
    partial = fed(auroc(), LOGITS, torch.where(LABELS == 2, torch.tensor(1), LABELS))
    rest = np.array([0, 1, 1, 1, 1, 1, 0, 0])
    probabilities = torch.softmax(LOGITS, dim=1).numpy()
    expected = np.mean([roc_auc_score(rest == label, probabilities[:, label]) for label in (0, 1)])
    assert partial == pytest.approx(expected)


def test_short_names_are_registered():
    table = aliases.ALIASES
    for name in ("rmse", "mae", "accuracy", "f1", "auroc", "average_precision"):
        assert name in table or any(getattr(entry, "name", None) == name for entry in table.values()), name


IMAGES = torch.rand(48, 1, 8, 8, generator=torch.Generator().manual_seed(0))


class Fixed(nn.Module):
    def __init__(self, images, scale=1.0, shift=0.0):
        super().__init__()
        self.register_buffer("images", images)
        self.scale = scale
        self.shift = shift
        self.offset = 0
        self.calls = []

    def forward(self, z, label=None):
        self.calls.append({"z": z.clone(), "label": None if label is None else label.clone(),
                           "training": self.training, "grad": torch.is_grad_enabled()})
        picked = self.images[self.offset:self.offset + len(z)]
        self.offset += len(z)
        return picked * self.scale + self.shift


def fake_ctx():
    return SimpleNamespace(rng=Rng(0), device=None)


def fid_of(metric, images=IMAGES, size=16, labels=None):
    ctx = fake_ctx()
    for start in range(0, len(images), size):
        batch = {"image": images[start:start + size]}
        if labels is not None:
            batch["label"] = labels[start:start + size]
        metric.update({}, batch, ctx)
    return metric.compute()


def test_perplexity_of_uniform_logits_is_the_vocabulary_size():
    targets = torch.randint(0, 7, (4, 5), generator=torch.Generator().manual_seed(1))
    assert fed(perplexity(), torch.zeros(4, 5, 7), targets) == pytest.approx(7.0)
    assert fed(perplexity(), torch.zeros(6, 11), torch.arange(6) % 11, parts=2) == pytest.approx(11.0)


def test_perplexity_of_confident_right_logits_tends_to_one():
    targets = torch.randint(0, 9, (3, 6), generator=torch.Generator().manual_seed(2))
    logits = 50.0 * nn.functional.one_hot(targets, 9).float()
    assert fed(perplexity(), logits, targets) == pytest.approx(1.0, abs=1e-6)
    assert fed(perplexity(), -logits, targets) > 1e6


def test_perplexity_matches_the_mean_cross_entropy_across_batches():
    generator = torch.Generator().manual_seed(3)
    logits = torch.randn(6, 4, 5, generator=generator, dtype=torch.float64)
    targets = torch.randint(0, 5, (6, 4), generator=generator)
    expected = math.exp(float(nn.functional.cross_entropy(logits.reshape(-1, 5), targets.reshape(-1))))
    assert fed(perplexity(), logits, targets, parts=3) == pytest.approx(expected, rel=1e-5)
    assert fed(perplexity(), logits, targets.reshape(6, 4, 1)) == pytest.approx(expected, rel=1e-5)


def test_perplexity_skips_ignored_tokens_and_reports_nan_when_empty():
    metric = perplexity()
    assert math.isnan(metric.compute())
    assert metric.to("cpu") is metric
    metric.update(torch.zeros(2, 3, 4), torch.full((2, 3), -100))
    assert math.isnan(metric.compute())
    targets = torch.tensor([[1, 2, -100]])
    metric.update(torch.zeros(1, 3, 4), targets)
    assert metric.tokens == 2
    assert metric.compute() == pytest.approx(4.0)
    metric.reset()
    assert math.isnan(metric.compute())


def test_perplexity_reads_the_torch_layout_and_rejects_mismatched_shapes():
    generator = torch.Generator().manual_seed(4)
    logits = torch.randn(2, 3, 6, generator=generator)
    targets = torch.randint(0, 6, (2, 3), generator=generator)
    expected = fed(perplexity(), logits, targets)
    assert fed(perplexity(), logits.movedim(-1, 1), targets) == pytest.approx(expected, rel=1e-6)
    assert fed(perplexity(classes_last=False), logits.movedim(-1, 1), targets) == pytest.approx(expected, rel=1e-6)
    with pytest.raises(ValueError, match=r"logits of shape \(2, 3, 4\) for targets of shape \(2, 5\)"):
        perplexity().update(torch.zeros(2, 3, 4), torch.zeros(2, 5, dtype=torch.long))
    with pytest.raises(ValueError):
        perplexity().update(torch.zeros(5), torch.zeros(5, dtype=torch.long))


def test_perplexity_takes_the_torch_layout_on_the_ambiguous_shape_like_cross_entropy(caplog):
    generator = torch.Generator().manual_seed(5)
    logits = torch.randn(2, 5, 5, generator=generator)
    targets = torch.randint(0, 5, (2, 5), generator=generator)
    torch_layout = math.exp(float(nn.functional.cross_entropy(logits, targets)))
    last = math.exp(float(nn.functional.cross_entropy(logits.reshape(-1, 5), targets.reshape(-1))))
    metric = perplexity()
    with caplog.at_level(logging.WARNING, logger="kalfa.std"):
        assert fed(metric, logits, targets, parts=2) == pytest.approx(torch_layout, rel=1e-6)
    assert sum("classes_last" in record.getMessage() for record in caplog.records) == 1
    assert fed(perplexity(classes_last=True), logits, targets) == pytest.approx(last, rel=1e-6)
    assert fed(perplexity(classes_last=False), logits, targets) == pytest.approx(torch_layout, rel=1e-6)


def test_fid_of_the_same_images_is_about_zero_and_a_shift_raises_it():
    metric = Fid(Fixed(IMAGES), latent=4, n=48)
    assert isinstance(metric, ContextMetric)
    assert metric.on == ("valid",)
    assert fid_of(metric) == pytest.approx(0.0, abs=1e-8)
    shifted = fid_of(Fid(Fixed(IMAGES, shift=0.5), latent=4, n=48))
    assert shifted > 1.0
    assert shifted == pytest.approx(256 * 0.25, rel=1e-6)


def test_pixel_extractor_pools_to_sixteen_by_sixteen_in_float64():
    features = pixels(IMAGES[:3].float())
    assert features.dtype == torch.float64
    assert tuple(features.shape) == (3, 256)
    assert tuple(pixels(torch.rand(2, 3, 32, 32)).shape) == (2, 768)
    assert tuple(pixels(torch.rand(2, 8, 8)).shape) == (2, 256)
    assert torch.allclose(features.reshape(3, 1, 16, 16)[:, :, ::2, ::2], IMAGES[:3].double())


def test_fid_conditions_on_labels_in_chunks_under_no_grad_with_ctx_noise():
    model = Fixed(IMAGES)
    labels = torch.arange(48) % 4
    metric = Fid(model, latent=3, label="label", n=40, batch=16)
    ctx = fake_ctx()
    metric.update({}, {"image": IMAGES[:32], "label": labels[:32]}, ctx)
    metric.update({}, {"image": IMAGES[32:], "label": labels[32:]}, ctx)
    metric.update({}, {"image": IMAGES[32:], "label": labels[32:]}, ctx)
    assert [len(call["z"]) for call in model.calls] == [16, 16, 8]
    assert torch.equal(torch.cat([call["label"] for call in model.calls]), labels[:40])
    assert not any(call["training"] or call["grad"] for call in model.calls)
    assert model.training
    assert metric.seen == 40
    assert sum(len(rows) for rows in metric.real_features) == 40
    assert sum(len(rows) for rows in metric.fake_features) == 40
    expected = Rng(0)
    noise = torch.cat([expected.normal("fid", (32, 3), like=IMAGES), expected.normal("fid", (8, 3), like=IMAGES)])
    assert torch.equal(torch.cat([call["z"] for call in model.calls]), noise)
    assert metric.compute() == pytest.approx(0.0, abs=1e-8)


def test_fid_without_enough_rows_is_nan_and_reset_clears_it():
    metric = Fid(Fixed(IMAGES), latent=2, n=1)
    assert math.isnan(metric.compute())
    assert math.isnan(fid_of(metric))
    metric = Fid(Fixed(IMAGES), latent=2, n=48)
    assert not math.isnan(fid_of(metric))
    metric.reset()
    assert metric.seen == 0
    assert math.isnan(metric.compute())


def test_fid_names_a_missing_batch_field():
    with pytest.raises(RunError, match="pixels"):
        Fid(Fixed(IMAGES), latent=2, real="pixels").update({}, {"image": IMAGES[:4]}, fake_ctx())
    with pytest.raises(RunError, match="label"):
        Fid(Fixed(IMAGES), latent=2, label="label").update({}, {"image": IMAGES[:4]}, fake_ctx())


def test_fid_resolves_an_import_path_extractor(tmp_path, monkeypatch, request):
    name = "kalfa_fid_brightness"
    source = "def brightness(images):\n    return images.mean(dim=(1, 2, 3)).reshape(-1, 1)\n"
    (tmp_path / f"{name}.py").write_text(source)
    monkeypatch.syspath_prepend(str(tmp_path))
    request.addfinalizer(lambda: sys.modules.pop(name, None))
    metric = Fid(Fixed(IMAGES, scale=0.5, shift=0.2), latent=2, n=48, extractor=f"{name}.brightness")
    real = IMAGES.mean(dim=(1, 2, 3)).double().numpy()
    fake = real * 0.5 + 0.2
    expected = (real.mean() - fake.mean()) ** 2 + (real.std(ddof=1) - fake.std(ddof=1)) ** 2
    assert fid_of(metric) == pytest.approx(expected, rel=1e-6)
    by_path = fid_of(Fid(Fixed(IMAGES, shift=0.3), latent=2, n=48, extractor="kalfa.std.metrics.pixels"))
    by_default = fid_of(Fid(Fixed(IMAGES, shift=0.3), latent=2, n=48))
    assert by_path == pytest.approx(by_default)
    flat_features = fid_of(Fid(Fixed(IMAGES, shift=0.3), latent=2, n=48, extractor=lambda images: images.flatten(1)))
    assert flat_features == pytest.approx(64 * 0.09, rel=1e-6)


def test_fid_rejects_an_extractor_that_is_not_a_function():
    with pytest.raises(BuildError, match="fid: extractor 'kalfa_no_such_module_here.features' is not an importable"):
        Fid(Fixed(IMAGES), latent=2, extractor="kalfa_no_such_module_here.features")
    with pytest.raises(BuildError, match="fid: extractor 'math.pi' is not a function"):
        Fid(Fixed(IMAGES), latent=2, extractor="math.pi")


def test_frechet_distance_matches_the_scipy_square_root():
    linalg = pytest.importorskip("scipy.linalg")
    generator = np.random.default_rng(4)
    mix = generator.normal(size=(5, 5))
    real = generator.normal(size=(200, 5))
    fake = generator.normal(loc=0.3, scale=1.5, size=(150, 5)) @ mix
    cov_real, cov_fake = np.cov(real, rowvar=False), np.cov(fake, rowvar=False)
    difference = real.mean(axis=0) - fake.mean(axis=0)
    root = linalg.sqrtm(cov_real @ cov_fake).real
    expected = difference @ difference + np.trace(cov_real) + np.trace(cov_fake) - 2.0 * np.trace(root)
    assert frechet_distance(real, fake) == pytest.approx(expected, rel=1e-8)
    assert frechet_distance(real, real) == pytest.approx(0.0, abs=1e-8)
    assert math.isnan(frechet_distance(real[:1], fake))
    with pytest.raises(RunError):
        frechet_distance(real, fake[:, :3])


def test_perplexity_and_fid_are_registered():
    for name in ("perplexity", "fid"):
        assert name in aliases.ALIASES, name
        assert aliases.ALIASES[name].slot == "metric"
