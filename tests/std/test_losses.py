import logging

import pytest
import torch

from kalfa.core import aliases
from kalfa.core.errors import RunError
from kalfa.std.losses import Constraint, Mdmm, Term, WeightedSum, cross_entropy
from kalfa.std.schedules import Cosine, LinearWarmup, StepDecay, Warmup


def mse(outputs, batch, ctx):
    return ((outputs["y"] - batch["t"]) ** 2).mean()


def mae(outputs, batch, ctx):
    return (outputs["y"] - batch["t"]).abs().mean()


def elbo(outputs, batch, ctx):
    recon = mse(outputs, batch, ctx)
    return {"loss": recon + 0.5, "recon": recon, "kl": torch.tensor(0.5)}


def sample():
    outputs = {"y": torch.tensor([[1.0], [2.0], [4.0]], requires_grad=True)}
    batch = {"t": torch.tensor([[1.0], [1.0], [1.0]])}
    return outputs, batch


MSE = 10 / 3
MAE = 4 / 3


def test_weighted_sum_adds_weighted_terms_and_reports_each():
    outputs, batch = sample()
    loss = WeightedSum([Term(mse, 1.0), {"of": mae, "weight": 0.5}, {"of": elbo, "term": "kl", "weight": 2.0}])
    out = loss(outputs, batch, None)
    assert list(out) == ["loss", "mse", "mae", "elbo.kl"]
    assert out["loss"].item() == pytest.approx(MSE + 0.5 * MAE + 2.0 * 0.5)
    assert out["mae"].item() == pytest.approx(MAE)
    assert out["loss"].requires_grad


def test_weighted_sum_picks_the_loss_key_of_a_dict_result():
    outputs, batch = sample()
    out = WeightedSum([{"of": elbo}])(outputs, batch, None)
    assert out["loss"].item() == pytest.approx(MSE + 0.5)
    assert out["elbo"].item() == pytest.approx(MSE + 0.5)


def test_weighted_sum_names_a_missing_term():
    outputs, batch = sample()
    with pytest.raises(RunError, match="no term 'kl'"):
        WeightedSum([{"of": mse, "term": "kl"}])(outputs, batch, None)
    with pytest.raises(RunError, match="its terms are"):
        WeightedSum([{"of": elbo, "term": "nope"}])(outputs, batch, None)


def test_weighted_sum_needs_terms():
    with pytest.raises(ValueError):
        WeightedSum([])
    with pytest.raises(TypeError):
        WeightedSum([3])


def test_mdmm_reports_objective_lambda_and_infeasibility():
    outputs, batch = sample()
    loss = Mdmm(mse, [{"of": mae, "epsilon": 0.05, "lmbda_init": 0.5, "scale": 2.0, "damping": 1.0}])
    assert [name for name, _ in loss.named_parameters()] == ["lmbda"]
    out = loss(outputs, batch, None)
    assert list(out) == ["loss", "objective", "lambda/mae", "inf/mae"]
    infeasibility = 0.05 - MAE
    assert out["loss"].item() == pytest.approx(MSE + 2.0 * (0.5 * infeasibility + infeasibility ** 2 / 2))
    assert out["objective"].item() == pytest.approx(MSE)
    assert out["inf/mae"].item() == pytest.approx(infeasibility)
    assert out["lambda/mae"].item() == pytest.approx(0.5)


def test_mdmm_lambda_gradient_is_scale_times_infeasibility():
    outputs, batch = sample()
    loss = Mdmm(mse, [Constraint(mae, epsilon=0.05, scale=2.0, lmbda_init=0.5)])
    loss(outputs, batch, None)["loss"].backward()
    assert loss.lmbda.grad.item() == pytest.approx(2.0 * (0.05 - MAE))
    assert loss.lmbda.grad.item() < 0
    assert outputs["y"].grad is not None


def test_mdmm_with_a_term_and_several_constraints():
    outputs, batch = sample()
    loss = Mdmm(elbo, [{"of": mae, "epsilon": 2.0}, {"of": elbo, "term": "kl", "epsilon": 1.0}], term="recon")
    out = loss(outputs, batch, None)
    assert out["objective"].item() == pytest.approx(MSE)
    assert out["inf/mae"].item() == pytest.approx(2.0 - MAE)
    assert out["inf/elbo.kl"].item() == pytest.approx(0.5)
    assert loss.lmbda.shape == (2,)
    assert list(loss.state_dict()) == ["lmbda"]


def test_mdmm_needs_constraints():
    with pytest.raises(ValueError):
        Mdmm(mse, [])


def test_mdmm_patches_land_on_constraint_attributes():
    loss = Mdmm(mse, [{"of": mae, "epsilon": 0.05}])
    loss.constraints[0].epsilon = 0.02
    outputs, batch = sample()
    assert loss(outputs, batch, None)["inf/mae"].item() == pytest.approx(0.02 - MAE)


def test_cross_entropy_takes_list_weights():
    loss = cross_entropy(weight=[1.0, 2.0], label_smoothing=0.1)
    assert torch.equal(loss.weight, torch.tensor([1.0, 2.0]))
    assert loss.label_smoothing == 0.1
    value = loss(torch.tensor([[2.0, 0.0], [0.0, 2.0]]), torch.tensor([0, 1]))
    assert 0.0 < value.item() < 1.0
    assert cross_entropy().weight is None


def test_cross_entropy_reads_both_layouts_and_warns_once_on_the_ambiguous_one(caplog):
    generator = torch.Generator().manual_seed(0)
    logits = torch.randn(2, 3, 7, generator=generator)
    target = torch.randint(0, 7, (2, 3), generator=generator)
    last = torch.nn.functional.cross_entropy(logits.reshape(-1, 7), target.reshape(-1))
    loss = cross_entropy()
    assert loss.classes_last is None
    assert float(loss(logits, target)) == pytest.approx(float(last))
    assert float(loss(logits.movedim(-1, 1), target)) == pytest.approx(float(last))
    square = torch.randn(2, 5, 5, generator=generator)
    truth = torch.randint(0, 5, (2, 5), generator=generator)
    with caplog.at_level(logging.WARNING, logger="kalfa.std"):
        expected = float(torch.nn.functional.cross_entropy(square, truth))
        assert float(loss(square, truth)) == pytest.approx(expected)
        assert float(loss(square, truth)) == pytest.approx(expected)
    warned = [record.getMessage() for record in caplog.records if "classes_last" in record.getMessage()]
    assert len(warned) == 1 and "cross_entropy: logits of shape (2, 5, 5)" in warned[0]
    assert float(cross_entropy(classes_last=True)(logits, target)) == pytest.approx(float(last))


def test_schedules_at_a_few_points():
    warmup = Warmup(10)
    assert [warmup(0), warmup(5), warmup(10), warmup(50)] == [0.0, 0.5, 1.0, 1.0]
    assert Warmup(0)(3) == 1.0
    cosine = Cosine(100, warmup=10, floor=0.1)
    assert cosine(0) == 0.0
    assert cosine(5) == pytest.approx(0.5)
    assert cosine(10) == pytest.approx(1.0)
    assert cosine(55) == pytest.approx(0.55)
    assert cosine(100) == pytest.approx(0.1)
    assert cosine(1000) == pytest.approx(0.1)
    plain = Cosine(10)
    assert [plain(0), round(plain(5), 6), plain(10)] == [1.0, 0.5, 0.0]
    linear = LinearWarmup(0.0, 1.0, 20000)
    assert [linear(0), linear(10000), linear(20000), linear(30000)] == [0.0, 0.5, 1.0, 1.0]
    decay = StepDecay(every=10, gamma=0.5)
    assert [decay(0), decay(9), decay(10), decay(25)] == [1.0, 1.0, 0.5, 0.25]


def test_short_names_are_registered():
    table = aliases.ALIASES
    for name in ("mse", "l1", "huber", "bce_logits", "poisson_nll", "cross_entropy", "weighted_sum", "mdmm", "adam",
                 "adamw", "sgd", "warmup", "cosine", "linear_warmup", "step_decay"):
        assert name in table or any(getattr(entry, "name", None) == name for entry in table.values()), name
    if "mae" in table:
        assert getattr(table["mae"], "slot", None) == "metric"
