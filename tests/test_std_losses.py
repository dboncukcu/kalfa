"""Criteria, the metric and the adapters."""

import math

import pytest
import torch

import kalfa  # noqa: F401
from helpers import batch, tiny_model
from kalfa.std.adapter import criterion as criterion_adapter
from kalfa.std.adapter import metric as metric_adapter
from kalfa.std.criterion import bce_logits, cross_entropy, huber, log_cosh, mae, mse
from kalfa.std.metric import recon_error, rmse
from kalfa.std.runtime import Context


def test_criterion_values():
    predictions = torch.tensor([[1.0], [3.0]])
    targets = torch.tensor([0.0, 1.0])
    assert float(mse(predictions, targets)) == pytest.approx(2.5)
    assert float(mae(predictions, targets)) == pytest.approx(1.5)
    assert float(huber(predictions, targets, delta=1.0)) == pytest.approx((0.5 + 1.5) / 2)
    assert float(log_cosh(predictions, targets)) == pytest.approx((math.log(math.cosh(1)) + math.log(math.cosh(2))) / 2)
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    assert float(cross_entropy(logits, torch.tensor([0, 1]))) == pytest.approx(float(
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1]))))
    assert float(bce_logits(torch.tensor([[0.0]]), torch.tensor([1.0]))) == pytest.approx(math.log(2))
    assert huber(predictions, targets, delta=1.0).requires_grad is False


def test_rmse_metric_accumulates():
    metric = rmse()
    metric.update(torch.tensor([[1.0], [3.0]]), torch.tensor([0.0, 1.0]))
    metric.update(torch.tensor([[2.0]]), torch.tensor([2.0]))
    assert metric.compute() == pytest.approx(math.sqrt(5 / 3))
    metric.reset()
    assert math.isnan(metric.compute())


def test_metrics_accumulate_above_the_range_of_a_half_precision_batch():
    metric = rmse()
    metric.update(torch.full((4, 2), 300.0, dtype=torch.float16), torch.zeros(4, 2, dtype=torch.float16))
    assert metric.compute() == pytest.approx(300.0)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="the machine has no mps device")
def test_metrics_update_on_mps_which_has_no_float64():
    for metric in (rmse(), recon_error()):
        metric.update(torch.zeros(4, 2, device="mps"), torch.ones(4, 2, device="mps"))
        assert metric.compute() == pytest.approx(1.0)


def test_criterion_adapter_reads_output_and_target_from_the_definition_keys():
    model = tiny_model(seed=1)
    data = batch()
    context = Context(data, {"model": model}, predicts="model", targets=["price"])
    adapter = criterion_adapter(mae)
    expected = float(mae(model(data["x"]), data["price"]).detach())
    assert float(adapter.loss(context, {"output": "y", "target": "price"}).detach()) == pytest.approx(expected)
    assert float(adapter.loss(context).detach()) == pytest.approx(expected)
    against_input = criterion_adapter(mse)
    assert float(against_input.loss(context, {"target": "input"}).detach()) == pytest.approx(
        float(mse(model(data["x"]), data["x"]).detach()))
    with pytest.raises(KeyError):
        adapter.loss(context, {"output": "ghost"})
    context.targets = ["a", "b"]
    with pytest.raises(ValueError, match="target"):
        adapter.loss(context)


def test_activity_and_trackers():
    from kalfa.std.runtime import active_entries, entry_active

    model = tiny_model(seed=1)
    keys = {"every": 2, "sets": ["valid"]}
    assert entry_active(keys, "valid", 2) and not entry_active(keys, "valid", 3) and not entry_active(keys, "train", 2)
    assert entry_active({}, "test", 7) and entry_active(None, "train", 1)
    table = {"a": criterion_adapter(mae), "b": criterion_adapter(mse)}
    assert [name for name, _, _ in active_entries(table, {"a": keys}, "train", 1)] == ["b"]
    tracker = criterion_adapter(mae).tracker("mae")
    for seed in (0, 1):
        data = batch(seed=seed)
        tracker.observe(Context(data, {"model": model}, predicts="model", targets=["price"]))
    values = [float(mae(model(batch(seed=seed)["x"]), batch(seed=seed)["price"]).detach()) for seed in (0, 1)]
    assert tracker.result() == {"mae": pytest.approx(sum(values) / 2)}
    metric = metric_adapter(rmse())
    tracker = metric.tracker("rmse", {"output": "y"})
    assert math.isnan(tracker.result()["rmse"])
    data = batch()
    tracker.observe(Context(data, {"model": model}, predicts="model", targets=["price"]))
    assert tracker.result()["rmse"] == pytest.approx(math.sqrt(float(mse(model(data["x"]),
                                                                          data["price"]).detach())))
    assert not hasattr(metric.metric, "seen") and metric.tracker("r").live is not metric.metric


def test_with_param_rebuilds_the_partial():
    import functools

    adapter = criterion_adapter(functools.partial(huber, delta=1.0))
    changed = adapter.with_param("delta", 0.1)
    assert changed.criterion.keywords == {"delta": 0.1} and adapter.criterion.keywords == {"delta": 1.0}


def test_a_target_selector_stacks_the_fields_and_rescales_each_one():
    import numpy
    from kalfa.std.pre import apply, fit, standard_scaler
    from kalfa.synthetic import scores_frame

    data = scores_frame(rows=40)
    prep = fit(data, {"y_*": {"target": True, "preprocessors": ["t"]}, "z": {"target": True, "preprocessors": ["t"]},
                      "x*": {"preprocessors": ["s"]}},
               {"s": standard_scaler(), "t": standard_scaler()}, [])
    frame = apply(data, prep, "valid")
    batch = {"x": torch.from_numpy(numpy.array(frame.data[prep.features].to_numpy(dtype="float32"), copy=True))}
    for name in ("y_a", "y_b", "y_c", "z"):
        batch[name] = torch.from_numpy(numpy.array(frame.data[name].to_numpy(dtype="float32"), copy=True))

    class Zero(torch.nn.Module):
        inputs = ["x"]
        outputs = ["y_hat", "z_hat"]

        def forward(self, value):
            return torch.zeros(len(value), 3), torch.zeros(len(value), 1)

    context = Context(batch, {"m": Zero()}, predicts="m", targets=["y_a", "y_b", "y_c", "z"], prep=prep,
                      set_name="valid", target_map={"y_hat": "y_*", "z_hat": "z"})
    stacked = context.target(None, "y_hat")
    assert stacked.shape == (40, 3)
    assert torch.allclose(stacked[:, 1], batch["y_b"])
    assert context.target(None, "z_hat").shape == (40,)
    assert context.target_fields(None, "y_hat") == ["y_a", "y_b", "y_c"]

    predictions, targets = context.rescaled("y_hat", None)
    for position, name in enumerate(("y_a", "y_b", "y_c")):
        original = data[name].to_numpy()
        assert numpy.allclose(targets[:, position].numpy(), original, atol=1e-4)
        assert numpy.allclose(predictions[:, position].numpy(), prep.inverse(name, numpy.zeros(40)), atol=1e-4)
    grouped = prep.fitted["t"]
    means = [float(grouped.obj.scaler.mean_[grouped.columns.index(name)]) for name in ("y_a", "y_b", "y_c")]
    assert len(set(round(value, 6) for value in means)) == 3 and grouped.columns == ["y_a", "y_b", "y_c", "z"]


def test_metrics_report_in_the_original_scale_and_losses_in_the_model_scale():
    import numpy
    from kalfa.std.pre import apply, fit, standard_scaler
    from kalfa.synthetic import housing_frame

    data = housing_frame(rows=40)
    prep = fit(data, {"x*": {"preprocessors": ["s"]}, "price": {"target": True, "preprocessors": ["t"]}},
               {"s": standard_scaler(), "t": standard_scaler()}, [])
    frame = apply(data, prep, "valid")
    x = torch.from_numpy(numpy.array(frame.data[prep.features].to_numpy(dtype="float32"), copy=True))
    price = torch.from_numpy(numpy.array(frame.data["price"].to_numpy(dtype="float32"), copy=True))

    class Zero(torch.nn.Module):
        inputs = ["x"]
        outputs = ["y"]

        def forward(self, value):
            return torch.zeros(len(value), 1)

    context = Context({"x": x, "price": price}, {"m": Zero()}, predicts="m", targets=["price"], prep=prep,
                      set_name="valid")
    model_scale = float(mae(torch.zeros(40, 1), price))
    assert float(criterion_adapter(mae).loss(context)) == pytest.approx(model_scale)
    as_metric = criterion_adapter(mae).tracker("mae", None, rescale=True)
    as_metric.observe(context)
    original_mae = float(numpy.abs(data["price"].to_numpy() - prep.inverse("price", numpy.zeros(40))).mean())
    assert as_metric.result()["mae"] == pytest.approx(original_mae, rel=1e-4) and original_mae > 10.0
    as_loss = criterion_adapter(mae).tracker("mae", None)
    as_loss.observe(context)
    assert as_loss.result()["mae"] == pytest.approx(model_scale)
    tracker = metric_adapter(rmse()).tracker("rmse", None, rescale=True)
    tracker.observe(context)
    original = float(numpy.sqrt(((data["price"].to_numpy() - prep.inverse("price", numpy.zeros(40))) ** 2).mean()))
    assert tracker.result()["rmse"] == pytest.approx(original, rel=1e-3) and original > 10.0
    plain = metric_adapter(rmse()).tracker("rmse", None)
    plain.observe(context)
    assert plain.result()["rmse"] < 5.0
    predictions, targets = context.rescaled(None, "input")
    assert numpy.allclose(targets.numpy(), data[prep.features].to_numpy(dtype="float32"), atol=1e-3)


def test_vae_objective_and_schedules():
    import functools

    from torch import nn

    from kalfa.std.objective import vae
    from kalfa.std.schedule import linear_warmup, step_decay, warmup_cosine
    from kalfa.std.layer import reparam
    from kalfa.std.turn import with_param

    class Encoder(nn.Module):
        inputs = ["image"]
        outputs = ["mu", "logvar"]

        def __init__(self):
            super().__init__()
            self.mu = nn.Linear(4, 2)
            self.logvar = nn.Linear(4, 2)

        def forward(self, x):
            return self.mu(x), self.logvar(x)

    class Decoder(nn.Module):
        inputs = ["z"]
        outputs = ["x_hat"]

        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(2, 4)

        def forward(self, z):
            return self.layer(z)

    torch.manual_seed(0)
    models = {"encoder": Encoder(), "decoder": Decoder()}
    batch = {"image": torch.randn(5, 4)}
    schedule = functools.partial(linear_warmup, start=0.0, end=1.0, steps=10)
    out = vae(models, batch, "encoder", "decoder", mse, w_rec=1.0, kl_schedule=schedule, step=5)
    assert set(out) == {"loss", "recon", "kl", "w_kl"} and out["w_kl"] == pytest.approx(0.5)
    assert float(out["loss"].detach()) == pytest.approx(float(out["recon"].detach())
                                                       + 0.5 * float(out["kl"].detach()))
    assert out["loss"].requires_grad
    objective = functools.partial(vae, encoder="encoder", decoder="decoder", recon=mse, kl_schedule=schedule)
    changed = with_param(objective, "w_rec", 0.5)
    assert changed.keywords["w_rec"] == 0.5 and changed.keywords["recon"] is mse and changed.func is vae
    assert linear_warmup(20, 0.0, 1.0, 10) == 1.0 and step_decay(25, 10, 0.5) == pytest.approx(0.25)
    assert warmup_cosine(5, 10, 100) == pytest.approx(0.5) and warmup_cosine(100, 10, 100) == pytest.approx(0.0)
    layer = reparam()
    mu, logvar = torch.zeros(3, 2), torch.zeros(3, 2)
    layer.eval()
    assert torch.equal(layer(mu, logvar), mu)
    layer.train()
    assert not torch.equal(layer(mu, logvar), mu)


def test_wgan_objectives_and_fid():
    from torch import nn

    from kalfa.std.metric import Fid, frechet_distance
    from kalfa.std.objective import wgan_g, wgan_gp_d

    class Generator(nn.Module):
        inputs = ["z"]
        outputs = ["image"]

        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(4, 12)

        def forward(self, z):
            return self.layer(z).reshape(len(z), 3, 2, 2)

    class Critic(nn.Module):
        inputs = ["image"]
        outputs = ["score"]

        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(12, 1)

        def forward(self, image):
            return self.layer(image.flatten(1))

    torch.manual_seed(0)
    models = {"generator": Generator(), "critic": Critic()}
    batch = {"image": torch.randn(6, 3, 2, 2)}
    loss = wgan_gp_d(models, batch, "generator", "critic", latent=4, gp_weight=10.0, rng=torch.Generator().manual_seed(1))
    assert loss.requires_grad and loss.shape == ()
    loss.backward()
    assert models["critic"].layer.weight.grad is not None and models["generator"].layer.weight.grad is None
    generator_loss = wgan_g(models, batch, "generator", "critic", latent=4)
    assert generator_loss.requires_grad
    metric = Fid("generator", latent=4, conditional=False, n=6, extractor=lambda images: images.flatten(1))
    metric.update(models, batch, rng=torch.Generator().manual_seed(2))
    assert metric.seen_real == 6 and metric.seen_fake == 6 and metric.compute() >= 0.0
    import numpy

    same = numpy.random.default_rng(0).normal(size=(50, 3))
    assert frechet_distance(same, same) == pytest.approx(0.0, abs=1e-6)


def test_ddpm_objective_and_sampler():
    import functools

    from torch import nn

    from kalfa.std.generate import ddpm_sampler
    from kalfa.std.objective import ddpm, diffusion_steps, noise_schedule
    from kalfa.std.schedule import linear_betas

    class Net(nn.Module):
        inputs = ["image", "t"]
        outputs = ["noise"]

        def __init__(self):
            super().__init__()
            self.layer = nn.Conv2d(1, 1, 1)

        def forward(self, image, t):
            return self.layer(image) + t.float().reshape(-1, 1, 1, 1) * 0.0

    schedule = functools.partial(linear_betas, steps=10)
    assert diffusion_steps(schedule) == 10
    betas, alphas, cumulative = noise_schedule(schedule, "cpu")
    assert betas.shape == (10,) and float(betas[0]) == pytest.approx(1e-4) and float(betas[-1]) == pytest.approx(0.02)
    assert torch.all(cumulative[1:] < cumulative[:-1])
    models = {"unet": Net()}
    loss = ddpm(models, {"image": torch.randn(3, 1, 4, 4)}, "unet", schedule, rng=torch.Generator().manual_seed(0))
    assert loss.requires_grad and loss.shape == ()
    samples = ddpm_sampler(models, None, torch.Generator().manual_seed(1), "unet", schedule, [1, 4, 4], n=2)
    assert samples.shape == (2, 1, 4, 4) and float(samples.abs().max()) <= 1.0
    with pytest.raises(ValueError, match="steps"):
        diffusion_steps(lambda step: 0.1)
