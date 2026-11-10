import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from kalfa.core import aliases
from kalfa.core.device import Device
from kalfa.core.errors import RunError
from kalfa.core.rng import Rng
from kalfa.core.wrap import Criterion, ModelEntry
from kalfa.std.objectives import (
    CosineBeta,
    Ddpm,
    Distill,
    LinearBeta,
    NoiseTables,
    Ntxent,
    VaeElbo,
    WganG,
    WganGpD,
)
from kalfa.std.schedules import LinearWarmup


def make_ctx(updates=0, targets=("label",), inputs=("x",), models=None, seed=0):
    data = SimpleNamespace(inputs=inputs, target_names=targets)
    return SimpleNamespace(rng=Rng(seed), updates=updates, data=data, device=None, models=models or {})


def numbers(shape, seed=0):
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed))


def finite(value):
    return bool(torch.isfinite(torch.as_tensor(value)).all())


def gradients(module):
    return [parameter.grad for parameter in module.parameters()]


def flows(module):
    found = gradients(module)
    return all(grad is not None for grad in found) and any(float(grad.abs().sum()) > 0 for grad in found)


def untouched(module):
    return all(grad is None for grad in gradients(module))


class Conditional(nn.Module):
    def __init__(self, width, out, classes=3):
        super().__init__()
        self.linear = nn.Linear(width, out)
        self.embed = nn.Embedding(classes, out)
        self.labels = []

    def forward(self, value, label):
        self.labels.append(label)
        return self.linear(value) + self.embed(label)


class Denoiser(nn.Module):
    def __init__(self, channels, steps):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.time = nn.Embedding(steps, channels)
        self.seen = []

    def forward(self, value, t):
        self.seen.append(t)
        return self.conv(value) + self.time(t)[:, :, None, None]


def test_linear_beta_rises_from_start_to_end():
    betas = LinearBeta(50, start=1.0e-4, end=0.02).betas
    assert betas.dtype == torch.float32 and betas.shape == (50,)
    assert betas[0].item() == pytest.approx(1.0e-4) and betas[-1].item() == pytest.approx(0.02)
    assert bool((betas[1:] > betas[:-1]).all())


def test_cosine_beta_is_inside_the_unit_interval_and_rises():
    for steps in (10, 1000):
        betas = CosineBeta(steps).betas
        assert betas.dtype == torch.float32 and betas.shape == (steps,)
        assert bool((betas > 0).all()) and bool((betas < 1).all())
        assert bool((betas[1:] >= betas[:-1]).all()) and betas[-1] > betas[0]


def test_cosine_beta_follows_the_squared_cosine_curve():
    steps = 20
    cumulative = torch.cumprod(1.0 - CosineBeta(steps, s=0.008).betas.double(), dim=0)
    curve = [math.cos((t / steps + 0.008) / 1.008 * math.pi / 2) ** 2 for t in range(steps + 1)]
    assert cumulative[4].item() == pytest.approx(curve[5] / curve[0], rel=1e-5)


def test_beta_schedules_reject_bad_values():
    with pytest.raises(ValueError, match="at least one step"):
        CosineBeta(0)
    with pytest.raises(ValueError, match="inside"):
        LinearBeta(10, start=0.0)
    with pytest.raises(ValueError, match="inside"):
        LinearBeta(10, end=1.5)


def test_noise_tables_cumulative_falls_and_placing_caches_per_device():
    tables = NoiseTables(LinearBeta(40))
    assert tables.steps == 40
    assert bool((tables.cumulative[1:] < tables.cumulative[:-1]).all())
    assert torch.allclose(tables.alphas, 1.0 - tables.betas)
    placed = tables.on("cpu")
    assert placed is tables.on(torch.device("cpu")) and placed is tables.on(Device("cpu"))
    betas, alphas, cumulative = placed
    assert torch.equal(cumulative, tables.cumulative) and cumulative.device.type == "cpu"
    double = tables.on("cpu", torch.float64)
    assert double is not placed and double.cumulative.dtype == torch.float64
    assert double.steps == 40


def recon_criterion():
    return Criterion(nn.MSELoss(), pred="x_hat", target="x")


def vae_sample():
    mu = numbers((6, 3), 1).requires_grad_(True)
    logvar = (0.1 * numbers((6, 3), 2)).requires_grad_(True)
    x_hat = numbers((6, 5), 3).requires_grad_(True)
    return {"x_hat": x_hat, "mu": mu, "logvar": logvar}, {"x": numbers((6, 5), 4)}


def test_vae_elbo_returns_its_terms_and_feeds_mu_and_logvar():
    outputs, batch = vae_sample()
    loss = VaeElbo(recon_criterion(), w_rec=2.0, w_kl=0.5)
    out = loss(outputs, batch, make_ctx())
    assert list(out) == ["loss", "recon", "kl", "w_kl"]
    mu, logvar = outputs["mu"].detach(), outputs["logvar"].detach()
    kl = (-0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(-1)).mean().item()
    recon = ((outputs["x_hat"].detach() - batch["x"]) ** 2).mean().item()
    assert out["kl"].item() == pytest.approx(kl, rel=1e-5)
    assert out["recon"].item() == pytest.approx(recon, rel=1e-5)
    assert out["w_kl"] == 0.5
    assert out["loss"].item() == pytest.approx(2.0 * recon + 0.5 * kl, rel=1e-5)
    out["loss"].backward()
    assert all(outputs[name].grad is not None for name in ("mu", "logvar", "x_hat"))


def test_vae_elbo_w_kl_schedule_follows_the_update_count():
    outputs, batch = vae_sample()
    loss = VaeElbo(recon_criterion(), w_kl=LinearWarmup(0.0, 1.0, 10))
    start = loss(outputs, batch, make_ctx(updates=0))
    middle = loss(outputs, batch, make_ctx(updates=5))
    late = loss(outputs, batch, make_ctx(updates=30))
    assert [start["w_kl"], middle["w_kl"], late["w_kl"]] == [0.0, 0.5, 1.0]
    assert start["loss"].item() == pytest.approx(start["recon"].item())
    assert middle["loss"].item() == pytest.approx(middle["recon"].item() + 0.5 * middle["kl"].item(), rel=1e-5)


def test_vae_elbo_takes_a_dict_recon_and_names_missing_outputs():
    outputs, batch = vae_sample()

    def recon(outputs, batch, ctx):
        value = ((outputs["x_hat"] - batch["x"]) ** 2).mean()
        return {"loss": value, "extra": value}

    out = VaeElbo(recon, w_kl=0.0)(outputs, batch, make_ctx())
    assert out["loss"].item() == pytest.approx(out["recon"].item())
    with pytest.raises(RunError, match="x_hat"):
        VaeElbo(recon_criterion(), mu="z_mu")(outputs, batch, make_ctx())


def gan_parts(seed=0):
    torch.manual_seed(seed)
    return nn.Linear(4, 8), nn.Linear(8, 1)


def test_wgan_gp_d_trains_the_critic_only():
    generator, critic = gan_parts()
    loss = WganGpD(generator, critic, latent=4)
    assert loss.on == ("train",)
    batch = {"image": numbers((6, 8), 5)}
    out = loss(None, batch, make_ctx())
    assert list(out) == ["loss", "wasserstein", "gp"]
    assert all(finite(value.detach()) for value in out.values())
    penalty = ((critic.weight.detach().norm() - 1.0) ** 2).item()
    assert out["gp"].item() == pytest.approx(penalty, rel=1e-4)
    assert out["loss"].item() == pytest.approx(-out["wasserstein"].item() + 10.0 * penalty, rel=1e-4)
    out["loss"].backward()
    assert flows(critic) and untouched(generator)


def test_wgan_gp_d_noise_comes_from_the_noise_stream():
    generator, critic = gan_parts()
    loss = WganGpD(generator, critic, latent=4)
    batch = {"image": numbers((6, 8), 5)}
    first = loss(None, batch, make_ctx(seed=3))["wasserstein"].item()
    again = loss(None, batch, make_ctx(seed=3))["wasserstein"].item()
    other = loss(None, batch, make_ctx(seed=4))["wasserstein"].item()
    assert first == again and first != other


def test_wgan_gp_d_computes_the_penalty_under_no_grad():
    generator, critic = gan_parts()
    with torch.no_grad():
        out = WganGpD(generator, critic, latent=4)(None, {"image": numbers((6, 8), 5)}, make_ctx())
    assert out["gp"].item() == pytest.approx(((critic.weight.norm() - 1.0) ** 2).item(), rel=1e-4)


def test_wgan_gp_d_names_a_shape_mismatch_and_a_missing_field():
    generator, critic = nn.Linear(4, 5), nn.Linear(8, 1)
    loss = WganGpD(generator, critic, latent=4)
    with pytest.raises(RunError, match="generator makes"):
        loss(None, {"image": numbers((6, 8))}, make_ctx())
    with pytest.raises(RunError, match="no field 'image'"):
        loss(None, {"x": numbers((6, 8))}, make_ctx())


def test_wgan_g_trains_the_generator_with_fresh_noise():
    generator, critic = gan_parts()
    loss = WganG(generator, critic, latent=4)
    batch = {"image": numbers((6, 8), 5)}
    out = loss(None, batch, make_ctx())
    z = Rng(0).normal("noise", (6, 4))
    assert out.item() == pytest.approx(-critic(generator(z)).mean().item(), rel=1e-5)
    out.backward()
    assert flows(generator)


def test_wgan_objectives_pass_the_label_field():
    torch.manual_seed(0)
    generator, critic = Conditional(4, 8), Conditional(8, 1)
    batch = {"image": numbers((6, 8), 5), "label": torch.tensor([0, 1, 2, 0, 1, 2])}
    out = WganGpD(generator, critic, latent=4, label="label")(None, batch, make_ctx())
    assert finite(out["loss"].detach())
    assert all(torch.equal(label, batch["label"]) for label in generator.labels + critic.labels)
    assert len(critic.labels) == 3
    out = WganG(generator, critic, latent=4, label="label")(None, batch, make_ctx())
    out.backward()
    assert generator.linear.weight.grad is not None and generator.embed.weight.grad is not None


def ddpm_batch():
    return {"image": numbers((5, 2, 3, 3), 6)}


def test_ddpm_loss_is_finite_and_trains_the_model():
    torch.manual_seed(0)
    model = Denoiser(2, 30)
    loss = Ddpm(model, LinearBeta(30), x="image")
    out = loss(None, ddpm_batch(), make_ctx())
    assert out.ndim == 0 and finite(out.detach()) and out.requires_grad
    t = model.seen[-1]
    assert t.dtype == torch.int64 and t.shape == (5,) and int(t.min()) >= 0 and int(t.max()) < 30
    out.backward()
    assert flows(model)
    assert ("cpu", torch.float32) in loss.tables.placed


def test_ddpm_draws_from_the_noise_stream_or_the_global_generator():
    torch.manual_seed(0)
    model = Denoiser(2, 30)
    loss = Ddpm(model, CosineBeta(30))
    first = loss(None, ddpm_batch(), make_ctx(seed=1)).item()
    assert first == loss(None, ddpm_batch(), make_ctx(seed=1)).item()
    assert finite(loss(None, ddpm_batch(), None).detach())


def test_ddpm_rebuilds_the_tables_when_the_schedule_changes():
    torch.manual_seed(0)
    loss = Ddpm(Denoiser(2, 30), LinearBeta(30))
    loss.schedule = CosineBeta(30)
    loss(None, ddpm_batch(), make_ctx())
    assert loss.tables.schedule is loss.schedule
    assert torch.equal(loss.tables.betas, loss.schedule.betas)


def distill_batch():
    return {"x": numbers((10, 8), 7), "label": torch.arange(10) % 3}


def test_distill_trains_the_student_not_the_teacher():
    torch.manual_seed(0)
    student, teacher = nn.Linear(8, 3), nn.Linear(8, 3)
    out = Distill(student, teacher, temperature=2.0, alpha=0.3)(None, distill_batch(), make_ctx())
    assert list(out) == ["loss", "ce", "kl"]
    assert all(finite(value.detach()) for value in out.values())
    assert out["loss"].item() == pytest.approx(0.3 * out["kl"].item() + 0.7 * out["ce"].item(), rel=1e-5)
    out["loss"].backward()
    assert flows(student) and untouched(teacher)


def test_distill_terms_match_their_definitions():
    torch.manual_seed(0)
    student, teacher = nn.Linear(8, 3), nn.Linear(8, 3)
    batch = distill_batch()
    out = Distill(student, teacher, temperature=2.0, alpha=1.0)(None, batch, make_ctx())
    logits, guide = student(batch["x"]).detach(), teacher(batch["x"]).detach()
    soft = torch.softmax(guide / 2.0, -1)
    kl = (soft * (soft.log() - torch.log_softmax(logits / 2.0, -1))).sum(-1).mean().item() * 4.0
    assert out["kl"].item() == pytest.approx(kl, rel=1e-5)
    assert out["loss"].item() == pytest.approx(kl, rel=1e-5)
    ce = torch.nn.functional.cross_entropy(logits, batch["label"]).item()
    assert out["ce"].item() == pytest.approx(ce, rel=1e-5)
    same = Distill(student, student, alpha=1.0)(None, batch, make_ctx())
    assert same["kl"].item() == pytest.approx(0.0, abs=1e-6)


def test_distill_reads_the_input_and_target_of_the_data_or_names_them():
    torch.manual_seed(0)
    student, teacher = nn.Linear(8, 3), nn.Linear(8, 3)
    batch = {"features": numbers((10, 8), 7), "kind": torch.arange(10) % 3, "other": torch.zeros(10)}
    loss = Distill(student, teacher)
    with pytest.raises(RunError, match="2 targets .*write label: <field>"):
        loss(None, batch, make_ctx(targets=("kind", "other"), inputs=("features",)))
    assert finite(loss(None, batch, make_ctx(targets=("kind",), inputs=("features",)))["loss"].detach())
    named = Distill(student, teacher, x="features", label="kind")
    assert finite(named(None, batch, make_ctx(targets=(), inputs=()))["loss"].detach())
    with pytest.raises(RunError, match="declares no inputs"):
        Distill(student, teacher, label="kind")(None, batch, make_ctx(inputs=()))


def encoder():
    return nn.Sequential(nn.Flatten(), nn.Linear(12, 16), nn.ReLU(), nn.Linear(16, 8))


def test_ntxent_of_identical_views_is_lower_than_of_random_views():
    torch.manual_seed(0)
    loss = Ntxent(encoder(), temperature=0.5)
    images = numbers((8, 3, 2, 2), 8)
    same = loss(None, {"image": torch.stack([images, images], 1)}, make_ctx())
    random = loss(None, {"image": torch.stack([images, numbers((8, 3, 2, 2), 9)], 1)}, make_ctx())
    assert finite(same.detach()) and finite(random.detach())
    assert same.item() < random.item()


def test_ntxent_trains_the_encoder_and_the_projector():
    torch.manual_seed(0)
    net, projector = encoder(), nn.Linear(8, 4)
    views = numbers((6, 2, 3, 2, 2), 10)
    out = Ntxent(net, projector, views="views")(None, {"views": views}, make_ctx())
    assert out.ndim == 0
    out.backward()
    assert flows(net) and flows(projector)


def test_ntxent_needs_two_views():
    with pytest.raises(RunError, match="two_views"):
        Ntxent(encoder())(None, {"image": numbers((8, 3, 2, 2))}, make_ctx())


def test_bind_releases_model_modules_but_keeps_inline_ones():
    torch.manual_seed(0)
    teacher = nn.Sequential(nn.Linear(8, 3), nn.Dropout(0.5))
    student = nn.Linear(8, 3)
    autoencoder = nn.Sequential(nn.Linear(12, 8), nn.Linear(8, 12))
    models = {"teacher": ModelEntry("teacher", teacher, ["x"], ["teacher"], freeze=True),
              "student": ModelEntry("student", student, ["x"], ["student"]),
              "ae": ModelEntry("ae", autoencoder, ["x"], ["ae"])}
    ctx = make_ctx(models=models)
    distill = Distill(student, teacher)
    distill.bind(ctx)
    assert list(distill.named_children()) == [] and distill.state_dict() == {}
    assert distill.teacher is teacher and distill.student is student
    distill.train(True)
    assert not teacher.training
    out = distill(None, distill_batch(), ctx)
    assert finite(out["loss"].detach())
    projector = nn.Linear(8, 4)
    ntxent = Ntxent(autoencoder[0], projector)
    ntxent.bind(ctx)
    assert [name for name, _ in ntxent.named_children()] == ["projector"]
    assert ntxent.encoder is autoencoder[0]


def test_objective_names_are_registered():
    for name in ("vae_elbo", "wgan_gp_d", "wgan_g", "ddpm", "distill", "ntxent", "linear_beta", "cosine_beta"):
        assert name in aliases.ALIASES, name
