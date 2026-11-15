import copy
import math

import torch
from torch import nn
from torch.nn import functional

from kalfa.core.aliases import register
from kalfa.core.errors import RunError
from kalfa.core.predict import batch_size, target_names
from kalfa.core.protocols import Objective
from kalfa.core.rng import Rng
from kalfa.core.wrap import loss_scalar


NOISE = "noise"


def title(owner):
    name = getattr(owner, "name", "")
    return f"loss {name!r}" if name else type(owner).__name__


def named(obj):
    return getattr(obj, "name", "") or getattr(obj, "__name__", "") or type(obj).__name__


def field(batch, name, owner):
    if name not in batch:
        raise RunError(f"{title(owner)}: the batch has no field {name!r}; its fields are {sorted(map(str, batch))}")
    return batch[name]


def first(result):
    if isinstance(result, (tuple, list)):
        return result[0]
    if isinstance(result, dict):
        return next(iter(result.values()))
    return result


def call(module, value, label):
    return module(value) if label is None else module(value, label)


def device_of(module, batch):
    for parameter in module.parameters():
        return parameter.device
    for value in batch.values() if isinstance(batch, dict) else ():
        if isinstance(value, torch.Tensor):
            return value.device
    return torch.device("cpu")


def normal(ctx, shape, like=None, device=None):
    rng = getattr(ctx, "rng", None)
    if rng is not None:
        return rng.normal(NOISE, shape, like=like, device=device)
    if like is None:
        return torch.randn(shape, device=device)
    return torch.randn(shape, device=like.device, dtype=like.dtype if like.dtype.is_floating_point else None)


def uniform(ctx, shape, like):
    rng = getattr(ctx, "rng", None)
    if rng is not None:
        return rng.uniform(NOISE, shape, like=like)
    return torch.rand(shape, device=like.device, dtype=like.dtype if like.dtype.is_floating_point else None)


def integers(ctx, high, count, device):
    rng = getattr(ctx, "rng", None)
    if rng is not None:
        return rng.integers(NOISE, 0, high, (count,), device=device)
    return torch.randint(0, int(high), (count,), device=device)


def owned_modules(ctx):
    found = []
    for entry in (getattr(ctx, "models", None) or {}).values():
        for module in (getattr(entry, "module", None), getattr(entry, "ema", None)):
            if isinstance(module, nn.Module):
                found.extend(module.modules())
    return found


def release(objective, ctx):
    owned = owned_modules(ctx)
    for name, module in list(objective.named_children()):
        if any(module is part for part in owned):
            delattr(objective, name)
            object.__setattr__(objective, name, module)


def gaussian_kl(mu, logvar):
    return -0.5 * torch.mean(torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1))


class BetaSchedule:
    def __init__(self, steps):
        self.steps = int(steps)
        if self.steps < 1:
            raise ValueError(f"a beta schedule needs at least one step, got {steps}")
        self.betas = self.compute()

    def compute(self):
        raise NotImplementedError


class LinearBeta(BetaSchedule):
    def __init__(self, steps, start=1.0e-4, end=0.02):
        self.start = float(start)
        self.end = float(end)
        if not (0.0 < self.start < 1.0 and 0.0 < self.end < 1.0):
            raise ValueError(f"linear_beta needs start and end inside (0, 1), got {start} and {end}")
        super().__init__(steps)

    def compute(self):
        return torch.linspace(self.start, self.end, self.steps, dtype=torch.float64).to(torch.float32)


class CosineBeta(BetaSchedule):
    def __init__(self, steps, s=0.008):
        self.s = float(s)
        super().__init__(steps)

    def compute(self):
        grid = torch.arange(self.steps + 1, dtype=torch.float64) / self.steps
        curve = torch.cos((grid + self.s) / (1.0 + self.s) * math.pi / 2.0) ** 2
        cumulative = curve / curve[0]
        betas = 1.0 - cumulative[1:] / cumulative[:-1]
        return betas.clamp(1.0e-8, 0.999).to(torch.float32)


class NoiseTables:
    def __init__(self, schedule):
        self.schedule = schedule
        self.steps = schedule.steps
        self.betas = schedule.betas
        self.alphas = 1.0 - self.betas
        self.cumulative = torch.cumprod(self.alphas, dim=0)
        self.placed = {}

    def on(self, device, dtype=torch.float32):
        where = Rng.place(device)
        key = (str(where), dtype)
        found = self.placed.get(key)
        if found is None:
            found = copy.copy(self)
            found.betas = self.betas.to(where, dtype)
            found.alphas = self.alphas.to(where, dtype)
            found.cumulative = self.cumulative.to(where, dtype)
            self.placed[key] = found
        return found

    def __iter__(self):
        return iter((self.betas, self.alphas, self.cumulative))


class VaeElbo(Objective):
    def __init__(self, recon, w_rec=1.0, w_kl=1.0, mu="mu", logvar="logvar"):
        super().__init__()
        self.recon = recon
        self.w_rec = float(w_rec)
        self.w_kl = w_kl
        self.mu = mu
        self.logvar = logvar

    def weight(self, ctx):
        if callable(self.w_kl):
            return float(self.w_kl(getattr(ctx, "updates", 0)))
        return float(self.w_kl)

    def forward(self, outputs, batch, ctx):
        outputs = {} if outputs is None else outputs
        missing = [name for name in (self.mu, self.logvar) if name not in outputs]
        if missing:
            raise RunError(f"{title(self)}: the step outputs have no {missing}; they are {sorted(map(str, outputs))}; "
                           "name the encoder outputs with mu and logvar")
        kl = gaussian_kl(outputs[self.mu].float(), outputs[self.logvar].float())
        recon = loss_scalar(self.recon(outputs, batch, ctx), named(self.recon))
        w_kl = self.weight(ctx)
        return {"loss": self.w_rec * recon + w_kl * kl, "recon": recon, "kl": kl, "w_kl": w_kl}


class WganGpD(Objective):
    on = ("train",)

    def __init__(self, generator, critic, latent, real="image", label=None, gp_weight=10.0):
        super().__init__()
        self.generator = generator
        self.critic = critic
        self.latent = int(latent)
        self.real = real
        self.label = label
        self.gp_weight = float(gp_weight)

    def bind(self, ctx):
        release(self, ctx)

    def forward(self, outputs, batch, ctx):
        real = field(batch, self.real, self).detach()
        label = None if self.label is None else field(batch, self.label, self)
        count = real.shape[0]
        z = normal(ctx, (count, self.latent), device=real.device)
        with torch.no_grad():
            fake = first(call(self.generator, z, label)).detach()
        if fake.shape != real.shape:
            raise RunError(f"{title(self)}: the generator makes {tuple(fake.shape)} but the real field "
                           f"{self.real!r} is {tuple(real.shape)}")
        score_real = first(call(self.critic, real, label))
        score_fake = first(call(self.critic, fake, label))
        with torch.enable_grad():
            mix = uniform(ctx, (count,) + (1,) * (real.ndim - 1), like=real)
            between = (mix * real + (1.0 - mix) * fake).requires_grad_(True)
            score = first(call(self.critic, between, label))
            gradients = torch.autograd.grad(score.sum(), between, create_graph=True)[0]
            penalty = ((gradients.flatten(1).norm(dim=1) - 1.0) ** 2).mean()
        wasserstein = score_real.mean() - score_fake.mean()
        return {"loss": -wasserstein + self.gp_weight * penalty, "wasserstein": wasserstein, "gp": penalty}


class WganG(Objective):
    def __init__(self, generator, critic, latent, label=None):
        super().__init__()
        self.generator = generator
        self.critic = critic
        self.latent = int(latent)
        self.label = label

    def bind(self, ctx):
        release(self, ctx)

    def forward(self, outputs, batch, ctx):
        label = None if self.label is None else field(batch, self.label, self)
        z = normal(ctx, (batch_size(batch), self.latent), device=device_of(self.generator, batch))
        fake = first(call(self.generator, z, label))
        return -first(call(self.critic, fake, label)).mean()


class Ddpm(Objective):
    def __init__(self, model, schedule, x="image"):
        super().__init__()
        self.model = model
        self.schedule = schedule
        self.x = x
        self.tables = NoiseTables(schedule)

    def bind(self, ctx):
        release(self, ctx)

    def forward(self, outputs, batch, ctx):
        if self.tables.schedule is not self.schedule:
            self.tables = NoiseTables(self.schedule)
        x = field(batch, self.x, self)
        tables = self.tables.on(x.device)
        count = x.shape[0]
        t = integers(ctx, tables.steps, count, x.device)
        noise = normal(ctx, x.shape, like=x)
        weight = tables.cumulative[t].to(x.dtype).reshape((count,) + (1,) * (x.ndim - 1))
        noised = weight.sqrt() * x + (1.0 - weight).sqrt() * noise
        predicted = first(self.model(noised, t))
        return functional.mse_loss(predicted.float(), noise.float())


class Distill(Objective):
    def __init__(self, student, teacher, x=None, label=None, temperature=1.0, alpha=0.5):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.x = x
        self.label = label
        self.temperature = float(temperature)
        self.alpha = float(alpha)

    def bind(self, ctx):
        release(self, ctx)

    def input_name(self, ctx):
        if self.x is not None:
            return self.x
        inputs = list(getattr(getattr(ctx, "data", None), "inputs", None) or ())
        if not inputs:
            raise RunError(f"{title(self)}: x is not written and the data declares no inputs; write x: <field>")
        return inputs[0]

    def label_name(self, ctx):
        if self.label is not None:
            return self.label
        names = target_names(getattr(ctx, "data", None))
        if len(names) != 1:
            raise RunError(f"{title(self)}: label is not written and the data has {len(names)} targets {names}; "
                           "write label: <field>")
        return names[0]

    def forward(self, outputs, batch, ctx):
        x = field(batch, self.input_name(ctx), self)
        labels = field(batch, self.label_name(ctx), self).reshape(-1).long()
        logits = first(self.student(x)).float()
        with torch.no_grad():
            guide = first(self.teacher(x)).float()
        scale = self.temperature
        soft = functional.softmax(guide / scale, dim=-1)
        guess = functional.log_softmax(logits / scale, dim=-1)
        kl = functional.kl_div(guess, soft, reduction="batchmean") * scale * scale
        ce = functional.cross_entropy(logits, labels)
        return {"loss": self.alpha * kl + (1.0 - self.alpha) * ce, "ce": ce, "kl": kl}


class Ntxent(Objective):
    def __init__(self, encoder, projector=None, views="image", temperature=0.5):
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.views = views
        self.temperature = float(temperature)

    def bind(self, ctx):
        release(self, ctx)

    def forward(self, outputs, batch, ctx):
        views = field(batch, self.views, self)
        if views.ndim < 3 or views.shape[1] != 2:
            raise RunError(f"{title(self)}: ntxent needs two views per item, a (B, 2, ...) field, but {self.views!r} "
                           f"is {tuple(views.shape)}; put two_views in the train transform")
        count = views.shape[0]
        embedded = first(self.encoder(views.flatten(0, 1)))
        if self.projector is not None:
            embedded = first(self.projector(embedded))
        z = functional.normalize(embedded.flatten(1).float(), dim=1)
        similarity = z @ z.t() / self.temperature
        diagonal = torch.eye(2 * count, device=z.device, dtype=torch.bool)
        similarity = similarity.masked_fill(diagonal, float("-inf"))
        partner = torch.arange(2 * count, device=z.device) ^ 1
        return functional.cross_entropy(similarity, partner)


REGISTRATIONS = (
    ("vae_elbo", "kalfa.std.objectives.VaeElbo", "loss",
     "w_rec times the recon criterion on the outputs plus w_kl (a number or a schedule of the update count) times the "
     "gaussian KL of the mu and logvar outputs; terms recon, kl, w_kl"),
    ("wgan_gp_d", "kalfa.std.objectives.WganGpD", "loss",
     "the critic loss of WGAN GP: fake minus real scores plus gp_weight times the gradient penalty on interpolates; "
     "the latent noise comes from the noise stream; terms wasserstein, gp; train only by default"),
    ("wgan_g", "kalfa.std.objectives.WganG", "loss",
     "the generator loss of WGAN: minus the critic score of fresh fakes"),
    ("ddpm", "kalfa.std.objectives.Ddpm", "loss",
     "the denoising loss: noise the x field at a random step of the beta schedule and regress the noise with "
     "model(x_t, t)"),
    ("distill", "kalfa.std.objectives.Distill", "loss",
     "alpha times the KL between student and teacher logits at temperature plus (1 - alpha) times cross entropy with "
     "the label field (default the only target of the data); terms ce, kl"),
    ("ntxent", "kalfa.std.objectives.Ntxent", "loss",
     "NT-Xent over two views per image: the encoder (and projector) embeds both views, partners are positives"),
    ("linear_beta", "kalfa.std.objectives.LinearBeta", "value", "betas rising linearly from start to end over steps"),
    ("cosine_beta", "kalfa.std.objectives.CosineBeta", "value", "the cosine beta schedule of Nichol and Dhariwal"),
)

for name, target, slot, doc in REGISTRATIONS:
    register(name, target, slot=slot, doc=doc)
