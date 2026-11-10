import itertools

import torch
from torch import nn

from kalfa.core.aliases import register
from kalfa.core.errors import RunError
from kalfa.core.rng import Rng
from kalfa.std.common import evaluating
from kalfa.std.data.text import size_of
from kalfa.std.objectives import NoiseTables


STREAM = "samples"


def module_of(model):
    if isinstance(model, nn.Module):
        return model
    inner = getattr(model, "module", None)
    if isinstance(inner, nn.Module):
        return inner
    if callable(model):
        return model
    raise RunError(f"a sampler needs a model to call, got {type(model).__name__}")


def device_of(ctx, model):
    device = getattr(ctx, "device", None)
    if device is not None:
        return Rng.place(device)
    if isinstance(model, nn.Module):
        for tensor in itertools.chain(model.parameters(), model.buffers()):
            return tensor.device
    return torch.device("cpu")


def count_of(sampler, n):
    count = int(sampler.n if n is None else n)
    if count < 1:
        raise RunError(f"{type(sampler).__name__} draws n samples; n must be positive, got {count}")
    return count


def rng_of(ctx):
    return getattr(ctx, "rng", None)


def normal(ctx, shape, device):
    rng = rng_of(ctx)
    if rng is None:
        return torch.randn(shape, device=device)
    return rng.normal(STREAM, shape, device=device)


def first(result):
    if isinstance(result, dict):
        result = next(iter(result.values()))
    while isinstance(result, (tuple, list)):
        result = result[0]
    return result


def finished(result):
    result = first(result)
    if isinstance(result, torch.Tensor):
        return result.detach().cpu()
    return result


class GanSampler:
    def __init__(self, model, latent, n_classes=None, n=64):
        self.model = model
        self.latent = int(latent)
        self.n_classes = None if n_classes is None else int(n_classes)
        self.n = int(n)

    def sample(self, ctx, n=None):
        count = count_of(self, n)
        model = module_of(self.model)
        device = device_of(ctx, model)
        with evaluating(model):
            z = normal(ctx, (count, self.latent), device)
            if self.n_classes:
                labels = torch.arange(count, device=device) % self.n_classes
                return finished(model(z, labels))
            return finished(model(z))


class LatentSampler:
    def __init__(self, model, latent, n=64):
        self.model = model
        self.latent = int(latent)
        self.n = int(n)

    def sample(self, ctx, n=None):
        count = count_of(self, n)
        model = module_of(self.model)
        device = device_of(ctx, model)
        with evaluating(model):
            return finished(model(normal(ctx, (count, self.latent), device)))


class DdpmSampler:
    def __init__(self, model, schedule, shape, n=64, steps=None):
        self.model = model
        self.schedule = schedule
        self.shape = tuple(int(part) for part in shape)
        self.n = int(n)
        self.steps = None if steps is None else int(steps)
        self.noise = None

    def tables(self, device):
        source = self.schedule
        if not isinstance(source, NoiseTables):
            if self.noise is None or self.noise.schedule is not source:
                self.noise = NoiseTables(source)
            source = self.noise
        placed = source.on(device)
        return int(placed.steps), placed.betas, placed.alphas, placed.cumulative

    def timesteps(self, total):
        if self.steps is None:
            return list(range(total - 1, -1, -1))
        if self.steps < 1:
            raise RunError(f"ddpm_sampler steps must be positive, got {self.steps}")
        count = min(self.steps, total)
        if count == 1:
            return [total - 1]
        spaced = torch.linspace(total - 1, 0, count, dtype=torch.float64).round().long().tolist()
        return list(dict.fromkeys(int(step) for step in spaced))

    def sample(self, ctx, n=None):
        count = count_of(self, n)
        model = module_of(self.model)
        device = device_of(ctx, model)
        total, betas, alphas, cumulative = self.tables(device)
        size = (count, *self.shape)
        schedule = self.timesteps(total)
        with evaluating(model):
            x = normal(ctx, size, device)
            for position, step in enumerate(schedule):
                t = torch.full((count,), step, dtype=torch.long, device=device)
                predicted = first(model(x, t)).to(x.dtype)
                if self.steps is None:
                    x = self.ancestral(ctx, x, predicted, step, betas, alphas, cumulative, size, device)
                else:
                    following = schedule[position + 1] if position + 1 < len(schedule) else None
                    x = self.implicit(x, predicted, step, following, cumulative)
        return x.clamp(-1.0, 1.0).detach().cpu()

    @staticmethod
    def ancestral(ctx, x, predicted, step, betas, alphas, cumulative, size, device):
        coefficient = betas[step] / (1.0 - cumulative[step]).sqrt()
        x = (x - coefficient * predicted) / alphas[step].sqrt()
        if step > 0:
            x = x + betas[step].sqrt() * normal(ctx, size, device)
        return x

    @staticmethod
    def implicit(x, predicted, step, following, cumulative):
        current = cumulative[step]
        previous = cumulative[following] if following is not None else torch.ones_like(current)
        start = ((x - (1.0 - current).sqrt() * predicted) / current.sqrt()).clamp(-1.0, 1.0)
        return previous.sqrt() * start + (1.0 - previous).sqrt() * predicted


class LmSampler:
    def __init__(self, model, tokenizer, prompt="", max_new_tokens=100, temperature=1.0, top_k=None, n=1):
        self.model = model
        self.tokenizer = tokenizer
        self.prompt = str(prompt)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_k = None if top_k is None else int(top_k)
        self.n = int(n)

    def tokenizer_of(self, ctx):
        tokenizer = self.tokenizer
        if tokenizer is None or isinstance(tokenizer, str):
            tokenizer = getattr(getattr(ctx, "data", None), "tokenizer", None)
        if tokenizer is None or isinstance(tokenizer, str) or not callable(getattr(tokenizer, "decode", None)):
            raise RunError("lm_sampler needs a tokenizer with encode and decode; write tokenizer: \"@data.tokenizer\"")
        return tokenizer

    @staticmethod
    def window_of(ctx):
        limit = getattr(getattr(ctx, "data", None), "seq_len", None)
        if isinstance(limit, int) and not isinstance(limit, bool) and limit > 0:
            return limit
        return None

    @staticmethod
    def logits_of(result, size=None, length=None):
        if isinstance(result, dict) and "logits" in result:
            result = result["logits"]
        result = first(result)
        if not isinstance(result, torch.Tensor) or result.ndim not in (2, 3):
            shape = tuple(result.shape) if isinstance(result, torch.Tensor) else type(result).__name__
            raise RunError(f"lm_sampler needs logits of shape (batch, tokens, vocabulary) from the model, got {shape}")
        if result.ndim == 3:
            result = result[:, -1, :] if LmSampler.classes_last(result.shape, size, length) else result[:, :, -1]
        return result.float()

    @staticmethod
    def classes_last(shape, size, length):
        tokens, classes = shape[1], shape[2]
        if tokens == length:
            return True
        if classes == length:
            return False
        if size is None or classes == size:
            return True
        if tokens == size:
            return False
        raise RunError(f"lm_sampler cannot find the vocabulary axis of logits of shape {tuple(shape)}: the model "
                       f"read {length} tokens and the tokenizer has {size}; give (batch, tokens, vocabulary) or "
                       "(batch, vocabulary, tokens)")

    def pick(self, logits, generator):
        if self.temperature <= 0:
            return logits.argmax(dim=-1, keepdim=True)
        logits = logits / self.temperature
        if self.top_k:
            keep = min(self.top_k, logits.shape[-1])
            least = torch.topk(logits, keep, dim=-1).values[:, -1:]
            logits = logits.masked_fill(logits < least, float("-inf"))
        return torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=generator)

    def sample(self, ctx, n=None):
        count = count_of(self, n)
        tokenizer = self.tokenizer_of(ctx)
        size = size_of(tokenizer)
        model = module_of(self.model)
        device = device_of(ctx, model)
        prompt = [int(token) for token in tokenizer.encode(self.prompt)] if self.prompt else []
        skip = 0 if prompt else 1
        ids = torch.tensor([prompt or [0]] * count, dtype=torch.long, device=device)
        limit = self.window_of(ctx)
        rng = rng_of(ctx)
        generator = None if rng is None else rng.stream(STREAM, device)
        with evaluating(model):
            for _ in range(self.max_new_tokens):
                window = ids[:, -limit:] if limit else ids
                logits = self.logits_of(model(window), size, window.shape[1])
                ids = torch.cat([ids, self.pick(logits, generator)], dim=1)
        return [tokenizer.decode(row) for row in ids[:, skip:].cpu().tolist()]


REGISTRATIONS = (
    ("gan_sampler", "kalfa.std.samplers.GanSampler",
     "n samples of model(z) or model(z, label) with z from the samples stream and labels cycling over n_classes"),
    ("latent_sampler", "kalfa.std.samplers.LatentSampler", "n samples of a decoder fed standard normal latents"),
    ("ddpm_sampler", "kalfa.std.samplers.DdpmSampler",
     "reverse diffusion from noise of the given shape with the beta schedule; steps limits the number of steps"),
    ("lm_sampler", "kalfa.std.samplers.LmSampler",
     "autoregressive text from the prompt: max_new_tokens draws at temperature, top_k limits the choices"),
)

for name, target, doc in REGISTRATIONS:
    register(name, target, slot="sampler", doc=doc)
