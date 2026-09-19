import functools

import pytest
import torch
from cirak.build import Graph, GraphNode
from cirak.registry import registry
from torch import nn

from helpers import build, tiny_model
from kalfa.std import STD_URIS
from kalfa.std.builder.kalfa.module import Module
from kalfa.std.common.generation import write_samples, write_turn_samples
from kalfa.std.pre.base import Field, Prep


GENERATE_URIS = sorted(uri for uri in STD_URIS if uri.startswith("/generate/"))
TOKENIZER_MESSAGE = "lm_sampler needs a fitted tokenizer among the preprocessors"
SCHEDULE_MESSAGE = "the noise schedule must be a schedule lego with a steps param (linear_betas)"


class Labelled(nn.Module):
    def forward(self, noise, labels):
        return torch.cat([noise, labels.reshape(-1, 1).float()], dim=1)


class ZeroNoise(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(1))
        self.steps = []

    def forward(self, x, t):
        self.steps.append(t.clone())
        return self.scale * x


class FirstTokenLogits(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
        self.bias = nn.Parameter(torch.zeros(vocab))

    def forward(self, ids):
        first = ids[:, :1].expand(-1, ids.shape[1])
        return nn.functional.one_hot(first, num_classes=self.vocab).float() + self.bias


def wrap(module, inputs, outputs=("out",)):
    graph = Graph(tuple(inputs), tuple(outputs), (GraphNode("node", module, tuple(inputs), tuple(outputs)),))
    return Module(graph, seed=1, name="net")


def char_prep(text):
    tokenizer = build("/pre/kalfa/char_tokenizer")
    tokenizer.fit([text])
    return Prep([Field("text", ["tok"], False, ["text"])], {"tok": {"text": tokenizer}}, {}, {}, []), tokenizer


def expected_reverse_diffusion(seed, n, shape, betas):
    rng = torch.Generator().manual_seed(seed)
    size = (n, *shape)
    x = torch.randn(size, generator=rng)
    for step in reversed(range(len(betas))):
        x = x / (1.0 - betas[step]).sqrt()
        if step > 0:
            x = x + betas[step].sqrt() * torch.randn(size, generator=rng)
    return x.clamp(-1.0, 1.0)


def test_the_generate_pack_holds_the_three_samplers_as_partials_with_refs():
    assert GENERATE_URIS == ["/generate/kalfa/ddpm_sampler", "/generate/kalfa/gan_sampler",
                             "/generate/kalfa/lm_sampler"]
    for uri in GENERATE_URIS:
        assert registry.facts(uri).partial is True
        assert registry.aliases()[uri.rsplit("/", 1)[1]] == uri
    assert registry.facts("/generate/kalfa/gan_sampler").refs == {"model": "model"}
    assert registry.facts("/generate/kalfa/lm_sampler").refs == {"model": "model"}
    assert registry.facts("/generate/kalfa/ddpm_sampler").refs == {"model": "model", "schedule": "schedule"}


def test_gan_sampler_feeds_seeded_latent_noise_to_the_generator():
    generator = tiny_model(in_features=4, out_features=3, seed=2)
    generator.train()
    sampler = build("/generate/kalfa/gan_sampler", latent=4, n=6)
    assert isinstance(sampler, functools.partial) and sampler.keywords == {"latent": 4, "n": 6}
    samples = sampler(models={"g": generator}, prep=None, rng=torch.Generator().manual_seed(0), model="g")
    assert tuple(samples.shape) == (6, 3) and samples.dtype == torch.float32 and samples.device.type == "cpu"
    assert not samples.requires_grad and generator.training is False
    noise = torch.randn((6, 4), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        assert torch.equal(samples, generator(noise))
    again = sampler(models={"g": generator}, prep=None, rng=torch.Generator().manual_seed(0), model="g")
    assert torch.equal(again, samples)
    other = sampler(models={"g": generator}, prep=None, rng=torch.Generator().manual_seed(1), model="g")
    assert not torch.equal(other, samples)
    unseeded = sampler(models={"g": generator}, prep=None, rng=None, model="g")
    assert tuple(unseeded.shape) == (6, 3)
    assert tuple(build("/generate/kalfa/gan_sampler", latent=2)(models={"g": tiny_model(in_features=2)}, prep=None,
                                                                 rng=None, model="g").shape) == (64, 1)


def test_conditional_gan_samples_cycle_through_the_classes():
    generator = wrap(Labelled(), ["z", "c"])
    sampler = build("/generate/kalfa/gan_sampler", latent=4, n=7, conditional=True, n_classes=3)
    samples = sampler(models={"g": generator}, prep=None, rng=torch.Generator().manual_seed(0), model="g")
    assert tuple(samples.shape) == (7, 5)
    assert samples[:, 4].tolist() == [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0]
    assert torch.equal(samples[:, :4], torch.randn((7, 4), generator=torch.Generator().manual_seed(0)))


def test_a_sampler_names_the_model_it_cannot_find():
    sampler = build("/generate/kalfa/gan_sampler", latent=4)
    with pytest.raises(KeyError) as caught:
        sampler(models={"g": tiny_model(), "f": tiny_model()}, prep=None, rng=None, model="h")
    assert caught.value.args[0] == "generate: 'h' is no model; the models are ['f', 'g']"


def test_ddpm_sampler_walks_the_noise_schedule_backwards_from_pure_noise():
    core = ZeroNoise()
    net = wrap(core, ["x", "t"])
    net.train()
    schedule = build("/schedule/kalfa/linear_betas", steps=3)
    assert isinstance(schedule, functools.partial) and schedule.keywords == {"steps": 3}
    sampler = build("/generate/kalfa/ddpm_sampler", shape=[1, 2, 2], n=4)
    samples = sampler(models={"unet": net}, prep=None, rng=torch.Generator().manual_seed(1), model="unet",
                      schedule=schedule)
    assert tuple(samples.shape) == (4, 1, 2, 2) and samples.dtype == torch.float32 and samples.device.type == "cpu"
    assert float(samples.abs().max()) <= 1.0 and net.training is False
    assert [step.tolist() for step in core.steps] == [[2] * 4, [1] * 4, [0] * 4]
    assert all(step.dtype == torch.long for step in core.steps)
    betas = torch.tensor([1e-4, 1e-4 + (0.02 - 1e-4) * 0.5, 0.02], dtype=torch.float32)
    assert torch.allclose(samples, expected_reverse_diffusion(1, 4, (1, 2, 2), betas), atol=1e-6)
    assert torch.equal(sampler(models={"unet": net}, prep=None, rng=torch.Generator().manual_seed(1), model="unet",
                               schedule=schedule), samples)
    default = build("/generate/kalfa/ddpm_sampler", shape=[2])
    assert tuple(default(models={"unet": net}, prep=None, rng=None, model="unet", schedule=schedule).shape) == (64, 2)


def test_ddpm_sampler_needs_a_schedule_with_steps():
    sampler = build("/generate/kalfa/ddpm_sampler", shape=[2], n=2)
    net = wrap(ZeroNoise(), ["x", "t"])
    for schedule in (lambda step: 0.01, functools.partial(lambda step, start: start, start=0.01)):
        with pytest.raises(ValueError) as caught:
            sampler(models={"unet": net}, prep=None, rng=None, model="unet", schedule=schedule)
        assert str(caught.value) == SCHEDULE_MESSAGE


def test_lm_sampler_extends_the_prompt_one_token_at_a_time_over_the_window():
    prep, tokenizer = char_prep("ab")
    assert tokenizer.chars == ["\n", "a", "b"] and tokenizer.size == 3
    core = FirstTokenLogits(3)
    net = wrap(core, ["input_ids"])
    sampler = build("/generate/kalfa/lm_sampler", prompt="ab", max_new_tokens=3, temperature=1e-9)
    assert sampler.keywords == {"prompt": "ab", "max_new_tokens": 3, "temperature": 1e-9}
    whole = sampler(models={"lm": net}, prep=prep, rng=torch.Generator().manual_seed(0), model="lm")
    assert isinstance(whole, str) and whole == "abaaa" and net.training is False
    windowed = sampler(models={"lm": net}, prep=prep, rng=torch.Generator().manual_seed(0), model="lm", context=1)
    assert windowed == "abbbb"
    core.seq_len = 1
    assert sampler(models={"lm": net}, prep=prep, rng=None, model="lm") == "abbbb"
    assert sampler(models={"lm": net}, prep=prep, rng=None, model="lm", context=5) == "abaaa"
    assert sampler(models={"lm": net}, prep=prep, rng=None, model="lm", max_new_tokens=0) == "ab"


def test_lm_sampler_draws_from_the_softmax_at_temperature_one():
    prep, tokenizer = char_prep("ab")
    net = wrap(FirstTokenLogits(3), ["input_ids"])
    sampler = build("/generate/kalfa/lm_sampler", prompt="ba", max_new_tokens=40)
    text = sampler(models={"lm": net}, prep=prep, rng=torch.Generator().manual_seed(2), model="lm")
    assert len(text) == 42 and text.startswith("ba") and set(text) <= set(tokenizer.chars)
    assert set(text[2:]) == set(tokenizer.chars)
    assert text.count("b") > 15
    again = sampler(models={"lm": net}, prep=prep, rng=torch.Generator().manual_seed(2), model="lm")
    assert again == text


def test_lm_sampler_needs_a_fitted_tokenizer():
    net = wrap(FirstTokenLogits(3), ["input_ids"])
    sampler = build("/generate/kalfa/lm_sampler", prompt="ab", max_new_tokens=1)
    with pytest.raises(ValueError) as caught:
        sampler(models={"lm": net}, prep=None, rng=None, model="lm")
    assert str(caught.value) == TOKENIZER_MESSAGE
    scaler = build("/pre/sklearn/standard_scaler")
    plain = Prep([Field("x0", ["std"], False, ["x0"])], {"std": {"x0": scaler}}, {}, {}, [])
    assert plain.tokenizer() is None
    with pytest.raises(ValueError) as caught:
        sampler(models={"lm": net}, prep=plain, rng=None, model="lm")
    assert str(caught.value) == TOKENIZER_MESSAGE


def test_write_samples_saves_the_tensor_and_draws_a_grid_of_images(tmp_path):
    images = torch.rand(3, 1, 2, 2, generator=torch.Generator().manual_seed(0))
    target = tmp_path / "generated" / "final"
    assert write_samples(images, target) is None
    assert sorted(path.name for path in target.iterdir()) == ["grid.png", "samples.pt"]
    assert torch.equal(torch.load(target / "samples.pt"), images)
    assert (target / "grid.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    flat = tmp_path / "flat"
    write_samples(torch.zeros(4, 2), flat)
    assert sorted(path.name for path in flat.iterdir()) == ["samples.pt"]
    text = tmp_path / "text"
    write_samples("ROMEO: love", text)
    assert sorted(path.name for path in text.iterdir()) == ["samples.txt"]
    assert (text / "samples.txt").read_text() == "ROMEO: love"


def test_write_turn_samples_stamps_the_files_with_the_turn(tmp_path):
    images = torch.rand(2, 3, 4, 4, generator=torch.Generator().manual_seed(1))
    assert write_turn_samples(images, tmp_path / "turns", 7) is None
    assert sorted(path.name for path in (tmp_path / "turns").iterdir()) == ["turn_0007.png", "turn_0007.pt"]
    assert torch.equal(torch.load(tmp_path / "turns" / "turn_0007.pt"), images)
    write_turn_samples("verona", tmp_path / "turns", 12)
    assert (tmp_path / "turns" / "turn_0012.txt").read_text() == "verona"
    write_turn_samples(torch.ones(3), tmp_path / "turns", 1234)
    assert sorted(path.name for path in (tmp_path / "turns").iterdir()) == ["turn_0007.png", "turn_0007.pt",
                                                                             "turn_0012.txt", "turn_1234.pt"]
