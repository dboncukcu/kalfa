from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch import nn

from kalfa.core import aliases
from kalfa.core.device import Device
from kalfa.core.errors import BuildError, RunError
from kalfa.core.record import RecordWriter
from kalfa.core.rng import Rng
from kalfa.std.actions import WriteSamples
from kalfa.std.objectives import LinearBeta, NoiseTables
from kalfa.std.samplers import DdpmSampler, GanSampler, LatentSampler, LmSampler
from kalfa.std.samples import image_grid, save_samples


VOCAB = "abcdefgh"


def fake_context(tmp_path=None, seed=0, rng=True, data=None):
    record = None if tmp_path is None else RecordWriter(tmp_path / "run")
    return SimpleNamespace(rng=Rng(seed) if rng else None, device=None, turn=3, record=record, generate=None,
                           data=data)


class Images(nn.Module):
    def __init__(self, latent=3, n_classes=None):
        super().__init__()
        self.linear = nn.Linear(latent, 16)
        self.embed = nn.Embedding(n_classes, latent) if n_classes else None
        self.labels = []
        self.seen = []
        self.frozen = nn.Linear(1, 1)

    def forward(self, z, labels=None):
        self.seen.append((self.training, self.frozen.training, torch.is_grad_enabled()))
        if labels is not None:
            self.labels.append(labels.clone())
            z = z + self.embed(labels)
        return torch.tanh(self.linear(z)).reshape(-1, 1, 4, 4)


class Zeros(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.steps = []

    def forward(self, x, t):
        self.steps.append(t.clone())
        return torch.zeros_like(x) * self.scale


class Language(nn.Module):
    def __init__(self, vocabulary=len(VOCAB), window=None, wrap=False):
        super().__init__()
        self.table = nn.Parameter(torch.randn(vocabulary, vocabulary, generator=torch.Generator().manual_seed(0)))
        self.window = window
        self.wrap = wrap
        self.lengths = []

    def forward(self, ids):
        self.lengths.append(ids.shape[1])
        if self.window is not None:
            assert ids.shape[1] <= self.window
        logits = self.table[ids]
        return (logits, None) if self.wrap else logits


class Letters:
    def encode(self, text):
        return [VOCAB.index(letter) for letter in text]

    def decode(self, ids):
        return "".join(VOCAB[index] for index in ids)


def schedule(steps=10):
    return SimpleNamespace(steps=steps, betas=torch.linspace(1.0e-4, 0.02, steps))


def test_gan_sampler_draws_images_from_the_samples_stream():
    model = Images()
    sampler = GanSampler(model, latent=3, n=5)
    first = sampler.sample(fake_context())
    again = sampler.sample(fake_context())
    other = sampler.sample(fake_context(seed=1))
    assert first.shape == (5, 1, 4, 4)
    assert first.device.type == "cpu" and not first.requires_grad
    assert torch.equal(first, again)
    assert not torch.equal(first, other)
    z = Rng(0).normal("samples", (5, 3))
    with torch.no_grad():
        assert torch.allclose(first, model(z))
    assert sampler.sample(fake_context(), 2).shape == (2, 1, 4, 4)


def test_gan_sampler_cycles_the_labels():
    model = Images(n_classes=3)
    sampler = GanSampler(model, latent=3, n_classes=3, n=7)
    samples = sampler.sample(fake_context())
    assert samples.shape == (7, 1, 4, 4)
    assert model.labels[-1].tolist() == [0, 1, 2, 0, 1, 2, 0]
    assert model.labels[-1].dtype == torch.long


def test_samplers_run_in_eval_without_grad_and_restore_the_modes():
    model = Images()
    model.train()
    model.frozen.eval()
    GanSampler(model, latent=3, n=2).sample(fake_context())
    assert model.seen[-1] == (False, False, False)
    assert model.training and not model.frozen.training and model.linear.training
    model.eval()
    LatentSampler(model, latent=3, n=2).sample(fake_context())
    assert not model.training and not model.linear.training


def test_sampler_takes_the_module_of_an_entry_and_the_context_device():
    model = Images()
    entry = SimpleNamespace(module=model)
    ctx = fake_context()
    ctx.device = Device("cpu")
    samples = LatentSampler(entry, latent=3, n=4).sample(ctx)
    assert samples.shape == (4, 1, 4, 4)


def test_latent_sampler_feeds_standard_normal_latents():
    model = Images()
    sampler = LatentSampler(model, latent=3, n=6)
    samples = sampler.sample(fake_context())
    with torch.no_grad():
        expected = model(Rng(0).normal("samples", (6, 3)))
    assert torch.allclose(samples, expected)
    assert torch.equal(samples, sampler.sample(fake_context()))


def test_samplers_fall_back_to_the_global_generator_without_rng():
    samples = LatentSampler(Images(), latent=3, n=3).sample(fake_context(rng=False))
    assert samples.shape == (3, 1, 4, 4)


def test_sampler_rejects_a_non_positive_count():
    with pytest.raises(RunError, match="positive"):
        LatentSampler(Images(), latent=3, n=0).sample(fake_context())


def test_ddpm_sampler_with_few_steps_returns_the_shape_in_range():
    model = Zeros()
    sampler = DdpmSampler(model, schedule(10), shape=[1, 4, 4], n=3, steps=4)
    samples = sampler.sample(fake_context())
    assert samples.shape == (3, 1, 4, 4)
    assert float(samples.min()) >= -1.0 and float(samples.max()) <= 1.0
    assert [int(t[0]) for t in model.steps] == [9, 6, 3, 0]
    assert all(t.shape == (3,) and t.dtype == torch.long for t in model.steps)
    assert torch.equal(samples, sampler.sample(fake_context()))
    assert model.scale.requires_grad and model.training


def test_ddpm_sampler_timesteps():
    sampler = DdpmSampler(Zeros(), schedule(10), shape=[1, 2, 2], steps=4)
    assert sampler.timesteps(10) == [9, 6, 3, 0]
    assert DdpmSampler(Zeros(), schedule(10), shape=[1], steps=1).timesteps(10) == [9]
    assert DdpmSampler(Zeros(), schedule(10), shape=[1], steps=50).timesteps(10) == list(range(9, -1, -1))
    assert DdpmSampler(Zeros(), schedule(10), shape=[1]).timesteps(10) == list(range(9, -1, -1))
    with pytest.raises(RunError, match="positive"):
        DdpmSampler(Zeros(), schedule(10), shape=[1], steps=0).timesteps(10)


def test_ddpm_sampler_single_implicit_step_predicts_the_clean_image():
    betas = schedule(10).betas
    samples = DdpmSampler(Zeros(), schedule(10), shape=[1, 2, 2], n=4, steps=1).sample(fake_context())
    noise = Rng(0).normal("samples", (4, 1, 2, 2))
    expected = (noise / torch.cumprod(1.0 - betas, dim=0)[-1].sqrt()).clamp(-1.0, 1.0)
    assert torch.allclose(samples, expected)


def test_ddpm_sampler_ancestral_walks_every_step():
    model = Zeros()
    sampler = DdpmSampler(model, schedule(5), shape=[2], n=3)
    samples = sampler.sample(fake_context())
    assert samples.shape == (3, 2)
    assert [int(t[0]) for t in model.steps] == [4, 3, 2, 1, 0]
    assert float(samples.abs().max()) <= 1.0
    assert torch.equal(samples, sampler.sample(fake_context()))
    single = DdpmSampler(Zeros(), schedule(1), shape=[2], n=3).sample(fake_context())
    noise = Rng(0).normal("samples", (3, 2))
    assert torch.allclose(single, (noise / (1.0 - schedule(1).betas[0]).sqrt()).clamp(-1.0, 1.0))


def test_lm_sampler_returns_n_strings_of_the_asked_length():
    sampler = LmSampler(Language(), Letters(), prompt="ab", max_new_tokens=6, n=3)
    texts = sampler.sample(fake_context())
    assert isinstance(texts, list) and len(texts) == 3
    assert all(isinstance(text, str) and len(text) == 8 and text.startswith("ab") for text in texts)
    assert texts == sampler.sample(fake_context())
    assert len(sampler.sample(fake_context(), 1)) == 1


def test_lm_sampler_without_prompt_returns_only_the_new_tokens():
    texts = LmSampler(Language(), Letters(), max_new_tokens=5, n=2).sample(fake_context())
    assert len(texts) == 2 and all(len(text) == 5 for text in texts)
    assert all(set(text) <= set(VOCAB) for text in texts)


def test_lm_sampler_seeds_change_the_draws():
    sampler = LmSampler(Language(), Letters(), prompt="a", max_new_tokens=20, n=4, temperature=2.0)
    assert sampler.sample(fake_context(seed=0)) != sampler.sample(fake_context(seed=5))


def test_lm_sampler_top_one_is_greedy():
    top = LmSampler(Language(), Letters(), prompt="c", max_new_tokens=6, n=3, top_k=1).sample(fake_context())
    greedy = LmSampler(Language(), Letters(), prompt="c", max_new_tokens=6, n=1, temperature=0).sample(fake_context())
    assert len(set(top)) == 1
    assert top[0] == greedy[0]


def test_lm_sampler_reads_a_tuple_output_and_crops_to_the_data_window():
    model = Language(window=4, wrap=True)
    data = SimpleNamespace(seq_len=4)
    texts = LmSampler(model, Letters(), prompt="abc", max_new_tokens=5, n=2).sample(fake_context(data=data))
    assert all(len(text) == 8 for text in texts)
    assert max(model.lengths) == 4


def test_lm_sampler_without_rng_and_without_tokenizer():
    texts = LmSampler(Language(), Letters(), prompt="a", max_new_tokens=3, n=2).sample(fake_context(rng=False))
    assert all(len(text) == 4 for text in texts)
    found = LmSampler(Language(), None, prompt="a", max_new_tokens=2)
    assert len(found.sample(fake_context(data=SimpleNamespace(tokenizer=Letters())))[0]) == 3
    named = LmSampler(Language(), "char", prompt="a", max_new_tokens=2)
    assert len(named.sample(fake_context(data=SimpleNamespace(tokenizer=Letters())))[0]) == 3
    with pytest.raises(RunError, match="tokenizer"):
        found.sample(fake_context())
    with pytest.raises(RunError, match="tokenizer"):
        named.sample(fake_context())


def test_write_samples_writes_the_turn_images(tmp_path):
    ctx = fake_context(tmp_path)
    written = WriteSamples(GanSampler(Images(), latent=3, n=9), n=4)(ctx)
    directory = tmp_path / "run" / "samples"
    assert [path.name for path in written] == ["turn_3.pt", "turn_3.png"]
    assert all(path.parent == directory and path.exists() for path in written)
    samples = torch.load(directory / "turn_3.pt")
    assert samples.shape == (4, 1, 4, 4)
    with Image.open(directory / "turn_3.png") as image:
        assert image.mode == "RGB" and image.width >= 4


def test_write_samples_writes_the_turn_text(tmp_path):
    ctx = fake_context(tmp_path)
    written = WriteSamples(LmSampler(Language(), Letters(), prompt="a", max_new_tokens=4), n=2)(ctx)
    path = tmp_path / "run" / "samples" / "turn_3.txt"
    assert written == [path.with_suffix(".pt"), path]
    blocks = path.read_text(encoding="utf-8").strip().split("\n\n")
    assert len(blocks) == 2 and all(len(block) == 5 and block.startswith("a") for block in blocks)
    assert torch.load(path.with_suffix(".pt"), weights_only=False) == blocks


def test_write_samples_uses_generate_and_needs_a_sampler(tmp_path):
    ctx = fake_context(tmp_path)
    ctx.generate = LatentSampler(Images(), latent=3)
    ctx.turn = 7
    written = WriteSamples(n=2)(ctx)
    assert {path.name for path in written} == {"turn_7.pt", "turn_7.png"}
    ctx.generate = None
    with pytest.raises(RunError, match="sampler"):
        WriteSamples()(ctx)


def test_write_samples_without_record_draws_and_writes_nothing():
    model = Images()
    assert WriteSamples(LatentSampler(model, latent=3), n=2)(fake_context()) == []
    assert len(model.seen) == 1


def test_save_samples_formats(tmp_path):
    flat = save_samples(torch.zeros(3, 5), tmp_path, "flat")
    assert [path.name for path in flat] == ["flat.pt"]
    colour = save_samples(torch.rand(2, 3, 4, 4), tmp_path, "colour")
    assert [path.name for path in colour] == ["colour.pt", "colour.png"]
    text = save_samples("hello", tmp_path, "one")
    assert [path.name for path in text] == ["one.pt", "one.txt"]
    assert (tmp_path / "one.txt").read_text(encoding="utf-8") == "hello\n"
    assert torch.load(tmp_path / "one.pt", weights_only=False) == ["hello"]
    nan = torch.full((2, 1, 3, 3), float("nan"))
    assert [path.name for path in save_samples(nan, tmp_path, "nan")] == ["nan.pt", "nan.png"]


def test_image_grid_layout():
    grid = image_grid(torch.rand(5, 1, 4, 4), columns=3, pad=1)
    assert grid.shape == (2 * 5 + 1, 3 * 5 + 1, 3)
    assert float(grid.min()) >= 0.0 and float(grid.max()) <= 1.0
    flat = image_grid(torch.full((2, 4, 4), 0.5))
    assert flat.shape[2] == 3


def test_sampler_names_are_registered():
    for name, target in (("gan_sampler", "GanSampler"), ("latent_sampler", "LatentSampler"),
                         ("ddpm_sampler", "DdpmSampler"), ("lm_sampler", "LmSampler"),
                         ("write_samples", "WriteSamples")):
        assert aliases.resolve(name).endswith(target)


def test_ddpm_sampler_takes_noise_tables_and_the_std_schedule():
    tables = NoiseTables(schedule(6))
    samples = DdpmSampler(Zeros(), tables, shape=[1, 2, 2], n=2, steps=3).sample(fake_context())
    assert samples.shape == (2, 1, 2, 2)
    beta = LinearBeta(8)
    model = Zeros()
    samples = DdpmSampler(model, beta, shape=[1, 2, 2], n=2, steps=3).sample(fake_context())
    assert samples.shape == (2, 1, 2, 2)
    assert [int(t[0]) for t in model.steps] == [7, 4, 0]


class TorchLayout(Language):
    def forward(self, ids):
        return super().forward(ids).permute(0, 2, 1)


class Sized(Letters):
    size = len(VOCAB)


def test_lm_sampler_reads_the_torch_layout_by_the_window_and_the_vocabulary():
    for tokenizer, data in ((Letters(), None), (Sized(), SimpleNamespace(seq_len=3))):
        expected = LmSampler(Language(), tokenizer, prompt="ab", max_new_tokens=6, temperature=0)
        permuted = LmSampler(TorchLayout(), tokenizer, prompt="ab", max_new_tokens=6, temperature=0)
        assert permuted.sample(fake_context(data=data)) == expected.sample(fake_context(data=data))
    empty = LmSampler(TorchLayout(), Sized(), max_new_tokens=4, temperature=0).sample(fake_context())
    assert empty == LmSampler(Language(), Sized(), max_new_tokens=4, temperature=0).sample(fake_context())


def test_lm_sampler_finds_the_vocabulary_axis():
    assert LmSampler.classes_last((2, 3, 12), 8, 3)
    assert not LmSampler.classes_last((2, 8, 3), 8, 3)
    assert LmSampler.classes_last((2, 5, 5), 8, 5)
    assert LmSampler.classes_last((2, 1, 8), 8, 3)
    assert not LmSampler.classes_last((2, 8, 1), 8, 3)
    assert LmSampler.classes_last((2, 1, 12), None, 3)
    with pytest.raises(RunError, match="vocabulary axis"):
        LmSampler.classes_last((2, 1, 12), 8, 3)
    logits = torch.arange(24.0).reshape(2, 4, 3)
    assert torch.equal(LmSampler.logits_of(logits, 3, 4), logits[:, -1, :])
    assert torch.equal(LmSampler.logits_of(logits, 4, 3), logits[:, :, -1])
    assert torch.equal(LmSampler.logits_of(logits, 4, 2), logits[:, :, -1])
    assert torch.equal(LmSampler.logits_of({"logits": logits[:, -1, :]}), logits[:, -1, :])


def test_write_samples_checks_for_a_sampler_at_setup():
    ctx = fake_context()
    action = WriteSamples(n=2)
    action.path = "rules.dump.run"
    with pytest.raises(BuildError, match=r"rules\.dump\.run: write_samples needs a sampler"):
        action.bind(ctx)
    ctx.generate = LatentSampler(Images(), latent=3)
    action.bind(ctx)
    WriteSamples(LatentSampler(Images(), latent=3)).bind(fake_context())
