import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional

from kalfa.core.rng import Rng
from kalfa.std.data.text import CharTokenizer
from kalfa.std.data.transforms import PRESETS, Normalize
from kalfa.std.losses import cross_entropy
from kalfa.std.samplers import LmSampler


class TorchLayoutLm(nn.Module):
    def __init__(self, vocab, wanted):
        super().__init__()
        self.vocab = vocab
        self.wanted = wanted
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids):
        batch, length = input_ids.shape
        logits = torch.zeros(batch, length, self.vocab)
        logits[:, -1, self.wanted] = 5.0
        return logits.permute(0, 2, 1)


def test_lm_sampler_reads_torch_layout_logits_by_the_vocabulary_size():
    tokenizer = CharTokenizer().fit(["abcdefgh"])
    wanted = tokenizer.encode("e")[0]
    assert wanted != tokenizer.size - 1
    ctx = SimpleNamespace(rng=Rng(0), data=SimpleNamespace(tokenizer=tokenizer, seq_len=4), device=None)
    sampler = LmSampler(TorchLayoutLm(tokenizer.size, wanted), tokenizer, prompt="ab", max_new_tokens=3,
                        temperature=0.0)
    assert sampler.sample(ctx) == ["abeee"]


def test_cross_entropy_takes_an_explicit_classes_last_knob_for_the_ambiguous_shape():
    generator = torch.Generator().manual_seed(0)
    logits = torch.randn(2, 5, 5, generator=generator)
    target = torch.randint(0, 5, (2, 5), generator=generator)
    last = functional.cross_entropy(logits.reshape(-1, 5), target.reshape(-1))
    assert float(cross_entropy(classes_last=True)(logits, target)) == pytest.approx(float(last))
    assert float(cross_entropy(classes_last=False)(logits, target)) == pytest.approx(
        float(functional.cross_entropy(logits, target)))


def test_normalize_keeps_an_explicit_std_next_to_a_preset_mean():
    assert Normalize(mean="imagenet").std == PRESETS["imagenet"][1]
    assert Normalize(mean="imagenet", std=0.5).std == [0.5]
    assert Normalize(mean="imagenet", std=1.0).std == [1.0]


def test_cross_entropy_and_lm_sampler_agree_on_the_unambiguous_lm_layout():
    generator = torch.Generator().manual_seed(1)
    logits = torch.randn(2, 3, 7, generator=generator)
    target = torch.randint(0, 7, (2, 3), generator=generator)
    last = functional.cross_entropy(logits.reshape(-1, 7), target.reshape(-1))
    assert float(cross_entropy()(logits, target)) == pytest.approx(float(last))
    assert math.isfinite(float(last))
