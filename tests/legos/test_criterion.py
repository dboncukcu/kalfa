import functools
import math

import pytest
import torch
from cirak.registry import registry

from helpers import build
from kalfa.std import STD_URIS


CRITERIA = sorted(uri for uri in STD_URIS if uri.startswith("/criterion/"))


def test_criterion_scope_is_the_seven_kalfa_criteria():
    assert CRITERIA == ["/criterion/kalfa/bce_logits", "/criterion/kalfa/cross_entropy", "/criterion/kalfa/huber",
                        "/criterion/kalfa/log_cosh", "/criterion/kalfa/mae", "/criterion/kalfa/mse",
                        "/criterion/kalfa/weighted_mse"]


def test_every_criterion_is_partial_and_aliased_by_its_name():
    for uri in CRITERIA:
        assert registry.facts(uri).partial is True
        assert registry.aliases()[uri.rsplit("/", 1)[1]] == uri
    assert isinstance(build("/criterion/kalfa/huber", delta=2.0), functools.partial)
    assert build("/criterion/kalfa/huber", delta=2.0).keywords == {"delta": 2.0}


def test_mse_is_the_mean_squared_error_over_every_element():
    mse = build("/criterion/kalfa/mse")
    value = mse(torch.tensor([[1.0], [3.0]]), torch.tensor([0.0, 1.0]))
    assert float(value) == 2.5


def test_mse_casts_integer_targets_to_the_prediction_dtype():
    mse = build("/criterion/kalfa/mse")
    value = mse(torch.tensor([[1.0], [3.0]]), torch.tensor([0, 1]))
    assert value.dtype == torch.float32
    assert float(value) == 2.5


def test_weighted_mse_scales_every_target_column_by_its_weight():
    weighted = build("/criterion/kalfa/weighted_mse", weights=[1.0, 0.5])
    value = weighted(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.zeros(2, 2))
    assert float(value) == 5.0


def test_weighted_mse_takes_a_tensor_of_weights():
    weighted = build("/criterion/kalfa/weighted_mse", weights=torch.tensor([1.0, 0.5]))
    assert float(weighted(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.zeros(2, 2))) == 5.0


def test_weighted_mse_refuses_a_weight_count_that_differs_from_the_columns():
    weighted = build("/criterion/kalfa/weighted_mse", weights=[1.0, 0.5, 2.0])
    with pytest.raises(ValueError, match=r"weighted_mse has 3 weights for 2 target columns"):
        weighted(torch.zeros(2, 2), torch.zeros(2, 2))


def test_mae_is_the_mean_absolute_error():
    mae = build("/criterion/kalfa/mae")
    assert float(mae(torch.tensor([[1.0], [3.0]]), torch.tensor([0.0, 1.0]))) == 1.5


def test_huber_is_quadratic_inside_delta_and_linear_outside():
    huber = build("/criterion/kalfa/huber", delta=1.0)
    assert float(huber(torch.tensor([[1.0], [3.0]]), torch.tensor([0.0, 1.0]))) == 1.0
    wide = build("/criterion/kalfa/huber", delta=3.0)
    assert float(wide(torch.tensor([[1.0], [3.0]]), torch.tensor([0.0, 1.0]))) == 1.25


def test_huber_defaults_to_delta_one():
    assert float(build("/criterion/kalfa/huber")(torch.tensor([[1.0], [3.0]]), torch.tensor([0.0, 1.0]))) == 1.0


def test_log_cosh_is_the_log_of_the_cosh_of_the_error():
    log_cosh = build("/criterion/kalfa/log_cosh")
    value = log_cosh(torch.tensor([[1.0], [0.0]]), torch.tensor([0.0, 0.0]))
    assert float(value) == pytest.approx(math.log(math.cosh(1.0)) / 2.0)
    assert float(log_cosh(torch.tensor([[2.0]]), torch.tensor([2.0]))) == 0.0


def test_cross_entropy_of_flat_logits_is_log_of_the_class_count():
    cross_entropy = build("/criterion/kalfa/cross_entropy")
    value = cross_entropy(torch.zeros(2, 2), torch.tensor([0, 1]))
    assert float(value) == pytest.approx(math.log(2.0))


def test_cross_entropy_averages_the_per_sample_terms():
    cross_entropy = build("/criterion/kalfa/cross_entropy")
    logits = torch.tensor([[math.log(3.0), 0.0], [0.0, 0.0]])
    value = cross_entropy(logits, torch.tensor([0, 1]))
    assert float(value) == pytest.approx((math.log(4.0 / 3.0) + math.log(2.0)) / 2.0)


def test_cross_entropy_weight_reweights_the_classes():
    cross_entropy = build("/criterion/kalfa/cross_entropy", weight=[1.0, 3.0])
    logits = torch.tensor([[math.log(3.0), 0.0], [0.0, 0.0]])
    value = cross_entropy(logits, torch.tensor([0, 1]))
    assert float(value) == pytest.approx((math.log(4.0 / 3.0) + 3.0 * math.log(2.0)) / 4.0)


def test_cross_entropy_label_smoothing_mixes_in_the_uniform_target():
    cross_entropy = build("/criterion/kalfa/cross_entropy", label_smoothing=0.5)
    value = cross_entropy(torch.tensor([[math.log(3.0), 0.0]]), torch.tensor([0]))
    nll = math.log(4.0 / 3.0)
    uniform = (math.log(4.0 / 3.0) + math.log(4.0)) / 2.0
    assert float(value) == pytest.approx(0.5 * nll + 0.5 * uniform)


def test_cross_entropy_flattens_sequence_logits_and_targets():
    cross_entropy = build("/criterion/kalfa/cross_entropy")
    value = cross_entropy(torch.zeros(1, 2, 4), torch.tensor([[1, 3]]))
    assert float(value) == pytest.approx(math.log(4.0))


def test_bce_logits_of_zero_logits_is_log_two():
    bce = build("/criterion/kalfa/bce_logits")
    value = bce(torch.tensor([[0.0], [0.0]]), torch.tensor([1.0, 0.0]))
    assert float(value) == pytest.approx(math.log(2.0))


def test_bce_logits_reads_the_sigmoid_of_the_logit():
    bce = build("/criterion/kalfa/bce_logits")
    assert float(bce(torch.tensor([math.log(3.0)]), torch.tensor([1]))) == pytest.approx(math.log(4.0 / 3.0))


def test_bce_logits_pos_weight_scales_the_positive_term():
    bce = build("/criterion/kalfa/bce_logits", pos_weight=3.0)
    value = bce(torch.tensor([[0.0], [0.0]]), torch.tensor([1.0, 0.0]))
    assert float(value) == pytest.approx(2.0 * math.log(2.0))
