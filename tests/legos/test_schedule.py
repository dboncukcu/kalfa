import functools
import math

from cirak.registry import registry

from helpers import build
from kalfa.std import STD_URIS


SCHEDULES = sorted(uri for uri in STD_URIS if uri.startswith("/schedule/"))


def test_schedule_scope_is_the_four_kalfa_schedules():
    assert SCHEDULES == ["/schedule/kalfa/linear_betas", "/schedule/kalfa/linear_warmup",
                         "/schedule/kalfa/step_decay", "/schedule/kalfa/warmup_cosine"]


def test_every_schedule_is_a_partial_aliased_by_its_name():
    for uri in SCHEDULES:
        assert registry.facts(uri).partial is True
        assert registry.aliases()[uri.rsplit("/", 1)[1]] == uri
    assert isinstance(build("/schedule/kalfa/step_decay", step_size=2, gamma=0.5), functools.partial)


def test_linear_warmup_ramps_from_start_to_end_then_holds():
    schedule = build("/schedule/kalfa/linear_warmup", start=0.0, end=1.0, steps=4)
    assert [schedule(step) for step in (0, 1, 2, 3, 4, 8)] == [0.0, 0.25, 0.5, 0.75, 1.0, 1.0]


def test_linear_warmup_clamps_a_negative_step_to_start():
    schedule = build("/schedule/kalfa/linear_warmup", start=0.2, end=1.0, steps=4)
    assert schedule(-3) == 0.2


def test_linear_warmup_without_steps_is_the_end_value():
    assert build("/schedule/kalfa/linear_warmup", start=0.0, end=0.7, steps=0)(0) == 0.7


def test_warmup_cosine_warms_linearly_then_decays_to_zero_at_total():
    schedule = build("/schedule/kalfa/warmup_cosine", warmup=4, total=12)
    assert [schedule(step) for step in (0, 1, 3, 4)] == [0.0, 0.25, 0.75, 1.0]
    assert schedule(8) == 0.5
    assert schedule(12) == 0.0
    assert schedule(20) == 0.0


def test_warmup_cosine_quarter_of_the_decay_is_cos_of_quarter_pi():
    schedule = build("/schedule/kalfa/warmup_cosine", warmup=0, total=8)
    assert schedule(2) == 0.5 * (1.0 + math.cos(math.pi / 4.0))


def test_warmup_cosine_is_zero_after_a_warmup_that_covers_the_total():
    schedule = build("/schedule/kalfa/warmup_cosine", warmup=4, total=4)
    assert schedule(2) == 0.5
    assert schedule(4) == 0.0
    assert schedule(9) == 0.0


def test_step_decay_multiplies_by_gamma_every_step_size_updates():
    schedule = build("/schedule/kalfa/step_decay", step_size=2, gamma=0.5)
    assert [schedule(step) for step in (0, 1, 2, 3, 4, 7)] == [1.0, 1.0, 0.5, 0.5, 0.25, 0.125]


def test_linear_betas_rises_from_start_to_end_over_the_steps():
    schedule = build("/schedule/kalfa/linear_betas", steps=5, start=0.1, end=0.5)
    assert [schedule(step) for step in range(5)] == [0.1, 0.2, 0.30000000000000004, 0.4, 0.5]


def test_linear_betas_defaults_to_the_ddpm_range():
    schedule = build("/schedule/kalfa/linear_betas", steps=1000)
    assert schedule(0) == 1e-4
    assert schedule(999) == 0.02


def test_linear_betas_with_one_step_is_the_end_value():
    assert build("/schedule/kalfa/linear_betas", steps=1, start=0.1, end=0.5)(0) == 0.5


def test_linear_betas_keeps_steps_in_its_keywords_for_the_diffusion_legos():
    assert build("/schedule/kalfa/linear_betas", steps=7).keywords == {"steps": 7}
