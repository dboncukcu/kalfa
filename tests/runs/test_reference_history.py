import json
import math
from pathlib import Path

import pytest

from kalfa.std.common.history import History


pytestmark = pytest.mark.slow

CRITERIA = ["mse_lin", "huber_hv", "mae_lin", "bce", "bce_pos"]
OBJECTIVE = ["ws/mse_lin", "ws/huber_hv", "ws", "total", "total/primary", "total/lambda/mae_lin",
             "total/inf/mae_lin", "total/lambda/ws.huber_hv", "total/inf/ws.huber_hv"]
METRICS = ["rmse_lin", "mae_orig", "recon_lin", "auroc", "acc"]
RATES = ["lr/main", "lr/main/lambdas", "lr/main/stem", "lr/aux"]
BOOKKEEPING = ("turn", "global_step", "rules", "seconds", "lr/", "minimizes/", "effect/")


def expected_keys(turn):
    keys = ["turn", "global_step"]
    for prefix in ("train", "val", "test"):
        keys += [f"{prefix}/{name}" for name in CRITERIA]
        if prefix != "train" and turn == 2:
            keys.append(f"{prefix}/probe_heavy")
        keys += [f"{prefix}/{name}" for name in OBJECTIVE]
        keys += [f"{prefix}/{name}" for name in METRICS]
        if prefix != "train" and turn == 2:
            keys.append(f"{prefix}/ap_every")
    keys += RATES + ["minimizes/main", "minimizes/aux"]
    if turn >= 2:
        keys.append("effect/tail_stem.trainable")
    if turn == 3:
        keys += ["effect/main.stem.lr", "effect/ws.terms.mse_lin"]
    return keys + ["seconds", "rules"]


def test_every_history_line_holds_exactly_the_expected_keys_in_order(reference):
    history = History.read(reference.record)
    assert [line["turn"] for line in history] == [1, 2, 3]
    for line in history:
        assert list(line) == expected_keys(line["turn"])
        assert all(math.isfinite(value) for key, value in line.items() if isinstance(value, float))


def test_the_global_step_counts_the_updates_under_accumulation(reference):
    history = History.read(reference.record)
    batches = json.loads((Path(reference.record) / "data.json").read_text())["loaders"]["train"]["batches"]
    per_turn = math.ceil(batches / 2)
    assert [line["global_step"] for line in history] == [per_turn, 2 * per_turn, 3 * per_turn]


def test_the_rules_fire_on_the_turns_their_triggers_name(reference):
    history = History.read(reference.record)
    assert [line["rules"] for line in history] == [["unfreeze", "swap_aux"], ["cool_stem"], ["cool_stem", "reweight"]]
    assert [line["minimizes/main"] for line in history] == ["total"] * 3
    assert [line["minimizes/aux"] for line in history] == ["bce", "bce_pos", "bce_pos"]
    assert [line.get("effect/tail_stem.trainable") for line in history] == [None, True, True]
    assert history[2]["effect/main.stem.lr"] == {"times": 0.5} and history[2]["effect/ws.terms.mse_lin"] == 2.0


def test_the_learning_rates_follow_the_groups_the_rule_and_the_schedule(reference):
    history = History.read(reference.record)
    assert [line["lr/main"] for line in history] == [1e-3] * 3
    assert [line["lr/main/lambdas"] for line in history] == [-1e-2] * 3
    assert [line["lr/main/stem"] for line in history] == [5e-4, 5e-4, 2.5e-4]
    aux = [line["lr/aux"] for line in history]
    assert aux[0] > aux[1] > aux[2] > 0.0 and aux[0] < 0.05


def test_the_multipliers_climb_away_from_their_start(reference):
    history = History.read(reference.record)
    assert history[-1]["train/total/lambda/mae_lin"] != 0.0
    assert history[-1]["train/total/lambda/ws.huber_hv"] != 0.0
    for line in history:
        assert line["train/total"] != line["train/total/primary"]
        assert line["train/ws"] == pytest.approx(line["train/ws/mse_lin"] * (2.0 if line["turn"] == 3 else 1.0)
                                                 + 0.5 * line["train/ws/huber_hv"], rel=1e-5)


def test_a_definition_with_every_and_sets_appears_only_where_it_says(reference):
    history = History.read(reference.record)
    for line in history:
        assert "train/probe_heavy" not in line and "train/ap_every" not in line
        assert ("val/probe_heavy" in line) == (line["turn"] == 2)
        assert ("test/ap_every" in line) == (line["turn"] == 2)


def test_the_step_lines_record_every_update_of_both_optimizers(reference):
    history = History.read(reference.record)
    steps = History.read_steps(reference.record)
    assert len(steps) == history[-1]["global_step"]
    for line in steps:
        assert list(line) == ["step", "turn", "loss/main", "lr/main", "grad_norm/main", "loss/aux", "lr/aux",
                              "grad_norm/aux"]
    assert [line["step"] for line in steps] == list(range(1, len(steps) + 1))
    assert [line["turn"] for line in steps] == sorted(line["turn"] for line in steps)
    assert steps[-1]["step"] == history[-1]["global_step"]


def test_the_report_carries_the_history_without_its_bookkeeping(reference):
    history = History.read(reference.record)
    traced = reference.report.outputs["history"]
    stripped = [{key: value for key, value in line.items() if not key.startswith(BOOKKEEPING)} for line in history]
    assert [set(line) for line in traced] == [set(line) for line in stripped]
    assert traced == stripped
