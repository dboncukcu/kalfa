"""const, pack, merge, identity, the history log and the monitor."""

import json
import logging
from io import StringIO

import kalfa  # noqa: F401
from kalfa.std.common.history import History
from kalfa.std.common.log import Monitor, logger_for, node_path, turn_line
from kalfa.std.lego.kalfa.history import history
from kalfa.std.lego.kalfa.values import const, identity, merge, pack
from kalfa.std.optimizer.torch.optimizers import Sgd
from helpers import tiny_model


def test_const_pack_identity():
    value = {"global_step": 0}
    copy = const(value)
    assert copy == value and copy is not value
    assert pack({"a": 1}) == {"a": 1} and pack(None) == {}
    assert identity(5) == 5


def test_merge_prefixes_and_orders_the_sets():
    parts = {"test_metrics": {"r": 3.0}, "train_metrics": {"l": 1.0}, "valid_metrics": {"r": 2.0, "l": 1.5}}
    prefixes = {"train": "train", "valid": "val", "test": "test"}
    assert merge(parts, prefixes) == {"train/l": 1.0, "val/r": 2.0, "val/l": 1.5, "test/r": 3.0}
    assert list(merge(parts, prefixes)) == ["train/l", "val/r", "val/l", "test/r"]
    assert merge({}, prefixes) == {}


def test_history_line_and_file(tmp_path):
    optimizer = Sgd({"m": tiny_model()}, {"lr": 0.3}, None, "l")
    line = History.line({"train/l": 0.5}, {"turn": 2, "global_step": 8}, {"m": optimizer}, {"fired": ["a"]})
    assert line == {"turn": 2, "global_step": 8, "train/l": 0.5, "lr/m": 0.3, "rules": ["a"]}
    monitor = Monitor()
    history(monitor, {"train/l": 0.5}, 1, {"turn": 2, "global_step": 8}, {"m": optimizer}, {"fired": []},
            record=str(tmp_path))
    history(monitor, {"train/l": 0.4}, 2, {"turn": 3, "global_step": 12}, {"m": optimizer}, {}, {"loss": "other"},
            str(tmp_path))
    lines = [json.loads(text) for text in (tmp_path / "history.jsonl").read_text().splitlines()]
    assert [line["turn"] for line in lines] == [2, 3] and lines[1]["rules"] == []
    assert lines[0]["minimizes/m"] == "l" and lines[1]["minimizes/m"] == "other"
    monitor.sink({"kind": "started", "path": "training.epochs", "total": 5})
    assert monitor.total == 5 and monitor.bar.total == 5
    monitor.close()
    assert monitor.bar is None
    assert history(None, {}, 0, {}, {}, {}, None) is None


def test_turn_line_and_node_path():
    line = {"turn": 3, "global_step": 12, "train/loss": 0.5, "val/rmse": float("nan"), "rules": ["to_huber"]}
    assert turn_line(line, 2.0) == "turn 3  train/loss 0.5  rules: to_huber  (2.0s)"
    assert turn_line({"turn": 1, "val/rmse": 0.25}) == "turn 1  val/rmse 0.25"
    assert node_path("training.epochs[2].body.turn") == "training.epochs[2].turn"
    assert node_path("training.epochs[2].body") == "training.epochs[2]"
    assert node_path("data.source") == "data.source"


def test_monitor_writes_the_kalfa_lines_to_the_stream_until_it_stops():
    stream = StringIO()
    with Monitor(logging.INFO, stream=stream) as monitor:
        logger_for("data.source").info("reading x.parquet")
        logger_for("data.source").debug("only at debug")
        monitor.sink({"kind": "started", "path": "data.source", "node": "source"})
    logger_for("data.source").info("after the monitor closed")
    text = stream.getvalue()
    assert "INFO   data.source     reading x.parquet" in text
    assert "only at debug" not in text and "started" not in text
    assert "after the monitor closed" not in text


def test_no_progress_keeps_the_bar_out():
    with Monitor(None, progress=False) as monitor:
        history(monitor, {"train/l": 0.5}, 1, {"turn": 1, "global_step": 4}, {}, {}, None)
        assert monitor.bar is None


def test_the_monitor_draws_the_inner_bar_and_logs_every_n_steps():
    stream = StringIO()
    with Monitor(logging.INFO, progress="steps", stream=stream, log_every=2) as monitor:
        monitor.turn_begins(3)
        assert monitor.inner is not None and monitor.inner.total == 3
        for step in (1, 2, 3):
            monitor.step({"step": step, "turn": 1, "loss/m": 0.5 / step, "lr/m": 0.1})
        assert monitor.inner.n == 3
        monitor.turn({"turn": 1, "global_step": 3, "train/l": 0.5})
        assert monitor.inner is None
    text = stream.getvalue()
    assert "training.step   step 2  loss/m 0.25  lr/m 0.1" in text and "step 1 " not in text
    assert "step 3 " not in text and "turn 1  train/l 0.5" in text
    with Monitor(None, progress=True) as monitor:
        monitor.turn_begins(5)
        assert monitor.inner is not None and monitor.inner.total == 5
        monitor.turn_begins(1)
        assert monitor.inner is None
    with Monitor(None, progress="turns") as monitor:
        monitor.turn_begins(5)
        assert monitor.inner is None


def test_turn_line_carries_every_value():
    line = {"turn": 2, "global_step": 8, "train/mse": 1.0, "val/mse": 2.0, "test/mse": 3.0, "train/rmse": 4.0,
            "val/rmse": 5.0, "test/rmse": 6.0, "train/mae": 7.0, "val/mae": 8.0, "lr/model": 0.001, "rules": []}
    text = turn_line(line)
    assert text.startswith("turn 2  train/mse 1")
    assert "val/mae 8" in text and "lr/model 0.001" in text and "rules" not in text


def test_monitor_at_debug_shows_every_node():
    stream = StringIO()
    with Monitor(logging.DEBUG, stream=stream) as monitor:
        monitor.sink({"kind": "started", "path": "training.epochs[0].body.turn", "node": "turn"})
        monitor.sink({"kind": "finished", "path": "training.epochs[0].body.turn", "node": "turn", "ms": 1500.0})
        monitor.sink({"kind": "failed", "path": "data.source", "node": "source", "error": "no file"})
    text = stream.getvalue()
    assert "DEBUG  training.epochs[0].turn  started" in text
    assert "finished (1.50s)" in text and "ERROR" in text and "failed: no file" in text
