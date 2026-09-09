"""const, pack, merge, identity, the history log, the progress display and the console log."""

import json
import logging
from io import StringIO

import kalfa  # noqa: F401
from kalfa.std.log import Progress, console, history, history_line, logger, node_path, progress, sink, turn_line
from kalfa.std.optimizer import sgd
from kalfa.std.util import const, identity, merge, pack
from helpers import tiny_model


def test_const_pack_identity():
    value = {"global_step": 0}
    copy = const(value)
    assert copy == value and copy is not value
    assert pack({"a": 1}) == {"a": 1} and pack(None) == {}
    assert identity(5) == 5


def test_merge_prefixes_and_orders_the_sets():
    parts = {"test_metrics": {"r": 3.0}, "train_metrics": {"l": 1.0}, "valid_metrics": {"r": 2.0, "l": 1.5}}
    assert merge(parts) == {"train/l": 1.0, "val/r": 2.0, "val/l": 1.5, "test/r": 3.0}
    assert list(merge(parts)) == ["train/l", "val/r", "val/l", "test/r"]
    assert merge({}) == {}


def test_history_line_and_file(tmp_path):
    optimizer = sgd({"m": tiny_model()}, {"lr": 0.3}, None, "l")
    line = history_line({"train/l": 0.5}, {"turn": 2, "global_step": 8}, {"m": optimizer}, {"fired": ["a"]})
    assert line == {"turn": 2, "global_step": 8, "train/l": 0.5, "lr/m": 0.3, "rules": ["a"]}
    bar = progress()
    assert Progress.current is bar
    history(bar, {"train/l": 0.5}, 1, {"turn": 2, "global_step": 8}, {"m": optimizer}, {"fired": []}, str(tmp_path))
    history(bar, {"train/l": 0.4}, 2, {"turn": 3, "global_step": 12}, {"m": optimizer}, {}, str(tmp_path))
    lines = [json.loads(text) for text in (tmp_path / "history.jsonl").read_text().splitlines()]
    assert [line["turn"] for line in lines] == [2, 3] and lines[1]["rules"] == []
    sink({"kind": "started", "path": "training.epochs", "total": 5})
    assert bar.total == 5 and bar.bar.total == 5
    bar.close()
    assert history(None, {}, 0, {}, {}, {}, None) is None


def test_turn_line_and_node_path():
    line = {"turn": 3, "global_step": 12, "train/loss": 0.5, "val/rmse": float("nan"), "rules": ["to_huber"]}
    assert turn_line(line, 2.0) == "turn 3  train/loss 0.5  rules: to_huber  (2.0s)"
    assert turn_line({"turn": 1, "val/rmse": 0.25}) == "turn 1  val/rmse 0.25"
    assert node_path("training.epochs[2].body.turn") == "training.epochs[2].turn"
    assert node_path("training.epochs[2].body") == "training.epochs[2]"
    assert node_path("data.source") == "data.source"


def test_console_writes_the_kalfa_lines_to_the_stream_until_it_stops():
    stream = StringIO()
    with console(logging.INFO, stream):
        logger("data.source").info("reading x.parquet")
        logger("data.source").debug("only at debug")
        sink({"kind": "started", "path": "data.source", "node": "source"})
    logger("data.source").info("after the console closed")
    text = stream.getvalue()
    assert "INFO   data.source     reading x.parquet" in text
    assert "only at debug" not in text and "started" not in text
    assert "after the console closed" not in text


def test_no_progress_keeps_the_bar_out():
    with console(None, progress=False):
        bar = progress()
        history(bar, {"train/l": 0.5}, 1, {"turn": 1, "global_step": 4}, {}, {}, None)
        assert bar.bar is None
    assert Progress.enabled


def test_turn_line_carries_every_value():
    line = {"turn": 2, "global_step": 8, "train/mse": 1.0, "val/mse": 2.0, "test/mse": 3.0, "train/rmse": 4.0,
            "val/rmse": 5.0, "test/rmse": 6.0, "train/mae": 7.0, "val/mae": 8.0, "lr/model": 0.001, "rules": []}
    text = turn_line(line)
    assert text.startswith("turn 2  train/mse 1")
    assert "val/mae 8" in text and "lr/model 0.001" in text and "rules" not in text


def test_console_at_debug_shows_every_node():
    stream = StringIO()
    with console(logging.DEBUG, stream):
        sink({"kind": "started", "path": "training.epochs[0].body.turn", "node": "turn"})
        sink({"kind": "finished", "path": "training.epochs[0].body.turn", "node": "turn", "ms": 1500.0})
        sink({"kind": "failed", "path": "data.source", "node": "source", "error": "no file"})
    text = stream.getvalue()
    assert "DEBUG  training.epochs[0].turn  started" in text
    assert "finished (1.50s)" in text and "ERROR" in text and "failed: no file" in text
