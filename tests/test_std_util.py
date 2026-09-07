"""const, pack, merge, identity, the history log and the progress display."""

import json

import kalfa  # noqa: F401
from kalfa.std.log import Progress, history, history_line, progress, sink
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
