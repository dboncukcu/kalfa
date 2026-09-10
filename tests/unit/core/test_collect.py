import json

import pytest

from kalfa.collect import collect, fold_summary, sweep_table


def test_collect_sweep_table(tmp_path):
    for name, lr in (("a", 0.1), ("b", 0.2)):
        run_dir = tmp_path / name
        run_dir.mkdir()
        (run_dir / "resolved.yaml").write_text(f"params:\n  lr: {lr}\n  seed: 1\nrecord: x\n")
        (run_dir / "history.jsonl").write_text(json.dumps({"turn": 1, "val/rmse": lr * 2, "test/rmse": lr}) + "\n")
    kind, text, target = collect([str(tmp_path / "a"), str(tmp_path / "b")])
    assert kind == "sweep" and "| lr |" in text.replace("| dir | turns | lr |", "| lr |") and target == str(tmp_path)
    table = json.loads((tmp_path / "sweep.json").read_text())
    assert table["varying"] == ["lr"] and table["rows"][1]["val/rmse"] == 0.4
    with pytest.raises(ValueError):
        collect([str(tmp_path / "nowhere")])


def test_fold_summary_math():
    runs = [{"dir": "r0", "config": {"params": {"fold": 0}}, "history": [{"test/rmse": 1.0, "train/l": 5.0}]},
            {"dir": "r1", "config": {"params": {"fold": 1}}, "history": [{"test/rmse": 3.0, "train/l": 5.0}]}]
    result = fold_summary(runs)
    assert result["summary"] == {"test/rmse": {"mean": 2.0, "std": pytest.approx(2 ** 0.5)}}
    assert [fold["turns"] for fold in result["folds"]] == [1, 1]
    assert sweep_table(runs)["varying"] == ["fold"]
