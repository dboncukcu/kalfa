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
    assert kind == "sweep" and target == str(tmp_path)
    assert "── RUNS" in text and "lr" in text and "|" not in text
    report = collect([str(tmp_path / "a"), str(tmp_path / "b")], markdown=True)[1]
    assert "| dir | turns | lr |" in report and report == (tmp_path / "sweep.md").read_text()
    table = json.loads((tmp_path / "sweep.json").read_text())
    assert table["varying"] == ["lr"] and table["rows"][1]["val/rmse"] == 0.4
    with pytest.raises(ValueError):
        collect([str(tmp_path / "nowhere")])


def test_collect_sweep_root_leaves_the_metrics_to_the_report(tmp_path):
    root = tmp_path / "sweep"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps(
        {"kind": "sweep", "objective": {"monitor": "val/rmse", "mode": "min", "at": "best"}}))
    for index, (lr, value) in enumerate(((0.1, 0.5), (0.2, 0.4))):
        point = root / f"{index:04d}"
        point.mkdir()
        (point / "sweep.json").write_text(json.dumps(
            {"id": index, "point": {"lr": lr}, "total": 3,
             "objective": {"monitor": "val/rmse", "mode": "min", "at": "best", "value": value, "turn": 2},
             "record": str(point)}))
        (point / "history.jsonl").write_text(
            json.dumps({"turn": 2, "val/rmse": value, "test/rmse": value + 0.1}) + "\n")
    (root / "0002").mkdir()
    (root / "0002" / "run.json").write_text(json.dumps({"status": "failed"}))

    kind, text, target = collect([str(root)])
    assert kind == "sweep" and target == str(root)
    assert "2 of 3 finished" in text and "test/rmse" not in text and "2 per point" in text
    assert "failed" in text and "0002" in text
    assert [line for line in text.splitlines() if line.startswith("  * ")][0].split()[1] == "1"
    report = collect([str(root)], markdown=True)[1]
    assert "| test/rmse |" in report and report == (root / "sweep.md").read_text()
    assert "best: point 1 with val/rmse=0.4 at turn 2" in report and "failed: 1 (0002)" in report


def test_fold_summary_math():
    runs = [{"dir": "r0", "config": {"params": {"fold": 0}}, "history": [{"test/rmse": 1.0, "train/l": 5.0}]},
            {"dir": "r1", "config": {"params": {"fold": 1}}, "history": [{"test/rmse": 3.0, "train/l": 5.0}]}]
    result = fold_summary(runs)
    assert result["summary"] == {"test/rmse": {"mean": 2.0, "std": pytest.approx(2 ** 0.5)}}
    assert [fold["turns"] for fold in result["folds"]] == [1, 1]
    assert sweep_table(runs)["varying"] == ["fold"]
