import json
import shlex
import shutil
from pathlib import Path

import pytest

from helpers import example, write_housing
from kalfa import sweep as sweeper
from kalfa.api import check
from kalfa.cli import build_parser, main
from kalfa.collect import collect
from kalfa.config import parse_sets
from kalfa.record import Record

CONFIG = example("14_sweep_grid")
OPTUNA = ["sweep.strategy={uri: optuna, params: {trials: 2, seed: 1}}",
          "sweep.space.lr={low: 1.0e-4, high: 1.0e-2, log: true}"]


def flags(sets):
    return [argument for text in sets for argument in ("--set", text)]


@pytest.fixture
def housing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_housing(tmp_path / "housing.parquet")
    return tmp_path


def test_plan_count_show_and_check(housing, capsys):
    plan = sweeper.plan([CONFIG], parse_sets([]))
    assert plan.total == 6 and plan.deterministic and plan.root == Path("runs/sweep_housing")
    assert plan.uri == "/strategy/kalfa/grid" and sweeper.point_of(plan, 4) == {"lr": 1e-4, "width": 64}
    assert check([CONFIG], parse_sets([])).problems == []
    kinds = [problem.kind for problem in check([CONFIG], parse_sets(["sweep.space.ghost=[1, 2]"])).problems]
    assert "unresolved_ref" in kinds
    kinds = [problem.kind for problem in check([CONFIG], parse_sets(["sweep.objective.mode=up"])).problems]
    assert "invalid_value" in kinds
    kinds = [problem.kind for problem in check([CONFIG], parse_sets(["sweep.space.lr={low: 0.0, high: 1.0}"])).problems]
    assert "sweep_space" in kinds
    assert main(["sweep", CONFIG, "--count"]) == 0
    assert capsys.readouterr().out.strip() == "6"
    assert main(["sweep", CONFIG, "--show", "4"]) == 0
    assert json.loads(capsys.readouterr().out) == {"lr": 1e-4, "width": 64}
    assert main(["sweep", CONFIG, "--show", "4", *flags(OPTUNA)]) == 1
    assert main(["sweep", CONFIG, "--id", "0", *flags(OPTUNA)]) == 1


def test_plan_writes_the_root_once_and_prepares_the_data(housing, capsys):
    assert main(["sweep", CONFIG, "--plan", "--record", "runs/planned"]) == 0
    root = Path("runs/planned")
    assert sorted(path.name for path in root.iterdir()) == ["manifest.json", "sweep.plan", "sweep.sh", "sweep.sub"]
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["kind"] == "sweep" and manifest["total"] == 6 and manifest["prepared"] is False
    assert manifest["strategy"] == "/strategy/kalfa/grid" and manifest["objective"]["monitor"] == "val/rmse"
    assert "N = 6" in (root / "sweep.plan").read_text() and "queue $(N)" in (root / "sweep.sub").read_text()
    line = next(line for line in (root / "sweep.sh").read_text().splitlines() if line.startswith("kalfa "))
    filled = {"$CONFIG": CONFIG, "$1": "0", "$ROOT": str(root)}
    parsed = build_parser().parse_args([filled.get(word, word) for word in shlex.split(line)[1:]])
    assert parsed.command == "sweep" and parsed.point_id == 0 and parsed.no_progress and parsed.log == "info"
    assert "condor_submit" in capsys.readouterr().out
    (root / "sweep.sh").write_text("edited\n")
    assert main(["sweep", CONFIG, "--plan", "--prepare-data", "--record", "runs/planned"]) == 0
    assert (root / "sweep.sh").read_text() == "edited\n" and (root / "data" / "manifest.json").exists()
    assert json.loads((root / "manifest.json").read_text())["prepared"] is True
    assert main(["sweep", CONFIG, "--plan", "--prepare-data", "--record", "runs/swept", "--set",
                 "data.batch=$width$"]) == 1
    assert "differs per point" in capsys.readouterr().err
    assert sweeper.swept_in_data({"data": {"batch": "$width$"}}, {"width": [1]}) == ["width"]


@pytest.mark.slow
def test_one_point_by_id(housing):
    assert main(["sweep", CONFIG, "--id", "4", "-p", "epochs=2"]) == 0
    record = Path("runs/sweep_housing/0004")
    info = json.loads((record / "sweep.json").read_text())
    assert info["id"] == 4 and info["point"] == {"lr": 1e-4, "width": 64} and info["total"] == 6
    assert info["strategy"] == "/strategy/kalfa/grid" and info["record"] == str(record)
    assert info["objective"]["monitor"] == "val/rmse" and info["objective"]["mode"] == "min"
    assert info["objective"]["at"] == "best" and info["objective"]["turn"] in (1, 2)
    resolved = (record / "resolved.yaml").read_text()
    assert "width: 64" in resolved and "lr: 0.0001" in resolved
    assert (record / "history.jsonl").exists() and not Path("runs/sweep_housing/0000").exists()
    manifest = json.loads((record / "manifest.json").read_text())
    assert manifest["kind"] == "point" and manifest["values"] == {"lr": 1e-4, "width": 64} and manifest["id"] == 4
    assert main(["sweep", CONFIG, "--plan", "--prepare-data", "--record", "runs/prepared_sweep"]) == 0
    assert main(["sweep", CONFIG, "--id", "1", "-p", "epochs=1", "--record", "runs/prepared_sweep"]) == 0
    point = json.loads((Path("runs/prepared_sweep") / "0001" / "manifest.json").read_text())
    assert point["prepared"].endswith("data") and (Path("runs/prepared_sweep") / "0001" / "fitted").exists()


@pytest.mark.slow
@pytest.mark.subprocess
def test_local_loop_and_collect_with_a_missing_folder(housing):
    assert main(["sweep", CONFIG, "-p", "epochs=1", "--record", "runs/grid"]) == 0
    root = Path("runs/grid")
    assert sorted(path.name for path in root.iterdir() if path.is_dir()) == [f"{index:04d}" for index in range(6)]
    assert (root / "manifest.json").exists() and (root / "sweep.sub").exists()
    shutil.rmtree(root / "0002")
    (root / "0003" / "sweep.json").unlink()
    kind, text, target = collect(["runs/grid"])
    assert kind == "sweep" and target == "runs/grid"
    assert (root / "sweep.csv").exists() and (root / "sweep.md").exists()
    summary = json.loads((root / "sweep.json").read_text())
    assert [row["id"] for row in summary["points"]] == [0, 1, 4, 5]
    assert summary["skipped"] == [{"dir": "0003", "status": "unfinished"}]
    assert summary["best"]["id"] in (0, 1, 4, 5) and "best: point" in text and "skipped: 0003" in text
    assert {"lr", "width", "objective", "turn", "turns", "val/rmse"} <= set(summary["points"][0])
    assert main(["collect", "runs/grid"]) == 0
    assert main(["sweep", CONFIG, "-p", "epochs=1", "--record", "runs/grid"]) == 1
    assert (root / "0002" / "sweep.json").exists() and not (root / "0003" / "sweep.json").exists()


@pytest.mark.slow
@pytest.mark.subprocess
def test_optuna_loop_feeds_the_objective_back(housing):
    assert main(["sweep", CONFIG, "-p", "epochs=1", "--record", "runs/opt", *flags(OPTUNA)]) == 0
    root = Path("runs/opt")
    points = [json.loads((root / f"{index:04d}" / "sweep.json").read_text()) for index in range(2)]
    assert all(1e-4 <= point["point"]["lr"] <= 1e-2 for point in points)
    assert all(point["point"]["width"] in (64, 128) for point in points)
    assert all(point["strategy"] == "/strategy/kalfa/optuna" and point["total"] == 2 for point in points)
    kind, text, target = collect(["runs/opt"])
    assert kind == "sweep" and "best: point" in text


def test_a_stopped_root_starts_no_point(housing, capsys):
    assert main(["sweep", CONFIG, "--plan", "--record", "runs/halted"]) == 0
    Record(Path("runs/halted")).request_stop("test")
    capsys.readouterr()
    assert main(["sweep", CONFIG, "-p", "epochs=1", "--record", "runs/halted"]) == 1
    out = capsys.readouterr().out
    assert "stop requested" in out and "0/" in out and not list(Path("runs/halted").glob("0*"))
    assert main(["sweep", CONFIG, "--id", "0", "--record", "runs/halted"]) == 1
    assert "the sweep is stopped" in capsys.readouterr().err
