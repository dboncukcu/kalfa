import json
import re
import shlex
import shutil
import stat
from pathlib import Path

import pytest

from data import write_housing
from helpers import CONFIGS, config_path
from kalfa import sweep
from kalfa.cli import build_parser, main
from kalfa.config import parse_sets
from kalfa.record import Record


CONFIG = config_path("sweep")
POINTS = [{"lr": 0.01, "width": 16}, {"lr": 0.01, "width": 32}, {"lr": 0.001, "width": 16},
          {"lr": 0.001, "width": 32}]
OPTUNA = ["sweep.strategy={uri: optuna, params: {trials: 2, seed: 1}}",
          "sweep.space.lr={low: 1.0e-4, high: 1.0e-2, log: true}"]
REFUSAL = ("/strategy/kalfa/optuna proposes points from the objectives fed back; it runs in the local loop "
           "(kalfa sweep cfg.yaml) and takes no --id or --show\n")


def flags(sets):
    return [argument for text in sets for argument in ("--set", text)]


def point_of(root, index):
    return json.loads((Path(root) / f"{index:04d}" / "sweep.json").read_text())


@pytest.fixture
def housing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_housing(tmp_path / "housing.parquet")
    return tmp_path


def test_plan_reads_the_sweep_section_without_running():
    plan = sweep.plan([CONFIG], parse_sets([]))
    assert plan.total == 4 and plan.deterministic and plan.root == Path("runs/grid")
    assert plan.uri == "/strategy/kalfa/grid" and list(plan.space) == ["lr", "width"] and plan.paths == [CONFIG]
    assert plan.objective == {"monitor": "val/rmse", "mode": "min", "at": "best"}
    assert [sweep.point_of(plan, index) for index in range(4)] == POINTS
    with pytest.raises(ValueError, match=r"^point 4 is outside the grid of 4 points$"):
        sweep.point_of(plan, 4)
    assert sweep.plan([CONFIG], parse_sets([]), record="elsewhere").root == Path("elsewhere")
    assert sweep.plan([CONFIG], parse_sets(["sweep.record=runs/other"])).root == Path("runs/other")


def test_strategy_of_refuses_what_is_no_strategy():
    with pytest.raises(sweep.SweepError, match=r"^sweep.strategy must be a strategy lego: a short name or \{uri, "
                                               r"params\}$"):
        sweep.strategy_of({"strategy": 5}, {})
    with pytest.raises(sweep.SweepError, match=r"^sweep.strategy 'nope' is not a known strategy lego$"):
        sweep.strategy_of({"strategy": "nope"}, {})
    with pytest.raises(sweep.SweepError, match=r"^/optimizer/torch/adam is a optimizer lego, sweep.strategy needs a "
                                               r"strategy$"):
        sweep.strategy_of({"strategy": "adam"}, {"adam": "/optimizer/torch/adam"})
    with pytest.raises(sweep.SweepError, match=r"^the config has no sweep section$"):
        sweep.plan([config_path("tiny")], parse_sets([]))
    with pytest.raises(ValueError, match=r"^sweep.space.lr: the list of choices is empty$"):
        sweep.plan([CONFIG], parse_sets(["sweep.space.lr=[]"]))


@pytest.mark.parametrize("name", ["random", "sobol"])
def test_random_and_sobol_draw_their_points_by_id(name):
    plan = sweep.plan([CONFIG], parse_sets([f"sweep.strategy={{uri: {name}, params: {{count: 3, seed: 1}}}}"]))
    assert plan.total == 3 and plan.deterministic and plan.uri == f"/strategy/kalfa/{name}"
    points = [sweep.point_of(plan, index) for index in range(3)]
    assert all(point["lr"] in (0.01, 0.001) and point["width"] in (16, 32) for point in points)
    assert points == [sweep.point_of(plan, index) for index in range(3)]
    with pytest.raises(ValueError, match=rf"^point 3 is outside the 3 {name} points$"):
        sweep.point_of(plan, 3)


def test_count_and_show_answer_without_running(capsys):
    assert main(["sweep", CONFIG, "--count"]) == 0
    assert capsys.readouterr().out == "4\n"
    assert main(["sweep", CONFIG, "--show", "1"]) == 0
    assert capsys.readouterr().out == '{"lr": 0.01, "width": 32}\n'
    assert main(["sweep", CONFIG, "--show", "9"]) == 1
    assert capsys.readouterr().err == "point 9 is outside the grid of 4 points\n"
    with pytest.raises(SystemExit) as failure:
        main(["sweep", CONFIG, "--point", "{}"])
    assert failure.value.code == 2
    assert capsys.readouterr().err == "kalfa: error: --point needs --id\n"


def test_plan_writes_the_root_once(housing, capsys):
    root = Path("runs/planned")
    assert main(["sweep", CONFIG, "--plan", "--record", str(root)]) == 0
    assert capsys.readouterr().out == (
        "runs/planned: manifest.json, sweep.plan, sweep.sub, sweep.sh\n"
        f"submit with: condor_submit runs/planned/sweep.sub; or run the points here: kalfa sweep {CONFIG} "
        "--record runs/planned\n")
    assert sorted(path.name for path in root.iterdir()) == ["manifest.json", "sweep.plan", "sweep.sh", "sweep.sub"]
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["kind"] == "sweep" and manifest["total"] == 4 and manifest["prepared"] is False
    assert manifest["strategy"] == "/strategy/kalfa/grid" and manifest["config"] == [CONFIG]
    assert manifest["objective"] == {"monitor": "val/rmse", "mode": "min", "at": "best"}
    assert manifest["space"] == {"lr": "Choices(values=[0.01, 0.001])", "width": "Choices(values=[16, 32])"}
    assert (root / "sweep.plan").read_text() == f"N = 4\nCONFIG = {CONFIG}\nROOT = {root.resolve()}\n"
    submit = (root / "sweep.sub").read_text()
    assert "include : sweep.plan\n" in submit and submit.endswith("queue $(N)\n")
    assert f"initialdir            = {CONFIGS}\n" in submit and "transfer_input_files  = sweep.yaml\n" in submit
    script = (root / "sweep.sh").read_text()
    assert script.startswith("#!/usr/bin/env bash\n") and (root / "sweep.sh").stat().st_mode & stat.S_IXUSR
    assert f'CONFIG="{CONFIG}"\n' in script and f'ROOT="{root.resolve()}"\n' in script
    line = next(line for line in script.splitlines() if line.startswith("kalfa "))
    filled = {"$CONFIG": CONFIG, "$1": "2", "$ROOT": str(root)}
    parsed = build_parser().parse_args([filled.get(word, word) for word in shlex.split(line)[1:]])
    assert parsed.command == "sweep" and parsed.config == [CONFIG] and parsed.point_id == 2
    assert parsed.record == str(root) and parsed.no_progress and parsed.log == "info"
    (root / "sweep.sh").write_text("edited\n")
    (root / "sweep.sub").write_text("mine\n")
    assert main(["sweep", CONFIG, "--plan", "--record", str(root)]) == 0
    assert capsys.readouterr().out.startswith("runs/planned: manifest.json, sweep.plan\n")
    assert (root / "sweep.sh").read_text() == "edited\n" and (root / "sweep.sub").read_text() == "mine\n"
    assert json.loads((root / "manifest.json").read_text())["started"] == manifest["started"]


def test_plan_prepare_data_runs_the_data_block_once(housing, capsys):
    root = Path("runs/planned")
    assert main(["sweep", CONFIG, "--plan", "--prepare-data", "--record", str(root)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("runs/planned: manifest.json, sweep.plan, sweep.sub, sweep.sh "
                          "(the data prepared under data/)\n")
    manifest = json.loads((root / "data" / "manifest.json").read_text())
    assert manifest["kind"] == "data" and manifest["sets"] == ["train", "valid", "test"]
    assert manifest["sizes"] == {"train": 1400, "valid": 300, "test": 300} and manifest["layout"] == "table"
    assert sorted(path.name for path in (root / "data").iterdir()) == [
        "data.json", "fitted", "manifest.json", "test.parquet", "train.parquet", "valid.parquet"]
    assert json.loads((root / "manifest.json").read_text())["prepared"] is True
    assert main(["sweep", CONFIG, "--plan", "--prepare-data", "--record", "runs/swept", "--set",
                 "data.batch=$width$"]) == 1
    assert capsys.readouterr().err == ("the data section reads the swept ['width'], so the data differs per point; "
                                       "--prepare-data cannot prepare it once\n")
    assert sweep.swept_in_data({"data": {"batch": "$width$"}}, {"width": [1]}) == ["width"]
    assert sweep.swept_in_data({"data": {"batch": 1, "source": {"params": {"path": "$lr$"}}}}, {"lr": [1]}) == ["lr"]
    assert sweep.swept_in_data({"model": {"nodes": ["$width$"]}}, {"width": [1]}) == []


@pytest.mark.slow
def test_id_runs_one_point_by_itself(housing, capsys):
    assert main(["sweep", CONFIG, "--id", "1", "-p", "epochs=1", "--record", "runs/one", "--no-progress"]) == 0
    out = capsys.readouterr().out
    assert re.fullmatch(r"point 1 \{'lr': 0\.01, 'width': 32\}: val/rmse=[\d.]+ at turn 1; record runs/one/0001\n", out)
    record = Path("runs/one/0001")
    assert sorted(path.name for path in Path("runs/one").iterdir()) == ["0001"]
    entry = json.loads((record / "sweep.json").read_text())
    assert entry["id"] == 1 and entry["point"] == {"lr": 0.01, "width": 32} and entry["total"] == 4
    assert entry["strategy"] == "/strategy/kalfa/grid" and entry["record"] == "runs/one/0001"
    assert entry["objective"]["monitor"] == "val/rmse" and entry["objective"]["mode"] == "min"
    assert entry["objective"]["at"] == "best" and entry["objective"]["turn"] == 1
    assert isinstance(entry["objective"]["value"], float) and entry["objective"]["value"] > 0
    manifest = json.loads((record / "manifest.json").read_text())
    assert manifest["kind"] == "point" and manifest["id"] == 1 and manifest["values"] == {"lr": 0.01, "width": 32}
    assert manifest["root"] == "runs/one" and manifest["prepared"] is None
    assert manifest["params"] == {"epochs": 1, "lr": 0.01, "seed": 3, "width": 32}
    resolved = (record / "resolved.yaml").read_text()
    assert re.search(r"^  width: 32 +# --set overrides ", resolved, re.M) and "out_features: 32" in resolved
    assert (record / "history.jsonl").exists() and (record / "final" / "state.pt").exists()
    assert json.loads((record / "run.json").read_text())["status"] == "ok"


@pytest.mark.slow
def test_id_starts_from_the_prepared_data_of_the_root(housing, capsys):
    assert main(["sweep", CONFIG, "--plan", "--prepare-data", "--record", "runs/prepared"]) == 0
    assert main(["sweep", CONFIG, "--id", "0", "--record", "runs/prepared", "--no-progress"]) == 0
    capsys.readouterr()
    record = Path("runs/prepared/0000")
    manifest = json.loads((record / "manifest.json").read_text())
    assert manifest["prepared"] == "runs/prepared/data" and (record / "fitted" / "preprocessors" / "plan.json").exists()
    assert point_of("runs/prepared", 0)["point"] == POINTS[0]


@pytest.mark.slow
@pytest.mark.subprocess
def test_local_loop_runs_every_point_and_collect_summarizes_the_root(housing, capsys):
    assert main(["sweep", CONFIG, "--record", "runs/grid", "--no-progress"]) == 0
    out = capsys.readouterr().out
    root = Path("runs/grid")
    assert "runs/grid: manifest.json, sweep.plan, sweep.sub, sweep.sh\n" in out
    assert all(f"runs/grid/{index:04d}: {point} -> val/rmse=" in out for index, point in enumerate(POINTS))
    assert out.endswith("4/4 points finished under runs/grid; summarize with: kalfa collect runs/grid\n")
    assert sorted(path.name for path in root.iterdir() if path.is_dir()) == ["0000", "0001", "0002", "0003"]
    entries = [point_of(root, index) for index in range(4)]
    assert [entry["point"] for entry in entries] == POINTS and [entry["id"] for entry in entries] == [0, 1, 2, 3]
    assert all(entry["objective"]["turn"] == 1 and entry["total"] == 4 for entry in entries)
    assert main(["collect", "runs/grid"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("── SWEEP ")
    assert "  objective  val/rmse (min, best)\n" in out and "  points     4 of 4 finished\n" in out
    assert out.endswith("wrote sweep.csv, sweep.json and sweep.md under runs/grid\n")
    summary = json.loads((root / "sweep.json").read_text())
    best = min(entries, key=lambda entry: entry["objective"]["value"])
    assert summary["best"] == {"id": best["id"], "point": best["point"], "value": best["objective"]["value"],
                               "turn": 1, "dir": f"runs/grid/{best['id']:04d}"}
    assert [row["id"] for row in summary["points"]] == [0, 1, 2, 3] and summary["skipped"] == []
    assert f"  best       point {best['id']}, val/rmse = " in out
    assert (root / "sweep.csv").read_text().splitlines()[0] == ("id,lr,width,objective,turn,turns,val/mse,val/rmse,"
                                                                 "test/mse,test/rmse")
    assert (root / "sweep.md").read_text().startswith("# sweep: val/rmse (min, best)\n\n4 of 4 points finished.\n")
    shutil.rmtree(root / "0002")
    (root / "0003" / "sweep.json").unlink()
    assert main(["collect", "runs/grid"]) == 0
    out = capsys.readouterr().out
    summary = json.loads((root / "sweep.json").read_text())
    assert [row["id"] for row in summary["points"]] == [0, 1]
    assert summary["skipped"] == [{"dir": "0003", "status": "unfinished"}]
    assert "  points     2 of 3 finished\n" in out and re.search(r"^  unfinished\s+1: 0003$", out, re.M)
    assert main(["sweep", CONFIG, "--record", "runs/grid", "--no-progress"]) == 1
    out = capsys.readouterr().out
    assert "runs/grid: manifest.json, sweep.plan\n" in out and "runs/grid/0000: done, val/rmse=" in out
    assert "runs/grid/0003: exists without sweep.json (unfinished or failed), skipped; remove it to rerun\n" in out
    assert f"runs/grid/0002: {POINTS[2]} -> val/rmse=" in out
    assert out.endswith("3/4 points finished under runs/grid; summarize with: kalfa collect runs/grid\n")
    assert point_of(root, 2)["point"] == POINTS[2] and not (root / "0003" / "sweep.json").exists()


@pytest.mark.slow
@pytest.mark.subprocess
def test_optuna_feeds_the_objective_back_and_runs_only_in_the_loop(housing, capsys):
    assert main(["sweep", CONFIG, "--show", "0", *flags(OPTUNA)]) == 1
    assert capsys.readouterr().err == REFUSAL
    assert main(["sweep", CONFIG, "--id", "0", "--record", "runs/opt", *flags(OPTUNA)]) == 1
    assert capsys.readouterr().err == REFUSAL
    assert main(["sweep", CONFIG, "--count", *flags(OPTUNA)]) == 0
    assert capsys.readouterr().out == "2\n"
    assert not Path("runs/opt").exists()
    assert main(["sweep", CONFIG, "--record", "runs/opt", "--no-progress", *flags(OPTUNA)]) == 0
    out = capsys.readouterr().out
    assert out.endswith("2/2 points finished under runs/opt; summarize with: kalfa collect runs/opt\n")
    points = [point_of("runs/opt", index) for index in range(2)]
    assert all(1e-4 <= point["point"]["lr"] <= 1e-2 and point["point"]["width"] in (16, 32) for point in points)
    assert all(point["strategy"] == "/strategy/kalfa/optuna" and point["total"] == 2 for point in points)
    assert [point["id"] for point in points] == [0, 1]
    manifest = json.loads(Path("runs/opt/manifest.json").read_text())
    assert manifest["strategy"] == "/strategy/kalfa/optuna" and manifest["total"] == 2
    assert manifest["space"]["lr"] == "Range(low=0.0001, high=0.01, log=True, integer=False, steps=None)"
    assert main(["collect", "runs/opt"]) == 0
    assert "  points     2 of 2 finished\n" in capsys.readouterr().out


def test_a_stopped_root_starts_no_point(housing, capsys):
    assert main(["sweep", CONFIG, "--plan", "--record", "runs/halted"]) == 0
    Record("runs/halted").request_stop("test")
    capsys.readouterr()
    assert main(["sweep", CONFIG, "--record", "runs/halted", "--no-progress"]) == 1
    out = capsys.readouterr().out
    assert ("runs/halted: stop requested (stop.json in the root), 0 of 4 points started; remove the file to go on\n"
            in out)
    assert out.endswith("0/4 points finished under runs/halted; summarize with: kalfa collect runs/halted\n")
    assert not [path for path in Path("runs/halted").iterdir() if path.is_dir()]
    assert main(["sweep", CONFIG, "--id", "0", "--record", "runs/halted"]) == 1
    assert capsys.readouterr().err == ("runs/halted: the sweep is stopped (stop.json in the root); remove the file to "
                                       "run more points\n")
