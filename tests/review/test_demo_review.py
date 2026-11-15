import json
import runpy
import shutil
import stat
from pathlib import Path

import numpy
import pandas
import pytest

import kalfa
from kalfa.board.answers import Board, jsonable
from kalfa.records import reader


ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "runs" / "demo"
EXAMPLES = ROOT / "examples"
STOPPED = {"13_distillation/student": ("stopped", 2, 3)}

pytestmark = pytest.mark.skipif(not DEMO.is_dir(), reason="runs/demo is missing; build it with examples/demo.py")


def recipes():
    return {folder.name: json.loads((folder / "example.json").read_text(encoding="utf-8"))
            for folder in sorted(EXAMPLES.iterdir()) if (folder / "example.json").is_file()}


def strict(value):
    return json.dumps(jsonable(value), allow_nan=False)


def patches_of(record):
    lines = (DEMO / record / "events.jsonl").read_text(encoding="utf-8").splitlines()
    return [entry["by"] for entry in map(json.loads, filter(None, lines)) if entry["kind"] == "patch"]


@pytest.fixture(scope="module")
def board():
    return Board(DEMO)


@pytest.fixture(scope="module")
def listing(board):
    return board.records()


def from_demo(path):
    return str(path).split("/", 1)[0] in recipes()


def plain_records(listing):
    return [entry["path"] for entry in listing["records"] if entry["kind"] == "record" and from_demo(entry["path"])]


def sweep_roots(listing):
    return [entry["path"] for entry in listing["records"] if entry["kind"] == "sweep" and from_demo(entry["path"])]


def test_demo_holds_every_example_and_every_recipe_output():
    missing = []
    for name, recipe in recipes().items():
        folder = DEMO / name
        if not folder.is_dir():
            missing.append(name)
            continue
        for step in recipe["steps"]:
            for key in ("record", "out", "html"):
                if step.get(key) and not (folder / step[key]).exists():
                    missing.append(f"{name}/{step[key]}")
    assert not missing, missing


def expected_status(path):
    return "stopped" if path in STOPPED else "done"


def test_manifests_say_done_with_the_recipe_turns():
    found = {}
    for manifest in sorted(DEMO.rglob("manifest.json")):
        if not from_demo(manifest.parent.relative_to(DEMO).as_posix()):
            continue
        data = reader.read_manifest(manifest.parent)
        found[manifest.parent.relative_to(DEMO).as_posix()] = (data["status"], data["turn"], data["turns_planned"])
    assert len(found) == 32
    assert {path: found[path] for path in STOPPED} == STOPPED
    assert all(status == "done" and turn == planned for path, (status, turn, planned) in found.items()
               if path not in STOPPED), found


def test_records_lists_every_record_and_sweep_done(listing):
    assert len(plain_records(listing)) == 22
    assert sorted(sweep_roots(listing)) == ["14_sweep/grid", "14_sweep/optuna", "14_sweep/random"]
    for entry in listing["records"]:
        if not from_demo(entry["path"]):
            continue
        if entry["kind"] == "record":
            status = entry["summary"].get("status")
            assert status == expected_status(entry["path"]) and "error" not in entry["summary"], entry["path"]
        else:
            assert entry["sweep"]["count"] == entry["sweep"]["done"] >= 3, entry["path"]
    strict(listing)


def test_every_record_answers_record_series_events_files(board, listing):
    for path in plain_records(listing):
        answer = board.record(path)
        strict(answer)
        assert answer["summary"]["status"] == expected_status(path) and answer["progress"], path
        for value in (board.series(path), board.updates(path), board.events(path), board.files(path)):
            strict(value)
        for plot in answer["plots"]:
            data, kind = board.file(path, plot)
            assert data and kind.startswith("image/"), (path, plot)


def test_every_prediction_table_answers(board, listing):
    seen = 0
    for path in plain_records(listing):
        for table in board.record(path)["tables"]:
            answer = board.predictions(path, name=table)
            assert answer is not None and answer["rows"] > 0, (path, table)
            strict(answer)
            seen += 1
    assert seen >= 20


def test_sweep_roots_rank_their_points(board, listing):
    for root in sweep_roots(listing):
        answer = board.sweep(root)
        strict(answer)
        assert len(answer["ranked"]) == len(answer["points"]) >= 3 and answer["best"] is not None, root
        for point in answer["points"]:
            assert board.record(f"{root}/{point['path']}")["summary"]["status"] == "done"
    grid = board.sweep("14_sweep/grid")
    assert grid["best"]["path"] == "0001" and grid["best"]["values"]["width"] == 32


def test_compare_answers_over_the_fold_records(board):
    answer = board.compare(["11_kfold/fold0", "11_kfold/fold1", "11_kfold/fold2"])
    strict(answer)
    assert answer["names"] == ["11_kfold/fold0", "11_kfold/fold1", "11_kfold/fold2"]


def test_rules_the_recipes_trigger_fired():
    assert patches_of("01_mlp_regression/housing") == ["to_huber", "to_l1", "to_log_cosh"]
    assert patches_of("04_cnn_images/pets") == ["unfreeze"]
    assert patches_of("15_multi_target/plant") == ["rebalance"]
    assert patches_of("16_mdmm/mdmm") == ["tighten"]
    assert patches_of("13_distillation/student") == ["cool_down"]
    lines = (DEMO / "13_distillation/student/events.jsonl").read_text(encoding="utf-8").splitlines()
    ends = [entry for entry in map(json.loads, filter(None, lines)) if entry["kind"] == "run_end"]
    assert len(ends) == 1 and ends[0]["turn"] == 2 and str(ends[0]["stopping"]).startswith("stop: "), ends


def test_vae_recipe_fires_the_reference_swap():
    assert "to_huber" in patches_of("06_vae/vae")


def test_window_predictions_on_new_rows_pair_each_horizon_with_its_own_hour(board):
    frame = pandas.read_parquet(DEMO / "03_timeseries_window/energy/predictions_energy_next.parquet")
    source = pandas.read_parquet(EXAMPLES / "03_timeseries_window/energy_next.parquet")
    answer = board.predictions("03_timeseries_window/energy", name="predictions_energy_next.parquet")
    assert [pair["target"] for pair in answer["pairs"]] == [f"load_{step}" for step in range(6)]
    for pair in answer["pairs"]:
        step = int(pair["pred"].rsplit("_", 1)[1])
        later = source["load"].reindex(frame["row"] + step).to_numpy()
        site = source["site_id"].reindex(frame["row"] + step).to_numpy() == source["site_id"].reindex(
            frame["row"]).to_numpy()
        keep = site & numpy.isfinite(later)
        aligned = float(numpy.sqrt(numpy.mean((frame[pair["pred"]].to_numpy()[keep] - later[keep]) ** 2)))
        assert pair["stats"]["rmse"] == pytest.approx(aligned, rel=0.1), (pair["pred"], pair["stats"]["rmse"], aligned)


def test_window_test_predictions_have_pairs(board):
    pairs = board.predictions("03_timeseries_window/energy")["pairs"]
    assert [(pair["pred"], pair["target"]) for pair in pairs] == [(f"pred_net_{step}", f"load_{step}")
                                                                  for step in range(6)]


def test_simclr_embeddings_are_not_paired_with_the_image_path(board):
    pairs = board.predictions("09_simclr/simclr")["pairs"]
    assert not [pair for pair in pairs if pair["target"] == "path"]


def test_resumed_record_keeps_the_params_of_its_source():
    first = reader.read_manifest(DEMO / "12_resume/resume_first")
    more = reader.read_manifest(DEMO / "12_resume/resume_more")
    assert first["params"] and more["params"] == first["params"]


def test_record_files_follow_the_umask_like_history():
    history = stat.S_IMODE((DEMO / "01_mlp_regression/housing/history.jsonl").stat().st_mode)
    for name in ("01_mlp_regression/housing/manifest.json", "01_mlp_regression/housing/predictions.parquet",
                 "14_sweep/report.html"):
        assert stat.S_IMODE((DEMO / name).stat().st_mode) == history, name


def full_config_turn(name, tmp_path, sets=(("train.epochs", 1),), params=()):
    folder = tmp_path / name
    shutil.copytree(EXAMPLES / name, folder, ignore=shutil.ignore_patterns("data", "__pycache__", "*.parquet"))
    runpy.run_path(str(folder / "make_data.py"), run_name="__main__")
    target = tmp_path / "record"
    kalfa.run(str(folder / "config.yaml"), sets=[("device", "cpu"), *sets], params=list(params), record=str(target))
    return [json.loads(line) for line in (target / "history.jsonl").read_text(encoding="utf-8").splitlines()]


@pytest.mark.slow
def test_ddpm_full_config_trains_on_its_own_data(tmp_path):
    assert full_config_turn("08_ddpm", tmp_path)[0]["update"] > 0


@pytest.mark.slow
def test_simclr_full_config_trains_on_its_own_data(tmp_path):
    assert full_config_turn("09_simclr", tmp_path)[0]["update"] > 0


@pytest.mark.slow
def test_wgan_full_config_steps_the_generator(tmp_path):
    sets = (("train.epochs", 1), ("metrics.fid.every", 1), ("metrics.fid.n", 16))
    assert "train/g" in full_config_turn("07_wgan_gp", tmp_path, sets)[0]


@pytest.mark.slow
def test_wgan_short_first_try_from_the_readme_finishes(tmp_path):
    readme = (EXAMPLES / "README.md").read_text(encoding="utf-8")
    assert "`--set metrics.fid.every=1` on top of `-p epochs=3`" in readme
    history = full_config_turn("07_wgan_gp", tmp_path, (("metrics.fid.every", 1),), (("epochs", 3),))
    assert len(history) == 3 and all("val/fid" in line for line in history)
