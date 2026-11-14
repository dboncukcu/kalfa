import json
import math
from datetime import date, datetime
from pathlib import Path

import numpy
import pandas
import pytest
import torch
from records.fixtures import make_record, make_sweep, write_json, write_lines

from kalfa.board.answers import TEXT_LIMIT, Board, EscapeError, jsonable
from kalfa.core.errors import KalfaError
from kalfa.records import reader


TEXT = "text/plain; charset=utf-8"
RECORD_KEYS = {"path", "summary", "plan", "progress", "best", "model_summary", "events", "files", "config", "fields",
               "has_predictions", "plots", "tables", "data_report", "failure", "marks"}
SWEEP_KEYS = {"root", "sweep", "points", "ranked", "best", "params", "objectives", "ids", "order", "space"}


def now_text():
    return datetime.now().isoformat(timespec="seconds")


def build(home):
    make_record(home / "reg")
    make_record(home / "cls", status="running", updated=now_text(), classes=["a", "b"], seed=1)
    make_sweep(home / "sweep", count=3)
    make_record(home / "group" / "inner", started="2026-11-06T09:00:00", predictions=False)
    samples = home / "reg" / "samples"
    samples.mkdir()
    (samples / "turn_1.png").write_bytes(b"png")
    (samples / "turn_1.pt").write_bytes(b"pt")
    (home / "reg" / "plots" / "deep").mkdir()
    (home / "reg" / "plots" / "deep" / "grid.svg").write_text("<svg/>", encoding="utf-8")
    (home / "reg" / "plots" / "notes.txt").write_text("notes", encoding="utf-8")
    (home / "loose").mkdir()
    return home


@pytest.fixture(scope="module")
def shared(tmp_path_factory):
    return build(tmp_path_factory.mktemp("answers"))


@pytest.fixture
def fresh(tmp_path):
    return build(tmp_path / "root")


def changed(before, after):
    return sorted(key for key in set(before) | set(after) if before.get(key) != after.get(key))


def test_board_needs_a_directory(tmp_path):
    with pytest.raises(KalfaError):
        Board(tmp_path / "missing")


def test_records_lists_records_sweeps_and_nested_records(shared):
    found = Board(shared).records()
    assert found["root"] == str(shared.resolve())
    assert [entry["path"] for entry in found["records"]] == ["cls", "reg", "sweep", "group/inner"]
    by_path = {entry["path"]: entry for entry in found["records"]}
    assert by_path["reg"]["kind"] == "record" and by_path["reg"]["summary"]["status"] == "done"
    assert by_path["cls"]["summary"]["status"] == "running"
    assert by_path["group/inner"]["kind"] == "record" and by_path["group/inner"]["summary"]["name"] == "inner"
    sweep = by_path["sweep"]
    assert sweep["kind"] == "sweep" and "summary" not in sweep
    assert sweep["sweep"]["points"] == 3 and sweep["sweep"]["objective"]["watch"] == "val/rmse"
    counts = {key: sweep["sweep"][key] for key in ("count", "done", "running", "failed", "stale")}
    assert counts == {"count": 3, "done": 2, "running": 0, "failed": 1, "stale": 0}
    json.dumps(jsonable(found), allow_nan=False)


def test_record_carries_every_part(shared):
    record = Board(shared).record("reg")
    assert set(record) == RECORD_KEYS
    assert record["path"] == "reg" and record["summary"]["name"] == "reg"
    assert record["plan"]["format"] == 1 and record["progress"]["turn"] == 3 and record["progress"]["total"] == 10
    assert record["best"]["series"] == "val/rmse" and record["best"]["mode"] == "min"
    assert record["best"] == record["summary"]["best"]
    assert record["model_summary"]["net"]["parameters"] == 97
    assert [event["kind"] for event in record["events"]] == ["run_start", "patch", "checkpoint", "run_end"]
    assert any(entry["path"] == "history.jsonl" for entry in record["files"])
    assert record["config"].startswith("seed: 7") and "train:" in record["config"]
    assert record["fields"] is None and record["has_predictions"] is True
    assert record["plots"] == ["plots/deep/grid.svg", "plots/loss_curve.png", "samples/turn_1.png"]
    assert record["tables"] == ["predictions.parquet"]
    json.dumps(jsonable(record), allow_nan=False)


def test_record_of_a_nested_record_and_a_sweep_point(fresh):
    inner = fresh / "group" / "inner"
    write_json(inner / "fitted" / "fields.json", {"x": {"columns": ["x0"]}})
    write_lines(inner / "events.jsonl", [{"at": "now", "turn": turn, "kind": "turn"} for turn in range(60)])
    board = Board(fresh)
    found = board.record("group/inner/")
    assert found["path"] == "group/inner" and found["has_predictions"] is False and found["tables"] == []
    assert found["fields"] == {"x": {"columns": ["x0"]}}
    assert len(found["events"]) == 50 and found["events"][0]["turn"] == 10 and found["events"][-1]["turn"] == 59
    point = board.record("sweep/0001")
    assert point["path"] == "sweep/0001" and point["summary"]["params"] == {"lr": 0.01, "width": 64}


def test_a_record_or_a_sweep_as_the_root_answers_with_a_dot(shared, tmp_path):
    make_record(tmp_path / "solo")
    board = Board(tmp_path / "solo")
    assert [entry["path"] for entry in board.records()["records"]] == ["."]
    assert board.record(".")["path"] == "." and board.record("")["summary"]["name"] == "solo"
    assert set(board.stamps()) == {"."}
    swept = Board(shared / "sweep")
    assert set(swept.stamps()) == {".", "0000", "0001", "0002"}
    assert swept.sweep(".")["root"] == "." and swept.record("0002")["path"] == "0002"


@pytest.mark.parametrize("relative", ["..", "../", "reg/../..", "reg/../cls", "/etc", "/", "reg//plots", "reg\\..",
                                      "a\x00b"])
def test_resolve_refuses_escapes(shared, relative):
    with pytest.raises(EscapeError):
        Board(shared).resolve(relative)


def test_resolve_accepts_the_root_and_nested_paths(shared):
    board = Board(shared)
    root = shared.resolve()
    assert board.resolve("") == root and board.resolve(".") == root and board.resolve(None) == root
    assert board.resolve("group/inner") == root / "group" / "inner" == board.resolve("./group/inner/")
    assert board.resolve(Path("group") / "inner") == root / "group" / "inner"


def test_symlinks_out_of_the_root_are_refused(tmp_path):
    root = build(tmp_path / "root")
    outside = make_record(tmp_path / "outside")
    (root / "link").symlink_to(outside, target_is_directory=True)
    (root / "reg" / "secret.yaml").symlink_to(outside / "resolved.yaml")
    board = Board(root)
    with pytest.raises(EscapeError):
        board.resolve("link")
    with pytest.raises(EscapeError):
        board.file("reg", "secret.yaml")
    assert "link" not in {entry["path"] for entry in board.records()["records"]}


def test_unknown_records_and_wrong_kinds_raise(shared):
    board = Board(shared)
    calls = [lambda: board.record("nowhere"), lambda: board.series("nowhere"), lambda: board.files("nowhere"),
             lambda: board.record("loose"), lambda: board.record("sweep"), lambda: board.sweep("reg"),
             lambda: board.compare(["reg", "nowhere"]), lambda: board.compare([]), lambda: board.stop("nowhere"),
             lambda: board.stop("loose"), lambda: board.file("reg", "missing.txt"), lambda: board.file("reg", "plots"),
             lambda: board.file("reg", None), lambda: board.file("nowhere", "plan.json"),
             lambda: board.predictions("nowhere")]
    for call in calls:
        with pytest.raises(KalfaError):
            call()


@pytest.mark.parametrize("name", ["../../etc/passwd", "../cls/resolved.yaml", "/etc/passwd",
                                  "plots/../../cls/plan.json", "plots//loss_curve.png"])
def test_file_refuses_names_outside_the_record(shared, name):
    with pytest.raises(EscapeError):
        Board(shared).file("reg", name)


def test_series_updates_events_and_files_follow_the_readers(shared):
    board = Board(shared)
    directory = shared / "reg"
    assert board.series("reg") == reader.series_of(directory)
    assert board.series("reg", keys=[]) == reader.series_of(directory)
    series = board.series("reg", keys=["val/rmse"], limit=2)
    assert set(series) == {"turn", "val/rmse"} and series["turn"] == [1, 3] and len(series["val/rmse"]) == 2
    updates = board.updates("reg", keys=["loss/main"], limit="5")
    assert updates == reader.updates_of(directory, ["loss/main"], 5) and len(updates["update"]) == 5
    assert board.updates("reg")["grad_norm/main"] == [0.5] * 12
    assert [event["kind"] for event in board.events("reg", kinds=["checkpoint"])] == ["checkpoint"]
    assert [event["kind"] for event in board.events("reg", limit=2)] == ["checkpoint", "run_end"]
    assert board.events("reg") == reader.events_of(directory)
    assert board.files("reg") == reader.files_of(directory)
    with pytest.raises(KalfaError):
        board.series("reg", limit="lots")


def test_file_content_types(fresh):
    record = fresh / "reg"
    for name in ("a.jpg", "a.JPEG", "a.gif"):
        (record / "samples" / name).write_bytes(b"image")
    (record / "notes.md").write_text("# notes\n", encoding="utf-8")
    (record / "stdout.txt").write_text("hello\n", encoding="utf-8")
    expected = {"resolved.yaml": TEXT, "history.jsonl": TEXT, "notes.md": TEXT, "stdout.txt": TEXT,
                "plan.json": "application/json", "plots/loss_curve.png": "image/png",
                "plots/deep/grid.svg": "image/svg+xml", "samples/a.jpg": "image/jpeg", "samples/a.JPEG": "image/jpeg",
                "samples/a.gif": "image/gif", "checkpoints/last.pt": "application/octet-stream",
                "predictions.parquet": "application/octet-stream"}
    board = Board(fresh)
    for name, kind in expected.items():
        data, found = board.file("reg", name)
        assert found == kind, name
        assert data == (record / name).read_bytes(), name


def test_file_tail_keeps_the_last_lines(fresh):
    board = Board(fresh)
    history = fresh / "reg" / "history.jsonl"
    lines = history.read_bytes().splitlines(keepends=True)
    assert board.file("reg", "history.jsonl", tail=1) == (lines[-1], TEXT)
    assert board.file("reg", "history.jsonl", tail="2")[0] == b"".join(lines[-2:])
    assert board.file("reg", "history.jsonl", tail=100)[0] == history.read_bytes()
    assert board.file("reg", "history.jsonl", tail=0)[0] == b""
    data, kind = board.file("reg", "plan.json", tail=3)
    assert kind == TEXT and data.count(b"\n") <= 3
    rows = [f"line {number}\n".encode() for number in range(30_000)]
    (fresh / "reg" / "long.log").write_bytes(b"".join(rows) + b"last without newline")
    assert board.file("reg", "long.log", tail=2)[0] == b"line 29999\nlast without newline"
    assert board.file("reg", "long.log", tail=10_000)[0] == b"".join(rows[-9_999:]) + b"last without newline"
    image, kind = board.file("reg", "plots/loss_curve.png", tail=1)
    assert image == b"png" and kind == "image/png"


def test_large_text_keeps_its_tail_after_a_skip_line(fresh):
    body = b"".join(f"row {number:07d}\n".encode() for number in range(40_000))
    (fresh / "reg" / "big.log").write_bytes(body)
    board = Board(fresh)
    data, kind = board.file("reg", "big.log")
    head, rest = data.split(b"\n", 1)
    skipped = int(head.split()[1])
    assert kind == TEXT and head == f"... {skipped} bytes skipped".encode()
    assert skipped + len(rest) == len(body) and body.endswith(rest) and len(rest) <= TEXT_LIMIT
    assert rest.startswith(b"row ") and rest.endswith(b"row 0039999\n")
    tailed, _ = board.file("reg", "big.log", tail=30_000)
    head, rest = tailed.split(b"\n", 1)
    assert head.startswith(b"... ") and head.endswith(b" bytes skipped") and len(rest) <= TEXT_LIMIT
    assert rest.startswith(b"row ") and rest.endswith(b"row 0039999\n")
    write_json(fresh / "reg" / "large.json", {"values": list(range(60_000))})
    data, kind = board.file("reg", "large.json")
    assert kind == TEXT and data.startswith(b"... ")


def test_predictions_regression_and_classification(shared):
    board = Board(shared)
    regression = board.predictions("reg", sample=10, bins=5)
    pair = regression["pairs"][0]
    assert regression["rows"] == 50 and pair["kind"] == "regression" and pair["target"] == "price"
    assert len(pair["stats"]["scatter"]["row"]) == 10 and len(pair["stats"]["pred_hist"]["counts"]) == 5
    classification = board.predictions("cls")
    assert classification["pairs"][0]["kind"] == "classification"
    assert classification["pairs"][0]["stats"]["classes"] == ["a", "b"]
    assert board.predictions("reg", where="town == 'ank'")["rows"] == 25
    assert board.predictions("reg", where="", sample=None, bins="")["rows"] == 50
    assert board.predictions("group/inner") is None
    assert board.predictions("reg", name="predictions_test.parquet") is None
    with pytest.raises(EscapeError):
        board.predictions("reg", name="../cls/predictions.parquet")
    with pytest.raises(KalfaError):
        board.predictions("reg", where="no such ===")
    with pytest.raises(KalfaError):
        board.predictions("reg", sample="many")
    json.dumps(jsonable(regression), allow_nan=False)
    json.dumps(jsonable(classification), allow_nan=False)


@pytest.mark.parametrize("where", ["town == 'ank' and row < 40", "town in ['ank', 'zzz']", "not (row > 10)",
                                   "(row % 2 == 0) | (row >= 45)", "-row < -5", "town != 'ank'"])
def test_predictions_accept_plain_comparisons(shared, where):
    assert board_rows(shared, where) <= 50


@pytest.mark.parametrize("where", ["town.str.len() > 2", "row.sum() > 0", "len(town) > 2", "row[0] > 1",
                                   "(lambda: 1)() == 1", "[x for x in row]", "row if row else 1", "`row` > 1",
                                   "row\x00 > 1", "town.upper == 'A'"])
def test_predictions_refuse_anything_but_plain_comparisons(shared, where):
    with pytest.raises(KalfaError, match="bad where"):
        Board(shared).predictions("reg", where=where)


def board_rows(root, where):
    return Board(root).predictions("reg", where=where)["rows"]


def test_compare_adds_the_relative_names(shared):
    board = Board(shared)
    compared = board.compare(["reg", "group/inner"], keys=["val/rmse"])
    assert compared["names"] == ["reg", "group/inner"]
    assert [record["name"] for record in compared["records"]] == ["reg", "inner"]
    assert compared["series"]["turn"] == [1, 2, 3] and set(compared["params"]) == {"shared", "differing"}
    assert set(compared["series"][compared["records"][0]["path"]]) == {"val/rmse"}
    alone = board.compare("reg")
    assert alone["names"] == ["reg"] and "val/rmse" in alone["series"][alone["records"][0]["path"]]
    json.dumps(jsonable(compared), allow_nan=False)


def test_sweep_lines_up_params_objectives_and_ranks(shared):
    swept = Board(shared).sweep("sweep")
    assert set(swept) == SWEEP_KEYS
    assert swept["root"] == "sweep" and swept["sweep"]["points"] == 3 and swept["ids"] == [0, 1, 2]
    assert swept["params"] == {"lr": [0.01, 0.01, 0.001], "width": [32, 64, 32]}
    assert [point["path"] for point in swept["points"]] == ["0000", "0001", "0002"]
    objectives = swept["objectives"]
    assert objectives[2] is None and all(isinstance(value, float) for value in objectives[:2])
    assert sorted(swept["ranked"]) == [0, 1, 2] and swept["ranked"][-1] == 2
    assert swept["best"]["id"] == swept["ranked"][0] and swept["best"]["objective"] == min(objectives[:2])
    json.dumps(jsonable(swept), allow_nan=False)


def test_stop_writes_stop_files_for_records_and_running_points(fresh):
    board = Board(fresh)
    assert board.stop("cls") == {"paths": ["cls/stop.json"]}
    assert json.loads((fresh / "cls" / "stop.json").read_text(encoding="utf-8"))["by"] == "board"
    assert board.stop("reg") == {"paths": []}
    assert not (fresh / "reg" / "stop.json").exists()
    make_record(fresh / "sweep" / "0003", status="running", updated=now_text(),
                sweep={"root": str(fresh / "sweep"), "id": 3, "values": {"lr": 0.001, "width": 64}})
    assert board.stop("sweep") == {"paths": ["sweep/0003/stop.json"]}


def test_report_writes_inside_the_root(fresh):
    board = Board(fresh)
    written = board.report(["reg", "cls"])
    assert written == str(fresh.resolve() / "reg" / "report.html") and Path(written).is_file()
    elsewhere = board.report("sweep", out="reports/sweep.html")
    assert Path(elsewhere) == fresh.resolve() / "reports" / "sweep.html" and Path(elsewhere).is_file()
    with pytest.raises(EscapeError):
        board.report(["reg"], out="../outside.html")
    with pytest.raises(KalfaError):
        board.report([])
    with pytest.raises(KalfaError):
        board.report(["nowhere"])


def test_stamps_change_when_a_record_grows(fresh):
    board = Board(fresh)
    before = board.stamps()
    assert set(before) == {"reg", "cls", "sweep", "sweep/0000", "sweep/0001", "sweep/0002", "group/inner"}
    assert all(isinstance(stamp, str) for stamp in before.values())
    assert board.stamps() == before
    with (fresh / "reg" / "history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"turn": 4, "val/rmse": 0.1}) + "\n")
    grown = board.stamps()
    assert changed(before, grown) == ["reg"]
    with (fresh / "sweep" / "0001" / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"kind": "turn"}) + "\n")
    point = board.stamps()
    assert changed(grown, point) == ["sweep", "sweep/0001"]
    manifest = json.loads((fresh / "group" / "inner" / "manifest.json").read_text(encoding="utf-8"))
    write_json(fresh / "group" / "inner" / "manifest.json", {**manifest, "updated": "2026-11-07T11:00:00"})
    later = board.stamps()
    assert changed(point, later) == ["group/inner"]
    make_record(fresh / "sweep" / "0003", status="running", updated=now_text())
    make_record(fresh / "extra")
    last = board.stamps()
    assert changed(later, last) == ["extra", "sweep", "sweep/0003"]


def test_jsonable_turns_values_into_plain_json():
    value = {
        "float32": numpy.float32(0.5), "int64": numpy.int64(7), "flag": numpy.bool_(True), "nan": float("nan"),
        "inf": numpy.float64("inf"), "array": numpy.array([[1.0, numpy.nan], [numpy.inf, 2.0]]),
        "tensor": torch.tensor([1.5, float("nan")]), "scalar": torch.tensor(3), "path": Path("a") / "b",
        "moment": datetime(2026, 11, 8, 10, 30), "day": date(2026, 11, 8), "pair": (1, (2, 3)), "set": {3, 1, 2},
        1: "int key", "nested": [{"deep": numpy.float64(-math.inf)}], "text": numpy.str_("x"), "none": None,
        "bytes": b"ok", "ints": numpy.arange(3, dtype=numpy.int32),
    }
    found = jsonable(value)
    assert found == {
        "float32": 0.5, "int64": 7, "flag": True, "nan": None, "inf": None, "array": [[1.0, None], [None, 2.0]],
        "tensor": [1.5, None], "scalar": 3, "path": "a/b", "moment": "2026-11-08T10:30:00", "day": "2026-11-08",
        "pair": [1, [2, 3]], "set": [1, 2, 3], "1": "int key", "nested": [{"deep": None}], "text": "x",
        "none": None, "bytes": "ok", "ints": [0, 1, 2],
    }
    assert type(found["int64"]) is int and type(found["flag"]) is bool and type(found["float32"]) is float
    assert type(found["text"]) is str and type(found["ints"][0]) is int
    assert json.loads(json.dumps(found, allow_nan=False)) == found
    assert jsonable(True) is True and jsonable(numpy.float64(2.0)) == 2.0 and jsonable(numpy.nan) is None


def test_predictions_clamp_the_bins_a_viewer_asks_for(shared):
    board = Board(shared)
    pair = board.predictions("reg", bins=2_000_000)["pairs"][0]
    assert len(pair["stats"]["pred_hist"]["counts"]) == 1000
    assert len(board.predictions("reg", bins="1000")["pairs"][0]["stats"]["pred_hist"]["counts"]) == 1000
    with pytest.raises(KalfaError):
        board.predictions("reg", bins=0)


def test_predictions_take_bins_per_histogram_for_one_pair(shared):
    board = Board(shared)
    found = board.predictions("reg", pair="pred_net", histograms={"residual_hist": "7", "truth_hist": 5000})
    stats = found["pairs"][0]["stats"]
    assert len(stats["residual_hist"]["counts"]) == 7 and len(stats["truth_hist"]["counts"]) == 1000
    both = board.predictions("reg", pair="pred_net", bins=9, histograms={"residual_hist": 4})["pairs"][0]["stats"]
    assert len(both["residual_hist"]["counts"]) == 4 and len(both["pred_hist"]["counts"]) == 9
    assert board.predictions("reg", histograms={})["pairs"][0]["stats"]["residual_hist"]["counts"]
    with pytest.raises(KalfaError, match="histograms needs pair"):
        board.predictions("reg", histograms={"residual_hist": 7})
    with pytest.raises(KalfaError, match="no histogram score_hist"):
        board.predictions("reg", pair="pred_net", histograms={"score_hist": 7})
    with pytest.raises(KalfaError, match="at least 1"):
        board.predictions("reg", pair="pred_net", histograms={"residual_hist": 0})
    with pytest.raises(KalfaError, match="histograms must map"):
        board.predictions("reg", pair="pred_net", histograms=["residual_hist"])


def picture_board(home, path, manifest):
    record = home / "runs" / "rec"
    record.mkdir(parents=True)
    pandas.DataFrame({"row": [0], "label": ["cat"], "pred_net": ["cat"], "path": [path]}).to_parquet(
        record / "predictions.parquet", index=False)
    write_json(record / "manifest.json", manifest)
    return Board(home / "runs")


def test_a_relative_image_path_is_found_where_the_run_started_or_next_to_its_config(tmp_path):
    from PIL import Image

    project = tmp_path / "project"
    (project / "data" / "cat").mkdir(parents=True)
    Image.new("RGB", (4, 4), (200, 0, 0)).save(project / "data" / "cat" / "0.png")
    started = picture_board(tmp_path / "a", "data/cat/0.png", {"host": {"cwd": str(project)}, "config_files": []})
    body, kind = started.image("rec", 0)
    assert kind == "image/png" and body.startswith(b"\x89PNG")
    configured = picture_board(tmp_path / "b", "data/cat/0.png",
                               {"host": {"cwd": str(tmp_path)}, "config_files": [str(project / "config.yaml")]})
    assert configured.image("rec", 0) == (body, "image/png")
    lost = picture_board(tmp_path / "c", "data/cat/0.png", {"host": {"cwd": str(tmp_path)}})
    with pytest.raises(KalfaError, match="no image file"):
        lost.image("rec", 0)
