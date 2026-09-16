"""The board: a reader of records under a root, served as JSON and one page."""

import json
import threading
import urllib.request
from pathlib import Path

import pandas

import kalfa.board
from helpers import housing_frame
from kalfa.board import Board, serve, static_path
from kalfa.record import Record
from kalfa.std.lego.kalfa.prep import fit
from kalfa.std.pre.sklearn.scalers import StandardScaler


def records(root):
    run = Record(root / "runs" / "one")
    run.manifest("run", name="one", params={"lr": 0.1})
    run.write_text("resolved.yaml",
                   "seed: 7\ntraining: {checkpoint: {uri: best, params: {monitor: val/rmse, mode: min}}}\n")
    pandas.DataFrame({"row": [0, 1, 2, 3], "price": [1.0, 2.0, 3.0, 4.0], "raw_y": [1.1, 2.2, 2.7, 4.4],
                      "pred_y": [1.1, 2.2, 2.7, 4.4], "flag_y": [False, False, False, True],
                      "site": ["a", "a", "b", "b"]}).to_parquet(
        run.directory / "predictions.parquet", index=False)
    (run.directory / "checkpoints").mkdir()
    (run.directory / "checkpoints" / "best.pt").write_bytes(b"pt")
    Record(run.directory / "fitted" / "calibrate").write_json("calibrate.json", {"cut": {"threshold": 0.5}})
    fit(housing_frame(rows=12, columns=3), {"x*": {"preprocessors": ["s"]}, "price": {"target": True}},
        {"s": StandardScaler()}, [], record=str(run.directory))
    run.append("history.jsonl", {"turn": 1, "global_step": 3, "train/l": 1.0, "val/rmse": 2.0, "lr/m": 0.1,
                                 "rules": []})
    run.append("history.jsonl", {"turn": 2, "global_step": 6, "train/l": 0.5, "val/rmse": 1.5, "lr/m": 0.1,
                                 "rules": ["a"]})
    run.append("steps.jsonl", {"step": 1, "turn": 1, "loss/m": 1.0, "lr/m": 0.1})
    (run.directory / "plots").mkdir()
    (run.directory / "plots" / "loss_curve.png").write_bytes(b"png")
    run.write_text("stdout.txt", "line one\nline two\nline three\n")
    run.write_json("architecture.json", {"models": {"m": {"boxes": [], "arrows": []}}})
    sweep = Record(root / "sweeps" / "grid")
    sweep.manifest("sweep", strategy="/strategy/kalfa/grid",
                   objective={"monitor": "val/rmse", "mode": "min", "at": "best"}, total=2)
    for index, value in enumerate((3.0, 1.0)):
        point = Record(sweep.directory / f"{index:04d}")
        point.manifest("point", name=f"{index:04d}", id=index, values={"lr": value}, root=str(sweep.directory),
                       turn="epoch")
        point.write_text("resolved.yaml", f"lr: {value}\n")
        point.append("history.jsonl", {"turn": 1, "global_step": 1, "val/rmse": value, "rules": []})
    Record(sweep.directory / "0001").write_json("sweep.json", {"id": 1, "point": {"lr": 1.0},
                                                               "objective": {"value": 1.0, "turn": 1}})
    Record(sweep.directory / "0001").write_json("run.json", {"status": "ok"})
    Record(root / "prepared").manifest("data", sets=["train"])
    return root


def test_the_board_reads_the_records_and_their_status(tmp_path):
    board = Board(records(tmp_path))
    tree = board.tree()
    assert sorted(tree["groups"]) == ["runs", "sweeps", "sweeps/grid"]
    assert [entry["kind"] for entry in tree["groups"]["sweeps/grid"]] == ["point", "point"]
    run = board.record("runs/one")
    assert run["status"]["state"] == "running" and run["plots"] == ["loss_curve.png"]
    assert run["resolved"].startswith("seed: 7\n")
    assert board.lines("runs/one", "history.jsonl", 1)["lines"][0]["turn"] == 2
    assert board.lines("runs/one", "steps.jsonl")["offset"] == 1
    sweep = board.sweep("sweeps/grid")
    assert [point["status"]["state"] for point in sweep["points"]] == ["running", "finished"]
    assert sweep["points"][0]["objective"] == {"value": 3.0, "turn": 1} and sweep["best"]["id"] == 1
    assert sweep["unit"] == "epoch" and [point["unit"] for point in sweep["points"]] == ["epoch", "epoch"]
    assert [entry["unit"] for entry in tree["groups"]["runs"]] == ["turn"]
    assert "-lr: 3.0" in board.diff("sweeps/grid/0000", "sweeps/grid/0001")["diff"]
    assert board.record("../outside") is None and board.file("../outside/resolved.yaml") is None
    assert board.file("runs/one/resolved.yaml").name == "resolved.yaml" and board.file("runs/one") is None
    assert board.file("runs/one/plots/loss_curve.png").read_bytes() == b"png" and board.record("nowhere") is None
    assert run["logs"] == ["stdout.txt"] and list(run["architecture"]["models"]) == ["m"]
    assert run["config"]["seed"] == 7
    assert run["best"] == {"monitor": "val/rmse", "mode": "min", "value": 1.5, "turn": 2}
    assert [item["name"] for item in run["checkpoints"]] == ["checkpoints/best.pt"]
    assert run["checkpoints"][0]["size"] == 2
    assert run["calibrations"] == {"cut": {"threshold": 0.5}} and run["predictions"] == ["predictions.parquet"]
    table = board.table()["rows"]
    assert [row["path"] for row in table] == ["runs/one", "sweeps/grid/0000", "sweeps/grid/0001"]
    assert table[0]["params"] == {"lr": 0.1} and table[0]["turns"] == 2 and table[0]["best"]["value"] == 1.5
    assert table[0]["last"] == {"val/rmse": 1.5} and table[1]["best"] is None
    predictions = board.predictions("runs/one")
    pair = predictions["pairs"][0]
    assert predictions["rows"] == 4 and (pair["pred"], pair["target"], pair["points"]) == ("pred_y", "price", 4)
    assert pair["rmse"] > 0 and len(pair["histogram"]["counts"]) == 40 and pair["worst"][0]["row"] == 3
    spread = pair["distribution"]
    assert len(spread["edges"]) == 41 and sum(spread["data"]) == 4 and sum(spread["pred"]) == 4
    assert spread["edges"][0] == 1.0 and spread["edges"][-1] == 4.4
    assert predictions["flags"] == ["flag_y"] and len(predictions["sample"]) == 4
    assert board.predictions("nowhere") is None
    coarse = board.predictions("runs/one", sample="all", bins=5)
    assert len(coarse["pairs"][0]["histogram"]["counts"]) == 5 and len(coarse["sample"]) == 4
    assert len(coarse["pairs"][0]["distribution"]["data"]) == 5
    assert len(board.predictions("runs/one", sample=2)["sample"]) == 2
    assert predictions["carried"] == ["site"] and "site" in predictions["sample"][0]
    kept = board.predictions("runs/one", where="site == 'b'")
    assert kept["rows"] == 2 and kept["total"] == 4 and kept["where"] == "site == 'b'" and kept["error"] is None
    assert kept["pairs"][0]["points"] == 2 and [row["row"] for row in kept["sample"]] == [2, 3]
    assert kept["pairs"][0]["rmse"] != predictions["pairs"][0]["rmse"]
    broken = board.predictions("runs/one", where="ghost > 1")
    assert broken["rows"] == 0 and broken["total"] == 4 and "ghost" in broken["error"]
    assert broken["pairs"] == []
    files = board.files("runs/one")
    assert "history.jsonl" in [item["name"] for item in files["files"]] and files["total"] > 0
    assert board.text("runs/one", "resolved.yaml")["text"].startswith("seed: 7")
    assert board.text("runs/one", "checkpoints/best.pt") is None
    assert board.text("runs/one", "../../resolved.yaml") is None
    prep = board.prep("runs/one")
    assert [item["name"] for item in prep["fields"]] == ["x0", "x1", "x2", "price"]
    assert prep["fields"][0]["columns"] == ["x0"] and prep["fields"][3]["target"] is True
    assert prep["preprocessors"]["s"]["grouped"] is True
    assert prep["preprocessors"]["s"]["columns"] == ["x0", "x1", "x2"]
    assert len(prep["preprocessors"]["s"]["state"]["scaler"]["mean_"]) == 3
    assert board.events("runs/one") == {"events": [], "total": 0} and board.prep("sweeps/grid/0000") is None
    assert board.tail("runs/one", "stdout.txt", 2) == {"lines": ["line two", "line three"], "name": "stdout.txt",
                                                        "total": 3}
    live = board.live()
    assert [entry["path"] for entry in live["live"]] == ["runs/one", "sweeps/grid", "sweeps/grid/0000"]
    one = live["live"][0]
    assert one["turn"] == 2 and one["step"] == 1 and one["step_in_turn"] == 1 - 6 and one["last"] == {"val/rmse": 1.5}
    grid = live["live"][1]
    assert grid["finished"] == 1 and grid["running"] == 1 and grid["total"] == 2 and grid["best"]["id"] == 1
    assert [entry["path"] for entry in live["recent"]] == ["sweeps/grid/0001"]
    assert board.tail("runs/one", "stderr.txt") == {"lines": [], "name": "stderr.txt"}
    assert board.tail("runs/one", "resolved.yaml") is None
    assert static_path("index.html").name == "index.html" and static_path("../__init__.py") is None
    assert static_path("board.css") is not None and static_path("nope.js") is None
    before = board.watched("runs/one")
    assert before["history.jsonl"][0] > 0 and before["plots"][0] == 1 and "tree" in before
    Record(tmp_path / "runs" / "one").append("history.jsonl",
                                             {"turn": 3, "global_step": 9, "val/rmse": 1.2, "rules": []})
    after = board.watched("runs/one")
    assert [name for name in after if after[name] != before[name]] == ["history.jsonl"]
    assert "0000/history.jsonl" in board.watched("sweeps/grid") and "runs/one/steps.jsonl" in board.watched("")
    assert "sweeps/grid/0000/history.jsonl" in board.watched("runs/one")
    assert "runs/one/history.jsonl" not in board.watched("runs/one")
    assert board.stop("nowhere") is None
    assert board.stop("sweeps/grid") == {"stopped": ["sweeps/grid", "sweeps/grid/0000"]}
    assert board.record("sweeps/grid/0000")["status"]["stop"]["by"] == "board"
    assert "stop.json" in board.watched("runs/one")


def test_the_server_answers_the_page_and_the_endpoints(tmp_path):
    server = serve(records(tmp_path), port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        page = urllib.request.urlopen(f"{base}/").read().decode()
        assert "kalfa board" in page and "/static/board.js" in page and "cdn.jsdelivr.net/npm/plotly" in page
        script = urllib.request.urlopen(f"{base}/static/board.js")
        assert script.headers["Content-Type"].startswith("text/javascript") and "/api/tree" in script.read().decode()
        tail = json.loads(urllib.request.urlopen(f"{base}/api/tail?path=runs/one&name=stdout.txt&lines=1").read())
        assert tail["lines"] == ["line three"]
        tree = json.loads(urllib.request.urlopen(f"{base}/api/tree").read())
        assert "runs" in tree["groups"]
        live = json.loads(urllib.request.urlopen(f"{base}/api/live").read())
        assert live["live"][0]["path"] == "runs/one"
        watch = urllib.request.urlopen(f"{base}/api/watch?path=runs/one", timeout=10)
        assert watch.headers["Content-Type"].startswith("text/event-stream")
        assert watch.readline().decode().startswith(": watching")
        Record(tmp_path / "runs" / "one").append("steps.jsonl", {"step": 2, "turn": 1, "loss/m": 0.9, "lr/m": 0.1})
        event = ""
        while not event.startswith("data:"):
            event = watch.readline().decode()
        assert json.loads(event[5:])["changed"] == ["steps.jsonl"]
        watch.close()
        history = json.loads(urllib.request.urlopen(f"{base}/api/history?path=runs/one&offset=1").read())
        assert history["offset"] == 2 and len(history["lines"]) == 1
        image = urllib.request.urlopen(f"{base}/file?path=runs/one/plots/loss_curve.png")
        assert image.headers["Content-Type"] == "image/png" and image.read() == b"png"
        try:
            urllib.request.urlopen(f"{base}/api/record?path=nowhere")
            assert False
        except urllib.error.HTTPError as error:
            assert error.code == 404
    finally:
        server.shutdown()
        server.server_close()


def test_the_server_stops_a_record_through_a_post(tmp_path):
    server = serve(records(tmp_path), port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        answer = urllib.request.urlopen(urllib.request.Request(f"{base}/api/stop?path=runs/one", method="POST"))
        assert json.loads(answer.read()) == {"stopped": ["runs/one"]}
        assert Record(tmp_path / "runs" / "one").stop_note()["by"] == "board"
        try:
            urllib.request.urlopen(urllib.request.Request(f"{base}/api/stop?path=sweeps/grid/0001", method="POST"))
            assert False
        except urllib.error.HTTPError as error:
            assert error.code == 409 and "ended already" in error.read().decode()
    finally:
        server.shutdown()
        server.server_close()


def test_static_path_holds_when_the_install_sits_behind_a_symlink(tmp_path, monkeypatch):
    package = Path(kalfa.board.__file__).parent
    link = tmp_path / "linked"
    link.symlink_to(package, target_is_directory=True)
    assert link.resolve() != link
    monkeypatch.setattr(kalfa.board, "__file__", str(link / "__init__.py"))
    assert static_path("index.html").name == "index.html"
    assert static_path("board.css") is not None
    assert static_path("../__init__.py") is None and static_path("../../record.py") is None
    assert static_path("nope.js") is None and static_path("") is None
