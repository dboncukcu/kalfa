import io
import json
import threading
import urllib.error
import urllib.request

import pandas
import pytest

import kalfa.board
from kalfa.board import Board, handler_for, serve, static_path
from kalfa.cli import main
from kalfa.errors import KalfaError
from kalfa.record import Record
from kalfa.std.common.files import read_lines
from kalfa.std.common.history import History


PAIRS = [("pred_y_hat_y_lin", "y_lin"), ("pred_y_hat_y_quad", "y_quad"), ("pred_aux_hat_y_heavy", "y_heavy"),
         ("pred_aux_hat_y_frac", "y_frac"), ("pred_tail_logit_is_hot", "is_hot")]


def fake_records(root):
    run = Record(root / "runs" / "one")
    run.manifest("run", name="one", params={"lr": 0.1}, turn="step")
    run.write_text("resolved.yaml", "seed: 7\ntraining: {epochs: 4, checkpoint: {uri: best, params: {monitor: "
                                    "val/rmse, "
                                    "mode: min}}}\n")
    run.append("history.jsonl", {"turn": 1, "global_step": 3, "train/l": 1.0, "val/rmse": 2.0, "seconds": 1.0,
                                 "rules": []})
    run.append("history.jsonl", {"turn": 2, "global_step": 6, "train/l": 0.5, "val/rmse": 1.5, "seconds": 1.0,
                                 "rules": ["a"]})
    run.append("steps.jsonl", {"step": 7, "turn": 3, "loss/m": 0.9, "lr/m": 0.1})
    run.write_text("stdout.txt", "line one\nline two\nline three\n")
    sweep = Record(root / "sweeps" / "grid")
    sweep.manifest("sweep", strategy="/strategy/kalfa/grid", total=2, space={"lr": "Choices(values=[3.0, 1.0])"},
                   objective={"monitor": "val/rmse", "mode": "min", "at": "best"})
    for index, value in enumerate((3.0, 1.0)):
        point = Record(sweep.directory / f"{index:04d}")
        point.manifest("point", name=f"{index:04d}", id=index, values={"lr": value}, root=str(sweep.directory),
                       turn="epoch")
        point.write_text("resolved.yaml", f"params: {{lr: {value}}}\n")
        point.append("history.jsonl", {"turn": 1, "global_step": 1, "val/rmse": value, "rules": []})
    done = Record(sweep.directory / "0001")
    objective = {"monitor": "val/rmse", "mode": "min", "at": "best", "value": 1.0, "turn": 1}
    done.write_json("sweep.json", {"id": 1, "point": {"lr": 1.0}, "objective": objective})
    done.write_json("run.json", {"status": "ok"})
    Record(root / "prepared").manifest("data", sets=["train"])
    return root


def call(board, path, method="GET"):
    handler_class = handler_for(board)
    handler = handler_class.__new__(handler_class)
    handler.path = path
    handler.command = method
    handler.request_version = "HTTP/1.1"
    handler.requestline = f"{method} {path} HTTP/1.1"
    handler.headers = {}
    handler.client_address = ("127.0.0.1", 0)
    handler.rfile = io.BytesIO()
    handler.wfile = io.BytesIO()
    getattr(handler, f"do_{method}")()
    head, _, body = handler.wfile.getvalue().partition(b"\r\n\r\n")
    lines = head.decode().split("\r\n")
    return int(lines[0].split(" ")[1]), dict(line.split(": ", 1) for line in lines[1:]), body


@pytest.fixture
def board(reference, root):
    return Board(root / "runs")


@pytest.fixture
def record_dir(root):
    return root / "runs" / "ref_fixed"


def test_board_lists_the_records_under_the_root(board, root, record_dir):
    entry = next(entry for entry in board.records() if entry["path"] == "ref_fixed")
    assert entry["kind"] == "run" and entry["name"] == "ref_fixed" and entry["unit"] == "epoch"
    assert entry["status"]["state"] == "finished" and entry["status"]["stop"] is None
    assert entry["started"] == json.loads((record_dir / "manifest.json").read_text())["started"]
    tree = board.tree()
    assert tree["root"] == str((root / "runs").resolve())
    assert "ref_fixed" in [entry["path"] for entry in tree["groups"]["."]]


def test_board_record_reads_every_note_of_the_run(board, record_dir):
    record = board.record("ref_fixed")
    history = History.read(record_dir)
    assert record["path"] == "ref_fixed" and record["status"]["state"] == "finished"
    assert record["manifest"]["kind"] == "run" and record["manifest"]["name"] == "ref_fixed"
    assert record["best"]["monitor"] == "val/rmse_lin" and record["best"]["mode"] == "min"
    assert (record["best"]["value"], record["best"]["turn"]) == history.best("val/rmse_lin", "min")
    assert [item["name"] for item in record["checkpoints"]] == ["checkpoints/best.pt", "checkpoints/last.pt",
                                                                 "final/state.pt"]
    assert all(item["size"] > 0 and item["modified"] for item in record["checkpoints"])
    assert list(record["calibrations"]) == ["hot_cut"] and record["calibrations"]["hot_cut"]["output"] == "tail_logit"
    assert record["predictions"] == ["predictions.parquet"] and record["samples"] == []
    assert record["plots"] == sorted(path.name for path in (record_dir / "plots").iterdir())
    assert len(record["plots"]) == 24 and record["logs"] == ["stdout.txt", "stderr.txt"]
    assert record["device"] == {"device": "cpu", "uri": "/device/kalfa/cpu", "params": {}}
    assert sorted(record["host"]) == ["cwd", "hostname", "pid"] and sorted(record["git"]) == ["commit", "dirty"]
    assert record["resume"] is None and record["sweep"] is None and record["run"]["status"] == "ok"
    assert sorted(record["architecture"]) == ["features", "models", "targets"]
    assert record["resolved"].startswith("params:\n") and record["config"]["seed"] == 11
    assert sorted(record["data"]) == ["after_set_transforms", "fit", "frames", "loaders", "set_transforms", "sets",
                                      "split", "stages"]
    assert len(record["events"]) == 60 and record["events"][-1]["kind"] == "finished"
    assert board.record("nowhere") is None and board.record("../outside") is None


def test_board_progress_live_and_table_read_the_history(board, record_dir):
    history = History.read(record_dir)
    progress = board.progress(record_dir)
    assert progress["turn"] == 3 and progress["turns_total"] == 3 and progress["eta"] is None
    assert progress["monitor"] == "val/rmse_lin" and progress["monitor_value"] == history[-1]["val/rmse_lin"]
    last_step = read_lines(record_dir / "steps.jsonl")[-1]
    assert progress["step"] == last_step["step"]
    assert progress["step_in_turn"] == last_step["step"] - history[-1]["global_step"]
    assert progress["steps_per_turn"] == 19 and sorted(progress["losses"]) == ["loss/aux", "loss/main"]
    assert len(progress["last"]) == 8 and all(key.startswith("val/") for key in progress["last"])
    live = board.live()
    assert "ref_fixed" not in [entry["path"] for entry in live["live"]]
    assert "ref_fixed" in [entry["path"] for entry in live["recent"]]
    row = next(row for row in board.table()["rows"] if row["path"] == "ref_fixed")
    assert row["turns"] == 3 and row["device"] == "cpu" and row["best"] == board.record("ref_fixed")["best"]
    assert row["params"] == json.loads((record_dir / "manifest.json").read_text())["params"]
    assert row["seconds"] == sum(line["seconds"] for line in history)
    assert row["last"] == {key: value for key, value in history[-1].items()
                           if key.startswith(("val/", "test/")) and "/total" not in key}


def test_board_predictions_pair_every_target_with_its_prediction(board, record_dir):
    table = pandas.read_parquet(record_dir / "predictions.parquet")
    found = board.predictions("ref_fixed")
    assert found["file"] == "predictions.parquet" and found["rows"] == found["total"] == len(table) == 278
    assert found["where"] is None and found["error"] is None and found["columns"] == list(table.columns)
    assert [(pair["pred"], pair["target"]) for pair in found["pairs"]] == PAIRS
    pair = found["pairs"][0]
    assert pair["points"] == 278 and pair["rmse"] > 0 and pair["mae"] > 0 and pair["r2"] is not None
    assert len(pair["histogram"]["counts"]) == 40 and len(pair["histogram"]["edges"]) == 41
    assert len(pair["distribution"]["edges"]) == 41 and len(pair["worst"]) == 15
    assert found["flags"] == ["flag_tail_logit"] and found["carried"] == ["sample_id", "site"]
    assert len(found["sample"]) == 278
    assert set(found["sample"][0]) == {"row", *[column for pair in PAIRS for column in pair], "flag_tail_logit",
                                       "sample_id", "site"}
    kept = board.predictions("ref_fixed", where="site == 's1'", sample=10)
    assert kept["rows"] == int((table["site"] == "s1").sum()) and kept["total"] == 278 and len(kept["sample"]) == 10
    assert kept["pairs"][0]["points"] == kept["rows"] and kept["where"] == "site == 's1'"
    broken = board.predictions("ref_fixed", where="ghost > 1")
    assert broken["rows"] == 0 and broken["pairs"] == [] and "ghost" in broken["error"]
    assert board.predictions("ref_fixed", name="resolved.yaml") is None and board.predictions("nowhere") is None


def test_board_serves_the_files_the_text_and_the_fitted_preprocessors(board, record_dir):
    files = board.files("ref_fixed")
    names = [item["name"] for item in files["files"]]
    assert "manifest.json" in names and "checkpoints/best.pt" in names and "fitted/preprocessors/plan.json" in names
    assert files["total"] == sum(item["size"] for item in files["files"]) > 0 and board.files("nowhere") is None
    text = board.text("ref_fixed", "manifest.json")
    assert json.loads(text["text"])["kind"] == "run" and text["truncated"] is False
    assert text["size"] == len(text["text"]) and text["name"] == "manifest.json"
    assert board.text("ref_fixed", "checkpoints/best.pt") is None
    assert board.text("ref_fixed", "../../reference.parquet") is None
    assert board.text("ref_fixed", "resolved.yaml", limit=10)["truncated"] is True
    prep = board.prep("ref_fixed")
    assert [field["name"] for field in prep["fields"]][:5] == ["num_0", "num_1", "num_2", "num_3", "inter"]
    assert sorted(prep["preprocessors"]) == ["abs_train", "fill0", "impute", "onehot", "ordinal", "robust", "squash",
                                             "std", "target_std", "to_float", "to_logit"]
    assert prep["preprocessors"]["std"]["grouped"] is True and prep["preprocessors"]["std"]["class"] == "StandardScaler"
    assert prep["sets"] == {"abs_train": ["train"]} and prep["drop"] == ["noise_id"] and board.prep("nowhere") is None
    assert board.events("ref_fixed")["total"] == len(read_lines(record_dir / "events.jsonl")) > 0
    assert len(board.events("ref_fixed", limit=5)["events"]) == 5
    assert board.lines("ref_fixed", "history.jsonl")["offset"] == 3
    assert [line["turn"] for line in board.lines("ref_fixed", "history.jsonl", 2)["lines"]] == [3]
    assert board.lines("ref_fixed", "steps.jsonl")["offset"] == len(read_lines(record_dir / "steps.jsonl"))
    assert (
        board.tail("ref_fixed", "stdout.txt", 2)["lines"] == (record_dir / "stdout.txt").read_text().splitlines()[-2:])
    assert board.tail("ref_fixed", "resolved.yaml") is None


def test_board_describe_and_best_of_read_the_resolved_config(board):
    text = board.describe("ref_fixed")["text"]
    assert text.startswith("resolved.yaml") and "── DATA" in text and "\x1b[" not in text
    assert board.describe("nowhere") is None
    history = History([{"turn": 1, "val/rmse": 2.0}, {"turn": 2, "val/rmse": 1.0}, {"turn": 3, "val/rmse": 1.5}])
    config = {"training": {"checkpoint": {"uri": "/checkpoint/kalfa/best",
                                          "params": {"monitor": "val/rmse", "mode": "min"}}}}
    assert board.best_of(config, history) == {"monitor": "val/rmse", "mode": "min", "value": 1.0, "turn": 2}
    assert board.best_of({"training": {"checkpoint": "/checkpoint/kalfa/last"}}, history) is None
    assert board.best_of(config, History()) is None
    assert board.best_of({"training": {"checkpoint": {"params": {"monitor": "val/ghost"}}}}, history) is None


def test_board_follows_a_running_record_and_a_sweep(tmp_path):
    board = Board(fake_records(tmp_path))
    tree = board.tree()
    assert sorted(tree["groups"]) == ["runs", "sweeps", "sweeps/grid"]
    assert [entry["kind"] for entry in tree["groups"]["sweeps/grid"]] == ["point", "point"]
    assert [entry["path"] for entry in tree["groups"]["sweeps"]] == ["sweeps/grid"]
    assert tree["groups"]["sweeps"][0]["status"]["state"] == "running"
    assert [entry["unit"] for entry in tree["groups"]["runs"]] == ["turn"]
    run = board.record("runs/one")
    assert run["status"]["state"] == "running"
    assert run["best"] == {"monitor": "val/rmse", "mode": "min", "value": 1.5, "turn": 2}
    assert run["checkpoints"] == [] and run["plots"] == [] and run["logs"] == ["stdout.txt"]
    progress = board.progress(tmp_path / "runs" / "one")
    assert progress["turn"] == 2 and progress["turns_total"] == 4 and progress["eta"] == 2.0
    assert progress["step"] == 7 and progress["step_in_turn"] == 1 and progress["losses"] == {"loss/m": 0.9}
    assert progress["last"] == {"val/rmse": 1.5} and progress["monitor_value"] == 1.5
    assert progress["steps_per_turn"] is None
    live = board.live()
    assert [entry["path"] for entry in live["live"]] == ["runs/one", "sweeps/grid", "sweeps/grid/0000"]
    assert live["live"][0]["turn"] == 2 and live["live"][1]["finished"] == 1 and live["live"][1]["running"] == 1
    assert live["live"][1]["total"] == 2 and live["live"][1]["best"]["id"] == 1
    assert [entry["path"] for entry in live["recent"]] == ["sweeps/grid/0001"]
    sweep = board.sweep("sweeps/grid")
    assert [point["status"]["state"] for point in sweep["points"]] == ["running", "finished"]
    assert sweep["points"][0]["objective"] == {"value": 3.0, "turn": 1} and sweep["best"]["id"] == 1
    assert sweep["space"] == {"lr": {"kind": "choices", "values": [3.0, 1.0]}} and sweep["unit"] == "epoch"
    table = board.table()["rows"]
    assert [row["path"] for row in table] == ["runs/one", "sweeps/grid/0000", "sweeps/grid/0001"]
    assert table[0]["params"] == {"lr": 0.1} and table[0]["turns"] == 2 and table[0]["seconds"] == 2.0
    assert table[0]["last"] == {"val/rmse": 1.5} and table[1]["best"] is None
    diff = board.diff("sweeps/grid/0000", "sweeps/grid/0001")["diff"]
    assert "-params: {lr: 3.0}" in diff and "+params: {lr: 1.0}" in diff
    assert board.diff("runs/one", "nowhere") is None
    assert board.tail("runs/one", "stdout.txt", 2) == {"lines": ["line two", "line three"], "name": "stdout.txt",
                                                        "total": 3}
    assert board.tail("runs/one", "stderr.txt") == {"lines": [], "name": "stderr.txt"}
    assert board.record("../outside") is None and board.resolve("../outside") is None
    assert board.file("runs/one") is None and board.file("../outside/resolved.yaml") is None
    assert board.file("runs/one/resolved.yaml") == (tmp_path / "runs" / "one" / "resolved.yaml").resolve()


def test_board_watches_the_stamps_and_stops_a_running_record(tmp_path):
    board = Board(fake_records(tmp_path))
    before = board.watched("runs/one")
    assert "tree" in before and before["history.jsonl"][0] > 0 and before["plots"] is None
    assert "sweeps/grid/0000/history.jsonl" in before and "runs/one/history.jsonl" not in before
    Record(tmp_path / "runs" / "one").append("history.jsonl", {"turn": 3, "global_step": 9, "val/rmse": 1.2,
                                                               "rules": []})
    after = board.watched("runs/one")
    assert [name for name in after if after[name] != before[name]] == ["history.jsonl"]
    assert "0000/history.jsonl" in board.watched("sweeps/grid") and "runs/one/steps.jsonl" in board.watched("")
    assert board.stop("nowhere") is None
    assert board.stop("sweeps/grid") == {"stopped": ["sweeps/grid", "sweeps/grid/0000"]}
    assert board.record("sweeps/grid/0000")["status"]["stop"]["by"] == "board"
    assert not Record(tmp_path / "sweeps" / "grid" / "0001").stop_requested()
    with pytest.raises(KalfaError, match="has ended already \\(finished\\); there is nothing to stop"):
        board.stop("sweeps/grid/0001")
    assert "stop.json" in board.watched("sweeps/grid")


def test_handler_answers_the_json_endpoints_without_a_server(board, record_dir):
    status, headers, body = call(board, "/api/tree")
    assert status == 200 and headers["Content-Type"] == "application/json" and headers["Cache-Control"] == "no-store"
    assert headers["Content-Length"] == str(len(body))
    assert "ref_fixed" in [entry["path"] for entry in json.loads(body)["groups"]["."]]
    status, headers, body = call(board, "/api/record?path=ref_fixed")
    assert status == 200 and json.loads(body)["manifest"]["kind"] == "run"
    status, headers, body = call(board, "/api/record?path=nowhere")
    assert status == 404
    assert headers["Content-Type"] == "application/json" and json.loads(body) == {"error": "not found"}
    for endpoint in ("live", "table", "predictions?path=ref_fixed&sample=5", "files?path=ref_fixed",
                     "text?path=ref_fixed&name=manifest.json", "prep?path=ref_fixed", "events?path=ref_fixed",
                     "history?path=ref_fixed&offset=2", "steps?path=ref_fixed", "sweep?path=ref_fixed",
                     "tail?path=ref_fixed&name=stderr.txt&lines=1", "diff?a=ref_fixed&b=ref_fixed",
                     "describe?path=ref_fixed"):
        status, headers, body = call(board, f"/api/{endpoint}")
        assert status == 200 and headers["Content-Type"] == "application/json" and isinstance(json.loads(body), dict)
    assert len(json.loads(call(board, "/api/predictions?path=ref_fixed&sample=5")[2])["sample"]) == 5
    history = json.loads(call(board, "/api/history?path=ref_fixed&offset=2")[2])
    assert history["offset"] == 3 and [line["turn"] for line in history["lines"]] == [3]
    tail = json.loads(call(board, "/api/tail?path=ref_fixed&name=stderr.txt&lines=1")[2])
    assert tail["lines"] == (record_dir / "stderr.txt").read_text().splitlines()[-1:]
    assert json.loads(call(board, "/api/diff?a=ref_fixed&b=ref_fixed")[2]) == {"diff": []}
    status, headers, body = call(board, "/file?path=ref_fixed/plots/loss_curve.png")
    assert status == 200 and headers["Content-Type"] == "image/png"
    assert body == (record_dir / "plots" / "loss_curve.png").read_bytes()
    assert call(board, "/file?path=ref_fixed/nothing.png")[0] == 404
    status, headers, body = call(board, "/")
    assert status == 200 and headers["Content-Type"] == "text/html; charset=utf-8" and b"kalfa board" in body
    status, headers, body = call(board, "/static/board.js")
    assert status == 200 and headers["Content-Type"] == "text/javascript; charset=utf-8" and b"/api/tree" in body
    assert call(board, "/static/../__init__.py")[0] == 404 and call(board, "/static/nope.js")[0] == 404
    status, headers, body = call(board, "/nothing")
    assert status == 404 and headers["Content-Type"] == "text/plain" and body == b"not found"


def test_post_stop_answers_409_on_the_finished_record(board, record_dir):
    status, headers, body = call(board, "/api/stop?path=ref_fixed", "POST")
    assert status == 409 and headers["Content-Type"] == "application/json"
    assert json.loads(body) == {"error": f"{record_dir.resolve()} has ended already (finished); there is nothing to "
                                         "stop"}
    assert not (record_dir / "stop.json").exists()
    status, headers, body = call(board, "/api/stop?path=nowhere", "POST")
    assert status == 404 and json.loads(body) == {"error": "not found"}
    status, headers, body = call(board, "/api/tree", "POST")
    assert status == 404 and headers["Content-Type"] == "text/plain" and body == b"not found"


def test_server_serves_the_page_streams_the_changes_and_stops_a_record(tmp_path):
    server = serve(fake_records(tmp_path), port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        page = urllib.request.urlopen(f"{base}/")
        assert page.headers["Content-Type"] == "text/html; charset=utf-8" and "kalfa board" in page.read().decode()
        script = urllib.request.urlopen(f"{base}/static/board.js")
        assert script.headers["Content-Type"] == "text/javascript; charset=utf-8"
        assert "/api/tree" in script.read().decode()
        live = json.loads(urllib.request.urlopen(f"{base}/api/live").read())
        assert live["live"][0]["path"] == "runs/one"
        watch = urllib.request.urlopen(f"{base}/api/watch?path=runs/one", timeout=10)
        assert watch.headers["Content-Type"] == "text/event-stream; charset=utf-8"
        assert watch.readline() == b": watching\n"
        Record(tmp_path / "runs" / "one").append("steps.jsonl", {"step": 8, "turn": 3, "loss/m": 0.8, "lr/m": 0.1})
        event = b""
        while not event.startswith(b"data:"):
            event = watch.readline()
        assert json.loads(event[5:]) == {"changed": ["steps.jsonl"]}
        watch.close()
        answer = urllib.request.urlopen(urllib.request.Request(f"{base}/api/stop?path=runs/one", method="POST"))
        assert json.loads(answer.read()) == {"stopped": ["runs/one"]}
        assert Record(tmp_path / "runs" / "one").stop_note()["by"] == "board"
        with pytest.raises(urllib.error.HTTPError) as refused:
            urllib.request.urlopen(urllib.request.Request(f"{base}/api/stop?path=sweeps/grid/0001", method="POST"))
        assert refused.value.code == 409
        assert "has ended already (finished); there is nothing to stop" in json.loads(refused.value.read())["error"]
        with pytest.raises(urllib.error.HTTPError) as missing:
            urllib.request.urlopen(f"{base}/api/record?path=nowhere")
        assert missing.value.code == 404
    finally:
        server.shutdown()
        server.server_close()


def test_static_path_refuses_what_lies_outside_the_static_folder():
    assert static_path("index.html").name == "index.html" and static_path("index.html").parent.name == "static"
    assert static_path("board.css").name == "board.css" and static_path("board.js").name == "board.js"
    assert static_path("../__init__.py") is None and static_path("../../record.py") is None
    assert static_path("static/../../record.py") is None and static_path("/etc/hosts") is None
    assert static_path("nope.js") is None and static_path("") is None and static_path(None) is None


def test_board_command_serves_the_root_on_the_host_and_port(monkeypatch, capsys, tmp_path):
    calls = []

    class Server:
        server_address = ("0.0.0.0", 9000)

        def serve_forever(self):
            calls.append("served")
            raise KeyboardInterrupt

        def server_close(self):
            calls.append("closed")

    def fake_serve(root, host="127.0.0.1", port=8080):
        calls.append((root, host, port))
        return Server()

    monkeypatch.setattr(kalfa.board, "serve", fake_serve)
    assert main(["board", str(tmp_path), "--host", "0.0.0.0", "--port", "9000"]) == 0
    assert calls == [(str(tmp_path), "0.0.0.0", 9000), "served", "closed"]
    assert capsys.readouterr().out == f"kalfa board over {tmp_path} at http://0.0.0.0:9000/ (ctrl-c stops it)\n"
    Server.server_address = ("127.0.0.1", 8080)
    calls.clear()
    assert main(["board", str(tmp_path)]) == 0
    assert calls == [(str(tmp_path), "127.0.0.1", 8080), "served", "closed"]
    assert capsys.readouterr().out == f"kalfa board over {tmp_path} at http://127.0.0.1:8080/ (ctrl-c stops it)\n"
