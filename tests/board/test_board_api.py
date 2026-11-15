import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

import kalfa
from helpers import housing_frame
from kalfa.board.answers import Board
from kalfa.board.server import make_server
from records.fixtures import make_record, make_sweep


pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    home = tmp_path_factory.mktemp("board")
    housing_frame(160, 0).to_parquet(home / "housing.parquet", index=False)
    config = {
        "seed": 3, "device": "cpu",
        "data": {"use": "table", "file": str(home / "housing.parquet"), "split": [0.7, 0.15, 0.15],
                 "features": {"x*": "standard_scaler"}, "targets": {"price": "standard_scaler"}, "batch": 32},
        "models": {"net": {"use": "mlp", "hidden": [8], "out": 1}},
        "losses": {"mse": "mse"}, "metrics": {"rmse": "rmse"},
        "optimizers": {"main": {"use": "adam", "lr": 0.01, "over": "@models.net", "minimizes": "@losses.mse"}},
        "train": {"epochs": 2, "checkpoint": {"watch": "val/rmse"}, "report": "best"},
        "plots": {"loss_curve": "loss_curve", "pred_vs_true": "pred_vs_true"},
    }
    kalfa.run(config, record=str(home / "runs" / "real"))
    make_record(home / "runs" / "fake", status="running", classes=["a", "b"])
    make_sweep(home / "runs" / "sweep", count=3)
    return home / "runs"


@pytest.fixture(scope="module")
def server(root):
    httpd = make_server(root, host="127.0.0.1", port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def get(server, path, raw=False):
    with urllib.request.urlopen(f"{server}{path}", timeout=10) as response:
        body = response.read()
        kind = response.headers.get("Content-Type", "")
    return (body, kind) if raw else json.loads(body)


def post(server, path, payload=None):
    data = json.dumps(payload or {}).encode("utf-8")
    request = urllib.request.Request(f"{server}{path}", data=data, headers={"Content-Type": "application/json"},
                                     method="POST")
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())


def status_of(server, path):
    try:
        with urllib.request.urlopen(f"{server}{path}", timeout=10) as response:
            return response.status, None
    except urllib.error.HTTPError as failure:
        with failure:
            return failure.code, json.loads(failure.read() or b"{}")


def test_answers_records_and_record(root):
    board = Board(root)
    found = board.records()
    assert found["root"] == str(root.resolve())
    by_path = {entry["path"]: entry for entry in found["records"]}
    assert {"real", "fake", "sweep"} <= set(by_path)
    assert by_path["real"]["kind"] == "record" and by_path["real"]["summary"]["status"] == "done"
    assert by_path["fake"]["summary"]["status"] in ("running", "stale")
    assert by_path["sweep"]["kind"] == "sweep" and by_path["sweep"]["sweep"]["points"] == 3
    record = board.record("real")
    assert record["summary"]["best"]["series"] == "val/rmse" and record["plan"]["format"] == 1
    assert record["progress"]["turn"] == 2 and record["model_summary"]["net"]["parameters"] > 0
    assert any(entry["path"] == "history.jsonl" for entry in record["files"])
    assert "train:" in record["config"] and record["has_predictions"] is True
    assert any(name.endswith("loss_curve.png") for name in record["plots"])
    series = board.series("real", keys=["val/rmse"])
    assert series["turn"] == [1, 2] and len(series["val/rmse"]) == 2
    assert board.updates("real")["update"]
    assert all(event["kind"] == "checkpoint" for event in board.events("real", kinds=["checkpoint"]))
    text, kind = board.file("real", "resolved.yaml")
    assert kind.startswith("text/") and b"train" in text
    image, kind = board.file("real", "plots/loss_curve.png")
    assert kind == "image/png" and image[:4] == b"\x89PNG"
    stats = board.predictions("real")
    assert stats["rows"] > 0 and stats["pairs"][0]["kind"] == "regression"
    classes = board.predictions("fake")
    assert classes["pairs"][0]["kind"] == "classification"
    compared = board.compare(["real", "fake"], keys=["val/rmse"])
    assert set(compared["records"][0]) >= {"name", "status"} and "turn" in compared["series"]
    swept = board.sweep("sweep")
    assert swept["sweep"]["points"] == 3 and len(swept["points"]) == 3 and "lr" in swept["params"]
    assert swept["best"] is not None and swept["ranked"][0] == swept["best"]["id"]


def test_answers_refuse_escapes_and_unknown_records(root):
    board = Board(root)
    with pytest.raises(kalfa.KalfaError):
        board.resolve("../")
    with pytest.raises(kalfa.KalfaError):
        board.record("nowhere")
    with pytest.raises(kalfa.KalfaError):
        board.file("real", "../../etc/passwd")


def test_http_endpoints(server):
    page, kind = get(server, "/", raw=True)
    assert kind.startswith("text/html") and b"app.js" in page
    records = get(server, "/api/records")
    assert {entry["path"] for entry in records["records"]} >= {"real", "fake", "sweep"}
    record = get(server, "/api/record/real")
    assert record["summary"]["name"] == "real"
    series = get(server, "/api/record/real/series?keys=val/rmse,train/mse&limit=100")
    assert set(series) >= {"turn", "val/rmse", "train/mse"}
    assert get(server, "/api/record/real/updates?limit=10")["update"]
    assert get(server, "/api/record/real/events?kinds=turn")
    files = get(server, "/api/record/real/files")
    assert any(entry["path"] == "plan.json" for entry in files)
    body, kind = get(server, "/api/record/real/file?name=plots/loss_curve.png", raw=True)
    assert kind == "image/png" and body[:4] == b"\x89PNG"
    body, kind = get(server, "/api/record/real/file?name=history.jsonl&tail=1", raw=True)
    assert kind.startswith("text/") and body.count(b"\n") <= 2
    stats = get(server, "/api/record/real/predictions?bins=10&sample=20")
    assert stats["pairs"][0]["stats"]["count"] > 0 and len(stats["pairs"][0]["stats"]["scatter"]["row"]) <= 20
    compared = get(server, "/api/compare?records=real,fake&keys=val/rmse")
    assert len(compared["records"]) == 2
    swept = get(server, "/api/sweep/sweep")
    assert len(swept["points"]) == 3
    code, body = status_of(server, "/api/record/nowhere")
    assert code == 404 and "error" in body
    code, body = status_of(server, "/api/record/real/file?name=../../x")
    assert code in (400, 404) and "error" in body
    css, kind = get(server, "/static/board.css", raw=True)
    assert kind.startswith("text/css")


def test_http_stop_and_live(server, root):
    written = post(server, "/api/record/fake/stop")
    assert written["paths"] and (root / "fake" / "stop.json").exists()
    code, body = status_of(server, "/api/record/nowhere/stop")
    assert code == 404 or code == 405

    received = []

    def listen():
        request = urllib.request.Request(f"{server}/api/live?every=0.2")
        with urllib.request.urlopen(request, timeout=15) as response:
            assert response.headers.get("Content-Type", "").startswith("text/event-stream")
            while len(received) < 1:
                line = response.readline()
                if not line:
                    break
                if line.startswith(b"data:"):
                    received.append(json.loads(line[5:].strip()))

    thread = threading.Thread(target=listen, daemon=True)
    thread.start()
    time.sleep(0.6)
    with Path(root / "fake" / "history.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"turn": 4, "update": 16, "val/rmse": 0.1, "seconds": 0.1, "rules": []}) + "\n")
    thread.join(timeout=10)
    assert received and "changed" in received[0] and any("fake" in item for item in received[0]["changed"])
