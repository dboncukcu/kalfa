import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from pathlib import Path

import pytest

from kalfa.board import server as serving
from kalfa.board.answers import BadValue, EscapeError
from kalfa.board.watch import Watcher
from kalfa.core.errors import KalfaError
from records.fixtures import make_record, make_sweep


class Marks:
    def __init__(self, marks=None):
        self.marks = dict(marks or {"a": 1, "b": 1})
        self.broken = False

    def stamps(self):
        if self.broken:
            raise RuntimeError("unreadable")
        return dict(self.marks)


class FakeBoard:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.marks = {"a": 1}

    def resolve(self, relative):
        if relative.startswith("hidden"):
            raise EscapeError(f"{relative} lies outside {self.root}")
        if relative == "nowhere":
            raise KalfaError(f"no record directory {relative} under {self.root}")
        return self.root / relative

    def records(self):
        return {"root": str(self.root), "records": [{"path": "a", "kind": "record"}]}

    def record(self, relative):
        self.resolve(relative)
        if relative == "broken":
            raise RuntimeError("the record broke")
        return {"path": relative}

    def series(self, relative, keys=None, limit=2000):
        self.resolve(relative)
        return {"call": "series", "path": relative, "keys": keys, "limit": limit}

    def updates(self, relative, keys=None, limit=2000):
        self.resolve(relative)
        return {"call": "updates", "path": relative, "keys": keys, "limit": limit}

    def events(self, relative, kinds=None, limit=None):
        self.resolve(relative)
        return [{"call": "events", "path": relative, "kinds": kinds, "limit": limit}]

    def files(self, relative):
        self.resolve(relative)
        return [{"path": "history.jsonl", "size": 3}]

    def file(self, relative, name, tail=None):
        self.resolve(relative)
        if name == "missing.txt":
            raise KalfaError(f"no file {name} in {relative}")
        return f"{relative}|{name}|{tail}".encode("utf-8"), "text/plain; charset=utf-8"

    def predictions(self, relative, where=None, sample=2000, bins=40, name="predictions.parquet", **choices):
        self.resolve(relative)
        if where == "bad":
            raise BadValue(f"bad where for {name}: nope")
        return {"call": "predictions", "path": relative, "where": where, "sample": sample, "bins": bins,
                "name": name, **choices}

    def compare(self, relatives, keys=None):
        for relative in relatives:
            self.resolve(relative)
        return {"names": relatives, "keys": keys}

    def sweep(self, relative):
        self.resolve(relative)
        return {"call": "sweep", "path": relative}

    def stop(self, relative):
        self.resolve(relative)
        return {"paths": [f"{relative}/stop.json"]}

    def report(self, relatives, out=None):
        time.sleep(0.05)
        if "fail" in relatives:
            raise KalfaError("nothing to report")
        if "crash" in relatives:
            raise ValueError("the report crashed")
        return self.root / (out or "report.html")

    def stamps(self):
        return dict(self.marks)


class Site:
    def __init__(self, httpd):
        self.httpd = httpd
        self.url = f"http://127.0.0.1:{httpd.server_address[1]}"

    @property
    def handler(self):
        return self.httpd.RequestHandlerClass

    @property
    def board(self):
        return self.handler.board


def start(root):
    httpd = serving.make_server(root, host="127.0.0.1", port=0)
    thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    thread.start()
    return httpd, thread


def finish(httpd, thread):
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=5)


def call(url, method="GET", payload=None, headers=None):
    data = payload if payload is None or isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, method=method, headers=dict(headers or {}))
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.headers, response.read()
    except urllib.error.HTTPError as failure:
        with failure:
            return failure.code, failure.headers, failure.read()


def answer(url, method="GET", payload=None, headers=None):
    status, _, body = call(url, method, payload, headers)
    return status, json.loads(body or b"null")


def until_data(response):
    lines = []
    while True:
        line = response.readline()
        assert line, f"the stream ended after {lines}"
        text = line.decode("utf-8").rstrip("\n")
        lines.append(text)
        if text.startswith("data:"):
            return lines, json.loads(text[len("data:"):])


def finished(site, job):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        status, state = answer(f"{site.url}/api/report/{job}")
        assert status == 200
        if state["status"] != "running":
            return state
        time.sleep(0.02)
    raise AssertionError(f"the report job {job} did not finish")


@pytest.fixture
def fake(tmp_path, monkeypatch):
    monkeypatch.setattr(serving, "Board", FakeBoard)
    monkeypatch.setattr(serving, "jsonable", lambda value: value)
    httpd, thread = start(tmp_path)
    yield Site(httpd)
    finish(httpd, thread)


def test_watcher_reports_changed_paths():
    marks = Marks()
    watcher = Watcher(marks, every=0.02).start()
    try:
        assert watcher.wait(0, timeout=0.1) is None
        marks.marks = {"a": 1, "b": 2, "c": 1}
        assert watcher.wait(0, timeout=5) == 1
        assert watcher.changed == ["b", "c"] and watcher.since(0) == (1, ["b", "c"])
        marks.marks = {"b": 2, "c": 1}
        assert watcher.wait(1, timeout=5) == 2
        assert watcher.changed == ["a"] and watcher.since(0) == (2, ["a", "b", "c"]) and watcher.since(1) == (2, ["a"])
        assert watcher.since(2) == (2, [])
    finally:
        watcher.stop()
    assert watcher.thread is None and watcher.stopping


def test_watcher_start_is_idempotent_and_restartable():
    watcher = Watcher(Marks(), every=0.02)
    watcher.start()
    first = watcher.thread
    assert watcher.start().thread is first and first.is_alive()
    watcher.stop()
    assert not first.is_alive()
    watcher.start()
    assert watcher.thread is not first and watcher.thread.is_alive() and not watcher.stopping
    watcher.stop()
    watcher.stop()


def test_watcher_survives_failing_stamps():
    marks = Marks()
    watcher = Watcher(marks, every=0.02).start()
    try:
        marks.broken = True
        assert watcher.wait(0, timeout=0.15) is None and watcher.failure == "RuntimeError: unreadable"
        assert watcher.thread.is_alive()
        marks.marks = {"a": 1, "b": 1, "d": 1}
        marks.broken = False
        assert watcher.wait(0, timeout=5) == 1 and watcher.changed == ["d"] and watcher.failure is None
    finally:
        watcher.stop()


def test_watcher_stop_wakes_waiters():
    watcher = Watcher(Marks(), every=10.0).start()
    found = []
    waiter = threading.Thread(target=lambda: found.append(watcher.wait(0, timeout=10)))
    waiter.start()
    time.sleep(0.05)
    began = time.monotonic()
    watcher.stop()
    waiter.join(timeout=5)
    assert found == [None] and time.monotonic() - began < 2


def test_watcher_since_after_missed_versions():
    marks = Marks()
    watcher = Watcher(marks, every=10.0)
    watcher.log = deque(maxlen=1)
    watcher.stamps = marks.stamps()
    marks.marks = {"a": 2, "b": 1}
    assert watcher.poll() is True
    marks.marks = {"a": 2, "b": 2, "e": 1}
    assert watcher.poll() is True and watcher.poll() is False
    assert watcher.since(1) == (2, ["b", "e"])
    assert watcher.since(0) == (2, ["a", "b", "e"])


def test_route_helpers():
    assert serving.split_record("a/b/c/series") == ("a/b/c", "series")
    assert serving.split_record("a/b/c") == ("a/b/c", None)
    assert serving.split_record("files") == ("files", None)
    assert serving.split_record("a//b/stop") == ("a/b", "stop")
    assert serving.escapes("../x") and serving.escapes("/etc") and serving.escapes("a/../../b")
    assert serving.escapes("a\\..\\b") and not serving.escapes("a/b..c")
    assert serving.content_type_of("board.css").startswith("text/css")
    assert serving.content_type_of("app.js").startswith("text/javascript")
    assert serving.content_type_of("x.PNG") == "image/png"
    assert serving.content_type_of("logo.svg") == "image/svg+xml"
    assert serving.content_type_of("blob.bin") == "application/octet-stream"
    assert serving.status_of(KalfaError("no record directory x under /r")) == 404
    assert serving.status_of(KalfaError("outside is not a record directory")) == 404
    assert serving.status_of(serving.NotFound("no route /x")) == 404
    assert serving.status_of(EscapeError("x lies outside /r")) == 400
    assert serving.status_of(BadValue("bins must be at least 1, not '0'")) == 400
    assert serving.status_of(serving.BadRequest("name is required")) == 400


def test_records_and_record_routes(fake):
    status, headers, body = call(f"{fake.url}/api/records")
    assert status == 200 and headers["Content-Type"] == "application/json"
    assert headers["Cache-Control"] == "no-store" and int(headers["Content-Length"]) == len(body)
    assert json.loads(body)["records"][0]["path"] == "a"
    assert answer(f"{fake.url}/api/record/a/b/c") == (200, {"path": "a/b/c"})
    assert answer(f"{fake.url}/api/record/a/b/c/") == (200, {"path": "a/b/c"})
    assert answer(f"{fake.url}/api/record/files") == (200, {"path": "files"})
    assert answer(f"{fake.url}/api/record/my%20run") == (200, {"path": "my run"})


def test_query_parameters_reach_the_board(fake):
    found = answer(f"{fake.url}/api/record/a/b/c/series?keys=val/rmse,train/mse&limit=5")[1]
    assert found == {"call": "series", "path": "a/b/c", "keys": ["val/rmse", "train/mse"], "limit": 5}
    found = answer(f"{fake.url}/api/record/a/series")[1]
    assert found["keys"] is None and found["limit"] == 2000
    found = answer(f"{fake.url}/api/record/a/updates?keys=loss/main&limit=")[1]
    assert found == {"call": "updates", "path": "a", "keys": ["loss/main"], "limit": 2000}
    found = answer(f"{fake.url}/api/record/a/events?kinds=turn,patch&limit=3")[1]
    assert found == [{"call": "events", "path": "a", "kinds": ["turn", "patch"], "limit": 3}]
    assert answer(f"{fake.url}/api/record/a/files")[1] == [{"path": "history.jsonl", "size": 3}]
    where = urllib.parse.quote("town == 'ank'")
    found = answer(f"{fake.url}/api/record/a/predictions?where={where}&sample=20&bins=10&name=p.parquet")[1]
    assert found == {"call": "predictions", "path": "a", "where": "town == 'ank'", "sample": 20, "bins": 10,
                     "name": "p.parquet"}
    found = answer(f"{fake.url}/api/record/a/predictions?where=")[1]
    assert found["where"] is None and found["sample"] == 2000 and found["name"] == "predictions.parquet"
    assert "histograms" not in found and "pair" not in found
    histograms = "residual_hist:60,%20truth_hist%3A30"
    found = answer(f"{fake.url}/api/record/a/predictions?pair=pred_net&histograms={histograms}")[1]
    assert found["pair"] == "pred_net" and found["histograms"] == {"residual_hist": 60, "truth_hist": 30}
    assert "histograms" not in answer(f"{fake.url}/api/record/a/predictions?histograms=")[1]
    for wrong, said in (("residual_hist", "name:count"), ("residual_hist:", "name:count"), (":7", "name:count"),
                        ("residual_hist:many", "must be an integer")):
        status, body = answer(f"{fake.url}/api/record/a/predictions?pair=pred_net&histograms={wrong}")
        assert status == 400 and said in body["error"]
    assert answer(f"{fake.url}/api/compare?records=a,b/c&keys=val/rmse")[1] == {"names": ["a", "b/c"],
                                                                              "keys": ["val/rmse"]}
    assert answer(f"{fake.url}/api/compare?records=a")[1] == {"names": ["a"], "keys": None}
    assert answer(f"{fake.url}/api/sweep/s/t") == (200, {"call": "sweep", "path": "s/t"})
    assert answer(f"{fake.url}/api/sweep/s/files") == (200, {"call": "sweep", "path": "s/files"})


def test_file_route_sends_bytes(fake):
    status, headers, body = call(f"{fake.url}/api/record/a/b/file?name=plots/x.png&tail=4")
    assert status == 200 and body == b"a/b|plots/x.png|4"
    assert headers["Content-Type"] == "text/plain; charset=utf-8" and headers["Cache-Control"] == "no-store"
    assert int(headers["Content-Length"]) == len(body)
    assert call(f"{fake.url}/api/record/a/file?name=resolved.yaml")[2] == b"a|resolved.yaml|None"


def test_errors_and_codes(fake):
    status, body = answer(f"{fake.url}/api/record/nowhere")
    assert status == 404 and "nowhere" in body["error"]
    assert answer(f"{fake.url}/api/record/hidden")[0] == 400
    assert answer(f"{fake.url}/api/record/%2e%2e/x")[0] == 400
    assert answer(f"{fake.url}/api/record/a/file?name=../../x")[0] == 400
    assert answer(f"{fake.url}/api/record/a/file")[0] == 400
    assert answer(f"{fake.url}/api/record/a/file?name=missing.txt")[0] == 404
    status, body = answer(f"{fake.url}/api/record/a/series?limit=many")
    assert status == 400 and "limit" in body["error"]
    assert answer(f"{fake.url}/api/record/a/predictions?where=bad")[0] == 400
    assert answer(f"{fake.url}/api/compare")[0] == 400
    assert answer(f"{fake.url}/api/compare?records=a,../b")[0] == 400
    assert answer(f"{fake.url}/api/record/")[0] == 404
    assert answer(f"{fake.url}/api/nothing")[0] == 404
    assert answer(f"{fake.url}/elsewhere")[0] == 404
    status, body = answer(f"{fake.url}/api/record/broken")
    assert status == 500 and body["error"] == "RuntimeError: the record broke"


def test_wrong_methods(fake):
    status, headers, body = call(f"{fake.url}/api/records", method="POST", payload={})
    assert status == 405 and headers["Allow"] == "GET" and "error" in json.loads(body)
    status, headers, _ = call(f"{fake.url}/api/record/a/stop")
    assert status == 405 and headers["Allow"] == "POST"
    assert call(f"{fake.url}/api/report")[0] == 405
    assert call(f"{fake.url}/api/record/a", method="DELETE")[0] == 405
    assert call(f"{fake.url}/api/nothing", method="PUT", payload={})[0] == 404


def test_static_files(fake):
    page = (serving.STATIC / "index.html").read_bytes()
    for path in ("/", "/index.html", "/static/index.html"):
        status, headers, body = call(f"{fake.url}{path}")
        assert status == 200 and headers["Content-Type"].startswith("text/html") and body == page
    assert call(f"{fake.url}/static/%2e%2e/server.py")[0] == 404
    assert call(f"{fake.url}/static/..%2f..%2fpyproject.toml")[0] == 404
    assert call(f"{fake.url}/static/nothing-here.css")[0] == 404
    assert call(f"{fake.url}/static/")[0] == 404


def test_stop_route(fake):
    assert answer(f"{fake.url}/api/record/a/b/stop", method="POST", payload={}) == (200, {"paths": ["a/b/stop.json"]})
    assert answer(f"{fake.url}/api/record/a/stop", method="POST") == (200, {"paths": ["a/stop.json"]})
    assert answer(f"{fake.url}/api/record/nowhere/stop", method="POST", payload={})[0] == 404
    own = {"Origin": fake.url}
    assert answer(f"{fake.url}/api/record/a/stop", method="POST", payload={}, headers=own)[0] == 200
    other = {"Origin": "http://example.com"}
    status, body = answer(f"{fake.url}/api/record/a/stop", method="POST", payload={}, headers=other)
    assert status == 403 and "origin" in body["error"]



def test_host_names():
    names = frozenset({"gpubox", "gpubox.lab"})
    for value in ("127.0.0.1", "127.0.0.1:8765", "LOCALHOST:8765", "localhost.", "[::1]:8765", "[::1]", "10.0.0.7:80",
                  "gpubox:8765", "GpuBox.lab"):
        assert serving.host_allowed(value, names), value
    for value in ("", "evil.example:8765", "localhost.evil.example", "evil.localhost", "localhost:", "localhost:http",
                  "127.0.0.1@evil.example", "[::1", "[::1]x", "::1", "gpubox.evil.example:8765", "127.0.0.1:8765:1"):
        assert not serving.host_allowed(value, names), value


def test_names_of_the_server(monkeypatch):
    monkeypatch.setattr(serving.socket, "gethostname", lambda: "GpuBox.lab.")
    assert serving.names_of("127.0.0.1") == {"127.0.0.1", "gpubox.lab", "gpubox"}
    assert serving.names_of("") == {"gpubox.lab", "gpubox"}
    assert serving.names_of("Board.Lan") == {"board.lan", "gpubox.lab", "gpubox"}


def test_foreign_host_names_are_refused(fake):
    port = fake.httpd.server_address[1]
    for host in (f"evil.example:{port}", f"localhost.evil.example:{port}", f"board.lan:{port}"):
        status, body = answer(f"{fake.url}/api/records", headers={"Host": host})
        assert status == 403 and "host" in body["error"], host
        rebound = {"Host": host, "Origin": f"http://{host}"}
        assert answer(f"{fake.url}/api/record/a/stop", method="POST", payload={}, headers=rebound)[0] == 403
        assert call(f"{fake.url}/", headers={"Host": host})[0] == 403
        assert call(f"{fake.url}/api/nothing", headers={"Host": host})[0] == 403
    for host in (f"localhost:{port}", f"[::1]:{port}", f"127.0.0.1:{port}", "127.0.0.1"):
        assert answer(f"{fake.url}/api/records", headers={"Host": host})[0] == 200, host
        own = {"Host": host, "Origin": f"http://{host}"}
        assert answer(f"{fake.url}/api/record/a/stop", method="POST", payload={}, headers=own)[0] == 200, host
    fake.handler.hosts = frozenset({"board.lan"})
    assert answer(f"{fake.url}/api/records", headers={"Host": f"board.lan:{port}"})[0] == 200


def test_fetch_site_guards_posts_and_where(fake):
    stop = f"{fake.url}/api/record/a/stop"
    for site in ("same-origin", "none", "Same-Origin"):
        assert answer(stop, method="POST", payload={}, headers={"Sec-Fetch-Site": site})[0] == 200, site
    for site in ("cross-site", "same-site"):
        status, body = answer(stop, method="POST", payload={}, headers={"Sec-Fetch-Site": site})
        assert status == 403 and "origin" in body["error"], site
    assert answer(stop, method="POST", payload={}, headers={"Origin": "null"})[0] == 403
    cross = {"Sec-Fetch-Site": "cross-site"}
    assert answer(f"{fake.url}/api/report", method="POST", payload={"records": ["a"]}, headers=cross)[0] == 403
    predictions = f"{fake.url}/api/record/a/predictions"
    assert answer(f"{predictions}?where=x", headers=cross)[0] == 403
    assert answer(f"{predictions}?where=", headers=cross)[0] == 403
    assert answer(f"{predictions}?where=x", headers={"Origin": "http://example.com"})[0] == 403
    assert answer(f"{predictions}?where=x", headers={"Sec-Fetch-Site": "same-origin"})[1]["where"] == "x"
    assert answer(f"{predictions}?where=x", headers={"Sec-Fetch-Site": "none"})[1]["where"] == "x"
    assert answer(f"{predictions}?where=x", headers={"Origin": fake.url})[1]["where"] == "x"
    assert answer(predictions, headers=cross)[0] == 200
    assert answer(f"{fake.url}/api/records", headers=cross)[0] == 200
    assert call(f"{fake.url}/", headers=cross)[0] == 200

def test_report_jobs(fake):
    status, body = answer(f"{fake.url}/api/report", method="POST", payload={"records": ["a", "b/c"], "out": None})
    assert status == 200 and set(body) == {"job"}
    assert answer(f"{fake.url}/api/report/{body['job']}")[1]["status"] in ("running", "done")
    state = finished(fake, body["job"])
    assert state == {"status": "done", "path": str(fake.board.root / "report.html"), "error": None}
    job = answer(f"{fake.url}/api/report", method="POST", payload={"records": ["a"], "out": "x.html"})[1]["job"]
    assert finished(fake, job)["path"].endswith("x.html")
    job = answer(f"{fake.url}/api/report", method="POST", payload={"records": ["fail"]})[1]["job"]
    assert finished(fake, job) == {"status": "failed", "path": None, "error": "nothing to report"}
    job = answer(f"{fake.url}/api/report", method="POST", payload={"records": ["crash"]})[1]["job"]
    assert finished(fake, job)["error"] == "ValueError: the report crashed"
    for payload in ({}, {"records": []}, {"records": "a"}, {"records": [""]}, {"records": ["a"], "out": 3},
                    {"records": ["../a"]}, b"not json", b"[1, 2]"):
        assert answer(f"{fake.url}/api/report", method="POST", payload=payload)[0] == 400, payload
    assert answer(f"{fake.url}/api/report", method="POST", payload={"records": ["nowhere"]})[0] == 404
    assert answer(f"{fake.url}/api/report/unknown")[0] == 404


def test_live_reports_changes(fake):
    with urllib.request.urlopen(f"{fake.url}/api/live?every=0.05", timeout=10) as response:
        assert response.headers["Content-Type"].startswith("text/event-stream")
        assert response.headers["Cache-Control"] == "no-cache"
        assert response.readline() == b": connected\n"
        fake.board.marks = {"a": 1, "b": 1}
        lines, data = until_data(response)
        assert lines[-2] == "event: change" and data == {"version": 1, "changed": ["b"]}
        fake.board.marks = {"b": 2}
        lines, data = until_data(response)
        assert data == {"version": 2, "changed": ["a", "b"]}
    assert fake.handler.watcher.thread is None


def test_live_keep_alive(fake, monkeypatch):
    monkeypatch.setattr(serving, "KEEP_ALIVE", 0.1)
    with urllib.request.urlopen(f"{fake.url}/api/live?every=0.05", timeout=10) as response:
        assert response.readline() == b": connected\n"
        assert response.readline() == b"\n"
        assert response.readline() == b": keep-alive\n"


def test_live_refuses_bad_every(fake):
    assert answer(f"{fake.url}/api/live?every=fast")[0] == 400
    assert answer(f"{fake.url}/api/live?every=0")[0] == 400


def test_live_uses_the_shared_watcher_on_demand(fake):
    watcher = fake.handler.watcher
    watcher.every = 0.05
    assert watcher.thread is None
    with urllib.request.urlopen(f"{fake.url}/api/live", timeout=10) as response:
        assert response.readline() == b": connected\n"
        assert watcher.thread is not None and watcher.thread.is_alive() and not fake.handler.live
        fake.board.marks = {"a": 1, "z": 2}
        lines, data = until_data(response)
        assert data == {"version": 1, "changed": ["z"]}


def test_server_closes_cleanly(tmp_path, monkeypatch):
    monkeypatch.setattr(serving, "Board", FakeBoard)
    monkeypatch.setattr(serving, "jsonable", lambda value: value)
    httpd, thread = start(tmp_path)
    site = Site(httpd)
    response = urllib.request.urlopen(f"{site.url}/api/live?every=0.05", timeout=10)
    try:
        assert response.readline() == b": connected\n"
        own = list(site.handler.live)
        assert len(own) == 1 and own[0].thread.is_alive()
        finish(httpd, thread)
        assert not thread.is_alive() and own[0].thread is None and site.handler.watcher.thread is None
        assert response.read() == b"\n"
    finally:
        response.close()
    assert not site.handler.live
    with pytest.raises(urllib.error.URLError):
        urllib.request.urlopen(f"{site.url}/api/records", timeout=2)


def test_make_server_refuses_a_missing_root(tmp_path):
    with pytest.raises(KalfaError):
        serving.make_server(tmp_path / "nowhere", host="127.0.0.1", port=0)


def test_make_server_refuses_a_taken_port(tmp_path, monkeypatch):
    monkeypatch.setattr(serving, "Board", FakeBoard)
    httpd, thread = start(tmp_path)
    try:
        with pytest.raises(KalfaError):
            serving.make_server(tmp_path, host="127.0.0.1", port=httpd.server_address[1])
    finally:
        finish(httpd, thread)


def test_serve_prints_opens_and_stops_on_ctrl_c(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(serving, "Board", FakeBoard)
    opened = []
    closed = []
    original = serving.BoardServer.server_close
    monkeypatch.setattr(serving.webbrowser, "open", opened.append)

    def interrupt(self):
        raise KeyboardInterrupt

    def close(self):
        closed.append(self.server_address[1])
        original(self)

    monkeypatch.setattr(serving.BoardServer, "service_actions", interrupt)
    monkeypatch.setattr(serving.BoardServer, "server_close", close)
    serving.serve(tmp_path, host="127.0.0.1", port=0, open_browser=True)
    printed = capsys.readouterr().out
    assert len(opened) == 1 and opened[0].startswith("http://127.0.0.1:") and opened[0] in printed
    assert closed and opened[0].endswith(f":{closed[0]}/")


@pytest.fixture(scope="module")
def real(tmp_path_factory):
    home = tmp_path_factory.mktemp("served")
    make_record(home / "runs" / "one", status="running")
    make_record(home / "runs" / "group" / "two", classes=["a", "b"])
    make_sweep(home / "sweep", count=3)
    httpd, thread = start(home)
    yield Site(httpd)
    finish(httpd, thread)


def real_answer(site, path, verb="GET", payload=None):
    return answer(f"{site.url}{path}", verb, payload)


def test_real_records(real):
    status, body = real_answer(real, "/api/records")
    assert status == 200 and {"runs/one", "runs/group/two", "sweep"} <= {entry["path"] for entry in body["records"]}


def test_real_nested_record(real):
    status, body = real_answer(real, "/api/record/runs/group/two")
    assert status == 200 and body["path"] == "runs/group/two"
    status, body = real_answer(real, "/api/record/runs/group/two/series?keys=val/rmse&limit=10")
    assert status == 200 and body["turn"] == [1, 2, 3] and len(body["val/rmse"]) == 3
    status, body = real_answer(real, "/api/record/runs/one/updates?keys=loss/main&limit=5")
    assert status == 200 and len(body["update"]) <= 5 and "loss/main" in body
    status, body = real_answer(real, "/api/record/runs/one/events?kinds=checkpoint")
    assert status == 200 and body and all(event["kind"] == "checkpoint" for event in body)
    status, body = real_answer(real, "/api/record/runs/one/files")
    assert status == 200 and any(entry["path"] == "plan.json" for entry in body)


def test_real_file(real):
    status, headers, body = call(f"{real.url}/api/record/runs/one/file?name=plots/loss_curve.png")
    assert status == 200 and headers["Content-Type"] == "image/png" and body == b"png"
    status, headers, body = call(f"{real.url}/api/record/runs/one/file?name=history.jsonl&tail=1")
    assert status == 200 and headers["Content-Type"].startswith("text/") and body.count(b"\n") <= 2
    assert answer(f"{real.url}/api/record/runs/one/file?name=../../x")[0] == 400
    assert answer(f"{real.url}/api/record/runs/one/file?name=nothing.txt")[0] == 404


def test_real_predictions(real):
    status, body = real_answer(real, "/api/record/runs/one/predictions?sample=10&bins=5")
    assert status == 200 and body["rows"] > 0 and body["pairs"][0]["kind"] == "regression"
    status, body = real_answer(real, "/api/record/runs/group/two/predictions")
    assert status == 200 and body["pairs"][0]["kind"] == "classification"


def test_real_compare_and_sweep(real):
    status, body = real_answer(real, "/api/compare?records=runs/one,runs/group/two&keys=val/rmse")
    assert status == 200 and len(body["records"]) == 2
    status, body = real_answer(real, "/api/sweep/sweep")
    assert status == 200 and len(body["points"]) == 3


def test_real_unknown_records(real):
    status, body = real_answer(real, "/api/record/nowhere")
    assert status == 404 and "error" in body
    assert answer(f"{real.url}/api/record/runs/%2e%2e/%2e%2e/x")[0] == 400
    assert real_answer(real, "/api/sweep/nowhere")[0] == 404


def test_real_stop(real):
    status, body = real_answer(real, "/api/record/runs/one/stop", "POST", {})
    assert status == 200 and body["paths"]
    assert (real.board.root / "runs" / "one" / "stop.json").exists()
    assert real_answer(real, "/api/record/nowhere/stop", "POST", {})[0] == 404


def test_real_report(real):
    status, body = real_answer(real, "/api/report", "POST", {"records": ["runs/group/two"]})
    assert status == 200
    state = finished(real, body["job"])
    assert state["status"] == "done" and state["path"].endswith(".html") and Path(state["path"]).exists()


def test_real_live(real):
    assert answer(f"{real.url}/api/records")[0] == 200
    with urllib.request.urlopen(f"{real.url}/api/live?every=0.1", timeout=10) as response:
        assert response.readline() == b": connected\n"
        with (real.board.root / "runs" / "group" / "two" / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"turn": 4, "update": 16, "val/rmse": 0.1, "seconds": 0.1, "rules": []}) + "\n")
        lines, data = until_data(response)
    assert data["version"] >= 1 and "runs/group/two" in data["changed"]
