import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime

import pytest

from kalfa.board import server as serving
from records.fixtures import make_record, make_sweep


def now_text():
    return datetime.now().isoformat(timespec="seconds")


class Site:
    def __init__(self, httpd, thread, home):
        self.httpd = httpd
        self.thread = thread
        self.home = home
        self.port = httpd.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"

    @property
    def handler(self):
        return self.httpd.RequestHandlerClass


def call(url, method="GET", payload=None, headers=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, method=method, headers=dict(headers or {}))
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.headers, response.read()
    except urllib.error.HTTPError as failure:
        with failure:
            return failure.code, failure.headers, failure.read()


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    home = tmp_path_factory.mktemp("phase4_review")
    root = home / "root"
    make_record(root / "reg")
    make_record(root / "cls", status="running", updated=now_text(), classes=["a", "b"], seed=1)
    make_sweep(root / "sweep", count=3)
    httpd = serving.make_server(root, host="127.0.0.1", port=0)
    thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    thread.start()
    yield Site(httpd, thread, home)
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=5)


def test_p4_r1_where_cannot_call_methods_with_side_effects(site):
    target = site.home / "pwned.csv"
    where = urllib.parse.quote(f"row.to_csv({str(target)!r})")
    status, _, body = call(f"{site.url}/api/record/reg/predictions?where={where}")
    assert status in (400, 404) and "error" in json.loads(body)
    assert not target.exists()


def test_p4_r2_a_foreign_host_header_is_refused(site):
    headers = {"Host": f"evil.example:{site.port}", "Origin": f"http://evil.example:{site.port}"}
    status, _, body = call(f"{site.url}/api/record/cls/stop", method="POST", payload={}, headers=headers)
    assert status == 403, (status, body)


@pytest.mark.parametrize("out", ["../outside.html", "/tmp/outside.html"])
def test_p4_r3_report_out_escaping_the_root_is_refused_with_400(site, out):
    status, _, body = call(f"{site.url}/api/report", method="POST", payload={"records": ["reg"], "out": out})
    assert status == 400, (status, body)


@pytest.mark.parametrize("query", ["bins=0", "bins=-1", "sample=-1"])
def test_p4_r4_bad_prediction_numbers_are_400_not_500(site, query):
    status, _, body = call(f"{site.url}/api/record/reg/predictions?{query}")
    assert status == 400, (status, body)


def test_p4_r5_a_missing_record_named_like_a_refusal_word_is_404(site):
    status, _, body = call(f"{site.url}/api/record/outside")
    assert status == 404 and "error" in json.loads(body)


@pytest.mark.parametrize("method, allowed", [("HEAD", (200, 405)), ("OPTIONS", (405,))])
def test_p4_r6_head_and_options_are_not_501(site, method, allowed):
    status, _, _ = call(f"{site.url}/api/records", method=method)
    assert status in allowed, status


def test_p4_r7_static_garbage_is_404_not_500(site):
    status, _, body = call(f"{site.url}/static/a%00b.css")
    assert status == 404, (status, body)


def test_live_loop_ends_when_the_client_leaves(site, monkeypatch):
    monkeypatch.setattr(serving, "KEEP_ALIVE", 0.1)
    response = urllib.request.urlopen(f"{site.url}/api/live?every=0.07", timeout=10)
    try:
        assert response.readline() == b": connected\n"
        assert len(site.handler.live) == 1
    finally:
        response.close()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and site.handler.live:
        time.sleep(0.02)
    assert not site.handler.live
