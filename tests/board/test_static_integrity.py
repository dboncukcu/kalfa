import re
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

import kalfa.board
from kalfa.board.server import make_server
from records.fixtures import make_record


STATIC = Path(kalfa.board.__file__).parent / "static"
VUE = "https://cdn.jsdelivr.net/npm/vue@3.4.38/dist/vue.esm-browser.prod.js"
PLOTLY = re.compile(r"https://cdn\.jsdelivr\.net/npm/plotly\.js-dist-min@\d+\.\d+\.\d+/plotly\.min\.js")
STATIC_IMPORT = re.compile(r"""\bimport\s+(?:[^;]*?\s+from\s+)?["']([^"']+)["']""", re.S)
DYNAMIC_IMPORT = re.compile(r"""\bimport\(\s*["']([^"']+)["']\s*\)""")
SCRIPT = re.compile(r"""<script[^>]*\ssrc=["']([^"']+)["']""")
TYPES = {".js": "text/javascript", ".css": "text/css", ".html": "text/html"}


def imports_in(name):
    text = (STATIC / name).read_text(encoding="utf-8")
    return STATIC_IMPORT.findall(text) + DYNAMIC_IMPORT.findall(text)


def modules_from(start="app.js"):
    seen = []
    queue = [start]
    while queue:
        name = queue.pop()
        if name in seen:
            continue
        seen.append(name)
        if not (STATIC / name).is_file():
            continue
        for target in imports_in(name):
            if target.startswith("http"):
                continue
            queue.append(((STATIC / name).parent / target).resolve().relative_to(STATIC.resolve()).as_posix())
    return seen


MODULES = modules_from()


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    root = tmp_path_factory.mktemp("integrity")
    make_record(root / "one")
    httpd = make_server(root, host="127.0.0.1", port=0)
    thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=5)


def fetch(server, path):
    try:
        with urllib.request.urlopen(f"{server}{path}", timeout=10) as response:
            return response.status, response.headers.get("Content-Type", ""), response.read()
    except urllib.error.HTTPError as failure:
        with failure:
            return failure.code, failure.headers.get("Content-Type", ""), failure.read()


def test_the_page_is_served_with_its_scripts(server):
    status, kind, body = fetch(server, "/")
    assert status == 200 and kind.startswith("text/html")
    page = body.decode("utf-8")
    assert '<script type="module" src="/static/app.js"></script>' in page
    assert 'href="/static/board.css"' in page and 'id="app"' in page
    sources = SCRIPT.findall(page)
    assert any(PLOTLY.fullmatch(source) for source in sources), "index.html does not load Plotly from jsdelivr"
    assert all(PLOTLY.fullmatch(source) or source.startswith("/static/") for source in sources), sources


def test_the_client_starts_at_the_app_module():
    assert MODULES[0] == "app.js" and len(MODULES) > 20


@pytest.mark.parametrize("name", MODULES)
def test_every_module_the_client_imports_is_served(server, name):
    status, kind, body = fetch(server, f"/static/{name}")
    assert status == 200, f"/static/{name} answers {status}"
    assert kind.startswith("text/javascript"), f"/static/{name} is sent as {kind}"
    assert body == (STATIC / name).read_bytes()


def test_the_stylesheet_is_served(server):
    status, kind, body = fetch(server, "/static/board.css")
    assert status == 200 and kind.startswith("text/css") and b":root" in body


def test_every_file_under_static_is_served(server):
    for path in sorted(STATIC.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        name = path.relative_to(STATIC).as_posix()
        status, kind, body = fetch(server, f"/static/{name}")
        assert status == 200, f"/static/{name} answers {status}"
        assert kind.startswith(TYPES.get(path.suffix, "application/octet-stream")), f"/static/{name}: {kind}"
        assert body == path.read_bytes()


def test_no_module_is_orphaned():
    on_disk = {path.relative_to(STATIC).as_posix() for path in STATIC.rglob("*.js")}
    assert on_disk == set(MODULES)


def test_remote_imports_follow_the_contract():
    for name in MODULES:
        remote = [target for target in imports_in(name) if target.startswith("http")]
        assert set(remote) <= {VUE}, f"{name} imports {remote}"


def test_missing_static_files_answer_404(server):
    status, kind, body = fetch(server, "/static/views/record/missing.js")
    assert status == 404 and kind.startswith("application/json") and b"error" in body
    status, kind, body = fetch(server, "/static/../server.py")
    assert status == 404
