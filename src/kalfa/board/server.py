import ipaddress
import json
import logging
import math
import re
import socket
import threading
import uuid
import webbrowser
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from socketserver import TCPServer
from urllib.parse import parse_qs, unquote, urlsplit

from kalfa.core.errors import KalfaError

from .answers import BadValue, Board, EscapeError, inside, jsonable
from .watch import Watcher


HOST = "127.0.0.1"
PORT = 8765
STATIC = Path(__file__).parent / "static"
KEEP_ALIVE = 15.0
SHORTEST = 0.05
BODY_LIMIT = 1_000_000
SUFFIXES = ("series", "updates", "events", "files", "file", "predictions", "image", "stop")
CHOICES = ("pair", "task", "score", "positive", "threshold")
FETCH_SITES = ("same-origin", "none")
PORT_TEXT = re.compile(r"(?::\d{1,5})?")
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".json": "application/json",
    ".map": "application/json",
    ".txt": "text/plain; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".ico": "image/x-icon",
}
LOGGER = logging.getLogger("kalfa.board")


class BadRequest(KalfaError):
    pass


class NotFound(KalfaError):
    pass


REFUSED = (BadRequest, EscapeError, BadValue)


def content_type_of(name):
    return CONTENT_TYPES.get(Path(name).suffix.lower(), "application/octet-stream")


def parts_of(rest):
    return [part for part in rest.split("/") if part]


def split_record(rest):
    parts = parts_of(rest)
    if len(parts) > 1 and parts[-1] in SUFFIXES:
        return "/".join(parts[:-1]), parts[-1]
    return "/".join(parts), None


def escapes(relative):
    text = str(relative).replace("\\", "/")
    return text.startswith("/") or ".." in text.split("/")


def checked(relative, what="the root"):
    if escapes(relative):
        raise BadRequest(f"{relative} escapes {what}")
    return relative


def status_of(error):
    return 400 if isinstance(error, REFUSED) else 404


def host_name(value):
    text = value.strip().lower()
    if text.startswith("["):
        name, closed, port = text[1:].partition("]")
        if not closed:
            return None
    else:
        name, colon, digits = text.partition(":")
        port = colon + digits
    if not PORT_TEXT.fullmatch(port):
        return None
    return name.rstrip(".")


def host_allowed(value, names=()):
    name = host_name(value)
    if not name:
        return False
    if name == "localhost" or name in names:
        return True
    try:
        ipaddress.ip_address(name)
    except ValueError:
        return False
    return True


def names_of(host):
    machine = socket.gethostname().strip().lower().rstrip(".")
    configured = str(host or "").strip().strip("[]").lower().rstrip(".")
    return frozenset(name for name in (configured, machine, machine.split(".")[0]) if name)


def text_of(query, key):
    values = query.get(key)
    return values[-1] if values else None


def listed(query, key):
    text = text_of(query, key)
    if text is None:
        return None
    items = [item.strip() for item in text.split(",") if item.strip()]
    return items or None


def number(query, key, kind=int):
    text = text_of(query, key)
    if text is None or not text.strip():
        return None
    try:
        value = kind(text)
    except ValueError:
        raise BadRequest(f"{key} must be {'an integer' if kind is int else 'a number'}, not {text!r}") from None
    if kind is float and (not math.isfinite(value) or value <= 0):
        raise BadRequest(f"{key} must be a positive number, not {text!r}")
    return value


def given(**options):
    return {key: value for key, value in options.items() if value is not None}


def choices_of(query):
    return given(**{key: text_of(query, key) or None for key in CHOICES})


def histograms_of(query):
    found = {}
    for item in listed(query, "histograms") or []:
        name, colon, count = (part.strip() for part in item.partition(":"))
        if not colon or not name or not count:
            raise BadRequest(f"histograms takes name:count pairs, like residual_hist:60, not {item!r}")
        try:
            found[name] = int(count)
        except ValueError:
            raise BadRequest(f"the bins of {name} must be an integer, not {count!r}") from None
    return found or None


def run_report(board, jobs, lock, job, records, out):
    try:
        path = board.report(records, out=out)
    except Exception as error:
        if isinstance(error, KalfaError):
            message = str(error)
        else:
            LOGGER.exception("the report job %s failed", job)
            message = f"{type(error).__name__}: {error}"
        result = {"status": "failed", "path": None, "error": message}
    else:
        result = {"status": "done", "path": str(path), "error": None}
    with lock:
        jobs[job] = result


class Handler(BaseHTTPRequestHandler):
    board = None
    watcher = None
    jobs = None
    live = None
    lock = None
    hosts = frozenset()

    def do_GET(self):
        self.dispatch("GET")

    def do_POST(self):
        self.dispatch("POST")

    def do_PUT(self):
        self.dispatch("PUT")

    def do_PATCH(self):
        self.dispatch("PATCH")

    def do_DELETE(self):
        self.dispatch("DELETE")

    def do_HEAD(self):
        self.dispatch("HEAD")

    def do_OPTIONS(self):
        self.dispatch("OPTIONS")

    def log_message(self, format, *args):
        LOGGER.debug(format, *args)

    def dispatch(self, method):
        self.answered = False
        self.body = b""
        url = urlsplit(self.path)
        path = unquote(url.path)
        query = parse_qs(url.query, keep_blank_values=True)
        try:
            self.body = self.read_body()
            if not self.local():
                self.send_json({"error": "requests for another host name are refused"}, 403, plain=True)
                return
            actions = self.routes(path)
            if actions is None:
                raise NotFound(f"no route {path}")
            if method not in actions:
                self.send_json({"error": f"{method} is not allowed on {path}"}, 405, plain=True,
                               headers={"Allow": ", ".join(actions)})
                return
            if (method != "GET" or "where" in query) and self.foreign():
                self.send_json({"error": "requests from another origin are refused"}, 403, plain=True)
                return
            actions[method](query)
        except ConnectionError:
            self.close_connection = True
        except KalfaError as error:
            self.fail(status_of(error), str(error))
        except Exception as error:
            if self.answered and isinstance(error, OSError):
                self.close_connection = True
                return
            LOGGER.exception("%s %s failed", method, self.path)
            self.fail(500, f"{type(error).__name__}: {error}")

    def routes(self, path):
        if path in ("/", "/index.html"):
            return {"GET": partial(self.get_static, "index.html")}
        if path.startswith("/static/"):
            return {"GET": partial(self.get_static, path[len("/static/"):])}
        if path == "/api/records":
            return {"GET": self.get_records}
        if path == "/api/compare":
            return {"GET": self.get_compare}
        if path == "/api/live":
            return {"GET": self.get_live}
        if path == "/api/report":
            return {"POST": self.post_report}
        if path.startswith("/api/report/"):
            return {"GET": partial(self.get_report, path[len("/api/report/"):])}
        if path.startswith("/api/sweep/"):
            relative = "/".join(parts_of(path[len("/api/sweep/"):]))
            return {"GET": partial(self.get_sweep, self.located(relative))}
        if path.startswith("/api/record/"):
            relative, suffix = split_record(path[len("/api/record/"):])
            return self.record_routes(self.located(relative), suffix)
        return None

    def located(self, relative):
        if not relative:
            raise NotFound("a record path is required")
        return checked(relative)

    def record_routes(self, relative, suffix):
        if suffix == "stop":
            return {"POST": partial(self.post_stop, relative)}
        readers = {None: self.get_record, "series": self.get_series, "updates": self.get_updates,
                   "events": self.get_events, "files": self.get_files, "file": self.get_file,
                   "predictions": self.get_predictions, "image": self.get_image}
        return {"GET": partial(readers[suffix], relative)}

    def get_static(self, name, query):
        base = STATIC.resolve()
        try:
            target = (base / name).resolve()
            found = target.is_relative_to(base) and target.is_file()
        except (OSError, ValueError):
            found = False
        if not found:
            raise NotFound(f"no static file {name!r}")
        self.send_body(200, target.read_bytes(), content_type_of(target.name), cache="no-cache")

    def get_records(self, query):
        self.send_json(self.board.records())

    def get_record(self, relative, query):
        self.send_json(self.board.record(relative))

    def get_series(self, relative, query):
        self.send_json(self.board.series(relative, **given(keys=listed(query, "keys"), limit=number(query, "limit"))))

    def get_updates(self, relative, query):
        self.send_json(self.board.updates(relative, **given(keys=listed(query, "keys"),
                                                            limit=number(query, "limit"))))

    def get_events(self, relative, query):
        self.send_json(self.board.events(relative, **given(kinds=listed(query, "kinds"),
                                                           limit=number(query, "limit"))))

    def get_files(self, relative, query):
        self.send_json(self.board.files(relative))

    def get_file(self, relative, query):
        name = text_of(query, "name")
        if not name:
            raise BadRequest("name is required")
        checked(name, "the record")
        body, kind = self.board.file(relative, name, **given(tail=number(query, "tail")))
        self.send_body(200, body, kind)

    def get_predictions(self, relative, query):
        name = text_of(query, "name") or None
        if name is not None:
            checked(name, "the record")
        options = given(where=text_of(query, "where") or None, sample=number(query, "sample"),
                        bins=number(query, "bins"), name=name, histograms=histograms_of(query), **choices_of(query))
        self.send_json(self.board.predictions(relative, **options))

    def get_image(self, relative, query):
        name = text_of(query, "name") or None
        if name is not None:
            checked(name, "the record")
        index = number(query, "index")
        if index is None:
            raise BadRequest("index is required")
        body, kind = self.board.image(relative, index, **given(name=name))
        self.send_body(200, body, kind, cache="private, max-age=60")

    def get_compare(self, query):
        relatives = listed(query, "records")
        if not relatives:
            raise BadRequest("records is required")
        for relative in relatives:
            checked(relative)
        self.send_json(self.board.compare(relatives, **given(keys=listed(query, "keys"))))

    def get_sweep(self, relative, query):
        self.send_json(self.board.sweep(relative))

    def post_stop(self, relative, query):
        self.send_json(self.board.stop(relative))

    def post_report(self, query):
        body = self.parsed()
        records = body.get("records")
        valid = isinstance(records, list) and all(isinstance(item, str) and item for item in records)
        if not valid or not records:
            raise BadRequest("records must be a non empty list of record paths")
        out = body.get("out")
        if out is not None and not isinstance(out, str):
            raise BadRequest("out must be a path or null")
        if out:
            inside(self.board.root, out)
        for relative in records:
            self.board.resolve(checked(relative))
        job = uuid.uuid4().hex[:12]
        with self.lock:
            self.jobs[job] = {"status": "running", "path": None, "error": None}
        worker = threading.Thread(target=run_report, args=(self.board, self.jobs, self.lock, job, records, out),
                                  name=f"kalfa-report-{job}", daemon=True)
        worker.start()
        self.send_json({"job": job}, plain=True)

    def get_report(self, job, query):
        with self.lock:
            found = self.jobs.get(job)
            found = None if found is None else dict(found)
        if found is None:
            raise NotFound(f"no report job {job}")
        self.send_json(found, plain=True)

    def get_live(self, query):
        every = number(query, "every", float)
        watcher = self.watcher
        own = every is not None and every != watcher.every
        if own:
            watcher = Watcher(self.board, max(every, SHORTEST))
            with self.lock:
                self.live.add(watcher)
        try:
            watcher.start()
            self.stream(watcher)
        finally:
            self.close_connection = True
            if own:
                watcher.stop()
                with self.lock:
                    self.live.discard(watcher)

    def stream(self, watcher):
        version = watcher.version
        self.answered = True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            self.emit(": connected\n\n")
            while not watcher.stopping:
                if watcher.wait(version, timeout=KEEP_ALIVE) is None:
                    if not watcher.stopping:
                        self.emit(": keep-alive\n\n")
                    continue
                version, changed = watcher.since(version)
                self.emit(f"event: change\ndata: {json.dumps({'version': version, 'changed': changed})}\n\n")
        except OSError:
            return

    def emit(self, text):
        self.wfile.write(text.encode("utf-8"))
        self.wfile.flush()

    def read_body(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            raise BadRequest("Content-Length must be an integer") from None
        if length < 0 or length > BODY_LIMIT:
            raise BadRequest(f"the request body must hold at most {BODY_LIMIT} bytes")
        return self.rfile.read(length) if length else b""

    def parsed(self):
        if not self.body.strip():
            return {}
        try:
            body = json.loads(self.body)
        except ValueError:
            raise BadRequest("the request body is not JSON") from None
        if not isinstance(body, dict):
            raise BadRequest("the request body must be a JSON object")
        return body

    def local(self):
        return all(host_allowed(value, self.hosts) for value in self.headers.get_all("Host") or [])

    def foreign(self):
        site = self.headers.get("Sec-Fetch-Site")
        if site is not None and site.strip().lower() not in FETCH_SITES:
            return True
        origin = self.headers.get("Origin")
        if origin is None:
            return False
        host = self.headers.get("Host") or ""
        return not host or urlsplit(origin).netloc.lower() != host.strip().lower()

    def send_json(self, value, status=200, plain=False, headers=None):
        body = json.dumps(value if plain else jsonable(value), ensure_ascii=False, default=str).encode("utf-8")
        self.send_body(status, body, "application/json", headers=headers)

    def send_body(self, status, body, kind, cache="no-store", headers=None):
        self.answered = True
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def fail(self, status, message):
        if self.answered:
            LOGGER.warning("%s failed after its answer began: %s", self.path, message)
            self.close_connection = True
            return
        try:
            self.send_json({"error": message}, status, plain=True)
        except OSError:
            self.close_connection = True


class BoardServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def server_bind(self):
        TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]

    def server_close(self):
        super().server_close()
        handler = self.RequestHandlerClass
        handler.watcher.stop()
        with handler.lock:
            watchers = list(handler.live)
        for watcher in watchers:
            watcher.stop()


def make_server(root, host=HOST, port=PORT):
    board = Board(root)
    attributes = {"board": board, "watcher": Watcher(board), "jobs": {}, "live": set(), "lock": threading.Lock(),
                  "hosts": names_of(host)}
    handler = type("BoardHandler", (Handler,), attributes)
    try:
        return BoardServer((host, int(port)), handler)
    except OSError as error:
        raise KalfaError(f"the board cannot listen on {host}:{port}: {error.strerror or error}") from error


def shown_host(host):
    return "127.0.0.1" if host in ("", "0.0.0.0") else host


def serve(root, host=HOST, port=PORT, open_browser=False):
    server = make_server(root, host, port)
    url = f"http://{shown_host(host)}:{server.server_address[1]}/"
    print(f"kalfa board on {url} over {server.RequestHandlerClass.board.root} (ctrl-c stops it)", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
