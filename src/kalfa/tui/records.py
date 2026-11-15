from pathlib import Path

from rich.text import Text
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import DataTable, Input, Sparkline, Static

from kalfa.commands.stop import STOPPABLE, state_in, stop_record
from kalfa.core.errors import KalfaError
from kalfa.records.reader import Cache, failure_of, log_of, records_under, series_of, summary_with
from kalfa.records.sweep import best_point, id_key, read_sweep, sweep_points
from kalfa.tui.jobs import OutputLog, cli_argv


COLUMNS = ("name", "status", "turn", "best")
ORDERS = ("started", "name", "status", "turn", "best")
SPARKS = 6
SPARK_POINTS = 400
REFRESH = 10.0
LOG_LINES = 40
TRACE_LINES = 12
SINGULAR = {"batches": "batch", "updates": "update"}


def number_text(value):
    return "-" if value is None else f"{value:.4g}" if isinstance(value, float) else str(value)


def seconds_text(value):
    value = float(value)
    if value < 60:
        return f"{value:.0f}s"
    if value < 3600:
        return f"{int(value // 60)}m {int(value % 60):02d}s"
    return f"{int(value // 3600)}h {int(value % 3600 // 60):02d}m"


def left_text(eta):
    if isinstance(eta, bool) or not isinstance(eta, (int, float)):
        return ""
    return "under a minute left" if eta < 60 else f"about {seconds_text(eta)} left"


def within_text(within):
    if not isinstance(within, dict):
        return ""
    phase = "" if within.get("phase") in (None, "train") else f"{within['phase']} "
    unit = str(within.get("unit") or "batches")
    done, total = within.get("done"), within.get("total")
    if isinstance(done, int) and isinstance(total, int) and total > 0:
        return f"{phase}{SINGULAR.get(unit, unit)} {done} of {total}"
    if isinstance(done, int) and done > 0:
        return f"{phase}{done} {unit}"
    return phase.strip()


def nonfinite_text(found):
    if not isinstance(found, dict) or not found.get("series"):
        return ""
    series = list(found["series"])
    more = f" and {len(series) - 1} more" if len(series) > 1 else ""
    return f"{series[0]}{more} is {found.get('value') or 'not finite'} from turn {found.get('turn')}"


def turn_text(summary):
    turn, total = summary.get("turn"), summary.get("total")
    text = "-" if turn is None else f"{turn}/{total}" if total else str(turn)
    fraction = summary.get("fraction")
    if summary.get("within") and isinstance(fraction, float):
        text += f" ({fraction:.0%})"
    return text


def best_text(best):
    if not isinstance(best, dict) or not best.get("series"):
        return ""
    text = f"{best['series']} {number_text(best.get('value'))}"
    return f"{text} @{best['turn']}" if best.get("turn") is not None else text


def text_of(title, rows, *blocks):
    lines = [title] + [f"  {label.ljust(9)} {value}" for label, value in rows if value]
    for name, values in blocks:
        if isinstance(values, dict) and values:
            width = max(len(str(key)) for key in values)
            lines += ["", name] + [f"  {str(key).ljust(width)}  {number_text(value)}" for key, value in values.items()]
    return "\n".join(lines)


def summary_text(summary):
    best = summary.get("best") if isinstance(summary.get("best"), dict) else {}
    seconds = number_text(float(summary.get("seconds") or 0.0))
    now = ", ".join(part for part in (within_text(summary.get("within")), left_text(summary.get("eta"))) if part)
    rows = [("status", f"{summary.get('status')}, turn {turn_text(summary)} {summary.get('unit') or 'turns'}"),
            ("now", now),
            ("started", summary.get("started") and f"{summary['started']}, {seconds} s"),
            ("device", " on ".join(str(part) for part in (summary.get("device"), summary.get("host")) if part)),
            ("best", best.get("series") and f"{best_text(best)} ({best.get('mode') or 'min'})"),
            ("failure", summary.get("failure")),
            ("warning", nonfinite_text(summary.get("nonfinite"))),
            ("stopped", summary.get("stop_reason")),
            ("rules", ", ".join(str(name) for name in summary.get("rules") or ()))]
    title = str(summary.get("name") or summary.get("path") or "record")
    return text_of(title, rows, ("params", summary.get("params")), ("last", summary.get("last")))


def trace_text(directory, count=TRACE_LINES):
    failure = failure_of(directory)
    trace = failure.get("traceback") if isinstance(failure, dict) else None
    if not isinstance(trace, str) or not trace.strip():
        return ""
    lines = trace.rstrip().splitlines()[-count:]
    return "\n\ntraceback (its end; log.txt and the board have all of it)\n" + "\n".join(f"  {line}" for line in lines)


def sweep_info(directory, cache=None):
    sweep = read_sweep(directory)
    sweep = sweep if isinstance(sweep, dict) else {}
    objective = sweep.get("objective") if isinstance(sweep.get("objective"), dict) else {}
    points = sweep_points(directory, cache)
    states = [point.get("status") for point in points]
    planned = sweep.get("points")
    return {"sweep": sweep, "objective": objective, "points": points, "best": best_point(points, objective),
            "total": planned if isinstance(planned, int) else len(points), "done": states.count("done"),
            "failed": states.count("failed"), "running": sum(state in STOPPABLE for state in states)}


def sweep_text(name, info):
    sweep, objective, best = info["sweep"], info["objective"], info["best"] or {}
    values = " ".join(f"{key}={number_text(value)}" for key, value in (best.get("values") or {}).items())
    rows = [("points", f"{len(info['points'])} of {info['total']}: {info['done']} done, {info['running']} running, "
                       f"{info['failed']} failed"),
            ("objective", objective.get("watch") and f"{objective['watch']} ({objective.get('mode') or 'min'}, "
                                                     f"at {objective.get('at') or 'best'})"),
            ("best", best and f"point {best['id']}: {number_text(best.get('objective'))}  {values}".rstrip()),
            ("keys", "enter lists the points, backspace comes back")]
    return text_of(f"{name} (sweep)", rows, ("space", sweep.get("space")))


def spark_series(directory, limit=SPARKS):
    found = []
    for key, values in series_of(directory, limit=SPARK_POINTS).items():
        finite = [float(value) for value in values if value is not None]
        if key != "turn" and finite:
            found.append((key, finite))
    return found[:limit]


def place_of(root, entry):
    path = entry.get("path") or "."
    return (Path(root).resolve().name if path == "." else path), Path(root, path)


def read_row(root, entry, cache=None):
    name, directory = place_of(root, entry)
    try:
        if entry.get("kind") == "sweep":
            info = sweep_info(directory, cache)
            best = info["best"]
            watch = info["objective"].get("watch") or "objective"
            cells = (name, "running" if info["running"] else "sweep", f"{info['done']}/{info['total']}",
                     f"#{best['id']} {watch} {number_text(best['objective'])}" if best else "")
            return {"cells": cells, "turn": info["done"], "best": best["objective"] if best else None, "words": name}
        summary = summary_with(cache, directory, entry.get("manifest"))
    except (OSError, ValueError):
        return {"cells": (name, "unreadable", "", ""), "turn": 0, "best": None, "words": name}
    params = " ".join(f"{key}={value}" for key, value in (summary.get("params") or {}).items())
    best = summary.get("best") if isinstance(summary.get("best"), dict) else {}
    cells = (name, str(summary.get("status")), turn_text(summary), best_text(summary.get("best")))
    return {"cells": cells, "turn": summary.get("turn") or 0, "best": best.get("value"), "words": f"{name} {params}"}


def row_cells(root, entry, cache=None):
    return read_row(root, entry, cache)["cells"]


def record_rows(root, cache=None):
    return [{**read_row(root, entry, cache), "key": entry["path"], "entry": entry} for entry in records_under(root)]


def point_rows(root, scope, cache=None):
    info = sweep_info(Path(root, scope), cache)
    watch = info["objective"].get("watch") or "objective"
    rows = []
    for point in sorted(info["points"], key=id_key):
        values = " ".join(f"{key}={number_text(value)}" for key, value in (point.get("values") or {}).items())
        summary = point.get("summary") if isinstance(point.get("summary"), dict) else {}
        key = point["path"] if scope in ("", ".") else f"{scope}/{point['path']}"
        valued = point.get("objective") is not None
        cells = (f"{point['path']}  {values}".rstrip(), str(point.get("status")), turn_text(summary),
                 f"{watch} {number_text(point['objective'])}" if valued else "")
        rows.append({"cells": cells, "turn": point.get("turn") or 0, "best": point.get("objective"),
                     "words": f"{point['path']} {values}", "key": key, "entry": {"path": key, "kind": "record"}})
    return rows


def matches(row, words):
    text = f"{row['words']} {row['cells'][1]}".lower()
    return all(word in text for word in words)


def ordered(rows, order, reverse=False):
    if order == "name":
        found = sorted(rows, key=lambda row: str(row["cells"][0]).lower())
    elif order == "status":
        found = sorted(rows, key=lambda row: str(row["cells"][1]))
    elif order in ("turn", "best"):
        valued = sorted((row for row in rows if isinstance(row[order], (int, float))), key=lambda row: row[order])
        return (valued[::-1] if reverse else valued) + [row for row in rows if not isinstance(row[order], (int, float))]
    else:
        found = list(rows)
    return found[::-1] if reverse else found


def detail_of(root, entry, cache=None):
    name, directory = place_of(root, entry)
    try:
        if entry.get("kind") == "sweep":
            return sweep_text(name, sweep_info(directory, cache)), []
        summary = summary_with(cache, directory, entry.get("manifest"))
        trace = trace_text(directory) if summary.get("status") == "failed" else ""
        return summary_text(summary) + trace, spark_series(directory)
    except (OSError, ValueError) as exc:
        return f"{name}: not readable now ({exc})", []


class Detail(Static):
    @property
    def renderable(self):
        return self.content


class RecordsView(Horizontal):
    BINDINGS = [("r", "reload", "refresh"), ("slash", "filter", "filter"), ("t", "order", "sort"),
                ("backspace", "back", "back"), ("p", "launch('predict')", "predict"),
                ("g", "launch('generate')", "generate"), ("o", "launch('plots')", "plots"),
                ("u", "resume", "resume"), ("l", "log", "log"), ("s", "stop", "stop"), ("b", "board", "board")]

    def __init__(self, **options):
        super().__init__(**options)
        self.rows = []
        self.entries = {}
        self.order = []
        self.scope = None
        self.sorting = ORDERS[0]
        self.reverse = False
        self.cache = Cache()

    def compose(self):
        with Vertical(id="records-main"):
            yield Input(placeholder="words to filter by name, status or param=value; / comes here, enter goes back",
                        id="records-filter")
            yield Static(id="records-order", markup=False)
            yield DataTable(id="records-table", cursor_type="row", zebra_stripes=True)
        with Vertical(id="records-side"):
            with VerticalScroll(id="records-info"):
                yield Detail(id="record-detail", markup=False)
                yield Vertical(id="record-sparks")
            yield OutputLog(id="records-log")

    def on_mount(self):
        table = self.query_one("#records-table", DataTable)
        for column in COLUMNS:
            table.add_column(column, key=column)
        self.action_reload()
        self.set_interval(REFRESH, self.reload_aside)

    def say(self, text):
        self.query_one("#records-log", OutputLog).say(text)

    def selected(self):
        row = self.query_one("#records-table", DataTable).cursor_row
        return self.order[row] if 0 <= row < len(self.order) else None

    def gather(self, root, scope):
        return record_rows(root, self.cache) if scope is None else point_rows(root, scope, self.cache)

    def action_reload(self):
        self.fill(self.gather(self.app.root, self.scope), self.scope)

    def reload_aside(self):
        root, scope = self.app.root, self.scope

        def collect():
            try:
                rows = self.gather(root, scope)
                self.app.call_from_thread(self.fill, rows, scope)
            except (OSError, ValueError, KalfaError, RuntimeError):
                return

        return self.run_worker(collect, thread=True, exclusive=True, group="records", exit_on_error=False)

    def fill(self, rows, scope=None, chosen=None):
        if scope != self.scope:
            return
        chosen = chosen if chosen is not None else self.selected()
        self.rows = rows
        self.entries = {row["key"]: row["entry"] for row in rows}
        self.draw(chosen)

    def draw(self, chosen=None):
        words = self.query_one("#records-filter", Input).value.lower().split()
        shown = ordered([row for row in self.rows if matches(row, words)], self.sorting, self.reverse)
        self.order = [row["key"] for row in shown]
        table = self.query_one("#records-table", DataTable)
        table.clear()
        for row in shown:
            table.add_row(*(Text(value) for value in row["cells"]), key=row["key"])
        if chosen in self.order:
            table.move_cursor(row=self.order.index(chosen))
        what = "records" if self.scope is None else f"points of {self.scope} (backspace goes back)"
        way = f"by {self.sorting}{', reversed' if self.reverse else ''}"
        self.query_one("#records-order", Static).update(f"{len(shown)} of {len(self.rows)} {what}, {way}")
        if not self.order:
            self.show(None)

    def on_input_changed(self, event):
        if event.input.id == "records-filter":
            self.draw(self.selected())

    def on_input_submitted(self, event):
        if event.input.id == "records-filter":
            self.query_one("#records-table", DataTable).focus()

    def action_filter(self):
        self.query_one("#records-filter", Input).focus()

    def action_order(self):
        self.sorting = ORDERS[(ORDERS.index(self.sorting) + 1) % len(ORDERS)]
        self.reverse = False
        self.draw(self.selected())

    def on_data_table_header_selected(self, event):
        column = event.column_key.value
        if column not in ORDERS:
            return
        self.reverse = not self.reverse if column == self.sorting else False
        self.sorting = column
        self.draw(self.selected())

    def on_data_table_row_selected(self, event):
        key = event.row_key.value
        entry = self.entries.get(key)
        if entry is not None and entry.get("kind") == "sweep":
            self.enter(key)

    def enter(self, scope, chosen=None):
        field = self.query_one("#records-filter", Input)
        with field.prevent(Input.Changed):
            field.value = ""
        self.scope = scope
        self.app.sub_title = self.app.root if scope is None else str(Path(self.app.root, scope))
        try:
            rows = self.gather(self.app.root, scope)
        except (OSError, ValueError) as exc:
            self.say(f"{scope}: not readable now ({exc})")
            rows = []
        self.fill(rows, scope, chosen)
        if chosen is None:
            self.query_one("#records-table", DataTable).move_cursor(row=0)

    def action_back(self):
        if self.scope is not None:
            self.enter(None, self.scope)

    def on_data_table_row_highlighted(self, event):
        self.show(event.row_key.value)

    def show(self, key):
        detail = self.query_one("#record-detail", Detail)
        sparks = self.query_one("#record-sparks", Vertical)
        sparks.remove_children()
        entry = self.entries.get(key)
        if entry is None:
            where = self.app.root if self.scope is None else Path(self.app.root, self.scope)
            detail.update("" if self.entries else f"no records under {where}")
            return
        text, series = detail_of(self.app.root, entry, self.cache)
        detail.update(text)
        widgets = [widget for name, values in series
                   for widget in (Static(f"{name}  {number_text(values[-1])}", markup=False),
                                  Sparkline(values, classes="spark"))]
        if widgets:
            sparks.mount(*widgets)

    def target(self, action, record_only=True):
        key = self.selected()
        entry = self.entries.get(key)
        if entry is None:
            self.say(f"{action}: no record is selected")
            return None
        directory = Path(self.app.root, key)
        if record_only and entry.get("kind") == "sweep":
            self.say(f"{action} works on a record and {key} is a sweep root; enter lists its points")
            return None
        return str(directory)

    def action_launch(self, command):
        record = self.target(command)
        if record:
            self.app.launch(cli_argv(command, record), command, "#records-log")

    def action_resume(self):
        record = self.target("resume")
        if record is None:
            return
        if state_in(record) == "running":
            self.say(f"resume: {record} is running; s asks it to stop after its turn")
            return
        self.app.launch(cli_argv("run", "--from", record), "resume", "#records-log")

    def action_log(self):
        record = self.target("log")
        if record is None:
            return
        lines = log_of(record, LOG_LINES)
        if not lines:
            self.say(f"log: {record} has no log.txt")
            return
        self.say(f"log: the last {len(lines)} lines of {record}/log.txt")
        for line in lines:
            self.say(line)

    def action_stop(self):
        record = self.target("stop", record_only=False)
        if record is None:
            return
        try:
            written = stop_record(record, by="tui")
        except KalfaError as exc:
            self.say(f"stop: {exc}")
            return
        self.say(f"stop asked: {', '.join(written)}" if written else f"stop: nothing runs under {record}")

    def action_board(self):
        board = self.app.board
        if board is not None and board.poll() is None:
            self.say(f"board: already runs as pid {board.pid}")
        else:
            self.app.board = self.app.launch(cli_argv("board", self.app.root, "--open"), "board", "#records-log", False)
