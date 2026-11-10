from pathlib import Path

from rich.text import Text
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import DataTable, Sparkline, Static

from kalfa.commands.stop import STOPPABLE, stop_record
from kalfa.core.errors import KalfaError
from kalfa.records.reader import records_under, series_of, summary_of
from kalfa.records.sweep import best_point, read_sweep, sweep_points
from kalfa.tui.jobs import OutputLog, cli_argv


COLUMNS = ("name", "status", "turn", "best")
SPARKS = 6
SPARK_POINTS = 400
REFRESH = 10.0


def number_text(value):
    return "-" if value is None else f"{value:.4g}" if isinstance(value, float) else str(value)


def turn_text(summary):
    turn, total = summary.get("turn"), summary.get("total")
    return "-" if turn is None else f"{turn}/{total}" if total else str(turn)


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
    rows = [("status", f"{summary.get('status')}, turn {turn_text(summary)} {summary.get('unit') or 'turns'}"),
            ("started", summary.get("started") and f"{summary['started']}, {seconds} s"),
            ("device", " on ".join(str(part) for part in (summary.get("device"), summary.get("host")) if part)),
            ("best", best.get("series") and f"{best_text(best)} ({best.get('mode') or 'min'})"),
            ("failure", summary.get("failure")),
            ("stopped", summary.get("stop_reason")),
            ("rules", ", ".join(str(name) for name in summary.get("rules") or ()))]
    title = str(summary.get("name") or summary.get("path") or "record")
    return text_of(title, rows, ("params", summary.get("params")), ("last", summary.get("last")))


def sweep_info(directory):
    sweep = read_sweep(directory)
    sweep = sweep if isinstance(sweep, dict) else {}
    objective = sweep.get("objective") if isinstance(sweep.get("objective"), dict) else {}
    points = sweep_points(directory)
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
            ("best", best and f"point {best['id']}: {number_text(best.get('objective'))}  {values}".rstrip())]
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


def row_cells(root, entry):
    name, directory = place_of(root, entry)
    try:
        if entry.get("kind") == "sweep":
            info = sweep_info(directory)
            best = info["best"]
            watch = info["objective"].get("watch") or "objective"
            return (name, "running" if info["running"] else "sweep", f"{info['done']}/{info['total']}",
                    f"#{best['id']} {watch} {number_text(best['objective'])}" if best else "")
        summary = summary_of(directory)
    except (OSError, ValueError):
        return (name, "unreadable", "", "")
    return (name, str(summary.get("status")), turn_text(summary), best_text(summary.get("best")))


def detail_of(root, entry):
    name, directory = place_of(root, entry)
    try:
        if entry.get("kind") == "sweep":
            return sweep_text(name, sweep_info(directory)), []
        return summary_text(summary_of(directory)), spark_series(directory)
    except (OSError, ValueError) as exc:
        return f"{name}: not readable now ({exc})", []


class Detail(Static):
    @property
    def renderable(self):
        return self.content


class RecordsView(Horizontal):
    BINDINGS = [("r", "reload", "refresh"), ("p", "launch('predict')", "predict"),
                ("g", "launch('generate')", "generate"), ("o", "launch('plots')", "plots"), ("s", "stop", "stop"),
                ("b", "board", "board")]
    entries = {}
    order = ()

    def compose(self):
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
        self.set_interval(REFRESH, self.action_reload)

    def say(self, text):
        self.query_one("#records-log", OutputLog).say(text)

    def selected(self):
        row = self.query_one("#records-table", DataTable).cursor_row
        return self.order[row] if 0 <= row < len(self.order) else None

    def action_reload(self):
        root = self.app.root
        found = records_under(root)
        chosen = self.selected()
        self.entries = {entry["path"]: entry for entry in found}
        self.order = [entry["path"] for entry in found]
        table = self.query_one("#records-table", DataTable)
        table.clear()
        for key in self.order:
            table.add_row(*(Text(value) for value in row_cells(root, self.entries[key])), key=key)
        if chosen in self.order:
            table.move_cursor(row=self.order.index(chosen))
        if not self.order:
            self.show(None)

    def on_data_table_row_highlighted(self, event):
        self.show(event.row_key.value)

    def show(self, key):
        detail = self.query_one("#record-detail", Detail)
        sparks = self.query_one("#record-sparks", Vertical)
        sparks.remove_children()
        entry = self.entries.get(key)
        if entry is None:
            detail.update("" if self.entries else f"no records under {self.app.root}")
            return
        text, series = detail_of(self.app.root, entry)
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
            self.say(f"{action} works on a record and {key} is a sweep root; kalfa tui {directory} lists its points")
            return None
        return str(directory)

    def action_launch(self, command):
        record = self.target(command)
        if record:
            self.app.launch(cli_argv(command, record), command, "#records-log")

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
