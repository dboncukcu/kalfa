import threading

from kalfa.core.errors import KalfaError


try:
    from textual.app import App
    from textual.widgets import Footer, Header, TabbedContent, TabPane

    from kalfa.tui.config import ConfigView
    from kalfa.tui.guide import GuideView
    from kalfa.tui.jobs import JobOutput, OutputLog, follow, spawn
    from kalfa.tui.legos import LegosView
    from kalfa.tui.records import RecordsView
except ImportError as missing:
    App = None
    TEXTUAL_ERROR = missing
else:
    TEXTUAL_ERROR = None


PRIMARY = {"records": "#records-table", "legos": "#legos-tree", "config": "#config-output", "guide": "#guide-sections"}
STYLE = """
#tabs, #tabs > ContentSwitcher, TabPane, #records-view, #legos-view, #config-view, #legos-tree { height: 1fr; }
#records-table { width: 3fr; height: 1fr; }
#records-side, #legos-side { width: 2fr; }
#legos-scroll { width: 3fr; }
#legos-tree { overflow-x: hidden; }
#records-info { height: 2fr; }
#record-detail, #record-sparks, #legos-page, .field { height: auto; padding: 0 1; }
.spark { height: 2; margin-bottom: 1; }
#records-log, #config-output { height: 1fr; border-top: solid $primary; }
.label { width: 10; padding: 1 1 0 1; }
"""


def require_textual():
    if App is None:
        raise KalfaError(f"the terminal interface needs the textual package (uv sync installs the dependencies of "
                         f"kalfa): {TEXTUAL_ERROR}")


class KalfaApp(App if App is not None else object):
    TITLE = "kalfa"
    CSS = STYLE
    AUTO_FOCUS = "#records-table"
    BINDINGS = [("1", "tab('records')", "records"), ("2", "tab('legos')", "legos"), ("3", "tab('config')", "config"),
                ("4", "tab('guide')", "guide"), ("escape", "tab", "leave field"), ("q", "quit", "quit")]

    def __init__(self, root=".", config=None):
        require_textual()
        super().__init__()
        self.root = str(root)
        self.config = config
        self.board = None
        self.closing = threading.Event()
        self.sub_title = self.root

    def compose(self):
        yield Header()
        with TabbedContent(initial="records", id="tabs"):
            with TabPane("1 records", id="records"):
                yield RecordsView(id="records-view")
            with TabPane("2 legos", id="legos"):
                yield LegosView(id="legos-view")
            with TabPane("3 config", id="config"):
                yield ConfigView(id="config-view")
            with TabPane("4 guide", id="guide"):
                yield GuideView(id="guide-view")
        yield Footer()

    def action_tab(self, name=None):
        tabs = self.query_one("#tabs", TabbedContent)
        tabs.active = name = name or tabs.active or "records"
        self.query_one(PRIMARY[name]).focus()

    def on_tabbed_content_tab_activated(self, event):
        focused, pane = self.focused, event.pane
        if pane is not None and pane.id in PRIMARY and (focused is None or pane not in focused.ancestors_with_self):
            self.query_one(PRIMARY[pane.id]).focus()

    def launch(self, argv, label, target, detach=True):
        output = self.query_one(target, OutputLog)
        try:
            process, path = spawn(argv, label, detach=detach)
        except OSError as exc:
            output.say(f"{label}: could not start: {exc}")
            return None
        output.say(f"{label}: pid {process.pid}, output in {path}")

        def emit(text):
            try:
                return output.post_message(JobOutput(text))
            except RuntimeError:
                return False

        threading.Thread(target=follow, args=(path, process, emit, self.closing), kwargs={"label": label},
                         daemon=True).start()
        return process

    def on_unmount(self):
        self.closing.set()
        if self.board is not None and self.board.poll() is None:
            self.board.terminate()
            self.board.wait(timeout=5)
