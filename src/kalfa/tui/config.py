import json

from textual.containers import Horizontal, Vertical
from textual.widgets import Input, Static

from kalfa.cli import record_configs
from kalfa.core.check import check
from kalfa.core.errors import KalfaError
from kalfa.tui.jobs import OutputLog, cli_argv, pairs_of, words_of


FIELDS = (("config", "config-path", "YAML files, later ones override, or a record"),
          ("--set", "config-sets", "PATH=VALUE pairs separated by spaces"),
          ("-p", "config-params", "NAME=VALUE pairs separated by spaces"))


class ConfigView(Vertical):
    BINDINGS = [("c", "checked(False)", "check"), ("p", "checked(True)", "plan"), ("x", "start('run')", "run"),
                ("w", "start('sweep')", "sweep")]

    def compose(self):
        config = str(self.app.config or "")
        for label, name, hint in FIELDS:
            with Horizontal(classes="field"):
                yield Static(label, classes="label", markup=False)
                yield Input(value=config if name == "config-path" else "", placeholder=hint, id=name)
        yield OutputLog(id="config-output")

    def say(self, text):
        self.query_one("#config-output", OutputLog).say(text)

    def fields(self):
        paths = words_of(self.query_one("#config-path", Input).value)
        if not paths:
            raise KalfaError("write a config path first")
        sets = pairs_of(self.query_one("#config-sets", Input).value, "--set")
        return paths, sets, pairs_of(self.query_one("#config-params", Input).value, "-p")

    def action_checked(self, plan=False):
        try:
            paths, sets, params = self.fields()
            self.say(f"check {' '.join(paths)}")
            found = check(paths=record_configs(paths), sets=sets, params=params)
        except Exception as exc:
            self.say(f"error: {exc}" if isinstance(exc, KalfaError) else f"error: {type(exc).__name__}: {exc}")
            return
        self.say(found.text())
        if plan and found.ok:
            self.say(json.dumps(found.plan, indent=1, default=str))

    def action_start(self, command):
        try:
            paths, sets, params = self.fields()
        except KalfaError as exc:
            self.say(f"error: {exc}")
            return
        self.app.launch(cli_argv(command, *paths, sets=sets, params=params), command, "#config-output")

    def on_input_submitted(self, event):
        self.query_one("#config-output", OutputLog).focus()
        self.action_checked()
