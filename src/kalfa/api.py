from dataclasses import dataclass, field
from pathlib import Path

import pandas

from . import summary as summarizing
from .core import check as checking
from .core import run as running
from .core.run import Monitor
from .records import reader
from .records import stats as statistics
from .records.compare import compare as compare_records, config_of


def call(use, **params):
    return {"use": use, **params}


@dataclass
class Rule:
    when: object = None
    after: str | None = None
    once: bool = True
    set: dict = field(default_factory=dict)
    stop: bool = False
    run: object = None

    def to_mapping(self):
        out = {}
        if self.when is not None:
            out["when"] = self.when
        if self.after is not None:
            out["after"] = self.after
        if not self.once:
            out["once"] = False
        if self.set:
            out["set"] = dict(self.set)
        if self.stop:
            out["stop"] = True
        if self.run is not None:
            out["run"] = self.run
        return out


@dataclass
class Train:
    epochs: int | None = None
    updates: dict | None = None
    step: object = None
    predicts: str | None = None
    evaluate: list | None = None
    checkpoint: dict | None = None
    report: str | None = None
    track: list | None = None
    log: list | None = None

    def to_mapping(self):
        return {key: value for key, value in self.__dict__.items() if value is not None}


def section_mapping(key, value):
    converter = getattr(value, "to_mapping", None)
    if callable(converter):
        return converter()
    if key == "rules" and isinstance(value, dict):
        return {name: section_mapping(name, rule) for name, rule in value.items()}
    return value


class Run:
    def __init__(self, **sections):
        self.sections = sections

    def to_mapping(self):
        return {key: section_mapping(key, value) for key, value in self.sections.items()}

    def check(self, sets=(), params=()):
        return checking.check(mappings=[self.to_mapping()], sets=running.pairs(sets),
                              params=running.pairs(params, "param"))

    def start(self, record=None, sets=(), params=(), monitor=None):
        return running.run(mappings=[self.to_mapping()], sets=sets, params=params, record=record,
                           monitor=monitor_of(monitor))


def sources(source):
    if isinstance(source, dict):
        return (), [source]
    if isinstance(source, (list, tuple)):
        return tuple(str(item) for item in source), []
    return (str(source),), []


def check(source, sets=(), params=()):
    paths, mappings = sources(source)
    return checking.check(paths=paths, sets=running.pairs(sets), params=running.pairs(params, "param"),
                          mappings=mappings)


def monitor_of(monitor):
    if monitor is True:
        return Monitor()
    if monitor is False:
        return Monitor(quiet=True)
    return monitor


def run(source, sets=(), params=(), record=None, monitor=None):
    paths, mappings = sources(source)
    return running.run(paths=paths, sets=sets, params=params, mappings=mappings, record=record,
                       monitor=monitor_of(monitor))


@dataclass
class Prediction:
    path: str
    table: pandas.DataFrame


def predict(directory, data=None, model=None, which=None, device=None):
    path, table = running.predict_record(str(directory), data=data, model=model, which=which, device=device)
    return Prediction(str(path), table)


class Record:
    def __init__(self, directory):
        self.directory = str(directory)

    @property
    def manifest(self):
        return reader.read_manifest(self.directory)

    @property
    def plan(self):
        return reader.read_plan(self.directory)

    @property
    def history(self):
        return pandas.DataFrame(reader.read_history(self.directory))

    @property
    def events(self):
        return reader.read_events(self.directory)

    @property
    def best(self):
        return reader.best_of(self.directory)

    def predict(self, data=None, model=None, which=None, device=None):
        return predict(self.directory, data=data, model=model, which=which, device=device).table

    def generate(self, n=None, which=None, device=None, out=None):
        from .commands.generate import generate_record

        return generate_record(self.directory, n=n, which=which, device=device, out=out)

    def plots(self, only=None, which=None, device=None):
        from .commands.plots import draw

        return draw(self.directory, only=only, which=which, device=device)

    def export(self, format="state_dict", model=None, which=None, out=None, **options):
        from .commands.export import export_record

        return export_record(self.directory, format=format, model=model, which=which, out=out, **options)

    def stop(self, by="api"):
        from .commands.stop import stop_record

        return stop_record(self.directory, by=by)

    def resume(self, sets=(), params=()):
        return running.resume(self.directory, sets=sets, params=params)

    @property
    def summary(self):
        return reader.summary_of(self.directory)

    @property
    def state(self):
        return reader.state_of(self.manifest)

    @property
    def progress(self):
        return reader.progress_of(self.directory)

    @property
    def updates(self):
        return pandas.DataFrame(reader.read_updates(self.directory))

    @property
    def files(self):
        return reader.files_of(self.directory)

    @property
    def model_summary(self):
        return reader.model_summary_of(self.directory)

    @property
    def data_report(self):
        return reader.data_report_of(self.directory)

    @property
    def config(self):
        return config_of(self.directory)

    @property
    def fields(self):
        return reader.fields_of(self.directory)

    @property
    def predictions(self):
        return statistics.read_predictions(self.directory)

    def series(self, keys=None, limit=None):
        return pandas.DataFrame(reader.series_of(self.directory, keys=keys_of(keys), limit=limit))

    def steps(self, keys=None, limit=None):
        return pandas.DataFrame(reader.updates_of(self.directory, keys=keys_of(keys), limit=limit))

    def patches(self):
        return reader.events_of(self.directory, kinds=["patch"])

    def stats(self, where=None, **options):
        return statistics.prediction_stats(self.directory, plan=self.plan, where=where, **options)

    def compare(self, *others, keys=None):
        directories = [self.directory] + [directory_of(other) for other in others]
        return compare_records(directories, keys=keys_of(keys))


def keys_of(keys):
    if keys is None:
        return None
    return [keys] if isinstance(keys, str) else list(keys)


def directory_of(record):
    return record.directory if isinstance(record, Record) else str(record)


def open(directory):
    return Record(directory)


def records(root):
    return [Record(directory) for directory in summarizing.under(root)]


def summarize(paths, by=None, mode=None, top=None):
    return summarizing.summarize(paths, by=by, mode=mode, top=top)


def generate(directory, n=None, which=None, device=None, out=None):
    from .commands.generate import generate_record

    return generate_record(str(directory), n=n, which=which, device=device, out=out)


def export(directory, format="state_dict", model=None, which=None, out=None, **options):
    from .commands.export import export_record

    return export_record(str(directory), format=format, model=model, which=which, out=out, **options)


def stop(directories, by="api"):
    from .commands.stop import stop_many

    return stop_many([str(item) for item in ([directories] if isinstance(directories, (str, Path)) else directories)],
                     by=by)


def sweep(source, sets=(), params=(), record=None, jobs=1, repair=False, monitor=None):
    from .commands import sweep as sweeping

    paths, mappings = sources(source)
    plan = sweeping.plan_of(paths=paths, sets=sets, params=params, mappings=mappings, record=record)
    return sweeping.run_local(plan, monitor=monitor, jobs=jobs, repair=repair, log=lambda text: None)


def report(paths, out=None, by=None, mode=None, top=None, points=None):
    from .commands.report import write_report

    return write_report(paths, out=out, by=by, mode=mode, top=top, points=points)
