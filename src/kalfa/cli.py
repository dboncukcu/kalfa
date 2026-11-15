import argparse
import json
import logging
import sys
import traceback
from pathlib import Path

from . import __version__
from . import summary
from .api import predict
from .core import run as running
from .core.check import check
from .core.errors import KalfaError, ProblemsError


def overrides(parser):
    parser.add_argument("--set", action="append", default=[], metavar="PATH=VALUE", help="override a leaf")
    parser.add_argument("-p", "--param", action="append", default=[], metavar="NAME=VALUE", help="set a param")


def training_commands(commands):
    checker = commands.add_parser("check", help="every problem of a config; nothing runs")
    checker.add_argument("config", nargs="+", metavar="CONFIG",
                         help="YAML files, later ones override, or a record directory")
    overrides(checker)
    checker.add_argument("--plan", action="store_true", help="print the plan as JSON after the problems")

    runner = commands.add_parser("run", help="train into the record directory, or continue a record")
    runner.add_argument("config", nargs="*", metavar="CONFIG", help="YAML files, later ones override")
    overrides(runner)
    runner.add_argument("--record", metavar="DIR", help="the record directory, over the config's record")
    runner.add_argument("--from", dest="from_record", metavar="RECORD",
                        help="continue this record from its last checkpoint into a new directory")
    runner.add_argument("--quiet", action="store_true", help="no progress line per turn")


def record_commands(commands):
    predictor = commands.add_parser("predict", help="predict with the weights of a record")
    predictor.add_argument("record", metavar="RECORD", help="the record directory")
    predictor.add_argument("--data", metavar="FILE", help="a .parquet or .csv file; without it the test set")
    predictor.add_argument("--model", metavar="NAME", help="the model to predict with, over train.predicts")
    predictor.add_argument("--which", choices=["best", "last", "final"], help="the weights to load")
    predictor.add_argument("--device", metavar="D", help="cpu, cuda, mps or auto")

    summarizer = commands.add_parser("summarize", help="a table over records and sweeps, best first")
    summarizer.add_argument("records", nargs="+", metavar="RECORD",
                            help="record directories, sweep roots, or a directory with records under it")
    summarizer.add_argument("--by", metavar="SERIES", help="the series to rank by, over the checkpoint watch")
    summarizer.add_argument("--mode", choices=["min", "max"], help="whether lower or higher is better")
    summarizer.add_argument("--top", type=int, metavar="K", help="only the best K rows")
    summarizer.add_argument("--out", metavar="DIR", help="write summary.csv and summary.md into this directory; with "
                                                        "--html the report file, or a directory for it")
    summarizer.add_argument("--markdown", action="store_true", help="print a markdown table instead of text")
    summarizer.add_argument("--columns", metavar="NAMES",
                            help="comma separated columns to print, or all; without it train and lr series are hidden")
    summarizer.add_argument("--html", action="store_true", help="write a single file report.html instead of a table")
    summarizer.add_argument("--points", metavar="IDS", help="with --html over a sweep root: the point ids to detail")

    generator = commands.add_parser("generate", help="draw samples from a record's generate section")
    generator.add_argument("record", metavar="RECORD", help="the record directory")
    generator.add_argument("--n", type=int, metavar="N", help="how many samples, over the sampler's n")
    generator.add_argument("--which", choices=["best", "last", "final"], help="the weights to load")
    generator.add_argument("--device", metavar="D", help="cpu, cuda, mps or auto")
    generator.add_argument("--out", metavar="DIR", help="write the samples here instead of <record>/samples")

    plotter = commands.add_parser("plots", help="draw the plots of a record again into <record>/plots")
    plotter.add_argument("record", metavar="RECORD", help="the record directory")
    plotter.add_argument("--only", metavar="NAMES", help="comma separated names from the plots section, over all")
    plotter.add_argument("--which", choices=["best", "last", "final"], help="the weights to load")
    plotter.add_argument("--device", metavar="D", help="cpu, cuda, mps or auto")

    exporter = commands.add_parser("export", help="export a model of a record")
    exporter.add_argument("record", metavar="RECORD", help="the record directory")
    exporter.add_argument("--format", choices=["state_dict", "onnx"], default="state_dict",
                          help="state_dict (default) or onnx")
    exporter.add_argument("--model", metavar="NAME", help="the model to export, over train.predicts")
    exporter.add_argument("--which", choices=["best", "last", "final"], help="the weights to load")
    exporter.add_argument("--out", metavar="PATH",
                          help="the output file, or a directory for it, over <record>/exports/<model>_<which>.<ext>")
    exporter.add_argument("--device", metavar="D", help="cpu, cuda, mps or auto")
    exporter.add_argument("--opset", type=int, default=17, help="onnx opset (default %(default)s)")
    exporter.add_argument("--static", action="store_true", help="onnx without a dynamic batch axis")


def sweep_commands(commands):
    sweeper = commands.add_parser("sweep", help="the points of the sweep section, locally or one per job")
    sweeper.add_argument("config", nargs="+", metavar="CONFIG", help="YAML files with a sweep section")
    overrides(sweeper)
    sweeper.add_argument("--record", metavar="ROOT", help="the sweep root, over sweep.record")
    sweeper.add_argument("--count", action="store_true", help="print the number of points and exit")
    sweeper.add_argument("--show", type=int, metavar="N", help="print the first N points and exit")
    sweeper.add_argument("--plan", action="store_true",
                         help="write sweep.json, points.jsonl, sweep.sh and sweep.sub under the root and exit")
    sweeper.add_argument("--point", type=int, metavar="N", help="run one point by id (for a job)")
    sweeper.add_argument("--repair", action="store_true", help="run only the failed or missing points")
    sweeper.add_argument("--jobs", type=int, default=1, metavar="K", help="points run in parallel locally")
    sweeper.add_argument("--quiet", action="store_true", help="no progress lines")

    stopper = commands.add_parser("stop", help="ask running records to stop after their turn")
    stopper.add_argument("records", nargs="+", metavar="RECORD", help="record directories or sweep roots")


def browsing_commands(commands):
    lister = commands.add_parser("legos", help="the short names: list, search or describe one")
    lister.add_argument("name", nargs="?", metavar="NAME|WORD", help="an exact name for its page, or a word to search")
    lister.add_argument("--slot", metavar="SLOT", help="only names of this slot (data, model, loss, metric, ...)")
    lister.add_argument("--docs", action="store_true", help="print the whole catalog as markdown")

    boarder = commands.add_parser("board", help="follow the records under a root in the browser")
    boarder.add_argument("root", nargs="?", default=".", metavar="ROOT", help="the directory holding the records")
    boarder.add_argument("--host", default="127.0.0.1", metavar="HOST")
    boarder.add_argument("--port", type=int, default=8765, metavar="PORT")
    boarder.add_argument("--open", action="store_true", help="open the page in the default browser")

    terminal = commands.add_parser("tui", help="the terminal interface over the records under a root")
    terminal.add_argument("root", nargs="?", default=".", metavar="ROOT", help="the directory holding the records")
    terminal.add_argument("--config", metavar="CONFIG", help="a config to load into the config tab")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="kalfa",
        description="YAML and Python front end for PyTorch training: a config names the data, the models, the "
                    "losses, the optimizers and the training; kalfa checks it and runs it into a record directory "
                    "that every other command reads.")
    parser.add_argument("--version", action="version", version=f"kalfa {__version__}")
    commands = parser.add_subparsers(dest="command", title="commands", metavar="<command>")
    for group in (training_commands, record_commands, sweep_commands, browsing_commands):
        group(commands)
    return parser


def layers(args):
    return running.pairs(args.set), running.pairs(args.param, "param")


def record_configs(paths):
    found = []
    for path in paths:
        resolved = Path(path) / "resolved.yaml"
        if Path(path).is_dir() and resolved.exists():
            running.record_root(path)
            found.append(str(resolved))
        else:
            found.append(str(path))
    return found


def cmd_check(args):
    sets, params = layers(args)
    checked = check(paths=record_configs(args.config), sets=sets, params=params)
    print(checked.text())
    if args.plan and checked.ok:
        print(json.dumps(checked.plan, indent=1, default=str))
    return 0 if checked.ok else 1


class StderrHandler(logging.StreamHandler):
    @property
    def stream(self):
        return sys.stderr

    @stream.setter
    def stream(self, value):
        pass


def configure_logging(quiet):
    logger = logging.getLogger("kalfa")
    if not any(isinstance(handler, StderrHandler) for handler in logger.handlers):
        handler = StderrHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.WARNING if quiet else logging.INFO)


def cmd_run(args):
    sets, params = layers(args)
    configure_logging(args.quiet)
    monitor = running.Monitor(quiet=args.quiet)
    if args.from_record:
        result = running.resume(args.from_record, sets=sets, params=params, target=args.record, monitor=monitor,
                                paths=record_configs(args.config))
    elif args.config:
        result = running.run(record_configs(args.config), sets=sets, params=params, record=args.record,
                             monitor=monitor)
    else:
        print("run takes a config, or --from RECORD", file=sys.stderr)
        return 2
    print(result.record)
    return 0


def cmd_predict(args):
    found = predict(args.record, data=args.data, model=args.model, which=args.which, device=args.device)
    print(found.path)
    return 0


def number(value):
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def best_line(found):
    best = found.best
    if not best or found.by is None or best.get("best") is None:
        return "best: none of the records has a value for " + (found.by or "the series")
    name = best.get("name", "?")
    turn = best.get("best_turn")
    where = f" at turn {number(turn)}" if turn is not None else ""
    return f"best: {name}  {found.by} {number(best.get('best'))}{where} ({found.mode})"


def cmd_summarize(args):
    if args.html:
        from .commands.report import write_report

        path = write_report(args.records, out=args.out, by=args.by, mode=args.mode, top=args.top, points=args.points)
        print(path)
        return 0
    found = summary.summarize(args.records, by=args.by, mode=args.mode, top=args.top)
    if args.markdown:
        print(summary.markdown_of(found, columns=args.columns or "all"))
    else:
        print(summary.text_of(found, columns=args.columns))
        print(best_line(found))
        if found.folds:
            print(summary.folds_text(found))
    if args.out:
        for path in summary.write(found, args.out):
            print(path)
    return 0


def cmd_generate(args):
    from .commands.generate import generate_record

    found = generate_record(args.record, n=args.n, which=args.which, device=args.device, out=args.out)
    for path in found.paths:
        print(path)
    return 0


def cmd_plots(args):
    from .commands.plots import redraw

    paths = redraw(args.record, only=args.only, which=args.which, device=args.device)
    for path in paths:
        print(path)
    if not paths:
        print(f"{args.record}: none of the plots had anything to draw", file=sys.stderr)
    return 0


def cmd_export(args):
    from .commands.export import export_record

    found = export_record(args.record, format=args.format, model=args.model, which=args.which, out=args.out,
                          device=args.device, opset=args.opset, dynamic=not args.static)
    print(found.path)
    return 0


def cmd_sweep(args):
    from .commands import sweep as sweeping

    sets, params = layers(args)
    configure_logging(args.quiet)
    plan = sweeping.plan_of(paths=record_configs(args.config), sets=sets, params=params, record=args.record)
    if args.count:
        print(plan.count if plan.count is not None else f"adaptive ({plan.strategy.trials} trials)")
        return 0
    if args.show is not None:
        if plan.points is None:
            print("adaptive strategy: the points are asked while the sweep runs")
            return 0
        for index, point in enumerate(plan.points[:args.show]):
            print(f"{index:04d}  " + "  ".join(f"{key}={value}" for key, value in point.items()))
        return 0
    if args.plan:
        for path in sweeping.write_plan(plan):
            print(path)
        return 0
    monitor = running.Monitor(quiet=args.quiet)
    if args.point is not None:
        result = sweeping.run_point(plan, args.point, monitor=monitor)
        print(result.record)
        return 0
    results = sweeping.run_local(plan, monitor=monitor, jobs=args.jobs, repair=args.repair)
    print(plan.root)
    return 0 if all(result is not None for result in results) else 2


def cmd_stop(args):
    from .commands.stop import stop_many

    for path in stop_many(args.records, by="cli"):
        print(path)
    return 0


def cmd_legos(args):
    from .commands import legos

    if args.docs:
        print(legos.docs_markdown())
    elif args.name and legos.is_name(args.name):
        if sys.stdout.isatty():
            from rich.console import Console

            Console().print(legos.entry_renderable(args.name))
        else:
            print(legos.entry_text(args.name))
    else:
        print(legos.catalog_text(word=args.name, slot=args.slot))
    return 0


def cmd_board(args):
    from .board import serve

    serve(args.root, host=args.host, port=args.port, open_browser=args.open)
    return 0


def cmd_tui(args):
    from .tui import open_tui

    open_tui(args.root, config=args.config)
    return 0


HANDLERS = {"check": cmd_check, "run": cmd_run, "predict": cmd_predict, "summarize": cmd_summarize,
            "generate": cmd_generate, "plots": cmd_plots, "export": cmd_export, "sweep": cmd_sweep, "stop": cmd_stop,
            "legos": cmd_legos, "board": cmd_board, "tui": cmd_tui}


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    handler = HANDLERS[args.command]
    try:
        return handler(args)
    except ProblemsError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt as exc:
        print("aborted", file=sys.stderr)
        print_notes(exc)
        return 130
    except KalfaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print_notes(exc)
        return 2
    except Exception:
        traceback.print_exc()
        return 2


def print_notes(exc):
    for note in getattr(exc, "__notes__", None) or ():
        print(note, file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
