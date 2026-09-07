"""The kalfa command line: run, check, predict, resume, collect, ls."""

import argparse
import os
import sys
import warnings

from cirak.errors import CirakError, ConfigError
from cirak.registry import registry
from tezgah import TezgahError

from . import PACKS
from .api import check as check_config
from .api import generate as generate_run
from .api import predict as predict_run
from .api import resume as resume_run
from .api import run as run_config
from .check import sets_text
from .collect import collect as collect_runs
from .kinds import kalfa_kind
from .config import parse_sets


class Style:
    def __init__(self, enabled):
        self.enabled = enabled

    def paint(self, text, code):
        return f"\x1b[{code}m{text}\x1b[0m" if self.enabled else text

    def bold(self, text):
        return self.paint(text, "1")

    def dim(self, text):
        return self.paint(text, "2")

    def red(self, text):
        return self.paint(text, "31")

    def green(self, text):
        return self.paint(text, "32")

    def yellow(self, text):
        return self.paint(text, "33")

    def cyan(self, text):
        return self.paint(text, "36")


def style_for(stream):
    is_tty = stream.isatty() if hasattr(stream, "isatty") else False
    return Style(is_tty and "NO_COLOR" not in os.environ)


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except ConfigError as exc:
        print_problems(exc.problems, sys.stderr)
        return 1
    except (CirakError, TezgahError, ValueError, FileNotFoundError) as exc:
        print(style_for(sys.stderr).red(str(exc)), file=sys.stderr)
        return 1


def build_parser():
    parser = argparse.ArgumentParser(prog="kalfa", description="YAML front end for PyTorch training")
    commands = parser.add_subparsers(dest="command", required=True)

    run_cmd = commands.add_parser("run", help="check, compile and train; the record directory holds everything")
    run_cmd.add_argument("config", nargs="+")
    _set_option(run_cmd)
    run_cmd.add_argument("--executor", default="serial")
    run_cmd.add_argument("--workers", type=int)
    run_cmd.set_defaults(handler=cmd_run)

    check_cmd = commands.add_parser("check", help="report every problem without running")
    check_cmd.add_argument("config", nargs="+")
    _set_option(check_cmd)
    check_cmd.add_argument("--layers", action="store_true", help="print the layer tree and the overridden leaves")
    check_cmd.add_argument("--dump", action="store_true", help="print the expanded flow the way flow.yaml records it")
    check_cmd.add_argument("--recipe", action="store_true", help="print the driver document the templates open")
    check_cmd.add_argument("--load", action="store_true",
                           help="run the data block and report the real set sizes (after filters)")
    check_cmd.set_defaults(handler=cmd_check)

    predict_cmd = commands.add_parser("predict", help="predict with a recorded run")
    predict_cmd.add_argument("run")
    predict_cmd.add_argument("--model", help="any model of the run, composites and .ema copies included")
    predict_cmd.add_argument("--which", choices=["best", "last"])
    predict_cmd.add_argument("--data", help="predict on this file instead of the run's test set")
    _device_option(predict_cmd)
    _set_option(predict_cmd)
    predict_cmd.set_defaults(handler=cmd_predict)

    generate_cmd = commands.add_parser("generate", help="run the generate lego of a recorded run")
    generate_cmd.add_argument("run")
    generate_cmd.add_argument("--which", choices=["best", "last"])
    _device_option(generate_cmd)
    _set_option(generate_cmd)
    generate_cmd.set_defaults(handler=cmd_generate)

    resume_cmd = commands.add_parser("resume", help="continue a run from last.pt or final/ into a new directory")
    resume_cmd.add_argument("run")
    _set_option(resume_cmd)
    resume_cmd.add_argument("--executor", default="serial")
    resume_cmd.add_argument("--workers", type=int)
    resume_cmd.set_defaults(handler=cmd_resume)

    sweep_cmd = commands.add_parser("sweep", help="run the points of the config's sweep section: the local loop "
                                                  "(every point in a subprocess), --id N for one point, --count, "
                                                  "--show N")
    sweep_cmd.add_argument("config", nargs="+")
    _set_option(sweep_cmd)
    sweep_cmd.add_argument("--record", help="root directory of the points (overrides sweep.record)")
    sweep_cmd.add_argument("--count", action="store_true", help="print the number of points and stop")
    sweep_cmd.add_argument("--show", type=int, metavar="N", help="print point N and stop (strategies deterministic "
                                                                 "by id)")
    sweep_cmd.add_argument("--id", type=int, metavar="N", dest="point_id",
                           help="run point N only, for a queue job; the record is <root>/<N>")
    sweep_cmd.add_argument("--point", help=argparse.SUPPRESS)
    sweep_cmd.set_defaults(handler=cmd_sweep)

    collect_cmd = commands.add_parser("collect", help="summarize fold runs (cv.json, cv.md), a list of runs or a "
                                                      "sweep root (sweep.csv, sweep.json, sweep.md, the best point)")
    collect_cmd.add_argument("runs", nargs="+")
    collect_cmd.add_argument("--out", help="directory for the summary files (default: the runs' parent)")
    collect_cmd.set_defaults(handler=cmd_collect)

    docs_cmd = commands.add_parser("docs", help="print the lego reference generated from the registry, or write it "
                                                "with --write DOCS.md")
    docs_cmd.add_argument("--write", metavar="PATH", help="write the reference to this file instead of printing it")
    docs_cmd.set_defaults(handler=cmd_docs)

    ls_cmd = commands.add_parser("ls", help="list alias packs and legos with their kinds and facts; a word without "
                                            "a leading slash searches names, aliases and descriptions")
    ls_cmd.add_argument("prefix", nargs="?", default=None, metavar="PREFIX|WORD")
    ls_cmd.add_argument("--kind")
    ls_cmd.set_defaults(handler=cmd_ls)
    return parser


def _device_option(command):
    command.add_argument("--device", metavar="DEVICE",
                         help="run on this device: a short name (cuda, mps, cpu, auto) or a lego call "
                              "('{uri: cuda, params: {index: 1}}'); the cpu without it")


def _device_value(args):
    from cirak.loader import parse_value

    if getattr(args, "device", None) is None:
        return None
    return parse_value(args.device)


def _set_option(command):
    command.add_argument("--set", action="append", default=[], metavar="PATH=VALUE",
                         help="override a value at a dotted path from the document root (--set training.epochs=5, "
                              "--set device=cuda); the value is read as YAML")
    command.add_argument("-p", "--param", action="append", default=[], metavar="NAME=VALUE",
                         help="override a params entry (-p lr=1e-4 is --set params.lr=1e-4)")


def _layer(args):
    try:
        return parse_sets(args.set, args.param)
    except ValueError as exc:
        raise SystemExit(_usage(str(exc)))


def _usage(message):
    print(f"kalfa: error: {message}", file=sys.stderr)
    return 2


def print_problems(problems, stream):
    style = style_for(stream)
    word = "problem" if len(problems) == 1 else "problems"
    print(style.bold(f"{len(problems)} {word} found:"), file=stream)
    for number, problem in enumerate(problems, 1):
        paint = style.red if problem.severity == "error" else style.yellow
        line = f"  {number}. {paint(f'[{problem.kind}]')} {problem.message}"
        if problem.file is not None:
            line += " " + style.cyan(f"({problem.file}:{problem.line})")
        if problem.hint is not None:
            line += style.dim(f"; {problem.hint}")
        print(line, file=stream)


def cmd_check(args) -> int:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prepared = check_config(args.config, _layer(args), load=args.load)
    style = style_for(sys.stdout)
    if args.layers:
        print(prepared.surface.layers_text())
    if prepared.problems:
        print_problems(prepared.problems, sys.stdout)
    else:
        print(style.green("no problems found"))
    if prepared.sizes is not None:
        print(sets_text(prepared.sizes))
    if prepared.loaded is not None:
        print(sets_text(prepared.loaded, loaded=True))
    if prepared.implicit:
        print("implicit bindings:")
        for path, param, key in prepared.implicit:
            print(f"  {path}: {param} <- {key}")
    if args.recipe:
        if prepared.document is None:
            print(style.red("the config could not be shaped, no recipe"), file=sys.stderr)
        else:
            from .recipe import recipe_text

            sys.stdout.write("---\n")
            sys.stdout.write(recipe_text(prepared.document))
    if args.dump:
        text = prepared.dump()
        if text is None:
            print(style.red("the flow could not be compiled, nothing to dump"), file=sys.stderr)
        else:
            sys.stdout.write("---\n")
            sys.stdout.write(text)
    return 1 if prepared.errors else 0


def cmd_run(args) -> int:
    style = style_for(sys.stdout)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = run_config(args.config, _layer(args), executor=args.executor, workers=args.workers)
    _print_warnings(caught)
    print(f"run {style.bold(result.report.run)}: {style.green('ok')}; device {result.device}; "
          f"record {style.cyan(result.record)}")
    return 0


def cmd_resume(args) -> int:
    style = style_for(sys.stdout)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = resume_run(args.run, _layer(args), executor=args.executor, workers=args.workers)
    _print_warnings(caught)
    print(f"resumed {style.bold(result.report.run)}: {style.green('ok')}; device {result.device}; "
          f"record {style.cyan(result.record)}")
    return 0


def cmd_predict(args) -> int:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = predict_run(args.run, model=args.model, which=args.which, data=args.data, sets=_layer(args),
                             device=_device_value(args))
    style = style_for(sys.stdout)
    print(f"predicted {len(result.table)} rows with {style.bold(result.model)}: {style.cyan(result.path)}")
    return 0


def cmd_generate(args) -> int:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = generate_run(args.run, which=args.which, sets=_layer(args), device=_device_value(args))
    style = style_for(sys.stdout)
    print(f"generated: {style.cyan(result.path)}")
    return 0


def cmd_collect(args) -> int:
    kind, text, target = collect_runs(args.runs, args.out)
    sys.stdout.write(text)
    files = "sweep.csv, sweep.json and sweep.md" if kind == "sweep" else "cv.json and cv.md"
    print(f"wrote {files} under {target}")
    return 0


def cmd_sweep(args) -> int:
    import json

    from . import sweep as sweeper

    style = style_for(sys.stdout)
    plan = sweeper.plan(args.config, _layer(args), record=args.record)
    if args.count:
        print(plan.total)
        return 0
    if args.show is not None:
        print(json.dumps(sweeper.point_of(plan, args.show)))
        return 0
    if args.point_id is not None:
        point = json.loads(args.point) if args.point else None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            entry = sweeper.run_point(args.config, args.set, args.param, plan, args.point_id, point)
        _print_warnings(caught)
        objective = entry["objective"]
        print(f"point {entry['id']} {entry['point']}: {objective['monitor']}={objective['value']:.6g} at turn "
              f"{objective['turn']}; record {style.cyan(entry['record'])}")
        return 0
    if args.point is not None:
        raise SystemExit(_usage("--point needs --id"))
    entries = sweeper.local_loop(args.config, args.set, args.param, plan, log=print)
    finished = [entry for entry in entries if entry]
    print(f"{len(finished)}/{plan.total} points finished under {style.cyan(str(plan.root))}; "
          f"summarize with: kalfa collect {plan.root}")
    return 0 if len(finished) == plan.total else 1


def cmd_docs(args) -> int:
    from .docs import render

    text = render()
    if args.write:
        from pathlib import Path

        Path(args.write).write_text(text)
        print(f"wrote {args.write}")
        return 0
    sys.stdout.write(text)
    return 0


def cmd_ls(args) -> int:
    style = style_for(sys.stdout)
    prefix = args.prefix
    if prefix is not None and not prefix.startswith("/"):
        entries = search_entries(prefix, args.kind)
        _print_entries(entries, style, aliases=True)
        return 0
    if prefix is None or prefix.startswith("/alias/"):
        for uri, path in sorted(registry.fragments().items()):
            if not uri.startswith("/alias/") or (prefix is not None and not uri.startswith(prefix.rstrip("/"))):
                continue
            print(style.bold(uri) + "  " + style.dim(str(path)))
            _print_pack(path, style, args.kind)
        if prefix is not None:
            return 0
        print()
    entries = registry.ls(prefix or "/")
    if args.kind is not None:
        entries = [entry for entry in entries if kalfa_kind(entry.uri) == args.kind]
    _print_entries(entries, style)
    return 0


def search_entries(word, kind=None):
    """Registered legos whose URI, alias names or description contain the word (case insensitive)."""
    needle = word.lower()
    found = []
    for uri in sorted(registry.uris()):
        entry = registry.lookup(uri)
        if entry is None or entry.fragment:
            continue
        if kind is not None and kalfa_kind(uri) != kind:
            continue
        haystack = " ".join([uri, entry.description or "", *entry.facts.alias]).lower()
        if needle in haystack:
            found.append(entry)
    return found


def pack_members():
    """Alias name and pack per URI, read from the registered alias packs."""
    from ruamel.yaml import YAML

    members = {}
    for uri, path in sorted(registry.fragments().items()):
        if not uri.startswith("/alias/"):
            continue
        table = (YAML(typ="safe").load(open(path).read()) or {}).get("alias") or {}
        for name, target in table.items():
            members.setdefault(target, []).append((name, uri.rsplit("/", 1)[-1]))
    return members


def _print_pack(path, style, kind):
    from ruamel.yaml import YAML

    table = (YAML(typ="safe").load(open(path).read()) or {}).get("alias") or {}
    width = max((len(name) for name in table), default=0)
    for name, uri in table.items():
        found = kalfa_kind(uri)
        if kind is not None and found != kind:
            continue
        print(f"  {style.cyan(name.ljust(width))}  {style.yellow((found or '').ljust(10))}  {uri}")


def _print_entries(entries, style, aliases=False):
    entries = [entry for entry in entries if not entry.fragment]
    if not entries:
        print(style.dim("nothing found"))
        return
    members = pack_members() if aliases else {}
    width = max(len(entry.uri) for entry in entries)
    kinds = [kalfa_kind(entry.uri) or entry.facts.kind or "" for entry in entries]
    kind_width = max(len(kind) for kind in kinds)
    for entry, kind in zip(entries, kinds):
        line = f"{style.cyan(entry.uri.ljust(width))}  {style.yellow(kind.ljust(kind_width))}  {style.dim(entry.description)}"
        facts = {name: value for name, value in entry.facts.declared().items() if name != "kind"}
        if facts:
            line += "  " + style.dim(_facts_text(facts))
        print(line)
        if aliases:
            names = {}
            for name, pack in members.get(entry.uri, []):
                names.setdefault(name, []).append(pack)
            for name in entry.facts.alias:
                names.setdefault(name, [])
            if names:
                text = ", ".join(f"{name} ({', '.join(packs)})" if packs else name for name, packs in names.items())
                print(f"  {style.dim('alias:')} {text}")


def _facts_text(facts):
    parts = []
    for name, value in facts.items():
        if isinstance(value, list):
            parts.append(f"{name}: {', '.join(str(item) for item in value)}")
        elif isinstance(value, dict):
            parts.append(f"{name}: " + ", ".join(f"{key}={item}" for key, item in value.items()))
        else:
            parts.append(f"{name}: {value}")
    return "[" + "; ".join(parts) + "]"


def _print_warnings(caught):
    style = style_for(sys.stderr)
    for entry in caught:
        print(style.yellow(f"warning: {entry.message}"), file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
