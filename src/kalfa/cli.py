import argparse
import json
import sys
import warnings
from pathlib import Path

from cirak.errors import CirakError, ConfigError
from cirak.loader import parse_value
from cirak.registry import registry
from tezgah import TezgahError

from . import api, collect, describe, docs, sweep
from .config import import_plugins, pack_tables, parse_sets
from .contract import Contract
from .kinds import kalfa_kind
from .recipe import recipe_text
from .std.common.log import Monitor, level_of
from .style import Style, style_for


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except ConfigError as exception:
        print_problems(exception.problems, sys.stderr)
        return 1
    except (CirakError, TezgahError, ValueError, FileNotFoundError) as exception:
        print(style_for(sys.stderr).red(str(exception)), file=sys.stderr)
        return 1


def build_parser():
    parser = argparse.ArgumentParser(prog="kalfa", description="YAML front end for PyTorch training")
    commands = parser.add_subparsers(dest="command", required=True)

    run_cmd = commands.add_parser("run", help="check, compile and train; the record directory holds everything")
    run_cmd.add_argument("config", nargs="+")
    set_option(run_cmd)
    contract_option(run_cmd)
    run_cmd.add_argument("--prepared", metavar="DIR", help="start from the data kalfa prepare wrote into DIR")
    run_cmd.add_argument("--executor", default="serial")
    run_cmd.add_argument("--workers", type=int)
    log_option(run_cmd)
    progress_option(run_cmd)
    run_cmd.set_defaults(handler=cmd_run)

    check_cmd = commands.add_parser("check", help="report every problem without running")
    check_cmd.add_argument("config", nargs="+")
    set_option(check_cmd)
    contract_option(check_cmd)
    check_cmd.add_argument("--layers", action="store_true", help="print the layer tree and the overridden leaves")
    check_cmd.add_argument("--dump", action="store_true", help="print the expanded flow the way flow.yaml records it")
    check_cmd.add_argument("--recipe", action="store_true", help="print the driver document the templates open")
    check_cmd.add_argument("--load", action="store_true",
                           help="run the data block and report the real set sizes (after filters)")
    check_cmd.add_argument("--prepared", metavar="DIR", help="check against the data kalfa prepare wrote into DIR")
    check_cmd.set_defaults(handler=cmd_check)

    describe_cmd = commands.add_parser("describe", help="the config as an analysis: data, model, training, after "
                                                       "and the columns, after the same checks")
    describe_cmd.add_argument("config", nargs="+")
    set_option(describe_cmd)
    contract_option(describe_cmd)
    describe_cmd.add_argument("--load", action="store_true",
                              help="run the data and model blocks: the real set sizes, the fitted column widths and "
                                   "the parameter counts")
    describe_cmd.add_argument("--prepared", metavar="DIR",
                              help="describe against the data kalfa prepare wrote into DIR")
    describe_cmd.add_argument("--section", action="append", default=[], choices=list(describe.ALL_SECTIONS),
                              help="print this section only (repeatable)")
    describe_cmd.add_argument("--wiring", action="store_true",
                              help="add the implicit bindings of the compiled pipeline")
    describe_cmd.add_argument("--save", metavar="PATH",
                              help="write the analysis to this file instead of printing it, with nothing clipped "
                                   "and no colors")
    describe_cmd.set_defaults(handler=cmd_describe)

    predict_cmd = commands.add_parser("predict", help="predict with a recorded run")
    predict_cmd.add_argument("run")
    predict_cmd.add_argument("--model", help="any model of the run, composites and .ema copies included")
    predict_cmd.add_argument("--which", choices=["best", "last", "final"])
    predict_cmd.add_argument("--data", help="predict on this file instead of the run's test set")
    predict_cmd.add_argument("--plots", nargs="?", const="all", metavar="NAMES",
                             help="draw the plots section on the predictions just made: every plot, or a comma "
                                  "separated list of definitions; the files take the suffix of the predictions file")
    device_option(predict_cmd)
    set_option(predict_cmd)
    contract_option(predict_cmd)
    log_option(predict_cmd)
    predict_cmd.set_defaults(handler=cmd_predict)

    generate_cmd = commands.add_parser("generate", help="run the generate lego of a recorded run")
    generate_cmd.add_argument("run")
    generate_cmd.add_argument("--which", choices=["best", "last", "final"])
    device_option(generate_cmd)
    set_option(generate_cmd)
    contract_option(generate_cmd)
    log_option(generate_cmd)
    generate_cmd.set_defaults(handler=cmd_generate)

    prepare_cmd = commands.add_parser("prepare", help="run the data block once and write the applied sets, the "
                                                      "fitted state and the data report into a directory a run "
                                                      "starts from with --prepared")
    prepare_cmd.add_argument("config", nargs="+")
    prepare_cmd.add_argument("--out", required=True, metavar="DIR", help="the directory to write")
    set_option(prepare_cmd)
    contract_option(prepare_cmd)
    log_option(prepare_cmd)
    prepare_cmd.set_defaults(handler=cmd_prepare)

    export_cmd = commands.add_parser("export", help="write a model of a recorded run in another format: onnx, "
                                                    "torchscript or state_dict, or an export lego of your own")
    export_cmd.add_argument("run")
    export_cmd.add_argument("--format", default="state_dict", metavar="NAME",
                            help="an export lego by alias or URI (default state_dict)")
    export_cmd.add_argument("--model", help="any model of the run, composites included; the predicts model without")
    export_cmd.add_argument("--which", choices=["best", "last", "final"])
    export_cmd.add_argument("--out", metavar="DIR", help="the directory to write into (default <run>/export)")
    device_option(export_cmd)
    set_option(export_cmd)
    contract_option(export_cmd)
    log_option(export_cmd)
    export_cmd.set_defaults(handler=cmd_export)

    plots_cmd = commands.add_parser("plots", help="redraw the plots section of a recorded run from its files and its "
                                                  "data; nothing is fitted or trained again")
    plots_cmd.add_argument("run")
    plots_cmd.add_argument("--only", metavar="NAMES", help="a comma separated list of the plot definitions to draw")
    device_option(plots_cmd)
    set_option(plots_cmd)
    contract_option(plots_cmd)
    log_option(plots_cmd)
    plots_cmd.set_defaults(handler=cmd_plots)

    resume_cmd = commands.add_parser("resume", help="continue a run from last.pt or final/ into a new directory")
    resume_cmd.add_argument("run")
    set_option(resume_cmd)
    contract_option(resume_cmd)
    resume_cmd.add_argument("--executor", default="serial")
    resume_cmd.add_argument("--workers", type=int)
    log_option(resume_cmd)
    progress_option(resume_cmd)
    resume_cmd.set_defaults(handler=cmd_resume)

    sweep_cmd = commands.add_parser("sweep", help="run the points of the config's sweep section: the local loop "
                                                  "(every point in a subprocess), --id N for one point, --count, "
                                                  "--show N")
    sweep_cmd.add_argument("config", nargs="+")
    set_option(sweep_cmd)
    sweep_cmd.add_argument("--record", help="root directory of the points (overrides sweep.record)")
    sweep_cmd.add_argument("--count", action="store_true", help="print the number of points and stop")
    sweep_cmd.add_argument("--show", type=int, metavar="N", help="print point N and stop (strategies deterministic "
                                                                 "by id)")
    sweep_cmd.add_argument("--id", type=int, metavar="N", dest="point_id",
                           help="run point N only, for a queue job; the record is <root>/<N>")
    sweep_cmd.add_argument("--plan", action="store_true",
                           help="write the root once: manifest.json, sweep.plan and the site files sweep.sub and "
                                "sweep.sh (never overwritten), then stop")
    sweep_cmd.add_argument("--prepare-data", action="store_true", dest="prepare_data",
                           help="with --plan, run the data block once into <root>/data; the points start from it")
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
    plugin_option(docs_cmd)
    docs_cmd.set_defaults(handler=cmd_docs)

    ls_cmd = commands.add_parser("ls", help="list alias packs and legos with their kinds and facts; a word without "
                                            "a leading slash searches names, aliases and descriptions")
    ls_cmd.add_argument("prefix", nargs="?", default=None, metavar="PREFIX|WORD")
    ls_cmd.add_argument("--kind")
    plugin_option(ls_cmd)
    ls_cmd.set_defaults(handler=cmd_ls)

    contract_cmd = commands.add_parser("contract", help="print the contract kalfa runs configs by (the wiring and "
                                                        "the flow blocks), or write it with --write for editing")
    contract_cmd.add_argument("--write", metavar="PATH", help="write the contract to this file instead of printing it")
    contract_cmd.set_defaults(handler=cmd_contract)
    return parser


def contract_option(command):
    command.add_argument("--contract", metavar="PATH",
                         help="run against this contract instead of the built in one (kalfa contract --write "
                              "exports it); a command over a record takes the record's copy without it")


def contract_of(args):
    path = getattr(args, "contract", None)
    return Contract.load(path) if path is not None else None


def plugin_option(command):
    command.add_argument("--plugin", action="append", default=[], metavar="MODULE",
                         help="import this module before listing, so that its legos come with; a module name on "
                              "sys.path or next to the working directory, or a path to a .py file (repeatable)")
    command.add_argument("--config", action="append", default=[], metavar="PATH",
                         help="import the modules of this config's plugins section before listing; the config is "
                              "read, not validated (repeatable)")


def load_plugins(args):
    if not args.plugin and not args.config:
        return False

    problems = import_plugins(args.config, args.plugin)
    if problems:
        print_problems(problems, sys.stderr)
    return any(problem.severity == "error" for problem in problems)


def device_option(command):
    command.add_argument("--device", metavar="DEVICE",
                         help="run on this device: a short name (cuda, mps, cpu, auto) or a lego call "
                              "('{uri: cuda, params: {index: 1}}'); the cpu without it")


def device_value(args):
    if getattr(args, "device", None) is None:
        return None
    return parse_value(args.device)


def log_option(command):
    command.add_argument("--log", nargs="?", const="info", choices=["info", "debug"], metavar="LEVEL",
                         help="print to stderr what the run is doing while it does it: info is the narrative "
                              "(device, data, models, one line per turn), debug adds every node of the pipeline "
                              "with its time and the decisions inside the legos; bare --log means info")


def progress_option(command):
    command.add_argument("--no-progress", action="store_true",
                         help="no progress bar; with --log the turn lines take its place, without it the run says "
                              "nothing until it ends")
    command.add_argument("--progress", choices=["turns", "steps"], default="turns",
                         help="the bar over the turns (the default), or an inner bar over the steps of a turn too")
    command.add_argument("--log-every", type=int, metavar="N",
                         help="under --log, a line every N steps with the loss, the learning rate and the "
                              "gradient norm of every optimizer")
    command.add_argument("--tensorboard", action="store_true",
                         help="write TensorBoard event files under <record>/tensorboard (the tensorboard package "
                              "is optional): every history and step value, the params as hparams")


def monitor_of(args):
    progress = False if args.no_progress else ("steps" if args.progress == "steps" else True)
    return Monitor(level_of(args.log), progress=progress, log_every=args.log_every, tensorboard=args.tensorboard)


def set_option(command):
    command.add_argument("--set", action="append", default=[], metavar="PATH=VALUE",
                         help="override a value at a dotted path from the document root (--set training.epochs=5, "
                              "--set device=cuda); the value is read as YAML")
    command.add_argument("-p", "--param", action="append", default=[], metavar="NAME=VALUE",
                         help="override a params entry (-p lr=1e-4 is --set params.lr=1e-4)")


def layer_of(args):
    try:
        return parse_sets(args.set, args.param)
    except ValueError as exception:
        raise SystemExit(usage(str(exception)))


def usage(message):
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
        prepared = api.check(args.config, layer_of(args), load=args.load, contract=contract_of(args),
                             prepared=args.prepared)
    style = style_for(sys.stdout)
    if args.layers:
        print(prepared.surface.layers_text())
    if prepared.problems:
        print_problems(prepared.problems, sys.stdout)
    else:
        print(style.green("no problems found"))
    if prepared.loaded is not None:
        print(describe.load_text(prepared, style))
    if args.recipe:
        if prepared.document is None:
            print(style.red("the config could not be shaped, no recipe"), file=sys.stderr)
        else:
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
    with monitor_of(args) as monitor, warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        monitor.echo_warnings(caught)
        result = api.run(args.config, layer_of(args), executor=args.executor, workers=args.workers,
                         contract=contract_of(args), monitor=monitor, prepared=args.prepared)
    print_warnings(caught)
    print(f"run {style.bold(result.report.run)}: {style.green('ok')}; device {result.device}; "
          f"record {style.cyan(result.record)}")
    return 0


def cmd_resume(args) -> int:
    style = style_for(sys.stdout)
    with monitor_of(args) as monitor, warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        monitor.echo_warnings(caught)
        result = api.resume(args.run, layer_of(args), executor=args.executor, workers=args.workers,
                            contract=contract_of(args), monitor=monitor)
    print_warnings(caught)
    print(f"resumed {style.bold(result.report.run)}: {style.green('ok')}; device {result.device}; "
          f"record {style.cyan(result.record)}")
    return 0


def cmd_describe(args) -> int:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prepared = api.check(args.config, layer_of(args), contract=contract_of(args), prepared=args.prepared)
    style = style_for(sys.stdout)
    if prepared.problems:
        print_problems(prepared.problems, sys.stdout)
    else:
        print(style.green("no problems found"))
    found = None
    if args.load:
        if prepared.document is None or prepared.errors:
            print(style.yellow("--load needs a config without errors; describing the config as written"),
                  file=sys.stderr)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                found = api.probe(prepared.document, prepared.contract)
    sections = list(args.section) if args.section else None
    if args.wiring and "wiring" not in (sections or ()):
        sections = list(sections or describe.DEFAULT_SECTIONS) + ["wiring"]
    if args.save:
        Path(args.save).write_text(describe.report(prepared, Style(False), sections, found))
        print(f"wrote {args.save}")
    elif style.enabled:
        sys.stdout.write(describe.render(prepared, style, sections, found))
    else:
        sys.stdout.write(describe.report(prepared, style, sections, found))
    return 1 if prepared.errors else 0


def cmd_predict(args) -> int:
    with Monitor(level_of(args.log)), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = api.predict(args.run, model=args.model, which=args.which, data=args.data, sets=layer_of(args),
                             device=device_value(args), contract=contract_of(args), plots=args.plots)
    style = style_for(sys.stdout)
    print(f"predicted {len(result.table)} rows with {style.bold(result.model)}: {style.cyan(result.path)}")
    if result.plots:
        print(f"plots {', '.join(result.plots)}: {style.cyan(str(Path(args.run) / 'plots'))}")
    return 0


def cmd_prepare(args) -> int:
    with Monitor(level_of(args.log)), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = api.prepare_data(args.config, layer_of(args), out=args.out, contract=contract_of(args))
    print_warnings(caught)
    style = style_for(sys.stdout)
    sizes = ", ".join(f"{name} {size}" for name, size in result.sizes.items())
    print(f"prepared {sizes}: {style.cyan(result.directory)}")
    return 0


def cmd_export(args) -> int:
    with Monitor(level_of(args.log)), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = api.export(args.run, format=args.format, model=args.model, which=args.which, out=args.out,
                            sets=layer_of(args), device=device_value(args), contract=contract_of(args))
    style = style_for(sys.stdout)
    if result.path is None:
        print(style.yellow(f"nothing written: {result.format} could not export {result.model}"), file=sys.stderr)
        return 1
    print(f"exported {style.bold(result.model)} as {result.format}: {style.cyan(result.path)}")
    return 0


def cmd_plots(args) -> int:
    with Monitor(level_of(args.log)), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = api.plots(args.run, only=args.only, sets=layer_of(args), device=device_value(args),
                           contract=contract_of(args))
    style = style_for(sys.stdout)
    print(f"plots {', '.join(result.names) or 'none'}: {style.cyan(str(Path(result.record) / 'plots'))}")
    return 0


def cmd_generate(args) -> int:
    with Monitor(level_of(args.log)), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = api.generate(args.run, which=args.which, sets=layer_of(args), device=device_value(args),
                              contract=contract_of(args))
    style = style_for(sys.stdout)
    print(f"generated: {style.cyan(result.path)}")
    return 0


def cmd_collect(args) -> int:
    kind, text, target = collect.collect(args.runs, args.out)
    sys.stdout.write(text)
    files = "sweep.csv, sweep.json and sweep.md" if kind == "sweep" else "cv.json and cv.md"
    print(f"wrote {files} under {target}")
    return 0


def cmd_sweep(args) -> int:
    style = style_for(sys.stdout)
    plan = sweep.plan(args.config, layer_of(args), record=args.record)
    if args.count:
        print(plan.total)
        return 0
    if args.show is not None:
        print(json.dumps(sweep.point_of(plan, args.show)))
        return 0
    if args.plan or args.prepare_data:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sweep.write_plan(plan, args.set, args.param, prepare=args.prepare_data, log=print)
        print_warnings(caught)
        print(f"submit with: condor_submit {plan.root / 'sweep.sub'}; or run the points here: kalfa sweep "
              f"{' '.join(args.config)} --record {plan.root}")
        return 0
    if args.point_id is not None:
        point = json.loads(args.point) if args.point else None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            entry = sweep.run_point(args.config, args.set, args.param, plan, args.point_id, point)
        print_warnings(caught)
        objective = entry["objective"]
        print(f"point {entry['id']} {entry['point']}: {objective['monitor']}={objective['value']:.6g} at turn "
              f"{objective['turn']}; record {style.cyan(entry['record'])}")
        return 0
    if args.point is not None:
        raise SystemExit(usage("--point needs --id"))
    entries = sweep.local_loop(args.config, args.set, args.param, plan, log=print)
    finished = [entry for entry in entries if entry]
    print(f"{len(finished)}/{plan.total} points finished under {style.cyan(str(plan.root))}; "
          f"summarize with: kalfa collect {plan.root}")
    return 0 if len(finished) == plan.total else 1


def cmd_docs(args) -> int:
    failed = load_plugins(args)
    text = docs.render(plugins=docs.plugin_uris() if args.plugin or args.config else None)
    if args.write:
        Path(args.write).write_text(text)
        print(f"wrote {args.write}")
    else:
        sys.stdout.write(text)
    return 1 if failed else 0


def cmd_contract(args) -> int:
    contract = Contract.load()
    if args.write:
        contract.write(args.write)
        print(f"wrote {args.write}")
    else:
        sys.stdout.write(contract.text())
    return 0


def cmd_ls(args) -> int:
    style = style_for(sys.stdout)
    code = 1 if load_plugins(args) else 0
    prefix = args.prefix
    if prefix is not None and not prefix.startswith("/"):
        entries = search_entries(prefix, args.kind)
        print_entries(entries, style, aliases=True)
        return code
    if prefix is None or prefix.startswith("/alias/"):
        tables = pack_tables()
        for uri, path in sorted(registry.fragments().items()):
            if not uri.startswith("/alias/") or (prefix is not None and not uri.startswith(prefix.rstrip("/"))):
                continue
            print(style.bold(uri) + "  " + style.dim(str(path)))
            print_pack(tables[uri], style, args.kind)
        if prefix is not None:
            return code
        print()
    entries = registry.ls(prefix or "/")
    if args.kind is not None:
        entries = [entry for entry in entries if kalfa_kind(entry.uri) == args.kind]
    print_entries(entries, style)
    return code


def search_entries(word, kind=None):
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
    members = {}
    for uri, table in pack_tables().items():
        for name, target in table.items():
            members.setdefault(target, []).append((name, uri.rsplit("/", 1)[-1]))
    return members


def print_pack(table, style, kind):
    width = max((len(name) for name in table), default=0)
    for name, uri in table.items():
        found = kalfa_kind(uri)
        if kind is not None and found != kind:
            continue
        print(f"  {style.cyan(name.ljust(width))}  {style.yellow((found or '').ljust(10))}  {uri}")


def print_entries(entries, style, aliases=False):
    entries = [entry for entry in entries if not entry.fragment]
    if not entries:
        print(style.dim("nothing found"))
        return
    members = pack_members() if aliases else {}
    width = max(len(entry.uri) for entry in entries)
    kinds = [kalfa_kind(entry.uri) or entry.facts.kind or "" for entry in entries]
    kind_width = max(len(kind) for kind in kinds)
    for entry, kind in zip(entries, kinds):
        line = (f"{style.cyan(entry.uri.ljust(width))}  {style.yellow(kind.ljust(kind_width))}  "
                f"{style.dim(entry.description)}")
        facts = {name: value for name, value in entry.facts.declared().items() if name != "kind"}
        if facts:
            line += "  " + style.dim(facts_text(facts))
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


def facts_text(facts):
    parts = []
    for name, value in facts.items():
        if isinstance(value, list):
            parts.append(f"{name}: {', '.join(str(item) for item in value)}")
        elif isinstance(value, dict):
            parts.append(f"{name}: " + ", ".join(f"{key}={item}" for key, item in value.items()))
        else:
            parts.append(f"{name}: {value}")
    return "[" + "; ".join(parts) + "]"


def print_warnings(caught):
    style = style_for(sys.stderr)
    for entry in caught:
        print(style.yellow(f"warning: {entry.message}"), file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
