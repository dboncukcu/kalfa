import argparse
import json
import platform
import re
import signal
import sys
import warnings
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from cirak.errors import CirakError, ConfigError
from cirak.loader import parse_value
from cirak.registry import registry
from tezgah import TezgahError

from . import __version__
from .config import import_plugins, pack_tables, parse_sets
from .contract import Contract
from .describe import ALL_SECTIONS, DEFAULT_SECTIONS
from .kinds import kalfa_kind
from .record import Record
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


def installed_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def version_text():
    parts = ", ".join(f"{name} {installed_version(name)}" for name in ("cirak", "tezgah", "torch"))
    return f"kalfa {__version__} ({parts}, python {platform.python_version()})"


def examples_text(examples):
    pairs = [re.split(r" {4,}", line, maxsplit=1) for line in examples]
    width = max((len(pair[0]) for pair in pairs), default=0)
    shown = [f"  {pair[0]:<{width}}   {pair[1]}" if len(pair) > 1 else f"  {pair[0]}" for pair in pairs]
    return "examples:\n" + "\n".join(shown)


def command(commands, name, summary, description, examples=()):
    return commands.add_parser(name, help=summary, description=description,
                               epilog=examples_text(examples) if examples else None,
                               formatter_class=argparse.RawDescriptionHelpFormatter)


def executor_options(command):
    command.add_argument("--executor", choices=["serial", "thread", "dask"], default="serial", metavar="NAME",
                         help="how tezgah runs the nodes: serial (the default), thread (unordered nodes side by side, "
                              "the aliasing warning becomes an error) or dask when it is installed")
    command.add_argument("--workers", type=int, metavar="N", help="the threads or workers of the executor")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="kalfa", formatter_class=argparse.RawDescriptionHelpFormatter,
        description="YAML front end for PyTorch training: a config names the data, the models, the losses, the\n"
                    "optimizers and the training; kalfa checks it, compiles it and runs it into a record directory\n"
                    "that every other command reads.",
        epilog="a first pass:\n"
               "  kalfa check cfg.yaml                  every problem of the config, nothing runs\n"
               "  kalfa describe cfg.yaml --measure     the config as an analysis, with the real sizes\n"
               "  kalfa run cfg.yaml -p lr=1e-4         check, compile and train into the record directory\n"
               "  kalfa predict runs/x --data new.parquet\n"
               "  kalfa board runs                      follow the records in the browser\n\n"
               "kalfa <command> --help explains a command with examples; CONFIG.md is the reference of the config,\n"
               "kalfa docs the reference of the legos, kalfa ls what a name resolves to.")
    parser.add_argument("--version", action="version", version=version_text(),
                        help="the versions of kalfa, cirak, tezgah, torch and python, what a bug report needs")
    commands = parser.add_subparsers(dest="command", required=True, title="commands", metavar="<command>")

    run_cmd = command(commands, "run", "check, compile and train into the record directory",
                      "Check the config, compile it and train. The record directory the config names is created and "
                      "holds everything: the resolved config, the history, the checkpoints, the predictions, the "
                      "plots. A non empty record directory is an error, nothing is overwritten. Ctrl-c while training "
                      "asks the run to stop after its turn and the after block still runs; a second ctrl-c aborts.",
                      ["kalfa run cfg.yaml",
                       "kalfa run base.yaml site.yaml -p lr=1e-4 -p epochs=50    later files override earlier ones",
                       "kalfa run cfg.yaml --set device=cuda --log --progress steps",
                       "kalfa run cfg.yaml --prepared data/prepared    start from the data kalfa prepare wrote"])
    run_cmd.add_argument("config", nargs="+", metavar="CONFIG",
                         help="one or more YAML files; later ones override earlier ones")
    overrides = run_cmd.add_argument_group("overrides")
    set_option(overrides)
    contract_option(overrides)
    running = run_cmd.add_argument_group("running")
    running.add_argument("--prepared", metavar="DIR", help="start from the data kalfa prepare wrote into DIR")
    executor_options(running)
    output = run_cmd.add_argument_group("output")
    log_option(output)
    progress_option(output)
    run_cmd.set_defaults(handler=cmd_run)

    check_cmd = command(commands, "check", "report every problem of a config without running",
                        "Read the config and report every problem without running anything: unknown and missing keys, "
                        "names that resolve to nothing, kinds that do not fit, a signature mismatch, columns no field "
                        "matches, targets that pair with nothing, and the warnings. The exit code is 1 when there are "
                        "errors. A record directory in place of the config reads its resolved.yaml under the contract "
                        "it ran by.",
                        ["kalfa check cfg.yaml",
                         "kalfa check cfg.yaml --measure    the set sizes after the transforms, from the data",
                         "kalfa check cfg.yaml --dump > flow.yaml    the expanded flow, the way flow.yaml records it",
                         "kalfa check runs/x    the config a run went by"])
    check_cmd.add_argument("config", nargs="+", metavar="CONFIG", help="one or more YAML files, or a record directory")
    overrides = check_cmd.add_argument_group("overrides")
    set_option(overrides)
    contract_option(overrides)
    overrides.add_argument("--prepared", metavar="DIR", help="check against the data kalfa prepare wrote into DIR")
    printing = check_cmd.add_argument_group("what else to print")
    printing.add_argument("--layers", action="store_true", help="the layer tree of the files and the overridden leaves")
    printing.add_argument("--dump", action="store_true", help="the expanded flow the way flow.yaml records it")
    printing.add_argument("--recipe", action="store_true", help="the driver document the templates open")
    printing.add_argument("--measure", action="store_true",
                          help="run the data block and report the set sizes after the transforms")
    check_cmd.set_defaults(handler=cmd_check)

    describe_cmd = command(commands, "describe", "the config as an analysis: data, model, training, after, columns",
                           "The same checks as kalfa check, then the config as an analysis: DATA (source, split, "
                           "batch, feed, the field table, the path of every set), MODEL (one line per model with its "
                           "layers), TRAINING (optimizers, losses, metrics, checkpoint, stop, the rule chain), AFTER "
                           "(report, predict, plots, generate, record) and COLUMNS (every source column with its "
                           "dtype, field, chain, role and tensor slot). The tables fit the width of the terminal.",
                           ["kalfa describe cfg.yaml",
                            "kalfa describe cfg.yaml --measure    real sizes, column widths, parameter counts",
                            "kalfa describe cfg.yaml --section model --section training",
                            "kalfa describe cfg.yaml --save report.txt    nothing clipped, no colors",
                            "kalfa describe runs/x    the config a run went by"])
    describe_cmd.add_argument("config", nargs="+", metavar="CONFIG",
                              help="one or more YAML files, or a record directory")
    overrides = describe_cmd.add_argument_group("overrides")
    set_option(overrides)
    contract_option(overrides)
    overrides.add_argument("--prepared", metavar="DIR", help="describe against the data kalfa prepare wrote into DIR")
    printing = describe_cmd.add_argument_group("what to print")
    printing.add_argument("--measure", action="store_true",
                          help="run the data and model blocks: the set sizes after the transforms, the fitted column "
                               "widths, the tensor slots and the parameter counts")
    printing.add_argument("--section", action="append", default=[], choices=list(ALL_SECTIONS),
                          metavar="NAME", help=f"this section only, repeatable: {', '.join(ALL_SECTIONS)}")
    printing.add_argument("--wiring", action="store_true", help="add the implicit bindings of the compiled pipeline")
    printing.add_argument("--save", metavar="PATH",
                          help="write the analysis to this file instead of printing it, nothing clipped, no colors")
    describe_cmd.set_defaults(handler=cmd_describe)

    predict_cmd = command(commands, "predict", "predict with a recorded run, on its test set or on a new file",
                          "Predict with a recorded run: the report model, or any model with --model, on the run's test "
                          "set or on a new file with --data. The predictions land beside the run in the original "
                          "units, with the raw outputs and the row ids: predictions.parquet, or "
                          "predictions_<file>.parquet with --data.",
                          ["kalfa predict runs/x",
                           "kalfa predict runs/x --data new.parquet --plots    and the plots on these predictions",
                           "kalfa predict runs/x --model encoder --which last --device cuda"])
    predict_cmd.add_argument("run", metavar="RUN", help="the record directory of a run")
    predict_cmd.add_argument("--model", metavar="NAME",
                             help="any model of the run, composites and .ema copies included; the report model without")
    predict_cmd.add_argument("--which", choices=["best", "last", "final"],
                             help="the weights to load; without it what training.report chose")
    predict_cmd.add_argument("--data", metavar="PATH", help="predict on this file instead of the run's test set")
    predict_cmd.add_argument("--plots", nargs="?", const="all", metavar="NAMES",
                             help="draw the plots section on the predictions just made: every plot, or a comma "
                                  "separated list of definitions; the files take the suffix of the predictions file")
    device_option(predict_cmd)
    overrides = predict_cmd.add_argument_group("overrides")
    set_option(overrides)
    contract_option(overrides)
    log_option(predict_cmd)
    predict_cmd.set_defaults(handler=cmd_predict)

    generate_cmd = command(commands, "generate", "run the generate lego of a recorded run",
                           "Run the generate section of a recorded run with its report model and write what the "
                           "sampler produces under samples/ of the run.",
                           ["kalfa generate runs/x", "kalfa generate runs/x --which last --device cuda"])
    generate_cmd.add_argument("run", metavar="RUN", help="the record directory of a run")
    generate_cmd.add_argument("--which", choices=["best", "last", "final"],
                              help="the weights to load; without it what training.report chose")
    device_option(generate_cmd)
    overrides = generate_cmd.add_argument_group("overrides")
    set_option(overrides)
    contract_option(overrides)
    log_option(generate_cmd)
    generate_cmd.set_defaults(handler=cmd_generate)

    prepare_cmd = command(commands, "prepare", "run the data block once into a directory runs start from",
                          "Run the data block once and write it into a directory: one parquet per set after the "
                          "transforms and the split, the fitted preprocessors and frame transforms, the data report "
                          "and a manifest. A run or a sweep point starts from it with --prepared and skips the data "
                          "block.",
                          ["kalfa prepare cfg.yaml --out data/prepared",
                           "kalfa run cfg.yaml --prepared data/prepared"])
    prepare_cmd.add_argument("config", nargs="+", metavar="CONFIG",
                             help="one or more YAML files; later ones override earlier ones")
    prepare_cmd.add_argument("--out", required=True, metavar="DIR", help="the directory to write; it must not exist")
    overrides = prepare_cmd.add_argument_group("overrides")
    set_option(overrides)
    contract_option(overrides)
    log_option(prepare_cmd)
    prepare_cmd.set_defaults(handler=cmd_prepare)

    export_cmd = command(commands, "export", "write a model of a recorded run in another format",
                         "Write a model of a recorded run in another format: onnx, pt2 or state_dict, or an export "
                         "lego of your own by alias or URI, under <run>/export or the directory --out names.",
                         ["kalfa export runs/x",
                          "kalfa export runs/x --format onnx --model encoder --which best",
                          "kalfa export runs/x --format /export/acme/mine --out exported"])
    export_cmd.add_argument("run", metavar="RUN", help="the record directory of a run")
    export_cmd.add_argument("--format", default="state_dict", metavar="NAME",
                            help="an export lego by alias or URI: onnx, pt2, state_dict (the default) or your own")
    export_cmd.add_argument("--model", metavar="NAME",
                            help="any model of the run, composites included; the predicts model without")
    export_cmd.add_argument("--which", choices=["best", "last", "final"],
                            help="the weights to load; without it what training.report chose")
    export_cmd.add_argument("--out", metavar="DIR", help="the directory to write into; <run>/export without it")
    device_option(export_cmd)
    overrides = export_cmd.add_argument_group("overrides")
    set_option(overrides)
    contract_option(overrides)
    log_option(export_cmd)
    export_cmd.set_defaults(handler=cmd_export)

    plots_cmd = command(commands, "plots", "redraw the plots section of a recorded run",
                        "Redraw the plots section of a recorded run from its files: the predictions and the history "
                        "from the record, the models from the report weights, the data replayed without a fit. Nothing "
                        "is fitted or trained again.",
                        ["kalfa plots runs/x",
                         "kalfa plots runs/x --only loss_curve,residuals",
                         "kalfa plots runs/x --set figures.format=pdf"])
    plots_cmd.add_argument("run", metavar="RUN", help="the record directory of a run")
    plots_cmd.add_argument("--only", metavar="NAMES", help="a comma separated list of the plot definitions to draw")
    device_option(plots_cmd)
    overrides = plots_cmd.add_argument_group("overrides")
    set_option(overrides)
    contract_option(overrides)
    log_option(plots_cmd)
    plots_cmd.set_defaults(handler=cmd_plots)

    resume_cmd = command(commands, "resume", "continue a run from its last checkpoint into a new directory",
                         "Continue a run from last.pt, or from final/ when it ended, into a new record directory next "
                         "to it: the models, the optimizers, the counters and the rule state carry over and the "
                         "history continues.",
                         ["kalfa resume runs/x",
                          "kalfa resume runs/x --set training.epochs=200    more epochs than the config wrote"])
    resume_cmd.add_argument("run", metavar="RUN", help="the record directory of the run to continue")
    overrides = resume_cmd.add_argument_group("overrides")
    set_option(overrides)
    contract_option(overrides)
    running = resume_cmd.add_argument_group("running")
    executor_options(running)
    output = resume_cmd.add_argument_group("output")
    log_option(output)
    progress_option(output)
    resume_cmd.set_defaults(handler=cmd_resume)

    sweep_cmd = command(commands, "sweep", "run the points of the sweep section, locally or one point per job",
                        "Run the sweep section of the config: the strategy draws the points and each one is an "
                        "ordinary run under <root>/<id>/. The local loop runs every point in a subprocess; on a queue "
                        "system --plan writes the root once and every job runs one point with --id.",
                        ["kalfa sweep cfg.yaml    every point, one after the other",
                         "kalfa sweep cfg.yaml --count    how many points",
                         "kalfa sweep cfg.yaml --show 3    the params of point 3",
                         "kalfa sweep cfg.yaml --plan --prepare-data --record sweeps/lr    root, plan and data",
                         "kalfa sweep cfg.yaml --id 3 --record sweeps/lr    one point, for a queue job",
                         "kalfa sweep cfg.yaml --no-progress --log info    a line per turn, no bar, for a job log",
                         "kalfa collect sweeps/lr    the table and the best point"])
    sweep_cmd.add_argument("config", nargs="+", metavar="CONFIG",
                           help="one or more YAML files with a sweep section")
    set_option(sweep_cmd)
    sweep_cmd.add_argument("--record", metavar="ROOT", help="the root directory of the points; overrides sweep.record")
    asking = sweep_cmd.add_argument_group("without running")
    asking.add_argument("--count", action="store_true", help="print the number of points and stop")
    asking.add_argument("--show", type=int, metavar="N",
                        help="print point N and stop; the strategies are deterministic by id")
    asking.add_argument("--plan", action="store_true",
                        help="write the root once: manifest.json, sweep.plan and the site files sweep.sub and "
                             "sweep.sh (never overwritten), then stop")
    asking.add_argument("--prepare-data", action="store_true", dest="prepare_data",
                        help="with --plan, run the data block once into <root>/data; the points start from it")
    sweep_cmd.add_argument("--id", type=int, metavar="N", dest="point_id",
                           help="run point N only, for a queue job; the record is <root>/<N>")
    sweep_cmd.add_argument("--point", help=argparse.SUPPRESS)
    output = sweep_cmd.add_argument_group("output")
    log_option(output)
    progress_option(output)
    sweep_cmd.set_defaults(handler=cmd_sweep)

    collect_cmd = command(commands, "collect", "summarize fold runs or a sweep root",
                          "Summarize a list of runs or the fold runs of a cross validation (cv.json, cv.md), or a "
                          "sweep root (sweep.csv, sweep.json, sweep.md and the best point), from the records alone.",
                          ["kalfa collect runs/cv_*", "kalfa collect sweeps/lr --out reports"])
    collect_cmd.add_argument("runs", nargs="+", metavar="RECORD", help="record directories, or one sweep root")
    collect_cmd.add_argument("--out", metavar="DIR", help="the directory of the summary files; the runs' parent "
                             "without it")
    collect_cmd.set_defaults(handler=cmd_collect)

    docs_cmd = command(commands, "docs", "the lego reference generated from the registry",
                       "Print the lego reference generated from the registry: every lego by kind with its URI, "
                       "aliases, signature, facts and description, and the alias packs. With --plugin or --config your "
                       "own legos come with.",
                       ["kalfa docs | less", "kalfa docs --write DOCS.md", "kalfa docs --plugin my_legos"])
    docs_cmd.add_argument("--write", metavar="PATH", help="write the reference to this file instead of printing it")
    plugin_option(docs_cmd)
    docs_cmd.set_defaults(handler=cmd_docs)

    ls_cmd = command(commands, "ls", "list alias packs and legos, or search them by a word",
                     "List the alias packs and the legos with their kinds and facts. A URI prefix narrows the list, a "
                     "word without a leading slash searches names, aliases and descriptions.",
                     ["kalfa ls",
                      "kalfa ls /alias/kalfa/tabular    what the pack binds",
                      "kalfa ls /criterion    every criterion",
                      "kalfa ls scaler    every lego with scaler in its name or description",
                      "kalfa ls --kind plot --plugin my_legos"])
    ls_cmd.add_argument("prefix", nargs="?", default=None, metavar="PREFIX|WORD",
                        help="a URI prefix (/criterion, /alias/kalfa/base) or a word to search for")
    ls_cmd.add_argument("--kind", metavar="KIND", help="only the legos of this kind: layer, criterion, plot, ...")
    plugin_option(ls_cmd)
    ls_cmd.set_defaults(handler=cmd_ls)

    board_cmd = command(commands, "board", "a page over the records under a root, live",
                        "Serve the records under a root as a page: what is running with its progress, every run as a "
                        "table, the curves, the architecture, the data pipeline, the predictions, the files, and a "
                        "stop button for a running record. It reads the records and writes nothing but stop.json. No "
                        "dependency beyond Python; on a batch system run it where the files are and reach it through "
                        "an ssh tunnel.",
                        ["kalfa board runs",
                         "kalfa board /scratch/sweeps --port 9000",
                         "ssh -L 8080:127.0.0.1:8080 login.node    then open http://127.0.0.1:8080"])
    board_cmd.add_argument("root", metavar="ROOT", help="the directory whose records are shown, searched recursively")
    board_cmd.add_argument("--host", default="127.0.0.1", metavar="HOST",
                           help="the address to listen on; 127.0.0.1 without it, this machine only")
    board_cmd.add_argument("--port", type=int, default=8080, metavar="PORT", help="the port; 8080 without it")
    log_option(board_cmd)
    board_cmd.set_defaults(handler=cmd_board)

    stop_cmd = command(commands, "stop", "ask a running record to stop after its current turn",
                       "Write stop.json into a running record, or into a sweep root and its running points. The loop "
                       "ends after the turn that sees it and the run finishes as an early stop would: the final state, "
                       "the predictions, the plots. A stopped run continues with kalfa resume.",
                       ["kalfa stop runs/x", "kalfa stop sweeps/lr    the root and every running point"])
    stop_cmd.add_argument("record", nargs="+", metavar="RECORD", help="record directories or sweep roots")
    stop_cmd.set_defaults(handler=cmd_stop)

    contract_cmd = command(commands, "contract", "the contract kalfa runs configs by",
                           "Print the contract, the wiring and the flow blocks that take a config to the pipeline. "
                           "Write it out with --write to edit it and run against the copy with --contract.",
                           ["kalfa contract", "kalfa contract --write contract.yaml",
                            "kalfa run cfg.yaml --contract contract.yaml"])
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
    progress = False if args.no_progress else ("turns" if args.progress == "turns" else True)
    return Monitor(level_of(args.log), progress=progress, log_every=args.log_every, tensorboard=args.tensorboard)


def output_flags(args):
    flags = ["--log", args.log] if args.log else []
    if args.no_progress:
        flags.append("--no-progress")
    elif args.progress != "turns":
        flags += ["--progress", args.progress]
    if args.log_every:
        flags += ["--log-every", str(args.log_every)]
    if args.tensorboard:
        flags.append("--tensorboard")
    return flags


class Interrupt:
    def __init__(self, monitor):
        self.monitor = monitor
        self.previous = None
        self.asked = False

    def __enter__(self):
        self.previous = signal.signal(signal.SIGINT, self.handle)
        return self

    def __exit__(self, *error):
        signal.signal(signal.SIGINT, self.previous)

    def handle(self, number, frame):
        if not self.monitor.training or self.asked:
            raise KeyboardInterrupt
        self.asked = True
        Record(self.monitor.record).request_stop("ctrl-c")
        self.monitor.say("stop requested: the run ends after this turn with its final state, predictions and "
                         "plots; ctrl-c again aborts it now")


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
    from . import api
    from .describe.render import measure_text

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prepared = api.check(args.config, layer_of(args), measure=args.measure, contract=contract_of(args),
                             prepared=args.prepared)
    style = style_for(sys.stdout)
    print(style.dim(version_text()))
    if args.layers:
        print(prepared.surface.layers_text())
    if prepared.problems:
        print_problems(prepared.problems, sys.stdout)
    else:
        print(style.green("no problems found"))
    if prepared.measured is not None:
        print(measure_text(prepared, style))
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
    from . import api

    style = style_for(sys.stdout)
    with monitor_of(args) as monitor, Interrupt(monitor), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        monitor.echo_warnings(caught)
        result = api.run(args.config, layer_of(args), executor=args.executor, workers=args.workers,
                         contract=contract_of(args), monitor=monitor, prepared=args.prepared)
    print_warnings(caught)
    print(f"run {style.bold(result.report.run)}: {style.green('ok')}; device {result.device}; "
          f"record {style.cyan(result.record)}")
    return 0


def cmd_resume(args) -> int:
    from . import api

    style = style_for(sys.stdout)
    with monitor_of(args) as monitor, Interrupt(monitor), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        monitor.echo_warnings(caught)
        result = api.resume(args.run, layer_of(args), executor=args.executor, workers=args.workers,
                            contract=contract_of(args), monitor=monitor)
    print_warnings(caught)
    print(f"resumed {style.bold(result.report.run)}: {style.green('ok')}; device {result.device}; "
          f"record {style.cyan(result.record)}")
    return 0


def cmd_describe(args) -> int:
    from . import api
    from .describe.render import render, report

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prepared = api.check(args.config, layer_of(args), contract=contract_of(args), prepared=args.prepared)
    style = style_for(sys.stdout)
    if prepared.problems:
        print_problems(prepared.problems, sys.stdout)
    else:
        print(style.green("no problems found"))
    found = None
    if args.measure:
        if prepared.document is None or prepared.errors:
            print(style.yellow("--measure needs a config without errors; describing the config as written"),
                  file=sys.stderr)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                found = api.probe(prepared.document, prepared.contract)
    sections = list(args.section) if args.section else None
    if args.wiring and "wiring" not in (sections or ()):
        sections = list(sections or DEFAULT_SECTIONS) + ["wiring"]
    if args.save:
        Path(args.save).write_text(report(prepared, Style(False), sections, found))
        print(f"wrote {args.save}")
    elif style.enabled:
        sys.stdout.write(render(prepared, style, sections, found))
    else:
        sys.stdout.write(report(prepared, style, sections, found))
    return 1 if prepared.errors else 0


def cmd_predict(args) -> int:
    from . import api

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
    from . import api

    with Monitor(level_of(args.log)), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = api.prepare_data(args.config, layer_of(args), out=args.out, contract=contract_of(args))
    print_warnings(caught)
    style = style_for(sys.stdout)
    sizes = ", ".join(f"{name} {size}" for name, size in result.sizes.items())
    print(f"prepared {sizes}: {style.cyan(result.directory)}")
    return 0


def cmd_export(args) -> int:
    from . import api

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
    from . import api

    with Monitor(level_of(args.log)), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = api.plots(args.run, only=args.only, sets=layer_of(args), device=device_value(args),
                           contract=contract_of(args))
    style = style_for(sys.stdout)
    print(f"plots {', '.join(result.names) or 'none'}: {style.cyan(str(Path(result.record) / 'plots'))}")
    return 0


def cmd_generate(args) -> int:
    from . import api

    with Monitor(level_of(args.log)), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = api.generate(args.run, which=args.which, sets=layer_of(args), device=device_value(args),
                              contract=contract_of(args))
    style = style_for(sys.stdout)
    print(f"generated: {style.cyan(result.path)}")
    return 0


def cmd_collect(args) -> int:
    from . import collect

    kind, text, target = collect.collect(args.runs, args.out)
    sys.stdout.write(text)
    files = "sweep.csv, sweep.json and sweep.md" if kind == "sweep" else "cv.json and cv.md"
    print(f"wrote {files} under {target}")
    return 0


def cmd_sweep(args) -> int:
    from . import sweep

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
        with monitor_of(args) as monitor, Interrupt(monitor), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            entry = sweep.run_point(args.config, args.set, args.param, plan, args.point_id, point, monitor=monitor)
        print_warnings(caught)
        objective = entry["objective"]
        print(f"point {entry['id']} {entry['point']}: {objective['monitor']}={objective['value']:.6g} at turn "
              f"{objective['turn']}; record {style.cyan(entry['record'])}")
        return 0
    if args.point is not None:
        raise SystemExit(usage("--point needs --id"))
    entries = sweep.local_loop(args.config, args.set, args.param, plan, log=print, options=output_flags(args))
    finished = [entry for entry in entries if entry]
    print(f"{len(finished)}/{plan.total} points finished under {style.cyan(str(plan.root))}; "
          f"summarize with: kalfa collect {plan.root}")
    return 0 if len(finished) == plan.total else 1


def cmd_docs(args) -> int:
    from . import docs

    failed = load_plugins(args)
    text = docs.render(plugins=docs.plugin_uris() if args.plugin or args.config else None)
    if args.write:
        Path(args.write).write_text(text)
        print(f"wrote {args.write}")
    else:
        sys.stdout.write(text)
    return 1 if failed else 0


def cmd_board(args) -> int:
    from . import board

    server = board.serve(args.root, host=args.host, port=args.port)
    style = style_for(sys.stdout)
    print(f"kalfa board over {style.cyan(args.root)} at http://{args.host}:{server.server_address[1]}/ "
          f"(ctrl-c stops it)")
    with Monitor(level_of(args.log)):
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
    return 0


def cmd_stop(args) -> int:
    from . import api

    style = style_for(sys.stdout)
    for record in args.record:
        written = api.stop(record)
        count = len(written) - 1
        points = f" and {count} running point{'s' if count > 1 else ''}" if count else ""
        print(f"stop requested for {style.cyan(record)}{points}; the run ends after its current turn")
    return 0


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


def module_of(entry):
    if isinstance(entry.target, str):
        return entry.target.partition(":")[0]
    return entry.target.__module__


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
        line += "  " + style.dim(module_of(entry))
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
