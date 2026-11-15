import importlib
import inspect
import json
import textwrap
from pathlib import Path

from rich.console import Group
from rich.padding import Padding
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

from kalfa.core import aliases
from kalfa.core.build import node_signature
from kalfa.core.errors import BuildError, KalfaError
from kalfa.std import notes


SLOT_ORDER = ("data", "model", "loss", "metric", "optimizer", "schedule", "step", "trigger", "action", "logger",
              "sampler", "plot", "strategy", "value")
UNKNOWN = "unknown"
SWEEP_MODULE = "kalfa.commands.sweep"
REGISTRARS = (SWEEP_MODULE,)
REQUIRED = "required"
WIDTH = 100
NOTE_INDENT = "      "
THEME = "ansi_dark"
PARENTS = Path(__file__).resolve().parents
ROOT = PARENTS[3] if PARENTS[2].name == "src" else None
EXAMPLES = "examples/*/*.yaml"
SECTIONS = ("models", "losses", "metrics", "optimizers", "plots")
SINGLE = ("data", "generate")
COLUMNS = ("features", "targets")
LEGO_KEYS = ("activation", "init", "schedule", "step", "strategy", "tokenizer", "when", "feed", "run", "log", "layers",
             "each", "split", "transform", "train_transform")
SEEN = {}
CATEGORIES = (
    ("data", "data", "slot", ("data",)),
    ("splits", "data.split", "module", ("kalfa.std.data.splits",)),
    ("table preprocessing", "data.features and data.targets", "module",
     ("kalfa.std.data.pre", "sklearn.preprocessing")),
    ("feeds", "data.feed", "module", ("kalfa.std.data.window",)),
    ("image transforms", "data.transform and data.train_transform", "module", ("kalfa.std.data.transforms",)),
    ("tokenizers", "data.tokenizer", "module", ("kalfa.std.data.text.CharTokenizer",)),
    ("models", "models.<name>", "slot", ("model",)),
    ("layers", "layers of a stack model", "name",
     ("linear", "conv2d", "batch_norm", "layer_norm", "dropout", "embedding", "flatten", "last_step")),
    ("activations", "the activation of mlp or layers of stack", "name", ("relu", "gelu", "silu", "sigmoid", "tanh")),
    ("init", "models.<name>.init", "module", ("kalfa.std.init",)),
    ("losses", "losses.<name>", "slot", ("loss",)),
    ("objectives", "losses.<name>", "name",
     ("vae_elbo", "wgan_gp_d", "wgan_g", "ddpm", "distill", "ntxent", "mdmm", "weighted_sum")),
    ("noise schedules", "the schedule of ddpm", "name", ("cosine_beta", "linear_beta")),
    ("metrics", "metrics.<name>", "slot", ("metric",)),
    ("optimizers", "optimizers.<name>", "slot", ("optimizer",)),
    ("lr schedules", "optimizers.<name>.schedule", "slot", ("schedule",)),
    ("steps", "train.step", "slot", ("step",)),
    ("triggers", "rules.<name>.when", "slot", ("trigger",)),
    ("actions", "rules.<name>.run", "slot", ("action",)),
    ("samplers", "generate", "slot", ("sampler",)),
    ("plots", "plots.<name>", "slot", ("plot",)),
    ("loggers", "train.log", "slot", ("logger",)),
    ("sweep strategies", "sweep.strategy", "slot", ("strategy",)),
    ("other", None, "slot", ()),
)
OTHER = "other"
PLACES = {category: place for category, place, _, _ in CATEGORIES}
RULES = ("name", "module", "slot")
FIELD = 8
PLUGIN = "(plugin)"


def load_registrars():
    for module in REGISTRARS:
        importlib.import_module(module)
    notes.apply()


def strategy_note(entry):
    target = importlib.import_module(SWEEP_MODULE).STRATEGIES.get(entry.name)
    if target is None or entry.slot == "strategy":
        return None
    path = aliases.target_path(target)
    names = [other.name for other in aliases.catalog(slot="strategy") if aliases.target_path(other.target) == path]
    return f"in a sweep section, {entry.name} is the {names[0]} strategy" if names else None


def group_of(entry):
    return entry.slot or UNKNOWN


def page_entry(name):
    load_registrars()
    return aliases.entry_of(name) if isinstance(name, str) else None


def entry_from(name_or_entry):
    if isinstance(name_or_entry, aliases.Entry):
        return name_or_entry
    return page_entry(name_or_entry)


def target_in(entry, modules):
    path = aliases.target_path(entry.target)
    return any(path == module or path.startswith(module + ".") for module in modules)


def fits(entry, rule, members):
    if rule == "name":
        return entry.name in members
    if rule == "module":
        return target_in(entry, members)
    return entry.slot in members


def category_of(name_or_entry):
    entry = entry_from(name_or_entry)
    if entry is None:
        return OTHER
    for rule in ("slot",) if entry.plugin else RULES:
        for category, _, kind, members in CATEGORIES:
            if kind == rule and fits(entry, rule, members):
                return category
    return OTHER


def place_of(name_or_entry):
    return PLACES.get(category_of(name_or_entry))


def filters(word, slot):
    return (word if isinstance(word, str) and word else None), (str(slot).lower() if slot else None)


def categories(word=None, slot=None):
    load_registrars()
    word, slot = filters(word, slot)
    groups = {}
    for entry in aliases.catalog(word=word):
        if slot is None or group_of(entry) == slot:
            groups.setdefault(category_of(entry), []).append(entry)
    return [(category, groups[category]) for category, *_ in CATEGORIES if category in groups]


def slot_rank(slot):
    if slot in SLOT_ORDER:
        return (0, SLOT_ORDER.index(slot), slot)
    if slot == UNKNOWN:
        return (2, 0, slot)
    return (1, 0, slot)


def flat(text):
    return " ".join(str(text or "").split())


def wrapped(text, lead, rest=None):
    body = flat(text)
    if not body:
        return [lead.rstrip()]
    rest = " " * len(lead) if rest is None else rest
    width = max(WIDTH, len(lead) + 40)
    return textwrap.wrap(body, width=width, initial_indent=lead, subsequent_indent=rest, break_long_words=False,
                         break_on_hyphens=False)


def shown(value):
    if inspect.isclass(value) or inspect.isroutine(value) or inspect.ismodule(value):
        return aliases.target_path(value)
    text = repr(value)
    if " at 0x" in text:
        return f"<{aliases.target_path(type(value))}>"
    return text


def parameters(entry):
    target = aliases.import_target(entry.target)
    signature = node_signature(target, entry.slot or "value")
    empty = inspect.Parameter.empty
    return [(name, parameter.default, parameter.default is empty) for name, parameter in signature.params.items()]


def signature_of(entry):
    if isinstance(entry, str):
        load_registrars()
        found = aliases.entry_of(entry)
        if found is None:
            raise KalfaError(f"no name {entry}")
        entry = found
    return [(name, REQUIRED if required else default) for name, default, required in parameters(entry)]


def is_name(text):
    load_registrars()
    return isinstance(text, str) and text in aliases.ALIASES


def present_slots():
    return sorted({group_of(entry) for entry in aliases.ALIASES.values()}, key=slot_rank)


def empty_text(word, slot):
    if word is None:
        if slot is None:
            return "no names are registered"
        return f"no name of slot {slot}; the slots are " + ", ".join(present_slots())
    lines = [f"no name matches {word}" + (f" in slot {slot}" if slot else "")]
    hint = aliases.suggestion(word)
    if hint:
        lines.append(hint)
    return "\n".join(lines)


def listed_doc(entry):
    return f"{PLUGIN} {flat(entry.doc)}".rstrip() if entry.plugin else flat(entry.doc)


def catalog_text(word=None, slot=None):
    word, slot = filters(word, slot)
    found = categories(word, slot)
    if not found:
        return empty_text(word, slot)
    lines = []
    for category, members in found:
        if lines:
            lines.append("")
        lines.append(category)
        width = max(len(entry.name) for entry in members)
        for entry in members:
            lines.extend(wrapped(listed_doc(entry), "  " + entry.name.ljust(width) + "  "))
    return "\n".join(lines)


def strings(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def nested_words(value):
    if isinstance(value, list):
        return [word for item in value for word in nested_words(item)]
    if not isinstance(value, dict):
        return []
    found = []
    for key, item in value.items():
        if key == "use" or key in LEGO_KEYS:
            found.extend(strings(item))
        found.extend(nested_words(item))
    return found


def section_words(tree):
    found = []
    for section in SECTIONS:
        entries = tree.get(section)
        if isinstance(entries, dict):
            found.extend(value for value in entries.values() if isinstance(value, str))
    found.extend(tree[section] for section in SINGLE if isinstance(tree.get(section), str))
    data = tree.get("data")
    if isinstance(data, dict):
        for key in COLUMNS:
            if isinstance(data.get(key), dict):
                found.extend(word for value in data[key].values() for word in strings(value))
    return found


def lego_words(tree):
    if not isinstance(tree, dict):
        return []
    return section_words(tree) + nested_words(tree)


def loaded(path):
    try:
        return YAML(typ="safe", pure=True).load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, YAMLError):
        return None


def words_in(path):
    try:
        stamp = path.stat().st_mtime_ns
    except OSError:
        return frozenset()
    found = SEEN.get(path)
    if found is None or found[0] != stamp:
        found = (stamp, frozenset(lego_words(loaded(path))))
        SEEN[path] = found
    return found[1]


def example_files():
    if ROOT is None or not (ROOT / "examples").is_dir():
        return []
    return list(ROOT.glob(EXAMPLES))


def used_in(name):
    entry = aliases.entry_of(name) if isinstance(name, str) else None
    spellings = {name} if entry is None else {name, aliases.target_path(entry.target)}
    return sorted(path.relative_to(ROOT).as_posix() for path in example_files() if spellings & set(words_in(path)))


def example_source(example):
    text = example if isinstance(example, str) else json.dumps(example, default=str)
    return text.strip("\n")


def example_lines(source):
    return ["  " + line if line.strip() else "" for line in source.splitlines()]


def fields_of(entry):
    category = category_of(entry)
    rows = [("field", "target", aliases.target_path(entry.target)), ("field", "slot", group_of(entry)),
            ("field", "category", category)]
    if PLACES.get(category):
        rows.append(("field", "goes in", PLACES[category]))
    if entry.plugin:
        rows.append(("field", "from", "a plugin"))
    note = strategy_note(entry)
    if note:
        rows.append(("field", "sweep", note))
    return rows


def param_rows(entry):
    try:
        found = parameters(entry)
    except BuildError as exc:
        return [("problem", f"not readable: {exc}")]
    rows = []
    for param, default, required in found:
        rows.append(("param", param, None if required else shown(default)))
        if flat(entry.params.get(param)):
            rows.append(("note", flat(entry.params[param])))
    return rows or [("problem", "none")]


def page_of(entry):
    rows = [("title", entry.name), *fields_of(entry)]
    if flat(entry.doc):
        rows += [("blank",), ("doc", flat(entry.doc))]
    rows += [("blank",), ("heading", "params:"), *param_rows(entry)]
    if entry.examples:
        rows += [("blank",), ("heading", "examples:")]
        for position, example in enumerate(entry.examples):
            if position:
                rows.append(("blank",))
            rows.append(("example", example_source(example)))
    used = used_in(entry.name)
    if used:
        rows += [("blank",), ("heading", "used in:"), *[("path", path) for path in used]]
    return rows


def param_line(row):
    return f"  {row[1]} (required)" if row[2] is None else f"  {row[1]} = {row[2]}"


TEXT = {
    "title": lambda row: [row[1]],
    "field": lambda row: [f"  {row[1].ljust(FIELD)}  {row[2]}"],
    "doc": lambda row: wrapped(row[1], "  "),
    "heading": lambda row: [row[1]],
    "param": lambda row: [param_line(row)],
    "note": lambda row: wrapped(row[1], NOTE_INDENT),
    "example": lambda row: example_lines(row[1]),
    "path": lambda row: ["  " + row[1]],
    "problem": lambda row: ["  " + row[1]],
    "blank": lambda row: [""],
}


def indented(renderable, width):
    return Padding(renderable, (0, 0, 0, width))


def rich_param(row):
    name = ("  " + row[1], "bold green")
    if row[2] is None:
        return Text.assemble(name, (" (required)", "red"))
    return Text.assemble(name, (" = ", "dim"), (row[2], "cyan"))


def rich_example(row):
    return indented(Syntax(row[1], "yaml", theme=THEME, background_color="default", word_wrap=True), 2)


RICH = {
    "title": lambda row: Text(row[1], style="bold cyan"),
    "field": lambda row: Text.assemble((f"  {row[1].ljust(FIELD)}  ", "dim"), row[2]),
    "doc": lambda row: indented(Text(row[1]), 2),
    "heading": lambda row: Text(row[1], style="bold yellow"),
    "param": rich_param,
    "note": lambda row: indented(Text(row[1]), len(NOTE_INDENT)),
    "example": rich_example,
    "path": lambda row: Text("  " + row[1], style="blue"),
    "problem": lambda row: Text("  " + row[1], style="yellow"),
    "blank": lambda row: Text(""),
}


def missing_text(name):
    lines = [f"no name {name}"]
    hint = aliases.suggestion(name)
    if hint:
        lines.append(hint)
    return "\n".join(lines)


def entry_text(name):
    entry = page_entry(name)
    if entry is None:
        return missing_text(name)
    return "\n".join(line for row in page_of(entry) for line in TEXT[row[0]](row))


def entry_renderable(name):
    entry = page_entry(name)
    if entry is None:
        return Text(missing_text(name))
    return Group(*[RICH[row[0]](row) for row in page_of(entry)])


def entry_label(entry):
    parts = [(entry.name, "bold")]
    if entry.plugin:
        parts.append(("  " + PLUGIN, "magenta"))
    if flat(entry.doc):
        parts.append(("  " + flat(entry.doc), "dim"))
    return Text.assemble(*parts)


def category_renderable(category, entries):
    rows = [RICH["title"](("title", category))]
    if PLACES.get(category):
        rows.append(RICH["field"](("field", "goes in", PLACES[category])))
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold green", no_wrap=True)
    grid.add_column()
    for entry in entries:
        grid.add_row(Text(entry.name), Text(listed_doc(entry)))
    return Group(*rows, Text(""), indented(grid, 2))


def prose(text):
    return flat(text).replace("<", "\\<").replace("*", "\\*")


def markdown_param(entry, param, default, required):
    head = f"`{param}` (required)" if required else f"`{param}={shown(default)}`"
    note = prose(entry.params.get(param))
    return f"* {head}: {note}" if note else f"* {head}"


def markdown_params(entry):
    try:
        found = parameters(entry)
    except BuildError as exc:
        return [f"Params: not readable: {prose(exc)}"]
    if not found:
        return ["Params: none."]
    return ["Params:", "", *[markdown_param(entry, *item) for item in found]]


def markdown_examples(entry):
    lines = ["Examples:" if len(entry.examples) > 1 else "Example:"]
    for example in entry.examples:
        lines += ["", "```yaml", example_source(example), "```"]
    return lines


def markdown_entry(entry):
    origin = ", from a plugin" if entry.plugin else ""
    lines = [f"### {entry.name}", "", f"Target `{aliases.target_path(entry.target)}`{origin}."]
    note = strategy_note(entry)
    if note:
        lines += ["", note[0].upper() + note[1:] + "."]
    if flat(entry.doc):
        lines += ["", prose(entry.doc)]
    lines += ["", *markdown_params(entry)]
    if entry.examples:
        lines += ["", *markdown_examples(entry)]
    used = used_in(entry.name)
    if used:
        lines += ["", "Used in " + ", ".join(f"`{path}`" for path in used) + "."]
    return lines


def markdown_place(place):
    return " ".join(f"`{word}`" if "." in word or " " not in place else word for word in place.split())


def docs_markdown():
    lines = ["# kalfa legos"]
    for category, members in categories():
        lines += ["", f"## {category}"]
        if PLACES.get(category):
            lines += ["", f"Goes in {markdown_place(PLACES[category])}."]
        for entry in members:
            lines += ["", *markdown_entry(entry)]
    return "\n".join(lines).rstrip()
