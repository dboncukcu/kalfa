import textwrap

from rich.console import Group
from rich.padding import Padding
from rich.syntax import Syntax
from rich.text import Text

from kalfa.commands.guide.about import RUN, RUN_KEYS
from kalfa.commands.guide.seen import PARTS
from kalfa.commands.guide.words import and_list, flat


TEXT_WIDTH = 100
THEME = "ansi_dark"


def status_text(doc):
    errors = sum(problem["level"] == "error" for problem in doc["problems"])
    warnings = len(doc["problems"]) - errors
    pairs = ((errors, "error"), (warnings, "warning"))
    counts = [f"{count} {word}{'s' if count != 1 else ''}" for count, word in pairs if count]
    if not counts:
        return "ok"
    return ("ok, " if errors == 0 else "") + and_list(counts)


def problem_line(problem, own=None):
    where = f"{problem['path']}: " if problem.get("path") and problem["path"] != own else ""
    hint = f" ({problem['hint']})" if problem.get("hint") else ""
    return f"{problem['level']} {problem['code']}: {where}{problem['message']}{hint}"


def section_name(path):
    if not path:
        return None
    first = PARTS.split(path)[0]
    return RUN if first in RUN_KEYS else first


def placed(doc):
    attached = [problem for section in doc["sections"] for entry in section["entries"]
                for problem in entry["problems"]]
    names = {section["name"] for section in doc["sections"]}
    own, loose = {}, []
    for problem in doc["problems"]:
        if problem in attached:
            continue
        name = section_name(problem.get("path"))
        if name in names:
            own.setdefault(name, []).append(problem)
        else:
            loose.append(problem)
    return own, loose


def wrapped(text, lead, rest=None):
    rest = lead if rest is None else rest
    return textwrap.wrap(flat(text), width=TEXT_WIDTH, initial_indent=lead, subsequent_indent=rest,
                         break_long_words=False, break_on_hyphens=False) or [lead.rstrip()]


def text(doc):
    own, loose = placed(doc)
    lines = [f"{doc['title']}: {status_text(doc)}"]
    lines += [line for problem in loose for line in wrapped(problem_line(problem), "  ", "    ")]
    for section in doc["sections"]:
        lines += ["", section["name"], "=" * len(section["name"])] + wrapped(section["about"], "")
        lines += [line for problem in own.get(section["name"], []) for line in wrapped(problem_line(problem), "  ")]
        for entry in section["entries"]:
            lines.append("")
            lines += ["  " + line for line in entry["yaml"].splitlines()]
            lines += [line for note in entry["notes"] for line in wrapped(note, "      ")]
            lines += [line for problem in entry["problems"]
                      for line in wrapped(problem_line(problem, entry["path"]), "    ", "      ")]
    return "\n".join(lines)


def problem_text(problem, own=None):
    return Text(problem_line(problem, own), style="red" if problem["level"] == "error" else "yellow")


def head_renderable(doc):
    _, loose = placed(doc)
    style = "bold green" if doc["ok"] and not doc["problems"] else "bold yellow" if doc["ok"] else "bold red"
    title = Text.assemble((doc["title"], "bold"), ": ", (status_text(doc), style))
    return Group(title, *(Padding(problem_text(problem), (0, 0, 0, 2)) for problem in loose))


def entry_renderable(entry):
    parts = [Syntax(entry["yaml"], "yaml", theme=THEME, background_color="default", word_wrap=True)]
    if entry["notes"]:
        parts.append(Padding(Text("\n".join(entry["notes"]), style="dim"), (0, 0, 0, 4)))
    parts += [Padding(problem_text(problem, entry["path"]), (0, 0, 0, 4)) for problem in entry["problems"]]
    return Padding(Group(*parts), (1, 0, 0, 2))


def section_renderable(section, doc=None):
    own, _ = placed(doc) if doc is not None else ({}, [])
    bar = Padding(Text(section["name"], style="bold"), (0, 1), style=f"bold white on {section['color']}", expand=True)
    about = Padding(Text(section["about"], style="italic"), (0, 1))
    problems = [Padding(problem_text(problem), (0, 0, 0, 2)) for problem in own.get(section["name"], [])]
    return Group(bar, about, *problems, *(entry_renderable(entry) for entry in section["entries"]), Text(""))


def renderable(doc):
    return Group(head_renderable(doc), Text(""), *(section_renderable(section, doc) for section in doc["sections"]))
