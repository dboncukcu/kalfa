import os
from pathlib import Path

from kalfa import summary as summarizing
from kalfa.commands.report.page import document
from kalfa.commands.report.sections import (
    DETAILS,
    chosen_records,
    curves_section,
    details_section,
    folds_section,
    heading,
    ranked,
    summary_section,
    sweep_section,
    title_of,
)
from kalfa.core.errors import KalfaError
from kalfa.core.record import atomic


REPORT = "report.html"
SUFFIXES = (".html", ".htm")
__all__ = ["DETAILS", "REPORT", "SUFFIXES", "render", "target_of", "write_report"]


def render(summary, points=None, curves=None, stats=True):
    records = ranked(summary)
    chosen, rest = chosen_records(summary, records, points)
    title = title_of(summary)
    sections = [("summary", "Summary"), ("curves", "Curves")]
    if summary.folds:
        sections.append(("folds", "Folds"))
    if summary.sweep is not None:
        sections.append(("sweep", "Sweep"))
    sections.append(("records", "Records"))
    body = [heading(summary, title, sections), summary_section(summary), curves_section(summary, records, curves)]
    if summary.folds:
        body.append(folds_section(summary))
    if summary.sweep is not None:
        body.append(sweep_section(summary))
    body.append(details_section(chosen, rest, stats))
    return document(title, "\n".join(body))


def target_of(paths, out):
    if out is not None:
        target = Path(out)
        if target.is_dir() or target.suffix.lower() not in SUFFIXES:
            return target / REPORT
        return target
    return Path(paths[0]) / REPORT


def write_report(paths, out=None, by=None, mode=None, top=None, points=None):
    paths = [paths] if isinstance(paths, (str, os.PathLike)) else list(paths)
    if not paths:
        raise KalfaError("a report needs at least one record or sweep root")
    found = summarizing.summarize(paths, by=by, mode=mode, top=top)
    text = render(found, points=points)
    target = target_of(paths, out)
    with atomic(target) as temporary:
        temporary.write_text(text, encoding="utf-8")
    return target
