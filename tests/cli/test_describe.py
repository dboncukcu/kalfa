import re

from helpers import config_path
from kalfa import __version__
from kalfa.cli import main


REFERENCE = config_path("reference")


def heads_of(text):
    return [line.split(" ")[1] for line in text.splitlines() if line.startswith("── ")]


def test_describe_prints_the_summary_and_the_default_sections(workdir, capsys):
    assert main(["describe", REFERENCE]) == 0
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines[0] == "no problems found"
    assert lines[1].startswith("reference.yaml ") and lines[1].endswith(f"kalfa {__version__}")
    assert "  layers      reference.yaml, /alias/kalfa/base, /alias/kalfa/tabular" in lines
    assert "  record      runs/ref_$datetime$" in lines and "  device      cpu" in lines
    assert heads_of(out) == ["DATA", "MODEL", "TRAINING", "AFTER", "COLUMNS"]
    assert re.search(r"^  source\s+parquet  reference\.parquet\s+2 000 rows · 22 columns$", out, re.M)
    assert re.search(r"^  split\s+random  0\.7 / 0\.15 / 0\.15  seed=11\s+train 1 400  ·  valid 300  ·  test 300$", out,
                     re.M)
    assert re.search(r"^  tower   trained  ·  optimizer main  ·  init weights=xavier bias=zeros", out, re.M)
    assert "  full   composite" in lines and "    x     ─→  tower      ─→  h" in lines
    assert re.search(r"^  checkpoint  best monitor=val/rmse_lin mode=min\s+report best$", out, re.M)
    assert "    turn ≥ 1                ─→  unfreeze   tail_stem.trainable := True" in lines
    assert "  report      best        predict full ─→ predictions.parquet" in lines
    assert re.search(r"^  x0\s+float64\s+num_\*\s+std\s+feature\s+x$", out, re.M) is None
    assert re.search(r"^  num_0\s+float64\s+num_\*\s+std\s+feature\s+x$", out, re.M)
    assert "\x1b[" not in out and "…" not in out


def test_describe_section_limits_the_output(workdir, capsys):
    assert main(["describe", REFERENCE, "--section", "model"]) == 0
    out = capsys.readouterr().out
    assert heads_of(out) == ["MODEL"] and out.startswith("no problems found\n\n── MODEL ")
    assert main(["describe", REFERENCE, "--section", "training", "--section", "data"]) == 0
    assert heads_of(capsys.readouterr().out) == ["DATA", "TRAINING"]


def test_describe_wiring_adds_the_implicit_bindings(workdir, capsys):
    assert main(["describe", REFERENCE, "--wiring"]) == 0
    out = capsys.readouterr().out
    assert heads_of(out) == ["DATA", "MODEL", "TRAINING", "AFTER", "COLUMNS", "WIRING"]
    wiring = out.split("── WIRING")[1].splitlines()[1:]
    assert "  data.prep: record ← record" in wiring and "  after.plots: record ← record" in wiring
    assert "  training.epochs.body.turn: monitor ← monitor" in wiring
    assert main(["describe", REFERENCE, "--section", "wiring"]) == 0
    assert heads_of(capsys.readouterr().out) == ["WIRING"]


def test_describe_measure_fills_the_real_sizes_widths_and_parameter_counts(workdir, capsys):
    assert main(["describe", REFERENCE, "--measure"]) == 0
    out = capsys.readouterr().out
    assert re.search(r"^  split\s+random  0\.7 / 0\.15 / 0\.15  seed=11\s+train 1 250  ·  valid 273  ·  test 278$", out,
                     re.M)
    assert re.search(r"^    ├─ train    1 250  ─→ fit .* ─→ table ─→ x \[64, 18\]$", out, re.M)
    assert re.search(r"^  tail_stem   frozen \(eval mode\)  ·  optimizer aux  ·  152 parameters \(0 trainable\)$", out,
                     re.M)
    assert re.search(r"^  lambdas   trained  ·  optimizer main  ·  2 parameters$", out, re.M)
    assert "the parameter counts need --measure" not in out
    assert "the produced widths and the tensor slots need --measure" not in out


def test_describe_save_writes_the_plain_report(workdir, capsys):
    report = workdir / "report.txt"
    assert main(["describe", REFERENCE, "--save", str(report)]) == 0
    assert capsys.readouterr().out == f"no problems found\nwrote {report}\n"
    text = report.read_text()
    assert heads_of(text) == ["DATA", "MODEL", "TRAINING", "AFTER", "COLUMNS"]
    assert text.startswith("reference.yaml ") and "\x1b[" not in text and "…" not in text
    assert "drop columns=[junk]" in text


def test_describe_exits_one_on_errors_and_describes_what_it_can(workdir, capsys):
    assert main(["describe", REFERENCE, "--set", "training.epochs=abc"]) == 1
    out = capsys.readouterr().out
    assert out.startswith("1 problem found:\n  1. [invalid_value] training.epochs must be a non negative integer "
                          "(--set:1)\nreference.yaml ")
    assert "  the config could not be shaped, no data analysis" in out.splitlines()
    assert heads_of(out) == ["DATA", "MODEL", "TRAINING", "AFTER", "COLUMNS"]
    assert main(["describe", REFERENCE, "--set", "training.epochs=abc", "--measure"]) == 1
    captured = capsys.readouterr()
    assert captured.err == "--measure needs a config without errors; describing the config as written\n"
    assert captured.out.startswith("1 problem found:")


def test_describe_reads_a_record_in_place_of_the_config(reference, capsys):
    assert main(["describe", reference.record, "--section", "after"]) == 0
    out = capsys.readouterr().out
    assert heads_of(out) == ["AFTER"]
    assert "  report      best        predict full ─→ predictions.parquet" in out.splitlines()
    assert "  record      runs/ref_$datetime$" in out.splitlines()
