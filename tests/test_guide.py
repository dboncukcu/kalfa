import sys
from pathlib import Path

import pytest
from rich.console import Console

from helpers import housing_frame, write_parquet
from kalfa.commands import guide
from kalfa.core import aliases
from kalfa.core import run as running


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"

BASE = """\
params: {width: 16}
seed: 3
data:
  use: table
  file: rows.csv
  split: [0.8, 0.2]
  features: {"x*": standard_scaler}
  targets: {price: standard_scaler}
models:
  ae: {use: mlp, hidden: [$width$], out: 1, outputs: [x_hat, z]}
losses:
  mse: mse
optimizers:
  main: {use: adam, lr: 0.01, over: "@models.ae", minimizes: "@losses.mse"}
train:
  epochs: 5
  checkpoint: {watch: val/mse}
"""


PLUGIN = "guide_record_legos"

LEGOS = """import torch


def log_cosh(pred, target):
    return torch.log(torch.cosh(pred - target)).mean()
"""

RECORDED = f"""\
seed: 3
device: cpu
data:
  use: table
  file: housing.parquet
  split: [0.7, 0.15, 0.15]
  features: {{"x*": standard_scaler}}
  targets: {{price: standard_scaler}}
  batch: 32
models:
  net: {{use: mlp, hidden: [8], out: 1}}
losses:
  log_cosh: {PLUGIN}.log_cosh
optimizers:
  main: {{use: adam, lr: 0.001, over: "@models.net", minimizes: "@losses.log_cosh"}}
train:
  epochs: 1
"""


def write(folder, name, text):
    path = folder / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def section(doc, name):
    return next(item for item in doc["sections"] if item["name"] == name)


def entry(doc, path):
    return next(item for part in doc["sections"] for item in part["entries"] if item["path"] == path)


def notes(doc, path):
    return " ".join(entry(doc, path)["notes"])


def test_a_later_file_overrides_and_the_notes_name_the_layer(tmp_path):
    base = write(tmp_path, "base.yaml", BASE)
    small = write(tmp_path, "small.yaml", "train: {epochs: 2}\nmetrics: {rmse: rmse}\n")
    doc = guide.document([base, small])
    assert doc["title"] == f"{base} + {small}"
    assert entry(doc, "train.epochs")["yaml"] == "epochs: 2"
    assert "small.yaml writes it over base.yaml" in notes(doc, "train.epochs")
    assert "written by small.yaml" in notes(doc, "metrics.rmse")
    assert "written by" not in notes(doc, "losses.mse") and "writes it" not in notes(doc, "losses.mse")
    assert [item["name"] for item in doc["sections"]][-1] == "metrics"


def test_set_and_param_flags_are_layers_too(tmp_path):
    base = write(tmp_path, "base.yaml", BASE)
    doc = guide.document([base], sets=["data.batch=32", "data.features.y=abs"], params=["width=8"])
    assert entry(doc, "data.batch")["yaml"] == "batch: 32" and "written by --set" in notes(doc, "data.batch")
    assert "merged from base.yaml and --set" in notes(doc, "data.features")
    assert entry(doc, "params.width")["yaml"] == "width: 8"
    assert "-p writes it over base.yaml" in notes(doc, "params.width")
    assert "$width$ is 8" in notes(doc, "models.ae")


def test_one_set_or_param_may_be_a_plain_string(tmp_path):
    base = write(tmp_path, "base.yaml", BASE)
    doc = guide.document(base, sets="data.batch=32", params="width=8")
    listed = guide.document([base], sets=["data.batch=32"], params=["width=8"])
    assert doc["problems"] == listed["problems"] and doc["ok"] is listed["ok"]
    assert entry(doc, "data.batch")["yaml"] == "batch: 32"
    assert entry(doc, "params.width")["yaml"] == "width: 8"


def test_an_unknown_top_level_section_is_shown_and_flagged(tmp_path):
    doc = guide.document([write(tmp_path, "c.yaml", BASE + "mystery: {a: 1}\n")])
    mystery = section(doc, "mystery")
    assert "does not know the top level key mystery" in mystery["about"]
    assert mystery["color"] not in {item["color"] for item in doc["sections"] if item["name"] != "mystery"}
    assert [item["label"] for item in mystery["entries"]] == ["mystery"]
    assert [problem["code"] for problem in entry(doc, "mystery")["problems"]] == ["unknown_section"]
    assert doc["ok"] is False


def test_each_rule_reads_as_one_sentence(tmp_path):
    rules = """\
generate: {use: latent_sampler, model: "@models.ae", latent: 1, n: 4}
rules:
  halve:
    when: {use: every, epochs: 2}
    once: false
    set: {optimizers.main.lr: {times: 0.5}}
  samples:
    after: halve
    run: {use: write_samples, n: 4}
  done:
    when: {use: plateau, watch: val/mse, patience: 3, mode: max}
    stop: true
"""
    doc = guide.document([write(tmp_path, "c.yaml", BASE + rules)])
    halve = entry(doc, "rules.halve")["notes"][0]
    assert halve.startswith("Fires every 2 turns and multiplies optimizers.main.lr by 0.5") and "once false" in halve
    samples = notes(doc, "rules.samples")
    assert "on the turn after rule halve fired and runs write_samples" in samples and "write_samples:" in samples
    done = entry(doc, "rules.done")["notes"][0]
    assert done == ("Fires after 3 turns without val/mse improving (higher is better) and stops the run after that "
                    "turn.")
    assert "rule halve multiplies lr by 0.5" in notes(doc, "optimizers.main")


def test_a_sweep_config_explains_its_space_and_the_params_it_varies():
    doc = guide.document([str(EXAMPLES / "14_sweep" / "grid.yaml")])
    assert doc["ok"] is True
    names = [item["name"] for item in doc["sections"]]
    assert names[-1] == "sweep" and "sweep root" in section(doc, "sweep")["about"]
    assert [item["label"] for item in section(doc, "sweep")["entries"]] == ["strategy", "space", "objective", "record"]
    assert "grid: every combination" in notes(doc, "sweep.strategy") and "9 points" in notes(doc, "sweep.strategy")
    assert "width: 64, 128 and 256, read as $width$ by models.net.hidden" in notes(doc, "sweep.space")
    assert "the sweep tries 64, 128 and 256" in notes(doc, "params.width")
    assert "$width$ is 128 here; the sweep tries 64, 128 and 256" in notes(doc, "models.net")
    assert "grid.yaml writes it over config.yaml" in notes(doc, "models.net")
    assert "ranks the points by the best value of val/rmse" in notes(doc, "sweep.objective")
    assert "include" in [item["label"] for item in section(doc, "run")["entries"]]


def test_a_grid_tries_steps_over_a_range_and_a_random_search_draws_from_it(tmp_path):
    random = EXAMPLES / "14_sweep" / "random.yaml"
    grid = write(tmp_path, "grid.yaml", f'include: ["{random}"]\nsweep:\n  strategy: {{use: grid, steps: 4}}\n')
    doc = guide.document([grid])
    assert doc["ok"] is True, doc["problems"]
    assert "the sweep tries 4 values in geometric steps from 0.0001 to 0.01" in notes(doc, "params.lr")
    assert "the sweep tries 4 integers in even steps from 16 to 256" in notes(doc, "params.width")
    assert "$width$ is 128 here; the sweep tries 4 integers in even steps" in notes(doc, "models.net")
    space = notes(doc, "sweep.space")
    assert "lr: 4 values in geometric steps from 0.0001 to 0.01" in space
    assert "width: 4 integers in even steps from 16 to 256" in space
    assert "the sweep runs 16 points" in notes(doc, "sweep.strategy")
    assert "draws" not in notes(doc, "params.lr") + notes(doc, "params.width") + space
    drawn = guide.document([str(random)])
    assert "the sweep draws log uniform from 0.0001 to 0.01" in notes(drawn, "params.lr")
    assert "the sweep draws integers from 16 to 256" in notes(drawn, "params.width")
    assert "lr: log uniform from 0.0001 to 0.01" in notes(drawn, "sweep.space")


def test_a_grid_without_steps_is_flagged_and_one_step_is_one_value(tmp_path):
    random = EXAMPLES / "14_sweep" / "random.yaml"
    stepless = guide.document([write(tmp_path, "stepless.yaml", f'include: ["{random}"]\nsweep: {{strategy: grid}}\n')])
    assert "lr: values in geometric steps from 0.0001 to 0.01" in notes(stepless, "sweep.space")
    assert "sweep_space" in [problem["code"] for problem in entry(stepless, "sweep.space")["problems"]]
    one = write(tmp_path, "one.yaml", f'include: ["{random}"]\nsweep:\n  strategy: {{use: grid, steps: 1}}\n')
    single = guide.document([one])
    assert "lr: only 0.0001" in notes(single, "sweep.space") and "width: only 16" in notes(single, "sweep.space")
    assert "the sweep tries only 0.0001" in notes(single, "params.lr")


def test_generate_is_explained_key_by_key():
    doc = guide.document([str(EXAMPLES / "08_ddpm" / "config.yaml")])
    labels = [item["label"] for item in section(doc, "generate")["entries"]]
    assert labels == ["use", "model", "schedule", "shape", "n", "steps"]
    assert notes(doc, "generate.use").startswith("ddpm_sampler: reverse diffusion")
    assert "@models.unet.ema is the EMA copy of the model unet" in notes(doc, "generate.model")
    assert "@losses.ddpm.schedule is the attribute schedule of the loss ddpm" in notes(doc, "generate.schedule")
    assert "read as @models.unet by losses.ddpm and generate" in notes(doc, "models.unet")
    assert "false: no model predicts" in notes(doc, "train.predicts")


def test_data_references_and_nested_params_are_spelled_out():
    doc = guide.document([str(EXAMPLES / "13_distillation" / "config.yaml")])
    student = notes(doc, "models.student")
    assert "@data.n_classes is n_classes, which the data class exposes once it is fitted" in student
    assert "train.predicts names it" in student
    teacher = notes(doc, "models.teacher")
    assert "$teacher_run$ is runs/teacher" in teacher and "freeze: every parameter is frozen" in teacher
    assert "loads the best checkpoint of the model net from the record runs/teacher" in teacher
    assert "read as $teacher_run$ by models.teacher.weights.run" in notes(doc, "params.teacher_run")
    assert "written by data.yaml" in notes(doc, "data.use")
    accuracy = notes(doc, "metrics.accuracy")
    assert "kalfa compares the output of student (the predicts model) with the target label" in accuracy


def test_a_param_inside_a_nested_value_is_given_its_value(tmp_path):
    text = BASE.replace("params: {width: 16}", "params: {width: 16, sizes: {small: 4}}")
    text = text.replace("hidden: [$width$]", "hidden: [$width$, $sizes.small$]")
    doc = guide.document([write(tmp_path, "c.yaml", text)])
    ae = notes(doc, "models.ae")
    assert "$width$ is 16" in ae and "$sizes.small$ is 4" in ae
    assert "read as $sizes$ by models.ae.hidden" in notes(doc, "params.sizes")


def test_the_defaults_of_a_criterion_follow_the_plan(tmp_path):
    doc = guide.document([write(tmp_path, "c.yaml", BASE)])
    mse = notes(doc, "losses.mse")
    assert "kalfa compares x_hat (the first output of the predicts model ae) with the target price" in mse
    assert "main minimizes it" in mse and "series train/mse, val/mse and test/mse" in mse
    assert "outputs: names what it returns x_hat and z" in notes(doc, "models.ae")
    assert "a shuffled split: 80% train and 20% valid, with no test set" in notes(doc, "data.split")


def test_yaml_that_cannot_be_read_gives_problems_and_no_sections(tmp_path):
    broken = write(tmp_path, "broken.yaml", "data:\n  use: table\n   file: x\n")
    doc = guide.document([broken])
    assert doc["ok"] is False and doc["sections"] == []
    assert [problem["code"] for problem in doc["problems"]] == ["yaml_error"]
    assert "yaml_error" in guide.text(doc)
    partial = guide.document([write(tmp_path, "base.yaml", BASE), broken])
    assert partial["ok"] is False and section(partial, "models")["entries"]


@pytest.mark.parametrize("text", ["models: [1, 2]\n", "rules: {x: 5}\n", "data: 3\n", "train: [1]\n",
                                  "optimizers: {main: {use: adam, over: 3}}\n", "sweep: {space: 3}\n", "- 1\n",
                                  "losses: {a: {use: mse, pred: [1]}}\n", "generate: sampler\n"])
def test_odd_shapes_never_raise(tmp_path, text):
    doc = guide.document([write(tmp_path, "c.yaml", text)])
    assert doc["ok"] is False
    assert isinstance(guide.text(doc), str)
    Console(record=True, width=80).print(guide.renderable(doc))


def test_entry_yaml_is_flow_when_short_and_block_when_long(tmp_path):
    doc = guide.document([write(tmp_path, "c.yaml", BASE)])
    main = 'main: {use: adam, lr: 0.01, over: "@models.ae", minimizes: "@losses.mse"}'
    assert entry(doc, "optimizers.main")["yaml"] == main
    assert entry(doc, "data.features")["yaml"] == 'features: {"x*": standard_scaler}'
    long = guide.document([str(EXAMPLES / "08_ddpm" / "config.yaml")])
    assert entry(long, "losses.ddpm")["yaml"].splitlines() == [
        "ddpm:", "  use: ddpm", '  model: "@models.unet"', "  x: image",
        "  schedule: {use: cosine_beta, steps: $diffusion_steps$}"]


def test_problems_without_an_entry_show_under_their_section(tmp_path):
    doc = guide.document([write(tmp_path, "c.yaml", BASE.replace("  epochs: 5\n", ""))])
    assert "train_unit" in [problem["code"] for problem in doc["problems"]]
    attached = [problem["code"] for item in section(doc, "train")["entries"] for problem in item["problems"]]
    assert "train_unit" not in attached
    plain = guide.text(doc)
    assert plain.index("error train_unit") > plain.index("train\n=====")


def test_a_record_directory_is_explained_from_its_resolved_config(tmp_path, monkeypatch):
    monkeypatch.delitem(sys.modules, PLUGIN, raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(aliases, "SEARCH_DIRS", [])
    base = tmp_path / "base"
    write_parquet(housing_frame(rows=60), base / "housing.parquet")
    (base / f"{PLUGIN}.py").write_text(LEGOS, encoding="utf-8")
    record = tmp_path / "runs" / "first"
    running.run(paths=[write(base, "config.yaml", RECORDED)], record=str(record))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.setattr(aliases, "SEARCH_DIRS", [])
    monkeypatch.setattr(sys, "path", [item for item in sys.path if not item.startswith(str(tmp_path))])
    monkeypatch.delitem(sys.modules, PLUGIN, raising=False)
    doc = guide.document([str(record)])
    assert doc["title"] == str(record)
    assert doc["ok"] and not doc["problems"], doc["problems"]
    assert "file: housing.parquet" in entry(doc, "data.file")["yaml"]
    assert not any(note.startswith(("written by", "merged from")) for part in doc["sections"]
                   for item in part["entries"] for note in item["notes"])
    assert aliases.SEARCH_DIRS == []
