import inspect
import json
from pathlib import Path

import pandas
import pytest

import kalfa
from helpers import housing_frame
from kalfa import cli
from kalfa.commands import sweep as sweeping
from kalfa.core.aliases import ALIASES


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "DOCS.md"
EXAMPLES = ROOT / "examples"

CONFIG = """
params:
  lr: 1.0e-3
device: cpu
data:
  use: table
  file: housing.parquet
  split: [0.7, 0.15, 0.15]
  features: {"x*": standard_scaler}
  targets: {price: standard_scaler}
  batch: 32
models:
  net: {use: mlp, hidden: [16], out: 1}
losses:
  mse: mse
metrics:
  rmse: rmse
optimizers:
  main: {use: adam, lr: $lr$, over: "@models.net", minimizes: "@losses.mse"}
train:
  epochs: 20
  checkpoint: {watch: val/rmse}
"""


def blocks(text, language):
    found = []
    lines = None
    for line in text.splitlines():
        stripped = line.strip()
        if lines is None:
            if stripped == f"```{language}":
                lines = []
        elif stripped.startswith("```"):
            found.append("\n".join(lines) + "\n")
            lines = None
        else:
            lines.append(line)
    return found


def section(text, heading):
    lines = text.splitlines()
    start = lines.index(heading) + 1
    level = len(heading) - len(heading.lstrip("#"))
    inside = False
    for position in range(start, len(lines)):
        line = lines[position]
        if line.lstrip().startswith("```"):
            inside = not inside
        elif not inside and line.startswith("#") and len(line) - len(line.lstrip("#")) <= level:
            return "\n".join(lines[start:position]) + "\n"
    return "\n".join(lines[start:]) + "\n"


def home(directory, text=CONFIG, frame=None):
    directory.mkdir(parents=True, exist_ok=True)
    (frame if frame is not None else housing_frame()).to_parquet(directory / "housing.parquet", index=False)
    config = directory / "config.yaml"
    config.write_text(text, encoding="utf-8")
    return config


@pytest.fixture
def saved_aliases():
    saved = dict(ALIASES)
    yield
    ALIASES.clear()
    ALIASES.update(saved)


def test_python_api_block_of_docs_runs(tmp_path, monkeypatch):
    home(tmp_path)
    housing_frame(rows=20, seed=3).drop(columns=["price"]).to_parquet(tmp_path / "new.parquet", index=False)
    monkeypatch.chdir(tmp_path)
    code = blocks(section(DOCS.read_text(encoding="utf-8"), "## Python API"), "python")[0]
    space = {"__name__": "docs_api"}
    exec(compile(code, str(DOCS), "exec"), space)
    assert space["checked"].ok, space["checked"].text()
    assert space["record"].state == "done"
    assert len(space["table"]) == 20


def test_resolve_gives_an_import_path_string_for_std_names():
    assert kalfa.resolve("mlp") == "kalfa.std.models.Mlp"
    assert isinstance(kalfa.resolve("adam"), str)


def test_lego_and_register_take_slot_and_doc_only_by_keyword(saved_aliases):
    assert inspect.signature(kalfa.lego).parameters["slot"].kind is inspect.Parameter.KEYWORD_ONLY
    with pytest.raises(TypeError):
        kalfa.lego("review_name", "loss")
    with pytest.raises(TypeError):
        kalfa.register("review_name", "torch.nn.MSELoss", "loss")


def test_unknown_use_message_points_at_kalfa_legos(tmp_path):
    config = home(tmp_path, CONFIG.replace("use: mlp,", "use: mlpp,"))
    checked = kalfa.check(str(config))
    problem = next(problem for problem in checked.problems if problem.code == "unknown_use")
    assert "kalfa legos lists the names" in str(problem) and "Aşama" not in str(problem)


def test_plan_json_keeps_no_filled_predicts_or_evaluate(tmp_path):
    config = home(tmp_path)
    record = tmp_path / "record"
    kalfa.run(str(config), sets={"train.epochs": 1}, record=str(record))
    plan = json.loads((record / "plan.json").read_text(encoding="utf-8"))
    assert "predicts" not in plan["train"] and "evaluate" not in plan["train"]
    assert plan["seed"] is None
    assert plan["nodes"]["losses.mse"]["keys"] == {"pred": "net", "target": "price"}


def test_distill_names_its_label_field_without_a_slot_conflict(tmp_path):
    text = CONFIG.replace("  net: {use: mlp, hidden: [16], out: 1}",
                          "  student: {use: mlp, hidden: [8], out: 1}\n  teacher: {use: mlp, hidden: [8], out: 1}")
    text = text.replace("  mse: mse", '  kd: {use: distill, student: "@models.student", teacher: "@models.teacher", '
                                      'label: price}')
    text = text.replace('over: "@models.net", minimizes: "@losses.mse"', 'over: "@models.student", minimizes: '
                                                                         '"@losses.kd"')
    text = text.replace("  rmse: rmse", '  rmse: {use: rmse, pred: student}')
    config = home(tmp_path, text)
    checked = kalfa.check(str(config))
    assert checked.ok and not checked.errors, checked.text()
    parameters = inspect.signature(kalfa.std.objectives.Distill).parameters
    assert "label" in parameters and "target" not in parameters


def test_step_slot_keys_that_are_constructor_params_pass(tmp_path):
    text = CONFIG.replace("  epochs: 20", "  epochs: 20\n  step: {use: supervised, grad_clip: 1.0, amp: false}")
    checked = kalfa.check(str(home(tmp_path, text)))
    assert checked.ok and not checked.problems, checked.text()
    assert "grad_clip" in inspect.signature(kalfa.std.steps.Supervised).parameters


def test_reference_sweep_has_four_points(tmp_path):
    text = blocks(section(DOCS.read_text(encoding="utf-8"), "## Reference config"), "yaml")[0]
    config = home(tmp_path, text)
    plan = sweeping.plan_of(paths=[str(config)], record=str(tmp_path / "sweep"))
    assert plan.count == 4


def test_dollar_dollar_is_a_literal_dollar(tmp_path):
    config = home(tmp_path, CONFIG + "record: runs/a$$b\n")
    checked = kalfa.check(str(config))
    assert checked.ok and checked.config.data["record"] == "runs/a$b"


def test_table_block_of_docs_checks_and_trains(tmp_path):
    frame = housing_frame()
    frame["size"] = frame["x0"].abs() * 100.0 + 1.0
    frame["notes"] = "n"
    frame["id"] = range(len(frame))
    data = blocks(section(DOCS.read_text(encoding="utf-8"), "#### table"), "yaml")[0]
    rest = CONFIG.split("models:", 1)[1]
    config = home(tmp_path, "device: cpu\nparams: {lr: 1.0e-3}\n" + data + "models:" + rest, frame)
    checked = kalfa.check(str(config))
    assert checked.ok, checked.text()
    record = tmp_path / "record"
    kalfa.run(str(config), sets={"train.epochs": 1}, record=str(record))
    assert json.loads((record / "manifest.json").read_text(encoding="utf-8"))["status"] == "done"


def test_named_python_api_objects_exist():
    for name in ("text_of", "markdown_of", "write"):
        assert callable(getattr(kalfa.summary, name))
    for name in ("summary_of", "series_of", "events_of", "records_under", "sweep_points"):
        assert callable(getattr(kalfa.records, name))
    for name in ("records", "generate", "export", "stop", "sweep", "report"):
        assert callable(getattr(kalfa.api, name))
    assert list(inspect.signature(kalfa.Monitor).parameters) == ["stream", "quiet"]
    assert list(inspect.signature(kalfa.Problem).parameters) == ["code", "message", "path", "hint", "severity"]
    assert issubclass(kalfa.core.errors.ProblemsError, kalfa.KalfaError)


def test_examples_without_config_or_data_script():
    folders = [path for path in EXAMPLES.iterdir() if (path / "example.json").is_file()]
    assert {path.name for path in folders if not (path / "config.yaml").is_file()} == {"14_sweep"}
    assert {path.name for path in folders if not (path / "make_data.py").is_file()} == {"11_kfold", "12_resume",
                                                                                         "14_sweep"}


def test_demo_prediction_columns_match_the_docs():
    demo = ROOT / "runs" / "demo"
    if not demo.is_dir():
        pytest.skip("no demo root")
    window = pandas.read_parquet(demo / "03_timeseries_window" / "energy" / "predictions.parquet")
    assert {"row", "load_0", "load_5", "pred_net_0", "pred_net_5"} <= set(window.columns), list(window.columns)
    churn = pandas.read_parquet(demo / "02_mlp_classification" / "churn" / "predictions.parquet")
    assert any(column.startswith("prob_") for column in churn.columns), list(churn.columns)
    housing = pandas.read_parquet(demo / "01_mlp_regression" / "housing" / "predictions.parquet")
    assert {"row", "price", "pred_net", "raw_net"} <= set(housing.columns), list(housing.columns)


def test_bare_init_and_schedule_names_are_calls(tmp_path):
    text = CONFIG.replace("hidden: [16], out: 1}", "hidden: [16], out: 1, init: xavier_init}")
    config = home(tmp_path, text)
    checked = kalfa.check(str(config))
    assert checked.ok, checked.text()
    assert checked.plan["nodes"]["models.net.init"]["use"] == "kalfa.std.init.xavier_init"
    record = tmp_path / "record"
    kalfa.run(str(config), sets={"train.epochs": 1}, record=str(record))
    assert json.loads((record / "manifest.json").read_text(encoding="utf-8"))["status"] == "done"
    wrong = kalfa.check(str(home(tmp_path / "wrong", text.replace("init: xavier_init", "init: xavier_inti"))))
    problem = next(problem for problem in wrong.errors if problem.code == "unknown_use")
    assert problem.path == "models.net.init" and "xavier_init" in (problem.hint or ""), wrong.text()
    bare = text.replace('minimizes: "@losses.mse"}', 'minimizes: "@losses.mse", schedule: cosine}')
    scheduled = kalfa.check(str(home(tmp_path / "schedule", bare)))
    found = [(problem.code, problem.path) for problem in scheduled.errors]
    assert found == [("missing_key", "optimizers.main.schedule")], scheduled.text()


def test_run_on_a_record_directory_reuses_its_stamped_record(tmp_path, monkeypatch, capsys):
    config = home(tmp_path, CONFIG + "record: runs/again_$datetime$\n")
    monkeypatch.chdir(tmp_path)
    first = kalfa.run(str(config), sets={"train.epochs": 1})
    capsys.readouterr()
    code = cli.main(["run", first.record, "--quiet"])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    again = Path(captured.out.strip().splitlines()[-1])
    assert again != Path(first.record) and again.name.startswith(Path(first.record).name + "_")
    assert (again / "manifest.json").is_file() and "already holds a run" in captured.err
