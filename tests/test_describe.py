import sys

import pytest

import kalfa  # noqa: F401
from conftest import CONFIG_01, CONFIGS, ROOT
from kalfa.api import check, probe
from kalfa.cli import main
from kalfa.describe import render
from kalfa.synthetic import write_churn

sys.path.insert(0, str(ROOT / "tests" / "fixtures"))
from regenerate_recipes import TARGETS  # noqa: E402

HEADS = ("── DATA ", "── MODEL ", "── TRAINING ", "── AFTER ", "── COLUMNS ")


@pytest.mark.parametrize("config, name", TARGETS)
def test_every_reference_config_describes(config, name, capsys):
    code = main(["describe", str(ROOT / config)])
    out = capsys.readouterr().out
    assert code in (0, 1)
    for head in HEADS:
        assert head in out, f"{config} has no {head.strip()} section"
    assert "Traceback" not in out


def test_describe_01(workdir, capsys):
    assert main(["describe", str(CONFIG_01)]) == 0
    out = capsys.readouterr().out
    assert "no problems found" in out
    assert "parquet  housing.parquet" in out and "2 000 rows · 9 columns" in out
    assert "train 1 400  ·  valid 300  ·  test 300" in out
    assert "x*     x0 … x7  (8)  std_scaler     feature" in out
    assert "x ─→ linear_relu 128 ─→ linear_relu 128 ─→ linear 1 ─→ y" in out
    assert "loss_mse      mse            train, valid, test  active at turn 1" in out
    assert "turn ≥ 50" in out and "to_huber    loss := loss_huber" in out
    assert "train/loss_huber < 0.01" in out and "after to_huber" in out
    assert "val/rmse plateau 6" in out and "best monitor=val/rmse" in out
    assert "predict model ─→ predictions.parquet" in out
    assert "the produced widths and the tensor slots need --load" in out


def test_describe_load_counts_parameters_and_slots(workdir, capsys):
    assert main(["describe", str(CONFIG_01), "--load"]) == 0
    out = capsys.readouterr().out
    assert "17 793 parameters" in out
    assert "─→ table ─→ x [128, 8]" in out
    assert "price   double  price  target_std     target   price" in out
    assert "x0      double  x*     std_scaler     feature  x[0]" in out
    assert "the produced widths and the tensor slots need --load" not in out


def test_describe_load_shows_one_hot_widths(workdir, capsys):
    write_churn(workdir / "churn.parquet")
    assert main(["describe", str(CONFIGS / "02_mlp_classification.yaml"), "--load", "--section", "columns"]) == 0
    out = capsys.readouterr().out
    assert "onehot  (3 columns)" in out and "x[4 … 6]" in out
    assert "onehot  (2 columns)" in out and "x[7 … 8]" in out
    assert "churned  large_string  churned  label" in out
    assert "── DATA " not in out


def test_describe_shows_the_target_table_and_the_slots(tmp_path, monkeypatch, capsys):
    from kalfa.synthetic import write_scores

    monkeypatch.chdir(tmp_path)
    write_scores(tmp_path / "scores.parquet")
    assert main(["describe", str(CONFIGS / "15_multi_target.yaml")]) == 0
    out = capsys.readouterr().out
    assert "output wire" in out and "predicts the target fields" in out
    assert "y_hat" in out and "y_a, y_b, y_c" in out
    assert "target   y_hat[0]" in out and "target   z_hat[0]" in out
    assert "y_hat ─→ y_*" in out and "compares" in out


def test_describe_sections_and_wiring(workdir, capsys):
    assert main(["describe", str(CONFIG_01), "--section", "model"]) == 0
    out = capsys.readouterr().out
    assert "── MODEL " in out and "── DATA " not in out and "── WIRING " not in out
    assert main(["describe", str(CONFIG_01), "--wiring"]) == 0
    out = capsys.readouterr().out
    assert "── WIRING " in out and "── DATA " in out
    assert "training.init: device ← device" in out
    assert main(["describe", str(CONFIG_01), "--section", "wiring"]) == 0
    out = capsys.readouterr().out
    assert "── WIRING " in out and "── DATA " not in out


def test_describe_reports_problems_and_keeps_going(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["describe", str(CONFIG_01)]) == 1
    out = capsys.readouterr().out
    assert "[source_missing]" in out and "── TRAINING " in out
    assert "sizes unknown" in out and "x ─→ linear_relu 128" in out


def test_describe_without_a_document_says_so(workdir, capsys):
    path = workdir / "unshaped.yaml"
    path.write_text("data: {}\n")
    assert main(["describe", str(path)]) == 1
    out = capsys.readouterr().out
    assert "the config could not be shaped" in out


def test_probe_builds_the_models(workdir):
    prepared = check([str(CONFIG_01)])
    assert prepared.errors == []
    found = probe(prepared.document)
    assert found.sizes == {"train": 1400, "valid": 300, "test": 300}
    assert found.features == 8
    assert found.parameters["model"] == (17793, 17793)
    assert found.shapes["model"]["y"] == (128, 1)
    assert found.notes == {}


def test_colors_do_not_change_the_layout(workdir):
    from kalfa.cli import Style
    from kalfa.describe import visible

    prepared = check([str(CONFIG_01)])
    painted = render(prepared, Style(True), None, None, width=96)
    plain = render(prepared, Style(False), None, None, width=96)
    assert "\x1b[36m" in painted and "\x1b[2m" in painted
    assert visible(painted) == plain


def test_params_are_a_block(workdir, capsys):
    config = (CONFIG_01).read_text().replace("  epochs: 100\n", "  epochs: 100\n  hidden: 256\n  dropout: 0.1\n"
                                                                "  alpha: 1\n  beta: 0.5\n")
    path = workdir / "many.yaml"
    path.write_text(config)
    assert main(["describe", str(path), "--section", "summary"]) == 0
    out = capsys.readouterr().out
    assert "  params\n" in out and "    epochs" in out and "hidden" in out
    assert "  device      cpu (no device key)" in out
    assert "  seed        7\n" in out


def test_render_takes_a_section_list(workdir):
    class Plain:
        def __getattr__(self, name):
            return lambda text: text

    prepared = check([str(CONFIG_01)])
    text = render(prepared, Plain(), ["data"], None, width=100)
    assert text.startswith("\n── DATA ") and "── MODEL " not in text
