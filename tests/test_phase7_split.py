import importlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "src" / "kalfa"
PACKAGES = {"kalfa.records.stats": ROOT / "records" / "stats", "kalfa.commands.report": ROOT / "commands" / "report"}
STATS_NAMES = ("PREDICTIONS", "IMAGE_TYPES", "IMAGE_SUFFIXES", "load_table", "read_predictions", "prediction_tables",
               "prediction_stats", "prediction_pairs", "task_pairs", "plan_targets", "plan_pairings", "spread_member",
               "kind_of", "classes_for", "fitted_classes", "read_fitted_fields", "read_plan", "label_text", "labels",
               "ordered_labels", "score_labels", "label_columns", "positive_column", "histogram", "regression_stats",
               "classification_stats", "values_stats", "best_threshold", "image_column")
REPORT_NAMES = ("REPORT", "DETAILS", "render", "write_report")


def test_stats_and_report_are_packages_of_small_modules():
    for name, folder in PACKAGES.items():
        module = importlib.import_module(name)
        assert Path(module.__file__).resolve() == (folder / "__init__.py").resolve(), name
        modules = sorted(folder.glob("*.py"))
        assert len(modules) >= 3, (name, modules)
        long = [(path.name, count) for path in modules
                if (count := len(path.read_text(encoding="utf-8").splitlines())) >= 450]
        assert not long, (name, long)
    assert not (ROOT / "records" / "stats.py").exists() and not (ROOT / "commands" / "report.py").exists()


def test_every_name_callers_use_stays_importable():
    stats = importlib.import_module("kalfa.records.stats")
    report = importlib.import_module("kalfa.commands.report")
    missing = [f"stats.{name}" for name in STATS_NAMES if not hasattr(stats, name)]
    missing += [f"report.{name}" for name in REPORT_NAMES if not hasattr(report, name)]
    assert not missing, missing
    from kalfa.records import prediction_pairs, prediction_stats, prediction_tables, read_predictions
    assert prediction_stats is stats.prediction_stats and read_predictions is stats.read_predictions
    assert prediction_pairs is stats.prediction_pairs and prediction_tables is stats.prediction_tables
