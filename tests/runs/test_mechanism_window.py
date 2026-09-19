import json
from pathlib import Path

import numpy
import pandas
import pytest

from data import energy_frame
from helpers import config_path, copied
from kalfa.api import check, resume
from kalfa.config import parse_sets
from kalfa.std.common.history import History


pytestmark = pytest.mark.slow


@pytest.fixture
def window(trained, at_root):
    return trained("window")


def test_the_group_column_must_stay_a_carried_column(at_root):
    prepared = check([config_path("window")], parse_sets([]))
    assert prepared.problems == [] and prepared.sizes == {"train": 252, "valid": 54, "test": 54}
    assert prepared.document["flow"]["data"]["params"]["prep"]["params"]["spectators"] == ["site_id"]
    dropped = check([config_path("window")], parse_sets(["data.drop=[site_id]"]))
    assert [problem.kind for problem in dropped.errors] == ["dropped_column_ref"] * 2
    fielded = check([config_path("window")], parse_sets(["data.fields.site_id={}"]))
    assert {problem.kind for problem in fielded.errors} == {"column_in_fields", "dtype_unsupported"}


def test_the_history_reports_every_set_and_no_checkpoint_is_written(window):
    history = History.read(window.record)
    assert [list(line) for line in history] == [["turn", "global_step", "train/loss_mse", "train/rmse", "val/loss_mse",
                                                 "val/rmse", "test/loss_mse", "test/rmse", "lr/model",
                                                 "minimizes/model", "seconds", "rules"]] * 2
    assert not (Path(window.record) / "checkpoints").exists()
    assert (Path(window.record) / "final" / "state.pt").exists()
    data = json.loads((Path(window.record) / "data.json").read_text())
    assert data["loaders"]["train"]["batches"] == 13 and [line["global_step"] for line in history] == [13, 26]


def test_the_predictions_hold_a_horizon_per_row_and_the_site(window):
    table = pandas.read_parquet(Path(window.record) / "predictions.parquet")
    assert list(table.columns) == ["row", *(f"load_{step}" for step in range(4)),
                                   *(f"raw_y_{step}" for step in range(4)), *(f"pred_y_{step}" for step in range(4)),
                                   "site_id"]
    assert len(table) == 45 and table["site_id"].value_counts().to_dict() == {"site_0": 15, "site_1": 15, "site_2": 15}
    source = energy_frame(sites=3, steps=120)
    rows = table["row"].to_numpy()
    for step in range(4):
        numpy.testing.assert_allclose(table[f"load_{step}"].to_numpy(), source.loc[rows + step, "load"].to_numpy(),
                                      rtol=1e-5)
    assert (source.loc[rows, "site_id"].to_numpy() == table["site_id"].to_numpy()).all()
    plots = sorted(path.name for path in (Path(window.record) / "plots").iterdir())
    assert plots == ["forecast.png", "loss_curve.png"]


def test_resume_without_checkpoints_starts_from_the_final_state(window, tmp_path):
    copy = copied(window.record, tmp_path / "window_copy")
    resumed = resume(copy, parse_sets(["training.epochs=3"]), when="resumed")
    assert [line["turn"] for line in History.read(resumed.record)] == [3]
    note = json.loads((Path(resumed.record) / "resume.json").read_text())
    assert note == {"resume_from": str(copy), "checkpoint": str(copy / "final" / "state.pt")}
