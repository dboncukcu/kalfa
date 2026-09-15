import json
from pathlib import Path

import pandas
import pytest

from helpers import minimal, write_config
from kalfa.api import check, predict, prepare_data, run
from kalfa.config import parse_sets
from kalfa.record import Record
from kalfa.std.common.history import History


pytestmark = pytest.mark.slow


def history_values(record):
    return [{key: value for key, value in line.items() if key not in ("seconds",)} for line in History.read(record)]


def test_a_run_from_prepared_data_matches_a_plain_run(workdir):
    config = minimal()
    config["params"]["epochs"] = 2
    config["data"]["transform"] = ["price > 0"]
    config["record"] = "runs/plain"
    path = write_config(workdir / "cfg.yaml", config)
    data = prepare_data([str(path)], parse_sets([]), out=str(workdir / "prepared"))
    folder = Path(data.directory)
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["kind"] == "data" and manifest["sizes"] == data.sizes and manifest["layout"] == "table"
    assert sorted(path.name for path in folder.glob("*.parquet")) == ["test.parquet", "train.parquet", "valid.parquet"]
    assert (folder / "fitted" / "preprocessors" / "plan.json").exists() and (folder / "data.json").exists()
    train = pandas.read_parquet(folder / "train.parquet")
    assert "row" in train.columns and len(train) == data.sizes["train"]
    plain = run([str(path)], parse_sets([]))
    config["record"] = "runs/from_prepared"
    again = write_config(workdir / "again.yaml", config)
    checked = check([str(again)], parse_sets([]), measure=True, prepared=str(folder))
    assert checked.errors == [] and checked.measured == data.sizes
    started = run([str(again)], parse_sets([]), prepared=str(folder))
    record = Path(started.record)
    assert history_values(plain.record) == history_values(record)
    assert (record / "fitted" / "preprocessors" / "plan.json").exists()
    assert json.loads((record / "manifest.json").read_text())["prepared"] == str(folder)
    assert json.loads((record / "host.json").read_text())["pid"] > 0
    assert Record(record).status()["state"] == "finished" and Record(workdir / "nowhere").status() is None
    prediction = predict(record, data=pandas.read_parquet("housing.parquet").head(10))
    assert len(prediction.table) == 10
    config["data"]["transform"] = ["price > 1"]
    other = write_config(workdir / "other.yaml", config)
    mismatch = check([str(other)], parse_sets([]), prepared=str(folder))
    assert [problem.kind for problem in mismatch.problems] == ["prepared_mismatch"]
    with pytest.raises(Exception, match="no prepared directory"):
        check([str(again)], parse_sets([]), prepared=str(workdir / "runs" / "plain"))
