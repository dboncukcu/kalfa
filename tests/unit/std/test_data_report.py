"""The data report step and the data pipeline drawing."""

import json

import kalfa  # noqa: F401
from helpers import housing_frame
from kalfa.std.feed.kalfa.table import table
from kalfa.std.lego.kalfa.data_report import data_report, stage_entries
from kalfa.std.lego.kalfa.prep import apply, fit
from kalfa.std.loader.kalfa.torch import torch_loader
from kalfa.std.plot.kalfa.data_pipeline import data_pipeline, histogram_columns, stage_lines
from kalfa.std.pre.sklearn.scalers import StandardScaler
from kalfa.std.transform.kalfa.table import derive


def pieces(tmp_path):
    df = housing_frame(rows=40)
    derived = derive(df, "bucket", "x0 > 0")
    kept = derived.query("price > 0")
    train, valid, test = kept.iloc[:28], kept.iloc[28:34], kept.iloc[34:]
    fields = {"x*": {"preprocessors": ["s"]}, "price": {"target": True, "preprocessors": ["s"]}}
    prep = fit(train, fields, {"s": StandardScaler()}, ["bucket"], record=str(tmp_path))
    frames = {f"{name}_frame": apply(part, prep, name) for name, part in
              (("train", train), ("valid", valid), ("test", test))}
    loaders = {f"{name}_loader": torch_loader(table(frame), name, 8) for name, frame in
               ((name.removesuffix("_frame"), frame) for name, frame in frames.items())}
    stages = {"df_0": df, "df_1": derived, "df_2": kept}
    split = {"train_df_0": train, "valid_df_0": valid, "test_df_0": test}
    after = {"train_df_1": train, "valid_df_1": valid, "test_df_1": test}
    return stages, split, after, prep, frames, loaders, train


def test_the_report_reads_every_stage_and_writes_data_json(tmp_path):
    stages, split, after, prep, frames, loaders, train = pieces(tmp_path)
    report = data_report(stages, split, after, [], prep, frames, loaders, record=str(tmp_path))
    assert [entry["stage"] for entry in report["stages"]] == ["df_0", "df_1", "df_2"]
    assert report["stages"][1]["added"] == ["bucket"] and report["stages"][2]["rows"] == len(train) + 12
    assert report["split"] == {"test": 6, "train": 28, "valid": 6} and report["frames"] == []
    assert report["fit"]["features"] == 8 and report["fit"]["targets"] == ["price"]
    assert report["fit"]["preprocessors"] == {"s": 9} and report["sets"]["train"] == {"rows": 28, "features": 8}
    assert report["loaders"]["train"] == {"batches": 4, "size": 8}
    assert json.loads((tmp_path / "data.json").read_text())["split"]["train"] == 28
    assert stage_entries({}) == []


def test_the_drawing_lists_the_stages_and_draws_the_longest_chains(tmp_path):
    stages, split, after, prep, frames, loaders, train = pieces(tmp_path)
    report = data_report(stages, split, after, [], prep, frames, loaders)
    kinds = [kind for kind, _ in stage_lines(report)]
    assert kinds == ["source", "transform", "transform", "split", "fit", "feed", "loaders"]
    chosen = histogram_columns(prep, train, frames["train_frame"], None)
    assert [item.name for item, _, _ in chosen] == ["x0", "x1", "x2"]
    assert histogram_columns(prep, train, frames["train_frame"], ["price"])[0][0].name == "price"
    assert histogram_columns(None, train, frames["train_frame"], None) == []
    data_pipeline(None, [], {}, str(tmp_path), data_report=report, train_df=train,
                  train_frame=frames["train_frame"], prep=prep)
    assert (tmp_path / "plots" / "data_pipeline.png").exists()
    assert data_pipeline(None, [], {}, str(tmp_path), data_report=None) is None
