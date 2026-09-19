import json

import numpy
import pandas
import torch
from cirak.registry import registry

from helpers import build
from kalfa.std import STD_URIS
from kalfa.std.pre.base import Grouped


DATA_LEGOS = ["/lego/kalfa/apply", "/lego/kalfa/apply_frames", "/lego/kalfa/csv_header", "/lego/kalfa/data_report",
              "/lego/kalfa/fit", "/lego/kalfa/fit_frames", "/lego/kalfa/given_sizes",
              "/lego/kalfa/image_folder_header", "/lego/kalfa/kfold_sizes", "/lego/kalfa/parquet_header",
              "/lego/kalfa/pixel_features", "/lego/kalfa/prepared_header", "/lego/kalfa/prepared_sizes",
              "/lego/kalfa/ratio_sizes", "/lego/kalfa/read_frames", "/lego/kalfa/read_prep",
              "/lego/kalfa/text_lines_header", "/lego/kalfa/transform_set"]


def source_table():
    return pandas.DataFrame({"raw_0": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "raw_1": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
                             "junk": list("abcdef"), "site": ["s0", "s1", "s0", "s1", "s0", "s1"],
                             "y": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]})


def pipeline(record=None):
    df_0 = source_table()
    df_1 = build("/transform/kalfa/rename", pattern="^raw_(\\d)$", to="num_\\1")(df_0)
    df_2 = build("/transform/kalfa/drop", columns=["junk"])(df_1)
    df_10 = build("/transform/kalfa/filter", query="num_0 < 6")(df_2)
    split = build("/split/kalfa/sequential", df=df_10, ratios=[0.6, 0.2, 0.2])
    after = {"train": build("/lego/kalfa/transform_set", df=split["train"], set="train",
                            transforms=[build("/transform/kalfa/filter", query="num_0 > 1")]),
             "valid": split["valid"], "test": split["test"]}
    frames = build("/lego/kalfa/fit_frames", df=after["train"],
                   frames=[build("/frame/kalfa/group_statistic", by="site", column="num_0")])
    framed = {name: build("/lego/kalfa/apply_frames", df=part, frames=frames) for name, part in after.items()}
    templates = {"std": build("/pre/sklearn/standard_scaler"), "onehot": build("/pre/kalfa/one_hot"),
                 "impute": build("/pre/kalfa/simple_imputer", indicator=True)}
    fields = {"num_*": {"preprocessors": ["std"]}, "site": {"preprocessors": ["onehot"]},
              "*_by_site": {"preprocessors": ["impute", "std"]}, "y": {"target": True, "preprocessors": ["std"]}}
    prep = build("/lego/kalfa/fit", df=framed["train"], fields=fields, preprocessors=templates, drop=[])
    applied = {name: build("/lego/kalfa/apply", df=part, prep=prep, set=name) for name, part in framed.items()}
    datasets = {name: build("/feed/kalfa/table", frame=part, frames=applied) for name, part in applied.items()}
    loaders = {name: build("/loader/kalfa/torch", data=data, set=name, size=2, eval_size=1)
               for name, data in datasets.items()}
    return {"stages": {"df_0": df_0, "df_1": df_1, "df_2": df_2, "df_10": df_10},
            "split": {f"{name}_df_0": part for name, part in split.items()},
            "after": {f"{name}_df_1": part for name, part in after.items()},
            "fitted": frames, "prep": prep,
            "frames": {f"{name}_frame": part for name, part in applied.items()},
            "loaders": {f"{name}_loader": loader for name, loader in loaders.items()},
            "transforms": [{"lego": "/transform/kalfa/rename", "with": {"pattern": "^raw_(\\d)$", "to": "num_\\1"}},
                           {"lego": "/transform/kalfa/drop", "with": {"columns": ["junk"]}},
                           {"lego": "/transform/kalfa/filter", "with": {"query": "num_0 < 6"}}],
            "set_transforms": {"train": [{"lego": "/transform/kalfa/filter", "with": {"query": "num_0 > 1"}}],
                               "valid": []},
            "record": record}


def test_every_data_step_of_the_lego_pack_is_registered():
    assert [uri for uri in DATA_LEGOS if uri not in STD_URIS] == []
    assert registry.facts("/lego/kalfa/data_report").returns == "data_report"
    assert registry.facts("/lego/kalfa/data_report").bus == {"record": "record"}


def test_data_report_describes_every_stage_of_the_data_block(tmp_path):
    inputs = pipeline(record=str(tmp_path))
    report = build("/lego/kalfa/data_report", **inputs)
    assert list(report) == ["stages", "set_transforms", "split", "after_set_transforms", "frames", "fit", "sets",
                            "loaders"]
    assert report["stages"] == [
        {"stage": "df_0", "rows": 6, "columns": 5},
        {"stage": "df_1", "rows": 6, "columns": 5, "added": ["num_0", "num_1"], "removed": ["raw_0", "raw_1"],
         "call": {"uri": "/transform/kalfa/rename", "params": {"pattern": "^raw_(\\d)$", "to": "num_\\1"}}},
        {"stage": "df_2", "rows": 6, "columns": 4, "added": [], "removed": ["junk"],
         "call": {"uri": "/transform/kalfa/drop", "params": {"columns": ["junk"]}}},
        {"stage": "df_10", "rows": 5, "columns": 4, "added": [], "removed": [],
         "call": {"uri": "/transform/kalfa/filter", "params": {"query": "num_0 < 6"}}}]
    assert report["set_transforms"] == {"train": [{"uri": "/transform/kalfa/filter", "params": {"query": "num_0 > 1"}}]}
    assert report["split"] == {"train": 3, "valid": 1, "test": 1}
    assert list(report["split"]) == ["test", "train", "valid"]
    assert report["after_set_transforms"] == {"train": 2, "valid": 1, "test": 1}
    assert report["frames"] == ["GroupStatistic"]
    assert isinstance(inputs["prep"].fitted["std"], Grouped)
    assert report["fit"] == {"fields": 5, "features": 6, "targets": ["y"],
                             "preprocessors": {"std": 4, "onehot": 1, "impute": 1},
                             "extras": ["num_0_mean_by_site_missing"]}
    assert report["sets"] == {"train": {"rows": 2, "features": 6}, "valid": {"rows": 1, "features": 6},
                              "test": {"rows": 1, "features": 6}}
    assert report["loaders"] == {"train": {"batches": 1, "size": 2}, "valid": {"batches": 1, "size": 1},
                                 "test": {"batches": 1, "size": 1}}
    assert json.loads((tmp_path / "data.json").read_text(encoding="utf-8")) == report


def test_data_report_without_a_record_writes_nothing_and_tolerates_missing_notes(tmp_path):
    inputs = pipeline()
    inputs["transforms"] = inputs["transforms"][:1]
    inputs["set_transforms"] = None
    report = build("/lego/kalfa/data_report", **inputs)
    assert [entry.get("call") for entry in report["stages"]] == [
        None, {"uri": "/transform/kalfa/rename", "params": {"pattern": "^raw_(\\d)$", "to": "num_\\1"}}, None, None]
    assert report["set_transforms"] == {}
    assert not (tmp_path / "data.json").exists()
    bare = build("/lego/kalfa/data_report", stages={"df_0": source_table()}, split={}, after={}, fitted=None,
                 prep=inputs["prep"], frames={}, loaders={})
    assert bare["stages"] == [{"stage": "df_0", "rows": 6, "columns": 5}]
    assert (bare["split"], bare["after_set_transforms"], bare["frames"], bare["sets"], bare["loaders"]) == (
        {}, {}, [], {}, {})
    notes = build("/lego/kalfa/data_report", stages={}, split={}, after={}, fitted=[], prep=inputs["prep"], frames={},
                  loaders={}, transforms=[{"lego": "/transform/kalfa/drop"}], set_transforms={"valid": [{}]})
    assert notes["stages"] == []
    assert notes["set_transforms"] == {"valid": [{"uri": None, "params": {}}]}


def test_data_report_counts_a_stream_stage_as_unknown_rows_and_a_dataset_source_by_fields(root):
    stream = build("/source/kalfa/parquet_stream", path=str(root / "housing.parquet"))
    images = build("/source/kalfa/image_folder", path=str(root / "images"))
    prep = pipeline()["prep"]
    report = build("/lego/kalfa/data_report", stages={"df_1": images, "df_0": stream}, split={"train_df_0": stream},
                   after={"train_df_1": images.subset([0, 1, 2])}, fitted=[], prep=prep, frames={}, loaders={})
    assert report["stages"] == [{"stage": "df_0", "rows": None, "columns": 9},
                                {"stage": "df_1", "rows": 64, "columns": 2, "added": ["image", "label"],
                                 "removed": [f"x{position}" for position in range(8)] + ["price"]}]
    assert report["split"] == {"train": None}
    assert report["after_set_transforms"] == {"train": 3}


def test_pixel_features_pools_images_to_size_by_size_and_flattens():
    extract = build("/lego/kalfa/pixel_features", size=2)
    images = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    features = extract(images)
    assert features.shape == (1, 4)
    assert features[0].tolist() == [2.5, 4.5, 10.5, 12.5]
    color = torch.rand(3, 3, 8, 8, generator=torch.Generator().manual_seed(0))
    pooled = build("/lego/kalfa/pixel_features")(color)
    assert pooled.shape == (3, 48)
    numpy.testing.assert_allclose(pooled[1, :4].numpy(), color[1, 0].reshape(4, 2, 4, 2).mean(dim=(1, 3))[0].numpy(),
                                  rtol=1e-6)
    assert build("/lego/kalfa/pixel_features", size=1)(color).shape == (3, 3)
    torch.testing.assert_close(build("/lego/kalfa/pixel_features", size=1)(color)[:, 0], color[:, 0].mean(dim=(1, 2)))
