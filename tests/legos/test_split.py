import json
import re
from pathlib import Path

import numpy
import pandas
import pytest
from cirak.registry import registry

from data import housing_frame
from helpers import build
from kalfa.std import STD_URIS
from kalfa.std.common.samples import Samples
from kalfa.std.common.stream import ParquetChunks, Stream
from kalfa.std.source.kalfa.samples import TextLines


SPLITS = {"/split/kalfa/random": "/lego/kalfa/ratio_sizes", "/split/kalfa/sequential": "/lego/kalfa/ratio_sizes",
          "/split/kalfa/kfold": "/lego/kalfa/kfold_sizes", "/split/kalfa/given": "/lego/kalfa/given_sizes",
          "/split/kalfa/prepared": "/lego/kalfa/prepared_sizes"}
SETS = ["train", "valid", "test"]


def table(rows=20, seed=1):
    generator = numpy.random.default_rng(seed)
    return pandas.DataFrame({"value": generator.normal(size=rows), "group": [f"g{position % 3}" for position in
                                                                              range(rows)]},
                            index=range(100, 100 + rows))


def labels(df):
    return df.index.tolist()


def write_prepared(folder, sets, layout="table"):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    sizes = {}
    for name, part in sets.items():
        sizes[name] = len(part)
        if layout == "samples":
            (folder / f"{name}.json").write_text(json.dumps([int(position) for position in part]))
        else:
            part.rename_axis("row").reset_index().to_parquet(folder / f"{name}.parquet", index=False)
    (folder / "manifest.json").write_text(json.dumps({"kind": "data", "layout": layout, "sets": list(sets),
                                                      "sizes": sizes, "header": {}}))
    return folder


def test_split_catalog_is_the_five_splits():
    assert sorted(uri for uri in STD_URIS if uri.startswith("/split/")) == sorted(SPLITS)


@pytest.mark.parametrize("uri", sorted(SPLITS))
def test_split_returns_the_three_sets_and_names_its_sizes_helper(uri):
    assert registry.facts(uri).returns == SETS
    assert registry.facts(uri).get("sizes") == SPLITS[uri]


def test_split_aliases_and_table_facts():
    aliases = registry.aliases()
    assert aliases["random_split"] == "/split/kalfa/random"
    assert aliases["sequential"] == "/split/kalfa/sequential"
    assert aliases["kfold"] == "/split/kalfa/kfold"
    assert aliases["given"] == "/split/kalfa/given"
    assert "prepared" not in aliases
    assert registry.facts("/split/kalfa/random").get("needs_table") is True
    assert registry.facts("/split/kalfa/kfold").get("needs_table") is True
    assert registry.facts("/split/kalfa/sequential").get("needs_table") is None
    assert registry.facts("/split/kalfa/sequential").refs == {"group": "column"}


def test_random_split_shuffles_with_the_seed_and_cuts_by_ratios():
    df = table(20)
    parts = build("/split/kalfa/random", df=df, ratios=[0.7, 0.15, 0.15], seed=3)
    assert list(parts) == SETS
    order = numpy.random.default_rng(3).permutation(20)
    assert order.tolist() == [16, 12, 18, 8, 3, 13, 15, 10, 6, 1, 2, 11, 17, 0, 14, 9, 4, 7, 5, 19]
    assert labels(parts["train"]) == [100 + position for position in order[:14]]
    assert labels(parts["valid"]) == [100 + position for position in order[14:17]]
    assert labels(parts["test"]) == [100 + position for position in order[17:]]
    assert labels(parts["train"])[:4] == [116, 112, 118, 108]
    assert labels(parts["valid"]) == [114, 109, 104]
    assert labels(parts["test"]) == [107, 105, 119]
    assert sorted(labels(parts["train"]) + labels(parts["valid"]) + labels(parts["test"])) == labels(df)
    pandas.testing.assert_frame_equal(parts["train"], df.iloc[order[:14]])


def test_random_split_is_reproducible_by_seed_only():
    df = table(20)
    first = build("/split/kalfa/random", df=df, ratios=[0.5, 0.25, 0.25], seed=11)
    second = build("/split/kalfa/random", df=df, ratios=[0.5, 0.25, 0.25], seed=11)
    other = build("/split/kalfa/random", df=df, ratios=[0.5, 0.25, 0.25], seed=12)
    for name in SETS:
        assert labels(first[name]) == labels(second[name])
    assert [len(first[name]) for name in SETS] == [10, 5, 5]
    assert labels(other["train"]) != labels(first["train"])


def test_random_split_rounds_the_cuts_half_to_even():
    df = table(10)
    parts = build("/split/kalfa/random", df=df, ratios=[0.5, 0.25, 0.25], seed=0)
    assert [len(parts[name]) for name in SETS] == [5, 2, 3]
    parts = build("/split/kalfa/random", df=table(3), ratios=[0.5, 0.5, 0.0], seed=0)
    assert [len(parts[name]) for name in SETS] == [2, 1, 0]
    assert list(parts["test"].columns) == ["value", "group"]


def test_random_split_of_a_dataset_source_subsets_positions(root):
    samples = build("/source/kalfa/text_lines", path=str(root / "text.txt"))
    parts = build("/split/kalfa/random", df=samples, ratios=[0.5, 0.25, 0.25], seed=7)
    order = numpy.random.default_rng(7).permutation(200)
    for name, part in parts.items():
        assert isinstance(part, Samples)
        assert part.source is samples.source
    numpy.testing.assert_array_equal(parts["train"].positions, order[:100])
    numpy.testing.assert_array_equal(parts["valid"].positions, order[100:150])
    numpy.testing.assert_array_equal(parts["test"].positions, order[150:])
    assert parts["train"][0] == samples[int(order[0])]


def test_random_split_refuses_a_stream(root):
    stream = build("/source/kalfa/parquet_stream", path=str(root / "housing.parquet"))
    with pytest.raises(ValueError, match=re.escape("a stream source cannot be shuffled; the lazy set splits with "
                                                   "sequential or given")):
        build("/split/kalfa/random", df=stream, ratios=[0.7, 0.15, 0.15], seed=0)


def test_ratios_are_validated():
    df = table(10)
    with pytest.raises(ValueError, match=re.escape("ratios must have three entries (train, valid, test), got "
                                                   "[0.5, 0.5]")):
        build("/split/kalfa/random", df=df, ratios=[0.5, 0.5], seed=0)
    with pytest.raises(ValueError, match=re.escape("ratios must be non negative and sum to 1, got [0.5, 0.6, 0.0]")):
        build("/split/kalfa/sequential", df=df, ratios=[0.5, 0.6, 0.0])
    with pytest.raises(ValueError, match=re.escape("ratios must be non negative and sum to 1, got [1.2, -0.2, 0.0]")):
        build("/split/kalfa/sequential", df=df, ratios=[1.2, -0.2, 0.0])


def test_sequential_cuts_the_rows_in_their_order():
    df = table(10)
    parts = build("/split/kalfa/sequential", df=df, ratios=[0.5, 0.25, 0.25])
    assert labels(parts["train"]) == [100, 101, 102, 103, 104]
    assert labels(parts["valid"]) == [105, 106]
    assert labels(parts["test"]) == [107, 108, 109]
    pandas.testing.assert_frame_equal(parts["train"], df.iloc[:5])
    whole = build("/split/kalfa/sequential", df=df, ratios=[1.0, 0.0, 0.0])
    assert labels(whole["train"]) == labels(df)
    assert whole["valid"].empty and whole["test"].empty
    assert list(whole["valid"].columns) == ["value", "group"]


def test_sequential_with_a_group_cuts_every_group_on_its_own():
    df = pandas.DataFrame({"value": numpy.arange(10.0), "g": ["b"] * 4 + ["a"] * 6})
    parts = build("/split/kalfa/sequential", df=df, ratios=[0.5, 0.25, 0.25], group="g")
    assert labels(parts["train"]) == [0, 1, 4, 5, 6]
    assert labels(parts["valid"]) == [2, 7, 8]
    assert labels(parts["test"]) == [3, 9]
    assert parts["train"]["g"].tolist() == ["b", "b", "a", "a", "a"]
    assert list(parts["train"].columns) == ["value", "g"]


def test_sequential_windows_a_stream(root):
    stream = build("/source/kalfa/parquet_stream", path=str(root / "housing.parquet"), chunk=700)
    parts = build("/split/kalfa/sequential", df=stream, ratios=[0.7, 0.15, 0.15])
    assert all(isinstance(part, Stream) for part in parts.values())
    assert [(part.start, part.stop) for part in parts.values()] == [(0, 1400), (1400, 1700), (1700, 2000)]
    assert [part.rows for part in parts.values()] == [1400, 300, 300]
    assert [part.count() for part in parts.values()] == [1400, 300, 300]
    numpy.testing.assert_array_equal(pandas.concat(parts["valid"].chunks()).index.to_numpy(),
                                     numpy.arange(1400, 1700))
    with pytest.raises(ValueError, match=re.escape("a stream source has no group column; drop group for a "
                                                   "sequential split")):
        build("/split/kalfa/sequential", df=stream, ratios=[0.7, 0.15, 0.15], group="x0")


def test_sequential_cuts_a_dataset_source_by_position(root):
    samples = build("/source/kalfa/image_folder", path=str(root / "images"))
    parts = build("/split/kalfa/sequential", df=samples, ratios=[0.5, 0.25, 0.25])
    numpy.testing.assert_array_equal(parts["train"].positions, numpy.arange(0, 32))
    numpy.testing.assert_array_equal(parts["valid"].positions, numpy.arange(32, 48))
    numpy.testing.assert_array_equal(parts["test"].positions, numpy.arange(48, 64))
    numpy.testing.assert_array_equal(parts["test"].column("label"), numpy.ones(16, dtype="int64"))
    with pytest.raises(ValueError, match=re.escape("a Dataset source has no group column; drop group for a "
                                                   "sequential split")):
        build("/split/kalfa/sequential", df=samples, ratios=[0.5, 0.25, 0.25], group="label")


def test_kfold_holds_out_a_fold_of_the_seeded_permutation_and_carves_valid():
    df = table(10)
    parts = build("/split/kalfa/kfold", df=df, k=3, fold=1, val=0.25, seed=0)
    order = numpy.random.default_rng(0).permutation(10)
    assert order.tolist() == [4, 6, 2, 7, 3, 5, 9, 0, 8, 1]
    assert labels(parts["test"]) == [103, 105, 109]
    assert labels(parts["valid"]) == [104, 106]
    assert labels(parts["train"]) == [102, 107, 100, 108, 101]
    pandas.testing.assert_frame_equal(parts["test"], df.iloc[[3, 5, 9]])
    first = build("/split/kalfa/kfold", df=df, k=3, fold=0, seed=0)
    assert labels(first["test"]) == [104, 106, 102, 107]
    assert labels(first["valid"]) == []
    assert list(first["valid"].columns) == ["value", "group"]
    assert labels(first["train"]) == [103, 105, 109, 100, 108, 101]
    last = build("/split/kalfa/kfold", df=df, k=3, fold=2, seed=0)
    assert labels(last["test"]) == [100, 108, 101]
    assert labels(last["train"]) == [104, 106, 102, 107, 103, 105, 109]


def test_kfold_of_a_dataset_source_subsets_positions(root):
    samples = build("/source/kalfa/image_folder", path=str(root / "images"))
    parts = build("/split/kalfa/kfold", df=samples, k=4, fold=3, val=0.5, seed=2)
    order = numpy.random.default_rng(2).permutation(64)
    numpy.testing.assert_array_equal(parts["test"].positions, order[48:])
    numpy.testing.assert_array_equal(parts["valid"].positions, order[:24])
    numpy.testing.assert_array_equal(parts["train"].positions, order[24:48])
    assert all(isinstance(part, Samples) for part in parts.values())


def test_kfold_validates_k_fold_and_the_source(root):
    df = table(10)
    with pytest.raises(ValueError, match=re.escape("kfold needs an integer k >= 2, got 1")):
        build("/split/kalfa/kfold", df=df, k=1, fold=0)
    with pytest.raises(ValueError, match=re.escape("kfold needs an integer k >= 2, got 3.0")):
        build("/split/kalfa/kfold", df=df, k=3.0, fold=0)
    with pytest.raises(ValueError, match=re.escape("kfold needs an integer k >= 2, got True")):
        build("/split/kalfa/kfold", df=df, k=True, fold=0)
    with pytest.raises(ValueError, match=re.escape("fold must be an integer in [0, 2], got 3")):
        build("/split/kalfa/kfold", df=df, k=3, fold=3)
    with pytest.raises(ValueError, match=re.escape("fold must be an integer in [0, 2], got -1")):
        build("/split/kalfa/kfold", df=df, k=3, fold=-1)
    with pytest.raises(ValueError, match=re.escape("fold must be an integer in [0, 2], got False")):
        build("/split/kalfa/kfold", df=df, k=3, fold=False)
    stream = build("/source/kalfa/parquet_stream", path=str(root / "housing.parquet"))
    with pytest.raises(ValueError, match=re.escape("a stream source cannot be folded; the lazy set splits with "
                                                   "sequential or given")):
        build("/split/kalfa/kfold", df=stream, k=3, fold=0)


def test_given_reads_valid_and_test_like_the_source(root):
    df = table(6)
    parts = build("/split/kalfa/given", df=df, valid=str(root / "new.parquet"), test=str(root / "new.csv"))
    assert parts["train"] is df
    assert len(parts["valid"]) == 100
    assert list(parts["valid"].columns)[:3] == ["sample_id", "raw_0", "raw_1"]
    assert len(parts["test"]) == 50
    assert list(parts["test"].columns) == [f"x{position}" for position in range(8)] + ["price"]
    pandas.testing.assert_frame_equal(parts["test"], housing_frame(rows=50, seed=9), check_exact=False, rtol=1e-12)


def test_given_without_a_path_is_an_empty_set():
    df = table(6)
    parts = build("/split/kalfa/given", df=df)
    assert parts["train"] is df
    assert parts["valid"].empty and parts["test"].empty
    assert list(parts["valid"].columns) == ["value", "group"]
    assert str(parts["valid"]["value"].dtype) == "float64"


def test_given_reads_txt_as_csv_and_refuses_other_suffixes(root, tmp_path):
    df = table(6)
    housing_frame(rows=4).to_csv(tmp_path / "rows.txt", index=False)
    parts = build("/split/kalfa/given", df=df, test=str(tmp_path / "rows.txt"))
    assert len(parts["test"]) == 4
    with pytest.raises(ValueError, match=re.escape("given: cannot read 'rows.json'; a table set is a .parquet or "
                                                   ".csv file")):
        build("/split/kalfa/given", df=df, valid="rows.json")


def test_given_on_a_stream_streams_the_given_files_with_the_same_chunk(root):
    stream = build("/source/kalfa/parquet_stream", path=str(root / "housing.parquet"), chunk=300)
    parts = build("/split/kalfa/given", df=stream, valid=str(root / "new.parquet"))
    assert parts["train"] is stream
    assert isinstance(parts["valid"], Stream)
    assert isinstance(parts["valid"].reader, ParquetChunks)
    assert parts["valid"].reader.chunk == 300
    assert parts["valid"].rows == 100
    assert parts["valid"].count() == 100
    assert parts["test"].rows == 0
    assert parts["test"].reader is stream.reader
    assert parts["test"].count() == 0


def test_given_on_a_dataset_source_reads_the_paths_with_the_same_reader(root, tmp_path):
    samples = build("/source/kalfa/text_lines", path=str(root / "text.txt"))
    (tmp_path / "held.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    parts = build("/split/kalfa/given", df=samples, test=str(tmp_path / "held.txt"))
    assert parts["train"] is samples
    assert isinstance(parts["test"], Samples)
    assert isinstance(parts["test"].source, TextLines)
    assert len(parts["test"]) == 2
    assert parts["test"][1] == {"text": "beta"}
    assert len(parts["valid"]) == 0
    assert parts["valid"].source is samples.source


def test_prepared_split_takes_the_sets_a_prepared_frame_is_marked_with(tmp_path):
    housing = housing_frame(rows=8)
    folder = write_prepared(tmp_path / "prepared", {"train": housing.iloc[[5, 1, 6]], "valid": housing.iloc[[2]],
                                                    "test": housing.iloc[[0, 7]]})
    df = build("/source/kalfa/prepared", path=str(folder))
    parts = build("/split/kalfa/prepared", df=df, path=str(folder))
    assert list(parts) == SETS
    assert labels(parts["train"]) == [5, 1, 6]
    assert labels(parts["valid"]) == [2]
    assert labels(parts["test"]) == [0, 7]
    for part in parts.values():
        assert list(part.columns) == list(housing.columns)
    pandas.testing.assert_frame_equal(parts["train"].rename_axis(None), housing.iloc[[5, 1, 6]])


def test_prepared_split_takes_the_positions_of_a_dataset_source_per_set(root, tmp_path):
    samples = build("/source/kalfa/text_lines", path=str(root / "text.txt"))
    folder = write_prepared(tmp_path / "prepared", {"train": [7, 3, 9], "test": [1]}, layout="samples")
    parts = build("/split/kalfa/prepared", df=samples, path=str(folder))
    assert list(parts) == ["train", "test"]
    numpy.testing.assert_array_equal(parts["train"].positions, [7, 3, 9])
    numpy.testing.assert_array_equal(parts["test"].positions, [1])
    assert parts["train"][1] == samples[3]


def test_prepared_split_needs_a_manifest(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="is no prepared directory; kalfa prepare writes one"):
        build("/split/kalfa/prepared", df=table(4), path=str(tmp_path / "empty"))


def test_ratio_sizes_are_the_cuts_of_the_row_count():
    assert build("/lego/kalfa/ratio_sizes", rows=2000, ratios=[0.7, 0.15, 0.15]) == {"train": 1400, "valid": 300,
                                                                                    "test": 300}
    assert build("/lego/kalfa/ratio_sizes", rows=10, ratios=[0.5, 0.25, 0.25]) == {"train": 5, "valid": 2, "test": 3}
    assert build("/lego/kalfa/ratio_sizes", rows=3, ratios=[0.5, 0.5, 0.0], seed=1) == {"train": 2, "valid": 1,
                                                                                       "test": 0}
    assert build("/lego/kalfa/ratio_sizes", rows=0, ratios=[0.7, 0.15, 0.15]) == {"train": 0, "valid": 0, "test": 0}


def test_ratio_sizes_without_rows_tell_which_sets_exist():
    assert build("/lego/kalfa/ratio_sizes", rows=None, ratios=[0.7, 0.3, 0.0]) == {"train": None, "valid": None,
                                                                                  "test": 0}
    assert build("/lego/kalfa/ratio_sizes", rows=None, ratios=[1.0, 0.0, 0.0], group="g") == {"train": None,
                                                                                             "valid": 0, "test": 0}
    with pytest.raises(ValueError, match="ratios must have three entries"):
        build("/lego/kalfa/ratio_sizes", rows=None, ratios=[0.5, 0.5])


def test_kfold_sizes_follow_the_fold_bounds():
    assert build("/lego/kalfa/kfold_sizes", rows=10, k=3, fold=1, val=0.25) == {"train": 5, "valid": 2, "test": 3}
    assert build("/lego/kalfa/kfold_sizes", rows=10, k=3, fold=0) == {"train": 6, "valid": 0, "test": 4}
    assert build("/lego/kalfa/kfold_sizes", rows=64, k=4, fold=3, val=0.5, seed=2) == {"train": 24, "valid": 24,
                                                                                      "test": 16}
    assert build("/lego/kalfa/kfold_sizes", rows=None, k=3, fold=1) == {"train": None, "valid": 0, "test": None}
    assert build("/lego/kalfa/kfold_sizes", rows=None, k=3, fold=1, val=0.2) == {"train": None, "valid": None,
                                                                                "test": None}
    with pytest.raises(ValueError, match=re.escape("fold must be an integer in [0, 2], got 5")):
        build("/lego/kalfa/kfold_sizes", rows=None, k=3, fold=5)


def test_given_sizes_read_the_given_files_through_the_header(root):
    header = registry.resolve("/lego/kalfa/parquet_header")
    found = build("/lego/kalfa/given_sizes", rows=2000, valid=str(root / "new.parquet"), header=header)
    assert found == {"train": 2000, "valid": 100, "test": 0}
    assert build("/lego/kalfa/given_sizes", rows=2000, valid=str(root / "new.parquet")) == {"train": 2000,
                                                                                            "valid": None, "test": 0}
    assert build("/lego/kalfa/given_sizes", rows=None, test="missing.parquet", header=header) == {
        "train": None, "valid": 0, "test": None}
    csv_header = registry.resolve("/lego/kalfa/csv_header")
    assert build("/lego/kalfa/given_sizes", rows=7, valid=str(root / "new.csv"), test=str(root / "housing.csv"),
                 header=csv_header) == {"train": 7, "valid": 50, "test": 2000}


def test_prepared_sizes_are_the_sizes_of_the_manifest(tmp_path):
    housing = housing_frame(rows=8)
    folder = write_prepared(tmp_path / "prepared", {"train": housing.iloc[:5], "valid": housing.iloc[5:6],
                                                    "test": housing.iloc[6:]})
    assert build("/lego/kalfa/prepared_sizes", rows=None, path=str(folder)) == {"train": 5, "valid": 1, "test": 2}
    assert build("/lego/kalfa/prepared_sizes", rows=999, path=str(folder)) == {"train": 5, "valid": 1, "test": 2}
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="is no prepared directory; kalfa prepare writes one"):
        build("/lego/kalfa/prepared_sizes", rows=None, path=str(tmp_path / "empty"))
