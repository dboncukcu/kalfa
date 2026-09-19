import json
import re
from pathlib import Path

import numpy
import pandas
import pytest
from cirak.registry import registry
from PIL import Image

from data import housing_frame, write_images
from helpers import build
from kalfa.std import STD_URIS
from kalfa.std.common.samples import Samples
from kalfa.std.common.stream import CsvChunks, ParquetChunks, Stream
from kalfa.std.source.kalfa.samples import ImageFolder, TextLines


SOURCES = sorted(uri for uri in STD_URIS if uri.startswith("/source/"))
HEADERS = {"/source/kalfa/parquet": "/lego/kalfa/parquet_header",
           "/source/kalfa/csv": "/lego/kalfa/csv_header",
           "/source/kalfa/parquet_stream": "/lego/kalfa/parquet_header",
           "/source/kalfa/csv_stream": "/lego/kalfa/csv_header",
           "/source/kalfa/image_folder": "/lego/kalfa/image_folder_header",
           "/source/kalfa/text_lines": "/lego/kalfa/text_lines_header",
           "/source/kalfa/prepared": "/lego/kalfa/prepared_header"}
HOUSING_COLUMNS = [f"x{position}" for position in range(8)] + ["price"]


def write_prepared(folder, sets, layout="table", source=None, header=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    sizes = {}
    for name, part in sets.items():
        sizes[name] = len(part)
        if layout == "samples":
            (folder / f"{name}.json").write_text(json.dumps([int(position) for position in part]))
        else:
            part.rename_axis("row").reset_index().to_parquet(folder / f"{name}.parquet", index=False)
    manifest = {"kind": "data", "layout": layout, "sets": list(sets), "sizes": sizes, "header": header or {},
                "source": source or {}}
    (folder / "manifest.json").write_text(json.dumps(manifest))
    return folder


def test_source_catalog_is_the_seven_sources():
    assert SOURCES == sorted(HEADERS)


@pytest.mark.parametrize("uri", sorted(HEADERS))
def test_source_declares_its_header_lego(uri):
    assert registry.facts(uri).get("header") == HEADERS[uri]
    assert registry.facts(uri).returns == "df"


def test_source_aliases_are_the_short_names():
    aliases = registry.aliases()
    assert aliases["parquet"] == "/source/kalfa/parquet"
    assert aliases["csv"] == "/source/kalfa/csv"
    assert aliases["image_folder"] == "/source/kalfa/image_folder"
    assert aliases["text_lines"] == "/source/kalfa/text_lines"
    assert "parquet_stream" not in aliases
    assert "csv_stream" not in aliases
    assert "prepared" not in aliases


def test_source_facts_mark_streams_and_samples():
    assert registry.facts("/source/kalfa/parquet_stream").get("stream") is True
    assert registry.facts("/source/kalfa/csv_stream").get("stream") is True
    assert registry.facts("/source/kalfa/image_folder").get("samples") is True
    assert registry.facts("/source/kalfa/text_lines").get("samples") is True
    for uri in ("/source/kalfa/parquet", "/source/kalfa/csv", "/source/kalfa/prepared"):
        assert registry.facts(uri).get("stream") is None
        assert registry.facts(uri).get("samples") is None


def test_parquet_reads_the_whole_table(root):
    df = build("/source/kalfa/parquet", path=str(root / "housing.parquet"))
    assert isinstance(df, pandas.DataFrame)
    assert list(df.columns) == HOUSING_COLUMNS
    assert len(df) == 2000
    pandas.testing.assert_frame_equal(df, housing_frame())


def test_parquet_reads_the_listed_columns_only(root):
    df = build("/source/kalfa/parquet", path=str(root / "housing.parquet"), columns=["x1", "price"])
    assert list(df.columns) == ["x1", "price"]
    assert len(df) == 2000
    numpy.testing.assert_array_equal(df["price"].to_numpy(), housing_frame()["price"].to_numpy())


def test_csv_reads_the_whole_table(root):
    df = build("/source/kalfa/csv", path=str(root / "housing.csv"))
    assert list(df.columns) == HOUSING_COLUMNS
    assert len(df) == 2000
    pandas.testing.assert_frame_equal(df, housing_frame(), check_exact=False, rtol=1e-12)


def test_csv_reads_the_listed_columns_only(root):
    df = build("/source/kalfa/csv", path=str(root / "new.csv"), columns=["x0", "price"])
    assert list(df.columns) == ["x0", "price"]
    assert len(df) == 50


def test_parquet_stream_is_a_stream_over_chunks(root):
    stream = build("/source/kalfa/parquet_stream", path=str(root / "housing.parquet"), chunk=700)
    assert isinstance(stream, Stream)
    assert isinstance(stream.reader, ParquetChunks)
    assert stream.rows == 2000
    assert stream.columns == HOUSING_COLUMNS
    chunks = list(stream.chunks())
    assert [len(chunk) for chunk in chunks] == [700, 700, 600]
    assert [(int(chunk.index[0]), int(chunk.index[-1])) for chunk in chunks] == [(0, 699), (700, 1399), (1400, 1999)]
    assert stream.count() == 2000
    assert stream.head().equals(chunks[0])
    pandas.testing.assert_frame_equal(pandas.concat(chunks), housing_frame())


def test_parquet_stream_default_chunk_reads_the_file_at_once(root):
    stream = build("/source/kalfa/parquet_stream", path=str(root / "housing.parquet"))
    assert stream.reader.chunk == 65536
    assert [len(chunk) for chunk in stream.chunks()] == [2000]


def test_parquet_stream_reads_the_listed_columns_only(root):
    stream = build("/source/kalfa/parquet_stream", path=str(root / "housing.parquet"), chunk=1000,
                   columns=["x0", "price"])
    assert stream.columns == ["x0", "price"]
    assert list(stream.head().columns) == ["x0", "price"]
    assert stream.count() == 2000


def test_parquet_stream_window_keeps_the_positions_of_the_file(root):
    stream = build("/source/kalfa/parquet_stream", path=str(root / "housing.parquet"), chunk=700)
    window = stream.window(500, 1300)
    assert (window.start, window.stop, window.rows) == (500, 1300, 800)
    chunks = list(window.chunks())
    assert [len(chunk) for chunk in chunks] == [200, 600]
    numpy.testing.assert_array_equal(pandas.concat(chunks).index.to_numpy(), numpy.arange(500, 1300))
    assert window.count() == 800
    pandas.testing.assert_frame_equal(pandas.concat(chunks), housing_frame().iloc[500:1300])


def test_parquet_stream_window_of_a_window_stays_inside_the_outer_one(root):
    stream = build("/source/kalfa/parquet_stream", path=str(root / "housing.parquet"), chunk=700)
    inner = stream.window(500, 1300).window(100, 2000)
    assert (inner.start, inner.stop, inner.rows) == (600, 1300, 700)
    assert inner.count() == 700
    beyond = stream.window(1900, 2500)
    assert (beyond.start, beyond.stop, beyond.rows) == (1900, 2000, 100)
    assert beyond.count() == 100


def test_parquet_stream_query_filters_every_chunk(root):
    housing = housing_frame()
    stream = build("/source/kalfa/parquet_stream", path=str(root / "housing.parquet"), chunk=700)
    selected = stream.query("price > 200")
    assert selected.queries == ("price > 200",)
    assert selected.rows == 2000
    expected = housing.index[housing["price"] > 200].to_numpy()
    assert selected.count() == len(expected)
    numpy.testing.assert_array_equal(pandas.concat(selected.chunks()).index.to_numpy(), expected)
    twice = selected.query("x0 < 0").window(0, 1000)
    assert twice.queries == ("price > 200", "x0 < 0")
    kept = housing.iloc[:1000]
    assert twice.count() == int(((kept["price"] > 200) & (kept["x0"] < 0)).sum())


def test_parquet_stream_empty_window_yields_nothing(root):
    stream = build("/source/kalfa/parquet_stream", path=str(root / "housing.parquet"), chunk=700)
    empty = stream.window(300, 900).empty()
    assert (empty.start, empty.stop, empty.rows) == (300, 300, 0)
    assert list(empty.chunks()) == []
    assert empty.head() is None
    assert empty.count() == 0
    assert stream.query("price > 1e9").head() is None


def test_parquet_stream_missing_file_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match=re.escape("parquet file 'nope.parquet' does not exist")):
        build("/source/kalfa/parquet_stream", path="nope.parquet")


def test_csv_stream_is_a_stream_over_chunks(root):
    stream = build("/source/kalfa/csv_stream", path=str(root / "housing.csv"), chunk=700)
    assert isinstance(stream, Stream)
    assert isinstance(stream.reader, CsvChunks)
    assert stream.rows == 2000
    assert stream.columns == HOUSING_COLUMNS
    chunks = list(stream.chunks())
    assert [len(chunk) for chunk in chunks] == [700, 700, 600]
    assert [(int(chunk.index[0]), int(chunk.index[-1])) for chunk in chunks] == [(0, 699), (700, 1399), (1400, 1999)]
    pandas.testing.assert_frame_equal(pandas.concat(chunks), housing_frame(), check_exact=False, rtol=1e-12)


def test_csv_stream_window_query_and_columns(root):
    housing = housing_frame()
    stream = build("/source/kalfa/csv_stream", path=str(root / "housing.csv"), chunk=700, columns=["x0", "price"])
    assert stream.columns == ["x0", "price"]
    window = stream.window(500, 1300).query("price > 200")
    kept = housing.iloc[500:1300]
    expected = kept.index[kept["price"] > 200].to_numpy()
    assert window.count() == len(expected)
    found = pandas.concat(window.chunks())
    assert list(found.columns) == ["x0", "price"]
    numpy.testing.assert_array_equal(found.index.to_numpy(), expected)


def test_csv_stream_missing_file_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match=re.escape("csv file 'nope.csv' does not exist")):
        build("/source/kalfa/csv_stream", path="nope.csv")


def test_image_folder_is_samples_of_image_and_label(root):
    samples = build("/source/kalfa/image_folder", path=str(root / "images"))
    assert isinstance(samples, Samples)
    assert isinstance(samples.source, ImageFolder)
    assert len(samples) == 64
    assert samples.fields == ["image", "label"]
    assert samples.dtypes == {"image": "image", "label": "int64"}
    assert samples.source.classes == ["a", "b"]
    numpy.testing.assert_array_equal(samples.positions, numpy.arange(64))
    numpy.testing.assert_array_equal(samples.index, numpy.arange(64))
    numpy.testing.assert_array_equal(samples.column("label"), [0] * 40 + [1] * 24)
    assert [path.name for path, _ in samples.source.samples[:3]] == ["000.png", "001.png", "002.png"]
    assert samples.source.samples[40][0] == root / "images" / "b" / "000.png"


def test_image_folder_item_is_a_loaded_image_and_its_class_position(root):
    samples = build("/source/kalfa/image_folder", path=str(root / "images"))
    first = samples[0]
    assert set(first) == {"image", "label"}
    assert first["label"] == 0
    assert isinstance(first["image"], Image.Image)
    assert first["image"].mode == "L"
    assert first["image"].size == (8, 8)
    with Image.open(root / "images" / "a" / "000.png") as image:
        numpy.testing.assert_array_equal(numpy.asarray(first["image"]), numpy.asarray(image))
    assert samples[40]["label"] == 1
    assert samples[63]["label"] == 1


def test_image_folder_only_the_label_reads_as_a_column(root):
    samples = build("/source/kalfa/image_folder", path=str(root / "images"))
    with pytest.raises(ValueError, match=re.escape("field 'image' is not readable as a column; only label is")):
        samples.column("image")


def test_image_folder_subset_and_query_select_positions(root):
    samples = build("/source/kalfa/image_folder", path=str(root / "images"))
    ones = samples.query("label == 1")
    assert isinstance(ones, Samples)
    assert len(ones) == 24
    numpy.testing.assert_array_equal(ones.positions, numpy.arange(40, 64))
    numpy.testing.assert_array_equal(ones.column("label"), numpy.ones(24, dtype="int64"))
    assert ones[0]["label"] == 1
    assert ones[0]["image"].tobytes() == samples[40]["image"].tobytes()
    assert len(samples.query("label != 1")) == 40
    assert len(samples.query("label in [0]")) == 40
    assert len(samples.query("label not in [0, 1]")) == 0
    picked = ones.subset([2, 0])
    numpy.testing.assert_array_equal(picked.positions, [42, 40])
    assert len(picked) == 2


def test_image_folder_query_takes_field_equality_only(root):
    samples = build("/source/kalfa/image_folder", path=str(root / "images"))
    with pytest.raises(ValueError, match=re.escape("a Dataset source takes field equality filters only (field == "
                                                   "value, field != value, field in [a, b]), got 'label > 0'")):
        samples.query("label > 0")
    with pytest.raises(ValueError, match=re.escape("filter names field 'class'; the fields are ['image', 'label']")):
        samples.query("class == 1")


def test_image_folder_skips_files_that_are_not_images(tmp_path):
    write_images(tmp_path / "pics", {"cat": 3, "dog": 2}, size=6, channels=3, seed=4)
    (tmp_path / "pics" / "cat" / "notes.txt").write_text("not an image")
    (tmp_path / "pics" / "readme.md").write_text("not a class")
    samples = build("/source/kalfa/image_folder", path=str(tmp_path / "pics"))
    assert len(samples) == 5
    assert samples.source.classes == ["cat", "dog"]
    numpy.testing.assert_array_equal(samples.column("label"), [0, 0, 0, 1, 1])
    assert samples[0]["image"].mode == "RGB"
    assert samples[0]["image"].size == (6, 6)


def test_image_folder_missing_root_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match=re.escape("image folder 'missing' does not exist")):
        build("/source/kalfa/image_folder", path="missing")


def test_text_lines_is_samples_of_non_blank_lines(root, tmp_path):
    samples = build("/source/kalfa/text_lines", path=str(root / "text.txt"))
    assert isinstance(samples, Samples)
    assert isinstance(samples.source, TextLines)
    assert len(samples) == 200
    assert samples.fields == ["text"]
    assert samples.dtypes == {"text": "string"}
    lines = (root / "text.txt").read_text(encoding="utf-8").splitlines()
    assert samples[0] == {"text": lines[0]}
    assert samples[0]["text"].startswith("ROMEO: ")
    assert samples[1]["text"] == lines[1]
    numpy.testing.assert_array_equal(samples.column("text"), numpy.asarray(lines))
    (tmp_path / "gaps.txt").write_text("first\n\n  \nsecond\nthird\n\n", encoding="utf-8")
    gaps = build("/source/kalfa/text_lines", path=str(tmp_path / "gaps.txt"))
    assert len(gaps) == 3
    assert [gaps[position]["text"] for position in range(3)] == ["first", "second", "third"]


def test_text_lines_only_the_text_reads_as_a_column_and_queries_compare_it(root):
    samples = build("/source/kalfa/text_lines", path=str(root / "text.txt"))
    with pytest.raises(ValueError, match=re.escape("field 'line' is not readable as a column; only text is")):
        samples.column("line")
    lines = (root / "text.txt").read_text(encoding="utf-8").splitlines()
    chosen = samples.query(f"text == {lines[5]!r}")
    assert len(chosen) == lines.count(lines[5])
    assert chosen[0]["text"] == lines[5]


def test_text_lines_missing_file_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match=re.escape("text file 'missing.txt' does not exist")):
        build("/source/kalfa/text_lines", path="missing.txt")


def test_prepared_reads_the_table_sets_back_marked_by_set(tmp_path):
    housing = housing_frame(rows=10)
    parts = {"train": housing.iloc[[3, 0, 7]], "valid": housing.iloc[[5]], "test": housing.iloc[[1, 9]]}
    folder = write_prepared(tmp_path / "prepared", parts)
    df = build("/source/kalfa/prepared", path=str(folder))
    assert isinstance(df, pandas.DataFrame)
    assert list(df.columns) == HOUSING_COLUMNS + ["kalfa_set"]
    assert df.index.name == "row"
    assert df.index.tolist() == [3, 0, 7, 5, 1, 9]
    assert df["kalfa_set"].tolist() == ["train"] * 3 + ["valid"] + ["test"] * 2
    pandas.testing.assert_frame_equal(df[HOUSING_COLUMNS].rename_axis(None), housing.iloc[[3, 0, 7, 5, 1, 9]])


def test_prepared_reads_only_the_sets_the_manifest_lists(tmp_path):
    housing = housing_frame(rows=6)
    folder = write_prepared(tmp_path / "prepared", {"train": housing.iloc[:4], "test": housing.iloc[4:]})
    df = build("/source/kalfa/prepared", path=str(folder))
    assert df["kalfa_set"].tolist() == ["train"] * 4 + ["test"] * 2
    none = write_prepared(tmp_path / "none", {})
    assert build("/source/kalfa/prepared", path=str(none)).empty


def test_prepared_samples_layout_rebuilds_the_dataset_source(root, tmp_path):
    source = {"uri": "/source/kalfa/text_lines", "params": {"path": str(root / "text.txt")}}
    folder = write_prepared(tmp_path / "prepared", {"train": [0, 1, 2], "test": [3]}, layout="samples",
                            source=source)
    samples = build("/source/kalfa/prepared", path=str(folder))
    assert isinstance(samples, Samples)
    assert isinstance(samples.source, TextLines)
    assert len(samples) == 200
    numpy.testing.assert_array_equal(samples.positions, numpy.arange(200))


def test_prepared_needs_a_manifest_of_kind_data(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match=re.escape(f"{tmp_path / 'empty'} is no prepared directory; kalfa prepare "
                                                   f"writes one")):
        build("/source/kalfa/prepared", path=str(tmp_path / "empty"))
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "manifest.json").write_text(json.dumps({"kind": "run"}))
    with pytest.raises(ValueError, match="is no prepared directory"):
        build("/source/kalfa/prepared", path=str(tmp_path / "run"))


def test_parquet_header_reads_columns_arrow_types_and_rows(root):
    header = build("/lego/kalfa/parquet_header", path=str(root / "housing.parquet"))
    assert header == {"columns": HOUSING_COLUMNS, "dtypes": {name: "double" for name in HOUSING_COLUMNS},
                      "rows": 2000}
    reference = build("/lego/kalfa/parquet_header", path=str(root / "new.parquet"), chunk=16)
    assert reference["rows"] == 100
    assert len(reference["columns"]) == 20
    assert reference["dtypes"]["sample_id"] == "int64"
    assert reference["dtypes"]["raw_0"] == "double"
    assert reference["dtypes"]["region"] == "large_string"
    assert reference["dtypes"]["is_hot"] == "int64"


def test_parquet_header_lists_the_named_columns_only(root):
    header = build("/lego/kalfa/parquet_header", path=str(root / "housing.parquet"), columns=["price", "x2"])
    assert header == {"columns": ["price", "x2"], "dtypes": {"price": "double", "x2": "double"}, "rows": 2000}
    with pytest.raises(KeyError, match=re.escape(f"source columns ['nope'] are not in '{root / 'housing.parquet'}'; "
                                                 f"write names the file has")):
        build("/lego/kalfa/parquet_header", path=str(root / "housing.parquet"), columns=["price", "nope"])


def test_csv_header_reads_columns_first_rows_dtypes_and_line_count(root, tmp_path):
    header = build("/lego/kalfa/csv_header", path=str(root / "new.csv"))
    assert header == {"columns": HOUSING_COLUMNS, "dtypes": {name: "float64" for name in HOUSING_COLUMNS},
                      "rows": 50}
    (tmp_path / "mixed.csv").write_text("a,b,c\n1,1.5,x\n2,2.5,y\n")
    mixed = build("/lego/kalfa/csv_header", path=str(tmp_path / "mixed.csv"), chunk=8)
    assert mixed["columns"] == ["a", "b", "c"]
    assert mixed["rows"] == 2
    assert mixed["dtypes"]["a"] == "int64"
    assert mixed["dtypes"]["b"] == "float64"
    assert mixed["dtypes"]["c"] == str(pandas.Series(["x", "y"]).dtype)
    (tmp_path / "bare.csv").write_text("a,b\n")
    assert build("/lego/kalfa/csv_header", path=str(tmp_path / "bare.csv"))["rows"] == 0


def test_csv_header_lists_the_named_columns_only(root):
    header = build("/lego/kalfa/csv_header", path=str(root / "new.csv"), columns=["price"])
    assert header == {"columns": ["price"], "dtypes": {"price": "float64"}, "rows": 50}
    with pytest.raises(KeyError, match=re.escape("source columns ['zz'] are not in")):
        build("/lego/kalfa/csv_header", path=str(root / "new.csv"), columns=["zz"])


def test_image_folder_header_counts_images_and_classes(root):
    header = build("/lego/kalfa/image_folder_header", path=str(root / "images"))
    assert header == {"columns": ["image", "label"], "dtypes": {"image": "image", "label": "int64"}, "rows": 64,
                      "classes": ["a", "b"]}


def test_text_lines_header_counts_lines(root, tmp_path):
    assert build("/lego/kalfa/text_lines_header", path=str(root / "text.txt")) == {
        "columns": ["text"], "dtypes": {"text": "string"}, "rows": 200}
    (tmp_path / "gaps.txt").write_text("one\n\ntwo\n", encoding="utf-8")
    assert build("/lego/kalfa/text_lines_header", path=str(tmp_path / "gaps.txt"))["rows"] == 2


def test_prepared_header_is_the_header_of_the_manifest(tmp_path):
    header = {"columns": ["a", "b"], "dtypes": {"a": "double", "b": "int64"}, "rows": 7}
    folder = write_prepared(tmp_path / "prepared", {}, header=header)
    found = build("/lego/kalfa/prepared_header", path=str(folder))
    assert found == header
    assert found is not header
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match=re.escape(f"{tmp_path / 'empty'} is no prepared directory; kalfa prepare "
                                                   f"writes one")):
        build("/lego/kalfa/prepared_header", path=str(tmp_path / "empty"))
