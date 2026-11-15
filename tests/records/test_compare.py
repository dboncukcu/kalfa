import importlib
from pathlib import Path

from kalfa.records.compare import aligned_series, compare, config_diff, config_of, differing_params
from kalfa.records.reader import is_number, read_history
from records.fixtures import make_record


module = importlib.import_module("kalfa.records.compare")
SKIP = ("turn", "update", "seconds", "rules")


def files_under(root):
    return sorted(path.relative_to(root).as_posix() for path in Path(root).rglob("*") if path.is_file())


def test_differing_params_splits_shared_and_differing():
    summaries = [{"params": {"lr": 0.01, "width": 32, "seed": 1}},
                 {"params": {"lr": 0.001, "width": 32, "seed": 1}},
                 {"params": {"lr": 0.01, "width": 32}}]
    result = differing_params(summaries)
    assert result == {"shared": {"width": 32}, "differing": {"lr": [0.01, 0.001, 0.01], "seed": [1, 1, None]}}


def test_differing_params_edge_cases():
    assert differing_params([]) == {"shared": {}, "differing": {}}
    assert differing_params([{"params": {"lr": 0.01}}]) == {"shared": {"lr": 0.01}, "differing": {}}
    mixed = differing_params([{"params": {"lr": 0.01}}, None, {}])
    assert mixed == {"shared": {}, "differing": {"lr": [0.01, None, None]}}


def test_aligned_series_fills_missing_turns_with_none(tmp_path):
    short = make_record(tmp_path / "short", turns=3, seed=1)
    long = make_record(tmp_path / "long", turns=5, seed=2)
    result = aligned_series([str(short), str(long)], ["val/rmse"])
    assert list(result) == ["turn", str(short), str(long)]
    assert result["turn"] == [1, 2, 3, 4, 5]
    short_history = read_history(short)
    long_history = read_history(long)
    assert result[str(short)]["val/rmse"] == [line["val/rmse"] for line in short_history] + [None, None]
    assert result[str(long)]["val/rmse"] == [line["val/rmse"] for line in long_history]


def test_aligned_series_keys_default_to_first_record_and_path_keys_are_strings(tmp_path):
    first = make_record(tmp_path / "first", turns=2)
    second = make_record(tmp_path / "second", turns=4)
    result = aligned_series([first, second], None)
    assert set(result) == {"turn", str(first), str(second)}
    expected = [key for key, value in read_history(first)[0].items() if key not in SKIP and is_number(value)]
    assert list(result[str(first)]) == expected
    assert list(result[str(second)]) == expected
    assert all(len(values) == 4 for values in result[str(second)].values())


def test_aligned_series_tolerates_record_without_history(tmp_path):
    empty = make_record(tmp_path / "empty", turns=0, status="running", predictions=False)
    full = make_record(tmp_path / "full", turns=3)
    result = aligned_series([str(full), str(empty)], ["train/mse", "missing/key"])
    assert result["turn"] == [1, 2, 3]
    assert result[str(empty)] == {"train/mse": [None, None, None], "missing/key": [None, None, None]}
    assert result[str(full)]["missing/key"] == [None, None, None]
    assert None not in result[str(full)]["train/mse"]
    assert aligned_series([], ["train/mse"]) == {"turn": []}
    assert aligned_series([str(empty)], None) == {"turn": [], str(empty): {}}


def test_config_of_returns_resolved_text_or_none(tmp_path):
    record = make_record(tmp_path / "record", params={"lr": 0.01}, turns_planned=7)
    text = config_of(record)
    assert text == (record / "resolved.yaml").read_text(encoding="utf-8")
    assert "epochs: 7" in text
    assert "lr: 0.01" in text
    assert config_of(str(record)) == text
    assert config_of(tmp_path / "nowhere") is None
    bare = make_record(tmp_path / "bare", turns=0, status="running", predictions=False)
    (bare / "resolved.yaml").unlink()
    assert config_of(bare) is None
    assert config_diff(record, bare) is None
    (bare / "resolved.yaml").write_bytes(b"seed: 7\n\xff\n")
    assert config_of(bare).startswith("seed: 7\n")


def test_config_diff_text(tmp_path):
    a = make_record(tmp_path / "a", params={"lr": 0.001})
    b = make_record(tmp_path / "b", params={"lr": 0.01})
    same = make_record(tmp_path / "same", params={"lr": 0.001})
    assert config_diff(a, same) == ""
    text = config_diff(a, b)
    lines = text.splitlines()
    assert lines[0] == "--- a"
    assert lines[1] == "+++ b"
    assert "-  lr: 0.001" in lines
    assert "+  lr: 0.01" in lines
    assert lines[2].startswith("@@ ")
    assert " train:" in lines
    (b / "resolved.yaml").unlink()
    assert config_diff(a, b) is None
    assert config_diff(b, a) is None


def test_compare_three_records(tmp_path):
    a = make_record(tmp_path / "a", turns=3, params={"lr": 0.001, "width": 32}, seed=1)
    b = make_record(tmp_path / "b", turns=5, params={"lr": 0.01, "width": 32}, seed=2)
    c = make_record(tmp_path / "c", turns=2, params={"lr": 0.001, "width": 64}, seed=3)
    before = files_under(tmp_path)
    result = compare([str(a), str(b), str(c)], keys=["val/rmse"])
    assert files_under(tmp_path) == before
    assert set(result) == {"records", "params", "series", "diffs"}
    assert [record["name"] for record in result["records"]] == ["a", "b", "c"]
    assert result["params"] == {"shared": {}, "differing": {"lr": [0.001, 0.01, 0.001], "width": [32, 32, 64]}}
    assert result["series"]["turn"] == [1, 2, 3, 4, 5]
    assert list(result["series"]) == ["turn", str(a), str(b), str(c)]
    assert result["series"][str(c)]["val/rmse"][2:] == [None, None, None]
    assert list(result["diffs"]) == ["a vs b", "a vs c"]
    assert "+  lr: 0.01" in result["diffs"]["a vs b"]
    assert result["diffs"]["a vs c"] == ""


def test_compare_single_and_path_inputs(tmp_path):
    only = make_record(tmp_path / "only", turns=2)
    result = compare([only])
    assert result["diffs"] == {}
    assert result["params"] == {"shared": {"lr": 0.001}, "differing": {}}
    assert result["series"]["turn"] == [1, 2]
    assert str(only) in result["series"]
    assert result["records"][0]["name"] == "only"


def test_records_package_exports_compare_names():
    import kalfa.records as package

    for name in ("differing_params", "aligned_series", "config_of", "config_diff"):
        assert getattr(package, name) is getattr(module, name)
    assert package.compare is module
    assert callable(package.compare.compare)


def test_aligned_series_takes_a_single_key_as_a_string(tmp_path):
    directory = make_record(tmp_path / "one")
    found = aligned_series([directory], "val/rmse")
    assert list(found[str(directory)]) == ["val/rmse"]
    assert found["turn"] == [1, 2, 3]
