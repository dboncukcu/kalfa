"""The alias packs agree with the legos' alias facts; the packs layer over base, lazy_tabular over tabular."""

from cirak.registry import registry
from ruamel.yaml import YAML

import kalfa
from kalfa.config import pack_tables
from kalfa.std import STD_URIS


def test_tabular_pack_is_flat_and_matches_the_facts():
    table = pack_tables()["/alias/kalfa/tabular"]
    assert "/alias/kalfa/tabular" in registry.fragments()
    for name, uri in table.items():
        entry = registry.lookup(uri)
        assert entry is not None, uri
        assert name in entry.facts.alias, (name, uri)
        assert uri in STD_URIS
    assert len(set(table.values())) <= len(table)
    for name in ("parquet", "table", "linear", "mse", "rmse", "adam", "supervised", "alternating", "plateau",
                 "after_epoch", "best", "last", "loss_curve", "standard_scaler", "auroc", "average_precision",
                 "l1_distance", "class_histogram", "architecture", "random_split"):
        assert name in table
    assert table["random_split"] == "/split/kalfa/random" and table["random"] == "/strategy/kalfa/random"


def test_the_packs_layer_over_the_base_pack():
    tables = pack_tables()
    base = tables["/alias/kalfa/base"]
    below = {"tabular": "/alias/kalfa/base", "vision": "/alias/kalfa/base", "text": "/alias/kalfa/base",
             "lazy_tabular": "/alias/kalfa/tabular"}
    packs = {name: tables[f"/alias/kalfa/{name}"] for name in below}
    for name, table in packs.items():
        own = YAML(typ="safe").load((kalfa.PACKS / f"{name}.yaml").read_text())
        assert own["include"] == [below[name]]
        assert not (set(own["alias"]) - set(table))
        for alias, uri in base.items():
            if alias not in own["alias"]:
                assert table[alias] == uri, (name, alias)
    shared = {alias for alias, uri in base.items()
              if all(table.get(alias) == uri for table in packs.values())}
    assert shared == set(base)
    for name in ("cast", "table", "linear", "mse", "rmse", "adam", "supervised", "best", "loss_curve"):
        assert name in base
    assert "image_folder" not in base and "image_folder" in packs["vision"]
    assert "parquet" not in base and "random_split" not in base


def test_lazy_tabular_is_the_tabular_pack_with_the_sources_streamed():
    packs = {name: pack_tables()[f"/alias/kalfa/{name}"] for name in ("tabular", "lazy_tabular")}
    assert packs["tabular"]["parquet"] == "/source/kalfa/parquet" and packs["tabular"]["csv"] == "/source/kalfa/csv"
    assert packs["lazy_tabular"]["parquet"] == "/source/kalfa/parquet_stream"
    assert packs["lazy_tabular"]["csv"] == "/source/kalfa/csv_stream"
    streamed = {"parquet", "csv"}
    assert {name for name, uri in packs["lazy_tabular"].items() if packs["tabular"].get(name) != uri} == streamed
    assert set(packs["tabular"]) == set(packs["lazy_tabular"])


def test_the_random_split_alias_is_the_written_out_short_form(workdir):
    from helpers import minimal, write_config
    from kalfa.api import check
    from kalfa.config import parse_sets

    short = check([str(write_config(workdir / "short.yaml", minimal()))], parse_sets([]))
    written = minimal()
    written["data"]["split"] = {"uri": "random_split", "params": {"ratios": [0.7, 0.15, 0.15], "seed": 7}}
    spelled = check([str(write_config(workdir / "long.yaml", written))], parse_sets([]))
    assert short.problems == [] and spelled.problems == []
    assert short.document["flow"]["data"]["params"]["split"] == spelled.document["flow"]["data"]["params"]["split"]
    assert short.sizes == spelled.sizes == {"train": 1400, "valid": 300, "test": 300}
    empty_valid = minimal()
    empty_valid["data"]["split"] = {"uri": "random_split", "params": {"ratios": [0.8, 0.0, 0.2], "seed": 1}}
    empty_valid["training"]["checkpoint"] = {"uri": "best", "params": {"monitor": "val/rmse"}}
    problems = check([str(write_config(workdir / "novalid.yaml", empty_valid))], parse_sets([])).problems
    assert "set_missing" in [problem.kind for problem in problems]
