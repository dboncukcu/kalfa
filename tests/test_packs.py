"""The alias packs agree with the legos' alias facts."""

from ruamel.yaml import YAML
from cirak.registry import registry

import kalfa
from kalfa.std import STD_URIS


def test_tabular_pack_is_flat_and_matches_the_facts():
    table = YAML(typ="safe").load((kalfa.PACKS / "tabular.yaml").read_text())["alias"]
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


def test_the_random_split_alias_is_the_written_out_short_form(workdir):
    from helpers import minimal, write_config
    from kalfa.api import check
    from kalfa.config import parse_sets

    short = check([str(write_config(workdir / "short.yaml", minimal()))], parse_sets([]))
    written = minimal()
    written["data"]["split"] = {"uri": "random_split", "params": {"ratios": [0.7, 0.15, 0.15], "seed": 1}}
    spelled = check([str(write_config(workdir / "long.yaml", written))], parse_sets([]))
    assert short.problems == [] and spelled.problems == []
    assert short.document["flow"]["data"]["params"]["split"] == spelled.document["flow"]["data"]["params"]["split"]
    assert short.sizes == spelled.sizes == {"train": 1400, "valid": 300, "test": 300}
    empty_valid = minimal()
    empty_valid["data"]["split"] = {"uri": "random_split", "params": {"ratios": [0.8, 0.0, 0.2], "seed": 1}}
    empty_valid["training"]["checkpoint"] = {"uri": "best", "params": {"monitor": "val/rmse"}}
    problems = check([str(write_config(workdir / "novalid.yaml", empty_valid))], parse_sets([])).problems
    assert "set_missing" in [problem.kind for problem in problems]
