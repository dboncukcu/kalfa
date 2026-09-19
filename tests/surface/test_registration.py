import hashlib
import subprocess
import sys

import pytest
from cirak.registry import registry

from kalfa import CONTRACT, PACKS, lego
from kalfa.config import pack_tables
from kalfa.contract import Contract
from kalfa.errors import KalfaError
from kalfa.kinds import cirak_kind, kalfa_kind, kind_of, kinds, names_of
from kalfa.registration import pack
from kalfa.std import STD_URIS


KINDS = ["source", "transform", "split", "frame", "pre", "feed", "loader", "layer", "init", "criterion", "objective",
         "metric", "adapter", "optimizer", "schedule", "turn", "trigger", "checkpoint", "rule", "generate", "plot",
         "strategy", "device", "rng", "export", "calibrate", "lego", "builder", "data"]


def zero(predictions, targets):
    return 0.0


def always(metrics, turn_index, state):
    return True


def test_lego_takes_the_kind_from_the_first_uri_segment():
    assert lego("/criterion/test/zero", zero, description="always zero", partial=True, alias="zero_test") is zero
    entry = registry.lookup("/criterion/test/zero")
    assert entry.target is zero
    assert entry.description == "always zero"
    assert entry.facts.kind == "criterion"
    assert entry.facts.partial is True
    assert entry.facts.alias == ("zero_test",)
    assert registry.aliases()["zero_test"] == "/criterion/test/zero"


def test_lego_works_as_a_decorator_and_stores_declared_facts():
    @lego("/objective/test/gp", needs_grad=True, partial=True, refs={"critic": "model"})
    def gp(models, batch, critic):
        return 0.0

    facts = registry.facts("/objective/test/gp")
    assert facts.kind == "objective"
    assert facts.get("needs_grad") is True
    assert facts.refs == {"critic": "model"}
    assert facts.declared() == {"kind": "objective", "partial": True, "refs": {"critic": "model"},
                                "needs_grad": True}
    assert registry.resolve("/objective/test/gp") is gp


def test_lego_refuses_an_unknown_kind():
    with pytest.raises(ValueError) as caught:
        lego("/nope/test/x", zero)
    assert str(caught.value) == f"/nope/test/x: 'nope' is not a kalfa kind; the kinds are {KINDS}"
    assert registry.lookup("/nope/test/x") is None


def test_lego_refuses_a_kind_argument():
    with pytest.raises(ValueError) as caught:
        lego("/criterion/test/x", zero, kind="metric")
    assert str(caught.value) == (
        "/criterion/test/x: kalfa.lego takes the kind from the first segment of the URI; drop kind=")
    assert registry.lookup("/criterion/test/x") is None


def test_trigger_legos_are_cirak_predicates():
    lego("/trigger/test/always", always, partial=True)
    assert registry.facts("/trigger/test/always").kind == "predicate"
    assert kind_of("/trigger/test/always") == "trigger"
    assert cirak_kind("trigger") == "predicate"
    assert cirak_kind("criterion") == "criterion"


def test_pack_declares_legos_under_one_prefix_with_lazy_targets():
    declare = pack("acme.legos.criterion.acme")
    assert declare("/criterion/acme/zero", "mod:zero", partial=True) == "acme.legos.criterion.acme.mod:zero"
    entry = registry.lookup("/criterion/acme/zero")
    assert entry.target == "acme.legos.criterion.acme.mod:zero"
    assert entry.facts.kind == "criterion"
    assert entry.facts.partial is True
    with pytest.raises(ValueError) as caught:
        declare("/criterion/acme/a/b", "mod:zero")
    assert str(caught.value) == (
        "/criterion/acme/a/b: acme.legos.criterion.acme declares /criterion/acme/<name>, nothing else")
    with pytest.raises(ValueError) as caught:
        declare("/criterion/acme/c", "mod.sub:zero")
    assert str(caught.value) == ("/criterion/acme/c: the target is 'module:name' inside acme.legos.criterion.acme, "
                                 "got 'mod.sub:zero'")


def test_kinds_are_the_declared_ones_plus_builder_and_data():
    assert kinds() == KINDS
    assert "predicate" not in kinds()
    assert kind_of("/lego/kalfa/const") == "lego"
    assert kind_of("/data/kalfa/feature_width") == "data"
    assert kind_of("/builder/kalfa/module") == "builder"
    assert kalfa_kind("/criterion/kalfa/mse") == "criterion"
    assert kalfa_kind("nope") is None
    assert kalfa_kind("/nope/a/b") is None
    assert kalfa_kind(None) is None


@pytest.mark.parametrize("uri", ["nope", "/x", 5, None])
def test_kind_of_refuses_what_is_no_lego_uri(uri):
    with pytest.raises(ValueError) as caught:
        kind_of(uri)
    assert str(caught.value) == f"{uri!r} is not a lego URI (/<kind>/<pack>/<name>)"


def test_names_of_wraps_a_string_and_keeps_a_list():
    assert names_of(None) == ()
    assert names_of("a") == ("a",)
    assert names_of(["a", "b"]) == ("a", "b")


def test_every_std_uri_resolves_and_keeps_its_kind():
    assert len(STD_URIS) > 300
    for uri in sorted(STD_URIS):
        entry = registry.lookup(uri)
        assert entry is not None and not entry.fragment, uri
        assert entry.facts.kind == cirak_kind(kind_of(uri)), uri
        assert callable(registry.resolve(uri)), uri


def test_std_uris_cover_every_kind_but_predicate_and_the_registry_has_nothing_else_but_cirak():
    assert {kind_of(uri) for uri in STD_URIS} == set(KINDS)
    extra = [uri for uri in registry.uris() if uri not in STD_URIS and not registry.lookup(uri).fragment]
    assert extra == ["/builder/cirak/compose"]


def test_packs_are_registered_fragments_from_the_package_directory():
    fragments = registry.fragments()
    assert sorted(uri for uri in fragments if uri.startswith("/alias/")) == [
        "/alias/kalfa/base", "/alias/kalfa/lazy_tabular", "/alias/kalfa/tabular", "/alias/kalfa/text",
        "/alias/kalfa/vision"]
    assert fragments["/alias/kalfa/base"] == str(PACKS / "base.yaml")
    assert registry.lookup("/alias/kalfa/text").description == "alias pack text: short names for the std legos"
    assert CONTRACT == Contract.default_path()


def test_base_pack_is_a_subset_of_the_tabular_vision_and_text_packs():
    tables = pack_tables()
    base = set(tables["/alias/kalfa/base"].items())
    for name in ("tabular", "vision", "text"):
        assert base < set(tables[f"/alias/kalfa/{name}"].items()), name


def test_lazy_tabular_rebinds_only_parquet_and_csv_to_the_stream_sources():
    tables = pack_tables()
    tabular = tables["/alias/kalfa/tabular"]
    lazy = tables["/alias/kalfa/lazy_tabular"]
    assert set(lazy) == set(tabular)
    assert {name: value for name, value in lazy.items() if tabular[name] != value} == {
        "parquet": "/source/kalfa/parquet_stream", "csv": "/source/kalfa/csv_stream"}
    assert tabular["parquet"] == "/source/kalfa/parquet"
    assert tabular["csv"] == "/source/kalfa/csv"


def test_every_alias_maps_to_a_std_uri_that_declares_it():
    repointed = {("/alias/kalfa/lazy_tabular", "parquet"): "/source/kalfa/parquet_stream",
                 ("/alias/kalfa/lazy_tabular", "csv"): "/source/kalfa/csv_stream"}
    for uri, table in pack_tables().items():
        for name, target in table.items():
            assert target in STD_URIS, (uri, name)
            if repointed.get((uri, name)) == target:
                assert registry.facts(target).alias == (), (uri, name, target)
                continue
            assert name in registry.facts(target).alias, (uri, name, target)
            assert kind_of(target) == target.split("/")[1]
    assert set(registry.aliases()) == {name for table in pack_tables().values() for name in table}


@pytest.mark.subprocess
def test_importing_kalfa_loads_no_torch_numpy_or_pandas():
    code = "import sys, kalfa; print(sorted(name for name in ('torch', 'numpy', 'pandas') if name in sys.modules))"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]"


def test_contract_loads_the_shipped_document():
    contract = Contract.load()
    assert contract.path == CONTRACT
    assert sorted(contract.document) == ["blocks", "wiring"]
    assert list(contract.blocks) == ["data", "models", "optimizers", "training", "after"]
    assert contract.sets == ["train", "valid", "test"]
    assert contract.history_prefix == {"train": "train", "valid": "val", "test": "test"}
    assert contract.prefixes(["valid", "calib"]) == {"valid": "val", "calib": "calib"}
    assert contract.run_inputs == ["device", "record", "monitor"]
    assert contract.roles() == ["weights", "bias", "scale"]
    assert contract.default_of("training", "sets") == ["valid", "test"]
    assert contract.default_of("data", "mask") is None
    assert contract.default_of("nope", "x") is None
    assert contract.template() == {"blocks": contract.blocks}
    assert contract.wiring["builder"] == "/builder/kalfa/module"
    assert contract.wiring["default_split"] == "/split/kalfa/random"
    assert contract.wiring["filter"] == "/transform/kalfa/filter"
    assert contract.wiring["loader"] == "/loader/kalfa/torch"
    assert contract.wiring["figures"] == "/lego/kalfa/figures"
    assert contract.plot_bus == {
        "prep": "prep", "train_loader": "train_loader", "valid_loader": "valid_loader", "test_loader": "test_loader",
        "composites": "composites", "device": "device", "counters": "counters_final",
        "optimizers": "optimizers_final", "emas": "emas_final", "rules": "rules_final", "losses": "losses",
        "losses_keys": "losses_keys", "data_report": "data_report", "train_df": "train_df",
        "train_frame": "train_frame"}


def test_contract_digest_is_the_sha256_of_its_text():
    contract = Contract.load()
    assert contract.text() == CONTRACT.read_text()
    assert contract.digest() == hashlib.sha256(CONTRACT.read_bytes()).hexdigest()
    assert len(contract.digest()) == 64


def test_contract_write_makes_an_identical_copy(tmp_path):
    contract = Contract.load()
    contract.write(tmp_path / "copy" / "contract.yaml")
    copy = Contract.load(tmp_path / "copy" / "contract.yaml")
    assert copy.path == tmp_path / "copy" / "contract.yaml"
    assert copy.text() == contract.text()
    assert copy.digest() == contract.digest()
    assert copy.document == contract.document


def test_contract_load_refuses_a_missing_or_malformed_file(tmp_path):
    with pytest.raises(KalfaError) as caught:
        Contract.load(tmp_path / "nope.yaml")
    assert str(caught.value) == (
        f"contract {tmp_path / 'nope.yaml'} does not exist; kalfa contract --write writes the default one")
    (tmp_path / "bare.yaml").write_text("wiring: {}\n")
    with pytest.raises(KalfaError) as caught:
        Contract.load(tmp_path / "bare.yaml")
    assert str(caught.value) == (
        f"contract {tmp_path / 'bare.yaml'} needs a blocks mapping; start from kalfa contract --write")
    (tmp_path / "empty.yaml").write_text("")
    with pytest.raises(KalfaError) as caught:
        Contract.load(tmp_path / "empty.yaml")
    assert str(caught.value) == (
        f"contract {tmp_path / 'empty.yaml'} needs a wiring mapping; start from kalfa contract --write")
