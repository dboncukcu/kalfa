import pytest
from cirak.registry import registry

from helpers import ROOT
from kalfa import docs, lego
from kalfa.config import pack_tables
from kalfa.kinds import kalfa_kind, kinds
from kalfa.std import STD_URIS


def thing(predictions, history, models, record, name=None):
    return None


def test_shipped_docs_equal_the_rendered_reference():
    assert (ROOT / "DOCS.md").read_text() == docs.render()


def test_render_lists_every_std_uri_once_by_kind():
    text = docs.render()
    assert text.startswith(docs.HEADER)
    assert "## Catalog\n" in text
    assert "## Skeleton steps\n" in text
    assert "## Plugin legos" not in text
    skeleton = docs.template_uris()
    catalog = text[text.index("## Catalog"):text.index("## Skeleton steps")]
    steps = text[text.index("## Skeleton steps"):text.index("## Alias packs")]
    for uri in STD_URIS:
        assert (catalog + steps).count(f"| `{uri}` |") == 1, uri
    for uri in STD_URIS:
        assert (f"| `{uri}` |" in steps) == (uri in skeleton), uri
        assert (f"| `{uri}` |" in catalog) == (uri not in skeleton), uri
    headings = [line for line in text.splitlines() if line.startswith("### ") and not line.startswith("### /")]
    assert headings == ["### Kinds", *[f"### {kind}" for kind in kinds()
                                      if any(kalfa_kind(uri) == kind for uri in STD_URIS - skeleton)]]


def test_render_counts_the_kinds_table():
    text = docs.render()
    criteria = sorted(uri for uri in STD_URIS if kalfa_kind(uri) == "criterion")
    assert criteria == ["/criterion/kalfa/bce_logits", "/criterion/kalfa/cross_entropy", "/criterion/kalfa/huber",
                        "/criterion/kalfa/log_cosh", "/criterion/kalfa/mae", "/criterion/kalfa/mse",
                        "/criterion/kalfa/weighted_mse"]
    assert "| `criterion` | losses, metrics | 7 |" in text
    assert "| `objective` | losses | 8 |" in text
    assert "| `rule` | the contract | 0 |" not in text
    assert "| Kind | Where it is written | Count |" in text


def test_render_lists_the_packs_with_their_kinds():
    text = docs.render()
    packs = text[text.index("## Alias packs"):]
    tables = pack_tables()
    assert [line for line in packs.splitlines() if line.startswith("### ")] == [f"### {uri}" for uri in tables]
    assert "| `mse` | `/criterion/kalfa/mse` | criterion |" in packs
    assert "| `parquet` | `/source/kalfa/parquet_stream` | source |" in packs
    assert "| `after_epoch` | `/trigger/kalfa/after_turn` | trigger |" in packs
    for uri, table in tables.items():
        section = packs[packs.index(f"### {uri}"):]
        for name, target in table.items():
            assert f"| `{name}` | `{target}` | {kalfa_kind(target)} |" in section, (uri, name)


def test_table_writes_one_markdown_row_per_uri():
    lines = []
    docs.table(lines, ["/criterion/kalfa/mse", "/checkpoint/kalfa/best"])
    assert lines == [
        "| URI | Alias | Signature | Facts | Description |",
        "|---|---|---|---|---|",
        "| `/criterion/kalfa/mse` | `mse` | `(predictions, targets)` | partial: True | Mean squared error |",
        "| `/checkpoint/kalfa/best` | `best` | `(monitor, mode='min', last=True)` | writes: best, last | Write best.pt "
        "when the monitored value improves and last.pt every turn; last: false writes best.pt alone, for a run that "
        "is never resumed |",
        "",
    ]


def test_signature_text_reads_the_callable():
    assert docs.signature_text(registry.resolve("/checkpoint/kalfa/best")) == "(monitor, mode='min', last=True)"
    assert docs.signature_text(registry.resolve("/criterion/kalfa/huber")) == "(predictions, targets, delta=1.0)"
    assert docs.signature_text(5) == ""


def test_facts_text_lists_the_declared_facts_without_kind_and_alias():
    assert docs.facts_text(registry.facts("/checkpoint/kalfa/best")) == "writes: best, last"
    assert docs.facts_text(registry.facts("/builder/kalfa/module")) == ("bus: prep=prep, train_loader=train_loader; "
                                                                        "roles: weights, bias, scale")
    assert docs.facts_text(registry.facts("/turn/kalfa/alternating")) == (
        "returns: models, optimizers, emas, counters, stream, metrics; "
        "bus: device=device, prep=prep, record=record, monitor=monitor; "
        "mutates: models, optimizers, emas, counters; extras: amp, grad_clip, accumulate")
    assert docs.facts_text(registry.facts("/criterion/kalfa/mse")) == "partial: True"
    assert docs.facts_text(registry.facts("/device/kalfa/cpu")) == ""


def test_cell_escapes_pipes_and_newlines():
    assert docs.cell("a|b\nc") == "a\\|b c"


def test_template_uris_are_the_skeleton_steps_the_contract_names():
    assert docs.template_uris() == {
        "/builder/kalfa/module", "/lego/kalfa/apply", "/lego/kalfa/apply_frames", "/lego/kalfa/architecture_note",
        "/lego/kalfa/calibrate", "/lego/kalfa/checkpoint", "/lego/kalfa/clone", "/lego/kalfa/const",
        "/lego/kalfa/csv_header", "/lego/kalfa/data_report", "/lego/kalfa/evaluate", "/lego/kalfa/figures",
        "/lego/kalfa/fit", "/lego/kalfa/fit_frames", "/lego/kalfa/generate", "/lego/kalfa/given_sizes",
        "/lego/kalfa/history", "/lego/kalfa/identity", "/lego/kalfa/image_folder_header", "/lego/kalfa/init_state",
        "/lego/kalfa/kfold_sizes", "/lego/kalfa/merge", "/lego/kalfa/pack", "/lego/kalfa/parquet_header",
        "/lego/kalfa/predict", "/lego/kalfa/prepared_header", "/lego/kalfa/prepared_sizes", "/lego/kalfa/ratio_sizes",
        "/lego/kalfa/read_frames", "/lego/kalfa/read_prep", "/lego/kalfa/run_all", "/lego/kalfa/save_final",
        "/lego/kalfa/select", "/lego/kalfa/text_lines_header", "/lego/kalfa/transform_set", "/loader/kalfa/torch",
        "/rule/kalfa/effects", "/rule/kalfa/open", "/rule/kalfa/rule", "/rule/kalfa/stop"}


@pytest.mark.xfail(strict=True, reason="bug: plugin_uris lists cirak's own /builder/cirak/compose as a plugin lego")
def test_plugin_uris_is_empty_without_plugins():
    assert docs.plugin_uris() == []


def test_plugin_uris_lists_a_registered_plugin_lego():
    lego("/plot/docs_plug/thing", thing, partial=True, requires="zzz")
    assert "/plot/docs_plug/thing" in docs.plugin_uris()
    assert not any(uri in STD_URIS for uri in docs.plugin_uris())


def test_render_with_plugins_adds_the_plugin_section():
    lego("/plot/docs_plug/thing", thing, partial=True, requires="zzz")
    text = docs.render(plugins=["/plot/docs_plug/thing"])
    section = text[text.index("## Plugin legos"):text.index("## Skeleton steps")]
    assert section == "\n".join([
        "## Plugin legos",
        "",
        docs.PLUGIN_NOTE,
        "### plot",
        "",
        "| URI | Alias | Signature | Facts | Description |",
        "|---|---|---|---|---|",
        "| `/plot/docs_plug/thing` |  | `(predictions, history, models, record, name=None)` | partial: True; "
        "requires: zzz | thing(predictions, history, models, record, name=None) |",
        "",
        "",
    ])
    empty = docs.render(plugins=[])
    assert "## Plugin legos\n\n" + docs.PLUGIN_NOTE + "\nNothing registered outside kalfa's std set.\n" in empty
    assert docs.render() == docs.render(uris=sorted(STD_URIS))
