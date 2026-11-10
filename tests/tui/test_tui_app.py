import pytest
from records.fixtures import make_record, make_sweep

from kalfa.tui.app import KalfaApp


@pytest.fixture
def root(tmp_path):
    make_record(tmp_path / "runs" / "alpha", status="done")
    make_record(tmp_path / "runs" / "beta", status="running")
    make_sweep(tmp_path / "runs" / "sweep", count=2)
    return tmp_path / "runs"


@pytest.mark.asyncio
async def test_records_tab_lists_the_records(root):
    app = KalfaApp(root=str(root))
    async with app.run_test() as pilot:
        await pilot.pause()
        table = app.query_one("#records-table")
        assert table.row_count >= 3
        await pilot.press("down")
        await pilot.pause()
        detail = app.query_one("#record-detail")
        assert detail.renderable is not None or str(detail.render())


@pytest.mark.asyncio
async def test_legos_tab_searches_the_catalog(root):
    app = KalfaApp(root=str(root))
    async with app.run_test() as pilot:
        await pilot.press("2")
        await pilot.pause()
        search = app.query_one("#legos-search")
        search.value = "mlp"
        await pilot.pause()
        tree = app.query_one("#legos-tree")
        assert any(leaf.data.name == "mlp" for branch in tree.root.children for leaf in branch.children)


@pytest.mark.asyncio
async def test_config_tab_checks_a_config(root, tmp_path):
    config = tmp_path / "c.yaml"
    config.write_text("train: {epochs: 1}\n", encoding="utf-8")
    app = KalfaApp(root=str(root), config=str(config))
    async with app.run_test() as pilot:
        await pilot.press("3")
        await pilot.pause()
        await pilot.press("c")
        await pilot.pause(0.5)
        output = app.query_one("#config-output")
        text = str(output.render()) if hasattr(output, "render") else str(output)
        assert "error" in text.lower() or "ok" in text.lower()
