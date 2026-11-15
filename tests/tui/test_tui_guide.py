from pathlib import Path

import pytest

from kalfa.tui.app import KalfaApp


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "01_mlp_regression" / "config.yaml"
DDPM = ROOT / "examples" / "08_ddpm" / "config.yaml"


def names_of(app):
    listing = app.query_one("#guide-sections")
    return [listing.get_option_at_index(index).id for index in range(listing.option_count)]


def highlighted(app):
    listing = app.query_one("#guide-sections")
    return listing.get_option_at_index(listing.highlighted).id


async def open_config(app, pilot, path):
    await pilot.press("3")
    await pilot.pause()
    app.query_one("#config-path").value = str(path)
    await pilot.press("4")
    await pilot.pause(0.2)


@pytest.mark.asyncio
async def test_choosing_a_section_scrolls_the_guide_to_it(tmp_path):
    app = KalfaApp(root=str(tmp_path), config=str(EXAMPLE))
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.press("4")
        await pilot.pause()
        view = app.query_one("#guide-view")
        scroll = app.query_one("#guide-scroll")
        assert scroll.scroll_y == 0
        for _ in range(names_of(app).index("optimizers")):
            await pilot.press("down")
        await pilot.press("enter")
        await pilot.pause(0.2)
        widget = view.sections["optimizers"]
        assert scroll.scroll_y > 0
        assert scroll.scroll_y == min(widget.virtual_region.y, scroll.max_scroll_y)


@pytest.mark.asyncio
async def test_another_config_opens_at_its_head_and_the_same_one_keeps_its_place(tmp_path):
    app = KalfaApp(root=str(tmp_path), config=str(EXAMPLE))
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.press("4")
        await pilot.pause()
        scroll = app.query_one("#guide-scroll")
        scroll.scroll_to(y=60, animate=False, immediate=True)
        await pilot.press("1", "4")
        await pilot.pause(0.2)
        assert scroll.scroll_y == 60
        await open_config(app, pilot, DDPM)
        assert scroll.scroll_y == 0 and highlighted(app) == "params"


@pytest.mark.asyncio
async def test_another_config_opens_at_the_chosen_section_or_at_its_head(tmp_path):
    app = KalfaApp(root=str(tmp_path), config=str(EXAMPLE))
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.press("4")
        await pilot.pause()
        for _ in range(names_of(app).index("optimizers")):
            await pilot.press("down")
        await pilot.pause(0.2)
        await open_config(app, pilot, DDPM)
        view, scroll = app.query_one("#guide-view"), app.query_one("#guide-scroll")
        assert highlighted(app) == "optimizers"
        assert scroll.scroll_y == min(view.sections["optimizers"].virtual_region.y, scroll.max_scroll_y) > 0
        await pilot.press("down", "down")
        await pilot.pause(0.2)
        assert view.picked == "generate" and scroll.scroll_y > 0
        await open_config(app, pilot, EXAMPLE)
        assert scroll.scroll_y == 0 and highlighted(app) == "params"


@pytest.mark.asyncio
async def test_r_reads_the_config_tab_again(tmp_path):
    app = KalfaApp(root=str(tmp_path), config=str(EXAMPLE))
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.press("4")
        await pilot.pause()
        assert "generate" not in names_of(app)
        app.query_one("#config-path").value = str(DDPM)
        await pilot.press("r")
        await pilot.pause()
        assert names_of(app) == ["params", "run", "data", "models", "losses", "optimizers", "train", "generate",
                                 "plots"]
        assert app.focused is app.query_one("#guide-sections")


@pytest.mark.asyncio
async def test_an_empty_path_says_what_to_do(tmp_path):
    app = KalfaApp(root=str(tmp_path))
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.press("4")
        await pilot.pause()
        assert names_of(app) == []
        message = str(app.query_one("#guide-message").render())
        assert "path is empty" in message and "config tab" in message


@pytest.mark.asyncio
async def test_bad_set_pairs_are_reported_in_the_guide(tmp_path):
    app = KalfaApp(root=str(tmp_path), config=str(EXAMPLE))
    async with app.run_test(size=(140, 40)) as pilot:
        app.query_one("#config-sets").value = "train.epochs"
        await pilot.press("4")
        await pilot.pause()
        assert names_of(app) == []
        assert "--set takes NAME=VALUE pairs" in str(app.query_one("#guide-message").render())
