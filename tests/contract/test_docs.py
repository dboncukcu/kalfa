import re

from helpers import ROOT
from kalfa.docs import render, template_uris


def listed_uris(section):
    return set(re.findall(r"^\| `(/[^`]+)`", section, re.M))


def test_the_reference_lists_the_std_and_the_packs():
    text = render()
    assert "| `/criterion/kalfa/mae` | `mae` |" in text and "### /alias/kalfa/tabular" in text
    assert "## strategy" in text and "## data" in text and "/lego/test/" not in text


def test_the_skeleton_section_is_exactly_what_the_template_calls():
    text = (ROOT / "DOCS.md").read_text()
    catalog, _, rest = text.partition("## Skeleton steps")
    skeleton, _, packs = rest.partition("## Alias packs")
    called = template_uris()
    assert called, "the template calls no registered lego"
    assert listed_uris(skeleton) == called
    assert not listed_uris(catalog) & called
    assert "/lego/kalfa/fit" in called and "/builder/kalfa/module" in called
    assert "/criterion/kalfa/mae" not in called and "/adapter/kalfa/criterion" not in called
    assert "not written in a config" in skeleton and listed_uris(packs) == set()
