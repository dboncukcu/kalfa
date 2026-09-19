import re
from pathlib import Path

from kalfa.std import STD_URIS


URI = re.compile(r'"(/[a-z_]+/[a-z_]+/[a-z0-9_]+)"')


def mentioned():
    found = set()
    for path in Path(__file__).parent.glob("test_*.py"):
        if path.name != Path(__file__).name:
            found |= set(URI.findall(path.read_text()))
    return found


def test_every_std_lego_is_exercised_by_a_test_of_its_kind():
    catalog = {uri for uri in STD_URIS if not uri.startswith("/alias/")}
    assert sorted(catalog - mentioned()) == []


def test_every_mentioned_uri_of_a_std_pack_is_a_std_lego():
    packs = {tuple(uri.split("/")[1:3]) for uri in STD_URIS}
    strayed = [uri for uri in mentioned() if tuple(uri.split("/")[1:3]) in packs and uri not in STD_URIS]
    assert sorted(strayed) == []


def test_every_std_lego_has_a_kind_file():
    kinds = {uri.split("/")[1] for uri in STD_URIS if not uri.startswith("/alias/")}
    files = {path.name for path in Path(__file__).parent.glob("test_*.py")}
    for kind in kinds - {"lego"}:
        assert f"test_{kind}.py" in files, kind
