import json
import subprocess
import sys

import pytest

from kalfa import cli
from kalfa.commands import legos
from kalfa.core import aliases


ORDER = ("data", "splits", "table preprocessing", "feeds", "image transforms", "tokenizers", "models", "layers",
         "activations", "init", "losses", "objectives", "noise schedules", "metrics", "optimizers", "lr schedules",
         "steps", "triggers", "actions", "samplers", "plots", "loggers", "sweep strategies", "other")
MEMBERS = {"table": "data", "kfold": "splits", "given": "splits", "standard_scaler": "table preprocessing",
           "one_hot": "table preprocessing", "log": "table preprocessing", "window": "feeds",
           "random_flip": "image transforms", "two_views": "image transforms", "char_tokenizer": "tokenizers",
           "mlp": "models", "linear": "layers", "last_step": "layers", "relu": "activations", "tanh": "activations",
           "normal_init": "init", "mse": "losses", "cross_entropy": "losses", "vae_elbo": "objectives",
           "weighted_sum": "objectives", "cosine_beta": "noise schedules", "rmse": "metrics", "adam": "optimizers",
           "warmup": "lr schedules", "alternating": "steps", "plateau": "triggers", "write_samples": "actions",
           "ddpm_sampler": "samplers", "loss_curve": "plots", "tensorboard": "loggers", "grid": "sweep strategies"}
PLACES = {"adam": "optimizers.<name>", "kfold": "data.split", "plateau": "rules.<name>.when", "warmup":
          "optimizers.<name>.schedule", "grid": "sweep.strategy", "normal_init": "models.<name>.init",
          "random_flip": "data.transform and data.train_transform", "supervised": "train.step"}
LISTING = ("import json\nfrom kalfa.commands import legos\nfrom kalfa.core import aliases\nlegos.load_registrars()\n"
           "print(json.dumps(sorted(name for name, entry in aliases.ALIASES.items() if not entry.plugin)))")


@pytest.fixture(scope="module")
def std_names():
    found = subprocess.run([sys.executable, "-c", LISTING], capture_output=True, text=True, check=True)
    return json.loads(found.stdout)


def names_of(found):
    return {category: [entry.name for entry in entries] for category, entries in found}


def test_every_std_name_sits_in_one_category_in_config_order(std_names):
    found = legos.categories()
    seen = [category for category, _ in found]
    assert seen == [category for category in ORDER if category in seen]
    placed = [entry.name for _, entries in found for entry in entries if entry.name in std_names]
    assert sorted(placed) == sorted(std_names) and len(placed) == len(set(placed))
    assert not [name for name in std_names if legos.category_of(name) == "other"]
    for name, category in MEMBERS.items():
        assert legos.category_of(name) == category, name
        assert legos.category_of(aliases.entry_of(name)) == category, name


def test_places_tell_where_a_lego_goes_in_a_config():
    for name, place in PLACES.items():
        assert legos.place_of(name) == place, name
    assert legos.place_of("qqzzxxw") is None


def test_search_and_slot_filter_the_categories():
    assert names_of(legos.categories("scaler")) == {
        "table preprocessing": ["minmax_scaler", "robust_scaler", "standard_scaler"]}
    found = names_of(legos.categories(slot="value"))
    assert "splits" in found and "layers" in found and "losses" not in found
    assert legos.categories("qqzzxxw") == []


def test_the_catalog_and_the_pages_follow_the_categories():
    text = legos.catalog_text()
    assert text.splitlines()[0] == "data" and "\ntable preprocessing\n" in text and "\nobjectives\n" in text
    page = legos.entry_text("adam").splitlines()
    assert any(line.split() == ["category", "optimizers"] for line in page)
    assert any(line.split()[:2] == ["goes", "in"] and line.split()[2:] == ["optimizers.<name>"] for line in page)
    docs = legos.docs_markdown()
    assert "## table preprocessing" in docs and "### standard_scaler" in docs


def test_the_cli_lists_categories(capsys):
    assert cli.main(["legos"]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0] == "data" and "image transforms" in out
