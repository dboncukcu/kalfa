import importlib.util
import os
import shutil

import pytest

from helpers import EXAMPLES
from kalfa.api import run
from kalfa.config import parse_sets


CPU = ["device=cpu"]
SMALL = {
    "01_mlp_regression": (["rng=indexed"], ["epochs=3"]),
    "02_mlp_classification": ([], ["epochs=3"]),
    "03_timeseries_window": ([], ["epochs=2"]),
    "04_cnn_images": (CPU + ["data.batch.size=16", "data.batch.eval_size=32",
                             "data.preprocessors.resize={uri: resize, params: {size: 32}}",
                             "data.preprocessors.augment={uri: random_crop_flip, params: {size: 32}, sets: [train]}"],
                      ["epochs=2", "unfreeze_at=1"]),
    "05_autoencoder": (["data.batch=16"], ["epochs=2"]),
    "06_vae_loss_dynamics": (["data.batch=16"], ["epochs=3", "kl_warmup_steps=6"]),
    "07_wgan_gp": (CPU + ["data.batch.size=8", "metrics.fid.params.n=16", "generate.params.n=4",
                          "metrics.fid.params.extractor={uri: /lego/kalfa/pixel_features}"], ["epochs=5"]),
    "08_ddpm": (CPU + ["data.batch=8", "optimizers.main.schedule={uri: warmup_cosine, params: {warmup: 5, total: 60}}",
                       "generate.params.n=4"], ["epochs=3", "diffusion_steps=20"]),
    "09_simclr": (CPU + ["data.batch.size=16", "data.preprocessors.simclr_aug.params.size=32"], ["epochs=2"]),
    "10_char_lm": (CPU + ["data.batch=4", "training.amp=false",
                          "model.models.gpt.nodes=[{uri: embedding, params: {num: {uri: vocab_size}, dim: 16}}, "
                          "{uri: gpt, params: {d_model: 16, layers: 1, heads: 2, seq_len: 16}}, "
                          "{uri: linear, params: {out_features: {uri: vocab_size}}}]",
                          "optimizers.main.schedule={uri: warmup_cosine, params: {warmup: 1, total: 12}}",
                          "generate.params.max_new_tokens=20"], ["total_steps=8", "turn_steps=2", "seq_len=16"]),
    "11_kfold_cv": ([], ["epochs=1", "fold=0"]),
    "12_resume": ([], ["epochs=2"]),
    "13_distillation": (CPU + ["data.batch.size=16", "data.batch.eval_size=32"], ["epochs=2"]),
    "14_sweep_grid": ([], ["epochs=1"]),
    "15_multi_target": ([], ["epochs=4"]),
    "alad": ([], ["epochs=2"]),
    "minimal": (["rng=indexed"], ["epochs=3"]),
}


def load_make_data(folder):
    spec = importlib.util.spec_from_file_location(f"make_data_{folder.name}", folder / "make_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def examples_copy(tmp_path_factory):
    copy = tmp_path_factory.mktemp("examples")
    shutil.copytree(EXAMPLES, copy, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("runs", "data", "*.parquet", "__pycache__"))
    return copy


@pytest.fixture(scope="session")
def prepared(examples_copy):
    done = set()

    def ensure(name):
        folder = examples_copy / name
        if name not in done:
            previous = os.getcwd()
            os.chdir(folder)
            try:
                load_make_data(folder).main()
            finally:
                os.chdir(previous)
            done.add(name)
        return folder
    return ensure


@pytest.fixture
def dataset(prepared, monkeypatch):
    def enter(name):
        folder = prepared(name)
        monkeypatch.chdir(folder)
        return folder
    return enter


@pytest.fixture(scope="session")
def records():
    return {}


@pytest.fixture
def trained(dataset, records):
    def get(name, when="fixed"):
        dataset(name)
        key = (name, when)
        if key not in records:
            sets, params = SMALL[name]
            records[key] = run(["config.yaml"], parse_sets(sets, params), when=when)
        return records[key]
    return get
