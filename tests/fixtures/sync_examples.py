"""Copy the reference configs and the test plugins into examples/: every folder of TABLE holds config.yaml (the
config of configs/ as it is), the configs it includes under their own names and the plugin modules it names.
The folders of STANDALONE own their config and their plugin, nothing is copied into them.
The test tests/test_examples.py fails when a copy drifts; rerun this script after editing a config or a plugin.

    uv run python tests/fixtures/sync_examples.py
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
PLUGINS = ROOT / "tests" / "plugins"
EXAMPLES = ROOT / "examples"

TABLE = {
    "01_mlp_regression": {"config": "01_mlp_regression.yaml"},
    "02_mlp_classification": {"config": "02_mlp_classification.yaml"},
    "03_timeseries_window": {"config": "03_timeseries_window.yaml"},
    "04_cnn_images": {"config": "04_cnn_images.yaml", "plugins": ["timm_legos.py"]},
    "05_autoencoder": {"config": "05_autoencoder.yaml"},
    "06_vae_loss_dynamics": {"config": "06_vae_loss_dynamics.yaml"},
    "07_wgan_gp": {"config": "07_wgan_gp.yaml", "plugins": ["dcgan_legos.py"]},
    "08_ddpm": {"config": "08_ddpm.yaml", "plugins": ["unet_legos.py"]},
    "09_simclr": {"config": "09_simclr.yaml", "plugins": ["timm_legos.py"]},
    "10_char_lm": {"config": "10_char_lm.yaml", "plugins": ["gpt_legos.py"]},
    "11_kfold_cv": {"config": "11_kfold_cv.yaml", "includes": ["01_mlp_regression.yaml"]},
    "12_resume": {"config": "12_resume.yaml", "includes": ["01_mlp_regression.yaml"]},
    "13_distillation": {"config": "13_distillation.yaml", "plugins": ["timm_legos.py"]},
    "14_sweep_grid": {"config": "14_sweep_grid.yaml", "includes": ["01_mlp_regression.yaml"]},
    "15_multi_target": {"config": "15_multi_target.yaml"},
}

STANDALONE = ("alad", "minimal")


def copies(name):
    """(source, target) pairs of one example folder."""
    entry = TABLE[name]
    folder = EXAMPLES / name
    pairs = [(CONFIGS / entry["config"], folder / "config.yaml")]
    pairs += [(CONFIGS / include, folder / include) for include in entry.get("includes", [])]
    pairs += [(PLUGINS / plugin, folder / plugin) for plugin in entry.get("plugins", [])]
    return pairs


def sync():
    for name in TABLE:
        (EXAMPLES / name).mkdir(parents=True, exist_ok=True)
        for source, target in copies(name):
            shutil.copyfile(source, target)


if __name__ == "__main__":
    sync()
    print(f"synced {len(TABLE)} example folders under {EXAMPLES}")
