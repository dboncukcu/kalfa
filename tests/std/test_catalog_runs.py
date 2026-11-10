import json
import math
import sys
from pathlib import Path

import numpy
import pandas
import pytest
import torch
from PIL import Image

import kalfa
from kalfa.core import run as running
from kalfa.core.errors import BuildError, KalfaError
from kalfa.core.state import read_checkpoint
from kalfa.records.reader import read_lines
from kalfa.std.objectives import CosineBeta


CLASSES = ("cat", "dog", "owl")
WORDS = ("the", "kalfa", "trains", "small", "models", "on", "tiny", "data", "with", "yaml", "and", "python")
NORMALIZE = {"use": "normalize", "mean": 0.5, "std": 0.5}

LEGOS = '''import torch
from torch import nn


class SmallCnn(nn.Module):
    def __init__(self, classes, channels=3, width=8):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(channels, width, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(2),
                                      nn.Flatten())
        self.head = nn.Linear(width * 4, classes)

    def forward(self, image):
        return self.head(self.features(image))


class Autoencoder(nn.Module):
    def __init__(self, latent=4, pixels=64):
        super().__init__()
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(pixels, 16), nn.ReLU(), nn.Linear(16, latent))
        self.decoder = nn.Sequential(nn.Linear(latent, 16), nn.ReLU(), nn.Linear(16, pixels), nn.Sigmoid(),
                                     nn.Unflatten(1, (1, 8, 8)))

    def forward(self, image):
        z = self.encoder(image)
        return self.decoder(z), z


class Vae(nn.Module):
    def __init__(self, latent=4, pixels=64):
        super().__init__()
        self.body = nn.Sequential(nn.Flatten(), nn.Linear(pixels, 16), nn.ReLU())
        self.mu = nn.Linear(16, latent)
        self.logvar = nn.Linear(16, latent)
        self.decoder = nn.Sequential(nn.Linear(latent, 16), nn.ReLU(), nn.Linear(16, pixels), nn.Sigmoid(),
                                     nn.Unflatten(1, (1, 8, 8)))

    def forward(self, image):
        hidden = self.body(image)
        mu, logvar = self.mu(hidden), self.logvar(hidden)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
        return {"x_hat": self.decoder(z), "mu": mu, "logvar": logvar}


class Generator(nn.Module):
    def __init__(self, latent, classes, pixels=64):
        super().__init__()
        self.embed = nn.Embedding(classes, 4)
        self.net = nn.Sequential(nn.Linear(latent + 4, 32), nn.ReLU(), nn.Linear(32, pixels), nn.Tanh())

    def forward(self, z, label):
        return self.net(torch.cat([z, self.embed(label)], dim=1)).view(-1, 1, 8, 8)


class Critic(nn.Module):
    def __init__(self, classes, pixels=64):
        super().__init__()
        self.embed = nn.Embedding(classes, 4)
        self.net = nn.Sequential(nn.Linear(pixels + 4, 32), nn.LeakyReLU(0.2), nn.Linear(32, 1))

    def forward(self, image, label):
        return self.net(torch.cat([image.flatten(1), self.embed(label)], dim=1))


class Denoiser(nn.Module):
    def __init__(self, steps, channels=1, width=8):
        super().__init__()
        self.embed = nn.Embedding(steps, width)
        self.inlet = nn.Conv2d(channels, width, 3, padding=1)
        self.outlet = nn.Conv2d(width, channels, 3, padding=1)

    def forward(self, x, t):
        hidden = torch.relu(self.inlet(x) + self.embed(t)[:, :, None, None])
        return self.outlet(hidden)


class Encoder(nn.Module):
    def __init__(self, channels=3, width=16):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(channels, 8, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(2),
                                 nn.Flatten(), nn.Linear(32, width))

    def forward(self, image):
        if image.ndim == 5:
            image = image.flatten(0, 1)
        return self.net(image)


class CharLm(nn.Module):
    def __init__(self, vocab, width=16):
        super().__init__()
        self.embed = nn.Embedding(vocab, width)
        self.rnn = nn.GRU(width, width, batch_first=True)
        self.head = nn.Linear(width, vocab)

    def forward(self, input_ids):
        hidden, _ = self.rnn(self.embed(input_ids))
        return self.head(hidden)
'''


def write_image(path, generator, base, gray):
    path.parent.mkdir(parents=True, exist_ok=True)
    shape = (8, 8) if gray else (8, 8, 3)
    array = generator.integers(-25, 25, shape) + base
    array[:, :4] += 40
    Image.fromarray(numpy.clip(array, 0, 255).astype("uint8")).save(path)


def image_tree(root, count, gray, seed):
    generator = numpy.random.default_rng(seed)
    for position, name in enumerate(CLASSES):
        for index in range(count):
            write_image(root / name / f"{index:02d}.png", generator, 50 + 70 * position, gray)
    return root


def flat_folder(root, count, seed):
    generator = numpy.random.default_rng(seed)
    for index in range(count):
        write_image(root / f"{index:02d}.png", generator, 60 + 5 * index, gray=True)
    return root


def text_lines(lines=24, seed=4):
    generator = numpy.random.default_rng(seed)
    return "\n".join(" ".join(generator.choice(WORDS, size=6)) for _ in range(lines)) + "\n"


def series_frame(rows=60, seed=5):
    generator = numpy.random.default_rng(seed)
    pieces = []
    for position, site in enumerate(("north", "south")):
        steps = numpy.arange(rows)
        x0 = numpy.sin(steps / 5.0 + position)
        x1 = numpy.cos(steps / 7.0)
        y = 2.0 * x0 + 0.5 * x1 + generator.normal(scale=0.05, size=rows) + 10.0 * position
        pieces.append(pandas.DataFrame({"site": site, "x0": x0, "x1": x1, "y": y}))
    return pandas.concat(pieces, ignore_index=True)


def stream_frame(rows=240, seed=6):
    generator = numpy.random.default_rng(seed)
    values = generator.normal(size=(rows, 3))
    frame = pandas.DataFrame(values, columns=["x1", "x2", "x3"])
    frame["y"] = values @ numpy.array([1.0, -2.0, 0.5]) + generator.normal(scale=0.1, size=rows)
    frame["id"] = numpy.arange(1000, 1000 + rows)
    return frame


@pytest.fixture(scope="module")
def home(tmp_path_factory):
    root = tmp_path_factory.mktemp("catalog")
    (root / "catalog_legos.py").write_text(LEGOS, encoding="utf-8")
    image_tree(root / "pets", 12, gray=False, seed=1)
    image_tree(root / "digits", 10, gray=True, seed=2)
    flat_folder(root / "shapes", 24, seed=3)
    (root / "lines.txt").write_text(text_lines(), encoding="utf-8")
    series_frame().to_parquet(root / "series.parquet", index=False)
    stream_frame().to_parquet(root / "rows.parquet", index=False)
    sys.path.insert(0, str(root))
    yield root
    sys.path.remove(str(root))
    sys.modules.pop("catalog_legos", None)


def run(config, home, name):
    config = {"device": "cpu", **config, "record": str(home / "runs" / name)}
    checked = kalfa.check(config)
    assert checked.ok, checked.text()
    result = kalfa.run(config)
    record = Path(result.record)
    assert json.loads((record / "manifest.json").read_text(encoding="utf-8"))["status"] == "done"
    return record, result


def history_of(record):
    return read_lines(record / "history.jsonl")


def events_of(record, kind):
    return [event for event in read_lines(record / "events.jsonl") if event["kind"] == kind]


def image_data(folder, transform=("to_tensor", NORMALIZE), batch=8, **more):
    return {"use": "image_folder", "folder": str(folder), "split": [0.6, 0.2, 0.2], "transform": list(transform),
            "batch": batch, **more}


def adam(over, minimizes, lr=0.01, **params):
    return {"use": "adam", "lr": lr, "over": over, "minimizes": minimizes, **params}


def classification_config(home):
    return {
        "seed": 3,
        "data": image_data(home / "pets", train_transform=["to_tensor", {"use": "random_flip", "p": 0.5}, NORMALIZE]),
        "models": {"net": {"use": "catalog_legos.SmallCnn", "classes": "@data.n_classes"}},
        "losses": {"ce": {"use": "cross_entropy", "weight": "@data.class_weights"}},
        "metrics": {"acc": "accuracy", "f1": {"use": "f1", "num_classes": "@data.n_classes"}},
        "optimizers": {"main": adam("@models.net", "@losses.ce")},
        "train": {"epochs": 2, "checkpoint": {"watch": "val/acc", "mode": "max"}},
        "plots": {"confusion_matrix": "confusion_matrix", "class_histogram": "class_histogram"},
    }


@pytest.fixture(scope="module")
def classified(home):
    return run(classification_config(home), home, "classifier")


@pytest.mark.slow
def test_image_classification_decodes_labels_and_writes_probabilities(home, classified):
    record, result = classified
    history = history_of(record)
    assert [line["turn"] for line in history] == [1, 2]
    assert {"train/ce", "val/acc", "val/f1", "test/acc", "test/f1"} <= set(history[-1])
    data = result.context.data
    assert data.classes == list(CLASSES) and data.n_classes == 3 and data.image_shape == (3, 8, 8)
    assert result.context.losses["ce"].function.weight.shape == (3,)
    assert sum(data.counts.values()) == 36 and len(data.rows("test")) == data.counts["test"]
    fields = json.loads((record / "fitted" / "fields.json").read_text(encoding="utf-8"))
    assert fields["kind"] == "image_folder" and fields["classes"] == list(CLASSES) and fields["targets"] == ["label"]
    assert fields["transform"] == ["to_tensor", "normalize"]
    assert fields["train_transform"] == ["to_tensor", "random_flip", "normalize"]
    assert fields["target_fields"][0]["classes"] == list(CLASSES)
    predictions = pandas.read_parquet(record / "predictions.parquet")
    probabilities = [f"prob_net_{name}" for name in CLASSES]
    assert {"row", "label", "pred_net", "raw_net", "path", *probabilities} <= set(predictions.columns)
    assert len(predictions) == data.counts["test"]
    assert predictions["row"].tolist() == data.rows("test").tolist()
    assert set(predictions["label"]) <= set(CLASSES) and set(predictions["pred_net"]) <= set(CLASSES)
    assert set(predictions["raw_net"].tolist()) <= {0, 1, 2}
    numpy.testing.assert_allclose(predictions[probabilities].sum(axis=1), 1.0, rtol=1e-6)
    assert all(Path(path).is_file() and Path(path).parent.name == label
               for path, label in zip(predictions["path"], predictions["label"]))
    for name in ("confusion_matrix", "class_histogram"):
        assert (record / "plots" / f"{name}.png").is_file(), name
    folder = kalfa.predict(record, data=str(home / "pets"))
    assert Path(folder.path) == record / "predictions_pets.parquet"
    assert len(folder.table) == 36 and {"row", "label", "pred_net", "path", *probabilities} <= set(folder.table.columns)
    assert set(folder.table["label"]) == set(CLASSES)
    assert folder.table["row"].tolist() == list(range(36))


@pytest.mark.slow
def test_autoencoder_draws_image_pairs_and_latent_samples(home):
    config = {
        "seed": 4,
        "data": image_data(home / "digits", transform=["to_tensor"], labels=False),
        "models": {"ae": {"use": "catalog_legos.Autoencoder", "latent": 4, "outputs": ["x_hat", "z"]}},
        "losses": {"rec": {"use": "mse", "pred": "x_hat", "target": "image"}},
        "metrics": {"recon_rmse": {"use": "rmse", "pred": "x_hat", "target": "image"}},
        "optimizers": {"main": adam("@models.ae", "@losses.rec")},
        "train": {"epochs": 2, "checkpoint": {"watch": "val/recon_rmse"}, "report": "best"},
        "generate": {"use": "latent_sampler", "model": "@models.ae.decoder", "latent": 4, "n": 6},
        "plots": {"image_pairs": {"use": "image_pairs", "n": 4}, "samples_grid": {"use": "samples_grid", "n": 6}},
    }
    record, result = run(config, home, "autoencoder")
    history = history_of(record)
    assert [line["turn"] for line in history] == [1, 2]
    assert {"train/rec", "val/rec", "val/recon_rmse", "test/recon_rmse"} <= set(history[-1])
    data = result.context.data
    assert data.labels is False and data.target_names == [] and data.image_shape == (1, 8, 8)
    assert result.context.metrics["recon_rmse"].rescale is False
    for name in ("samples/samples.pt", "samples/samples.png", "plots/image_pairs.png", "plots/samples_grid.png",
                 "checkpoints/best.pt"):
        assert (record / name).is_file(), name
    samples = torch.load(record / "samples" / "samples.pt")
    assert samples.shape == (6, 1, 8, 8) and samples.dtype == torch.float32
    predictions = pandas.read_parquet(record / "predictions.parquet")
    assert {"row", "pred_z_0", "pred_z_3", "path", "label"} <= set(predictions.columns)
    assert not any(column.startswith("pred_x_hat") for column in predictions.columns)
    assert len(predictions) == data.counts["test"]


@pytest.mark.slow
def test_vae_warms_the_kl_weight_and_a_rule_swaps_the_reconstruction(home):
    config = {
        "seed": 5,
        "data": image_data(home / "digits", transform=["to_tensor"], labels=False),
        "models": {"vae": {"use": "catalog_legos.Vae", "latent": 4, "outputs": ["x_hat", "mu", "logvar"]}},
        "losses": {
            "mse": {"use": "mse", "pred": "x_hat", "target": "image"},
            "huber": {"use": "huber", "pred": "x_hat", "target": "image"},
            "elbo": {"use": "vae_elbo", "recon": "@losses.mse", "w_rec": 1.0,
                     "w_kl": {"use": "linear_warmup", "start": 0.0, "end": 1.0, "steps": 6}},
        },
        "optimizers": {"main": adam("@models.vae", "@losses.elbo")},
        "train": {"epochs": 2},
        "rules": {"shift": {"when": {"use": "after_epoch", "at": 1},
                            "set": {"losses.elbo.w_rec": 0.5, "losses.elbo.recon": "@losses.huber"}}},
        "plots": {"loss_curve": {"use": "loss_curve", "series": ["train/elbo/recon", "train/elbo/kl", "val/mse"]}},
    }
    record, result = run(config, home, "vae")
    history = history_of(record)
    assert [line["turn"] for line in history] == [1, 2]
    assert {"train/elbo", "train/elbo/recon", "train/elbo/kl", "train/elbo/w_kl", "train/mse", "train/huber",
            "val/elbo/kl", "val/mse", "test/elbo/w_kl"} <= set(history[0])
    assert 0.0 <= history[0]["train/elbo/w_kl"] < history[1]["train/elbo/w_kl"] <= 1.0
    assert history[0]["val/elbo/w_kl"] == pytest.approx(0.5) and history[1]["val/elbo/w_kl"] == pytest.approx(1.0)
    assert history[0]["rules"] == ["shift"] and history[1]["rules"] == []
    patches = events_of(record, "patch")
    assert [(event["path"], event["value"], event["turn"]) for event in patches] == [
        ("losses.elbo.w_rec", 0.5, 1), ("losses.elbo.recon", {"ref": "losses.huber"}, 1)]
    ctx = result.context
    assert ctx.losses["elbo"].w_rec == 0.5 and ctx.losses["elbo"].recon is ctx.losses["huber"]
    plan = json.loads((record / "plan.json").read_text(encoding="utf-8"))
    assert plan["nodes"]["losses.elbo"]["params"]["w_rec"] == 0.5
    assert plan["nodes"]["losses.elbo"]["params"]["recon"] == {"ref": "losses.huber"}
    assert (record / "plots" / "loss_curve.png").is_file()


@pytest.mark.slow
def test_wgan_gp_alternates_scores_fid_and_writes_samples_by_rule(home):
    config = {
        "seed": 6,
        "data": image_data(home / "digits", batch=6),
        "models": {
            "generator": {"use": "catalog_legos.Generator", "latent": 8, "classes": "@data.n_classes",
                          "inputs": ["z", "label"], "outputs": ["fake"]},
            "critic": {"use": "catalog_legos.Critic", "classes": "@data.n_classes", "inputs": ["image", "label"],
                       "outputs": ["score"]},
        },
        "losses": {
            "d": {"use": "wgan_gp_d", "generator": "@models.generator", "critic": "@models.critic", "latent": 8,
                  "real": "image", "label": "label", "gp_weight": 10.0},
            "g": {"use": "wgan_g", "generator": "@models.generator", "critic": "@models.critic", "latent": 8,
                  "label": "label"},
        },
        "metrics": {"fid": {"use": "fid", "model": "@models.generator", "latent": 8, "real": "image", "label": "label",
                            "n": 16, "every": 2}},
        "optimizers": {
            "d": adam("@models.critic", "@losses.d", lr=1.0e-4, betas=[0.0, 0.9]),
            "g": adam("@models.generator", "@losses.g", lr=1.0e-4, betas=[0.0, 0.9]),
        },
        "train": {"epochs": 2, "step": {"use": "alternating", "order": ["d", "g"], "repeat": {"d": 2, "g": 1},
                                        "fresh_batch": True}},
        "rules": {"samples": {"when": {"use": "every", "epochs": 1}, "once": False,
                              "run": {"use": "write_samples", "n": 4}}},
        "generate": {"use": "gan_sampler", "model": "@models.generator", "latent": 8, "n_classes": "@data.n_classes",
                     "n": 6},
        "plots": {"loss_curve": {"use": "loss_curve", "series": ["train/d", "train/g", "val/fid"]},
                  "samples_grid": {"use": "samples_grid", "file": "turn_2"}},
    }
    record, result = run(config, home, "wgan")
    history = history_of(record)
    assert [line["turn"] for line in history] == [1, 2]
    assert {"train/d", "train/d/wasserstein", "train/d/gp", "train/g", "val/g", "test/g"} <= set(history[0])
    assert "val/fid" not in history[0] and math.isfinite(history[1]["val/fid"]) and history[1]["val/fid"] >= 0.0
    assert history[-1]["update"] == 2 and history[-1]["minimizes/d"] == "d" and history[-1]["minimizes/g"] == "g"
    assert [event["rules"] for event in events_of(record, "turn")] == [["samples"], ["samples"]]
    for name in ("samples/turn_1.pt", "samples/turn_1.png", "samples/turn_2.pt", "samples/turn_2.png",
                 "samples/samples.pt", "samples/samples.png", "plots/samples_grid.png", "plots/loss_curve.png"):
        assert (record / name).is_file(), name
    assert torch.load(record / "samples" / "turn_1.pt").shape == (4, 1, 8, 8)
    assert torch.load(record / "samples" / "samples.pt").shape == (6, 1, 8, 8)
    assert not (record / "predictions.parquet").exists()
    assert result.context.predicts() is None
    summary = json.loads((record / "model_summary.json").read_text(encoding="utf-8"))
    assert summary["critic"]["outputs"] == {"score": [6, 1]} and list(summary["generator"]["outputs"]) == ["fake"]
    assert summary["generator"]["graph_from"] == "first training call"
    assert result.context.step.repeat == {"d": 2, "g": 1} and result.context.step.fresh_batch is True


@pytest.mark.slow
def test_ddpm_trains_the_denoiser_without_predictions_and_samples_in_few_steps(home):
    config = {
        "seed": 7,
        "data": image_data(home / "shapes"),
        "models": {"unet": {"use": "catalog_legos.Denoiser", "steps": 10}},
        "losses": {"ddpm": {"use": "ddpm", "model": "@models.unet", "schedule": {"use": "cosine_beta", "steps": 10},
                            "x": "image"}},
        "optimizers": {"main": adam("@models.unet", "@losses.ddpm")},
        "train": {"epochs": 2, "predicts": False},
        "generate": {"use": "ddpm_sampler", "model": "@models.unet", "schedule": "@losses.ddpm.schedule",
                     "shape": [1, 8, 8], "n": 4, "steps": 3},
        "plots": {"samples_grid": "samples_grid"},
    }
    record, result = run(config, home, "ddpm")
    history = history_of(record)
    assert [line["turn"] for line in history] == [1, 2]
    assert {"train/ddpm", "val/ddpm", "test/ddpm"} <= set(history[-1])
    ctx = result.context
    data = ctx.data
    assert data.labels is False and data.classes is None and data.target_names == [] and data.image_shape == (1, 8, 8)
    assert ctx.predicts() is None and "test" in ctx.loaders()
    assert not (record / "predictions.parquet").exists()
    plan = json.loads((record / "plan.json").read_text(encoding="utf-8"))
    assert plan["train"]["predicts"] is False
    assert isinstance(ctx.generate.schedule, CosineBeta) and ctx.generate.schedule is ctx.losses["ddpm"].schedule
    samples = torch.load(record / "samples" / "samples.pt")
    assert samples.shape == (4, 1, 8, 8) and float(samples.abs().max()) <= 1.0
    assert (record / "samples" / "samples.png").is_file() and (record / "plots" / "samples_grid.png").is_file()


@pytest.mark.slow
def test_simclr_trains_on_two_views_and_predicts_embeddings(home):
    config = {
        "seed": 8,
        "data": image_data(home / "pets", transform=["to_tensor"], labels=False,
                           train_transform=[{"use": "two_views",
                                             "each": ["random_flip", {"use": "random_crop", "size": 8, "padding": 1},
                                                      "to_tensor"]}]),
        "models": {"encoder": {"use": "catalog_legos.Encoder", "width": 16, "outputs": ["h"]},
                   "head": {"use": "mlp", "hidden": [16], "out": 8, "in_features": 16, "inputs": ["h"]}},
        "losses": {"ntx": {"use": "ntxent", "encoder": "@models.encoder", "projector": "@models.head",
                           "views": "image", "temperature": 0.5, "on": ["train"]}},
        "optimizers": {"main": {"use": "adamw", "lr": 1.0e-3, "over": ["@models.encoder", "@models.head"],
                                "minimizes": "@losses.ntx"}},
        "train": {"epochs": 2, "predicts": "@models.encoder"},
    }
    record, result = run(config, home, "simclr")
    history = history_of(record)
    assert [line["turn"] for line in history] == [1, 2]
    assert "train/ntx" in history[-1] and math.isfinite(history[-1]["train/ntx"])
    assert not any(key.startswith(("val/", "test/")) for key in history[-1])
    ctx = result.context
    loaders = ctx.loaders()
    batch = next(iter(loaders["train"]))
    assert tuple(batch["image"].shape) == (8, 2, 3, 8, 8) and "label" not in batch
    assert tuple(next(iter(loaders["test"]))["image"].shape)[1:] == (3, 8, 8)
    assert ctx.data.image_shape == (3, 8, 8)
    predictions = pandas.read_parquet(record / "predictions.parquet")
    assert {"row", "path", "label", *(f"pred_h_{index}" for index in range(16))} <= set(predictions.columns)
    assert len(predictions) == ctx.data.counts["test"]
    assert not hasattr(ctx.losses["ntx"], "_modules") or "encoder" not in ctx.losses["ntx"]._modules


@pytest.mark.slow
def test_distillation_freezes_a_teacher_loaded_from_the_classifier_record(home, classified):
    source, _ = classified
    config = {
        "seed": 9,
        "data": image_data(home / "pets"),
        "models": {
            "teacher": {"use": "catalog_legos.SmallCnn", "classes": "@data.n_classes",
                        "weights": {"run": str(source), "model": "net", "which": "last"}, "freeze": True},
            "student": {"use": "catalog_legos.SmallCnn", "classes": "@data.n_classes", "width": 4},
        },
        "losses": {"kd": {"use": "distill", "student": "@models.student", "teacher": "@models.teacher",
                          "temperature": 2.0, "alpha": 0.7}},
        "metrics": {"acc": "accuracy"},
        "optimizers": {"main": adam("@models.student", "@losses.kd")},
        "train": {"epochs": 2, "predicts": "@models.student", "checkpoint": {"watch": "val/acc", "mode": "max"}},
        "plots": {"loss_curve": {"use": "loss_curve", "series": ["train/kd/ce", "train/kd/kl", "val/acc"]}},
    }
    record, result = run(config, home, "distill")
    ctx = result.context
    teacher, student = ctx.models["teacher"], ctx.models["student"]
    assert teacher.freeze is True and all(not parameter.requires_grad for parameter in teacher.module.parameters())
    ctx.set_modes(True)
    assert not teacher.module.training and student.module.training
    ctx.set_modes(False)
    loaded = read_checkpoint(source / "checkpoints" / "last.pt")["nodes"]["models.net"]
    for name, value in teacher.module.state_dict().items():
        assert torch.equal(value, loaded[name]), name
    history = history_of(record)
    assert [line["turn"] for line in history] == [1, 2]
    assert {"train/kd", "train/kd/ce", "train/kd/kl", "val/kd/kl", "val/acc", "test/acc"} <= set(history[-1])
    predictions = pandas.read_parquet(record / "predictions.parquet")
    assert {"row", "label", "pred_student", "raw_student", "prob_student_cat", "path"} <= set(predictions.columns)
    assert set(predictions["pred_student"]) <= set(CLASSES)
    summary = json.loads((record / "model_summary.json").read_text(encoding="utf-8"))
    assert summary["teacher"]["trainable"] == 0 and summary["student"]["trainable"] > 0
    first = len(next(iter(ctx.loaders()["valid"]))["image"])
    assert summary["teacher"]["outputs"] == {"teacher": [first, 3]}
    assert summary["student"]["outputs"] == {"student": [first, 3]}
    assert (record / "plots" / "loss_curve.png").is_file()
    checkpoint = read_checkpoint(record / "checkpoints" / "last.pt")
    assert "losses.kd" not in checkpoint["nodes"] and {"models.teacher", "models.student"} <= set(checkpoint["nodes"])


@pytest.mark.slow
def test_char_language_model_counts_updates_scores_perplexity_and_samples_text(home):
    config = {
        "seed": 10,
        "data": {"use": "text_lines", "file": str(home / "lines.txt"), "seq_len": 8, "split": [0.7, 0.15, 0.15],
                 "batch": 8},
        "models": {"lm": {"use": "catalog_legos.CharLm", "vocab": "@data.vocab_size", "width": 16}},
        "losses": {"ce": "cross_entropy"},
        "metrics": {"ppl": "perplexity"},
        "optimizers": {"main": {"use": "adamw", "lr": 0.01, "over": "@models.lm", "minimizes": "@losses.ce"}},
        "train": {"updates": {"total": 6, "turn": 3}, "checkpoint": {"watch": "val/ppl"}},
        "generate": {"use": "lm_sampler", "model": "@models.lm", "tokenizer": "@data.tokenizer", "prompt": "the ",
                     "max_new_tokens": 12, "temperature": 0.8, "n": 2},
        "plots": {"samples_grid": "samples_grid"},
    }
    record, result = run(config, home, "charlm")
    history = history_of(record)
    assert [line["turn"] for line in history] == [1, 2] and [line["update"] for line in history] == [3, 6]
    assert {"train/ce", "val/ce", "val/ppl", "test/ppl"} <= set(history[-1])
    assert math.isfinite(history[-1]["val/ppl"]) and history[-1]["val/ppl"] > 1.0
    assert history[-1]["val/ppl"] == pytest.approx(math.exp(history[-1]["val/ce"]), rel=1e-3)
    manifest = json.loads((record / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["unit"] == "updates" and manifest["turns_planned"] == 2
    assert not (record / "predictions.parquet").exists()
    ctx = result.context
    data = ctx.data
    assert data.vocab_size == ctx.models["lm"].module.embed.num_embeddings > 10
    assert ctx.losses["ce"].target == "targets" and ctx.metrics["ppl"].rescale is False
    assert (record / "fitted" / "tokenizer.json").is_file()
    blocks = torch.load(record / "samples" / "samples.pt", weights_only=False)
    assert len(blocks) == 2 and all(block.startswith("the ") and 4 <= len(block) <= 16 for block in blocks)
    text = (record / "samples" / "samples.txt").read_text(encoding="utf-8")
    assert text == "\n\n".join(blocks) + "\n"
    assert (record / "plots" / "samples_grid.png").is_file()
    with pytest.raises(KalfaError, match="TextLines writes no predictions table; kalfa generate draws samples"):
        running.predict_record(record, device="cpu")


@pytest.mark.slow
def test_window_forecast_with_context_predicts_horizons(home):
    config = {
        "seed": 11,
        "data": {"use": "table", "file": str(home / "series.parquet"),
                 "split": {"use": "sequential", "ratios": [0.6, 0.2, 0.2], "group": "site"},
                 "features": {"x*": "standard_scaler"}, "targets": {"y": "standard_scaler"}, "spectators": ["site"],
                 "feed": {"use": "window", "size": 6, "horizon": 3, "context": True, "group": "site"}, "batch": 16},
        "models": {"net": {"use": "stack", "layers": [
            {"use": "torch.nn.GRU", "input_size": 2, "hidden_size": 8, "batch_first": True}, "last_step",
            {"use": "linear", "out_features": 3}]}},
        "losses": {"mse": "mse"},
        "metrics": {"rmse": "rmse", "mae": "mae"},
        "optimizers": {"main": adam("@models.net", "@losses.mse")},
        "train": {"epochs": 2, "checkpoint": {"watch": "val/rmse"}},
        "plots": {"forecast": {"use": "forecast_samples", "n": 4}},
    }
    record, result = run(config, home, "window")
    ctx = result.context
    data = ctx.data
    assert data.counts == {"train": 72, "valid": 24, "test": 24}
    loaders = ctx.loaders()
    assert [len(loaders[name].dataset) for name in ("train", "valid", "test")] == [56, 20, 20]
    assert data.report()["windows"] == {"train": 56, "valid": 20, "test": 20}
    batch = next(iter(loaders["valid"]))
    assert tuple(batch["x"].shape) == (16, 6, 2) and tuple(batch["y"].shape) == (16, 3)
    history = history_of(record)
    assert [line["turn"] for line in history] == [1, 2]
    assert {"train/mse", "val/rmse", "val/mae", "test/rmse"} <= set(history[-1])
    assert ctx.metrics["rmse"].rescale is True and ctx.losses["mse"].rescale is False
    predictions = pandas.read_parquet(record / "predictions.parquet")
    columns = {"row", "y_0", "y_1", "y_2", "pred_net_0", "pred_net_1", "pred_net_2", "raw_net_0", "raw_net_2"}
    assert columns <= set(predictions.columns) and len(predictions) == 20
    frame = pandas.read_parquet(home / "series.parquet")
    rows = predictions["row"].to_numpy()
    assert set(rows) <= set(range(48, 60)) | set(range(108, 120))
    assert predictions["row"].tolist() == data.rows("test").tolist()
    numpy.testing.assert_allclose(predictions["y_0"], frame.loc[rows, "y"], rtol=1e-4)
    numpy.testing.assert_allclose(predictions["y_2"], frame.loc[rows + 2, "y"], rtol=1e-4)
    assert numpy.isfinite(predictions[["pred_net_0", "pred_net_1", "pred_net_2"]].to_numpy()).all()
    assert (record / "plots" / "forecast_samples.png").is_file()
    again = kalfa.predict(record, data=str(home / "series.parquet")).table
    rows = again["row"].to_numpy()
    assert {"y_0", "y_2", "pred_net_2", "site"} <= set(again.columns) and "y" not in again.columns
    numpy.testing.assert_allclose(again["y_2"], frame.loc[rows + 2, "y"], rtol=1e-4)
    assert (again["site"].to_numpy() == frame.loc[rows + 2, "site"].to_numpy()).all()


@pytest.mark.slow
def test_streamed_table_predicts_the_test_set(home):
    config = {
        "seed": 12,
        "data": {"use": "table", "file": str(home / "rows.parquet"),
                 "split": {"use": "sequential", "ratios": [0.7, 0.15, 0.15]}, "features": {"x*": "standard_scaler"},
                 "targets": {"y": "standard_scaler"}, "spectators": ["id"], "batch": 32, "stream": {"chunk": 64}},
        "models": {"net": {"use": "mlp", "hidden": [8], "out": 1}},
        "losses": {"mse": "mse"},
        "metrics": {"rmse": "rmse"},
        "optimizers": {"main": adam("@models.net", "@losses.mse")},
        "train": {"epochs": 2},
        "plots": {"pred_vs_true": "pred_vs_true"},
    }
    record, result = run(config, home, "stream")
    ctx = result.context
    data = ctx.data
    assert data.engine is not None and data.engine.options == {"chunk": 64, "fit_rows": 100000}
    assert data.counts == {"train": 168, "valid": 36, "test": 36}
    with pytest.raises(BuildError, match="streamed"):
        data.rows("train")
    history = history_of(record)
    assert [line["turn"] for line in history] == [1, 2] and history[-1]["update"] == 2 * 6
    assert {"train/mse", "val/mse", "val/rmse", "test/rmse"} <= set(history[-1])
    predictions = pandas.read_parquet(record / "predictions.parquet")
    assert predictions["row"].tolist() == list(range(204, 240))
    assert predictions["id"].tolist() == list(range(1204, 1240))
    frame = pandas.read_parquet(home / "rows.parquet")
    numpy.testing.assert_allclose(predictions["y"], frame.loc[204:, "y"], rtol=1e-4)
    assert numpy.isfinite(predictions["pred_net"].to_numpy()).all()
    assert (record / "plots" / "pred_vs_true.png").is_file()
    again = kalfa.predict(record)
    assert again.table["row"].tolist() == list(range(204, 240))
    numpy.testing.assert_allclose(again.table["y"], predictions["y"])
    bad = {**config, "data": {**config["data"], "stream": {"chunks": 64}}}
    checked = kalfa.check(bad)
    assert [(problem.code, problem.path) for problem in checked.errors] == [("data_value", "data.stream")]
