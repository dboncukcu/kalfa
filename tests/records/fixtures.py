import json
from pathlib import Path

import numpy
import pandas


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1), encoding="utf-8")


def write_lines(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")


def fake_plan(targets=("price",), classes=None, watch="val/rmse", mode="min", outputs=("net",)):
    fields = {"x": {"columns": ["x0", "x1", "x2", "x3"], "preprocessors": ["standard_scaler"]}}
    for name in targets:
        fields[name] = {"target": True, "preprocessors": ["label" if classes else "standard_scaler"]}
    return {
        "format": 1, "kalfa": "3.0.0.dev0", "seed": 7, "device": None, "record": None,
        "nodes": {
            "data": {"slot": "data", "use": "kalfa.std.data.table.Table",
                     "params": {"file": "housing.parquet", "split": [0.7, 0.15, 0.15],
                                "targets": {name: "label" if classes else "standard_scaler" for name in targets}},
                     "keys": {}, "fixed": [], "signature": "checked"},
            "models.net": {"slot": "model", "use": "kalfa.std.models.Mlp", "params": {"hidden": [16], "out": 1},
                           "keys": {"outputs": list(outputs)}, "fixed": [], "signature": "checked"},
            "losses.mse": {"slot": "loss", "use": "torch.nn.MSELoss", "params": {},
                           "keys": {"pred": outputs[0], "target": targets[0]}, "fixed": [], "signature": "checked"},
            "metrics.rmse": {"slot": "metric", "use": "kalfa.std.metrics.rmse", "params": {},
                             "keys": {"pred": outputs[0], "target": targets[0], "rescale": True}, "fixed": [],
                             "signature": "checked"},
            "optimizers.main": {"slot": "optimizer", "use": "torch.optim.Adam", "params": {"lr": 0.001},
                                "keys": {"over": [{"ref": "models.net"}], "minimizes": {"ref": "losses.mse"}},
                                "fixed": [], "signature": "checked"},
            "train.step": {"slot": "step", "use": "kalfa.std.steps.Supervised", "params": {}, "keys": {},
                           "fixed": [], "signature": "checked"},
        },
        "data": {"inputs": ["x"], "exposes": ["n_features", "n_classes"], "fields": fields,
                 "classes": list(classes) if classes else None},
        "train": {"epochs": 10, "checkpoint": {"watch": watch, "mode": mode}, "report": "best"},
        "rules": [{"id": "to_huber", "when": {"use": "after_epoch", "at": 2}, "after": None, "once": True,
                   "patch": [{"path": "optimizers.main.minimizes", "value": {"ref": "losses.huber"}}],
                   "stop": False, "run": None}],
        "series": ["train/mse", "val/mse", "test/mse", "val/rmse", "test/rmse", "lr/main", "minimizes/main"],
        "mutable": ["optimizers.main.lr", "optimizers.main.minimizes", "train.epochs"],
    }


def fake_history(turns=3, seed=0, series=("train/mse", "val/mse", "val/rmse", "test/rmse")):
    generator = numpy.random.default_rng(seed)
    lines = []
    for turn in range(1, turns + 1):
        line = {"turn": turn, "update": turn * 4}
        for key in series:
            line[key] = round(float(1.0 / turn + generator.normal(scale=0.01)), 6)
        line["lr/main"] = 0.001
        line["minimizes/main"] = "mse" if turn < 2 else "huber"
        line["seconds"] = 0.5
        line["rules"] = ["to_huber"] if turn == 2 else []
        lines.append(line)
    return lines


def fake_predictions(rows=50, seed=0, classes=None):
    generator = numpy.random.default_rng(seed)
    frame = pandas.DataFrame({"row": numpy.arange(rows)})
    if classes:
        truth = generator.integers(0, len(classes), size=rows)
        logits = generator.normal(size=(rows, len(classes)))
        logits[numpy.arange(rows), truth] += 1.5
        probs = numpy.exp(logits) / numpy.exp(logits).sum(axis=1, keepdims=True)
        guess = probs.argmax(axis=1)
        frame["label"] = numpy.asarray(classes)[truth]
        frame["pred_net"] = numpy.asarray(classes)[guess]
        frame["raw_net"] = guess
        for position, name in enumerate(classes):
            frame[f"prob_net_{name}"] = probs[:, position]
    else:
        truth = generator.normal(loc=200.0, scale=20.0, size=rows)
        frame["price"] = truth
        frame["pred_net"] = truth + generator.normal(scale=5.0, size=rows)
        frame["raw_net"] = (frame["pred_net"] - 200.0) / 20.0
    frame["town"] = numpy.where(numpy.arange(rows) % 2 == 0, "ank", "ist")
    return frame


def make_record(directory, name=None, turns=3, status="done", params=None, classes=None, started="2026-11-07T10:00:00",
                updated="2026-11-07T10:05:00", finished="2026-11-07T10:05:00", turns_planned=10, sweep=None,
                failure=None, predictions=True, seed=0, watch="val/rmse", mode="min"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {"format": 1, "kalfa": "3.0.0.dev0", "name": name or directory.name, "started": started,
                "updated": updated, "finished": finished if status in ("done", "failed", "stopped") else None,
                "status": status, "config_files": ["config.yaml"], "params": dict(params or {"lr": 0.001}),
                "unit": "epochs", "turns_planned": turns_planned, "host": {"hostname": "box"}, "device": "cpu",
                "git": None, "sweep": sweep, "resumed_from": None, "failure": failure, "turn": turns}
    write_json(directory / "manifest.json", manifest)
    write_json(directory / "plan.json", fake_plan(targets=("label",) if classes else ("price",), classes=classes,
                                                  watch=watch, mode=mode))
    history = fake_history(turns, seed)
    write_lines(directory / "history.jsonl", history)
    write_lines(directory / "updates.jsonl", [{"update": update, "turn": (update - 1) // 4 + 1,
                                               "loss/main": round(1.0 / update, 6), "lr/main": 0.001,
                                               "grad_norm/main": 0.5}
                                              for update in range(1, turns * 4 + 1)])
    events = [{"at": started, "turn": 0, "kind": "run_start", "total": turns_planned},
              {"at": started, "turn": 2, "kind": "patch", "by": "to_huber", "path": "optimizers.main.minimizes",
               "value": {"ref": "losses.huber"}, "asked": {"ref": "losses.huber"}},
              {"at": started, "turn": 2, "kind": "checkpoint", "tags": ["best", "last"]}]
    if status == "done":
        events.append({"at": finished, "turn": turns, "kind": "run_end", "stopping": None})
    write_lines(directory / "events.jsonl", events)
    (directory / "resolved.yaml").write_text(f"seed: 7\ntrain:\n  epochs: {turns_planned}\nparams:\n  lr: "
                                             f"{manifest['params'].get('lr', 0.001)}\n", encoding="utf-8")
    write_json(directory / "model_summary.json", {"net": {"class": "kalfa.std.models.Mlp", "parameters": 97,
                                                           "trainable": 97, "tree": [], "outputs": {"net": [1]}}})
    (directory / "checkpoints").mkdir(exist_ok=True)
    (directory / "checkpoints" / "last.pt").write_bytes(b"x")
    (directory / "plots").mkdir(exist_ok=True)
    (directory / "plots" / "loss_curve.png").write_bytes(b"png")
    if predictions:
        fake_predictions(seed=seed, classes=classes).to_parquet(directory / "predictions.parquet", index=False)
    return directory


def make_sweep(root, count=4, objective=None, space=None, seed=0):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    objective = objective or {"watch": "val/rmse", "mode": "min", "at": "best"}
    space = space or {"lr": [0.01, 0.001], "width": [32, 64]}
    write_json(root / "sweep.json", {"format": 1, "strategy": {"use": "grid"}, "space": space,
                                     "objective": objective, "points": count, "started": "2026-11-07T10:00:00"})
    points = []
    values = [{"lr": lr, "width": width} for lr in space["lr"] for width in space["width"]][:count]
    for index, point in enumerate(values):
        status = "failed" if index == count - 1 and count > 2 else "done"
        directory = make_record(root / f"{index:04d}", name=f"{index:04d}", params=dict(point), status=status,
                                sweep={"root": str(root), "id": index, "values": dict(point)}, seed=seed + index,
                                failure={"type": "RuntimeError", "message": "boom"} if status == "failed" else None)
        points.append(directory)
    return root, points
