import builtins
import io
from pathlib import Path
from types import SimpleNamespace

import pytest

import kalfa
from helpers import housing_frame
from kalfa.core import run as running
from kalfa.core.aliases import ALIASES, register
from kalfa.core.errors import BuildError
from kalfa.core.protocols import Logger
from kalfa.core.record import RecordWriter
from kalfa.std import loggers
from kalfa.std.loggers import Echo, TensorBoard


LINE = {"turn": 2, "update": 10, "train/mse": 3.0, "val/rmse": 0.123456, "lr/main": 0.01, "lr/main/0": 0.02,
        "minimizes/main": "mse", "seconds": 0.5, "rules": ["warm"], "flag": True}
UPDATE = {"update": 7, "turn": 2, "loss/main": 0.5, "lr/main": 0.01, "grad_norm/main": 1.25}


class FakeWriter:
    def __init__(self, made, log_dir):
        self.log_dir = log_dir
        self.scalars = []
        self.flushes = 0
        self.closed = False
        made.append(self)

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))

    def flush(self):
        self.flushes += 1

    def close(self):
        self.closed = True


@pytest.fixture
def made(monkeypatch):
    writers = []
    monkeypatch.setattr(loggers, "summary_writer", lambda: lambda log_dir: FakeWriter(writers, log_dir))
    return writers


@pytest.fixture(scope="module")
def home(tmp_path_factory):
    root = tmp_path_factory.mktemp("loggers")
    housing_frame(160, 0).to_parquet(root / "housing.parquet", index=False)
    return root


def housing_config(home, log, epochs=1):
    return {
        "seed": 3,
        "device": "cpu",
        "data": {"use": "table", "file": str(home / "housing.parquet"), "split": [0.7, 0.15, 0.15],
                 "features": {"x*": "standard_scaler"}, "targets": {"price": "standard_scaler"}, "batch": 32},
        "models": {"net": {"use": "mlp", "hidden": [8], "out": 1}},
        "losses": {"mse": "mse"},
        "metrics": {"rmse": "rmse"},
        "optimizers": {"main": {"use": "adam", "lr": 0.01, "over": "@models.net", "minimizes": "@losses.mse"}},
        "train": {"epochs": epochs, "checkpoint": {"watch": "val/rmse"}, "report": "best", "log": log},
    }


def test_echo_prints_the_chosen_series_to_its_stream():
    stream = io.StringIO()
    echo = Echo(series=["val/rmse", "train/mse", "absent", "minimizes/main"], stream=stream)
    ctx = SimpleNamespace(record=None)
    echo.bind(ctx)
    echo.update(ctx, UPDATE)
    echo.turn(ctx, LINE)
    echo.close(ctx)
    assert stream.getvalue() == "turn 2  val/rmse 0.1235  train/mse 3\n"


def test_echo_prints_every_numeric_series_to_stderr(capsys):
    Echo().turn(SimpleNamespace(), LINE)
    assert capsys.readouterr().err == "turn 2  train/mse 3  val/rmse 0.1235  lr/main 0.01  lr/main/0 0.02\n"


def test_echo_takes_one_series_as_a_string():
    assert Echo(series="val/rmse").series == ["val/rmse"]
    assert Echo().series is None


def test_tensorboard_writes_turn_series_and_update_series_into_the_record(made, tmp_path):
    ctx = SimpleNamespace(record=RecordWriter(tmp_path))
    logger = TensorBoard(flush_every=2)
    logger.bind(ctx)
    assert logger.writer is None and made == []
    logger.update(ctx, UPDATE)
    logger.turn(ctx, LINE)
    writer = made[0]
    assert writer.log_dir == str(tmp_path / "tensorboard")
    assert writer.scalars == [("loss/main", 0.5, 7), ("lr/main", 0.01, 7), ("grad_norm/main", 1.25, 7),
                              ("train/mse", 3.0, 2), ("val/rmse", 0.123456, 2), ("lr/main/0", 0.02, 2)]
    assert writer.flushes == 0
    logger.turn(ctx, {**LINE, "turn": 3})
    assert writer.flushes == 1 and len(made) == 1
    logger.close(ctx)
    assert writer.closed and writer.flushes == 2 and logger.writer is None
    logger.close(ctx)


def test_tensorboard_without_updates_writes_the_turn_lr(made, tmp_path):
    ctx = SimpleNamespace(record=None)
    logger = TensorBoard(dir=str(tmp_path / "board"), updates=False)
    logger.bind(ctx)
    logger.update(ctx, UPDATE)
    assert made == []
    logger.turn(ctx, LINE)
    writer = made[0]
    assert writer.log_dir == str(tmp_path / "board")
    assert ("lr/main", 0.01, 2) in writer.scalars and writer.flushes == 1
    assert {tag for tag, value, step in writer.scalars} == {"train/mse", "val/rmse", "lr/main", "lr/main/0"}
    logger.close(ctx)


def test_tensorboard_needs_a_dir_without_a_record(made):
    logger = TensorBoard()
    logger.path = "train.log[0]"
    ctx = SimpleNamespace(record=None)
    logger.bind(ctx)
    with pytest.raises(BuildError, match=r"train\.log\[0\]: the tensorboard logger needs a dir"):
        logger.turn(ctx, LINE)
    assert made == []


def test_tensorboard_flush_every_is_a_positive_integer():
    with pytest.raises(BuildError, match="flush_every"):
        TensorBoard(flush_every=0)


def test_tensorboard_without_the_package_names_the_extra(monkeypatch):
    original = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "tensorboard" or name.startswith(("tensorboard.", "torch.utils.tensorboard")):
            raise ImportError(f"No module named {name!r}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    with pytest.raises(BuildError) as raised:
        TensorBoard().bind(SimpleNamespace(record=None))
    assert str(raised.value) == ("tensorboard logger needs the tensorboard package (uv sync installs the dependencies "
                                 "of kalfa)")


def test_tensorboard_writes_event_files(tmp_path):
    ctx = SimpleNamespace(record=RecordWriter(tmp_path))
    logger = TensorBoard()
    logger.bind(ctx)
    logger.update(ctx, UPDATE)
    logger.turn(ctx, LINE)
    logger.turn(ctx, {**LINE, "turn": 3})
    logger.close(ctx)
    assert list((tmp_path / "tensorboard").glob("events.out.tfevents.*"))


@pytest.mark.slow
def test_echo_logger_is_built_as_a_train_log_node(home, tmp_path, capsys):
    config = housing_config(home, [{"use": "echo", "series": ["val/rmse"]}])
    node = kalfa.check(config).plan["nodes"]["train.log[0]"]
    assert node["slot"] == "logger" and node["use"] == "kalfa.std.loggers.Echo"
    kalfa.run(config, record=str(tmp_path / "echo"))
    lines = [line for line in capsys.readouterr().err.splitlines() if line.startswith("turn 1  ")]
    assert len(lines) == 1 and lines[0].startswith("turn 1  val/rmse ")


@pytest.mark.slow
def test_tensorboard_logger_opens_in_the_record_of_a_run(made, home, tmp_path):
    result = kalfa.run(housing_config(home, ["tensorboard"], epochs=2), record=str(tmp_path / "tb"))
    writer = made[0]
    assert len(made) == 1 and writer.log_dir == str(Path(result.record) / "tensorboard")
    turns = [step for tag, value, step in writer.scalars if tag == "val/rmse"]
    updates = [step for tag, value, step in writer.scalars if tag == "loss/main"]
    assert turns == [1, 2] and updates == list(range(1, len(updates) + 1)) and len(updates) > 2
    assert writer.closed and writer.flushes == 3
    opened = running.open_record(result.record, data=False)
    assert opened.loggers == [] and len(made) == 1


class Closing(Logger):
    closed = 0

    def turn(self, ctx, line):
        raise RuntimeError("closing logger")

    def close(self, ctx):
        type(self).closed += 1


@pytest.mark.slow
def test_a_failing_run_still_closes_its_loggers(home, tmp_path):
    register("t_loggers_closing", Closing, slot="logger", doc="raises on its first turn")
    try:
        with pytest.raises(RuntimeError, match="closing logger"):
            kalfa.run(housing_config(home, ["t_loggers_closing"]), record=str(tmp_path / "closing"))
    finally:
        ALIASES.pop("t_loggers_closing", None)
    assert Closing.closed == 1
