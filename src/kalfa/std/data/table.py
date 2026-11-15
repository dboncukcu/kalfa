import json
import logging
from pathlib import Path

import joblib
import numpy
import pandas
import torch
from torch.utils.data import Dataset

from kalfa.core import aliases
from kalfa.core.errors import BuildError, error, warning
from kalfa.std.data import flow
from kalfa.std.data.base import Loaders, config_directories, seeded_split
from kalfa.std.data.columns import (
    Block,
    Target,
    apply_step,
    as_matrix,
    chain_of,
    floats_of,
    inverts,
    is_pattern,
    label_step,
    mapping_of,
    matches,
    matches_any,
    name_problems,
    names_after,
    names_of,
    owners_of,
    specificity,
    step_names,
    step_of,
    target_of,
    tensor_of,
)
from kalfa.std.data.options import (
    BATCH_DEFAULTS,
    FILTER_KEYS,
    SET_NAMES,
    STREAM_SPLIT,
    batch_of,
    batch_spec,
    filters_of,
    locate,
    query,
    read_table,
    split_check,
    split_of,
    split_target,
    stream_check,
    stream_split_check,
)
from kalfa.std.data.weights import scale_of, weights_of


__all__ = ["BATCH_DEFAULTS", "Block", "FILTER_KEYS", "REGISTRATIONS", "SET_NAMES", "STREAM_SPLIT", "Table",
           "TableDataset", "Target", "VALIDATORS", "apply_step", "as_matrix", "batch_of", "batch_spec", "chain_of",
           "filters_of", "floats_of", "inverts", "is_pattern", "label_step", "locate", "mapping_of", "matches",
           "matches_any", "names_after", "names_of", "owners_of", "query", "read_table", "register_aliases",
           "specificity", "split_check", "split_of", "split_target", "step_names", "step_of", "stream_check",
           "stream_split_check", "target_of", "tensor_of"]

logger = logging.getLogger("kalfa.data")

REGISTRATIONS = (
    ("table", "kalfa.std.data.table.Table", "data",
     "a parquet or csv table: split into sets, feature chains fitted on the train rows; the batch carries x "
     "(float32 features) and every target by name"),
    ("random", "kalfa.std.data.splits.RandomSplit", "value",
     "shuffle the rows with a seed and cut them by ratios into train, valid and test"),
    ("sequential", "kalfa.std.data.splits.SequentialSplit", "value",
     "cut the rows in their order by ratios, within each group when group names a column"),
    ("kfold", "kalfa.std.data.splits.KFoldSplit", "value",
     "fold number fold of folds shuffled folds is the test set, the rest train; valid_ratio carves a valid share"),
    ("given", "kalfa.std.data.splits.GivenSplit", "value",
     "a column's values name the set of every row: train, valid and test"),
    ("standard_scaler", "sklearn.preprocessing.StandardScaler", "value", "zero mean and unit variance per column"),
    ("minmax_scaler", "sklearn.preprocessing.MinMaxScaler", "value",
     "every column scaled into [0, 1], or into feature_range"),
    ("robust_scaler", "sklearn.preprocessing.RobustScaler", "value",
     "centered on the median and scaled by the interquartile range per column"),
    ("one_hot", "kalfa.std.data.pre.one_hot", "value",
     "one column per category, named <column>_<category>; unknown categories give zeros"),
    ("label", "kalfa.std.data.pre.Label", "value",
     "integer class codes of a label column; on a target it makes a classification task"),
    ("abs", "kalfa.std.data.pre.Abs", "value", "absolute value"),
    ("log", "kalfa.std.data.pre.Log", "value", "log(1 + x / norm) / log(base), inverted exactly"),
    ("cast", "kalfa.std.data.pre.Cast", "value", "cast to a numpy dtype"),
)


def register_aliases():
    for name, target, slot, doc in REGISTRATIONS:
        aliases.register(name, target, slot=slot, doc=doc)


class TableDataset(Dataset):
    def __init__(self, x, targets, rows):
        self.x = x
        self.targets = targets
        self.rows = rows

    def __len__(self):
        return int(self.x.shape[0])

    def __getitem__(self, position):
        item = {"x": self.x[position]}
        for name, values in self.targets.items():
            item[name] = values[position]
        return item


VALIDATORS = (
    ("features", lambda value: mapping_of(value, "features")),
    ("targets", lambda value: mapping_of(value, "targets")),
    ("stream", stream_check),
    ("filter", filters_of),
    ("drop", lambda value: names_of(value, "drop")),
    ("spectators", lambda value: names_of(value, "spectators")),
    ("batch", batch_spec),
    ("split", split_check),
)


class Table(Loaders):
    inputs = ("x",)
    exposes = ("n_features", "feature_names", "target_names", "n_classes", "class_weights", "classes")

    @classmethod
    def check(cls, params, problems, directories=()):
        if params.get("split") is None:
            problems.append(error("split_required", "data.split is required: a list of ratios such as "
                                  "[0.7, 0.15, 0.15] or a call such as {use: kfold, folds: 5, fold: 0}", "data.split"))
        file = params.get("file")
        if isinstance(file, str) and file:
            try:
                locate(file, directories)
            except BuildError as failure:
                problems.append(warning("data_file_missing", str(failure), "data.file",
                                        hint="the run looks in the working directory, then next to the config"))
        for name, validator in VALIDATORS:
            if name not in params:
                continue
            try:
                validator(params[name])
            except BuildError as failure:
                problems.append(error("data_value", str(failure), f"data.{name}"))
        for section in ("features", "targets"):
            if isinstance(params.get(section), dict):
                for key, chain in params[section].items():
                    name_problems(chain, f"data.{section}.{key}", problems)
        name_problems(params.get("split"), "data.split", problems)
        try:
            stream_split_check(params.get("stream"), params.get("split"))
        except BuildError as failure:
            problems.append(error("data_value", str(failure), "data.split"))
        try:
            weights_of(params.get("weights"), params.get("targets"), params.get("drop"))
        except BuildError as failure:
            problems.append(error("data_value", str(failure), "data.weights"))

    def __init__(self, file, split=None, features=None, targets=None, filter=None, drop=None, spectators=None,
                 feed=None, batch=128, stream=False, weights=None):
        self.file = file
        self.feed = feed
        self.engine = None
        if stream:
            from kalfa.std.data.stream import StreamEngine

            self.engine = StreamEngine(self, stream)
        self.split = split_of(split)
        self.features = mapping_of(features, "features")
        self.targets = mapping_of(targets, "targets")
        if "x" in self.targets:
            raise BuildError("a target cannot be named x; x is the feature tensor")
        self.filter = filter
        self.all_filter, self.set_filters = filters_of(filter)
        self.drop = names_of(drop, "drop")
        self.weights = weights_of(weights, self.targets, self.drop)
        self.weight_scales = {}
        self.spectator_patterns = names_of(spectators, "spectators")
        self.batch = batch_spec(batch)
        self.blocks = []
        self.fitted_targets = []
        self.feature_names = []
        self.n_features = 0
        self.target_names = list(self.targets)
        self.n_classes = None
        self.classes = None
        self.class_weights = None
        self.datasets = {}
        self.spectator_frames = {}
        self.counts = {}
        self.read_rows = None
        self.read_columns = None
        self.kept_rows = None
        self.rng = None
        self.loaders_by_set = None
        self.fitted = False

    def sets_of(self, ctx):
        if self.engine is not None:
            sets = self.engine.sets_of(ctx)
            self.read_rows = self.engine.reader.count()
            self.read_columns = len(self.engine.reader.header())
            return sets
        frame = read_table(locate(self.file, config_directories(ctx)))
        self.read_rows = len(frame)
        self.read_columns = len(frame.columns)
        frame = self.filtered(frame)
        self.kept_rows = len(frame)
        sets = {name: part for name, part in seeded_split(self.split, ctx).split(frame).items() if len(part)}
        for name, text in self.set_filters.items():
            if name in sets:
                sets[name] = query(sets[name], text, f"filter.{name}")
                if not len(sets[name]):
                    raise BuildError(f"filter.{name}: the query {text!r} leaves the {name} set empty")
        if "train" not in sets:
            raise BuildError("the split leaves no train rows")
        return {name: self.dropped(self.weighed(part)) for name, part in sets.items()}

    def materialize(self, sets, ctx):
        self.rng = getattr(ctx, "rng", None)
        self.datasets = {}
        self.spectator_frames = {}
        self.counts = {}
        previous = None
        for name, part in sets.items():
            if isinstance(part, pandas.DataFrame):
                self.datasets[name] = self.dataset_of(part, name, context=previous)
                self.spectator_frames[name] = self.spectators_of(part)
                self.counts[name] = len(part)
                previous = self.context_of(part)
            else:
                self.datasets[name] = self.engine.dataset(part, name)
                self.spectator_frames[name] = None
                self.counts[name] = self.engine.count(part)
                previous = None
        self.loaders_by_set = None
        rows = ", ".join(f"{name} {count}" for name, count in self.counts.items())
        logger.info(f"data: {rows} rows; {self.n_features} features; targets {self.target_names}")
        windows = self.windows()
        if windows is not None:
            counted = ", ".join(f"{name} {count}" for name, count in windows.items())
            logger.info(f"data: {counted} windows from the {type(self.feed).__name__} feed")

    def windows(self):
        if self.feed is None or not self.datasets:
            return None
        return {name: len(dataset) for name, dataset in self.datasets.items()}

    def fit(self, ctx):
        sets = self.sets_of(ctx)
        train = sets["train"] if self.engine is None else self.engine.fit_sample(sets)
        self.blocks = self.blocks_of(list(train.columns))
        self.fit_blocks(train)
        self.fitted_targets = self.fit_targets(train)
        self.weight_scales = self.fit_weights(train)
        self.expose()
        self.fitted = True
        self.materialize(sets, ctx)

    def rebuild(self, ctx):
        if not self.fitted:
            raise BuildError("the data is not fitted; load(directory) first")
        self.materialize(self.sets_of(ctx), ctx)

    def filtered(self, frame):
        return query(frame, self.all_filter, "filter") if self.all_filter else frame

    def dropped(self, frame):
        dropping = [column for column in self.drop if column in frame.columns]
        return frame.drop(columns=dropping) if dropping else frame

    def weighed(self, frame):
        if not self.weights:
            return frame
        added = {}
        for weight in self.weights:
            if weight.name in frame.columns and weight.name != weight.column:
                raise BuildError(f"weights.{weight.name}: the table already has a column {weight.name!r}; give the "
                                 "weight field another name")
            added[weight.name] = weight.values(frame)
        return frame.assign(**added)

    def weight_columns(self):
        names = [weight.name for weight in self.weights]
        return set(names) | {weight.column for weight in self.weights if weight.column is not None}

    def fit_weights(self, train):
        scales = {}
        for weight in self.weights:
            scales[weight.name] = scale_of(weight, train[weight.name].to_numpy(dtype="float64"))
            logger.info(f"data: weight {weight.name} is scaled by {scales[weight.name]:.6g} so its train mean is one")
        return scales

    def weight_tensors(self, frame):
        found = {}
        for weight in self.weights:
            if weight.name not in frame.columns:
                continue
            if weight.name not in self.weight_scales:
                raise BuildError(f"weights.{weight.name}: the weight was not fitted; the fitted files hold no scale "
                                 "for it")
            values = frame[weight.name].to_numpy(dtype="float64") * self.weight_scales[weight.name]
            found[weight.name] = tensor_of(values, "float32")
        return found

    def blocks_of(self, columns):
        missing = [name for name in self.targets if name not in columns]
        if missing:
            raise BuildError(f"target columns {missing} are not in the table; the columns are {columns}")
        spectators = [column for column in columns if matches_any(column, self.spectator_patterns)]
        group = getattr(self.feed, "group", None)
        weights = self.weight_columns()
        candidates = [column for column in columns if column not in self.targets and column not in spectators
                      and column != group and column not in weights]
        owners = owners_of(candidates, list(self.features))
        blocks = []
        for pattern, spec in self.features.items():
            matched = [column for column in candidates if owners.get(column) == pattern]
            if not matched:
                if any(matches(column, pattern) for column in candidates):
                    logger.info(f"features pattern {pattern!r}: more specific patterns take every column it matches")
                    continue
                raise BuildError(f"features pattern {pattern!r} matches no column; the columns are {candidates}")
            blocks.append(Block(pattern, matched, chain_of(spec, f"features.{pattern}")))
        unmatched = [column for column in candidates if column not in owners]
        if unmatched:
            logger.warning(f"columns {unmatched} match no features pattern and are not targets or spectators; "
                           f"they are dropped")
        return blocks

    def fit_blocks(self, train):
        for block in self.blocks:
            where = f"features.{block.pattern}"
            names = list(block.columns)
            values = as_matrix(train[block.columns].to_numpy())
            for step in block.chain:
                values = apply_step(step, values, where, fit=True)
                names = names_after(step, names, values.shape[1])
            floats_of(values, where)
            block.names = names
        self.feature_names = [name for block in self.blocks for name in block.names]
        self.n_features = len(self.feature_names)

    def fit_targets(self, train):
        out = []
        for name, spec in self.targets.items():
            where = f"targets.{name}"
            chain = chain_of(spec, where)
            values = as_matrix(train[[name]].to_numpy())
            for step in chain:
                values = apply_step(step, values, where, fit=True)
            target = Target(name, chain, label_step(chain), int(values.shape[1]))
            if target.label is not None:
                codes = numpy.asarray(values, dtype="int64").reshape(-1)
                target.classes = target.label.classes_.tolist()
                counts = numpy.bincount(codes, minlength=len(target.classes)).astype("float64")
                weights = 1.0 / counts
                target.class_weights = torch.as_tensor(weights / weights.mean(), dtype=torch.float32)
            else:
                floats_of(values, where)
            out.append(target)
        return out

    def expose(self):
        self.target_names = [target.name for target in self.fitted_targets]
        first = next((target for target in self.fitted_targets if target.label is not None), None)
        self.n_classes = None if first is None else len(first.classes)
        self.classes = None if first is None else list(first.classes)
        self.class_weights = None if first is None else first.class_weights

    def features_of(self, frame):
        missing = [column for block in self.blocks for column in block.columns if column not in frame.columns]
        if missing:
            raise BuildError(f"the table lacks the feature columns {missing}")
        pieces = []
        for block in self.blocks:
            where = f"features.{block.pattern}"
            values = as_matrix(frame[block.columns].to_numpy())
            for step in block.chain:
                values = apply_step(step, values, where, fit=False)
            pieces.append(floats_of(values, where))
        if not pieces:
            return torch.zeros((len(frame), 0), dtype=torch.float32)
        return tensor_of(numpy.concatenate(pieces, axis=1) if len(pieces) > 1 else pieces[0], "float32")

    def target_tensor(self, target, frame):
        where = f"targets.{target.name}"
        values = as_matrix(frame[[target.name]].to_numpy())
        for step in target.chain:
            values = apply_step(step, values, where, fit=False)
        if target.label is not None:
            return tensor_of(numpy.asarray(values).reshape(-1), "int64")
        return tensor_of(floats_of(values, where), "float32")

    def tensors_of(self, frame, set_name, strict=True):
        targets = {}
        for target in self.fitted_targets:
            if target.name not in frame.columns:
                if strict:
                    raise BuildError(f"the {set_name} set lacks the target column {target.name!r}")
                continue
            targets[target.name] = self.target_tensor(target, frame)
        targets.update(self.weight_tensors(frame))
        return self.features_of(frame), targets, numpy.asarray(frame.index)

    def groups_of(self, frame):
        group = getattr(self.feed, "group", None)
        if group is None:
            return None
        if group not in frame.columns:
            raise BuildError(f"feed.group names the column {group!r}, which the table lacks")
        return frame[group].to_numpy()

    def context_of(self, frame):
        if self.feed is None or not getattr(self.feed, "context", False):
            return None
        x, targets, rows = self.tensors_of(frame, "context", strict=False)
        return x, targets, rows, self.groups_of(frame)

    def dataset_of(self, frame, set_name, strict=True, context=None):
        if not len(frame):
            raise BuildError(f"the {set_name} set has no rows")
        x, targets, rows = self.tensors_of(frame, set_name, strict)
        if self.feed is not None:
            return self.feed.dataset(x, targets, rows, self.groups_of(frame), context)
        return TableDataset(x, targets, rows)

    def spectators_of(self, frame):
        return frame[[column for column in frame.columns if matches_any(column, self.spectator_patterns)]]

    def spectators(self, set_name):
        return self.spectator_frames.get(set_name)

    def loader_for(self, frame, set_name="test"):
        if not self.fitted:
            raise BuildError("the data is not fitted; call fit(ctx) or load(directory) first")
        if not isinstance(frame, pandas.DataFrame):
            raise BuildError(f"loader_for takes a DataFrame, got {type(frame).__name__}")
        prepared = self.dropped(self.filtered(frame))
        return self.make_loader(self.dataset_of(prepared, set_name, strict=False), train=False)

    def fitted_target(self, name):
        for target in self.fitted_targets:
            if target.name == name:
                return target
        raise BuildError(f"unknown target {name!r}; the targets are {self.target_names}")

    def inverse(self, name, values):
        target = self.fitted_target(name)
        is_tensor = isinstance(values, torch.Tensor)
        array = values.detach().cpu().numpy() if is_tensor else numpy.asarray(values)
        steps = [step for step in target.chain if inverts(step)]
        if steps:
            if array.size % target.width:
                raise BuildError(f"inverse({name!r}): the shape {array.shape} does not split into columns of width "
                                 f"{target.width}")
            flat = numpy.asarray(array, dtype="float64").reshape(-1, target.width)
            for step in reversed(steps):
                flat = as_matrix(step.inverse_transform(flat))
            out = numpy.asarray(flat, dtype="float32").reshape(array.shape)
        else:
            out = numpy.asarray(array, dtype="float32")
        if is_tensor:
            return torch.from_numpy(numpy.array(out, dtype="float32", copy=True)).to(values.device)
        return out

    def decode(self, name, values):
        target = self.fitted_target(name)
        if target.label is None:
            raise BuildError(f"target {name!r} has no label preprocessor; decode needs one")
        array = values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else numpy.asarray(values)
        if array.ndim >= 2 and array.shape[-1] != 1:
            if array.shape[-1] != len(target.classes):
                raise BuildError(f"decode({name!r}): takes class indices or scores of width {len(target.classes)}, "
                                 f"got the shape {array.shape}")
            array = array.argmax(axis=-1)
        codes = numpy.asarray(array).astype("int64").reshape(-1)
        return numpy.asarray(target.label.inverse_transform(codes))

    def rescales(self, name):
        for target in self.fitted_targets:
            if target.name == name:
                return any(inverts(step) for step in target.chain)
        return False

    def is_label(self, name):
        return any(target.name == name and target.label is not None for target in self.fitted_targets)

    def save(self, directory):
        if not self.fitted:
            raise BuildError("the data is not fitted; there is nothing to save")
        root = Path(directory) / "fitted"
        root.mkdir(parents=True, exist_ok=True)
        joblib.dump({"features": [block.chain for block in self.blocks],
                     "targets": [target.chain for target in self.fitted_targets]}, root / "preprocessors.joblib")
        fields = {
            "features": [{"pattern": block.pattern, "columns": list(block.columns), "names": list(block.names),
                          "chain": step_names(block.chain)} for block in self.blocks],
            "targets": list(self.target_names),
            "target_fields": [{"name": target.name, "width": target.width, "chain": step_names(target.chain),
                               "classes": target.classes,
                               "class_weights": None if target.class_weights is None
                               else target.class_weights.tolist()}
                              for target in self.fitted_targets],
            "feature_names": list(self.feature_names),
            "classes": self.classes,
            "drops": list(self.drop),
            "spectators": list(self.spectator_patterns),
            "filter": self.all_filter,
            "weights": [{"name": name, "scale": scale} for name, scale in self.weight_scales.items()],
        }
        (root / "fields.json").write_text(json.dumps(fields, indent=2))

    def load(self, directory):
        root = Path(directory) / "fitted"
        try:
            fields = json.loads((root / "fields.json").read_text())
            chains = joblib.load(root / "preprocessors.joblib")
        except FileNotFoundError as error:
            raise BuildError(f"{root} lacks {Path(error.filename).name}; save(directory) writes the fitted "
                             f"files") from error
        self.blocks = [Block(item["pattern"], list(item["columns"]), chain, list(item["names"]))
                       for item, chain in zip(fields["features"], chains["features"])]
        self.fitted_targets = []
        for item, chain in zip(fields["target_fields"], chains["targets"]):
            target = Target(item["name"], chain, label_step(chain), int(item["width"]), item.get("classes"))
            if item.get("class_weights") is not None:
                target.class_weights = torch.tensor(item["class_weights"], dtype=torch.float32)
            self.fitted_targets.append(target)
        self.feature_names = list(fields["feature_names"])
        self.n_features = len(self.feature_names)
        self.drop = list(fields.get("drops") or [])
        self.spectator_patterns = list(fields.get("spectators") or [])
        self.all_filter = fields.get("filter")
        self.weight_scales = {item["name"]: float(item["scale"]) for item in fields.get("weights") or []}
        self.expose()
        self.datasets = {}
        self.spectator_frames = {}
        self.counts = {}
        self.read_rows = None
        self.read_columns = None
        self.kept_rows = None
        self.loaders_by_set = None
        self.fitted = True
        return self

    def report(self):
        columns = {}
        for block in self.blocks:
            for name in block.names:
                columns[name] = step_names(block.chain)
        found = {"rows": dict(self.counts), "features": self.n_features, "targets": list(self.target_names),
                 "classes": None if self.classes is None else list(self.classes), "columns": columns,
                 "windows": self.windows()}
        if self.weights:
            found["weights"] = {weight.name: {"from": weight.source(), "scale": self.weight_scales.get(weight.name)}
                                for weight in self.weights}
        return found

    def feature_blocks(self):
        if self.fitted:
            return [(block.pattern, list(block.columns), self.features.get(block.pattern), block.chain)
                    for block in self.blocks]
        return [(pattern, None, spec, ()) for pattern, spec in self.features.items()]

    def target_specs(self):
        if self.fitted:
            return [(target.name, self.targets.get(target.name), target.chain, target.classes, target.width)
                    for target in self.fitted_targets]
        return [(name, spec, (), None, None) for name, spec in self.targets.items()]

    def set_details(self):
        notes = {name: [f"where {text}"] for name, text in self.set_filters.items()}
        if self.engine is not None:
            notes.setdefault("train", []).insert(0, "streamed")
        for name, count in (self.windows() or {}).items():
            notes.setdefault(name, []).append(f"{count} windows")
        return {name: ", ".join(parts) for name, parts in notes.items()}

    def row_lane(self, graph, source):
        rows = source
        if self.engine is not None:
            chunk, fit = self.engine.options["chunk"], self.engine.options["fit_rows"]
            rows = graph.step(rows, "stream", "stream", "stream",
                              f"chunks of {chunk} rows, chains fit on at most {fit} train rows")
        if self.all_filter and self.engine is None:
            rows = graph.step(rows, "filter", "filter", "filter", self.all_filter, self.kept_rows)
        rows = flow.split_node(graph, rows, self.split)
        if self.all_filter and self.engine is not None:
            rows = graph.step(rows, "filter", "filter", "filter", self.all_filter)
        made = flow.set_nodes(graph, rows, self.counts, self.set_details(), flow.shares_of(self.split))
        return list(made.values()) or [rows]

    def column_lane(self, graph, source):
        columns = source
        if self.drop:
            columns = graph.step(source, "drop", "drop", "drop", flow.listed(self.drop), len(self.drop))
        ends = []
        for position, (pattern, names, spec, chain) in enumerate(self.feature_blocks()):
            key = graph.step(columns, f"features:{position}", "features", pattern, flow.listed(names),
                             None if names is None else len(names))
            ends += graph.chain([key], f"features:{position}", flow.steps_of(spec, chain), "chain", pattern,
                                "feature chain, fitted on the train rows")
        x = graph.node("tensor:x", "tensor", "x", f"{self.n_features} features" if self.fitted else "features")
        graph.join(ends or [columns], x)
        tensors = [x]
        for name, spec, chain, classes, width in self.target_specs():
            key = graph.step(columns, f"targets:{name}", "targets", name,
                             flow.listed(classes) if classes else "target column", len(classes) if classes else None)
            ends = graph.chain([key], f"targets:{name}", flow.steps_of(spec, chain), "chain", name,
                               "target chain, fitted on the train rows")
            if classes:
                about = f"{len(classes)} classes"
            else:
                about = "" if width is None else f"{width} value" if width == 1 else f"{width} values"
            tensor = graph.node(f"tensor:{name}", "tensor", name, about)
            graph.join(ends, tensor)
            tensors.append(tensor)
        for weight in self.weights:
            key = graph.step(source, f"weights:{weight.name}", "weights", weight.name, weight.detail())
            tensor = graph.node(f"tensor:{weight.name}", "tensor", weight.name, "row weights, mean one on train")
            graph.join([key], tensor)
            tensors.append(tensor)
        return tensors

    def pipeline(self):
        graph = flow.Flow()
        columns = None if self.read_columns is None else f"{self.read_columns} columns"
        source = flow.source_node(graph, self.file, self.read_rows, [columns])
        ends = self.row_lane(graph, source) + self.column_lane(graph, source)
        if self.feed is not None:
            ends = [flow.feed_node(graph, ends, self.feed, self.windows())]
        flow.batch_node(graph, ends, self.batch)
        return graph.result()


register_aliases()
