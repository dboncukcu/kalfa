from dataclasses import dataclass, field

import torch

from kalfa.registration import lego
from kalfa.std.builder.base import Model
from kalfa.std.common.figure import Figure
from kalfa.std.common.log import logger_for
from kalfa.std.common.runtime import call_model
from kalfa.std.plot.base import report_loader


logger = logger_for("after.plots")


@dataclass
class Box:
    name: str
    kind: str
    lines: list
    column: int
    row: int = 0
    detail: list = field(default_factory=list)


@dataclass
class Layout:
    boxes: dict = field(default_factory=dict)
    arrows: list = field(default_factory=list)
    widths: dict = field(default_factory=dict)

    def add(self, key, kind, lines, column, detail=()):
        self.boxes[key] = Box(key, kind, list(lines), column, detail=list(detail))

    def link(self, source, target, wire=""):
        self.arrows.append((source, target, wire))

    def columns(self):
        table = {}
        for box in self.boxes.values():
            box.row = len(table.setdefault(box.column, []))
            table[box.column].append(box)
        return table

    def ordered(self):
        table = self.columns()
        feeding = {}
        for source, target, _ in self.arrows:
            feeding.setdefault(target, []).append(source)
        for column in sorted(table):
            boxes = table[column]
            if column == 0 or len(boxes) < 2:
                continue

            def mean_row(box):
                rows = [self.boxes[name].row for name in feeding.get(box.name, [])
                        if name in self.boxes and self.boxes[name].column < column]
                return (sum(rows) / len(rows) if rows else box.row, box.row)

            boxes.sort(key=mean_row)
            for row, box in enumerate(boxes):
                box.row = row
        return table


def shape_text(value):
    if isinstance(value, torch.Tensor):
        return "x".join(str(size) for size in value.shape)
    if isinstance(value, (tuple, list)):
        return ", ".join(shape_text(item) for item in value)
    return type(value).__name__


def shape_lines(shapes):
    return [f"{shapes[0]} -> {shapes[1]}"] if shapes else []


def widths_of(shape):
    return [part.split("x", 1)[1] if "x" in part else part for part in shape.split(", ")] if shape else []


def summary_of(module):
    extra = module.extra_repr().replace("in_features=", "").replace("out_features=", "").replace(", bias=True", "")
    extra = extra.replace(", inplace=False", "")
    return extra if len(extra) <= 34 else extra[:33] + "…"


def child_lines(module, key, shapes):
    lines = []
    for name, child in module.named_children():
        summary = summary_of(child)
        traced = shapes.get(f"{key}.{name}")
        lines.append(f"{name}: {type(child).__name__}" + (f"({summary})" if summary else "")
                     + (f" -> {traced[1]}" if traced else ""))
    return lines


def graph_of(model):
    return model.graph if isinstance(model, Model) else None


def traced_shapes(model, batch, device):
    shapes = {}
    handles = []

    def hook_for(key):
        def hook(module, inputs, output):
            shapes[key] = (shape_text(inputs[0] if len(inputs) == 1 else list(inputs)), shape_text(output))
        return hook

    graph = graph_of(model)
    targets = {node.name: model.node_module(node) for node in graph.nodes} if graph is not None else {}
    for key, module in list(targets.items()):
        for name, child in (module.named_children() if module is not None else []):
            targets[f"{key}.{name}"] = child
    targets[None] = model
    for key, module in targets.items():
        if module is not None:
            handles.append(module.register_forward_hook(hook_for(key)))
    sample = {wire: value[:2].to(device) for wire, value in batch.items() if isinstance(value, torch.Tensor)}
    try:
        with torch.no_grad():
            call_model(model, sample)
    except Exception as exception:
        logger.debug(f"architecture: no shapes, the batch does not feed the model: {exception}")
    finally:
        for handle in handles:
            handle.remove()
    return shapes


def kind_of(node, module):
    if node.ref is not None:
        return "model"
    return "torch" if type(module).__module__.startswith("torch.") else "lego"


def input_lines(wire, features):
    if not features:
        return [wire]
    shown = ", ".join(features[:4]) + (f" +{len(features) - 4}" if len(features) > 4 else "")
    return [wire, f"{len(features)} features", shown]


def graph_layout(model, label, shapes, features=None):
    layout = Layout()
    graph = graph_of(model)
    if graph is None:
        layout.add(label, "lego", [label, type(model).__name__, *shape_lines(shapes.get(None))], 0,
                   child_lines(model, None, shapes))
        return layout, 0
    depth = {wire: 0 for wire in graph.inputs}
    producer = {}
    entry = shapes.get(None)
    for wire, width in zip(graph.inputs, widths_of(entry[0]) if entry else []):
        layout.widths[wire] = width
    for wire in graph.inputs:
        named = features if features and (wire == "x" or len(graph.inputs) == 1) else None
        layout.add(f"in:{wire}", "input", input_lines(wire, named), 0, named or [])
        producer[wire] = f"in:{wire}"
    for node in graph.nodes:
        column = 1 + max((depth.get(wire, 0) for wire in node.inputs), default=0)
        module = model.node_module(node)
        what = f"model {node.ref}" if node.ref is not None else type(module).__name__
        detail = child_lines(module, node.name, shapes) if node.ref is None and module is not None else []
        shown = detail if len(detail) <= 12 else [*detail[:11], f"+{len(detail) - 11} more"]
        layout.add(node.name, kind_of(node, module), [node.name, what, *shape_lines(shapes.get(node.name)), *shown],
                   column, detail)
        traced = shapes.get(node.name)
        for wire, width in zip(node.outputs, widths_of(traced[1]) if traced else []):
            layout.widths[wire] = width
        for wire in node.inputs:
            layout.link(producer.get(wire, f"in:{wire}"), node.name, wire)
        for wire in node.outputs:
            depth[wire] = column
            producer[wire] = node.name
    last = max(box.column for box in layout.boxes.values()) + 1
    for wire in graph.outputs:
        layout.add(f"out:{wire}", "output", [wire], last)
        layout.link(producer.get(wire, f"in:{wire}"), f"out:{wire}", wire)
    return layout, last


def training_layout(layout, model, label, last, predicts, losses, losses_keys, optimizers):
    graph = graph_of(model)
    outputs = list(graph.outputs) if graph is not None else []
    owners = {name: item for name, item in (optimizers or {}).items() if label in item.models}
    minimized = {item.loss for item in owners.values()}
    for name, entry in (losses or {}).items():
        keys = (losses_keys or {}).get(name) or {}
        if entry.reads == "predictions" and label == predicts:
            output = keys.get("output") or (outputs[0] if outputs else None)
            layout.add(f"loss:{name}", "loss", [name, f"vs {keys.get('target') or 'the target'}"], last + 1)
            if output is not None:
                layout.link(f"out:{output}", f"loss:{name}")
        elif name in minimized:
            layout.add(f"loss:{name}", "loss", [name, "objective" if entry.reads == "models" else "criterion"],
                       last + 1)
            for wire in outputs:
                layout.link(f"out:{wire}", f"loss:{name}")
    for name, item in owners.items():
        lr = item.params.get("lr")
        layout.add(f"opt:{name}", "optimizer", [name, item.name + (f" lr {lr}" if lr is not None else "")], last + 2)
        if f"loss:{item.loss}" in layout.boxes:
            layout.link(f"loss:{item.loss}", f"opt:{name}")


def arrow_label(layout, source, wire):
    box = layout.boxes.get(source)
    name = wire if wire and (box is None or box.lines[0] != wire) else ""
    width = layout.widths.get(wire, "")
    if name and width:
        return f"{name} [{width}]"
    return name or (f"[{width}]" if width else "")


def along_arc(start, end, rad, t=0.62):
    middle = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
    dx, dy = end[0] - start[0], end[1] - start[1]
    control = (middle[0] + rad * dy, middle[1] - rad * dx)
    x = (1 - t) ** 2 * start[0] + 2 * (1 - t) * t * control[0] + t ** 2 * end[0]
    y = (1 - t) ** 2 * start[1] + 2 * (1 - t) * t * control[1] + t ** 2 * end[1]
    return x, y


def port_positions(count, low, high):
    if count <= 1:
        return [(low + high) / 2]
    return [low + (high - low) * (position + 1) / (count + 1) for position in range(count)]


def draw_layout(figures, layout, label):
    from matplotlib.patches import FancyBboxPatch

    columns = layout.ordered()
    span = max(columns) + 1
    widths = {column: min(3.6, max(1.5, 0.068 * max(len(line) for box in boxes for line in box.lines) + 0.5))
              for column, boxes in columns.items()}
    heights = {box.name: 0.2 * len(box.lines) + 0.34 for boxes in columns.values() for box in boxes}
    gap_x, gap_y = 1.35, 0.42
    lefts, cursor = {}, 0.45
    for column in range(span):
        lefts[column] = cursor
        cursor += widths.get(column, 1.5) + gap_x
    total_w = cursor - gap_x + 0.45
    tallest = max(sum(heights[box.name] for box in boxes) + gap_y * (len(boxes) - 1) for boxes in columns.values())
    total_h = tallest + 1.3
    drawing, axes = figures.sized(total_w, total_h)
    drawing.subplots_adjust(left=0, right=1, bottom=0, top=1)
    axis = axes[0][0]
    axis.set_xlim(0, total_w)
    axis.set_ylim(0, total_h)
    axis.axis("off")
    palette = figures.categorical
    colors = {"input": figures.ink_muted, "torch": palette[0], "lego": palette[2], "model": palette[6],
              "output": palette[3], "loss": palette[7], "optimizer": palette[1]}
    middle = (total_h - 0.7) / 2 + 0.35
    frames = {}
    for column, boxes in columns.items():
        width = widths[column]
        stack = sum(heights[box.name] for box in boxes) + gap_y * (len(boxes) - 1)
        top = middle + stack / 2
        for box in boxes:
            x, height = lefts[column], heights[box.name]
            y = top - height
            top = y - gap_y
            axis.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.06,rounding_size=0.12",
                                          facecolor=colors[box.kind], edgecolor=colors[box.kind], alpha=0.9,
                                          linewidth=1.0))
            axis.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.06,rounding_size=0.12",
                                          facecolor=figures.surface, edgecolor="none", alpha=0.78))
            axis.text(x + width / 2, y + height / 2, "\n".join(box.lines), ha="center", va="center", fontsize=7.5,
                      color=figures.ink, linespacing=1.35)
            frames[box.name] = (x, y, width, height, box.column)
    leaving = {name: [] for name in frames}
    arriving = {name: [] for name in frames}
    for position, (source, target, wire) in enumerate(layout.arrows):
        if source in frames and target in frames:
            leaving[source].append(position)
            arriving[target].append(position)
    for position, (source, target, wire) in enumerate(layout.arrows):
        if source not in frames or target not in frames:
            continue
        x1, y1, w1, h1, column1 = frames[source]
        x2, y2, w2, h2, column2 = frames[target]
        outs = port_positions(len(leaving[source]), y1 + h1 * 0.25, y1 + h1 * 0.75)
        ins = port_positions(len(arriving[target]), y2 + h2 * 0.25, y2 + h2 * 0.75)
        start = (x1 + w1 + 0.03, outs[::-1][leaving[source].index(position)])
        end = (x2 - 0.03, ins[::-1][arriving[target].index(position)])
        skips = column2 - column1 > 1
        rad = (0.24 if end[1] >= start[1] else -0.24) if skips else 0.0
        axis.annotate("", xy=end, xytext=start,
                      arrowprops={"arrowstyle": "-|>", "color": figures.ink_secondary, "lw": 0.9, "shrinkA": 0,
                                  "shrinkB": 0, "connectionstyle": f"arc3,rad={rad}", "alpha": 0.9})
        text = arrow_label(layout, source, wire)
        if text:
            x, y = along_arc(start, end, rad)
            axis.text(x, y + 0.02, text, ha="center", va="bottom", fontsize=6.5, color=figures.ink_secondary,
                      bbox={"boxstyle": "round,pad=0.15", "facecolor": figures.surface, "edgecolor": "none",
                            "alpha": 0.9})
    kinds = [kind for kind in colors if any(box.kind == kind for box in layout.boxes.values())]
    names = {"input": "input wire", "torch": "torch layer", "lego": "kalfa layer", "model": "model",
             "output": "output wire", "loss": "loss", "optimizer": "optimizer"}
    for position, kind in enumerate(kinds):
        x = 0.45 + position * 1.35
        axis.add_patch(FancyBboxPatch((x, 0.22), 0.28, 0.16, boxstyle="round,pad=0.02", facecolor=colors[kind],
                                      edgecolor="none", alpha=0.85))
        axis.text(x + 0.36, 0.3, names[kind], ha="left", va="center", fontsize=6.5, color=figures.ink_secondary)
    figures.title(drawing, label)
    return drawing


@lego("/plot/kalfa/architecture", partial=True, alias="architecture",
      description="kalfa's own drawing of every report model under plots/<name>_<model>.png: one box per graph "
                  "node with the name from the config, what it is (a torch layer, a lego, another model), the "
                  "shapes one batch traced through it and the layers inside it, the wires as arrows with their "
                  "width, the input wire with its feature columns, the boundary wires as boxes, and the losses "
                  "and the optimizers beside the outputs they read; matplotlib only, any device")
def architecture(predictions, history, models, record, loaders=None, device=None, predicts=None, losses=None,
                 losses_keys=None, optimizers=None, prep=None, name=None, figures=None):
    figures = figures or Figure()
    loader = report_loader(loaders)[1]
    batch = next(iter(loader), None) if loader is not None else None
    stem = name or "architecture"
    features = list(prep.features) if prep is not None else None
    for label, model in (models or {}).items():
        if label.endswith(".ema"):
            continue
        where = device.torch if device is not None else next(iter(model.parameters()), torch.zeros(1)).device
        shapes = traced_shapes(model, batch, where) if batch is not None else {}
        layout, last = graph_layout(model, label, shapes, features)
        training_layout(layout, model, label, last, predicts, losses, losses_keys, optimizers)
        figures.save(draw_layout(figures, layout, label), record, f"{stem}_{label}", tight=False)
        logger.debug(f"architecture: drew {label}")
    return None


@lego("/plot/kalfa/architecture_text", partial=True, alias="architecture_text",
      description="The report models printed as text under plots/<name>.txt, the module repr of each")
def architecture_text(predictions, history, models, record, name=None, figures=None):
    figures = figures or Figure()
    lines = []
    for label, model in (models or {}).items():
        lines.append(f"== {label}")
        lines.append(repr(model))
        lines.append("")
    figures.target(record, f"{name or 'architecture_text'}.txt").write_text("\n".join(lines))
    return None
