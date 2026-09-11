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


@dataclass
class Layout:
    boxes: dict = field(default_factory=dict)
    arrows: list = field(default_factory=list)

    def add(self, key, kind, lines, column):
        self.boxes[key] = Box(key, kind, list(lines), column)

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


def graph_layout(model, label, shapes):
    layout = Layout()
    graph = graph_of(model)
    if graph is None:
        layout.add(label, "lego", [label, type(model).__name__, *shape_lines(shapes.get(None))], 0)
        return layout, 0
    depth = {wire: 0 for wire in graph.inputs}
    producer = {}
    for wire in graph.inputs:
        layout.add(f"in:{wire}", "input", [wire], 0)
        producer[wire] = f"in:{wire}"
    for node in graph.nodes:
        column = 1 + max((depth.get(wire, 0) for wire in node.inputs), default=0)
        module = model.node_module(node)
        what = f"model {node.ref}" if node.ref is not None else type(module).__name__
        layout.add(node.name, kind_of(node, module), [node.name, what, *shape_lines(shapes.get(node.name))], column)
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


def labelled(layout, source, wire):
    if not wire:
        return False
    box = layout.boxes.get(source)
    return box is None or box.lines[0] != wire


def draw_layout(figures, layout, label):
    from matplotlib.patches import FancyBboxPatch

    columns = layout.ordered()
    span = max(columns) + 1
    widths = {column: min(3.4, max(1.5, 0.068 * max(len(line) for box in boxes for line in box.lines) + 0.5))
              for column, boxes in columns.items()}
    heights = {column: max(0.24 * len(box.lines) + 0.34 for box in boxes) for column, boxes in columns.items()}
    height = max(heights.values())
    pitch_y = height + 0.5
    gap_x = 1.25
    lefts, cursor = {}, 0.45
    for column in range(span):
        lefts[column] = cursor
        cursor += widths.get(column, 1.5) + gap_x
    total_w = cursor - gap_x + 0.45
    rows = max(len(boxes) for boxes in columns.values())
    total_h = rows * pitch_y + 1.2
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
    ports = {}
    for column, boxes in columns.items():
        width = widths[column]
        for box in boxes:
            x = lefts[column]
            y = middle + ((len(boxes) - 1) / 2 - box.row) * pitch_y - height / 2
            axis.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.06,rounding_size=0.12",
                                          facecolor=colors[box.kind], edgecolor=colors[box.kind], alpha=0.9,
                                          linewidth=1.0))
            axis.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.06,rounding_size=0.12",
                                          facecolor=figures.surface, edgecolor="none", alpha=0.78))
            axis.text(x + width / 2, y + height / 2, "\n".join(box.lines), ha="center", va="center", fontsize=7.5,
                      color=figures.ink, linespacing=1.35)
            ports[box.name] = ((x, y + height / 2), (x + width, y + height / 2), box.column)
    for source, target, wire in layout.arrows:
        if source not in ports or target not in ports:
            continue
        (_, (x1, y1), column1), ((x2, y2), _, column2) = ports[source], ports[target]
        skips = column2 - column1 > 1
        rad = (0.16 if y2 >= y1 else -0.16) if skips else 0.0
        axis.annotate("", xy=(x2 - 0.03, y2), xytext=(x1 + 0.03, y1),
                      arrowprops={"arrowstyle": "-|>", "color": figures.ink_secondary, "lw": 0.9, "shrinkA": 0,
                                  "shrinkB": 0, "connectionstyle": f"arc3,rad={rad}", "alpha": 0.9})
        if labelled(layout, source, wire):
            along = 0.64
            axis.text(x1 + (x2 - x1) * along, y1 + (y2 - y1) * along + 0.02, wire, ha="center", va="bottom",
                      fontsize=6.5, color=figures.ink_secondary,
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
                  "node with the name from the config, what it is (a torch layer, a lego, another model) and the "
                  "shapes one batch traced through it, the wires as arrows labelled only where the wire is not "
                  "the node's name, the boundary wires as boxes, and the losses and the optimizers beside the "
                  "outputs they read; matplotlib only, any device")
def architecture(predictions, history, models, record, loaders=None, device=None, predicts=None, losses=None,
                 losses_keys=None, optimizers=None, name=None, figures=None):
    figures = figures or Figure()
    loader = report_loader(loaders)[1]
    batch = next(iter(loader), None) if loader is not None else None
    stem = name or "architecture"
    for label, model in (models or {}).items():
        if label.endswith(".ema"):
            continue
        where = device.torch if device is not None else next(iter(model.parameters()), torch.zeros(1)).device
        shapes = traced_shapes(model, batch, where) if batch is not None else {}
        layout, last = graph_layout(model, label, shapes)
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
