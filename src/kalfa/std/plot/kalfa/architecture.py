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


def draw_layout(figures, layout, label):
    from matplotlib.patches import FancyBboxPatch

    columns = layout.columns()
    span = max(columns) + 1
    rows = max(len(items) for items in columns.values())
    width, height, gap_x, gap_y = 2.3, 1.0, 3.3, 1.5
    drawing, axes = figures.sized(1.9 * span + 0.6, 0.95 * rows + 1.1)
    axis = axes[0][0]
    palette = figures.categorical
    colors = {"input": figures.ink_muted, "torch": palette[0], "lego": palette[2], "model": palette[6],
              "output": palette[3], "loss": palette[7], "optimizer": palette[1]}
    centers = {}
    for box in layout.boxes.values():
        count = len(columns[box.column])
        x = box.column * gap_x
        y = -(box.row - (count - 1) / 2) * gap_y
        centers[box.name] = (x, y)
        axis.add_patch(FancyBboxPatch((x - width / 2, y - height / 2), width, height, boxstyle="round,pad=0.08",
                                      facecolor=colors[box.kind], edgecolor=colors[box.kind], alpha=0.85,
                                      linewidth=1.0))
        axis.add_patch(FancyBboxPatch((x - width / 2, y - height / 2), width, height, boxstyle="round,pad=0.08",
                                      facecolor=figures.surface, edgecolor="none", alpha=0.72))
        axis.text(x, y, "\n".join(box.lines), ha="center", va="center", fontsize=7, color=figures.ink)
    for source, target, wire in layout.arrows:
        if source not in centers or target not in centers:
            continue
        (x1, y1), (x2, y2) = centers[source], centers[target]
        axis.annotate("", xy=(x2 - width / 2, y2), xytext=(x1 + width / 2, y1),
                      arrowprops={"arrowstyle": "-|>", "color": figures.ink_secondary, "lw": 0.9,
                                  "shrinkA": 0, "shrinkB": 0})
        if wire:
            axis.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.1, wire, ha="center", va="bottom", fontsize=6.5,
                      color=figures.ink_secondary)
    axis.set_xlim(-width, (span - 1) * gap_x + width)
    axis.set_ylim(-(rows - 1) / 2 * gap_y - height, (rows - 1) / 2 * gap_y + height)
    axis.axis("off")
    figures.title(drawing, label)
    return drawing


@lego("/plot/kalfa/architecture", partial=True, alias="architecture",
      description="kalfa's own drawing of every report model under plots/<name>_<model>.png: one box per graph "
                  "node with the name from the config, what it is (a torch layer, a lego, another model) and the "
                  "shapes one batch traced through it, the wires as labelled arrows, the boundary wires as boxes, "
                  "and the losses and the optimizers beside the outputs they read; matplotlib only, any device")
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
        figures.save(draw_layout(figures, layout, label), record, f"{stem}_{label}")
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
