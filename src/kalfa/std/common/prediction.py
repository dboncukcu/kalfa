import numpy
import pandas
import torch

from kalfa.std.common.device import Device
from kalfa.std.common.runtime import call_model, expand_targets, named_outputs


def observed_columns(model, loader, dataset, device):
    raw = {}
    observed = {name: [] for name in dataset.targets}
    with torch.no_grad():
        for batch in loader:
            batch = device.move(batch)
            outputs = named_outputs(model, call_model(model, batch))
            for wire, value in outputs.items():
                raw.setdefault(wire, []).append(value.detach().cpu().numpy().reshape(len(value), -1))
            for name in dataset.targets:
                value = batch[name]
                observed[name].append(value.detach().cpu().numpy().reshape(len(value), -1))
    return raw, observed


def prediction_table(model, loader, prep, dataset, device=None, target_map=None):
    model.eval()
    raw, observed = observed_columns(model, loader, dataset, device or Device.cpu())
    set_name = dataset.frame.set
    columns = {"row": numpy.asarray(dataset.rows())}
    widths = {}
    for name in dataset.targets:
        values = numpy.concatenate(observed[name]) if observed[name] else numpy.zeros((0, 1))
        widths[name] = values.shape[1]
        if values.shape[1] == 1:
            columns[name] = prep.inverse(name, values[:, 0], set_name)
        else:
            for position in range(values.shape[1]):
                columns[f"{name}_{position}"] = prep.inverse(name, values[:, position], set_name)
    total = sum(widths.values())
    single = dataset.targets[0] if len(dataset.targets) == 1 else None
    for wire, pieces in raw.items():
        matrix = numpy.concatenate(pieces) if pieces else numpy.zeros((0, 1))
        width = matrix.shape[1]
        if width == 1:
            columns[f"raw_{wire}"] = matrix[:, 0]
        else:
            for position in range(width):
                columns[f"raw_{wire}_{position}"] = matrix[:, position]
        if target_map:
            names = expand_targets(target_map.get(wire), dataset.targets)
            write_blocks(columns, prep, matrix, names, widths, set_name, [f"pred_{wire}_{name}" for name in names])
            continue
        decoder = prep.decoder(single) if single is not None else None
        if decoder is not None:
            columns[f"pred_{wire}"] = prep.decode(single, matrix, set_name)
        elif width == total and total:
            names = list(dataset.targets)
            write_blocks(columns, prep, matrix, names, widths, set_name,
                         [f"pred_{wire}" if single is not None else f"pred_{wire}_{name}" for name in names])
    return pandas.DataFrame(columns)


def write_blocks(columns, prep, matrix, names, widths, set_name, labels):
    offset = 0
    for name, label in zip(names, labels):
        span = widths.get(name, 1)
        block = matrix[:, offset:offset + span]
        offset += span
        if block.shape[1] == 0:
            continue
        if prep.decoder(name) is not None:
            columns[label] = prep.decode(name, block, set_name)
        elif span == 1:
            columns[label] = prep.inverse(name, block[:, 0], set_name)
        else:
            for position in range(span):
                columns[f"{label}_{position}"] = prep.inverse(name, block[:, position], set_name)
