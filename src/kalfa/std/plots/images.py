import math

import numpy

from kalfa.core.errors import KalfaError
from kalfa.std.plots.common import directory_of


TEXT_LINES = 20
TEXT_WIDTH = 120


def first_output(outputs):
    import torch

    if isinstance(outputs, torch.Tensor):
        return "output", outputs
    items = outputs.items() if isinstance(outputs, dict) else enumerate(outputs or ())
    for name, value in items:
        if isinstance(value, torch.Tensor):
            return str(name), value
    return None, None


def image_field(ctx, batch):
    import torch

    for name in getattr(getattr(ctx, "data", None), "inputs", None) or ():
        if isinstance(batch.get(name), torch.Tensor):
            return str(name)
    for name, value in batch.items():
        if isinstance(value, torch.Tensor) and value.ndim in (3, 4):
            return str(name)
    return None


def shaped(output, images, name):
    if output.shape[1:] == images.shape[1:]:
        return output
    if output.ndim >= 2 and len(output) == len(images) and output[0].numel() == images[0].numel():
        return output.reshape(images.shape)
    if output.ndim in (3, 4):
        return output
    raise KalfaError(f"image_pairs: the output {name!r} of shape {tuple(output.shape)} cannot be drawn like the "
                     f"inputs of shape {tuple(images.shape)}")


def pixels(images):
    images = images.detach().float().cpu()
    if images.ndim == 4:
        if images.shape[1] not in (1, 3):
            images = images[:, :1]
        images = images.permute(0, 2, 3, 1)
        if images.shape[-1] == 1:
            images = images[..., 0]
    return images.numpy()


def show(axis, image, low, high):
    scaled = numpy.clip((image - low) / (high - low), 0.0, 1.0) if high > low else numpy.zeros_like(image)
    if scaled.ndim == 2:
        axis.imshow(scaled, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    else:
        axis.imshow(scaled, interpolation="nearest")
    axis.set_xticks([])
    axis.set_yticks([])


def image_pairs(record, figures, n=8, set="test", ctx=None):
    import torch

    if ctx is None:
        raise KalfaError("image_pairs needs the run; redraw it with the record open")
    loaders = ctx.loaders()
    loader = loaders.get(set)
    if loader is None:
        raise KalfaError(f"image_pairs: the data has no {set!r} set; its sets are {sorted(loaders)}")
    modes = getattr(ctx, "set_modes", None)
    if callable(modes):
        modes(False)
    device = getattr(ctx, "device", None)
    count = max(int(n), 1)
    field = name = None
    shown, guessed = [], []
    with torch.no_grad():
        for batch in loader:
            if device is not None:
                batch = device.move(batch)
            if not isinstance(batch, dict):
                raise KalfaError("image_pairs needs batches that map field names to tensors")
            field = field or image_field(ctx, batch)
            if field is None:
                raise KalfaError(f"image_pairs: none of the batch fields {sorted(batch)} holds images")
            images = batch[field]
            if images.ndim not in (3, 4):
                raise KalfaError(f"image_pairs: the input {field!r} of shape {tuple(images.shape)} is not a batch of "
                                 f"images")
            name, output = first_output(ctx.step.predict(batch, ctx))
            if output is None:
                raise KalfaError("image_pairs: the run predicts no tensor; train.predicts names the model to draw")
            shown.append(images.detach().float().cpu())
            guessed.append(shaped(output, images, name).detach().float().cpu())
            if sum(len(piece) for piece in shown) >= count:
                break
    if not shown:
        return None
    inputs = pixels(torch.cat(shown)[:count])
    outputs = pixels(torch.cat(guessed)[:count])
    total = min(len(inputs), len(outputs))
    values = numpy.concatenate([inputs[:total].reshape(-1), outputs[:total].reshape(-1)])
    values = values[numpy.isfinite(values)]
    low, high = (float(values.min()), float(values.max())) if values.size else (0.0, 1.0)
    figure = figures.new(max(1.3 * total + 0.8, 3.2), 3.2)
    axes = figure.subplots(2, total, squeeze=False)
    for row, (images, label) in enumerate(((inputs, field), (outputs, name))):
        for column in range(total):
            show(axes[row][column], images[column], low, high)
        axes[row][0].set_ylabel(label)
    figure.suptitle(f"{field} and {name} ({set}, n = {total})", x=0.02, ha="left")
    figures.save(figure, "image_pairs")
    return None


def sample_images(samples, path):
    import torch

    if isinstance(samples, (list, tuple)) and samples and all(isinstance(item, torch.Tensor) for item in samples):
        try:
            samples = torch.stack([item.detach().cpu() for item in samples])
        except RuntimeError as error:
            raise KalfaError(f"samples_grid: the tensors in {path.name} differ in shape: {error}") from None
    if not isinstance(samples, torch.Tensor) or samples.ndim not in (3, 4) or not len(samples):
        if isinstance(samples, torch.Tensor):
            kind = f"a tensor of shape {tuple(samples.shape)}"
        else:
            kind = f"a {type(samples).__name__}"
        raise KalfaError(f"samples_grid: {path.name} holds {kind}, not a batch of images (N, C, H, W)")
    images = samples.detach().float().cpu()
    if images.ndim == 4 and images.shape[1] not in (1, 3):
        images = images[:, :1]
    return images


def text_sheet(path, figures, lines=None):
    if lines is None:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    shown = [line.expandtabs(4) for line in lines[:TEXT_LINES]]
    shown = [line if len(line) <= TEXT_WIDTH else line[:TEXT_WIDTH - 3] + "..." for line in shown]
    figure = figures.new(10.0, 0.8 + 0.22 * max(len(shown), 1))
    axis = figure.subplots()
    axis.axis("off")
    axis.text(0.0, 1.0, "\n".join(shown) or "(empty)", family="monospace", fontsize=9, va="top", ha="left",
              transform=axis.transAxes, parse_math=False)
    axis.set_title(f"{path.name} (first {len(shown)} of {len(lines)} lines)", loc="left")
    figures.save(figure, "samples_grid")
    return None


def samples_grid(record, figures, n=8, file="samples"):
    import torch

    from kalfa.std.samples import image_grid, is_text

    directory = directory_of(record) / "samples"
    stem = str(file)
    names = [stem] if stem.endswith((".pt", ".txt")) else [f"{stem}.pt", f"{stem}.txt"]
    path = next((directory / name for name in names if (directory / name).is_file()), None)
    if path is None:
        wanted = " or ".join(f"samples/{name}" for name in names)
        raise KalfaError(f"samples_grid: {wanted} is missing in {directory.parent}; a sampler under generate or a "
                         f"write_samples rule writes it")
    if path.suffix == ".txt":
        return text_sheet(path, figures)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if is_text(loaded):
        return text_sheet(path, figures, "\n\n".join(loaded).splitlines())
    images = sample_images(loaded, path)
    count = min(len(images), max(int(n), 1))
    grid = image_grid(images[:count], columns=math.ceil(math.sqrt(count)))
    height, width = grid.shape[:2]
    scale = 8.0 / max(height, width)
    figure = figures.new(max(width * scale + 0.4, 3.0), height * scale + 0.8)
    axis = figure.subplots()
    axis.imshow(grid, interpolation="nearest")
    axis.axis("off")
    axis.set_title(f"{path.name} ({count} of {len(images)} samples)", loc="left")
    figures.save(figure, "samples_grid")
    return None
