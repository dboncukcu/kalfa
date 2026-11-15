import math
from pathlib import Path

import numpy
import torch
from PIL import Image

from kalfa.core.record import atomic


def image_grid(images, columns=8, pad=2):
    images = torch.nan_to_num(torch.as_tensor(images).detach().float().cpu())
    if images.ndim == 3:
        images = images.unsqueeze(1)
    count, channels, height, width = images.shape
    columns = max(1, min(columns, count))
    rows = math.ceil(count / columns)
    low, high = float(images.min()), float(images.max())
    if high > low:
        images = (images - low) / (high - low)
    else:
        images = images.clamp(0.0, 1.0)
    canvas = torch.ones((channels, rows * (height + pad) + pad, columns * (width + pad) + pad))
    for index in range(count):
        row, column = divmod(index, columns)
        top = pad + row * (height + pad)
        left = pad + column * (width + pad)
        canvas[:, top:top + height, left:left + width] = images[index]
    if channels == 1:
        canvas = canvas.expand(3, -1, -1)
    return canvas.permute(1, 2, 0).numpy()


def write_png(grid, path, least=256, most=8):
    pixels = numpy.clip(numpy.rint(numpy.asarray(grid) * 255.0), 0, 255).astype(numpy.uint8)
    height, width = pixels.shape[:2]
    scale = max(1, min(most, math.ceil(least / max(height, width, 1))))
    image = Image.fromarray(numpy.ascontiguousarray(pixels))
    if scale > 1:
        image = image.resize((width * scale, height * scale), Image.Resampling.NEAREST)
    with atomic(path) as temporary:
        image.save(temporary, format="PNG")
    return Path(path)


def is_text(samples):
    return isinstance(samples, (list, tuple)) and bool(samples) and all(isinstance(item, str) for item in samples)


def is_images(samples):
    return isinstance(samples, torch.Tensor) and samples.ndim == 4 and samples.shape[1] in (1, 3)


def save_samples(samples, directory, stem, figures=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if isinstance(samples, str):
        samples = [samples]
    if isinstance(samples, torch.Tensor):
        samples = samples.detach().cpu()
    path = directory / f"{stem}.pt"
    with atomic(path) as temporary:
        torch.save(samples, temporary)
    written = [path]
    if is_text(samples):
        text = directory / f"{stem}.txt"
        with atomic(text) as temporary:
            temporary.write_text("\n\n".join(samples) + "\n", encoding="utf-8")
        written.append(text)
    elif is_images(samples):
        written.append(write_png(image_grid(samples), directory / f"{stem}.png"))
    return written
