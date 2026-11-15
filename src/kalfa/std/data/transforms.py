import inspect

import numpy
import torch
import torch.nn.functional as functional
from PIL import Image

from kalfa.core import aliases
from kalfa.core.aliases import register
from kalfa.core.errors import BuildError, RunError
from kalfa.std.data.columns import target_of


PRESETS = {
    "imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "cifar10": ([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
    "mnist": ([0.1307], [0.3081]),
}

GRAY_WEIGHTS = (0.299, 0.587, 0.114)
PIL_MODES = ("1", "L", "LA", "RGB", "RGBA", "I", "I;16", "F")


def built(target, params, where):
    try:
        made = target(**params)
    except BuildError:
        raise
    except Exception as failure:
        raise BuildError(f"{where}: {aliases.target_path(target)} could not be built: {type(failure).__name__}: "
                         f"{failure}") from failure
    if not callable(made):
        raise BuildError(f"{where}: {aliases.target_path(target)} gave a {type(made).__name__}, which is not a "
                         f"callable image -> image")
    return made


def transform_of(spec, where):
    if isinstance(spec, str):
        target = target_of(spec, where)
        if inspect.isclass(target):
            return built(target, {}, where)
        if callable(target):
            return target
        raise BuildError(f"{where}: {spec!r} is a {type(target).__name__}, not a transform")
    if isinstance(spec, dict):
        if "use" not in spec:
            raise BuildError(f"{where}: a transform call needs use, got the keys {sorted(spec)}")
        params = {key: value for key, value in spec.items() if key != "use"}
        return built(target_of(spec["use"], where), params, where)
    if inspect.isclass(spec):
        return built(spec, {}, where)
    if callable(spec):
        return spec
    raise BuildError(f"{where}: a transform is a short name, an import path, a call or a callable object, got "
                     f"{type(spec).__name__}")


def name_of(spec):
    if isinstance(spec, str):
        return spec
    if isinstance(spec, dict):
        return str(spec.get("use"))
    path = aliases.target_path(spec if inspect.isclass(spec) or inspect.isroutine(spec) else type(spec))
    for name, entry in aliases.ALIASES.items():
        if entry.slot == "value" and aliases.target_path(entry.target) == path:
            return name
    return path


class Pipeline:
    def __init__(self, steps=(), names=()):
        self.steps = list(steps)
        self.names = list(names)

    def __call__(self, image):
        for step in self.steps:
            image = step(image)
        return image


def pipeline_of(specs, where):
    if specs is None:
        return Pipeline()
    if isinstance(specs, Pipeline):
        return specs
    items = list(specs) if isinstance(specs, (list, tuple)) else [specs]
    steps = [transform_of(item, f"{where}[{position}]") for position, item in enumerate(items)]
    return Pipeline(steps, [name_of(item) for item in items])


def pil_array(image):
    if image.mode not in PIL_MODES:
        image = image.convert("RGB")
    return numpy.array(image)


def array_tensor(array):
    if array.dtype == numpy.uint8:
        values = array.astype(numpy.float32) / 255.0
    elif array.dtype == numpy.uint16:
        values = array.astype(numpy.float32) / 65535.0
    else:
        values = array.astype(numpy.float32)
    if values.ndim == 2:
        values = values[None]
    elif values.ndim == 3:
        values = values.transpose(2, 0, 1)
    else:
        raise RunError(f"to_tensor takes an array of shape (H, W) or (H, W, C), got {array.shape}")
    return torch.from_numpy(numpy.ascontiguousarray(values, dtype=numpy.float32))


class ToTensor:
    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            return image
        if isinstance(image, Image.Image):
            return array_tensor(pil_array(image))
        if isinstance(image, numpy.ndarray):
            return array_tensor(image)
        raise RunError(f"to_tensor takes a PIL image, an array or a tensor, got {type(image).__name__}")


def image_tensor(value):
    tensor = value if isinstance(value, torch.Tensor) else ToTensor()(value)
    if tensor.dtype == torch.uint8:
        tensor = tensor.to(torch.float32) / 255.0
    elif tensor.dtype != torch.float32:
        tensor = tensor.to(torch.float32)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim < 3:
        raise RunError(f"an image tensor is (C, H, W), got the shape {tuple(tensor.shape)}")
    return tensor


def preset_of(name):
    key = str(name).lower()
    if key not in PRESETS:
        raise BuildError(f"normalize: unknown preset {name!r}; the presets are {sorted(PRESETS)}")
    return PRESETS[key]


def values_of(value, what):
    if isinstance(value, torch.Tensor):
        value = value.flatten().tolist()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [float(value)]
    if (isinstance(value, (list, tuple)) and value
            and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value)):
        return [float(item) for item in value]
    raise BuildError(f"normalize: {what} is a number, a list of numbers or a preset name ({', '.join(PRESETS)}), "
                     f"got {value!r}")


def per_channel(values, channels, what):
    if len(values) == channels:
        return values
    if len(values) == 1:
        return values * channels
    raise RunError(f"normalize: {len(values)} {what} values for an image of {channels} channels")


class Normalize:
    def __init__(self, mean=0.5, std=None):
        if isinstance(mean, str):
            mean, preset_std = preset_of(mean)
            if std is None:
                std = preset_std
        if std is None:
            std = 0.5
        if isinstance(std, str):
            std = preset_of(std)[1]
        self.mean = values_of(mean, "mean")
        self.std = values_of(std, "std")
        if any(value <= 0.0 for value in self.std):
            raise BuildError(f"normalize: std must be positive, got {self.std}")
        if len(self.mean) != len(self.std) and 1 not in (len(self.mean), len(self.std)):
            raise BuildError(f"normalize: mean has {len(self.mean)} values and std {len(self.std)}")

    def __call__(self, image):
        floating = isinstance(image, torch.Tensor) and image.is_floating_point() and image.ndim >= 3
        tensor = image if floating else image_tensor(image)
        channels = int(tensor.shape[-3])
        shape = (channels, 1, 1)
        mean = torch.tensor(per_channel(self.mean, channels, "mean"), dtype=tensor.dtype, device=tensor.device)
        std = torch.tensor(per_channel(self.std, channels, "std"), dtype=tensor.dtype, device=tensor.device)
        return (tensor - mean.view(shape)) / std.view(shape)


class TwoViews:
    def __init__(self, each):
        self.each = pipeline_of(each, "two_views.each")

    def __call__(self, image):
        return torch.stack([image_tensor(self.each(image)), image_tensor(self.each(image))])


def size_of(size, what):
    if isinstance(size, int) and not isinstance(size, bool) and size > 0:
        return size, size
    if (isinstance(size, (list, tuple)) and len(size) == 2
            and all(isinstance(item, int) and not isinstance(item, bool) and item > 0 for item in size)):
        return int(size[0]), int(size[1])
    raise BuildError(f"{what}: size is a positive integer or [height, width], got {size!r}")


def extent(image):
    if isinstance(image, Image.Image):
        return image.height, image.width
    return int(image.shape[-2]), int(image.shape[-1])


def crop(image, top, left, height, width):
    if isinstance(image, Image.Image):
        return image.crop((left, top, left + width, top + height))
    out = image.new_zeros((*image.shape[:-2], height, width))
    full_height, full_width = extent(image)
    first_row, first_column = max(top, 0), max(left, 0)
    last_row, last_column = min(top + height, full_height), min(left + width, full_width)
    if last_row > first_row and last_column > first_column:
        piece = image[..., first_row:last_row, first_column:last_column]
        out[..., first_row - top:last_row - top, first_column - left:last_column - left] = piece
    return out


def planar(image):
    return image if isinstance(image, Image.Image) else image_tensor(image)


class Resize:
    def __init__(self, size):
        self.size = size_of(size, "resize")

    def __call__(self, image):
        height, width = self.size
        image = planar(image)
        if isinstance(image, Image.Image):
            return image.resize((width, height), Image.Resampling.BILINEAR)
        batched = image if image.ndim == 4 else image.unsqueeze(0)
        out = functional.interpolate(batched, size=(height, width), mode="bilinear", align_corners=False,
                                     antialias=True)
        return out if image.ndim == 4 else out.squeeze(0)


class CenterCrop:
    def __init__(self, size):
        self.size = size_of(size, "center_crop")

    def __call__(self, image):
        height, width = self.size
        image = planar(image)
        full_height, full_width = extent(image)
        top = int(round((full_height - height) / 2.0))
        left = int(round((full_width - width) / 2.0))
        return crop(image, top, left, height, width)


def offset(span):
    if span <= 0:
        return int(round(span / 2.0))
    return int(torch.randint(0, span + 1, (1,)).item())


class RandomCrop:
    def __init__(self, size, padding=0):
        self.size = size_of(size, "random_crop")
        if isinstance(padding, bool) or not isinstance(padding, int) or padding < 0:
            raise BuildError(f"random_crop: padding is a non negative integer, got {padding!r}")
        self.padding = padding

    def __call__(self, image):
        height, width = self.size
        image = planar(image)
        full_height, full_width = extent(image)
        top = offset(full_height + 2 * self.padding - height) - self.padding
        left = offset(full_width + 2 * self.padding - width) - self.padding
        return crop(image, top, left, height, width)


class RandomFlip:
    def __init__(self, p=0.5):
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not 0.0 <= float(p) <= 1.0:
            raise BuildError(f"random_flip: p is a probability in [0, 1], got {p!r}")
        self.p = float(p)

    def __call__(self, image):
        image = planar(image)
        if float(torch.rand(())) >= self.p:
            return image
        if isinstance(image, Image.Image):
            return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return torch.flip(image, dims=(-1,))


class Grayscale:
    def __call__(self, image):
        image = planar(image)
        if isinstance(image, Image.Image):
            return image if image.mode == "L" else image.convert("L")
        channels = int(image.shape[-3])
        if channels == 1:
            return image
        if channels in (3, 4):
            weights = image.new_tensor(GRAY_WEIGHTS).view(3, 1, 1)
            return (image[..., :3, :, :] * weights).sum(dim=-3, keepdim=True)
        raise RunError(f"grayscale takes an image of 1, 3 or 4 channels, got {channels}")


REGISTRATIONS = (
    ("to_tensor", "kalfa.std.data.transforms.ToTensor",
     "a PIL image or an array becomes a float tensor (C, H, W) in [0, 1]; a tensor passes through"),
    ("normalize", "kalfa.std.data.transforms.Normalize",
     "(x - mean) / std per channel; mean and std are numbers, lists or a preset name: imagenet, cifar10, mnist"),
    ("two_views", "kalfa.std.data.transforms.TwoViews",
     "applies the each pipeline twice with fresh randomness and stacks the two results into (2, C, H, W)"),
    ("resize", "kalfa.std.data.transforms.Resize",
     "resize a PIL image or a tensor to size (an int or [height, width])"),
    ("center_crop", "kalfa.std.data.transforms.CenterCrop", "crop the center of a PIL image or a tensor to size"),
    ("random_crop", "kalfa.std.data.transforms.RandomCrop", "crop a random window of size after padding"),
    ("random_flip", "kalfa.std.data.transforms.RandomFlip", "flip an image horizontally with probability p"),
    ("grayscale", "kalfa.std.data.transforms.Grayscale", "an image in one channel"),
)

for name, target, doc in REGISTRATIONS:
    register(name, target, slot="value", doc=doc)
