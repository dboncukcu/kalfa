import inspect
import weakref

import torch
from torch.nn.parameter import is_lazy
from torch.overrides import TorchFunctionMode, resolve_name


PASS = {"view", "view_as", "reshape", "reshape_as", "permute", "transpose", "t", "T", "mT", "H", "mH", "flatten",
        "unflatten", "squeeze", "unsqueeze", "contiguous", "to", "float", "double", "half", "bfloat16", "int", "long",
        "short", "bool", "byte", "char", "type", "type_as", "cpu", "cuda", "detach", "clone", "getitem", "expand",
        "expand_as", "broadcast_to", "narrow", "select", "movedim", "moveaxis", "swapaxes", "swapdims", "split",
        "chunk", "unbind", "tensor_split", "as_strided", "data", "requires_grad", "atleast_1d", "atleast_2d",
        "atleast_3d", "real", "align_to", "rename", "refine_names", "pin_memory", "share_memory"}
FRESH = {"zeros_like", "ones_like", "empty_like", "full_like", "rand_like", "randn_like", "randint_like", "new_zeros",
         "new_ones", "new_empty", "new_full", "new_tensor", "setitem"}
BINARY = {"add", "sub", "mul", "matmul", "div", "truediv", "floordiv", "mod", "pow", "and", "or", "xor", "lshift",
          "rshift"}
RENAMED = {"truediv": "div", "floordiv": "floor_divide"}


def resolved(func):
    try:
        name = resolve_name(func)
    except Exception:
        name = None
    return name or getattr(func, "__name__", None) or ""


def plain(func):
    parts = resolved(func).split(".")
    if parts[-1] in ("__get__", "__set__", "__delete__"):
        if parts[-1] != "__get__" or len(parts) < 2:
            return None
        parts = parts[:-1]
    word = parts[-1]
    if len(word) > 4 and word.startswith("__") and word.endswith("__"):
        word = word[2:-2]
        if word[:1] in ("r", "i") and word[1:] in BINARY:
            word = word[1:]
    elif word.endswith("_") and not word.startswith("_"):
        word = word.rstrip("_")
    return RENAMED.get(word, word) or None


def short(func):
    name = resolved(func)
    return name[len("torch."):] if name.startswith("torch.") else name


def tensors_in(value):
    found = []
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, torch.Tensor):
            found.append(item)
        elif isinstance(item, dict):
            stack.extend(reversed(list(item.values())))
        elif isinstance(item, (tuple, list)):
            stack.extend(reversed(item))
    return found


def qualified(module):
    return f"{type(module).__module__}.{type(module).__qualname__}"


def dtype_text(value):
    if isinstance(value, torch.Tensor):
        return str(value.dtype).replace("torch.", "")
    return type(value).__name__


def named(result, names):
    if isinstance(result, torch.Tensor):
        return [(names[0] if names else "output", result)]
    if isinstance(result, dict):
        return [(str(key), value) for key, value in result.items()]
    if isinstance(result, (tuple, list)):
        return [(names[position] if position < len(names) else f"output_{position}", value)
                for position, value in enumerate(result)]
    return []


class Tracer:
    def __init__(self, module, limit):
        self.root = module
        self.limit = int(limit)
        self.paths = {id(found): name for name, found in module.named_modules()}
        self.sizes = {}
        self.nodes = []
        self.edges = []
        self.links = set()
        self.groups = []
        self.frames = []
        self.producer = {}
        self.counted = 0
        self.truncated = False
        self.batch = None
        self.depth = 0
        self.busy = False

    def shape(self, value):
        if not isinstance(value, torch.Tensor):
            return None
        shape = [int(size) for size in value.shape]
        if shape and self.batch is not None and shape[0] == self.batch:
            return shape[1:]
        return shape

    def first_shape(self, value):
        found = tensors_in(value)
        return self.shape(found[0]) if found else None

    def parameters(self, module):
        key = id(module)
        if key not in self.sizes:
            self.sizes[key] = sum(parameter.numel() for parameter in module.parameters() if not is_lazy(parameter))
        return self.sizes[key]

    def mark(self, value, node):
        for tensor in tensors_in(value):
            self.producer[id(tensor)] = (node, weakref.ref(tensor))

    def sources(self, value):
        found = []
        for tensor in tensors_in(value):
            known = self.producer.get(id(tensor))
            if known is not None and known[1]() is tensor:
                found.append((known[0], tensor))
        return found

    def room(self):
        if self.counted >= self.limit:
            self.truncated = True
            return False
        self.counted += 1
        return True

    def add(self, kind, label, detail, path=None, cls=None, shape=None, parameters=None, group=None):
        node = {"id": f"n{len(self.nodes)}", "kind": kind, "label": label, "detail": detail, "path": path,
                "class": cls, "shape": shape, "parameters": parameters, "group": group}
        self.nodes.append(node)
        return node["id"]

    def link(self, found, target):
        for origin, tensor in found:
            if origin == target or (origin, target) in self.links:
                continue
            self.links.add((origin, target))
            self.edges.append({"from": origin, "to": target, "shape": self.shape(tensor)})

    def group_at(self, position):
        frame = self.frames[position]
        if frame["group"] is None:
            module = frame["module"]
            parent = self.group_below(position)
            frame["group"] = f"g{len(self.groups)}"
            self.groups.append({"id": frame["group"], "path": frame["path"], "class": qualified(module),
                                "label": frame["path"], "detail": type(module).__name__,
                                "parameters": self.parameters(module), "parent": parent})
        return frame["group"]

    def group_below(self, position):
        for back in range(position - 1, -1, -1):
            if self.frames[back]["kind"] == "group":
                return self.group_at(back)
        return None

    def start(self, inputs, names, keywords=None):
        keywords = dict(keywords or {})
        found = tensors_in([inputs, keywords])
        if found and found[0].ndim:
            self.batch = int(found[0].shape[0])
        given = [(names[position] if position < len(names) else f"input_{position}", value)
                 for position, value in enumerate(inputs)]
        for name, value in [*given, *((str(key), value) for key, value in keywords.items())]:
            if not tensors_in(value):
                continue
            node = self.add("input", name, dtype_text(value), shape=self.shape(value))
            self.mark(value, node)

    def frame(self, module, args, kwargs):
        if self.depth:
            self.depth += 1
            return {"module": module, "kind": "inside"}
        path = self.paths.get(id(module))
        if next(module.children(), None) is None:
            self.depth += 1
            return {"module": module, "kind": "leaf", "path": path, "sources": self.sources((args, kwargs))}
        if module is self.root or path is None:
            return {"module": module, "kind": "root"}
        return {"module": module, "kind": "group", "path": path, "group": None}

    def enter(self, module, args, kwargs):
        self.busy = True
        try:
            self.frames.append(self.frame(module, args, kwargs))
        finally:
            self.busy = False

    def leave(self, module, args, kwargs, output):
        self.busy = True
        try:
            frame = None
            while self.frames:
                frame = self.frames.pop()
                if frame["kind"] in ("inside", "leaf"):
                    self.depth -= 1
                if frame["module"] is module:
                    break
            if frame is not None and frame["kind"] == "leaf" and frame["module"] is module:
                self.leaf(module, frame, output)
        finally:
            self.busy = False

    def leaf(self, module, frame, output):
        if not self.room():
            return
        path = frame["path"]
        node = self.add("module", type(module).__name__, path or "", path=path, cls=qualified(module),
                        shape=self.first_shape(output), parameters=self.parameters(module),
                        group=self.group_below(len(self.frames)))
        self.link(frame["sources"], node)
        self.mark(output, node)

    def op(self, func, args, kwargs, result):
        name = plain(func)
        if not name or name in FRESH:
            return
        found = self.sources((args, kwargs))
        if not found or not tensors_in(result):
            return
        if name in PASS:
            self.mark(result, found[0][0])
            return
        if not self.room():
            return
        node = self.add("op", name, short(func), shape=self.first_shape(result),
                        group=self.group_below(len(self.frames)))
        self.link(found, node)
        self.mark(result, node)

    def finish(self, result, names):
        for name, value in named(result, names):
            if not tensors_in(value):
                continue
            node = self.add("output", name, dtype_text(value), shape=self.shape(value))
            self.link(self.sources(value), node)

    def graph(self):
        return {"method": "eager", "truncated": self.truncated, "nodes": self.nodes, "edges": self.edges,
                "groups": self.groups}

    def release(self):
        self.producer = {}
        self.frames = []


class Watch(TorchFunctionMode):
    def __init__(self, tracer):
        super().__init__()
        self.tracer = tracer

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        result = func(*args, **kwargs)
        tracer = self.tracer
        if not tracer.busy and not tracer.depth:
            tracer.op(func, args, kwargs, result)
        return result


def trace(module, inputs, input_names, output_names, limit=300, keywords=None):
    tracer = Tracer(module, limit)
    handles = []
    try:
        for found in module.modules():
            handles.append(found.register_forward_pre_hook(tracer.enter, with_kwargs=True))
            handles.append(found.register_forward_hook(tracer.leave, with_kwargs=True))
        inputs = tuple(inputs)
        keywords = dict(keywords or {})
        tracer.start(inputs, list(input_names or []), keywords)
        with torch.no_grad(), Watch(tracer):
            result = module(*inputs, **keywords)
        tracer.finish(result, list(output_names or []))
        return tracer.graph()
    except Exception:
        return None
    finally:
        for handle in handles:
            handle.remove()
        tracer.release()


def argument_names(module, count):
    try:
        signature = inspect.signature(module.forward)
    except (TypeError, ValueError):
        return [f"input_{position}" for position in range(count)]
    kinds = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    names = [name for name, parameter in signature.parameters.items() if parameter.kind in kinds]
    return [names[position] if position < len(names) else f"input_{position}" for position in range(count)]
