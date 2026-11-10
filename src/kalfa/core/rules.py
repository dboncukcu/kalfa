import logging
from dataclasses import dataclass, field

from .context import TrainSettings
from .errors import RunError, error
from .refs import SINGULAR, TABLES, jsonable, join_tokens, ref_json, ref_parts, tokenize
from .spec import is_call
from .spec import kind as type_name


RULE_KEYS = ("when", "after", "once", "set", "stop", "run")
ROOTS = TABLES + ("train", "rules")
INNER = ("module", "metric", "function")
OPERATIONS = ("times", "plus")


@dataclass
class RuleSpec:
    name: str
    when: object = None
    after: str | None = None
    once: bool = True
    set: dict = field(default_factory=dict)
    stop: bool = False
    run: object = None


@dataclass
class Patch:
    path: str
    value: object
    asked: object
    by: str
    turn: int


def rules_by_name(ctx):
    table = getattr(ctx, "rules_by_name", None)
    if not table:
        table = {rule.name: rule for rule in (getattr(ctx, "rules", None) or [])}
        ctx.rules_by_name = table
    return table


def describe(trigger):
    if trigger is None:
        return "rule"
    method = getattr(trigger, "describe", None)
    return method() if callable(method) else type(trigger).__name__


def logger_of(ctx):
    return getattr(ctx, "logger", None) or logging.getLogger("kalfa.run")


class Rule:
    def __init__(self, spec, when=None, run=None):
        self.spec = spec
        self.name = spec.name
        self.when = when
        self.run = run
        self.after = spec.after
        self.once = spec.once
        self.fired_turn = None
        self.done = False

    def ready(self, ctx):
        if self.after is None:
            return True
        table = rules_by_name(ctx)
        other = table.get(self.after)
        if other is None:
            raise RunError(f"rule {self.name!r} waits for {self.after!r}, which is no rule; "
                           f"the rules are {sorted(table)}")
        return other.fired_turn is not None and other.fired_turn < ctx.turn

    def fire(self, ctx):
        if self.done and self.once:
            return False
        if not self.ready(ctx):
            return False
        if self.when is not None and not self.when.fires(ctx):
            return False
        for path, value in self.spec.set.items():
            apply_patch(ctx, path, value, by=self.name, turn=ctx.turn)
        if self.spec.stop:
            ctx.stop(f"{self.name}: {describe(self.when)}")
            ctx.stopped_by = self.name
        if self.run is not None:
            self.run(ctx)
        self.fired_turn = ctx.turn
        self.done = self.once
        ctx.fired.append(self.name)
        logger_of(ctx).info("rule %s fired at turn %s%s", self.name, ctx.turn, self.effects())
        if not self.once and self.when is not None:
            reset = getattr(self.when, "reset", None)
            if callable(reset):
                reset()
        return True

    def effects(self):
        parts = [f"{path}={value}" for path, value in self.spec.set.items()]
        if self.spec.stop:
            parts.append("stop")
        if self.run is not None:
            parts.append(f"run {type(self.run).__name__}")
        return ": " + ", ".join(parts) if parts else ""

    def state_dict(self):
        state = {"fired_turn": self.fired_turn, "done": self.done}
        if self.when is not None:
            state["when"] = self.when.state_dict()
        return state

    def load_state_dict(self, state):
        self.fired_turn = state.get("fired_turn")
        self.done = bool(state.get("done", False))
        if self.when is not None and "when" in state:
            self.when.load_state_dict(state["when"])


def patch_value_problem(value):
    if not isinstance(value, dict):
        return None
    if is_call(value):
        return "a call cannot be a patch value; set a scalar, {times: x}, {plus: x} or an @reference"
    if len(value) == 1 and next(iter(value)) in OPERATIONS:
        amount = next(iter(value.values()))
        if isinstance(amount, (int, float)) and not isinstance(amount, bool):
            return None
        return f"{next(iter(value))} takes a number, got {amount!r}"
    if set(value) in ({"ref"}, {"ref", "attr"}):
        return None
    return "a patch value is a scalar, {times: x}, {plus: x} or an @reference"


def parse_rules(raw, problems):
    if raw is None:
        return []
    if not isinstance(raw, dict):
        problems.append(error("rules_type", f"rules is a mapping of rule name to rule, got {type_name(raw)}", "rules"))
        return []
    keys_hint = "the rule keys are " + ", ".join(RULE_KEYS)
    specs = []
    for raw_name, body in raw.items():
        name = str(raw_name)
        path = f"rules.{name}"
        if not isinstance(body, dict):
            problems.append(error("rule_type", f"a rule is a mapping, got {type_name(body)}", path, hint=keys_hint))
            continue
        spec = RuleSpec(name)
        for key in body:
            if key not in RULE_KEYS:
                problems.append(error("rule_key", f"unknown rule key {key!r}", f"{path}.{key}", hint=keys_hint))
        when = body.get("when")
        if when is not None and not is_call(when) and not isinstance(when, str):
            problems.append(error("rule_value", f"when is a trigger call, got {type_name(when)}", f"{path}.when",
                                  hint="for example when: {use: plateau, watch: val/rmse, patience: 6}"))
        else:
            spec.when = when
        after = body.get("after")
        if after is not None:
            if not isinstance(after, str) or after not in raw:
                problems.append(error("rule_after", f"after names {after!r}, which is no rule", f"{path}.after",
                                      hint="the rules are " + ", ".join(str(key) for key in raw)))
            elif after == name:
                problems.append(error("rule_after", "a rule cannot wait for itself", f"{path}.after"))
            else:
                spec.after = after
        once = body.get("once", True)
        if isinstance(once, bool):
            spec.once = once
        else:
            problems.append(error("rule_value", f"once is true or false, got {once!r}", f"{path}.once"))
        settings = body.get("set")
        rejected = False
        if "set" in body and not isinstance(settings, dict):
            problems.append(error("rule_set", f"set is a mapping of plan path to value, got {type_name(settings)}",
                                  f"{path}.set", hint="for example set: {optimizers.main.lr: {times: 0.5}}"))
        elif settings:
            for key, value in settings.items():
                try:
                    tokenize(key)
                except RunError as failure:
                    problems.append(error("rule_path", str(failure), f"{path}.set"))
                    rejected = True
                    continue
                message = patch_value_problem(value)
                if message is not None:
                    problems.append(error("rule_value", f"{key}: {message}", f"{path}.set"))
                    rejected = True
                    continue
                spec.set[str(key)] = value
        stop = body.get("stop", False)
        if isinstance(stop, bool):
            spec.stop = stop
        else:
            problems.append(error("rule_value", f"stop is true or false, got {stop!r}", f"{path}.stop"))
        run = body.get("run")
        if run is not None and not is_call(run) and not isinstance(run, str):
            problems.append(error("rule_value", f"run is an action call, got {type_name(run)}", f"{path}.run"))
        else:
            spec.run = run
        malformed_set = rejected or ("set" in body and not isinstance(settings, dict))
        if not spec.set and not spec.stop and spec.run is None and not malformed_set:
            problems.append(error("rule_empty", "a rule needs set, stop or run to do anything", path))
        specs.append(spec)
    return specs


def holder_with(holder, key):
    if hasattr(holder, key):
        return holder
    for inner in INNER:
        wrapped = getattr(holder, inner, None)
        if wrapped is not None and hasattr(wrapped, key):
            return wrapped
    return None


def read(holder, token, path):
    if isinstance(holder, dict):
        if token not in holder:
            raise RunError(f"{path}: {token!r} is not a key of the mapping; its keys are {sorted(map(str, holder))}")
        return holder[token]
    if isinstance(holder, (list, tuple)):
        if not isinstance(token, int) or not 0 <= token < len(holder):
            raise RunError(f"{path}: {token!r} is no index of a list of {len(holder)}")
        return holder[token]
    if isinstance(token, int):
        raise RunError(f"{path}: {type_name(holder)} takes no index")
    owner = holder_with(holder, token)
    if owner is not None:
        return getattr(owner, token)
    params = getattr(holder, "params", None)
    if isinstance(params, dict) and token in params:
        return params[token]
    raise RunError(f"{path}: {type_name(holder)} has no attribute {token!r}")


def walk(holder, rest, path):
    if not rest:
        raise RunError(f"{path}: the path names an object, not one of its params")
    for token in rest[:-1]:
        holder = read(holder, token, path)
    kind = "train" if isinstance(holder, TrainSettings) else "generic"
    return holder, rest[-1], kind


def root_of(ctx, tokens, path):
    root = tokens[0]
    if root in TABLES:
        if len(tokens) < 2:
            raise RunError(f"{path}: a path names an entry like {root}.<name>")
        table = getattr(ctx, root, None) or {}
        name = tokens[1]
        if name not in table:
            raise RunError(f"{path}: no {SINGULAR[root]} named {name!r}; the {root} are {sorted(table)}")
        return table[name], tokens[2:]
    if root == "train":
        if len(tokens) > 1 and tokens[1] == "step":
            if ctx.step is None:
                raise RunError(f"{path}: there is no step object yet")
            return ctx.step, tokens[2:]
        if ctx.train is None:
            raise RunError(f"{path}: the train settings are not built yet")
        return ctx.train, tokens[1:]
    if root == "rules":
        if len(tokens) < 4 or tokens[2] not in ("when", "run"):
            raise RunError(f"{path}: a rule path is rules.<name>.when.<param> or rules.<name>.run.<param>")
        table = rules_by_name(ctx)
        rule = table.get(tokens[1])
        if rule is None:
            raise RunError(f"{path}: no rule named {tokens[1]!r}; the rules are {sorted(table)}")
        part = getattr(rule, tokens[2], None)
        if part is None:
            raise RunError(f"{path}: rule {tokens[1]!r} has no {tokens[2]}")
        return part, tokens[3:]
    raise RunError(f"{path}: unknown path root {root!r}; the roots are {', '.join(ROOTS)}")


def read_path(ctx, path):
    tokens = tokenize(path)
    holder, rest = root_of(ctx, tokens, path)
    for token in rest:
        holder = read(holder, token, path)
    return holder


def locate(ctx, tokens, path):
    root = tokens[0]
    holder, rest = root_of(ctx, tokens, path)
    if root in TABLES:
        if not rest:
            raise RunError(f"{path}: a patch names an entry and one of its params, like {root}.<name>.<param>")
        if root == "optimizers" and len(rest) == 1:
            return holder, rest[0], "optimizer"
        if root == "models" and rest == ["freeze"]:
            return holder, "freeze", "model"
    return walk(holder, rest, path)


def current_of(holder, key, kind):
    if kind == "optimizer":
        if key == "minimizes":
            return getattr(holder, "minimizes", None)
        params = getattr(holder, "params", None)
        if isinstance(params, dict) and key in params:
            return params[key]
        groups = getattr(getattr(holder, "real", None), "param_groups", None)
        if groups:
            return groups[0].get(key)
        return None
    if kind == "model":
        return getattr(holder, "freeze", None)
    try:
        return read(holder, key, "")
    except RunError:
        return None


def as_int(value, path):
    try:
        return int(value)
    except (TypeError, ValueError):
        raise RunError(f"{path}: expected an integer, got {value!r}") from None


def write(holder, key, value, kind, path):
    if kind == "optimizer":
        if key == "minimizes":
            holder.minimizes = value
            return
        if hasattr(holder, "set_param"):
            holder.set_param(key, value)
            return
    elif kind == "model":
        if hasattr(holder, "set_freeze"):
            holder.set_freeze(value)
        else:
            holder.freeze = value
        return
    elif kind == "train":
        if key == "epochs":
            holder.epochs = as_int(value, path)
            return
        if key == "updates" and isinstance(holder.updates, dict) and not isinstance(value, dict):
            holder.updates["total"] = as_int(value, path)
            return
    if isinstance(holder, dict):
        holder[key] = value
        return
    if isinstance(holder, tuple):
        raise RunError(f"{path}: a tuple cannot be patched in place")
    if isinstance(holder, list):
        if not isinstance(key, int) or not 0 <= key < len(holder):
            raise RunError(f"{path}: {key!r} is no index of a list of {len(holder)}")
        holder[key] = value
        return
    if isinstance(key, int):
        raise RunError(f"{path}: {type_name(holder)} takes no index")
    owner = holder_with(holder, key)
    if owner is None:
        raise RunError(f"{path}: {type_name(holder)} has no attribute {key!r}")
    setattr(owner, key, value)


def resolve_ref(ctx, text):
    label = f"@{text}"
    tokens = tokenize(text)
    root, rest = tokens[0], tokens[1:]
    if root in TABLES:
        if not rest:
            raise RunError(f"{label} names a section; a reference names an entry like @{root}.<name>")
        table = getattr(ctx, root, None) or {}
        name = rest[0]
        if name not in table:
            raise RunError(f"{label}: no {SINGULAR[root]} named {name!r}; the {root} are {sorted(table)}")
        obj = table[name]
        rest = rest[1:]
        if root == "models":
            if rest and rest[0] == "ema":
                obj = getattr(obj, "ema", None)
                if obj is None:
                    raise RunError(f"{label}: model {name!r} has no ema copy")
                rest = rest[1:]
            else:
                obj = getattr(obj, "module", obj)
    elif root in ("data", "generate"):
        obj = getattr(ctx, root, None)
        if obj is None:
            raise RunError(f"{label}: there is no {root} object")
    else:
        raise RunError(f"{label}: unknown reference root {root!r}; a reference starts with "
                       + ", ".join(TABLES + ("data", "generate")))
    for token in rest:
        obj = read(obj, token, label)
    return obj


def resolve_value(ctx, current, asked):
    text = ref_parts(asked)
    if text is not None:
        return resolve_ref(ctx, text)
    if isinstance(asked, dict) and len(asked) == 1:
        operation, amount = next(iter(asked.items()))
        if operation in OPERATIONS:
            if current is None:
                raise RunError(f"{operation} needs a current value to work from, and there is none")
            return current * amount if operation == "times" else current + amount
    return asked


def node_prefix(nodes, tokens):
    for count in range(len(tokens) - 1, 0, -1):
        text = join_tokens(tokens[:count])
        if text in nodes:
            return text, tokens[count:]
    return None, tokens


def assign_nested(container, tokens, value):
    for token in tokens[:-1]:
        if isinstance(container, dict) and token in container:
            container = container[token]
        elif isinstance(container, list) and isinstance(token, int) and 0 <= token < len(container):
            container = container[token]
        else:
            return
    last = tokens[-1]
    if isinstance(container, dict) and not isinstance(last, int):
        container[last] = value
    elif isinstance(container, list) and isinstance(last, int) and 0 <= last < len(container):
        container[last] = value


def bucket_of(node, key):
    keys = node.get("keys")
    params = node.get("params")
    if isinstance(keys, dict) and key in keys:
        return keys
    if isinstance(params, dict) and key in params:
        return params
    from .build import SLOT_KEYS

    if key in SLOT_KEYS.get(node.get("slot"), ()):
        return node.setdefault("keys", {})
    return node.setdefault("params", {})


def plan_write(plan, tokens, value):
    if not isinstance(plan, dict):
        return
    if tokens[0] == "train" and (len(tokens) < 2 or tokens[1] != "step"):
        section = plan.get("train")
        if isinstance(section, dict) and len(tokens) > 1:
            assign_nested(section, tokens[1:], value)
        return
    nodes = plan.get("nodes")
    if not isinstance(nodes, dict):
        return
    node_path, rest = node_prefix(nodes, tokens)
    if node_path is None or not rest or not isinstance(nodes[node_path], dict):
        return
    assign_nested(bucket_of(nodes[node_path], rest[0]), rest, value)


def apply_patch(ctx, path, asked, by="", turn=None):
    turn = ctx.turn if turn is None else turn
    tokens = tokenize(path)
    holder, key, kind = locate(ctx, tokens, path)
    current = current_of(holder, key, kind)
    value = resolve_value(ctx, current, asked)
    write(holder, key, value, kind, path)
    value_json = ref_json(asked) if ref_parts(asked) is not None else jsonable(value, ctx)
    plan_write(getattr(ctx, "plan", None), tokens, value_json)
    record = getattr(ctx, "record", None)
    if record is not None:
        record.event("patch", turn=turn, by=by, path=path, value=value_json, asked=jsonable(asked, ctx))
        write_plan(ctx, record)
    patch = Patch(path, value, asked, by, turn)
    patches = getattr(ctx, "patches", None)
    if isinstance(patches, list):
        patches.append(patch)
    return patch


def write_plan(ctx, record):
    plan = getattr(ctx, "plan", None)
    writer = getattr(record, "write_json", None)
    if isinstance(plan, dict) and callable(writer):
        writer("plan.json", plan)


def replay(ctx, events, upto_turn):
    record = ctx.record
    ctx.record = None
    count = 0
    try:
        for event in events:
            if not isinstance(event, dict) or event.get("kind") != "patch":
                continue
            turn = event.get("turn") or 0
            if turn > upto_turn:
                continue
            by = f"{event.get('by') or 'patch'} (replay)"
            apply_patch(ctx, event["path"], event["value"], by=by, turn=turn)
            count += 1
    finally:
        ctx.record = record
    if count and record is not None:
        write_plan(ctx, record)
    return count
