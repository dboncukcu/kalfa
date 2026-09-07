"""The aliasing check: bus keys that name the same mutable object, and unordered siblings that touch one of them
while another mutates it.

A serial run orders nodes by their bus dependencies alone, so two siblings without a dependency run in an order
that happens to be right; the thread executor runs them side by side and the outcome depends on timing. The alias
table comes from the legos' facts: ``mutates`` (the objects come back under the output of the same name) and
``aliases`` (the output holds the inputs), plus the loop carry and the frame renames.
"""

import functools

from cirak.errors import warning


class Groups:
    """Union find over key names: keys in one group name the same object."""

    def __init__(self):
        self.parent = {}

    def find(self, key):
        self.parent.setdefault(key, key)
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def union(self, first, second):
        root_first, root_second = self.find(first), self.find(second)
        if root_first != root_second:
            self.parent[root_second] = root_first

    def of(self, keys):
        return {self.find(key) for key in keys}


def keys_of(binding):
    """Every bus key a binding spec names: a key, a bundle mapping, a list."""
    if isinstance(binding, str):
        return [binding]
    if isinstance(binding, dict):
        return [key for value in binding.values() for key in keys_of(value)]
    if isinstance(binding, (list, tuple)):
        return [key for value in binding for key in keys_of(value)]
    return []


def unwrap(fn):
    while isinstance(fn, functools.partial):
        fn = fn.func
    return fn


class Analysis:
    def __init__(self, registry):
        self.groups = Groups()
        self.mutated = {}
        self.touched = {}
        self.frames = []
        self.targets = {}
        for uri in registry.uris():
            entry = registry.lookup(uri)
            if entry is not None and not entry.fragment:
                self.targets[id(entry.target)] = entry.facts

    def facts_of(self, fn):
        return self.targets.get(id(unwrap(fn)))

    def visit(self, resolution, path):
        kind = getattr(resolution, "kind", None)
        if kind == "step":
            self.step(resolution)
        elif kind == "pipeline":
            for outer, inner in (resolution.inputs or {}).items():
                self.groups.union(outer, inner)
            for inner, outer in (resolution.outputs or {}).items():
                self.groups.union(inner, outer)
            mutated, touched = set(), set(resolution.reads) | set(resolution.writes)
            for name, child in resolution.table.items():
                self.visit(child, f"{path}.{name}" if path else name)
                mutated |= self.mutated[id(child)]
                touched |= self.touched[id(child)]
            self.frames.append((path, resolution))
            self.mutated[id(resolution)] = mutated
            self.touched[id(resolution)] = touched
        elif kind in ("loop", "map"):
            node = resolution.node
            for carry, target in (getattr(node, "next", None) or {}).items():
                self.groups.union(carry, target)
            for carry, final in (getattr(node, "outputs", None) or {}).items():
                self.groups.union(carry, final)
            self.visit(resolution.body, f"{path}.body")
            self.mutated[id(resolution)] = set(self.mutated[id(resolution.body)])
            self.touched[id(resolution)] = set(self.touched[id(resolution.body)]) | set(resolution.reads) | set(
                resolution.writes)
        elif kind == "branch":
            mutated, touched = set(), set(resolution.reads) | set(resolution.writes)
            for label, case in (resolution.cases or {}).items():
                self.visit(case, f"{path}.{label}")
                mutated |= self.mutated[id(case)]
                touched |= self.touched[id(case)]
            self.mutated[id(resolution)] = mutated
            self.touched[id(resolution)] = touched
        else:
            self.mutated[id(resolution)] = set()
            self.touched[id(resolution)] = set(getattr(resolution, "reads", [])) | set(
                getattr(resolution, "writes", []))

    def step(self, resolution):
        step = resolution.node
        facts = self.facts_of(step.fn)
        binding = resolution.binding or {}
        select = step.select or {name: name for name in step.outputs}
        mutated = set()
        if facts is not None:
            for param in facts.mutates:
                keys = keys_of(binding.get(param))
                mutated.update(keys)
                if param in select:
                    for key in keys:
                        self.groups.union(key, select[param])
            for param in facts.aliases:
                for key in keys_of(binding.get(param)):
                    for output in step.outputs:
                        self.groups.union(key, output)
        self.mutated[id(resolution)] = mutated
        self.touched[id(resolution)] = set(resolution.reads) | set(resolution.writes)


def ancestors_of(deps):
    """The transitive upstream set of every node of a frame."""
    found = {}

    def visit(name, seen):
        if name in found:
            return found[name]
        seen.add(name)
        upstream = set()
        for parent in deps.get(name, ()):
            upstream.add(parent)
            if parent not in seen:
                upstream |= visit(parent, seen)
        found[name] = upstream
        return upstream

    for name in deps:
        visit(name, set())
    return found


def aliasing_problems(pipeline, registry):
    """Warnings for every pair of unordered siblings where one mutates an object the other touches."""
    resolved = getattr(pipeline, "resolved", None)
    if resolved is None:
        return []
    analysis = Analysis(registry)
    analysis.visit(resolved, "")
    problems = []
    for path, frame in analysis.frames:
        names = list(frame.table)
        ancestors = ancestors_of({name: set(frame.deps.get(name, ())) for name in names})
        for position, first in enumerate(names):
            for second in names[position + 1:]:
                if first in ancestors.get(second, ()) or second in ancestors.get(first, ()):
                    continue
                for writer, reader in ((first, second), (second, first)):
                    shared = _shared(analysis, frame.table[writer], frame.table[reader])
                    if shared:
                        where = f"{path}." if path else ""
                        problems.append(warning(
                            "aliasing",
                            f"{where}{writer} mutates {sorted(shared)} while {where}{reader} touches the same "
                            f"object with no order between them; the thread executor runs them side by side",
                            hint="order the two nodes (wait_for, or a key one writes and the other reads) or "
                                 "return a copy instead of mutating in place"))
    return problems


def _shared(analysis, writer, reader):
    groups = analysis.groups
    mutated = {groups.find(key): key for key in analysis.mutated[id(writer)]}
    touched = groups.of(analysis.touched[id(reader)])
    return {key for group, key in mutated.items() if group in touched}
