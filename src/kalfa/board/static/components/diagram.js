const SIZE = { line: 14, padX: 10, padY: 7, minWidth: 64, chars: 30, head: 22 };
const SPACE = { rank: 46, order: 16, dummy: 8, port: 18, pad: 12 };
const CHAR = { label: 7.1, detail: 6.1, mono: 6.7, corner: 6.0, head: 6.7 };
const SWEEPS = 8;
const PASSES = 6;

function textOf(value) {
  return value === null || value === undefined ? "" : String(value);
}

function clip(value, chars) {
  const text = textOf(value);
  return text.length > chars ? text.slice(0, Math.max(1, chars - 1)) + "…" : text;
}

function trimmed(value, digits) {
  return value.toFixed(digits).replace(/\.0+$/, "").replace(/(\.\d*?)0+$/, "$1");
}

export function abbreviate(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "";
  const size = Math.abs(value);
  for (const [step, unit] of [[1e12, "T"], [1e9, "B"], [1e6, "M"], [1e3, "k"]]) {
    if (size >= step) {
      const scaled = value / step;
      const digits = Math.abs(scaled) >= 100 ? 0 : Math.abs(scaled) >= 10 ? 1 : 2;
      return trimmed(scaled, digits) + unit;
    }
  }
  return Number.isInteger(value) ? String(value) : trimmed(value, 2);
}

export function shapeText(shape) {
  return Array.isArray(shape) ? "[" + shape.join(", ") + "]" : "";
}

function isNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

function cornerOf(count, parameters) {
  if (isNumber(count)) return abbreviate(count);
  if (isNumber(parameters) && parameters > 0) return abbreviate(parameters);
  return "";
}

function kindClass(kind) {
  return "kind-" + (textOf(kind).toLowerCase().replace(/[^a-z0-9_-]/g, "") || "plain");
}

function present(value) {
  return value !== undefined && value !== null;
}

function indexGraph(graph) {
  const source = graph && typeof graph === "object" ? graph : {};
  const nodes = (Array.isArray(source.nodes) ? source.nodes : []).filter(node => node && present(node.id));
  const groups = new Map();
  for (const group of Array.isArray(source.groups) ? source.groups : []) {
    if (group && present(group.id)) groups.set(String(group.id), group);
  }
  const edges = (Array.isArray(source.edges) ? source.edges : []).filter(edge => edge && present(edge.from) && present(edge.to));
  const chains = new Map();
  const chain = (id) => {
    const start = present(id) ? String(id) : null;
    if (start === null || !groups.has(start)) return [];
    if (chains.has(start)) return chains.get(start);
    const found = [];
    const seen = new Set();
    let current = start;
    while (current !== null && groups.has(current) && !seen.has(current)) {
      seen.add(current);
      found.unshift(current);
      const parent = groups.get(current).parent;
      current = present(parent) ? String(parent) : null;
    }
    chains.set(start, found);
    return found;
  };
  return { nodes, groups, edges, chain };
}

function visibleOf(base, closed) {
  const units = new Map();
  const owner = new Map();
  base.nodes.forEach((node, index) => {
    const chain = base.chain(node.group);
    const at = chain.findIndex(id => closed.has(id));
    let key;
    if (at >= 0) {
      const gid = chain[at];
      key = "group:" + gid;
      if (!units.has(key)) {
        units.set(key, { key, gid, collapsed: true, kind: "group", group: base.groups.get(gid), members: 0, index,
                         chain: chain.slice(0, at) });
      }
      units.get(key).members += 1;
    } else {
      key = "node:" + node.id;
      units.set(key, { key, collapsed: false, kind: textOf(node.kind), node, index, chain });
    }
    owner.set(String(node.id), key);
  });
  const merged = new Map();
  base.edges.forEach((edge) => {
    const from = owner.get(String(edge.from));
    const to = owner.get(String(edge.to));
    if (!from || !to || from === to) return;
    const key = from + ">" + to;
    if (!merged.has(key)) merged.set(key, { key, from, to, count: 0, labels: new Set(), shapes: new Set() });
    const item = merged.get(key);
    item.count += 1;
    item.labels.add(textOf(edge.label));
    item.shapes.add(shapeText(edge.shape));
  });
  const edges = [...merged.values()].map(item => ({
    key: item.key, from: item.from, to: item.to, count: item.count,
    label: item.labels.size === 1 ? [...item.labels][0] : "",
    shape: item.shapes.size === 1 ? [...item.shapes][0] : "",
  }));
  return { units, edges };
}

function unitText(unit) {
  if (unit.collapsed) {
    const group = unit.group || {};
    const inside = unit.members === 1 ? "1 node inside" : unit.members + " nodes inside";
    return { label: textOf(group.label || unit.gid), detail: [textOf(group.detail), inside].filter(Boolean).join(" · "),
             shape: "", corner: cornerOf(undefined, group.parameters) };
  }
  const node = unit.node;
  return { label: textOf(node.label || node.id), detail: textOf(node.detail), shape: shapeText(node.shape),
           corner: cornerOf(node.count, node.parameters) };
}

function measure(text) {
  const label = clip(text.label, SIZE.chars);
  const detail = clip(text.detail, SIZE.chars + 8);
  const shape = clip(text.shape, SIZE.chars + 8);
  const head = label.length * CHAR.label + (text.corner ? text.corner.length * CHAR.corner + 12 : 0);
  const widest = Math.max(head, detail.length * CHAR.detail, shape.length * CHAR.mono);
  const rows = 1 + (detail ? 1 : 0) + (shape ? 1 : 0);
  return { label, detail, shape, corner: text.corner, width: Math.max(SIZE.minWidth, Math.ceil(widest + 2 * SIZE.padX)),
           height: rows * SIZE.line + 2 * SIZE.padY };
}

function boxesOf(base, units) {
  const boxes = new Map([["", { key: "", gid: null, items: [], depth: 0, index: 0 }]]);
  for (const unit of units.values()) {
    let parent = "";
    unit.chain.forEach((gid, position) => {
      const key = "box:" + gid;
      if (!boxes.has(key)) {
        boxes.set(key, { key, gid, group: base.groups.get(gid) || {}, items: [], depth: position + 1, parent, index: unit.index });
        boxes.get(parent).items.push(key);
      }
      const box = boxes.get(key);
      box.index = Math.min(box.index, unit.index);
      parent = key;
    });
    boxes.get(parent).items.push(unit.key);
    unit.container = parent;
  }
  return boxes;
}

function ancestry(unit) {
  return ["", ...unit.chain.map(gid => "box:" + gid)];
}

function plan(view) {
  const wants = new Map();
  const want = (container, from, to) => {
    if (!wants.has(container)) wants.set(container, new Map());
    const key = from + ">" + to;
    if (!wants.get(container).has(key)) wants.get(container).set(key, { key, from, to });
    return { container, from, to, key };
  };
  const routes = [];
  for (const edge of view.edges) {
    const up = ancestry(view.units.get(edge.from));
    const down = ancestry(view.units.get(edge.to));
    let low = 0;
    while (low + 1 < up.length && low + 1 < down.length && up[low + 1] === down[low + 1]) low += 1;
    const first = low + 1 < up.length ? up[low + 1] : edge.from;
    const last = low + 1 < down.length ? down[low + 1] : edge.to;
    const segments = [];
    for (let level = up.length - 1; level > low; level -= 1) {
      segments.push(want(up[level], level === up.length - 1 ? edge.from : up[level + 1], "out:" + last));
    }
    segments.push(want(up[low], first, last));
    for (let level = low + 1; level < down.length; level += 1) {
      segments.push(want(down[level], "in:" + first, level === down.length - 1 ? edge.to : down[level + 1]));
    }
    routes.push({ edge, segments });
  }
  return { wants, routes };
}

function isPort(key) {
  return key.startsWith("in:") || key.startsWith("out:");
}

function layered(keys, links, indexOf) {
  const succs = new Map(keys.map(key => [key, []]));
  for (const link of links) if (succs.has(link.from) && succs.has(link.to)) succs.get(link.from).push(link.to);
  const order = [...keys].sort((a, b) => indexOf(a) - indexOf(b));
  const state = new Map();
  const finished = [];
  const back = new Set();
  for (const start of order) {
    if (state.has(start)) continue;
    const stack = [[start, 0]];
    state.set(start, 1);
    while (stack.length) {
      const top = stack[stack.length - 1];
      const next = succs.get(top[0])[top[1]];
      if (next === undefined) {
        state.set(top[0], 2);
        finished.push(top[0]);
        stack.pop();
        continue;
      }
      top[1] += 1;
      if (!state.has(next)) {
        state.set(next, 1);
        stack.push([next, 0]);
      } else if (state.get(next) === 1) {
        back.add(top[0] + ">" + next);
      }
    }
  }
  const topo = finished.reverse();
  const preds = new Map(keys.map(key => [key, []]));
  const forward = new Map(keys.map(key => [key, []]));
  for (const link of links) {
    if (!succs.has(link.from) || !succs.has(link.to) || back.has(link.from + ">" + link.to)) continue;
    preds.get(link.to).push(link.from);
    forward.get(link.from).push(link.to);
  }
  const layer = new Map();
  for (const key of topo) layer.set(key, Math.max(0, ...preds.get(key).map(other => layer.get(other) + 1)));
  for (const key of [...topo].reverse()) {
    if (preds.get(key).length || !forward.get(key).length) continue;
    layer.set(key, Math.max(layer.get(key), Math.min(...forward.get(key).map(other => layer.get(other))) - 1));
  }
  return { layer, back };
}

function crossings(layers, succs) {
  let count = 0;
  for (let index = 0; index + 1 < layers.length; index += 1) {
    const below = new Map(layers[index + 1].map((key, position) => [key, position]));
    const pairs = [];
    layers[index].forEach((key, position) => {
      for (const other of succs.get(key) || []) if (below.has(other)) pairs.push([position, below.get(other)]);
    });
    for (let first = 0; first < pairs.length; first += 1) {
      for (let second = first + 1; second < pairs.length; second += 1) {
        const a = pairs[first];
        const b = pairs[second];
        if ((a[0] - b[0]) * (a[1] - b[1]) < 0) count += 1;
      }
    }
  }
  return count;
}

function reorder(layers, preds, succs) {
  let current = layers.map(layer => layer.slice());
  let best = current.map(layer => layer.slice());
  let fewest = crossings(best, succs);
  for (let sweep = 0; sweep < SWEEPS && fewest > 0; sweep += 1) {
    const down = sweep % 2 === 0;
    const indices = [...current.keys()];
    const visit = down ? indices.slice(1) : indices.reverse().slice(1);
    for (const index of visit) {
      const reference = current[down ? index - 1 : index + 1];
      const position = new Map(reference.map((key, place) => [key, place]));
      const near = down ? preds : succs;
      const scored = current[index].map((key, place) => {
        const linked = (near.get(key) || []).filter(other => position.has(other));
        const score = linked.length ? linked.reduce((sum, other) => sum + position.get(other), 0) / linked.length : place;
        return { key, score, place };
      });
      scored.sort((a, b) => a.score - b.score || a.place - b.place);
      current[index] = scored.map(item => item.key);
    }
    const count = crossings(current, succs);
    if (count < fewest) {
      fewest = count;
      best = current.map(layer => layer.slice());
    }
  }
  return best;
}

function centers(layers, extent, gap, preds, succs) {
  const center = new Map();
  for (const layer of layers) {
    let cursor = 0;
    layer.forEach((key, place) => {
      if (place) cursor += gap(layer[place - 1], key);
      center.set(key, cursor + extent(key) / 2);
      cursor += extent(key);
    });
    for (const key of layer) center.set(key, center.get(key) - cursor / 2);
  }
  for (let pass = 0; pass < PASSES; pass += 1) {
    const down = pass % 2 === 0;
    const near = down ? preds : succs;
    for (const layer of down ? layers : [...layers].reverse()) {
      const wanted = layer.map((key) => {
        const linked = (near.get(key) || []).filter(other => center.has(other));
        return linked.length ? linked.reduce((sum, other) => sum + center.get(other), 0) / linked.length : center.get(key);
      });
      const placed = wanted.slice();
      for (let place = 1; place < layer.length; place += 1) {
        const least = placed[place - 1] + extent(layer[place - 1]) / 2 + gap(layer[place - 1], layer[place]) + extent(layer[place]) / 2;
        if (placed[place] < least) placed[place] = least;
      }
      const shift = layer.length ? wanted.reduce((sum, value, place) => sum + value - placed[place], 0) / layer.length : 0;
      layer.forEach((key, place) => center.set(key, placed[place] + shift));
    }
  }
  return center;
}

function layoutBox(key, scene) {
  if (scene.layouts.has(key)) return scene.layouts.get(key);
  const box = scene.boxes.get(key);
  const lr = scene.lr;
  const root = key === "";
  const pad = root ? 0 : SPACE.pad;
  const head = root ? 0 : SIZE.head;
  const requested = [...(scene.plan.wants.get(key) || new Map()).values()];
  const items = new Map();
  for (const item of box.items) {
    const size = item.startsWith("box:") ? layoutBox(item, scene) : scene.sizes.get(item);
    const index = item.startsWith("box:") ? scene.boxes.get(item).index : scene.units.get(item).index;
    items.set(item, { key: item, w: size.width, h: size.height, index, kind: "item" });
  }
  for (const link of requested) {
    for (const [end, other] of [[link.from, link.to], [link.to, link.from]]) {
      if (!isPort(end)) continue;
      const index = items.has(other) ? items.get(other).index : 0;
      if (!items.has(end)) items.set(end, { key: end, w: 0, h: 0, index, kind: end.startsWith("in:") ? "in" : "out" });
      else items.get(end).index = Math.min(items.get(end).index, index);
    }
  }
  const real = [...items.values()].filter(item => item.kind === "item").map(item => item.key);
  const indexOf = name => items.get(name).index;
  const inner = requested.filter(link => items.has(link.from) && items.has(link.to) && !isPort(link.from) && !isPort(link.to));
  const { layer, back } = layered(real, inner, indexOf);
  const hasIn = requested.some(link => link.from.startsWith("in:"));
  const hasOut = requested.some(link => link.to.startsWith("out:"));
  if (hasIn) for (const name of real) layer.set(name, layer.get(name) + 1);
  const deepest = Math.max(hasIn ? 0 : -1, ...real.map(name => layer.get(name)));
  for (const item of items.values()) {
    if (item.kind === "in") layer.set(item.key, 0);
    if (item.kind === "out") layer.set(item.key, deepest + 1);
  }
  const count = Math.max(0, ...[...layer.values()].map(value => value + 1));
  const layers = Array.from({ length: count }, () => []);
  for (const item of [...items.values()].sort((a, b) => a.index - b.index)) layers[layer.get(item.key)].push(item.key);
  const chains = new Map();
  const preds = new Map();
  const succs = new Map();
  const join = (from, to) => {
    if (!succs.has(from)) succs.set(from, []);
    if (!preds.has(to)) preds.set(to, []);
    succs.get(from).push(to);
    preds.get(to).push(from);
  };
  for (const link of requested) {
    if (!items.has(link.from) || !items.has(link.to)) continue;
    const start = layer.get(link.from);
    const end = layer.get(link.to);
    const middle = [];
    if (!back.has(link.from + ">" + link.to) && end > start) {
      let previous = link.from;
      for (let step = start + 1; step < end; step += 1) {
        const dummy = link.key + "#" + step;
        items.set(dummy, { key: dummy, w: 0, h: 2, index: items.get(link.from).index + 0.5, kind: "dummy" });
        layer.set(dummy, step);
        layers[step].push(dummy);
        join(previous, dummy);
        middle.push(dummy);
        previous = dummy;
      }
      join(previous, link.to);
    }
    chains.set(link.key, middle);
  }
  const ordered = reorder(layers, preds, succs);
  const extent = name => (lr ? items.get(name).h : items.get(name).w);
  const length = name => (lr ? items.get(name).w : items.get(name).h);
  const loose = name => items.get(name).kind !== "item";
  const gap = (a, b) => (loose(a) || loose(b) ? SPACE.dummy : SPACE.order);
  const center = centers(ordered, extent, gap, preds, succs);
  const rankStart = pad + (lr ? 0 : head);
  const orderStart = pad + (lr ? head : 0);
  const starts = [];
  const sizes = [];
  let cursor = rankStart + (hasIn ? SPACE.port : 0);
  let any = false;
  ordered.forEach((names, index) => {
    const ports = names.length && names.every(name => items.get(name).kind === "in" || items.get(name).kind === "out");
    if (ports) {
      starts[index] = 0;
      sizes[index] = 0;
      return;
    }
    starts[index] = cursor;
    sizes[index] = Math.max(0, ...names.map(length));
    cursor += sizes[index] + SPACE.rank;
    any = true;
  });
  const contentEnd = any ? cursor - SPACE.rank : cursor;
  let low = Infinity;
  let high = -Infinity;
  for (const [name, value] of center) {
    low = Math.min(low, value - extent(name) / 2);
    high = Math.max(high, value + extent(name) / 2);
  }
  if (!Number.isFinite(low)) {
    low = 0;
    high = 0;
  }
  const group = box.group || {};
  const title = root ? 0 : textOf(group.label).length * CHAR.head + textOf(group.detail).length * CHAR.detail + 24 +
    (cornerOf(undefined, group.parameters).length * CHAR.corner);
  let rankSize = contentEnd + (hasOut ? SPACE.port : 0) + pad;
  let orderSize = orderStart + (high - low) + pad;
  const offset = orderStart - low;
  if (lr) rankSize = Math.max(rankSize, title);
  else orderSize = Math.max(orderSize, title);
  const rects = new Map();
  for (const [name, item] of items) {
    if (item.kind === "dummy") continue;
    const index = layer.get(name);
    const middle = center.get(name) + offset;
    let along;
    if (item.kind === "in") along = 0;
    else if (item.kind === "out") along = rankSize;
    else along = starts[index] + (sizes[index] - length(name)) / 2;
    rects.set(name, lr ? { x: along, y: middle - item.h / 2, w: item.w, h: item.h, kind: item.kind }
      : { x: middle - item.w / 2, y: along, w: item.w, h: item.h, kind: item.kind });
  }
  const point = (along, across) => (lr ? { x: along, y: across } : { x: across, y: along });
  const exit = (name) => {
    const rect = rects.get(name);
    if (rect.kind !== "item") return { x: rect.x, y: rect.y };
    return lr ? { x: rect.x + rect.w, y: rect.y + rect.h / 2 } : { x: rect.x + rect.w / 2, y: rect.y + rect.h };
  };
  const entry = (name) => {
    const rect = rects.get(name);
    if (rect.kind !== "item") return { x: rect.x, y: rect.y };
    return lr ? { x: rect.x, y: rect.y + rect.h / 2 } : { x: rect.x + rect.w / 2, y: rect.y };
  };
  const routes = new Map();
  for (const link of requested) {
    if (!rects.has(link.from) || !rects.has(link.to)) continue;
    const points = [exit(link.from)];
    for (const dummy of chains.get(link.key) || []) {
      const index = layer.get(dummy);
      const across = center.get(dummy) + offset;
      points.push(point(starts[index], across), point(starts[index] + sizes[index], across));
    }
    points.push(entry(link.to));
    routes.set(link.key, points);
  }
  const width = lr ? rankSize : orderSize;
  const height = lr ? orderSize : rankSize;
  const result = { width, height, rects, routes };
  scene.layouts.set(key, result);
  return result;
}

function placeBox(key, ox, oy, scene, out) {
  const layout = scene.layouts.get(key);
  scene.offsets.set(key, { x: ox, y: oy });
  for (const [name, rect] of layout.rects) {
    if (rect.kind !== "item") continue;
    const x = ox + rect.x;
    const y = oy + rect.y;
    if (name.startsWith("box:")) {
      const box = scene.boxes.get(name);
      const group = box.group || {};
      const corner = cornerOf(undefined, group.parameters);
      const room = Math.max(4, Math.floor((rect.w - 24 - corner.length * CHAR.corner) / CHAR.head));
      const label = clip(group.label || box.gid, room);
      const detail = clip(group.detail, Math.max(0, room - label.length - 1));
      out.boxes.push({ key: name, gid: box.gid, x, y, w: rect.w, h: rect.h, depth: box.depth, label, detail, corner,
                       title: [textOf(group.label || box.gid), textOf(group.detail),
                               isNumber(group.parameters) ? group.parameters.toLocaleString() + " parameters" : "",
                               "click the header to fold this group"].filter(Boolean).join("\n") });
      placeBox(name, x, y, scene, out);
    } else {
      const unit = scene.units.get(name);
      const shown = scene.texts.get(name);
      out.nodes.push({ key: name, x, y, w: rect.w, h: rect.h, unit, kind: unit.kind, ...shown, title: titleOf(unit) });
    }
  }
}

function titleOf(unit) {
  if (unit.collapsed) {
    const group = unit.group || {};
    return [textOf(group.label || unit.gid), textOf(group.detail), unit.members + " nodes inside",
            isNumber(group.parameters) ? group.parameters.toLocaleString() + " parameters" : "",
            "click to unfold"].filter(Boolean).join("\n");
  }
  const node = unit.node;
  return [textOf(node.label || node.id), textOf(node.detail), shapeText(node.shape),
          isNumber(node.count) ? "count " + node.count.toLocaleString() : "",
          isNumber(node.parameters) ? node.parameters.toLocaleString() + " parameters" : ""].filter(Boolean).join("\n");
}

function shifted(points, offset) {
  return points.map(item => ({ x: item.x + offset.x, y: item.y + offset.y }));
}

function joined(segments, scene) {
  let points = [];
  for (let index = 0; index < segments.length; index += 1) {
    const segment = segments[index];
    const route = scene.layouts.get(segment.container).routes.get(segment.key);
    if (!route) return null;
    let part = shifted(route, scene.offsets.get(segment.container));
    if (index > 0 && segments[index - 1].to.startsWith("out:")) part = part.slice(1);
    if (segment.from.startsWith("in:")) points = points.slice(0, -1);
    points = points.concat(part);
  }
  return points.filter((item, index) => !index || Math.abs(item.x - points[index - 1].x) + Math.abs(item.y - points[index - 1].y) > 0.5);
}

function curve(points, lr) {
  if (!points.length) return "";
  let path = `M${points[0].x.toFixed(1)} ${points[0].y.toFixed(1)}`;
  for (let index = 1; index < points.length; index += 1) {
    const a = points[index - 1];
    const b = points[index];
    if (lr) {
      const bend = Math.max(Math.abs(b.x - a.x) / 2, 12);
      path += ` C${(a.x + bend).toFixed(1)} ${a.y.toFixed(1)} ${(b.x - bend).toFixed(1)} ${b.y.toFixed(1)} ${b.x.toFixed(1)} ${b.y.toFixed(1)}`;
    } else {
      const bend = Math.max(Math.abs(b.y - a.y) / 2, 12);
      path += ` C${a.x.toFixed(1)} ${(a.y + bend).toFixed(1)} ${b.x.toFixed(1)} ${(b.y - bend).toFixed(1)} ${b.x.toFixed(1)} ${b.y.toFixed(1)}`;
    }
  }
  return path;
}

export function layoutGraph(graph, closed, direction) {
  const lr = direction !== "TB";
  const base = indexGraph(graph);
  const view = visibleOf(base, closed || new Set());
  const boxes = boxesOf(base, view.units);
  const texts = new Map();
  const sizes = new Map();
  for (const unit of view.units.values()) {
    const shown = measure(unitText(unit));
    texts.set(unit.key, shown);
    sizes.set(unit.key, shown);
  }
  const scene = { lr, boxes, units: view.units, texts, sizes, plan: plan(view), layouts: new Map(), offsets: new Map() };
  layoutBox("", scene);
  const out = { nodes: [], boxes: [], edges: [] };
  placeBox("", 0, 0, scene, out);
  for (const route of scene.plan.routes) {
    const points = joined(route.segments, scene);
    if (!points || points.length < 2) continue;
    const edge = route.edge;
    const middle = Math.max(0, Math.floor((points.length - 2) / 2));
    const a = points[middle];
    const b = points[middle + 1];
    const from = scene.texts.get(edge.from);
    const to = scene.texts.get(edge.to);
    out.edges.push({ key: edge.key, from: edge.from, to: edge.to, d: curve(points, lr), points, label: clip(edge.label, 24),
                     lx: (a.x + b.x) / 2, ly: (a.y + b.y) / 2 - 4,
                     title: [`${from.label} → ${to.label}`, edge.label, edge.shape,
                             edge.count > 1 ? edge.count + " edges" : ""].filter(Boolean).join("\n") });
  }
  let left = Infinity;
  let top = Infinity;
  let right = -Infinity;
  let bottom = -Infinity;
  const grow = (x, y) => {
    left = Math.min(left, x);
    top = Math.min(top, y);
    right = Math.max(right, x);
    bottom = Math.max(bottom, y);
  };
  for (const item of [...out.nodes, ...out.boxes]) {
    grow(item.x, item.y);
    grow(item.x + item.w, item.y + item.h);
  }
  for (const edge of out.edges) for (const item of edge.points) grow(item.x, item.y);
  const bounds = Number.isFinite(left) ? { x: left, y: top, width: right - left, height: bottom - top } : { x: 0, y: 0, width: 0, height: 0 };
  out.boxes.sort((a, b) => a.depth - b.depth);
  return { ...out, bounds, groups: [...base.groups.keys()] };
}

const STYLED = ["fill", "fill-opacity", "stroke", "stroke-width", "stroke-dasharray", "stroke-opacity", "opacity",
                "font-family", "font-size", "font-weight", "paint-order", "stroke-linejoin"];

function fileName(title) {
  return (textOf(title).trim().replace(/[^\w.]+/g, "_").replace(/^_+|_+$/g, "") || "diagram") + ".svg";
}

export const Diagram = {
  name: "KDiagram",
  props: {
    graph: { type: Object, default: null },
    direction: { type: String, default: "LR" },
    height: { type: Number, default: 460 },
    collapsed: { type: Array, default: () => [] },
    title: { type: String, default: "" },
  },
  emits: ["select"],
  data() {
    return {
      dir: this.direction === "TB" ? "TB" : "LR",
      closedIds: (this.collapsed || []).map(String),
      zoom: 1,
      pan: { x: 0, y: 0 },
      drag: null,
      dragged: false,
      selected: null,
      marker: "kdiagram-arrow-" + Math.random().toString(36).slice(2, 9),
    };
  },
  computed: {
    closedSet() { return new Set(this.closedIds); },
    layout() { return layoutGraph(this.graph, this.closedSet, this.dir); },
    signature() {
      const graph = this.graph && typeof this.graph === "object" ? this.graph : {};
      const ids = list => (Array.isArray(list) ? list.map(item => (item ? textOf(item.id) : "")).join(",") : "");
      return ids(graph.nodes) + "|" + ids(graph.groups) + "|" + (Array.isArray(graph.edges) ? graph.edges.length : 0);
    },
    collapsedKey() { return (this.collapsed || []).map(String).join(","); },
    transform() { return `translate(${this.pan.x} ${this.pan.y}) scale(${this.zoom})`; },
    empty() { return !this.layout.nodes.length; },
    groupCount() { return this.layout.groups.length; },
    labeled() { return this.layout.edges.filter(edge => edge.label); },
  },
  watch: {
    signature() {
      this.closedIds = (this.collapsed || []).map(String);
      this.selected = null;
      this.later();
    },
    collapsedKey() {
      this.closedIds = (this.collapsed || []).map(String);
      this.later();
    },
    direction(value) {
      this.dir = value === "TB" ? "TB" : "LR";
      this.later();
    },
  },
  mounted() {
    this.later();
  },
  beforeUnmount() {
    this.stopListening();
  },
  methods: {
    later() {
      this.$nextTick(() => this.fit());
    },
    fit() {
      const svg = this.$refs.svg;
      const bounds = this.layout.bounds;
      if (!svg || !bounds.width || !bounds.height) {
        this.zoom = 1;
        this.pan = { x: 0, y: 0 };
        return;
      }
      const width = svg.clientWidth || svg.getBoundingClientRect().width || 800;
      const height = this.height;
      const zoom = Math.max(0.05, Math.min(1.25, (width - 32) / bounds.width, (height - 32) / bounds.height));
      this.zoom = zoom;
      this.pan = { x: (width - bounds.width * zoom) / 2 - bounds.x * zoom, y: (height - bounds.height * zoom) / 2 - bounds.y * zoom };
    },
    turn(value) {
      if (this.dir === value) return;
      this.dir = value;
      this.later();
    },
    toggle(gid) {
      if (this.dragged) return;
      const id = String(gid);
      this.closedIds = this.closedSet.has(id) ? this.closedIds.filter(item => item !== id) : [...this.closedIds, id];
      this.later();
    },
    openAll() {
      this.closedIds = [];
      this.later();
    },
    closeAll() {
      this.closedIds = [...this.layout.groups];
      this.later();
    },
    pick(node) {
      if (this.dragged) return;
      if (node.unit.collapsed) {
        this.toggle(node.unit.gid);
        return;
      }
      this.selected = node.key;
      this.$emit("select", node.unit.node);
    },
    clear() {
      if (this.dragged || this.selected === null) return;
      this.selected = null;
      this.$emit("select", null);
    },
    lit(edge) {
      return this.selected !== null && (edge.from === this.selected || edge.to === this.selected);
    },
    nodeClass(node) {
      return [kindClass(node.kind), { selected: node.key === this.selected, collapsed: node.unit.collapsed }];
    },
    wheel(event) {
      const svg = this.$refs.svg;
      if (!svg) return;
      const rect = svg.getBoundingClientRect();
      const cx = event.clientX - rect.left;
      const cy = event.clientY - rect.top;
      const zoom = Math.min(4, Math.max(0.05, this.zoom * Math.exp(-event.deltaY * 0.0015)));
      const ratio = zoom / this.zoom;
      this.pan = { x: cx - (cx - this.pan.x) * ratio, y: cy - (cy - this.pan.y) * ratio };
      this.zoom = zoom;
    },
    press(event) {
      if (event.button !== 0) return;
      this.dragged = false;
      this.drag = { x: event.clientX, y: event.clientY, pan: { ...this.pan }, moved: false };
      window.addEventListener("mousemove", this.move);
      window.addEventListener("mouseup", this.release);
    },
    move(event) {
      const drag = this.drag;
      if (!drag) return;
      const dx = event.clientX - drag.x;
      const dy = event.clientY - drag.y;
      if (!drag.moved && Math.abs(dx) + Math.abs(dy) < 4) return;
      drag.moved = true;
      this.pan = { x: drag.pan.x + dx, y: drag.pan.y + dy };
    },
    release() {
      this.stopListening();
      this.dragged = Boolean(this.drag && this.drag.moved);
      this.drag = null;
    },
    stopListening() {
      window.removeEventListener("mousemove", this.move);
      window.removeEventListener("mouseup", this.release);
    },
    download() {
      const svg = this.$refs.svg;
      const bounds = this.layout.bounds;
      if (!svg || !bounds.width) return;
      const margin = 16;
      const copy = svg.cloneNode(true);
      const originals = svg.querySelectorAll("*");
      const copies = copy.querySelectorAll("*");
      originals.forEach((element, index) => {
        const target = copies[index];
        if (!target) return;
        const computed = window.getComputedStyle(element);
        const style = STYLED.map(name => `${name}: ${computed.getPropertyValue(name)}`).join("; ");
        target.setAttribute("style", style);
      });
      const width = Math.ceil(bounds.width + 2 * margin);
      const height = Math.ceil(bounds.height + 2 * margin);
      copy.setAttribute("xmlns", "http://www.w3.org/2000/svg");
      copy.setAttribute("viewBox", `${bounds.x - margin} ${bounds.y - margin} ${width} ${height}`);
      copy.setAttribute("width", String(width));
      copy.setAttribute("height", String(height));
      copy.removeAttribute("class");
      const canvas = copy.querySelector("g.canvas");
      if (canvas) canvas.removeAttribute("transform");
      const back = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      back.setAttribute("x", String(bounds.x - margin));
      back.setAttribute("y", String(bounds.y - margin));
      back.setAttribute("width", String(width));
      back.setAttribute("height", String(height));
      back.setAttribute("fill", window.getComputedStyle(this.$el).getPropertyValue("--surface").trim() || "#ffffff");
      copy.insertBefore(back, copy.firstChild);
      const text = new XMLSerializer().serializeToString(copy);
      const url = URL.createObjectURL(new Blob([text], { type: "image/svg+xml" }));
      const link = document.createElement("a");
      link.href = url;
      link.download = fileName(this.title);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    },
  },
  template: `
    <div class="kdiagram">
      <div class="kdiagram-tools">
        <span class="muted small kdiagram-hint" v-if="!empty">drag to pan, wheel to zoom{{ groupCount ? ", click a group header to fold it" : "" }}</span>
        <span class="spacer"></span>
        <button type="button" class="tool" @click="fit" title="zoom and pan so the whole diagram is in view">fit</button>
        <button type="button" class="tool" :class="{ on: dir === 'LR' }" @click="turn('LR')" title="lay the diagram out from left to right">LR</button>
        <button type="button" class="tool" :class="{ on: dir === 'TB' }" @click="turn('TB')" title="lay the diagram out from top to bottom">TB</button>
        <button type="button" class="tool" :disabled="!groupCount" @click="openAll" title="unfold every group">expand all</button>
        <button type="button" class="tool" :disabled="!groupCount" @click="closeAll" title="fold every group">collapse all</button>
        <button type="button" class="tool" :disabled="empty" @click="download" title="download the diagram as an SVG file">svg</button>
      </div>
      <div v-if="empty" class="empty">nothing to draw</div>
      <svg v-else ref="svg" class="kdiagram-canvas" :class="{ panning: drag && drag.moved }" width="100%" :height="height"
           @wheel.prevent="wheel" @mousedown="press" @click="clear">
        <defs>
          <marker :id="marker" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" class="kdiagram-arrow"></path>
          </marker>
          <marker :id="marker + '-on'" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" class="kdiagram-arrow on"></path>
          </marker>
        </defs>
        <g class="canvas" :transform="transform">
          <g v-for="box in layout.boxes" :key="box.key" class="kdiagram-group" :class="'depth-' + Math.min(box.depth, 4)">
            <rect class="group-box" :x="box.x" :y="box.y" :width="box.w" :height="box.h" rx="8"></rect>
            <g class="group-head" @click.stop="toggle(box.gid)">
              <rect class="group-hit" :x="box.x" :y="box.y" :width="box.w" height="22" rx="8"></rect>
              <text class="group-label" :x="box.x + 10" :y="box.y + 15">{{ box.label }}<tspan v-if="box.detail" class="group-detail" dx="6">{{ box.detail }}</tspan></text>
              <text v-if="box.corner" class="group-corner" :x="box.x + box.w - 10" :y="box.y + 15" text-anchor="end">{{ box.corner }}</text>
              <title>{{ box.title }}</title>
            </g>
          </g>
          <g v-for="edge in layout.edges" :key="edge.key" class="kdiagram-edge" :class="{ on: lit(edge) }">
            <path class="edge-line" :d="edge.d" :marker-end="'url(#' + marker + (lit(edge) ? '-on' : '') + ')'"></path>
            <path class="edge-hit" :d="edge.d"></path>
            <title>{{ edge.title }}</title>
          </g>
          <text v-for="edge in labeled" :key="'label:' + edge.key" class="edge-label"
                :x="edge.lx" :y="edge.ly" text-anchor="middle">{{ edge.label }}</text>
          <g v-for="node in layout.nodes" :key="node.key" class="kdiagram-node" :class="nodeClass(node)" @click.stop="pick(node)">
            <rect class="node-box" :x="node.x" :y="node.y" :width="node.w" :height="node.h" rx="6"></rect>
            <text class="node-label" :x="node.x + 10" :y="node.y + 18">{{ node.label }}</text>
            <text v-if="node.corner" class="node-corner" :x="node.x + node.w - 8" :y="node.y + 17" text-anchor="end">{{ node.corner }}</text>
            <text v-if="node.detail" class="node-detail" :x="node.x + 10" :y="node.y + 32">{{ node.detail }}</text>
            <text v-if="node.shape" class="node-shape" :x="node.x + 10" :y="node.y + (node.detail ? 46 : 32)">{{ node.shape }}</text>
            <title>{{ node.title }}</title>
          </g>
        </g>
      </svg>
    </div>`,
};

export default Diagram;
