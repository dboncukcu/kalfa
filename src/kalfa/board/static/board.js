const SET_COLORS = { train: "#2f6fdd", val: "#e8722a", test: "#17a673", calib: "#8e5bd8" };
const PALETTE = ["#2f6fdd", "#e8722a", "#17a673", "#8e5bd8", "#d6437a", "#c99a06", "#1c9aa8", "#7a7a7a"];
const SKIP_KEYS = new Set(["turn", "global_step", "step", "seconds", "rules"]);
const KIND_COLORS = { input: "#8a8f98", torch: "#2f6fdd", lego: "#17a673", model: "#8e5bd8", output: "#e0a106",
                      loss: "#d8433c", optimizer: "#e8722a", source: "#8a8f98", transform: "#17a673", split: "#2f6fdd",
                      frames: "#8e5bd8", fit: "#e0a106", feed: "#d6437a", loaders: "#e8722a" };
const KIND_NAMES = { input: "input wire", torch: "torch layer", lego: "kalfa layer", model: "model", output: "output wire",
                     loss: "loss", optimizer: "optimizer", source: "source", transform: "transform", split: "split",
                     frames: "frame transforms", fit: "fit on train", feed: "feed", loaders: "loaders" };
const SET_ORDER = ["train", "valid", "test"];

function orderedSets(names) {
  return [...names].sort((a, b) => (SET_ORDER.indexOf(a) + 1 || 99) - (SET_ORDER.indexOf(b) + 1 || 99) || a.localeCompare(b));
}

function wrapWords(text, width) {
  const lines = [];
  let current = "";
  for (const word of text.split(", ")) {
    if (current && current.length + word.length + 2 > width) { lines.push(current); current = word; }
    else current = current ? `${current}, ${word}` : word;
  }
  if (current) lines.push(current);
  return lines;
}

function downloadSvg(svg, title, width, height, background) {
  const copy = svg.cloneNode(true);
  const originals = svg.querySelectorAll("*");
  copy.querySelectorAll("*").forEach((node, index) => {
    const style = getComputedStyle(originals[index]);
    for (const key of ["fill", "stroke", "stroke-width", "stroke-dasharray", "font-size", "font-family", "opacity", "font-weight"]) {
      if (style[key]) node.setAttribute(key, style[key]);
    }
  });
  copy.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  if (!copy.getAttribute("viewBox")) copy.setAttribute("viewBox", `0 0 ${width} ${height}`);
  copy.setAttribute("width", width * 2);
  copy.setAttribute("height", height * 2);
  const blob = new Blob([new XMLSerializer().serializeToString(copy)], { type: "image/svg+xml;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const image = new Image();
  const name = (title || "chart").replace(/[^\w.-]+/g, "_");
  image.onload = () => {
    const canvas = document.createElement("canvas");
    canvas.width = width * 2; canvas.height = height * 2;
    const context = canvas.getContext("2d");
    context.fillStyle = background && background !== "rgba(0, 0, 0, 0)" ? background : "#ffffff";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.drawImage(image, 0, 0);
    URL.revokeObjectURL(url);
    canvas.toBlob(png => {
      const link = document.createElement("a");
      link.href = URL.createObjectURL(png);
      link.download = `${name}.png`;
      link.click();
      setTimeout(() => URL.revokeObjectURL(link.href), 1000);
    }, "image/png");
  };
  image.src = url;
}

async function api(route, params) {
  const query = new URLSearchParams(params || {}).toString();
  const response = await fetch(route + (query ? "?" + query : ""));
  if (!response.ok) return null;
  return response.json();
}

function fmt(value) {
  if (value === null || value === undefined || value === "") return "";
  if (typeof value !== "number") return String(value);
  if (!Number.isFinite(value)) return String(value);
  if (Number.isInteger(value) && Math.abs(value) < 1e9) return String(value);
  const size = Math.abs(value);
  if (size !== 0 && (size >= 1e5 || size < 1e-3)) return value.toExponential(3);
  return String(Number(value.toPrecision(5)));
}

function count(value) {
  return typeof value === "number" ? value.toLocaleString("en-US") : fmt(value);
}

function ms(value) {
  if (typeof value !== "number") return "";
  if (value < 1000) return `${Math.round(value)} ms`;
  if (value < 60000) return `${(value / 1000).toFixed(1)} s`;
  const minutes = Math.floor(value / 60000);
  const seconds = Math.round((value - minutes * 60000) / 1000);
  return minutes < 60 ? `${minutes} min ${seconds} s` : `${Math.floor(minutes / 60)} h ${minutes % 60} min`;
}

function ago(iso) {
  const when = Date.parse(iso);
  if (Number.isNaN(when)) return iso || "";
  const seconds = Math.max(0, (Date.now() - when) / 1000);
  if (seconds < 60) return "just now";
  if (seconds < 3600) return `${Math.round(seconds / 60)} min ago`;
  if (seconds < 86400) return `${Math.round(seconds / 3600)} h ago`;
  return `${Math.round(seconds / 86400)} d ago`;
}

function clock(iso) {
  if (!iso) return "";
  const at = iso.indexOf("T");
  return at >= 0 ? iso.slice(at + 1, at + 13) : iso;
}

function splitKey(key) {
  const slash = key.indexOf("/");
  if (slash < 0) return { set: "", name: key };
  return { set: key.slice(0, slash), name: key.slice(slash + 1) };
}

function setColor(set, position) {
  return SET_COLORS[set] || PALETTE[(position || 0) % PALETTE.length];
}

function downsample(points, limit) {
  if (points.length <= limit) return points;
  const size = points.length / (limit / 2);
  const kept = [];
  for (let start = 0; start < points.length; start += size) {
    const bucket = points.slice(Math.floor(start), Math.floor(start + size));
    if (!bucket.length) continue;
    let low = bucket[0], high = bucket[0];
    for (const point of bucket) {
      if (point[1] < low[1]) low = point;
      if (point[1] > high[1]) high = point;
    }
    kept.push(...(low[0] <= high[0] ? [low, high] : [high, low]).filter((point, index, pair) => index === 0 || point !== pair[0]));
  }
  return kept;
}

function niceStep(span, target) {
  const rough = span / Math.max(target, 1);
  const power = Math.pow(10, Math.floor(Math.log10(rough)));
  for (const factor of [1, 2, 2.5, 5, 10]) {
    if (rough <= factor * power) return factor * power;
  }
  return 10 * power;
}

function ticksOf(min, max, target) {
  if (!(max > min)) return [min];
  const step = niceStep(max - min, target);
  const found = [];
  for (let value = Math.ceil(min / step) * step; value <= max + step / 1e6; value += step) found.push(Number(value.toFixed(10)));
  return found;
}

const Chart = {
  props: { lines: { type: Array, default: () => [] }, title: String, xlabel: String, logy: Boolean, marks: { type: Array, default: () => [] } },
  data() { return { hover: null, hidden: {}, width: 720, height: 240, pad: { left: 58, right: 16, top: 12, bottom: 30 } }; },
  computed: {
    visible() {
      return this.lines.filter(line => !this.hidden[line.name]).map(line => ({
        ...line, points: line.points.filter(point => Number.isFinite(point[1]) && (!this.logy || point[1] > 0)) })).filter(line => line.points.length);
    },
    domain() {
      const xs = this.visible.flatMap(line => line.points.map(point => point[0]));
      const ys = this.visible.flatMap(line => line.points.map(point => this.logy ? Math.log10(point[1]) : point[1]));
      if (!xs.length) return null;
      let xmin = Math.min(...xs), xmax = Math.max(...xs), ymin = Math.min(...ys), ymax = Math.max(...ys);
      if (xmin === xmax) { xmin -= 0.5; xmax += 0.5; }
      if (ymin === ymax) { ymin -= Math.abs(ymin) * 0.05 || 0.5; ymax += Math.abs(ymax) * 0.05 || 0.5; }
      const margin = (ymax - ymin) * 0.06;
      return { xmin, xmax, ymin: ymin - margin, ymax: ymax + margin };
    },
    scaled() {
      if (!this.domain) return [];
      return this.visible.map(line => ({
        name: line.name, color: line.color,
        path: line.points.map((point, index) => `${index ? "L" : "M"}${this.sx(point[0]).toFixed(1)} ${this.sy(point[1]).toFixed(1)}`).join(" ") }));
    },
    yticks() {
      if (!this.domain) return [];
      const { ymin, ymax } = this.domain;
      if (this.logy) {
        const found = [];
        for (let power = Math.ceil(ymin); power <= Math.floor(ymax); power++) found.push({ pos: this.sy(Math.pow(10, power)), label: fmt(Math.pow(10, power)) });
        if (found.length >= 2) return found;
      }
      return ticksOf(ymin, ymax, 5).map(value => ({ pos: this.sy(this.logy ? Math.pow(10, value) : value), label: fmt(this.logy ? Math.pow(10, value) : value) }));
    },
    xticks() {
      if (!this.domain) return [];
      return ticksOf(this.domain.xmin, this.domain.xmax, 6).map(value => ({ pos: this.sx(value), label: fmt(value) }));
    },
    placedMarks() {
      if (!this.domain) return [];
      return this.marks.filter(mark => mark >= this.domain.xmin && mark <= this.domain.xmax).map(mark => ({ x: this.sx(mark) }));
    },
    tooltip() {
      if (this.hover === null || !this.domain) return null;
      const dots = [];
      let label = null, anchor = null;
      for (const line of this.visible) {
        let best = null;
        for (const point of line.points) {
          if (best === null || Math.abs(point[0] - this.hover) < Math.abs(best[0] - this.hover)) best = point;
        }
        if (best === null) continue;
        if (anchor === null || Math.abs(best[0] - this.hover) < Math.abs(anchor - this.hover)) { anchor = best[0]; label = fmt(best[0]); }
        dots.push({ name: line.name, color: line.color, y: this.sy(best[1]), text: fmt(best[1]), x: best[0] });
      }
      if (anchor === null) return null;
      return { x: this.sx(anchor), label, dots: dots.filter(dot => dot.x === anchor) };
    },
    tooltipLeft() {
      return this.tooltip ? this.tooltip.x / this.width * 100 : 0;
    },
  },
  methods: {
    save(event) {
      const svg = event.currentTarget.closest(".chart").querySelector("svg");
      downloadSvg(svg, this.title, this.width, this.height, getComputedStyle(svg.closest(".card") || svg).backgroundColor);
    },
    sx(x) { return this.pad.left + (x - this.domain.xmin) / (this.domain.xmax - this.domain.xmin) * (this.width - this.pad.left - this.pad.right); },
    sy(y) {
      const value = this.logy ? Math.log10(y) : y;
      return this.height - this.pad.bottom - (value - this.domain.ymin) / (this.domain.ymax - this.domain.ymin) * (this.height - this.pad.top - this.pad.bottom);
    },
    onMove(event) {
      if (!this.domain) return;
      const box = event.currentTarget.getBoundingClientRect();
      const x = (event.clientX - box.left) / box.width * this.width;
      this.hover = this.domain.xmin + (x - this.pad.left) / (this.width - this.pad.left - this.pad.right) * (this.domain.xmax - this.domain.xmin);
    },
    toggle(name) { this.hidden = { ...this.hidden, [name]: !this.hidden[name] }; },
  },
  template: "#chart-template",
};

const Diagram = {
  props: { boxes: { type: Array, default: () => [] }, arrows: { type: Array, default: () => [] }, title: String },
  data() { return { hover: null, marker: `arrow-${Math.random().toString(36).slice(2, 8)}` }; },
  computed: {
    layout() {
      const CHAR = 6.7, LINE = 15, PAD_X = 14, PAD_Y = 10, GAP_X = 72, GAP_Y = 24, EDGE = 20;
      const columns = {};
      for (const box of this.boxes) (columns[box.column] = columns[box.column] || []).push(box);
      const indices = Object.keys(columns).map(Number).sort((a, b) => a - b);
      if (!indices.length) return { boxes: [], edges: [], width: 200, height: 60 };
      const widths = {}, heights = {};
      for (const index of indices) {
        const items = columns[index];
        const chars = Math.max(4, ...items.flatMap(box => box.lines.map(line => Math.min(line.length, 46))));
        widths[index] = Math.min(340, Math.max(96, chars * CHAR + 2 * PAD_X));
        heights[index] = Math.max(...items.map(box => box.lines.length)) * LINE + 2 * PAD_Y;
      }
      const height = Math.max(...Object.values(heights));
      const pitch = height + GAP_Y;
      const rows = Math.max(...indices.map(index => columns[index].length));
      const lefts = {};
      let cursor = EDGE;
      for (const index of indices) { lefts[index] = cursor; cursor += widths[index] + GAP_X; }
      const width = cursor - GAP_X + EDGE;
      const middle = EDGE + rows * pitch / 2;
      const placed = {};
      for (const index of indices) {
        const items = columns[index];
        for (const box of items) {
          const y = middle + (box.row - (items.length - 1) / 2) * pitch - height / 2;
          placed[box.name] = { ...box, x: lefts[index], y, width: widths[index], height,
                               shown: box.lines.map(line => line.length > 46 ? line.slice(0, 45) + "…" : line) };
        }
      }
      const edges = [];
      for (const [source, target, label, dashed] of this.arrows) {
        const from = placed[source], to = placed[target];
        if (!from || !to) continue;
        const x1 = from.x + from.width, y1 = from.y + from.height / 2, x2 = to.x, y2 = to.y + to.height / 2;
        const bend = Math.max(28, (x2 - x1) / 2);
        const shown = label && from.lines[0] !== label ? label : "";
        edges.push({ key: `${source}->${target}`, path: `M${x1} ${y1} C${x1 + bend} ${y1}, ${x2 - bend} ${y2}, ${x2} ${y2}`,
                     label: shown, lx: (x1 + x2) / 2, ly: (y1 + y2) / 2 - 6, dashed: !!dashed });
      }
      return { boxes: Object.values(placed), edges, width, height: EDGE + rows * pitch + EDGE };
    },
    kinds() { return [...new Set(this.boxes.map(box => box.kind))]; },
    tooltipStyle() {
      if (!this.hover) return {};
      return { left: `${this.hover.x + this.hover.width + 10}px`, top: `${this.hover.y}px` };
    },
  },
  methods: {
    color(kind) { return KIND_COLORS[kind] || "#7a7a7a"; },
    kindName(kind) { return KIND_NAMES[kind] || kind; },
    pick(box) { this.$emit("pick", box); },
    save(event) {
      const svg = event.currentTarget.closest(".diagram").querySelector("svg");
      downloadSvg(svg, this.title, this.layout.width, this.layout.height, getComputedStyle(svg.closest(".card") || svg).backgroundColor);
    },
  },
  template: "#diagram-template",
};

const app = Vue.createApp({
  components: { chart: Chart, diagram: Diagram },
  data() {
    return {
      tree: { root: "", groups: {} }, filter: "", collapsed: {}, refreshed: "",
      current: { path: null, kind: null }, record: null, history: { lines: [], offset: 0 }, steps: { lines: [], offset: 0 },
      tab: "overview", logy: false, metricFilter: "", logs: { name: "stdout.txt", lines: [], total: 0 }, describeText: null,
      texts: {}, modelPick: null, showModuleText: false,
      sweep: null, selected: [], overlay: {}, diff: null, sortKey: null, sortDesc: false, lightbox: null,
    };
  },
  computed: {
    groups() {
      const needle = this.filter.toLowerCase();
      return Object.entries(this.tree.groups).map(([name, entries]) => ({
        name, entries: entries.filter(entry => !needle || entry.name.toLowerCase().includes(needle) || entry.path.toLowerCase().includes(needle)) }))
        .filter(group => group.entries.length);
    },
    live() {
      if (!this.current.path) return false;
      if (this.current.kind === "sweep") return !!(this.sweep && this.sweep.points.some(point => this.stateOf(point) === "running"));
      return this.state === "running";
    },
    state() { return this.record ? this.stateOf(this.record) : "pending"; },
    sweepState() {
      if (!this.sweep) return "pending";
      const states = this.sweep.points.map(point => this.stateOf(point));
      if (states.includes("running")) return "running";
      if (states.length && states.every(state => state === "finished")) return "finished";
      if (states.includes("failed")) return "failed";
      return "pending";
    },
    progress() {
      if (!this.sweep) return 0;
      const total = this.sweep.manifest.total || this.sweep.points.length || 1;
      return Math.min(100, 100 * this.sweep.points.filter(point => this.stateOf(point) === "finished").length / total);
    },
    params() { return Object.entries((this.record && this.record.manifest && this.record.manifest.params) || {}); },
    turns() { return this.history.lines.length; },
    elapsed() {
      const manifest = this.record && this.record.manifest;
      const seen = this.record && this.record.status && this.record.status.last_seen;
      if (!manifest || !manifest.started || !seen) return "";
      const span = Date.parse(seen) - Date.parse(manifest.started);
      return Number.isFinite(span) && span > 0 ? `ran ${ms(span)}` : "";
    },
    pointObjective() {
      const manifest = this.record && this.record.manifest;
      if (!manifest || manifest.kind !== "point" || this.record.sweep) return "";
      return manifest.objective ? `${manifest.objective.monitor} ${manifest.objective.mode || "min"}, not scored yet` : "";
    },
    tabNames() { return ["overview", "curves", "steps", "model", "data", "plots", "samples", "config", "notes", "events", "logs", "describe"]; },
    architectureModels() { return Object.keys((this.record && this.record.architecture && this.record.architecture.models) || {}); },
    moduleText() { return this.record && this.record.plots.includes("architecture_text.txt") ? this.texts["architecture_text.txt"] || "reading" : null; },
    currentArchitecture() {
      const models = (this.record && this.record.architecture && this.record.architecture.models) || {};
      const label = this.modelPick && models[this.modelPick] ? this.modelPick : this.architectureModels[0];
      return label ? { label, ...models[label] } : null;
    },
    pipeline() {
      const data = this.record && this.record.data;
      if (!data || !(data.stages || []).length) return null;
      const boxes = [], arrows = [];
      const stages = data.stages;
      let column = 0, previous = "source";
      boxes.push({ name: "source", kind: "source", column, row: 0, lines: ["source", `${count(stages[0].rows)} rows`, `${stages[0].columns} columns`] });
      for (const stage of stages.slice(1)) {
        column += 1;
        const added = stage.added || [], removed = stage.removed || [];
        const change = [added.length ? `+${added.length}` : "", removed.length ? `−${removed.length}` : ""].filter(Boolean).join(" ");
        boxes.push({ name: stage.stage, kind: "transform", column, row: 0, note: [...added.map(name => `+ ${name}`), ...removed.map(name => `− ${name}`)],
                     lines: [stage.stage, `${count(stage.rows)} rows`, `${stage.columns} columns${change ? "  " + change : ""}`] });
        arrows.push([previous, stage.stage, "", false]);
        previous = stage.stage;
      }
      const sets = orderedSets(Object.keys(data.split || {}));
      if (!sets.length) return { boxes, arrows };
      column += 1;
      sets.forEach((set, row) => { boxes.push({ name: `split:${set}`, kind: "split", column, row, lines: [set, `${count(data.split[set])} rows`] }); arrows.push([previous, `split:${set}`, "", false]); });
      let last = set => `split:${set}`;
      const after = data.after_set_transforms || {};
      if (sets.some(set => after[set] !== undefined && after[set] !== data.split[set])) {
        column += 1;
        sets.forEach((set, row) => { boxes.push({ name: `after:${set}`, kind: "transform", column, row, lines: [`${set} transforms`, `${count(after[set])} rows`] }); arrows.push([`split:${set}`, `after:${set}`, "", false]); });
        last = set => `after:${set}`;
      }
      const fit = data.fit || {};
      column += 1;
      const fitted = Object.entries(fit.preprocessors || {}).map(([name, columns]) => `${name} ${columns}`).join(", ");
      const lines = ["fit on train", ...(data.frames || []).length ? [`frames ${data.frames.join(", ")}`] : [], ...wrapWords(fitted || "no preprocessors", 30),
                     `${fit.features || 0} features, ${(fit.targets || []).length} targets`];
      boxes.push({ name: "fit", kind: "fit", column, row: 0, lines, note: (fit.targets || []).length ? ["targets", ...fit.targets] : [] });
      sets.forEach(set => arrows.push([last(set), "fit", "", set !== "train"]));
      column += 1;
      sets.forEach((set, row) => { const entry = (data.sets || {})[set] || {}; boxes.push({ name: `feed:${set}`, kind: "feed", column, row, lines: [set, `${count(entry.rows)} rows`, `${entry.features ?? "?"} features`] }); arrows.push(["fit", `feed:${set}`, "", false]); });
      column += 1;
      sets.forEach((set, row) => { const entry = (data.loaders || {})[set] || {}; boxes.push({ name: `loader:${set}`, kind: "loaders", column, row, lines: [`${set} loader`, `${count(entry.batches)} x ${entry.size}`] }); arrows.push([`feed:${set}`, `loader:${set}`, "", false]); });
      return { boxes, arrows };
    },
    series() {
      const found = {};
      for (const line of this.history.lines) {
        for (const [key, value] of Object.entries(line)) {
          if (SKIP_KEYS.has(key) || typeof value !== "number") continue;
          (found[key] = found[key] || []).push([line.turn, value]);
        }
      }
      return found;
    },
    metricCharts() {
      const needle = this.metricFilter.toLowerCase();
      const byName = {};
      for (const [key, points] of Object.entries(this.series)) {
        const { set, name } = splitKey(key);
        if (set === "lr") continue;
        if (needle && !key.toLowerCase().includes(needle)) continue;
        (byName[name] = byName[name] || []).push({ name: set || key, color: null, points });
      }
      return Object.entries(byName).map(([name, lines]) => ({
        name, lines: lines.map((line, index) => ({ ...line, color: setColor(line.name, index) })) }));
    },
    rateLines() {
      return Object.entries(this.series).filter(([key]) => key.startsWith("lr/")).map(([key, points], index) => ({ name: key.slice(3), color: PALETTE[index % PALETTE.length], points }));
    },
    rateRows() {
      return Object.entries(this.series).filter(([key]) => key.startsWith("lr/")).map(([key, points]) => ({ key, last: points[points.length - 1][1] }));
    },
    latest() {
      return Object.entries(this.series).filter(([key]) => !key.startsWith("lr/")).map(([key, points]) => {
        const { set, name } = splitKey(key);
        let low = points[0], high = points[0];
        for (const point of points) { if (point[1] < low[1]) low = point; if (point[1] > high[1]) high = point; }
        const tail = points.slice(-40);
        const min = Math.min(...tail.map(point => point[1])), max = Math.max(...tail.map(point => point[1]));
        const spark = tail.map((point, index) => `${(index / Math.max(tail.length - 1, 1) * 118 + 1).toFixed(1)},${(22 - (max > min ? (point[1] - min) / (max - min) * 20 : 10) + 1).toFixed(1)}`).join(" ");
        return { key, set, name, last: points[points.length - 1][1], min: low[1], minTurn: low[0], max: high[1], maxTurn: high[0], spark };
      });
    },
    rulesFired() {
      return this.history.lines.filter(line => (line.rules || []).length).map(line => `turn ${line.turn}: ${line.rules.join(", ")}`);
    },
    ruleMarks() { return this.history.lines.filter(line => (line.rules || []).length).map(line => line.turn); },
    stepCharts() {
      const found = {};
      for (const line of this.steps.lines) {
        for (const [key, value] of Object.entries(line)) {
          if (SKIP_KEYS.has(key) || typeof value !== "number") continue;
          (found[key] = found[key] || []).push([line.step, value]);
        }
      }
      return Object.entries(found).map(([key, points], index) => ({ name: key, lines: [{ name: key, color: PALETTE[index % PALETTE.length], points: downsample(points, 2000) }] }));
    },
    turnMarks() {
      const marks = [];
      let previous = null;
      for (const line of this.steps.lines) {
        if (previous !== null && line.turn !== previous) marks.push(line.step);
        previous = line.turn;
      }
      return marks.length <= 60 ? marks : [];
    },
    gallery() { return this.record ? (this.tab === "plots" ? this.record.plots : this.record.samples) : []; },
    images() { return this.gallery.filter(name => /\.(png|jpe?g|gif|svg|webp)$/i.test(name)); },
    textFiles() { return this.gallery.filter(name => /\.(txt|md|json|csv)$/i.test(name)); },
    dataStages() { return (this.record && this.record.data && this.record.data.stages) || []; },
    runNodes() {
      const tree = this.record && this.record.run && this.record.run.tree;
      if (!tree) return [];
      const total = tree.ms || 1;
      const rows = [];
      const walk = (node, depth) => {
        rows.push({ path: node.path, node: node.node, ms: node.ms, status: node.status, depth, share: Math.max(0.5, 100 * (node.ms || 0) / total) });
        for (const child of node.nodes || []) walk(child, depth + 1);
      };
      walk(tree, 0);
      return rows;
    },
    pointKeys() {
      return this.sweep ? Array.from(new Set(this.sweep.points.flatMap(point => Object.keys(point.values || {})))) : [];
    },
    sortedPoints() {
      if (!this.sweep) return [];
      const points = [...this.sweep.points];
      if (!this.sortKey) return points;
      const value = point => {
        if (this.sortKey === "id") return point.id;
        if (this.sortKey === "turns") return point.turns;
        if (this.sortKey === "objective") return point.objective ? point.objective.value : Infinity;
        return point.values[this.sortKey];
      };
      points.sort((first, second) => {
        const a = value(first), b = value(second);
        if (a === b) return 0;
        if (a === undefined || a === null) return 1;
        if (b === undefined || b === null) return -1;
        return (a < b ? -1 : 1) * (this.sortDesc ? -1 : 1);
      });
      return points;
    },
    overlayLines() {
      const monitor = this.sweep && this.sweep.objective.monitor;
      if (!monitor) return [];
      return this.selected.filter(path => this.overlay[path]).map((path, index) => ({
        name: path.split("/").pop(), color: PALETTE[index % PALETTE.length],
        points: this.overlay[path].filter(line => typeof line[monitor] === "number").map(line => [line.turn, line[monitor]]) }));
    },
  },
  methods: {
    fmt, count, ms, ago, clock, setColor,
    stateOf(entry) { return (entry.status && entry.status.state) || "pending"; },
    shortUri(uri) { return typeof uri === "string" ? uri.split("/").pop() : ""; },
    text(value) { return typeof value === "object" && value !== null ? JSON.stringify(value) : String(value); },
    entries(mapping, skip) { return Object.entries(mapping || {}).filter(([key]) => !(skip || []).includes(key)); },
    fileUrl(kind, name) { return `/file?path=${encodeURIComponent(`${this.current.path}/${kind}/${name}`)}`; },
    tabCount(name) {
      if (!this.record) return 0;
      if (name === "model") return this.architectureModels.length;
      if (name === "plots") return this.record.plots.length;
      if (name === "samples") return this.record.samples.length;
      if (name === "events") return this.record.events.length;
      return 0;
    },
    diffClass(line) {
      if (line.startsWith("+++") || line.startsWith("---")) return "hunk";
      if (line.startsWith("@@")) return "hunk";
      if (line.startsWith("+")) return "add";
      if (line.startsWith("-")) return "del";
      return "";
    },
    toggleGroup(name) { this.collapsed = { ...this.collapsed, [name]: !this.collapsed[name] }; },
    pickModel(box) {
      if (box.kind !== "model") return;
      const label = (box.lines[1] || "").replace(/^model /, "");
      if (this.architectureModels.includes(label)) this.modelPick = label;
    },
    async loadTree() {
      const tree = await api("/api/tree");
      if (tree) this.tree = tree;
      this.refreshed = new Date().toLocaleTimeString();
      if (!this.record || this.current.kind === "sweep") return;
      const entry = Object.values(this.tree.groups).flat().find(item => item.path === this.current.path);
      if (entry && this.stateOf(entry) !== this.state) await this.loadRecord();
    },
    async loadTexts() {
      const found = {};
      const names = this.record ? this.record.plots.filter(name => /\.(txt|md|json|csv)$/i.test(name)) : [];
      for (const name of names) {
        const response = await fetch(this.fileUrl("plots", name));
        found[name] = response.ok ? await response.text() : "unreadable";
      }
      this.texts = found;
    },
    async open(entry) { await this.openPath(entry.path, entry.kind); },
    async openPath(path, kind) {
      this.current = { path, kind };
      this.record = null; this.sweep = null; this.history = { lines: [], offset: 0 }; this.steps = { lines: [], offset: 0 };
      this.selected = []; this.overlay = {}; this.diff = null; this.describeText = null; this.lightbox = null;
      this.logs = { name: "stdout.txt", lines: [], total: 0 }; this.modelPick = null;
      if (kind === "sweep") { this.tab = "overview"; await this.loadSweep(); return; }
      if (["plots", "samples", "describe"].includes(this.tab)) this.tab = "overview";
      await this.loadRecord();
    },
    async loadRecord() {
      const record = await api("/api/record", { path: this.current.path });
      if (!record) return;
      this.record = record;
      await this.loadHistory();
      if (this.tab === "steps") await this.loadSteps();
      if (this.tab === "logs") await this.loadLogs(this.logs.name);
      if (this.tab === "plots" || this.tab === "model") await this.loadTexts();
    },
    async loadHistory() {
      const found = await api("/api/history", { path: this.current.path, offset: this.history.offset });
      if (found) this.history = { lines: this.history.lines.concat(found.lines), offset: found.offset };
    },
    async loadSteps() {
      const found = await api("/api/steps", { path: this.current.path, offset: this.steps.offset });
      if (found) this.steps = { lines: this.steps.lines.concat(found.lines), offset: found.offset };
    },
    async loadLogs(name) {
      const found = await api("/api/tail", { path: this.current.path, name, lines: 300 });
      if (found) this.logs = { name, lines: found.lines, total: found.total || 0 };
    },
    async switchTab(name) {
      this.tab = name;
      if (name === "steps" && !this.steps.lines.length) await this.loadSteps();
      if (name === "plots" || name === "model") await this.loadTexts();
      if (name === "logs") await this.loadLogs(this.record.logs.includes(this.logs.name) ? this.logs.name : (this.record.logs[0] || "stdout.txt"));
      if (name === "describe" && !this.describeText) {
        const found = await api("/api/describe", { path: this.current.path });
        this.describeText = found ? found.text : "the record cannot be described";
      }
    },
    async loadSweep() {
      const sweep = await api("/api/sweep", { path: this.current.path });
      if (sweep) this.sweep = sweep;
      await this.loadOverlay();
    },
    async toggleSelected(path) {
      this.selected = this.selected.includes(path) ? this.selected.filter(item => item !== path) : [...this.selected, path];
      await this.loadOverlay();
      await this.loadDiff();
    },
    async loadOverlay() {
      const found = {};
      for (const path of this.selected) {
        const history = await api("/api/history", { path });
        if (history) found[path] = history.lines;
      }
      this.overlay = found;
    },
    async loadDiff() {
      if (this.selected.length < 2) { this.diff = null; return; }
      const found = await api("/api/diff", { a: this.selected[0], b: this.selected[1] });
      this.diff = found ? (found.diff.length ? found.diff : ["no difference"]) : ["no resolved.yaml to compare"];
    },
    sortBy(key) {
      if (this.sortKey === key) this.sortDesc = !this.sortDesc; else { this.sortKey = key; this.sortDesc = false; }
    },
    stepLightbox(direction) {
      const names = this.gallery;
      const index = names.indexOf(this.lightbox);
      if (index < 0) return;
      this.lightbox = names[(index + direction + names.length) % names.length];
    },
    async copy(text) {
      try { await navigator.clipboard.writeText(text || ""); } catch (error) { console.warn("clipboard unavailable", error); }
    },
    async tick() {
      if (!this.current.path) return;
      if (this.current.kind === "sweep") { await this.loadSweep(); return; }
      if (!["running", "pending"].includes(this.state)) return;
      await this.loadRecord();
      if (this.tab === "steps") await this.loadSteps();
    },
  },
  mounted() {
    this.loadTree();
    setInterval(() => this.tick(), 3000);
    setInterval(() => this.loadTree(), 10000);
    window.addEventListener("keydown", event => {
      if (event.key === "Escape") this.lightbox = null;
      if (this.lightbox && event.key === "ArrowRight") this.stepLightbox(1);
      if (this.lightbox && event.key === "ArrowLeft") this.stepLightbox(-1);
    });
  },
});

app.mount("#app");
