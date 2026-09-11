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
const TABS = ["monitor", "overview", "curves", "steps", "model", "data", "predictions", "prep", "plots", "samples", "config", "notes", "files", "timeline", "events", "logs", "describe"];
const PAGES = ["table", "compare"];
const IMAGE = /\.(png|jpe?g|gif|svg|webp)$/i;
const TEXT = /\.(txt|md|json|csv|yaml|yml)$/i;

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

function shortUri(uri) {
  return typeof uri === "string" ? uri.split("/").filter(Boolean).pop() || uri : "";
}

function paramsText(params) {
  return Object.entries(params || {}).map(([key, value]) => `${key}=${typeof value === "object" && value !== null ? JSON.stringify(value) : value}`).join(" ");
}

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
    const pair = low[0] <= high[0] ? [low, high] : [high, low];
    kept.push(pair[0]);
    if (pair[1] !== pair[0]) kept.push(pair[1]);
  }
  return kept;
}

function human(size) {
  if (typeof size !== "number") return "";
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  if (size < 1024 * 1024 * 1024) return `${(size / 1024 / 1024).toFixed(1)} MB`;
  return `${(size / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

function darkMode() {
  return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
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
    canvas.width = width * 2;
    canvas.height = height * 2;
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

function parseHash(hash) {
  const text = (hash || "").replace(/^#/, "");
  const question = text.indexOf("?");
  const path = decodeURIComponent((question < 0 ? text : text.slice(0, question)).replace(/^\/+/, "").replace(/\/+$/, ""));
  const params = {};
  if (question >= 0) {
    for (const [key, value] of new URLSearchParams(text.slice(question + 1))) params[key] = value;
  }
  return { path, params };
}

function buildHash(path, params) {
  const query = new URLSearchParams();
  for (const [key, value] of Object.entries(params || {})) {
    if (value !== null && value !== undefined && value !== "" && value !== false) query.set(key, String(value));
  }
  const text = query.toString();
  return `#/${path.split("/").map(encodeURIComponent).join("/")}${text ? "?" + text : ""}`;
}

function chartOptions(view) {
  const dark = darkMode();
  const kind = view.kind || "line";
  const series = view.lines.map(line => ({
    name: line.name,
    data: kind === "bar" ? line.points.map(point => ({ x: String(point[0]), y: point[1] }))
      : line.points.filter(point => Number.isFinite(point[1]) && (!view.logy || point[1] > 0)).map(point => [point[0], point[1]]) }));
  return {
    chart: { type: kind, height: view.height, background: "transparent", fontFamily: "inherit", foreColor: dark ? "#b7bcc4" : "#4a4f57",
             animations: { enabled: false }, zoom: { enabled: kind !== "bar", type: kind === "scatter" ? "xy" : "x", autoScaleYaxis: kind !== "scatter" },
             toolbar: { show: true, offsetY: -4, tools: { download: true, selection: true, zoom: true, zoomin: true, zoomout: true, pan: true, reset: true } } },
    series,
    colors: view.lines.map(line => line.color),
    stroke: { width: kind === "scatter" ? 0 : 1.6, curve: "straight", dashArray: view.lines.map(line => line.dashed ? 5 : 0) },
    markers: { size: kind === "scatter" ? 3 : 0, hover: { size: 4 } },
    plotOptions: { bar: { columnWidth: "90%" } },
    dataLabels: { enabled: false },
    legend: { position: "top", horizontalAlign: "left", showForSingleSeries: true, onItemClick: { toggleDataSeries: true } },
    grid: { borderColor: dark ? "#2d3238" : "#e3e6ea" },
    xaxis: kind === "bar" ? { type: "category", title: { text: view.xlabel || "" }, tickAmount: 10, labels: { rotate: 0 } }
      : { type: "numeric", title: { text: view.xlabel || "" }, tickAmount: 8, labels: { formatter: value => fmt(Number(value)) }, tooltip: { enabled: false } },
    yaxis: { logarithmic: !!view.logy, title: { text: view.ylabel || "" }, labels: { formatter: value => fmt(Number(value)) } },
    tooltip: { shared: kind !== "scatter", intersect: kind === "scatter", theme: dark ? "dark" : "light",
               x: { formatter: value => `${view.xlabel || "x"} ${fmt(Number(value))}` }, y: { formatter: value => fmt(Number(value)) } },
    annotations: { xaxis: (view.marks || []).map(mark => ({ x: mark, borderColor: dark ? "#4a515a" : "#cfd4da", strokeDashArray: 3 })) },
    theme: { mode: dark ? "dark" : "light" },
  };
}

const Chart = {
  props: { lines: { type: Array, default: () => [] }, title: String, xlabel: String, ylabel: String, logy: Boolean,
           marks: { type: Array, default: () => [] }, height: { type: Number, default: 260 }, expand: String, kind: { type: String, default: "line" } },
  data() { return { chart: null }; },
  computed: { options() { return chartOptions(this); } },
  watch: {
    options() { if (this.chart) this.chart.updateOptions(this.options, false, false); },
  },
  mounted() {
    this.chart = new ApexCharts(this.$refs.host, this.options);
    this.chart.render();
  },
  beforeUnmount() {
    if (this.chart) this.chart.destroy();
    this.chart = null;
  },
  template: `
    <div class="chart">
      <div class="chart-head" v-if="title || expand">
        <span class="chart-title">{{ title }}</span>
        <a v-if="expand" class="small chart-save" :href="expand" title="open this chart in a large view">expand</a>
      </div>
      <div ref="host"></div>
    </div>`,
};

const Spark = {
  props: { points: { type: Array, default: () => [] }, name: String, color: String, xlabel: { type: String, default: "turn" } },
  data() { return { hover: null }; },
  computed: {
    geometry() {
      const values = this.points.map(point => point[1]);
      if (!values.length) return { polyline: "", scaled: [] };
      const min = Math.min(...values), max = Math.max(...values);
      const scaled = this.points.map((point, index) => ({
        x: index / Math.max(this.points.length - 1, 1) * 118 + 1,
        y: 23 - (max > min ? (point[1] - min) / (max - min) * 20 : 10),
        point }));
      return { polyline: scaled.map(item => `${item.x.toFixed(1)},${item.y.toFixed(1)}`).join(" "), scaled };
    },
  },
  methods: {
    fmt,
    onMove(event) {
      const box = event.currentTarget.getBoundingClientRect();
      const x = (event.clientX - box.left) / box.width * 120;
      let best = null;
      for (const item of this.geometry.scaled) {
        if (best === null || Math.abs(item.x - x) < Math.abs(best.x - x)) best = item;
      }
      this.hover = best;
    },
  },
  template: `
    <div class="spark" @mousemove="onMove" @mouseleave="hover = null">
      <svg viewBox="0 0 120 24" preserveAspectRatio="none">
        <polyline :points="geometry.polyline" fill="none" :stroke="color" stroke-width="1.5"></polyline>
        <circle v-if="hover" :cx="hover.x" :cy="hover.y" r="2.2" :fill="color"></circle>
      </svg>
      <div class="tooltip" v-if="hover">
        <div class="tooltip-x">{{ xlabel }} {{ hover.point[0] }}</div>
        <div>{{ name }} <b>{{ fmt(hover.point[1]) }}</b></div>
      </div>
    </div>`,
};

function portSpread(count, low, high) {
  if (count <= 1) return [(low + high) / 2];
  return Array.from({ length: count }, (_, position) => low + (high - low) * (position + 1) / (count + 1));
}

const Diagram = {
  props: { boxes: { type: Array, default: () => [] }, arrows: { type: Array, default: () => [] }, widths: { type: Object, default: () => ({}) }, title: String },
  data() { return { hover: null, marker: `arrow-${Math.random().toString(36).slice(2, 8)}` }; },
  computed: {
    layout() {
      const CHAR = 6.7, LINE = 15, PAD_X = 14, PAD_Y = 10, GAP_X = 84, GAP_Y = 22, EDGE = 20;
      const columns = {};
      for (const box of this.boxes) (columns[box.column] = columns[box.column] || []).push(box);
      const indices = Object.keys(columns).map(Number).sort((a, b) => a - b);
      if (!indices.length) return { boxes: [], edges: [], width: 200, height: 60 };
      const widths = {};
      for (const index of indices) {
        const chars = Math.max(4, ...columns[index].flatMap(box => box.lines.map(line => Math.min(line.length, 46))));
        widths[index] = Math.min(340, Math.max(96, chars * CHAR + 2 * PAD_X));
      }
      const heightOf = box => box.lines.length * LINE + 2 * PAD_Y;
      const stacks = {};
      for (const index of indices) stacks[index] = columns[index].reduce((sum, box) => sum + heightOf(box), 0) + GAP_Y * (columns[index].length - 1);
      const tallest = Math.max(...Object.values(stacks));
      const lefts = {};
      let cursor = EDGE;
      for (const index of indices) { lefts[index] = cursor; cursor += widths[index] + GAP_X; }
      const width = cursor - GAP_X + EDGE;
      const middle = EDGE + tallest / 2;
      const placed = {};
      for (const index of indices) {
        let top = middle - stacks[index] / 2;
        for (const box of columns[index]) {
          const height = heightOf(box);
          placed[box.name] = { ...box, x: lefts[index], y: top, width: widths[index], height,
                               shown: box.lines.map(line => line.length > 46 ? line.slice(0, 45) + "…" : line) };
          top += height + GAP_Y;
        }
      }
      const leaving = {}, arriving = {};
      this.arrows.forEach(([source, target], position) => {
        if (!placed[source] || !placed[target]) return;
        (leaving[source] = leaving[source] || []).push(position);
        (arriving[target] = arriving[target] || []).push(position);
      });
      const edges = [];
      this.arrows.forEach(([source, target, label, dashed], position) => {
        const from = placed[source], to = placed[target];
        if (!from || !to) return;
        const outs = portSpread(leaving[source].length, from.y + from.height * 0.25, from.y + from.height * 0.75);
        const ins = portSpread(arriving[target].length, to.y + to.height * 0.25, to.y + to.height * 0.75);
        const x1 = from.x + from.width, y1 = outs[leaving[source].indexOf(position)], x2 = to.x, y2 = ins[arriving[target].indexOf(position)];
        const bend = Math.max(30, (x2 - x1) / 2);
        const skips = to.column - from.column > 1;
        const off = skips ? ((y1 + y2) / 2 <= middle ? -70 : 70) : 0;
        const name = label && from.lines[0] !== label ? label : "";
        const wide = label && this.widths[label] ? `[${this.widths[label]}]` : "";
        const shown = name && wide ? `${name} ${wide}` : (name || wide);
        const t = 0.6, u = 1 - t;
        const lx = u * u * u * x1 + 3 * u * u * t * (x1 + bend) + 3 * u * t * t * (x2 - bend) + t * t * t * x2;
        const ly = u * u * u * y1 + 3 * u * u * t * (y1 + off) + 3 * u * t * t * (y2 + off) + t * t * t * y2;
        edges.push({ key: `${source}->${target}`, path: `M${x1} ${y1} C${x1 + bend} ${y1 + off}, ${x2 - bend} ${y2 + off}, ${x2} ${y2}`,
                     label: shown, lx, ly: ly - 7, dashed: !!dashed });
      });
      return { boxes: Object.values(placed), edges, width, height: EDGE + tallest + EDGE };
    },
    kinds() { return [...new Set(this.boxes.map(box => box.kind))]; },
    tooltipStyle() {
      if (!this.hover) return {};
      return { left: `${this.hover.x + this.hover.width + 10}px`, top: `${this.hover.y}px` };
    },
    hoverNote() {
      if (!this.hover) return [];
      const note = this.hover.note && this.hover.note.length ? this.hover.note : (this.hover.detail && this.hover.detail.length ? this.hover.detail : []);
      return note;
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
  components: { chart: Chart, spark: Spark, diagram: Diagram },
  data() {
    return {
      route: parseHash(location.hash),
      tree: { root: "", groups: {} }, filter: "", collapsed: {}, refreshed: "",
      record: null, history: { lines: [], offset: 0 }, steps: { lines: [], offset: 0 },
      logs: { name: "", lines: [], total: 0 }, texts: {}, describeText: null, showModuleText: false,
      sweep: null, overlay: {}, diff: null, modalHeight: 520, liveBoard: { live: [], recent: [] },
      tableRows: [], tableLoaded: false, tableFilter: "", compareData: null, predictionsData: null, prepData: null,
      filesData: null, fileView: null, eventsData: null, playing: false, frame: 0, player: null,
    };
  },
  computed: {
    path() { return this.route.path; },
    page() { return PAGES.includes(this.route.path) ? this.route.path : (this.route.path ? "record" : "home"); },
    isSweep() { return !!(this.record && this.record.manifest && this.record.manifest.kind === "sweep"); },
    kind() { return (this.record && this.record.manifest && this.record.manifest.kind) || "run"; },
    tab() {
      if (TABS.includes(this.route.params.tab)) return this.route.params.tab;
      return this.record && ["running", "pending"].includes(this.state) ? "monitor" : "overview";
    },
    predictionFile() {
      const names = (this.record && this.record.predictions) || [];
      return names.includes(this.route.params.set) ? this.route.params.set : (names[0] || "");
    },
    pairInfo() {
      const pairs = (this.predictionsData && this.predictionsData.pairs) || [];
      return pairs.find(pair => pair.pred === this.route.params.pair) || pairs[0] || null;
    },
    scatterLines() {
      if (!this.pairInfo || !this.predictionsData) return [];
      const pair = this.pairInfo;
      return [{ name: `${pair.pred} against ${pair.target}`, color: PALETTE[0],
                points: this.predictionsData.sample.map(row => [row[pair.target], row[pair.pred]]).filter(point => Number.isFinite(point[0]) && Number.isFinite(point[1])) }];
    },
    histogramLines() {
      if (!this.pairInfo) return [];
      const { edges, counts } = this.pairInfo.histogram;
      return [{ name: "residual", color: PALETTE[1], points: counts.map((value, index) => [fmt((edges[index] + edges[index + 1]) / 2), value]) }];
    },
    prepPreprocessors() {
      const table = (this.prepData && this.prepData.preprocessors) || {};
      return Object.entries(table).map(([name, entry]) => {
        const state = entry.state || {};
        const inner = state.scaler && typeof state.scaler === "object" ? state.scaler : state;
        const perColumn = entry.grouped ? Object.entries(inner).filter(([, value]) => Array.isArray(value) && value.length === entry.columns.length && value.every(item => typeof item === "number")) : [];
        const scalars = Object.entries(inner).filter(([key]) => !perColumn.some(([name]) => name === key));
        return { name, ...entry, perColumn, scalars };
      });
    },
    timeline() {
      const events = (this.eventsData && this.eventsData.events) || [];
      const rows = [];
      const open = {};
      for (const event of events) {
        const when = Date.parse(event.t);
        if (!Number.isFinite(when)) continue;
        const key = event.kind.startsWith("iter") ? `${event.path} #${event.index ?? event.iteration ?? ""}` : event.path;
        if (event.kind === "started" || event.kind === "iter_started") {
          const row = { key, node: event.kind === "started" ? event.node : `${event.node} iteration`, path: event.path, start: when, end: null, status: "", depth: (event.path || "").split(".").length - 1 };
          rows.push(row);
          open[key] = row;
        } else if (open[key]) {
          open[key].end = when;
          open[key].status = event.status || "";
          open[key].ms = event.ms;
          delete open[key];
        }
      }
      if (!rows.length) return { rows: [], total: 0 };
      const first = Math.min(...rows.map(row => row.start));
      const last = Math.max(...rows.map(row => row.end || row.start));
      const total = Math.max(1, last - first);
      return { rows: rows.map(row => ({ ...row, left: 100 * (row.start - first) / total, width: Math.max(0.3, 100 * ((row.end || last) - row.start) / total) })), total };
    },
    tableColumns() {
      const params = new Set(), metrics = new Set();
      for (const row of this.tableRows) {
        Object.keys(row.params || {}).forEach(key => params.add(key));
        Object.keys(row.last || {}).forEach(key => metrics.add(key));
      }
      return { params: [...params], metrics: [...metrics].slice(0, 8) };
    },
    tableSorted() {
      const needle = this.tableFilter.toLowerCase();
      let rows = this.tableRows.filter(row => !needle || row.name.toLowerCase().includes(needle) || row.path.toLowerCase().includes(needle)
        || Object.entries(row.params || {}).some(([key, value]) => `${key}=${value}`.toLowerCase().includes(needle)));
      const key = this.sortKey;
      if (!key) return rows;
      const value = row => {
        if (key === "name") return row.name;
        if (key === "state") return this.stateOf(row);
        if (key === "turns") return row.turns;
        if (key === "seconds") return row.seconds;
        if (key === "best") return row.best ? row.best.value : null;
        if (key.startsWith("param:")) return (row.params || {})[key.slice(6)];
        if (key.startsWith("last:")) return (row.last || {})[key.slice(5)];
        return null;
      };
      rows = [...rows].sort((first, second) => {
        const a = value(first), b = value(second);
        if (a === b) return 0;
        if (a === undefined || a === null) return 1;
        if (b === undefined || b === null) return -1;
        return (a < b ? -1 : 1) * (this.sortDesc ? -1 : 1);
      });
      return rows;
    },
    sweepParamCharts() {
      if (!this.sweep) return [];
      const scored = this.sweep.points.filter(point => point.objective);
      return this.pointKeys.map(key => {
        const points = scored.filter(point => typeof point.values[key] === "number").map(point => [point.values[key], point.objective.value]);
        if (new Set(points.map(point => point[0])).size < 2) return null;
        return { name: key, lines: [{ name: this.sweep.objective.monitor || "objective", color: PALETTE[0], points }] };
      }).filter(Boolean);
    },
    compareCharts() {
      if (!this.compareData) return [];
      const byName = {};
      const label = which => (this.compareData[which].record.manifest && this.compareData[which].record.manifest.name) || this.compareData[which].path;
      for (const which of ["a", "b"]) {
        const series = {};
        for (const line of this.compareData[which].history) {
          for (const [key, value] of Object.entries(line)) {
            if (SKIP_KEYS.has(key) || typeof value !== "number" || key.startsWith("lr/") || key.startsWith("minimizes/")) continue;
            (series[key] = series[key] || []).push([line.turn, value]);
          }
        }
        for (const [key, points] of Object.entries(series)) {
          const { set, name } = splitKey(key);
          (byName[name] = byName[name] || []).push({ name: `${label(which)} ${set}`, color: setColor(set), dashed: which === "b", points });
        }
      }
      return Object.entries(byName).map(([name, lines]) => ({ name, lines }));
    },
    item() { return this.route.params.item || null; },
    logy() { return this.route.params.log === "1"; },
    metricFilter() { return this.route.params.q || ""; },
    logName() {
      const names = (this.record && this.record.logs) || [];
      return names.includes(this.route.params.file) ? this.route.params.file : (names[0] || "");
    },
    selected() { return this.route.params.pick ? this.route.params.pick.split(",").filter(Boolean) : []; },
    overlayPaths() {
      if (this.selected.length) return this.selected;
      if (!this.sweep) return [];
      return this.sweep.points.filter(point => this.stateOf(point) === "running").slice(0, 8).map(point => point.path);
    },
    sortKey() { return this.route.params.sort || null; },
    sortDesc() { return this.route.params.desc === "1"; },
    groups() {
      const needle = this.filter.toLowerCase();
      return Object.entries(this.tree.groups).map(([name, entries]) => ({
        name, entries: entries.filter(entry => !needle || entry.name.toLowerCase().includes(needle) || entry.path.toLowerCase().includes(needle)) }))
        .filter(group => group.entries.length);
    },
    state() { return this.record ? this.stateOf(this.record) : "pending"; },
    live() {
      if (!this.path) return false;
      if (this.isSweep) return !!(this.sweep && this.sweep.points.some(point => this.stateOf(point) === "running"));
      return ["running", "pending"].includes(this.state);
    },
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
    tabNames() { return TABS; },
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
        (byName[name] = byName[name] || []).push({ name: set || key, points });
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
        const definition = this.definitionOf(name);
        return { key, set, name, ...definition, last: points[points.length - 1][1], min: low[1], minTurn: low[0], max: high[1], maxTurn: high[0], tail: points.slice(-60) };
      });
    },
    rulesFired() {
      return this.history.lines.filter(line => (line.rules || []).length).map(line => `${this.turnLabel} ${line.turn}: ${line.rules.join(", ")}`);
    },
    turnLabel() {
      const noted = this.record && this.record.manifest && this.record.manifest.turn;
      if (noted === "epoch") return "epoch";
      if (noted === "steps") return "turn";
      const lines = this.history.lines;
      const loaders = this.record && this.record.data && this.record.data.loaders;
      const batches = loaders && loaders.train && loaders.train.batches;
      if (!lines.length || !batches) return "turn";
      const per = lines.length > 1 ? lines[1].global_step - lines[0].global_step : lines[0].global_step;
      return per === batches ? "epoch" : "turn";
    },
    minimized() {
      const found = {};
      for (const line of this.history.lines) {
        for (const [key, value] of Object.entries(line)) {
          if (!key.startsWith("minimizes/") || typeof value !== "string") continue;
          (found[key.slice(10)] = found[key.slice(10)] || []).push([line.turn, value]);
        }
      }
      return found;
    },
    training() {
      const names = new Set([...Object.keys(this.minimized), ...Object.keys(this.series).filter(key => key.startsWith("lr/")).map(key => key.slice(3))]);
      return [...names].map(name => {
        const path = this.minimized[name] || [];
        const segments = [];
        for (const [turn, loss] of path) {
          const last = segments[segments.length - 1];
          if (last && last.loss === loss) last.to = turn; else segments.push({ loss, from: turn, to: turn });
        }
        const current = path.length ? path[path.length - 1][1] : null;
        const definition = current ? this.definitionOf(current) : null;
        const terms = definition && definition.params && typeof definition.params.terms === "object" ? definition.params.terms : null;
        const prefix = current ? `train/${current}/` : null;
        const parts = terms ? Object.entries(terms).map(([term, weight]) => `${fmt(weight)} × ${term}`)
          : (prefix ? Object.keys(this.series).filter(key => key.startsWith(prefix)).map(key => key.slice(prefix.length)) : []);
        const rates = this.series[`lr/${name}`];
        return { name, current, uri: definition ? definition.uri : "", segments, parts, lr: rates ? rates[rates.length - 1][1] : null };
      });
    },
    ruleMarks() { return this.history.lines.filter(line => (line.rules || []).length).map(line => line.turn); },
    plan() {
      const training = (this.record && this.record.config && this.record.config.training) || {};
      if (training.steps && typeof training.steps === "object") {
        const total = training.steps.total, per = training.steps.turn;
        return typeof total === "number" && typeof per === "number" && per ? Math.ceil(total / per) : null;
      }
      return typeof training.epochs === "number" ? training.epochs : null;
    },
    monitorKey() {
      const training = (this.record && this.record.config && this.record.config.training) || {};
      const checkpoint = training.checkpoint;
      return checkpoint && typeof checkpoint === "object" && checkpoint.params ? checkpoint.params.monitor || "" : "";
    },
    progressInfo() {
      const lines = this.history.lines;
      const done = lines.length;
      const timed = lines.filter(line => typeof line.seconds === "number");
      const mean = timed.length ? timed.reduce((sum, line) => sum + line.seconds, 0) / timed.length : null;
      const remaining = this.plan !== null ? Math.max(0, this.plan - done) : null;
      const eta = mean !== null && remaining !== null ? mean * remaining : null;
      const last = lines.length ? lines[lines.length - 1] : null;
      const step = this.steps.lines.length ? this.steps.lines[this.steps.lines.length - 1] : null;
      const loaders = this.record && this.record.data && this.record.data.loaders;
      const perTurn = loaders && loaders.train ? loaders.train.batches : null;
      const inTurn = step && last ? step.step - (last.global_step || 0) : (step ? step.step : null);
      const share = this.plan ? Math.min(100, 100 * (done + (perTurn && inTurn !== null && inTurn > 0 ? Math.min(inTurn / perTurn, 1) : 0)) / this.plan) : 0;
      return { done, plan: this.plan, eta, mean, share, step: step ? step.step : null, inTurn, perTurn,
               losses: step ? Object.entries(step).filter(([key, value]) => key.startsWith("loss/") && typeof value === "number") : [],
               monitorValue: last && this.monitorKey ? last[this.monitorKey] : null };
    },
    monitorChart() {
      if (!this.monitorKey) return null;
      const { name } = splitKey(this.monitorKey);
      return this.metricCharts.find(entry => entry.name === name) || null;
    },
    activeLossChart() {
      const names = Object.keys(this.minimized);
      const current = names.length ? this.minimized[names[0]][this.minimized[names[0]].length - 1][1] : null;
      if (!current) return null;
      return this.metricCharts.find(entry => entry.name === current) || null;
    },
    epochs() {
      const rows = this.history.lines.map(line => ({ turn: line.turn, seconds: typeof line.seconds === "number" ? line.seconds : null, step: line.global_step }));
      const timed = rows.filter(row => row.seconds !== null);
      const max = timed.length ? Math.max(...timed.map(row => row.seconds)) : 0;
      const total = timed.reduce((sum, row) => sum + row.seconds, 0);
      return { rows: rows.map(row => ({ ...row, share: row.seconds !== null && max > 0 ? 100 * row.seconds / max : 0 })), total, mean: timed.length ? total / timed.length : 0 };
    },
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
    expanded() {
      const name = this.route.params.chart;
      if (!name || !this.record) return null;
      if (this.isSweep) return name === "overlay" && this.overlayLines.length ? { name: this.sweep.objective.monitor || "objective", lines: this.overlayLines, xlabel: "turn", marks: [] } : null;
      if (this.tab === "steps") {
        const found = this.stepCharts.find(entry => entry.name === name);
        return found ? { name, lines: found.lines, xlabel: "step", marks: this.turnMarks } : null;
      }
      if (name === "learning rate") return this.rateLines.length ? { name, lines: this.rateLines, xlabel: this.turnLabel, marks: [], logy: true } : null;
      const found = this.metricCharts.find(entry => entry.name === name);
      return found ? { name, lines: found.lines, xlabel: this.turnLabel, marks: this.ruleMarks } : null;
    },
    gallery() {
      if (!this.record) return [];
      return this.tab === "samples" ? this.record.samples : this.record.plots;
    },
    images() { return this.gallery.filter(name => IMAGE.test(name)); },
    textFiles() { return this.gallery.filter(name => TEXT.test(name)); },
    itemIsText() { return !!this.item && TEXT.test(this.item); },
    itemText() { return this.item ? this.texts[this.textKey(this.item)] : undefined; },
    architectureModels() { return Object.keys((this.record && this.record.architecture && this.record.architecture.models) || {}); },
    currentArchitecture() {
      const models = (this.record && this.record.architecture && this.record.architecture.models) || {};
      const label = models[this.route.params.model] ? this.route.params.model : this.architectureModels[0];
      return label ? { label, ...models[label] } : null;
    },
    moduleText() {
      if (!this.record || !this.record.plots.includes("architecture_text.txt")) return null;
      return this.texts[this.textKey("architecture_text.txt")];
    },
    pipeline() {
      const data = this.record && this.record.data;
      if (!data || !(data.stages || []).length) return null;
      const boxes = [], arrows = [];
      const stages = data.stages;
      let column = 0, previous = "source";
      const source = (this.record.config && this.record.config.data && this.record.config.data.source) || {};
      const sourceParams = typeof source === "object" ? source.params || {} : {};
      boxes.push({ name: "source", kind: "source", column, row: 0, note: source.uri ? [`uri ${source.uri}`, ...Object.entries(sourceParams).map(([key, value]) => `${key}: ${typeof value === "object" ? JSON.stringify(value) : value}`)] : [],
                   lines: [source.uri ? shortUri(source.uri) : "source", ...(sourceParams.path ? [String(sourceParams.path).split("/").pop()] : []), `${count(stages[0].rows)} rows · ${stages[0].columns} columns`] });
      for (const stage of stages.slice(1)) {
        column += 1;
        const added = stage.added || [], removed = stage.removed || [];
        const change = [added.length ? `+${added.length}` : "", removed.length ? `−${removed.length}` : ""].filter(Boolean).join(" ");
        const params = stage.call ? paramsText(stage.call.params) : "";
        const note = [`stage ${stage.stage}`, ...(stage.call ? [`uri ${stage.call.uri}`, ...Object.entries(stage.call.params || {}).map(([key, value]) => `${key}: ${typeof value === "object" ? JSON.stringify(value) : value}`)] : []), "",
                      ...added.map(name => `+ ${name}`), ...removed.map(name => `− ${name}`)];
        boxes.push({ name: stage.stage, kind: "transform", column, row: 0, note,
                     lines: [stage.call ? shortUri(stage.call.uri) : stage.stage, ...(params ? [params] : []), `${count(stage.rows)} rows · ${stage.columns} columns${change ? "  " + change : ""}`] });
        arrows.push([previous, stage.stage, "", false]);
        previous = stage.stage;
      }
      const sets = orderedSets(Object.keys(data.split || {}));
      if (!sets.length) return { boxes, arrows };
      column += 1;
      sets.forEach((set, row) => { boxes.push({ name: `split:${set}`, kind: "split", column, row, lines: [set, `${count(data.split[set])} rows`] }); arrows.push([previous, `split:${set}`, "", false]); });
      let last = set => `split:${set}`;
      const after = data.after_set_transforms || {};
      const perSet = data.set_transforms || {};
      if (sets.some(set => (perSet[set] || []).length || (after[set] !== undefined && after[set] !== data.split[set]))) {
        column += 1;
        sets.forEach((set, row) => {
          const calls = (perSet[set] || []).map(call => `${shortUri(call.uri)} ${paramsText(call.params)}`.trim());
          boxes.push({ name: `after:${set}`, kind: "transform", column, row, note: (perSet[set] || []).map(call => `uri ${call.uri} ${paramsText(call.params)}`),
                       lines: [`${set} transforms`, ...(calls.length ? calls : ["none"]), `${count(after[set] !== undefined ? after[set] : data.split[set])} rows`] });
          arrows.push([`split:${set}`, `after:${set}`, "", false]);
        });
        last = set => `after:${set}`;
      }
      const fit = data.fit || {};
      column += 1;
      const fitted = Object.entries(fit.preprocessors || {}).map(([name, columns]) => `${name} ${columns}`).join(", ");
      const lines = ["fit on train", ...((data.frames || []).length ? [`frames ${data.frames.join(", ")}`] : []), ...wrapWords(fitted || "no preprocessors", 30),
                     `${fit.features || 0} features, ${(fit.targets || []).length} targets`];
      boxes.push({ name: "fit", kind: "fit", column, row: 0, lines, note: (fit.targets || []).length ? ["targets", ...fit.targets] : [] });
      sets.forEach(set => arrows.push([last(set), "fit", "", set !== "train"]));
      column += 1;
      sets.forEach((set, row) => { const entry = (data.sets || {})[set] || {}; boxes.push({ name: `feed:${set}`, kind: "feed", column, row, lines: [set, `${count(entry.rows)} rows`, `${entry.features ?? "?"} features`] }); arrows.push(["fit", `feed:${set}`, "", false]); });
      column += 1;
      sets.forEach((set, row) => { const entry = (data.loaders || {})[set] || {}; boxes.push({ name: `loader:${set}`, kind: "loaders", column, row, lines: [`${set} loader`, `${count(entry.batches)} x ${entry.size}`] }); arrows.push([`feed:${set}`, `loader:${set}`, "", false]); });
      return { boxes, arrows };
    },
    dataStages() { return (this.record && this.record.data && this.record.data.stages) || []; },
    setTransforms() {
      const perSet = (this.record && this.record.data && this.record.data.set_transforms) || {};
      return orderedSets(Object.keys(perSet)).map(set => ({ set, calls: perSet[set] }));
    },
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
      return this.overlayPaths.filter(path => this.overlay[path]).map((path, index) => ({
        name: path.split("/").pop(), color: PALETTE[index % PALETTE.length],
        points: this.overlay[path].filter(line => typeof line[monitor] === "number").map(line => [line.turn, line[monitor]]) }));
    },
  },
  watch: {
    path: { immediate: true, handler() { this.enterRecord(); } },
    predictionFile() { if (this.tab === "predictions") this.loadPredictions(); },
    "route.params.a"() { if (this.page === "compare") this.loadCompare(); },
    "route.params.b"() { if (this.page === "compare") this.loadCompare(); },
    tab: { immediate: true, handler() { this.enterTab(); } },
    item() { this.enterItem(); },
    overlayPaths() { this.loadOverlay(); },
    logName() { if (this.tab === "logs") this.loadLogs(); },
  },
  methods: {
    fmt, count, ms, ago, clock, setColor, shortUri, paramsText,
    stateOf(entry) { return (entry.status && entry.status.state) || "pending"; },
    definitionOf(name) {
      const config = (this.record && this.record.config) || {};
      const head = name.split("/")[0];
      for (const kind of ["loss", "metric"]) {
        const table = config[kind === "loss" ? "losses" : "metrics"] || {};
        const entry = table[head];
        if (entry && typeof entry === "object") return { kind, uri: entry.uri || "", params: entry.params || {}, output: entry.output || "", target: entry.target || "" };
      }
      if (config.training && config.training.loss === head) return { kind: "loss", uri: "", params: {}, output: "", target: "" };
      return { kind: "", uri: "", params: {}, output: "", target: "" };
    },
    text(value) { return typeof value === "object" && value !== null ? JSON.stringify(value) : String(value); },
    entries(mapping, skip) { return Object.entries(mapping || {}).filter(([key]) => !(skip || []).includes(key)); },
    fileUrl(kind, name) { return `/file?path=${encodeURIComponent(`${this.path}/${kind}/${name}`)}`; },
    textKey(name) { return `${this.path}/${this.tab === "samples" ? "samples" : "plots"}/${name}`; },
    link(changes, path) {
      const params = { ...this.route.params, ...(changes || {}) };
      return buildHash(path === undefined ? this.path : path, params);
    },
    recordLink(path) { return buildHash(path, {}); },
    go(changes, replace) {
      const hash = this.link(changes);
      if (hash === location.hash) return;
      if (replace) {
        history.replaceState(null, "", hash);
        this.route = parseHash(hash);
      } else {
        location.hash = hash;
      }
    },
    onHash() {
      const parsed = parseHash(location.hash);
      if (JSON.stringify(parsed) !== JSON.stringify(this.route)) this.route = parsed;
    },
    tabLink(name) {
      const kept = { tab: name, item: null, model: null, file: null, q: null, log: null, chart: null };
      if (name === "curves" || name === "steps") kept.log = this.route.params.log;
      return this.link(kept);
    },
    tabCount(name) {
      if (!this.record) return 0;
      if (name === "model") return this.architectureModels.length;
      if (name === "plots") return this.record.plots.length;
      if (name === "samples") return this.record.samples.length;
      if (name === "events") return this.record.events.length;
      return 0;
    },
    diffClass(line) {
      if (line.startsWith("+++") || line.startsWith("---") || line.startsWith("@@")) return "hunk";
      if (line.startsWith("+")) return "add";
      if (line.startsWith("-")) return "del";
      return "";
    },
    toggleGroup(name) { this.collapsed = { ...this.collapsed, [name]: !this.collapsed[name] }; },
    async loadLive() {
      const found = await api("/api/live");
      if (found) this.liveBoard = found;
    },
    monitorLink(path) { return buildHash(path, { tab: "monitor" }); },
    async loadTree() {
      const tree = await api("/api/tree");
      if (tree) this.tree = tree;
      this.refreshed = new Date().toLocaleTimeString();
      if (!this.path) await this.loadLive();
      if (!this.record || this.isSweep) return;
      const entry = Object.values(this.tree.groups).flat().find(item => item.path === this.path);
      if (entry && this.stateOf(entry) !== this.state) await this.loadRecord();
    },
    async enterRecord() {
      this.record = null; this.sweep = null; this.history = { lines: [], offset: 0 }; this.steps = { lines: [], offset: 0 };
      this.overlay = {}; this.diff = null; this.describeText = null; this.showModuleText = false;
      this.logs = { name: "", lines: [], total: 0 };
      this.predictionsData = null; this.prepData = null; this.filesData = null; this.fileView = null; this.eventsData = null;
      this.stopPlaying();
      if (this.page === "table") { await this.loadTable(); return; }
      if (this.page === "compare") { await this.loadCompare(); return; }
      if (!this.path) return;
      await this.loadRecord();
      if (this.isSweep) { await this.loadSweep(); await this.loadOverlay(); return; }
      await this.enterTab();
    },
    async loadRecord() {
      const path = this.path;
      const record = await api("/api/record", { path });
      if (!record || path !== this.path) return;
      this.record = record;
      if (record.manifest && record.manifest.kind === "sweep") return;
      await this.loadHistory();
    },
    async loadHistory() {
      const path = this.path;
      const found = await api("/api/history", { path, offset: this.history.offset });
      if (found && path === this.path) this.history = { lines: this.history.lines.concat(found.lines), offset: found.offset };
    },
    async loadSteps() {
      const path = this.path;
      const found = await api("/api/steps", { path, offset: this.steps.offset });
      if (found && path === this.path) this.steps = { lines: this.steps.lines.concat(found.lines), offset: found.offset };
    },
    async loadLogs() {
      if (!this.record || !this.logName) return;
      const path = this.path, name = this.logName;
      const found = await api("/api/tail", { path, name, lines: 300 });
      if (found && path === this.path) this.logs = { name, lines: found.lines, total: found.total || 0 };
    },
    async loadTexts() {
      if (!this.record) return;
      const path = this.path;
      const wanted = [...this.record.plots.filter(name => TEXT.test(name)).map(name => ["plots", name]),
                      ...this.record.samples.filter(name => TEXT.test(name)).map(name => ["samples", name])];
      for (const [kind, name] of wanted) {
        const key = `${path}/${kind}/${name}`;
        if (this.texts[key] !== undefined) continue;
        const response = await fetch(`/file?path=${encodeURIComponent(key)}`);
        const content = response.ok ? await response.text() : "unreadable";
        this.texts = { ...this.texts, [key]: content };
      }
    },
    async loadDescribe() {
      if (this.describeText !== null) return;
      const path = this.path;
      const found = await api("/api/describe", { path });
      if (path === this.path) this.describeText = found ? found.text : "the record cannot be described";
    },
    async loadTable() {
      const found = await api("/api/table");
      if (found) { this.tableRows = found.rows; this.tableLoaded = true; }
    },
    async loadCompare() {
      const { a, b } = this.route.params;
      if (!a || !b) { this.compareData = null; return; }
      const [recordA, recordB, historyA, historyB, diff] = await Promise.all([
        api("/api/record", { path: a }), api("/api/record", { path: b }), api("/api/history", { path: a }), api("/api/history", { path: b }),
        api("/api/diff", { a, b })]);
      if (!recordA || !recordB) { this.compareData = { missing: true }; return; }
      this.compareData = { a: { path: a, record: recordA, history: historyA ? historyA.lines : [] },
                           b: { path: b, record: recordB, history: historyB ? historyB.lines : [] },
                           diff: diff ? (diff.diff.length ? diff.diff : ["no difference"]) : ["no resolved.yaml to compare"] };
    },
    async loadPredictions() {
      if (!this.record || !this.predictionFile) { this.predictionsData = null; return; }
      const path = this.path;
      const found = await api("/api/predictions", { path, name: this.predictionFile, sample: 2000 });
      if (path === this.path) this.predictionsData = found;
    },
    async loadPrep() {
      if (this.prepData !== null) return;
      const path = this.path;
      const found = await api("/api/prep", { path });
      if (path === this.path) this.prepData = found || { fields: [], preprocessors: {}, missing: true };
    },
    async loadFiles() {
      const path = this.path;
      const found = await api("/api/files", { path });
      if (path === this.path) this.filesData = found;
    },
    async openFile(name) {
      const path = this.path;
      const found = await api("/api/text", { path, name });
      if (path === this.path) this.fileView = found || { name, text: "this file is not shown as text", truncated: false, size: 0 };
    },
    async loadEvents() {
      const path = this.path;
      const found = await api("/api/events", { path });
      if (path === this.path) this.eventsData = found;
    },
    isText(name) { return /\.(yaml|yml|json|jsonl|txt|md|csv|py|sh|sub|plan|log)$/i.test(name); },
    togglePlaying() {
      if (this.playing) { this.stopPlaying(); return; }
      if (!this.record || !this.record.samples.length) return;
      this.playing = true;
      this.player = setInterval(() => { this.frame = (this.frame + 1) % this.record.samples.length; }, 400);
    },
    stopPlaying() {
      if (this.player) clearInterval(this.player);
      this.player = null;
      this.playing = false;
    },
    async enterTab() {
      if (!this.record || this.isSweep) return;
      if (this.tab === "predictions") await this.loadPredictions();
      if (this.tab === "prep") await this.loadPrep();
      if (this.tab === "files") await this.loadFiles();
      if (this.tab === "timeline") await this.loadEvents();
      if (this.tab !== "samples") this.stopPlaying();
      if ((this.tab === "steps" || this.tab === "monitor") && !this.steps.lines.length) await this.loadSteps();
      if (this.tab === "logs" || this.tab === "monitor") await this.loadLogs();
      if (this.tab === "plots" || this.tab === "samples" || this.tab === "model") await this.loadTexts();
      if (this.tab === "describe") await this.loadDescribe();
      await this.enterItem();
    },
    async enterItem() {
      if (this.item && TEXT.test(this.item)) await this.loadTexts();
    },
    async loadSweep() {
      const path = this.path;
      const sweep = await api("/api/sweep", { path });
      if (sweep && path === this.path) this.sweep = sweep;
    },
    async loadOverlay() {
      if (!this.isSweep) return;
      const path = this.path;
      const found = {};
      for (const point of this.overlayPaths) {
        const history = await api("/api/history", { path: point });
        if (history) found[point] = history.lines;
      }
      if (path !== this.path) return;
      this.overlay = found;
      await this.loadDiff();
    },
    async loadDiff() {
      if (this.selected.length < 2) { this.diff = null; return; }
      const found = await api("/api/diff", { a: this.selected[0], b: this.selected[1] });
      this.diff = found ? (found.diff.length ? found.diff : ["no difference"]) : ["no resolved.yaml to compare"];
    },
    togglePick(path) {
      const picks = this.selected.includes(path) ? this.selected.filter(entry => entry !== path) : [...this.selected, path];
      this.go({ pick: picks.join(",") }, true);
    },
    sortBy(key) {
      if (this.sortKey === key) this.go({ desc: this.sortDesc ? null : "1" }, true);
      else this.go({ sort: key, desc: null }, true);
    },
    setLog(value) { this.go({ log: value ? "1" : null }, true); },
    setFilter(value) { this.go({ q: value }, true); },
    closeItem() { this.go({ item: null }, true); },
    closeChart() { this.go({ chart: null }, true); },
    openChart(name) {
      this.modalHeight = Math.max(360, Math.round(window.innerHeight * 0.72));
      this.go({ chart: name });
    },
    stepItem(direction) {
      const names = this.gallery;
      const index = names.indexOf(this.item);
      if (index < 0 || !names.length) return;
      this.go({ item: names[(index + direction + names.length) % names.length] }, true);
    },
    pickModel(box) {
      if (box.kind !== "model") return;
      const label = (box.lines[1] || "").replace(/^model /, "");
      if (this.architectureModels.includes(label)) this.go({ model: label });
    },
    async copy(text) {
      try { await navigator.clipboard.writeText(text || ""); } catch (error) { console.warn("clipboard unavailable", error); }
    },
    async tick() {
      if (!this.path) { await this.loadLive(); return; }
      if (this.page === "table") { await this.loadTable(); return; }
      if (this.page === "compare" || !this.record) return;
      if (this.isSweep) { await this.loadSweep(); await this.loadOverlay(); return; }
      if (!["running", "pending"].includes(this.state)) return;
      await this.loadRecord();
      if (this.tab === "steps" || this.tab === "monitor") await this.loadSteps();
      if (this.tab === "logs" || this.tab === "monitor") await this.loadLogs();
      if (this.tab === "timeline") await this.loadEvents();
    },
    fmtBytes: human,
  },
  mounted() {
    this.modalHeight = Math.max(360, Math.round(window.innerHeight * 0.72));
    this.loadTree();
    this.loadLive();
    window.addEventListener("hashchange", () => this.onHash());
    setInterval(() => this.tick(), 3000);
    setInterval(() => this.loadTree(), 10000);
    window.addEventListener("keydown", event => {
      if (event.key === "Escape") {
        if (this.route.params.chart) this.closeChart();
        else if (this.item) this.closeItem();
        return;
      }
      if (!this.item) return;
      if (event.key === "ArrowRight") this.stepItem(1);
      if (event.key === "ArrowLeft") this.stepItem(-1);
    });
  },
});

app.mount("#app");
