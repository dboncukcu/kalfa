const SET_COLORS = { train: "#2f6fdd", val: "#e8722a", test: "#17a673", calib: "#8e5bd8" };
const PALETTE = ["#2f6fdd", "#e8722a", "#17a673", "#8e5bd8", "#d6437a", "#c99a06", "#1c9aa8", "#7a7a7a"];
const SKIP_KEYS = new Set(["turn", "global_step", "step", "seconds", "rules"]);
const KIND_COLORS = { input: "#8a8f98", torch: "#2f6fdd", lego: "#17a673", model: "#8e5bd8", output: "#e0a106",
                      loss: "#d8433c", optimizer: "#e8722a", source: "#8a8f98", transform: "#17a673", split: "#2f6fdd",
                      frames: "#8e5bd8", fit: "#e0a106", feed: "#d6437a", loaders: "#e8722a" };
const KIND_NAMES = { input: "input wire", torch: "torch layer", lego: "kalfa layer", model: "model", output: "output wire",
                     loss: "loss", optimizer: "optimizer", source: "source", transform: "transform", split: "split",
                     frames: "frame transforms", fit: "fit on train", feed: "feed", loaders: "loaders" };
const NODE_KINDS = new Set(["torch", "lego", "model"]);
const RAMP_LIGHT = ["#7d1a15", "#b5332a", "#d6604f", "#9aa0a8", "#4a8fe0", "#2360b4", "#14396f"];
const RAMP_DARK = ["#f2a7a0", "#e06a5e", "#c03d33", "#8b9099", "#2f6fd0", "#5f9ae8", "#9ec5f4"];
const SCORE_BANDS_LIGHT = ["#7d1a15", "#c94a3e", "#9aa0a8", "#3d7fd0", "#14396f"];
const SCORE_BANDS_DARK = ["#f2a7a0", "#d4544a", "#8b9099", "#3f7ad6", "#9ec5f4"];
const BANDS_LIGHT = ["#0d366b", "#256abf", "#3987e5", "#5598e7", "#86b6ef"];
const BANDS_DARK = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#184f95"];
const MARK_LIGHT = "#5f6672";
const MARK_DARK = "#9aa3af";
const STATE_COLORS = { running: "#e39b12", finished: "#17a673", failed: "#d8433c", pending: "#9aa0a8" };
const EXTRA_AXES = ["objective", "turns", "id", "state"];
const SET_ORDER = ["train", "valid", "test"];
const TABS = ["monitor", "overview", "curves", "steps", "model", "data", "predictions", "prep", "plots", "samples", "config", "notes", "files", "timeline", "events", "logs", "describe"];
const TAB_LABELS = { curves: "loss and metrics", steps: "optimizer steps" };
const PAGES = ["table", "compare"];
const IMAGE = /\.(png|jpe?g|gif|svg|webp|bmp)$/i;
const TEXT = /\.(txt|md|json|csv|yaml|yml)$/i;
const FILE_TEXT = /\.(yaml|yml|json|jsonl|txt|md|csv|tsv|py|sh|sub|plan|log|toml|ini|cfg|rst|html|xml)$/i;
const FILE_PDF = /\.pdf$/i;

function fileKind(name) {
  if (IMAGE.test(name)) return "image";
  if (FILE_PDF.test(name)) return "pdf";
  return FILE_TEXT.test(name) ? "text" : "binary";
}

function defaultPlot() {
  return { logy: false, width: 1.6, markers: 3, dots: 0, curve: "straight", grid: true, bins: 40 };
}

function scatterPoints(lines, kind) {
  return (lines || []).filter(line => (line.kind || kind || "line") === "scatter").reduce((sum, line) => sum + (line.points || []).length, 0);
}

function markerSize(points, settings) {
  if (settings && typeof settings.markers === "number") return settings.markers;
  return points < 40 ? 6 : points < 200 ? 5 : points < 1000 ? 4 : 3;
}

async function api(route, params) {
  const query = new URLSearchParams(params || {}).toString();
  const response = await fetch(route + (query ? "?" + query : ""));
  if (!response.ok) return null;
  return response.json();
}

async function post(route, params) {
  const query = new URLSearchParams(params || {}).toString();
  const response = await fetch(route + (query ? "?" + query : ""), { method: "POST" });
  const body = await response.json().catch(() => ({}));
  return { ok: response.ok, ...body };
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

function tick(value) {
  if (!Number.isFinite(value)) return "";
  const size = Math.abs(value);
  if (size < 1e-12) return "0";
  if (size >= 1e5 || size < 1e-3) return value.toExponential(1);
  return String(Number(value.toPrecision(3)));
}

function rampColor(share) {
  const ramp = darkMode() ? RAMP_DARK : RAMP_LIGHT;
  return ramp[Math.round(Math.min(1, Math.max(0, share)) * (ramp.length - 1))];
}

function bandColors(score) {
  if (score) return darkMode() ? SCORE_BANDS_DARK : SCORE_BANDS_LIGHT;
  return darkMode() ? BANDS_DARK : BANDS_LIGHT;
}

function markColor() {
  return darkMode() ? MARK_DARK : MARK_LIGHT;
}

function axisNorm(axis, value) {
  if (!axis || value === null || value === undefined) return null;
  if (axis.kind === "choices") {
    const index = axis.values.findIndex(item => item === value);
    if (index < 0) return null;
    return axis.values.length > 1 ? index / (axis.values.length - 1) : 0.5;
  }
  const low = axis.log ? Math.log10(axis.low) : axis.low;
  const high = axis.log ? Math.log10(axis.high) : axis.high;
  const at = axis.log ? Math.log10(value) : value;
  if (!Number.isFinite(at) || !Number.isFinite(low) || !Number.isFinite(high)) return null;
  if (high === low) return 0.5;
  return Math.min(1, Math.max(0, (at - low) / (high - low)));
}

function axisTicks(axis) {
  if (axis.kind === "choices") return axis.values.map(value => ({ at: axisNorm(axis, value), text: fmt(value) }));
  const low = axis.log ? Math.log10(axis.low) : axis.low;
  const high = axis.log ? Math.log10(axis.high) : axis.high;
  if (!Number.isFinite(low) || !Number.isFinite(high)) return [];
  return [0, 0.25, 0.5, 0.75, 1].map(at => ({ at, text: tick(axis.log ? Math.pow(10, low + at * (high - low)) : low + at * (high - low)) }));
}

function median(values) {
  const sorted = [...values].sort((first, second) => first - second);
  if (!sorted.length) return null;
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function extent(values) {
  let low = Infinity, high = -Infinity;
  for (const value of values) { if (value < low) low = value; if (value > high) high = value; }
  return [low, high];
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

function paramCount(value) {
  if (typeof value !== "number" || value <= 0) return "";
  if (value < 1000) return `${value} params`;
  if (value < 1e6) return `${(value / 1e3).toFixed(1)}k params`;
  return `${(value / 1e6).toFixed(2)}M params`;
}

function callLines(uri, params) {
  return [`uri ${uri}`, ...Object.entries(params || {}).map(([key, value]) => `${key}: ${typeof value === "object" && value !== null ? JSON.stringify(value) : value}`)];
}

function parseShape(line) {
  const parts = typeof line === "string" && !line.includes(": ") ? line.split(" -> ") : [];
  return parts.length === 2 ? parts : null;
}

function parseDetail(line) {
  const item = { name: line, kind: "", class: "", summary: "", shapes: null, parameters: 0, children: [] };
  const match = /^(.+?): ([A-Za-z_]\w*)(?:\((.*)\))?(?: -> (.+))?$/.exec(line);
  if (!match) return item;
  return { ...item, name: match[1], class: match[2], summary: match[3] || "", shapes: match[4] ? ["", match[4]] : null };
}

function layerItems(box) {
  return Array.isArray(box.layers) ? box.layers : (box.detail || []).map(parseDetail);
}

function layerLine(item) {
  const summary = item.summary.length > 34 ? item.summary.slice(0, 33) + "…" : item.summary;
  const shape = item.shapes && item.shapes[1] ? ` -> ${item.shapes[1]}` : "";
  return `${item.name}: ${item.class}${summary ? `(${summary})` : ""}${shape}`;
}

function nodeItem(box) {
  return { name: box.name, kind: box.kind, class: box.lines[1] || "", summary: "", shapes: parseShape(box.lines[2]),
           parameters: box.parameters || 0, children: layerItems(box) };
}

function modelBox(box, models) {
  const shape = parseShape(box.lines[2]);
  if (box.kind === "input") {
    const features = box.detail || [];
    return { ...box, note: features, groups: features.length ? [{ title: `${features.length} features`, items: features }] : [] };
  }
  if (box.kind === "model") {
    const label = (box.lines[1] || "").replace(/^model /, "");
    const nodes = ((models[label] && models[label].boxes) || []).filter(item => NODE_KINDS.has(item.kind)).map(nodeItem);
    return { ...box, note: nodes.map(layerLine), blocks: nodes, chain: false, open: models[label] ? label : "" };
  }
  if (!NODE_KINDS.has(box.kind)) return { ...box, note: [] };
  const items = layerItems(box);
  const head = box.lines.slice(0, shape ? 3 : 2);
  const hint = [items.length ? `${items.length} layers` : "", paramCount(box.parameters)].filter(Boolean).join(" · ");
  return { ...box, lines: hint ? [...head, hint] : head, note: items.map(layerLine), blocks: items, chain: head[1] === "Sequential" };
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

function iconSvg(state, share) {
  const ground = { idle: "#9aa0a8", running: "#e39b12", failed: "#d8433c" }[state] || "#2f6fdd";
  const bar = share === null || share === undefined ? ""
    : `<rect x="4" y="28" width="${(24 * Math.max(0, Math.min(1, share))).toFixed(1)}" height="2" rx="1" fill="#fff" opacity=".9"/>`;
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" rx="7" fill="${ground}"/>`
    + `<rect x="5" y="17" width="9" height="9" rx="1.6" fill="#fff"/><rect x="11.5" y="10.5" width="9" height="9" rx="1.6" fill="#fff" opacity=".82"/>`
    + `<rect x="18" y="4" width="9" height="9" rx="1.6" fill="#fff" opacity=".64"/>${bar}</svg>`;
}

function setIcon(svg) {
  let link = document.querySelector("link[rel~=icon]");
  if (!link) { link = document.createElement("link"); link.rel = "icon"; document.head.appendChild(link); }
  link.type = "image/svg+xml";
  link.href = `data:image/svg+xml,${encodeURIComponent(svg)}`;
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
  const settings = { ...defaultPlot(), ...(view.settings || {}) };
  const kind = view.kind || "line";
  const logy = !!view.logy;
  const up = value => (logy ? Math.log10(value) : value);
  const down = value => (logy ? Math.pow(10, value) : value);
  const back = value => (view.xlog ? Math.pow(10, Number(value)) : Number(value));
  const kinds = view.lines.map(line => line.kind || kind);
  const series = view.lines.map((line, index) => ({
    name: line.name, type: kinds[index],
    data: line.points.filter(point => Number.isFinite(point[1]) && (!logy || point[1] > 0)).map(point => [Number(point[0]), up(point[1])]) }));
  const xs = series.flatMap(entry => entry.data.map(point => point[0]));
  const [low, high] = extent(xs);
  const [bottom, top] = extent(series.flatMap(entry => entry.data.map(point => point[1])));
  const integral = xs.length > 0 && !view.xlog && xs.every(Number.isInteger);
  let ticks = 8;
  let span = {};
  if (view.xlog && Number.isFinite(high - low)) ticks = Math.max(1, Math.round(high - low));
  else if (integral && high > low && kind !== "bar") {
    const step = Math.max(1, Math.ceil((high - low) / 8));
    ticks = Math.ceil((high - low) / step);
    span = { min: low, max: low + ticks * step };
  }
  const markers = markerSize(scatterPoints(view.lines, kind), view.settings);
  const pad = bottom === top && Number.isFinite(top) ? (Math.abs(top) * 0.1 || 1) : 0;
  const marks = (view.marks || []).map(mark => (typeof mark === "number" ? { x: mark } : mark));
  const shownMarks = marks.filter((_, index) => index % (Math.ceil(marks.length / 60) || 1) === 0);
  const labelled = Math.ceil(shownMarks.length / 12) || 1;
  const noteOf = value => {
    if (!view.turns) return "";
    const found = marks.filter(mark => mark.text && mark.x <= value).pop();
    return found ? ` · ${found.text}` : "";
  };
  return {
    chart: { type: kinds.some(item => item !== kind) ? "line" : kind, height: view.height, background: "transparent", fontFamily: "inherit",
             group: view.group || undefined, id: view.group ? view.uid : undefined,
             foreColor: dark ? "#b7bcc4" : "#4a4f57", animations: { enabled: false },
             zoom: { enabled: kind !== "bar", type: kind === "scatter" ? "xy" : "x", autoScaleYaxis: kind !== "scatter" },
             toolbar: { show: true, offsetY: -4, tools: { download: true, selection: true, zoom: true, zoomin: true, zoomout: true, pan: true, reset: true } } },
    series,
    colors: view.lines.map(line => line.color),
    stroke: { width: kinds.map(item => (item === "scatter" ? 0 : settings.width)), curve: settings.curve, dashArray: view.lines.map(line => (line.dashed ? 5 : 0)) },
    markers: { size: kinds.map(item => (item === "scatter" ? markers : settings.dots)), hover: { size: Math.max(4, markers) } },
    plotOptions: { bar: { columnWidth: "90%" } },
    dataLabels: { enabled: false },
    legend: { position: "top", horizontalAlign: "left", showForSingleSeries: true, onItemClick: { toggleDataSeries: true } },
    grid: { show: settings.grid, borderColor: dark ? "#2d3238" : "#e3e6ea" },
    xaxis: { type: "numeric", ...span, title: { text: view.xlabels === false ? "" : (view.xlabel || "") }, tickAmount: ticks,
             labels: { show: view.xlabels !== false, formatter: value => (integral ? String(Math.round(back(value))) : tick(back(value))),
                       rotate: 0, hideOverlappingLabels: true },
             tooltip: { enabled: false } },
    yaxis: { title: { text: view.ylabel || "" }, labels: { formatter: value => tick(down(Number(value))), minWidth: view.ywidth || undefined },
             ...(pad ? { min: bottom - pad, max: top + pad } : {}) },
    tooltip: { shared: kind !== "scatter", intersect: kind === "scatter", theme: dark ? "dark" : "light",
               x: { formatter: value => `${view.xlabel || "x"} ${fmt(back(value))}${noteOf(back(value))}` }, y: { formatter: value => fmt(down(Number(value))) } },
    annotations: { yaxis: (view.ylines || []).filter(y => !logy || y > 0).map(y => ({ y: up(y), borderColor: dark ? "#868d97" : "#7d838d", strokeDashArray: 4 })),
                   xaxis: shownMarks.map((mark, index) => ({
      x: mark.x, borderColor: dark ? "#4a515a" : "#cfd4da", strokeDashArray: 3,
      ...(mark.text && index % labelled === 0 ? { label: { text: mark.text, position: "top", orientation: "horizontal", borderWidth: 0, offsetY: -2,
                                                           style: { background: "transparent", color: dark ? "#868d97" : "#7d838d", fontSize: "10px", fontFamily: "inherit" } } } : {}) })) },
    theme: { mode: dark ? "dark" : "light" },
  };
}

const Chart = {
  props: { lines: { type: Array, default: () => [] }, title: String, xlabel: String, ylabel: String, logy: Boolean,
           marks: { type: Array, default: () => [] }, height: { type: Number, default: 260 }, expand: String, kind: { type: String, default: "line" },
           xlog: Boolean, turns: Boolean, settings: { type: Object, default: () => ({}) },
           group: { type: String, default: "" }, xlabels: { type: Boolean, default: true }, ylines: { type: Array, default: () => [] },
           ywidth: { type: Number, default: 0 } },
  data() { return { chart: null, uid: `chart-${Math.random().toString(36).slice(2, 10)}` }; },
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
  methods: {
    download() {
      if (!this.chart) return;
      const name = (this.title || this.ylabel || "chart").replace(/[^\w.-]+/g, "_");
      this.chart.dataURI().then(({ imgURI }) => {
        const link = document.createElement("a");
        link.href = imgURI;
        link.download = `${name}.png`;
        link.click();
      });
    },
  },
  template: `
    <div class="chart">
      <div class="chart-head" v-if="title || expand">
        <span class="chart-title">{{ title }}</span>
        <a v-if="expand" class="small chart-save" :href="expand" title="open this chart in a large view with its settings">expand</a>
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

const Names = {
  props: { items: { type: Array, default: () => [] }, limit: { type: Number, default: 12 } },
  data() { return { all: false }; },
  computed: { shown() { return this.all || this.items.length <= this.limit ? this.items : this.items.slice(0, this.limit); } },
  watch: { items() { this.all = false; } },
  template: `
    <div class="names">
      <span class="chip" v-for="(name, index) in shown" :key="index" :title="name">{{ name }}</span>
      <button class="tiny" v-if="items.length > limit" @click="all = !all">{{ all ? 'fewer' : '+' + (items.length - limit) + ' more' }}</button>
    </div>`,
};

const CHAR = 6.7, LINE = 15, PIN = 14, PIN_SPACE = 30, BOX_PAD = 8, GAP_X = 96, GAP_Y = 26, EDGE = 26, MAX_TEXT = 46;

function wireColor(name) {
  if (!name) return "";
  let hash = 0;
  for (const char of name) hash = (hash * 31 + char.charCodeAt(0)) % 1000003;
  return PALETTE[hash % PALETTE.length];
}

function effectText(value) {
  if (value && typeof value === "object" && "times" in value) return `×${value.times}`;
  if (value && typeof value === "object" && "plus" in value) return value.plus < 0 ? `−${Math.abs(value.plus)}` : `+${value.plus}`;
  return typeof value === "object" && value !== null ? JSON.stringify(value) : String(value);
}

function fitText(text, chars) { return text.length > chars ? text.slice(0, chars - 1) + "…" : text; }

function unique(items) { return [...new Set(items)]; }

function wireLabel(wire, widths) { return wire ? (widths[wire] ? `${wire} [${widths[wire]}]` : wire) : ""; }

function chainGraph(items, key, dashed) {
  const boxes = items.map((item, index) => ({
    name: `${key}/${index}`, kind: item.kind || "torch", column: index, item, inputs: [""], outputs: [""],
    lines: [`${item.name}: ${item.class}${item.summary ? `(${item.summary})` : ""}`, ...(item.parameters ? [paramCount(item.parameters)] : [])],
    note: [`${item.class}${item.summary ? `(${item.summary})` : ""}`, item.shapes ? `${item.shapes[0] || "?"} → ${item.shapes[1]}` : "", paramCount(item.parameters)].filter(Boolean),
    blocks: item.children || [], chain: item.class === "Sequential" }));
  const arrows = [];
  boxes.forEach((box, index) => {
    const previous = boxes[index - 1];
    if (!previous) arrows.push({ source: "in:", target: box.name, wire: "", label: box.item.shapes ? box.item.shapes[0] || "" : "", dashed });
    else arrows.push({ source: previous.name, target: box.name, wire: "", label: previous.item.shapes ? previous.item.shapes[1] : "", dashed });
  });
  const last = boxes[boxes.length - 1];
  if (last) arrows.push({ source: last.name, target: "out:", wire: "", label: last.item.shapes ? last.item.shapes[1] : "", dashed });
  return { boxes, arrows };
}

function modelGraph(models, label) {
  const model = models[label];
  if (!model) return null;
  const boxes = (model.boxes || []).filter(box => NODE_KINDS.has(box.kind)).map(box => modelBox(box, models));
  const names = new Set(boxes.map(box => box.name));
  const arrows = (model.arrows || [])
    .filter(([source, target]) => (names.has(source) || source.startsWith("in:")) && (names.has(target) || target.startsWith("out:")))
    .map(([source, target, wire, dashed]) => ({ source, target, wire: wire || "", label: wireLabel(wire || "", model.widths || {}), dashed: !!dashed }));
  return { boxes, arrows };
}

function innerOf(box, key, models) {
  if (box.kind === "model" && box.open) {
    const graph = modelGraph(models, box.open);
    return graph ? { graph } : null;
  }
  if (box.blocks && box.blocks.length) return { graph: chainGraph(box.blocks, key, !box.chain) };
  const lines = [...(box.text || [])];
  for (const group of box.groups || []) {
    lines.push(group.title);
    for (const item of group.items) lines.push(`- ${item}`);
  }
  if (!lines.length) return null;
  return { lines: lines.length > 200 ? [...lines.slice(0, 199), `… and ${lines.length - 199} more`] : lines };
}

function expandable(box) {
  return !!((box.blocks && box.blocks.length) || box.open || (box.text && box.text.length) || (box.groups && box.groups.length));
}

function schematic(graph, open, prefix, models) {
  const boxes = graph.boxes.map(box => ({ ...box }));
  const arrows = graph.arrows;
  for (const box of boxes) {
    box.inputs = box.inputs || unique(arrows.filter(arrow => arrow.target === box.name).map(arrow => arrow.wire));
    box.outputs = box.outputs || unique(arrows.filter(arrow => arrow.source === box.name).map(arrow => arrow.wire));
    box.key = `${prefix}${box.name}`;
    box.shown = box.lines.map(line => fitText(line, MAX_TEXT));
    box.head = box.shown.length * LINE + 6;
    box.inner = null;
    box.body = null;
    if (open[box.key] && expandable(box)) {
      const inner = innerOf(box, box.key, models);
      if (inner && inner.graph) box.inner = schematic(inner.graph, open, `${box.key}/`, models);
      else if (inner) box.body = inner.lines;
    }
    const pinRows = Math.max(box.inputs.length, box.outputs.length);
    box.inLabel = Math.max(0, ...box.inputs.map(name => name.length)) * 6;
    box.outLabel = Math.max(0, ...box.outputs.map(name => name.length)) * 6;
    box.left = Math.max(PIN_SPACE, box.inLabel + 44);
    box.right = Math.max(PIN_SPACE, box.outLabel + 44);
    const pinWidth = box.inLabel + box.outLabel + 28;
    const headWidth = Math.max(...box.shown.map(line => line.length)) * CHAR + 2 * BOX_PAD;
    const bodyWidth = box.inner ? box.inner.width + box.left + box.right
      : (box.body ? Math.max(0, ...box.body.map(line => line.length)) * 6.2 + 2 * BOX_PAD + 12 : 0);
    box.width = Math.max(96, headWidth, pinWidth, bodyWidth);
    const content = box.inner ? Math.max(box.inner.height, pinRows * PIN)
      : (box.body ? Math.max(box.body.length * 13 + 4, pinRows * PIN) : pinRows * PIN);
    box.height = box.head + content + BOX_PAD;
  }
  const ranks = unique(boxes.map(box => box.column)).sort((a, b) => a - b);
  const columns = ranks.map(rank => boxes.filter(box => box.column === rank));
  const widths = columns.map(items => Math.max(...items.map(box => box.width)));
  const stacks = columns.map(items => items.reduce((sum, box) => sum + box.height, 0) + GAP_Y * (items.length - 1));
  const tallest = Math.max(0, ...stacks);
  let cursor = 0;
  columns.forEach((items, index) => {
    let top = (tallest - stacks[index]) / 2;
    for (const box of items) { box.x = cursor; box.y = top; box.rank = index; box.width = widths[index]; top += box.height + GAP_Y; }
    cursor += widths[index] + GAP_X;
  });
  const lanes = laneCounts(boxes, arrows, tallest / 2, ranks.length - 1);
  const laneTop = lanes.top ? 10 + 6 * lanes.top : 0, laneBottom = lanes.bottom ? 10 + 6 * lanes.bottom : 0;
  for (const box of boxes) box.y += laneTop;
  return { width: Math.max(0, cursor - GAP_X), height: tallest + laneTop + laneBottom, top: laneTop, bottom: laneBottom, boxes, arrows };
}

function laneCounts(boxes, arrows, middle, last) {
  const byName = Object.fromEntries(boxes.map(box => [box.name, box]));
  const pinY = (box, side, wire) => box.y + box.head + PIN / 2 + Math.max(0, (side === "in" ? box.inputs : box.outputs).indexOf(wire)) * PIN;
  const counts = { top: 0, bottom: 0 };
  for (const arrow of arrows) {
    const source = byName[arrow.source], target = byName[arrow.target];
    const a = source ? { rank: source.rank, y: pinY(source, "out", arrow.wire) } : { rank: -1, y: target ? pinY(target, "in", arrow.wire) : 0 };
    const b = target ? { rank: target.rank, y: pinY(target, "in", arrow.wire) } : { rank: Infinity, y: a.y };
    if (!skipping(a, b, last)) continue;
    counts[(a.y + b.y) / 2 <= middle ? "top" : "bottom"] += 1;
  }
  return counts;
}

function pinAt(box, side, index) {
  const y = box.y + box.head + PIN / 2 + index * PIN;
  return side === "in" ? { x: box.x - 8, y, rank: box.rank } : { x: box.x + box.width + 8, y, rank: box.rank };
}

function endOf(name, side, placed, outer, layout, wire) {
  const box = placed[name];
  if (box) {
    const pins = side === "out" ? box.outputs : box.inputs;
    return pinAt(box, side, Math.max(0, pins.indexOf(wire)));
  }
  const virtual = name.startsWith("in:") ? "in" : (name.startsWith("out:") ? "out" : null);
  if (!outer || !virtual) return null;
  const pins = virtual === "in" ? outer.inputs : outer.outputs;
  let index = pins.indexOf(name.slice(virtual.length + 1));
  if (index < 0) {
    const seen = unique(layout.arrows.map(arrow => (virtual === "in" ? arrow.source : arrow.target)).filter(item => item.startsWith(`${virtual}:`)));
    index = Math.min(Math.max(0, seen.indexOf(name)), Math.max(0, pins.length - 1));
  }
  const y = outer.y + outer.head + PIN / 2 + index * PIN;
  return virtual === "in" ? { x: outer.x + outer.inLabel + 10, y, rank: -1 } : { x: outer.x + outer.width - outer.outLabel - 10, y, rank: Infinity };
}

function skipping(a, b, last) {
  if (Number.isFinite(a.rank) && Number.isFinite(b.rank)) return b.rank - a.rank > 1;
  return (a.rank === -1 && b.rank > 0) || (b.rank === Infinity && a.rank < last);
}

function channels(wires) {
  const gaps = {};
  for (const wire of wires) (gaps[wire.a.rank] = gaps[wire.a.rank] || []).push(wire);
  for (const group of Object.values(gaps)) {
    const bundles = [];
    const byPin = {};
    for (const wire of group) {
      const pin = `${wire.a.x},${wire.a.y}`;
      if (!byPin[pin]) { byPin[pin] = { wires: [] }; bundles.push(byPin[pin]); }
      byPin[pin].wires.push(wire);
    }
    for (const bundle of bundles) bundle.y = Math.min(...bundle.wires.map(wire => wire.b.y));
    bundles.sort((first, second) => first.y - second.y || first.wires[0].a.y - second.wires[0].a.y);
    bundles.forEach((bundle, index) => { for (const wire of bundle.wires) wire.share = (index + 1) / (bundles.length + 1); });
  }
}

function labelAt(label, x1, x2, y) {
  const width = label.length * 6.2 + 6;
  return x2 - x1 >= width + 8 ? { lx: (x1 + x2 - width) / 2 + 3, ly: y - 5 } : null;
}

function channel(wire, bend) {
  const { a, b } = wire;
  const own = a.x + (b.x - a.x) * (wire.share || 0.5);
  const xm = typeof bend === "number" ? Math.min(b.x - 6, Math.max(a.x + 6, bend)) : own;
  const label = wire.arrow.label || "";
  const placed = labelAt(label, a.x, xm, a.y) || labelAt(label, xm, b.x, b.y) || { lx: xm + 4, ly: (a.y + b.y) / 2 };
  return { path: `M${a.x} ${a.y} H${xm} V${b.y} H${b.x}`, xm, y1: a.y, y2: b.y, ...placed };
}

function detour(wire, middle, top, height, lanes) {
  const { a, b } = wire;
  const above = (a.y + b.y) / 2 <= middle;
  const lane = above ? (lanes.top += 1) : (lanes.bottom += 1);
  const y = above ? top - 4 - 6 * lane : top + height + 4 + 6 * lane;
  const out = a.x + 8 + 5 * lane, back = b.x - 8 - 5 * lane;
  const width = (wire.arrow.label || "").length * 6.2 + 6;
  return { path: `M${a.x} ${a.y} H${out} V${y} H${back} V${b.y} H${b.x}`, lx: (out + back - width) / 2 + 3, ly: y - 5 };
}

function render(layout, ox, oy, outer, out, level, state) {
  const placed = {};
  for (const box of layout.boxes) {
    const shift = state.moved[box.key] || { dx: 0, dy: 0 };
    const item = { ...box, x: ox + box.x + shift.dx, y: oy + box.y + shift.dy, level };
    placed[box.name] = item;
    out.boxes.push(item);
    if (box.inner) render(box.inner, item.x + item.left + (item.width - item.left - item.right - box.inner.width) / 2, item.y + box.head, item, out, level + 1, state);
  }
  const wires = [];
  const last = Math.max(-1, ...layout.boxes.map(box => box.rank));
  layout.arrows.forEach((arrow, position) => {
    const a = endOf(arrow.source, "out", placed, outer, layout, arrow.wire);
    const b = endOf(arrow.target, "in", placed, outer, layout, arrow.wire);
    if (a && b) wires.push({ key: `${level}/${position}/${arrow.source}>${arrow.target}`, a, b, arrow, direct: b.x > a.x + 12 && !skipping(a, b, last) });
  });
  channels(wires.filter(wire => wire.direct));
  const lanes = { top: 0, bottom: 0 };
  const content = { top: oy + layout.top, height: layout.height - layout.top - layout.bottom };
  const shown = new Set();
  for (const wire of wires) {
    const routed = wire.direct ? channel(wire, state.bends[wire.key]) : detour(wire, content.top + content.height / 2, content.top, content.height, lanes);
    const mark = `${wire.a.x},${wire.a.y}|${wire.arrow.label}`;
    const label = wire.arrow.label && !shown.has(mark) ? wire.arrow.label : "";
    shown.add(mark);
    out.wires.push({ key: wire.key, ...routed, label, dashed: wire.arrow.dashed, color: wireColor(wire.arrow.wire) });
  }
}

const Diagram = {
  props: { boxes: { type: Array, default: () => [] }, arrows: { type: Array, default: () => [] }, widths: { type: Object, default: () => ({}) },
           models: { type: Object, default: () => ({}) }, title: String, store: { type: String, default: "" }, expand: { type: String, default: "" },
           modal: Boolean, height: { type: Number, default: 0 } },
  emits: ["close"],
  data() {
    return { hover: null, open: {}, moved: {}, bends: {}, zoom: 1, pan: { x: 0, y: 0 }, fitted: false, drag: null,
             mouse: { x: 0, y: 0 }, marker: `arrow-${Math.random().toString(36).slice(2, 8)}` };
  },
  watch: {
    store() { this.load(); },
    open() { this.persist(); },
    moved() { this.persist(); },
    bends() { this.persist(); },
    zoom() { this.persist(); },
    pan() { this.persist(); },
  },
  computed: {
    storeKey() { return this.store ? `kalfa-board-diagram:${this.store}` : ""; },
    graph() {
      return { boxes: this.boxes,
               arrows: this.arrows.map(([source, target, wire, dashed]) => ({ source, target, wire: wire || "", label: wireLabel(wire || "", this.widths), dashed: !!dashed })) };
    },
    layout() {
      const inner = schematic(this.graph, this.open, "", this.models);
      const out = { boxes: [], wires: [], width: inner.width + 2 * EDGE, height: inner.height + 2 * EDGE, top: inner.top, bottom: inner.bottom };
      render(inner, EDGE, EDGE, null, out, 0, { moved: this.moved, bends: this.bends });
      return out;
    },
    bounds() {
      const boxes = this.layout.boxes;
      if (!boxes.length) return { x: 0, y: 0, width: 200, height: 100 };
      const x = Math.min(...boxes.map(box => box.x)) - 14, y = Math.min(...boxes.map(box => box.y)) - 14 - this.layout.top;
      return { x, y, width: Math.max(...boxes.map(box => box.x + box.width)) + 14 - x,
               height: Math.max(...boxes.map(box => box.y + box.height)) + 14 + this.layout.bottom - y };
    },
    viewHeight() { return this.height || Math.min(720, Math.max(300, this.layout.height)); },
    canvas() { return `translate(${this.pan.x} ${this.pan.y}) scale(${this.zoom})`; },
    kinds() { return [...new Set(this.boxes.map(box => box.kind))]; },
    tooltipStyle() {
      const lines = this.hoverNote;
      const width = Math.max(8, ...lines.map(line => line.length)) * 6.6 + 20;
      const height = lines.length * 17 + 14;
      const left = this.mouse.x + 16 + width > window.innerWidth ? this.mouse.x - 16 - width : this.mouse.x + 16;
      const top = this.mouse.y + 16 + height > window.innerHeight ? this.mouse.y - 16 - height : this.mouse.y + 16;
      return { position: "fixed", left: `${Math.max(4, left)}px`, top: `${Math.max(4, top)}px` };
    },
    hoverNote() {
      if (this.drag) return [];
      const note = (this.hover && this.hover.note) || [];
      if (note.length <= 14) return note;
      return [...note.slice(0, 12), `and ${note.length - 12} more${expandable(this.hover) ? ", click the block to open it" : ""}`];
    },
  },
  mounted() {
    this.load();
    if (!this.fitted) this.fit();
  },
  methods: {
    color(kind) { return KIND_COLORS[kind] || "#7a7a7a"; },
    kindName(kind) { return KIND_NAMES[kind] || kind; },
    palette() { return PALETTE; },
    markerFor(color) { const index = PALETTE.indexOf(color); return index < 0 ? this.marker : `${this.marker}-${index}`; },
    clickable(box) { return expandable(box); },
    pinY(box, index) { return box.y + box.head + PIN / 2 + index * PIN; },
    toggle(box) { if (expandable(box)) this.open = { ...this.open, [box.key]: !this.open[box.key] }; },
    enter(box) { this.hover = box.inner || box.body ? null : box; },
    onMove(event) { this.mouse = { x: event.clientX, y: event.clientY }; },
    saved() {
      try { return this.storeKey ? JSON.parse(localStorage.getItem(this.storeKey) || "null") : null; } catch (error) { return null; }
    },
    load() {
      const saved = this.saved() || {};
      const view = (saved.views || {})[this.modal ? "modal" : "inline"];
      this.open = saved.open || {};
      this.moved = saved.moved || {};
      this.bends = saved.bends || {};
      this.zoom = view ? view.zoom : 1;
      this.pan = view ? view.pan : { x: 0, y: 0 };
      this.fitted = !!view;
      if (!view) this.$nextTick(() => this.fit());
    },
    persist() {
      if (!this.storeKey) return;
      const views = { ...((this.saved() || {}).views || {}), [this.modal ? "modal" : "inline"]: { zoom: this.zoom, pan: this.pan } };
      const state = { open: this.open, moved: this.moved, bends: this.bends, views };
      try { localStorage.setItem(this.storeKey, JSON.stringify(state)); } catch (error) { console.warn("storage unavailable", error); }
    },
    reset() {
      this.open = {}; this.moved = {}; this.bends = {};
      this.$nextTick(() => this.fit());
    },
    fit() {
      const svg = this.$refs.svg;
      const width = svg ? svg.clientWidth : 800, height = this.viewHeight, bounds = this.bounds;
      const zoom = Math.min(1.5, (width - 24) / bounds.width, (height - 24) / bounds.height);
      this.zoom = zoom;
      this.pan = { x: (width - bounds.width * zoom) / 2 - bounds.x * zoom, y: (height - bounds.height * zoom) / 2 - bounds.y * zoom };
      this.fitted = true;
    },
    onWheel(event) {
      const rect = this.$refs.svg.getBoundingClientRect();
      const cx = event.clientX - rect.left, cy = event.clientY - rect.top;
      const zoom = Math.min(4, Math.max(0.15, this.zoom * Math.exp(-event.deltaY * 0.0012)));
      const ratio = zoom / this.zoom;
      this.pan = { x: cx - (cx - this.pan.x) * ratio, y: cy - (cy - this.pan.y) * ratio };
      this.zoom = zoom;
    },
    startDrag(kind, target, event) {
      if (event.button !== 0) return;
      const drag = { kind, target, startX: event.clientX, startY: event.clientY, went: false, pan: { ...this.pan } };
      if (kind === "box") drag.from = { ...(this.moved[target.key] || { dx: 0, dy: 0 }) };
      if (kind === "bend") drag.from = target.xm;
      this.drag = drag;
      window.addEventListener("mousemove", this.onDrag);
      window.addEventListener("mouseup", this.endDrag);
    },
    onDrag(event) {
      const drag = this.drag;
      if (!drag) return;
      const dx = event.clientX - drag.startX, dy = event.clientY - drag.startY;
      if (Math.abs(dx) + Math.abs(dy) > 3) drag.went = true;
      if (!drag.went) return;
      if (drag.kind === "pan") this.pan = { x: drag.pan.x + dx, y: drag.pan.y + dy };
      else if (drag.kind === "box") this.moved = { ...this.moved, [drag.target.key]: { dx: drag.from.dx + dx / this.zoom, dy: drag.from.dy + dy / this.zoom } };
      else if (drag.kind === "bend") this.bends = { ...this.bends, [drag.target.key]: drag.from + dx / this.zoom };
    },
    endDrag() {
      window.removeEventListener("mousemove", this.onDrag);
      window.removeEventListener("mouseup", this.endDrag);
      const drag = this.drag;
      this.drag = null;
      if (drag && drag.kind === "box" && !drag.went) this.toggle(drag.target);
    },
    save() {
      const svg = this.$refs.svg;
      const group = svg.querySelector("g.canvas");
      const bounds = this.bounds;
      const before = { transform: group.getAttribute("transform"), viewBox: svg.getAttribute("viewBox") };
      group.setAttribute("transform", `translate(${-bounds.x} ${-bounds.y})`);
      svg.setAttribute("viewBox", `0 0 ${bounds.width} ${bounds.height}`);
      try {
        downloadSvg(svg, this.title, bounds.width, bounds.height, getComputedStyle(svg).backgroundColor);
      } finally {
        group.setAttribute("transform", before.transform);
        if (before.viewBox === null) svg.removeAttribute("viewBox"); else svg.setAttribute("viewBox", before.viewBox);
      }
    },
  },
  template: "#diagram-template",
};

const Parallel = {
  props: { axes: { type: Array, default: () => [] }, rows: { type: Array, default: () => [] },
           brush: { type: Object, default: () => ({}) }, height: { type: Number, default: 380 } },
  emits: ["update:brush", "pick"],
  data() { return { hover: null, drag: null }; },
  computed: {
    layout() {
      const width = 1000, left = 70, right = 70, top = 30, bottom = 54;
      const plot = this.height - top - bottom;
      const gaps = Math.max(this.axes.length - 1, 1);
      return { width, left, right, top, bottom, plot,
               columns: this.axes.map((axis, index) => {
                 const x = this.axes.length > 1 ? left + index * (width - left - right) / gaps : width / 2;
                 return { axis, key: axis.key, label: axis.label || axis.key, x,
                          ticks: axisTicks(axis).map(item => ({ ...item, y: top + (1 - item.at) * plot })) };
               }) };
    },
    lines() {
      const { columns, top, plot } = this.layout;
      const drawn = this.rows.map(row => {
        const seats = columns.map(column => {
          const at = axisNorm(column.axis, row.values[column.key]);
          return at === null ? null : { x: column.x, y: top + (1 - at) * plot };
        }).filter(Boolean);
        return { ...row, polyline: seats.map(seat => `${seat.x.toFixed(1)},${seat.y.toFixed(1)}`).join(" "),
                 seats, end: seats[seats.length - 1] };
      }).filter(row => row.seats.length > 1);
      return drawn.sort((first, second) => (second.share === null ? 2 : second.share) - (first.share === null ? 2 : first.share));
    },
    tags() {
      return this.lines.filter(row => row.place && row.place <= 3 && row.end && row.scored);
    },
    bands() {
      const { top, plot, columns } = this.layout;
      return Object.entries(this.brush).map(([key, range]) => {
        const column = columns.find(item => item.key === key);
        return column ? { key, x: column.x - 6, y: top + (1 - range[1]) * plot,
                          height: Math.max(3, (range[1] - range[0]) * plot) } : null;
      }).filter(Boolean);
    },
    detail() {
      if (!this.hover) return [];
      return this.layout.columns.map(column => ({ key: column.label, value: fmt(this.hover.values[column.key]) }));
    },
  },
  methods: {
    fmt,
    colorOf(row) { return row.share === null ? (darkMode() ? "#5a6067" : "#c8ccd2") : rampColor(row.share); },
    widthOf(row) {
      if (this.hover) return this.hover.path === row.path ? 3.4 : 1.3;
      if (!row.scored) return 1.2;
      return row.place === 1 ? 3 : 1.8;
    },
    fadeOf(row) {
      if (this.hover) return this.hover.path === row.path ? 1 : 0.12;
      return row.scored ? 0.9 : 0.3;
    },
    at(event) {
      const svg = this.$refs.svg;
      const seat = svg.createSVGPoint();
      seat.x = event.clientX;
      seat.y = event.clientY;
      const local = seat.matrixTransform(svg.getScreenCTM().inverse());
      return Math.min(1, Math.max(0, 1 - (local.y - this.layout.top) / this.layout.plot));
    },
    startBrush(column, event) {
      this.drag = { key: column.key, from: this.at(event), went: false };
      window.addEventListener("mousemove", this.onBrush);
      window.addEventListener("mouseup", this.endBrush);
    },
    onBrush(event) {
      if (!this.drag) return;
      const to = this.at(event);
      const range = [Math.min(this.drag.from, to), Math.max(this.drag.from, to)];
      if (range[1] - range[0] < 0.015) return;
      this.drag.went = true;
      this.$emit("update:brush", { ...this.brush, [this.drag.key]: range });
    },
    endBrush() {
      window.removeEventListener("mousemove", this.onBrush);
      window.removeEventListener("mouseup", this.endBrush);
      const drag = this.drag;
      this.drag = null;
      if (!drag || drag.went) return;
      const next = { ...this.brush };
      delete next[drag.key];
      this.$emit("update:brush", next);
    },
    save() {
      const svg = this.$refs.svg;
      downloadSvg(svg, "parallel", this.layout.width, this.height, getComputedStyle(svg).backgroundColor);
    },
  },
  template: "#parallel-template",
};

const Strip = {
  props: { levels: { type: Array, default: () => [] }, low: Number, high: Number, monitor: String },
  data() { return { hover: null }; },
  computed: {
    layout() {
      const width = 320, left = 52, right = 16, top = 8, step = 24;
      const height = top + this.levels.length * step + 22;
      const place = value => left + (this.high > this.low ? (value - this.low) / (this.high - this.low) : 0.5) * (width - left - right);
      return { width, height, left, right, top, step, place,
               axis: [this.low, (this.low + this.high) / 2, this.high].map(value => ({ x: place(value), text: tick(value) })),
               rows: this.levels.map((level, index) => ({
                 ...level, y: top + index * step + step / 2,
                 dots: level.values.map(value => ({ x: place(value), value })),
                 middle: level.values.length ? place(median(level.values)) : null })) };
    },
  },
  methods: { fmt },
  template: `
    <div class="strip">
      <svg :viewBox="'0 0 ' + layout.width + ' ' + layout.height" preserveAspectRatio="xMidYMid meet" @mouseleave="hover = null">
        <g v-for="row in layout.rows" :key="row.label">
          <text class="strip-label" :x="layout.left - 8" :y="row.y + 3">{{ row.label }}</text>
          <line class="strip-rule" :x1="layout.left" :x2="layout.width - layout.right" :y1="row.y" :y2="row.y"></line>
          <line class="strip-median" v-if="row.middle !== null" :x1="row.middle" :x2="row.middle" :y1="row.y - 7" :y2="row.y + 7"></line>
          <circle v-for="(dot, index) in row.dots" :key="index" :cx="dot.x" :cy="row.y" r="4"
                  :class="{picked: hover && hover.value === dot.value && hover.label === row.label}"
                  @mouseenter="hover = {value: dot.value, label: row.label, n: row.values.length}"></circle>
          <text class="strip-n" :x="layout.width - layout.right + 2" :y="row.y + 3">{{ row.values.length }}</text>
        </g>
        <g class="strip-axis">
          <text v-for="(item, index) in layout.axis" :key="index" :x="item.x" :y="layout.height - 6">{{ item.text }}</text>
        </g>
      </svg>
      <div class="strip-note">
        <template v-if="hover">{{ hover.label }} · {{ monitor }} <b>{{ fmt(hover.value) }}</b></template>
        <template v-else>dot per point, tick is the median, right number is n</template>
      </div>
    </div>`,
};

const app = Vue.createApp({
  components: { chart: Chart, spark: Spark, diagram: Diagram, names: Names, parallel: Parallel, strip: Strip },
  data() {
    return {
      route: parseHash(location.hash),
      tree: { root: "", groups: {} }, filter: "", collapsed: {}, opened: {}, refreshed: "",
      record: null, history: { lines: [], offset: 0 }, steps: { lines: [], offset: 0 },
      logs: { name: "", lines: [], total: 0 }, texts: {}, describeText: null, showModuleText: false,
      sweep: null, overlay: {}, diff: null, modalHeight: 520, liveBoard: { live: [], recent: [] }, brush: {},
      tableRows: [], tableLoaded: false, tableFilter: "", compareData: null, predictionsData: null, prepData: null,
      whereDraft: "",
      filesData: null, fileView: null, eventsData: null, playing: false, frame: 0, player: null,
      refresh: (() => { try { return localStorage.getItem("kalfa-board-refresh") || "realtime"; } catch (error) { return "realtime"; } })(),
      source: null, timer: null, treeTimer: null, connected: false, queue: {}, plot: defaultPlot(), busy: 0,
      booted: false, entering: null,
      sidebar: (() => { try { return localStorage.getItem("kalfa-board-sidebar") !== "closed"; } catch (error) { return true; } })(),
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
      const points = this.predictionsData.sample.map(row => [row[pair.target], row[pair.pred]]).filter(point => Number.isFinite(point[0]) && Number.isFinite(point[1]));
      const [low, high] = extent(points.flatMap(point => point));
      const diagonal = Number.isFinite(low) && Number.isFinite(high) && high > low ? [{ name: "y = x", color: "#9aa0a8", kind: "line", dashed: true, points: [[low, low], [high, high]] }] : [];
      return [{ name: `${pair.pred} against ${pair.target}`, color: PALETTE[0], points }, ...diagonal];
    },
    sampleAll() { return this.route.params.sample === "all"; },
    whereFilter() { return this.route.params.where || ""; },
    histogramLines() {
      if (!this.pairInfo) return [];
      const { edges, counts } = this.pairInfo.histogram;
      return [{ name: "residual", color: PALETTE[1], points: counts.map((value, index) => [(edges[index] + edges[index + 1]) / 2, value]) }];
    },
    distributionLines() {
      const pair = this.pairInfo;
      if (!pair || !pair.distribution || !pair.distribution.edges || pair.distribution.edges.length < 2) return null;
      const { edges, data, pred } = pair.distribution;
      const centers = data.map((value, index) => (edges[index] + edges[index + 1]) / 2);
      const top = [{ name: pair.target, color: PALETTE[0], kind: "bar", points: centers.map((x, index) => [x, data[index]]) },
                   { name: pair.pred, color: PALETTE[1], kind: "bar", points: centers.map((x, index) => [x, pred[index]]) }];
      const ratio = [{ name: `${pair.pred} / ${pair.target}`, color: PALETTE[1], kind: "scatter",
                       points: centers.map((x, index) => [x, data[index] > 0 ? pred[index] / data[index] : NaN]).filter(point => Number.isFinite(point[1])) }];
      return { top, ratio };
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
    sweepSpace() { return (this.sweep && this.sweep.space) || {}; },
    sweepUnit() { return (this.sweep && this.sweep.unit) || "turn"; },
    tableUnit() {
      const units = new Set(this.tableRows.map(row => row.unit));
      return units.size === 1 && units.has("epoch") ? "epoch" : "turn";
    },
    compareUnit() {
      if (!this.compareData || this.compareData.missing) return "turn";
      return ["a", "b"].every(which => (this.compareData[which].record.manifest || {}).turn === "epoch") ? "epoch" : "turn";
    },
    expandedHasPoints() {
      const view = this.expanded;
      if (!view) return false;
      return (view.panels || [view]).some(panel => panel.kind === "scatter" || (panel.lines || []).some(line => line.kind === "scatter"));
    },
    objectiveName() { return (this.sweep && this.sweep.objective && this.sweep.objective.monitor) || "objective"; },
    objectiveMode() { return (this.sweep && this.sweep.objective && this.sweep.objective.mode) || "min"; },
    scoreExtent() {
      const values = this.sweep ? this.sweep.points.filter(point => point.objective).map(point => point.objective.value) : [];
      return values.length ? extent(values) : [0, 1];
    },
    scoreAxis() {
      const [low, high] = this.scoreExtent;
      return { key: "objective", kind: "range", log: false, low, high, label: this.objectiveName };
    },
    paramAxes() { return Object.entries(this.sweepSpace).map(([key, axis]) => ({ ...axis, key, label: key })); },
    rampSteps() { return darkMode() ? RAMP_DARK : RAMP_LIGHT; },
    parallelAxes() { return [...this.paramAxes, this.scoreAxis]; },
    sweepRows() {
      if (!this.sweep) return [];
      const sign = this.objectiveMode === "min" ? 1 : -1;
      const ranked = this.sweep.points.filter(point => point.objective)
        .sort((first, second) => sign * (first.objective.value - second.objective.value));
      const order = new Map(ranked.map((point, index) => [point.path, index]));
      return this.sweep.points.map(point => {
        const score = point.objective ? point.objective.value : null;
        const state = this.stateOf(point);
        const at = order.get(point.path);
        return { path: point.path, id: point.id, label: point.path.split("/").pop(), state, turns: point.turns,
                 score, scored: score !== null, place: at === undefined ? null : at + 1,
                 share: at === undefined ? null : (ranked.length > 1 ? at / (ranked.length - 1) : 0),
                 values: { ...point.values, objective: score, turns: point.turns, id: point.id, state } };
      });
    },
    brushed() {
      const keys = Object.keys(this.brush);
      if (!keys.length) return this.sweepRows;
      const axes = Object.fromEntries(this.parallelAxes.map(axis => [axis.key, axis]));
      return this.sweepRows.filter(row => keys.every(key => {
        const at = axisNorm(axes[key], row.values[key]);
        return at !== null && at >= this.brush[key][0] - 1e-6 && at <= this.brush[key][1] + 1e-6;
      }));
    },
    brushNotes() {
      const axes = Object.fromEntries(this.parallelAxes.map(axis => [axis.key, axis]));
      return Object.entries(this.brush).map(([key, range]) => {
        const axis = axes[key];
        if (!axis) return { key, text: key };
        if (axis.kind === "choices") {
          const kept = axis.values.filter(value => {
            const at = axisNorm(axis, value);
            return at >= range[0] - 1e-6 && at <= range[1] + 1e-6;
          });
          return { key, text: `${key} = ${kept.map(fmt).join(", ") || "none"}` };
        }
        const low = axis.log ? Math.log10(axis.low) : axis.low;
        const high = axis.log ? Math.log10(axis.high) : axis.high;
        const back = at => (axis.log ? Math.pow(10, low + at * (high - low)) : low + at * (high - low));
        return { key, text: `${key} ${tick(back(range[0]))} to ${tick(back(range[1]))}` };
      });
    },
    sweepStats() {
      const rows = this.sweepRows;
      const scores = rows.filter(row => row.scored).map(row => row.score);
      const [low, high] = scores.length ? extent(scores) : [null, null];
      const count = name => rows.filter(row => row.state === name).length;
      return { total: rows.length, planned: (this.sweep && this.sweep.manifest.total) || rows.length,
               scored: scores.length, failed: count("failed"), running: count("running"), pending: count("pending"),
               low, high, spread: low && high && low !== 0 ? (high - low) / Math.abs(low) : null };
    },
    axisChoices() { return [...this.paramAxes.map(axis => axis.key), "objective", "turns", "id"]; },
    colorChoices() { return [...this.paramAxes.map(axis => axis.key), "objective", "state"]; },
    explore() {
      const params = this.route.params;
      const keys = this.paramAxes.map(axis => axis.key);
      const categorical = this.paramAxes.find(axis => axis.kind === "choices");
      return { x: params.x || keys[0] || "id", y: params.y || "objective",
               color: params.color || (categorical ? categorical.key : "state") };
    },
    exploreChart() {
      if (!this.sweep) return null;
      const { x, y, color } = this.explore;
      const axes = Object.fromEntries(this.parallelAxes.map(axis => [axis.key, axis]));
      const ready = row => typeof row.values[x] === "number" && typeof row.values[y] === "number";
      const rows = this.brushed.filter(ready);
      const xlog = !!(axes[x] && axes[x].kind === "range" && axes[x].log);
      const lines = this.colorGroups(rows, color).map(group => ({
        name: group.name, color: group.color, kind: "scatter",
        points: group.rows.map(row => [xlog ? Math.log10(row.values[x]) : row.values[x], row.values[y]]) }));
      return { lines: lines.filter(line => line.points.length), xlog, xlabel: this.axisLabel(x), ylabel: this.axisLabel(y), shown: rows.length };
    },
    paramCharts() {
      if (!this.sweep) return [];
      const rows = this.brushed.filter(row => row.scored);
      const [low, high] = this.scoreExtent;
      return this.paramAxes.map(axis => {
        if (axis.kind === "choices") {
          return { key: axis.key, shape: "strip", low, high,
                   levels: axis.values.map(value => ({ label: fmt(value),
                     values: rows.filter(row => row.values[axis.key] === value).map(row => row.score) })) };
        }
        const points = rows.filter(row => typeof row.values[axis.key] === "number")
          .map(row => [axis.log ? Math.log10(row.values[axis.key]) : row.values[axis.key], row.score]);
        return { key: axis.key, shape: "scatter", xlog: !!axis.log,
                 lines: [{ name: this.objectiveName, color: markColor(), kind: "scatter", points }] };
      });
    },
    hasFailures() { return this.sweepRows.some(row => row.state === "failed"); },
    failureMap() {
      const rows = this.sweepRows;
      return this.paramAxes.filter(axis => axis.kind === "choices").map(axis => ({
        key: axis.key,
        levels: axis.values.map(value => {
          const here = rows.filter(row => row.values[axis.key] === value);
          const failed = here.filter(row => row.state === "failed").length;
          const scored = here.filter(row => row.scored).length;
          return { label: fmt(value), total: here.length, failed, scored, other: here.length - failed - scored };
        }) }));
    },
    progressChart() {
      const rows = this.sweepRows.filter(row => row.scored && typeof row.id === "number")
        .sort((first, second) => first.id - second.id);
      if (rows.length < 2) return null;
      const pick = this.objectiveMode === "min" ? Math.min : Math.max;
      let running = null;
      const best = rows.map(row => {
        running = running === null ? row.score : pick(running, row.score);
        return [row.id, running];
      });
      return { lines: [{ name: this.objectiveName, color: markColor(), kind: "scatter", points: rows.map(row => [row.id, row.score]) },
                       { name: "best so far", color: rampColor(0), kind: "line", points: best }] };
    },
    compareCharts() {
      if (!this.compareData) return [];
      const byName = {};
      const label = which => (this.compareData[which].record.manifest && this.compareData[which].record.manifest.name) || this.compareData[which].path;
      for (const which of ["a", "b"]) {
        const series = {};
        for (const line of this.compareData[which].history) {
          for (const [key, value] of Object.entries(line)) {
            if (SKIP_KEYS.has(key) || typeof value !== "number" || key.startsWith("lr/") || key.startsWith("minimizes/") || key.startsWith("effect/")) continue;
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
    expandedDiagram() {
      const which = this.route.params.diagram;
      if (!this.record || this.isSweep) return null;
      if (which === "model" && this.currentArchitecture) return "model";
      if (which === "data" && this.pipeline) return "data";
      return null;
    },
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
      const all = this.tree.groups;
      const hit = entry => !needle || entry.name.toLowerCase().includes(needle) || entry.path.toLowerCase().includes(needle);
      const held = new Set(Object.values(all).flat().map(entry => entry.path));
      const attach = (entry, forced) => {
        const mine = forced || hit(entry);
        const children = (all[entry.path] || []).map(child => attach(child, mine));
        return { ...entry, children: mine ? children : children.filter(child => child.keep),
                 keep: mine || children.some(child => child.keep) };
      };
      return Object.entries(all).filter(([name]) => !held.has(name))
        .map(([name, entries]) => ({ name, entries: entries.map(entry => attach(entry, false)).filter(entry => entry.keep) }))
        .filter(group => group.entries.length);
    },
    state() { return this.record ? this.stateOf(this.record) : "pending"; },
    liveProgress() {
      const runs = this.liveBoard.live.filter(entry => entry.kind !== "sweep" && typeof entry.turns_total === "number" && entry.turns_total > 0);
      const done = runs.reduce((sum, entry) => sum + Math.min(entry.turn || 0, entry.turns_total), 0);
      const planned = runs.reduce((sum, entry) => sum + entry.turns_total, 0);
      return { running: this.liveBoard.live.length, done, planned, share: planned ? done / planned : null };
    },
    badge() {
      if (this.record && !this.isSweep && this.state === "failed") return "failed";
      if (this.record && !this.isSweep && ["running", "pending"].includes(this.state)) return "running";
      const live = this.liveBoard.live, latest = this.liveBoard.recent[0];
      const failedAt = latest && this.stateOf(latest) === "failed" ? Date.parse(latest.status.last_seen || "") : NaN;
      const started = Math.max(-Infinity, ...live.map(entry => Date.parse(entry.started || "")).filter(Number.isFinite));
      if (Number.isFinite(failedAt) && (!live.length || failedAt > started)) return "failed";
      return live.length ? "running" : "idle";
    },
    chrome() {
      const summary = this.liveProgress;
      const percent = summary.share === null ? "" : ` (${Math.round(100 * summary.share)}%)`;
      return { title: summary.running ? `Kalfa Board${percent}` : "Kalfa Board", badge: this.badge, share: this.badge === "running" ? summary.share : null };
    },
    logoSvg() { return iconSvg(this.chrome.badge, this.chrome.share); },
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
        if (set === "lr" || set === "effect") continue;
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
      return Object.entries(this.series).filter(([key]) => key.startsWith("lr/")).map(([key, points]) => {
        const segments = [];
        for (const [turn, value] of points) {
          const last = segments[segments.length - 1];
          if (last && last.value === value) last.to = turn;
          else segments.push({ value, from: turn, to: turn });
        }
        const shown = segments.length > 10 ? [...segments.slice(0, 4), null, ...segments.slice(-4)] : segments;
        return { key, last: points[points.length - 1][1], segments: shown, changes: segments.length };
      });
    },
    latest() {
      return Object.entries(this.series).filter(([key]) => !key.startsWith("lr/") && !key.startsWith("effect/")).map(([key, points]) => {
        const { set, name } = splitKey(key);
        let low = points[0], high = points[0];
        for (const point of points) { if (point[1] < low[1]) low = point; if (point[1] > high[1]) high = point; }
        const definition = this.definitionOf(name);
        return { key, set, name, ...definition, last: points[points.length - 1][1], min: low[1], minTurn: low[0], max: high[1], maxTurn: high[0], tail: points.slice(-60) };
      });
    },
    rulesFired() {
      return this.history.lines.filter(line => (line.rules || []).length).map(line => ({ turn: line.turn, names: line.rules.join(", ") }));
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
        const terms = definition && definition.params && typeof definition.params.terms === "object" ? this.effectiveTerms(current, definition.params.terms) : null;
        const prefix = current ? `train/${current}/` : null;
        const parts = terms ? Object.entries(terms).map(([term, weight]) => `${fmt(weight)} × ${term}`)
          : (prefix ? Object.keys(this.series).filter(key => key.startsWith(prefix)).map(key => key.slice(prefix.length)) : []);
        const rates = this.series[`lr/${name}`];
        return { name, current, uri: definition ? definition.uri : "", segments, parts, lr: rates ? rates[rates.length - 1][1] : null };
      });
    },
    ruleMarks() { return this.history.lines.filter(line => (line.rules || []).length).map(line => ({ x: line.turn, text: line.rules.join(", ") })); },
    effectRows() {
      const lines = this.history.lines;
      const targets = [...new Set(lines.flatMap(line => Object.keys(line).filter(key => key.startsWith("effect/"))))].map(key => key.slice(7));
      const relative = value => !!(value && typeof value === "object" && ("times" in value || "plus" in value));
      return targets.map(target => {
        const [owner, ...rest] = target.split(".");
        const rate = rest.join(".") === "lr" && lines.some(line => typeof line[`lr/${owner}`] === "number") ? `lr/${owner}` : null;
        if (!rate && lines.some(line => relative(line[`effect/${target}`]))) {
          const firings = lines.filter(line => relative(line[`effect/${target}`])).map(line => ({ turn: line.turn, text: effectText(line[`effect/${target}`]) }));
          return { target, written: this.initialOf(target), firings, segments: [] };
        }
        const segments = [];
        for (const line of lines) {
          const value = rate ? line[rate] : (line[`effect/${target}`] === undefined ? this.initialOf(target) : line[`effect/${target}`]);
          const shown = JSON.stringify(value === undefined ? null : value);
          const last = segments[segments.length - 1];
          if (last && last.shown === shown) last.to = line.turn;
          else segments.push({ value, shown, from: line.turn, to: line.turn });
        }
        return { target, segments, firings: [] };
      });
    },
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
        if (line.turn !== previous) marks.push({ x: line.step, text: `${this.turnLabel} ${line.turn}` });
        previous = line.turn;
      }
      return marks;
    },
    expanded() {
      const name = this.route.params.chart;
      if (!name || !this.record) return null;
      if (this.isSweep) return this.expandedSweep(name);
      if (this.tab === "steps" || (this.tab === "monitor" && name.startsWith("loss/"))) {
        const found = this.stepCharts.find(entry => entry.name === name);
        return found ? { name, lines: found.lines, xlabel: "step", marks: this.turnMarks, turns: true } : null;
      }
      if (this.tab === "predictions" && this.pairInfo) {
        const pair = this.pairInfo;
        if (name === "scatter") return { name: `${pair.pred} against ${pair.target}`, lines: this.scatterLines, kind: "scatter", xlabel: pair.target, ylabel: pair.pred, marks: [] };
        if (name === "histogram") return { name: "residual (prediction minus target)", lines: this.histogramLines, kind: "bar", xlabel: "residual", ylabel: "points", marks: [], bins: true };
        if (name === "distribution" && this.distributionLines) {
          return { name: `${pair.pred} and ${pair.target} over the same bins`, bins: true,
                   panels: [{ lines: this.distributionLines.top, kind: "bar", ylabel: "points", xlabels: false, logy: true },
                            { lines: this.distributionLines.ratio, kind: "scatter", xlabel: pair.target, ylabel: "pred / true", ylines: [1] }] };
        }
        return null;
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
      if (!label) return null;
      return { label, ...models[label], boxes: (models[label].boxes || []).map(box => modelBox(box, models)) };
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
      const sourceLines = source.uri ? callLines(source.uri, sourceParams) : [];
      boxes.push({ name: "source", kind: "source", column, row: 0, note: sourceLines, text: sourceLines,
                   lines: [source.uri ? shortUri(source.uri) : "source", ...(sourceParams.path ? [String(sourceParams.path).split("/").pop()] : []), `${count(stages[0].rows)} rows · ${stages[0].columns} columns`] });
      for (const stage of stages.slice(1)) {
        column += 1;
        const added = stage.added || [], removed = stage.removed || [];
        const change = [added.length ? `+${added.length}` : "", removed.length ? `−${removed.length}` : ""].filter(Boolean).join(" ");
        const params = stage.call ? paramsText(stage.call.params) : "";
        const call = stage.call ? callLines(stage.call.uri, stage.call.params) : [];
        const note = [`stage ${stage.stage}`, ...call, ...added.map(name => `+ ${name}`), ...removed.map(name => `− ${name}`)];
        const groups = [added.length ? { title: `+ ${added.length} columns added`, items: added } : null,
                        removed.length ? { title: `− ${removed.length} columns removed`, items: removed } : null].filter(Boolean);
        boxes.push({ name: stage.stage, kind: "transform", column, row: 0, note, text: call, groups,
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
          const detail = (perSet[set] || []).flatMap(call => callLines(call.uri, call.params));
          boxes.push({ name: `after:${set}`, kind: "transform", column, row, note: detail, text: detail,
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
      const fitText = [...Object.entries(fit.preprocessors || {}).map(([name, columns]) => `${name}: ${columns} columns`),
                       ...(data.frames || []).map(name => `frame transform ${name}`)];
      const fitGroups = [(fit.targets || []).length ? { title: "targets", items: fit.targets } : null,
                         (fit.extras || []).length ? { title: "extras", items: fit.extras } : null].filter(Boolean);
      boxes.push({ name: "fit", kind: "fit", column, row: 0, lines, text: fitText, groups: fitGroups,
                   note: (fit.targets || []).length ? ["targets", ...fit.targets] : [] });
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
      const kept = new Set(this.brushed.map(row => row.path));
      const points = this.sweep.points.filter(point => kept.has(point.path));
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
    path: { immediate: true, handler() { this.entering = this.enterRecord(); } },
    chrome: {
      immediate: true,
      handler(now, before) {
        if (before && now.title === before.title && now.badge === before.badge && now.share === before.share) return;
        document.title = now.title;
        setIcon(iconSvg(now.badge, now.share));
      },
    },
    refresh() {
      try { localStorage.setItem("kalfa-board-refresh", this.refresh); } catch (error) { console.warn("storage unavailable", error); }
      this.applyRefresh();
    },
    predictionFile() { if (this.tab === "predictions") this.loadPredictions(); },
    sampleAll() { if (this.tab === "predictions") this.loadPredictions(); },
    whereFilter: { immediate: true, handler(now) {
      this.whereDraft = now;
      if (this.tab === "predictions" && this.predictionsData) this.loadPredictions();
    } },
    "plot.bins"() { if (this.tab === "predictions" && this.predictionsData) this.loadPredictions(); },
    expanded(now, before) {
      if (!now || before) return;
      const lines = now.panels ? now.panels.flatMap(panel => panel.lines.map(line => ({ ...line, kind: line.kind || panel.kind }))) : now.lines;
      this.plot = { ...defaultPlot(), logy: !!(now.logy || this.logy), bins: this.plot.bins, markers: markerSize(scatterPoints(lines, now.kind), null) };
    },
    "route.params.a"() { if (this.page === "compare") this.loadCompare(); },
    "route.params.b"() { if (this.page === "compare") this.loadCompare(); },
    tab: { immediate: true, handler() { this.enterTab(); } },
    item() { this.enterItem(); },
    overlayPaths() { this.loadOverlay(); },
    logName() { if (this.tab === "logs") this.loadLogs(); },
  },
  methods: {
    fmt, count, ms, ago, clock, setColor, shortUri, paramsText, paramCount,
    tabLabel(name) { return TAB_LABELS[name] || name; },
    axisLabel(key) { return key === "turns" ? `${this.sweepUnit}s` : key; },
    expandedSweep(name) {
      if (name === "overlay" && this.overlayLines.length) return { name: this.sweep.objective.monitor || "objective", lines: this.overlayLines, xlabel: this.sweepUnit, marks: [] };
      if (name === "explorer" && this.exploreChart && this.exploreChart.lines.length) {
        const chart = this.exploreChart;
        return { name: `${chart.ylabel} against ${chart.xlabel}, colored by ${this.explore.color}`, lines: chart.lines, kind: "scatter",
                 xlabel: chart.xlabel, ylabel: chart.ylabel, xlog: chart.xlog, marks: [] };
      }
      if (name.startsWith("param:")) {
        const found = this.paramCharts.find(entry => entry.key === name.slice(6) && entry.shape === "scatter");
        return found ? { name: `${found.key} against ${this.objectiveName}`, lines: found.lines, kind: "scatter", xlabel: found.key, ylabel: this.objectiveName, xlog: found.xlog, marks: [] } : null;
      }
      if (name === "progress" && this.progressChart) return { name: `${this.objectiveName} over the sweep order`, lines: this.progressChart.lines, xlabel: "point", ylabel: this.objectiveName, marks: [] };
      return null;
    },
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
    chainText(value) { return typeof value === "number" ? fmt(value) : this.text(value); },
    pairLabel(pair) {
      const pairs = (this.predictionsData && this.predictionsData.pairs) || [];
      const suffix = `_${pair.target}`;
      if (pairs.filter(item => item.target === pair.target).length < 2 || pair.pred === `pred_${pair.target}` || !pair.pred.endsWith(suffix)) return pair.target;
      return `${pair.target} (${pair.pred.slice("pred_".length, -suffix.length)})`;
    },
    initialOf(target) {
      const config = (this.record && this.record.config) || {};
      const [owner, ...rest] = target.split(".");
      const loss = (config.losses || {})[owner];
      if (loss && typeof loss === "object") return rest.reduce((value, part) => (value && typeof value === "object" ? value[part] : undefined), loss.params || {});
      const optimizer = (config.optimizers || {})[owner];
      if (optimizer && typeof optimizer === "object") return (optimizer.params || {})[rest.join(".")];
      const model = ((config.model || {}).models || {})[owner];
      if (model && typeof model === "object" && rest.join(".") === "trainable") return model.trainable === undefined ? true : model.trainable;
      return undefined;
    },
    effectiveTerms(loss, terms) {
      const last = this.history.lines[this.history.lines.length - 1] || {};
      const whole = last[`effect/${loss}.terms`];
      const found = whole && typeof whole === "object" ? { ...whole } : { ...terms };
      const prefix = `effect/${loss}.terms.`;
      for (const [key, value] of Object.entries(last)) {
        if (key.startsWith(prefix)) found[key.slice(prefix.length)] = value;
      }
      return found;
    },
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
      const kept = { tab: name, item: null, model: null, file: null, q: null, log: null, chart: null, diagram: null };
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
    childrenOpen(entry) {
      const choice = this.opened[entry.path];
      if (choice !== undefined) return choice;
      if (this.filter) return true;
      return entry.children.some(child => child.path === this.path);
    },
    toggleChildren(entry) { this.opened = { ...this.opened, [entry.path]: !this.childrenOpen(entry) }; },
    colorGroups(rows, key) {
      if (key === "state") {
        return [...new Set(rows.map(row => row.state))].map(name => ({
          name, color: STATE_COLORS[name] || PALETTE[7], rows: rows.filter(row => row.state === name) }));
      }
      const axis = this.parallelAxes.find(item => item.key === key);
      if (axis && axis.kind === "choices" && axis.values.length <= 3) {
        return axis.values.map((value, index) => ({ name: `${key} ${fmt(value)}`, color: PALETTE[index],
          rows: rows.filter(row => row.values[key] === value) })).filter(group => group.rows.length);
      }
      const score = key === "objective";
      const colors = score && this.objectiveMode === "max" ? [...bandColors(true)].reverse() : bandColors(score);
      if (axis && axis.kind === "choices") {
        const step = Math.max(1, Math.floor(colors.length / Math.max(axis.values.length - 1, 1)));
        return axis.values.map((value, index) => ({ name: `${key} ${fmt(value)}`, color: colors[Math.min(index * step, colors.length - 1)],
          rows: rows.filter(row => row.values[key] === value) })).filter(group => group.rows.length);
      }
      const values = rows.map(row => row.values[key]).filter(value => typeof value === "number");
      if (!values.length) return [{ name: key, color: PALETTE[0], rows }];
      const [low, high] = extent(values);
      if (high === low) return [{ name: `${key} ${tick(low)}`, color: colors[0], rows }];
      const edges = colors.map((item, index) => low + (index + 1) * (high - low) / colors.length);
      return colors.map((color, index) => ({
        name: `${tick(index ? edges[index - 1] : low)} to ${tick(edges[index])}`, color,
        rows: rows.filter(row => {
          const value = row.values[key];
          return typeof value === "number" && value <= edges[index] && (index === 0 || value > edges[index - 1]);
        }) })).filter(group => group.rows.length);
    },
    setExplore(which, value) { this.go({ [which]: value }); },
    setBrush(next) { this.brush = next; },
    openPoint(path) { location.hash = this.recordLink(path); },
    clearBrush() { this.brush = {}; },
    dropBrush(key) { const next = { ...this.brush }; delete next[key]; this.brush = next; },
    toggleSidebar() {
      this.sidebar = !this.sidebar;
      try { localStorage.setItem("kalfa-board-sidebar", this.sidebar ? "open" : "closed"); } catch (error) { console.warn("storage unavailable", error); }
    },
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
      this.record = null; this.sweep = null; this.brush = {}; this.history = { lines: [], offset: 0 }; this.steps = { lines: [], offset: 0 };
      this.overlay = {}; this.diff = null; this.describeText = null; this.showModuleText = false;
      this.logs = { name: "", lines: [], total: 0 };
      this.predictionsData = null; this.prepData = null; this.filesData = null; this.fileView = null; this.eventsData = null;
      this.stopPlaying();
      if (this.refresh === "realtime") this.connectWatch();
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
      await this.loadMore("history");
    },
    loadMore(name) {
      const chained = (this.queue[name] || Promise.resolve()).then(async () => {
        const path = this.path;
        const found = await api(`/api/${name}`, { path, offset: this[name].offset });
        if (found && path === this.path) this[name] = { lines: this[name].lines.concat(found.lines), offset: found.offset };
      });
      this.queue[name] = chained.catch(error => console.warn(`${name} did not load`, error));
      return chained;
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
      const found = await this.heavy(api("/api/describe", { path }));
      if (path === this.path) this.describeText = found ? found.text : "the record cannot be described";
    },
    async loadTable() {
      const found = await this.heavy(api("/api/table"));
      if (found) { this.tableRows = found.rows; this.tableLoaded = true; }
    },
    async loadCompare() {
      const { a, b } = this.route.params;
      if (!a || !b) { this.compareData = null; return; }
      const [recordA, recordB, historyA, historyB, diff] = await this.heavy(Promise.all([
        api("/api/record", { path: a }), api("/api/record", { path: b }), api("/api/history", { path: a }), api("/api/history", { path: b }),
        api("/api/diff", { a, b })]));
      if (!recordA || !recordB) { this.compareData = { missing: true }; return; }
      this.compareData = { a: { path: a, record: recordA, history: historyA ? historyA.lines : [] },
                           b: { path: b, record: recordB, history: historyB ? historyB.lines : [] },
                           diff: diff ? (diff.diff.length ? diff.diff : ["no difference"]) : ["no resolved.yaml to compare"] };
    },
    heavy(promise) {
      this.busy += 1;
      return promise.finally(() => { this.busy -= 1; });
    },
    async loadPredictions() {
      if (!this.record || !this.predictionFile) { this.predictionsData = null; return; }
      const path = this.path, name = this.predictionFile;
      const where = this.whereFilter;
      const found = await this.heavy(api("/api/predictions", { path, name, sample: this.sampleAll ? "all" : 2000, bins: this.plot.bins, where }));
      if (path === this.path && name === this.predictionFile && where === this.whereFilter) this.predictionsData = found;
    },
    applyWhere() { this.go({ where: (this.whereDraft || "").trim() || null }); },
    clearWhere() { this.whereDraft = ""; this.go({ where: null }); },
    async loadPrep() {
      if (this.prepData !== null) return;
      const path = this.path;
      const found = await this.heavy(api("/api/prep", { path }));
      if (path === this.path) this.prepData = found || { fields: [], preprocessors: {}, missing: true };
    },
    async loadFiles() {
      const path = this.path;
      const found = await this.heavy(api("/api/files", { path }));
      if (path === this.path) this.filesData = found;
    },
    recordFile(name) { return `/file?path=${encodeURIComponent(`${this.path}/${name}`)}`; },
    async openFile(name) {
      const kind = fileKind(name);
      if (kind === "image" || kind === "pdf") { this.fileView = { name, kind, url: this.recordFile(name) }; return; }
      if (kind === "binary") {
        this.fileView = { name, kind, text: `a .${name.split(".").pop()} file is not shown here; save it and open it with the tool that reads it` };
        return;
      }
      const path = this.path;
      const found = await this.heavy(api("/api/text", { path, name }));
      if (path === this.path) this.fileView = found ? { ...found, kind } : { name, kind, text: "this file could not be read", truncated: false, size: 0 };
    },
    async loadEvents() {
      const path = this.path;
      const found = await this.heavy(api("/api/events", { path }));
      if (path === this.path) this.eventsData = found;
    },
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
      if ((this.tab === "steps" || this.tab === "monitor") && !this.steps.lines.length) await this.loadMore("steps");
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
    closeDiagram() { this.go({ diagram: null }, true); },
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
    downloadModal() {
      const charts = this.expanded && this.expanded.panels ? (this.$refs.panels || []) : [this.$refs.modal];
      charts.forEach(chart => { if (chart) chart.download(); });
    },
    resetPlot() { this.plot = { ...defaultPlot(), logy: this.logy }; },
    async copy(text) {
      try { await navigator.clipboard.writeText(text || ""); } catch (error) { console.warn("clipboard unavailable", error); }
    },
    async tick() {
      await this.loadLive();
      if (!this.path) return;
      if (this.page === "table") { await this.loadTable(); return; }
      if (this.page === "compare" || !this.record) return;
      if (this.isSweep) { await this.loadSweep(); await this.loadOverlay(); return; }
      if (!["running", "pending"].includes(this.state)) return;
      await this.loadRecord();
      if (this.tab === "steps" || this.tab === "monitor") await this.loadMore("steps");
      if (this.tab === "logs" || this.tab === "monitor") await this.loadLogs();
      if (this.tab === "timeline") await this.loadEvents();
    },
    stopText(entry) {
      const stop = entry && entry.status && entry.status.stop;
      if (!stop) return "";
      const asked = ["running", "pending"].includes(this.stateOf(entry)) ? "stop requested" : "stopped on request";
      return asked + (stop.by ? ` by ${stop.by}` : "") + (stop.at ? ` at ${new Date(stop.at).toLocaleTimeString()}` : "");
    },
    canStop(entry) {
      if (!entry || (entry.status && entry.status.stop)) return false;
      if (entry.kind === "sweep" || (entry.manifest && entry.manifest.kind === "sweep")) return this.live;
      return ["running", "pending"].includes(this.stateOf(entry));
    },
    async requestStop(path) {
      const asked = window.confirm(`Stop ${path} after its current turn? The run ends like an early stop, with its final state, predictions and plots.`);
      if (!asked) return;
      const result = await post("/api/stop", { path });
      if (!result.ok) { window.alert(result.error || "the board could not write the stop file"); return; }
      if (this.page === "home") { await this.loadLive(); return; }
      if (this.isSweep) { await this.loadSweep(); await this.loadRecord(); return; }
      await this.loadRecord();
    },
    fmtBytes: human,
    applyRefresh() {
      if (this.timer) clearInterval(this.timer);
      if (this.treeTimer) clearInterval(this.treeTimer);
      this.timer = null; this.treeTimer = null;
      this.disconnectWatch();
      if (this.refresh === "realtime") { this.connectWatch(); return; }
      if (this.refresh === "off") return;
      const seconds = Number(this.refresh) || 5;
      this.timer = setInterval(() => this.tick(), seconds * 1000);
      this.treeTimer = setInterval(() => this.loadTree(), Math.max(seconds * 4, 20) * 1000);
    },
    connectWatch() {
      this.disconnectWatch();
      if (typeof EventSource === "undefined") { this.refresh = "3"; return; }
      const target = this.page === "record" ? this.path : "";
      this.source = new EventSource(`/api/watch?path=${encodeURIComponent(target)}`);
      this.source.onopen = () => { this.connected = true; };
      this.source.onerror = () => { this.connected = false; };
      this.source.onmessage = event => {
        try { this.onChange(JSON.parse(event.data).changed || []); } catch (error) { console.warn("bad watch event", error); }
      };
    },
    disconnectWatch() {
      if (this.source) this.source.close();
      this.source = null;
      this.connected = false;
    },
    async onChange(changed) {
      this.refreshed = new Date().toLocaleTimeString();
      const own = name => changed.some(item => item === name || item === `${this.path}/${name}`);
      const elsewhere = changed.some(item => item === "tree" || item.endsWith("/history.jsonl") || item.endsWith("/run.json"));
      if (changed.includes("tree")) await this.loadTree();
      if (this.page === "home" || elsewhere) await this.loadLive();
      if (this.page === "home") return;
      if (this.page === "table") { await this.loadTable(); return; }
      if (this.page === "compare" || !this.record) return;
      if (this.isSweep) { await this.loadSweep(); await this.loadOverlay(); return; }
      const rest = changed.filter(name => !name.includes("/") && !["tree", "history.jsonl", "steps.jsonl", "stdout.txt", "stderr.txt", "events.jsonl"].includes(name));
      if (rest.length) await this.loadRecord();
      else if (own("history.jsonl")) await this.loadMore("history");
      if (own("steps.jsonl") && (this.tab === "steps" || this.tab === "monitor")) await this.loadMore("steps");
      if ((own("stdout.txt") || own("stderr.txt")) && (this.tab === "logs" || this.tab === "monitor")) await this.loadLogs();
      if (own("events.jsonl") && this.tab === "timeline") await this.loadEvents();
      if (own("predictions.parquet") && this.tab === "predictions") await this.loadPredictions();
    },
  },
  mounted() {
    this.modalHeight = Math.max(360, Math.round(window.innerHeight * 0.72));
    Promise.all([this.loadTree(), this.loadLive(), this.entering])
      .catch(error => console.warn("the first reads did not all land", error))
      .finally(() => { this.booted = true; });
    window.addEventListener("hashchange", () => this.onHash());
    this.applyRefresh();
    window.addEventListener("keydown", event => {
      if (event.key === "Escape") {
        if (this.route.params.chart) this.closeChart();
        else if (this.route.params.diagram) this.closeDiagram();
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
