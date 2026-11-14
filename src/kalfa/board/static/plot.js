const FALLBACK = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"];
const AXES = ["turn", "update", "x", "step"];
const CONFIG = { responsive: true, displaylogo: false, modeBarButtonsToRemove: ["lasso2d", "select2d"] };
const MODES = ["lines", "markers", "lines+markers"];
const DASHES = ["solid", "dash", "dot"];
const SAMPLES = 120;
const MARK_LABELS = 12;
const UNSAFE = ["__proto__", "constructor", "prototype"];

export const PALETTES = {
  theme: null,
  colorblind: ["#0072b2", "#e69f00", "#009e73", "#cc79a7", "#56b4e9", "#d55e00", "#f0e442", "#999999"],
  classic: ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"],
  viridis: ["#440154", "#482878", "#3e4989", "#31688e", "#26828e", "#1f9e89", "#35b779", "#6ece58", "#b5de2b", "#fde725"],
};

function plotly() {
  return typeof window !== "undefined" ? window.Plotly : undefined;
}

export function numberOf(value) {
  if (value === "" || value === null || value === undefined || typeof value === "boolean") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function between(value, low, high, fallback) {
  const number = numberOf(value);
  return number === null ? fallback : Math.min(high, Math.max(low, number));
}

export function paletteOf(palette) {
  if (Array.isArray(palette) && palette.length) return palette;
  const found = typeof palette === "string" ? PALETTES[palette] : null;
  return Array.isArray(found) ? found : null;
}

export function themeColors(palette) {
  const style = getComputedStyle(document.documentElement);
  const read = (name, fallback) => (style.getPropertyValue(name) || "").trim() || fallback;
  return {
    ink: read("--ink", "#17191c"),
    muted: read("--muted", "#7a7974"),
    line: read("--line", "#e4e3df"),
    surface: read("--surface", "#fcfcfb"),
    accent: read("--accent", "#2a78d6"),
    font: read("--font", "system-ui, sans-serif"),
    series: paletteOf(palette) || FALLBACK.map((color, index) => read("--series-" + (index + 1), color)),
  };
}

export function colorAt(index) {
  const series = themeColors().series;
  return series[index % series.length];
}

function axis(colors, title, log, extra) {
  return {
    title: { text: title || "", font: { size: 11 } },
    type: log ? "log" : "-",
    gridcolor: colors.line,
    linecolor: colors.line,
    zerolinecolor: colors.line,
    tickfont: { color: colors.muted },
    automargin: true,
    ...(extra || {}),
  };
}

function veil(color) {
  return /^#[0-9a-f]{6}$/i.test(color) ? color + "d9" : color;
}

function legendOf(place, colors, height) {
  const font = { color: colors.muted };
  if (place === "right") return { orientation: "v", x: 1.02, xanchor: "left", y: 1, yanchor: "top", font };
  if (place === "bottom") {
    const below = 56 / Math.max(120, (height || 300) - 80);
    return { orientation: "h", x: 0, xanchor: "left", y: -Math.min(0.5, below), yanchor: "top", font };
  }
  if (place === "inside") {
    return { orientation: "v", x: 0.99, xanchor: "right", y: 0.99, yanchor: "top", font,
      bgcolor: veil(colors.surface), bordercolor: colors.line, borderwidth: 1 };
  }
  return { orientation: "h", x: 0, y: 1.02, yanchor: "bottom", font };
}

export function baseLayout(options = {}) {
  const colors = themeColors(options.palette);
  const layout = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: colors.font, size: 11, color: colors.ink },
    colorway: colors.series,
    margin: { l: 48, r: 12, t: options.title ? 30 : 10, b: 36 },
    title: options.title
      ? { text: options.title, font: { size: 12 }, x: 0, xanchor: "left", xref: "paper", y: 1, yref: "container",
        yanchor: "top", pad: { t: 8 } }
      : undefined,
    height: options.height || undefined,
    autosize: true,
    showlegend: options.legend !== undefined ? Boolean(options.legend) : undefined,
    legend: legendOf(options.legendPlace, colors, options.height),
    hoverlabel: { bgcolor: colors.surface, bordercolor: colors.line, font: { color: colors.ink } },
    xaxis: axis(colors, options.xlabel, options.logx, options.xaxis),
    yaxis: axis(colors, options.ylabel, options.logy, options.yaxis),
  };
  if (options.grid === false) {
    layout.xaxis.showgrid = false;
    layout.yaxis.showgrid = false;
  }
  return layout;
}

function observe(el) {
  if (el.__kalfaSize || typeof ResizeObserver === "undefined") return;
  let frame = 0;
  const observer = new ResizeObserver(() => {
    if (frame) cancelAnimationFrame(frame);
    frame = requestAnimationFrame(() => {
      frame = 0;
      const Plotly = plotly();
      if (!Plotly || !el.isConnected || !el.data) return;
      const done = Plotly.Plots.resize(el);
      if (done && done.catch) done.catch(() => null);
    });
  });
  observer.observe(el);
  el.__kalfaSize = observer;
}

export function render(el, traces, layout) {
  if (!el) return;
  const Plotly = plotly();
  if (!Plotly) {
    el.textContent = "Plotly did not load from the CDN, so the charts cannot be drawn";
    return;
  }
  const done = Plotly.react(el, traces, layout, CONFIG);
  if (done && done.catch) done.catch((error) => console.error(error));
  observe(el);
}

export function purge(el) {
  if (!el) return;
  if (el.__kalfaSize) {
    el.__kalfaSize.disconnect();
    el.__kalfaSize = null;
  }
  const Plotly = plotly();
  if (Plotly && el.data) Plotly.purge(el);
}

function copyOf(value) {
  return JSON.parse(JSON.stringify(value));
}

export function merge(target, source) {
  for (const [key, value] of Object.entries(source || {})) {
    if (UNSAFE.includes(key)) continue;
    const nested = value && typeof value === "object" && !Array.isArray(value);
    const inside = target[key] && typeof target[key] === "object" && !Array.isArray(target[key]);
    if (nested && inside) merge(target[key], value);
    else target[key] = value;
  }
  return target;
}

function valuesOf(traces, key) {
  const found = [];
  for (const trace of traces) {
    const values = trace[key];
    if (!Array.isArray(values)) continue;
    const across = trace.type === "bar" && (trace.orientation === "h" ? "y" : "x") === key;
    const widths = across && Array.isArray(trace.width) ? trace.width : null;
    if (trace.type === "bar" && !across) found.push(0);
    values.forEach((value, index) => {
      if (typeof value !== "number" || !Number.isFinite(value)) return;
      const width = widths && typeof widths[index] === "number" ? widths[index] : 0;
      if (width) found.push(value - width / 2, value + width / 2);
      else found.push(value);
    });
  }
  return found;
}

function spread(values, positive) {
  let low = Infinity;
  let high = -Infinity;
  for (const value of values) {
    if (positive && value <= 0) continue;
    if (value < low) low = value;
    if (value > high) high = value;
  }
  return Number.isFinite(low) ? [low, high] : null;
}

function limit(layoutAxis, low, high, values) {
  if (!layoutAxis || layoutAxis.type === "category") return;
  const log = layoutAxis.type === "log";
  let from = numberOf(low);
  let to = numberOf(high);
  if (log && from !== null && from <= 0) from = null;
  if (log && to !== null && to <= 0) to = null;
  if (from === null && to === null) return;
  const found = spread(values, log);
  const start = from !== null ? from : found ? found[0] : null;
  const end = to !== null ? to : found ? found[1] : null;
  if (start === null || end === null || !(end > start)) return;
  layoutAxis.range = log ? [Math.log10(start), Math.log10(end)] : [start, end];
  layoutAxis.autorange = false;
}

function finish(el, traces, layout, options) {
  limit(layout.xaxis, options.xmin, options.xmax, valuesOf(traces, "x"));
  limit(layout.yaxis, options.ymin, options.ymax, valuesOf(traces, "y"));
  const top = !options.legendPlace || options.legendPlace === "top";
  if (layout.title && layout.showlegend && top) layout.margin.t = Math.max(layout.margin.t, 56);
  layout.uirevision = JSON.stringify([options.revision || "", traces.map((trace) => trace.name || ""),
    Boolean(options.logx), Boolean(options.logy)]);
  const extra = options.layout && typeof options.layout === "object" && !Array.isArray(options.layout)
    ? copyOf(options.layout)
    : null;
  render(el, traces, extra ? merge(layout, extra) : layout);
}

function axisOf(table, options) {
  if (options.x && Array.isArray(table[options.x])) return options.x;
  return AXES.find((key) => Array.isArray(table[key])) || null;
}

export function linesOf(series, options = {}) {
  if (Array.isArray(series)) return series.filter(Boolean);
  if (!series || typeof series !== "object") return [];
  const xKey = axisOf(series, options);
  const wanted = options.keys || Object.keys(series).filter((key) => !AXES.includes(key) && Array.isArray(series[key]));
  const x = xKey ? series[xKey] : null;
  return wanted.filter((key) => Array.isArray(series[key])).map((key) => ({
    name: key,
    x: x || series[key].map((value, index) => index + 1),
    y: series[key],
  }));
}

export function smooth(values, weight) {
  let last = 0;
  let count = 0;
  return (values || []).map((value) => {
    if (typeof value !== "number" || !Number.isFinite(value)) return null;
    last = last * weight + (1 - weight) * value;
    count += 1;
    return last / (1 - Math.pow(weight, count));
  });
}

function marksOf(marks, colors) {
  const found = Array.isArray(marks) ? marks.filter((mark) => mark && numberOf(mark.x) !== null) : [];
  const labelled = found.length <= MARK_LABELS;
  return {
    shapes: found.map((mark) => ({
      type: "line",
      xref: "x",
      yref: "paper",
      x0: mark.x,
      x1: mark.x,
      y0: 0,
      y1: 1,
      line: { width: 1, dash: "dot", color: colors.muted },
    })),
    annotations: !labelled ? [] : found.map((mark) => ({
      xref: "x",
      yref: "paper",
      x: mark.x,
      y: 1,
      text: String(mark.text || ""),
      showarrow: false,
      textangle: -90,
      xanchor: "left",
      yanchor: "top",
      font: { size: 9, color: colors.muted },
    })),
  };
}

function modeOf(mode, fallback, count) {
  if (MODES.includes(mode)) return mode;
  return fallback || (count <= 40 ? "lines+markers" : "lines");
}

function dashOf(dash, fallback) {
  return DASHES.includes(dash) ? dash : fallback;
}

export function lines(el, series, options = {}) {
  const colors = themeColors(options.palette);
  const found = linesOf(series, options);
  const weight = between(options.smoothing, 0, 0.99, 0);
  const opacity = between(options.opacity, 0.05, 1, 1);
  const traces = [];
  found.forEach((line, index) => {
    const count = (line.y || []).length;
    const mode = modeOf(options.mode, line.mode, count);
    const style = {
      width: between(options.width, 0.5, 12, 2),
      color: line.color || colors.series[index % colors.series.length],
      dash: dashOf(options.dash, line.dash),
    };
    const marker = { size: between(options.size, 1, 30, 6) };
    const trace = {
      type: "scatter",
      mode,
      name: line.name,
      x: line.x,
      y: line.y,
      connectgaps: true,
      opacity,
      line: style,
      marker,
      hovertemplate: "%{y:.4g}",
    };
    if (weight > 0) {
      const group = "line " + index;
      traces.push({
        type: "scatter",
        mode: mode === "markers" ? "markers" : "lines",
        name: line.name,
        x: line.x,
        y: line.y,
        connectgaps: true,
        opacity: opacity * 0.25,
        line: { ...style },
        marker: { ...marker },
        legendgroup: group,
        showlegend: false,
        hoverinfo: "skip",
      });
      trace.y = smooth(line.y, weight);
      trace.customdata = line.y;
      trace.legendgroup = group;
      trace.hovertemplate = "%{y:.4g} (raw %{customdata:.4g})";
    }
    traces.push(trace);
  });
  const layout = baseLayout({ xlabel: options.xlabel || axisOf(series || {}, options) || "", ...options });
  layout.hovermode = "x unified";
  layout.showlegend = options.legend !== undefined ? Boolean(options.legend) : found.length > 1;
  if (weight > 0) layout.legend.traceorder = "normal";
  const marks = marksOf(options.marks, colors);
  if (marks.shapes.length) {
    layout.shapes = marks.shapes;
    layout.annotations = marks.annotations;
  }
  finish(el, traces, layout, options);
}

function pointsOf(points) {
  if (Array.isArray(points)) return points.filter(Boolean);
  return points && typeof points === "object" ? [points] : [];
}

function extent(groups, positive) {
  const values = [];
  for (const group of groups) {
    for (const value of [...(group.x || []), ...(group.y || [])]) {
      if (typeof value === "number" && Number.isFinite(value)) values.push(value);
    }
  }
  return spread(values, positive);
}

function diagonal(groups, logx, logy) {
  const range = extent(groups, Boolean(logx || logy));
  if (!range) return null;
  const [low, high] = range;
  if (Boolean(logx) === Boolean(logy) || !(high > low)) return [low, high];
  const steps = [];
  for (let index = 0; index <= SAMPLES; index += 1) {
    const at = index / SAMPLES;
    steps.push(low + at * (high - low), low * Math.pow(high / low, at));
  }
  return [...new Set(steps)].sort((first, second) => first - second);
}

function hoverOf(values, letter) {
  const named = (values || []).some((value) => value !== null && value !== undefined && typeof value !== "number");
  return named ? "%{" + letter + "}" : "%{" + letter + ":.4g}";
}

function sized(size, own, fallback) {
  const wanted = numberOf(size);
  if (wanted === null) return own || fallback;
  const ratio = Math.min(30, Math.max(1, wanted)) / fallback;
  if (Array.isArray(own)) return own.map((value) => value * ratio);
  return typeof own === "number" ? own * ratio : ratio * fallback;
}

export function scatter(el, points, options = {}) {
  const colors = themeColors(options.palette);
  const groups = pointsOf(points);
  const traces = groups.map((group, index) => ({
    type: group.x && group.x.length > 5000 ? "scattergl" : "scatter",
    mode: "markers",
    name: group.name || "",
    x: group.x,
    y: group.y,
    text: group.text,
    marker: {
      size: sized(options.size, group.size, 8),
      color: group.color || colors.series[index % colors.series.length],
      opacity: between(options.opacity, 0.05, 1, group.opacity || 0.85),
      line: { width: 1, color: colors.surface },
    },
    hovertemplate: (group.text ? "%{text}<br>" : "") + hoverOf(group.x, "x") + ", " + hoverOf(group.y, "y")
      + "<extra></extra>",
  }));
  const line = options.diagonal ? diagonal(groups, options.logx, options.logy) : null;
  if (line) {
    traces.push({
      type: "scatter", mode: "lines", name: "y = x", x: line, y: line, hoverinfo: "skip",
      line: { width: 1, dash: "dot", color: colors.muted },
    });
  }
  const layout = baseLayout(options);
  layout.hovermode = "closest";
  layout.showlegend = options.legend !== undefined ? Boolean(options.legend) : groups.length > 1;
  finish(el, traces, layout, options);
}

function barsOf(edges, counts) {
  if (Array.isArray(edges) && (edges.length === 0 || typeof edges[0] === "number")) {
    return [{ edges, counts: counts || [] }];
  }
  if (Array.isArray(edges)) return edges.filter(Boolean);
  return edges && typeof edges === "object" ? [edges] : [];
}

export function histogram(el, edges, counts, options) {
  const numeric = Array.isArray(edges) && (edges.length === 0 || typeof edges[0] === "number");
  const settings = (numeric ? options : counts) || {};
  const colors = themeColors(settings.palette);
  const bars = barsOf(edges, numeric ? counts : null);
  const traces = bars.map((bar, index) => {
    const sides = bar.edges || [];
    const centers = sides.slice(0, -1).map((edge, position) => (edge + sides[position + 1]) / 2);
    const widths = sides.slice(0, -1).map((edge, position) => sides[position + 1] - edge);
    return {
      type: "bar",
      name: bar.name || "",
      x: centers,
      y: bar.counts || [],
      width: widths,
      opacity: between(settings.opacity, 0.05, 1, bars.length > 1 ? 0.6 : 0.9),
      marker: { color: bar.color || colors.series[index % colors.series.length], line: { width: 0 } },
      hovertemplate: "%{x:.4g}: %{y}<extra></extra>",
    };
  });
  const layout = baseLayout({ ylabel: "count", ...settings });
  layout.barmode = "overlay";
  layout.bargap = 0;
  layout.showlegend = settings.legend !== undefined ? Boolean(settings.legend) : bars.length > 1;
  finish(el, traces, layout, settings);
}

function cellsOf(matrix, labels) {
  if (Array.isArray(matrix)) {
    const names = Array.isArray(labels) ? labels : (labels && labels.x) || null;
    const rows = Array.isArray(labels) ? labels : (labels && labels.y) || names;
    return { z: matrix, x: names, y: rows };
  }
  const data = matrix || {};
  const z = data.z || data.matrix || [];
  const x = data.x || data.labels || null;
  return { z, x, y: data.y || x };
}

export function heatmap(el, matrix, labels, options) {
  const settings = (Array.isArray(matrix) ? options : labels) || {};
  const cells = cellsOf(matrix, Array.isArray(matrix) ? labels : null);
  const colors = themeColors(settings.palette);
  const x = cells.x ? cells.x.map(String) : undefined;
  const y = cells.y ? cells.y.map(String) : undefined;
  const traces = [{
    type: "heatmap",
    z: cells.z,
    x,
    y,
    colorscale: [[0, colors.surface], [1, settings.color || colors.series[0]]],
    showscale: settings.scale !== undefined ? Boolean(settings.scale) : false,
    texttemplate: settings.text === false ? undefined : "%{z}",
    hovertemplate: "%{y} / %{x}: %{z}<extra></extra>",
    opacity: between(settings.opacity, 0.05, 1, 1),
    xgap: 2,
    ygap: 2,
  }];
  const flat = { ...settings, logx: false, logy: false, xmin: null, xmax: null, ymin: null, ymax: null };
  const layout = baseLayout({ xlabel: "predicted", ylabel: "truth", ...flat });
  layout.xaxis.type = "category";
  layout.yaxis.type = "category";
  layout.yaxis.autorange = "reversed";
  layout.xaxis.gridcolor = "rgba(0,0,0,0)";
  layout.yaxis.gridcolor = "rgba(0,0,0,0)";
  finish(el, traces, layout, flat);
}

export function bars(el, data, options = {}) {
  const colors = themeColors(options.palette);
  const labels = (data && (data.labels || data.x)) || [];
  const values = (data && (data.values || data.y)) || [];
  const horizontal = options.horizontal !== false;
  const traces = [{
    type: "bar",
    orientation: horizontal ? "h" : "v",
    x: horizontal ? values : labels.map(String),
    y: horizontal ? labels.map(String) : values,
    opacity: between(options.opacity, 0.05, 1, 1),
    marker: { color: (data && data.color) || colors.series[0] },
    hovertemplate: horizontal ? "%{y}: %{x:.4g}<extra></extra>" : "%{x}: %{y:.4g}<extra></extra>",
  }];
  const layout = baseLayout(options);
  if (horizontal) {
    layout.yaxis.type = "category";
    layout.yaxis.autorange = "reversed";
  } else {
    layout.xaxis.type = "category";
  }
  layout.showlegend = false;
  finish(el, traces, layout, options);
}

export function parallel(el, data, options = {}) {
  const colors = themeColors(options.palette);
  const axes = data && Array.isArray(data.axes) ? data.axes : [];
  const last = axes.length ? axes[axes.length - 1] : null;
  const dimensions = axes.map((item) => {
    const dimension = { label: String(item.label || item.key || ""), values: item.values || [] };
    if (Array.isArray(item.ticks) && Array.isArray(item.text)) {
      dimension.tickvals = item.ticks;
      dimension.ticktext = item.text;
    }
    return dimension;
  });
  const traces = [{
    type: "parcoords",
    dimensions,
    line: {
      color: last ? last.values : [],
      colorscale: "Viridis",
      reversescale: Boolean(data && data.mode === "min"),
      showscale: true,
    },
    labelfont: { color: colors.ink, size: 11 },
    tickfont: { color: colors.muted, size: 10 },
    rangefont: { color: colors.muted, size: 9 },
  }];
  const flat = { ...options, logx: false, logy: false, xmin: null, xmax: null, ymin: null, ymax: null };
  const layout = baseLayout(flat);
  delete layout.xaxis;
  delete layout.yaxis;
  layout.margin = { l: 64, r: 64, t: 56, b: 28 };
  layout.showlegend = false;
  finish(el, traces, layout, flat);
}

function fileName(name) {
  return String(name || "chart").trim().replace(/[^\w.-]+/g, "_").replace(/^_+|_+$/g, "") || "chart";
}

export function exportImage(el, options = {}) {
  const Plotly = plotly();
  if (!Plotly || !el || !el.data) return Promise.reject(new Error("the chart is not drawn yet"));
  try {
    const colors = themeColors();
    const format = options.format === "svg" ? "svg" : "png";
    const width = Math.round(between(options.width || null, 100, 8000, el.clientWidth || 900));
    const height = Math.round(between(options.height || null, 100, 8000, el.clientHeight || 600));
    const scale = between(options.scale || null, 0.25, 8, format === "svg" ? 1 : 2);
    const layout = { ...copyOf(el.layout || {}), width, height, autosize: false,
      paper_bgcolor: colors.surface, plot_bgcolor: colors.surface };
    const figure = { data: copyOf(el.data), layout, config: { displaylogo: false } };
    const done = Plotly.downloadImage(figure, { format, width, height, scale, filename: fileName(options.filename) });
    return Promise.resolve(done);
  } catch (error) {
    return Promise.reject(error);
  }
}

export function draw(kind, el, data, options = {}) {
  if (kind === "scatter") return scatter(el, data, options);
  if (kind === "histogram") return histogram(el, data, options);
  if (kind === "heatmap") return heatmap(el, data, options);
  if (kind === "bars") return bars(el, data, options);
  if (kind === "parallel") return parallel(el, data, options);
  return lines(el, data, options);
}
