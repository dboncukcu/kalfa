import { computed, reactive, ref, watch }
  from "https://cdn.jsdelivr.net/npm/vue@3.4.38/dist/vue.esm-browser.prod.js";

import { PALETTES, numberOf, themeColors } from "../plot.js";
import { store } from "../store.js";


const PREFIX = "kalfa.chart.";
const shared = new Map();

export const PALETTE_NAMES = Object.keys(PALETTES);
export const PLACES = ["top", "bottom", "right", "inside"];

const CHOICES = {
  xscale: ["", "linear", "log"],
  yscale: ["", "linear", "log"],
  dash: ["", "solid", "dash", "dot"],
  mode: ["auto", "lines", "markers", "lines+markers"],
  legend: ["auto", "on", "off"],
  place: PLACES,
  palette: PALETTE_NAMES,
};

export function blankSettings() {
  return {
    xscale: "",
    yscale: "",
    xmin: "",
    xmax: "",
    ymin: "",
    ymax: "",
    xtitle: "",
    ytitle: "",
    grid: true,
    width: "",
    dash: "",
    mode: "auto",
    size: "",
    opacity: "",
    smoothing: 0,
    bins: "",
    title: "",
    legend: "auto",
    place: "top",
    palette: "theme",
    boxWidth: "",
    boxHeight: "",
    imageWidth: "",
    imageHeight: "",
    imageScale: "",
    layout: "",
  };
}

function cleaned(found) {
  const settings = blankSettings();
  if (!found || typeof found !== "object" || Array.isArray(found)) return settings;
  for (const key of Object.keys(settings)) {
    const value = found[key];
    if (typeof settings[key] === "boolean") {
      if (typeof value === "boolean") settings[key] = value;
    } else if (typeof value === "string" || (typeof value === "number" && Number.isFinite(value))) {
      settings[key] = value;
    }
  }
  for (const [key, allowed] of Object.entries(CHOICES)) {
    if (!allowed.includes(settings[key])) settings[key] = blankSettings()[key];
  }
  settings.smoothing = Math.min(0.99, Math.max(0, numberOf(settings.smoothing) || 0));
  return settings;
}

export function changedSettings(settings) {
  const blank = blankSettings();
  const found = {};
  for (const key of Object.keys(blank)) {
    if (settings && settings[key] !== undefined && settings[key] !== blank[key]) found[key] = settings[key];
  }
  return found;
}

export function isTuned(settings) {
  return Object.keys(changedSettings(settings)).length > 0;
}

export function loadSettings(key) {
  try {
    const text = window.localStorage.getItem(PREFIX + key);
    return cleaned(text ? JSON.parse(text) : null);
  } catch (error) {
    return blankSettings();
  }
}

export function saveSettings(key, settings) {
  const changed = changedSettings(settings);
  try {
    if (Object.keys(changed).length) window.localStorage.setItem(PREFIX + key, JSON.stringify(changed));
    else window.localStorage.removeItem(PREFIX + key);
  } catch (error) {
    return;
  }
}

export function chartSettings(key) {
  if (!shared.has(key)) shared.set(key, reactive(loadSettings(key)));
  return shared.get(key);
}

export function forgetSettings(key) {
  Object.assign(chartSettings(key), blankSettings());
  try {
    window.localStorage.removeItem(PREFIX + key);
  } catch (error) {
    return;
  }
}

export function parseLayout(text) {
  const trimmed = String(text || "").trim();
  if (!trimmed) return { layout: null, error: "" };
  try {
    const parsed = JSON.parse(trimmed);
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) return { layout: parsed, error: "" };
    return { layout: null, error: "the layout must be a JSON object, like {\"margin\": {\"l\": 80}}" };
  } catch (error) {
    return { layout: null, error: "not valid JSON yet: " + error.message };
  }
}

export function drawOptions(settings) {
  const found = {};
  if (settings.xscale) found.logx = settings.xscale === "log";
  if (settings.yscale) found.logy = settings.yscale === "log";
  for (const key of ["xmin", "xmax", "ymin", "ymax", "width", "size", "opacity"]) {
    const value = numberOf(settings[key]);
    if (value !== null) found[key] = value;
  }
  const smoothing = numberOf(settings.smoothing);
  if (smoothing) found.smoothing = smoothing;
  if (settings.xtitle) found.xlabel = settings.xtitle;
  if (settings.ytitle) found.ylabel = settings.ytitle;
  if (settings.grid === false) found.grid = false;
  if (settings.dash) found.dash = settings.dash;
  if (settings.mode && settings.mode !== "auto") found.mode = settings.mode;
  if (settings.title) found.title = settings.title;
  if (settings.legend === "on" || settings.legend === "off") found.legend = settings.legend === "on";
  if (settings.place && settings.place !== "top") found.legendPlace = settings.place;
  if (settings.palette && settings.palette !== "theme") found.palette = settings.palette;
  const layout = parseLayout(settings.layout).layout;
  if (layout) found.layout = layout;
  return found;
}

export function payloadOf(key, settings, options) {
  const found = drawOptions(settings);
  const base = options || {};
  const bins = numberOf(settings.bins);
  return {
    key,
    logx: found.logx !== undefined ? found.logx : Boolean(base.logx),
    logy: found.logy !== undefined ? found.logy : Boolean(base.logy),
    xmin: numberOf(settings.xmin),
    xmax: numberOf(settings.xmax),
    ymin: numberOf(settings.ymin),
    ymax: numberOf(settings.ymax),
    bins: bins !== null && bins >= 1 ? Math.round(bins) : null,
    smoothing: numberOf(settings.smoothing) || 0,
    settings: changedSettings(settings),
  };
}

function featuresOf(kind, horizontal, binsBox) {
  const grid = kind !== "heatmap";
  return {
    x: grid && !(kind === "bars" && !horizontal),
    y: grid && !(kind === "bars" && horizontal),
    grid,
    line: kind === "lines",
    marker: kind === "lines" || kind === "scatter",
    bins: kind === "histogram" || Boolean(binsBox),
  };
}

function defaultsOf(kind) {
  return {
    width: 2,
    size: kind === "scatter" ? 8 : 6,
    opacity: kind === "scatter" ? 0.85 : kind === "histogram" ? 0.9 : 1,
  };
}

export const ChartSettings = {
  name: "KChartSettings",
  props: {
    settings: { type: Object, required: true },
    kind: { type: String, default: "lines" },
    auto: { type: Object, default: () => ({}) },
    size: { type: Object, default: () => ({}) },
    binsBox: { type: Boolean, default: false },
  },
  emits: ["reset", "download"],
  setup(props, { emit }) {
    const draft = ref(props.settings.layout || "");
    const advanced = ref(Boolean(props.settings.layout));
    const palettes = PALETTE_NAMES;
    const places = PLACES;
    const has = computed(() => featuresOf(props.kind, props.auto.horizontal !== false, props.binsBox));
    const defaults = computed(() => defaultsOf(props.kind));
    const swatches = computed(() => (store.paint >= 0 ? themeColors(props.settings.palette).series : []));
    const layoutError = computed(() => parseLayout(draft.value).error);

    watch(() => props.settings.layout, (value) => {
      if (value !== draft.value) draft.value = value || "";
    });

    const scaleText = (log) => "auto (" + (log ? "log" : "linear") + ")";
    const shown = (name) => (props.settings[name] === "" ? defaults.value[name] : props.settings[name]);
    const readout = (name) => (props.settings[name] === "" ? "auto" : String(props.settings[name]));
    const put = (name, event) => {
      props.settings[name] = Number(event.target.value);
    };
    const clear = (name) => {
      props.settings[name] = "";
    };
    const fit = (value) => (value ? "fit, " + value : "fit");
    const typed = (event) => {
      draft.value = event.target.value;
      if (!parseLayout(draft.value).error) props.settings.layout = draft.value.trim() ? draft.value : "";
    };
    const opened = (event) => {
      advanced.value = event.target.open;
    };
    const resetAll = () => {
      draft.value = "";
      emit("reset");
    };
    const download = (format) => emit("download", format);
    return {
      draft, advanced, palettes, places, has, swatches, layoutError, scaleText, shown, readout, put, clear, fit,
      typed, opened, resetAll, download,
    };
  },
  template: `
    <aside class="chart-panel" aria-label="chart settings">
      <section class="chart-group">
        <div class="chart-group-title">axes</div>
        <div v-if="has.x" class="chart-row">
          <span class="chart-label">x</span>
          <select v-model="settings.xscale" aria-label="x scale">
            <option value="">{{ scaleText(auto.logx) }}</option>
            <option value="linear">linear</option>
            <option value="log">log</option>
          </select>
          <input type="number" step="any" placeholder="min" aria-label="x minimum" v-model.lazy="settings.xmin">
          <input type="number" step="any" placeholder="max" aria-label="x maximum" v-model.lazy="settings.xmax">
        </div>
        <div v-if="has.y" class="chart-row">
          <span class="chart-label">y</span>
          <select v-model="settings.yscale" aria-label="y scale">
            <option value="">{{ scaleText(auto.logy) }}</option>
            <option value="linear">linear</option>
            <option value="log">log</option>
          </select>
          <input type="number" step="any" placeholder="min" aria-label="y minimum" v-model.lazy="settings.ymin">
          <input type="number" step="any" placeholder="max" aria-label="y maximum" v-model.lazy="settings.ymax">
        </div>
        <div v-if="has.x || has.y" class="chart-hint">an empty range end follows the data</div>
        <div class="chart-row">
          <span class="chart-label">titles</span>
          <input type="text" spellcheck="false" aria-label="x axis title" :placeholder="auto.xlabel || 'x axis'"
            v-model.lazy="settings.xtitle">
          <input type="text" spellcheck="false" aria-label="y axis title" :placeholder="auto.ylabel || 'y axis'"
            v-model.lazy="settings.ytitle">
        </div>
        <label v-if="has.grid" class="chart-check"><input type="checkbox" v-model="settings.grid"> grid</label>
      </section>

      <section class="chart-group">
        <div class="chart-group-title">{{ has.line ? 'lines and markers' : 'marks' }}</div>
        <div v-if="has.line" class="chart-row">
          <span class="chart-label">line width</span>
          <input type="range" min="0.5" max="8" step="0.5" aria-label="line width" :value="shown('width')"
            @input="put('width', $event)">
          <button type="button" class="chart-auto" :disabled="settings.width === ''" title="back to auto"
            @click="clear('width')">{{ readout('width') }}</button>
        </div>
        <div v-if="has.line" class="chart-row">
          <span class="chart-label">line style</span>
          <select v-model="settings.dash" aria-label="line style">
            <option value="">auto</option>
            <option value="solid">solid</option>
            <option value="dash">dashed</option>
            <option value="dot">dotted</option>
          </select>
        </div>
        <div v-if="has.line" class="chart-row">
          <span class="chart-label">draw</span>
          <select v-model="settings.mode" aria-label="lines or markers">
            <option value="auto">auto</option>
            <option value="lines">lines</option>
            <option value="markers">markers</option>
            <option value="lines+markers">lines and markers</option>
          </select>
        </div>
        <div v-if="has.marker" class="chart-row">
          <span class="chart-label">marker size</span>
          <input type="range" min="1" max="20" step="1" aria-label="marker size" :value="shown('size')"
            @input="put('size', $event)">
          <button type="button" class="chart-auto" :disabled="settings.size === ''" title="back to auto"
            @click="clear('size')">{{ readout('size') }}</button>
        </div>
        <div class="chart-row">
          <span class="chart-label">opacity</span>
          <input type="range" min="0.05" max="1" step="0.05" aria-label="opacity" :value="shown('opacity')"
            @input="put('opacity', $event)">
          <button type="button" class="chart-auto" :disabled="settings.opacity === ''" title="back to auto"
            @click="clear('opacity')">{{ readout('opacity') }}</button>
        </div>
        <div v-if="has.line" class="chart-row">
          <span class="chart-label">smoothing</span>
          <input type="range" min="0" max="0.99" step="0.01" aria-label="smoothing" v-model.number="settings.smoothing">
          <button type="button" class="chart-auto" :disabled="!settings.smoothing" title="turn smoothing off"
            @click="settings.smoothing = 0">{{ settings.smoothing ? settings.smoothing : 'off' }}</button>
        </div>
        <div v-if="has.line && settings.smoothing" class="chart-hint">exponential moving average; the raw line stays faint</div>
        <div v-if="has.bins" class="chart-row">
          <span class="chart-label">bins</span>
          <input type="number" min="1" max="1000" step="1" placeholder="auto" aria-label="bins"
            v-model.lazy="settings.bins">
        </div>
        <div v-if="has.bins" class="chart-hint">the server counts the rows again into this many bins</div>
      </section>

      <section class="chart-group">
        <div class="chart-group-title">look</div>
        <div class="chart-row">
          <span class="chart-label">title</span>
          <input type="text" spellcheck="false" aria-label="title on the chart" placeholder="none"
            v-model.lazy="settings.title">
        </div>
        <div class="chart-row">
          <span class="chart-label">legend</span>
          <select v-model="settings.legend" aria-label="legend">
            <option value="auto">auto</option>
            <option value="on">on</option>
            <option value="off">off</option>
          </select>
          <select v-model="settings.place" aria-label="legend place" :disabled="settings.legend === 'off'">
            <option v-for="place in places" :key="place" :value="place">{{ place }}</option>
          </select>
        </div>
        <div class="chart-row">
          <span class="chart-label">palette</span>
          <select v-model="settings.palette" aria-label="palette">
            <option v-for="name in palettes" :key="name" :value="name">{{ name }}</option>
          </select>
        </div>
        <div class="chart-swatches">
          <span v-for="(color, index) in swatches" :key="index" class="chart-swatch" :style="{ background: color }"></span>
        </div>
      </section>

      <section class="chart-group">
        <div class="chart-group-title">this view</div>
        <div class="chart-row">
          <span class="chart-label">width</span>
          <input type="number" min="200" step="10" aria-label="view width" :placeholder="fit(size.width)"
            v-model.lazy="settings.boxWidth">
          <span class="chart-unit">px</span>
        </div>
        <div class="chart-row">
          <span class="chart-label">height</span>
          <input type="number" min="200" step="10" aria-label="view height" :placeholder="fit(size.height)"
            v-model.lazy="settings.boxHeight">
          <span class="chart-unit">px</span>
        </div>
      </section>

      <section class="chart-group">
        <div class="chart-group-title">download</div>
        <div class="chart-row">
          <span class="chart-label">size</span>
          <input type="number" min="100" step="10" aria-label="image width" :placeholder="String(size.width || 900)"
            v-model.lazy="settings.imageWidth">
          <span class="chart-unit">by</span>
          <input type="number" min="100" step="10" aria-label="image height" :placeholder="String(size.height || 600)"
            v-model.lazy="settings.imageHeight">
        </div>
        <div class="chart-row">
          <span class="chart-label">scale</span>
          <input type="number" min="0.25" max="8" step="0.25" aria-label="image scale" placeholder="2 for png, 1 for svg"
            v-model.lazy="settings.imageScale">
        </div>
        <div class="chart-row chart-actions">
          <span class="chart-label"></span>
          <button type="button" class="tool" @click="download('png')">png</button>
          <button type="button" class="tool" @click="download('svg')">svg</button>
        </div>
      </section>

      <details class="chart-group" :open="advanced" @toggle="opened">
        <summary class="chart-group-title">advanced</summary>
        <div class="chart-hint">Plotly layout as JSON, merged over the layout of this chart</div>
        <textarea rows="6" spellcheck="false" aria-label="Plotly layout JSON" :value="draft" @input="typed"
          placeholder='{"yaxis": {"tickformat": ".1e"}, "margin": {"l": 80}}'></textarea>
        <div v-if="layoutError" class="chart-error">{{ layoutError }}</div>
      </details>

      <div class="chart-row chart-actions">
        <button type="button" class="tool chart-reset" @click="resetAll">reset</button>
        <span class="chart-hint">saved in this browser for this chart</span>
      </div>
      <div class="chart-hint">Drag a box to zoom, double click to zoom out, click a legend entry to hide its series.</div>
    </aside>`,
};

export default ChartSettings;
