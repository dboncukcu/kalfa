import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch }
  from "https://cdn.jsdelivr.net/npm/vue@3.4.38/dist/vue.esm-browser.prod.js";

import { draw, exportImage, numberOf, purge } from "../plot.js";
import { store } from "../store.js";
import { ChartSettings, chartSettings, drawOptions, forgetSettings, isTuned, payloadOf, saveSettings }
  from "./settings.js";


function labelsOf(kind, options) {
  const fallback = { histogram: ["", "count"], heatmap: ["predicted", "truth"] }[kind] || ["", ""];
  return { xlabel: options.xlabel || fallback[0], ylabel: options.ylabel || fallback[1] };
}

export const Chart = {
  name: "KChart",
  components: { KChartSettings: ChartSettings },
  props: {
    title: { type: String, default: "" },
    kind: { type: String, default: "lines" },
    data: { default: null },
    options: { type: Object, default: () => ({}) },
    height: { type: Number, default: 260 },
    logToggle: { type: Boolean, default: true },
    note: { type: String, default: "" },
    settingsKey: { type: String, default: "" },
    binsBox: { type: Boolean, default: false },
  },
  emits: ["settings"],
  setup(props, { emit }) {
    const host = ref(null);
    const large = ref(null);
    const open = ref(false);
    const panel = ref(true);
    const failure = ref("");
    const measured = ref({ width: 0, height: 0 });
    let pressed = false;
    let announced = "";

    const base = () => props.options || {};
    const key = computed(() => props.settingsKey || props.kind + ":" + props.title);
    const settings = computed(() => chartSettings(key.value));
    const tuned = computed(() => isTuned(settings.value));
    const logy = computed(() => {
      const scale = settings.value.yscale;
      return scale ? scale === "log" : Boolean(base().logy);
    });
    const auto = computed(() => ({
      logx: Boolean(base().logx),
      logy: Boolean(base().logy),
      horizontal: base().horizontal !== false,
      ...labelsOf(props.kind, base()),
    }));
    const box = computed(() => ({
      width: numberOf(settings.value.boxWidth),
      height: numberOf(settings.value.boxHeight),
    }));
    const largeStyle = computed(() => {
      const style = {};
      if (box.value.width) style.width = Math.max(200, box.value.width) + "px";
      if (box.value.height) {
        style.height = Math.max(200, box.value.height) + "px";
        style.flex = "none";
      }
      return style;
    });

    const optionsFor = (height) => ({ ...base(), ...drawOptions(settings.value), revision: key.value, height });
    const paint = (el, height) => {
      if (!el || props.data === null || props.data === undefined) return;
      try {
        draw(props.kind, el, props.data, optionsFor(height));
        failure.value = "";
      } catch (error) {
        failure.value = "cannot draw this chart: " + error.message;
      }
    };
    const bigHeight = () => {
      if (box.value.height) return Math.max(200, box.value.height);
      return Math.max(240, (large.value && large.value.clientHeight) || window.innerHeight * 0.6);
    };
    const paintLarge = () => {
      const el = large.value;
      if (!open.value || !el) return;
      paint(el, bigHeight());
      const size = { width: el.clientWidth, height: el.clientHeight };
      if (size.width !== measured.value.width || size.height !== measured.value.height) measured.value = size;
    };
    const redraw = () => {
      paint(host.value, props.height);
      if (open.value) nextTick(paintLarge);
    };
    const announce = (force) => {
      const payload = payloadOf(key.value, settings.value, base());
      const text = JSON.stringify(payload);
      if (text === announced) return;
      announced = text;
      if (force) emit("settings", payload);
    };

    const onKey = (event) => {
      if (event.key === "Escape") close();
    };
    const close = () => {
      purge(large.value);
      open.value = false;
      window.removeEventListener("keydown", onKey);
    };
    const expand = () => {
      open.value = true;
      window.addEventListener("keydown", onKey);
      nextTick(paintLarge);
    };
    const press = (event) => {
      pressed = event.target === event.currentTarget;
    };
    const release = (event) => {
      if (pressed && event.target === event.currentTarget) close();
      pressed = false;
    };
    const togglePanel = () => {
      panel.value = !panel.value;
    };
    const toggleLog = () => {
      const next = !logy.value;
      settings.value.yscale = next === Boolean(base().logy) ? "" : next ? "log" : "linear";
    };
    const reset = () => forgetSettings(key.value);
    const download = (format) => {
      const current = settings.value;
      exportImage(large.value, {
        format,
        width: numberOf(current.imageWidth) || measured.value.width,
        height: numberOf(current.imageHeight) || measured.value.height,
        scale: numberOf(current.imageScale),
        filename: props.title || key.value,
      }).catch((error) => {
        failure.value = "cannot download this chart: " + error.message;
      });
    };

    watch([key, () => JSON.stringify(settings.value)], (now, before) => {
      if (now[0] === before[0]) saveSettings(key.value, settings.value);
      redraw();
      announce(true);
    });
    watch(() => props.data, redraw);
    watch(() => props.kind, redraw);
    watch(() => JSON.stringify(props.options || {}), redraw);
    watch(() => store.paint, redraw);
    onMounted(() => {
      redraw();
      announce(tuned.value);
    });
    onBeforeUnmount(() => {
      window.removeEventListener("keydown", onKey);
      purge(host.value);
      purge(large.value);
    });
    return {
      host, large, open, panel, failure, measured, settings, tuned, logy, auto, largeStyle, expand, close, press,
      release, togglePanel, toggleLog, reset, download,
    };
  },
  template: `
    <div class="card chart-card">
      <div class="card-head">
        <span class="card-title" :title="title">{{ title }}</span>
        <span class="card-tools">
          <button v-if="logToggle" type="button" class="tool" :class="{ on: logy }" @click="toggleLog">log y</button>
          <button type="button" class="tool" :class="{ tuned: tuned }" @click="expand"
            :title="tuned ? 'open this chart large; it has saved settings' : 'open this chart large with its settings'">expand</button>
        </span>
      </div>
      <div v-if="note" class="card-note muted small">{{ note }}</div>
      <div v-if="failure" class="error-box">{{ failure }}</div>
      <div ref="host" class="plot" :style="{ height: height + 'px' }"></div>
      <teleport to="body">
        <div v-if="open" class="modal chart-modal" @mousedown="press" @click="release">
          <div class="modal-box chart-box" role="dialog" aria-modal="true" :aria-label="title || 'chart'">
            <div class="card-head">
              <span class="card-title" :title="title">{{ title }}</span>
              <span class="card-tools">
                <button v-if="logToggle" type="button" class="tool" :class="{ on: logy }" @click="toggleLog">log y</button>
                <button type="button" class="tool" :class="{ on: panel }" @click="togglePanel">settings</button>
                <button type="button" class="tool" @click="close">close</button>
              </span>
            </div>
            <div v-if="note" class="card-note muted small">{{ note }}</div>
            <div v-if="failure" class="error-box">{{ failure }}</div>
            <div class="chart-body">
              <div class="chart-stage">
                <div ref="large" class="plot large" :style="largeStyle"></div>
              </div>
              <k-chart-settings v-if="panel" :settings="settings" :kind="kind" :auto="auto" :size="measured"
                :bins-box="binsBox" @reset="reset" @download="download"></k-chart-settings>
            </div>
          </div>
        </div>
      </teleport>
    </div>`,
};

export default Chart;
