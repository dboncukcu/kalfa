import { messageOf } from "../../components/error.js";
import { when } from "../../format.js";


const LIMIT = 200;
const TAIL = 300;
const OUTPUT = "log.txt";
const CODE = "overflow: auto; max-height: 50vh; margin: 0; padding: 10px; background: var(--surface-2); border-radius: 6px; white-space: pre";
const SKIP = ["at", "turn", "kind"];
const TONES = {
  run_start: "var(--done)",
  run_end: "var(--done)",
  resume: "var(--running)",
  checkpoint: "var(--running)",
  patch: "var(--stale)",
  rule: "var(--stale)",
  warning: "var(--stale)",
  stop: "var(--failed)",
  stopping: "var(--failed)",
  failure: "var(--failed)",
  error: "var(--failed)",
};
const ROW = "display: flex; flex-wrap: wrap; gap: 8px; align-items: baseline; padding: 3px 0";

function shown(value) {
  if (value === null || value === undefined) return "none";
  if (Array.isArray(value)) return value.map(shown).join(", ");
  if (typeof value === "object") {
    if (typeof value.ref === "string") return "@" + value.ref;
    return JSON.stringify(value);
  }
  return String(value);
}

export default {
  name: "LogTab",
  props: {
    record: { type: Object, default: null },
    path: { type: String, default: "" },
    api: { default: null },
    version: { type: Number, default: 0 },
  },
  data() {
    return { fetched: null, error: "", loading: false, kind: "", output: "", outputError: "" };
  },
  computed: {
    row() { return ROW; },
    code() { return CODE; },
    tail() { return TAIL; },
    hasOutput() {
      const files = this.record && Array.isArray(this.record.files) ? this.record.files : [];
      return files.some(item => item && item.path === OUTPUT);
    },
    outputUrl() { return this.api ? this.api.fileUrl(this.path, OUTPUT, null) : ""; },
    events() {
      if (Array.isArray(this.fetched)) return this.fetched;
      return this.record && Array.isArray(this.record.events) ? this.record.events : [];
    },
    kinds() {
      const found = [];
      for (const event of this.events) {
        if (event && event.kind && !found.includes(event.kind)) found.push(event.kind);
      }
      return found;
    },
    groups() {
      const groups = [];
      for (const event of this.events) {
        if (!event || (this.kind && event.kind !== this.kind)) continue;
        const turn = event.turn === undefined || event.turn === null ? null : event.turn;
        const last = groups[groups.length - 1];
        if (last && last.turn === turn) last.events.push(event);
        else groups.push({ key: String(groups.length), turn, events: [event] });
      }
      return groups.reverse().map(group => ({ ...group, events: group.events.slice().reverse() }));
    },
    countText() {
      const asked = Array.isArray(this.fetched) ? ", up to " + LIMIT + " asked" : ", the latest the record carries";
      return this.events.length + " events" + asked;
    },
    button() { return Array.isArray(this.fetched) ? "refresh" : "load the last " + LIMIT; },
  },
  watch: {
    record() {
      if (Array.isArray(this.fetched)) this.load();
      this.loadOutput();
    },
  },
  mounted() { this.loadOutput(); },
  methods: {
    async loadOutput() {
      if (!this.api || !this.hasOutput) return;
      const path = this.path;
      try {
        const text = await this.api.getText(this.api.fileUrl(path, OUTPUT, TAIL));
        if (path !== this.path) return;
        this.output = text;
        this.outputError = "";
      } catch (error) {
        if (path === this.path) this.outputError = messageOf(error);
      }
    },
    moment(value) { return when(value); },
    title(group) { return group.turn === null ? "no turn" : "turn " + group.turn; },
    tone(kind) { return { "--tone": TONES[kind] || "var(--muted)", minWidth: "84px", textAlign: "center" }; },
    details(event) {
      return Object.entries(event).filter(([key]) => !SKIP.includes(key)).map(([key, value]) => key + " " + shown(value))
        .join(", ");
    },
    async load() {
      if (!this.api) {
        this.error = "the api prop is missing";
        return;
      }
      const path = this.path;
      this.loading = true;
      try {
        const found = await this.api.getJson(this.api.recordUrl(path, "events", { limit: LIMIT }));
        if (path !== this.path) return;
        this.fetched = Object.freeze(Array.isArray(found) ? found : []);
        this.error = "";
      } catch (error) {
        if (path === this.path) this.error = messageOf(error);
      } finally {
        this.loading = false;
      }
    },
  },
  template: `
  <div class="page pane-log">
    <section class="card" v-if="hasOutput">
      <div class="card-head">
        <span class="card-title">output</span>
        <span class="card-tools">
          <span class="muted small">the last {{ tail }} lines of log.txt, followed while the record runs</span>
          <a class="tool" :href="outputUrl" download="log.txt">save</a>
        </span>
      </div>
      <div class="error-box" v-if="outputError">{{ outputError }}</div>
      <pre class="mono" :style="code">{{ output }}</pre>
    </section>
    <div class="muted small">The events of the record from events.jsonl, newest first and grouped by turn: the start and the end
      of the run, checkpoints, the patches rules applied, stops and failures.</div>
    <div class="toolbar">
      <button type="button" class="tool" :disabled="loading" @click="load">{{ button }}</button>
      <select class="filter" v-model="kind" style="width: auto">
        <option value="">every kind</option>
        <option v-for="item in kinds" :key="item" :value="item">{{ item }}</option>
      </select>
      <span class="muted small">{{ countText }}</span>
      <span class="muted small" v-if="loading">reading</span>
    </div>
    <div class="error-box" v-if="error">{{ error }}</div>
    <div class="empty" v-if="!groups.length">no events yet</div>
    <section class="card" v-for="group in groups" :key="group.key">
      <div class="card-head"><span class="card-title">{{ title(group) }}</span><span class="muted small">{{ group.events.length }}</span></div>
      <div v-for="(event, index) in group.events" :key="index" :style="row">
        <span class="badge" :style="tone(event.kind)">{{ event.kind || "event" }}</span>
        <span class="muted mono">{{ moment(event.at) }}</span>
        <span class="mono" style="overflow-wrap: anywhere">{{ details(event) }}</span>
      </div>
    </section>
  </div>`,
};
