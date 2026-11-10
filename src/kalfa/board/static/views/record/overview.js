import Badge from "../../components/badge.js";
import Table from "../../components/table.js";
import { number, percent, seconds, when } from "../../format.js";


const LAST_COLUMNS = [
  { key: "name", label: "series", className: "mono" },
  { key: "value", label: "last value", numeric: true },
];
const PARAM_COLUMNS = [
  { key: "name", label: "param", className: "mono" },
  { key: "value", label: "value", numeric: true },
];

function present(value) {
  return value !== null && value !== undefined && value !== "";
}

function shown(value) {
  if (!present(value)) return "";
  if (typeof value === "number") return number(value);
  if (Array.isArray(value)) return "[" + value.map(shown).join(", ") + "]";
  if (typeof value === "object") {
    if (typeof value.ref === "string") return "@" + value.ref;
    return JSON.stringify(value);
  }
  return String(value);
}

function described(value) {
  if (value && typeof value === "object" && !Array.isArray(value) && typeof value.use === "string") {
    const rest = Object.entries(value).filter(([key]) => key !== "use").map(([key, item]) => key + " " + shown(item));
    return [value.use].concat(rest).join(", ");
  }
  return shown(value);
}

export default {
  name: "OverviewTab",
  components: { "k-badge": Badge, "k-table": Table },
  props: {
    record: { type: Object, default: null },
    path: { type: String, default: "" },
    api: { default: null },
    version: { type: Number, default: 0 },
  },
  computed: {
    lastColumns() { return LAST_COLUMNS; },
    paramColumns() { return PARAM_COLUMNS; },
    summary() { return (this.record && this.record.summary) || {}; },
    progress() { return (this.record && this.record.progress) || {}; },
    plan() { return (this.record && this.record.plan) || {}; },
    status() { return this.summary.status || this.progress.status || "unknown"; },
    best() { return this.summary.best || (this.record && this.record.best) || null; },
    bestValue() { return this.best && present(this.best.value) ? number(this.best.value) : "no value yet"; },
    bestMode() { return this.best && this.best.mode === "max" ? "highest" : "lowest"; },
    bestTurn() { return this.best && present(this.best.turn) ? String(this.best.turn) : ""; },
    turn() { return present(this.summary.turn) ? this.summary.turn : (this.progress.turn || 0); },
    total() { return present(this.summary.total) ? this.summary.total : this.progress.total; },
    unit() { return this.summary.unit || this.progress.unit || ""; },
    elapsed() { return present(this.summary.seconds) ? seconds(this.summary.seconds) : seconds(0); },
    started() { return present(this.summary.started) ? "started " + when(this.summary.started) : ""; },
    fraction() {
      const total = Number(this.total);
      const turn = Number(this.turn);
      if (!total || !Number.isFinite(total) || !Number.isFinite(turn)) return null;
      return Math.max(0, Math.min(1, turn / total));
    },
    hasFraction() { return this.fraction !== null; },
    fractionText() { return this.hasFraction ? percent(this.fraction) : ""; },
    fill() { return { width: (this.hasFraction ? (this.fraction * 100).toFixed(1) : "0") + "%" }; },
    turnText() { return present(this.total) ? "turn " + this.turn + " of " + this.total : "turn " + this.turn; },
    lastRows() {
      const watched = this.best ? this.best.series : null;
      return Object.entries(this.summary.last || {}).map(([name, value]) => ({ name, value, best: name === watched }));
    },
    manyLast() { return this.lastRows.length > 8; },
    minimizes() { return Object.entries(this.summary.minimizes || {}).map(([name, loss]) => ({ name, loss: String(loss) })); },
    params() {
      return Object.entries(this.summary.params || {}).map(([name, value]) => ({
        name,
        value: typeof value === "number" || typeof value === "string" ? value : shown(value),
      }));
    },
    manyParams() { return this.params.length > 8; },
    fired() { return Array.isArray(this.summary.rules) ? this.summary.rules : []; },
    rules() {
      const fired = this.fired;
      const planned = Array.isArray(this.plan.rules) ? this.plan.rules : [];
      return planned.map((rule, index) => ({
        key: rule.id || String(index),
        id: rule.id || "rule " + (index + 1),
        fired: fired.includes(rule.id),
        when: described(rule.when),
        patch: (Array.isArray(rule.patch) ? rule.patch : []).map(item => item.path + " = " + shown(item.value)).join("; "),
        stop: Boolean(rule.stop),
        once: Boolean(rule.once),
      }));
    },
    identity() {
      const summary = this.summary;
      const sweep = summary.sweep;
      const rows = [
        ["name", summary.name],
        ["device", summary.device],
        ["host", summary.host],
        ["started", present(summary.started) ? when(summary.started) : ""],
        ["updated", present(summary.updated) ? when(summary.updated) : ""],
        ["finished", present(summary.finished) ? when(summary.finished) : ""],
        ["elapsed", present(summary.seconds) ? seconds(summary.seconds) : ""],
        ["failure", summary.failure],
        ["stopped by", summary.stopped_by],
        ["stop reason", summary.stop_reason],
        ["resumed from", summary.resumed_from],
        ["sweep point", sweep && present(sweep.id) ? String(sweep.id) : ""],
        ["seed", this.plan.seed],
        ["kalfa", this.plan.kalfa],
      ];
      return rows.filter(row => present(row[1])).map(([name, value]) => ({ name, value: shown(value) }));
    },
  },
  methods: {
    bestRow(row) { return row && row.best ? "selected" : ""; },
  },
  template: `
  <div class="page pane-overview">
    <div class="error-box" v-if="summary.failure">failed: {{ summary.failure }}</div>
    <div class="notice-box" v-if="summary.stop_reason">stopped: {{ summary.stop_reason }}</div>
    <div class="cards">
      <section class="card">
        <div class="stat-label">status</div>
        <div class="stat-value"><k-badge :status="status"></k-badge></div>
        <span class="progress" v-if="hasFraction" style="width: 100%">
          <span class="progress-track"><span class="progress-fill" :style="fill"></span></span>
          <span class="progress-text">{{ fractionText }}</span>
        </span>
        <div class="muted small">{{ turnText }}<span v-if="unit"> {{ unit }}</span></div>
      </section>
      <section class="card">
        <div class="stat-label">best</div>
        <div class="stat-value mono">{{ bestValue }}</div>
        <div class="muted small" v-if="best">{{ best.series }}, {{ bestMode }} is best<span v-if="bestTurn">, reached at turn {{ bestTurn }}</span></div>
        <div class="muted small" v-else>the plan names no checkpoint watch series</div>
      </section>
      <section class="card">
        <div class="stat-label">elapsed</div>
        <div class="stat-value">{{ elapsed }}</div>
        <div class="muted small">{{ started }}</div>
      </section>
      <section class="card">
        <div class="stat-label">rules fired</div>
        <div class="stat-value">{{ fired.length }}</div>
        <div class="muted small" v-if="fired.length"><span class="chip mono" v-for="name in fired" :key="name">{{ name }}</span></div>
        <div class="muted small" v-else>no rule has fired yet</div>
      </section>
    </div>
    <div class="grid">
      <section class="card">
        <div class="card-head"><span class="card-title">last values</span><span class="muted small">the last line of history.jsonl</span></div>
        <k-table :columns="lastColumns" :rows="lastRows" row-key="name" :filterable="manyLast" placeholder="filter series"
          :row-class="bestRow" empty="no values yet"></k-table>
        <div class="muted small" v-for="row in minimizes" :key="row.name">optimizer {{ row.name }} minimizes {{ row.loss }}</div>
      </section>
      <section class="card">
        <div class="card-head"><span class="card-title">params</span></div>
        <k-table :columns="paramColumns" :rows="params" row-key="name" :filterable="manyParams" placeholder="filter params"
          empty="no params"></k-table>
      </section>
      <section class="card">
        <div class="card-head"><span class="card-title">identity</span></div>
        <div class="table-wrap">
          <table class="data">
            <tbody>
              <tr v-for="row in identity" :key="row.name"><td class="muted">{{ row.name }}</td><td class="mono">{{ row.value }}</td></tr>
            </tbody>
          </table>
        </div>
      </section>
    </div>
    <section class="card" v-if="rules.length">
      <div class="card-head"><span class="card-title">rules of the plan</span></div>
      <div class="table-wrap">
        <table class="data">
          <thead><tr><th>rule</th><th>when</th><th>patch</th><th>fired</th></tr></thead>
          <tbody>
            <tr v-for="rule in rules" :key="rule.key" :class="{ selected: rule.fired }">
              <td class="mono">{{ rule.id }}<span class="muted" v-if="rule.once"> once</span><span class="muted" v-if="rule.stop"> stops</span></td>
              <td class="mono">{{ rule.when }}</td>
              <td class="mono">{{ rule.patch }}</td>
              <td>{{ rule.fired ? "yes" : "" }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>
  </div>`,
};
