import Chart from "../../components/chart.js";
import { messageOf } from "../../components/error.js";


const SETS = ["train", "val", "test"];
const FAMILIES = ["loss", "lr", "grad_norm"];
const AXES = ["turn", "update"];
const LIMIT = 2000;

function historyName(key) {
  const cut = key.indexOf("/");
  return cut > 0 && SETS.includes(key.slice(0, cut)) ? key.slice(cut + 1) : key;
}

function updateName(key) {
  const cut = key.indexOf("/");
  const family = cut > 0 ? key.slice(0, cut) : key;
  return FAMILIES.includes(family) ? family : null;
}

function groupsOf(table, axis, nameOf) {
  if (!table || !Array.isArray(table[axis])) return [];
  const groups = new Map();
  for (const key of Object.keys(table)) {
    if (AXES.includes(key) || !Array.isArray(table[key])) continue;
    const name = nameOf(key);
    if (name === null) continue;
    if (!groups.has(name)) groups.set(name, []);
    groups.get(name).push({ name: key, x: table[axis], y: table[key] });
  }
  return Array.from(groups, ([title, series]) => ({ title, series }));
}

export default {
  name: "CurvesTab",
  components: { "k-chart": Chart },
  props: {
    record: { type: Object, default: null },
    path: { type: String, default: "" },
    api: { default: null },
    version: { type: Number, default: 0 },
  },
  data() {
    return { history: null, steps: null, error: "", loading: false, logy: false, filter: "", ruled: true };
  },
  computed: {
    fired() { return this.record && Array.isArray(this.record.marks) ? this.record.marks : []; },
    historyGroups() { return groupsOf(this.history, "turn", historyName); },
    stepGroups() { return groupsOf(this.steps, "update", updateName); },
    historyCharts() { return this.decorate(this.historyGroups, "turn"); },
    stepCharts() { return this.decorate(this.stepGroups, "update"); },
    turns() { return this.history && Array.isArray(this.history.turn) ? this.history.turn.length : 0; },
    updates() { return this.steps && Array.isArray(this.steps.update) ? this.steps.update.length : 0; },
    countText() { return this.turns + " turns, " + this.updates + " optimizer updates"; },
    waiting() { return this.loading && !this.history; },
  },
  watch: {
    record() { this.load(); },
  },
  mounted() { this.load(); },
  methods: {
    marksOn(axis) {
      if (!this.ruled) return [];
      return this.fired
        .map(mark => ({ x: mark[axis], text: (mark.rules || []).join(", ") }))
        .filter(mark => typeof mark.x === "number");
    },
    decorate(groups, xlabel) {
      const wanted = this.filter.trim().toLowerCase();
      const logy = this.logy;
      const marks = this.marksOn(xlabel);
      return groups
        .filter(group => !wanted || group.title.toLowerCase().includes(wanted))
        .map(group => ({
          key: group.title + (logy ? " log" : ""),
          title: group.title,
          series: group.series,
          options: { xlabel, ylabel: group.title, logy, marks },
        }));
    },
    toggleLog() { this.logy = !this.logy; },
    toggleRules() { this.ruled = !this.ruled; },
    async load() {
      if (!this.api) {
        this.error = "the api prop is missing";
        return;
      }
      const path = this.path;
      this.loading = true;
      try {
        const found = await Promise.all([
          this.api.getJson(this.api.recordUrl(path, "series", { limit: LIMIT })),
          this.api.getJson(this.api.recordUrl(path, "updates", { limit: LIMIT })),
        ]);
        if (path !== this.path) return;
        this.history = Object.freeze(found[0] || {});
        this.steps = Object.freeze(found[1] || {});
        this.error = "";
      } catch (error) {
        if (path === this.path) this.error = messageOf(error);
      } finally {
        this.loading = false;
      }
    },
  },
  template: `
  <div class="page pane-curves">
    <div class="muted small">One chart per series of history.jsonl, one point per turn; train, val and test of the same name share
      a chart. Below, the optimizer updates of updates.jsonl: the batch loss, the learning rate and the gradient norm of every
      optimizer. A dotted line marks every turn a rule fired, with the names of the rules. Expand opens a chart large.</div>
    <div class="toolbar">
      <input class="filter" type="search" v-model="filter" placeholder="filter charts" spellcheck="false">
      <button type="button" class="tool" :class="{ on: logy }" @click="toggleLog">log y on every chart</button>
      <button type="button" class="tool" :class="{ on: ruled }" v-if="fired.length" @click="toggleRules">rule marks</button>
      <button type="button" class="tool" :disabled="loading" @click="load">refresh</button>
      <span class="muted small">{{ countText }}</span>
    </div>
    <div class="error-box" v-if="error">{{ error }}</div>
    <div class="loading" v-if="waiting">reading the series</div>
    <template v-if="history">
      <h3 class="section-title">per turn</h3>
      <div class="grid" v-if="historyCharts.length">
        <k-chart v-for="chart in historyCharts" :key="chart.key" :title="chart.title" kind="lines" :data="chart.series"
          :options="chart.options"></k-chart>
      </div>
      <div class="empty" v-else>no series written yet</div>
    </template>
    <template v-if="steps">
      <h3 class="section-title">per optimizer update</h3>
      <div class="grid" v-if="stepCharts.length">
        <k-chart v-for="chart in stepCharts" :key="chart.key" :title="chart.title" kind="lines" :data="chart.series"
          :options="chart.options"></k-chart>
      </div>
      <div class="empty" v-else>no update written yet</div>
    </template>
  </div>`,
};
