import { computed, onBeforeUnmount, onMounted, ref }
  from "https://cdn.jsdelivr.net/npm/vue@3.4.38/dist/vue.esm-browser.prod.js";

import { fileUrl, getJson, getText, postJson, sweepUrl } from "../api.js";
import { Chart } from "../components/chart.js";
import { ErrorBox } from "../components/error.js";
import { Table } from "../components/table.js";
import { EMPTY, number, text, when } from "../format.js";
import { onLive, recordLink, touches } from "../store.js";


const POLL = 1000;

export function pointPath(root, point) {
  return root && root !== "." ? root + "/" + point.path : point.path;
}

export function baseName(path) {
  const parts = String(path || "").split(/[\\/]/).filter(Boolean);
  return parts.length ? parts[parts.length - 1] : "";
}

export function isLogSpace(spec) {
  return Boolean(spec) && typeof spec === "object" && !Array.isArray(spec) && Array.isArray(spec.log);
}

export function pointColumns(root, answer) {
  const ranked = (answer && answer.ranked) || [];
  const keys = Object.keys((answer && answer.params) || {});
  const objective = ((answer && answer.sweep) || {}).objective || {};
  const columns = [
    { key: "id", label: "point", value: (point) => point.id, href: (point) => recordLink(pointPath(root, point)),
      numeric: true, title: (point) => point.path },
    { key: "rank", label: "rank", value: (point) => (ranked.indexOf(point.id) >= 0 ? ranked.indexOf(point.id) + 1 : null),
      numeric: true },
    { key: "status", label: "status", value: (point) => point.status, badge: true },
    { key: "progress", label: "turn/total", value: (point) => point.turn,
      progress: (point) => ({ value: point.turn || 0, total: point.total || null }), numeric: true },
    { key: "objective", label: objective.watch || "objective", value: (point) => point.objective, numeric: true,
      descending: objective.mode === "max", hint: "objective (" + (objective.mode || "min") + ")" },
  ];
  for (const key of keys) {
    columns.push({ key: "param:" + key, label: key, value: (point) => (point.values || {})[key],
      text: (point) => text((point.values || {})[key]), numeric: true });
  }
  return columns;
}

export function scatters(answer) {
  if (!answer) return [];
  const params = answer.params || {};
  const ids = answer.ids || [];
  const objectives = answer.objectives || [];
  const space = ((answer.sweep || {}).space) || {};
  const objective = ((answer.sweep || {}).objective) || {};
  const best = answer.best;
  return Object.keys(params).map((key) => {
    const groups = [{ name: "points", x: params[key], y: objectives, text: ids.map((id) => "point " + id) }];
    if (best && best.values && best.values[key] !== undefined) {
      groups.push({ name: "best", x: [best.values[key]], y: [best.objective], text: ["best: point " + best.id],
        size: 12 });
    }
    return { key, groups, options: { xlabel: key, ylabel: objective.watch || "objective", logx: isLogSpace(space[key]) } };
  });
}

export const Sweep = {
  name: "SweepView",
  components: { KChart: Chart, KError: ErrorBox, KTable: Table },
  props: { path: { type: String, default: "" } },
  setup(props) {
    const answer = ref(null);
    const error = ref("");
    const loading = ref(false);
    const job = ref(null);
    const report = ref({ status: "", path: "", error: "", url: "", name: "" });
    let ticket = 0;
    let timer = null;
    let off = null;
    let alive = true;

    const load = async () => {
      const mine = ++ticket;
      loading.value = true;
      try {
        const found = await getJson(sweepUrl(props.path));
        if (mine !== ticket) return;
        answer.value = found;
        error.value = "";
      } catch (failure) {
        if (mine === ticket) error.value = failure.message;
      } finally {
        if (mine === ticket) loading.value = false;
      }
    };

    const forget = () => {
      if (report.value.url) URL.revokeObjectURL(report.value.url);
    };
    const fetchReport = async (path) => {
      const name = baseName(path) || "report.html";
      try {
        const body = await getText(fileUrl(props.path, name));
        if (!alive) return;
        forget();
        const url = URL.createObjectURL(new Blob([body], { type: "text/html" }));
        report.value = { status: "done", path, error: "", url, name };
      } catch (failure) {
        report.value = { status: "done", path, error: "written, but the board cannot read it back: " + failure.message,
          url: "", name };
      }
    };
    const poll = async () => {
      timer = null;
      if (!alive || job.value === null) return;
      try {
        const state = await getJson("/api/report/" + encodeURIComponent(job.value));
        if (!alive) return;
        if (state.status === "running") {
          timer = setTimeout(poll, POLL);
          return;
        }
        job.value = null;
        if (state.status === "done") await fetchReport(state.path || "");
        else report.value = { status: "failed", path: "", error: state.error || "the report failed", url: "", name: "" };
      } catch (failure) {
        job.value = null;
        report.value = { status: "failed", path: "", error: failure.message, url: "", name: "" };
      }
    };
    const writeReport = async () => {
      if (job.value !== null) return;
      report.value = { status: "running", path: "", error: "", url: "", name: "" };
      try {
        const started = await postJson("/api/report", { records: [props.path], out: null });
        job.value = started && started.job !== undefined ? started.job : null;
        if (job.value === null) throw new Error("the server did not start a report job");
        timer = setTimeout(poll, POLL);
      } catch (failure) {
        report.value = { status: "failed", path: "", error: failure.message, url: "", name: "" };
      }
    };

    const info = computed(() => (answer.value && answer.value.sweep) || {});
    const objective = computed(() => info.value.objective || {});
    const points = computed(() => (answer.value && answer.value.points) || []);
    const best = computed(() => (answer.value && answer.value.best) || null);
    const columns = computed(() => pointColumns(props.path, answer.value));
    const charts = computed(() => scatters(answer.value));
    const counts = computed(() => {
      const found = {};
      for (const point of points.value) found[point.status || "unknown"] = (found[point.status || "unknown"] || 0) + 1;
      return Object.entries(found);
    });
    const bestLink = computed(() => (best.value ? recordLink(pointPath(props.path, best.value)) : ""));
    const valuesOf = (point) => Object.entries((point && point.values) || {});
    const strategy = computed(() => (info.value.strategy && info.value.strategy.use) || EMPTY);

    onMounted(() => {
      load();
      off = onLive((changed) => {
        if (touches(changed, [props.path])) load();
      });
    });
    onBeforeUnmount(() => {
      alive = false;
      ticket += 1;
      if (timer) clearTimeout(timer);
      if (off) off();
      forget();
    });
    return {
      answer, error, loading, job, report, info, objective, points, best, columns, charts, counts, bestLink, valuesOf,
      strategy, writeReport, number, text, when,
    };
  },
  template: `
    <section class="page sweep">
      <div class="page-head">
        <h2 class="section-title">sweep <span class="mono">{{ path }}</span></h2>
        <span class="spacer"></span>
        <button type="button" class="button" :disabled="report.status === 'running'" @click="writeReport">
          {{ report.status === "running" ? "writing report.html" : "write report.html" }}
        </button>
      </div>
      <div v-if="report.status === 'done'" class="notice">
        <span>report written to <span class="mono">{{ report.path }}</span></span>
        <a v-if="report.url" class="link" :href="report.url" target="_blank" rel="noopener">open</a>
        <a v-if="report.url" class="link" :href="report.url" :download="report.name">download</a>
      </div>
      <k-error :error="report.error"></k-error>
      <k-error :error="error"></k-error>
      <div v-if="loading && !answer" class="loading">reading the sweep</div>
      <template v-if="answer">
        <div class="cards">
          <div class="card stat">
            <div class="stat-label">strategy</div>
            <div class="stat-value">{{ strategy }}</div>
            <div class="muted small">started {{ when(info.started) }}</div>
          </div>
          <div class="card stat">
            <div class="stat-label">points</div>
            <div class="stat-value">{{ points.length }} / {{ info.points || points.length }}</div>
            <div class="muted small"><span v-for="pair in counts" :key="pair[0]" class="chip">{{ pair[1] }} {{ pair[0] }}</span></div>
          </div>
          <div class="card stat">
            <div class="stat-label">objective</div>
            <div class="stat-value mono">{{ objective.watch || "none" }}</div>
            <div class="muted small">{{ objective.mode || "min" }} at {{ objective.at || "best" }}</div>
          </div>
          <div class="card stat best-card">
            <div class="stat-label">best</div>
            <template v-if="best">
              <div class="stat-value"><a class="link" :href="bestLink">point {{ best.id }}</a> {{ number(best.objective) }}</div>
              <div class="small"><span v-for="pair in valuesOf(best)" :key="pair[0]" class="chip">{{ pair[0] }} = {{ text(pair[1]) }}</span></div>
            </template>
            <div v-else class="muted">no point has an objective yet</div>
          </div>
        </div>
        <h3 class="section-title">points</h3>
        <k-table :columns="columns" :rows="points" row-key="id" sort-key="rank" placeholder="filter points"
          empty="the sweep has no points yet"></k-table>
        <h3 class="section-title">objective against each param</h3>
        <div v-if="!charts.length" class="empty">the sweep has no params</div>
        <div class="grid">
          <k-chart v-for="chart in charts" :key="chart.key" kind="scatter" :title="chart.key" :data="chart.groups"
            :options="chart.options"></k-chart>
        </div>
      </template>
    </section>`,
};

export default Sweep;
