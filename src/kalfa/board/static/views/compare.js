import { computed, onBeforeUnmount, onMounted, ref }
  from "https://cdn.jsdelivr.net/npm/vue@3.4.38/dist/vue.esm-browser.prod.js";

import { getJson, query } from "../api.js";
import { Badge } from "../components/badge.js";
import { Chart } from "../components/chart.js";
import { ErrorBox } from "../components/error.js";
import { Progress } from "../components/progress.js";
import { bestText, seconds, text } from "../format.js";
import { compareLink, onLive, recordLink, touches } from "../store.js";


export function listOf(value) {
  return String(value || "").split(",").map((item) => item.trim()).filter(Boolean);
}

export function tablesOf(series) {
  return Object.keys(series || {}).filter((key) => key !== "turn").map((key) => ({ key, table: series[key] || {} }));
}

function hasValue(values) {
  return Array.isArray(values) && values.some((value) => value !== null && value !== undefined);
}

export function seriesKeys(tables) {
  const keys = [];
  for (const entry of tables) {
    for (const key of Object.keys(entry.table)) {
      if (!keys.includes(key)) keys.push(key);
    }
  }
  return keys;
}

export function diffLines(textValue) {
  return String(textValue || "").split("\n").map((line) => {
    let tone = "";
    if (line.startsWith("+++") || line.startsWith("---")) tone = "head";
    else if (line.startsWith("@@")) tone = "hunk";
    else if (line.startsWith("+")) tone = "add";
    else if (line.startsWith("-")) tone = "del";
    return { line, tone };
  });
}

export const Compare = {
  name: "CompareView",
  components: { KBadge: Badge, KChart: Chart, KError: ErrorBox, KProgress: Progress },
  props: { query: { type: Object, default: () => ({}) } },
  setup(props) {
    const paths = listOf(props.query.records);
    const keys = ref(props.query.keys || "");
    const answer = ref(null);
    const error = ref("");
    const loading = ref(false);
    let ticket = 0;
    let off = null;

    const load = async () => {
      if (!paths.length) return;
      const mine = ++ticket;
      loading.value = true;
      try {
        const found = await getJson("/api/compare" + query({ records: paths, keys: listOf(keys.value) }));
        if (mine !== ticket) return;
        answer.value = found;
        error.value = "";
      } catch (failure) {
        if (mine === ticket) error.value = failure.message;
      } finally {
        if (mine === ticket) loading.value = false;
      }
    };

    const labels = computed(() => {
      const found = answer.value || {};
      const names = Array.isArray(found.names) ? found.names : [];
      const records = Array.isArray(found.records) ? found.records : [];
      return tablesOf(found.series).map((entry, index) => names[index] || (records[index] || {}).name || paths[index]
        || entry.key);
    });
    const tables = computed(() => tablesOf((answer.value || {}).series));
    const turns = computed(() => ((answer.value || {}).series || {}).turn || []);
    const common = computed(() => seriesKeys(tables.value)
      .filter((key) => tables.value.every((entry) => hasValue(entry.table[key]))));
    const partial = computed(() => seriesKeys(tables.value).filter((key) => !common.value.includes(key)
      && tables.value.some((entry) => hasValue(entry.table[key]))));
    const charts = computed(() => common.value.map((key) => ({
      key,
      lines: tables.value.map((entry, index) => ({ name: labels.value[index], x: turns.value, y: entry.table[key] })),
    })));
    const records = computed(() => ((answer.value || {}).records || []).map((summary, index) => ({
      summary, path: paths[index] || summary.path, label: labels.value[index] || summary.name,
    })));
    const differing = computed(() => Object.entries(((answer.value || {}).params || {}).differing || {}));
    const shared = computed(() => Object.entries(((answer.value || {}).params || {}).shared || {}));
    const diffs = computed(() => Object.entries((answer.value || {}).diffs || {})
      .map(([title, body]) => ({ title, body, lines: body === null || body === undefined ? [] : diffLines(body) })));
    const apply = () => {
      window.location.hash = compareLink(paths, listOf(keys.value).join(","));
    };

    onMounted(() => {
      load();
      off = onLive((changed) => {
        if (touches(changed, paths)) load();
      });
    });
    onBeforeUnmount(() => {
      ticket += 1;
      if (off) off();
    });
    return {
      paths, keys, answer, error, loading, charts, partial, records, differing, shared, diffs, apply, text, seconds,
      bestText, recordLink,
    };
  },
  template: `
    <section class="page compare">
      <div class="page-head">
        <h2 class="section-title">compare</h2>
        <span class="muted small">{{ paths.length }} records</span>
      </div>
      <div v-if="!paths.length" class="empty">pick two or more records on the <a class="link" href="#/runs">runs page</a>
        and press compare</div>
      <template v-else>
        <form class="toolbar" @submit.prevent="apply">
          <label class="small muted" for="compare-keys">series</label>
          <input id="compare-keys" class="filter" v-model="keys" placeholder="val/rmse, train/mse (blank: every series of the first record)">
          <button type="submit" class="button">apply</button>
        </form>
        <k-error :error="error"></k-error>
        <div v-if="loading && !answer" class="loading">reading the records</div>
        <template v-if="answer">
          <div class="table-wrap">
            <table class="data">
              <thead><tr><th>record</th><th>status</th><th>progress</th><th>best</th><th>seconds</th></tr></thead>
              <tbody>
                <tr v-for="item in records" :key="item.path">
                  <td><a class="link" :href="recordLink(item.path, 'overview')">{{ item.label }}</a></td>
                  <td><k-badge :status="item.summary.status"></k-badge></td>
                  <td><k-progress :value="item.summary.turn" :total="item.summary.total"></k-progress></td>
                  <td class="num">{{ bestText(item.summary.best) }}</td>
                  <td class="num">{{ seconds(item.summary.seconds) }}</td>
                </tr>
              </tbody>
            </table>
          </div>
          <h3 class="section-title">series</h3>
          <div v-if="!charts.length" class="empty">the records share no series with values</div>
          <div class="grid">
            <k-chart v-for="chart in charts" :key="chart.key" :title="chart.key" :data="chart.lines"
              :options="{ xlabel: 'turn' }"></k-chart>
          </div>
          <div v-if="partial.length" class="muted small">not in every record: {{ partial.join(", ") }}</div>
          <h3 class="section-title">params that differ</h3>
          <div v-if="!differing.length" class="empty">every param is the same</div>
          <div v-else class="table-wrap">
            <table class="data">
              <thead><tr><th>param</th><th v-for="item in records" :key="item.path">{{ item.label }}</th></tr></thead>
              <tbody>
                <tr v-for="pair in differing" :key="pair[0]">
                  <td class="mono">{{ pair[0] }}</td>
                  <td v-for="(value, index) in pair[1]" :key="index" class="num">{{ text(value) }}</td>
                </tr>
              </tbody>
            </table>
          </div>
          <div v-if="shared.length" class="muted small shared">shared:
            <span v-for="pair in shared" :key="pair[0]" class="chip">{{ pair[0] }} = {{ text(pair[1]) }}</span>
          </div>
          <h3 class="section-title">config diff</h3>
          <div v-if="!diffs.length" class="empty">one record has nothing to diff against</div>
          <div v-for="diff in diffs" :key="diff.title" class="card">
            <div class="card-head"><span class="card-title">{{ diff.title }}</span></div>
            <div v-if="diff.body === null || diff.body === undefined" class="empty">a resolved.yaml is missing</div>
            <div v-else-if="!diff.body" class="empty">the resolved configs are the same</div>
            <pre v-else class="diff"><span v-for="(item, index) in diff.lines" :key="index" :class="item.tone">{{ item.line }}</span></pre>
          </div>
        </template>
      </template>
    </section>`,
};

export default Compare;
