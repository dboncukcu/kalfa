import { computed } from "https://cdn.jsdelivr.net/npm/vue@3.4.38/dist/vue.esm-browser.prod.js";

import { ErrorBox } from "../components/error.js";
import { Table } from "../components/table.js";
import { EMPTY, seconds, text, when } from "../format.js";
import {
  compareLink,
  isSweepEntry,
  nameOf,
  progressOf,
  recordLink,
  setSelected,
  statusOf,
  store,
  sweepLink,
} from "../store.js";


const summaryOf = (entry) => entry.summary || {};
const bestOf = (entry) => summaryOf(entry).best || {};
const paramsOf = (entry) => summaryOf(entry).params || {};

export function paramKeys(entries) {
  const keys = [];
  for (const entry of entries) {
    for (const key of Object.keys(paramsOf(entry))) {
      if (!keys.includes(key)) keys.push(key);
    }
  }
  return keys;
}

export function recordColumns(entries) {
  const columns = [
    { key: "name", label: "name", value: nameOf, href: (entry) => recordLink(entry.path, "overview"),
      title: (entry) => entry.path },
    { key: "status", label: "status", value: statusOf, badge: true },
    { key: "progress", label: "turn/total", value: (entry) => summaryOf(entry).turn,
      progress: (entry) => progressOf(entry), numeric: true },
    { key: "seconds", label: "seconds", value: (entry) => summaryOf(entry).seconds,
      text: (entry) => seconds(summaryOf(entry).seconds), numeric: true },
    { key: "best_series", label: "best of", value: (entry) => bestOf(entry).series },
    { key: "best", label: "best", value: (entry) => bestOf(entry).value, numeric: true,
      title: (entry) => (bestOf(entry).turn ? "at turn " + bestOf(entry).turn : "") },
    { key: "started", label: "started", value: (entry) => summaryOf(entry).started,
      text: (entry) => when(summaryOf(entry).started), descending: true },
  ];
  for (const key of paramKeys(entries)) {
    columns.push({ key: "param:" + key, label: key, value: (entry) => paramsOf(entry)[key],
      text: (entry) => text(paramsOf(entry)[key]), numeric: true, hint: "param " + key });
  }
  return columns;
}

export function failedText(sweep) {
  const parts = [];
  if (sweep.failed) parts.push(sweep.failed + " failed");
  if (sweep.stale) parts.push(sweep.stale + " stale");
  return parts.join(", ") || EMPTY;
}

export function sweepColumns() {
  const sweepOf = (entry) => entry.sweep || {};
  const objectiveOf = (entry) => sweepOf(entry).objective || {};
  return [
    { key: "name", label: "sweep", value: (entry) => entry.path, href: (entry) => sweepLink(entry.path) },
    { key: "status", label: "status", value: statusOf, badge: true },
    { key: "points", label: "points done", value: (entry) => sweepOf(entry).done,
      progress: (entry) => progressOf(entry), numeric: true },
    { key: "running", label: "running", value: (entry) => sweepOf(entry).running, numeric: true },
    { key: "failed", label: "failed", value: (entry) => sweepOf(entry).failed, numeric: true,
      hint: "points that failed or went stale", text: (entry) => failedText(sweepOf(entry)) },
    { key: "strategy", label: "strategy", value: (entry) => (sweepOf(entry).strategy || {}).use },
    { key: "objective", label: "objective", value: (entry) => objectiveOf(entry).watch,
      text: (entry) => (objectiveOf(entry).watch ? objectiveOf(entry).watch + " (" + (objectiveOf(entry).mode || "min")
        + ")" : EMPTY) },
    { key: "started", label: "started", value: (entry) => sweepOf(entry).started,
      text: (entry) => when(sweepOf(entry).started), descending: true },
  ];
}

export const Runs = {
  name: "RunsView",
  components: { KError: ErrorBox, KTable: Table },
  setup() {
    const records = computed(() => store.records.filter((entry) => !isSweepEntry(entry)));
    const sweeps = computed(() => store.records.filter(isSweepEntry));
    const columns = computed(() => recordColumns(records.value));
    const sweepTable = sweepColumns();
    const chosen = computed(() => store.selected.filter((path) => records.value.some((entry) => entry.path === path)));
    const compareHref = computed(() => compareLink(chosen.value));
    const select = (paths) => setSelected(paths);
    const clear = () => setSelected([]);
    return { store, records, sweeps, columns, sweepTable, chosen, compareHref, select, clear };
  },
  template: `
    <section class="page runs">
      <k-error :error="store.error"></k-error>
      <div v-if="!store.loaded" class="loading">reading the records</div>
      <template v-else>
        <div class="page-head">
          <h2 class="section-title">records</h2>
          <span class="muted small mono">{{ store.root }}</span>
        </div>
        <k-table :columns="columns" :rows="records" row-key="path" selectable :selected="store.selected"
          @update:selected="select" sort-key="started" :sort-desc="true" placeholder="filter by name, status or param"
          empty="no records under this root">
          <template #tools>
            <span class="spacer"></span>
            <span class="muted small" v-if="chosen.length">{{ chosen.length }} selected</span>
            <a class="button" :class="{ disabled: !chosen.length }" :href="chosen.length ? compareHref : null">compare</a>
            <button type="button" class="button ghost" v-if="chosen.length" @click="clear">clear</button>
          </template>
        </k-table>
        <template v-if="sweeps.length">
          <h2 class="section-title">sweeps</h2>
          <k-table :columns="sweepTable" :rows="sweeps" row-key="path" :filterable="false" sort-key="started"
            :sort-desc="true"></k-table>
        </template>
      </template>
    </section>`,
};

export default Runs;
