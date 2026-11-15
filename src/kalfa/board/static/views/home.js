import { computed } from "https://cdn.jsdelivr.net/npm/vue@3.4.38/dist/vue.esm-browser.prod.js";

import { Badge } from "../components/badge.js";
import { ErrorBox } from "../components/error.js";
import { Progress } from "../components/progress.js";
import { EMPTY, ago, bestText, liveNote, nonfiniteText, seconds, when } from "../format.js";
import { isActive, isSweepEntry, linkOf, nameOf, progressOf, statusOf, store } from "../store.js";


const RECENT = 20;

export function metaOf(entry) {
  if (isSweepEntry(entry)) {
    const sweep = entry.sweep || {};
    const strategy = sweep.strategy && sweep.strategy.use ? sweep.strategy.use : "sweep";
    const parts = [strategy + " sweep", sweep.running ? sweep.running + " running" : "none running"];
    if (sweep.stale) parts.push(sweep.stale + " stale");
    if (sweep.failed) parts.push(sweep.failed + " failed");
    return parts.join(", ");
  }
  const summary = entry.summary || {};
  const parts = [];
  if (liveNote(summary)) parts.push(liveNote(summary));
  if (summary.best && summary.best.series) parts.push("best " + bestText(summary.best));
  if (summary.updated) parts.push("updated " + ago(summary.updated));
  if (summary.seconds) parts.push(seconds(summary.seconds));
  return parts.join(", ") || EMPTY;
}

export const Home = {
  name: "HomeView",
  components: { KBadge: Badge, KError: ErrorBox, KProgress: Progress },
  setup() {
    const active = computed(() => store.records.filter(isActive));
    const recent = computed(() => store.records.filter((entry) => !isActive(entry)).slice(0, RECENT));
    const kindOf = (entry) => (isSweepEntry(entry) ? "sweep" : "record");
    const startedOf = (entry) => when(isSweepEntry(entry) ? (entry.sweep || {}).started : (entry.summary || {}).started);
    const failureOf = (entry) => (entry.summary && entry.summary.failure) || "";
    const warningOf = (entry) => nonfiniteText(entry.summary && entry.summary.nonfinite);
    return {
      store, active, recent, kindOf, startedOf, failureOf, warningOf, linkOf, nameOf, progressOf, statusOf, metaOf,
    };
  },
  template: `
    <section class="page home">
      <k-error :error="store.error"></k-error>
      <div v-if="!store.loaded" class="loading">reading the records</div>
      <template v-else>
        <h2 class="section-title">running</h2>
        <div v-if="!active.length" class="empty">nothing is running under {{ store.root || "the root" }}</div>
        <div class="live-grid">
          <a v-for="entry in active" :key="entry.path" class="card live-card" :href="linkOf(entry)">
            <div class="live-head">
              <span class="live-name">{{ nameOf(entry) }}</span>
              <k-badge :status="statusOf(entry)"></k-badge>
            </div>
            <div class="muted small mono">{{ entry.path }}</div>
            <k-progress :value="progressOf(entry).value" :total="progressOf(entry).total" :part="progressOf(entry).part"
              :label="progressOf(entry).value + ' / ' + (progressOf(entry).total || '?') + ' ' + progressOf(entry).unit">
            </k-progress>
            <div class="muted small">{{ metaOf(entry) }}</div>
            <div v-if="warningOf(entry)" class="fail-text small">{{ warningOf(entry) }}</div>
          </a>
        </div>
        <h2 class="section-title">recent</h2>
        <div v-if="!recent.length" class="empty">no other records under {{ store.root || "the root" }}</div>
        <div v-else class="table-wrap">
          <table class="data">
            <thead>
              <tr><th>name</th><th>kind</th><th>status</th><th>progress</th><th>started</th><th>details</th></tr>
            </thead>
            <tbody>
              <tr v-for="entry in recent" :key="entry.path">
                <td><a class="link" :href="linkOf(entry)">{{ nameOf(entry) }}</a>
                  <div class="muted small mono" v-if="entry.path !== nameOf(entry)">{{ entry.path }}</div></td>
                <td class="muted">{{ kindOf(entry) }}</td>
                <td><k-badge :status="statusOf(entry)"></k-badge></td>
                <td><k-progress :value="progressOf(entry).value" :total="progressOf(entry).total"></k-progress></td>
                <td class="muted">{{ startedOf(entry) }}</td>
                <td class="small">{{ metaOf(entry) }} <span v-if="failureOf(entry)" class="fail-text">{{ failureOf(entry) }}</span>
                  <span v-if="warningOf(entry)" class="fail-text">{{ warningOf(entry) }}</span></td>
              </tr>
            </tbody>
          </table>
        </div>
      </template>
    </section>`,
};

export default Home;
