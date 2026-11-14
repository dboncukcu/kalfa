import { computed, defineAsyncComponent, onBeforeUnmount, onErrorCaptured, onMounted, ref, watch }
  from "https://cdn.jsdelivr.net/npm/vue@3.4.38/dist/vue.esm-browser.prod.js";

import { client as api } from "../api.js";
import { Badge } from "../components/badge.js";
import { ErrorBox } from "../components/error.js";
import { Loading } from "../components/loading.js";
import { Progress } from "../components/progress.js";
import { bestText, liveNote, nonfiniteText, seconds, when } from "../format.js";
import { TABS, entryOf, isSweepEntry, onLive, recordLink, sweepLink, touches } from "../store.js";


function lazy(loader) {
  return defineAsyncComponent({ loader, errorComponent: ErrorBox, loadingComponent: Loading, delay: 150 });
}

export const TAB_VIEWS = {
  overview: lazy(() => import("./record/overview.js")),
  curves: lazy(() => import("./record/curves.js")),
  model: lazy(() => import("./record/model.js")),
  data: lazy(() => import("./record/data.js")),
  predictions: lazy(() => import("./record/predictions.js")),
  plots: lazy(() => import("./record/plots.js")),
  config: lazy(() => import("./record/config.js")),
  files: lazy(() => import("./record/files.js")),
  log: lazy(() => import("./record/log.js")),
};
const STOPPABLE = ["running", "stale"];

export function parentOf(path) {
  const parts = String(path || "").split("/").filter(Boolean);
  return parts.slice(0, -1).join("/");
}

export const Record = {
  name: "RecordView",
  components: { KBadge: Badge, KError: ErrorBox, KProgress: Progress },
  props: {
    path: { type: String, default: "" },
    tab: { type: String, default: "overview" },
  },
  setup(props) {
    const record = ref(null);
    const error = ref("");
    const loading = ref(false);
    const version = ref(0);
    const tabError = ref("");
    const stopping = ref(false);
    const stopNote = ref("");
    let ticket = 0;
    let off = null;

    const isRecord = async (path) => {
      try {
        await api.getJson(api.recordUrl(path));
        return true;
      } catch (failure) {
        return false;
      }
    };
    const load = async () => {
      const mine = ++ticket;
      loading.value = true;
      try {
        const found = await api.getJson(api.recordUrl(props.path));
        if (mine !== ticket) return;
        record.value = found;
        version.value += 1;
        error.value = "";
      } catch (failure) {
        if (mine !== ticket) return;
        const longer = props.path + "/" + props.tab;
        if (failure.status === 404 && await isRecord(longer)) {
          window.location.hash = recordLink(longer, "overview");
          return;
        }
        error.value = failure.message;
      } finally {
        if (mine === ticket) loading.value = false;
      }
    };

    const summary = computed(() => (record.value && record.value.summary) || {});
    const progress = computed(() => (record.value && record.value.progress) || {});
    const best = computed(() => (record.value && record.value.best) || summary.value.best || null);
    const status = computed(() => summary.value.status || progress.value.status || "unknown");
    const current = computed(() => (TABS.includes(props.tab) ? props.tab : "overview"));
    const view = computed(() => TAB_VIEWS[current.value] || TAB_VIEWS.overview);
    const canStop = computed(() => STOPPABLE.includes(status.value));
    const asSweep = computed(() => isSweepEntry(entryOf(props.path)));
    const sweepParent = computed(() => (summary.value.sweep ? parentOf(props.path) : ""));
    const tabLink = (tab) => recordLink(props.path, tab);
    const note = computed(() => liveNote(progress.value));
    const warning = computed(() => nonfiniteText(summary.value.nonfinite));

    const stop = async () => {
      if (stopping.value) return;
      stopping.value = true;
      stopNote.value = "";
      try {
        const written = await api.postJson(api.recordUrl(props.path, "stop"), {});
        const paths = (written && written.paths) || [];
        stopNote.value = paths.length ? "asked to stop: " + paths.join(", ") : "nothing to stop";
        load();
      } catch (failure) {
        stopNote.value = "";
        error.value = failure.message;
      } finally {
        stopping.value = false;
      }
    };

    watch(() => props.tab, () => {
      tabError.value = "";
    });
    onErrorCaptured((failure) => {
      tabError.value = "the " + current.value + " tab failed: " + (failure && failure.message ? failure.message
        : String(failure));
      return false;
    });
    onMounted(() => {
      load();
      off = onLive((changed) => {
        if (touches(changed, [props.path])) load();
      });
    });
    onBeforeUnmount(() => {
      ticket += 1;
      if (off) off();
    });
    return {
      api, record, error, loading, version, tabError, stopping, stopNote, summary, progress, best, status, current,
      view, canStop, asSweep, sweepParent, tabLink, note, warning, stop, TABS, bestText, seconds, when, sweepLink,
    };
  },
  template: `
    <section class="page record">
      <k-error :error="error"></k-error>
      <div v-if="asSweep" class="notice">
        <span class="mono">{{ path }}</span> is a sweep: <a class="link" :href="sweepLink(path)">open the sweep page</a>
      </div>
      <div v-if="loading && !record && !error" class="loading">reading the record</div>
      <template v-if="record">
        <header class="record-head card">
          <div class="record-title">
            <h2>{{ summary.name || path }}</h2>
            <k-badge :status="status"></k-badge>
            <span class="spacer"></span>
            <button v-if="canStop" type="button" class="button danger" :disabled="stopping" @click="stop"
              title="write stop.json; the run stops after its current turn">{{ stopping ? "stopping" : "stop" }}</button>
          </div>
          <div class="record-meta">
            <span class="mono muted">{{ path }}</span>
            <span v-if="sweepParent">point of <a class="link" :href="sweepLink(sweepParent)">{{ sweepParent }}</a></span>
            <k-progress :value="progress.turn || summary.turn || 0" :total="progress.total || summary.total || null"
              :part="progress.fraction"
              :label="(progress.turn || summary.turn || 0) + ' / ' + (progress.total || summary.total || '?') + ' ' + (progress.unit || summary.unit || 'turns')">
            </k-progress>
            <span v-if="note" class="muted">{{ note }}</span>
            <span v-if="best && best.series">best <b>{{ bestText(best) }}</b><span v-if="best.turn" class="muted"> at turn {{ best.turn }}</span></span>
            <span class="muted">{{ seconds(progress.seconds || summary.seconds) }}</span>
            <span class="muted">started {{ when(summary.started) }}</span>
          </div>
          <div v-if="summary.failure" class="error-box">{{ summary.failure }}</div>
          <div v-if="warning" class="error-box">{{ warning }}</div>
          <div v-if="stopNote" class="notice small">{{ stopNote }}</div>
        </header>
        <nav class="tabs">
          <a v-for="name in TABS" :key="name" class="tab" :class="{ active: name === current }" :href="tabLink(name)">{{ name }}</a>
        </nav>
        <k-error :error="tabError"></k-error>
        <div class="tab-body">
          <component :is="view" :key="current" :record="record" :path="path" :api="api" :version="version"></component>
        </div>
      </template>
    </section>`,
};

export default Record;
