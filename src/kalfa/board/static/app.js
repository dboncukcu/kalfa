import { computed, createApp, onMounted, ref, watch }
  from "https://cdn.jsdelivr.net/npm/vue@3.4.38/dist/vue.esm-browser.prod.js";

import {
  EVERY,
  TABS,
  applyTheme,
  compareLink,
  cycleTheme,
  isActive,
  latestSweep,
  refresh,
  setEvery,
  startLive,
  store,
  sweepLink,
  watchScheme,
} from "./store.js";
import { Compare } from "./views/compare.js";
import { Home } from "./views/home.js";
import { Record } from "./views/record.js";
import { Runs } from "./views/runs.js";
import { Sweep } from "./views/sweep.js";


const PAGES = ["home", "runs", "compare", "sweep", "record"];

function decodePart(part) {
  try {
    return decodeURIComponent(part);
  } catch (error) {
    return part;
  }
}

export function parseHash(hash) {
  const raw = String(hash || "").replace(/^#/, "").replace(/^\//, "");
  const mark = raw.indexOf("?");
  const route = mark >= 0 ? raw.slice(0, mark) : raw;
  const query = Object.fromEntries(new URLSearchParams(mark >= 0 ? raw.slice(mark + 1) : ""));
  const parts = route.split("/").filter(Boolean).map(decodePart);
  const page = parts.length ? parts[0] : "home";
  const rest = parts.slice(1);
  if (page === "record") {
    const named = rest.length > 1 && TABS.includes(rest[rest.length - 1]);
    const tab = named ? rest[rest.length - 1] : "overview";
    const path = (named ? rest.slice(0, -1) : rest).join("/");
    return { page, path, tab, query, raw };
  }
  if (page === "sweep") return { page, path: rest.join("/"), tab: "", query, raw };
  return { page: PAGES.includes(page) ? page : "missing", path: "", tab: "", query, raw };
}

const Board = {
  name: "KalfaBoard",
  components: { HomeView: Home, RunsView: Runs, CompareView: Compare, SweepView: Sweep, RecordView: Record },
  setup() {
    const route = ref(parseHash(window.location.hash));
    const onHash = () => {
      store.fault = "";
      route.value = parseHash(window.location.hash);
    };
    const running = computed(() => store.records.filter(isActive).length);
    const sweepHref = computed(() => {
      const sweep = latestSweep();
      return sweep ? sweepLink(sweep.path) : "";
    });
    const compareHref = computed(() => compareLink(store.selected));
    const liveTitle = computed(() => (store.connected ? "live: the server reports changes every " + store.every + "s"
      : "not connected to the live stream, retrying"));
    const changeEvery = (event) => setEvery(event.target.value);

    watch(running, (count) => {
      document.title = count ? "kalfa board (" + count + " running)" : "kalfa board";
    }, { immediate: true });
    onMounted(() => {
      window.addEventListener("hashchange", onHash);
      watchScheme();
      refresh();
      startLive();
    });
    return { store, route, running, sweepHref, compareHref, liveTitle, changeEvery, cycleTheme, EVERY };
  },
  template: `
    <header class="topbar">
      <a class="brand" href="#/">kalfa</a>
      <nav class="nav">
        <a :class="{ active: route.page === 'home' }" href="#/">home</a>
        <a :class="{ active: route.page === 'runs' }" href="#/runs">runs</a>
        <a :class="{ active: route.page === 'compare' }" :href="compareHref">compare<span v-if="store.selected.length" class="count">{{ store.selected.length }}</span></a>
        <a v-if="sweepHref" :class="{ active: route.page === 'sweep' }" :href="sweepHref">sweep</a>
        <span v-else class="nav-off" title="no sweep under this root">sweep</span>
      </nav>
      <span class="root mono" :title="store.root">{{ store.root }}</span>
      <span class="spacer"></span>
      <span class="live" :class="{ on: store.connected }" :title="liveTitle">
        <span class="dot"></span>{{ store.connected ? "live" : "offline" }}<span v-if="running" class="count">{{ running }}</span>
      </span>
      <select class="every" :value="store.every" @change="changeEvery" title="how often the server looks for changes">
        <option v-for="value in EVERY" :key="value" :value="value">{{ value }}s</option>
      </select>
      <button type="button" class="tool" @click="cycleTheme" title="theme: auto follows the system">{{ store.theme }}</button>
    </header>
    <div v-if="store.fault" class="error-box page-fault">{{ store.fault }}</div>
    <main class="main">
      <home-view v-if="route.page === 'home'"></home-view>
      <runs-view v-else-if="route.page === 'runs'"></runs-view>
      <compare-view v-else-if="route.page === 'compare'" :key="'compare:' + route.raw" :query="route.query"></compare-view>
      <sweep-view v-else-if="route.page === 'sweep'" :key="'sweep:' + route.path" :path="route.path"></sweep-view>
      <record-view v-else-if="route.page === 'record'" :key="'record:' + route.path" :path="route.path" :tab="route.tab"></record-view>
      <div v-else class="empty">there is no page at #/{{ route.raw }}; go <a class="link" href="#/">home</a></div>
    </main>`,
};

const app = createApp(Board);
app.config.errorHandler = (error) => {
  console.error(error);
  store.fault = "the page hit an error: " + (error && error.message ? error.message : String(error));
};
applyTheme();
app.mount("#app");
window.kalfaMounted = true;
