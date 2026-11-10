import { reactive } from "https://cdn.jsdelivr.net/npm/vue@3.4.38/dist/vue.esm-browser.prod.js";

import { encodePath, getJson, live } from "./api.js";


export const TABS = ["overview", "curves", "model", "data", "predictions", "plots", "config", "files", "log"];
export const ACTIVE = ["running", "stale"];
export const EVERY = [1, 2, 5, 10, 30];
export const THEMES = ["auto", "light", "dark"];

function readSetting(key, fallback) {
  try {
    const value = window.localStorage.getItem("kalfa." + key);
    return value === null ? fallback : value;
  } catch (error) {
    return fallback;
  }
}

function writeSetting(key, value) {
  try {
    window.localStorage.setItem("kalfa." + key, String(value));
  } catch (error) {
    return;
  }
}

function startEvery() {
  const value = Number(readSetting("every", "2"));
  return EVERY.includes(value) ? value : 2;
}

function startTheme() {
  const value = readSetting("theme", "auto");
  return THEMES.includes(value) ? value : "auto";
}

export const store = reactive({
  root: "",
  records: [],
  loaded: false,
  error: "",
  selected: [],
  version: 0,
  changed: [],
  connected: false,
  every: startEvery(),
  theme: startTheme(),
  paint: 0,
  fault: "",
});

const listeners = new Set();
let stopLive = null;
let loading = 0;

export function onLive(callback) {
  listeners.add(callback);
  return () => listeners.delete(callback);
}

function notify(changed) {
  for (const callback of [...listeners]) {
    try {
      callback(changed);
    } catch (error) {
      console.error(error);
    }
  }
}

export async function refresh() {
  const ticket = ++loading;
  try {
    const answer = await getJson("/api/records");
    if (ticket !== loading) return;
    store.root = answer && answer.root ? answer.root : "";
    store.records = answer && Array.isArray(answer.records) ? answer.records : [];
    store.error = "";
  } catch (error) {
    if (ticket === loading) store.error = error.message;
  } finally {
    store.loaded = true;
  }
}

function changedHandler(changed) {
  store.version += 1;
  store.changed = Array.isArray(changed) ? changed : [];
  refresh();
  notify(changed);
}

export function startLive() {
  if (stopLive) stopLive();
  stopLive = live(changedHandler, (connected) => {
    store.connected = connected;
  }, store.every);
}

export function setEvery(value) {
  const every = Number(value);
  store.every = EVERY.includes(every) ? every : 2;
  writeSetting("every", store.every);
  startLive();
}

export function applyTheme() {
  const root = document.documentElement;
  if (store.theme === "auto") root.removeAttribute("data-theme");
  else root.setAttribute("data-theme", store.theme);
  store.paint += 1;
}

export function cycleTheme() {
  const next = THEMES[(THEMES.indexOf(store.theme) + 1) % THEMES.length];
  store.theme = next;
  writeSetting("theme", next);
  applyTheme();
}

export function watchScheme() {
  if (!window.matchMedia) return;
  const scheme = window.matchMedia("(prefers-color-scheme: dark)");
  const repaint = () => {
    if (store.theme === "auto") store.paint += 1;
  };
  if (scheme.addEventListener) scheme.addEventListener("change", repaint);
}

export function touches(changed, paths) {
  if (!Array.isArray(changed)) return true;
  if (!changed.length) return false;
  if (paths.some((path) => path === "" || path === ".")) return true;
  return changed.some((item) => paths.some((path) => item === path || item.startsWith(path + "/")
    || path.startsWith(item + "/")));
}

export function setSelected(paths) {
  store.selected = [...new Set((paths || []).filter(Boolean))];
}

export function toggleSelected(path) {
  if (store.selected.includes(path)) setSelected(store.selected.filter((item) => item !== path));
  else setSelected([...store.selected, path]);
}

export function recordLink(path, tab = "overview") {
  return "#/record/" + encodePath(path) + "/" + (tab || "overview");
}

export function sweepLink(path) {
  return "#/sweep/" + encodePath(path);
}

export function compareLink(paths, keys = "") {
  const search = new URLSearchParams();
  search.set("records", (paths || []).join(","));
  if (keys) search.set("keys", keys);
  return "#/compare?" + search.toString();
}

export function isSweepEntry(entry) {
  return Boolean(entry) && entry.kind === "sweep";
}

export function statusOf(entry) {
  if (!entry) return "unknown";
  if (isSweepEntry(entry)) {
    const sweep = entry.sweep || {};
    if (sweep.running > 0) return "running";
    if (sweep.stale > 0) return "stale";
    const planned = sweep.points || sweep.count || 0;
    if (sweep.count > 0 && sweep.done >= sweep.count && sweep.count >= planned) return "done";
    return sweep.count > 0 ? "partial" : "empty";
  }
  return (entry.summary && entry.summary.status) || "unknown";
}

export function isActive(entry) {
  return ACTIVE.includes(statusOf(entry));
}

export function nameOf(entry) {
  if (!entry) return "";
  if (entry.summary && entry.summary.name) return entry.summary.name;
  return entry.path;
}

export function progressOf(entry) {
  if (isSweepEntry(entry)) {
    const sweep = entry.sweep || {};
    return { value: sweep.done || 0, total: sweep.points || sweep.count || null, unit: "points" };
  }
  const summary = (entry && entry.summary) || {};
  return { value: summary.turn || 0, total: summary.total || null, unit: summary.unit || "turns" };
}

export function linkOf(entry) {
  return isSweepEntry(entry) ? sweepLink(entry.path) : recordLink(entry.path, "overview");
}

export function entryOf(path) {
  return store.records.find((entry) => entry.path === path) || null;
}

export function latestSweep() {
  return store.records.find(isSweepEntry) || null;
}
