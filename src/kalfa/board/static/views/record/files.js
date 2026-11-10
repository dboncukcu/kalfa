import { messageOf } from "../../components/error.js";
import { bytes, when } from "../../format.js";


const TEXTS = [".yaml", ".yml", ".json", ".jsonl", ".txt", ".md", ".csv", ".py", ".sh", ".sub", ".log"];
const IMAGES = [".png", ".jpg", ".jpeg", ".gif", ".svg"];
const TAIL = 200;
const LINK = "border: 0; background: none; padding: 0; font: inherit; cursor: pointer; text-align: left";
const CODE = "overflow: auto; max-height: 70vh; padding: 10px; background: var(--surface-2); border-radius: 6px; white-space: pre";

function kindOf(name) {
  const lower = String(name).toLowerCase();
  if (TEXTS.some(suffix => lower.endsWith(suffix))) return "text";
  if (IMAGES.some(suffix => lower.endsWith(suffix))) return "image";
  return "binary";
}

function basename(name) {
  const parts = String(name).split("/");
  return parts[parts.length - 1];
}

export default {
  name: "FilesTab",
  props: {
    record: { type: Object, default: null },
    path: { type: String, default: "" },
    api: { default: null },
    version: { type: Number, default: 0 },
  },
  data() {
    return { filter: "", view: null, error: "", loading: false };
  },
  computed: {
    link() { return LINK; },
    code() { return CODE; },
    tail() { return TAIL; },
    all() { return this.record && Array.isArray(this.record.files) ? this.record.files : []; },
    files() {
      const wanted = this.filter.trim().toLowerCase();
      return wanted ? this.all.filter(item => String(item.path).toLowerCase().includes(wanted)) : this.all;
    },
    countText() {
      if (this.files.length === this.all.length) return this.all.length + " files";
      return this.files.length + " of " + this.all.length + " files";
    },
    partial() { return Boolean(this.view && this.view.kind === "text" && !this.view.whole); },
    splitClass() { return this.view ? "grid" : ""; },
  },
  watch: {
    record() {
      if (this.partial) this.open(this.view.name, false, true);
    },
  },
  methods: {
    bytes,
    basename,
    moment(value) { return when(value); },
    url(name, tail) { return this.api ? this.api.fileUrl(this.path, name, tail || null) : ""; },
    isOpen(name) { return Boolean(this.view && this.view.name === name); },
    rowClass(name) { return this.isOpen(name) ? "selected" : ""; },
    close() { this.view = null; },
    async open(name, whole, quiet) {
      const kind = kindOf(name);
      if (kind !== "text") {
        this.view = { name, kind, text: "", whole: true };
        return;
      }
      if (!this.api) {
        this.error = "the api prop is missing";
        return;
      }
      const path = this.path;
      if (!quiet) this.loading = true;
      try {
        const text = await this.api.getText(this.url(name, whole ? 0 : TAIL));
        if (path !== this.path) return;
        this.view = { name, kind, text, whole: Boolean(whole) };
        this.error = "";
      } catch (error) {
        if (path === this.path) this.error = messageOf(error);
      } finally {
        this.loading = false;
      }
    },
  },
  template: `
  <div class="page pane-files">
    <div class="muted small">Every file of the record with its size and the time it last changed. A name opens it here: the last
      lines of a text file, images inline; save downloads any file as it is. An open text file follows the record while it runs.</div>
    <div class="toolbar">
      <input class="filter" type="search" v-model="filter" placeholder="filter files" spellcheck="false">
      <span class="muted small">{{ countText }}</span>
      <span class="muted small" v-if="loading">reading</span>
    </div>
    <div class="error-box" v-if="error">{{ error }}</div>
    <div :class="splitClass" style="align-items: start">
      <div class="table-wrap">
        <table class="data">
          <thead><tr><th>file</th><th class="num">size</th><th>modified</th><th></th></tr></thead>
          <tbody>
            <tr v-for="item in files" :key="item.path" :class="rowClass(item.path)">
              <td class="mono"><button type="button" class="link" :style="link" @click="open(item.path, false)">{{ item.path }}</button></td>
              <td class="num mono">{{ bytes(item.size) }}</td>
              <td class="muted">{{ moment(item.modified) }}</td>
              <td><a class="tool" :href="url(item.path, 0)" :download="basename(item.path)">save</a></td>
            </tr>
            <tr v-if="!files.length"><td class="empty" colspan="4">no files</td></tr>
          </tbody>
        </table>
      </div>
      <section class="card" v-if="view">
        <div class="card-head">
          <span class="card-title mono">{{ view.name }}</span>
          <span class="card-tools">
            <span class="muted small" v-if="partial">the last {{ tail }} lines</span>
            <button type="button" class="tool" v-if="partial" @click="open(view.name, true)">whole file</button>
            <a class="tool" :href="url(view.name, 0)" :download="basename(view.name)">save</a>
            <button type="button" class="tool" @click="close">close</button>
          </span>
        </div>
        <pre class="mono" v-if="view.kind === 'text'" :style="code">{{ view.text }}</pre>
        <img v-else-if="view.kind === 'image'" :src="url(view.name, 0)" :alt="view.name" style="max-width: 100%; background: #fff">
        <div class="muted" v-else>{{ view.name }} is not shown here; save downloads it</div>
      </section>
    </div>
  </div>`,
};
