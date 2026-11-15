const IMAGES = [".png", ".jpg", ".jpeg", ".gif", ".svg"];
const THUMB = "display: block; width: 100%; height: 200px; object-fit: contain; cursor: zoom-in; background: #fff; border-radius: 4px";
const LARGE = "flex: 1; min-height: 0; width: 100%; object-fit: contain; background: #fff; border-radius: 4px";

function pathOf(item) {
  if (typeof item === "string") return item;
  return item && typeof item.path === "string" ? item.path : "";
}

function isImage(path) {
  const lower = path.toLowerCase();
  return IMAGES.some(suffix => lower.endsWith(suffix));
}

export default {
  name: "PlotsTab",
  props: {
    record: { type: Object, default: null },
    path: { type: String, default: "" },
    api: { default: null },
    version: { type: Number, default: 0 },
  },
  data() {
    return { shown: "" };
  },
  computed: {
    thumb() { return THUMB; },
    large() { return LARGE; },
    items() {
      const plots = this.record && Array.isArray(this.record.plots) ? this.record.plots : [];
      return plots.map(pathOf).filter(Boolean).map(path => {
        const parts = path.split("/");
        return { path, name: parts[parts.length - 1], folder: parts.length > 1 ? parts[0] : "", image: isImage(path) };
      });
    },
    groups() {
      const groups = new Map();
      for (const item of this.items) {
        if (!groups.has(item.folder)) groups.set(item.folder, []);
        groups.get(item.folder).push(item);
      }
      return Array.from(groups, ([folder, items]) => ({ folder: folder || "record", items }));
    },
    images() { return this.items.filter(item => item.image); },
    current() { return this.images.find(item => item.path === this.shown) || null; },
    position() {
      const index = this.images.findIndex(item => item.path === this.shown);
      return index < 0 ? "" : (index + 1) + " of " + this.images.length;
    },
  },
  mounted() { window.addEventListener("keydown", this.onKey); },
  beforeUnmount() { window.removeEventListener("keydown", this.onKey); },
  methods: {
    url(path) { return this.api ? this.api.fileUrl(this.path, path) : ""; },
    show(item) { if (item.image) this.shown = item.path; },
    close() { this.shown = ""; },
    step(delta) {
      const images = this.images;
      if (!images.length) return;
      const index = images.findIndex(item => item.path === this.shown);
      const next = (Math.max(index, 0) + delta + images.length) % images.length;
      this.shown = images[next].path;
    },
    onKey(event) {
      if (!this.current) return;
      if (event.key === "Escape") this.close();
      else if (event.key === "ArrowRight") this.step(1);
      else if (event.key === "ArrowLeft") this.step(-1);
    },
  },
  template: `
  <div class="page pane-plots">
    <div class="muted small">The figures the run drew under plots/ and the samples it generated under samples/. A click opens a
      figure large; the arrow keys move between figures and escape closes.</div>
    <div class="empty" v-if="!items.length">no figures yet; the plots section draws them when the run ends</div>
    <template v-for="group in groups" :key="group.folder">
      <h3 class="section-title">{{ group.folder }} <span class="muted small">{{ group.items.length }}</span></h3>
      <div class="live-grid">
        <figure class="card" v-for="item in group.items" :key="item.path" style="margin: 0">
          <img v-if="item.image" :src="url(item.path)" :alt="item.name" :title="item.path" loading="lazy" :style="thumb"
            @click="show(item)">
          <a v-else class="tool" :href="url(item.path)" :download="item.name">save {{ item.name }}</a>
          <figcaption class="mono muted" style="margin-top: 6px; overflow-wrap: anywhere">{{ item.name }}</figcaption>
        </figure>
      </div>
    </template>
    <div class="modal" v-if="current" @click.self="close">
      <div class="modal-box">
        <div class="card-head">
          <span class="card-title mono">{{ current.path }}</span>
          <span class="card-tools">
            <span class="muted small">{{ position }}</span>
            <button type="button" class="tool" @click="step(-1)">previous</button>
            <button type="button" class="tool" @click="step(1)">next</button>
            <a class="tool" :href="url(current.path)" :download="current.name">save</a>
            <button type="button" class="tool" @click="close">close</button>
          </span>
        </div>
        <img :src="url(current.path)" :alt="current.name" :style="large">
      </div>
    </div>
  </div>`,
};
