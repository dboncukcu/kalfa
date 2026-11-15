const CODE = "overflow: auto; max-height: 75vh; padding: 10px; background: var(--surface-2); border-radius: 6px; white-space: pre";

export default {
  name: "ConfigTab",
  props: {
    record: { type: Object, default: null },
    path: { type: String, default: "" },
    api: { default: null },
    version: { type: Number, default: 0 },
  },
  data() {
    return { notice: "", timer: null };
  },
  computed: {
    code() { return CODE; },
    text() { return this.record && typeof this.record.config === "string" ? this.record.config : ""; },
    lines() { return this.text ? this.text.split("\n").length : 0; },
    raw() { return this.api ? this.api.fileUrl(this.path, "resolved.yaml") : ""; },
    mutable() {
      const plan = (this.record && this.record.plan) || {};
      return Array.isArray(plan.mutable) ? plan.mutable : [];
    },
  },
  beforeUnmount() {
    if (this.timer) clearTimeout(this.timer);
  },
  methods: {
    flash(message) {
      this.notice = message;
      if (this.timer) clearTimeout(this.timer);
      this.timer = setTimeout(() => {
        this.notice = "";
        this.timer = null;
      }, 1800);
    },
    selectText() {
      const box = this.$refs.yaml;
      const selection = window.getSelection ? window.getSelection() : null;
      if (!box || !selection) return false;
      const range = document.createRange();
      range.selectNodeContents(box);
      selection.removeAllRanges();
      selection.addRange(range);
      try {
        return Boolean(document.execCommand && document.execCommand("copy"));
      } catch (error) {
        return false;
      }
    },
    async copy() {
      try {
        await navigator.clipboard.writeText(this.text);
        this.flash("copied");
      } catch (error) {
        this.flash(this.selectText() ? "copied" : "selected; copy it with the keyboard");
      }
    },
  },
  template: `
  <div class="page pane-config">
    <div class="muted small">The config as the run resolved it (resolved.yaml), after the sets and params of the command line;
      continuing the record reads this file.</div>
    <div class="empty" v-if="!text">this record has no resolved.yaml</div>
    <section class="card" v-else>
      <div class="card-head">
        <span class="card-title mono">resolved.yaml</span>
        <span class="card-tools">
          <span class="muted small">{{ lines }} lines</span>
          <span class="muted small" v-if="notice">{{ notice }}</span>
          <button type="button" class="tool" @click="copy">copy</button>
          <a class="tool" :href="raw" download="resolved.yaml">save</a>
        </span>
      </div>
      <pre class="mono" ref="yaml" :style="code">{{ text }}</pre>
    </section>
    <section class="card" v-if="mutable.length">
      <div class="card-head"><span class="card-title">mutable paths</span><span class="muted small">what a rule may patch while the run trains</span></div>
      <div><span class="chip mono" v-for="item in mutable" :key="item">{{ item }}</span></div>
    </section>
  </div>`,
};
