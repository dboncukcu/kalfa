import { Diagram } from "../../components/diagram.js";
import Table from "../../components/table.js";
import { integer } from "../../format.js";


const FACTS = ["kind", "labels", "image_shape", "mode", "transform", "train_transform", "drops", "spectators", "filter"];
const SET_ORDER = ["train", "valid", "test"];
const CROWDED = 40;
const COLUMN_COLUMNS = [
  { key: "name", label: "column", className: "mono" },
  { key: "steps", label: "steps", className: "mono", sortable: false },
];

function present(value) {
  if (value === null || value === undefined || value === "") return false;
  return !(Array.isArray(value) && !value.length);
}

function shown(value) {
  if (!present(value)) return "";
  if (Array.isArray(value)) return value.map(shown).join(", ");
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function listOf(value) {
  if (Array.isArray(value)) return value.map(String);
  if (value && typeof value === "object") return Object.keys(value);
  return present(value) ? [String(value)] : [];
}

function chainOf(value) {
  return Array.isArray(value) ? value.map(shown).join(" → ") : shown(value);
}

function fieldRow(name, value) {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return {
      name,
      type: shown(value.type || value.dtype),
      role: value.target ? "target" : "input",
      columns: listOf(value.columns),
      chain: chainOf(value.preprocessors || value.chain),
    };
  }
  return { name, type: shown(value), role: "", columns: [], chain: "" };
}

function isMapping(value) {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function setOrder(names) {
  const known = SET_ORDER.filter(name => names.includes(name));
  return [...known, ...names.filter(name => !SET_ORDER.includes(name))];
}

function perSet(value, sets) {
  if (!isMapping(value)) return false;
  const keys = Object.keys(value);
  return keys.length > 0 && keys.every(key => sets.includes(key) && (value[key] === null || typeof value[key] === "number"));
}

export default {
  name: "DataTab",
  components: { KDiagram: Diagram, KTable: Table },
  props: {
    record: { type: Object, default: null },
    path: { type: String, default: "" },
    api: { default: null },
    version: { type: Number, default: 0 },
  },
  data() {
    return { selected: null };
  },
  created() {
    this.drawn = null;
  },
  watch: {
    path() { this.selected = null; },
  },
  computed: {
    dataReport() {
      const found = this.record && this.record.data_report;
      return isMapping(found) ? found : null;
    },
    report() { return this.dataReport && isMapping(this.dataReport.report) ? this.dataReport.report : {}; },
    kind() { return this.dataReport ? shown(this.dataReport.kind) : ""; },
    pipeline() {
      const found = this.dataReport && this.dataReport.pipeline;
      if (!isMapping(found) || !Array.isArray(found.nodes) || !found.nodes.length) return null;
      const key = JSON.stringify(found);
      if (this.drawn && this.drawn.key === key) return this.drawn.graph;
      const graph = { nodes: found.nodes, edges: Array.isArray(found.edges) ? found.edges : [], groups: Array.isArray(found.groups) ? found.groups : [] };
      this.drawn = { key, graph };
      return graph;
    },
    crowded() { return Boolean(this.pipeline) && this.pipeline.nodes.length > CROWDED; },
    collapsed() {
      if (!this.crowded) return [];
      return this.pipeline.groups.filter(group => !group.parent).map(group => group.id);
    },
    diagramHeight() { return this.crowded ? 560 : 420; },
    diagramTitle() {
      const name = String(this.path || "").split("/").filter(Boolean).pop() || "record";
      return name + "_data";
    },
    groupLabels() {
      const labels = {};
      for (const group of this.pipeline ? this.pipeline.groups : []) labels[group.id] = shown(group.label);
      return labels;
    },
    selectedRows() {
      const node = this.selected;
      if (!node) return [];
      const rows = [{ name: "kind", value: shown(node.kind) }];
      if (present(node.detail)) rows.push({ name: "detail", value: shown(node.detail) });
      if (typeof node.count === "number") rows.push({ name: "count", value: integer(node.count) });
      if (node.group) rows.push({ name: "group", value: this.groupLabels[node.group] || shown(node.group) });
      const into = this.pipeline ? this.pipeline.edges.filter(edge => edge.to === node.id && present(edge.label)) : [];
      if (into.length) rows.push({ name: "arrives as", value: into.map(edge => shown(edge.label)).join(", ") });
      return rows;
    },
    setCounts() { return isMapping(this.report.rows) ? this.report.rows : {}; },
    setExtras() {
      const known = [...Object.keys(this.setCounts), ...SET_ORDER];
      return Object.keys(this.report).filter(key => key !== "rows" && perSet(this.report[key], known));
    },
    sets() {
      const names = new Set(Object.keys(this.setCounts));
      for (const key of this.setExtras) Object.keys(this.report[key]).forEach(name => names.add(name));
      return setOrder([...names]);
    },
    setRows() {
      return this.sets.map(name => {
        const row = { name, rows: this.setCounts[name] };
        for (const key of this.setExtras) row[key] = this.report[key][name];
        return row;
      });
    },
    setTotal() { return Object.values(this.setCounts).reduce((sum, count) => sum + (Number(count) || 0), 0); },
    columnColumns() { return COLUMN_COLUMNS; },
    columnRows() {
      if (!isMapping(this.report.columns)) return [];
      return Object.entries(this.report.columns).map(([name, steps]) => ({ name, steps: chainOf(steps) || "as read" }));
    },
    manyColumns() { return this.columnRows.length > 12; },
    reportFacts() {
      const skipped = new Set(["rows", "columns", ...this.setExtras]);
      return Object.entries(this.report)
        .filter(([key, value]) => !skipped.has(key) && present(value))
        .map(([name, value]) => ({ name, value: shown(value) }));
    },
    plan() { return (this.record && this.record.plan) || {}; },
    planData() { return this.plan.data && typeof this.plan.data === "object" ? this.plan.data : {}; },
    source() {
      const nodes = this.plan.nodes && typeof this.plan.nodes === "object" ? this.plan.nodes : {};
      return nodes.data && typeof nodes.data === "object" ? nodes.data : null;
    },
    sourceRows() {
      const params = this.source && this.source.params && typeof this.source.params === "object" ? this.source.params : {};
      return Object.entries(params).map(([name, value]) => ({ name, value: shown(value) }));
    },
    targets() {
      const params = (this.source && this.source.params) || {};
      return listOf(params.targets);
    },
    inputs() { return listOf(this.planData.inputs); },
    exposes() { return listOf(this.planData.exposes); },
    classes() { return listOf(this.planData.classes); },
    fieldRows() {
      const fields = this.planData.fields && typeof this.planData.fields === "object" ? this.planData.fields : {};
      const targets = this.targets;
      return Object.entries(fields).map(([name, value]) => {
        const row = fieldRow(name, value);
        if (!row.role && targets.includes(name)) row.role = "target";
        return row;
      });
    },
    fitted() { return this.record && this.record.fields && typeof this.record.fields === "object" ? this.record.fields : null; },
    features() {
      const features = this.fitted && Array.isArray(this.fitted.features) ? this.fitted.features : [];
      return features.map((item, index) => ({
        key: String(index),
        pattern: shown(item.pattern),
        columns: listOf(item.columns),
        names: listOf(item.names),
        chain: chainOf(item.chain),
      }));
    },
    targetFields() {
      const fields = this.fitted && Array.isArray(this.fitted.target_fields) ? this.fitted.target_fields : [];
      return fields.map((item, index) => ({
        key: String(index),
        name: shown(item.name),
        width: shown(item.width),
        chain: chainOf(item.chain),
        classes: listOf(item.classes),
        weights: Array.isArray(item.class_weights) ? item.class_weights.map(shown).join(", ") : "",
      }));
    },
    fittedTargets() {
      if (!this.fitted || this.targetFields.length) return [];
      return listOf(this.fitted.targets);
    },
    fittedClasses() { return this.fitted ? listOf(this.fitted.classes) : []; },
    featureCount() { return this.fitted && Array.isArray(this.fitted.feature_names) ? this.fitted.feature_names.length : 0; },
    facts() {
      if (!this.fitted) return [];
      return FACTS.filter(name => present(this.fitted[name])).map(name => ({ name, value: shown(this.fitted[name]) }));
    },
  },
  methods: {
    integer,
    pick(node) { this.selected = node || null; },
  },
  template: `
  <div class="page pane-data">
    <template v-if="dataReport">
      <div class="muted small">The data as the run read it (data.json): where the rows come from, how they are filtered and split
        into sets, the chains fitted on the train rows and the tensors every batch carries. Click a box for its details.
        Below, the sets and columns the run reported, then the data as the plan saw it and as the run fitted it.</div>
      <section class="card" v-if="pipeline">
        <div class="card-head"><span class="card-title">pipeline</span><span class="muted mono" v-if="kind">{{ kind }}</span></div>
        <k-diagram :graph="pipeline" direction="LR" :height="diagramHeight" :collapsed="collapsed" :title="diagramTitle"
          @select="pick"></k-diagram>
        <div class="table-wrap" v-if="selected">
          <table class="data">
            <thead><tr><th colspan="2">{{ selected.label }}</th></tr></thead>
            <tbody>
              <tr v-for="row in selectedRows" :key="row.name">
                <td class="muted">{{ row.name }}</td>
                <td class="mono" style="white-space: normal; overflow-wrap: anywhere">{{ row.value }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>
      <div class="empty" v-else>the run could not draw the data flowchart; the tables below still describe the data</div>
      <div class="grid">
        <section class="card" v-if="setRows.length">
          <div class="card-head"><span class="card-title">sets</span><span class="muted small">{{ integer(setTotal) }} rows in all</span></div>
          <div class="table-wrap">
            <table class="data">
              <thead><tr><th>set</th><th class="num">rows</th><th class="num" v-for="key in setExtras" :key="key">{{ key }}</th></tr></thead>
              <tbody>
                <tr v-for="row in setRows" :key="row.name">
                  <td class="mono">{{ row.name }}</td>
                  <td class="num mono">{{ integer(row.rows) }}</td>
                  <td class="num mono" v-for="key in setExtras" :key="key">{{ integer(row[key]) }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>
        <section class="card" v-if="reportFacts.length">
          <div class="card-head"><span class="card-title">report</span></div>
          <div class="table-wrap">
            <table class="data">
              <tbody>
                <tr v-for="row in reportFacts" :key="row.name">
                  <td class="muted">{{ row.name }}</td>
                  <td class="mono" style="white-space: normal; overflow-wrap: anywhere">{{ row.value }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>
      </div>
      <section class="card" v-if="columnRows.length">
        <div class="card-head"><span class="card-title">columns into the model</span><span class="muted small">{{ columnRows.length }}</span></div>
        <k-table :columns="columnColumns" :rows="columnRows" row-key="name" :filterable="manyColumns" placeholder="filter columns"
          empty="no columns"></k-table>
      </section>
    </template>
    <div class="muted small" v-if="!dataReport">The data as the plan saw it before training (plan.json) and as the run fitted it (fitted/fields.json):
      the feature blocks with their columns and preprocessor chains, the targets and their classes.</div>
    <div class="grid">
      <section class="card">
        <div class="card-head"><span class="card-title">source</span><span class="muted mono" v-if="source">{{ source.use }}</span></div>
        <div class="table-wrap" v-if="sourceRows.length">
          <table class="data">
            <tbody>
              <tr v-for="row in sourceRows" :key="row.name">
                <td class="muted">{{ row.name }}</td>
                <td class="mono" style="white-space: normal; overflow-wrap: anywhere">{{ row.value }}</td>
              </tr>
            </tbody>
          </table>
        </div>
        <div class="empty" v-if="!source">the plan has no data node</div>
      </section>
      <section class="card">
        <div class="card-head"><span class="card-title">wiring</span></div>
        <div class="table-wrap">
          <table class="data">
            <tbody>
              <tr><td class="muted">inputs</td><td class="mono">{{ inputs.join(", ") }}</td></tr>
              <tr v-if="exposes.length"><td class="muted">exposes</td><td class="mono">{{ exposes.join(", ") }}</td></tr>
              <tr v-if="targets.length"><td class="muted">targets</td><td class="mono">{{ targets.join(", ") }}</td></tr>
              <tr v-if="classes.length"><td class="muted">classes</td><td class="mono" style="white-space: normal">{{ classes.join(", ") }}</td></tr>
            </tbody>
          </table>
        </div>
      </section>
    </div>
    <section class="card" v-if="fieldRows.length">
      <div class="card-head"><span class="card-title">fields in the plan</span><span class="muted small">{{ fieldRows.length }}</span></div>
      <div class="table-wrap">
        <table class="data">
          <thead><tr><th>name</th><th>type</th><th>role</th><th>columns</th><th>preprocessors</th></tr></thead>
          <tbody>
            <tr v-for="row in fieldRows" :key="row.name">
              <td class="mono">{{ row.name }}</td>
              <td class="mono">{{ row.type }}</td>
              <td>{{ row.role }}</td>
              <td class="mono" style="white-space: normal">{{ row.columns.join(", ") }}</td>
              <td class="mono">{{ row.chain }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>
    <template v-if="fitted">
      <section class="card" v-if="features.length">
        <div class="card-head"><span class="card-title">fitted features</span><span class="muted small">{{ featureCount }} columns into the model</span></div>
        <div class="table-wrap">
          <table class="data">
            <thead><tr><th>pattern</th><th>columns</th><th>chain</th></tr></thead>
            <tbody>
              <tr v-for="item in features" :key="item.key">
                <td class="mono">{{ item.pattern }}</td>
                <td style="white-space: normal">
                  <details>
                    <summary style="cursor: pointer">{{ item.columns.length }} columns</summary>
                    <div class="mono">{{ item.columns.join(", ") }}</div>
                    <div class="muted mono" v-if="item.names.length">model names: {{ item.names.join(", ") }}</div>
                  </details>
                </td>
                <td class="mono">{{ item.chain }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>
      <section class="card" v-if="targetFields.length || fittedTargets.length">
        <div class="card-head"><span class="card-title">fitted targets</span></div>
        <div class="table-wrap" v-if="targetFields.length">
          <table class="data">
            <thead><tr><th>name</th><th class="num">width</th><th>chain</th><th>classes</th><th>class weights</th></tr></thead>
            <tbody>
              <tr v-for="item in targetFields" :key="item.key">
                <td class="mono">{{ item.name }}</td>
                <td class="num mono">{{ item.width }}</td>
                <td class="mono">{{ item.chain }}</td>
                <td class="mono" style="white-space: normal">{{ item.classes.join(", ") }}</td>
                <td class="mono" style="white-space: normal">{{ item.weights }}</td>
              </tr>
            </tbody>
          </table>
        </div>
        <div class="mono" v-else>{{ fittedTargets.join(", ") }}</div>
      </section>
      <section class="card" v-if="facts.length || fittedClasses.length">
        <div class="card-head"><span class="card-title">fitted settings</span></div>
        <div class="table-wrap">
          <table class="data">
            <tbody>
              <tr v-if="fittedClasses.length"><td class="muted">classes</td><td class="mono" style="white-space: normal">{{ fittedClasses.join(", ") }}</td></tr>
              <tr v-for="row in facts" :key="row.name"><td class="muted">{{ row.name }}</td><td class="mono" style="white-space: normal">{{ row.value }}</td></tr>
            </tbody>
          </table>
        </div>
      </section>
    </template>
    <div class="empty" v-else>no fitted/fields.json in this record</div>
  </div>`,
};
