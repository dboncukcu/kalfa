import { Diagram, shapeText } from "../../components/diagram.js";
import { integer, percent } from "../../format.js";


const COLUMNS = "minmax(120px, 1.3fr) minmax(100px, 1fr) 96px minmax(80px, 160px) 44px";
const ROW = "display: inline-grid; grid-template-columns: " + COLUMNS + "; gap: 10px; align-items: center; width: calc(100% - 20px)";
const LEAF = "display: grid; grid-template-columns: " + COLUMNS + "; gap: 10px; align-items: center; padding: 2px 0 2px 20px";

function shapeOf(value) {
  if (Array.isArray(value)) return value.join(" × ");
  if (value && typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function shown(value) {
  if (value === null || value === undefined) return "";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

const ModuleNode = {
  name: "ModuleNode",
  props: {
    node: { type: Object, required: true },
    total: { type: Number, default: 0 },
    depth: { type: Number, default: 0 },
  },
  computed: {
    row() { return ROW; },
    leaf() { return LEAF; },
    children() { return Array.isArray(this.node.children) ? this.node.children : []; },
    type() { return this.node.class || ""; },
    parameters() { return integer(Number(this.node.parameters) || 0); },
    share() {
      const own = Number(this.node.parameters) || 0;
      return this.total > 0 ? Math.max(0, Math.min(1, own / this.total)) : 0;
    },
    shareText() { return percent(this.share); },
    fill() { return { width: (this.share * 100).toFixed(1) + "%" }; },
    opened() { return this.depth === 0; },
  },
  template: `
    <details v-if="children.length" class="module" :open="opened">
      <summary style="cursor: pointer">
        <span :style="row">
          <span class="mono">{{ node.name }}</span>
          <span class="muted small">{{ type }}</span>
          <span class="num mono">{{ parameters }}</span>
          <span class="progress-track"><span class="progress-fill" :style="fill"></span></span>
          <span class="num muted small">{{ shareText }}</span>
        </span>
      </summary>
      <div class="module-children" style="padding-left: 18px">
        <module-node v-for="(child, index) in children" :key="index" :node="child" :total="total" :depth="depth + 1"></module-node>
      </div>
    </details>
    <div v-else class="module" :style="leaf">
      <span class="mono">{{ node.name }}</span>
      <span class="muted small">{{ type }}</span>
      <span class="num mono">{{ parameters }}</span>
      <span class="progress-track"><span class="progress-fill" :style="fill"></span></span>
      <span class="num muted small">{{ shareText }}</span>
    </div>`,
};

ModuleNode.components = { "module-node": ModuleNode };

const CROWDED = 60;

function graphOf(model) {
  const graph = model.graph;
  return graph && typeof graph === "object" && Array.isArray(graph.nodes) && graph.nodes.length ? graph : null;
}

function missingText(model) {
  if (!Object.prototype.hasOwnProperty.call(model, "graph")) {
    return "This record was written before kalfa drew model diagrams, so the module tree is shown instead.";
  }
  const note = model.graph_note ? " (" + model.graph_note + ")" : "";
  return "kalfa drew no diagram for this model" + note + ". It traces a model on the first batch with the fields the " +
    "model reads, and when a loss calls the model with inputs of its own, such as a diffusion denoiser or a generator, " +
    "on the inputs of its first call in training, which exist once the first turn is over. The module tree is shown " +
    "instead.";
}

function topGroups(graph) {
  const groups = Array.isArray(graph.groups) ? graph.groups : [];
  return groups.filter(group => group && (group.parent === null || group.parent === undefined)).map(group => String(group.id));
}

function outputRows(outputs) {
  return Object.entries(outputs || {}).map(([label, shape]) => ({ label, shape: shapeOf(shape) }));
}

function shapeRows(outputs, perSample) {
  const sampled = new Map(perSample.map(item => [item.label, item.shape]));
  const rows = outputs.map(item => ({ key: item.label, label: item.label, batch: item.shape, sample: sampled.get(item.label) || "" }));
  const named = new Set(outputs.map(item => item.label));
  for (const item of perSample) {
    if (!named.has(item.label)) rows.push({ key: "sample:" + item.label, label: item.label, batch: "", sample: item.shape });
  }
  return rows;
}

export default {
  name: "ModelTab",
  components: { "module-node": ModuleNode, "k-diagram": Diagram },
  props: {
    record: { type: Object, default: null },
    path: { type: String, default: "" },
    api: { default: null },
    version: { type: Number, default: 0 },
  },
  data() {
    return { views: {}, picked: {} };
  },
  computed: {
    nodes() {
      const plan = (this.record && this.record.plan) || {};
      return plan.nodes && typeof plan.nodes === "object" ? plan.nodes : {};
    },
    models() {
      const summary = (this.record && this.record.model_summary) || {};
      return Object.entries(summary).map(([name, entry]) => {
        const model = entry && typeof entry === "object" ? entry : {};
        const node = this.nodes["models." + name] || {};
        const params = node.params && typeof node.params === "object" ? node.params : {};
        const parameters = Number(model.parameters) || 0;
        const trainable = Number(model.trainable) || 0;
        const graph = graphOf(model);
        const kinds = graph ? graph.nodes.map(item => item && item.kind) : [];
        const outputs = outputRows(model.outputs);
        const perSample = graph ? graph.nodes.filter(item => item && item.kind === "output")
          .map(item => ({ label: String(item.label), shape: shapeText(item.shape) || "?" })) : [];
        return {
          name,
          type: model.class || node.use || "",
          parameters,
          total: integer(parameters),
          trainable: integer(trainable),
          frozen: model.trainable !== undefined && model.trainable !== null && trainable !== parameters,
          tree: Array.isArray(model.tree) ? model.tree : [],
          shapes: shapeRows(outputs, perSample),
          params: Object.entries(params).map(([key, value]) => ({ key, value: shown(value) })),
          graph,
          drawnFromCall: Boolean(graph) && model.graph_from === "first training call",
          missing: graph ? "" : missingText(model),
          collapsed: graph && graph.nodes.length > CROWDED ? topGroups(graph) : [],
          modules: kinds.filter(kind => kind === "module").length,
          ops: kinds.filter(kind => kind === "op").length,
          truncated: Boolean(graph && graph.truncated),
        };
      });
    },
    details() {
      return Object.fromEntries(this.models.map(model => [model.name, this.detail(model)]));
    },
  },
  methods: {
    viewOf(model) {
      const chosen = this.views[model.name];
      if (chosen === "diagram" && !model.graph) return "tree";
      return chosen || (model.graph ? "diagram" : "tree");
    },
    show(model, view) {
      this.views = { ...this.views, [model.name]: view };
    },
    select(model, node) {
      this.picked = { ...this.picked, [model.name]: node ? String(node.id) : null };
    },
    detail(model) {
      const id = this.picked[model.name];
      if (!model.graph || id === null || id === undefined) return null;
      const byId = new Map(model.graph.nodes.filter(Boolean).map(item => [String(item.id), item]));
      const node = byId.get(id);
      if (!node) return null;
      const groups = new Map((model.graph.groups || []).filter(Boolean).map(group => [String(group.id), group]));
      const name = (other) => {
        const found = byId.get(String(other));
        return found ? String(found.label) + (found.path ? " " + found.path : "") : String(other);
      };
      const edges = Array.isArray(model.graph.edges) ? model.graph.edges.filter(Boolean) : [];
      const group = node.group === null || node.group === undefined ? null : groups.get(String(node.group));
      const rows = [
        ["path", node.path || ""],
        ["class", node.class || ""],
        ["detail", node.detail && node.detail !== node.path ? node.detail : ""],
        ["shape", shapeText(node.shape)],
        ["parameters", typeof node.parameters === "number" ? integer(node.parameters) : ""],
        ["inside", group ? [group.label, group.detail].filter(Boolean).join(", ") : ""],
      ].filter(row => row[1] !== "" && row[1] !== null && row[1] !== undefined);
      return {
        label: String(node.label),
        kind: String(node.kind || ""),
        rows,
        inputs: edges.filter(edge => String(edge.to) === id).map(edge => ({ name: name(edge.from), shape: shapeText(edge.shape) })),
        outputs: edges.filter(edge => String(edge.from) === id).map(edge => ({ name: name(edge.to), shape: shapeText(edge.shape) })),
      };
    },
  },
  template: `
  <div class="page pane-model">
    <div class="muted small">Every model of the run from model_summary.json, written when the run starts and again when it
      ends: its parameter counts, a diagram of one forward pass on the first batch, its modules as a tree and the shapes of
      its outputs. In the diagram, shapes are per sample (the batch size left out) and the corner number is the parameter
      count; in the tree, the bar is the share of the model's parameters a module holds.</div>
    <div class="empty" v-if="!models.length">no model_summary.json yet; the run writes it before its first turn</div>
    <section class="card" v-for="model in models" :key="model.name">
      <div class="card-head">
        <span class="card-title">{{ model.name }}</span>
        <span class="muted mono">{{ model.type }}</span>
      </div>
      <div class="toolbar">
        <span class="chip">{{ model.total }} parameters</span>
        <span class="chip">{{ model.trainable }} trainable<span v-if="model.frozen">, the rest frozen</span></span>
        <span class="chip" v-if="model.graph">{{ model.modules }} module calls, {{ model.ops }} operations</span>
        <span class="chip" v-if="model.drawnFromCall"
          title="the batch does not hold this model's inputs, so the diagram follows its first call in training">drawn from its first call in training</span>
      </div>
      <div class="muted small mono" v-if="model.params.length">
        <span class="chip" v-for="item in model.params" :key="item.key">{{ item.key }} {{ item.value }}</span>
      </div>
      <div class="model-views">
        <button type="button" class="tab" :class="{ active: viewOf(model) === 'diagram' }" :disabled="!model.graph"
          :title="model.graph ? 'one forward pass on the first batch' : 'no diagram for this model'" @click="show(model, 'diagram')">diagram</button>
        <button type="button" class="tab" :class="{ active: viewOf(model) === 'tree' }" @click="show(model, 'tree')">module tree</button>
        <button type="button" class="tab" :class="{ active: viewOf(model) === 'outputs' }" @click="show(model, 'outputs')">output shapes</button>
      </div>
      <template v-if="viewOf(model) === 'diagram'">
        <div class="notice-box model-note" v-if="model.truncated">The diagram stops after the first {{ model.modules + model.ops }}
          module calls and operations; the module tree shows the whole model.</div>
        <div class="muted small model-note" v-if="model.collapsed.length">This model has many nodes, so its top groups start
          folded; click a folded group to open it, or use expand all.</div>
        <div class="model-diagram">
          <k-diagram :graph="model.graph" :collapsed="model.collapsed" :height="540" :title="model.name + '_diagram'"
                     @select="select(model, $event)"></k-diagram>
          <div class="model-detail">
            <template v-if="details[model.name]">
              <div class="model-detail-title">{{ details[model.name].label }} <span class="chip">{{ details[model.name].kind }}</span></div>
              <dl>
                <template v-for="row in details[model.name].rows" :key="row[0]">
                  <dt>{{ row[0] }}</dt>
                  <dd :class="{ mono: row[0] !== 'inside' }">{{ row[1] }}</dd>
                </template>
              </dl>
              <div v-if="details[model.name].inputs.length">
                <div class="muted small">takes</div>
                <ul><li v-for="(item, index) in details[model.name].inputs" :key="index">{{ item.name }} <span class="mono muted">{{ item.shape }}</span></li></ul>
              </div>
              <div v-if="details[model.name].outputs.length">
                <div class="muted small">feeds</div>
                <ul><li v-for="(item, index) in details[model.name].outputs" :key="index">{{ item.name }} <span class="mono muted">{{ item.shape }}</span></li></ul>
              </div>
            </template>
            <div class="muted small" v-else>Click a node to see its path, class, shape, parameters and the nodes it takes from
              and feeds.</div>
          </div>
        </div>
      </template>
      <template v-else-if="viewOf(model) === 'tree'">
        <div class="notice-box model-note" v-if="model.missing">{{ model.missing }}</div>
        <div class="module-tree" v-if="model.tree.length" style="margin-top: 8px">
          <module-node v-for="(node, index) in model.tree" :key="index" :node="node" :total="model.parameters" :depth="0"></module-node>
        </div>
        <div class="empty" v-else>no submodules recorded</div>
      </template>
      <template v-else>
        <div class="muted small model-note">The shapes the model's outputs had on the first batch, the batch size first; the
          per sample column comes from the diagram.</div>
        <div class="table-wrap" v-if="model.shapes.length">
          <table class="data">
            <thead><tr><th>output</th><th>shape on the first batch</th><th>per sample</th></tr></thead>
            <tbody>
              <tr v-for="item in model.shapes" :key="item.key">
                <td class="mono">{{ item.label }}</td>
                <td class="mono">{{ item.batch }}</td>
                <td class="mono">{{ item.sample }}</td>
              </tr>
            </tbody>
          </table>
        </div>
        <div class="empty" v-else>no output shapes recorded: the model could not run on the first batch with the fields it reads</div>
      </template>
    </section>
  </div>`,
};
