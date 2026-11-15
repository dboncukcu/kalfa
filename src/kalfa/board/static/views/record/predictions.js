import Chart from "../../components/chart.js";
import { messageOf } from "../../components/error.js";
import Table from "../../components/table.js";
import { integer, number, percent } from "../../format.js";


const DEFAULT = "predictions.parquet";
const PREFIX = "kalfa.predictions.";
const DELAY = 350;
const GROUPS = 20;
const CHOICES = ["task", "score", "positive", "threshold"];
const HISTOGRAMS = {
  residuals: "residual_hist",
  histograms: "truth_hist",
  scores: "score_hist",
  norms: "norm_hist",
  values: "hist",
};
const TASK_HISTOGRAMS = {
  regression: ["residuals", "histograms"],
  binary: ["scores"],
  embedding: ["norms"],
  values: ["values"],
};
const WORST_COLUMNS = [
  { key: "row", label: "row", numeric: true },
  { key: "truth", label: "truth", numeric: true },
  { key: "pred", label: "prediction", numeric: true },
  { key: "error", label: "error", numeric: true },
];
const CLASS_COLUMNS = [
  { key: "label", label: "class", className: "mono" },
  { key: "precision", label: "precision", numeric: true, text: row => percent(row.precision, 1) },
  { key: "recall", label: "recall", numeric: true, text: row => percent(row.recall, 1) },
  { key: "f1", label: "f1", numeric: true, text: row => number(row.f1) },
  { key: "support", label: "support", numeric: true, text: row => integer(row.support) },
];
const REASONS = {
  "binary,multiclass": "the target has two classes",
  "binary,regression": "the numeric target holds two values and the larger one is the positive class",
  "multiclass,regression": "the target holds 3 to 20 whole numbers and every prediction is one of them",
  multiclass: "the target holds class labels",
  regression: "the target is numeric",
  embedding: "spread pred_ columns with no target beside them",
  values: "one pred_ column with no target beside it",
};
const NOTES = {
  binary: "two classes: the confusion matrix at a threshold, and ROC, precision recall and a threshold sweep when a score column exists",
  multiclass: "several classes: the confusion matrix, precision, recall and f1 per class, and ROC when every class has a prob_ column",
  regression: "a numeric target: rmse, mae, r2, truth against prediction and the residuals",
  embedding: "a vector per row: a PCA map of the rows and the length of each vector",
  values: "one value per row: its distribution",
};
const RULES = {
  "0.5": "0.5, the score is a probability",
  best_f1: "the best f1 on this table",
  given: "set here",
};

function tablesOf(record) {
  return record && Array.isArray(record.tables) ? record.tables : [];
}

function chosenTable(tables, current) {
  if (!tables.length) return DEFAULT;
  if (tables.includes(current)) return current;
  return tables.includes(DEFAULT) ? DEFAULT : tables[0];
}

function readStored(key) {
  try {
    const text = window.localStorage.getItem(PREFIX + key);
    const value = text ? JSON.parse(text) : null;
    return value && typeof value === "object" && !Array.isArray(value) ? value : {};
  } catch (error) {
    return {};
  }
}

function writeStored(key, value) {
  try {
    if (value && Object.keys(value).length) window.localStorage.setItem(PREFIX + key, JSON.stringify(value));
    else window.localStorage.removeItem(PREFIX + key);
  } catch (error) {
    return;
  }
}

function cleaned(choice) {
  const found = {};
  for (const key of CHOICES) {
    const value = choice ? choice[key] : null;
    if (value !== null && value !== undefined && value !== "") found[key] = value;
  }
  return found;
}

function stairs(edges, counts) {
  const x = [];
  const y = [];
  (counts || []).forEach((count, index) => {
    x.push(edges[index], edges[index + 1]);
    y.push(count, count);
  });
  return { x, y };
}

function peakOf(lists) {
  let peak = 0;
  for (const list of lists) {
    for (const value of list || []) {
      if (typeof value === "number" && value > peak) peak = value;
    }
  }
  return peak || 1;
}

function shortened(matrix) {
  return (matrix || []).map((row) => row.map((value) => (typeof value === "number" ? Number(value.toFixed(3)) : value)));
}

function labelOf(value) {
  return value === null || value === undefined ? "missing" : String(value);
}

function rowText(row, index) {
  return "row " + labelOf(row) + ", index " + index;
}

function pcaGroups(pca) {
  const xs = pca.x || [];
  const ys = pca.y || [];
  const rows = pca.row || [];
  const indexes = pca.index || [];
  const colors = Array.isArray(pca.color) ? pca.color.map(labelOf) : null;
  const names = colors ? [...new Set(colors)] : [];
  if (!colors || names.length > GROUPS) {
    return [{ name: "rows", x: xs, y: ys, text: xs.map((value, index) => rowText(rows[index], indexes[index])) }];
  }
  return names.map((name) => {
    const group = { name, x: [], y: [], text: [] };
    colors.forEach((color, index) => {
      if (color !== name) return;
      group.x.push(xs[index]);
      group.y.push(ys[index]);
      group.text.push(name + ", " + rowText(rows[index], indexes[index]));
    });
    return group;
  });
}

export default {
  name: "PredictionsTab",
  components: { "k-chart": Chart, "k-table": Table },
  props: {
    record: { type: Object, default: null },
    path: { type: String, default: "" },
    api: { default: null },
    version: { type: Number, default: 0 },
  },
  data() {
    return {
      stats: null,
      error: "",
      loading: false,
      where: "",
      applied: "",
      name: chosenTable(tablesOf(this.record), DEFAULT),
      selected: "",
      choice: {},
      slider: null,
      binsByChart: {},
      matrix: "counts",
      metric: "f1",
    };
  },
  computed: {
    worstColumns() { return WORST_COLUMNS; },
    classColumns() { return CLASS_COLUMNS; },
    tables() { return tablesOf(this.record); },
    manyTables() { return this.tables.length > 1; },
    available() { return Boolean(this.record && (this.record.has_predictions || this.tables.length)); },
    tablesKey() { return (this.available ? "1:" : "0:") + this.tables.join(","); },
    pairs() { return this.stats && Array.isArray(this.stats.pairs) ? this.stats.pairs : []; },
    manyPairs() { return this.pairs.length > 1; },
    pair() { return this.pairs.find((item) => item.pred === this.selected) || this.pairs[0] || null; },
    numbers() { return (this.pair && this.pair.stats) || {}; },
    task() { return this.pair ? this.pair.task : ""; },
    isRegression() { return this.task === "regression"; },
    isBinary() { return this.task === "binary"; },
    isMulticlass() { return this.task === "multiclass"; },
    isEmbedding() { return this.task === "embedding"; },
    isValues() { return this.task === "values"; },
    chosen() { return Object.keys(this.choice).length > 0; },
    noPairs() { return Boolean(this.stats) && !this.pairs.length; },
    rowsText() {
      if (!this.stats) return "";
      return integer(this.stats.rows) + " rows" + (this.applied ? " where " + this.applied : "");
    },
    overview() { return (this.stats && this.stats.overview) || { pairs: 0, tasks: {}, rows: [] }; },
    overviewRows() { return Array.isArray(this.overview.rows) ? this.overview.rows : []; },
    overviewNote() {
      const parts = Object.entries(this.overview.tasks || {}).map(([task, count]) => count + " " + task);
      return this.overviewRows.length + " pairs: " + parts.join(", ") + "; click a row to open it";
    },
    pairNote() {
      const item = this.pair;
      if (!item) return "";
      if (item.target) return item.pred + " against " + item.target;
      const last = item.columns[item.columns.length - 1];
      return item.columns.length > 1 ? item.columns.length + " columns from " + item.columns[0] + " to " + last : item.pred;
    },
    reason() {
      const item = this.pair;
      if (!item) return "";
      const why = REASONS[item.tasks.join(",")] || "";
      const head = "auto reads it as " + item.auto + (why ? " since " + why : "");
      return item.task === item.auto ? head : head + "; you chose " + item.task;
    },
    taskNote() { return NOTES[this.task] || ""; },
    taskOptions() {
      const item = this.pair;
      if (!item) return [];
      return [{ value: "", label: "auto, " + item.auto }, ...item.tasks.map((task) => ({ value: task, label: task }))];
    },
    taskChoice() { return this.choice.task || ""; },
    scoreOptions() { return Array.isArray(this.numbers.candidates) ? this.numbers.candidates : []; },
    positiveOptions() { return Array.isArray(this.numbers.labels) ? this.numbers.labels : []; },
    range() {
      const sweep = this.numbers.sweep;
      const steps = sweep && Array.isArray(sweep.threshold) ? sweep.threshold : [];
      const low = steps.length ? steps[0] : 0;
      const high = steps.length ? steps[steps.length - 1] : 1;
      const top = high > low ? high : low + 1;
      return { min: low, max: top, step: (top - low) / 200 };
    },
    sliderText() { return number(this.slider); },
    ruleText() { return RULES[this.numbers.threshold_rule] || ""; },
    metrics() {
      const s = this.numbers;
      if (this.isRegression) {
        return [["rows used", integer(s.count)], ["rmse", number(s.rmse)], ["mae", number(s.mae)], ["r2", number(s.r2)],
                ["bias", number(s.bias)]];
      }
      if (this.isBinary) {
        const found = [["rows used", integer(s.count)], ["accuracy", percent(s.accuracy, 1)],
                       ["precision", percent(s.precision, 1)], ["recall", percent(s.recall, 1)], ["f1", number(s.f1)],
                       ["specificity", percent(s.specificity, 1)], ["balanced accuracy", percent(s.balanced_accuracy, 1)]];
        if (s.roc) found.push(["roc auc", number(s.roc.auc)]);
        if (s.pr) found.push(["average precision", number(s.pr.ap)]);
        return found;
      }
      if (this.isMulticlass) {
        return [["rows used", integer(s.count)], ["classes", integer((s.classes || []).length)],
                ["accuracy", percent(s.accuracy, 1)], ["macro f1", number(s.macro_f1)], ["weighted f1", number(s.weighted_f1)]];
      }
      if (this.isEmbedding) {
        const explained = (s.pca && s.pca.explained) || [];
        return [["rows used", integer(s.count)], ["dimensions", integer(s.dims)],
                ["first component", percent(explained[0], 1)], ["second component", percent(explained[1], 1)]];
      }
      if (this.isValues) {
        if (s.labels) return [["rows used", integer(s.count)], ["labels", integer((s.labels.labels || []).length)]];
        return [["rows used", integer(s.count)], ["mean", number(s.mean)], ["std", number(s.std)]];
      }
      return [];
    },
    quantiles() {
      return Object.entries(this.numbers.residual_quantiles || {}).map(([name, value]) => ({ name, value: number(value) }));
    },
    worst() { return Array.isArray(this.numbers.worst) ? this.numbers.worst : []; },
    scatterData() {
      const scatter = this.isRegression ? this.numbers.scatter : null;
      if (!scatter) return null;
      return [{ name: "rows", x: scatter.truth, y: scatter.pred, text: (scatter.row || []).map((row) => "row " + row) }];
    },
    scatterOptions() {
      const item = this.pair || {};
      return { diagonal: true, xlabel: "truth, " + item.target, ylabel: "prediction, " + item.pred, legend: false };
    },
    residualData() {
      const found = this.isRegression ? this.numbers.residual_hist : null;
      return found ? { edges: found.edges || [], counts: found.counts || [] } : null;
    },
    residualOptions() { return { xlabel: "prediction minus truth", ylabel: "rows" }; },
    histogramData() {
      const s = this.numbers;
      if (!this.isRegression || !s.truth_hist || !s.pred_hist) return null;
      return [
        { name: "truth", edges: s.truth_hist.edges || [], counts: s.truth_hist.counts || [] },
        { name: "prediction", edges: s.pred_hist.edges || [], counts: s.pred_hist.counts || [] },
      ];
    },
    histogramOptions() { return { xlabel: this.pair ? this.pair.target : "", ylabel: "rows" }; },
    matrixLabels() {
      const s = this.numbers;
      if (this.isBinary) return Array.isArray(s.labels) ? s.labels : [];
      return Array.isArray(s.classes) ? s.classes : [];
    },
    confusionData() {
      const s = this.numbers;
      if (!(this.isBinary || this.isMulticlass) || !Array.isArray(s.confusion)) return null;
      const z = this.matrix === "rates" ? shortened(s.confusion_normalized) : s.confusion;
      return { z, x: this.matrixLabels, y: this.matrixLabels };
    },
    confusionTitle() {
      const head = this.matrix === "rates" ? "confusion, share of each true class" : "confusion, rows";
      return this.isBinary && this.numbers.score ? head + " at threshold " + number(this.numbers.threshold) : head;
    },
    confusionOptions() {
      const item = this.pair || {};
      return { xlabel: "predicted, " + item.pred, ylabel: "truth, " + item.target };
    },
    rocData() {
      const roc = this.isBinary ? this.numbers.roc : null;
      if (!roc) return null;
      return [{ name: "roc", x: roc.fpr, y: roc.tpr, mode: "lines" },
              { name: "chance", x: [0, 1], y: [0, 1], dash: "dot", mode: "lines" }];
    },
    rocTitle() { return this.numbers.roc && this.isBinary ? "ROC, auc " + number(this.numbers.roc.auc) : ""; },
    rocOptions() { return { xlabel: "false positive rate", ylabel: "true positive rate" }; },
    prData() {
      const pr = this.isBinary ? this.numbers.pr : null;
      return pr ? [{ name: "precision recall", x: pr.recall, y: pr.precision, mode: "lines" }] : null;
    },
    prTitle() { return this.numbers.pr ? "precision recall, average precision " + number(this.numbers.pr.ap) : ""; },
    prOptions() { return { xlabel: "recall", ylabel: "precision", legend: false }; },
    scoreData() {
      const found = this.isBinary ? this.numbers.score_hist : null;
      if (!found) return null;
      const labels = this.positiveOptions;
      const negative = stairs(found.edges || [], found.negative);
      const positive = stairs(found.edges || [], found.positive);
      const peak = peakOf([found.negative, found.positive]);
      const at = this.numbers.threshold;
      return [
        { name: "truth " + labelOf(labels[0]), x: negative.x, y: negative.y, mode: "lines" },
        { name: "truth " + labelOf(labels[1]), x: positive.x, y: positive.y, mode: "lines" },
        { name: "threshold", x: [at, at], y: [0, peak], mode: "lines", dash: "dash" },
      ];
    },
    scoreChartOptions() { return { xlabel: "score, " + (this.numbers.score || ""), ylabel: "rows" }; },
    sweepData() {
      const sweep = this.isBinary ? this.numbers.sweep : null;
      if (!sweep) return null;
      const at = this.numbers.threshold;
      return [
        { name: "precision", x: sweep.threshold, y: sweep.precision, mode: "lines" },
        { name: "recall", x: sweep.threshold, y: sweep.recall, mode: "lines" },
        { name: "f1", x: sweep.threshold, y: sweep.f1, mode: "lines" },
        { name: "threshold", x: [at, at], y: [0, 1], mode: "lines", dash: "dash" },
      ];
    },
    sweepOptions() { return { xlabel: "threshold", ylabel: "rate" }; },
    perClass() {
      return Object.entries(this.numbers.per_class || {}).map(([label, item]) => ({ label, ...(item || {}) }));
    },
    classBars() {
      if (!this.isMulticlass || !this.perClass.length) return null;
      return { labels: this.perClass.map((row) => row.label), values: this.perClass.map((row) => row[this.metric]) };
    },
    classBarsOptions() { return { horizontal: true, xlabel: this.metric }; },
    classBarsHeight() { return Math.max(220, Math.min(640, 60 + 26 * this.perClass.length)); },
    multiRocData() {
      const roc = this.isMulticlass ? this.numbers.roc : null;
      if (!roc || typeof roc !== "object") return null;
      const lines = Object.entries(roc).map(([label, curve]) => ({
        name: label + ", auc " + number(curve.auc), x: curve.fpr, y: curve.tpr, mode: "lines",
      }));
      lines.push({ name: "chance", x: [0, 1], y: [0, 1], dash: "dot", mode: "lines" });
      return lines;
    },
    multiRocOptions() { return { xlabel: "false positive rate", ylabel: "true positive rate" }; },
    pcaData() {
      const pca = this.isEmbedding ? this.numbers.pca : null;
      return pca && Array.isArray(pca.x) && pca.x.length ? pcaGroups(pca) : null;
    },
    pcaTitle() {
      const pca = this.numbers.pca || {};
      const shown = (pca.x || []).length;
      const base = "PCA of " + (this.pair ? this.pair.pred : "") + ", " + integer(shown) + " rows";
      return pca.color_by ? base + " colored by " + pca.color_by : base;
    },
    pcaOptions() {
      const explained = (this.numbers.pca && this.numbers.pca.explained) || [];
      return { xlabel: "first component, " + percent(explained[0], 1), ylabel: "second component, " + percent(explained[1], 1) };
    },
    normData() {
      const found = this.isEmbedding ? this.numbers.norm_hist : null;
      return found ? { edges: found.edges || [], counts: found.counts || [] } : null;
    },
    normOptions() { return { xlabel: "vector length", ylabel: "rows" }; },
    valuesData() {
      const found = this.isValues ? this.numbers.hist : null;
      return found ? { edges: found.edges || [], counts: found.counts || [] } : null;
    },
    valuesOptions() { return { xlabel: this.pair ? this.pair.pred : "", ylabel: "rows" }; },
    labelBars() {
      const found = this.isValues ? this.numbers.labels : null;
      return found ? { labels: found.labels || [], values: found.counts || [] } : null;
    },
    labelBarsOptions() { return { horizontal: true, xlabel: "rows" }; },
    labelBarsHeight() {
      const found = this.labelBars ? this.labelBars.labels.length : 0;
      return Math.max(220, Math.min(640, 60 + 24 * found));
    },
    valueQuantiles() {
      return Object.entries(this.numbers.quantiles || {}).map(([name, value]) => ({ name, value: number(value) }));
    },
    gallery() {
      return this.pair && this.pair.images && Array.isArray(this.numbers.gallery) ? this.numbers.gallery : [];
    },
    galleryNote() {
      if (this.isBinary || this.isMulticlass) return "wrong predictions first";
      if (this.isRegression) return "largest errors first";
      return "the first rows";
    },
    histograms() {
      const item = this.pair;
      if (!item || item.pred !== this.selected) return {};
      const found = {};
      for (const chart of TASK_HISTOGRAMS[this.choice.task || item.auto] || []) {
        const count = this.binsByChart[this.chartKey(chart)];
        if (count) found[HISTOGRAMS[chart]] = count;
      }
      return found;
    },
    histogramsText() {
      return Object.entries(this.histograms).map(([name, count]) => name + ":" + count).join(",");
    },
    histogramsWanted() { return this.histogramsText ? this.selected + "|" + this.histogramsText : ""; },
  },
  watch: {
    tablesKey() {
      const wanted = chosenTable(this.tables, this.name);
      if (wanted !== this.name) {
        this.name = wanted;
        return;
      }
      this.load();
    },
    name() {
      this.restore();
      this.load();
    },
    path() {
      this.restore();
      this.load();
    },
    histogramsWanted() {
      this.syncBins();
    },
  },
  created() {
    this.timer = null;
    this.ticket = 0;
    this.sent = { pred: "", text: "" };
    this.binsQueued = false;
  },
  mounted() {
    this.restore();
    this.load();
  },
  beforeUnmount() {
    if (this.timer) clearTimeout(this.timer);
  },
  methods: {
    number,
    percent,
    integer,
    tableKey() { return this.path + "|" + this.name; },
    pairKey(pred) { return this.tableKey() + "|" + pred; },
    chartKey(chart) { return "predictions|" + this.path + "|" + (this.pair ? this.pair.pred : "") + "|" + chart; },
    restore() {
      this.selected = String(readStored(this.tableKey()).pred || "");
      this.choice = cleaned(readStored(this.pairKey(this.selected)));
    },
    pick(pred) {
      if (this.pair && pred === this.pair.pred) return;
      const before = this.chosen;
      this.selected = pred;
      writeStored(this.tableKey(), { pred });
      this.choice = cleaned(readStored(this.pairKey(pred)));
      this.syncSlider();
      if (before || this.chosen) this.load();
    },
    remember(patch) {
      if (!this.pair) return;
      if (this.timer) clearTimeout(this.timer);
      this.timer = null;
      this.selected = this.pair.pred;
      this.choice = cleaned({ ...this.choice, ...patch });
      writeStored(this.tableKey(), { pred: this.selected });
      writeStored(this.pairKey(this.selected), this.choice);
      this.load();
    },
    setTask(value) { this.remember({ task: value, threshold: null }); },
    setScore(value) { this.remember({ score: value, threshold: null }); },
    setPositive(value) { this.remember({ positive: value, score: null, threshold: null }); },
    slide(value) {
      const found = Number(value);
      if (!this.pair || !Number.isFinite(found)) return;
      const key = this.pairKey(this.pair.pred);
      this.slider = found;
      if (this.timer) clearTimeout(this.timer);
      this.timer = setTimeout(() => {
        this.timer = null;
        if (this.pair && this.pairKey(this.pair.pred) === key) this.remember({ threshold: found });
      }, DELAY);
    },
    resetThreshold() { this.remember({ threshold: null }); },
    resetChoice() {
      if (!this.pair) return;
      if (this.timer) clearTimeout(this.timer);
      this.timer = null;
      this.choice = {};
      writeStored(this.pairKey(this.pair.pred), {});
      this.load();
    },
    setMatrix(mode) { this.matrix = mode; },
    setMetric(metric) { this.metric = metric; },
    syncSlider() {
      const found = this.pair && this.pair.task === "binary" ? this.pair.stats.threshold : null;
      this.slider = typeof found === "number" ? found : null;
    },
    onSettings(payload) {
      if (!payload || typeof payload.key !== "string") return;
      const wanted = Number.isInteger(payload.bins) && payload.bins > 0 ? payload.bins : null;
      const known = this.binsByChart[payload.key] || null;
      if (wanted === known) return;
      if (wanted === null) delete this.binsByChart[payload.key];
      else this.binsByChart[payload.key] = wanted;
    },
    countedText() {
      return this.pair && this.sent.pred === this.pair.pred ? this.sent.text : "";
    },
    syncBins() {
      if (this.binsQueued) return;
      this.binsQueued = true;
      this.$nextTick(() => {
        this.binsQueued = false;
        if (this.histogramsText !== this.countedText()) this.load();
      });
    },
    apply() {
      this.applied = this.where.trim();
      this.load();
    },
    clear() {
      this.where = "";
      this.applied = "";
      this.load();
    },
    pairLabel(item) {
      if (item.target) return item.pred + " against " + item.target;
      return item.columns.length > 1 ? item.pred + ", " + item.columns.length + " columns" : item.pred;
    },
    valueText(row) {
      if (row.value === null || row.value === undefined) return "none";
      if (row.key === "count") return integer(row.value);
      if (row.key === "accuracy" || row.key === "explained") return percent(row.value, 1);
      return number(row.value);
    },
    imageUrl(entry) {
      return this.api ? this.api.imageUrl(this.path, entry.index, this.name) : "";
    },
    entryWrong(entry) {
      return (this.isBinary || this.isMulticlass) && entry.truth !== entry.pred;
    },
    entryText(entry) {
      const parts = [];
      if (entry.truth !== null && entry.truth !== undefined) parts.push("truth " + labelOf(typeof entry.truth === "number" ? number(entry.truth) : entry.truth));
      if (entry.pred !== null && entry.pred !== undefined) parts.push("pred " + labelOf(typeof entry.pred === "number" ? number(entry.pred) : entry.pred));
      return parts.join(", ") || "row " + labelOf(entry.row);
    },
    entryDetail(entry) {
      if (entry.error !== null && entry.error !== undefined) return "error " + number(entry.error);
      if (entry.score !== null && entry.score !== undefined) return "score " + number(entry.score);
      return "row " + labelOf(entry.row);
    },
    params() {
      const params = { where: this.applied || null, name: this.name || null };
      if (this.chosen && this.selected) {
        params.pair = this.selected;
        for (const key of CHOICES) {
          if (this.choice[key] !== undefined) params[key] = this.choice[key];
        }
      }
      if (this.histogramsText) {
        params.pair = this.selected;
        params.histograms = { ...this.histograms };
      }
      return params;
    },
    async load() {
      if (!this.api) {
        this.error = "the api prop is missing";
        return;
      }
      if (!this.available) {
        this.stats = null;
        return;
      }
      const path = this.path;
      const ticket = ++this.ticket;
      const chosen = this.chosen;
      const params = this.params();
      this.sent = { pred: params.pair || "", text: this.histogramsText };
      this.loading = true;
      try {
        const found = await this.api.getPredictions(path, params);
        if (path !== this.path || ticket !== this.ticket) return;
        this.stats = found ? Object.freeze(found) : null;
        this.error = "";
        if (this.pairs.length && !this.pairs.some((item) => item.pred === this.selected)) {
          this.selected = this.pairs[0].pred;
          this.choice = cleaned(readStored(this.pairKey(this.selected)));
          if (this.chosen) {
            this.load();
            return;
          }
        }
        this.syncSlider();
      } catch (error) {
        if (path !== this.path || ticket !== this.ticket) return;
        if (chosen && params.pair && error.status === 400 && !messageOf(error).startsWith("bad where")) {
          this.choice = {};
          writeStored(this.pairKey(params.pair), {});
          this.load();
          return;
        }
        this.stats = null;
        this.error = messageOf(error);
      } finally {
        if (ticket === this.ticket) this.loading = false;
      }
    },
  },
  template: `
  <div class="page pane-predictions">
    <div class="muted small">The prediction table of the record, scored on the server. Each pred_ column pairs with its target
      and gets a task: binary, multiclass, regression, embedding for spread columns without a target, or values for a single
      column without one. The task selector reads a pair another way, a binary pair takes its score column, positive class and
      threshold, and the choices stay in this browser for each record, table and pair. A where filter is a pandas query over the
      columns, for example price &gt; 200 and town == 'ank'.</div>
    <div class="empty" v-if="!available">this record has no predictions table yet; a finished run or kalfa predict writes one</div>
    <template v-else>
      <div class="toolbar">
        <input class="filter mono" type="search" v-model="where" placeholder="where, a pandas query" spellcheck="false"
          @keyup.enter="apply">
        <button type="button" class="tool" :disabled="loading" @click="apply">apply</button>
        <button type="button" class="tool" v-if="applied" @click="clear">clear</button>
        <select class="filter pred-select" v-if="manyTables" v-model="name">
          <option v-for="table in tables" :key="table" :value="table">{{ table }}</option>
        </select>
        <span class="muted small">{{ rowsText }}</span>
        <span class="muted small" v-if="loading && stats">updating</span>
      </div>
      <div class="error-box" v-if="error">{{ error }}</div>
      <div class="loading" v-if="loading && !stats">scoring the predictions</div>
      <div class="empty" v-if="noPairs">{{ name }} has no pred_ column</div>
      <section class="card pred-overview" v-if="manyPairs">
        <div class="card-head"><span class="card-title">overview</span><span class="muted small">{{ overviewNote }}</span></div>
        <div class="table-wrap">
          <table class="data">
            <thead>
              <tr><th>output</th><th>prediction</th><th>target</th><th>task</th><th>measure</th><th class="num">value</th><th class="num">rows</th></tr>
            </thead>
            <tbody>
              <tr v-for="row in overviewRows" :key="row.pred" :class="{ selected: pair && row.pred === pair.pred }"
                @click="pick(row.pred)">
                <td class="mono">{{ row.output }}</td>
                <td class="mono">{{ row.pred }}</td>
                <td class="mono">{{ row.target || 'none' }}</td>
                <td><span class="pred-task" :class="'is-' + row.task">{{ row.task }}</span></td>
                <td class="muted">{{ row.key }}</td>
                <td class="num mono">{{ valueText(row) }}</td>
                <td class="num">{{ integer(row.count) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>
      <div class="toolbar pred-pairs" v-if="manyPairs">
        <button type="button" class="tool" v-for="item in pairs" :key="item.pred" :class="{ on: pair && pair.pred === item.pred }"
          @click="pick(item.pred)">{{ pairLabel(item) }}</button>
      </div>
      <template v-if="pair">
        <h3 class="section-title">{{ pair.output }} <span class="pred-task" :class="'is-' + pair.task">{{ pair.task }}</span>
          <span class="muted small">{{ pairNote }}</span></h3>
        <div class="muted small">{{ reason }}. {{ taskNote }}.</div>
        <div class="pred-controls">
          <label class="pred-control">
            <span class="muted small">task</span>
            <select class="filter pred-select" :value="taskChoice" @change="setTask($event.target.value)">
              <option v-for="option in taskOptions" :key="option.value" :value="option.value">{{ option.label }}</option>
            </select>
          </label>
          <template v-if="isBinary">
            <label class="pred-control" v-if="scoreOptions.length">
              <span class="muted small">score</span>
              <select class="filter pred-select" :value="numbers.score" @change="setScore($event.target.value)">
                <option v-for="column in scoreOptions" :key="column" :value="column">{{ column }}</option>
              </select>
            </label>
            <label class="pred-control">
              <span class="muted small">positive</span>
              <select class="filter pred-select" :value="pair.positive" @change="setPositive($event.target.value)">
                <option v-for="label in positiveOptions" :key="label" :value="label">{{ label }}</option>
              </select>
            </label>
            <label class="pred-control pred-threshold" v-if="numbers.score && slider !== null">
              <span class="muted small">threshold</span>
              <input type="range" :min="range.min" :max="range.max" :step="range.step" :value="slider"
                @input="slide($event.target.value)">
              <span class="mono">{{ sliderText }}</span>
              <span class="muted small">{{ ruleText }}</span>
            </label>
            <button type="button" class="tool" v-if="choice.threshold !== undefined" @click="resetThreshold">default threshold</button>
            <span class="muted small" v-if="!numbers.score">no score column, so the predicted labels are used and there are no curves</span>
          </template>
          <button type="button" class="tool" v-if="chosen" @click="resetChoice">back to auto</button>
        </div>
        <div class="cards">
          <section class="card" v-for="item in metrics" :key="item[0]">
            <div class="stat-label">{{ item[0] }}</div>
            <div class="stat-value mono">{{ item[1] }}</div>
          </section>
        </div>
        <template v-if="isRegression">
          <div class="grid">
            <k-chart v-if="scatterData" title="truth against prediction" kind="scatter" :data="scatterData"
              :options="scatterOptions" :height="360" :log-toggle="false" :settings-key="chartKey('scatter')"></k-chart>
            <k-chart v-if="residualData" title="residuals" kind="histogram" :data="residualData" :options="residualOptions"
              :height="360" :settings-key="chartKey('residuals')" @settings="onSettings"></k-chart>
            <k-chart v-if="histogramData" title="truth and prediction histograms" kind="histogram" :data="histogramData"
              :options="histogramOptions" :height="360" :settings-key="chartKey('histograms')" @settings="onSettings"></k-chart>
          </div>
          <div class="grid">
            <section class="card">
              <div class="card-head"><span class="card-title">residual quantiles</span><span class="muted small">prediction minus truth</span></div>
              <div class="table-wrap">
                <table class="data">
                  <tbody>
                    <tr v-for="item in quantiles" :key="item.name"><td class="muted">{{ item.name }}</td><td class="num mono">{{ item.value }}</td></tr>
                  </tbody>
                </table>
              </div>
            </section>
            <section class="card" v-if="worst.length">
              <div class="card-head"><span class="card-title">worst rows</span><span class="muted small">by absolute error</span></div>
              <k-table :columns="worstColumns" :rows="worst" row-key="row" :filterable="false" empty="no rows"></k-table>
            </section>
          </div>
        </template>
        <template v-if="isBinary || isMulticlass">
          <div class="toolbar">
            <span class="muted small">confusion matrix</span>
            <button type="button" class="tool" :class="{ on: matrix === 'counts' }" @click="setMatrix('counts')">counts</button>
            <button type="button" class="tool" :class="{ on: matrix === 'rates' }" @click="setMatrix('rates')">rates</button>
            <template v-if="isMulticlass">
              <span class="muted small pred-gap">per class bars</span>
              <button type="button" class="tool" :class="{ on: metric === 'precision' }" @click="setMetric('precision')">precision</button>
              <button type="button" class="tool" :class="{ on: metric === 'recall' }" @click="setMetric('recall')">recall</button>
              <button type="button" class="tool" :class="{ on: metric === 'f1' }" @click="setMetric('f1')">f1</button>
            </template>
          </div>
          <div class="grid">
            <k-chart v-if="confusionData" :title="confusionTitle" kind="heatmap" :data="confusionData" :options="confusionOptions"
              :height="360" :log-toggle="false" :settings-key="chartKey('confusion')"></k-chart>
            <template v-if="isBinary">
              <k-chart v-if="rocData" :title="rocTitle" kind="lines" :data="rocData" :options="rocOptions" :height="360"
                :log-toggle="false" :settings-key="chartKey('roc')"></k-chart>
              <k-chart v-if="prData" :title="prTitle" kind="lines" :data="prData" :options="prOptions" :height="360"
                :log-toggle="false" :settings-key="chartKey('pr')"></k-chart>
              <k-chart v-if="scoreData" title="score by true class" kind="lines" :data="scoreData" :options="scoreChartOptions"
                :height="360" :settings-key="chartKey('scores')" bins-box @settings="onSettings"></k-chart>
              <k-chart v-if="sweepData" title="threshold sweep" kind="lines" :data="sweepData" :options="sweepOptions"
                :height="360" :log-toggle="false" :settings-key="chartKey('sweep')"></k-chart>
            </template>
            <template v-if="isMulticlass">
              <k-chart v-if="classBars" :title="metric + ' per class'" kind="bars" :data="classBars" :options="classBarsOptions"
                :height="classBarsHeight" :log-toggle="false" :settings-key="chartKey('classes')"></k-chart>
              <k-chart v-if="multiRocData" title="ROC, one class against the rest" kind="lines" :data="multiRocData"
                :options="multiRocOptions" :height="360" :log-toggle="false" :settings-key="chartKey('roc')"></k-chart>
            </template>
          </div>
          <section class="card">
            <div class="card-head"><span class="card-title">per class table</span><span class="muted small">support counts the true rows</span></div>
            <k-table :columns="classColumns" :rows="perClass" row-key="label" :filterable="false" empty="no classes"></k-table>
          </section>
        </template>
        <template v-if="isEmbedding">
          <div class="grid">
            <k-chart v-if="pcaData" :title="pcaTitle" kind="scatter" :data="pcaData" :options="pcaOptions" :height="420"
              :log-toggle="false" :settings-key="chartKey('pca')"></k-chart>
            <k-chart v-if="normData" title="vector length" kind="histogram" :data="normData" :options="normOptions"
              :height="420" :settings-key="chartKey('norms')" @settings="onSettings"></k-chart>
          </div>
        </template>
        <template v-if="isValues">
          <div class="grid">
            <k-chart v-if="valuesData" title="distribution" kind="histogram" :data="valuesData" :options="valuesOptions"
              :height="360" :settings-key="chartKey('values')" @settings="onSettings"></k-chart>
            <k-chart v-if="labelBars" title="predicted labels" kind="bars" :data="labelBars" :options="labelBarsOptions"
              :height="labelBarsHeight" :log-toggle="false" :settings-key="chartKey('labels')"></k-chart>
            <section class="card" v-if="valueQuantiles.length">
              <div class="card-head"><span class="card-title">quantiles</span></div>
              <div class="table-wrap">
                <table class="data">
                  <tbody>
                    <tr v-for="item in valueQuantiles" :key="item.name"><td class="muted">{{ item.name }}</td><td class="num mono">{{ item.value }}</td></tr>
                  </tbody>
                </table>
              </div>
            </section>
          </div>
        </template>
        <section class="card" v-if="gallery.length">
          <div class="card-head"><span class="card-title">images</span><span class="muted small">{{ gallery.length }} rows, {{ galleryNote }}</span></div>
          <div class="pred-gallery">
            <figure class="pred-shot" v-for="entry in gallery" :key="entry.index" :class="{ wrong: entryWrong(entry) }">
              <img :src="imageUrl(entry)" loading="lazy" :alt="entryText(entry)">
              <figcaption>
                <span class="mono">{{ entryText(entry) }}</span>
                <span class="muted">{{ entryDetail(entry) }}</span>
              </figcaption>
            </figure>
          </div>
        </section>
      </template>
    </template>
  </div>`,
};
