import { computed, ref } from "https://cdn.jsdelivr.net/npm/vue@3.4.38/dist/vue.esm-browser.prod.js";

import { EMPTY, number, text } from "../format.js";
import { Badge } from "./badge.js";
import { Progress } from "./progress.js";
import { Spark } from "./spark.js";


export function missing(value) {
  return value === null || value === undefined || value === "" || (typeof value === "number" && Number.isNaN(value));
}

export function compareValues(first, second) {
  if (typeof first === "number" && typeof second === "number") return first - second;
  return String(first).localeCompare(String(second), undefined, { numeric: true, sensitivity: "base" });
}

export function valueOf(row, column) {
  if (column.value) return column.value(row);
  return row ? row[column.key] : undefined;
}

export function textOf(row, column) {
  if (column.text) return column.text(row);
  const value = valueOf(row, column);
  if (missing(value)) return EMPTY;
  return typeof value === "number" ? number(value) : text(value);
}

export function sortRows(rows, column, descending) {
  if (!column) return rows;
  const sign = descending ? -1 : 1;
  return [...rows].sort((first, second) => {
    const left = valueOf(first, column);
    const right = valueOf(second, column);
    const leftMissing = missing(left);
    const rightMissing = missing(right);
    if (leftMissing || rightMissing) return leftMissing === rightMissing ? 0 : leftMissing ? 1 : -1;
    return compareValues(left, right) * sign;
  });
}

export const Table = {
  name: "KTable",
  components: { KBadge: Badge, KProgress: Progress, KSpark: Spark },
  props: {
    columns: { type: Array, default: () => [] },
    rows: { type: Array, default: () => [] },
    rowKey: { type: String, default: "path" },
    selectable: { type: Boolean, default: false },
    canSelect: { type: Function, default: null },
    selected: { type: Array, default: () => [] },
    filterable: { type: Boolean, default: true },
    placeholder: { type: String, default: "filter" },
    sortKey: { type: String, default: "" },
    sortDesc: { type: Boolean, default: false },
    empty: { type: String, default: "nothing to show" },
    rowClass: { type: Function, default: null },
  },
  emits: ["update:selected", "select"],
  setup(props, { emit }) {
    const needle = ref("");
    const key = ref(props.sortKey);
    const descending = ref(props.sortDesc);
    const keyOf = (row) => String(row ? row[props.rowKey] : "");
    const matches = (row, wanted) => props.columns.some((column) => String(textOf(row, column)).toLowerCase()
      .includes(wanted));
    const shown = computed(() => {
      const wanted = needle.value.trim().toLowerCase();
      const rows = (props.rows || []).filter((row) => !wanted || matches(row, wanted));
      return sortRows(rows, props.columns.find((column) => column.key === key.value), descending.value);
    });
    const sortBy = (column) => {
      if (column.sortable === false) return;
      if (key.value === column.key) descending.value = !descending.value;
      else {
        key.value = column.key;
        descending.value = Boolean(column.descending);
      }
    };
    const arrow = (column) => (key.value === column.key ? (descending.value ? "▼" : "▲") : "");
    const selectableRow = (row) => (props.canSelect ? Boolean(props.canSelect(row)) : true);
    const isSelected = (row) => props.selected.includes(keyOf(row));
    const publish = (keys) => {
      const unique = [...new Set(keys)];
      emit("update:selected", unique);
      emit("select", unique);
    };
    const toggle = (row) => {
      const id = keyOf(row);
      publish(isSelected(row) ? props.selected.filter((item) => item !== id) : [...props.selected, id]);
    };
    const pickable = computed(() => shown.value.filter(selectableRow));
    const allSelected = computed(() => pickable.value.length > 0 && pickable.value.every(isSelected));
    const toggleAll = () => {
      const ids = pickable.value.map(keyOf);
      if (allSelected.value) publish(props.selected.filter((item) => !ids.includes(item)));
      else publish([...props.selected, ...ids]);
    };
    const href = (row, column) => (column.href ? column.href(row) : "");
    const cellClass = (column) => [column.numeric ? "num" : "", column.className || ""].join(" ").trim();
    const classOf = (row) => [props.rowClass ? props.rowClass(row) : "", isSelected(row) ? "selected" : ""]
      .join(" ").trim();
    const progressOf = (row, column) => column.progress(row) || { value: 0, total: null };
    const titleOf = (row, column) => (column.title ? column.title(row) : "");
    const count = computed(() => shown.value.length);
    return {
      needle, shown, sortBy, arrow, selectableRow, isSelected, toggle, allSelected, toggleAll, href, cellClass,
      classOf, progressOf, titleOf, count, keyOf, valueOf, textOf,
    };
  },
  template: `
    <div class="table-box">
      <div class="table-tools" v-if="filterable">
        <input class="filter" type="search" v-model="needle" :placeholder="placeholder">
        <span class="muted small">{{ count }} of {{ rows.length }}</span>
        <slot name="tools"></slot>
      </div>
      <div class="table-wrap">
        <table class="data">
          <thead>
            <tr>
              <th v-if="selectable" class="check">
                <input type="checkbox" :checked="allSelected" @change="toggleAll" title="select every shown row">
              </th>
              <th v-for="column in columns" :key="column.key" :class="[cellClass(column), column.sortable === false ? '' : 'sortable']"
                @click="sortBy(column)" :title="column.hint || ''">
                {{ column.label || column.key }} <span class="arrow">{{ arrow(column) }}</span>
              </th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="row in shown" :key="keyOf(row)" :class="classOf(row)">
              <td v-if="selectable" class="check">
                <input v-if="selectableRow(row)" type="checkbox" :checked="isSelected(row)" @change="toggle(row)">
              </td>
              <td v-for="column in columns" :key="column.key" :class="cellClass(column)" :title="titleOf(row, column)">
                <k-badge v-if="column.badge" :status="valueOf(row, column)"></k-badge>
                <k-progress v-else-if="column.progress" :value="progressOf(row, column).value"
                  :total="progressOf(row, column).total" :label="progressOf(row, column).label || ''"></k-progress>
                <k-spark v-else-if="column.spark" :values="column.spark(row)"></k-spark>
                <a v-else-if="href(row, column)" class="link" :href="href(row, column)">{{ textOf(row, column) }}</a>
                <span v-else>{{ textOf(row, column) }}</span>
              </td>
            </tr>
            <tr v-if="!shown.length">
              <td class="empty" :colspan="columns.length + (selectable ? 1 : 0)">{{ empty }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>`,
};

export default Table;
