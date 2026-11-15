import { computed } from "https://cdn.jsdelivr.net/npm/vue@3.4.38/dist/vue.esm-browser.prod.js";


export function sparkPoints(values, width, height, count) {
  const tail = (Array.isArray(values) ? values : []).slice(-count);
  const finite = tail.map((value, index) => [index, value])
    .filter((pair) => typeof pair[1] === "number" && Number.isFinite(pair[1]));
  if (finite.length < 2) return "";
  const numbers = finite.map((pair) => pair[1]);
  const low = Math.min(...numbers);
  const high = Math.max(...numbers);
  const span = high - low || 1;
  const last = Math.max(tail.length - 1, 1);
  return finite.map((pair) => {
    const x = 1 + (pair[0] / last) * (width - 2);
    const y = height - 1 - ((pair[1] - low) / span) * (height - 2);
    return x.toFixed(1) + "," + y.toFixed(1);
  }).join(" ");
}

export const Spark = {
  name: "KSpark",
  props: {
    values: { type: Array, default: () => [] },
    width: { type: Number, default: 80 },
    height: { type: Number, default: 20 },
    count: { type: Number, default: 40 },
    title: { type: String, default: "" },
  },
  setup(props) {
    const points = computed(() => sparkPoints(props.values, props.width, props.height, props.count));
    const box = computed(() => "0 0 " + props.width + " " + props.height);
    return { points, box };
  },
  template: `
    <svg class="spark" :width="width" :height="height" :viewBox="box" preserveAspectRatio="none" role="img">
      <title v-if="title">{{ title }}</title>
      <polyline v-if="points" :points="points" fill="none" stroke="currentColor" stroke-width="1.5"
        stroke-linejoin="round" stroke-linecap="round"></polyline>
    </svg>`,
};

export default Spark;
