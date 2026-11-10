import { fraction, share } from "../format.js";


export const Progress = {
  name: "KProgress",
  props: {
    value: { type: Number, default: 0 },
    total: { type: Number, default: null },
    label: { type: String, default: "" },
  },
  setup(props) {
    const part = () => share(props.value, props.total);
    const width = () => (part() === null ? "0%" : (100 * part()).toFixed(1) + "%");
    const caption = () => props.label || fraction(props.value, props.total);
    return { part, width, caption };
  },
  template: `
    <span class="progress" :class="{ unknown: part() === null }">
      <span class="progress-track"><span class="progress-fill" :style="{ width: width() }"></span></span>
      <span class="progress-text">{{ caption() }}</span>
    </span>`,
};

export default Progress;
