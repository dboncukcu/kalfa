const KNOWN = ["running", "stale", "done", "stopped", "failed", "partial", "unknown"];

export const Badge = {
  name: "KBadge",
  props: { status: { type: String, default: "unknown" } },
  setup(props) {
    const label = () => props.status || "unknown";
    const tone = () => (KNOWN.includes(label()) ? label() : "unknown");
    return { label, tone };
  },
  template: '<span class="badge" :class="\'is-\' + tone()">{{ label() }}</span>',
};

export default Badge;
