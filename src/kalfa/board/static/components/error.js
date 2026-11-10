export function messageOf(error) {
  if (!error) return "";
  if (typeof error === "string") return error;
  return error.message || String(error);
}

export const ErrorBox = {
  name: "KError",
  props: { error: { default: null } },
  setup(props) {
    const message = () => messageOf(props.error);
    return { message };
  },
  template: '<div v-if="message()" class="error-box" role="alert">{{ message() }}</div>',
};

export default ErrorBox;
