export const EMPTY = "·";

export function isNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

function compact(text) {
  const [mantissa, power] = text.split("e");
  const trimmed = mantissa.indexOf(".") < 0 ? mantissa : mantissa.replace(/0+$/, "").replace(/\.$/, "");
  return power === undefined ? trimmed : trimmed + "e" + power;
}

export function number(value, digits = 4) {
  if (value === null || value === undefined || value === "") return EMPTY;
  if (typeof value === "boolean") return String(value);
  if (typeof value !== "number") return String(value);
  if (!Number.isFinite(value)) return String(value);
  if (value === 0) return "0";
  if (Number.isInteger(value) && Math.abs(value) < 1e6) return String(value);
  const exponent = Math.floor(Math.log10(Math.abs(value)));
  if (exponent < -4 || exponent >= digits) return compact(value.toExponential(digits - 1));
  return compact(value.toPrecision(digits));
}

export function integer(value) {
  return isNumber(value) ? Math.round(value).toLocaleString() : EMPTY;
}

function pad(value) {
  return String(Math.floor(value)).padStart(2, "0");
}

export function seconds(value) {
  if (!isNumber(value)) return EMPTY;
  if (value < 60) return value.toFixed(1) + "s";
  if (value < 3600) return Math.floor(value / 60) + "m " + pad(value % 60) + "s";
  return Math.floor(value / 3600) + "h " + pad((value % 3600) / 60) + "m";
}

export function moment(iso) {
  if (!iso) return null;
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? null : date;
}

export function when(iso) {
  const date = moment(iso);
  if (!date) return iso ? String(iso) : EMPTY;
  const today = new Date();
  const time = date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  if (date.toDateString() === today.toDateString()) return time;
  const day = date.toLocaleDateString([], { month: "short", day: "numeric" });
  if (date.getFullYear() === today.getFullYear()) return day + " " + time;
  return date.toLocaleDateString() + " " + time;
}

export function ago(iso) {
  const date = moment(iso);
  if (!date) return EMPTY;
  const elapsed = Math.max(0, (Date.now() - date.getTime()) / 1000);
  if (elapsed < 60) return Math.round(elapsed) + "s ago";
  if (elapsed < 3600) return Math.round(elapsed / 60) + "m ago";
  if (elapsed < 86400) return Math.round(elapsed / 3600) + "h ago";
  return Math.round(elapsed / 86400) + "d ago";
}

export function percent(value, digits = 0) {
  return isNumber(value) ? (100 * value).toFixed(digits) + "%" : EMPTY;
}

export function bytes(value) {
  if (!isNumber(value)) return EMPTY;
  const units = ["B", "KB", "MB", "GB", "TB"];
  let size = value;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  return (unit === 0 ? String(size) : size.toFixed(size < 10 ? 1 : 0)) + " " + units[unit];
}

export function fraction(turn, total) {
  const done = isNumber(turn) ? String(turn) : "0";
  return isNumber(total) && total > 0 ? done + "/" + total : done;
}

export function share(turn, total) {
  if (!isNumber(turn) || !isNumber(total) || total <= 0) return null;
  return Math.max(0, Math.min(1, turn / total));
}

export function text(value) {
  if (value === null || value === undefined || value === "") return EMPTY;
  if (typeof value === "number") return number(value);
  if (typeof value === "object") {
    try {
      return JSON.stringify(value);
    } catch (error) {
      return String(value);
    }
  }
  return String(value);
}

export function bestText(best) {
  if (!best || typeof best !== "object" || !best.series) return EMPTY;
  return best.series + " " + number(best.value);
}
