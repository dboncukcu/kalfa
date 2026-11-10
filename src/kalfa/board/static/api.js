export function encodePath(path) {
  return String(path || "").split("/").filter(Boolean).map(encodeURIComponent).join("/");
}

function component(value) {
  return encodeURIComponent(value).replace(/%2C/gi, ",").replace(/%2F/gi, "/");
}

export function query(params) {
  const parts = [];
  for (const [key, value] of Object.entries(params || {})) {
    if (value === null || value === undefined) continue;
    const item = Array.isArray(value) ? value.join(",") : String(value);
    if (item) parts.push(component(key) + "=" + component(item));
  }
  return parts.length ? "?" + parts.join("&") : "";
}

export function recordUrl(path, suffix = "", params = null) {
  const tail = suffix ? "/" + suffix : "";
  return "/api/record/" + encodePath(path) + tail + query(params);
}

export function fileUrl(path, name, tail = null) {
  return recordUrl(path, "file", { name, tail });
}

export function histogramsText(histograms) {
  if (!histograms || typeof histograms !== "object" || Array.isArray(histograms)) return histograms || null;
  const parts = Object.entries(histograms)
    .filter(([name, count]) => name && Number.isInteger(count) && count > 0)
    .map(([name, count]) => name + ":" + count);
  return parts.length ? parts.join(",") : null;
}

export function predictionsUrl(path, params = null) {
  const found = { ...(params || {}) };
  if (found.histograms !== undefined) found.histograms = histogramsText(found.histograms);
  return recordUrl(path, "predictions", found);
}

export function imageUrl(path, index, name = null) {
  return recordUrl(path, "image", { index, name });
}

export function sweepUrl(path) {
  return "/api/sweep/" + encodePath(path);
}

async function reach(path, init) {
  try {
    return await fetch(path, { cache: "no-store", ...init });
  } catch (error) {
    throw new Error("cannot reach the board server (" + error.message + ")");
  }
}

function failure(response, path, body) {
  const message = body && typeof body === "object" && body.error
    ? body.error
    : response.status + " " + (response.statusText || "error") + " for " + path;
  const error = new Error(message);
  error.status = response.status;
  return error;
}

async function readJson(response, path) {
  const text = await response.text();
  let body = null;
  let parsed = false;
  if (text) {
    try {
      body = JSON.parse(text);
      parsed = true;
    } catch (error) {
      parsed = false;
    }
  }
  if (!response.ok) throw failure(response, path, body);
  if (text && !parsed) throw new Error("the answer of " + path + " is not JSON");
  return body;
}

export async function getJson(path) {
  const response = await reach(path, { headers: { Accept: "application/json" } });
  return readJson(response, path);
}

export async function postJson(path, body = {}) {
  const response = await reach(path, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(body === undefined ? {} : body),
  });
  return readJson(response, path);
}

export function getPredictions(path, params = null) {
  return getJson(predictionsUrl(path, params));
}

export async function getText(path) {
  const response = await reach(path, {});
  if (!response.ok) {
    let body = null;
    try {
      body = JSON.parse(await response.text());
    } catch (error) {
      body = null;
    }
    throw failure(response, path, body);
  }
  return response.text();
}

export function live(onChange, onState = null, every = 2) {
  let source = null;
  let timer = null;
  let delay = 2000;
  let stopped = false;
  let dropped = false;
  const state = (connected) => {
    if (onState) onState(connected);
  };
  const connect = () => {
    if (stopped) return;
    if (typeof EventSource === "undefined") {
      state(false);
      return;
    }
    source = new EventSource("/api/live" + query({ every }));
    source.addEventListener("open", () => {
      delay = 2000;
      state(true);
      if (dropped) {
        dropped = false;
        onChange(null);
      }
    });
    source.addEventListener("change", (event) => {
      let changed = [];
      try {
        const data = JSON.parse(event.data);
        changed = Array.isArray(data.changed) ? data.changed : [];
      } catch (error) {
        changed = [];
      }
      onChange(changed);
    });
    source.addEventListener("error", () => {
      state(false);
      if (source) source.close();
      source = null;
      if (stopped) return;
      dropped = true;
      timer = setTimeout(() => {
        timer = null;
        connect();
      }, delay);
      delay = Math.min(delay * 2, 30000);
    });
  };
  connect();
  return () => {
    stopped = true;
    if (timer) clearTimeout(timer);
    if (source) source.close();
    source = null;
    state(false);
  };
}

export const client = Object.freeze({
  encodePath, query, recordUrl, fileUrl, histogramsText, predictionsUrl, imageUrl, sweepUrl, getJson, postJson,
  getText, getPredictions, live,
});
