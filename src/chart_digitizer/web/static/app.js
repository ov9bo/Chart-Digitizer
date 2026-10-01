// Chart Digitizer: browser side. Plain ES module, no dependencies.
// Corner coordinates are always source-image pixels (EXIF-corrected), matching the CLI.

const $ = (id) => document.getElementById(id);
const SVGNS = "http://www.w3.org/2000/svg";
const CORNERS = [
  { tag: "BL", x: "xmin", y: "ymin" },
  { tag: "BR", x: "xmax", y: "ymin" },
  { tag: "TR", x: "xmax", y: "ymax" },
  { tag: "TL", x: "xmin", y: "ymax" },
];
const FLAG = { 0: "measured", 1: "filled", out_of_range: "outside" };
const FLAG_TEXT = { measured: "meas", filled: "fill", outside: "out" };

const state = {
  upload: null, // what /api/upload or /api/sample returned
  sourcePath: "", // for the command-line hint
  samplesDir: "",
  corners: [], // up to four [x, y] pairs
  source: "none", // none | auto | manual | placing | missing
  selected: 0,
  drag: null,
  sampleMode: "points",
  curveMode: "dark",
  result: null,
  running: false,
  outRoot: "out",
};

// ---------------------------------------------------------------- small helpers

function el(tag, attrs = {}, parent = null) {
  const node = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (parent) parent.appendChild(node);
  return node;
}

function fmt(value, digits = 4) {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  if (value !== 0 && (Math.abs(value) >= 1e6 || Math.abs(value) < 1e-3)) return value.toExponential(2);
  return String(Number(value.toPrecision(digits)));
}

let toastTimer = 0;
function toast(message) {
  const t = $("toast");
  t.textContent = message;
  t.hidden = false;
  t.style.animation = "none";
  void t.offsetWidth;
  t.style.animation = "";
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.hidden = true), 2600);
}

function setLamp(kind, text) {
  const lamp = $("lamp");
  lamp.dataset.state = kind;
  lamp.textContent = text;
}

class ApiError extends Error {
  constructor(info) {
    super(info.message);
    this.info = info;
  }
}

async function api(url, { method = "GET", body, headers = {} } = {}) {
  let response;
  try {
    response = await fetch(url, { method, body, headers });
  } catch (err) {
    throw new ApiError({
      stage: "connection",
      message: `Could not reach the digitizer server (${err.message}).`,
      hint: "Is `uv run digitize-ui` still running in a terminal?",
    });
  }
  let payload;
  try {
    payload = await response.json();
  } catch {
    throw new ApiError({ stage: "http", message: `The server answered ${response.status} without JSON.`, hint: null });
  }
  if (!response.ok || payload.error) {
    throw new ApiError(payload.error || { stage: "http", message: `HTTP ${response.status}`, hint: null });
  }
  return payload;
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    const area = Object.assign(document.createElement("textarea"), { value: text });
    document.body.appendChild(area);
    area.select();
    document.execCommand("copy");
    area.remove();
  }
}

// ---------------------------------------------------------------- source image

async function loadSamples() {
  try {
    const { samples, samples_dir, out_root } = await api("/api/samples");
    state.outRoot = out_root;
    state.samplesDir = samples_dir;
    $("outRoot").textContent = out_root;
    $("outRoot").title = out_root;
    const list = $("sampleList");
    list.replaceChildren();
    for (const name of samples) {
      const chip = Object.assign(document.createElement("button"), { type: "button", className: "chip", textContent: name });
      chip.addEventListener("click", () => loadSample(name));
      list.appendChild(chip);
    }
    $("samples").hidden = samples.length === 0;
    setLamp("idle", "Awaiting image");
  } catch (err) {
    showError(err.info);
    setLamp("error", "Offline");
  }
}

async function withLoading(label, work) {
  const drop = $("drop");
  drop.classList.add("working");
  $("dropTitle").textContent = label;
  $("dropSub").textContent = "reading · looking for the plot frame";
  setLamp("busy", "Reading");
  try {
    acceptUpload(await work());
  } catch (err) {
    showError(err.info || { stage: "load", message: String(err), hint: null });
    setLamp("error", "Load failed");
    resetDropText();
  } finally {
    drop.classList.remove("working");
  }
}

function uploadFile(file) {
  if (!file) return;
  if (file.type && !file.type.startsWith("image/")) {
    toast(`${file.name} is not an image`);
    return;
  }
  state.sourcePath = file.name;
  withLoading(file.name, () =>
    api("/api/upload", { method: "POST", body: file, headers: { "X-Filename": encodeURIComponent(file.name) } }),
  );
}

function loadSample(name) {
  state.sourcePath = state.samplesDir ? `${state.samplesDir}/${name}` : name;
  withLoading(name, () =>
    api("/api/sample", { method: "POST", body: JSON.stringify({ name }), headers: { "Content-Type": "application/json" } }),
  );
}

function resetDropText() {
  const u = state.upload;
  $("dropTitle").textContent = u ? u.name : "Drop a chart photo";
  $("dropSub").textContent = u ? `${u.width} × ${u.height} px · click or drop to replace` : "or click to browse · PNG JPG TIFF WEBP";
}

// Fill the form from the image's own YAML file (the examples ship with one).
function applySettings(settings) {
  const axes = settings.axes || {};
  const pair = (value, ids) => {
    if (Array.isArray(value) && value.length === 2) ids.forEach((id, i) => { $(id).value = value[i]; });
  };
  pair(axes.x_range, ["xmin", "xmax"]);
  pair(axes.y_range, ["ymin", "ymax"]);
  pair(axes.grid_step, ["gdx", "gdy"]);
  $("xlog").checked = Boolean(axes.x_log);
  $("ylog").checked = Boolean(axes.y_log);

  const sampling = settings.sampling || {};
  const mode = ["points", "x_values", "step"].find((m) => sampling[m] !== undefined);
  if (mode) {
    $("sampleMode").querySelector(`[data-mode="${mode}"]`).click();
    const field = { points: "nPoints", x_values: "xValues", step: "step" }[mode];
    $(field).value = Array.isArray(sampling[mode]) ? sampling[mode].join(", ") : sampling[mode];
  }

  const color = settings.curve?.color;
  $("curveMode").querySelector(`[data-mode="${color ? "color" : "dark"}"]`).click();
  if (color) {
    $("colorText").value = color;
    if (/^#[0-9a-f]{6}$/i.test(color)) $("colorPick").value = color;
  }
  if (settings.clutter && "enabled" in settings.clutter) $("clutter").checked = Boolean(settings.clutter.enabled);
}

function acceptUpload(upload) {
  state.upload = upload;
  if (upload.settings) applySettings(upload.settings);
  if (upload.settings_error) toast(upload.settings_error);
  state.result = null;
  hideError();
  $("drop").classList.add("loaded");
  resetDropText();
  $("outName").placeholder = upload.stem;
  $("empty").hidden = true;
  $("editor").hidden = false;

  const stage = $("stage");
  stage.setAttribute("viewBox", `0 0 ${upload.width} ${upload.height}`);
  const img = $("stageImg");
  img.setAttribute("href", upload.preview);
  img.setAttribute("width", upload.width);
  img.setAttribute("height", upload.height);
  loupeImage.src = upload.preview;

  if (upload.corners) {
    state.corners = upload.corners.map((p) => [...p]);
    setSource("auto");
    $("notice").hidden = true;
  } else {
    state.corners = [];
    setSource("missing");
    const e = upload.corners_error || {};
    $("notice").innerHTML = "";
    $("notice").append(
      Object.assign(document.createElement("b"), { textContent: "No plot frame found. " }),
      document.createTextNode(`${e.message || ""} Press `),
      Object.assign(document.createElement("b"), { textContent: "Place" }),
      document.createTextNode(" and click the four corners: bottom-left, bottom-right, top-right, top-left."),
    );
    $("notice").hidden = false;
  }
  state.selected = 0;
  $("btnPlace").disabled = false;
  $("btnRedetect").disabled = !upload.corners;
  $("btnRun").disabled = false;

  setTabsEnabled(false);
  $("results").hidden = true;
  $("ledgerSub").textContent = "ready to run";
  selectTab("frame");
  renderFrame();
  updateCli();
  setLamp("ready", "Ready");
  requestAnimationFrame(renderFrame); // handle sizes depend on the laid-out stage
}

// ---------------------------------------------------------------- frame editor

const loupeImage = new Image();

function setSource(source) {
  state.source = source;
  const badge = $("frameBadge");
  badge.dataset.kind = source;
  badge.textContent = {
    none: "No image",
    auto: "Found automatically",
    manual: "Set by hand",
    placing: `Click corner ${state.corners.length + 1} of 4`,
    missing: "Not found",
  }[source];
  $("btnPlace").classList.toggle("on", source === "placing");
  $("btnPlace").textContent = source === "placing" ? "Cancel" : "Place";
  $("stage").classList.toggle("placing", source === "placing");
  $("hint").textContent =
    source === "placing"
      ? `Click the ${["bottom-left", "bottom-right", "top-right", "top-left"][state.corners.length]} corner of the plot frame (${CORNERS[state.corners.length].x}, ${CORNERS[state.corners.length].y})`
      : "Drag a corner to adjust · 1–4 select · arrow keys nudge (Shift ×10)";
}

function screenScale() {
  const ctm = $("stage").getScreenCTM();
  return ctm && ctm.a ? 1 / ctm.a : 1;
}

function toImage(evt) {
  const stage = $("stage");
  const pt = stage.createSVGPoint();
  pt.x = evt.clientX;
  pt.y = evt.clientY;
  const p = pt.matrixTransform(stage.getScreenCTM().inverse());
  const { width, height } = state.upload;
  return [Math.min(Math.max(p.x, 0), width), Math.min(Math.max(p.y, 0), height)];
}

function axisValues() {
  const read = (id) => Number.parseFloat($(id).value);
  return { xmin: read("xmin"), xmax: read("xmax"), ymin: read("ymin"), ymax: read("ymax") };
}

function renderFrame() {
  if (!state.upload) return;
  const s = screenScale();
  const { width: W, height: H } = state.upload;
  const pts = state.corners;
  const full = pts.length === 4;
  const poly = pts.map((p) => p.join(",")).join(" ");

  $("quad").setAttribute("points", poly);
  $("quad").setAttribute("stroke-width", 2 * s);
  $("quad").style.strokeDasharray = `${10 * s} ${6 * s}`;
  $("shade").setAttribute("d", full ? `M0 0H${W}V${H}H0Z M${pts.map((p) => p.join(" ")).join(" L")}Z` : "");

  const axes = axisValues();
  const handles = $("handles");
  handles.replaceChildren();
  pts.forEach(([x, y], i) => {
    const g = el("g", { class: `handle${i === state.selected ? " sel" : ""}`, "data-i": i }, handles);
    el("circle", { class: "hit", cx: x, cy: y, r: 22 * s }, g);
    el("circle", { class: "ring", cx: x, cy: y, r: 13 * s, "stroke-width": 2 * s }, g);
    el("circle", { class: "dot", cx: x, cy: y, r: 2.5 * s }, g);
    const c = CORNERS[i];
    const left = i === 0 || i === 3;
    const below = i < 2;
    const label = el("text", {
      x: x + (left ? -18 : 18) * s,
      y: y + (below ? 30 : -20) * s,
      "font-size": 12 * s,
      "stroke-width": 3 * s,
      "text-anchor": left ? "end" : "start",
    }, g);
    label.textContent = `${c.tag} (${fmt(axes[c.x])}, ${fmt(axes[c.y])})`;
    g.addEventListener("pointerdown", (evt) => startDrag(evt, i));
  });

  const edges = $("edgeLabels");
  edges.replaceChildren();
  if (full) {
    const mid = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
    const [bx, by] = mid(pts[0], pts[1]);
    const [lx, ly] = mid(pts[0], pts[3]);
    const t1 = el("text", { x: bx, y: by + 24 * s, "font-size": 12 * s, "stroke-width": 3 * s, "text-anchor": "middle" }, edges);
    t1.textContent = `x ${$("xlog").checked ? "log " : ""}${fmt(axes.xmin)} → ${fmt(axes.xmax)}`;
    const t2 = el("text", { x: lx + 18 * s, y: ly, "font-size": 12 * s, "stroke-width": 3 * s }, edges);
    t2.textContent = `y ${$("ylog").checked ? "log " : ""}${fmt(axes.ymin)} → ${fmt(axes.ymax)}`;
  }
  renderCornerList();
}

function renderCornerList() {
  const list = $("cornerList");
  list.replaceChildren();
  if (!state.upload) return;
  CORNERS.forEach((c, i) => {
    const li = document.createElement("li");
    const p = state.corners[i];
    li.className = !p ? "unset" : i === state.selected ? "active" : "";
    li.innerHTML = `<b>${c.tag}</b><span>${p ? `${p[0].toFixed(1)}, ${p[1].toFixed(1)}` : "—"}</span>`;
    li.addEventListener("click", () => {
      if (!p) return;
      state.selected = i;
      renderFrame();
      $("stage").focus();
    });
    list.appendChild(li);
  });
}

function startDrag(evt, i) {
  if (state.source === "placing") return;
  evt.preventDefault();
  evt.stopPropagation();
  state.selected = i;
  const [px, py] = toImage(evt);
  const [cx, cy] = state.corners[i];
  state.drag = { i, dx: cx - px, dy: cy - py, pointer: evt.pointerId };
  $("stage").setPointerCapture(evt.pointerId);
  renderFrame();
  showLoupe(evt, state.corners[i]);
}

function onStageMove(evt) {
  if (!state.upload) return;
  const p = toImage(evt);
  if (state.drag) {
    const { i, dx, dy } = state.drag;
    const { width, height } = state.upload;
    state.corners[i] = [Math.min(Math.max(p[0] + dx, 0), width), Math.min(Math.max(p[1] + dy, 0), height)];
    if (state.source !== "manual") setSource("manual");
    renderFrame();
    showLoupe(evt, state.corners[i]);
    readout(state.corners[i]);
    return;
  }
  if (state.source === "placing") showLoupe(evt, p);
  readout(p);
}

function onStageUp() {
  if (!state.drag) return;
  state.drag = null;
  hideLoupe();
  updateCli();
}

function onStageClick(evt) {
  if (state.source !== "placing") return;
  state.corners.push(toImage(evt));
  state.selected = state.corners.length - 1;
  if (state.corners.length === 4) {
    setSource("manual");
    hideLoupe();
    $("notice").hidden = true;
    updateCli();
    toast("Frame set · drag any corner to refine");
  } else {
    setSource("placing");
  }
  renderFrame();
}

function togglePlacing() {
  if (!state.upload) return;
  if (state.source === "placing") {
    state.corners = state.placingBackup;
    setSource(state.placingBackupSource);
    hideLoupe();
  } else {
    state.placingBackup = state.corners;
    state.placingBackupSource = state.source;
    state.corners = [];
    setSource("placing");
  }
  renderFrame();
}

function redetect() {
  if (!state.upload?.corners) return;
  state.corners = state.upload.corners.map((p) => [...p]);
  setSource("auto");
  $("notice").hidden = true;
  renderFrame();
  updateCli();
}

function onStageKey(evt) {
  if (!state.upload || state.corners.length !== 4) return;
  if (/^[1-4]$/.test(evt.key)) {
    state.selected = Number(evt.key) - 1;
    renderFrame();
    return;
  }
  const move = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[evt.key];
  if (!move) return;
  evt.preventDefault();
  const step = screenScale() * (evt.shiftKey ? 10 : 1);
  const p = state.corners[state.selected];
  const { width, height } = state.upload;
  state.corners[state.selected] = [
    Math.min(Math.max(p[0] + move[0] * step, 0), width),
    Math.min(Math.max(p[1] + move[1] * step, 0), height),
  ];
  setSource("manual");
  renderFrame();
  readout(state.corners[state.selected]);
  updateCli();
}

// Magnifier: a zoomed circle of the preview around the point being placed or dragged.
function showLoupe(evt, [x, y]) {
  const canvas = $("loupe");
  if (!loupeImage.complete || !loupeImage.naturalWidth) return;
  const ctx = canvas.getContext("2d");
  const k = loupeImage.naturalWidth / state.upload.width; // preview px per source px
  const zoom = 4;
  const span = canvas.width / zoom / (window.devicePixelRatio > 1 ? 1 : 1);
  const sx = x * k;
  const sy = y * k;
  const srcSpan = span / 2; // 360 canvas px shows 90 preview px at zoom 4 -> keep it crisp
  ctx.imageSmoothingEnabled = false;
  ctx.fillStyle = "#1b1813";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(loupeImage, sx - srcSpan / 2, sy - srcSpan / 2, srcSpan, srcSpan, 0, 0, canvas.width, canvas.height);
  const c = canvas.width / 2;
  ctx.strokeStyle = "#e0431f";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(c, 0); ctx.lineTo(c, c - 14); ctx.moveTo(c, c + 14); ctx.lineTo(c, canvas.height);
  ctx.moveTo(0, c); ctx.lineTo(c - 14, c); ctx.moveTo(c + 14, c); ctx.lineTo(canvas.width, c);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(c, c, 3, 0, Math.PI * 2);
  ctx.stroke();

  const box = $("editor").getBoundingClientRect();
  const size = 180;
  let left = evt.clientX - box.left + 24;
  let top = evt.clientY - box.top - size - 24;
  if (left + size > box.width) left = evt.clientX - box.left - size - 24;
  if (top < 0) top = evt.clientY - box.top + 24;
  canvas.style.left = `${left}px`;
  canvas.style.top = `${top}px`;
  canvas.hidden = false;
}

function hideLoupe() {
  $("loupe").hidden = true;
}

// Cursor readout in data units, through the homography that maps the frame to the unit square.
function homography(src, dst) {
  const A = [];
  const b = [];
  for (let i = 0; i < 4; i++) {
    const [x, y] = src[i];
    const [u, v] = dst[i];
    A.push([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.push(u);
    A.push([0, 0, 0, x, y, 1, -v * x, -v * y]); b.push(v);
  }
  for (let c = 0; c < 8; c++) {
    let pivot = c;
    for (let r = c + 1; r < 8; r++) if (Math.abs(A[r][c]) > Math.abs(A[pivot][c])) pivot = r;
    if (Math.abs(A[pivot][c]) < 1e-12) return null;
    [A[c], A[pivot]] = [A[pivot], A[c]];
    [b[c], b[pivot]] = [b[pivot], b[c]];
    for (let r = 0; r < 8; r++) {
      if (r === c) continue;
      const f = A[r][c] / A[c][c];
      for (let k = c; k < 8; k++) A[r][k] -= f * A[c][k];
      b[r] -= f * b[c];
    }
  }
  return b.map((v, i) => v / A[i][i]);
}

function toData([x, y]) {
  if (state.corners.length !== 4) return null;
  const h = homography(state.corners, [[0, 0], [1, 0], [1, 1], [0, 1]]);
  if (!h) return null;
  const w = h[6] * x + h[7] * y + 1;
  const u = (h[0] * x + h[1] * y + h[2]) / w;
  const v = (h[3] * x + h[4] * y + h[5]) / w;
  const a = axisValues();
  const lerp = (lo, hi, t, log) => (log ? 10 ** (Math.log10(lo) + t * (Math.log10(hi) - Math.log10(lo))) : lo + t * (hi - lo));
  return [lerp(a.xmin, a.xmax, u, $("xlog").checked), lerp(a.ymin, a.ymax, v, $("ylog").checked)];
}

function readout(p) {
  const d = toData(p);
  $("coordReadout").textContent =
    `px ${p[0].toFixed(0)}, ${p[1].toFixed(0)}` + (d ? `   ·   x ${fmt(d[0])}  y ${fmt(d[1])}` : "");
}

// ---------------------------------------------------------------- settings

function num(id) {
  const raw = $(id).value.trim();
  const value = Number(raw);
  const ok = raw !== "" && Number.isFinite(value);
  $(id).classList.toggle("invalid", !ok);
  return ok ? value : raw; // unparseable text goes to the server, whose config check explains it
}

function optionalNum(id) {
  if ($(id).value.trim() === "") {
    $(id).classList.remove("invalid");
    return null;
  }
  return num(id);
}

function numberList(id) {
  const parts = $(id).value.split(/[\s,;]+/).filter(Boolean);
  const values = parts.map((p) => (Number.isFinite(Number(p)) ? Number(p) : p));
  $(id).classList.toggle("invalid", parts.length === 0 || values.some((v) => typeof v === "string"));
  return values;
}

function buildSettings() {
  const axes = {
    x_range: [num("xmin"), num("xmax")],
    y_range: [num("ymin"), num("ymax")],
    x_log: $("xlog").checked,
    y_log: $("ylog").checked,
  };
  const gdx = optionalNum("gdx");
  const gdy = optionalNum("gdy");
  if (gdx !== null || gdy !== null) axes.grid_step = [gdx ?? "", gdy ?? ""];

  const sampling =
    state.sampleMode === "points" ? { points: num("nPoints") }
      : state.sampleMode === "x_values" ? { x_values: numberList("xValues") }
        : { step: num("step") };

  const settings = {
    axes,
    sampling,
    clutter: { enabled: $("clutter").checked },
    corners: state.source === "manual" && state.corners.length === 4 ? { points: state.corners } : { mode: "auto" },
    debug: $("debug").checked,
    name: $("outName").value.trim(),
  };
  if (state.curveMode === "color") settings.curve = { color: $("colorText").value.trim() };
  return settings;
}

function updateCli() {
  const settings = buildSettings();
  const q = (s) => (/[^\w@%+=:,./\\-]/.test(s) ? `"${s}"` : s); // quote what a shell reads specially, such as #
  const a = settings.axes;
  const parts = ["uv run digitize", q(state.sourcePath || "<image>")];
  parts.push(`--x-range ${a.x_range.join(",")}`, `--y-range ${a.y_range.join(",")}`);
  if (a.x_log) parts.push("--x-log");
  if (a.y_log) parts.push("--y-log");
  if (a.grid_step) parts.push(`--grid-step ${a.grid_step.join(",")}`);
  const s = settings.sampling;
  if (s.points !== undefined) parts.push(`--points ${s.points}`);
  if (s.x_values) parts.push(`--x-values ${s.x_values.join(",")}`);
  if (s.step !== undefined) parts.push(`--step ${s.step}`);
  if (settings.corners.points) parts.push(`--corners ${settings.corners.points.flat().map((v) => v.toFixed(1)).join(",")}`);
  if (settings.curve) parts.push(`--curve-color ${q(settings.curve.color)}`);
  if (!settings.clutter.enabled) parts.push("--no-clutter");
  parts.push(`--out ${q(state.outRoot)}`);
  if (settings.debug) parts.push("--debug");
  let text = parts.join(" \\\n    ");
  if (settings.name) text += `\n# the command line names the folder after the image; here it is "${settings.name}"`;
  $("cli").textContent = text;
}

// ---------------------------------------------------------------- run

let busyTimer = 0;

async function run() {
  if (!state.upload || state.running) return;
  if (state.source === "placing") {
    toast(`Finish placing the corners first (${state.corners.length} of 4)`);
    return;
  }
  if (state.source === "missing") {
    toast("No frame was found: place the four corners first");
    togglePlacing();
    return;
  }
  const settings = buildSettings();
  updateCli();
  state.running = true;
  hideError();
  $("btnRun").disabled = true;
  $("btnRun").classList.add("busy");
  $("busy").hidden = false;
  setLamp("busy", "Digitizing");
  const t0 = performance.now();
  busyTimer = setInterval(() => ($("busyTime").textContent = `${((performance.now() - t0) / 1000).toFixed(1)} s`), 100);
  try {
    const result = await api("/api/digitize", {
      method: "POST",
      body: JSON.stringify({ id: state.upload.id, settings }),
      headers: { "Content-Type": "application/json" },
    });
    renderResult(result);
    setLamp("done", `${result.stats.kept} points`);
  } catch (err) {
    showError(err.info || { stage: "internal", message: String(err), hint: null });
    setLamp("error", "Failed");
  } finally {
    clearInterval(busyTimer);
    state.running = false;
    $("btnRun").disabled = false;
    $("btnRun").classList.remove("busy");
    $("busy").hidden = true;
  }
}

function showError(info) {
  $("errStage").textContent = info.stage || "error";
  $("errMessage").textContent = info.message || "";
  $("errHint").textContent = info.hint || "";
  const card = $("errorCard");
  card.hidden = true;
  void card.offsetWidth;
  card.hidden = false;
  $("ledgerSub").textContent = "run failed";
}

function hideError() {
  $("errorCard").hidden = true;
}

function renderResult(r) {
  state.result = r;
  $("ledgerSub").textContent = `${r.name} · ${new Date().toLocaleTimeString()}`;
  $("stKept").textContent = r.stats.kept === r.stats.total ? r.stats.kept : `${r.stats.kept}/${r.stats.total}`;
  $("stFilled").textContent = `${(r.stats.filled_fraction * 100).toFixed(1)}%`;
  $("stFrame").textContent = r.stats.corners_source;
  $("stTime").textContent = `${r.elapsed_s.toFixed(1)}s`;

  const warnings = $("warnings");
  warnings.replaceChildren(...r.warnings.map((w) => Object.assign(document.createElement("li"), { textContent: w })));

  const body = $("table").tBodies[0];
  body.replaceChildren();
  r.points.forEach((p, i) => {
    const kind = FLAG[p.interpolated_flag] || "measured";
    const tr = document.createElement("tr");
    tr.className = kind;
    tr.dataset.i = i;
    tr.style.animationDelay = `${Math.min(i * 12, 600)}ms`;
    tr.innerHTML = `<td>${i + 1}</td><td>${fmt(p.x, 6)}</td><td>${fmt(p.y, 6)}</td><td><span class="flag ${kind}">${FLAG_TEXT[kind]}</span></td>`;
    tr.addEventListener("mouseenter", () => highlight(i, false));
    tr.addEventListener("mouseleave", () => highlight(-1, false));
    body.appendChild(tr);
  });

  const files = r.files;
  $("dlCsv").href = files["points.csv"];
  $("dlJson").href = files["points.json"];
  $("dlOverlay").href = files["overlay.png"];
  $("dlReplot").href = files["replot.png"];
  $("overlayImg").src = files["overlay.png"];
  $("overlayLink").href = files["overlay.png"];
  $("replotImg").src = files["replot.png"];
  $("replotLink").href = files["replot.png"];
  $("outDir").textContent = r.out_dir;

  renderStages(r.debug);
  $("results").hidden = false;
  setTabsEnabled(true);
  for (const tab of document.querySelectorAll(".tabs button")) tab.classList.toggle("fresh", tab.dataset.tab !== "frame");
  selectTab("chart");
}

function renderStages(images) {
  const grid = $("stageGrid");
  grid.replaceChildren();
  if (!images.length) {
    grid.innerHTML = `<div class="stage-empty"><p>No stage images this time.</p>Switch on “Keep an image from every stage” and run again.</div>`;
    return;
  }
  images.forEach((d, i) => {
    const fig = document.createElement("figure");
    fig.className = "stage-card";
    fig.style.animationDelay = `${i * 40}ms`;
    const [lead, ...rest] = d.name.split("_");
    const label = /^\d+$/.test(lead) ? `<b>${lead}</b>${rest.join(" ")}` : d.name.replaceAll("_", " ");
    fig.innerHTML = `<a href="${d.url}" target="_blank" rel="noopener"><img loading="lazy" src="${d.url}" alt=""></a><figcaption>${label}</figcaption>`;
    grid.appendChild(fig);
  });
}

// ---------------------------------------------------------------- tabs

function setTabsEnabled(enabled) {
  for (const tab of document.querySelectorAll(".tabs button")) {
    if (tab.dataset.tab !== "frame") tab.disabled = !enabled;
    tab.classList.remove("fresh");
  }
}

function selectTab(name) {
  for (const tab of document.querySelectorAll(".tabs button")) {
    const on = tab.dataset.tab === name;
    tab.setAttribute("aria-selected", on);
    if (on) tab.classList.remove("fresh");
  }
  for (const view of document.querySelectorAll(".view")) view.hidden = view.dataset.view !== name;
  if (name === "chart") renderChart();
  if (name === "frame") requestAnimationFrame(renderFrame);
}

// ---------------------------------------------------------------- data chart

function niceTicks(lo, hi, count = 6) {
  const span = Math.abs(hi - lo) || 1;
  const raw = span / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) || mag * 10;
  return ticksBy(lo, hi, step);
}

function ticksBy(lo, hi, step) {
  const [a, b] = lo < hi ? [lo, hi] : [hi, lo];
  const out = [];
  for (let v = Math.ceil(a / step - 1e-9) * step; v <= b + step * 1e-9 && out.length < 200; v += step) out.push(Number(v.toPrecision(12)));
  return out;
}

function logTicks(lo, hi) {
  const [a, b] = lo < hi ? [lo, hi] : [hi, lo];
  const out = [];
  for (let e = Math.floor(Math.log10(a)); e <= Math.ceil(Math.log10(b)); e++) {
    for (const m of [1, 2, 5]) {
      const v = m * 10 ** e;
      if (v >= a * (1 - 1e-9) && v <= b * (1 + 1e-9)) out.push({ v, major: m === 1 });
    }
  }
  return out;
}

function scale(lo, hi, log, p0, p1) {
  const f = log ? Math.log10 : (v) => v;
  const a = f(lo);
  const b = f(hi);
  return (v) => p0 + ((f(v) - a) / (b - a)) * (p1 - p0);
}

function renderChart() {
  const r = state.result;
  const svg = $("chart");
  if (!r || svg.closest(".view").hidden) return;
  const { width, height } = svg.getBoundingClientRect();
  if (!width || !height) return;
  const ax = r.meta.axes;
  const m = { l: 64, r: 22, t: 18, b: 52 };
  const X = scale(ax.x_range[0], ax.x_range[1], ax.x_log, m.l, width - m.r);
  const Y = scale(ax.y_range[0], ax.y_range[1], ax.y_log, height - m.b, m.t);
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.replaceChildren();

  const ticksFor = (range, log, step) =>
    log ? logTicks(...range)
      : (step ? ticksBy(range[0], range[1], step) : niceTicks(...range)).map((v) => ({ v, major: true }));
  const xt = ticksFor(ax.x_range, ax.x_log, ax.grid_step?.[0]);
  const yt = ticksFor(ax.y_range, ax.y_log, ax.grid_step?.[1]);

  const grid = el("g", { class: "grid" }, svg);
  for (const t of xt) el("line", { x1: X(t.v), x2: X(t.v), y1: m.t, y2: height - m.b, class: t.major ? "major" : "" }, grid);
  for (const t of yt) el("line", { y1: Y(t.v), y2: Y(t.v), x1: m.l, x2: width - m.r, class: t.major ? "major" : "" }, grid);

  const axis = el("g", { class: "axis" }, svg);
  el("path", { d: `M${m.l} ${m.t}V${height - m.b}H${width - m.r}`, fill: "none", "stroke-width": 1.5 }, axis);
  const every = (ticks) => Math.max(1, Math.ceil(ticks.filter((t) => t.major).length / 12));
  const xEvery = every(xt);
  xt.filter((t) => t.major).forEach((t, i) => {
    el("line", { x1: X(t.v), x2: X(t.v), y1: height - m.b, y2: height - m.b + 5 }, axis);
    if (i % xEvery === 0) el("text", { x: X(t.v), y: height - m.b + 18, "text-anchor": "middle" }, axis).textContent = fmt(t.v);
  });
  const yEvery = every(yt);
  yt.filter((t) => t.major).forEach((t, i) => {
    el("line", { x1: m.l - 5, x2: m.l, y1: Y(t.v), y2: Y(t.v) }, axis);
    if (i % yEvery === 0) el("text", { x: m.l - 9, y: Y(t.v) + 3.5, "text-anchor": "end" }, axis).textContent = fmt(t.v);
  });
  el("text", { class: "axis-title", x: (m.l + width - m.r) / 2, y: height - 10, "text-anchor": "middle" }, svg).textContent = ax.x_log ? "x (log)" : "x";
  el("text", { class: "axis-title", x: 18, y: (m.t + height - m.b) / 2, "text-anchor": "middle", transform: `rotate(-90 18 ${(m.t + height - m.b) / 2})` }, svg).textContent = ax.y_log ? "y (log)" : "y";

  // Curve through the in-range points, broken where a point is out of range.
  let d = "";
  let pen = false;
  for (const p of r.points) {
    if (p.y === null || p.x === null) { pen = false; continue; }
    d += `${pen ? "L" : "M"}${X(p.x).toFixed(1)} ${Y(p.y).toFixed(1)}`;
    pen = true;
  }
  const curve = el("path", { class: "curve", d }, svg);
  curve.style.setProperty("--len", Math.ceil(curve.getTotalLength?.() || 4000));

  const dots = el("g", {}, svg);
  const baseY = height - m.b;
  const n = r.points.length;
  const radius = n > 150 ? 2.2 : n > 60 ? 3 : 4.2;
  r.points.forEach((p, i) => {
    const kind = FLAG[p.interpolated_flag] || "measured";
    if (p.x === null) return;
    const c = el("circle", { class: `pt ${kind}`, cx: X(p.x), cy: p.y === null ? baseY : Y(p.y), r: kind === "outside" ? radius * 0.7 : radius, "data-i": i }, dots);
    c.style.animationDelay = `${600 + (i / n) * 600}ms`;
  });
  el("line", { class: "cross", id: "crossV", y1: m.t, y2: height - m.b, x1: -10, x2: -10 }, svg);
  el("line", { class: "cross", id: "crossH", x1: m.l, x2: width - m.r, y1: -10, y2: -10 }, svg);
  chartGeometry = { X, Y, baseY, m, width, height };
}

let chartGeometry = null;

function onChartMove(evt) {
  const r = state.result;
  if (!r || !chartGeometry) return;
  const box = $("chart").getBoundingClientRect();
  const mx = evt.clientX - box.left;
  let best = -1;
  let bestDist = Infinity;
  r.points.forEach((p, i) => {
    if (p.x === null) return;
    const dist = Math.abs(chartGeometry.X(p.x) - mx);
    if (dist < bestDist) { bestDist = dist; best = i; }
  });
  highlight(bestDist < 40 ? best : -1, true);
}

function highlight(i, fromChart) {
  const svg = $("chart");
  for (const c of svg.querySelectorAll(".pt.hot")) c.classList.remove("hot");
  for (const row of document.querySelectorAll("#table tr.hot")) row.classList.remove("hot");
  const tip = $("tip");
  const r = state.result;
  if (i < 0 || !r || !chartGeometry) {
    tip.hidden = true;
    for (const line of svg.querySelectorAll(".cross")) line.classList.remove("on");
    return;
  }
  const p = r.points[i];
  const row = document.querySelector(`#table tr[data-i="${i}"]`);
  row?.classList.add("hot");
  if (fromChart) row?.scrollIntoView({ block: "nearest" });
  const dot = svg.querySelector(`.pt[data-i="${i}"]`);
  if (!dot) return;
  dot.classList.add("hot");
  const cx = Number(dot.getAttribute("cx"));
  const cy = Number(dot.getAttribute("cy"));
  svg.querySelector("#crossV").setAttribute("x1", cx);
  svg.querySelector("#crossV").setAttribute("x2", cx);
  svg.querySelector("#crossH").setAttribute("y1", cy);
  svg.querySelector("#crossH").setAttribute("y2", cy);
  for (const line of svg.querySelectorAll(".cross")) line.classList.add("on");
  const kind = FLAG[p.interpolated_flag] || "measured";
  tip.innerHTML = `#${i + 1} · x <b>${fmt(p.x, 6)}</b> · y <b>${fmt(p.y, 6)}</b>${kind === "measured" ? "" : ` · ${kind === "filled" ? "gap-filled" : "outside trace"}`}`;
  const wrap = $("chart").parentElement.getBoundingClientRect();
  const svgBox = svg.getBoundingClientRect();
  tip.style.left = `${cx + svgBox.left - wrap.left}px`;
  tip.style.top = `${cy + svgBox.top - wrap.top}px`;
  tip.hidden = false;
}

// ---------------------------------------------------------------- wiring

function setSegment(groupId, mode) {
  for (const b of $(groupId).querySelectorAll("button")) b.setAttribute("aria-checked", b.dataset.mode === mode);
}

function wire() {
  $("file").addEventListener("change", (e) => {
    uploadFile(e.target.files[0]);
    e.target.value = "";
  });

  // Whole-page drag and drop.
  let depth = 0;
  const hasFiles = (e) => [...(e.dataTransfer?.types || [])].includes("Files");
  window.addEventListener("dragenter", (e) => { if (!hasFiles(e)) return; e.preventDefault(); depth++; $("dropVeil").hidden = false; });
  window.addEventListener("dragover", (e) => { if (hasFiles(e)) e.preventDefault(); });
  window.addEventListener("dragleave", () => { depth = Math.max(0, depth - 1); if (!depth) $("dropVeil").hidden = true; });
  window.addEventListener("drop", (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    depth = 0;
    $("dropVeil").hidden = true;
    uploadFile(e.dataTransfer.files[0]);
  });

  const stage = $("stage");
  stage.addEventListener("pointermove", onStageMove);
  stage.addEventListener("pointerup", onStageUp);
  stage.addEventListener("pointercancel", onStageUp);
  stage.addEventListener("pointerleave", () => { if (!state.drag) hideLoupe(); });
  stage.addEventListener("click", onStageClick);
  stage.addEventListener("keydown", onStageKey);
  $("btnPlace").addEventListener("click", togglePlacing);
  $("btnRedetect").addEventListener("click", redetect);

  for (const id of ["xmin", "xmax", "ymin", "ymax", "xlog", "ylog"]) {
    $(id).addEventListener("input", () => { renderFrame(); updateCli(); });
  }
  for (const id of ["gdx", "gdy", "nPoints", "xValues", "step", "outName", "clutter", "debug", "colorText"]) {
    $(id).addEventListener("input", updateCli);
  }

  $("sampleMode").addEventListener("click", (e) => {
    const mode = e.target.closest("button")?.dataset.mode;
    if (!mode) return;
    state.sampleMode = mode;
    setSegment("sampleMode", mode);
    for (const label of document.querySelectorAll("[data-for]")) label.hidden = label.dataset.for !== mode;
    document.querySelector(`[data-for="${mode}"] input`).focus();
    updateCli();
  });
  $("curveMode").addEventListener("click", (e) => {
    const mode = e.target.closest("button")?.dataset.mode;
    if (!mode) return;
    state.curveMode = mode;
    setSegment("curveMode", mode);
    $("colorRow").hidden = mode !== "color";
    updateCli();
  });
  $("colorPick").addEventListener("input", (e) => { $("colorText").value = e.target.value; updateCli(); });
  $("colorText").addEventListener("input", (e) => {
    if (/^#[0-9a-f]{6}$/i.test(e.target.value.trim())) $("colorPick").value = e.target.value.trim();
  });

  $("sheet").addEventListener("submit", (e) => { e.preventDefault(); run(); });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); run(); }
    if (e.key === "Escape" && state.source === "placing") togglePlacing();
  });

  for (const tab of document.querySelectorAll(".tabs button")) {
    tab.addEventListener("click", () => !tab.disabled && selectTab(tab.dataset.tab));
  }

  $("chart").addEventListener("pointermove", onChartMove);
  $("chart").addEventListener("pointerleave", () => highlight(-1, false));
  new ResizeObserver(() => { renderChart(); renderFrame(); }).observe(document.querySelector(".views"));

  $("btnCopy").addEventListener("click", async () => {
    const rows = state.result?.points.map((p) => [p.x ?? "", p.y ?? "", p.interpolated_flag].join("\t")) || [];
    await copyText(["x\ty\tinterpolated_flag", ...rows].join("\n"));
    toast(`Copied ${rows.length} rows · paste into a spreadsheet`);
  });
  $("btnCopyCli").addEventListener("click", async () => {
    await copyText($("cli").textContent.replace(/\n#.*$/m, ""));
    toast("Command copied");
  });
}

function decorateHero() {
  // Faint grid and sample dots for the empty-state drawing.
  const grid = document.querySelector(".hero-grid");
  for (let x = 102; x < 480; x += 42) el("line", { x1: x, x2: x, y1: 30, y2: 270 }, grid);
  for (let y = 222; y > 30; y -= 48) el("line", { x1: 60, x2: 480, y1: y, y2: y }, grid);
  const dots = document.querySelector(".hero-dots");
  const path = document.querySelector(".hero-curve");
  const length = path.getTotalLength();
  for (let k = 1; k <= 9; k++) {
    const p = path.getPointAtLength((length * k) / 10);
    const c = el("circle", { cx: p.x, cy: p.y, r: 4.5 }, dots);
    c.style.animationDelay = `${1.6 + k * 0.08}s`;
  }
}

wire();
decorateHero();
updateCli();
loadSamples();
