"use strict";

const API_URL = "/api/predict";
const FRAME_SIZE = 256;          // the training images were 256x256 squares
const THUMB_SIZE = 160;
const LIVE_INTERVAL_MS = 700;    // pause between live-detect requests
const HISTORY_KEY = "leaflens.history";
const HISTORY_LIMIT = 12;

const TYPE_LABELS = {
  fungus: "Fungal disease",
  oomycete: "Water-mould disease",
  bacterium: "Bacterial disease",
  virus: "Viral disease",
  pest: "Pest damage",
  disorder: "Nutrient or stress problem",
  unknown: "Possible problem",
};

const $ = (id) => document.getElementById(id);
const els = {
  tabCamera: $("tab-camera"),
  tabUpload: $("tab-upload"),
  panelCamera: $("panel-camera"),
  panelUpload: $("panel-upload"),
  heroCamera: $("hero-camera"),
  heroUpload: $("hero-upload"),
  video: $("video"),
  snapshot: $("snapshot"),
  viewfinder: $("viewfinder"),
  closeCamera: $("close-camera"),
  overlay: $("cam-overlay"),
  camTitle: $("cam-title"),
  camText: $("cam-text"),
  startCamera: $("start-camera"),
  nativeCamera: $("native-camera"),
  livePill: $("live-pill"),
  liveToggle: $("live-toggle"),
  shutter: $("shutter"),
  shutterLabel: $("shutter-label"),
  retake: $("retake"),
  altCapture: $("alt-capture"),
  flip: $("flip-camera"),
  dropzone: $("dropzone"),
  fileInput: $("file-input"),
  nativeInput: $("native-input"),
  result: $("result"),
  resultEmpty: $("result-empty"),
  resultLoading: $("result-loading"),
  resultBody: $("result-body"),
  loadingImg: $("loading-img"),
  historyGrid: $("history-grid"),
  historyEmpty: $("history-empty"),
  clearHistory: $("clear-history"),
  historyTemplate: $("tpl-history-item"),
};

const state = { stream: null, facing: "environment", live: false, liveTimer: 0, busy: false, frozen: false };

// ------------------------------------------------------------------ helpers

const escapeHtml = (value) =>
  String(value).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

const percent = (value) => `${Math.round(value * 100)}%`;

const statusOf = (data) => (data.uncertain ? "uncertain" : data.prediction.healthy ? "healthy" : "disease");

const isNarrow = () => window.matchMedia("(max-width: 960px)").matches;

/** Centre-crop any image/video source to a square canvas, the shape the model was trained on. */
function squareCanvas(source, width, height, size = FRAME_SIZE) {
  const side = Math.min(width, height);
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(source, (width - side) / 2, (height - side) / 2, side, side, 0, 0, size, size);
  return canvas;
}

const canvasToBlob = (canvas) =>
  new Promise((resolve, reject) =>
    canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error("Couldn't encode the image."))), "image/jpeg", 0.92)
  );

function loadImage(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => { URL.revokeObjectURL(url); resolve(img); };
    img.onerror = () => { URL.revokeObjectURL(url); reject(new Error("That file couldn't be opened as an image. Try a JPG or PNG photo.")); };
    img.src = url;
  });
}

async function classify(blob) {
  const body = new FormData();
  body.append("image", blob, "leaf.jpg");
  let response;
  try {
    response = await fetch(API_URL, { method: "POST", body });
  } catch {
    throw new Error("Couldn't reach the LeafLens server. Check that app.py is still running.");
  }
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `The server returned an error (${response.status}).`);
  return data;
}

// ------------------------------------------------------------------ results

function showResultState(which) {
  els.resultEmpty.hidden = which !== "empty";
  els.resultLoading.hidden = which !== "loading";
  els.resultBody.hidden = which !== "body";
}

function factBlock(title, items, highlight = false) {
  if (!items || !items.length) return "";
  return `<section class="fact${highlight ? " is-act" : ""}">
      <h3>${escapeHtml(title)}</h3>
      <ul>${items.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
    </section>`;
}

function renderResult(data, image) {
  const { prediction: p } = data;
  const info = p.info;
  const status = statusOf(data);
  const badge = status === "uncertain" ? "Not sure" : p.healthy ? "No disease found" : TYPE_LABELS[info.type] || "Disease detected";

  const facts = p.healthy
    ? factBlock("Keep it healthy", info.prevention, true)
    : factBlock("What to look for", info.symptoms) +
      factBlock("What to do now", info.treatment, true) +
      factBlock("Prevent it next time", info.prevention);

  const about = `
    <div>
      <p class="summary">${escapeHtml(info.summary)}</p>
      ${info.cause ? `<p class="cause">Cause: ${escapeHtml(info.cause)}</p>` : ""}
    </div>
    <div class="facts">${facts}</div>`;

  const notice = data.uncertain
    ? `<div class="notice">
        <strong>Not confident about this photo</strong>
        The closest match is only ${percent(p.confidence)} likely. Retake it with one leaf filling the square, in even
        light, against a plain background. If this plant isn't in the supported list below, the result won't be meaningful.
      </div>
      <details class="details">
        <summary>Show details for the closest match</summary>
        <div>${about}</div>
      </details>`
    : about;

  const matches = data.top
    .map(
      (m) => `<li class="match">
        <span class="match-name">${escapeHtml(m.name)} <span>· ${escapeHtml(m.plant)}</span></span>
        <span class="match-pct">${percent(m.confidence)}</span>
        <span class="match-bar"><i style="width:${(m.confidence * 100).toFixed(1)}%"></i></span>
      </li>`
    )
    .join("");

  els.resultBody.innerHTML = `
    <div class="result-head">
      <img class="result-thumb" src="${image}" alt="The leaf that was analysed">
      <div>
        <span class="badge is-${status}"><span class="dot"></span>${badge}</span>
        <h2 class="result-title">${escapeHtml(p.name)}</h2>
        <p class="result-plant">${escapeHtml(p.plant)}</p>
      </div>
      <div class="ring is-${status}" style="--p:${(p.confidence * 100).toFixed(1)}">
        <div><strong>${percent(p.confidence)}</strong><small>confidence</small></div>
      </div>
    </div>
    ${notice}
    <div class="matches">
      <h3>Top matches</h3>
      <ol>${matches}</ol>
    </div>
    <div class="result-foot">
      <span class="result-meta">Analysed in ${Math.round(data.inference_ms)} ms</span>
      <button class="btn btn-ghost" type="button" data-action="scan-again">Scan another leaf</button>
    </div>`;
  showResultState("body");
}

function renderError(message) {
  els.resultBody.innerHTML = `
    <div class="error-box">
      <h2>Something went wrong</h2>
      <p>${escapeHtml(message)}</p>
      <button class="btn btn-ghost" type="button" data-action="scan-again">Try again</button>
    </div>`;
  showResultState("body");
}

async function analyse(canvas) {
  if (state.busy) return;
  state.busy = true;
  const image = canvas.toDataURL("image/jpeg", 0.9);
  els.loadingImg.src = image;
  showResultState("loading");
  if (isNarrow()) els.result.scrollIntoView({ behavior: "smooth", block: "start" });

  try {
    const data = await classify(await canvasToBlob(canvas));
    renderResult(data, image);
    addToHistory(data, squareCanvas(canvas, canvas.width, canvas.height, THUMB_SIZE).toDataURL("image/jpeg", 0.8));
  } catch (err) {
    renderError(err.message);
  } finally {
    state.busy = false;
  }
}

function scanAgain() {
  if (isNarrow()) document.querySelector(".scanner").scrollIntoView({ behavior: "smooth", block: "start" });
  if (!els.panelUpload.hidden) els.fileInput.click();
  else if (!state.stream) startCamera();
  else retake();
}

els.result.addEventListener("click", (event) => {
  if (event.target.closest("[data-action='scan-again']")) scanAgain();
});

// ------------------------------------------------------------------- camera

function setOverlay(title, text, { start = true, native = false } = {}) {
  els.overlay.hidden = false;
  els.camTitle.textContent = title;
  els.camText.textContent = text;
  els.startCamera.hidden = !start;
  els.nativeCamera.hidden = !native;
}

function resetOverlay() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setOverlay(
      "Live camera needs a secure connection",
      "Browsers only allow the live camera on https:// or localhost. You can still take a picture with your phone's camera app.",
      { start: false, native: true }
    );
  } else {
    setOverlay("Take a picture of a leaf", "Open the camera, fill the square with a single leaf, then press Take picture.");
  }
}

async function cameraCount() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices.filter((d) => d.kind === "videoinput").length;
  } catch {
    return 1;
  }
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) return resetOverlay();
  stopCamera();
  setOverlay("Starting camera…", "Allow camera access when your browser asks.", { start: false });

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: { ideal: state.facing }, width: { ideal: 1280 }, height: { ideal: 1280 } },
    });
    state.stream = stream;
    els.video.srcObject = stream;
    await els.video.play();

    // Laptop webcams usually don't report a facing mode; mirror them so moving the leaf feels natural.
    const facing = stream.getVideoTracks()[0].getSettings().facingMode || "user";
    els.video.classList.toggle("mirrored", facing === "user");

    els.overlay.hidden = true;
    els.closeCamera.hidden = false;
    els.shutter.disabled = false;
    els.liveToggle.disabled = false;
    els.flip.disabled = (await cameraCount()) < 2;
  } catch (err) {
    stopCamera();
    const blocked = err.name === "NotAllowedError" || err.name === "SecurityError";
    setOverlay(
      blocked ? "Camera access was blocked" : "Couldn't start the camera",
      blocked
        ? "Allow camera access in your browser's site settings, or take a photo instead."
        : "No camera was found, or another app is using it.",
      { start: true, native: true }
    );
  }
}

function stopCamera() {
  setLive(false);
  state.stream?.getTracks().forEach((track) => track.stop());
  state.stream = null;
  els.video.srcObject = null;
  setFrozen(false);
  els.closeCamera.hidden = true;
  els.shutter.disabled = true;
  els.liveToggle.disabled = true;
  els.flip.disabled = true;
}

function grabFrame() {
  const { video } = els;
  if (!video.videoWidth) return null;
  return squareCanvas(video, video.videoWidth, video.videoHeight);
}

/** Show the picture just taken in place of the live feed, with a Retake button. */
function setFrozen(frozen) {
  state.frozen = frozen;
  if (frozen) setLive(false);
  els.snapshot.hidden = !frozen;
  els.viewfinder.classList.toggle("is-frozen", frozen);
  els.shutter.hidden = frozen;
  els.shutterLabel.hidden = frozen;
  els.retake.hidden = !frozen;
  els.liveToggle.disabled = frozen || !state.stream;
}

function retake() {
  if (!state.stream) return startCamera();
  setFrozen(false);
  els.shutter.focus({ preventScroll: true });
}

function openCamera() {
  selectTab("camera");
  if (!state.stream) startCamera();
  else if (state.frozen) retake();
}

els.startCamera.addEventListener("click", startCamera);
els.nativeCamera.addEventListener("click", () => els.nativeInput.click());
els.altCapture.addEventListener("click", () => els.nativeInput.click());
els.heroCamera.addEventListener("click", openCamera);
els.heroUpload.addEventListener("click", () => selectTab("upload"));
els.retake.addEventListener("click", retake);
els.closeCamera.addEventListener("click", () => {
  stopCamera();
  resetOverlay();
});

els.flip.addEventListener("click", () => {
  state.facing = state.facing === "environment" ? "user" : "environment";
  startCamera();
});

els.shutter.addEventListener("click", () => {
  const frame = grabFrame();
  if (!frame || state.busy) return;
  els.viewfinder.classList.remove("flash");
  void els.viewfinder.offsetWidth; // restart the flash animation
  els.viewfinder.classList.add("flash");
  els.snapshot.src = frame.toDataURL("image/jpeg", 0.9);
  els.snapshot.classList.toggle("mirrored", els.video.classList.contains("mirrored"));
  setFrozen(true);
  analyse(frame);
});

document.addEventListener("visibilitychange", () => {
  if (document.hidden && state.stream) {
    stopCamera();
    resetOverlay();
  }
});

// ---------------------------------------------------------------- live mode

function showPill(html, status) {
  els.livePill.className = `live-pill is-${status}`;
  els.livePill.innerHTML = `<span class="dot"></span><span>${html}</span>`;
}

function setLive(on) {
  state.live = on && Boolean(state.stream);
  els.liveToggle.checked = state.live;
  els.livePill.hidden = !state.live;
  clearTimeout(state.liveTimer);
  if (state.live) {
    showPill("Looking for a leaf…", "uncertain");
    liveTick();
  }
}

async function liveTick() {
  if (!state.live) return;
  const frame = state.busy ? null : grabFrame(); // don't compete with a full capture
  if (frame) {
    try {
      const data = await classify(await canvasToBlob(frame));
      if (!state.live) return;
      const p = data.prediction;
      if (data.uncertain) showPill("Hold steady on a single leaf…", "uncertain");
      else showPill(`<strong>${escapeHtml(p.name)}</strong> · ${escapeHtml(p.plant)} · ${percent(p.confidence)}`, statusOf(data));
    } catch (err) {
      if (state.live) showPill(escapeHtml(err.message), "uncertain");
    }
  }
  if (state.live) state.liveTimer = setTimeout(liveTick, LIVE_INTERVAL_MS);
}

els.liveToggle.addEventListener("change", () => setLive(els.liveToggle.checked));

// ------------------------------------------------------------------ uploads

async function handleFile(file) {
  if (!file) return;
  if (file.type && !file.type.startsWith("image/")) {
    renderError("That file isn't an image. Choose a JPG or PNG photo of a leaf.");
    return;
  }
  try {
    const img = await loadImage(file);
    await analyse(squareCanvas(img, img.naturalWidth, img.naturalHeight));
  } catch (err) {
    renderError(err.message);
  }
}

for (const input of [els.fileInput, els.nativeInput]) {
  input.addEventListener("change", () => {
    handleFile(input.files[0]);
    input.value = ""; // allow picking the same file again
  });
}

["dragenter", "dragover"].forEach((type) =>
  els.dropzone.addEventListener(type, (event) => {
    event.preventDefault();
    els.dropzone.classList.add("is-dragging");
  })
);
["dragleave", "drop"].forEach((type) =>
  els.dropzone.addEventListener(type, () => els.dropzone.classList.remove("is-dragging"))
);
els.dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  handleFile(event.dataTransfer.files[0]);
});

document.addEventListener("paste", (event) => {
  const item = [...(event.clipboardData?.items || [])].find((i) => i.kind === "file" && i.type.startsWith("image/"));
  if (item) handleFile(item.getAsFile());
});

// --------------------------------------------------------------------- tabs

function selectTab(name) {
  for (const tab of [els.tabCamera, els.tabUpload]) {
    const selected = tab.id === `tab-${name}`;
    tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;
  }
  els.panelCamera.hidden = name !== "camera";
  els.panelUpload.hidden = name !== "upload";
  if (name !== "camera") {
    stopCamera();
    resetOverlay();
  }
}

els.tabCamera.addEventListener("click", openCamera);
els.tabUpload.addEventListener("click", () => selectTab("upload"));
document.querySelector(".tabs").addEventListener("keydown", (event) => {
  if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
  const next = els.tabCamera.getAttribute("aria-selected") === "true" ? els.tabUpload : els.tabCamera;
  selectTab(next.id.replace("tab-", ""));
  next.focus();
});

// ------------------------------------------------------------------ history

function loadHistory() {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY)) || [];
  } catch {
    return [];
  }
}

function storeHistory(items) {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(items));
    return true;
  } catch {
    return false;
  }
}

function addToHistory(data, thumb) {
  const items = [{ at: Date.now(), thumb, data }, ...loadHistory()].slice(0, HISTORY_LIMIT);
  // If storage is full, drop the oldest scans until it fits.
  while (items.length && !storeHistory(items)) items.pop();
  renderHistory();
}

const timeFormat = new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });

function renderHistory() {
  const items = loadHistory();
  els.historyGrid.replaceChildren();
  els.historyEmpty.hidden = items.length > 0;
  els.clearHistory.hidden = items.length === 0;

  for (const item of items) {
    const node = els.historyTemplate.content.cloneNode(true);
    const p = item.data.prediction;
    const status = statusOf(item.data);
    node.querySelector("img").src = item.thumb;
    node.querySelector(".history-name").textContent = `${p.plant} · ${p.name}`;
    const meta = node.querySelector(".history-meta");
    meta.classList.add(`is-${status}`);
    meta.innerHTML = `<span class="dot"></span>${status === "uncertain" ? "Uncertain" : percent(p.confidence)} · ${timeFormat.format(item.at)}`;
    node.querySelector("button").addEventListener("click", () => {
      renderResult(item.data, item.thumb);
      els.result.scrollIntoView({ behavior: "smooth", block: "start" });
    });
    els.historyGrid.append(node);
  }
}

els.clearHistory.addEventListener("click", () => {
  if (!confirm("Clear all saved scans from this device?")) return;
  try {
    localStorage.removeItem(HISTORY_KEY);
  } catch {
    /* storage unavailable: nothing to clear */
  }
  renderHistory();
});

// --------------------------------------------------------------------- init

// On phones, the native camera app is a reliable fallback even when the live camera works.
els.altCapture.hidden = !(navigator.mediaDevices?.getUserMedia && window.matchMedia("(pointer: coarse)").matches);
resetOverlay();
renderHistory();
