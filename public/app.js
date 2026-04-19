const form = document.getElementById("analyze-form");
const input = document.getElementById("image-input");
const uploadZone = document.getElementById("upload-zone");
const fileName = document.getElementById("file-name");
const uploadSubtitle = document.getElementById("upload-subtitle");
const promptInput = document.getElementById("prompt");
const modelSelect = document.getElementById("model-select");
const layoutPicker = document.getElementById("layout-picker");
const layoutDescription = document.getElementById("layout-description");
const previewWrap = document.getElementById("preview-wrap");
const previewImage = document.getElementById("preview-image");
const previewVideo = document.getElementById("preview-video");
const videoDurationWrap = document.getElementById("video-duration-wrap");
const videoDurationLabel = document.getElementById("video-duration-label");
const videoDurationSlider = document.getElementById("video-duration-slider");
const videoDurationValue = document.getElementById("video-duration-value");
const overlayCanvas = document.getElementById("overlay-canvas");
const overlayContext = overlayCanvas.getContext("2d");
const subjectBoxStatus = document.getElementById("subject-box-status");
const result = document.getElementById("result");
const explainButton = document.getElementById("submit-button");
const focusCutButton = document.getElementById("focus-cut-button");
const statusPill = document.getElementById("status-pill");
const generatedLink = document.getElementById("generated-link");
const outputVideoWrap = document.getElementById("output-video-wrap");
const outputVideo = document.getElementById("output-video");
const focusProgress = document.getElementById("focus-progress");
const focusProgressStage = document.getElementById("focus-progress-stage");
const focusProgressPercent = document.getElementById("focus-progress-percent");
const focusProgressFill = document.getElementById("focus-progress-fill");
const focusProgressEta = document.getElementById("focus-progress-eta");
const focusProgressDetail = document.getElementById("focus-progress-detail");
const resultMeta = document.getElementById("result-meta");

const EXPLAIN_BUTTON_TEXT = "Explain Media";
const FOCUS_CUT_BUTTON_TEXT = "Focus and Cut";
const FOCUS_POLL_INTERVAL_MS = 900;
const DEFAULT_LAYOUT_MODE = "split";
const DEFAULT_FOCUS_CUT_SOURCE_LABEL =
  "The_Future_Mark_Zuckerberg_first2m_source.mp4";

let selectedFile = null;
let previewUrl = null;
let mediaPipeReady = null;
let imageDetector = null;
let videoDetector = null;
let videoLoopHandle = null;
let currentSubjectHint = "";
let lastVideoTime = -1;
let mediaPipeUnavailable = false;
let activeModelId = "";
let modelSwitchInFlight = false;
let activeRequestStartedAt = 0;
let lastElapsedLabel = "";
let selectedProcessDurationSec = 0;
let inputVideoDurationSec = 0;
let selectedLayoutMode = DEFAULT_LAYOUT_MODE;

const LAYOUT_DESCRIPTIONS = {
  fill: "Fill: crop to cover the full 9:16 frame while keeping the main subject in view.",
  fit: "Fit: keep the full source frame visible inside 9:16 and use background treatment around it.",
  three: "Three: use a 3-panel composition for multi-person or multi-source clips with one clear primary panel.",
  four: "Four: use a 2x2 grid so each person or source keeps a visible region in frame.",
  split: "Split: divide the frame into two readable sections for speaker plus demo, gameplay, or comparison content.",
};

input.addEventListener("change", () => {
  const file = input.files?.[0];
  if (file && !isSupportedFile(file)) {
    result.textContent = "Please choose an image or video file.";
    setSelectedFile(null);
    return;
  }

  setSelectedFile(file || null);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  await runExplainRequest();
});

modelSelect.addEventListener("change", async () => {
  await activateSelectedModel();
});

layoutPicker.addEventListener("click", (event) => {
  const button = event.target.closest(".layout-option");
  if (!button) {
    return;
  }

  setSelectedLayoutMode(button.dataset.layout || DEFAULT_LAYOUT_MODE);
});

videoDurationSlider.addEventListener("input", () => {
  selectedProcessDurationSec = Number(videoDurationSlider.value || 0);
  renderVideoDurationUi();
});

focusCutButton.addEventListener("click", async () => {
  if (selectedFile && !isVideoFile(selectedFile)) {
    result.textContent = "Focus and Cut works on video uploads only.";
    return;
  }

  await runFocusCutRequest();
});

["dragenter", "dragover"].forEach((eventName) => {
  uploadZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    event.stopPropagation();
    uploadZone.classList.add("drag-active");
  });
});

["dragleave", "dragend", "drop"].forEach((eventName) => {
  uploadZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    event.stopPropagation();
    uploadZone.classList.remove("drag-active");
  });
});

uploadZone.addEventListener("drop", (event) => {
  const file = event.dataTransfer?.files?.[0];
  if (!file) {
    return;
  }

  if (!isSupportedFile(file)) {
    result.textContent = "Please drop an image or video file.";
    return;
  }

  setSelectedFile(file);
  syncInputFiles(file);
});

window.addEventListener("dragover", (event) => {
  event.preventDefault();
});

window.addEventListener("drop", (event) => {
  event.preventDefault();
});

refreshStatus();
const statusTimer = setInterval(refreshStatus, 3000);

async function refreshStatus() {
  try {
    const response = await fetch("/api/status");
    const data = await response.json();
    syncModelSelector(data);
    if (uploadSubtitle) {
      const maxUploadMb = Number(data?.maxUploadMb);
      if (Number.isFinite(maxUploadMb) && maxUploadMb > 0) {
        uploadSubtitle.textContent = `Images or short videos up to ${maxUploadMb} MB`;
      }
    }
    const missingModel = data.models?.find((model) => !model.modelExists);
    const missingMmproj = data.models?.find((model) => !model.mmprojExists);

    if (data.ok) {
      statusPill.textContent = data.activeModelName
        ? `Models ready (${data.activeModelName} active)`
        : "Models ready";
      statusPill.classList.add("ready");
      return;
    }

    if (missingModel) {
      statusPill.textContent = `Missing ${missingModel.name} model`;
      statusPill.classList.remove("ready");
      return;
    }

    if (missingMmproj) {
      statusPill.textContent = `Missing ${missingMmproj.name} mmproj`;
      statusPill.classList.remove("ready");
      return;
    }

    if (!data.ffmpegExists || !data.ffprobeExists) {
      statusPill.textContent = "Missing ffmpeg";
      statusPill.classList.remove("ready");
      return;
    }

    statusPill.textContent = "Starting backend";
    statusPill.classList.remove("ready");
  } catch {
    statusPill.textContent = "Status unavailable";
    statusPill.classList.remove("ready");
  }
}

async function runExplainRequest() {
  if (!selectedFile) {
    result.textContent = "Please choose an image or video first.";
    return;
  }

  const formData = new FormData();
  formData.append("media", selectedFile);
  formData.append("prompt", promptInput.value);
  if (activeModelId) {
    formData.append("modelId", activeModelId);
  }
  formData.append("layoutMode", selectedLayoutMode);
  if (isVideoFile(selectedFile) && selectedProcessDurationSec > 0) {
    formData.append("maxDurationSec", String(selectedProcessDurationSec));
  }

  activeRequestStartedAt = performance.now();
  setActionState("explain", true);
  hideGeneratedLink();
  hideOutputVideo();
  hideFocusProgress();
  clearResultMeta();
  result.textContent = isVideoFile(selectedFile)
    ? `Uploading video, sampling frames, and running ${getActiveModelName()}...`
    : `Uploading image and running ${getActiveModelName()}...`;

  try {
    const response = await fetch("/api/explain", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Request failed.");
    }

    lastElapsedLabel = `Explain completed in ${formatElapsedMs(performance.now() - activeRequestStartedAt)}`;
    result.textContent = `${lastElapsedLabel}\n\n${formatExplainResult(data)}`;
    showElapsedTime("Explain completed in", performance.now() - activeRequestStartedAt);
    await annotateSubject(data?.meta?.subjectHint || "");
  } catch (error) {
    result.textContent = `Error: ${error.message}`;
    clearResultMeta();
    setSubjectBoxStatus("Could not draw a subject box.", "warn");
  } finally {
    activeRequestStartedAt = 0;
    setActionState("explain", false);
  }
}

function formatExplainResult(data) {
  const analyses = data?.meta?.analyses || data?.analyses || [];
  const header =
    data?.meta?.type === "video" && data?.meta?.frameCount
      ? `[Video summary from ${data.meta.frameCount} sampled frames]\n\n`
      : "";

  if (!analyses.length) {
    return header + (data?.answer || "No model output.");
  }

  const blocks = analyses.map((analysis) => {
    const label = analysis.modelName || analysis.modelId || "Model";
    if (analysis.error) {
      return `[${label}]\nError: ${analysis.error}`;
    }
    return `[${label}]\n${analysis.answer || "No response."}`;
  });

  return `${header}${blocks.join("\n\n")}`;
}

async function runFocusCutRequest() {
  const formData = new FormData();
  if (selectedFile) {
    formData.append("media", selectedFile);
  }
  formData.append("subjectHint", currentSubjectHint);
  if (activeModelId) {
    formData.append("modelId", activeModelId);
  }
  formData.append("layoutMode", selectedLayoutMode);
  if (selectedProcessDurationSec > 0) {
    formData.append("maxDurationSec", String(selectedProcessDurationSec));
  }

  activeRequestStartedAt = performance.now();
  setActionState("focus-cut", true);
  hideGeneratedLink();
  hideOutputVideo();
  clearOverlay();
  showFocusProgress();
  clearResultMeta();
  updateFocusProgressUi({
    stage: "upload",
    message: selectedFile
      ? "Uploading video and creating render job."
      : `Using default source video (${DEFAULT_FOCUS_CUT_SOURCE_LABEL}) and creating render job.`,
    progress: 1,
    etaSeconds: null,
  });
  result.textContent = currentSubjectHint
    ? `Creating a split 9:16 output that follows "${currentSubjectHint}"${selectedFile ? "" : ` from ${DEFAULT_FOCUS_CUT_SOURCE_LABEL}`}...`
    : `Finding the main subject and creating a split 9:16 output${selectedFile ? "" : ` from ${DEFAULT_FOCUS_CUT_SOURCE_LABEL}`}...`;
  setSubjectBoxStatus(
    "Tracking subject and rendering split 9:16 output...",
    "",
  );

  try {
    const response = await fetch("/api/focus-cut", {
      method: "POST",
      body: formData,
    });

    const startData = await response.json();
    if (!response.ok) {
      throw new Error(startData.error || "Focus and Cut failed.");
    }

    const data = await waitForFocusCutJob(startData.jobId);
    currentSubjectHint = data.subjectHint || currentSubjectHint;
    showGeneratedLink(data.outputUrl, data.downloadName);
    showOutputVideo(data.outputUrl);
    previewRenderedVideo(data.outputUrl);
    updateFocusProgressUi({
      stage: "complete",
      message: `Focus cut complete in ${formatElapsedMs(performance.now() - activeRequestStartedAt)}.`,
      progress: 100,
      etaSeconds: 0,
    });
    lastElapsedLabel = `Focus cut completed in ${formatElapsedMs(performance.now() - activeRequestStartedAt)}`;
    showElapsedTime("Focus cut completed in", performance.now() - activeRequestStartedAt);
    result.textContent =
      `${lastElapsedLabel}\n\n` +
      `[Focus and cut ready]\n\n` +
      `Subject: ${data.subjectHint || data.renderSubject}\n` +
      `Layout: ${selectedLayoutMode}\n` +
      (selectedLayoutMode === "split"
        ? "Output: split 9:16 layout with subject focus and full-frame context."
        : "Output: 9:16 vertical crop centered on the main subject.");
    setSubjectBoxStatus(
      `Focused 9:16 video ready${data.subjectHint ? ` for "${data.subjectHint}"` : ""}.`,
      "ready",
    );
  } catch (error) {
    hideFocusProgress();
    result.textContent = `Error: ${error.message}`;
    clearResultMeta();
    setSubjectBoxStatus("Could not create the focused 9:16 crop.", "warn");
  } finally {
    activeRequestStartedAt = 0;
    setActionState("focus-cut", false);
  }
}

async function waitForFocusCutJob(jobId) {
  if (!jobId) {
    throw new Error("Server did not return a focus-cut job id.");
  }

  while (true) {
    const response = await fetch(`/api/focus-cut/jobs/${encodeURIComponent(jobId)}?t=${Date.now()}`);
    const job = await response.json();
    if (!response.ok) {
      throw new Error(job.error || "Could not fetch focus-cut job status.");
    }

    updateFocusProgressUi(job);

    if (job.status === "completed") {
      return job;
    }
    if (job.status === "failed") {
      throw new Error(job.error || "Focus and Cut failed.");
    }

    await delay(FOCUS_POLL_INTERVAL_MS);
  }
}

function showFocusProgress() {
  focusProgress.classList.remove("hidden");
}

function hideFocusProgress() {
  focusProgress.classList.add("hidden");
}

function updateFocusProgressUi(job) {
  showFocusProgress();
  const percent = Math.max(0, Math.min(100, Math.round(Number(job?.progress) || 0)));
  focusProgressFill.style.width = `${percent}%`;
  focusProgressPercent.textContent = `${percent}%`;
  focusProgressStage.textContent = formatFocusStage(job?.stage);
  focusProgressDetail.textContent = job?.message || "";

  if (typeof job?.etaSeconds === "number" && Number.isFinite(job.etaSeconds)) {
    focusProgressEta.textContent =
      job.etaSeconds <= 0
        ? "Done"
        : `${formatDuration(Math.round(job.etaSeconds))} left`;
  } else {
    focusProgressEta.textContent = percent >= 100 ? "Done" : "Estimating time left...";
  }
}

function formatFocusStage(stage) {
  const value = String(stage || "").toLowerCase();
  if (value === "queued" || value === "upload") {
    return "Queued";
  }
  if (value === "preparing") {
    return "Preparing";
  }
  if (value === "subject") {
    return "Detecting Subject";
  }
  if (value === "planning") {
    return "AI Layout Planning";
  }
  if (value === "rendering") {
    return "Rendering";
  }
  if (value === "muxing") {
    return "Finalizing";
  }
  if (value === "complete" || value === "completed") {
    return "Complete";
  }
  if (value === "failed") {
    return "Failed";
  }
  return "Processing";
}

function formatDuration(seconds) {
  const total = Math.max(0, Math.floor(seconds));
  const mins = Math.floor(total / 60);
  const secs = total % 60;
  if (mins > 0) {
    return `${mins}m ${secs}s`;
  }
  return `${secs}s`;
}

function delay(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

function setSelectedFile(file) {
  selectedFile = file;
  currentSubjectHint = "";
  inputVideoDurationSec = 0;
  selectedProcessDurationSec = 0;
  stopVideoDetection();
  hideGeneratedLink();
  hideOutputVideo();
  hideFocusProgress();
  clearOverlay();
  hideVideoDurationControls();

  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }

  if (!file) {
    previewWrap.classList.add("hidden");
    previewImage.removeAttribute("src");
    previewImage.classList.add("hidden");
    previewVideo.pause();
    previewVideo.removeAttribute("src");
    previewVideo.classList.add("hidden");
    overlayCanvas.classList.add("hidden");
    fileName.textContent = "";
    fileName.classList.add("hidden");
    subjectBoxStatus.textContent = "";
    subjectBoxStatus.className = "subject-box-status hidden";
    return;
  }

  previewUrl = URL.createObjectURL(file);
  previewWrap.classList.remove("hidden");
  fileName.textContent = file.name;
  fileName.classList.remove("hidden");

  if (isVideoFile(file)) {
    previewVideo.src = previewUrl;
    previewVideo.classList.remove("hidden");
    previewImage.removeAttribute("src");
    previewImage.classList.add("hidden");
    previewVideo.load();
    previewVideo.onloadedmetadata = () => {
      inputVideoDurationSec = Number(previewVideo.duration || 0);
      selectedProcessDurationSec = Math.max(1, Math.floor(inputVideoDurationSec || 0));
      configureVideoDurationControls();
    };
    previewVideo.onloadeddata = () => {
      syncOverlaySize(previewVideo);
    };
    return;
  }

  previewImage.src = previewUrl;
  previewImage.classList.remove("hidden");
  previewImage.onload = () => {
    syncOverlaySize(previewImage);
  };
  previewVideo.pause();
  previewVideo.removeAttribute("src");
  previewVideo.classList.add("hidden");
}

function syncInputFiles(file) {
  try {
    const dataTransfer = new DataTransfer();
    dataTransfer.items.add(file);
    input.files = dataTransfer.files;
  } catch {
    // Some browsers may block programmatic file assignment; selectedFile is still used for upload.
  }
}

function setSelectedLayoutMode(layoutMode) {
  const nextMode = DEFAULT_LAYOUT_MODE;
  selectedLayoutMode = nextMode;

  for (const option of layoutPicker.querySelectorAll(".layout-option")) {
    const isActive = option.dataset.layout === nextMode;
    option.classList.toggle("active", isActive);
    option.setAttribute("aria-pressed", isActive ? "true" : "false");
  }

  layoutDescription.textContent = LAYOUT_DESCRIPTIONS[nextMode];
}

setSelectedLayoutMode(DEFAULT_LAYOUT_MODE);

function configureVideoDurationControls() {
  if (!inputVideoDurationSec || !Number.isFinite(inputVideoDurationSec)) {
    hideVideoDurationControls();
    return;
  }

  const maxSeconds = Math.max(1, Math.floor(inputVideoDurationSec));
  videoDurationSlider.min = "1";
  videoDurationSlider.max = String(maxSeconds);
  videoDurationSlider.step = "1";
  videoDurationSlider.value = String(
    Math.min(maxSeconds, Math.max(1, Math.round(selectedProcessDurationSec || maxSeconds))),
  );
  selectedProcessDurationSec = Number(videoDurationSlider.value);
  videoDurationWrap.classList.remove("hidden");
  renderVideoDurationUi();
}

function hideVideoDurationControls() {
  videoDurationWrap.classList.add("hidden");
  videoDurationValue.textContent = "";
  videoDurationLabel.textContent = "Process duration";
}

function renderVideoDurationUi() {
  if (!inputVideoDurationSec || !Number.isFinite(inputVideoDurationSec)) {
    hideVideoDurationControls();
    return;
  }

  const totalSeconds = Math.max(1, Math.floor(inputVideoDurationSec));
  const chosenSeconds = Math.max(1, Math.min(totalSeconds, Math.round(selectedProcessDurationSec || totalSeconds)));
  selectedProcessDurationSec = chosenSeconds;
  videoDurationLabel.textContent = "Process first part of video";
  videoDurationValue.textContent = `Processing first ${formatDuration(chosenSeconds)} of ${formatDuration(totalSeconds)}`;
}

function isSupportedFile(file) {
  return isVideoFile(file) || isImageFile(file);
}

function isImageFile(file) {
  return file.type.startsWith("image/") || /\.(png|jpg|jpeg|webp|gif|bmp)$/i.test(file.name);
}

function isVideoFile(file) {
  return file.type.startsWith("video/") || /\.(mp4|mov|m4v|webm|avi)$/i.test(file.name);
}

async function annotateSubject(subjectHint) {
  currentSubjectHint = subjectHint.trim();

  if (!selectedFile) {
    return;
  }

  if (!currentSubjectHint) {
    clearOverlay();
    setSubjectBoxStatus("No clear subject hint returned by the model.", "warn");
    return;
  }

  setSubjectBoxStatus(`Finding subject box for "${currentSubjectHint}"...`, "");

  try {
    await ensureMediaPipe();
  } catch (error) {
    clearOverlay();
    setSubjectBoxStatus(`MediaPipe failed to load: ${error.message}`, "warn");
    return;
  }

  if (isVideoFile(selectedFile)) {
    await startVideoDetection();
    return;
  }

  const detector = await getImageDetector();
  const detections = detector.detect(previewImage).detections || [];
  renderBestDetection(previewImage, detections, currentSubjectHint);
}

async function ensureMediaPipe() {
  if (mediaPipeUnavailable) {
    throw new Error("MediaPipe is unavailable in this browser session.");
  }

  const mediaPipeVision = window.vision;
  if (!mediaPipeVision?.FilesetResolver || !mediaPipeVision?.ObjectDetector) {
    mediaPipeUnavailable = true;
    throw new Error("MediaPipe script did not load.");
  }

  if (!mediaPipeReady) {
    mediaPipeReady = mediaPipeVision.FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm",
    );
  }

  return mediaPipeReady;
}

async function getImageDetector() {
  if (!imageDetector) {
    const vision = await ensureMediaPipe();
    imageDetector = await window.vision.ObjectDetector.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite",
      },
      scoreThreshold: 0.2,
      runningMode: "IMAGE",
      maxResults: 8,
    });
  }

  return imageDetector;
}

async function getVideoDetector() {
  if (!videoDetector) {
    const vision = await ensureMediaPipe();
    videoDetector = await window.vision.ObjectDetector.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite",
      },
      scoreThreshold: 0.2,
      runningMode: "VIDEO",
      maxResults: 8,
    });
  }

  return videoDetector;
}

async function startVideoDetection() {
  stopVideoDetection();
  const detector = await getVideoDetector();

  const tick = () => {
    if (!previewVideo || previewVideo.classList.contains("hidden")) {
      return;
    }

    if (previewVideo.readyState >= 2) {
      if (previewVideo.currentTime !== lastVideoTime) {
        lastVideoTime = previewVideo.currentTime;
        syncOverlaySize(previewVideo);
        const detections =
          detector.detectForVideo(previewVideo, performance.now()).detections || [];
        renderBestDetection(previewVideo, detections, currentSubjectHint);
      }
    }

    if (!previewVideo.paused && !previewVideo.ended) {
      videoLoopHandle = requestAnimationFrame(tick);
    }
  };

  previewVideo.onplay = () => {
    stopVideoDetection();
    videoLoopHandle = requestAnimationFrame(tick);
  };

  previewVideo.onpause = async () => {
    stopVideoDetection();
    if (previewVideo.readyState >= 2) {
      const detections =
        detector.detectForVideo(previewVideo, performance.now()).detections || [];
      renderBestDetection(previewVideo, detections, currentSubjectHint);
    }
  };

  previewVideo.onseeked = async () => {
    if (previewVideo.readyState >= 2) {
      const detections =
        detector.detectForVideo(previewVideo, performance.now()).detections || [];
      renderBestDetection(previewVideo, detections, currentSubjectHint);
    }
  };

  if (previewVideo.readyState >= 2) {
    const detections =
      detector.detectForVideo(previewVideo, performance.now()).detections || [];
    renderBestDetection(previewVideo, detections, currentSubjectHint);
  }
}

function stopVideoDetection() {
  if (videoLoopHandle) {
    cancelAnimationFrame(videoLoopHandle);
    videoLoopHandle = null;
  }
  previewVideo.onplay = null;
  previewVideo.onpause = null;
  previewVideo.onseeked = null;
  lastVideoTime = -1;
}

function setActionState(activeAction, isBusy) {
  explainButton.disabled = isBusy;
  focusCutButton.disabled = isBusy;
  modelSelect.disabled = isBusy || modelSwitchInFlight;
  explainButton.textContent =
    isBusy && activeAction === "explain" ? "Thinking..." : EXPLAIN_BUTTON_TEXT;
  focusCutButton.textContent =
    isBusy && activeAction === "focus-cut" ? "Focusing..." : FOCUS_CUT_BUTTON_TEXT;
}

function syncModelSelector(status) {
  const models = Array.isArray(status?.models) ? status.models : [];
  if (!models.length) {
    return;
  }

  const optionsChanged =
    modelSelect.options.length !== models.length ||
    models.some((model, index) => modelSelect.options[index]?.value !== model.id);

  if (optionsChanged) {
    modelSelect.innerHTML = "";
    for (const model of models) {
      const option = document.createElement("option");
      option.value = model.id;
      option.textContent =
        model.modelExists && model.mmprojExists
          ? model.name
          : `${model.name} (missing files)`;
      option.disabled = !(model.modelExists && model.mmprojExists);
      modelSelect.appendChild(option);
    }
  }

  const reportedActiveId = String(status?.activeModelId || "").trim();
  const fallbackModelId = models.find((model) => model.modelExists && model.mmprojExists)?.id || models[0].id;
  if (!modelSwitchInFlight) {
    activeModelId = reportedActiveId || fallbackModelId;
    if (activeModelId) {
      modelSelect.value = activeModelId;
    }
  }
}

function getActiveModelName() {
  const selected = modelSelect.options[modelSelect.selectedIndex];
  return selected?.textContent?.replace(/\s+\(missing files\)$/, "") || "selected model";
}

async function activateSelectedModel() {
  const targetModelId = String(modelSelect.value || "").trim();
  if (!targetModelId || targetModelId === activeModelId) {
    return;
  }

  modelSwitchInFlight = true;
  setActionState("model", true);
  statusPill.textContent = `Switching to ${getActiveModelName()}...`;
  statusPill.classList.remove("ready");
  result.textContent = `Switching model to ${getActiveModelName()}...`;

  try {
    const response = await fetch("/api/activate-model", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        modelId: targetModelId,
      }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Failed to activate model.");
    }
    activeModelId = data.activeModelId || targetModelId;
    statusPill.textContent = `Models ready (${data.activeModelName || getActiveModelName()} active)`;
    statusPill.classList.add("ready");
    result.textContent = `${data.activeModelName || getActiveModelName()} loaded. Previous model was unloaded from RAM.`;
  } catch (error) {
    result.textContent = `Error: ${error.message}`;
    await refreshStatus();
  } finally {
    modelSwitchInFlight = false;
    setActionState("model", false);
  }
}

function showGeneratedLink(url, downloadName) {
  generatedLink.href = `${url}${url.includes("?") ? "&" : "?"}t=${Date.now()}`;
  generatedLink.download = downloadName || "";
  generatedLink.classList.remove("hidden");
}

function hideGeneratedLink() {
  generatedLink.removeAttribute("href");
  generatedLink.removeAttribute("download");
  generatedLink.classList.add("hidden");
}

function showOutputVideo(url) {
  if (!outputVideoWrap || !outputVideo) {
    return;
  }

  const resolvedUrl = `${url}${url.includes("?") ? "&" : "?"}t=${Date.now()}`;
  outputVideo.pause();
  outputVideo.src = resolvedUrl;
  outputVideoWrap.classList.remove("hidden");
  outputVideo.load();
}

function hideOutputVideo() {
  if (!outputVideoWrap || !outputVideo) {
    return;
  }

  outputVideo.pause();
  outputVideo.removeAttribute("src");
  outputVideo.load();
  outputVideoWrap.classList.add("hidden");
}

function showElapsedTime(prefix, elapsedMs) {
  if (!resultMeta) {
    return;
  }
  resultMeta.textContent = `${prefix} ${formatElapsedMs(elapsedMs)}`;
  resultMeta.classList.remove("hidden");
}

function clearResultMeta() {
  if (!resultMeta) {
    return;
  }
  resultMeta.textContent = "";
  resultMeta.classList.add("hidden");
}

function formatElapsedMs(elapsedMs) {
  const totalMs = Math.max(0, Math.round(Number(elapsedMs) || 0));
  if (totalMs < 1000) {
    return `${totalMs} ms`;
  }

  const totalSeconds = totalMs / 1000;
  if (totalSeconds < 60) {
    return `${totalSeconds.toFixed(2)} s`;
  }

  const mins = Math.floor(totalSeconds / 60);
  const secs = totalSeconds - mins * 60;
  return `${mins}m ${secs.toFixed(1)}s`;
}

function previewRenderedVideo(url) {
  stopVideoDetection();
  clearOverlay();
  previewWrap.classList.remove("hidden");
  previewImage.removeAttribute("src");
  previewImage.classList.add("hidden");
  previewVideo.pause();
  previewVideo.src = `${url}${url.includes("?") ? "&" : "?"}t=${Date.now()}`;
  previewVideo.classList.remove("hidden");
  previewVideo.onloadeddata = () => {
    syncOverlaySize(previewVideo);
    overlayCanvas.classList.add("hidden");
  };
  previewVideo.load();
}

function renderBestDetection(element, detections, subjectHint) {
  syncOverlaySize(element);

  const best = pickBestDetection(detections, subjectHint, element);
  if (!best) {
    clearOverlay();
    setSubjectBoxStatus(`No matching MediaPipe box found for "${subjectHint}".`, "warn");
    return;
  }

  drawDetection(best, element);
  setSubjectBoxStatus(
    `Main subject: ${subjectHint} -> box on ${best.categories?.[0]?.categoryName || "detected object"}`,
    "ready",
  );
}

function pickBestDetection(detections, subjectHint, element) {
  if (!detections.length) {
    return null;
  }

  const normalizedHint = normalizeWords(subjectHint);
  const width = element.videoWidth || element.naturalWidth || element.clientWidth || 1;
  const height = element.videoHeight || element.naturalHeight || element.clientHeight || 1;

  let best = null;
  let bestScore = -Infinity;

  for (const detection of detections) {
    const bbox = detection.boundingBox;
    if (!bbox) {
      continue;
    }

    const labels = (detection.categories || [])
      .map((category) => normalizeWords(category.categoryName || ""))
      .flat();
    const matchScore = scoreLabelMatch(normalizedHint, labels);
    const confidence = detection.categories?.[0]?.score || 0;
    const area = (bbox.width * bbox.height) / (width * height);
    const centerX = (bbox.originX + bbox.width / 2) / width;
    const centerY = (bbox.originY + bbox.height / 2) / height;
    const centerDistance = Math.hypot(centerX - 0.5, centerY - 0.5);
    const score = matchScore * 100 + confidence * 10 + area * 8 - centerDistance * 3;

    if (score > bestScore) {
      bestScore = score;
      best = detection;
    }
  }

  return best;
}

function scoreLabelMatch(subjectWords, labels) {
  if (!subjectWords.length || !labels.length) {
    return 0;
  }

  const aliases = new Map([
    ["person", ["person", "man", "woman", "boy", "girl", "human", "people"]],
    ["dog", ["dog", "puppy"]],
    ["cat", ["cat", "kitten"]],
    ["car", ["car", "vehicle", "truck", "bus"]],
    ["bicycle", ["bicycle", "bike"]],
    ["motorcycle", ["motorcycle", "bike"]],
    ["bird", ["bird"]],
    ["horse", ["horse"]],
    ["boat", ["boat", "ship"]],
    ["tv", ["tv", "monitor", "screen"]],
  ]);

  let score = 0;
  for (const word of subjectWords) {
    const group = aliases.get(word) || [word];
    for (const label of labels) {
      if (group.includes(label) || label.includes(word) || word.includes(label)) {
        score += 1;
      }
    }
  }

  return score;
}

function normalizeWords(text) {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, " ")
    .split(/\s+/)
    .filter(Boolean);
}

function syncOverlaySize(element) {
  const width = element.clientWidth;
  const height = element.clientHeight;
  if (!width || !height) {
    return;
  }

  overlayCanvas.width = width;
  overlayCanvas.height = height;
  overlayCanvas.classList.remove("hidden");
}

function drawDetection(detection, element) {
  const bbox = detection.boundingBox;
  if (!bbox) {
    return;
  }

  clearOverlay();
  overlayCanvas.classList.remove("hidden");

  const sourceWidth = element.videoWidth || element.naturalWidth || element.clientWidth;
  const sourceHeight = element.videoHeight || element.naturalHeight || element.clientHeight;
  const scaleX = overlayCanvas.width / sourceWidth;
  const scaleY = overlayCanvas.height / sourceHeight;

  const x = bbox.originX * scaleX;
  const y = bbox.originY * scaleY;
  const width = bbox.width * scaleX;
  const height = bbox.height * scaleY;
  const label = detection.categories?.[0]?.categoryName || "subject";

  overlayContext.lineWidth = 4;
  overlayContext.strokeStyle = "#ff7b42";
  overlayContext.fillStyle = "rgba(255, 123, 66, 0.16)";
  overlayContext.strokeRect(x, y, width, height);
  overlayContext.fillRect(x, y, width, height);

  overlayContext.font = "600 14px 'IBM Plex Mono'";
  const text = `${label}`;
  const textWidth = overlayContext.measureText(text).width;
  const chipHeight = 24;
  overlayContext.fillStyle = "#1f241f";
  overlayContext.fillRect(x, Math.max(0, y - chipHeight), textWidth + 16, chipHeight);
  overlayContext.fillStyle = "#f6f1e8";
  overlayContext.fillText(text, x + 8, Math.max(16, y - 8));
}

function clearOverlay() {
  overlayContext.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  overlayCanvas.classList.add("hidden");
}

function setSubjectBoxStatus(text, variant) {
  subjectBoxStatus.textContent = text;
  subjectBoxStatus.className = "subject-box-status";
  if (!text) {
    subjectBoxStatus.classList.add("hidden");
    return;
  }
  if (variant) {
    subjectBoxStatus.classList.add(variant);
  }
}
