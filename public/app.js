const form = document.getElementById("analyze-form");
const input = document.getElementById("image-input");
const uploadZone = document.getElementById("upload-zone");
const fileName = document.getElementById("file-name");
const promptInput = document.getElementById("prompt");
const previewWrap = document.getElementById("preview-wrap");
const previewImage = document.getElementById("preview-image");
const previewVideo = document.getElementById("preview-video");
const overlayCanvas = document.getElementById("overlay-canvas");
const overlayContext = overlayCanvas.getContext("2d");
const subjectBoxStatus = document.getElementById("subject-box-status");
const result = document.getElementById("result");
const explainButton = document.getElementById("submit-button");
const focusCutButton = document.getElementById("focus-cut-button");
const statusPill = document.getElementById("status-pill");
const generatedLink = document.getElementById("generated-link");

const EXPLAIN_BUTTON_TEXT = "Explain Media";
const FOCUS_CUT_BUTTON_TEXT = "Focus and Cut";

let selectedFile = null;
let previewUrl = null;
let mediaPipeReady = null;
let imageDetector = null;
let videoDetector = null;
let videoLoopHandle = null;
let currentSubjectHint = "";
let lastVideoTime = -1;
let mediaPipeUnavailable = false;

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

focusCutButton.addEventListener("click", async () => {
  if (!selectedFile) {
    result.textContent = "Please choose a video first.";
    return;
  }

  if (!isVideoFile(selectedFile)) {
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

    if (data.ok) {
      statusPill.textContent = "Model ready";
      statusPill.classList.add("ready");
      clearInterval(statusTimer);
      return;
    }

    if (!data.mmprojExists) {
      statusPill.textContent = "Missing mmproj";
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

  setActionState("explain", true);
  hideGeneratedLink();
  result.textContent = isVideoFile(selectedFile)
    ? "Uploading video, sampling frames, and asking the model..."
    : "Uploading image and asking the model...";

  try {
    const response = await fetch("/api/explain", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Request failed.");
    }

    if (data?.meta?.type === "video" && data?.meta?.frameCount) {
      result.textContent = `[Video summary from ${data.meta.frameCount} sampled frames]\n\n${data.answer}`;
      await annotateSubject(data.meta.subjectHint || "");
      return;
    }

    result.textContent = data.answer;
    await annotateSubject(data?.meta?.subjectHint || "");
  } catch (error) {
    result.textContent = `Error: ${error.message}`;
    setSubjectBoxStatus("Could not draw a subject box.", "warn");
  } finally {
    setActionState("explain", false);
  }
}

async function runFocusCutRequest() {
  const formData = new FormData();
  formData.append("media", selectedFile);
  formData.append("subjectHint", currentSubjectHint);

  setActionState("focus-cut", true);
  hideGeneratedLink();
  clearOverlay();
  result.textContent = currentSubjectHint
    ? `Creating a 9:16 crop that follows "${currentSubjectHint}"...`
    : "Finding the main subject and creating a 9:16 vertical crop...";
  setSubjectBoxStatus("Tracking the subject and rendering a vertical clip...", "");

  try {
    const response = await fetch("/api/focus-cut", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Focus and Cut failed.");
    }

    currentSubjectHint = data.subjectHint || currentSubjectHint;
    showGeneratedLink(data.outputUrl, data.downloadName);
    previewRenderedVideo(data.outputUrl);
    result.textContent =
      `[Focus and cut ready]\n\n` +
      `Subject: ${data.subjectHint || data.renderSubject}\n` +
      "Output: 9:16 vertical crop centered on the main subject.";
    setSubjectBoxStatus(
      `Focused 9:16 video ready${data.subjectHint ? ` for "${data.subjectHint}"` : ""}.`,
      "ready",
    );
  } catch (error) {
    result.textContent = `Error: ${error.message}`;
    setSubjectBoxStatus("Could not create the focused 9:16 crop.", "warn");
  } finally {
    setActionState("focus-cut", false);
  }
}

function setSelectedFile(file) {
  selectedFile = file;
  currentSubjectHint = "";
  stopVideoDetection();
  hideGeneratedLink();
  clearOverlay();

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
  explainButton.textContent =
    isBusy && activeAction === "explain" ? "Thinking..." : EXPLAIN_BUTTON_TEXT;
  focusCutButton.textContent =
    isBusy && activeAction === "focus-cut" ? "Focusing..." : FOCUS_CUT_BUTTON_TEXT;
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
