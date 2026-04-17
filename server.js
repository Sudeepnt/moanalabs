const express = require("express");
const multer = require("multer");
const { spawn } = require("child_process");
const path = require("path");
const fs = require("fs");
const os = require("os");
const fsp = require("fs/promises");

const APP_PORT = Number(process.env.APP_PORT || 3055);
const LLAMA_PORT = Number(process.env.LLAMA_PORT || 32111);
const LLAMA_HOST = process.env.LLAMA_HOST || "127.0.0.1";
const LLAMA_BIN =
  process.env.LLAMA_BIN || "/Users/sudeepnt/llama.cpp/build/bin/llama-server";
const FFMPEG_BIN = process.env.FFMPEG_BIN || "/opt/homebrew/bin/ffmpeg";
const FFPROBE_BIN = process.env.FFPROBE_BIN || "/opt/homebrew/bin/ffprobe";
const PYTHON_BIN = process.env.PYTHON_BIN || "python3";
const INITIAL_CTX_SIZE = Number(process.env.CTX_SIZE || 16384);
const MAX_CTX_SIZE = Number(process.env.MAX_CTX_SIZE || 65536);
const GPU_LAYERS = process.env.GPU_LAYERS || "999";
const LLAMA_BASE_URL = `http://${LLAMA_HOST}:${LLAMA_PORT}`;
const MAX_VIDEO_FRAMES = Number(process.env.MAX_VIDEO_FRAMES || 6);
const MAX_UPLOAD_MB = Math.max(1, Number(process.env.MAX_UPLOAD_MB || 1024));
const MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024;
const DETECTOR_MODEL_PATH =
  process.env.DETECTOR_MODEL_PATH || path.join(__dirname, "models", "efficientdet_lite0.tflite");
const GENERATED_DIR = path.join(__dirname, "generated");
function resolveFirstExistingPath(candidates) {
  for (const candidate of candidates) {
    if (candidate && fs.existsSync(candidate)) {
      return candidate;
    }
  }
  return candidates.find(Boolean);
}
const MODEL_CONFIGS = {
  qwen: {
    id: "qwen",
    name: "Qwen3VL",
    modelPath:
      process.env.QWEN_MODEL_PATH ||
      process.env.MODEL_PATH ||
      resolveFirstExistingPath([
        "/Users/sudeepnt/Desktop/New Folder With Items 2/DMain/AI Model Stuff/Qwen3VL-4B-Instruct-Q4_K_M.gguf",
        "/Users/sudeepnt/Desktop/DMain/AI Model Stuff/Qwen3VL-4B-Instruct-Q4_K_M.gguf",
      ]),
    mmprojPath:
      process.env.QWEN_MMPROJ_PATH ||
      process.env.MMPROJ_PATH ||
      resolveFirstExistingPath([
        "/Users/sudeepnt/Desktop/New Folder With Items 2/DMain/AI Model Stuff/mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf",
        "/Users/sudeepnt/Desktop/DMain/AI Model Stuff/mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf",
      ]),
  },
  lfm: {
    id: "lfm",
    name: "LFM2.5-VL",
    modelPath:
      process.env.LFM_MODEL_PATH ||
      resolveFirstExistingPath([
        "/Users/sudeepnt/Desktop/New Folder With Items 2/DMain/AI Model Stuff/LFM2.5-VL-1.6B-Q4_0.gguf",
        "/Users/sudeepnt/Desktop/DMain/AI Model Stuff/LFM2.5-VL-1.6B-Q4_0.gguf",
      ]),
    mmprojPath:
      process.env.LFM_MMPROJ_PATH ||
      resolveFirstExistingPath([
        "/Users/sudeepnt/Desktop/New Folder With Items 2/DMain/AI Model Stuff/mmproj-LFM2.5-VL-1.6b-Q8_0.gguf",
        "/Users/sudeepnt/Desktop/DMain/AI Model Stuff/mmproj-LFM2.5-VL-1.6b-Q8_0.gguf",
      ]),
  },
};
const MODEL_ORDER = ["qwen", "lfm"];
const DEFAULT_MODEL_ID = MODEL_ORDER[0];

const app = express();
const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: MAX_UPLOAD_BYTES,
  },
});

let llamaChild = null;
let llamaStartPromise = null;
let activeModelId = null;
const currentCtxSizes = Object.fromEntries(MODEL_ORDER.map((modelId) => [modelId, INITIAL_CTX_SIZE]));
const focusCutJobs = new Map();
const FOCUS_JOB_MAX_AGE_MS = 1000 * 60 * 45;
const FOCUS_PROGRESS_PREFIX = "__PROGRESS__";

fs.mkdirSync(GENERATED_DIR, { recursive: true });
app.use(express.static(path.join(__dirname, "public")));
app.use("/generated", express.static(GENERATED_DIR));

app.get("/api/status", async (req, res) => {
  try {
    const status = await getStatus();
    res.json(status);
  } catch (error) {
    res.status(500).json({
      ok: false,
      error: error.message,
    });
  }
});

app.post("/api/activate-model", express.json(), async (req, res) => {
  try {
    const modelId = String(req.body?.modelId || "").trim().toLowerCase();
    if (!MODEL_CONFIGS[modelId]) {
      return res.status(400).json({ error: `Unknown model id: ${modelId}` });
    }

    await ensureLlamaServer(modelId);
    res.json({
      ok: true,
      activeModelId,
      activeModelName: getModelConfig(activeModelId).name,
    });
  } catch (error) {
    res.status(500).json({
      error: error.message,
    });
  }
});

app.post("/api/explain", upload.single("media"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "Please upload an image or video." });
    }

    const requestedModelId = resolveRequestedModelId(req.body?.modelId);
    await ensureLlamaServer(requestedModelId);

    if (isVideoMimeType(req.file.mimetype, req.file.originalname)) {
      const prompt =
        (req.body.prompt || "").trim() ||
        "These are sampled frames from one video in chronological order. Explain what happens across the video, mention the key objects or people, and include any readable on-screen text.";

      const { analyses, frameCount, subjectHint } = await analyzeVideo(req.file, prompt, requestedModelId);
      return res.json({
        answer: formatCombinedAnswer(analyses),
        analyses,
        meta: {
          type: "video",
          frameCount,
          subjectHint,
          activeModelId: requestedModelId,
          activeModelName: getModelConfig(requestedModelId).name,
          analyses,
        },
      });
    }

    if (!isImageMimeType(req.file.mimetype, req.file.originalname)) {
      return res.status(400).json({
        error: "Unsupported file type. Please upload an image or video.",
      });
    }

    const prompt =
      (req.body.prompt || "").trim() ||
      "Explain this image clearly. Mention the main objects, any visible text, and what is likely happening.";

    const contentParts = [
      {
        type: "image_url",
        image_url: {
          url: bufferToDataUrl(req.file),
        },
      },
    ];
    const analyses = await runSequentialModelAnalyses(
      contentParts,
      prompt,
      "Identify the single main subject in this image that could fit inside one bounding box. Only return a person, animal, vehicle, or everyday object. Do not return terrain, sky, water, mountains, buildings, or broad scenery. If no single boxable subject stands out, return none. Return only 1 to 4 lowercase words, no punctuation.",
      [requestedModelId],
    );
    const subjectHint = choosePreferredSubjectHint(analyses);

    res.json({
      answer: formatCombinedAnswer(analyses),
      analyses,
      meta: {
        type: "image",
        subjectHint,
        activeModelId: requestedModelId,
        activeModelName: getModelConfig(requestedModelId).name,
        analyses,
      },
    });
  } catch (error) {
    res.status(500).json({
      error: error.message,
    });
  }
});

app.post("/api/focus-cut", upload.single("media"), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "Please upload a video first." });
  }

  if (!isVideoMimeType(req.file.mimetype, req.file.originalname)) {
    return res.status(400).json({
      error: "Focus and Cut works on video uploads only.",
    });
  }

  const job = createFocusCutJob({
    originalName: req.file.originalname || "video.mp4",
  });
  const initialSubjectHint = normalizeSubjectHint((req.body.subjectHint || "").trim());
  const requestedModelId = resolveRequestedModelId(req.body.modelId);
  const filePayload = {
    buffer: Buffer.from(req.file.buffer),
    originalname: req.file.originalname || "video.mp4",
    mimetype: req.file.mimetype || "video/mp4",
  };

  processFocusCutJob(job.id, filePayload, initialSubjectHint, requestedModelId).catch((error) => {
    failFocusCutJob(job.id, error);
  });

  res.status(202).json({
    ok: true,
    jobId: job.id,
    status: job.status,
    progress: job.progress,
    stage: job.stage,
    message: job.message,
    statusUrl: `/api/focus-cut/jobs/${job.id}`,
  });
});

app.get("/api/focus-cut/jobs/:jobId", (req, res) => {
  const job = focusCutJobs.get(req.params.jobId);
  if (!job) {
    return res.status(404).json({
      ok: false,
      error: "Focus-cut job not found.",
    });
  }

  res.json(serializeFocusCutJob(job));
});

app.use((error, req, res, next) => {
  if (error instanceof multer.MulterError && error.code === "LIMIT_FILE_SIZE") {
    return res.status(413).json({
      error: `Upload too large. Max allowed is ${MAX_UPLOAD_MB} MB.`,
      maxUploadMb: MAX_UPLOAD_MB,
    });
  }

  if (error instanceof multer.MulterError) {
    return res.status(400).json({
      error: `Upload failed: ${error.message}`,
    });
  }

  if (error) {
    return res.status(500).json({
      error: error.message || "Unexpected server error.",
    });
  }

  return next();
});

const server = app.listen(APP_PORT, async () => {
  console.log(`App UI ready at http://127.0.0.1:${APP_PORT}`);
  try {
    await ensureLlamaServer(DEFAULT_MODEL_ID);
    console.log(`Model backend ready at ${LLAMA_BASE_URL} (${getModelConfig(DEFAULT_MODEL_ID).name})`);
  } catch (error) {
    console.error(`Model startup failed: ${error.message}`);
  }
});

server.on("error", (error) => {
  console.error(`App startup failed: ${error.message}`);
  process.exit(1);
});

async function getStatus() {
  const models = MODEL_ORDER.map((modelId) => {
    const config = getModelConfig(modelId);
    return {
      id: modelId,
      name: config.name,
      modelPath: config.modelPath,
      mmprojPath: config.mmprojPath,
      modelExists: fs.existsSync(config.modelPath),
      mmprojExists: fs.existsSync(config.mmprojPath),
      currentCtxSize: currentCtxSizes[modelId],
    };
  });
  const llamaExists = fs.existsSync(LLAMA_BIN);
  const ffmpegExists = fs.existsSync(FFMPEG_BIN);
  const ffprobeExists = fs.existsSync(FFPROBE_BIN);
  const detectorModelExists = fs.existsSync(DETECTOR_MODEL_PATH);
  const backendUp = await isBackendHealthy();
  const allModelFilesReady = models.every((model) => model.modelExists && model.mmprojExists);

  return {
    ok: backendUp && allModelFilesReady && llamaExists && ffmpegExists && ffprobeExists && detectorModelExists,
    llamaExists,
    ffmpegExists,
    ffprobeExists,
    detectorModelExists,
    backendUp,
    appPort: APP_PORT,
    llamaPort: LLAMA_PORT,
    activeModelId,
    activeModelName: activeModelId ? getModelConfig(activeModelId).name : "",
    currentCtxSize: activeModelId ? currentCtxSizes[activeModelId] : currentCtxSizes[DEFAULT_MODEL_ID],
    maxCtxSize: MAX_CTX_SIZE,
    maxUploadMb: MAX_UPLOAD_MB,
    modelPath: activeModelId ? getModelConfig(activeModelId).modelPath : getModelConfig(DEFAULT_MODEL_ID).modelPath,
    mmprojPath: activeModelId ? getModelConfig(activeModelId).mmprojPath : getModelConfig(DEFAULT_MODEL_ID).mmprojPath,
    models,
  };
}

function getModelConfig(modelId = DEFAULT_MODEL_ID) {
  const config = MODEL_CONFIGS[modelId];
  if (!config) {
    throw new Error(`Unknown model id: ${modelId}`);
  }
  return config;
}

function resolveRequestedModelId(rawModelId) {
  const candidate = String(rawModelId || "").trim().toLowerCase();
  if (candidate && MODEL_CONFIGS[candidate]) {
    return candidate;
  }
  return activeModelId || DEFAULT_MODEL_ID;
}

async function ensureLlamaServer(modelId = DEFAULT_MODEL_ID) {
  if (llamaStartPromise) {
    await llamaStartPromise;
  }

  if (activeModelId === modelId && (await isBackendHealthy())) {
    return;
  }

  const config = getModelConfig(modelId);

  if (!fs.existsSync(LLAMA_BIN)) {
    throw new Error(`llama-server not found at ${LLAMA_BIN}`);
  }

  if (!fs.existsSync(config.modelPath)) {
    throw new Error(`Model file not found at ${config.modelPath}`);
  }

  if (!fs.existsSync(config.mmprojPath)) {
    throw new Error(`mmproj file not found at ${config.mmprojPath}`);
  }

  if (!fs.existsSync(FFMPEG_BIN)) {
    throw new Error(`ffmpeg not found at ${FFMPEG_BIN}`);
  }

  if (!fs.existsSync(FFPROBE_BIN)) {
    throw new Error(`ffprobe not found at ${FFPROBE_BIN}`);
  }

  llamaStartPromise = (async () => {
    await stopLlamaServer();
    await startLlamaServer(modelId);
  })();
  try {
    await llamaStartPromise;
  } finally {
    llamaStartPromise = null;
  }
}

async function startLlamaServer(modelId) {
  const config = getModelConfig(modelId);
  const args = [
    "-m",
    config.modelPath,
    "--mmproj",
    config.mmprojPath,
    "--host",
    LLAMA_HOST,
    "--port",
    String(LLAMA_PORT),
    "-c",
    String(currentCtxSizes[modelId]),
    "-ngl",
    String(GPU_LAYERS),
    "-np",
    "1",
  ];

  llamaChild = spawn(LLAMA_BIN, args, {
    stdio: ["ignore", "pipe", "pipe"],
  });

  llamaChild.stdout.on("data", (chunk) => {
    process.stdout.write(`[llama] ${chunk}`);
  });

  llamaChild.stderr.on("data", (chunk) => {
    process.stderr.write(`[llama] ${chunk}`);
  });

  llamaChild.on("exit", (code, signal) => {
    console.log(`llama-server exited (code=${code}, signal=${signal})`);
    llamaChild = null;
    activeModelId = null;
  });

  await waitForBackend(180000);
  activeModelId = modelId;
}

async function isBackendHealthy() {
  try {
    const response = await fetch(`${LLAMA_BASE_URL}/health`, {
      signal: AbortSignal.timeout(1500),
    });
    return response.ok;
  } catch {
    return false;
  }
}

async function waitForBackend(timeoutMs) {
  const startedAt = Date.now();

  while (Date.now() - startedAt < timeoutMs) {
    if (await isBackendHealthy()) {
      return;
    }

    if (!llamaChild) {
      throw new Error("llama-server exited before becoming ready.");
    }

    await sleep(1000);
  }

  throw new Error("Timed out while waiting for llama-server to start.");
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function askModel(contentParts, prompt, options = {}) {
  const modelId = options.modelId || DEFAULT_MODEL_ID;
  await ensureLlamaServer(modelId);
  return askModelWithRetry(contentParts, prompt, 0, {
    ...options,
    modelId,
  });
}

async function askModelWithRetry(contentParts, prompt, attempt, options = {}) {
  const modelId = options.modelId || DEFAULT_MODEL_ID;
  const payload = {
    temperature: options.temperature ?? 0.2,
    top_k: options.topK ?? 20,
    top_p: options.topP ?? 0.8,
    max_tokens: options.maxTokens ?? 500,
    messages: [
      {
        role: "user",
        content: [{ type: "text", text: prompt }, ...contentParts],
      },
    ],
  };

  const response = await fetch(`${LLAMA_BASE_URL}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const data = await response.json();

  if (!response.ok) {
    const message = data?.error?.message || "Model request failed.";

    if (attempt === 0 && shouldRetryWithLargerContext(message)) {
      await increaseContextForMessage(message, modelId);
      return askModelWithRetry(contentParts, prompt, attempt + 1, options);
    }

    throw new Error(message);
  }

  return data?.choices?.[0]?.message?.content?.trim() || "The model returned an empty response.";
}

async function inferMainSubject(contentParts, instruction, options = {}) {
  const subject = await askModel(contentParts, instruction, {
    temperature: 0,
    topK: 1,
    topP: 1,
    maxTokens: 16,
    modelId: options.modelId,
  });

  return normalizeSubjectHint(subject);
}

function normalizeSubjectHint(subject) {
  const normalized = subject
    .toLowerCase()
    .replace(/[`"'*_[\]{}()]/g, " ")
    .replace(/[^a-z0-9\s-]/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .split(" ")
    .slice(0, 4)
    .join(" ");

  const blockedWords = [
    "mountain",
    "mountains",
    "valley",
    "river",
    "fjord",
    "landscape",
    "scenery",
    "sky",
    "water",
    "ocean",
    "sea",
    "rock",
    "rocks",
    "rocky",
    "formation",
    "cliff",
    "terrain",
    "forest",
    "trees",
  ];

  if (!normalized || normalized === "none" || normalized === "no subject") {
    return "";
  }

  const words = normalized.split(" ");
  if (words.some((word) => blockedWords.includes(word))) {
    return "";
  }

  return normalized;
}

function choosePreferredSubjectHint(analyses) {
  return analyses.find((analysis) => analysis.subjectHint)?.subjectHint || "";
}

function formatCombinedAnswer(analyses) {
  return analyses
    .map((analysis) => {
      const header = `[${analysis.modelName}]`;
      if (analysis.error) {
        return `${header}\nError: ${analysis.error}`;
      }
      return `${header}\n${analysis.answer}`;
    })
    .join("\n\n");
}

async function runSequentialModelAnalyses(contentParts, prompt, subjectPrompt, modelIds = MODEL_ORDER) {
  const analyses = [];

  for (const modelId of modelIds) {
    const config = getModelConfig(modelId);

    try {
      const answer = await askModel(contentParts, prompt, { modelId });
      const subjectHint = await inferMainSubject(contentParts, subjectPrompt, { modelId });
      analyses.push({
        modelId,
        modelName: config.name,
        answer,
        subjectHint,
      });
    } catch (error) {
      analyses.push({
        modelId,
        modelName: config.name,
        answer: "",
        subjectHint: "",
        error: error.message,
      });
    }
  }

  if (!analyses.some((analysis) => !analysis.error)) {
    throw new Error(analyses.map((analysis) => `${analysis.modelName}: ${analysis.error}`).join(" | "));
  }

  return analyses;
}

function shouldRetryWithLargerContext(message) {
  return /exceeds the available context size/i.test(message);
}

async function increaseContextForMessage(message, modelId) {
  const requiredTokens = parseRequiredTokens(message);
  const targetCtxSize = chooseTargetContextSize(requiredTokens, modelId);

  if (targetCtxSize <= currentCtxSizes[modelId]) {
    throw new Error(message);
  }

  await restartLlamaServer(targetCtxSize, modelId);
}

function parseRequiredTokens(message) {
  const match = message.match(/request \((\d+) tokens\)/i);
  if (!match) {
    return currentCtxSize + 1;
  }

  return Number(match[1]);
}

function chooseTargetContextSize(requiredTokens, modelId) {
  const currentCtxSize = currentCtxSizes[modelId];
  const needed = Math.max(requiredTokens + 1024, currentCtxSize * 2);
  let next = 4096;

  while (next < needed) {
    next *= 2;
  }

  if (next > MAX_CTX_SIZE) {
    throw new Error(
      `This request needs more context than the current auto-grow limit allows. Required about ${requiredTokens} tokens, current context ${currentCtxSize}, max auto context ${MAX_CTX_SIZE}. Increase MAX_CTX_SIZE if your machine has enough memory.`,
    );
  }

  return next;
}

async function restartLlamaServer(targetCtxSize, modelId) {
  currentCtxSizes[modelId] = targetCtxSize;
  await stopLlamaServer();
  await ensureLlamaServer(modelId);
}

async function stopLlamaServer() {
  if (!llamaChild) {
    return;
  }

  const child = llamaChild;

  await new Promise((resolve) => {
    child.once("exit", () => {
      resolve();
    });

    child.kill("SIGTERM");
  });
}

async function analyzeVideo(file, prompt, modelId) {
  const tempDir = await fsp.mkdtemp(path.join(os.tmpdir(), "qwen-vl-video-"));

  try {
    const inputPath = path.join(tempDir, safeFileName(file.originalname || "upload-video"));
    await fsp.writeFile(inputPath, file.buffer);

    const { framePaths, contentParts } = await buildVideoContentParts(inputPath, tempDir);

    const orderedPrompt =
      `${prompt}\n\n` +
      `These ${framePaths.length} frames are in time order from earliest to latest. ` +
      "Describe the progression of events, not just isolated images.";

    const analyses = await runSequentialModelAnalyses(
      contentParts,
      orderedPrompt,
      "Identify the single main subject across these video frames that could fit inside one bounding box. Only return a person, animal, vehicle, or everyday object. Do not return terrain, sky, water, mountains, buildings, or broad scenery. If no single boxable subject stands out, return none. Return only 1 to 4 lowercase words, no punctuation.",
      [modelId],
    );

    return {
      analyses,
      frameCount: framePaths.length,
      subjectHint: choosePreferredSubjectHint(analyses),
    };
  } finally {
    await fsp.rm(tempDir, { recursive: true, force: true });
  }
}

async function buildVideoContentParts(inputPath, tempDir) {
  const duration = await getVideoDuration(inputPath);
  const framePaths = await extractVideoFrames(inputPath, tempDir, duration);

  if (framePaths.length === 0) {
    throw new Error("No frames could be extracted from the video.");
  }

  const contentParts = [];

  for (let index = 0; index < framePaths.length; index += 1) {
    contentParts.push({
      type: "text",
      text: `Frame ${index + 1} of ${framePaths.length} (chronological order).`,
    });
    contentParts.push({
      type: "image_url",
      image_url: {
        url: await filePathToDataUrl(framePaths[index], "image/jpeg"),
      },
    });
  }

  return {
    framePaths,
    contentParts,
  };
}

async function getVideoDuration(inputPath) {
  const result = await runCommand(FFPROBE_BIN, [
    "-v",
    "error",
    "-show_entries",
    "format=duration",
    "-of",
    "default=noprint_wrappers=1:nokey=1",
    inputPath,
  ]);

  const duration = Number(result.stdout.trim());
  if (!Number.isFinite(duration) || duration <= 0) {
    return 12;
  }

  return duration;
}

async function extractVideoFrames(inputPath, tempDir, duration) {
  const intervalSeconds = Math.max(duration / MAX_VIDEO_FRAMES, 1);
  const outputPattern = path.join(tempDir, "frame-%03d.jpg");

  await runCommand(FFMPEG_BIN, [
    "-hide_banner",
    "-loglevel",
    "error",
    "-i",
    inputPath,
    "-vf",
    `fps=1/${intervalSeconds}`,
    "-frames:v",
    String(MAX_VIDEO_FRAMES),
    "-q:v",
    "3",
    outputPattern,
  ]);

  const files = await fsp.readdir(tempDir);
  return files
    .filter((name) => name.startsWith("frame-") && name.endsWith(".jpg"))
    .sort()
    .map((name) => path.join(tempDir, name));
}

async function filePathToDataUrl(filePath, mimeType) {
  const buffer = await fsp.readFile(filePath);
  return `data:${mimeType};base64,${buffer.toString("base64")}`;
}

function bufferToDataUrl(file) {
  return `data:${file.mimetype};base64,${file.buffer.toString("base64")}`;
}

function createFocusCutJob({ originalName }) {
  pruneFocusCutJobs();
  const id = `focus-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
  const now = Date.now();
  const job = {
    id,
    status: "queued",
    stage: "queued",
    message: "Queued for focus cut.",
    progress: 0,
    etaSeconds: null,
    createdAt: now,
    updatedAt: now,
    startedAt: null,
    finishedAt: null,
    originalName: originalName || "video.mp4",
    outputName: null,
    outputUrl: null,
    downloadName: null,
    renderSubject: "",
    subjectHint: "",
    inferredFromFrames: false,
    frameCount: 0,
    error: "",
  };
  focusCutJobs.set(id, job);
  return job;
}

function serializeFocusCutJob(job) {
  return {
    ok: true,
    jobId: job.id,
    status: job.status,
    stage: job.stage,
    message: job.message,
    progress: job.progress,
    etaSeconds: job.etaSeconds,
    createdAt: job.createdAt,
    updatedAt: job.updatedAt,
    startedAt: job.startedAt,
    finishedAt: job.finishedAt,
    outputUrl: job.outputUrl,
    downloadName: job.downloadName,
    outputName: job.outputName,
    renderSubject: job.renderSubject,
    subjectHint: job.subjectHint,
    inferredFromFrames: job.inferredFromFrames,
    frameCount: job.frameCount,
    error: job.error,
  };
}

function pruneFocusCutJobs() {
  const cutoff = Date.now() - FOCUS_JOB_MAX_AGE_MS;
  for (const [jobId, job] of focusCutJobs.entries()) {
    const finishedAt = job.finishedAt || job.updatedAt;
    if (finishedAt < cutoff) {
      focusCutJobs.delete(jobId);
    }
  }
}

function clamp01(value) {
  if (!Number.isFinite(value)) {
    return 0;
  }
  if (value < 0) {
    return 0;
  }
  if (value > 1) {
    return 1;
  }
  return value;
}

function stageProgressToOverall(stage, stageRatio) {
  const ratio = clamp01(stageRatio);
  const weights = {
    preparing: 0.03,
    subject: 0.07,
    planning: 0.62,
    rendering: 0.25,
    muxing: 0.03,
  };
  const order = ["preparing", "subject", "planning", "rendering", "muxing"];
  if (stage === "complete") {
    return 1;
  }
  if (!weights[stage]) {
    return ratio;
  }
  let base = 0;
  for (const key of order) {
    if (key === stage) {
      break;
    }
    base += weights[key];
  }
  return clamp01(base + weights[stage] * ratio);
}

function toFinitePositiveInt(value) {
  const num = Number(value);
  if (!Number.isFinite(num) || num <= 0) {
    return null;
  }
  return Math.round(num);
}

function computeEtaSecondsFromJob(job, progress) {
  if (!job.startedAt || progress <= 0.01 || progress >= 0.999) {
    return null;
  }
  const elapsedSeconds = (Date.now() - job.startedAt) / 1000;
  if (!Number.isFinite(elapsedSeconds) || elapsedSeconds <= 0) {
    return null;
  }
  const eta = elapsedSeconds / progress - elapsedSeconds;
  if (!Number.isFinite(eta) || eta < 0) {
    return null;
  }
  return Math.round(eta);
}

function updateFocusCutJob(jobId, patch) {
  const job = focusCutJobs.get(jobId);
  if (!job) {
    return null;
  }

  Object.assign(job, patch);
  job.updatedAt = Date.now();

  if (typeof patch.progress === "number" && job.status === "running") {
    job.progress = Math.min(99, Math.max(0, Math.round(patch.progress)));
  }
  if (job.status === "completed") {
    job.progress = 100;
    job.etaSeconds = 0;
    job.finishedAt = job.finishedAt || Date.now();
  }
  if (job.status === "failed") {
    job.finishedAt = job.finishedAt || Date.now();
  }
  return job;
}

function failFocusCutJob(jobId, error) {
  const message = error instanceof Error ? error.message : String(error);
  updateFocusCutJob(jobId, {
    status: "failed",
    stage: "failed",
    message: "Focus cut failed.",
    error: message,
    etaSeconds: null,
  });
}

function parseFocusProgressLine(line) {
  if (!line.startsWith(FOCUS_PROGRESS_PREFIX)) {
    return null;
  }
  try {
    const payload = JSON.parse(line.slice(FOCUS_PROGRESS_PREFIX.length));
    if (!payload || typeof payload !== "object") {
      return null;
    }
    return payload;
  } catch {
    return null;
  }
}

function applyPythonProgressToJob(jobId, payload) {
  const job = focusCutJobs.get(jobId);
  if (!job || job.status !== "running") {
    return;
  }

  const stage = String(payload.stage || "").trim() || "planning";
  const total = toFinitePositiveInt(payload.total);
  const current = toFinitePositiveInt(payload.current);
  let stageRatio = typeof payload.ratio === "number" ? payload.ratio : null;
  if (stageRatio === null && total && current !== null) {
    stageRatio = current / total;
  }
  const overall = stageProgressToOverall(stage, stageRatio === null ? 0 : stageRatio);
  const etaFromModel = Number.isFinite(payload.eta_seconds) ? Math.max(0, Math.round(payload.eta_seconds)) : null;
  const etaSeconds = etaFromModel !== null ? etaFromModel : computeEtaSecondsFromJob(job, overall);

  updateFocusCutJob(jobId, {
    stage,
    message: String(payload.message || ""),
    progress: Math.round(overall * 100),
    etaSeconds,
  });
}

async function processFocusCutJob(jobId, file, initialSubjectHint, modelId) {
  const job = focusCutJobs.get(jobId);
  if (!job) {
    return;
  }

  updateFocusCutJob(jobId, {
    status: "running",
    stage: "preparing",
    message: "Preparing video upload.",
    progress: Math.round(stageProgressToOverall("preparing", 0.1) * 100),
    etaSeconds: null,
    startedAt: Date.now(),
    error: "",
  });

  await ensureLlamaServer(modelId);
  updateFocusCutJob(jobId, {
    stage: "preparing",
    message: "Model backend ready.",
    progress: Math.round(stageProgressToOverall("preparing", 1.0) * 100),
  });

  const tempDir = await fsp.mkdtemp(path.join(os.tmpdir(), "qwen-vl-focus-cut-"));

  try {
    const inputPath = path.join(tempDir, safeFileName(file.originalname || "upload-video"));
    await fsp.writeFile(inputPath, file.buffer);

    let subjectHint = normalizeSubjectHint(initialSubjectHint || "");
    let frameCount = 0;

    if (!subjectHint) {
      updateFocusCutJob(jobId, {
        stage: "subject",
        message: "Detecting the main subject.",
        progress: Math.round(stageProgressToOverall("subject", 0.2) * 100),
      });
      const { framePaths, contentParts } = await buildVideoContentParts(inputPath, tempDir);
      frameCount = framePaths.length;
      subjectHint = await inferMainSubject(
        contentParts,
        "Identify the single main subject across these video frames that could fit inside one bounding box. Only return a person, animal, vehicle, or everyday object. Do not return terrain, sky, water, mountains, buildings, or broad scenery. If no single boxable subject stands out, return none. Return only 1 to 4 lowercase words, no punctuation.",
        { modelId },
      );
      updateFocusCutJob(jobId, {
        stage: "subject",
        message: "Main subject identified.",
        progress: Math.round(stageProgressToOverall("subject", 1.0) * 100),
        frameCount,
        inferredFromFrames: frameCount > 0,
      });
    }

    const renderSubject = subjectHint || "subject";
    const outputName = buildGeneratedVideoName(file.originalname || "video.mp4");
    const outputPath = path.join(GENERATED_DIR, outputName);

    updateFocusCutJob(jobId, {
      stage: "planning",
      message: "Planning BIG/MID layout.",
      progress: Math.round(stageProgressToOverall("planning", 0.0) * 100),
      outputName,
      outputUrl: `/generated/${outputName}`,
      downloadName: outputName,
      renderSubject,
      subjectHint,
    });

    await renderFocusCutVideo(inputPath, outputPath, renderSubject, (payload) => {
      applyPythonProgressToJob(jobId, payload);
    });

    updateFocusCutJob(jobId, {
      status: "completed",
      stage: "complete",
      message: "Focus cut complete.",
      progress: 100,
      etaSeconds: 0,
      outputName,
      outputUrl: `/generated/${outputName}`,
      downloadName: outputName,
      renderSubject,
      subjectHint,
      frameCount,
      inferredFromFrames: frameCount > 0,
      error: "",
    });
  } finally {
    await fsp.rm(tempDir, { recursive: true, force: true });
  }
}

function safeFileName(fileName) {
  return fileName.replace(/[^a-zA-Z0-9._-]/g, "_");
}

function buildGeneratedVideoName(fileName) {
  const parsed = path.parse(safeFileName(fileName || "video.mp4"));
  const stem = parsed.name || "video";
  return `${stem}-focus-cut-${Date.now()}.mp4`;
}

function isImageMimeType(mimeType, fileName = "") {
  return mimeType.startsWith("image/") || /\.(png|jpg|jpeg|webp|gif|bmp)$/i.test(fileName);
}

function isVideoMimeType(mimeType, fileName = "") {
  return mimeType.startsWith("video/") || /\.(mp4|mov|m4v|webm|avi)$/i.test(fileName);
}

async function renderFocusCutVideo(inputPath, outputPath, subjectHint, onProgress) {
  if (!fs.existsSync(DETECTOR_MODEL_PATH)) {
    throw new Error(`MediaPipe detector model not found at ${DETECTOR_MODEL_PATH}`);
  }

  const args = [
    path.join(__dirname, "tools", "annotate_subject_video.py"),
    "--input",
    inputPath,
    "--output",
    outputPath,
    "--model",
    DETECTOR_MODEL_PATH,
    "--subject",
    subjectHint,
    "--llama-url",
    LLAMA_BASE_URL,
    "--crop-vertical",
    "--no-box",
  ];
  await runCommandWithProgress(PYTHON_BIN, args, onProgress);
}

function runCommand(command, args) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      stdio: ["ignore", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });

    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });

    child.on("error", (error) => {
      reject(error);
    });

    child.on("exit", (code) => {
      if (code === 0) {
        resolve({ stdout, stderr });
        return;
      }

      reject(new Error(stderr.trim() || `${command} exited with code ${code}`));
    });
  });
}

function runCommandWithProgress(command, args, onProgress) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      stdio: ["ignore", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";
    let stdoutBuffer = "";

    const flushStdoutLines = () => {
      let newlineIndex = stdoutBuffer.indexOf("\n");
      while (newlineIndex >= 0) {
        const rawLine = stdoutBuffer.slice(0, newlineIndex);
        const line = rawLine.trim();
        if (line) {
          const payload = parseFocusProgressLine(line);
          if (payload && typeof onProgress === "function") {
            onProgress(payload);
          }
        }
        stdoutBuffer = stdoutBuffer.slice(newlineIndex + 1);
        newlineIndex = stdoutBuffer.indexOf("\n");
      }
    };

    child.stdout.on("data", (chunk) => {
      const text = chunk.toString();
      stdout += text;
      stdoutBuffer += text;
      flushStdoutLines();
    });

    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });

    child.on("error", (error) => {
      reject(error);
    });

    child.on("exit", (code) => {
      if (stdoutBuffer.trim()) {
        const payload = parseFocusProgressLine(stdoutBuffer.trim());
        if (payload && typeof onProgress === "function") {
          onProgress(payload);
        }
      }
      if (code === 0) {
        resolve({ stdout, stderr });
        return;
      }
      reject(new Error(stderr.trim() || `${command} exited with code ${code}`));
    });
  });
}

function shutdown() {
  if (llamaChild) {
    llamaChild.kill("SIGTERM");
  }
  process.exit(0);
}

process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);
