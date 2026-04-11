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
const MODEL_PATH =
  process.env.MODEL_PATH ||
  "/Users/sudeepnt/Desktop/DMain/AI Model Stuff/Qwen3VL-4B-Instruct-Q4_K_M.gguf";
const MMPROJ_PATH =
  process.env.MMPROJ_PATH ||
  "/Users/sudeepnt/Desktop/DMain/AI Model Stuff/mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf";
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
const DETECTOR_MODEL_PATH =
  process.env.DETECTOR_MODEL_PATH || path.join(__dirname, "models", "efficientdet_lite0.tflite");
const GENERATED_DIR = path.join(__dirname, "generated");

const app = express();
const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 120 * 1024 * 1024,
  },
});

let llamaChild = null;
let llamaStartPromise = null;
let currentCtxSize = INITIAL_CTX_SIZE;

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

app.post("/api/explain", upload.single("media"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "Please upload an image or video." });
    }

    await ensureLlamaServer();

    if (isVideoMimeType(req.file.mimetype, req.file.originalname)) {
      const prompt =
        (req.body.prompt || "").trim() ||
        "These are sampled frames from one video in chronological order. Explain what happens across the video, mention the key objects or people, and include any readable on-screen text.";

      const { answer, frameCount, subjectHint } = await analyzeVideo(req.file, prompt);
      return res.json({
        answer,
        meta: {
          type: "video",
          frameCount,
          subjectHint,
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
    const answer = await askModel(contentParts, prompt);
    const subjectHint = await inferMainSubject(
      contentParts,
      "Identify the single main subject in this image that could fit inside one bounding box. Only return a person, animal, vehicle, or everyday object. Do not return terrain, sky, water, mountains, buildings, or broad scenery. If no single boxable subject stands out, return none. Return only 1 to 4 lowercase words, no punctuation.",
    );

    res.json({
      answer,
      meta: {
        type: "image",
        subjectHint,
      },
    });
  } catch (error) {
    res.status(500).json({
      error: error.message,
    });
  }
});

app.post("/api/focus-cut", upload.single("media"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "Please upload a video first." });
    }

    if (!isVideoMimeType(req.file.mimetype, req.file.originalname)) {
      return res.status(400).json({
        error: "Focus and Cut works on video uploads only.",
      });
    }

    await ensureLlamaServer();

    const tempDir = await fsp.mkdtemp(path.join(os.tmpdir(), "qwen-vl-focus-cut-"));

    try {
      const inputPath = path.join(tempDir, safeFileName(req.file.originalname || "upload-video"));
      await fsp.writeFile(inputPath, req.file.buffer);

      let subjectHint = normalizeSubjectHint((req.body.subjectHint || "").trim());
      let frameCount = 0;

      if (!subjectHint) {
        const { framePaths, contentParts } = await buildVideoContentParts(inputPath, tempDir);
        frameCount = framePaths.length;
        subjectHint = await inferMainSubject(
          contentParts,
          "Identify the single main subject across these video frames that could fit inside one bounding box. Only return a person, animal, vehicle, or everyday object. Do not return terrain, sky, water, mountains, buildings, or broad scenery. If no single boxable subject stands out, return none. Return only 1 to 4 lowercase words, no punctuation.",
        );
      }

      const renderSubject = subjectHint || "subject";
      const outputName = buildGeneratedVideoName(req.file.originalname || "video.mp4");
      const outputPath = path.join(GENERATED_DIR, outputName);

      await renderFocusCutVideo(inputPath, outputPath, renderSubject);

      res.json({
        ok: true,
        outputUrl: `/generated/${outputName}`,
        downloadName: outputName,
        subjectHint,
        renderSubject,
        inferredFromFrames: frameCount > 0,
        frameCount,
      });
    } finally {
      await fsp.rm(tempDir, { recursive: true, force: true });
    }
  } catch (error) {
    res.status(500).json({
      error: error.message,
    });
  }
});

const server = app.listen(APP_PORT, async () => {
  console.log(`App UI ready at http://127.0.0.1:${APP_PORT}`);
  try {
    await ensureLlamaServer();
    console.log(`Model backend ready at ${LLAMA_BASE_URL}`);
  } catch (error) {
    console.error(`Model startup failed: ${error.message}`);
  }
});

server.on("error", (error) => {
  console.error(`App startup failed: ${error.message}`);
  process.exit(1);
});

async function getStatus() {
  const modelExists = fs.existsSync(MODEL_PATH);
  const mmprojExists = fs.existsSync(MMPROJ_PATH);
  const llamaExists = fs.existsSync(LLAMA_BIN);
  const ffmpegExists = fs.existsSync(FFMPEG_BIN);
  const ffprobeExists = fs.existsSync(FFPROBE_BIN);
  const detectorModelExists = fs.existsSync(DETECTOR_MODEL_PATH);
  const backendUp = await isBackendHealthy();

  return {
    ok:
      backendUp &&
      modelExists &&
      mmprojExists &&
      llamaExists &&
      ffmpegExists &&
      ffprobeExists &&
      detectorModelExists,
    modelExists,
    mmprojExists,
    llamaExists,
    ffmpegExists,
    ffprobeExists,
    detectorModelExists,
    backendUp,
    appPort: APP_PORT,
    llamaPort: LLAMA_PORT,
    currentCtxSize,
    maxCtxSize: MAX_CTX_SIZE,
    modelPath: MODEL_PATH,
    mmprojPath: MMPROJ_PATH,
  };
}

async function ensureLlamaServer() {
  if (await isBackendHealthy()) {
    return;
  }

  if (llamaStartPromise) {
    return llamaStartPromise;
  }

  if (!fs.existsSync(LLAMA_BIN)) {
    throw new Error(`llama-server not found at ${LLAMA_BIN}`);
  }

  if (!fs.existsSync(MODEL_PATH)) {
    throw new Error(`Model file not found at ${MODEL_PATH}`);
  }

  if (!fs.existsSync(MMPROJ_PATH)) {
    throw new Error(`mmproj file not found at ${MMPROJ_PATH}`);
  }

  if (!fs.existsSync(FFMPEG_BIN)) {
    throw new Error(`ffmpeg not found at ${FFMPEG_BIN}`);
  }

  if (!fs.existsSync(FFPROBE_BIN)) {
    throw new Error(`ffprobe not found at ${FFPROBE_BIN}`);
  }

  llamaStartPromise = startLlamaServer();
  try {
    await llamaStartPromise;
  } finally {
    llamaStartPromise = null;
  }
}

async function startLlamaServer() {
  const args = [
    "-m",
    MODEL_PATH,
    "--mmproj",
    MMPROJ_PATH,
    "--host",
    LLAMA_HOST,
    "--port",
    String(LLAMA_PORT),
    "-c",
    String(currentCtxSize),
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
  });

  await waitForBackend(180000);
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
  return askModelWithRetry(contentParts, prompt, 0, options);
}

async function askModelWithRetry(contentParts, prompt, attempt, options = {}) {
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
      await increaseContextForMessage(message);
      return askModelWithRetry(contentParts, prompt, attempt + 1, options);
    }

    throw new Error(message);
  }

  return data?.choices?.[0]?.message?.content?.trim() || "The model returned an empty response.";
}

async function inferMainSubject(contentParts, instruction) {
  const subject = await askModel(contentParts, instruction, {
    temperature: 0,
    topK: 1,
    topP: 1,
    maxTokens: 16,
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

function shouldRetryWithLargerContext(message) {
  return /exceeds the available context size/i.test(message);
}

async function increaseContextForMessage(message) {
  const requiredTokens = parseRequiredTokens(message);
  const targetCtxSize = chooseTargetContextSize(requiredTokens);

  if (targetCtxSize <= currentCtxSize) {
    throw new Error(message);
  }

  await restartLlamaServer(targetCtxSize);
}

function parseRequiredTokens(message) {
  const match = message.match(/request \((\d+) tokens\)/i);
  if (!match) {
    return currentCtxSize + 1;
  }

  return Number(match[1]);
}

function chooseTargetContextSize(requiredTokens) {
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

async function restartLlamaServer(targetCtxSize) {
  currentCtxSize = targetCtxSize;

  if (llamaStartPromise) {
    await llamaStartPromise;
  }

  await stopLlamaServer();
  await ensureLlamaServer();
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

async function analyzeVideo(file, prompt) {
  const tempDir = await fsp.mkdtemp(path.join(os.tmpdir(), "qwen-vl-video-"));

  try {
    const inputPath = path.join(tempDir, safeFileName(file.originalname || "upload-video"));
    await fsp.writeFile(inputPath, file.buffer);

    const { framePaths, contentParts } = await buildVideoContentParts(inputPath, tempDir);

    const orderedPrompt =
      `${prompt}\n\n` +
      `These ${framePaths.length} frames are in time order from earliest to latest. ` +
      "Describe the progression of events, not just isolated images.";

    const answer = await askModel(contentParts, orderedPrompt);
    const subjectHint = await inferMainSubject(
      contentParts,
      "Identify the single main subject across these video frames that could fit inside one bounding box. Only return a person, animal, vehicle, or everyday object. Do not return terrain, sky, water, mountains, buildings, or broad scenery. If no single boxable subject stands out, return none. Return only 1 to 4 lowercase words, no punctuation.",
    );

    return {
      answer,
      frameCount: framePaths.length,
      subjectHint,
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

async function renderFocusCutVideo(inputPath, outputPath, subjectHint) {
  if (!fs.existsSync(DETECTOR_MODEL_PATH)) {
    throw new Error(`MediaPipe detector model not found at ${DETECTOR_MODEL_PATH}`);
  }

  await runCommand(PYTHON_BIN, [
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
    "--source-inset",
  ]);
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

function shutdown() {
  if (llamaChild) {
    llamaChild.kill("SIGTERM");
  }
  process.exit(0);
}

process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);
