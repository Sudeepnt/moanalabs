# Qwen3VL local media explainer

This is a tiny local app that:

- starts `llama-server`
- loads your local `Qwen3VL-4B-Instruct-Q4_K_M.gguf`
- attaches the matching `mmproj` vision file
- serves a small HTML UI for image and short-video explanation
- uses MediaPipe Object Detector in the browser to draw a box around the model's predicted main subject

## Context window

The app now starts `llama-server` with a larger default context window and will automatically restart with a bigger context if a request exceeds the current limit.

- default context: `16384`
- auto-grow ceiling: `65536`

## Video support

This app handles video by sampling a small number of frames with `ffmpeg` and sending those frames to the model in time order. It is not native end-to-end video inference, but it works well for short clips and scene summaries.

## Subject boxes

After each explanation, the backend asks Qwen for a short main-subject hint, and the frontend uses MediaPipe to choose the closest matching detection box on the preview image or current video frame.

This works best for people, animals, vehicles, and other common objects. Scene-level subjects such as mountains, oceans, or skylines may not produce a useful box because the detector is object-based.

## Run it

```bash
cd "/Users/sudeepnt/Desktop/DMain/Codex Projects/qwen vl"
npm install
npm start
```

Then open:

```text
http://127.0.0.1:3055
```

## Defaults

- Model: `/Users/sudeepnt/Desktop/DMain/AI Model Stuff/Qwen3VL-4B-Instruct-Q4_K_M.gguf`
- Vision projector: `/Users/sudeepnt/Desktop/DMain/AI Model Stuff/mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf`
- `llama-server`: `/Users/sudeepnt/llama.cpp/build/bin/llama-server`
- `ffmpeg`: `/opt/homebrew/bin/ffmpeg`
- `ffprobe`: `/opt/homebrew/bin/ffprobe`

## Optional env vars

```bash
APP_PORT=3055
LLAMA_PORT=32111
CTX_SIZE=16384
MAX_CTX_SIZE=65536
MODEL_PATH="/absolute/path/to/model.gguf"
MMPROJ_PATH="/absolute/path/to/mmproj.gguf"
LLAMA_BIN="/absolute/path/to/llama-server"
FFMPEG_BIN="/absolute/path/to/ffmpeg"
FFPROBE_BIN="/absolute/path/to/ffprobe"
GPU_LAYERS=999
MAX_VIDEO_FRAMES=6
```
