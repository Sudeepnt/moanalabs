#!/usr/bin/env python3

import argparse
import base64
import json
import math
import os
import subprocess
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Iterable, Optional

import cv2
import mediapipe as mp

FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")


ANIMAL_LABELS = {
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
}

HUMAN_LABELS = {
    "person",
    "man",
    "woman",
    "boy",
    "girl",
    "human",
}

SUBJECT_ALIASES = {
    "ostrich": {"bird", "dog", "horse", "cow", "zebra", "giraffe", "elephant"},
    "bird": {"bird"},
    "dog": {"dog"},
    "cat": {"cat"},
    "person": {"person"},
    "man": HUMAN_LABELS,
    "woman": HUMAN_LABELS,
    "speaker": HUMAN_LABELS,
    "host": HUMAN_LABELS,
    "presenter": HUMAN_LABELS,
    "guest": HUMAN_LABELS,
}


@dataclass
class Box:
    x: float
    y: float
    w: float
    h: float
    label: str
    conf: float

    def as_rect(self):
        return (int(self.x), int(self.y), int(self.w), int(self.h))

    def area(self):
        return max(self.w, 0) * max(self.h, 0)

    def center(self):
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)


@dataclass
class CropWindow:
    x: float
    y: float
    w: float
    h: float


@dataclass
class FocusPlanPoint:
    time_sec: float
    center_x: float
    label: str
    snap_cut: bool


def iou(a: Optional[Box], b: Optional[Box]) -> float:
    if a is None or b is None:
        return 0.0
    ax2, ay2 = a.x + a.w, a.y + a.h
    bx2, by2 = b.x + b.w, b.y + b.h
    ix1, iy1 = max(a.x, b.x), max(a.y, b.y)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a.area() + b.area() - inter
    return inter / union if union > 0 else 0.0


def smooth_box(prev: Optional[Box], curr: Box, alpha: float = 0.65) -> Box:
    if prev is None:
        return curr
    return Box(
        x=prev.x * (1 - alpha) + curr.x * alpha,
        y=prev.y * (1 - alpha) + curr.y * alpha,
        w=prev.w * (1 - alpha) + curr.w * alpha,
        h=prev.h * (1 - alpha) + curr.h * alpha,
        label=curr.label,
        conf=curr.conf,
    )


def is_human_subject(subject: str) -> bool:
    normalized = subject.lower().strip()
    return normalized in {"person", "man", "woman", "speaker", "host", "presenter", "guest"}


def score_candidate_box(
    box: Box,
    subject: str,
    frame_w: int,
    frame_h: int,
    prev_box: Optional[Box] = None,
    guided_center_x: Optional[float] = None,
) -> float:
    subject = subject.lower().strip()
    aliases = SUBJECT_ALIASES.get(subject, {subject})
    area_ratio = box.area() / float(frame_w * frame_h)
    center_x, center_y = box.center()
    center_dx = abs(center_x / frame_w - 0.5)
    center_dy = abs(center_y / frame_h - 0.5)
    center_penalty = math.hypot(center_dx, center_dy)

    score = box.conf * 20.0
    if box.label in aliases:
        score += 100.0
    elif box.label == "person" and is_human_subject(subject):
        score += 90.0
    elif box.label in ANIMAL_LABELS and subject == "ostrich":
        score += 45.0
    elif box.label in ANIMAL_LABELS:
        score += 10.0

    if 0.01 <= area_ratio <= 0.45:
        score += 8.0
    else:
        score -= 8.0

    score -= center_penalty * 4.0
    score += iou(prev_box, box) * 35.0

    if guided_center_x is not None:
        guided_dx = abs(center_x - guided_center_x) / max(frame_w, 1)
        score -= guided_dx * 90.0

    return score


def choose_box(
    detections,
    subject: str,
    prev_box: Optional[Box],
    frame_w: int,
    frame_h: int,
    guided_center_x: Optional[float] = None,
) -> Optional[Box]:
    subject = subject.lower().strip()
    best_box = None
    best_score = -10**9

    for det in detections:
        if not det.categories:
            continue
        cat = det.categories[0]
        label = (cat.category_name or "").lower()
        conf = float(cat.score or 0.0)
        bb = det.bounding_box
        box = Box(bb.origin_x, bb.origin_y, bb.width, bb.height, label, conf)
        score = score_candidate_box(
            box,
            subject,
            frame_w,
            frame_h,
            prev_box=prev_box,
            guided_center_x=guided_center_x,
        )

        if score > best_score:
            best_score = score
            best_box = box

    if best_box is None:
        return None

    min_accept = 12.0 if prev_box is not None else 20.0
    return best_box if best_score >= min_accept else None


def draw_box(frame, box: Box, subject_name: str):
    x, y, w, h = box.as_rect()
    x = max(0, x)
    y = max(0, y)
    w = max(1, w)
    h = max(1, h)
    color = (66, 123, 255)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 4)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)

    label = f"{subject_name} | {box.label} {box.conf:.2f}"
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    chip_y1 = max(0, y - text_h - 18)
    chip_y2 = chip_y1 + text_h + 14
    cv2.rectangle(frame, (x, chip_y1), (x + text_w + 18, chip_y2), (31, 36, 31), -1)
    cv2.putText(
        frame,
        label,
        (x + 9, chip_y2 - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (246, 241, 232),
        2,
        cv2.LINE_AA,
    )


def draw_simple_box(frame, x: int, y: int, w: int, h: int, color, label: str):
    x = max(0, x)
    y = max(0, y)
    w = max(1, w)
    h = max(1, h)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    if not label:
        return

    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    chip_y1 = max(0, y - text_h - 10)
    chip_y2 = chip_y1 + text_h + 8
    cv2.rectangle(frame, (x, chip_y1), (x + text_w + 12, chip_y2), (22, 24, 28), -1)
    cv2.putText(
        frame,
        label,
        (x + 6, chip_y2 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (246, 241, 232),
        1,
        cv2.LINE_AA,
    )


def smooth_value(prev: Optional[float], curr: float, alpha: float = 0.2) -> float:
    if prev is None:
        return curr
    return prev * (1 - alpha) + curr * alpha


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def move_towards(curr: float, target: float, max_delta: float) -> float:
    delta = target - curr
    if abs(delta) <= max_delta:
        return target
    return curr + math.copysign(max_delta, delta)


def constrain_center_to_box(
    center_x: float,
    box: Optional[Box],
    crop_w: float,
    frame_w: int,
    subject_padding_ratio: float = 0.06,
) -> float:
    if box is None:
        return center_x

    half_w = crop_w / 2.0
    pad = crop_w * subject_padding_ratio
    left_bound = box.x + box.w + pad - half_w
    right_bound = box.x - pad + half_w

    min_center = max(half_w, left_bound)
    max_center = min(frame_w - half_w, right_bound)

    if min_center <= max_center:
        return clamp(center_x, min_center, max_center)

    # If the subject is wider than the crop can safely contain, at least center on it.
    return clamp(box.center()[0], half_w, frame_w - half_w)


def constrain_center_near_subject(
    center_x: float,
    box: Optional[Box],
    crop_w: float,
    frame_w: int,
    tolerance_ratio: float,
) -> float:
    if box is None:
        return center_x

    half_w = crop_w / 2.0
    subject_center = box.center()[0]
    tolerance = crop_w * tolerance_ratio
    lower = max(half_w, subject_center - tolerance)
    upper = min(frame_w - half_w, subject_center + tolerance)
    if lower > upper:
        return clamp(subject_center, half_w, frame_w - half_w)
    return clamp(center_x, lower, upper)


def keep_subject_inside_dead_zone(
    prev_center_x: float,
    box: Optional[Box],
    crop_w: float,
    frame_w: int,
    dead_zone_ratio: float,
) -> float:
    if box is None:
        return prev_center_x

    half_w = crop_w / 2.0
    zone_half = crop_w * dead_zone_ratio
    subject_anchor_x = box.center()[0]
    zone_left = prev_center_x - zone_half
    zone_right = prev_center_x + zone_half

    if zone_left <= subject_anchor_x <= zone_right:
        return clamp(prev_center_x, half_w, frame_w - half_w)

    if subject_anchor_x < zone_left:
        return clamp(subject_anchor_x + zone_half, half_w, frame_w - half_w)

    return clamp(subject_anchor_x - zone_half, half_w, frame_w - half_w)


def build_face_detector():
    return mp.solutions.face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.35,
    )


def detect_face_boxes(face_detector, rgb_frame, frame_w: int, frame_h: int) -> list[Box]:
    result = face_detector.process(rgb_frame)
    detections = result.detections or []
    boxes = []
    for detection in detections:
        rel = detection.location_data.relative_bounding_box
        x = rel.xmin * frame_w
        y = rel.ymin * frame_h
        w = rel.width * frame_w
        h = rel.height * frame_h
        score = float(detection.score[0]) if detection.score else 0.0
        boxes.append(Box(x=x, y=y, w=w, h=h, label="face", conf=score))
    return boxes


def choose_face_box(
    face_boxes: list[Box],
    prev_face_box: Optional[Box],
    frame_w: int,
    frame_h: int,
    guided_center_x: Optional[float] = None,
    subject_box: Optional[Box] = None,
) -> Optional[Box]:
    best_box = None
    best_score = -10**9

    for box in face_boxes:
        area_ratio = box.area() / float(frame_w * frame_h)
        center_x, center_y = box.center()
        center_dx = abs(center_x / frame_w - 0.5)
        center_dy = abs(center_y / frame_h - 0.38)
        center_penalty = math.hypot(center_dx, center_dy)

        score = box.conf * 24.0
        if 0.015 <= area_ratio <= 0.16:
            score += 10.0
        else:
            score -= 12.0

        score -= center_penalty * 5.0
        score += iou(prev_face_box, box) * 20.0

        if guided_center_x is not None:
            guided_dx = abs(center_x - guided_center_x) / max(frame_w, 1)
            score -= guided_dx * 80.0

        if subject_box is not None:
            subject_x2 = subject_box.x + subject_box.w
            subject_y2 = subject_box.y + subject_box.h
            in_subject = (
                subject_box.x <= center_x <= subject_x2
                and subject_box.y <= center_y <= subject_y2
            )
            if in_subject:
                score += 75.0
            else:
                score -= 120.0

        if score > best_score:
            best_score = score
            best_box = box

    if best_box is None:
        return None

    min_accept = 8.0 if prev_face_box is not None else 14.0
    return best_box if best_score >= min_accept else None


def expand_face_to_tracking_box(face_box: Box, frame_w: int, frame_h: int) -> Box:
    x = face_box.x - face_box.w * 0.55
    y = face_box.y - face_box.h * 0.25
    w = face_box.w * 2.05
    h = face_box.h * 2.45

    x = clamp(x, 0, frame_w - 1)
    y = clamp(y, 0, frame_h - 1)
    w = min(w, frame_w - x)
    h = min(h, frame_h - y)

    return Box(
        x=x,
        y=y,
        w=w,
        h=h,
        label="speaker",
        conf=face_box.conf,
    )


def derive_speaker_anchor_box(subject_box: Optional[Box], frame_w: int, frame_h: int) -> Optional[Box]:
    if subject_box is None:
        return None

    x = subject_box.x + subject_box.w * 0.28
    y = subject_box.y + subject_box.h * 0.02
    w = subject_box.w * 0.44
    h = subject_box.h * 0.38

    x = clamp(x, 0, frame_w - 1)
    y = clamp(y, 0, frame_h - 1)
    w = min(w, frame_w - x)
    h = min(h, frame_h - y)

    return Box(
        x=x,
        y=y,
        w=w,
        h=h,
        label=subject_box.label,
        conf=subject_box.conf,
    )


def compute_crop_window(
    frame_w: int,
    frame_h: int,
    box: Optional[Box],
    prev_center_x: Optional[float],
    target_aspect: float,
    speaker_mode: bool = False,
    guided_center_x: Optional[float] = None,
    force_cut: bool = False,
) -> tuple[CropWindow, float]:
    crop_h = frame_h
    crop_w = min(frame_w, int(round(crop_h * target_aspect)))
    if crop_w <= 0:
        crop_w = frame_w

    desired_center_x = frame_w / 2.0
    if box is not None:
        desired_center_x = box.center()[0]
    elif guided_center_x is not None:
        desired_center_x = guided_center_x
    elif prev_center_x is not None:
        desired_center_x = prev_center_x

    if force_cut or prev_center_x is None:
        center_x = desired_center_x
    elif speaker_mode:
        center_x = keep_subject_inside_dead_zone(
            prev_center_x,
            box,
            crop_w,
            frame_w,
            dead_zone_ratio=0.22,
        )
    else:
        dead_zone_half = crop_w * 0.10
        target_center_x = prev_center_x
        delta = desired_center_x - prev_center_x

        if abs(delta) > dead_zone_half:
            target_center_x = desired_center_x - math.copysign(dead_zone_half, delta)

        center_x = smooth_value(prev_center_x, target_center_x, alpha=0.22)
        center_x = move_towards(prev_center_x, center_x, crop_w * 0.035)

    half_w = crop_w / 2.0
    if not speaker_mode:
        center_x = constrain_center_to_box(center_x, box, crop_w, frame_w)
    center_x = clamp(center_x, half_w, frame_w - half_w)
    x = int(round(center_x - half_w))
    y = 0
    return CropWindow(x=x, y=y, w=crop_w, h=crop_h), center_x


def crop_frame(frame, crop: CropWindow, output_w: int, output_h: int):
    x1 = int(clamp(crop.x, 0, frame.shape[1] - crop.w))
    y1 = int(clamp(crop.y, 0, frame.shape[0] - crop.h))
    x2 = x1 + int(crop.w)
    y2 = y1 + int(crop.h)
    cropped = frame[y1:y2, x1:x2]
    return cv2.resize(cropped, (output_w, output_h), interpolation=cv2.INTER_LINEAR)


def fit_inside(src_w: int, src_h: int, max_w: int, max_h: int) -> tuple[int, int]:
    if src_w <= 0 or src_h <= 0:
        return max_w, max_h
    scale = min(max_w / src_w, max_h / src_h)
    return max(1, int(round(src_w * scale))), max(1, int(round(src_h * scale)))


def encode_frame_as_data_url(frame, jpeg_quality: int = 92) -> str:
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if not ok:
        raise RuntimeError("Failed to encode guidance frame for Qwen.")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def select_candidate_boxes(
    detections,
    subject_name: str,
    frame_w: int,
    frame_h: int,
    limit: int = 3,
) -> list[Box]:
    scored = []
    for det in detections:
        if not det.categories:
            continue
        cat = det.categories[0]
        label = (cat.category_name or "").lower()
        conf = float(cat.score or 0.0)
        bb = det.bounding_box
        box = Box(bb.origin_x, bb.origin_y, bb.width, bb.height, label, conf)
        score = score_candidate_box(box, subject_name, frame_w, frame_h)
        if score < 6.0:
            continue
        scored.append((score, box))

    scored.sort(key=lambda item: item[0], reverse=True)
    selected = []
    for _, box in scored:
        if any(iou(box, existing) > 0.45 for existing in selected):
            continue
        selected.append(box)
        if len(selected) >= limit:
            break

    selected.sort(key=lambda box: box.center()[0])
    return selected


def annotate_focus_candidates(frame, candidates: list[Box]):
    annotated = frame.copy()
    for index, candidate in enumerate(candidates):
        label = chr(ord("A") + index)
        x, y, w, h = candidate.as_rect()
        draw_simple_box(
            annotated,
            x,
            y,
            w,
            h,
            (66, 123, 255),
            f"{label} {candidate.label}",
        )
    return annotated


def request_qwen_focus_choices(
    llama_url: str,
    batch_entries: list[dict],
    subject_name: str,
) -> dict[int, str]:
    prompt_lines = [
        "Choose exactly one candidate box for each frame.",
        "Goal: keep one real subject in frame for a vertical crop at all times.",
        "Prefer the person currently speaking. If speaking is unclear, choose the clearest main subject.",
        "Never choose empty space between people.",
        "Return exactly one line per frame in the format number:letter.",
        "Example:",
        "1:A",
        "2:B",
        "",
    ]

    content = [{"type": "text", "text": "\n".join(prompt_lines)}]

    for entry in batch_entries:
        candidate_text = ", ".join(
            f"{chr(ord('A') + idx)}={candidate.label}"
            for idx, candidate in enumerate(entry["candidates"])
        )
        content.append(
            {
                "type": "text",
                "text": (
                    f"Frame {entry['frame_number']} at {entry['time_sec']:.1f}s. "
                    f"Subject type: {subject_name}. Candidates: {candidate_text}."
                ),
            }
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": entry["image_url"],
                },
            }
        )

    payload = {
        "temperature": 0,
        "top_k": 1,
        "top_p": 1,
        "max_tokens": max(80, len(batch_entries) * 10),
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
    }

    request = urllib.request.Request(
        f"{llama_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raise RuntimeError(error.read().decode("utf-8", errors="ignore") or str(error)) from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Could not reach local Qwen server at {llama_url}: {error}") from error

    text = data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
    selections = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        left, right = line.split(":", 1)
        try:
            frame_number = int(left.strip())
        except ValueError:
            continue
        choice = right.strip().upper()[:1]
        if "A" <= choice <= "Z":
            selections[frame_number] = choice
    return selections


def fallback_candidate_choice(
    candidates: list[Box],
    prev_center_x: Optional[float],
    frame_w: int,
) -> int:
    if not candidates:
        return 0
    if prev_center_x is None:
        return min(
            range(len(candidates)),
            key=lambda idx: abs(candidates[idx].center()[0] - frame_w / 2.0),
        )
    return min(
        range(len(candidates)),
        key=lambda idx: abs(candidates[idx].center()[0] - prev_center_x),
    )


def build_focus_plan(
    input_path: str,
    model_path: str,
    subject_name: str,
    llama_url: Optional[str],
    sample_interval_sec: Optional[float] = None,
    batch_size: Optional[int] = None,
) -> list[FocusPlanPoint]:
    if not llama_url:
        return []

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video for focus plan: {input_path}")

    human_subject = is_human_subject(subject_name)
    if sample_interval_sec is None:
        sample_interval_sec = 0.25 if human_subject else 1.0
    if batch_size is None:
        batch_size = 4 if human_subject else 12

    detector = build_detector(model_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 23.976
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sample_every_n = max(1, int(round(fps * sample_interval_sec)))

    plan = []
    pending_entries = []
    prev_plan_center_x = None

    def flush_pending():
        nonlocal pending_entries, prev_plan_center_x, plan
        if not pending_entries:
            return
        selections = request_qwen_focus_choices(llama_url, pending_entries, subject_name)
        for entry in pending_entries:
            fallback_index = fallback_candidate_choice(
                entry["candidates"],
                prev_plan_center_x,
                width,
            )
            choice_letter = selections.get(entry["frame_number"], chr(ord("A") + fallback_index))
            choice_index = ord(choice_letter) - ord("A")
            if choice_index < 0 or choice_index >= len(entry["candidates"]):
                choice_index = fallback_index
            chosen = entry["candidates"][choice_index]
            center_x = chosen.center()[0]
            normalized_x = center_x / max(width, 1)
            snap_cut = (
                prev_plan_center_x is not None
                and abs(center_x - prev_plan_center_x) > width * 0.22
            )
            plan.append(
                FocusPlanPoint(
                    time_sec=entry["time_sec"],
                    center_x=normalized_x,
                    label=chosen.label,
                    snap_cut=snap_cut,
                )
            )
            prev_plan_center_x = center_x
        pending_entries = []

    try:
        for frame_idx, frame in iter_video_frames(cap):
            if frame_idx % sample_every_n != 0:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(frame_idx * 1000.0 / fps)
            result = detector.detect_for_video(mp_image, timestamp_ms)
            candidates = select_candidate_boxes(result.detections, subject_name, width, height)
            time_sec = frame_idx / fps

            if not candidates:
                if prev_plan_center_x is not None:
                    plan.append(
                        FocusPlanPoint(
                            time_sec=time_sec,
                            center_x=prev_plan_center_x / max(width, 1),
                            label=subject_name,
                            snap_cut=False,
                        )
                    )
                continue

            if len(candidates) == 1:
                chosen = candidates[0]
                center_x = chosen.center()[0]
                snap_cut = (
                    prev_plan_center_x is not None
                    and abs(center_x - prev_plan_center_x) > width * 0.22
                )
                plan.append(
                    FocusPlanPoint(
                        time_sec=time_sec,
                        center_x=center_x / max(width, 1),
                        label=chosen.label,
                        snap_cut=snap_cut,
                    )
                )
                prev_plan_center_x = center_x
                continue

            pending_entries.append(
                {
                    "frame_number": frame_idx,
                    "time_sec": time_sec,
                    "candidates": candidates,
                    "image_url": encode_frame_as_data_url(
                        annotate_focus_candidates(frame, candidates)
                    ),
                }
            )
            if len(pending_entries) >= batch_size:
                flush_pending()

        flush_pending()
    finally:
        cap.release()

    return plan


def translate_box_to_crop(box: Box, crop: CropWindow, scale_x: float, scale_y: float) -> Box:
    return Box(
        x=(box.x - crop.x) * scale_x,
        y=(box.y - crop.y) * scale_y,
        w=box.w * scale_x,
        h=box.h * scale_y,
        label=box.label,
        conf=box.conf,
    )


def render_source_inset(
    output_frame,
    source_frame,
    subject_box: Optional[Box],
    crop_window: Optional[CropWindow],
):
    out_h, out_w = output_frame.shape[:2]
    src_h, src_w = source_frame.shape[:2]
    if src_w <= 0 or src_h <= 0:
        return

    panel_w = int(out_w * 0.34)
    panel_max_h = int(out_h * 0.22)
    content_w, content_h = fit_inside(src_w, src_h, panel_w - 16, panel_max_h - 16)
    panel_w = content_w + 16
    panel_h = content_h + 16
    margin = 18
    panel_x = out_w - panel_w - margin
    panel_y = out_h - panel_h - margin

    overlay = output_frame.copy()
    cv2.rectangle(
        overlay,
        (panel_x - 4, panel_y - 4),
        (panel_x + panel_w + 4, panel_y + panel_h + 4),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.28, output_frame, 0.72, 0, output_frame)

    cv2.rectangle(
        output_frame,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h),
        (245, 247, 249),
        -1,
    )
    cv2.rectangle(
        output_frame,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h),
        (36, 40, 46),
        2,
    )

    inner_x = panel_x + 8
    inner_y = panel_y + 8
    resized = cv2.resize(source_frame, (content_w, content_h), interpolation=cv2.INTER_LINEAR)
    output_frame[inner_y : inner_y + content_h, inner_x : inner_x + content_w] = resized

    scale_x = content_w / src_w
    scale_y = content_h / src_h

    if crop_window is not None:
        draw_simple_box(
            output_frame,
            int(round(inner_x + crop_window.x * scale_x)),
            int(round(inner_y + crop_window.y * scale_y)),
            int(round(crop_window.w * scale_x)),
            int(round(crop_window.h * scale_y)),
            (255, 170, 60),
            "crop",
        )

    if subject_box is not None:
        draw_simple_box(
            output_frame,
            int(round(inner_x + subject_box.x * scale_x)),
            int(round(inner_y + subject_box.y * scale_y)),
            int(round(subject_box.w * scale_x)),
            int(round(subject_box.h * scale_y)),
            (66, 123, 255),
            "subject",
        )

    title = "Source View"
    (text_w, text_h), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    title_y = max(18, panel_y - 8)
    cv2.rectangle(
        output_frame,
        (panel_x, title_y - text_h - 10),
        (panel_x + text_w + 12, title_y),
        (31, 36, 31),
        -1,
    )
    cv2.putText(
        output_frame,
        title,
        (panel_x + 6, title_y - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (246, 241, 232),
        1,
        cv2.LINE_AA,
    )


def iter_video_frames(cap) -> Iterable[tuple[int, any]]:
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield idx, frame
        idx += 1


def build_detector(model_path: str):
    BaseOptions = mp.tasks.BaseOptions
    ObjectDetector = mp.tasks.vision.ObjectDetector
    ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
    RunningMode = mp.tasks.vision.RunningMode
    options = ObjectDetectorOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        score_threshold=0.12,
        max_results=10,
    )
    return ObjectDetector.create_from_options(options)


def parse_aspect_ratio(value: str) -> float:
    text = value.strip()
    if ":" in text:
        left, right = text.split(":", 1)
        ratio = float(left) / float(right)
    else:
        ratio = float(text)
    if ratio <= 0:
        raise ValueError("Aspect ratio must be positive.")
    return ratio


def compute_output_size(target_aspect: float) -> tuple[int, int]:
    if target_aspect <= 1.0:
        out_h = 1280 if target_aspect < 0.8 else 720
        out_w = max(1, int(round(out_h * target_aspect)))
    else:
        out_w = 1280
        out_h = max(1, int(round(out_w / target_aspect)))
    return out_w, out_h


def annotate_video(
    input_path: str,
    output_path: str,
    model_path: str,
    subject_name: str,
    target_aspect: Optional[float] = None,
    draw_subject_box: bool = True,
    source_inset: bool = False,
    focus_plan: Optional[list[FocusPlanPoint]] = None,
):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 23.976
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_width = width
    out_height = height

    crop_active = target_aspect is not None
    if crop_active:
        out_width, out_height = compute_output_size(target_aspect)

    temp_dir = tempfile.mkdtemp(prefix="ostrich_box_")
    temp_video = os.path.join(temp_dir, "boxed_no_audio.mp4")
    writer = cv2.VideoWriter(
        temp_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (out_width, out_height),
    )
    if not writer.isOpened():
        raise RuntimeError("Could not open output video writer.")

    detector = build_detector(model_path)
    speaker_mode = is_human_subject(subject_name)
    face_detector = build_face_detector() if speaker_mode else None
    prev_box = None
    prev_face_box = None
    missing_count = 0
    prev_center_x = None
    box_alpha = 0.28 if speaker_mode and crop_active else 0.65
    max_missing = 30 if speaker_mode and crop_active else 12
    active_plan_index = -1
    active_plan_point = None

    try:
        for frame_idx, frame in iter_video_frames(cap):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(frame_idx * 1000.0 / fps)
            current_time_sec = frame_idx / fps

            force_cut = False
            if focus_plan:
                while (
                    active_plan_index + 1 < len(focus_plan)
                    and focus_plan[active_plan_index + 1].time_sec <= current_time_sec
                ):
                    active_plan_index += 1
                    active_plan_point = focus_plan[active_plan_index]
                    force_cut = active_plan_point.snap_cut

            guided_center_x = None
            if active_plan_point is not None:
                guided_center_x = active_plan_point.center_x * width

            result = detector.detect_for_video(mp_image, timestamp_ms)
            subject_box = choose_box(
                result.detections,
                subject_name,
                prev_box,
                width,
                height,
                guided_center_x=guided_center_x,
            )
            tracking_box = derive_speaker_anchor_box(subject_box, width, height) if speaker_mode else subject_box

            if speaker_mode and face_detector is not None:
                face_boxes = detect_face_boxes(face_detector, rgb, width, height)
                face_box = choose_face_box(
                    face_boxes,
                    prev_face_box,
                    width,
                    height,
                    guided_center_x=guided_center_x,
                    subject_box=subject_box,
                )
                if face_box is not None:
                    if (
                        prev_face_box is None
                        or abs(face_box.center()[0] - prev_face_box.center()[0]) > width * 0.10
                    ):
                        prev_face_box = face_box
                        force_cut = True
                    else:
                        prev_face_box = smooth_box(prev_face_box, face_box, alpha=0.55)
                    tracking_box = expand_face_to_tracking_box(prev_face_box, width, height)
                elif subject_box is None and prev_face_box is not None and missing_count < max_missing:
                    tracking_box = expand_face_to_tracking_box(prev_face_box, width, height)
                    subject_box = tracking_box
                elif subject_box is not None:
                    prev_face_box = None

            current_box = tracking_box
            display_box = subject_box if subject_box is not None else tracking_box
            if display_box is not None:
                prev_box = smooth_box(prev_box, display_box, alpha=box_alpha)
                missing_count = 0
            elif prev_box is not None and missing_count < max_missing:
                missing_count += 1
            else:
                prev_box = None

            output_frame = frame
            focus_box = current_box if speaker_mode and current_box is not None else prev_box
            output_box = display_box if display_box is not None else focus_box
            crop = None

            if crop_active:
                if (
                    speaker_mode
                    and current_box is not None
                    and prev_center_x is not None
                    and abs(current_box.center()[0] - prev_center_x) > width * 0.18
                ):
                    force_cut = True

                crop, prev_center_x = compute_crop_window(
                    width,
                    height,
                    focus_box,
                    prev_center_x,
                    target_aspect,
                    speaker_mode=speaker_mode,
                    guided_center_x=guided_center_x,
                    force_cut=force_cut,
                )
                output_frame = crop_frame(frame, crop, out_width, out_height)
                if focus_box is not None:
                    scale_x = out_width / crop.w
                    scale_y = out_height / crop.h
                    if output_box is not None:
                        output_box = translate_box_to_crop(output_box, crop, scale_x, scale_y)
            elif prev_box is not None:
                prev_center_x = prev_box.center()[0]

            if draw_subject_box and output_box is not None:
                draw_box(output_frame, output_box, subject_name)

            if source_inset and crop_active:
                render_source_inset(output_frame, frame, display_box, crop)

            writer.write(output_frame)
    finally:
        cap.release()
        writer.release()

    mux_audio(temp_video, input_path, output_path)


def mux_audio(video_path: str, source_path: str, output_path: str):
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        video_path,
        "-i",
        source_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "21",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-profile:v",
        "high",
        "-level:v",
        "4.0",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-ar",
        "48000",
        "-shortest",
        output_path,
    ]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--subject", default="ostrich")
    parser.add_argument("--crop-vertical", action="store_true")
    parser.add_argument("--aspect-ratio")
    parser.add_argument("--no-box", action="store_true")
    parser.add_argument("--source-inset", action="store_true")
    parser.add_argument("--llama-url")
    args = parser.parse_args()

    focus_plan = build_focus_plan(
        args.input,
        args.model,
        args.subject.lower().strip(),
        args.llama_url or os.environ.get("LLAMA_BASE_URL"),
    )

    target_aspect = None
    if args.aspect_ratio:
        target_aspect = parse_aspect_ratio(args.aspect_ratio)
    elif args.crop_vertical:
        target_aspect = 9.0 / 16.0

    annotate_video(
        args.input,
        args.output,
        args.model,
        args.subject.lower().strip(),
        target_aspect=target_aspect,
        draw_subject_box=not args.no_box,
        source_inset=args.source_inset,
        focus_plan=focus_plan,
    )
    print(args.output)


if __name__ == "__main__":
    main()
