#!/usr/bin/env python3

import argparse
import base64
import json
import math
import os
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Iterable, Optional

import cv2
import mediapipe as mp
import pytesseract

FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")
FOCUS_STYLE_RULES_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "config", "focus-style-rules.md")
)
FOCUS_STYLE_SETTINGS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "config", "focus-style-settings.json")
)
DEFAULT_FOCUS_STYLE_RULES = """BIG look (default)

- BIG is the law. Use BIG for almost the whole video.
- BIG means a tight center crop that fills the full 9:16 screen.
- Use BIG for talking-head sections, single speakers, reactions, and emotional moments.
- Use BIG whenever the main subject is clear and the important content still survives inside the crop.
- If unsure, stay in BIG.

MID look (rare exception)

- MID is earned. MID must be justified.
- Use MID only when BIG would physically cut off important content or make it unclear.
- Use MID only for the exact duration of that moment.
- As soon as the reason is gone, go back to BIG immediately.

Valid reasons for MID

- A physical object being shown would be cropped in BIG, such as a product, shirt, phone, book, or other item.
- A second person who actually matters in that moment would be cut out in BIG.
- The environment is the point of the shot, such as revealing a room, store, setup, or location.
- Important on-screen text, graphics, charts, prices, labels, or captions would be cropped or become unclear in BIG.
- A hand gesture is pointing to something that would be lost in BIG.

What must not trigger MID

- Text existing somewhere on screen is not enough.
- Two people merely being present is not enough.
- Background or environment being visible is not enough.
- A wide-looking source frame is not enough.
- Model uncertainty is not enough. Uncertainty defaults to BIG.
- No human visible is not enough by itself.

Decision rule

- Mentally apply the BIG crop first.
- Ask what important content would actually be lost in BIG.
- If the answer is nothing or only background, choose BIG.
- If the answer matches one of the valid MID reasons above, choose MID for that moment only.
- Never let MID become the default style.
"""
DEFAULT_FOCUS_STYLE_SETTINGS = {
    "default": {
        "sample_interval_sec": 6.0,
        "batch_size": 4,
        "ai_per_frame": False,
        "scene_change_threshold": 24.0,
        "scene_change_min_gap_sec": 0.9,
        "text_check_interval_sec": 0.25,
        "text_hold_sec": 0.5,
    },
    "speaker": {
        "sample_interval_sec": 4.0,
        "batch_size": 4,
        "ai_per_frame": False,
        "scene_change_threshold": 16.0,
        "scene_change_min_gap_sec": 0.75,
        "text_check_interval_sec": 0.2,
        "text_hold_sec": 0.65,
    },
}

PROGRESS_PREFIX = "__PROGRESS__"


def emit_progress(
    stage: str,
    current: Optional[int] = None,
    total: Optional[int] = None,
    message: str = "",
    eta_seconds: Optional[float] = None,
    **extra,
) -> None:
    payload = {
        "stage": stage,
        "message": message,
    }
    if current is not None:
        payload["current"] = int(max(current, 0))
    if total is not None and total > 0:
        payload["total"] = int(total)
    if "current" in payload and "total" in payload and payload["total"] > 0:
        payload["ratio"] = max(0.0, min(1.0, payload["current"] / payload["total"]))
    if eta_seconds is not None and math.isfinite(eta_seconds):
        payload["eta_seconds"] = max(0.0, float(eta_seconds))
    for key, value in extra.items():
        if value is not None:
            payload[key] = value
    print(f"{PROGRESS_PREFIX}{json.dumps(payload, separators=(',', ':'))}", flush=True)


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
    prefer_wide: bool = False


@dataclass
class SpeakerCue:
    face_box: Box
    score: float


@dataclass
class LayoutMetrics:
    use_wide: bool
    subject_span_ratio: float
    candidate_count: int


def load_focus_style_rules_text() -> str:
    try:
        with open(FOCUS_STYLE_RULES_PATH, "r", encoding="utf-8") as handle:
            text = handle.read().strip()
            if text:
                return text
    except OSError:
        pass
    return DEFAULT_FOCUS_STYLE_RULES


def build_layout_system_prompt(style_rules: str) -> str:
    return "\n".join(
        [
            "You are the editorial layout controller for vertical 9:16 reframing.",
            "Choose between two layouts only:",
            "- BIG = fullscreen_crop",
            "- MID = center_with_blur_bg",
            "",
            "Default behavior:",
            "- BIG is the default for almost the whole video.",
            "- MID is an exception and must be rare.",
            "- If you are unsure, choose BIG.",
            "",
            "Core rule:",
            "- Mentally apply the BIG crop first.",
            "- Ask what important content would actually be lost in BIG.",
            "- If nothing important would be lost, choose BIG.",
            "- Choose MID only when BIG would physically cut off important content or make it unclear.",
            "- As soon as that risk is gone, go back to BIG.",
            "- Text somewhere on screen is not enough by itself.",
            "- Two people merely being present is not enough by itself.",
            "- Background or a wide-looking frame is not enough by itself.",
            "- No human visible is not enough by itself.",
            "",
            "Project editorial rules:",
            style_rules,
            "",
            "Output rules:",
            "- Return exactly one line per frame.",
            "- Format: number:BIG or number:MID",
            "- Do not add explanations.",
        ]
    )


def is_context_limit_error(message: str) -> bool:
    text = (message or "").lower()
    return "exceeds the available context size" in text or "exceed_context_size_error" in text


def request_choices_with_backoff(
    request_fn,
    llama_url: str,
    batch_entries: list[dict],
    subject_name: str,
) -> dict[int, str]:
    if not batch_entries:
        return {}

    try:
        return request_fn(llama_url, batch_entries, subject_name)
    except RuntimeError as error:
        if len(batch_entries) <= 1 or not is_context_limit_error(str(error)):
            raise

    mid = max(1, len(batch_entries) // 2)
    left = request_choices_with_backoff(request_fn, llama_url, batch_entries[:mid], subject_name)
    right = request_choices_with_backoff(request_fn, llama_url, batch_entries[mid:], subject_name)
    merged = {}
    merged.update(left)
    merged.update(right)
    return merged


def load_focus_style_settings(human_subject: bool) -> dict:
    settings = json.loads(json.dumps(DEFAULT_FOCUS_STYLE_SETTINGS))
    try:
        with open(FOCUS_STYLE_SETTINGS_PATH, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, dict):
            for key, value in loaded.items():
                if key not in settings or not isinstance(value, dict):
                    continue
                settings[key].update(value)
    except (OSError, json.JSONDecodeError):
        pass

    profile = settings["speaker" if human_subject else "default"].copy()
    merged = settings["default"].copy()
    merged.update(profile)
    return merged


def compute_scene_change_score(prev_small_gray, curr_small_gray) -> float:
    if prev_small_gray is None or curr_small_gray is None:
        return 0.0
    diff = cv2.absdiff(curr_small_gray, prev_small_gray)
    return float(diff.mean())


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


def build_face_mesh():
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=4,
        refine_landmarks=False,
        min_detection_confidence=0.35,
        min_tracking_confidence=0.35,
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


def detect_speaker_cues(face_mesh, rgb_frame, frame_w: int, frame_h: int) -> list[SpeakerCue]:
    result = face_mesh.process(rgb_frame)
    faces = result.multi_face_landmarks or []
    cues = []

    for face in faces:
        xs = [point.x * frame_w for point in face.landmark]
        ys = [point.y * frame_h for point in face.landmark]
        if not xs or not ys:
            continue

        min_x = clamp(min(xs), 0, frame_w - 1)
        min_y = clamp(min(ys), 0, frame_h - 1)
        max_x = clamp(max(xs), 0, frame_w)
        max_y = clamp(max(ys), 0, frame_h)
        face_w = max(max_x - min_x, 1.0)
        face_h = max(max_y - min_y, 1.0)
        mouth_gap = abs(face.landmark[13].y - face.landmark[14].y) * frame_h
        mouth_width = abs(face.landmark[78].x - face.landmark[308].x) * frame_w
        cue_score = max(mouth_gap / face_h, 0.0) + max(mouth_gap / max(mouth_width, 1.0), 0.0) * 0.35

        cues.append(
            SpeakerCue(
                face_box=Box(
                    x=min_x,
                    y=min_y,
                    w=face_w,
                    h=face_h,
                    label="face",
                    conf=cue_score,
                ),
                score=cue_score,
            )
        )

    return cues


def score_candidate_speaker_cue(candidate: Box, speaker_cues: list[SpeakerCue]) -> float:
    best_score = 0.0
    cand_x2 = candidate.x + candidate.w
    cand_y2 = candidate.y + candidate.h

    for cue in speaker_cues:
        center_x, center_y = cue.face_box.center()
        if candidate.x <= center_x <= cand_x2 and candidate.y <= center_y <= cand_y2:
            best_score = max(best_score, cue.score)

    return best_score


def choose_dominant_speaker_index(speaker_scores: list[float]) -> Optional[int]:
    if len(speaker_scores) < 2:
        return 0 if speaker_scores else None

    ranked = sorted(enumerate(speaker_scores), key=lambda item: item[1], reverse=True)
    best_index, best_score = ranked[0]
    second_score = ranked[1][1]

    if best_score >= 0.085 and best_score - second_score >= 0.018:
        return best_index

    return None


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


def extract_crop_region(frame, crop: CropWindow):
    x1 = int(clamp(crop.x, 0, frame.shape[1] - crop.w))
    y1 = int(clamp(crop.y, 0, frame.shape[0] - crop.h))
    x2 = x1 + int(crop.w)
    y2 = y1 + int(crop.h)
    return frame[y1:y2, x1:x2]


def resize_to_cover(frame, output_w: int, output_h: int):
    src_h, src_w = frame.shape[:2]
    scale = max(output_w / max(src_w, 1), output_h / max(src_h, 1))
    resized_w = max(1, int(round(src_w * scale)))
    resized_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    offset_x = max(0, (resized_w - output_w) // 2)
    offset_y = max(0, (resized_h - output_h) // 2)
    return resized[offset_y : offset_y + output_h, offset_x : offset_x + output_w]


def resize_to_fit(frame, max_w: int, max_h: int):
    src_h, src_w = frame.shape[:2]
    scale = min(max_w / max(src_w, 1), max_h / max(src_h, 1))
    resized_w = max(1, int(round(src_w * scale)))
    resized_h = max(1, int(round(src_h * scale)))
    return cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)


def build_blurred_background(frame, output_w: int, output_h: int):
    background = resize_to_cover(frame, output_w, output_h)
    background = cv2.GaussianBlur(background, (0, 0), 28)
    background = cv2.GaussianBlur(background, (0, 0), 18)
    background = cv2.convertScaleAbs(background, alpha=0.82, beta=-6)
    return background


def compose_blurred_fill_frame(
    frame,
    crop: CropWindow,
    output_w: int,
    output_h: int,
):
    crop_region = extract_crop_region(frame, crop)
    background = build_blurred_background(crop_region, output_w, output_h)

    fg_w = output_w
    fg_h = max(1, int(round(output_w * crop.h / max(crop.w, 1))))
    if fg_h > output_h:
        foreground = resize_to_fit(crop_region, output_w, output_h)
        fg_h, fg_w = foreground.shape[:2]
    else:
        foreground = cv2.resize(crop_region, (fg_w, fg_h), interpolation=cv2.INTER_LINEAR)

    offset_x = max(0, (output_w - fg_w) // 2)
    offset_y = max(0, (output_h - fg_h) // 2)

    composed = background.copy()
    composed[offset_y : offset_y + fg_h, offset_x : offset_x + fg_w] = foreground
    return composed, fg_w / crop.w, fg_h / crop.h, offset_x, offset_y


def compose_center_with_blur_bg_frame(frame, output_w: int, output_h: int):
    background = build_blurred_background(frame, output_w, output_h)
    foreground = resize_to_fit(frame, output_w, output_h)
    fg_h, fg_w = foreground.shape[:2]
    offset_x = max(0, (output_w - fg_w) // 2)
    offset_y = max(0, (output_h - fg_h) // 2)

    composed = background.copy()
    composed[offset_y : offset_y + fg_h, offset_x : offset_x + fg_w] = foreground
    return composed, fg_w / max(frame.shape[1], 1), fg_h / max(frame.shape[0], 1), offset_x, offset_y


def fit_inside(src_w: int, src_h: int, max_w: int, max_h: int) -> tuple[int, int]:
    if src_w <= 0 or src_h <= 0:
        return max_w, max_h
    scale = min(max_w / src_w, max_h / src_h)
    return max(1, int(round(src_w * scale))), max(1, int(round(src_h * scale)))


def encode_frame_as_data_url(
    frame,
    jpeg_quality: int = 58,
    max_w: int = 224,
    max_h: int = 224,
) -> str:
    src_h, src_w = frame.shape[:2]
    if src_w > max_w or src_h > max_h:
        frame = resize_to_fit(frame, max_w, max_h)
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if not ok:
        raise RuntimeError("Failed to encode guidance frame for Qwen.")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def normalize_ocr_token(text: str) -> str:
    return "".join(char for char in (text or "") if char.isalnum())


def detect_text_regions(frame) -> list[Box]:
    probe = resize_to_fit(frame, 960, 540)
    gray = cv2.cvtColor(probe, cv2.COLOR_BGR2GRAY)
    processed = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )
    data = pytesseract.image_to_data(
        processed,
        output_type=pytesseract.Output.DICT,
        config="--psm 11",
    )

    scale_x = frame.shape[1] / max(probe.shape[1], 1)
    scale_y = frame.shape[0] / max(probe.shape[0], 1)
    regions = []
    for idx, raw_text in enumerate(data.get("text", [])):
        token = normalize_ocr_token(raw_text)
        if not token:
            continue
        try:
            conf = float(data["conf"][idx])
        except (ValueError, TypeError, KeyError):
            conf = -1.0
        if conf < 45.0:
            continue
        width = max(0, int(data["width"][idx]))
        height = max(0, int(data["height"][idx]))
        if width * height < 120:
            continue
        if len(token) < 3 and conf < 65.0:
            continue
        x = max(0, int(round(int(data["left"][idx]) * scale_x)))
        y = max(0, int(round(int(data["top"][idx]) * scale_y)))
        w = max(1, int(round(width * scale_x)))
        h = max(1, int(round(height * scale_y)))
        regions.append(Box(x=x, y=y, w=w, h=h, label="text", conf=conf))
    return regions


def detect_text_presence(frame) -> bool:
    return bool(detect_text_regions(frame))


def text_requires_mid(frame, crop: CropWindow) -> bool:
    regions = detect_text_regions(frame)
    if not regions:
        return False

    crop_left = crop.x
    crop_right = crop.x + crop.w
    crop_margin = crop.w * 0.03

    for region in regions:
        text_left = region.x
        text_right = region.x + region.w
        if text_left < crop_left or text_right > crop_right:
            return True
        if text_left < crop_left + crop_margin or text_right > crop_right - crop_margin:
            return True
    return False


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


def annotate_layout_preview(
    frame,
    crop: Optional[CropWindow],
    chosen_box: Optional[Box],
    candidates: list[Box],
):
    annotated = frame.copy()
    if crop is not None:
        overlay = annotated.copy()
        x1 = int(round(crop.x))
        y1 = int(round(crop.y))
        x2 = int(round(crop.x + crop.w))
        y2 = int(round(crop.y + crop.h))
        cv2.rectangle(overlay, (0, 0), (annotated.shape[1], annotated.shape[0]), (0, 0, 0), -1)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.28, annotated, 0.72, 0, annotated)

    for candidate in candidates:
        x, y, w, h = candidate.as_rect()
        draw_simple_box(annotated, x, y, w, h, (66, 123, 255), "")

    if chosen_box is not None:
        x, y, w, h = chosen_box.as_rect()
        draw_simple_box(annotated, x, y, w, h, (66, 123, 255), "subject")

    if crop is not None:
        draw_simple_box(
            annotated,
            int(round(crop.x)),
            int(round(crop.y)),
            int(round(crop.w)),
            int(round(crop.h)),
            (255, 170, 60),
            "crop preview",
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
        "In dialogue scenes, do not choose the listener when the speaker is visible.",
        "Use higher speaker cue values as evidence of who is speaking when those values are clearly different.",
        "If the speaking cue is close or unclear, keep continuity with the previous speaker instead of switching randomly.",
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
            f"{chr(ord('A') + idx)}={candidate.label} cue={entry['speaker_scores'][idx]:.3f}"
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
                    "url": entry.get("focus_image_url") or entry["image_url"],
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


def request_layout_choices(
    llama_url: str,
    batch_entries: list[dict],
    subject_name: str,
) -> dict[int, str]:
    style_rules = load_focus_style_rules_text()
    prompt_lines = [
        "Decide BIG or MID for each frame.",
        "The image shows the source frame with the BIG crop preview drawn on it.",
        "BIG is the default answer.",
        "If the crop preview keeps the important content, choose BIG.",
        "Choose MID only when the crop preview would physically cut off important content or make it unclear.",
        "Text alone does not justify MID. Only cropped or unclear important text justifies MID.",
        "No human visible does not justify MID by itself.",
        "Return exactly one line per frame in the format number:BIG or number:MID.",
        "Example:",
        "1:BIG",
        "2:MID",
        "",
    ]

    content = [{"type": "text", "text": "\n".join(prompt_lines)}]

    for entry in batch_entries:
        content.append(
            {
                "type": "text",
                "text": f"Frame {entry['frame_number']} at {entry['time_sec']:.1f}s.",
            }
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": entry["layout_image_url"],
                },
            }
        )

    payload = {
        "temperature": 0,
        "top_k": 1,
        "top_p": 1,
        "max_tokens": max(80, len(batch_entries) * 8),
        "messages": [
            {
                "role": "system",
                "content": build_layout_system_prompt(style_rules),
            },
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
        choice_text = right.strip().upper()
        if choice_text.startswith("MID") or choice_text.startswith("W"):
            selections[frame_number] = "MID"
        elif choice_text.startswith("BIG") or choice_text.startswith("C"):
            selections[frame_number] = "BIG"
    return selections


def fallback_candidate_choice(
    candidates: list[Box],
    prev_center_x: Optional[float],
    frame_w: int,
    speaker_scores: Optional[list[float]] = None,
) -> int:
    if not candidates:
        return 0

    dominant_speaker_index = choose_dominant_speaker_index(speaker_scores or [])
    if dominant_speaker_index is not None:
        return dominant_speaker_index

    if prev_center_x is None:
        return max(
            range(len(candidates)),
            key=lambda idx: (speaker_scores[idx] if speaker_scores else 0.0) * 120.0
            - abs(candidates[idx].center()[0] - frame_w / 2.0),
        )
    return max(
        range(len(candidates)),
        key=lambda idx: (speaker_scores[idx] if speaker_scores else 0.0) * 120.0
        - abs(candidates[idx].center()[0] - prev_center_x),
    )


def union_boxes(boxes: list[Box]) -> Optional[Box]:
    if not boxes:
        return None
    min_x = min(box.x for box in boxes)
    min_y = min(box.y for box in boxes)
    max_x = max(box.x + box.w for box in boxes)
    max_y = max(box.y + box.h for box in boxes)
    exemplar = boxes[0]
    return Box(
        x=min_x,
        y=min_y,
        w=max_x - min_x,
        h=max_y - min_y,
        label=exemplar.label,
        conf=exemplar.conf,
    )


def choose_layout_metrics(
    candidates: list[Box],
    subject_box: Optional[Box],
    crop_w: float,
    frame_w: int,
    frame_h: int,
    speaker_mode: bool,
    speaker_visible: bool = True,
    model_prefers_wide: bool = False,
) -> LayoutMetrics:
    if crop_w >= frame_w * 0.98 or not candidates:
        return LayoutMetrics(use_wide=False, subject_span_ratio=0.0, candidate_count=len(candidates))

    min_area_ratio = 0.018 if speaker_mode else 0.012
    relevant = [
        candidate
        for candidate in candidates
        if candidate.area() / float(max(frame_w * frame_h, 1)) >= min_area_ratio
    ]
    if subject_box is not None and not any(iou(subject_box, candidate) > 0.18 for candidate in relevant):
        relevant.append(subject_box)
    relevant = sorted(relevant, key=lambda box: box.center()[0])[:3]

    if len(relevant) < 2:
        return LayoutMetrics(use_wide=False, subject_span_ratio=0.0, candidate_count=len(relevant))

    group_box = union_boxes(relevant)
    if group_box is None:
        return LayoutMetrics(use_wide=False, subject_span_ratio=0.0, candidate_count=len(relevant))

    required_crop_w = group_box.w + max(group_box.w * 0.16, frame_w * 0.06)
    required_ratio = required_crop_w / max(crop_w, 1.0)
    subject_span_ratio = group_box.w / max(frame_w, 1)
    edge_margin_ratio = min(group_box.x, frame_w - (group_box.x + group_box.w)) / max(frame_w, 1)
    max_gap_ratio = 0.0
    if len(relevant) >= 2:
        max_gap_ratio = max(
            abs(relevant[idx + 1].center()[0] - relevant[idx].center()[0]) / max(frame_w, 1)
            for idx in range(len(relevant) - 1)
        )

    subject_share = 1.0
    if subject_box is not None and group_box.w > 0:
        subject_share = subject_box.w / group_box.w

    wide_score = 0.0
    if subject_span_ratio >= 0.60:
        wide_score += 0.60
    if subject_span_ratio >= 0.72:
        wide_score += 0.80
    if required_ratio >= 1.05:
        wide_score += 1.00
    if required_ratio >= 1.15:
        wide_score += 1.20
    if max_gap_ratio >= 0.38:
        wide_score += 0.60
    if edge_margin_ratio <= 0.04:
        wide_score += 0.40
    if speaker_mode and subject_share < 0.72:
        wide_score += 0.25

    use_wide = False
    if model_prefers_wide:
        use_wide = (
            required_ratio >= 1.02
            or subject_span_ratio >= 0.54
            or max_gap_ratio >= 0.32
        )
    elif speaker_mode:
        use_wide = False
    elif not speaker_mode:
        use_wide = wide_score >= 2.80

    return LayoutMetrics(use_wide=use_wide, subject_span_ratio=subject_span_ratio, candidate_count=len(relevant))


def compute_focus_plan_mid_ratio(plan: list[FocusPlanPoint], total_duration_sec: float) -> float:
    if not plan or total_duration_sec <= 0:
        return 0.0

    mid_duration = 0.0
    for index, point in enumerate(plan):
        start = max(0.0, point.time_sec)
        end = total_duration_sec
        if index + 1 < len(plan):
            end = max(start, plan[index + 1].time_sec)
        if point.prefer_wide:
            mid_duration += max(0.0, end - start)
    return mid_duration / total_duration_sec


def enforce_big_default_focus_plan(
    plan: list[FocusPlanPoint],
    total_duration_sec: float,
    max_mid_ratio: float = 0.20,
) -> list[FocusPlanPoint]:
    if not plan:
        return plan

    ratio = compute_focus_plan_mid_ratio(plan, total_duration_sec)
    if ratio <= max_mid_ratio:
        return plan

    adjusted = [
        FocusPlanPoint(
            time_sec=point.time_sec,
            center_x=point.center_x,
            label=point.label,
            snap_cut=point.snap_cut,
            prefer_wide=point.prefer_wide,
        )
        for point in plan
    ]

    in_mid_run = False
    for point in adjusted:
        if not point.prefer_wide:
            in_mid_run = False
            continue
        if in_mid_run:
            point.prefer_wide = False
        else:
            in_mid_run = True

    ratio = compute_focus_plan_mid_ratio(adjusted, total_duration_sec)
    if ratio <= max_mid_ratio:
        return adjusted

    for point in adjusted:
        point.prefer_wide = False
    return adjusted


def log_focus_plan_layout(plan: list[FocusPlanPoint]) -> None:
    mid_count = sum(1 for point in plan if point.prefer_wide)
    total = len(plan)
    ratio = mid_count / max(total, 1)
    print(f"[Layout] Plan: {total} points, MID={mid_count}, BIG={total - mid_count}, ratio={ratio:.1%}")


def build_focus_plan(
    input_path: str,
    model_path: str,
    subject_name: str,
    llama_url: Optional[str],
    target_aspect: Optional[float] = None,
    sample_interval_sec: Optional[float] = None,
    batch_size: Optional[int] = None,
    scene_change_threshold: Optional[float] = None,
    scene_change_min_gap_sec: Optional[float] = None,
) -> list[FocusPlanPoint]:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video for focus plan: {input_path}")

    human_subject = is_human_subject(subject_name)
    style_settings = load_focus_style_settings(human_subject)
    per_frame_ai = bool(llama_url) and bool(style_settings.get("ai_per_frame", False))
    use_ai_layout = bool(llama_url)
    if sample_interval_sec is None:
        sample_interval_sec = float(style_settings["sample_interval_sec"])
    if batch_size is None:
        batch_size = int(style_settings["batch_size"])
    if scene_change_threshold is None:
        scene_change_threshold = float(style_settings["scene_change_threshold"])
    if scene_change_min_gap_sec is None:
        scene_change_min_gap_sec = float(style_settings["scene_change_min_gap_sec"])

    detector = build_detector(model_path)
    face_mesh = build_face_mesh() if human_subject else None
    fps = cap.get(cv2.CAP_PROP_FPS) or 23.976
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sample_every_n = 1 if per_frame_ai else max(1, int(round(fps * sample_interval_sec)))
    text_check_every_n = max(1, int(round(fps * max(float(style_settings["text_check_interval_sec"]), 0.05))))
    text_hold_frames = max(1, int(round(fps * max(float(style_settings["text_hold_sec"]), 0.05))))
    text_probe_every_n = max(1, int(round(fps * 0.12)))
    preview_aspect = target_aspect or (9.0 / 16.0)

    plan = []
    pending_entries = []
    prev_plan_center_x = None
    prev_small_gray = None
    last_scene_sample_time = -10**9
    text_hold_until_frame = -1
    planning_started_at = time.monotonic()
    last_progress_emit_at = 0.0

    emit_progress(
        "planning",
        current=0,
        total=frame_count if frame_count > 0 else None,
        message="Planning BIG/MID layout with AI.",
    )

    def append_plan_point(time_sec: float, center_x: float, label: str, prefer_wide: bool):
        nonlocal prev_plan_center_x, plan
        snap_cut = (
            prev_plan_center_x is not None
            and abs(center_x - prev_plan_center_x) > width * 0.22
        )
        plan.append(
            FocusPlanPoint(
                time_sec=time_sec,
                center_x=center_x / max(width, 1),
                label=label,
                snap_cut=snap_cut,
                prefer_wide=prefer_wide,
            )
        )
        prev_plan_center_x = center_x

    def should_query_layout_ai(entry: dict, candidates: list[Box]) -> bool:
        if not use_ai_layout:
            return False
        if entry.get("text_visible", False):
            return True
        if len(candidates) > 1:
            return True
        if entry.get("scene_change_triggered", False):
            return True
        return False

    def flush_pending():
        nonlocal pending_entries
        if not pending_entries:
            return

        focus_entries = [entry for entry in pending_entries if len(entry["candidates"]) > 1]
        if use_ai_layout:
            for entry in focus_entries:
                if entry.get("focus_image_url"):
                    continue
                entry["focus_image_url"] = encode_frame_as_data_url(
                    annotate_focus_candidates(entry["frame"], entry["candidates"])
                )
        focus_selections = (
            request_choices_with_backoff(
                request_qwen_focus_choices,
                llama_url,
                focus_entries,
                subject_name,
            )
            if (use_ai_layout and not per_frame_ai and focus_entries)
            else {}
        )

        layout_entries = []

        for entry in pending_entries:
            candidates = entry["candidates"]
            if not candidates:
                if per_frame_ai:
                    preview_crop, _ = compute_crop_window(
                        width,
                        height,
                        None,
                        prev_plan_center_x,
                        preview_aspect,
                        speaker_mode=human_subject,
                        guided_center_x=None,
                        force_cut=False,
                    )
                    layout_entries.append(
                        {
                            **entry,
                            "chosen": None,
                            "preview_crop": preview_crop,
                            "layout_image_url": encode_frame_as_data_url(
                                annotate_layout_preview(
                                    entry["frame"],
                                    preview_crop,
                                    None,
                                    candidates,
                                )
                            ),
                        }
                    )
                    continue

                if not entry.get("text_visible", False):
                    if prev_plan_center_x is not None:
                        append_plan_point(entry["time_sec"], prev_plan_center_x, subject_name, False)
                    else:
                        append_plan_point(entry["time_sec"], width / 2.0, subject_name, False)
                    continue

                preview_crop, _ = compute_crop_window(
                    width,
                    height,
                    None,
                    prev_plan_center_x,
                    preview_aspect,
                    speaker_mode=human_subject,
                    guided_center_x=None,
                    force_cut=False,
                )
                prefer_wide = text_requires_mid(entry["frame"], preview_crop)
                append_plan_point(
                    entry["time_sec"],
                    prev_plan_center_x if prev_plan_center_x is not None else width / 2.0,
                    subject_name,
                    prefer_wide,
                )
                continue

            chosen = None
            if len(candidates) == 1:
                chosen = candidates[0]
            else:
                fallback_index = fallback_candidate_choice(
                    candidates,
                    prev_plan_center_x,
                    width,
                    speaker_scores=entry["speaker_scores"],
                )
                dominant_speaker_index = choose_dominant_speaker_index(entry["speaker_scores"])
                choice_letter = focus_selections.get(entry["frame_number"], chr(ord("A") + fallback_index))
                choice_index = ord(choice_letter) - ord("A")
                if choice_index < 0 or choice_index >= len(candidates):
                    choice_index = fallback_index
                if dominant_speaker_index is not None:
                    choice_index = dominant_speaker_index
                chosen = candidates[choice_index]

            preview_focus_box = derive_speaker_anchor_box(chosen, width, height) if human_subject else chosen
            preview_crop, _ = compute_crop_window(
                width,
                height,
                preview_focus_box,
                prev_plan_center_x,
                preview_aspect,
                speaker_mode=human_subject,
                guided_center_x=None,
                force_cut=False,
            )
            if per_frame_ai:
                layout_entries.append(
                    {
                        **entry,
                        "chosen": chosen,
                        "preview_crop": preview_crop,
                        "layout_image_url": encode_frame_as_data_url(
                            annotate_layout_preview(
                                entry["frame"],
                                preview_crop,
                                chosen,
                                candidates,
                            )
                        ),
                    }
                )
                continue
            if not should_query_layout_ai(entry, candidates):
                append_plan_point(
                    entry["time_sec"],
                    chosen.center()[0],
                    chosen.label,
                    False,
                )
                continue
            layout_entries.append(
                {
                    **entry,
                    "chosen": chosen,
                    "preview_crop": preview_crop,
                    "layout_image_url": encode_frame_as_data_url(
                        annotate_layout_preview(
                            entry["frame"],
                            preview_crop,
                            chosen,
                            candidates,
                        )
                    ),
                }
            )

        layout_selections = (
            request_choices_with_backoff(
                request_layout_choices,
                llama_url,
                layout_entries,
                subject_name,
            )
            if use_ai_layout and layout_entries
            else {}
        )

        for entry in layout_entries:
            prefer_wide = layout_selections.get(entry["frame_number"], "BIG") == "MID"
            chosen = entry["chosen"]
            center_x = chosen.center()[0] if chosen is not None else (prev_plan_center_x if prev_plan_center_x is not None else width / 2.0)
            label = chosen.label if chosen is not None else subject_name
            append_plan_point(entry["time_sec"], center_x, label, prefer_wide)

        pending_entries = []

    try:
        for frame_idx, frame in iter_video_frames(cap):
            now = time.monotonic()
            if frame_count > 0 and (
                frame_idx == 0
                or frame_idx + 1 >= frame_count
                or now - last_progress_emit_at >= 0.35
            ):
                ratio = (frame_idx + 1) / max(frame_count, 1)
                elapsed = now - planning_started_at
                eta_seconds = (elapsed / max(ratio, 1e-6) - elapsed) if ratio > 0.001 else None
                emit_progress(
                    "planning",
                    current=frame_idx + 1,
                    total=frame_count,
                    message="Planning BIG/MID layout with AI.",
                    eta_seconds=eta_seconds,
                )
                last_progress_emit_at = now

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            small_gray = cv2.cvtColor(
                cv2.resize(rgb, (96, 54), interpolation=cv2.INTER_AREA),
                cv2.COLOR_RGB2GRAY,
            )
            time_sec = frame_idx / fps
            scene_change_score = compute_scene_change_score(prev_small_gray, small_gray)
            prev_small_gray = small_gray

            if not per_frame_ai and frame_idx % text_check_every_n == 0:
                try:
                    text_visible = detect_text_presence(frame)
                except pytesseract.TesseractError:
                    text_visible = False
                if text_visible:
                    text_hold_until_frame = frame_idx + text_hold_frames
            elif not per_frame_ai and frame_idx <= text_hold_until_frame:
                text_visible = True
            else:
                text_visible = False

            interval_triggered = frame_idx % sample_every_n == 0
            scene_change_triggered = (
                scene_change_threshold > 0
                and scene_change_score >= scene_change_threshold
                and (time_sec - last_scene_sample_time) >= scene_change_min_gap_sec
            )
            text_triggered = (not per_frame_ai) and text_visible and frame_idx % text_probe_every_n == 0
            if not interval_triggered and not scene_change_triggered and not text_triggered:
                continue
            if scene_change_triggered:
                last_scene_sample_time = time_sec

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(frame_idx * 1000.0 / fps)
            result = detector.detect_for_video(mp_image, timestamp_ms)
            candidates = select_candidate_boxes(result.detections, subject_name, width, height)
            speaker_scores = [0.0 for _ in candidates]
            speaker_cues = []

            if human_subject and face_mesh is not None and candidates:
                speaker_cues = detect_speaker_cues(face_mesh, rgb, width, height)
                speaker_scores = [
                    score_candidate_speaker_cue(candidate, speaker_cues) for candidate in candidates
                ]
            elif human_subject and face_mesh is not None:
                speaker_cues = detect_speaker_cues(face_mesh, rgb, width, height)

            human_visible = bool(speaker_cues) or any(
                candidate.label in HUMAN_LABELS for candidate in candidates
            )
            pending_entries.append(
                {
                    "frame_number": frame_idx,
                    "time_sec": time_sec,
                    "frame": frame.copy(),
                    "candidates": candidates,
                    "speaker_scores": speaker_scores,
                    "candidate_count": len(candidates),
                    "human_visible": human_visible,
                    "text_visible": text_visible,
                    "scene_change_score": scene_change_score,
                    "scene_change_triggered": scene_change_triggered,
                }
            )
            if len(pending_entries) >= batch_size:
                flush_pending()

        flush_pending()
    finally:
        cap.release()

    emit_progress(
        "planning",
        current=frame_count if frame_count > 0 else None,
        total=frame_count if frame_count > 0 else None,
        message="Layout plan ready.",
        eta_seconds=0.0,
    )
    total_duration_sec = frame_count / fps if frame_count > 0 and fps > 0 else (plan[-1].time_sec if plan else 0.0)
    focus_plan = enforce_big_default_focus_plan(plan, total_duration_sec)
    log_focus_plan_layout(focus_plan)
    if compute_focus_plan_mid_ratio(focus_plan, total_duration_sec) > 0.20:
        raise RuntimeError("Layout plan is still too MID-heavy after enforcement.")
    return focus_plan


def translate_box_to_crop(box: Box, crop: CropWindow, scale_x: float, scale_y: float) -> Box:
    return Box(
        x=(box.x - crop.x) * scale_x,
        y=(box.y - crop.y) * scale_y,
        w=box.w * scale_x,
        h=box.h * scale_y,
        label=box.label,
        conf=box.conf,
    )


def translate_box_with_scale(
    box: Box,
    scale_x: float,
    scale_y: float,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> Box:
    return Box(
        x=box.x * scale_x + offset_x,
        y=box.y * scale_y + offset_y,
        w=box.w * scale_x,
        h=box.h * scale_y,
        label=box.label,
        conf=box.conf,
    )


def blend_frames(frame_a, frame_b, alpha: float):
    if alpha <= 0.0:
        return frame_a
    if alpha >= 1.0:
        return frame_b
    return cv2.addWeighted(frame_a, 1.0 - alpha, frame_b, alpha, 0.0)


def blend_boxes(box_a: Optional[Box], box_b: Optional[Box], alpha: float) -> Optional[Box]:
    if box_a is None:
        return box_b
    if box_b is None:
        return box_a
    if alpha <= 0.0:
        return box_a
    if alpha >= 1.0:
        return box_b
    return Box(
        x=box_a.x * (1.0 - alpha) + box_b.x * alpha,
        y=box_a.y * (1.0 - alpha) + box_b.y * alpha,
        w=box_a.w * (1.0 - alpha) + box_b.w * alpha,
        h=box_a.h * (1.0 - alpha) + box_b.h * alpha,
        label=box_b.label,
        conf=box_a.conf * (1.0 - alpha) + box_b.conf * alpha,
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
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_width = width
    out_height = height

    crop_active = target_aspect is not None
    use_blurred_fill = crop_active and target_aspect is not None and target_aspect < 0.8
    crop_target_aspect = target_aspect
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
    crop_width = min(width, int(round(height * crop_target_aspect))) if crop_active else width
    transition_frames = max(1, int(round(fps * 0.22))) if use_blurred_fill else 1
    wide_enter_frames = max(2, int(round(fps * 0.18))) if use_blurred_fill else 1
    wide_exit_frames = max(wide_enter_frames + 1, int(round(fps * 0.28))) if use_blurred_fill else 1
    wide_target = False
    wide_enter_streak = 0
    wide_exit_streak = 0
    layout_blend = 0.0
    render_started_at = time.monotonic()
    last_render_emit_at = 0.0
    emit_progress(
        "rendering",
        current=0,
        total=frame_count if frame_count > 0 else None,
        message="Rendering 9:16 video frames.",
    )
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
            model_prefers_wide = False
            if active_plan_point is not None:
                guided_center_x = active_plan_point.center_x * width
                model_prefers_wide = active_plan_point.prefer_wide
            effective_speaker_mode = speaker_mode or (
                active_plan_point is not None
                and active_plan_point.label.lower() in HUMAN_LABELS
            )

            result = detector.detect_for_video(mp_image, timestamp_ms)
            candidate_boxes = (
                select_candidate_boxes(result.detections, subject_name, width, height, limit=3)
                if crop_active
                else []
            )
            subject_box = choose_box(
                result.detections,
                subject_name,
                prev_box,
                width,
                height,
                guided_center_x=guided_center_x,
            )
            tracking_box = derive_speaker_anchor_box(subject_box, width, height) if speaker_mode else subject_box
            speaker_visible = not effective_speaker_mode

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
                    speaker_visible = True
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
                elif subject_box is None:
                    speaker_visible = False

            if effective_speaker_mode and subject_box is not None and subject_box.label in HUMAN_LABELS:
                speaker_visible = True

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
            focus_box = current_box if effective_speaker_mode and current_box is not None else prev_box
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
                    crop_target_aspect,
                    speaker_mode=speaker_mode,
                    guided_center_x=guided_center_x,
                    force_cut=force_cut,
                )
                if use_blurred_fill:
                    layout_metrics = choose_layout_metrics(
                        candidate_boxes,
                        display_box,
                        crop_width,
                        width,
                        height,
                        effective_speaker_mode,
                        speaker_visible=speaker_visible,
                        model_prefers_wide=model_prefers_wide,
                    )
                    if layout_metrics.use_wide:
                        wide_enter_streak += 1
                        wide_exit_streak = 0
                    else:
                        wide_exit_streak += 1
                        wide_enter_streak = 0

                    if not wide_target and wide_enter_streak >= wide_enter_frames:
                        wide_target = True
                    elif wide_target and wide_exit_streak >= wide_exit_frames:
                        wide_target = False

                    target_blend = 1.0 if wide_target else 0.0
                    layout_blend = move_towards(layout_blend, target_blend, 1.0 / transition_frames)

                    full_scale_x = out_width / crop.w
                    full_scale_y = out_height / crop.h
                    full_offset_x = 0
                    full_offset_y = 0
                    full_frame = None
                    wide_frame = None
                    wide_scale_x = wide_scale_y = 1.0
                    wide_offset_x = wide_offset_y = 0
                    if layout_blend <= 0.001:
                        output_frame = crop_frame(frame, crop, out_width, out_height)
                    elif layout_blend >= 0.999:
                        wide_frame, wide_scale_x, wide_scale_y, wide_offset_x, wide_offset_y = compose_center_with_blur_bg_frame(
                            frame,
                            out_width,
                            out_height,
                        )
                        output_frame = wide_frame
                    else:
                        full_frame = crop_frame(frame, crop, out_width, out_height)
                        wide_frame, wide_scale_x, wide_scale_y, wide_offset_x, wide_offset_y = compose_center_with_blur_bg_frame(
                            frame,
                            out_width,
                            out_height,
                        )
                        output_frame = blend_frames(full_frame, wide_frame, layout_blend)
                else:
                    output_frame = crop_frame(frame, crop, out_width, out_height)
                    scale_x = out_width / crop.w
                    scale_y = out_height / crop.h
                    offset_x = 0
                    offset_y = 0
                if focus_box is not None:
                    if output_box is not None:
                        if use_blurred_fill:
                            full_output_box = translate_box_to_crop(
                                output_box,
                                crop,
                                full_scale_x,
                                full_scale_y,
                            )
                            full_output_box = Box(
                                x=full_output_box.x + full_offset_x,
                                y=full_output_box.y + full_offset_y,
                                w=full_output_box.w,
                                h=full_output_box.h,
                                label=full_output_box.label,
                                conf=full_output_box.conf,
                            )
                            if layout_blend <= 0.001:
                                output_box = full_output_box
                            elif layout_blend >= 0.999:
                                output_box = translate_box_with_scale(
                                    output_box,
                                    wide_scale_x,
                                    wide_scale_y,
                                    wide_offset_x,
                                    wide_offset_y,
                                )
                            else:
                                wide_output_box = translate_box_with_scale(
                                    output_box,
                                    wide_scale_x,
                                    wide_scale_y,
                                    wide_offset_x,
                                    wide_offset_y,
                                )
                                output_box = blend_boxes(full_output_box, wide_output_box, layout_blend)
                        else:
                            output_box = translate_box_to_crop(output_box, crop, scale_x, scale_y)
                            output_box = Box(
                                x=output_box.x + offset_x,
                                y=output_box.y + offset_y,
                                w=output_box.w,
                                h=output_box.h,
                                label=output_box.label,
                                conf=output_box.conf,
                            )
            elif prev_box is not None:
                prev_center_x = prev_box.center()[0]
                layout_blend = 0.0

            if draw_subject_box and output_box is not None and not (crop_active and source_inset):
                draw_box(output_frame, output_box, subject_name)

            if source_inset and crop_active:
                inset_crop = crop
                if use_blurred_fill and layout_blend >= 0.5:
                    inset_crop = None
                render_source_inset(output_frame, frame, display_box, inset_crop)

            writer.write(output_frame)
            now = time.monotonic()
            if frame_count > 0 and (
                frame_idx == 0
                or frame_idx + 1 >= frame_count
                or now - last_render_emit_at >= 0.20
            ):
                ratio = (frame_idx + 1) / max(frame_count, 1)
                elapsed = now - render_started_at
                eta_seconds = (elapsed / max(ratio, 1e-6) - elapsed) if ratio > 0.001 else None
                emit_progress(
                    "rendering",
                    current=frame_idx + 1,
                    total=frame_count,
                    message="Rendering 9:16 video frames.",
                    eta_seconds=eta_seconds,
                )
                last_render_emit_at = now
    finally:
        cap.release()
        writer.release()

    emit_progress("muxing", current=0, total=1, message="Muxing audio and finalizing file.")
    mux_audio(temp_video, input_path, output_path)
    emit_progress("muxing", current=1, total=1, message="Audio muxing complete.", eta_seconds=0.0)


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
    parser.add_argument("--focus-plan-interval", type=float)
    parser.add_argument("--focus-plan-batch-size", type=int)
    parser.add_argument("--scene-change-threshold", type=float)
    parser.add_argument("--scene-change-min-gap-sec", type=float)
    args = parser.parse_args()

    target_aspect = None
    if args.aspect_ratio:
        target_aspect = parse_aspect_ratio(args.aspect_ratio)
    elif args.crop_vertical:
        target_aspect = 9.0 / 16.0

    focus_plan = build_focus_plan(
        args.input,
        args.model,
        args.subject.lower().strip(),
        args.llama_url or os.environ.get("LLAMA_BASE_URL"),
        target_aspect=target_aspect,
        sample_interval_sec=args.focus_plan_interval,
        batch_size=args.focus_plan_batch_size,
        scene_change_threshold=args.scene_change_threshold,
        scene_change_min_gap_sec=args.scene_change_min_gap_sec,
    )

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
    emit_progress("complete", current=1, total=1, message="Focus cut complete.", eta_seconds=0.0)
    print(args.output)


if __name__ == "__main__":
    main()
