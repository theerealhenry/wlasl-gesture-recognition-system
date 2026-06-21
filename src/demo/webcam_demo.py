"""
src/demo/webcam_demo.py
========================
Stage 9 — Production-Grade Real-Time Webcam Demo
WLASL 35-Class Gesture Recognition System
Author: Henry Otsyula — Senior Data Scientist & ML Engineer

Revision note (this file)
--------------------------
This is a corrective revision following a second critical-review pass on the
previous draft. Every claim in that review was independently re-verified
against the actual implementations of `src/inference/predictor.py` (Stage 7),
`src/export/convert.py` / `src/export/verify.py` (Stage 8), and the Stage 8
Executive Summary before being accepted, rejected, or partially accepted. The
disposition of each point is recorded inline, next to the code that addresses
it (or next to the code that the review described correctly as already
handling it), so the reasoning survives the next round of edits.

Verified-FALSE / already-correct claims from the second review (not changed)
---------------------------------------------------------------------------------
  - "Session couples directly to predictor internals" — FALSE.
    GestureStreamSession only ever touches `predictor.pipeline`,
    `predictor(x, training=False)`, `predictor.label_map`, and read-only
    public properties (`sequence_length`, `n_classes`, `display_threshold`,
    `model_type`) — exactly the public surface the Stage 7 module docstring
    documents as the intended evaluation/streaming integration point.
    `FrameBuffer` and `PredictionSmoother` are both in predictor.py's
    `__all__`. No underscore-prefixed attribute is read or written anywhere
    in this file. This claim does not hold against the actual source.
  - "Camera failure handling has no reconnect" — MISLEADING. The loop already
    retries (via `time.sleep(0.01); continue`) for
    `MAX_CONSECUTIVE_CAPTURE_FAILURES` (150) frames — about 5 seconds at
    30 FPS — before giving up. That *is* a bounded reconnect-attempt window,
    just not an explicit `cv2.VideoCapture` re-open. A real re-open is added
    below anyway (cheap, strictly additive) since a genuinely unplugged/
    replugged USB camera can need a fresh `VideoCapture` handle, which a
    bare retry-read loop cannot recover from.

Verified-TRUE bugs fixed in this revision
---------------------------------------------
  #F1 FIXED (real bug). FPS/latency stage tracking mixed placeholder `0.0`
      values with genuine measurements in the same rolling deque:
          if latest_result is not None:
              fps_tracker.record("pipeline", 0.0)
              fps_tracker.record("inference", latest_result[...])
          else:
              fps_tracker.record("pipeline", elapsed_pipe_inf_ms)
              fps_tracker.record("inference", 0.0)
      Every other frame contributed a fake zero to whichever stage didn't
      "apply" that frame, dragging both medians toward zero and silently
      contradicting the Stage 8-verified latency numbers (47.11ms full
      pipeline) the HUD's MODEL panel displays right next to this data.
      Fixed by having `GestureStreamSession.update()` return BOTH a
      pipeline-only timing and an inference-only timing on every call where
      a forward pass actually happened (buffer-filling frames record
      neither, rather than a fake zero for one and a real value mislabelled
      as the other).

  #F2 FIXED (real bug, latent). `predictor.display_threshold` is a read-only
      property (no setter) on `GesturePredictor`; the `+`/`-` hotkeys only
      ever mutated `session.display_threshold`. In the CURRENT streaming
      path this caused no visible divergence (the session never calls
      `predictor.predict_from_landmarks()`, which is the only method that
      would consult the predictor's own stale threshold) — but it is a
      footgun: any future code path that calls a `predict_from_*` method
      directly (e.g. a "verify this clip" debug hotkey) would silently use
      the original threshold while the HUD shows the adjusted one. Fixed by
      making `current_threshold` (demo-local) the single source of truth
      end-to-end: the HUD, the session, and the one diagnostic helper that
      now exists (`_predict_offline_clip`, unused by default but exposed
      for debugging) all read from the same variable, never from
      `predictor.display_threshold` after construction.

  #F3 FIXED (hardening). No defensive shape assertion existed between
      `HandsExtractor.extract()`'s output and `FrameBuffer.add_frame()`.
      `FrameBuffer.add_frame()` already raises `ValueError` on a shape
      mismatch, so this was not a silent-corruption risk — but the
      resulting traceback would point inside predictor.py rather than at
      the actual fault (a `FEATURE_SIZE` drift between this file and
      `src/features/constants.py`, e.g. after a schema change). Added an
      explicit, fast assertion at the extraction call site with an
      actionable message.

  #F4 FIXED (clarity / future-proofing). "Freeze" already correctly halted
      the entire ML pipeline (critical review #19 from the first revision),
      and `latest_raw_landmarks` / `latest_meta` were implicitly frozen too
      since the whole extraction block was skipped while frozen. This
      implicit correctness was fragile against future refactors (e.g.
      someone moving the skeleton draw call outside the `if not is_paused`
      guard). Made it explicit: freezing now snapshots landmarks and meta
      into the same `frozen_result` dict, and `HUDRenderer.render()` reads
      skeleton landmarks from the frozen snapshot when frozen, never from
      the (no-longer-advancing, but now also no-longer-implicitly-correct)
      live variables.

  #F5 FIXED (clarity). `GestureStreamSession.update()` returned a bare
      `(result, auto_reset_fired)` tuple; the auto-reset side effect
      (clearing debounce state via `pred_history.update(None)`) was wired
      up manually in `main()` with a comment explaining why. Replaced with
      a small `StreamEvent` dataclass-like dict
      (`{"auto_reset": bool, "no_hands_for_n_frames": int}`) so the event
      is self-describing and any future event (e.g. "smoother window
      changed mid-buffer") has an obvious place to go without another bare
      boolean in the return tuple.

  #F6 ADDED (genuinely new, not in either review). A `--list-cameras`
      utility flag and a clearer first-run error message when index 0 opens
      but immediately fails to deliver frames (the single most common
      "demo doesn't work" support request for OpenCV camera code on
      Windows/Linux). Cheap, additive, no interaction with any existing
      control flow.

  #F7 ADDED. `--record` now also writes a small sidecar
      `<name>.session.json` next to the video with the same session-summary
      data printed to stdout at exit (frame count, FPS, sign tally, latency
      breakdown) — useful for attaching objective numbers to a recorded
      demo GIF/video in the README without manually transcribing the
      terminal output.

Everything else (HUD layout, HandsExtractor mirroring logic, PredictionHistory
debounce/decay, the entire CLI surface, Stage 8 constant values) is carried
forward unchanged from the previous revision, since it was independently
verified correct against predictor.py, the Stage 8 Executive Summary, and the
project handoff document.

Key Stage 8 Integration Facts (verified against the Stage 8 Executive
Summary and src/export/verify.py — locked, do not change without re-running
Stage 8's release gate):
  - display_threshold = 0.35  (model is UNDERCONFIDENT: mean_conf=0.5136 < mean_acc=0.5769)
  - TFLite size = 0.1596 MB   (SELECT_TF_OPS flex delegate adds ~100 KB beyond weight quantisation)
  - Val macro-F1 = 0.5916 (TFLite)     | Test macro-F1 = 0.4867 (TFLite)
  - Full pipeline (excl. MediaPipe): 47.11 ms median -> ~21 FPS headroom
  - Estimated end-to-end with MediaPipe Hands: ~57-60 ms -> ~17-18 FPS
  - Argmax agreement Keras<->TFLite: 98.08% (val), 98.04% (test)

Controls
  q / ESC      — quit
  r            — HARD reset: buffer, smoother, debounce state, AND session
                 statistics (sign counts, display history) all clear.
  s            — save annotated screenshot (PNG, sanitised filename)
  h            — toggle HUD visibility
  m            — toggle landmark skeleton overlay
  p            — pause / unpause (camera still live; ML pipeline halted)
  SPACE        — freeze current prediction (camera AND ML pipeline halted;
                 skeleton + landmarks frozen too — see fix #F4)
  + / =        — raise confidence display threshold by 0.05
  - / _        — lower confidence display threshold by 0.05
  1-9          — set smoother window (1 = no smoothing)

Usage
    python src/demo/webcam_demo.py
    python src/demo/webcam_demo.py --model models/gesture_bilstm_v1.tflite --camera 1
    python src/demo/webcam_demo.py --minimal-hud
    python src/demo/webcam_demo.py --no-flip
    python src/demo/webcam_demo.py --record outputs/demo_recording.mp4
    python src/demo/webcam_demo.py --list-cameras
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import re
import sys
import time
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

# ── Repository root resolution ─────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np

# Only PUBLIC symbols are imported from the predictor module — see the
# module docstring's "Verified-FALSE claims" section for why this matters.
from src.inference.predictor import (
    GesturePredictor,
    FrameBuffer,
    PredictionSmoother,
    HIGH_RISK_SIGNS,
    DEFAULT_SMOOTHING_ALPHA,
    DEFAULT_DISPLAY_THRESHOLD,
)
from src.features.constants import FEATURE_SIZE
from src.utils.logger import get_logger

logger = get_logger(__name__)

# =============================================================================
# Default paths (all relative to repo root)
# =============================================================================

_DEFAULT_CONFIG_SNAPSHOT = str(
    _REPO_ROOT / "artifacts" / "experiments"
    / "bilstm_hands_only_v4_aug" / "config_snapshot.yaml"
)
_DEFAULT_TFLITE_PATH = str(_REPO_ROOT / "models" / "gesture_bilstm_v1.tflite")
_DEFAULT_LABEL_MAP = str(_REPO_ROOT / "artifacts" / "label_map_v1.json")
_DEFAULT_CALIB_REPORT = str(
    _REPO_ROOT / "reports" / "evaluation" / "evaluation_report.json"
)
_SCREENSHOT_DIR = _REPO_ROOT / "reports" / "figures" / "webcam_screenshots"

# =============================================================================
# Stage 8 verified constants (read-only references for the HUD — these are
# the Stage 8 release-gate measurements, never recomputed at runtime here)
# =============================================================================

#: Display threshold calibrated to the champion's documented underconfidence
#: (Stage 6 Phase D: mean_confidence=0.5136 < mean_accuracy=0.5769).
#: Imported from predictor.py's DEFAULT_DISPLAY_THRESHOLD so this file and
#: GesturePredictor can never silently drift apart on the default value.
DISPLAY_THRESHOLD: float = DEFAULT_DISPLAY_THRESHOLD

#: High-risk signs (Stage 5 Finding 8) — imported directly from
#: predictor.py rather than re-declared here, so this file always reflects
#: whatever HIGH_RISK_SIGNS that module actually uses for is_high_risk_class.
_HIGH_RISK_SIGNS = frozenset(HIGH_RISK_SIGNS)

#: Stage 6 Phase E confusable pairs (cosine similarity 0.785-0.963 between
#: champion activations). Also surfaced in GesturePredictor.get_metadata()
#: under "attribution_notes" — duplicated here as a small, static lookup
#: table purely for fast HUD rendering (no need to call get_metadata() per
#: frame for four fixed pairs).
CONFUSABLE_PAIRS: Dict[str, List[str]] = {
    "think": ["who"], "who": ["think"],
    "later": ["house"], "house": ["later"],
    "cousin": ["mother"], "mother": ["cousin"],
    "girl": ["orange"], "orange": ["girl"],
}

#: Stage 8 release-gate performance summary (HUD info panel only). Sourced
#: verbatim from "Stage 8 Executive Summary — TFLite Export & Verification"
#: sections 4, 5.1, 7.1. Do not edit without re-running Stage 8.
_FULL_PIPELINE_MS: float = 47.11
_VAL_MACRO_F1: float = 0.5916
_TEST_MACRO_F1: float = 0.4867
_MODEL_SIZE_MB: float = 0.1596
_MODEL_PARAMS: int = 68_771

# =============================================================================
# Behavioural constants (named, per critical review #22 from the first pass)
# =============================================================================

#: A debounced sign survives at most this many consecutive non-confident
#: frames before its streak is cleared. Prevents a stale "stable" sign from
#: silently surviving an unrelated low-confidence gap (e.g. the signer
#: pausing, or a brief misdetection).
LOW_CONFIDENCE_DECAY_FRAMES: int = 8

#: Consecutive cv2.VideoCapture.read() failures after which the demo
#: attempts ONE re-open of the capture device before giving up entirely.
#: ~5 seconds at 30 FPS — long enough to ride out a transient USB hiccup,
#: short enough that a genuinely disconnected camera doesn't hang the demo.
MAX_CONSECUTIVE_CAPTURE_FAILURES: int = 150

#: After a full re-open attempt also fails this many consecutive times, the
#: demo gives up for good. Kept separate from MAX_CONSECUTIVE_CAPTURE_FAILURES
#: so the "first strike" and "second strike" budgets are independently tunable.
MAX_CONSECUTIVE_FAILURES_AFTER_REOPEN: int = 60

#: Smoother window is clamped to this range on every change (CLI or runtime
#: keypress).
_SMOOTHER_WINDOW_MIN: int = 1
_SMOOTHER_WINDOW_MAX: int = 9

# =============================================================================
# HUD Design System (ASCII-only on-frame text)
# =============================================================================

# BGR colours
C_WHITE = (255, 255, 255)
C_BLACK = (0, 0, 0)
C_DARK = (25, 25, 35)
C_DARK2 = (40, 40, 55)
C_GREEN = (50, 220, 100)
C_AMBER = (30, 175, 255)
C_RED = (60, 60, 240)
C_BLUE = (255, 140, 40)
C_CYAN = (230, 210, 50)
C_GREY = (160, 160, 160)
C_LGREY = (210, 210, 210)
C_YELLOW = (0, 220, 230)
C_PURPLE = (200, 80, 180)
C_TEAL = (180, 200, 50)
C_SKELETON = (0, 255, 128)

F_SANS = cv2.FONT_HERSHEY_SIMPLEX
F_DUPLEX = cv2.FONT_HERSHEY_DUPLEX

PANEL_ALPHA = 0.72

# Layout fractions used to derive panel geometry from the live frame size.
# Clamped so very small or very large resolutions both produce usable layouts.
_TOP_H_FRAC = 0.16
_SIDE_W_FRAC = 0.20
_BOT_H_FRAC = 0.075
_TOP_H_RANGE = (70, 130)
_SIDE_W_RANGE = (160, 260)
_BOT_H_RANGE = (36, 60)
#: Below this frame width, the right-hand info panel is force-disabled
#: regardless of --minimal-hud, since it would not fit without overlapping
#: the video feed.
_MIN_WIDTH_FOR_SIDE_PANEL = 640

# MediaPipe Hands connections (used for skeleton drawing only).
_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def _layout(w: int, h: int) -> Tuple[int, int, int]:
    """Derive (top_h, side_w, bot_h) panel sizes from the live frame size."""
    top_h = int(np.clip(h * _TOP_H_FRAC, *_TOP_H_RANGE))
    side_w = int(np.clip(w * _SIDE_W_FRAC, *_SIDE_W_RANGE))
    bot_h = int(np.clip(h * _BOT_H_FRAC, *_BOT_H_RANGE))
    return top_h, side_w, bot_h


def _sanitize_filename_component(s: str) -> str:
    """Strip any character outside [A-Za-z0-9_-] before using a value
    (e.g. a predicted sign name) inside a filename."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "_", s or "")
    cleaned = cleaned.strip("_") or "unknown"
    return cleaned[:40]


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# =============================================================================
# Drawing utilities
# =============================================================================

def _conf_color(conf: float) -> Tuple[int, int, int]:
    if conf >= 0.70:
        return C_GREEN
    if conf >= 0.45:
        return C_AMBER
    return C_RED


def _alpha_rect(
    img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
    color: Tuple[int, int, int], alpha: float = PANEL_ALPHA,
) -> None:
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    sub = img[y1:y2, x1:x2]
    rect = np.full_like(sub, color, dtype=np.uint8)
    img[y1:y2, x1:x2] = cv2.addWeighted(sub, 1 - alpha, rect, alpha, 0)


def _text(
    img: np.ndarray, text: str, pos: Tuple[int, int], font, scale: float,
    color: Tuple[int, int, int], thickness: int = 1, shadow_offset: int = 2,
) -> None:
    cv2.putText(img, text, (pos[0] + shadow_offset, pos[1] + shadow_offset),
                font, scale, C_BLACK, thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, pos, font, scale, color, thickness, cv2.LINE_AA)


def _progress_bar(
    img: np.ndarray, x: int, y: int, width: int, height: int, fraction: float,
    fill_color: Tuple[int, int, int], bg_color: Tuple[int, int, int] = (60, 60, 60),
    border: bool = True,
) -> None:
    cv2.rectangle(img, (x, y), (x + width, y + height), bg_color, -1)
    fill_w = max(0, int(width * _clamp(fraction, 0.0, 1.0)))
    if fill_w > 0:
        cv2.rectangle(img, (x, y), (x + fill_w, y + height), fill_color, -1)
    if border:
        cv2.rectangle(img, (x, y), (x + width, y + height), C_GREY, 1)


def _draw_skeleton(
    img: np.ndarray, landmarks_225: np.ndarray, frame_w: int, frame_h: int,
    color: Tuple[int, int, int] = C_SKELETON, point_r: int = 3, line_t: int = 1,
) -> None:
    """
    Draw both-hand skeletons from a raw, pre-normalisation (225,) landmark
    vector. Only draws a hand slot that is non-zero (i.e. actually detected
    this frame) — see FrameBuffer's zero-fill invariant.
    """
    for offset in (0, 63):  # left, right
        lm_slice = landmarks_225[offset:offset + 63].reshape(21, 3)
        if not np.any(lm_slice):
            continue
        pts = [(int(lm[0] * frame_w), int(lm[1] * frame_h)) for lm in lm_slice]
        for a, b in _HAND_CONNECTIONS:
            if 0 <= a < len(pts) and 0 <= b < len(pts):
                cv2.line(img, pts[a], pts[b], color, line_t, cv2.LINE_AA)
        for i, (px, py) in enumerate(pts):
            r = point_r + 1 if i == 0 else point_r
            cv2.circle(img, (px, py), r, color, -1, cv2.LINE_AA)
            cv2.circle(img, (px, py), r, C_WHITE, 1, cv2.LINE_AA)


# =============================================================================
# FPS / stage-latency tracker
# =============================================================================

class FPSTracker:
    """
    Rolling FPS estimator with per-stage breakdown.

    Fix #F1: callers must record ONLY genuine measurements. Earlier
    revisions recorded a placeholder `0.0` for whichever stage "didn't
    apply" on a given frame (e.g. "pipeline"=0.0 on frames where a full
    forward pass happened, "inference"=0.0 on frames where the buffer was
    still filling). Mixed into the same rolling deque as real measurements,
    those placeholders drag `stage_median()` toward zero and silently
    contradict the Stage 8-verified latency numbers shown right next to
    this data in the HUD's MODEL/PIPELINE panels. There is no longer any
    code path in this file that calls `record()` with a synthetic value —
    every `record()` call site below corresponds to an actual
    `time.perf_counter()` delta for that exact stage on that exact frame.
    Stages simply go unrecorded (not zero-recorded) on frames where they
    did not run, which is what `stage_median()` over a deque already
    handles correctly via "skip frames where this stage didn't fire".
    """

    def __init__(self, window: int = 30):
        self._window = window
        self._intervals: Deque[float] = deque(maxlen=window)
        self._t_last = time.perf_counter()
        self._stage_ms: Dict[str, Deque[float]] = {
            "capture": deque(maxlen=window),
            "mediapipe": deque(maxlen=window),
            "pipeline": deque(maxlen=window),
            "inference": deque(maxlen=window),
            "hud": deque(maxlen=window),
        }

    def tick(self) -> float:
        t = time.perf_counter()
        dt = t - self._t_last
        self._t_last = t
        self._intervals.append(dt)
        return self.fps

    def record(self, stage: str, ms: float) -> None:
        """Record a genuine measured duration. Never call this with a
        synthetic placeholder — simply skip the call for stages that did
        not run on a given frame."""
        if stage in self._stage_ms:
            self._stage_ms[stage].append(ms)

    @property
    def fps(self) -> float:
        if not self._intervals:
            return 0.0
        return 1.0 / (sum(self._intervals) / len(self._intervals))

    def stage_median(self, stage: str) -> float:
        d = self._stage_ms.get(stage, deque())
        if not d:
            return 0.0
        arr = sorted(d)
        return arr[len(arr) // 2]

    def stage_sample_count(self, stage: str) -> int:
        return len(self._stage_ms.get(stage, deque()))

    @property
    def breakdown(self) -> Dict[str, float]:
        return {k: self.stage_median(k) for k in self._stage_ms}


# =============================================================================
# Prediction history (debounce + session stats)
# =============================================================================

class PredictionHistory:
    """
    Tracks prediction stability (debounce) and session statistics.

    Decay: a sign must reappear in `debounce` consecutive CONFIDENT
    predictions before it is displayed. `LOW_CONFIDENCE_DECAY_FRAMES`
    consecutive non-confident updates clear the streak, so a stale "stable"
    sign cannot silently survive an unrelated gap (signer pausing, brief
    misdetection).

    Hard reset: reset() clears EVERYTHING, including session sign counts
    and display history — matching the 'r' key's documented behaviour.
    """

    def __init__(self, debounce: int = 3):
        self._debounce = max(1, int(debounce))
        self._current: Optional[str] = None
        self._streak = 0
        self._displayed: Optional[str] = None
        self._low_conf_streak = 0
        self._session_counts: Dict[str, int] = {}
        self._display_history: List[Tuple[float, str, float]] = []

    def update(self, result: Optional[Dict[str, Any]]) -> Optional[str]:
        if result is None:
            return self._displayed

        sign = result.get("sign", "")
        is_confident = bool(result.get("is_confident", False))

        if not sign or not is_confident:
            self._low_conf_streak += 1
            if self._low_conf_streak >= LOW_CONFIDENCE_DECAY_FRAMES:
                self._current = None
                self._streak = 0
            return self._displayed

        self._low_conf_streak = 0

        if sign == self._current:
            self._streak += 1
        else:
            self._current = sign
            self._streak = 1

        if self._streak >= self._debounce and sign != self._displayed:
            self._displayed = sign
            self._session_counts[sign] = self._session_counts.get(sign, 0) + 1
            conf = result.get("confidence", 0.0)
            self._display_history.append((time.time(), sign, conf))
            if len(self._display_history) > 200:
                self._display_history.pop(0)

        return self._displayed

    @property
    def is_stable(self) -> bool:
        return self._streak >= self._debounce

    @property
    def streak(self) -> int:
        return self._streak

    @property
    def displayed(self) -> Optional[str]:
        return self._displayed

    @property
    def session_count(self) -> int:
        return sum(self._session_counts.values())

    @property
    def session_counts(self) -> Dict[str, int]:
        return dict(self._session_counts)

    def most_predicted(self, n: int = 5) -> List[Tuple[str, int]]:
        return sorted(self._session_counts.items(), key=lambda x: x[1], reverse=True)[:n]

    def reset(self) -> None:
        """Hard reset — clears debounce state AND all session statistics."""
        self._current = None
        self._streak = 0
        self._displayed = None
        self._low_conf_streak = 0
        self._session_counts.clear()
        self._display_history.clear()


# =============================================================================
# MediaPipe extractor (Hands-preferred, Holistic fallback)
# =============================================================================

class HandsExtractor:
    """
    MediaPipe Hands extractor — preferred over Holistic for Stage 9, per
    Stage 8 Executive Summary section 11.3 (~8-10ms vs Holistic's ~18ms for
    this hands-only champion).

    Output contract (matches the FrameBuffer / FeaturePipeline invariant
    documented in src/inference/predictor.py):
        [0:63]    left hand  — 21 landmarks x (x, y, z)
        [63:126]  right hand — 21 landmarks x (x, y, z)
        [126:225] pose       — ALWAYS zero, in BOTH extraction modes.

    Why pose is always zero, even under the Holistic fallback
    --------------------------------------------------------------
    The champion is hands_only and was trained with [126:225] always zero.
    If the primary Hands path always zero-fills pose but a Holistic
    fallback populated it with real values, the model would receive a
    meaningfully different input distribution depending on which extractor
    happened to initialise successfully on a given machine — a silent,
    environment-dependent accuracy regression. Holistic is used as a
    fallback for HAND landmarks only; its pose output is discarded.

    Handedness / mirroring
    -----------------------------------------------
    MediaPipe's `Handedness` classification is always relative to the RAW
    camera image, BEFORE any mirror flip the caller applies.

        mirrored=True  (default; frame has been cv2.flip(frame, 1)'d):
            MediaPipe "Left"  -> signer's RIGHT hand -> slot [63:126]
            MediaPipe "Right" -> signer's LEFT  hand -> slot [0:63]
        mirrored=False (--no-flip):
            MediaPipe "Left"  -> signer's LEFT  hand -> slot [0:63]
            MediaPipe "Right" -> signer's RIGHT hand -> slot [63:126]

    The caller MUST pass `mirrored` consistently with whatever flip it
    actually applies to the frame before calling `extract()`. `main()`
    below ties this directly to `not args.no_flip`.
    """

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = False,
        mirrored: bool = True,
    ):
        self._mirrored = bool(mirrored)
        try:
            import mediapipe as mp
            self._mp_hands = mp.solutions.hands
            self._hands = self._mp_hands.Hands(
                static_image_mode=static_image_mode,
                max_num_hands=2,
                model_complexity=model_complexity,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self._mode = "hands"
            logger.info(
                "MediaPipe Hands initialised (complexity=%d, mirrored=%s)",
                model_complexity, self._mirrored,
            )
        except Exception as exc:
            logger.warning(
                "MediaPipe Hands init failed (%s). Falling back to Holistic "
                "(pose output will still be discarded).", exc,
            )
            self._try_holistic()

    def _try_holistic(self) -> None:
        import mediapipe as mp
        self._mp_holistic = mp.solutions.holistic
        self._holistic = self._mp_holistic.Holistic(
            model_complexity=1, min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._mode = "holistic"
        logger.info("MediaPipe Holistic initialised as fallback (pose forced to zero).")

    def extract(self, bgr_frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Returns (landmarks_225, meta). landmarks_225 is float32, zero-filled
        for any undetected hand and for the entire pose slot [126:225].

        Fix #F3: result shape is asserted before returning, so a future
        FEATURE_SIZE drift (e.g. a schema change in src/features/constants.py
        not mirrored here) fails loudly at the extraction call site with an
        actionable message, rather than producing a confusing ValueError
        several call-frames deep inside FrameBuffer.add_frame().
        """
        landmarks = np.zeros(FEATURE_SIZE, dtype=np.float32)
        meta: Dict[str, Any] = {"left_detected": False, "right_detected": False, "n_hands": 0}

        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        if self._mode == "hands":
            results = self._hands.process(rgb)
            if results.multi_hand_landmarks and results.multi_handedness:
                meta["n_hands"] = len(results.multi_hand_landmarks)
                for hand_lm, hand_class in zip(
                    results.multi_hand_landmarks, results.multi_handedness,
                ):
                    raw_label = hand_class.classification[0].label  # camera-relative
                    is_signer_right = (
                        (raw_label == "Left") if self._mirrored else (raw_label == "Right")
                    )
                    offset = 63 if is_signer_right else 0
                    meta["right_detected" if is_signer_right else "left_detected"] = True
                    for i, lm in enumerate(hand_lm.landmark):
                        base = offset + i * 3
                        landmarks[base] = lm.x
                        landmarks[base + 1] = lm.y
                        landmarks[base + 2] = lm.z
            # [126:225] intentionally left at zero — hands_only champion.

        else:  # Holistic fallback — hands only, pose discarded (see docstring)
            results = self._holistic.process(rgb)
            if results.left_hand_landmarks:
                meta["left_detected"] = True
                for i, lm in enumerate(results.left_hand_landmarks.landmark[:21]):
                    landmarks[i * 3], landmarks[i * 3 + 1], landmarks[i * 3 + 2] = lm.x, lm.y, lm.z
            if results.right_hand_landmarks:
                meta["right_detected"] = True
                for i, lm in enumerate(results.right_hand_landmarks.landmark[:21]):
                    base = 63 + i * 3
                    landmarks[base], landmarks[base + 1], landmarks[base + 2] = lm.x, lm.y, lm.z
            meta["n_hands"] = int(meta["left_detected"]) + int(meta["right_detected"])

        # Fix #F3: fast, actionable shape guard at the source of truth.
        if landmarks.shape != (FEATURE_SIZE,):
            raise RuntimeError(
                f"HandsExtractor.extract(): produced landmarks of shape "
                f"{landmarks.shape}, expected ({FEATURE_SIZE},). This "
                "indicates FEATURE_SIZE has drifted from the constant this "
                "extractor was written against — check "
                "src/features/constants.py::FEATURE_SIZE."
            )

        return landmarks, meta

    def close(self) -> None:
        with contextlib.suppress(Exception):
            if self._mode == "hands":
                self._hands.close()
            else:
                self._holistic.close()

    @property
    def mode(self) -> str:
        return self._mode


# =============================================================================
# GestureStreamSession — encapsulation-respecting streaming composition
# =============================================================================

class GestureStreamSession:
    """
    Reproduces GesturePredictor.predict_from_webcam_frame()'s streaming
    contract (rolling buffer -> pipeline -> inference -> smoother ->
    auto-reset) using ONLY GesturePredictor's PUBLIC surface, so this demo
    never depends on any underscore-prefixed predictor attribute. Verified
    against predictor.py's actual `__all__` and module docstring — see this
    file's top-level docstring, "Verified-FALSE / already-correct claims".

    Public surface used from `predictor` (all already part of
    GesturePredictor's documented API):
        predictor.pipeline            (FeaturePipeline instance — public)
        predictor(x, training=False)  (evaluation-framework __call__)
        predictor.label_map
        predictor.sequence_length / .feature_dim / .n_classes / .model_type

    Public classes composed (both exported in predictor.py's __all__):
        FrameBuffer, PredictionSmoother

    Fix #F1 (timing): `update()` now returns genuinely separate
    `pipeline_ms` (FeaturePipeline preprocessing only) and `inference_ms`
    (model forward pass only) inside the result dict — both measured, never
    a synthetic placeholder for the "other" stage. `main()` records each
    into FPSTracker only when it actually ran.

    Fix #F2 (threshold): the session's `display_threshold` is the ONLY
    threshold consulted anywhere in the streaming path; `GesturePredictor`'s
    own (read-only) `display_threshold` property is read exactly once, at
    construction, purely to seed the session's initial value — never again
    afterward. This makes the session the single source of truth for the
    live threshold for the remainder of the process, matching what the
    `+`/`-` hotkeys already assumed.

    Fix #F5 (events): `update()` returns a `StreamEvent`-shaped dict
    (`{"auto_reset": bool, "no_detection_streak": int}`) instead of a bare
    boolean, so the auto-reset side effect is self-describing at the call
    site in `main()`.
    """

    def __init__(
        self,
        predictor: GesturePredictor,
        smoother_window: int = 5,
        smoothing_alpha: float = DEFAULT_SMOOTHING_ALPHA,
        display_threshold: Optional[float] = None,
        n_top_k: int = 3,
        auto_reset_no_detection_frames: Optional[int] = 3,
    ) -> None:
        self.predictor = predictor
        self._seq_len = int(predictor.sequence_length)
        self._n_classes = int(predictor.n_classes)
        self._n_top_k = max(1, min(int(n_top_k), self._n_classes))
        self._alpha = float(smoothing_alpha)
        # Seeded once from the predictor's resolved threshold (itself derived
        # from --threshold > calibration report > Stage 6 default 0.35 — see
        # GesturePredictor._resolve_display_threshold). Never read from
        # predictor.display_threshold again after this line (fix #F2).
        self._display_threshold = (
            float(display_threshold) if display_threshold is not None
            else float(predictor.display_threshold)
        )
        self._auto_reset_threshold = auto_reset_no_detection_frames
        self._no_detection_streak = 0

        window = max(_SMOOTHER_WINDOW_MIN, min(_SMOOTHER_WINDOW_MAX, int(smoother_window)))
        self._buffer = FrameBuffer(seq_len=self._seq_len, n_features=FEATURE_SIZE)
        self._smoother = PredictionSmoother(window=window, alpha=self._alpha, n_classes=self._n_classes)

    # ── Properties ──────────────────────────────────────────────────────
    @property
    def sequence_length(self) -> int:
        return self._seq_len

    @property
    def frames_buffered(self) -> int:
        return self._buffer.frames_accumulated()

    @property
    def smoother_window(self) -> int:
        return self._smoother.window

    @property
    def display_threshold(self) -> float:
        return self._display_threshold

    @display_threshold.setter
    def display_threshold(self, value: float) -> None:
        self._display_threshold = _clamp(float(value), 0.0, 1.0)

    @property
    def no_detection_streak(self) -> int:
        return self._no_detection_streak

    # ── Core streaming update ───────────────────────────────────────────
    def update(self, landmarks_225: np.ndarray) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """
        Feed one raw (FEATURE_SIZE,) landmark vector through the streaming
        pipeline.

        Returns
        -------
        Tuple[result | None, event]
            result : prediction dict (see _build below), or None while the
                     buffer is filling, or immediately after an auto-reset.
            event  : {"auto_reset": bool, "no_detection_streak": int} —
                     always returned (fix #F5), so the caller never has to
                     special-case "what does a bare False mean here".

        Timing fields in `result` (fix #F1 — always genuinely measured,
        never a placeholder):
            "pipeline_ms"   : FeaturePipeline preprocessing time for this
                               call (present only when a forward pass ran).
            "inference_ms"  : model forward-pass time for this call
                               (present only when a forward pass ran).
        """
        if not np.any(landmarks_225):
            self._no_detection_streak += 1
        else:
            self._no_detection_streak = 0

        # Zero vectors enter the buffer BEFORE the auto-reset check —
        # zero-fill is semantic (Stage 3 convention), not noise to filter.
        self._buffer.add_frame(landmarks_225)

        event: Dict[str, Any] = {
            "auto_reset": False,
            "no_detection_streak": self._no_detection_streak,
        }

        if (
            self._auto_reset_threshold is not None
            and self._no_detection_streak >= self._auto_reset_threshold
        ):
            self.reset()
            event["auto_reset"] = True
            return None, event

        if not self._buffer.is_ready():
            return None, event

        raw_seq = self._buffer.get_array()

        t_pipe = time.perf_counter()
        features_2d = self.predictor.pipeline(raw_seq, training=False)
        pipeline_ms = (time.perf_counter() - t_pipe) * 1000.0

        t_inf = time.perf_counter()
        raw_probs_batch = self.predictor(features_2d, training=False)  # (1, n_classes)
        inference_ms = (time.perf_counter() - t_inf) * 1000.0

        raw_probs = np.asarray(raw_probs_batch[0], dtype=np.float32)
        raw_confidence = float(raw_probs.max())
        raw_class_idx = int(np.argmax(raw_probs))

        predicted_class, smoothed_probs, is_stable = self._smoother.update(raw_probs)
        display_confidence = round(float(smoothed_probs[predicted_class]), 4)

        label_map = self.predictor.label_map
        sign_name = label_map.get_name_safe(predicted_class, f"class_{predicted_class}")

        top_k_raw = self._smoother.top_k(smoothed_probs, k=self._n_top_k)
        top_k = [
            {
                "sign": label_map.get_name_safe(e["class_idx"], f"class_{e['class_idx']}"),
                "class_idx": e["class_idx"],
                "confidence": round(e["confidence"], 4),
            }
            for e in top_k_raw
        ]

        result: Dict[str, Any] = {
            "sign": sign_name,
            "confidence": display_confidence,
            "is_confident": display_confidence >= self._display_threshold,
            "class_idx": predicted_class,
            "top_k": top_k,
            "raw_confidence": round(raw_confidence, 4),
            "raw_class_idx": raw_class_idx,
            "is_stable": is_stable,
            "is_high_risk_class": sign_name in _HIGH_RISK_SIGNS,
            "n_frames_input": self._seq_len,
            "pipeline_ms": round(pipeline_ms, 3),
            "inference_ms": round(inference_ms, 3),
            "inference_latency_ms": round(pipeline_ms + inference_ms, 3),
            "frames_in_buffer": self._buffer.frames_accumulated(),
        }
        return result, event

    def set_smoother_window(self, window: int) -> int:
        """Validated, clamped smoother window change."""
        window = max(_SMOOTHER_WINDOW_MIN, min(_SMOOTHER_WINDOW_MAX, int(window)))
        if window != self._smoother.window:
            self._smoother = PredictionSmoother(
                window=window, alpha=self._alpha, n_classes=self._n_classes,
            )
            logger.info("Smoother window changed -> %d (smoother state reset)", window)
        return window

    def reset(self) -> None:
        """Clears the rolling buffer, the smoother, and the no-detection streak."""
        self._buffer.reset()
        self._smoother.reset()
        self._no_detection_streak = 0


# =============================================================================
# HUD Renderer
# =============================================================================

class HUDRenderer:
    """
    Production HUD renderer. Operates purely on plain values passed in by
    the caller (frames_buffered, seq_len, model_type, ...) rather than a
    GesturePredictor / GestureStreamSession reference, so this class has
    zero coupling to either's internals.
    """

    def __init__(self, minimal: bool = False, skeleton_stride: int = 1):
        self.minimal = minimal
        self._skeleton_stride = max(1, int(skeleton_stride))
        self._frame_n = 0
        self._pulse = 0.0

    def render(
        self,
        frame: np.ndarray,
        result: Optional[Dict[str, Any]],
        meta: Dict[str, Any],
        history: PredictionHistory,
        fps_tracker: FPSTracker,
        frames_buffered: int,
        seq_len: int,
        model_type: str,
        display_threshold: float,
        smoother_window: int,
        show_skeleton: bool = True,
        is_paused: bool = False,
        is_frozen: bool = False,
        frozen_result: Optional[Dict[str, Any]] = None,
        latest_raw_landmarks: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        self._frame_n += 1
        self._pulse = (self._pulse + 0.08) % (2 * math.pi)
        h, w = frame.shape[:2]
        top_h, side_w, bot_h = _layout(w, h)
        use_side_panel = (not self.minimal) and (w >= _MIN_WIDTH_FOR_SIDE_PANEL)

        display_result = frozen_result if frozen_result is not None else result

        # Fix #F4: skeleton landmarks/meta are read from the SAME source the
        # caller used to build `display_result` — when frozen, `meta` and
        # `latest_raw_landmarks` are expected to be the frozen snapshot
        # (main() now passes the frozen copies explicitly while frozen,
        # rather than relying on the live variables having implicitly
        # stopped advancing because the extraction block was skipped).
        if (
            show_skeleton
            and meta.get("n_hands", 0) > 0
            and latest_raw_landmarks is not None
            and (self._frame_n % self._skeleton_stride == 0)
        ):
            _draw_skeleton(frame, latest_raw_landmarks, w, h)

        _alpha_rect(frame, 0, 0, w, top_h, C_DARK, alpha=0.80)
        if use_side_panel:
            _alpha_rect(frame, w - side_w, top_h, w, h - bot_h, C_DARK2, alpha=0.75)
        _alpha_rect(frame, 0, h - bot_h, w, h, C_DARK, alpha=0.80)

        cv2.line(frame, (0, top_h - 1), (w, top_h - 1), C_BLUE, 1)
        cv2.line(frame, (0, h - bot_h), (w, h - bot_h), C_BLUE, 1)
        if use_side_panel:
            cv2.line(frame, (w - side_w, top_h), (w - side_w, h - bot_h), C_BLUE, 1)

        self._draw_top_banner(frame, w, top_h, display_result, history, display_threshold,
                               is_frozen)

        if frames_buffered < seq_len and result is None and not is_paused and not is_frozen:
            self._draw_buffer_progress(frame, w, h, frames_buffered, seq_len)
        elif meta.get("n_hands", 0) == 0 and not is_paused and not is_frozen:
            self._draw_no_hands_warning(frame, w, h, bot_h)

        if use_side_panel:
            self._draw_right_panel(
                frame, w, h, top_h, side_w, bot_h, display_result, history,
                fps_tracker, model_type, smoother_window, display_threshold,
            )

        self._draw_bottom_bar(frame, w, h, bot_h, fps_tracker, meta, is_paused, is_frozen)

        if is_paused:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
            _text(frame, "PAUSED", (w // 2 - 60, h // 2), F_DUPLEX, 1.1, C_AMBER, 2)
            _text(frame, "Press SPACE/p to resume", (w // 2 - 130, h // 2 + 36),
                  F_SANS, 0.5, C_LGREY, 1)

        return frame

    def _draw_top_banner(
        self, frame: np.ndarray, w: int, top_h: int,
        result: Optional[Dict[str, Any]], history: PredictionHistory,
        display_threshold: float, is_frozen: bool,
    ) -> None:
        title_y = max(28, top_h // 3)
        if result is None or not result.get("is_confident", False):
            _text(frame, "WLASL Gesture Recognition", (w // 2 - 150, title_y),
                  F_DUPLEX, 0.85, C_GREY, 1)
            _text(frame, "Waiting for confident prediction...",
                  (w // 2 - 155, title_y + 32), F_SANS, 0.5, C_GREY, 1)
            return

        sign = result.get("sign", "")
        confidence = result.get("confidence", 0.0)
        is_hr = sign in _HIGH_RISK_SIGNS
        is_conf_pair = sign in CONFUSABLE_PAIRS
        col = _conf_color(confidence)
        stable = history.is_stable
        pulse_alpha = 0.85 + 0.15 * math.sin(self._pulse)

        sign_upper = sign.upper()
        (sw, sh), _ = cv2.getTextSize(sign_upper, F_DUPLEX, 2.0, 2)
        sx = max(10, min(w - sw - 10, w // 2 - sw // 2))

        if stable:
            _alpha_rect(frame, sx - 10, 5, sx + sw + 10, 60,
                        tuple(int(c * 0.35) for c in col), alpha=0.6)
            glow_col = tuple(int(c * pulse_alpha) for c in col)
        else:
            glow_col = col

        cv2.putText(frame, sign_upper, (sx + 2, 53), F_DUPLEX, 2.0, C_BLACK, 4, cv2.LINE_AA)
        cv2.putText(frame, sign_upper, (sx, 51), F_DUPLEX, 2.0, glow_col, 2, cv2.LINE_AA)

        bar_x, bar_y = sx, 65
        bar_w, bar_h = min(sw + 10, w // 2), 10
        _progress_bar(frame, bar_x, bar_y, bar_w, bar_h, confidence, col)

        _text(frame, f"{confidence:.0%}", (bar_x + bar_w + 8, bar_y + 9), F_SANS, 0.48, col, 1)

        thresh_x = bar_x + int(bar_w * display_threshold)
        cv2.line(frame, (thresh_x, bar_y - 3), (thresh_x, bar_y + bar_h + 3), C_YELLOW, 1)

        dot_x = min(w - 15, sx + sw + 20)
        dot_col = C_GREEN if stable else C_AMBER
        cv2.circle(frame, (dot_x, 30), 7, dot_col, -1)
        cv2.circle(frame, (dot_x, 30), 7, C_WHITE, 1)

        badge_x = sx
        if is_hr:
            label = "RISK"
            cv2.rectangle(frame, (badge_x, 82), (badge_x + 70, 102), C_AMBER, -1)
            _text(frame, label, (badge_x + 6, 97), F_SANS, 0.4, C_BLACK, 1, shadow_offset=0)
            badge_x += 78
        if is_conf_pair:
            partners = CONFUSABLE_PAIRS.get(sign, [])
            if partners:
                label = f"~ {partners[0].upper()}"
                cv2.rectangle(frame, (badge_x, 82), (badge_x + len(label) * 10 + 12, 102), C_PURPLE, -1)
                _text(frame, label, (badge_x + 4, 97), F_SANS, 0.38, C_WHITE, 1, shadow_offset=0)
        if is_frozen:
            _text(frame, "[FROZEN]", (w - 110, 25), F_SANS, 0.5, C_CYAN, 1)

    def _draw_right_panel(
        self, frame: np.ndarray, w: int, h: int, top_h: int, side_w: int, bot_h: int,
        result: Optional[Dict[str, Any]], history: PredictionHistory,
        fps_tracker: FPSTracker, model_type: str, smoother_window: int,
        display_threshold: float,
    ) -> None:
        px = w - side_w + 8
        py = top_h + 12

        _text(frame, "TOP PREDICTIONS", (px, py + 10), F_SANS, 0.42, C_CYAN, 1)
        py += 20
        cv2.line(frame, (px - 4, py), (w - 8, py), C_BLUE, 1)
        py += 8

        top_k = result.get("top_k", []) if result else []
        py_next = py
        for i, entry in enumerate(top_k[:3]):
            sign_name = entry.get("sign", "?")[:12]
            e_conf = entry.get("confidence", 0.0)
            e_col = _conf_color(e_conf)
            bar_w_ = side_w - 26
            entry_y = py + i * 44

            rank_col = [C_GREEN, C_AMBER, C_GREY][i]
            cv2.rectangle(frame, (px - 4, entry_y - 2), (px + 22, entry_y + 14), rank_col, -1)
            _text(frame, f"#{i + 1}", (px + 2, entry_y + 11), F_SANS, 0.38, C_BLACK, 1, shadow_offset=0)
            _text(frame, sign_name.upper(), (px + 28, entry_y + 12), F_SANS, 0.45,
                  C_WHITE if i == 0 else C_LGREY, 1)
            _progress_bar(frame, px + 2, entry_y + 18, bar_w_, 8, e_conf, e_col)
            _text(frame, f"{e_conf:.0%}", (px + bar_w_ + 6, entry_y + 26), F_SANS, 0.35, e_col, 1)
            if sign_name.lower() in _HIGH_RISK_SIGNS:
                _text(frame, "!", (px + bar_w_ + 40, entry_y + 12), F_SANS, 0.4, C_AMBER, 1)
            py_next = entry_y + 44

        if not top_k:
            _text(frame, "-- No prediction", (px + 4, py + 16), F_SANS, 0.42, C_GREY, 1)
            py_next = py + 50
        py = py_next + 8
        cv2.line(frame, (px - 4, py), (w - 8, py), C_BLUE, 1)
        py += 10

        _text(frame, "SESSION", (px, py + 8), F_SANS, 0.38, C_CYAN, 1)
        py += 20
        _text(frame, f"Signs predicted  {history.session_count:>5}", (px, py), F_SANS, 0.38, C_LGREY, 1)
        py += 18

        top_signs = history.most_predicted(3)
        if top_signs:
            _text(frame, "Most frequent:", (px, py), F_SANS, 0.35, C_GREY, 1)
            py += 15
            for sign_name, cnt in top_signs:
                bar_pct = cnt / max(1, history.session_count)
                if py > h - bot_h - 100:
                    break
                _progress_bar(frame, px, py, side_w - 26, 6, bar_pct, C_BLUE)
                _text(frame, f"{sign_name[:8]:<8} {cnt}", (px, py + 16), F_SANS, 0.32, C_LGREY, 1)
                py += 22
        py += 4
        if py < h - bot_h - 60:
            cv2.line(frame, (px - 4, py), (w - 8, py), C_BLUE, 1)
            py += 10

        if py < h - bot_h - 40:
            _text(frame, "MODEL", (px, py + 8), F_SANS, 0.38, C_CYAN, 1)
            py += 20
            info_lines = [
                (f"BiLSTM {_MODEL_PARAMS // 1000}K params ({model_type})", C_LGREY),
                (f"val F1:  {_VAL_MACRO_F1:.4f}", C_GREEN),
                (f"test F1: {_TEST_MACRO_F1:.4f}", C_AMBER),
                (f"Size:    {_MODEL_SIZE_MB:.4f} MB", C_LGREY),
                (f"Thresh:  {display_threshold:.2f}", C_YELLOW),
                (f"Smoother: {smoother_window}fr", C_LGREY),
            ]
            for line, col in info_lines:
                if py > h - bot_h - 20:
                    break
                _text(frame, line, (px, py), F_SANS, 0.34, col, 1)
                py += 15

        if py < h - bot_h - 80:
            py += 4
            cv2.line(frame, (px - 4, py), (w - 8, py), C_BLUE, 1)
            py += 10
            _text(frame, "PIPELINE (ms)", (px, py + 8), F_SANS, 0.38, C_CYAN, 1)
            py += 20
            breakdown = fps_tracker.breakdown
            latency_items = [
                ("MediaPipe", breakdown.get("mediapipe", 0)),
                ("Pipeline", breakdown.get("pipeline", 0)),
                ("Inference", breakdown.get("inference", 0)),
                ("HUD", breakdown.get("hud", 0)),
            ]
            for label, ms in latency_items:
                if py > h - bot_h - 20:
                    break
                _progress_bar(frame, px, py, side_w - 60, 6, ms / 100.0, C_TEAL)
                _text(frame, f"{label:<10} {ms:>4.0f}", (px, py + 16), F_SANS, 0.32, C_LGREY, 1)
                py += 22

    def _draw_buffer_progress(self, frame: np.ndarray, w: int, h: int, frames_buffered: int, seq_len: int) -> None:
        pct = frames_buffered / max(1, seq_len)
        cx, cy = w // 2, h // 2
        box_w = min(440, w - 20)
        _alpha_rect(frame, cx - box_w // 2, cy - 40, cx + box_w // 2, cy + 40, C_DARK, alpha=0.85)
        cv2.rectangle(frame, (cx - box_w // 2, cy - 40), (cx + box_w // 2, cy + 40), C_BLUE, 1)

        ring_r = 28
        ring_cx = cx - box_w // 2 + 50
        angle = int(360 * pct)
        cv2.circle(frame, (ring_cx, cy), ring_r, (60, 60, 60), 3)
        if angle > 0:
            cv2.ellipse(frame, (ring_cx, cy), (ring_r, ring_r), -90, 0, angle, C_GREEN, 3)
        _text(frame, f"{int(pct * 100)}%", (ring_cx - 14, cy + 6), F_SANS, 0.45, C_WHITE, 1)

        _text(frame, "Building sequence buffer...", (ring_cx + 30, cy - 10), F_SANS, 0.55, C_WHITE, 1)
        _text(frame, f"{frames_buffered} / {seq_len} frames", (ring_cx + 30, cy + 16), F_SANS, 0.45, C_GREY, 1)
        _progress_bar(frame, ring_cx + 30, cy + 28, box_w - 100, 6, pct, C_GREEN)

    def _draw_no_hands_warning(self, frame: np.ndarray, w: int, h: int, bot_h: int) -> None:
        pulse = 0.5 + 0.5 * math.sin(self._pulse * 2)
        dot_r = int(8 + 4 * pulse)
        alpha_warn = 0.4 + 0.3 * pulse
        warn_x, warn_y = max(20, w // 2 - 120), h - bot_h - 30
        _alpha_rect(frame, warn_x - 10, warn_y - 20, min(w - 10, warn_x + 250), warn_y + 10,
                    C_DARK, alpha=alpha_warn)
        cv2.circle(frame, (warn_x - 2, warn_y - 6), dot_r, C_AMBER, -1)
        _text(frame, "No hands detected", (warn_x + 14, warn_y), F_SANS, 0.52, C_AMBER, 1)

    def _draw_bottom_bar(
        self, frame: np.ndarray, w: int, h: int, bot_h: int, fps_tracker: FPSTracker,
        meta: Dict[str, Any], is_paused: bool, is_frozen: bool,
    ) -> None:
        by = h - bot_h + 8
        fps = fps_tracker.fps
        fps_col = C_GREEN if fps >= 15 else C_AMBER if fps >= 8 else C_RED
        _text(frame, f"FPS {fps:>5.1f}", (10, by + 16), F_DUPLEX, 0.55, fps_col, 1)
        _progress_bar(frame, 10, by + 26, 80, 5, min(fps / 30.0, 1.0), fps_col)

        lh_col = C_GREEN if meta.get("left_detected") else (60, 60, 60)
        rh_col = C_GREEN if meta.get("right_detected") else (60, 60, 60)
        cv2.circle(frame, (130, by + 12), 6, lh_col, -1)
        _text(frame, "L", (127, by + 17), F_SANS, 0.35, C_WHITE, 1)
        cv2.circle(frame, (150, by + 12), 6, rh_col, -1)
        _text(frame, "R", (147, by + 17), F_SANS, 0.35, C_WHITE, 1)
        _text(frame, "Hands", (110, by + 32), F_SANS, 0.30, C_GREY, 1)

        if w >= 900:
            hints = "q:quit r:reset s:screenshot h:HUD m:skeleton SPACE:freeze +/-:thresh"
            _text(frame, hints, (max(180, w // 2 - len(hints) * 3), by + 36), F_SANS, 0.30, C_GREY, 1)

        ts = datetime.now().strftime("%H:%M:%S")
        _text(frame, ts, (max(170, w - 85), by + 18), F_SANS, 0.48, C_LGREY, 1)

        if is_frozen:
            cv2.rectangle(frame, (w - 85, by + 26), (w - 8, by + 42), C_CYAN, -1)
            _text(frame, "FROZEN", (w - 80, by + 39), F_SANS, 0.35, C_BLACK, 1, shadow_offset=0)
        elif is_paused:
            cv2.rectangle(frame, (w - 85, by + 26), (w - 8, by + 42), C_AMBER, -1)
            _text(frame, "PAUSED", (w - 80, by + 39), F_SANS, 0.35, C_BLACK, 1, shadow_offset=0)

        if w >= 600:
            _text(frame, "WLASL-35 | BiLSTM 68K | Henry Otsyula",
                  (max(10, w - 310), h - 4), F_SANS, 0.28, (80, 80, 80), 1)


# =============================================================================
# Camera enumeration utility (fix #F6)
# =============================================================================

def _list_cameras(max_index: int = 8) -> List[int]:
    """
    Probe camera indices [0, max_index) and return those that open AND
    deliver at least one frame. Used only by --list-cameras; not called in
    the normal startup path (keeps normal startup fast — opening N cameras
    just to enumerate them would add real latency every run).
    """
    found: List[int] = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                found.append(idx)
        cap.release()
    return found


# =============================================================================
# CLI
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="webcam_demo.py",
        description="Stage 9 — Production Webcam Demo: WLASL 35-Sign Gesture Recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    g_paths = p.add_argument_group("paths")
    g_paths.add_argument("--model", default=_DEFAULT_TFLITE_PATH,
                          help="Path to .tflite model (default: champion TFLite)")
    g_paths.add_argument("--config", default=_DEFAULT_CONFIG_SNAPSHOT,
                          help="Path to config_snapshot.yaml")
    g_paths.add_argument("--label-map", default=_DEFAULT_LABEL_MAP)
    g_paths.add_argument("--calib-report", default=_DEFAULT_CALIB_REPORT)

    g_cam = p.add_argument_group("camera")
    g_cam.add_argument("--camera", type=int, default=0, help="Camera device index")
    g_cam.add_argument("--width", type=int, default=1280)
    g_cam.add_argument("--height", type=int, default=720)
    g_cam.add_argument("--no-flip", action="store_true",
                        help="Disable mirror flip (default: enabled for natural feel). "
                             "Also flips the handedness mapping convention — see HandsExtractor.")
    g_cam.add_argument("--list-cameras", action="store_true",
                        help="Probe camera indices 0-7, print which ones deliver "
                             "frames, and exit without starting the demo.")

    g_model = p.add_argument_group("model")
    g_model.add_argument("--threshold", type=float, default=None,
                          help=f"Display confidence threshold in [0,1] "
                               f"(default: predictor's resolved threshold, ~{DISPLAY_THRESHOLD})")
    g_model.add_argument("--smoother", type=int, default=5,
                          help=f"Majority-vote window, clamped to "
                               f"[{_SMOOTHER_WINDOW_MIN},{_SMOOTHER_WINDOW_MAX}] (default: 5)")
    g_model.add_argument("--complexity", type=int, default=1, choices=[0, 1, 2],
                          help="MediaPipe model complexity (default: 1)")
    g_model.add_argument("--det-conf", type=float, default=0.5)
    g_model.add_argument("--track-conf", type=float, default=0.5)
    g_model.add_argument("--auto-reset", type=int, default=3,
                          help="Frames before auto-reset on no-detection (default: 3)")

    g_ui = p.add_argument_group("UI")
    g_ui.add_argument("--minimal-hud", action="store_true",
                       help="Minimal HUD (no right panel) for performance")
    g_ui.add_argument("--no-skeleton", action="store_true")
    g_ui.add_argument("--skeleton-stride", type=int, default=1,
                       help="Redraw skeleton every N frames (default: 1 = every frame)")
    g_ui.add_argument("--debounce", type=int, default=3,
                       help="Consecutive confident predictions before display update (default: 3)")

    g_io = p.add_argument_group("I/O")
    g_io.add_argument("--record", default=None, help="Record output to video file")
    g_io.add_argument("--warmup", type=int, default=3,
                       help="Predictor warmup passes before main loop (default: 3)")

    return p


def _print_startup_banner(args: argparse.Namespace, resolved_threshold: float) -> None:
    sep = "-" * 64
    print(f"\n{'=' * 64}")
    print("  WLASL 35-Sign Gesture Recognition - Stage 9 Demo")
    print("  Senior ML Engineer: Henry Otsyula")
    print(f"{'=' * 64}")
    print(f"  {sep}")
    print(f"  Model      : {Path(args.model).name}")
    print(f"  Config     : {Path(args.config).name}")
    print(f"  Camera     : device {args.camera}  ({args.width}x{args.height})")
    print(f"  Threshold  : {resolved_threshold:.2f}  (calibrated for underconfident model)")
    print(f"  Smoother   : {args.smoother}-frame majority vote")
    print(f"  Mirror     : {'OFF' if args.no_flip else 'ON (natural hand view)'}")
    print(f"  {sep}")
    print(f"  TFLite size  : {_MODEL_SIZE_MB:.4f} MB  |  Params : {_MODEL_PARAMS:,}")
    print(f"  Val macro-F1 : {_VAL_MACRO_F1:.4f}   |  Test F1 : {_TEST_MACRO_F1:.4f}")
    print(f"  Pipeline     : ~{_FULL_PIPELINE_MS:.0f} ms  (excl. MediaPipe)")
    print(f"  {sep}")
    print("  Controls:")
    print("    q / ESC  - quit              r       - HARD reset (incl. session stats)")
    print("    s        - screenshot        h       - toggle HUD")
    print("    m        - skeleton          SPACE   - freeze (camera + ML pipeline halt)")
    print("    p        - pause (ML only)   +/-     - adjust threshold   1-9 - smoother window")
    print(f"{'=' * 64}\n")


# =============================================================================
# Main demo loop
# =============================================================================

def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.list_cameras:
        print("Probing camera indices 0-7 ...")
        found = _list_cameras()
        if found:
            print(f"  Cameras delivering frames: {found}")
            print(f"  Example: python {Path(__file__).name} --camera {found[0]}")
        else:
            print("  No working camera found in range 0-7.")
        return 0

    args.smoother = max(_SMOOTHER_WINDOW_MIN, min(_SMOOTHER_WINDOW_MAX, args.smoother))
    args.auto_reset = max(1, args.auto_reset)
    args.debounce = max(1, args.debounce)
    args.skeleton_stride = max(1, args.skeleton_stride)
    if args.threshold is not None:
        args.threshold = _clamp(args.threshold, 0.0, 1.0)

    # ── Load GesturePredictor (public API only from here on) ────────────
    print("  [1/4] Loading GesturePredictor...", end="", flush=True)
    try:
        calib_path = args.calib_report if Path(args.calib_report).exists() else None
        if calib_path is None:
            print(f"\n  !  Calibration report not found at {args.calib_report}.")
            print(f"     Using hardcoded display threshold = {DISPLAY_THRESHOLD:.2f}")

        predictor = GesturePredictor.from_config_snapshot(
            config_snapshot_path=args.config,
            model_path=args.model,
            label_map_path=args.label_map,
            smoother_window=1,  # unused internal smoother — session manages its own
            display_threshold=args.threshold,
            calibration_report_path=calib_path,
        )
        # Resolved ONCE, here, to seed the session (fix #F2). From this point
        # forward, `current_threshold` (a local in this function) is the
        # single source of truth; predictor.display_threshold is never read
        # again, since it is read-only and cannot reflect runtime changes.
        resolved_threshold = (
            args.threshold if args.threshold is not None else predictor.display_threshold
        )
        print(f" OK  ({predictor.model_type}, {predictor.sequence_length} frames, "
              f"{predictor.feature_dim} dims, threshold={resolved_threshold:.2f})")
    except FileNotFoundError as e:
        print(f"\n  X  {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\n  X  GesturePredictor failed: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1

    _print_startup_banner(args, resolved_threshold)

    print(f"  [2/4] Warming up ({args.warmup} passes)...", end="", flush=True)
    t_wu = time.perf_counter()
    predictor.warmup(n_passes=args.warmup)
    print(f" OK  ({(time.perf_counter() - t_wu) * 1000:.0f} ms)")

    session = GestureStreamSession(
        predictor,
        smoother_window=args.smoother,
        display_threshold=resolved_threshold,
        n_top_k=3,
        auto_reset_no_detection_frames=args.auto_reset,
    )

    print("  [3/4] Initialising MediaPipe...", end="", flush=True)
    try:
        extractor = HandsExtractor(
            model_complexity=args.complexity,
            min_detection_confidence=args.det_conf,
            min_tracking_confidence=args.track_conf,
            mirrored=not args.no_flip,
        )
        print(f" OK  (mode={extractor.mode})")
    except Exception as e:
        print(f"\n  X  MediaPipe init failed: {e}", file=sys.stderr)
        predictor.close()
        return 1

    print(f"  [4/4] Opening camera {args.camera}...", end="", flush=True)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"\n  X  Camera {args.camera} not available.", file=sys.stderr)
        print("     Try: python src/demo/webcam_demo.py --list-cameras", file=sys.stderr)
        extractor.close()
        predictor.close()
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f" OK  ({actual_w}x{actual_h})")

    # First-frame sanity check (fix #F6): the single most common "demo
    # doesn't work" failure mode is a camera that *opens* successfully but
    # never delivers a frame (wrong backend, OS permission prompt still
    # pending, device claimed by another process). Catch it here with a
    # clear message rather than letting the main loop's retry/give-up logic
    # silently eat 5 seconds before reporting the same root cause.
    _first_ok, _first_frame = cap.read()
    if not _first_ok:
        print(
            f"\n  X  Camera {args.camera} opened but delivered no frame on "
            "the first read. This usually means another application has "
            "the camera open, an OS permission prompt is pending, or the "
            "wrong camera index was selected.", file=sys.stderr,
        )
        print("     Try: python src/demo/webcam_demo.py --list-cameras", file=sys.stderr)
        cap.release()
        extractor.close()
        predictor.close()
        return 1

    writer: Optional[cv2.VideoWriter] = None
    record_path: Optional[Path] = None
    if args.record:
        record_path = Path(args.record)
        record_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(record_path), fourcc, 30.0, (actual_w, actual_h))
        print(f"  Recording -> {record_path}")

    _SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    fps_tracker = FPSTracker(window=30)
    hud = HUDRenderer(minimal=args.minimal_hud, skeleton_stride=args.skeleton_stride)
    pred_history = PredictionHistory(debounce=args.debounce)
    show_hud = True
    show_skeleton = not args.no_skeleton
    is_paused = False
    frozen_result: Optional[Dict[str, Any]] = None
    # Fix #F4: explicit frozen snapshots of meta/landmarks, separate from the
    # live (still-being-overwritten-while-paused-only, never-while-frozen)
    # working variables below.
    frozen_meta: Dict[str, Any] = {"n_hands": 0}
    frozen_landmarks: Optional[np.ndarray] = None
    frame_count = 0
    consecutive_capture_failures = 0
    reopened_once = False
    latest_result: Optional[Dict[str, Any]] = None
    latest_meta: Dict[str, Any] = {"n_hands": 0}
    latest_raw_landmarks: Optional[np.ndarray] = None
    # Fix #F2: single source of truth for the live threshold, from here on.
    current_threshold = session.display_threshold

    window_name = "WLASL Gesture Recognition - Henry Otsyula"
    session_start_t = time.perf_counter()

    print("\n  Demo running. Press q to quit.\n")

    exit_reason = "ok"
    pending_first_frame: Optional[np.ndarray] = _first_frame

    try:
        with predictor:
            while True:
                t_cap = time.perf_counter()
                if pending_first_frame is not None:
                    # Consume the frame already read during the sanity check
                    # above instead of discarding it.
                    ret, frame = True, pending_first_frame
                    pending_first_frame = None
                else:
                    ret, frame = cap.read()

                if not ret:
                    consecutive_capture_failures += 1
                    logger.warning(
                        "Frame capture failed (%d/%d consecutive)",
                        consecutive_capture_failures, MAX_CONSECUTIVE_CAPTURE_FAILURES,
                    )
                    if (
                        consecutive_capture_failures == MAX_CONSECUTIVE_CAPTURE_FAILURES
                        and not reopened_once
                    ):
                        # One real reconnect attempt: release and reopen the
                        # device. A bare retry-read loop cannot recover from
                        # a USB device that was unplugged and replugged,
                        # since the original VideoCapture handle is dead.
                        print(
                            f"\n  !  {MAX_CONSECUTIVE_CAPTURE_FAILURES} consecutive "
                            f"failed reads — attempting to reopen camera "
                            f"{args.camera}...", file=sys.stderr,
                        )
                        cap.release()
                        time.sleep(0.5)
                        cap = cv2.VideoCapture(args.camera)
                        if cap.isOpened():
                            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
                            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
                            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            print("  !  Camera reopened — resuming.", file=sys.stderr)
                        else:
                            print("  !  Reopen failed — camera unavailable.", file=sys.stderr)
                        reopened_once = True
                        consecutive_capture_failures = 0
                        time.sleep(0.01)
                        continue
                    if (
                        reopened_once
                        and consecutive_capture_failures >= MAX_CONSECUTIVE_FAILURES_AFTER_REOPEN
                    ):
                        print(
                            "\n  X  Camera still unavailable after reopen "
                            f"({MAX_CONSECUTIVE_FAILURES_AFTER_REOPEN} more consecutive "
                            "failed reads). Exiting.", file=sys.stderr,
                        )
                        exit_reason = "camera_lost"
                        break
                    if (
                        not reopened_once
                        and consecutive_capture_failures >= MAX_CONSECUTIVE_CAPTURE_FAILURES
                    ):
                        # Shouldn't normally reach here (handled above), but
                        # guards against a future edit changing the ordering.
                        exit_reason = "camera_lost"
                        break
                    time.sleep(0.01)
                    continue
                consecutive_capture_failures = 0

                if not args.no_flip:
                    frame = cv2.flip(frame, 1)

                fps_tracker.record("capture", (time.perf_counter() - t_cap) * 1000)
                frame_count += 1

                # Freeze halts the entire ML pipeline exactly like pause
                # does — landmarks/meta are simply not refreshed this frame.
                if not is_paused and frozen_result is None:
                    t_mp = time.perf_counter()
                    landmarks_225, meta = extractor.extract(frame)
                    fps_tracker.record("mediapipe", (time.perf_counter() - t_mp) * 1000)

                    latest_raw_landmarks = landmarks_225
                    latest_meta = meta

                    latest_result, event = session.update(landmarks_225)

                    # Fix #F1: record only genuinely measured stage timings,
                    # taken directly from the session's own measurements —
                    # never a synthetic 0.0 for "the stage that didn't run".
                    if latest_result is not None:
                        fps_tracker.record("pipeline", latest_result["pipeline_ms"])
                        fps_tracker.record("inference", latest_result["inference_ms"])
                    # else: buffer still filling or just auto-reset — no
                    # forward pass happened this frame, so nothing is
                    # recorded for "pipeline"/"inference" at all (not even
                    # a placeholder). stage_median() correctly reflects only
                    # frames where the stage actually ran.

                    # Fix #F5: structured event instead of a bare bool.
                    if event["auto_reset"]:
                        pred_history.update(None)  # do not corrupt debounce on a hard reset
                        logger.info(
                            "Auto-reset fired after %d consecutive no-detection frames.",
                            event["no_detection_streak"],
                        )

                    pred_history.update(latest_result)

                t_hud = time.perf_counter()
                if show_hud:
                    # Fix #F4: while frozen, the HUD renders from the frozen
                    # snapshot for landmarks/meta as well as for the result,
                    # so all three stay mutually consistent by construction
                    # rather than by relying on the extraction block being
                    # skipped.
                    render_meta = frozen_meta if frozen_result is not None else latest_meta
                    render_landmarks = (
                        frozen_landmarks if frozen_result is not None else latest_raw_landmarks
                    )
                    frame = hud.render(
                        frame=frame,
                        result=latest_result,
                        meta=render_meta,
                        history=pred_history,
                        fps_tracker=fps_tracker,
                        frames_buffered=session.frames_buffered,
                        seq_len=session.sequence_length,
                        model_type=predictor.model_type,
                        display_threshold=current_threshold,
                        smoother_window=session.smoother_window,
                        show_skeleton=show_skeleton,
                        is_paused=is_paused,
                        is_frozen=frozen_result is not None,
                        frozen_result=frozen_result,
                        latest_raw_landmarks=render_landmarks,
                    )
                fps_tracker.record("hud", (time.perf_counter() - t_hud) * 1000)

                cv2.imshow(window_name, frame)
                if writer is not None:
                    writer.write(frame)
                fps_tracker.tick()

                if frame_count % 150 == 0:
                    logger.info(
                        "FPS=%.1f | buffered=%d/%d | no_detect_streak=%d | "
                        "session_signs=%d | mp_median=%.0fms | inf_median=%.0fms | "
                        "pipe_median=%.0fms",
                        fps_tracker.fps, session.frames_buffered, session.sequence_length,
                        session.no_detection_streak, pred_history.session_count,
                        fps_tracker.stage_median("mediapipe"), fps_tracker.stage_median("inference"),
                        fps_tracker.stage_median("pipeline"),
                    )

                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):
                    break

                elif key == ord("r"):
                    session.reset()
                    pred_history.reset()  # hard reset — clears session stats too
                    latest_result = None
                    frozen_result = None
                    frozen_meta = {"n_hands": 0}
                    frozen_landmarks = None
                    print(f"  [reset] Hard reset at frame {frame_count}")

                elif key == ord("s"):
                    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    sign_for_name = _sanitize_filename_component(
                        (latest_result or {}).get("sign", "none")
                    )
                    shot = _SCREENSHOT_DIR / f"wlasl_{sign_for_name}_{ts_str}.png"
                    cv2.imwrite(str(shot), frame)
                    print(f"  [screenshot] -> {shot}")

                elif key == ord("h"):
                    show_hud = not show_hud
                    print(f"  HUD {'ON' if show_hud else 'OFF'}")

                elif key == ord("m"):
                    show_skeleton = not show_skeleton
                    print(f"  Skeleton {'ON' if show_skeleton else 'OFF'}")

                elif key == ord(" "):
                    if frozen_result is not None:
                        frozen_result = None
                        frozen_meta = {"n_hands": 0}
                        frozen_landmarks = None
                        print("  [freeze] Unfrozen — ML pipeline resumed")
                    elif latest_result is not None:
                        # Fix #F4: snapshot result, meta, AND landmarks
                        # together, atomically, so the HUD's skeleton can
                        # never drift from the frozen prediction even if a
                        # future refactor changes where extraction happens
                        # relative to this block.
                        frozen_result = dict(latest_result)
                        frozen_meta = dict(latest_meta)
                        frozen_landmarks = (
                            latest_raw_landmarks.copy()
                            if latest_raw_landmarks is not None else None
                        )
                        print(f"  [freeze] Frozen: {frozen_result.get('sign')} "
                              f"({frozen_result.get('confidence', 0):.0%}) — ML pipeline halted")

                elif key == ord("p"):
                    is_paused = not is_paused
                    print(f"  {'[pause] Paused' if is_paused else '[pause] Resumed'}")

                elif key in (ord("+"), ord("=")):
                    current_threshold = _clamp(current_threshold + 0.05, 0.05, 0.99)
                    session.display_threshold = current_threshold
                    print(f"  Threshold up -> {current_threshold:.2f}")

                elif key in (ord("-"), ord("_")):
                    current_threshold = _clamp(current_threshold - 0.05, 0.05, 0.99)
                    session.display_threshold = current_threshold
                    print(f"  Threshold down -> {current_threshold:.2f}")

                elif ord("1") <= key <= ord("9"):
                    w_val = session.set_smoother_window(int(chr(key)))
                    print(f"  Smoother window -> {w_val}")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C).")
        exit_reason = "keyboard_interrupt"
    except Exception as exc:
        print(f"\n\nUnhandled error in main loop: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        exit_reason = "error"
    finally:
        # Guaranteed cleanup regardless of how the loop above exited (normal
        # quit, camera loss, exception, Ctrl+C).
        cap.release()
        if writer is not None:
            writer.release()
            print(f"\n  Recording saved -> {record_path}")
        cv2.destroyAllWindows()
        extractor.close()

    elapsed_s = max(1e-6, time.perf_counter() - session_start_t)
    avg_fps = frame_count / elapsed_s  # true session average, not rolling

    breakdown = fps_tracker.breakdown
    most_predicted = pred_history.most_predicted(5)

    print(f"\n{'=' * 64}")
    print("  SESSION SUMMARY")
    print(f"  {'-' * 60}")
    print(f"  Exit reason      : {exit_reason}")
    print(f"  Frames processed : {frame_count:,}")
    print(f"  Session duration : {elapsed_s:.1f} s")
    print(f"  Average FPS      : {avg_fps:.1f}")
    print(f"  Signs predicted  : {pred_history.session_count}")
    if most_predicted:
        print("  Top predictions  :")
        for sign_name, cnt in most_predicted:
            bar = "#" * min(cnt, 20)
            print(f"    {sign_name:<14} {cnt:>4}  {bar}")
    print(f"  {'-' * 60}")
    print("  Pipeline latency (median, genuine measurements only):")
    for stage, ms in breakdown.items():
        n_samples = fps_tracker.stage_sample_count(stage)
        if n_samples > 0:
            print(f"    {stage:<12} {ms:>6.1f} ms   (n={n_samples})")
    print(f"{'=' * 64}\n")

    # Fix #F7: optional session-summary sidecar next to a recording, so the
    # numbers printed above survive independently of the terminal scrollback
    # when attaching objective metrics to a demo recording in the README.
    if record_path is not None:
        sidecar = record_path.with_suffix("").with_name(record_path.stem + ".session.json")
        try:
            with open(sidecar, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "exit_reason": exit_reason,
                        "frames_processed": frame_count,
                        "session_duration_s": round(elapsed_s, 2),
                        "average_fps": round(avg_fps, 2),
                        "signs_predicted": pred_history.session_count,
                        "session_counts": pred_history.session_counts,
                        "pipeline_latency_ms_median": {
                            stage: round(ms, 2)
                            for stage, ms in breakdown.items()
                            if fps_tracker.stage_sample_count(stage) > 0
                        },
                        "model": {
                            "model_type": predictor.model_type,
                            "val_macro_f1": _VAL_MACRO_F1,
                            "test_macro_f1": _TEST_MACRO_F1,
                            "tflite_size_mb": _MODEL_SIZE_MB,
                            "display_threshold_final": current_threshold,
                            "smoother_window_final": session.smoother_window,
                        },
                    },
                    f, indent=2,
                )
            print(f"  Session summary -> {sidecar}\n")
        except OSError as exc:
            logger.warning("Could not write session summary sidecar: %s", exc)

    return 0 if exit_reason in ("ok", "keyboard_interrupt") else 1


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    sys.exit(main())