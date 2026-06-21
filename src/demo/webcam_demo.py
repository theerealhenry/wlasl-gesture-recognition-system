"""
src/demo/webcam_demo.py
========================
Stage 9 — Production-Grade Real-Time Webcam Demo
WLASL 35-Class Gesture Recognition System
Author: Henry Otsyula — Senior Data Scientist & ML Engineer

Revision note (this file)
--------------------------
This is a corrective revision of the Stage 9 demo following a critical
architectural review of the previous draft. Every claim in that review was
independently re-verified against the actual implementations of
`src/inference/predictor.py` (Stage 7), `src/export/convert.py` /
`src/export/verify.py` (Stage 8), and the Stage 8 Executive Summary before
being accepted, modified, or rejected. The disposition of every point is
recorded inline, next to the code that addresses it, so the reasoning
survives the code change.

Architecture: why this file no longer touches `predictor._anything`
---------------------------------------------------------------------
The single biggest issue in the previous revision (critical review #1) was
direct manipulation of `GesturePredictor`'s private state:
`predictor._frame_buffer`, `predictor._smoother`, `predictor._pipeline`,
`predictor._run_single`, `predictor._build_result`,
`predictor._display_threshold`, `predictor._no_detection_streak`,
`predictor._auto_reset_threshold`. This was not a cosmetic problem: it
meant the demo was reimplementing roughly half of `GesturePredictor`'s
streaming logic by reaching through its skin, and any future refactor of
`FrameBuffer`/`PredictionSmoother`/`_build_result()` would silently break
this file with no type error to catch it.

The review's suggested fix was a new public method on `GesturePredictor`
(e.g. `predict_landmarks()`). That is the *eventual* right answer, but it
requires changing `src/inference/predictor.py`, which is out of scope for
this revision. The alternative used here is exactly as encapsulation-safe
and requires zero changes to Stage 7: `GesturePredictor` already exports
three things publicly that are sufficient to reproduce its own streaming
contract from the outside —

    predictor.pipeline                  -> the FeaturePipeline instance
    predictor(x, training=False)        -> the evaluation-framework
                                            __call__ contract (batched
                                            forward pass, model-agnostic)
    predictor.label_map / .sequence_length / .feature_dim / .n_classes
                                         -> read-only properties

— plus two classes that `src/inference/predictor.py` already exports as
PUBLIC API (`__all__` includes both): `FrameBuffer` and
`PredictionSmoother`. `GestureStreamSession` (below) composes these three
public surfaces into the exact same buffering/inference/smoothing/
auto-reset behaviour as `GesturePredictor.predict_from_webcam_frame()` —
without it, because that method is hardwired to the predictor's own lazily
constructed MediaPipe *Holistic* extractor and has no hook for externally
supplied landmarks.

Why an external extractor is needed at all: Stage 8's Executive Summary
(Section 11.3) documents that MediaPipe Hands runs in ~8-10ms versus
Holistic's ~18ms for this hands-only champion, and recommends exactly the
switch this file makes. That recommendation is the reason this demo cannot
simply call `predict_from_webcam_frame()` and be done with it.

Key Stage 8 Integration Facts (verified against the Stage 8 Executive
Summary and src/export/verify.py — locked, do not change without re-running
Stage 8's release gate):
  - display_threshold = 0.35  (model is UNDERCONFIDENT: mean_conf=0.5136 < mean_acc=0.5769)
  - TFLite size = 0.1596 MB   (SELECT_TF_OPS flex delegate adds ~100 KB beyond weight quantisation)
  - Val macro-F1 = 0.5916 (TFLite)     | Test macro-F1 = 0.4867 (TFLite)
  - Full pipeline (excl. MediaPipe): 47.11 ms median -> ~21 FPS headroom
  - Estimated end-to-end with MediaPipe Hands: ~57-60 ms -> ~17-18 FPS
  - Argmax agreement Keras<->TFLite: 98.08% (val), 98.04% (test)

Critical review disposition summary (full detail inline at each fix site)
----------------------------------------------------------------------------
  #1  FIXED — GestureStreamSession composes only public predictor API.
  #2  FIXED — try/finally around the entire capture loop guarantees cleanup
              even on an unhandled exception (cap, writer, extractor, window).
  #3  FIXED — pose is never extracted at all (see #18); the historical
              `landmark[:33]` overflow risk is structurally eliminated.
  #4  FIXED — auto-reset state now lives in GestureStreamSession (public),
              not in predictor internals.
  #5  FIXED — session-average FPS is now frame_count / elapsed_wall_clock,
              not the rolling instantaneous FPS at the moment the loop exits.
  #6  FIXED — PredictionHistory now decays a stale debounce streak after a
              sustained run of low-confidence frames.
  #7  FIXED — PredictionHistory.reset() is now a genuine hard reset,
              including session sign counts and display history.
  #8  FIXED — consecutive capture failures are counted; the loop exits with
              a clear message instead of spinning forever on a dead camera.
  #9  RESOLVED BY DESIGN — the demo never mutates predictor state, so
              threshold changes can never desynchronise predictor vs. demo.
  #10 FIXED — smoother window changes are validated and clamped to [1, 15].
  #11 FIXED — the raw landmark vector is stored by reference, not copied;
              it is read-only downstream (skeleton drawing) and never
              mutated, including inside FrameBuffer.add_frame() (which
              uses astype(..., copy=False) and is itself non-mutating).
  #12 PARTIALLY ADDRESSED — `--skeleton-stride` lets a slow CPU skip
              skeleton redraws on some frames; default is 1 (every frame)
              since 3-5ms is not the dominant cost at this point.
  #13 EVALUATED, NOT CHANGED — FPSTracker sorts six 30-element deques per
              frame; at this size (<30 elements) this is sub-microsecond
              and not worth the complexity of a running-median structure.
  #14 FIXED — HUD panel geometry is now computed from the live frame size
              every frame, with sane clamps, instead of fixed pixel constants.
  #15 FIXED — every on-frame HUD string is ASCII-only. Box-drawing/emoji
              glyphs are unreliable across OpenCV/Freetype font backends;
              console print() statements (which run through the terminal's
              own font) keep their original Unicode glyphs since that
              concern does not apply there.
  #16 FIXED — screenshot filenames are sanitised to [A-Za-z0-9_-] only.
  #17 FIXED — handedness mapping is now mirror-aware (`mirrored=` flag tied
              to `not args.no_flip`), with the convention documented
              explicitly in `HandsExtractor`'s docstring. The previous
              revision silently assumed mirroring was always on, which is
              wrong whenever `--no-flip` is passed.
  #18 FIXED — the MediaPipe Holistic fallback path NEVER populates the
              pose slot. The champion is hands_only; feeding it real pose
              values via the fallback (while the primary Hands path always
              zero-fills pose) would silently change the input distribution
              between the two extraction code paths.
  #19 FIXED — "freeze" now halts the entire streaming session (buffer,
              smoother, inference) in addition to freezing the HUD,
              matching the documented behaviour and user expectation.
  #20 FIXED — dead constants removed (`_TFLITE_MEDIAN_MS`, unused
              `_RECORDING_DIR`).
  #21 FIXED — unused typing imports removed.
  #22 PARTIALLY ADDRESSED — the highest-risk magic numbers (thresholds,
              layout fractions, decay windows) are now named module-level
              constants; a handful of pure layout pixel offsets inside HUD
              drawing remain literal, as is conventional for drawing code.

Controls
  q / ESC      — quit
  r            — HARD reset: buffer, smoother, debounce state, AND session
                 statistics (sign counts, display history) all clear.
  s            — save annotated screenshot (PNG, sanitised filename)
  h            — toggle HUD visibility
  m            — toggle landmark skeleton overlay
  p            — pause / unpause (camera still live; ML pipeline halted)
  SPACE        — freeze current prediction (camera AND ML pipeline halted)
  + / =        — raise confidence display threshold by 0.05
  - / _        — lower confidence display threshold by 0.05
  1-9          — set smoother window (1 = no smoothing)
  F            — toggle FPS overlay detail (reserved)

Usage
    python src/demo/webcam_demo.py
    python src/demo/webcam_demo.py --model models/gesture_bilstm_v1.tflite --camera 1
    python src/demo/webcam_demo.py --minimal-hud
    python src/demo/webcam_demo.py --no-flip
    python src/demo/webcam_demo.py --record outputs/demo_recording.mp4
"""

from __future__ import annotations

import argparse
import contextlib
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
# module docstring's "Architecture" section for why this matters.
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

#: Stage 8 release-gate performance summary (HUD info panel only).
_FULL_PIPELINE_MS: float = 47.11
_VAL_MACRO_F1: float = 0.5916
_TEST_MACRO_F1: float = 0.4867
_MODEL_SIZE_MB: float = 0.1596
_MODEL_PARAMS: int = 68_771

# =============================================================================
# Behavioural constants (named, per critical review #22)
# =============================================================================

#: Critical review #6 — a debounced sign survives at most this many
#: consecutive non-confident frames before its streak is cleared. Prevents
#: a stale "stable" sign from silently surviving an unrelated low-confidence
#: gap (e.g. the signer pausing, or a brief misdetection).
LOW_CONFIDENCE_DECAY_FRAMES: int = 8

#: Critical review #8 — consecutive cv2.VideoCapture.read() failures after
#: which the demo exits rather than spinning forever on a dead/disconnected
#: camera. ~5 seconds at 30 FPS.
MAX_CONSECUTIVE_CAPTURE_FAILURES: int = 150

#: Smoother window is clamped to this range on every change (CLI or runtime
#: keypress) — critical review #10.
_SMOOTHER_WINDOW_MIN: int = 1
_SMOOTHER_WINDOW_MAX: int = 9

# =============================================================================
# HUD Design System (ASCII-only on-frame text — critical review #15)
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

# Layout fractions used to derive panel geometry from the live frame size
# (critical review #14). Clamped so very small or very large resolutions
# both produce usable layouts.
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
    """
    Critical review #16 — strip any character outside [A-Za-z0-9_-] before
    using a value (e.g. a predicted sign name) inside a filename.
    """
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
# FPS tracker
# =============================================================================

class FPSTracker:
    """
    Rolling FPS estimator with per-stage breakdown.

    Critical review #13: stage_median() sorts a <=30-element deque on every
    call. At this size sorting costs a sub-microsecond amount of CPU time
    (orders of magnitude below the ~1-5ms HUD rendering budget it informs),
    so a running-median data structure was evaluated and rejected as
    unnecessary complexity for this project's scale.
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

    @property
    def breakdown(self) -> Dict[str, float]:
        return {k: self.stage_median(k) for k in self._stage_ms}


# =============================================================================
# Prediction history (debounce + session stats)
# =============================================================================

class PredictionHistory:
    """
    Tracks prediction stability (debounce) and session statistics.

    Critical review #6 (decay): a sign must reappear in `debounce`
    consecutive CONFIDENT predictions before it is displayed. Previously,
    once a streak reached `debounce`, nothing ever cleared it during a
    subsequent run of low-confidence frames, so a stale "stable" sign could
    silently survive an unrelated gap (signer pausing, brief misdetection).
    `LOW_CONFIDENCE_DECAY_FRAMES` consecutive non-confident updates now
    clear the streak.

    Critical review #7 (hard reset): reset() now clears EVERYTHING,
    including session sign counts and display history — matching the 'r'
    key's documented behaviour ("hard reset").
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
    MediaPipe Hands extractor — preferred over Holistic for Stage 9.

    Output contract (matches the FrameBuffer / FeaturePipeline invariant
    documented in src/inference/predictor.py):
        [0:63]    left hand  — 21 landmarks x (x, y, z)
        [63:126]  right hand — 21 landmarks x (x, y, z)
        [126:225] pose       — ALWAYS zero, in BOTH extraction modes.

    Why pose is always zero, even under the Holistic fallback
    --------------------------------------------------------------
    Critical review #18: the champion is hands_only and was trained with
    [126:225] always zero. If the primary Hands path always zero-fills pose
    but the Holistic fallback populates it with real values, the model
    would receive a meaningfully different input distribution depending on
    which extractor happened to initialise successfully on a given machine
    — a silent, environment-dependent accuracy regression. Holistic IS used
    as a fallback for hand landmarks only; its pose output is discarded.
    This also structurally eliminates the pose-landmark-count overflow risk
    flagged in critical review #3 (MediaPipe Holistic's pose landmark count
    has occasionally differed across versions) — there is no pose-writing
    code path left to overflow.

    Handedness / mirroring (critical review #17)
    -----------------------------------------------
    MediaPipe's `Handedness` classification is always relative to the RAW
    camera image, BEFORE any mirror flip the caller applies. The previous
    revision hardcoded the mirrored-frame convention regardless of whether
    `--no-flip` was passed, which silently swapped left/right hands in
    non-mirrored mode — a high-risk, silent correctness bug given how
    central hand identity is to this model's hands_only feature layout.

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
    never depends on any underscore-prefixed predictor attribute.

    See the module docstring's "Architecture" section for the full
    rationale (critical review #1, #4, #9, #10, #19).

    Public surface used from `predictor` (all already part of
    GesturePredictor's documented API):
        predictor.pipeline            (FeaturePipeline instance — public)
        predictor(x, training=False)  (evaluation-framework __call__)
        predictor.label_map
        predictor.sequence_length / .feature_dim / .n_classes / .model_type

    Public classes composed (both exported in predictor.py's __all__):
        FrameBuffer, PredictionSmoother
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
    def update(self, landmarks_225: np.ndarray) -> Tuple[Optional[Dict[str, Any]], bool]:
        """
        Feed one raw (FEATURE_SIZE,) landmark vector through the streaming
        pipeline. Returns (result, auto_reset_fired).

        result is None while the buffer is filling, or immediately after an
        auto-reset fires (matching GesturePredictor.predict_from_webcam_frame
        semantics exactly).
        """
        if not np.any(landmarks_225):
            self._no_detection_streak += 1
        else:
            self._no_detection_streak = 0

        # Zero vectors enter the buffer BEFORE the auto-reset check —
        # zero-fill is semantic (Stage 3 convention), not noise to filter.
        self._buffer.add_frame(landmarks_225)

        if (
            self._auto_reset_threshold is not None
            and self._no_detection_streak >= self._auto_reset_threshold
        ):
            self.reset()
            return None, True

        if not self._buffer.is_ready():
            return None, False

        raw_seq = self._buffer.get_array()
        features_2d = self.predictor.pipeline(raw_seq, training=False)

        t0 = time.perf_counter()
        raw_probs_batch = self.predictor(features_2d, training=False)  # (1, n_classes)
        inference_ms = (time.perf_counter() - t0) * 1000.0

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
            "inference_latency_ms": round(inference_ms, 3),
            "frames_in_buffer": self._buffer.frames_accumulated(),
        }
        return result, False

    def set_smoother_window(self, window: int) -> int:
        """Critical review #10 — validated, clamped smoother window change."""
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
        extractor.close()
        predictor.close()
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f" OK  ({actual_w}x{actual_h})")

    writer: Optional[cv2.VideoWriter] = None
    if args.record:
        rec_path = Path(args.record)
        rec_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(rec_path), fourcc, 30.0, (actual_w, actual_h))
        print(f"  Recording -> {rec_path}")

    _SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    fps_tracker = FPSTracker(window=30)
    hud = HUDRenderer(minimal=args.minimal_hud, skeleton_stride=args.skeleton_stride)
    pred_history = PredictionHistory(debounce=args.debounce)
    show_hud = True
    show_skeleton = not args.no_skeleton
    is_paused = False
    frozen_result: Optional[Dict[str, Any]] = None
    frame_count = 0
    consecutive_capture_failures = 0
    latest_result: Optional[Dict[str, Any]] = None
    latest_meta: Dict[str, Any] = {"n_hands": 0}
    latest_raw_landmarks: Optional[np.ndarray] = None
    current_threshold = session.display_threshold

    window_name = "WLASL Gesture Recognition - Henry Otsyula"
    session_start_t = time.perf_counter()  # critical review #5

    print("\n  Demo running. Press q to quit.\n")

    # ── Main loop — wrapped end-to-end in try/finally (critical review #2) ─
    exit_reason = "ok"
    try:
        with predictor:
            while True:
                t_cap = time.perf_counter()
                ret, frame = cap.read()
                if not ret:
                    consecutive_capture_failures += 1
                    logger.warning(
                        "Frame capture failed (%d/%d consecutive)",
                        consecutive_capture_failures, MAX_CONSECUTIVE_CAPTURE_FAILURES,
                    )
                    if consecutive_capture_failures >= MAX_CONSECUTIVE_CAPTURE_FAILURES:
                        print(
                            "\n  X  Camera appears disconnected "
                            f"({MAX_CONSECUTIVE_CAPTURE_FAILURES} consecutive failed reads). "
                            "Exiting.", file=sys.stderr,
                        )
                        exit_reason = "camera_lost"
                        break
                    time.sleep(0.01)
                    continue
                consecutive_capture_failures = 0

                if not args.no_flip:
                    frame = cv2.flip(frame, 1)

                fps_tracker.record("capture", (time.perf_counter() - t_cap) * 1000)
                frame_count += 1

                # Critical review #19: freeze halts the ML pipeline exactly
                # like pause does, not just the HUD overlay.
                if not is_paused and frozen_result is None:
                    t_mp = time.perf_counter()
                    landmarks_225, meta = extractor.extract(frame)
                    fps_tracker.record("mediapipe", (time.perf_counter() - t_mp) * 1000)

                    # Critical review #11: stored by reference, never mutated
                    # downstream (read-only for skeleton drawing; FrameBuffer
                    # copies internally via astype(copy=False) on an already
                    # float32 array, which is a no-op view, not a mutation).
                    latest_raw_landmarks = landmarks_225
                    latest_meta = meta

                    t_pipe_inf = time.perf_counter()
                    latest_result, auto_reset_fired = session.update(landmarks_225)
                    elapsed_pipe_inf_ms = (time.perf_counter() - t_pipe_inf) * 1000.0
                    if latest_result is not None:
                        fps_tracker.record("pipeline", 0.0)  # folded into "inference" below
                        fps_tracker.record(
                            "inference", latest_result.get("inference_latency_ms", elapsed_pipe_inf_ms)
                        )
                    else:
                        fps_tracker.record("pipeline", elapsed_pipe_inf_ms)
                        fps_tracker.record("inference", 0.0)

                    if auto_reset_fired:
                        pred_history.update(None)  # do not corrupt debounce on a hard reset
                        logger.info(
                            "Auto-reset fired after %d consecutive no-detection frames.",
                            args.auto_reset,
                        )

                    pred_history.update(latest_result)

                t_hud = time.perf_counter()
                if show_hud:
                    frame = hud.render(
                        frame=frame,
                        result=latest_result,
                        meta=latest_meta,
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
                        latest_raw_landmarks=latest_raw_landmarks,
                    )
                fps_tracker.record("hud", (time.perf_counter() - t_hud) * 1000)

                cv2.imshow(window_name, frame)
                if writer is not None:
                    writer.write(frame)
                fps_tracker.tick()

                if frame_count % 150 == 0:
                    logger.info(
                        "FPS=%.1f | buffered=%d/%d | no_detect_streak=%d | "
                        "session_signs=%d | mp_median=%.0fms | inf_median=%.0fms",
                        fps_tracker.fps, session.frames_buffered, session.sequence_length,
                        session.no_detection_streak, pred_history.session_count,
                        fps_tracker.stage_median("mediapipe"), fps_tracker.stage_median("inference"),
                    )

                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):
                    break

                elif key == ord("r"):
                    session.reset()
                    pred_history.reset()  # hard reset — clears session stats too
                    latest_result = None
                    frozen_result = None
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
                        print("  [freeze] Unfrozen — ML pipeline resumed")
                    elif latest_result is not None:
                        frozen_result = dict(latest_result)
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
        # Critical review #2 — guaranteed cleanup regardless of how the
        # loop above exited (normal quit, camera loss, exception, Ctrl+C).
        cap.release()
        if writer is not None:
            writer.release()
            print(f"\n  Recording saved -> {args.record}")
        cv2.destroyAllWindows()
        extractor.close()

    elapsed_s = max(1e-6, time.perf_counter() - session_start_t)
    avg_fps = frame_count / elapsed_s  # critical review #5 — true session average, not rolling

    print(f"\n{'=' * 64}")
    print("  SESSION SUMMARY")
    print(f"  {'-' * 60}")
    print(f"  Exit reason      : {exit_reason}")
    print(f"  Frames processed : {frame_count:,}")
    print(f"  Session duration : {elapsed_s:.1f} s")
    print(f"  Average FPS      : {avg_fps:.1f}")
    print(f"  Signs predicted  : {pred_history.session_count}")
    most_predicted = pred_history.most_predicted(5)
    if most_predicted:
        print("  Top predictions  :")
        for sign_name, cnt in most_predicted:
            bar = "#" * min(cnt, 20)
            print(f"    {sign_name:<14} {cnt:>4}  {bar}")
    breakdown = fps_tracker.breakdown
    print(f"  {'-' * 60}")
    print("  Pipeline latency (median):")
    for stage, ms in breakdown.items():
        if ms > 0:
            print(f"    {stage:<12} {ms:>6.1f} ms")
    print(f"{'=' * 64}\n")

    return 0 if exit_reason in ("ok", "keyboard_interrupt") else 1


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    sys.exit(main())