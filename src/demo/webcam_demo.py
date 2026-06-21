"""
src/demo/webcam_demo.py
========================
Stage 9 — Production-Grade Real-Time Webcam Demo
WLASL 35-Class Gesture Recognition System
Author: Henry Otsyula — Senior Data Scientist & ML Engineer

Architecture:
  MediaPipe Hands (optimised) / Holistic fallback
  → FrameBuffer (rolling 100-frame window, full 225-dim raw landmarks)
  → FeaturePipeline (wrist-relative norm → z-clip ±0.10 → pad/truncate)
  → gesture_bilstm_v1.tflite (SELECT_TF_OPS flex delegate, 68,771 params)
  → PredictionSmoother (5-frame majority vote + exponential smoothing)
  → Production HUD (multi-panel, animated, calibration-aware display)

Key Stage 8 Integration Facts (locked — do not change):
  - display_threshold = 0.35  (model is UNDERCONFIDENT: mean_conf=0.5136 < mean_acc=0.5769)
  - TFLite size = 0.1596 MB   (SELECT_TF_OPS adds ~100 KB beyond weight quantisation)
  - Val macro-F1 = 0.5916     (TFLite, 52 clips, 7 unseen signers)
  - Test macro-F1 = 0.4867    (TFLite slightly outperforms Keras: quantisation regularisation)
  - Full pipeline (excl. MediaPipe): 47.11 ms median → ~21 FPS headroom
  - Estimated end-to-end: ~70 ms → ~14-15 FPS on this development machine
  - Argmax agreement Keras↔TFLite: 98.08% (val), 98.04% (test)

Critical Implementation Rules (from Part 8 + Stage 8 guide):
  1. GesturePredictor.from_config_snapshot() — NEVER reconstruct via load_config()
  2. warmup(n_passes=3) BEFORE main loop — JIT latency can exceed 700ms
  3. NEVER use Keras SavedModel in loop — 3862ms/frame (0.26 FPS)
  4. FrameBuffer stores FULL 225-dim raw vectors — slicing happens inside pipeline
  5. Zero-fill frames MUST enter the buffer (semantic, not noise)
  6. auto_reset_no_detection_frames=3 — signer pausing must not pollute buffer
  7. MediaPipe Hands preferred over Holistic (~8-10ms vs ~18ms per frame)
     → Populate [0:63] left hand, [63:126] right hand, zero-fill [126:225] pose

Controls:
  q / ESC      — quit
  r            — hard reset (buffer + smoother + prediction history)
  s            — save annotated screenshot (PNG)
  h            — toggle HUD visibility
  m            — toggle mini-map (landmark skeleton overlay)
  p            — pause / unpause
  + / =        — raise confidence display threshold by 0.05
  - / _        — lower confidence display threshold by 0.05
  1-5          — set smoother window (1=no smoothing, 5=default)
  F            — toggle FPS overlay detail
  SPACE        — freeze current prediction (useful for demos)

Usage:
    # Default (champion TFLite):
    python src/demo/webcam_demo.py

    # Custom model / camera:
    python src/demo/webcam_demo.py --model models/gesture_bilstm_v1.tflite --camera 1

    # High-performance mode (reduced HUD overhead):
    python src/demo/webcam_demo.py --minimal-hud

    # Record output video:
    python src/demo/webcam_demo.py --record outputs/demo_recording.mp4
"""

from __future__ import annotations

import argparse
import contextlib
import math
import sys
import time
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Repository root resolution ─────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np

from src.inference.predictor import GesturePredictor
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
_DEFAULT_LABEL_MAP   = str(_REPO_ROOT / "artifacts" / "label_map_v1.json")
_DEFAULT_CALIB_REPORT = str(
    _REPO_ROOT / "reports" / "evaluation" / "evaluation_report.json"
)
_SCREENSHOT_DIR = _REPO_ROOT / "reports" / "figures" / "webcam_screenshots"
_RECORDING_DIR  = _REPO_ROOT / "reports" / "recordings"

# =============================================================================
# Stage 8 calibration constants (locked — derived from Stage 8 gate results)
# =============================================================================

#: Display threshold calibrated to champion's underconfidence (Stage 6 Phase D).
#: mean_confidence=0.5136 < mean_accuracy=0.5769. 0.35 prevents suppressing
#: correct predictions.
DISPLAY_THRESHOLD = 0.35

#: High-risk signs from Stage 5 Finding 8 — flagged in HUD with amber border.
HIGH_RISK_SIGNS = frozenset({"clothes", "think", "birthday", "name", "book"})

#: Stage 6 Phase E confusable pairs — displayed when top-1 prediction is from a pair.
CONFUSABLE_PAIRS: Dict[str, List[str]] = {
    "think": ["who"],      "who": ["think"],
    "later": ["house"],    "house": ["later"],
    "cousin": ["mother"],  "mother": ["cousin"],
    "girl": ["orange"],    "orange": ["girl"],
}

#: TFLite performance summary (Stage 8 gate, for HUD info panel)
_TFLITE_MEDIAN_MS   = 46.86
_FULL_PIPELINE_MS   = 47.11
_VAL_MACRO_F1       = 0.5916
_TEST_MACRO_F1      = 0.4867
_MODEL_SIZE_MB      = 0.1596
_MODEL_PARAMS       = 68_771

# =============================================================================
# HUD Design System
# =============================================================================

# BGR colours
C_WHITE     = (255, 255, 255)
C_BLACK     = (0,   0,   0)
C_DARK      = (25,  25,  35)       # near-black panel background
C_DARK2     = (40,  40,  55)       # slightly lighter panel
C_GREEN     = (50,  220, 100)      # high confidence
C_AMBER     = (30,  175, 255)      # medium confidence
C_RED       = (60,  60,  240)      # low confidence / error
C_BLUE      = (255, 140, 40)       # accent (BGR: warm blue)
C_CYAN      = (230, 210, 50)       # info accent
C_GREY      = (160, 160, 160)      # secondary text
C_LGREY     = (210, 210, 210)      # light grey
C_YELLOW    = (0,   220, 230)      # warning colour
C_PURPLE    = (200, 80,  180)      # confusable pair indicator
C_TEAL      = (180, 200, 50)       # stability indicator
C_SKELETON  = (0,   255, 128)      # landmark skeleton colour

# Fonts
F_SANS      = cv2.FONT_HERSHEY_SIMPLEX
F_DUPLEX    = cv2.FONT_HERSHEY_DUPLEX
F_COMPLEX   = cv2.FONT_HERSHEY_COMPLEX_SMALL

# Panel geometry (computed relative to frame size at runtime)
PANEL_ALPHA = 0.72          # background overlay transparency
TOP_H       = 110           # top banner height
SIDE_W      = 230           # right panel width
BOT_H       = 50            # bottom bar height

# Landmark indices for skeleton drawing (MediaPipe hand connections)
_HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),           # thumb
    (0,5),(5,6),(6,7),(7,8),           # index
    (0,9),(9,10),(10,11),(11,12),      # middle
    (0,13),(13,14),(14,15),(15,16),    # ring
    (0,17),(17,18),(18,19),(19,20),    # pinky
    (5,9),(9,13),(13,17),              # palm
]


# =============================================================================
# Drawing utilities
# =============================================================================

def _conf_color(conf: float) -> Tuple[int, int, int]:
    """Map confidence to a BGR colour (green → amber → red)."""
    if conf >= 0.70:
        return C_GREEN
    if conf >= 0.45:
        return C_AMBER
    return C_RED


def _alpha_rect(
    img: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    color: Tuple[int, int, int],
    alpha: float = PANEL_ALPHA,
    radius: int = 0,
) -> np.ndarray:
    """Draw a filled rectangle with transparency onto img in-place."""
    sub = img[y1:y2, x1:x2]
    rect = np.full_like(sub, color, dtype=np.uint8)
    blended = cv2.addWeighted(sub, 1 - alpha, rect, alpha, 0)
    img[y1:y2, x1:x2] = blended

    if radius > 0:
        # Draw rounded-corner border
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)

    return img


def _text_shadow(
    img: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    font,
    scale: float,
    color: Tuple[int, int, int],
    thickness: int = 1,
    shadow_offset: int = 2,
) -> None:
    """Draw text with a drop shadow for readability on any background."""
    cv2.putText(img, text, (pos[0] + shadow_offset, pos[1] + shadow_offset),
                font, scale, C_BLACK, thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, pos, font, scale, color, thickness, cv2.LINE_AA)


def _progress_bar(
    img: np.ndarray,
    x: int, y: int, width: int, height: int,
    fraction: float,
    fill_color: Tuple[int, int, int],
    bg_color: Tuple[int, int, int] = (60, 60, 60),
    border: bool = True,
    glow: bool = False,
) -> None:
    """Draw a filled progress bar."""
    cv2.rectangle(img, (x, y), (x + width, y + height), bg_color, -1)
    fill_w = max(0, int(width * min(fraction, 1.0)))
    if fill_w > 0:
        cv2.rectangle(img, (x, y), (x + fill_w, y + height), fill_color, -1)
        if glow and fill_w > 4:
            # Subtle glow effect on filled portion
            glow_sub = img[y:y+height, x:x+fill_w].copy()
            glow_bright = cv2.addWeighted(
                glow_sub, 0.7,
                np.full_like(glow_sub, fill_color), 0.3, 0
            )
            img[y:y+height, x:x+fill_w] = glow_bright
    if border:
        cv2.rectangle(img, (x, y), (x + width, y + height), C_GREY, 1)


def _draw_skeleton(
    img: np.ndarray,
    landmarks_225: np.ndarray,
    frame_w: int,
    frame_h: int,
    hand: str = "right",          # "left" | "right" | "both"
    color: Tuple[int, int, int] = C_SKELETON,
    point_r: int = 3,
    line_t: int = 1,
) -> None:
    """
    Project wrist-relative normalised landmarks back onto frame pixel coords
    and draw the hand skeleton. Only draws detected (non-zero) hands.

    NOTE: landmarks_225 here is the RAW (pre-normalisation) vector from the
    buffer, so coordinates are already in [0,1] camera-space.
    """
    hands_to_draw = []
    if hand in ("left", "both"):
        hands_to_draw.append(("left", 0))      # left hand: dims [0:63]
    if hand in ("right", "both"):
        hands_to_draw.append(("right", 63))    # right hand: dims [63:126]

    for side, offset in hands_to_draw:
        lm_slice = landmarks_225[offset:offset + 63].reshape(21, 3)

        # Skip zero-fill (undetected) hands
        if not np.any(lm_slice):
            continue

        # Project to pixel coords (x,y only — z is depth, not used for display)
        pts = []
        for lm in lm_slice:
            px = int(lm[0] * frame_w)
            py = int(lm[1] * frame_h)
            pts.append((px, py))

        # Draw connections
        for a, b in _HAND_CONNECTIONS:
            if 0 <= a < len(pts) and 0 <= b < len(pts):
                cv2.line(img, pts[a], pts[b], color, line_t, cv2.LINE_AA)

        # Draw joint dots
        for i, (px, py) in enumerate(pts):
            r = point_r + 1 if i == 0 else point_r   # wrist larger
            cv2.circle(img, (px, py), r, color, -1, cv2.LINE_AA)
            cv2.circle(img, (px, py), r, C_WHITE, 1, cv2.LINE_AA)


# =============================================================================
# FPS tracker
# =============================================================================

class FPSTracker:
    """Rolling FPS estimator with per-stage breakdown."""

    def __init__(self, window: int = 30):
        self._window    = window
        self._intervals: deque = deque(maxlen=window)
        self._t_last    = time.perf_counter()
        self._stage_ms: Dict[str, deque] = {
            "capture":    deque(maxlen=window),
            "mediapipe":  deque(maxlen=window),
            "pipeline":   deque(maxlen=window),
            "inference":  deque(maxlen=window),
            "hud":        deque(maxlen=window),
            "total":      deque(maxlen=window),
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
        mid = len(arr) // 2
        return arr[mid]

    @property
    def breakdown(self) -> Dict[str, float]:
        return {k: self.stage_median(k) for k in self._stage_ms}


# =============================================================================
# Prediction history tracker (for debounce / stability)
# =============================================================================

class PredictionHistory:
    """
    Track prediction stability over time.

    Implements the Stage 9 debounce requirement: don't show a new sign
    label until it has appeared in 3 consecutive prediction windows.
    Also tracks per-sign prediction streaks and session counts.
    """

    def __init__(self, debounce: int = 3):
        self._debounce       = debounce
        self._recent:  deque = deque(maxlen=10)
        self._current        = None
        self._streak         = 0
        self._displayed      = None
        self._session_counts: Dict[str, int] = {}
        self._last_displayed_at = 0.0
        self._display_history: List[Tuple[float, str, float]] = []  # (t, sign, conf)

    def update(self, result: Optional[Dict[str, Any]]) -> Optional[str]:
        """
        Update history with a new prediction result.
        Returns the sign that should be DISPLAYED (after debounce), or None.
        """
        if result is None:
            return self._displayed

        sign = result.get("sign", "")
        if not sign or not result.get("is_confident", False):
            return self._displayed

        self._recent.append(sign)

        if sign == self._current:
            self._streak += 1
        else:
            self._current = sign
            self._streak  = 1

        # Debounce: display only after N consecutive consistent predictions
        if self._streak >= self._debounce:
            if sign != self._displayed:
                self._displayed = sign
                self._session_counts[sign] = self._session_counts.get(sign, 0) + 1
                self._last_displayed_at = time.time()
                conf = result.get("confidence", 0.0)
                self._display_history.append((time.time(), sign, conf))
                if len(self._display_history) > 100:
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
    def sign_counts(self) -> Dict[str, int]:
        return dict(self._session_counts)

    def most_predicted(self, n: int = 5) -> List[Tuple[str, int]]:
        return sorted(self._session_counts.items(), key=lambda x: x[1], reverse=True)[:n]

    def reset(self) -> None:
        self._recent.clear()
        self._current    = None
        self._streak     = 0
        self._displayed  = None


# =============================================================================
# MediaPipe extractor (hands-only, with Holistic fallback)
# =============================================================================

class HandsExtractor:
    """
    MediaPipe Hands extractor — preferred over Holistic for Stage 9.

    Stage 8 note: MediaPipe Hands produces the same 21 landmarks per hand
    that the champion model was trained on. It runs in ~8-10ms vs Holistic's
    ~18ms, reducing estimated end-to-end latency from ~70ms to ~57ms.

    The preprocessing contract (from pipeline.py) is:
      [0:63]   left hand  — 21 landmarks × (x,y,z)
      [63:126] right hand — 21 landmarks × (x,y,z)
      [126:225] pose      — ALWAYS zero for hands-only champion
    Zero-fill frames ENTER the buffer (semantic, not noise).
    """

    def __init__(
        self,
        model_complexity:     int   = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence:  float = 0.5,
        static_image_mode:    bool  = False,
    ):
        try:
            import mediapipe as mp
            self._mp_hands = mp.solutions.hands
            self._mp_drawing = mp.solutions.drawing_utils
            self._mp_drawing_styles = mp.solutions.drawing_styles

            self._hands = self._mp_hands.Hands(
                static_image_mode=static_image_mode,
                max_num_hands=2,
                model_complexity=model_complexity,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self._mode = "hands"
            logger.info(
                "MediaPipe Hands initialised "
                "(complexity=%d, det_conf=%.2f, track_conf=%.2f)",
                model_complexity,
                min_detection_confidence,
                min_tracking_confidence,
            )
        except Exception as exc:
            logger.warning(
                "MediaPipe Hands init failed (%s). Falling back to Holistic.",
                exc,
            )
            self._try_holistic()

    def _try_holistic(self) -> None:
        """Holistic fallback (uses all 225 dims including pose)."""
        import mediapipe as mp
        self._mp_holistic = mp.solutions.holistic
        self._holistic = self._mp_holistic.Holistic(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._mode = "holistic"
        logger.info("MediaPipe Holistic initialised as fallback.")

    def extract(self, bgr_frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Extract landmarks from a BGR frame.

        Returns:
          landmarks_225 : (225,) float32 — raw MediaPipe output (camera-space coords).
                          Zero-filled for undetected hands/pose.
          meta          : dict with detection info for HUD display.
        """
        landmarks = np.zeros(225, dtype=np.float32)
        meta: Dict[str, Any] = {
            "left_detected":  False,
            "right_detected": False,
            "pose_detected":  False,
            "n_hands":        0,
            "raw_results":    None,
        }

        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        if self._mode == "hands":
            results = self._hands.process(rgb)
            meta["raw_results"] = results

            if results.multi_hand_landmarks and results.multi_handedness:
                meta["n_hands"] = len(results.multi_hand_landmarks)

                for hand_lm, hand_class in zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness,
                ):
                    label = hand_class.classification[0].label
                    # MediaPipe labels are from the signer's perspective;
                    # after mirror flip, "Left" in frame = signer's right hand.
                    if label == "Left":
                        # Signer's right hand → slot [63:126]
                        offset = 63
                        meta["right_detected"] = True
                    else:
                        # Signer's left hand → slot [0:63]
                        offset = 0
                        meta["left_detected"] = True

                    for i, lm in enumerate(hand_lm.landmark):
                        base = offset + i * 3
                        landmarks[base]     = lm.x
                        landmarks[base + 1] = lm.y
                        landmarks[base + 2] = lm.z

            # Pose slot [126:225] stays zero (hands-only champion)

        else:
            # Holistic fallback — uses full 225-dim layout
            results = self._holistic.process(rgb)
            meta["raw_results"] = results

            if results.left_hand_landmarks:
                meta["left_detected"] = True
                for i, lm in enumerate(results.left_hand_landmarks.landmark):
                    landmarks[i*3], landmarks[i*3+1], landmarks[i*3+2] = lm.x, lm.y, lm.z

            if results.right_hand_landmarks:
                meta["right_detected"] = True
                for i, lm in enumerate(results.right_hand_landmarks.landmark):
                    base = 63 + i * 3
                    landmarks[base], landmarks[base+1], landmarks[base+2] = lm.x, lm.y, lm.z

            if results.pose_landmarks:
                meta["pose_detected"] = True
                for i, lm in enumerate(results.pose_landmarks.landmark):
                    base = 126 + i * 3
                    landmarks[base], landmarks[base+1], landmarks[base+2] = lm.x, lm.y, lm.z

            meta["n_hands"] = int(meta["left_detected"]) + int(meta["right_detected"])

        return landmarks, meta

    def close(self) -> None:
        with contextlib.suppress(Exception):
            if self._mode == "hands":
                self._hands.close()
            else:
                self._holistic.close()


# =============================================================================
# HUD Renderer
# =============================================================================

class HUDRenderer:
    """
    Production-grade HUD renderer.

    Panels:
      ┌─────────────────────────────────────────┐
      │  TOP BANNER  — sign name + confidence   │
      ├──────────────────────┬──────────────────┤
      │                      │  RIGHT PANEL     │
      │   VIDEO FEED         │  Top-3 preds     │
      │                      │  Session stats   │
      │                      │  Model info      │
      │                      │  Stage timing    │
      ├──────────────────────┴──────────────────┤
      │  BOTTOM BAR  — FPS | status | controls  │
      └─────────────────────────────────────────┘

    Adaptive: right panel collapses in minimal_hud mode.
    """

    def __init__(self, minimal: bool = False):
        self.minimal    = minimal
        self._frame_n   = 0
        self._pulse     = 0.0       # animation phase [0, 2π]

    def render(
        self,
        frame:          np.ndarray,
        result:         Optional[Dict[str, Any]],
        meta:           Dict[str, Any],
        history:        PredictionHistory,
        fps_tracker:    FPSTracker,
        predictor:      GesturePredictor,
        display_threshold: float,
        show_skeleton:  bool = True,
        is_paused:      bool = False,
        frozen_result:  Optional[Dict[str, Any]] = None,
        smoother_window: int = 5,
    ) -> np.ndarray:
        """Composite all HUD elements onto frame and return."""
        self._frame_n += 1
        self._pulse = (self._pulse + 0.08) % (2 * math.pi)

        h, w = frame.shape[:2]

        # Use frozen result for display if set
        display_result = frozen_result if frozen_result is not None else result

        # ── Skeleton overlay (behind panels) ──────────────────────────────
        if show_skeleton and meta.get("n_hands", 0) > 0:
            latest_lm = meta.get("_latest_raw_landmarks")
            if latest_lm is not None:
                _draw_skeleton(frame, latest_lm, w, h, hand="both",
                               color=C_SKELETON, point_r=3, line_t=1)

        # ── Panel backgrounds ──────────────────────────────────────────────
        _alpha_rect(frame, 0, 0, w, TOP_H, C_DARK, alpha=0.80)
        if not self.minimal:
            _alpha_rect(frame, w - SIDE_W, TOP_H, w, h - BOT_H, C_DARK2, alpha=0.75)
        _alpha_rect(frame, 0, h - BOT_H, w, h, C_DARK, alpha=0.80)

        # Thin separator lines
        cv2.line(frame, (0, TOP_H - 1),    (w, TOP_H - 1),    C_BLUE, 1)
        cv2.line(frame, (0, h - BOT_H),    (w, h - BOT_H),    C_BLUE, 1)
        if not self.minimal:
            cv2.line(frame, (w - SIDE_W, TOP_H), (w - SIDE_W, h - BOT_H), C_BLUE, 1)

        # ── TOP BANNER ─────────────────────────────────────────────────────
        self._draw_top_banner(frame, w, display_result, history, display_threshold,
                              is_paused, frozen_result is not None)

        # ── BUFFER FILL / NO-HANDS OVERLAY ─────────────────────────────────
        frames_buffered = predictor.frames_buffered
        seq_len         = predictor.sequence_length
        if frames_buffered < seq_len and result is None and not is_paused:
            self._draw_buffer_progress(frame, w, h, frames_buffered, seq_len)
        elif meta.get("n_hands", 0) == 0 and not is_paused:
            self._draw_no_hands_warning(frame, w, h)

        # ── RIGHT PANEL ────────────────────────────────────────────────────
        if not self.minimal:
            self._draw_right_panel(
                frame, w, h, display_result, history, fps_tracker,
                predictor, smoother_window, display_threshold
            )

        # ── BOTTOM BAR ─────────────────────────────────────────────────────
        self._draw_bottom_bar(frame, w, h, fps_tracker, meta, is_paused, frozen_result is not None)

        # ── Pause overlay ──────────────────────────────────────────────────
        if is_paused:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
            _text_shadow(frame, "⏸  PAUSED",
                         (w // 2 - 80, h // 2),
                         F_DUPLEX, 1.2, C_AMBER, 2)
            _text_shadow(frame, "Press SPACE to resume",
                         (w // 2 - 120, h // 2 + 40),
                         F_SANS, 0.55, C_LGREY, 1)

        return frame

    # ── Top banner ───────────────────────────────────────────────────────────

    def _draw_top_banner(
        self,
        frame: np.ndarray, w: int,
        result: Optional[Dict[str, Any]],
        history: PredictionHistory,
        display_threshold: float,
        is_paused: bool,
        is_frozen: bool,
    ) -> None:
        if result is None or not result.get("is_confident", False):
            # Dim waiting state
            _text_shadow(frame, "WLASL Gesture Recognition",
                         (w // 2 - 150, 40), F_DUPLEX, 0.85, C_GREY, 1)
            _text_shadow(frame, "Waiting for confident prediction...",
                         (w // 2 - 155, 75), F_SANS, 0.5, C_GREY, 1)
            return

        sign       = result.get("sign", "")
        confidence = result.get("confidence", 0.0)
        is_hr      = sign in HIGH_RISK_SIGNS
        is_conf_pair = sign in CONFUSABLE_PAIRS
        col        = _conf_color(confidence)
        stable     = history.is_stable

        # Animated pulse for stable predictions
        pulse_alpha = 0.85 + 0.15 * math.sin(self._pulse)

        # Sign name — large, centred, with glow effect
        sign_upper = sign.upper()
        (sw, sh), _ = cv2.getTextSize(sign_upper, F_DUPLEX, 2.0, 2)
        sx = max(10, w // 2 - sw // 2)

        # Glow / background highlight behind sign name
        if stable:
            glow_col = tuple(int(c * pulse_alpha) for c in col)
            _alpha_rect(frame, sx - 10, 5, sx + sw + 10, 60,
                        tuple(int(c * 0.35) for c in col), alpha=0.6)
        else:
            glow_col = col

        # Sign text with shadow
        cv2.putText(frame, sign_upper, (sx + 2, 53), F_DUPLEX, 2.0, C_BLACK, 4, cv2.LINE_AA)
        cv2.putText(frame, sign_upper, (sx, 51), F_DUPLEX, 2.0, glow_col, 2, cv2.LINE_AA)

        # Confidence bar
        bar_x, bar_y = sx, 65
        bar_w, bar_h = min(sw + 10, w // 2), 10
        _progress_bar(frame, bar_x, bar_y, bar_w, bar_h,
                      confidence, col, glow=stable)

        # Confidence percentage
        conf_str = f"{confidence:.0%}"
        _text_shadow(frame, conf_str, (bar_x + bar_w + 8, bar_y + 9),
                     F_SANS, 0.48, col, 1)

        # Threshold indicator on bar
        thresh_x = bar_x + int(bar_w * display_threshold)
        cv2.line(frame, (thresh_x, bar_y - 3), (thresh_x, bar_y + bar_h + 3),
                 C_YELLOW, 1)

        # Stability indicator dot (right of sign)
        dot_x = sx + sw + 20
        dot_col = C_GREEN if stable else C_AMBER
        dot_r   = 7
        cv2.circle(frame, (dot_x, 30), dot_r, dot_col, -1)
        cv2.circle(frame, (dot_x, 30), dot_r, C_WHITE, 1)

        # Risk / confusable badges
        badge_x = sx
        if is_hr:
            cv2.rectangle(frame, (badge_x, 82), (badge_x + 120, 102), C_AMBER, -1)
            _text_shadow(frame, "⚠ HIGH-RISK CLASS", (badge_x + 4, 97),
                         F_SANS, 0.38, C_BLACK, 1, shadow_offset=0)
            badge_x += 128

        if is_conf_pair:
            partners = CONFUSABLE_PAIRS.get(sign, [])
            if partners:
                label = f"≈ {partners[0].upper()}"
                cv2.rectangle(frame, (badge_x, 82), (badge_x + len(label) * 10 + 10, 102),
                               C_PURPLE, -1)
                _text_shadow(frame, label, (badge_x + 4, 97),
                             F_SANS, 0.38, C_WHITE, 1, shadow_offset=0)

        if is_frozen:
            _text_shadow(frame, "❄ FROZEN", (w - 100, 25), F_SANS, 0.5, C_CYAN, 1)

    # ── Right panel ──────────────────────────────────────────────────────────

    def _draw_right_panel(
        self,
        frame: np.ndarray,
        w: int, h: int,
        result: Optional[Dict[str, Any]],
        history: PredictionHistory,
        fps_tracker: FPSTracker,
        predictor: GesturePredictor,
        smoother_window: int,
        display_threshold: float,
    ) -> None:
        px = w - SIDE_W + 8    # panel content x
        py = TOP_H + 12        # panel content y start

        # ── Section: Top-3 predictions ────────────────────────────────────
        _text_shadow(frame, "TOP PREDICTIONS", (px, py + 10),
                     F_SANS, 0.42, C_CYAN, 1)
        py += 20
        cv2.line(frame, (px - 4, py), (w - 8, py), C_BLUE, 1)
        py += 8

        top_k = result.get("top_k", []) if result else []
        for i, entry in enumerate(top_k[:3]):
            sign_name = entry.get("sign", "?")[:12]
            e_conf    = entry.get("confidence", 0.0)
            e_col     = _conf_color(e_conf)
            bar_w_    = SIDE_W - 26
            entry_y   = py + i * 44

            # Rank indicator
            rank_col = [C_GREEN, C_AMBER, C_GREY][i]
            cv2.rectangle(frame, (px - 4, entry_y - 2),
                          (px + 22, entry_y + 14), rank_col, -1)
            _text_shadow(frame, f"#{i+1}", (px + 2, entry_y + 11),
                         F_SANS, 0.38, C_BLACK, 1, shadow_offset=0)

            # Sign name
            _text_shadow(frame, sign_name.upper(), (px + 28, entry_y + 12),
                         F_SANS, 0.45, C_WHITE if i == 0 else C_LGREY, 1)

            # Confidence bar
            _progress_bar(frame, px + 2, entry_y + 18, bar_w_, 8,
                          e_conf, e_col, glow=(i == 0))

            # Confidence pct
            _text_shadow(frame, f"{e_conf:.0%}", (px + bar_w_ + 6, entry_y + 26),
                         F_SANS, 0.35, e_col, 1)

            # High-risk indicator
            if sign_name.lower() in HIGH_RISK_SIGNS:
                cv2.putText(frame, "⚠", (px + bar_w_ + 40, entry_y + 12),
                            F_SANS, 0.35, C_AMBER, 1)

            py_next = entry_y + 44

        if not top_k:
            _text_shadow(frame, "—  No prediction", (px + 4, py + 16),
                         F_SANS, 0.42, C_GREY, 1)
            py_next = py + 50
        else:
            py = py_next

        py = py_next + 8
        cv2.line(frame, (px - 4, py), (w - 8, py), C_BLUE, 1)
        py += 10

        # ── Section: Session stats ─────────────────────────────────────────
        _text_shadow(frame, "SESSION", (px, py + 8), F_SANS, 0.38, C_CYAN, 1)
        py += 20
        _text_shadow(frame, f"Signs predicted  {history.session_count:>5}",
                     (px, py), F_SANS, 0.38, C_LGREY, 1)
        py += 18

        # Most-predicted this session
        top_signs = history.most_predicted(3)
        if top_signs:
            _text_shadow(frame, "Most frequent:", (px, py), F_SANS, 0.35, C_GREY, 1)
            py += 15
            for sign_name, cnt in top_signs:
                bar_pct = cnt / max(1, history.session_count)
                _progress_bar(frame, px, py, SIDE_W - 26, 6, bar_pct, C_BLUE)
                _text_shadow(frame, f"{sign_name[:8]:<8} {cnt}",
                             (px, py + 16), F_SANS, 0.32, C_LGREY, 1)
                py += 22
        py += 4
        cv2.line(frame, (px - 4, py), (w - 8, py), C_BLUE, 1)
        py += 10

        # ── Section: Model info ────────────────────────────────────────────
        _text_shadow(frame, "MODEL", (px, py + 8), F_SANS, 0.38, C_CYAN, 1)
        py += 20

        info_lines = [
            (f"BiLSTM {_MODEL_PARAMS//1000}K params", C_LGREY),
            (f"val F1:  {_VAL_MACRO_F1:.4f}",         C_GREEN),
            (f"test F1: {_TEST_MACRO_F1:.4f}",         C_AMBER),
            (f"Size:    {_MODEL_SIZE_MB:.4f} MB",      C_LGREY),
            (f"Thresh:  {display_threshold:.2f}",      C_YELLOW),
            (f"Smoother: {smoother_window}fr",         C_LGREY),
        ]
        for line, col in info_lines:
            if py > h - BOT_H - 20:
                break
            _text_shadow(frame, line, (px, py), F_SANS, 0.34, col, 1)
            py += 15

        py += 4
        if py < h - BOT_H - 50:
            cv2.line(frame, (px - 4, py), (w - 8, py), C_BLUE, 1)
            py += 10

        # ── Section: Latency breakdown ────────────────────────────────────
        if py < h - BOT_H - 80:
            _text_shadow(frame, "PIPELINE (ms)", (px, py + 8), F_SANS, 0.38, C_CYAN, 1)
            py += 20
            breakdown = fps_tracker.breakdown
            latency_items = [
                ("MediaPipe", breakdown.get("mediapipe", 0)),
                ("Pipeline",  breakdown.get("pipeline",  0)),
                ("Inference", breakdown.get("inference", 0)),
                ("HUD",       breakdown.get("hud",       0)),
            ]
            max_ms = max(v for _, v in latency_items) or 1.0
            for label, ms in latency_items:
                if py > h - BOT_H - 20:
                    break
                _progress_bar(frame, px, py, SIDE_W - 60, 6,
                              ms / 100.0, C_TEAL)   # 100ms scale
                _text_shadow(frame, f"{label:<10} {ms:>4.0f}",
                             (px, py + 16), F_SANS, 0.32, C_LGREY, 1)
                py += 22

    # ── Buffer progress ───────────────────────────────────────────────────────

    def _draw_buffer_progress(
        self,
        frame: np.ndarray, w: int, h: int,
        frames_buffered: int, seq_len: int,
    ) -> None:
        pct = frames_buffered / seq_len
        cx, cy = w // 2, h // 2

        # Background pill
        _alpha_rect(frame, cx - 220, cy - 40, cx + 220, cy + 40, C_DARK, alpha=0.85)
        cv2.rectangle(frame, (cx - 220, cy - 40), (cx + 220, cy + 40), C_BLUE, 1)

        # Circular progress ring
        ring_r  = 28
        angle   = int(360 * pct)
        cv2.circle(frame, (cx - 170, cy), ring_r, (60, 60, 60), 3)
        if angle > 0:
            cv2.ellipse(frame, (cx - 170, cy), (ring_r, ring_r),
                        -90, 0, angle, C_GREEN, 3)
        _text_shadow(frame, f"{int(pct*100)}%",
                     (cx - 170 - 14, cy + 6), F_SANS, 0.45, C_WHITE, 1)

        # Text
        _text_shadow(frame, "Building sequence buffer...",
                     (cx - 110, cy - 10), F_SANS, 0.55, C_WHITE, 1)
        _text_shadow(frame, f"{frames_buffered} / {seq_len} frames",
                     (cx - 110, cy + 16), F_SANS, 0.45, C_GREY, 1)

        # Linear progress bar
        bar_y = cy + 28
        _progress_bar(frame, cx - 110, bar_y, 200, 6, pct, C_GREEN)

    # ── No-hands warning ──────────────────────────────────────────────────────

    def _draw_no_hands_warning(
        self, frame: np.ndarray, w: int, h: int
    ) -> None:
        # Pulsing amber dot
        pulse = 0.5 + 0.5 * math.sin(self._pulse * 2)
        dot_r = int(8 + 4 * pulse)
        alpha_warn = 0.4 + 0.3 * pulse

        warn_x, warn_y = w // 2 - 120, h - BOT_H - 30
        _alpha_rect(frame, warn_x - 10, warn_y - 20,
                    warn_x + 250, warn_y + 10, C_DARK, alpha=alpha_warn)
        cv2.circle(frame, (warn_x - 2, warn_y - 6), dot_r, C_AMBER, -1)
        _text_shadow(frame, "No hands detected",
                     (warn_x + 14, warn_y), F_SANS, 0.52, C_AMBER, 1)

    # ── Bottom bar ────────────────────────────────────────────────────────────

    def _draw_bottom_bar(
        self,
        frame: np.ndarray, w: int, h: int,
        fps_tracker: FPSTracker,
        meta: Dict[str, Any],
        is_paused: bool,
        is_frozen: bool,
    ) -> None:
        by = h - BOT_H + 8

        # FPS display (left)
        fps = fps_tracker.fps
        fps_col = C_GREEN if fps >= 15 else C_AMBER if fps >= 8 else C_RED
        _text_shadow(frame, f"FPS {fps:>5.1f}", (10, by + 16), F_DUPLEX, 0.55, fps_col, 1)

        # FPS bar
        _progress_bar(frame, 10, by + 26, 80, 5, min(fps / 30.0, 1.0), fps_col)

        # Hand detection indicators (centre-left)
        lh_col = C_GREEN if meta.get("left_detected")  else (60, 60, 60)
        rh_col = C_GREEN if meta.get("right_detected") else (60, 60, 60)
        cv2.circle(frame, (130, by + 12), 6, lh_col, -1)
        _text_shadow(frame, "L", (127, by + 17), F_SANS, 0.35, C_WHITE, 1)
        cv2.circle(frame, (150, by + 12), 6, rh_col, -1)
        _text_shadow(frame, "R", (147, by + 17), F_SANS, 0.35, C_WHITE, 1)
        _text_shadow(frame, "Hands", (110, by + 32), F_SANS, 0.30, C_GREY, 1)

        # Controls hint (centre)
        hints = "q:quit  r:reset  s:screenshot  h:HUD  m:skeleton  SPACE:freeze  ±:threshold"
        hint_x = max(180, w // 2 - len(hints) * 3)
        _text_shadow(frame, hints, (hint_x, by + 36), F_SANS, 0.30, C_GREY, 1)

        # Timestamp (right)
        ts = datetime.now().strftime("%H:%M:%S")
        _text_shadow(frame, ts, (w - 85, by + 18), F_SANS, 0.48, C_LGREY, 1)

        # Status badge
        if is_frozen:
            cv2.rectangle(frame, (w - 85, by + 26), (w - 8, by + 42), C_CYAN, -1)
            _text_shadow(frame, "FROZEN", (w - 80, by + 39), F_SANS, 0.35, C_BLACK, 1, shadow_offset=0)
        elif is_paused:
            cv2.rectangle(frame, (w - 85, by + 26), (w - 8, by + 42), C_AMBER, -1)
            _text_shadow(frame, "PAUSED", (w - 80, by + 39), F_SANS, 0.35, C_BLACK, 1, shadow_offset=0)

        # Branding watermark (subtle, bottom-right corner)
        _text_shadow(frame, "WLASL-35 | BiLSTM 68K | Henry Otsyula",
                     (w - 310, h - 4), F_SANS, 0.28, (80, 80, 80), 1)


# =============================================================================
# Main demo loop
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="webcam_demo.py",
        description="Stage 9 — Production Webcam Demo: WLASL 35-Sign Gesture Recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    g_paths = p.add_argument_group("paths")
    g_paths.add_argument("--model",        default=_DEFAULT_TFLITE_PATH,
                         help="Path to .tflite model (default: champion TFLite)")
    g_paths.add_argument("--config",       default=_DEFAULT_CONFIG_SNAPSHOT,
                         help="Path to config_snapshot.yaml")
    g_paths.add_argument("--label-map",    default=_DEFAULT_LABEL_MAP)
    g_paths.add_argument("--calib-report", default=_DEFAULT_CALIB_REPORT)

    g_cam = p.add_argument_group("camera")
    g_cam.add_argument("--camera",  type=int, default=0, help="Camera device index")
    g_cam.add_argument("--width",   type=int, default=1280)
    g_cam.add_argument("--height",  type=int, default=720)
    g_cam.add_argument("--no-flip", action="store_true",
                       help="Disable mirror flip (default: enabled for natural feel)")

    g_model = p.add_argument_group("model")
    g_model.add_argument("--threshold",    type=float, default=DISPLAY_THRESHOLD,
                         help=f"Display confidence threshold (default: {DISPLAY_THRESHOLD})")
    g_model.add_argument("--smoother",     type=int,   default=5,
                         help="Majority-vote window (1=off, default: 5)")
    g_model.add_argument("--complexity",   type=int,   default=1, choices=[0, 1, 2],
                         help="MediaPipe model complexity (default: 1)")
    g_model.add_argument("--det-conf",     type=float, default=0.5,
                         help="MediaPipe min_detection_confidence (default: 0.5)")
    g_model.add_argument("--track-conf",   type=float, default=0.5,
                         help="MediaPipe min_tracking_confidence (default: 0.5)")
    g_model.add_argument("--auto-reset",   type=int,   default=3,
                         help="Frames before auto-reset on no-detection (default: 3)")

    g_ui = p.add_argument_group("UI")
    g_ui.add_argument("--minimal-hud",  action="store_true",
                      help="Minimal HUD (no right panel) for performance")
    g_ui.add_argument("--no-skeleton",  action="store_true",
                      help="Disable landmark skeleton overlay")
    g_ui.add_argument("--debounce",     type=int, default=3,
                      help="Consecutive predictions before display update (default: 3)")

    g_io = p.add_argument_group("I/O")
    g_io.add_argument("--record",    default=None,
                      help="Record output to video file (e.g. outputs/demo.mp4)")
    g_io.add_argument("--warmup",    type=int, default=3,
                      help="Warmup inference passes before main loop (default: 3)")

    return p


def _print_startup_banner(args: argparse.Namespace) -> None:
    """Print a rich terminal banner before opening the window."""
    SEP = "─" * 64
    print(f"\n{'═'*64}")
    print("  WLASL 35-Sign Gesture Recognition — Stage 9 Demo")
    print("  Senior ML Engineer: Henry Otsyula")
    print(f"{'═'*64}")
    print(f"  {SEP}")
    print(f"  Model      : {Path(args.model).name}")
    print(f"  Config     : {Path(args.config).name}")
    print(f"  Camera     : device {args.camera}  ({args.width}×{args.height})")
    print(f"  Threshold  : {args.threshold:.2f}  (calibrated for underconfident model)")
    print(f"  Smoother   : {args.smoother}-frame majority vote")
    print(f"  Mirror     : {'OFF' if args.no_flip else 'ON (natural hand view)'}")
    print(f"  {SEP}")
    print(f"  TFLite size  : {_MODEL_SIZE_MB:.4f} MB  |  Params : {_MODEL_PARAMS:,}")
    print(f"  Val macro-F1 : {_VAL_MACRO_F1:.4f}   |  Test F1 : {_TEST_MACRO_F1:.4f}")
    print(f"  Pipeline     : ~{_FULL_PIPELINE_MS:.0f} ms  (excl. MediaPipe)")
    print(f"  {SEP}")
    print("  Controls:")
    print("    q / ESC  — quit            r       — reset buffer + smoother")
    print("    s        — screenshot      h       — toggle HUD")
    print("    m        — skeleton        SPACE   — freeze prediction")
    print("    + / -    — adjust threshold  1-5   — smoother window")
    print(f"{'═'*64}\n")


def main() -> int:
    parser = build_arg_parser()
    args   = parser.parse_args()

    _print_startup_banner(args)

    # ── Load GesturePredictor ──────────────────────────────────────────────────
    print("  [1/4] Loading GesturePredictor...", end="", flush=True)
    try:
        calib_path = args.calib_report if Path(args.calib_report).exists() else None
        if calib_path is None:
            print(f"\n  ⚠  Calibration report not found at {args.calib_report}.")
            print(f"     Using hardcoded display threshold = {args.threshold:.2f}")

        predictor = GesturePredictor.from_config_snapshot(
            config_snapshot_path=args.config,
            model_path=args.model,
            label_map_path=args.label_map,
            smoother_window=args.smoother,
            display_threshold=args.threshold,
            calibration_report_path=calib_path,
            auto_reset_no_detection_frames=args.auto_reset,
            n_top_k=3,
            flag_high_risk_classes=True,
        )
        print(f" ✓  ({predictor.model_type}, {predictor.sequence_length} frames, "
              f"{predictor.feature_dim} dims, threshold={predictor.display_threshold:.2f})")
    except FileNotFoundError as e:
        print(f"\n  ✗  {e}", file=sys.stderr)
        print("     Run: python pipelines/run_export_verification.py", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\n  ✗  GesturePredictor failed: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1

    # ── Warmup ────────────────────────────────────────────────────────────────
    print(f"  [2/4] Warming up ({args.warmup} passes)...", end="", flush=True)
    t_wu = time.perf_counter()
    predictor.warmup(n_passes=args.warmup)
    print(f" ✓  ({(time.perf_counter() - t_wu)*1000:.0f} ms)")

    # ── MediaPipe extractor ────────────────────────────────────────────────────
    print("  [3/4] Initialising MediaPipe...", end="", flush=True)
    try:
        extractor = HandsExtractor(
            model_complexity=args.complexity,
            min_detection_confidence=args.det_conf,
            min_tracking_confidence=args.track_conf,
        )
        print(f" ✓  (mode={extractor._mode})")
    except Exception as e:
        print(f"\n  ✗  MediaPipe init failed: {e}", file=sys.stderr)
        predictor.close()
        return 1

    # ── Camera ────────────────────────────────────────────────────────────────
    print(f"  [4/4] Opening camera {args.camera}...", end="", flush=True)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"\n  ✗  Camera {args.camera} not available.", file=sys.stderr)
        extractor.close()
        predictor.close()
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # minimise capture latency

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f" ✓  ({actual_w}×{actual_h})")

    # ── Optional video recorder ────────────────────────────────────────────────
    writer: Optional[cv2.VideoWriter] = None
    if args.record:
        rec_path = Path(args.record)
        rec_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(rec_path), fourcc, 30.0, (actual_w, actual_h))
        print(f"  Recording → {rec_path}")

    _SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    # ── State variables ────────────────────────────────────────────────────────
    fps_tracker    = FPSTracker(window=30)
    hud            = HUDRenderer(minimal=args.minimal_hud)
    pred_history   = PredictionHistory(debounce=args.debounce)
    show_hud       = True
    show_skeleton  = not args.no_skeleton
    is_paused      = False
    frozen_result: Optional[Dict[str, Any]] = None
    frame_count    = 0
    latest_result: Optional[Dict[str, Any]] = None
    latest_meta:   Dict[str, Any] = {"n_hands": 0}
    current_threshold = predictor.display_threshold
    current_smoother  = args.smoother

    WINDOW_NAME = "WLASL Gesture Recognition  —  Henry Otsyula"

    print(f"\n  ▶  Demo running. Press q to quit.\n")

    # ── Main loop ──────────────────────────────────────────────────────────────
    with predictor:
        while True:
            # ── Frame capture ────────────────────────────────────────────────
            t_cap = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                logger.warning("Frame capture failed — retrying")
                time.sleep(0.01)
                continue

            if not args.no_flip:
                frame = cv2.flip(frame, 1)

            fps_tracker.record("capture", (time.perf_counter() - t_cap) * 1000)
            frame_count += 1

            # ── MediaPipe extraction ─────────────────────────────────────────
            if not is_paused:
                t_mp = time.perf_counter()
                landmarks_225, meta = extractor.extract(frame)
                fps_tracker.record("mediapipe", (time.perf_counter() - t_mp) * 1000)

                # Stash raw landmarks for skeleton drawing
                meta["_latest_raw_landmarks"] = landmarks_225.copy()
                latest_meta = meta

                # ── GesturePredictor ─────────────────────────────────────────
                # We bypass predict_from_webcam_frame's internal MediaPipe
                # since we've already extracted landmarks above (avoids double-extraction).
                # Instead, we manually feed the buffer.
                t_pred = time.perf_counter()

                # Ensure predictor's frame buffer is updated manually
                predictor._frame_buffer.add_frame(landmarks_225)

                # Auto-reset: check no-detection streak
                if not np.any(landmarks_225):
                    predictor._no_detection_streak += 1
                else:
                    predictor._no_detection_streak = 0

                if (predictor._auto_reset_threshold is not None and
                        predictor._no_detection_streak >= predictor._auto_reset_threshold):
                    predictor.reset()
                    pred_history.reset()
                    latest_result = None
                    fps_tracker.record("pipeline", 0)
                    fps_tracker.record("inference", 0)
                else:
                    # Pipeline + inference if buffer is ready
                    if predictor._frame_buffer.is_ready():
                        raw_seq = predictor._frame_buffer.get_array()

                        # FeaturePipeline (wrist norm + z-clip + pad)
                        t_pipe = time.perf_counter()
                        features_2d = predictor._pipeline(raw_seq, training=False)
                        fps_tracker.record("pipeline", (time.perf_counter() - t_pipe) * 1000)

                        features_b = features_2d[np.newaxis, ...].astype(np.float32)

                        # TFLite inference
                        t_inf = time.perf_counter()
                        raw_probs, elapsed_ms = predictor._run_single(features_b)
                        fps_tracker.record("inference", elapsed_ms)

                        # Smoother update
                        predicted_class, smoothed_probs, is_stable = predictor._smoother.update(raw_probs)
                        display_confidence = float(smoothed_probs[predicted_class])
                        raw_confidence     = float(raw_probs.max())

                        latest_result = predictor._build_result(
                            predicted_class=predicted_class,
                            smoothed_probs=smoothed_probs,
                            display_confidence=display_confidence,
                            raw_confidence=raw_confidence,
                            raw_class_idx=int(np.argmax(raw_probs)),
                            is_stable=is_stable,
                            n_frames_input=predictor.sequence_length,
                            inference_latency_ms=elapsed_ms,
                        )
                    else:
                        latest_result = None

                # Update prediction history (debounce)
                pred_history.update(latest_result)

            # ── HUD rendering ────────────────────────────────────────────────
            t_hud = time.perf_counter()
            if show_hud:
                frame = hud.render(
                    frame       = frame,
                    result      = latest_result,
                    meta        = latest_meta,
                    history     = pred_history,
                    fps_tracker = fps_tracker,
                    predictor   = predictor,
                    display_threshold = current_threshold,
                    show_skeleton = show_skeleton,
                    is_paused   = is_paused,
                    frozen_result = frozen_result,
                    smoother_window = current_smoother,
                )
            fps_tracker.record("hud", (time.perf_counter() - t_hud) * 1000)

            # ── Display ──────────────────────────────────────────────────────
            cv2.imshow(WINDOW_NAME, frame)

            if writer is not None:
                writer.write(frame)

            fps_tracker.tick()

            # ── Periodic console log ─────────────────────────────────────────
            if frame_count % 150 == 0:
                logger.info(
                    "FPS=%.1f | buffered=%d/%d | no_detect_streak=%d | "
                    "session_signs=%d | mp_median=%.0fms | inf_median=%.0fms",
                    fps_tracker.fps,
                    predictor.frames_buffered,
                    predictor.sequence_length,
                    predictor._no_detection_streak,
                    pred_history.session_count,
                    fps_tracker.stage_median("mediapipe"),
                    fps_tracker.stage_median("inference"),
                )

            # ── Key handling ─────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):               # q or ESC — quit
                break

            elif key == ord("r"):                   # Hard reset
                predictor.reset()
                pred_history.reset()
                latest_result = None
                frozen_result = None
                print(f"  ↺  Reset  (frame {frame_count})")

            elif key == ord("s"):                   # Screenshot
                ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                sign_name = (latest_result or {}).get("sign", "none")
                shot = _SCREENSHOT_DIR / f"wlasl_{sign_name}_{ts_str}.png"
                cv2.imwrite(str(shot), frame)
                print(f"  📸 Screenshot → {shot}")

            elif key == ord("h"):                   # Toggle HUD
                show_hud = not show_hud
                print(f"  HUD {'ON' if show_hud else 'OFF'}")

            elif key == ord("m"):                   # Toggle skeleton
                show_skeleton = not show_skeleton
                print(f"  Skeleton {'ON' if show_skeleton else 'OFF'}")

            elif key == ord(" "):                   # Freeze / unfreeze
                if frozen_result is not None:
                    frozen_result = None
                    print("  ❄  Prediction unfrozen")
                elif latest_result is not None:
                    frozen_result = dict(latest_result)
                    print(f"  ❄  Frozen: {frozen_result.get('sign')} "
                          f"({frozen_result.get('confidence', 0):.0%})")

            elif key == ord("p"):                   # Pause
                is_paused = not is_paused
                print(f"  {'⏸ Paused' if is_paused else '▶ Resumed'}")

            elif key in (ord("+"), ord("=")):       # Raise threshold
                current_threshold = min(0.99, current_threshold + 0.05)
                predictor._display_threshold = current_threshold
                print(f"  Threshold ↑ {current_threshold:.2f}")

            elif key in (ord("-"), ord("_")):       # Lower threshold
                current_threshold = max(0.05, current_threshold - 0.05)
                predictor._display_threshold = current_threshold
                print(f"  Threshold ↓ {current_threshold:.2f}")

            elif key in (ord("1"), ord("2"), ord("3"), ord("4"), ord("5")):
                w_val = int(chr(key))
                current_smoother = w_val
                predictor._smoother._window = w_val
                print(f"  Smoother window → {w_val}")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    if writer is not None:
        writer.release()
        print(f"\n  Recording saved → {args.record}")
    cv2.destroyAllWindows()
    extractor.close()

    # ── Session summary ───────────────────────────────────────────────────────
    avg_fps = fps_tracker.fps
    print(f"\n{'═'*64}")
    print("  SESSION SUMMARY")
    print(f"  {'─'*60}")
    print(f"  Frames processed : {frame_count:,}")
    print(f"  Average FPS      : {avg_fps:.1f}")
    print(f"  Signs predicted  : {pred_history.session_count}")
    if pred_history.most_predicted():
        print(f"  Top predictions  :")
        for sign_name, cnt in pred_history.most_predicted(5):
            bar = "█" * min(cnt, 20)
            print(f"    {sign_name:<14} {cnt:>4}  {bar}")
    breakdown = fps_tracker.breakdown
    print(f"  {'─'*60}")
    print(f"  Pipeline latency (median):")
    for stage, ms in breakdown.items():
        if ms > 0:
            print(f"    {stage:<12} {ms:>6.1f} ms")
    print(f"{'═'*64}\n")

    return 0


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    sys.exit(main())