"""
src/inference/predictor.py
============================
Stage 7 — Unified Inference Engine for the WLASL 35-class gesture
recognition system.

This module is the SOLE entry point through which any consumer — Stage 8's
TFLite verification, Stage 9's real-time webcam demo, a future Android
wrapper, or an ad-hoc notebook cell — runs the champion model. Centralising
inference here is what turns this project from a collection of training
scripts into a deployable system: every preprocessing parameter (sequence
length, landmark config, z-clip, wrist normalisation) is read from exactly
one ``FeaturePipeline`` instance, so a future change to any of those values
cannot accidentally be applied in one consumer and missed in another.

Why this file looks the way it does (grounded in what Stage 6 actually found)
--------------------------------------------------------------------------------
This module was written AFTER Stage 6 (evaluation, benchmarking, calibration,
interpretability) completed, and several Stage 6 findings are load-bearing
design decisions here, not cosmetic defaults:

  1. The champion is UNDERCONFIDENT, not overconfident.
     Stage 6 calibration (``src/evaluation/calibration.py``) on the real
     val-set softmax outputs found mean confidence 0.5136 against mean
     accuracy 0.5769 for the champion (``bilstm_hands_only_v4_aug``) — the
     model's own confidence systematically UNDERSTATES how often it is
     right. A naive ``confidence >= 0.50`` display gate would therefore
     suppress a meaningful fraction of CORRECT predictions. This module's
     ``DEFAULT_DISPLAY_THRESHOLD = 0.35`` encodes that finding directly;
     see ``GesturePredictor.__init__`` for how a caller can instead supply
     a calibration-report-derived threshold (Stage 6's
     ``threshold_curve.optimal_threshold_accuracy``) when one is available
     on disk, rather than relying on the hardcoded fallback forever.

  2. The champion's real hyperparameters come from
     ``artifacts/experiments/bilstm_hands_only_v4_aug/config_snapshot.yaml``,
     not from the earlier narrative draft of the Stage 7 spec. That draft
     stated ``hidden_units=128`` (64/direction) for the "champion override".
     The verified, on-disk config snapshot shows ``hidden_units: 64``
     (32/direction, concat=64) — consistent with the documented parameter
     count of 68,771 and the ablation-default ``model/bilstm.yaml``. This
     module therefore never hardcodes champion hyperparameters; it always
     takes the *actual* ``ExperimentConfig`` the model was trained with —
     see ``GesturePredictor.from_config_snapshot()``, which loads that
     exact YAML rather than re-deriving it from CLI-style overrides that
     are easy to get subtly wrong (e.g. the champion's
     ``data.landmark_config: hands_only`` is NOT set by any of
     ``configs/data/seq*.yaml`` — it was applied as a run-time override
     during Stage 5 and is only durably recorded in the snapshot).

  3. ``early_stopping_monitor`` in the champion's config is ``val_accuracy``,
     not ``val_macro_f1`` as the Stage 5 handoff narrative claimed. This
     module takes no position on that discrepancy (already flagged in
     ``benchmark.py`` / ``calibration.py``) — it is mentioned here only so
     a reader of this file isn't confused by the apparent mismatch when
     cross-referencing the handoff document.

  4. TFLite export (Stage 8) has NOT happened yet — only the Keras
     SavedModel at ``models/bilstm_hands_only_v4_aug_saved_model/`` exists
     on disk today. ``GesturePredictor`` auto-detects the model format from
     the path (``*.tflite`` vs. a SavedModel directory) so this exact same
     class serves both the current pre-Stage-8 world (Keras only) and the
     post-Stage-8 world (TFLite primary, Keras as an accuracy-comparison
     fallback for ``src/export/verify.py``) without any code change here.

  5. ``HIGH_RISK_SIGNS`` from Stage 6 (``clothes``, ``think``, ``birthday``,
     ``name``, ``book`` — Stage 5 Finding 8 / Stage 6 per-class analysis)
     are surfaced on every prediction (``is_high_risk_class``) so any
     downstream HUD or report can flag low-trust predictions instead of
     presenting every class with equal implied confidence.

Evaluation-framework compatibility (a deliberate, senior-level design choice)
---------------------------------------------------------------------------------
``src/evaluation/metrics.py`` (Stage 6) is framework-agnostic by
construction: every function there accepts "a callable satisfying
``model(x_batch, training=False) -> array-like(batch, n_classes)``" as its
``model`` argument — that contract is what already lets a raw Keras model,
a ``TFLiteCallable`` (``benchmark.py``), or a mock test double all work with
``get_predictions()`` / ``compute_evaluation_summary()`` unmodified.

``GesturePredictor`` implements ``__call__`` to satisfy that EXACT contract.
This means a ``GesturePredictor`` instance can be passed directly as the
``model`` argument to ``get_predictions()``, ``benchmark_inference()``, or
``compute_evaluation_summary()`` with zero adapter code — Stage 8's
``verify.py`` and any future ``signer_analysis.py`` re-run against the
*real* inference path (pipeline + model, exactly as Stage 9's webcam demo
will run it) rather than against the bare Keras model in isolation. See
``GesturePredictor.__call__`` and the "Integration Contract" section below.

Module-level architecture
--------------------------
    src/inference/predictor.py
    ├── PredictionSmoother   — sliding-window majority vote (hard prediction)
    │                          + exponential smoothing (display confidence)
    ├── FrameBuffer          — rolling fixed-length raw-landmark accumulator
    ├── GesturePredictor     — unified inference class (primary public API)
    │   ├── __init__()                  — load pipeline, label map, model, smoother
    │   ├── from_config_snapshot()      — classmethod: reconstruct from a
    │   │                                 saved run's config_snapshot.yaml
    │   ├── predict_from_landmarks()    — (T_raw, 225) → prediction dict
    │   ├── predict_from_video()        — video file path → prediction dict
    │   ├── predict_from_webcam_frame() — single BGR frame → dict | None
    │   ├── __call__()                  — evaluation-framework-compatible
    │   │                                 batch callable (see above)
    │   ├── reset()                     — clear buffer + smoother state
    │   └── close() / context manager   — release MediaPipe resources
    └── tests/test_predictor.py  (Stage 7 completion gate — not in this file)

Integration contract with other stages
-----------------------------------------
Stage 8 (``src/export/verify.py``), once a verified ``.tflite`` exists::

    predictor = GesturePredictor.from_config_snapshot(
        config_snapshot_path="artifacts/experiments/bilstm_hands_only_v4_aug/config_snapshot.yaml",
        model_path="models/gesture_bilstm_v1.tflite",
        smoother_window=1,            # disable majority voting for a clean
                                      # apples-to-apples accuracy comparison
    )
    y_true, y_pred, y_prob = get_predictions(
        predictor, val_ds, n_classes=35, return_probs=True,
    )  # predictor used directly as `model` — no adapter needed

Stage 9 (``src/demo/webcam_demo.py``)::

    predictor = GesturePredictor.from_config_snapshot(
        "artifacts/experiments/bilstm_hands_only_v4_aug/config_snapshot.yaml",
        model_path="models/gesture_bilstm_v1.tflite",  # or the SavedModel dir
                                                        # today, pre-Stage-8
    )
    while True:
        ret, frame = cap.read()
        result = predictor.predict_from_webcam_frame(frame)
        if result is None:
            continue  # buffer still filling, or auto-reset just fired
        if result["is_confident"]:
            draw_sign_label(overlay, result["sign"], result["confidence"])
        ...

Thread safety
--------------
A ``GesturePredictor`` instance is NOT thread-safe: ``PredictionSmoother``
and ``FrameBuffer`` hold mutable streaming state, and a single
``tf.lite.Interpreter`` instance must not be invoked concurrently from
multiple threads. Construct one instance per worker/thread if parallelising.
"""

from __future__ import annotations

import time
from collections import Counter, deque
from pathlib import Path
from typing import (
    Any,
    Deque,
    Dict,
    List,
    Optional,
    Tuple,
    TypedDict,
    Union,
)

import numpy as np

from src.features.constants import FEATURE_SIZE
from src.features.pipeline import FeaturePipeline
from src.utils.label_map import LabelMap, get_label_map
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Soft dependency: HIGH_RISK_SIGNS from Stage 6's metrics.py.
#
# Importing this is a deliberate, narrow coupling — it lets every prediction
# carry the same "is_high_risk_class" flag that Stage 6's per-class reports
# already use, so a HUD or report doesn't need a second, possibly-drifting
# copy of this list. Guarded with a fallback literal (kept byte-for-byte
# identical to metrics.HIGH_RISK_SIGNS) so this module never hard-fails to
# import if src/evaluation is unavailable in a minimal deployment image
# (e.g. Dockerfile.inference, which intentionally excludes src/evaluation).
# ---------------------------------------------------------------------------
try:
    from src.evaluation.metrics import HIGH_RISK_SIGNS
except ImportError:  # pragma: no cover - minimal inference-only deployments
    HIGH_RISK_SIGNS: Tuple[str, ...] = (
        "clothes", "think", "birthday", "name", "book",
    )
    logger.debug(
        "src.evaluation.metrics not importable — using the inlined "
        "HIGH_RISK_SIGNS fallback. This is expected in minimal inference-only "
        "deployments (e.g. Dockerfile.inference) that exclude src/evaluation.",
        extra={"stage": "inference"},
    )


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Majority-vote window for PredictionSmoother. 5 frames at 30 FPS ≈ 167ms
#: latency — matches the project standard from the Stage 7 spec (Part 7,
#: Stage 9) and is small enough to feel responsive in the webcam demo while
#: still eliminating single-frame MediaPipe jitter.
DEFAULT_SMOOTHER_WINDOW: int = 5

#: Exponential smoothing factor for the HUD confidence display. At
#: alpha=0.4 the effective memory is ~1/(1-0.4) ≈ 1.67 frames — responsive
#: to new evidence without visibly oscillating. Tuned independently of
#: DEFAULT_SMOOTHER_WINDOW: majority voting answers "what sign?" (discrete),
#: exponential smoothing answers "how confident?" (continuous display only).
DEFAULT_SMOOTHING_ALPHA: float = 0.4

#: Calibration-aware display threshold (Stage 6 finding, not a guess).
#: src/evaluation/calibration.py's reliability diagram on the real val
#: predictions found mean_confidence=0.5136 vs mean_accuracy=0.5769 for the
#: champion — the model is UNDERCONFIDENT. A naive 0.50 cutoff would
#: needlessly suppress a meaningful fraction of correct predictions. 0.35
#: is the Stage 6-documented value; see __init__'s calibration_report_path
#: parameter for loading a per-deployment-recomputed threshold instead.
DEFAULT_DISPLAY_THRESHOLD: float = 0.35

#: Default number of top-k alternative signs returned alongside the winner,
#: for HUD bar charts / debugging.
DEFAULT_TOP_K: int = 3

#: Consecutive all-zero (no-detection) frames in predict_from_webcam_frame()
#: after which the rolling buffer and smoother are auto-reset, so a signer
#: pausing or stepping out of frame doesn't leave stale buffer content to
#: contaminate the next sign's prediction window. Matches the Stage 7 spec's
#: Stage 9 HUD behaviour (Part 7, Stage 9) but is implemented here, inside
#: the predictor, rather than left to every consumer to reimplement. Set to
#: None at construction to disable.
DEFAULT_AUTO_RESET_NO_DETECTION_FRAMES: Optional[int] = 3

#: Documented parameter count of the verified champion
#: (bilstm_hands_only_v4_aug: BiLSTM, 2 layers, hidden_units=64 →
#: 32 units/direction, hands_only, seq_len=100). Used only as a SOFT sanity
#: check (a logged warning, never a hard failure) when a loaded Keras model
#: happens to match the champion's (seq_len, landmark_config) shape but
#: reports a different parameter count — which would indicate the wrong
#: SavedModel was loaded by path. Never asserted for non-champion-shaped
#: models, since this module is not hardcoded to one model.
_EXPECTED_CHAMPION_PARAM_COUNT: int = 68_771

#: Bytes per float32 parameter — used only for an uncompressed weight-size
#: estimate, identical to the formula used in benchmark.py / architectures.py.
_BYTES_PER_FLOAT32: int = 4

#: Best-effort reference hash from the champion's verified config_snapshot.yaml
#: (artifacts/experiments/bilstm_hands_only_v4_aug/config_snapshot.yaml).
#: Used only for an informational match/mismatch log line in
#: from_config_snapshot() — never asserted, since a person may deliberately
#: load a different model's snapshot through this same classmethod.
_KNOWN_CHAMPION_CONFIG_HASH: str = (
    "5809193d37e0d480e409b8e3112e70c8de9008497a29727b411a7128e73287a6"
)

#: Default label map path, resolved relative to the repository root exactly
#: like GestureDataset's default (artifacts/label_map_v1.json).
_REPO_ROOT: Path = Path(__file__).resolve().parents[2]
_DEFAULT_LABEL_MAP_PATH: Path = _REPO_ROOT / "artifacts" / "label_map_v1.json"


# ---------------------------------------------------------------------------
# Typed result schema (documentation + IDE/static-analysis aid only — these
# functions still return plain dicts at runtime, consistent with the rest of
# the codebase, e.g. src/evaluation/metrics.py).
# ---------------------------------------------------------------------------

class TopKEntry(TypedDict):
    sign: str
    class_idx: int
    confidence: float


class PredictionResult(TypedDict, total=False):
    sign: str
    confidence: float
    is_confident: bool
    class_idx: int
    top_k: List[TopKEntry]
    raw_confidence: float
    raw_class_idx: int
    is_stable: bool
    is_high_risk_class: bool
    n_frames_input: int
    inference_latency_ms: float
    frames_in_buffer: int  # webcam streaming path only


# ---------------------------------------------------------------------------
# PredictionSmoother
# ---------------------------------------------------------------------------

class PredictionSmoother:
    """
    Dual-mechanism temporal smoother for streaming gesture predictions.

    Mechanism 1 — majority voting (the reported HARD prediction)
        Maintains the last ``window`` per-frame argmax class indices. The
        reported class is the mode of that window. Ties are broken by
        recency (the most recently seen tied class wins) — appropriate for
        real-time signing, where the signer has plausibly moved on to a new
        sign by the time a tie occurs.

    Mechanism 2 — exponential smoothing (the displayed CONFIDENCE)
        ``smoothed[t] = alpha * raw[t] + (1 - alpha) * smoothed[t-1]``,
        applied to the full probability vector. Drives a smooth HUD
        confidence bar without affecting which class is reported — the hard
        prediction always comes from majority voting, never from the
        smoothed vector's argmax. Coupling the two (e.g. taking argmax of
        the smoothed vector) would understate genuine model indecision
        between two visually similar signs by averaging them into a single
        misleadingly confident-looking blend.

    Calibration context (Stage 6)
        The champion is underconfident (mean confidence 0.5136 vs mean
        accuracy 0.5769 on the val set) — smoothed confidences for CORRECT
        predictions will typically sit in the 0.3-0.7 range, not near 1.0.
        This class does not apply any display threshold itself — that
        decision belongs to ``GesturePredictor`` (``display_threshold``),
        so different consumers (demo vs. batch evaluation) can apply
        different thresholds against the same smoothed signal.

    Not thread-safe: internal deque/array state is mutated by every call to
    ``update()``.

    Parameters
    ----------
    window : int, default 5
        Number of frames for majority voting. Must be >= 1; window=1
        degenerates to "no smoothing" (the per-frame argmax), which is the
        correct setting for a clean accuracy comparison (Stage 8 verify.py).
    alpha : float, default 0.4
        Exponential smoothing factor in (0.0, 1.0] for the displayed
        confidence vector.
    n_classes : int, default 35
        Number of output classes — initialises the uniform-prior smoothed
        probability buffer.

    Raises
    ------
    ValueError
        If ``window < 1``, ``alpha`` is outside ``(0.0, 1.0]``, or
        ``n_classes < 2``.
    """

    def __init__(
        self,
        window: int = DEFAULT_SMOOTHER_WINDOW,
        alpha: float = DEFAULT_SMOOTHING_ALPHA,
        n_classes: int = 35,
    ) -> None:
        if window < 1:
            raise ValueError(f"PredictionSmoother: window={window} must be >= 1.")
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"PredictionSmoother: alpha={alpha} must be in (0.0, 1.0].")
        if n_classes < 2:
            raise ValueError(f"PredictionSmoother: n_classes={n_classes} must be >= 2.")

        self._window: int = int(window)
        self._alpha: float = float(alpha)
        self._n_classes: int = int(n_classes)

        self._history: Deque[int] = deque(maxlen=self._window)
        self._smoothed_probs: np.ndarray = np.full(
            self._n_classes, 1.0 / self._n_classes, dtype=np.float32,
        )
        self._stable_count: int = 0
        self._last_winner: Optional[int] = None

    def update(self, raw_probs: np.ndarray) -> Tuple[int, np.ndarray, bool]:
        """
        Feed one frame's raw softmax output through both smoothing mechanisms.

        Parameters
        ----------
        raw_probs : np.ndarray, shape (n_classes,)
            Raw softmax probabilities from the model for this single frame.

        Returns
        -------
        Tuple[int, np.ndarray, bool]
            predicted_class : majority-vote winner over the window.
            smoothed_probs  : (n_classes,) float32, exponentially smoothed
                              probability vector, for HUD display.
            is_stable       : True once the same class has won for
                              >= ``window`` consecutive frames.
        """
        raw_probs = np.asarray(raw_probs, dtype=np.float32)
        if raw_probs.shape != (self._n_classes,):
            raise ValueError(
                f"PredictionSmoother.update(): raw_probs has shape "
                f"{raw_probs.shape}, expected ({self._n_classes},)."
            )

        # --- Mechanism 1: majority vote with recency tiebreak ---
        frame_class = int(np.argmax(raw_probs))
        self._history.append(frame_class)

        counts = Counter(self._history)
        max_count = max(counts.values())
        candidates = {c for c, cnt in counts.items() if cnt == max_count}
        if len(candidates) == 1:
            winner = next(iter(candidates))
        else:
            # Multiple classes tied for the mode: prefer whichever appeared
            # most recently in the window.
            winner = next(c for c in reversed(self._history) if c in candidates)

        if winner == self._last_winner:
            self._stable_count += 1
        else:
            self._stable_count = 1
            self._last_winner = winner
        is_stable = self._stable_count >= self._window

        # --- Mechanism 2: exponential smoothing of the probability vector ---
        self._smoothed_probs = (
            self._alpha * raw_probs + (1.0 - self._alpha) * self._smoothed_probs
        ).astype(np.float32)

        return winner, self._smoothed_probs.copy(), is_stable

    def top_k(self, smoothed_probs: np.ndarray, k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
        """
        Return the top-``k`` classes by smoothed probability, descending.

        Sign names are NOT resolved here — ``GesturePredictor`` owns the
        label map and resolves names; this keeps ``PredictionSmoother``
        free of any label-map dependency.

        Returns
        -------
        List[dict] — each ``{"class_idx": int, "confidence": float}``.
        """
        k = max(1, min(k, self._n_classes))
        top_idx = np.argpartition(smoothed_probs, -k)[-k:]
        top_idx = top_idx[np.argsort(smoothed_probs[top_idx])[::-1]]
        return [
            {"class_idx": int(idx), "confidence": float(smoothed_probs[idx])}
            for idx in top_idx
        ]

    def reset(self) -> None:
        """Clear all history and return the smoother to its initial state."""
        self._history.clear()
        self._smoothed_probs[:] = 1.0 / self._n_classes
        self._stable_count = 0
        self._last_winner = None

    @property
    def window(self) -> int:
        return self._window

    def __repr__(self) -> str:
        return (
            f"PredictionSmoother(window={self._window}, alpha={self._alpha}, "
            f"n_classes={self._n_classes}, history_len={len(self._history)})"
        )


# ---------------------------------------------------------------------------
# FrameBuffer
# ---------------------------------------------------------------------------

class FrameBuffer:
    """
    Rolling fixed-length accumulator for raw, full-dimensional landmark frames.

    Invariant — ALWAYS stores raw FEATURE_SIZE-dim vectors, never pre-sliced
    -----------------------------------------------------------------------
    ``FeaturePipeline.__call__`` (and ``pre_augmentation``) require the full
    225-dim vector as input: wrist-relative normalisation and z-clipping
    index into the full feature space via ``LEFT_HAND_SLICE`` /
    ``RIGHT_HAND_SLICE`` / ``arr[:, 2::3]`` BEFORE landmark-config selection
    happens (selection is step 7 of 8 inside the pipeline). If this buffer
    stored pre-sliced ``hands_only`` (126-dim) vectors instead, every call
    to the pipeline would immediately fail its own step-1 shape guard. This
    class enforces that invariant via ``n_features`` (always
    ``FEATURE_SIZE`` = 225 in practice) — see ``add_frame()``.

    Uses ``collections.deque(maxlen=seq_len)`` for O(1) append with
    automatic oldest-frame eviction once full — the correct data structure
    for a rolling window, even though at 30 FPS the difference versus a
    naive list is not perceptible at this project's scale.

    Not thread-safe.

    Parameters
    ----------
    seq_len : int
        Target window length in frames (e.g. 100 for the champion).
    n_features : int, default FEATURE_SIZE (225)
        Expected per-frame feature dimension. Always the FULL raw dimension,
        never a landmark-config-sliced one — see invariant above.
    """

    def __init__(self, seq_len: int, n_features: int = FEATURE_SIZE) -> None:
        if seq_len < 1:
            raise ValueError(f"FrameBuffer: seq_len={seq_len} must be >= 1.")
        if n_features < 1:
            raise ValueError(f"FrameBuffer: n_features={n_features} must be >= 1.")
        self._seq_len: int = int(seq_len)
        self._n_features: int = int(n_features)
        self._buffer: Deque[np.ndarray] = deque(maxlen=self._seq_len)

    def add_frame(self, landmark_vec: np.ndarray) -> None:
        """
        Append one frame's raw landmark vector.

        Parameters
        ----------
        landmark_vec : np.ndarray, shape (n_features,)
            Raw vector from ``LandmarkExtractor.extract_frame()`` — zero-filled
            (per the Stage 3 convention) if MediaPipe detected nothing this frame.

        Raises
        ------
        ValueError
            If ``landmark_vec`` does not have shape ``(n_features,)``.
        """
        landmark_vec = np.asarray(landmark_vec)
        if landmark_vec.shape != (self._n_features,):
            raise ValueError(
                f"FrameBuffer.add_frame(): expected shape ({self._n_features},), "
                f"got {landmark_vec.shape}. Always pass the full "
                f"{self._n_features}-dim raw landmark vector — landmark-config "
                "slicing happens inside FeaturePipeline, never in the buffer."
            )
        self._buffer.append(landmark_vec.astype(np.float32, copy=False))

    def is_ready(self) -> bool:
        """True once the buffer holds exactly ``seq_len`` frames."""
        return len(self._buffer) == self._seq_len

    def get_array(self) -> np.ndarray:
        """
        Return the current window as a ``(seq_len, n_features)`` array copy.

        Raises
        ------
        RuntimeError
            If called before ``is_ready()`` is True.
        """
        if not self.is_ready():
            raise RuntimeError(
                f"FrameBuffer.get_array() called with only "
                f"{len(self._buffer)}/{self._seq_len} frames buffered. "
                "Check is_ready() before calling get_array()."
            )
        return np.array(self._buffer, dtype=np.float32)

    def frames_accumulated(self) -> int:
        """Current number of frames buffered (<= seq_len)."""
        return len(self._buffer)

    def reset(self) -> None:
        """Clear all accumulated frames."""
        self._buffer.clear()

    def __repr__(self) -> str:
        return (
            f"FrameBuffer(seq_len={self._seq_len}, n_features={self._n_features}, "
            f"buffered={len(self._buffer)}/{self._seq_len})"
        )


# ---------------------------------------------------------------------------
# GesturePredictor
# ---------------------------------------------------------------------------

class GesturePredictor:
    """
    Unified inference engine for WLASL gesture recognition (Stage 7).

    Guarantees, by construction
    ------------------------------
    1. Preprocessing consistency — a single ``FeaturePipeline`` instance
       (built from the SAME ``ExperimentConfig`` the model was trained with)
       is used for every input source, so training/inference preprocessing
       can never silently diverge.
    2. Calibration-aware output — ``is_confident`` is gated on a threshold
       calibrated to the Stage 6 finding that this model is underconfident,
       not the naive 0.50 cutoff (see ``DEFAULT_DISPLAY_THRESHOLD``).
    3. Zero training-mode contamination — ``FeaturePipeline`` is invoked
       with ``training=False`` unconditionally; this is Critical Rule #8
       from the project handoff and is enforced here, not left to caller
       discipline.
    4. Evaluation-framework compatible — ``__call__`` satisfies the
       ``model(x_batch, training=False) -> probs`` contract used throughout
       ``src/evaluation``, so this class doubles as a drop-in ``model`` for
       ``get_predictions()`` / ``benchmark_inference()`` (see module
       docstring "Integration contract").

    Supported inference entry points
    ------------------------------------
        predict_from_landmarks()      (T_raw, 225) raw array     → dict
        predict_from_video()          video file path            → dict
        predict_from_webcam_frame()   single BGR frame           → dict | None
        __call__()                    (batch, seq_len, feat_dim) → (batch, n_classes)
                                       already-pipelined input — evaluation use

    Model format auto-detection
    -------------------------------
        *.tflite file        → tf.lite.Interpreter (Stage 8 deployment target)
        directory / *.keras  → tf.keras SavedModel (current state, pre-Stage-8;
                                also Stage 8's accuracy-comparison fallback)

    Parameters
    ----------
    model_path : str | Path
        Path to a ``.tflite`` file or a Keras SavedModel directory.
    config : Any
        The frozen ``ExperimentConfig`` the model was TRAINED with. Prefer
        constructing this via ``from_config_snapshot()`` rather than
        re-deriving it from ``load_config(...)`` CLI-style arguments — see
        the module docstring's point 2 for why that matters for this
        project's champion specifically.
    label_map_path : str | Path, optional
        Defaults to ``artifacts/label_map_v1.json`` (the project default).
    smoother_window : int, default 5
        See ``PredictionSmoother``. Pass ``1`` to disable majority voting
        for a clean per-clip accuracy comparison (Stage 8).
    smoothing_alpha : float, default 0.4
        See ``PredictionSmoother``.
    display_threshold : float, optional
        Explicit override. If omitted, resolved via
        ``calibration_report_path`` (if given and present on disk) or else
        ``DEFAULT_DISPLAY_THRESHOLD`` (0.35, the Stage 6 finding).
    calibration_report_path : str | Path, optional
        Path to a Stage 6 ``evaluation_report.json``
        (``compute_calibration_summary()`` output). If present, its
        ``threshold_curve.optimal_threshold_accuracy.threshold`` is used as
        the display threshold instead of the hardcoded default — letting a
        re-calibrated deployment pick up an updated threshold without a
        code change. Failures to parse fall back to the default with a
        logged warning rather than raising.
    n_top_k : int, default 3
        Number of alternative signs returned per prediction.
    flag_high_risk_classes : bool, default True
        Whether to annotate predictions with ``is_high_risk_class`` using
        Stage 6's ``HIGH_RISK_SIGNS``.
    auto_reset_no_detection_frames : int, optional
        See ``DEFAULT_AUTO_RESET_NO_DETECTION_FRAMES``. ``None`` disables.

    Raises
    ------
    FileNotFoundError
        If ``model_path`` or ``label_map_path`` does not exist.
    ValueError
        If the label map's class count disagrees with ``config.num_classes``,
        if the label map appears to contain placeholder names (the exact
        Stage 6 bug class — see module docstring), or if any constructor
        parameter is out of its valid range.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        config: Any,
        label_map_path: Optional[Union[str, Path]] = None,
        smoother_window: int = DEFAULT_SMOOTHER_WINDOW,
        smoothing_alpha: float = DEFAULT_SMOOTHING_ALPHA,
        display_threshold: Optional[float] = None,
        calibration_report_path: Optional[Union[str, Path]] = None,
        n_top_k: int = DEFAULT_TOP_K,
        flag_high_risk_classes: bool = True,
        auto_reset_no_detection_frames: Optional[int] = DEFAULT_AUTO_RESET_NO_DETECTION_FRAMES,
    ) -> None:
        if n_top_k < 1:
            raise ValueError(f"GesturePredictor: n_top_k={n_top_k} must be >= 1.")
        if auto_reset_no_detection_frames is not None and auto_reset_no_detection_frames < 1:
            raise ValueError(
                "GesturePredictor: auto_reset_no_detection_frames must be "
                "None or >= 1."
            )

        self._config = config
        self._n_top_k = int(n_top_k)
        self._flag_high_risk = bool(flag_high_risk_classes)
        self._auto_reset_threshold = auto_reset_no_detection_frames
        self._no_detection_streak: int = 0
        self._extractor: Any = None  # lazy-initialised MediaPipe extractor

        # ── Step 1: FeaturePipeline — single source of preprocessing truth ──
        self._pipeline = FeaturePipeline(config)
        expected_shape = (int(config.data.sequence_length), self._pipeline.feature_dim)
        if self._pipeline.output_shape != expected_shape:
            raise ValueError(
                f"GesturePredictor: pipeline.output_shape={self._pipeline.output_shape} "
                f"does not match expected {expected_shape} derived from the supplied "
                "config. This should be impossible for a self-consistent "
                "ExperimentConfig — check that 'config' was not mutated after "
                "FeaturePipeline construction."
            )
        if bool(config.augmentation.enabled):
            logger.info(
                "GesturePredictor: config.augmentation.enabled=True, but this is "
                "irrelevant at inference — FeaturePipeline is always invoked with "
                "training=False, which unconditionally bypasses augmentation "
                "(Critical Rule #8).",
                extra={"stage": "inference"},
            )

        self._seq_len: int = int(config.data.sequence_length)
        self._n_classes: int = int(config.num_classes)

        # ── Step 2: Label map — validated against the exact Stage 6 bug class ──
        label_map_path = Path(label_map_path) if label_map_path else _DEFAULT_LABEL_MAP_PATH
        self._label_map: LabelMap = get_label_map(label_map_path)

        if self._label_map.num_classes != self._n_classes:
            raise ValueError(
                f"GesturePredictor: label map at {label_map_path} has "
                f"{self._label_map.num_classes} classes but config.num_classes="
                f"{self._n_classes}. Ensure the label map matches the config "
                "this model was trained with."
            )

        # Placeholder-name guard. LabelMap._load() already rejects duplicate
        # sign names internally (see src/utils/label_map.py), so duplicates
        # cannot reach this point — this check exists specifically to catch
        # the Stage 6 incident where a schema mismatch silently produced
        # "class_0".."class_34" placeholders, which made every confusable-
        # pair / high-risk-class analysis meaningless without raising any
        # error on its own. Fail loudly here instead.
        sample_n = min(5, self._n_classes)
        sample_names = [
            self._label_map.get_name_safe(i, f"PLACEHOLDER_{i}") for i in range(sample_n)
        ]
        if any("PLACEHOLDER" in n or n.startswith("class_") for n in sample_names):
            raise ValueError(
                f"GesturePredictor: label map at {label_map_path} appears to "
                f"contain placeholder names: {sample_names}. This is the exact "
                "failure mode identified in Stage 6 — verify the JSON schema "
                "matches LabelMap's expected format "
                '({"_metadata": {...}, "classes": {"0": "before", ...}}).'
            )

        # ── Step 3: Display threshold resolution (calibration-aware) ──
        self._display_threshold, threshold_source = self._resolve_display_threshold(
            display_threshold, calibration_report_path,
        )

        # ── Step 4: Model loading (auto-detected format) ──
        self._model_type: str = ""
        self._keras_model: Any = None
        self._interpreter: Any = None
        self._input_index: Optional[int] = None
        self._output_index: Optional[int] = None
        self._tflite_fixed_input_shape: Optional[Tuple[int, ...]] = None
        self._tflite_has_dynamic_batch: bool = False
        self._model_path = Path(model_path)
        self._load_model(self._model_path)

        # ── Step 5: Smoother + rolling frame buffer ──
        self._smoother = PredictionSmoother(
            window=smoother_window, alpha=smoothing_alpha, n_classes=self._n_classes,
        )
        self._frame_buffer = FrameBuffer(seq_len=self._seq_len, n_features=FEATURE_SIZE)

        logger.info(
            "GesturePredictor ready | model_type=%s | seq_len=%d | "
            "landmark_config=%s | feature_dim=%d | n_classes=%d | "
            "display_threshold=%.3f (source=%s) | smoother_window=%d | "
            "config_hash=%s",
            self._model_type, self._seq_len, self._pipeline.landmark_config,
            self._pipeline.feature_dim, self._n_classes, self._display_threshold,
            threshold_source, self._smoother.window,
            str(getattr(config, "config_hash", "unknown"))[:12],
            extra={"stage": "inference"},
        )

    # ══════════════════════════════════════════════════════════════════════
    # Alternative constructor — the recommended entry point
    # ══════════════════════════════════════════════════════════════════════

    @classmethod
    def from_config_snapshot(
        cls,
        config_snapshot_path: Union[str, Path],
        model_path: Union[str, Path],
        label_map_path: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> "GesturePredictor":
        """
        Construct a ``GesturePredictor`` from a saved run's
        ``config_snapshot.yaml`` rather than re-deriving the config via
        ``load_config(model=..., data=..., augmentation=..., overrides=...)``.

        This is the RECOMMENDED construction path for this project. The
        champion's true training config — in particular
        ``data.landmark_config: hands_only`` — is not reproducible purely
        from ``configs/data/seq100.yaml`` (whose Pydantic default is
        ``"full"``); it was applied as a run-time CLI override during
        Stage 5 and is durably recorded ONLY in
        ``artifacts/experiments/<run_name>/config_snapshot.yaml``. Loading
        that file directly removes any risk of silently reconstructing the
        wrong preprocessing configuration.

        Parameters
        ----------
        config_snapshot_path : str | Path
            e.g. ``"artifacts/experiments/bilstm_hands_only_v4_aug/config_snapshot.yaml"``.
        model_path : str | Path
            Path to the corresponding ``.tflite`` file or SavedModel directory.
        label_map_path : str | Path, optional
            Defaults to ``artifacts/label_map_v1.json``.
        **kwargs
            Forwarded to ``GesturePredictor.__init__`` (e.g.
            ``smoother_window``, ``display_threshold``,
            ``calibration_report_path``).

        Returns
        -------
        GesturePredictor

        Raises
        ------
        FileNotFoundError
            If ``config_snapshot_path`` does not exist.
        """
        from omegaconf import OmegaConf

        from src.utils.config import ExperimentConfig

        snapshot_path = Path(config_snapshot_path)
        if not snapshot_path.exists():
            raise FileNotFoundError(
                f"GesturePredictor.from_config_snapshot(): config snapshot not "
                f"found at {snapshot_path}. This file is written once per run "
                "by src/utils/reproducibility.py's setup_experiment() under "
                "artifacts/experiments/<run_name>/config_snapshot.yaml."
            )

        raw = OmegaConf.to_container(OmegaConf.load(snapshot_path), resolve=True)
        config = ExperimentConfig(**raw)

        observed_hash = str(getattr(config, "config_hash", ""))
        if observed_hash and not _KNOWN_CHAMPION_CONFIG_HASH.startswith(observed_hash[:12]):
            logger.debug(
                "from_config_snapshot(): loaded config_hash=%s does not match "
                "the known champion reference hash (%s...). This is expected "
                "and harmless if you are intentionally loading a different "
                "model's snapshot.",
                observed_hash[:12], _KNOWN_CHAMPION_CONFIG_HASH[:12],
                extra={"stage": "inference"},
            )

        return cls(
            model_path=model_path,
            config=config,
            label_map_path=label_map_path,
            **kwargs,
        )

    # ══════════════════════════════════════════════════════════════════════
    # Construction helpers
    # ══════════════════════════════════════════════════════════════════════

    def _resolve_display_threshold(
        self,
        explicit: Optional[float],
        calibration_report_path: Optional[Union[str, Path]],
    ) -> Tuple[float, str]:
        """
        Resolve the display-confidence threshold, preferring (in order):
        an explicit constructor argument, a Stage 6 calibration report on
        disk, then the Stage 6-documented hardcoded default.

        Returns
        -------
        Tuple[float, str] — (threshold, source_description)
        """
        if explicit is not None:
            if not (0.0 <= explicit <= 1.0):
                raise ValueError(
                    f"GesturePredictor: display_threshold={explicit} must be in [0.0, 1.0]."
                )
            return float(explicit), "explicit"

        if calibration_report_path is not None:
            path = Path(calibration_report_path)
            if path.exists():
                try:
                    import json

                    with open(path, encoding="utf-8") as f:
                        report = json.load(f)
                    threshold = (
                        report.get("threshold_curve", {})
                        .get("optimal_threshold_accuracy", {})
                        .get("threshold")
                    )
                    if threshold is not None and 0.0 <= float(threshold) <= 1.0:
                        return float(threshold), f"calibration_report:{path.name}"
                    logger.warning(
                        "GesturePredictor: calibration report at %s did not "
                        "contain a usable threshold_curve.optimal_threshold_accuracy"
                        ".threshold — falling back to the default (%.2f).",
                        path, DEFAULT_DISPLAY_THRESHOLD,
                        extra={"stage": "inference"},
                    )
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    logger.warning(
                        "GesturePredictor: failed to parse calibration report "
                        "at %s (%s: %s) — falling back to the default (%.2f).",
                        path, type(exc).__name__, exc, DEFAULT_DISPLAY_THRESHOLD,
                        extra={"stage": "inference"},
                    )
            else:
                logger.warning(
                    "GesturePredictor: calibration_report_path=%s does not "
                    "exist — falling back to the default (%.2f).",
                    path, DEFAULT_DISPLAY_THRESHOLD,
                    extra={"stage": "inference"},
                )

        return DEFAULT_DISPLAY_THRESHOLD, "default(stage6_underconfidence_finding)"

    def _load_model(self, model_path: Path) -> None:
        """Auto-detect and load the model, dispatching to the TFLite or Keras path."""
        if not model_path.exists():
            raise FileNotFoundError(
                f"GesturePredictor: model not found at {model_path}. Expected "
                "either a .tflite file (Stage 8 deployment target) or a Keras "
                "SavedModel directory (e.g. "
                "models/bilstm_hands_only_v4_aug_saved_model/)."
            )

        if model_path.is_file() and model_path.suffix == ".tflite":
            self._load_tflite(model_path)
        elif model_path.is_dir() or model_path.suffix == ".keras":
            self._load_keras(model_path)
        else:
            raise ValueError(
                f"GesturePredictor: cannot determine model format from "
                f"{model_path}. Expected a '.tflite' file or a SavedModel "
                "directory."
            )

    def _load_tflite(self, model_path: Path) -> None:
        import tensorflow as tf

        interpreter = tf.lite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        if len(input_details) != 1 or len(output_details) != 1:
            raise ValueError(
                f"GesturePredictor: TFLite model at {model_path} has "
                f"{len(input_details)} input(s) and {len(output_details)} "
                "output(s); expected exactly one of each (a single landmark-"
                "sequence input feeding a Dense(n_classes, softmax) head)."
            )

        self._interpreter = interpreter
        self._input_index = input_details[0]["index"]
        self._output_index = output_details[0]["index"]

        raw_shape = tuple(int(d) for d in input_details[0]["shape"])
        self._tflite_has_dynamic_batch = len(raw_shape) > 0 and raw_shape[0] == -1
        self._tflite_fixed_input_shape = raw_shape

        expected = (1, self._seq_len, self._pipeline.feature_dim)
        if not self._tflite_has_dynamic_batch and raw_shape != expected:
            raise ValueError(
                f"GesturePredictor: TFLite model input shape {raw_shape} does "
                f"not match expected {expected} (derived from the supplied "
                "config's sequence_length and landmark_config). Ensure this "
                ".tflite file was exported from a model trained with the same "
                "seq_len / landmark_config as the supplied config."
            )

        self._model_type = "tflite"
        file_size_mb = round(model_path.stat().st_size / (1024 ** 2), 4)
        logger.info(
            "TFLite interpreter loaded | path=%s | input_shape=%s | "
            "dynamic_batch=%s | file_size=%.4fMB",
            model_path, raw_shape, self._tflite_has_dynamic_batch, file_size_mb,
            extra={"stage": "inference"},
        )

    def _load_keras(self, model_path: Path) -> None:
        import tensorflow as tf

        model = tf.keras.models.load_model(str(model_path))
        param_count = int(model.count_params())

        is_champion_shape = (
            self._seq_len == 100 and self._pipeline.landmark_config == "hands_only"
        )
        if is_champion_shape and param_count != _EXPECTED_CHAMPION_PARAM_COUNT:
            logger.warning(
                "GesturePredictor: loaded Keras model at %s has %d parameters; "
                "the verified champion (bilstm_hands_only_v4_aug, seq_len=100, "
                "landmark_config=hands_only) has %d. This shape matches the "
                "champion's preprocessing config but a different parameter "
                "count — verify this is the intended SavedModel before "
                "trusting its predictions.",
                model_path, param_count, _EXPECTED_CHAMPION_PARAM_COUNT,
                extra={"stage": "inference"},
            )

        self._keras_model = model
        self._model_type = "keras"
        model_size_mb = round(param_count * _BYTES_PER_FLOAT32 / (1024 ** 2), 4)
        logger.info(
            "Keras SavedModel loaded | path=%s | params=%d | "
            "estimated_size=%.4fMB (uncompressed float32 weights)",
            model_path, param_count, model_size_mb,
            extra={"stage": "inference"},
        )

    # ══════════════════════════════════════════════════════════════════════
    # Core forward pass
    # ══════════════════════════════════════════════════════════════════════

    def _run_single(self, features_batched: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Run one forward pass on a ``(1, seq_len, feature_dim)`` tensor.

        Returns
        -------
        Tuple[np.ndarray, float] — (probs of shape (n_classes,), elapsed_ms)
        """
        t0 = time.perf_counter()
        if self._model_type == "tflite":
            probs = self._run_tflite(features_batched)
        else:
            probs = self._run_keras(features_batched)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return probs, elapsed_ms

    def _run_tflite(self, features_batched: np.ndarray) -> np.ndarray:
        sample = np.asarray(features_batched, dtype=np.float32)
        actual_shape = tuple(sample.shape)

        if actual_shape != self._tflite_fixed_input_shape:
            if self._tflite_has_dynamic_batch:
                self._interpreter.resize_tensor_input(self._input_index, list(actual_shape))
                self._interpreter.allocate_tensors()
                self._input_index = self._interpreter.get_input_details()[0]["index"]
                self._output_index = self._interpreter.get_output_details()[0]["index"]
                self._tflite_fixed_input_shape = actual_shape
            else:
                expected = (1, self._seq_len, self._pipeline.feature_dim)
                raise ValueError(
                    f"GesturePredictor: input shape {actual_shape} does not "
                    f"match the TFLite interpreter's fixed shape {expected}. "
                    "Check that the input came from this predictor's own "
                    "FeaturePipeline."
                )

        self._interpreter.set_tensor(self._input_index, sample)
        self._interpreter.invoke()
        raw = np.array(self._interpreter.get_tensor(self._output_index))
        return raw[0]

    def _run_keras(self, features_batched: np.ndarray) -> np.ndarray:
        import tensorflow as tf

        x = tf.constant(features_batched, dtype=tf.float32)
        # training=False is unconditional and explicit — belt-and-suspenders
        # alongside FeaturePipeline's own training=False gate (Critical Rule #8):
        # even if augmentation were somehow applied upstream, dropout/recurrent
        # dropout layers are disabled here regardless.
        logits = self._keras_model(x, training=False)
        return np.asarray(logits)[0]

    # ══════════════════════════════════════════════════════════════════════
    # Evaluation-framework compatibility
    # ══════════════════════════════════════════════════════════════════════

    def __call__(self, x_batch: Any, training: bool = False) -> np.ndarray:
        """
        Satisfy the ``model(x_batch, training=False) -> probs`` contract used
        throughout ``src/evaluation`` (``metrics.get_predictions``,
        ``benchmark.benchmark_inference``, ``calibration.*``).

        This lets a ``GesturePredictor`` instance be passed directly as the
        ``model`` argument to those functions — e.g. Stage 8's
        ``verify.py`` can evaluate the REAL deployment path (pipeline +
        model) rather than the bare Keras/TFLite model in isolation — with
        no adapter class required.

        IMPORTANT: ``x_batch`` here is ALREADY-PIPELINED model input of
        shape ``(batch, seq_len, feature_dim)`` (exactly what
        ``GestureDataset.load_split()`` yields), NOT raw landmarks. This
        method does not touch ``self._smoother`` / ``self._frame_buffer`` —
        those exist only for the streaming entry points
        (``predict_from_webcam_frame``). ``training`` is accepted for
        interface compatibility and ignored (inference is always
        ``training=False`` regardless of this argument's value, per
        Critical Rule #8).

        Parameters
        ----------
        x_batch : array-like, shape (batch, seq_len, feature_dim) or
                  (seq_len, feature_dim) for a single sample.
        training : bool, ignored.

        Returns
        -------
        np.ndarray, shape (batch, n_classes), float32.
        """
        x = np.asarray(x_batch, dtype=np.float32)
        if x.ndim == 2:
            x = x[np.newaxis, ...]
        if x.ndim != 3:
            raise ValueError(
                f"GesturePredictor.__call__(): x_batch has shape {x.shape}; "
                "expected (seq_len, feature_dim) or (batch, seq_len, feature_dim)."
            )
        outputs = np.stack(
            [self._run_single(x[i : i + 1])[0] for i in range(x.shape[0])],
            axis=0,
        )
        return outputs.astype(np.float32)

    # ══════════════════════════════════════════════════════════════════════
    # Public inference entry points
    # ══════════════════════════════════════════════════════════════════════

    def predict_from_landmarks(
        self,
        landmarks: np.ndarray,
        update_smoother: bool = True,
    ) -> PredictionResult:
        """
        Predict the sign class from a raw ``(T_raw, 225)`` landmark array.

        The foundational inference method — ``predict_from_video()`` and
        ``predict_from_webcam_frame()`` both ultimately call this same
        preprocessing + forward-pass logic.

        Parameters
        ----------
        landmarks : np.ndarray, shape (T_raw, 225)
            Raw landmark array (e.g. loaded via ``np.load()`` from a Stage 3
            ``.npy`` file, or from ``LandmarkExtractor.extract_frame()``
            stacked across frames). Must be the FULL 225-dim array —
            ``FeaturePipeline`` performs landmark-config slicing internally.
        update_smoother : bool, default True
            Whether to feed this prediction into ``PredictionSmoother``.
            Set ``False`` for batch/offline evaluation where clips are
            independent (e.g. Stage 8's val-set verification) — temporal
            smoothing across unrelated clips would corrupt the comparison.

        Returns
        -------
        PredictionResult (dict) — see module docstring's ``PredictionResult``
        TypedDict for the full schema.
        """
        features_2d = self._pipeline(landmarks, training=False)
        features_batched = features_2d[np.newaxis, ...].astype(np.float32)

        raw_probs, elapsed_ms = self._run_single(features_batched)
        raw_confidence = float(raw_probs.max())
        raw_class_idx = int(np.argmax(raw_probs))

        if update_smoother:
            predicted_class, smoothed_probs, is_stable = self._smoother.update(raw_probs)
            display_confidence = float(smoothed_probs[predicted_class])
        else:
            predicted_class = raw_class_idx
            smoothed_probs = raw_probs
            display_confidence = raw_confidence
            is_stable = False

        return self._build_result(
            predicted_class=predicted_class,
            smoothed_probs=smoothed_probs,
            display_confidence=display_confidence,
            raw_confidence=raw_confidence,
            raw_class_idx=raw_class_idx,
            is_stable=is_stable,
            n_frames_input=int(landmarks.shape[0]),
            inference_latency_ms=elapsed_ms,
        )

    def predict_from_video(self, video_path: Union[str, Path]) -> PredictionResult:
        """
        Extract landmarks from a video file and predict the sign class.

        Used by Stage 8 offline verification / CLI tooling. Runs in batch
        mode (``update_smoother=False``) since a single video file is one
        independent clip, not part of a continuous stream.

        Lazily imports ``cv2`` and ``LandmarkExtractor`` (MediaPipe) — these
        heavy, optional dependencies are never imported for the
        landmarks-array or already-pipelined-batch entry points.

        Raises
        ------
        FileNotFoundError
            If ``video_path`` does not exist.
        RuntimeError
            If the file cannot be opened or yields zero frames.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"predict_from_video(): video not found: {video_path}")

        try:
            import cv2
        except ImportError as exc:
            raise ImportError(
                "predict_from_video() requires opencv-python. "
                "Install with: pip install opencv-python"
            ) from exc

        self._ensure_extractor()

        landmarks_list: List[np.ndarray] = []
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"predict_from_video(): OpenCV could not open {video_path}")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                landmarks_list.append(self._extractor.extract_frame(frame))
        finally:
            cap.release()

        if not landmarks_list:
            raise RuntimeError(f"predict_from_video(): no frames extracted from {video_path}")

        landmarks_array = np.stack(landmarks_list, axis=0)
        return self.predict_from_landmarks(landmarks_array, update_smoother=False)

    def predict_from_webcam_frame(self, frame: np.ndarray) -> Optional[PredictionResult]:
        """
        Process one BGR webcam frame; return a prediction once the rolling
        buffer is full, else ``None``.

        Called once per captured frame. Returns ``None`` for the first
        ``seq_len - 1`` frames while the buffer fills, and on every frame
        immediately after an auto-reset (see ``auto_reset_no_detection_frames``
        in ``__init__``). From frame ``seq_len`` onward, returns an updated
        prediction every call as the rolling window advances.

        No-detection auto-reset
        -------------------------
        If ``LandmarkExtractor.extract_frame()`` returns an all-zero vector
        (no hands/pose detected) for ``auto_reset_no_detection_frames``
        consecutive calls, the buffer and smoother are automatically reset
        so stale window content doesn't bleed into the next sign once the
        signer returns. This mirrors the Stage 9 HUD spec but is implemented
        once here rather than in every consumer.

        Parameters
        ----------
        frame : np.ndarray
            A single BGR uint8 frame from ``cv2.VideoCapture.read()``.

        Returns
        -------
        PredictionResult | None
            ``None`` if the buffer is still filling (or was just
            auto-reset). Otherwise a prediction dict with the additional
            key ``frames_in_buffer`` (always equal to ``seq_len``).
        """
        self._ensure_extractor()

        landmark_vec = self._extractor.extract_frame(frame)

        if not np.any(landmark_vec):
            self._no_detection_streak += 1
        else:
            self._no_detection_streak = 0

        self._frame_buffer.add_frame(landmark_vec)

        if (
            self._auto_reset_threshold is not None
            and self._no_detection_streak >= self._auto_reset_threshold
        ):
            logger.info(
                "predict_from_webcam_frame(): %d consecutive no-detection "
                "frames — auto-resetting buffer and smoother.",
                self._no_detection_streak,
                extra={"stage": "inference"},
            )
            self.reset()
            return None

        if not self._frame_buffer.is_ready():
            return None

        raw_sequence = self._frame_buffer.get_array()
        # pad_or_truncate inside the pipeline is a guaranteed no-op here
        # (T_raw == seq_len exactly); wrist normalisation and z-clipping
        # still run, as they must on every call regardless of length.
        features_2d = self._pipeline(raw_sequence, training=False)
        features_batched = features_2d[np.newaxis, ...].astype(np.float32)

        raw_probs, elapsed_ms = self._run_single(features_batched)
        raw_confidence = float(raw_probs.max())
        raw_class_idx = int(np.argmax(raw_probs))

        predicted_class, smoothed_probs, is_stable = self._smoother.update(raw_probs)
        display_confidence = float(smoothed_probs[predicted_class])

        result = self._build_result(
            predicted_class=predicted_class,
            smoothed_probs=smoothed_probs,
            display_confidence=display_confidence,
            raw_confidence=raw_confidence,
            raw_class_idx=raw_class_idx,
            is_stable=is_stable,
            n_frames_input=self._seq_len,
            inference_latency_ms=elapsed_ms,
        )
        result["frames_in_buffer"] = self._frame_buffer.frames_accumulated()
        return result

    # ══════════════════════════════════════════════════════════════════════
    # Result assembly
    # ══════════════════════════════════════════════════════════════════════

    def _build_result(
        self,
        predicted_class: int,
        smoothed_probs: np.ndarray,
        display_confidence: float,
        raw_confidence: float,
        raw_class_idx: int,
        is_stable: bool,
        n_frames_input: int,
        inference_latency_ms: float,
    ) -> PredictionResult:
        sign_name = self._label_map.get_name_safe(predicted_class, f"class_{predicted_class}")
        top_k_raw = self._smoother.top_k(smoothed_probs, k=self._n_top_k)
        top_k: List[TopKEntry] = [
            {
                "sign": self._label_map.get_name_safe(e["class_idx"], f"class_{e['class_idx']}"),
                "class_idx": e["class_idx"],
                "confidence": round(e["confidence"], 4),
            }
            for e in top_k_raw
        ]

        result: PredictionResult = {
            "sign": sign_name,
            "confidence": round(display_confidence, 4),
            "is_confident": display_confidence >= self._display_threshold,
            "class_idx": predicted_class,
            "top_k": top_k,
            "raw_confidence": round(raw_confidence, 4),
            "raw_class_idx": raw_class_idx,
            "is_stable": is_stable,
            "n_frames_input": n_frames_input,
            "inference_latency_ms": round(inference_latency_ms, 3),
        }
        if self._flag_high_risk:
            result["is_high_risk_class"] = sign_name in HIGH_RISK_SIGNS
        return result

    # ══════════════════════════════════════════════════════════════════════
    # MediaPipe extractor (lazy)
    # ══════════════════════════════════════════════════════════════════════

    def _ensure_extractor(self) -> None:
        """Lazily construct the MediaPipe LandmarkExtractor on first use."""
        if self._extractor is None:
            from src.features.extractor import LandmarkExtractor

            self._extractor = LandmarkExtractor()
            logger.debug(
                "MediaPipe LandmarkExtractor initialised on first video/webcam call.",
                extra={"stage": "inference"},
            )

    # ══════════════════════════════════════════════════════════════════════
    # State management
    # ══════════════════════════════════════════════════════════════════════

    def reset(self) -> None:
        """Clear the rolling frame buffer, the smoother, and the no-detection streak."""
        self._frame_buffer.reset()
        self._smoother.reset()
        self._no_detection_streak = 0
        logger.debug("GesturePredictor state reset.", extra={"stage": "inference"})

    def close(self) -> None:
        """Release the MediaPipe extractor's resources, if one was created."""
        if self._extractor is not None:
            close_fn = getattr(self._extractor, "close", None)
            if callable(close_fn):
                close_fn()
            self._extractor = None
            logger.debug("GesturePredictor: MediaPipe extractor closed.", extra={"stage": "inference"})

    def __enter__(self) -> "GesturePredictor":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    # ══════════════════════════════════════════════════════════════════════
    # Properties
    # ══════════════════════════════════════════════════════════════════════

    @property
    def model_type(self) -> str:
        """``"tflite"`` or ``"keras"``."""
        return self._model_type

    @property
    def sequence_length(self) -> int:
        return self._seq_len

    @property
    def feature_dim(self) -> int:
        return self._pipeline.feature_dim

    @property
    def landmark_config(self) -> str:
        return self._pipeline.landmark_config

    @property
    def n_classes(self) -> int:
        return self._n_classes

    @property
    def display_threshold(self) -> float:
        return self._display_threshold

    @property
    def is_buffer_ready(self) -> bool:
        return self._frame_buffer.is_ready()

    @property
    def frames_buffered(self) -> int:
        return self._frame_buffer.frames_accumulated()

    @property
    def consecutive_no_detection_frames(self) -> int:
        return self._no_detection_streak

    @property
    def label_map(self) -> LabelMap:
        return self._label_map

    @property
    def pipeline(self) -> FeaturePipeline:
        return self._pipeline

    def get_metadata(self) -> Dict[str, Any]:
        """
        Complete, JSON-serialisable description of this predictor instance.

        Suitable for logging alongside webcam-demo session recordings or
        embedding in a Stage 11 model-card appendix describing the exact
        deployed inference configuration (distinct from
        ``gesture_model_metadata.json``'s training-time record).
        """
        return {
            "model_type": self._model_type,
            "model_path": str(self._model_path),
            "sequence_length": self._seq_len,
            "landmark_config": self._pipeline.landmark_config,
            "feature_dim": self._pipeline.feature_dim,
            "n_classes": self._n_classes,
            "label_map_version": self._label_map.version,
            "display_threshold": self._display_threshold,
            "smoother_window": self._smoother.window,
            "n_top_k": self._n_top_k,
            "flag_high_risk_classes": self._flag_high_risk,
            "high_risk_signs": list(HIGH_RISK_SIGNS),
            "auto_reset_no_detection_frames": self._auto_reset_threshold,
            "config_hash": str(getattr(self._config, "config_hash", "unknown")),
            "pipeline_metadata": self._pipeline.get_pipeline_metadata(),
        }

    def __repr__(self) -> str:
        return (
            f"GesturePredictor(model_type={self._model_type!r}, "
            f"seq_len={self._seq_len}, landmark_config={self._pipeline.landmark_config!r}, "
            f"n_classes={self._n_classes}, display_threshold={self._display_threshold}, "
            f"smoother_window={self._smoother.window}, "
            f"buffer={self._frame_buffer.frames_accumulated()}/{self._seq_len})"
        )


# ---------------------------------------------------------------------------
# Import-time self-check
# ---------------------------------------------------------------------------

def _self_check() -> None:
    """Cheap, dependency-free sanity check on module constants."""
    assert DEFAULT_SMOOTHER_WINDOW >= 1
    assert 0.0 < DEFAULT_SMOOTHING_ALPHA <= 1.0
    assert 0.0 <= DEFAULT_DISPLAY_THRESHOLD <= 1.0
    assert DEFAULT_TOP_K >= 1
    assert _BYTES_PER_FLOAT32 == 4
    assert len(HIGH_RISK_SIGNS) == 5, (
        f"predictor.py: HIGH_RISK_SIGNS has {len(HIGH_RISK_SIGNS)} entries; "
        "expected the 5 Stage 5/6 high-risk classes."
    )


if __debug__:
    _self_check()


__all__ = [
    "PredictionSmoother",
    "FrameBuffer",
    "GesturePredictor",
    "PredictionResult",
    "TopKEntry",
    "DEFAULT_SMOOTHER_WINDOW",
    "DEFAULT_SMOOTHING_ALPHA",
    "DEFAULT_DISPLAY_THRESHOLD",
    "DEFAULT_TOP_K",
    "DEFAULT_AUTO_RESET_NO_DETECTION_FRAMES",
]