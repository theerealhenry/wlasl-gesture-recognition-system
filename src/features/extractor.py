"""
src/features/extractor.py
==========================
MediaPipe Holistic landmark extractor for the WLASL gesture recognition pipeline.

Overview
--------
This module implements Stage 3 of the pipeline: converting raw video clips into
structured landmark sequences that can be consumed by the downstream feature
engineering and model training stages.

Every video in ``data/splits/{train,val,test}.csv`` is processed through
MediaPipe Holistic, which returns hand and pose keypoints for every frame.
The per-frame landmarks are packed into a flat 225-element vector:

    [0  : 63 ] — left hand  (21 landmarks × x,y,z)
    [63 : 126] — right hand (21 landmarks × x,y,z)
    [126: 225] — pose       (33 landmarks × x,y,z)

The result for each clip is a NumPy array of shape ``(num_frames, 225)`` —
the clip's *actual* raw frame count, never padded. Padding/truncation to the
model's sequence length is Stage 4's responsibility (FeaturePipeline), so the
same .npy files can serve all sequence-length ablation experiments.

Missing landmark handling
--------------------------
MediaPipe sometimes fails to detect hands — especially with fast motion, partial
occlusion, unusual angles, or poor lighting. The policy is:

  PER FRAME:   If MediaPipe fails to detect a hand or pose component, zero-fill
               that component's 63 or 99 values. Zeros are used (not NaN,
               not interpolated) for simplicity and to preserve sequence length.
               The missing-detection event is recorded in the ExtractionResult.

  PER CLIP:    If more than ``max_missing_frame_pct`` (default 30%) of a clip's
               frames have zero-filled hands (both hands absent simultaneously),
               the clip is skipped and ``None`` is returned from extract_video().
               The skip is logged and recorded in the batch ExtractionStats.

This conservative skip threshold (30%) preserves most clips while discarding
those where MediaPipe is fundamentally unable to track the signer — typically
due to very poor video quality or unusual camera angles.

Storage layout
--------------
Extracted .npy files are written to:

    data/landmarks/<split>/<sign_label>/<video_id>.npy

The directory tree mirrors the split structure from Stage 1. Within each split
directory, signs are separated into sub-directories for easier per-class loading.
Output dtype is float32 throughout.

Resumability
------------
The extractor checks whether the output .npy file already exists before
processing any video. If it does, the file is verified (shape check) and
reused. This makes the full extraction run safely restartable after crashes
or interruptions — no re-processing of completed clips.

Shape verification on load: if the cached .npy has the wrong number of
columns (≠ 225), it is treated as corrupt and reprocessed from scratch.

Usage
-----
The typical usage pattern is through ``pipelines/run_landmark_extraction.py``,
not by importing directly. For notebook use or testing:

    from src.features.extractor import LandmarkExtractor
    from src.utils.config import load_config

    cfg = load_config(model='lstm', data='seq30', augmentation='none')
    extractor = LandmarkExtractor(config=cfg)

    # Process a single video (returns (N, 225) array or None if skipped)
    result = extractor.extract_video(
        video_path="data/raw/book/00123.mp4",
        output_path="data/landmarks/train/book/00123.npy",
        video_id="00123",
    )

    # Low-level: extract a single frame (for real-time inference in Stage 7)
    import cv2
    cap = cv2.VideoCapture("video.mp4")
    ret, frame = cap.read()
    frame_vec = extractor.extract_frame(frame)   # shape (225,)
    cap.release()

Inference integration
---------------------
``extract_frame()`` is the key method for Stage 7 (GesturePredictor). It
accepts a single BGR frame (H×W×3 uint8) and returns the 225-element feature
vector. The method is stateless — the MediaPipe Holistic instance is held as
a reentrant context managed internally. GesturePredictor should create one
``LandmarkExtractor`` instance and call ``extract_frame()`` per webcam frame.

Thread safety
-------------
Each ``LandmarkExtractor`` instance owns its MediaPipe Holistic context and
is NOT thread-safe. For parallel extraction, create one instance per worker
process (not thread). The batch extractor in this module uses sequential
processing because MediaPipe's GIL behaviour makes multiprocessing preferable
to multithreading, and the extraction is I/O-bound for most clips.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False

try:
    from tqdm import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

from src.utils.logger import get_logger
from src.features import (
    FEATURE_SIZE,
    N_HAND_FEATURES,
    N_POSE_FEATURES,
    N_HAND_LANDMARKS,
    N_POSE_LANDMARKS,
    N_COORDS_PER_LANDMARK,
    LEFT_HAND_SLICE,
    RIGHT_HAND_SLICE,
    POSE_SLICE,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Repository root
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Extraction constants
# ---------------------------------------------------------------------------

#: Log a progress line every N clips (at INFO level)
_LOG_INTERVAL: int = 50

#: Default path for the batch preprocessing summary JSON
_DEFAULT_SUMMARY_PATH = _REPO_ROOT / "data" / "preprocessing_summary.json"

#: Minimum frames a valid clip must have after reading (< this → skip)
_MIN_VALID_FRAMES: int = 5

#: MediaPipe model_complexity: 0=lite, 1=full, 2=heavy. 1 is the project default.
_DEFAULT_MODEL_COMPLEXITY: int = 1

#: MediaPipe confidence thresholds
_MIN_DETECTION_CONFIDENCE: float = 0.5
_MIN_TRACKING_CONFIDENCE: float = 0.5


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ExtractionResult:
    """
    Result metadata for a single clip extraction.

    Produced by ``LandmarkExtractor.extract_video()`` for every clip
    regardless of whether it was skipped, loaded from cache, or freshly
    processed. Stored in ``ExtractionStats`` for batch-level reporting.

    Attributes
    ----------
    video_id : str
        WLASL video identifier.
    sign_label : str
        Human-readable sign name.
    split : str
        "train", "val", or "test".
    output_path : str
        Absolute path to the written .npy file (empty if skipped).
    status : str
        One of "extracted", "cached", "skipped", "error".
    num_frames : int
        Number of frames in the output array (0 if skipped).
    missing_left_hand_frames : int
        Frames where left-hand landmarks were zero-filled.
    missing_right_hand_frames : int
        Frames where right-hand landmarks were zero-filled.
    missing_pose_frames : int
        Frames where pose landmarks were zero-filled.
    missing_both_hands_frames : int
        Frames where both hands were absent (used for skip decision).
    missing_pct : float
        Fraction of frames with both hands absent [0.0, 1.0].
    skip_reason : str
        If status=="skipped", the reason (e.g. "missing_rate_exceeded").
    processing_time_sec : float
        Wall-clock seconds spent on this clip (0.0 for cached).
    error_message : str
        If status=="error", the exception message.
    """
    video_id: str
    sign_label: str
    split: str
    output_path: str = ""
    status: str = "extracted"          # extracted | cached | skipped | error
    num_frames: int = 0
    missing_left_hand_frames: int = 0
    missing_right_hand_frames: int = 0
    missing_pose_frames: int = 0
    missing_both_hands_frames: int = 0
    missing_pct: float = 0.0
    skip_reason: str = ""
    processing_time_sec: float = 0.0
    error_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExtractionStats:
    """
    Aggregate statistics for a batch extraction run.

    Written to ``data/preprocessing_summary.json`` after the run completes.
    Provides the numbers needed for the Stage 2 missing-landmark analysis
    notebook without requiring a second pass over all .npy files.

    Attributes
    ----------
    run_id : str
        ISO 8601 timestamp for this run.
    split : str
        Which split was processed ("train", "val", "test", "all").
    total_clips : int
        Number of clips submitted for processing.
    extracted : int
        Clips freshly extracted in this run.
    cached : int
        Clips skipped because .npy already existed and was valid.
    skipped : int
        Clips skipped due to missing-rate or video-read failures.
    errors : int
        Clips that raised unexpected exceptions.
    total_frames_extracted : int
        Sum of frame counts across all successfully processed clips.
    mean_frames_per_clip : float
        Average frame count (extracted + cached clips only).
    mean_missing_pct : float
        Average both-hands-absent rate across all processed clips.
    max_missing_pct : float
        Worst-case missing rate observed.
    per_clip_results : list[ExtractionResult]
        Full per-clip metadata. Written to JSON for Notebook 02.
    elapsed_sec : float
        Total wall-clock time for the batch run.
    """
    run_id: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    split: str = "all"
    total_clips: int = 0
    extracted: int = 0
    cached: int = 0
    skipped: int = 0
    errors: int = 0
    total_frames_extracted: int = 0
    mean_frames_per_clip: float = 0.0
    mean_missing_pct: float = 0.0
    max_missing_pct: float = 0.0
    per_clip_results: list[ExtractionResult] = field(default_factory=list)
    elapsed_sec: float = 0.0

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def success_count(self) -> int:
        """Clips that are usable (extracted + cached)."""
        return self.extracted + self.cached

    @property
    def skip_rate(self) -> float:
        """Fraction of submitted clips that were skipped."""
        return self.skipped / self.total_clips if self.total_clips > 0 else 0.0

    @property
    def error_rate(self) -> float:
        return self.errors / self.total_clips if self.total_clips > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["success_count"] = self.success_count
        d["skip_rate"] = round(self.skip_rate, 4)
        d["error_rate"] = round(self.error_rate, 4)
        return d

    def save(self, output_path: str | Path) -> Path:
        """Write the stats to a JSON file, merging with any existing data."""
        path = Path(output_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        # Merge with existing summary (multiple partial runs append to one file)
        existing_runs: list[dict[str, Any]] = []
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    existing = json.load(f)
                if isinstance(existing, list):
                    existing_runs = existing
                elif isinstance(existing, dict):
                    # Legacy single-run format — wrap it
                    existing_runs = [existing]
            except (json.JSONDecodeError, OSError):
                logger.warning(
                    f"Could not read existing summary at {path}. Overwriting.",
                    extra={"stage": "extraction"},
                )

        existing_runs.append(self.to_dict())

        with open(path, "w", encoding="utf-8") as f:
            json.dump(existing_runs, f, indent=2, default=str)

        logger.info(
            f"Extraction summary saved: {path} | "
            f"extracted={self.extracted} | "
            f"cached={self.cached} | "
            f"skipped={self.skipped} | "
            f"errors={self.errors} | "
            f"elapsed={self.elapsed_sec:.1f}s",
            extra={"stage": "extraction"},
        )
        return path

    def print_summary(self) -> None:
        """Log a human-readable summary at INFO level."""
        logger.info("=" * 65, extra={"stage": "extraction"})
        logger.info("EXTRACTION SUMMARY", extra={"stage": "extraction"})
        logger.info("=" * 65, extra={"stage": "extraction"})
        logger.info(
            f"  Total submitted  : {self.total_clips}",
            extra={"stage": "extraction"},
        )
        logger.info(
            f"  Freshly extracted: {self.extracted}",
            extra={"stage": "extraction"},
        )
        logger.info(
            f"  Loaded from cache: {self.cached}",
            extra={"stage": "extraction"},
        )
        logger.info(
            f"  Skipped (policy) : {self.skipped}  "
            f"(skip_rate={self.skip_rate:.1%})",
            extra={"stage": "extraction"},
        )
        logger.info(
            f"  Errors           : {self.errors}",
            extra={"stage": "extraction"},
        )
        logger.info(
            f"  Usable clips     : {self.success_count}",
            extra={"stage": "extraction"},
        )
        if self.success_count > 0:
            logger.info(
                f"  Mean frames/clip : {self.mean_frames_per_clip:.1f}",
                extra={"stage": "extraction"},
            )
            logger.info(
                f"  Mean missing %   : {self.mean_missing_pct:.1%}  "
                f"(max={self.max_missing_pct:.1%})",
                extra={"stage": "extraction"},
            )
        logger.info(
            f"  Elapsed          : {self.elapsed_sec:.1f}s",
            extra={"stage": "extraction"},
        )
        logger.info("=" * 65, extra={"stage": "extraction"})


# ---------------------------------------------------------------------------
# LandmarkExtractor
# ---------------------------------------------------------------------------

class LandmarkExtractor:
    """
    Extracts MediaPipe Holistic landmarks from WLASL video clips.

    This class is the single component responsible for converting raw video
    pixels into structured skeletal representations. It operates at two
    granularities:

    - **Batch mode** (``extract_dataset()``): processes all clips listed in
      the split CSVs, writing .npy files to ``data/landmarks/``. Used by
      ``pipelines/run_landmark_extraction.py``.

    - **Single-clip mode** (``extract_video()``): processes one video file.
      Used for sample runs and testing.

    - **Single-frame mode** (``extract_frame()``): processes one BGR frame
      (H×W×3 uint8). This is the method called by ``GesturePredictor``
      (Stage 7) at inference time. Stateless — can be called repeatedly on
      consecutive frames.

    Parameters
    ----------
    config : omegaconf.DictConfig | None
        Project config loaded via ``load_config()``. If None, sensible defaults
        are used (max_missing_frame_pct=0.30, model_complexity=1). Recommended
        to always pass the config in production.
    landmarks_dir : str | Path | None
        Root directory for extracted .npy files. Defaults to
        ``<repo_root>/data/landmarks``. Overridden by ``output_path`` in
        ``extract_video()`` if supplied explicitly.
    model_complexity : int
        MediaPipe Holistic model complexity (0=lite, 1=full, 2=heavy).
        Default 1. Must match the value used during inference.
    min_detection_confidence : float
        MediaPipe minimum detection confidence. Default 0.5.
    min_tracking_confidence : float
        MediaPipe minimum tracking confidence. Default 0.5.

    Raises
    ------
    ImportError
        If mediapipe or opencv-python is not installed.
    """

    def __init__(
        self,
        config=None,
        landmarks_dir: Optional[str | Path] = None,
        model_complexity: int = _DEFAULT_MODEL_COMPLEXITY,
        min_detection_confidence: float = _MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = _MIN_TRACKING_CONFIDENCE,
    ) -> None:
        if not _CV2_AVAILABLE:
            raise ImportError(
                "opencv-python is required for LandmarkExtractor. "
                "Install with: pip install opencv-python==4.8.1.78"
            )
        if not _MP_AVAILABLE:
            raise ImportError(
                "mediapipe is required for LandmarkExtractor. "
                "Install with: pip install mediapipe==0.10.14"
            )

        self._config = config
        self._landmarks_dir = Path(landmarks_dir) if landmarks_dir else (
            _REPO_ROOT / "data" / "landmarks"
        )

        # Extract relevant config values with safe defaults
        self._max_missing_pct: float = (
            config.data.max_missing_frame_pct
            if config is not None and hasattr(config, "data")
            and hasattr(config.data, "max_missing_frame_pct")
            else 0.30
        )

        # MediaPipe initialisation parameters
        self._model_complexity = model_complexity
        self._min_detection_confidence = min_detection_confidence
        self._min_tracking_confidence = min_tracking_confidence

        # Internal MediaPipe Holistic instance — lazily initialised on first use.
        # This avoids importing MediaPipe at module level (fast imports) and allows
        # the extractor to be constructed in contexts where MediaPipe may not
        # need to run (e.g. loading cached .npy files in a notebook).
        self._holistic: Optional[Any] = None
        self._mp_holistic_module: Optional[Any] = None
        self._mp_drawing: Optional[Any] = None

        logger.info(
            f"LandmarkExtractor initialised | "
            f"max_missing_pct={self._max_missing_pct:.0%} | "
            f"model_complexity={model_complexity} | "
            f"landmarks_dir={self._landmarks_dir}",
            extra={"stage": "extraction"},
        )

    # ------------------------------------------------------------------
    # MediaPipe lifecycle management
    # ------------------------------------------------------------------

    def _init_mediapipe(self) -> None:
        """
        Lazily initialise the MediaPipe Holistic context.

        Called automatically on the first call to ``extract_frame()`` or
        ``extract_video()``. Can also be called explicitly to warm up the
        model before the main extraction loop.

        MediaPipe's Python API uses context managers internally, but the
        ``mp.solutions.holistic.Holistic`` object is reentrant for sequential
        frame processing. We hold a single instance for the lifetime of the
        extractor and release it explicitly in ``close()``.
        """
        if self._holistic is not None:
            return  # Already initialised

        logger.info(
            "Initialising MediaPipe Holistic model...",
            extra={"stage": "extraction"},
        )
        t0 = time.time()

        self._mp_holistic_module = mp.solutions.holistic
        self._mp_drawing = mp.solutions.drawing_utils

        self._holistic = self._mp_holistic_module.Holistic(
            static_image_mode=False,
            model_complexity=self._model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,    # Not needed; saves compute
            smooth_segmentation=False,
            refine_face_landmarks=False,  # Not needed; saves compute
            min_detection_confidence=self._min_detection_confidence,
            min_tracking_confidence=self._min_tracking_confidence,
        )

        elapsed = time.time() - t0
        logger.info(
            f"MediaPipe Holistic ready | "
            f"model_complexity={self._model_complexity} | "
            f"init_time={elapsed:.2f}s",
            extra={"stage": "extraction"},
        )

    def close(self) -> None:
        """
        Release the MediaPipe Holistic context and free GPU/CPU resources.

        Call this when batch extraction is complete. Not calling it is not
        catastrophic (the GC will eventually clean up), but explicit release
        is good practice and prevents resource warnings in tests.
        """
        if self._holistic is not None:
            self._holistic.close()
            self._holistic = None
            logger.debug(
                "MediaPipe Holistic context released.",
                extra={"stage": "extraction"},
            )

    def __enter__(self) -> "LandmarkExtractor":
        """Support use as a context manager: ``with LandmarkExtractor() as ext:``"""
        self._init_mediapipe()
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Core single-frame extraction — public, reused by Stage 7
    # ------------------------------------------------------------------

    def extract_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract the 225-element landmark vector from a single BGR frame.

        This is the **primary public method for real-time inference** (Stage 7).
        ``GesturePredictor`` calls this on each webcam frame. The method is
        stateless in the sense that output depends only on the input frame,
        but note that MediaPipe's tracking state is maintained internally —
        for optimal tracking, frames should be passed in sequential order.

        The feature vector layout is:
            [0  :63 ] left hand  — 21 landmarks × (x, y, z)
            [63 :126] right hand — 21 landmarks × (x, y, z)
            [126:225] pose       — 33 landmarks × (x, y, z)

        When MediaPipe fails to detect a component (e.g. left hand not visible),
        the corresponding slice is zero-filled. Zeros are semantically
        distinguishable from valid detections (valid landmarks cluster around
        [0, 1] normalised screen space; true zero is rare but possible).

        Parameters
        ----------
        frame : np.ndarray
            BGR uint8 image array (H × W × 3). Standard OpenCV format.
            Must not be None or empty.

        Returns
        -------
        np.ndarray
            Shape ``(225,)`` float32. Zero-filled components where MediaPipe
            failed to detect.

        Raises
        ------
        ValueError
            If frame is None, empty, or not a 3-channel image.
        RuntimeError
            If MediaPipe has not been initialised (call _init_mediapipe() or
            use the extractor as a context manager before calling this method
            from non-batch code paths).
        """
        if frame is None or frame.size == 0:
            raise ValueError(
                "extract_frame() received an empty or None frame. "
                "Ensure the video capture is reading valid frames."
            )
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"extract_frame() expects a 3-channel BGR image (H×W×3), "
                f"got shape {frame.shape}."
            )

        if self._holistic is None:
            self._init_mediapipe()

        # Convert BGR → RGB (MediaPipe expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Mark as non-writeable for zero-copy MediaPipe processing
        rgb_frame.flags.writeable = False
        results = self._holistic.process(rgb_frame)
        rgb_frame.flags.writeable = True

        # Pack the 225-element feature vector
        feature_vec = np.zeros(FEATURE_SIZE, dtype=np.float32)

        # --- Left hand [0:63] ---
        if results.left_hand_landmarks:
            feature_vec[LEFT_HAND_SLICE] = self._pack_hand_landmarks(
                results.left_hand_landmarks
            )

        # --- Right hand [63:126] ---
        if results.right_hand_landmarks:
            feature_vec[RIGHT_HAND_SLICE] = self._pack_hand_landmarks(
                results.right_hand_landmarks
            )

        # --- Pose [126:225] ---
        if results.pose_landmarks:
            feature_vec[POSE_SLICE] = self._pack_pose_landmarks(
                results.pose_landmarks
            )

        return feature_vec

    # ------------------------------------------------------------------
    # Detection status helpers (used by extract_video for skip logic)
    # ------------------------------------------------------------------

    def _extract_frame_with_status(
        self, frame: np.ndarray
    ) -> tuple[np.ndarray, bool, bool, bool]:
        """
        Extract landmarks and return per-component detection flags.

        Returns
        -------
        tuple[np.ndarray, bool, bool, bool]
            (feature_vec, left_detected, right_detected, pose_detected)
        """
        if frame is None or frame.size == 0:
            return (
                np.zeros(FEATURE_SIZE, dtype=np.float32),
                False, False, False,
            )

        if self._holistic is None:
            self._init_mediapipe()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self._holistic.process(rgb_frame)
        rgb_frame.flags.writeable = True

        feature_vec = np.zeros(FEATURE_SIZE, dtype=np.float32)

        left_detected = results.left_hand_landmarks is not None
        right_detected = results.right_hand_landmarks is not None
        pose_detected = results.pose_landmarks is not None

        if left_detected:
            feature_vec[LEFT_HAND_SLICE] = self._pack_hand_landmarks(
                results.left_hand_landmarks
            )
        if right_detected:
            feature_vec[RIGHT_HAND_SLICE] = self._pack_hand_landmarks(
                results.right_hand_landmarks
            )
        if pose_detected:
            feature_vec[POSE_SLICE] = self._pack_pose_landmarks(
                results.pose_landmarks
            )

        return feature_vec, left_detected, right_detected, pose_detected

    # ------------------------------------------------------------------
    # Landmark packing helpers
    # ------------------------------------------------------------------

    def _pack_hand_landmarks(self, hand_landmarks) -> np.ndarray:
        """
        Flatten 21 MediaPipe hand landmarks into a (63,) float32 array.

        The layout is [x0, y0, z0, x1, y1, z1, ..., x20, y20, z20] where
        landmark 0 is the wrist (WRIST_LANDMARK_INDEX=0). This ordering is
        used by FeaturePipeline for wrist-relative normalisation:
        the wrist position is at indices [0:3] of the 63-element hand vector.

        Coordinates are MediaPipe's normalised screen coordinates:
        x and y ∈ [0, 1] (relative to frame dimensions);
        z ∈ approximately [-0.2, 0.2] (depth, smaller = closer to camera).

        Parameters
        ----------
        hand_landmarks : mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList
            MediaPipe hand landmarks (21 points).

        Returns
        -------
        np.ndarray
            Shape (63,) float32.
        """
        vec = np.empty(N_HAND_FEATURES, dtype=np.float32)
        for i, lm in enumerate(hand_landmarks.landmark):
            base = i * N_COORDS_PER_LANDMARK
            vec[base]     = lm.x
            vec[base + 1] = lm.y
            vec[base + 2] = lm.z
        return vec

    def _pack_pose_landmarks(self, pose_landmarks) -> np.ndarray:
        """
        Flatten 33 MediaPipe pose landmarks into a (99,) float32 array.

        MediaPipe Holistic's pose model (BlazePose) provides 33 body
        landmarks: nose, eyes, ears, shoulders, elbows, wrists, hips,
        knees, ankles, and feet. Visibility scores are NOT included in
        the feature vector to keep the representation consistent with
        the 225-value specification.

        Layout: [x0, y0, z0, x1, y1, z1, ..., x32, y32, z32]

        Parameters
        ----------
        pose_landmarks : mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList
            MediaPipe pose landmarks (33 points).

        Returns
        -------
        np.ndarray
            Shape (99,) float32.
        """
        vec = np.empty(N_POSE_FEATURES, dtype=np.float32)
        for i, lm in enumerate(pose_landmarks.landmark):
            base = i * N_COORDS_PER_LANDMARK
            vec[base]     = lm.x
            vec[base + 1] = lm.y
            vec[base + 2] = lm.z
        return vec

    # ------------------------------------------------------------------
    # Single-clip extraction
    # ------------------------------------------------------------------

    def extract_video(
        self,
        video_path: str | Path,
        output_path: str | Path,
        video_id: str = "",
        sign_label: str = "",
        split: str = "",
        force: bool = False,
    ) -> Optional[ExtractionResult]:
        """
        Extract landmarks from one video clip and save to a .npy file.

        This is the workhorse method called by the batch extractor for every
        clip in the dataset. It handles the full lifecycle:

        1. Check for cached .npy (skip if valid, force=False)
        2. Open the video with OpenCV
        3. Process each frame through MediaPipe Holistic
        4. Apply the missing-landmark skip policy
        5. Write the (num_frames, 225) array to disk
        6. Return an ExtractionResult with full statistics

        The output array shape is ``(num_frames, 225)`` where ``num_frames``
        is the raw clip length. It is NOT padded to ``seq_len=30``.

        Parameters
        ----------
        video_path : str | Path
            Path to the input video file. Can be relative to repo root or
            absolute.
        output_path : str | Path
            Path for the output .npy file. Parent directory created if needed.
        video_id : str
            WLASL video identifier (for logging and ExtractionResult).
        sign_label : str
            Sign name (for logging and ExtractionResult).
        split : str
            Split name — "train", "val", or "test" (for ExtractionResult).
        force : bool
            If True, reprocess even if a valid .npy already exists.

        Returns
        -------
        ExtractionResult
            Full statistics for this clip. ``result.status`` is one of:
            "extracted", "cached", "skipped", "error".

        Notes
        -----
        This method does NOT raise exceptions on per-clip failures — it
        catches all exceptions, logs them, and returns an ExtractionResult
        with status="error". This allows the batch extractor to continue
        past individual corrupt clips.
        """
        t0 = time.time()
        output_path = Path(output_path).resolve()

        # ----------------------------------------------------------------
        # Resolve video path — support both relative and absolute paths
        # ----------------------------------------------------------------
        vp = Path(video_path)
        if not vp.is_absolute():
            vp = _REPO_ROOT / vp
        vp = vp.resolve()

        # ----------------------------------------------------------------
        # Resumability check: load cached .npy if valid
        # ----------------------------------------------------------------
        if not force and output_path.exists():
            cached_result = self._try_load_cached(
                output_path, video_id, sign_label, split
            )
            if cached_result is not None:
                cached_result.processing_time_sec = round(time.time() - t0, 3)
                return cached_result

        # ----------------------------------------------------------------
        # Validate video file exists
        # ----------------------------------------------------------------
        if not vp.exists():
            logger.warning(
                f"Video file not found: {vp} | video_id={video_id}",
                extra={"stage": "extraction", "video_id": video_id},
            )
            return ExtractionResult(
                video_id=video_id,
                sign_label=sign_label,
                split=split,
                status="error",
                error_message=f"File not found: {vp}",
                processing_time_sec=round(time.time() - t0, 3),
            )

        # ----------------------------------------------------------------
        # Open video and extract landmarks frame by frame
        # ----------------------------------------------------------------
        try:
            landmarks_list, detection_flags = self._process_video_frames(
                vp, video_id
            )
        except Exception as exc:
            logger.error(
                f"Error processing video {video_id} ({sign_label}): "
                f"{type(exc).__name__}: {exc}",
                extra={"stage": "extraction", "video_id": video_id},
            )
            return ExtractionResult(
                video_id=video_id,
                sign_label=sign_label,
                split=split,
                status="error",
                error_message=f"{type(exc).__name__}: {exc}",
                processing_time_sec=round(time.time() - t0, 3),
            )

        if not landmarks_list:
            logger.warning(
                f"No frames extracted from {video_id} ({sign_label}). "
                "Video may be corrupt or unreadable.",
                extra={"stage": "extraction", "video_id": video_id},
            )
            return ExtractionResult(
                video_id=video_id,
                sign_label=sign_label,
                split=split,
                status="skipped",
                skip_reason="no_frames_extracted",
                processing_time_sec=round(time.time() - t0, 3),
            )

        # ----------------------------------------------------------------
        # Compute per-frame detection statistics
        # ----------------------------------------------------------------
        n_frames = len(landmarks_list)
        left_flags, right_flags, pose_flags = zip(*detection_flags)

        missing_left   = sum(1 for f in left_flags  if not f)
        missing_right  = sum(1 for f in right_flags if not f)
        missing_pose   = sum(1 for f in pose_flags  if not f)
        # "both hands absent" — the condition used for the skip threshold
        missing_both   = sum(
            1 for l, r in zip(left_flags, right_flags) if not l and not r
        )
        missing_pct    = missing_both / n_frames if n_frames > 0 else 0.0

        # ----------------------------------------------------------------
        # Skip policy: too many frames with no hand detection
        # ----------------------------------------------------------------
        if missing_pct > self._max_missing_pct:
            logger.info(
                f"Skipping {video_id} ({sign_label}): "
                f"{missing_pct:.1%} frames missing both hands "
                f"(threshold={self._max_missing_pct:.0%})",
                extra={"stage": "extraction", "video_id": video_id},
            )
            return ExtractionResult(
                video_id=video_id,
                sign_label=sign_label,
                split=split,
                status="skipped",
                num_frames=n_frames,
                missing_left_hand_frames=missing_left,
                missing_right_hand_frames=missing_right,
                missing_pose_frames=missing_pose,
                missing_both_hands_frames=missing_both,
                missing_pct=round(missing_pct, 4),
                skip_reason="missing_rate_exceeded",
                processing_time_sec=round(time.time() - t0, 3),
            )

        # ----------------------------------------------------------------
        # Stack into array and write to disk
        # ----------------------------------------------------------------
        landmarks_array = np.stack(landmarks_list, axis=0).astype(np.float32)
        # landmarks_array.shape: (num_frames, 225)

        if landmarks_array.shape[1] != FEATURE_SIZE:
            # Should never happen if _pack_* methods are correct.
            raise RuntimeError(
                f"Feature size mismatch: expected {FEATURE_SIZE}, "
                f"got {landmarks_array.shape[1]} for video_id={video_id}. "
                "This is a bug in the landmark packing code."
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(output_path), landmarks_array)

        processing_time = round(time.time() - t0, 3)
        logger.debug(
            f"Extracted {video_id} ({sign_label}) | "
            f"shape={landmarks_array.shape} | "
            f"missing_left={missing_left}/{n_frames} | "
            f"missing_right={missing_right}/{n_frames} | "
            f"missing_both={missing_both}/{n_frames} ({missing_pct:.1%}) | "
            f"time={processing_time:.2f}s",
            extra={"stage": "extraction", "video_id": video_id},
        )

        return ExtractionResult(
            video_id=video_id,
            sign_label=sign_label,
            split=split,
            output_path=str(output_path),
            status="extracted",
            num_frames=n_frames,
            missing_left_hand_frames=missing_left,
            missing_right_hand_frames=missing_right,
            missing_pose_frames=missing_pose,
            missing_both_hands_frames=missing_both,
            missing_pct=round(missing_pct, 4),
            processing_time_sec=processing_time,
        )

    # ------------------------------------------------------------------
    # Frame-level video processing
    # ------------------------------------------------------------------

    def _process_video_frames(
        self,
        video_path: Path,
        video_id: str,
    ) -> tuple[list[np.ndarray], list[tuple[bool, bool, bool]]]:
        """
        Open a video file and extract landmark vectors for every frame.

        Uses OpenCV to read frames sequentially. MediaPipe processes each
        frame. Returns two parallel lists:
        - ``landmarks_list``: per-frame (225,) feature vectors
        - ``detection_flags``: per-frame (left_detected, right_detected, pose_detected)

        The ordering of landmark vectors matches the ordering used by
        ``_pack_hand_landmarks`` and ``_pack_pose_landmarks``:
        LEFT_HAND first, then RIGHT_HAND, then POSE.

        Parameters
        ----------
        video_path : Path
            Absolute path to the video file.
        video_id : str
            For debug logging only.

        Returns
        -------
        tuple[list[np.ndarray], list[tuple[bool, bool, bool]]]
            (landmarks_list, detection_flags)

        Raises
        ------
        RuntimeError
            If OpenCV cannot open the file (corrupt video or wrong codec).
        """
        if self._holistic is None:
            self._init_mediapipe()

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(
                f"OpenCV could not open video: {video_path}. "
                "The file may be corrupt or encoded in an unsupported codec."
            )

        landmarks_list: list[np.ndarray] = []
        detection_flags: list[tuple[bool, bool, bool]] = []

        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame is None or frame.size == 0:
                    logger.debug(
                        f"Null frame at index {frame_idx} in {video_id}. "
                        "Zero-filling.",
                        extra={"stage": "extraction"},
                    )
                    landmarks_list.append(np.zeros(FEATURE_SIZE, dtype=np.float32))
                    detection_flags.append((False, False, False))
                    frame_idx += 1
                    continue

                feat_vec, left_det, right_det, pose_det = (
                    self._extract_frame_with_status(frame)
                )
                landmarks_list.append(feat_vec)
                detection_flags.append((left_det, right_det, pose_det))
                frame_idx += 1

        finally:
            cap.release()

        if len(landmarks_list) < _MIN_VALID_FRAMES:
            logger.warning(
                f"Only {len(landmarks_list)} frames extracted from {video_id}. "
                f"Minimum required: {_MIN_VALID_FRAMES}. "
                "Treating as empty (will be skipped).",
                extra={"stage": "extraction", "video_id": video_id},
            )
            return [], []

        return landmarks_list, detection_flags

    # ------------------------------------------------------------------
    # Cache management helpers
    # ------------------------------------------------------------------

    def _try_load_cached(
        self,
        output_path: Path,
        video_id: str,
        sign_label: str,
        split: str,
    ) -> Optional[ExtractionResult]:
        """
        Attempt to validate and return metadata for a cached .npy file.

        Performs a shape verification (ndim==2, shape[1]==225) before
        treating the cache as valid. An incorrect number of columns
        indicates the file was created with a different feature specification
        and should be reprocessed.

        Parameters
        ----------
        output_path : Path
            Path to the existing .npy file.
        video_id, sign_label, split : str
            Metadata for the ExtractionResult.

        Returns
        -------
        ExtractionResult | None
            ExtractionResult with status="cached" if valid.
            None if the file is invalid/corrupt (triggers reprocessing).
        """
        try:
            # Load only the array header (fast) to check shape
            arr = np.load(str(output_path), mmap_mode="r")

            if arr.ndim != 2 or arr.shape[1] != FEATURE_SIZE:
                logger.warning(
                    f"Cached .npy for {video_id} has unexpected shape "
                    f"{arr.shape} (expected (N, {FEATURE_SIZE})). "
                    "Reprocessing.",
                    extra={"stage": "extraction", "video_id": video_id},
                )
                return None

            n_frames = arr.shape[0]
            logger.debug(
                f"Using cached: {video_id} | shape={arr.shape}",
                extra={"stage": "extraction", "video_id": video_id},
            )
            return ExtractionResult(
                video_id=video_id,
                sign_label=sign_label,
                split=split,
                output_path=str(output_path),
                status="cached",
                num_frames=n_frames,
            )

        except Exception as exc:
            logger.warning(
                f"Could not read cached .npy for {video_id}: {exc}. "
                "Reprocessing.",
                extra={"stage": "extraction", "video_id": video_id},
            )
            return None

    # ------------------------------------------------------------------
    # Batch extraction — processes the full split CSVs
    # ------------------------------------------------------------------

    def extract_dataset(
        self,
        split_csv_paths: dict[str, str | Path],
        force: bool = False,
        sample_only: bool = False,
        summary_path: Optional[str | Path] = None,
    ) -> ExtractionStats:
        """
        Extract landmarks for all clips listed in the split CSV files.

        This is the top-level method called by ``run_landmark_extraction.py``.
        It iterates over all clips in the provided splits, calls
        ``extract_video()`` for each, accumulates statistics, and writes
        ``data/preprocessing_summary.json``.

        Parameters
        ----------
        split_csv_paths : dict[str, str | Path]
            Mapping of split_name → CSV path. Keys must be from
            {"train", "val", "test"}. Pass only the splits to process;
            e.g. ``{"train": "data/splits/train.csv"}`` to process only train.
        force : bool
            If True, reprocess all clips even if .npy files exist.
        sample_only : bool
            If True, process exactly one clip per sign per split. Produces
            35 clips per split for quick Stage 2 validation. The selected
            clip is the first one alphabetically by video_id for each sign.
        summary_path : str | Path | None
            Path for the JSON summary output. Defaults to
            ``data/preprocessing_summary.json``.

        Returns
        -------
        ExtractionStats
            Aggregate statistics for the completed run.

        Notes
        -----
        MediaPipe is initialised once at the start of this method and
        released at the end. Do not call ``close()`` manually between calls
        to ``extract_dataset()`` if processing multiple splits.
        """
        import pandas as pd

        if not split_csv_paths:
            raise ValueError(
                "split_csv_paths must not be empty. "
                "Provide at least one split CSV path."
            )

        split_name_repr = "+".join(sorted(split_csv_paths.keys()))
        stats = ExtractionStats(split=split_name_repr)
        batch_start = time.time()

        # Warm up MediaPipe before the main loop (avoids a cold-start
        # penalty on the first clip's timing measurement)
        self._init_mediapipe()

        all_results: list[ExtractionResult] = []

        for split_name, csv_path in sorted(split_csv_paths.items()):
            csv_path = Path(csv_path)
            if not csv_path.exists():
                logger.error(
                    f"Split CSV not found: {csv_path}. Skipping {split_name}.",
                    extra={"stage": "extraction"},
                )
                continue

            logger.info(
                f"Processing split: {split_name} | csv={csv_path}",
                extra={"stage": "extraction"},
            )

            try:
                import pandas as pd
                df = pd.read_csv(csv_path)
            except Exception as exc:
                logger.error(
                    f"Could not read CSV {csv_path}: {exc}",
                    extra={"stage": "extraction"},
                )
                continue

            # Validate required columns
            required_cols = {"video_id", "sign_label", "video_path"}
            missing_cols = required_cols - set(df.columns)
            if missing_cols:
                logger.error(
                    f"Split CSV {csv_path} is missing required columns: "
                    f"{missing_cols}. Skipping {split_name}.",
                    extra={"stage": "extraction"},
                )
                continue

            # Sample mode: keep one clip per sign (first alphabetically by video_id)
            if sample_only:
                df = (
                    df.sort_values("video_id")
                    .groupby("sign_label", sort=False)
                    .first()
                    .reset_index()
                )
                logger.info(
                    f"Sample mode: selected {len(df)} clips "
                    f"(1 per sign) from {split_name}.",
                    extra={"stage": "extraction"},
                )

            split_results = self._process_split_df(
                df=df,
                split_name=split_name,
                force=force,
            )
            all_results.extend(split_results)

        # ----------------------------------------------------------------
        # Aggregate statistics
        # ----------------------------------------------------------------
        stats.total_clips   = len(all_results)
        stats.extracted     = sum(1 for r in all_results if r.status == "extracted")
        stats.cached        = sum(1 for r in all_results if r.status == "cached")
        stats.skipped       = sum(1 for r in all_results if r.status == "skipped")
        stats.errors        = sum(1 for r in all_results if r.status == "error")
        stats.per_clip_results = all_results

        usable = [r for r in all_results if r.status in ("extracted", "cached")]
        if usable:
            frame_counts = [r.num_frames for r in usable if r.num_frames > 0]
            missing_pcts  = [r.missing_pct for r in usable]

            stats.total_frames_extracted = sum(frame_counts)
            stats.mean_frames_per_clip   = (
                sum(frame_counts) / len(frame_counts) if frame_counts else 0.0
            )
            stats.mean_missing_pct = (
                sum(missing_pcts) / len(missing_pcts) if missing_pcts else 0.0
            )
            stats.max_missing_pct = max(missing_pcts) if missing_pcts else 0.0

        stats.elapsed_sec = round(time.time() - batch_start, 1)

        # ----------------------------------------------------------------
        # Write summary JSON and print to log
        # ----------------------------------------------------------------
        out_path = Path(summary_path) if summary_path else _DEFAULT_SUMMARY_PATH
        stats.save(out_path)
        stats.print_summary()

        return stats

    def _process_split_df(
        self,
        df,
        split_name: str,
        force: bool,
    ) -> list[ExtractionResult]:
        """
        Iterate over one split's DataFrame and extract each clip.

        Provides per-50-clip progress logging and graceful error handling.
        Each clip is processed independently so a single corrupt video does
        not abort the rest of the split.

        Parameters
        ----------
        df : pd.DataFrame
            The split DataFrame (must have video_id, sign_label, video_path).
        split_name : str
            "train", "val", or "test" (used in output path and logging).
        force : bool
            Passed through to extract_video().

        Returns
        -------
        list[ExtractionResult]
            One result per row in df.
        """
        results: list[ExtractionResult] = []
        n_clips = len(df)
        split_start = time.time()

        iterator = (
            tqdm(df.itertuples(index=False), total=n_clips, desc=f"{split_name}")
            if _TQDM_AVAILABLE
            else df.itertuples(index=False)
        )

        for i, row in enumerate(iterator):
            video_id    = str(row.video_id)
            sign_label  = str(row.sign_label)
            video_path  = str(row.video_path)
            class_idx   = getattr(row, "class_idx", -1)

            # Determine output path:
            # data/landmarks/<split>/<sign_label>/<video_id>.npy
            output_path = (
                self._landmarks_dir
                / split_name
                / sign_label
                / f"{video_id}.npy"
            )

            result = self.extract_video(
                video_path=video_path,
                output_path=output_path,
                video_id=video_id,
                sign_label=sign_label,
                split=split_name,
                force=force,
            )
            results.append(result)

            # Progress log every N clips
            if (i + 1) % _LOG_INTERVAL == 0 or (i + 1) == n_clips:
                elapsed = time.time() - split_start
                rate = (i + 1) / elapsed if elapsed > 0 else 0.0
                eta = (n_clips - (i + 1)) / rate if rate > 0 else 0.0

                n_done     = sum(1 for r in results if r.status in ("extracted", "cached"))
                n_skipped  = sum(1 for r in results if r.status == "skipped")
                n_errors   = sum(1 for r in results if r.status == "error")

                logger.info(
                    f"{split_name} | "
                    f"{i + 1}/{n_clips} clips | "
                    f"done={n_done} | "
                    f"skipped={n_skipped} | "
                    f"errors={n_errors} | "
                    f"rate={rate:.1f}/s | "
                    f"ETA={eta:.0f}s",
                    extra={"stage": "extraction"},
                )

        return results

    # ------------------------------------------------------------------
    # Drawing utility (for Notebook 02 visualisation)
    # ------------------------------------------------------------------

    def draw_landmarks(
        self,
        frame: np.ndarray,
        feature_vec: Optional[np.ndarray] = None,
        results=None,
    ) -> np.ndarray:
        """
        Draw MediaPipe landmarks on a frame copy for visualisation.

        Intended for use in ``notebooks/02_landmark_inspection.ipynb``.
        Two usage modes:

        1. Pass ``results`` (a raw MediaPipe Holistic result object) — uses
           MediaPipe's built-in drawing utilities directly.
        2. Pass ``feature_vec`` (a (225,) array) — reconstructs landmark
           positions from the packed vector and draws circles.

        Parameters
        ----------
        frame : np.ndarray
            BGR image. Not modified in place — a copy is returned.
        feature_vec : np.ndarray | None
            (225,) landmark vector (post-extraction, pre-normalisation).
            Used when you have the .npy data but not the raw MediaPipe result.
        results : mediapipe Holistic result | None
            Raw output from ``self._holistic.process()``. Used for direct
            rendering with MediaPipe's drawing utilities.

        Returns
        -------
        np.ndarray
            Annotated BGR frame (copy of input).
        """
        if not _MP_AVAILABLE or not _CV2_AVAILABLE:
            logger.warning(
                "draw_landmarks() requires mediapipe and opencv-python.",
                extra={"stage": "extraction"},
            )
            return frame.copy()

        annotated = frame.copy()

        if results is not None and self._mp_drawing is not None:
            mp_styles = mp.solutions.drawing_styles
            mp_hol    = mp.solutions.holistic

            self._mp_drawing.draw_landmarks(
                annotated,
                results.left_hand_landmarks,
                mp_hol.HAND_CONNECTIONS,
                self._mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                self._mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2),
            )
            self._mp_drawing.draw_landmarks(
                annotated,
                results.right_hand_landmarks,
                mp_hol.HAND_CONNECTIONS,
                self._mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                self._mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2),
            )
            self._mp_drawing.draw_landmarks(
                annotated,
                results.pose_landmarks,
                mp_hol.POSE_CONNECTIONS,
                mp_styles.get_default_pose_landmarks_style(),
            )

        elif feature_vec is not None:
            if feature_vec.shape != (FEATURE_SIZE,):
                logger.warning(
                    f"draw_landmarks(): feature_vec has shape {feature_vec.shape}, "
                    f"expected ({FEATURE_SIZE},). Skipping draw.",
                    extra={"stage": "extraction"},
                )
                return annotated

            h, w = frame.shape[:2]
            lh_vec = feature_vec[LEFT_HAND_SLICE].reshape(N_HAND_LANDMARKS, 3)
            rh_vec = feature_vec[RIGHT_HAND_SLICE].reshape(N_HAND_LANDMARKS, 3)
            pose_vec = feature_vec[POSE_SLICE].reshape(N_POSE_LANDMARKS, 3)

            # Draw left hand (blue)
            for lm in lh_vec:
                cx, cy = int(lm[0] * w), int(lm[1] * h)
                if 0 <= cx < w and 0 <= cy < h:
                    cv2.circle(annotated, (cx, cy), 4, (255, 0, 0), -1)

            # Draw right hand (orange)
            for lm in rh_vec:
                cx, cy = int(lm[0] * w), int(lm[1] * h)
                if 0 <= cx < w and 0 <= cy < h:
                    cv2.circle(annotated, (cx, cy), 4, (0, 165, 255), -1)

            # Draw pose (green, smaller)
            for lm in pose_vec:
                cx, cy = int(lm[0] * w), int(lm[1] * h)
                if 0 <= cx < w and 0 <= cy < h:
                    cv2.circle(annotated, (cx, cy), 2, (0, 255, 0), -1)

        return annotated

    # ------------------------------------------------------------------
    # Convenience: load an existing .npy file
    # ------------------------------------------------------------------

    @staticmethod
    def load_landmarks(npy_path: str | Path) -> np.ndarray:
        """
        Load a previously extracted landmark array from disk.

        Validates shape on load. Raises ValueError if the array has the
        wrong number of columns (corruption or version mismatch).

        Parameters
        ----------
        npy_path : str | Path
            Path to the .npy file.

        Returns
        -------
        np.ndarray
            Shape ``(num_frames, 225)`` float32.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the array shape is not (N, 225).
        """
        path = Path(npy_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Landmark file not found: {path}")

        arr = np.load(str(path))

        if arr.ndim != 2 or arr.shape[1] != FEATURE_SIZE:
            raise ValueError(
                f"Invalid landmark array shape at {path}: {arr.shape}. "
                f"Expected (N, {FEATURE_SIZE}). "
                "The file may be corrupt or from an incompatible extractor version."
            )

        return arr.astype(np.float32)

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        mp_status = "initialised" if self._holistic is not None else "lazy (not yet initialised)"
        return (
            f"LandmarkExtractor("
            f"model_complexity={self._model_complexity}, "
            f"max_missing_pct={self._max_missing_pct:.0%}, "
            f"mediapipe={mp_status}, "
            f"landmarks_dir='{self._landmarks_dir}')"
        )