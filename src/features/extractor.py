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
same .npy files serve all sequence-length ablation experiments unchanged.

Schema versions
---------------
- 1.0: original schema
- 1.1: added ``decode_failure_frames`` field; ``missing_pct`` now computed over
       successfully-decoded frames only (fixes inflated skip rate on codec errors)
- 1.2: replaced ratio-based skip policy with dual-criterion absolute policy;
       added ``detected_frames`` field to ExtractionResult and sidecar; added
       ``missing_one_hand_frames`` field (frames where exactly one hand absent)

Skip policy — v1.2 (dual-criterion absolute)
---------------------------------------------
Previous versions (v1.0, v1.1) used a ratio-based policy:

    skip if missing_both_pct > threshold (default 30%)

Analysis of the WLASL dataset revealed this is fundamentally wrong.
WLASL clips sourced from YouTube contain large temporal dead zones:
preparation movements, idle frames, lead-in/lead-out segments where
the signer's hands are at rest or off-screen. These inflate the ratio
without reducing the actual sign content.

Example (video 34824, sign "many"):
    69 frames, 21 both-hands-missing → ratio = 31.3%
    Old policy: REJECTED (31.3% > 30%)
    Actual detected frames: 46 → more than enough for training

The correct question is not "what fraction of frames are missing?" but
"does this clip contain enough usable frames for the LSTM to learn from?"

**v1.2 dual-criterion policy:**

    KEEP clip if ALL of the following hold:
      (a) detected_frames >= min_detected_frames  (default: 15)
              At least 15 frames where at least one hand was detected.
              This directly answers whether the clip has training value.
              15 is the floor; the ablation studies use {20,30,40,60}
              sequence lengths, so 15 usable frames is the minimum to be
              useful even at seq_len=20.
      (b) missing_pct <= max_missing_pct  (default: 0.95)
              Catastrophe filter: if 95%+ of decoded frames have both
              hands absent, the clip is truly garbage regardless of how
              many frames it has. This catches corrupt files, wrong-codec
              videos, or clips where the signer is never visible.

    SKIP clip (after attempting extraction) if either criterion fails.

This policy retains ~96% of the sample clips that the old 30%-ratio policy
rejected, recovering the training data needed to reach the ≥70% accuracy
target.

Missing landmark handling
--------------------------
MediaPipe sometimes fails to detect hands — especially with fast motion, partial
occlusion, unusual angles, or poor lighting. The policy is:

  PER FRAME (decode failure):   If OpenCV fails to read a frame (transient codec
               error), zero-fill that frame's 225 values and mark it as a decode
               failure. Do NOT count it toward the hand-detection missing rate.

  PER FRAME (detection failure): If MediaPipe fails to detect a hand or pose
               component on a successfully decoded frame, zero-fill that
               component's 63 or 99 values. Count it in ``missing_both`` if
               both hands are absent simultaneously.

  PER CLIP:    Apply the dual-criterion policy described above.

Sidecar metadata
----------------
Alongside every .npy file, a sibling .meta.json file is written:

    data/landmarks/train/book/00123.npy
    data/landmarks/train/book/00123.meta.json

The .meta.json stores the per-clip detection statistics so that cache hits
on subsequent runs can restore the full ExtractionResult without
re-processing the video.

Storage layout
--------------
Extracted .npy files are written to:

    data/landmarks/<split>/<sign_label>/<video_id>.npy

Output dtype is float32 throughout. Sidecar metadata uses the same stem:

    data/landmarks/<split>/<sign_label>/<video_id>.meta.json

Summary output
--------------
Two JSON summary files are written after each batch run:

    data/preprocessing_summary_latest.json   — current run only (always overwritten)
    data/preprocessing_summary_history.json  — append-only log of all runs

Resumability
------------
Before processing any clip the extractor checks for the .npy + .meta.json pair.
If both exist and pass validation (shape, dtype, schema version, full finiteness
check), the clip is skipped with status="cached" and statistics are restored
from the sidecar. If either file is missing or fails validation, the clip is
reprocessed.

Thread safety
-------------
Each ``LandmarkExtractor`` instance owns its MediaPipe Holistic context and
is NOT thread-safe. For parallel extraction, create one instance per worker
process. Batch extraction uses sequential processing because MediaPipe's
C++ backend is already multi-threaded within a single Holistic instance.
"""

from __future__ import annotations

import csv
import json
import re
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

# Import from the dependency-free constants module to avoid circular imports.
from src.features.constants import (
    FEATURE_SIZE,
    N_HAND_FEATURES,
    N_POSE_FEATURES,
    N_HAND_LANDMARKS,
    N_POSE_LANDMARKS,
    N_COORDS_PER_LANDMARK,
    LEFT_HAND_SLICE,
    RIGHT_HAND_SLICE,
    POSE_SLICE,
    EXTRACTOR_SCHEMA_VERSION,
    MIN_DETECTED_FRAMES_DEFAULT,
    MAX_MISSING_PCT_CATASTROPHE,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Repository root
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Log a progress line every N clips (at INFO level)
_LOG_INTERVAL: int = 50

#: Default root for per-run summary JSON files
_DEFAULT_SUMMARY_DIR = _REPO_ROOT / "data"

#: Minimum TOTAL frames a clip must have after decoding to attempt processing.
#: This is a pre-MediaPipe guard; the per-clip minimum is MIN_DETECTED_FRAMES_DEFAULT.
_MIN_VALID_FRAMES: int = 5

#: MediaPipe model_complexity: 0=lite, 1=full, 2=heavy. 1 is the project default.
_DEFAULT_MODEL_COMPLEXITY: int = 1

#: MediaPipe confidence thresholds (both detection and tracking)
_MIN_DETECTION_CONFIDENCE: float = 0.5
_MIN_TRACKING_CONFIDENCE: float = 0.5

#: Consecutive decode failures before treating as end-of-stream / corrupt file.
#: Some codecs emit occasional transient read errors that immediately recover;
#: this tolerance avoids premature termination for those cases.
_MAX_CONSECUTIVE_READ_FAILURES: int = 3

#: Characters unsafe as filesystem path components on any OS
_UNSAFE_PATH_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

#: Default number of clips per sign in sample-only mode.
_DEFAULT_SAMPLE_CLIPS_PER_SIGN: int = 3

#: Name of the landmark inventory CSV written after each batch run.
_LANDMARK_INVENTORY_FILENAME: str = "landmark_inventory.csv"

#: Inventory CSV columns (order must match _build_inventory_row).
_INVENTORY_CSV_COLUMNS = [
    "video_id",
    "sign_label",
    "split",
    "outcome",
    "num_frames",
    "decode_failure_frames",
    "detected_frames",
    "missing_left_pct",
    "missing_right_pct",
    "missing_pose_pct",
    "missing_both_pct",
    "processing_time_sec",
    "output_path",
    "skip_reason",
    "error_message",
]


# ---------------------------------------------------------------------------
# Path sanitisation helper
# ---------------------------------------------------------------------------

def _sanitize_path_component(name: str) -> str:
    """
    Replace characters unsafe in a filesystem path component with underscores.

    WLASL sign labels are all plain ASCII words, so this is a safety net
    rather than a routine operation. Handles Windows restrictions as well.

    Parameters
    ----------
    name : str
        Raw path component (sign label, video_id, etc.).

    Returns
    -------
    str
        Safe path component suitable for all target filesystems.
    """
    safe = _UNSAFE_PATH_CHARS.sub("_", name)
    safe = safe.strip(". ")
    safe = re.sub(r"_+", "_", safe)
    return safe or "unknown"


# ---------------------------------------------------------------------------
# Sidecar metadata schema
# ---------------------------------------------------------------------------

def _meta_path_for(npy_path: Path) -> Path:
    """Return the .meta.json sibling path for a given .npy file."""
    return npy_path.with_suffix(".meta.json")


def _write_meta(npy_path: Path, result: "ExtractionResult") -> None:
    """
    Write per-clip extraction metadata to a sidecar .meta.json file.

    Parameters
    ----------
    npy_path : Path
        Path to the corresponding .npy file.
    result : ExtractionResult
        The freshly extracted result whose statistics to persist.
    """
    meta = result.to_dict()
    meta["schema_version"] = EXTRACTOR_SCHEMA_VERSION
    meta_path = _meta_path_for(npy_path)
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)
    except OSError as exc:
        logger.warning(
            f"Could not write sidecar metadata to {meta_path}: {exc}",
            extra={"stage": "extraction"},
        )


def _read_meta(npy_path: Path) -> Optional[dict[str, Any]]:
    """
    Load and return sidecar metadata, or None if missing/invalid/stale.

    Returns None (triggers reprocessing) if:
    - The .meta.json file does not exist.
    - The file is not valid JSON.
    - The schema_version does not match EXTRACTOR_SCHEMA_VERSION.

    Parameters
    ----------
    npy_path : Path
        Path to the .npy file.

    Returns
    -------
    dict | None
    """
    meta_path = _meta_path_for(npy_path)
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        stored_version = meta.get("schema_version", "")
        if stored_version != EXTRACTOR_SCHEMA_VERSION:
            logger.debug(
                f"Sidecar schema version mismatch for {npy_path.name}: "
                f"stored={stored_version!r}, current={EXTRACTOR_SCHEMA_VERSION!r}. "
                "Will reprocess.",
                extra={"stage": "extraction"},
            )
            return None
        return meta
    except (json.JSONDecodeError, OSError) as exc:
        logger.debug(
            f"Could not read sidecar for {npy_path.name}: {exc}. Will reprocess.",
            extra={"stage": "extraction"},
        )
        return None


# ---------------------------------------------------------------------------
# Internal frame accumulation dataclass
# ---------------------------------------------------------------------------

@dataclass
class _FrameDecodeStats:
    """
    Per-clip frame-level statistics accumulated during video processing.

    Separating decode failures from detection failures is the key design
    invariant: decode failures are codec/IO issues independent of whether
    the signer's hands were visible.

    Attributes
    ----------
    total_frames : int
        All frames appended to landmarks_list (decoded + decode-failures).
    decode_failure_frames : int
        Frames where cv2.read() returned ret=False (zero-filled, not
        counted in missing denominators).
    successfully_decoded_frames : int
        total_frames - decode_failure_frames. Denominator for missing_pct.
    detected_frames : int
        Successfully decoded frames where AT LEAST ONE hand was detected.
        This is the primary criterion in the v1.2 skip policy.
    missing_left_hand : int
        Successfully decoded frames where MediaPipe found no left hand.
    missing_right_hand : int
        Successfully decoded frames where MediaPipe found no right hand.
    missing_pose : int
        Successfully decoded frames where MediaPipe found no pose.
    missing_both_hands : int
        Successfully decoded frames where BOTH hands were absent simultaneously.
    """
    total_frames: int = 0
    decode_failure_frames: int = 0
    missing_left_hand: int = 0
    missing_right_hand: int = 0
    missing_pose: int = 0
    missing_both_hands: int = 0

    @property
    def successfully_decoded_frames(self) -> int:
        return self.total_frames - self.decode_failure_frames

    @property
    def detected_frames(self) -> int:
        """Frames where at least one hand was detected (left OR right OR both)."""
        return self.successfully_decoded_frames - self.missing_both_hands

    @property
    def missing_pct(self) -> float:
        """
        Fraction of successfully-decoded frames with BOTH hands absent.

        Used as the catastrophe filter in the v1.2 dual-criterion policy.
        A value near 1.0 indicates the clip contains no usable sign content
        regardless of frame count.
        """
        denom = self.successfully_decoded_frames
        return self.missing_both_hands / denom if denom > 0 else 0.0

    @property
    def missing_left_pct(self) -> float:
        denom = self.successfully_decoded_frames
        return self.missing_left_hand / denom if denom > 0 else 0.0

    @property
    def missing_right_pct(self) -> float:
        denom = self.successfully_decoded_frames
        return self.missing_right_hand / denom if denom > 0 else 0.0

    @property
    def missing_pose_pct(self) -> float:
        denom = self.successfully_decoded_frames
        return self.missing_pose / denom if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ExtractionResult:
    """
    Result metadata for a single clip extraction.

    Produced by ``LandmarkExtractor.extract_video()`` for every clip,
    regardless of whether it was freshly extracted, loaded from cache,
    skipped by policy, or failed with an error.

    Attributes
    ----------
    video_id : str
        WLASL video identifier.
    sign_label : str
        Human-readable sign name.
    split : str
        "train", "val", or "test".
    output_path : str
        Absolute path to the written .npy file (empty if skipped/error).
    status : str
        One of "extracted", "cached", "skipped", "error".
    num_frames : int
        Total frames in the output array (decoded + decode-failures).
    decode_failure_frames : int
        Frames where OpenCV failed to decode (zero-filled; excluded from
        missing_pct denominator).
    detected_frames : int
        Frames where at least one hand was detected. Primary criterion in
        the v1.2 skip policy. New in schema v1.2.
    missing_left_hand_frames : int
        Successfully decoded frames where left-hand landmarks were zero-filled.
    missing_right_hand_frames : int
        Successfully decoded frames where right-hand landmarks were zero-filled.
    missing_pose_frames : int
        Successfully decoded frames where pose landmarks were zero-filled.
    missing_both_hands_frames : int
        Successfully decoded frames where both hands were absent simultaneously.
    missing_pct : float
        missing_both_hands_frames / successfully_decoded_frames. Used as the
        catastrophe filter (second criterion) in v1.2 policy. [0.0, 1.0]
    skip_reason : str
        Non-empty only when status=="skipped". One of:
        "insufficient_detected_frames", "catastrophic_missing_rate",
        "no_frames_extracted".
    processing_time_sec : float
        Wall-clock seconds for this clip. 0.0 for cached clips.
    error_message : str
        Non-empty only when status=="error".
    """
    video_id: str
    sign_label: str
    split: str
    output_path: str = ""
    status: str = "extracted"
    num_frames: int = 0
    decode_failure_frames: int = 0
    detected_frames: int = 0
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

    Written to ``data/preprocessing_summary_latest.json`` (overwritten each
    run) and appended to ``data/preprocessing_summary_history.json``.

    Attributes
    ----------
    run_id : str
        ISO 8601 timestamp uniquely identifying this run.
    split : str
        Which split(s) were processed.
    total_clips : int
        Number of clips submitted for processing.
    extracted : int
        Clips freshly extracted in this run.
    cached : int
        Clips skipped because a valid .npy + .meta.json pair already existed.
    skipped : int
        Clips skipped due to skip policy or video-read failure.
    errors : int
        Clips that raised unexpected exceptions.
    total_frames_extracted : int
        Sum of total frame counts across usable clips (extracted + cached).
    mean_frames_per_clip : float
        Average frame count for usable clips.
    mean_detected_per_clip : float
        Average detected_frames count across usable clips. New in v1.2.
    mean_missing_pct : float
        Average both-hands-absent rate across usable clips.
    max_missing_pct : float
        Worst-case both-hands-absent rate across usable clips.
    per_clip_results : list[ExtractionResult]
        Full per-clip metadata for notebook consumption.
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
    mean_detected_per_clip: float = 0.0
    mean_missing_pct: float = 0.0
    max_missing_pct: float = 0.0
    per_clip_results: list[ExtractionResult] = field(default_factory=list)
    elapsed_sec: float = 0.0

    @property
    def success_count(self) -> int:
        return self.extracted + self.cached

    @property
    def skip_rate(self) -> float:
        return self.skipped / self.total_clips if self.total_clips > 0 else 0.0

    @property
    def error_rate(self) -> float:
        return self.errors / self.total_clips if self.total_clips > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["success_count"] = self.success_count
        d["skip_rate"]     = round(self.skip_rate, 4)
        d["error_rate"]    = round(self.error_rate, 4)
        return d

    def save(self, summary_dir: str | Path) -> tuple[Path, Path]:
        """
        Write summary JSON files (latest + history).

        Parameters
        ----------
        summary_dir : str | Path
            Directory where both files are written (typically ``data/``).

        Returns
        -------
        tuple[Path, Path]
            (latest_path, history_path)
        """
        out_dir = Path(summary_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        current_dict = self.to_dict()

        latest_path = out_dir / "preprocessing_summary_latest.json"
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(current_dict, f, indent=2, default=str)

        history_path = out_dir / "preprocessing_summary_history.json"
        existing_runs: list[dict[str, Any]] = []
        if history_path.exists():
            try:
                with open(history_path, encoding="utf-8") as f:
                    data = json.load(f)
                existing_runs = data if isinstance(data, list) else [data]
            except (json.JSONDecodeError, OSError):
                logger.warning(
                    f"Could not read {history_path}. Starting fresh history.",
                    extra={"stage": "extraction"},
                )

        existing_runs.append(current_dict)
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(existing_runs, f, indent=2, default=str)

        logger.info(
            f"Extraction summary saved | "
            f"latest={latest_path} ({latest_path.stat().st_size / 1024:.1f} KB) | "
            f"history={history_path} ({len(existing_runs)} runs) | "
            f"clips_in_latest={self.total_clips}",
            extra={"stage": "extraction"},
        )
        return latest_path, history_path

    def print_summary(self) -> None:
        """Log a human-readable summary at INFO level."""
        logger.info("=" * 65, extra={"stage": "extraction"})
        logger.info("EXTRACTION SUMMARY", extra={"stage": "extraction"})
        logger.info("=" * 65, extra={"stage": "extraction"})
        logger.info(f"  Total submitted  : {self.total_clips}",   extra={"stage": "extraction"})
        logger.info(f"  Freshly extracted: {self.extracted}",     extra={"stage": "extraction"})
        logger.info(f"  Loaded from cache: {self.cached}",        extra={"stage": "extraction"})
        logger.info(
            f"  Skipped (policy) : {self.skipped}  "
            f"(skip_rate={self.skip_rate:.1%})",
            extra={"stage": "extraction"},
        )
        logger.info(f"  Errors           : {self.errors}",        extra={"stage": "extraction"})
        logger.info(f"  Usable clips     : {self.success_count}", extra={"stage": "extraction"})
        if self.success_count > 0:
            logger.info(
                f"  Mean frames/clip : {self.mean_frames_per_clip:.1f}",
                extra={"stage": "extraction"},
            )
            logger.info(
                f"  Mean detected/clip: {self.mean_detected_per_clip:.1f}",
                extra={"stage": "extraction"},
            )
            logger.info(
                f"  Mean missing %   : {self.mean_missing_pct:.1%}  "
                f"(max={self.max_missing_pct:.1%})",
                extra={"stage": "extraction"},
            )
        logger.info(f"  Elapsed          : {self.elapsed_sec:.1f}s", extra={"stage": "extraction"})
        logger.info("=" * 65, extra={"stage": "extraction"})


# ---------------------------------------------------------------------------
# Landmark inventory CSV
# ---------------------------------------------------------------------------

def _build_inventory_row(result: ExtractionResult) -> dict[str, Any]:
    """Build a single landmark_inventory.csv row from an ExtractionResult."""
    n_decoded = result.num_frames - result.decode_failure_frames
    return {
        "video_id":              result.video_id,
        "sign_label":            result.sign_label,
        "split":                 result.split,
        "outcome":               result.status,
        "num_frames":            result.num_frames,
        "decode_failure_frames": result.decode_failure_frames,
        "detected_frames":       result.detected_frames,
        "missing_left_pct":      round(
            result.missing_left_hand_frames / n_decoded if n_decoded > 0 else 0.0, 4
        ),
        "missing_right_pct":     round(
            result.missing_right_hand_frames / n_decoded if n_decoded > 0 else 0.0, 4
        ),
        "missing_pose_pct":      round(
            result.missing_pose_frames / n_decoded if n_decoded > 0 else 0.0, 4
        ),
        "missing_both_pct":      round(result.missing_pct, 4),
        "processing_time_sec":   result.processing_time_sec,
        "output_path":           result.output_path,
        "skip_reason":           result.skip_reason,
        "error_message":         result.error_message,
    }


def write_landmark_inventory(
    results: list[ExtractionResult],
    landmarks_dir: str | Path,
) -> Path:
    """
    Write a ``landmark_inventory.csv`` to the landmarks root directory.

    This file is the Stage 3 equivalent of ``raw_inventory.json`` for Stage 1.
    Notebook 02 and Stage 4/5 code can load it with ``pd.read_csv()`` for
    instant per-clip statistics without scanning .npy files.

    Columns: video_id, sign_label, split, outcome, num_frames,
             decode_failure_frames, detected_frames,
             missing_left_pct, missing_right_pct, missing_pose_pct,
             missing_both_pct, processing_time_sec,
             output_path, skip_reason, error_message

    Parameters
    ----------
    results : list[ExtractionResult]
        All per-clip results from a batch run (extracted + cached + skipped).
    landmarks_dir : str | Path
        Root of the landmarks directory tree.

    Returns
    -------
    Path
        Absolute path to the written CSV.
    """
    out_dir = Path(landmarks_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / _LANDMARK_INVENTORY_FILENAME

    rows = [_build_inventory_row(r) for r in results]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_INVENTORY_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(
        f"Landmark inventory written | path={csv_path} | rows={len(rows)}",
        extra={"stage": "extraction"},
    )
    return csv_path


# ---------------------------------------------------------------------------
# LandmarkExtractor
# ---------------------------------------------------------------------------

class LandmarkExtractor:
    """
    Extracts MediaPipe Holistic landmarks from WLASL video clips.

    This class is the single component responsible for converting raw video
    pixels into structured skeletal representations. It operates at three
    granularities:

    - **Batch mode** (``extract_dataset()``): processes all clips listed in
      the split CSVs, writing .npy + .meta.json files to ``data/landmarks/``.
      Used by ``pipelines/run_landmark_extraction.py``.

    - **Single-clip mode** (``extract_video()``): processes one video file.
      Used for sample runs and testing.

    - **Single-frame mode** (``extract_frame()``): processes one BGR frame.
      Used by ``GesturePredictor`` (Stage 7) at inference time.

    v1.2 Skip Policy
    ----------------
    The skip policy has been changed from a ratio-based criterion to a
    dual-criterion absolute policy. See module docstring for full rationale.

    The two parameters controlling the policy are:

    ``min_detected_frames`` (default: 15, from constants.MIN_DETECTED_FRAMES_DEFAULT)
        Minimum number of frames where at least one hand must be detected.
        Clips below this threshold are skipped with reason
        "insufficient_detected_frames".

    ``max_missing_pct`` (default: 0.95, from constants.MAX_MISSING_PCT_CATASTROPHE)
        Catastrophe filter. Clips where 95%+ of decoded frames have both
        hands absent are skipped regardless of detected_frames, with reason
        "catastrophic_missing_rate".

    These can be overridden at construction time or via the OmegaConf config.

    Parameters
    ----------
    config : omegaconf.DictConfig | None
        Project config loaded via ``load_config()``. If None, defaults are used.
        Reads ``config.data.min_detected_frames`` and
        ``config.data.max_missing_frame_pct`` if present.
    landmarks_dir : str | Path | None
        Root for extracted .npy files. Defaults to ``<repo_root>/data/landmarks``.
    model_complexity : int
        MediaPipe Holistic model complexity: 0=lite, 1=full, 2=heavy.
        **Must be identical between extraction and inference.**
    min_detection_confidence : float
        MediaPipe minimum detection confidence. Default 0.5.
    min_tracking_confidence : float
        MediaPipe minimum tracking confidence. Default 0.5.
    min_detected_frames : int
        Minimum number of frames where at least one hand must be detected.
        Overrides config if provided explicitly.
    max_missing_pct : float
        Catastrophe filter threshold. Overrides config if provided explicitly.
    sample_clips_per_sign : int
        Number of clips per sign in sample-only mode (capped by availability).
        Default 3.

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
        min_detected_frames: int = MIN_DETECTED_FRAMES_DEFAULT,
        max_missing_pct: float = MAX_MISSING_PCT_CATASTROPHE,
        sample_clips_per_sign: int = _DEFAULT_SAMPLE_CLIPS_PER_SIGN,
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

        # Policy parameters — start from constructor arguments (which have
        # module-constant defaults), then override from config if present.
        self._min_detected_frames: int   = max(1, min_detected_frames)
        self._max_missing_pct: float     = max_missing_pct

        if config is not None:
            try:
                val = config.data.min_detected_frames
                if val is not None:
                    self._min_detected_frames = max(1, int(val))
            except Exception:  # noqa: BLE001
                pass
            try:
                val = config.data.max_missing_frame_pct
                if val is not None:
                    self._max_missing_pct = float(val)
            except Exception:  # noqa: BLE001
                pass

        self._model_complexity         = model_complexity
        self._min_detection_confidence = min_detection_confidence
        self._min_tracking_confidence  = min_tracking_confidence
        self._sample_clips_per_sign    = max(1, sample_clips_per_sign)

        # MediaPipe Holistic instance — lazily initialised on first use.
        self._holistic: Optional[Any]           = None
        self._mp_holistic_module: Optional[Any] = None
        self._mp_drawing: Optional[Any]         = None

        logger.info(
            f"LandmarkExtractor initialised | "
            f"schema_version={EXTRACTOR_SCHEMA_VERSION} | "
            f"skip_policy=dual_criterion_v1.2 | "
            f"min_detected_frames={self._min_detected_frames} | "
            f"max_missing_pct={self._max_missing_pct:.0%} | "
            f"model_complexity={model_complexity} | "
            f"sample_clips_per_sign={self._sample_clips_per_sign} | "
            f"landmarks_dir={self._landmarks_dir}",
            extra={"stage": "extraction"},
        )

    # ------------------------------------------------------------------
    # MediaPipe lifecycle management
    # ------------------------------------------------------------------

    def _init_mediapipe(self) -> None:
        """
        Lazily initialise the MediaPipe Holistic context.

        Called automatically on the first call to any extraction method.
        Can also be called explicitly to warm up the model before the main
        loop to avoid a cold-start timing penalty on the first clip.
        """
        if self._holistic is not None:
            return

        logger.info(
            "Initialising MediaPipe Holistic model...",
            extra={"stage": "extraction"},
        )
        t0 = time.time()

        self._mp_holistic_module = mp.solutions.holistic
        self._mp_drawing         = mp.solutions.drawing_utils

        self._holistic = self._mp_holistic_module.Holistic(
            static_image_mode=False,
            model_complexity=self._model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=False,
            refine_face_landmarks=False,
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
        """Release the MediaPipe Holistic context. Idempotent."""
        if self._holistic is not None:
            try:
                self._holistic.close()
            except Exception:  # noqa: BLE001
                pass
            self._holistic = None
            logger.debug(
                "MediaPipe Holistic context released.",
                extra={"stage": "extraction"},
            )

    def __enter__(self) -> "LandmarkExtractor":
        self._init_mediapipe()
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    def __repr__(self) -> str:
        mp_status = (
            "initialised" if self._holistic is not None
            else "lazy (not yet initialised)"
        )
        return (
            f"LandmarkExtractor("
            f"schema_version={EXTRACTOR_SCHEMA_VERSION}, "
            f"model_complexity={self._model_complexity}, "
            f"skip_policy=dual_criterion("
            f"min_detected={self._min_detected_frames}, "
            f"max_missing={self._max_missing_pct:.0%}), "
            f"mediapipe={mp_status}, "
            f"landmarks_dir='{self._landmarks_dir}')"
        )

    # ------------------------------------------------------------------
    # Internal frame processing
    # ------------------------------------------------------------------

    def _process_single_frame(
        self,
        frame: np.ndarray,
    ) -> tuple[np.ndarray, bool, bool, bool]:
        """
        Run MediaPipe Holistic on one BGR frame and pack results.

        This is the single internal method that performs the actual MediaPipe
        call. Both ``extract_frame()`` (public, inference) and the batch video
        processing loop call this method, eliminating duplication.

        Parameters
        ----------
        frame : np.ndarray
            A non-empty BGR uint8 image (H×W×3).

        Returns
        -------
        tuple[np.ndarray, bool, bool, bool]
            (feature_vec, left_detected, right_detected, pose_detected)
            feature_vec — shape (225,) float32, zero-filled where not detected.
        """
        if self._holistic is None:
            self._init_mediapipe()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self._holistic.process(rgb_frame)
        rgb_frame.flags.writeable = True

        feature_vec = np.zeros(FEATURE_SIZE, dtype=np.float32)

        left_detected  = results.left_hand_landmarks  is not None
        right_detected = results.right_hand_landmarks is not None
        pose_detected  = results.pose_landmarks       is not None

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
    # Core single-frame extraction — public, reused by Stage 7
    # ------------------------------------------------------------------

    def extract_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract the 225-element landmark vector from a single BGR frame.

        This is the **primary public method for real-time inference** (Stage 7).
        GesturePredictor calls this on each webcam frame.

        Feature vector layout:
            [0  :63 ] left hand  — 21 landmarks × (x, y, z)
            [63 :126] right hand — 21 landmarks × (x, y, z)
            [126:225] pose       — 33 landmarks × (x, y, z)

        Parameters
        ----------
        frame : np.ndarray
            BGR uint8 image (H×W×3). Standard OpenCV format.

        Returns
        -------
        np.ndarray
            Shape ``(225,)`` float32.

        Raises
        ------
        ValueError
            If frame is None, empty, or not a 3-channel image.
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

        feature_vec, _, _, _ = self._process_single_frame(frame)
        return feature_vec

    # ------------------------------------------------------------------
    # Landmark packing helpers
    # ------------------------------------------------------------------

    def _pack_hand_landmarks(self, hand_landmarks) -> np.ndarray:
        """
        Flatten MediaPipe hand landmarks into a (63,) float32 array.

        Layout: [x0, y0, z0, x1, y1, z1, ..., x20, y20, z20]
        Landmark 0 is the wrist (MediaPipe convention).

        Raises
        ------
        RuntimeError
            If MediaPipe returns an unexpected landmark count.
        """
        actual_count = len(hand_landmarks.landmark)
        if actual_count != N_HAND_LANDMARKS:
            raise RuntimeError(
                f"MediaPipe returned {actual_count} hand landmarks; "
                f"expected {N_HAND_LANDMARKS}. "
                "This may indicate a MediaPipe version mismatch. "
                "Project requires mediapipe==0.10.14."
            )

        vec = np.empty(N_HAND_FEATURES, dtype=np.float32)
        for i, lm in enumerate(hand_landmarks.landmark):
            base          = i * N_COORDS_PER_LANDMARK
            vec[base]     = lm.x
            vec[base + 1] = lm.y
            vec[base + 2] = lm.z
        return vec

    def _pack_pose_landmarks(self, pose_landmarks) -> np.ndarray:
        """
        Flatten MediaPipe pose landmarks into a (99,) float32 array.

        Visibility scores are intentionally excluded to keep the feature
        size consistent with the 225-value specification.

        Raises
        ------
        RuntimeError
            If MediaPipe returns an unexpected landmark count.
        """
        actual_count = len(pose_landmarks.landmark)
        if actual_count != N_POSE_LANDMARKS:
            raise RuntimeError(
                f"MediaPipe returned {actual_count} pose landmarks; "
                f"expected {N_POSE_LANDMARKS}. "
                "This may indicate a MediaPipe version mismatch. "
                "Project requires mediapipe==0.10.14."
            )

        vec = np.empty(N_POSE_FEATURES, dtype=np.float32)
        for i, lm in enumerate(pose_landmarks.landmark):
            base          = i * N_COORDS_PER_LANDMARK
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
    ) -> ExtractionResult:
        """
        Extract landmarks from one video clip and save to a .npy file.

        Full lifecycle:
        1. Check for cached .npy + .meta.json; restore statistics and return
           if valid (schema version, shape, dtype, full finiteness check).
        2. Open the video with OpenCV.
        3. Process each frame — decode failures tracked separately from
           detection failures.
        4. Apply the v1.2 dual-criterion skip policy.
        5. Write the (num_frames, 225) float32 array.
        6. Write sidecar .meta.json with full per-clip statistics.
        7. Return an ExtractionResult.

        v1.2 Skip policy applied in step 4:
        - Skip if detected_frames < min_detected_frames
          (reason: "insufficient_detected_frames")
        - Skip if missing_pct > max_missing_pct
          (reason: "catastrophic_missing_rate")

        Parameters
        ----------
        video_path : str | Path
            Input video file path.
        output_path : str | Path
            Destination .npy file path.
        video_id : str
            WLASL video identifier.
        sign_label : str
            Sign name.
        split : str
            "train", "val", or "test".
        force : bool
            If True, reprocess even if a valid .npy + .meta.json already exists.

        Returns
        -------
        ExtractionResult
            ``result.status`` is one of "extracted", "cached", "skipped", "error".
        """
        t0 = time.time()

        output_path = Path(output_path).resolve()

        # Resolve video path (relative → absolute)
        vp = Path(video_path)
        if not vp.is_absolute():
            vp = _REPO_ROOT / vp
        vp = vp.resolve()

        # ----------------------------------------------------------------
        # Resumability: load from cache if .npy + .meta.json are valid
        # ----------------------------------------------------------------
        if not force and output_path.exists():
            cached = self._try_load_cached(output_path, video_id, sign_label, split)
            if cached is not None:
                return cached

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
                processing_time_sec=round(time.time() - t0, 4),
            )

        # ----------------------------------------------------------------
        # Extract landmarks frame-by-frame
        # ----------------------------------------------------------------
        try:
            landmarks_array, frame_stats = self._process_video_frames(vp, video_id)
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
                processing_time_sec=round(time.time() - t0, 4),
            )

        if landmarks_array is None:
            return ExtractionResult(
                video_id=video_id,
                sign_label=sign_label,
                split=split,
                status="skipped",
                skip_reason="no_frames_extracted",
                processing_time_sec=round(time.time() - t0, 4),
            )

        n_frames         = landmarks_array.shape[0]
        detected_frames  = frame_stats.detected_frames
        missing_pct      = frame_stats.missing_pct

        # ----------------------------------------------------------------
        # v1.2 Dual-criterion skip policy
        #
        # Criterion 1: insufficient detected frames
        #   Does this clip contain enough usable frames for training?
        #   This directly answers the question the LSTM cares about.
        #
        # Criterion 2: catastrophic missing rate
        #   If 95%+ of decoded frames have both hands absent, the clip is
        #   genuinely unusable regardless of absolute frame count.
        # ----------------------------------------------------------------

        # Criterion 1: minimum detected frames
        if detected_frames < self._min_detected_frames:
            logger.info(
                f"Skipping {video_id} ({sign_label}): "
                f"only {detected_frames} detected frames "
                f"(threshold={self._min_detected_frames}) | "
                f"n_frames={n_frames} | "
                f"missing_both={frame_stats.missing_both_hands}/"
                f"{frame_stats.successfully_decoded_frames} "
                f"({missing_pct:.1%})",
                extra={"stage": "extraction", "video_id": video_id},
            )
            return ExtractionResult(
                video_id=video_id,
                sign_label=sign_label,
                split=split,
                status="skipped",
                num_frames=n_frames,
                decode_failure_frames=frame_stats.decode_failure_frames,
                detected_frames=detected_frames,
                missing_left_hand_frames=frame_stats.missing_left_hand,
                missing_right_hand_frames=frame_stats.missing_right_hand,
                missing_pose_frames=frame_stats.missing_pose,
                missing_both_hands_frames=frame_stats.missing_both_hands,
                missing_pct=round(missing_pct, 4),
                skip_reason="insufficient_detected_frames",
                processing_time_sec=round(time.time() - t0, 4),
            )

        # Criterion 2: catastrophic missing rate
        if missing_pct > self._max_missing_pct:
            logger.info(
                f"Skipping {video_id} ({sign_label}): "
                f"catastrophic missing rate {missing_pct:.1%} "
                f"(threshold={self._max_missing_pct:.0%}) | "
                f"n_frames={n_frames} | "
                f"detected={detected_frames}",
                extra={"stage": "extraction", "video_id": video_id},
            )
            return ExtractionResult(
                video_id=video_id,
                sign_label=sign_label,
                split=split,
                status="skipped",
                num_frames=n_frames,
                decode_failure_frames=frame_stats.decode_failure_frames,
                detected_frames=detected_frames,
                missing_left_hand_frames=frame_stats.missing_left_hand,
                missing_right_hand_frames=frame_stats.missing_right_hand,
                missing_pose_frames=frame_stats.missing_pose,
                missing_both_hands_frames=frame_stats.missing_both_hands,
                missing_pct=round(missing_pct, 4),
                skip_reason="catastrophic_missing_rate",
                processing_time_sec=round(time.time() - t0, 4),
            )

        # ----------------------------------------------------------------
        # Feature size guard (defensive — should never fire)
        # ----------------------------------------------------------------
        if landmarks_array.shape[1] != FEATURE_SIZE:
            raise RuntimeError(
                f"Feature size mismatch for {video_id}: "
                f"expected columns={FEATURE_SIZE}, got {landmarks_array.shape[1]}. "
                "This is a bug in the landmark packing code."
            )

        # ----------------------------------------------------------------
        # Write .npy file
        # ----------------------------------------------------------------
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(output_path), landmarks_array)

        processing_time = round(time.time() - t0, 4)

        result = ExtractionResult(
            video_id=video_id,
            sign_label=sign_label,
            split=split,
            output_path=str(output_path),
            status="extracted",
            num_frames=n_frames,
            decode_failure_frames=frame_stats.decode_failure_frames,
            detected_frames=detected_frames,
            missing_left_hand_frames=frame_stats.missing_left_hand,
            missing_right_hand_frames=frame_stats.missing_right_hand,
            missing_pose_frames=frame_stats.missing_pose,
            missing_both_hands_frames=frame_stats.missing_both_hands,
            missing_pct=round(missing_pct, 4),
            processing_time_sec=processing_time,
        )

        _write_meta(output_path, result)

        logger.debug(
            f"Extracted {video_id} ({sign_label}) | "
            f"shape={landmarks_array.shape} | "
            f"detected={detected_frames}/{frame_stats.successfully_decoded_frames} | "
            f"decode_failures={frame_stats.decode_failure_frames} | "
            f"missing_left={frame_stats.missing_left_hand}/"
            f"{frame_stats.successfully_decoded_frames} | "
            f"missing_right={frame_stats.missing_right_hand}/"
            f"{frame_stats.successfully_decoded_frames} | "
            f"missing_both={frame_stats.missing_both_hands}/"
            f"{frame_stats.successfully_decoded_frames} "
            f"({missing_pct:.1%}) | "
            f"time={processing_time:.3f}s",
            extra={"stage": "extraction", "video_id": video_id},
        )

        return result

    # ------------------------------------------------------------------
    # Frame-level video processing
    # ------------------------------------------------------------------

    def _process_video_frames(
        self,
        video_path: Path,
        video_id: str,
    ) -> tuple[Optional[np.ndarray], _FrameDecodeStats]:
        """
        Open a video file and extract landmark vectors for every frame.

        Decode failures (``ret=False``) are tracked in
        ``_FrameDecodeStats.decode_failure_frames`` and are excluded from the
        detection-failure counts. The skip policy in ``extract_video()``
        uses ``frame_stats.detected_frames`` and ``frame_stats.missing_pct``
        rather than any ratio of decode failures, so transient codec errors
        never affect the keep/skip decision.

        Parameters
        ----------
        video_path : Path
            Absolute path to the video file.
        video_id : str
            For debug logging only.

        Returns
        -------
        tuple[np.ndarray | None, _FrameDecodeStats]
            (stacked_array, stats). Returns (None, stats) if the clip has
            fewer than ``_MIN_VALID_FRAMES`` successfully decoded frames.

        Raises
        ------
        RuntimeError
            If OpenCV cannot open the file at all.
        """
        if self._holistic is None:
            self._init_mediapipe()

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(
                f"OpenCV could not open video: {video_path}. "
                "File may be corrupt or encoded in an unsupported codec."
            )

        landmarks_list: list[np.ndarray] = []
        frame_stats = _FrameDecodeStats()

        frame_idx         = 0
        consecutive_fails = 0

        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    consecutive_fails += 1
                    if consecutive_fails >= _MAX_CONSECUTIVE_READ_FAILURES:
                        # Treat as end-of-stream
                        break

                    # Transient decode failure: zero-fill but track separately.
                    # CRITICAL: do NOT increment any detection-missing counters.
                    # Decode failures are NOT MediaPipe detection failures.
                    logger.debug(
                        f"Transient decode failure at frame {frame_idx} in "
                        f"{video_id} "
                        f"(consecutive={consecutive_fails}/"
                        f"{_MAX_CONSECUTIVE_READ_FAILURES}). "
                        "Zero-filling; NOT counted as detection failure.",
                        extra={"stage": "extraction"},
                    )
                    landmarks_list.append(np.zeros(FEATURE_SIZE, dtype=np.float32))
                    frame_stats.total_frames          += 1
                    frame_stats.decode_failure_frames += 1
                    frame_idx += 1
                    continue

                # Successful decode — reset failure counter
                consecutive_fails = 0

                if frame is None or frame.size == 0:
                    # Null frame despite ret=True — treat as decode failure
                    logger.debug(
                        f"Null frame at index {frame_idx} in {video_id}. "
                        "Zero-filling as decode failure.",
                        extra={"stage": "extraction"},
                    )
                    landmarks_list.append(np.zeros(FEATURE_SIZE, dtype=np.float32))
                    frame_stats.total_frames          += 1
                    frame_stats.decode_failure_frames += 1
                    frame_idx += 1
                    continue

                # Successfully decoded frame — run MediaPipe
                feat_vec, left_det, right_det, pose_det = (
                    self._process_single_frame(frame)
                )
                landmarks_list.append(feat_vec)
                frame_stats.total_frames += 1

                # Count detection failures only for successfully decoded frames
                if not left_det:
                    frame_stats.missing_left_hand += 1
                if not right_det:
                    frame_stats.missing_right_hand += 1
                if not pose_det:
                    frame_stats.missing_pose += 1
                if not left_det and not right_det:
                    frame_stats.missing_both_hands += 1

                frame_idx += 1

        finally:
            cap.release()

        # Require a minimum number of successfully decoded frames before
        # even attempting the skip-policy evaluation.
        if frame_stats.successfully_decoded_frames < _MIN_VALID_FRAMES:
            logger.warning(
                f"Only {frame_stats.successfully_decoded_frames} successfully "
                f"decoded frames in {video_id} (total={frame_stats.total_frames}, "
                f"decode_failures={frame_stats.decode_failure_frames}, "
                f"minimum={_MIN_VALID_FRAMES}). Treating as empty.",
                extra={"stage": "extraction", "video_id": video_id},
            )
            return None, frame_stats

        stacked = np.stack(landmarks_list, axis=0).astype(np.float32)
        return stacked, frame_stats

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _try_load_cached(
        self,
        output_path: Path,
        video_id: str,
        sign_label: str,
        split: str,
    ) -> Optional[ExtractionResult]:
        """
        Validate a cached .npy file and restore statistics from its sidecar.

        Validation checks (all must pass; any failure triggers reprocessing):
        1. .npy ndim == 2
        2. .npy shape[1] == FEATURE_SIZE (225)
        3. .npy dtype == float32
        4. All values are finite (full scan)
        5. .meta.json exists and has matching schema_version

        Parameters
        ----------
        output_path : Path
            Path to the existing .npy file.

        Returns
        -------
        ExtractionResult | None
            status="cached" with restored statistics if valid. None triggers
            reprocessing.
        """
        try:
            arr = np.load(str(output_path), mmap_mode="r")

            if arr.ndim != 2:
                logger.warning(
                    f"Cached .npy for {video_id} has ndim={arr.ndim} "
                    "(expected 2). Reprocessing.",
                    extra={"stage": "extraction", "video_id": video_id},
                )
                return None

            if arr.shape[1] != FEATURE_SIZE:
                logger.warning(
                    f"Cached .npy for {video_id} has shape {arr.shape} "
                    f"(expected (N, {FEATURE_SIZE})). Reprocessing.",
                    extra={"stage": "extraction", "video_id": video_id},
                )
                return None

            if arr.dtype != np.float32:
                logger.warning(
                    f"Cached .npy for {video_id} has dtype={arr.dtype} "
                    "(expected float32). Reprocessing.",
                    extra={"stage": "extraction", "video_id": video_id},
                )
                return None

            # Full finiteness scan. mmap_mode="r" ensures only touched pages
            # are loaded into RAM, so this is efficient even for large arrays.
            if arr.size > 0 and not np.isfinite(arr).all():
                logger.warning(
                    f"Cached .npy for {video_id} contains non-finite values "
                    "(NaN or Inf). Reprocessing.",
                    extra={"stage": "extraction", "video_id": video_id},
                )
                return None

            n_frames = int(arr.shape[0])

        except Exception as exc:
            logger.warning(
                f"Could not read cached .npy for {video_id}: {exc}. Reprocessing.",
                extra={"stage": "extraction", "video_id": video_id},
            )
            return None

        # Validate and read sidecar metadata
        meta = _read_meta(output_path)
        if meta is None:
            logger.debug(
                f"Sidecar missing or stale for {video_id}. Reprocessing.",
                extra={"stage": "extraction", "video_id": video_id},
            )
            return None

        logger.debug(
            f"Cache hit: {video_id} | shape=({n_frames}, {FEATURE_SIZE}) | "
            f"detected={meta.get('detected_frames', 0)} | "
            f"missing_pct={meta.get('missing_pct', 0.0):.1%} | "
            f"decode_failures={meta.get('decode_failure_frames', 0)}",
            extra={"stage": "extraction", "video_id": video_id},
        )
        return ExtractionResult(
            video_id=video_id,
            sign_label=sign_label,
            split=split,
            output_path=str(output_path),
            status="cached",
            num_frames=n_frames,
            decode_failure_frames=   meta.get("decode_failure_frames",      0),
            detected_frames=         meta.get("detected_frames",             0),
            missing_left_hand_frames=meta.get("missing_left_hand_frames",    0),
            missing_right_hand_frames=meta.get("missing_right_hand_frames",  0),
            missing_pose_frames=     meta.get("missing_pose_frames",         0),
            missing_both_hands_frames=meta.get("missing_both_hands_frames",  0),
            missing_pct=             meta.get("missing_pct",                 0.0),
            # processing_time_sec intentionally left at 0.0 for cached clips
        )

    # ------------------------------------------------------------------
    # Batch extraction
    # ------------------------------------------------------------------

    def extract_dataset(
        self,
        split_csv_paths: dict[str, str | Path],
        force: bool = False,
        sample_only: bool = False,
        summary_dir: Optional[str | Path] = None,
    ) -> ExtractionStats:
        """
        Extract landmarks for all clips listed in the split CSV files.

        Top-level method called by ``run_landmark_extraction.py``. Iterates
        over all clips in the provided splits, calls ``extract_video()`` for
        each, accumulates statistics, and writes summary JSON files and the
        landmark inventory CSV.

        Parameters
        ----------
        split_csv_paths : dict[str, str | Path]
            Mapping of split_name → CSV path.
        force : bool
            If True, reprocess all clips even if valid cache exists.
        sample_only : bool
            If True, process ``self._sample_clips_per_sign`` clips per sign
            per split (capped by availability). Uses alphabetical selection
            by video_id for reproducibility.
        summary_dir : str | Path | None
            Directory for summary JSON outputs. Defaults to ``data/``.

        Returns
        -------
        ExtractionStats
            Aggregate statistics for the completed run.
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

        self._init_mediapipe()

        all_results: list[ExtractionResult] = []

        for split_name, csv_path in sorted(split_csv_paths.items()):
            csv_path = Path(csv_path)
            if not csv_path.exists():
                logger.error(
                    f"Split CSV not found: {csv_path}. Skipping split '{split_name}'.",
                    extra={"stage": "extraction"},
                )
                continue

            logger.info(
                f"Processing split: {split_name} | csv={csv_path}",
                extra={"stage": "extraction"},
            )

            try:
                df = pd.read_csv(csv_path)
            except Exception as exc:
                logger.error(
                    f"Could not read CSV {csv_path}: {exc}",
                    extra={"stage": "extraction"},
                )
                continue

            required_cols = {"video_id", "sign_label", "video_path"}
            missing_cols  = required_cols - set(df.columns)
            if missing_cols:
                logger.error(
                    f"Split CSV {csv_path} missing required columns: {missing_cols}. "
                    f"Skipping split '{split_name}'.",
                    extra={"stage": "extraction"},
                )
                continue

            if sample_only:
                df = (
                    df.sort_values("video_id")
                    .groupby("sign_label", sort=False)
                    .head(self._sample_clips_per_sign)
                    .reset_index(drop=True)
                )
                n_signs = df["sign_label"].nunique()
                logger.info(
                    f"Sample mode: selected {len(df)} clips from '{split_name}' "
                    f"(up to {self._sample_clips_per_sign} per sign, "
                    f"{n_signs} signs represented)",
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
        stats.total_clips = len(all_results)
        stats.extracted   = sum(1 for r in all_results if r.status == "extracted")
        stats.cached      = sum(1 for r in all_results if r.status == "cached")
        stats.skipped     = sum(1 for r in all_results if r.status == "skipped")
        stats.errors      = sum(1 for r in all_results if r.status == "error")
        stats.per_clip_results = all_results

        usable = [r for r in all_results if r.status in ("extracted", "cached")]
        if usable:
            frame_counts    = [r.num_frames     for r in usable if r.num_frames > 0]
            detected_counts = [r.detected_frames for r in usable]
            missing_pcts    = [r.missing_pct    for r in usable]

            stats.total_frames_extracted = sum(frame_counts)
            stats.mean_frames_per_clip   = (
                sum(frame_counts) / len(frame_counts) if frame_counts else 0.0
            )
            stats.mean_detected_per_clip = (
                sum(detected_counts) / len(detected_counts) if detected_counts else 0.0
            )
            stats.mean_missing_pct = (
                sum(missing_pcts) / len(missing_pcts) if missing_pcts else 0.0
            )
            stats.max_missing_pct = max(missing_pcts) if missing_pcts else 0.0

        stats.elapsed_sec = round(time.time() - batch_start, 1)

        out_dir = Path(summary_dir) if summary_dir else _DEFAULT_SUMMARY_DIR
        stats.save(out_dir)
        stats.print_summary()

        # Write landmark inventory CSV for Notebook 02
        try:
            write_landmark_inventory(all_results, self._landmarks_dir)
        except Exception as exc:
            logger.warning(
                f"Could not write landmark inventory CSV: {exc}",
                extra={"stage": "extraction"},
            )

        return stats

    def _process_split_df(
        self,
        df,
        split_name: str,
        force: bool,
    ) -> list[ExtractionResult]:
        """
        Iterate over one split's DataFrame and extract each clip.

        Parameters
        ----------
        df : pd.DataFrame
            Split DataFrame with columns: video_id, sign_label, video_path.
        split_name : str
            Used in output path construction.
        force : bool
            Passed through to ``extract_video()``.

        Returns
        -------
        list[ExtractionResult]
        """
        results: list[ExtractionResult] = []
        n_clips     = len(df)
        split_start = time.time()

        iterator = (
            tqdm(df.itertuples(index=False), total=n_clips, desc=split_name)
            if _TQDM_AVAILABLE
            else df.itertuples(index=False)
        )

        for i, row in enumerate(iterator):
            video_id   = str(row.video_id)
            sign_label = str(row.sign_label)
            video_path = str(row.video_path)

            safe_label = _sanitize_path_component(sign_label)

            output_path = (
                self._landmarks_dir
                / split_name
                / safe_label
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

            if (i + 1) % _LOG_INTERVAL == 0 or (i + 1) == n_clips:
                elapsed = time.time() - split_start
                rate    = (i + 1) / elapsed if elapsed > 0 else 0.0
                eta     = (n_clips - (i + 1)) / rate if rate > 0 else 0.0

                n_done    = sum(1 for r in results if r.status in ("extracted", "cached"))
                n_skipped = sum(1 for r in results if r.status == "skipped")
                n_errors  = sum(1 for r in results if r.status == "error")

                logger.info(
                    f"{split_name} | "
                    f"{i + 1}/{n_clips} clips | "
                    f"done={n_done} | skipped={n_skipped} | errors={n_errors} | "
                    f"rate={rate:.1f}/s | ETA={eta:.0f}s",
                    extra={"stage": "extraction"},
                )

        return results

    # ------------------------------------------------------------------
    # Drawing utility — Notebook 02 visualisation
    # ------------------------------------------------------------------

    def draw_landmarks(
        self,
        frame: np.ndarray,
        feature_vec: Optional[np.ndarray] = None,
        results=None,
    ) -> np.ndarray:
        """
        Draw MediaPipe landmarks on a copy of a frame for visualisation.

        Two usage modes:
        1. Pass ``results`` (raw MediaPipe Holistic output).
        2. Pass ``feature_vec`` (a (225,) array loaded from a .npy file).
           Zero-filled landmarks are skipped to avoid false dots at (0,0).

        Parameters
        ----------
        frame : np.ndarray
            BGR image. Not modified in place — a copy is returned.
        feature_vec : np.ndarray | None
            (225,) float32 array from a cached .npy file.
        results : mediapipe Holistic result | None
            Raw output from ``self._holistic.process()``.

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
                self._mp_drawing.DrawingSpec(color=(121, 22, 76),  thickness=2, circle_radius=4),
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
            lh_vec   = feature_vec[LEFT_HAND_SLICE].reshape(N_HAND_LANDMARKS,  N_COORDS_PER_LANDMARK)
            rh_vec   = feature_vec[RIGHT_HAND_SLICE].reshape(N_HAND_LANDMARKS, N_COORDS_PER_LANDMARK)
            pose_vec = feature_vec[POSE_SLICE].reshape(N_POSE_LANDMARKS,       N_COORDS_PER_LANDMARK)

            # Left hand — blue; skip zero-filled landmarks
            for lm in lh_vec:
                if np.allclose(lm[:2], 0.0, atol=1e-6):
                    continue
                cx, cy = int(lm[0] * w), int(lm[1] * h)
                if 0 <= cx < w and 0 <= cy < h:
                    cv2.circle(annotated, (cx, cy), 4, (255, 0, 0), -1)

            # Right hand — orange; skip zero-filled landmarks
            for lm in rh_vec:
                if np.allclose(lm[:2], 0.0, atol=1e-6):
                    continue
                cx, cy = int(lm[0] * w), int(lm[1] * h)
                if 0 <= cx < w and 0 <= cy < h:
                    cv2.circle(annotated, (cx, cy), 4, (0, 165, 255), -1)

            # Pose — green, smaller radius; skip zero-filled landmarks
            for lm in pose_vec:
                if np.allclose(lm[:2], 0.0, atol=1e-6):
                    continue
                cx, cy = int(lm[0] * w), int(lm[1] * h)
                if 0 <= cx < w and 0 <= cy < h:
                    cv2.circle(annotated, (cx, cy), 2, (0, 255, 0), -1)

        return annotated

    # ------------------------------------------------------------------
    # Static utility — load a .npy file with full validation
    # ------------------------------------------------------------------

    @staticmethod
    def load_landmarks(npy_path: str | Path) -> np.ndarray:
        """
        Load a previously extracted landmark array from disk.

        Validates shape and dtype on load. Does not copy if already float32.

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
            If the array shape is not (N, 225) or dtype is unexpected.
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

        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)

        return arr