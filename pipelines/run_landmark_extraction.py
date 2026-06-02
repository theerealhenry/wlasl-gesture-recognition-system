"""
pipelines/run_landmark_extraction.py
======================================
Stage 3 pipeline entry point — MediaPipe Holistic landmark extraction.

Overview
--------
This script processes every video clip in the signer-aware train/val/test
splits produced by Stage 1, runs MediaPipe Holistic on each frame, and caches
the resulting landmark arrays to disk as .npy files.

It is a completely separate script from ``run_preprocessing.py`` (Stage 1) by
design. The two stages have incompatible dependency and runtime profiles:

    Stage 1   — no MediaPipe dependency; ~2 minutes; stage-level resumability
    Stage 3   — requires MediaPipe Holistic; 30–90 minutes; file-level resumability

Mixing them would create an import-time MediaPipe dependency on every Stage 1
run, couple different resumability models, and make the codebase harder to
reason about.

Output layout
-------------
Landmark arrays are written to:

    data/landmarks/<split>/<sign_label>/<video_id>.npy

Each file contains a float32 array of shape ``(num_frames, 225)`` where:

    [0:63]    left hand  — 21 landmarks × (x, y, z)
    [63:126]  right hand — 21 landmarks × (x, y, z)
    [126:225] pose       — 33 landmarks × (x, y, z)

Alongside every .npy file a sibling .meta.json sidecar is written by the
extractor (v1.1 schema). The sidecar stores per-clip detection statistics
including the new ``decode_failure_frames`` field, which separates transient
OpenCV codec errors from genuine MediaPipe detection failures.

IMPORTANT: Arrays store the clip's actual frame count — they are NOT padded
to seq_len=30. Padding/truncation are deferred to ``FeaturePipeline`` (Stage 4)
so the same .npy files serve all sequence-length ablation experiments
({20, 30, 40, 60} frames) without any re-extraction.

Resumability
------------
The extractor itself checks for a valid .npy + .meta.json pair before doing any
work (shape, dtype, schema version v1.1, full finiteness scan). This pipeline
respects those cache hits and adds an optional ``--verify-existing`` path to
spot-check cached files independently before trusting them.

Summary outputs
---------------
Two pipeline-level JSON summary files are written after each run:

    data/preprocessing_summary_latest.json  — current run (always overwritten)
    data/preprocessing_summary_history.json — append-only audit log

These are written by ``_write_pipeline_summary()`` in this script and contain
pipeline-layer metadata (CLI args, run mode, health-check results) in addition
to the per-clip statistics delegated to the extractor's ``ExtractionStats``.
They are distinct from the extractor-internal summaries that
``LandmarkExtractor.extract_dataset()`` writes when called directly; this
script drives extraction clip-by-clip via ``extract_video()`` so it controls
the summary format entirely.

A ``landmark_inventory.csv`` is written by the extractor after each batch run
(via ``write_landmark_inventory``). Notebook 02 loads this CSV for the
missing-landmark analysis.

Schema alignment with extractor.py v1.1
-----------------------------------------
This script is aligned with ``src/features/extractor.py`` schema version 1.1.
Key behavioural changes from v1.0 that are reflected here:

  - ``decode_failure_frames`` field is now propagated through all
    ``_RunStats.record_*`` methods and included in summary outputs.
  - ``missing_pct`` is computed over successfully-decoded frames only (not
    total frames). Health-check thresholds have been adjusted accordingly:
    a global missing rate > 15% now indicates genuine MediaPipe quality
    issues, not codec noise inflating the denominator.
  - Cache-hit clips return ``status="cached"`` (not ``status="skipped"``).
    The routing logic in ``_run_extraction_loop`` has been updated to match.
  - The extractor handles .npy writes internally (no atomic tmp-file logic
    needed here). Post-write verification is still performed via
    ``_verify_npy_file()`` as an independent safety check.
  - ``write_landmark_inventory`` is called inside ``_finalise_run()`` via the
    extractor's results list, not called a second time from this script.

Usage
-----
Sample-only run (Stage 2 validation gate — ~2–5 minutes):
    python pipelines/run_landmark_extraction.py --sample-only

Full extraction, all splits (30–90 minutes):
    python pipelines/run_landmark_extraction.py --split all

Single split only:
    python pipelines/run_landmark_extraction.py --split train

Force re-extraction (overwrite existing .npy + .meta.json files):
    python pipelines/run_landmark_extraction.py --split all --force

Verify existing cached files before trusting them:
    python pipelines/run_landmark_extraction.py --split all --verify-existing

Dry run (validate inputs and log plan, write nothing):
    python pipelines/run_landmark_extraction.py --dry-run

Override MediaPipe thresholds:
    python pipelines/run_landmark_extraction.py --split all \
        --max-missing-frame-pct 0.40 \
        --min-detection-confidence 0.4

Verbose debug logging:
    python pipelines/run_landmark_extraction.py --sample-only --verbose

Exit codes
----------
0  — Extraction completed successfully (includes cache-hit-only runs and
     partial runs where some clips were skipped by policy).
1  — Input error: split CSV missing, no clips found, invalid argument value.
2  — Unexpected exception, extractor init failure, or post-run health check
     failure that exceeds the error rate threshold.

Design principles
-----------------
- No assert statements — explicit RuntimeError throughout
- No print() — structured logging via configure_logging / get_logger
- No MLflow — tracking begins at Stage 5 (run_training.py)
- Single source of truth for which videos to process: data/splits/*.csv
- Extraction delegated entirely to LandmarkExtractor.extract_video()
- Post-write verification independent of extractor's own cache check
- Per-clip exception isolation: one bad video never aborts the whole run
- ETA computed from a fixed pre-loop count — never drifts as files are written
- Aligned with extractor.py schema version 1.1 throughout
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Bootstrap: ensure repo root is on sys.path so src/ imports resolve
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Utils — configure logging before any other import that may emit log lines
# ---------------------------------------------------------------------------
from src.utils.logger import configure_logging, get_logger
from src.utils.reproducibility import set_seeds

# ---------------------------------------------------------------------------
# Feature modules
# ---------------------------------------------------------------------------
from src.features.extractor import (
    LandmarkExtractor,
    ExtractionResult,
    write_landmark_inventory,
)
from src.features.constants import FEATURE_SIZE, EXTRACTOR_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Default paths and constants
# ---------------------------------------------------------------------------
_DEFAULT_SPLITS_DIR    = str(_REPO_ROOT / "data" / "splits")
_DEFAULT_LANDMARKS_DIR = str(_REPO_ROOT / "data" / "landmarks")
_DEFAULT_SUMMARY_DIR   = str(_REPO_ROOT / "data")
_DEFAULT_LOG_DIR       = str(_REPO_ROOT / "logs")

_VALID_SPLITS = ("train", "val", "test", "all")
_LOG_INTERVAL = 50    # clips between progress log lines

# Must match extractor.py v1.1 default: min(3, available_clips) per sign
_SAMPLE_CLIPS_PER_SIGN = 3

_SEED = 42

# Extractor defaults kept in sync with extractor.py module constants
_DEFAULT_MAX_MISSING_FRAME_PCT = 0.30
_DEFAULT_MIN_DETECTION_CONF    = 0.5
_DEFAULT_MIN_TRACKING_CONF     = 0.5

# Health check thresholds (calibrated for v1.1 missing_pct denominator)
# With correctly-computed missing_pct (over successfully-decoded frames only),
# these are tighter than the v1.0 values that were inflated by codec errors.
_HEALTH_POLICY_SKIP_RATE_WARN  = 0.10   # >10% skipped by missing-rate policy
_HEALTH_ERROR_RATE_WARN        = 0.05   # >5% clips with extraction errors
_HEALTH_GLOBAL_MISSING_RATE    = 0.15   # >15% of frames missing both hands

# Characters unsafe as filesystem path components on any OS
_UNSAFE_PATH_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


# ---------------------------------------------------------------------------
# Filesystem safety helper
# ---------------------------------------------------------------------------

def _sanitize_path_component(name: str) -> str:
    """
    Replace characters that are unsafe in a filesystem path component.

    WLASL sign labels are plain ASCII words; this is a safety net for edge
    cases and Windows compatibility rather than routine processing.

    Parameters
    ----------
    name : str
        Raw path component (sign label or video_id).

    Returns
    -------
    str
        Safe path component, guaranteed non-empty.

    Examples
    --------
    >>> _sanitize_path_component("before")
    'before'
    >>> _sanitize_path_component("sign/with:special*chars")
    'sign_with_special_chars'
    """
    safe = _UNSAFE_PATH_CHARS.sub("_", name)
    safe = safe.strip(". ")
    safe = re.sub(r"_+", "_", safe)
    return safe or "unknown"


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_landmark_extraction.py",
        description=(
            "Stage 3: MediaPipe Holistic landmark extraction for WLASL gesture recognition.\n\n"
            "Reads split CSVs produced by Stage 1, runs MediaPipe Holistic on every\n"
            "video clip, and writes (num_frames, 225) float32 .npy arrays to\n"
            "data/landmarks/<split>/<sign>/<video_id>.npy.\n\n"
            "Aligned with extractor.py schema version 1.1:\n"
            "  - decode_failure_frames tracked separately from detection failures\n"
            "  - missing_pct denominator is successfully_decoded_frames (not total)\n"
            "  - sidecar .meta.json schema version field enforced on cache reads\n\n"
            "Always run --sample-only first to validate the extractor before\n"
            "committing to the full 30–90 minute extraction."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Stage 2 validation gate (3 clips/sign, all splits — ~2-5 minutes)
  python pipelines/run_landmark_extraction.py --sample-only

  # Full extraction, all splits (30-90 minutes)
  python pipelines/run_landmark_extraction.py --split all

  # Training split only
  python pipelines/run_landmark_extraction.py --split train

  # Force re-extraction (overwrite existing .npy + .meta.json)
  python pipelines/run_landmark_extraction.py --split all --force

  # Resume and verify previously cached files
  python pipelines/run_landmark_extraction.py --split all --verify-existing

  # Dry run — validate inputs and log plan, write nothing
  python pipelines/run_landmark_extraction.py --dry-run

  # Verbose debug output
  python pipelines/run_landmark_extraction.py --sample-only --verbose

  # Override MediaPipe thresholds (lower confidence for difficult clips)
  python pipelines/run_landmark_extraction.py --split all \\
      --max-missing-frame-pct 0.40 --min-detection-confidence 0.4

Exit codes: 0=success, 1=input error/no clips, 2=unexpected failure
        """,
    )

    # ----------------------------------------------------------------
    # Paths
    # ----------------------------------------------------------------
    parser.add_argument(
        "--splits-dir",
        default=_DEFAULT_SPLITS_DIR,
        metavar="DIR",
        help=f"Directory containing train/val/test CSVs (default: {_DEFAULT_SPLITS_DIR})",
    )
    parser.add_argument(
        "--landmarks-dir",
        default=_DEFAULT_LANDMARKS_DIR,
        metavar="DIR",
        help=(
            f"Root output directory for .npy landmark files "
            f"(default: {_DEFAULT_LANDMARKS_DIR})"
        ),
    )
    parser.add_argument(
        "--summary-dir",
        default=_DEFAULT_SUMMARY_DIR,
        metavar="DIR",
        help=(
            "Directory for pipeline-level extraction summary JSON files. "
            "Two files are written: preprocessing_summary_latest.json "
            "(current run, always overwritten) and "
            "preprocessing_summary_history.json (append-only audit log). "
            f"(default: {_DEFAULT_SUMMARY_DIR})"
        ),
    )
    parser.add_argument(
        "--log-dir",
        default=_DEFAULT_LOG_DIR,
        metavar="DIR",
        help=f"Directory for structured log files (default: {_DEFAULT_LOG_DIR})",
    )

    # ----------------------------------------------------------------
    # Run mode (mutually exclusive)
    # ----------------------------------------------------------------
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--sample-only",
        action="store_true",
        help=(
            f"Process only {_SAMPLE_CLIPS_PER_SIGN} clip(s) per sign per split "
            "(alphabetically first by video_id). "
            "Use as the Stage 2 validation gate before full extraction. "
            "Takes ~2–5 minutes and validates the extractor output before "
            "committing to the full 30–90 minute run."
        ),
    )
    mode_group.add_argument(
        "--split",
        choices=_VALID_SPLITS,
        default="all",
        metavar="{train,val,test,all}",
        help="Which split(s) to process (default: all).",
    )

    # ----------------------------------------------------------------
    # Behaviour flags
    # ----------------------------------------------------------------
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-extract and overwrite .npy + .meta.json files that already exist. "
            "Without this flag, existing valid files are skipped (resumable by default). "
            "Implies --verify-existing is redundant (will be warned)."
        ),
    )
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help=(
            "Spot-check each cached .npy file before trusting it: "
            "validates ndim, shape, dtype, and first-row finiteness. "
            "Corrupt or schema-mismatched files are automatically reprocessed. "
            "Use after a filesystem incident or when resuming an interrupted run."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate all inputs and log the extraction plan without writing "
            "any files or calling MediaPipe. Useful for verifying clip counts "
            "and output paths before committing to a long run."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging to console (file always gets DEBUG).",
    )

    # ----------------------------------------------------------------
    # Extractor configuration
    # ----------------------------------------------------------------
    parser.add_argument(
        "--max-missing-frame-pct",
        type=float,
        default=_DEFAULT_MAX_MISSING_FRAME_PCT,
        metavar="RATIO",
        help=(
            "Skip a clip if the fraction of *successfully decoded* frames where "
            "both hands are absent exceeds this threshold "
            f"(default: {_DEFAULT_MAX_MISSING_FRAME_PCT}). "
            "Note: in extractor v1.1, decode failures (OpenCV codec errors) are "
            "excluded from this denominator, giving an accurate MediaPipe-only "
            "missing rate."
        ),
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=_DEFAULT_MIN_DETECTION_CONF,
        metavar="CONF",
        help=(
            f"MediaPipe Holistic minimum detection confidence "
            f"(default: {_DEFAULT_MIN_DETECTION_CONF}). "
            "Must be identical between extraction and inference (Stage 7)."
        ),
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=_DEFAULT_MIN_TRACKING_CONF,
        metavar="CONF",
        help=(
            f"MediaPipe Holistic minimum tracking confidence "
            f"(default: {_DEFAULT_MIN_TRACKING_CONF})."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_SEED,
        metavar="N",
        help=(
            f"Random seed for global reproducibility (default: {_SEED}). "
            "Sample clip selection within --sample-only mode is alphabetically "
            "deterministic and does not consume randomness."
        ),
    )

    return parser


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_args(args: argparse.Namespace, logger) -> bool:
    """
    Validate all CLI argument constraints before any work begins.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments from ``_build_parser()``.
    logger
        Active logger instance.

    Returns
    -------
    bool
        True if all constraints pass; False signals the caller to exit(1).
    """
    valid = True

    if not (0.0 < args.max_missing_frame_pct <= 1.0):
        logger.error(
            f"--max-missing-frame-pct must be in (0, 1]. "
            f"Got: {args.max_missing_frame_pct}",
            extra={"stage": "extraction"},
        )
        valid = False

    if not (0.0 < args.min_detection_confidence <= 1.0):
        logger.error(
            f"--min-detection-confidence must be in (0, 1]. "
            f"Got: {args.min_detection_confidence}",
            extra={"stage": "extraction"},
        )
        valid = False

    if not (0.0 < args.min_tracking_confidence <= 1.0):
        logger.error(
            f"--min-tracking-confidence must be in (0, 1]. "
            f"Got: {args.min_tracking_confidence}",
            extra={"stage": "extraction"},
        )
        valid = False

    if args.force and args.verify_existing:
        logger.warning(
            "--force re-extracts all clips; --verify-existing is redundant "
            "and will be ignored.",
            extra={"stage": "extraction"},
        )
        # Not a hard failure — just warn.

    return valid


# ---------------------------------------------------------------------------
# Split CSV loading
# ---------------------------------------------------------------------------

def _load_split_df(splits_dir: str, split_name: str, logger) -> Optional[pd.DataFrame]:
    """
    Load and schema-validate a single split CSV produced by Stage 1.

    Required columns: video_id, sign_label, class_idx, signer_id, video_path.

    Parameters
    ----------
    splits_dir : str
        Directory containing the split CSVs.
    split_name : str
        One of "train", "val", "test".
    logger
        Active logger.

    Returns
    -------
    pd.DataFrame | None
        Validated DataFrame, or None if the file is missing or malformed.
    """
    csv_path = Path(splits_dir) / f"{split_name}.csv"

    if not csv_path.exists():
        logger.error(
            f"Split CSV not found: {csv_path}. "
            "Run Stage 1 (run_preprocessing.py) first to generate split files.",
            extra={"stage": "extraction"},
        )
        return None

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        logger.error(
            f"Failed to read {csv_path}: {type(exc).__name__}: {exc}",
            extra={"stage": "extraction"},
        )
        return None

    required_cols = {"video_id", "sign_label", "class_idx", "signer_id", "video_path"}
    missing_cols  = required_cols - set(df.columns)
    if missing_cols:
        logger.error(
            f"Split CSV {csv_path} is missing required columns: {missing_cols}. "
            "Re-run Stage 1 to regenerate the splits.",
            extra={"stage": "extraction"},
        )
        return None

    logger.info(
        f"Loaded split: {split_name} | clips={len(df)} | "
        f"signs={df['sign_label'].nunique()} | "
        f"signers={df['signer_id'].nunique()}",
        extra={"stage": "extraction"},
    )
    return df


# ---------------------------------------------------------------------------
# Clip collection
# ---------------------------------------------------------------------------

def _collect_clips(
    splits_dir: str,
    split_arg: str,
    sample_only: bool,
    logger,
) -> list[dict[str, Any]]:
    """
    Build the ordered list of clips to process.

    Each entry is a dict with keys:
        video_id, sign_label, safe_sign_label, class_idx, signer_id,
        split, video_path

    In ``--sample-only`` mode, exactly ``_SAMPLE_CLIPS_PER_SIGN`` clip(s) per
    sign per split are selected. Selection is alphabetical by video_id within
    each sign group — fully deterministic without consuming any randomness.

    Parameters
    ----------
    splits_dir : str
        Path to the splits directory.
    split_arg : str
        "train", "val", "test", or "all".
    sample_only : bool
        If True, restrict to ``_SAMPLE_CLIPS_PER_SIGN`` clips per sign per split.
    logger
        Active logger.

    Returns
    -------
    list[dict]
        Ordered clip records. Empty list signals a loading error.
    """
    split_names = ["train", "val", "test"] if split_arg == "all" else [split_arg]
    clips: list[dict[str, Any]] = []

    for split_name in split_names:
        df = _load_split_df(splits_dir, split_name, logger)
        if df is None:
            logger.error(
                f"Cannot continue — split '{split_name}' could not be loaded.",
                extra={"stage": "extraction"},
            )
            return []

        if sample_only:
            # Alphabetical sort by video_id before groupby guarantees a fixed,
            # reproducible selection independent of CSV row order.
            df = (
                df.sort_values("video_id")
                .groupby("sign_label", sort=True)
                .head(_SAMPLE_CLIPS_PER_SIGN)
                .reset_index(drop=True)
            )
            n_signs = df["sign_label"].nunique()
            logger.info(
                f"Sample mode: selected {len(df)} clips from '{split_name}' "
                f"(up to {_SAMPLE_CLIPS_PER_SIGN} per sign, "
                f"{n_signs} signs represented)",
                extra={"stage": "extraction"},
            )

        for _, row in df.iterrows():
            sign_label = str(row["sign_label"])
            clips.append({
                "video_id":        str(row["video_id"]),
                "sign_label":      sign_label,
                "safe_sign_label": _sanitize_path_component(sign_label),
                "class_idx":       int(row["class_idx"]),
                "signer_id":       int(row["signer_id"]),
                "split":           split_name,
                "video_path":      str(row["video_path"]),
            })

    n_signs_total = len({c["sign_label"] for c in clips})
    logger.info(
        f"Total clips queued: {len(clips)} | "
        f"splits={split_names} | signs={n_signs_total} | "
        f"sample_only={sample_only}",
        extra={"stage": "extraction"},
    )
    return clips


# ---------------------------------------------------------------------------
# Output path helpers
# ---------------------------------------------------------------------------

def _get_output_path(
    landmarks_dir: str,
    split_name: str,
    safe_sign_label: str,
    video_id: str,
) -> Path:
    """
    Canonical output .npy path for one clip.

    Schema: <landmarks_dir>/<split>/<safe_sign_label>/<video_id>.npy

    Parameters
    ----------
    landmarks_dir : str
        Root landmarks directory.
    split_name : str
        "train", "val", or "test".
    safe_sign_label : str
        Filesystem-safe sign label.
    video_id : str
        WLASL video identifier.

    Returns
    -------
    Path
        Absolute output .npy path.
    """
    return Path(landmarks_dir) / split_name / safe_sign_label / f"{video_id}.npy"


def _get_sidecar_path(npy_path: Path) -> Path:
    """Return the .meta.json sidecar path for a given .npy file."""
    return npy_path.with_suffix(".meta.json")


# ---------------------------------------------------------------------------
# Video path resolution
# ---------------------------------------------------------------------------

def _resolve_video_path(video_path: str, video_id: str, logger) -> Optional[Path]:
    """
    Resolve and validate a video path from the split CSV.

    Handles both relative (repo-root-relative) and absolute paths.

    Parameters
    ----------
    video_path : str
        Raw path string from the split CSV.
    video_id : str
        WLASL identifier (for logging).
    logger
        Active logger.

    Returns
    -------
    Path | None
        Resolved absolute path if the file exists on disk, else None.
    """
    if not video_path or video_path.lower() in ("nan", "none", ""):
        logger.warning(
            f"Empty/null video_path for video_id={video_id} — cannot process.",
            extra={"stage": "extraction", "video_id": video_id},
        )
        return None

    p = Path(video_path)
    resolved = p if p.is_absolute() else (_REPO_ROOT / p)

    if not resolved.exists():
        logger.warning(
            f"Video file not found on disk: {resolved} | video_id={video_id}. "
            "Check the raw_dir path used in the Stage 1 inventory build.",
            extra={"stage": "extraction", "video_id": video_id},
        )
        return None

    return resolved


# ---------------------------------------------------------------------------
# .npy verification (independent of extractor's cache check)
# ---------------------------------------------------------------------------

def _verify_npy_file(
    npy_path: Path,
    video_id: str,
    logger,
    full_check: bool = False,
) -> bool:
    """
    Independently validate a .npy landmark array file.

    Always checks:
    - Loadable by numpy (no pickle)
    - ndim == 2
    - shape[1] == FEATURE_SIZE (225)
    - dtype == float32

    With ``full_check=True`` (used after fresh writes):
    - All values are finite (full scan across every row)

    With ``full_check=False`` (fast spot-check on cache-hit verification):
    - Only the first row is checked for finiteness

    Note: This is complementary to the extractor's own ``_try_load_cached``
    validation. Running both on a freshly extracted file provides defence-in-
    depth against partial writes or memory corruption.

    Parameters
    ----------
    npy_path : Path
        Path to the .npy file.
    video_id : str
        WLASL identifier (for logging).
    logger
        Active logger.
    full_check : bool
        If True, scan all values. If False, spot-check only the first row.

    Returns
    -------
    bool
        True if all checks pass.
    """
    try:
        arr = np.load(str(npy_path), mmap_mode="r", allow_pickle=False)
    except Exception as exc:
        logger.error(
            f"Cannot load .npy: {npy_path.name} | video_id={video_id} | "
            f"{type(exc).__name__}: {exc}",
            extra={"stage": "extraction", "video_id": video_id},
        )
        return False

    if arr.ndim != 2:
        logger.error(
            f"ndim error: expected 2D array, got shape={arr.shape} | "
            f"video_id={video_id} | path={npy_path}",
            extra={"stage": "extraction", "video_id": video_id},
        )
        return False

    if arr.shape[1] != FEATURE_SIZE:
        logger.error(
            f"Feature-size error: expected {FEATURE_SIZE} columns, "
            f"got {arr.shape[1]} | video_id={video_id} | path={npy_path}",
            extra={"stage": "extraction", "video_id": video_id},
        )
        return False

    if arr.dtype != np.float32:
        logger.warning(
            f"Unexpected dtype: expected float32, got {arr.dtype} | "
            f"video_id={video_id} | path={npy_path}. "
            "The extractor guarantees float32 — this suggests a foreign file.",
            extra={"stage": "extraction", "video_id": video_id},
        )
        # Dtype mismatch is a warning, not a hard failure here. The extractor
        # raises on dtype mismatch during its own cache validation; we simply
        # flag it in the pipeline layer for visibility.

    rows_to_check = arr if full_check else (arr[:1] if arr.shape[0] > 0 else arr)
    if rows_to_check.size > 0 and not np.isfinite(rows_to_check).all():
        logger.error(
            f"Non-finite values (NaN/Inf) detected | "
            f"video_id={video_id} | path={npy_path}",
            extra={"stage": "extraction", "video_id": video_id},
        )
        return False

    return True


# ---------------------------------------------------------------------------
# Sidecar metadata reader (pipeline-layer, reads extractor sidecar format)
# ---------------------------------------------------------------------------

def _read_sidecar(npy_path: Path) -> Optional[dict[str, Any]]:
    """
    Load extractor sidecar metadata for a cached .npy file.

    Reads the .meta.json written by ``LandmarkExtractor._write_meta()``.
    Returns None if the file is absent, unreadable, or has a schema version
    that does not match ``EXTRACTOR_SCHEMA_VERSION``.

    Importantly, schema version mismatches are surfaced as a warning here
    so the pipeline layer can decide whether to reprocess. The extractor
    itself enforces this check during its own cache validation; this
    function provides a parallel read for the pipeline's statistics
    accumulation logic.

    Parameters
    ----------
    npy_path : Path
        Path to the .npy file.

    Returns
    -------
    dict | None
        Parsed sidecar JSON with all ExtractionResult fields, or None.
    """
    meta_path = _get_sidecar_path(npy_path)
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        stored_version = meta.get("schema_version", "")
        if stored_version != EXTRACTOR_SCHEMA_VERSION:
            # Schema version mismatch: the extractor will reprocess this
            # clip on its next cache check, so we treat it as a cache miss
            # for statistics purposes too.
            return None
        return meta
    except (json.JSONDecodeError, OSError):
        return None


# ---------------------------------------------------------------------------
# Dry-run plan reporter
# ---------------------------------------------------------------------------

def _report_dry_run_plan(
    clips: list[dict[str, Any]],
    landmarks_dir: str,
    force: bool,
    logger,
) -> None:
    """
    Log the extraction plan in dry-run mode without doing any real work.

    Parameters
    ----------
    clips : list[dict]
        Full clip list as produced by ``_collect_clips``.
    landmarks_dir : str
        Root landmarks directory.
    force : bool
        Whether ``--force`` was passed.
    logger
        Active logger.
    """
    would_extract     = 0
    would_skip_cached = 0
    by_split: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "cached": 0})

    for clip in clips:
        npy_path = _get_output_path(
            landmarks_dir,
            clip["split"],
            clip["safe_sign_label"],
            clip["video_id"],
        )
        by_split[clip["split"]]["total"] += 1
        if npy_path.exists() and not force:
            would_skip_cached += 1
            by_split[clip["split"]]["cached"] += 1
        else:
            would_extract += 1

    logger.info("[DRY RUN] Extraction plan:", extra={"stage": "extraction"})
    logger.info(
        f"  Total queued  : {len(clips)}",
        extra={"stage": "extraction"},
    )
    logger.info(
        f"  Would extract : {would_extract}",
        extra={"stage": "extraction"},
    )
    logger.info(
        f"  Would skip    : {would_skip_cached} "
        "(already cached, --force not set)",
        extra={"stage": "extraction"},
    )
    for split_name, counts in sorted(by_split.items()):
        cached = counts["cached"]
        to_do  = counts["total"] - cached
        logger.info(
            f"  {split_name:6s}: {counts['total']:4d} total | "
            f"{cached:3d} already cached | {to_do:3d} would extract",
            extra={"stage": "extraction"},
        )


# ---------------------------------------------------------------------------
# Per-run statistics accumulator
# ---------------------------------------------------------------------------

class _RunStats:
    """
    Accumulates per-clip and aggregate statistics for the current pipeline run.

    Aligned with extractor.py v1.1:
      - ``decode_failure_frames`` is tracked separately from detection failures
      - ``missing_pct`` stored per-clip is over successfully-decoded frames
      - Cache-hit status is ``"cached"`` not ``"skipped"``

    Attributes
    ----------
    n_queued : int
        Total clips submitted to the extraction loop.
    n_extracted : int
        Clips freshly extracted this run.
    n_skipped_cached : int
        Clips skipped because a valid .npy + .meta.json already existed.
    n_skipped_policy : int
        Clips skipped due to the missing-frame-rate threshold.
    n_skipped_error : int
        Clips skipped due to video read failure or MediaPipe exception.
    n_dry_run : int
        Clips processed in dry-run mode (no actual work done).
    total_frames : int
        Sum of frame counts across freshly extracted clips.
    total_missing_both : int
        Sum of missing-both-hands frame counts across extracted clips.
        Computed over successfully-decoded frames (v1.1 semantics).
    total_decode_failures : int
        Sum of decode_failure_frames across extracted clips.
    total_proc_sec : float
        Total wall-clock processing time for extracted clips.
    """

    def __init__(self, run_id: str, max_missing_frame_pct: float) -> None:
        self._run_id          = run_id
        self._started_utc     = datetime.now(timezone.utc).isoformat()
        self._max_missing_pct = max_missing_frame_pct

        self._records: list[dict[str, Any]] = []

        # Counters
        self.n_queued             = 0
        self.n_extracted          = 0
        self.n_skipped_cached     = 0
        self.n_skipped_policy     = 0
        self.n_skipped_error      = 0
        self.n_dry_run            = 0
        self.total_frames         = 0
        self.total_missing_both   = 0   # v1.1: over successfully-decoded frames
        self.total_decode_failures = 0  # v1.1: separate from detection failures
        self.total_proc_sec       = 0.0

        # Per-sign breakdown
        self._sign_frames:        dict[str, int] = defaultdict(int)
        self._sign_missing:       dict[str, int] = defaultdict(int)
        self._sign_extracted:     dict[str, int] = defaultdict(int)
        self._sign_skipped:       dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Recording methods — one per outcome type
    # ------------------------------------------------------------------

    def record_extracted(
        self,
        clip: dict[str, Any],
        result: ExtractionResult,
        proc_sec: float,
    ) -> None:
        """Record a freshly extracted clip (status='extracted')."""
        sign = clip["sign_label"]
        self.n_extracted           += 1
        self.total_frames          += result.num_frames
        self.total_missing_both    += result.missing_both_hands_frames
        self.total_decode_failures += result.decode_failure_frames   # v1.1
        self.total_proc_sec        += proc_sec
        self._sign_frames[sign]    += result.num_frames
        self._sign_missing[sign]   += result.missing_both_hands_frames
        self._sign_extracted[sign] += 1

        self._records.append({
            "video_id":              clip["video_id"],
            "sign_label":            sign,
            "class_idx":             clip["class_idx"],
            "signer_id":             clip["signer_id"],
            "split":                 clip["split"],
            "video_path":            clip["video_path"],
            "output_path":           result.output_path,
            "outcome":               "extracted",
            "proc_sec":              round(proc_sec, 4),
            "n_frames":              result.num_frames,
            "decode_failure_frames": result.decode_failure_frames,       # v1.1
            "n_missing_both":        result.missing_both_hands_frames,
            "missing_pct":           round(result.missing_pct, 4),       # v1.1 denominator
        })

    def record_cached(
        self,
        clip: dict[str, Any],
        output_path: str,
        n_frames: int = 0,
        missing_pct: float = 0.0,
        missing_both: int = 0,
        decode_failure_frames: int = 0,   # v1.1
    ) -> None:
        """
        Record a cache-hit clip (status='cached').

        Statistics are restored from the v1.1 .meta.json sidecar when
        available so aggregate missing-rate figures remain accurate even for
        clips that were not processed in this run.
        """
        sign = clip["sign_label"]
        self.n_skipped_cached      += 1
        self._sign_frames[sign]    += n_frames
        self._sign_missing[sign]   += missing_both
        self._sign_extracted[sign] += 1   # counts toward usable total

        self._records.append({
            "video_id":              clip["video_id"],
            "sign_label":            sign,
            "class_idx":             clip["class_idx"],
            "signer_id":             clip["signer_id"],
            "split":                 clip["split"],
            "video_path":            clip["video_path"],
            "output_path":           output_path,
            "outcome":               "cached",
            "proc_sec":              0.0,
            "n_frames":              n_frames,
            "decode_failure_frames": decode_failure_frames,  # v1.1
            "n_missing_both":        missing_both,
            "missing_pct":           round(missing_pct, 4),
        })

    def record_skipped_policy(
        self,
        clip: dict[str, Any],
        result: ExtractionResult,
        proc_sec: float,
    ) -> None:
        """Record a clip skipped by the missing-frame-pct policy."""
        sign = clip["sign_label"]
        self.n_skipped_policy      += 1
        self._sign_skipped[sign]   += 1
        self.total_proc_sec        += proc_sec

        self._records.append({
            "video_id":              clip["video_id"],
            "sign_label":            sign,
            "class_idx":             clip["class_idx"],
            "signer_id":             clip["signer_id"],
            "split":                 clip["split"],
            "video_path":            clip["video_path"],
            "output_path":           "",
            "outcome":               "skipped_policy",
            "proc_sec":              round(proc_sec, 4),
            "n_frames":              result.num_frames,
            "decode_failure_frames": result.decode_failure_frames,  # v1.1
            "n_missing_both":        result.missing_both_hands_frames,
            "missing_pct":           round(result.missing_pct, 4),
            "skip_reason":           result.skip_reason,
        })

    def record_error(
        self,
        clip: dict[str, Any],
        error_msg: str,
        proc_sec: float = 0.0,
    ) -> None:
        """Record a clip that failed with an exception or missing video file."""
        sign = clip["sign_label"]
        self.n_skipped_error     += 1
        self._sign_skipped[sign] += 1
        self.total_proc_sec      += proc_sec

        self._records.append({
            "video_id":              clip["video_id"],
            "sign_label":            sign,
            "class_idx":             clip["class_idx"],
            "signer_id":             clip["signer_id"],
            "split":                 clip["split"],
            "video_path":            clip["video_path"],
            "output_path":           "",
            "outcome":               "error",
            "proc_sec":              round(proc_sec, 4),
            "n_frames":              0,
            "decode_failure_frames": 0,
            "n_missing_both":        0,
            "missing_pct":           0.0,
            "error_message":         error_msg,
        })

    def record_dry_run(self, clip: dict[str, Any]) -> None:
        """Record a clip in dry-run mode."""
        self.n_dry_run += 1
        self._records.append({
            "video_id":  clip["video_id"],
            "sign_label": clip["sign_label"],
            "class_idx":  clip["class_idx"],
            "signer_id":  clip["signer_id"],
            "split":      clip["split"],
            "video_path": clip["video_path"],
            "output_path": "",
            "outcome":    "dry_run",
            "proc_sec":   0.0,
        })

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def n_usable(self) -> int:
        """Clips usable for training: freshly extracted + cache-hit."""
        return self.n_extracted + self.n_skipped_cached

    @property
    def global_missing_rate(self) -> float:
        """
        Global both-hands-absent rate over successfully-decoded frames.

        Aligns with v1.1 semantics: decode_failure_frames are excluded from
        the denominator, so this reflects genuine MediaPipe detection quality.
        """
        denom = self.total_frames - self.total_decode_failures
        return self.total_missing_both / denom if denom > 0 else 0.0

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self, status: str = "completed") -> dict[str, Any]:
        """
        Serialise aggregate and per-clip statistics to a JSON-ready dict.

        Parameters
        ----------
        status : str
            "completed" or "PARTIAL_INTERRUPTED".

        Returns
        -------
        dict
            Full summary payload for ``_write_pipeline_summary()``.
        """
        n_eff = max(self.n_queued, 1)

        per_sign: dict[str, Any] = {}
        all_signs = (
            set(self._sign_frames)
            | set(self._sign_extracted)
            | set(self._sign_skipped)
        )
        for sign in sorted(all_signs):
            total_f = self._sign_frames.get(sign, 0)
            miss_f  = self._sign_missing.get(sign, 0)
            per_sign[sign] = {
                "usable":         self._sign_extracted.get(sign, 0),
                "skipped":        self._sign_skipped.get(sign, 0),
                "total_frames":   total_f,
                "missing_frames": miss_f,
                "missing_rate":   round(miss_f / total_f, 4) if total_f > 0 else 0.0,
            }

        return {
            "_run_metadata": {
                "run_id":                self._run_id,
                "status":                status,
                "started_utc":           self._started_utc,
                "completed_utc":         datetime.now(timezone.utc).isoformat(),
                "extractor_schema":      EXTRACTOR_SCHEMA_VERSION,
                "max_missing_frame_pct": self._max_missing_pct,
            },
            "aggregate": {
                "n_queued":               self.n_queued,
                "n_extracted":            self.n_extracted,
                "n_cached":               self.n_skipped_cached,
                "n_skipped_policy":       self.n_skipped_policy,
                "n_skipped_error":        self.n_skipped_error,
                "n_usable":               self.n_usable,
                "policy_skip_rate":       round(self.n_skipped_policy / n_eff, 4),
                "error_rate":             round(self.n_skipped_error   / n_eff, 4),
                "total_frames":           self.total_frames,
                "total_decode_failures":  self.total_decode_failures,  # v1.1
                "total_missing_both":     self.total_missing_both,
                "global_missing_rate":    round(self.global_missing_rate, 4),  # v1.1
                "total_proc_sec":         round(self.total_proc_sec, 1),
                "mean_proc_sec_per_clip": round(
                    self.total_proc_sec / max(self.n_extracted, 1), 3
                ),
            },
            "per_sign": per_sign,
            "per_clip": self._records,
        }


# ---------------------------------------------------------------------------
# Pipeline-level summary writer
# ---------------------------------------------------------------------------

def _write_pipeline_summary(
    summary: dict[str, Any],
    summary_dir: str,
    logger,
) -> None:
    """
    Write the current run's pipeline-level summary to two JSON files.

    Files written:
    - ``preprocessing_summary_latest.json``: full payload (overwritten each run)
    - ``preprocessing_summary_history.json``: compact entry appended per run
      (``per_clip`` list excluded to keep file size manageable)

    The ``_latest`` file is what Notebook 02 reads for the missing-landmark
    analysis. The ``_history`` file is the audit trail.

    Parameters
    ----------
    summary : dict
        Current run summary from ``_RunStats.to_dict()``.
    summary_dir : str
        Directory to write both files.
    logger
        Active logger.
    """
    out_dir = Path(summary_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    latest_path  = out_dir / "preprocessing_summary_latest.json"
    history_path = out_dir / "preprocessing_summary_history.json"

    # Full payload — always overwritten
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    # Compact history entry — per_clip excluded for file-size control
    compact = {k: v for k, v in summary.items() if k != "per_clip"}

    existing_runs: list[dict[str, Any]] = []
    if history_path.exists():
        try:
            with open(history_path, encoding="utf-8") as f:
                data = json.load(f)
            existing_runs = data if isinstance(data, list) else [data]
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                f"Could not read history file (starting fresh): {exc}",
                extra={"stage": "extraction"},
            )

    existing_runs.append(compact)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(existing_runs, f, indent=2, default=str)

    n_clips = len(summary.get("per_clip", []))
    logger.info(
        f"Pipeline summary written | "
        f"latest={latest_path} ({latest_path.stat().st_size / 1024:.1f} KB) | "
        f"history={history_path} ({len(existing_runs)} runs) | "
        f"clips_in_latest={n_clips}",
        extra={"stage": "extraction"},
    )


# ---------------------------------------------------------------------------
# Core extraction loop
# ---------------------------------------------------------------------------

def _run_extraction_loop(
    clips: list[dict[str, Any]],
    extractor: Optional[LandmarkExtractor],
    landmarks_dir: str,
    run_stats: _RunStats,
    force: bool,
    verify_existing: bool,
    dry_run: bool,
    logger,
) -> None:
    """
    Iterate over clip records, delegate to the extractor, and accumulate stats.

    Design decisions:
    - The extractor owns all .npy write logic. This loop drives clip iteration
      and statistics accumulation only.
    - ETA is computed from a fixed pre-loop count so it never drifts as files
      are written during the run.
    - Cache-hit routing uses ``result.status == "cached"`` (v1.1 extractor
      returns this; the old "skipped" routing from v1.0 is removed).
    - Post-write verification (``_verify_npy_file``) is an independent check
      applied after each fresh extraction as defence-in-depth.
    - Each clip is isolated: any exception is caught, logged, and counted as
      an error without aborting the rest of the run.
    - ``--verify-existing`` triggers a spot-check on cache-hit files; corrupt
      ones are deleted so the extractor reprocesses them on the same call.
    - ``extractor`` is None only when ``dry_run=True``; this is asserted before
      the loop begins.

    Parameters
    ----------
    clips : list[dict]
        Ordered clip records from ``_collect_clips()``.
    extractor : LandmarkExtractor | None
        Initialised extractor. Must be non-None when dry_run=False.
    landmarks_dir : str
        Root output directory for .npy files.
    run_stats : _RunStats
        Statistics accumulator.
    force : bool
        Re-extract even if a valid cached .npy + .meta.json exists.
    verify_existing : bool
        Spot-check cached .npy files before trusting them.
    dry_run : bool
        Log plan only — never call MediaPipe or write files.
    logger
        Active logger.
    """
    run_stats.n_queued = len(clips)
    loop_start = time.time()

    # ----------------------------------------------------------------
    # Dry-run short-circuit
    # ----------------------------------------------------------------
    if dry_run:
        _report_dry_run_plan(clips, landmarks_dir, force, logger)
        for clip in clips:
            run_stats.record_dry_run(clip)
        return

    if extractor is None:
        raise RuntimeError(
            "_run_extraction_loop: extractor must be non-None when dry_run=False. "
            "This is a bug in the pipeline initialisation logic."
        )

    # Pre-compute how many clips actually need extraction (fixed before loop).
    # This count is the ETA denominator so ETA never drifts as files are written.
    n_to_process_initially = sum(
        1 for c in clips
        if force or not _get_output_path(
            landmarks_dir, c["split"], c["safe_sign_label"], c["video_id"]
        ).exists()
    )
    logger.info(
        f"Clips requiring extraction: {n_to_process_initially} | "
        f"already cached: {len(clips) - n_to_process_initially}",
        extra={"stage": "extraction"},
    )

    n_newly_processed = 0   # extraction calls made (for ETA denominator)

    for i, clip in enumerate(clips):
        video_id        = clip["video_id"]
        sign_label      = clip["sign_label"]
        safe_sign_label = clip["safe_sign_label"]
        split_name      = clip["split"]
        video_path_str  = clip["video_path"]

        npy_path = _get_output_path(
            landmarks_dir, split_name, safe_sign_label, video_id
        )

        # ----------------------------------------------------------------
        # Fast path: cache-hit check before calling the extractor.
        # The extractor performs its own (more thorough) cache check,
        # but this fast path avoids the function call overhead for the
        # majority of clips on subsequent runs.
        # ----------------------------------------------------------------
        if npy_path.exists() and not force:
            if verify_existing:
                ok = _verify_npy_file(npy_path, video_id, logger, full_check=False)
                if not ok:
                    logger.info(
                        f"Cached file failed spot-check — deleting for reprocessing: "
                        f"{npy_path.name} | video_id={video_id}",
                        extra={"stage": "extraction", "video_id": video_id},
                    )
                    try:
                        npy_path.unlink(missing_ok=True)
                        _get_sidecar_path(npy_path).unlink(missing_ok=True)
                    except OSError as exc:
                        logger.warning(
                            f"Could not delete corrupt cached file {npy_path}: {exc}",
                            extra={"stage": "extraction"},
                        )
                    # Fall through to extraction below
                else:
                    meta = _read_sidecar(npy_path)
                    run_stats.record_cached(
                        clip,
                        output_path=str(npy_path),
                        n_frames=             meta.get("num_frames",               0)   if meta else 0,
                        missing_pct=          meta.get("missing_pct",              0.0) if meta else 0.0,
                        missing_both=         meta.get("missing_both_hands_frames",0)   if meta else 0,
                        decode_failure_frames=meta.get("decode_failure_frames",    0)   if meta else 0,
                    )
                    logger.debug(
                        f"Cache hit (verified): {video_id} ({sign_label})",
                        extra={"stage": "extraction", "video_id": video_id},
                    )
                    continue
            else:
                meta = _read_sidecar(npy_path)
                run_stats.record_cached(
                    clip,
                    output_path=str(npy_path),
                    n_frames=             meta.get("num_frames",               0)   if meta else 0,
                    missing_pct=          meta.get("missing_pct",              0.0) if meta else 0.0,
                    missing_both=         meta.get("missing_both_hands_frames",0)   if meta else 0,
                    decode_failure_frames=meta.get("decode_failure_frames",    0)   if meta else 0,
                )
                logger.debug(
                    f"Cache hit: {video_id} ({sign_label})",
                    extra={"stage": "extraction", "video_id": video_id},
                )
                continue

        # ----------------------------------------------------------------
        # Validate video file on disk before calling the extractor.
        # This surfaces missing-file errors with a clean warning rather
        # than letting the extractor raise a RuntimeError.
        # ----------------------------------------------------------------
        resolved_video_path = _resolve_video_path(video_path_str, video_id, logger)
        if resolved_video_path is None:
            run_stats.record_error(
                clip, error_msg="video_file_not_found"
            )
            n_newly_processed += 1
            continue

        # ----------------------------------------------------------------
        # Delegate extraction to LandmarkExtractor.
        # The extractor handles: MediaPipe processing, decode-failure
        # tracking (v1.1), skip-policy application, .npy write, and
        # .meta.json sidecar write.
        # ----------------------------------------------------------------
        clip_start = time.time()
        try:
            result: ExtractionResult = extractor.extract_video(
                video_path=str(resolved_video_path),
                output_path=str(npy_path),
                video_id=video_id,
                sign_label=sign_label,
                split=split_name,
                force=force,
            )
        except Exception as exc:
            proc_sec = time.time() - clip_start
            logger.error(
                f"Extraction exception | video_id={video_id} | sign={sign_label} | "
                f"{type(exc).__name__}: {exc}",
                extra={"stage": "extraction", "video_id": video_id},
            )
            logger.debug(traceback.format_exc(), extra={"stage": "extraction"})
            run_stats.record_error(
                clip,
                error_msg=f"{type(exc).__name__}: {exc}",
                proc_sec=proc_sec,
            )
            n_newly_processed += 1
            continue

        proc_sec = time.time() - clip_start

        # ----------------------------------------------------------------
        # Route by result.status
        # ----------------------------------------------------------------

        if result.status == "cached":
            # The extractor's own cache check triggered (e.g. the pipeline
            # fast-path above was bypassed by --force=False on a race).
            run_stats.record_cached(
                clip,
                output_path=result.output_path,
                n_frames=             result.num_frames,
                missing_pct=          result.missing_pct,
                missing_both=         result.missing_both_hands_frames,
                decode_failure_frames=result.decode_failure_frames,
            )
            n_newly_processed += 1
            continue

        if result.status == "skipped":
            logger.info(
                f"Skipped (policy: >{run_stats._max_missing_pct:.0%} "
                f"both-hands absent on decoded frames) | "
                f"video_id={video_id} | sign={sign_label} | "
                f"missing_pct={result.missing_pct:.1%} | "
                f"decode_failures={result.decode_failure_frames}",
                extra={"stage": "extraction", "video_id": video_id},
            )
            run_stats.record_skipped_policy(clip, result, proc_sec)
            n_newly_processed += 1
            continue

        if result.status == "error":
            logger.warning(
                f"Extraction error | video_id={video_id} | sign={sign_label} | "
                f"{result.error_message}",
                extra={"stage": "extraction", "video_id": video_id},
            )
            run_stats.record_error(
                clip,
                error_msg=result.error_message,
                proc_sec=proc_sec,
            )
            n_newly_processed += 1
            continue

        # status == "extracted" ─────────────────────────────────────────
        # Post-write verification: independent check on the file just written.
        # The extractor guarantees float32 and (N, 225); this catches any
        # partial write or OS-level corruption that slipped through.
        if not _verify_npy_file(npy_path, video_id, logger, full_check=True):
            logger.error(
                f"Post-write verification failed — removing corrupt .npy: "
                f"{npy_path} | video_id={video_id}",
                extra={"stage": "extraction", "video_id": video_id},
            )
            try:
                npy_path.unlink(missing_ok=True)
                _get_sidecar_path(npy_path).unlink(missing_ok=True)
            except OSError:
                pass
            run_stats.record_error(
                clip,
                error_msg="post_write_verification_failed",
                proc_sec=proc_sec,
            )
            n_newly_processed += 1
            continue

        logger.debug(
            f"Extracted: {video_id} ({sign_label}) | "
            f"frames={result.num_frames} | "
            f"decode_failures={result.decode_failure_frames} | "
            f"missing_pct={result.missing_pct:.1%} | "
            f"time={proc_sec:.2f}s",
            extra={"stage": "extraction", "video_id": video_id},
        )
        run_stats.record_extracted(clip, result, proc_sec)
        n_newly_processed += 1

        # ----------------------------------------------------------------
        # Progress logging every _LOG_INTERVAL newly-processed clips.
        # ETA is based on n_to_process_initially (fixed before loop).
        # ----------------------------------------------------------------
        if n_newly_processed % _LOG_INTERVAL == 0:
            elapsed   = time.time() - loop_start
            rate      = n_newly_processed / elapsed if elapsed > 0 else 0.0
            remaining = max(n_to_process_initially - n_newly_processed, 0)
            eta_sec   = remaining / rate if rate > 0 else 0.0

            logger.info(
                f"Progress | {i + 1}/{len(clips)} queued | "
                f"extracted={run_stats.n_extracted} | "
                f"cached={run_stats.n_skipped_cached} | "
                f"policy_skip={run_stats.n_skipped_policy} | "
                f"errors={run_stats.n_skipped_error} | "
                f"rate={rate:.1f} clips/s | "
                f"ETA={eta_sec / 60:.1f}min",
                extra={"stage": "extraction"},
            )

    # ----------------------------------------------------------------
    # Final summary line (covers runs shorter than _LOG_INTERVAL)
    # ----------------------------------------------------------------
    elapsed = time.time() - loop_start
    logger.info(
        f"Extraction loop complete | elapsed={elapsed:.1f}s | "
        f"extracted={run_stats.n_extracted} | "
        f"cached={run_stats.n_skipped_cached} | "
        f"policy_skip={run_stats.n_skipped_policy} | "
        f"errors={run_stats.n_skipped_error}",
        extra={"stage": "extraction"},
    )


# ---------------------------------------------------------------------------
# Post-run reporting and health checks
# ---------------------------------------------------------------------------

def _log_extraction_report(run_stats: _RunStats, logger) -> None:
    """
    Emit a structured, human-readable extraction report at INFO level.

    Uses the v1.1 semantics for missing_pct (over successfully-decoded
    frames) and surfaces decode_failure_frames as a separate line item
    for full transparency.

    Parameters
    ----------
    run_stats : _RunStats
        Completed run statistics.
    logger
        Active logger.
    """
    n_eff = max(run_stats.n_queued, 1)

    logger.info("=" * 65, extra={"stage": "extraction"})
    logger.info("STAGE 3 — EXTRACTION REPORT", extra={"stage": "extraction"})
    logger.info("=" * 65, extra={"stage": "extraction"})
    logger.info(f"  Queued              : {run_stats.n_queued}",          extra={"stage": "extraction"})
    logger.info(f"  Extracted (fresh)   : {run_stats.n_extracted}",       extra={"stage": "extraction"})
    logger.info(f"  Loaded (cache)      : {run_stats.n_skipped_cached}",  extra={"stage": "extraction"})
    logger.info(f"  Usable total        : {run_stats.n_usable}",          extra={"stage": "extraction"})
    logger.info(
        f"  Skipped (policy)    : {run_stats.n_skipped_policy}  "
        f"(>{run_stats._max_missing_pct:.0%} both-hands absent — "
        f"{run_stats.n_skipped_policy / n_eff:.1%} of queued)",
        extra={"stage": "extraction"},
    )
    logger.info(
        f"  Skipped (error)     : {run_stats.n_skipped_error}  "
        f"({run_stats.n_skipped_error / n_eff:.1%} of queued)",
        extra={"stage": "extraction"},
    )
    if run_stats.total_frames > 0:
        logger.info(
            f"  Total frames        : {run_stats.total_frames:,}",
            extra={"stage": "extraction"},
        )
        logger.info(
            f"  Decode failures     : {run_stats.total_decode_failures:,}  "
            f"(codec errors, excluded from missing_pct denominator — v1.1)",
            extra={"stage": "extraction"},
        )
        logger.info(
            f"  Global missing rate : {run_stats.global_missing_rate:.2%}  "
            f"({run_stats.total_missing_both:,}/{run_stats.total_frames - run_stats.total_decode_failures:,} "
            "decoded frames zero-filled both hands)",
            extra={"stage": "extraction"},
        )
    logger.info(
        f"  Processing time     : {run_stats.total_proc_sec:.1f}s",
        extra={"stage": "extraction"},
    )
    logger.info("=" * 65, extra={"stage": "extraction"})


def _validate_extraction_health(
    run_stats: _RunStats,
    logger,
) -> bool:
    """
    Check overall extraction health and emit actionable warnings.

    Thresholds are calibrated for extractor v1.1 where ``missing_pct``
    is computed over successfully-decoded frames only. Because codec errors
    no longer inflate the denominator, a global missing rate > 15% is a
    genuine MediaPipe quality signal rather than codec noise.

    Parameters
    ----------
    run_stats : _RunStats
        Completed run statistics.
    logger
        Active logger.

    Returns
    -------
    bool
        True if all health checks pass.
    """
    healthy = True
    n_eff   = max(run_stats.n_queued, 1)

    policy_rate = run_stats.n_skipped_policy / n_eff
    if policy_rate > _HEALTH_POLICY_SKIP_RATE_WARN:
        logger.warning(
            f"Policy skip rate {policy_rate:.1%} exceeds "
            f"{_HEALTH_POLICY_SKIP_RATE_WARN:.0%} threshold "
            f"(expected 5–8% for WLASL with v1.1 extractor). "
            "Possible causes: unusual signing angles, poor lighting, "
            "or --max-missing-frame-pct set too low. "
            "Consider raising --max-missing-frame-pct or lowering "
            "--min-detection-confidence.",
            extra={"stage": "extraction"},
        )
        healthy = False

    error_rate = run_stats.n_skipped_error / n_eff
    if error_rate > _HEALTH_ERROR_RATE_WARN:
        logger.warning(
            f"Error rate {error_rate:.1%} exceeds "
            f"{_HEALTH_ERROR_RATE_WARN:.0%} threshold. "
            "Review error records in preprocessing_summary_latest.json "
            "under the 'per_clip' key for details.",
            extra={"stage": "extraction"},
        )
        healthy = False

    if run_stats.global_missing_rate > _HEALTH_GLOBAL_MISSING_RATE:
        logger.warning(
            f"Global missing-landmark rate {run_stats.global_missing_rate:.1%} "
            f"exceeds {_HEALTH_GLOBAL_MISSING_RATE:.0%}. "
            "This is measured over successfully-decoded frames (v1.1) so it "
            "reflects genuine MediaPipe detection quality, not codec errors. "
            "Consider reviewing video quality for high-miss signs or adjusting "
            "--min-detection-confidence.",
            extra={"stage": "extraction"},
        )
        healthy = False

    return healthy


def _log_output_inventory(landmarks_dir: str, logger) -> None:
    """
    Walk the landmarks directory and log .npy file counts per split and sign.

    Uses a generator expression (``sum(1 for _ in glob())``) to avoid
    materialising the full file list into memory.

    Parameters
    ----------
    landmarks_dir : str
        Root landmarks directory.
    logger
        Active logger.
    """
    root = Path(landmarks_dir)
    if not root.exists():
        logger.warning(
            f"Landmarks directory not found: {root}",
            extra={"stage": "extraction"},
        )
        return

    total_files   = 0
    split_counts: dict[str, int]       = {}
    sign_totals:  dict[str, int]       = defaultdict(int)

    for split_dir in sorted(root.iterdir()):
        if not split_dir.is_dir():
            continue
        n_in_split = 0
        for sign_dir in sorted(split_dir.iterdir()):
            if not sign_dir.is_dir():
                continue
            n_files = sum(1 for _ in sign_dir.glob("*.npy"))
            n_in_split += n_files
            sign_totals[sign_dir.name] += n_files
        split_counts[split_dir.name] = n_in_split
        total_files += n_in_split

    signs_with_files = sum(1 for v in sign_totals.values() if v > 0)
    logger.info(
        f"Landmark inventory | total_npy={total_files} | "
        f"signs_covered={signs_with_files}",
        extra={"stage": "extraction"},
    )
    for split_name, count in sorted(split_counts.items()):
        logger.info(
            f"  {split_name:6s}: {count:4d} .npy files",
            extra={"stage": "extraction"},
        )


# ---------------------------------------------------------------------------
# Post-run finalisation: landmark inventory CSV
# ---------------------------------------------------------------------------

def _finalise_run(
    run_stats: _RunStats,
    landmarks_dir: str,
    logger,
) -> None:
    """
    Write the landmark_inventory.csv using the per-clip records from _RunStats.

    This function creates a minimal ExtractionResult list from the already-
    accumulated _RunStats records and delegates to ``write_landmark_inventory``
    in the extractor module. This keeps the CSV format consistent with what
    ``LandmarkExtractor.extract_dataset()`` produces when called directly.

    Parameters
    ----------
    run_stats : _RunStats
        Completed run statistics containing all per-clip records.
    landmarks_dir : str
        Root landmarks directory where the CSV is written.
    logger
        Active logger.
    """
    results: list[ExtractionResult] = []

    for rec in run_stats._records:
        outcome = rec.get("outcome", "error")

        # Map pipeline outcome strings back to ExtractionResult status values
        status_map = {
            "extracted":      "extracted",
            "cached":         "cached",
            "skipped_policy": "skipped",
            "error":          "error",
            "dry_run":        "skipped",  # dry-run clips appear as skipped
        }
        status = status_map.get(outcome, "error")

        result = ExtractionResult(
            video_id=                 rec.get("video_id",              ""),
            sign_label=               rec.get("sign_label",            ""),
            split=                    rec.get("split",                 ""),
            output_path=              rec.get("output_path",           ""),
            status=                   status,
            num_frames=               rec.get("n_frames",              0),
            decode_failure_frames=    rec.get("decode_failure_frames", 0),   # v1.1
            missing_left_hand_frames= 0,   # not tracked at pipeline level
            missing_right_hand_frames=0,
            missing_pose_frames=      0,
            missing_both_hands_frames=rec.get("n_missing_both",       0),
            missing_pct=              rec.get("missing_pct",           0.0),
            skip_reason=              rec.get("skip_reason",           ""),
            processing_time_sec=      rec.get("proc_sec",              0.0),
            error_message=            rec.get("error_message",         ""),
        )
        results.append(result)

    try:
        csv_path = write_landmark_inventory(results, landmarks_dir)
        logger.info(
            f"Landmark inventory CSV written: {csv_path}",
            extra={"stage": "extraction"},
        )
    except Exception as exc:
        logger.warning(
            f"Could not write landmark inventory CSV: {type(exc).__name__}: {exc}",
            extra={"stage": "extraction"},
        )


# ---------------------------------------------------------------------------
# Extractor initialisation
# ---------------------------------------------------------------------------

def _init_extractor(args: argparse.Namespace, logger) -> Optional[LandmarkExtractor]:
    """
    Initialise LandmarkExtractor with runtime configuration and warm up MediaPipe.

    The warm-up call (``extractor._init_mediapipe()``) loads the MediaPipe model
    before the main loop so the first clip does not incur a cold-start timing
    penalty (model load takes ~2–5 seconds).

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    logger
        Active logger.

    Returns
    -------
    LandmarkExtractor | None
        Initialised extractor, or None on failure (caller exits with code 2).
    """
    logger.info(
        f"Initialising LandmarkExtractor | "
        f"schema_version={EXTRACTOR_SCHEMA_VERSION} | "
        f"max_missing_pct={args.max_missing_frame_pct:.0%} | "
        f"min_detection_conf={args.min_detection_confidence} | "
        f"min_tracking_conf={args.min_tracking_confidence}",
        extra={"stage": "extraction"},
    )
    try:
        extractor = LandmarkExtractor(
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
        )
        # Warm up MediaPipe before the main loop
        extractor._init_mediapipe()
        return extractor
    except Exception as exc:
        logger.error(
            f"Failed to initialise LandmarkExtractor: "
            f"{type(exc).__name__}: {exc}. "
            "Ensure mediapipe==0.10.14 and opencv-python==4.8.1.78 are installed "
            "in the active Conda environment.",
            extra={"stage": "extraction"},
        )
        logger.debug(traceback.format_exc(), extra={"stage": "extraction"})
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """
    Execute the Stage 3 landmark extraction pipeline.

    Returns
    -------
    int
        0 = success, 1 = input/validation error, 2 = runtime failure.
    """
    parser = _build_parser()
    args   = parser.parse_args()

    # ----------------------------------------------------------------
    # Logging — must be configured before any other operation
    # ----------------------------------------------------------------
    log_level = "DEBUG" if args.verbose else "INFO"
    run_label = "stage3_sample" if args.sample_only else f"stage3_{args.split}"
    if args.dry_run:
        run_label += "_dryrun"

    log_file = configure_logging(
        log_dir=args.log_dir,
        run_name=run_label,
        level=log_level,
        file_level="DEBUG",
    )
    logger = get_logger(__name__, stage="extraction")

    # ----------------------------------------------------------------
    # Header
    # ----------------------------------------------------------------
    logger.info(
        "=" * 65, extra={"stage": "extraction"}
    )
    logger.info(
        "WLASL Gesture Recognition — Stage 3: Landmark Extraction",
        extra={"stage": "extraction"},
    )
    logger.info(
        f"Extractor schema version : {EXTRACTOR_SCHEMA_VERSION}",
        extra={"stage": "extraction"},
    )
    logger.info(
        f"Log file                 : {log_file}",
        extra={"stage": "extraction"},
    )

    mode_str = (
        f"SAMPLE ({_SAMPLE_CLIPS_PER_SIGN} clips/sign/split)"
        if args.sample_only
        else args.split.upper()
    )
    logger.info(
        f"Mode: {mode_str} | force={args.force} | "
        f"verify_existing={args.verify_existing} | "
        f"dry_run={args.dry_run}",
        extra={"stage": "extraction"},
    )
    logger.info(
        f"Extractor config | "
        f"max_missing_pct={args.max_missing_frame_pct:.0%} | "
        f"min_detection_conf={args.min_detection_confidence} | "
        f"min_tracking_conf={args.min_tracking_confidence}",
        extra={"stage": "extraction"},
    )
    logger.info(
        "=" * 65, extra={"stage": "extraction"}
    )

    # ----------------------------------------------------------------
    # Reproducibility
    # ----------------------------------------------------------------
    set_seeds(args.seed)

    # ----------------------------------------------------------------
    # Argument validation
    # ----------------------------------------------------------------
    if not _validate_args(args, logger):
        return 1

    # ----------------------------------------------------------------
    # Collect clips
    # ----------------------------------------------------------------
    split_arg = "all" if args.sample_only else args.split
    clips = _collect_clips(
        splits_dir=args.splits_dir,
        split_arg=split_arg,
        sample_only=args.sample_only,
        logger=logger,
    )

    if not clips:
        logger.error(
            "No clips to process. "
            "Verify split CSVs exist and contain video_path entries: "
            f"{args.splits_dir}",
            extra={"stage": "extraction"},
        )
        return 1

    # ----------------------------------------------------------------
    # Run statistics accumulator
    # ----------------------------------------------------------------
    run_id    = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_stats = _RunStats(
        run_id=run_id,
        max_missing_frame_pct=args.max_missing_frame_pct,
    )

    # ----------------------------------------------------------------
    # Dry-run short-circuit
    # ----------------------------------------------------------------
    if args.dry_run:
        logger.info(
            "[DRY RUN] Logging extraction plan only. No files will be written.",
            extra={"stage": "extraction"},
        )
        _run_extraction_loop(
            clips=clips,
            extractor=None,
            landmarks_dir=args.landmarks_dir,
            run_stats=run_stats,
            force=args.force,
            verify_existing=False,
            dry_run=True,
            logger=logger,
        )
        _log_extraction_report(run_stats, logger)
        logger.info(
            "[DRY RUN] Complete. Re-run without --dry-run to extract landmarks.",
            extra={"stage": "extraction"},
        )
        return 0

    # ----------------------------------------------------------------
    # Initialise MediaPipe extractor
    # ----------------------------------------------------------------
    extractor = _init_extractor(args, logger)
    if extractor is None:
        return 2

    logger.info(
        f"LandmarkExtractor ready | feature_size={FEATURE_SIZE} values/frame",
        extra={"stage": "extraction"},
    )

    # Ensure output root exists
    Path(args.landmarks_dir).mkdir(parents=True, exist_ok=True)
    logger.info(
        f"Landmarks output root: {args.landmarks_dir}",
        extra={"stage": "extraction"},
    )

    # ----------------------------------------------------------------
    # Main extraction loop
    # ----------------------------------------------------------------
    pipeline_start = time.time()

    try:
        _run_extraction_loop(
            clips=clips,
            extractor=extractor,
            landmarks_dir=args.landmarks_dir,
            run_stats=run_stats,
            force=args.force,
            verify_existing=args.verify_existing,
            dry_run=False,
            logger=logger,
        )
    except KeyboardInterrupt:
        logger.warning(
            f"Extraction interrupted by user (KeyboardInterrupt). "
            f"{run_stats.n_extracted} clips extracted before interruption. "
            "Re-run without --force to resume — existing .npy files are not "
            "re-processed.",
            extra={"stage": "extraction"},
        )
        # Write partial summary before exiting so the user can review progress
        try:
            partial = run_stats.to_dict(status="PARTIAL_INTERRUPTED")
            _write_pipeline_summary(partial, args.summary_dir, logger)
        except Exception:
            pass
        return 2

    except Exception as exc:
        logger.error(
            f"Unexpected exception in extraction loop: "
            f"{type(exc).__name__}: {exc}",
            extra={"stage": "extraction"},
        )
        logger.debug(traceback.format_exc(), extra={"stage": "extraction"})
        try:
            partial = run_stats.to_dict(status="PARTIAL_INTERRUPTED")
            _write_pipeline_summary(partial, args.summary_dir, logger)
        except Exception:
            pass
        return 2

    finally:
        # Always release MediaPipe resources, even on exception paths.
        if extractor is not None:
            extractor.close()

    # ----------------------------------------------------------------
    # Post-run reporting
    # ----------------------------------------------------------------
    total_elapsed = time.time() - pipeline_start

    _log_extraction_report(run_stats, logger)
    health_ok = _validate_extraction_health(run_stats, logger)

    if not health_ok:
        logger.warning(
            "One or more health checks exceeded their thresholds. "
            "Review warnings above and check preprocessing_summary_latest.json "
            "before proceeding to Stage 4.",
            extra={"stage": "extraction"},
        )

    # ----------------------------------------------------------------
    # Write pipeline-level summary JSON (latest + history)
    # ----------------------------------------------------------------
    try:
        summary = run_stats.to_dict(status="completed")
        _write_pipeline_summary(summary, args.summary_dir, logger)
    except Exception as exc:
        logger.error(
            f"Failed to write pipeline summary: {type(exc).__name__}: {exc}. "
            "The .npy files themselves are the critical output — proceeding.",
            extra={"stage": "extraction"},
        )

    # ----------------------------------------------------------------
    # Write landmark_inventory.csv (aligned with extractor CSV format)
    # ----------------------------------------------------------------
    _finalise_run(run_stats, args.landmarks_dir, logger)

    # ----------------------------------------------------------------
    # Landmark directory inventory (file count verification)
    # ----------------------------------------------------------------
    _log_output_inventory(args.landmarks_dir, logger)

    # ----------------------------------------------------------------
    # Sample-mode verification guidance
    # ----------------------------------------------------------------
    if args.sample_only and run_stats.n_extracted > 0:
        logger.info(
            "SAMPLE EXTRACTION COMPLETE",
            extra={"stage": "extraction"},
        )
        logger.info(
            "Manually verify a few output arrays before proceeding:",
            extra={"stage": "extraction"},
        )
        logger.info(
            "  import numpy as np",
            extra={"stage": "extraction"},
        )
        logger.info(
            "  arr = np.load('data/landmarks/train/<sign>/<video_id>.npy')",
            extra={"stage": "extraction"},
        )
        logger.info(
            "  assert arr.ndim == 2 and arr.shape[1] == 225  # (N, 225) float32",
            extra={"stage": "extraction"},
        )
        logger.info(
            "  print(arr.shape, arr.min(), arr.max())  # values approximately [0, 1]",
            extra={"stage": "extraction"},
        )
        logger.info(
            "If shapes and value ranges look correct, open and execute "
            "notebooks/02_landmark_inspection.ipynb.",
            extra={"stage": "extraction"},
        )
        logger.info(
            "Then run full extraction: "
            "python pipelines/run_landmark_extraction.py --split all",
            extra={"stage": "extraction"},
        )

    # ----------------------------------------------------------------
    # Footer
    # ----------------------------------------------------------------
    logger.info("=" * 65, extra={"stage": "extraction"})
    logger.info("STAGE 3 COMPLETE", extra={"stage": "extraction"})
    logger.info(
        f"  Extracted (fresh) : {run_stats.n_extracted}",
        extra={"stage": "extraction"},
    )
    logger.info(
        f"  Usable total      : {run_stats.n_usable}",
        extra={"stage": "extraction"},
    )
    logger.info(
        f"  Total elapsed     : {total_elapsed:.1f}s "
        f"({total_elapsed / 60:.1f} minutes)",
        extra={"stage": "extraction"},
    )
    logger.info(
        f"  Summary dir       : {args.summary_dir}",
        extra={"stage": "extraction"},
    )
    logger.info(
        f"  Landmarks dir     : {args.landmarks_dir}",
        extra={"stage": "extraction"},
    )

    if not args.sample_only:
        logger.info(
            "Next: Stage 4 — build src/features/pipeline.py (FeaturePipeline) "
            "and src/features/augmentation.py, then run "
            "notebooks/03_feature_engineering_experiments.ipynb.",
            extra={"stage": "extraction"},
        )
    logger.info("=" * 65, extra={"stage": "extraction"})

    # Return exit code 2 if health checks failed, 0 if all passed.
    # This allows CI scripts to distinguish between "extraction ran but
    # something looks off" and "extraction ran cleanly".
    return 0 if health_ok else 2


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(main())