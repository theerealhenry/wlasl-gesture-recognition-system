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
reason about. See the architectural note in ``src/features/__init__.py``.

Output layout
-------------
Landmark arrays are written to:

    data/landmarks/<split>/<sign_label>/<video_id>.npy

Each file contains a float32 array of shape ``(num_frames, 225)`` where:

    [0:63]    left hand  — 21 landmarks × (x, y, z)
    [63:126]  right hand — 21 landmarks × (x, y, z)
    [126:225] pose       — 33 landmarks × (x, y, z)

Alongside every .npy file a sibling .meta.json sidecar is written by the
extractor. The sidecar stores per-clip detection statistics (missing rates,
frame count, schema version). On subsequent runs, cache-hit clips have their
statistics restored from the sidecar so that aggregate missing-rate figures
remain accurate without re-processing any video.

IMPORTANT: Arrays store the clip's actual frame count — they are NOT padded to
seq_len=30. Padding and truncation are deferred to ``FeaturePipeline`` (Stage 4)
so the same .npy files can serve all sequence-length ablation experiments
({20, 30, 40, 60} frames) without any re-extraction.

Resumability
------------
Before processing a clip the extractor checks for a valid .npy + .meta.json
pair. If both exist and pass validation (shape, dtype, schema version, finite
values), the clip is skipped unless ``--force`` is passed. This makes the
script safe to restart after crashes or interruptions — no wasted work.

If ``--verify-existing`` is passed, every cache-hit .npy is spot-checked
before being trusted. Use this after a filesystem incident or when resuming
after an incomplete previous run.

Atomic writes
-------------
Every .npy file is first written to a ``<video_id>.npy.tmp`` sibling, then
atomically renamed to the final ``<video_id>.npy``. This prevents a half-
written file from being mistaken for a valid cache entry if the process is
interrupted mid-write.

Extraction summary
------------------
Two JSON summary files are written after each run:

    data/preprocessing_summary_latest.json  — current run only (always overwritten)
    data/preprocessing_summary_history.json — append-only audit log of all runs

Notebook 02 reads ``_latest`` for the missing-landmark analysis. The
``_history`` file provides an audit trail for comparing multiple runs.

Usage
-----
Sample-only run (1 clip per sign per split — Stage 2 validation gate):
    python pipelines/run_landmark_extraction.py --sample-only

Full extraction, all splits (30–90 minutes):
    python pipelines/run_landmark_extraction.py --split all

Single split only:
    python pipelines/run_landmark_extraction.py --split train

Force re-extraction (overwrite existing .npy files):
    python pipelines/run_landmark_extraction.py --split all --force

Verify existing cached files while resuming:
    python pipelines/run_landmark_extraction.py --split all --verify-existing

Dry run (validate inputs and log plan, write nothing):
    python pipelines/run_landmark_extraction.py --dry-run

Verbose debug logging:
    python pipelines/run_landmark_extraction.py --sample-only --verbose

Exit codes
----------
0  — Extraction completed successfully (includes partial runs where some
     clips were skipped due to skip policy or cache-hit).
1  — Input error: split CSV missing, no clips found, bad argument value.
2  — Unexpected exception or post-run health check failure.

Design principles
-----------------
- No assert statements in production paths — explicit RuntimeError throughout
- No print() anywhere — structured logging via configure_logging / get_logger
- No MLflow — tracking begins at Stage 5 (run_training.py)
- Single source of truth for which videos to process: data/splits/*.csv
- Atomic .npy writes: tmp-file + rename guards against partial-write corruption
- Per-clip exception isolation: one bad video never aborts the whole run
- ETA computed from a fixed pre-loop baseline — never drifts as files are written
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
# Utils — configure logging before any other import that might log
# ---------------------------------------------------------------------------
from src.utils.logger import configure_logging, get_logger
from src.utils.reproducibility import set_seeds

# ---------------------------------------------------------------------------
# Feature modules
# The extractor contains all MediaPipe logic; this script is pure orchestration.
# ---------------------------------------------------------------------------
from src.features.extractor import LandmarkExtractor, ExtractionResult
from src.features.constants import FEATURE_SIZE


# ---------------------------------------------------------------------------
# Default paths and constants
# ---------------------------------------------------------------------------
_DEFAULT_SPLITS_DIR    = str(_REPO_ROOT / "data" / "splits")
_DEFAULT_LANDMARKS_DIR = str(_REPO_ROOT / "data" / "landmarks")
_DEFAULT_SUMMARY_DIR   = str(_REPO_ROOT / "data")
_DEFAULT_LOG_DIR       = str(_REPO_ROOT / "logs")

_VALID_SPLITS          = ("train", "val", "test", "all")
_LOG_INTERVAL          = 50    # clips between progress log lines
_SAMPLE_CLIPS_PER_SIGN = 1     # clips per sign in --sample-only mode
_SEED                  = 42

# Extractor defaults — kept in sync with extractor.py defaults
_DEFAULT_MAX_MISSING_FRAME_PCT = 0.30
_DEFAULT_MIN_DETECTION_CONF    = 0.5
_DEFAULT_MIN_TRACKING_CONF     = 0.5

# Characters unsafe as filesystem path components on any OS (Windows or POSIX)
_UNSAFE_PATH_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


# ---------------------------------------------------------------------------
# Filesystem safety helper
# ---------------------------------------------------------------------------

def _sanitize_path_component(name: str) -> str:
    """
    Replace characters that are unsafe in a filesystem path component.

    Handles sign labels that might contain slashes, colons, or other special
    characters. The WLASL sign labels are all plain ASCII words, so this
    function is a safety net rather than a routine operation.

    Substitution rule: any character in ``_UNSAFE_PATH_CHARS`` is replaced
    with an underscore. Leading/trailing whitespace and dots are stripped.

    Parameters
    ----------
    name : str
        Raw path component (e.g. a sign label or video_id).

    Returns
    -------
    str
        Safe path component suitable for all target filesystems.

    Examples
    --------
    >>> _sanitize_path_component("before")
    'before'
    >>> _sanitize_path_component("sign/with:special*chars")
    'sign_with_special_chars'
    """
    safe = _UNSAFE_PATH_CHARS.sub("_", name)
    safe = safe.strip(". ")
    # Collapse multiple consecutive underscores to a single one
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
            "Always run --sample-only first to validate the extractor before\n"
            "committing to the full 30–90 minute extraction."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Stage 2 validation gate — 1 clip per sign, all splits (~2-5 minutes)
  python pipelines/run_landmark_extraction.py --sample-only

  # Full extraction, all splits (30-90 minutes)
  python pipelines/run_landmark_extraction.py --split all

  # Training split only
  python pipelines/run_landmark_extraction.py --split train

  # Force re-extraction (overwrite existing .npy + .meta.json files)
  python pipelines/run_landmark_extraction.py --split all --force

  # Resume and verify all previously cached files
  python pipelines/run_landmark_extraction.py --split all --verify-existing

  # Dry run — validate inputs and log plan, write nothing
  python pipelines/run_landmark_extraction.py --dry-run

  # Verbose debug output
  python pipelines/run_landmark_extraction.py --sample-only --verbose

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
            "Directory for extraction summary JSON files. Two files are written: "
            "preprocessing_summary_latest.json (current run) and "
            f"preprocessing_summary_history.json (audit log). "
            f"(default: {_DEFAULT_SUMMARY_DIR})"
        ),
    )
    parser.add_argument(
        "--log-dir",
        default=_DEFAULT_LOG_DIR,
        metavar="DIR",
        help=f"Directory for log files (default: {_DEFAULT_LOG_DIR})",
    )

    # ----------------------------------------------------------------
    # Run mode
    # ----------------------------------------------------------------
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--sample-only",
        action="store_true",
        help=(
            f"Process only {_SAMPLE_CLIPS_PER_SIGN} clip(s) per sign per split "
            "(alphabetically first by video_id). "
            "Use this as the Stage 2 validation gate before full extraction."
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
            "Without this flag, existing valid files are skipped (resumable by default)."
        ),
    )
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help=(
            "On cache-hit clips, verify the existing .npy before trusting it. "
            "Checks ndim, shape, dtype, and finiteness. Corrupt files are "
            "automatically reprocessed. Use after a filesystem incident or "
            "when resuming an interrupted run."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate all inputs and log the extraction plan without writing "
            "any files or calling MediaPipe. Useful for verifying clip counts "
            "and output paths before a long run."
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
            "Skip a clip if the fraction of frames where both hands are absent "
            f"exceeds this threshold (default: {_DEFAULT_MAX_MISSING_FRAME_PCT}). "
            "E.g. 0.30 skips clips where >30%% of frames have no hand detection."
        ),
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=_DEFAULT_MIN_DETECTION_CONF,
        metavar="CONF",
        help=(
            f"MediaPipe Holistic min detection confidence "
            f"(default: {_DEFAULT_MIN_DETECTION_CONF})"
        ),
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=_DEFAULT_MIN_TRACKING_CONF,
        metavar="CONF",
        help=(
            f"MediaPipe Holistic min tracking confidence "
            f"(default: {_DEFAULT_MIN_TRACKING_CONF})"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_SEED,
        metavar="N",
        help=(
            f"Random seed passed to set_seeds() for global reproducibility "
            f"(default: {_SEED}). "
            "Sample selection within --sample-only mode is alphabetically "
            "deterministic and does not consume randomness."
        ),
    )

    return parser


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------

def _validate_args(args: argparse.Namespace, logger) -> bool:
    """
    Validate all CLI argument constraints before any work begins.

    Centralises argument checks so ``main()`` stays readable. Returns False
    (caller should exit 1) if any constraint is violated.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments from ``_build_parser()``.
    logger
        Active logger.

    Returns
    -------
    bool
        True if all constraints pass, False if the caller should abort.
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
            "--force implies re-extraction of all clips; "
            "--verify-existing is redundant and will be ignored.",
            extra={"stage": "extraction"},
        )

    return valid


# ---------------------------------------------------------------------------
# Split loading and clip collection
# ---------------------------------------------------------------------------

def _load_split_df(splits_dir: str, split_name: str, logger) -> Optional[pd.DataFrame]:
    """
    Load and schema-validate a single split CSV.

    Parameters
    ----------
    splits_dir : str
        Directory containing the split CSVs produced by Stage 1.
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
            f"Failed to read {csv_path}: {exc}",
            extra={"stage": "extraction"},
        )
        return None

    required_cols = {"video_id", "sign_label", "class_idx", "signer_id", "video_path"}
    missing_cols = required_cols - set(df.columns)
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


def _collect_clips(
    splits_dir: str,
    split_arg: str,
    sample_only: bool,
    logger,
) -> list[dict[str, Any]]:
    """
    Build the ordered list of clips to process.

    Each entry is a dict with keys:
        video_id, sign_label, class_idx, signer_id, split, video_path,
        safe_sign_label  (filesystem-safe version of sign_label)

    In ``--sample-only`` mode, exactly ``_SAMPLE_CLIPS_PER_SIGN`` clip(s) per
    sign per split are selected. Selection is purely alphabetical by video_id
    within each sign group — deterministic without any randomness. The ``seed``
    CLI argument does not affect sample selection.

    Parameters
    ----------
    splits_dir : str
        Path to the splits directory.
    split_arg : str
        "train", "val", "test", or "all".
    sample_only : bool
        If True, restrict to one clip per sign per split.
    logger
        Active logger.

    Returns
    -------
    list[dict]
        Ordered list of clip records. An empty list signals a loading error.
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
            # Select the alphabetically-first clip per sign (by video_id).
            # Sorting before groupby + first() guarantees a fixed, reproducible
            # selection that is independent of CSV row order.
            df = (
                df.sort_values("video_id")
                .groupby("sign_label", sort=True)
                .head(_SAMPLE_CLIPS_PER_SIGN)
                .reset_index(drop=True)
            )
            n_signs = df["sign_label"].nunique()
            logger.info(
                f"Sample mode: selected {len(df)} clips from '{split_name}' "
                f"({_SAMPLE_CLIPS_PER_SIGN} per sign, {n_signs} signs represented)",
                extra={"stage": "extraction"},
            )

        for _, row in df.iterrows():
            sign_label = str(row["sign_label"])
            clips.append({
                "video_id":       str(row["video_id"]),
                "sign_label":     sign_label,
                "safe_sign_label": _sanitize_path_component(sign_label),
                "class_idx":      int(row["class_idx"]),
                "signer_id":      int(row["signer_id"]),
                "split":          split_name,
                "video_path":     str(row["video_path"]),
            })

    n_signs_total = len({c["sign_label"] for c in clips})
    logger.info(
        f"Total clips queued: {len(clips)} | "
        f"splits={split_names} | signs={n_signs_total} | sample_only={sample_only}",
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
        Filesystem-safe sign label (produced by ``_sanitize_path_component``).
    video_id : str
        WLASL video identifier (already safe — numeric string).

    Returns
    -------
    Path
        Absolute output .npy path.
    """
    return Path(landmarks_dir) / split_name / safe_sign_label / f"{video_id}.npy"


def _get_tmp_path(npy_path: Path) -> Path:
    """Return the temporary write path for atomic .npy writes."""
    return npy_path.with_suffix(".npy.tmp")


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
    if not video_path or video_path.lower() == "nan":
        logger.warning(
            f"Empty video_path for video_id={video_id} — cannot process.",
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
# .npy verification
# ---------------------------------------------------------------------------

def _verify_npy_file(
    npy_path: Path,
    video_id: str,
    logger,
    full_check: bool = False,
) -> bool:
    """
    Validate a .npy landmark array file.

    Always checks:
    - Loadable by numpy without ``allow_pickle``
    - ndim == 2
    - shape[1] == FEATURE_SIZE (225)
    - dtype == float32

    When ``full_check=True`` (used after fresh writes):
    - All values are finite (no NaN or Inf across the entire array)

    When ``full_check=False`` (used on cache-hit verification):
    - Only the first row is checked for finiteness (fast mmap-based spot-check)

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
            f"Cannot load .npy: {npy_path} | video_id={video_id} | {exc}",
            extra={"stage": "extraction", "video_id": video_id},
        )
        return False

    if arr.ndim != 2:
        logger.error(
            f"ndim error: expected 2D, got shape={arr.shape} | "
            f"video_id={video_id} | path={npy_path}",
            extra={"stage": "extraction", "video_id": video_id},
        )
        return False

    if arr.shape[1] != FEATURE_SIZE:
        logger.error(
            f"Feature-size error: expected {FEATURE_SIZE} cols, "
            f"got {arr.shape[1]} | video_id={video_id} | path={npy_path}",
            extra={"stage": "extraction", "video_id": video_id},
        )
        return False

    if arr.dtype != np.float32:
        logger.warning(
            f"dtype warning: expected float32, got {arr.dtype} | "
            f"video_id={video_id} | path={npy_path}",
            extra={"stage": "extraction", "video_id": video_id},
        )
        # dtype mismatch is a warning, not a hard failure — the extractor
        # guarantees float32, so this indicates a foreign file.

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
# Dry-run plan reporter
# ---------------------------------------------------------------------------

def _report_dry_run_plan(
    clips: list[dict[str, Any]],
    landmarks_dir: str,
    force: bool,
    logger,
) -> None:
    """
    Log the extraction plan without doing any actual work.

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
    would_extract = 0
    would_skip_exists = 0
    by_split: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "exists": 0})

    for clip in clips:
        npy_path = _get_output_path(
            landmarks_dir,
            clip["split"],
            clip["safe_sign_label"],
            clip["video_id"],
        )
        by_split[clip["split"]]["total"] += 1
        if npy_path.exists() and not force:
            would_skip_exists += 1
            by_split[clip["split"]]["exists"] += 1
        else:
            would_extract += 1

    logger.info("[DRY RUN] Extraction plan:", extra={"stage": "extraction"})
    logger.info(
        f"  Total queued   : {len(clips)}",
        extra={"stage": "extraction"},
    )
    logger.info(
        f"  Would extract  : {would_extract}",
        extra={"stage": "extraction"},
    )
    logger.info(
        f"  Would skip     : {would_skip_exists} (already exist, --force not set)",
        extra={"stage": "extraction"},
    )
    for split_name, counts in sorted(by_split.items()):
        exists = counts["exists"]
        to_do  = counts["total"] - exists
        logger.info(
            f"  {split_name:6s}: {counts['total']:4d} total | "
            f"{exists:3d} already exist | {to_do:3d} would extract",
            extra={"stage": "extraction"},
        )


# ---------------------------------------------------------------------------
# Per-run statistics accumulator
# ---------------------------------------------------------------------------

class _RunStats:
    """
    Accumulates per-clip and aggregate statistics for the current run.

    Designed to be a lightweight dict accumulator — negligible overhead per clip.

    Attributes
    ----------
    n_queued : int
        Total clips submitted to the extraction loop.
    n_extracted : int
        Clips freshly extracted and written to disk this run.
    n_skipped_cached : int
        Clips skipped because a valid .npy + .meta.json already existed.
    n_skipped_policy : int
        Clips skipped because the missing-frame rate exceeded the threshold.
    n_skipped_error : int
        Clips skipped because of a video read failure or MediaPipe exception.
    n_dry_run : int
        Clips logged in dry-run mode (no actual work done).
    total_frames : int
        Sum of frame counts across all freshly extracted clips.
    total_missing : int
        Sum of both-hands-absent frame counts across extracted clips.
    total_proc_sec : float
        Total wall-clock processing time for extracted clips.
    """

    def __init__(self, run_id: str, max_missing_frame_pct: float) -> None:
        self._run_id             = run_id
        self._started_utc        = datetime.now(timezone.utc).isoformat()
        # Store the runtime threshold so reports print the actual value used,
        # not a hardcoded module constant.
        self._max_missing_pct    = max_missing_frame_pct

        # Per-clip records list (keyed by (video_id, split) for merge dedup)
        self._records: list[dict[str, Any]] = []

        # Aggregate counters
        self.n_queued          = 0
        self.n_extracted       = 0
        self.n_skipped_cached  = 0
        self.n_skipped_policy  = 0
        self.n_skipped_error   = 0
        self.n_dry_run         = 0
        self.total_frames      = 0
        self.total_missing     = 0
        self.total_proc_sec    = 0.0

        # Per-sign breakdown
        self._sign_frames:    dict[str, int] = defaultdict(int)
        self._sign_missing:   dict[str, int] = defaultdict(int)
        self._sign_extracted: dict[str, int] = defaultdict(int)
        self._sign_skipped:   dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_extracted(
        self,
        clip: dict[str, Any],
        result: ExtractionResult,
        proc_sec: float,
        output_path: str,
    ) -> None:
        """Record a successfully extracted clip."""
        sign = clip["sign_label"]
        self.n_extracted += 1
        self.total_frames  += result.num_frames
        self.total_missing += result.missing_both_hands_frames
        self.total_proc_sec += proc_sec
        self._sign_frames[sign]    += result.num_frames
        self._sign_missing[sign]   += result.missing_both_hands_frames
        self._sign_extracted[sign] += 1

        self._records.append({
            "video_id":          clip["video_id"],
            "sign_label":        sign,
            "class_idx":         clip["class_idx"],
            "signer_id":         clip["signer_id"],
            "split":             clip["split"],
            "video_path":        clip["video_path"],
            "output_path":       output_path,
            "outcome":           "extracted",
            "proc_sec":          round(proc_sec, 4),
            "n_frames":          result.num_frames,
            "n_missing_frames":  result.missing_both_hands_frames,
            "missing_rate":      round(result.missing_pct, 4),
        })

    def record_cached(
        self,
        clip: dict[str, Any],
        output_path: str,
        n_frames: int = 0,
        missing_pct: float = 0.0,
        missing_both: int = 0,
    ) -> None:
        """
        Record a cache-hit clip.

        Statistics are passed in from the .meta.json sidecar when available
        so that aggregate missing-rate figures remain accurate.
        """
        sign = clip["sign_label"]
        self.n_skipped_cached += 1
        self._sign_frames[sign]    += n_frames
        self._sign_missing[sign]   += missing_both
        self._sign_extracted[sign] += 1   # counts towards usable total

        self._records.append({
            "video_id":          clip["video_id"],
            "sign_label":        sign,
            "class_idx":         clip["class_idx"],
            "signer_id":         clip["signer_id"],
            "split":             clip["split"],
            "video_path":        clip["video_path"],
            "output_path":       output_path,
            "outcome":           "skipped_cached",
            "proc_sec":          0.0,
            "n_frames":          n_frames,
            "n_missing_frames":  missing_both,
            "missing_rate":      round(missing_pct, 4),
        })

    def record_skipped_policy(
        self,
        clip: dict[str, Any],
        result: ExtractionResult,
        proc_sec: float,
    ) -> None:
        """Record a clip skipped due to the missing-frame-pct policy."""
        sign = clip["sign_label"]
        self.n_skipped_policy += 1
        self._sign_skipped[sign] += 1
        self.total_proc_sec += proc_sec

        self._records.append({
            "video_id":    clip["video_id"],
            "sign_label":  sign,
            "class_idx":   clip["class_idx"],
            "signer_id":   clip["signer_id"],
            "split":       clip["split"],
            "video_path":  clip["video_path"],
            "output_path": "",
            "outcome":     "skipped_policy",
            "proc_sec":    round(proc_sec, 4),
            "n_frames":    result.num_frames,
            "n_missing_frames": result.missing_both_hands_frames,
            "missing_rate": round(result.missing_pct, 4),
            "skip_reason": result.skip_reason,
        })

    def record_error(
        self,
        clip: dict[str, Any],
        error_msg: str,
        proc_sec: float = 0.0,
    ) -> None:
        """Record a clip that failed due to an exception or missing video file."""
        sign = clip["sign_label"]
        self.n_skipped_error += 1
        self._sign_skipped[sign] += 1
        self.total_proc_sec += proc_sec

        self._records.append({
            "video_id":    clip["video_id"],
            "sign_label":  sign,
            "class_idx":   clip["class_idx"],
            "signer_id":   clip["signer_id"],
            "split":       clip["split"],
            "video_path":  clip["video_path"],
            "output_path": "",
            "outcome":     "skipped_error",
            "proc_sec":    round(proc_sec, 4),
            "n_frames":    0,
            "n_missing_frames": 0,
            "missing_rate": 0.0,
            "error":       error_msg,
        })

    def record_dry_run(self, clip: dict[str, Any]) -> None:
        """Record a clip in dry-run mode (no actual work performed)."""
        self.n_dry_run += 1
        self._records.append({
            "video_id":    clip["video_id"],
            "sign_label":  clip["sign_label"],
            "class_idx":   clip["class_idx"],
            "signer_id":   clip["signer_id"],
            "split":       clip["split"],
            "video_path":  clip["video_path"],
            "output_path": "",
            "outcome":     "dry_run",
            "proc_sec":    0.0,
        })

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self, status: str = "completed") -> dict[str, Any]:
        """
        Serialise aggregate and per-clip statistics to a JSON-ready dict.

        Parameters
        ----------
        status : str
            "completed" or "PARTIAL_INTERRUPTED" — recorded in the metadata.

        Returns
        -------
        dict
            Full summary payload suitable for ``_write_summary()``.
        """
        n_usable     = self.n_extracted + self.n_skipped_cached
        global_miss  = (
            self.total_missing / self.total_frames
            if self.total_frames > 0 else 0.0
        )
        n_queued_eff = max(self.n_queued, 1)

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
                "run_id":               self._run_id,
                "status":               status,
                "started_utc":          self._started_utc,
                "completed_utc":        datetime.now(timezone.utc).isoformat(),
                "max_missing_frame_pct": self._max_missing_pct,
            },
            "aggregate": {
                "n_queued":              self.n_queued,
                "n_extracted":           self.n_extracted,
                "n_skipped_cached":      self.n_skipped_cached,
                "n_skipped_policy":      self.n_skipped_policy,
                "n_skipped_error":       self.n_skipped_error,
                "n_usable":              n_usable,
                "policy_skip_rate":      round(self.n_skipped_policy / n_queued_eff, 4),
                "error_rate":            round(self.n_skipped_error   / n_queued_eff, 4),
                "total_frames":          self.total_frames,
                "total_missing_frames":  self.total_missing,
                "global_missing_rate":   round(global_miss, 4),
                "total_proc_sec":        round(self.total_proc_sec, 1),
                "mean_proc_sec_per_clip": round(
                    self.total_proc_sec / max(self.n_extracted, 1), 3
                ),
            },
            "per_sign": per_sign,
            "per_clip": self._records,
        }

    @property
    def n_usable(self) -> int:
        """Clips usable for training: freshly extracted + cache-hit."""
        return self.n_extracted + self.n_skipped_cached


# ---------------------------------------------------------------------------
# Summary JSON writer (latest + history)
# ---------------------------------------------------------------------------

def _write_summary(
    summary: dict[str, Any],
    summary_dir: str,
    logger,
) -> None:
    """
    Write the current run's summary to two files:

    - ``preprocessing_summary_latest.json``: always overwritten; used by
      Notebook 02 for the missing-landmark analysis.
    - ``preprocessing_summary_history.json``: append-only audit log;
      each run appends one entry. The ``per_clip`` list is intentionally
      excluded from the history file to keep it compact.

    The ``per_clip`` list is retained in the ``_latest`` file only.
    For WLASL-35 (350 clips), the ``_latest`` file stays well under 1 MB.
    For datasets > 10 k clips, consider passing ``include_per_clip=False``.

    Parameters
    ----------
    summary : dict
        Current run's summary dict from ``_RunStats.to_dict()``.
    summary_dir : str
        Directory to write both JSON files into.
    logger
        Active logger.
    """
    out_dir = Path(summary_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    latest_path  = out_dir / "preprocessing_summary_latest.json"
    history_path = out_dir / "preprocessing_summary_history.json"

    # --- Latest (full payload, always overwritten) ---
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    # --- History (compact entry without per_clip, appended) ---
    compact = {k: v for k, v in summary.items() if k != "per_clip"}

    existing_runs: list[dict[str, Any]] = []
    if history_path.exists():
        try:
            with open(history_path, encoding="utf-8") as f:
                data = json.load(f)
            existing_runs = data if isinstance(data, list) else [data]
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                f"Could not read history file (will start fresh): {exc}",
                extra={"stage": "extraction"},
            )

    existing_runs.append(compact)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(existing_runs, f, indent=2, default=str)

    n_clips = len(summary.get("per_clip", []))
    logger.info(
        f"Extraction summary written | "
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
    Iterate through clip records, extract landmarks, and write .npy files.

    Key design decisions implemented here:
    - ETA is computed from a fixed pre-loop baseline so it never drifts as
      files are written (fix for critical issue #2).
    - .npy files are written atomically via tmp-file + rename (fix for
      critical issue #8).
    - Each clip is isolated: any exception is caught, logged, and counted as
      skipped_error without aborting the rest of the run.
    - Cache-hit files can be verified with ``verify_existing=True`` (fix for
      high-priority issue #7).
    - The ``extractor`` parameter is Optional — None is only valid when
      ``dry_run=True``, which is checked before the loop begins (fix for
      issue #6).

    Parameters
    ----------
    clips : list[dict]
        Ordered clip records from ``_collect_clips()``.
    extractor : LandmarkExtractor | None
        Initialised extractor. Must be non-None when ``dry_run=False``.
    landmarks_dir : str
        Root output directory for .npy files.
    run_stats : _RunStats
        Statistics accumulator.
    force : bool
        Re-extract even if a valid .npy + .meta.json exists.
    verify_existing : bool
        Spot-check cached .npy files before trusting them.
    dry_run : bool
        If True, log plan only — never call MediaPipe or write files.
    logger
        Active logger.
    """
    run_stats.n_queued = len(clips)
    loop_start = time.time()

    if dry_run:
        _report_dry_run_plan(clips, landmarks_dir, force, logger)
        for clip in clips:
            run_stats.record_dry_run(clip)
        return

    # Pre-compute how many clips actually need extraction (not already cached).
    # This count is fixed before the loop starts so ETA never drifts.
    n_to_extract_initially = sum(
        1 for c in clips
        if force or not _get_output_path(
            landmarks_dir, c["split"], c["safe_sign_label"], c["video_id"]
        ).exists()
    )
    logger.info(
        f"Clips to extract: {n_to_extract_initially} | "
        f"clips already cached: {len(clips) - n_to_extract_initially}",
        extra={"stage": "extraction"},
    )

    n_newly_processed = 0   # extraction calls made (for ETA denominator)

    for i, clip in enumerate(clips):
        video_id        = clip["video_id"]
        sign_label      = clip["sign_label"]
        safe_sign_label = clip["safe_sign_label"]
        split_name      = clip["split"]
        video_path      = clip["video_path"]

        npy_path = _get_output_path(
            landmarks_dir, split_name, safe_sign_label, video_id
        )

        # ----------------------------------------------------------------
        # Resumability: check for existing valid .npy
        # ----------------------------------------------------------------
        if npy_path.exists() and not force:
            if verify_existing:
                if not _verify_npy_file(npy_path, video_id, logger, full_check=False):
                    logger.info(
                        f"Cached file failed verification — reprocessing: "
                        f"{npy_path.name} | video_id={video_id}",
                        extra={"stage": "extraction", "video_id": video_id},
                    )
                    # Fall through to extraction below
                else:
                    # Cache hit and verified — restore stats from sidecar
                    meta = _read_sidecar(npy_path)
                    run_stats.record_cached(
                        clip,
                        output_path=str(npy_path),
                        n_frames=meta.get("num_frames", 0)               if meta else 0,
                        missing_pct=meta.get("missing_pct", 0.0)         if meta else 0.0,
                        missing_both=meta.get("missing_both_hands_frames", 0) if meta else 0,
                    )
                    logger.debug(
                        f"Cache hit: {video_id} ({sign_label}) | path={npy_path}",
                        extra={"stage": "extraction", "video_id": video_id},
                    )
                    continue
            else:
                # No verification — trust existing file and restore sidecar stats
                meta = _read_sidecar(npy_path)
                run_stats.record_cached(
                    clip,
                    output_path=str(npy_path),
                    n_frames=meta.get("num_frames", 0)               if meta else 0,
                    missing_pct=meta.get("missing_pct", 0.0)         if meta else 0.0,
                    missing_both=meta.get("missing_both_hands_frames", 0) if meta else 0,
                )
                logger.debug(
                    f"Cache hit (unverified): {video_id} ({sign_label})",
                    extra={"stage": "extraction", "video_id": video_id},
                )
                continue

        # ----------------------------------------------------------------
        # Validate video file on disk
        # ----------------------------------------------------------------
        resolved_path = _resolve_video_path(video_path, video_id, logger)
        if resolved_path is None:
            run_stats.record_error(clip, error_msg="video_file_not_found")
            n_newly_processed += 1
            continue

        # ----------------------------------------------------------------
        # Extract landmarks via LandmarkExtractor
        # ----------------------------------------------------------------
        clip_start = time.time()
        output_path_for_extractor = npy_path   # extractor writes its own file

        try:
            result: ExtractionResult = extractor.extract_video(  # type: ignore[union-attr]
                video_path=str(resolved_path),
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
        # Route by result status
        # ----------------------------------------------------------------
        if result.status == "skipped":
            logger.info(
                f"Skipped (policy: >{run_stats._max_missing_pct:.0%} both-hands "
                f"absent) | video_id={video_id} | sign={sign_label} | "
                f"missing_pct={result.missing_pct:.1%}",
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
            run_stats.record_error(clip, error_msg=result.error_message, proc_sec=proc_sec)
            n_newly_processed += 1
            continue

        if result.status == "cached":
            # extractor.extract_video() returned a cache hit (--force was False
            # and the extractor itself found a valid .npy). This happens when
            # the orchestration layer races the extractor's own cache check.
            run_stats.record_cached(
                clip,
                output_path=result.output_path,
                n_frames=result.num_frames,
                missing_pct=result.missing_pct,
                missing_both=result.missing_both_hands_frames,
            )
            n_newly_processed += 1
            continue

        # status == "extracted" — the extractor has written the .npy itself
        # ----------------------------------------------------------------
        # Post-write verification — every freshly written array is checked
        # ----------------------------------------------------------------
        if not _verify_npy_file(npy_path, video_id, logger, full_check=True):
            logger.error(
                f"Verification failed — removing corrupt .npy: {npy_path}",
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
            f"missing={result.missing_pct:.1%} | "
            f"time={proc_sec:.2f}s",
            extra={"stage": "extraction", "video_id": video_id},
        )
        run_stats.record_extracted(
            clip, result, proc_sec, output_path=str(npy_path)
        )
        n_newly_processed += 1

        # ----------------------------------------------------------------
        # Progress logging every _LOG_INTERVAL newly-processed clips.
        # ETA is based on n_to_extract_initially (fixed before loop) so
        # it never drifts as files are written to disk.
        # ----------------------------------------------------------------
        if n_newly_processed % _LOG_INTERVAL == 0:
            elapsed = time.time() - loop_start
            rate    = n_newly_processed / elapsed if elapsed > 0 else 0.0
            remaining = max(n_to_extract_initially - n_newly_processed, 0)
            eta_sec = remaining / rate if rate > 0 else 0.0

            logger.info(
                f"Progress | {i + 1}/{len(clips)} queued | "
                f"{run_stats.n_extracted} extracted | "
                f"{run_stats.n_skipped_cached} cached | "
                f"{run_stats.n_skipped_policy} policy-skip | "
                f"{run_stats.n_skipped_error} errors | "
                f"rate={rate:.1f} clips/s | "
                f"ETA={eta_sec / 60:.1f}min",
                extra={"stage": "extraction"},
            )

    # ----------------------------------------------------------------
    # Final summary line (covers runs shorter than _LOG_INTERVAL)
    # ----------------------------------------------------------------
    elapsed = time.time() - loop_start
    logger.info(
        f"Extraction loop done | elapsed={elapsed:.1f}s | "
        f"extracted={run_stats.n_extracted} | "
        f"cached={run_stats.n_skipped_cached} | "
        f"policy_skip={run_stats.n_skipped_policy} | "
        f"errors={run_stats.n_skipped_error}",
        extra={"stage": "extraction"},
    )


# ---------------------------------------------------------------------------
# Sidecar metadata helpers (pipeline-layer access)
# ---------------------------------------------------------------------------

def _get_sidecar_path(npy_path: Path) -> Path:
    """Return the .meta.json sidecar path for a given .npy file."""
    return npy_path.with_suffix(".meta.json")


def _read_sidecar(npy_path: Path) -> Optional[dict[str, Any]]:
    """
    Load sidecar metadata for a cached .npy file.

    Returns None if the sidecar is absent or unreadable, which causes
    the cache-hit record to be stored with zero-value statistics rather
    than failing the run.

    Parameters
    ----------
    npy_path : Path
        Path to the .npy file.

    Returns
    -------
    dict | None
        Parsed sidecar JSON, or None on any failure.
    """
    meta_path = _get_sidecar_path(npy_path)
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


# ---------------------------------------------------------------------------
# Post-run reporting helpers
# ---------------------------------------------------------------------------

def _log_extraction_report(run_stats: _RunStats, logger) -> None:
    """
    Emit a structured, human-readable extraction report at INFO level.

    Uses the actual runtime max_missing_frame_pct threshold from RunStats
    (not a hardcoded module constant) so the report accurately reflects
    what threshold was in effect for this run.

    Parameters
    ----------
    run_stats : _RunStats
        Completed run statistics accumulator.
    logger
        Active logger.
    """
    total_frames  = run_stats.total_frames
    total_missing = run_stats.total_missing
    global_miss   = (total_missing / total_frames) if total_frames > 0 else 0.0
    n_queued_eff  = max(run_stats.n_queued, 1)

    logger.info("=" * 65, extra={"stage": "extraction"})
    logger.info("STAGE 3 — EXTRACTION REPORT", extra={"stage": "extraction"})
    logger.info("=" * 65, extra={"stage": "extraction"})
    logger.info(f"  Queued            : {run_stats.n_queued}", extra={"stage": "extraction"})
    logger.info(f"  Extracted (fresh) : {run_stats.n_extracted}", extra={"stage": "extraction"})
    logger.info(f"  Loaded (cache)    : {run_stats.n_skipped_cached}", extra={"stage": "extraction"})
    logger.info(
        f"  Usable total      : {run_stats.n_usable}",
        extra={"stage": "extraction"},
    )
    logger.info(
        f"  Skipped (policy)  : {run_stats.n_skipped_policy}  "
        f"(>{run_stats._max_missing_pct:.0%} both-hands absent — "
        f"{run_stats.n_skipped_policy / n_queued_eff:.1%} of queued)",
        extra={"stage": "extraction"},
    )
    logger.info(
        f"  Skipped (error)   : {run_stats.n_skipped_error}  "
        f"({run_stats.n_skipped_error / n_queued_eff:.1%} of queued)",
        extra={"stage": "extraction"},
    )
    if total_frames > 0:
        logger.info(
            f"  Total frames      : {total_frames:,}",
            extra={"stage": "extraction"},
        )
        logger.info(
            f"  Missing rate      : {global_miss:.2%}  "
            f"({total_missing:,}/{total_frames:,} frames zero-filled)",
            extra={"stage": "extraction"},
        )
    logger.info(
        f"  Processing time   : {run_stats.total_proc_sec:.1f}s",
        extra={"stage": "extraction"},
    )
    logger.info("=" * 65, extra={"stage": "extraction"})


def _validate_extraction_health(
    run_stats: _RunStats,
    logger,
) -> bool:
    """
    Check overall extraction health and emit actionable warnings.

    Returns False if any threshold is exceeded. This is not a hard failure —
    the caller decides whether to exit with code 2. The thresholds are
    informed by the project handoff document's expected values.

    Thresholds
    ----------
    - Policy skip rate > 10%: higher than expected 5–8% → MediaPipe issue?
    - Error rate > 5%: filesystem or video corruption concern
    - Global missing rate > 15%: MediaPipe detection quality concern

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
    n_eff = max(run_stats.n_queued, 1)

    policy_rate = run_stats.n_skipped_policy / n_eff
    if policy_rate > 0.10:
        logger.warning(
            f"Policy skip rate {policy_rate:.1%} exceeds 10% threshold "
            f"(expected 5–8% for WLASL). "
            "Consider raising --max-missing-frame-pct or reviewing "
            "--min-detection-confidence settings.",
            extra={"stage": "extraction"},
        )
        healthy = False

    error_rate = run_stats.n_skipped_error / n_eff
    if error_rate > 0.05:
        logger.warning(
            f"Error rate {error_rate:.1%} exceeds 5% threshold. "
            "Review skipped_error records in preprocessing_summary_latest.json.",
            extra={"stage": "extraction"},
        )
        healthy = False

    if run_stats.total_frames > 0:
        global_miss = run_stats.total_missing / run_stats.total_frames
        if global_miss > 0.15:
            logger.warning(
                f"Global missing-landmark rate {global_miss:.1%} exceeds 15%. "
                "Consider reviewing video quality for high-miss signs, or "
                "raising --min-detection-confidence.",
                extra={"stage": "extraction"},
            )
            healthy = False

    return healthy


def _log_output_inventory(landmarks_dir: str, logger) -> None:
    """
    Walk the landmarks directory and log file counts per split and sign.

    Uses a generator expression (``sum(1 for _ in glob())``) instead of
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

    total_files = 0
    split_counts: dict[str, int] = {}
    sign_totals:  dict[str, int] = defaultdict(int)

    for split_dir in sorted(root.iterdir()):
        if not split_dir.is_dir():
            continue
        n_in_split = 0
        for sign_dir in sorted(split_dir.iterdir()):
            if not sign_dir.is_dir():
                continue
            n_files = sum(1 for _ in sign_dir.glob("*.npy"))
            n_in_split   += n_files
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
# MediaPipe extractor initialisation
# ---------------------------------------------------------------------------

def _init_extractor(args: argparse.Namespace, logger) -> Optional[LandmarkExtractor]:
    """
    Initialise the LandmarkExtractor with runtime configuration.

    Returns None on failure (caller should return exit code 2).

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    logger
        Active logger.

    Returns
    -------
    LandmarkExtractor | None
    """
    logger.info(
        "Initialising MediaPipe Holistic extractor...",
        extra={"stage": "extraction"},
    )
    try:
        extractor = LandmarkExtractor(
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
        )
        # Warm up MediaPipe before the main loop to avoid cold-start timing
        # on the first clip (model load takes 2–5 seconds).
        extractor._init_mediapipe()
        return extractor
    except Exception as exc:
        logger.error(
            f"Failed to initialise LandmarkExtractor: {type(exc).__name__}: {exc}. "
            "Ensure mediapipe==0.10.14 is installed in the active environment.",
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
        Exit code: 0=success, 1=input/validation error, 2=runtime failure.
    """
    parser = _build_parser()
    args   = parser.parse_args()

    # ----------------------------------------------------------------
    # Logging — must be configured before any other operation
    # ----------------------------------------------------------------
    log_level  = "DEBUG" if args.verbose else "INFO"
    run_label  = "stage3_sample" if args.sample_only else f"stage3_{args.split}"
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
        "WLASL Gesture Recognition — Stage 3: Landmark Extraction",
        extra={"stage": "extraction"},
    )
    logger.info(f"Log file: {log_file}", extra={"stage": "extraction"})

    mode_str = (
        f"SAMPLE ({_SAMPLE_CLIPS_PER_SIGN} clip/sign/split)"
        if args.sample_only
        else args.split.upper()
    )
    logger.info(
        f"Mode: {mode_str} | force={args.force} | "
        f"verify_existing={args.verify_existing} | dry_run={args.dry_run}",
        extra={"stage": "extraction"},
    )
    logger.info(
        f"Extractor config | "
        f"max_missing_frame_pct={args.max_missing_frame_pct:.0%} | "
        f"min_detection_conf={args.min_detection_confidence} | "
        f"min_tracking_conf={args.min_tracking_confidence}",
        extra={"stage": "extraction"},
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
            "No clips to process. Verify split CSVs exist and contain "
            f"video_path entries: {args.splits_dir}",
            extra={"stage": "extraction"},
        )
        return 1

    # ----------------------------------------------------------------
    # Run statistics accumulator
    # ----------------------------------------------------------------
    run_id     = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_stats  = _RunStats(
        run_id=run_id,
        max_missing_frame_pct=args.max_missing_frame_pct,
    )

    # ----------------------------------------------------------------
    # Dry-run short-circuit — no MediaPipe needed
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
        f"Output root: {args.landmarks_dir}",
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
            f"Extraction interrupted (KeyboardInterrupt). "
            f"{run_stats.n_extracted} clips extracted before interruption. "
            "Re-run to continue — existing .npy files are not re-processed.",
            extra={"stage": "extraction"},
        )
        # Write partial summary before exiting so the user can review progress
        summary = run_stats.to_dict(status="PARTIAL_INTERRUPTED")
        try:
            _write_summary(summary, args.summary_dir, logger)
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
        summary = run_stats.to_dict(status="PARTIAL_INTERRUPTED")
        try:
            _write_summary(summary, args.summary_dir, logger)
        except Exception:
            pass
        return 2

    # ----------------------------------------------------------------
    # Post-run reporting
    # ----------------------------------------------------------------
    total_elapsed = time.time() - pipeline_start

    _log_extraction_report(run_stats, logger)
    health_ok = _validate_extraction_health(run_stats, logger)

    if not health_ok:
        logger.warning(
            "One or more health checks exceeded their thresholds. "
            "Review warnings above before proceeding to Stage 4.",
            extra={"stage": "extraction"},
        )

    # ----------------------------------------------------------------
    # Write summary JSON (latest + history)
    # ----------------------------------------------------------------
    try:
        summary = run_stats.to_dict(status="completed")
        _write_summary(summary, args.summary_dir, logger)
    except Exception as exc:
        logger.error(
            f"Failed to write extraction summary: {exc}",
            extra={"stage": "extraction"},
        )
        # Non-fatal — the .npy files are the critical output

    # ----------------------------------------------------------------
    # Landmark directory inventory
    # ----------------------------------------------------------------
    _log_output_inventory(args.landmarks_dir, logger)

    # ----------------------------------------------------------------
    # Sample-mode verification instructions
    # ----------------------------------------------------------------
    if args.sample_only and run_stats.n_extracted > 0:
        logger.info(
            "SAMPLE EXTRACTION COMPLETE — Manual verification:",
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
            "  assert arr.ndim == 2 and arr.shape[1] == 225  # (N, 225)",
            extra={"stage": "extraction"},
        )
        logger.info(
            "  print(arr.shape, arr.min(), arr.max())  # values in ~[0,1]",
            extra={"stage": "extraction"},
        )
        logger.info(
            "If shapes and ranges look correct, open "
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
            "and run notebooks/03_feature_engineering_experiments.ipynb.",
            extra={"stage": "extraction"},
        )
    logger.info("=" * 65, extra={"stage": "extraction"})

    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(main())