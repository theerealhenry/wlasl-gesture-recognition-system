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
extractor (v1.2 schema). The sidecar stores per-clip detection statistics
including ``detected_frames`` (frames where at least one hand was detected,
the primary v1.2 skip criterion) and ``decode_failure_frames`` (separate
from genuine MediaPipe detection failures).

IMPORTANT: Arrays store the clip's actual frame count — they are NOT padded
to seq_len=30. Padding/truncation are deferred to ``FeaturePipeline`` (Stage 4)
so the same .npy files serve all sequence-length ablation experiments
({20, 30, 40, 60} frames) without any re-extraction.

Skip policy — v1.2 (dual-criterion absolute)
---------------------------------------------
Notebook 02 analysis of the sample run revealed that the old ratio-based
policy (skip if missing_both_pct > 30%) was catastrophically wrong for WLASL:

    Expected skip rate : 5–8%
    Observed skip rate : 76% (158/208 clips skipped)
    Projected usable   : ~84 clips from 350 → ~2.4 clips per class
    Impact             : 70% accuracy target mathematically impossible

Root cause: WLASL clips contain large temporal dead zones (preparation
movements, idle frames, lead-in/lead-out segments) that inflate the ratio
without reducing training value. One-handed signs also naturally inflate
``missing_left`` or ``missing_right``, and when neither hand overlaps, the
``missing_both`` rate approaches the percentage of "resting" frames.

The correct question is "does this clip contain enough usable frames for
the LSTM to learn from?" ``detected_frames`` directly answers this.

v1.2 dual-criterion policy (implemented in extractor.py, enforced here):

    KEEP clip if ALL of the following hold:
      (a) detected_frames >= min_detected_frames  (default: 15)
              At least 15 frames where at least one hand was detected.
      (b) missing_pct <= max_missing_pct          (default: 0.95)
              Catastrophe filter: 95%+ both-absent means the clip is
              genuinely unusable (corrupt file, signer never visible).

    SKIP clip if either criterion fails.

Expected outcomes with v1.2:
    Expected skip rate : ~3–6%
    Expected usable    : ~330–340 clips from 350
    Mean clips/class   : ~9.4–9.7

Resumability
------------
The extractor itself checks for a valid .npy + .meta.json pair before doing
any work (shape, dtype, schema version v1.2, full finiteness scan). This
pipeline respects those cache hits and adds an optional ``--verify-existing``
path to spot-check cached files independently before trusting them.

Summary outputs
---------------
Two pipeline-level JSON summary files are written after each run:

    data/preprocessing_summary_latest.json  — current run (always overwritten)
    data/preprocessing_summary_history.json — append-only audit log

A ``landmark_inventory.csv`` is also written by the extractor after each
batch run. Notebook 02 loads this CSV for the missing-landmark analysis.

Per-hand missing rate fix (Notebook 02 Bug 9.1)
-------------------------------------------------
The original v1.1 pipeline zeroed ``n_missing_left``, ``n_missing_right``,
and ``n_missing_pose`` in ``_finalise_run()``, making the per-hand columns in
``landmark_inventory.csv`` always 0.0. This is fixed in v1.2 by propagating
per-hand missing counts through ``_RunStats.record_extracted()`` and
``_RunStats.record_cached()``, and correctly mapping them in ``_finalise_run()``.

Schema alignment with extractor.py v1.2
-----------------------------------------
This script is fully aligned with ``src/features/extractor.py`` schema
version 1.2. Key changes from v1.1:

  - ``detected_frames`` field is now propagated through all
    ``_RunStats.record_*`` methods and included in summary outputs.
  - Primary skip criterion is ``detected_frames < min_detected_frames``,
    not a missing ratio. The ``--max-missing-frame-pct`` CLI argument now
    sets the catastrophe filter threshold (default: 0.95) rather than
    the primary threshold (which was 0.30 in v1.1).
  - New ``--min-detected-frames`` CLI argument controls the primary criterion.
  - Cache-hit clips restore ``detected_frames`` from .meta.json sidecar.
  - Health check thresholds recalibrated for v1.2 expected outcomes.
  - Per-hand missing rate bug (Notebook 02 Bug 9.1) is fixed.

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

Override skip policy thresholds:
    python pipelines/run_landmark_extraction.py --split all \\
        --min-detected-frames 20 \\
        --max-missing-frame-pct 0.90

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
- Aligned with extractor.py schema version 1.2 throughout
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
# Feature modules — import constants first (no heavy deps), then extractor
# ---------------------------------------------------------------------------
from src.features.constants import (
    FEATURE_SIZE,
    EXTRACTOR_SCHEMA_VERSION,
    MIN_DETECTED_FRAMES_DEFAULT,
    MAX_MISSING_PCT_CATASTROPHE,
    HEALTH_POLICY_SKIP_RATE_WARN,
    HEALTH_ERROR_RATE_WARN,
    HEALTH_GLOBAL_MISSING_RATE_WARN,
)
from src.features.extractor import (
    LandmarkExtractor,
    ExtractionResult,
    write_landmark_inventory,
)


# ---------------------------------------------------------------------------
# Default paths and constants
# ---------------------------------------------------------------------------
_DEFAULT_SPLITS_DIR    = str(_REPO_ROOT / "data" / "splits")
_DEFAULT_LANDMARKS_DIR = str(_REPO_ROOT / "data" / "landmarks")
_DEFAULT_SUMMARY_DIR   = str(_REPO_ROOT / "data")
_DEFAULT_LOG_DIR       = str(_REPO_ROOT / "logs")

_VALID_SPLITS = ("train", "val", "test", "all")
_LOG_INTERVAL = 50    # clips between progress log lines

# Must match extractor.py default: min(3, available_clips) per sign
_SAMPLE_CLIPS_PER_SIGN = 3

_SEED = 42

# Characters unsafe as filesystem path components on any OS
_UNSAFE_PATH_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


# ---------------------------------------------------------------------------
# Filesystem safety helper
# ---------------------------------------------------------------------------

def _sanitize_path_component(name: str) -> str:
    """
    Replace characters unsafe in a filesystem path component with underscores.

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
            "Extractor schema version: 1.2 (dual-criterion absolute skip policy)\n"
            "  Primary criterion : detected_frames >= min_detected_frames (default: 15)\n"
            "  Secondary criterion: missing_both_pct <= max_missing_pct (default: 0.95)\n\n"
            "Notebook 02 analysis found the previous 30% ratio policy produced a 76%\n"
            "skip rate on WLASL (expected 5-8%). The v1.2 policy is expected to retain\n"
            "~94-97% of clips.\n\n"
            "Always run --sample-only first to validate the extractor before\n"
            "committing to the full 30-90 minute extraction."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Stage 2 validation gate (3 clips/sign, all splits — ~2-5 minutes)
  python pipelines/run_landmark_extraction.py --sample-only

  # Full extraction, all splits (30-90 minutes) — PRIMARY USAGE
  python pipelines/run_landmark_extraction.py --split all

  # Training split only
  python pipelines/run_landmark_extraction.py --split train

  # Force re-extraction (overwrite existing .npy + .meta.json)
  python pipelines/run_landmark_extraction.py --split all --force

  # Resume and verify previously cached files (after interruption)
  python pipelines/run_landmark_extraction.py --split all --verify-existing

  # Dry run — validate inputs and log plan, write nothing
  python pipelines/run_landmark_extraction.py --dry-run

  # Verbose debug output
  python pipelines/run_landmark_extraction.py --sample-only --verbose

  # Override v1.2 skip policy thresholds
  python pipelines/run_landmark_extraction.py --split all \\
      --min-detected-frames 20 --max-missing-frame-pct 0.90

  # Run threshold diagnostic on existing sample summary
  python pipelines/run_landmark_extraction.py --threshold-diagnostic \\
      data/preprocessing_summary_latest.json

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
    mode_group.add_argument(
        "--threshold-diagnostic",
        metavar="SUMMARY_JSON",
        help=(
            "Run a skip-threshold sensitivity analysis on an existing "
            "preprocessing_summary_latest.json and exit. Prints expected "
            "retention at multiple threshold values. "
            "Useful for selecting optimal --min-detected-frames before "
            "committing to a full extraction run."
        ),
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
            "Note: cached files that pass validation are always skipped unless --force "
            "is set, even if their sidecar schema version is current."
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
    # v1.2 Skip policy configuration
    # ----------------------------------------------------------------
    parser.add_argument(
        "--min-detected-frames",
        type=int,
        default=MIN_DETECTED_FRAMES_DEFAULT,
        metavar="N",
        help=(
            "PRIMARY skip criterion (v1.2): minimum number of frames where at "
            "least one hand must be detected for the clip to be retained. "
            f"Default: {MIN_DETECTED_FRAMES_DEFAULT}. "
            "Clips with fewer detected frames are skipped with reason "
            "'insufficient_detected_frames'. "
            "Notebook 02 analysis: 15 is the floor below which there is "
            "insufficient temporal context for seq_len=20 training."
        ),
    )
    parser.add_argument(
        "--max-missing-frame-pct",
        type=float,
        default=MAX_MISSING_PCT_CATASTROPHE,
        metavar="RATIO",
        help=(
            "SECONDARY skip criterion — catastrophe filter (v1.2): skip if "
            "the fraction of successfully-decoded frames with BOTH hands absent "
            f"exceeds this threshold. Default: {MAX_MISSING_PCT_CATASTROPHE} (95%%). "
            "This catches corrupt files and clips where the signer is never visible. "
            "Note: in v1.1 this was the PRIMARY criterion at 0.30, which produced "
            "a 76%% skip rate on WLASL. In v1.2 it is a catastrophe filter only."
        ),
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.5,
        metavar="CONF",
        help=(
            "MediaPipe Holistic minimum detection confidence (default: 0.5). "
            "MUST be identical between extraction (Stage 3) and inference (Stage 7)."
        ),
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.5,
        metavar="CONF",
        help=(
            "MediaPipe Holistic minimum tracking confidence (default: 0.5). "
            "MUST be identical between extraction (Stage 3) and inference (Stage 7)."
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
# Threshold diagnostic (Notebook 02 recommendation, Section 10.3)
# ---------------------------------------------------------------------------

def run_threshold_diagnostic(summary_json_path: str, logger) -> int:
    """
    Print skip-threshold sensitivity analysis from an existing summary JSON.

    Reads the per-clip records from ``preprocessing_summary_latest.json``
    and prints expected retention at multiple ``min_detected_frames`` values.
    This helps select the optimal threshold before committing to a full run.

    Uses ``detected_frames`` (v1.2 primary criterion) rather than
    ``missing_pct`` (v1.1 ratio criterion).

    Parameters
    ----------
    summary_json_path : str
        Path to preprocessing_summary_latest.json.
    logger
        Active logger.

    Returns
    -------
    int
        Exit code: 0 on success, 1 on error.
    """
    path = Path(summary_json_path)
    if not path.exists():
        logger.error(
            f"Summary JSON not found: {path}. "
            "Run --sample-only first to generate it.",
            extra={"stage": "extraction"},
        )
        return 1

    try:
        with open(path, encoding="utf-8") as f:
            summary = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.error(
            f"Could not read summary JSON: {exc}",
            extra={"stage": "extraction"},
        )
        return 1

    # Support both pipeline-level summary (_RunStats.to_dict) and
    # extractor-level summary (ExtractionStats.to_dict) formats
    per_clip = summary.get("per_clip", [])
    if not per_clip:
        logger.error(
            "No per_clip data found in summary JSON. "
            "The file may be from an older run or may be the history file.",
            extra={"stage": "extraction"},
        )
        return 1

    # Collect detected_frames and missing_pct for all non-error clips
    records = [
        {
            "detected_frames": r.get("detected_frames", r.get("n_frames", 0)),
            "missing_pct":     r.get("missing_pct", 0.0),
        }
        for r in per_clip
        if r.get("outcome", r.get("status", "error")) != "error"
    ]
    total = len(records)

    if total == 0:
        logger.error(
            "No non-error clips found in per_clip data.",
            extra={"stage": "extraction"},
        )
        return 1

    thresholds_detected = [5, 10, 15, 20, 25, 30]
    thresholds_missing  = [0.70, 0.80, 0.90, 0.95, 1.00]

    logger.info("=" * 65, extra={"stage": "extraction"})
    logger.info("THRESHOLD DIAGNOSTIC", extra={"stage": "extraction"})
    logger.info(
        f"  Based on {total} clips from: {path}",
        extra={"stage": "extraction"},
    )
    logger.info("=" * 65, extra={"stage": "extraction"})

    logger.info(
        "PRIMARY CRITERION: min_detected_frames (v1.2)",
        extra={"stage": "extraction"},
    )
    logger.info(
        f"  {'min_detected':>14} | {'Retained':>10} | {'Skipped':>10} | {'Ret. rate':>10}",
        extra={"stage": "extraction"},
    )
    logger.info("  " + "-" * 52, extra={"stage": "extraction"})
    for t in thresholds_detected:
        retained = sum(1 for r in records if r["detected_frames"] >= t)
        skipped  = total - retained
        logger.info(
            f"  {t:>14} | {retained:>10} | {skipped:>10} | {retained/total:>10.1%}",
            extra={"stage": "extraction"},
        )

    logger.info("", extra={"stage": "extraction"})
    logger.info(
        "SECONDARY CRITERION: max_missing_pct — catastrophe filter (v1.2)",
        extra={"stage": "extraction"},
    )
    logger.info(
        f"  {'max_missing_pct':>14} | {'Retained':>10} | {'Skipped':>10} | {'Ret. rate':>10}",
        extra={"stage": "extraction"},
    )
    logger.info("  " + "-" * 52, extra={"stage": "extraction"})
    for t in thresholds_missing:
        retained = sum(1 for r in records if r["missing_pct"] <= t)
        skipped  = total - retained
        logger.info(
            f"  {t:>14.0%} | {retained:>10} | {skipped:>10} | {retained/total:>10.1%}",
            extra={"stage": "extraction"},
        )

    logger.info("", extra={"stage": "extraction"})
    logger.info(
        f"Recommended v1.2 defaults: "
        f"--min-detected-frames {MIN_DETECTED_FRAMES_DEFAULT} "
        f"--max-missing-frame-pct {MAX_MISSING_PCT_CATASTROPHE}",
        extra={"stage": "extraction"},
    )
    logger.info("=" * 65, extra={"stage": "extraction"})

    return 0


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

    if args.min_detected_frames < 1:
        logger.error(
            f"--min-detected-frames must be >= 1. Got: {args.min_detected_frames}",
            extra={"stage": "extraction"},
        )
        valid = False

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
        # Dtype mismatch is a warning, not a hard failure here.

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
    that does not match ``EXTRACTOR_SCHEMA_VERSION`` (v1.2).

    Parameters
    ----------
    npy_path : Path
        Path to the .npy file.

    Returns
    -------
    dict | None
        Parsed sidecar JSON, or None on any validation failure.
    """
    meta_path = _get_sidecar_path(npy_path)
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        stored_version = meta.get("schema_version", "")
        if stored_version != EXTRACTOR_SCHEMA_VERSION:
            # Schema version mismatch: extractor will reprocess on next cache
            # check, so treat as a miss for statistics purposes too.
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

    Aligned with extractor.py v1.2:
      - ``detected_frames`` tracked per clip and aggregated (v1.2 primary criterion)
      - ``decode_failure_frames`` tracked separately from detection failures
      - ``missing_pct`` stored per-clip is over successfully-decoded frames
      - Per-hand missing counts propagated (fixes Notebook 02 Bug 9.1)
      - Cache-hit status is ``"cached"``
      - Policy-skip reason is one of: "insufficient_detected_frames",
        "catastrophic_missing_rate", "no_frames_extracted"

    Attributes
    ----------
    n_queued : int
        Total clips submitted to the extraction loop.
    n_extracted : int
        Clips freshly extracted this run.
    n_skipped_cached : int
        Clips skipped because a valid .npy + .meta.json already existed.
    n_skipped_policy : int
        Clips skipped due to the v1.2 skip policy (either criterion).
    n_skipped_error : int
        Clips skipped due to video read failure or MediaPipe exception.
    n_dry_run : int
        Clips processed in dry-run mode (no actual work done).
    total_frames : int
        Sum of total frame counts across freshly extracted clips.
    total_detected : int
        Sum of detected_frames across freshly extracted clips.
    total_missing_both : int
        Sum of missing_both_hands frame counts across extracted clips.
        Computed over successfully-decoded frames (v1.2 semantics).
    total_missing_left : int
        Sum of missing_left_hand frame counts (fixes Bug 9.1).
    total_missing_right : int
        Sum of missing_right_hand frame counts (fixes Bug 9.1).
    total_missing_pose : int
        Sum of missing_pose frame counts (fixes Bug 9.1).
    total_decode_failures : int
        Sum of decode_failure_frames across extracted clips.
    total_proc_sec : float
        Total wall-clock processing time for extracted clips.
    """

    def __init__(self, run_id: str, min_detected_frames: int, max_missing_pct: float) -> None:
        self._run_id              = run_id
        self._started_utc         = datetime.now(timezone.utc).isoformat()
        self._min_detected_frames = min_detected_frames
        self._max_missing_pct     = max_missing_pct

        self._records: list[dict[str, Any]] = []

        # Counters
        self.n_queued              = 0
        self.n_extracted           = 0
        self.n_skipped_cached      = 0
        self.n_skipped_policy      = 0
        self.n_skipped_error       = 0
        self.n_dry_run             = 0
        self.total_frames          = 0
        self.total_detected        = 0    # v1.2: sum of detected_frames
        self.total_missing_both    = 0
        self.total_missing_left    = 0    # v1.2: fix for Bug 9.1
        self.total_missing_right   = 0    # v1.2: fix for Bug 9.1
        self.total_missing_pose    = 0    # v1.2: fix for Bug 9.1
        self.total_decode_failures = 0
        self.total_proc_sec        = 0.0

        # Per-sign breakdown
        self._sign_frames:    dict[str, int] = defaultdict(int)
        self._sign_detected:  dict[str, int] = defaultdict(int)
        self._sign_missing:   dict[str, int] = defaultdict(int)
        self._sign_extracted: dict[str, int] = defaultdict(int)
        self._sign_skipped:   dict[str, int] = defaultdict(int)

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
        self.n_extracted             += 1
        self.total_frames            += result.num_frames
        self.total_detected          += result.detected_frames        # v1.2
        self.total_missing_both      += result.missing_both_hands_frames
        self.total_missing_left      += result.missing_left_hand_frames  # Bug 9.1 fix
        self.total_missing_right     += result.missing_right_hand_frames # Bug 9.1 fix
        self.total_missing_pose      += result.missing_pose_frames       # Bug 9.1 fix
        self.total_decode_failures   += result.decode_failure_frames
        self.total_proc_sec          += proc_sec
        self._sign_frames[sign]      += result.num_frames
        self._sign_detected[sign]    += result.detected_frames
        self._sign_missing[sign]     += result.missing_both_hands_frames
        self._sign_extracted[sign]   += 1

        self._records.append({
            "video_id":                clip["video_id"],
            "sign_label":              sign,
            "class_idx":               clip["class_idx"],
            "signer_id":               clip["signer_id"],
            "split":                   clip["split"],
            "video_path":              clip["video_path"],
            "output_path":             result.output_path,
            "outcome":                 "extracted",
            "proc_sec":                round(proc_sec, 4),
            "n_frames":                result.num_frames,
            "decode_failure_frames":   result.decode_failure_frames,
            "detected_frames":         result.detected_frames,           # v1.2
            "n_missing_left":          result.missing_left_hand_frames,  # Bug 9.1
            "n_missing_right":         result.missing_right_hand_frames, # Bug 9.1
            "n_missing_pose":          result.missing_pose_frames,       # Bug 9.1
            "n_missing_both":          result.missing_both_hands_frames,
            "missing_pct":             round(result.missing_pct, 4),
        })

    def record_cached(
        self,
        clip: dict[str, Any],
        output_path: str,
        n_frames: int = 0,
        detected_frames: int = 0,           # v1.2
        missing_pct: float = 0.0,
        missing_both: int = 0,
        missing_left: int = 0,              # Bug 9.1 fix
        missing_right: int = 0,             # Bug 9.1 fix
        missing_pose: int = 0,              # Bug 9.1 fix
        decode_failure_frames: int = 0,
    ) -> None:
        """
        Record a cache-hit clip (status='cached').

        All per-hand statistics are restored from the v1.2 .meta.json sidecar
        so aggregate figures remain accurate for clips not processed this run.
        """
        sign = clip["sign_label"]
        self.n_skipped_cached        += 1
        self._sign_frames[sign]      += n_frames
        self._sign_detected[sign]    += detected_frames
        self._sign_missing[sign]     += missing_both
        self._sign_extracted[sign]   += 1   # counts toward usable total

        self._records.append({
            "video_id":                clip["video_id"],
            "sign_label":              sign,
            "class_idx":               clip["class_idx"],
            "signer_id":               clip["signer_id"],
            "split":                   clip["split"],
            "video_path":              clip["video_path"],
            "output_path":             output_path,
            "outcome":                 "cached",
            "proc_sec":                0.0,
            "n_frames":                n_frames,
            "decode_failure_frames":   decode_failure_frames,
            "detected_frames":         detected_frames,   # v1.2
            "n_missing_left":          missing_left,      # Bug 9.1
            "n_missing_right":         missing_right,     # Bug 9.1
            "n_missing_pose":          missing_pose,      # Bug 9.1
            "n_missing_both":          missing_both,
            "missing_pct":             round(missing_pct, 4),
        })

    def record_skipped_policy(
        self,
        clip: dict[str, Any],
        result: ExtractionResult,
        proc_sec: float,
    ) -> None:
        """Record a clip skipped by the v1.2 dual-criterion policy."""
        sign = clip["sign_label"]
        self.n_skipped_policy      += 1
        self._sign_skipped[sign]   += 1
        self.total_proc_sec        += proc_sec

        self._records.append({
            "video_id":                clip["video_id"],
            "sign_label":              sign,
            "class_idx":               clip["class_idx"],
            "signer_id":               clip["signer_id"],
            "split":                   clip["split"],
            "video_path":              clip["video_path"],
            "output_path":             "",
            "outcome":                 "skipped_policy",
            "proc_sec":                round(proc_sec, 4),
            "n_frames":                result.num_frames,
            "decode_failure_frames":   result.decode_failure_frames,
            "detected_frames":         result.detected_frames,      # v1.2
            "n_missing_left":          result.missing_left_hand_frames,
            "n_missing_right":         result.missing_right_hand_frames,
            "n_missing_pose":          result.missing_pose_frames,
            "n_missing_both":          result.missing_both_hands_frames,
            "missing_pct":             round(result.missing_pct, 4),
            "skip_reason":             result.skip_reason,
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
            "video_id":                clip["video_id"],
            "sign_label":              sign,
            "class_idx":               clip["class_idx"],
            "signer_id":               clip["signer_id"],
            "split":                   clip["split"],
            "video_path":              clip["video_path"],
            "output_path":             "",
            "outcome":                 "error",
            "proc_sec":                round(proc_sec, 4),
            "n_frames":                0,
            "decode_failure_frames":   0,
            "detected_frames":         0,
            "n_missing_left":          0,
            "n_missing_right":         0,
            "n_missing_pose":          0,
            "n_missing_both":          0,
            "missing_pct":             0.0,
            "error_message":           error_msg,
        })

    def record_dry_run(self, clip: dict[str, Any]) -> None:
        """Record a clip in dry-run mode."""
        self.n_dry_run += 1
        self._records.append({
            "video_id":   clip["video_id"],
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

        Denominator excludes decode_failure_frames (v1.2 semantics), so this
        reflects genuine MediaPipe detection quality rather than codec noise.
        """
        denom = self.total_frames - self.total_decode_failures
        return self.total_missing_both / denom if denom > 0 else 0.0

    @property
    def global_detection_rate(self) -> float:
        """
        Fraction of successfully-decoded frames where at least one hand was detected.

        This is 1 - global_missing_rate, exposed separately for clarity in
        health reporting.
        """
        denom = self.total_frames - self.total_decode_failures
        return self.total_detected / denom if denom > 0 else 0.0

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
            total_f    = self._sign_frames.get(sign, 0)
            detected_f = self._sign_detected.get(sign, 0)
            miss_f     = self._sign_missing.get(sign, 0)
            per_sign[sign] = {
                "usable":           self._sign_extracted.get(sign, 0),
                "skipped":          self._sign_skipped.get(sign, 0),
                "total_frames":     total_f,
                "detected_frames":  detected_f,
                "missing_frames":   miss_f,
                "detection_rate":   round(detected_f / total_f, 4) if total_f > 0 else 0.0,
                "missing_rate":     round(miss_f / total_f, 4)     if total_f > 0 else 0.0,
            }

        return {
            "_run_metadata": {
                "run_id":                self._run_id,
                "status":                status,
                "started_utc":           self._started_utc,
                "completed_utc":         datetime.now(timezone.utc).isoformat(),
                "extractor_schema":      EXTRACTOR_SCHEMA_VERSION,
                "skip_policy":           "dual_criterion_v1.2",
                "min_detected_frames":   self._min_detected_frames,
                "max_missing_pct":       self._max_missing_pct,
            },
            "aggregate": {
                "n_queued":              self.n_queued,
                "n_extracted":           self.n_extracted,
                "n_cached":              self.n_skipped_cached,
                "n_skipped_policy":      self.n_skipped_policy,
                "n_skipped_error":       self.n_skipped_error,
                "n_usable":              self.n_usable,
                "policy_skip_rate":      round(self.n_skipped_policy / n_eff, 4),
                "error_rate":            round(self.n_skipped_error   / n_eff, 4),
                "total_frames":          self.total_frames,
                "total_detected":        self.total_detected,           # v1.2
                "total_decode_failures": self.total_decode_failures,
                "total_missing_left":    self.total_missing_left,       # Bug 9.1 fix
                "total_missing_right":   self.total_missing_right,      # Bug 9.1 fix
                "total_missing_pose":    self.total_missing_pose,       # Bug 9.1 fix
                "total_missing_both":    self.total_missing_both,
                "global_missing_rate":   round(self.global_missing_rate,    4),
                "global_detection_rate": round(self.global_detection_rate,  4),  # v1.2
                "total_proc_sec":        round(self.total_proc_sec, 1),
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
    - Cache-hit routing uses ``result.status == "cached"`` (v1.2 extractor).
    - Post-write verification (``_verify_npy_file``) is an independent check
      applied after each fresh extraction as defence-in-depth.
    - Each clip is isolated: any exception is caught, logged, and counted as
      an error without aborting the rest of the run.
    - ``--verify-existing`` triggers a spot-check on cache-hit files; corrupt
      ones are deleted so the extractor reprocesses them on the same call.
    - ``extractor`` is None only when ``dry_run=True``.

    Parameters
    ----------
    clips : list[dict]
        Ordered clip records from ``_collect_clips()``.
    extractor : LandmarkExtractor | None
        Initialised extractor. Must be non-None when dry_run=False.
    landmarks_dir : str
        Root output directory for .npy files.
    run_stats : _RunStats
        Statistics accumulator (v1.2: includes detected_frames, per-hand counts).
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

    # Pre-compute how many clips actually need extraction (fixed before loop
    # so ETA never drifts as files are written during the run).
    n_to_process_initially = sum(
        1 for c in clips
        if force or not _get_output_path(
            landmarks_dir, c["split"], c["safe_sign_label"], c["video_id"]
        ).exists()
    )
    logger.info(
        f"Clips requiring extraction: {n_to_process_initially} | "
        f"already cached (fast path): {len(clips) - n_to_process_initially}",
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
                        detected_frames=      meta.get("detected_frames",          0)   if meta else 0,  # v1.2
                        missing_pct=          meta.get("missing_pct",              0.0) if meta else 0.0,
                        missing_both=         meta.get("missing_both_hands_frames",0)   if meta else 0,
                        missing_left=         meta.get("missing_left_hand_frames", 0)   if meta else 0,  # Bug 9.1
                        missing_right=        meta.get("missing_right_hand_frames",0)   if meta else 0,  # Bug 9.1
                        missing_pose=         meta.get("missing_pose_frames",      0)   if meta else 0,  # Bug 9.1
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
                    detected_frames=      meta.get("detected_frames",          0)   if meta else 0,  # v1.2
                    missing_pct=          meta.get("missing_pct",              0.0) if meta else 0.0,
                    missing_both=         meta.get("missing_both_hands_frames",0)   if meta else 0,
                    missing_left=         meta.get("missing_left_hand_frames", 0)   if meta else 0,  # Bug 9.1
                    missing_right=        meta.get("missing_right_hand_frames",0)   if meta else 0,  # Bug 9.1
                    missing_pose=         meta.get("missing_pose_frames",      0)   if meta else 0,  # Bug 9.1
                    decode_failure_frames=meta.get("decode_failure_frames",    0)   if meta else 0,
                )
                logger.debug(
                    f"Cache hit: {video_id} ({sign_label})",
                    extra={"stage": "extraction", "video_id": video_id},
                )
                continue

        # ----------------------------------------------------------------
        # Validate video file on disk before calling the extractor.
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
        # tracking (v1.1+), v1.2 dual-criterion policy, .npy write, and
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
            # Extractor's own cache check triggered (e.g. race with --force=False).
            run_stats.record_cached(
                clip,
                output_path=result.output_path,
                n_frames=             result.num_frames,
                detected_frames=      result.detected_frames,
                missing_pct=          result.missing_pct,
                missing_both=         result.missing_both_hands_frames,
                missing_left=         result.missing_left_hand_frames,
                missing_right=        result.missing_right_hand_frames,
                missing_pose=         result.missing_pose_frames,
                decode_failure_frames=result.decode_failure_frames,
            )
            n_newly_processed += 1
            continue

        if result.status == "skipped":
            logger.info(
                f"Skipped (v1.2 policy: {result.skip_reason}) | "
                f"video_id={video_id} | sign={sign_label} | "
                f"detected={result.detected_frames} | "
                f"min_detected={run_stats._min_detected_frames} | "
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
            f"detected={result.detected_frames} | "
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

    Uses v1.2 semantics:
    - Primary criterion is detected_frames (not missing ratio)
    - decode_failure_frames tracked separately
    - Per-hand missing rates included (Bug 9.1 fix)
    - global_detection_rate reported alongside global_missing_rate

    Parameters
    ----------
    run_stats : _RunStats
        Completed run statistics.
    logger
        Active logger.
    """
    n_eff = max(run_stats.n_queued, 1)

    logger.info("=" * 65, extra={"stage": "extraction"})
    logger.info("STAGE 3 — EXTRACTION REPORT (schema v1.2)", extra={"stage": "extraction"})
    logger.info("=" * 65, extra={"stage": "extraction"})
    logger.info(f"  Queued              : {run_stats.n_queued}",          extra={"stage": "extraction"})
    logger.info(f"  Extracted (fresh)   : {run_stats.n_extracted}",       extra={"stage": "extraction"})
    logger.info(f"  Loaded (cache)      : {run_stats.n_skipped_cached}",  extra={"stage": "extraction"})
    logger.info(f"  Usable total        : {run_stats.n_usable}",          extra={"stage": "extraction"})
    logger.info(
        f"  Skipped (policy)    : {run_stats.n_skipped_policy}  "
        f"({run_stats.n_skipped_policy / n_eff:.1%} of queued) "
        f"[v1.2: primary=detected<{run_stats._min_detected_frames}, "
        f"secondary=missing>{run_stats._max_missing_pct:.0%}]",
        extra={"stage": "extraction"},
    )
    logger.info(
        f"  Skipped (error)     : {run_stats.n_skipped_error}  "
        f"({run_stats.n_skipped_error / n_eff:.1%} of queued)",
        extra={"stage": "extraction"},
    )
    if run_stats.total_frames > 0:
        n_decoded = run_stats.total_frames - run_stats.total_decode_failures
        logger.info(
            f"  Total frames        : {run_stats.total_frames:,}",
            extra={"stage": "extraction"},
        )
        logger.info(
            f"  Decode failures     : {run_stats.total_decode_failures:,}  "
            f"(codec errors, excluded from detection rate denominator)",
            extra={"stage": "extraction"},
        )
        logger.info(
            f"  Detected frames     : {run_stats.total_detected:,} / {n_decoded:,} decoded  "
            f"(detection rate={run_stats.global_detection_rate:.1%})",
            extra={"stage": "extraction"},
        )
        logger.info(
            f"  Global missing rate : {run_stats.global_missing_rate:.2%}  "
            f"({run_stats.total_missing_both:,}/{n_decoded:,} both-hands absent)",
            extra={"stage": "extraction"},
        )
        if run_stats.n_extracted > 0:
            logger.info(
                f"  Per-hand missing    : "
                f"left={run_stats.total_missing_left:,} | "
                f"right={run_stats.total_missing_right:,} | "
                f"pose={run_stats.total_missing_pose:,}",
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

    Thresholds are calibrated for extractor v1.2 (dual-criterion policy):
    - Expected policy skip rate: ~3–6% (not 5–8% as in v1.1)
    - Expected global missing rate: can be higher due to one-handed signs
      retained at up to 95% both-absent rate. Threshold raised to 35%.
    - Error rate threshold unchanged at 5%.

    Parameters
    ----------
    run_stats : _RunStats
        Completed run statistics.
    logger
        Active logger.

    Returns
    -------
    bool
        True if all health checks pass (no threshold exceeded).
    """
    healthy = True
    n_eff   = max(run_stats.n_queued, 1)

    policy_rate = run_stats.n_skipped_policy / n_eff
    if policy_rate > HEALTH_POLICY_SKIP_RATE_WARN:
        logger.warning(
            f"Policy skip rate {policy_rate:.1%} exceeds "
            f"{HEALTH_POLICY_SKIP_RATE_WARN:.0%} threshold. "
            f"With the v1.2 dual-criterion policy (min_detected_frames="
            f"{run_stats._min_detected_frames}), expected skip rate is ~3–6%%. "
            "Possible causes: video files contain unusually short or corrupted "
            "clips with very few detectable frames. "
            "Review skipped clips in preprocessing_summary_latest.json "
            "(per_clip, outcome='skipped_policy') for patterns.",
            extra={"stage": "extraction"},
        )
        healthy = False

    error_rate = run_stats.n_skipped_error / n_eff
    if error_rate > HEALTH_ERROR_RATE_WARN:
        logger.warning(
            f"Error rate {error_rate:.1%} exceeds "
            f"{HEALTH_ERROR_RATE_WARN:.0%} threshold. "
            "Review error records in preprocessing_summary_latest.json "
            "under the 'per_clip' key for details.",
            extra={"stage": "extraction"},
        )
        healthy = False

    if run_stats.global_missing_rate > HEALTH_GLOBAL_MISSING_RATE_WARN:
        logger.warning(
            f"Global missing-landmark rate {run_stats.global_missing_rate:.1%} "
            f"exceeds {HEALTH_GLOBAL_MISSING_RATE_WARN:.0%}. "
            "This is measured over successfully-decoded frames so it "
            "reflects genuine MediaPipe detection quality. "
            "Note: one-handed signs naturally inflate this rate. "
            "Consider reviewing per-sign missing rates in the landmark "
            "inventory CSV and checking video quality for high-miss signs.",
            extra={"stage": "extraction"},
        )
        healthy = False

    # Warn if usable clip count is dangerously low for training
    if run_stats.n_usable < 200:
        logger.warning(
            f"Usable clip count ({run_stats.n_usable}) is below 200. "
            "The ≥70% validation accuracy target requires at least ~200 "
            "training clips. If this is a full extraction run (not sample), "
            "check the per-sign breakdown in preprocessing_summary_latest.json "
            "and consider lowering --min-detected-frames.",
            extra={"stage": "extraction"},
        )
        # This is advisory only — does not set healthy=False.
        # Sample runs will always be below 200; health=True is correct for them.

    return healthy


def _log_output_inventory(landmarks_dir: str, logger) -> None:
    """
    Walk the landmarks directory and log .npy file counts per split and sign.

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

    This function creates ExtractionResult objects from the already-accumulated
    _RunStats records and delegates to ``write_landmark_inventory`` in the
    extractor module. This keeps the CSV format consistent with what
    ``LandmarkExtractor.extract_dataset()`` produces when called directly.

    v1.2 changes from v1.1:
    - ``detected_frames`` is populated from ``rec.get("detected_frames", 0)``
    - Per-hand counts (n_missing_left/right/pose) are populated correctly
      (fixes Notebook 02 Bug 9.1 — previously hardcoded to 0)

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
            "dry_run":        "skipped",
        }
        status = status_map.get(outcome, "error")

        result = ExtractionResult(
            video_id=                   rec.get("video_id",              ""),
            sign_label=                 rec.get("sign_label",            ""),
            split=                      rec.get("split",                 ""),
            output_path=                rec.get("output_path",           ""),
            status=                     status,
            num_frames=                 rec.get("n_frames",              0),
            decode_failure_frames=      rec.get("decode_failure_frames", 0),
            detected_frames=            rec.get("detected_frames",       0),   # v1.2
            missing_left_hand_frames=   rec.get("n_missing_left",        0),   # Bug 9.1 fix
            missing_right_hand_frames=  rec.get("n_missing_right",       0),   # Bug 9.1 fix
            missing_pose_frames=        rec.get("n_missing_pose",        0),   # Bug 9.1 fix
            missing_both_hands_frames=  rec.get("n_missing_both",        0),
            missing_pct=                rec.get("missing_pct",           0.0),
            skip_reason=                rec.get("skip_reason",           ""),
            processing_time_sec=        rec.get("proc_sec",              0.0),
            error_message=              rec.get("error_message",         ""),
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
    Initialise LandmarkExtractor with v1.2 runtime configuration.

    Passes ``min_detected_frames`` (v1.2 primary criterion) and
    ``max_missing_pct`` (v1.2 catastrophe filter) from CLI args.

    The warm-up call (``extractor._init_mediapipe()``) loads the MediaPipe
    model before the main loop to avoid a cold-start timing penalty (~2–5s)
    on the first clip.

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
        f"skip_policy=dual_criterion_v1.2 | "
        f"min_detected_frames={args.min_detected_frames} | "
        f"max_missing_pct={args.max_missing_frame_pct:.0%} | "
        f"min_detection_conf={args.min_detection_confidence} | "
        f"min_tracking_conf={args.min_tracking_confidence}",
        extra={"stage": "extraction"},
    )
    try:
        extractor = LandmarkExtractor(
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
            min_detected_frames=args.min_detected_frames,
            max_missing_pct=args.max_missing_frame_pct,
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

    if args.threshold_diagnostic:
        run_label = "stage3_threshold_diagnostic"
    elif args.sample_only:
        run_label = "stage3_sample"
    else:
        run_label = f"stage3_{args.split}"

    if getattr(args, "dry_run", False):
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
    logger.info("=" * 65, extra={"stage": "extraction"})
    logger.info(
        "WLASL Gesture Recognition — Stage 3: Landmark Extraction",
        extra={"stage": "extraction"},
    )
    logger.info(
        f"Extractor schema version : {EXTRACTOR_SCHEMA_VERSION}",
        extra={"stage": "extraction"},
    )
    logger.info(
        f"Skip policy              : dual_criterion_v1.2 "
        f"(primary: detected_frames >= {args.min_detected_frames}, "
        f"secondary: missing_pct <= {args.max_missing_frame_pct:.0%})",
        extra={"stage": "extraction"},
    )
    logger.info(
        f"Log file                 : {log_file}",
        extra={"stage": "extraction"},
    )
    logger.info("=" * 65, extra={"stage": "extraction"})

    # ----------------------------------------------------------------
    # Threshold diagnostic mode (short-circuit before anything else)
    # ----------------------------------------------------------------
    if args.threshold_diagnostic:
        return run_threshold_diagnostic(args.threshold_diagnostic, logger)

    # ----------------------------------------------------------------
    # Mode summary
    # ----------------------------------------------------------------
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
        min_detected_frames=args.min_detected_frames,
        max_missing_pct=args.max_missing_frame_pct,
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
    # Write landmark_inventory.csv
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
        logger.info("SAMPLE EXTRACTION COMPLETE", extra={"stage": "extraction"})
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
    # Full extraction — next-step guidance
    # ----------------------------------------------------------------
    if not args.sample_only:
        # Warn specifically if sample .npy files exist from the v1.1 run
        # (schema v1.1 sidecars will be rejected by v1.2 cache check;
        # those clips will be automatically re-extracted — this is correct).
        old_sidecars = list(Path(args.landmarks_dir).rglob("*.meta.json"))
        if old_sidecars:
            # Spot-check the first sidecar's schema version
            try:
                with open(old_sidecars[0], encoding="utf-8") as f:
                    sample_meta = json.load(f)
                stored_ver = sample_meta.get("schema_version", "")
                if stored_ver != EXTRACTOR_SCHEMA_VERSION:
                    logger.info(
                        f"Found {len(old_sidecars)} existing .meta.json sidecar(s) "
                        f"with schema version '{stored_ver}' "
                        f"(current: '{EXTRACTOR_SCHEMA_VERSION}'). "
                        "These will be automatically re-extracted by the v1.2 "
                        "extractor's cache validation — no action required.",
                        extra={"stage": "extraction"},
                    )
            except Exception:
                pass

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
        f"  Policy skip rate  : {run_stats.n_skipped_policy / max(run_stats.n_queued, 1):.1%} "
        f"(v1.2 expected: ~3-6%%)",
        extra={"stage": "extraction"},
    )
    logger.info(
        f"  Detection rate    : {run_stats.global_detection_rate:.1%} of decoded frames",
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

    # Return exit code 2 if health checks failed (log warnings were emitted),
    # 0 if all health checks passed.
    return 0 if health_ok else 2


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(main())