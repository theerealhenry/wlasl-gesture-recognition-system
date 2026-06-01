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

IMPORTANT: Arrays store the clip's actual frame count — they are NOT padded to
seq_len=30. Padding and truncation are deferred to ``FeaturePipeline`` (Stage 4)
so the same .npy files can serve all sequence-length ablation experiments
({20, 30, 40, 60} frames) without any re-extraction.

Resumability
------------
Before processing a clip, the extractor checks whether its .npy already exists.
If it does, the clip is skipped unless ``--force`` is passed. This makes the
script safe to restart after crashes or interruptions — no wasted work.

Extraction summary
------------------
At the end of each run, a JSON summary is written (or merged) to
``data/preprocessing_summary.json``. It records per-video stats (frames
extracted, missing-landmark rate, processing time, skip reason) and aggregate
statistics over the run. This file is the audit trail for the extraction phase
and is used by Notebook 02 for the missing-landmark analysis.

Usage
-----
Sample-only run (1 clip per sign, ~35 clips — Stage 2 validation gate):
    python pipelines/run_landmark_extraction.py --sample-only

Full extraction, all splits (30–90 minutes):
    python pipelines/run_landmark_extraction.py --split all

Single split (train only):
    python pipelines/run_landmark_extraction.py --split train

Force re-extraction (overwrite existing .npy files):
    python pipelines/run_landmark_extraction.py --split all --force

Dry run (validate inputs, log what would be processed, write nothing):
    python pipelines/run_landmark_extraction.py --dry-run

Verbose debug logging:
    python pipelines/run_landmark_extraction.py --verbose

Exit codes
----------
0  — Extraction completed successfully (includes partial runs where some
     clips were skipped due to skip policy).
1  — One or more ERROR-severity failures blocked the run
     (e.g. splits CSV not found, no clips to process).
2  — Unexpected exception terminated the pipeline.

Integration with the utils package
-----------------------------------
- configure_logging() / get_logger() — all logging, no print() statements
- set_seeds(42) — called before any data operations for reproducibility
- load_config() — reads base.yaml and data/seq30.yaml for extraction params

The LandmarkExtractor from src.features.extractor is the workhorse of this
script. All MediaPipe logic lives there; this script is pure orchestration.

Design principles
-----------------
- No assert statements in production-critical paths — explicit RuntimeError
- No print() — structured logging throughout
- No MLflow — tracking begins at Stage 5 (run_training.py)
- Single source of truth for which videos to process: data/splits/*.csv
- Extraction is embarrassingly parallel but implemented serially for
  determinism and MediaPipe thread-safety. If speed is critical, pass
  --workers N for multiprocessing (one MediaPipe instance per worker).
"""

from __future__ import annotations

import argparse
import json
import os
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
# Utils — must be imported before feature modules so logging is ready first
# ---------------------------------------------------------------------------
from src.utils.logger import configure_logging, get_logger
from src.utils.reproducibility import set_seeds

# ---------------------------------------------------------------------------
# Feature modules (Stage 3)
# ---------------------------------------------------------------------------
from src.features.extractor import LandmarkExtractor, ExtractionResult, ExtractionStats
from src.features import FEATURE_SIZE


# ---------------------------------------------------------------------------
# Default paths and constants
# ---------------------------------------------------------------------------
_DEFAULT_SPLITS_DIR     = str(_REPO_ROOT / "data" / "splits")
_DEFAULT_LANDMARKS_DIR  = str(_REPO_ROOT / "data" / "landmarks")
_DEFAULT_SUMMARY_PATH   = str(_REPO_ROOT / "data" / "preprocessing_summary.json")
_DEFAULT_LOG_DIR        = str(_REPO_ROOT / "logs")

_VALID_SPLITS           = ("train", "val", "test", "all")
_LOG_INTERVAL           = 50           # clips between progress log lines
_SAMPLE_CLIPS_PER_SIGN  = 1            # how many clips per sign in --sample-only mode
_SEED                   = 42

# MediaPipe / extractor defaults (overridable via CLI to match config.yaml)
_DEFAULT_MAX_MISSING_FRAME_PCT = 0.30  # skip clip if >30% frames have no landmarks
_DEFAULT_STATIC_IMAGE_MODE     = False
_DEFAULT_MIN_DETECTION_CONF    = 0.5
_DEFAULT_MIN_TRACKING_CONF     = 0.5


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
            "Always run --sample-only first to validate the extractor before committing\n"
            "to the full 30–90 minute extraction."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Stage 2 validation gate — 1 clip per sign (~35 clips, ~2-5 minutes)
  python pipelines/run_landmark_extraction.py --sample-only

  # Full extraction, all splits (30-90 minutes)
  python pipelines/run_landmark_extraction.py --split all

  # Training split only
  python pipelines/run_landmark_extraction.py --split train

  # Force re-extraction (overwrite existing .npy files)
  python pipelines/run_landmark_extraction.py --split all --force

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
        help=f"Root output directory for .npy landmark files (default: {_DEFAULT_LANDMARKS_DIR})",
    )
    parser.add_argument(
        "--summary-path",
        default=_DEFAULT_SUMMARY_PATH,
        metavar="PATH",
        help=f"Path for extraction summary JSON (default: {_DEFAULT_SUMMARY_PATH})",
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
            f"(~{35 * _SAMPLE_CLIPS_PER_SIGN} total). "
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
            "Re-extract and overwrite .npy files that already exist. "
            "Without this flag, existing files are skipped (resumable by default)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate all inputs and log the extraction plan without writing "
            "any .npy files or modifying the summary JSON. Useful for verifying "
            "clip counts and output paths before a long run."
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
            "Skip a clip if the fraction of frames with no landmarks detected "
            f"exceeds this threshold (default: {_DEFAULT_MAX_MISSING_FRAME_PCT}). "
            "E.g. 0.30 skips clips where >30%% of frames have zero-filled hands."
        ),
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=_DEFAULT_MIN_DETECTION_CONF,
        metavar="CONF",
        help=f"MediaPipe Holistic min detection confidence (default: {_DEFAULT_MIN_DETECTION_CONF})",
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=_DEFAULT_MIN_TRACKING_CONF,
        metavar="CONF",
        help=f"MediaPipe Holistic min tracking confidence (default: {_DEFAULT_MIN_TRACKING_CONF})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_SEED,
        metavar="N",
        help=f"Random seed (affects sample selection in --sample-only mode) (default: {_SEED})",
    )

    return parser


# ---------------------------------------------------------------------------
# Split loading
# ---------------------------------------------------------------------------

def _load_split_df(splits_dir: str, split_name: str, logger) -> Optional[pd.DataFrame]:
    """
    Load a single split CSV and validate its schema.

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
    seed: int,
    logger,
) -> list[dict[str, Any]]:
    """
    Build the ordered list of clips to process.

    Each entry in the returned list is a dict with keys:
        video_id, sign_label, class_idx, signer_id, split, video_path

    In ``--sample-only`` mode exactly ``_SAMPLE_CLIPS_PER_SIGN`` clips per
    sign per split are selected. The selection is deterministic given ``seed``.

    Parameters
    ----------
    splits_dir : str
        Path to the splits directory.
    split_arg : str
        "train", "val", "test", or "all".
    sample_only : bool
        If True, restrict to one clip per sign.
    seed : int
        Random seed for sample selection.
    logger
        Active logger.

    Returns
    -------
    list[dict]
        Ordered list of clip records to process. Empty list signals an error.
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
            # Deterministic sample: pick _SAMPLE_CLIPS_PER_SIGN clips per sign.
            # Sort by video_id first so the sample is independent of CSV row order.
            sampled_rows = []
            for sign, group in df.groupby("sign_label"):
                group_sorted = group.sort_values("video_id").reset_index(drop=True)
                n_to_take = min(_SAMPLE_CLIPS_PER_SIGN, len(group_sorted))
                sampled_rows.append(group_sorted.iloc[:n_to_take])
            if sampled_rows:
                df = pd.concat(sampled_rows, ignore_index=True)
            logger.info(
                f"Sample mode: selected {len(df)} clips from {split_name} "
                f"({_SAMPLE_CLIPS_PER_SIGN} per sign)",
                extra={"stage": "extraction"},
            )

        for _, row in df.iterrows():
            clips.append({
                "video_id":  str(row["video_id"]),
                "sign_label": str(row["sign_label"]),
                "class_idx":  int(row["class_idx"]),
                "signer_id":  int(row["signer_id"]),
                "split":      split_name,
                "video_path": str(row["video_path"]),
            })

    logger.info(
        f"Total clips queued for extraction: {len(clips)} | "
        f"splits={split_names} | sample_only={sample_only}",
        extra={"stage": "extraction"},
    )
    return clips


# ---------------------------------------------------------------------------
# Output path helper
# ---------------------------------------------------------------------------

def _get_output_path(
    landmarks_dir: str,
    split_name: str,
    sign_label: str,
    video_id: str,
) -> Path:
    """
    Compute the canonical output .npy path for a landmark array.

    Schema: <landmarks_dir>/<split>/<sign_label>/<video_id>.npy

    Parameters
    ----------
    landmarks_dir : str
        Root landmarks directory.
    split_name : str
        "train", "val", or "test".
    sign_label : str
        Human-readable sign name (used as subdirectory).
    video_id : str
        WLASL video identifier.

    Returns
    -------
    Path
        Absolute output path.
    """
    return Path(landmarks_dir) / split_name / sign_label / f"{video_id}.npy"


# ---------------------------------------------------------------------------
# Extraction summary helpers
# ---------------------------------------------------------------------------

class _RunStats:
    """
    Accumulates per-video and aggregate statistics for the current run.

    Designed to be lightweight — just a dict accumulator — so it adds
    negligible overhead per clip.
    """

    def __init__(self, run_id: str, args: argparse.Namespace) -> None:
        self._run_id = run_id
        self._started_utc = datetime.now(timezone.utc).isoformat()
        self._args = args

        # Per-video records (appended as clips complete)
        self._records: list[dict[str, Any]] = []

        # Aggregate counters
        self.n_queued         = 0
        self.n_extracted      = 0
        self.n_skipped_exists = 0   # already existed, not re-processed
        self.n_skipped_policy = 0   # too many missing frames
        self.n_skipped_error  = 0   # video unreadable / MediaPipe crash
        self.n_dry_run        = 0   # dry run — not written
        self.total_frames     = 0
        self.total_missing    = 0
        self.total_proc_sec   = 0.0

        # Per-sign tracking for missing-landmark analysis
        self._sign_missing_frames: dict[str, int]   = defaultdict(int)
        self._sign_total_frames:   dict[str, int]   = defaultdict(int)
        self._sign_skipped:        dict[str, int]   = defaultdict(int)
        self._sign_extracted:      dict[str, int]   = defaultdict(int)

    def record(
        self,
        clip: dict[str, Any],
        outcome: str,                     # "extracted" | "skipped_exists" |
                                          # "skipped_policy" | "skipped_error" | "dry_run"
        result: Optional["ExtractionResult"] = None,
        proc_sec: float = 0.0,
        error_msg: str = "",
        output_path: str = "",
    ) -> None:
        """Append a single clip record and update aggregate counters."""
        sign = clip["sign_label"]
        record: dict[str, Any] = {
            "video_id":    clip["video_id"],
            "sign_label":  sign,
            "class_idx":   clip["class_idx"],
            "signer_id":   clip["signer_id"],
            "split":       clip["split"],
            "video_path":  clip["video_path"],
            "output_path": output_path,
            "outcome":     outcome,
            "proc_sec":    round(proc_sec, 4),
        }

        if result is not None:
            record["n_frames"]         = result.n_frames
            record["n_missing_frames"] = result.n_missing_frames
            record["missing_rate"]     = round(result.missing_rate, 4)
            record["array_shape"]      = list(result.landmarks.shape) if result.landmarks is not None else []
            self.total_frames  += result.n_frames
            self.total_missing += result.n_missing_frames
            self._sign_missing_frames[sign] += result.n_missing_frames
            self._sign_total_frames[sign]   += result.n_frames
        else:
            record["n_frames"]         = 0
            record["n_missing_frames"] = 0
            record["missing_rate"]     = 0.0
            record["array_shape"]      = []

        if error_msg:
            record["error"] = error_msg

        self._records.append(record)
        self.total_proc_sec += proc_sec

        if outcome == "extracted":
            self.n_extracted += 1
            self._sign_extracted[sign] += 1
        elif outcome == "skipped_exists":
            self.n_skipped_exists += 1
        elif outcome == "skipped_policy":
            self.n_skipped_policy += 1
            self._sign_skipped[sign] += 1
        elif outcome == "skipped_error":
            self.n_skipped_error += 1
            self._sign_skipped[sign] += 1
        elif outcome == "dry_run":
            self.n_dry_run += 1

    def to_summary_dict(self) -> dict[str, Any]:
        """Serialise aggregate + per-video stats to a JSON-serialisable dict."""
        elapsed = (
            (datetime.now(timezone.utc) - datetime.fromisoformat(self._started_utc)).total_seconds()
            if self._started_utc
            else 0.0
        )

        global_missing_rate = (
            self.total_missing / self.total_frames
            if self.total_frames > 0 else 0.0
        )

        per_sign: dict[str, Any] = {}
        for sign in sorted(
            set(self._sign_total_frames) | set(self._sign_extracted) | set(self._sign_skipped)
        ):
            total_f = self._sign_total_frames.get(sign, 0)
            miss_f  = self._sign_missing_frames.get(sign, 0)
            per_sign[sign] = {
                "extracted":       self._sign_extracted.get(sign, 0),
                "skipped":         self._sign_skipped.get(sign, 0),
                "total_frames":    total_f,
                "missing_frames":  miss_f,
                "missing_rate":    round(miss_f / total_f, 4) if total_f > 0 else 0.0,
            }

        return {
            "_run_metadata": {
                "run_id":             self._run_id,
                "started_utc":        self._started_utc,
                "completed_utc":      datetime.now(timezone.utc).isoformat(),
                "elapsed_sec":        round(elapsed, 1),
                "split":              getattr(self._args, "split", "all"),
                "sample_only":        getattr(self._args, "sample_only", False),
                "force":              getattr(self._args, "force", False),
                "dry_run":            getattr(self._args, "dry_run", False),
                "max_missing_frame_pct": getattr(
                    self._args, "max_missing_frame_pct", _DEFAULT_MAX_MISSING_FRAME_PCT
                ),
            },
            "aggregate": {
                "n_queued":           self.n_queued,
                "n_extracted":        self.n_extracted,
                "n_skipped_exists":   self.n_skipped_exists,
                "n_skipped_policy":   self.n_skipped_policy,
                "n_skipped_error":    self.n_skipped_error,
                "n_dry_run":          self.n_dry_run,
                "total_frames":       self.total_frames,
                "total_missing":      self.total_missing,
                "global_missing_rate": round(global_missing_rate, 4),
                "total_proc_sec":     round(self.total_proc_sec, 1),
                "mean_proc_sec_per_clip": (
                    round(self.total_proc_sec / max(self.n_extracted, 1), 3)
                ),
            },
            "per_sign":   per_sign,
            "per_clip":   self._records,
        }


def _merge_and_write_summary(
    summary: dict[str, Any],
    summary_path: str,
    logger,
) -> None:
    """
    Merge the current run's summary into the cumulative preprocessing_summary.json.

    If the file already exists (from a previous sample or partial run), the new
    run's data is merged in: ``per_clip`` records are deduplicated by
    ``(video_id, split)``, keeping the most recent outcome. ``per_sign``
    statistics are recomputed from the merged clip records.

    Parameters
    ----------
    summary : dict
        Current run's summary dict from ``_RunStats.to_summary_dict()``.
    summary_path : str
        Path to write the merged JSON.
    logger
        Active logger.
    """
    path = Path(summary_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing_clips: dict[tuple[str, str], dict[str, Any]] = {}

    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                existing = json.load(f)
            for clip_rec in existing.get("per_clip", []):
                key = (clip_rec.get("video_id", ""), clip_rec.get("split", ""))
                existing_clips[key] = clip_rec
            logger.info(
                f"Merging into existing summary | "
                f"existing_clips={len(existing_clips)} | path={path}",
                extra={"stage": "extraction"},
            )
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                f"Could not read existing summary (will overwrite): {exc}",
                extra={"stage": "extraction"},
            )
            existing_clips = {}

    # Merge: new run's records overwrite existing records for the same clip
    for clip_rec in summary.get("per_clip", []):
        key = (clip_rec.get("video_id", ""), clip_rec.get("split", ""))
        existing_clips[key] = clip_rec

    merged_clips = sorted(
        existing_clips.values(),
        key=lambda r: (r.get("split", ""), r.get("sign_label", ""), r.get("video_id", "")),
    )

    # Recompute per-sign stats from merged clip records
    sign_frames:   dict[str, int] = defaultdict(int)
    sign_missing:  dict[str, int] = defaultdict(int)
    sign_extracted: dict[str, int] = defaultdict(int)
    sign_skipped:  dict[str, int] = defaultdict(int)

    for r in merged_clips:
        sign = r.get("sign_label", "")
        outcome = r.get("outcome", "")
        sign_frames[sign]   += r.get("n_frames", 0)
        sign_missing[sign]  += r.get("n_missing_frames", 0)
        if outcome == "extracted":
            sign_extracted[sign] += 1
        elif outcome in ("skipped_policy", "skipped_error"):
            sign_skipped[sign] += 1

    per_sign_merged: dict[str, Any] = {}
    for sign in sorted(set(sign_frames) | set(sign_extracted) | set(sign_skipped)):
        total_f = sign_frames.get(sign, 0)
        miss_f  = sign_missing.get(sign, 0)
        per_sign_merged[sign] = {
            "extracted":       sign_extracted.get(sign, 0),
            "skipped":         sign_skipped.get(sign, 0),
            "total_frames":    total_f,
            "missing_frames":  miss_f,
            "missing_rate":    round(miss_f / total_f, 4) if total_f > 0 else 0.0,
        }

    # Compute merged aggregate from clip records
    n_extracted = sum(1 for r in merged_clips if r.get("outcome") == "extracted")
    n_skipped_exists = sum(1 for r in merged_clips if r.get("outcome") == "skipped_exists")
    n_skipped_policy = sum(1 for r in merged_clips if r.get("outcome") == "skipped_policy")
    n_skipped_error  = sum(1 for r in merged_clips if r.get("outcome") == "skipped_error")
    total_f_all = sum(r.get("n_frames", 0) for r in merged_clips)
    total_m_all = sum(r.get("n_missing_frames", 0) for r in merged_clips)

    merged_summary = {
        "_run_metadata": summary.get("_run_metadata", {}),
        "aggregate": {
            "n_total_clips_in_summary": len(merged_clips),
            "n_extracted":              n_extracted,
            "n_skipped_exists":         n_skipped_exists,
            "n_skipped_policy":         n_skipped_policy,
            "n_skipped_error":          n_skipped_error,
            "total_frames":             total_f_all,
            "total_missing":            total_m_all,
            "global_missing_rate":      round(total_m_all / total_f_all, 4) if total_f_all > 0 else 0.0,
            "total_proc_sec":           round(sum(
                r.get("proc_sec", 0) for r in merged_clips
            ), 1),
        },
        "per_sign": per_sign_merged,
        "per_clip": merged_clips,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(merged_summary, f, indent=2, default=str)

    logger.info(
        f"Extraction summary written: {path} | "
        f"total_clips_in_summary={len(merged_clips)} | "
        f"size={path.stat().st_size / 1024:.1f} KB",
        extra={"stage": "extraction"},
    )


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------

def _validate_video_path(video_path: str, video_id: str, logger) -> Optional[Path]:
    """
    Resolve and validate a video path from the split CSV.

    Handles both relative (repo-root-relative) and absolute paths.

    Parameters
    ----------
    video_path : str
        Path string from the split CSV.
    video_id : str
        WLASL identifier (for logging only).
    logger
        Active logger.

    Returns
    -------
    Path | None
        Resolved absolute path if the file exists, else None.
    """
    if not video_path or video_path == "nan":
        logger.warning(
            f"Empty video_path for video_id={video_id} — clip cannot be processed.",
            extra={"stage": "extraction", "video_id": video_id},
        )
        return None

    p = Path(video_path)
    resolved = p if p.is_absolute() else (_REPO_ROOT / p)

    if not resolved.exists():
        logger.warning(
            f"Video file not found on disk: {resolved} | video_id={video_id}. "
            "Verify the raw_dir path in Stage 1 inventory build.",
            extra={"stage": "extraction", "video_id": video_id},
        )
        return None

    return resolved


def _verify_output_array(
    npy_path: Path,
    video_id: str,
    logger,
) -> bool:
    """
    Spot-check an existing or freshly written .npy file.

    Verifies:
    - File is loadable by numpy
    - Array is 2-dimensional
    - Second dimension is exactly FEATURE_SIZE (225)
    - dtype is float32
    - No NaN or Inf values

    Parameters
    ----------
    npy_path : Path
        Path to the .npy file to verify.
    video_id : str
        WLASL identifier (for logging).
    logger
        Active logger.

    Returns
    -------
    bool
        True if the array passes all checks.
    """
    try:
        arr = np.load(str(npy_path), allow_pickle=False)
    except Exception as exc:
        logger.error(
            f"Cannot load .npy for verification: {npy_path}: {exc}",
            extra={"stage": "extraction", "video_id": video_id},
        )
        return False

    if arr.ndim != 2:
        logger.error(
            f"Shape error: expected 2D array, got shape={arr.shape} | "
            f"video_id={video_id} | path={npy_path}",
            extra={"stage": "extraction", "video_id": video_id},
        )
        return False

    if arr.shape[1] != FEATURE_SIZE:
        logger.error(
            f"Feature size error: expected {FEATURE_SIZE} features/frame, "
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

    if not np.isfinite(arr).all():
        logger.error(
            f"Non-finite values (NaN or Inf) detected in array | "
            f"video_id={video_id} | path={npy_path}",
            extra={"stage": "extraction", "video_id": video_id},
        )
        return False

    return True


# ---------------------------------------------------------------------------
# Dry-run reporting
# ---------------------------------------------------------------------------

def _report_dry_run_plan(
    clips: list[dict[str, Any]],
    landmarks_dir: str,
    force: bool,
    logger,
) -> None:
    """
    Log the extraction plan without doing any actual work.

    Computes how many clips would be extracted vs skipped (already exists)
    and logs a per-sign breakdown.

    Parameters
    ----------
    clips : list[dict]
        Clip records to process.
    landmarks_dir : str
        Root landmarks directory.
    force : bool
        Whether --force was passed (affects skipped-exists count).
    logger
        Active logger.
    """
    would_extract = 0
    would_skip_exists = 0
    by_split: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "exists": 0})

    for clip in clips:
        npy_path = _get_output_path(
            landmarks_dir, clip["split"], clip["sign_label"], clip["video_id"]
        )
        by_split[clip["split"]]["total"] += 1
        if npy_path.exists() and not force:
            would_skip_exists += 1
            by_split[clip["split"]]["exists"] += 1
        else:
            would_extract += 1

    logger.info(
        "[DRY RUN] Extraction plan:",
        extra={"stage": "extraction"},
    )
    logger.info(
        f"  Total queued  : {len(clips)}",
        extra={"stage": "extraction"},
    )
    logger.info(
        f"  Would extract : {would_extract} "
        f"({'all' if not would_skip_exists else 'new/changed'})",
        extra={"stage": "extraction"},
    )
    logger.info(
        f"  Would skip    : {would_skip_exists} (already exist, --force not set)",
        extra={"stage": "extraction"},
    )
    for split_name, counts in sorted(by_split.items()):
        logger.info(
            f"  {split_name:6s}: {counts['total']:4d} clips | "
            f"{counts['exists']:3d} already exist | "
            f"{counts['total'] - counts['exists']:3d} would extract",
            extra={"stage": "extraction"},
        )


# ---------------------------------------------------------------------------
# Core extraction loop
# ---------------------------------------------------------------------------

def _run_extraction_loop(
    clips: list[dict[str, Any]],
    extractor: LandmarkExtractor,
    landmarks_dir: str,
    run_stats: _RunStats,
    force: bool,
    dry_run: bool,
    logger,
) -> None:
    """
    Iterate through clip records, extract landmarks, and write .npy files.

    This is the hot loop. It handles:
    - Resumability: skip existing .npy files unless --force
    - Per-clip timing and stats collection
    - Progress logging every _LOG_INTERVAL clips with ETA
    - Error isolation: a crash on one clip is logged and counted, never fatal
    - Post-write verification: spot-check every written .npy array

    Parameters
    ----------
    clips : list[dict]
        Ordered list of clip records to process.
    extractor : LandmarkExtractor
        Initialised MediaPipe Holistic extractor.
    landmarks_dir : str
        Root landmarks output directory.
    run_stats : _RunStats
        Accumulator for per-clip and aggregate statistics.
    force : bool
        If True, re-extract and overwrite existing .npy files.
    dry_run : bool
        If True, log plan only — do not write files or call MediaPipe.
    logger
        Active logger.
    """
    run_stats.n_queued = len(clips)
    loop_start = time.time()
    n_processed = 0   # clips where actual processing was attempted

    if dry_run:
        _report_dry_run_plan(clips, landmarks_dir, force, logger)
        for clip in clips:
            run_stats.record(clip, outcome="dry_run")
        return

    for i, clip in enumerate(clips):
        video_id   = clip["video_id"]
        sign_label = clip["sign_label"]
        split_name = clip["split"]
        video_path = clip["video_path"]

        # ----------------------------------------------------------------
        # Resumability: check for existing .npy
        # ----------------------------------------------------------------
        npy_path = _get_output_path(landmarks_dir, split_name, sign_label, video_id)

        if npy_path.exists() and not force:
            logger.debug(
                f"Skipping (already exists): {npy_path.name} | "
                f"video_id={video_id} | sign={sign_label}",
                extra={"stage": "extraction", "video_id": video_id},
            )
            run_stats.record(clip, outcome="skipped_exists", output_path=str(npy_path))
            continue

        # ----------------------------------------------------------------
        # Validate video file on disk
        # ----------------------------------------------------------------
        resolved_path = _validate_video_path(video_path, video_id, logger)
        if resolved_path is None:
            run_stats.record(
                clip,
                outcome="skipped_error",
                error_msg="video_file_not_found",
            )
            continue

        # ----------------------------------------------------------------
        # Extract landmarks
        # ----------------------------------------------------------------
        clip_start = time.time()

        try:
            result: Optional[ExtractionResult] = extractor.extract_video(
                str(resolved_path)
            )
        except Exception as exc:
            proc_sec = time.time() - clip_start
            logger.error(
                f"Extraction raised exception | video_id={video_id} | "
                f"sign={sign_label} | {type(exc).__name__}: {exc}",
                extra={"stage": "extraction", "video_id": video_id},
            )
            logger.debug(traceback.format_exc(), extra={"stage": "extraction"})
            run_stats.record(
                clip,
                outcome="skipped_error",
                proc_sec=proc_sec,
                error_msg=f"{type(exc).__name__}: {exc}",
            )
            n_processed += 1
            continue

        proc_sec = time.time() - clip_start

        # ----------------------------------------------------------------
        # Skip policy: too many missing frames
        # ----------------------------------------------------------------
        if result is None:
            # LandmarkExtractor returns None when skip policy triggered
            logger.info(
                f"Skipped (policy: >{extractor.max_missing_frame_pct:.0%} missing) | "
                f"video_id={video_id} | sign={sign_label}",
                extra={"stage": "extraction", "video_id": video_id},
            )
            run_stats.record(
                clip,
                outcome="skipped_policy",
                proc_sec=proc_sec,
                error_msg="missing_frame_pct_exceeded",
            )
            n_processed += 1
            continue

        # ----------------------------------------------------------------
        # Write .npy to disk
        # ----------------------------------------------------------------
        try:
            npy_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(npy_path), result.landmarks)
        except OSError as exc:
            logger.error(
                f"Failed to write .npy: {npy_path} | video_id={video_id} | {exc}",
                extra={"stage": "extraction", "video_id": video_id},
            )
            run_stats.record(
                clip,
                outcome="skipped_error",
                result=result,
                proc_sec=proc_sec,
                error_msg=f"write_failed: {exc}",
            )
            n_processed += 1
            continue

        # ----------------------------------------------------------------
        # Post-write verification — spot-check every written array
        # ----------------------------------------------------------------
        if not _verify_output_array(npy_path, video_id, logger):
            logger.error(
                f"Verification failed — removing corrupt .npy: {npy_path}",
                extra={"stage": "extraction", "video_id": video_id},
            )
            try:
                npy_path.unlink(missing_ok=True)
            except OSError:
                pass
            run_stats.record(
                clip,
                outcome="skipped_error",
                result=result,
                proc_sec=proc_sec,
                error_msg="verification_failed_array_corrupt",
            )
            n_processed += 1
            continue

        # ----------------------------------------------------------------
        # Success
        # ----------------------------------------------------------------
        logger.debug(
            f"Extracted: {video_id} ({sign_label}) | "
            f"frames={result.n_frames} | "
            f"missing={result.missing_rate:.1%} | "
            f"shape={result.landmarks.shape} | "
            f"{proc_sec:.2f}s",
            extra={"stage": "extraction", "video_id": video_id},
        )

        run_stats.record(
            clip,
            outcome="extracted",
            result=result,
            proc_sec=proc_sec,
            output_path=str(npy_path),
        )
        n_processed += 1

        # ----------------------------------------------------------------
        # Progress logging every _LOG_INTERVAL processed clips
        # ----------------------------------------------------------------
        if n_processed % _LOG_INTERVAL == 0:
            elapsed = time.time() - loop_start
            total_to_process = sum(
                1 for c in clips
                if force or not _get_output_path(
                    landmarks_dir, c["split"], c["sign_label"], c["video_id"]
                ).exists()
            )
            # Approximate ETA based on clips processed so far
            rate = n_processed / elapsed if elapsed > 0 else 0.0
            remaining = max(total_to_process - n_processed, 0)
            eta_sec = remaining / rate if rate > 0 else 0.0
            eta_min = eta_sec / 60

            logger.info(
                f"Progress | {i + 1}/{len(clips)} queued | "
                f"{run_stats.n_extracted} extracted | "
                f"{run_stats.n_skipped_policy} skipped (policy) | "
                f"{run_stats.n_skipped_error} errors | "
                f"rate={rate:.1f} clips/s | "
                f"ETA={eta_min:.1f}min",
                extra={"stage": "extraction"},
            )

    # ----------------------------------------------------------------
    # Final progress line (catches runs where len(clips) < _LOG_INTERVAL)
    # ----------------------------------------------------------------
    elapsed = time.time() - loop_start
    logger.info(
        f"Extraction loop complete | "
        f"elapsed={elapsed:.1f}s | "
        f"extracted={run_stats.n_extracted} | "
        f"skipped_exists={run_stats.n_skipped_exists} | "
        f"skipped_policy={run_stats.n_skipped_policy} | "
        f"skipped_error={run_stats.n_skipped_error}",
        extra={"stage": "extraction"},
    )


# ---------------------------------------------------------------------------
# Post-run reporting
# ---------------------------------------------------------------------------

def _log_extraction_report(run_stats: _RunStats, logger) -> None:
    """
    Emit a structured, human-readable extraction report at INFO level.

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

    logger.info("=" * 65, extra={"stage": "extraction"})
    logger.info("STAGE 3 — EXTRACTION REPORT", extra={"stage": "extraction"})
    logger.info("=" * 65, extra={"stage": "extraction"})
    logger.info(
        f"  Queued            : {run_stats.n_queued}",
        extra={"stage": "extraction"},
    )
    logger.info(
        f"  Extracted         : {run_stats.n_extracted}",
        extra={"stage": "extraction"},
    )
    logger.info(
        f"  Skipped (exists)  : {run_stats.n_skipped_exists}",
        extra={"stage": "extraction"},
    )
    logger.info(
        f"  Skipped (policy)  : {run_stats.n_skipped_policy}  "
        f"(>{_DEFAULT_MAX_MISSING_FRAME_PCT:.0%} missing frames)",
        extra={"stage": "extraction"},
    )
    logger.info(
        f"  Skipped (error)   : {run_stats.n_skipped_error}",
        extra={"stage": "extraction"},
    )
    if run_stats.total_frames > 0:
        logger.info(
            f"  Total frames      : {total_frames:,}",
            extra={"stage": "extraction"},
        )
        logger.info(
            f"  Missing rate      : {global_miss:.2%} "
            f"({total_missing:,}/{total_frames:,} frames zero-filled)",
            extra={"stage": "extraction"},
        )
    logger.info(
        f"  Processing time   : {run_stats.total_proc_sec:.1f}s",
        extra={"stage": "extraction"},
    )

    # Expected skip rate from handoff doc: 5–8% of clips
    skip_policy_rate = (
        run_stats.n_skipped_policy / max(run_stats.n_queued, 1)
    )
    if run_stats.n_skipped_policy > 0:
        logger.info(
            f"  Policy skip rate  : {skip_policy_rate:.1%} "
            f"(expected 5–8% per project specification)",
            extra={"stage": "extraction"},
        )

    if run_stats.n_skipped_error > 0:
        logger.warning(
            f"  {run_stats.n_skipped_error} clip(s) failed due to errors. "
            "Check logs above for details. These clips will not be available "
            "for training. Review data/preprocessing_summary.json for the full list.",
            extra={"stage": "extraction"},
        )

    logger.info("=" * 65, extra={"stage": "extraction"})


def _validate_extraction_health(run_stats: _RunStats, logger) -> bool:
    """
    Check overall extraction health and emit actionable warnings.

    Returns False if any critical threshold is exceeded — not a hard failure,
    but the caller should log a prominent warning.

    Thresholds (based on project specification in handoff document):
    - Policy skip rate > 10%: may indicate MediaPipe configuration issue
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

    # Skip rate
    skip_rate = run_stats.n_skipped_policy / max(run_stats.n_queued, 1)
    if skip_rate > 0.10:
        logger.warning(
            f"Policy skip rate is {skip_rate:.1%} — above 10% threshold. "
            "Consider lowering --max-missing-frame-pct or checking MediaPipe "
            "confidence settings. Expected: 5–8% per project specification.",
            extra={"stage": "extraction"},
        )
        healthy = False

    # Error rate
    error_rate = run_stats.n_skipped_error / max(run_stats.n_queued, 1)
    if error_rate > 0.05:
        logger.warning(
            f"Error skip rate is {error_rate:.1%} — above 5% threshold. "
            "Review skipped_error records in preprocessing_summary.json.",
            extra={"stage": "extraction"},
        )
        healthy = False

    # Global missing frame rate
    if run_stats.total_frames > 0:
        global_miss = run_stats.total_missing / run_stats.total_frames
        if global_miss > 0.15:
            logger.warning(
                f"Global missing-landmark rate is {global_miss:.1%} — above 15%. "
                "Higher than expected for WLASL dataset. "
                "Consider increasing --min-detection-confidence or reviewing "
                "video quality for affected signs.",
                extra={"stage": "extraction"},
            )
            healthy = False

    return healthy


# ---------------------------------------------------------------------------
# Landmark directory inventory
# ---------------------------------------------------------------------------

def _log_output_inventory(landmarks_dir: str, logger) -> None:
    """
    Walk the landmarks directory tree and log file counts per split and sign.

    Called at the end of the run to confirm the output structure is correct.

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
            f"Landmarks directory does not exist: {root}",
            extra={"stage": "extraction"},
        )
        return

    total_files = 0
    split_counts: dict[str, int] = {}
    sign_counts: dict[str, int] = defaultdict(int)

    for split_dir in sorted(root.iterdir()):
        if not split_dir.is_dir():
            continue
        split_name = split_dir.name
        n_in_split = 0
        for sign_dir in sorted(split_dir.iterdir()):
            if not sign_dir.is_dir():
                continue
            n_files = len(list(sign_dir.glob("*.npy")))
            n_in_split += n_files
            sign_counts[sign_dir.name] += n_files
        split_counts[split_name] = n_in_split
        total_files += n_in_split

    logger.info(
        f"Landmark directory inventory | total_npy_files={total_files}",
        extra={"stage": "extraction"},
    )
    for split_name, count in sorted(split_counts.items()):
        logger.info(
            f"  {split_name:6s}: {count:4d} files",
            extra={"stage": "extraction"},
        )

    signs_with_files = {s: c for s, c in sign_counts.items() if c > 0}
    logger.info(
        f"  Signs with ≥1 .npy file: {len(signs_with_files)}/35",
        extra={"stage": "extraction"},
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """
    Execute Stage 3 landmark extraction pipeline.

    Returns
    -------
    int
        Exit code: 0=success, 1=input error, 2=unexpected failure.
    """
    parser = _build_parser()
    args = parser.parse_args()

    # ----------------------------------------------------------------
    # Logging — configure before ANY other operation
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
        "WLASL Gesture Recognition — Stage 3: Landmark Extraction",
        extra={"stage": "extraction"},
    )
    logger.info(f"Log file: {log_file}", extra={"stage": "extraction"})
    logger.info(
        f"Mode: {'SAMPLE (' + str(_SAMPLE_CLIPS_PER_SIGN) + ' clip/sign)' if args.sample_only else args.split.upper()} | "
        f"force={args.force} | dry_run={args.dry_run}",
        extra={"stage": "extraction"},
    )
    logger.info(
        f"MediaPipe config | "
        f"max_missing_frame_pct={args.max_missing_frame_pct:.0%} | "
        f"min_detection_confidence={args.min_detection_confidence} | "
        f"min_tracking_confidence={args.min_tracking_confidence}",
        extra={"stage": "extraction"},
    )

    # ----------------------------------------------------------------
    # Reproducibility
    # ----------------------------------------------------------------
    set_seeds(args.seed)

    # ----------------------------------------------------------------
    # Validate argument constraints
    # ----------------------------------------------------------------
    if not (0.0 < args.max_missing_frame_pct <= 1.0):
        logger.error(
            f"--max-missing-frame-pct must be in (0, 1]. Got: {args.max_missing_frame_pct}",
            extra={"stage": "extraction"},
        )
        return 1

    if not (0.0 < args.min_detection_confidence <= 1.0):
        logger.error(
            f"--min-detection-confidence must be in (0, 1]. Got: {args.min_detection_confidence}",
            extra={"stage": "extraction"},
        )
        return 1

    if not (0.0 < args.min_tracking_confidence <= 1.0):
        logger.error(
            f"--min-tracking-confidence must be in (0, 1]. Got: {args.min_tracking_confidence}",
            extra={"stage": "extraction"},
        )
        return 1

    # ----------------------------------------------------------------
    # Collect clips to process
    # ----------------------------------------------------------------
    split_arg = "all" if args.sample_only else args.split
    clips = _collect_clips(
        splits_dir=args.splits_dir,
        split_arg=split_arg,
        sample_only=args.sample_only,
        seed=args.seed,
        logger=logger,
    )

    if not clips:
        logger.error(
            "No clips to process. Verify split CSVs exist in "
            f"{args.splits_dir} and contain video_path entries.",
            extra={"stage": "extraction"},
        )
        return 1

    # ----------------------------------------------------------------
    # Run statistics accumulator
    # ----------------------------------------------------------------
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_stats = _RunStats(run_id=run_id, args=args)

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
            extractor=None,  # type: ignore[arg-type]
            landmarks_dir=args.landmarks_dir,
            run_stats=run_stats,
            force=args.force,
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
    # Initialise MediaPipe Holistic extractor
    # ----------------------------------------------------------------
    logger.info(
        "Initialising MediaPipe Holistic extractor...",
        extra={"stage": "extraction"},
    )

    try:
        extractor = LandmarkExtractor(
            max_missing_frame_pct=args.max_missing_frame_pct,
            static_image_mode=_DEFAULT_STATIC_IMAGE_MODE,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
        )
    except Exception as exc:
        logger.error(
            f"Failed to initialise LandmarkExtractor: {type(exc).__name__}: {exc}. "
            "Ensure mediapipe==0.10.14 is installed in the active environment.",
            extra={"stage": "extraction"},
        )
        logger.debug(traceback.format_exc(), extra={"stage": "extraction"})
        return 2

    logger.info(
        f"LandmarkExtractor ready | "
        f"feature_size={FEATURE_SIZE} values/frame | "
        f"max_missing={extractor.max_missing_frame_pct:.0%}",
        extra={"stage": "extraction"},
    )

    # ----------------------------------------------------------------
    # Confirm output directory layout
    # ----------------------------------------------------------------
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
            dry_run=False,
            logger=logger,
        )
    except KeyboardInterrupt:
        logger.warning(
            f"Extraction interrupted by user (KeyboardInterrupt). "
            f"{run_stats.n_extracted} clips extracted before interruption. "
            "Re-run to continue from where you left off (resumable by default).",
            extra={"stage": "extraction"},
        )
        # Still write partial summary so the user can review progress
        _write_partial_summary(run_stats, args.summary_path, logger)
        return 2

    except Exception as exc:
        logger.error(
            f"Unexpected exception in extraction loop: {type(exc).__name__}: {exc}",
            extra={"stage": "extraction"},
        )
        logger.debug(traceback.format_exc(), extra={"stage": "extraction"})
        _write_partial_summary(run_stats, args.summary_path, logger)
        return 2

    # ----------------------------------------------------------------
    # Post-run validation and reporting
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
    # Write / merge extraction summary JSON
    # ----------------------------------------------------------------
    try:
        summary_dict = run_stats.to_summary_dict()
        _merge_and_write_summary(summary_dict, args.summary_path, logger)
    except Exception as exc:
        logger.error(
            f"Failed to write extraction summary: {exc}",
            extra={"stage": "extraction"},
        )
        # Non-fatal — the .npy files are the critical output

    # ----------------------------------------------------------------
    # Landmark directory inventory (confirms output structure)
    # ----------------------------------------------------------------
    _log_output_inventory(args.landmarks_dir, logger)

    # ----------------------------------------------------------------
    # Manual verification instructions (sample-only mode)
    # ----------------------------------------------------------------
    if args.sample_only and run_stats.n_extracted > 0:
        logger.info(
            "SAMPLE EXTRACTION COMPLETE — Manual verification recommended:",
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
            "  assert arr.ndim == 2 and arr.shape[1] == 225",
            extra={"stage": "extraction"},
        )
        logger.info(
            "  print(arr.shape, arr.min(), arr.max())",
            extra={"stage": "extraction"},
        )
        logger.info(
            "If shapes and value ranges look correct, proceed to "
            "notebooks/02_landmark_inspection.ipynb.",
            extra={"stage": "extraction"},
        )
        logger.info(
            "Then run full extraction: "
            "python pipelines/run_landmark_extraction.py --split all",
            extra={"stage": "extraction"},
        )

    # ----------------------------------------------------------------
    # Summary footer
    # ----------------------------------------------------------------
    logger.info("=" * 65, extra={"stage": "extraction"})
    logger.info("STAGE 3 COMPLETE", extra={"stage": "extraction"})
    logger.info(
        f"  Extracted     : {run_stats.n_extracted} clips",
        extra={"stage": "extraction"},
    )
    logger.info(
        f"  Total elapsed : {total_elapsed:.1f}s "
        f"({total_elapsed / 60:.1f} minutes)",
        extra={"stage": "extraction"},
    )
    logger.info(
        f"  Summary       : {args.summary_path}",
        extra={"stage": "extraction"},
    )
    logger.info(
        f"  Landmarks dir : {args.landmarks_dir}",
        extra={"stage": "extraction"},
    )

    if not args.sample_only:
        logger.info(
            "Next step: Stage 4 — build src/features/pipeline.py (FeaturePipeline) "
            "and run notebooks/03_feature_engineering_experiments.ipynb.",
            extra={"stage": "extraction"},
        )
    logger.info("=" * 65, extra={"stage": "extraction"})

    return 0


def _write_partial_summary(
    run_stats: _RunStats,
    summary_path: str,
    logger,
) -> None:
    """
    Write a partial extraction summary after an interruption or error.

    Records what was completed before the interruption so the user can
    inspect the partial state and resume the run cleanly.

    Parameters
    ----------
    run_stats : _RunStats
        Partially completed run statistics.
    summary_path : str
        Path to write the JSON.
    logger
        Active logger.
    """
    try:
        summary_dict = run_stats.to_summary_dict()
        # Mark as partial in the metadata
        if "_run_metadata" in summary_dict:
            summary_dict["_run_metadata"]["status"] = "PARTIAL_INTERRUPTED"
        _merge_and_write_summary(summary_dict, summary_path, logger)
        logger.info(
            f"Partial extraction summary written: {summary_path}",
            extra={"stage": "extraction"},
        )
    except Exception as exc:
        logger.error(
            f"Could not write partial summary: {exc}",
            extra={"stage": "extraction"},
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(main())