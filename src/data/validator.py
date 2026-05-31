"""
src/data/validator.py
======================
Data integrity validation for the WLASL gesture recognition pipeline.

Overview
--------
The validator runs 8 integrity checks against the clip inventory produced by
``WLASLResolver``. It produces a machine-readable validation report written to
``data/data_validation_report.json``. A ``pipeline_can_proceed`` flag is set
in the report — downstream pipeline scripts must check this before proceeding.

Check severity model
--------------------
ERROR   — structural problems that make the pipeline logically impossible
          to run correctly (e.g. missing signs, duplicate video IDs, missing
          signer IDs). Sets pipeline_can_proceed=False.

WARNING — data quality issues that reduce model quality but do not make
          training impossible (e.g. signs with fewer than 20 clips, videos
          with unusual frame counts, class imbalance). Pipeline proceeds.

INFO    — informational checks that always pass (e.g. total dataset size).
          Never blocks the pipeline.

The 8 checks
------------
1.  all_signs_present          — all 35 selected signs have at least 1 clip
2.  minimum_clips_per_sign     — every sign has ≥ min_clips_per_sign found clips
3.  video_readability          — every found clip is readable by OpenCV
4.  frame_count_range          — all clips have frame counts in [min_frames, max_frames]
5.  no_duplicate_video_ids     — video IDs are globally unique
6.  signer_ids_complete        — signer_id != -1 for all clips (required for split)
7.  dataset_size               — total size computed (always passes; informational)
8.  class_imbalance            — max/min clip-count ratio across signs ≤ threshold

Report schema
-------------
Written to ``data/data_validation_report.json``. The top-level structure:

    {
        "metadata": {
            "generated_utc": "...",
            "inventory_path": "...",
            "total_checks": 8,
            "checks_passed": N,
            "checks_warned": N,
            "checks_failed": N,
            "pipeline_can_proceed": true | false,
            "blocking_checks": [...]
        },
        "checks": {
            "<check_name>": {
                "passed": bool,
                "severity": "error" | "warning" | "info",
                "message": "...",
                "details": { ... }
            }
        },
        "per_sign_stats": {
            "<sign_name>": {
                "clips_found": int,
                "unique_signers": int,
                "mean_duration_sec": float,
                "mean_frame_count": float,
                "min_frame_count": int,
                "max_frame_count": int,
                "total_size_mb": float
            }
        },
        "dataset_totals": {
            "total_clips": int,
            "total_clips_found": int,
            "total_size_mb": float,
            "total_unique_signers": int,
            "mean_clips_per_sign": float,
            "min_clips_per_sign": int,
            "max_clips_per_sign": int
        }
    }

Usage
-----
    from src.data.validator import DataValidator

    validator = DataValidator(
        inventory_df=inventory_df,
        raw_dir="data/raw",
        min_clips_per_sign=20,
        min_frames=10,
        max_frames=300,
        imbalance_threshold=3.0,
    )
    report = validator.run_all_checks()
    report.save("data/data_validation_report.json")

    if not report.pipeline_can_proceed:
        raise RuntimeError("Pipeline blocked. See data/data_validation_report.json")
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

try:
    from tqdm import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Repository root
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Severity constants
# ---------------------------------------------------------------------------
SEVERITY_ERROR   = "error"
SEVERITY_WARNING = "warning"
SEVERITY_INFO    = "info"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    """
    Result of a single validation check.

    Attributes
    ----------
    check_name : str
        Snake-case identifier matching the key in the report JSON.
    passed : bool
        True if the check succeeded (no issue detected).
    severity : str
        One of "error", "warning", "info". Determines whether failure
        blocks the pipeline.
    message : str
        Human-readable description of the outcome.
    details : dict
        Structured detail data (sign-level or clip-level breakdown).
    elapsed_sec : float
        Wall-clock time the check took. Useful for profiling on large datasets.
    """
    check_name: str
    passed: bool
    severity: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    elapsed_sec: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("check_name")   # check_name is the dict key, not a field value
        return d


@dataclass
class ValidationReport:
    """
    Aggregated result of all validation checks.

    This object is both the runtime result and the serialisable report.
    Call ``save()`` to write ``data_validation_report.json``.

    Attributes
    ----------
    pipeline_can_proceed : bool
        True if no ERROR-severity check failed. Downstream scripts must
        check this before proceeding.
    checks : dict[str, CheckResult]
        Individual check results, keyed by check_name.
    per_sign_stats : dict
        Per-sign clip and quality statistics.
    dataset_totals : dict
        Aggregate dataset-level statistics.
    inventory_path : str
        Path to the inventory used to produce this report.
    generated_utc : str
        ISO 8601 timestamp of report generation.
    """
    pipeline_can_proceed: bool
    checks: dict[str, CheckResult]
    per_sign_stats: dict[str, Any]
    dataset_totals: dict[str, Any]
    inventory_path: str = ""
    generated_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def n_passed(self) -> int:
        return sum(1 for r in self.checks.values() if r.passed)

    @property
    def n_warned(self) -> int:
        return sum(
            1 for r in self.checks.values()
            if not r.passed and r.severity == SEVERITY_WARNING
        )

    @property
    def n_failed(self) -> int:
        return sum(
            1 for r in self.checks.values()
            if not r.passed and r.severity == SEVERITY_ERROR
        )

    @property
    def blocking_checks(self) -> list[str]:
        return [
            name for name, r in self.checks.items()
            if not r.passed and r.severity == SEVERITY_ERROR
        ]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": {
                "generated_utc": self.generated_utc,
                "inventory_path": self.inventory_path,
                "total_checks": len(self.checks),
                "checks_passed": self.n_passed,
                "checks_warned": self.n_warned,
                "checks_failed": self.n_failed,
                "pipeline_can_proceed": self.pipeline_can_proceed,
                "blocking_checks": self.blocking_checks,
            },
            "checks": {
                name: result.to_dict()
                for name, result in self.checks.items()
            },
            "per_sign_stats": self.per_sign_stats,
            "dataset_totals": self.dataset_totals,
        }

    def save(self, output_path: str | Path) -> Path:
        """
        Write the report to a JSON file.

        Parameters
        ----------
        output_path : str | Path
            Destination. Parent directories are created if needed.

        Returns
        -------
        Path
            Resolved path to the written file.
        """
        path = Path(output_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

        logger.info(
            f"Validation report saved: {path} | "
            f"passed={self.n_passed} | "
            f"warned={self.n_warned} | "
            f"failed={self.n_failed} | "
            f"can_proceed={self.pipeline_can_proceed}",
            extra={"stage": "validation"},
        )
        return path

    def print_summary(self) -> None:
        """Print a human-readable summary to the logger at INFO level."""
        status = "✓ PROCEED" if self.pipeline_can_proceed else "✗ BLOCKED"
        logger.info(
            f"Validation summary [{status}] | "
            f"passed={self.n_passed} | "
            f"warned={self.n_warned} | "
            f"failed={self.n_failed}",
            extra={"stage": "validation"},
        )
        for name, result in self.checks.items():
            icon = "✓" if result.passed else ("⚠" if result.severity == SEVERITY_WARNING else "✗")
            logger.info(
                f"  {icon} [{result.severity.upper():7s}] {name}: {result.message}",
                extra={"stage": "validation"},
            )
        if not self.pipeline_can_proceed:
            logger.error(
                f"Pipeline BLOCKED by {self.n_failed} error(s): {self.blocking_checks}. "
                "Resolve these issues before proceeding to Stage 2.",
                extra={"stage": "validation"},
            )


# ---------------------------------------------------------------------------
# DataValidator
# ---------------------------------------------------------------------------

class DataValidator:
    """
    Runs 8 integrity checks on the WLASL clip inventory.

    Parameters
    ----------
    inventory_df : pd.DataFrame
        The inventory DataFrame produced by WLASLResolver.build_inventory().
    raw_dir : str | Path
        Root directory for raw video files. Used for direct file-level checks.
    min_clips_per_sign : int
        Minimum acceptable found-clip count per sign. Default 20.
    min_frames : int
        Minimum frame count for a valid clip. Default 10.
    max_frames : int
        Maximum frame count for a valid clip. Default 300.
    imbalance_threshold : float
        Maximum acceptable ratio of max/min clips across signs. Default 3.0.
    num_classes : int
        Expected number of distinct sign classes. Default 35.
    """

    def __init__(
        self,
        inventory_df: pd.DataFrame,
        raw_dir: str | Path,
        min_clips_per_sign: int = 20,
        min_frames: int = 10,
        max_frames: int = 300,
        imbalance_threshold: float = 3.0,
        num_classes: int = 35,
    ) -> None:
        self._df = inventory_df.copy()
        self._raw_dir = Path(raw_dir).resolve()
        self._min_clips = min_clips_per_sign
        self._min_frames = min_frames
        self._max_frames = max_frames
        self._imbalance_threshold = imbalance_threshold
        self._num_classes = num_classes

        # Only check found clips for video-level checks
        self._found_df = self._df[self._df["found"] == True].copy()

        logger.info(
            f"DataValidator initialised | "
            f"total_clips={len(self._df)} | "
            f"found={len(self._found_df)} | "
            f"missing={len(self._df) - len(self._found_df)} | "
            f"num_classes={num_classes}",
            extra={"stage": "validation"},
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run_all_checks(self, inventory_path: str = "") -> ValidationReport:
        """
        Execute all 8 validation checks in sequence.

        Checks are designed to fail gracefully — one failing check does not
        prevent subsequent checks from running. All results are collected
        before the final report is assembled.

        Parameters
        ----------
        inventory_path : str
            Path to the inventory file (recorded in the report metadata).

        Returns
        -------
        ValidationReport
            Aggregated result with pipeline_can_proceed flag.
        """
        logger.info(
            "Starting validation suite | 8 checks",
            extra={"stage": "validation"},
        )

        # Execute all checks and collect results
        check_methods = [
            ("all_signs_present",      self._check_all_signs_present),
            ("minimum_clips_per_sign", self._check_minimum_clips_per_sign),
            ("video_readability",      self._check_video_readability),
            ("frame_count_range",      self._check_frame_count_range),
            ("no_duplicate_video_ids", self._check_no_duplicate_video_ids),
            ("signer_ids_complete",    self._check_signer_ids_complete),
            ("dataset_size",           self._check_dataset_size),
            ("class_imbalance",        self._check_class_imbalance),
        ]

        checks: dict[str, CheckResult] = {}
        for check_name, check_fn in check_methods:
            logger.info(
                f"Running check: {check_name}",
                extra={"stage": "validation"},
            )
            t0 = time.time()
            try:
                result = check_fn()
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    f"Check '{check_name}' raised an unexpected exception: {exc}",
                    extra={"stage": "validation"},
                )
                result = CheckResult(
                    check_name=check_name,
                    passed=False,
                    severity=SEVERITY_ERROR,
                    message=f"Check raised exception: {exc}",
                    details={"exception": str(exc)},
                )
            result.elapsed_sec = round(time.time() - t0, 3)
            checks[check_name] = result

            status = "PASSED" if result.passed else f"FAILED [{result.severity.upper()}]"
            logger.info(
                f"  {check_name}: {status} — {result.message} "
                f"({result.elapsed_sec:.2f}s)",
                extra={"stage": "validation"},
            )

        # Determine pipeline_can_proceed
        pipeline_can_proceed = all(
            r.passed or r.severity != SEVERITY_ERROR
            for r in checks.values()
        )

        # Compute aggregate statistics for the report
        per_sign_stats = self._compute_per_sign_stats()
        dataset_totals = self._compute_dataset_totals()

        report = ValidationReport(
            pipeline_can_proceed=pipeline_can_proceed,
            checks=checks,
            per_sign_stats=per_sign_stats,
            dataset_totals=dataset_totals,
            inventory_path=inventory_path,
        )
        report.print_summary()
        return report

    # ------------------------------------------------------------------
    # Check 1 — All 35 signs present
    # ------------------------------------------------------------------

    def _check_all_signs_present(self) -> CheckResult:
        """
        Verify that all expected sign classes have at least one found clip.

        A sign missing entirely from the found inventory means we have no
        training data for that class — the model cannot learn it. This is a
        hard blocker.
        """
        if self._found_df.empty:
            return CheckResult(
                check_name="all_signs_present",
                passed=False,
                severity=SEVERITY_ERROR,
                message="No clips found on disk at all — inventory is empty.",
                details={"found_signs": [], "missing_signs": []},
            )

        found_signs = set(self._found_df["sign_label"].unique())
        # We don't have label map access here, so infer expected from inventory
        expected_signs = set(self._df["sign_label"].unique())
        missing_from_disk = expected_signs - found_signs

        if missing_from_disk:
            return CheckResult(
                check_name="all_signs_present",
                passed=False,
                severity=SEVERITY_ERROR,
                message=(
                    f"{len(missing_from_disk)} sign(s) have NO clips found on disk: "
                    f"{sorted(missing_from_disk)}"
                ),
                details={
                    "found_signs": sorted(found_signs),
                    "missing_signs": sorted(missing_from_disk),
                    "expected_count": len(expected_signs),
                    "found_count": len(found_signs),
                },
            )

        return CheckResult(
            check_name="all_signs_present",
            passed=True,
            severity=SEVERITY_INFO,
            message=f"All {len(found_signs)} signs present with ≥1 clip on disk.",
            details={
                "found_signs": sorted(found_signs),
                "found_count": len(found_signs),
            },
        )

    # ------------------------------------------------------------------
    # Check 2 — Minimum clips per sign
    # ------------------------------------------------------------------

    def _check_minimum_clips_per_sign(self) -> CheckResult:
        """
        Verify that every sign meets the minimum clip count threshold.

        Signs below the threshold will be underrepresented in training,
        increasing per-class variance and reducing reliability. This is
        a warning, not a blocker — we can proceed but should note the
        limitation.
        """
        clips_per_sign = (
            self._found_df.groupby("sign_label").size().to_dict()
        )
        below_threshold = {
            sign: count
            for sign, count in clips_per_sign.items()
            if count < self._min_clips
        }

        if below_threshold:
            return CheckResult(
                check_name="minimum_clips_per_sign",
                passed=False,
                severity=SEVERITY_WARNING,
                message=(
                    f"{len(below_threshold)} sign(s) have fewer than "
                    f"{self._min_clips} clips. Model performance for these "
                    f"signs may be lower."
                ),
                details={
                    "threshold": self._min_clips,
                    "below_threshold": dict(sorted(below_threshold.items(), key=lambda x: x[1])),
                    "clips_per_sign": dict(sorted(clips_per_sign.items())),
                },
            )

        return CheckResult(
            check_name="minimum_clips_per_sign",
            passed=True,
            severity=SEVERITY_INFO,
            message=(
                f"All signs meet the minimum clip threshold of {self._min_clips}. "
                f"Range: {min(clips_per_sign.values())}–{max(clips_per_sign.values())} clips."
            ),
            details={
                "threshold": self._min_clips,
                "clips_per_sign": dict(sorted(clips_per_sign.items())),
                "min_clips": min(clips_per_sign.values()),
                "max_clips": max(clips_per_sign.values()),
            },
        )

    # ------------------------------------------------------------------
    # Check 3 — Video readability
    # ------------------------------------------------------------------

    def _check_video_readability(self) -> CheckResult:
        """
        Verify that every found clip is readable by OpenCV.

        Corrupt or truncated video files cannot be processed by MediaPipe.
        This check attempts to open each file and read its metadata (not
        actual frames) via cv2.VideoCapture.

        The check also populates ``frame_count`` and ``duration_sec``
        columns in the internal found DataFrame for use by Check 4 and
        the per-sign statistics.
        """
        if not _CV2_AVAILABLE:
            logger.warning(
                "OpenCV not available. Skipping video readability check. "
                "Install: pip install opencv-python",
                extra={"stage": "validation"},
            )
            return CheckResult(
                check_name="video_readability",
                passed=True,
                severity=SEVERITY_WARNING,
                message="OpenCV not installed — readability check skipped.",
                details={"opencv_available": False},
            )

        corrupt: list[dict[str, str]] = []
        frame_counts: list[int] = []
        durations: list[float] = []

        video_paths = self._found_df["video_path"].tolist()
        iterator = (
            tqdm(video_paths, desc="Checking readability", unit="clip")
            if _TQDM_AVAILABLE else video_paths
        )

        for rel_path in iterator:
            abs_path = str(_REPO_ROOT / rel_path) if rel_path else ""
            if not abs_path or not Path(abs_path).exists():
                corrupt.append({
                    "video_path": rel_path,
                    "reason": "file_not_found_at_path",
                })
                frame_counts.append(0)
                durations.append(0.0)
                continue

            readable, n_frames, duration = _check_video_readable(abs_path)

            if not readable:
                corrupt.append({
                    "video_path": rel_path,
                    "reason": "opencv_cannot_open",
                })
                frame_counts.append(0)
                durations.append(0.0)
            else:
                frame_counts.append(n_frames)
                durations.append(duration)

        # Attach frame_count and duration_sec to the found DataFrame
        # so Check 4 and statistics can use them without re-reading files
        self._found_df = self._found_df.copy()
        self._found_df["frame_count"] = frame_counts
        self._found_df["duration_sec"] = durations

        if corrupt:
            return CheckResult(
                check_name="video_readability",
                passed=False,
                severity=SEVERITY_ERROR,
                message=(
                    f"{len(corrupt)} clip(s) could not be opened by OpenCV. "
                    "These will cause failures in the preprocessing pipeline."
                ),
                details={
                    "total_checked": len(video_paths),
                    "corrupt_count": len(corrupt),
                    "corrupt_clips": corrupt[:50],   # cap to 50 for readability
                },
            )

        return CheckResult(
            check_name="video_readability",
            passed=True,
            severity=SEVERITY_INFO,
            message=(
                f"All {len(video_paths)} clips readable by OpenCV. "
                f"Mean frame count: {sum(frame_counts)/len(frame_counts):.0f}."
            ),
            details={
                "total_checked": len(video_paths),
                "corrupt_count": 0,
                "mean_frame_count": round(sum(frame_counts) / max(len(frame_counts), 1), 1),
                "mean_duration_sec": round(sum(durations) / max(len(durations), 1), 2),
            },
        )

    # ------------------------------------------------------------------
    # Check 4 — Frame count range
    # ------------------------------------------------------------------

    def _check_frame_count_range(self) -> CheckResult:
        """
        Verify that all clips have frame counts within [min_frames, max_frames].

        Clips with too few frames do not provide enough temporal signal for
        the LSTM. Clips with too many frames will be truncated or padded
        aggressively, wasting compute. Both extremes are logged as warnings.

        Note: This check uses ``frame_count`` populated by Check 3. If
        Check 3 was skipped (no OpenCV), this check uses an empty series
        and reports INFO.
        """
        if "frame_count" not in self._found_df.columns:
            return CheckResult(
                check_name="frame_count_range",
                passed=True,
                severity=SEVERITY_INFO,
                message="Frame count check skipped (video_readability not run).",
                details={"skipped": True},
            )

        valid_mask = self._found_df["frame_count"] > 0
        frame_counts = self._found_df.loc[valid_mask, "frame_count"]

        too_short = self._found_df[
            (self._found_df["frame_count"] > 0) &
            (self._found_df["frame_count"] < self._min_frames)
        ]
        too_long = self._found_df[
            self._found_df["frame_count"] > self._max_frames
        ]

        out_of_range_ids = (
            too_short["video_id"].tolist() + too_long["video_id"].tolist()
        )

        details: dict[str, Any] = {
            "min_frames_threshold": self._min_frames,
            "max_frames_threshold": self._max_frames,
            "min_frames_found": int(frame_counts.min()) if len(frame_counts) > 0 else 0,
            "max_frames_found": int(frame_counts.max()) if len(frame_counts) > 0 else 0,
            "mean_frames_found": round(float(frame_counts.mean()), 1) if len(frame_counts) > 0 else 0,
            "too_short_count": len(too_short),
            "too_long_count": len(too_long),
            "out_of_range_video_ids": out_of_range_ids[:50],
        }

        if out_of_range_ids:
            return CheckResult(
                check_name="frame_count_range",
                passed=False,
                severity=SEVERITY_WARNING,
                message=(
                    f"{len(too_short)} clips too short (<{self._min_frames} frames), "
                    f"{len(too_long)} clips too long (>{self._max_frames} frames). "
                    "These will be handled by the extractor's skip/truncation policy."
                ),
                details=details,
            )

        return CheckResult(
            check_name="frame_count_range",
            passed=True,
            severity=SEVERITY_INFO,
            message=(
                f"All clips in acceptable frame range [{self._min_frames}–{self._max_frames}]. "
                f"Range found: [{details['min_frames_found']}–{details['max_frames_found']}]."
            ),
            details=details,
        )

    # ------------------------------------------------------------------
    # Check 5 — No duplicate video IDs
    # ------------------------------------------------------------------

    def _check_no_duplicate_video_ids(self) -> CheckResult:
        """
        Verify that video IDs are globally unique across all signs.

        Duplicate IDs would cause the same clip to appear in multiple
        rows of the inventory, potentially in both train and val splits,
        creating data leakage. This is a hard blocker.
        """
        all_ids = self._df["video_id"].tolist()
        id_counts = pd.Series(all_ids).value_counts()
        duplicates = id_counts[id_counts > 1]

        if len(duplicates) > 0:
            dup_details = [
                {
                    "video_id": vid,
                    "count": int(cnt),
                    "signs": self._df[self._df["video_id"] == vid]["sign_label"].tolist(),
                }
                for vid, cnt in duplicates.head(20).items()
            ]
            return CheckResult(
                check_name="no_duplicate_video_ids",
                passed=False,
                severity=SEVERITY_ERROR,
                message=(
                    f"{len(duplicates)} duplicate video ID(s) found. "
                    "Duplicates must be resolved before splitting — they cause "
                    "data leakage between train and val splits."
                ),
                details={
                    "duplicate_count": len(duplicates),
                    "duplicate_examples": dup_details,
                },
            )

        return CheckResult(
            check_name="no_duplicate_video_ids",
            passed=True,
            severity=SEVERITY_INFO,
            message=f"All {len(all_ids)} video IDs are unique.",
            details={"total_video_ids": len(all_ids)},
        )

    # ------------------------------------------------------------------
    # Check 6 — Signer IDs complete
    # ------------------------------------------------------------------

    def _check_signer_ids_complete(self) -> CheckResult:
        """
        Verify that every found clip has a valid signer_id.

        signer_id = -1 indicates an unknown signer (assigned during
        inventory build when the manifest entry was missing this field).
        Clips with unknown signer IDs cannot be safely assigned to a
        split — assigning them to train risks their appearing in val
        under a different sign. This is a hard blocker.
        """
        unknown_signer = self._found_df[self._found_df["signer_id"] == -1]

        if len(unknown_signer) > 0:
            by_sign = unknown_signer.groupby("sign_label").size().to_dict()
            return CheckResult(
                check_name="signer_ids_complete",
                passed=False,
                severity=SEVERITY_ERROR,
                message=(
                    f"{len(unknown_signer)} clip(s) have unknown signer_id (-1). "
                    "Signer-aware splitting is impossible without complete signer IDs."
                ),
                details={
                    "unknown_count": len(unknown_signer),
                    "unknown_video_ids": unknown_signer["video_id"].tolist()[:50],
                    "by_sign": by_sign,
                },
            )

        unique_signers = self._found_df["signer_id"].nunique()
        return CheckResult(
            check_name="signer_ids_complete",
            passed=True,
            severity=SEVERITY_INFO,
            message=(
                f"All {len(self._found_df)} found clips have valid signer IDs. "
                f"Unique signers: {unique_signers}."
            ),
            details={
                "total_clips_with_signer_id": len(self._found_df),
                "unique_signers": unique_signers,
                "signer_id_range": [
                    int(self._found_df["signer_id"].min()),
                    int(self._found_df["signer_id"].max()),
                ],
            },
        )

    # ------------------------------------------------------------------
    # Check 7 — Dataset size (informational)
    # ------------------------------------------------------------------

    def _check_dataset_size(self) -> CheckResult:
        """
        Compute total dataset size and basic clip statistics.

        This check always passes. It exists to surface dataset size
        information in the validation report for documentation purposes.
        """
        total_size_mb = self._found_df["file_size_mb"].sum()
        total_size_gb = total_size_mb / 1024

        return CheckResult(
            check_name="dataset_size",
            passed=True,
            severity=SEVERITY_INFO,
            message=(
                f"Dataset: {len(self._found_df)} clips | "
                f"{total_size_mb:.1f} MB ({total_size_gb:.2f} GB) | "
                f"{self._found_df['signer_id'].nunique()} unique signers."
            ),
            details={
                "total_clips_found": len(self._found_df),
                "total_size_mb": round(total_size_mb, 2),
                "total_size_gb": round(total_size_gb, 3),
                "unique_signers": int(self._found_df["signer_id"].nunique()),
                "unique_signs": int(self._found_df["sign_label"].nunique()),
            },
        )

    # ------------------------------------------------------------------
    # Check 8 — Class imbalance
    # ------------------------------------------------------------------

    def _check_class_imbalance(self) -> CheckResult:
        """
        Check whether any sign dominates the dataset disproportionately.

        Computes the max/min clip-count ratio. If this exceeds
        ``imbalance_threshold`` (default 3.0), the model may implicitly
        learn to favour high-frequency classes. This is a warning, not
        a blocker — class weight balancing in training can mitigate it.
        """
        clips_per_sign = self._found_df.groupby("sign_label").size()
        min_clips = int(clips_per_sign.min())
        max_clips = int(clips_per_sign.max())

        if min_clips == 0:
            ratio = float("inf")
        else:
            ratio = max_clips / min_clips

        min_signs = clips_per_sign[clips_per_sign == min_clips].index.tolist()
        max_signs = clips_per_sign[clips_per_sign == max_clips].index.tolist()

        details: dict[str, Any] = {
            "imbalance_ratio": round(ratio, 2),
            "threshold": self._imbalance_threshold,
            "min_clips": min_clips,
            "max_clips": max_clips,
            "signs_with_min_clips": min_signs,
            "signs_with_max_clips": max_signs,
            "clips_per_sign": dict(clips_per_sign.sort_values().to_dict()),
        }

        if ratio > self._imbalance_threshold:
            return CheckResult(
                check_name="class_imbalance",
                passed=False,
                severity=SEVERITY_WARNING,
                message=(
                    f"Class imbalance ratio {ratio:.2f}x exceeds threshold "
                    f"{self._imbalance_threshold}x. "
                    f"Min: {min_clips} clips ({min_signs}), "
                    f"Max: {max_clips} clips ({max_signs}). "
                    "Consider enabling class_weight_balancing in TrainingConfig."
                ),
                details=details,
            )

        return CheckResult(
            check_name="class_imbalance",
            passed=True,
            severity=SEVERITY_INFO,
            message=(
                f"Class balance within acceptable ratio: {ratio:.2f}x "
                f"(threshold {self._imbalance_threshold}x). "
                f"Range: {min_clips}–{max_clips} clips per sign."
            ),
            details=details,
        )

    # ------------------------------------------------------------------
    # Statistics helpers
    # ------------------------------------------------------------------

    def _compute_per_sign_stats(self) -> dict[str, dict[str, Any]]:
        """
        Compute per-sign clip and quality statistics for the report.

        Uses the frame_count / duration_sec columns if populated by
        Check 3. Falls back to zeros if OpenCV was not available.
        """
        stats: dict[str, dict[str, Any]] = {}
        has_frame_data = "frame_count" in self._found_df.columns

        for sign, group in self._found_df.groupby("sign_label"):
            entry: dict[str, Any] = {
                "clips_found": len(group),
                "unique_signers": int(group["signer_id"].nunique()),
                "total_size_mb": round(float(group["file_size_mb"].sum()), 2),
            }

            if has_frame_data:
                valid_frames = group["frame_count"][group["frame_count"] > 0]
                if len(valid_frames) > 0:
                    entry["mean_frame_count"] = round(float(valid_frames.mean()), 1)
                    entry["min_frame_count"] = int(valid_frames.min())
                    entry["max_frame_count"] = int(valid_frames.max())
                    entry["std_frame_count"] = round(float(valid_frames.std()), 1)
                else:
                    entry.update({
                        "mean_frame_count": 0, "min_frame_count": 0,
                        "max_frame_count": 0, "std_frame_count": 0,
                    })

                valid_dur = group["duration_sec"][group["duration_sec"] > 0]
                if len(valid_dur) > 0:
                    entry["mean_duration_sec"] = round(float(valid_dur.mean()), 2)
                    entry["min_duration_sec"] = round(float(valid_dur.min()), 2)
                    entry["max_duration_sec"] = round(float(valid_dur.max()), 2)
                else:
                    entry.update({
                        "mean_duration_sec": 0.0,
                        "min_duration_sec": 0.0,
                        "max_duration_sec": 0.0,
                    })

            stats[str(sign)] = entry

        return stats

    def _compute_dataset_totals(self) -> dict[str, Any]:
        """Compute aggregate dataset-level statistics."""
        clips_per_sign = self._found_df.groupby("sign_label").size()

        totals: dict[str, Any] = {
            "total_clips_in_inventory": len(self._df),
            "total_clips_found": len(self._found_df),
            "total_clips_missing": len(self._df) - len(self._found_df),
            "total_size_mb": round(float(self._found_df["file_size_mb"].sum()), 2),
            "total_unique_signers": int(self._found_df["signer_id"].nunique()),
            "total_unique_signs": int(self._found_df["sign_label"].nunique()),
            "mean_clips_per_sign": round(float(clips_per_sign.mean()), 1),
            "min_clips_per_sign": int(clips_per_sign.min()),
            "max_clips_per_sign": int(clips_per_sign.max()),
            "std_clips_per_sign": round(float(clips_per_sign.std()), 1),
        }

        if "frame_count" in self._found_df.columns:
            valid = self._found_df[self._found_df["frame_count"] > 0]
            if len(valid) > 0:
                totals["mean_frame_count"] = round(float(valid["frame_count"].mean()), 1)
                totals["mean_duration_sec"] = round(float(valid["duration_sec"].mean()), 2)
                totals["total_duration_hours"] = round(
                    float(valid["duration_sec"].sum()) / 3600, 3
                )

        return totals


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def _check_video_readable(video_path: str) -> tuple[bool, int, float]:
    """
    Attempt to open a video file with OpenCV and read its metadata.

    Does NOT load any frames into memory — only reads header metadata
    via CAP_PROP_FRAME_COUNT and CAP_PROP_FPS. This makes the check
    fast enough to run on thousands of clips in seconds.

    Parameters
    ----------
    video_path : str
        Absolute path to the video file.

    Returns
    -------
    tuple[bool, int, float]
        (readable, frame_count, duration_seconds)
        If not readable, returns (False, 0, 0.0).
    """
    if not _CV2_AVAILABLE:
        return True, 0, 0.0

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            return False, 0, 0.0

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        # Guard against invalid fps values (0 or negative)
        if fps is None or fps <= 0:
            fps = 30.0

        duration = frame_count / fps if frame_count > 0 else 0.0
        return True, max(frame_count, 0), round(duration, 3)

    except Exception:  # noqa: BLE001
        return False, 0, 0.0