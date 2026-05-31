"""
src/data/splitter.py
=====================
Signer-aware train/val/test split for the WLASL gesture recognition pipeline.

Why signer-aware splitting is critical
---------------------------------------
A naïve random split ignoring signers allows the same individual's signing
style to appear in both training and validation sets. Because LSTM models
can memorise idiosyncratic motion patterns (hand shape velocity, trajectory
curvature, rest position), this creates a form of data leakage that inflates
validation accuracy by 10–25 percentage points compared to the true
generalisation performance on unseen signers.

The honest question for a production sign language system is: *can the model
recognise a sign performed by someone it has never seen before?* Signer-aware
splitting forces exactly this evaluation.

Algorithm
---------
The ``SignerAwareSplitter`` uses a greedy bin-packing approach:

1.  Enumerate all unique signers in the found inventory.
2.  Sort signers by their total clip count (descending). Placing large
    contributors first gives the bin-packing better balance.
3.  Iterate through signers. At each step, assign the current signer to the
    split whose current clip fraction is furthest below its target ratio.
4.  Post-assignment: verify zero signer overlap across splits (hard assertion).
5.  Post-assignment: verify all 35 signs appear in train and val splits.
    If a sign is entirely absent from val, log a prominent warning (the model
    cannot be evaluated on that sign — a real data limitation).

Output CSVs
-----------
Three CSVs written to ``data/splits/``:

    train.csv  val.csv  test.csv

Columns:

    video_id       str     WLASL identifier
    sign_label     str     Human-readable sign name
    class_idx      int     Integer 0–34 from label_map_v1.json
    signer_id      int     WLASL signer identifier
    split          str     "train" / "val" / "test"
    video_path     str     Relative path from repo root
    frame_count    int     From validation step (0 if unknown)
    duration_sec   float   Duration in seconds (0.0 if unknown)
    file_size_mb   float   File size (0.0 if missing)

Split summary
-------------
Written to ``data/splits/split_summary.json``:

    {
        "created_utc": "...",
        "seed": 42,
        "algorithm": "greedy_bin_packing_signer_aware",
        "targets": {"train": 0.70, "val": 0.15, "test": 0.15},
        "actual_ratios": {"train": 0.703, "val": 0.148, "test": 0.149},
        "clip_counts": {"train": 834, "val": 176, "test": 177},
        "signer_counts": {"train": 89, "val": 19, "test": 21},
        "signer_overlap": {"train_val": 0, "train_test": 0, "val_test": 0},
        "classes_in_each_split": {"train": 35, "val": 35, "test": 34},
        "signs_missing_from_val": [],
        "signs_missing_from_test": ["thanksgiving"],
        "per_class_per_split": {
            "book": {"train": 45, "val": 10, "test": 9}
        }
    }

Reproducibility guarantee
-------------------------
The split is fully deterministic given the same ``seed``. The same seed always
produces the same signer assignments, which always produces the same CSV rows.
Changing the seed changes the split — this is intentional and enables
sensitivity analysis across different signer partitions.

Usage
-----
    from src.data.splitter import SignerAwareSplitter

    splitter = SignerAwareSplitter(
        inventory_df=inventory_df,      # from WLASLResolver (found=True only)
        splits_dir="data/splits",
        targets={"train": 0.70, "val": 0.15, "test": 0.15},
        seed=42,
        min_clips_per_class_per_split=3,
    )
    result = splitter.split()

    # CSVs are written automatically; also accessible as DataFrames:
    train_df = result.train
    val_df   = result.val
    test_df  = result.test
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Repository root
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Required CSV columns — enforced on output
# ---------------------------------------------------------------------------
_SPLIT_CSV_COLUMNS = [
    "video_id",
    "sign_label",
    "class_idx",
    "signer_id",
    "split",
    "video_path",
    "frame_count",
    "duration_sec",
    "file_size_mb",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SplitResult:
    """
    Container for the three split DataFrames and the summary statistics.

    Attributes
    ----------
    train : pd.DataFrame
        Training split. Typically 70% of clips by count.
    val : pd.DataFrame
        Validation split. Typically 15% of clips by count.
    test : pd.DataFrame
        Test split. Typically 15% of clips by count.
    summary : dict
        The full split summary dictionary (mirrors split_summary.json).
    splits_dir : Path
        Directory where CSV files were written.
    """
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    summary: dict[str, Any]
    splits_dir: Path

    @property
    def all_splits(self) -> dict[str, pd.DataFrame]:
        return {"train": self.train, "val": self.val, "test": self.test}

    def __repr__(self) -> str:
        return (
            f"SplitResult("
            f"train={len(self.train)}, "
            f"val={len(self.val)}, "
            f"test={len(self.test)}, "
            f"splits_dir='{self.splits_dir}')"
        )


# ---------------------------------------------------------------------------
# SignerAwareSplitter
# ---------------------------------------------------------------------------

class SignerAwareSplitter:
    """
    Assigns WLASL clips to train/val/test splits with zero signer overlap.

    Every clip produced by a given signer is assigned exclusively to one
    split. No signer appears in more than one split. This guarantees that
    the validation and test sets measure generalisation to unseen signers
    rather than recognition of familiar motion patterns.

    Parameters
    ----------
    inventory_df : pd.DataFrame
        The inventory DataFrame from WLASLResolver. Only found=True clips
        are included in splits; missing clips are excluded and logged.
    splits_dir : str | Path
        Directory where train.csv, val.csv, test.csv, and split_summary.json
        will be written. Created if it does not exist.
    targets : dict[str, float]
        Target clip-count proportions. Must sum to 1.0.
        Default: {"train": 0.70, "val": 0.15, "test": 0.15}
    seed : int
        Random seed for reproducibility. Default 42.
    min_clips_per_class_per_split : int
        Minimum clips per sign class per split to flag in warnings.
        Default 3. Not enforced — only logged.

    Raises
    ------
    ValueError
        If targets do not sum to 1.0 (within floating-point tolerance).
        If inventory_df is missing required columns.
    """

    _REQUIRED_COLUMNS = {"video_id", "sign_label", "class_idx", "signer_id", "found", "video_path"}

    def __init__(
        self,
        inventory_df: pd.DataFrame,
        splits_dir: str | Path,
        targets: Optional[dict[str, float]] = None,
        seed: int = 42,
        min_clips_per_class_per_split: int = 3,
    ) -> None:
        self._splits_dir = Path(splits_dir).resolve()
        self._seed = seed
        self._min_clips_flag = min_clips_per_class_per_split
        self._targets: dict[str, float] = targets or {
            "train": 0.70, "val": 0.15, "test": 0.15,
        }

        # Validate targets
        total = sum(self._targets.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Split targets must sum to 1.0, got {total:.6f}. "
                f"Targets: {self._targets}"
            )

        if not {"train", "val", "test"}.issubset(self._targets):
            raise ValueError(
                "Targets must contain keys 'train', 'val', and 'test'. "
                f"Got: {list(self._targets.keys())}"
            )

        # Validate required columns
        missing_cols = self._REQUIRED_COLUMNS - set(inventory_df.columns)
        if missing_cols:
            raise ValueError(
                f"inventory_df is missing required columns: {missing_cols}. "
                "Ensure the DataFrame comes from WLASLResolver.build_inventory()."
            )

        # Work only with found clips
        self._all_df = inventory_df.copy()
        self._df = inventory_df[inventory_df["found"] == True].copy()

        n_missing = len(inventory_df) - len(self._df)
        if n_missing > 0:
            logger.warning(
                f"{n_missing} clips excluded from splits (found=False). "
                "These clips are missing from disk and cannot be used in training.",
                extra={"stage": "splitting"},
            )

        # Ensure optional columns exist with defaults
        for col, default in [("frame_count", 0), ("duration_sec", 0.0), ("file_size_mb", 0.0)]:
            if col not in self._df.columns:
                self._df[col] = default

        logger.info(
            f"SignerAwareSplitter initialised | "
            f"clips={len(self._df)} | "
            f"signers={self._df['signer_id'].nunique()} | "
            f"signs={self._df['sign_label'].nunique()} | "
            f"targets={self._targets} | "
            f"seed={seed}",
            extra={"stage": "splitting"},
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def split(self, force: bool = False) -> SplitResult:
        """
        Execute the signer-aware split and write output CSVs.

        Parameters
        ----------
        force : bool
            If False (default) and all three CSVs already exist in
            ``splits_dir``, load and return them without re-splitting.
            Pass True to force a full re-split.

        Returns
        -------
        SplitResult
            Container with train/val/test DataFrames and summary statistics.
        """
        # Resumability: return cached split if CSVs already exist
        if not force and self._cached_csvs_exist():
            logger.info(
                "Loading cached split CSVs from splits_dir. "
                "Pass force=True to re-split.",
                extra={"stage": "splitting"},
            )
            return self._load_cached_split()

        logger.info(
            f"Starting signer-aware split | "
            f"algorithm=greedy_bin_packing | seed={self._seed}",
            extra={"stage": "splitting"},
        )

        # Step 1: Compute per-signer clip counts
        signer_clip_counts = (
            self._df.groupby("signer_id").size().to_dict()
        )
        logger.info(
            f"Signer analysis | "
            f"unique_signers={len(signer_clip_counts)} | "
            f"max_clips_per_signer={max(signer_clip_counts.values())} | "
            f"min_clips_per_signer={min(signer_clip_counts.values())}",
            extra={"stage": "splitting"},
        )

        # Step 2: Greedy bin-packing assignment
        signer_assignments = self._assign_signers(signer_clip_counts)

        # Step 3: Build split DataFrames
        split_frames: dict[str, pd.DataFrame] = {}
        for split_name in ("train", "val", "test"):
            assigned_signers = {
                sid for sid, sp in signer_assignments.items() if sp == split_name
            }
            split_df = self._df[
                self._df["signer_id"].isin(assigned_signers)
            ].copy()
            split_df["split"] = split_name
            split_frames[split_name] = split_df

        # Step 4: Validate — zero signer overlap is a hard requirement
        self._assert_no_signer_overlap(split_frames)

        # Step 5: Validate class coverage — warn if signs missing from val/test
        self._check_class_coverage(split_frames)

        # Step 6: Warn on thin class-split combinations
        self._check_thin_class_splits(split_frames)

        # Step 7: Write CSVs
        self._splits_dir.mkdir(parents=True, exist_ok=True)
        for split_name, split_df in split_frames.items():
            self._write_split_csv(split_df, split_name)

        # Step 8: Build and write summary
        summary = self._build_summary(split_frames, signer_assignments)
        self._write_summary(summary)

        self._log_split_overview(split_frames, summary)

        return SplitResult(
            train=split_frames["train"],
            val=split_frames["val"],
            test=split_frames["test"],
            summary=summary,
            splits_dir=self._splits_dir,
        )

    # ------------------------------------------------------------------
    # Core splitting algorithm
    # ------------------------------------------------------------------

    def _assign_signers(
        self,
        signer_clip_counts: dict[int, int],
    ) -> dict[int, str]:
        """
        Greedy bin-packing signer assignment.

        Assigns each signer exclusively to one split by iterating through
        signers (large-to-small by clip count) and placing each in the split
        that is currently furthest below its target ratio.

        The descending-by-clip-count ordering is key: placing large signers
        first prevents a situation where one split overshoots its target badly
        because many small signers were assigned arbitrarily.

        Parameters
        ----------
        signer_clip_counts : dict[int, int]
            Mapping of signer_id → total clip count.

        Returns
        -------
        dict[int, str]
            Mapping of signer_id → split_name.
        """
        rng = random.Random(self._seed)

        # Group signers by clip count, then shuffle within each group
        # This introduces controlled randomness within ties (same clip count)
        # while still placing larger signers first.
        signers_sorted = sorted(
            signer_clip_counts.keys(),
            key=lambda s: (-signer_clip_counts[s], rng.random()),
        )

        # Running clip totals per split
        running_totals: dict[str, int] = {"train": 0, "val": 0, "test": 0}
        assignments: dict[int, str] = {}

        for signer_id in signers_sorted:
            clips = signer_clip_counts[signer_id]
            total_assigned = sum(running_totals.values()) + clips

            # Choose the split furthest below its target ratio
            best_split = min(
                self._targets.keys(),
                key=lambda s: (
                    running_totals[s] / max(total_assigned, 1)
                ) - self._targets[s],
            )

            assignments[signer_id] = best_split
            running_totals[best_split] += clips

            logger.debug(
                f"Assigned signer {signer_id} ({clips} clips) → {best_split} | "
                f"running: train={running_totals['train']} "
                f"val={running_totals['val']} "
                f"test={running_totals['test']}",
                extra={"stage": "splitting"},
            )

        # Log final allocation
        total = sum(running_totals.values())
        for split_name, count in running_totals.items():
            ratio = count / total if total > 0 else 0.0
            logger.info(
                f"  {split_name:6s}: {count:4d} clips ({ratio:.1%}) | "
                f"target={self._targets[split_name]:.1%}",
                extra={"stage": "splitting"},
            )

        return assignments

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _assert_no_signer_overlap(
        self,
        split_frames: dict[str, pd.DataFrame],
    ) -> None:
        """
        Assert that no signer appears in more than one split.

        This is the core methodological guarantee of signer-aware splitting.
        If this assertion fails, the split algorithm has a bug — this should
        never happen in correct operation.

        Raises
        ------
        AssertionError
            If any signer_id appears in two or more splits.
        """
        signer_sets = {
            name: set(df["signer_id"].unique())
            for name, df in split_frames.items()
        }

        all_clean = True
        for s1, s2 in combinations(list(split_frames.keys()), 2):
            overlap = signer_sets[s1] & signer_sets[s2]
            if overlap:
                all_clean = False
                logger.error(
                    f"SIGNER OVERLAP DETECTED between {s1} and {s2}: "
                    f"{len(overlap)} signer(s) appear in both splits: {overlap}. "
                    "This is a bug in the splitting algorithm.",
                    extra={"stage": "splitting"},
                )

        assert all_clean, (
            "Signer overlap detected in split assignment. "
            "The signer-aware split guarantee has been violated. "
            "Check the splitting algorithm."
        )

        logger.info(
            "✓ Signer overlap check passed — zero overlap across all split pairs.",
            extra={"stage": "splitting"},
        )

    def _check_class_coverage(
        self,
        split_frames: dict[str, pd.DataFrame],
    ) -> None:
        """
        Check whether all sign classes appear in train and val.

        A sign missing from the val split means the model cannot be evaluated
        on it — a real data limitation that should be documented.

        Signs missing from train cannot be learned at all — this is an
        ERROR that requires adjusting the split (e.g. changing the seed).
        """
        all_classes = set(self._df["sign_label"].unique())

        for split_name in ("train", "val", "test"):
            split_classes = set(split_frames[split_name]["sign_label"].unique())
            missing = all_classes - split_classes

            if missing:
                severity = "ERROR" if split_name == "train" else "WARNING"
                logger.warning(
                    f"[{severity}] {len(missing)} sign(s) have NO clips in "
                    f"the {split_name} split: {sorted(missing)}. "
                    + (
                        "The model cannot learn these signs. "
                        "Try a different seed or manually adjust signer assignments."
                        if split_name == "train"
                        else
                        f"These signs cannot be evaluated in {split_name}. "
                        "Document this in LIMITATIONS.md."
                    ),
                    extra={"stage": "splitting"},
                )
            else:
                logger.info(
                    f"✓ All {len(all_classes)} signs present in {split_name} split.",
                    extra={"stage": "splitting"},
                )

    def _check_thin_class_splits(
        self,
        split_frames: dict[str, pd.DataFrame],
    ) -> None:
        """
        Warn on class-split combinations with very few clips.

        A sign with fewer than ``min_clips_per_class_per_split`` clips in
        the val or test set will produce high-variance per-class metrics.
        This is a known limitation of small datasets and should be documented.
        """
        for split_name in ("val", "test"):
            df = split_frames[split_name]
            per_class = df.groupby("sign_label").size()
            thin = per_class[per_class < self._min_clips_flag]
            if len(thin) > 0:
                logger.warning(
                    f"{len(thin)} sign(s) have <{self._min_clips_flag} clips "
                    f"in {split_name} — per-class metrics will be noisy: "
                    f"{thin.to_dict()}",
                    extra={"stage": "splitting"},
                )

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def _write_split_csv(self, df: pd.DataFrame, split_name: str) -> Path:
        """
        Write a split DataFrame to a CSV file with enforced column ordering.

        Parameters
        ----------
        df : pd.DataFrame
            The split DataFrame to write.
        split_name : str
            One of "train", "val", "test". Used in the filename.

        Returns
        -------
        Path
            Path to the written CSV.
        """
        # Ensure all required columns exist
        for col in _SPLIT_CSV_COLUMNS:
            if col not in df.columns:
                df = df.copy()
                df[col] = 0 if col in ("frame_count",) else (
                    0.0 if col in ("duration_sec", "file_size_mb") else ""
                )

        output_df = df[_SPLIT_CSV_COLUMNS].sort_values(
            ["sign_label", "signer_id", "video_id"]
        ).reset_index(drop=True)

        csv_path = self._splits_dir / f"{split_name}.csv"
        output_df.to_csv(csv_path, index=False)

        logger.info(
            f"Written: {csv_path} | "
            f"rows={len(output_df)} | "
            f"signs={output_df['sign_label'].nunique()} | "
            f"signers={output_df['signer_id'].nunique()}",
            extra={"stage": "splitting"},
        )
        return csv_path

    def _build_summary(
        self,
        split_frames: dict[str, pd.DataFrame],
        signer_assignments: dict[int, str],
    ) -> dict[str, Any]:
        """
        Build the split summary dictionary for JSON output.

        Parameters
        ----------
        split_frames : dict[str, pd.DataFrame]
            The three split DataFrames.
        signer_assignments : dict[int, str]
            Signer-to-split assignment mapping.

        Returns
        -------
        dict
            Complete split summary.
        """
        total_clips = sum(len(df) for df in split_frames.values())

        clip_counts = {s: len(df) for s, df in split_frames.items()}
        actual_ratios = {
            s: round(n / total_clips, 4) if total_clips > 0 else 0.0
            for s, n in clip_counts.items()
        }
        signer_counts = {
            s: int(df["signer_id"].nunique()) for s, df in split_frames.items()
        }
        signer_sets = {
            s: set(df["signer_id"].unique()) for s, df in split_frames.items()
        }

        # Overlap verification (for the summary — assertion already passed)
        overlap_counts = {
            f"{s1}_{s2}": len(signer_sets[s1] & signer_sets[s2])
            for s1, s2 in combinations(list(split_frames.keys()), 2)
        }

        # Class coverage per split
        all_classes = set(self._df["sign_label"].unique())
        classes_in_each_split = {
            s: int(df["sign_label"].nunique()) for s, df in split_frames.items()
        }
        signs_missing_from_val = sorted(
            all_classes - set(split_frames["val"]["sign_label"].unique())
        )
        signs_missing_from_test = sorted(
            all_classes - set(split_frames["test"]["sign_label"].unique())
        )

        # Per-class per-split clip counts
        per_class_per_split: dict[str, dict[str, int]] = {}
        for split_name, df in split_frames.items():
            per_class = df.groupby("sign_label").size().to_dict()
            for sign, count in per_class.items():
                if sign not in per_class_per_split:
                    per_class_per_split[sign] = {"train": 0, "val": 0, "test": 0}
                per_class_per_split[sign][split_name] = int(count)

        return {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "seed": self._seed,
            "algorithm": "greedy_bin_packing_signer_aware",
            "targets": self._targets,
            "actual_ratios": actual_ratios,
            "clip_counts": clip_counts,
            "signer_counts": signer_counts,
            "signer_overlap": overlap_counts,
            "classes_in_each_split": classes_in_each_split,
            "signs_missing_from_val": signs_missing_from_val,
            "signs_missing_from_test": signs_missing_from_test,
            "total_clips_split": total_clips,
            "total_clips_excluded_missing": len(self._all_df) - len(self._df),
            "per_class_per_split": dict(sorted(per_class_per_split.items())),
        }

    def _write_summary(self, summary: dict[str, Any]) -> Path:
        """Write the split summary to data/splits/split_summary.json."""
        path = self._splits_dir / "split_summary.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(
            f"Split summary written: {path}",
            extra={"stage": "splitting"},
        )
        return path

    def _log_split_overview(
        self,
        split_frames: dict[str, pd.DataFrame],
        summary: dict[str, Any],
    ) -> None:
        """Log a concise human-readable split overview."""
        logger.info("=" * 65, extra={"stage": "splitting"})
        logger.info("SPLIT SUMMARY", extra={"stage": "splitting"})
        logger.info("=" * 65, extra={"stage": "splitting"})
        for split_name in ("train", "val", "test"):
            n_clips = summary["clip_counts"][split_name]
            n_signers = summary["signer_counts"][split_name]
            ratio = summary["actual_ratios"][split_name]
            target = summary["targets"][split_name]
            n_classes = summary["classes_in_each_split"][split_name]
            logger.info(
                f"  {split_name:6s} | "
                f"{n_clips:5d} clips ({ratio:.1%}, target {target:.0%}) | "
                f"{n_signers:3d} signers | "
                f"{n_classes:2d} classes",
                extra={"stage": "splitting"},
            )
        logger.info(
            f"  Signer overlap — "
            f"train∩val: {summary['signer_overlap']['train_val']} | "
            f"train∩test: {summary['signer_overlap']['train_test']} | "
            f"val∩test: {summary['signer_overlap']['val_test']}",
            extra={"stage": "splitting"},
        )
        if summary["signs_missing_from_val"]:
            logger.warning(
                f"  Signs missing from val: {summary['signs_missing_from_val']}",
                extra={"stage": "splitting"},
            )
        logger.info("=" * 65, extra={"stage": "splitting"})

    # ------------------------------------------------------------------
    # Resumability helpers
    # ------------------------------------------------------------------

    def _cached_csvs_exist(self) -> bool:
        """Return True if all three split CSVs exist in splits_dir."""
        return all(
            (self._splits_dir / f"{name}.csv").exists()
            for name in ("train", "val", "test")
        )

    def _load_cached_split(self) -> SplitResult:
        """
        Load split DataFrames from existing CSVs.

        Returns
        -------
        SplitResult
            Populated from the cached CSV files.
        """
        split_frames: dict[str, pd.DataFrame] = {}
        for split_name in ("train", "val", "test"):
            csv_path = self._splits_dir / f"{split_name}.csv"
            df = pd.read_csv(csv_path)
            split_frames[split_name] = df
            logger.info(
                f"Loaded cached split: {csv_path} | rows={len(df)}",
                extra={"stage": "splitting"},
            )

        # Load summary if it exists
        summary_path = self._splits_dir / "split_summary.json"
        summary: dict[str, Any] = {}
        if summary_path.exists():
            with open(summary_path, encoding="utf-8") as f:
                summary = json.load(f)

        return SplitResult(
            train=split_frames["train"],
            val=split_frames["val"],
            test=split_frames["test"],
            summary=summary,
            splits_dir=self._splits_dir,
        )