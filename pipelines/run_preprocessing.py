"""
pipelines/run_preprocessing.py
================================
Stage 1 pipeline entry point — Data Ingestion, Validation, and Splitting.

Overview
--------
This script orchestrates the full Stage 1 pipeline:

    1. Resolve local WLASL videos against WLASL_v0.3.json
       (WLASLResolver → data/raw_inventory.json)

    2. Validate the inventory — 8 integrity checks
       (DataValidator → data/data_validation_report.json)

    3. Split into train/val/test with zero signer overlap
       (SignerAwareSplitter → data/splits/{train,val,test}.csv)

Each stage is independently resumable: if the output artifact already exists,
the stage is skipped unless ``--force`` is passed.

Usage
-----
Full pipeline (standard run):
    python pipelines/run_preprocessing.py \\
        --manifest data/raw/WLASL_v0.3.json \\
        --raw-dir data/raw

Validate only (skip resolve and split):
    python pipelines/run_preprocessing.py --validate-only

Split only (skip resolve and validate):
    python pipelines/run_preprocessing.py --split-only

Force full re-run (overwrite all cached artifacts):
    python pipelines/run_preprocessing.py --force

Attempt to download missing clips after resolve:
    python pipelines/run_preprocessing.py --download-missing

Dry-run download (show what would be downloaded, don't fetch):
    python pipelines/run_preprocessing.py --download-missing --dry-run

Exit codes
----------
0  — All stages completed (or were skipped as already done) and
     pipeline_can_proceed=True.
1  — One or more ERROR-severity validation checks failed.
     Check data/data_validation_report.json for details.
2  — An unexpected exception terminated the pipeline. Check logs.

Integration with the utils package
-----------------------------------
All logging uses configure_logging() → get_logger(__name__). The
active MLflow run is NOT started here — Stage 1 is pure data preparation.
MLflow tracking begins in Stage 5 (run_training.py).

Reproducibility
---------------
set_seeds(42) is called at the top of main() before any data operations.
The split algorithm is fully deterministic given the same seed and
inventory order.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
import pandas as pd

# ---------------------------------------------------------------------------
# Bootstrap: ensure repo root is on sys.path so src/ imports resolve
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Utils — must be imported before data modules so logging is configured first
# ---------------------------------------------------------------------------
from src.utils.logger import configure_logging, get_logger
from src.utils.reproducibility import set_seeds
from src.utils.label_map import get_label_map

# ---------------------------------------------------------------------------
# Data modules
# ---------------------------------------------------------------------------
from src.data.downloader import WLASLResolver
from src.data.validator import DataValidator
from src.data.splitter import SignerAwareSplitter

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DEFAULT_MANIFEST       = str(_REPO_ROOT / "data" / "raw" / "WLASL_v0.3.json")
_DEFAULT_RAW_DIR        = str(_REPO_ROOT / "data" / "raw")
_DEFAULT_LABEL_MAP      = str(_REPO_ROOT / "artifacts" / "label_map_v1.json")
_DEFAULT_INVENTORY_PATH = str(_REPO_ROOT / "data" / "raw_inventory.json")
_DEFAULT_REPORT_PATH    = str(_REPO_ROOT / "data" / "data_validation_report.json")
_DEFAULT_SPLITS_DIR     = str(_REPO_ROOT / "data" / "splits")
_DEFAULT_LOG_DIR        = str(_REPO_ROOT / "logs")

# Validation thresholds — can be overridden via CLI
_MIN_CLIPS_PER_SIGN    = 20
_MIN_FRAMES            = 10
_MAX_FRAMES            = 300
_IMBALANCE_THRESHOLD   = 3.0
_SEED                  = 42


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_preprocessing.py",
        description=(
            "Stage 1: WLASL data ingestion, validation, and signer-aware splitting.\n\n"
            "By default, runs all three stages in sequence. Use --validate-only "
            "or --split-only to run a subset."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard full pipeline run
  python pipelines/run_preprocessing.py

  # Custom manifest location
  python pipelines/run_preprocessing.py --manifest /path/to/WLASL_v0.3.json

  # Force re-run (ignore cached artifacts)
  python pipelines/run_preprocessing.py --force

  # Download missing clips via yt-dlp
  python pipelines/run_preprocessing.py --download-missing

  # Validate only (requires existing inventory)
  python pipelines/run_preprocessing.py --validate-only

Exit codes: 0=success, 1=validation blocked pipeline, 2=unexpected error
        """,
    )

    # --- Paths ---
    parser.add_argument(
        "--manifest",
        default=_DEFAULT_MANIFEST,
        metavar="PATH",
        help=f"Path to WLASL_v0.3.json (default: {_DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--raw-dir",
        default=_DEFAULT_RAW_DIR,
        metavar="DIR",
        help=f"Root directory of raw WLASL videos (default: {_DEFAULT_RAW_DIR})",
    )
    parser.add_argument(
        "--label-map",
        default=_DEFAULT_LABEL_MAP,
        metavar="PATH",
        help=f"Path to label_map_v1.json (default: {_DEFAULT_LABEL_MAP})",
    )
    parser.add_argument(
        "--inventory-path",
        default=_DEFAULT_INVENTORY_PATH,
        metavar="PATH",
        help=f"Output path for raw_inventory.json (default: {_DEFAULT_INVENTORY_PATH})",
    )
    parser.add_argument(
        "--report-path",
        default=_DEFAULT_REPORT_PATH,
        metavar="PATH",
        help=f"Output path for data_validation_report.json (default: {_DEFAULT_REPORT_PATH})",
    )
    parser.add_argument(
        "--splits-dir",
        default=_DEFAULT_SPLITS_DIR,
        metavar="DIR",
        help=f"Output directory for split CSVs (default: {_DEFAULT_SPLITS_DIR})",
    )
    parser.add_argument(
        "--log-dir",
        default=_DEFAULT_LOG_DIR,
        metavar="DIR",
        help=f"Directory for log files (default: {_DEFAULT_LOG_DIR})",
    )

    # --- Stage selectors ---
    stage_group = parser.add_mutually_exclusive_group()
    stage_group.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Run validation only. Requires raw_inventory.json to already exist. "
            "Skips resolve and split stages."
        ),
    )
    stage_group.add_argument(
        "--split-only",
        action="store_true",
        help=(
            "Run split only. Requires raw_inventory.json and "
            "data_validation_report.json with pipeline_can_proceed=True. "
            "Skips resolve and validate stages."
        ),
    )

    # --- Behaviour flags ---
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Force re-run of all stages, overwriting any cached artifacts. "
            "Without this flag, stages skip if their output already exists."
        ),
    )
    parser.add_argument(
        "--download-missing",
        action="store_true",
        help=(
            "After resolving the inventory, attempt to download clips that "
            "could not be found locally via yt-dlp. Requires yt-dlp: "
            "'pip install yt-dlp'."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "When combined with --download-missing, logs what would be "
            "downloaded without actually fetching any files."
        ),
    )
    parser.add_argument(
        "--deep-search",
        action="store_true",
        help=(
            "Enable recursive filesystem search for missing videos during "
            "inventory build. Slower but finds videos in unexpected subdirectories."
        ),
    )

    # --- Validation thresholds ---
    parser.add_argument(
        "--min-clips",
        type=int,
        default=_MIN_CLIPS_PER_SIGN,
        metavar="N",
        help=f"Minimum clips per sign for validation Check 2 (default: {_MIN_CLIPS_PER_SIGN})",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=_MIN_FRAMES,
        metavar="N",
        help=f"Minimum frame count for Check 4 (default: {_MIN_FRAMES})",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=_MAX_FRAMES,
        metavar="N",
        help=f"Maximum frame count for Check 4 (default: {_MAX_FRAMES})",
    )
    parser.add_argument(
        "--imbalance-threshold",
        type=float,
        default=_IMBALANCE_THRESHOLD,
        metavar="RATIO",
        help=f"Max/min clip ratio for imbalance Check 8 (default: {_IMBALANCE_THRESHOLD})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_SEED,
        metavar="N",
        help=f"Random seed for split reproducibility (default: {_SEED})",
    )
    parser.add_argument(
        "--split-targets",
        nargs=3,
        type=float,
        default=[0.70, 0.15, 0.15],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Train/val/test split ratios (default: 0.70 0.15 0.15)",
    )

    # --- Verbosity ---
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level console output.",
    )

    return parser


# ---------------------------------------------------------------------------
# Stage functions
# ---------------------------------------------------------------------------

def run_resolve_stage(args: argparse.Namespace, logger) -> "pd.DataFrame":
    """
    Stage 1a — Resolve local WLASL videos against the manifest.

    Loads WLASL_v0.3.json, filters to the 35 selected signs, locates
    videos on disk, and writes raw_inventory.json.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    logger : StructuredAdapter
        Active logger with stage context.

    Returns
    -------
    pd.DataFrame
        The full inventory (found and missing clips).
    """
    import pandas as pd  # local import — pandas is heavy; keep top-level clean

    stage_start = time.time()
    logger.info(
        "=" * 65,
        extra={"stage": "ingestion"},
    )
    logger.info(
        "STAGE 1a — Resolve Inventory",
        extra={"stage": "ingestion"},
    )
    logger.info(
        f"  manifest   : {args.manifest}",
        extra={"stage": "ingestion"},
    )
    logger.info(
        f"  raw_dir    : {args.raw_dir}",
        extra={"stage": "ingestion"},
    )
    logger.info(
        f"  output     : {args.inventory_path}",
        extra={"stage": "ingestion"},
    )
    logger.info("=" * 65, extra={"stage": "ingestion"})

    # Load label map
    label_map = get_label_map(args.label_map)
    logger.info(
        f"Label map loaded | {label_map.num_classes} signs | "
        f"version={label_map.version}",
        extra={"stage": "ingestion"},
    )

    # Build inventory
    resolver = WLASLResolver(
        manifest_path=args.manifest,
        raw_dir=args.raw_dir,
        label_map=label_map,
        deep_search=args.deep_search,
    )

    inventory_df = resolver.build_inventory(
        force=args.force,
        inventory_path=args.inventory_path,
    )

    # Optionally attempt to download missing clips
    if args.download_missing:
        missing_count = (~inventory_df["found"]).sum()
        if missing_count > 0:
            logger.info(
                f"Attempting download of {missing_count} missing clips | "
                f"dry_run={args.dry_run}",
                extra={"stage": "ingestion"},
            )
            download_results = resolver.download_missing(
                max_attempts=3,
                dry_run=args.dry_run,
            )
            succeeded = sum(1 for v in download_results.values() if v)
            logger.info(
                f"Download results: {succeeded}/{len(download_results)} clips retrieved.",
                extra={"stage": "ingestion"},
            )
            # Rebuild inventory to reflect newly downloaded clips
            if succeeded > 0 and not args.dry_run:
                inventory_df = resolver.build_inventory(
                    force=True,
                    inventory_path=args.inventory_path,
                )
        else:
            logger.info(
                "No missing clips — --download-missing is a no-op.",
                extra={"stage": "ingestion"},
            )

    # Save inventory
    resolver.save_inventory(
        output_path=args.inventory_path,
        include_per_sign_summary=True,
    )

    elapsed = time.time() - stage_start
    found = int(inventory_df["found"].sum())
    total = len(inventory_df)
    logger.info(
        f"Stage 1a complete | {found}/{total} clips found | {elapsed:.1f}s",
        extra={"stage": "ingestion"},
    )
    return inventory_df


def run_validation_stage(
    args: argparse.Namespace,
    inventory_df: "pd.DataFrame",
    logger,
) -> "ValidationReport":
    """
    Stage 1b — Run 8 integrity checks on the inventory.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    inventory_df : pd.DataFrame
        Inventory DataFrame from Stage 1a.
    logger : StructuredAdapter
        Active logger with stage context.

    Returns
    -------
    ValidationReport
        The complete validation report with pipeline_can_proceed flag.
    """
    from src.data.validator import DataValidator

    stage_start = time.time()
    logger.info("=" * 65, extra={"stage": "validation"})
    logger.info("STAGE 1b — Validate Inventory", extra={"stage": "validation"})
    logger.info(f"  output: {args.report_path}", extra={"stage": "validation"})
    logger.info("=" * 65, extra={"stage": "validation"})

    # Check resumability
    if not args.force and Path(args.report_path).exists():
        logger.info(
            f"Validation report already exists: {args.report_path}. "
            "Loading cached report. Pass --force to re-validate.",
            extra={"stage": "validation"},
        )
        with open(args.report_path, encoding="utf-8") as f:
            cached = json.load(f)
        can_proceed = cached.get("metadata", {}).get("pipeline_can_proceed", False)

        from src.data.validator import ValidationReport, CheckResult
        # Reconstruct lightweight report object for return
        report = ValidationReport(
            pipeline_can_proceed=can_proceed,
            checks={},
            per_sign_stats=cached.get("per_sign_stats", {}),
            dataset_totals=cached.get("dataset_totals", {}),
            inventory_path=args.inventory_path,
            generated_utc=cached.get("metadata", {}).get("generated_utc", ""),
        )
        logger.info(
            f"Cached report: pipeline_can_proceed={can_proceed}",
            extra={"stage": "validation"},
        )
        return report

    validator = DataValidator(
        inventory_df=inventory_df,
        raw_dir=args.raw_dir,
        min_clips_per_sign=args.min_clips,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
        imbalance_threshold=args.imbalance_threshold,
        num_classes=35,
    )

    report = validator.run_all_checks(inventory_path=args.inventory_path)
    report.save(args.report_path)

    elapsed = time.time() - stage_start
    logger.info(
        f"Stage 1b complete | "
        f"passed={report.n_passed} | "
        f"warned={report.n_warned} | "
        f"failed={report.n_failed} | "
        f"can_proceed={report.pipeline_can_proceed} | "
        f"{elapsed:.1f}s",
        extra={"stage": "validation"},
    )
    return report


def run_split_stage(
    args: argparse.Namespace,
    inventory_df: "pd.DataFrame",
    logger,
) -> "SplitResult":
    """
    Stage 1c — Create signer-aware train/val/test splits.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    inventory_df : pd.DataFrame
        Inventory DataFrame from Stage 1a (all clips, found and missing).
    logger : StructuredAdapter
        Active logger with stage context.

    Returns
    -------
    SplitResult
        Container with the three split DataFrames and summary statistics.
    """
    from src.data.splitter import SignerAwareSplitter

    stage_start = time.time()
    logger.info("=" * 65, extra={"stage": "splitting"})
    logger.info("STAGE 1c — Signer-Aware Split", extra={"stage": "splitting"})
    logger.info(f"  output_dir: {args.splits_dir}", extra={"stage": "splitting"})
    logger.info("=" * 65, extra={"stage": "splitting"})

    train_ratio, val_ratio, test_ratio = args.split_targets
    targets = {"train": train_ratio, "val": val_ratio, "test": test_ratio}

    splitter = SignerAwareSplitter(
        inventory_df=inventory_df,
        splits_dir=args.splits_dir,
        targets=targets,
        seed=args.seed,
        min_clips_per_class_per_split=3,
    )

    result = splitter.split(force=args.force)

    elapsed = time.time() - stage_start
    logger.info(
        f"Stage 1c complete | "
        f"train={len(result.train)} | "
        f"val={len(result.val)} | "
        f"test={len(result.test)} | "
        f"{elapsed:.1f}s",
        extra={"stage": "splitting"},
    )
    return result


# ---------------------------------------------------------------------------
# Inventory loading helper (for --validate-only and --split-only modes)
# ---------------------------------------------------------------------------

def _load_inventory_from_disk(inventory_path: str, logger) -> "pd.DataFrame":
    """
    Load the inventory DataFrame from a saved JSON file.

    Used when --validate-only or --split-only is passed and the resolve
    stage is skipped.

    Parameters
    ----------
    inventory_path : str
        Path to raw_inventory.json.
    logger : StructuredAdapter
        Active logger.

    Returns
    -------
    pd.DataFrame
        Inventory DataFrame reconstructed from JSON.

    Raises
    ------
    SystemExit
        If the inventory file does not exist.
    """
    import pandas as pd

    path = Path(inventory_path)
    if not path.exists():
        logger.error(
            f"Inventory file not found: {inventory_path}. "
            "Run without --validate-only / --split-only first to build the inventory.",
            extra={"stage": "ingestion"},
        )
        sys.exit(2)

    logger.info(
        f"Loading inventory from disk: {inventory_path}",
        extra={"stage": "ingestion"},
    )
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    clips = data.get("clips", [])
    if not clips:
        logger.error(
            f"Inventory at {inventory_path} contains no clips. "
            "The file may be empty or corrupt.",
            extra={"stage": "ingestion"},
        )
        sys.exit(2)

    df = pd.DataFrame(clips)
    logger.info(
        f"Inventory loaded | clips={len(df)} | "
        f"found={df['found'].sum() if 'found' in df.columns else '?'}",
        extra={"stage": "ingestion"},
    )
    return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """
    Execute Stage 1 pipeline stages according to CLI arguments.

    Returns
    -------
    int
        Exit code: 0 = success, 1 = validation blocked, 2 = unexpected error.
    """
    parser = _build_parser()
    args = parser.parse_args()

    # ----------------------------------------------------------------
    # Logging — must be configured before ANY other operation
    # ----------------------------------------------------------------
    log_level = "DEBUG" if args.verbose else "INFO"
    log_file = configure_logging(
        log_dir=args.log_dir,
        run_name="stage1_preprocessing",
        level=log_level,
        file_level="DEBUG",
    )
    logger = get_logger(__name__, stage="pipeline")

    logger.info(
        "WLASL Gesture Recognition — Stage 1: Data Ingestion & Validation",
        extra={"stage": "pipeline"},
    )
    logger.info(
        f"Log file: {log_file}",
        extra={"stage": "pipeline"},
    )

    # ----------------------------------------------------------------
    # Reproducibility — seed before any data operations
    # ----------------------------------------------------------------
    set_seeds(args.seed)

    # ----------------------------------------------------------------
    # Validate CLI arguments
    # ----------------------------------------------------------------
    try:
        train_r, val_r, test_r = args.split_targets
        if abs(train_r + val_r + test_r - 1.0) > 1e-6:
            logger.error(
                f"--split-targets must sum to 1.0. Got: {train_r}+{val_r}+{test_r}"
                f"={train_r + val_r + test_r:.6f}",
                extra={"stage": "pipeline"},
            )
            return 2
    except Exception as exc:
        logger.error(
            f"Invalid --split-targets: {exc}",
            extra={"stage": "pipeline"},
        )
        return 2

    pipeline_start = time.time()
    inventory_df = None

    try:
        # ----------------------------------------------------------------
        # Stage 1a — Resolve inventory
        # ----------------------------------------------------------------
        if not args.validate_only and not args.split_only:
            inventory_df = run_resolve_stage(args, logger)
        else:
            # Load from disk for skipped resolve stage
            inventory_df = _load_inventory_from_disk(args.inventory_path, logger)

        # ----------------------------------------------------------------
        # Stage 1b — Validate
        # ----------------------------------------------------------------
        if not args.split_only:
            report = run_validation_stage(args, inventory_df, logger)

            if not report.pipeline_can_proceed:
                logger.error(
                    f"Pipeline BLOCKED. {report.n_failed} ERROR-severity "
                    f"check(s) failed. See: {args.report_path}",
                    extra={"stage": "pipeline"},
                )
                logger.error(
                    f"Blocking checks: {report.blocking_checks}",
                    extra={"stage": "pipeline"},
                )
                return 1
        else:
            # Split-only mode: verify existing report allows proceeding
            report_path = Path(args.report_path)
            if report_path.exists():
                with open(report_path, encoding="utf-8") as f:
                    cached_report = json.load(f)
                can_proceed = cached_report.get("metadata", {}).get(
                    "pipeline_can_proceed", True
                )
                if not can_proceed:
                    logger.error(
                        "Existing validation report shows pipeline_can_proceed=False. "
                        "Resolve validation errors before splitting.",
                        extra={"stage": "pipeline"},
                    )
                    return 1
                logger.info(
                    "Validation report: pipeline_can_proceed=True. Proceeding to split.",
                    extra={"stage": "pipeline"},
                )
            else:
                logger.warning(
                    f"Validation report not found at {args.report_path}. "
                    "Proceeding to split without validation — not recommended.",
                    extra={"stage": "pipeline"},
                )

        # ----------------------------------------------------------------
        # Stage 1c — Split
        # ----------------------------------------------------------------
        if not args.validate_only:
            split_result = run_split_stage(args, inventory_df, logger)

            # Final integrity check — zero signer overlap confirmed by splitter,
            # but re-verify here for belt-and-suspenders logging.
            train_signers = set(split_result.train["signer_id"].unique())
            val_signers   = set(split_result.val["signer_id"].unique())
            test_signers  = set(split_result.test["signer_id"].unique())
            assert len(train_signers & val_signers) == 0, "Train/val signer overlap!"
            assert len(train_signers & test_signers) == 0, "Train/test signer overlap!"
            assert len(val_signers & test_signers) == 0, "Val/test signer overlap!"
            logger.info(
                "✓ Final signer overlap verification passed.",
                extra={"stage": "pipeline"},
            )

    except SystemExit:
        raise

    except AssertionError as exc:
        logger.error(
            f"Integrity assertion failed: {exc}",
            extra={"stage": "pipeline"},
        )
        return 2

    except KeyboardInterrupt:
        logger.warning(
            "Pipeline interrupted by user (KeyboardInterrupt).",
            extra={"stage": "pipeline"},
        )
        return 2

    except Exception as exc:
        logger.error(
            f"Unexpected error terminated the pipeline: {type(exc).__name__}: {exc}",
            extra={"stage": "pipeline"},
        )
        import traceback
        logger.debug(traceback.format_exc(), extra={"stage": "pipeline"})
        return 2

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    elapsed = time.time() - pipeline_start
    logger.info("=" * 65, extra={"stage": "pipeline"})
    logger.info("STAGE 1 COMPLETE", extra={"stage": "pipeline"})
    logger.info(f"  Total elapsed: {elapsed:.1f}s", extra={"stage": "pipeline"})
    logger.info(
        f"  Artifacts produced:",
        extra={"stage": "pipeline"},
    )
    for artifact in [
        args.inventory_path,
        args.report_path,
        str(Path(args.splits_dir) / "train.csv"),
        str(Path(args.splits_dir) / "val.csv"),
        str(Path(args.splits_dir) / "test.csv"),
        str(Path(args.splits_dir) / "split_summary.json"),
    ]:
        p = Path(artifact)
        exists = "✓" if p.exists() else "✗"
        size = f"({p.stat().st_size / 1024:.1f} KB)" if p.exists() else "(not found)"
        logger.info(
            f"  {exists} {artifact} {size}",
            extra={"stage": "pipeline"},
        )
    logger.info("=" * 65, extra={"stage": "pipeline"})
    logger.info(
        "Next step: Stage 2 — Run notebooks/02_landmark_inspection.ipynb "
        "after completing Stage 3 preprocessing on a sample.",
        extra={"stage": "pipeline"},
    )
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(main())