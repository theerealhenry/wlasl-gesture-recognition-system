"""
src/data/
=========
Data ingestion, validation, and splitting for the WLASL gesture recognition pipeline.

This package handles everything between raw video files on disk and the clean,
validated, split-aware DataFrames consumed by the preprocessing pipeline.

Module responsibilities
-----------------------
downloader.py
    Reads WLASL_v0.3.json, filters to the 35 selected signs, locates videos in
    the local filesystem, and produces a structured inventory manifest. Includes
    an optional yt-dlp fallback for any clips that are genuinely missing.

    Design note: the dataset has already been downloaded. This module operates
    in "resolver" mode by default — it locates and inventories existing files
    rather than fetching from the internet. The download path is retained for
    completeness and future reuse.

validator.py
    Runs 8 integrity checks against the resolved inventory and produces a
    machine-readable validation report (data/data_validation_report.json).
    Sets a pipeline_can_proceed flag that downstream scripts must check before
    proceeding. Any ERROR-severity failure blocks the pipeline.

splitter.py
    Implements a signer-aware train/val/test split. All clips from any given
    signer are assigned exclusively to one split — a signer never appears in
    two splits. This is the methodologically correct approach for evaluating
    generalisation to unseen signers, and the only honest test of model quality.

Typical usage
-------------
    # Via the pipeline entry point (recommended):
    python pipelines/run_preprocessing.py --manifest data/raw/WLASL_v0.3.json

    # Programmatic usage:
    from src.data.downloader import WLASLResolver
    from src.data.validator import DataValidator
    from src.data.splitter import SignerAwareSplitter

    resolver  = WLASLResolver(manifest_path, raw_dir, label_map)
    inventory = resolver.build_inventory()

    validator = DataValidator(inventory, raw_dir)
    report    = validator.run_all_checks()

    if report.pipeline_can_proceed:
        splitter = SignerAwareSplitter(inventory, splits_dir, seed=42)
        splits   = splitter.split()

Pipeline contract
-----------------
Every public function in this package uses get_logger(__name__) for all output.
print() is never used. All file I/O is explicit — paths are resolved relative to
the repository root, never assumed from the working directory.
"""

from src.data.downloader import WLASLResolver
from src.data.validator import DataValidator, ValidationReport
from src.data.splitter import SignerAwareSplitter, SplitResult

__all__ = [
    "WLASLResolver",
    "DataValidator",
    "ValidationReport",
    "SignerAwareSplitter",
    "SplitResult",
]