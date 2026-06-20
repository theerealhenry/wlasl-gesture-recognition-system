"""
pipelines/run_export_verification.py
======================================
Stage 8 — TFLite Export and Verification Pipeline.

This is the authoritative CLI entry point for Stage 8. It orchestrates the
complete chain:

  1.  Pre-flight checks (SavedModel, config snapshot, label map, dataset)
  2.  TFLite export via src/export/convert.py::export_champion()
  3.  Accuracy verification (Keras vs TFLite on val + test splits)
  4.  Per-class F1 delta analysis (reuses Step 3 predictions — no re-inference)
  5.  Production latency benchmarking (optional: --skip-benchmark / --dry-run)
  6.  Release gate evaluation
  7.  Verification report + model metadata authoring
  8.  Figure generation (size comparison, accuracy comparison, per-class delta)
  9.  Concise release summary printed to stdout + structured exit code

Exit codes
-----------
  0 — PASS: all hard gate criteria met; gesture_bilstm_v1.tflite is release-ready
  1 — FAIL: one or more hard gate criteria failed; see gate report for details
  2 — ERROR: pipeline aborted due to a configuration or infrastructure failure

Usage
------
Default (champion run, all defaults):
    python pipelines/run_export_verification.py

Custom paths:
    python pipelines/run_export_verification.py \\
        --config-snapshot artifacts/experiments/bilstm_hands_only_v4_aug/config_snapshot.yaml \\
        --saved-model models/bilstm_hands_only_v4_aug_saved_model \\
        --output models/gesture_bilstm_v1.tflite \\
        --n-calls 200 \\
        --warmup 20 \\
        --figures-dir reports/figures \\
        --stage6-report reports/evaluation/evaluation_report.json

Dry-run (skip latency benchmark, fast end-to-end sanity check):
    python pipelines/run_export_verification.py --dry-run

Skip export (re-verify an already-exported .tflite):
    python pipelines/run_export_verification.py --skip-export

Design principles
------------------
  - Single-pass inference: run_accuracy_verification() runs inference exactly
    once per split. All downstream consumers (per-class delta, release gate,
    metadata, figures) reuse the cached ``_predictions`` arrays.
  - GestureDataset is constructed exactly once and reused across all steps.
  - Every intermediate result is persisted to disk before the next step so a
    partial run can be debugged without restarting from scratch.
  - The release gate is NEVER marked as passing on a stub/skipped benchmark;
    instead, latency-related gate criteria are explicitly marked as "skipped"
    (neither pass nor fail) when --skip-benchmark is active.
  - Structured logging goes to logs/<timestamp>_stage8_export.log alongside
    the console output.
  - Non-zero exit on gate failure so CI/CD pipelines treat a failed gate as a
    build failure.
  - Gate is always assembled and summary always printed, even on partial
    failure — the summary is the primary debugging tool.

Champion model reference
--------------------------
  SavedModel:      models/bilstm_hands_only_v4_aug_saved_model/
  Config snapshot: artifacts/experiments/bilstm_hands_only_v4_aug/config_snapshot.yaml
  TFLite output:   models/gesture_bilstm_v1.tflite
  MLflow run ID:   cb16f689d2294001a2ff2d3e02419d27
  val macro-F1:    0.6011  (Stage 6 reference)
  test macro-F1:   0.4581  (Stage 6 reference)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Repository root — ensure all src.* imports resolve regardless of cwd.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Project imports (lightweight only at module scope — TF/MediaPipe deferred)
# ---------------------------------------------------------------------------
from src.utils.logger import configure_logging, get_logger

# ---------------------------------------------------------------------------
# Module-level constants — single source of truth for paths + thresholds
# ---------------------------------------------------------------------------

_CHAMPION_CONFIG_SNAPSHOT: str = (
    "artifacts/experiments/bilstm_hands_only_v4_aug/config_snapshot.yaml"
)
_CHAMPION_SAVED_MODEL: str = "models/bilstm_hands_only_v4_aug_saved_model"
_CHAMPION_TFLITE_OUTPUT: str = "models/gesture_bilstm_v1.tflite"
_STAGE6_REPORT_DEFAULT: str = "reports/evaluation/evaluation_report.json"
_FIGURES_DIR_DEFAULT: str = "reports/figures"
_VERIFICATION_REPORT_DEFAULT: str = "reports/evaluation/tflite_verification_report.json"
_METADATA_OUTPUT_DEFAULT: str = "models/gesture_model_metadata.json"
_EXPORT_MANIFEST_DEFAULT: str = "models/export_manifest.json"

_DEFAULT_N_CALLS: int = 200
_DEFAULT_WARMUP: int = 20

# Gate thresholds — must mirror verify.py constants exactly.
_DELTA_THRESHOLD: float = 0.03
_AGREEMENT_THRESHOLD: float = 0.95
_MAX_TFLITE_SIZE_MB: float = 10.0
_LATENCY_TARGET_MS: float = 100.0

# Console formatting
_SEP_WIDTH: int = 62

# Sentinel value for "benchmark was skipped" — distinguishes from 0.0 ms
_SKIPPED_MS: float = float("nan")


# ===========================================================================
# Argument parsing
# ===========================================================================

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_export_verification.py",
        description=(
            "Stage 8 — TFLite Export and Verification Pipeline for the WLASL "
            "35-class gesture recognition champion model."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Paths ───────────────────────────────────────────────────────────
    path_grp = parser.add_argument_group("paths")
    path_grp.add_argument(
        "--config-snapshot",
        default=_CHAMPION_CONFIG_SNAPSHOT,
        help=f"Path to config_snapshot.yaml. Default: {_CHAMPION_CONFIG_SNAPSHOT}",
    )
    path_grp.add_argument(
        "--saved-model",
        default=_CHAMPION_SAVED_MODEL,
        help=f"Path to Keras SavedModel directory. Default: {_CHAMPION_SAVED_MODEL}",
    )
    path_grp.add_argument(
        "--output",
        default=_CHAMPION_TFLITE_OUTPUT,
        help=f"Destination .tflite path. Default: {_CHAMPION_TFLITE_OUTPUT}",
    )
    path_grp.add_argument(
        "--stage6-report",
        default=_STAGE6_REPORT_DEFAULT,
        help=(
            "Path to Stage 6 evaluation_report.json (calibration metrics + CI bounds). "
            "Falls back to hardcoded Stage 6 constants if absent. "
            f"Default: {_STAGE6_REPORT_DEFAULT}"
        ),
    )
    path_grp.add_argument(
        "--figures-dir",
        default=_FIGURES_DIR_DEFAULT,
        help=f"Directory for Stage 8 figures. Default: {_FIGURES_DIR_DEFAULT}",
    )
    path_grp.add_argument(
        "--verification-report",
        default=_VERIFICATION_REPORT_DEFAULT,
        help=f"Output path for tflite_verification_report.json. Default: {_VERIFICATION_REPORT_DEFAULT}",
    )
    path_grp.add_argument(
        "--metadata-output",
        default=_METADATA_OUTPUT_DEFAULT,
        help=f"Output path for gesture_model_metadata.json. Default: {_METADATA_OUTPUT_DEFAULT}",
    )
    path_grp.add_argument(
        "--export-manifest",
        default=_EXPORT_MANIFEST_DEFAULT,
        help=f"Output path for export_manifest.json. Default: {_EXPORT_MANIFEST_DEFAULT}",
    )
    path_grp.add_argument(
        "--log-dir",
        default="logs",
        help="Directory for log files. Default: logs",
    )
    path_grp.add_argument(
        "--label-map",
        default="artifacts/label_map_v1.json",
        help="Path to label_map_v1.json. Default: artifacts/label_map_v1.json",
    )

    # ── Benchmark parameters ─────────────────────────────────────────────
    bench_grp = parser.add_argument_group("benchmark")
    bench_grp.add_argument(
        "--n-calls",
        type=int,
        default=_DEFAULT_N_CALLS,
        help=f"Number of timed inference calls. Default: {_DEFAULT_N_CALLS}",
    )
    bench_grp.add_argument(
        "--warmup",
        type=int,
        default=_DEFAULT_WARMUP,
        help=f"Warmup calls discarded before timing. Default: {_DEFAULT_WARMUP}",
    )

    # ── Control flags ────────────────────────────────────────────────────
    ctrl_grp = parser.add_argument_group("control")
    ctrl_grp.add_argument(
        "--skip-export",
        action="store_true",
        help=(
            "Skip the TFLite export step. Requires the .tflite file to already "
            "exist at --output."
        ),
    )
    ctrl_grp.add_argument(
        "--skip-benchmark",
        action="store_true",
        help=(
            "Skip the latency benchmark. Gate latency criteria will be marked "
            "SKIPPED (not PASS or FAIL) in the release summary."
        ),
    )
    ctrl_grp.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Run with n_calls=5, warmup=2 for all benchmarks. Useful for rapid "
            "end-to-end validation without waiting for the full 200-call benchmark. "
            "Latency numbers from a dry run are NOT suitable for the gate."
        ),
    )
    ctrl_grp.add_argument(
        "--no-quantise",
        action="store_true",
        help="Export an unquantised float32 .tflite (diagnostic only).",
    )
    ctrl_grp.add_argument(
        "--allow-non-champion-params",
        action="store_true",
        help=(
            "Log a warning instead of raising on champion-shape / wrong-param-count "
            "mismatch. Use only when intentionally exporting a non-champion run."
        ),
    )
    ctrl_grp.add_argument(
        "--strict-architecture-check",
        action="store_true",
        help=(
            "Hard-fail if layer architecture doesn't match the champion's "
            "(Masking + 2× Bidirectional). Default: warn only."
        ),
    )
    ctrl_grp.add_argument(
        "--skip-sanity-check",
        action="store_true",
        help="Skip the post-conversion TFLite interpreter sanity inference pass.",
    )
    ctrl_grp.add_argument(
        "--build-representative-dataset",
        action="store_true",
        help=(
            "Force-build the representative dataset generator. Only relevant if "
            "experimenting with full-integer quantisation."
        ),
    )
    ctrl_grp.add_argument(
        "--n-classes",
        type=int,
        default=35,
        help="Number of output classes. Default: 35",
    )
    ctrl_grp.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed. Default: 42",
    )

    return parser


# ===========================================================================
# Pretty-printing helpers
# ===========================================================================

def _sep(char: str = "=") -> str:
    return char * _SEP_WIDTH


def _header(title: str) -> str:
    return "\n" + _sep() + f"\n  {title}\n" + _sep()


def _step(n: int, total: int, description: str) -> str:
    return f"\n[{n}/{total}] {description}"


def _ok(msg: str) -> str:
    return f"  ✓  {msg}"


def _warn(msg: str) -> str:
    return f"  ⚠  {msg}"


def _fail(msg: str) -> str:
    return f"  ✗  {msg}"


def _skip_msg(msg: str) -> str:
    return f"  -  {msg}"


def _kv(key: str, value: Any, width: int = 28) -> str:
    return f"  {key:<{width}}: {value}"


def _is_nan(v: Any) -> bool:
    """Safe NaN check that works for Python floats and numpy scalars."""
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return False


def _fmt_mb(v: Any) -> str:
    if v is None or _is_nan(v):
        return "N/A"
    return f"{float(v):.4f} MB"


def _fmt_ms(v: Any) -> str:
    if v is None or _is_nan(v):
        return "N/A"
    return f"{float(v):.2f} ms"


def _fmt_f1(v: Any) -> str:
    if v is None or _is_nan(v):
        return "N/A"
    return f"{float(v):.4f}"


def _fmt_pct(v: Any) -> str:
    if v is None or _is_nan(v):
        return "N/A"
    return f"{float(v):.4f}"


def _fmt_delta(v: Any, threshold: float = _DELTA_THRESHOLD) -> str:
    if v is None or _is_nan(v):
        return "N/A (not measured)"
    fv = float(v)
    flag = "✓" if abs(fv) <= threshold else "✗"
    return f"{fv:+.4f}  [{flag}]"


def _fmt_agreement(v: Any, threshold: float = _AGREEMENT_THRESHOLD) -> str:
    if v is None or _is_nan(v):
        return "N/A (not measured)"
    fv = float(v)
    flag = "✓" if fv >= threshold else "✗"
    return f"{fv:.4f}  [{flag}]"


# ===========================================================================
# Pre-flight checks
# ===========================================================================

def _preflight(args: argparse.Namespace, logger: Any) -> None:
    """
    Validate that all required inputs exist before doing any heavy work.
    Raises SystemExit(2) with clear, actionable messages if anything is missing.
    """
    errors: List[str] = []

    config_snap = Path(args.config_snapshot)
    if not config_snap.exists():
        errors.append(
            f"Config snapshot not found: {config_snap}\n"
            "  → Run Stage 5 training; setup_experiment() writes the snapshot "
            "to artifacts/experiments/<run_name>/config_snapshot.yaml"
        )

    label_map = Path(args.label_map)
    if not label_map.exists():
        errors.append(
            f"Label map not found: {label_map}\n"
            "  → Verify artifacts/label_map_v1.json exists (created in Stage 1)"
        )

    if not args.skip_export:
        saved_model = Path(args.saved_model)
        if not saved_model.exists():
            errors.append(
                f"Keras SavedModel not found: {saved_model}\n"
                "  → Run Stage 5 training to generate the SavedModel"
            )
        else:
            pb = saved_model / "saved_model.pb"
            if not pb.exists():
                errors.append(
                    f"SavedModel is missing saved_model.pb: {saved_model}\n"
                    "  → The SavedModel appears incomplete — re-run Stage 5 training"
                )
    else:
        # --skip-export: the .tflite file must already be present
        tflite_p = Path(args.output)
        if not tflite_p.exists():
            errors.append(
                f"--skip-export is set but .tflite file not found: {tflite_p}\n"
                "  → Remove --skip-export to run the export, or supply the "
                "correct --output path to an existing .tflite file"
            )

    # Verify the benchmark parameters are sane (catches common mis-inversions)
    if args.n_calls <= args.warmup:
        errors.append(
            f"--n-calls ({args.n_calls}) must be greater than --warmup ({args.warmup}).\n"
            f"  → Use the project defaults: --n-calls {_DEFAULT_N_CALLS} --warmup {_DEFAULT_WARMUP}"
        )

    if errors:
        print(_header("PRE-FLIGHT CHECK FAILED"), file=sys.stderr)
        for err in errors:
            print(_fail(err), file=sys.stderr)
        print(file=sys.stderr)
        sys.exit(2)

    logger.info(
        "Pre-flight checks passed | config_snapshot=%s | saved_model=%s | "
        "skip_export=%s | skip_benchmark=%s | dry_run=%s",
        args.config_snapshot,
        args.saved_model if not args.skip_export else "(skipped)",
        args.skip_export,
        args.skip_benchmark,
        args.dry_run,
    )


# ===========================================================================
# Step 1 — TFLite export
# ===========================================================================

def _step1_export(
    args: argparse.Namespace,
    logger: Any,
) -> Dict[str, Any]:
    """
    Export the champion SavedModel to a verified .tflite file.

    Calls export_champion() which internally:
      1. Loads + verifies the SavedModel identity (param count, input/output
         shapes, config hash, layer architecture).
      2. Runs dynamic-range quantisation with SELECT_TF_OPS (required for
         BiLSTM TensorListReserve ops in TF 2.13).
      3. Optionally runs a post-conversion sanity inference pass.
      4. Computes and returns a SHA-256 checksum of the .tflite file.

    Returns the full conversion result dict for downstream consumption.
    """
    from src.export.convert import export_champion, write_export_manifest

    logger.info(
        "Starting TFLite export | saved_model=%s | output=%s | quantise=%s",
        args.saved_model,
        args.output,
        not args.no_quantise,
    )
    t0 = time.perf_counter()

    # Ensure output directory exists before export
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    export_result = export_champion(
        config_snapshot_path=args.config_snapshot,
        saved_model_path=args.saved_model,
        output_path=args.output,
        build_representative_dataset=args.build_representative_dataset,
        quantise=not args.no_quantise,
        verify_model=True,
        run_sanity_inference=not args.skip_sanity_check,
        strict_champion_param_check=not args.allow_non_champion_params,
        strict_architecture_check=args.strict_architecture_check,
    )

    elapsed = time.perf_counter() - t0
    logger.info(
        "TFLite export complete | tflite_size=%.4f MB | conversion_time=%.2fs | "
        "used_select_tf_ops=%s | sha256=%s...",
        export_result.get("tflite_disk_mb", float("nan")),
        export_result.get("conversion_time_s", elapsed),
        export_result.get("used_select_tf_ops", "unknown"),
        str(export_result.get("sha256_checksum", ""))[:16],
    )

    # Write export manifest immediately — persisted before any later step can fail
    try:
        manifest_path = Path(args.export_manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        written = write_export_manifest(
            export_result,
            output_dir=manifest_path.parent,
            filename=manifest_path.name,
        )
        logger.info("Export manifest written → %s", written)
    except Exception as exc:
        # Manifest write failure is non-fatal — log and continue
        logger.warning(
            "Failed to write export manifest (non-fatal): %s: %s",
            type(exc).__name__, exc,
        )

    return export_result


# ===========================================================================
# Step 2 — GestureDataset + FeaturePipeline construction
# ===========================================================================

def _step2_build_dataset_and_pipeline(
    args: argparse.Namespace,
    logger: Any,
) -> Tuple[Any, Any]:
    """
    Reconstruct the FeaturePipeline and GestureDataset from the config snapshot.

    Using from_config_snapshot() guarantees that data.landmark_config=hands_only
    (and every other override applied during Stage 5 training) is correctly
    reproduced — this is the only path that avoids the risk of silently
    rebuilding the wrong preprocessing configuration.

    Returns (pipeline, dataset).
    """
    from src.export.convert import load_config_snapshot
    from src.features.dataset import GestureDataset
    from src.features.pipeline import FeaturePipeline

    logger.info("Loading config from snapshot: %s", args.config_snapshot)
    config = load_config_snapshot(args.config_snapshot)

    logger.info(
        "Building FeaturePipeline | seq_len=%d | landmark_config=%s | feature_dim=%s",
        config.data.sequence_length,
        config.data.landmark_config,
        getattr(config.data, "feature_dim", "derived"),
    )
    pipeline = FeaturePipeline(config)
    logger.info(
        "FeaturePipeline ready | output_shape=%s | feature_dim=%d",
        pipeline.output_shape,
        pipeline.feature_dim,
    )

    logger.info(
        "Constructing GestureDataset | splits_dir=%s | landmarks_dir=%s",
        config.data.splits_dir,
        config.data.landmark_dir,
    )
    dataset = GestureDataset(
        config,
        pipeline,
        splits_dir=config.data.splits_dir,
        landmarks_dir=config.data.landmark_dir,
    )
    logger.info(
        "GestureDataset ready | n_train=%d | n_val=%d | n_test=%d",
        dataset.n_train,
        dataset.n_val,
        dataset.n_test,
    )

    # Sanity-check: val and test must be non-empty
    if dataset.n_val == 0:
        raise RuntimeError(
            "GestureDataset has zero validation clips. "
            "Check data/splits/val.csv and data/landmarks/ are present. "
            "Expected n_val=52 for the champion WLASL-35 configuration."
        )
    if dataset.n_test == 0:
        logger.warning(
            "GestureDataset has zero test clips — test-set verification will be "
            "skipped. Expected n_test=51 for the champion configuration."
        )

    return pipeline, dataset


# ===========================================================================
# Step 3 — Accuracy verification
# ===========================================================================

def _step3_accuracy_verification(
    args: argparse.Namespace,
    logger: Any,
    dataset: Any,
) -> Dict[str, Any]:
    """
    Compare Keras SavedModel vs TFLite interpreter accuracy on val + test splits.

    Both models are loaded through GesturePredictor.from_config_snapshot()
    with smoother_window=1 (disables majority voting for a clean per-clip
    argmax comparison). Inference is run exactly ONCE per split per model.

    The returned dict MUST contain ``_predictions`` — a sub-dict keyed by
    split name containing the raw (y_true, y_pred_keras, y_pred_tflite,
    y_prob_keras, y_prob_tflite) arrays. Step 4 (per-class delta) reuses
    these arrays to avoid a second full inference pass.

    If ``_predictions`` is absent from the returned dict (e.g. due to an
    older implementation of run_accuracy_verification), Step 4 is gracefully
    skipped with a logged warning rather than crashing.
    """
    from src.export.verify import run_accuracy_verification

    logger.info(
        "Running accuracy verification | keras=%s | tflite=%s | n_classes=%d",
        args.saved_model,
        args.output,
        args.n_classes,
    )

    verification_result = run_accuracy_verification(
        keras_model_path=args.saved_model,
        tflite_path=args.output,
        config_snapshot_path=args.config_snapshot,
        val_dataset=dataset,
        n_classes=args.n_classes,
    )

    # Validate expected structure — surface helpful errors early
    for split in ("val", "test"):
        if split not in verification_result:
            logger.warning(
                "run_accuracy_verification() returned no '%s' key — "
                "gate criteria for this split will be unavailable.",
                split,
            )
            continue
        sr = verification_result[split]
        logger.info(
            "[%s] keras_macro_f1=%.4f | tflite_macro_f1=%.4f | "
            "delta=%+.4f | argmax_agreement=%.4f | confidence_shift=%+.4f | "
            "n_samples=%d | n_disagreements=%d",
            split,
            sr.get("keras_macro_f1", float("nan")),
            sr.get("tflite_macro_f1", float("nan")),
            sr.get("delta_macro_f1", float("nan")),
            sr.get("argmax_agreement", float("nan")),
            sr.get("confidence_shift", float("nan")),
            sr.get("n_samples", 0),
            sr.get("n_disagreements", -1),
        )

    # Log presence/absence of _predictions for downstream step awareness
    if "_predictions" not in verification_result:
        logger.warning(
            "run_accuracy_verification() did not return '_predictions'. "
            "Per-class delta analysis (Step 4) will be skipped to avoid "
            "re-running inference. Ensure the verify.py implementation "
            "includes the B1/B8 fix that caches prediction arrays."
        )

    return verification_result


# ===========================================================================
# Step 4 — Per-class F1 delta analysis (zero re-inference)
# ===========================================================================

def _step4_per_class_delta(
    args: argparse.Namespace,
    logger: Any,
    verification_result: Dict[str, Any],
    dataset: Any,
) -> List[Dict[str, Any]]:
    """
    Compute per-class F1 delta between Keras and TFLite on the val split.

    Reuses prediction arrays cached in verification_result["_predictions"]["val"]
    to guarantee zero re-inference (B1/B8 fix). If ``_predictions`` is absent,
    falls back to fetching sign names from the dataset label map and returning
    an empty delta list, logging a warning that Step 4 was skipped.

    Returns list of per-class delta rows (may be empty on fallback).
    """
    from src.export.verify import compute_per_class_tflite_delta

    # --- Guard: _predictions must be present ---
    if "_predictions" not in verification_result:
        logger.warning(
            "Step 4 SKIPPED — '_predictions' not in verification_result. "
            "This means per-class delta figures and the LIMITATIONS.md annotation "
            "of confusable-pair quantisation sensitivity will be unavailable. "
            "Fix: ensure run_accuracy_verification() caches prediction arrays."
        )
        return []

    val_preds = verification_result["_predictions"].get("val")
    if val_preds is None:
        logger.warning(
            "Step 4 SKIPPED — verification_result['_predictions']['val'] is None."
        )
        return []

    # --- Resolve sign names from verification_result or dataset label map ---
    sign_names: Optional[List[str]] = verification_result.get("sign_names")
    if not sign_names:
        logger.info(
            "sign_names absent from verification_result — "
            "deriving from dataset.label_map"
        )
        try:
            lm = dataset.label_map
            sign_names = [
                lm.get_name_safe(i, f"class_{i}")
                for i in range(args.n_classes)
            ]
        except AttributeError:
            # dataset may not expose label_map directly; try the LabelMap util
            try:
                from src.utils.label_map import get_label_map
                lm = get_label_map(args.label_map)
                sign_names = [
                    lm.get_name_safe(i, f"class_{i}")
                    for i in range(args.n_classes)
                ]
            except Exception as exc:
                logger.error(
                    "Could not resolve sign names for per-class delta: %s. "
                    "Step 4 skipped.", exc
                )
                return []

    required_pred_keys = ("y_true", "y_pred_keras", "y_pred_tflite")
    missing_keys = [k for k in required_pred_keys if k not in val_preds]
    if missing_keys:
        logger.warning(
            "Step 4 SKIPPED — prediction cache is missing required keys: %s. "
            "Expected keys: %s",
            missing_keys,
            list(required_pred_keys),
        )
        return []

    logger.info(
        "Computing per-class TFLite delta | n_classes=%d | n_val_samples=%d",
        args.n_classes,
        len(val_preds["y_true"]),
    )

    per_class_delta = compute_per_class_tflite_delta(
        y_true=val_preds["y_true"],
        y_pred_keras=val_preds["y_pred_keras"],
        y_pred_tflite=val_preds["y_pred_tflite"],
        sign_names=sign_names,
        n_classes=args.n_classes,
    )

    n_meaningful = sum(1 for r in per_class_delta if r.get("meaningful_degradation"))
    n_confusable_affected = sum(
        1 for r in per_class_delta
        if r.get("is_confusable_pair") and abs(r.get("f1_delta", 0.0)) > 1e-6
    )
    n_high_risk_affected = sum(
        1 for r in per_class_delta
        if r.get("is_high_risk") and abs(r.get("f1_delta", 0.0)) > 1e-6
    )

    logger.info(
        "Per-class delta complete | %d/%d meaningful degradations | "
        "%d confusable-pair classes with non-zero delta | "
        "%d high-risk classes with non-zero delta",
        n_meaningful,
        args.n_classes,
        n_confusable_affected,
        n_high_risk_affected,
    )

    return per_class_delta


# ===========================================================================
# Step 5 — Production latency benchmarking
# ===========================================================================

def _make_stub_latency_result() -> Dict[str, Any]:
    """
    Return a stub latency result when --skip-benchmark is active.

    All numeric values are NaN so _is_nan() correctly identifies them as
    "not measured" throughout the pipeline. The gate reads
    ``meets_100ms_target`` — we deliberately set it to None (not False) here
    so assemble_release_gate() can distinguish "benchmark skipped" from
    "benchmark ran and failed". The gate implementation must handle None
    for this field.
    """
    _nan = float("nan")
    _stub_stats = {
        "median_ms":  _nan,
        "mean_ms":    _nan,
        "p95_ms":     _nan,
        "p99_ms":     _nan,
        "min_ms":     _nan,
        "max_ms":     _nan,
        "std_ms":     _nan,
        "fps":        _nan,
        "n_calls":    0,
        "warmup":     0,
        "description": "SKIPPED",
    }
    return {
        "tflite":                    dict(_stub_stats, description="tflite_SKIPPED"),
        "keras":                     dict(_stub_stats, description="keras_SKIPPED"),
        "pipeline":                  dict(_stub_stats, description="pipeline_SKIPPED"),
        "full_pipeline_ms":          _nan,
        "meets_100ms_target":        None,   # None = "not measured", not False = "failed"
        "speedup_keras_vs_tflite_x": None,
        "_stub":                     True,
        "_stub_reason":              "benchmark skipped (--skip-benchmark or --dry-run)",
    }


def _step5_latency_benchmark(
    args: argparse.Namespace,
    logger: Any,
    pipeline: Any,
) -> Dict[str, Any]:
    """
    Benchmark the production TFLite file, Keras SavedModel, and FeaturePipeline.

    Applies --dry-run overrides (n_calls=5, warmup=2) if requested.
    On failure, logs the error and returns a stub result rather than crashing
    the entire pipeline — latency is important but should not prevent the
    accuracy gate from being reported.
    """
    from src.export.verify import run_production_latency_benchmark

    n_calls = 5 if args.dry_run else args.n_calls
    warmup  = 2 if args.dry_run else args.warmup

    if args.dry_run:
        logger.info(
            "Dry-run mode: benchmark n_calls overridden to %d, warmup to %d. "
            "Latency results from a dry run are indicative only and NOT "
            "suitable for the production release gate.",
            n_calls,
            warmup,
        )

    logger.info(
        "Running production latency benchmark | "
        "n_calls=%d | warmup=%d | tflite=%s | saved_model=%s",
        n_calls,
        warmup,
        args.output,
        args.saved_model,
    )

    try:
        latency_result = run_production_latency_benchmark(
            tflite_path=args.output,
            keras_model_path=args.saved_model,
            pipeline=pipeline,
            n_calls=n_calls,
            warmup=warmup,
        )
    except Exception as exc:
        logger.error(
            "Latency benchmark raised an exception — returning stub result "
            "so gate can still be assembled. Exception: %s: %s",
            type(exc).__name__,
            exc,
        )
        stub = _make_stub_latency_result()
        stub["_stub_reason"] = f"benchmark FAILED: {type(exc).__name__}: {exc}"
        stub["meets_100ms_target"] = False  # failed benchmark → conservative fail
        return stub

    # Log a concise summary
    t_stats  = latency_result.get("tflite",   {})
    p_stats  = latency_result.get("pipeline", {})
    k_stats  = latency_result.get("keras",    {})
    full_ms  = latency_result.get("full_pipeline_ms", float("nan"))
    meets    = latency_result.get("meets_100ms_target", False)
    speedup  = latency_result.get("speedup_keras_vs_tflite_x")

    logger.info(
        "Latency complete | tflite_median=%.2fms | pipeline_median=%.2fms | "
        "full_pipeline=%.2fms | keras_median=%.2fms | speedup=%.2fx | "
        "meets_100ms=%s",
        t_stats.get("median_ms", float("nan")),
        p_stats.get("median_ms", float("nan")),
        full_ms,
        k_stats.get("median_ms", float("nan")),
        speedup or 0.0,
        meets,
    )

    return latency_result


# ===========================================================================
# Step 6 — Release gate
# ===========================================================================

def _step6_assemble_gate(
    args: argparse.Namespace,
    logger: Any,
    verification_result: Dict[str, Any],
    latency_result: Dict[str, Any],
) -> Any:
    """
    Assemble the release gate from verification + latency results.

    When latency was skipped (stub result with meets_100ms_target=None),
    assemble_release_gate() should treat the latency criteria as "not
    evaluated" rather than "failed". This is the correct conservative
    interpretation for CI/CD pipelines that defer latency checks.

    Returns a ReleaseGateResult instance (always — never raises).
    """
    from src.export.verify import assemble_release_gate

    try:
        gate = assemble_release_gate(
            verification_result=verification_result,
            latency_result=latency_result,
            tflite_path=args.output,
        )
    except Exception as exc:
        # Gate assembly should never fail, but if it does, return a definitive
        # "failed" gate with an explanation rather than crashing the pipeline.
        logger.error(
            "assemble_release_gate() raised an exception: %s: %s. "
            "Creating a failed gate with the error as the hard failure.",
            type(exc).__name__,
            exc,
        )
        from src.export.verify import ReleaseGateResult
        gate = ReleaseGateResult()
        # Override release_ready to False by injecting a hard failure
        # (the dataclass may have a mechanism for this, otherwise we return
        # a minimal object with a meaningful report)
        gate.tflite_file_exists = Path(args.output).exists()
        gate.tflite_size_mb     = float("nan")
        gate.size_under_10mb    = False
        return gate

    if gate.release_ready:
        logger.info(
            "Release gate: PASS — gesture_bilstm_v1.tflite is approved for Stage 9"
        )
    else:
        logger.warning(
            "Release gate: FAIL — %d hard failure(s): %s",
            len(gate.hard_failures),
            " | ".join(gate.hard_failures),
        )

    if gate.warnings:
        for w in gate.warnings:
            logger.warning("Gate warning: %s", w)

    return gate


# ===========================================================================
# Step 7 — Write verification report + model metadata
# ===========================================================================

def _step7_write_reports(
    args: argparse.Namespace,
    logger: Any,
    gate: Any,
    verification_result: Dict[str, Any],
    latency_result: Dict[str, Any],
    per_class_delta: List[Dict[str, Any]],
    export_result: Optional[Dict[str, Any]],
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Write tflite_verification_report.json and gesture_model_metadata.json.

    These two artefacts are written separately so a failure in one does not
    block the other. Both paths are returned (or None on failure) so the
    release summary can report their status accurately.

    IMPORTANT: ``save_verification_report`` and ``write_model_metadata`` have
    specific, documented signatures — this function passes only the arguments
    those signatures define.
    """
    from src.export.verify import save_verification_report, write_model_metadata

    verification_report_path: Optional[Path] = None
    metadata_path: Optional[Path] = None

    # ── Verification report ──────────────────────────────────────────────
    try:
        Path(args.verification_report).parent.mkdir(parents=True, exist_ok=True)
        verification_report_path = save_verification_report(
            gate=gate,
            verification_result=verification_result,
            latency_result=latency_result,
            per_class_delta=per_class_delta if per_class_delta else None,
            conversion_result=export_result,
            output_path=args.verification_report,
        )
        logger.info("Verification report written → %s", verification_report_path)
    except TypeError as exc:
        # save_verification_report may not accept all kwargs — try with minimal args
        logger.warning(
            "save_verification_report() raised TypeError (%s) — retrying with "
            "minimal required arguments.",
            exc,
        )
        try:
            Path(args.verification_report).parent.mkdir(parents=True, exist_ok=True)
            verification_report_path = save_verification_report(
                gate=gate,
                verification_result=verification_result,
                latency_result=latency_result,
                output_path=args.verification_report,
            )
            logger.info(
                "Verification report written (minimal args) → %s",
                verification_report_path,
            )
        except Exception as exc2:
            logger.error(
                "Failed to write verification report: %s: %s", type(exc2).__name__, exc2
            )
    except Exception as exc:
        logger.error(
            "Failed to write verification report: %s: %s", type(exc).__name__, exc
        )

    # ── Model metadata ───────────────────────────────────────────────────
    # Resolve Stage 6 calibration report path (optional — fall back gracefully)
    stage6_report_path: Optional[str] = None
    stage6_report = Path(args.stage6_report)
    if stage6_report.exists():
        stage6_report_path = str(stage6_report)
        logger.info("Stage 6 report found — calibration metrics will be loaded: %s", stage6_report)
    else:
        logger.warning(
            "Stage 6 report not found at %s — "
            "gesture_model_metadata.json will use hardcoded Stage 6 Phase D "
            "calibration constants.",
            stage6_report,
        )

    # Build a robust conversion_result for metadata even when --skip-export
    if export_result is not None:
        conversion_result_for_meta = export_result
    else:
        # --skip-export path: reconstruct the minimum required fields from disk
        tflite_path = Path(args.output)
        tflite_disk_mb = (
            tflite_path.stat().st_size / (1024 ** 2)
            if tflite_path.exists()
            else float("nan")
        )
        conversion_result_for_meta = {
            "tflite_disk_mb":                round(tflite_disk_mb, 4),
            "param_memory_mb":               round(68_771 * 4 / (1024 ** 2), 4),
            "savedmodel_disk_mb":            float("nan"),
            "size_reduction_vs_params_x":    float("nan"),
            "size_reduction_vs_savedmodel_x": float("nan"),
            "sha256_checksum":               None,
            "quantised":                     True,
            "quantisation_mode":             "dynamic_range",
            "used_select_tf_ops":            True,
            "_source":                       "reconstructed from disk (--skip-export)",
        }
        logger.info(
            "Built synthetic conversion_result from disk (--skip-export): "
            "tflite_disk_mb=%.4f MB",
            tflite_disk_mb,
        )

    try:
        Path(args.metadata_output).parent.mkdir(parents=True, exist_ok=True)
        write_model_metadata(
            tflite_path=args.output,
            conversion_result=conversion_result_for_meta,
            verification_result=verification_result,
            latency_result=latency_result,
            config_snapshot_path=args.config_snapshot,
            stage6_report_path=stage6_report_path,
            output_path=args.metadata_output,
        )
        metadata_path = Path(args.metadata_output)
        logger.info("Model metadata written → %s", metadata_path)
    except TypeError as exc:
        # Gracefully handle signature mismatch — try without optional args
        logger.warning(
            "write_model_metadata() raised TypeError (%s) — "
            "retrying with core required arguments.",
            exc,
        )
        try:
            write_model_metadata(
                tflite_path=args.output,
                conversion_result=conversion_result_for_meta,
                verification_result=verification_result,
                latency_result=latency_result,
                config_snapshot_path=args.config_snapshot,
                output_path=args.metadata_output,
            )
            metadata_path = Path(args.metadata_output)
            logger.info("Model metadata written (core args) → %s", metadata_path)
        except Exception as exc2:
            logger.error(
                "Failed to write model metadata: %s: %s", type(exc2).__name__, exc2
            )
    except Exception as exc:
        logger.error(
            "Failed to write model metadata: %s: %s", type(exc).__name__, exc
        )

    return verification_report_path, metadata_path


# ===========================================================================
# Step 8 — Figure generation
# ===========================================================================

def _step8_generate_figures(
    args: argparse.Namespace,
    logger: Any,
    export_result: Optional[Dict[str, Any]],
    verification_result: Dict[str, Any],
    per_class_delta: List[Dict[str, Any]],
) -> Dict[str, str]:
    """
    Generate the three Stage 8 figures:
      - tflite_size_comparison.png
      - tflite_accuracy_comparison.png
      - tflite_per_class_delta.png

    Each figure is produced independently; a failure in one does not prevent
    the others from being produced.  The returned dict maps figure filename
    to its resolved path or an "ERROR: ..." message.
    """
    from src.export.verify import (
        plot_tflite_accuracy_comparison,
        plot_tflite_per_class_delta,
        plot_tflite_size_comparison,
    )

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    produced: Dict[str, str] = {}

    # Build a robust conversion_result for the size comparison figure
    tflite_path = Path(args.output)
    tflite_disk_mb = (
        tflite_path.stat().st_size / (1024 ** 2)
        if tflite_path.exists()
        else float("nan")
    )

    if export_result is not None:
        conversion_result_for_fig = export_result
    else:
        conversion_result_for_fig = {
            "param_memory_mb":               round(68_771 * 4 / (1024 ** 2), 4),
            "savedmodel_disk_mb":            float("nan"),
            "tflite_disk_mb":                round(tflite_disk_mb, 4),
            "size_reduction_vs_params_x":    float("nan"),
            "size_reduction_vs_savedmodel_x": float("nan"),
        }

    figure_tasks = [
        (
            "tflite_size_comparison.png",
            plot_tflite_size_comparison,
            {"conversion_result": conversion_result_for_fig},
        ),
        (
            "tflite_accuracy_comparison.png",
            plot_tflite_accuracy_comparison,
            {"verification_result": verification_result},
        ),
        (
            "tflite_per_class_delta.png",
            plot_tflite_per_class_delta,
            {"per_class_delta": per_class_delta, "n_classes": args.n_classes},
        ),
    ]

    for fname, fn, kwargs in figure_tasks:
        out_path = figures_dir / fname
        try:
            fig = fn(**kwargs, output_path=out_path, figure_dpi=150)
            # Release matplotlib resources
            try:
                import matplotlib.pyplot as plt
                plt.close(fig)
            except Exception:
                pass
            produced[fname] = str(out_path.resolve())
            logger.info("Figure produced → %s", out_path)
        except Exception as exc:
            err_msg = f"ERROR: {type(exc).__name__}: {exc}"
            produced[fname] = err_msg
            logger.warning(
                "Failed to generate figure %s: %s: %s",
                fname,
                type(exc).__name__,
                exc,
            )

    return produced


# ===========================================================================
# Release summary printer
# ===========================================================================

def _print_release_summary(
    gate: Any,
    export_result: Optional[Dict[str, Any]],
    verification_result: Dict[str, Any],
    latency_result: Dict[str, Any],
    per_class_delta: List[Dict[str, Any]],
    figures: Dict[str, str],
    report_paths: Dict[str, Optional[Path]],
    args: argparse.Namespace,
    elapsed_total: float,
) -> None:
    """
    Print a structured, human-readable release summary to stdout.

    This summary is always printed regardless of gate outcome so it can serve
    as a debugging tool for failed runs.
    """
    val  = verification_result.get("val",  {})
    test = verification_result.get("test", {})
    is_stub_latency = latency_result.get("_stub", False)
    tflite_stats    = latency_result.get("tflite",   {})
    pipeline_stats  = latency_result.get("pipeline", {})
    keras_stats     = latency_result.get("keras",    {})

    tflite_size_mb = None
    if export_result:
        tflite_size_mb = export_result.get("tflite_disk_mb")
    if tflite_size_mb is None or _is_nan(tflite_size_mb):
        tflite_p = Path(args.output)
        if tflite_p.exists():
            tflite_size_mb = tflite_p.stat().st_size / (1024 ** 2)

    # ── Header ──────────────────────────────────────────────────────────
    print(_header("Stage 8 — TFLite Export & Verification Summary"))

    # ── TFLite artefact ─────────────────────────────────────────────────
    print("\nTFLite Artefact:")
    print(_kv("path",          str(Path(args.output).resolve())))
    print(_kv("size",          _fmt_mb(tflite_size_mb)))
    if export_result:
        sha = str(export_result.get("sha256_checksum") or "N/A")
        print(_kv("sha256",        sha[:24] + "..." if len(sha) > 24 else sha))
        print(_kv("quantisation",  export_result.get("quantisation_mode", "N/A")))
        print(_kv("select_tf_ops", export_result.get("used_select_tf_ops", "N/A")))
        print(_kv("param_memory",  _fmt_mb(export_result.get("param_memory_mb"))))
        print(_kv("savedmodel_size", _fmt_mb(export_result.get("savedmodel_disk_mb"))))
        sr_params = export_result.get("size_reduction_vs_params_x")
        sr_sm     = export_result.get("size_reduction_vs_savedmodel_x")
        print(_kv("size_reduction (params)",    f"{sr_params}×" if sr_params else "N/A"))
        print(_kv("size_reduction (savedmodel)", f"{sr_sm}×"    if sr_sm     else "N/A"))
    else:
        print(_kv("export",        "SKIPPED (--skip-export)"))

    # ── Accuracy comparison ──────────────────────────────────────────────
    print(f"\nAccuracy Comparison  (Δ threshold ±{_DELTA_THRESHOLD}):")
    for split_name, split_data in [("val", val), ("test", test)]:
        if not split_data:
            print(_kv(f"  {split_name}", "NOT AVAILABLE"))
            continue
        print(f"\n  {split_name.upper()} split  (n={split_data.get('n_samples', '?')}):")
        print(_kv("    Keras macro-F1",    _fmt_f1(split_data.get("keras_macro_f1"))))
        print(_kv("    TFLite macro-F1",   _fmt_f1(split_data.get("tflite_macro_f1"))))
        print(_kv("    Δ macro-F1",        _fmt_delta(split_data.get("delta_macro_f1"))))
        print(_kv("    Keras accuracy",    _fmt_pct(split_data.get("keras_accuracy"))))
        print(_kv("    TFLite accuracy",   _fmt_pct(split_data.get("tflite_accuracy"))))
        print(_kv(
            "    Argmax agreement",
            _fmt_agreement(split_data.get("argmax_agreement")),
        ))
        n_d = split_data.get("n_disagreements", "?")
        n_s = split_data.get("n_samples", "?")
        print(_kv("    Disagreements",   f"{n_d}/{n_s} clips"))
        print(_kv(
            "    Keras→TFLite wrong",
            split_data.get("n_keras_right_tflite_wrong", "?"),
        ))
        print(_kv(
            "    TFLite→Keras right",
            split_data.get("n_keras_wrong_tflite_right", "?"),
        ))

    # ── Calibration continuity ───────────────────────────────────────────
    if val:
        print("\nCalibration Continuity:")
        print(_kv("  Keras mean confidence",    _fmt_f1(val.get("keras_mean_confidence"))))
        print(_kv("  TFLite mean confidence",   _fmt_f1(val.get("tflite_mean_confidence"))))
        conf_shift = val.get("confidence_shift")
        shift_str = (
            f"{float(conf_shift):+.4f}"
            if conf_shift is not None and not _is_nan(conf_shift)
            else "N/A"
        )
        print(_kv("  Confidence shift",          shift_str))
        print(_kv("  Mean abs prob diff",         _fmt_f1(val.get("mean_abs_diff"))))

    # ── Latency ─────────────────────────────────────────────────────────
    if is_stub_latency:
        stub_reason = latency_result.get("_stub_reason", "skipped")
        print(f"\nLatency:  SKIPPED ({stub_reason})")
        print(_skip_msg(
            "Latency gate criteria not evaluated. Re-run without "
            "--skip-benchmark for production gate."
        ))
    else:
        print(f"\nLatency  (target ≤ {_LATENCY_TARGET_MS:.0f} ms full pipeline):")
        print(_kv("  TFLite median",      _fmt_ms(tflite_stats.get("median_ms"))))
        print(_kv("  TFLite p95",         _fmt_ms(tflite_stats.get("p95_ms"))))
        fps = tflite_stats.get("fps")
        print(_kv("  TFLite FPS",         f"{float(fps):.1f}" if fps and not _is_nan(fps) else "N/A"))
        print(_kv("  Pipeline median",    _fmt_ms(pipeline_stats.get("median_ms"))))
        print(_kv("  Full pipeline",      _fmt_ms(latency_result.get("full_pipeline_ms"))))
        meets = latency_result.get("meets_100ms_target")
        meets_str = "✓ Yes" if meets is True else "✗ No" if meets is False else "N/A"
        print(_kv("  Meets 100ms target", meets_str))
        sx = latency_result.get("speedup_keras_vs_tflite_x")
        print(_kv("  Keras speedup",      f"{float(sx):.2f}×" if sx and not _is_nan(sx) else "N/A"))
        print(_kv("  Keras median",       _fmt_ms(keras_stats.get("median_ms"))))

    # ── Per-class highlights ─────────────────────────────────────────────
    if per_class_delta:
        meaningful = [r for r in per_class_delta if r.get("meaningful_degradation")]
        confusable_affected = [
            r for r in per_class_delta
            if r.get("is_confusable_pair") and abs(r.get("f1_delta", 0.0)) > 1e-6
        ]
        print(f"\nPer-Class Delta Analysis  ({len(per_class_delta)} classes):")
        if meaningful:
            print(f"  Meaningful degradations (|Δ| > 0.10, non-singleton):  {len(meaningful)}")
            for row in sorted(meaningful, key=lambda r: abs(r.get("f1_delta", 0)), reverse=True)[:5]:
                risk  = " ⚠HIGH-RISK" if row.get("is_high_risk")      else ""
                conf  = " [confusable]" if row.get("is_confusable_pair") else ""
                print(
                    f"    {row['sign']:>16}  keras={row.get('keras_f1', 0):.4f}  "
                    f"tflite={row.get('tflite_f1', 0):.4f}  "
                    f"Δ={row.get('f1_delta', 0):+.4f}{risk}{conf}"
                )
            if len(meaningful) > 5:
                print(f"    ... and {len(meaningful) - 5} more  (see per-class delta report)")
        else:
            print(_ok("No meaningful degradations from quantisation"))
        if confusable_affected:
            print(
                f"  Confusable-pair classes with non-zero Δ: "
                f"{', '.join(r['sign'] for r in confusable_affected)}"
            )
    elif not per_class_delta:
        print("\nPer-Class Delta:  SKIPPED (prediction cache unavailable)")

    # ── Warnings (non-blocking) ──────────────────────────────────────────
    if gate is not None and gate.warnings:
        print("\nWarnings (non-blocking):")
        for w in gate.warnings:
            print(_warn(w))

    # ── Hard failures ────────────────────────────────────────────────────
    if gate is not None and gate.hard_failures:
        print("\nHard Failures (blocking release):")
        for f in gate.hard_failures:
            print(_fail(f))

    # ── Artefacts written ────────────────────────────────────────────────
    print("\nArtefacts:")
    tflite_exists = Path(args.output).exists()
    print(f"  {'✓' if tflite_exists else '✗'}  {'TFLite model':<30}  {args.output}")

    manifest_exists = Path(args.export_manifest).exists()
    print(f"  {'✓' if manifest_exists else '✗'}  {'Export manifest':<30}  {args.export_manifest}")

    for label, path_obj in [
        ("Verification report", report_paths.get("verification")),
        ("Model metadata",      report_paths.get("metadata")),
    ]:
        if path_obj is not None and Path(path_obj).exists():
            print(f"  ✓  {label:<30}  {path_obj}")
        else:
            target = str(getattr(args, label.lower().replace(" ", "_").replace("model_", "metadata_").replace("verification_report", "verification_report"), "unknown"))
            print(f"  ✗  {label:<30}  (not written)")

    if figures:
        print("\nFigures:")
        for fname, fpath in figures.items():
            ok = not fpath.startswith("ERROR")
            status = "✓" if ok else "✗"
            detail = fpath if ok else fpath[7:]   # strip "ERROR: " prefix
            print(f"  {status}  {fname:<44}  {detail}")

    # ── Pipeline timing ──────────────────────────────────────────────────
    print(f"\nTotal pipeline time:  {elapsed_total:.1f} s")
    if args.dry_run:
        print(_warn(
            "Dry-run mode was active — benchmark n_calls=5. "
            "Re-run without --dry-run for production gate numbers."
        ))

    # ── Gate verdict ─────────────────────────────────────────────────────
    print("\n" + _sep())
    if gate is None:
        print("  GATE NOT EVALUATED  (pipeline aborted before gate assembly)")
    elif is_stub_latency and gate.release_ready:
        print("  CONDITIONALLY RELEASE READY  ⚠")
        print("  Accuracy + size gates PASS.  Latency was SKIPPED.")
        print("  Re-run without --skip-benchmark to complete gate evaluation.")
    elif gate.release_ready:
        print("  RELEASE READY  ✓")
        print("  gesture_bilstm_v1.tflite is approved for Stage 9")
    else:
        print("  RELEASE BLOCKED  ✗")
        print(f"  {len(gate.hard_failures)} hard failure(s) must be resolved before Stage 9")
    print(_sep())
    print()


# ===========================================================================
# Main orchestrator
# ===========================================================================

def main() -> int:
    """
    Orchestrate the complete Stage 8 TFLite export and verification pipeline.

    Returns:
        0 — PASS:  all hard gate criteria met
        1 — FAIL:  one or more hard gate criteria not met
        2 — ERROR: pipeline aborted due to infrastructure/configuration failure
    """
    parser = _build_parser()
    args   = parser.parse_args()

    # ── Dry-run adjustments ──────────────────────────────────────────────
    # Dry-run does NOT skip the benchmark — it runs it with minimal calls so
    # we still exercise the benchmark code path end-to-end. The latency numbers
    # are flagged as indicative-only in the summary.
    # (Contrast with --skip-benchmark which skips the step entirely.)

    # ── Configure logging ─────────────────────────────────────────────────
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    log_file = configure_logging(
        log_dir=args.log_dir,
        run_name="stage8_export_verification",
        level="INFO",
        file_level="DEBUG",
    )
    logger = get_logger(__name__, stage="export")

    # ── Banner ────────────────────────────────────────────────────────────
    print(_header("Stage 8 — TFLite Export & Verification Pipeline"))
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(_kv("Started",          ts))
    print(_kv("Log file",         str(log_file)))
    print(_kv("Config snapshot",  args.config_snapshot))
    print(_kv("Saved model",      args.saved_model))
    print(_kv("TFLite output",    args.output))
    print(_kv("N classes",        args.n_classes))
    print(_kv("Seed",             args.seed))
    print(_kv("Skip export",      args.skip_export))
    print(_kv("Skip benchmark",   args.skip_benchmark))
    print(_kv("Dry run",          args.dry_run))
    if args.dry_run:
        print(_warn(
            "Dry-run mode: benchmark will use n_calls=5, warmup=2. "
            "Do NOT use dry-run latency numbers for the production gate."
        ))

    # ── Pre-flight checks ─────────────────────────────────────────────────
    _preflight(args, logger)

    # ── Pipeline state variables (always defined so summary always works) ─
    pipeline:            Any                        = None
    dataset:             Any                        = None
    export_result:       Optional[Dict[str, Any]]   = None
    verification_result: Dict[str, Any]             = {}
    per_class_delta:     List[Dict[str, Any]]       = []
    latency_result:      Dict[str, Any]             = _make_stub_latency_result()
    gate:                Any                        = None
    figures:             Dict[str, str]             = {}
    report_paths:        Dict[str, Optional[Path]]  = {
        "verification": None,
        "metadata":     None,
    }

    t_total_start = time.perf_counter()

    # Determine total step count for the progress indicator
    n_steps = 8
    step    = 0

    exit_code = 0

    try:
        # ── Step 1: TFLite export ────────────────────────────────────────
        step += 1
        if args.skip_export:
            print(_step(step, n_steps, "TFLite export  [SKIPPED — using existing file]"))
            tflite_p = Path(args.output)
            size_mb  = tflite_p.stat().st_size / (1024 ** 2) if tflite_p.exists() else float("nan")
            print(_ok(f"Existing TFLite file: {args.output}  ({_fmt_mb(size_mb)})"))
        else:
            print(_step(step, n_steps, "TFLite export"))
            export_result = _step1_export(args, logger)
            print(_ok(
                f"Exported {export_result.get('tflite_disk_mb', 0):.4f} MB  "
                f"(conversion {export_result.get('conversion_time_s', 0):.1f}s  "
                f"sha256={str(export_result.get('sha256_checksum', ''))[:12]}...)"
            ))
            sanity = export_result.get("sanity_check", {})
            if sanity.get("output_finite") is True:
                print(_ok("Sanity inference: output finite ✓"))
            elif sanity.get("output_finite") is False:
                print(_fail("Sanity inference: non-finite output detected!"))

        # ── Step 2: Dataset + pipeline ───────────────────────────────────
        step += 1
        print(_step(step, n_steps, "Building GestureDataset + FeaturePipeline"))
        pipeline, dataset = _step2_build_dataset_and_pipeline(args, logger)
        print(_ok(
            f"Dataset ready: "
            f"train={dataset.n_train}  "
            f"val={dataset.n_val}  "
            f"test={dataset.n_test}  "
            f"landmark_config={pipeline.landmark_config}  "
            f"feature_dim={pipeline.feature_dim}"
        ))

        # ── Step 3: Accuracy verification ────────────────────────────────
        step += 1
        print(_step(step, n_steps, "Accuracy verification (Keras vs TFLite)"))
        verification_result = _step3_accuracy_verification(args, logger, dataset)

        # Print concise per-split summary
        for split_name in ("val", "test"):
            sr = verification_result.get(split_name, {})
            if not sr:
                print(_warn(f"{split_name}: no results returned"))
                continue
            delta    = sr.get("delta_macro_f1", float("nan"))
            agree    = sr.get("argmax_agreement", float("nan"))
            delta_ok = not _is_nan(delta) and abs(delta) <= _DELTA_THRESHOLD
            agree_ok = not _is_nan(agree) and agree >= _AGREEMENT_THRESHOLD
            status   = "✓" if (delta_ok and agree_ok) else "⚠"
            print(f"  {status}  {split_name}:  "
                  f"keras_f1={_fmt_f1(sr.get('keras_macro_f1'))}  "
                  f"tflite_f1={_fmt_f1(sr.get('tflite_macro_f1'))}  "
                  f"Δ={delta:+.4f}  agreement={agree:.4f}  "
                  f"n={sr.get('n_samples', '?')}")

        # ── Step 4: Per-class delta ───────────────────────────────────────
        step += 1
        print(_step(step, n_steps, "Per-class F1 delta analysis"))
        per_class_delta = _step4_per_class_delta(
            args, logger, verification_result, dataset
        )
        if per_class_delta:
            n_md  = sum(1 for r in per_class_delta if r.get("meaningful_degradation"))
            n_all = len(per_class_delta)
            print(_ok(
                f"{n_md}/{n_all} meaningful degradations | "
                f"{n_all} classes analysed (no re-inference)"
            ))
        else:
            print(_skip_msg("Per-class delta skipped (prediction cache unavailable)"))

        # ── Step 5: Latency benchmark ─────────────────────────────────────
        step += 1
        if args.skip_benchmark:
            print(_step(step, n_steps, "Latency benchmark  [SKIPPED]"))
            # latency_result is already the stub from initialisation
            print(_skip_msg(
                "Latency gate criteria will not be evaluated. "
                "Re-run without --skip-benchmark for a complete gate."
            ))
        else:
            label = "Latency benchmark" + (" (dry-run: n_calls=5)" if args.dry_run else "")
            print(_step(step, n_steps, label))
            latency_result = _step5_latency_benchmark(args, logger, pipeline)
            if latency_result.get("_stub"):
                print(_warn(
                    f"Benchmark returned stub: {latency_result.get('_stub_reason', 'unknown')}"
                ))
            else:
                full_ms = latency_result.get("full_pipeline_ms", float("nan"))
                meets   = latency_result.get("meets_100ms_target", False)
                t_med   = latency_result.get("tflite", {}).get("median_ms", float("nan"))
                fps     = latency_result.get("tflite", {}).get("fps", float("nan"))
                status  = "✓" if meets else "✗"
                print(
                    f"  {status}  TFLite median={_fmt_ms(t_med)}  "
                    f"full_pipeline={_fmt_ms(full_ms)}  "
                    f"FPS={fps:.0f}  "
                    f"meets_100ms={'Yes' if meets else 'No'}"
                )

        # ── Step 6: Release gate ──────────────────────────────────────────
        step += 1
        print(_step(step, n_steps, "Release gate evaluation"))
        gate = _step6_assemble_gate(
            args, logger, verification_result, latency_result
        )
        if gate.release_ready and not latency_result.get("_stub"):
            print(_ok("ALL hard gate criteria met — RELEASE READY"))
        elif gate.release_ready and latency_result.get("_stub"):
            print(_warn(
                "Accuracy + size gates PASS. "
                "Latency was SKIPPED — re-run for full gate."
            ))
        else:
            for hf in gate.hard_failures:
                print(_fail(hf))

        # ── Step 7: Write reports + metadata ─────────────────────────────
        step += 1
        print(_step(step, n_steps, "Writing verification report + model metadata"))
        r_path, m_path = _step7_write_reports(
            args, logger, gate,
            verification_result, latency_result,
            per_class_delta, export_result,
        )
        report_paths["verification"] = r_path
        report_paths["metadata"]     = m_path

        if r_path:
            print(_ok(f"Verification report → {r_path}"))
        else:
            print(_fail("Verification report NOT written (check logs)"))
        if m_path:
            print(_ok(f"Model metadata → {m_path}"))
        else:
            print(_fail("Model metadata NOT written (check logs)"))

        # ── Step 8: Figures ───────────────────────────────────────────────
        step += 1
        print(_step(step, n_steps, "Generating Stage 8 figures"))
        figures = _step8_generate_figures(
            args, logger, export_result, verification_result, per_class_delta
        )
        n_ok  = sum(1 for v in figures.values() if not v.startswith("ERROR"))
        n_err = len(figures) - n_ok
        if n_err == 0:
            print(_ok(f"{n_ok}/3 figures produced"))
        else:
            print(_warn(f"{n_ok}/3 figures produced  ({n_err} failed — see above)"))

    except SystemExit:
        raise
    except KeyboardInterrupt:
        print("\n\nAborted by user (KeyboardInterrupt).", file=sys.stderr)
        logger.warning("Pipeline aborted by user (KeyboardInterrupt)")
        exit_code = 2
    except Exception as exc:
        print(f"\n\n{'!'*60}", file=sys.stderr)
        print(f"PIPELINE ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"{'!'*60}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        logger.exception("Pipeline aborted due to unhandled exception")
        exit_code = 2

    # ── Final summary — ALWAYS printed ───────────────────────────────────
    elapsed_total = time.perf_counter() - t_total_start
    _print_release_summary(
        gate=gate,
        export_result=export_result,
        verification_result=verification_result,
        latency_result=latency_result,
        per_class_delta=per_class_delta,
        figures=figures,
        report_paths=report_paths,
        args=args,
        elapsed_total=elapsed_total,
    )

    # ── Persist gate report to JSON for CI/CD consumption ─────────────────
    gate_json_path = Path(args.verification_report).parent / "release_gate.json"
    if gate is not None:
        try:
            gate_json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(gate_json_path, "w", encoding="utf-8") as fh:
                # Use gate.to_dict() if available, else a minimal fallback
                try:
                    gate_dict = gate.to_dict()
                except AttributeError:
                    import dataclasses
                    gate_dict = dataclasses.asdict(gate) if dataclasses.is_dataclass(gate) else {}
                json.dump(gate_dict, fh, indent=2, default=str)
            logger.info("Release gate JSON written → %s", gate_json_path)
        except Exception as exc:
            logger.warning("Failed to write release_gate.json: %s", exc)

    # ── Determine final exit code ─────────────────────────────────────────
    if exit_code == 2:
        return 2
    if gate is None:
        return 2
    if gate.release_ready:
        return 0
    return 1


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    sys.exit(main())