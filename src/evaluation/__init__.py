"""
src/evaluation
===============
Stage 6 — Evaluation, Benchmarking, and Interpretability.

All four Phase A modules are now implemented and exported from this package:

    metrics.py          — core metric primitives (macro-F1, confusion matrix,
                          per-class breakdown, bootstrap CI, prediction
                          extraction). Foundation module — every other Stage 6
                          module either calls into this or consumes its output.

    benchmark.py        — latency and model-size benchmarking. Times real
                          inference against the champion Keras SavedModel and
                          an optional scratch TFLite export. Produces the data
                          for reports/figures/latency_benchmark.png.

    calibration.py      — confidence calibration diagnostics (reliability
                          diagram, ECE, MCE, confidence-threshold curve).
                          Operates exclusively on already-extracted
                          (y_true, y_proba) arrays from the Phase B1 cache.

    signer_analysis.py  — per-signer generalisation diagnostics (per-signer
                          accuracy, dual macro-F1, Wilson CIs, spread bootstrap,
                          high-risk correlation, failure-mode metadata summary).
                          Operates exclusively on (y_true, y_pred, signer_ids)
                          from the Phase B1 / Phase C prediction caches.

Phase B1 / Phase C prediction caches
--------------------------------------
    reports/evaluation/predictions/val_predictions.npz
        Written by pipelines/run_evaluation.py Phase B1.
        Schema: y_true, y_pred, y_prob, clip_ids, signer_ids,
                detected_frame_count, missing_pct.

    reports/evaluation/predictions/test_predictions.npz
        Written by pipelines/run_evaluation.py Phase C (one-shot, gated).
        Same schema as val_predictions.npz.
        MUST be preceded by a pre-commitment entry in
        reports/test_evaluation_log.md.

Champion model context (for reference)
-----------------------------------------
    bilstm_hands_only_v4_aug
    val_macro_f1  : 0.6011  (52 clips, 7 unseen signers)
    input_shape   : (1, 100, 126)  — seq_len=100, hands_only
    param_count   : 68,771
    early_stopping_monitor: val_accuracy  (NOT val_macro_f1 — discrepancy
        between config_snapshot.yaml and Stage 5 handoff narrative; flagged
        in reports/evaluation/evaluation_report.json per Phase F requirements)
    MLflow run ID : cb16f689d2294001a2ff2d3e02419d27
    SavedModel    : models/bilstm_hands_only_v4_aug_saved_model/

Import convention
------------------
This package uses absolute imports (``from src.evaluation.metrics import …``)
rather than relative imports (``from .metrics import …``) for consistency with
the rest of the codebase. All pipeline entry points, training scripts, and
tests use the ``src.*`` absolute-import style; making this package the sole
exception would create a confusing inconsistency.

If an IDE's import resolver underlines these imports, the usual fix is to open
the editor at the repository root (the directory containing ``src/``,
``pipelines/``, ``configs/``) or add
``"python.analysis.extraPaths": ["./src"]`` to ``.vscode/settings.json``.

Re-exported public API
------------------------
All four module public APIs are re-exported here so callers can write either:

    from src.evaluation import compute_macro_f1, plot_reliability_diagram

or the more explicit:

    from src.evaluation.metrics import compute_macro_f1
    from src.evaluation.calibration import plot_reliability_diagram

Both forms are valid and produce identical imports. The explicit form is
preferred inside this package itself (to keep each module independently
importable and testable); the short form is provided as a convenience for
notebook cells and pipeline entry points.

``__all__`` is kept explicit (not auto-derived from ``globals()``). This is
slightly more maintenance overhead — a new function added to any sub-module
must be added here too — but it guarantees ``from src.evaluation import *``
never accidentally exposes private helpers. This trade-off is intentional.

Maintenance note
-----------------
When adding a new sub-module to Stage 6 (e.g. an ``error_analysis.py`` for
Phase D), add its public functions to BOTH the import block below AND
``__all__`` in the same commit that creates the module. A package __init__
whose exports lag behind its on-disk modules is worse than one that exports
nothing yet: ``from src.evaluation import new_function`` raising ImportError
long after new_function.py exists is a confusing debugging experience.
"""

# ---------------------------------------------------------------------------
# metrics.py — core primitives; every other module depends on these
# ---------------------------------------------------------------------------
from src.evaluation.metrics import (
    DEFAULT_BOOTSTRAP_CI,
    DEFAULT_N_BOOTSTRAP,
    DEFAULT_SEED,
    HIGH_RISK_SIGNS,
    N_CLASSES,
    bootstrap_macro_f1_ci,
    compute_accuracy,
    compute_confusion_matrix,
    compute_confusion_matrix_from_predictions,
    compute_evaluation_summary,
    compute_macro_f1,
    compute_per_class_metrics,
    compute_support_counts,
    get_predictions,
    get_val_predictions,
    rank_classes_by_f1,
)

# ---------------------------------------------------------------------------
# benchmark.py — latency and model-size benchmarking (Phase A2)
# ---------------------------------------------------------------------------
from src.evaluation.benchmark import (
    DEFAULT_N_CALLS,
    DEFAULT_WARMUP,
    TFLiteCallable,
    benchmark_champion_summary,
    benchmark_inference,
    benchmark_keras_savedmodel,
    benchmark_model_registry,
    benchmark_pipeline_preprocessing,
    benchmark_tflite_inference,
    build_latency_comparison_rows,
    compute_file_size_mb,
    compute_keras_model_size_mb,
    convert_to_scratch_tflite,
    time_callable,
)

# ---------------------------------------------------------------------------
# calibration.py — confidence calibration diagnostics (Phase A3)
# ---------------------------------------------------------------------------
from src.evaluation.calibration import (
    DEFAULT_N_BINS,
    DEFAULT_N_THRESHOLD_POINTS,
    SPARSE_BIN_THRESHOLD,
    TEMPERATURE_SCALING_NOTE,
    compute_calibration_summary,
    compute_confidence_threshold_curve,
    compute_reliability_diagram,
    plot_confidence_threshold_curve,
    plot_reliability_diagram,
)

# ---------------------------------------------------------------------------
# signer_analysis.py — per-signer generalisation diagnostics (Phase A4)
# ---------------------------------------------------------------------------
from src.evaluation.signer_analysis import (
    DEFAULT_BOOTSTRAP_CI as SIGNER_DEFAULT_BOOTSTRAP_CI,   # re-export with
    DEFAULT_N_BOOTSTRAP as SIGNER_DEFAULT_N_BOOTSTRAP,      # unambiguous names
    HIGH_RISK_SIGNS as SIGNER_HIGH_RISK_SIGNS,              # (same values as
    SPARSE_SIGNER_THRESHOLD,                                 # metrics.py but
    UNSEEN_SIGNER_FRAMING_NOTE,                              # different module)
    compute_per_signer_accuracy,
    compute_signer_analysis_summary,
    compute_signer_failure_mode_summary,
    compute_signer_high_risk_correlation,
    compute_signer_spread_bootstrap_ci,
    plot_signer_generalisation,
)


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------

__all__ = [
    # ── metrics.py ────────────────────────────────────────────────────────
    "N_CLASSES",
    "HIGH_RISK_SIGNS",
    "DEFAULT_N_BOOTSTRAP",
    "DEFAULT_BOOTSTRAP_CI",
    "DEFAULT_SEED",
    "get_predictions",
    "get_val_predictions",
    "compute_macro_f1",
    "compute_accuracy",
    "compute_confusion_matrix",
    "compute_confusion_matrix_from_predictions",
    "compute_support_counts",
    "compute_per_class_metrics",
    "rank_classes_by_f1",
    "bootstrap_macro_f1_ci",
    "compute_evaluation_summary",
    # ── benchmark.py ──────────────────────────────────────────────────────
    "DEFAULT_N_CALLS",
    "DEFAULT_WARMUP",
    "TFLiteCallable",
    "benchmark_champion_summary",
    "benchmark_inference",
    "benchmark_keras_savedmodel",
    "benchmark_model_registry",
    "benchmark_pipeline_preprocessing",
    "benchmark_tflite_inference",
    "build_latency_comparison_rows",
    "compute_file_size_mb",
    "compute_keras_model_size_mb",
    "convert_to_scratch_tflite",
    "time_callable",
    # ── calibration.py ────────────────────────────────────────────────────
    "DEFAULT_N_BINS",
    "DEFAULT_N_THRESHOLD_POINTS",
    "SPARSE_BIN_THRESHOLD",
    "TEMPERATURE_SCALING_NOTE",
    "compute_reliability_diagram",
    "compute_confidence_threshold_curve",
    "compute_calibration_summary",
    "plot_reliability_diagram",
    "plot_confidence_threshold_curve",
    # ── signer_analysis.py ────────────────────────────────────────────────
    "SIGNER_DEFAULT_BOOTSTRAP_CI",
    "SIGNER_DEFAULT_N_BOOTSTRAP",
    "SIGNER_HIGH_RISK_SIGNS",
    "SPARSE_SIGNER_THRESHOLD",
    "UNSEEN_SIGNER_FRAMING_NOTE",
    "compute_per_signer_accuracy",
    "compute_signer_spread_bootstrap_ci",
    "compute_signer_high_risk_correlation",
    "compute_signer_failure_mode_summary",
    "compute_signer_analysis_summary",
    "plot_signer_generalisation",
]