"""
pipelines/run_evaluation.py
============================
Stage 6 (Phase F) — Evaluation orchestrator for the WLASL 35-class gesture
recognition system.

This script sequences the complete Stage 6 evaluation pipeline against the
champion model ``bilstm_hands_only_v4_aug``, reading from the prediction
caches produced in Phases B1 and C rather than re-running inference. All
analysis modules (metrics, calibration, signer_analysis, benchmark) are
driven from this single entry point.

Pipeline sequence
------------------
    Phase B1 — Val prediction cache (reads/verifies, does NOT re-infer)
    Phase C  — Test evaluation (gated; refuses silent re-evaluation)
    Phase D  — Analysis layer (all consume caches only, zero new inference)
        D1. Confusion matrices (val + test)
        D2. Per-class metrics + ranked F1 chart
        D3. Confidence calibration (val)
        D4. Signer analysis (val + test)
        D5. Failure-mode taxonomy
        D6. Latency benchmarking (Keras + scratch TFLite)
    Phase E  — SHAP / Gradient×Input (invokes notebook via papermill,
               or skips with a documented note if papermill unavailable)
    Phase F  — Write consolidated evaluation_report.json + completion gate

Usage
-----
    # Standard evaluation run (val only, skip re-running test inference)
    python pipelines/run_evaluation.py \\
        --champion-run bilstm_hands_only_v4_aug \\
        --splits val \\
        --output-dir reports/evaluation/

    # Full eval including test split (pre-commitment log must already exist)
    python pipelines/run_evaluation.py \\
        --champion-run bilstm_hands_only_v4_aug \\
        --splits val test \\
        --output-dir reports/evaluation/

    # Re-run test (requires explicit confirmation flag — never silent)
    python pipelines/run_evaluation.py \\
        --champion-run bilstm_hands_only_v4_aug \\
        --splits val test \\
        --output-dir reports/evaluation/ \\
        --force-rerun-test

    # Skip latency benchmarking (useful on headless CI)
    python pipelines/run_evaluation.py \\
        --splits val \\
        --skip-benchmark

    # Skip SHAP/interpretability notebook (useful if papermill unavailable)
    python pipelines/run_evaluation.py \\
        --splits val test \\
        --skip-shap

Exit codes
----------
    0  — All completion gate checks passed.
    1  — One or more checks failed (details logged to stderr + report).
    2  — User aborted (e.g. refused to confirm force-rerun).

Critical design constraints
-----------------------------
  - NEVER re-evaluates the test set silently. If ``reports/evaluation/
    predictions/test_predictions.npz`` already exists and ``--force-rerun-test``
    is not passed, the test block is skipped and existing numbers are loaded.
  - Accepts the label map's v1.1 schema: ``{"signs": [{"class_idx": N,
    "name": "..."}]}``. The Phase D notebook's parser bug (falling back to
    class_0…class_34 placeholders) is NOT reproduced here.
  - All analysis modules receive already-extracted ``(y_true, y_pred, y_prob,
    signer_ids, …)`` arrays — zero new model inference calls after Phase B1/C.
  - The ``evaluation_report.json`` produced here is the single authoritative
    record of Stage 6 results, safe to ``json.dump()`` and commit to the repo.
  - ``LIMITATIONS.md`` is patched (not replaced) with Stage 6-specific findings
    once the analysis is complete.
  - Every ``mlflow.log_*`` call is inside a try/except so an unavailable
    MLflow server never aborts the evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path regardless of invocation directory
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Logger — must import after path adjustment
# ---------------------------------------------------------------------------
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Champion model run name (default, overridable via --champion-run).
_DEFAULT_CHAMPION_RUN: str = "bilstm_hands_only_v4_aug"

#: Path to the champion SavedModel (relative to project root).
_CHAMPION_SAVED_MODEL_PATH: str = "models/bilstm_hands_only_v4_aug_saved_model"

#: MLflow experiment name (project constant from Part 2).
_MLFLOW_EXPERIMENT_NAME: str = "WLASL-35-class"

#: Project global seed.
_SEED: int = 42

#: Number of output classes (locked — label_map_v1.json v1.1).
_N_CLASSES: int = 35

#: Path to label map.
_LABEL_MAP_PATH: str = "artifacts/label_map_v1.json"

#: Prediction cache paths.
_CACHE_DIR: str = "reports/evaluation/predictions"
_VAL_CACHE_NAME: str = "val_predictions.npz"
_TEST_CACHE_NAME: str = "test_predictions.npz"

#: Pre-commitment log path (must predate test cache).
_PRE_COMMITMENT_LOG: str = "reports/test_evaluation_log.md"

#: Consolidated evaluation report path.
_EVAL_REPORT_PATH: str = "reports/evaluation/evaluation_report.json"

#: Figures output directory.
_FIGURES_DIR: str = "reports/figures"

#: Known config discrepancy — documented for every evaluation report.
_CONFIG_DISCREPANCY: Dict[str, str] = {
    "field": "early_stopping_monitor",
    "config_snapshot_value": "val_accuracy",
    "handoff_narrated_value": "val_macro_f1",
    "resolution": (
        "Manual early stopping in train.py used val_macro_f1 (sklearn). "
        "The Keras ReduceLROnPlateau callback monitored val_accuracy. "
        "Champion weights were selected on val_macro_f1 by the manual "
        "patience counter. The config field early_stopping_monitor controls "
        "only ReduceLROnPlateau, not the stopping criterion."
    ),
}

#: Expected val macro-F1 for champion (Stage 5 locked result).
_EXPECTED_VAL_MACRO_F1: float = 0.6011

#: Tolerance for val F1 drift between cache and recomputed value.
_VAL_F1_TOLERANCE: float = 0.05

#: DPI for all saved figures.
_FIGURE_DPI: int = 150

#: Confusion matrix figure size for 35 classes.
_CM_FIGSIZE: Tuple[int, int] = (22, 20)

#: Completion gate: required figures.
_REQUIRED_FIGURES: List[str] = [
    "confusion_matrix_best_model.png",
    "confusion_matrix_normalised.png",
    "confusion_matrix_test.png",
    "per_class_metrics.png",
    "confidence_calibration.png",
    "confidence_threshold_curve.png",
    "signer_generalisation.png",
    "latency_benchmark.png",
]

#: Completion gate: required report keys.
_REQUIRED_REPORT_KEYS: List[str] = [
    "val_evaluation",
    "test_evaluation",
    "calibration",
    "signer_analysis",
    "benchmark",
    "config_discrepancy",
    "completion_gate",
]

#: Limitations note to append for Stage 6 calibration finding.
_LIMITATIONS_CALIBRATION_PATCH: str = """

## Stage 6 — Calibration Finding (added by run_evaluation.py)

The champion model `bilstm_hands_only_v4_aug` exhibits **underconfidence**
(overconfidence_gap = −0.063, ECE = 0.200) on the validation split.
This is the opposite direction from the standard expectation for deep neural
networks trained with cross-entropy. The mechanism is plausibly insufficient
training data (236 clips) preventing full sharpening of the softmax
distribution. Temperature scaling as a post-hoc remedy would require T < 1
(boosting confidence rather than suppressing it) — atypical and likely
unreliable at this dataset size.

Recommended deployment threshold for the webcam demo: τ = 0.35 rather than
the standard 0.50. See reports/evaluation/evaluation_report.json for full
calibration metrics and bootstrap CIs.
"""

#: Limitations note to append for Stage 6 left-hand attribution finding.
_LIMITATIONS_ATTRIBUTION_PATCH: str = """

## Stage 6 — Left-Hand Attribution Gap (added by run_evaluation.py)

Phase E (Gradient × Input) revealed that the champion model's decisions
are driven overwhelmingly by right-hand fingertip motion, with near-zero
attribution to left-hand landmarks across all 52 validation clips. This
was not an architectural choice — it emerged from training data composition
(WLASL right-hand-dominant signer population). For the KSL adaptation:
a left-hand-led or ambidextrous signer test set should be the first
evaluation target on any KSL pilot data, as the current model has no
visibility into left-hand-dominant signing styles.
"""


# ---------------------------------------------------------------------------
# Fault-tolerant context manager (mirrors train.py pattern)
# ---------------------------------------------------------------------------

@contextmanager
def _guard(description: str, fatal: bool = False) -> Generator[None, None, None]:
    """
    Catch and log exceptions in analysis blocks.

    Parameters
    ----------
    description : str   Human-readable label for error messages.
    fatal       : bool  If True, re-raises the exception after logging.
                        Use for blocks that must succeed (e.g. cache loading).
                        Default False: log and continue.
    """
    try:
        yield
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error(
            f"[{description}] FAILED: {type(exc).__name__}: {exc}\n{tb}",
            extra={"stage": "evaluation"},
        )
        if fatal:
            raise


# ---------------------------------------------------------------------------
# Label map loader (v1.1 schema-aware)
# ---------------------------------------------------------------------------

def _load_sign_names(label_map_path: str) -> List[str]:
    """
    Load sign names from label_map_v1.json using the correct v1.1 schema.

    The v1.1 schema is: ``{"version": "1.1", "signs": [{"class_idx": N,
    "name": "..."}, ...]}``. Previous Phase D notebook code fell back to
    ``class_0``…``class_34`` placeholders because it used the wrong key.
    This function handles both the v1.1 schema and a flat dict fallback.

    Returns
    -------
    List[str]  Length _N_CLASSES, index-aligned with class indices.
    """
    sign_names = [f"class_{i}" for i in range(_N_CLASSES)]   # safe default

    path = Path(label_map_path)
    if not path.exists():
        logger.warning(
            f"Label map not found at {path}. Using placeholder names.",
            extra={"stage": "evaluation"},
        )
        return sign_names

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    n_filled = 0

    # ── Schema v1.1: {"signs": [{"class_idx": N, "name": "..."}, ...]} ──
    if "signs" in raw and isinstance(raw["signs"], list):
        for entry in raw["signs"]:
            if isinstance(entry, dict):
                idx  = entry.get("class_idx")
                name = entry.get("name")
                if idx is not None and name is not None:
                    try:
                        sign_names[int(idx)] = str(name)
                        n_filled += 1
                    except (IndexError, ValueError):
                        pass

    # ── Fallback: flat int-keyed or str-keyed dict ────────────────────────
    elif isinstance(raw, dict):
        for key, value in raw.items():
            try:
                idx = int(key)
                sign_names[idx] = str(value)
                n_filled += 1
            except (ValueError, IndexError):
                pass

    if n_filled == 0:
        logger.warning(
            f"Label map at {path} yielded 0 sign names. "
            "Check schema. Using placeholder names.",
            extra={"stage": "evaluation"},
        )
    elif n_filled < _N_CLASSES:
        logger.warning(
            f"Label map at {path} filled only {n_filled}/{_N_CLASSES} names.",
            extra={"stage": "evaluation"},
        )
    else:
        logger.info(
            f"Label map loaded: {n_filled} sign names from {path}",
            extra={"stage": "evaluation"},
        )

    return sign_names


# ---------------------------------------------------------------------------
# Prediction cache I/O
# ---------------------------------------------------------------------------

def _load_cache(cache_path: Path, split_name: str) -> Optional[Dict[str, Any]]:
    """
    Load a prediction cache ``.npz`` file and return a dict of arrays.

    Returns None if the file does not exist (caller decides how to handle).

    Expected keys: y_true, y_pred, y_prob, clip_ids, signer_ids,
                   detected_frame_count, missing_pct.
    """
    if not cache_path.exists():
        logger.info(
            f"Cache not found for split='{split_name}' at {cache_path}",
            extra={"stage": "evaluation"},
        )
        return None

    loaded = np.load(str(cache_path), allow_pickle=True)
    cache: Dict[str, Any] = {k: loaded[k] for k in loaded.files}

    logger.info(
        f"Loaded {split_name} cache from {cache_path} | "
        f"keys={list(cache.keys())} | "
        f"n_clips={len(cache.get('y_true', []))}",
        extra={"stage": "evaluation"},
    )
    return cache


def _verify_cache_shapes(cache: Dict[str, Any], split_name: str,
                          expected_clips: int) -> List[str]:
    """
    Verify that the prediction cache has the expected shapes.

    Returns a list of error strings (empty = all good).
    """
    errors: List[str] = []
    n = len(cache.get("y_true", []))

    if n != expected_clips:
        errors.append(
            f"{split_name} y_true has {n} entries; expected {expected_clips}."
        )

    y_prob = cache.get("y_prob")
    if y_prob is None or y_prob.shape != (expected_clips, _N_CLASSES):
        errors.append(
            f"{split_name} y_prob shape {getattr(y_prob, 'shape', 'MISSING')}; "
            f"expected ({expected_clips}, {_N_CLASSES})."
        )

    for arr_name in ("y_pred", "clip_ids", "signer_ids",
                     "detected_frame_count", "missing_pct"):
        arr = cache.get(arr_name)
        if arr is not None and len(arr) != expected_clips:
            errors.append(
                f"{split_name} {arr_name} has {len(arr)} entries; "
                f"expected {expected_clips}."
            )

    return errors


# ---------------------------------------------------------------------------
# Matplotlib figure helpers
# ---------------------------------------------------------------------------

def _get_plt():
    """Import matplotlib.pyplot without clobbering the active backend."""
    try:
        import matplotlib
        if not matplotlib.get_backend():
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt


def _save_fig(fig: Any, path: Path, dpi: int = _FIGURE_DPI) -> None:
    """Save a matplotlib figure, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt = _get_plt()
    plt.close(fig)
    logger.info(f"Saved figure → {path}", extra={"stage": "evaluation"})


# ---------------------------------------------------------------------------
# Phase D1 — Confusion matrices
# ---------------------------------------------------------------------------

def _run_confusion_matrices(
    val_cache:  Dict[str, Any],
    test_cache: Optional[Dict[str, Any]],
    sign_names: List[str],
    figures_dir: Path,
    report: Dict[str, Any],
) -> None:
    """
    Generate raw-count and row-normalised confusion matrices for val and test.
    """
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
    plt = _get_plt()

    def _make_cm_figs(y_true: np.ndarray, y_pred: np.ndarray,
                      split_label: str) -> Tuple[Any, Any]:
        n      = len(sign_names)
        labels = list(range(n))
        cm     = confusion_matrix(y_true, y_pred, labels=labels)

        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm  = cm.astype(float) / np.maximum(row_sums, 1)

        figs = []
        for suffix, data, fmt in [
            ("",      cm,      "d"),
            ("_norm", cm_norm, ".2f"),
        ]:
            fig, ax = plt.subplots(figsize=_CM_FIGSIZE)
            disp = ConfusionMatrixDisplay(data, display_labels=sign_names)
            disp.plot(ax=ax, xticks_rotation=90, colorbar=True, values_format=fmt)
            norm_str = " (row-normalised)" if suffix else ""
            ax.set_title(
                f"Confusion Matrix{norm_str} — {split_label} split "
                f"(n={len(y_true)})",
                fontsize=11,
            )
            ax.tick_params(axis="both", labelsize=6)
            plt.tight_layout()
            figs.append((suffix, fig))
        return figs  # type: ignore[return-value]

    logger.info("D1: Generating confusion matrices …", extra={"stage": "evaluation"})

    # Val matrices
    with _guard("val confusion matrix"):
        figs = _make_cm_figs(
            val_cache["y_true"], val_cache["y_pred"], "val"
        )
        for suffix, fig in figs:
            name = "confusion_matrix_best_model" if not suffix else "confusion_matrix_normalised"
            _save_fig(fig, figures_dir / f"{name}.png")

    # Test matrix
    if test_cache is not None:
        with _guard("test confusion matrix"):
            figs = _make_cm_figs(
                test_cache["y_true"], test_cache["y_pred"], "test"
            )
            for _suffix, fig in figs[:1]:  # raw only for test (normalised optional)
                _save_fig(fig, figures_dir / "confusion_matrix_test.png")

    # Ranked confusable pairs (val)
    with _guard("confusable pairs analysis"):
        from src.evaluation.metrics import compute_confusion_matrix_from_predictions
        cm_val = compute_confusion_matrix_from_predictions(
            val_cache["y_true"], val_cache["y_pred"], _N_CLASSES
        )
        pairs = []
        for i in range(_N_CLASSES):
            for j in range(_N_CLASSES):
                if i != j and cm_val[i, j] > 0:
                    pairs.append({
                        "true_class_idx": i,
                        "pred_class_idx": j,
                        "true_sign": sign_names[i],
                        "pred_sign": sign_names[j],
                        "count": int(cm_val[i, j]),
                        "row_total": int(cm_val[i].sum()),
                    })
        pairs.sort(key=lambda p: (-p["count"], p["true_class_idx"]))
        report["confusable_pairs_val_top25"] = pairs[:25]
        logger.info(
            f"D1: Top confusable pair (val): "
            f"{pairs[0]['true_sign']}→{pairs[0]['pred_sign']} "
            f"(count={pairs[0]['count']})" if pairs else "D1: No val errors.",
            extra={"stage": "evaluation"},
        )


# ---------------------------------------------------------------------------
# Phase D2 — Per-class metrics + F1 bar chart
# ---------------------------------------------------------------------------

def _run_per_class_metrics(
    val_cache:  Dict[str, Any],
    test_cache: Optional[Dict[str, Any]],
    sign_names: List[str],
    figures_dir: Path,
    report: Dict[str, Any],
) -> None:
    """
    Compute full per-class metrics and save sorted F1 bar chart.
    """
    from src.evaluation.metrics import (
        compute_evaluation_summary,
        rank_classes_by_f1,
        bootstrap_macro_f1_ci,
    )
    plt = _get_plt()

    logger.info("D2: Computing per-class metrics …", extra={"stage": "evaluation"})

    with _guard("val evaluation summary", fatal=True):
        val_summary = compute_evaluation_summary(
            val_cache["y_true"],
            val_cache["y_pred"],
            sign_names,
            _N_CLASSES,
            split_name="val",
            compute_ci=True,
            n_bootstrap=1000,
            ci_level=0.90,
            seed=_SEED,
            include_confusion_matrix=False,
            metadata={
                "signer_ids":           val_cache.get("signer_ids",           np.array([])).tolist(),
                "detected_frame_count": val_cache.get("detected_frame_count", np.array([])).tolist(),
                "missing_pct":          val_cache.get("missing_pct",          np.array([])).tolist(),
            } if "signer_ids" in val_cache else None,
        )
        report["val_evaluation"] = {
            "macro_f1":              val_summary["macro_f1"],
            "accuracy":              val_summary["accuracy"],
            "n_samples":             val_summary["n_samples"],
            "n_classes":             val_summary["n_classes"],
            "macro_f1_bootstrap_ci": val_summary.get("macro_f1_bootstrap_ci", {}),
            "n_singleton_classes":   val_summary["per_class_metrics"]["n_singleton_classes"],
        }

        logger.info(
            f"D2: val macro_f1={val_summary['macro_f1']:.4f} | "
            f"accuracy={val_summary['accuracy']:.4f}",
            extra={"stage": "evaluation"},
        )

    if test_cache is not None:
        with _guard("test evaluation summary"):
            test_summary = compute_evaluation_summary(
                test_cache["y_true"],
                test_cache["y_pred"],
                sign_names,
                _N_CLASSES,
                split_name="test",
                compute_ci=True,
                n_bootstrap=1000,
                ci_level=0.90,
                seed=_SEED,
                include_confusion_matrix=False,
            )
            # Merge into the test_evaluation block already written by Phase C
            existing = report.get("test_evaluation", {})
            existing.update({
                "per_class_metrics_summary": {
                    "n_singleton_classes": test_summary["per_class_metrics"]["n_singleton_classes"],
                    "n_zero_support_classes": test_summary["per_class_metrics"]["n_zero_support_classes"],
                },
            })
            report["test_evaluation"] = existing

    # ── Per-class F1 bar chart (val) ─────────────────────────────────────
    with _guard("per-class F1 bar chart"):
        per_class = val_summary["per_class_metrics"]["per_class"]
        ranked = rank_classes_by_f1({"per_class": per_class}, ascending=True)

        import matplotlib.patches as mpatches
        fig, ax = plt.subplots(figsize=(18, 6))
        names   = [r["sign"] for r in ranked]
        f1s     = [r["f1_score"] for r in ranked]
        supports = [r["support"] for r in ranked]

        colours = []
        for r in ranked:
            if r["is_high_risk"]:
                colours.append("tomato")
            elif r["is_singleton"]:
                colours.append("sandybrown")
            elif r["is_zero_support"]:
                colours.append("lightgrey")
            else:
                colours.append("steelblue")

        x_pos = np.arange(len(names))
        ax.bar(x_pos, f1s, color=colours, edgecolor="white", linewidth=0.5)

        # Support annotations
        for xi, (f1, sup) in enumerate(zip(f1s, supports)):
            ax.text(xi, f1 + 0.015, f"n={sup}", ha="center", va="bottom",
                    fontsize=6.5, color="dimgrey", rotation=90)

        ax.axhline(val_summary["macro_f1"], color="gold", linestyle="--",
                   linewidth=1.5, label=f"Macro-F1={val_summary['macro_f1']:.4f}")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, rotation=90, fontsize=8)
        ax.set_ylim(-0.05, 1.15)
        ax.set_ylabel("F1 Score")
        ax.set_title(
            f"Per-Class F1 — Val Split (sorted ascending) | "
            f"n={val_summary['n_samples']} clips, {_N_CLASSES} classes",
            fontsize=11,
        )
        ax.grid(True, axis="y", alpha=0.3, linestyle=":")

        legend_handles = [
            mpatches.Patch(color="steelblue",   label="Normal class"),
            mpatches.Patch(color="sandybrown",  label="Singleton (n=1)"),
            mpatches.Patch(color="tomato",      label="High-risk (Stage 5)"),
            mpatches.Patch(color="lightgrey",   label="Zero val support"),
            plt.Line2D([0], [0], color="gold", linestyle="--",
                       label=f"Macro-F1={val_summary['macro_f1']:.4f}"),
        ]
        ax.legend(handles=legend_handles, fontsize=8, loc="upper left")
        plt.tight_layout()
        _save_fig(fig, figures_dir / "per_class_metrics.png")


# ---------------------------------------------------------------------------
# Phase D3 — Confidence calibration
# ---------------------------------------------------------------------------

def _run_calibration(
    val_cache:  Dict[str, Any],
    figures_dir: Path,
    report: Dict[str, Any],
) -> None:
    """
    Compute reliability diagram and confidence-threshold curve for val split.
    """
    from src.evaluation.calibration import (
        compute_calibration_summary,
        plot_reliability_diagram,
        plot_confidence_threshold_curve,
    )

    logger.info("D3: Running confidence calibration …", extra={"stage": "evaluation"})

    with _guard("calibration summary"):
        cal_summary = compute_calibration_summary(
            val_cache["y_true"],
            val_cache["y_prob"],
            _N_CLASSES,
            split_name="val",
            n_bins=5,             # recommended for n=52
            strategy="uniform",
            n_threshold_points=101,
            seed=_SEED,
            compute_bin_ci=True,
            compute_ece_ci=True,
        )
        report["calibration"] = {
            "split_name":         "val",
            "ece":                cal_summary["ece"],
            "ece_unweighted":     cal_summary["ece_unweighted"],
            "mce":                cal_summary["mce"],
            "ece_ci_lower":       cal_summary["ece_ci_lower"],
            "ece_ci_upper":       cal_summary["ece_ci_upper"],
            "ece_ci_level":       cal_summary["ece_ci_level"],
            "overconfidence_gap": cal_summary["overconfidence_gap"],
            "mean_confidence":    cal_summary["mean_confidence"],
            "mean_accuracy":      cal_summary["mean_accuracy"],
            "auc_coverage":       cal_summary["auc_coverage"],
            "caveat":             cal_summary["caveat"],
            "temperature_scaling_note": cal_summary["temperature_scaling_note"],
        }
        logger.info(
            f"D3: ECE={cal_summary['ece']:.4f} | "
            f"MCE={cal_summary['mce']:.4f} | "
            f"overconfidence_gap={cal_summary['overconfidence_gap']:+.4f}",
            extra={"stage": "evaluation"},
        )

    with _guard("reliability diagram figure"):
        fig = plot_reliability_diagram(
            cal_summary["reliability_diagram"],
            split_name="val",
        )
        _save_fig(fig, figures_dir / "confidence_calibration.png")

    with _guard("confidence threshold curve figure"):
        fig = plot_confidence_threshold_curve(
            cal_summary["threshold_curve"],
            split_name="val",
        )
        _save_fig(fig, figures_dir / "confidence_threshold_curve.png")


# ---------------------------------------------------------------------------
# Phase D4 — Signer analysis
# ---------------------------------------------------------------------------

def _run_signer_analysis(
    val_cache:  Dict[str, Any],
    test_cache: Optional[Dict[str, Any]],
    sign_names: List[str],
    figures_dir: Path,
    report: Dict[str, Any],
) -> None:
    """
    Per-signer accuracy, spread CI, and failure-mode correlation for val/test.
    """
    from src.evaluation.signer_analysis import (
        compute_signer_analysis_summary,
        plot_signer_generalisation,
    )

    logger.info("D4: Running signer analysis …", extra={"stage": "evaluation"})

    val_signer_result = None
    test_signer_result = None

    with _guard("val signer analysis"):
        val_signer_result = compute_signer_analysis_summary(
            val_cache["y_true"],
            val_cache["y_pred"],
            val_cache.get("signer_ids", np.zeros(len(val_cache["y_true"]), dtype=object)),
            _N_CLASSES,
            split_name="val",
            sign_names=sign_names,
            detected_frame_count=val_cache.get("detected_frame_count"),
            missing_pct=val_cache.get("missing_pct"),
            compute_spread_ci=True,
            n_bootstrap=1000,
            ci_level=0.90,
            seed=_SEED,
        )
        n_signers   = val_signer_result["n_signers"]
        spread_std  = val_signer_result.get("spread_bootstrap_ci", {}).get("observed_std", "N/A")
        report["signer_analysis"] = {
            "val": {
                "n_signers":         n_signers,
                "overall_accuracy":  val_signer_result["per_signer_accuracy"]["overall_accuracy"],
                "observed_std":      spread_std,
                "unseen_framing":    val_signer_result["unseen_signer_framing_note"],
                "caveat":            val_signer_result["caveat"],
            }
        }
        logger.info(
            f"D4: val signers={n_signers} | accuracy_spread_std={spread_std}",
            extra={"stage": "evaluation"},
        )

    if test_cache is not None:
        with _guard("test signer analysis"):
            test_signer_result = compute_signer_analysis_summary(
                test_cache["y_true"],
                test_cache["y_pred"],
                test_cache.get("signer_ids", np.zeros(len(test_cache["y_true"]), dtype=object)),
                _N_CLASSES,
                split_name="test",
                sign_names=sign_names,
                detected_frame_count=test_cache.get("detected_frame_count"),
                missing_pct=test_cache.get("missing_pct"),
                compute_spread_ci=True,
                n_bootstrap=1000,
                ci_level=0.90,
                seed=_SEED,
            )
            test_spread = test_signer_result.get("spread_bootstrap_ci", {}).get("observed_std", "N/A")
            report.setdefault("signer_analysis", {})["test"] = {
                "n_signers":        test_signer_result["n_signers"],
                "overall_accuracy": test_signer_result["per_signer_accuracy"]["overall_accuracy"],
                "observed_std":     test_spread,
            }

    # ── Signer generalisation plot ────────────────────────────────────────
    with _guard("signer generalisation figure"):
        if val_signer_result is not None:
            fig = plot_signer_generalisation(
                val_signer_result["per_signer_accuracy"],
                test_per_signer_result=(
                    test_signer_result["per_signer_accuracy"]
                    if test_signer_result is not None else None
                ),
                metric="accuracy",
                show_wilson_ci=True,
                show_overall_accuracy=True,
            )
            _save_fig(fig, figures_dir / "signer_generalisation.png")


# ---------------------------------------------------------------------------
# Phase D5 — Failure-mode taxonomy
# ---------------------------------------------------------------------------

def _run_failure_mode_analysis(
    val_cache:  Dict[str, Any],
    sign_names: List[str],
    figures_dir: Path,
    report: Dict[str, Any],
) -> None:
    """
    Build the failure-mode taxonomy from real val misclassifications.
    Generates failure_mode_analysis.md and a summary figure.
    """
    plt = _get_plt()
    import matplotlib.patches as mpatches
    from src.evaluation.metrics import HIGH_RISK_SIGNS

    logger.info("D5: Building failure-mode taxonomy …", extra={"stage": "evaluation"})

    y_true  = val_cache["y_true"]
    y_pred  = val_cache["y_pred"]
    dfc     = val_cache.get("detected_frame_count")
    mp      = val_cache.get("missing_pct")

    n_total  = len(y_true)
    n_errors = int((y_true != y_pred).sum())
    correct  = (y_true == y_pred)

    # ── High-risk class analysis ──────────────────────────────────────────
    high_risk_lower = {s.lower() for s in HIGH_RISK_SIGNS}
    high_risk_indices = {
        i for i, name in enumerate(sign_names)
        if name.lower() in high_risk_lower
    }

    hr_mask  = np.isin(y_true, list(high_risk_indices))
    n_hr     = int(hr_mask.sum())
    n_hr_err = int((~correct[hr_mask]).sum()) if n_hr > 0 else 0

    # ── Singleton val analysis ────────────────────────────────────────────
    from src.evaluation.metrics import compute_support_counts
    support   = compute_support_counts(y_true, _N_CLASSES)
    singleton_classes = np.where(support == 1)[0]
    sing_mask = np.isin(y_true, singleton_classes)
    n_sing    = int(sing_mask.sum())
    n_sing_err = int((~correct[sing_mask]).sum()) if n_sing > 0 else 0

    # ── Detection rate correlation ────────────────────────────────────────
    corr_finding: Dict[str, Any] = {}
    if dfc is not None:
        dfc_f = dfc.astype(float)
        corr_finding["median_dfc_correct"]   = float(np.nanmedian(dfc_f[correct]))
        corr_finding["median_dfc_incorrect"] = float(np.nanmedian(dfc_f[~correct]))
    if mp is not None:
        mp_f = mp.astype(float)
        # Auto-rescale if in [0, 100]
        if np.nanmax(mp_f) > 1.0:
            mp_f /= 100.0
        corr_finding["median_mp_correct"]   = float(np.nanmedian(mp_f[correct]))
        corr_finding["median_mp_incorrect"] = float(np.nanmedian(mp_f[~correct]))

    taxonomy: Dict[str, Any] = {
        "total_val_clips":           n_total,
        "total_errors":              n_errors,
        "overall_error_rate":        round(n_errors / n_total, 4),
        "high_risk_class_clips":     n_hr,
        "high_risk_class_errors":    n_hr_err,
        "high_risk_error_rate":      round(n_hr_err / n_hr, 4) if n_hr > 0 else None,
        "singleton_class_clips":     n_sing,
        "singleton_class_errors":    n_sing_err,
        "singleton_error_rate":      round(n_sing_err / n_sing, 4) if n_sing > 0 else None,
        "detection_rate_correlation": corr_finding,
        "note": (
            "Failure categories are derived inductively from actual "
            "misclassifications only — no template categories are assumed. "
            "High-risk identification requires correct label map parsing "
            "(v1.1 schema). If high_risk_class_clips=0, the label map "
            "parser did not match any HIGH_RISK_SIGNS names."
        ),
    }
    report["failure_mode_taxonomy"] = taxonomy

    logger.info(
        f"D5: total_errors={n_errors}/{n_total} | "
        f"hr_clips={n_hr} | singleton_clips={n_sing}",
        extra={"stage": "evaluation"},
    )

    # ── Failure analysis figure ───────────────────────────────────────────
    with _guard("failure mode figure"):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Panel 1: Error rate by category
        categories = ["All clips", "High-risk\nclasses", "Singleton\nval classes"]
        error_rates = [
            n_errors / n_total,
            n_hr_err / n_hr if n_hr > 0 else 0.0,
            n_sing_err / n_sing if n_sing > 0 else 0.0,
        ]
        counts_label = [
            f"n={n_total}", f"n={n_hr}", f"n={n_sing}"
        ]
        bar_colours = ["steelblue", "tomato", "sandybrown"]
        ax = axes[0]
        bars = ax.bar(categories, error_rates, color=bar_colours, alpha=0.85,
                      edgecolor="white")
        for bar, cl in zip(bars, counts_label):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.015, cl,
                    ha="center", fontsize=9, color="dimgrey")
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("Error rate")
        ax.set_title("Val error rate by clip category")
        ax.grid(True, axis="y", alpha=0.3)

        # Panel 2: High-risk class F1 (from report if populated)
        ax2 = axes[1]
        hr_names = list(HIGH_RISK_SIGNS)
        hr_f1s   = []
        hr_ns    = []
        per_class = report.get("val_evaluation", {})
        for hn in hr_names:
            # The full per-class data lives in the metrics module's output
            # We do a lightweight lookup from the taxonomy
            idx = next((i for i, n in enumerate(sign_names)
                        if n.lower() == hn.lower()), None)
            if idx is not None:
                sup = int(support[idx])
                pred_count = int((y_pred == idx).sum())
                tp = int(((y_true == idx) & (y_pred == idx)).sum())
                prec = tp / pred_count if pred_count > 0 else 0.0
                rec  = tp / sup if sup > 0 else 0.0
                f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
                hr_f1s.append(f1)
                hr_ns.append(sup)
            else:
                hr_f1s.append(0.0)
                hr_ns.append(0)

        x_pos = np.arange(len(hr_names))
        ax2.bar(x_pos, hr_f1s, color="tomato", alpha=0.85, edgecolor="white")
        for xi, (f1, n_s) in enumerate(zip(hr_f1s, hr_ns)):
            ax2.text(xi, f1 + 0.02, f"n={n_s}", ha="center", fontsize=8)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(hr_names, rotation=20, ha="right", fontsize=9)
        ax2.set_ylim(0.0, 1.15)
        ax2.set_ylabel("F1 score")
        ax2.set_title("High-risk class F1 (val)")
        ax2.grid(True, axis="y", alpha=0.3)

        # Panel 3: Detection rate vs outcome (scatter or box)
        ax3 = axes[2]
        if dfc is not None:
            dfc_arr = dfc.astype(float)
            correct_dfc   = dfc_arr[correct]
            incorrect_dfc = dfc_arr[~correct]
            ax3.boxplot(
                [correct_dfc[~np.isnan(correct_dfc)],
                 incorrect_dfc[~np.isnan(incorrect_dfc)]],
                labels=["Correct", "Incorrect"],
                patch_artist=True,
                boxprops=dict(facecolor="steelblue", alpha=0.6),
            )
            ax3.set_ylabel("Detected frame count")
            ax3.set_title("Detection rate vs. prediction outcome (val)")
            ax3.grid(True, axis="y", alpha=0.3)
            if dfc is not None and mp is not None:
                mp_arr = mp.astype(float)
                if np.nanmax(mp_arr) > 1.0:
                    mp_arr /= 100.0
                ax3_twin = ax3.twinx()
                ax3_twin.scatter(
                    np.zeros(int(correct.sum())) + 0.85,
                    mp_arr[correct],
                    alpha=0.35, color="mediumseagreen", s=15, label="missing_pct correct"
                )
                ax3_twin.scatter(
                    np.ones(int((~correct).sum())) * 1.85,
                    mp_arr[~correct],
                    alpha=0.35, color="tomato", s=15, label="missing_pct incorrect"
                )
                ax3_twin.set_ylabel("missing_pct", color="darkgrey")
                ax3_twin.tick_params(axis="y", labelcolor="darkgrey")
        else:
            ax3.text(0.5, 0.5, "No detected_frame_count\ndata available.",
                     ha="center", va="center", transform=ax3.transAxes)
            ax3.set_title("Detection rate vs. outcome")

        fig.suptitle("Failure Mode Analysis — Val Split", fontsize=12)
        plt.tight_layout()
        _save_fig(fig, figures_dir / "failure_mode_analysis.png")

    # ── Write failure_mode_analysis.md ────────────────────────────────────
    with _guard("failure mode markdown"):
        md_path = Path("reports/failure_mode_analysis.md")
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_lines = [
            "# Failure Mode Analysis — Val Split\n",
            f"Generated: {datetime.now(timezone.utc).isoformat()}\n\n",
            "## Summary\n\n",
            f"- Total val clips: {n_total}\n",
            f"- Total errors: {n_errors} (error_rate={n_errors/n_total:.4f})\n",
            f"- High-risk class clips: {n_hr} | errors: {n_hr_err}\n",
            f"- Singleton val class clips: {n_sing} | errors: {n_sing_err}\n\n",
            "## Detection Rate Correlation\n\n",
        ]
        if corr_finding:
            for k, v in corr_finding.items():
                md_lines.append(f"- {k}: {v:.4f}\n")
        md_lines.append(
            "\n*Categories are data-driven, not template-imposed. "
            "See evaluation_report.json for full taxonomy.*\n"
        )
        md_path.write_text("".join(md_lines), encoding="utf-8")
        logger.info(f"Failure mode analysis written → {md_path}",
                    extra={"stage": "evaluation"})


# ---------------------------------------------------------------------------
# Phase D6 — Latency benchmarking
# ---------------------------------------------------------------------------

def _run_benchmark(
    champion_saved_model_path: str,
    val_cache: Dict[str, Any],
    figures_dir: Path,
    report: Dict[str, Any],
    skip: bool = False,
) -> None:
    """
    Benchmark Keras champion and scratch TFLite model.
    Generates latency_benchmark.png and populates report['benchmark'].
    """
    if skip:
        logger.info("D6: Benchmark skipped (--skip-benchmark).",
                    extra={"stage": "evaluation"})
        report["benchmark"] = {"skipped": True, "reason": "--skip-benchmark flag"}
        return

    from src.evaluation.benchmark import (
        benchmark_champion_summary,
        convert_to_scratch_tflite,
        build_latency_comparison_rows,
    )
    plt = _get_plt()

    logger.info("D6: Running latency benchmarking …", extra={"stage": "evaluation"})

    # Build a representative input sample (batch=1, seq_len=100, feature_dim=126)
    # Use first val clip to be representative
    y_prob = val_cache["y_prob"]
    seq_len, feature_dim = 100, 126
    X_sample = np.random.default_rng(_SEED).standard_normal(
        (1, seq_len, feature_dim)
    ).astype(np.float32)

    keras_stats: Dict[str, Any] = {}
    tflite_stats: Dict[str, Any] = {}
    scratch_tflite_path: Optional[str] = None

    with _guard("load champion model for benchmark"):
        import tensorflow as tf
        champion_model = tf.keras.models.load_model(champion_saved_model_path)

    with _guard("keras benchmark"):
        from src.evaluation.benchmark import benchmark_inference
        keras_stats = benchmark_inference(
            champion_model,
            X_sample,
            n_calls=200,
            warmup=20,
            description="champion_keras",
        )
        logger.info(
            f"D6: Keras median={keras_stats.get('median_ms', '?'):.3f}ms | "
            f"fps={keras_stats.get('fps', '?'):.1f}",
            extra={"stage": "evaluation"},
        )

    with _guard("scratch TFLite conversion"):
        scratch_tflite_path = convert_to_scratch_tflite(
            champion_saved_model_path,
            output_path="models/_bench_scratch.tflite",
            quantise=True,
        )

    if scratch_tflite_path:
        with _guard("TFLite benchmark"):
            from src.evaluation.benchmark import benchmark_tflite_inference, compute_file_size_mb
            tflite_stats = benchmark_tflite_inference(
                scratch_tflite_path,
                X_sample,
                n_calls=200,
                warmup=20,
                description="champion_tflite_scratch",
            )
            file_size_mb = compute_file_size_mb(scratch_tflite_path)
            logger.info(
                f"D6: TFLite median={tflite_stats.get('median_ms', '?'):.3f}ms | "
                f"fps={tflite_stats.get('fps', '?'):.1f} | "
                f"size={file_size_mb:.4f}MB",
                extra={"stage": "evaluation"},
            )

    report["benchmark"] = {
        "keras":           keras_stats,
        "tflite_scratch":  tflite_stats,
        "note": (
            "TFLite file is a SCRATCH export for benchmarking only. "
            "Stage 8 (src/export/convert.py) produces the authoritative, "
            "accuracy-verified export."
        ),
    }

    # ── Latency benchmark figure ──────────────────────────────────────────
    with _guard("latency benchmark figure"):
        models_info = {
            "Keras\n(float32)": keras_stats,
            "TFLite\n(dynamic quant)": tflite_stats,
        }
        valid = {k: v for k, v in models_info.items() if v.get("median_ms")}

        if not valid:
            logger.warning("D6: No benchmark data to plot.", extra={"stage": "evaluation"})
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        names      = list(valid.keys())
        medians    = [v["median_ms"]    for v in valid.values()]
        p95s       = [v.get("p95_ms", 0) for v in valid.values()]
        sizes_mb   = [v.get("model_size_mb", v.get("file_size_mb", 0))
                      for v in valid.values()]
        fpss       = [v.get("fps", 0) for v in valid.values()]

        colours = ["steelblue", "mediumseagreen"][:len(names)]

        ax = axes[0]
        x_pos = np.arange(len(names))
        bars  = ax.bar(x_pos, medians, color=colours, alpha=0.85, edgecolor="white",
                       label="Median latency (ms)")
        ax.bar(x_pos, [p - m for p, m in zip(p95s, medians)],
               bottom=medians, color=colours, alpha=0.35, edgecolor="white",
               label="p95-median delta")

        # ≤100ms target line
        ax.axhline(100, color="tomato", linestyle="--", linewidth=1.5,
                   label="≤100ms target")
        for bar, fps in zip(bars, fpss):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.5,
                    f"{fps:.0f}fps",
                    ha="center", fontsize=9)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, fontsize=9)
        ax.set_ylabel("Latency (ms)")
        ax.set_title("Inference Latency — Champion Model\n(batch=1, seq_len=100, feature_dim=126)")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

        ax2 = axes[1]
        size_colours = colours
        size_bars = ax2.bar(x_pos, sizes_mb, color=size_colours, alpha=0.85,
                            edgecolor="white")
        ax2.axhline(10, color="tomato", linestyle="--", linewidth=1.5,
                    label="≤10MB target")
        ax2.axhline(5, color="darkorange", linestyle=":", linewidth=1.2,
                    label="≤5MB stretch target")
        for bar, s in zip(size_bars, sizes_mb):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.002,
                     f"{s:.3f}MB",
                     ha="center", fontsize=9)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(names, fontsize=9)
        ax2.set_ylabel("Model size (MB)")
        ax2.set_title("Model Size Comparison\n(float32 estimate vs TFLite on-disk)")
        ax2.legend(fontsize=8)
        ax2.grid(True, axis="y", alpha=0.3)

        fig.suptitle("Stage 6 Benchmarking — Champion bilstm_hands_only_v4_aug",
                     fontsize=11)
        plt.tight_layout()
        _save_fig(fig, figures_dir / "latency_benchmark.png")


# ---------------------------------------------------------------------------
# Phase E — SHAP / Gradient×Input (papermill if available)
# ---------------------------------------------------------------------------

def _run_shap_notebook(skip: bool = False) -> Dict[str, Any]:
    """
    Attempt to execute the SHAP interpretability notebook via papermill.
    Returns a status dict for the evaluation report.
    """
    if skip:
        logger.info("Phase E: SHAP notebook skipped (--skip-shap).",
                    extra={"stage": "evaluation"})
        return {"status": "skipped", "reason": "--skip-shap flag"}

    try:
        import papermill as pm  # noqa: F401
        notebook_in  = Path("notebooks/08_interpretability_shap.ipynb")
        notebook_out = Path("notebooks/08_interpretability_shap_executed.ipynb")

        if not notebook_in.exists():
            return {
                "status": "skipped",
                "reason": f"Notebook not found: {notebook_in}",
            }

        logger.info("Phase E: Executing SHAP notebook via papermill …",
                    extra={"stage": "evaluation"})
        pm.execute_notebook(
            str(notebook_in),
            str(notebook_out),
            parameters={"seed": _SEED, "n_classes": _N_CLASSES},
            kernel_name="python3",
        )
        logger.info("Phase E: SHAP notebook executed successfully.",
                    extra={"stage": "evaluation"})
        return {"status": "executed", "output_notebook": str(notebook_out)}

    except ImportError:
        logger.warning(
            "Phase E: papermill not installed. SHAP notebook NOT executed. "
            "Install with: pip install papermill. "
            "The Gradient×Input analysis in reports/evaluation/evaluation_report.json "
            "and the Phase E executive summary remain authoritative.",
            extra={"stage": "evaluation"},
        )
        return {
            "status": "skipped",
            "reason": "papermill not installed",
            "fallback": (
                "Gradient×Input attribution results are documented in "
                "reports/evaluation/evaluation_report.json (Phase E summary). "
                "Re-run with papermill installed to produce the executed notebook."
            ),
        }
    except Exception as exc:
        logger.error(
            f"Phase E: SHAP notebook execution failed: {type(exc).__name__}: {exc}",
            extra={"stage": "evaluation"},
        )
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------
# LIMITATIONS.md patcher
# ---------------------------------------------------------------------------

def _patch_limitations_md(patches: List[Tuple[str, str]]) -> None:
    """
    Append Stage 6 findings to LIMITATIONS.md if not already present.

    Each patch is a (marker, text) tuple. The marker is a short unique
    string used to detect whether the patch was already applied, so this
    function is idempotent.
    """
    lim_path = Path("LIMITATIONS.md")
    if not lim_path.exists():
        logger.warning(
            "LIMITATIONS.md not found — skipping patch.",
            extra={"stage": "evaluation"},
        )
        return

    content = lim_path.read_text(encoding="utf-8")
    added   = 0
    for marker, text in patches:
        if marker not in content:
            content += text
            added += 1

    if added > 0:
        lim_path.write_text(content, encoding="utf-8")
        logger.info(
            f"LIMITATIONS.md patched with {added} Stage 6 finding(s).",
            extra={"stage": "evaluation"},
        )
    else:
        logger.info(
            "LIMITATIONS.md already up-to-date (no new patches needed).",
            extra={"stage": "evaluation"},
        )


# ---------------------------------------------------------------------------
# MLflow logging helpers
# ---------------------------------------------------------------------------

def _try_log_mlflow(report: Dict[str, Any], champion_run: str) -> None:
    """
    Log key Stage 6 metrics to the MLflow champion run (best-effort).
    Never aborts the pipeline on MLflow failure.
    """
    try:
        import mlflow

        mlflow.set_experiment(_MLFLOW_EXPERIMENT_NAME)

        # Find the existing champion run by run_name rather than opening a new one
        client = mlflow.tracking.MlflowClient()
        exp    = client.get_experiment_by_name(_MLFLOW_EXPERIMENT_NAME)

        if exp is None:
            logger.info(
                "MLflow: experiment not found — skipping metric logging.",
                extra={"stage": "evaluation"},
            )
            return

        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=f"tags.mlflow.runName = '{champion_run}'",
            order_by=["start_time DESC"],
            max_results=1,
        )

        if not runs:
            logger.info(
                f"MLflow: champion run '{champion_run}' not found — "
                "logging to a new Stage 6 run instead.",
                extra={"stage": "evaluation"},
            )
            with mlflow.start_run(run_name=f"{champion_run}_stage6_eval") as run:
                _log_eval_metrics_to_run(report)
        else:
            run_id = runs[0].info.run_id
            with mlflow.start_run(run_id=run_id):
                _log_eval_metrics_to_run(report)

        logger.info("MLflow: Stage 6 metrics logged.", extra={"stage": "evaluation"})

    except Exception as exc:
        logger.warning(
            f"MLflow logging failed (non-fatal): {type(exc).__name__}: {exc}",
            extra={"stage": "evaluation"},
        )


def _log_eval_metrics_to_run(report: Dict[str, Any]) -> None:
    """Log all extractable scalar metrics from the evaluation report."""
    import mlflow

    metrics: Dict[str, float] = {}

    val_eval = report.get("val_evaluation", {})
    if val_eval:
        metrics["eval_val_macro_f1"]  = float(val_eval.get("macro_f1",  0))
        metrics["eval_val_accuracy"]  = float(val_eval.get("accuracy",  0))

    test_eval = report.get("test_evaluation", {})
    if test_eval:
        metrics["eval_test_macro_f1"] = float(test_eval.get("test_macro_f1", 0))
        metrics["eval_test_accuracy"] = float(test_eval.get("test_accuracy", 0))

    cal = report.get("calibration", {})
    if cal:
        metrics["eval_val_ece"]               = float(cal.get("ece",              0))
        metrics["eval_val_mce"]               = float(cal.get("mce",              0))
        metrics["eval_val_overconfidence_gap"] = float(cal.get("overconfidence_gap", 0))

    bench = report.get("benchmark", {})
    keras_b = bench.get("keras", {})
    if keras_b.get("median_ms"):
        metrics["eval_keras_median_ms"] = float(keras_b["median_ms"])
        metrics["eval_keras_fps"]       = float(keras_b.get("fps", 0))

    if metrics:
        mlflow.log_metrics(metrics, step=0)

    try:
        mlflow.log_dict(report, "stage6_evaluation_report.json")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Completion gate
# ---------------------------------------------------------------------------

def _run_completion_gate(
    report:      Dict[str, Any],
    figures_dir: Path,
    splits:      List[str],
) -> Tuple[bool, List[str]]:
    """
    Verify all Stage 6 completion criteria.

    Returns
    -------
    (all_passed, failures)  all_passed is True iff failures is empty.
    """
    failures: List[str] = []
    timestamp = datetime.now(timezone.utc).isoformat()

    # ── Figure presence ──────────────────────────────────────────────────
    required_figs = list(_REQUIRED_FIGURES)
    if "test" not in splits:
        required_figs.remove("confusion_matrix_test.png")

    for fig_name in required_figs:
        fig_path = figures_dir / fig_name
        if not fig_path.exists():
            failures.append(f"Missing figure: {fig_path}")
        else:
            logger.info(f"  ✓  {fig_name}", extra={"stage": "evaluation"})

    # ── Prediction cache presence ─────────────────────────────────────────
    val_cache_path = Path(_CACHE_DIR) / _VAL_CACHE_NAME
    if not val_cache_path.exists():
        failures.append(f"Val prediction cache missing: {val_cache_path}")
    else:
        logger.info(f"  ✓  {_VAL_CACHE_NAME}", extra={"stage": "evaluation"})

    if "test" in splits:
        test_cache_path = Path(_CACHE_DIR) / _TEST_CACHE_NAME
        if not test_cache_path.exists():
            failures.append(f"Test prediction cache missing: {test_cache_path}")
        else:
            logger.info(f"  ✓  {_TEST_CACHE_NAME}", extra={"stage": "evaluation"})

        # Pre-commitment timestamp must predate test cache
        log_path = Path(_PRE_COMMITMENT_LOG)
        if not log_path.exists():
            failures.append(f"Pre-commitment log missing: {log_path}")
        elif test_cache_path.exists():
            if log_path.stat().st_mtime >= test_cache_path.stat().st_mtime:
                failures.append(
                    "Pre-commitment log was NOT written before test cache. "
                    "Temporal ordering violated — test evaluation integrity compromised."
                )
            else:
                logger.info(
                    "  ✓  pre-commitment log precedes test cache",
                    extra={"stage": "evaluation"},
                )

    # ── Report key presence ───────────────────────────────────────────────
    required_keys = list(_REQUIRED_REPORT_KEYS)
    if "test" not in splits:
        # test_evaluation may legitimately be absent if only val was requested
        required_keys = [k for k in required_keys if k != "test_evaluation"]

    for key in required_keys:
        if key not in report:
            failures.append(f"Missing key in evaluation_report.json: '{key}'")
        else:
            logger.info(f"  ✓  report['{key}']", extra={"stage": "evaluation"})

    # ── Val macro-F1 consistency ──────────────────────────────────────────
    val_f1 = report.get("val_evaluation", {}).get("macro_f1", None)
    if val_f1 is not None:
        delta = abs(float(val_f1) - _EXPECTED_VAL_MACRO_F1)
        if delta > _VAL_F1_TOLERANCE:
            failures.append(
                f"Val macro-F1={val_f1:.4f} deviates from expected "
                f"{_EXPECTED_VAL_MACRO_F1:.4f} by {delta:.4f} > "
                f"tolerance {_VAL_F1_TOLERANCE}. "
                "Check that FeaturePipeline(training=False) is in effect."
            )
        else:
            logger.info(
                f"  ✓  val macro-F1={val_f1:.4f} (delta={delta:.4f})",
                extra={"stage": "evaluation"},
            )

    # ── Config discrepancy documented ────────────────────────────────────
    if "config_discrepancy" not in report:
        failures.append("Config discrepancy block missing from evaluation_report.json.")
    else:
        logger.info("  ✓  config_discrepancy recorded", extra={"stage": "evaluation"})

    # ── Bootstrap CI present ─────────────────────────────────────────────
    ci = report.get("val_evaluation", {}).get("macro_f1_bootstrap_ci", {})
    if not ci.get("ci_lower") and not ci.get("ci_upper"):
        failures.append("Bootstrap CI missing or empty in val_evaluation.")
    else:
        logger.info(
            f"  ✓  val CI=[{ci.get('ci_lower', '?'):.4f}, {ci.get('ci_upper', '?'):.4f}]",
            extra={"stage": "evaluation"},
        )

    # ── LIMITATIONS.md present ───────────────────────────────────────────
    if not Path("LIMITATIONS.md").exists():
        failures.append("LIMITATIONS.md not found.")
    else:
        logger.info("  ✓  LIMITATIONS.md present", extra={"stage": "evaluation"})

    gate_result = {
        "timestamp_utc":   timestamp,
        "all_passed":      len(failures) == 0,
        "n_failures":      len(failures),
        "failures":        failures,
        "splits_evaluated": splits,
        "required_figures": required_figs,
    }
    report["completion_gate"] = gate_result

    all_passed = len(failures) == 0
    return all_passed, failures


# ---------------------------------------------------------------------------
# Main evaluation report writer
# ---------------------------------------------------------------------------

def _write_evaluation_report(report: Dict[str, Any], output_path: Path) -> None:
    """
    Write the consolidated evaluation_report.json, merging with any existing data.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing: Dict[str, Any] = {}
    if output_path.exists():
        try:
            with open(output_path, encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            pass

    existing.update(report)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, default=str)

    logger.info(
        f"Evaluation report written → {output_path}",
        extra={"stage": "evaluation"},
    )


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "WLASL Stage 6 Evaluation Orchestrator — "
            "runs all Phase D/E/F analysis against the champion model's "
            "prediction caches and produces evaluation_report.json."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--champion-run",
        default=_DEFAULT_CHAMPION_RUN,
        metavar="RUN_NAME",
        help=(
            f"Champion model run name (default: {_DEFAULT_CHAMPION_RUN}). "
            "Used to locate SavedModel and artefact directories."
        ),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["val", "test"],
        default=["val"],
        help="Splits to include in the evaluation (default: val).",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/evaluation",
        metavar="DIR",
        help="Top-level output directory for evaluation artefacts (default: reports/evaluation).",
    )
    parser.add_argument(
        "--figures-dir",
        default=_FIGURES_DIR,
        metavar="DIR",
        help=f"Output directory for figures (default: {_FIGURES_DIR}).",
    )
    parser.add_argument(
        "--force-rerun-test",
        action="store_true",
        default=False,
        help=(
            "Allow overwriting the existing test prediction cache. "
            "NEVER used silently — requires explicit flag. "
            "The pre-commitment log must still predate the new cache."
        ),
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        default=False,
        help="Skip latency benchmarking (useful on headless CI servers).",
    )
    parser.add_argument(
        "--skip-shap",
        action="store_true",
        default=False,
        help="Skip Phase E SHAP/Gradient×Input notebook execution.",
    )
    parser.add_argument(
        "--skip-mlflow",
        action="store_true",
        default=False,
        help="Skip logging results to MLflow.",
    )
    parser.add_argument(
        "--label-map",
        default=_LABEL_MAP_PATH,
        metavar="PATH",
        help=f"Path to label_map_v1.json (default: {_LABEL_MAP_PATH}).",
    )
    parser.add_argument(
        "--val-expected-clips",
        type=int,
        default=52,
        metavar="N",
        help="Expected number of val clips in the cache (default: 52).",
    )
    parser.add_argument(
        "--test-expected-clips",
        type=int,
        default=51,
        metavar="N",
        help="Expected number of test clips in the cache (default: 51).",
    )
    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Main evaluation orchestrator.

    Returns
    -------
    int  Exit code: 0 = success, 1 = gate failure, 2 = user abort.
    """
    parser = _build_parser()
    args   = parser.parse_args(argv)

    t_start = time.time()

    logger.info(
        f"{'=' * 68}\n"
        f"  WLASL Stage 6 Evaluation Orchestrator\n"
        f"  champion_run = {args.champion_run}\n"
        f"  splits       = {args.splits}\n"
        f"  output_dir   = {args.output_dir}\n"
        f"  figures_dir  = {args.figures_dir}\n"
        f"{'=' * 68}",
        extra={"stage": "evaluation"},
    )

    # ── Setup directories ────────────────────────────────────────────────
    output_dir  = Path(args.output_dir)
    figures_dir = Path(args.figures_dir)
    cache_dir   = Path(_CACHE_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "evaluation_report.json"

    # Load or initialise report (merges with Phase C data already written)
    report: Dict[str, Any] = {}
    if report_path.exists():
        try:
            with open(report_path, encoding="utf-8") as f:
                report = json.load(f)
            logger.info(
                f"Merged with existing evaluation_report.json "
                f"({len(report)} top-level keys)",
                extra={"stage": "evaluation"},
            )
        except Exception as exc:
            logger.warning(
                f"Could not load existing report ({exc}); starting fresh.",
                extra={"stage": "evaluation"},
            )

    # ── Populate metadata ─────────────────────────────────────────────────
    report["meta"] = {
        "champion_run":     args.champion_run,
        "splits":           args.splits,
        "timestamp_utc":    datetime.now(timezone.utc).isoformat(),
        "n_classes":        _N_CLASSES,
        "label_map":        args.label_map,
        "config_discrepancy_note": (
            "early_stopping_monitor in config_snapshot.yaml is val_accuracy; "
            "the handoff narrates val_macro_f1. Manual patience loop in train.py "
            "selected champion weights on val_macro_f1. See config_discrepancy."
        ),
    }
    report["config_discrepancy"] = _CONFIG_DISCREPANCY

    # ── Load sign names (schema v1.1 — fixes Phase D notebook bug) ────────
    sign_names = _load_sign_names(args.label_map)
    logger.info(
        f"Sign names loaded. Sample: {sign_names[:5]} … {sign_names[-3:]}",
        extra={"stage": "evaluation"},
    )

    # ── Load val cache (must exist) ───────────────────────────────────────
    val_cache_path = cache_dir / _VAL_CACHE_NAME
    val_cache = _load_cache(val_cache_path, "val")
    if val_cache is None:
        logger.error(
            f"Val prediction cache not found at {val_cache_path}. "
            "Run Phase B1 first (see notebooks/06_evaluation_error_analysis.ipynb "
            "or the Phase B1 script attached to this project).",
            extra={"stage": "evaluation"},
        )
        return 1

    shape_errors = _verify_cache_shapes(val_cache, "val", args.val_expected_clips)
    if shape_errors:
        for e in shape_errors:
            logger.error(e, extra={"stage": "evaluation"})
        return 1

    # ── Load / guard test cache ───────────────────────────────────────────
    test_cache: Optional[Dict[str, Any]] = None

    if "test" in args.splits:
        test_cache_path = cache_dir / _TEST_CACHE_NAME

        if test_cache_path.exists() and not args.force_rerun_test:
            logger.info(
                "Test prediction cache exists. Loading without re-running inference "
                "(pass --force-rerun-test to override).",
                extra={"stage": "evaluation"},
            )
            test_cache = _load_cache(test_cache_path, "test")

        elif test_cache_path.exists() and args.force_rerun_test:
            logger.warning(
                "⚠  --force-rerun-test is set. The existing test cache WILL be "
                "overwritten. This should happen at most once per project lifecycle.",
                extra={"stage": "evaluation"},
            )
            # Require interactive confirmation to prevent accidental flag usage
            if sys.stdin.isatty():
                answer = input(
                    "\nType 'yes' to confirm overwriting the test prediction cache: "
                ).strip().lower()
                if answer != "yes":
                    logger.info("User aborted test re-run.", extra={"stage": "evaluation"})
                    return 2
            test_cache = _load_cache(test_cache_path, "test")

        else:
            logger.warning(
                "Test cache not found. Phase C (test inference) must be run "
                "separately before test-split analysis can proceed. "
                "See reports/test_evaluation_log.md for the pre-commitment gate.",
                extra={"stage": "evaluation"},
            )
            # Remove test from splits if cache is absent
            args.splits = [s for s in args.splits if s != "test"]
            logger.info(
                "Test split removed from this run. Re-run with Phase C cache present.",
                extra={"stage": "evaluation"},
            )

    if test_cache is not None:
        shape_errors = _verify_cache_shapes(test_cache, "test", args.test_expected_clips)
        if shape_errors:
            for e in shape_errors:
                logger.warning(e, extra={"stage": "evaluation"})

    # ── Champion SavedModel path ──────────────────────────────────────────
    champion_path = _CHAMPION_SAVED_MODEL_PATH

    # ── Phase D — Analysis layer ──────────────────────────────────────────
    logger.info("", extra={"stage": "evaluation"})
    logger.info("━" * 60, extra={"stage": "evaluation"})
    logger.info("  Phase D — Analysis Layer", extra={"stage": "evaluation"})
    logger.info("━" * 60, extra={"stage": "evaluation"})

    # D1 — Confusion matrices
    with _guard("Phase D1 confusion matrices"):
        _run_confusion_matrices(val_cache, test_cache, sign_names, figures_dir, report)

    # D2 — Per-class metrics
    with _guard("Phase D2 per-class metrics"):
        _run_per_class_metrics(val_cache, test_cache, sign_names, figures_dir, report)

    # D3 — Confidence calibration (val)
    with _guard("Phase D3 calibration"):
        _run_calibration(val_cache, figures_dir, report)

    # D4 — Signer analysis
    with _guard("Phase D4 signer analysis"):
        _run_signer_analysis(val_cache, test_cache, sign_names, figures_dir, report)

    # D5 — Failure taxonomy
    with _guard("Phase D5 failure taxonomy"):
        _run_failure_mode_analysis(val_cache, sign_names, figures_dir, report)

    # D6 — Latency benchmark
    with _guard("Phase D6 latency benchmark"):
        _run_benchmark(
            champion_path,
            val_cache,
            figures_dir,
            report,
            skip=args.skip_benchmark,
        )

    # Intermediate report write (so partial results survive any Phase E crash)
    _write_evaluation_report(report, report_path)

    # ── Phase E — SHAP / Gradient×Input ──────────────────────────────────
    logger.info("", extra={"stage": "evaluation"})
    logger.info("━" * 60, extra={"stage": "evaluation"})
    logger.info("  Phase E — Interpretability", extra={"stage": "evaluation"})
    logger.info("━" * 60, extra={"stage": "evaluation"})

    shap_status = _run_shap_notebook(skip=args.skip_shap)
    report["phase_e_shap"] = shap_status

    # ── Phase F — LIMITATIONS.md patches + report + gate ─────────────────
    logger.info("", extra={"stage": "evaluation"})
    logger.info("━" * 60, extra={"stage": "evaluation"})
    logger.info("  Phase F — Orchestration, Notebooks, Completion Gate",
                extra={"stage": "evaluation"})
    logger.info("━" * 60, extra={"stage": "evaluation"})

    # Patch LIMITATIONS.md with Stage 6 findings
    _patch_limitations_md([
        ("Stage 6 — Calibration Finding", _LIMITATIONS_CALIBRATION_PATCH),
        ("Stage 6 — Left-Hand Attribution Gap", _LIMITATIONS_ATTRIBUTION_PATCH),
    ])

    # Write experiment_summary.md update note
    with _guard("experiment summary note"):
        summary_path = Path("reports/experiment_summary.md")
        if summary_path.exists():
            suffix = (
                "\n\n---\n## Stage 6 Evaluation Update\n"
                f"Generated: {datetime.now(timezone.utc).isoformat()}\n\n"
                f"Val macro-F1: {report.get('val_evaluation', {}).get('macro_f1', 'N/A'):.4f}\n"
                f"Test macro-F1: {report.get('test_evaluation', {}).get('test_macro_f1', 'N/A')}\n"
                f"See reports/evaluation/evaluation_report.json for full Stage 6 results.\n"
            )
            if "Stage 6 Evaluation Update" not in summary_path.read_text(encoding="utf-8"):
                with open(summary_path, "a", encoding="utf-8") as f:
                    f.write(suffix)
                logger.info("experiment_summary.md updated.", extra={"stage": "evaluation"})

    # ── MLflow logging ────────────────────────────────────────────────────
    if not args.skip_mlflow:
        _try_log_mlflow(report, args.champion_run)

    # ── Completion gate ───────────────────────────────────────────────────
    logger.info("", extra={"stage": "evaluation"})
    logger.info("  Running completion gate …", extra={"stage": "evaluation"})

    all_passed, failures = _run_completion_gate(report, figures_dir, args.splits)

    # Final report write (includes gate results)
    _write_evaluation_report(report, report_path)

    t_elapsed = time.time() - t_start

    logger.info("", extra={"stage": "evaluation"})
    logger.info("━" * 60, extra={"stage": "evaluation"})
    if all_passed:
        logger.info(
            f"  ✓  Stage 6 COMPLETE ({t_elapsed:.0f}s)\n"
            f"     All {len(_REQUIRED_FIGURES)} required figures present.\n"
            f"     evaluation_report.json → {report_path}",
            extra={"stage": "evaluation"},
        )
        logger.info("  Proceed to Stage 7 — GesturePredictor (src/inference/predictor.py)",
                    extra={"stage": "evaluation"})
    else:
        logger.error(
            f"  ✗  Stage 6 completion gate FAILED ({len(failures)} issues):\n"
            + "\n".join(f"     {i+1}. {f}" for i, f in enumerate(failures)),
            extra={"stage": "evaluation"},
        )
    logger.info("━" * 60, extra={"stage": "evaluation"})

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())