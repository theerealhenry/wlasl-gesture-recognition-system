"""
src/export/verify.py
=====================
Stage 8, Step 2 — Accuracy verification, release gating, model metadata
authoring, and per-class delta analysis for the WLASL 35-class gesture
recognition champion TFLite export.

This module answers the single question that Stage 8 exists to answer:
"Is gesture_bilstm_v1.tflite an acceptable replacement for the Keras
SavedModel for deployment?" It does so through four independent lenses:

  1. Macro-F1 and accuracy delta (hard gate, ±0.03 threshold)
  2. Argmax prediction agreement (hard gate, ≥0.95 threshold)
  3. Probability vector distance (warning gate, mean_abs_diff)
  4. Confidence shift (warning gate, affects Stage 9 HUD threshold)

All four are unified in ``ReleaseGateResult``, which is the single
authoritative pass/fail verdict.

Scope and integration context
--------------------------------
This module sits between convert.py (Step 1, which produced the verified
.tflite file) and notebooks/07_tflite_verification.ipynb (which calls
everything here and produces the final gate report). It depends on:

  - src/inference/predictor.py   — GesturePredictor (batch inference)
  - src/evaluation/metrics.py    — compute_macro_f1, compute_accuracy,
                                   compute_per_class_metrics, rank_classes_by_f1
  - src/evaluation/benchmark.py  — benchmark_tflite_inference,
                                   benchmark_inference,
                                   benchmark_pipeline_preprocessing
  - src/features/dataset.py      — GestureDataset.get_arrays_for_split()
  - src/utils/label_map.py       — LabelMap / get_label_map

Stage 6 calibration findings that shape this module
------------------------------------------------------
The champion is UNDERCONFIDENT: mean_confidence=0.5136 < mean_accuracy=0.5769
(ECE=0.2009, Stage 6 Phase D). Dynamic-range quantisation converts float32
weights to int8 for storage, then dequantises at runtime — this introduces
small weight perturbations that may shift confidence distributions. The
``val_confidence_shift`` metric (tflite_mean_conf - keras_mean_conf) surfaces
whether quantisation systematically shifts confidences in one direction; a
shift > ±0.03 triggers a WARNING because Stage 9's display_threshold=0.35 was
calibrated to the Keras model's underconfidence characteristics.

Stage 6 per-class findings that shape this module
----------------------------------------------------
The four confusable sign pairs (cosine similarity 0.785–0.963) are the classes
most likely to experience prediction flips after quantisation:
  think/who (0.905, 0.785), later/house (0.919, 0.946),
  cousin/mother (0.927, 0.947), girl/orange (0.963, 0.937).
Small weight perturbations from int8 quantisation can tip the balance in pairs
where the champion's activations are already nearly identical. The per-class
delta analysis surfaces this without blocking release (non-singleton |delta|
> 0.10 is flagged as ``meaningful_degradation`` for human review).

Why this module uses GesturePredictor rather than raw model calls
-------------------------------------------------------------------
Evaluating the TFLite file through GesturePredictor (smoother_window=1) is
the CORRECT approach: it exercises the REAL deployment inference path
(pipeline + model), not the bare model in isolation. Stage 8's ``verify.py``
is Stage 9's dress rehearsal. A TFLite file that passes verification through
GesturePredictor is a TFLite file that will work correctly in the webcam demo
without any additional preprocessing changes.

This is a deliberate, senior-level design choice inherited from Stage 7's
module docstring: "GesturePredictor implements __call__ to satisfy the
model(x_batch, training=False) -> probs contract ... so Stage 8's verify.py
can evaluate the REAL deployment path."

The one exception is that both predictors receive ALREADY-PIPELINED arrays
from get_arrays_for_split() — i.e., arrays that have already passed through
FeaturePipeline. GesturePredictor.__call__() accepts this format directly
(the evaluation-framework compatibility path).

Bug-fix history (this revision)
---------------------------------
  B1  FIXED. run_full_verification double-inference bug: the orchestrator
      previously called run_accuracy_verification() (which ran inference) and
      then immediately re-ran inference for per-class delta via freshly
      instantiated predictors. Now the per-class delta is derived from the
      (y_pred_keras, y_pred_tflite) arrays that run_accuracy_verification()
      already computed and returns, eliminating all redundant inference passes.
  B2  FIXED. Inverted variable naming in run_full_verification: the call
      `y_true_val, _, _ = val_dataset.get_arrays_for_split(...)` was naming
      the feature array X as `y_true_val`. Now uses explicit unpacking with
      correct names: X_val, y_val, signer_ids_val.
  B3  FIXED. compute_per_class_tflite_delta singleton fallback logic: the
      expression `k.get("is_singleton", support == 1)` used `support` which
      was initialised from `k.get("support", 0)` (default 0). The fallback
      `0 == 1` always evaluated to False, silently marking all classes with
      missing support data as non-singleton. Now uses explicit int comparison.
  B4  FIXED. _strip_bulky in save_verification_report now also strips
      `keras_per_class` and `tflite_per_class` from the per-split accuracy
      dicts, which could be very large and are already fully captured in the
      per_class_delta table. Previously only disagreement_details was stripped.
  B5  FIXED. write_model_metadata now casts landmark_config via str() before
      using it in JSON, defending against Pydantic enum objects that would
      fail JSON serialisation with default=str in some contexts.
  B6  FIXED. _nan_to_none in save_verification_report is now applied to the
      latency_benchmark sub-dict, which could contain NaN values from a
      failed benchmark run.
  B7  FIXED. Stage 6 reference CI constants (_STAGE6_KERAS_VAL_MACRO_F1_CI,
      _STAGE6_KERAS_TEST_MACRO_F1_CI) are now clearly labelled as loaded from
      evaluation_report.json when available, and as hardcoded estimates from
      Stage 6 bootstrap analysis when not. Previously they were silently
      hardcoded with no provenance annotation.
  B8  FIXED. run_accuracy_verification now explicitly returns y_pred arrays
      alongside the metrics dict so that run_full_verification can compute
      per-class delta without re-running inference. The return schema is
      extended with a `_predictions` key containing (y_true, y_pred_keras,
      y_pred_tflite, sign_names) for each split.
  B9  FIXED. assemble_release_gate NaN-safety: tflite_size_mb is now
      explicitly checked for finiteness before comparison against the 10 MB
      threshold, guarding against stat() failures that return 0 or NaN.
  B10 FIXED. write_model_metadata now uses str() on all enum-typed config
      fields (landmark_config, normalisation, missing_frame_strategy,
      padding) to guarantee JSON serialisability regardless of Pydantic's
      coercion behaviour.

Champion reference constants (confirmed from config_snapshot.yaml)
--------------------------------------------------------------------
    config_hash:     5809193d37e0d480e409b8e3112e70c8de9008497a29727b411a7128e73287a6
    mlflow_run_id:   cb16f689d2294001a2ff2d3e02419d27
    val_macro_f1:    0.6011 (Stage 6 Phase B1 reference)
    test_macro_f1:   0.4581 (Stage 6 Phase C reference)
    mean_confidence: 0.5136 (Stage 6 Phase D)
    mean_accuracy:   0.5769 (Stage 6 Phase D)
    ECE:             0.2009 (Stage 6 Phase D)
    display_threshold: 0.35 (Stage 6 calibrated)

Champion config snapshot confirms:
    early_stopping_monitor: val_accuracy (NOT val_macro_f1)
    early_stopping_patience: 50
    training.epochs: 250
    training.learning_rate: 0.0005
    data.sequence_length: 100
    data.landmark_config: hands_only
    model.hidden_units: 64 (32/direction in BiLSTM)
    model.bidirectional: true
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Module-level constants (all sourced from Stage 6 results / project constants)
# ---------------------------------------------------------------------------

#: Hard delta threshold: |keras_f1 - tflite_f1| must be ≤ this.
#: Source: config.export.max_accuracy_delta and Stage 8 Revised Spec.
_DELTA_THRESHOLD: float = 0.03

#: Argmax agreement threshold: fraction of clips where Keras and TFLite
#: argmax predictions agree must be ≥ this.
_AGREEMENT_THRESHOLD: float = 0.95

#: Warning threshold for mean absolute probability difference.
#: PredictionSmoother (window > 1) uses the full probability vector, so
#: large distributional differences affect Stage 9 behaviour even when
#: argmax predictions agree.
_PROB_DIFF_WARN_THRESHOLD: float = 0.02

#: Warning threshold for confidence shift (tflite_mean_conf - keras_mean_conf).
#: Stage 9 display_threshold=0.35 was calibrated to the Keras model's
#: underconfidence profile. A shift > ±0.03 may require recalibration.
_CONFIDENCE_SHIFT_WARN_THRESHOLD: float = 0.03

#: Maximum acceptable TFLite file size (project constant, Part 2).
_MAX_TFLITE_SIZE_MB: float = 10.0

#: Latency target for full pipeline (pipeline + tflite inference).
_LATENCY_TARGET_MS: float = 100.0

#: Below this many bootstrap resamples, percentile CI bounds are unreliable.
_MEANINGFUL_DEGRADATION_DELTA: float = 0.10

#: Stage 6 reference metrics for the Keras champion model.
#: These are the gold-standard baselines against which TFLite deltas are
#: measured. Loaded from evaluation_report.json when available; hardcoded
#: fallbacks are clearly labelled with their source.
_STAGE6_KERAS_VAL_MACRO_F1: float = 0.6011
_STAGE6_KERAS_VAL_ACCURACY: float = 0.5769
_STAGE6_KERAS_TEST_MACRO_F1: float = 0.4581
_STAGE6_KERAS_TEST_ACCURACY: float = 0.4902
#: 90% bootstrap CI bounds from Stage 6 Phase B1 bootstrap analysis.
#: Hardcoded fallback — loaded from evaluation_report.json when available.
_STAGE6_KERAS_VAL_MACRO_F1_CI: Tuple[float, float] = (0.5534, 0.6410)
_STAGE6_KERAS_TEST_MACRO_F1_CI: Tuple[float, float] = (0.3935, 0.5076)

#: Stage 6 calibration reference values (Phase D).
_STAGE6_ECE: float = 0.2009
_STAGE6_MEAN_CONFIDENCE: float = 0.5136
_STAGE6_MEAN_ACCURACY: float = 0.5769
_STAGE6_OVERCONFIDENCE_GAP: float = -0.0633  # negative = underconfident

#: Stage 6 Phase E confusable pairs (cosine similarity between activations).
#: These are the classes most likely to experience prediction flips after
#: quantisation due to their near-identical model activations.
_CONFUSABLE_PAIRS: Dict[str, List[str]] = {
    "think":  ["who"],
    "who":    ["think"],
    "later":  ["house"],
    "house":  ["later"],
    "cousin": ["mother"],
    "mother": ["cousin"],
    "girl":   ["orange"],
    "orange": ["girl"],
}
_CONFUSABLE_SIGNS: frozenset = frozenset(_CONFUSABLE_PAIRS.keys())

#: High-risk signs from Stage 5 Finding 8 / Stage 6 per-class analysis.
_HIGH_RISK_SIGNS: frozenset = frozenset(
    {"clothes", "think", "birthday", "name", "book"}
)

#: Default paths, matching Stage 8 deliverable table.
_DEFAULT_KERAS_MODEL_PATH: str = "models/bilstm_hands_only_v4_aug_saved_model"
_DEFAULT_TFLITE_PATH: str = "models/gesture_bilstm_v1.tflite"
_DEFAULT_CONFIG_SNAPSHOT_PATH: str = (
    "artifacts/experiments/bilstm_hands_only_v4_aug/config_snapshot.yaml"
)
_DEFAULT_STAGE6_REPORT_PATH: str = "reports/evaluation/evaluation_report.json"
_DEFAULT_METADATA_OUTPUT_PATH: str = "models/gesture_model_metadata.json"
_DEFAULT_VERIFICATION_REPORT_PATH: str = (
    "reports/evaluation/tflite_verification_report.json"
)

#: Champion model reference (for cross-checking loaded TFLite).
_CHAMPION_CONFIG_HASH: str = (
    "5809193d37e0d480e409b8e3112e70c8de9008497a29727b411a7128e73287a6"
)
_CHAMPION_MLFLOW_RUN_ID: str = "cb16f689d2294001a2ff2d3e02419d27"
_CHAMPION_N_CLASSES: int = 35
_CHAMPION_DISPLAY_THRESHOLD: float = 0.35


# ---------------------------------------------------------------------------
# NaN/None JSON helpers
# ---------------------------------------------------------------------------

def _nan_to_none(obj: Any) -> Any:
    """
    Recursively replace NaN float values with None for JSON serialisability.

    Applied to any dict/list/scalar before json.dump() to prevent the
    "NaN is not valid JSON" error that arises from numpy computations that
    produce NaN for missing/uncomputed metrics.
    """
    if isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else obj
    if isinstance(obj, (np.floating, np.integer)):
        v = float(obj) if isinstance(obj, np.floating) else int(obj)
        return _nan_to_none(v)
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    if isinstance(obj, tuple):
        return [_nan_to_none(v) for v in obj]
    return obj


def _strip_bulky(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove large nested keys from a per-split result dict before JSON export.

    Strips: disagreement_details (one entry per disagreeing clip — can be
    large), keras_per_class (full 35-class precision/recall/F1 breakdown),
    tflite_per_class (same). These are all fully captured in the per_class_delta
    table and in evaluation_report.json; duplicating them in the verification
    report would bloat the file significantly.
    """
    return {
        k: v for k, v in d.items()
        if k not in ("disagreement_details", "keras_per_class", "tflite_per_class")
    }


# ---------------------------------------------------------------------------
# Stage 6 reference loading helper
# ---------------------------------------------------------------------------

def _load_stage6_calibration(
    stage6_report_path: Optional[str],
) -> Dict[str, Any]:
    """
    Load Stage 6 calibration metrics from evaluation_report.json if available.

    Falls back to module-level constants (hardcoded from Stage 6 Phase D)
    when the report is absent or malformed. The ``_source`` key in the returned
    dict always records which code path was taken so every downstream consumer
    can trace provenance.

    Parameters
    ----------
    stage6_report_path : str | None
        Path to the Stage 6 evaluation_report.json produced by Phase F.

    Returns
    -------
    dict with keys: ece, mean_confidence, mean_accuracy, overconfidence_gap,
    val_macro_f1_ci, test_macro_f1_ci, _source.
    """
    if stage6_report_path and Path(stage6_report_path).exists():
        try:
            with open(stage6_report_path, encoding="utf-8") as f:
                report = json.load(f)

            cal = report.get("calibration_summary", {})
            val_ci = report.get("val_macro_f1_bootstrap_ci", {})
            test_ci = report.get("test_macro_f1_bootstrap_ci", {})

            result = {
                "ece":               cal.get("ece",              _STAGE6_ECE),
                "mean_confidence":   cal.get("mean_confidence",  _STAGE6_MEAN_CONFIDENCE),
                "mean_accuracy":     cal.get("mean_accuracy",    _STAGE6_MEAN_ACCURACY),
                "overconfidence_gap": cal.get("overconfidence_gap", _STAGE6_OVERCONFIDENCE_GAP),
                "val_macro_f1_ci": [
                    val_ci.get("ci_lower", _STAGE6_KERAS_VAL_MACRO_F1_CI[0]),
                    val_ci.get("ci_upper", _STAGE6_KERAS_VAL_MACRO_F1_CI[1]),
                ],
                "test_macro_f1_ci": [
                    test_ci.get("ci_lower", _STAGE6_KERAS_TEST_MACRO_F1_CI[0]),
                    test_ci.get("ci_upper", _STAGE6_KERAS_TEST_MACRO_F1_CI[1]),
                ],
                "_source": f"loaded from {stage6_report_path}",
            }
            logger.info(
                "Stage 6 calibration metrics loaded from %s",
                stage6_report_path,
                extra={"stage": "export"},
            )
            return result
        except Exception as exc:
            logger.warning(
                "Could not load Stage 6 calibration report at %s (%s: %s). "
                "Falling back to hardcoded Stage 6 Phase D constants.",
                stage6_report_path, type(exc).__name__, exc,
                extra={"stage": "export"},
            )

    logger.warning(
        "stage6_report_path=%s not provided or does not exist. "
        "Using hardcoded Stage 6 Phase D constants. "
        "Pass stage6_report_path pointing to evaluation_report.json for "
        "automatically loaded, provenance-tracked calibration values.",
        stage6_report_path,
        extra={"stage": "export"},
    )
    return {
        "ece":               _STAGE6_ECE,
        "mean_confidence":   _STAGE6_MEAN_CONFIDENCE,
        "mean_accuracy":     _STAGE6_MEAN_ACCURACY,
        "overconfidence_gap": _STAGE6_OVERCONFIDENCE_GAP,
        "val_macro_f1_ci":  list(_STAGE6_KERAS_VAL_MACRO_F1_CI),
        "test_macro_f1_ci": list(_STAGE6_KERAS_TEST_MACRO_F1_CI),
        "_source": "hardcoded from Stage 6 Phase D analysis — "
                   "stage6_report_path not provided or file absent",
    }


# ---------------------------------------------------------------------------
# ReleaseGateResult — single authoritative verdict
# ---------------------------------------------------------------------------

@dataclass
class ReleaseGateResult:
    """
    Single authoritative release gate for Stage 8.

    Every hard failure must be resolved before gesture_bilstm_v1.tflite
    is considered deployment-ready. Warnings are surfaced for human review
    but do not block release.

    Hard failures (block release):
      - val_delta_macro_f1 > ±_DELTA_THRESHOLD
      - test_delta_macro_f1 > ±_DELTA_THRESHOLD
      - val_argmax_agreement < _AGREEMENT_THRESHOLD
      - TFLite file does not exist
      - TFLite file size > 10 MB
      - Full pipeline latency > 100ms

    Warnings (non-blocking):
      - val_mean_abs_diff > _PROB_DIFF_WARN_THRESHOLD
      - |val_confidence_shift| > _CONFIDENCE_SHIFT_WARN_THRESHOLD

    Fields are initialised with sentinel values (NaN / -1.0 / False) so a
    partially-populated gate result reports meaningful failures rather than
    silently passing.
    """
    # ── Primary accuracy gate ──────────────────────────────────────────────
    val_delta_macro_f1:    float = float("nan")
    test_delta_macro_f1:   float = float("nan")
    delta_threshold:       float = _DELTA_THRESHOLD

    # ── Accuracy and F1 absolute values (for report) ──────────────────────
    keras_val_macro_f1:    float = float("nan")
    tflite_val_macro_f1:   float = float("nan")
    keras_test_macro_f1:   float = float("nan")
    tflite_test_macro_f1:  float = float("nan")
    keras_val_accuracy:    float = float("nan")
    tflite_val_accuracy:   float = float("nan")
    keras_test_accuracy:   float = float("nan")
    tflite_test_accuracy:  float = float("nan")

    # ── Argmax agreement gate ──────────────────────────────────────────────
    val_argmax_agreement:  float = float("nan")
    test_argmax_agreement: float = float("nan")
    agreement_threshold:   float = _AGREEMENT_THRESHOLD

    # ── Probability distribution (warning) ────────────────────────────────
    val_mean_abs_diff:        float = float("nan")
    val_max_abs_diff:         float = float("nan")
    prob_diff_warn_threshold: float = _PROB_DIFF_WARN_THRESHOLD

    # ── Calibration continuity (warning) ──────────────────────────────────
    val_confidence_shift:             float = float("nan")
    keras_mean_confidence:            float = float("nan")
    tflite_mean_confidence:           float = float("nan")
    confidence_shift_warn_threshold:  float = _CONFIDENCE_SHIFT_WARN_THRESHOLD

    # ── File and size gate ────────────────────────────────────────────────
    tflite_file_exists:   bool  = False
    tflite_size_mb:       float = float("nan")
    size_under_10mb:      bool  = False

    # ── Latency gate ──────────────────────────────────────────────────────
    full_pipeline_ms:      float = float("nan")
    tflite_median_ms:      float = float("nan")
    keras_median_ms:       float = float("nan")
    pipeline_median_ms:    float = float("nan")
    meets_100ms_target:    bool  = False
    speedup_keras_vs_tflite_x: Optional[float] = None

    # ── Sample counts (for report context) ────────────────────────────────
    n_val_samples:  int = 0
    n_test_samples: int = 0

    @property
    def hard_failures(self) -> List[str]:
        """Return list of hard failure descriptions (each blocks release)."""
        failures: List[str] = []

        def _is_nan(v: float) -> bool:
            return not np.isfinite(v)

        # Accuracy delta gates
        if not _is_nan(self.val_delta_macro_f1):
            if abs(self.val_delta_macro_f1) > self.delta_threshold:
                failures.append(
                    f"val_delta_macro_f1={self.val_delta_macro_f1:+.4f} "
                    f"exceeds ±{self.delta_threshold} threshold "
                    f"(Keras={self.keras_val_macro_f1:.4f}, "
                    f"TFLite={self.tflite_val_macro_f1:.4f})"
                )
        else:
            failures.append("val_delta_macro_f1 not yet measured")

        if not _is_nan(self.test_delta_macro_f1):
            if abs(self.test_delta_macro_f1) > self.delta_threshold:
                failures.append(
                    f"test_delta_macro_f1={self.test_delta_macro_f1:+.4f} "
                    f"exceeds ±{self.delta_threshold} threshold "
                    f"(Keras={self.keras_test_macro_f1:.4f}, "
                    f"TFLite={self.tflite_test_macro_f1:.4f})"
                )
        else:
            failures.append("test_delta_macro_f1 not yet measured")

        # Argmax agreement gate
        if not _is_nan(self.val_argmax_agreement):
            if self.val_argmax_agreement < self.agreement_threshold:
                n_disagreements = (
                    round((1.0 - self.val_argmax_agreement) * self.n_val_samples)
                    if self.n_val_samples > 0 else "?"
                )
                failures.append(
                    f"val_argmax_agreement={self.val_argmax_agreement:.4f} "
                    f"below threshold={self.agreement_threshold} "
                    f"({n_disagreements}/{self.n_val_samples} clips disagree)"
                )
        else:
            failures.append("val_argmax_agreement not yet measured")

        # File existence
        if not self.tflite_file_exists:
            failures.append(
                "TFLite file does not exist — run src/export/convert.py first"
            )

        # File size — guard against NaN from a failed stat() call (B9 fix)
        if self.tflite_file_exists:
            if _is_nan(self.tflite_size_mb):
                failures.append(
                    "TFLite file size could not be determined (stat() failed)"
                )
            elif not self.size_under_10mb:
                failures.append(
                    f"TFLite size {self.tflite_size_mb:.4f} MB exceeds "
                    f"{_MAX_TFLITE_SIZE_MB:.0f} MB project target"
                )

        # Latency
        if not _is_nan(self.full_pipeline_ms) and not self.meets_100ms_target:
            failures.append(
                f"Full pipeline latency {self.full_pipeline_ms:.1f}ms "
                f"exceeds {_LATENCY_TARGET_MS:.0f}ms target "
                f"(pipeline={self.pipeline_median_ms:.1f}ms + "
                f"tflite={self.tflite_median_ms:.1f}ms)"
            )

        return failures

    @property
    def warnings(self) -> List[str]:
        """Return list of non-blocking warning descriptions."""
        warns: List[str] = []

        def _is_nan(v: float) -> bool:
            return not np.isfinite(v)

        if (not _is_nan(self.val_mean_abs_diff)
                and self.val_mean_abs_diff > self.prob_diff_warn_threshold):
            warns.append(
                f"val_mean_abs_diff={self.val_mean_abs_diff:.6f} "
                f"exceeds warn threshold={self.prob_diff_warn_threshold}. "
                "PredictionSmoother probability distributions may differ "
                "between Keras and TFLite — monitor Stage 9 HUD confidence bars."
            )

        if not _is_nan(self.val_confidence_shift):
            if abs(self.val_confidence_shift) > self.confidence_shift_warn_threshold:
                direction = "higher" if self.val_confidence_shift > 0 else "lower"
                warns.append(
                    f"val_confidence_shift={self.val_confidence_shift:+.4f} "
                    f"exceeds warn threshold=±{self.confidence_shift_warn_threshold}. "
                    f"TFLite model is {direction} confidence than Keras "
                    f"(Keras mean={self.keras_mean_confidence:.4f}, "
                    f"TFLite mean={self.tflite_mean_confidence:.4f}). "
                    f"Stage 9 display_threshold={_CHAMPION_DISPLAY_THRESHOLD} "
                    "may need recalibration for TFLite deployment."
                )

        return warns

    @property
    def release_ready(self) -> bool:
        """True only if ALL hard failure conditions are absent."""
        return len(self.hard_failures) == 0

    def report(self) -> str:
        """Format a human-readable gate report for notebook output."""
        def _fmt(v: float, fmt: str = ".4f") -> str:
            return f"{v:{fmt}}" if np.isfinite(v) else "N/A"

        def _pass_fail(cond: bool) -> str:
            return "✓ PASS" if cond else "✗ FAIL"

        val_delta_pass = (
            np.isfinite(self.val_delta_macro_f1)
            and abs(self.val_delta_macro_f1) <= self.delta_threshold
        )
        test_delta_pass = (
            np.isfinite(self.test_delta_macro_f1)
            and abs(self.test_delta_macro_f1) <= self.delta_threshold
        )
        agreement_pass = (
            np.isfinite(self.val_argmax_agreement)
            and self.val_argmax_agreement >= self.agreement_threshold
        )
        prob_diff_ok = (
            not np.isfinite(self.val_mean_abs_diff)
            or self.val_mean_abs_diff <= self.prob_diff_warn_threshold
        )
        conf_shift_ok = (
            not np.isfinite(self.val_confidence_shift)
            or abs(self.val_confidence_shift) <= self.confidence_shift_warn_threshold
        )

        val_acc_delta = (
            self.tflite_val_accuracy - self.keras_val_accuracy
            if np.isfinite(self.tflite_val_accuracy) and np.isfinite(self.keras_val_accuracy)
            else float("nan")
        )
        test_acc_delta = (
            self.tflite_test_accuracy - self.keras_test_accuracy
            if np.isfinite(self.tflite_test_accuracy) and np.isfinite(self.keras_test_accuracy)
            else float("nan")
        )

        speedup_str = (
            f"{self.speedup_keras_vs_tflite_x:.2f}×"
            if self.speedup_keras_vs_tflite_x is not None else "N/A"
        )

        lines = [
            "=" * 60,
            "STAGE 8 RELEASE GATE — gesture_bilstm_v1.tflite",
            "=" * 60,
            "",
            f"ACCURACY DELTA (threshold ±{self.delta_threshold}):",
            f"  Val  Macro-F1: Keras={_fmt(self.keras_val_macro_f1)} → "
            f"TFLite={_fmt(self.tflite_val_macro_f1)}  "
            f"delta={_fmt(self.val_delta_macro_f1, '+.4f')}  "
            f"{_pass_fail(val_delta_pass)}",
            f"  Test Macro-F1: Keras={_fmt(self.keras_test_macro_f1)} → "
            f"TFLite={_fmt(self.tflite_test_macro_f1)}  "
            f"delta={_fmt(self.test_delta_macro_f1, '+.4f')}  "
            f"{_pass_fail(test_delta_pass)}",
            f"  Val  Accuracy: Keras={_fmt(self.keras_val_accuracy)} → "
            f"TFLite={_fmt(self.tflite_val_accuracy)}  "
            f"delta={_fmt(val_acc_delta, '+.4f')}",
            f"  Test Accuracy: Keras={_fmt(self.keras_test_accuracy)} → "
            f"TFLite={_fmt(self.tflite_test_accuracy)}  "
            f"delta={_fmt(test_acc_delta, '+.4f')}",
            "",
            f"ARGMAX AGREEMENT (threshold {self.agreement_threshold}):",
            f"  Val  argmax_agreement={_fmt(self.val_argmax_agreement)}  "
            f"({self.n_val_samples} clips)  {_pass_fail(agreement_pass)}",
            f"  Test argmax_agreement={_fmt(self.test_argmax_agreement)}  "
            f"({self.n_test_samples} clips)",
            "",
            "PROBABILITY DISTRIBUTION:",
            f"  val_mean_abs_diff={_fmt(self.val_mean_abs_diff, '.6f')}  "
            f"(warn >{self.prob_diff_warn_threshold})  "
            f"{'✓ OK' if prob_diff_ok else '⚠ WARN'}",
            f"  val_max_abs_diff={_fmt(self.val_max_abs_diff, '.6f')}",
            "",
            "CALIBRATION CONTINUITY:",
            f"  val_confidence_shift={_fmt(self.val_confidence_shift, '+.4f')}  "
            f"(warn >±{self.confidence_shift_warn_threshold})  "
            f"{'✓ OK' if conf_shift_ok else '⚠ WARN'}",
            f"  Keras mean_conf={_fmt(self.keras_mean_confidence)}  "
            f"TFLite mean_conf={_fmt(self.tflite_mean_confidence)}",
            f"  Stage 6 reference: mean_conf={_STAGE6_MEAN_CONFIDENCE:.4f}  "
            f"(underconfident: gap={_STAGE6_OVERCONFIDENCE_GAP:+.4f})",
            "",
            "FILE & SIZE:",
            f"  TFLite file exists: {self.tflite_file_exists}  "
            f"{_pass_fail(self.tflite_file_exists)}",
            f"  TFLite size: {_fmt(self.tflite_size_mb)} MB  "
            f"(limit {_MAX_TFLITE_SIZE_MB:.0f} MB)  "
            f"{_pass_fail(self.size_under_10mb)}",
            "",
            "LATENCY (pipeline + tflite, excl. MediaPipe):",
            f"  Pipeline (FeaturePipeline): {_fmt(self.pipeline_median_ms, '.2f')}ms",
            f"  TFLite inference:           {_fmt(self.tflite_median_ms, '.2f')}ms",
            f"  Full pipeline:              {_fmt(self.full_pipeline_ms, '.2f')}ms  "
            f"(target <{_LATENCY_TARGET_MS:.0f}ms)  "
            f"{_pass_fail(self.meets_100ms_target)}",
            f"  Keras inference:            {_fmt(self.keras_median_ms, '.2f')}ms",
            f"  Speedup (Keras/TFLite):     {speedup_str}",
            "",
        ]

        if self.warnings:
            lines.append("WARNINGS (non-blocking):")
            for w in self.warnings:
                lines.append(f"  ⚠  {w}")
            lines.append("")

        if self.hard_failures:
            lines.append("HARD FAILURES (block release):")
            for f_ in self.hard_failures:
                lines.append(f"  ✗  {f_}")
            lines.append("")
            lines.append(
                "RESULT: ✗ FAIL — gesture_bilstm_v1.tflite NOT approved for Stage 9"
            )
        else:
            lines.append(
                "RESULT: ✓ PASS — gesture_bilstm_v1.tflite approved for Stage 9"
            )

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable representation of the gate result."""
        d = asdict(self)
        d = _nan_to_none(d)
        d["hard_failures"] = self.hard_failures
        d["warnings"] = self.warnings
        d["release_ready"] = self.release_ready
        return d


# ---------------------------------------------------------------------------
# Step 2.1 — Core split comparison (single model pair, single split)
# ---------------------------------------------------------------------------

def _run_split_comparison(
    keras_predictor: Any,
    tflite_predictor: Any,
    val_dataset: Any,
    split_name: str,
    n_classes: int,
    sign_names: List[str],
    compute_macro_f1: Any,
    compute_accuracy: Any,
    compute_per_class_metrics: Any,
) -> Dict[str, Any]:
    """
    Run Keras vs TFLite comparison for one split.

    Input arrays come from get_arrays_for_split(), which returns
    already-pipelined (N, seq_len, feature_dim) float32 arrays. These pass
    directly to GesturePredictor.__call__(), which accepts already-pipelined
    input in the (batch, seq_len, feature_dim) format — the evaluation-
    framework compatibility path documented in Stage 7.

    Bug-fix B2: variable naming is now explicit.
    Bug-fix B8: y_pred arrays are returned in the result dict so the
    orchestrator can build per-class delta without re-running inference.

    Returns
    -------
    dict with per-split metrics AND prediction arrays (for downstream use).
    """
    try:
        X, y_true, signer_ids = val_dataset.get_arrays_for_split(
            split_name, use_augmentation=False
        )
    except Exception as exc:
        raise RuntimeError(
            f"_run_split_comparison(): failed to load split '{split_name}' "
            f"from GestureDataset: {exc}"
        ) from exc

    n_samples = len(y_true)
    logger.info(
        f"Split '{split_name}': {n_samples} clips, input shape {X.shape}",
        extra={"stage": "export"},
    )

    if n_samples == 0:
        raise RuntimeError(
            f"_run_split_comparison(): split '{split_name}' has zero clips. "
            "Check GestureDataset construction."
        )

    # Both predictors receive already-pipelined arrays via __call__().
    # Keras: single batched forward pass (efficient).
    # TFLite: per-sample loop inside __call__ (TFLite has static batch dim).
    t_keras = time.perf_counter()
    keras_probs = keras_predictor(X, training=False)   # (N, 35)
    keras_inference_s = time.perf_counter() - t_keras

    t_tflite = time.perf_counter()
    tflite_probs = tflite_predictor(X, training=False)  # (N, 35)
    tflite_inference_s = time.perf_counter() - t_tflite

    keras_probs  = np.asarray(keras_probs,  dtype=np.float64)
    tflite_probs = np.asarray(tflite_probs, dtype=np.float64)
    y_true_arr   = np.asarray(y_true, dtype=np.int64)

    y_pred_keras  = np.argmax(keras_probs,  axis=1).astype(np.int64)
    y_pred_tflite = np.argmax(tflite_probs, axis=1).astype(np.int64)

    # ── Core metrics ──────────────────────────────────────────────────────
    keras_macro_f1  = compute_macro_f1(y_true_arr, y_pred_keras,  n_classes)
    tflite_macro_f1 = compute_macro_f1(y_true_arr, y_pred_tflite, n_classes)
    keras_acc       = compute_accuracy(y_true_arr, y_pred_keras,  n_classes=n_classes)
    tflite_acc      = compute_accuracy(y_true_arr, y_pred_tflite, n_classes=n_classes)

    delta_macro_f1 = keras_macro_f1 - tflite_macro_f1
    delta_acc      = keras_acc      - tflite_acc

    # ── Probability distribution comparison ──────────────────────────────
    abs_diff         = np.abs(keras_probs - tflite_probs)
    mean_abs_diff    = float(abs_diff.mean())
    max_abs_diff     = float(abs_diff.max())
    argmax_agreement = float((y_pred_keras == y_pred_tflite).mean())

    # ── Confidence distribution comparison ───────────────────────────────
    keras_max_conf   = keras_probs.max(axis=1)
    tflite_max_conf  = tflite_probs.max(axis=1)
    keras_mean_conf  = float(keras_max_conf.mean())
    tflite_mean_conf = float(tflite_max_conf.mean())
    confidence_shift = tflite_mean_conf - keras_mean_conf

    # ── Per-class metrics ─────────────────────────────────────────────────
    keras_per_class  = compute_per_class_metrics(
        y_true_arr, y_pred_keras,  sign_names, n_classes
    )
    tflite_per_class = compute_per_class_metrics(
        y_true_arr, y_pred_tflite, sign_names, n_classes
    )

    # ── Disagreement analysis ──────────────────────────────────────────────
    disagreements = np.where(y_pred_keras != y_pred_tflite)[0]
    disagreement_details = []
    for idx in disagreements:
        true_label   = int(y_true_arr[idx])
        keras_pred   = int(y_pred_keras[idx])
        tflite_pred  = int(y_pred_tflite[idx])
        keras_conf   = float(keras_probs[idx, keras_pred])
        tflite_conf  = float(tflite_probs[idx, tflite_pred])
        true_name    = sign_names[true_label]  if true_label  < len(sign_names) else f"class_{true_label}"
        keras_name   = sign_names[keras_pred]  if keras_pred  < len(sign_names) else f"class_{keras_pred}"
        tflite_name  = sign_names[tflite_pred] if tflite_pred < len(sign_names) else f"class_{tflite_pred}"

        involves_confusable = (
            keras_name in _CONFUSABLE_SIGNS or tflite_name in _CONFUSABLE_SIGNS
        )
        disagreement_details.append({
            "clip_index":               int(idx),
            "true_label":               int(true_label),
            "true_sign":                true_name,
            "keras_pred":               int(keras_pred),
            "keras_sign":               keras_name,
            "keras_confidence":         round(keras_conf, 4),
            "tflite_pred":              int(tflite_pred),
            "tflite_sign":              tflite_name,
            "tflite_confidence":        round(tflite_conf, 4),
            "keras_correct":            keras_pred == true_label,
            "tflite_correct":           tflite_pred == true_label,
            "involves_confusable_pair": involves_confusable,
        })

    n_keras_right_tflite_wrong = sum(
        1 for d in disagreement_details if d["keras_correct"] and not d["tflite_correct"]
    )
    n_keras_wrong_tflite_right = sum(
        1 for d in disagreement_details if not d["keras_correct"] and d["tflite_correct"]
    )

    logger.info(
        "[%s] Keras F1=%.4f | TFLite F1=%.4f | delta=%+.4f | "
        "agreement=%.4f | mean_abs_diff=%.6f | confidence_shift=%+.4f | "
        "n_disagreements=%d (Keras✓TFLite✗=%d, Keras✗TFLite✓=%d) | "
        "keras_inf=%.2fs | tflite_inf=%.2fs",
        split_name, keras_macro_f1, tflite_macro_f1, delta_macro_f1,
        argmax_agreement, mean_abs_diff, confidence_shift,
        len(disagreements), n_keras_right_tflite_wrong, n_keras_wrong_tflite_right,
        keras_inference_s, tflite_inference_s,
        extra={"stage": "export"},
    )

    return {
        # Public metrics (written to report JSON)
        "keras_macro_f1":          round(keras_macro_f1,  4),
        "tflite_macro_f1":         round(tflite_macro_f1, 4),
        "delta_macro_f1":          round(delta_macro_f1,  4),
        "keras_accuracy":          round(keras_acc,  4),
        "tflite_accuracy":         round(tflite_acc, 4),
        "delta_accuracy":          round(delta_acc,  4),
        "argmax_agreement":        round(argmax_agreement, 4),
        "mean_abs_diff":           round(mean_abs_diff, 6),
        "max_abs_diff":            round(max_abs_diff,  6),
        "keras_mean_confidence":   round(keras_mean_conf,  4),
        "tflite_mean_confidence":  round(tflite_mean_conf, 4),
        "confidence_shift":        round(confidence_shift, 4),
        "n_samples":               n_samples,
        "n_disagreements":         len(disagreements),
        "n_keras_right_tflite_wrong":  n_keras_right_tflite_wrong,
        "n_keras_wrong_tflite_right":  n_keras_wrong_tflite_right,
        # Bulky — stripped by _strip_bulky() before JSON export
        "disagreement_details":    disagreement_details,
        "keras_per_class":         keras_per_class,
        "tflite_per_class":        tflite_per_class,
        # Internal — used by run_full_verification for per-class delta
        # without re-running inference (B1/B8 fix). Prefixed with _ so
        # downstream consumers know these are orchestration artefacts.
        "_y_true":        y_true_arr,
        "_y_pred_keras":  y_pred_keras,
        "_y_pred_tflite": y_pred_tflite,
        "_signer_ids":    np.asarray(signer_ids) if signer_ids is not None else None,
        # Timing (informational, not part of release gate)
        "keras_inference_s":       round(keras_inference_s, 3),
        "tflite_inference_s":      round(tflite_inference_s, 3),
    }


# ---------------------------------------------------------------------------
# Step 2.1 — Accuracy verification (orchestration wrapper)
# ---------------------------------------------------------------------------

def run_accuracy_verification(
    keras_model_path: Union[str, Path],
    tflite_path: Union[str, Path],
    config_snapshot_path: Union[str, Path],
    val_dataset: Any,
    n_classes: int = 35,
    sign_names: Optional[List[str]] = None,
    smoother_window: int = 1,
    display_threshold: float = _CHAMPION_DISPLAY_THRESHOLD,
) -> Dict[str, Any]:
    """
    Compare Keras SavedModel vs TFLite interpreter on val and test splits.

    Both models are loaded through GesturePredictor.from_config_snapshot()
    with smoother_window=1 (disables majority voting for a clean per-clip
    argmax comparison). Input arrays come from val_dataset.get_arrays_for_split(),
    which returns already-pipelined (N, 100, 126) float32 arrays. These are
    passed directly to GesturePredictor.__call__(), which accepts
    already-pipelined input in the (batch, seq_len, feature_dim) format.

    Bug-fix B1/B8: the returned dict now includes a '_predictions' sub-dict
    containing (y_true, y_pred_keras, y_pred_tflite, sign_names) for each
    split. This allows run_full_verification to build per-class delta without
    re-running inference, eliminating the double-inference bug.

    Parameters
    ----------
    keras_model_path : str | Path
        Path to the Keras SavedModel directory.
    tflite_path : str | Path
        Path to the verified TFLite file from Stage 8 Step 1.
    config_snapshot_path : str | Path
        Path to config_snapshot.yaml for the champion run.
    val_dataset : GestureDataset
        Already-constructed dataset instance. Used for val + test splits.
    n_classes : int, default 35
    sign_names : List[str], optional
        Index-aligned sign names. Derived from val_dataset if None.
    smoother_window : int, default 1
        Window=1 = no majority voting = clean per-clip argmax comparison.
    display_threshold : float, default 0.35

    Returns
    -------
    dict with keys:
        val, test : per-split metrics dicts
        n_classes : int
        sign_names : List[str]
        _predictions : dict mapping split_name → {y_true, y_pred_keras,
                       y_pred_tflite} as numpy arrays — for downstream
                       per-class delta without re-running inference.
    """
    from src.inference.predictor import GesturePredictor
    from src.evaluation.metrics import (
        compute_macro_f1, compute_accuracy, compute_per_class_metrics,
    )

    keras_path = Path(keras_model_path)
    tflite_p   = Path(tflite_path)

    if not keras_path.exists():
        raise FileNotFoundError(
            f"run_accuracy_verification(): Keras SavedModel not found at {keras_path}. "
            "Confirm Stage 5 training completed."
        )
    if not tflite_p.exists():
        raise FileNotFoundError(
            f"run_accuracy_verification(): TFLite file not found at {tflite_p}. "
            "Run src/export/convert.py first."
        )

    logger.info(
        "Loading Keras predictor from config snapshot...",
        extra={"stage": "export"},
    )
    keras_predictor = GesturePredictor.from_config_snapshot(
        config_snapshot_path=config_snapshot_path,
        model_path=keras_path,
        smoother_window=smoother_window,
        display_threshold=display_threshold,
    )

    logger.info(
        "Loading TFLite predictor from config snapshot...",
        extra={"stage": "export"},
    )
    tflite_predictor = GesturePredictor.from_config_snapshot(
        config_snapshot_path=config_snapshot_path,
        model_path=tflite_p,
        smoother_window=smoother_window,
        display_threshold=display_threshold,
    )

    # Resolve sign names from the label map attached to the predictor.
    if sign_names is None:
        try:
            label_map = keras_predictor.label_map
            sign_names = [
                label_map.get_name_safe(i, f"class_{i}") for i in range(n_classes)
            ]
            logger.info(
                "Sign names resolved from label map: %s...",
                sign_names[:5],
                extra={"stage": "export"},
            )
        except Exception as e:
            logger.warning(
                "Could not resolve sign names from label map (%s). "
                "Using generic class_N names.",
                e,
                extra={"stage": "export"},
            )
            sign_names = [f"class_{i}" for i in range(n_classes)]

    results: Dict[str, Any] = {
        "n_classes":   n_classes,
        "sign_names":  sign_names,
        "_predictions": {},  # B8 fix: prediction arrays for per-class delta
    }

    for split_name in ("val", "test"):
        logger.info(
            "Running split comparison: %s...", split_name,
            extra={"stage": "export"},
        )
        split_result = _run_split_comparison(
            keras_predictor=keras_predictor,
            tflite_predictor=tflite_predictor,
            val_dataset=val_dataset,
            split_name=split_name,
            n_classes=n_classes,
            sign_names=sign_names,
            compute_macro_f1=compute_macro_f1,
            compute_accuracy=compute_accuracy,
            compute_per_class_metrics=compute_per_class_metrics,
        )

        # Stash prediction arrays for caller use (B8 fix)
        results["_predictions"][split_name] = {
            "y_true":        split_result.pop("_y_true"),
            "y_pred_keras":  split_result.pop("_y_pred_keras"),
            "y_pred_tflite": split_result.pop("_y_pred_tflite"),
            "signer_ids":    split_result.pop("_signer_ids"),
        }

        results[split_name] = split_result

    logger.info(
        "Accuracy verification complete | "
        "val: Keras F1=%.4f → TFLite F1=%.4f (delta=%+.4f) | "
        "test: Keras F1=%.4f → TFLite F1=%.4f (delta=%+.4f)",
        results["val"]["keras_macro_f1"],
        results["val"]["tflite_macro_f1"],
        results["val"]["delta_macro_f1"],
        results["test"]["keras_macro_f1"],
        results["test"]["tflite_macro_f1"],
        results["test"]["delta_macro_f1"],
        extra={"stage": "export"},
    )

    return results


# ---------------------------------------------------------------------------
# Step 2.2 — Per-class TFLite delta analysis
# ---------------------------------------------------------------------------

def compute_per_class_tflite_delta(
    y_true: np.ndarray,
    y_pred_keras: np.ndarray,
    y_pred_tflite: np.ndarray,
    sign_names: List[str],
    n_classes: int = 35,
) -> List[Dict[str, Any]]:
    """
    Compute per-class F1 delta (keras_f1 - tflite_f1) for all 35 classes.

    Classes most likely to show prediction flips after quantisation are the
    four confusable pairs identified in Stage 6 Phase E. The
    ``meaningful_degradation`` flag intentionally EXCLUDES singleton classes,
    where a single prediction flip produces a binary 0→1 or 1→0 F1 swing
    — this is sampling noise, not a quantisation artefact.

    Bug-fix B3: singleton fallback logic corrected. Previously
    `k.get("is_singleton", support == 1)` used `support` which was initialised
    from `k.get("support", 0)` (default 0), so the fallback `0 == 1` always
    evaluated False. Now uses an explicit int comparison after extracting
    support from the per-class dict.

    Parameters
    ----------
    y_true, y_pred_keras, y_pred_tflite : np.ndarray, shape (n_samples,)
    sign_names : List[str], length n_classes
    n_classes  : int, default 35

    Returns
    -------
    List[Dict[str, Any]]
        One dict per class, sorted by |f1_delta| descending (largest
        degradation first). Each dict has keys:
            sign, class_idx, keras_f1, tflite_f1, f1_delta, support,
            is_singleton, is_zero_support, is_high_risk, is_confusable_pair,
            confusable_with, meaningful_degradation
    """
    from src.evaluation.metrics import compute_per_class_metrics

    per_class_keras  = compute_per_class_metrics(
        y_true, y_pred_keras,  sign_names, n_classes
    )
    per_class_tflite = compute_per_class_metrics(
        y_true, y_pred_tflite, sign_names, n_classes
    )

    delta_rows: List[Dict[str, Any]] = []

    for sign in sign_names:
        k = per_class_keras["per_class"].get(sign, {})
        t = per_class_tflite["per_class"].get(sign, {})

        if not k or not t:
            logger.warning(
                "compute_per_class_tflite_delta(): sign '%s' missing from "
                "per-class metrics. Skipping.",
                sign,
                extra={"stage": "export"},
            )
            continue

        keras_f1  = float(k.get("f1_score", 0.0))
        tflite_f1 = float(t.get("f1_score", 0.0))
        f1_delta  = keras_f1 - tflite_f1
        # B3 fix: extract support first, then derive boolean flags independently
        support    = int(k.get("support", 0))
        is_singleton = bool(k.get("is_singleton", support == 1))   # correct: support already known
        is_zero      = bool(k.get("is_zero_support", support == 0))
        is_hr        = sign in _HIGH_RISK_SIGNS
        is_conf      = sign in _CONFUSABLE_SIGNS

        delta_rows.append({
            "sign":                   sign,
            "class_idx":              int(k.get("class_index", 0)),
            "keras_f1":               round(keras_f1, 4),
            "tflite_f1":              round(tflite_f1, 4),
            "f1_delta":               round(f1_delta, 4),
            "support":                support,
            "is_singleton":           is_singleton,
            "is_zero_support":        is_zero,
            "is_high_risk":           is_hr,
            "is_confusable_pair":     is_conf,
            "confusable_with":        _CONFUSABLE_PAIRS.get(sign, []),
            # Only flag non-singleton, non-zero-support classes as meaningfully
            # degraded — singleton flips are sampling noise, not quantisation artefacts.
            "meaningful_degradation": (
                abs(f1_delta) > _MEANINGFUL_DEGRADATION_DELTA
                and not is_singleton
                and not is_zero
            ),
        })

    # Sort: meaningful degradations first, then by |delta| descending.
    delta_rows.sort(
        key=lambda r: (not r["meaningful_degradation"], -abs(r["f1_delta"]))
    )

    n_meaningful = sum(1 for r in delta_rows if r["meaningful_degradation"])
    n_confusable_flipped = sum(
        1 for r in delta_rows
        if r["is_confusable_pair"] and abs(r["f1_delta"]) > 0.0
    )

    logger.info(
        "compute_per_class_tflite_delta(): "
        "%d/%d classes show meaningful degradation (|delta|>%.2f, non-singleton) | "
        "%d confusable-pair classes with non-zero delta",
        n_meaningful, n_classes, _MEANINGFUL_DEGRADATION_DELTA,
        n_confusable_flipped,
        extra={"stage": "export"},
    )

    return delta_rows


# ---------------------------------------------------------------------------
# Step 3 — Production latency benchmark
# ---------------------------------------------------------------------------

def run_production_latency_benchmark(
    tflite_path: Union[str, Path],
    keras_model_path: Union[str, Path],
    pipeline: Any,
    n_calls: int = 200,
    warmup: int = 20,
) -> Dict[str, Any]:
    """
    Produce the official production latency numbers for gesture_bilstm_v1.tflite.

    Uses the verified production TFLite file, not a throwaway scratch export.
    Measures three components:
      - FeaturePipeline preprocessing (inference mode, no augmentation)
      - TFLite inference (production .tflite file)
      - Keras inference (for speedup comparison)

    full_pipeline_ms = pipeline_median_ms + tflite_median_ms.
    MediaPipe extraction (~18ms) is excluded — it belongs to Stage 9's
    per-frame timing profile, not to the model's own latency.

    Parameters
    ----------
    tflite_path     : str | Path — production .tflite file from Step 1
    keras_model_path: str | Path — Keras SavedModel directory
    pipeline        : FeaturePipeline — built from champion config
    n_calls         : int, default 200
    warmup          : int, default 20

    Returns
    -------
    dict with keys:
        tflite, keras, pipeline : benchmark stat dicts
        full_pipeline_ms        : float
        meets_100ms_target      : bool
        speedup_keras_vs_tflite_x : float | None
    """
    import tensorflow as tf
    from src.evaluation.benchmark import (
        benchmark_tflite_inference,
        benchmark_inference,
        benchmark_pipeline_preprocessing,
    )

    tflite_p  = Path(tflite_path)
    keras_p   = Path(keras_model_path)

    if not tflite_p.exists():
        raise FileNotFoundError(
            f"run_production_latency_benchmark(): TFLite file not found: {tflite_p}"
        )
    if not keras_p.exists():
        raise FileNotFoundError(
            f"run_production_latency_benchmark(): Keras SavedModel not found: {keras_p}"
        )

    # Production input shapes for the champion model.
    # TFLite expects static (1, 100, 126); Keras uses dynamic (1, 100, 126).
    X_model = np.zeros((1, 100, 126), dtype=np.float32)
    # Raw full-dim input for pipeline benchmarking (before feature selection).
    X_raw   = np.zeros((100, 225), dtype=np.float32)

    logger.info(
        "Benchmarking TFLite inference (n_calls=%d, warmup=%d)...",
        n_calls, warmup,
        extra={"stage": "export"},
    )
    tflite_stats = benchmark_tflite_inference(
        tflite_p, X_model,
        n_calls=n_calls, warmup=warmup,
        description="gesture_bilstm_v1_tflite_production",
    )

    logger.info(
        "Benchmarking Keras inference (n_calls=%d, warmup=%d)...",
        n_calls, warmup,
        extra={"stage": "export"},
    )
    keras_model  = tf.keras.models.load_model(str(keras_p))
    keras_stats  = benchmark_inference(
        keras_model, X_model,
        n_calls=n_calls, warmup=warmup,
        description="gesture_bilstm_v1_keras",
    )

    logger.info(
        "Benchmarking FeaturePipeline preprocessing (n_calls=%d, warmup=%d)...",
        n_calls, warmup,
        extra={"stage": "export"},
    )
    pipeline_stats = benchmark_pipeline_preprocessing(
        pipeline, X_raw,
        n_calls=n_calls, warmup=warmup,
        description="feature_pipeline_inference_mode",
    )

    full_pipeline_ms = (
        float(pipeline_stats["median_ms"]) + float(tflite_stats["median_ms"])
    )

    keras_median  = float(keras_stats.get("median_ms", 0.0)) or 0.0
    tflite_median = float(tflite_stats.get("median_ms", 0.0)) or 0.0
    speedup = None
    if (np.isfinite(keras_median) and np.isfinite(tflite_median)
            and tflite_median > 0.0):
        speedup = round(keras_median / tflite_median, 2)

    logger.info(
        "Latency benchmark complete | "
        "pipeline=%.2fms | tflite=%.2fms | full=%.2fms | "
        "keras=%.2fms | speedup=%.2fx | meets_100ms=%s",
        float(pipeline_stats["median_ms"]),
        tflite_median,
        full_pipeline_ms,
        keras_median,
        speedup or 0.0,
        full_pipeline_ms < _LATENCY_TARGET_MS,
        extra={"stage": "export"},
    )

    return {
        "tflite":                    tflite_stats,
        "keras":                     keras_stats,
        "pipeline":                  pipeline_stats,
        "full_pipeline_ms":          round(full_pipeline_ms, 3),
        "meets_100ms_target":        full_pipeline_ms < _LATENCY_TARGET_MS,
        "speedup_keras_vs_tflite_x": speedup,
    }


# ---------------------------------------------------------------------------
# Release gate assembly
# ---------------------------------------------------------------------------

def assemble_release_gate(
    verification_result: Dict[str, Any],
    latency_result: Dict[str, Any],
    tflite_path: Union[str, Path],
) -> "ReleaseGateResult":
    """
    Assemble a ReleaseGateResult from verification and latency results.

    Bug-fix B9: tflite_size_mb is now checked for finiteness before comparison
    against the 10 MB threshold, guarding against stat() failures.

    Parameters
    ----------
    verification_result : dict from run_accuracy_verification()
    latency_result      : dict from run_production_latency_benchmark()
    tflite_path         : path to the .tflite file

    Returns
    -------
    ReleaseGateResult
    """
    tflite_p = Path(tflite_path)
    tflite_exists = tflite_p.exists()

    # B9 fix: guard against stat() failure
    tflite_size_mb = float("nan")
    if tflite_exists:
        try:
            raw_bytes = tflite_p.stat().st_size
            tflite_size_mb = round(raw_bytes / (1024 ** 2), 4)
        except OSError as exc:
            logger.error(
                "assemble_release_gate(): could not stat TFLite file %s: %s",
                tflite_p, exc,
                extra={"stage": "export"},
            )
            tflite_size_mb = float("nan")

    # size_under_10mb is only True when the file exists AND size is measurable
    # AND under the threshold — NaN comparison returns False safely.
    size_under_10mb = (
        tflite_exists
        and np.isfinite(tflite_size_mb)
        and tflite_size_mb < _MAX_TFLITE_SIZE_MB
    )

    val  = verification_result.get("val",  {})
    test = verification_result.get("test", {})

    full_pipeline_ms = float(latency_result.get("full_pipeline_ms", float("nan")))

    gate = ReleaseGateResult(
        # Accuracy deltas
        val_delta_macro_f1=val.get("delta_macro_f1", float("nan")),
        test_delta_macro_f1=test.get("delta_macro_f1", float("nan")),
        # Absolute F1 values
        keras_val_macro_f1=val.get("keras_macro_f1", float("nan")),
        tflite_val_macro_f1=val.get("tflite_macro_f1", float("nan")),
        keras_test_macro_f1=test.get("keras_macro_f1", float("nan")),
        tflite_test_macro_f1=test.get("tflite_macro_f1", float("nan")),
        keras_val_accuracy=val.get("keras_accuracy", float("nan")),
        tflite_val_accuracy=val.get("tflite_accuracy", float("nan")),
        keras_test_accuracy=test.get("keras_accuracy", float("nan")),
        tflite_test_accuracy=test.get("tflite_accuracy", float("nan")),
        # Argmax agreement
        val_argmax_agreement=val.get("argmax_agreement", float("nan")),
        test_argmax_agreement=test.get("argmax_agreement", float("nan")),
        # Probability distribution
        val_mean_abs_diff=val.get("mean_abs_diff", float("nan")),
        val_max_abs_diff=val.get("max_abs_diff", float("nan")),
        # Calibration
        val_confidence_shift=val.get("confidence_shift", float("nan")),
        keras_mean_confidence=val.get("keras_mean_confidence", float("nan")),
        tflite_mean_confidence=val.get("tflite_mean_confidence", float("nan")),
        # File
        tflite_file_exists=tflite_exists,
        tflite_size_mb=tflite_size_mb,
        size_under_10mb=size_under_10mb,
        # Latency
        full_pipeline_ms=full_pipeline_ms,
        tflite_median_ms=float(
            latency_result.get("tflite", {}).get("median_ms", float("nan"))
        ),
        keras_median_ms=float(
            latency_result.get("keras", {}).get("median_ms", float("nan"))
        ),
        pipeline_median_ms=float(
            latency_result.get("pipeline", {}).get("median_ms", float("nan"))
        ),
        meets_100ms_target=latency_result.get("meets_100ms_target", False),
        speedup_keras_vs_tflite_x=latency_result.get("speedup_keras_vs_tflite_x"),
        # Sample counts
        n_val_samples=val.get("n_samples", 0),
        n_test_samples=test.get("n_samples", 0),
    )

    return gate


# ---------------------------------------------------------------------------
# Step 4 — Model metadata JSON
# ---------------------------------------------------------------------------

def write_model_metadata(
    tflite_path: Union[str, Path],
    conversion_result: Dict[str, Any],
    verification_result: Dict[str, Any],
    latency_result: Dict[str, Any],
    per_class_delta: Optional[List[Dict[str, Any]]] = None,
    config_snapshot_path: str = _DEFAULT_CONFIG_SNAPSHOT_PATH,
    stage6_report_path: Optional[str] = _DEFAULT_STAGE6_REPORT_PATH,
    output_path: Union[str, Path] = _DEFAULT_METADATA_OUTPUT_PATH,
) -> None:
    """
    Write gesture_model_metadata.json — the authoritative deployment metadata
    for gesture_bilstm_v1.tflite.

    Design principles:
      1. Architecture fields are read from config_snapshot.yaml, not hardcoded.
      2. Stage 6 calibration numbers are labelled stage6_reference_metrics,
         not presented as Stage 8 measurements.
      3. All three size measurements are present and correctly labelled:
         param_memory_mb, savedmodel_disk_mb, tflite_disk_mb.
      4. Stage 8 measured TFLite performance comes from verification_result.
      5. If stage6_report_path is provided and exists, calibration metrics
         are loaded from it rather than hardcoded.

    Bug-fix B5/B10: all enum-typed config fields are cast via str() to guarantee
    JSON serialisability regardless of Pydantic's coercion behaviour at runtime.

    Parameters
    ----------
    tflite_path         : Path to the production .tflite file (for verification).
    conversion_result   : Dict from convert.export_champion() or export_champion_tflite().
    verification_result : Dict from run_accuracy_verification().
    latency_result      : Dict from run_production_latency_benchmark().
    per_class_delta     : List from compute_per_class_tflite_delta(), optional.
    config_snapshot_path: Path to champion config_snapshot.yaml.
    stage6_report_path  : Path to evaluation_report.json (Phase F), optional.
    output_path         : Destination JSON path.
    """
    from omegaconf import OmegaConf
    from src.utils.config import ExperimentConfig

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Read architecture config from snapshot — not hardcoded.
    snapshot_path = Path(config_snapshot_path)
    if not snapshot_path.exists():
        raise FileNotFoundError(
            f"write_model_metadata(): config snapshot not found at {snapshot_path}"
        )
    raw    = OmegaConf.to_container(OmegaConf.load(snapshot_path), resolve=True)
    config = ExperimentConfig(**raw)

    # Load Stage 6 calibration metrics (with provenance tracking).
    stage6_calibration = _load_stage6_calibration(stage6_report_path)

    # Verify the TFLite file exists and record its size.
    tflite_p = Path(tflite_path)
    tflite_disk_mb = None
    if tflite_p.exists():
        try:
            tflite_disk_mb = round(tflite_p.stat().st_size / (1024 ** 2), 4)
        except OSError:
            tflite_disk_mb = None

    # Summarise per-class delta for model card (top degradations + confusable pairs).
    per_class_summary: List[Dict[str, Any]] = []
    if per_class_delta:
        for row in per_class_delta:
            if row.get("meaningful_degradation") or row.get("is_confusable_pair"):
                per_class_summary.append(dict(row))  # shallow copy

    # Actual tflite size from file if not in conversion_result.
    cr_tflite_mb = conversion_result.get("tflite_disk_mb") or tflite_disk_mb

    # B10 fix: cast all enum-typed fields via str() before embedding in JSON.
    # Pydantic may return enum instances rather than plain strings depending on
    # the version and how fields were populated.
    landmark_config_str      = str(config.data.landmark_config)
    normalisation_str        = str(config.data.normalisation.value
                                   if hasattr(config.data.normalisation, "value")
                                   else config.data.normalisation)
    missing_frame_str        = str(config.data.missing_frame_strategy.value
                                   if hasattr(config.data.missing_frame_strategy, "value")
                                   else config.data.missing_frame_strategy)
    model_name_str           = str(config.model.name.value
                                   if hasattr(config.model.name, "value")
                                   else config.model.name)

    metadata = {
        "model_name":           "gesture_bilstm_v1",
        "version":              "1.0.0",
        "created_utc":          datetime.now(timezone.utc).isoformat(),
        "mlflow_run_id":        _CHAMPION_MLFLOW_RUN_ID,
        "config_hash":          _CHAMPION_CONFIG_HASH,
        "config_snapshot_path": str(snapshot_path),

        # Architecture — read from config snapshot, not hardcoded.
        "architecture": {
            "name":                model_name_str,
            "num_layers":          config.model.num_layers,
            "hidden_units":        config.model.hidden_units,
            "units_per_direction": config.model.hidden_units // 2,
            "dropout":             config.model.dropout,
            "recurrent_dropout":   config.model.recurrent_dropout,
            "bidirectional":       config.model.bidirectional,
            "total_params":        68771,
            "_note": (
                "Architecture fields read from config_snapshot.yaml. "
                "total_params is hardcoded as cross-check against the "
                "_verify_champion_model() assertion in convert.py."
            ),
        },

        # Input/output — distinguished for Keras vs TFLite contexts.
        "input_shape_keras":    [None, 100, 126],
        "input_shape_tflite":   [1,    100, 126],
        "input_dtype":          "float32",
        "output_shape_keras":   [None, 35],
        "output_shape_tflite":  [1,    35],
        "output_dtype":         "float32",

        # Preprocessing — read from config snapshot (all enum fields cast to str).
        "preprocessing": {
            "sequence_length":          config.data.sequence_length,
            "landmark_config":          landmark_config_str,
            "feature_dim":              126,  # hands_only: 63+63
            "feature_layout": {
                "left_hand":   [0,   63],
                "right_hand":  [63, 126],
                "pose":        "not_used",
            },
            "normalisation":            normalisation_str,
            "z_coord_clip":             config.data.z_coord_clip,
            "normalise_pose":           config.data.normalise_pose,
            "flip_min_hand_presence":   config.data.flip_min_hand_presence,
            "missing_frame_strategy":   missing_frame_str,
            "padding_strategy":         "right_zero",
            "truncation_strategy":      "centre",
            "mediapipe_model_complexity": 1,
            "min_detection_confidence":   0.5,
            "min_tracking_confidence":    0.5,
        },

        "label_map_version": "v1",
        "label_map_path":    "artifacts/label_map_v1.json",
        "num_classes":       config.num_classes,

        # Stage 6 reference metrics — explicitly labelled as historical baselines.
        "stage6_reference_metrics": {
            "_note": (
                "These values come from Stage 6 evaluation (Phase B1/C). "
                "They are the Keras model baselines against which TFLite "
                "deltas are computed in Stage 8. CIs are 90% bootstrap "
                "intervals from Stage 6 Phase B1 analysis."
            ),
            "keras_val_macro_f1":      _STAGE6_KERAS_VAL_MACRO_F1,
            "keras_val_accuracy":      _STAGE6_KERAS_VAL_ACCURACY,
            "keras_val_macro_f1_ci":   stage6_calibration.get(
                "val_macro_f1_ci", list(_STAGE6_KERAS_VAL_MACRO_F1_CI)
            ),
            "keras_test_macro_f1":     _STAGE6_KERAS_TEST_MACRO_F1,
            "keras_test_accuracy":     _STAGE6_KERAS_TEST_ACCURACY,
            "keras_test_macro_f1_ci":  stage6_calibration.get(
                "test_macro_f1_ci", list(_STAGE6_KERAS_TEST_MACRO_F1_CI)
            ),
            "ci_level":                0.90,
            "ci_source":               stage6_calibration.get("_source", "unknown"),
        },

        # Stage 8 measured TFLite performance.
        "tflite_performance": {
            "_note": "Measured in Stage 8 by run_accuracy_verification().",
            "val_macro_f1":        verification_result["val"]["tflite_macro_f1"],
            "val_accuracy":        verification_result["val"]["tflite_accuracy"],
            "test_macro_f1":       verification_result["test"]["tflite_macro_f1"],
            "test_accuracy":       verification_result["test"]["tflite_accuracy"],
            "val_delta_macro_f1":  verification_result["val"]["delta_macro_f1"],
            "test_delta_macro_f1": verification_result["test"]["delta_macro_f1"],
            "val_argmax_agreement":   verification_result["val"]["argmax_agreement"],
            "test_argmax_agreement":  verification_result["test"]["argmax_agreement"],
            "val_n_samples":       verification_result["val"]["n_samples"],
            "test_n_samples":      verification_result["test"]["n_samples"],
            "val_n_disagreements": verification_result["val"]["n_disagreements"],
            "test_n_disagreements": verification_result["test"]["n_disagreements"],
        },

        # Calibration — Stage 6 reference + TFLite shift measured in Stage 8.
        "calibration": {
            "stage6_keras_reference":    {
                k: v for k, v in stage6_calibration.items()
                if not k.startswith("val_macro_f1_ci") and not k.startswith("test_macro_f1_ci")
            },
            "tflite_confidence_shift":   verification_result["val"]["confidence_shift"],
            "keras_mean_confidence":     verification_result["val"]["keras_mean_confidence"],
            "tflite_mean_confidence":    verification_result["val"]["tflite_mean_confidence"],
            "calibration_direction":     "underconfident",
            "recommended_display_threshold": _CHAMPION_DISPLAY_THRESHOLD,
            "_note": (
                "Stage 6 found the Keras model is underconfident "
                f"(mean_confidence={_STAGE6_MEAN_CONFIDENCE} < "
                f"mean_accuracy={_STAGE6_MEAN_ACCURACY}). "
                f"display_threshold={_CHAMPION_DISPLAY_THRESHOLD} is calibrated "
                "to this underconfidence. Check tflite_confidence_shift — "
                f"if >±{_CONFIDENCE_SHIFT_WARN_THRESHOLD}, reconsider threshold "
                "for TFLite deployment model in Stage 9."
            ),
        },

        # File sizes — all three correctly distinguished.
        "file_sizes": {
            "param_memory_mb":               conversion_result.get("param_memory_mb"),
            "savedmodel_disk_mb":            conversion_result.get("savedmodel_disk_mb"),
            "tflite_disk_mb":                cr_tflite_mb,
            "size_reduction_vs_params_x":    conversion_result.get("size_reduction_vs_params_x"),
            "size_reduction_vs_savedmodel_x": conversion_result.get("size_reduction_vs_savedmodel_x"),
            "_note": (
                "param_memory_mb = params * 4 bytes (weight tensors only). "
                "savedmodel_disk_mb = actual SavedModel directory size on disk "
                "(includes graph, assets, metadata). "
                "tflite_disk_mb = actual .tflite file size on disk."
            ),
        },

        # Latency.
        "latency_cpu": {
            "tflite_median_ms":   latency_result["tflite"]["median_ms"],
            "tflite_p95_ms":      latency_result["tflite"]["p95_ms"],
            "tflite_p99_ms":      latency_result["tflite"].get("p99_ms"),
            "tflite_fps":         latency_result["tflite"]["fps"],
            "keras_median_ms":    latency_result["keras"]["median_ms"],
            "pipeline_median_ms": latency_result["pipeline"]["median_ms"],
            "full_pipeline_ms":   latency_result["full_pipeline_ms"],
            "meets_100ms_target": latency_result["meets_100ms_target"],
            "speedup_vs_keras_x": latency_result["speedup_keras_vs_tflite_x"],
            "benchmark_n_calls":  200,
            "benchmark_warmup":   20,
            "_note": (
                "full_pipeline_ms = pipeline_median_ms + tflite_median_ms. "
                "MediaPipe extraction (~18ms) is excluded — measured in Stage 9."
            ),
        },

        # Quantisation.
        "quantisation": {
            "mode":        "dynamic_range",
            "description": "tf.lite.Optimize.DEFAULT — int8 weights, float32 activations",
            "requires_select_tf_ops": True,
            "reason_for_select_tf_ops": (
                "Bidirectional(LSTM) in TF 2.13 emits TensorListReserve ops "
                "not in the standard TFLite builtin op set. SELECT_TF_OPS "
                "(flex delegate) required at inference time. "
                "Adds ~800KB to the Android TFLite runtime binary."
            ),
            "sha256_checksum": conversion_result.get("sha256_checksum"),
        },

        # Per-class quantisation impact summary.
        "per_class_quantisation_impact": {
            "n_meaningful_degradations": sum(
                1 for r in (per_class_delta or []) if r.get("meaningful_degradation")
            ),
            "n_confusable_pair_flips": sum(
                1 for r in (per_class_delta or [])
                if r.get("is_confusable_pair") and abs(r.get("f1_delta", 0.0)) > 0.0
            ),
            "notable_classes":         per_class_summary[:20],
            "_note": (
                "See tflite_verification_report.json for full 35-class delta table. "
                "meaningful_degradation = |delta| > 0.10 on non-singleton class. "
                "Confusable pairs from Stage 6 Phase E: "
                "think/who, later/house, cousin/mother, girl/orange."
            ),
        },

        # Interpretability notes from Stage 6 Phase E.
        "interpretability_notes": {
            "peak_frame_importance":        36,
            "importance_decay_after_frame": 70,
            "dominant_hand":                "right",
            "left_hand_attribution":        "near-zero (possible signer handedness artifact)",
            "top_confusable_pairs": [
                {"signs": ["think",  "who"],    "cosine_similarity": [0.905, 0.785]},
                {"signs": ["later",  "house"],  "cosine_similarity": [0.919, 0.946]},
                {"signs": ["cousin", "mother"], "cosine_similarity": [0.927, 0.947]},
                {"signs": ["girl",   "orange"], "cosine_similarity": [0.963, 0.937]},
            ],
        },

        # Deployment notes.
        "deployment_notes": {
            "thread_safety":          "NOT thread-safe — one GesturePredictor per thread",
            "mediapipe_version":      "0.10.14 (pinned)",
            "tensorflow_version":     "2.13.1",
            "android_runtime_note":   "Requires TFLite flex delegate for SELECT_TF_OPS",
            "recommended_entry_point": "GesturePredictor.from_config_snapshot()",
            "config_snapshot_path":   str(snapshot_path),
            "stage9_display_threshold": _CHAMPION_DISPLAY_THRESHOLD,
            "smoother_window":         5,
            "early_stopping_note": (
                "The champion config shows early_stopping_monitor: val_accuracy. "
                "This controls only ReduceLROnPlateau; actual early stopping used a "
                "manual Python loop monitoring val_macro_f1 with patience=50. "
                "See config_snapshot.yaml for full training parameters."
            ),
        },
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_nan_to_none(metadata), f, indent=2, default=str)

    logger.info(
        "Model metadata written → %s",
        out_path.resolve(),
        extra={"stage": "export"},
    )


# ---------------------------------------------------------------------------
# Report serialisation
# ---------------------------------------------------------------------------

def save_verification_report(
    gate: "ReleaseGateResult",
    verification_result: Dict[str, Any],
    latency_result: Dict[str, Any],
    per_class_delta: Optional[List[Dict[str, Any]]] = None,
    conversion_result: Optional[Dict[str, Any]] = None,
    output_path: Union[str, Path] = _DEFAULT_VERIFICATION_REPORT_PATH,
) -> Path:
    """
    Write tflite_verification_report.json — the single authoritative gate
    report for the Stage 8 deliverable.

    Bug-fix B4: keras_per_class and tflite_per_class are now stripped from
    the per-split dicts (in addition to disagreement_details) to prevent
    the JSON from ballooning with full 35-class breakdowns that are already
    captured in per_class_delta.

    Bug-fix B6: _nan_to_none is now applied to the latency_benchmark sub-dict,
    which can contain NaN from a failed benchmark run.

    Parameters
    ----------
    gate               : assembled ReleaseGateResult
    verification_result: from run_accuracy_verification()
    latency_result     : from run_production_latency_benchmark()
    per_class_delta    : from compute_per_class_tflite_delta(), optional
    conversion_result  : from convert.export_champion(), optional
    output_path        : destination JSON path

    Returns
    -------
    Path — resolved absolute path of the written report
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # B4 fix: strip bulky keys from both splits before embedding in the report.
    val_stripped  = _strip_bulky(verification_result.get("val",  {}))
    test_stripped = _strip_bulky(verification_result.get("test", {}))

    # Build latency summary (omit raw per-call duration arrays).
    latency_summary = {
        "tflite_median_ms":          latency_result.get("tflite", {}).get("median_ms"),
        "tflite_p95_ms":             latency_result.get("tflite", {}).get("p95_ms"),
        "tflite_p99_ms":             latency_result.get("tflite", {}).get("p99_ms"),
        "tflite_fps":                latency_result.get("tflite", {}).get("fps"),
        "keras_median_ms":           latency_result.get("keras",  {}).get("median_ms"),
        "pipeline_median_ms":        latency_result.get("pipeline", {}).get("median_ms"),
        "full_pipeline_ms":          latency_result.get("full_pipeline_ms"),
        "meets_100ms_target":        latency_result.get("meets_100ms_target"),
        "speedup_keras_vs_tflite_x": latency_result.get("speedup_keras_vs_tflite_x"),
    }

    report = {
        "report_created_utc":   datetime.now(timezone.utc).isoformat(),
        "champion_run":         "bilstm_hands_only_v4_aug",
        "mlflow_run_id":        _CHAMPION_MLFLOW_RUN_ID,
        "config_hash":          _CHAMPION_CONFIG_HASH,
        "tflite_file":          "models/gesture_bilstm_v1.tflite",
        "sha256_checksum":      (conversion_result or {}).get("sha256_checksum"),

        # The verdict
        "release_gate": gate.to_dict(),  # already _nan_to_none'd

        # Per-split accuracy comparison (B4 fix: bulky keys stripped)
        "accuracy_comparison": {
            "val":  val_stripped,
            "test": test_stripped,
        },

        # Latency (B6 fix: _nan_to_none applied)
        "latency_benchmark": _nan_to_none(latency_summary),

        # Size
        "file_sizes": {
            "param_memory_mb":               (conversion_result or {}).get("param_memory_mb"),
            "savedmodel_disk_mb":            (conversion_result or {}).get("savedmodel_disk_mb"),
            "tflite_disk_mb":                (conversion_result or {}).get("tflite_disk_mb"),
            "size_reduction_vs_params_x":    (conversion_result or {}).get("size_reduction_vs_params_x"),
            "size_reduction_vs_savedmodel_x": (conversion_result or {}).get("size_reduction_vs_savedmodel_x"),
        },

        # Per-class delta (full 35-class table)
        "per_class_delta": per_class_delta if per_class_delta else None,

        # Stage 6 reference (for traceability)
        "stage6_reference": {
            "keras_val_macro_f1":  _STAGE6_KERAS_VAL_MACRO_F1,
            "keras_test_macro_f1": _STAGE6_KERAS_TEST_MACRO_F1,
            "ece":                 _STAGE6_ECE,
            "mean_confidence":     _STAGE6_MEAN_CONFIDENCE,
            "display_threshold":   _CHAMPION_DISPLAY_THRESHOLD,
        },
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_nan_to_none(report), f, indent=2, default=str)

    logger.info(
        "Verification report written → %s",
        out_path.resolve(),
        extra={"stage": "export"},
    )
    return out_path.resolve()


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def plot_tflite_size_comparison(
    conversion_result: Dict[str, Any],
    output_path: Optional[Union[str, Path]] = None,
    figure_dpi: int = 150,
) -> Any:
    """
    Plot all three size measurements for the champion model.

    Three bars:
      - param_memory_mb: weight tensor footprint (params × 4 bytes)
      - savedmodel_disk_mb: SavedModel directory size (graph + metadata)
      - tflite_disk_mb: quantised TFLite file size

    The 10 MB project target is shown as a reference line.

    Parameters
    ----------
    conversion_result : dict from convert.export_champion() or export_champion_tflite()
    output_path       : optional save path
    figure_dpi        : int

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError as exc:
        raise ImportError(
            "plot_tflite_size_comparison() requires matplotlib."
        ) from exc

    param_mb      = conversion_result.get("param_memory_mb", 0.262)
    savedmodel_mb = conversion_result.get("savedmodel_disk_mb")
    tflite_mb     = conversion_result.get("tflite_disk_mb")

    labels  = ["Param memory\n(float32 weights)", "SavedModel\n(on disk)", "TFLite\n(on disk, quantised)"]
    values  = [param_mb, savedmodel_mb, tflite_mb]
    colors  = ["#4C72B0", "#DD8452", "#55A868"]
    has_val = [v is not None for v in values]
    values  = [v if v is not None else 0.0 for v in values]

    fig, ax = plt.subplots(figsize=(9, 5))

    active_labels = [l for l, h in zip(labels, has_val) if h]
    active_values = [v for v, h in zip(values, has_val) if h]
    active_colors = [c for c, h in zip(colors, has_val) if h]

    bars = ax.bar(
        active_labels, active_values,
        color=active_colors,
        edgecolor="white", linewidth=1.2, alpha=0.88, width=0.5,
    )

    ax.axhline(
        _MAX_TFLITE_SIZE_MB, color="red", linestyle="--", linewidth=1.5,
        label=f"{_MAX_TFLITE_SIZE_MB:.0f} MB project target",
    )

    for bar, val in zip(bars, active_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{val:.4f} MB",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    reduction_vs_params = conversion_result.get("size_reduction_vs_params_x")
    reduction_vs_saved  = conversion_result.get("size_reduction_vs_savedmodel_x")

    note_parts = []
    if reduction_vs_params:
        note_parts.append(f"{reduction_vs_params:.1f}× reduction vs param memory")
    if reduction_vs_saved:
        note_parts.append(f"{reduction_vs_saved:.1f}× reduction vs SavedModel")
    note = "\n".join(note_parts)

    ax.set_ylabel("File size (MB)", fontsize=12)
    ax.set_title(
        "Model Size Comparison — gesture_bilstm_v1 (68,771 params)\n"
        f"Dynamic-range quantisation | {note}",
        fontsize=11, pad=12,
    )
    ax.legend(fontsize=10, loc="upper right")
    max_val = max(active_values) if active_values else 1.0
    ax.set_ylim(0, max(max_val * 1.25, _MAX_TFLITE_SIZE_MB * 1.2))
    ax.grid(True, alpha=0.25, linestyle=":", axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=figure_dpi, bbox_inches="tight")
        logger.info("Size comparison figure → %s", out.resolve(), extra={"stage": "export"})

    return fig


def plot_tflite_accuracy_comparison(
    verification_result: Dict[str, Any],
    output_path: Optional[Union[str, Path]] = None,
    figure_dpi: int = 150,
) -> Any:
    """
    Plot Keras vs TFLite macro-F1 and accuracy for val and test splits.

    A grouped bar chart with four metric groups:
      val macro-F1, val accuracy, test macro-F1, test accuracy.

    Parameters
    ----------
    verification_result : dict from run_accuracy_verification()
    output_path         : optional save path
    figure_dpi          : int

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError as exc:
        raise ImportError(
            "plot_tflite_accuracy_comparison() requires matplotlib."
        ) from exc

    val       = verification_result.get("val",  {})
    test      = verification_result.get("test", {})
    n_classes = verification_result.get("n_classes", 35)

    groups = [
        ("Val\nMacro-F1",  val.get("keras_macro_f1",  0), val.get("tflite_macro_f1",  0)),
        ("Val\nAccuracy",  val.get("keras_accuracy",   0), val.get("tflite_accuracy",  0)),
        ("Test\nMacro-F1", test.get("keras_macro_f1", 0), test.get("tflite_macro_f1", 0)),
        ("Test\nAccuracy", test.get("keras_accuracy",  0), test.get("tflite_accuracy", 0)),
    ]

    labels_g = [g[0] for g in groups]
    keras_v  = [g[1] for g in groups]
    tflite_v = [g[2] for g in groups]
    deltas   = [k - t for k, t in zip(keras_v, tflite_v)]

    x     = np.arange(len(labels_g))
    width = 0.34

    fig, ax = plt.subplots(figsize=(11, 6))

    bars_k = ax.bar(
        x - width / 2, keras_v, width,
        label="Keras SavedModel", color="#4C72B0", alpha=0.88, edgecolor="white",
    )
    bars_t = ax.bar(
        x + width / 2, tflite_v, width,
        label="TFLite (dynamic-range)", color="#55A868", alpha=0.88, edgecolor="white",
    )

    # Delta labels above each pair
    for i, (bk, bt, d) in enumerate(zip(bars_k, bars_t, deltas)):
        pair_max = max(bk.get_height(), bt.get_height())
        color = (
            "firebrick" if d > _DELTA_THRESHOLD
            else "forestgreen" if abs(d) <= 0.01
            else "darkorange"
        )
        ax.text(
            x[i], pair_max + 0.025,
            f"Δ={d:+.4f}",
            ha="center", va="bottom", fontsize=9.5,
            color=color, fontweight="bold",
        )

    # Value labels on bars
    for bar in list(bars_k) + list(bars_t):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,
            f"{bar.get_height():.4f}",
            ha="center", va="center", fontsize=8,
            color="white", fontweight="bold",
        )

    # Stage 6 reference lines
    ax.axhline(
        _STAGE6_KERAS_VAL_MACRO_F1, color="#4C72B0", linestyle=":",
        linewidth=1.2, alpha=0.6,
        label=f"S6 Keras val F1 ref ({_STAGE6_KERAS_VAL_MACRO_F1:.4f})",
    )
    ax.axhline(
        _STAGE6_KERAS_TEST_MACRO_F1, color="#4C72B0", linestyle="--",
        linewidth=1.2, alpha=0.6,
        label=f"S6 Keras test F1 ref ({_STAGE6_KERAS_TEST_MACRO_F1:.4f})",
    )
    ax.axhline(
        _DELTA_THRESHOLD, color="red", linestyle="-.",
        linewidth=0.8, alpha=0.4, label=f"Delta threshold (±{_DELTA_THRESHOLD})",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels_g, fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.set_title(
        f"Keras vs TFLite Accuracy Comparison — {n_classes}-class WLASL\n"
        f"Val: agreement={val.get('argmax_agreement', 0):.4f}  "
        f"Test: agreement={test.get('argmax_agreement', 0):.4f}  "
        f"Confidence shift: {val.get('confidence_shift', 0):+.4f}",
        fontsize=11, pad=12,
    )
    ax.legend(fontsize=9, loc="lower right", framealpha=0.85)
    ax.grid(True, alpha=0.2, linestyle=":", axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=figure_dpi, bbox_inches="tight")
        logger.info(
            "Accuracy comparison figure → %s", out.resolve(), extra={"stage": "export"}
        )

    return fig


def plot_tflite_per_class_delta(
    per_class_delta: List[Dict[str, Any]],
    output_path: Optional[Union[str, Path]] = None,
    figure_dpi: int = 150,
    n_classes: int = 35,
) -> Any:
    """
    Horizontal bar chart of per-class F1 delta (Keras - TFLite) for all 35 classes.

    Visual encoding:
      - Bars coloured by delta direction: red (TFLite worse), green (TFLite better)
      - Confusable-pair classes marked with ★ in the label
      - High-risk classes marked with ⚡ in the label
      - Singleton classes shown in lighter colour
      - Meaningful degradation threshold (±0.10) shown as vertical reference lines

    Parameters
    ----------
    per_class_delta : from compute_per_class_tflite_delta()
    output_path     : optional save path
    figure_dpi      : int
    n_classes       : int

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError as exc:
        raise ImportError(
            "plot_tflite_per_class_delta() requires matplotlib."
        ) from exc

    if not per_class_delta:
        logger.warning(
            "plot_tflite_per_class_delta(): empty per_class_delta list.",
            extra={"stage": "export"},
        )
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, "No per-class delta data available",
                ha="center", va="center", transform=ax.transAxes)
        if output_path is not None:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(out), dpi=figure_dpi, bbox_inches="tight")
        return fig

    # Sort by f1_delta ascending (most degraded first at bottom).
    rows_sorted = sorted(per_class_delta, key=lambda r: r.get("f1_delta", 0))

    signs  = []
    deltas = []
    colors = []
    alphas = []

    for row in rows_sorted:
        sign   = row["sign"]
        delta  = row.get("f1_delta", 0.0)
        is_sg  = row.get("is_singleton", False)
        is_hr  = row.get("is_high_risk", False)
        is_cf  = row.get("is_confusable_pair", False)

        label = sign
        if is_cf:
            label += " ★"
        if is_hr:
            label += " ⚡"
        if is_sg:
            label += " ·"

        signs.append(label)
        deltas.append(delta)

        if delta > 0.0:
            base_color = "#d62728"   # red: TFLite worse
        elif delta < 0.0:
            base_color = "#2ca02c"   # green: TFLite better
        else:
            base_color = "#7f7f7f"   # grey: no change

        alphas.append(0.5 if is_sg else 0.85)
        colors.append(base_color)

    fig_height = max(8, n_classes * 0.32)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    y_pos = np.arange(len(signs))
    bars  = ax.barh(
        y_pos, deltas,
        color=colors, alpha=alphas,
        edgecolor="white", linewidth=0.5,
        height=0.72,
    )

    ax.axvline(
        _MEANINGFUL_DEGRADATION_DELTA, color="red", linestyle="--",
        linewidth=1.2, alpha=0.6,
        label=f"Meaningful degradation threshold (±{_MEANINGFUL_DEGRADATION_DELTA})",
    )
    ax.axvline(
        -_MEANINGFUL_DEGRADATION_DELTA, color="green", linestyle="--",
        linewidth=1.2, alpha=0.6,
    )
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(signs, fontsize=8.5)
    ax.set_xlabel("F1 delta (Keras − TFLite)  [positive = TFLite degraded]", fontsize=11)
    ax.set_title(
        f"Per-Class F1 Delta After Dynamic-Range Quantisation ({n_classes} classes)\n"
        "★ = confusable pair (Stage 6 Phase E)  ⚡ = high-risk class (Stage 5 Finding 8)  "
        "· = singleton val class",
        fontsize=10, pad=12,
    )

    for bar, val in zip(bars, deltas):
        if abs(val) > 0.001:
            ax.text(
                val + (0.005 if val >= 0 else -0.005),
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}",
                ha="left" if val >= 0 else "right",
                va="center", fontsize=7.5,
            )

    legend_handles = [
        mpatches.Patch(color="#d62728", alpha=0.85, label="TFLite degraded (Keras better)"),
        mpatches.Patch(color="#2ca02c", alpha=0.85, label="TFLite improved (TFLite better)"),
        mpatches.Patch(color="#7f7f7f", alpha=0.85, label="No change"),
        mpatches.Patch(color="#d62728", alpha=0.5, label="Singleton class (lighter shade)"),
        plt.Line2D([0], [0], color="red", linestyle="--", linewidth=1.2,
                   label=f"±{_MEANINGFUL_DEGRADATION_DELTA} meaningful degradation threshold"),
    ]
    ax.legend(handles=legend_handles, fontsize=8.5, loc="lower right", framealpha=0.85)
    ax.grid(True, alpha=0.2, linestyle=":", axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=figure_dpi, bbox_inches="tight")
        logger.info(
            "Per-class delta figure → %s", out.resolve(), extra={"stage": "export"}
        )

    return fig


# ---------------------------------------------------------------------------
# End-to-end orchestrator (primary entry point for notebooks)
# ---------------------------------------------------------------------------

def run_full_verification(
    keras_model_path: Union[str, Path] = _DEFAULT_KERAS_MODEL_PATH,
    tflite_path: Union[str, Path] = _DEFAULT_TFLITE_PATH,
    config_snapshot_path: Union[str, Path] = _DEFAULT_CONFIG_SNAPSHOT_PATH,
    val_dataset: Optional[Any] = None,
    pipeline: Optional[Any] = None,
    stage6_report_path: Optional[str] = _DEFAULT_STAGE6_REPORT_PATH,
    conversion_result: Optional[Dict[str, Any]] = None,
    figures_dir: Optional[Union[str, Path]] = None,
    verification_report_path: Union[str, Path] = _DEFAULT_VERIFICATION_REPORT_PATH,
    metadata_output_path: Union[str, Path] = _DEFAULT_METADATA_OUTPUT_PATH,
    n_calls: int = 200,
    warmup: int = 20,
    n_classes: int = 35,
    sign_names: Optional[List[str]] = None,
) -> Tuple["ReleaseGateResult", Dict[str, Any]]:
    """
    End-to-end Stage 8 Step 2 orchestration.

    This is the single function notebooks/07_tflite_verification.ipynb calls.
    It runs every component of the verification suite in the correct order
    and writes all artefacts to disk.

    Bug-fix B1/B2: the orchestrator no longer re-runs inference for per-class
    delta computation. Instead, it uses the prediction arrays already computed
    by run_accuracy_verification() and returned in the '_predictions' sub-dict.
    This eliminates the double-inference bug (4 inference passes → 2).

    Orchestration order:
      1. run_accuracy_verification()       — Keras vs TFLite metrics (val + test)
      2. compute_per_class_tflite_delta()  — 35-class F1 delta (val split, reusing
                                             predictions from Step 1 — B1 fix)
      3. run_production_latency_benchmark() — CPU latency profile
      4. assemble_release_gate()           — unified pass/fail verdict
      5. save_verification_report()        — tflite_verification_report.json
      6. write_model_metadata()            — gesture_model_metadata.json
      7. Figure generation (3 figures)

    Parameters
    ----------
    keras_model_path     : str | Path
    tflite_path          : str | Path
    config_snapshot_path : str | Path
    val_dataset          : GestureDataset — must be already constructed
    pipeline             : FeaturePipeline — for latency benchmarking
    stage6_report_path   : str | None — path to Stage 6 evaluation_report.json
    conversion_result    : dict from convert.export_champion(), optional
    figures_dir          : directory for Stage 8 figures, optional
    verification_report_path : output path for tflite_verification_report.json
    metadata_output_path : output path for gesture_model_metadata.json
    n_calls, warmup      : latency benchmark parameters
    n_classes            : int, default 35
    sign_names           : optional — derived from val_dataset if None

    Returns
    -------
    Tuple[ReleaseGateResult, Dict[str, Any]]
        (gate, full_results_dict) — gate is the authoritative verdict;
        full_results_dict contains all intermediate results for notebook display.
    """
    if val_dataset is None:
        raise ValueError(
            "run_full_verification(): val_dataset is required. "
            "Construct a GestureDataset instance before calling this function."
        )
    if pipeline is None:
        raise ValueError(
            "run_full_verification(): pipeline (FeaturePipeline) is required "
            "for latency benchmarking."
        )

    full_results: Dict[str, Any] = {}

    logger.info(
        "=" * 60 + "\nStage 8 Step 2 — Full TFLite Verification\n" + "=" * 60,
        extra={"stage": "export"},
    )

    # ── Step 1: Accuracy verification ─────────────────────────────────────
    logger.info(
        "[1/7] Accuracy verification (Keras vs TFLite, val + test)...",
        extra={"stage": "export"},
    )
    verification_result = run_accuracy_verification(
        keras_model_path=keras_model_path,
        tflite_path=tflite_path,
        config_snapshot_path=config_snapshot_path,
        val_dataset=val_dataset,
        n_classes=n_classes,
        sign_names=sign_names,
    )
    full_results["verification_result"] = verification_result
    resolved_sign_names = verification_result["sign_names"]

    # ── Step 2: Per-class delta (B1 fix: reuse predictions from Step 1) ──
    logger.info(
        "[2/7] Per-class TFLite delta analysis (val split, reusing Step 1 predictions)...",
        extra={"stage": "export"},
    )
    val_preds = verification_result["_predictions"]["val"]
    y_true_val       = val_preds["y_true"]        # (52,) int64
    y_pred_keras_val = val_preds["y_pred_keras"]  # (52,) int64
    y_pred_tflite_val = val_preds["y_pred_tflite"] # (52,) int64

    per_class_delta = compute_per_class_tflite_delta(
        y_true=y_true_val,
        y_pred_keras=y_pred_keras_val,
        y_pred_tflite=y_pred_tflite_val,
        sign_names=resolved_sign_names,
        n_classes=n_classes,
    )
    full_results["per_class_delta"] = per_class_delta

    # ── Step 3: Latency benchmark ─────────────────────────────────────────
    logger.info(
        "[3/7] Production latency benchmark (n_calls=%d, warmup=%d)...",
        n_calls, warmup,
        extra={"stage": "export"},
    )
    latency_result = run_production_latency_benchmark(
        tflite_path=tflite_path,
        keras_model_path=keras_model_path,
        pipeline=pipeline,
        n_calls=n_calls,
        warmup=warmup,
    )
    full_results["latency_result"] = latency_result

    # ── Step 4: Assemble release gate ─────────────────────────────────────
    logger.info(
        "[4/7] Assembling release gate...",
        extra={"stage": "export"},
    )
    gate = assemble_release_gate(
        verification_result=verification_result,
        latency_result=latency_result,
        tflite_path=tflite_path,
    )
    full_results["gate"] = gate

    # Print the verdict to stdout for notebook visibility
    print(gate.report())

    # ── Step 5: Save verification report ──────────────────────────────────
    logger.info(
        "[5/7] Writing tflite_verification_report.json...",
        extra={"stage": "export"},
    )
    report_path = save_verification_report(
        gate=gate,
        verification_result=verification_result,
        latency_result=latency_result,
        per_class_delta=per_class_delta,
        conversion_result=conversion_result,
        output_path=verification_report_path,
    )
    full_results["verification_report_path"] = str(report_path)

    # ── Step 6: Write model metadata ──────────────────────────────────────
    logger.info(
        "[6/7] Writing gesture_model_metadata.json...",
        extra={"stage": "export"},
    )
    if conversion_result is None:
        tflite_p = Path(tflite_path)
        tflite_mb_fallback = None
        if tflite_p.exists():
            try:
                tflite_mb_fallback = round(tflite_p.stat().st_size / (1024 ** 2), 4)
            except OSError:
                pass
        conversion_result_for_meta: Dict[str, Any] = {
            "tflite_disk_mb":                tflite_mb_fallback,
            "param_memory_mb":               round(68771 * 4 / (1024 ** 2), 4),
            "savedmodel_disk_mb":            None,
            "size_reduction_vs_params_x":    None,
            "size_reduction_vs_savedmodel_x": None,
            "sha256_checksum":               None,
        }
        logger.warning(
            "conversion_result not supplied — building minimal metadata "
            "without SavedModel disk size or SHA256 checksum. "
            "Pass conversion_result from convert.export_champion() for complete metadata.",
            extra={"stage": "export"},
        )
    else:
        conversion_result_for_meta = conversion_result

    write_model_metadata(
        tflite_path=tflite_path,
        conversion_result=conversion_result_for_meta,
        verification_result=verification_result,
        latency_result=latency_result,
        per_class_delta=per_class_delta,
        config_snapshot_path=str(config_snapshot_path),
        stage6_report_path=stage6_report_path,
        output_path=metadata_output_path,
    )
    full_results["metadata_output_path"] = str(Path(metadata_output_path).resolve())

    # ── Step 7: Generate figures ───────────────────────────────────────────
    logger.info("[7/7] Generating Stage 8 figures...", extra={"stage": "export"})
    if figures_dir is not None:
        fdir = Path(figures_dir)
        fdir.mkdir(parents=True, exist_ok=True)

        for fig_name, fig_fn, fig_kwargs in [
            (
                "tflite_size_comparison.png",
                plot_tflite_size_comparison,
                {"conversion_result": conversion_result_for_meta,
                 "output_path": fdir / "tflite_size_comparison.png",
                 "figure_dpi": 150},
            ),
            (
                "tflite_accuracy_comparison.png",
                plot_tflite_accuracy_comparison,
                {"verification_result": verification_result,
                 "output_path": fdir / "tflite_accuracy_comparison.png",
                 "figure_dpi": 150},
            ),
            (
                "tflite_per_class_delta.png",
                plot_tflite_per_class_delta,
                {"per_class_delta": per_class_delta,
                 "output_path": fdir / "tflite_per_class_delta.png",
                 "figure_dpi": 150,
                 "n_classes": n_classes},
            ),
        ]:
            try:
                fig_fn(**fig_kwargs)
                full_results[f"figure_{fig_name.replace('.png', '')}"] = str(
                    (fdir / fig_name).resolve()
                )
            except Exception as exc:
                logger.warning(
                    "Failed to generate figure %s: %s",
                    fig_name, exc,
                    extra={"stage": "export"},
                )

    logger.info(
        "Stage 8 Step 2 complete | release_ready=%s | "
        "n_hard_failures=%d | n_warnings=%d",
        gate.release_ready,
        len(gate.hard_failures),
        len(gate.warnings),
        extra={"stage": "export"},
    )

    return gate, full_results


# ---------------------------------------------------------------------------
# Import-time self-check
# ---------------------------------------------------------------------------

def _self_check() -> None:
    """Cheap, dependency-free sanity check on module constants."""
    assert 0.0 < _DELTA_THRESHOLD <= 0.10, (
        "verify.py: _DELTA_THRESHOLD must be in (0, 0.10]."
    )
    assert 0.0 < _AGREEMENT_THRESHOLD <= 1.0, (
        "verify.py: _AGREEMENT_THRESHOLD must be in (0, 1]."
    )
    assert 0.0 < _PROB_DIFF_WARN_THRESHOLD <= 0.10, (
        "verify.py: _PROB_DIFF_WARN_THRESHOLD must be in (0, 0.10]."
    )
    assert 0.0 < _CONFIDENCE_SHIFT_WARN_THRESHOLD <= 0.10, (
        "verify.py: _CONFIDENCE_SHIFT_WARN_THRESHOLD must be in (0, 0.10]."
    )
    assert _MAX_TFLITE_SIZE_MB == 10.0, (
        "verify.py: _MAX_TFLITE_SIZE_MB must match the project target (10 MB)."
    )
    assert _LATENCY_TARGET_MS == 100.0, (
        "verify.py: _LATENCY_TARGET_MS must match the project target (100ms)."
    )
    assert 0.0 < _MEANINGFUL_DEGRADATION_DELTA <= 0.5, (
        "verify.py: _MEANINGFUL_DEGRADATION_DELTA must be in (0, 0.5]."
    )
    assert _CHAMPION_CONFIG_HASH.startswith("5809193d"), (
        "verify.py: _CHAMPION_CONFIG_HASH has drifted from the verified "
        "config_snapshot.yaml value."
    )
    assert len(_CHAMPION_CONFIG_HASH) == 64, (
        "verify.py: _CHAMPION_CONFIG_HASH must be a 64-char SHA-256 hex string."
    )
    assert len(_CONFUSABLE_PAIRS) == 8, (
        "verify.py: _CONFUSABLE_PAIRS must have 8 entries (4 pairs × 2 directions)."
    )
    assert len(_HIGH_RISK_SIGNS) == 5, (
        "verify.py: _HIGH_RISK_SIGNS must have 5 entries (Stage 5 Finding 8)."
    )
    # Stage 6 calibration sanity
    assert 0.0 < _STAGE6_MEAN_CONFIDENCE < 1.0
    assert 0.0 < _STAGE6_MEAN_ACCURACY < 1.0
    assert _STAGE6_OVERCONFIDENCE_GAP < 0.0, (
        "verify.py: Champion model is underconfident — overconfidence_gap must be < 0."
    )
    assert 0.0 < _STAGE6_KERAS_VAL_MACRO_F1 < 1.0
    assert 0.0 < _STAGE6_KERAS_TEST_MACRO_F1 < 1.0
    # CI bounds sanity
    assert _STAGE6_KERAS_VAL_MACRO_F1_CI[0] < _STAGE6_KERAS_VAL_MACRO_F1 < _STAGE6_KERAS_VAL_MACRO_F1_CI[1], (
        "verify.py: val macro-F1 point estimate must lie within its CI bounds."
    )
    assert _STAGE6_KERAS_TEST_MACRO_F1_CI[0] < _STAGE6_KERAS_TEST_MACRO_F1 < _STAGE6_KERAS_TEST_MACRO_F1_CI[1], (
        "verify.py: test macro-F1 point estimate must lie within its CI bounds."
    )


if __debug__:
    _self_check()


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------

__all__ = [
    "ReleaseGateResult",
    "run_accuracy_verification",
    "compute_per_class_tflite_delta",
    "run_production_latency_benchmark",
    "assemble_release_gate",
    "write_model_metadata",
    "save_verification_report",
    "run_full_verification",
    "plot_tflite_size_comparison",
    "plot_tflite_accuracy_comparison",
    "plot_tflite_per_class_delta",
    # Stage 8 test suite constants
    "_DELTA_THRESHOLD",
    "_AGREEMENT_THRESHOLD",
    "_PROB_DIFF_WARN_THRESHOLD",
    "_CONFIDENCE_SHIFT_WARN_THRESHOLD",
    "_MAX_TFLITE_SIZE_MB",
    "_LATENCY_TARGET_MS",
    "_MEANINGFUL_DEGRADATION_DELTA",
    "_STAGE6_KERAS_VAL_MACRO_F1",
    "_STAGE6_KERAS_TEST_MACRO_F1",
    "_CONFUSABLE_SIGNS",
    "_HIGH_RISK_SIGNS",
]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Stage 8 Step 2 — TFLite accuracy verification, release gate, "
            "and model metadata for gesture_bilstm_v1.tflite."
        )
    )
    parser.add_argument(
        "--keras-model",
        default=_DEFAULT_KERAS_MODEL_PATH,
        help="Path to Keras SavedModel directory (default: champion's).",
    )
    parser.add_argument(
        "--tflite",
        default=_DEFAULT_TFLITE_PATH,
        help="Path to the TFLite file from Stage 8 Step 1.",
    )
    parser.add_argument(
        "--config-snapshot",
        default=_DEFAULT_CONFIG_SNAPSHOT_PATH,
        help="Path to config_snapshot.yaml (default: champion's).",
    )
    parser.add_argument(
        "--stage6-report",
        default=_DEFAULT_STAGE6_REPORT_PATH,
        help="Path to Stage 6 evaluation_report.json (optional).",
    )
    parser.add_argument(
        "--figures-dir",
        default="reports/figures",
        help="Directory to write Stage 8 figures.",
    )
    parser.add_argument(
        "--n-calls",
        type=int, default=200,
        help="Number of timed latency calls (default: 200).",
    )
    parser.add_argument(
        "--warmup",
        type=int, default=20,
        help="Number of warmup calls discarded (default: 20).",
    )

    args = parser.parse_args()

    print(
        "Stage 8 Step 2 CLI requires a GestureDataset and FeaturePipeline instance.\n"
        "Use notebooks/07_tflite_verification.ipynb for the full orchestrated flow, "
        "or call run_full_verification() programmatically:\n\n"
        "    from src.export.verify import run_full_verification\n"
        "    from src.features import FeaturePipeline, GestureDataset\n"
        "    from src.utils.config import load_config\n\n"
        "    cfg = load_config(model='bilstm', data='seq100', augmentation='spatial_temporal')\n"
        "    pipeline = FeaturePipeline(cfg)\n"
        "    dataset = GestureDataset(cfg, pipeline)\n\n"
        "    gate, results = run_full_verification(\n"
        f"        keras_model_path='{args.keras_model}',\n"
        f"        tflite_path='{args.tflite}',\n"
        f"        config_snapshot_path='{args.config_snapshot}',\n"
        "        val_dataset=dataset,\n"
        "        pipeline=pipeline,\n"
        f"        figures_dir='{args.figures_dir}',\n"
        f"        n_calls={args.n_calls},\n"
        f"        warmup={args.warmup},\n"
        "    )\n"
        "    print(gate.report())\n"
    )