"""
src/evaluation/calibration.py
==============================
Confidence calibration diagnostics for the WLASL 35-class gesture recognition
system. This is Stage 6 (Evaluation, Benchmarking, and Interpretability),
Phase A3, per the Stage 6 (Revised) plan.

Scope of this module (Phase A3, exactly)
------------------------------------------
The Stage 6 (Revised) plan defines Phase A3:

    "calibration.py — compute_reliability_diagram(),
    compute_confidence_threshold_curve() — operate on (y_true, y_prob),
    not y_pred. Built and unit-tested against synthetic probability arrays
    before touching real predictions."

Concretely, this module:

  1. Operates EXCLUSIVELY on already-extracted ``(y_true, y_proba)`` arrays —
     it NEVER calls ``model.predict()`` or touches the TF Dataset directly.
     Both arrays come from the Phase B1 prediction cache
     (``reports/evaluation/predictions/val_predictions.npz``), written by
     ``pipelines/run_evaluation.py`` using ``get_predictions(...,
     return_probs=True)`` from ``metrics.py``.

  2. Is completely framework-agnostic — only ``numpy`` and ``scipy`` (for
     bootstrap confidence intervals on bin accuracies). No TensorFlow import
     anywhere in this file.

  3. Is independently unit-testable with synthetic probability arrays and
     synthetic true labels, requiring no model, dataset, or project
     infrastructure.

  4. Never writes figures itself — it returns structured data dicts that
     the notebooks (``notebooks/06_evaluation_error_analysis.ipynb``) and
     ``pipelines/run_evaluation.py`` pass to dedicated plotting helpers
     (also in this module). Separating computation from rendering keeps the
     computation layer trivially testable and makes figure styling a
     presentation concern, not a metric concern.

Why calibration matters for this project
------------------------------------------
The champion model (``bilstm_hands_only_v4_aug``) achieves
``val_macro_f1=0.6011`` on 52 clips. The softmax probabilities it emits
are the raw output of a ``Dense(35, softmax)`` layer that was trained with
``sparse_categorical_crossentropy`` loss and no explicit calibration. Two
well-established empirical regularities apply:

  1. Softmax outputs from deep networks are systematically OVERCONFIDENT:
     when the model says "90% confident", the actual accuracy on those
     predictions tends to be considerably lower than 90%. This has been
     extensively documented (Guo et al., 2017, "On Calibration of Modern
     Neural Networks").

  2. Overconfidence is WORSE on small, imbalanced datasets. With 236
     training clips and 21 singleton validation classes, the model has very
     limited information to calibrate its uncertainty against — the softmax
     temperature is effectively set by gradient descent on the training
     distribution, not by the validation distribution the model is evaluated
     on.

The reliability diagram produced by ``compute_reliability_diagram()`` makes
this overconfidence quantitative: each bar shows the actual accuracy of
predictions that fell in a given confidence bucket, compared to the ideal
diagonal (where 70%-confident predictions should be right 70% of the time).
The Expected Calibration Error (ECE) summarises this as a single number.

The confidence-threshold curve produced by
``compute_confidence_threshold_curve()`` answers the deployment question:
"if I only accept predictions where the model says it is at least X%
confident, what accuracy do I achieve and what fraction of inputs do I have
to reject?" For a real-time ASL-to-KSL transfer scenario, this trade-off
between coverage and precision is operationally important.

Calibration methodology decisions
------------------------------------
Equal-width binning (fixed-width bins across [0, 1])
    The most common approach in the literature and easiest to explain to a
    client. The alternative — equal-mass binning (adaptive-width bins with
    equal sample counts) — would be more statistically stable given the
    tiny val set (52 clips) but is harder to interpret visually (the bar
    widths vary, and bin edges are data-dependent). Equal-width is the
    documented project choice.

n_bins=10 (default), configurable
    10 bins × 52 clips ≈ 5 clips/bin on average. This is very sparse —
    many bins will be empty or contain 1–2 clips. The module therefore:
    (a) reports ``bin_count`` for every bin so chart consumers can
    annotate the actual sample count above each bar rather than presenting
    a misleadingly smooth curve;
    (b) flags bins with ``is_sparse`` when ``bin_count < _SPARSE_BIN_THRESHOLD``
    (default: 5), so downstream consumers can style or annotate them
    differently;
    (c) supports ``n_bins=5`` as the "low-sample-size" alternative that
    ~10 clips/bin, which is still sparse but produces a coarser but more
    reliable diagram.

Max confidence per prediction (class with highest softmax probability)
    The reliability diagram uses only the WINNING class's softmax probability
    as the "model confidence" for that prediction. This is the standard
    definition and the one that corresponds to the argmax prediction that
    ``compute_macro_f1()`` in ``metrics.py`` is based on. It is NOT an
    average over all 35 class probabilities — that would be a meaningless
    mixed-class metric.

ECE = weighted average of |confidence - accuracy|
    Weighted by the fraction of samples in each non-empty bin (i.e. bins
    with zero samples are excluded from the average). This is the standard
    ECE formulation (Naeini et al., 2015). The unweighted version (average
    over non-empty bins regardless of their size) is also computed and
    reported as ``ece_unweighted`` for completeness.

Maximum Calibration Error (MCE)
    MCE = max over non-empty bins of |confidence - accuracy|. Reported
    alongside ECE because ECE can be dominated by many small-error high-
    mass bins while a single catastrophically miscalibrated high-confidence
    bin (a scenario plausible on this 52-clip val set) would be visible
    in MCE but obscured in ECE.

Temperature scaling note (documented-not-implemented)
    The standard post-hoc calibration fix is temperature scaling: dividing
    the logits (pre-softmax activations) by a learned scalar T before
    softmax, optimised on the validation set. This is NOT implemented here
    because (a) it requires access to the pre-softmax logits, which this
    module does not receive; (b) with 52 val clips, the temperature estimate
    would itself be highly unreliable; (c) Stage 6's goal is DIAGNOSIS, not
    calibration repair — the repair belongs in a future model iteration.
    The limitation and the remedy are documented in LIMITATIONS.md.

Post-review disposition notes
-------------------------------
This module was designed from scratch with the lessons of the ``metrics.py``
and ``benchmark.py`` peer-review process in mind. The following design
decisions reflect those lessons explicitly:

  1. All validation helpers follow the same pattern as ``metrics.py``:
     ``_validate_class_count()``, ``_to_proba_array()``, etc. — one
     validation point per input, shared across all public functions.

  2. ``compute_reliability_diagram()`` returns ``is_sparse`` flags per bin
     (analogous to ``is_singleton`` in ``metrics.py``) so downstream
     consumers never have to re-derive sparseness from the count alone.

  3. Empty bins are represented in the output dict with ``None`` values (not
     0.0, not NaN) for ``mean_confidence`` and ``actual_accuracy``, so a
     chart consumer can explicitly decide how to handle them (skip, draw
     empty rectangle, label as "0 samples") without guessing the module's
     intent.

  4. Threshold curve uses a linspace over [0, 1] that always INCLUDES both
     endpoints so that ``coverage(0) == 1.0`` (all predictions accepted)
     and ``coverage(1) = 0.0 or small`` (only perfect-confidence
     predictions, usually 0) are always present in the output.

  5. The ``compute_calibration_summary()`` consolidation function (analogous
     to ``compute_evaluation_summary()`` in ``metrics.py``) bundles both
     results so ``run_evaluation.py`` can call it once and receive a single
     JSON-serialisable dict for ``evaluation_report.json``.

Champion model context
------------------------
  - Input shape:   (1, 100, 126) — seq_len=100, landmark_config=hands_only
  - Output shape:  (1, 35) — softmax probabilities
  - val_macro_f1:  0.6011 (52 clips, 7 unseen signers)
  - early_stopping_monitor: val_accuracy (NOTE: handoff narrates
    val_macro_f1 — the actual config_snapshot.yaml says val_accuracy.
    This discrepancy is flagged in evaluation_report.json per Phase F
    requirements; this module takes no position on it.)
  - ``y_proba`` consumed here comes from:
    ``get_predictions(model, val_ds, return_probs=True)`` in metrics.py,
    which returns float64 arrays of shape (52, 35).

Module-level exports
----------------------
    compute_reliability_diagram          — bin predictions by confidence,
                                           compute actual accuracy per bin,
                                           ECE, MCE
    compute_confidence_threshold_curve   — coverage vs accuracy vs mean
                                           confidence across confidence
                                           thresholds τ ∈ [0, 1]
    compute_calibration_summary          — consolidation wrapper (both above
                                           + metadata) for evaluation_report
    plot_reliability_diagram             — figure renderer (reliability diagram)
    plot_confidence_threshold_curve      — figure renderer (coverage/accuracy
                                           trade-off)
    SPARSE_BIN_THRESHOLD                 — public constant: bins with fewer
                                           samples than this are flagged sparse
    DEFAULT_N_BINS                       — default bin count (10)
    DEFAULT_N_THRESHOLD_POINTS           — default τ grid density (101)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Default number of equal-width confidence bins for the reliability diagram.
#: 10 bins × 52 val clips ≈ 5.2 clips/bin — very sparse. See module docstring.
DEFAULT_N_BINS: int = 10

#: Default number of threshold points for the confidence-threshold curve.
#: 101 points gives a 1-percentage-point resolution across [0, 1], which is
#: sufficient for a 52-clip val set where the effective resolution is much
#: coarser (each clip contributes ~1.9pp to accuracy).
DEFAULT_N_THRESHOLD_POINTS: int = 101

#: Bins with fewer than this many samples are flagged as sparse in the
#: reliability diagram output. With the WLASL val set (52 clips, 10 bins),
#: most bins will be sparse. The threshold of 5 is a practical floor below
#: which per-bin accuracy estimates are dominated by single-sample noise.
SPARSE_BIN_THRESHOLD: int = 5

#: Number of bootstrap resamples for per-bin accuracy confidence intervals.
#: Lower than metrics.py's DEFAULT_N_BOOTSTRAP (1000) because we call this
#: per-bin inside compute_reliability_diagram(), so total cost is
#: DEFAULT_N_BINS × DEFAULT_N_BINS_CI_BOOTSTRAP = 10 × 200 = 2000 calls.
#: Each resample on ≤52 samples is microsecond-scale, so total overhead
#: is negligible.
_N_BINS_CI_BOOTSTRAP: int = 200

#: Bootstrap confidence level for per-bin accuracy CIs. 80% rather than 90%
#: because the per-bin sample counts are so small (often 1–5) that a 90%
#: interval would typically span [0, 1] and communicate nothing useful.
_BIN_CI_LEVEL: float = 0.80

#: Project global seed, matching DEFAULT_SEED in metrics.py and the global
#: seed in base.yaml. Used for all bootstrap resampling here.
DEFAULT_SEED: int = 42

#: Below this many total samples, reliability diagram computation is
#: technically possible but statistically meaningless — warn rather than raise.
_MIN_SAMPLES_FOR_CALIBRATION: int = 5

#: Coverage below which a threshold is considered "effectively rejecting
#: everything" — flagged in the threshold curve output.
_MIN_MEANINGFUL_COVERAGE: float = 0.05

#: The calibration limitation string embedded in every summary dict.
#: Mirrors the _BOOTSTRAP_SIGNER_CAVEAT pattern in metrics.py so every
#: downstream consumer (evaluation_report.json, LIMITATIONS.md) inherits
#: it automatically.
_CALIBRATION_CAVEAT: str = (
    "Calibration estimates are based on 52 validation clips. At this sample "
    "size, most confidence bins contain fewer than 5 samples, making per-bin "
    "accuracy estimates highly unreliable. ECE and MCE should be treated as "
    "rough indicators of calibration quality, not precise numerical estimates. "
    "Temperature scaling (dividing pre-softmax logits by a scalar T) is the "
    "standard post-hoc calibration remedy but requires access to pre-softmax "
    "activations and a larger calibration set to be reliable — see "
    "LIMITATIONS.md."
)

#: Documented-not-implemented note for temperature scaling, to be embedded
#: in LIMITATIONS.md and evaluation_report.json verbatim.
TEMPERATURE_SCALING_NOTE: str = (
    "Temperature scaling mitigation: The recommended post-hoc calibration "
    "approach is temperature scaling (Guo et al., 2017): divide pre-softmax "
    "logits by a learned scalar T > 1 (for overconfident models) before "
    "applying softmax. This is not implemented in Stage 6 because (a) this "
    "module receives post-softmax probabilities, not logits; (b) with 52 "
    "val clips, the temperature estimate would be unreliable; (c) Stage 6 "
    "goal is diagnosis, not repair. A future model iteration should add a "
    "calibration head or apply Platt scaling on a dedicated calibration split."
)


# ---------------------------------------------------------------------------
# Internal validation helpers
# ---------------------------------------------------------------------------

def _validate_n_bins(n_bins: int, caller: str) -> None:
    """Raise ValueError if n_bins is not a positive integer >= 2."""
    if not isinstance(n_bins, (int, np.integer)) or n_bins < 2:
        raise ValueError(
            f"{caller}: n_bins={n_bins!r} must be an integer >= 2. "
            f"Default is {DEFAULT_N_BINS}. "
            "With 52 val clips, n_bins=5 is recommended for more "
            "reliable (if coarser) bin estimates."
        )


def _validate_class_count(n_classes: int, caller: str) -> None:
    """Raise ValueError if n_classes is not a sane positive integer >= 2."""
    if not isinstance(n_classes, (int, np.integer)) or n_classes < 2:
        raise ValueError(
            f"{caller}: n_classes={n_classes!r} must be an integer >= 2. "
            "Pass cfg.num_classes explicitly (35 for the current WLASL label map)."
        )


def _to_label_array(arr: Any, name: str) -> np.ndarray:
    """
    Coerce an array-like into a flat 1-D int64 numpy array of class indices.

    Mirrors the same helper in metrics.py for consistency. Accepts numpy
    arrays, Python lists, and TF EagerTensors (which implement __array__).

    Raises
    ------
    ValueError
        If ``arr`` is empty or has a shape that suggests one-hot encoding.
    """
    out = np.asarray(arr)
    if out.size == 0:
        raise ValueError(f"{name} is empty. Cannot compute calibration metrics.")
    if out.ndim > 1:
        squeezed = np.squeeze(out)
        if squeezed.ndim > 1:
            raise ValueError(
                f"{name} has shape {out.shape}, which looks like one-hot or "
                "multi-dimensional label encoding. Expected flat integer class "
                "indices, e.g. from GestureDataset (sparse_categorical labels). "
                "Convert with np.argmax(arr, axis=-1) if one-hot."
            )
        out = squeezed
    if out.ndim == 0:
        out = out.reshape(1)
    return out.astype(np.int64)


def _to_proba_array(arr: Any, n_classes: int, caller: str) -> np.ndarray:
    """
    Coerce an array-like into a 2-D float64 (n_samples, n_classes) probability
    array. Validates shape, value range, and approximate row-sum normalisation.

    Parameters
    ----------
    arr      : array-like, shape (n_samples, n_classes)
    n_classes: int
    caller   : str — used in error messages

    Returns
    -------
    np.ndarray, dtype float64, shape (n_samples, n_classes)

    Raises
    ------
    ValueError
        If shape is wrong, values are outside [0, 1], or rows don't sum to ~1.
    """
    out = np.asarray(arr, dtype=np.float64)

    if out.ndim != 2:
        raise ValueError(
            f"{caller}: y_proba must be 2-D (n_samples, n_classes), "
            f"got shape {out.shape}. "
            "This array should come from get_predictions(..., return_probs=True) "
            "in metrics.py, which always returns a (n_samples, n_classes) array."
        )

    if out.shape[1] != n_classes:
        raise ValueError(
            f"{caller}: y_proba has {out.shape[1]} columns but n_classes={n_classes}. "
            "Ensure n_classes matches the label map used during inference "
            "(35 for the current WLASL project, artifacts/label_map_v1.json)."
        )

    if out.size == 0:
        raise ValueError(
            f"{caller}: y_proba is empty (shape {out.shape}). "
            "Check that the prediction cache was written correctly."
        )

    # Value range: softmax outputs are always in [0, 1], but floating-point
    # arithmetic can produce tiny negatives or values slightly above 1.
    # Use a tolerance to avoid false positives on numerically clean outputs.
    _tol = 1e-6
    if out.min() < -_tol or out.max() > 1.0 + _tol:
        raise ValueError(
            f"{caller}: y_proba contains values outside [0, 1] "
            f"(min={out.min():.6f}, max={out.max():.6f}). "
            "Expected softmax probabilities in [0, 1]. "
            "If this is a logits array, apply softmax before passing it here."
        )

    # Row-sum check: softmax rows should sum to 1 within floating-point tolerance.
    row_sums = out.sum(axis=1)
    max_deviation = float(np.abs(row_sums - 1.0).max())
    if max_deviation > 1e-3:
        raise ValueError(
            f"{caller}: y_proba rows do not sum to 1.0 "
            f"(max |row_sum - 1| = {max_deviation:.6f}). "
            "Expected softmax probabilities. Check that the model's final "
            "layer is Dense(n_classes, activation='softmax')."
        )

    return out


def _validate_equal_length(a: np.ndarray, b: np.ndarray,
                            name_a: str, name_b: str) -> None:
    """Raise ValueError if two arrays differ in length (first dimension)."""
    if len(a) != len(b):
        raise ValueError(
            f"{name_a} (len={len(a)}) and {name_b} (len={len(b)}) must have "
            "the same length — one entry per clip. Check that y_true and "
            "y_proba came from the same prediction cache."
        )


def _validate_label_range(y_true: np.ndarray, n_classes: int, caller: str) -> None:
    """Raise ValueError if any label falls outside [0, n_classes)."""
    if y_true.size == 0:
        return
    min_v, max_v = int(y_true.min()), int(y_true.max())
    if min_v < 0 or max_v >= n_classes:
        bad = max_v if max_v >= n_classes else min_v
        raise ValueError(
            f"{caller}: y_true contains class index {bad}, outside [0, {n_classes}). "
            "Check that n_classes matches the label map "
            "(artifacts/label_map_v1.json — 35 signs for this project)."
        )


# ---------------------------------------------------------------------------
# Core calibration computations
# ---------------------------------------------------------------------------

def _extract_max_confidence(y_proba: np.ndarray) -> np.ndarray:
    """
    Extract the maximum softmax probability per prediction.

    This is the "model confidence" used throughout this module: for each
    prediction, the confidence is the softmax probability assigned to the
    winning class (argmax class). This is the natural companion to the
    argmax prediction that drives macro-F1 in metrics.py.

    Parameters
    ----------
    y_proba : np.ndarray, shape (n_samples, n_classes), float64

    Returns
    -------
    np.ndarray, shape (n_samples,), float64 — max probability per row.
    """
    return y_proba.max(axis=1)


def _is_correct(y_true: np.ndarray, y_proba: np.ndarray) -> np.ndarray:
    """
    Boolean array indicating whether argmax(y_proba) == y_true for each sample.

    Parameters
    ----------
    y_true  : np.ndarray, shape (n_samples,), int64
    y_proba : np.ndarray, shape (n_samples, n_classes), float64

    Returns
    -------
    np.ndarray, shape (n_samples,), bool
    """
    y_pred = y_proba.argmax(axis=1).astype(np.int64)
    return y_pred == y_true


def _bootstrap_bin_accuracy_ci(
    correct_flags: np.ndarray,
    ci_level: float = _BIN_CI_LEVEL,
    n_bootstrap: int = _N_BINS_CI_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
) -> Tuple[float, float]:
    """
    Compute a bootstrap confidence interval for the accuracy of a single bin.

    Parameters
    ----------
    correct_flags : np.ndarray, shape (n_bin_samples,), bool
        True where the prediction was correct for samples in this bin.
    ci_level  : float, confidence level in (0, 1)
    n_bootstrap: int, number of bootstrap resamples
    seed       : int, RNG seed for reproducibility

    Returns
    -------
    Tuple[float, float]
        (ci_lower, ci_upper) — percentile-based interval. Both None-safe:
        if n_bin_samples == 0, returns (0.0, 0.0). If n_bin_samples == 1,
        returns (0.0, 1.0) (degenerate case: either 0% or 100% is "correct").
    """
    n = len(correct_flags)
    if n == 0:
        return (0.0, 0.0)
    if n == 1:
        # Single sample: accuracy is deterministically 0.0 or 1.0; CI is trivial.
        acc = float(correct_flags[0])
        return (acc, acc)

    rng = np.random.default_rng(seed)
    boot_acc = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_acc[i] = correct_flags[idx].mean()

    alpha = 1.0 - ci_level
    ci_lower = float(np.percentile(boot_acc, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_acc, 100.0 * (1.0 - alpha / 2.0)))
    return (ci_lower, ci_upper)


def compute_reliability_diagram(
    y_true: Any,
    y_proba: Any,
    n_classes: int,
    n_bins: int = DEFAULT_N_BINS,
    seed: int = DEFAULT_SEED,
    compute_bin_ci: bool = True,
    ci_level: float = _BIN_CI_LEVEL,
) -> Dict[str, Any]:
    """
    Compute a reliability diagram (calibration curve) for a multi-class
    classifier using the maximum-confidence convention.

    For each confidence bin b = [b_lo, b_hi):
      - ``mean_confidence``  : average max-softmax-probability of predictions
                               falling in this bin (x-axis of the diagram).
      - ``actual_accuracy``  : fraction of those predictions that were correct
                               (y-axis of the diagram). None if bin is empty.
      - ``bin_count``        : number of predictions in the bin.
      - ``is_sparse``        : True if bin_count < SPARSE_BIN_THRESHOLD (5).
      - ``is_empty``         : True if bin_count == 0.
      - ``calibration_gap``  : actual_accuracy - mean_confidence. Positive =
                               underconfident; negative = overconfident. None
                               if bin is empty.
      - ``ci_lower/upper``   : bootstrap CI on actual_accuracy (only if
                               compute_bin_ci=True and bin is non-empty).

    Aggregate statistics (ECE, MCE, ACE):
      - ``ece``              : Expected Calibration Error — weighted average
                               |confidence - accuracy| over non-empty bins,
                               weighted by bin sample fraction.
      - ``ece_unweighted``   : Unweighted average |confidence - accuracy| over
                               non-empty bins (all bins count equally).
      - ``mce``              : Maximum Calibration Error — max |conf - acc|
                               over non-empty bins.
      - ``mean_confidence``  : Global mean max-softmax probability (average
                               over all samples, not bins).
      - ``mean_accuracy``    : Overall accuracy (fraction of correct argmax
                               predictions). This matches ``compute_accuracy()``
                               in metrics.py.
      - ``overconfidence_gap``: mean_confidence - mean_accuracy. Positive
                               implies overconfidence. Expected positive for
                               this model.
      - ``n_empty_bins``     : Number of bins with zero samples (expected to
                               be high on the 52-clip val set).
      - ``n_sparse_bins``    : Number of bins with fewer than
                               ``SPARSE_BIN_THRESHOLD`` samples.

    Stage 6 context
    ----------------
    With the champion model's 52 val clips and ``n_bins=10``:
      - Expected mean confidence: ~0.55–0.75 (softmax peak for a 35-class
        model that gets ~58% of predictions right).
      - Expected overconfidence_gap: +0.10 to +0.20 (model says 65%
        confident on average, but is actually right ~58% of the time).
      - Expected n_empty_bins: 3–6 (most confidence mass in [0.4, 0.9]).
      - ECE: expected ~0.10–0.20 on this small, imbalanced dataset.

    Parameters
    ----------
    y_true   : array-like, shape (n_samples,)
        True class indices (int). One entry per validation clip.
    y_proba  : array-like, shape (n_samples, n_classes)
        Softmax probability arrays. Come from
        ``get_predictions(..., return_probs=True)`` in metrics.py.
    n_classes: int — number of output classes (35).
    n_bins   : int, default 10
        Number of equal-width confidence bins spanning [0, 1].
        Recommended: 5 for this 52-clip val set (more stable), 10 for
        the diagram shape that matches most published reliability diagrams.
    seed     : int, default 42 — for bootstrap CI resampling.
    compute_bin_ci : bool, default True
        If True, compute bootstrap CI for each non-empty bin's accuracy.
        Set False for a fast smoke-test run during development.
    ci_level : float in (0, 1), default 0.80
        Bootstrap confidence level for per-bin accuracy CIs.
        80% (not 90%) because per-bin sample counts are so small that
        a 90% interval often spans [0, 1] and communicates nothing useful.

    Returns
    -------
    dict with keys:
        bins         : List[dict] — one dict per bin (ordered by confidence).
                       Each bin dict has: bin_index, bin_lo, bin_hi,
                       mean_confidence, actual_accuracy, bin_count,
                       is_sparse, is_empty, calibration_gap, ci_lower,
                       ci_upper (latter two None if empty or CI skipped).
        ece          : float — Expected Calibration Error.
        ece_unweighted: float — Unweighted ECE.
        mce          : float — Maximum Calibration Error.
        mean_confidence  : float — global mean max probability.
        mean_accuracy    : float — global fraction correct.
        overconfidence_gap: float — mean_confidence - mean_accuracy.
        n_bins           : int — total number of bins (including empty).
        n_empty_bins     : int
        n_sparse_bins    : int
        n_samples        : int
        n_classes        : int
        ci_level         : float — CI level used (or None if ci skipped).
        caveat           : str — calibration reliability caveat.
        temperature_scaling_note: str — documented-not-implemented note.

    Raises
    ------
    ValueError
        If n_bins < 2, n_classes < 2, arrays are empty, shapes mismatch,
        label values are out of range, or y_proba doesn't look like softmax.
    """
    # ── Validate inputs ───────────────────────────────────────────────────
    _validate_n_bins(n_bins, "compute_reliability_diagram")
    _validate_class_count(n_classes, "compute_reliability_diagram")

    y_true_arr  = _to_label_array(y_proba if False else y_true, "y_true")
    y_proba_arr = _to_proba_array(y_proba, n_classes, "compute_reliability_diagram")
    _validate_equal_length(y_true_arr, y_proba_arr, "y_true", "y_proba")
    _validate_label_range(y_true_arr, n_classes, "compute_reliability_diagram")

    n_samples = len(y_true_arr)
    if n_samples < _MIN_SAMPLES_FOR_CALIBRATION:
        logger.warning(
            f"compute_reliability_diagram(): n_samples={n_samples} is below "
            f"{_MIN_SAMPLES_FOR_CALIBRATION}. Calibration estimates will be "
            "statistically meaningless. Proceeding anyway.",
            extra={"stage": "evaluation"},
        )

    # ── Derived per-sample quantities ─────────────────────────────────────
    max_conf  = _extract_max_confidence(y_proba_arr)   # shape (n_samples,)
    correct   = _is_correct(y_true_arr, y_proba_arr)   # shape (n_samples,)

    # Bin edges: n_bins+1 points from 0.0 to 1.0 inclusive.
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    # ── Per-bin computation ───────────────────────────────────────────────
    bins_data: List[Dict[str, Any]] = []
    ece_numerator_sum   = 0.0   # sum of (bin_fraction × |gap|) for ECE
    ece_unweighted_gaps: List[float] = []
    mce = 0.0

    for b in range(n_bins):
        lo = float(bin_edges[b])
        hi = float(bin_edges[b + 1])

        # Half-open [lo, hi) except last bin which is fully closed [lo, hi].
        # This exactly mirrors scikit-learn's calibration_curve behaviour.
        if b < n_bins - 1:
            in_bin = (max_conf >= lo) & (max_conf < hi)
        else:
            # Last bin: include hi=1.0 (softmax outputs can be exactly 1.0
            # when one class dominates all floating-point mass).
            in_bin = (max_conf >= lo) & (max_conf <= hi)

        bin_count = int(in_bin.sum())
        is_empty  = bin_count == 0
        is_sparse = (not is_empty) and (bin_count < SPARSE_BIN_THRESHOLD)

        if is_empty:
            bin_dict: Dict[str, Any] = {
                "bin_index":        b,
                "bin_lo":           round(lo, 6),
                "bin_hi":           round(hi, 6),
                "mean_confidence":  None,
                "actual_accuracy":  None,
                "bin_count":        0,
                "is_sparse":        False,
                "is_empty":         True,
                "calibration_gap":  None,
                "ci_lower":         None,
                "ci_upper":         None,
            }
        else:
            bin_conf_vals = max_conf[in_bin]
            bin_correct   = correct[in_bin]

            bin_mean_conf = float(bin_conf_vals.mean())
            bin_acc       = float(bin_correct.mean())
            gap           = bin_acc - bin_mean_conf
            abs_gap       = abs(gap)

            # ECE contribution: weighted by bin fraction
            bin_fraction = bin_count / n_samples
            ece_numerator_sum += bin_fraction * abs_gap
            ece_unweighted_gaps.append(abs_gap)

            # MCE: track the worst-case bin
            mce = max(mce, abs_gap)

            # Per-bin bootstrap CI on accuracy
            ci_lower_val: Optional[float] = None
            ci_upper_val: Optional[float] = None
            if compute_bin_ci:
                ci_lower_val, ci_upper_val = _bootstrap_bin_accuracy_ci(
                    bin_correct, ci_level=ci_level,
                    n_bootstrap=_N_BINS_CI_BOOTSTRAP, seed=seed ^ b,
                )

            bin_dict = {
                "bin_index":        b,
                "bin_lo":           round(lo, 6),
                "bin_hi":           round(hi, 6),
                "mean_confidence":  round(bin_mean_conf, 6),
                "actual_accuracy":  round(bin_acc, 6),
                "bin_count":        bin_count,
                "is_sparse":        is_sparse,
                "is_empty":         False,
                "calibration_gap":  round(gap, 6),
                "ci_lower":         round(ci_lower_val, 6) if ci_lower_val is not None else None,
                "ci_upper":         round(ci_upper_val, 6) if ci_upper_val is not None else None,
            }

        bins_data.append(bin_dict)

    # ── Aggregate statistics ──────────────────────────────────────────────
    n_empty_bins  = sum(1 for bd in bins_data if bd["is_empty"])
    n_sparse_bins = sum(1 for bd in bins_data if bd["is_sparse"])

    ece              = round(float(ece_numerator_sum), 6)
    ece_unweighted   = (
        round(float(np.mean(ece_unweighted_gaps)), 6)
        if ece_unweighted_gaps else 0.0
    )
    mce_final        = round(float(mce), 6)
    mean_confidence  = round(float(max_conf.mean()), 6)
    mean_accuracy    = round(float(correct.mean()), 6)
    overconf_gap     = round(float(mean_confidence - mean_accuracy), 6)

    result: Dict[str, Any] = {
        "bins":                   bins_data,
        "ece":                    ece,
        "ece_unweighted":         ece_unweighted,
        "mce":                    mce_final,
        "mean_confidence":        mean_confidence,
        "mean_accuracy":          mean_accuracy,
        "overconfidence_gap":     overconf_gap,
        "n_bins":                 n_bins,
        "n_empty_bins":           n_empty_bins,
        "n_sparse_bins":          n_sparse_bins,
        "n_samples":              n_samples,
        "n_classes":              n_classes,
        "ci_level":               ci_level if compute_bin_ci else None,
        "caveat":                 _CALIBRATION_CAVEAT,
        "temperature_scaling_note": TEMPERATURE_SCALING_NOTE,
    }

    overconf_direction = "overconfident" if overconf_gap > 0 else "underconfident"
    logger.info(
        f"compute_reliability_diagram() | "
        f"n_samples={n_samples} | n_bins={n_bins} | "
        f"ECE={ece:.4f} | ECE_unweighted={ece_unweighted:.4f} | "
        f"MCE={mce_final:.4f} | "
        f"mean_confidence={mean_confidence:.4f} | "
        f"mean_accuracy={mean_accuracy:.4f} | "
        f"overconfidence_gap={overconf_gap:+.4f} ({overconf_direction}) | "
        f"n_empty_bins={n_empty_bins}/{n_bins} | "
        f"n_sparse_bins={n_sparse_bins}/{n_bins}",
        extra={"stage": "evaluation"},
    )

    if overconf_gap > 0.05:
        logger.warning(
            f"compute_reliability_diagram(): model is overconfident by "
            f"{overconf_gap:.4f} on average (mean_confidence={mean_confidence:.4f}, "
            f"mean_accuracy={mean_accuracy:.4f}). "
            "This is the expected direction for softmax models on small datasets. "
            "See LIMITATIONS.md for temperature scaling discussion.",
            extra={"stage": "evaluation"},
        )

    if n_empty_bins > n_bins // 2:
        logger.warning(
            f"compute_reliability_diagram(): {n_empty_bins}/{n_bins} bins are "
            "empty. With 52 val clips and 10 bins the confidence mass is "
            "concentrated in a narrow range. Consider n_bins=5 for a coarser "
            "but more populated diagram.",
            extra={"stage": "evaluation"},
        )

    return result


def compute_confidence_threshold_curve(
    y_true: Any,
    y_proba: Any,
    n_classes: int,
    n_threshold_points: int = DEFAULT_N_THRESHOLD_POINTS,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    """
    Compute the confidence-threshold curve: how coverage and accuracy change
    as we vary the minimum-confidence threshold τ ∈ [0, 1].

    For each threshold τ:
      - A prediction is ACCEPTED if ``max(softmax(x)) >= τ``.
      - ``coverage(τ)``   : fraction of all predictions accepted.
      - ``accuracy(τ)``   : accuracy on accepted predictions. None if no
                            predictions are accepted (coverage == 0).
      - ``n_accepted(τ)`` : absolute count of accepted predictions.
      - ``macro_f1(τ)``   : sklearn macro-F1 on accepted predictions, forcing
                            all n_classes into the denominator (consistent with
                            metrics.py). None if coverage == 0.
      - ``mean_confidence(τ)`` : mean max-softmax of accepted predictions.
                                  None if coverage == 0.

    Design note: τ = 0.0 always gives coverage=1.0 (all predictions accepted),
    and τ = 1.0 typically gives coverage near 0 (only exact-1.0 softmax
    predictions, which may be 0). Both endpoints are always included in the
    grid so the full trade-off curve is always visible.

    Deployment relevance
    ---------------------
    For a real-time gesture recognition system (Stage 9 webcam demo), the
    operator can set a threshold τ to achieve a desired precision level at
    the cost of some coverage. The curve quantifies this trade-off:

      - At τ=0.0: 100% coverage, ~58% accuracy (champion's val_acc).
      - At τ=0.5: moderate filtering. Typical improvement: +5–15% accuracy
        at the cost of 20–40% reduced coverage on a 35-class system.
      - At τ=0.8: high-confidence regime. Accuracy likely >80%, but only
        a small fraction of predictions accepted (most of the 35 rare classes
        will be below threshold).

    The accompanying figure (``reports/figures/confidence_threshold_curve.png``)
    shows both curves on a single plot so the operator can identify the
    "elbow" where accuracy gains become marginal relative to coverage cost.

    Area Under Curve (AUC-coverage)
    --------------------------------
    The area under the accuracy-vs-coverage curve (AUC, computed with the
    trapezoidal rule) is reported as a single-number quality metric. A
    perfectly calibrated model with 100% accuracy everywhere would have
    AUC=1.0; a random classifier would have AUC = 1/n_classes. The
    champion's expected AUC is ~0.55–0.65 (accuracy rises significantly
    as coverage drops, but never reaches 100%).

    Parameters
    ----------
    y_true   : array-like, shape (n_samples,)
    y_proba  : array-like, shape (n_samples, n_classes), softmax probs
    n_classes: int — 35 for this project.
    n_threshold_points : int, default 101
        Number of evenly spaced threshold values in [0, 1] (inclusive).
        101 gives 1% resolution. With 52 clips, the effective resolution
        is much coarser — each clip is 1.9% of accuracy — so 101 points
        is sufficient to capture all meaningful transitions.
    seed     : int, default 42 — not used for randomness here, kept for
               API consistency with compute_reliability_diagram().

    Returns
    -------
    dict with keys:
        thresholds         : List[float] — τ values (n_threshold_points,).
        coverage           : List[float] — fraction accepted at each τ.
        accuracy           : List[float | None] — accuracy on accepted at each τ.
        macro_f1           : List[float | None] — macro-F1 on accepted at each τ.
        n_accepted         : List[int] — count accepted at each τ.
        mean_confidence    : List[float | None] — mean max-prob of accepted.
        auc_coverage       : float — area under accuracy-vs-coverage curve
                             (trapezoidal rule, excluding None entries).
        optimal_threshold_accuracy : dict — threshold that maximises accuracy
                             (may not exist if coverage=0 at all τ > 0).
        optimal_threshold_f1       : dict — threshold that maximises macro-F1.
        n_threshold_points : int
        n_samples          : int
        n_classes          : int
        caveat             : str

    Raises
    ------
    ValueError
        If n_threshold_points < 2, arrays are invalid per standard checks.
    """
    # ── Validate inputs ───────────────────────────────────────────────────
    if not isinstance(n_threshold_points, (int, np.integer)) or n_threshold_points < 2:
        raise ValueError(
            f"compute_confidence_threshold_curve(): "
            f"n_threshold_points={n_threshold_points!r} must be >= 2. "
            f"Default is {DEFAULT_N_THRESHOLD_POINTS}."
        )
    _validate_class_count(n_classes, "compute_confidence_threshold_curve")

    y_true_arr  = _to_label_array(y_true, "y_true")
    y_proba_arr = _to_proba_array(y_proba, n_classes, "compute_confidence_threshold_curve")
    _validate_equal_length(y_true_arr, y_proba_arr, "y_true", "y_proba")
    _validate_label_range(y_true_arr, n_classes, "compute_confidence_threshold_curve")

    n_samples = len(y_true_arr)
    if n_samples < _MIN_SAMPLES_FOR_CALIBRATION:
        logger.warning(
            f"compute_confidence_threshold_curve(): n_samples={n_samples} "
            f"is below {_MIN_SAMPLES_FOR_CALIBRATION}.",
            extra={"stage": "evaluation"},
        )

    # ── Per-sample derived quantities ─────────────────────────────────────
    max_conf = _extract_max_confidence(y_proba_arr)   # (n_samples,)
    y_pred   = y_proba_arr.argmax(axis=1).astype(np.int64)
    correct  = (y_pred == y_true_arr)

    # Import sklearn f1_score here — keeps the module's core numpy-only
    # contract while allowing the richer per-threshold macro-F1 computation.
    from sklearn.metrics import f1_score as _f1_score

    labels_range = list(range(n_classes))
    thresholds   = np.linspace(0.0, 1.0, n_threshold_points)

    # ── Per-threshold computation ─────────────────────────────────────────
    threshold_list:   List[float]               = []
    coverage_list:    List[float]               = []
    accuracy_list:    List[Optional[float]]     = []
    macro_f1_list:    List[Optional[float]]     = []
    n_accepted_list:  List[int]                 = []
    mean_conf_list:   List[Optional[float]]     = []

    for tau in thresholds:
        accepted = max_conf >= tau
        n_acc    = int(accepted.sum())

        threshold_list.append(round(float(tau), 6))
        coverage_list.append(round(float(n_acc / n_samples), 6))
        n_accepted_list.append(n_acc)

        if n_acc == 0:
            accuracy_list.append(None)
            macro_f1_list.append(None)
            mean_conf_list.append(None)
        else:
            acc_val = round(float(correct[accepted].mean()), 6)
            accuracy_list.append(acc_val)

            # macro-F1 on the accepted subset.
            # We force all n_classes labels into the denominator (same as
            # metrics.py) so that this number is directly comparable to the
            # full-coverage macro-F1 reported there. Classes absent from the
            # accepted subset contribute F1=0 (zero_division=0).
            f1_val = round(float(_f1_score(
                y_true_arr[accepted],
                y_pred[accepted],
                average="macro",
                labels=labels_range,
                zero_division=0,
            )), 6)
            macro_f1_list.append(f1_val)

            mean_conf_list.append(round(float(max_conf[accepted].mean()), 6))

    # ── Aggregate statistics ──────────────────────────────────────────────
    # AUC under accuracy-vs-coverage curve (trapezoidal rule).
    # Exclude (threshold, None) points. Sort by coverage ascending for a
    # well-defined monotone x-axis.
    valid_pairs = [
        (cov, acc)
        for cov, acc in zip(coverage_list, accuracy_list)
        if acc is not None
    ]
    # Sort by coverage ASCENDING (τ increases → coverage decreases, so
    # iterating thresholds gives coverage decreasing; we sort ascending for AUC).
    valid_pairs.sort(key=lambda p: p[0])
    if len(valid_pairs) >= 2:
        cov_arr = np.array([p[0] for p in valid_pairs])
        acc_arr = np.array([p[1] for p in valid_pairs])
        auc_coverage = round(float(np.trapz(acc_arr, cov_arr)), 6)
    else:
        auc_coverage = 0.0

    # Optimal threshold for accuracy (maximum accuracy with coverage > 0)
    # and for macro-F1 (maximum macro-F1 with coverage > 0).
    def _optimal_threshold(
        values: List[Optional[float]],
        metric_name: str,
    ) -> Dict[str, Any]:
        """Find threshold index that maximises a metric (None values skipped)."""
        best_idx   = None
        best_val   = -1.0
        for i, (val, cov) in enumerate(zip(values, coverage_list)):
            if val is not None and val > best_val:
                best_val = val
                best_idx = i
        if best_idx is None:
            return {"threshold": None, metric_name: None, "coverage": None,
                    "n_accepted": None}
        return {
            "threshold":   float(threshold_list[best_idx]),
            metric_name:   float(values[best_idx]),
            "coverage":    float(coverage_list[best_idx]),
            "n_accepted":  int(n_accepted_list[best_idx]),
        }

    optimal_acc = _optimal_threshold(accuracy_list, "accuracy")
    optimal_f1  = _optimal_threshold(macro_f1_list, "macro_f1")

    result: Dict[str, Any] = {
        "thresholds":                   threshold_list,
        "coverage":                     coverage_list,
        "accuracy":                     accuracy_list,
        "macro_f1":                     macro_f1_list,
        "n_accepted":                   n_accepted_list,
        "mean_confidence":              mean_conf_list,
        "auc_coverage":                 auc_coverage,
        "optimal_threshold_accuracy":   optimal_acc,
        "optimal_threshold_f1":         optimal_f1,
        "n_threshold_points":           n_threshold_points,
        "n_samples":                    n_samples,
        "n_classes":                    n_classes,
        "caveat":                       _CALIBRATION_CAVEAT,
    }

    # Extract τ=0 and τ=0.5 stats for the log line.
    # τ=0 is always index 0 (coverage=1.0); τ=0.5 is the midpoint.
    mid_idx       = n_threshold_points // 2
    acc_at_0      = accuracy_list[0]
    acc_at_half   = accuracy_list[mid_idx]
    cov_at_half   = coverage_list[mid_idx]
    f1_at_0       = macro_f1_list[0]

    logger.info(
        f"compute_confidence_threshold_curve() | "
        f"n_samples={n_samples} | n_threshold_points={n_threshold_points} | "
        f"τ=0: coverage=1.00, acc={acc_at_0}, macro_f1={f1_at_0} | "
        f"τ=0.5: coverage={cov_at_half}, acc={acc_at_half} | "
        f"auc_coverage={auc_coverage:.4f} | "
        f"optimal_acc_threshold={optimal_acc['threshold']} "
        f"(acc={optimal_acc.get('accuracy')}, cov={optimal_acc.get('coverage')}) | "
        f"optimal_f1_threshold={optimal_f1['threshold']} "
        f"(f1={optimal_f1.get('macro_f1')}, cov={optimal_f1.get('coverage')})",
        extra={"stage": "evaluation"},
    )

    return result


def compute_calibration_summary(
    y_true: Any,
    y_proba: Any,
    n_classes: int,
    split_name: str = "val",
    n_bins: int = DEFAULT_N_BINS,
    n_threshold_points: int = DEFAULT_N_THRESHOLD_POINTS,
    seed: int = DEFAULT_SEED,
    compute_bin_ci: bool = True,
    ci_level: float = _BIN_CI_LEVEL,
) -> Dict[str, Any]:
    """
    Bundle reliability diagram and confidence-threshold curve into a single
    JSON-serialisable summary for ``evaluation_report.json``.

    This is the single function that ``pipelines/run_evaluation.py`` and
    ``notebooks/06_evaluation_error_analysis.ipynb`` call to get all
    calibration metrics for a split. Analogous to ``compute_evaluation_summary()``
    in ``metrics.py``.

    IMPORTANT — this function operates on arrays ONLY (no model inference).
    The caller is responsible for passing already-extracted ``(y_true, y_proba)``
    from the Phase B1 / Phase C prediction cache. This ensures the test set
    inference pass (Phase C) remains a controlled, logged, one-shot event
    rather than happening silently inside this function.

    All inputs are validated exactly once and the validated arrays are passed
    to both constituent functions — avoiding redundant re-validation (same
    pattern as the post-review fix in ``compute_evaluation_summary()`` in
    metrics.py, item 11).

    Parameters
    ----------
    y_true            : array-like, shape (n_samples,)
    y_proba           : array-like, shape (n_samples, n_classes), softmax probs
    n_classes         : int
    split_name        : str, default "val" — echoed into the result dict.
    n_bins            : int, default 10
    n_threshold_points: int, default 101
    seed              : int, default 42
    compute_bin_ci    : bool, default True
    ci_level          : float in (0, 1), default 0.80

    Returns
    -------
    dict with keys:
        split_name         : str
        n_samples          : int
        n_classes          : int
        reliability_diagram: dict — see compute_reliability_diagram()
        threshold_curve    : dict — see compute_confidence_threshold_curve()
        ece                : float — top-level convenience alias
        mce                : float — top-level convenience alias
        overconfidence_gap : float — top-level convenience alias
        auc_coverage       : float — top-level convenience alias
        caveat             : str
        temperature_scaling_note: str

    Raises
    ------
    ValueError
        Via constituent functions (arrays invalid, shapes wrong, etc.)
    """
    _validate_class_count(n_classes, "compute_calibration_summary")

    # Validate once; pass already-validated arrays to constituent functions
    # to avoid triple-validation overhead.
    y_true_arr  = _to_label_array(y_true, "y_true")
    y_proba_arr = _to_proba_array(y_proba, n_classes, "compute_calibration_summary")
    _validate_equal_length(y_true_arr, y_proba_arr, "y_true", "y_proba")
    _validate_label_range(y_true_arr, n_classes, "compute_calibration_summary")

    n_samples = len(y_true_arr)

    logger.info(
        f"compute_calibration_summary() | split='{split_name}' | "
        f"n_samples={n_samples} | n_classes={n_classes} | "
        f"n_bins={n_bins} | n_threshold_points={n_threshold_points}",
        extra={"stage": "evaluation"},
    )

    reliability = compute_reliability_diagram(
        y_true_arr, y_proba_arr, n_classes,
        n_bins=n_bins, seed=seed,
        compute_bin_ci=compute_bin_ci, ci_level=ci_level,
    )

    threshold_curve = compute_confidence_threshold_curve(
        y_true_arr, y_proba_arr, n_classes,
        n_threshold_points=n_threshold_points, seed=seed,
    )

    summary: Dict[str, Any] = {
        "split_name":             split_name,
        "n_samples":              n_samples,
        "n_classes":              n_classes,
        "reliability_diagram":    reliability,
        "threshold_curve":        threshold_curve,
        # Top-level convenience aliases for evaluation_report.json
        "ece":                    reliability["ece"],
        "ece_unweighted":         reliability["ece_unweighted"],
        "mce":                    reliability["mce"],
        "overconfidence_gap":     reliability["overconfidence_gap"],
        "mean_confidence":        reliability["mean_confidence"],
        "mean_accuracy":          reliability["mean_accuracy"],
        "auc_coverage":           threshold_curve["auc_coverage"],
        "caveat":                 _CALIBRATION_CAVEAT,
        "temperature_scaling_note": TEMPERATURE_SCALING_NOTE,
    }

    logger.info(
        f"compute_calibration_summary() COMPLETE | split='{split_name}' | "
        f"ECE={reliability['ece']:.4f} | MCE={reliability['mce']:.4f} | "
        f"overconfidence_gap={reliability['overconfidence_gap']:+.4f} | "
        f"auc_coverage={threshold_curve['auc_coverage']:.4f}",
        extra={"stage": "evaluation"},
    )

    return summary


# ---------------------------------------------------------------------------
# Figure rendering helpers
# ---------------------------------------------------------------------------

def plot_reliability_diagram(
    reliability_result: Dict[str, Any],
    output_path: Optional[Union[str, Path]] = None,
    split_name: str = "val",
    figure_dpi: int = 150,
    show_sparse_annotation: bool = True,
    show_bin_counts: bool = True,
    show_ci: bool = True,
) -> Any:
    """
    Render a reliability diagram (calibration curve) from the dict returned
    by ``compute_reliability_diagram()``.

    Visual design
    -------------
    - Blue bars: actual accuracy per bin (the "observed" calibration).
    - Red dashed diagonal: perfect calibration (confidence == accuracy).
    - Light red shaded region: the overconfidence zone (bars below diagonal).
    - Light blue shaded region: the underconfidence zone (bars above diagonal).
    - Error bars: 80% bootstrap CI on per-bin accuracy (when available).
    - Empty bins: shown as a very light grey hatched bar (height=0 marker).
    - Sparse bins: annotated with an asterisk (*) above the bar.
    - Bin count labels: printed above each bar.
    - ECE, MCE, overconfidence gap: annotated in the plot legend/title.

    Parameters
    ----------
    reliability_result  : dict from compute_reliability_diagram()
    output_path         : str | Path | None — if provided, save the figure here.
                          If None, return the figure object for notebook display.
    split_name          : str — shown in the figure title.
    figure_dpi          : int — DPI for saved figure.
    show_sparse_annotation: bool — annotate sparse bins with '*'.
    show_bin_counts     : bool — print n=X above each bar.
    show_ci             : bool — draw error bars from bootstrap CI.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object. Caller is responsible for plt.close(fig) if not
        embedding in a notebook.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError as exc:
        raise ImportError(
            "plot_reliability_diagram() requires matplotlib. "
            "Install with: pip install matplotlib"
        ) from exc

    bins      = reliability_result["bins"]
    n_bins    = reliability_result["n_bins"]
    ece       = reliability_result["ece"]
    mce       = reliability_result["mce"]
    ovgap     = reliability_result["overconfidence_gap"]
    n_samples = reliability_result["n_samples"]
    ci_level  = reliability_result.get("ci_level")

    fig, (ax_main, ax_hist) = plt.subplots(
        2, 1, figsize=(9, 9),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.35},
    )

    # ── Main panel: reliability diagram ───────────────────────────────────
    ax = ax_main

    bin_width = 1.0 / n_bins
    bin_centers = [
        bd["bin_lo"] + bin_width / 2.0
        for bd in bins
    ]

    # Perfect calibration diagonal
    ax.plot([0, 1], [0, 1], "r--", linewidth=1.5, label="Perfect calibration", zorder=5)

    # Shading for over/under confidence regions
    ax.fill_between([0, 1], [0, 0], [0, 1],
                    alpha=0.06, color="cornflowerblue",
                    label="Underconfidence zone")
    ax.fill_between([0, 1], [0, 1], [1, 1],
                    alpha=0.06, color="tomato",
                    label="Overconfidence zone")

    for bd, center in zip(bins, bin_centers):
        if bd["is_empty"]:
            # Empty bin: draw a very faint hatched marker at height 0
            ax.bar(center, 0.005, width=bin_width * 0.85,
                   color="lightgrey", edgecolor="grey", linewidth=0.5,
                   hatch="////", alpha=0.5, zorder=2)
            continue

        acc   = bd["actual_accuracy"]
        is_sp = bd["is_sparse"]
        count = bd["bin_count"]

        # Bar colour: light steel blue for normal, muted orange for sparse
        color = "steelblue" if not is_sp else "sandybrown"
        ax.bar(center, acc, width=bin_width * 0.85,
               color=color, edgecolor="white", linewidth=0.8,
               alpha=0.85, zorder=3)

        # CI error bars
        if show_ci and bd.get("ci_lower") is not None and bd.get("ci_upper") is not None:
            ci_lo = bd["ci_lower"]
            ci_hi = bd["ci_upper"]
            ax.errorbar(
                center, acc,
                yerr=[[acc - ci_lo], [ci_hi - acc]],
                fmt="none", color="navy", linewidth=1.5,
                capsize=4, capthick=1.5, zorder=4,
            )

        # Count annotation
        if show_bin_counts:
            label_parts = [f"n={count}"]
            if show_sparse_annotation and is_sp:
                label_parts.append("*")
            ax.text(
                center, min(acc + 0.04, 0.97),
                " ".join(label_parts),
                ha="center", va="bottom",
                fontsize=7.5,
                color="dimgrey" if not is_sp else "darkorange",
            )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Mean confidence (max softmax probability)", fontsize=11)
    ax.set_ylabel("Actual accuracy", fontsize=11)

    ci_level_str = f" ({int(ci_level * 100)}% CI)" if ci_level and show_ci else ""
    title_lines = [
        f"Reliability Diagram — {split_name} split  (n={n_samples})",
        f"ECE={ece:.4f}  |  MCE={mce:.4f}  |  "
        f"Overconfidence gap={ovgap:+.4f}{ci_level_str}",
    ]
    ax.set_title("\n".join(title_lines), fontsize=11, pad=10)

    # Custom legend entries
    legend_handles = [
        plt.Line2D([0], [0], color="red", linestyle="--", label="Perfect calibration"),
        mpatches.Patch(color="steelblue", alpha=0.85, label="Calibrated bin"),
        mpatches.Patch(color="sandybrown", alpha=0.85,
                       label=f"Sparse bin (n < {SPARSE_BIN_THRESHOLD})*"),
        mpatches.Patch(color="lightgrey", hatch="////", alpha=0.5, label="Empty bin"),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.25, linestyle=":")

    # ── Lower panel: confidence histogram ─────────────────────────────────
    # Shows where the model's confidence mass actually lies — essential
    # context for interpreting sparse/empty bins in the main panel.
    # We can reconstruct the histogram from bin counts.
    bin_edges   = np.array([bd["bin_lo"] for bd in bins] + [bins[-1]["bin_hi"]])
    bin_heights = np.array([bd["bin_count"] for bd in bins])

    ax_hist.bar(
        bin_edges[:-1], bin_heights,
        width=np.diff(bin_edges), align="edge",
        color="steelblue", alpha=0.6, edgecolor="white", linewidth=0.5,
    )
    ax_hist.axhline(SPARSE_BIN_THRESHOLD, color="darkorange", linestyle=":",
                    linewidth=1.2, label=f"Sparse threshold (n={SPARSE_BIN_THRESHOLD})")
    ax_hist.set_xlabel("Confidence", fontsize=10)
    ax_hist.set_ylabel("Count", fontsize=10)
    ax_hist.set_title("Confidence distribution across bins", fontsize=10)
    ax_hist.set_xlim(0.0, 1.0)
    ax_hist.legend(fontsize=8, loc="upper left")
    ax_hist.grid(True, alpha=0.25, linestyle=":")

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=figure_dpi, bbox_inches="tight")
        logger.info(
            f"Reliability diagram saved → {out.resolve()}",
            extra={"stage": "evaluation"},
        )

    return fig


def plot_confidence_threshold_curve(
    threshold_result: Dict[str, Any],
    output_path: Optional[Union[str, Path]] = None,
    split_name: str = "val",
    figure_dpi: int = 150,
) -> Any:
    """
    Render the confidence-threshold curve from the dict returned by
    ``compute_confidence_threshold_curve()``.

    Visual design
    -------------
    Two y-axes on a single panel:
      - Left y-axis (blue): Coverage (fraction of predictions accepted).
      - Right y-axis (green): Accuracy on accepted predictions.
    A secondary thin orange line shows macro-F1 on the accepted subset.
    Vertical dashed lines mark optimal_threshold_accuracy and
    optimal_threshold_f1.

    Parameters
    ----------
    threshold_result : dict from compute_confidence_threshold_curve()
    output_path      : str | Path | None
    split_name       : str
    figure_dpi       : int

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "plot_confidence_threshold_curve() requires matplotlib."
        ) from exc

    thresholds   = threshold_result["thresholds"]
    coverage     = threshold_result["coverage"]
    accuracy     = threshold_result["accuracy"]
    macro_f1     = threshold_result["macro_f1"]
    n_samples    = threshold_result["n_samples"]
    auc_cov      = threshold_result["auc_coverage"]
    opt_acc      = threshold_result["optimal_threshold_accuracy"]
    opt_f1       = threshold_result["optimal_threshold_f1"]

    # Replace None with np.nan for plotting (matplotlib handles NaN as gaps)
    acc_plot = [a if a is not None else float("nan") for a in accuracy]
    f1_plot  = [f if f is not None else float("nan") for f in macro_f1]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    # Coverage line (left axis)
    ax1.plot(thresholds, coverage, color="cornflowerblue", linewidth=2.2,
             label="Coverage", zorder=4)
    ax1.set_xlabel("Confidence threshold τ", fontsize=11)
    ax1.set_ylabel("Coverage (fraction accepted)", color="cornflowerblue", fontsize=11)
    ax1.tick_params(axis="y", labelcolor="cornflowerblue")
    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(-0.02, 1.05)

    # Accuracy line (right axis)
    ax2.plot(thresholds, acc_plot, color="mediumseagreen", linewidth=2.2,
             label="Accuracy on accepted", zorder=4)
    ax2.plot(thresholds, f1_plot, color="darkorange", linewidth=1.5,
             linestyle="--", label="Macro-F1 on accepted", zorder=4)
    ax2.set_ylabel("Accuracy / Macro-F1", color="mediumseagreen", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="mediumseagreen")
    ax2.set_ylim(-0.02, 1.05)

    # Vertical markers for optimal thresholds
    if opt_acc.get("threshold") is not None:
        ax1.axvline(opt_acc["threshold"], color="mediumseagreen",
                    linestyle=":", linewidth=1.5,
                    label=f"Opt acc τ={opt_acc['threshold']:.2f} "
                          f"(acc={opt_acc.get('accuracy', '?'):.3f}, "
                          f"cov={opt_acc.get('coverage', '?'):.2f})",
                    zorder=3)
    if opt_f1.get("threshold") is not None and opt_f1["threshold"] != opt_acc.get("threshold"):
        ax1.axvline(opt_f1["threshold"], color="darkorange",
                    linestyle=":", linewidth=1.5,
                    label=f"Opt F1 τ={opt_f1['threshold']:.2f} "
                          f"(F1={opt_f1.get('macro_f1', '?'):.3f}, "
                          f"cov={opt_f1.get('coverage', '?'):.2f})",
                    zorder=3)

    # Horizontal reference line at coverage = 1.0 (τ=0)
    ax1.axhline(1.0, color="lightgrey", linestyle=":", linewidth=0.8, zorder=1)

    title = (
        f"Confidence-Threshold Curve — {split_name} split  (n={n_samples})\n"
        f"AUC-coverage={auc_cov:.4f}"
    )
    ax1.set_title(title, fontsize=11, pad=10)
    ax1.grid(True, alpha=0.2, linestyle=":")

    # Combined legend (both axes)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="center left")

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=figure_dpi, bbox_inches="tight")
        logger.info(
            f"Confidence-threshold curve saved → {out.resolve()}",
            extra={"stage": "evaluation"},
        )

    return fig


# ---------------------------------------------------------------------------
# Import-time self-check
# ---------------------------------------------------------------------------

def _self_check() -> None:
    """
    Cheap, dependency-free sanity check on module constants.

    Mirrors the pattern in metrics.py and benchmark.py. Validates internal
    constant consistency without asserting any project-specific class count
    (consistent with the post-review metrics.py revision, item 12).
    """
    assert DEFAULT_N_BINS >= 2, (
        f"calibration.py: DEFAULT_N_BINS={DEFAULT_N_BINS} must be >= 2."
    )
    assert DEFAULT_N_THRESHOLD_POINTS >= 2, (
        f"calibration.py: DEFAULT_N_THRESHOLD_POINTS={DEFAULT_N_THRESHOLD_POINTS} "
        "must be >= 2."
    )
    assert SPARSE_BIN_THRESHOLD >= 1, (
        f"calibration.py: SPARSE_BIN_THRESHOLD={SPARSE_BIN_THRESHOLD} must be >= 1."
    )
    assert 0.0 < _BIN_CI_LEVEL < 1.0, (
        f"calibration.py: _BIN_CI_LEVEL={_BIN_CI_LEVEL} must be in (0, 1)."
    )
    assert _N_BINS_CI_BOOTSTRAP >= 10, (
        f"calibration.py: _N_BINS_CI_BOOTSTRAP={_N_BINS_CI_BOOTSTRAP} must be >= 10."
    )
    assert 0.0 < _MIN_MEANINGFUL_COVERAGE <= 1.0, (
        f"calibration.py: _MIN_MEANINGFUL_COVERAGE={_MIN_MEANINGFUL_COVERAGE} "
        "must be in (0, 1]."
    )
    assert len(_CALIBRATION_CAVEAT) > 50, (
        "calibration.py: _CALIBRATION_CAVEAT string is unexpectedly short — "
        "check it was not accidentally truncated."
    )
    assert len(TEMPERATURE_SCALING_NOTE) > 50, (
        "calibration.py: TEMPERATURE_SCALING_NOTE string is unexpectedly short."
    )


if __debug__:
    _self_check()


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------

__all__ = [
    "DEFAULT_N_BINS",
    "DEFAULT_N_THRESHOLD_POINTS",
    "DEFAULT_SEED",
    "SPARSE_BIN_THRESHOLD",
    "TEMPERATURE_SCALING_NOTE",
    "compute_reliability_diagram",
    "compute_confidence_threshold_curve",
    "compute_calibration_summary",
    "plot_reliability_diagram",
    "plot_confidence_threshold_curve",
]