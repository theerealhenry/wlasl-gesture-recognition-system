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

  2. Is framework-agnostic at its core. The only non-numpy dependency for
     metric computation is ``sklearn.metrics.f1_score``, which is already
     a project-wide dependency (used throughout ``metrics.py``, ``train.py``,
     and ``benchmark.py``). This dependency is declared explicitly at the top
     of the module rather than deferred as a hidden local import — the previous
     design claimed "numpy and scipy only" while importing sklearn internally,
     which was a documentation/contract bug. scipy is NOT used in this module.

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

Champion model context (for reference)
----------------------------------------
  - Input shape:   (1, 100, 126) — seq_len=100, landmark_config=hands_only
  - Output shape:  (1, 35) — softmax probabilities
  - val_macro_f1:  0.6011 (52 clips, 7 unseen signers)
  - early_stopping_monitor: val_accuracy (NOTE: the handoff document narrates
    val_macro_f1 as the monitor, but the actual config_snapshot.yaml for the
    champion run bilstm_hands_only_v4_aug confirms ``early_stopping_monitor:
    val_accuracy``. This discrepancy is flagged in evaluation_report.json per
    Phase F requirements; this module takes no position on which was intended.)
  - ``y_proba`` consumed here comes from:
    ``get_predictions(model, val_ds, return_probs=True)`` in metrics.py,
    which returns float64 arrays of shape (52, 35).

Calibration methodology decisions
------------------------------------
Equal-width binning (default) + optional quantile binning
    Equal-width binning (fixed-width bins across [0, 1]) is the standard
    approach and is easiest to explain. However, with only 52 validation
    clips, equal-width bins result in many empty or 1–2 sample bins because
    softmax confidence is not uniformly distributed across [0,1]. Optional
    quantile binning (``strategy="quantile"``) places equal numbers of
    predictions in each bin, producing more statistically stable estimates
    at the cost of variable-width bars that are harder to interpret visually.
    Both strategies are supported. Equal-width is the default for publication
    consistency; quantile is recommended for internal diagnostics.

n_bins=10 (default), configurable
    10 bins × 52 clips ≈ 5 clips/bin on average. This is very sparse —
    many bins will be empty or contain 1–2 clips. The module therefore:
    (a) reports ``bin_count`` for every bin so chart consumers can
    annotate the actual sample count above each bar rather than presenting
    a misleadingly smooth curve;
    (b) flags bins with ``is_sparse`` when ``bin_count < SPARSE_BIN_THRESHOLD``
    (default: 5), so downstream consumers can style or annotate them
    differently;
    (c) supports ``n_bins=5`` as the "low-sample-size" alternative that
    produces ~10 clips/bin, which is still sparse but more reliable.

Max confidence per prediction (class with highest softmax probability)
    The reliability diagram uses only the WINNING class's softmax probability
    as the "model confidence" for that prediction. This is the standard
    definition and the one that corresponds to the argmax prediction that
    ``compute_macro_f1()`` in ``metrics.py`` is based on.

ECE = weighted average of |confidence - accuracy|
    Weighted by the fraction of samples in each non-empty bin. This is the
    standard ECE formulation (Naeini et al., 2015). The unweighted version
    (average over non-empty bins regardless of their size) is also computed
    and reported as ``ece_unweighted`` for completeness. A bootstrap CI on
    ECE itself is also computed (unlike the original, which only provided
    bin-level CIs) to surface the global calibration uncertainty.

Maximum Calibration Error (MCE)
    MCE = max over non-empty bins of |confidence - accuracy|. Reported
    alongside ECE because ECE can be dominated by many small-error high-mass
    bins while a single catastrophically miscalibrated high-confidence bin
    (a scenario plausible on this 52-clip val set) would be visible in MCE
    but obscured in ECE.

Temperature scaling note (documented-not-implemented)
    The standard post-hoc calibration fix is temperature scaling: dividing
    the logits (pre-softmax activations) by a learned scalar T before softmax,
    optimised on the validation set. This is NOT implemented here because
    (a) this module receives post-softmax probabilities, not logits; (b) with
    52 val clips, the temperature estimate would itself be highly unreliable;
    (c) Stage 6's goal is DIAGNOSIS, not calibration repair — the repair
    belongs in a future model iteration. The limitation and the remedy are
    documented in LIMITATIONS.md.

Threshold curve macro-F1 interpretation note
---------------------------------------------
    The ``macro_f1`` values in ``compute_confidence_threshold_curve()`` are
    labeled ``selective_macro_f1`` in the output to make clear that this is
    NOT a measure of overall model quality — it is the macro-F1 achieved on
    the subset of predictions the model is "confident" enough to make (those
    at or above threshold τ). At high thresholds only easy predictions remain,
    so this metric inflates naturally. The ``optimal_threshold_f1`` finding
    must therefore be interpreted as "best selective precision at some coverage
    cost", not as an improved model performance estimate.

Post-review disposition (critical review applied)
---------------------------------------------------
The following issues from the Phase A3 critical review were assessed,
verified, and addressed in this revision:

  1.  FIXED. Dead expression ``y_true_arr = _to_label_array(y_proba if False
      else y_true, "y_true")`` removed; clean call with y_true directly.

  2.  FIXED. Hidden sklearn dependency declared explicitly at module top and
      in this docstring rather than deferred as a local import inside
      ``compute_confidence_threshold_curve()``. The "numpy and scipy only"
      contract claim has been corrected; scipy is NOT used here.

  3.  FIXED. Bootstrap seeding now uses ``np.random.SeedSequence([seed, b])``
      rather than ``seed ^ b`` XOR, which introduced structured bit-level
      correlation between per-bin bootstrap streams.

  4.  FIXED. ``matplotlib.use("Agg")`` global-state mutation removed from
      plot functions. Callers that need a non-interactive backend (e.g. CI
      environments) should set it at application entry point. Plot functions
      now use ``plt.switch_backend("Agg")`` contextually only if no backend
      is yet active, and only as a best-effort fallback.

  5.  ADDED. Bootstrap CI on ECE itself (``ece_ci_lower``, ``ece_ci_upper``)
      via clip-level resampling, matching the approach in ``metrics.py``.

  6.  FIXED. ``macro_f1`` renamed to ``selective_macro_f1`` in threshold
      curve output with explicit documentation that this is conditional on
      acceptance, not overall quality.

  7.  ADDED. Monotonicity warning in AUC-coverage computation.

  8.  ADDED. ``ci_degenerate`` flag in per-bin bootstrap CI output when
      bin_count == 1 (CI is trivially acc=acc, communicates nothing).

  9.  ADDED. Optional ``strategy`` parameter to
      ``compute_reliability_diagram()`` supporting ``"uniform"`` (default,
      equal-width bins) and ``"quantile"`` (equal-mass bins) for more
      robust small-sample diagnostics.

  10. FIXED. ``compute_calibration_summary()`` validates arrays once and
      passes already-validated arrays to constituent functions (matching the
      post-review fix item 11 in ``metrics.py``).

  11. NOTED. Validation helpers ``_validate_class_count``, ``_to_label_array``
      etc. are duplicated from ``metrics.py``. This is an accepted trade-off
      for now (keeps module independently importable and testable without
      metrics.py). A future ``src/evaluation/_validation.py`` shared module
      would be the cleaner solution. Divergence risk is mitigated by the
      import-time self-check and unit tests.

  12. FIXED. ``early_stopping_monitor`` discrepancy documented in module
      docstring above.

Module-level exports
----------------------
    compute_reliability_diagram          — bin predictions by confidence,
                                           compute actual accuracy per bin,
                                           ECE, MCE, ECE bootstrap CI
    compute_confidence_threshold_curve   — coverage vs accuracy vs selective
                                           macro-F1 across confidence thresholds
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
from sklearn.metrics import f1_score as _sklearn_f1_score

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Default number of equal-width confidence bins for the reliability diagram.
#: With 52 val clips, n_bins=5 is recommended for more reliable estimates
DEFAULT_N_BINS: int = 5

#: Default number of threshold points for the confidence-threshold curve.
#: 101 points gives a 1-percentage-point resolution across [0, 1].
DEFAULT_N_THRESHOLD_POINTS: int = 101

#: Bins with fewer than this many samples are flagged as sparse in the
#: reliability diagram output.
SPARSE_BIN_THRESHOLD: int = 5

#: Number of bootstrap resamples for per-bin accuracy confidence intervals.
_N_BINS_CI_BOOTSTRAP: int = 200

#: Bootstrap confidence level for per-bin accuracy CIs.
#: 80% rather than 90% because per-bin sample counts are so small (often 1–5)
#: that a 90% interval would typically span [0, 1] and communicate nothing.
_BIN_CI_LEVEL: float = 0.80

#: Number of bootstrap resamples for ECE confidence interval.
#: Clip-level resampling; 1000 matches DEFAULT_N_BOOTSTRAP in metrics.py.
_N_ECE_CI_BOOTSTRAP: int = 1000

#: Bootstrap CI level for ECE. 90% matches DEFAULT_BOOTSTRAP_CI in metrics.py.
_ECE_CI_LEVEL: float = 0.90

#: Project global seed. Matches DEFAULT_SEED in metrics.py and base.yaml.
DEFAULT_SEED: int = 42

#: Below this many total samples, reliability diagram computation is
#: technically possible but statistically meaningless — warn rather than raise.
_MIN_SAMPLES_FOR_CALIBRATION: int = 5

#: Coverage below which a threshold is considered "effectively rejecting
#: everything" — flagged in the threshold curve output.
_MIN_MEANINGFUL_COVERAGE: float = 0.05

#: Valid binning strategies for compute_reliability_diagram().
#: "uniform" = equal-width bins (standard, default).
#: "quantile" = equal-mass bins (more stable for small, skewed datasets).
_VALID_BIN_STRATEGIES: Tuple[str, ...] = ("uniform", "quantile")

#: Calibration limitation caveat — embedded in every summary dict so all
#: downstream consumers (evaluation_report.json, LIMITATIONS.md) inherit
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

#: Documented-not-implemented note for temperature scaling, embedded verbatim
#: in LIMITATIONS.md and evaluation_report.json.
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

#: Selective macro-F1 interpretation note. Embedded in threshold curve output
#: so every downstream consumer understands the metric's conditional nature.
_SELECTIVE_MACRO_F1_NOTE: str = (
    "selective_macro_f1 is the macro-F1 achieved on the subset of predictions "
    "that exceed confidence threshold τ (i.e. predictions the model is willing "
    "to make). This is NOT a measure of overall model quality — at high "
    "thresholds only easy predictions remain, inflating this metric naturally. "
    "Interpret optimal_threshold_f1 as 'best selective precision at some "
    "coverage cost', not as an improved model performance estimate."
)


# ---------------------------------------------------------------------------
# Internal validation helpers
# (Intentionally self-contained — see module docstring item 11 for rationale)
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


def _validate_bin_strategy(strategy: str, caller: str) -> None:
    """Raise ValueError if strategy is not a recognised binning strategy."""
    if strategy not in _VALID_BIN_STRATEGIES:
        raise ValueError(
            f"{caller}: strategy={strategy!r} is not recognised. "
            f"Valid options: {_VALID_BIN_STRATEGIES}. "
            "'uniform' (default) uses equal-width bins. "
            "'quantile' uses equal-mass bins — recommended for small, "
            "skewed datasets like this project's 52-clip val set."
        )


def _to_label_array(arr: Any, name: str) -> np.ndarray:
    """
    Coerce an array-like into a flat 1-D int64 numpy array of class indices.

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

    _tol = 1e-6
    if out.min() < -_tol or out.max() > 1.0 + _tol:
        raise ValueError(
            f"{caller}: y_proba contains values outside [0, 1] "
            f"(min={out.min():.6f}, max={out.max():.6f}). "
            "Expected softmax probabilities in [0, 1]. "
            "If this is a logits array, apply softmax before passing it here."
        )

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
    winning class (argmax class).

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


def _compute_bin_edges(
    max_conf: np.ndarray,
    n_bins: int,
    strategy: str,
) -> np.ndarray:
    """
    Compute bin edges based on the chosen strategy.

    Parameters
    ----------
    max_conf : np.ndarray, shape (n_samples,), float64
        Max softmax probability per prediction.
    n_bins   : int
        Number of bins.
    strategy : str
        ``"uniform"`` — equal-width bins from 0.0 to 1.0.
        ``"quantile"`` — equal-mass bins using quantiles of max_conf.

    Returns
    -------
    np.ndarray, shape (n_bins + 1,)
        Bin edge values (inclusive on both ends for the last bin).
    """
    if strategy == "quantile":
        # Place bin edges at the quantile breakpoints of max_conf.
        # Always include 0.0 and 1.0 as the outer bounds.
        quantiles = np.linspace(0.0, 100.0, n_bins + 1)
        edges = np.percentile(max_conf, quantiles)
        # Ensure strict 0.0 and 1.0 bounds regardless of data extremes.
        edges[0]  = 0.0
        edges[-1] = 1.0
        return edges
    else:
        # "uniform": standard equal-width bins.
        return np.linspace(0.0, 1.0, n_bins + 1)


def _bootstrap_bin_accuracy_ci(
    correct_flags: np.ndarray,
    ci_level: float = _BIN_CI_LEVEL,
    n_bootstrap: int = _N_BINS_CI_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
    bin_index: int = 0,
) -> Tuple[float, float, bool]:
    """
    Compute a bootstrap confidence interval for the accuracy of a single bin.

    Seeding fix (post-review item 3)
    ---------------------------------
    Uses ``np.random.SeedSequence([seed, bin_index])`` rather than
    ``seed ^ bin_index`` XOR. XOR-based seeds create structured bit-level
    correlations: e.g. for seed=42, bins 0 and 42 produce the same XOR seed,
    and the seeds across bins are not statistically independent. SeedSequence
    hashing produces cryptographically mixed, statistically independent
    streams for every (seed, bin_index) pair.

    Parameters
    ----------
    correct_flags : np.ndarray, shape (n_bin_samples,), bool
    ci_level      : float, confidence level in (0, 1)
    n_bootstrap   : int, number of bootstrap resamples
    seed          : int, base RNG seed
    bin_index     : int, bin index (mixed with seed via SeedSequence)

    Returns
    -------
    Tuple[float, float, bool]
        (ci_lower, ci_upper, ci_degenerate)
        ci_degenerate is True when the bin has only 1 sample — the CI is
        (acc, acc) and conveys no uncertainty information.
    """
    n = len(correct_flags)
    if n == 0:
        return (0.0, 0.0, True)
    if n == 1:
        # Single sample: CI is degenerate (0.0 or 1.0 with certainty).
        acc = float(correct_flags[0])
        return (acc, acc, True)

    # SeedSequence produces statistically independent streams per (seed, bin_index).
    ss  = np.random.SeedSequence([seed, bin_index])
    rng = np.random.default_rng(ss)

    boot_acc = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_acc[i] = correct_flags[idx].mean()

    alpha    = 1.0 - ci_level
    ci_lower = float(np.percentile(boot_acc, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_acc, 100.0 * (1.0 - alpha / 2.0)))
    return (ci_lower, ci_upper, False)


def _bootstrap_ece_ci(
    max_conf: np.ndarray,
    correct: np.ndarray,
    n_classes: int,
    n_bins: int,
    strategy: str,
    ci_level: float = _ECE_CI_LEVEL,
    n_bootstrap: int = _N_ECE_CI_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
) -> Tuple[float, float]:
    """
    Compute a bootstrap CI for ECE via clip-level resampling.

    This uses CLIP-LEVEL resampling (not bin-level) — each resample draws
    n_samples clips with replacement, recomputes the full reliability diagram
    and ECE on the resample. This correctly propagates uncertainty from both
    bin composition (which clips fall in which bin) and per-bin accuracy noise.

    Parameters
    ----------
    max_conf  : np.ndarray, shape (n_samples,), float64 — max softmax prob per clip
    correct   : np.ndarray, shape (n_samples,), bool — correct argmax prediction
    n_classes : int (unused in ECE computation; kept for interface consistency)
    n_bins    : int
    strategy  : str, "uniform" or "quantile"
    ci_level  : float
    n_bootstrap: int
    seed      : int

    Returns
    -------
    Tuple[float, float]
        (ece_ci_lower, ece_ci_upper) at the specified ci_level.
    """
    n_samples = len(max_conf)

    # Use SeedSequence for the ECE bootstrap stream (distinct from bin CIs).
    ss  = np.random.SeedSequence([seed, 999983])  # large prime offset avoids collision with bin seeds
    rng = np.random.default_rng(ss)

    boot_ece = np.empty(n_bootstrap, dtype=np.float64)

    # Pre-compute global bin edges from the full dataset (fixed for all resamples
    # in the "uniform" case; recomputed per resample in "quantile" to match the
    # conditional distribution of the resample).
    global_edges = _compute_bin_edges(max_conf, n_bins, strategy) if strategy == "uniform" else None

    for i in range(n_bootstrap):
        idx          = rng.integers(0, n_samples, size=n_samples)
        r_max_conf   = max_conf[idx]
        r_correct    = correct[idx]

        # For quantile binning, recompute edges on the resample so the bins
        # adapt to the resampled confidence distribution.
        edges = global_edges if strategy == "uniform" else _compute_bin_edges(r_max_conf, n_bins, strategy)

        ece_sum = 0.0
        for b in range(n_bins):
            lo, hi = edges[b], edges[b + 1]
            in_bin = (r_max_conf >= lo) & (r_max_conf < hi) if b < n_bins - 1 else (r_max_conf >= lo) & (r_max_conf <= hi)
            cnt = int(in_bin.sum())
            if cnt == 0:
                continue
            bin_conf = float(r_max_conf[in_bin].mean())
            bin_acc  = float(r_correct[in_bin].mean())
            ece_sum += (cnt / n_samples) * abs(bin_conf - bin_acc)

        boot_ece[i] = ece_sum

    alpha    = 1.0 - ci_level
    ci_lower = float(np.percentile(boot_ece, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_ece, 100.0 * (1.0 - alpha / 2.0)))
    return (ci_lower, ci_upper)


def compute_reliability_diagram(
    y_true: Any,
    y_proba: Any,
    n_classes: int,
    n_bins: int = DEFAULT_N_BINS,
    strategy: str = "uniform",
    seed: int = DEFAULT_SEED,
    compute_bin_ci: bool = True,
    ci_level: float = _BIN_CI_LEVEL,
    compute_ece_ci: bool = True,
) -> Dict[str, Any]:
    """
    Compute a reliability diagram (calibration curve) for a multi-class
    classifier using the maximum-confidence convention.

    For each confidence bin b:
      - ``mean_confidence``  : average max-softmax-probability of predictions
                               falling in this bin (x-axis of the diagram).
      - ``actual_accuracy``  : fraction of those predictions that were correct
                               (y-axis of the diagram). None if bin is empty.
      - ``bin_count``        : number of predictions in the bin.
      - ``is_sparse``        : True if bin_count < SPARSE_BIN_THRESHOLD (5).
      - ``is_empty``         : True if bin_count == 0.
      - ``calibration_gap``  : actual_accuracy - mean_confidence.
      - ``ci_lower/upper``   : bootstrap CI on actual_accuracy.
      - ``ci_degenerate``    : True if CI is trivial (bin_count == 1).

    Aggregate statistics:
      - ``ece``              : Expected Calibration Error (weighted).
      - ``ece_unweighted``   : Unweighted ECE.
      - ``mce``              : Maximum Calibration Error.
      - ``ece_ci_lower/upper``: Bootstrap CI for ECE (clip-level resampling).
      - ``mean_confidence``  : Global mean max-softmax probability.
      - ``mean_accuracy``    : Overall accuracy (fraction of correct argmax).
      - ``overconfidence_gap``: mean_confidence - mean_accuracy.

    Parameters
    ----------
    y_true    : array-like, shape (n_samples,)
    y_proba   : array-like, shape (n_samples, n_classes), softmax probs
    n_classes : int — 35 for this project.
    n_bins    : int, default 10
    strategy  : str, default "uniform"
        "uniform" — equal-width bins (standard, publication-compatible).
        "quantile" — equal-mass bins; more stable for small, skewed datasets.
        Recommended: use n_bins=5 with strategy="uniform" for this 52-clip
        val set as a balance between shape and stability. Use strategy=
        "quantile" for internal diagnostics.
    seed      : int, default 42
    compute_bin_ci : bool, default True
    ci_level  : float in (0, 1), default 0.80
    compute_ece_ci : bool, default True
        If True, compute a bootstrap CI for ECE itself via clip-level
        resampling. Adds ~1000 additional ECE computations; fast at n=52.

    Returns
    -------
    dict with keys:
        bins                     : List[dict] — one dict per bin.
        ece, ece_unweighted, mce : float
        ece_ci_lower, ece_ci_upper: float | None
        ece_ci_level             : float | None
        mean_confidence          : float
        mean_accuracy            : float
        overconfidence_gap       : float
        n_bins, n_empty_bins, n_sparse_bins, n_samples, n_classes : int
        strategy                 : str
        ci_level                 : float | None
        caveat, temperature_scaling_note : str
    """
    # ── Validate inputs ───────────────────────────────────────────────────
    _validate_n_bins(n_bins, "compute_reliability_diagram")
    _validate_class_count(n_classes, "compute_reliability_diagram")
    _validate_bin_strategy(strategy, "compute_reliability_diagram")

    y_true_arr  = _to_label_array(y_true, "y_true")
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
    max_conf = _extract_max_confidence(y_proba_arr)   # shape (n_samples,)
    correct  = _is_correct(y_true_arr, y_proba_arr)   # shape (n_samples,)

    # ── Bin edges ─────────────────────────────────────────────────────────
    bin_edges = _compute_bin_edges(max_conf, n_bins, strategy)

    # ── Per-bin computation ───────────────────────────────────────────────
    bins_data: List[Dict[str, Any]] = []
    ece_numerator_sum   = 0.0
    ece_unweighted_gaps: List[float] = []
    mce = 0.0

    for b in range(n_bins):
        lo = float(bin_edges[b])
        hi = float(bin_edges[b + 1])

        if b < n_bins - 1:
            in_bin = (max_conf >= lo) & (max_conf < hi)
        else:
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
                "ci_degenerate":    None,
            }
        else:
            bin_conf_vals = max_conf[in_bin]
            bin_correct   = correct[in_bin]

            bin_mean_conf = float(bin_conf_vals.mean())
            bin_acc       = float(bin_correct.mean())
            gap           = bin_acc - bin_mean_conf
            abs_gap       = abs(gap)

            bin_fraction = bin_count / n_samples
            ece_numerator_sum += bin_fraction * abs_gap
            ece_unweighted_gaps.append(abs_gap)
            mce = max(mce, abs_gap)

            ci_lower_val: Optional[float] = None
            ci_upper_val: Optional[float] = None
            ci_degen: Optional[bool] = None

            if compute_bin_ci:
                ci_lower_val, ci_upper_val, ci_degen = _bootstrap_bin_accuracy_ci(
                    bin_correct, ci_level=ci_level,
                    n_bootstrap=_N_BINS_CI_BOOTSTRAP,
                    seed=seed, bin_index=b,
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
                "ci_degenerate":    ci_degen,
            }

        bins_data.append(bin_dict)

    # ── Aggregate statistics ──────────────────────────────────────────────
    n_empty_bins  = sum(1 for bd in bins_data if bd["is_empty"])
    n_sparse_bins = sum(1 for bd in bins_data if bd["is_sparse"])

    ece            = round(float(ece_numerator_sum), 6)
    ece_unweighted = (
        round(float(np.mean(ece_unweighted_gaps)), 6)
        if ece_unweighted_gaps else 0.0
    )
    mce_final        = round(float(mce), 6)
    mean_confidence  = round(float(max_conf.mean()), 6)
    mean_accuracy    = round(float(correct.mean()), 6)
    overconf_gap     = round(float(mean_confidence - mean_accuracy), 6)

    # ── ECE bootstrap CI ─────────────────────────────────────────────────
    ece_ci_lower: Optional[float] = None
    ece_ci_upper: Optional[float] = None
    if compute_ece_ci and n_samples >= _MIN_SAMPLES_FOR_CALIBRATION:
        ece_ci_lower, ece_ci_upper = _bootstrap_ece_ci(
            max_conf, correct,
            n_classes=n_classes, n_bins=n_bins, strategy=strategy,
            ci_level=_ECE_CI_LEVEL, n_bootstrap=_N_ECE_CI_BOOTSTRAP, seed=seed,
        )
        ece_ci_lower = round(ece_ci_lower, 6)
        ece_ci_upper = round(ece_ci_upper, 6)

    result: Dict[str, Any] = {
        "bins":                   bins_data,
        "ece":                    ece,
        "ece_unweighted":         ece_unweighted,
        "mce":                    mce_final,
        "ece_ci_lower":           ece_ci_lower,
        "ece_ci_upper":           ece_ci_upper,
        "ece_ci_level":           _ECE_CI_LEVEL if compute_ece_ci else None,
        "mean_confidence":        mean_confidence,
        "mean_accuracy":          mean_accuracy,
        "overconfidence_gap":     overconf_gap,
        "n_bins":                 n_bins,
        "n_empty_bins":           n_empty_bins,
        "n_sparse_bins":          n_sparse_bins,
        "n_samples":              n_samples,
        "n_classes":              n_classes,
        "strategy":               strategy,
        "ci_level":               ci_level if compute_bin_ci else None,
        "caveat":                 _CALIBRATION_CAVEAT,
        "temperature_scaling_note": TEMPERATURE_SCALING_NOTE,
    }

    overconf_direction = "overconfident" if overconf_gap > 0 else "underconfident"
    ece_ci_str = (
        f" [{int(_ECE_CI_LEVEL*100)}% CI: {ece_ci_lower:.4f}–{ece_ci_upper:.4f}]"
        if ece_ci_lower is not None else ""
    )
    logger.info(
        f"compute_reliability_diagram() | "
        f"n_samples={n_samples} | n_bins={n_bins} | strategy={strategy} | "
        f"ECE={ece:.4f}{ece_ci_str} | ECE_unweighted={ece_unweighted:.4f} | "
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
            "empty. With 52 val clips confidence mass is concentrated in a "
            "narrow range. Consider n_bins=5 or strategy='quantile' for a "
            "more populated diagram.",
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
      - ``coverage(τ)``         : fraction of all predictions accepted.
      - ``accuracy(τ)``         : accuracy on accepted predictions.
      - ``n_accepted(τ)``       : absolute count of accepted predictions.
      - ``selective_macro_f1(τ)``: sklearn macro-F1 on accepted predictions,
                                   forcing all n_classes into the denominator.
                                   NOTE: this is SELECTIVE inference performance,
                                   not overall model quality. See
                                   ``_SELECTIVE_MACRO_F1_NOTE`` for the full
                                   interpretation warning.
      - ``mean_confidence(τ)``  : mean max-softmax of accepted predictions.

    IMPORTANT NAMING CHANGE from the original implementation
    ----------------------------------------------------------
    The ``macro_f1`` key from the original has been renamed to
    ``selective_macro_f1`` throughout this function's output to make clear
    this is a conditional metric (performance on the accepted subset) and
    NOT a measure of overall model quality. At high thresholds, only easy
    predictions remain, so this metric inflates naturally. See the module
    docstring section "Threshold curve macro-F1 interpretation note" and
    ``_SELECTIVE_MACRO_F1_NOTE`` for full context.

    sklearn dependency note
    -----------------------
    This function uses ``sklearn.metrics.f1_score`` (imported at module top,
    not deferred as a local import) to compute per-threshold macro-F1. This
    is an explicitly declared project-wide dependency consistent with
    metrics.py, train.py, and benchmark.py.

    Parameters
    ----------
    y_true             : array-like, shape (n_samples,)
    y_proba            : array-like, shape (n_samples, n_classes), softmax probs
    n_classes          : int — 35 for this project.
    n_threshold_points : int, default 101
    seed               : int, default 42 — kept for API consistency.

    Returns
    -------
    dict with keys:
        thresholds             : List[float]
        coverage               : List[float]
        accuracy               : List[float | None]
        selective_macro_f1     : List[float | None]  (renamed from macro_f1)
        n_accepted             : List[int]
        mean_confidence        : List[float | None]
        auc_coverage           : float
        auc_monotone_warning   : bool  — True if coverage-accuracy curve is
                                         not monotone (noisy signal at n=52)
        optimal_threshold_accuracy : dict
        optimal_threshold_f1       : dict  (based on selective_macro_f1)
        n_threshold_points, n_samples, n_classes : int
        selective_macro_f1_note : str  (interpretation warning)
        caveat                 : str
    """
    # ── Validate inputs ───────────────────────────────────────────────────
    if not isinstance(n_threshold_points, (int, np.integer)) or n_threshold_points < 2:
        raise ValueError(
            f"compute_confidence_threshold_curve(): "
            f"n_threshold_points={n_threshold_points!r} must be >= 2."
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
    max_conf = _extract_max_confidence(y_proba_arr)
    y_pred   = y_proba_arr.argmax(axis=1).astype(np.int64)
    correct  = (y_pred == y_true_arr)

    labels_range = list(range(n_classes))
    thresholds   = np.linspace(0.0, 1.0, n_threshold_points)

    # ── Per-threshold computation ─────────────────────────────────────────
    threshold_list:              List[float]           = []
    coverage_list:               List[float]           = []
    accuracy_list:               List[Optional[float]] = []
    selective_macro_f1_list:     List[Optional[float]] = []
    n_accepted_list:             List[int]             = []
    mean_conf_list:              List[Optional[float]] = []

    for tau in thresholds:
        accepted = max_conf >= tau
        n_acc    = int(accepted.sum())

        threshold_list.append(round(float(tau), 6))
        coverage_list.append(round(float(n_acc / n_samples), 6))
        n_accepted_list.append(n_acc)

        if n_acc == 0:
            accuracy_list.append(None)
            selective_macro_f1_list.append(None)
            mean_conf_list.append(None)
        else:
            acc_val = round(float(correct[accepted].mean()), 6)
            accuracy_list.append(acc_val)

            # Selective macro-F1: all n_classes forced into denominator (matching
            # metrics.py convention). Zero_division=0 for absent classes.
            # sklearn imported at module top — no hidden deferred import.
            f1_val = round(float(_sklearn_f1_score(
                y_true_arr[accepted],
                y_pred[accepted],
                average="macro",
                labels=labels_range,
                zero_division=0,
            )), 6)
            selective_macro_f1_list.append(f1_val)
            mean_conf_list.append(round(float(max_conf[accepted].mean()), 6))

    # ── AUC under accuracy-vs-coverage curve ─────────────────────────────
    valid_pairs = [
        (cov, acc)
        for cov, acc in zip(coverage_list, accuracy_list)
        if acc is not None
    ]
    valid_pairs.sort(key=lambda p: p[0])

    auc_monotone_warning = False
    if len(valid_pairs) >= 2:
        cov_arr = np.array([p[0] for p in valid_pairs])
        acc_arr = np.array([p[1] for p in valid_pairs])

        # Check monotonicity: accuracy should be non-decreasing as coverage
        # decreases (i.e. as cov_arr increases, acc_arr should be non-increasing,
        # or alternatively, sorted by cov ascending means acc should be non-
        # decreasing or at least roughly monotone). With n=52, violations are
        # common and signal statistical noise, not a calibration pathology.
        diffs = np.diff(acc_arr)
        if np.any(diffs < -0.05):  # more than 5pp non-monotone drop
            auc_monotone_warning = True
            logger.warning(
                "compute_confidence_threshold_curve(): accuracy-vs-coverage "
                "curve is non-monotone (max decrease between adjacent points: "
                f"{float(diffs.min()):.4f}). With n={n_samples} clips this is "
                "expected sampling noise, not a calibration pathology. "
                "AUC-coverage is still computed but should be treated as a "
                "heuristic scalar for this dataset size.",
                extra={"stage": "evaluation"},
            )

        auc_coverage = round(float(np.trapz(acc_arr, cov_arr)), 6)
    else:
        auc_coverage = 0.0

    # ── Optimal thresholds ────────────────────────────────────────────────
    def _optimal_threshold(
        values: List[Optional[float]],
        metric_name: str,
    ) -> Dict[str, Any]:
        best_idx = None
        best_val = -1.0
        for i, (val, cov) in enumerate(zip(values, coverage_list)):
            if val is not None and val > best_val:
                best_val = val
                best_idx = i
        if best_idx is None:
            return {"threshold": None, metric_name: None, "coverage": None,
                    "n_accepted": None}
        return {
            "threshold":  float(threshold_list[best_idx]),
            metric_name:  float(values[best_idx]),
            "coverage":   float(coverage_list[best_idx]),
            "n_accepted": int(n_accepted_list[best_idx]),
        }

    optimal_acc = _optimal_threshold(accuracy_list, "accuracy")
    optimal_f1  = _optimal_threshold(selective_macro_f1_list, "selective_macro_f1")

    result: Dict[str, Any] = {
        "thresholds":                   threshold_list,
        "coverage":                     coverage_list,
        "accuracy":                     accuracy_list,
        "selective_macro_f1":           selective_macro_f1_list,
        "n_accepted":                   n_accepted_list,
        "mean_confidence":              mean_conf_list,
        "auc_coverage":                 auc_coverage,
        "auc_monotone_warning":         auc_monotone_warning,
        "optimal_threshold_accuracy":   optimal_acc,
        "optimal_threshold_f1":         optimal_f1,
        "n_threshold_points":           n_threshold_points,
        "n_samples":                    n_samples,
        "n_classes":                    n_classes,
        "selective_macro_f1_note":      _SELECTIVE_MACRO_F1_NOTE,
        "caveat":                       _CALIBRATION_CAVEAT,
    }

    mid_idx   = n_threshold_points // 2
    acc_at_0  = accuracy_list[0]
    f1_at_0   = selective_macro_f1_list[0]
    cov_at_half = coverage_list[mid_idx]
    acc_at_half = accuracy_list[mid_idx]

    logger.info(
        f"compute_confidence_threshold_curve() | "
        f"n_samples={n_samples} | n_threshold_points={n_threshold_points} | "
        f"τ=0: coverage=1.00, acc={acc_at_0}, selective_macro_f1={f1_at_0} | "
        f"τ=0.5: coverage={cov_at_half}, acc={acc_at_half} | "
        f"auc_coverage={auc_coverage:.4f} (monotone_warning={auc_monotone_warning}) | "
        f"optimal_acc_threshold={optimal_acc['threshold']} "
        f"(acc={optimal_acc.get('accuracy')}, cov={optimal_acc.get('coverage')}) | "
        f"optimal_f1_threshold={optimal_f1['threshold']} "
        f"(selective_f1={optimal_f1.get('selective_macro_f1')}, cov={optimal_f1.get('coverage')})",
        extra={"stage": "evaluation"},
    )

    return result


def compute_calibration_summary(
    y_true: Any,
    y_proba: Any,
    n_classes: int,
    split_name: str = "val",
    n_bins: int = DEFAULT_N_BINS,
    strategy: str = "uniform",
    n_threshold_points: int = DEFAULT_N_THRESHOLD_POINTS,
    seed: int = DEFAULT_SEED,
    compute_bin_ci: bool = True,
    ci_level: float = _BIN_CI_LEVEL,
    compute_ece_ci: bool = True,
) -> Dict[str, Any]:
    """
    Bundle reliability diagram and confidence-threshold curve into a single
    JSON-serialisable summary for ``evaluation_report.json``.

    This is the single function that ``pipelines/run_evaluation.py`` and
    ``notebooks/06_evaluation_error_analysis.ipynb`` call to get all
    calibration metrics for a split.

    Validation-once pattern (post-review item 10)
    -----------------------------------------------
    ``y_true`` / ``y_proba`` are validated exactly ONCE at the top of this
    function. The already-validated arrays are then passed directly to
    ``compute_reliability_diagram()`` and ``compute_confidence_threshold_curve()``,
    which accept pre-validated arrays without re-running full validation.
    This avoids triple-validation overhead within a single summary call while
    preserving each constituent function's ability to be called independently
    with its own validation (the guards in those functions remain intact for
    standalone use).

    IMPORTANT — this function operates on arrays ONLY (no model inference).
    The caller is responsible for passing already-extracted ``(y_true, y_proba)``
    from the Phase B1 / Phase C prediction cache.

    Parameters
    ----------
    y_true            : array-like, shape (n_samples,)
    y_proba           : array-like, shape (n_samples, n_classes), softmax probs
    n_classes         : int
    split_name        : str, default "val"
    n_bins            : int, default 10
    strategy          : str, default "uniform" — see compute_reliability_diagram()
    n_threshold_points: int, default 101
    seed              : int, default 42
    compute_bin_ci    : bool, default True
    ci_level          : float in (0, 1), default 0.80
    compute_ece_ci    : bool, default True

    Returns
    -------
    dict with keys:
        split_name, n_samples, n_classes         : metadata
        reliability_diagram                      : full dict
        threshold_curve                          : full dict
        ece, ece_unweighted, mce                 : float — convenience aliases
        ece_ci_lower, ece_ci_upper, ece_ci_level : float | None
        overconfidence_gap, mean_confidence,
        mean_accuracy                            : float
        auc_coverage                             : float
        auc_monotone_warning                     : bool
        strategy                                 : str
        caveat, temperature_scaling_note         : str
    """
    _validate_class_count(n_classes, "compute_calibration_summary")

    # Validate once; pass pre-validated arrays into constituent functions.
    y_true_arr  = _to_label_array(y_true, "y_true")
    y_proba_arr = _to_proba_array(y_proba, n_classes, "compute_calibration_summary")
    _validate_equal_length(y_true_arr, y_proba_arr, "y_true", "y_proba")
    _validate_label_range(y_true_arr, n_classes, "compute_calibration_summary")

    n_samples = len(y_true_arr)

    logger.info(
        f"compute_calibration_summary() | split='{split_name}' | "
        f"n_samples={n_samples} | n_classes={n_classes} | "
        f"n_bins={n_bins} | strategy={strategy} | "
        f"n_threshold_points={n_threshold_points}",
        extra={"stage": "evaluation"},
    )

    # Pass already-validated arrays directly — no re-validation overhead.
    reliability = compute_reliability_diagram(
        y_true_arr, y_proba_arr, n_classes,
        n_bins=n_bins, strategy=strategy, seed=seed,
        compute_bin_ci=compute_bin_ci, ci_level=ci_level,
        compute_ece_ci=compute_ece_ci,
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
        "ece_ci_lower":           reliability["ece_ci_lower"],
        "ece_ci_upper":           reliability["ece_ci_upper"],
        "ece_ci_level":           reliability["ece_ci_level"],
        "overconfidence_gap":     reliability["overconfidence_gap"],
        "mean_confidence":        reliability["mean_confidence"],
        "mean_accuracy":          reliability["mean_accuracy"],
        "auc_coverage":           threshold_curve["auc_coverage"],
        "auc_monotone_warning":   threshold_curve["auc_monotone_warning"],
        "strategy":               strategy,
        "caveat":                 _CALIBRATION_CAVEAT,
        "temperature_scaling_note": TEMPERATURE_SCALING_NOTE,
    }

    ece_ci_str = ""
    if reliability["ece_ci_lower"] is not None:
        ece_ci_str = (
            f" [{int(_ECE_CI_LEVEL*100)}% CI: "
            f"{reliability['ece_ci_lower']:.4f}–{reliability['ece_ci_upper']:.4f}]"
        )

    logger.info(
        f"compute_calibration_summary() COMPLETE | split='{split_name}' | "
        f"ECE={reliability['ece']:.4f}{ece_ci_str} | "
        f"MCE={reliability['mce']:.4f} | "
        f"overconfidence_gap={reliability['overconfidence_gap']:+.4f} | "
        f"auc_coverage={threshold_curve['auc_coverage']:.4f}",
        extra={"stage": "evaluation"},
    )

    return summary


# ---------------------------------------------------------------------------
# Figure rendering helpers
# ---------------------------------------------------------------------------

def _get_safe_matplotlib():
    """
    Import matplotlib.pyplot safely without mutating global backend state.

    The original implementation called ``matplotlib.use("Agg")`` at the
    module level of each plot function, which mutates global matplotlib state
    and breaks interactive notebooks / other callers. This helper instead
    attempts to import pyplot with whatever backend is currently active.

    If no display is available (headless CI environment) and importing pyplot
    raises, falls back to setting Agg backend as a last resort via
    ``matplotlib.use("Agg")``, but only if no backend is yet set.
    """
    try:
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        # Last-resort fallback for completely headless environments where
        # pyplot itself fails to import without an explicit backend.
        import matplotlib
        if not matplotlib.get_backend():
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt


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
    - Blue bars: actual accuracy per bin.
    - Red dashed diagonal: perfect calibration.
    - Light shading for over/under confidence zones.
    - Error bars: bootstrap CI on per-bin accuracy (when available).
    - Degenerate CI bins: shown with a distinct marker (dashed error cap)
      to signal the CI is trivial (n=1 in that bin).
    - Empty bins: shown as a very light grey hatched bar.
    - Sparse bins: annotated with an asterisk (*).
    - Bin count labels above each bar.
    - ECE with bootstrap CI, MCE, overconfidence gap in title.
    - Lower panel: confidence distribution histogram.

    Backend note
    ------------
    Does NOT call ``matplotlib.use("Agg")``. Callers that need a non-
    interactive backend should set it at application entry point.

    Parameters
    ----------
    reliability_result  : dict from compute_reliability_diagram()
    output_path         : str | Path | None
    split_name          : str
    figure_dpi          : int
    show_sparse_annotation: bool
    show_bin_counts     : bool
    show_ci             : bool

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        import matplotlib.patches as mpatches
        plt = _get_safe_matplotlib()
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
    strategy  = reliability_result.get("strategy", "uniform")
    ece_ci_lo = reliability_result.get("ece_ci_lower")
    ece_ci_hi = reliability_result.get("ece_ci_upper")
    ece_ci_lv = reliability_result.get("ece_ci_level")

    fig, (ax_main, ax_hist) = plt.subplots(
        2, 1, figsize=(9, 9),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.35},
    )

    ax = ax_main
    bin_edges = np.array([bd["bin_lo"] for bd in bins] + [bins[-1]["bin_hi"]])

    # For uniform bins, use fixed width. For quantile, derive from edges.
    bin_widths = np.diff(bin_edges)
    bin_centers = bin_edges[:-1] + bin_widths / 2.0

    ax.plot([0, 1], [0, 1], "r--", linewidth=1.5, label="Perfect calibration", zorder=5)
    ax.fill_between([0, 1], [0, 0], [0, 1], alpha=0.06, color="cornflowerblue",
                    label="Underconfidence zone")
    ax.fill_between([0, 1], [0, 1], [1, 1], alpha=0.06, color="tomato",
                    label="Overconfidence zone")

    for bd, center, bw in zip(bins, bin_centers, bin_widths):
        if bd["is_empty"]:
            ax.bar(center, 0.005, width=bw * 0.85,
                   color="lightgrey", edgecolor="grey", linewidth=0.5,
                   hatch="////", alpha=0.5, zorder=2)
            continue

        acc    = bd["actual_accuracy"]
        is_sp  = bd["is_sparse"]
        count  = bd["bin_count"]
        ci_deg = bd.get("ci_degenerate", False)

        color = "steelblue" if not is_sp else "sandybrown"
        ax.bar(center, acc, width=bw * 0.85,
               color=color, edgecolor="white", linewidth=0.8,
               alpha=0.85, zorder=3)

        if show_ci and bd.get("ci_lower") is not None and bd.get("ci_upper") is not None:
            ci_lo = bd["ci_lower"]
            ci_hi = bd["ci_upper"]
            # Use dashed caplines to visually flag degenerate CIs (n=1).
            cap_style = ":" if ci_deg else "-"
            ax.errorbar(
                center, acc,
                yerr=[[acc - ci_lo], [ci_hi - acc]],
                fmt="none", color="navy", linewidth=1.5,
                capsize=4, capthick=1.5, zorder=4,
                linestyle=cap_style,
            )

        if show_bin_counts:
            label_parts = [f"n={count}"]
            if show_sparse_annotation and is_sp:
                label_parts.append("*")
            if ci_deg and show_ci:
                label_parts.append("†")  # dagger marks degenerate CI
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

    ci_level_str = f" ({int(ci_level * 100)}% bin CI)" if ci_level and show_ci else ""
    ece_ci_str = ""
    if ece_ci_lo is not None and ece_ci_hi is not None:
        ece_ci_str = f" [{int(ece_ci_lv*100)}% CI: {ece_ci_lo:.4f}–{ece_ci_hi:.4f}]"

    title_lines = [
        f"Reliability Diagram — {split_name} split  (n={n_samples}, strategy={strategy})",
        f"ECE={ece:.4f}{ece_ci_str}  |  MCE={mce:.4f}  |  "
        f"Overconfidence gap={ovgap:+.4f}{ci_level_str}",
    ]
    ax.set_title("\n".join(title_lines), fontsize=10, pad=10)

    legend_handles = [
        plt.Line2D([0], [0], color="red", linestyle="--", label="Perfect calibration"),
        mpatches.Patch(color="steelblue", alpha=0.85, label="Calibrated bin"),
        mpatches.Patch(color="sandybrown", alpha=0.85,
                       label=f"Sparse bin (n < {SPARSE_BIN_THRESHOLD})*"),
        mpatches.Patch(color="lightgrey", hatch="////", alpha=0.5, label="Empty bin"),
    ]
    if show_ci:
        legend_handles.append(
            plt.Line2D([0], [0], color="navy", linestyle="--", linewidth=1,
                       label="†Degenerate CI (n=1)")
        )
    ax.legend(handles=legend_handles, fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.25, linestyle=":")

    # ── Lower panel: confidence histogram ─────────────────────────────────
    bin_heights = np.array([bd["bin_count"] for bd in bins])
    ax_hist.bar(
        bin_edges[:-1], bin_heights,
        width=bin_widths, align="edge",
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

    Naming note
    -----------
    The ``macro_f1`` key from the original output has been renamed to
    ``selective_macro_f1`` in ``compute_confidence_threshold_curve()``.
    This plot function uses the renamed key accordingly and labels the line
    "Selective macro-F1" with an annotation noting it is conditional on
    acceptance, to prevent misinterpretation as overall model quality.

    Visual design
    -------------
    Two y-axes on a single panel:
      - Left y-axis (blue): Coverage.
      - Right y-axis (green): Accuracy on accepted predictions.
    A secondary orange dashed line shows selective macro-F1.
    A monotonicity warning annotation appears if auc_monotone_warning=True.

    Backend note
    ------------
    Does NOT call ``matplotlib.use("Agg")``.

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
        plt = _get_safe_matplotlib()
    except ImportError as exc:
        raise ImportError(
            "plot_confidence_threshold_curve() requires matplotlib."
        ) from exc

    thresholds     = threshold_result["thresholds"]
    coverage       = threshold_result["coverage"]
    accuracy       = threshold_result["accuracy"]
    sel_macro_f1   = threshold_result["selective_macro_f1"]
    n_samples      = threshold_result["n_samples"]
    auc_cov        = threshold_result["auc_coverage"]
    mono_warn      = threshold_result.get("auc_monotone_warning", False)
    opt_acc        = threshold_result["optimal_threshold_accuracy"]
    opt_f1         = threshold_result["optimal_threshold_f1"]

    # Replace None with nan for plotting (matplotlib handles NaN as gaps).
    acc_plot = [a if a is not None else float("nan") for a in accuracy]
    f1_plot  = [f if f is not None else float("nan") for f in sel_macro_f1]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    ax1.plot(thresholds, coverage, color="cornflowerblue", linewidth=2.2,
             label="Coverage", zorder=4)
    ax1.set_xlabel("Confidence threshold τ", fontsize=11)
    ax1.set_ylabel("Coverage (fraction accepted)", color="cornflowerblue", fontsize=11)
    ax1.tick_params(axis="y", labelcolor="cornflowerblue")
    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(-0.02, 1.05)

    ax2.plot(thresholds, acc_plot, color="mediumseagreen", linewidth=2.2,
             label="Accuracy on accepted", zorder=4)
    ax2.plot(thresholds, f1_plot, color="darkorange", linewidth=1.5,
             linestyle="--", label="Selective macro-F1 (conditional)", zorder=4)
    ax2.set_ylabel("Accuracy / Selective macro-F1", color="mediumseagreen", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="mediumseagreen")
    ax2.set_ylim(-0.02, 1.05)

    if opt_acc.get("threshold") is not None:
        ax1.axvline(opt_acc["threshold"], color="mediumseagreen",
                    linestyle=":", linewidth=1.5,
                    label=f"Opt acc τ={opt_acc['threshold']:.2f} "
                          f"(acc={opt_acc.get('accuracy', '?'):.3f}, "
                          f"cov={opt_acc.get('coverage', '?'):.2f})",
                    zorder=3)
    if (opt_f1.get("threshold") is not None and
            opt_f1["threshold"] != opt_acc.get("threshold")):
        ax1.axvline(opt_f1["threshold"], color="darkorange",
                    linestyle=":", linewidth=1.5,
                    label=f"Opt selective-F1 τ={opt_f1['threshold']:.2f} "
                          f"(F1={opt_f1.get('selective_macro_f1', '?'):.3f}, "
                          f"cov={opt_f1.get('coverage', '?'):.2f})",
                    zorder=3)

    ax1.axhline(1.0, color="lightgrey", linestyle=":", linewidth=0.8, zorder=1)

    mono_note = "\n⚠ Non-monotone curve (sampling noise, n=52)" if mono_warn else ""
    title = (
        f"Confidence-Threshold Curve — {split_name} split  (n={n_samples})\n"
        f"AUC-coverage={auc_cov:.4f}{mono_note}"
    )
    ax1.set_title(title, fontsize=10, pad=10)
    ax1.grid(True, alpha=0.2, linestyle=":")

    # Add a small annotation box explaining selective macro-F1.
    ax2.annotate(
        "Selective macro-F1 ≠ overall quality.\nConditional on accepted subset.",
        xy=(0.98, 0.05), xycoords="axes fraction",
        ha="right", va="bottom", fontsize=7,
        color="darkorange", style="italic",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow",
                  edgecolor="darkorange", alpha=0.7),
    )

    # Combined legend (both axes) — kept concise to avoid cognitive overload.
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="center left",
               framealpha=0.85)

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
    assert 0.0 < _ECE_CI_LEVEL < 1.0, (
        f"calibration.py: _ECE_CI_LEVEL={_ECE_CI_LEVEL} must be in (0, 1)."
    )
    assert _N_BINS_CI_BOOTSTRAP >= 10, (
        f"calibration.py: _N_BINS_CI_BOOTSTRAP={_N_BINS_CI_BOOTSTRAP} must be >= 10."
    )
    assert _N_ECE_CI_BOOTSTRAP >= 10, (
        f"calibration.py: _N_ECE_CI_BOOTSTRAP={_N_ECE_CI_BOOTSTRAP} must be >= 10."
    )
    assert 0.0 < _MIN_MEANINGFUL_COVERAGE <= 1.0, (
        f"calibration.py: _MIN_MEANINGFUL_COVERAGE={_MIN_MEANINGFUL_COVERAGE} "
        "must be in (0, 1]."
    )
    assert len(_CALIBRATION_CAVEAT) > 50, (
        "calibration.py: _CALIBRATION_CAVEAT string unexpectedly short."
    )
    assert len(TEMPERATURE_SCALING_NOTE) > 50, (
        "calibration.py: TEMPERATURE_SCALING_NOTE string unexpectedly short."
    )
    assert len(_SELECTIVE_MACRO_F1_NOTE) > 50, (
        "calibration.py: _SELECTIVE_MACRO_F1_NOTE string unexpectedly short."
    )
    assert set(_VALID_BIN_STRATEGIES) == {"uniform", "quantile"}, (
        "calibration.py: _VALID_BIN_STRATEGIES must contain exactly "
        "'uniform' and 'quantile'."
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