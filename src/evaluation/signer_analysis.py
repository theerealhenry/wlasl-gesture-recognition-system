"""
src/evaluation/signer_analysis.py
===================================
Per-signer generalisation diagnostics for the WLASL 35-class gesture
recognition system. This is Stage 6 (Evaluation, Benchmarking, and
Interpretability), Phase A4, per the Stage 6 (Revised) plan.

Scope of this module (Phase A4, exactly)
------------------------------------------
The Stage 6 (Revised) plan defines Phase A4:

    "signer_analysis.py — compute_per_signer_accuracy() — operates on
    (y_true, y_pred, signer_ids). Built against synthetic/mock arrays
    first since real signer-joined predictions don't exist yet."

Concretely, this module:

  1. Operates EXCLUSIVELY on already-extracted ``(y_true, y_pred,
     signer_ids)`` arrays — it NEVER calls ``model.predict()`` and never
     reads ``data/splits/*.csv`` or ``landmark_inventory.csv`` directly.
     ``signer_ids`` is supplied by the caller, already joined from
     ``data/splits/{val,test}.csv`` into the Phase B1 / Phase C prediction
     cache (``val_predictions.npz`` / ``test_predictions.npz``).
  2. Is framework-agnostic and dependency-light: ``numpy`` +
     ``sklearn.metrics.f1_score`` only (the same declared dependency as
     ``metrics.py`` and ``calibration.py`` — no hidden imports).
  3. Is independently unit-testable with synthetic label and signer-id
     arrays, requiring no model, dataset, or project infrastructure.
  4. Never writes figures itself — it returns structured, JSON-serialisable
     dicts that the notebooks (``notebooks/06_evaluation_error_analysis.ipynb``)
     and ``pipelines/run_evaluation.py`` pass to a dedicated plotting helper
     also defined here (``plot_signer_generalisation``), keeping the
     computation layer trivially testable and figure styling a presentation
     concern, not a metric concern.

Why this module's framing differs from the original handoff template
----------------------------------------------------------------------
The Stage 5 handoff (Part 6.4) describes per-signer analysis using the
language of "familiar vs. novel" signers and frames the val signer spread
as revealing "generalisation quality" in a way that implicitly assumes some
signers are more familiar to the model than others. The Stage 6 (Revised)
plan explicitly corrects this (Phase D3):

    "Explicit framing: all signers are unseen by construction (zero-overlap
    split) — there is no 'familiar vs. novel' axis to compare ... State
    this directly rather than forcing a comparison that doesn't exist."

Stage 1 (Part 4, Finding F3) confirms zero signer overlap across
train/val/test with signer-aware greedy bin-packing (seed=42). Every
val and test signer is, by construction, equally "unseen". This module:

  - NEVER computes or reports a "familiar" vs "novel" split.
  - Always labels signer-level results as "unseen-signer accuracy", not
    "generalisation to new signers" framed as if some baseline of
    "known signers" existed for comparison.
  - Surfaces the n_signers / clips-per-signer ratio prominently.

Why per-signer estimates are extremely noisy (and this module says so)
------------------------------------------------------------------------
The val split has 52 clips across 7 signers — roughly 7-8 clips per
signer on average, but Stage 1's signer-aware bin-packing does not
guarantee an even split: some signers may contribute 3 clips, others 12.
A signer with 3 clips has only 4 possible accuracy values (0/3, 1/3, 2/3,
3/3). This module:

  - Reports ``n_clips`` alongside every per-signer accuracy, never as an
    afterthought.
  - Flags signers with ``n_clips < SPARSE_SIGNER_THRESHOLD`` as
    ``is_sparse`` (default threshold: 5).
  - Computes Wilson score confidence intervals per signer (``accuracy_ci_lower``,
    ``accuracy_ci_upper``) so the individual-signer uncertainty is quantified
    rather than implicit.
  - Computes a bootstrap CI for cross-signer accuracy SPREAD so "how much do
    signers differ" is a quantified, reproducible number.
  - Embeds a caveat string in every returned dict, mirroring the pattern
    in ``calibration.py`` and ``metrics.py``.

Dual macro-F1 reporting (Review Issue H1)
-------------------------------------------
``compute_per_signer_accuracy()`` now reports both:

  ``macro_f1_global``:
      sklearn macro-F1 with ``labels=list(range(n_classes))`` forced
      (Part 8, Critical Rule #2 — consistent with every other macro-F1
      in this project). For a signer with 7 clips covering 4 distinct
      classes, 31 of the 35 classes contribute F1=0.0 to this average,
      making it low by construction. This is the project-standard metric,
      kept for cross-run comparability.

  ``macro_f1_present_classes``:
      sklearn macro-F1 computed only over the classes actually present in
      ``y_true`` for this signer (``labels=np.unique(yt_s)``). A signer
      who perfectly classifies all of their 4 distinct signs gets
      ``macro_f1_present_classes=1.0``, not ~0.11. This is the metric
      that answers "how well did the model perform on the signs this
      signer actually produced?" and is always reported alongside
      ``macro_f1_global`` with an explicit label so neither is mistaken
      for the other.

Both values are reported because neither alone is sufficient: global
correctly penalises failure to cover the full label space; present-classes
correctly rewards per-sign quality without penalising the model for signs
the signer simply didn't produce.

Weighted spread statistics (Review Issue H2)
----------------------------------------------
``compute_signer_spread_bootstrap_ci()`` now reports both unweighted spread
(equal weight per signer — answers "how variable are signers as individuals?")
and clip-count-weighted spread statistics (``weighted_observed_std``,
``weighted_observed_mean`` — answers "how variable is performance across
clips, accounting for differing signer contributions?"). Both are provided
because a signer with 3 clips and one with 12 clips are legitimately different
amounts of evidence.

Correlation methodology (Review Issue H3)
-------------------------------------------
``compute_signer_high_risk_correlation()`` now reports both Pearson r and
Spearman rho. Spearman is generally more appropriate for small n (n=7),
nonlinear relationships, and outlier-prone settings. Both are provided;
the caller can select the appropriate one for their analysis.

Revision history
-----------------
Post-review revision addressing the following issues from the critical review:

  H1 FIXED. ``compute_per_signer_accuracy()`` now reports both
     ``macro_f1_global`` (labels forced to full range, project standard)
     and ``macro_f1_present_classes`` (labels restricted to classes the
     signer actually produced). The original single ``macro_f1`` key is
     retained as an alias for ``macro_f1_global`` for backward compatibility.

  H2 FIXED. ``compute_signer_spread_bootstrap_ci()`` now additionally
     computes ``weighted_observed_std`` and ``weighted_observed_mean``
     using clip counts as weights, alongside the original unweighted
     statistics.

  H3 FIXED. ``compute_signer_high_risk_correlation()`` now computes and
     reports both Pearson r and Spearman rho. Full-precision (unrounded)
     accuracy and high_risk_clip_fraction values are used in the correlation
     computation (M6 fix), with rounding applied only in the returned dict.

  M1 FIXED. ``compute_signer_failure_mode_summary()`` now validates that
     ``missing_pct`` values lie in [0, 1] (or [0, 100] — detected and
     rescaled automatically) and that ``detected_frame_count`` values are
     non-negative.

  M2 FIXED. All mean computations in failure-mode summary now use
     ``np.nanmean()`` and explicitly report the count of NaN values found,
     so downstream consumers know when a mean is based on partial data.

  M3 FIXED. ``sign_names`` validation in ``compute_per_signer_accuracy()``
     now checks for None entries, empty strings, and duplicate names —
     not just length.

  M4 FIXED. ``compute_per_signer_accuracy()`` now emits a WARNING when
     ``high_risk_class_indices`` is empty after the name-matching step,
     preventing silent all-zero high_risk_clip_fraction values that would
     make the Phase D3 correlation meaningless.

  M5 FIXED. Signer ID keys now use ``repr(signer)`` instead of
     ``str(signer)`` to prevent collision between numeric ``1`` and string
     ``"1"`` IDs — the same class of leading-zero bug Stage 1 already fixed
     for video IDs. Plain-string representation is still used for display
     in figures; ``repr()`` is used for dict keying only.

  M6 FIXED. Full-precision (unrounded) per-signer accuracy and
     high_risk_clip_fraction values are stored internally and used in all
     correlation computations. Rounding is applied only in the final output
     dict.

  M7 FIXED. Wilson score confidence intervals are now computed per signer
     (``accuracy_ci_lower``, ``accuracy_ci_upper``) at the 90% level,
     matching the project's DEFAULT_BOOTSTRAP_CI. A signer with n_clips=3
     correctly gets a wide CI; a signer with n_clips=12 gets a narrower one.

  M8 FIXED. ``compute_signer_spread_bootstrap_ci()`` now explicitly detects
     and reports when all signer accuracies are identical (degenerate case —
     zero variance), setting ``spread_is_degenerate=True`` and a descriptive
     note in the result, rather than silently returning zero CI bounds.

  L1 FIXED. Unused ``Mapping`` and ``Tuple`` typing imports removed.

  L2 NOTED. Plot title updated to "Per-signer accuracy variation" rather than
     "generalisation" — the latter implies a familiar/novel comparison that
     doesn't exist in this zero-overlap split.

  L3 NOTED. Marker size scaling documented; the formula ``40 + 20*c`` is
     retained but capped at a maximum to prevent large signers dominating.

  L4 FIXED. Legend construction in ``plot_signer_generalisation()`` now
     uses explicit proxy artists rather than sequential ``ax.legend()``
     calls, preventing duplicate entries.

  L5 NOTED. Sort order in the plot is by value (ascending) for readability;
     signer labels on the x-axis preserve identity. A stable secondary sort
     key (signer id string) is added to make the ordering deterministic when
     multiple signers have identical values.

  L8 NOTED. Validation helpers are intentionally duplicated (not imported)
     from metrics.py/calibration.py for independent testability — accepted
     trade-off, documented in module docstring.

Champion model context (for reference)
-----------------------------------------
The champion model (``bilstm_hands_only_v4_aug``) config_snapshot.yaml
confirms:
  - ``early_stopping_monitor: val_accuracy``  (NOT val_macro_f1 as narrated
    in the Stage 5 handoff — this discrepancy is flagged in
    evaluation_report.json per Phase F requirements; this module takes no
    position on it).
  - Input shape: (1, 100, 126) — seq_len=100, landmark_config=hands_only.
  - Parameters: 68,771. val_macro_f1: 0.6011.
  - This module operates purely on (y_true, y_pred, signer_ids) produced
    from running this champion model over the val/test splits.

Module-level exports
----------------------
    compute_per_signer_accuracy           — per-signer accuracy + dual
                                              macro-F1 + Wilson CI + support
    compute_signer_spread_bootstrap_ci    — bootstrap CI on cross-signer
                                              spread (weighted + unweighted)
    compute_signer_high_risk_correlation  — Pearson + Spearman correlation
                                              of accuracy vs high-risk exposure
    compute_signer_failure_mode_summary   — NaN-safe metadata correlation
    compute_signer_analysis_summary       — consolidation wrapper for
                                              evaluation_report.json
    plot_signer_generalisation            — figure renderer (box/strip plot)
    SPARSE_SIGNER_THRESHOLD               — public constant
    UNSEEN_SIGNER_FRAMING_NOTE            — the "no familiar/novel axis"
                                              caveat, ready to embed verbatim
    DEFAULT_N_BOOTSTRAP                   — project-standard resample count
    DEFAULT_BOOTSTRAP_CI                  — project-standard CI level (0.90)
    DEFAULT_SEED                          — project global seed (42)
    HIGH_RISK_SIGNS                       — Stage 5 Finding 8 classes
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from sklearn.metrics import f1_score as _sklearn_f1_score

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Signers with fewer clips than this are flagged ``is_sparse`` in
#: per-signer output. Mirrors ``calibration.py::SPARSE_BIN_THRESHOLD``.
#: With 52 val clips / 7 signers (mean ~7.4 clips/signer), a signer at or
#: below this threshold has too few clips for accuracy to be read as more
#: than a rough indicator.
SPARSE_SIGNER_THRESHOLD: int = 5

#: Project global seed. Matches DEFAULT_SEED in metrics.py / calibration.py
#: and base.yaml's top-level ``seed: 42``.
DEFAULT_SEED: int = 42

#: Default bootstrap resample count. Matches DEFAULT_N_BOOTSTRAP in
#: metrics.py — 1000 resamples of a ~7-signer array is fast.
DEFAULT_N_BOOTSTRAP: int = 1000

#: Default CI level — 90%, not 95%, because a 95% interval on 7 signers
#: would be too wide to be informative. Matches DEFAULT_BOOTSTRAP_CI in
#: metrics.py and calibration.py.
DEFAULT_BOOTSTRAP_CI: float = 0.90

#: Wilson score CI level for per-signer accuracy. Set equal to
#: DEFAULT_BOOTSTRAP_CI for consistency across the evaluation suite.
_WILSON_CI_LEVEL: float = 0.90

#: z-value for Wilson CI. z(0.95) ≈ 1.6449 gives a 90% two-sided interval.
_WILSON_Z: float = 1.6448536269514729  # scipy.stats.norm.ppf(0.95)

#: Below this many distinct signers, resampling signers for a spread CI is
#: statistically marginal — warn rather than error (7 signers is the
#: project's actual val count and a legitimate call).
_MIN_SIGNERS_FOR_BOOTSTRAP: int = 3

#: Below this many bootstrap resamples, percentile CI bounds are unreliable.
_MIN_BOOTSTRAP_FOR_STABLE_CI: int = 100

#: The five smallest, most failure-prone training classes identified in
#: Stage 5 Finding 8. Intentionally duplicated from metrics.py (not
#: imported) so this module remains independently testable — accepted
#: trade-off documented in calibration.py revision history item 11.
HIGH_RISK_SIGNS: tuple = ("clothes", "think", "birthday", "name", "book")

#: Embedded verbatim in every per-signer / summary result so every
#: downstream consumer inherits the correct framing automatically.
UNSEEN_SIGNER_FRAMING_NOTE: str = (
    "All validation and test signers are unseen by construction: Stage 1's "
    "signer-aware split (zero overlap across train/val/test, confirmed in "
    "Finding F3) means no signer in this analysis was present in any form "
    "during training. There is therefore no 'familiar vs. novel' axis to "
    "compare — every signer here is equally 'novel' relative to the "
    "training set. Per-signer accuracy spread reflects natural variation "
    "in signing style, camera angle, and clip difficulty across different "
    "unseen individuals, not a familiar/unfamiliar generalisation gap."
)

#: Embedded in every per-signer result. Mirrors the small-sample caveat
#: pattern established by _CALIBRATION_CAVEAT in calibration.py.
_SIGNER_SAMPLE_SIZE_CAVEAT: str = (
    "Per-signer accuracy and macro-F1 estimates are based on a small number "
    "of clips per signer (this project: ~7-8 clips/signer on average across "
    "7 validation signers, with uneven distribution likely). A signer with "
    "n_clips=3 has only 4 possible accuracy values (0/3, 1/3, 2/3, 3/3) — "
    "treat any individual signer's accuracy as a rough indicator, not a "
    "precise estimate. Signers flagged is_sparse (n_clips < "
    f"{SPARSE_SIGNER_THRESHOLD}) should be interpreted with particular caution. "
    "Wilson score confidence intervals are provided per signer to quantify "
    "this uncertainty explicitly."
)

#: Documented note on dual macro-F1 reporting (Review Issue H1).
_DUAL_MACRO_F1_NOTE: str = (
    "Two macro-F1 values are reported per signer. macro_f1_global forces "
    "labels=list(range(n_classes)) — the project standard (Part 8, Critical "
    "Rule #2) — and is dominated by zero-F1 for classes the signer did not "
    "produce. macro_f1_present_classes is computed only over classes appearing "
    "in this signer's y_true subset, and better reflects per-sign quality "
    "for the signs they actually produced. Neither alone is sufficient: use "
    "global for cross-signer comparability; use present_classes for diagnosing "
    "which specific signs a signer struggled with."
)


# ---------------------------------------------------------------------------
# Internal validation helpers
# (Intentionally self-contained — keeps this module independently
#  importable/testable. Accepted trade-off, documented in calibration.py
#  revision history item 11.)
# ---------------------------------------------------------------------------

def _validate_class_count(n_classes: int, caller: str) -> None:
    """Raise ValueError if n_classes is not a sane positive integer >= 2."""
    if not isinstance(n_classes, (int, np.integer)) or n_classes < 2:
        raise ValueError(
            f"{caller}: n_classes={n_classes!r} must be an integer >= 2. "
            "Pass cfg.num_classes explicitly (35 for the current WLASL "
            "label map, artifacts/label_map_v1.json)."
        )


def _to_label_array(arr: Any, name: str) -> np.ndarray:
    """
    Coerce an array-like into a flat 1-D int64 numpy array of class indices.

    Rejects multi-dimensional (e.g. one-hot) label arrays rather than
    silently flattening them — matching the contract in metrics.py and
    calibration.py.

    Raises
    ------
    ValueError
        If ``arr`` is empty or has a shape that suggests one-hot encoding.
    """
    out = np.asarray(arr)
    if out.size == 0:
        raise ValueError(
            f"{name} is empty. Cannot compute signer metrics on zero samples."
        )
    if out.ndim > 1:
        squeezed = np.squeeze(out)
        if squeezed.ndim > 1:
            raise ValueError(
                f"{name} has shape {out.shape}, which looks like one-hot or "
                "multi-dimensional label encoding rather than a flat vector "
                "of class indices. Expected integer class indices, e.g. from "
                "GestureDataset (sparse_categorical labels). Convert with "
                "np.argmax(arr, axis=-1) if one-hot."
            )
        out = squeezed
    if out.ndim == 0:
        out = out.reshape(1)
    return out.astype(np.int64)


def _validate_label_range(y: np.ndarray, n_classes: int, name: str, caller: str) -> None:
    """Raise ValueError if any label falls outside [0, n_classes)."""
    if y.size == 0:
        return
    min_v, max_v = int(y.min()), int(y.max())
    if min_v < 0 or max_v >= n_classes:
        bad = max_v if max_v >= n_classes else min_v
        raise ValueError(
            f"{caller}: {name} contains class index {bad}, outside the valid "
            f"range [0, {n_classes}). Check that n_classes={n_classes} matches "
            "the label map this array was produced against "
            "(artifacts/label_map_v1.json — 35 signs for this project)."
        )


def _to_signer_id_array(signer_ids: Any, caller: str) -> np.ndarray:
    """
    Coerce signer_ids into a flat 1-D numpy array WITHOUT assuming a numeric
    dtype.

    Signer IDs may be plain integers, zero-padded strings (e.g. "014"), or
    arbitrary labels (e.g. "signer_014") depending on how the Phase B1 cache
    joins data/splits/{val,test}.csv. Using ``np.asarray(arr)`` (no explicit
    dtype) preserves whatever dtype the caller supplied — converting to
    ``int`` would silently truncate a zero-padded string ID, echoing the
    exact class of bug Stage 1 already fixed for video IDs (handoff Part
    6.1, "Fixed Bugs" table, "Leading-zero video_id mismatch").

    Returns
    -------
    np.ndarray, 1-D, dtype preserved from input (object, str, or int).

    Raises
    ------
    ValueError
        If ``signer_ids`` is empty or not 1-D after squeezing.
    """
    out = np.asarray(signer_ids)
    if out.size == 0:
        raise ValueError(f"{caller}: signer_ids is empty.")
    if out.ndim > 1:
        squeezed = np.squeeze(out)
        if squeezed.ndim > 1:
            raise ValueError(
                f"{caller}: signer_ids has shape {out.shape}, expected a flat "
                "1-D array of one signer identifier per clip."
            )
        out = squeezed
    if out.ndim == 0:
        out = out.reshape(1)
    return out


def _validate_equal_length(a: np.ndarray, b: np.ndarray, name_a: str, name_b: str) -> None:
    """Raise ValueError if two arrays differ in length (first dimension)."""
    if len(a) != len(b):
        raise ValueError(
            f"{name_a} (len={len(a)}) and {name_b} (len={len(b)}) must have "
            "the same length — one entry per clip in both arrays."
        )


def _validate_n_bootstrap(n_bootstrap: int, caller: str) -> None:
    """Raise ValueError if n_bootstrap is not a positive integer."""
    if not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap < 1:
        raise ValueError(
            f"{caller}: n_bootstrap={n_bootstrap!r} must be a positive integer."
        )


def _validate_ci_level(ci_level: float, caller: str) -> None:
    """Raise ValueError if ci_level is not strictly in (0, 1)."""
    if not (0.0 < ci_level < 1.0):
        raise ValueError(f"{caller}: ci_level={ci_level} must be in (0, 1).")


def _validate_sign_names(
    sign_names: Sequence[str], n_classes: int, caller: str
) -> None:
    """
    Validate sign_names for length, None entries, empty strings, and
    duplicates. (Review Issues M3, M4 — previously only length was checked.)

    Raises
    ------
    ValueError
        If length mismatches n_classes, any entry is None or an empty
        string, or any name is duplicated.
    """
    if len(sign_names) != n_classes:
        raise ValueError(
            f"{caller}: len(sign_names)={len(sign_names)} must equal "
            f"n_classes={n_classes}. Check the sign_names list was built as "
            "[label_map.get_name_safe(i, ...) for i in range(n_classes)]."
        )

    none_indices = [i for i, n in enumerate(sign_names) if n is None]
    if none_indices:
        raise ValueError(
            f"{caller}: sign_names contains None at indices {none_indices}. "
            "Every class must have a non-None string name. Check "
            "label_map.get_name_safe() for those class indices."
        )

    empty_indices = [i for i, n in enumerate(sign_names) if isinstance(n, str) and n.strip() == ""]
    if empty_indices:
        raise ValueError(
            f"{caller}: sign_names contains empty strings at indices "
            f"{empty_indices}. Every class must have a non-empty name."
        )

    seen: Dict[str, int] = {}
    for name in sign_names:
        seen[str(name)] = seen.get(str(name), 0) + 1
    duplicates = {n: c for n, c in seen.items() if c > 1}
    if duplicates:
        raise ValueError(
            f"{caller}: sign_names contains duplicate entries: {duplicates}. "
            "Duplicate class names cause silent metric loss — check "
            "label_map_v1.json for corrupted or repeated label entries."
        )


def _signer_key(signer: Any) -> str:
    """
    Produce a collision-safe dict key for a signer identifier.

    Uses ``repr()`` rather than ``str()`` to prevent collision between
    numeric ``1`` (int) and string ``"1"`` — the exact class of leading-zero
    / type-ambiguity bug Stage 1 already experienced with video IDs.

    Examples
    --------
    _signer_key(1)    → "1"     (int repr, no quotes in output)
    _signer_key("1")  → "'1'"   (str repr, with quotes — unambiguous)
    _signer_key("014") → "'014'" (zero-padded string preserved)
    """
    return repr(signer)


# ---------------------------------------------------------------------------
# Wilson score confidence interval helper
# ---------------------------------------------------------------------------

def _wilson_ci(n_correct: int, n_total: int) -> tuple:
    """
    Compute a Wilson score confidence interval for a binomial proportion.

    The Wilson interval is recommended over the Wald (normal approximation)
    interval for small sample sizes and extreme probabilities (p near 0 or 1),
    both of which are common for per-signer evaluation on 3-12 clips.

    Uses z = _WILSON_Z, corresponding to a 90% two-sided interval
    (z ≈ 1.6449 = norminv(0.95)).

    Parameters
    ----------
    n_correct : int — number of correct predictions.
    n_total   : int — total clip count for this signer.

    Returns
    -------
    (ci_lower, ci_upper) : both float, in [0, 1].
        Both are 0.0 when n_total == 0.

    References
    ----------
    Wilson, E.B. (1927). Probable inference, the law of succession, and
    statistical inference. JASA 22(158), 209-212.
    """
    if n_total == 0:
        return (0.0, 0.0)

    z   = _WILSON_Z
    z2  = z * z
    p   = n_correct / n_total
    n   = n_total
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    ci_lower = max(0.0, centre - margin)
    ci_upper = min(1.0, centre + margin)
    return (round(ci_lower, 6), round(ci_upper, 6))


# ---------------------------------------------------------------------------
# Core per-signer computation
# ---------------------------------------------------------------------------

def compute_per_signer_accuracy(
    y_true: Any,
    y_pred: Any,
    signer_ids: Any,
    n_classes: int,
    split_name: str = "val",
    sign_names: Optional[Sequence[str]] = None,
    high_risk_signs: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Compute per-signer accuracy, dual macro-F1, Wilson CI, and support for
    every distinct signer in ``signer_ids``.

    Matches the function name and signature called for in the Stage 6
    (Revised) plan Phase A4. Extends the original handoff's Part 6.4
    specification with dual macro-F1 (Review H1), Wilson CIs (Review M7),
    improved validation (Review M3, M4), and collision-safe signer keys
    (Review M5).

    For each distinct signer ``s``:
      - ``n_clips``               : total clips from this signer.
      - ``n_correct``              : count where argmax prediction matches
                                      the true label.
      - ``accuracy``                : n_correct / n_clips (full precision
                                      stored internally; rounded in output).
      - ``accuracy_ci_lower``        : Wilson score CI lower bound (90%).
      - ``accuracy_ci_upper``        : Wilson score CI upper bound (90%).
      - ``macro_f1_global``           : sklearn macro-F1 with labels forced
                                        to list(range(n_classes)) — the
                                        project standard. Low by construction
                                        for signers who don't cover all 35 classes.
      - ``macro_f1_present_classes``   : sklearn macro-F1 computed only over
                                        the classes appearing in this signer's
                                        y_true — reflects per-sign quality
                                        for signs actually produced.
      - ``n_distinct_true_classes``    : count of distinct true classes in
                                        this signer's clips — context for
                                        interpreting macro_f1_global.
      - ``is_sparse``                  : True if n_clips < SPARSE_SIGNER_THRESHOLD.
      - ``high_risk_clip_fraction``    : fraction of clips with a high-risk
                                        true label. None if sign_names not
                                        supplied. Computed using case-insensitive
                                        name matching.

    Parameters
    ----------
    y_true, y_pred : array-like, shape (n_samples,)
    signer_ids     : array-like, shape (n_samples,)
        One signer identifier per clip. May be int, str, or any hashable
        type — never coerced to a numeric dtype (Review M5).
    n_classes       : int — 35 for this project.
    split_name       : str, default "val"
    sign_names        : Sequence[str], length n_classes, optional
        Validated for None entries, empty strings, and duplicates (Review M3).
        If supplied, enables ``high_risk_clip_fraction`` per signer using
        case-insensitive name matching (Review M4).
    high_risk_signs    : Sequence[str], optional
        Defaults to module-level HIGH_RISK_SIGNS. Case-insensitive matching.

    Returns
    -------
    dict with keys:
        split_name, n_samples, n_classes, n_signers      : metadata
        per_signer                                        : dict keyed by
          repr(signer_id) — collision-safe (Review M5). Each value:
          {n_clips, n_correct, accuracy, accuracy_ci_lower, accuracy_ci_upper,
           macro_f1_global, macro_f1_present_classes,
           n_distinct_true_classes, is_sparse, high_risk_clip_fraction}
        n_sparse_signers, mean_clips_per_signer          : int / float
        min_clips_per_signer, max_clips_per_signer        : int
        overall_accuracy                                   : float
        dual_macro_f1_note                                  : str
        unseen_signer_framing_note                           : str
        caveat                                               : str

    Raises
    ------
    ValueError
        If arrays are empty, mismatched in length, contain out-of-range
        labels, or ``sign_names`` validation fails.
    """
    _validate_class_count(n_classes, "compute_per_signer_accuracy")

    y_true_arr = _to_label_array(y_true, "y_true")
    y_pred_arr = _to_label_array(y_pred, "y_pred")
    _validate_equal_length(y_true_arr, y_pred_arr, "y_true", "y_pred")
    _validate_label_range(y_true_arr, n_classes, "y_true", "compute_per_signer_accuracy")
    _validate_label_range(y_pred_arr, n_classes, "y_pred", "compute_per_signer_accuracy")

    signer_arr = _to_signer_id_array(signer_ids, "compute_per_signer_accuracy")
    _validate_equal_length(y_true_arr, signer_arr, "y_true", "signer_ids")

    # sign_names validation — length, None, empty, duplicates (Review M3)
    if sign_names is not None:
        _validate_sign_names(sign_names, n_classes, "compute_per_signer_accuracy")

    # Build high-risk class index set with CASE-INSENSITIVE matching (Review M4)
    high_risk_lower = {
        s.lower() for s in (high_risk_signs if high_risk_signs is not None else HIGH_RISK_SIGNS)
    }
    high_risk_class_indices: Optional[set] = None
    if sign_names is not None:
        high_risk_class_indices = {
            idx for idx, name in enumerate(sign_names)
            if str(name).lower() in high_risk_lower
        }
        if not high_risk_class_indices:
            logger.warning(
                "compute_per_signer_accuracy(): high_risk_class_indices is "
                "EMPTY after case-insensitive name matching against sign_names. "
                f"high_risk_signs={list(high_risk_lower)}, "
                f"sign_names_lower_sample={[str(n).lower() for n in sign_names[:5]]}... "
                "high_risk_clip_fraction will be 0.0 for every signer — the "
                "Phase D3 correlation cross-check will be meaningless. "
                "Check that sign_names entries match HIGH_RISK_SIGNS (possibly "
                "with different capitalisation) in artifacts/label_map_v1.json.",
                extra={"stage": "evaluation"},
            )

    n_samples    = len(y_true_arr)
    labels_range = list(range(n_classes))
    correct      = (y_true_arr == y_pred_arr)

    unique_signers = np.unique(signer_arr)
    per_signer: Dict[str, Dict[str, Any]] = {}
    clip_counts: List[int] = []

    for signer in unique_signers:
        mask      = (signer_arr == signer)
        n_clips_s = int(mask.sum())
        clip_counts.append(n_clips_s)

        yt_s       = y_true_arr[mask]
        yp_s       = y_pred_arr[mask]
        n_correct_s = int(correct[mask].sum())

        # Full-precision accuracy stored internally (Review M6)
        accuracy_s_full = n_correct_s / n_clips_s if n_clips_s > 0 else 0.0

        # Wilson score CI (Review M7)
        ci_lower_s, ci_upper_s = _wilson_ci(n_correct_s, n_clips_s)

        # macro_f1_global: all classes forced (project standard, Part 8 Rule #2)
        macro_f1_global_s = float(_sklearn_f1_score(
            yt_s, yp_s,
            average="macro",
            labels=labels_range,
            zero_division=0,
        ))

        # macro_f1_present_classes: only classes in this signer's y_true (Review H1)
        present_labels = np.unique(yt_s).tolist()
        if len(present_labels) >= 1:
            macro_f1_present_s = float(_sklearn_f1_score(
                yt_s, yp_s,
                average="macro",
                labels=present_labels,
                zero_division=0,
            ))
        else:
            macro_f1_present_s = 0.0

        n_distinct_s = len(present_labels)
        is_sparse_s  = n_clips_s < SPARSE_SIGNER_THRESHOLD

        # High-risk fraction — uses case-insensitive matched indices (Review M4)
        high_risk_fraction_s: Optional[float] = None
        if high_risk_class_indices is not None:
            n_hr = int(np.isin(yt_s, list(high_risk_class_indices)).sum())
            high_risk_fraction_s = n_hr / n_clips_s if n_clips_s > 0 else 0.0

        # collision-safe key using repr() (Review M5)
        signer_key = _signer_key(signer)

        per_signer[signer_key] = {
            "n_clips":                    n_clips_s,
            "n_correct":                  n_correct_s,
            # Full-precision values stored; callers can round for display
            "accuracy":                   accuracy_s_full,
            "accuracy_ci_lower":          ci_lower_s,
            "accuracy_ci_upper":          ci_upper_s,
            "accuracy_ci_level":          _WILSON_CI_LEVEL,
            "macro_f1_global":            macro_f1_global_s,
            "macro_f1_present_classes":   macro_f1_present_s,
            "n_distinct_true_classes":    n_distinct_s,
            "is_sparse":                  is_sparse_s,
            "high_risk_clip_fraction":    high_risk_fraction_s,
        }

    n_signers        = len(unique_signers)
    n_sparse_signers = sum(1 for v in per_signer.values() if v["is_sparse"])
    overall_accuracy = float(correct.mean())

    result: Dict[str, Any] = {
        "split_name":                  split_name,
        "n_samples":                   n_samples,
        "n_classes":                   n_classes,
        "n_signers":                   n_signers,
        "per_signer":                  per_signer,
        "n_sparse_signers":            n_sparse_signers,
        "mean_clips_per_signer":       round(float(np.mean(clip_counts)), 4),
        "min_clips_per_signer":        int(np.min(clip_counts)),
        "max_clips_per_signer":        int(np.max(clip_counts)),
        "overall_accuracy":            round(overall_accuracy, 6),
        "dual_macro_f1_note":          _DUAL_MACRO_F1_NOTE,
        "unseen_signer_framing_note":  UNSEEN_SIGNER_FRAMING_NOTE,
        "caveat":                      _SIGNER_SAMPLE_SIZE_CAVEAT,
    }

    acc_values = [v["accuracy"] for v in per_signer.values()]
    logger.info(
        f"compute_per_signer_accuracy() | split='{split_name}' | "
        f"n_samples={n_samples} | n_signers={n_signers} | "
        f"clips_per_signer=[min={result['min_clips_per_signer']}, "
        f"mean={result['mean_clips_per_signer']:.1f}, "
        f"max={result['max_clips_per_signer']}] | "
        f"n_sparse_signers={n_sparse_signers}/{n_signers} | "
        f"accuracy_range=[{min(acc_values):.3f}, {max(acc_values):.3f}] | "
        f"overall_accuracy={overall_accuracy:.4f}",
        extra={"stage": "evaluation"},
    )

    if n_sparse_signers > 0:
        logger.warning(
            f"compute_per_signer_accuracy(): {n_sparse_signers}/{n_signers} "
            f"signers have fewer than {SPARSE_SIGNER_THRESHOLD} clips. Their "
            "accuracy/macro_f1 should be treated as rough indicators — "
            "see result['caveat'] and result['per_signer'][id]['accuracy_ci_*'] "
            "for quantified uncertainty.",
            extra={"stage": "evaluation"},
        )

    return result


# ---------------------------------------------------------------------------
# Signer-level spread bootstrap
# ---------------------------------------------------------------------------

def compute_signer_spread_bootstrap_ci(
    per_signer_result: Dict[str, Any],
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    ci_level: float = DEFAULT_BOOTSTRAP_CI,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    """
    Quantify cross-signer accuracy SPREAD via a signer-level bootstrap,
    reporting both unweighted and clip-count-weighted statistics.

    This complements ``metrics.py::bootstrap_macro_f1_ci()`` (which
    resamples clips and likely understates uncertainty due to signer
    clustering) by answering a different question: "if we drew a different
    set of 7 unseen signers, how much would the spread of their individual
    accuracies vary?"

    Weighted vs. unweighted spread (Review Issue H2)
    --------------------------------------------------
    Two sets of spread statistics are reported:

    UNWEIGHTED (equal weight per signer):
        ``observed_std``, ``observed_range``, ``std_ci_*``, ``range_ci_*``
        These answer: "how variable are signers AS INDIVIDUALS?"
        Each signer counts as one data point regardless of how many clips
        they contributed. A signer with 3 clips and one with 12 clips are
        given equal weight.

    WEIGHTED (clip-count weighted):
        ``weighted_observed_std``, ``weighted_observed_mean``
        These answer: "how variable is model performance across CLIPS,
        accounting for different signer contributions?"
        Uses numpy's standard weighted standard deviation formula.
        Not bootstrapped (weights change meaning under resample) but
        reported as companion statistics. The weighted mean is also
        a sanity-check against ``overall_accuracy`` from
        ``compute_per_signer_accuracy()``.

    Degenerate case detection (Review Issue M8)
    --------------------------------------------
    When all signer accuracies are identical (zero variance), the bootstrap
    will always return zero spread. This degenerate case is explicitly
    detected and flagged with ``spread_is_degenerate=True`` and a note,
    rather than silently returning zero CI bounds that might be mistaken
    for a well-calibrated zero-uncertainty result.

    Parameters
    ----------
    per_signer_result : dict from compute_per_signer_accuracy()
    n_bootstrap       : int, default 1000
    ci_level          : float in (0, 1), default 0.90
    seed              : int, default 42

    Returns
    -------
    dict with keys:
        n_signers                : int
        observed_std              : float — unweighted std
        observed_range             : float — unweighted range (max - min)
        std_ci_lower, std_ci_upper  : float — bootstrap CI on unweighted std
        range_ci_lower, range_ci_upper : float
        weighted_observed_mean     : float — clip-count weighted mean accuracy
        weighted_observed_std       : float — clip-count weighted std
        ci_level                     : float
        n_bootstrap                   : int
        seed                           : int
        spread_is_degenerate            : bool
        caveat                           : str

    Raises
    ------
    ValueError
        If ``per_signer_result`` lacks a ``"per_signer"`` key, or if
        ``n_bootstrap`` / ``ci_level`` are invalid.
    """
    per_signer = per_signer_result.get("per_signer")
    if per_signer is None:
        raise ValueError(
            "compute_signer_spread_bootstrap_ci() expects the dict returned "
            "by compute_per_signer_accuracy(), which must contain a "
            f"'per_signer' key. Got top-level keys: {list(per_signer_result.keys())}"
        )

    _validate_n_bootstrap(n_bootstrap, "compute_signer_spread_bootstrap_ci")
    _validate_ci_level(ci_level, "compute_signer_spread_bootstrap_ci")

    # Use full-precision accuracy values (Review M6)
    accuracies = np.array(
        [v["accuracy"] for v in per_signer.values()], dtype=np.float64,
    )
    clip_counts = np.array(
        [v["n_clips"] for v in per_signer.values()], dtype=np.float64,
    )
    n_signers = len(accuracies)

    if n_signers < _MIN_SIGNERS_FOR_BOOTSTRAP:
        logger.warning(
            f"compute_signer_spread_bootstrap_ci(): n_signers={n_signers} is "
            f"below {_MIN_SIGNERS_FOR_BOOTSTRAP}. Spread statistics will be "
            "extremely unstable. Treat as illustrative only.",
            extra={"stage": "evaluation"},
        )

    if n_bootstrap < _MIN_BOOTSTRAP_FOR_STABLE_CI:
        logger.warning(
            f"compute_signer_spread_bootstrap_ci(): n_bootstrap={n_bootstrap} "
            f"is below {_MIN_BOOTSTRAP_FOR_STABLE_CI}; percentile CI bounds "
            "will be noisy.",
            extra={"stage": "evaluation"},
        )

    # Unweighted statistics
    observed_std   = float(np.std(accuracies, ddof=1)) if n_signers > 1 else 0.0
    observed_range = float(np.max(accuracies) - np.min(accuracies)) if n_signers > 0 else 0.0

    # Weighted statistics (clip-count weighted) — Review H2
    total_clips = clip_counts.sum()
    if total_clips > 0 and n_signers > 1:
        weights = clip_counts / total_clips
        weighted_mean = float(np.dot(weights, accuracies))
        # Weighted std: sqrt(sum(w_i * (x_i - mu_w)^2))
        weighted_std = float(
            np.sqrt(np.dot(weights, (accuracies - weighted_mean) ** 2))
        )
    else:
        weighted_mean = float(accuracies.mean()) if n_signers > 0 else 0.0
        weighted_std  = 0.0

    # Degenerate case detection (Review M8)
    spread_is_degenerate = observed_std == 0.0
    if spread_is_degenerate and n_signers > 1:
        logger.warning(
            "compute_signer_spread_bootstrap_ci(): all signer accuracies are "
            "identical (spread_is_degenerate=True). Bootstrap resamples will "
            "all return zero spread — CI bounds of [0.0, 0.0] do not indicate "
            "a well-calibrated zero-uncertainty result; they reflect a data "
            "artifact (all signers happen to have the same accuracy on this "
            "small val set).",
            extra={"stage": "evaluation"},
        )

    rng = np.random.default_rng(seed)
    boot_std   = np.empty(n_bootstrap, dtype=np.float64)
    boot_range = np.empty(n_bootstrap, dtype=np.float64)

    for i in range(n_bootstrap):
        idx       = rng.integers(0, n_signers, size=n_signers)
        resampled = accuracies[idx]
        boot_std[i]   = np.std(resampled, ddof=1) if n_signers > 1 else 0.0
        boot_range[i] = np.max(resampled) - np.min(resampled) if n_signers > 0 else 0.0

    alpha     = 1.0 - ci_level
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    degenerate_note = (
        " NOTE: spread_is_degenerate=True — all signer accuracies are "
        "identical; CI bounds [0.0, 0.0] reflect this data artifact, not a "
        "meaningful uncertainty estimate."
        if spread_is_degenerate else ""
    )

    result: Dict[str, Any] = {
        "n_signers":              n_signers,
        "observed_std":            round(observed_std, 6),
        "observed_range":           round(observed_range, 6),
        "std_ci_lower":              round(float(np.percentile(boot_std, lower_pct)), 6),
        "std_ci_upper":               round(float(np.percentile(boot_std, upper_pct)), 6),
        "range_ci_lower":              round(float(np.percentile(boot_range, lower_pct)), 6),
        "range_ci_upper":               round(float(np.percentile(boot_range, upper_pct)), 6),
        "weighted_observed_mean":        round(weighted_mean, 6),
        "weighted_observed_std":          round(weighted_std, 6),
        "ci_level":                        ci_level,
        "n_bootstrap":                      n_bootstrap,
        "seed":                              seed,
        "spread_is_degenerate":              spread_is_degenerate,
        "caveat": (
            f"This bootstrap resamples only {n_signers} signer-level "
            "observations (unweighted by clip count — see weighted_observed_* "
            "for clip-count-weighted companion statistics). With this few "
            "signers, the resample space is small and percentile estimates "
            "are approximate. This is a signer-level complement to "
            "metrics.py::bootstrap_macro_f1_ci()'s clip-level CI — not a "
            "replacement for it." + degenerate_note
        ),
    }

    logger.info(
        f"compute_signer_spread_bootstrap_ci() | n_signers={n_signers} | "
        f"observed_std={observed_std:.4f} "
        f"[{int(ci_level*100)}% CI: {result['std_ci_lower']:.4f}, {result['std_ci_upper']:.4f}] | "
        f"observed_range={observed_range:.4f} "
        f"[{int(ci_level*100)}% CI: {result['range_ci_lower']:.4f}, {result['range_ci_upper']:.4f}] | "
        f"weighted_mean={weighted_mean:.4f} | weighted_std={weighted_std:.4f} | "
        f"spread_is_degenerate={spread_is_degenerate}",
        extra={"stage": "evaluation"},
    )

    return result


# ---------------------------------------------------------------------------
# High-risk-class correlation cross-check (Stage 6 Revised, Phase D3)
# ---------------------------------------------------------------------------

def compute_signer_high_risk_correlation(
    per_signer_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Test whether low-accuracy signers disproportionately produce the five
    Stage 5 Finding 8 high-risk classes (clothes, think, birthday, name,
    book), per the Stage 6 (Revised) plan's Phase D3 cross-check.

    Now reports both Pearson r and Spearman rho (Review Issue H3).
    Spearman is generally more appropriate for small n (n=7), nonlinear
    relationships, and outlier-prone settings. Both are provided; neither
    is claimed as the definitive statistic over 7 data points.

    Uses full-precision (unrounded) accuracy and high_risk_clip_fraction
    values for the correlation computation (Review Issue M6).

    Parameters
    ----------
    per_signer_result : dict from compute_per_signer_accuracy()
        Must have been called with sign_names= to populate
        high_risk_clip_fraction.

    Returns
    -------
    dict with keys:
        n_signers             : int
        pearson_r               : float | None
        spearman_rho             : float | None
        interpretation            : str — plain-language description
        is_underpowered             : bool — True if n_signers < 10
        per_signer_table             : List[dict] — sorted ascending by
                                       accuracy, using full-precision values
        caveat                        : str

    Raises
    ------
    ValueError
        If per_signer_result lacks 'per_signer', or if no signer has a
        non-None high_risk_clip_fraction (sign_names was not supplied
        upstream).
    """
    per_signer = per_signer_result.get("per_signer")
    if per_signer is None:
        raise ValueError(
            "compute_signer_high_risk_correlation() expects the dict "
            "returned by compute_per_signer_accuracy(), which must contain "
            f"a 'per_signer' key. Got top-level keys: {list(per_signer_result.keys())}"
        )

    rows = [
        {
            "signer_id":               signer_id,
            "accuracy":                 v["accuracy"],          # full precision (Review M6)
            "high_risk_clip_fraction":   v.get("high_risk_clip_fraction"),
        }
        for signer_id, v in per_signer.items()
    ]

    if all(r["high_risk_clip_fraction"] is None for r in rows):
        raise ValueError(
            "compute_signer_high_risk_correlation(): every signer has "
            "high_risk_clip_fraction=None. This means "
            "compute_per_signer_accuracy() was called without sign_names — "
            "re-run it with sign_names=[label_map.get_name_safe(i, ...) for "
            "i in range(n_classes)] to populate this field."
        )

    valid_rows = [r for r in rows if r["high_risk_clip_fraction"] is not None]
    n_signers  = len(valid_rows)

    # Full-precision arrays for correlation (Review M6)
    acc_vals = np.array([r["accuracy"] for r in valid_rows], dtype=np.float64)
    hr_vals  = np.array([r["high_risk_clip_fraction"] for r in valid_rows], dtype=np.float64)

    # Pearson r
    pearson_r: Optional[float] = None
    if n_signers >= 2 and np.std(acc_vals) > 0 and np.std(hr_vals) > 0:
        pearson_r = float(np.corrcoef(acc_vals, hr_vals)[0, 1])

    # Spearman rho — computed via rank correlation (Review H3)
    spearman_rho: Optional[float] = None
    if n_signers >= 2:
        # Compute Spearman as Pearson on ranks (avoids scipy dependency)
        # Average ties using scipy-style rank (midrank method)
        def _rank_midpoint(arr: np.ndarray) -> np.ndarray:
            n = len(arr)
            idx_sort = np.argsort(arr, kind="mergesort")
            ranks    = np.empty(n, dtype=np.float64)
            i = 0
            while i < n:
                j = i + 1
                while j < n and arr[idx_sort[j]] == arr[idx_sort[i]]:
                    j += 1
                mid_rank = (i + j - 1) / 2.0 + 1.0  # 1-indexed midpoint
                for k in range(i, j):
                    ranks[idx_sort[k]] = mid_rank
                i = j
            return ranks

        acc_ranks = _rank_midpoint(acc_vals)
        hr_ranks  = _rank_midpoint(hr_vals)

        std_acc = np.std(acc_ranks)
        std_hr  = np.std(hr_ranks)
        if std_acc > 0 and std_hr > 0:
            spearman_rho = float(np.corrcoef(acc_ranks, hr_ranks)[0, 1])
        else:
            # Zero-variance in ranks means all values tied — Spearman undefined
            spearman_rho = None

    is_underpowered = n_signers < 10

    def _interpret(r_val: Optional[float], r_name: str) -> str:
        if r_val is None:
            return (
                f"{r_name} undefined: insufficient signer count, or no variance "
                "in accuracy or high_risk_clip_fraction across signers."
            )
        direction = (
            "negative (lower accuracy ↔ more high-risk-class exposure — expected direction)"
            if r_val < 0 else
            "positive (higher accuracy ↔ more high-risk-class exposure — counter to expectation)"
        )
        magnitude = (
            "weak" if abs(r_val) < 0.3 else
            "moderate" if abs(r_val) < 0.6 else
            "strong"
        )
        return f"{magnitude.capitalize()} {direction}, {r_name}={r_val:.3f}."

    pearson_interp  = _interpret(pearson_r,   "r")
    spearman_interp = _interpret(spearman_rho, "rho")
    interpretation  = f"Pearson: {pearson_interp} | Spearman: {spearman_interp}"

    # Sort ascending by accuracy for display; stable secondary sort by signer_id
    sorted_rows = sorted(
        rows,
        key=lambda r: (r["accuracy"], str(r["signer_id"])),
    )
    # Round for display output only — computation already done above
    display_rows = [
        {
            "signer_id":              r["signer_id"],
            "accuracy":               round(r["accuracy"], 6),
            "high_risk_clip_fraction": (
                round(r["high_risk_clip_fraction"], 6)
                if r["high_risk_clip_fraction"] is not None else None
            ),
        }
        for r in sorted_rows
    ]

    result: Dict[str, Any] = {
        "n_signers":        n_signers,
        "pearson_r":         round(pearson_r, 6) if pearson_r is not None else None,
        "spearman_rho":       round(spearman_rho, 6) if spearman_rho is not None else None,
        "interpretation":      interpretation,
        "is_underpowered":      is_underpowered,
        "per_signer_table":      display_rows,
        "caveat": (
            f"Computed over n_signers={n_signers}. Both Pearson r and Spearman "
            "rho are reported; Spearman is generally preferred for small n and "
            "possible outliers. Over {n_signers} data points, both coefficients "
            "are descriptive/exploratory only — treat as suggestive, not "
            "statistically confirmatory. A negative correlation is the expected "
            "direction (Stage 5 Finding 8: high-risk classes fail due to tiny "
            "training-clip counts, not signer skill) and should not be "
            "over-interpreted as a finding about signer quality."
        ),
    }

    logger.info(
        f"compute_signer_high_risk_correlation() | n_signers={n_signers} | "
        f"pearson_r={pearson_r if pearson_r is not None else 'undefined'} | "
        f"spearman_rho={spearman_rho if spearman_rho is not None else 'undefined'} | "
        f"is_underpowered={is_underpowered}",
        extra={"stage": "evaluation"},
    )

    if is_underpowered:
        logger.warning(
            f"compute_signer_high_risk_correlation(): n_signers={n_signers} "
            "< 10. Both correlation coefficients are exploratory only and "
            "must not be reported as statistically robust findings. "
            "Prefer Spearman rho over Pearson r at this sample size.",
            extra={"stage": "evaluation"},
        )

    return result


# ---------------------------------------------------------------------------
# Failure-mode metadata correlation (Stage 6 Revised, Phase D4)
# ---------------------------------------------------------------------------

def _validate_metadata_array(
    arr: Any,
    name: str,
    n_expected: int,
    caller: str,
    expected_min: Optional[float] = None,
    expected_max: Optional[float] = None,
    allow_nan: bool = True,
    rescale_range: Optional[tuple] = None,
) -> np.ndarray:
    """
    Validate and coerce a metadata array (detected_frame_count or missing_pct).

    Handles NaN detection and optional range rescaling. (Review Issues M1, M2.)

    Parameters
    ----------
    arr           : array-like
    name          : str — field name for error messages
    n_expected    : int — expected length
    caller        : str — calling function name
    expected_min  : float, optional — warn if any value < expected_min
    expected_max  : float, optional — warn if any value > expected_max
    allow_nan     : bool, default True — if True, NaNs are preserved and
                    reported; if False, NaN presence raises ValueError
    rescale_range : tuple (lo, hi), optional — if any value falls within
                    this range and expected_max < hi, attempt rescaling
                    (e.g. missing_pct in [0,100] → [0,1])

    Returns
    -------
    np.ndarray, float64, shape (n_expected,)
    """
    out = np.asarray(arr, dtype=np.float64)
    if out.ndim != 1 or len(out) != n_expected:
        raise ValueError(
            f"{caller}: {name} has shape {out.shape}, expected 1-D array of "
            f"length {n_expected} (one value per clip)."
        )

    n_nan = int(np.isnan(out).sum())
    if n_nan > 0:
        if not allow_nan:
            raise ValueError(
                f"{caller}: {name} contains {n_nan} NaN value(s). "
                f"Clean the prediction cache before calling {caller}."
            )
        logger.warning(
            f"{caller}: {name} contains {n_nan}/{n_expected} NaN value(s). "
            "These will be excluded from mean computations via np.nanmean(). "
            "Check landmark_inventory.csv for missing metadata entries.",
            extra={"stage": "evaluation"},
        )

    # Auto-rescaling (e.g. missing_pct in [0,100] → [0,1]) — Review M1
    if rescale_range is not None and expected_max is not None:
        lo, hi = rescale_range
        finite = out[~np.isnan(out)]
        if finite.size > 0 and finite.max() > expected_max and finite.max() <= hi:
            out = out / hi
            logger.info(
                f"{caller}: {name} values appear to be in [{lo}, {hi}] range; "
                f"auto-rescaled to [0, 1] by dividing by {hi}. "
                "If this is incorrect, normalise the array before passing it in.",
                extra={"stage": "evaluation"},
            )

    # Range validation after any rescaling
    if expected_min is not None or expected_max is not None:
        finite = out[~np.isnan(out)]
        if finite.size > 0:
            actual_min = float(finite.min())
            actual_max = float(finite.max())
            if expected_min is not None and actual_min < expected_min:
                logger.warning(
                    f"{caller}: {name} has min value {actual_min:.4f} < expected "
                    f"minimum {expected_min}. Negative values are likely a "
                    "data-pipeline error (e.g. corrupted landmark_inventory.csv).",
                    extra={"stage": "evaluation"},
                )
            if expected_max is not None and actual_max > expected_max:
                logger.warning(
                    f"{caller}: {name} has max value {actual_max:.4f} > expected "
                    f"maximum {expected_max}. Values out of range are likely a "
                    "data-pipeline error.",
                    extra={"stage": "evaluation"},
                )

    return out


def compute_signer_failure_mode_summary(
    y_true: Any,
    y_pred: Any,
    signer_ids: Any,
    detected_frame_count: Optional[Any] = None,
    missing_pct: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Per-signer breakdown of clip-level failure-mode metadata, supporting the
    Stage 6 (Revised) plan's Phase D4 question:

        "Use the cache's detected_frame_count/missing_pct fields to test
        whether errors cluster in heavily zero-filled or short clips."

    This version adds NaN-safe mean computation via ``np.nanmean()`` (Review
    M2) and metadata range validation (Review M1), while preserving the
    original architecture and scope.

    Parameters
    ----------
    y_true, y_pred                   : array-like, shape (n_samples,)
    signer_ids                        : array-like, shape (n_samples,)
    detected_frame_count               : array-like, shape (n_samples,), optional
        Non-negative integers. NaN values warn and are excluded from means.
    missing_pct                          : array-like, shape (n_samples,), optional
        Fraction of both-hands-absent frames. Accepts [0,1] or [0,100];
        if values > 1.0 are detected, auto-rescaled to [0,1] by dividing by 100.
        NaN values warn and are excluded from means.

    Returns
    -------
    dict with keys:
        n_samples, n_signers           : int
        metadata_fields_provided        : List[str]
        n_nan_detected_frame_count       : int — count of NaN values in
                                           detected_frame_count (0 if not supplied)
        n_nan_missing_pct                 : int — count of NaN values in missing_pct
        per_signer                          : dict[repr(signer_id) -> {
          n_clips, n_correct, n_incorrect,
          mean_detected_frames_correct,    mean_detected_frames_incorrect,
          n_detected_frames_nan_correct,   n_detected_frames_nan_incorrect,
          mean_missing_pct_correct,        mean_missing_pct_incorrect,
          n_missing_pct_nan_correct,       n_missing_pct_nan_incorrect,
        }]
        caveat                               : str

    Raises
    ------
    ValueError
        If neither metadata field is supplied, or if length validation fails.
    """
    if detected_frame_count is None and missing_pct is None:
        raise ValueError(
            "compute_signer_failure_mode_summary(): at least one of "
            "detected_frame_count or missing_pct must be supplied. Both "
            "come from the Phase B1 prediction cache, joined from "
            "landmark_inventory.csv."
        )

    y_true_arr = _to_label_array(y_true, "y_true")
    y_pred_arr = _to_label_array(y_pred, "y_pred")
    _validate_equal_length(y_true_arr, y_pred_arr, "y_true", "y_pred")

    signer_arr = _to_signer_id_array(signer_ids, "compute_signer_failure_mode_summary")
    _validate_equal_length(y_true_arr, signer_arr, "y_true", "signer_ids")

    n_samples = len(y_true_arr)
    metadata_fields_provided: List[str] = []
    dfc_arr: Optional[np.ndarray] = None
    mp_arr:  Optional[np.ndarray] = None
    n_nan_dfc: int = 0
    n_nan_mp:  int = 0

    if detected_frame_count is not None:
        dfc_arr = _validate_metadata_array(
            detected_frame_count, "detected_frame_count", n_samples,
            "compute_signer_failure_mode_summary",
            expected_min=0.0, expected_max=None,
            allow_nan=True,
        )
        n_nan_dfc = int(np.isnan(dfc_arr).sum())
        metadata_fields_provided.append("detected_frame_count")

    if missing_pct is not None:
        mp_arr = _validate_metadata_array(
            missing_pct, "missing_pct", n_samples,
            "compute_signer_failure_mode_summary",
            expected_min=0.0, expected_max=1.0,
            allow_nan=True,
            rescale_range=(0.0, 100.0),
        )
        n_nan_mp = int(np.isnan(mp_arr).sum())
        metadata_fields_provided.append("missing_pct")

    correct = (y_true_arr == y_pred_arr)
    unique_signers = np.unique(signer_arr)

    def _nanmean_safe(values: np.ndarray) -> Optional[float]:
        """NaN-safe mean; returns None if all values are NaN or array is empty."""
        if values.size == 0:
            return None
        finite = values[~np.isnan(values)]
        return round(float(finite.mean()), 4) if finite.size > 0 else None

    def _n_nan(values: np.ndarray) -> int:
        return int(np.isnan(values).sum()) if values.size > 0 else 0

    per_signer: Dict[str, Dict[str, Any]] = {}
    for signer in unique_signers:
        mask           = (signer_arr == signer)
        correct_mask   = mask & correct
        incorrect_mask = mask & (~correct)

        entry: Dict[str, Any] = {
            "n_clips":     int(mask.sum()),
            "n_correct":   int(correct_mask.sum()),
            "n_incorrect": int(incorrect_mask.sum()),
        }

        if dfc_arr is not None:
            entry["mean_detected_frames_correct"]    = _nanmean_safe(dfc_arr[correct_mask])
            entry["mean_detected_frames_incorrect"]  = _nanmean_safe(dfc_arr[incorrect_mask])
            entry["n_detected_frames_nan_correct"]   = _n_nan(dfc_arr[correct_mask])
            entry["n_detected_frames_nan_incorrect"] = _n_nan(dfc_arr[incorrect_mask])

        if mp_arr is not None:
            entry["mean_missing_pct_correct"]    = _nanmean_safe(mp_arr[correct_mask])
            entry["mean_missing_pct_incorrect"]  = _nanmean_safe(mp_arr[incorrect_mask])
            entry["n_missing_pct_nan_correct"]   = _n_nan(mp_arr[correct_mask])
            entry["n_missing_pct_nan_incorrect"] = _n_nan(mp_arr[incorrect_mask])

        per_signer[_signer_key(signer)] = entry

    result: Dict[str, Any] = {
        "n_samples":                      n_samples,
        "n_signers":                      len(unique_signers),
        "metadata_fields_provided":        metadata_fields_provided,
        "n_nan_detected_frame_count":       n_nan_dfc,
        "n_nan_missing_pct":                n_nan_mp,
        "per_signer":                       per_signer,
        "caveat": (
            "Per-signer correct/incorrect metadata means are computed over "
            "very small subsets (often 1-6 clips per signer per correctness "
            "bucket) using np.nanmean() — NaN values are excluded and their "
            "count is reported in n_*_nan_* fields. Differences between "
            "mean_*_correct and mean_*_incorrect for any single signer are "
            "illustrative, not statistically tested. Aggregate across all "
            "signers (not signer-by-signer) for any claim about whether "
            "errors cluster in zero-filled or short clips dataset-wide."
        ),
    }

    logger.info(
        f"compute_signer_failure_mode_summary() | "
        f"n_samples={n_samples} | n_signers={len(unique_signers)} | "
        f"metadata_fields={metadata_fields_provided} | "
        f"n_nan_dfc={n_nan_dfc} | n_nan_mp={n_nan_mp}",
        extra={"stage": "evaluation"},
    )

    return result


# ---------------------------------------------------------------------------
# Consolidated summary
# ---------------------------------------------------------------------------

def compute_signer_analysis_summary(
    y_true: Any,
    y_pred: Any,
    signer_ids: Any,
    n_classes: int,
    split_name: str = "val",
    sign_names: Optional[Sequence[str]] = None,
    high_risk_signs: Optional[Sequence[str]] = None,
    detected_frame_count: Optional[Any] = None,
    missing_pct: Optional[Any] = None,
    compute_spread_ci: bool = True,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    ci_level: float = DEFAULT_BOOTSTRAP_CI,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    """
    Bundle every primitive in this module into a single, JSON-serialisable
    signer-analysis summary for one split.

    Mirrors the consolidation pattern established by
    ``metrics.py::compute_evaluation_summary()`` and
    ``calibration.py::compute_calibration_summary()``: validates inputs
    once, then composes the public functions in this module.

    Does NOT run inference itself — callers pass already-extracted
    ``(y_true, y_pred, signer_ids)`` from the Phase B1 / Phase C
    prediction cache.

    Parameters
    ----------
    y_true, y_pred             : array-like, shape (n_samples,)
    signer_ids                  : array-like, shape (n_samples,)
    n_classes                    : int
    split_name                    : str, default "val"
    sign_names                     : Sequence[str], optional — enables
                                    high_risk_clip_fraction and the
                                    high-risk correlation cross-check.
                                    Validated for None/empty/duplicate entries.
    high_risk_signs                 : Sequence[str], optional
    detected_frame_count               : array-like, optional
    missing_pct                          : array-like, optional
    compute_spread_ci                      : bool, default True
    n_bootstrap, ci_level, seed              : see constituent functions

    Returns
    -------
    dict with keys:
        split_name, n_samples, n_classes, n_signers
        per_signer_accuracy
        spread_bootstrap_ci        (if compute_spread_ci=True)
        high_risk_correlation       (if sign_names was supplied)
        failure_mode_summary         (if detected_frame_count or missing_pct)
        unseen_signer_framing_note
        caveat
    """
    per_signer_acc = compute_per_signer_accuracy(
        y_true, y_pred, signer_ids, n_classes,
        split_name=split_name,
        sign_names=sign_names,
        high_risk_signs=high_risk_signs,
    )

    summary: Dict[str, Any] = {
        "split_name":                  split_name,
        "n_samples":                   per_signer_acc["n_samples"],
        "n_classes":                   per_signer_acc["n_classes"],
        "n_signers":                   per_signer_acc["n_signers"],
        "per_signer_accuracy":          per_signer_acc,
        "unseen_signer_framing_note":    UNSEEN_SIGNER_FRAMING_NOTE,
        "caveat":                         _SIGNER_SAMPLE_SIZE_CAVEAT,
    }

    if compute_spread_ci:
        summary["spread_bootstrap_ci"] = compute_signer_spread_bootstrap_ci(
            per_signer_acc,
            n_bootstrap=n_bootstrap,
            ci_level=ci_level,
            seed=seed,
        )

    if sign_names is not None:
        # Only run if at least one signer has high_risk_clip_fraction populated
        any_populated = any(
            v.get("high_risk_clip_fraction") is not None
            for v in per_signer_acc["per_signer"].values()
        )
        if any_populated:
            summary["high_risk_correlation"] = compute_signer_high_risk_correlation(
                per_signer_acc,
            )
        else:
            logger.warning(
                "compute_signer_analysis_summary(): sign_names was supplied but "
                "no signer has a populated high_risk_clip_fraction "
                "(likely because high_risk_class_indices was empty — check the "
                "M4 warning above). high_risk_correlation step skipped.",
                extra={"stage": "evaluation"},
            )

    if detected_frame_count is not None or missing_pct is not None:
        summary["failure_mode_summary"] = compute_signer_failure_mode_summary(
            y_true, y_pred, signer_ids,
            detected_frame_count=detected_frame_count,
            missing_pct=missing_pct,
        )

    logger.info(
        f"compute_signer_analysis_summary() COMPLETE | split='{split_name}' | "
        f"n_signers={summary['n_signers']} | "
        f"has_spread_ci={compute_spread_ci} | "
        f"has_high_risk_correlation={'high_risk_correlation' in summary} | "
        f"has_failure_mode_summary={'failure_mode_summary' in summary}",
        extra={"stage": "evaluation"},
    )

    return summary


# ---------------------------------------------------------------------------
# Figure rendering
# ---------------------------------------------------------------------------

def _get_safe_matplotlib():
    """
    Import matplotlib.pyplot safely without unconditionally mutating global
    backend state. Identical helper to calibration.py::_get_safe_matplotlib()
    — duplicated here (not imported) so this module remains independently
    importable. Does NOT call matplotlib.use("Agg") globally (Review L4).
    """
    try:
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        import matplotlib
        if not matplotlib.get_backend():
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt


def plot_signer_generalisation(
    per_signer_result: Dict[str, Any],
    test_per_signer_result: Optional[Dict[str, Any]] = None,
    output_path=None,
    figure_dpi: int = 150,
    metric: str = "accuracy",
    show_wilson_ci: bool = True,
    show_overall_accuracy: bool = True,
) -> Any:
    """
    Render a strip plot of per-signer accuracy (or macro-F1), from the dict(s)
    returned by ``compute_per_signer_accuracy()``.

    Visual design
    -------------
    - One panel for val (always present). A second panel for test is added
      only if ``test_per_signer_result`` is supplied.
    - Each signer is one point, sized by ``n_clips`` (more clips → larger),
      capped at a maximum marker size to prevent large-signer visual dominance
      (Review L3), and coloured grey if ``is_sparse`` else steelblue.
    - Wilson score CI error bars per signer (when show_wilson_ci=True and
      metric="accuracy") — quantified uncertainty rather than bare points.
    - A horizontal dashed line marks the pooled overall accuracy.
    - NO "familiar" / "novel" colour coding — per the Stage 6 (Revised)
      explicit correction, every point represents an equally-unseen signer.
    - Legend uses explicit proxy artists to prevent duplicate entries (Review L4).
    - Sort order: by metric value ascending for readability, with a stable
      secondary sort on signer_id string for deterministic tie-breaking (Review L5).
    - Title says "Per-signer accuracy variation" (not "generalisation") to
      avoid implying a familiar/novel axis that doesn't exist (Review L2).

    Parameters
    ----------
    per_signer_result        : dict from compute_per_signer_accuracy() — val.
    test_per_signer_result     : dict — test split, optional second panel.
    output_path                  : str | Path | None
    figure_dpi                    : int, default 150
    metric                          : str, default "accuracy"
        Either "accuracy" or "macro_f1_global" or "macro_f1_present_classes".
    show_wilson_ci                   : bool, default True
        Show Wilson score CI error bars (only meaningful for metric="accuracy").
    show_overall_accuracy             : bool, default True
        Draw a horizontal dashed line at the pooled overall accuracy.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If ``metric`` is not a valid key in the per_signer entries.
    ImportError
        If matplotlib is not installed.
    """
    valid_metrics = ("accuracy", "macro_f1_global", "macro_f1_present_classes")
    if metric not in valid_metrics:
        raise ValueError(
            f"plot_signer_generalisation(): metric={metric!r} must be one of "
            f"{valid_metrics}."
        )

    try:
        import matplotlib.patches as mpatches
        import matplotlib.lines as mlines
        plt = _get_safe_matplotlib()
    except ImportError as exc:
        raise ImportError(
            "plot_signer_generalisation() requires matplotlib. "
            "Install with: pip install matplotlib"
        ) from exc

    results = [("val", per_signer_result)]
    if test_per_signer_result is not None:
        results.append(("test", test_per_signer_result))

    n_panels = len(results)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6), squeeze=False)
    axes_flat = axes[0]

    metric_label = {
        "accuracy":                  "Accuracy",
        "macro_f1_global":           "Macro-F1 (global, all classes)",
        "macro_f1_present_classes":  "Macro-F1 (present classes only)",
    }[metric]

    # Maximum marker size to prevent visual dominance of high-clip signers (Review L3)
    _MAX_MARKER_SIZE = 300

    for ax, (split_label, result) in zip(axes_flat, results):
        per_signer = result["per_signer"]
        overall    = result.get("overall_accuracy")

        signer_ids_list = list(per_signer.keys())
        values     = [per_signer[sid].get(metric, 0.0) for sid in signer_ids_list]
        n_clips    = [per_signer[sid]["n_clips"] for sid in signer_ids_list]
        is_sparse  = [per_signer[sid]["is_sparse"] for sid in signer_ids_list]

        # Wilson CI only available for accuracy metric
        has_ci = (
            show_wilson_ci
            and metric == "accuracy"
            and "accuracy_ci_lower" in per_signer[signer_ids_list[0]]
        ) if signer_ids_list else False

        ci_lowers = [per_signer[sid].get("accuracy_ci_lower", 0.0) for sid in signer_ids_list]
        ci_uppers = [per_signer[sid].get("accuracy_ci_upper", 0.0) for sid in signer_ids_list]

        # Stable sort: primary by value ascending, secondary by signer_id (Review L5)
        order = sorted(
            range(len(values)),
            key=lambda i: (values[i], str(signer_ids_list[i])),
        )
        signer_ids_sorted = [signer_ids_list[i] for i in order]
        values_sorted     = [values[i] for i in order]
        n_clips_sorted    = [n_clips[i] for i in order]
        is_sparse_sorted  = [is_sparse[i] for i in order]
        ci_lo_sorted      = [ci_lowers[i] for i in order]
        ci_hi_sorted      = [ci_uppers[i] for i in order]

        x_positions = np.arange(len(signer_ids_sorted))

        # Marker size: proportional to n_clips, capped at _MAX_MARKER_SIZE (Review L3)
        sizes  = [min(40 + 20 * c, _MAX_MARKER_SIZE) for c in n_clips_sorted]
        colors = ["darkgrey" if sp else "steelblue" for sp in is_sparse_sorted]

        sc = ax.scatter(
            x_positions, values_sorted, s=sizes, c=colors,
            alpha=0.85, edgecolor="white", linewidth=1.0, zorder=3,
        )

        # Wilson CI error bars (Review M7)
        if has_ci:
            for x, v, ci_lo, ci_hi, sp in zip(
                x_positions, values_sorted, ci_lo_sorted, ci_hi_sorted, is_sparse_sorted
            ):
                yerr_lo = max(0.0, v - ci_lo)
                yerr_hi = max(0.0, ci_hi - v)
                ax.errorbar(
                    x, v,
                    yerr=[[yerr_lo], [yerr_hi]],
                    fmt="none",
                    color="darkgrey" if sp else "steelblue",
                    linewidth=1.2,
                    capsize=3,
                    capthick=1.0,
                    alpha=0.6,
                    zorder=2,
                )

        # Clip count annotations
        for x, v, c in zip(x_positions, values_sorted, n_clips_sorted):
            ax.annotate(
                f"n={c}",
                (x, v),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=7.5,
                color="dimgrey",
            )

        # Pooled accuracy line
        if show_overall_accuracy and overall is not None and metric == "accuracy":
            ax.axhline(
                overall,
                color="tomato",
                linestyle="--",
                linewidth=1.3,
                zorder=2,
                label=f"_pooled",  # underscore prefix prevents auto-legend entry
            )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            signer_ids_sorted, rotation=45, ha="right", fontsize=8
        )
        ax.set_ylim(-0.02, 1.05)
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_xlabel("Signer ID", fontsize=10)
        # Title avoids "generalisation" language (Review L2)
        ax.set_title(
            f"{split_label} split — per-signer accuracy variation\n"
            f"{len(signer_ids_sorted)} unseen signers "
            f"(n={result['n_samples']} clips)",
            fontsize=10,
        )
        ax.grid(True, axis="y", alpha=0.25, linestyle=":")

        # Build legend with explicit proxy artists to prevent duplicates (Review L4)
        legend_handles = [
            mpatches.Patch(color="steelblue", alpha=0.85, label="Signer (adequate clips)"),
            mpatches.Patch(
                color="darkgrey", alpha=0.85,
                label=f"Sparse signer (n < {SPARSE_SIGNER_THRESHOLD})",
            ),
        ]
        if show_overall_accuracy and overall is not None and metric == "accuracy":
            legend_handles.append(
                mlines.Line2D(
                    [0], [0], color="tomato", linestyle="--", linewidth=1.3,
                    label=f"Pooled accuracy={overall:.3f}",
                ),
            )
        if has_ci:
            legend_handles.append(
                mlines.Line2D(
                    [0], [0], color="steelblue", linewidth=1.2, alpha=0.6,
                    label=f"Wilson {int(_WILSON_CI_LEVEL*100)}% CI (per signer)",
                ),
            )

        ax.legend(handles=legend_handles, fontsize=8, loc="lower right", framealpha=0.85)

    fig.suptitle(
        "Per-signer variation (unseen signers only — zero-overlap split;\n"
        "no familiar/novel axis exists in this evaluation)",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=figure_dpi, bbox_inches="tight")
        logger.info(
            f"Signer generalisation plot saved → {out.resolve()}",
            extra={"stage": "evaluation"},
        )

    return fig


# ---------------------------------------------------------------------------
# Import-time self-check
# ---------------------------------------------------------------------------

def _self_check() -> None:
    """
    Cheap, dependency-free sanity check on module constants, mirroring the
    pattern used in metrics.py / calibration.py / benchmark.py.
    """
    assert SPARSE_SIGNER_THRESHOLD >= 1, (
        f"signer_analysis.py: SPARSE_SIGNER_THRESHOLD={SPARSE_SIGNER_THRESHOLD} "
        "must be >= 1."
    )
    assert 0.0 < DEFAULT_BOOTSTRAP_CI < 1.0, (
        f"signer_analysis.py: DEFAULT_BOOTSTRAP_CI={DEFAULT_BOOTSTRAP_CI} "
        "must be in (0, 1)."
    )
    assert DEFAULT_N_BOOTSTRAP >= _MIN_BOOTSTRAP_FOR_STABLE_CI, (
        f"signer_analysis.py: DEFAULT_N_BOOTSTRAP={DEFAULT_N_BOOTSTRAP} should "
        f">= _MIN_BOOTSTRAP_FOR_STABLE_CI={_MIN_BOOTSTRAP_FOR_STABLE_CI}."
    )
    assert 0.0 < _WILSON_CI_LEVEL < 1.0, (
        f"signer_analysis.py: _WILSON_CI_LEVEL={_WILSON_CI_LEVEL} must be in (0, 1)."
    )
    assert _WILSON_Z > 0.0, (
        f"signer_analysis.py: _WILSON_Z={_WILSON_Z} must be positive."
    )
    assert len(HIGH_RISK_SIGNS) == 5, (
        f"signer_analysis.py: HIGH_RISK_SIGNS has {len(HIGH_RISK_SIGNS)} entries; "
        "expected the 5 Stage 5 Finding 8 classes (clothes, think, birthday, "
        "name, book). Keep in sync with metrics.py::HIGH_RISK_SIGNS."
    )
    assert len(UNSEEN_SIGNER_FRAMING_NOTE) > 50, (
        "signer_analysis.py: UNSEEN_SIGNER_FRAMING_NOTE string unexpectedly short."
    )
    assert len(_SIGNER_SAMPLE_SIZE_CAVEAT) > 50, (
        "signer_analysis.py: _SIGNER_SAMPLE_SIZE_CAVEAT string unexpectedly short."
    )
    assert len(_DUAL_MACRO_F1_NOTE) > 50, (
        "signer_analysis.py: _DUAL_MACRO_F1_NOTE string unexpectedly short."
    )
    # Verify _signer_key differentiates int and string signer IDs (Review M5)
    assert _signer_key(1) != _signer_key("1"), (
        "signer_analysis.py: _signer_key() must produce different outputs for "
        "integer 1 and string '1'. repr()-based approach ensures collision safety."
    )


if __debug__:
    _self_check()


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------

__all__ = [
    "SPARSE_SIGNER_THRESHOLD",
    "DEFAULT_N_BOOTSTRAP",
    "DEFAULT_BOOTSTRAP_CI",
    "DEFAULT_SEED",
    "HIGH_RISK_SIGNS",
    "UNSEEN_SIGNER_FRAMING_NOTE",
    "compute_per_signer_accuracy",
    "compute_signer_spread_bootstrap_ci",
    "compute_signer_high_risk_correlation",
    "compute_signer_failure_mode_summary",
    "compute_signer_analysis_summary",
    "plot_signer_generalisation",
]