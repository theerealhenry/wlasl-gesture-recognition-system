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
--------------------------------------------------------------------
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
val and test signer is, by construction, equally "unseen" — there is no
subset of signers the model has ever seen sign anything. This module
therefore:

  - NEVER computes or reports a "familiar" vs "novel" split.
  - Always labels signer-level results as "unseen-signer accuracy", not
    "generalisation to new signers" framed as if some baseline of
    "known signers" existed for comparison.
  - Surfaces the n_signers / clips-per-signer ratio prominently, because
    that ratio (not familiarity) is what determines how trustworthy any
    individual signer's accuracy estimate is.

Why per-signer estimates are extremely noisy (and this module says so)
------------------------------------------------------------------------
The val split has 52 clips across 7 signers — roughly 7-8 clips per
signer on average, but Stage 1's signer-aware bin-packing does not
guarantee an even split: some signers may contribute 3 clips, others 12.
A signer with 3 clips has only 4 possible accuracy values (0/3, 1/3, 2/3,
3/3) — far too coarse to support any claim stronger than "this is a rough
indicator." This module:

  - Reports ``n_clips`` alongside every per-signer accuracy, never as an
    afterthought.
  - Flags signers with ``n_clips < SPARSE_SIGNER_THRESHOLD`` as
    ``is_sparse`` (default threshold: 5), mirroring the ``is_sparse`` /
    ``SPARSE_BIN_THRESHOLD`` pattern already established in
    ``calibration.py``.
  - Computes a bootstrap CI for cross-signer accuracy SPREAD (not just a
    point estimate of each signer's accuracy) so "how much do signers
    differ" is a quantified, reproducible number rather than an eyeballed
    box-plot impression.
  - Embeds a caveat string in every returned dict, exactly mirroring the
    ``_CALIBRATION_CAVEAT`` / ``_BOOTSTRAP_SIGNER_CAVEAT`` pattern in
    ``calibration.py`` / ``metrics.py``, so every downstream consumer
    (``evaluation_report.json``, ``LIMITATIONS.md``, the Stage 11 report)
    inherits the small-sample warning automatically.

Relationship to metrics.py's bootstrap caveat
------------------------------------------------
``metrics.py::bootstrap_macro_f1_ci()`` already documents (Revision history
item 1) that its clip-level bootstrap is NOT signer-aware and likely
UNDERSTATES true uncertainty because validation clips cluster by signer.
That module explicitly defers a signer-stratified treatment to this one.
This module is the natural home for that signer-grouped view, but it does
NOT implement a signer-stratified block-bootstrap replacement for
``bootstrap_macro_f1_ci()`` — that would conflate two different questions
("what is the CI on overall macro-F1, accounting for signer clustering?"
vs "how much does per-signer accuracy vary across signers, unconditional
on overall macro-F1?"). This module answers the second question only.
``compute_signer_spread_bootstrap_ci()`` here resamples SIGNERS (not
clips) to quantify how much the spread of per-signer accuracy would change
under a different random sample of signers — which is the complementary,
signer-level diagnostic ``metrics.py`` defers to this module.

Failure-mode hypothesis testing (Stage 6 Revised, Phase D3 + D4)
--------------------------------------------------------------------
The Stage 6 (Revised) plan asks two specific questions of this module's
output:

    "Cross-check: do low-accuracy signers disproportionately produce the
    known high-risk classes (think, clothes, birthday, name, book)?"

    (Phase D4) "Use the cache's detected_frame_count / missing_pct fields
    to test whether errors cluster in heavily zero-filled or short clips."

``compute_signer_high_risk_correlation()`` answers the first question
directly: for each signer, what fraction of their clips are one of the
five Stage 5 Finding 8 high-risk classes, and does that fraction correlate
with the signer's accuracy? ``compute_per_signer_accuracy()`` accepts
optional ``detected_frame_count`` / ``missing_pct`` metadata (mirroring
the optional ``metadata`` passthrough already established in
``metrics.py::compute_evaluation_summary()``) so the second question can
be answered without a second data-joining step.

Design decisions worth flagging explicitly
---------------------------------------------
Framework-agnostic by construction
    Identical principle to ``metrics.py`` / ``calibration.py``: every
    function here operates on plain numpy arrays already extracted by the
    caller. No model, dataset, or TensorFlow dependency anywhere in this
    module — confirmed independently testable with synthetic arrays.

Per-signer accuracy AND per-signer macro-F1 are both computed
    The handoff (Part 6.4) asks only for per-signer accuracy. This module
    also computes per-signer macro-F1 (forced over all n_classes, the same
    ``labels=list(range(n_classes))`` discipline as every other metric in
    this project — Part 8, Critical Rule #2) because a signer's 7-8 clips
    typically touch only a handful of distinct signs; accuracy alone can
    look deceptively high or low depending on which few classes that
    signer happens to have signed. Per-signer macro-F1 is reported
    alongside accuracy, never as a replacement for it, since macro-F1 over
    a 7-8 clip, ~5-distinct-class subset is itself a very coarse number —
    both are presented together with their respective caveats so neither
    is over-read in isolation.

Signer IDs are treated as opaque labels, never assumed numeric or ordered
    ``signer_ids`` may be strings (e.g. "signer_014") or integers depending
    on how the Phase B1 cache joins ``data/splits/val.csv``. Every function
    here accepts ``Sequence[Any]`` for signer ids and converts to a numpy
    object array internally rather than assuming a numeric dtype — this
    avoids a silent ``astype(int)`` truncation bug if signer IDs are ever
    zero-padded strings (the Stage 1 handoff explicitly notes a previous
    leading-zero video_id bug; signer IDs are not assumed immune to the
    same class of issue).

No I/O, anywhere
    Like ``metrics.py`` and ``calibration.py``, this module performs zero
    file I/O. ``pipelines/run_evaluation.py`` is responsible for loading
    ``val_predictions.npz`` / ``test_predictions.npz`` and joining
    ``signer_ids`` from ``data/splits/{val,test}.csv`` before calling into
    this module.

Module-level exports
----------------------
    compute_per_signer_accuracy           — per-signer accuracy + macro-F1
                                              + support, exactly the
                                              handoff-specified function
    compute_signer_spread_bootstrap_ci    — bootstrap CI on cross-signer
                                              accuracy spread (std / range)
    compute_signer_high_risk_correlation  — does low accuracy correlate
                                              with high-risk class exposure?
    compute_signer_failure_mode_summary   — optional detected_frame_count /
                                              missing_pct correlation (Phase D4)
    compute_signer_analysis_summary       — consolidation wrapper for
                                              evaluation_report.json
    plot_signer_generalisation            — figure renderer (box/strip plot)
    SPARSE_SIGNER_THRESHOLD               — public constant: signers with
                                              fewer clips than this are
                                              flagged sparse
    UNSEEN_SIGNER_FRAMING_NOTE            — the "no familiar/novel axis"
                                              caveat, ready to embed verbatim
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

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
#: below this threshold has too few clips for its accuracy to be read as
#: more than a rough indicator.
SPARSE_SIGNER_THRESHOLD: int = 5

#: Project global seed. Matches DEFAULT_SEED in metrics.py / calibration.py
#: and base.yaml's top-level `seed: 42`.
DEFAULT_SEED: int = 42

#: Default bootstrap resample count for signer-spread CI. Matches the
#: project-wide convention in metrics.py (DEFAULT_N_BOOTSTRAP).
DEFAULT_N_BOOTSTRAP: int = 1000

#: Default confidence level for the signer-spread bootstrap, matching
#: metrics.py's DEFAULT_BOOTSTRAP_CI (90%, not the more conventional 95%,
#: because a 95% interval on 7 signers would be too wide to be informative).
DEFAULT_BOOTSTRAP_CI: float = 0.90

#: Below this many distinct signers, resampling SIGNERS (not clips) for a
#: spread CI is close to meaningless — warn rather than error, since 7
#: signers (this project's actual val signer count) is a legitimate,
#: expected call and must not raise.
_MIN_SIGNERS_FOR_BOOTSTRAP: int = 3

#: Below this many bootstrap resamples, percentile CI bounds are unreliable.
_MIN_BOOTSTRAP_FOR_STABLE_CI: int = 100

#: The five smallest, most failure-prone training classes identified in
#: Stage 5 Finding 8. Matches metrics.py::HIGH_RISK_SIGNS exactly — kept
#: as an independent module-level constant (not imported from metrics.py)
#: so this module remains independently importable/testable without a
#: hard dependency on metrics.py, mirroring the accepted trade-off already
#: documented in calibration.py (item 11 in its revision history).
HIGH_RISK_SIGNS: Tuple[str, ...] = ("clothes", "think", "birthday", "name", "book")

#: Embedded verbatim in every per-signer / summary result so every
#: downstream consumer (evaluation_report.json, LIMITATIONS.md, the
#: Stage 11 report) inherits the correct framing automatically, without
#: depending on someone reading this module's docstring.
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
#: pattern already established by _CALIBRATION_CAVEAT in calibration.py.
_SIGNER_SAMPLE_SIZE_CAVEAT: str = (
    "Per-signer accuracy and macro-F1 estimates are based on a small number "
    "of clips per signer (this project: ~7-8 clips/signer on average across "
    "7 validation signers, with uneven distribution likely). A signer with "
    "n_clips=3 has only 4 possible accuracy values (0/3, 1/3, 2/3, 3/3) — "
    "treat any individual signer's accuracy as a rough indicator, not a "
    "precise estimate. Signers flagged is_sparse (n_clips < "
    f"{SPARSE_SIGNER_THRESHOLD}) should be interpreted with particular caution."
)


# ---------------------------------------------------------------------------
# Internal validation helpers
# (Intentionally self-contained, mirroring calibration.py item 11 —
#  keeps this module independently importable/testable without a hard
#  dependency on metrics.py.)
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

    Identical contract to the equivalent helper in metrics.py / calibration.py:
    rejects multi-dimensional (e.g. one-hot) label arrays rather than
    silently flattening them.

    Raises
    ------
    ValueError
        If ``arr`` is empty or has a shape that suggests one-hot encoding.
    """
    out = np.asarray(arr)
    if out.size == 0:
        raise ValueError(f"{name} is empty. Cannot compute signer metrics on zero samples.")
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
    ``int`` here would silently and incorrectly truncate a zero-padded
    string ID, echoing the exact class of bug Stage 1 already had to fix
    once for video_ids (handoff Part 6.1, "Fixed Bugs" table).

    Parameters
    ----------
    signer_ids : array-like
    caller     : str — used only in error messages.

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
    Compute per-signer accuracy, macro-F1, and support for every distinct
    signer in ``signer_ids``.

    Matches the function name and signature called for in the Stage 6
    (Revised) plan Phase A4: ``compute_per_signer_accuracy() — operates on
    (y_true, y_pred, signer_ids)``. Also satisfies the original handoff's
    Part 6.4 specification ("group val/test clips by signer_id; for each
    signer: n_clips, n_correct, accuracy") while extending it with
    macro-F1, high-risk-class exposure, and explicit small-sample flags —
    see module docstring "Per-signer accuracy AND per-signer macro-F1".

    For each distinct signer ``s``:
      - ``n_clips``               : total clips from this signer.
      - ``n_correct``              : count where argmax prediction matches
                                      the true label.
      - ``accuracy``                : n_correct / n_clips.
      - ``macro_f1``                 : sklearn macro-F1 over this signer's
                                       clips, with ``labels=list(range(n_classes))``
                                       forced (Part 8, Critical Rule #2 — the
                                       same all-classes-forced discipline used
                                       everywhere else in this project). Because
                                       a single signer typically only signs a
                                       handful of distinct classes, this number
                                       is heavily zero-inflated by construction
                                       and should be read alongside
                                       ``n_distinct_true_classes``, not in
                                       isolation.
      - ``n_distinct_true_classes``  : how many distinct true classes this
                                       signer's clips cover — context for
                                       interpreting this signer's macro_f1.
      - ``is_sparse``                : True if n_clips < SPARSE_SIGNER_THRESHOLD.
      - ``high_risk_clip_fraction``  : fraction of this signer's clips whose
                                       true label is one of HIGH_RISK_SIGNS
                                       (only computed if ``sign_names`` is
                                       supplied; otherwise None — see
                                       ``compute_signer_high_risk_correlation``
                                       for the dedicated cross-check this
                                       feeds into).

    Parameters
    ----------
    y_true, y_pred : array-like, shape (n_samples,)
        Already-extracted true labels and argmax predictions — one entry
        per clip. NOT one-hot. This function performs zero inference.
    signer_ids     : array-like, shape (n_samples,)
        One signer identifier per clip, index-aligned with y_true/y_pred.
        May be int, str, or any hashable type — see
        ``_to_signer_id_array()`` docstring for why this is never coerced
        to a numeric dtype.
    n_classes       : int — 35 for this project.
    split_name       : str, default "val" — carried into the output dict
                       and log line only.
    sign_names        : Sequence[str], length n_classes, optional
        If supplied, enables ``high_risk_clip_fraction`` per signer (which
        class index each clip's true label corresponds to is otherwise
        unrecoverable without the name mapping). Omit to skip this field
        (it will be ``None`` for every signer).
    high_risk_signs    : Sequence[str], optional
        Defaults to module-level ``HIGH_RISK_SIGNS`` (the 5 Stage 5
        Finding 8 classes). Only used if ``sign_names`` is also supplied.

    Returns
    -------
    dict with keys:
        split_name              : str
        n_samples                : int
        n_classes                : int
        n_signers                : int
        per_signer                : dict[signer_id_str -> {n_clips, n_correct,
                                    accuracy, macro_f1, n_distinct_true_classes,
                                    is_sparse, high_risk_clip_fraction}]
        n_sparse_signers           : int — count of is_sparse signers
        mean_clips_per_signer       : float
        min_clips_per_signer        : int
        max_clips_per_signer        : int
        overall_accuracy             : float — same n_samples denominator,
                                       sanity-check value (should equal
                                       sklearn accuracy_score(y_true, y_pred))
        unseen_signer_framing_note    : str — see UNSEEN_SIGNER_FRAMING_NOTE
        caveat                        : str — see _SIGNER_SAMPLE_SIZE_CAVEAT

    Raises
    ------
    ValueError
        If arrays are empty, mismatched in length, contain out-of-range
        labels, or ``sign_names`` is supplied with the wrong length.
    """
    _validate_class_count(n_classes, "compute_per_signer_accuracy")

    y_true_arr = _to_label_array(y_true, "y_true")
    y_pred_arr = _to_label_array(y_pred, "y_pred")
    _validate_equal_length(y_true_arr, y_pred_arr, "y_true", "y_pred")
    _validate_label_range(y_true_arr, n_classes, "y_true", "compute_per_signer_accuracy")
    _validate_label_range(y_pred_arr, n_classes, "y_pred", "compute_per_signer_accuracy")

    signer_arr = _to_signer_id_array(signer_ids, "compute_per_signer_accuracy")
    _validate_equal_length(y_true_arr, signer_arr, "y_true", "signer_ids")

    if sign_names is not None and len(sign_names) != n_classes:
        raise ValueError(
            f"compute_per_signer_accuracy(): len(sign_names)={len(sign_names)} "
            f"must equal n_classes={n_classes} when sign_names is supplied."
        )

    high_risk = set(high_risk_signs) if high_risk_signs is not None else set(HIGH_RISK_SIGNS)
    high_risk_class_indices: Optional[set] = None
    if sign_names is not None:
        high_risk_class_indices = {
            idx for idx, name in enumerate(sign_names) if name in high_risk
        }

    n_samples    = len(y_true_arr)
    labels_range = list(range(n_classes))
    correct      = (y_true_arr == y_pred_arr)

    unique_signers = np.unique(signer_arr)
    per_signer: Dict[str, Dict[str, Any]] = {}
    clip_counts: List[int] = []

    for signer in unique_signers:
        mask = (signer_arr == signer)
        n_clips_s = int(mask.sum())
        clip_counts.append(n_clips_s)

        yt_s = y_true_arr[mask]
        yp_s = y_pred_arr[mask]
        n_correct_s = int(correct[mask].sum())
        accuracy_s  = n_correct_s / n_clips_s if n_clips_s > 0 else 0.0

        macro_f1_s = float(_sklearn_f1_score(
            yt_s, yp_s,
            average="macro",
            labels=labels_range,
            zero_division=0,
        ))

        n_distinct_s = int(np.unique(yt_s).size)
        is_sparse_s  = n_clips_s < SPARSE_SIGNER_THRESHOLD

        high_risk_fraction_s: Optional[float] = None
        if high_risk_class_indices is not None:
            n_high_risk_clips = int(np.isin(yt_s, list(high_risk_class_indices)).sum())
            high_risk_fraction_s = n_high_risk_clips / n_clips_s if n_clips_s > 0 else 0.0

        # Signer IDs may be numpy scalar types (e.g. np.str_, np.int64) —
        # cast to a plain Python str for a JSON-serialisable, stable dict key.
        signer_key = str(signer)

        per_signer[signer_key] = {
            "n_clips":                  n_clips_s,
            "n_correct":                n_correct_s,
            "accuracy":                 round(accuracy_s, 6),
            "macro_f1":                 round(macro_f1_s, 6),
            "n_distinct_true_classes":  n_distinct_s,
            "is_sparse":                is_sparse_s,
            "high_risk_clip_fraction":  (
                round(high_risk_fraction_s, 6) if high_risk_fraction_s is not None else None
            ),
        }

    n_signers        = len(unique_signers)
    n_sparse_signers = sum(1 for v in per_signer.values() if v["is_sparse"])
    overall_accuracy = float(correct.mean())

    result: Dict[str, Any] = {
        "split_name":                 split_name,
        "n_samples":                  n_samples,
        "n_classes":                  n_classes,
        "n_signers":                  n_signers,
        "per_signer":                 per_signer,
        "n_sparse_signers":            n_sparse_signers,
        "mean_clips_per_signer":       round(float(np.mean(clip_counts)), 4),
        "min_clips_per_signer":        int(np.min(clip_counts)),
        "max_clips_per_signer":        int(np.max(clip_counts)),
        "overall_accuracy":            round(overall_accuracy, 6),
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
            "accuracy/macro_f1 should be treated as rough indicators only — "
            "see result['caveat'].",
            extra={"stage": "evaluation"},
        )

    return result


# ---------------------------------------------------------------------------
# Signer-level spread bootstrap (complements metrics.py's clip-level bootstrap)
# ---------------------------------------------------------------------------

def compute_signer_spread_bootstrap_ci(
    per_signer_result: Dict[str, Any],
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    ci_level: float = DEFAULT_BOOTSTRAP_CI,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    """
    Quantify how much cross-signer accuracy SPREAD would change under a
    different random sample of signers, via a signer-level bootstrap.

    This is the complementary diagnostic ``metrics.py::bootstrap_macro_f1_ci()``
    explicitly defers to this module (see that function's Revision history
    item 1 and its ``caveat`` field): that function resamples clips and
    likely UNDERSTATES uncertainty because clips cluster by signer. This
    function instead resamples SIGNERS — answering "if we drew a different
    set of 7 unseen signers, how much would the spread of their individual
    accuracies vary?" rather than "what is the CI on the pooled macro-F1?".
    The two are different questions and neither replaces the other.

    Procedure
    ---------
    1. Treat each signer's per-signer accuracy (from
       ``compute_per_signer_accuracy()``) as one observation.
    2. Resample ``n_signers`` signer-accuracies with replacement,
       ``n_bootstrap`` times.
    3. For each resample, compute the spread statistic: standard deviation
       of the resampled accuracies (``ddof=1`` when n>=2, else 0.0) and the
       range (max - min).
    4. Report the empirical percentile interval of both statistics.

    Weighting note
    -----------------
    This function resamples signers UNWEIGHTED by their clip count — a
    signer with 3 clips and a signer with 12 clips are each one observation
    in the resample. This is deliberate: the question being asked is about
    variability ACROSS signers as individuals, not variability across
    clips. A clip-weighted version would just reduce back toward the
    clip-level bootstrap that ``metrics.py`` already provides.

    With only 7 (this project's actual val signer count) input observations,
    bootstrap resampling of signers is itself a coarse instrument — there
    are only a finite number of distinct resamples possible, and percentile
    estimates from such a small base population are inherently approximate.
    This is documented in the returned ``caveat`` field.

    Parameters
    ----------
    per_signer_result : dict
        The dict returned by ``compute_per_signer_accuracy()`` — must
        contain a ``"per_signer"`` key.
    n_bootstrap        : int, default 1000
    ci_level             : float in (0, 1), default 0.90
    seed                 : int, default 42

    Returns
    -------
    dict with keys:
        n_signers                       : int
        observed_std                     : float — std of the actual
                                            (non-resampled) per-signer
                                            accuracies
        observed_range                    : float — max - min of the actual
                                            per-signer accuracies
        std_ci_lower, std_ci_upper          : float
        range_ci_lower, range_ci_upper       : float
        ci_level                              : float
        n_bootstrap                           : int
        seed                                   : int
        caveat                                  : str

    Raises
    ------
    ValueError
        If ``per_signer_result`` lacks a ``"per_signer"`` key, or if
        ``n_bootstrap`` / ``ci_level`` are out of range.
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

    accuracies = np.array(
        [v["accuracy"] for v in per_signer.values()], dtype=np.float64,
    )
    n_signers = len(accuracies)

    if n_signers < _MIN_SIGNERS_FOR_BOOTSTRAP:
        logger.warning(
            f"compute_signer_spread_bootstrap_ci(): n_signers={n_signers} is "
            f"below {_MIN_SIGNERS_FOR_BOOTSTRAP}. Spread statistics will be "
            "extremely unstable. Proceeding anyway, but treat the result as "
            "illustrative only.",
            extra={"stage": "evaluation"},
        )

    if n_bootstrap < _MIN_BOOTSTRAP_FOR_STABLE_CI:
        logger.warning(
            f"compute_signer_spread_bootstrap_ci(): n_bootstrap={n_bootstrap} "
            f"is below {_MIN_BOOTSTRAP_FOR_STABLE_CI}; percentile CI bounds "
            "will be noisy.",
            extra={"stage": "evaluation"},
        )

    observed_std   = float(np.std(accuracies, ddof=1)) if n_signers > 1 else 0.0
    observed_range = float(np.max(accuracies) - np.min(accuracies)) if n_signers > 0 else 0.0

    rng = np.random.default_rng(seed)
    boot_std   = np.empty(n_bootstrap, dtype=np.float64)
    boot_range = np.empty(n_bootstrap, dtype=np.float64)

    for i in range(n_bootstrap):
        idx       = rng.integers(0, n_signers, size=n_signers)
        resampled = accuracies[idx]
        boot_std[i]   = np.std(resampled, ddof=1) if n_signers > 1 else 0.0
        boot_range[i] = np.max(resampled) - np.min(resampled)

    alpha     = 1.0 - ci_level
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    result: Dict[str, Any] = {
        "n_signers":        n_signers,
        "observed_std":      round(observed_std, 6),
        "observed_range":     round(observed_range, 6),
        "std_ci_lower":        round(float(np.percentile(boot_std, lower_pct)), 6),
        "std_ci_upper":         round(float(np.percentile(boot_std, upper_pct)), 6),
        "range_ci_lower":        round(float(np.percentile(boot_range, lower_pct)), 6),
        "range_ci_upper":         round(float(np.percentile(boot_range, upper_pct)), 6),
        "ci_level":                ci_level,
        "n_bootstrap":              n_bootstrap,
        "seed":                      seed,
        "caveat": (
            f"This bootstrap resamples only {n_signers} signer-level "
            "observations (unweighted by clip count). With this few "
            "signers, the resample space is small and percentile estimates "
            "are approximate. This is a signer-level complement to "
            "metrics.py::bootstrap_macro_f1_ci()'s clip-level CI, not a "
            "replacement for it — see this module's docstring for the "
            "distinction between the two questions each answers."
        ),
    }

    logger.info(
        f"compute_signer_spread_bootstrap_ci() | n_signers={n_signers} | "
        f"observed_std={observed_std:.4f} "
        f"[{int(ci_level*100)}% CI: {result['std_ci_lower']:.4f}, {result['std_ci_upper']:.4f}] | "
        f"observed_range={observed_range:.4f} "
        f"[{int(ci_level*100)}% CI: {result['range_ci_lower']:.4f}, {result['range_ci_upper']:.4f}]",
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
    book), per the Stage 6 (Revised) plan's Phase D3 cross-check:

        "Cross-check: do low-accuracy signers disproportionately produce
        the known high-risk classes (think, clothes, birthday, name,
        book)?"

    Requires ``per_signer_result`` to have been produced by
    ``compute_per_signer_accuracy(..., sign_names=...)`` — i.e.
    ``high_risk_clip_fraction`` must be populated (non-None) for every
    signer. This function does NOT have access to the raw label arrays;
    it operates purely on the already-computed per-signer summary,
    consistent with this module's "no I/O, pure computation" design.

    Methodology
    -----------
    Computes the Pearson correlation coefficient between each signer's
    ``accuracy`` and their ``high_risk_clip_fraction`` across all signers.
    A negative correlation is the expected-and-unsurprising direction
    (more high-risk clips → lower accuracy, since high-risk classes are
    failure-prone by construction — Stage 5 Finding 8). This function
    reports the correlation as a DESCRIPTIVE statistic, not a causal claim,
    and explicitly flags when n_signers is too small for the correlation
    to be statistically meaningful (this project: 7 val signers — a
    Pearson r computed over 7 points should be treated as suggestive at
    best, never confirmatory).

    Parameters
    ----------
    per_signer_result : dict
        The dict returned by ``compute_per_signer_accuracy(...,
        sign_names=...)``. Must contain a ``"per_signer"`` key whose
        entries have a non-None ``high_risk_clip_fraction``.

    Returns
    -------
    dict with keys:
        n_signers                        : int
        pearson_r                          : float | None — None if
                                            high_risk_clip_fraction was not
                                            computed (sign_names was omitted
                                            upstream) or if fewer than 2
                                            signers have variance in either
                                            variable.
        interpretation                      : str — plain-language direction
                                            and magnitude descriptor
        is_underpowered                      : bool — True if n_signers < 10
                                            (an arbitrary but explicit floor
                                            below which any correlation
                                            coefficient is unreliable)
        per_signer_table                      : List[dict] — [{signer_id,
                                            accuracy, high_risk_clip_fraction}]
                                            sorted by accuracy ascending, for
                                            direct visual/tabular inspection
                                            alongside the correlation
        caveat                                  : str

    Raises
    ------
    ValueError
        If ``per_signer_result`` lacks a ``"per_signer"`` key, or if no
        signer has a non-None ``high_risk_clip_fraction`` (indicating
        ``compute_per_signer_accuracy()`` was called without
        ``sign_names``).
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
            "accuracy":                 stats["accuracy"],
            "high_risk_clip_fraction":   stats.get("high_risk_clip_fraction"),
        }
        for signer_id, stats in per_signer.items()
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

    acc_vals = np.array([r["accuracy"] for r in valid_rows], dtype=np.float64)
    hr_vals  = np.array([r["high_risk_clip_fraction"] for r in valid_rows], dtype=np.float64)

    pearson_r: Optional[float] = None
    if n_signers >= 2 and np.std(acc_vals) > 0 and np.std(hr_vals) > 0:
        pearson_r = float(np.corrcoef(acc_vals, hr_vals)[0, 1])

    is_underpowered = n_signers < 10

    if pearson_r is None:
        interpretation = (
            "Correlation undefined: fewer than 2 signers, or no variance in "
            "accuracy or high_risk_clip_fraction across signers (e.g. every "
            "signer has the same accuracy, or none of their clips are "
            "high-risk classes)."
        )
    else:
        direction = "negative (lower accuracy ↔ more high-risk-class exposure)" \
            if pearson_r < 0 else \
            "positive (higher accuracy ↔ more high-risk-class exposure — counter to expectation)"
        magnitude = (
            "weak" if abs(pearson_r) < 0.3 else
            "moderate" if abs(pearson_r) < 0.6 else
            "strong"
        )
        interpretation = (
            f"{magnitude.capitalize()} {direction}, r={pearson_r:.3f}."
        )

    sorted_rows = sorted(rows, key=lambda r: r["accuracy"])

    result: Dict[str, Any] = {
        "n_signers":          n_signers,
        "pearson_r":           round(pearson_r, 6) if pearson_r is not None else None,
        "interpretation":        interpretation,
        "is_underpowered":         is_underpowered,
        "per_signer_table":          sorted_rows,
        "caveat": (
            f"Computed over n_signers={n_signers}. Pearson correlation over "
            "this few points is descriptive/exploratory only — treat as "
            "suggestive, not statistically confirmatory. A negative "
            "correlation is the expected direction (Stage 5 Finding 8: "
            "high-risk classes are failure-prone by construction, mostly "
            "due to their own tiny training-clip counts, not the signer's "
            "general skill) and should not be over-interpreted as a finding "
            "about signer quality."
        ),
    }

    logger.info(
        f"compute_signer_high_risk_correlation() | n_signers={n_signers} | "
        f"pearson_r={pearson_r if pearson_r is not None else 'undefined'} | "
        f"is_underpowered={is_underpowered}",
        extra={"stage": "evaluation"},
    )

    if is_underpowered:
        logger.warning(
            f"compute_signer_high_risk_correlation(): n_signers={n_signers} "
            "< 10. This correlation is exploratory only and must not be "
            "reported as a statistically robust finding.",
            extra={"stage": "evaluation"},
        )

    return result


# ---------------------------------------------------------------------------
# Failure-mode metadata correlation (Stage 6 Revised, Phase D4)
# ---------------------------------------------------------------------------

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

    This function answers the SIGNER-conditioned slice of that question:
    for each signer, what is the mean ``detected_frame_count`` /
    ``missing_pct`` among their correctly-classified clips versus their
    misclassified clips? A signer whose errors cluster in heavily
    zero-filled or short clips (rather than being spread evenly) supports
    the "errors cluster in heavily zero-filled or short clips" hypothesis
    being a property of the DATA (specific difficult clips), not of any
    particular signer being generally harder to classify.

    This function does NOT replace a dataset-wide (non-signer-conditioned)
    version of this analysis — that belongs in
    ``src/evaluation/metrics.py`` or a dedicated failure-taxonomy module
    consuming the full prediction cache directly (Stage 6 Revised, Phase D4
    "failure taxonomy ... built inductively from D1's actual misclassified
    clips"). This function exists specifically to let Phase D3's signer
    analysis and Phase D4's failure taxonomy share one signer-level cut of
    the same metadata without a second data-join.

    Parameters
    ----------
    y_true, y_pred         : array-like, shape (n_samples,)
    signer_ids               : array-like, shape (n_samples,)
    detected_frame_count       : array-like, shape (n_samples,), optional
        Number of MediaPipe-detected frames per clip (joined from
        ``landmark_inventory.csv`` into the Phase B1 cache). If omitted,
        this metric is skipped (per-signer entries omit the corresponding
        keys) — this function tolerates partial metadata rather than
        requiring both fields simultaneously.
    missing_pct                 : array-like, shape (n_samples,), optional
        Fraction of both-hands-absent frames per clip. Same omission
        behaviour as ``detected_frame_count``.

    Returns
    -------
    dict with keys:
        n_samples                            : int
        n_signers                              : int
        metadata_fields_provided                : List[str] — subset of
                                                 ["detected_frame_count",
                                                 "missing_pct"] actually supplied
        per_signer                               : dict[signer_id_str -> {
                                                 n_clips, n_correct, n_incorrect,
                                                 mean_detected_frames_correct,
                                                 mean_detected_frames_incorrect,
                                                 mean_missing_pct_correct,
                                                 mean_missing_pct_incorrect,
                                                 }] — mean_* keys are only
                                                 present for fields that were
                                                 supplied, and are None when
                                                 a signer has zero clips in
                                                 the correct/incorrect subset
                                                 needed to compute that mean.
        caveat                                    : str

    Raises
    ------
    ValueError
        If neither ``detected_frame_count`` nor ``missing_pct`` is
        supplied, or via standard array-length validation.
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

    metadata_fields_provided: List[str] = []
    dfc_arr: Optional[np.ndarray] = None
    mp_arr:  Optional[np.ndarray] = None

    if detected_frame_count is not None:
        dfc_arr = np.asarray(detected_frame_count, dtype=np.float64)
        _validate_equal_length(y_true_arr, dfc_arr, "y_true", "detected_frame_count")
        metadata_fields_provided.append("detected_frame_count")

    if missing_pct is not None:
        mp_arr = np.asarray(missing_pct, dtype=np.float64)
        _validate_equal_length(y_true_arr, mp_arr, "y_true", "missing_pct")
        metadata_fields_provided.append("missing_pct")

    correct = (y_true_arr == y_pred_arr)
    unique_signers = np.unique(signer_arr)

    def _safe_mean(values: np.ndarray) -> Optional[float]:
        return round(float(values.mean()), 4) if values.size > 0 else None

    per_signer: Dict[str, Dict[str, Any]] = {}
    for signer in unique_signers:
        mask          = (signer_arr == signer)
        correct_mask  = mask & correct
        incorrect_mask = mask & (~correct)

        entry: Dict[str, Any] = {
            "n_clips":      int(mask.sum()),
            "n_correct":    int(correct_mask.sum()),
            "n_incorrect":  int(incorrect_mask.sum()),
        }

        if dfc_arr is not None:
            entry["mean_detected_frames_correct"]   = _safe_mean(dfc_arr[correct_mask])
            entry["mean_detected_frames_incorrect"] = _safe_mean(dfc_arr[incorrect_mask])

        if mp_arr is not None:
            entry["mean_missing_pct_correct"]    = _safe_mean(mp_arr[correct_mask])
            entry["mean_missing_pct_incorrect"]  = _safe_mean(mp_arr[incorrect_mask])

        per_signer[str(signer)] = entry

    result: Dict[str, Any] = {
        "n_samples":                  len(y_true_arr),
        "n_signers":                  len(unique_signers),
        "metadata_fields_provided":    metadata_fields_provided,
        "per_signer":                  per_signer,
        "caveat": (
            "Per-signer correct/incorrect metadata means are computed over "
            "very small subsets (often 1-6 clips per signer per correctness "
            "bucket). Differences between mean_*_correct and "
            "mean_*_incorrect for any single signer are illustrative, not "
            "statistically tested. Aggregate across all signers (not "
            "signer-by-signer) for any claim about whether errors cluster "
            "in zero-filled or short clips dataset-wide."
        ),
    }

    logger.info(
        f"compute_signer_failure_mode_summary() | "
        f"n_samples={result['n_samples']} | n_signers={result['n_signers']} | "
        f"metadata_fields={metadata_fields_provided}",
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
    signer-analysis summary for one split — the one function Notebook 06
    and ``pipelines/run_evaluation.py`` should call to get "the signer
    numbers" for a split.

    Mirrors the consolidation pattern already established by
    ``metrics.py::compute_evaluation_summary()`` and
    ``calibration.py::compute_calibration_summary()``: validates inputs
    once, then composes the other public functions in this module, so no
    caller has to remember to separately call
    ``compute_per_signer_accuracy``, ``compute_signer_spread_bootstrap_ci``,
    ``compute_signer_high_risk_correlation``, and
    ``compute_signer_failure_mode_summary`` and hand-assemble the result.

    Does NOT run inference itself — callers pass already-extracted
    ``(y_true, y_pred, signer_ids)`` from the Phase B1 / Phase C prediction
    cache, exactly like every other Stage 6 consolidation function.

    Parameters
    ----------
    y_true, y_pred             : array-like, shape (n_samples,)
    signer_ids                  : array-like, shape (n_samples,)
    n_classes                    : int
    split_name                    : str, default "val"
    sign_names                     : Sequence[str], optional — enables
                                    high_risk_clip_fraction and therefore
                                    the high-risk correlation cross-check.
                                    Strongly recommended; omit only if the
                                    label map is unavailable.
    high_risk_signs                 : Sequence[str], optional — see
                                    compute_per_signer_accuracy().
    detected_frame_count               : array-like, optional — enables
                                    compute_signer_failure_mode_summary().
    missing_pct                          : array-like, optional — same.
    compute_spread_ci                      : bool, default True — set False
                                    to skip the signer-level bootstrap
                                    (e.g. for a fast dev iteration).
    n_bootstrap, ci_level, seed              : see
                                    compute_signer_spread_bootstrap_ci().

    Returns
    -------
    dict with keys:
        split_name                    : str
        n_samples, n_classes, n_signers : int
        per_signer_accuracy              : dict (see compute_per_signer_accuracy())
        spread_bootstrap_ci               : dict, only if compute_spread_ci=True
        high_risk_correlation               : dict, only if sign_names was
                                            supplied (see
                                            compute_signer_high_risk_correlation())
        failure_mode_summary                  : dict, only if
                                            detected_frame_count or
                                            missing_pct was supplied
        unseen_signer_framing_note               : str
        caveat                                     : str

    Raises
    ------
    ValueError
        If any constituent validation fails.
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
        "unseen_signer_framing_note":     UNSEEN_SIGNER_FRAMING_NOTE,
        "caveat":                          _SIGNER_SAMPLE_SIZE_CAVEAT,
    }

    if compute_spread_ci:
        summary["spread_bootstrap_ci"] = compute_signer_spread_bootstrap_ci(
            per_signer_acc, n_bootstrap=n_bootstrap, ci_level=ci_level, seed=seed,
        )

    if sign_names is not None:
        summary["high_risk_correlation"] = compute_signer_high_risk_correlation(
            per_signer_acc,
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
    backend state. Identical helper to calibration.py's
    ``_get_safe_matplotlib()`` — duplicated here (not imported) so this
    module remains independently importable, consistent with this file's
    "no hard dependency on sibling Stage 6 modules" design choice.
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
    output_path: Optional[Union[str, Path]] = None,
    figure_dpi: int = 150,
    metric: str = "accuracy",
) -> Any:
    """
    Render a strip/box plot of per-signer accuracy (or macro-F1), from the
    dict(s) returned by ``compute_per_signer_accuracy()``.

    Visual design
    -------------
    - One panel for val (always present). A second panel for test is added
      only if ``test_per_signer_result`` is supplied (Stage 6 Revised plan,
      Phase D3: "optionally 7 test signers as a second panel").
    - Each signer is one point (strip plot), sized by ``n_clips`` (more
      clips → larger, more trustworthy-looking marker) and coloured grey
      if ``is_sparse`` else steelblue, making the small-sample caveat
      visually legible rather than only textual.
    - A horizontal dashed line marks the overall (pooled) accuracy for
      context against the per-signer spread.
    - NO "familiar" / "novel" colour coding or legend category anywhere in
      this figure — per the explicit Stage 6 (Revised) correction, every
      point represents an equally-unseen signer (see
      ``UNSEEN_SIGNER_FRAMING_NOTE``, also printed in the figure caption).

    Backend note
    ------------
    Does NOT call ``matplotlib.use("Agg")`` globally — see
    ``_get_safe_matplotlib()``, matching the convention in calibration.py.

    Parameters
    ----------
    per_signer_result        : dict from compute_per_signer_accuracy() — val split.
    test_per_signer_result     : dict from compute_per_signer_accuracy() —
                                test split, optional second panel.
    output_path                  : str | Path | None
    figure_dpi                    : int
    metric                          : str, default "accuracy"
        Either "accuracy" or "macro_f1" — which per-signer field to plot.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If ``metric`` is not "accuracy" or "macro_f1".
    ImportError
        If matplotlib is not installed.
    """
    if metric not in ("accuracy", "macro_f1"):
        raise ValueError(
            f"plot_signer_generalisation(): metric={metric!r} must be "
            "'accuracy' or 'macro_f1'."
        )

    try:
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
    axes = axes[0]

    metric_label = "Accuracy" if metric == "accuracy" else "Macro-F1"

    for ax, (split_label, result) in zip(axes, results):
        per_signer = result["per_signer"]
        overall    = result["overall_accuracy"] if metric == "accuracy" else None

        signer_ids = list(per_signer.keys())
        values     = [per_signer[sid][metric] for sid in signer_ids]
        n_clips    = [per_signer[sid]["n_clips"] for sid in signer_ids]
        is_sparse  = [per_signer[sid]["is_sparse"] for sid in signer_ids]

        # Sort by value for a readable strip plot.
        order = np.argsort(values)
        signer_ids = [signer_ids[i] for i in order]
        values     = [values[i] for i in order]
        n_clips    = [n_clips[i] for i in order]
        is_sparse  = [is_sparse[i] for i in order]

        x_positions = np.arange(len(signer_ids))
        sizes  = [40 + 25 * c for c in n_clips]
        colors = ["darkgrey" if sp else "steelblue" for sp in is_sparse]

        ax.scatter(x_positions, values, s=sizes, c=colors, alpha=0.85,
                   edgecolor="white", linewidth=1.0, zorder=3)

        for x, v, c in zip(x_positions, values, n_clips):
            ax.annotate(f"n={c}", (x, v), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=7.5, color="dimgrey")

        if overall is not None:
            ax.axhline(overall, color="tomato", linestyle="--", linewidth=1.3,
                       label=f"Pooled {metric_label.lower()}={overall:.3f}", zorder=2)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(signer_ids, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(-0.02, 1.05)
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_xlabel("Signer ID", fontsize=10)
        ax.set_title(
            f"{split_label} split — {len(signer_ids)} unseen signers "
            f"(n={result['n_samples']} clips)",
            fontsize=10,
        )
        ax.grid(True, axis="y", alpha=0.25, linestyle=":")
        ax.legend(fontsize=8, loc="lower right")

        # Sparse-signer legend marker, added once via a proxy artist.
        if any(is_sparse):
            ax.scatter([], [], s=80, c="darkgrey", edgecolor="white",
                      label=f"is_sparse (n_clips < {SPARSE_SIGNER_THRESHOLD})")
            ax.legend(fontsize=8, loc="lower right")

    fig.suptitle(
        "Per-signer generalisation — all signers unseen by construction "
        "(zero-overlap split; no familiar/novel axis)",
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

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
        f"be >= _MIN_BOOTSTRAP_FOR_STABLE_CI={_MIN_BOOTSTRAP_FOR_STABLE_CI}."
    )
    assert len(HIGH_RISK_SIGNS) == 5, (
        f"signer_analysis.py: HIGH_RISK_SIGNS has {len(HIGH_RISK_SIGNS)} "
        "entries; expected the 5 Stage 5 Finding 8 classes (clothes, think, "
        "birthday, name, book). Keep in sync with metrics.py::HIGH_RISK_SIGNS "
        "if this was an intentional update."
    )
    assert len(UNSEEN_SIGNER_FRAMING_NOTE) > 50, (
        "signer_analysis.py: UNSEEN_SIGNER_FRAMING_NOTE string unexpectedly short."
    )
    assert len(_SIGNER_SAMPLE_SIZE_CAVEAT) > 50, (
        "signer_analysis.py: _SIGNER_SAMPLE_SIZE_CAVEAT string unexpectedly short."
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