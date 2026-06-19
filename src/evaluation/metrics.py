"""
src/evaluation/metrics.py
==========================
Core evaluation metric primitives for the WLASL 35-class gesture recognition
system. This is the foundation module of Stage 6 (Evaluation, Benchmarking,
and Interpretability) — every other Stage 6 module (benchmark.py,
calibration.py, signer_analysis.py, the SHAP notebook, run_evaluation.py)
either calls into this module directly or consumes the dicts/arrays it
produces. Nothing downstream re-implements confusion-matrix, F1, or
prediction-extraction logic.

Why this module exists as a separate, deliberately narrow layer
------------------------------------------------------------------
Stages 1-5 already taught two hard lessons about this dataset that this
module is built to never re-violate:

  1. macro-F1, not accuracy, is the project's primary metric (Part 8,
     Critical Rule #2). With 21 of 35 validation classes having exactly one
     supporting clip, accuracy and macro-F1 can diverge sharply, and every
     champion-selection decision across Stage 5's 23 runs was made on
     macro-F1. Every function here that reports a headline number reports
     macro-F1 first.

  2. The validation set is small enough (52 clips) that point estimates are
     close to meaningless on their own. Stage 5's own analysis noted that
     "epoch-to-epoch swings of 3-5pp are structural noise" — but that was an
     informal observation, not a number anyone could act on. This module's
     headline addition, ``bootstrap_macro_f1_ci()``, turns that informal
     observation into a quantified, reproducible interval
     (e.g. macro-F1 = 0.60 [90% CI: 0.52, 0.68]) so that Stage 6 and Stage 11
     report real uncertainty instead of a bare point estimate.

Revision history
-----------------
This is a post-review revision. A peer review of the original Stage 6
draft (preserved in project records) raised twelve points; each was
independently re-verified against this codebase before being accepted,
modified, or rejected. The disposition of every point is recorded here so
the reasoning survives the code change itself:

  1.  ACCEPTED (documented, not "fixed"). The bootstrap in
      ``bootstrap_macro_f1_ci()`` resamples clip indices uniformly and is
      NOT signer-aware. Val clips are not fully independent — they cluster
      by signer (52 clips / 7 signers), and errors correlate within a
      signer. A true fix would require a signer-stratified block bootstrap,
      which needs signer_ids threaded through a function that currently
      only sees (y_true, y_pred). Rather than silently bolt on a
      signer-aware mode here (which `signer_analysis.py` is the right home
      for), this module now states the limitation explicitly in the
      returned dict (`"resampling_unit": "clip"`, plus a `"caveat"` string)
      so every consumer — evaluation_report.json, LIMITATIONS.md, the
      Stage 11 report — inherits the caveat automatically instead of
      depending on someone reading this docstring.
  2.  ACCEPTED. ``macro_avg`` / ``weighted_avg`` now carry a ``support``
      key (== total_support) so a consumer reading only the avg block
      still knows the sample size behind it.
  3.  ACCEPTED. ``get_predictions()`` now raises a clear, actionable error
      if ``return_probs=True`` produces zero probability batches despite
      zero-length label batches being silently skipped — see the
      docstring update there.
  4.  ACCEPTED. Labels are now shape-validated (``ndim <= 1`` after an
      explicit squeeze, with a hard rejection of genuinely 2-D one-hot-style
      label batches) at extraction time, before concatenation, so a
      one-hot-label misconfiguration fails immediately and legibly instead
      of silently producing a wrong-length flattened array.
  5.  ACCEPTED (softened). The bootstrap docstring no longer asserts that
      penalising an absent singleton class is THE statistically "correct"
      behaviour. It is the project's deliberate, documented choice
      (consistent with how every other macro-F1 call in this codebase
      forces `labels=list(range(n_classes))`), not a claim that a
      stratified bootstrap would be wrong. Both are legitimate estimators
      of different things.
  6.  ACCEPTED. ``compute_per_class_metrics()`` now rejects duplicate
      ``sign_names`` outright — silent dict-overwrite-by-name was a real
      and serious data-loss bug (e.g. two classes erroneously sharing a
      label-map entry would silently report only one of them).
  7.  ACCEPTED. ``rank_classes_by_f1()`` now validates that every per-class
      entry actually has an ``f1_score`` key before sorting, since
      ``evaluation_report.json`` is an on-disk artefact that may be
      reloaded by a different process/notebook than the one that wrote it.
  8.  ACCEPTED. ``compute_accuracy()`` now validates label well-formedness
      to the same standard as every other metric in this module
      (non-negative integers; full ``[0, n_classes)`` range check when
      ``n_classes`` is supplied). ``n_classes`` is optional and keyword-only
      to preserve the simple two-argument call sites already in use.
  9.  ACCEPTED. ``compute_evaluation_summary()`` gained an optional
      ``metadata`` passthrough (clip_ids / signer_ids / detected_frame_count
      / missing_pct, per the Phase B1 prediction-cache schema) so failure
      analysis doesn't need to re-join CSVs once this summary is the
      canonical artefact. This module still performs zero I/O — metadata is
      accepted as already-loaded arrays/lists, not file paths.
  10. ACCEPTED (narrow allowance, opt-in). ``get_predictions()`` now
      accepts an ``allow_unbatched`` flag: if a single inference call
      returns a 1-D ``(n_classes,)`` vector (some TFLite interpreter
      wrappers do this for batch=1) it is treated as one sample rather than
      hard-rejected. Default is ``False`` — production Keras inference
      always returns a 2-D batch, so silently reshaping by default would
      mask real shape bugs. Stage 8 / benchmark.py call sites can opt in.
  11. ACCEPTED (light touch). ``compute_evaluation_summary()`` validates
      ``y_true``/``y_pred`` exactly once and passes the already-validated
      arrays into the constituent calls, which avoids three redundant
      re-validations per summary. At WLASL-35 scale (<= 211 clips) this is
      a clarity improvement more than a performance one; it is not
      generalised into a broader "skip validation" mode since the per-call
      guards are what make every function here safely independently
      callable and independently unit-testable.
  12. ACCEPTED. The hard ``assert N_CLASSES == 35`` import imports-time
      self-check has been removed. It directly contradicted this module's
      own design principle (every function takes ``n_classes`` explicitly
      specifically so the module is not hardcoded to 35) and would have
      crashed on import for a future WLASL-50 or KSL label map before a
      single function call. ``N_CLASSES`` remains as a documentation
      constant only; the self-check now verifies internal constant
      consistency (e.g. ``HIGH_RISK_SIGNS`` cardinality, CI-level bounds)
      without asserting a class count that this module explicitly does not
      own. The authoritative, runtime class-count check is — and always
      should be — against ``cfg.num_classes``, mirroring how
      ``architectures.py::_check_n_classes`` handles the equivalent
      situation: log loudly, never crash at import time.

Design decisions worth flagging explicitly
---------------------------------------------
Framework-agnostic by construction
    Every function here operates on plain ``numpy`` arrays. ``get_predictions()``
    is the one function that touches a live model, and even it never imports
    TensorFlow: ``np.asarray(model(x, training=False))`` works directly on a
    TF EagerTensor because EagerTensor implements ``__array__``. This keeps
    the module's hard dependency surface to ``numpy`` + ``scikit-learn`` only,
    which means:
      (a) every pure-computation function here is trivially unit-testable
          with synthetic arrays and a fake callable model — no TensorFlow
          installation required in the test environment, and
      (b) this module can be reused unmodified if the project ever evaluates
          a non-Keras model (e.g. the Stage 8 TFLite interpreter — see
          benchmark.py) without modification, since "a callable that returns
          something array-like" is the only contract `get_predictions()`
          imposes on `model`.

``labels=list(range(n_classes))`` everywhere, non-negotiably
    Every sklearn metric call in this module passes an explicit ``labels``
    list spanning all ``n_classes`` and ``zero_division=0``. Without this,
    sklearn silently restricts its output to whichever classes happen to
    appear in a given batch of predictions — which, with 21 singleton
    validation classes, would silently produce a *different* (and inflated)
    macro-F1 every time a rare class happened to be absent or misclassified
    out of the prediction set. This is the single most consequential
    correctness detail in this module.

Per-class outputs always carry support + small-sample flags
    ``compute_per_class_metrics()`` never returns a bare F1 number for a
    class without also returning that class's ``support`` and an
    ``is_singleton`` / ``is_zero_support`` flag. A perfect F1=1.0 on a class
    with one validation clip is one lucky guess, not a learned pattern, and
    no downstream chart or table should be able to present it without that
    context attached. ``is_high_risk`` similarly flags the five classes
    (``clothes``, ``think``, ``birthday``, ``name``, ``book``) that Stage 5's
    Finding 8 already identified as small-sample failure-prone.

Clean schema, not raw sklearn passthrough
    ``sklearn.metrics.classification_report(output_dict=True)`` returns keys
    like ``"f1-score"`` (hyphenated) and ``"macro avg"`` (spaced) — both
    awkward for downstream JSON consumers and for dict/attribute access.
    This module computes the same underlying numbers but re-keys them into a
    consistent ``snake_case`` schema (``f1_score``, ``macro_avg``,
    ``weighted_avg``) before returning. The underlying sklearn computation is
    unchanged; only the presentation layer is cleaned up.

Bootstrap is clip-level by construction, not an extra step
    ``y_true`` / ``y_pred`` here are already one entry per *clip* (each
    array index is the outcome of one validation clip's prediction, not one
    frame). A standard bootstrap resample of *array indices* is therefore
    automatically a clip-level resample — there is no separate "frame vs
    clip" branch to implement. It is NOT, however, signer-level — see
    Revision history item 1 above and the ``resampling_unit`` /
    ``caveat`` fields now returned by ``bootstrap_macro_f1_ci()``.

Module-level exports
---------------------
    get_predictions             — run inference over a dataset, return arrays
    get_val_predictions         — thin spec-naming wrapper around the above
    compute_macro_f1            — sklearn macro-F1, all classes forced into the average
    compute_accuracy            — plain accuracy (secondary metric only — see Part 8)
    compute_confusion_matrix    — (model, ds, n_classes) -> (n_classes, n_classes) array
    compute_confusion_matrix_from_predictions — pure (y_true, y_pred) variant
    compute_support_counts      — per-class true-label counts
    compute_per_class_metrics   — enriched, support-flagged classification report
    rank_classes_by_f1          — sort a per_class_metrics dict for charting
    bootstrap_macro_f1_ci       — clip-level bootstrap CI for macro-F1
    compute_evaluation_summary  — single canonical dict bundling all of the above
    HIGH_RISK_SIGNS             — the 5 known small-sample classes from Stage 5
    N_CLASSES                   — documentation constant (35); functions always
                                   take n_classes as an explicit argument
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Documentation constant: number of output classes in this project.
#: Every function in this module takes ``n_classes`` as an explicit argument
#: rather than reading this constant, so the module remains correct if a
#: future WLASL-50 (or KSL, per Stage 11 Q4/Q5) label map changes the class
#: count. NOTE (post-review): this constant is informational only and is
#: deliberately NOT enforced by an import-time assert — see Revision
#: history item 12. The authoritative runtime check is always
#: ``cfg.num_classes``.
N_CLASSES: int = 35

#: The five smallest, most failure-prone training classes identified in
#: Stage 5 (Finding 8 / Part 4 high-risk class table): clothes (2 train
#: clips), think (3 train clips, F1=0.0 in 8/9 champion runs), birthday,
#: name, book. ``compute_per_class_metrics()`` flags these by default so
#: every per-class report — not just the Stage 5 notebook — surfaces them.
#: Callers may override via the ``high_risk_signs`` parameter (e.g. once
#: Stage 6's own confusion-matrix analysis identifies additional
#: failure-prone classes that weren't visible from training metrics alone).
HIGH_RISK_SIGNS: Tuple[str, ...] = ("clothes", "think", "birthday", "name", "book")

#: Default bootstrap resample count for ``bootstrap_macro_f1_ci()``.
#: 1000 resamples of a ~50-clip array completes in well under a second
#: (sklearn's f1_score call overhead dominates, not array size) — there is
#: no performance reason to use fewer, and percentile estimates of the CI
#: bounds get visibly noisier below a few hundred resamples.
DEFAULT_N_BOOTSTRAP: int = 1000

#: Default confidence level. The Stage 6 plan calls for a 90% CI specifically
#: (not the more conventional 95%) because the validation set is small
#: (52 clips) — a 95% CI on this much data would be so wide it communicates
#: little beyond "we don't know"; 90% is the documented project choice.
DEFAULT_BOOTSTRAP_CI: float = 0.90

#: Default seed. Matches the project's global seed (Part 2 project
#: constants table) so that "the bootstrap CI" means the same fixed number
#: every time it is recomputed, not a moving target across notebook re-runs.
DEFAULT_SEED: int = 42

#: Bootstrap resampling becomes statistically meaningless below this many
#: samples (Stage 6 will call this on val=52 clips and test=51 clips, both
#: comfortably above this floor — it exists to catch a misconfigured split,
#: e.g. accidentally passing the 3-clip 'think' class subset alone).
_MIN_SAMPLES_FOR_BOOTSTRAP: int = 5

#: Below this many bootstrap resamples, percentile-based CI bounds are
#: unreliable; warn (not error) since a caller may legitimately want a fast
#: smoke-test run with fewer resamples during development.
_MIN_BOOTSTRAP_FOR_STABLE_CI: int = 100

#: Documented limitation string attached to every bootstrap CI result.
#: Surfaced verbatim in evaluation_report.json and quotable directly into
#: LIMITATIONS.md / the Stage 11 report — see Revision history item 1.
_BOOTSTRAP_SIGNER_CAVEAT: str = (
    "Resampling is class-stratified (each class's own clips are resampled "
    "independently, preserving its original count) — this eliminates the "
    "forced-zero bias from singleton classes dropping out of a resample, "
    "but it is still NOT signer-aware. Within a class's index pool, clips "
    "may still cluster by signer, so this interval likely UNDERSTATES true "
    "uncertainty relative to a signer-stratified block bootstrap. Treat "
    "this interval as a lower bound on uncertainty, not a tight estimate."
)


# ---------------------------------------------------------------------------
# Internal validation helpers
# ---------------------------------------------------------------------------

def _validate_class_count(n_classes: int, caller: str) -> None:
    """Raise ValueError if n_classes is not a sane positive integer >= 2."""
    if not isinstance(n_classes, (int, np.integer)) or n_classes < 2:
        raise ValueError(
            f"{caller}: n_classes={n_classes!r} must be an integer >= 2. "
            "Pass cfg.num_classes explicitly (35 for the current WLASL "
            "label map, artifacts/label_map_v1.json) — check the caller "
            "isn't passing a stray default."
        )


def _to_label_array(arr: Any, name: str, *, allow_multi_dim: bool = False) -> np.ndarray:
    """
    Coerce an arbitrary array-like into a flat 1-D ``int64`` numpy array.

    Accepts plain numpy arrays, Python lists/tuples, and TF tensors
    (EagerTensor implements ``__array__``, so ``np.asarray()`` converts it
    without this module ever importing ``tensorflow``). This is the single
    conversion point that keeps the rest of the module framework-agnostic —
    see the module docstring's "Framework-agnostic by construction" section.

    Shape guard (post-review, item 4)
    -----------------------------------
    A genuinely 2-D label array (e.g. one-hot labels of shape
    ``(batch, n_classes)`` accidentally yielded by a mis-configured
    dataset) must NOT be silently flattened — ``reshape(-1)`` on a
    ``(16, 35)`` one-hot batch produces 560 "labels" instead of 16, and the
    resulting length-mismatch error surfaces far from its actual cause.
    By default (``allow_multi_dim=False``) any array with more than one
    non-singleton dimension is rejected immediately with a message that
    names the likely cause. Pass ``allow_multi_dim=True`` only for inputs
    that are legitimately multi-dimensional before this function is
    expected to squeeze them (none of this module's current call sites do).

    Parameters
    ----------
    arr  : array-like
    name : str — used only in error messages.
    allow_multi_dim : bool, default False

    Returns
    -------
    np.ndarray, dtype int64, shape (n,)

    Raises
    ------
    ValueError
        If ``arr`` is empty, or has a shape that cannot be safely
        interpreted as a flat label vector (e.g. one-hot encoded labels).
    """
    out = np.asarray(arr)
    if out.size == 0:
        raise ValueError(f"{name} is empty. Cannot compute metrics on zero samples.")

    if out.ndim > 1:
        # Squeeze out singleton dims first (e.g. (n, 1) -> (n,) is always safe).
        squeezed = np.squeeze(out)
        if squeezed.ndim > 1 and not allow_multi_dim:
            raise ValueError(
                f"{name} has shape {out.shape}, which looks like one-hot or "
                "multi-dimensional label encoding rather than a flat vector "
                "of class indices. This module expects integer class indices "
                "(e.g. from GestureDataset, whose labels are int32 class "
                "indices, NOT one-hot vectors — see architectures.py: "
                "loss='sparse_categorical_crossentropy'). If this array is "
                "genuinely one-hot, convert it with np.argmax(arr, axis=-1) "
                "before passing it in."
            )
        out = squeezed

    if out.ndim == 0:
        out = out.reshape(1)

    return out.astype(np.int64)


def _validate_label_array(
    arr: Any, n_classes: int, name: str, *, allow_multi_dim: bool = False,
) -> np.ndarray:
    """
    Coerce + validate that every value in ``arr`` lies in ``[0, n_classes)``.

    This guard exists because sklearn's metric functions, when given an
    explicit ``labels=`` argument, silently *exclude* any sample whose true
    or predicted label falls outside that list rather than raising — which
    would quietly understate the effective sample size on a corrupted label
    array instead of failing loudly. Catching it here, at the boundary,
    converts a silent statistical distortion into an immediate, actionable
    error.

    Raises
    ------
    ValueError
        If ``arr`` is empty, has an unsafe shape (see ``_to_label_array``),
        or contains any value outside ``[0, n_classes)``.
    """
    out = _to_label_array(arr, name, allow_multi_dim=allow_multi_dim)
    min_v, max_v = int(out.min()), int(out.max())
    if min_v < 0 or max_v >= n_classes:
        bad = max_v if max_v >= n_classes else min_v
        raise ValueError(
            f"{name} contains class index {bad}, outside the valid range "
            f"[0, {n_classes}). Check that n_classes={n_classes} matches the "
            "label map this array was produced against "
            "(artifacts/label_map_v1.json — 35 signs for this project)."
        )
    return out


def _validate_equal_length(a: np.ndarray, b: np.ndarray, name_a: str, name_b: str) -> None:
    """Raise ValueError if two 1-D arrays differ in length."""
    if len(a) != len(b):
        raise ValueError(
            f"{name_a} (len={len(a)}) and {name_b} (len={len(b)}) must have "
            "the same length — one entry per clip in both arrays."
        )


def _validate_unique_sign_names(sign_names: Sequence[str]) -> None:
    """
    Raise ValueError if ``sign_names`` contains duplicates.

    Post-review fix (item 6): ``compute_per_class_metrics()`` builds its
    result dict keyed by sign name, and ``sklearn.metrics.classification_report``
    is called with ``target_names=sign_names``. Duplicate names cause two
    distinct, silent failure modes: sklearn's report dict has only one entry
    per name (the later one effectively wins inside sklearn's own internals),
    and this module's ``per_class[name] = ...`` assignment would additionally
    overwrite any earlier entry sharing that name. Either way, one class's
    metrics silently vanish rather than erroring. A duplicate sign name is
    almost certainly a label-map corruption bug and must fail loudly here,
    not produce a quietly-incomplete report.
    """
    seen: Dict[str, int] = {}
    for name in sign_names:
        seen[name] = seen.get(name, 0) + 1
    duplicates = {name: count for name, count in seen.items() if count > 1}
    if duplicates:
        raise ValueError(
            f"sign_names contains duplicate entries: {duplicates}. "
            "Duplicate class names cause silent metric loss (sklearn's "
            "classification_report and this module's per-class dict are "
            "both keyed by name). Check label_map_v1.json / "
            "label_map.get_name_safe() for a corrupted or repeated label."
        )


# ---------------------------------------------------------------------------
# Prediction extraction
# ---------------------------------------------------------------------------

def get_predictions(
    model: Any,
    dataset: Any,
    n_classes: Optional[int] = None,
    return_probs: bool = False,
    split_name: str = "val",
    allow_unbatched: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Run inference over every batch of ``dataset`` and return concatenated
    label / prediction arrays (optionally including softmax probabilities).

    Mirrors the inference pattern already used in ``MacroF1Evaluator``
    (``src/models/train.py``): the model's ``__call__`` API
    (``model(x_batch, training=False)``) is used rather than
    ``model.predict()``, deliberately avoiding Keras's internal predict loop
    and its progress-bar / batching side effects inside evaluation code.

    Framework-agnostic by design: this function never imports ``tensorflow``.
    ``np.asarray(model(x_batch, training=False))`` converts a TF EagerTensor
    to a numpy array directly (EagerTensor implements ``__array__``), and the
    same code path works unchanged for any other callable that returns
    something array-convertible — including, in Stage 8, a thin wrapper
    around a TFLite ``Interpreter`` (see ``src/evaluation/benchmark.py``).

    Parameters
    ----------
    model : Any
        A callable accepting ``model(x_batch, training=False)`` and
        returning a ``(batch, n_classes)`` array-like of class probabilities
        (or logits — only ``argmax`` is used unless ``return_probs=True``).
    dataset : Any
        Any iterable of ``(x_batch, y_batch)`` pairs — a ``tf.data.Dataset``
        in production, or a plain list of tuples in tests.
    n_classes : int, optional
        If provided, validates that (a) the model's output width matches
        ``n_classes`` exactly and (b) every true label falls in
        ``[0, n_classes)``. Strongly recommended in production call sites;
        omit only in tests that intentionally use a smaller synthetic
        class count.
    return_probs : bool, default False
        If True, also return the concatenated ``(n_samples, n_classes)``
        probability array. Needed by calibration.py (reliability diagrams
        require confidence, not just the argmax) and by the SHAP notebook.
    split_name : str, default "val"
        Used only in log messages / error text, so a failure clearly states
        which split (val vs. test) was being evaluated.
    allow_unbatched : bool, default False
        If True, a model output of shape ``(n_classes,)`` (1-D, no batch
        axis) is treated as a single-sample batch and promoted to
        ``(1, n_classes)`` before further processing. Some TFLite
        ``Interpreter`` wrappers return unbatched output for batch=1
        (Stage 8 concern). Default False because production Keras
        inference always returns a proper 2-D batch axis; silently
        accepting 1-D output by default would mask genuine shape bugs in
        the common case. Opt in explicitly from benchmark.py / Stage 8
        call sites that know they're wrapping a TFLite interpreter.

    Returns
    -------
    (y_true, y_pred) : Tuple[np.ndarray, np.ndarray]
        Both ``int64``, shape ``(n_samples,)``, if ``return_probs=False``.
    (y_true, y_pred, y_proba) : Tuple[np.ndarray, np.ndarray, np.ndarray]
        ``y_proba`` is ``float64``, shape ``(n_samples, n_classes)``,
        if ``return_probs=True``.

    Raises
    ------
    ValueError
        If a batch's model output is not 2-D (after the optional
        ``allow_unbatched`` promotion), its width disagrees with an
        explicitly supplied ``n_classes``, or a label batch has an unsafe
        shape (e.g. one-hot encoded labels — see ``_to_label_array``).
    RuntimeError
        If ``dataset`` yields zero batches (e.g. an empty or mis-loaded
        split) — this is treated as a hard failure rather than silently
        returning empty arrays, since an empty val/test split downstream
        would otherwise surface as a much more confusing sklearn error.
        Also raised (post-review, item 3) if ``return_probs=True`` but,
        despite non-zero label batches, zero probability arrays ended up
        collected — an internal-consistency failure that must never be
        allowed to reach ``np.concatenate([])``.
    """
    y_true_batches: List[np.ndarray] = []
    y_pred_batches: List[np.ndarray] = []
    prob_batches: List[np.ndarray] = []

    n_batches = 0
    n_samples = 0

    for x_batch, y_batch in dataset:
        raw_output = model(x_batch, training=False)
        probs = np.asarray(raw_output)

        if probs.ndim == 1 and allow_unbatched:
            probs = probs.reshape(1, -1)

        if probs.ndim != 2:
            raise ValueError(
                f"get_predictions(): model output has shape {probs.shape} on the "
                f"'{split_name}' split; expected a 2-D (batch, n_classes) tensor. "
                "Check that the model's final layer is Dense(n_classes, softmax) "
                "and that x_batch has the expected (batch, seq_len, feature_dim) "
                "shape. If this is an unbatched (n_classes,) TFLite output, pass "
                "allow_unbatched=True."
            )

        if n_classes is not None and probs.shape[1] != n_classes:
            raise ValueError(
                f"get_predictions(): model output width {probs.shape[1]} on the "
                f"'{split_name}' split does not match n_classes={n_classes}. "
                "Check cfg.num_classes against the model's output layer — this "
                "usually indicates the wrong champion model or config was loaded."
            )

        preds  = np.argmax(probs, axis=-1)
        # allow_multi_dim=False here is deliberate: a (batch, n_classes)
        # one-hot label batch must be rejected loudly, not silently
        # flattened into batch * n_classes garbage labels (post-review item 4).
        labels = _to_label_array(y_batch, "y_batch", allow_multi_dim=False)

        if len(labels) != probs.shape[0]:
            raise ValueError(
                f"get_predictions(): batch size mismatch on the '{split_name}' "
                f"split — y_batch has {len(labels)} labels but the model "
                f"produced {probs.shape[0]} predictions. Check the dataset's "
                "batching is consistent between x and y."
            )

        y_true_batches.append(labels)
        y_pred_batches.append(preds)
        if return_probs:
            prob_batches.append(probs)

        n_batches += 1
        n_samples += len(labels)

    if n_batches == 0:
        raise RuntimeError(
            f"get_predictions(): the '{split_name}' dataset produced zero batches. "
            "Check that the split was loaded correctly — e.g. "
            "GestureDataset.load_split(split_name, training=False) — and that "
            "n_val / n_test is non-zero for this dataset instance."
        )

    y_true = np.concatenate(y_true_batches).astype(np.int64)
    y_pred = np.concatenate(y_pred_batches).astype(np.int64)

    if n_classes is not None:
        _validate_label_array(y_true, n_classes, "y_true")

    logger.info(
        f"get_predictions() | split='{split_name}' | n_samples={n_samples} | "
        f"n_batches={n_batches} | return_probs={return_probs}",
        extra={"stage": "evaluation"},
    )

    if return_probs:
        # Post-review fix (item 3): n_batches > 0 guarantees prob_batches is
        # non-empty here (every loop iteration that appends to y_true_batches
        # also appends to prob_batches when return_probs=True), but this is
        # re-asserted explicitly rather than left as an implicit invariant,
        # so a future refactor that breaks the invariant fails loudly here
        # instead of inside np.concatenate([]) with a confusing message.
        if not prob_batches:
            raise RuntimeError(
                f"get_predictions(): return_probs=True but zero probability "
                f"batches were collected on the '{split_name}' split despite "
                f"n_batches={n_batches} > 0. This indicates an internal "
                "inconsistency (e.g. a zero-row batch) — investigate the "
                "dataset's batch sizes before trusting any result."
            )
        y_proba = np.concatenate(prob_batches).astype(np.float64)
        return y_true, y_pred, y_proba

    return y_true, y_pred


def get_val_predictions(model: Any, val_ds: Any) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference over the validation set and return ``(y_true, y_pred)``.

    Thin, spec-naming wrapper preserving the exact function name and
    signature called for in the Stage 6 plan (Part 6.1 of the handoff):
    ``get_val_predictions(model, val_ds) -> Tuple[np.ndarray, np.ndarray]``.
    Internally delegates to the more general ``get_predictions()``, which
    also supports the test split and optional probability extraction —
    use ``get_predictions(model, test_ds, split_name="test")`` directly for
    the one-shot test evaluation in Stage 6 Phase C.

    Parameters
    ----------
    model  : Any — see ``get_predictions()``.
    val_ds : Any — the validation ``tf.data.Dataset``.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        ``(y_true, y_pred)``, both ``int64``, shape ``(n_val,)``.
    """
    y_true, y_pred = get_predictions(model, val_ds, return_probs=False, split_name="val")
    return y_true, y_pred


# ---------------------------------------------------------------------------
# Headline metrics
# ---------------------------------------------------------------------------

def compute_macro_f1(y_true: Any, y_pred: Any, n_classes: int) -> float:
    """
    Compute macro-averaged F1 across all ``n_classes``, forcing every class
    into the average regardless of whether it appears in this particular
    batch of predictions.

    This is the project's primary metric (Part 8, Critical Rule #2) — every
    champion-selection decision in Stage 5 was made on this exact
    computation: ``sklearn.metrics.f1_score(average="macro",
    labels=list(range(n_classes)), zero_division=0)``.

    Why ``labels=list(range(n_classes))`` is mandatory, not optional
    --------------------------------------------------------------------
    Without it, sklearn computes the average only over classes present in
    ``y_true`` or ``y_pred`` for *this specific call*. With 21 singleton
    validation classes, whether a given class appears in a sub-sample (e.g.
    inside ``bootstrap_macro_f1_ci()``'s resampling loop) is partly a matter
    of chance — and silently shrinking the averaging denominator would
    inflate the reported macro-F1 in exactly the resamples where rare
    classes happen to drop out. Pinning the label set to all 35 classes
    every time is what makes macro-F1 comparable across runs, across
    resamples, and across the val/test split.

    Parameters
    ----------
    y_true, y_pred : array-like, shape (n_samples,)
    n_classes      : int

    Returns
    -------
    float in [0.0, 1.0]

    Raises
    ------
    ValueError
        If ``n_classes < 2``, either array is empty, contains a label
        outside ``[0, n_classes)``, or the two arrays differ in length.
    """
    _validate_class_count(n_classes, "compute_macro_f1")
    y_true_arr = _validate_label_array(y_true, n_classes, "y_true")
    y_pred_arr = _validate_label_array(y_pred, n_classes, "y_pred")
    _validate_equal_length(y_true_arr, y_pred_arr, "y_true", "y_pred")

    return float(f1_score(
        y_true_arr, y_pred_arr,
        average="macro",
        labels=list(range(n_classes)),
        zero_division=0,
    ))


def compute_accuracy(
    y_true: Any,
    y_pred: Any,
    n_classes: Optional[int] = None,
) -> float:
    """
    Compute plain (micro) accuracy.

    Secondary metric only (Part 8, Critical Rule #2) — never use this for
    champion selection or ranking decisions on this dataset. With 21
    singleton validation classes, a model can achieve respectable accuracy
    by performing well on a handful of common classes while completely
    failing rare ones; macro-F1 (``compute_macro_f1``) is the metric that
    exposes that failure mode and accuracy is not.

    Validation strictness (post-review fix, item 8)
    --------------------------------------------------
    Earlier drafts of this function used a bare ``_to_label_array()`` with
    no range checking, while every other metric in this module enforces
    ``[0, n_classes)``. That asymmetry meant a corrupted prediction (e.g.
    ``y_pred`` containing a stray ``999``) would silently compute "happily"
    here while every other function in the module would reject it — an
    inconsistency that makes this function's output untrustworthy exactly
    when something has gone wrong elsewhere in the pipeline.

    ``n_classes`` is optional (keyword-friendly, defaults to ``None``) to
    preserve simple two-argument call sites that don't have a class count
    handy. When omitted, this function still enforces "non-negative
    integer labels" (via ``_to_label_array``'s dtype coercion plus an
    explicit non-negativity check) but cannot catch an out-of-range index
    on the high end without knowing ``n_classes``. Supplying ``n_classes``
    is strongly recommended in every production call site, exactly as it
    is required for every other function in this module.

    Parameters
    ----------
    y_true, y_pred : array-like, shape (n_samples,)
    n_classes : int, optional
        If supplied, both arrays are validated to lie in ``[0, n_classes)``,
        matching the strictness of every other metric function here. If
        omitted, only non-negativity is enforced.

    Returns
    -------
    float in [0.0, 1.0]

    Raises
    ------
    ValueError
        If either array is empty, has an unsafe shape, contains a negative
        label, the two arrays differ in length, or (when ``n_classes`` is
        supplied) any label falls outside ``[0, n_classes)``.
    """
    if n_classes is not None:
        _validate_class_count(n_classes, "compute_accuracy")
        y_true_arr = _validate_label_array(y_true, n_classes, "y_true")
        y_pred_arr = _validate_label_array(y_pred, n_classes, "y_pred")
    else:
        y_true_arr = _to_label_array(y_true, "y_true")
        y_pred_arr = _to_label_array(y_pred, "y_pred")
        for arr, name in ((y_true_arr, "y_true"), (y_pred_arr, "y_pred")):
            if arr.min() < 0:
                raise ValueError(
                    f"compute_accuracy(): {name} contains a negative class "
                    f"index ({int(arr.min())}), which is never valid. Pass "
                    "n_classes explicitly for full [0, n_classes) range "
                    "validation, matching every other metric in this module."
                )

    _validate_equal_length(y_true_arr, y_pred_arr, "y_true", "y_pred")
    return float(accuracy_score(y_true_arr, y_pred_arr))


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def compute_confusion_matrix_from_predictions(
    y_true: Any, y_pred: Any, n_classes: int,
) -> np.ndarray:
    """
    Pure ``(y_true, y_pred) -> confusion matrix`` computation, with no model
    or dataset dependency.

    Separated from ``compute_confusion_matrix()`` (which additionally runs
    inference) so that the actual matrix computation is trivially unit
    testable with synthetic label arrays — no model, dataset, or TensorFlow
    required — and so that the one-shot test evaluation and the confusion
    matrix figures can reuse a single already-computed ``(y_true, y_pred)``
    pair instead of re-running inference.

    Parameters
    ----------
    y_true, y_pred : array-like, shape (n_samples,)
    n_classes      : int

    Returns
    -------
    np.ndarray, shape (n_classes, n_classes), dtype int64
        Row = true class, column = predicted class (sklearn convention).
    """
    _validate_class_count(n_classes, "compute_confusion_matrix_from_predictions")
    y_true_arr = _validate_label_array(y_true, n_classes, "y_true")
    y_pred_arr = _validate_label_array(y_pred, n_classes, "y_pred")
    _validate_equal_length(y_true_arr, y_pred_arr, "y_true", "y_pred")

    return confusion_matrix(y_true_arr, y_pred_arr, labels=list(range(n_classes)))


def compute_confusion_matrix(model: Any, ds: Any, n_classes: int) -> np.ndarray:
    """
    Run inference over ``ds`` and return the resulting ``(n_classes,
    n_classes)`` confusion matrix.

    Thin orchestration wrapper matching the exact signature specified in the
    Stage 6 handoff (Part 6.1): ``compute_confusion_matrix(model, ds,
    n_classes) -> np.ndarray``. Delegates inference to ``get_predictions()``
    and matrix computation to ``compute_confusion_matrix_from_predictions()``
    — this function adds no logic of its own beyond composing the two, by
    design (see module docstring: pure computation and I/O are deliberately
    kept in separate functions).

    Parameters
    ----------
    model     : Any — see ``get_predictions()``.
    ds        : Any — any iterable of ``(x_batch, y_batch)`` pairs.
    n_classes : int

    Returns
    -------
    np.ndarray, shape (n_classes, n_classes), dtype int64
    """
    y_true, y_pred = get_predictions(model, ds, n_classes=n_classes, return_probs=False)
    return compute_confusion_matrix_from_predictions(y_true, y_pred, n_classes)


# ---------------------------------------------------------------------------
# Per-class support and metrics
# ---------------------------------------------------------------------------

def compute_support_counts(y_true: Any, n_classes: int) -> np.ndarray:
    """
    Count true-label occurrences per class.

    Parameters
    ----------
    y_true    : array-like, shape (n_samples,)
    n_classes : int

    Returns
    -------
    np.ndarray, shape (n_classes,), dtype int64
        ``support[i]`` = number of samples with true label ``i``.
    """
    _validate_class_count(n_classes, "compute_support_counts")
    y_true_arr = _validate_label_array(y_true, n_classes, "y_true")
    return np.bincount(y_true_arr, minlength=n_classes)[:n_classes].astype(np.int64)


def compute_per_class_metrics(
    y_true: Any,
    y_pred: Any,
    sign_names: Sequence[str],
    n_classes: int,
    high_risk_signs: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Compute a per-class precision / recall / F1 / support breakdown, with
    every class forced to appear and every entry annotated with small-sample
    context.

    Built on ``sklearn.metrics.classification_report(output_dict=True,
    labels=list(range(n_classes)), zero_division=0)`` — the explicit
    ``labels`` argument is what forces all ``n_classes`` (not just classes
    observed in this prediction set) to appear in the output. Without it,
    a class entirely absent from both ``y_true`` and ``y_pred`` in a given
    call would silently vanish from the report rather than appearing with
    F1=0.0 / support=0 — exactly the failure mode this project's 21
    singleton validation classes make likely on any sub-sample.

    The raw sklearn dict is re-keyed into a clean, consistent schema
    (``f1_score`` instead of ``"f1-score"``, ``macro_avg`` instead of
    ``"macro avg"``) — see the module docstring's "Clean schema" section —
    and every per-class entry gains three small-sample flags:

      is_singleton     : support == 1   (this project: 21/35 val classes)
      is_zero_support  : support == 0   (no true examples in this call)
      is_high_risk     : name in high_risk_signs (Stage 5 Finding 8 classes)

    Both ``macro_avg`` and ``weighted_avg`` now also carry a ``support``
    key (post-review fix, item 2) equal to ``total_support``, so a consumer
    reading only the averaged block (e.g. a summary table that doesn't
    unpack ``per_class``) still has the sample size that average was
    computed over, without having to cross-reference a sibling key.

    Parameters
    ----------
    y_true, y_pred   : array-like, shape (n_samples,)
    sign_names       : Sequence[str], length n_classes, must be unique
        Index-aligned with class indices, e.g. from
        ``dataset.label_map.get_name_safe(i, f"class_{i}")`` for
        ``i in range(n_classes)``.
    n_classes        : int
    high_risk_signs  : Sequence[str], optional
        Defaults to module-level ``HIGH_RISK_SIGNS`` (the 5 Stage 5
        small-sample classes). Pass an explicit list to flag a different
        set once Stage 6's own confusion-matrix analysis identifies
        additional failure-prone classes.

    Returns
    -------
    dict with keys:
        per_class               : dict[sign_name -> {class_index, precision,
                                   recall, f1_score, support, is_singleton,
                                   is_zero_support, is_high_risk}]
        macro_avg                : {precision, recall, f1_score, support}
        weighted_avg              : {precision, recall, f1_score, support}
        accuracy                  : float
        n_classes                 : int
        n_singleton_classes       : int  — count of support == 1 classes
        n_zero_support_classes    : int  — count of support == 0 classes
        total_support              : int — should equal len(y_true)

    Raises
    ------
    ValueError
        If ``len(sign_names) != n_classes``, if ``sign_names`` contains
        duplicates (post-review fix, item 6), or via the standard
        label-array validation (empty arrays, out-of-range labels, length
        mismatch).
    """
    _validate_class_count(n_classes, "compute_per_class_metrics")
    if len(sign_names) != n_classes:
        raise ValueError(
            f"compute_per_class_metrics(): len(sign_names)={len(sign_names)} "
            f"must equal n_classes={n_classes}. Check the sign_names list was "
            "built as [label_map.get_name_safe(i, ...) for i in range(n_classes)]."
        )
    _validate_unique_sign_names(sign_names)

    y_true_arr = _validate_label_array(y_true, n_classes, "y_true")
    y_pred_arr = _validate_label_array(y_pred, n_classes, "y_pred")
    _validate_equal_length(y_true_arr, y_pred_arr, "y_true", "y_pred")

    high_risk = set(high_risk_signs) if high_risk_signs is not None else set(HIGH_RISK_SIGNS)
    sign_names = list(sign_names)

    raw_report = classification_report(
        y_true_arr,
        y_pred_arr,
        labels=list(range(n_classes)),
        target_names=sign_names,
        output_dict=True,
        zero_division=0,
    )

    support = compute_support_counts(y_true_arr, n_classes)
    total_support = int(support.sum())

    per_class: Dict[str, Dict[str, Any]] = {}
    n_singleton      = 0
    n_zero_support   = 0

    for idx, name in enumerate(sign_names):
        entry        = raw_report.get(name, {})
        cls_support  = int(support[idx])
        is_singleton = cls_support == 1
        is_zero      = cls_support == 0

        if is_singleton:
            n_singleton += 1
        if is_zero:
            n_zero_support += 1

        per_class[name] = {
            "class_index":      idx,
            "precision":        float(entry.get("precision", 0.0)),
            "recall":           float(entry.get("recall", 0.0)),
            "f1_score":         float(entry.get("f1-score", 0.0)),
            "support":          cls_support,
            "is_singleton":     is_singleton,
            "is_zero_support":  is_zero,
            "is_high_risk":     name in high_risk,
        }

    macro_avg    = raw_report.get("macro avg", {})
    weighted_avg = raw_report.get("weighted avg", {})

    summary: Dict[str, Any] = {
        "per_class": per_class,
        "macro_avg": {
            "precision": float(macro_avg.get("precision", 0.0)),
            "recall":    float(macro_avg.get("recall", 0.0)),
            "f1_score":  float(macro_avg.get("f1-score", 0.0)),
            "support":   total_support,
        },
        "weighted_avg": {
            "precision": float(weighted_avg.get("precision", 0.0)),
            "recall":    float(weighted_avg.get("recall", 0.0)),
            "f1_score":  float(weighted_avg.get("f1-score", 0.0)),
            "support":   total_support,
        },
        "accuracy":                float(raw_report.get("accuracy", 0.0)),
        "n_classes":                n_classes,
        "n_singleton_classes":      n_singleton,
        "n_zero_support_classes":   n_zero_support,
        "total_support":            total_support,
    }

    if n_singleton > 0:
        logger.warning(
            f"compute_per_class_metrics(): {n_singleton}/{n_classes} classes have "
            "exactly 1 supporting sample ('singleton classes'). Their F1 is "
            "necessarily binary (0.0 or 1.0) and not statistically meaningful "
            "in isolation — always report alongside support. "
            "See LIMITATIONS.md and Stage 5 Finding 8.",
            extra={"stage": "evaluation"},
        )

    return summary


def rank_classes_by_f1(
    per_class_metrics: Dict[str, Any],
    ascending: bool = True,
) -> List[Dict[str, Any]]:
    """
    Flatten and sort the ``per_class`` section of a
    ``compute_per_class_metrics()`` result for charting (Stage 6's
    "per-class metrics table, sorted ascending by F1").

    Schema validation (post-review fix, item 7)
    -----------------------------------------------
    ``evaluation_report.json`` is a persisted on-disk artefact that may be
    loaded by a different process or notebook than the one that produced
    it — a stale or hand-edited copy could plausibly be missing keys.
    Before sorting, every entry is checked for the presence of
    ``f1_score`` (and ``support``, since callers commonly want to display
    it alongside rank); a malformed entry raises immediately with the
    offending class name rather than failing deep inside the sort
    comparator with a bare ``KeyError``.

    Parameters
    ----------
    per_class_metrics : dict
        The full dict returned by ``compute_per_class_metrics()`` (must
        contain a ``"per_class"`` key).
    ascending : bool, default True
        Ascending F1 surfaces the worst-performing classes first — the
        order Stage 6's plan explicitly asks for ("sort ascending by F1").

    Returns
    -------
    List[dict]
        One dict per class, each the original per-class entry plus
        ``"sign"`` (the class name) and ``"rank"`` (1-indexed in the
        requested sort order).

    Raises
    ------
    ValueError
        If ``per_class_metrics`` does not contain a ``"per_class"`` key, or
        if any per-class entry is missing the required ``f1_score`` /
        ``support`` keys.
    """
    per_class = per_class_metrics.get("per_class")
    if per_class is None:
        raise ValueError(
            "rank_classes_by_f1() expects the dict returned by "
            "compute_per_class_metrics(), which must contain a 'per_class' key. "
            f"Got top-level keys: {list(per_class_metrics.keys())}"
        )

    required_keys = {"f1_score", "support"}
    malformed: Dict[str, set] = {}
    for name, stats in per_class.items():
        if not isinstance(stats, Mapping):
            malformed[name] = required_keys
            continue
        missing = required_keys - stats.keys()
        if missing:
            malformed[name] = missing

    if malformed:
        raise ValueError(
            "rank_classes_by_f1(): per_class entries are missing required "
            f"keys: {malformed}. This usually means evaluation_report.json "
            "was loaded from a stale or hand-edited copy that predates the "
            "current compute_per_class_metrics() schema. Regenerate it via "
            "run_evaluation.py rather than editing the JSON by hand."
        )

    rows = [{"sign": name, **stats} for name, stats in per_class.items()]
    rows.sort(key=lambda r: r["f1_score"], reverse=not ascending)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


# ---------------------------------------------------------------------------
# Bootstrap confidence interval for macro-F1
# ---------------------------------------------------------------------------

def bootstrap_macro_f1_ci(
    y_true: Any,
    y_pred: Any,
    n_classes: int,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    ci_level: float = DEFAULT_BOOTSTRAP_CI,
    seed: int = DEFAULT_SEED,
    include_distribution: bool = False,
) -> Dict[str, Any]:
    """
    Quantify macro-F1 uncertainty via a clip-level bootstrap, turning Stage
    5's informal observation ("epoch-to-epoch swings of 3-5pp are structural
    noise") into a reproducible, reportable confidence interval.

    Procedure
    ---------
    1. Resample ``n_samples`` indices from ``[0, n_samples)`` *with
       replacement*, ``n_bootstrap`` times.
    2. For each resample, recompute macro-F1 on the resampled
       ``(y_true, y_pred)`` pair using the exact same all-classes-forced
       computation as ``compute_macro_f1()``.
    3. Report the empirical ``ci_level`` percentile interval of the
       resulting distribution (e.g. the 5th/95th percentiles for a 90% CI).

    Why this is clip-level "for free"
    ------------------------------------
    ``y_true`` / ``y_pred`` already have exactly one entry per validation
    (or test) *clip* — there is no frame-level granularity in these arrays
    to begin with (frame-level structure was consumed by the model's
    sequence input long before prediction). Resampling array indices is
    therefore automatically a clip-level bootstrap; there is no separate
    implementation step for "clip-level, not frame-level" beyond using these
    already-aggregated arrays as the resampling unit.

    IMPORTANT — this is NOT a signer-aware bootstrap (post-review item 1)
    ---------------------------------------------------------------------
    The WLASL splits are signer-independent (Stage 1, zero overlap across
    train/val/test), but within the validation set itself, clips still
    cluster by signer: 52 clips come from only 7 signers. Prediction errors
    are plausibly correlated within a signer (a signer with an unusual
    signing style or camera angle may produce several wrong predictions
    together, not independently). This function resamples *clips*
    uniformly at random — it does NOT resample signers and then take all of
    that signer's clips (a signer-stratified block bootstrap), which would
    be the more conservative, signer-aware alternative.

    Practical consequence: this interval likely UNDERSTATES true
    uncertainty, because the effective sample size is probably smaller than
    52 once within-signer correlation is accounted for. The returned dict
    carries this caveat verbatim (``resampling_unit`` and ``caveat`` keys)
    specifically so every downstream consumer — ``evaluation_report.json``,
    ``LIMITATIONS.md``, the Stage 11 report — inherits it automatically. A
    true signer-stratified bootstrap is left to ``signer_analysis.py``,
    which has access to ``signer_ids`` and is the more natural home for a
    signer-grouped resampling scheme; bolting that capability onto this
    function would require threading signer metadata through a module that
    is otherwise deliberately model/dataset/metadata-agnostic.

    On singleton-class disappearance widening the interval
    ------------------------------------------------------------
    With 21/35 validation classes having exactly one supporting clip, many
    bootstrap resamples will, by chance, omit a singleton class's one clip
    entirely. Because ``labels=list(range(n_classes))`` is still enforced
    inside the per-resample macro-F1 computation, an omitted class
    contributes F1=0.0 to that resample's average (``zero_division=0``)
    rather than being dropped from the denominator. This is the project's
    deliberate, documented choice — consistent with how every other
    macro-F1 computation in this codebase forces all 35 classes into the
    average — and it is what makes the resulting interval answer the
    question "what if a rare class's only example had gone the other way?"
    It is not asserted here to be the uniquely "correct" statistical
    estimator in any general sense: a stratified-by-class bootstrap (which
    guarantees every class appears in every resample) is a legitimate
    alternative that answers a different question ("given that every class
    is represented, how much does within-class sampling noise alone move
    macro-F1?"). The diagnostic ``mean_classes_with_zero_true_support``
    reports how often class-disappearance happens on average under this
    project's chosen (unstratified) scheme, so the CI's width is
    explainable rather than a mysterious black box.

    Parameters
    ----------
    y_true, y_pred : array-like, shape (n_samples,)
    n_classes      : int
    n_bootstrap    : int, default 1000
    ci_level       : float in (0, 1), default 0.90
        Project default is 90%, not the more conventional 95%, because a
        95% interval on ~50 clips would be wide enough to communicate little
        beyond "we don't know" — see ``DEFAULT_BOOTSTRAP_CI`` docstring.
    seed           : int, default 42 (the project's global seed)
        Bootstrap resampling uses ``numpy.random.default_rng(seed)``. The
        same ``(y_true, y_pred, n_classes, n_bootstrap, ci_level, seed)``
        tuple always reproduces an identical interval — required for the
        result to be citable in the Stage 11 report rather than a number
        that shifts every time the notebook is re-run.
    include_distribution : bool, default False
        If True, also returns the full array of ``n_bootstrap`` resampled
        macro-F1 values (e.g. for plotting a histogram in Notebook 06).
        Defaults to False to keep the result dict small when only the
        summary statistics are needed (e.g. for ``evaluation_report.json``).

    Returns
    -------
    dict with keys:
        point_estimate                      : float — macro-F1 on the full,
                                                 un-resampled data
        ci_level                            : float
        ci_lower, ci_upper, ci_width        : float
        bootstrap_mean, bootstrap_std,
        bootstrap_median                    : float — distribution summary
        mean_classes_with_zero_true_support : float — average, across all
                                                 resamples, of how many of the
                                                 n_classes classes had zero
                                                 true examples in that resample
        resampling_unit                      : str — always "clip" (see caveat)
        caveat                               : str — the signer-clustering
                                                 caveat described above, ready
                                                 to be embedded verbatim in
                                                 evaluation_report.json /
                                                 LIMITATIONS.md
        n_bootstrap, n_samples, seed         : int
        bootstrap_distribution               : List[float], only if
                                                 include_distribution=True

    Raises
    ------
    ValueError
        If ``n_samples < 5``, or ``ci_level`` is not in ``(0, 1)``, or via
        the standard label-array validation.
    """
    _validate_class_count(n_classes, "bootstrap_macro_f1_ci")
    y_true_arr = _validate_label_array(y_true, n_classes, "y_true")
    y_pred_arr = _validate_label_array(y_pred, n_classes, "y_pred")
    _validate_equal_length(y_true_arr, y_pred_arr, "y_true", "y_pred")

    n_samples = len(y_true_arr)
    if n_samples < _MIN_SAMPLES_FOR_BOOTSTRAP:
        raise ValueError(
            f"bootstrap_macro_f1_ci(): n_samples={n_samples} is below the "
            f"minimum of {_MIN_SAMPLES_FOR_BOOTSTRAP} required for a "
            "meaningful resample. Check that the correct split was passed "
            "(this looks too small to be the full val or test set)."
        )
    if not (0.0 < ci_level < 1.0):
        raise ValueError(f"bootstrap_macro_f1_ci(): ci_level={ci_level} must be in (0, 1).")
    if n_bootstrap < _MIN_BOOTSTRAP_FOR_STABLE_CI:
        logger.warning(
            f"bootstrap_macro_f1_ci(): n_bootstrap={n_bootstrap} is below "
            f"{_MIN_BOOTSTRAP_FOR_STABLE_CI}; percentile CI bounds will be "
            f"noisy. Recommended >= {DEFAULT_N_BOOTSTRAP} for a reportable result.",
            extra={"stage": "evaluation"},
        )

    point_estimate = compute_macro_f1(y_true_arr, y_pred_arr, n_classes)

    rng          = np.random.default_rng(seed)
    labels_range = list(range(n_classes))
    boot_f1               = np.empty(n_bootstrap, dtype=np.float64)
    zero_support_counts   = np.empty(n_bootstrap, dtype=np.int64)

    # ── Class-stratified resampling (bug fix — see Revision history item 13) ──
    # Uniform resampling of n_samples indices lets any class disappear from a
    # given resample purely by chance. With labels=labels_range +
    # zero_division=0 still in force, every absent class is FORCED to F1=0
    # for that resample — even though the model never had a chance to be
    # right or wrong about it. With this dataset's many singleton classes,
    # ~9/35 classes dropped out per resample on the test set, which
    # systematically biased the whole bootstrap distribution below the point
    # estimate (computed on the real data, where every class has genuine
    # support) — producing an invalid interval whose upper bound sat below
    # the point estimate.
    #
    # Fix: resample WITHIN each class's own index pool, preserving each
    # class's original count exactly. Every class with >=1 real example
    # therefore has >=1 example in EVERY resample, so the forced-zero
    # artifact cannot occur for it. Every resample's macro-F1 is still
    # computed over the same fixed 35-class label set as the point estimate
    # — no redefinition of the statistic, just a fix to how clips are drawn.
    class_index_pools = [np.where(y_true_arr == c)[0] for c in range(n_classes)]
    class_pool_sizes  = np.array([len(p) for p in class_index_pools])
    n_zero_support_original = int(np.sum(class_pool_sizes == 0))

    if n_zero_support_original > 0:
        logger.info(
            f"bootstrap_macro_f1_ci(): {n_zero_support_original}/{n_classes} "
            "classes have zero true examples in the ORIGINAL data — these "
            "are forced to F1=0 in both the point estimate and every "
            "resample, consistently (this is not a resampling artifact).",
            extra={"stage": "evaluation"},
        )

    for i in range(n_bootstrap):
        resampled_parts = [
            rng.choice(pool, size=len(pool), replace=True)
            for pool in class_index_pools
            if len(pool) > 0
        ]
        idx  = np.concatenate(resampled_parts)
        yt_i = y_true_arr[idx]
        yp_i = y_pred_arr[idx]

        boot_f1[i] = f1_score(
            yt_i, yp_i,
            average="macro",
            labels=labels_range,
            zero_division=0,
        )
        # Under class-stratified resampling this is now an INVARIANT
        # (== n_zero_support_original on every iteration), not a fluctuating
        # quantity. Still computed per-iteration as a self-check — any
        # deviation from n_zero_support_original would indicate a logic
        # error in the stratification above, and should be investigated.
        zero_support_counts[i] = n_classes - len(np.unique(yt_i))

    alpha     = 1.0 - ci_level
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)
    ci_lower  = float(np.percentile(boot_f1, lower_pct))
    ci_upper  = float(np.percentile(boot_f1, upper_pct))

    if not (ci_lower <= point_estimate <= ci_upper):
        logger.warning(
            f"bootstrap_macro_f1_ci(): point_estimate={point_estimate:.4f} "
            f"falls OUTSIDE the computed {int(ci_level*100)}% CI "
            f"[{ci_lower:.4f}, {ci_upper:.4f}]. A percentile bootstrap "
            "interval should almost never exclude the observed statistic "
            "this is a strong signal of a biased resampling distribution. "
            "If you see this after the class-stratified fix, investigate "
            "before trusting the result.",
            extra={"stage": "evaluation"},
        )

    result: Dict[str, Any] = {
        "point_estimate":                       point_estimate,
        "ci_level":                              ci_level,
        "ci_lower":                              ci_lower,
        "ci_upper":                              ci_upper,
        "ci_width":                              ci_upper - ci_lower,
        "bootstrap_mean":                        float(np.mean(boot_f1)),
        "bootstrap_std":                         float(np.std(boot_f1, ddof=1)),
        "bootstrap_median":                      float(np.median(boot_f1)),
        "mean_classes_with_zero_true_support":   float(np.mean(zero_support_counts)),
        "resampling_unit":                       "clip",
        "caveat":                                _BOOTSTRAP_SIGNER_CAVEAT,
        "n_bootstrap":                            n_bootstrap,
        "n_samples":                              n_samples,
        "seed":                                   seed,
    }
    if include_distribution:
        result["bootstrap_distribution"] = boot_f1.tolist()

    logger.info(
        f"bootstrap_macro_f1_ci() | point={point_estimate:.4f} | "
        f"{int(ci_level * 100)}% CI=[{ci_lower:.4f}, {ci_upper:.4f}] "
        f"(width={result['ci_width']:.4f}) | "
        f"n_bootstrap={n_bootstrap} | n_samples={n_samples} | seed={seed} | "
        f"mean_classes_zero_support={result['mean_classes_with_zero_true_support']:.1f}/{n_classes} | "
        "resampling_unit=clip (NOT signer-aware — see result['caveat'])",
        extra={"stage": "evaluation"},
    )

    return result


# ---------------------------------------------------------------------------
# Consolidated summary
# ---------------------------------------------------------------------------

def compute_evaluation_summary(
    y_true: Any,
    y_pred: Any,
    sign_names: Sequence[str],
    n_classes: int,
    split_name: str = "val",
    high_risk_signs: Optional[Sequence[str]] = None,
    compute_ci: bool = True,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    ci_level: float = DEFAULT_BOOTSTRAP_CI,
    seed: int = DEFAULT_SEED,
    include_confusion_matrix: bool = True,
    metadata: Optional[Mapping[str, Sequence[Any]]] = None,
) -> Dict[str, Any]:
    """
    Bundle every primitive in this module into a single, JSON-serialisable
    evaluation summary for one split.

    This is the one function Notebook 06 and ``pipelines/run_evaluation.py``
    should call to get "the numbers" for a split — it exists so that no
    caller has to remember to separately call ``compute_macro_f1``,
    ``compute_accuracy``, ``compute_per_class_metrics``,
    ``compute_confusion_matrix_from_predictions``, and
    ``bootstrap_macro_f1_ci`` and then hand-assemble the result. All values
    in the returned dict are plain Python ``float`` / ``int`` / ``bool`` /
    ``list`` / ``dict`` — safe to ``json.dump()`` directly.

    Does NOT run inference itself — callers pass already-extracted
    ``(y_true, y_pred)`` (e.g. from ``get_predictions()`` or
    ``get_val_predictions()``). This keeps the one-shot test-set evaluation
    able to run inference exactly once and then call this function purely
    on the resulting arrays, rather than this function silently triggering
    a second, redundant inference pass over the test set.

    Validation is performed once (post-review item 11)
    -------------------------------------------------------
    ``y_true`` / ``y_pred`` are validated exactly once at the top of this
    function; the validated arrays (not the raw inputs) are then passed to
    every constituent computation, avoiding three-to-four redundant
    re-validations of the same data per summary call. Each constituent
    function remains independently callable and independently
    unit-testable with its own validation intact — this optimisation only
    removes duplicate work *within* a single ``compute_evaluation_summary``
    call, it does not introduce an unvalidated code path anywhere else.

    Metadata passthrough (post-review fix, item 9)
    ----------------------------------------------------
    The Stage 6 plan's validation prediction cache
    (``reports/evaluation/predictions/val_predictions.npz``) carries
    ``clip_ids``, ``signer_ids``, ``detected_frame_count``, and
    ``missing_pct`` alongside ``y_true``/``y_pred``/``y_prob``, specifically
    to support failure-mode hypotheses (e.g. "do errors cluster in heavily
    zero-filled or short clips?") without re-joining
    ``landmark_inventory.csv`` downstream. ``compute_evaluation_summary()``
    accepts this metadata as an optional mapping of equal-length sequences
    and echoes it back, length-validated, under the ``"metadata"`` key —
    this module still performs zero file I/O; ``metadata`` must already be
    loaded into memory by the caller (e.g. from the ``.npz`` cache).

    Parameters
    ----------
    y_true, y_pred            : array-like, shape (n_samples,)
    sign_names                 : Sequence[str], length n_classes, unique
    n_classes                  : int
    split_name                 : str, default "val" — carried into the
                                  output dict and log line only.
    high_risk_signs             : Sequence[str], optional — see
                                  ``compute_per_class_metrics()``.
    compute_ci                  : bool, default True — set False to skip the
                                  bootstrap (e.g. for a fast dev iteration).
    n_bootstrap, ci_level, seed : see ``bootstrap_macro_f1_ci()``.
    include_confusion_matrix    : bool, default True — set False if the
                                  caller will compute/plot the matrix
                                  separately and doesn't need it duplicated
                                  in this summary dict.
    metadata                    : Mapping[str, Sequence], optional
        e.g. ``{"clip_ids": [...], "signer_ids": [...],
        "detected_frame_count": [...], "missing_pct": [...]}``. Every
        sequence must have length ``== len(y_true)``. Stored verbatim
        (as lists) under the ``"metadata"`` key of the returned dict.

    Returns
    -------
    dict with keys:
        split_name             : str
        n_samples               : int
        n_classes               : int
        macro_f1                 : float
        accuracy                  : float
        per_class_metrics          : dict (see compute_per_class_metrics())
        confusion_matrix            : List[List[int]], only if
                                       include_confusion_matrix=True
        macro_f1_bootstrap_ci        : dict (see bootstrap_macro_f1_ci()),
                                       only if compute_ci=True
        metadata                     : dict[str, list], only if a
                                       ``metadata`` mapping was supplied

    Raises
    ------
    ValueError
        If any constituent validation fails, or if a ``metadata`` sequence's
        length disagrees with ``len(y_true)``.
    """
    # Validate once; reuse the validated arrays everywhere below
    # (post-review item 11).
    y_true_arr = _validate_label_array(y_true, n_classes, "y_true")
    y_pred_arr = _validate_label_array(y_pred, n_classes, "y_pred")
    _validate_equal_length(y_true_arr, y_pred_arr, "y_true", "y_pred")
    n_samples = len(y_true_arr)

    macro_f1  = compute_macro_f1(y_true_arr, y_pred_arr, n_classes)
    accuracy  = compute_accuracy(y_true_arr, y_pred_arr, n_classes=n_classes)
    per_class = compute_per_class_metrics(
        y_true_arr, y_pred_arr, sign_names, n_classes, high_risk_signs,
    )

    summary: Dict[str, Any] = {
        "split_name":         split_name,
        "n_samples":          n_samples,
        "n_classes":          n_classes,
        "macro_f1":           macro_f1,
        "accuracy":           accuracy,
        "per_class_metrics":  per_class,
    }

    if include_confusion_matrix:
        cm = compute_confusion_matrix_from_predictions(y_true_arr, y_pred_arr, n_classes)
        summary["confusion_matrix"] = cm.tolist()

    if compute_ci:
        summary["macro_f1_bootstrap_ci"] = bootstrap_macro_f1_ci(
            y_true_arr, y_pred_arr, n_classes,
            n_bootstrap=n_bootstrap, ci_level=ci_level, seed=seed,
        )

    if metadata is not None:
        clean_metadata: Dict[str, List[Any]] = {}
        for key, values in metadata.items():
            values_list = list(values)
            if len(values_list) != n_samples:
                raise ValueError(
                    f"compute_evaluation_summary(): metadata['{key}'] has "
                    f"length {len(values_list)}, but y_true/y_pred have "
                    f"length {n_samples}. Every metadata sequence must be "
                    "index-aligned with y_true/y_pred (one entry per clip)."
                )
            clean_metadata[key] = values_list
        summary["metadata"] = clean_metadata

    logger.info(
        f"compute_evaluation_summary() | split='{split_name}' | "
        f"macro_f1={macro_f1:.4f} | accuracy={accuracy:.4f} | "
        f"n_samples={summary['n_samples']} | "
        f"n_singleton={per_class['n_singleton_classes']}/{n_classes}"
        + (f" | metadata_keys={list(metadata.keys())}" if metadata is not None else ""),
        extra={"stage": "evaluation"},
    )

    return summary


# ---------------------------------------------------------------------------
# Import-time self-check
# ---------------------------------------------------------------------------

def _self_check() -> None:
    """
    Cheap, dependency-free sanity check on module constants, mirroring the
    pattern used in ``src/models/architectures.py``.

    Post-review change (item 12): this no longer asserts ``N_CLASSES == 35``.
    That assertion directly contradicted the module's own design principle
    — every function here takes ``n_classes`` explicitly so the module
    works for any label map, present or future (WLASL-50, KSL, ...) — and
    would have raised at import time, before a single function call,
    the moment a future label map changed the class count. The
    authoritative, runtime check belongs against ``cfg.num_classes`` at the
    point a model/dataset is actually constructed, exactly mirroring
    ``architectures.py::_check_n_classes()``: log loudly on a mismatch,
    never crash at import time over a documentation constant.

    What remains checked here is internal self-consistency of constants
    that this module DOES own and assert ownership over: the cardinality of
    ``HIGH_RISK_SIGNS`` (a specific, named Stage 5 finding, not a function
    of label-map size) and the bootstrap defaults' sane ranges.
    """
    if N_CLASSES != 35:
        logger.info(
            f"metrics.py: N_CLASSES={N_CLASSES} (documentation constant) "
            "differs from the original WLASL-35 project value. This is "
            "informational only — every function in this module takes "
            "n_classes as an explicit argument and does not read this "
            "constant. Verify cfg.num_classes at the call site if this is "
            "unexpected.",
            extra={"stage": "evaluation"},
        )

    assert len(HIGH_RISK_SIGNS) == 5, (
        f"metrics.py: HIGH_RISK_SIGNS has {len(HIGH_RISK_SIGNS)} entries; "
        "expected the 5 Stage 5 Finding 8 classes (clothes, think, birthday, "
        "name, book). If this was an intentional update, this assertion "
        "should be revised alongside it."
    )
    assert 0.0 < DEFAULT_BOOTSTRAP_CI < 1.0
    assert DEFAULT_N_BOOTSTRAP >= _MIN_BOOTSTRAP_FOR_STABLE_CI


if __debug__:
    _self_check()


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------

__all__ = [
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
]