"""
src/evaluation/benchmark.py
=============================
Latency and model-size benchmarking primitives for the WLASL 35-class
gesture recognition system. This is Stage 6 (Evaluation, Benchmarking, and
Interpretability), Phase A2, per the Stage 6 (Revised) plan.

Scope of this module (Phase A2, exactly)
------------------------------------------
The Stage 6 (Revised) plan defines Phase A2 narrowly and deliberately:

    "benchmark.py — 200 calls, 20 warmup, median/p95/p99/min/max/std/FPS.
    Built and tested against the champion Keras SavedModel directly; no
    dependency on Phase B. Also do the throwaway dynamic-range TFLite
    conversion here ... purely for benchmarking — clearly commented as
    non-final, superseded by Stage 8's real export."

Concretely, that means this module:

  1. Times real inference calls against the actual champion Keras
     SavedModel — it does NOT estimate latency from FLOPs or param counts.
  2. Has zero dependency on the Phase B1 validation-prediction cache
     (``val_predictions.npz``) — every function here is independently
     runnable the moment a SavedModel exists on disk.
  3. Produces a SCRATCH, NON-FINAL TFLite file purely so a realistic
     "quantised model latency" number can appear in Stage 6's
     ``latency_benchmark.png`` without waiting on Stage 8
     (``src/export/convert.py``), which owns the authoritative,
     accuracy-verified, metadata-tracked export. Every function that
     touches this scratch file says so loudly, in the log line and the
     docstring, so nobody mistakes it for a deployment artefact.
  4. Does NOT touch MediaPipe extraction latency, overlay-rendering
     latency, or the full webcam per-frame pipeline. Those stages live in
     ``src/inference/predictor.py`` (Stage 7) and
     ``src/demo/webcam_demo.py`` (Stage 9), neither of which exists yet.
     ``benchmark_pipeline_preprocessing()`` benchmarks the one
     pre-existing, already-built stage (``FeaturePipeline.__call__``) so
     that the per-stage timing table can be filled in incrementally as
     later stages land, rather than faking numbers for stages that don't
     exist.

Why a separate timing harness from metrics.py
-----------------------------------------------
``metrics.py`` answers "is the model right?" (macro-F1, confusion matrices,
bootstrap CI). This module answers a different, equally important question
for a project whose stated thesis (handoff Part 1.2) is real-time CPU
deployment: "is the model fast enough?" Mixing the two concerns into one
file would make ``metrics.py`` — already the foundation every other Stage 6
module imports — slower to reason about. The two modules share exactly one
design principle, copied deliberately from ``metrics.py``'s "framework-
agnostic by construction" section: a model here is "a callable that returns
something array-like", never assumed to be specifically a ``tf.keras.Model``.
``TFLiteCallable`` (below) is the concrete adapter that makes a
``tf.lite.Interpreter`` satisfy that same contract, so the exact same
``metrics.py`` functions (``get_predictions``, ``compute_evaluation_summary``)
that already run against the Keras champion will run unmodified against a
TFLite interpreter the moment Stage 8 produces a verified ``.tflite`` file.

Timing methodology (why these specific choices)
----------------------------------------------------
Warmup is discarded, not averaged in
    The first few calls into any inference path pay one-time costs that a
    deployed app pays exactly once at startup, never again. ``warmup=20``
    (project default) is enough for all of these to stabilise.

Input tensors are pre-converted ONCE, outside the timed loop
    ``benchmark_inference()`` converts ``X_sample`` to a ``tf.constant``
    exactly once before entering the timing loop. The same principle applies
    to ``benchmark_tflite_inference()``.

Output is forced to materialise inside the timed region
    ``np.asarray(raw_output)`` inside the timed closure blocks until the
    output value is actually available, ensuring the timed region captures
    the complete forward pass. This applies consistently to BOTH the Keras
    and TFLite paths.

FPS is derived from MEDIAN latency, not from (total_time / n_calls)
    Median is robust to the occasional GC pause or OS scheduling hiccup
    that the mean would be dragged around by. p95/p99 are reported
    separately so that tail latency is visible without distorting the
    headline FPS number.

n_calls=200, warmup=20 are defaults, not hard-coded
    Every function accepts ``n_calls`` / ``warmup`` explicitly. A warning
    fires below ``_MIN_CALLS_FOR_STABLE_PERCENTILES`` (30).

Post-review fixes applied
---------------------------
The following issues from the Phase A2 critical review were assessed,
verified, and addressed in this revision:

  #1  FIXED. ``TFLiteCallable.__call__()`` now validates that the sample
      shape matches the interpreter's fixed input shape and raises a clear
      ``ValueError`` rather than producing a cryptic TFLite error. Shape
      resize support is also added for models exported with dynamic batch
      dimensions (shape_signature containing -1).

  #2  FIXED. ``benchmark_model_registry()`` now calls
      ``tf.keras.backend.clear_session()`` after each model to release
      TensorFlow's C++-side allocator pools, not just ``gc.collect()``.

  #3  FIXED. ``benchmark_inference()`` now falls back to
      ``model(x_tensor)`` (without the ``training`` kwarg) if the callable
      raises ``TypeError`` on ``model(x_tensor, training=False)``.
      Documentation updated to accurately reflect the narrowed-but-safe
      contract: "Keras-style callables with optional training= kwarg."

  #4  FIXED. ``benchmark_tflite_inference()``'s inner closure now wraps
      ``interpreter.get_tensor()`` in ``np.asarray()`` for consistent
      output materialisation semantics with the Keras path.

  #5  FIXED. ``np.median(durations_ms)`` used instead of
      ``np.percentile(durations_ms, 50)`` for clarity and intent.

  #6  FIXED. ``benchmark_pipeline_preprocessing()`` now guards against
      ``pipeline.output_shape is None`` before calling ``list()``.

  #7  FIXED. ``benchmark_champion_summary()`` now guards the speedup
      computation with ``np.isfinite()`` on both median values.

  #8  FIXED. ``time_callable()`` now warns when ``warmup >= n_calls``,
      which almost certainly indicates an accidental argument swap.

  #9  N/A. Warmup failure propagation is acceptable at this project scale;
      a raw exception from warmup is informative (it identifies the failure
      point before timing begins).

  #10 FIXED. ``benchmark_tflite_inference()`` now logs a WARNING when
      ``X_sample`` dtype differs from the interpreter's expected dtype and
      a silent cast occurs.

  #11 FIXED. ``TFLiteCallable`` now caches ``input_index`` and
      ``output_index`` in ``__init__`` rather than re-looking them up on
      every ``__call__``.

  #12 N/A (future enhancement, out of Stage 6 scope).

  #13 N/A. NaN median_ms is not a realistic concern for this project;
      the current sort behaviour is adequate.

  #14 FIXED. ``convert_to_scratch_tflite()`` now logs conversion wall-clock
      time to help identify unexpectedly slow quantisation.

Additional fixes (not in the review):
  - ``_ensure_batched_float_array()`` now also checks for zero-size arrays.
  - ``benchmark_champion_summary()`` validates ``keras_model`` is not None.
  - ``compute_file_size_mb()`` path is resolved before stat() for clarity.
  - The ``_SCRATCH_TFLITE_DEFAULT_PATH`` parent directory is created lazily
    (already handled by ``convert_to_scratch_tflite``), so no import-time
    side-effect occurs.
  - All public functions now follow a strict ``(n_classes,)`` convention:
    we never reference ``cfg.data.feature_dim`` (which does not exist as a
    DataConfig field); feature_dim always comes from ``pipeline.feature_dim``.

Champion model context (for reference)
-----------------------------------------
The champion model (``bilstm_hands_only_v4_aug``) config snapshot confirms:

  - ``early_stopping_monitor: val_accuracy``  (NOT val_macro_f1 as narrated
    in the Stage 5 handoff — this discrepancy is flagged in
    ``reports/evaluation/evaluation_report.json`` per Phase F requirements;
    benchmark.py takes no position on it and does not reproduce the error).
  - Input shape: (1, 100, 126)  — seq_len=100, landmark_config=hands_only.
  - Parameters: 68,771. Estimated float32 weight size: 0.262 MB.

Module-level exports
---------------------
    time_callable                    — generic 0-arg-callable timing harness
    benchmark_inference               — time a Keras-style model's forward pass
    benchmark_keras_savedmodel        — load + benchmark a SavedModel path
    benchmark_pipeline_preprocessing  — time FeaturePipeline.__call__ (inference mode)
    convert_to_scratch_tflite         — THROWAWAY dynamic-range TFLite export
    TFLiteCallable                    — Interpreter adapter satisfying the
                                         metrics.py "model(x, training=False)"
                                         contract
    benchmark_tflite_inference        — time a .tflite interpreter's forward pass
    compute_keras_model_size_mb       — float32 weight-size estimate (Keras)
    compute_file_size_mb              — on-disk file size (e.g. .tflite)
    benchmark_model_registry          — benchmark several SavedModels under
                                         one shared input sample
    build_latency_comparison_rows     — flatten a registry result into rows
                                         for a DataFrame / bar chart
    benchmark_champion_summary        — one-call Keras-vs-TFLite side-by-side
    DEFAULT_N_CALLS, DEFAULT_WARMUP   — project-standard benchmarking defaults
"""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Project-standard timed call count, matching the handoff's
#: ``benchmark_inference(model, X_sample, n_calls=200, warmup=20)`` spec
#: (Part 6.2) and the Stage 6 (Revised) plan's Phase A2 description.
DEFAULT_N_CALLS: int = 200

#: Project-standard warmup count discarded before timing begins.
DEFAULT_WARMUP: int = 20

#: Below this many timed calls, percentile estimates (especially p99) are
#: dominated by sampling noise — warn rather than error, since a fast
#: dev-loop smoke test (e.g. n_calls=10) is a legitimate use case.
_MIN_CALLS_FOR_STABLE_PERCENTILES: int = 30

#: Default scratch (non-final) TFLite output path. The leading underscore
#: is a deliberate naming signal — "this is scratch, not a deliverable".
_SCRATCH_TFLITE_DEFAULT_PATH: str = "models/_bench_scratch.tflite"

#: Bytes-per-parameter for an uncompressed float32 Keras weight tensor.
#: Matches the identical constant used in architectures.py and factory.py
#: so that size estimates are numerically consistent across the codebase.
_BYTES_PER_FLOAT32: int = 4


# ---------------------------------------------------------------------------
# Internal validation helpers
# ---------------------------------------------------------------------------

def _validate_n_calls(n_calls: int, caller: str) -> None:
    """Raise ValueError if n_calls is not a positive integer."""
    if not isinstance(n_calls, (int, np.integer)) or n_calls < 1:
        raise ValueError(
            f"{caller}: n_calls={n_calls!r} must be a positive integer. "
            f"Project default is {DEFAULT_N_CALLS}."
        )


def _validate_warmup(warmup: int, caller: str) -> None:
    """Raise ValueError if warmup is not a non-negative integer."""
    if not isinstance(warmup, (int, np.integer)) or warmup < 0:
        raise ValueError(
            f"{caller}: warmup={warmup!r} must be a non-negative integer. "
            f"Project default is {DEFAULT_WARMUP}."
        )


def _ensure_batched_float_array(X: Any, caller: str) -> np.ndarray:
    """
    Coerce X_sample into a 3-D float32 ``(batch, seq_len, feature_dim)`` array.

    Accepts a single un-batched sample ``(seq_len, feature_dim)`` and promotes
    it to a batch of 1. Rejects genuinely 2-D or >3-D inputs with a clear,
    actionable error. Also rejects zero-size arrays, which would silently
    produce meaningless benchmark numbers.

    Parameters
    ----------
    X : array-like
    caller : str — used only in the error message.

    Returns
    -------
    np.ndarray, dtype float32, ndim == 3, size > 0
    """
    X = np.asarray(X, dtype=np.float32)

    if X.ndim == 2:
        X = X[np.newaxis, ...]
    elif X.ndim != 3:
        raise ValueError(
            f"{caller}: X_sample has shape {X.shape}; expected either "
            "(seq_len, feature_dim) for a single sample or "
            "(batch, seq_len, feature_dim) for a batch. "
            "Check that this array came from FeaturePipeline output "
            "(e.g. pipeline(raw_arr, training=False)), not raw, "
            "un-pipelined landmarks."
        )

    if X.size == 0:
        raise ValueError(
            f"{caller}: X_sample has zero elements (shape {X.shape}). "
            "Benchmarking a zero-size array produces meaningless results. "
            "Provide a representative non-empty input sample."
        )

    return X


# ---------------------------------------------------------------------------
# Core timing harness — framework-agnostic, zero TensorFlow dependency
# ---------------------------------------------------------------------------

def time_callable(
    fn: Callable[[], Any],
    n_calls: int = DEFAULT_N_CALLS,
    warmup: int = DEFAULT_WARMUP,
    description: str = "callable",
) -> Dict[str, Any]:
    """
    Time ``n_calls`` invocations of a zero-argument callable, after
    discarding ``warmup`` untimed invocations.

    This is the single timing primitive every other benchmarking function in
    this module is built on top of. Centralising the ``time.perf_counter()``
    loop here guarantees every latency number reported anywhere in Stage 6
    was measured the same way.

    ``fn`` takes no arguments and is expected to be a closure that already
    captures everything it needs (a pre-converted input tensor, a bound
    model, etc.) — see ``benchmark_inference()`` for the canonical example
    of why pre-binding matters for measurement validity.

    Parameters
    ----------
    fn : Callable[[], Any]
        Zero-argument callable to time. Its return value is discarded here
        (callers are responsible for ensuring ``fn`` itself forces any lazy
        computation to materialise before returning).
    n_calls : int, default 200
        Number of TIMED invocations.
    warmup : int, default 20
        Number of UNTIMED invocations executed first and discarded.
    description : str, default "callable"
        Human-readable label, used only in log lines and echoed back in the
        result dict for downstream tables/charts.

    Returns
    -------
    dict with keys:
        description   : str
        n_calls       : int
        warmup        : int
        median_ms      : float
        mean_ms        : float
        p95_ms         : float
        p99_ms         : float
        min_ms         : float
        max_ms         : float
        std_ms         : float  (sample std, ddof=1; 0.0 if n_calls == 1)
        fps            : float  (1000 / median_ms; inf if median_ms == 0)
        total_time_s    : float  (wall-clock time for the timed region only,
                                  excludes warmup)

    Raises
    ------
    ValueError
        If ``n_calls < 1`` or ``warmup < 0``.
    """
    _validate_n_calls(n_calls, "time_callable")
    _validate_warmup(warmup, "time_callable")

    # Post-review fix #8: warn if warmup >= n_calls, which almost certainly
    # indicates an accidental argument swap or a misconfigured call.
    if warmup >= n_calls:
        logger.warning(
            f"time_callable('{description}'): warmup={warmup} >= n_calls={n_calls}. "
            "This is almost certainly a misconfiguration — the warmup phase "
            "would consume all or more calls than the timed phase. "
            f"Project defaults: n_calls={DEFAULT_N_CALLS}, warmup={DEFAULT_WARMUP}.",
            extra={"stage": "evaluation"},
        )

    # ── Warmup — discarded ──────────────────────────────────────────────
    for _ in range(warmup):
        fn()

    # ── Timed region ─────────────────────────────────────────────────────
    durations_s = np.empty(n_calls, dtype=np.float64)
    t_region_start = time.perf_counter()
    for i in range(n_calls):
        t0 = time.perf_counter()
        fn()
        durations_s[i] = time.perf_counter() - t0
    total_time_s = time.perf_counter() - t_region_start

    durations_ms = durations_s * 1000.0

    if n_calls < _MIN_CALLS_FOR_STABLE_PERCENTILES:
        logger.warning(
            f"time_callable('{description}'): n_calls={n_calls} is below "
            f"{_MIN_CALLS_FOR_STABLE_PERCENTILES}; p95/p99 estimates are "
            "noisy (p99 of a small sample is effectively just the max). "
            f"Recommended >= {DEFAULT_N_CALLS} for a citable result.",
            extra={"stage": "evaluation"},
        )

    # Post-review fix #5: use np.median() directly instead of
    # np.percentile(..., 50) — clearer intent, avoids percentile
    # interpolation semantics, slightly faster.
    median_ms = float(np.median(durations_ms))
    p95_ms    = float(np.percentile(durations_ms, 95))
    p99_ms    = float(np.percentile(durations_ms, 99))
    mean_ms   = float(np.mean(durations_ms))
    std_ms    = float(np.std(durations_ms, ddof=1)) if n_calls > 1 else 0.0
    min_ms    = float(np.min(durations_ms))
    max_ms    = float(np.max(durations_ms))

    # FPS from median (robust) latency, not from total_time_s / n_calls.
    fps = float(1000.0 / median_ms) if median_ms > 0 else float("inf")

    result: Dict[str, Any] = {
        "description":  description,
        "n_calls":      int(n_calls),
        "warmup":       int(warmup),
        "median_ms":    round(median_ms, 4),
        "mean_ms":      round(mean_ms, 4),
        "p95_ms":       round(p95_ms, 4),
        "p99_ms":       round(p99_ms, 4),
        "min_ms":       round(min_ms, 4),
        "max_ms":       round(max_ms, 4),
        "std_ms":       round(std_ms, 4),
        "fps":          round(fps, 2),
        "total_time_s": round(total_time_s, 4),
    }

    logger.info(
        f"time_callable('{description}') | median={median_ms:.3f}ms | "
        f"p95={p95_ms:.3f}ms | p99={p99_ms:.3f}ms | min={min_ms:.3f}ms | "
        f"max={max_ms:.3f}ms | fps={fps:.1f} | n_calls={n_calls} warmup={warmup}",
        extra={"stage": "evaluation"},
    )
    return result


# ---------------------------------------------------------------------------
# Keras model benchmarking
# ---------------------------------------------------------------------------

def _call_model(model: Any, x_tensor: Any) -> np.ndarray:
    """
    Call a model for a single forward pass, materialising the output.

    Attempts ``model(x_tensor, training=False)`` first. If the callable
    raises ``TypeError`` (e.g. it does not accept a ``training`` keyword
    argument — common for non-Keras wrappers), falls back to
    ``model(x_tensor)``.

    This resolves the contradiction identified in the critical review (#3):
    the original implementation claimed to be "framework-agnostic" but
    required the Keras call signature. The fallback makes the claim true for
    any callable that returns something array-convertible.

    Parameters
    ----------
    model : Any
        A callable accepting ``model(x, training=False)`` or ``model(x)``.
    x_tensor : Any
        Pre-converted input tensor (``tf.constant`` or numpy array).

    Returns
    -------
    np.ndarray — output materialised via ``np.asarray()``.
    """
    try:
        raw_output = model(x_tensor, training=False)
    except TypeError:
        logger.debug(
            "_call_model(): callable does not accept training= kwarg; "
            "falling back to model(x) without training=False.",
            extra={"stage": "evaluation"},
        )
        raw_output = model(x_tensor)
    return np.asarray(raw_output)


def benchmark_inference(
    model: Any,
    X_sample: Any,
    n_calls: int = DEFAULT_N_CALLS,
    warmup: int = DEFAULT_WARMUP,
    description: str = "keras_inference",
) -> Dict[str, Any]:
    """
    Benchmark a Keras-style model's forward-pass latency on a fixed input sample.

    Matches the exact signature specified in the handoff (Part 6.2):
    ``benchmark_inference(model, X_sample, n_calls=200, warmup=20)``.

    The model callable contract is: accepts ``model(x, training=False)`` or
    ``model(x)`` and returns something array-convertible. The ``training``
    kwarg is attempted first and silently dropped if the callable raises
    ``TypeError`` — see ``_call_model()``. This makes the function truly
    framework-agnostic while preferring the Keras convention for the primary
    deployment target.

    ``X_sample`` is converted to a ``tf.constant`` exactly once before
    entering the timed region. The model output is materialised via
    ``np.asarray()`` inside the timed closure to capture the complete
    forward pass rather than just kernel dispatch.

    Parameters
    ----------
    model : Any
        A callable satisfying model(x_tensor[, training=False]) → array-like.
        Must accept the call signature ``model(x_tensor, training=False)``
        or ``model(x_tensor)`` and return something array-convertible.
    X_sample : array-like, shape (seq_len, feature_dim) or (batch, seq_len, feature_dim)
        A representative input. For the champion (seq_len=100,
        landmark_config=hands_only), this is shape (100, 126) or (1, 100, 126).
        A 2-D input is automatically promoted to a batch of 1.
    n_calls : int, default 200
    warmup : int, default 20
    description : str, default "keras_inference"
        Label echoed into the result dict — pass the run name when benchmarking
        a specific run so results remain distinguishable in a multi-model comparison.

    Returns
    -------
    dict
        Everything from ``time_callable()``, plus:
            batch_size   : int
            input_shape  : list[int]
            backend      : str, always "keras"

    Raises
    ------
    ValueError
        If ``X_sample`` cannot be interpreted as (seq_len, feature_dim) or
        (batch, seq_len, feature_dim), or if it is zero-size.
    """
    import tensorflow as tf

    X = _ensure_batched_float_array(X_sample, "benchmark_inference")

    if X.shape[0] != 1:
        logger.info(
            f"benchmark_inference('{description}'): X_sample has "
            f"batch_size={X.shape[0]}. Latency benchmarks conventionally "
            "use batch_size=1 to measure single-prediction (real-time "
            "deployment) latency — the per-frame scenario Stage 9's webcam "
            "demo actually runs. Proceeding with the supplied batch as-is; "
            "pass a (seq_len, feature_dim) sample for the deployment-"
            "realistic measurement.",
            extra={"stage": "evaluation"},
        )

    # Pre-convert ONCE, outside the timed loop.
    x_tensor = tf.constant(X, dtype=tf.float32)

    def _call() -> np.ndarray:
        # _call_model materialises the output via np.asarray().
        return _call_model(model, x_tensor)

    stats = time_callable(_call, n_calls=n_calls, warmup=warmup, description=description)
    stats["batch_size"]  = int(X.shape[0])
    stats["input_shape"] = list(X.shape)
    stats["backend"]     = "keras"
    return stats


def benchmark_keras_savedmodel(
    saved_model_path: Union[str, Path],
    X_sample: Any,
    n_calls: int = DEFAULT_N_CALLS,
    warmup: int = DEFAULT_WARMUP,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load a Keras SavedModel from disk and benchmark its inference latency.

    Convenience wrapper around ``benchmark_inference()`` for the common
    case of benchmarking a saved run directly by path (e.g. the champion at
    ``models/bilstm_hands_only_v4_aug_saved_model/``) without the caller
    needing to load the model separately first.

    Parameters
    ----------
    saved_model_path : str | Path
        Path to a ``tf.keras`` SavedModel directory.
    X_sample : array-like — see ``benchmark_inference()``.
    n_calls, warmup : see ``benchmark_inference()``.
    description : str, optional
        Defaults to the SavedModel directory's name if not supplied.

    Returns
    -------
    dict
        Everything from ``benchmark_inference()``, plus:
            param_count       : int
            model_size_mb     : float  (uncompressed float32 estimate)
            saved_model_path  : str    (resolved absolute path)

    Raises
    ------
    FileNotFoundError
        If ``saved_model_path`` does not exist.
    """
    import tensorflow as tf

    path = Path(saved_model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"benchmark_keras_savedmodel(): SavedModel directory not found: "
            f"{path}. Check the path against models/<run_name>_saved_model/."
        )

    desc = description or path.name
    model = tf.keras.models.load_model(str(path))

    stats = benchmark_inference(model, X_sample, n_calls=n_calls, warmup=warmup, description=desc)
    stats["param_count"]      = int(model.count_params())
    stats["model_size_mb"]    = compute_keras_model_size_mb(model)
    stats["saved_model_path"] = str(path.resolve())
    return stats


# ---------------------------------------------------------------------------
# FeaturePipeline preprocessing benchmarking
# ---------------------------------------------------------------------------

def benchmark_pipeline_preprocessing(
    pipeline: Any,
    raw_landmarks_sample: Any,
    n_calls: int = DEFAULT_N_CALLS,
    warmup: int = DEFAULT_WARMUP,
    description: str = "feature_pipeline_preprocessing",
) -> Dict[str, Any]:
    """
    Benchmark ``FeaturePipeline.__call__(raw_arr, training=False)``: wrist-
    relative normalisation, z-coordinate soft-clip, pad/centre-crop to
    ``seq_len``, and landmark-config selection.

    ``training=False`` is non-negotiable here and matches the inference
    contract enforced everywhere else in this project (Critical Rule #8).
    This function never benchmarks the augmented code path.

    Scope note — this is a PARTIAL per-stage profile
    ----------------------------------------------------
    The handoff's full per-stage timing table (Part 6.2) has seven rows.
    Of these, only "wrist normalisation" + "z-clip+pad" (bundled here as
    ``FeaturePipeline.__call__``) are buildable in Stage 6. Re-run this
    unchanged once Stage 7 / Stage 9 land and append their numbers to the
    same comparison table.

    Parameters
    ----------
    pipeline : FeaturePipeline
        An already-constructed pipeline instance (``FeaturePipeline(cfg)``).
    raw_landmarks_sample : array-like, shape (T_raw, 225)
        One clip's raw, un-pipelined landmark array — NOT the
        (seq_len, feature_dim) output of a previous pipeline call.
    n_calls, warmup : see ``time_callable()``.
    description : str, default "feature_pipeline_preprocessing"

    Returns
    -------
    dict
        Everything from ``time_callable()``, plus:
            raw_input_shape : list[int]
            output_shape    : list[int] | not present if unavailable or None.

    Post-review fix #6: guards against ``pipeline.output_shape is None``
    before calling ``list()``, which would raise ``TypeError``.
    """
    raw_arr = np.asarray(raw_landmarks_sample, dtype=np.float32)

    def _call() -> np.ndarray:
        return np.asarray(pipeline(raw_arr, training=False))

    stats = time_callable(_call, n_calls=n_calls, warmup=warmup, description=description)
    stats["raw_input_shape"] = list(raw_arr.shape)

    # Post-review fix #6: getattr with None default, then guard before list().
    shape = getattr(pipeline, "output_shape", None)
    if shape is not None:
        try:
            stats["output_shape"] = list(shape)
        except TypeError:
            logger.debug(
                f"benchmark_pipeline_preprocessing(): pipeline.output_shape={shape!r} "
                "cannot be converted to list; omitting from result dict.",
                extra={"stage": "evaluation"},
            )

    return stats


# ---------------------------------------------------------------------------
# Scratch (non-final) TFLite conversion — Phase A2 requirement
# ---------------------------------------------------------------------------

def convert_to_scratch_tflite(
    saved_model_dir: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    quantise: bool = True,
) -> str:
    """
    Produce a THROWAWAY dynamic-range TFLite file — for Stage 6 latency
    benchmarking ONLY.

    !!! THIS IS NOT THE STAGE 8 DELIVERABLE !!!
    -----------------------------------------------
    Stage 8 (``src/export/convert.py``) owns the authoritative TFLite
    export: accuracy verified against the full val set, a model metadata
    JSON (``models/gesture_model_metadata.json``), and the file that
    actually ships (``models/gesture_bilstm_v1.tflite``). This function
    exists solely so Stage 6's ``latency_benchmark.png`` can report a
    realistic quantised-model latency number without waiting for Stage 8.

    No accuracy verification is performed here. The default output filename
    is prefixed with an underscore as a "scratch, do not commit" signal.

    Post-review fix #14: conversion wall-clock time is now logged so that
    unexpectedly slow quantisation can be identified immediately.

    Parameters
    ----------
    saved_model_dir : str | Path
        Path to a ``tf.keras`` SavedModel directory.
    output_path : str | Path, optional
        Destination for the ``.tflite`` file. Defaults to
        ``models/_bench_scratch.tflite``.
    quantise : bool, default True
        If True, applies ``tf.lite.Optimize.DEFAULT`` (dynamic-range
        quantisation). If False, produces an unquantised float32 TFLite
        model — useful for isolating TFLite interpreter overhead from
        quantisation overhead.

    Returns
    -------
    str
        Resolved absolute path to the written ``.tflite`` file.

    Raises
    ------
    FileNotFoundError
        If ``saved_model_dir`` does not exist.
    """
    import tensorflow as tf

    src_path = Path(saved_model_dir)
    if not src_path.exists():
        raise FileNotFoundError(
            f"convert_to_scratch_tflite(): SavedModel directory not found: "
            f"{src_path}."
        )

    out_path = Path(output_path) if output_path else Path(_SCRATCH_TFLITE_DEFAULT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    converter = tf.lite.TFLiteConverter.from_saved_model(str(src_path))
    if quantise:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Post-review fix #14: measure and log conversion time.
    t_convert_start = time.perf_counter()
    tflite_bytes    = converter.convert()
    conversion_time_s = time.perf_counter() - t_convert_start

    out_path.write_bytes(tflite_bytes)
    file_size_mb = compute_file_size_mb(out_path)

    logger.warning(
        f"convert_to_scratch_tflite(): wrote SCRATCH/THROWAWAY TFLite file "
        f"→ {out_path.resolve()} "
        f"(quantise={quantise}, size={file_size_mb:.4f} MB, "
        f"conversion_time={conversion_time_s:.2f}s). "
        "This file has NOT been accuracy-verified and is NOT the Stage 8 "
        "deliverable — use it only with benchmark_tflite_inference() for "
        "latency measurement. Stage 8's src/export/convert.py produces the "
        "real, verified export.",
        extra={"stage": "evaluation"},
    )

    return str(out_path.resolve())


# ---------------------------------------------------------------------------
# TFLite interpreter adapter — satisfies metrics.py's "model" contract
# ---------------------------------------------------------------------------

class TFLiteCallable:
    """
    Adapter exposing a ``tf.lite.Interpreter`` through the same
    ``model(x_batch, training=False) -> probs`` call signature that every
    function in ``src/evaluation/metrics.py`` expects of ``model``.

    Why this exists
    ----------------
    ``metrics.py::get_predictions()`` is framework-agnostic by design: it
    accepts "a callable accepting model(x_batch, training=False) and
    returning a (batch, n_classes) array-like." This class is that adapter
    for TFLite. Once Stage 8 produces a verified ``.tflite`` file (or once
    this module's scratch conversion produces one for benchmarking), wrapping
    it in ``TFLiteCallable`` makes every existing ``metrics.py`` function
    work against it with zero code changes in ``metrics.py``.

    Batching behaviour
    -------------------
    TFLite ``Interpreter`` instances built from a SavedModel with a static
    input shape typically fix the batch dimension at conversion time (usually
    1). ``__call__`` loops over ``x_batch``'s batch dimension one sample at
    a time. This is appropriate for CORRECTNESS evaluation (running the full
    52-clip val set through ``get_predictions()``); for LATENCY measurement,
    use ``benchmark_tflite_inference()`` directly, which bypasses this wrapper.

    Shape handling (post-review fix #1)
    ---------------------------------------
    The interpreter's fixed input shape (from ``allocate_tensors()``) is
    stored as ``self.input_shape`` in ``__init__``. On each ``__call__``,
    the per-sample shape is compared against the expected shape. If they
    match, inference proceeds directly. If the model was exported with a
    dynamic batch dimension (shape_signature containing -1 in position 0),
    ``resize_tensor_input`` + ``allocate_tensors`` are called to adapt the
    interpreter, and the result is re-stored. A genuine shape mismatch (wrong
    seq_len or feature_dim) raises ``ValueError`` immediately with a clear
    message rather than propagating a cryptic TFLite error.

    Index caching (post-review fix #11)
    ----------------------------------------
    ``input_index`` and ``output_index`` are cached in ``__init__`` rather
    than re-looked-up on every ``__call__``, avoiding repeated dict access
    into the interpreter's detail list.

    Parameters
    ----------
    tflite_path : str | Path
        Path to a ``.tflite`` file — either the Stage 6 scratch export from
        ``convert_to_scratch_tflite()`` or the verified Stage 8 export.

    Raises
    ------
    FileNotFoundError
        If ``tflite_path`` does not exist.
    ValueError
        If the model does not have exactly one input and one output tensor.
    """

    def __init__(self, tflite_path: Union[str, Path]) -> None:
        import tensorflow as tf

        path = Path(tflite_path)
        if not path.exists():
            raise FileNotFoundError(f"TFLiteCallable: file not found: {path}")

        self._path = path
        self.interpreter = tf.lite.Interpreter(model_path=str(path))
        self.interpreter.allocate_tensors()

        input_details  = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        if len(input_details) != 1 or len(output_details) != 1:
            raise ValueError(
                f"TFLiteCallable: expected exactly one input and one output "
                f"tensor; got {len(input_details)} input(s) and "
                f"{len(output_details)} output(s). This adapter assumes the "
                "single-input, single-output classification signature used "
                "throughout this project (a Dense(n_classes, softmax) head "
                "fed by one landmark-sequence input)."
            )

        self._input_detail  = input_details[0]
        self._output_detail = output_details[0]
        self._input_dtype   = self._input_detail["dtype"]

        # Post-review fix #11: cache indices once in __init__ to avoid
        # repeated dict lookups inside the hot inference path of __call__.
        self._input_index  = self._input_detail["index"]
        self._output_index = self._output_detail["index"]

        # Store whether the model has a dynamic batch dimension (shape[0] == -1).
        # This drives the resize-or-reject decision in __call__.
        raw_shape = tuple(int(d) for d in self._input_detail["shape"])
        self._has_dynamic_batch = (len(raw_shape) > 0 and raw_shape[0] == -1)
        self._fixed_input_shape = raw_shape

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """The interpreter's fixed input shape, e.g. (1, 100, 126)."""
        return self._fixed_input_shape

    @property
    def output_shape(self) -> Tuple[int, ...]:
        """The interpreter's fixed output shape, e.g. (1, 35)."""
        return tuple(int(d) for d in self._output_detail["shape"])

    def __call__(self, x_batch: Any, training: bool = False) -> np.ndarray:
        """
        Run inference for every sample in ``x_batch`` and return a stacked
        ``(batch, n_classes)`` probability array.

        ``training`` is accepted and silently ignored — TFLite inference has
        no notion of a training mode; the graph is already frozen.

        Post-review fix #1: shape validation and conditional resize.
        Each sample's shape (1, seq_len, feature_dim) is compared to the
        interpreter's expected input shape. For static-batch models the shapes
        must match exactly. For dynamic-batch models (shape[0] == -1), the
        interpreter is resized and reallocated on the first call (or whenever
        the sample shape changes from the previous call).
        """
        x = np.asarray(x_batch, dtype=self._input_dtype)
        if x.ndim == 2:
            x = x[np.newaxis, ...]

        outputs = []
        for i in range(x.shape[0]):
            sample = x[i : i + 1]  # shape (1, seq_len, feature_dim)

            expected = self._fixed_input_shape
            actual   = tuple(sample.shape)

            if actual != expected:
                if self._has_dynamic_batch:
                    # Resize interpreter for dynamic-batch models.
                    self.interpreter.resize_tensor_input(self._input_index, list(actual))
                    self.interpreter.allocate_tensors()
                    # Re-cache indices after reallocation (they may change).
                    self._input_index  = self.interpreter.get_input_details()[0]["index"]
                    self._output_index = self.interpreter.get_output_details()[0]["index"]
                    self._fixed_input_shape = actual
                else:
                    raise ValueError(
                        f"TFLiteCallable.__call__(): sample shape {actual} does not "
                        f"match the interpreter's fixed input shape {expected}. "
                        "This model was exported with a static batch dimension and "
                        "cannot be resized. Ensure X_sample was produced by the same "
                        "FeaturePipeline config (sequence_length, landmark_config) "
                        "used to export this .tflite file. Expected for champion: "
                        "(1, 100, 126) — seq_len=100, landmark_config=hands_only."
                    )

            self.interpreter.set_tensor(self._input_index, sample)
            self.interpreter.invoke()
            outputs.append(np.array(self.interpreter.get_tensor(self._output_index))[0])

        return np.stack(outputs, axis=0)


def benchmark_tflite_inference(
    tflite_path: Union[str, Path],
    X_sample: Any,
    n_calls: int = DEFAULT_N_CALLS,
    warmup: int = DEFAULT_WARMUP,
    description: str = "tflite_inference",
) -> Dict[str, Any]:
    """
    Benchmark a ``.tflite`` model's inference latency directly via the
    ``set_tensor`` / ``invoke`` / ``get_tensor`` interpreter API.

    Deliberately bypasses ``TFLiteCallable`` — that adapter's per-sample
    Python loop exists for *correctness* convenience, but would contaminate
    a *latency* measurement. This function always measures single-sample
    (batch=1) latency, matching the real deployment scenario.

    Post-review fix #4: the inner closure now wraps ``get_tensor()`` in
    ``np.asarray()`` for consistent output materialisation semantics with
    the Keras benchmarking path (``benchmark_inference()``).

    Post-review fix #10: a WARNING is logged when ``X_sample`` dtype differs
    from the interpreter's expected input dtype and a silent cast occurs.

    Parameters
    ----------
    tflite_path : str | Path
        Path to a ``.tflite`` file (scratch or verified).
    X_sample : array-like
        A representative input. If batched (batch > 1), only the first
        sample is used.
    n_calls, warmup : see ``time_callable()``.
    description : str, default "tflite_inference"

    Returns
    -------
    dict
        Everything from ``time_callable()``, plus:
            batch_size    : int, always 1
            input_shape   : list[int]
            backend       : str, always "tflite"
            tflite_path   : str (resolved absolute path)
            file_size_mb  : float (on-disk size of the .tflite file)

    Raises
    ------
    FileNotFoundError
        If ``tflite_path`` does not exist.
    ValueError
        If ``X_sample``'s shape does not match the interpreter's fixed
        input shape after truncation to batch=1.
    """
    import tensorflow as tf

    path = Path(tflite_path)
    if not path.exists():
        raise FileNotFoundError(
            f"benchmark_tflite_inference(): file not found: {path}."
        )

    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    input_detail  = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    expected_dtype = input_detail["dtype"]

    # Post-review fix #10: warn on silent dtype cast.
    X_raw = np.asarray(X_sample)
    if X_raw.dtype != expected_dtype:
        logger.warning(
            f"benchmark_tflite_inference('{description}'): X_sample dtype "
            f"'{X_raw.dtype}' differs from interpreter's expected dtype "
            f"'{expected_dtype}'. A silent cast will occur. "
            "Verify that this cast is intentional and does not hide a "
            "data-pipeline misconfiguration (e.g. passing float64 data "
            "to a float32 model).",
            extra={"stage": "evaluation"},
        )

    X = X_raw.astype(expected_dtype)
    if X.ndim == 2:
        X = X[np.newaxis, ...]

    if X.shape[0] != 1:
        logger.info(
            f"benchmark_tflite_inference('{description}'): X_sample "
            f"batch_size={X.shape[0]} truncated to 1. TFLite single-"
            "inference latency (batch=1) is the deployment-realistic "
            "measurement — Stage 9's webcam demo processes one rolling "
            "sequence window per frame, never a batch.",
            extra={"stage": "evaluation"},
        )
        X = X[:1]

    expected_shape = tuple(int(d) for d in input_detail["shape"])
    if tuple(X.shape) != expected_shape:
        raise ValueError(
            f"benchmark_tflite_inference(): X_sample shape {X.shape} does "
            f"not match the TFLite model's fixed input shape "
            f"{expected_shape}. Check that X_sample was produced by the "
            "same FeaturePipeline config (sequence_length, landmark_config) "
            "used to export this .tflite file. "
            "For the champion: expected (1, 100, 126) — seq_len=100, "
            "landmark_config=hands_only."
        )

    input_index  = input_detail["index"]
    output_index = output_detail["index"]

    def _call() -> np.ndarray:
        interpreter.set_tensor(input_index, X)
        interpreter.invoke()
        # Post-review fix #4: wrap get_tensor() in np.asarray() for
        # consistent output materialisation semantics with the Keras path.
        return np.asarray(interpreter.get_tensor(output_index))

    stats = time_callable(_call, n_calls=n_calls, warmup=warmup, description=description)
    stats["batch_size"]   = 1
    stats["input_shape"]  = list(X.shape)
    stats["backend"]      = "tflite"
    stats["tflite_path"]  = str(path.resolve())
    stats["file_size_mb"] = compute_file_size_mb(path)
    return stats


# ---------------------------------------------------------------------------
# Model size helpers
# ---------------------------------------------------------------------------

def compute_keras_model_size_mb(model: Any) -> float:
    """
    Estimate the uncompressed float32 weight size of a Keras model, in MB.

    Uses ``total_params * 4 bytes / 1024**2`` — the identical formula used
    by ``architectures.py::_log_model_summary()`` and
    ``factory.py::get_model_summary_dict()``. Kept numerically identical to
    those two call sites deliberately: a size reported here must never
    diverge from the size already logged to MLflow / ``run_manifest.json``
    for the same model.

    Note: this estimates the WEIGHT storage size, not the full on-disk
    SavedModel size (which includes computation graphs, assets, and metadata).
    For the actual on-disk size of a file, use ``compute_file_size_mb()``.

    Parameters
    ----------
    model : tf.keras.Model

    Returns
    -------
    float, MB, rounded to 4 decimal places.
    """
    total_params = int(model.count_params())
    return round(total_params * _BYTES_PER_FLOAT32 / (1024 ** 2), 4)


def compute_file_size_mb(path: Union[str, Path]) -> float:
    """
    Return the on-disk size of a file in MB.

    Used for ``.tflite`` files, whose size already reflects quantisation
    and therefore cannot be derived from ``param_count`` alone the way an
    uncompressed Keras model's can.

    Parameters
    ----------
    path : str | Path

    Returns
    -------
    float, MB, rounded to 4 decimal places.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    """
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"compute_file_size_mb(): file not found: {p}")
    return round(p.stat().st_size / (1024 ** 2), 4)


# ---------------------------------------------------------------------------
# Multi-model comparison
# ---------------------------------------------------------------------------

def benchmark_model_registry(
    model_paths: Mapping[str, Union[str, Path]],
    X_sample: Any,
    n_calls: int = DEFAULT_N_CALLS,
    warmup: int = DEFAULT_WARMUP,
) -> Dict[str, Dict[str, Any]]:
    """
    Benchmark several SavedModel directories under one shared input sample.

    Produces the per-model data behind the handoff's "Model size comparison
    table" (Part 6.2: Dense / LSTM / GRU / BiLSTM) and the
    ``reports/figures/latency_benchmark.png`` figure built later in Phase D.

    Memory management (post-review fix #2)
    -----------------------------------------
    After each model is benchmarked, BOTH ``tf.keras.backend.clear_session()``
    AND ``gc.collect()`` are called. ``clear_session()`` releases TensorFlow's
    C++-side allocator pools (graph caches, kernel caches) that Python's GC
    cannot see. Without this, benchmarking many models in sequence causes
    steadily growing memory consumption that can eventually OOM.

    A single model failing to load or run does NOT abort the whole
    comparison; that model's entry gets an ``"error"`` key instead, and
    benchmarking continues for the rest.

    Parameters
    ----------
    model_paths : Mapping[str, str | Path]
        e.g. ``{"dense": "models/dense_baseline_saved_model",
                "bilstm": "models/bilstm_hands_only_v4_aug_saved_model"}``.
    X_sample : array-like
        Shared input. Must match every model's expected input shape — if a
        model in the registry was trained on a different
        ``(seq_len, feature_dim)`` than ``X_sample`` provides, that model's
        entry will contain an ``"error"`` rather than aborting the run.
        Benchmark such a model separately via ``benchmark_keras_savedmodel()``
        with its own correctly-shaped sample.
    n_calls, warmup : see ``time_callable()``.

    Returns
    -------
    dict[str, dict]
        Keyed identically to ``model_paths``. Each value is either the
        dict from ``benchmark_keras_savedmodel()``, or
        ``{"error": "...", "model_path": "..."}`` on failure.
    """
    import tensorflow as tf

    results: Dict[str, Dict[str, Any]] = {}

    for name, path in model_paths.items():
        try:
            stats = benchmark_keras_savedmodel(
                path, X_sample, n_calls=n_calls, warmup=warmup, description=name,
            )
            results[name] = stats
        except Exception as exc:
            logger.error(
                f"benchmark_model_registry(): '{name}' ({path}) failed: "
                f"{type(exc).__name__}: {exc}. Skipping — other models will "
                "still be benchmarked.",
                extra={"stage": "evaluation"},
            )
            results[name] = {
                "error":      f"{type(exc).__name__}: {exc}",
                "model_path": str(path),
            }
        finally:
            # Post-review fix #2: call clear_session() to release TF's C++-side
            # allocator pools (graph caches, kernel caches) that gc.collect()
            # alone cannot reach. This bounds peak memory to roughly one model
            # at a time when benchmarking a large registry.
            try:
                tf.keras.backend.clear_session()
            except Exception as clear_exc:
                logger.debug(
                    f"benchmark_model_registry(): clear_session() failed for "
                    f"'{name}': {clear_exc}. Continuing.",
                    extra={"stage": "evaluation"},
                )
            gc.collect()

    n_failed = sum(1 for v in results.values() if "error" in v)
    logger.info(
        f"benchmark_model_registry() complete | "
        f"n_models={len(model_paths)} | n_failed={n_failed}",
        extra={"stage": "evaluation"},
    )
    return results


def build_latency_comparison_rows(
    results: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Flatten a ``benchmark_model_registry()`` (or similarly-shaped) result
    into row-dicts suitable for a ``pandas.DataFrame`` or a Phase D bar
    chart (``reports/figures/latency_benchmark.png``).

    Failed entries (containing an ``"error"`` key) are included with
    ``None`` numeric fields rather than silently dropped, so a chart/table
    consumer can see which models failed to benchmark instead of the
    comparison quietly shrinking by one row.

    Rows are sorted by ``median_ms`` ascending (fastest first), with failed
    entries (``median_ms is None``) sorted to the end.

    Parameters
    ----------
    results : Mapping[str, Mapping[str, Any]]
        Typically the return value of ``benchmark_model_registry()``.

    Returns
    -------
    list[dict]
        One dict per model with keys: name, median_ms, p95_ms, p99_ms, fps,
        param_count, model_size_mb, backend, error.
    """
    rows: List[Dict[str, Any]] = []
    for name, stats in results.items():
        if "error" in stats:
            rows.append({
                "name":          name,
                "median_ms":     None,
                "p95_ms":        None,
                "p99_ms":        None,
                "fps":           None,
                "param_count":   None,
                "model_size_mb": None,
                "backend":       None,
                "error":         stats["error"],
            })
            continue

        rows.append({
            "name":          name,
            "median_ms":     stats.get("median_ms"),
            "p95_ms":        stats.get("p95_ms"),
            "p99_ms":        stats.get("p99_ms"),
            "fps":           stats.get("fps"),
            "param_count":   stats.get("param_count"),
            "model_size_mb": stats.get("model_size_mb", stats.get("file_size_mb")),
            "backend":       stats.get("backend", "unknown"),
            "error":         None,
        })

    rows.sort(key=lambda r: (r["median_ms"] is None, r["median_ms"] or float("inf")))
    return rows


# ---------------------------------------------------------------------------
# Consolidated champion summary — Keras vs. scratch TFLite
# ---------------------------------------------------------------------------

def benchmark_champion_summary(
    keras_model: Any,
    X_sample: Any,
    tflite_path: Optional[Union[str, Path]] = None,
    n_calls: int = DEFAULT_N_CALLS,
    warmup: int = DEFAULT_WARMUP,
) -> Dict[str, Any]:
    """
    One-call convenience: benchmark the Keras champion and (optionally) a
    TFLite export side-by-side, with the relative speed-up — exactly the
    comparison ``reports/figures/latency_benchmark.png`` needs.

    Deliberately does NOT call ``convert_to_scratch_tflite()`` itself: the
    caller decides when to pay that one-time conversion cost (e.g. once at
    the top of Notebook 06's evaluation cell, not on every call to this
    summary function during iterative development).

    Post-review fix #7: the speedup computation now guards against
    non-finite values in either median_ms before dividing, preventing
    ZeroDivisionError and NaN propagation into the result dict.

    Parameters
    ----------
    keras_model : tf.keras.Model
        Already-loaded champion model. Must not be None.
    X_sample : array-like, shape (seq_len, feature_dim)
        A representative input. For the champion: shape (100, 126).
    tflite_path : str | Path, optional
        Path to a scratch or verified ``.tflite`` file. If omitted, only
        the Keras side of the comparison is computed.
    n_calls, warmup : see ``time_callable()``.

    Returns
    -------
    dict with keys:
        keras             : dict — see ``benchmark_inference()``, plus
                             param_count / model_size_mb.
        tflite             : dict, only if tflite_path was supplied — see
                             ``benchmark_tflite_inference()``.
        speedup_median_x   : float, only if tflite_path was supplied AND
                             both median values are finite and positive —
                             keras median_ms / tflite median_ms. Values > 1
                             mean TFLite is faster.

    Raises
    ------
    ValueError
        If ``keras_model`` is None.
    """
    if keras_model is None:
        raise ValueError(
            "benchmark_champion_summary(): keras_model is None. "
            "Provide an already-loaded tf.keras.Model instance, e.g. "
            "tf.keras.models.load_model('models/bilstm_hands_only_v4_aug_saved_model/')."
        )

    keras_stats = benchmark_inference(
        keras_model, X_sample, n_calls=n_calls, warmup=warmup, description="champion_keras",
    )
    keras_stats["param_count"]   = int(keras_model.count_params())
    keras_stats["model_size_mb"] = compute_keras_model_size_mb(keras_model)

    summary: Dict[str, Any] = {"keras": keras_stats}

    if tflite_path is not None:
        tflite_stats = benchmark_tflite_inference(
            tflite_path, X_sample, n_calls=n_calls, warmup=warmup, description="champion_tflite",
        )
        summary["tflite"] = tflite_stats

        # Post-review fix #7: guard against non-finite / zero median values
        # before computing speedup to prevent ZeroDivisionError and NaN.
        keras_median  = keras_stats.get("median_ms", 0.0) or 0.0
        tflite_median = tflite_stats.get("median_ms", 0.0) or 0.0

        if (
            np.isfinite(keras_median)
            and np.isfinite(tflite_median)
            and tflite_median > 0.0
        ):
            summary["speedup_median_x"] = round(keras_median / tflite_median, 3)
        else:
            logger.warning(
                f"benchmark_champion_summary(): cannot compute speedup — "
                f"keras_median={keras_median}ms, tflite_median={tflite_median}ms "
                "are not both finite positive values.",
                extra={"stage": "evaluation"},
            )
            summary["speedup_median_x"] = None

    log_msg = f"benchmark_champion_summary() | keras_median={keras_stats['median_ms']:.3f}ms"
    if tflite_path is not None and "tflite" in summary:
        log_msg += (
            f" | tflite_median={summary['tflite']['median_ms']:.3f}ms | "
            f"speedup={summary.get('speedup_median_x', 'n/a')}x"
        )
    logger.info(log_msg, extra={"stage": "evaluation"})

    return summary


# ---------------------------------------------------------------------------
# Import-time self-check
# ---------------------------------------------------------------------------

def _self_check() -> None:
    """
    Cheap, dependency-free sanity check on module constants, mirroring the
    pattern used in ``metrics.py`` and ``architectures.py``.
    """
    assert DEFAULT_N_CALLS >= _MIN_CALLS_FOR_STABLE_PERCENTILES, (
        f"benchmark.py: DEFAULT_N_CALLS={DEFAULT_N_CALLS} should be >= "
        f"_MIN_CALLS_FOR_STABLE_PERCENTILES={_MIN_CALLS_FOR_STABLE_PERCENTILES} "
        "so the project default never itself triggers the small-sample warning."
    )
    assert DEFAULT_WARMUP >= 0, "DEFAULT_WARMUP must be non-negative."
    assert DEFAULT_WARMUP < DEFAULT_N_CALLS, (
        f"DEFAULT_WARMUP={DEFAULT_WARMUP} must be strictly less than "
        f"DEFAULT_N_CALLS={DEFAULT_N_CALLS}."
    )
    assert _BYTES_PER_FLOAT32 == 4, (
        "_BYTES_PER_FLOAT32 must be 4 to match architectures.py and factory.py."
    )


if __debug__:
    _self_check()


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------

__all__ = [
    "DEFAULT_N_CALLS",
    "DEFAULT_WARMUP",
    "time_callable",
    "benchmark_inference",
    "benchmark_keras_savedmodel",
    "benchmark_pipeline_preprocessing",
    "convert_to_scratch_tflite",
    "TFLiteCallable",
    "benchmark_tflite_inference",
    "compute_keras_model_size_mb",
    "compute_file_size_mb",
    "benchmark_model_registry",
    "build_latency_comparison_rows",
    "benchmark_champion_summary",
]