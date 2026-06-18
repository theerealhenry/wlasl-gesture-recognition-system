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
     latency, or the full webcam per-frame pipeline (handoff Part 6.2's
     "Frame capture ~2ms / MediaPipe ~18ms / ... / Overlay ~3ms" table).
     Those stages live in ``src/inference/predictor.py`` (Stage 7) and
     ``src/demo/webcam_demo.py`` (Stage 9), neither of which exists yet.
     ``benchmark_pipeline_preprocessing()`` below benchmarks the one
     pre-existing, already-built stage (``FeaturePipeline.__call__``) so
     that table can be filled in incrementally as later stages land,
     rather than faking numbers for stages that don't exist.

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
    deployed app pays exactly once at startup, never again: TensorFlow's
    eager-mode op dispatch caches, the OS's allocator growing its arena to
    a steady-state working set, and (for TFLite) the interpreter's internal
    buffer allocation. Counting these in the timed distribution would
    inflate p99 and max with a cost no real prediction after frame 1 ever
    pays again. ``warmup=20`` (project default) is enough for all of these
    to stabilise without materially lengthening a benchmark run.

Input tensors are pre-converted ONCE, outside the timed loop
    ``benchmark_inference()`` converts ``X_sample`` to a ``tf.constant``
    exactly once, before entering the timing loop. Converting a numpy array
    to a ``tf.Tensor`` inside the loop would add Python-level marshalling
    overhead to every single measurement — overhead that has nothing to do
    with the model's actual forward-pass cost, and that a real deployment
    (Stage 7's ``GesturePredictor``, holding a pre-allocated rolling buffer)
    would never pay per-prediction either. The same principle applies to
    ``benchmark_tflite_inference()``: the input array is built once, and
    only ``set_tensor`` / ``invoke`` / ``get_tensor`` are inside the loop.

Output is forced to materialise inside the timed region
    ``np.asarray(raw_output)`` inside the timed closure blocks until the
    EagerTensor's value is actually available, ensuring the timed region
    captures the complete forward pass rather than just kernel-dispatch
    latency. (On the CPU-only target this project trains on, TensorFlow's
    eager ops are synchronous either way — this is a defensive habit, not
    a correctness fix specific to this codebase, and it costs nothing.)

FPS is derived from MEDIAN latency, not from (total_time / n_calls)
    Median is robust to the occasional GC pause or OS scheduling hiccup
    that the mean (and "naive total/n_calls FPS") would be dragged around
    by. p95/p99 are reported separately precisely so that tail latency is
    visible without being allowed to distort the single headline FPS number.

n_calls=200, warmup=20 are defaults, not hard-coded
    Every function accepts ``n_calls`` / ``warmup`` explicitly (matching
    the exact signature specified in the handoff, Part 6.2:
    ``benchmark_inference(model, X_sample, n_calls=200, warmup=20)``).
    A warning (not an error) fires below ``_MIN_CALLS_FOR_STABLE_PERCENTILES``
    (30) because p99 of a 30-sample distribution is, in effect, just the
    max — still useful for a fast dev-loop smoke test, just not citable in
    the Stage 11 report.

Module-level exports
---------------------
    time_callable                    — generic 0-arg-callable timing harness
    benchmark_inference               — time a Keras model's forward pass
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
#: is a deliberate naming signal — "this is scratch, not a deliverable" —
#: mirroring the Stage 6 (Revised) plan's own naming
#: ("models/_bench_champion.tflite ... gitignored or clearly marked scratch").
_SCRATCH_TFLITE_DEFAULT_PATH: str = "models/_bench_scratch.tflite"

#: Bytes-per-parameter for an uncompressed float32 Keras weight tensor.
#: Matches the identical constant used in architectures.py::_log_model_summary
#: and factory.py::get_model_summary_dict — kept in sync deliberately so a
#: size reported by this module is never numerically different from the
#: size already logged to MLflow / run_manifest.json for the same model.
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

    Accepts a single un-batched sample ``(seq_len, feature_dim)`` — the
    common case when a caller pulls one clip out of
    ``FeaturePipeline(arr, training=False)`` — and promotes it to a batch
    of 1. Anything that is not 2-D or 3-D after this is almost certainly
    the wrong array (e.g. raw un-pipelined landmarks at 225 dims when the
    model expects hands_only at 126), so it is rejected immediately rather
    than allowed to fail deep inside ``model.__call__`` with a cryptic
    TensorFlow shape error.

    Parameters
    ----------
    X : array-like
    caller : str — used only in the error message.

    Returns
    -------
    np.ndarray, dtype float32, ndim == 3
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 2:
        X = X[np.newaxis, ...]
    elif X.ndim != 3:
        raise ValueError(
            f"{caller}: X_sample has shape {X.shape}; expected either "
            "(seq_len, feature_dim) for a single sample or "
            "(batch, seq_len, feature_dim) for a batch. Check that this "
            "array came from FeaturePipeline output "
            "(e.g. pipeline(raw_arr, training=False)), not raw, "
            "un-pipelined landmarks."
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
    this module is built on top of (``benchmark_inference()``,
    ``benchmark_tflite_inference()``, ``benchmark_pipeline_preprocessing()``
    are all thin wrappers that build a closure and hand it to this
    function). Centralising the actual ``time.perf_counter()`` loop here
    guarantees every latency number reported anywhere in Stage 6 was
    measured the same way — same warmup discard policy, same percentile
    definitions, same FPS derivation.

    ``fn`` takes no arguments and is expected to be a closure that already
    captures everything it needs (a pre-converted input tensor, a bound
    model, etc.) — see ``benchmark_inference()`` for the canonical example
    of why pre-binding matters for measurement validity.

    Parameters
    ----------
    fn : Callable[[], Any]
        Zero-argument callable to time. Its return value is discarded here
        (callers are responsible for ensuring ``fn`` itself forces any lazy
        computation to materialise before returning — see
        ``benchmark_inference()``'s use of ``np.asarray()``).
    n_calls : int, default 200
        Number of TIMED invocations. Project default per the handoff spec
        (Part 6.2).
    warmup : int, default 20
        Number of UNTIMED invocations executed first and discarded. Absorbs
        one-time JIT/cache/allocator-growth costs that a deployed app pays
        once at startup, never per-prediction.
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

    median_ms = float(np.percentile(durations_ms, 50))
    p95_ms    = float(np.percentile(durations_ms, 95))
    p99_ms    = float(np.percentile(durations_ms, 99))
    mean_ms   = float(np.mean(durations_ms))
    std_ms    = float(np.std(durations_ms, ddof=1)) if n_calls > 1 else 0.0
    min_ms    = float(np.min(durations_ms))
    max_ms    = float(np.max(durations_ms))

    # FPS from median (robust) latency, not from total_time_s / n_calls —
    # the latter is dragged around by any single slow call (GC pause, OS
    # scheduling hiccup) in exactly the way p95/p99 are designed to expose
    # rather than hide inside one blended number.
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

def benchmark_inference(
    model: Any,
    X_sample: Any,
    n_calls: int = DEFAULT_N_CALLS,
    warmup: int = DEFAULT_WARMUP,
    description: str = "keras_inference",
) -> Dict[str, Any]:
    """
    Benchmark a Keras model's forward-pass latency on a fixed input sample.

    Matches the exact signature specified in the handoff (Part 6.2):
    ``benchmark_inference(model, X_sample, n_calls=200, warmup=20)``. Built
    and tested directly against the champion Keras SavedModel — no
    dependency on the Phase B1 prediction cache (``val_predictions.npz``);
    this function only needs a model and one representative input array.

    Measurement validity — why the tensor is converted ONCE, outside the loop
    --------------------------------------------------------------------------
    ``X_sample`` is converted to a ``tf.constant`` exactly once before
    entering ``time_callable()``'s timed region. Re-converting a numpy
    array to a ``tf.Tensor`` on every call would add Python-level
    marshalling overhead to every single measurement — overhead a real
    deployment never pays per-prediction either (Stage 7's
    ``GesturePredictor`` will hold a pre-allocated rolling buffer, not
    reconstruct a tensor from scratch per frame).

    The model's raw output is forced to materialise
    (``np.asarray(raw_output)``) inside the timed closure, so the measured
    region captures the complete forward pass rather than just eager-mode
    kernel dispatch.

    Parameters
    ----------
    model : tf.keras.Model (or any callable satisfying model(x, training=False))
        Must accept the call signature ``model(x_tensor, training=False)``
        and return something array-convertible — identical contract to
        ``metrics.py::get_predictions()``'s ``model`` parameter. This means
        a pre-loaded champion SavedModel, a freshly-built (untrained) model
        from ``build_model()``, or anything else satisfying the contract
        all work unmodified.
    X_sample : array-like, shape (seq_len, feature_dim) or (batch, seq_len, feature_dim)
        A representative input. For the champion (per
        artifacts/experiments/bilstm_hands_only_v4_aug/config_snapshot.yaml:
        sequence_length=100, landmark_config=hands_only), this is shape
        (100, 126) for a single sample. A 2-D input is automatically
        promoted to a batch of 1.
    n_calls : int, default 200
    warmup : int, default 20
    description : str, default "keras_inference"
        Label echoed into the result dict and log line — pass the run name
        (e.g. "bilstm_hands_only_v4_aug") when benchmarking a specific run
        so results remain distinguishable in a multi-model comparison.

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
        (batch, seq_len, feature_dim) — see ``_ensure_batched_float_array()``.
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

    # Pre-convert ONCE, outside the timed loop — see docstring above.
    x_tensor = tf.constant(X, dtype=tf.float32)

    def _call() -> np.ndarray:
        raw_output = model(x_tensor, training=False)
        # Force materialisation — see docstring "Measurement validity".
        return np.asarray(raw_output)

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
        Defaults to the SavedModel directory's name (e.g.
        "bilstm_hands_only_v4_aug_saved_model") if not supplied.

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
    contract enforced everywhere else in this project (Critical Rule #8,
    handoff Part 8: "training=False at inference — FeaturePipeline and
    GesturePredictor must never apply augmentation at inference."). This
    function never benchmarks the augmented code path — that would measure
    something no deployed prediction ever does.

    Scope note — this is a PARTIAL per-stage profile
    ----------------------------------------------------
    The handoff's full per-stage timing table (Part 6.2) has seven rows:
    frame capture, MediaPipe extraction, wrist normalisation, z-clip+pad,
    inference, prediction smoothing, overlay rendering. Of these, only
    "wrist normalisation" + "z-clip+pad" (bundled together here as
    ``FeaturePipeline.__call__``) are buildable in Stage 6 — the rest
    require ``src/inference/predictor.py`` (Stage 7) and
    ``src/demo/webcam_demo.py`` (Stage 9), neither of which exists yet.
    Re-run this same function unchanged once those stages land and append
    their numbers to the same comparison table rather than estimating them
    here.

    Parameters
    ----------
    pipeline : FeaturePipeline
        An already-constructed pipeline instance (``FeaturePipeline(cfg)``).
    raw_landmarks_sample : array-like, shape (T_raw, 225)
        One clip's raw, un-pipelined landmark array (i.e. straight from a
        ``.npy`` file in ``data/landmarks/``, before any pipeline
        processing) — NOT the (seq_len, feature_dim) output of a previous
        pipeline call. ``benchmark_inference()`` is what times the model on
        already-pipelined input; this function times the pipeline itself.
    n_calls, warmup : see ``time_callable()``.
    description : str, default "feature_pipeline_preprocessing"

    Returns
    -------
    dict
        Everything from ``time_callable()``, plus:
            raw_input_shape : list[int]
            output_shape    : list[int], only if ``pipeline.output_shape``
                               is available.
    """
    raw_arr = np.asarray(raw_landmarks_sample, dtype=np.float32)

    def _call() -> np.ndarray:
        return np.asarray(pipeline(raw_arr, training=False))

    stats = time_callable(_call, n_calls=n_calls, warmup=warmup, description=description)
    stats["raw_input_shape"] = list(raw_arr.shape)
    if hasattr(pipeline, "output_shape"):
        stats["output_shape"] = list(pipeline.output_shape)
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
    export: accuracy verified against the full val set
    (``cfg.export.max_accuracy_delta = 0.03``), a model metadata JSON
    (``models/gesture_model_metadata.json``), and the file that actually
    ships (``models/gesture_bilstm_v1.tflite``). This function exists
    solely so Stage 6's ``latency_benchmark.png`` can report a realistic
    quantised-model latency number without waiting for Stage 8 — per the
    Stage 6 (Revised) plan's Phase A2: "Also do the throwaway dynamic-range
    TFLite conversion here ... purely for benchmarking — clearly commented
    as non-final, superseded by Stage 8's real export."

    No accuracy verification is performed here, and none should be
    inferred from the fact that this function ran successfully — a model
    can convert cleanly and still have degraded post-quantisation accuracy.
    The default output filename is prefixed with an underscore
    (``models/_bench_scratch.tflite``) as a deliberate "this is scratch, do
    not commit, do not deploy" signal; add it to ``.gitignore`` if it isn't
    already covered by a ``models/_*`` pattern.

    Parameters
    ----------
    saved_model_dir : str | Path
        Path to a ``tf.keras`` SavedModel directory (e.g.
        ``models/bilstm_hands_only_v4_aug_saved_model/``).
    output_path : str | Path, optional
        Destination for the ``.tflite`` file. Defaults to
        ``models/_bench_scratch.tflite``.
    quantise : bool, default True
        If True, applies ``tf.lite.Optimize.DEFAULT`` (dynamic-range
        quantisation — no representative dataset required, matching
        ``cfg.export.quantisation_mode = "dynamic_range"``). If False,
        produces an unquantised float32 TFLite model — useful for isolating
        "TFLite interpreter overhead" from "quantisation overhead" when
        interpreting the benchmark comparison.

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

    tflite_bytes = converter.convert()
    out_path.write_bytes(tflite_bytes)

    logger.warning(
        f"convert_to_scratch_tflite(): wrote SCRATCH/THROWAWAY TFLite file "
        f"→ {out_path.resolve()} (quantise={quantise}, "
        f"size={compute_file_size_mb(out_path):.4f} MB). This file has NOT "
        "been accuracy-verified and is NOT the Stage 8 deliverable — use it "
        "only with benchmark_tflite_inference() for latency measurement. "
        "Stage 8's src/export/convert.py produces the real, verified export.",
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
    ``metrics.py::get_predictions()`` documents itself as framework-agnostic
    by design: "model: Any — a callable accepting model(x_batch,
    training=False) and returning a (batch, n_classes) array-like" — and
    explicitly states this design is what would let it be "reused unmodified
    ... for the Stage 8 TFLite interpreter — see benchmark.py". This class
    is that adapter. Once Stage 8 produces a verified ``.tflite`` file (or
    once this module's own scratch conversion produces one for benchmarking),
    wrapping it in ``TFLiteCallable`` makes every existing ``metrics.py``
    function — ``get_predictions``, ``compute_confusion_matrix``,
    ``compute_evaluation_summary`` — work against it with zero code changes
    in ``metrics.py``.

    Batching behaviour
    -------------------
    TFLite ``Interpreter`` instances built from a SavedModel with a static
    input shape typically fix the batch dimension at conversion time
    (usually 1). Rather than require every caller to know and handle this,
    ``__call__`` loops over ``x_batch``'s batch dimension one sample at a
    time, feeding each through ``set_tensor`` / ``invoke`` / ``get_tensor``
    individually and stacking results back into one ``(batch, n_classes)``
    array. This adds a small amount of Python-loop overhead that is
    irrelevant for *correctness* evaluation (running the full 52-clip val
    set through ``get_predictions()``) but would contaminate a *latency*
    measurement — which is exactly why ``benchmark_tflite_inference()``
    below times the raw interpreter calls directly and does NOT route
    through this wrapper.

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
        If the model does not have exactly one input and one output
        tensor (this project's architectures all use a single
        ``Dense(n_classes, softmax)`` head fed by a single landmark-sequence
        input — a multi-input/output model would indicate a different,
        unsupported export).
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

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """The interpreter's fixed input shape, e.g. (1, 100, 126)."""
        return tuple(int(d) for d in self._input_detail["shape"])

    @property
    def output_shape(self) -> Tuple[int, ...]:
        """The interpreter's fixed output shape, e.g. (1, 35)."""
        return tuple(int(d) for d in self._output_detail["shape"])

    def __call__(self, x_batch: Any, training: bool = False) -> np.ndarray:
        """
        Run inference for every sample in ``x_batch`` and return a stacked
        ``(batch, n_classes)`` probability array.

        ``training`` is accepted and silently ignored — it exists purely so
        this object satisfies the same call signature as a Keras model.
        TFLite inference has no notion of a training mode; the graph is
        already frozen at conversion time.
        """
        x = np.asarray(x_batch, dtype=self._input_dtype)
        if x.ndim == 2:
            x = x[np.newaxis, ...]

        input_index  = self._input_detail["index"]
        output_index = self._output_detail["index"]

        outputs = []
        for i in range(x.shape[0]):
            sample = x[i : i + 1]
            self.interpreter.set_tensor(input_index, sample)
            self.interpreter.invoke()
            outputs.append(np.array(self.interpreter.get_tensor(output_index))[0])

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
    Python loop exists for *correctness* convenience when evaluating a full
    val/test split (where a few microseconds of loop overhead is
    irrelevant), but would contaminate a *latency* measurement, which needs
    to isolate the interpreter's own per-call cost. This function always
    measures single-sample (batch=1) latency, matching the real deployment
    scenario: Stage 9's webcam demo processes one rolling sequence window
    at a time, never a batch.

    Parameters
    ----------
    tflite_path : str | Path
        Path to a ``.tflite`` file (scratch or verified).
    X_sample : array-like
        A representative input. If batched (batch > 1), only the first
        sample is used and a log message records this — see the
        "single-sample latency" rationale above.
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

    X = np.asarray(X_sample, dtype=input_detail["dtype"])
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
            "used to export this .tflite file."
        )

    input_index  = input_detail["index"]
    output_index = output_detail["index"]

    def _call() -> np.ndarray:
        interpreter.set_tensor(input_index, X)
        interpreter.invoke()
        return interpreter.get_tensor(output_index)

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
    for the same model, or a reviewer comparing the two would (reasonably)
    suspect a bug rather than a unit/rounding difference.

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
    p = Path(path)
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

    Each model is loaded, benchmarked, and explicitly garbage-collected
    before the next is loaded — bounding peak memory to roughly one model
    at a time, which matters once this is pointed at the full 23-run
    registry rather than just the four Group-1 architecture-comparison
    candidates.

    A single model failing to load or run (most commonly: a shape mismatch,
    since different Stage 5 groups used different ``seq_len`` /
    ``landmark_config`` combinations — Group 1 used full/225-dim/seq60,
    the champion uses hands_only/126-dim/seq100) does NOT abort the whole
    comparison; that model's entry gets an ``"error"`` key instead, and
    benchmarking continues for the rest.

    Parameters
    ----------
    model_paths : Mapping[str, str | Path]
        e.g. ``{"dense": "models/dense_baseline_saved_model", "lstm": ...,
        "gru": ..., "bilstm": "models/bilstm_hands_only_v4_aug_saved_model"}``.
    X_sample : array-like
        Shared input. Must match every model's expected input shape — if a
        model in the registry was trained on a different
        ``(seq_len, feature_dim)`` than ``X_sample`` provides, that model's
        entry will contain an ``"error"`` rather than aborting the run.
        Benchmark such a model separately via ``benchmark_keras_savedmodel()``
        with its own correctly-shaped sample instead.
    n_calls, warmup : see ``time_callable()``.

    Returns
    -------
    dict[str, dict]
        Keyed identically to ``model_paths``. Each value is either the
        dict from ``benchmark_keras_savedmodel()``, or
        ``{"error": "...", "model_path": "..."}`` on failure.
    """
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
            # Bound peak memory across a potentially large registry —
            # explicit, not relying on Python's refcounting alone, since
            # TF Keras models can hold C++-side resources GC doesn't see.
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

    rows.sort(key=lambda r: (r["median_ms"] is None, r["median_ms"]))
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

    Parameters
    ----------
    keras_model : tf.keras.Model
        Already-loaded champion model, e.g.
        ``tf.keras.models.load_model("models/bilstm_hands_only_v4_aug_saved_model/")``.
    X_sample : array-like, shape (seq_len, feature_dim)
        A representative input. For the champion: shape (100, 126).
    tflite_path : str | Path, optional
        Path to a scratch (this module's ``convert_to_scratch_tflite()``)
        or verified (Stage 8) ``.tflite`` file. If omitted, only the Keras
        side of the comparison is computed.
    n_calls, warmup : see ``time_callable()``.

    Returns
    -------
    dict with keys:
        keras             : dict — see benchmark_inference(), plus
                             param_count / model_size_mb.
        tflite             : dict, only if tflite_path was supplied — see
                             benchmark_tflite_inference().
        speedup_median_x   : float, only if tflite_path was supplied —
                             keras median_ms / tflite median_ms. Values > 1
                             mean TFLite is faster, the expected direction
                             once dynamic-range quantisation and the
                             lighter-weight interpreter runtime are in play.
    """
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
        if tflite_stats["median_ms"] > 0:
            summary["speedup_median_x"] = round(
                keras_stats["median_ms"] / tflite_stats["median_ms"], 3
            )

    log_msg = f"benchmark_champion_summary() | keras_median={keras_stats['median_ms']:.3f}ms"
    if tflite_path is not None:
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
    assert DEFAULT_WARMUP >= 0
    assert _BYTES_PER_FLOAT32 == 4


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