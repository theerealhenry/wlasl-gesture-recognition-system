"""
src/export/convert.py
=======================
Stage 8, Step 1 — Authoritative TFLite export for the WLASL 35-class
gesture recognition champion (bilstm_hands_only_v4_aug, val macro-F1=0.6011).

This module produces the deployment artefact — models/gesture_bilstm_v1.tflite —
that Stage 9's webcam demo and GesturePredictor (Stage 7) load via
model_path="*.tflite". It is deliberately NOT the same code path as
src/evaluation/benchmark.py's convert_to_scratch_tflite() — that function
exists purely to get a realistic latency number into Stage 6's figures and
explicitly disclaims production status. This file is the production status.

Why this file is shaped the way it is (grounded in this project's actual state)
--------------------------------------------------------------------------------

1.  The champion's true preprocessing contract is hands_only / seq_len=100 /
    126-dim input, which is NOT derivable from configs/data/seq100.yaml alone
    (that YAML's Pydantic default is landmark_config="full"). It is durably
    recorded ONLY in artifacts/experiments/bilstm_hands_only_v4_aug/
    config_snapshot.yaml (config_hash=5809193d...). Every function here that
    needs the model's shape contract takes an ExperimentConfig loaded from
    that snapshot, never a freshly-assembled load_config(...) call.

2.  _verify_champion_model() is config-DRIVEN, not just hardcoded-constant
    driven. Stage 5 produced 23 SavedModels under models/; loading the wrong
    one by path is a one-line directory-name typo away. The hard checks here
    derive the model's expected input/output shape from the supplied
    ExperimentConfig (sequence_length, landmark_config → feature_dim via
    LANDMARK_CONFIGS, num_classes) and additionally cross-check the documented
    champion parameter count (68,771) whenever the resolved shape happens to
    match the champion's shape. As of this revision it ALSO cross-checks the
    champion's known layer architecture (Bidirectional(LSTM) x2 + Masking)
    whenever the shape+params both match the champion, so a hypothetical
    GRU/attention variant that happened to land on the exact same parameter
    count cannot silently masquerade as the champion (critical-review
    "Most Important Architectural Concern").

3.  SELECT_TF_OPS is NOT applied unconditionally. Bidirectional(LSTM(...))
    under TF 2.13 emits TensorListReserve/TensorListStack ops that the default
    TFLite builtin op set cannot lower. Rather than always paying the ~800 KB
    flex-delegate runtime cost, this module attempts a builtins-only conversion
    FIRST and falls back to SELECT_TF_OPS + _experimental_lower_tensor_list_ops
    =False only if that fails. This was independently confirmed necessary by
    Stage 6's benchmark.py integration testing for this exact architecture.

4.  Dynamic-range quantisation does not require a representative dataset
    (it quantises weights only, not activations) — but export.
    representative_dataset_size: 100 is a declared, non-null field in the
    champion's own config_snapshot.yaml, and a representative-dataset generator
    function is also the mandatory ingredient for any future full-integer-
    quantisation experiment. make_representative_dataset_fn() is built directly
    against GestureDataset.get_arrays_for_split() — the REAL method signature
    in src/features/dataset.py, which already returns fully-pipelined
    (n, seq_len, feature_dim) arrays. There is no need to re-invoke
    FeaturePipeline here.

    As of this revision, building this (GestureDataset-backed) generator is
    SKIPPED BY DEFAULT for DYNAMIC_RANGE and FLOAT16 exports (the project
    default and the only two implemented modes) — see
    export_champion()'s build_representative_dataset resolution. Neither mode
    consumes calibration data, so constructing a GestureDataset (which
    preloads all 339 clips into RAM, ~20-30 MB, plus a FeaturePipeline pass)
    purely to build a generator that is never called is wasted work
    (critical-review point #4). The generator-building code path is fully
    retained and exercised the moment a caller opts in (or once
    FULL_INTEGER is implemented).

5.  With only 52 val clips, n_actual = min(representative_dataset_size, 52)
    is the honest count — logged explicitly rather than silently capped.

6.  A post-conversion sanity pass (_sanity_check_tflite) loads the freshly
    written .tflite file with a real tf.lite.Interpreter, runs one zero-input
    forward pass, and confirms the output is finite and the right shape. If
    this sanity pass FAILS, the just-written .tflite file is deleted before
    the exception propagates (critical-review point #8) — a broken artefact
    must never be left sitting at the production path
    (models/gesture_bilstm_v1.tflite) where GesturePredictor or Stage 9's
    webcam demo could load it on a subsequent, unrelated run.

7.  This module never asserts config.export.quantisation_mode ==
    "dynamic_range" as the only valid value. QuantisationMode.FLOAT16 is
    implemented. QuantisationMode.FULL_INTEGER raises NotImplementedError with
    the exact reason (the same Bidirectional(LSTM) / TensorList INT8-kernel
    gap documented in the Stage 8 spec) rather than silently producing a
    broken converter configuration. quantisation_mode is now coerced through
    QuantisationMode(...) wherever it crosses a boundary that could plausibly
    hand this function a raw string (e.g. a hand-edited config_snapshot.yaml
    re-loaded via OmegaConf) rather than trusting it is already an enum
    member (critical-review point #5).

8.  The result dict uses the EXACT key names from the Stage 8 Revised
    Specification: tflite_disk_mb, savedmodel_disk_mb, param_memory_mb,
    size_reduction_vs_params_x, size_reduction_vs_savedmodel_x,
    conversion_time_s, used_select_tf_ops, keras_params, quantised,
    quantisation_mode, output_path. These feed directly into verify.py's
    write_model_metadata() and tflite_verification_report.json.

9.  A SHA-256 checksum of the output .tflite file is computed AFTER the
    post-conversion sanity check succeeds, not before (critical-review point
    #7). The checksum is meant to certify "this is the verified artefact
    that shipped" — computing it before the sanity pass would mean a file
    that fails sanity checking (and is then deleted) briefly had a checksum
    computed for it, which is a wasted computation at best and a confusing
    half-state at worst if the function is ever modified to return partial
    results.

10. SavedModel directory structure is validated (saved_model.pb + variables/
    directory must exist) before attempting to load the model, catching a
    half-written or corrupted SavedModel immediately with a clear message.

11. Keras shape attributes (model.input_shape / model.output_shape) are
    normalised through a single helper, _normalise_keras_shape(), before any
    comparison or arithmetic. tf.keras.Model.input_shape can legitimately be
    a bare tuple (Sequential, single-input Functional — the only pattern this
    project's src/models/architectures.py produces) OR a list of tuples
    (multi-input Functional models). Comparing a raw, un-normalised
    model.output_shape against a plain tuple is fragile against that second
    case (critical-review point #2); normalising first makes the comparison
    correct regardless of which Keras API style produced the model, with no
    behavioural change for this project's actual (Sequential-style) models.

Non-goals of this file (left to src/export/verify.py, Stage 8 Step 2)
------------------------------------------------------------------------
  - Full val/test-set accuracy comparison (Keras vs TFLite macro-F1 delta).
  - Per-class delta analysis, probability-distribution comparison.
  - models/gesture_model_metadata.json authoring.
  - Production latency benchmarking (reuses src/evaluation/benchmark.py's
    benchmark_tflite_inference() against the file this module produces).

This module performs exactly one job, correctly and defensively:
SavedModel on disk → verified, sanity-checked, checksummed .tflite file on disk.

Champion model reference (confirmed from config_snapshot.yaml)
--------------------------------------------------------------
  architecture:    BiLSTM, 2 layers, hidden_units=64, bidirectional=True
  units/direction: 32 (hidden_units // 2), concat output = 64
  total_params:    68,771
  input_shape:     (None, 100, 126)  — seq_len=100, hands_only
  output_shape:    (None, 35)        — 35-class softmax
  config_hash:     5809193d37e0d480e409b8e3112e70c8de9008497a29727b411a7128e73287a6
  mlflow_run_id:   cb16f689d2294001a2ff2d3e02419d27
  val_macro_f1:    0.6011
  landmark_config: hands_only (126 dims = 63 LH + 63 RH)
  early_stopping:  patience=50, monitor=val_accuracy (ReduceLROnPlateau),
                   manual macro-F1 patience loop for champion selection
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from src.utils.config import ExperimentConfig, QuantisationMode
from src.utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Module-level constants
# =============================================================================

#: Documented parameter count of the verified champion
#: (bilstm_hands_only_v4_aug: BiLSTM, 2 layers, hidden_units=64 →
#: 32 units/direction, hands_only landmark_config, seq_len=100).
#: Source: handoff Part 4 Stage 5 champion table + config_snapshot.yaml.
_EXPECTED_CHAMPION_PARAMS: int = 68_771

#: Champion's verified Keras (dynamic batch) input/output shapes.
#: Used as FALLBACK when no ExperimentConfig is supplied to
#: _verify_champion_model; when a config IS supplied, expectations are
#: derived from it instead.
_EXPECTED_CHAMPION_INPUT_SHAPE: Tuple[Optional[int], int, int] = (None, 100, 126)
_EXPECTED_CHAMPION_OUTPUT_SHAPE: Tuple[Optional[int], int] = (None, 35)

#: TFLite uses static batch=1 shapes (interpreter allocates tensors at
#: conversion time). Kept separate from Keras shapes to avoid confusion.
_TFLITE_EXPECTED_INPUT_SHAPE: Tuple[int, int, int] = (1, 100, 126)
_TFLITE_EXPECTED_OUTPUT_SHAPE: Tuple[int, int] = (1, 35)

#: Reference config_hash prefix from the champion's verified config_snapshot.yaml.
#: Used only for an informational log line — never asserted — since
#: intentionally exporting a different run is legitimate.
_KNOWN_CHAMPION_CONFIG_HASH: str = (
    "5809193d37e0d480e409b8e3112e70c8de9008497a29727b411a7128e73287a6"
)

#: Champion's known layer-architecture signature, used ONLY as an additional
#: non-fatal-by-default cross-check (critical-review "Most Important
#: Architectural Concern") when shape AND parameter count both already match
#: the champion. Expressed as a sequence of (substring-of-class-name,
#: min_occurrences) pairs rather than exact layer objects, since this project
#: builds models via src/models/architectures.py's Sequential-style factory
#: and the class-name substring is stable across TF point releases while
#: exact layer configs are not part of this module's concern.
_EXPECTED_CHAMPION_LAYER_SIGNATURE: Tuple[Tuple[str, int], ...] = (
    ("Masking", 1),
    ("Bidirectional", 2),
)

#: Project-default paths, matching the Stage 8 deliverable table.
_DEFAULT_SAVED_MODEL_PATH: str = "models/bilstm_hands_only_v4_aug_saved_model"
_DEFAULT_TFLITE_OUTPUT_PATH: str = "models/gesture_bilstm_v1.tflite"
_DEFAULT_CONFIG_SNAPSHOT_PATH: str = (
    "artifacts/experiments/bilstm_hands_only_v4_aug/config_snapshot.yaml"
)

#: Bytes per float32 parameter — identical formula used throughout the
#: codebase (architectures.py, factory.py, benchmark.py, predictor.py) so
#: size estimates never diverge from numbers already logged elsewhere.
_BYTES_PER_FLOAT32: int = 4

#: Project global seed default, matching base.yaml's top-level seed: 42.
_DEFAULT_SEED: int = 42

#: Maximum acceptable TFLite file size (Stage 8 project constant, Part 2).
_MAX_TFLITE_SIZE_MB: float = 10.0

#: Minimum softmax output sum sanity check — output should sum to ~1.0.
_SOFTMAX_SUM_TOLERANCE: float = 0.05

#: Quantisation modes that consume a representative_dataset generator.
#: Neither implemented mode actually uses it today (DYNAMIC_RANGE quantises
#: weights only; FLOAT16 casts weights directly with no calibration step) —
#: this set exists so the "do we need to build it?" decision in
#: export_champion() is a single, explicit, easily-audited source of truth
#: rather than an implicit assumption buried in conditional logic.
_QUANTISATION_MODES_REQUIRING_REPR_DATASET: frozenset = frozenset(
    {QuantisationMode.FULL_INTEGER}
)


# =============================================================================
# Shape normalisation helper (critical-review point #2)
# =============================================================================

def _normalise_keras_shape(
    raw_shape: Any,
) -> Tuple[Optional[int], ...]:
    """
    Normalise a Keras ``model.input_shape`` / ``model.output_shape`` value
    into a single, flat tuple suitable for direct comparison against
    ``_EXPECTED_CHAMPION_INPUT_SHAPE`` / a config-derived expected shape.

    Why this exists
    -----------------
    ``tf.keras.Model.input_shape`` returns a bare tuple, e.g.
    ``(None, 100, 126)``, for the Sequential / single-input Functional
    models this project's ``src/models/architectures.py`` actually builds
    (every architecture there — Dense, LSTM, GRU, BiLSTM — has exactly one
    ``Input`` and one ``Dense(n_classes, softmax)`` output). However, the
    Keras API contract more generally allows ``input_shape`` to be a
    **list of tuples** for genuinely multi-input Functional models (e.g.
    ``[(None, 100, 126), (None, 10)]``). A bare ``tuple(model.input_shape)``
    call — as used directly in earlier revisions of this module — silently
    "succeeds" on a list-of-tuples input by producing a tuple-of-tuples,
    which then fails EVERY downstream equality check with a confusing
    "shape mismatch" error that does not name the real problem.

    This function:
      - Accepts a bare tuple/list of ints (or ``tf.TensorShape``) → returns
        it as a plain tuple of ``int | None``.
      - Accepts a single-element list/tuple wrapping such a shape (the
        single-input Functional case) → unwraps one level and returns the
        inner shape.
      - Raises ``ValueError`` with a clear message for any genuinely
        multi-input/output shape, since no function in this module is
        equipped to verify a multi-input model's contract — that would be
        a deliberate, separate extension, not a silent guess.

    Parameters
    ----------
    raw_shape : Any
        Typically ``model.input_shape`` or ``model.output_shape``.

    Returns
    -------
    Tuple[Optional[int], ...]

    Raises
    ------
    ValueError
        If ``raw_shape`` represents a multi-input/output (list-of-shapes)
        model, which this module does not support.
    """
    # tf.TensorShape implements __iter__ and behaves like a sequence of
    # int|None; converting via list() handles it identically to a plain tuple.
    try:
        as_list = list(raw_shape)
    except TypeError as exc:
        raise ValueError(
            f"_normalise_keras_shape(): could not interpret {raw_shape!r} "
            "as an iterable shape. Expected a tuple/list of int|None or a "
            "tf.TensorShape."
        ) from exc

    if len(as_list) == 0:
        raise ValueError(
            f"_normalise_keras_shape(): received an empty shape {raw_shape!r}."
        )

    # Detect "list of shapes" (multi-input/output Functional model): every
    # element is itself a non-empty sequence (tuple/list/TensorShape), as
    # opposed to a single shape where every element is an int or None.
    def _is_shape_like(x: Any) -> bool:
        return (x is None) or isinstance(x, (int, np.integer))

    if all(_is_shape_like(x) for x in as_list):
        # Already a flat shape, e.g. [None, 100, 126].
        return tuple(int(x) if x is not None else None for x in as_list)

    # Not flat — likely a list of per-input/output shapes.
    if len(as_list) == 1:
        # Single-input Functional model: Keras sometimes wraps the one
        # shape in a list. Unwrap exactly one level and retry.
        return _normalise_keras_shape(as_list[0])

    raise ValueError(
        f"_normalise_keras_shape(): received what appears to be a "
        f"multi-input/output shape specification ({raw_shape!r}). This "
        "module's champion-verification logic assumes a single-input, "
        "single-output classification model (the only pattern "
        "src/models/architectures.py produces — Dense/LSTM/GRU/BiLSTM, "
        "each with exactly one Input and one Dense(n_classes, softmax) "
        "output). A genuinely multi-input/output model is out of scope "
        "for this export pipeline."
    )


# =============================================================================
# Config snapshot loading
# =============================================================================

def load_config_snapshot(
    config_snapshot_path: Union[str, Path],
) -> ExperimentConfig:
    """
    Reconstruct a fully-validated ExperimentConfig from a saved run's
    config_snapshot.yaml.

    This is the RECOMMENDED — and in this module, the ONLY — way to obtain
    the config used for export. load_config(model=..., data=..., ...) CLI-
    style reconstruction is deliberately not supported here: the champion's
    data.landmark_config: hands_only override was applied at CLI-runtime
    during Stage 5 and is recoverable ONLY from the snapshot written by
    setup_experiment() — the identical reasoning already applied in
    GesturePredictor.from_config_snapshot().

    Parameters
    ----------
    config_snapshot_path : str | Path
        e.g. "artifacts/experiments/bilstm_hands_only_v4_aug/config_snapshot.yaml".

    Returns
    -------
    ExperimentConfig
        Fully validated, frozen, with its own recomputed config_hash.

    Raises
    ------
    FileNotFoundError
        If the snapshot file does not exist.
    """
    from omegaconf import OmegaConf

    path = Path(config_snapshot_path)
    if not path.exists():
        raise FileNotFoundError(
            f"load_config_snapshot(): config snapshot not found at {path}. "
            "This file is written once per run by "
            "src/utils/reproducibility.py's setup_experiment() under "
            "artifacts/experiments/<run_name>/config_snapshot.yaml. "
            f"For the champion, expected: {_DEFAULT_CONFIG_SNAPSHOT_PATH}"
        )

    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    config = ExperimentConfig(**raw)

    # Critical-review point #5: quantisation_mode must be a QuantisationMode
    # enum member, not a raw string, by the time any function in this module
    # touches it. Pydantic v2 normally coerces this automatically on
    # ExperimentConfig construction, but a hand-edited / stale snapshot
    # YAML re-loaded via plain OmegaConf (bypassing the Pydantic validator
    # that would normally run at config-build time) is exactly the failure
    # mode the critical review flagged. Coerce explicitly and fail loudly
    # with an actionable message rather than deferring the failure to a
    # confusing ".value on a str" AttributeError deep inside
    # _configure_converter().
    raw_qmode = config.export.quantisation_mode
    if not isinstance(raw_qmode, QuantisationMode):
        try:
            coerced_qmode = QuantisationMode(raw_qmode)
        except ValueError as exc:
            raise ValueError(
                f"load_config_snapshot(): config.export.quantisation_mode="
                f"{raw_qmode!r} (type={type(raw_qmode).__name__}) is not a "
                f"valid QuantisationMode value. Valid values: "
                f"{[m.value for m in QuantisationMode]}. The config snapshot "
                f"at {path} may be stale or hand-edited — regenerate it via "
                "a fresh training run if possible."
            ) from exc
        # ExperimentConfig may be a frozen/validated Pydantic model; mutate
        # only the nested export sub-model's field via object.__setattr__ if
        # direct assignment is blocked by frozen=True. Try direct assignment
        # first since most of this project's configs are not frozen at the
        # sub-model level (only the top-level config_hash is immutable).
        try:
            config.export.quantisation_mode = coerced_qmode
        except (TypeError, ValueError):
            object.__setattr__(config.export, "quantisation_mode", coerced_qmode)
        logger.warning(
            "load_config_snapshot(): coerced config.export.quantisation_mode "
            "from raw value %r to QuantisationMode.%s. This indicates the "
            "snapshot was loaded through a path that bypassed normal Pydantic "
            "validation — verify %s was not hand-edited.",
            raw_qmode, coerced_qmode.name, path,
            extra={"stage": "export"},
        )

    observed_hash = str(getattr(config, "config_hash", ""))
    if observed_hash and not observed_hash.startswith(_KNOWN_CHAMPION_CONFIG_HASH[:8]):
        logger.info(
            "load_config_snapshot(): loaded config_hash=%s does not match "
            "the known champion reference prefix (%s...). This is expected "
            "and harmless if you are intentionally exporting a different run.",
            observed_hash[:12], _KNOWN_CHAMPION_CONFIG_HASH[:8],
            extra={"stage": "export"},
        )
    else:
        logger.info(
            "load_config_snapshot(): config hash verified matches champion "
            "prefix (%s...)",
            observed_hash[:8] if observed_hash else "unknown",
            extra={"stage": "export"},
        )

    logger.info(
        "Config snapshot loaded | path=%s | landmark_config=%s | "
        "sequence_length=%d | num_classes=%d | quantisation_mode=%s | "
        "config_hash=%s",
        path,
        config.data.landmark_config,
        config.data.sequence_length,
        config.num_classes,
        config.export.quantisation_mode.value,
        observed_hash[:12] if observed_hash else "unknown",
        extra={"stage": "export"},
    )
    return config


# =============================================================================
# SavedModel directory validation
# =============================================================================

def _validate_savedmodel_directory(src: Path) -> None:
    """
    Validate that a SavedModel directory has the expected structure.

    A SavedModel directory must contain saved_model.pb and a variables/
    subdirectory. Checking this before tf.keras.models.load_model() provides
    a clear, actionable error message rather than a cryptic TensorFlow
    internal failure when the path is malformed or incomplete.

    Parameters
    ----------
    src : Path
        Path to the SavedModel directory.

    Raises
    ------
    FileNotFoundError
        If the directory does not exist.
    ValueError
        If the directory is missing saved_model.pb or variables/.
    """
    if not src.exists():
        raise FileNotFoundError(
            f"_validate_savedmodel_directory(): SavedModel directory not "
            f"found: {src}. "
            f"For the champion, expected: {_DEFAULT_SAVED_MODEL_PATH}"
        )

    if not src.is_dir():
        raise ValueError(
            f"_validate_savedmodel_directory(): {src} exists but is not a "
            "directory. A Keras SavedModel must be a directory containing "
            "saved_model.pb and variables/. If you have a .keras file, "
            "convert it to SavedModel format first."
        )

    pb_file = src / "saved_model.pb"
    if not pb_file.exists():
        raise ValueError(
            f"_validate_savedmodel_directory(): {src} is missing "
            "saved_model.pb. This indicates a corrupted or incomplete "
            "SavedModel export. Re-run Stage 5 training to regenerate it."
        )

    variables_dir = src / "variables"
    if not variables_dir.exists() or not variables_dir.is_dir():
        raise ValueError(
            f"_validate_savedmodel_directory(): {src} is missing the "
            "variables/ subdirectory. This indicates a corrupted or "
            "incomplete SavedModel export. Re-run Stage 5 training."
        )

    logger.info(
        "_validate_savedmodel_directory(): SavedModel structure OK | path=%s",
        src,
        extra={"stage": "export"},
    )


# =============================================================================
# Architecture signature check (critical-review "Most Important Architectural
# Concern")
# =============================================================================

def _check_layer_architecture_signature(
    model: Any,
    expected_signature: Sequence[Tuple[str, int]] = _EXPECTED_CHAMPION_LAYER_SIGNATURE,
) -> Dict[str, Any]:
    """
    Verify the loaded model's layer composition matches the champion's known
    architecture signature (Masking + 2x Bidirectional), independent of
    shape or parameter count.

    Why this exists
    -----------------
    Shape (input/output) and parameter count are strong but not airtight
    identity signals: it is combinatorially unlikely but not impossible for
    a differently-architected model (e.g. a single-layer GRU with a wider
    hidden size, or an attention-based model) to land on the exact champion
    parameter count of 68,771 while having a completely different internal
    structure and therefore completely different — and untrustworthy —
    learned behaviour. This function inspects ``model.layers`` directly and
    counts how many layers' class names contain each substring in
    ``expected_signature``, which is robust to TF point-release renaming of
    private implementation details while still being a real structural check
    (not just another shape/count proxy).

    This check is informational/diagnostic by default — see
    ``strict_architecture_check`` in ``_verify_champion_model`` — because:
      (a) it is a NEW addition in this revision and should not silently turn
          a previously-passing export pipeline into a hard failure for
          legitimate non-champion exports that happen to share the
          champion's shape (e.g. a deliberately-exported sibling ablation
          run with a different architecture but same I/O contract — which
          is exactly the scenario the parameter-count check already guards,
          just from a different angle), and
      (b) a Sequential model's ``model.layers`` ordering and class naming is
          stable for this project's TF 2.13.1 pin, but treating an
          architecture-name substring match as an unconditional hard gate
          would be a stronger claim than this module needs to make to
          satisfy its actual job (catching wrong-SavedModel-by-path
          mistakes among Stage 5's 23 candidates).

    Parameters
    ----------
    model : tf.keras.Model
        An already-loaded SavedModel.
    expected_signature : Sequence[Tuple[str, int]], default
        ``_EXPECTED_CHAMPION_LAYER_SIGNATURE`` — e.g.
        ``(("Masking", 1), ("Bidirectional", 2))``: at least 1 layer whose
        class name contains "Masking", and at least 2 whose class name
        contains "Bidirectional".

    Returns
    -------
    dict
        {matches: bool, observed_layer_classes: List[str],
         missing_or_insufficient: List[str]}
    """
    try:
        layer_classes = [type(layer).__name__ for layer in model.layers]
    except Exception as exc:  # pragma: no cover - defensive, model.layers is standard
        logger.warning(
            "_check_layer_architecture_signature(): could not enumerate "
            "model.layers (%s: %s). Skipping architecture signature check.",
            type(exc).__name__, exc,
            extra={"stage": "export"},
        )
        return {
            "matches": None,
            "observed_layer_classes": [],
            "missing_or_insufficient": [],
            "skipped_reason": f"{type(exc).__name__}: {exc}",
        }

    missing_or_insufficient: List[str] = []
    for substring, min_count in expected_signature:
        actual_count = sum(1 for cls_name in layer_classes if substring in cls_name)
        if actual_count < min_count:
            missing_or_insufficient.append(
                f"expected >= {min_count} layer(s) containing '{substring}', "
                f"found {actual_count}"
            )

    matches = len(missing_or_insufficient) == 0

    return {
        "matches": matches,
        "observed_layer_classes": layer_classes,
        "missing_or_insufficient": missing_or_insufficient,
    }


# =============================================================================
# Step 1.1 — Model loading with config-driven integrity verification
# =============================================================================

def _verify_champion_model(
    model: Any,
    config: Optional[ExperimentConfig] = None,
    strict_champion_param_check: bool = True,
    strict_architecture_check: bool = False,
) -> Dict[str, Any]:
    """
    Hard sanity checks before any conversion occurs.

    Fails loudly so a wrong SavedModel — there are 23 candidates under
    models/ from Stage 5's experiment matrix — never silently becomes a
    deployment artefact.

    Expectation resolution
    -----------------------
    If config is supplied, the expected input/output shape is DERIVED from it
    (config.data.sequence_length, the feature_dim implied by
    config.data.landmark_config via LANDMARK_CONFIGS, and config.num_classes)
    rather than from hardcoded champion constants. This keeps the function
    correct if it is ever pointed at a different run's SavedModel +
    config_snapshot.

    If config is omitted, falls back to the champion's known constants
    (_EXPECTED_CHAMPION_INPUT_SHAPE / _EXPECTED_CHAMPION_OUTPUT_SHAPE).

    Both the model's and the expected shapes are passed through
    ``_normalise_keras_shape()`` before comparison (critical-review point
    #2), so a Functional-API model that reports its shape as a
    single-element list-of-tuples is compared correctly rather than failing
    on a spurious tuple-of-tuples mismatch.

    Parameter-count check
    -----------------------
    The documented champion parameter count (68,771) is checked WHENEVER the
    resolved expected shape happens to equal the champion's shape. A mismatch
    with strict_champion_param_check=True raises immediately: matching
    input/output shape but a different parameter count is exactly the signature
    of accidentally loading a different ablation run (e.g. bilstm_hands_only_v2)
    that happens to share the champion's I/O contract.

    Architecture signature check (new in this revision)
    -------------------------------------------------------
    Whenever BOTH the shape and the parameter count already match the
    champion, an additional structural check
    (``_check_layer_architecture_signature``) confirms the model actually
    contains a Masking layer and two Bidirectional layers — the champion's
    real architecture — rather than trusting shape+params alone (see
    ``_check_layer_architecture_signature`` docstring for the full
    rationale). This is diagnostic-only (logged, never raises) unless
    ``strict_architecture_check=True`` is explicitly passed, since it is a
    new, additive check and a parameter-count collision this specific is
    already astronomically unlikely.

    Parameters
    ----------
    model : tf.keras.Model
        An already-loaded SavedModel.
    config : ExperimentConfig, optional
        The config the model was trained with (recommended: loaded via
        load_config_snapshot()).
    strict_champion_param_check : bool, default True
        If True, a parameter-count mismatch against the champion (when the
        resolved shape matches the champion's shape) raises ValueError.
        If False, it only logs a WARNING.
    strict_architecture_check : bool, default False
        If True, a layer-architecture-signature mismatch (only evaluated
        when shape AND params already match the champion) raises ValueError
        instead of logging a WARNING.

    Returns
    -------
    dict
        {actual_params, actual_input_shape, actual_output_shape,
         expected_input_shape, expected_output_shape,
         params_match_champion, architecture_check, config_hash}

    Raises
    ------
    ValueError
        On any shape mismatch, on a champion-shape parameter mismatch when
        strict_champion_param_check=True, or on a champion-shape+params
        architecture-signature mismatch when strict_architecture_check=True.
    """
    actual_params = int(model.count_params())
    actual_input = _normalise_keras_shape(model.input_shape)
    actual_output = _normalise_keras_shape(model.output_shape)

    if config is not None:
        from src.features.constants import LANDMARK_CONFIGS

        lm_config = str(config.data.landmark_config)
        if lm_config not in LANDMARK_CONFIGS:
            raise ValueError(
                f"_verify_champion_model(): config.data.landmark_config="
                f"'{lm_config}' is not a recognised key in LANDMARK_CONFIGS "
                f"({sorted(LANDMARK_CONFIGS.keys())}). The supplied config "
                "appears corrupted or stale — re-derive it from "
                "load_config_snapshot()."
            )
        lm_slice = LANDMARK_CONFIGS[lm_config]
        expected_feature_dim = lm_slice.stop - lm_slice.start
        expected_input_shape: Tuple[Optional[int], ...] = (
            None,
            int(config.data.sequence_length),
            expected_feature_dim,
        )
        expected_output_shape: Tuple[Optional[int], ...] = (
            None,
            int(config.num_classes),
        )
    else:
        expected_input_shape = _EXPECTED_CHAMPION_INPUT_SHAPE
        expected_output_shape = _EXPECTED_CHAMPION_OUTPUT_SHAPE
        logger.warning(
            "_verify_champion_model(): no config supplied — falling back to "
            "hardcoded champion shape expectations %s -> %s. Pass the "
            "ExperimentConfig from load_config_snapshot() for a config-driven, "
            "non-champion-specific check.",
            expected_input_shape,
            expected_output_shape,
            extra={"stage": "export"},
        )

    if actual_input != expected_input_shape:
        raise ValueError(
            f"_verify_champion_model(): model input shape {actual_input} does "
            f"not match expected {expected_input_shape}. "
            "Confirm the correct SavedModel path was supplied, and that the "
            "config's sequence_length / landmark_config matches the model "
            "that was actually trained and saved at that path."
        )

    if actual_output != expected_output_shape:
        raise ValueError(
            f"_verify_champion_model(): model output shape {actual_output} "
            f"does not match expected {expected_output_shape}. "
            "Check config.num_classes against the model's final Dense layer."
        )

    is_champion_shape = (
        expected_input_shape == _EXPECTED_CHAMPION_INPUT_SHAPE
        and expected_output_shape == _EXPECTED_CHAMPION_OUTPUT_SHAPE
    )
    params_match_champion = actual_params == _EXPECTED_CHAMPION_PARAMS

    if is_champion_shape and not params_match_champion:
        msg = (
            f"_verify_champion_model(): loaded model has {actual_params:,} "
            f"parameters; champion bilstm_hands_only_v4_aug has "
            f"{_EXPECTED_CHAMPION_PARAMS:,}. Shape matches the champion's "
            "I/O contract but the parameter count does not — this is the "
            "signature of an accidentally-loaded sibling ablation run "
            "(e.g. bilstm_hands_only_v2/v3, which share this shape but "
            "differ in trained weights/architecture details). "
            "Verify the SavedModel path."
        )
        if strict_champion_param_check:
            raise ValueError(msg)
        logger.warning(msg, extra={"stage": "export"})

    # ── Architecture signature check (new) ───────────────────────────────
    # Only meaningful — and only run — once shape AND params already match
    # the champion; otherwise the shape/param checks above have already
    # raised (or warned) and an architecture mismatch would be redundant
    # noise on top of an already-flagged problem.
    architecture_check: Dict[str, Any] = {
        "matches": None,
        "observed_layer_classes": [],
        "missing_or_insufficient": [],
    }
    if is_champion_shape and params_match_champion:
        architecture_check = _check_layer_architecture_signature(model)
        if architecture_check["matches"] is False:
            arch_msg = (
                f"_verify_champion_model(): model shape and parameter count "
                f"({actual_params:,}) both match the champion "
                "bilstm_hands_only_v4_aug, but its layer architecture does "
                f"not match the champion's known signature "
                f"{_EXPECTED_CHAMPION_LAYER_SIGNATURE}: "
                f"{architecture_check['missing_or_insufficient']}. "
                f"Observed layer classes: {architecture_check['observed_layer_classes']}. "
                "This is an extremely unlikely but not impossible "
                "parameter-count collision between two structurally "
                "different models. Verify the SavedModel path before "
                "trusting this export."
            )
            if strict_architecture_check:
                raise ValueError(arch_msg)
            logger.warning(arch_msg, extra={"stage": "export"})
        elif architecture_check["matches"] is True:
            logger.info(
                "_verify_champion_model(): architecture signature confirmed "
                "(Masking + 2x Bidirectional layers present) — shape, "
                "parameter count, and layer structure all match the "
                "champion.",
                extra={"stage": "export"},
            )

    observed_hash = (
        str(getattr(config, "config_hash", "")) if config is not None else ""
    )

    diagnostics: Dict[str, Any] = {
        "actual_params": actual_params,
        "actual_input_shape": list(actual_input),
        "actual_output_shape": list(actual_output),
        "expected_input_shape": list(expected_input_shape),
        "expected_output_shape": list(expected_output_shape),
        "params_match_champion": params_match_champion,
        "architecture_check": architecture_check,
        "config_hash": observed_hash,
    }

    logger.info(
        "Champion model verified | params=%s | input=%s | output=%s | "
        "params_match_champion=%s | architecture_matches=%s",
        f"{actual_params:,}",
        actual_input,
        actual_output,
        params_match_champion,
        architecture_check["matches"],
        extra={"stage": "export"},
    )
    return diagnostics


# =============================================================================
# Step 1.2 — Representative dataset generation
# =============================================================================

def make_representative_dataset_fn(
    dataset: Any,
    split: str = "val",
    n_samples: int = 100,
    seed: int = _DEFAULT_SEED,
) -> Callable[[], Any]:
    """
    Build a TFLite representative-dataset generator from an already-built
    GestureDataset.

    Neither implemented quantisation mode (DYNAMIC_RANGE, the project
    default; FLOAT16) actually consumes this generator — DYNAMIC_RANGE
    quantises weights only, and FLOAT16 casts weights directly without any
    calibration step. It is built anyway, ON REQUEST (see
    ``export_champion()``'s ``build_representative_dataset`` parameter and
    ``_QUANTISATION_MODES_REQUIRING_REPR_DATASET``), because (a)
    ``config.export.representative_dataset_size`` is a declared, non-null
    field in the champion's own config_snapshot.yaml, and (b) it is the
    mandatory ingredient for any future full-integer-quantisation
    experiment. Building it unconditionally on every export — the prior
    behaviour — paid the cost of constructing a full ``GestureDataset``
    (which preloads all 339 clips into RAM) for zero benefit on the default
    DYNAMIC_RANGE path (critical-review point #4); this function itself is
    unchanged, only its caller's default behaviour around invoking it.

    Uses dataset.get_arrays_for_split(split, use_augmentation=False), which
    already returns fully-pipelined (N, seq_len, feature_dim) float32 arrays.
    No separate FeaturePipeline call is needed or correct here — calling the
    pipeline again would double-apply preprocessing.

    Sample-size honesty
    ---------------------
    With only 52 val clips, n_actual = min(n_samples, 52) — logged explicitly
    when capped, since a calibration set this small would make any future
    full-integer attempt unreliable and that limitation must be visible in
    the export log, not silently absorbed.

    Parameters
    ----------
    dataset : GestureDataset
        An already-constructed GestureDataset instance (same config as
        the model being exported).
    split : str, default "val"
        Which split to draw representative samples from.
    n_samples : int, default 100
        From config.export.representative_dataset_size.
    seed : int, default 42
        Sampling seed — matches the project's global seed.

    Returns
    -------
    Callable[[], Generator]
        A zero-argument generator function suitable for
        converter.representative_dataset. Each yielded element is
        [sample] where sample has shape (1, seq_len, feature_dim).

    Raises
    ------
    ValueError
        If the requested split has zero clips.
    """
    X, _, _ = dataset.get_arrays_for_split(split, use_augmentation=False)
    n_total = len(X)

    if n_total == 0:
        raise ValueError(
            f"make_representative_dataset_fn(): split '{split}' has zero "
            "clips in this GestureDataset instance. Check that Stage 1-3 "
            "data preparation completed for this split."
        )

    n_actual = min(n_samples, n_total)
    rng = np.random.default_rng(seed)

    if n_actual < n_total:
        indices = rng.choice(n_total, size=n_actual, replace=False)
    else:
        indices = np.arange(n_total)

    X_repr = np.ascontiguousarray(X[indices].astype(np.float32))

    if n_actual < n_samples:
        logger.warning(
            "make_representative_dataset_fn(): requested n_samples=%d but "
            "split '%s' only has %d clips; using all %d. A representative "
            "set this small would make any FULL_INTEGER quantisation "
            "calibration unreliable — dynamic-range quantisation (the "
            "current default) is unaffected since it does not consume this "
            "generator.",
            n_samples,
            split,
            n_total,
            n_actual,
            extra={"stage": "export"},
        )
    else:
        logger.info(
            "make_representative_dataset_fn(): built representative set | "
            "split=%s | n_samples=%d/%d | seed=%d | shape=%s",
            split,
            n_actual,
            n_total,
            seed,
            X_repr.shape,
            extra={"stage": "export"},
        )

    def representative_data_gen() -> Any:
        for i in range(n_actual):
            # Yields list of one sample: (1, seq_len, feature_dim)
            # batch dimension required by the TFLite converter
            yield [X_repr[i : i + 1]]

    return representative_data_gen


# =============================================================================
# Step 1.3 — Converter configuration per quantisation mode
# =============================================================================

def _configure_converter(
    converter: Any,
    quantisation_mode: QuantisationMode,
    representative_dataset_fn: Optional[Callable] = None,
    use_select_tf_ops: bool = False,
) -> Any:
    """
    Apply quantisation-mode-specific settings to an already-constructed
    tf.lite.TFLiteConverter.

    DYNAMIC_RANGE (project default for the champion)
        tf.lite.Optimize.DEFAULT with no representative dataset attached.
        int8 weights, float32 activations. ~4x size reduction. No
        calibration data needed.

    FLOAT16
        tf.lite.Optimize.DEFAULT + target_spec.supported_types=[tf.float16].
        ~2x size reduction, full activation precision preserved, zero
        calibration-data requirement. Representative dataset is NOT attached
        because float16 quantisation does not use it (weights are cast from
        float32 to float16 directly — there are no activation statistics to
        calibrate). This is consistent with DYNAMIC_RANGE's handling and
        avoids silently running the generator for no effect.
        ``target_spec.supported_ops`` is additionally pinned to
        ``[TFLITE_BUILTINS]`` here as a defensive, forward-compatibility
        measure (critical-review point #1): FLOAT16 conversion behaviour for
        ops outside the standard builtin set is not guaranteed stable across
        TF versions, and this project already has an explicit,
        independently-tested builtins-first/SELECT_TF_OPS-fallback strategy
        one level up in ``export_champion_tflite`` — pinning here just makes
        FLOAT16's own first attempt as predictable as DYNAMIC_RANGE's.

    FULL_INTEGER
        Raises NotImplementedError. Bidirectional(LSTM(...)) under TF 2.13
        emits TensorListReserve/TensorListStack ops that do not have full
        INT8 kernels without MLIR lowering tricks this project does not
        implement. At 0.262 MB float32 the model is already well under the
        10 MB target — the added complexity is not justified.

    SELECT_TF_OPS flex delegate
        Applied independently of quantisation mode whenever use_select_tf_ops=
        True. Sets target_spec.supported_ops=[TFLITE_BUILTINS, SELECT_TF_OPS]
        and _experimental_lower_tensor_list_ops=False to avoid an MLIR
        lowering attempt that fails for this BiLSTM's TensorList pattern.
        This OVERRIDES the FLOAT16-specific builtins-only pin above, exactly
        as intended — SELECT_TF_OPS is the fallback path activated only
        after a builtins-only attempt has already failed once.

    Parameters
    ----------
    converter : tf.lite.TFLiteConverter
    quantisation_mode : QuantisationMode
    representative_dataset_fn : Callable, optional
        Attached only when the quantisation mode can meaningfully use it
        (currently: no standard mode consumes it, but it is available for
        future FULL_INTEGER implementation).
    use_select_tf_ops : bool, default False

    Returns
    -------
    tf.lite.TFLiteConverter
        The same converter instance, mutated in place.

    Raises
    ------
    NotImplementedError
        If quantisation_mode == QuantisationMode.FULL_INTEGER.
    ValueError
        If quantisation_mode is not a recognised QuantisationMode member.
    """
    import tensorflow as tf

    # Critical-review point #5: defend against a raw string slipping through
    # despite load_config_snapshot()'s coercion (e.g. a caller constructing
    # quantisation_mode manually rather than via a loaded config).
    if not isinstance(quantisation_mode, QuantisationMode):
        try:
            quantisation_mode = QuantisationMode(quantisation_mode)
        except ValueError as exc:
            raise ValueError(
                f"_configure_converter(): quantisation_mode={quantisation_mode!r} "
                f"is not a valid QuantisationMode. Valid values: "
                f"{[m.value for m in QuantisationMode]}."
            ) from exc

    if quantisation_mode == QuantisationMode.DYNAMIC_RANGE:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # Deliberately NOT attaching representative_dataset_fn here:
        # dynamic-range quantises weights only; activation calibration data
        # is a no-op for this mode and attaching it would obscure which
        # code path actually drove the conversion.

    elif quantisation_mode == QuantisationMode.FLOAT16:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        # float16 quantisation does not use calibration data either —
        # weights are cast directly from float32 to float16 without
        # activation statistics. Not attaching the generator is correct.
        #
        # Defensive op-set pin (critical-review point #1): without this,
        # FLOAT16 conversion behaviour for non-builtin ops is undocumented
        # and could vary across TF point releases. Pinning to
        # TFLITE_BUILTINS here means FLOAT16's first attempt fails the same
        # way DYNAMIC_RANGE's does for this BiLSTM (TensorList ops), which
        # is then handled by the SAME builtins-first/SELECT_TF_OPS-fallback
        # logic in export_champion_tflite() — not a special case.
        if not use_select_tf_ops:
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

    elif quantisation_mode == QuantisationMode.FULL_INTEGER:
        raise NotImplementedError(
            "_configure_converter(): QuantisationMode.FULL_INTEGER is not "
            "supported for this architecture. Bidirectional(LSTM(...)) under "
            "TF 2.13 emits TensorListReserve/TensorListStack ops that lack "
            "full INT8 kernels without MLIR lowering tricks not implemented "
            "in this project. Use QuantisationMode.DYNAMIC_RANGE (project "
            "default — already ~4x size reduction on a 0.262 MB model) or "
            "QuantisationMode.FLOAT16 instead."
        )
    else:
        raise ValueError(
            f"_configure_converter(): unrecognised quantisation_mode "
            f"{quantisation_mode!r}. Expected a QuantisationMode member "
            f"(DYNAMIC_RANGE, FLOAT16, or FULL_INTEGER)."
        )

    if use_select_tf_ops:
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        # Prevent MLIR from attempting (and failing) to lower TensorList ops.
        # This is a private API required for TF 2.13 + Bidirectional(LSTM).
        # Confirmed necessary by Stage 6's benchmark.py integration tests.
        # Adds ~800 KB to the Android TFLite runtime binary — known, accepted.
        converter._experimental_lower_tensor_list_ops = False

    return converter


# =============================================================================
# Step 1.4 — Post-export sanity inference
# =============================================================================

def _sanity_check_tflite(
    tflite_path: Path,
    expected_input_shape: Optional[Tuple[Optional[int], ...]] = None,
    expected_output_shape: Optional[Tuple[Optional[int], ...]] = None,
) -> Dict[str, Any]:
    """
    Load the freshly-written .tflite file with a real interpreter and run
    one zero-input forward pass.

    This is the cheapest possible point to catch a broken export: a
    successful converter.convert() call does NOT guarantee the resulting
    file actually runs correctly under tf.lite.Interpreter (flex-delegate
    misconfiguration and corrupted writes are both silent at the .convert()
    call site). Catching it here means Stage 8 Step 2's full val-set
    comparison, or worse, Stage 9's live webcam demo, never has to discover
    a fundamentally broken file.

    IMPORTANT — this function does NOT delete the file on failure
    -------------------------------------------------------------------
    Deleting a failed export is the caller's responsibility
    (``export_champion_tflite``), not this function's — this function's
    only job is to inspect and report. Centralising the delete-on-failure
    behaviour at the call site (where the destination path, the "is this
    the production path or a throwaway diagnostic path" context, and any
    future additional sanity gates all live) keeps this function a pure,
    side-effect-free (besides reading the file) verifier that is easy to
    unit test in isolation.

    The TFLite interpreter uses static shapes (batch=1), so shape comparison
    uses _TFLITE_EXPECTED_INPUT_SHAPE / _TFLITE_EXPECTED_OUTPUT_SHAPE as
    fallbacks rather than the Keras-style (None, ...) shapes.

    Parameters
    ----------
    tflite_path : Path
        Path to the just-written .tflite file.
    expected_input_shape, expected_output_shape : tuple, optional
        If supplied, the interpreter's actual fixed shapes are compared
        against these. None or -1 at position 0 is treated as a wildcard
        (dynamic batch; the interpreter fixes batch=1).

    Returns
    -------
    dict
        {input_shape, output_shape, output_finite, output_sum_row0,
         n_input_tensors, n_output_tensors}

    Raises
    ------
    ValueError
        If the model does not have exactly one input/output tensor, if the
        shapes disagree with the supplied expectations, or if the forward
        pass produces non-finite output.
    """
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if len(input_details) != 1 or len(output_details) != 1:
        raise ValueError(
            f"_sanity_check_tflite(): expected exactly one input and one "
            f"output tensor; got {len(input_details)} input(s) and "
            f"{len(output_details)} output(s) in {tflite_path}."
        )

    in_shape = tuple(int(d) for d in input_details[0]["shape"])
    out_shape = tuple(int(d) for d in output_details[0]["shape"])

    def _shapes_compatible(
        actual: Tuple[int, ...],
        expected: Tuple[Optional[int], ...],
    ) -> bool:
        """Allow None or -1 as wildcard at any position."""
        if len(actual) != len(expected):
            return False
        return all(
            (e is None or e == -1 or e == a) for a, e in zip(actual, expected)
        )

    if expected_input_shape is not None:
        if not _shapes_compatible(in_shape, expected_input_shape):
            raise ValueError(
                f"_sanity_check_tflite(): TFLite input shape {in_shape} is "
                f"incompatible with expected {expected_input_shape}. "
                "The exported file does not match the model's documented contract. "
                "_shapes_compatible() already treats None/-1 as a wildcard at any "
                "position, so a Keras-style (None, seq_len, feature_dim) expectation "
                "is matched correctly against TFLite's static-batch shape without "
                "any further fallback."
            )

    if expected_output_shape is not None:
        if not _shapes_compatible(out_shape, expected_output_shape):
            raise ValueError(
                f"_sanity_check_tflite(): TFLite output shape {out_shape} is "
                f"incompatible with expected {expected_output_shape}."
            )

    dummy = np.zeros(in_shape, dtype=input_details[0]["dtype"])
    interpreter.set_tensor(input_details[0]["index"], dummy)
    interpreter.invoke()
    output = np.asarray(interpreter.get_tensor(output_details[0]["index"]))

    if not np.all(np.isfinite(output)):
        n_nan = int(np.isnan(output).sum())
        n_inf = int(np.isinf(output).sum())
        raise ValueError(
            f"_sanity_check_tflite(): forward pass on a zero-input sample "
            f"produced non-finite output (NaN={n_nan}, Inf={n_inf}) in "
            f"{tflite_path}. Investigate the quantisation/flex-delegate "
            "configuration before treating this file as deployable — a "
            "softmax output should never contain NaN/Inf even on a "
            "degenerate all-zero input."
        )

    output_sum_row0 = None
    if output.size > 0:
        row0 = output.reshape(output.shape[0], -1)[0]
        output_sum_row0 = float(row0.sum())
        # Softmax should sum to ~1.0; warn but don't fail on slight deviation
        if abs(output_sum_row0 - 1.0) > _SOFTMAX_SUM_TOLERANCE:
            logger.warning(
                "_sanity_check_tflite(): output row 0 sums to %.4f (expected "
                "≈1.0 for softmax). This may indicate quantisation artefacts "
                "or a non-softmax final activation.",
                output_sum_row0,
                extra={"stage": "export"},
            )

    logger.info(
        "TFLite sanity check passed | input_shape=%s | output_shape=%s | "
        "output_sum_row0=%.4f (≈1.0 expected for softmax)",
        in_shape,
        out_shape,
        output_sum_row0 if output_sum_row0 is not None else float("nan"),
        extra={"stage": "export"},
    )

    return {
        "input_shape": list(in_shape),
        "output_shape": list(out_shape),
        "output_finite": True,
        "output_sum_row0": (
            round(output_sum_row0, 6) if output_sum_row0 is not None else None
        ),
        "n_input_tensors": len(input_details),
        "n_output_tensors": len(output_details),
    }


# =============================================================================
# Step 1.5 — SHA-256 checksum for deployment reproducibility
# =============================================================================

def _compute_file_sha256(path: Path) -> str:
    """
    Compute the SHA-256 hex digest of a file for deployment reproducibility.

    Reads the file in 1 MB chunks to handle large files efficiently without
    loading the entire contents into memory. For the champion TFLite file
    (~0.065 MB) this makes no practical difference, but the chunked approach
    is correct regardless of file size.

    Called by ``export_champion_tflite`` ONLY after the post-conversion
    sanity check has already succeeded (critical-review point #7) — the
    checksum is meant to certify "this is the verified artefact that
    shipped", and a file that fails sanity checking is deleted, not
    checksummed.

    Parameters
    ----------
    path : Path
        Path to the file to hash.

    Returns
    -------
    str
        64-character lowercase hex string (SHA-256 digest).
    """
    h = hashlib.sha256()
    chunk_size = 1024 * 1024  # 1 MB
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# =============================================================================
# Step 1.6 — Core export function
# =============================================================================

def export_champion_tflite(
    saved_model_path: Union[str, Path],
    output_path: Union[str, Path],
    config: Optional[ExperimentConfig] = None,
    representative_dataset_fn: Optional[Callable] = None,
    quantise: bool = True,
    quantisation_mode: Optional[QuantisationMode] = None,
    verify_model: bool = True,
    strict_champion_param_check: bool = True,
    strict_architecture_check: bool = False,
    run_sanity_inference: bool = True,
) -> Dict[str, Any]:
    """
    Authoritative, verified TFLite export for the WLASL champion model.

    This is the Stage 8 deliverable producer — the file this function writes
    is what GesturePredictor.from_config_snapshot(model_path=...) and Stage
    9's webcam demo load. It is NOT the throwaway scratch export in
    src/evaluation/benchmark.py::convert_to_scratch_tflite().

    Result dict key alignment with Stage 8 Revised Specification
    -------------------------------------------------------------
    The returned dict uses the EXACT keys defined in Step 4 of the spec:
        output_path, tflite_disk_mb, savedmodel_disk_mb, param_memory_mb,
        conversion_time_s, quantised, quantisation_mode, used_select_tf_ops,
        keras_params, size_reduction_vs_params_x, size_reduction_vs_savedmodel_x,
        sha256_checksum, model_diagnostics, sanity_check.

    These feed directly into verify.py's write_model_metadata() and
    tflite_verification_report.json without any key renaming.

    Conversion strategy: builtins-first, flex-delegate fallback
    -----------------------------------------------------------
    A first conversion attempt is made WITHOUT SELECT_TF_OPS. Only if that
    raises is a second attempt made with the flex delegate enabled. For the
    champion BiLSTM under TF 2.13 this fallback is expected to fire — Stage
    6's benchmark.py integration testing already confirmed Bidirectional(LSTM)
    requires it — but structuring the code as "attempt, then fall back" keeps
    this function correct and lean for any future non-recurrent model exported
    through the same path.

    A fresh TFLiteConverter is constructed for the fallback attempt rather
    than reusing the failed one — TFLiteConverter instances are not guaranteed
    to be safely re-configurable after a failed .convert() call.

    Failure cleanup (critical-review point #8)
    ---------------------------------------------
    If the post-conversion sanity check (``run_sanity_inference=True``)
    raises, the just-written file at ``output_path`` is deleted BEFORE the
    exception propagates to the caller. A broken .tflite file must never be
    left sitting at a path that ``GesturePredictor`` (Stage 7) or Stage 9's
    webcam demo might load on a subsequent, unrelated invocation — silently
    leaving it there would mean a later, unrelated run could pick up a
    known-broken artefact with no indication anything is wrong. The SHA-256
    checksum is computed only AFTER the sanity check succeeds (critical-
    review point #7), so a deleted-on-failure file never has a checksum
    recorded for it.

    Parameters
    ----------
    saved_model_path : str | Path
        Path to the Keras SavedModel directory.
    output_path : str | Path
        Destination .tflite path (e.g. "models/gesture_bilstm_v1.tflite").
        Parent directories are created if absent.
    config : ExperimentConfig, optional
        The model's training config (recommended: from load_config_snapshot()).
        Drives shape verification and default quantisation mode.
    representative_dataset_fn : Callable, optional
        From make_representative_dataset_fn(). Ignored for DYNAMIC_RANGE and
        FLOAT16 (neither uses calibration data); required in practice for any
        future FULL_INTEGER attempt (currently unsupported).
    quantise : bool, default True
        If False, produces an unquantised float32 TFLite file (diagnostic).
    quantisation_mode : QuantisationMode, optional
        Overrides config.export.quantisation_mode.
    verify_model : bool, default True
        Run _verify_champion_model() before conversion.
    strict_champion_param_check : bool, default True
        Forwarded to _verify_champion_model().
    strict_architecture_check : bool, default False
        Forwarded to _verify_champion_model(). See that function's docstring
        for why this defaults to a warning rather than a hard failure.
    run_sanity_inference : bool, default True
        Run _sanity_check_tflite() after writing the file. On failure, the
        written file is deleted before the exception propagates.

    Returns
    -------
    dict
        {
          output_path, tflite_disk_mb, savedmodel_disk_mb, param_memory_mb,
          conversion_time_s, quantised, quantisation_mode, used_select_tf_ops,
          keras_params, size_reduction_vs_params_x, size_reduction_vs_savedmodel_x,
          sha256_checksum, model_diagnostics, sanity_check
        }

    Raises
    ------
    FileNotFoundError
        If saved_model_path does not exist or lacks required structure.
    ValueError
        If model verification fails, or the conversion fails under BOTH the
        builtins-only and SELECT_TF_OPS attempts.
    """
    import tensorflow as tf

    src = Path(saved_model_path)
    out = Path(output_path)

    # ── Validate SavedModel directory structure before loading ──────────────
    _validate_savedmodel_directory(src)
    out.parent.mkdir(parents=True, exist_ok=True)

    # ── Measure SavedModel disk footprint (actual directory size) ───────────
    savedmodel_disk_bytes = sum(
        f.stat().st_size for f in src.rglob("*") if f.is_file()
    )
    savedmodel_disk_mb = savedmodel_disk_bytes / (1024 ** 2)

    logger.info(
        "Loading SavedModel from %s (disk_size=%.2f MB)...",
        src,
        savedmodel_disk_mb,
        extra={"stage": "export"},
    )
    model = tf.keras.models.load_model(str(src))

    # ── Compute parameter memory footprint (weights only, not graph/assets) ─
    keras_params = int(model.count_params())
    param_memory_mb = (keras_params * _BYTES_PER_FLOAT32) / (1024 ** 2)

    # ── Run champion identity verification ────────────────────────────────────
    diagnostics: Dict[str, Any] = {}
    if verify_model:
        diagnostics = _verify_champion_model(
            model,
            config=config,
            strict_champion_param_check=strict_champion_param_check,
            strict_architecture_check=strict_architecture_check,
        )

    # ── Resolve quantisation mode ─────────────────────────────────────────────
    resolved_mode: Optional[QuantisationMode]
    if not quantise:
        resolved_mode = None
    elif quantisation_mode is not None:
        resolved_mode = quantisation_mode
    elif config is not None:
        resolved_mode = config.export.quantisation_mode
    else:
        resolved_mode = QuantisationMode.DYNAMIC_RANGE
        logger.warning(
            "export_champion_tflite(): quantise=True but neither "
            "quantisation_mode nor config was supplied — defaulting to "
            "QuantisationMode.DYNAMIC_RANGE. Pass config from "
            "load_config_snapshot() to use the run's declared mode instead.",
            extra={"stage": "export"},
        )

    # Critical-review point #5: coerce one more time at the point of use, in
    # case a caller passed a raw string directly as quantisation_mode=...
    # rather than going through load_config_snapshot()'s coercion.
    if resolved_mode is not None and not isinstance(resolved_mode, QuantisationMode):
        try:
            resolved_mode = QuantisationMode(resolved_mode)
        except ValueError as exc:
            raise ValueError(
                f"export_champion_tflite(): resolved quantisation_mode="
                f"{resolved_mode!r} is not a valid QuantisationMode. Valid "
                f"values: {[m.value for m in QuantisationMode]}."
            ) from exc

    # ── Convert: builtins-only first, SELECT_TF_OPS fallback on failure ──────
    t_start = time.perf_counter()
    used_select_tf_ops = False
    tflite_bytes: bytes

    if resolved_mode is not None:
        try:
            converter = tf.lite.TFLiteConverter.from_saved_model(str(src))
            converter = _configure_converter(
                converter,
                resolved_mode,
                representative_dataset_fn,
                use_select_tf_ops=False,
            )
            tflite_bytes = converter.convert()
            logger.info(
                "Builtins-only TFLite conversion succeeded for mode=%s.",
                resolved_mode.value,
                extra={"stage": "export"},
            )
        except NotImplementedError:
            # FULL_INTEGER is an explicit, intentional rejection — never
            # silently retried with a different mode.
            raise
        except Exception as exc:
            logger.warning(
                "Builtins-only TFLite conversion failed (%s: %s) — retrying "
                "with SELECT_TF_OPS flex delegate. This is EXPECTED for the "
                "BiLSTM champion under TF 2.13 (Bidirectional(LSTM(...)) "
                "emits TensorList ops outside the standard builtin op set).",
                type(exc).__name__,
                exc,
                extra={"stage": "export"},
            )
            # Fresh converter — TFLiteConverter state after failure is undefined.
            converter = tf.lite.TFLiteConverter.from_saved_model(str(src))
            converter = _configure_converter(
                converter,
                resolved_mode,
                representative_dataset_fn,
                use_select_tf_ops=True,
            )
            tflite_bytes = converter.convert()
            used_select_tf_ops = True
            logger.info(
                "SELECT_TF_OPS fallback conversion succeeded for mode=%s.",
                resolved_mode.value,
                extra={"stage": "export"},
            )
    else:
        # quantise=False: plain float32 TFLite, no Optimize flags at all.
        converter = tf.lite.TFLiteConverter.from_saved_model(str(src))
        try:
            tflite_bytes = converter.convert()
            logger.info(
                "Unquantised TFLite conversion succeeded.",
                extra={"stage": "export"},
            )
        except Exception as exc:
            logger.warning(
                "Unquantised TFLite conversion failed (%s: %s) — retrying "
                "with SELECT_TF_OPS flex delegate.",
                type(exc).__name__,
                exc,
                extra={"stage": "export"},
            )
            converter = tf.lite.TFLiteConverter.from_saved_model(str(src))
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS,
            ]
            converter._experimental_lower_tensor_list_ops = False
            tflite_bytes = converter.convert()
            used_select_tf_ops = True

    conversion_s = time.perf_counter() - t_start

    # ── Write output file ─────────────────────────────────────────────────────
    out.write_bytes(tflite_bytes)
    tflite_disk_mb = out.stat().st_size / (1024 ** 2)

    # ── Size reduction ratios (computed before sanity check; pure arithmetic,
    #    independent of whether the file ultimately passes sanity checking) ──
    size_reduction_vs_params_x = (
        round(param_memory_mb / tflite_disk_mb, 2)
        if tflite_disk_mb > 0
        else None
    )
    size_reduction_vs_savedmodel_x = (
        round(savedmodel_disk_mb / tflite_disk_mb, 2)
        if tflite_disk_mb > 0
        else None
    )

    # ── Post-conversion sanity inference — BEFORE checksum (critical-review
    #    point #7), and with delete-on-failure (critical-review point #8) ────
    sanity_check: Dict[str, Any] = {}
    if run_sanity_inference:
        expected_in = tuple(diagnostics.get("expected_input_shape", [])) or None
        expected_out = tuple(diagnostics.get("expected_output_shape", [])) or None
        # Critical-review-adjacent fix: when verify_model=False (or it ran
        # but produced no expected-shape diagnostics for some reason),
        # `diagnostics` is empty and expected_in/expected_out would silently
        # be None, meaning _sanity_check_tflite would skip shape validation
        # entirely. Fall back to config-derived expectations, then to the
        # hardcoded champion constants, so the sanity check's shape
        # validation is never silently disabled just because verify_model
        # was turned off.
        if expected_in is None or expected_out is None:
            if config is not None:
                from src.features.constants import LANDMARK_CONFIGS

                lm_slice = LANDMARK_CONFIGS[str(config.data.landmark_config)]
                expected_in = expected_in or (
                    None,
                    int(config.data.sequence_length),
                    lm_slice.stop - lm_slice.start,
                )
                expected_out = expected_out or (None, int(config.num_classes))
            else:
                expected_in = expected_in or _EXPECTED_CHAMPION_INPUT_SHAPE
                expected_out = expected_out or _EXPECTED_CHAMPION_OUTPUT_SHAPE

        try:
            sanity_check = _sanity_check_tflite(
                out,
                expected_input_shape=expected_in,
                expected_output_shape=expected_out,
            )
        except Exception:
            # Critical-review point #8: never leave a broken artefact at the
            # production path. Delete before re-raising so a subsequent,
            # unrelated run cannot accidentally load a known-bad file.
            logger.error(
                "export_champion_tflite(): post-conversion sanity check "
                "FAILED for %s — deleting the written file before "
                "propagating the error. This file must never be treated as "
                "deployable.",
                out,
                extra={"stage": "export"},
            )
            try:
                out.unlink(missing_ok=True)
            except OSError as unlink_exc:
                logger.error(
                    "export_champion_tflite(): failed to delete broken "
                    "artefact at %s (%s: %s). MANUALLY DELETE THIS FILE "
                    "before it is mistaken for a valid export.",
                    out, type(unlink_exc).__name__, unlink_exc,
                    extra={"stage": "export"},
                )
            raise

    # ── Compute SHA-256 checksum — only after sanity check has succeeded
    #    (or was explicitly skipped) (critical-review point #7) ──────────────
    sha256_checksum = _compute_file_sha256(out)

    result: Dict[str, Any] = {
        "output_path": str(out.resolve()),
        "tflite_disk_mb": round(tflite_disk_mb, 4),
        "savedmodel_disk_mb": round(savedmodel_disk_mb, 2),
        "param_memory_mb": round(param_memory_mb, 4),
        "conversion_time_s": round(conversion_s, 2),
        "quantised": quantise,
        "quantisation_mode": resolved_mode.value if resolved_mode is not None else None,
        "used_select_tf_ops": used_select_tf_ops,
        "keras_params": keras_params,
        "size_reduction_vs_params_x": size_reduction_vs_params_x,
        "size_reduction_vs_savedmodel_x": size_reduction_vs_savedmodel_x,
        "sha256_checksum": sha256_checksum,
        "model_diagnostics": diagnostics,
        "sanity_check": sanity_check,
    }

    logger.info(
        "TFLite export complete | output=%s | "
        "tflite_size=%.4f MB | savedmodel_size=%.2f MB | "
        "param_memory=%.4f MB | "
        "reduction_vs_params=%.2fx | reduction_vs_savedmodel=%.2fx | "
        "conversion_time=%.2fs | quantised=%s mode=%s | "
        "used_select_tf_ops=%s | sha256=%s",
        out,
        tflite_disk_mb,
        savedmodel_disk_mb,
        param_memory_mb,
        size_reduction_vs_params_x or 0.0,
        size_reduction_vs_savedmodel_x or 0.0,
        conversion_s,
        quantise,
        result["quantisation_mode"],
        used_select_tf_ops,
        sha256_checksum[:16] + "...",
        extra={"stage": "export"},
    )

    # ── Size target check ─────────────────────────────────────────────────────
    if tflite_disk_mb > _MAX_TFLITE_SIZE_MB:
        logger.warning(
            "export_champion_tflite(): output file is %.2f MB, exceeding the "
            "project's %.0f MB TFLite size target (Part 2 project constants) "
            "despite quantisation. This is unexpected for the "
            "68,771-parameter champion (~0.065 MB expected) — verify the "
            "correct SavedModel was exported.",
            tflite_disk_mb,
            _MAX_TFLITE_SIZE_MB,
            extra={"stage": "export"},
        )

    return result


# =============================================================================
# Step 1.7 — High-level orchestration (config snapshot → verified .tflite)
# =============================================================================

def export_champion(
    config_snapshot_path: Union[str, Path] = _DEFAULT_CONFIG_SNAPSHOT_PATH,
    saved_model_path: Union[str, Path] = _DEFAULT_SAVED_MODEL_PATH,
    output_path: Union[str, Path] = _DEFAULT_TFLITE_OUTPUT_PATH,
    build_representative_dataset: Optional[bool] = None,
    representative_dataset_split: str = "val",
    quantise: bool = True,
    verify_model: bool = True,
    run_sanity_inference: bool = True,
    strict_champion_param_check: bool = True,
    strict_architecture_check: bool = False,
) -> Dict[str, Any]:
    """
    End-to-end Step 1 orchestration: config snapshot → SavedModel → verified,
    sanity-checked, checksummed .tflite file.

    This is the function pipelines/run_export.py (or an ad-hoc notebook
    cell) should call. It wires together everything else in this module
    plus, when needed, FeaturePipeline + GestureDataset for the
    representative-dataset generator.

    GestureDataset construction cost
    ---------------------------------
    GestureDataset.__init__ preloads ALL THREE splits (~339 clips, ~20-30 MB
    into RAM) even though only the val split's clips are needed here —
    _validate_split_coverage() requires a non-empty training split to
    construct at all. This cost is now PAID ONLY WHEN ACTUALLY NEEDED — see
    build_representative_dataset resolution below (critical-review point #4).

    build_representative_dataset resolution (revised — critical-review #4)
    --------------------------------------------------------------------------
    - True  → always build it (e.g. a caller explicitly wants the generator
              available for inspection, or is about to call FULL_INTEGER
              once implemented).
    - False → never build it (fastest path).
    - None (default) → build it ONLY if the resolved quantisation mode is
              in ``_QUANTISATION_MODES_REQUIRING_REPR_DATASET`` (currently:
              FULL_INTEGER only). For the project default — DYNAMIC_RANGE —
              and for FLOAT16, this now means the GestureDataset/
              FeaturePipeline construction is SKIPPED ENTIRELY by default,
              removing the wasted preload-all-339-clips cost the critical
              review correctly identified. The moment FULL_INTEGER is
              implemented (or a caller passes quantisation_mode=
              FULL_INTEGER explicitly via a future override), this same
              default resolution starts building it automatically — no
              code change required here.

    Parameters
    ----------
    config_snapshot_path : str | Path
        Path to config_snapshot.yaml. Defaults to champion's snapshot.
    saved_model_path : str | Path
        Path to the Keras SavedModel directory. Defaults to champion's.
    output_path : str | Path
        Destination .tflite path. Defaults to "models/gesture_bilstm_v1.tflite".
    build_representative_dataset : bool, optional
        See resolution logic above.
    representative_dataset_split : str, default "val"
        Which split to draw representative samples from.
    quantise : bool, default True
    verify_model : bool, default True
    run_sanity_inference : bool, default True
    strict_champion_param_check : bool, default True
    strict_architecture_check : bool, default False
        See ``_verify_champion_model`` docstring.

    Returns
    -------
    dict
        Everything from export_champion_tflite(), plus:
            config_snapshot_path : str (resolved)
            config_hash          : str
    """
    config = load_config_snapshot(config_snapshot_path)

    resolved_quant_mode = config.export.quantisation_mode
    if not isinstance(resolved_quant_mode, QuantisationMode):
        resolved_quant_mode = QuantisationMode(resolved_quant_mode)

    need_repr = build_representative_dataset
    if need_repr is None:
        # Critical-review point #4: only pay the GestureDataset construction
        # cost when the resolved quantisation mode would actually consume
        # the generator. Neither DYNAMIC_RANGE (project default) nor
        # FLOAT16 does.
        need_repr = resolved_quant_mode in _QUANTISATION_MODES_REQUIRING_REPR_DATASET
        if not quantise:
            need_repr = False

    representative_fn: Optional[Callable] = None
    if need_repr:
        from src.features.dataset import GestureDataset
        from src.features.pipeline import FeaturePipeline

        logger.info(
            "export_champion(): building GestureDataset for representative "
            "dataset generation (split='%s', quantisation_mode=%s)...",
            representative_dataset_split,
            resolved_quant_mode.value,
            extra={"stage": "export"},
        )
        pipeline = FeaturePipeline(config)
        dataset = GestureDataset(config, pipeline)
        representative_fn = make_representative_dataset_fn(
            dataset,
            split=representative_dataset_split,
            n_samples=int(config.export.representative_dataset_size),
            seed=int(config.seed),
        )
    else:
        logger.info(
            "export_champion(): skipping representative-dataset construction "
            "— quantisation_mode=%s does not consume calibration data "
            "(GestureDataset preload of all splits avoided). Pass "
            "build_representative_dataset=True to force construction.",
            resolved_quant_mode.value,
            extra={"stage": "export"},
        )

    result = export_champion_tflite(
        saved_model_path=saved_model_path,
        output_path=output_path,
        config=config,
        representative_dataset_fn=representative_fn,
        quantise=quantise,
        quantisation_mode=None,  # resolved from config.export.quantisation_mode
        verify_model=verify_model,
        strict_champion_param_check=strict_champion_param_check,
        strict_architecture_check=strict_architecture_check,
        run_sanity_inference=run_sanity_inference,
    )

    result["config_snapshot_path"] = str(Path(config_snapshot_path).resolve())
    result["config_hash"] = config.config_hash

    return result


# =============================================================================
# Step 1.8 — Export manifest writer (convenience)
# =============================================================================

def write_export_manifest(
    result: Dict[str, Any],
    output_dir: Union[str, Path] = "models",
    filename: str = "export_manifest.json",
) -> Path:
    """
    Persist the export result dict as a JSON manifest alongside the .tflite file.

    This is an OPTIONAL convenience function — export_champion_tflite()
    returns all the same data in its result dict. Callers that want the
    manifest on disk (e.g. CI/CD pipelines, notebooks) call this after the
    export completes. It is NOT called automatically, keeping the core
    export function's responsibilities narrow.

    The manifest includes:
      - All result dict keys from export_champion_tflite()
      - created_utc timestamp for provenance
      - tensorflow_version for reproducibility

    Parameters
    ----------
    result : dict
        The dict returned by export_champion_tflite() or export_champion().
    output_dir : str | Path
        Directory where the manifest is written. Defaults to "models/".
    filename : str
        Manifest filename. Defaults to "export_manifest.json".

    Returns
    -------
    Path
        Resolved absolute path to the written manifest file.
    """
    from datetime import datetime, timezone

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / filename

    try:
        import tensorflow as tf

        tf_version = tf.__version__
    except ImportError:
        tf_version = "unknown"

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tensorflow_version": tf_version,
        **result,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    logger.info(
        "Export manifest written → %s",
        manifest_path.resolve(),
        extra={"stage": "export"},
    )
    return manifest_path.resolve()


# =============================================================================
# Import-time self-check
# =============================================================================

def _self_check() -> None:
    """Cheap, dependency-free sanity check on module constants."""
    assert _EXPECTED_CHAMPION_PARAMS == 68_771, (
        "convert.py: _EXPECTED_CHAMPION_PARAMS has drifted from the "
        "documented champion parameter count (handoff Part 4). "
        "Expected 68,771 for bilstm_hands_only_v4_aug."
    )
    assert _EXPECTED_CHAMPION_INPUT_SHAPE == (None, 100, 126), (
        "convert.py: _EXPECTED_CHAMPION_INPUT_SHAPE has drifted from the "
        "documented champion input shape (seq_len=100, hands_only=126)."
    )
    assert _EXPECTED_CHAMPION_OUTPUT_SHAPE == (None, 35), (
        "convert.py: _EXPECTED_CHAMPION_OUTPUT_SHAPE has drifted from the "
        "documented 35-class output shape."
    )
    assert _TFLITE_EXPECTED_INPUT_SHAPE == (1, 100, 126), (
        "convert.py: _TFLITE_EXPECTED_INPUT_SHAPE must be (1, 100, 126) "
        "for the TFLite static-batch shape."
    )
    assert _TFLITE_EXPECTED_OUTPUT_SHAPE == (1, 35), (
        "convert.py: _TFLITE_EXPECTED_OUTPUT_SHAPE must be (1, 35)."
    )
    assert _BYTES_PER_FLOAT32 == 4, (
        "convert.py: _BYTES_PER_FLOAT32 must be 4 to match architectures.py, "
        "factory.py, benchmark.py, and predictor.py."
    )
    assert _KNOWN_CHAMPION_CONFIG_HASH.startswith("5809193d"), (
        "convert.py: _KNOWN_CHAMPION_CONFIG_HASH has changed from the "
        "value in artifacts/experiments/bilstm_hands_only_v4_aug/"
        "config_snapshot.yaml. Verify before trusting the informational "
        "config_hash log line."
    )
    assert len(_KNOWN_CHAMPION_CONFIG_HASH) == 64, (
        "convert.py: _KNOWN_CHAMPION_CONFIG_HASH must be a 64-char "
        "SHA-256 hex string."
    )
    assert 0.0 < _SOFTMAX_SUM_TOLERANCE <= 0.5, (
        "convert.py: _SOFTMAX_SUM_TOLERANCE must be in (0.0, 0.5]."
    )
    assert _MAX_TFLITE_SIZE_MB == 10.0, (
        "convert.py: _MAX_TFLITE_SIZE_MB must match the project target "
        "(Part 2: ≤10 MB post-quantisation TFLite)."
    )
    assert len(_EXPECTED_CHAMPION_LAYER_SIGNATURE) >= 1, (
        "convert.py: _EXPECTED_CHAMPION_LAYER_SIGNATURE must not be empty."
    )
    assert all(
        isinstance(substr, str) and isinstance(cnt, int) and cnt >= 1
        for substr, cnt in _EXPECTED_CHAMPION_LAYER_SIGNATURE
    ), (
        "convert.py: _EXPECTED_CHAMPION_LAYER_SIGNATURE entries must be "
        "(str, positive int) pairs."
    )
    assert QuantisationMode.FULL_INTEGER in _QUANTISATION_MODES_REQUIRING_REPR_DATASET, (
        "convert.py: FULL_INTEGER must be in "
        "_QUANTISATION_MODES_REQUIRING_REPR_DATASET — it is the only mode "
        "this module would ever need calibration data for."
    )
    assert QuantisationMode.DYNAMIC_RANGE not in _QUANTISATION_MODES_REQUIRING_REPR_DATASET, (
        "convert.py: DYNAMIC_RANGE must NOT be in "
        "_QUANTISATION_MODES_REQUIRING_REPR_DATASET — it quantises weights "
        "only and never consumes a representative dataset."
    )


if __debug__:
    _self_check()


# =============================================================================
# Public API surface
# =============================================================================

__all__ = [
    "load_config_snapshot",
    "make_representative_dataset_fn",
    "export_champion_tflite",
    "export_champion",
    "write_export_manifest",
    # Exposed for Stage 8 test suite (test_tflite_export.py)
    "_verify_champion_model",
    "_check_layer_architecture_signature",
    "_normalise_keras_shape",
    "_sanity_check_tflite",
    "_validate_savedmodel_directory",
    "_EXPECTED_CHAMPION_PARAMS",
    "_EXPECTED_CHAMPION_INPUT_SHAPE",
    "_EXPECTED_CHAMPION_OUTPUT_SHAPE",
    "_EXPECTED_CHAMPION_LAYER_SIGNATURE",
    "_TFLITE_EXPECTED_INPUT_SHAPE",
    "_TFLITE_EXPECTED_OUTPUT_SHAPE",
    "_KNOWN_CHAMPION_CONFIG_HASH",
]


# =============================================================================
# CLI entry point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Stage 8 Step 1 — Export the WLASL champion model "
            "(bilstm_hands_only_v4_aug) to a verified, sanity-checked, "
            "checksummed .tflite file."
        )
    )
    parser.add_argument(
        "--config-snapshot",
        default=_DEFAULT_CONFIG_SNAPSHOT_PATH,
        help="Path to config_snapshot.yaml (default: champion's).",
    )
    parser.add_argument(
        "--saved-model",
        default=_DEFAULT_SAVED_MODEL_PATH,
        help="Path to the Keras SavedModel directory (default: champion's).",
    )
    parser.add_argument(
        "--output",
        default=_DEFAULT_TFLITE_OUTPUT_PATH,
        help="Destination .tflite path (default: models/gesture_bilstm_v1.tflite).",
    )
    parser.add_argument(
        "--no-quantise",
        action="store_true",
        help="Export an unquantised float32 .tflite file (diagnostic only).",
    )
    parser.add_argument(
        "--build-representative-dataset",
        action="store_true",
        help=(
            "Force-build the representative dataset generator even though "
            "the resolved quantisation mode (DYNAMIC_RANGE or FLOAT16) does "
            "not consume it. Useful for inspecting/testing the generator "
            "ahead of a future FULL_INTEGER implementation."
        ),
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip the pre-conversion model verification check.",
    )
    parser.add_argument(
        "--skip-sanity-check",
        action="store_true",
        help="Skip the post-conversion interpreter sanity inference.",
    )
    parser.add_argument(
        "--allow-non-champion-params",
        action="store_true",
        help=(
            "Do not hard-fail on a champion-shape/wrong-parameter-count "
            "mismatch (log a warning instead). Use only when intentionally "
            "exporting a non-champion run that shares the champion's I/O shape."
        ),
    )
    parser.add_argument(
        "--strict-architecture-check",
        action="store_true",
        help=(
            "Hard-fail (instead of warn) if shape and parameter count both "
            "match the champion but the layer architecture (Masking + "
            "2x Bidirectional) does not."
        ),
    )
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help=(
            "Write an export_manifest.json alongside the .tflite file "
            "with provenance metadata (TF version, timestamp, checksums)."
        ),
    )
    args = parser.parse_args()

    export_result = export_champion(
        config_snapshot_path=args.config_snapshot,
        saved_model_path=args.saved_model,
        output_path=args.output,
        build_representative_dataset=(
            True if args.build_representative_dataset else None
        ),
        quantise=not args.no_quantise,
        verify_model=not args.skip_verify,
        run_sanity_inference=not args.skip_sanity_check,
        strict_champion_param_check=not args.allow_non_champion_params,
        strict_architecture_check=args.strict_architecture_check,
    )

    if args.write_manifest:
        manifest_dir = Path(args.output).parent
        manifest_path = write_export_manifest(export_result, output_dir=manifest_dir)
        export_result["manifest_path"] = str(manifest_path)

    print(json.dumps(export_result, indent=2, default=str))