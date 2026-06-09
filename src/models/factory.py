"""
src/models/factory.py
======================
Model factory for the WLASL 35-class gesture recognition system.

This module provides a single, config-driven entry point for model
construction. Every training script, pipeline entry point, and test that
needs a model calls ``build_model(cfg)`` — never a builder function directly.
This ensures:

  1. Model selection is always driven by ``cfg.model.name``, making
     experiment runs fully reproducible from config files alone.
  2. The dispatch table is defined in one place; adding a new architecture
     requires editing only ``architectures.py`` (add builder) and this file
     (add dispatch entry).
  3. Optional pipeline shape validation catches mismatches between the
     feature engineering config and the model input config before TF graph
     construction, producing actionable error messages instead of cryptic
     tensor shape errors inside ``model.fit()``.
  4. All post-build model statistics needed by MLflow and run manifests are
     centralised in ``get_model_summary_dict()``.

Single responsibility principle
---------------------------------
This file does NOT contain any model construction logic. All layer
definitions, compilation settings, and architectural invariants live in
``src/models/architectures.py``. The factory's only responsibilities are:

  - Name normalisation and dispatch.
  - Pre-build validation (config sanity, pipeline shape consistency).
  - Post-build validation (return type guard).
  - Post-build summary extraction.
  - A clean public API that downstream code can depend on without
    importing from ``architectures.py`` directly.

Pipeline shape validation
--------------------------
When a ``FeaturePipeline`` instance is passed to ``build_model()``, the
factory validates that::

    pipeline.output_shape == (cfg.data.sequence_length, pipeline.feature_dim)

IMPORTANT: ``cfg.data.feature_dim`` is NOT a field on ``DataConfig``.
The feature dimension is derived from ``cfg.data.landmark_config`` inside
``FeaturePipeline.__init__()`` via the ``LANDMARK_CONFIGS`` slice mapping.
The factory therefore reads the resolved dimension from
``pipeline.feature_dim`` (the property on FeaturePipeline) rather than
from the config directly. This avoids a silent ``AttributeError`` if code
tries to access ``cfg.data.feature_dim`` which does not exist.

The pipeline shape check catches the most common silent misconfiguration:
constructing a pipeline with ``seq_len=60`` but a model config expecting
``seq_len=80``, or using ``landmark_config="hands_only"`` (feature_dim=126)
while the model was built expecting 225 features. Both of these would
otherwise surface only when ``model.fit()`` encounters the first batch —
producing a cryptic ``InvalidArgumentError: Incompatible shapes`` deep
inside the TF graph.

The validation is NOT done by comparing ``cfg.data.feature_dim`` (which
does not exist as a direct config field) but by comparing the pipeline's
own ``output_shape`` property against the model's declared input shape
(``seq_len, feature_dim``), both derived from the same pipeline instance.

OmegaConf attribute safety
----------------------------
``dense.yaml`` does not define ``num_layers`` or ``recurrent_dropout``.
``get_model_summary_dict()`` uses ``getattr(cfg.model, field, default)``
when extracting optional fields, making it safe for any model type.

Module-level exports
---------------------
    build_model            — primary public API: config + optional pipeline → compiled model
    get_model_summary_dict — post-build statistics dict for MLflow / manifests
    get_valid_model_names  — sorted list of recognised model name strings
    VALID_MODEL_NAMES      — frozenset of valid names (for fast membership tests)
"""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np

from src.models.architectures import (
    build_bilstm,
    build_dense,
    build_gru,
    build_lstm,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

#: Mapping from canonical (lower-case) model name → builder function.
#:
#: This is the single source of truth for the set of valid model names.
#: ``VALID_MODEL_NAMES`` and ``get_valid_model_names()`` are both derived
#: from this table, so adding a new architecture never requires editing more
#: than two files: ``architectures.py`` (add builder) and here (add entry).
_BUILDERS: dict[str, Any] = {
    "dense":  build_dense,
    "lstm":   build_lstm,
    "gru":    build_gru,
    "bilstm": build_bilstm,
}

#: Frozenset of valid model name strings (lower-case).
#: Use for fast ``O(1)`` membership tests in CLI argument parsers and tests::
#:
#:     if name not in VALID_MODEL_NAMES:
#:         raise ValueError(...)
VALID_MODEL_NAMES: frozenset[str] = frozenset(_BUILDERS.keys())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise_model_name(cfg: Any) -> str:
    """
    Extract and normalise the model name from the config.

    Reads ``cfg.model.name``, strips surrounding whitespace, and converts
    to lower-case. This means ``"BiLSTM"``, ``"  BILSTM  "``, and
    ``"bilstm"`` all resolve to ``"bilstm"``.

    Accessing ``cfg.model.name`` in a single guarded block ensures the
    factory emits a clear ``AttributeError`` if the config object is
    malformed or the model sub-config is missing — rather than propagating
    a cryptic attribute error from deep inside a builder call.

    Parameters
    ----------
    cfg : ExperimentConfig
        Full frozen experiment config from ``load_config()``.

    Returns
    -------
    str
        Lower-case, stripped model name.

    Raises
    ------
    AttributeError
        If ``cfg.model`` or ``cfg.model.name`` is missing.
    TypeError
        If ``cfg.model.name`` is not a string.
    ValueError
        If the name is empty after stripping.
    """
    try:
        raw_name = cfg.model.name
    except AttributeError as exc:
        raise AttributeError(
            "build_model(): cfg.model.name is missing or cfg.model is not "
            "set. Ensure load_config() completed successfully and that your "
            "model YAML (configs/model/*.yaml) defines a 'name' field. "
            f"Original error: {exc}"
        ) from exc

    if not isinstance(raw_name, str):
        raise TypeError(
            f"cfg.model.name must be a string, got {type(raw_name).__name__}: "
            f"{raw_name!r}. "
            "Check that your model YAML uses a plain string value "
            "(e.g. name: bilstm) without quotes or special characters."
        )

    name = raw_name.strip().lower()
    if not name:
        raise ValueError(
            "cfg.model.name is empty after stripping whitespace. "
            f"Got: {raw_name!r}. "
            f"Valid names: {sorted(_BUILDERS.keys())}."
        )
    return name


def _validate_pipeline_shape(cfg: Any, pipeline: Any) -> None:
    """
    Verify that the pipeline's output shape matches the model's expected input.

    This guard catches the most common silent misconfiguration: constructing
    a ``FeaturePipeline`` with one ``seq_len`` or ``landmark_config`` but
    then building a model intended for different dimensions. Without it the
    mismatch surfaces only inside ``model.fit()`` as a cryptic TensorFlow
    ``InvalidArgumentError: Incompatible shapes``.

    Shape source of truth
    ---------------------
    ``cfg.data.feature_dim`` is NOT a field on ``DataConfig``. The feature
    dimension is computed inside ``FeaturePipeline.__init__()`` from
    ``cfg.data.landmark_config`` via the ``LANDMARK_CONFIGS`` slice mapping::

        hands_only → 126 dims
        pose_only  →  99 dims
        full       → 225 dims

    The factory therefore reads the resolved dimension from
    ``pipeline.feature_dim`` (a property on ``FeaturePipeline``) rather
    than attempting to access ``cfg.data.feature_dim``.

    The validation compares ``pipeline.output_shape`` — the single tuple
    that ``FeaturePipeline`` guarantees for every ``__call__`` return —
    against the pair ``(cfg.data.sequence_length, pipeline.feature_dim)``.
    Both sides must agree exactly.

    Parameters
    ----------
    cfg : ExperimentConfig
        Reads ``cfg.data.sequence_length`` (int).
    pipeline : FeaturePipeline
        Must expose:
            - ``.output_shape`` property → ``(int, int)``
            - ``.feature_dim``   property → ``int``
            - ``.sequence_length`` property → ``int``

    Raises
    ------
    AttributeError
        If ``pipeline`` does not expose ``.output_shape``.
    TypeError
        If ``pipeline.output_shape`` is not a tuple or list.
    ValueError
        If the shape is not length 2, or if the dimensions disagree.
    """
    # ── 1. Attribute presence guard ───────────────────────────────────────
    if not hasattr(pipeline, "output_shape"):
        raise AttributeError(
            "build_model() received a pipeline object without an "
            "'output_shape' attribute. Expected a FeaturePipeline instance "
            "with a property returning (seq_len, feature_dim). "
            f"Got: {type(pipeline).__name__}."
        )

    raw_shape = pipeline.output_shape

    # ── 2. Type guard ─────────────────────────────────────────────────────
    if not isinstance(raw_shape, (tuple, list)):
        raise TypeError(
            f"pipeline.output_shape must return a tuple or list, "
            f"got {type(raw_shape).__name__}: {raw_shape!r}. "
            "Check FeaturePipeline.output_shape implementation."
        )

    actual_shape = tuple(raw_shape)

    # ── 3. Length guard ───────────────────────────────────────────────────
    if len(actual_shape) != 2:
        raise ValueError(
            f"pipeline.output_shape must have exactly 2 elements "
            f"(seq_len, feature_dim), got {len(actual_shape)}: {actual_shape}. "
            "Check FeaturePipeline.output_shape implementation."
        )

    # ── 4. Value comparison ───────────────────────────────────────────────
    # Expected dimensions come from:
    #   seq_len     → cfg.data.sequence_length  (direct DataConfig field)
    #   feature_dim → pipeline.feature_dim      (derived from landmark_config)
    #
    # NOTE: cfg.data.feature_dim does NOT exist as a DataConfig field.
    # Using pipeline.feature_dim is the only correct approach here.
    expected_seq_len     = int(cfg.data.sequence_length)
    expected_feature_dim = int(pipeline.feature_dim)
    expected_shape       = (expected_seq_len, expected_feature_dim)

    if actual_shape == expected_shape:
        logger.debug(
            f"Pipeline shape validated | "
            f"output_shape={actual_shape} | "
            f"seq_len={expected_seq_len} | "
            f"feature_dim={expected_feature_dim} | "
            f"landmark_config={getattr(pipeline, 'landmark_config', '?')}",
            extra={"stage": "factory"},
        )
        return

    # Build a targeted diagnostic that names exactly which dimension diverged.
    seq_match  = actual_shape[0] == expected_seq_len
    feat_match = actual_shape[1] == expected_feature_dim
    mismatch_parts: list[str] = []

    if not seq_match:
        mismatch_parts.append(
            f"sequence_length: pipeline reports {actual_shape[0]}, "
            f"cfg.data.sequence_length={expected_seq_len}"
        )
    if not feat_match:
        lm_config = getattr(cfg.data, "landmark_config", "?")
        mismatch_parts.append(
            f"feature_dim: pipeline reports {actual_shape[1]}, "
            f"expected {expected_feature_dim} "
            f"(derived from landmark_config='{lm_config}')"
        )

    raise ValueError(
        "Pipeline output shape does not match model config — "
        "this would cause a shape error inside model.fit().\n"
        f"  pipeline.output_shape           = {actual_shape}\n"
        f"  expected (seq_len, feature_dim) = {expected_shape}\n"
        f"  Divergent field(s):\n"
        + "\n".join(f"    - {p}" for p in mismatch_parts)
        + "\n"
        "Ensure the FeaturePipeline was constructed from the same config "
        "object passed to build_model(). Check that configs/data/*.yaml "
        "and configs/model/*.yaml are consistent."
    )


def _verify_model_return(model: Any, model_name: str) -> None:
    """
    Verify that a builder returned a valid compiled ``tf.keras.Model``.

    This guard makes future architecture additions safer: if a new builder
    accidentally returns ``None``, a dict, or a partially-constructed object,
    the error is caught immediately with a clear message rather than surfacing
    as a cryptic ``AttributeError`` on ``model.count_params()`` or during
    ``model.fit()``.

    Parameters
    ----------
    model : Any
        Return value from a builder function.
    model_name : str
        Normalised model name, used in the error message.

    Raises
    ------
    TypeError
        If ``model`` is not an instance of ``tf.keras.Model``.
    """
    import tensorflow as tf

    if not isinstance(model, tf.keras.Model):
        raise TypeError(
            f"Builder for model '{model_name}' did not return a tf.keras.Model. "
            f"Got {type(model).__name__}: {model!r}. "
            "Every builder in architectures.py must return a compiled "
            "tf.keras.Model instance."
        )


# ---------------------------------------------------------------------------
# Primary public API
# ---------------------------------------------------------------------------

def build_model(cfg: Any, pipeline: Optional[Any] = None) -> Any:
    """
    Build and return a compiled Keras model driven by ``cfg.model.name``.

    This is the single entry point for model construction used by every
    training script, pipeline entry point, and test in the project. It
    normalises the model name, optionally validates the pipeline shape,
    dispatches to the correct builder, verifies the return type, and logs
    timing and parameter counts.

    Name normalisation
    ------------------
    ``cfg.model.name`` is read defensively, stripped, and lower-cased before
    lookup, so ``"BiLSTM"``, ``"bilstm"``, and ``"  BILSTM  "`` all resolve
    correctly. Config attribute errors are caught and re-raised with clear
    messages.

    Pipeline shape validation (optional but strongly recommended)
    -------------------------------------------------------------
    When ``pipeline`` is provided, ``build_model()`` verifies that::

        pipeline.output_shape == (cfg.data.sequence_length, pipeline.feature_dim)

    IMPORTANT: ``cfg.data.feature_dim`` is NOT a DataConfig field. The
    feature dimension is derived from ``cfg.data.landmark_config`` inside
    ``FeaturePipeline``. The factory always reads the resolved dimension from
    ``pipeline.feature_dim`` — never from the config directly.

    ``pipeline`` should always be provided in production training scripts.
    It may be omitted in unit tests that build models with synthetic configs
    and no pipeline object.

    Builder return type validation
    --------------------------------
    The return value of every builder is verified to be a ``tf.keras.Model``
    instance before the function returns. This makes future architecture
    additions safer.

    Parameters
    ----------
    cfg : ExperimentConfig
        Full frozen experiment config from ``load_config()``. Must have:
            cfg.model.name          — one of {"dense", "lstm", "gru", "bilstm"}
            cfg.data.sequence_length — positive int
            cfg.data.landmark_config — one of {"full", "hands_only", "pose_only"}
            cfg.num_classes          — 35 for this project
        Additional fields are read by the individual builders as needed.
        Note: ``cfg.data.feature_dim`` does NOT exist as a DataConfig field;
        feature dimensions are resolved via the pipeline instance.
    pipeline : FeaturePipeline | None, optional
        If provided, its ``.output_shape`` and ``.feature_dim`` are used to
        validate consistency between the pipeline and the config before
        dispatching to the builder. Strongly recommended in all production
        paths. Default: None (shape validation skipped — use only in tests
        that explicitly document the omission).

    Returns
    -------
    tf.keras.Model
        Compiled model with:
            - input shape:  (None, seq_len, feature_dim)
            - output shape: (None, n_classes)
            - loss:         sparse_categorical_crossentropy
            - optimizer:    Adam(lr=cfg.training.learning_rate)
            - metrics:      [accuracy]

    Raises
    ------
    AttributeError
        If ``cfg.model`` or ``cfg.model.name`` is missing.
        If ``pipeline`` is provided but does not expose ``.output_shape``.
    TypeError
        If ``cfg.model.name`` is not a string.
        If ``pipeline.output_shape`` is not a tuple or list.
        If the builder returns something other than a ``tf.keras.Model``.
    ValueError
        If ``cfg.model.name`` (normalised) is not in ``VALID_MODEL_NAMES``.
        If ``pipeline.output_shape`` has the wrong length or mismatches config.
        If any builder-level validation fails (propagated from architectures.py).

    Examples
    --------
    Standard production usage in train.py::

        pipeline = FeaturePipeline(cfg)
        model = build_model(cfg, pipeline=pipeline)

    Unit test without pipeline (shape validation skipped)::

        model = build_model(cfg)
        assert model.name == "lstm_classifier"
        assert model.count_params() > 0
    """
    t0 = time.perf_counter()

    # ── Step 1: Normalise and validate model name ─────────────────────────
    name = _normalise_model_name(cfg)

    if name not in _BUILDERS:
        raise ValueError(
            f"Unknown model name: '{cfg.model.name}' (normalised: '{name}'). "
            f"Valid names: {sorted(_BUILDERS.keys())}. "
            "Check cfg.model.name in your model YAML "
            "(configs/model/dense.yaml, lstm.yaml, gru.yaml, bilstm.yaml)."
        )

    # ── Step 2: Pipeline shape validation (optional but recommended) ──────
    if pipeline is not None:
        _validate_pipeline_shape(cfg, pipeline)
    else:
        logger.debug(
            "build_model(): pipeline=None — input shape validation skipped. "
            "Provide a FeaturePipeline instance in production training scripts "
            "to catch seq_len / feature_dim mismatches before model.fit().",
            extra={"stage": "factory"},
        )

    # ── Step 3: Log build context ─────────────────────────────────────────
    seq_len     = int(cfg.data.sequence_length)
    feature_dim = int(pipeline.feature_dim) if pipeline is not None else "unknown"
    lm_config   = getattr(cfg.data, "landmark_config", "unknown")

    logger.info(
        f"Building model | "
        f"name='{name}' | "
        f"seq_len={seq_len} | "
        f"feature_dim={feature_dim} | "
        f"landmark_config={lm_config} | "
        f"num_classes={cfg.num_classes}",
        extra={"stage": "factory"},
    )

    # ── Step 4: Dispatch to the correct builder ───────────────────────────
    builder = _BUILDERS[name]
    model   = builder(cfg)

    # ── Step 5: Verify return type ────────────────────────────────────────
    _verify_model_return(model, name)

    # ── Step 6: Post-build logging ─────────────────────────────────────────
    elapsed      = time.perf_counter() - t0
    total_params = model.count_params()
    size_mb      = total_params * 4 / (1024 ** 2)

    logger.info(
        f"Model ready | "
        f"name='{name}' | "
        f"keras_name='{model.name}' | "
        f"total_params={total_params:,} | "
        f"size_mb_estimate={size_mb:.3f} | "
        f"build_time={elapsed:.2f}s",
        extra={"stage": "factory"},
    )

    return model


# ---------------------------------------------------------------------------
# Post-build summary
# ---------------------------------------------------------------------------

def get_model_summary_dict(model: Any) -> dict[str, Any]:
    """
    Return a JSON-serialisable dictionary of post-build model statistics.

    Called in two contexts:

    1. **MLflow param logging** (``run_training.py``):
       ``mlflow.log_params(summary)`` records the actual post-build param
       counts alongside the hyperparameter values from the config. Using
       ``model.count_params()`` (not a config estimate) guarantees the
       logged count matches the actual trained model.

    2. **Run manifest** (``train.py`` Step 5d):
       ``run_manifest.json`` includes ``"model_param_count"`` so readers of
       the manifest can verify a checkpoint without loading the full model.

    Implementation notes
    --------------------
    - ``model.count_params()`` — Keras-standard method, always available on
      compiled or uncompiled models.
    - Trainable parameters use ``np.prod(v.shape)`` over
      ``model.trainable_weights`` to avoid ``w.numpy()`` calls (which can
      fail under non-eager execution / distribution strategies).
    - Weight file size estimate uses ``float32 = 4 bytes/param`` for the
      uncompressed Keras SavedModel. TFLite dynamic-range quantisation
      reduces this by approximately 4× in practice.
    - Shapes are returned as plain lists so the dict is JSON-serialisable
      without custom encoders. The batch dimension is ``None`` and is
      preserved as-is.

    Parameters
    ----------
    model : tf.keras.Model
        A compiled or uncompiled Keras model returned by ``build_model()``.

    Returns
    -------
    dict[str, Any]
        All values are JSON-serialisable. Keys:

        param_count            (int)   total parameters (trainable + frozen)
        trainable_params       (int)   trainable parameters
        non_trainable_params   (int)   frozen / non-trainable parameters
        model_size_mb_estimate (float) uncompressed float32 weight size in MB
        model_name             (str)   Keras model name (e.g. "lstm_classifier")
        input_shape            (list)  e.g. [None, 60, 225]
        output_shape           (list)  e.g. [None, 35]

    Raises
    ------
    AttributeError
        If ``model`` does not expose the standard Keras model attributes
        (``count_params``, ``trainable_weights``, ``input_shape``,
        ``output_shape``). These are present on every ``tf.keras.Model``.

    Examples
    --------
    ::

        model   = build_model(cfg, pipeline=pipeline)
        summary = get_model_summary_dict(model)

        # MLflow
        mlflow.log_params({
            "total_params":  summary["param_count"],
            "model_size_mb": summary["model_size_mb_estimate"],
        })

        # Run manifest
        run_manifest["model_param_count"] = summary["param_count"]
    """
    total_params = model.count_params()
    trainable    = int(sum(np.prod(v.shape) for v in model.trainable_weights))
    non_train    = total_params - trainable
    size_mb      = round(total_params * 4 / (1024 ** 2), 4)

    def _shape_to_list(shape: Any) -> Any:
        """Recursively convert a shape tuple to a JSON-serialisable list."""
        if isinstance(shape, (list, tuple)):
            return [_shape_to_list(s) for s in shape]
        return shape  # int or None — both JSON-serialisable

    return {
        "param_count":            total_params,
        "trainable_params":       trainable,
        "non_trainable_params":   non_train,
        "model_size_mb_estimate": size_mb,
        "model_name":             str(model.name),
        "input_shape":            _shape_to_list(model.input_shape),
        "output_shape":           _shape_to_list(model.output_shape),
    }


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def get_valid_model_names() -> list[str]:
    """
    Return a sorted list of all recognised model name strings.

    Used in two places:

    1. **CLI argument validation** in ``run_training.py``::

           valid = get_valid_model_names()
           if args.model not in valid:
               parser.error(f"--model must be one of {valid}")

    2. **Test assertions** in ``test_model_factory.py``::

           assert "bilstm" in get_valid_model_names()
           assert len(get_valid_model_names()) == 4

    The list is always sorted for deterministic comparison regardless of
    dict insertion order.

    Returns
    -------
    list[str]
        Sorted list, e.g. ``["bilstm", "dense", "gru", "lstm"]``.
    """
    return sorted(_BUILDERS.keys())


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------

__all__ = [
    "VALID_MODEL_NAMES",
    "build_model",
    "get_model_summary_dict",
    "get_valid_model_names",
]