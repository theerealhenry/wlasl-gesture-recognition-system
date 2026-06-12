"""
src/models/architectures.py
============================
Model architecture builder functions for the WLASL 35-class gesture
recognition system.

This module defines all four architecture variants used in the Stage 5
multi-model experiment matrix. Every builder accepts a validated config
object and returns a compiled ``tf.keras.Model`` ready for training.

Why a single file for all architectures
-----------------------------------------
Keeping all four builders here allows:

  1. Side-by-side comparison of architectural choices during code review.
  2. Clean import surface for the factory (``src/models/factory.py``).
  3. A single place to enforce invariants that must hold across ALL
     architectures (output units, compilation settings, Masking presence).
  4. Prevention of the common mistake of inconsistent compile settings
     when builders live in separate files.

Architecture overview
----------------------

+------------------+--------------------------------------------------+
| Name             | Purpose                                          |
+==================+==================================================+
| Dense baseline   | Proves temporal modelling is necessary.          |
|                  | Receives flattened (seq_len × feature_dim)       |
|                  | vector — no mechanism to learn temporal          |
|                  | dependencies. Expected accuracy: 40–55%.         |
+------------------+--------------------------------------------------+
| LSTM             | Primary workhorse for Groups 2, 3, 4 ablations. |
|                  | Two-layer stacked LSTM with Masking.             |
|                  | Expected accuracy: 60–70%.                       |
+------------------+--------------------------------------------------+
| GRU              | Streamlined alternative to LSTM. Fewer params   |
|                  | per unit; expected to match LSTM accuracy at     |
|                  | lower latency. Same stacking pattern.            |
+------------------+--------------------------------------------------+
| BiLSTM           | Champion model candidate. Bidirectional reads    |
|                  | forward + backward; useful because sign          |
|                  | resolution (how a sign ends) is as discriminative|
|                  | as onset. hidden_units//2 per direction keeps   |
|                  | total output dimension comparable to LSTM.       |
+------------------+--------------------------------------------------+

Critical invariants (never violate)
--------------------------------------
  - ``Dense(n_classes, activation="softmax")`` output for all models.
    n_classes is read from ``cfg.num_classes`` (35 for this project).
    ``N_CLASSES`` in this module is a documentation constant and a
    fallback for the self-check; the builders always use cfg.num_classes.

  - ``Masking(mask_value=0.0)`` as the first non-Input layer in EVERY
    recurrent architecture (LSTM, GRU, BiLSTM). 35.28% of frames in the
    WLASL dataset are zero-filled (both-hands-absent frames). Without the
    Masking layer the recurrent cells update their hidden state on
    zero-fill frames, wasting capacity on uninformative content. With it,
    Keras skips the recurrent update for any timestep where the entire
    feature vector is zero — the semantically correct inductive bias.

    Why mask_value=0.0 is safe for this pipeline
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The critique raised a valid general concern: normalisation can
    accidentally map real frames to all-zeros, causing Masking to
    suppress them. This is NOT possible in this pipeline for the
    following reason, verified against pipeline.py:

    ``_wrist_relative_normalise()`` computes detection masks from the
    ORIGINAL un-modified array (``arr[:, SLICE].any(axis=1)``) and then
    subtracts ONLY the wrist landmark (landmark 0, 3 values) from all 21
    landmarks in the detected hand. After subtraction:

      - The wrist itself becomes (0, 0, z) where z is the depth coordinate.
        This is a zero x,y but non-zero z in most frames.
      - Landmarks 1–20 (all non-wrist) remain non-zero — they were already
        expressed relative to the frame origin; subtracting one landmark
        (the wrist) cannot make all of them simultaneously zero unless
        every finger joint coincides with the wrist position, which is
        anatomically impossible.
      - The .any() mask over the full 63-element LH or RH slice remains
        True for any genuinely detected hand frame.

    Zero-fill frames (all-zero slices) originate exclusively from
    MediaPipe detection failures and zero-padding, never from
    normalisation of real detections. Changing to a sentinel like -999.0
    would break the zero-fill semantic that the entire feature engineering
    pipeline (AugmentationPipeline, GestureDataset, extractor.py) is built
    on. mask_value=0.0 is the correct, pipeline-consistent choice.

  - ``loss="sparse_categorical_crossentropy"``.
    Labels in ``GestureDataset`` are integer class indices (int32), not
    one-hot vectors. Using ``categorical_crossentropy`` would require
    one-hot labels and silently produce wrong gradients otherwise.

  - ``metrics=["accuracy"]``.
    macro-F1 is NOT registered as a Keras compile metric. It is computed
    via ``sklearn.metrics.f1_score(average="macro")`` in the
    ``MacroF1Callback`` (``src/models/train.py``) and logged to MLflow.
    Attempting to add it as a Keras metric produces numerically different
    results from sklearn (different averaging implementation) and would
    cause mismatches between per-epoch logs and evaluation reports.

  - ``return_sequences=True`` for every recurrent layer except the final
    one in the stack. A second LSTM/GRU/BiLSTM layer needs the full
    sequence of hidden states from the first layer, not the single final
    state. Collapsing to the final state at the first layer discards all
    temporal structure before the second layer can process it.

  - ``hidden_units // 2`` per direction in BiLSTM.
    A ``Bidirectional(LSTM(u))`` layer produces a ``2u``-dimensional output
    (forward + backward concatenated). Using ``hidden_units`` per direction
    would double the output dimension compared to the unidirectional LSTM,
    confounding the architecture comparison with a parameter-count advantage.
    Using ``hidden_units // 2`` produces the same output width as the
    unidirectional LSTM, keeping parameter counts approximately comparable.
    Note: total param counts are NOT identical (BiLSTM is slightly lower
    for this input size due to smaller hidden state in L2), but the
    comparison is fair enough to isolate directionality as the variable.

Dense baseline design
---------------------
The Dense baseline proves temporal modelling is necessary.
Its first Dense layer is capped at min(512, flattened_dim // 2) to
prevent memory explosion during the seq_len ablation (Group 3). At
seq_len=100 the flattened dimension is 22,500; an uncapped Dense(512)
first layer would create 11.5M parameters vs the LSTM's ~110K, which
distorts the ablation comparison at high sequence lengths. The cap keeps
the first Dense layer parameter count proportional to the actual input
size across all seq_len values.

The Dense model intentionally destroys temporal structure via Flatten —
this is the defining property of the baseline, not GlobalAveragePooling1D
(which preserves some temporal information via averaging and would weaken
the scientific argument). The baseline must have MORE parameters than the
LSTM to ensure any accuracy gap is attributable to architecture, not
parameter count.

CuDNN note for recurrent layers
---------------------------------
In TensorFlow 2.13.x, setting ``recurrent_dropout > 0`` disables the
CuDNN-accelerated LSTM/GRU kernel in favour of a software implementation.
This project targets CPU training (236 clips; no CuDNN available by
default), so this has no practical impact. If GPU training is added in a
future stage, remove ``recurrent_dropout`` or set it to 0.0 in the
champion config to re-enable the fast kernel.

Regularisation strategy
------------------------
  - Recurrent dropout (``recurrent_dropout``) applied to the recurrent
    connections (h_{t-1} → h_t gates). Default 0.1 (lighter than input
    dropout to avoid gradient instability in deep stacks).
  - Input dropout (``dropout``) applied to input-to-hidden connections
    and the Dense head. Default 0.3.
  - NO L2 weight regularisation, NO BatchNormalization, NO LayerNorm.
    With 236 training clips and batch_size=32 (≈7 batches), batch
    statistics are too noisy for BatchNorm. LayerNorm is not part of the
    Stage 5 experiment specification. The dominant regularisation
    mechanisms are Stage 4 augmentation and Stage 1 signer-independent
    splitting.

Relation to Stage 5 experiment groups
-----------------------------------------
  Group 1 — Architecture comparison:
      All four builders with seq60, augmentation=none, same seed.
      Isolates architecture as the only variable.

  Groups 2, 3, 4 — Ablation studies:
      ``build_lstm()`` only. Architecture is fixed; augmentation strategy,
      sequence length, and landmark config vary one at a time.

  Champion run:
      ``build_bilstm()`` (or the Group 1 winner) with optimal settings from
      Groups 2–4. ``hidden_units=128`` rather than the ablation value of 64.

Module-level exports
--------------------
    build_dense   — Dense feedforward baseline (temporal structure destroyed)
    build_lstm    — Two-layer stacked LSTM
    build_gru     — Two-layer stacked GRU
    build_bilstm  — Two-layer stacked Bidirectional LSTM
    N_CLASSES     — Documentation constant: 35 (builders use cfg.num_classes)
"""

from __future__ import annotations

import numpy as np
from typing import Any, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Documentation constant: number of output classes in this project.
#: All builders read ``cfg.num_classes`` at runtime — they do NOT reference
#: this constant directly. This value exists for two purposes:
#:   1. Self-documentation and the import-time self-check.
#:   2. Fallback verification against cfg.num_classes in _check_n_classes().
#: If the label map changes (e.g. WLASL-50), update cfg.num_classes in
#: base.yaml; this constant is then only for audit/documentation.
N_CLASSES: int = 35

#: Practical minimum hidden units for recurrent layers. Below this threshold
#: the recurrent state vector is too small to discriminate between 35 sign
#: classes with any reliability. This is a practical heuristic, not a
#: mathematical lower bound — a mathematically a 4-unit LSTM can produce 35
#: output classes, but will not learn meaningful representations on this task.
_MIN_HIDDEN_UNITS: int = 16

#: Upper bound for warning on recurrent dropout. Higher values cause gradient
#: instability in stacked recurrent networks on short sequences (≤100 frames).
_MAX_RECURRENT_DROPOUT: float = 0.5

#: Upper bound for warning on input/head dropout. Above this level the Dense
#: head receives too little signal to converge on a 236-clip dataset.
_MAX_DROPOUT: float = 0.7

#: Minimum sequence length. Enforced to catch YAML misconfiguration before
#: TensorFlow graph construction begins.
_MIN_SEQ_LEN: int = 1

#: Minimum feature dimension. Enforced to catch landmark_config mismatches
#: before graph construction.
_MIN_FEATURE_DIM: int = 1

#: Cap on the first Dense hidden layer in ``build_dense()``.
#: Prevents memory explosion during the seq_len ablation (Group 3).
#: At seq_len=100, feature_dim=225: flattened=22,500. Without a cap,
#: Dense(512) would create 22,500 × 512 = 11.5M weights in the first layer
#: alone. The cap scales the first hidden layer proportionally to the input
#: so that parameter counts do not explode at high seq_len values.
_DENSE_LAYER1_MAX: int = 512

#: Valid range for learning rate. Values outside (0, 1] are almost certainly
#: configuration errors; negative LR or LR > 1 causes divergence silently.
_LR_MIN: float = 1e-7
_LR_MAX: float = 1.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# architectures.py — replace the entire _get_input_shape function

def _get_input_shape(cfg: Any, pipeline: Optional[Any] = None) -> tuple[int, int]:
    """
    Extract and validate ``(seq_len, feature_dim)`` from the config and pipeline.

    feature_dim is NOT a DataConfig field — it is derived from
    cfg.data.landmark_config inside FeaturePipeline. When a pipeline instance
    is available (production path via build_model(cfg, pipeline=pipeline)),
    feature_dim is read from pipeline.feature_dim. When no pipeline is
    provided (unit tests with synthetic configs), feature_dim is derived
    from landmark_config via the LANDMARK_CONFIGS slice mapping.

    Parameters
    ----------
    cfg : ExperimentConfig
    pipeline : FeaturePipeline | None

    Returns
    -------
    tuple[int, int]
        (sequence_length, feature_dim)
    """
    from src.features.constants import LANDMARK_CONFIGS

    seq_len = int(cfg.data.sequence_length)

    if seq_len < _MIN_SEQ_LEN:
        raise ValueError(
            f"cfg.data.sequence_length={seq_len} is below the minimum "
            f"({_MIN_SEQ_LEN}). Check your data config YAML."
        )

    if pipeline is not None and hasattr(pipeline, "feature_dim"):
        feature_dim = int(pipeline.feature_dim)
    else:
        # Fallback for unit tests without a pipeline instance.
        lm_config = getattr(cfg.data, "landmark_config", "full")
        if lm_config not in LANDMARK_CONFIGS:
            raise ValueError(
                f"cfg.data.landmark_config='{lm_config}' is not in LANDMARK_CONFIGS. "
                f"Valid values: {sorted(LANDMARK_CONFIGS.keys())}."
            )
        lm_slice    = LANDMARK_CONFIGS[lm_config]
        feature_dim = lm_slice.stop - lm_slice.start

    if feature_dim < _MIN_FEATURE_DIM:
        raise ValueError(
            f"feature_dim={feature_dim} is below the minimum ({_MIN_FEATURE_DIM}). "
            "Check cfg.data.landmark_config. "
            "Valid values: full=225, hands_only=126, pose_only=99."
        )

    return seq_len, feature_dim


def _check_n_classes(cfg: Any) -> int:
    """
    Read and validate the number of output classes from the config.

    Using ``cfg.num_classes`` (confirmed present in ``ExperimentConfig`` via
    ``dataset.py``) makes builders config-driven and forward-compatible with
    WLASL-50 or larger label maps without touching this file.

    The value is cross-checked against the module-level ``N_CLASSES`` constant
    at runtime; a mismatch between config and this file is logged as an ERROR
    (not a crash) so that training can still proceed with the config value,
    which is the authoritative source.

    Parameters
    ----------
    cfg : ExperimentConfig
        Reads ``cfg.num_classes`` (int).

    Returns
    -------
    int
        Number of output classes (35 for this project).

    Raises
    ------
    ValueError
        If ``cfg.num_classes < 2`` — cannot classify fewer than 2 classes.
    """
    n = int(cfg.num_classes)

    if n < 2:
        raise ValueError(
            f"cfg.num_classes={n} is invalid. A classifier requires at least "
            "2 output classes. Check base.yaml (num_classes: 35)."
        )

    if n != N_CLASSES:
        logger.error(
            f"cfg.num_classes={n} differs from architectures.py N_CLASSES={N_CLASSES}. "
            "Using cfg.num_classes as the authoritative value. "
            "Update N_CLASSES in architectures.py if the label map has changed.",
            extra={"stage": "model"},
        )

    return n


def _validate_recurrent_params(
    hidden_units: int,
    dropout: float,
    recurrent_dropout: float,
    caller: str,
) -> None:
    """
    Validate recurrent layer hyper-parameters with full range checks.

    Raises ``ValueError`` for values that will cause TensorFlow errors or
    silent failures. Logs ``WARNING`` for values that are technically valid
    but likely to cause training problems on this specific dataset.

    Parameters
    ----------
    hidden_units : int
    dropout : float
    recurrent_dropout : float
    caller : str
        Name of the calling function, used in error messages.

    Raises
    ------
    ValueError
        If ``hidden_units`` is below ``_MIN_HIDDEN_UNITS``.
        If ``dropout`` is outside ``[0.0, 1.0)``.
        If ``recurrent_dropout`` is outside ``[0.0, 1.0)``.
    """
    if hidden_units < _MIN_HIDDEN_UNITS:
        raise ValueError(
            f"{caller}: hidden_units={hidden_units} is below the practical "
            f"minimum of {_MIN_HIDDEN_UNITS}. At this size the recurrent "
            "state vector is too small to learn reliable representations "
            "for 35 sign classes on a 236-clip dataset. "
            f"Use at least {_MIN_HIDDEN_UNITS} (ablation default: 64)."
        )

    # Full range check — catches negative dropout AND values >= 1.0
    # (which TensorFlow silently accepts but produces all-zero outputs).
    if not (0.0 <= dropout < 1.0):
        raise ValueError(
            f"{caller}: dropout={dropout} is outside the valid range [0.0, 1.0). "
            "Dropout probability must be non-negative and strictly less than 1. "
            "Check cfg.model.dropout in your model YAML."
        )

    if not (0.0 <= recurrent_dropout < 1.0):
        raise ValueError(
            f"{caller}: recurrent_dropout={recurrent_dropout} is outside the "
            "valid range [0.0, 1.0). Check cfg.model.recurrent_dropout in your "
            "model YAML."
        )

    if dropout > _MAX_DROPOUT:
        logger.warning(
            f"{caller}: dropout={dropout} exceeds {_MAX_DROPOUT}. "
            "This level of regularisation may prevent convergence on a "
            "236-clip dataset. Consider values in [0.2, 0.5].",
            extra={"stage": "model"},
        )

    if recurrent_dropout > _MAX_RECURRENT_DROPOUT:
        logger.warning(
            f"{caller}: recurrent_dropout={recurrent_dropout} exceeds "
            f"{_MAX_RECURRENT_DROPOUT}. High recurrent dropout causes gradient "
            "instability in stacked recurrent networks on short sequences. "
            "Consider values in [0.0, 0.2].",
            extra={"stage": "model"},
        )


def _validate_dropout(dropout: float, caller: str) -> None:
    """
    Validate a single dropout value for non-recurrent layers (e.g. Dense head).

    Raises
    ------
    ValueError
        If ``dropout`` is outside ``[0.0, 1.0)``.
    """
    if not (0.0 <= dropout < 1.0):
        raise ValueError(
            f"{caller}: dropout={dropout} is outside the valid range [0.0, 1.0). "
            "Check cfg.model.dropout in your model YAML."
        )
    if dropout > _MAX_DROPOUT:
        logger.warning(
            f"{caller}: dropout={dropout} exceeds {_MAX_DROPOUT}. "
            "May prevent convergence on a 236-clip dataset.",
            extra={"stage": "model"},
        )


def _compile_model(model: Any, cfg: Any, model_name: str) -> None:
    """
    Compile a Keras model with the project-standard settings.

    Enforced in a single function so compilation settings are IDENTICAL
    across all four architectures. Divergent learning rates, losses, or
    metrics between architectures would corrupt Group 1 ablation results.

    Compilation details
    -------------------
    - ``Adam`` with ``learning_rate`` from ``cfg.training.learning_rate``.
      Validated to be in (_LR_MIN, _LR_MAX] before use.
    - ``sparse_categorical_crossentropy``: labels from ``GestureDataset``
      are integer class indices (int32), NOT one-hot vectors.
    - ``["accuracy"]``: the only Keras compile metric. macro-F1 is computed
      via sklearn in ``MacroF1Callback`` (``src/models/train.py``) and
      logged to MLflow. Adding macro-F1 here would produce numerically
      different results from sklearn and corrupt evaluation comparisons.

    Parameters
    ----------
    model : tf.keras.Model
        Uncompiled model.
    cfg : ExperimentConfig
        Reads ``cfg.training.learning_rate``.
    model_name : str
        For log messages only.

    Raises
    ------
    ValueError
        If ``cfg.training.learning_rate`` is outside (_LR_MIN, _LR_MAX].
    """
    import tensorflow as tf

    lr = float(cfg.training.learning_rate)

    if not (_LR_MIN < lr <= _LR_MAX):
        raise ValueError(
            f"_compile_model({model_name}): learning_rate={lr} is outside "
            f"the valid range ({_LR_MIN}, {_LR_MAX}]. "
            "Check cfg.training.learning_rate in base.yaml. "
            "Typical values: 1e-3 (Adam default), 5e-4, 1e-4."
        )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    logger.debug(
        f"Compiled '{model_name}' | "
        f"optimizer=Adam(lr={lr:.2e}) | "
        f"loss=sparse_categorical_crossentropy | "
        f"metrics=[accuracy] | "
        f"note: macro_f1 computed via sklearn in MacroF1Callback",
        extra={"stage": "model"},
    )


def _log_model_summary(model: Any, model_name: str) -> None:
    """
    Log parameter counts and estimated weight file size at INFO level.

    Uses ``model.count_params()`` for the total (the Keras-standard method)
    and ``np.prod(v.shape)`` to count trainable parameters per weight tensor.
    This avoids ``w.numpy()`` calls which can be problematic under
    non-eager execution modes (distributed strategies, graph mode).

    Parameters
    ----------
    model : tf.keras.Model
        Compiled or uncompiled model.
    model_name : str
        Name used in the log line.
    """
    total     = model.count_params()
    trainable = int(sum(np.prod(v.shape) for v in model.trainable_weights))
    non_train = total - trainable

    # float32 = 4 bytes; convert to MB. This is the uncompressed weight size.
    # TFLite dynamic-range quantisation reduces this by ~4×.
    size_mb = total * 4 / (1024 ** 2)

    logger.info(
        f"Model '{model_name}' built | "
        f"total_params={total:,} | "
        f"trainable={trainable:,} | "
        f"non_trainable={non_train:,} | "
        f"estimated_weight_size={size_mb:.2f} MB (float32, pre-quantisation)",
        extra={"stage": "model"},
    )


# ---------------------------------------------------------------------------
# Architecture 1: Dense Baseline
# ---------------------------------------------------------------------------

def build_dense(cfg: Any, pipeline: Optional[Any] = None) -> Any:
    """
    Build and compile a Dense feedforward baseline model.

    Purpose
    -------
    The Dense baseline proves that temporal modelling is necessary. By
    receiving the same (seq_len, feature_dim) input as the recurrent models
    but immediately flattening it, the Dense model has no mechanism to learn
    temporal dependencies — all frame order information is destroyed at
    Flatten. Its expected accuracy (40–55%) compared to the LSTM (60–70%)
    is the quantitative justification for using sequence models.

    Why Flatten and not GlobalAveragePooling1D
    -------------------------------------------
    ``GlobalAveragePooling1D`` computes a temporal mean, which does preserve
    some temporal information (different from a random ordering of frames).
    Using it would weaken the "temporal modelling is necessary" claim because
    the baseline would no longer be a purely non-temporal model. ``Flatten``
    is the only correct choice: it makes the model position-independent in the
    worst possible way (each position is an independent input feature), which
    is exactly the null hypothesis we are testing against.

    Architecture
    ------------
    Input → Flatten → Dense(H1, relu) → Dropout → Dense(256, relu)
          → Dropout → Dense(n_classes, softmax)

    Where H1 = min(_DENSE_LAYER1_MAX, flattened_dim // 2).

    The first-layer cap prevents memory explosion during the seq_len ablation
    (Group 3). Without it:
      - seq_len=60:  flattened=13,500 → Dense(512) → 6.9M params (acceptable)
      - seq_len=100: flattened=22,500 → Dense(512) → 11.5M params (≈100× LSTM)

    The cap scales H1 proportionally to the actual flattened dimension, keeping
    the Dense model parameter count bounded relative to the recurrent baselines
    across all seq_len ablation values.

    Dense-specific notes
    --------------------
    - NO Masking layer. Masking operates on the sequence axis; after Flatten
      there is no sequence axis. Zero-fill frames are baked into the flat
      vector and the model must learn to ignore them — exactly the harder
      generalisation problem that motivates the recurrent architecture.
    - NO BatchNormalization. With 236 clips and batch_size=32 (≈7 batches),
      batch statistics are too noisy to be useful.

    Parameters
    ----------
    cfg : ExperimentConfig
        Reads: cfg.data.sequence_length, cfg.data.feature_dim,
               cfg.model.dropout, cfg.training.learning_rate, cfg.num_classes.

    Returns
    -------
    tf.keras.Model
        Compiled Dense model with input shape (seq_len, feature_dim).

    Raises
    ------
    ValueError
        If input dimensions are invalid, dropout is out of range, or
        learning rate is out of range.
    """
    import tensorflow as tf

    seq_len, feature_dim = _get_input_shape(cfg, pipeline)
    n_classes = _check_n_classes(cfg)
    dropout   = float(cfg.model.dropout)

    _validate_dropout(dropout, "build_dense")

    flattened_dim = seq_len * feature_dim
    # Cap the first hidden layer to prevent memory explosion at high seq_len.
    # min(512, 22500//2=11250) → 512 for seq60; still 512 for seq100 since
    # 512 < 22500//2. The real protection comes at smaller feature dims or
    # if _DENSE_LAYER1_MAX is tightened for future large-seq ablations.
    # The key invariant: H1 ≤ flattened_dim // 2, so the first layer is
    # always a compression step, never an expansion.
    h1 = min(_DENSE_LAYER1_MAX, max(256, flattened_dim // 2))
    h2 = 256

    logger.info(
        f"Building Dense baseline | "
        f"input=({seq_len}, {feature_dim}) | "
        f"flattened_dim={flattened_dim:,} | "
        f"h1={h1} (cap={_DENSE_LAYER1_MAX}) | "
        f"h2={h2} | "
        f"n_classes={n_classes} | "
        f"dropout={dropout}",
        extra={"stage": "model"},
    )

    # ── Model construction ────────────────────────────────────────────────
    inputs = tf.keras.Input(
        shape=(seq_len, feature_dim),
        name="landmark_sequence",
    )

    # Destroy temporal structure — this is the defining property of the baseline.
    # Output shape: (batch, seq_len * feature_dim)
    x = tf.keras.layers.Flatten(name="flatten_temporal")(inputs)

    # Hidden layer 1 — capped to prevent memory explosion in seq_len ablation
    x = tf.keras.layers.Dense(
        h1,
        activation="relu",
        name="dense_hidden_1",
        kernel_initializer="glorot_uniform",
    )(x)
    x = tf.keras.layers.Dropout(dropout, name="dropout_1")(x)

    # Hidden layer 2 — fixed at 256 for all seq_len configs
    x = tf.keras.layers.Dense(
        h2,
        activation="relu",
        name="dense_hidden_2",
        kernel_initializer="glorot_uniform",
    )(x)
    x = tf.keras.layers.Dropout(dropout, name="dropout_2")(x)

    # Output — n_classes softmax probabilities
    outputs = tf.keras.layers.Dense(
        n_classes,
        activation="softmax",
        name="class_probabilities",
        kernel_initializer="glorot_uniform",
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="dense_baseline")

    _compile_model(model, cfg, "dense_baseline")
    _log_model_summary(model, "dense_baseline")

    return model


# ---------------------------------------------------------------------------
# Architecture 2: Stacked LSTM
# ---------------------------------------------------------------------------

def build_lstm(cfg: Any, pipeline: Optional[Any] = None) -> Any:
    """
    Build and compile a stacked LSTM model for gesture sequence classification.

    Purpose
    -------
    The LSTM is the primary workhorse for the Stage 5 ablation studies.
    Groups 2 (augmentation), 3 (sequence length), and 4 (landmark config)
    all fix the architecture to LSTM so that only one variable changes per
    group. LSTM is preferred over GRU/BiLSTM for the ablations because it
    is the canonical sequence model — deviations in the ablation candidates
    can be attributed to the varying factor, not to a hidden architectural
    advantage of GRU or BiLSTM.

    Architecture
    ------------
    Input → Masking(0.0) → LSTM(u, ret_seq=True) → ... → LSTM(u, ret_seq=False)
          → Dense(u//2, relu) → Dropout → Dense(n_classes, softmax)

    Where u = cfg.model.hidden_units (64 for ablations, 128 for champion run).
    Stack depth = cfg.model.num_layers (2 by default).

    Why Masking is mandatory
    ------------------------
    35.28% of all frames in the WLASL dataset are zero-filled (both hands
    absent simultaneously — confirmed in Notebook 03). Without Masking,
    the LSTM processes these frames and updates its hidden state h_{t-1}→h_t
    despite receiving zero information, wasting recurrent capacity.

    The zero-fill convention is safe for mask_value=0.0 because
    ``FeaturePipeline._wrist_relative_normalise()`` uses per-slot detection
    masks derived from the ORIGINAL pre-normalisation array and ONLY modifies
    detected frames. A genuinely detected hand, after wrist-relative
    normalisation, has its wrist at (0, 0, z) but its 20 remaining landmarks
    at non-zero positions — so the full 63-element LH slice is never all-zero
    for a detected hand. The Masking layer correctly identifies and suppresses
    ONLY the genuine zero-fill (detection-failure) frames.

    return_sequences
    ----------------
    ``return_sequences=not is_final_recurrent`` ensures:
      - All intermediate layers return the full (seq_len, hidden_units) tensor.
      - The final recurrent layer returns only the last hidden state (hidden_units,).
      - Single-layer configs (num_layers=1) work correctly: return_sequences=False.

    CuDNN note
    ----------
    ``recurrent_dropout > 0`` disables the TensorFlow CuDNN LSTM kernel.
    This project targets CPU training (236 clips); no CuDNN penalty applies.
    For GPU deployment, set recurrent_dropout=0.0 to re-enable the fast path.

    Parameters
    ----------
    cfg : ExperimentConfig
        Reads: cfg.data.sequence_length, cfg.data.feature_dim,
               cfg.model.hidden_units, cfg.model.num_layers,
               cfg.model.dropout, cfg.model.recurrent_dropout,
               cfg.training.learning_rate, cfg.num_classes.

    Returns
    -------
    tf.keras.Model
        Compiled LSTM model.

    Raises
    ------
    ValueError
        If hidden_units < _MIN_HIDDEN_UNITS, dropout or recurrent_dropout
        are out of range, learning rate is invalid, or num_layers < 1.
    """
    import tensorflow as tf

    seq_len, feature_dim = _get_input_shape(cfg, pipeline)
    n_classes         = _check_n_classes(cfg)
    hidden_units      = int(cfg.model.hidden_units)
    num_layers        = int(cfg.model.num_layers)
    dropout           = float(cfg.model.dropout)
    recurrent_dropout = float(cfg.model.recurrent_dropout)

    _validate_recurrent_params(hidden_units, dropout, recurrent_dropout, "build_lstm")

    if num_layers < 1:
        raise ValueError(
            f"build_lstm: num_layers={num_layers} must be ≥ 1. "
            "Check cfg.model.num_layers in your model YAML."
        )
    if num_layers > 3:
        logger.warning(
            f"build_lstm: num_layers={num_layers} > 3. Gradient flow degrades "
            "with deep recurrent stacks on sequences ≤100 frames. "
            "The standard ablation value is 2.",
            extra={"stage": "model"},
        )

    logger.info(
        f"Building LSTM | "
        f"input=({seq_len}, {feature_dim}) | "
        f"hidden_units={hidden_units} | "
        f"num_layers={num_layers} | "
        f"dropout={dropout} | "
        f"recurrent_dropout={recurrent_dropout} | "
        f"n_classes={n_classes}",
        extra={"stage": "model"},
    )

    # ── Model construction ────────────────────────────────────────────────
    inputs = tf.keras.Input(
        shape=(seq_len, feature_dim),
        name="landmark_sequence",
    )

    # Masking — MANDATORY first non-Input layer.
    # mask_value=0.0 matches the zero-fill convention; safe per pipeline analysis
    # in the module docstring.
    x = tf.keras.layers.Masking(
        mask_value=0.0,
        name="zero_fill_mask",
    )(inputs)

    # Stacked LSTM layers
    for layer_idx in range(num_layers):
        is_final_recurrent = (layer_idx == num_layers - 1)
        layer_name = f"lstm_layer_{layer_idx + 1}"

        x = tf.keras.layers.LSTM(
            units=hidden_units,
            return_sequences=not is_final_recurrent,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            # recurrent_initializer="orthogonal" (Keras default for LSTM):
            # orthogonal initialisation is better than glorot for recurrent
            # connections because it preserves gradient norm across timesteps.
            name=layer_name,
        )(x)

    # Dense projection head
    x = tf.keras.layers.Dense(
        hidden_units // 2,
        activation="relu",
        name="dense_projection",
        kernel_initializer="glorot_uniform",
    )(x)
    x = tf.keras.layers.Dropout(dropout, name="dropout_head")(x)

    # Output
    outputs = tf.keras.layers.Dense(
        n_classes,
        activation="softmax",
        name="class_probabilities",
        kernel_initializer="glorot_uniform",
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="lstm_classifier")

    _compile_model(model, cfg, "lstm_classifier")
    _log_model_summary(model, "lstm_classifier")

    return model


# ---------------------------------------------------------------------------
# Architecture 3: Stacked GRU
# ---------------------------------------------------------------------------

def build_gru(cfg: Any, pipeline: Optional[Any] = None) -> Any:
    """
    Build and compile a stacked GRU model for gesture sequence classification.

    Purpose
    -------
    The GRU tests whether the LSTM's cell-state mechanism provides a
    measurable benefit on this specific dataset. GRU merges the cell state
    and hidden state into a single gated state vector, using two gates
    (reset, update) instead of LSTM's three (input, forget, output). This
    reduces the parameter count per unit by approximately one-third while
    preserving the ability to model long-range dependencies.

    Expected result: GRU accuracy within 1–3 percentage points of LSTM with
    meaningfully lower latency (fewer FLOPs per timestep). If accuracy matches,
    GRU is the preferred deployment candidate — smaller model, faster TFLite
    inference on mobile (the target deployment platform).

    Architecture
    ------------
    Input → Masking(0.0) → GRU(u, ret_seq=True) → ... → GRU(u, ret_seq=False)
          → Dense(u//2, relu) → Dropout → Dense(n_classes, softmax)

    Identical structural pattern to ``build_lstm()`` — only the recurrent cell
    type differs. This architectural parity is the scientific requirement: when
    Group 1 results show GRU within 1–3pp of LSTM, the reviewer can attribute
    the difference to the cell mechanism, not to hidden architectural asymmetry.

    GRU-specific notes
    ------------------
    - ``reset_after=True`` (TF 2.13 default): uses the CuDNN-compatible
      matrix formulation. No practical difference on CPU training; ensures
      portability to GPU environments without retraining.
    - ``recurrent_dropout > 0`` disables the CuDNN GRU kernel, same as
      LSTM. Not a concern for this project's CPU training target.

    Parameters
    ----------
    cfg : ExperimentConfig
        Same fields as ``build_lstm``.

    Returns
    -------
    tf.keras.Model
        Compiled GRU model.

    Raises
    ------
    ValueError
        If hidden_units < _MIN_HIDDEN_UNITS, dropout or recurrent_dropout
        are out of range, learning rate is invalid, or num_layers < 1.
    """
    import tensorflow as tf

    seq_len, feature_dim = _get_input_shape(cfg, pipeline)
    n_classes         = _check_n_classes(cfg)
    hidden_units      = int(cfg.model.hidden_units)
    num_layers        = int(cfg.model.num_layers)
    dropout           = float(cfg.model.dropout)
    recurrent_dropout = float(cfg.model.recurrent_dropout)

    _validate_recurrent_params(hidden_units, dropout, recurrent_dropout, "build_gru")

    if num_layers < 1:
        raise ValueError(
            f"build_gru: num_layers={num_layers} must be ≥ 1."
        )

    logger.info(
        f"Building GRU | "
        f"input=({seq_len}, {feature_dim}) | "
        f"hidden_units={hidden_units} | "
        f"num_layers={num_layers} | "
        f"dropout={dropout} | "
        f"recurrent_dropout={recurrent_dropout} | "
        f"n_classes={n_classes}",
        extra={"stage": "model"},
    )

    # ── Model construction ────────────────────────────────────────────────
    inputs = tf.keras.Input(
        shape=(seq_len, feature_dim),
        name="landmark_sequence",
    )

    # Masking — MANDATORY. Same zero-fill convention and safety rationale as LSTM.
    x = tf.keras.layers.Masking(
        mask_value=0.0,
        name="zero_fill_mask",
    )(inputs)

    # Stacked GRU layers — identical depth and return_sequences logic to LSTM
    for layer_idx in range(num_layers):
        is_final_recurrent = (layer_idx == num_layers - 1)
        layer_name = f"gru_layer_{layer_idx + 1}"

        x = tf.keras.layers.GRU(
            units=hidden_units,
            return_sequences=not is_final_recurrent,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            # reset_after=True: CuDNN-compatible formulation (TF default).
            name=layer_name,
        )(x)

    # Dense projection head — identical to LSTM for fair comparison
    x = tf.keras.layers.Dense(
        hidden_units // 2,
        activation="relu",
        name="dense_projection",
        kernel_initializer="glorot_uniform",
    )(x)
    x = tf.keras.layers.Dropout(dropout, name="dropout_head")(x)

    # Output
    outputs = tf.keras.layers.Dense(
        n_classes,
        activation="softmax",
        name="class_probabilities",
        kernel_initializer="glorot_uniform",
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="gru_classifier")

    _compile_model(model, cfg, "gru_classifier")
    _log_model_summary(model, "gru_classifier")

    return model


# ---------------------------------------------------------------------------
# Architecture 4: Stacked Bidirectional LSTM
# ---------------------------------------------------------------------------

def build_bilstm(cfg: Any, pipeline: Optional[Any] = None) -> Any:
    """
    Build and compile a stacked Bidirectional LSTM (champion model candidate).

    Purpose
    -------
    BiLSTM is the champion model candidate — the architecture most likely to
    exceed the ≥70% validation macro-F1 target. It reads each sequence in
    both the forward and backward temporal directions, giving every timestep
    access to context from both the past (pre-stroke preparation) and the
    future (post-stroke resolution). For ASL gesture recognition this is
    particularly valuable because:

      - Sign onset (how a sign begins from rest) is discriminative.
      - Sign resolution (how a hand returns to rest) is equally discriminative.
      - Co-articulation effects (the influence of surrounding signs on the
        current sign shape) are better captured bidirectionally.

    Expected accuracy: best chance of ≥70% val macro-F1, especially when
    combined with spatial_temporal augmentation and seq_len=80 (the
    highest-priority ablation per Notebook 04 — 97% truncation at seq_len=60
    indicates most sign content lies beyond 60 frames).

    Architecture
    ------------
    Input → Masking(0.0)
          → Bidirectional(LSTM(u//2, ret_seq=True),  merge=concat) [× (num_layers-1)]
          → Bidirectional(LSTM(u//2, ret_seq=False), merge=concat)
          → Dense(u//2, relu) → Dropout
          → Dense(n_classes, softmax)

    Where u = cfg.model.hidden_units (64 for ablations, 128 for champion run).

    CRITICAL: hidden_units // 2 per direction
    -----------------------------------------
    A ``Bidirectional(LSTM(k))`` layer with ``merge_mode="concat"`` produces
    a ``2k``-dimensional output (forward k-dim concatenated with backward k-dim).
    Using ``hidden_units`` per direction would produce ``2 × hidden_units``
    output, increasing parameter counts and making the Group 1 comparison
    confounded by parameter count.

    Using ``hidden_units // 2`` per direction:
        Output dim = 2 × (hidden_units // 2) = hidden_units

    This keeps the output dimension identical to the unidirectional LSTM.
    Total parameter counts are approximately comparable (BiLSTM is slightly
    lower due to smaller hidden state in L2 — verified analytically for
    hidden_units=64, seq_len=60):
        LSTM total ≈ 110,499 params
        BiLSTM total ≈ 94,115 params

    The difference is small relative to the 236-clip dataset and does not
    meaningfully confound the architecture comparison.

    BiLSTM-specific validation
    --------------------------
    ``hidden_units`` must be even to split symmetrically between forward and
    backward directions. An odd value is rejected immediately (before any TF
    graph construction) with a clear error message.

    Masking with Bidirectional
    --------------------------
    Keras's ``Bidirectional`` wrapper correctly propagates the mask from the
    preceding ``Masking`` layer to BOTH the forward and backward LSTM cells.
    Zero-fill frames are suppressed in both temporal directions.

    Parameters
    ----------
    cfg : ExperimentConfig
        Reads: cfg.data.sequence_length, cfg.data.feature_dim,
               cfg.model.hidden_units (must be even),
               cfg.model.num_layers, cfg.model.dropout,
               cfg.model.recurrent_dropout, cfg.training.learning_rate,
               cfg.num_classes.

    Returns
    -------
    tf.keras.Model
        Compiled BiLSTM model.

    Raises
    ------
    ValueError
        If hidden_units is odd, hidden_units // 2 < _MIN_HIDDEN_UNITS,
        dropout or recurrent_dropout are out of range, learning rate is
        invalid, or num_layers < 1.
    """
    import tensorflow as tf

    seq_len, feature_dim = _get_input_shape(cfg, pipeline)
    n_classes         = _check_n_classes(cfg)
    hidden_units      = int(cfg.model.hidden_units)
    num_layers        = int(cfg.model.num_layers)
    dropout           = float(cfg.model.dropout)
    recurrent_dropout = float(cfg.model.recurrent_dropout)

    _validate_recurrent_params(hidden_units, dropout, recurrent_dropout, "build_bilstm")

    # BiLSTM-specific: hidden_units must be even for symmetric direction split.
    if hidden_units % 2 != 0:
        raise ValueError(
            f"build_bilstm: hidden_units={hidden_units} must be even. "
            "BiLSTM uses hidden_units // 2 per direction; an odd value "
            "produces an asymmetric split. "
            "Use a power-of-2 value: 32, 64, 128, 256."
        )

    units_per_direction = hidden_units // 2

    if units_per_direction < _MIN_HIDDEN_UNITS:
        raise ValueError(
            f"build_bilstm: hidden_units // 2 = {units_per_direction} units per "
            f"direction is below the practical minimum of {_MIN_HIDDEN_UNITS}. "
            f"Set hidden_units to at least {_MIN_HIDDEN_UNITS * 2} "
            f"(currently {hidden_units})."
        )

    if num_layers < 1:
        raise ValueError(
            f"build_bilstm: num_layers={num_layers} must be ≥ 1."
        )

    logger.info(
        f"Building BiLSTM | "
        f"input=({seq_len}, {feature_dim}) | "
        f"hidden_units={hidden_units} "
        f"({units_per_direction} per direction, output={hidden_units} via concat) | "
        f"num_layers={num_layers} | "
        f"dropout={dropout} | "
        f"recurrent_dropout={recurrent_dropout} | "
        f"n_classes={n_classes}",
        extra={"stage": "model"},
    )

    # ── Model construction ────────────────────────────────────────────────
    inputs = tf.keras.Input(
        shape=(seq_len, feature_dim),
        name="landmark_sequence",
    )

    # Masking — MANDATORY. Bidirectional wrapper correctly propagates this mask
    # to both forward and backward LSTM cells.
    x = tf.keras.layers.Masking(
        mask_value=0.0,
        name="zero_fill_mask",
    )(inputs)

    # Stacked Bidirectional LSTM layers
    for layer_idx in range(num_layers):
        is_final_recurrent = (layer_idx == num_layers - 1)
        layer_name = f"bilstm_layer_{layer_idx + 1}"

        # Inner LSTM: units_per_direction so that the Bidirectional concat
        # output = 2 × units_per_direction = hidden_units (comparable to
        # the unidirectional LSTM output width).
        inner_lstm = tf.keras.layers.LSTM(
            units=units_per_direction,
            return_sequences=not is_final_recurrent,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
        )

        x = tf.keras.layers.Bidirectional(
            inner_lstm,
            merge_mode="concat",  # forward ⊕ backward → (batch, 2×u_per_dir)
            name=layer_name,
        )(x)

    # Dense projection head — identical to LSTM/GRU for fair comparison.
    # Input width: hidden_units (= 2 × units_per_direction after concat).
    # hidden_units // 2 projects back to units_per_direction.
    x = tf.keras.layers.Dense(
        hidden_units // 2,
        activation="relu",
        name="dense_projection",
        kernel_initializer="glorot_uniform",
    )(x)
    x = tf.keras.layers.Dropout(dropout, name="dropout_head")(x)

    # Output
    outputs = tf.keras.layers.Dense(
        n_classes,
        activation="softmax",
        name="class_probabilities",
        kernel_initializer="glorot_uniform",
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="bilstm_classifier")

    _compile_model(model, cfg, "bilstm_classifier")
    _log_model_summary(model, "bilstm_classifier")

    return model


# ---------------------------------------------------------------------------
# Import-time self-check
# ---------------------------------------------------------------------------

def _self_check() -> None:
    """
    Verify module-level constants are internally consistent at import time.

    This check runs under Python's default ``__debug__`` mode (i.e. not when
    running with ``python -O``). It provides a fast, dependency-free sanity
    check on the module constant.

    What it checks
    --------------
    N_CLASSES == 35: verifies the module constant has not been accidentally
    edited. This is meaningful because it is compared to the hard-coded
    project value (35 signs, locked in artifacts/label_map_v1.json). It does
    NOT verify against the actual label map file — that is the responsibility
    of ``_check_n_classes(cfg)`` which runs at build time with the live config.

    The two-layer verification strategy
    ------------------------------------
    - Import time (here): cheap constant check — catches accidental edits to
      N_CLASSES in this file.
    - Build time (_check_n_classes): reads cfg.num_classes and cross-checks
      against the actual runtime config, which was itself loaded from
      label_map_v1.json by the data pipeline. This is the authoritative check.
    """
    assert N_CLASSES == 35, (
        f"architectures.py: N_CLASSES={N_CLASSES} was edited but must remain 35. "
        "The project is locked to 35 signs (artifacts/label_map_v1.json, schema v1.1). "
        "To use a different class count, update cfg.num_classes in base.yaml — "
        "the builders read cfg.num_classes at runtime, not this constant."
    )


if __debug__:
    _self_check()


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------

__all__ = [
    "N_CLASSES",
    "build_dense",
    "build_lstm",
    "build_gru",
    "build_bilstm",
]