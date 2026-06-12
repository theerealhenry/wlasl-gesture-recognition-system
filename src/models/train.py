"""
src/models/train.py
====================
Core training loop for the WLASL 35-class gesture recognition pipeline.

This module exposes a single public function — ``train_one_run()`` — that
trains one fully-specified model configuration, tracks every metric and
artefact to MLflow, and returns a summary dict used by the pipeline
orchestrators (``run_training.py``, ``run_all_experiments.py``) to select
the best configuration across groups.

Architecture overview
----------------------
``train_one_run()`` owns the entire lifecycle of one experiment run:

    1.  Seed the global RNG state (reproducibility).
    2.  Construct FeaturePipeline and GestureDataset.
    3.  Build or receive the model via the factory.
    4.  Construct callbacks (ReduceLROnPlateau, MacroF1Evaluator).
    5.  Execute the per-epoch training loop with per-epoch load_split().
    6.  Manual early stopping driven by val_macro_f1 (primary metric).
    7.  Restore and persist the best-macro-F1 checkpoint (weights-only).
    8.  Save all per-run artefacts (confusion matrices, training curves,
        per-class metrics JSON, run manifest, model hash).
    9.  Log everything to MLflow.

Critical design constraints (all non-negotiable)
-------------------------------------------------
Per-epoch ``load_split()``
    GestureDataset.load_split("train", training=True) MUST be called once
    per epoch — never once before the outer loop. Each call increments the
    internal epoch counter, which XORs with each clip's stable index to
    produce different augmentation per epoch. Calling model.fit(train_ds,
    epochs=N) with a single dataset object bypasses this mechanism and all
    N epochs receive identical augmentation.

macro-F1 as the primary metric
    val_macro_f1 (computed via sklearn) drives all checkpoint and early-
    stopping decisions. val_accuracy is a secondary metric only. With 21
    singleton validation classes, accuracy and macro-F1 can diverge
    significantly; the model checkpoint must be selected on macro-F1.

class_weight_balancing
    Applied to every run without exception when
    cfg.training.class_weight_balancing=True (the default). Without it,
    classes with 2–3 training clips (clothes, think) receive near-zero
    gradient contribution.

Weights-only checkpointing
    ``model.save_weights()`` / ``model.load_weights()`` is used instead
    of the full SavedModel format for intra-run checkpointing. This is
    approximately 5–10× faster per save, critical when the model improves
    frequently in the first 20 epochs. The final SavedModel export uses
    ``model.save()`` exactly once after the loop ends.

ReduceLROnPlateau state
    In TF 2.13.x, ReduceLROnPlateau maintains its internal ``wait`` counter
    and ``best`` value across sequential ``model.fit(epochs=1)`` calls
    when the same callback object is passed each time. This is the observed
    and tested behaviour on TF 2.13.1. The current LR is read via
    ``tf.keras.backend.get_value()`` after each epoch so LR reductions
    are detected and logged even if Keras does not print them (verbose=0).

MacroF1Evaluator (not a Keras callback)
    The reviewer correctly noted that manually invoking a Keras callback
    outside model.fit() is architecturally confusing. The class is renamed
    to ``MacroF1Evaluator`` and explicitly documented as a standalone
    evaluator object, not a Keras callback. It is never passed to
    ``model.fit(callbacks=...)``.

NaN/Inf training loss detection
    If training loss becomes NaN or Inf (gradient explosion), training is
    aborted immediately with a RuntimeError rather than silently consuming
    the remaining patience budget.

Fault-tolerant artefact generation
    All artefact-saving functions are wrapped in try/except. A matplotlib
    crash or disk-write failure after hours of successful training will
    LOG an error and continue — the training result is never lost due to
    an artefact failure.

High-risk classes
    After training, F1 scores for clothes, think, birthday, name, and book
    are extracted and logged. Zero-F1 for any of these triggers a WARNING
    and is recorded in the run manifest for Stage 6 analysis.

Artefacts produced per run
---------------------------
    artifacts/experiments/{run_name}/
        run_manifest.json          — identity, metrics, high-risk F1s
        config_snapshot.yaml       — exact config that produced this run
        metrics.json               — full epoch-by-epoch history
        per_class_metrics.json     — sklearn classification report
        confusion_matrix.png       — raw counts
        confusion_matrix_norm.png  — row-normalised rates
        training_curves.png        — loss, accuracy, and macro-F1 over epochs
        model_hash.txt             — MD5 of best model weights

    models/{run_name}_best_weights/   — Keras weights-only checkpoint dir
    models/{run_name}_saved_model/    — TF SavedModel (written once at end)
"""

from __future__ import annotations

import hashlib
import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # non-interactive backend; must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.features.dataset import GestureDataset
from src.features.pipeline import FeaturePipeline
from src.models.factory import build_model, get_model_summary_dict
from src.utils.logger import get_logger
from src.utils.reproducibility import setup_experiment

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Signs whose per-class F1 must be explicitly logged after every run.
#: These are the smallest training classes in the WLASL-35 dataset.
_HIGH_RISK_SIGNS: Tuple[str, ...] = (
    "clothes", "think", "birthday", "name", "book",
)

#: Fraction of training clips used for train-subset macro-F1 estimation.
#: Applies to the deterministic (non-augmented) arrays to give a stable
#: trend signal for the train/val gap analysis. 50% of 236 clips = 118 clips.
_TRAIN_F1_SAMPLE_FRACTION: float = 0.50

#: Compute train-subset macro-F1 every this many epochs.
#: At WLASL scale (~236 clips) one forward pass is fast; every 5 epochs
#: balances visibility vs compute cost.
_TRAIN_F1_EVAL_INTERVAL: int = 5

#: Minimum improvement in val_macro_f1 that resets the patience counter.
#: 0.001 = 0.1 percentage points — avoids saving on floating-point noise.
_F1_MIN_DELTA: float = 0.001

#: DPI for all saved figures.
_FIGURE_DPI: int = 150

#: Confusion matrix figure size in inches for 35 classes.
_CM_FIGSIZE: Tuple[int, int] = (20, 18)

#: Maximum training loss value before triggering NaN/explosion guard.
#: Losses above this threshold in the FIRST epoch indicate a mis-configured
#: learning rate or data pipeline issue and abort training immediately.
_LOSS_EXPLOSION_THRESHOLD: float = 1e6

#: Weights checkpoint subdirectory suffix (inside models/).
_WEIGHTS_CHECKPOINT_SUFFIX: str = "_best_weights"

#: SavedModel subdirectory suffix (inside models/).
_SAVEDMODEL_SUFFIX: str = "_saved_model"


# ---------------------------------------------------------------------------
# MacroF1Evaluator (standalone evaluator, NOT a Keras callback)
# ---------------------------------------------------------------------------

class MacroF1Evaluator:
    """
    Standalone evaluator that computes val macro-F1 via sklearn each epoch.

    This class is intentionally NOT a subclass of tf.keras.callbacks.Callback.
    The original implementation was called "MacroF1Callback" but was invoked
    manually outside model.fit() — a pattern that is architecturally confusing
    (something named "callback" but not used as one). Renaming to
    ``MacroF1Evaluator`` makes the intent explicit.

    Design contract
    ---------------
    - COMPUTATION only: loops through ``val_ds``, calls sklearn f1_score,
      appends result to ``val_macro_f1_history``.
    - DOES NOT log to MLflow or call any Keras internal API.
    - CALLED EXPLICITLY by the epoch loop in ``train_one_run()`` after
      each ``model.fit(epochs=1)`` call.
    - The ``labels=list(range(n_classes))`` argument to ``f1_score`` ensures
      all 35 classes appear in the denominator. Without this, sklearn silently
      excludes classes absent from the validation predictions, inflating macro-F1
      for runs where rare classes predict zero correct samples.

    Val scoring strategy
    --------------------
    This evaluator runs one forward pass over the full val set per epoch.
    ``model.fit(validation_data=val_ds)`` already runs a val pass for
    val_loss and val_accuracy (Keras metrics). This results in two val passes
    per epoch — one for Keras metrics, one for sklearn macro-F1.

    At WLASL scale (52 val clips, CPU training) the overhead is negligible.
    For larger datasets, consider computing macro-F1 from the val_accuracy
    pass by registering a custom Keras metric — but that implementation is
    numerically different from sklearn and would corrupt evaluation comparisons.

    Parameters
    ----------
    val_ds : tf.data.Dataset
        The validation dataset. Loaded once before the epoch loop and reused.
    n_classes : int
        Number of output classes (35). Passed as ``labels`` to f1_score.
    model : tf.keras.Model
        Set via ``set_model()`` before the first ``evaluate()`` call.
    """

    def __init__(self, val_ds: Any, n_classes: int) -> None:
        self.val_ds               = val_ds
        self.n_classes            = n_classes
        self.model: Optional[Any] = None
        self.val_macro_f1_history: List[float] = []
        self.best_val_macro_f1    = 0.0

    def set_model(self, model: Any) -> None:
        """Bind the model. Called once before the epoch loop begins."""
        self.model = model

    def evaluate(self, epoch: int) -> float:
        """
        Run inference on the full val set and return macro-F1.

        Uses ``self.model(x_batch, training=False)`` (the call API) rather
        than ``self.model.predict()`` to avoid triggering Keras's internal
        predict machinery inside an epoch loop, which can produce unexpected
        interaction with batch-normalisation layers or progress-bar callbacks.

        Parameters
        ----------
        epoch : int  (0-indexed, for log messages only)

        Returns
        -------
        float  macro-F1 in [0.0, 1.0].

        Raises
        ------
        RuntimeError  If set_model() has not been called.
        """
        if self.model is None:
            raise RuntimeError(
                "MacroF1Evaluator.evaluate() called before set_model(). "
                "Call evaluator.set_model(model) before the epoch loop."
            )

        import tensorflow as tf

        y_true: List[int] = []
        y_pred: List[int] = []

        for x_batch, y_batch in self.val_ds:
            logits = self.model(x_batch, training=False)
            preds  = tf.argmax(logits, axis=1).numpy().tolist()
            y_pred.extend(preds)
            y_true.extend(y_batch.numpy().tolist())

        f1 = float(f1_score(
            y_true, y_pred,
            average="macro",
            labels=list(range(self.n_classes)),
            zero_division=0,
        ))

        self.val_macro_f1_history.append(f1)
        if f1 > self.best_val_macro_f1:
            self.best_val_macro_f1 = f1

        logger.debug(
            f"MacroF1Evaluator | epoch={epoch + 1} | "
            f"val_macro_f1={f1:.4f} | best_so_far={self.best_val_macro_f1:.4f}",
            extra={"stage": "training"},
        )
        return f1


# ---------------------------------------------------------------------------
# Weights-only checkpoint helpers
# ---------------------------------------------------------------------------

def _save_weights(model: Any, checkpoint_dir: Path) -> None:
    """
    Save model weights to a Keras weights checkpoint directory.

    Uses ``model.save_weights()`` which is 5–10× faster than a full
    SavedModel export. Safe to call on every improvement without performance
    penalty.

    The checkpoint is saved as ``checkpoint_dir/weights`` using TF's
    built-in checkpoint format (multiple files: .index, .data-00000-of-00001).

    Parameters
    ----------
    model          : tf.keras.Model
    checkpoint_dir : Path  (created if it does not exist)
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    weight_path = str(checkpoint_dir / "weights")
    model.save_weights(weight_path)
    logger.debug(
        f"Weights checkpoint saved → {checkpoint_dir}",
        extra={"stage": "training"},
    )


def _load_weights(model: Any, checkpoint_dir: Path) -> bool:
    """
    Restore model weights from a Keras weights checkpoint directory.

    Parameters
    ----------
    model          : tf.keras.Model  (must have identical architecture)
    checkpoint_dir : Path

    Returns
    -------
    bool  True if weights were successfully restored, False otherwise.
    """
    weight_path = str(checkpoint_dir / "weights")
    try:
        model.load_weights(weight_path)
        logger.info(
            f"Weights restored from {checkpoint_dir}",
            extra={"stage": "training"},
        )
        return True
    except Exception as exc:
        logger.warning(
            f"Could not restore weights from {checkpoint_dir}: "
            f"{type(exc).__name__}: {exc}",
            extra={"stage": "training"},
        )
        return False


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _get_lr(optimizer: Any) -> float:
    """
    Read the current learning rate from a Keras optimizer safely.

    In TF 2.13, ``optimizer.learning_rate`` is a ``tf.Variable`` after
    compilation. ``tf.keras.backend.get_value()`` handles both the
    tf.Variable case (post-compile) and a plain Python float (pre-compile
    or mocked), making this safe across all execution contexts.
    """
    import tensorflow as tf
    return float(tf.keras.backend.get_value(optimizer.learning_rate))


def _is_loss_nan_or_exploded(loss: float) -> bool:
    """
    Return True if the loss value indicates training has collapsed.

    Detects three failure modes:
      - NaN (gradient explosion that wrapped to undefined)
      - Inf (gradient explosion that diverged to infinity)
      - Extremely large finite value (pre-explosion instability)
    """
    import math
    return math.isnan(loss) or math.isinf(loss) or loss > _LOSS_EXPLOSION_THRESHOLD


def _get_config_attr(cfg_section: Any, attr: str, default: Any) -> Any:
    """
    Safely retrieve an attribute from an OmegaConf-backed config section.

    OmegaConf raises ``omegaconf.errors.ConfigAttributeError`` (not
    ``AttributeError``) for missing keys. ``getattr()`` catches standard
    AttributeError but not the OmegaConf variant. This helper catches both,
    making it safe for optional fields absent from some model YAMLs (e.g.
    ``num_layers`` and ``recurrent_dropout`` are absent from ``dense.yaml``).

    Parameters
    ----------
    cfg_section : OmegaConf DictConfig or Pydantic model section
    attr        : attribute name to retrieve
    default     : value to return if attr is absent or raises

    Returns
    -------
    Any  The attribute value, or ``default`` if not found.
    """
    try:
        value = getattr(cfg_section, attr, None)
        # OmegaConf may return MISSING sentinel; treat as absent
        if value is None:
            return default
        # Convert OmegaConf MISSING to default (handles structured config edge cases)
        try:
            from omegaconf import MISSING as _MISSING
            if value is _MISSING:
                return default
        except ImportError:
            pass
        return value
    except Exception:
        return default


def _compute_train_f1_subset(
    model:   Any,
    dataset: GestureDataset,
    cfg:     Any,
    epoch:   int,
) -> Optional[float]:
    """
    Compute train macro-F1 on a deterministic 50% sample of training clips.

    Uses the DETERMINISTIC (non-augmented) training arrays from
    ``GestureDataset.get_arrays_for_split("train", use_augmentation=False)``.
    This is intentional: the augmented dataset changes every epoch, so using
    it would produce an epoch-correlated noisy signal. The deterministic
    arrays give a stable read on how well the model fits the raw training
    distribution — useful for diagnosing overfitting.

    The sample seed is ``cfg.seed XOR epoch`` so different clips are sampled
    each evaluation, but results are fully reproducible for any fixed
    (cfg, epoch) pair.

    Returns
    -------
    float or None  macro-F1 on the sample. None if an exception is raised
                   (non-fatal: the epoch loop continues without this metric).
    """
    try:
        import tensorflow as tf

        X, y, _ = dataset.get_arrays_for_split("train", use_augmentation=False)
        n_total  = len(y)
        n_sample = max(1, int(n_total * _TRAIN_F1_SAMPLE_FRACTION))

        rng     = np.random.default_rng(int(cfg.seed) ^ int(epoch))
        indices = rng.choice(n_total, size=n_sample, replace=False)

        X_sample = X[indices]
        y_sample = y[indices]

        logits   = model(tf.constant(X_sample, dtype=tf.float32), training=False)
        y_pred   = tf.argmax(logits, axis=1).numpy()

        return float(f1_score(
            y_sample, y_pred,
            average="macro",
            labels=list(range(int(cfg.num_classes))),
            zero_division=0,
        ))
    except Exception as exc:
        logger.warning(
            f"train_macro_f1 subset computation failed at epoch {epoch + 1}: "
            f"{type(exc).__name__}: {exc}",
            extra={"stage": "training"},
        )
        return None


def _get_val_predictions(
    model:  Any,
    val_ds: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the best model over the full validation set and return (y_true, y_pred).

    Used once after the training loop for final artefact generation. This is
    a third val pass (after Keras val pass and MacroF1Evaluator pass per
    epoch) but is called only once at the end, so total cost is marginal.

    Returns
    -------
    y_true : np.ndarray  shape (n_val,)  int32
    y_pred : np.ndarray  shape (n_val,)  int32
    """
    import tensorflow as tf

    y_true_list: List[int] = []
    y_pred_list: List[int] = []

    for x_batch, y_batch in val_ds:
        logits = model(x_batch, training=False)
        preds  = tf.argmax(logits, axis=1).numpy().tolist()
        y_pred_list.extend(preds)
        y_true_list.extend(y_batch.numpy().tolist())

    return (
        np.array(y_true_list, dtype=np.int32),
        np.array(y_pred_list, dtype=np.int32),
    )


def _build_sign_name_list(dataset: GestureDataset, n_classes: int) -> List[str]:
    """
    Build a list of sign names indexed by class_idx (length = n_classes).

    Relies on ``dataset.label_map.get_name_safe(idx, fallback)`` which is
    confirmed present on LabelMap in the project codebase.
    """
    return [
        dataset.label_map.get_name_safe(i, f"class_{i}")
        for i in range(n_classes)
    ]


def _build_high_risk_f1_dict(
    report: Dict[str, Any],
) -> Dict[str, float]:
    """
    Extract F1 scores for the five highest-risk classes from a sklearn report.

    Uses only the ``report`` dict — the unused ``sign_names`` parameter from
    the original implementation has been removed.

    Parameters
    ----------
    report : dict from sklearn.metrics.classification_report(output_dict=True)

    Returns
    -------
    dict mapping sign_name -> f1_score (0.0 if the class is absent from
    the validation predictions, which indicates complete failure to learn it).
    """
    result: Dict[str, float] = {}
    for sign in _HIGH_RISK_SIGNS:
        if sign in report:
            result[sign] = float(report[sign]["f1-score"])
        else:
            result[sign] = 0.0

        if result[sign] == 0.0:
            logger.warning(
                f"High-risk class '{sign}' has F1 = 0.0 after training. "
                "This class likely failed to learn meaningful patterns. "
                "Check training clip count and class weight. "
                "See LIMITATIONS.md §6 for analysis.",
                extra={"stage": "training"},
            )

    return result


# ---------------------------------------------------------------------------
# Fault-tolerant artefact generation
# ---------------------------------------------------------------------------

@contextmanager
def _artefact_guard(description: str) -> Generator[None, None, None]:
    """
    Context manager that catches all exceptions in artefact-generation blocks.

    A crash in matplotlib, a full disk, or a permissions error should NEVER
    abort a training run after the model has been trained. This guard logs
    the error at ERROR level and continues. The training result dict is always
    returned regardless of artefact failures.

    Usage::

        with _artefact_guard("confusion matrix"):
            _save_confusion_matrices(...)
    """
    try:
        yield
    except Exception as exc:
        logger.error(
            f"Artefact generation failed ({description}): "
            f"{type(exc).__name__}: {exc}. "
            "Training result is unaffected — the artefact will be missing.",
            extra={"stage": "training"},
        )


def _save_confusion_matrices(
    y_true:       np.ndarray,
    y_pred:       np.ndarray,
    sign_names:   List[str],
    run_name:     str,
    artifact_dir: Path,
) -> None:
    """
    Save raw-count and row-normalised confusion matrices as PNG files.

    Two versions:
      confusion_matrix.png      — raw integer counts
      confusion_matrix_norm.png — row-normalised rates (0.0–1.0)

    Raw counts matter for rare classes: one miss on clothes (2 val clips) is
    a 50% error rate. Row-normalisation reveals systematic confusable pairs
    independent of class frequency.

    Division-by-zero guard: ``np.maximum(row_sums, 1)`` prevents NaN in the
    normalised matrix for classes with zero true samples (possible when a
    class has all its val clips missing from the loaded dataset).
    """
    n  = len(sign_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))

    # Row-normalised — guard against zero-true-sample rows
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = cm.astype(float) / np.maximum(row_sums, 1)

    for suffix, data, fmt in [
        ("",      cm,      "d"),
        ("_norm", cm_norm, ".2f"),
    ]:
        fig, ax = plt.subplots(figsize=_CM_FIGSIZE)
        disp = ConfusionMatrixDisplay(data, display_labels=sign_names)
        disp.plot(ax=ax, xticks_rotation=90, colorbar=True, values_format=fmt)
        title_suffix = " (row-normalised)" if suffix else ""
        ax.set_title(f"Confusion Matrix{title_suffix} — {run_name}", fontsize=11)
        ax.tick_params(axis="both", labelsize=7)
        plt.tight_layout()

        out_path = artifact_dir / f"confusion_matrix{suffix}.png"
        fig.savefig(out_path, dpi=_FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        logger.info(
            f"Saved confusion matrix{title_suffix} → {out_path}",
            extra={"stage": "training"},
        )


def _save_training_curves(
    history_log:  List[Dict[str, Any]],
    best_epoch:   int,
    run_name:     str,
    artifact_dir: Path,
) -> None:
    """
    Save a two-panel training curves figure (loss left, metrics right).

    Both panels include a gold dashed vertical line at the best macro-F1 epoch.
    The right panel overlays train_acc, val_acc, and val_macro_f1 on the same
    axes to make the overfitting gap and the relationship between accuracy and
    macro-F1 immediately visible.
    """
    epochs_logged = list(range(len(history_log)))

    train_loss = [h["train_loss"]  for h in history_log]
    val_loss   = [h["val_loss"]    for h in history_log]
    train_acc  = [h["train_acc"]   for h in history_log]
    val_acc    = [h["val_acc"]     for h in history_log]
    val_f1     = [h["val_macro_f1"] for h in history_log]

    # Train macro-F1 is only logged every N epochs — fill gaps with None
    train_f1_epochs = [
        (i, h["train_macro_f1"])
        for i, h in enumerate(history_log)
        if "train_macro_f1" in h
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left panel: Loss ────────────────────────────────────────────────
    axes[0].plot(epochs_logged, train_loss, label="train_loss", color="royalblue",  linewidth=1.5)
    axes[0].plot(epochs_logged, val_loss,   label="val_loss",   color="tomato",     linewidth=1.5)
    axes[0].axvline(
        best_epoch, color="gold", linestyle="--", linewidth=1.5,
        label=f"best epoch ({best_epoch + 1})",
    )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # ── Right panel: Accuracy + macro-F1 ────────────────────────────────
    axes[1].plot(epochs_logged, train_acc, label="train_acc",    color="royalblue",     linewidth=1.5)
    axes[1].plot(epochs_logged, val_acc,   label="val_acc",      color="tomato",        linewidth=1.5)
    axes[1].plot(epochs_logged, val_f1,    label="val_macro_f1", color="mediumseagreen",linewidth=1.5, linestyle=":")

    if train_f1_epochs:
        tf_x, tf_y = zip(*train_f1_epochs)
        axes[1].plot(tf_x, tf_y, label="train_macro_f1 (sampled)",
                     color="steelblue", linestyle=":", linewidth=1.0,
                     marker="o", markersize=4)

    axes[1].axvline(
        best_epoch, color="gold", linestyle="--", linewidth=1.5,
        label=f"best epoch ({best_epoch + 1})",
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Accuracy / Macro-F1")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0.0, 1.05)

    fig.suptitle(f"Training curves — {run_name}", fontsize=12)
    plt.tight_layout()

    out_path = artifact_dir / "training_curves.png"
    fig.savefig(out_path, dpi=_FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(
        f"Saved training curves → {out_path}",
        extra={"stage": "training"},
    )


def _save_per_class_metrics(
    y_true:       np.ndarray,
    y_pred:       np.ndarray,
    sign_names:   List[str],
    n_classes:    int,
    artifact_dir: Path,
) -> Dict[str, Any]:
    """
    Compute and save the sklearn classification report as JSON.

    Uses explicit ``labels=list(range(n_classes))`` and
    ``target_names=sign_names`` to guarantee all 35 classes appear in the
    report even if some classes have zero true samples or zero predicted
    samples in the validation set. Without explicit ``labels``, sklearn
    silently drops classes absent from the prediction, which would make
    the report appear more complete than it is.

    Returns the report dict so callers can extract high-risk class F1s
    without re-running inference.
    """
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(n_classes)),
        target_names=sign_names,
        output_dict=True,
        zero_division=0,
    )
    out_path = artifact_dir / "per_class_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(
        f"Saved per-class metrics → {out_path}",
        extra={"stage": "training"},
    )
    return report


def _save_run_manifest(
    cfg:               Any,
    run_name:          str,
    experiment_group:  str,
    best_epoch:        int,
    best_val_macro_f1: float,
    best_val_acc:      float,
    total_epochs:      int,
    model_param_count: int,
    high_risk_f1:      Dict[str, float],
    artifact_dir:      Path,
    mlflow_run_id:     str,
    weights_restored:  bool,
) -> None:
    """
    Write run_manifest.json — the authoritative index entry for this run.

    Includes ``weights_restored`` to flag runs where the best-checkpoint
    could not be loaded (fallback to final state). Stage 6 analysis should
    treat such runs with caution.
    """
    manifest = {
        "run_name":             run_name,
        "experiment_group":     experiment_group,
        "mlflow_run_id":        mlflow_run_id,
        "best_epoch":           best_epoch,
        "best_val_macro_f1":    best_val_macro_f1,
        "best_val_acc":         best_val_acc,
        "total_epochs_trained": total_epochs,
        "model_param_count":    model_param_count,
        "config_hash":          cfg.config_hash,
        "timestamp_utc":        datetime.now(timezone.utc).isoformat(),
        "high_risk_class_f1":   high_risk_f1,
        "best_weights_restored": weights_restored,
    }
    out_path = artifact_dir / "run_manifest.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(
        f"Saved run manifest → {out_path}",
        extra={"stage": "training"},
    )


def _save_config_snapshot(cfg: Any, artifact_dir: Path) -> None:
    """
    Save a YAML (preferred) or JSON representation of the full config.
    """
    out_path = artifact_dir / "config_snapshot.yaml"
    try:
        import json as _json
        from omegaconf import OmegaConf

        # model_dump with mode="json" serialises Enums to their .value strings,
        # making the dict safe for OmegaConf.create() which cannot handle Enum objects.
        cfg_dict = cfg.model_dump(mode="json")
        yaml_str = OmegaConf.to_yaml(OmegaConf.create(cfg_dict))
        out_path.write_text(yaml_str, encoding="utf-8")
    except Exception as exc:
        logger.debug(
            f"OmegaConf YAML serialisation failed ({exc}); falling back to JSON.",
            extra={"stage": "training"},
        )
        out_path_json = artifact_dir / "config_snapshot.json"
        with open(out_path_json, "w", encoding="utf-8") as f:
            _json.dump(cfg.model_dump(mode="json"), f, indent=2)


def _save_metrics_history(
    history_log:  List[Dict[str, Any]],
    artifact_dir: Path,
) -> None:
    """Write the full epoch-by-epoch metrics log as metrics.json."""
    out_path = artifact_dir / "metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(history_log, f, indent=2)


def _compute_model_hash(model: Any) -> str:
    """
    Compute an MD5 hex digest over the model's weight tensors.

    Uses an incremental hash (``md5.update()`` per tensor) to avoid
    materialising all weights as a single large ``bytes`` object in RAM —
    safer for large future models even though WLASL models are small.

    This is not a cryptographic hash — it is a cheap integrity fingerprint
    that lets reviewers verify that a reported ``.tflite`` file corresponds
    to a specific MLflow run without reloading the full model.

    Returns
    -------
    str  32-character hex MD5 digest, or "unavailable" on failure.
    """
    try:
        md5 = hashlib.md5()
        for weight in model.weights:
            md5.update(weight.numpy().tobytes())
        return md5.hexdigest()
    except Exception as exc:
        logger.warning(
            f"Could not compute model hash: {type(exc).__name__}: {exc}",
            extra={"stage": "training"},
        )
        return "unavailable"


def _save_model_hash(model_hash: str, artifact_dir: Path) -> None:
    """Write model_hash.txt."""
    (artifact_dir / "model_hash.txt").write_text(model_hash, encoding="utf-8")


# ---------------------------------------------------------------------------
# MLflow parameter logging helpers
# ---------------------------------------------------------------------------

def _log_mlflow_params(cfg: Any, pipeline: Any, dataset: Any, model_summary: Dict) -> None:
    """
    Log all run parameters to the active MLflow run.

    Uses ``_get_config_attr()`` for every field that is absent from some
    model YAMLs (``num_layers``, ``recurrent_dropout`` are absent from
    ``dense.yaml``). Direct attribute access would raise
    ``ConfigAttributeError`` for Dense runs.

    All parameter values must be JSON-serialisable scalars (str, int, float,
    bool). MLflow silently coerces non-serialisable values to strings, which
    makes comparison across runs unreliable.

    Parameters
    ----------
    cfg          : ExperimentConfig
    pipeline     : FeaturePipeline   (resolved feature_dim, not cfg.data.feature_dim)
    dataset      : GestureDataset    (n_train, n_val for sanity checking in UI)
    model_summary: dict from get_model_summary_dict()
    """
    import mlflow

    mlflow.log_params({
        # ── Model architecture ─────────────────────────────────────────────
        "model_type":          cfg.model.name.value if hasattr(cfg.model.name, "value") else str(cfg.model.name),
        "hidden_units":        int(cfg.model.hidden_units),
        # num_layers and recurrent_dropout absent from dense.yaml → use safe getter
        "num_layers":          int(_get_config_attr(cfg.model, "num_layers",        1)),
        "dropout":             float(cfg.model.dropout),
        "recurrent_dropout":   float(_get_config_attr(cfg.model, "recurrent_dropout", 0.0)),
        "bidirectional":       bool(_get_config_attr(cfg.model, "bidirectional",    False)),
        # ── Data ───────────────────────────────────────────────────────────
        "seq_len":             int(cfg.data.sequence_length),
        "landmark_config":     str(cfg.data.landmark_config),
        # feature_dim is derived from landmark_config inside FeaturePipeline;
        # cfg.data.feature_dim does NOT exist as a DataConfig field.
        "feature_dim":         int(pipeline.feature_dim),
        "n_train":             int(dataset.n_train),
        "n_val":               int(dataset.n_val),
        "n_classes":           int(cfg.num_classes),
        # ── Training ──────────────────────────────────────────────────────
        "batch_size":          int(cfg.training.batch_size),
        "learning_rate":       float(cfg.training.learning_rate),
        "epochs_max":          int(cfg.training.epochs),
        "early_stopping_patience": int(cfg.training.early_stopping_patience),
        "class_weight_balancing":  bool(cfg.training.class_weight_balancing),
        "seed":                int(cfg.seed),
        # ── Augmentation ──────────────────────────────────────────────────
        "augmentation_enabled":    bool(cfg.augmentation.enabled),
        "temporal_jitter":         bool(cfg.augmentation.temporal_jitter),
        "speed_jitter":            bool(cfg.augmentation.speed_jitter),
        "spatial_flip":            bool(cfg.augmentation.spatial_flip),
        "gaussian_noise_std":      float(cfg.augmentation.gaussian_noise_std),
        "rotation_deg":            float(cfg.augmentation.rotation_deg),
        # ── Model statistics (post-build) ──────────────────────────────────
        "total_params":        int(model_summary["param_count"]),
        "model_size_mb":       float(model_summary["model_size_mb_estimate"]),
        # ── Meta ──────────────────────────────────────────────────────────
        "config_hash":         str(cfg.config_hash),
    })


def _set_mlflow_tags(cfg: Any, experiment_group: str) -> None:
    """
    Set MLflow tags for filtering and grouping in the MLflow UI.

    Tags use string values only (MLflow tag API requirement).
    """
    import mlflow

    mlflow.set_tags({
        "experiment_group": str(experiment_group),
        "model_type":       str(cfg.model.name),
        "landmark_config":  str(cfg.data.landmark_config),
        "seq_len":          str(cfg.data.sequence_length),
        "augmentation":     "enabled" if cfg.augmentation.enabled else "disabled",
        "signer_split":     "true",
        "num_classes":      "35",
        "primary_metric":   "val_macro_f1",
    })


# ---------------------------------------------------------------------------
# Primary public function
# ---------------------------------------------------------------------------

def train_one_run(
    cfg:              Any,
    run_name:         str,
    experiment_group: str,
    model:            Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Train one model configuration end-to-end and return a results summary.

    This is the single function called by every pipeline entry point. It owns
    the full lifecycle: data loading, model construction, the per-epoch
    training loop, artefact generation, and MLflow logging.

    The function must be called inside an active ``mlflow.start_run()``
    context. The caller (``run_training.py``) is responsible for opening
    and closing the MLflow run; this function only calls ``mlflow.log_*``.

    Parameters
    ----------
    cfg : ExperimentConfig
        Full frozen experiment config from ``load_config()``.
    run_name : str
        Unique name for this run (e.g. "lstm_seq60_spatial_temporal").
        Used for MLflow run name, artefact directory, and model save path.
    experiment_group : str
        Group label for MLflow tag filtering (e.g. "architecture",
        "augmentation", "sequence_length", "landmark_config", "champion").
    model : tf.keras.Model | None, optional
        Pre-built compiled model. If None (default), the factory builds one
        from ``cfg``. Providing a pre-built model is intended for unit tests.

    Returns
    -------
    dict with keys:
        run_name                (str)
        experiment_group        (str)
        best_val_macro_f1       (float)
        best_val_acc            (float)
        best_epoch              (int)    0-indexed
        total_epochs_trained    (int)
        mlflow_run_id           (str)
        config_hash             (str)
        model_param_count       (int)
        high_risk_class_f1      (dict)
        artifact_dir            (str)    absolute path
        model_save_path         (str)    absolute path to SavedModel
        best_weights_restored   (bool)   False means fallback to final weights
        model_type              (str)    for run_all_experiments.py selection
        seq_len                 (int)
        landmark_config         (str)
        augmentation            (bool)

    Raises
    ------
    ValueError      If run_name is empty.
    RuntimeError    If the training set is empty, model construction fails,
                    or training loss diverges to NaN/Inf in the first epoch.
    """
    import mlflow
    import tensorflow as tf

    if not run_name or not run_name.strip():
        raise ValueError("run_name must be a non-empty string.")

    run_name    = run_name.strip()
    t_run_start = time.time()

    # ── 0. Suppress TF autolog (we log all metrics explicitly) ──────────
    try:
        mlflow.tensorflow.autolog(disable=True)
    except Exception:
        pass

    # ── 1. Seed global RNG and create artefact directory ────────────────
    artifact_dir = Path(f"artifacts/experiments/{run_name}")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    setup_experiment(
        config=cfg,
        run_name=run_name,
        output_dir=str(artifact_dir),
    )

    logger.info(
        f"train_one_run START | "
        f"run_name='{run_name}' | "
        f"group='{experiment_group}' | "
        f"config_hash={cfg.config_hash[:12]}",
        extra={"stage": "training"},
    )

    # ── 2. Feature pipeline and dataset ─────────────────────────────────
    pipeline = FeaturePipeline(cfg)

    dataset = GestureDataset(
        cfg,
        pipeline,
        splits_dir="data/splits",
        landmarks_dir="data/landmarks",
    )

    n_classes  = int(cfg.num_classes)
    sign_names = _build_sign_name_list(dataset, n_classes)

    # Class weights — conditionally applied
    if cfg.training.class_weight_balancing:
        class_weights: Optional[Dict[int, float]] = dataset.compute_class_weights()
        logger.info(
            f"Class weighting ENABLED | "
            f"n_weighted={len(class_weights)} | "
            f"min={min(class_weights.values()):.4f} | "
            f"max={max(class_weights.values()):.4f} | "
            f"ratio={max(class_weights.values()) / max(min(class_weights.values()), 1e-9):.2f}x",
            extra={"stage": "training"},
        )
    else:
        class_weights = None
        logger.info(
            "Class weighting DISABLED (cfg.training.class_weight_balancing=False).",
            extra={"stage": "training"},
        )

    # Validation dataset: loaded once, reused every epoch. Never augmented.
    # Loaded BEFORE the epoch counter is used so that the counter increment
    # from this load (counter goes 0 → 1) is accounted for before training.
    val_ds = dataset.load_split("val", training=False)

    # ── 3. Build model ───────────────────────────────────────────────────
    if model is None:
        model = build_model(cfg, pipeline=pipeline)
    else:
        logger.info(
            f"Using pre-supplied model '{model.name}' "
            f"({model.count_params():,} params).",
            extra={"stage": "training"},
        )

    model_summary = get_model_summary_dict(model)

    # ── 4. Callbacks ─────────────────────────────────────────────────────
    #
    # ReduceLROnPlateau on val_accuracy.
    # In TF 2.13.x, this callback DOES retain its internal wait counter and
    # best-value across sequential model.fit(epochs=1) calls when the same
    # object is passed each time — verified behaviour. verbose=0 silences
    # stdout; we detect and log LR reductions manually via _get_lr().
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_accuracy",
        factor=float(cfg.training.reduce_lr_factor),
        patience=int(cfg.training.reduce_lr_patience),
        min_lr=float(cfg.training.reduce_lr_min_lr),
        min_delta=0.001,
        verbose=0,
    )

    # MacroF1Evaluator — standalone evaluator, NOT in keras_callbacks.
    # Invoked manually after each model.fit() call.
    macro_f1_evaluator = MacroF1Evaluator(val_ds=val_ds, n_classes=n_classes)
    macro_f1_evaluator.set_model(model)

    # Keras callbacks list. EarlyStopping on Keras val_accuracy is deliberately
    # EXCLUDED: it cannot function correctly across sequential model.fit(epochs=1)
    # calls because its internal state (wait counter) resets each call in some
    # TF versions. Manual early stopping on val_macro_f1 (below) is the sole
    # stopping mechanism. ReduceLROnPlateau state has been empirically verified
    # to persist in TF 2.13.x.
    keras_callbacks = [reduce_lr]

    # ── 5. Capture MLflow run ID — params and tags already logged by caller ──
    #
    # run_training.py logs all params via _log_mlflow_params() and all tags
    # via _set_mlflow_tags() BEFORE calling train_one_run(). Repeating those
    # calls here would cause MlflowException: "Changing param values is not
    # allowed" when the Enum serialisation differs between the two callers.
    # train_one_run() is responsible for per-epoch metrics and artefacts only.
    mlflow_run_id = ""
    if mlflow.active_run():
        mlflow_run_id = mlflow.active_run().info.run_id
    else:
        logger.warning(
            "train_one_run(): no active MLflow run found. "
            "Metrics and artefacts will not be logged. "
            "Call train_one_run() inside an mlflow.start_run() context.",
            extra={"stage": "training"},
        )

    # ── 6. Per-epoch training loop ────────────────────────────────────────
    #
    # MANDATORY CONTRACT: dataset.load_split("train", training=True) is
    # called ONCE PER EPOCH. Each call increments GestureDataset's internal
    # epoch counter, which XORs with each clip's stable index to seed per-
    # clip augmentation differently every epoch. Using model.fit(train_ds,
    # epochs=N) with N>1 bypasses this mechanism — augmentation is identical
    # for all N epochs.
    # ──────────────────────────────────────────────────────────────────────

    # Initialise best_val_macro_f1 to -1.0 so that the very first epoch
    # always saves a checkpoint regardless of its F1 value. Using 0.0 can
    # prevent the first epoch from saving if F1 < _F1_MIN_DELTA = 0.001.
    best_val_macro_f1: float = -1.0
    best_val_acc:      float  = 0.0
    best_epoch:        int    = 0
    patience_counter:  int    = 0
    prev_lr:           float  = _get_lr(model.optimizer)
    history_log:       List[Dict[str, Any]] = []

    models_dir             = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    weights_checkpoint_dir = models_dir / f"{run_name}{_WEIGHTS_CHECKPOINT_SUFFIX}"
    saved_model_dir        = models_dir / f"{run_name}{_SAVEDMODEL_SUFFIX}"

    max_epochs    = int(cfg.training.epochs)
    patience_cap  = int(cfg.training.early_stopping_patience)
    stopped_early = False

    for epoch in range(max_epochs):
        epoch_start = time.time()

        # ── Load FRESH augmented training dataset for this epoch ──────────
        # This is the call that increments GestureDataset._epoch_counter.
        train_ds = dataset.load_split("train", training=True)

        # ── One epoch of training ─────────────────────────────────────────
        hist = model.fit(
            train_ds,
            epochs=1,
            validation_data=val_ds,
            class_weight=class_weights,
            callbacks=keras_callbacks,
            verbose=0,
        )

        # ── NaN / explosion guard ─────────────────────────────────────────
        train_loss = float(hist.history["loss"][0])
        if _is_loss_nan_or_exploded(train_loss):
            msg = (
                f"Training aborted: loss={train_loss} at epoch {epoch + 1}. "
                "This indicates gradient explosion or a data pipeline issue. "
                "Check learning_rate (try 10× smaller) and input data statistics."
            )
            logger.error(msg, extra={"stage": "training"})
            raise RuntimeError(msg)

        # ── Extract Keras metrics ─────────────────────────────────────────
        val_loss  = float(hist.history["val_loss"][0])
        train_acc = float(hist.history["accuracy"][0])
        val_acc   = float(hist.history["val_accuracy"][0])

        # ── Compute val macro-F1 via sklearn ──────────────────────────────
        # Called AFTER model.fit() completes for this epoch. This is a second
        # val pass per epoch (Keras already scored val_accuracy above). At
        # WLASL scale (52 val clips) the overhead is negligible.
        val_macro_f1 = macro_f1_evaluator.evaluate(epoch)

        # ── LR change detection ───────────────────────────────────────────
        current_lr = _get_lr(model.optimizer)
        if current_lr < prev_lr - 1e-10:
            logger.info(
                f"ReduceLROnPlateau | "
                f"lr: {prev_lr:.3e} → {current_lr:.3e} at epoch {epoch + 1}",
                extra={"stage": "training"},
            )
        prev_lr = current_lr

        # ── Train macro-F1 on a deterministic subset (every N epochs) ─────
        train_macro_f1: Optional[float] = None
        if epoch % _TRAIN_F1_EVAL_INTERVAL == 0:
            train_macro_f1 = _compute_train_f1_subset(model, dataset, cfg, epoch)

        # ── Build epoch metrics dict ──────────────────────────────────────
        epoch_time = time.time() - epoch_start
        epoch_metrics: Dict[str, Any] = {
            "train_loss":    round(train_loss, 6),
            "val_loss":      round(val_loss,   6),
            "train_acc":     round(train_acc,  6),
            "val_acc":       round(val_acc,    6),
            "val_macro_f1":  round(val_macro_f1, 6),
            "learning_rate": current_lr,
            "epoch_time_s":  round(epoch_time, 2),
        }
        if train_macro_f1 is not None:
            epoch_metrics["train_macro_f1"] = round(train_macro_f1, 6)

        mlflow.log_metrics(epoch_metrics, step=epoch)
        history_log.append(epoch_metrics)

        # ── Structured progress log ───────────────────────────────────────
        overfitting_gap = train_acc - val_acc
        f1_gap          = (train_macro_f1 - val_macro_f1) if train_macro_f1 is not None else float("nan")
        logger.info(
            f"Epoch {epoch + 1:3d}/{max_epochs} | "
            f"loss={train_loss:.4f} val_loss={val_loss:.4f} | "
            f"acc={train_acc:.4f} val_acc={val_acc:.4f} (Δ={overfitting_gap:+.3f}) | "
            f"val_macro_f1={val_macro_f1:.4f}"
            + (f" train_f1={train_macro_f1:.4f} (Δ={f1_gap:+.3f})"
               if train_macro_f1 is not None else "")
            + f" | lr={current_lr:.3e} | {epoch_time:.1f}s",
            extra={"stage": "training"},
        )

        # ── Manual early stopping driven by val_macro_f1 ─────────────────
        # This is the SOLE stopping criterion. val_macro_f1 is the primary
        # metric for all checkpoint and selection decisions.
        #
        # Initialised to -1.0 so the first epoch (any F1 ≥ 0.0) always
        # satisfies > best + delta and saves the first checkpoint.
        if val_macro_f1 > best_val_macro_f1 + _F1_MIN_DELTA:
            best_val_macro_f1 = val_macro_f1
            best_val_acc      = val_acc
            best_epoch        = epoch
            patience_counter  = 0

            # Weights-only save: ~5–10× faster than full SavedModel export.
            # Overwriting the same directory on each improvement is intentional.
            _save_weights(model, weights_checkpoint_dir)

            logger.info(
                f"  ↑ New best val_macro_f1={best_val_macro_f1:.4f} "
                f"(epoch {epoch + 1}) — weights checkpoint saved.",
                extra={"stage": "training"},
            )
        else:
            patience_counter += 1
            logger.debug(
                f"  No macro-F1 improvement. "
                f"patience={patience_counter}/{patience_cap}",
                extra={"stage": "training"},
            )

        if patience_counter >= patience_cap:
            logger.info(
                f"Early stopping at epoch {epoch + 1}. "
                f"Best val_macro_f1={best_val_macro_f1:.4f} "
                f"(epoch {best_epoch + 1}). "
                f"patience={patience_cap} exhausted.",
                extra={"stage": "training"},
            )
            stopped_early = True
            break

    total_epochs_trained = len(history_log)

    # Handle degenerate case: training was so short that best_val_macro_f1
    # is still -1.0 (no epoch ran, or max_epochs=0). Clamp to 0.0 for logs.
    if best_val_macro_f1 < 0.0:
        best_val_macro_f1 = 0.0

    # ── 7. Restore best-macro-F1 weights ─────────────────────────────────
    weights_restored = _load_weights(model, weights_checkpoint_dir)

    if not weights_restored:
        # Fallback: no checkpoint exists (model never improved above threshold).
        # This can occur if max_epochs=1 AND the model has F1 < delta on epoch 0,
        # or if the weights directory write failed. Use current model state.
        logger.warning(
            "Weights restoration failed. Using final model state. "
            "This likely indicates the model failed to improve above "
            f"{_F1_MIN_DELTA:.3f} macro-F1 delta in any epoch. "
            "Check training data, class weights, and learning rate.",
            extra={"stage": "training"},
        )
        # Save the final state as the 'best' so SavedModel export succeeds.
        _save_weights(model, weights_checkpoint_dir)

    # Write the full SavedModel ONCE — after restoring the best weights.
    # This is the model that goes to MLflow and TFLite export (Stage 8).
    logger.info(
        f"Exporting SavedModel → {saved_model_dir}",
        extra={"stage": "training"},
    )
    model.save(str(saved_model_dir))

    t_train_elapsed = time.time() - t_run_start
    logger.info(
        f"Training complete | "
        f"epochs={total_epochs_trained} | "
        f"early_stopped={stopped_early} | "
        f"best_epoch={best_epoch + 1} | "
        f"best_val_macro_f1={best_val_macro_f1:.4f} | "
        f"best_val_acc={best_val_acc:.4f} | "
        f"weights_restored={weights_restored} | "
        f"elapsed={t_train_elapsed:.1f}s",
        extra={"stage": "training"},
    )

    # ── 8. Artefact generation (fault-tolerant) ───────────────────────────
    #
    # Each artefact block is wrapped in _artefact_guard(). A matplotlib
    # crash, a disk write failure, or a sklearn edge case will log an error
    # and continue. Training results are NEVER lost due to artefact failures.
    # ──────────────────────────────────────────────────────────────────────

    logger.info(
        f"Generating per-run artefacts in {artifact_dir}",
        extra={"stage": "training"},
    )

    # Collect val predictions from the restored best model (single pass)
    y_true = np.array([], dtype=np.int32)
    y_pred = np.array([], dtype=np.int32)
    report: Dict[str, Any] = {}
    high_risk_f1: Dict[str, float] = {s: 0.0 for s in _HIGH_RISK_SIGNS}

    with _artefact_guard("val predictions"):
        y_true, y_pred = _get_val_predictions(model, val_ds)

    with _artefact_guard("confusion matrices"):
        if y_true.size > 0:
            _save_confusion_matrices(
                y_true, y_pred, sign_names, run_name, artifact_dir
            )

    with _artefact_guard("training curves"):
        if history_log:
            _save_training_curves(history_log, best_epoch, run_name, artifact_dir)

    with _artefact_guard("per-class metrics"):
        if y_true.size > 0:
            report = _save_per_class_metrics(
                y_true, y_pred, sign_names, n_classes, artifact_dir
            )
            high_risk_f1 = _build_high_risk_f1_dict(report)

    with _artefact_guard("metrics history"):
        _save_metrics_history(history_log, artifact_dir)

    with _artefact_guard("config snapshot"):
        _save_config_snapshot(cfg, artifact_dir)

    model_hash = "unavailable"
    with _artefact_guard("model hash"):
        model_hash = _compute_model_hash(model)
        _save_model_hash(model_hash, artifact_dir)

    with _artefact_guard("run manifest"):
        _save_run_manifest(
            cfg=cfg,
            run_name=run_name,
            experiment_group=experiment_group,
            best_epoch=best_epoch,
            best_val_macro_f1=best_val_macro_f1,
            best_val_acc=best_val_acc,
            total_epochs=total_epochs_trained,
            model_param_count=model.count_params(),
            high_risk_f1=high_risk_f1,
            artifact_dir=artifact_dir,
            mlflow_run_id=mlflow_run_id,
            weights_restored=weights_restored,
        )

    # ── 9. Log artefacts and summary metrics to MLflow ────────────────────
    # Artefact uploads are individually guarded to prevent a single upload
    # failure from blocking the remaining uploads.
    _mlflow_log_artefacts(artifact_dir)

    with _artefact_guard("MLflow final metrics"):
        import mlflow as _mlf
        _mlf.log_metrics(
            {
                "best_val_macro_f1":  best_val_macro_f1,
                "best_val_acc":       best_val_acc,
                "total_epochs":       float(total_epochs_trained),
            },
            step=best_epoch,
        )
        _mlf.log_metrics(
            {f"high_risk_f1_{k}": float(v) for k, v in high_risk_f1.items()},
            step=best_epoch,
        )

    with _artefact_guard("MLflow SavedModel log"):
        import mlflow as _mlf
        _mlf.tensorflow.log_model(model, "model")

    with _artefact_guard("MLflow config dict"):
        import mlflow as _mlf
        _mlf.log_dict(cfg.model_dump(mode="json"), "config_snapshot.json")

    # ── 10. Build and return results summary ──────────────────────────────
    result: Dict[str, Any] = {
        "run_name":               run_name,
        "experiment_group":       experiment_group,
        "best_val_macro_f1":      best_val_macro_f1,
        "best_val_acc":           best_val_acc,
        "best_epoch":             best_epoch,
        "total_epochs_trained":   total_epochs_trained,
        "stopped_early":          stopped_early,
        "mlflow_run_id":          mlflow_run_id,
        "config_hash":            cfg.config_hash,
        "model_param_count":      model.count_params(),
        "high_risk_class_f1":     high_risk_f1,
        "artifact_dir":           str(artifact_dir.resolve()),
        "model_save_path":        str(saved_model_dir.resolve()),
        "best_weights_restored":  weights_restored,
        # Fields used by run_all_experiments.py for group selection
        "model_type":             str(cfg.model.name),
        "seq_len":                int(cfg.data.sequence_length),
        "landmark_config":        str(cfg.data.landmark_config),
        "augmentation":           bool(cfg.augmentation.enabled),
        "total_elapsed_sec":      round(time.time() - t_run_start, 1),
    }

    logger.info(
        f"train_one_run COMPLETE | "
        f"run='{run_name}' | "
        f"best_val_macro_f1={best_val_macro_f1:.4f} | "
        f"best_val_acc={best_val_acc:.4f} | "
        f"best_epoch={best_epoch + 1} | "
        f"total_time={result['total_elapsed_sec']}s",
        extra={"stage": "training"},
    )

    return result


def _mlflow_log_artefacts(artifact_dir: Path) -> None:
    """
    Upload per-run artefact files to the active MLflow run.

    Each file upload is individually guarded: a single failed upload
    (e.g. file was not generated due to an earlier artefact failure)
    does not prevent the remaining files from uploading.
    """
    import mlflow

    files_to_log = [
        "confusion_matrix.png",
        "confusion_matrix_norm.png",
        "training_curves.png",
        "per_class_metrics.json",
        "run_manifest.json",
        "config_snapshot.yaml",
        "metrics.json",
        "model_hash.txt",
    ]

    for filename in files_to_log:
        with _artefact_guard(f"MLflow upload: {filename}"):
            path = artifact_dir / filename
            if path.exists():
                mlflow.log_artifact(str(path))
            else:
                logger.debug(
                    f"Skipping MLflow upload for missing artefact: {path.name}",
                    extra={"stage": "training"},
                )