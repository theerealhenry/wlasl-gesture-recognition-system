"""
pipelines/run_training.py
==========================
CLI entry point for training a single WLASL gesture recognition experiment.

This script is the outermost shell for a single experiment run. Its only
responsibilities are:

    1.  Parse and validate CLI arguments.
    2.  Load and validate the composed config (model + data + augmentation +
        optional experiment override).
    3.  Configure structured logging.
    4.  Seed the global RNG state via setup_experiment() for reproducibility.
    5.  Open an MLflow run context and log all parameters and tags.
    6.  Delegate the full training lifecycle to train_one_run() in
        src/models/train.py.
    7.  Exit with a CI/CD-compatible exit code:
            0 — success
            1 — configuration / validation error (bad arguments, missing files)
            2 — training failure (data pipeline error, NaN loss, etc.)
            3 — unexpected exception

Usage
-----
Single run (primary config composition pattern):

    python pipelines/run_training.py \\
        --model bilstm \\
        --data seq60 \\
        --augmentation spatial_temporal \\
        --run-name bilstm_seq60_spatial_temporal

With an optional experiment-level config overlay:

    python pipelines/run_training.py \\
        --model bilstm \\
        --data seq60 \\
        --augmentation spatial_temporal \\
        --experiment best_model \\
        --run-name champion_model_v1

With config overrides (CLI takes highest priority, dot-notation):

    python pipelines/run_training.py \\
        --model lstm \\
        --data seq60 \\
        --augmentation none \\
        --run-name lstm_seq60_lr_sweep \\
        --override training.learning_rate=0.0005 \\
        --override training.epochs=60

With landmark configuration override (Group 4 ablation):

    python pipelines/run_training.py \\
        --model lstm \\
        --data seq60 \\
        --augmentation spatial_temporal \\
        --run-name lstm_seq60_hands_only \\
        --override data.landmark_config=hands_only

Dry run (full config + dataset + model validation, then exit):

    python pipelines/run_training.py \\
        --model bilstm \\
        --data seq60 \\
        --augmentation spatial_temporal \\
        --dry-run

MLflow parameter contract
-------------------------
Every run logs an IDENTICAL set of MLflow parameter keys regardless of
architecture. Optional fields absent from some model YAMLs (num_layers,
recurrent_dropout for dense.yaml) are safely read via _safe_cfg_attr() with
documented defaults. This uniformity is what allows mlflow.search_runs() in
Notebook 05 to build a consistent comparison table across all 17 experiment runs.

Exit codes
----------
    0  — Training completed successfully
    1  — Configuration / validation error (bad arguments, missing files,
          Pydantic schema failure)
    2  — Training failure (RuntimeError from train_one_run: empty dataset,
          NaN/Inf loss, model construction failure)
    3  — Unexpected exception (bug or environment issue)

Design constraints
------------------
- No print() statements — all output via get_logger(__name__). The sole
  exception is the pre-logger error path in _parse_overrides() and
  _validate_run_name(), where the logger is not yet initialised.
- setup_experiment(cfg, run_name, output_dir) is called immediately after
  config load to seed np.random, random, and tf.random before any pipeline
  object is constructed. This is the critical reproducibility call.
- The MLflow run context is opened HERE, not in train_one_run(). This gives
  the CLI full control over run naming and parameter logging order.
- train_one_run() is called inside the open mlflow.start_run() context.
  It calls mlflow.log_metrics() and mlflow.log_artifact() internally.
- feature_dim is read from pipeline.feature_dim, NOT from cfg.data — that
  field does not exist on DataConfig. The pipeline resolves it from
  cfg.data.landmark_config at construction time.
- recurrent_dropout and num_layers are absent from dense.yaml; they are
  read via _safe_cfg_attr() which narrows exception types to AttributeError,
  ValueError, and the OmegaConf ConfigAttributeError.
- mlflow.start_run() is guarded against nesting: if an active run exists,
  RuntimeError is raised immediately. Nested runs cause silent tracking
  corruption in MLflow and must never be permitted.
- Artefact directory collision is checked before training begins. If
  artifacts/experiments/{run_name}/ already contains a run_manifest.json,
  training is aborted with EXIT_CONFIG_ERROR and a clear message.
- All exception handlers log at the correct granularity: known error types
  log their message; unexpected exceptions log the full traceback once only.
"""

from __future__ import annotations

import argparse
import ast
import os
import platform
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Exit code constants — CI/CD-compatible
# ---------------------------------------------------------------------------

EXIT_SUCCESS           = 0
EXIT_CONFIG_ERROR      = 1
EXIT_TRAINING_FAILURE  = 2
EXIT_UNEXPECTED_ERROR  = 3

# ---------------------------------------------------------------------------
# Valid config name sets — validated pre-import to catch typos before TF loads
# ---------------------------------------------------------------------------

_VALID_MODELS: frozenset[str] = frozenset({"dense", "lstm", "gru", "bilstm"})
_VALID_DATA:   frozenset[str] = frozenset({"seq20", "seq30", "seq40", "seq60", "seq80", "seq100"})
_VALID_AUGMENTATION: frozenset[str] = frozenset({"none", "temporal", "spatial_temporal"})

# Known experiment configs (advisory — not fatal if user has custom configs)
_KNOWN_EXPERIMENTS: frozenset[str] = frozenset({
    "baseline",
    "ablation_augmentation",
    "ablation_sequence",
    "ablation_landmarks",
    "best_model",
})

# Valid experiment group labels for MLflow tag filtering in Notebook 05
_VALID_EXPERIMENT_GROUPS: frozenset[str] = frozenset({
    "architecture",
    "augmentation",
    "sequence_length",
    "landmark_config",
    "champion",
    "custom",
})

# Required keys in the dict returned by train_one_run() — validated post-training
# to catch train.py API drift early (Issue #4).
_REQUIRED_RESULT_KEYS: frozenset[str] = frozenset({
    "run_name",
    "experiment_group",
    "best_val_macro_f1",
    "best_val_acc",
    "best_epoch",
    "total_epochs_trained",
    "stopped_early",
    "mlflow_run_id",
    "config_hash",
    "model_param_count",
    "high_risk_class_f1",
    "artifact_dir",
    "model_save_path",
    "best_weights_restored",
    "model_type",
    "seq_len",
    "landmark_config",
    "augmentation",
})


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """
    Build the argument parser for run_training.py.

    Returns
    -------
    argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="run_training.py",
        description=(
            "Train a single WLASL 35-class gesture recognition model. "
            "Composes model + data + augmentation YAML configs, validates them, "
            "seeds the global RNG, opens an MLflow run, and delegates to "
            "train_one_run() in src/models/train.py."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Group 1 — Architecture comparison (isolates architecture, no augmentation)
  python pipelines/run_training.py --model bilstm --data seq60 --augmentation none \\
      --run-name bilstm_seq60_no_aug --group architecture

  # Group 2 — Augmentation ablation (fixed LSTM + seq60)
  python pipelines/run_training.py --model lstm --data seq60 --augmentation spatial_temporal \\
      --run-name lstm_seq60_spatial_temporal --group augmentation

  # Group 3 — Sequence length ablation (seq80 is highest priority)
  python pipelines/run_training.py --model lstm --data seq80 --augmentation spatial_temporal \\
      --run-name lstm_seq80_spatial_temporal --group sequence_length

  # Group 4 — Landmark config ablation (hands_only)
  python pipelines/run_training.py --model lstm --data seq80 --augmentation spatial_temporal \\
      --run-name lstm_seq80_hands_only --group landmark_config \\
      --override data.landmark_config=hands_only

  # Champion model run
  python pipelines/run_training.py --model bilstm --data seq80 --augmentation spatial_temporal \\
      --experiment best_model --run-name champion_model_v1 --group champion

  # Dry run — validates config + dataset + model, then exits without training
  python pipelines/run_training.py --model bilstm --data seq60 --augmentation none --dry-run
        """,
    )

    # ── Required flags ────────────────────────────────────────────────────────

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=sorted(_VALID_MODELS),
        metavar="MODEL",
        help=(
            f"Model architecture config name. One of: {sorted(_VALID_MODELS)}. "
            "Loads configs/model/<MODEL>.yaml."
        ),
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        choices=sorted(_VALID_DATA),
        metavar="DATA",
        help=(
            f"Data/sequence config name. One of: {sorted(_VALID_DATA)}. "
            "Loads configs/data/<DATA>.yaml."
        ),
    )
    parser.add_argument(
        "--augmentation",
        type=str,
        required=True,
        choices=sorted(_VALID_AUGMENTATION),
        metavar="AUG",
        help=(
            f"Augmentation strategy config name. One of: {sorted(_VALID_AUGMENTATION)}. "
            "Loads configs/augmentation/<AUG>.yaml."
        ),
    )

    # ── Optional config composition ───────────────────────────────────────────

    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        metavar="EXPERIMENT",
        help=(
            f"Optional experiment-level config overlay. Known values: {sorted(_KNOWN_EXPERIMENTS)}. "
            "Loads configs/experiment/<EXPERIMENT>.yaml. Applied last — highest YAML priority."
        ),
    )

    # ── Run identity ──────────────────────────────────────────────────────────

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        metavar="RUN_NAME",
        help=(
            "Unique name for this MLflow run and artefact directory. "
            "Must contain only [A-Za-z0-9_-], max 128 chars. "
            "If omitted, auto-generated as '<model>_<data>_<augmentation>_<uuid8>'."
        ),
    )
    parser.add_argument(
        "--group",
        type=str,
        default="custom",
        choices=sorted(_VALID_EXPERIMENT_GROUPS),
        metavar="GROUP",
        help=(
            "Experiment group label for MLflow tag-based filtering in Notebook 05. "
            f"One of: {sorted(_VALID_EXPERIMENT_GROUPS)}. Default: 'custom'."
        ),
    )

    # ── Config overrides ──────────────────────────────────────────────────────

    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Dot-notation config override. May be specified multiple times. "
            "Values are parsed via ast.literal_eval before falling back to str. "
            "Examples: --override training.learning_rate=0.0005 "
            "--override data.landmark_config=hands_only "
            "--override training.class_weight_balancing=true"
        ),
    )

    # ── Execution control ─────────────────────────────────────────────────────

    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help=(
            "Load config, build pipeline + dataset + model, log a full summary, "
            "then exit without training. Validates the entire pre-training stack "
            "including data access and model construction. Does not open an MLflow run."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help=(
            "Allow overwriting an existing artefact directory for this run_name. "
            "Without --force, run_training.py aborts if "
            "artifacts/experiments/{run_name}/run_manifest.json already exists. "
            "Use with care — previous artefacts will be overwritten."
        ),
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=None,
        metavar="URI",
        help=(
            "MLflow tracking URI. Overrides cfg.mlflow.tracking_uri. "
            "Examples: mlruns  |  http://localhost:5000"
        ),
    )
    parser.add_argument(
        "--splits-dir",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Override for split CSV directory. "
            "Default: cfg.data.splits_dir (data/splits)."
        ),
    )
    parser.add_argument(
        "--landmarks-dir",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Override for landmarks root directory. "
            "Default: cfg.data.landmark_dir (data/landmarks)."
        ),
    )

    return parser


# ---------------------------------------------------------------------------
# Override parsing — with ast.literal_eval for structured values (Issue #11)
# ---------------------------------------------------------------------------

def _parse_overrides(override_list: List[str]) -> Dict[str, Any]:
    """
    Parse a list of 'KEY=VALUE' override strings into a typed dict.

    Type coercion priority (from highest to lowest):
        1. ast.literal_eval  — handles bool, int, float, list, dict, None
        2. str               — fallback for plain string values like "hands_only"

    Note: ast.literal_eval is used instead of raw bool/int/float parsing so
    that complex overrides like --override model.some_list=[64,128] work
    correctly. Plain string values that are not Python literals fall through
    to the str fallback.

    Parameters
    ----------
    override_list : list[str]
        Raw override strings from argparse.

    Returns
    -------
    dict[str, Any]

    Raises
    ------
    SystemExit (EXIT_CONFIG_ERROR)
        If any override string is malformed (missing '=' separator).
    """
    if not override_list:
        return {}

    result: Dict[str, Any] = {}
    errors: List[str] = []

    for raw in override_list:
        if "=" not in raw:
            errors.append(
                f"  Malformed override '{raw}': expected 'KEY=VALUE' format. "
                "Example: --override training.learning_rate=0.0005"
            )
            continue

        key, _, value_str = raw.partition("=")
        key = key.strip()
        value_str = value_str.strip()

        if not key:
            errors.append(f"  Empty key in override: '{raw}'")
            continue

        # Attempt ast.literal_eval for structured types (bool, int, float,
        # list, dict, None). Fall back to raw string for non-literal values
        # like landmark config names ("hands_only", "full").
        value: Any
        try:
            value = ast.literal_eval(value_str)
        except (ValueError, SyntaxError):
            # Not a Python literal — treat as plain string
            value = value_str

        result[key] = value

    if errors:
        print(
            f"[run_training] CONFIG ERROR: {len(errors)} malformed override(s):\n"
            + "\n".join(errors),
            file=sys.stderr,
        )
        sys.exit(EXIT_CONFIG_ERROR)

    return result


# ---------------------------------------------------------------------------
# Run name generation and validation
# ---------------------------------------------------------------------------

def _generate_run_name(model: str, data: str, augmentation: str) -> str:
    """
    Auto-generate a globally-unique, filesystem-safe run name.

    Format: <model>_<data>_<augmentation>_<uuid8>

    A UUID4 suffix (8 hex chars) rather than a millisecond timestamp is
    used to guarantee uniqueness even when run_all_experiments.py launches
    multiple runs in tight succession within the same millisecond.

    Parameters
    ----------
    model, data, augmentation : str

    Returns
    -------
    str  e.g. "bilstm_seq60_spatial_temporal_3f7a1c4b"
    """
    suffix = uuid.uuid4().hex[:8]
    return f"{model}_{data}_{augmentation}_{suffix}"


def _validate_run_name(run_name: str) -> str:
    """
    Validate and return a normalised run name.

    Rules:
      - Non-empty after stripping whitespace.
      - Contains only [A-Za-z0-9_-] (filesystem and MLflow safe).
      - At most 128 characters (MLflow UI truncates longer names).

    Parameters
    ----------
    run_name : str

    Returns
    -------
    str  Validated (stripped) run name.

    Raises
    ------
    SystemExit (EXIT_CONFIG_ERROR)  On any rule violation.
    """
    import re

    name = run_name.strip()

    if not name:
        print(
            "[run_training] CONFIG ERROR: --run-name is empty after stripping.",
            file=sys.stderr,
        )
        sys.exit(EXIT_CONFIG_ERROR)

    if len(name) > 128:
        print(
            f"[run_training] CONFIG ERROR: --run-name is {len(name)} characters; "
            "maximum is 128.",
            file=sys.stderr,
        )
        sys.exit(EXIT_CONFIG_ERROR)

    if not re.match(r"^[A-Za-z0-9_\-]+$", name):
        sanitised = re.sub(r"[^A-Za-z0-9_\-]", "_", name)
        print(
            f"[run_training] CONFIG ERROR: --run-name '{name}' contains "
            "unsafe characters (only A-Z a-z 0-9 _ - are allowed). "
            f"Suggested sanitised name: '{sanitised}'",
            file=sys.stderr,
        )
        sys.exit(EXIT_CONFIG_ERROR)

    return name


# ---------------------------------------------------------------------------
# Safe config attribute reader — narrow exception scope (Issue #5)
# ---------------------------------------------------------------------------

def _safe_cfg_attr(cfg_section: Any, attr: str, default: Any) -> Any:
    """
    Safely retrieve an optional attribute from a Pydantic/OmegaConf section.

    Catches only the specific exception types that legitimately arise from
    absent config fields:
      - AttributeError        — standard Python missing attribute
      - ValueError            — Pydantic validation failure on access
      - omegaconf ConfigAttributeError — OmegaConf-specific missing key

    Any other exception is re-raised, preserving traceability of genuine bugs.

    This is used for fields absent from some model YAMLs:
      - num_layers        — absent from dense.yaml
      - recurrent_dropout — absent from dense.yaml

    Parameters
    ----------
    cfg_section : Any     Pydantic model section (e.g. cfg.model).
    attr        : str     Attribute name.
    default     : Any     Return value if the attribute is absent.

    Returns
    -------
    Any  The attribute value, or default if absent.
    """
    _CAUGHT: Tuple[type, ...] = (AttributeError, ValueError)

    try:
        from omegaconf.errors import ConfigAttributeError as _OmegaAttrErr
        _CAUGHT = (AttributeError, ValueError, _OmegaAttrErr)
    except ImportError:
        pass

    try:
        val = getattr(cfg_section, attr, _SENTINEL := object())
        if val is _SENTINEL:
            return default
        if val is None:
            return default
        # Guard against OmegaConf MISSING sentinel
        try:
            from omegaconf import MISSING as _OMEGACONF_MISSING
            if val is _OMEGACONF_MISSING:
                return default
        except ImportError:
            pass
        return val
    except _CAUGHT:
        return default
    # All other exceptions propagate — they indicate genuine bugs.


# ---------------------------------------------------------------------------
# Result contract validation (Issue #4)
# ---------------------------------------------------------------------------

def _validate_result_contract(result: Dict[str, Any], run_name: str) -> None:
    """
    Verify that train_one_run() returned all required keys.

    This guard catches API drift in train.py (e.g. a key rename from
    'best_epoch' to 'epoch_best') before downstream code silently receives
    a KeyError deep inside _log_run_footer() or run_all_experiments.py.

    Parameters
    ----------
    result   : dict  Return value of train_one_run().
    run_name : str   For error message context.

    Raises
    ------
    RuntimeError  If any required key is absent.
    """
    missing: Set[str] = _REQUIRED_RESULT_KEYS - result.keys()
    if missing:
        raise RuntimeError(
            f"train_one_run() result for run '{run_name}' is missing required keys: "
            f"{sorted(missing)}. "
            "This indicates an API contract change in src/models/train.py. "
            "Update _REQUIRED_RESULT_KEYS in run_training.py to match the new contract, "
            "or fix the missing keys in train_one_run()."
        )


# ---------------------------------------------------------------------------
# Artefact directory collision check (Issue #7)
# ---------------------------------------------------------------------------

def _check_artefact_collision(run_name: str, force: bool) -> None:
    """
    Abort if artifacts/experiments/{run_name}/run_manifest.json already exists
    and --force has not been specified.

    Presence of run_manifest.json is the definitive signal that a previous
    training run for this name completed successfully. Overwriting it silently
    would corrupt the experiment record for that run.

    Parameters
    ----------
    run_name : str
    force    : bool  If True, skip the check (--force flag).

    Raises
    ------
    SystemExit (EXIT_CONFIG_ERROR)  If collision detected and --force absent.
    """
    if force:
        return

    manifest_path = Path("artifacts") / "experiments" / run_name / "run_manifest.json"
    if manifest_path.exists():
        print(
            f"[run_training] CONFIG ERROR: artefact directory for run '{run_name}' "
            f"already exists and contains a completed run manifest:\n"
            f"  {manifest_path.resolve()}\n"
            "Choose a different --run-name, or use --force to overwrite.",
            file=sys.stderr,
        )
        sys.exit(EXIT_CONFIG_ERROR)


# ---------------------------------------------------------------------------
# Environment snapshot for reproducibility tags (Issue #12)
# ---------------------------------------------------------------------------

def _collect_environment_snapshot() -> Dict[str, str]:
    """
    Collect a minimal environment snapshot for MLflow tags.

    Captures the key dependency versions and platform information needed
    to reproduce a training run 6+ months later. All values are strings
    (MLflow tag API requirement).

    Returns
    -------
    dict[str, str]  Keys prefixed with "env." for namespace clarity in the MLflow UI.
    """
    snap: Dict[str, str] = {}

    snap["env.python_version"] = platform.python_version()
    snap["env.platform"]       = platform.platform()

    try:
        import tensorflow as tf
        snap["env.tensorflow_version"] = tf.__version__
    except ImportError:
        snap["env.tensorflow_version"] = "unavailable"

    try:
        import numpy as np
        snap["env.numpy_version"] = np.__version__
    except ImportError:
        snap["env.numpy_version"] = "unavailable"

    try:
        import mediapipe as mp
        snap["env.mediapipe_version"] = mp.__version__
    except ImportError:
        snap["env.mediapipe_version"] = "unavailable"

    try:
        import mlflow
        snap["env.mlflow_version"] = mlflow.__version__
    except ImportError:
        snap["env.mlflow_version"] = "unavailable"

    # Git commit hash — non-fatal if not in a git repo
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        snap["env.git_commit"] = result.stdout.strip() if result.returncode == 0 else "unavailable"
    except Exception:
        snap["env.git_commit"] = "unavailable"

    # CUDA availability
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        snap["env.gpu_count"] = str(len(gpus))
        snap["env.cuda_available"] = "true" if gpus else "false"
    except Exception:
        snap["env.gpu_count"]      = "unavailable"
        snap["env.cuda_available"] = "unavailable"

    return snap


# ---------------------------------------------------------------------------
# MLflow parameter and tag logging
# ---------------------------------------------------------------------------

def _log_mlflow_params(
    cfg:              Any,
    pipeline:         Any,
    dataset:          Any,
    model_summary:    Dict[str, Any],
    run_name:         str,
    experiment_group: str,
) -> None:
    """
    Log all run parameters to the active MLflow run.

    CONTRACT: every run across all 17 Stage 5 experiments MUST log an
    IDENTICAL set of parameter keys. mlflow.search_runs() in Notebook 05
    builds a comparison table from these keys — missing keys produce NaN.

    Fields absent from some model YAMLs (num_layers, recurrent_dropout for
    dense.yaml) are read via _safe_cfg_attr() which narrows exception scope
    to the specific types that legitimately arise from absent config fields.

    NOTE: feature_dim is read from pipeline.feature_dim, NOT cfg.data —
    that field does not exist on DataConfig. The pipeline resolves it from
    cfg.data.landmark_config at construction time via LANDMARK_CONFIGS.

    NOTE: experiment_group is logged as a TAG (via _set_mlflow_tags), not
    as a parameter here. Tags are the correct MLflow mechanism for filtering
    and grouping; duplicating the value as a parameter adds no information.
    """
    import mlflow

    model_type_str = (
        cfg.model.name.value
        if hasattr(cfg.model.name, "value")
        else str(cfg.model.name)
    )
    padding_str = (
        cfg.data.padding.value
        if hasattr(cfg.data.padding, "value")
        else str(cfg.data.padding)
    )
    normalisation_str = (
        cfg.data.normalisation.value
        if hasattr(cfg.data.normalisation, "value")
        else str(cfg.data.normalisation)
    )

    # Compute steps_per_epoch from dataset counts and batch_size (Issue #9)
    batch_size = int(cfg.training.batch_size)
    train_steps = max(1, (dataset.n_train + batch_size - 1) // batch_size)
    val_steps   = max(1, (dataset.n_val   + batch_size - 1) // batch_size)

    # ── Model architecture ─────────────────────────────────────────────────
    mlflow.log_params({
        "model_type":          model_type_str,
        "hidden_units":        int(cfg.model.hidden_units),
        "num_layers":          int(_safe_cfg_attr(cfg.model, "num_layers",        1)),
        "dropout":             float(cfg.model.dropout),
        "recurrent_dropout":   float(_safe_cfg_attr(cfg.model, "recurrent_dropout", 0.0)),
        "bidirectional":       bool(_safe_cfg_attr(cfg.model, "bidirectional",    False)),
        "dense_units":         int(_safe_cfg_attr(cfg.model, "dense_units",       64)),
        "activation":          str(_safe_cfg_attr(cfg.model, "activation",        "relu")),
    })

    # ── Data ──────────────────────────────────────────────────────────────
    mlflow.log_params({
        "seq_len":             int(cfg.data.sequence_length),
        "landmark_config":     str(cfg.data.landmark_config),
        "feature_dim":         int(pipeline.feature_dim),   # from pipeline, NOT cfg.data
        "n_train":             int(dataset.n_train),
        "n_val":               int(dataset.n_val),
        "n_test":              int(dataset.n_test),
        "n_classes":           int(cfg.num_classes),
        "train_steps_per_epoch": train_steps,
        "val_steps":           val_steps,
        "padding":             padding_str,
        "normalisation":       normalisation_str,
        "z_coord_clip":        float(cfg.data.z_coord_clip),
        "normalise_pose":      bool(cfg.data.normalise_pose),
        "flip_min_hand_pres":  float(cfg.data.flip_min_hand_presence),
    })

    # ── Training loop ──────────────────────────────────────────────────────
    mlflow.log_params({
        "batch_size":                 int(cfg.training.batch_size),
        "learning_rate":              float(cfg.training.learning_rate),
        "epochs_max":                 int(cfg.training.epochs),
        "early_stopping_patience":    int(cfg.training.early_stopping_patience),
        "early_stopping_monitor":     str(cfg.training.early_stopping_monitor),
        "reduce_lr_patience":         int(cfg.training.reduce_lr_patience),
        "reduce_lr_factor":           float(cfg.training.reduce_lr_factor),
        "reduce_lr_min_lr":           float(cfg.training.reduce_lr_min_lr),
        "class_weight_balancing":     bool(cfg.training.class_weight_balancing),
        "seed":                       int(cfg.seed),
    })

    # ── Augmentation ───────────────────────────────────────────────────────
    mlflow.log_params({
        "augmentation_enabled":       bool(cfg.augmentation.enabled),
        "temporal_jitter":            bool(cfg.augmentation.temporal_jitter),
        "frame_drop_prob":            float(cfg.augmentation.frame_drop_prob),
        "speed_jitter":               bool(cfg.augmentation.speed_jitter),
        "spatial_flip":               bool(cfg.augmentation.spatial_flip),
        "gaussian_noise_std":         float(cfg.augmentation.gaussian_noise_std),
        "gaussian_noise_det_only":    bool(cfg.augmentation.gaussian_noise_detected_only),
        "rotation_deg":               float(cfg.augmentation.rotation_deg),
    })

    # ── Model statistics (post-build actuals, not config estimates) ────────
    mlflow.log_params({
        "total_params":       int(model_summary["param_count"]),
        "trainable_params":   int(model_summary["trainable_params"]),
        "model_size_mb":      float(model_summary["model_size_mb_estimate"]),
        "model_keras_name":   str(model_summary["model_name"]),
    })

    # ── Config identity (for unambiguous experiment attribution) ───────────
    mlflow.log_params({
        "run_name":           run_name,
        "config_hash":        str(cfg.config_hash),
        "mlflow_experiment":  str(cfg.mlflow.experiment_name),
    })
    # NOTE: experiment_group is intentionally NOT logged here as a param —
    # it is logged as a MLflow tag in _set_mlflow_tags(), which is the
    # correct mechanism for categorical run-grouping metadata.


def _set_mlflow_tags(
    cfg:              Any,
    experiment_group: str,
    run_name:         str,
    args:             argparse.Namespace,
    env_snapshot:     Dict[str, str],
) -> None:
    """
    Set MLflow tags for filtering and grouping in the Tracking UI and Notebook 05.

    All tag values must be strings (MLflow API requirement). Tags include:
      - Core experiment identity (group, model, seq_len, augmentation)
      - Data provenance claims (signer_split, num_classes)
      - Metric policy (primary_metric = val_macro_f1)
      - Environment snapshot (env.* prefix)

    Parameters
    ----------
    cfg              : ExperimentConfig
    experiment_group : str
    run_name         : str
    args             : argparse.Namespace
    env_snapshot     : dict  from _collect_environment_snapshot()
    """
    import mlflow

    model_type_str = (
        cfg.model.name.value
        if hasattr(cfg.model.name, "value")
        else str(cfg.model.name)
    )

    tags = {
        # Core run identity
        "experiment_group":   str(experiment_group),
        "model_type":         model_type_str,
        "landmark_config":    str(cfg.data.landmark_config),
        "seq_len":            str(cfg.data.sequence_length),
        "augmentation":       "enabled" if cfg.augmentation.enabled else "disabled",
        "augmentation_name":  str(args.augmentation),
        "data_config":        str(args.data),
        "model_config":       str(args.model),
        # Data provenance
        "signer_split":       "true",
        "num_classes":        "35",
        # Metric policy — critical for champion selection reproducibility
        "primary_metric":     "val_macro_f1",
        "secondary_metric":   "val_accuracy",
        # Config fingerprint
        "config_hash_short":  str(cfg.config_hash)[:12],
        # Post-run inspection targets
        "monitor_classes":    "clothes,think,birthday,name,book",
        "run_name":           run_name,
    }

    # Merge environment snapshot (env.* keys)
    tags.update(env_snapshot)

    mlflow.set_tags(tags)


# ---------------------------------------------------------------------------
# Pre-training and post-training summary logs
# ---------------------------------------------------------------------------

def _log_run_header(
    logger:           Any,
    args:             argparse.Namespace,
    run_name:         str,
    experiment_group: str,
    cfg:              Any,
    pipeline:         Any,
    dataset:          Any,
    model_summary:    Dict[str, Any],
) -> None:
    """
    Emit a structured pre-training summary at INFO level.

    Provides a single grep-able log block that fully describes the run
    before training begins, useful for CI log inspection and post-mortem
    analysis of interrupted runs.
    """
    model_name = (
        cfg.model.name.value
        if hasattr(cfg.model.name, "value")
        else str(cfg.model.name)
    )

    aug = cfg.augmentation
    if not aug.enabled:
        aug_summary = "DISABLED"
    else:
        parts = []
        if aug.temporal_jitter:
            parts.append(f"jitter(p={aug.frame_drop_prob})")
        if aug.speed_jitter:
            parts.append("speed")
        if aug.gaussian_noise_std > 0:
            parts.append(f"noise(std={aug.gaussian_noise_std},per_slot)")
        if aug.rotation_deg > 0:
            parts.append(f"rot(±{aug.rotation_deg}°)")
        if aug.spatial_flip:
            parts.append(f"flip(min={cfg.data.flip_min_hand_presence})")
        aug_summary = " → ".join(parts) if parts else "enabled(no_ops)"

    logger.info("=" * 72, extra={"stage": "run_training"})
    logger.info("WLASL Stage 5 Training Run", extra={"stage": "run_training"})
    logger.info(f"  run_name:       {run_name}", extra={"stage": "run_training"})
    logger.info(f"  group:          {experiment_group}", extra={"stage": "run_training"})
    logger.info(f"  config_hash:    {cfg.config_hash[:24]}", extra={"stage": "run_training"})
    logger.info(
        f"  model:          {model_name} | "
        f"hidden={cfg.model.hidden_units} | "
        f"layers={_safe_cfg_attr(cfg.model, 'num_layers', 1)} | "
        f"drop={cfg.model.dropout} | "
        f"params={model_summary['param_count']:,} | "
        f"size={model_summary['model_size_mb_estimate']:.3f}MB",
        extra={"stage": "run_training"},
    )
    logger.info(
        f"  data:           seq_len={cfg.data.sequence_length} | "
        f"landmark={cfg.data.landmark_config} | "
        f"feature_dim={pipeline.feature_dim} | "
        f"train={dataset.n_train} val={dataset.n_val} test={dataset.n_test}",
        extra={"stage": "run_training"},
    )
    logger.info(f"  augmentation:   {aug_summary}", extra={"stage": "run_training"})
    logger.info(
        f"  training:       epochs_max={cfg.training.epochs} | "
        f"batch={cfg.training.batch_size} | "
        f"lr={cfg.training.learning_rate:.2e} | "
        f"patience={cfg.training.early_stopping_patience} | "
        f"class_weights={'yes' if cfg.training.class_weight_balancing else 'NO — WARNING'}",
        extra={"stage": "run_training"},
    )
    logger.info(
        f"  mlflow:         experiment='{cfg.mlflow.experiment_name}'",
        extra={"stage": "run_training"},
    )
    logger.info("=" * 72, extra={"stage": "run_training"})


def _log_run_footer(
    logger:  Any,
    result:  Dict[str, Any],
    elapsed: float,
) -> None:
    """
    Emit a structured post-training summary at INFO level.

    Called only after _validate_result_contract() passes, so all key
    accesses here are guaranteed to succeed.
    """
    logger.info("=" * 72, extra={"stage": "run_training"})
    logger.info("Training run COMPLETE", extra={"stage": "run_training"})
    logger.info(f"  run_name:           {result['run_name']}", extra={"stage": "run_training"})
    logger.info(f"  best_val_macro_f1:  {result['best_val_macro_f1']:.4f}", extra={"stage": "run_training"})
    logger.info(f"  best_val_acc:       {result['best_val_acc']:.4f}", extra={"stage": "run_training"})
    logger.info(f"  best_epoch:         {result['best_epoch'] + 1}", extra={"stage": "run_training"})
    logger.info(
        f"  total_epochs:       {result['total_epochs_trained']} "
        f"({'early-stopped' if result.get('stopped_early') else 'full'})",
        extra={"stage": "run_training"},
    )
    logger.info(f"  weights_restored:   {result['best_weights_restored']}", extra={"stage": "run_training"})
    logger.info(f"  mlflow_run_id:      {result['mlflow_run_id']}", extra={"stage": "run_training"})
    logger.info(f"  artefact_dir:       {result['artifact_dir']}", extra={"stage": "run_training"})
    logger.info(f"  saved_model:        {result['model_save_path']}", extra={"stage": "run_training"})
    logger.info(
        f"  total_elapsed:      {elapsed:.1f}s ({elapsed / 60:.1f}min)",
        extra={"stage": "run_training"},
    )

    # High-risk class inspection
    hr = result.get("high_risk_class_f1", {})
    if hr:
        hr_str    = " | ".join(f"{sign}={f1:.3f}" for sign, f1 in sorted(hr.items()))
        zero_f1   = [s for s, f1 in hr.items() if f1 == 0.0]
        low_f1    = [s for s, f1 in hr.items() if 0.0 < f1 < 0.20]
        logger.info(f"  high_risk_class_f1: {hr_str}", extra={"stage": "run_training"})
        if zero_f1:
            logger.warning(
                f"  ATTENTION: {len(zero_f1)} high-risk class(es) have F1=0.0: "
                f"{zero_f1}. These signs failed to learn. "
                "Document in LIMITATIONS.md.",
                extra={"stage": "run_training"},
            )
        if low_f1:
            logger.warning(
                f"  ATTENTION: {len(low_f1)} high-risk class(es) have F1<0.20: {low_f1}.",
                extra={"stage": "run_training"},
            )

    if not result.get("best_weights_restored", True):
        logger.warning(
            "  ATTENTION: best-checkpoint weights could NOT be restored. "
            "SavedModel uses final (not best) weights. "
            "Flag this run in experiment_summary.md.",
            extra={"stage": "run_training"},
        )

    # Target gate reporting
    f1 = result["best_val_macro_f1"]
    if f1 >= 0.70:
        logger.info(
            f"  TARGET MET: val_macro_f1={f1:.4f} >= 0.70 ✓",
            extra={"stage": "run_training"},
        )
    elif f1 >= 0.60:
        logger.info(
            f"  TARGET VIABLE: val_macro_f1={f1:.4f} "
            "(minimum viability 0.60 ✓, target 0.70 ✗). "
            "Consider champion run with larger hidden_units.",
            extra={"stage": "run_training"},
        )
    else:
        logger.warning(
            f"  TARGET NOT MET: val_macro_f1={f1:.4f} < 0.60. "
            "Inspect training curves for underfitting or gradient instability.",
            extra={"stage": "run_training"},
        )

    logger.info("=" * 72, extra={"stage": "run_training"})


def _log_dry_run_summary(
    logger:           Any,
    cfg:              Any,
    pipeline:         Any,
    dataset:          Any,
    model_summary:    Dict[str, Any],
    run_name:         str,
    experiment_group: str,
) -> None:
    """Emit a dry-run summary and return cleanly."""
    model_name = (
        cfg.model.name.value
        if hasattr(cfg.model.name, "value")
        else str(cfg.model.name)
    )
    logger.info(
        "DRY RUN — configuration validated successfully. No training performed.",
        extra={"stage": "run_training"},
    )
    logger.info(f"  config_hash:    {cfg.config_hash[:24]}", extra={"stage": "run_training"})
    logger.info(f"  run_name:       {run_name}", extra={"stage": "run_training"})
    logger.info(f"  group:          {experiment_group}", extra={"stage": "run_training"})
    logger.info(
        f"  architecture:   {model_name} | "
        f"{model_summary['param_count']:,} params | "
        f"{model_summary['model_size_mb_estimate']:.3f} MB",
        extra={"stage": "run_training"},
    )
    logger.info(
        f"  input_shape:    (batch, {cfg.data.sequence_length}, {pipeline.feature_dim})",
        extra={"stage": "run_training"},
    )
    logger.info(
        f"  output_shape:   (batch, {cfg.num_classes})",
        extra={"stage": "run_training"},
    )
    logger.info(
        f"  dataset:        train={dataset.n_train} | val={dataset.n_val} | test={dataset.n_test}",
        extra={"stage": "run_training"},
    )
    logger.info(
        f"  class_weights:  {'enabled' if cfg.training.class_weight_balancing else 'DISABLED — WARNING'}",
        extra={"stage": "run_training"},
    )
    if not cfg.training.class_weight_balancing:
        logger.warning(
            "class_weight_balancing=False. All Stage 5 runs require True "
            "(6.50× imbalance ratio). Add --override training.class_weight_balancing=True.",
            extra={"stage": "run_training"},
        )
    logger.info(
        "DRY RUN complete. Remove --dry-run to begin training.",
        extra={"stage": "run_training"},
    )


# ---------------------------------------------------------------------------
# Core run execution
# ---------------------------------------------------------------------------

def _execute_run(
    args:             argparse.Namespace,
    run_name:         str,
    experiment_group: str,
    overrides:        Dict[str, Any],
) -> Tuple[int, Optional[Dict[str, Any]]]:
    """
    Execute the full run: config load → seed → MLflow → training → result.

    Separated from main() so run_all_experiments.py can call it directly
    without subprocess overhead, sharing the same Python / TF process.

    All exceptions are caught and converted to (exit_code, None) returns.
    The function never raises.

    Parameters
    ----------
    args             : argparse.Namespace
    run_name         : str   (validated)
    experiment_group : str
    overrides        : dict[str, Any]

    Returns
    -------
    Tuple[int, Optional[dict]]
        (exit_code, result_dict_or_None)
    """
    import mlflow

    # Lazy imports — keep TF / MediaPipe out of the pre-import path so
    # that --help and argument validation complete in <1 second.
    from src.features.dataset import GestureDataset
    from src.features.pipeline import FeaturePipeline
    from src.models.factory import build_model, get_model_summary_dict
    from src.models.train import train_one_run
    from src.utils.config import load_config
    from src.utils.logger import get_logger
    from src.utils.reproducibility import setup_experiment

    logger = get_logger(__name__)

    t_script_start = time.time()

    # ── Separate MLflow tag overrides from real config overrides ──────────
    #
    # run_all_experiments.py injects _tag_* keys into the overrides dict to
    # carry session provenance metadata (batch_id, dataset_split_version).
    # These keys MUST NOT reach load_config() → OmegaConf → Pydantic because
    # ExperimentConfig uses extra="forbid" and will raise ValidationError.
    #
    # Split strategy:
    #   _tag_*  → strip prefix, collect as {tag_name: value} for mlflow.set_tags()
    #   other _ → drop silently (informational markers, no config/MLflow target)
    #   rest    → pass to load_config() as real config overrides
    mlflow_tag_overrides: Dict[str, str] = {}
    clean_overrides:      Dict[str, Any] = {}

    for k, v in overrides.items():
        if k.startswith("_tag_"):
            mlflow_tag_overrides[k[5:]] = str(v)   # strip "_tag_" prefix
        elif k.startswith("_"):
            logger.debug(
                f"Dropping private informational override key '{k}' — "
                "no config field or MLflow tag destination.",
                extra={"stage": "run_training"},
            )
        else:
            clean_overrides[k] = v

    overrides = clean_overrides   # only real config keys from here on

    # ── Step 1: Load and validate config ──────────────────────────────────
    logger.info(
        f"Loading config | model={args.model} | data={args.data} | "
        f"augmentation={args.augmentation} | experiment={args.experiment} | "
        f"overrides={overrides}",
        extra={"stage": "run_training"},
    )

    try:
        cfg = load_config(
            model=args.model,
            data=args.data,
            augmentation=args.augmentation,
            experiment=args.experiment,
            overrides=overrides if overrides else None,
        )
    except FileNotFoundError as exc:
        logger.error(
            f"Config file not found: {exc}",
            extra={"stage": "run_training"},
        )
        return EXIT_CONFIG_ERROR, None
    except Exception as exc:
        logger.error(
            f"Config validation failed: {type(exc).__name__}: {exc}",
            extra={"stage": "run_training"},
        )
        return EXIT_CONFIG_ERROR, None

    # ── Step 2: Seed global RNG — CRITICAL for reproducibility ────────────
    #
    # setup_experiment() calls np.random.seed(), random.seed(), and
    # tf.random.set_seed() using cfg.seed. This MUST happen before any
    # pipeline object (FeaturePipeline, GestureDataset, model) is constructed
    # so that weight initialisation, dataset shuffling, and augmentation are
    # all deterministic for this (config, seed) pair.
    #
    # The output_dir for this call is the artefact directory for the run.
    # setup_experiment() may write a run_manifest stub or similar.
    artifact_dir = Path("artifacts") / "experiments" / run_name
    artifact_dir.mkdir(parents=True, exist_ok=True)

    try:
        setup_experiment(
            config=cfg,
            run_name=run_name,
            output_dir=str(artifact_dir),
        )
        logger.info(
            f"Global RNG seeded | seed={cfg.seed} | config_hash={cfg.config_hash[:12]}",
            extra={"stage": "run_training"},
        )
    except Exception as exc:
        logger.error(
            f"setup_experiment() failed: {type(exc).__name__}: {exc}",
            extra={"stage": "run_training"},
        )
        return EXIT_CONFIG_ERROR, None

    # ── Step 3: Mandatory class_weight_balancing guard ────────────────────
    if not cfg.training.class_weight_balancing:
        logger.warning(
            "cfg.training.class_weight_balancing=False. "
            "All Stage 5 runs require True (6.50× imbalance; clothes has 2 clips). "
            "Add --override training.class_weight_balancing=True to enforce.",
            extra={"stage": "run_training"},
        )

    # ── Step 4: Configure MLflow tracking URI ─────────────────────────────
    tracking_uri = args.mlflow_tracking_uri or cfg.mlflow.tracking_uri
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(
        f"MLflow tracking URI: {mlflow.get_tracking_uri()}",
        extra={"stage": "run_training"},
    )

    # ── Step 5: Build FeaturePipeline and GestureDataset ──────────────────
    # Constructed BEFORE opening the MLflow run so data errors fail fast
    # without creating orphan run entries in the MLflow tracking store.
    try:
        pipeline = FeaturePipeline(cfg)
    except Exception as exc:
        logger.error(
            f"FeaturePipeline construction failed: {type(exc).__name__}: {exc}",
            extra={"stage": "run_training"},
        )
        return EXIT_CONFIG_ERROR, None

    splits_dir    = args.splits_dir    or None
    landmarks_dir = args.landmarks_dir or None

    try:
        dataset = GestureDataset(
            cfg,
            pipeline,
            splits_dir=splits_dir,
            landmarks_dir=landmarks_dir,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error(
            f"GestureDataset construction failed: {type(exc).__name__}: {exc}",
            extra={"stage": "run_training"},
        )
        return EXIT_CONFIG_ERROR, None
    except Exception as exc:
        # Unexpected — log full traceback once, do not repeat it.
        logger.error(
            f"Unexpected error building GestureDataset: {type(exc).__name__}: {exc}\n"
            f"{traceback.format_exc()}",
            extra={"stage": "run_training"},
        )
        return EXIT_UNEXPECTED_ERROR, None

    # ── Step 6: Build model ────────────────────────────────────────────────
    try:
        model = build_model(cfg, pipeline=pipeline)
    except (ValueError, TypeError) as exc:
        logger.error(
            f"Model construction failed: {type(exc).__name__}: {exc}",
            extra={"stage": "run_training"},
        )
        return EXIT_CONFIG_ERROR, None
    except Exception as exc:
        # Unexpected — log full traceback once.
        logger.error(
            f"Unexpected error building model: {type(exc).__name__}: {exc}\n"
            f"{traceback.format_exc()}",
            extra={"stage": "run_training"},
        )
        return EXIT_UNEXPECTED_ERROR, None

    model_summary = get_model_summary_dict(model)

    # ── Step 7: Dry-run exit point ─────────────────────────────────────────
    # NOTE: dry-run validates the full pre-training stack (config, pipeline,
    # dataset, model). This is intentional — "full validation" mode, not just
    # config loading. The flag is accurately described in the --help text.
    if args.dry_run:
        _log_dry_run_summary(
            logger, cfg, pipeline, dataset,
            model_summary, run_name, experiment_group,
        )
        return EXIT_SUCCESS, None

    # ── Step 8: Collect environment snapshot before opening the MLflow run ─
    env_snapshot = _collect_environment_snapshot()

    # ── Step 9: Guard against MLflow run nesting (Issue #2) ───────────────
    # Nested runs cause silent tracking corruption. If run_single_experiment()
    # is called from within an active run context, fail loudly and immediately.
    if mlflow.active_run() is not None:
        active_id = mlflow.active_run().info.run_id
        logger.error(
            f"An MLflow run is already active (run_id={active_id}). "
            "Cannot start a new run inside an existing one — this would create "
            "a nested run which corrupts the experiment tracking record. "
            "Ensure run_all_experiments.py ends each run before starting the next.",
            extra={"stage": "run_training"},
        )
        return EXIT_CONFIG_ERROR, None

    # ── Step 10: Open MLflow run context and train ─────────────────────────
    logger.info(
        f"Starting MLflow run '{run_name}' in experiment '{cfg.mlflow.experiment_name}'",
        extra={"stage": "run_training"},
    )

    mlflow.set_experiment(cfg.mlflow.experiment_name)

    result: Optional[Dict[str, Any]] = None

    try:
        # nested=False is explicit — we verified no active run above.
        with mlflow.start_run(run_name=run_name, nested=False) as mlflow_run:
            mlflow_run_id = mlflow_run.info.run_id
            logger.info(
                f"MLflow run opened | run_id={mlflow_run_id}",
                extra={"stage": "run_training"},
            )

            # Log all parameters and tags BEFORE training begins.
            # A run interrupted at epoch 3 still has its full config logged.
            _log_mlflow_params(
                cfg, pipeline, dataset, model_summary,
                run_name, experiment_group,
            )
            _set_mlflow_tags(cfg, experiment_group, run_name, args, env_snapshot)

            # Apply provenance tags injected by run_all_experiments.py.
            # These were carried as _tag_* keys in the overrides dict and
            # extracted above. Applied here after the run context is open
            # so they are attached to the correct MLflow run record.
            if mlflow_tag_overrides:
                mlflow.set_tags(mlflow_tag_overrides)
                logger.debug(
                    f"Applied {len(mlflow_tag_overrides)} orchestrator provenance "
                    f"tag(s): {list(mlflow_tag_overrides.keys())}",
                    extra={"stage": "run_training"},
                )

            # Emit the human-readable pre-training header to the log
            _log_run_header(
                logger, args, run_name, experiment_group,
                cfg, pipeline, dataset, model_summary,
            )

            # ── Delegate full training lifecycle to train_one_run ──────────
            # train_one_run() owns: per-epoch load_split() loop, artefact
            # generation, per-epoch mlflow.log_metrics(), SavedModel export.
            # It MUST be called inside an active mlflow.start_run() context.
            result = train_one_run(
                cfg=cfg,
                run_name=run_name,
                experiment_group=experiment_group,
                model=model,
            )

    except RuntimeError as exc:
        # Known failure mode from train_one_run() — NaN loss, empty dataset, etc.
        logger.error(
            f"Training failed (RuntimeError): {exc}",
            extra={"stage": "run_training"},
        )
        return EXIT_TRAINING_FAILURE, None
    except KeyboardInterrupt:
        logger.warning(
            "Training interrupted by user (KeyboardInterrupt). "
            "Partial MLflow run may exist.",
            extra={"stage": "run_training"},
        )
        return EXIT_TRAINING_FAILURE, None
    except Exception as exc:
        # Unexpected — log full traceback once only (not duplicated).
        logger.error(
            f"Unexpected error during training: {type(exc).__name__}: {exc}\n"
            f"{traceback.format_exc()}",
            extra={"stage": "run_training"},
        )
        return EXIT_UNEXPECTED_ERROR, None

    # ── Step 11: Validate result contract ─────────────────────────────────
    # Done outside the MLflow context (run is already closed) so a contract
    # violation does not corrupt the run's status to FAILED.
    if result is not None:
        try:
            _validate_result_contract(result, run_name)
        except RuntimeError as exc:
            logger.error(str(exc), extra={"stage": "run_training"})
            return EXIT_UNEXPECTED_ERROR, None

    # ── Step 12: Post-run summary ──────────────────────────────────────────
    elapsed = time.time() - t_script_start
    if result is not None:
        _log_run_footer(logger, result, elapsed)

    return EXIT_SUCCESS, result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    """
    Parse CLI arguments and execute the training run.

    Parameters
    ----------
    argv : list[str] | None
        Override sys.argv (used in unit tests). None = use sys.argv[1:].

    Returns
    -------
    int  Exit code (0 = success, 1 = config error, 2 = training failure,
         3 = unexpected error).
    """
    parser = _build_parser()
    args   = parser.parse_args(argv)

    # ── Phase 1: Pre-import argument validation ────────────────────────────
    # Catch typos before TF/MediaPipe load (which takes 5–15 seconds).

    if args.experiment is not None and args.experiment not in _KNOWN_EXPERIMENTS:
        # Advisory only — users may have custom experiment configs.
        print(
            f"[run_training] WARNING: --experiment '{args.experiment}' is not in "
            f"the known set {sorted(_KNOWN_EXPERIMENTS)}. "
            f"Ensure configs/experiment/{args.experiment}.yaml exists.",
            file=sys.stderr,
        )

    # Parse --override flags with ast.literal_eval support
    overrides = _parse_overrides(args.override)

    # Generate or validate run name
    run_name = (
        _generate_run_name(args.model, args.data, args.augmentation)
        if args.run_name is None
        else args.run_name
    )
    run_name         = _validate_run_name(run_name)
    experiment_group = args.group

    # Artefact directory collision check — must happen pre-import
    # (before TF loads) to fail fast on user error.
    _check_artefact_collision(run_name, force=args.force)

    # ── Phase 2: Configure structured logging ─────────────────────────────
    # configure_logging() is called here in main() — the correct entry point.
    # It is NOT imported or called inside _execute_run(), which only calls
    # get_logger(__name__) to obtain an already-configured logger.
    try:
        from src.utils.logger import configure_logging
        configure_logging(
            level="INFO",
            log_dir="logs",
            run_name=run_name,
        )
    except Exception as exc:
        print(
            f"[run_training] WARNING: configure_logging() failed: {exc}. "
            "Proceeding with default Python logging.",
            file=sys.stderr,
        )

    # ── Phase 3: Execute run ───────────────────────────────────────────────
    exit_code, _ = _execute_run(args, run_name, experiment_group, overrides)
    return exit_code


# ---------------------------------------------------------------------------
# Public API for run_all_experiments.py
# ---------------------------------------------------------------------------

def run_single_experiment(
    model:               str,
    data:                str,
    augmentation:        str,
    run_name:            str,
    experiment_group:    str,
    experiment:          Optional[str]      = None,
    overrides:           Optional[Dict[str, Any]] = None,
    splits_dir:          Optional[str]      = None,
    landmarks_dir:       Optional[str]      = None,
    mlflow_tracking_uri: Optional[str]      = None,
    force:               bool               = False,
) -> Dict[str, Any]:
    """
    Programmatic entry point for run_all_experiments.py.

    Bypasses argparse so the orchestrator can call this function directly
    without spawning subprocesses, sharing the same Python process and TF
    graph context across all runs.

    Parameters
    ----------
    model, data, augmentation : str  — config names (must be in valid sets)
    run_name           : str         — unique run identifier
    experiment_group   : str         — MLflow tag group
    experiment         : str | None  — optional experiment config overlay
    overrides          : dict | None — pre-parsed config overrides
    splits_dir         : str | None  — override for split CSV directory
    landmarks_dir      : str | None  — override for landmarks directory
    mlflow_tracking_uri: str | None  — override for MLflow tracking URI
    force              : bool        — allow artefact directory overwrite

    Returns
    -------
    dict  Training result from train_one_run() — all _REQUIRED_RESULT_KEYS present.

    Raises
    ------
    ValueError    If any config name is not in its valid set.
    RuntimeError  If training fails or result contract validation fails.
    """
    if model not in _VALID_MODELS:
        raise ValueError(
            f"model='{model}' is invalid. Must be one of: {sorted(_VALID_MODELS)}."
        )
    if data not in _VALID_DATA:
        raise ValueError(
            f"data='{data}' is invalid. Must be one of: {sorted(_VALID_DATA)}."
        )
    if augmentation not in _VALID_AUGMENTATION:
        raise ValueError(
            f"augmentation='{augmentation}' is invalid. "
            f"Must be one of: {sorted(_VALID_AUGMENTATION)}."
        )
    if experiment_group not in _VALID_EXPERIMENT_GROUPS:
        raise ValueError(
            f"experiment_group='{experiment_group}' is invalid. "
            f"Must be one of: {sorted(_VALID_EXPERIMENT_GROUPS)}."
        )

    run_name = _validate_run_name(run_name)
    _check_artefact_collision(run_name, force=force)

    # Build a synthetic argparse.Namespace matching _execute_run()'s expectations
    ns = argparse.Namespace(
        model=model,
        data=data,
        augmentation=augmentation,
        experiment=experiment,
        run_name=run_name,
        group=experiment_group,
        override=[],                   # overrides already parsed as dict
        dry_run=False,
        force=force,
        mlflow_tracking_uri=mlflow_tracking_uri,
        splits_dir=splits_dir,
        landmarks_dir=landmarks_dir,
    )

    exit_code, result = _execute_run(
        ns,
        run_name,
        experiment_group,
        overrides or {},
    )

    if exit_code != EXIT_SUCCESS or result is None:
        raise RuntimeError(
            f"run_single_experiment failed for run_name='{run_name}' "
            f"with exit_code={exit_code}. Check the training log for details."
        )

    return result


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(main())