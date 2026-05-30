"""
src/utils/reproducibility.py
==============================
Reproducibility utilities for the WLASL gesture recognition pipeline.

Ensures that every experiment run is deterministic and that the exact
software environment that produced each result is recorded permanently
in MLflow alongside the model artefacts.

Design principles:
  - Seeds are set atomically across all randomness sources before any
    other operation in a training run.
  - Environment metadata is logged to MLflow so future engineers can
    verify whether a result can be reproduced on their hardware.
  - A run manifest is written to disk at the start of every experiment,
    independent of MLflow, so reproducibility information survives even
    if the MLflow store is lost.
  - GPU determinism is enabled when a GPU is available, with a clear
    warning that this may reduce training speed.

Usage (at the top of any pipeline entry point):
    from src.utils.reproducibility import set_seeds, log_environment, save_run_manifest
    from src.utils.logger import get_logger

    logger = get_logger(__name__)
    set_seeds(seed=42, logger=logger)
    log_environment(run_name="bilstm_seq30_aug")
    save_run_manifest(config=cfg, output_dir="artifacts/experiments/bilstm_seq30_aug")
"""

import os
import json
import random
import platform
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.utils.logger import get_logger

# Lazy imports — only required during experiment runs, not at import time
# This prevents import errors when running tests without a full TF install.
def _import_tf():
    import tensorflow as tf
    return tf

def _import_mediapipe():
    import mediapipe as mp
    return mp

def _import_mlflow():
    import mlflow
    return mlflow


# ---------------------------------------------------------------------------
# Seed management
# ---------------------------------------------------------------------------

def set_seeds(seed: int = 42, logger=None) -> None:
    """
    Set all randomness sources to a fixed seed for full reproducibility.

    Sources covered:
      - Python built-in random module
      - NumPy random number generator
      - TensorFlow global random seed (covers Keras layer initialisation,
        dropout masks, and data shuffling within tf.data pipelines)
      - PYTHONHASHSEED environment variable (affects dict ordering and set
        iteration in Python 3.3+)
      - TF GPU determinism (CuDNN ops become deterministic; may slow training)

    Parameters
    ----------
    seed : int
        The seed value. Must be reproducible across runs — store in config.yaml.
        Default: 42.
    logger : StructuredAdapter, optional
        Logger to use. If None, a module-level logger is created.

    Notes
    -----
    TensorFlow's GPU ops (CuDNN) are non-deterministic by default for performance.
    Setting TF_DETERMINISTIC_OPS=1 forces determinism at the cost of ~10–20%
    slower training on GPU. On CPU (this project's primary target), there is no
    speed penalty.

    Even with all seeds set, results may vary across:
      - Different hardware (CPU vs GPU floating-point rounding differences)
      - Different TF/CUDA/cuDNN versions
      - Different OS thread scheduling

    This is documented in the run manifest so future engineers understand the limits.
    """
    if logger is None:
        logger = get_logger(__name__)

    # 1. Python stdlib random
    random.seed(seed)

    # 2. NumPy
    np.random.seed(seed)

    # 3. PYTHONHASHSEED — must be set before the Python interpreter starts
    #    for full effect, but setting here still affects most hash-dependent ops.
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 4. TensorFlow
    try:
        tf = _import_tf()
        tf.random.set_seed(seed)

        # 5. GPU determinism — forces CuDNN ops to be deterministic
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            logger.warning(
                "GPU detected: TF_DETERMINISTIC_OPS=1 enabled. "
                "Expect ~10-20% slower GPU training for full reproducibility.",
                extra={"stage": "reproducibility"},
            )
        else:
            logger.debug(
                "CPU-only mode: GPU determinism flags set but have no effect.",
                extra={"stage": "reproducibility"},
            )

    except ImportError:
        logger.warning(
            "TensorFlow not available — TF seeds not set.",
            extra={"stage": "reproducibility"},
        )

    logger.info(
        f"All randomness sources seeded with seed={seed}",
        extra={"stage": "reproducibility"},
    )


# ---------------------------------------------------------------------------
# Environment metadata collection
# ---------------------------------------------------------------------------

def collect_environment_metadata() -> dict[str, Any]:
    """
    Collect a comprehensive snapshot of the current software environment.

    Returns a dictionary suitable for logging to MLflow, writing to JSON,
    or embedding in a run manifest. Designed to be self-contained: no
    external services are called.

    Returns
    -------
    dict
        Keys: tf_version, keras_version, mediapipe_version, numpy_version,
        sklearn_version, pandas_version, python_version, python_implementation,
        os_system, os_release, os_machine, cuda_available, gpu_devices,
        cpu_count, timestamp_utc, git_commit_hash, git_branch.
    """
    meta: dict[str, Any] = {}

    # --- Python ---
    meta["python_version"] = platform.python_version()
    meta["python_implementation"] = platform.python_implementation()  # CPython / PyPy

    # --- OS ---
    meta["os_system"] = platform.system()       # Linux / Darwin / Windows
    meta["os_release"] = platform.release()
    meta["os_machine"] = platform.machine()     # x86_64 / arm64

    # --- CPU ---
    meta["cpu_count"] = os.cpu_count()

    # --- TensorFlow ---
    try:
        tf = _import_tf()
        meta["tf_version"] = tf.__version__
        meta["keras_version"] = tf.keras.__version__
        gpus = tf.config.list_physical_devices("GPU")
        meta["cuda_available"] = len(gpus) > 0
        meta["gpu_devices"] = [gpu.name for gpu in gpus] if gpus else []
    except ImportError:
        meta["tf_version"] = "not installed"
        meta["keras_version"] = "not installed"
        meta["cuda_available"] = False
        meta["gpu_devices"] = []

    # --- MediaPipe ---
    try:
        mp = _import_mediapipe()
        meta["mediapipe_version"] = mp.__version__
    except ImportError:
        meta["mediapipe_version"] = "not installed"

    # --- NumPy ---
    meta["numpy_version"] = np.__version__

    # --- Scikit-learn ---
    try:
        import sklearn
        meta["sklearn_version"] = sklearn.__version__
    except ImportError:
        meta["sklearn_version"] = "not installed"

    # --- pandas ---
    try:
        import pandas as pd
        meta["pandas_version"] = pd.__version__
    except ImportError:
        meta["pandas_version"] = "not installed"

    # --- MLflow ---
    try:
        mlflow = _import_mlflow()
        meta["mlflow_version"] = mlflow.__version__
    except ImportError:
        meta["mlflow_version"] = "not installed"

    # --- Git ---
    meta["git_commit_hash"] = _get_git_commit_hash()
    meta["git_branch"] = _get_git_branch()

    # --- Timestamp ---
    meta["timestamp_utc"] = datetime.now(timezone.utc).isoformat()

    return meta


def _get_git_commit_hash() -> str:
    """Return the current git commit hash, or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _get_git_branch() -> str:
    """Return the current git branch name, or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


# ---------------------------------------------------------------------------
# MLflow environment logging
# ---------------------------------------------------------------------------

def log_environment(
    run_name: Optional[str] = None,
    extra_tags: Optional[dict[str, str]] = None,
    logger=None,
) -> dict[str, Any]:
    """
    Log the full environment snapshot to the active MLflow run.

    This function must be called INSIDE an active mlflow.start_run() context.
    It logs all environment metadata as MLflow params and sets descriptive tags.

    Parameters
    ----------
    run_name : str, optional
        Human-readable name for this run (used as an MLflow tag).
    extra_tags : dict, optional
        Additional MLflow tags to set (e.g. experiment_group, signer_split).
    logger : StructuredAdapter, optional
        Logger instance. Created if not provided.

    Returns
    -------
    dict
        The full environment metadata dictionary (also logged to MLflow).

    Example
    -------
    with mlflow.start_run(run_name="bilstm_seq30_aug"):
        set_seeds(42)
        env = log_environment(
            run_name="bilstm_seq30_aug",
            extra_tags={"experiment_group": "architecture_comparison"}
        )
    """
    if logger is None:
        logger = get_logger(__name__)

    mlflow = _import_mlflow()
    env = collect_environment_metadata()

    # Log as MLflow params — these are searchable in the MLflow UI
    mlflow.log_params({
        "env.tf_version":         env.get("tf_version", "unknown"),
        "env.keras_version":      env.get("keras_version", "unknown"),
        "env.mediapipe_version":  env.get("mediapipe_version", "unknown"),
        "env.numpy_version":      env.get("numpy_version", "unknown"),
        "env.python_version":     env.get("python_version", "unknown"),
        "env.os_system":          env.get("os_system", "unknown"),
        "env.os_machine":         env.get("os_machine", "unknown"),
        "env.cuda_available":     str(env.get("cuda_available", False)),
        "env.cpu_count":          str(env.get("cpu_count", "unknown")),
        "env.git_commit":         env.get("git_commit_hash", "unknown")[:8],  # short hash
        "env.git_branch":         env.get("git_branch", "unknown"),
    })

    # Set MLflow tags — more descriptive, not searchable as params
    tags = {
        "mlflow.runName": run_name or "unnamed_run",
        "project": "wlasl-gesture-recognition",
        "timestamp_utc": env["timestamp_utc"],
    }
    if extra_tags:
        tags.update(extra_tags)
    mlflow.set_tags(tags)

    # GPU devices as a tag (can be a list, not suitable as a param)
    gpu_info = ", ".join(env.get("gpu_devices", [])) or "none"
    mlflow.set_tag("env.gpu_devices", gpu_info)

    logger.info(
        f"Environment metadata logged to MLflow | "
        f"TF={env.get('tf_version')} | "
        f"MediaPipe={env.get('mediapipe_version')} | "
        f"GPU={'yes' if env.get('cuda_available') else 'no'}",
        extra={"stage": "reproducibility"},
    )

    return env


# ---------------------------------------------------------------------------
# Run manifest — disk-level reproducibility record
# ---------------------------------------------------------------------------

def save_run_manifest(
    config: Any,
    output_dir: str,
    seed: int = 42,
    logger=None,
) -> Path:
    """
    Write a JSON run manifest to disk before training begins.

    The manifest is a complete record of everything needed to reproduce this
    experiment: the full config, all software versions, git state, seed, and
    timestamps. It is independent of MLflow — if MLflow is unavailable or its
    store is deleted, the manifest is the fallback.

    Parameters
    ----------
    config : OmegaConf DictConfig or dict
        The full experiment configuration (will be serialised to JSON).
    output_dir : str
        Directory where the manifest is written. Created if not exists.
        Recommended: "artifacts/experiments/<run_name>/"
    seed : int
        The seed value used for this run.
    logger : StructuredAdapter, optional
        Logger instance.

    Returns
    -------
    Path
        Absolute path to the written manifest file.

    File written
    ------------
    <output_dir>/run_manifest.json
    """
    if logger is None:
        logger = get_logger(__name__)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    manifest_path = out_path / "run_manifest.json"

    # Serialise config — handle OmegaConf DictConfig transparently
    try:
        from omegaconf import OmegaConf
        config_dict = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    except (ImportError, Exception):
        # Fallback: try treating config as a plain dict
        try:
            config_dict = dict(config)
        except Exception:
            config_dict = {"error": "config not serialisable"}

    manifest = {
        "manifest_version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "config": config_dict,
        "environment": collect_environment_metadata(),
        "reproduction_instructions": (
            "To reproduce: "
            "(1) checkout git commit shown in environment.git_commit_hash, "
            "(2) install requirements.txt pinned versions, "
            "(3) run with the config block above using the seed shown."
        ),
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    logger.info(
        f"Run manifest written to {manifest_path.resolve()}",
        extra={"stage": "reproducibility"},
    )

    return manifest_path


# ---------------------------------------------------------------------------
# Model hash — verify model file integrity
# ---------------------------------------------------------------------------

def compute_model_hash(model_path: str, algorithm: str = "md5") -> str:
    """
    Compute a hash of a model file or SavedModel directory for integrity checking.

    Used to verify that a loaded model file matches the one logged during training.
    The hash is stored in the run manifest and can be logged to MLflow.

    Parameters
    ----------
    model_path : str
        Path to a .tflite file or a SavedModel directory.
    algorithm : str
        Hash algorithm. One of 'md5', 'sha256'. Default: 'md5' (faster).

    Returns
    -------
    str
        Hex digest of the model file(s).

    Example
    -------
    >>> h = compute_model_hash("models/gesture_bilstm_v1.tflite")
    >>> print(h)  # e.g. "a3f2c1d4..."
    """
    path = Path(model_path)
    h = hashlib.new(algorithm)

    if path.is_file():
        # Single file (e.g. .tflite)
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    elif path.is_dir():
        # SavedModel directory — hash all files in sorted order for consistency
        for file_path in sorted(path.rglob("*")):
            if file_path.is_file():
                h.update(str(file_path.relative_to(path)).encode())
                with open(file_path, "rb") as f:
                    for chunk in iter(lambda: f.read(65536), b""):
                        h.update(chunk)
    else:
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    return h.hexdigest()


# ---------------------------------------------------------------------------
# Convenience: full setup in one call
# ---------------------------------------------------------------------------

def setup_experiment(
    config: Any,
    run_name: str,
    output_dir: str,
    extra_mlflow_tags: Optional[dict[str, str]] = None,
    logger=None,
) -> dict[str, Any]:
    """
    Convenience function that performs all reproducibility setup in one call.

    Does, in order:
      1. set_seeds(config.seed)
      2. save_run_manifest(config, output_dir)
      3. log_environment(run_name, extra_mlflow_tags)  [requires active MLflow run]

    Parameters
    ----------
    config : OmegaConf DictConfig
        Full experiment config. Must have a `seed` attribute.
    run_name : str
        MLflow run name and manifest directory name.
    output_dir : str
        Directory for the run manifest.
    extra_mlflow_tags : dict, optional
        Additional MLflow tags.
    logger : StructuredAdapter, optional
        Logger instance.

    Returns
    -------
    dict
        Environment metadata dictionary.

    Example
    -------
    with mlflow.start_run(run_name=run_name):
        env = setup_experiment(
            config=cfg,
            run_name=run_name,
            output_dir=f"artifacts/experiments/{run_name}",
            extra_mlflow_tags={"experiment_group": "architecture_comparison"},
        )
    """
    if logger is None:
        logger = get_logger(__name__)

    seed = getattr(config, "seed", 42)
    set_seeds(seed=seed, logger=logger)
    save_run_manifest(config=config, output_dir=output_dir, seed=seed, logger=logger)
    env = log_environment(run_name=run_name, extra_tags=extra_mlflow_tags, logger=logger)

    return env