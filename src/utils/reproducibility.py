"""
src/utils/reproducibility.py
=============================
Full reproducibility stack for the WLASL gesture recognition pipeline.

Responsibilities:
  1. Set all random seeds (Python, NumPy, TensorFlow) before any training operation.
  2. Set TensorFlow determinism environment variables BEFORE importing TensorFlow —
     many TF env vars are read at import time and have no effect if set afterwards.
  3. Collect comprehensive environment metadata (library versions, hardware, git state,
     requirements hash) so that every experiment can be exactly reproduced.
  4. Log all metadata to the active MLflow run.
  5. Save a run manifest JSON to disk as a reproducibility artefact.
  6. Compute cryptographic hashes of model files for integrity verification.

Critical ordering:
    # CORRECT — env vars set before TF is imported anywhere
    from src.utils.reproducibility import set_seeds
    set_seeds(42)

    import tensorflow as tf   # TF reads env vars at this point

    # WRONG — TF already imported by another module; env vars too late
    import tensorflow as tf
    set_seeds(42)

Usage in pipeline entry points:
    from src.utils.reproducibility import setup_experiment

    with mlflow.start_run(run_name="bilstm_seq30"):
        manifest_path = setup_experiment(
            config=cfg,
            run_name="bilstm_seq30",
            output_dir="artifacts/experiments/bilstm_seq30",
        )
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)
_ENV_METADATA_CACHE: Optional[dict[str, Any]] = None


# =============================================================================
# Step 1 — Set determinism environment variables
# This function must be called, and TF imported, in the correct order.
# =============================================================================

def _set_determinism_env_vars(seed: int) -> None:
    """
    Set environment variables that control TensorFlow determinism.

    MUST be called before tensorflow is imported anywhere in the process.
    These variables are read by TF at import time; setting them after import
    has no effect.

    Variables set:
        PYTHONHASHSEED          — controls Python's hash randomisation
        TF_DETERMINISTIC_OPS    — enables deterministic GPU ops
        TF_CUDNN_DETERMINISTIC  — forces deterministic cuDNN algorithms
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    logger.debug(
        f"Determinism env vars set | "
        f"PYTHONHASHSEED={seed} | "
        f"TF_DETERMINISTIC_OPS=1 | "
        f"TF_CUDNN_DETERMINISTIC=1"
    )


def set_seeds(seed: int = 42) -> None:
    """
    Set all random seeds for full reproducibility.

    Sets: Python random, NumPy random, TensorFlow random, PYTHONHASHSEED,
    TF_DETERMINISTIC_OPS, TF_CUDNN_DETERMINISTIC.

    IMPORTANT: This function should be called at the very start of any pipeline
    entry point, before importing tensorflow or any library that uses random
    state. If TensorFlow has already been imported when this is called, the
    environment variables will have no effect on TF's GPU operation selection —
    a warning is logged in that case.

    Parameters
    ----------
    seed : int
        The global random seed. Default 42.

    Example
    -------
    # In a pipeline entry point (e.g. pipelines/run_training.py):
    from src.utils.reproducibility import set_seeds
    set_seeds(42)

    import tensorflow as tf   # Import TF AFTER set_seeds
    """
    # Check if TF is already imported — if so, env vars are too late for GPU determinism
    tf_already_imported = "tensorflow" in sys.modules

    # Set env vars first — even if TF is imported, PYTHONHASHSEED still helps
    _set_determinism_env_vars(seed)

    # Python built-in random
    random.seed(seed)

    # NumPy
    try:
        import numpy as np
        np.random.seed(seed)
        logger.debug(f"NumPy seed set: {seed}")
    except ImportError:
        logger.warning("NumPy not available — skipping NumPy seed.")

    # TensorFlow
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)

        try:
            tf.config.experimental.enable_op_determinism()
            logger.debug("TensorFlow op determinism enabled.")
        except Exception:  # noqa: BLE001
            logger.debug(
                "TensorFlow op determinism API unavailable on this TF version."
            )

        logger.debug(f"TensorFlow seed set: {seed}")

        if tf_already_imported:
            logger.warning(
                "TensorFlow was already imported before set_seeds() was called. "
                "TF_DETERMINISTIC_OPS and TF_CUDNN_DETERMINISTIC environment variables "
                "may have no effect on GPU operation selection for this run. "
                "For full GPU determinism, call set_seeds() before any TF import."
            )
    except ImportError:
        logger.warning("TensorFlow not available — skipping TF seed.")

    logger.info(f"All random seeds set | seed={seed}")


# =============================================================================
# Step 2 — Environment metadata collection
# =============================================================================

def _get_git_info() -> dict[str, str]:
    """
    Collect git repository state.

    Returns commit hash, branch name, and whether the working tree is dirty
    (has uncommitted changes). A commit hash alone does not guarantee
    reproducibility if there are uncommitted changes — dirty=True is a warning.
    """
    info: dict[str, str] = {
        "git_commit_hash": "unavailable",
        "git_branch": "unavailable",
        "git_dirty": "unavailable",
    }

    def _run(cmd: list[str]) -> str:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).parents[2],
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    try:
        commit = _run(["git", "rev-parse", "HEAD"])
        if commit:
            info["git_commit_hash"] = commit

        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        if branch:
            info["git_branch"] = branch

        remote = _run(
            ["git", "config", "--get", "remote.origin.url"]
        )
        if remote:
            info["git_remote"] = remote

        # git status --porcelain outputs nothing if the tree is clean
        status = _run(["git", "status", "--porcelain"])
        info["git_dirty"] = "true" if status else "false"

        if info["git_dirty"] == "true":
            logger.warning(
                "Git working tree is dirty (uncommitted changes present). "
                "The commit hash alone does not guarantee full reproducibility. "
                "Commit your changes before running experiments for exact reproduction."
            )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        logger.debug("Git not available or not in a git repository.")

    return info


def _get_cuda_info() -> dict[str, str]:
    """
    Collect CUDA and cuDNN version information if a GPU is available.
    """
    info: dict[str, str] = {
        "git_commit_hash": "unavailable",
        "git_branch": "unavailable",
        "git_dirty": "unavailable",
        "git_remote": "unavailable",
    }

    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            info["cuda_available"] = "true"
            info["gpu_devices"] = ", ".join(g.name for g in gpus)

            # TF exposes CUDA/cuDNN versions in build info
            build_info = tf.sysconfig.get_build_info()
            info["cuda_version"] = str(build_info.get("cuda_version", "unavailable"))
            info["cudnn_version"] = str(build_info.get("cudnn_version", "unavailable"))
    except Exception:  # noqa: BLE001
        pass

    return info


def _get_requirements_hash() -> str:
    """
    Compute SHA256 of requirements.txt.

    Library version numbers in environment metadata can be spoofed or
    ambiguous (e.g. local editable installs). The requirements.txt hash
    provides a stronger guarantee that the exact dependency set is recorded.
    Returns "unavailable" if requirements.txt is not found.
    """
    for candidate in [
        Path("requirements.txt"),
        Path(__file__).parents[2] / "requirements.txt",
    ]:
        if candidate.exists():
            content = candidate.read_bytes()
            return hashlib.sha256(content).hexdigest()
    logger.debug("requirements.txt not found — skipping requirements hash.")
    return "unavailable"


def _get_pip_freeze() -> str:
    """
    Capture a pip freeze snapshot.

    Returns the full list of installed packages as a newline-separated string,
    or "unavailable" if pip is not accessible. This supplements version numbers
    in metadata with the actual installed state, including local editable installs.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    logger.debug("pip freeze failed — skipping pip snapshot.")
    return "unavailable"


def _get_library_versions() -> dict[str, str]:
    """Collect versions of all key pipeline libraries."""
    versions: dict[str, str] = {}

    libraries = [
        ("tensorflow", "tensorflow"),
        ("keras", "keras"),
        ("mediapipe", "mediapipe"),
        ("numpy", "numpy"),
        ("sklearn", "sklearn"),
        ("pandas", "pandas"),
        ("mlflow", "mlflow"),
        ("omegaconf", "omegaconf"),
        ("pydantic", "pydantic"),
        ("shap", "shap"),
        ("cv2", "cv2"),
    ]

    for import_name, display_name in libraries:
        try:
            mod = __import__(import_name)
            versions[f"{display_name}_version"] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[f"{display_name}_version"] = "not installed"

    return versions


def collect_environment_metadata(include_pip_freeze: bool = True) -> dict[str, Any]:
    """
    Collect comprehensive environment metadata for reproducibility records.

    Captures: library versions, Python version, OS, CPU, GPU/CUDA info,
    git commit and branch, git dirty status, requirements.txt hash,
    and optionally a full pip freeze snapshot.

    Parameters
    ----------
    include_pip_freeze : bool
        Whether to include a full pip freeze snapshot. Can be slow (~1s)
        on environments with many packages. Default True.

    Returns
    -------
    dict[str, Any]
        Flat or nested dict suitable for logging to MLflow or writing to JSON.
    """
    global _ENV_METADATA_CACHE

    if _ENV_METADATA_CACHE is not None:
        return _ENV_METADATA_CACHE.copy()

    metadata: dict[str, Any] = {
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "os": platform.system(),
        "os_version": platform.version(),
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
    }

    metadata.update(_get_library_versions())
    metadata.update(_get_git_info())
    metadata.update(_get_cuda_info())

    metadata["requirements_hash"] = _get_requirements_hash()

    metadata["determinism"] = {
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "tf_deterministic_ops": os.environ.get("TF_DETERMINISTIC_OPS"),
        "tf_cudnn_deterministic": os.environ.get("TF_CUDNN_DETERMINISTIC"),
    }

    if include_pip_freeze:
        metadata["pip_freeze"] = _get_pip_freeze()

    _ENV_METADATA_CACHE = metadata.copy()

    return metadata


# =============================================================================
# Step 3 — MLflow integration
# =============================================================================

def log_environment(
    run_name: str = "",
    extra_tags: Optional[dict[str, str]] = None,
    include_pip_freeze: bool = False,
) -> None:
    """
    Log environment metadata to the active MLflow run.

    Must be called inside an mlflow.start_run() context. If no MLflow run
    is active, logs a warning and returns without error.

    Parameters
    ----------
    run_name : str
        Human-readable run name, added as an MLflow tag.
    extra_tags : dict[str, str], optional
        Additional MLflow tags to set (e.g. experiment_group, model_type).
    include_pip_freeze : bool
        Whether to log the full pip freeze as an MLflow param. Disabled by
        default because it produces a very long param value. The pip freeze
        is always saved to the run manifest on disk.
    """
    try:
        import mlflow  # type: ignore[import]
    except ImportError:
        logger.warning("MLflow not installed — skipping environment logging to MLflow.")
        return

    active = mlflow.active_run()
    if active is None:
        logger.warning(
            "log_environment() called but no MLflow run is active. "
            "Call this inside an mlflow.start_run() context."
        )
        return

    metadata = collect_environment_metadata(include_pip_freeze=include_pip_freeze)

    # Log as MLflow params — exclude pip_freeze (too long for MLflow params)
    loggable_params = {
        k: str(v)[:500]  # MLflow param value limit is 500 chars
        for k, v in metadata.items()
        if k != "pip_freeze" and v != "unavailable"
    }
    mlflow.log_params(loggable_params)

    # Set tags
    tags: dict[str, str] = {"run_name": run_name}
    if extra_tags:
        tags.update(extra_tags)
    mlflow.set_tags(tags)

    # Log git dirty as a prominent tag for visibility in the MLflow UI
    if metadata.get("git_dirty") == "true":
        mlflow.set_tag("git_dirty", "true")
        mlflow.set_tag("reproducibility_warning", "uncommitted_changes_present")

    logger.info(
        f"Environment metadata logged to MLflow run {active.info.run_id[:8]}",
        extra={"stage": "reproducibility"},
    )


# =============================================================================
# Step 4 — Run manifest
# =============================================================================

def save_run_manifest(
    config: Any,
    output_dir: str,
    seed: int = 42,
    extra_metadata: Optional[dict[str, Any]] = None,
    include_pip_freeze: bool = True,
) -> Path:
    """
    Save a complete run manifest JSON to disk.

    The manifest contains the full config, environment snapshot, seed, and
    reproduction instructions. It is the disk-based reproducibility record,
    independent of MLflow — useful when MLflow is unavailable or when you
    want to reproduce a run on a different machine.

    Parameters
    ----------
    config : ExperimentConfig | dict
        The experiment configuration. Accepts both Pydantic v2 models
        (with model_dump()) and plain dicts.
    output_dir : str
        Directory where run_manifest.json is written. Created if needed.
    seed : int
        The random seed used for this run.
    extra_metadata : dict, optional
        Additional key-value pairs to include in the manifest.

    Returns
    -------
    Path
        Absolute path to the written manifest file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_file = output_path / "run_manifest.json"

    # Serialise config — support both Pydantic v2 models and plain dicts
    if hasattr(config, "model_dump"):
        config_dict = config.model_dump()
    elif hasattr(config, "dict"):
        # Pydantic v1 fallback
        config_dict = config.dict()
    elif isinstance(config, dict):
        config_dict = config
    else:
        logger.warning(
            f"save_run_manifest: config type {type(config).__name__} is not a "
            f"Pydantic model or dict. Attempting str() serialisation."
        )
        config_dict = {"raw": str(config)}

    env_metadata = collect_environment_metadata(
    include_pip_freeze=include_pip_freeze
    )

    manifest: dict[str, Any] = {
        "manifest_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "config": config_dict,
        "environment": env_metadata,
        "reproduction_instructions": {
            "step_1": "Clone the repository at the git_commit_hash recorded in environment.",
            "step_2": "Install dependencies: pip install -r requirements.txt",
            "step_3": (
                "Verify requirements hash matches environment.requirements_hash: "
                "sha256sum requirements.txt"
            ),
            "step_4": "Run: python pipelines/run_training.py with the same config.",
            "note": (
                "If environment.git_dirty=true, exact reproduction is not guaranteed "
                "because uncommitted changes were present when this run was executed."
            ),
        },
    }

    if extra_metadata:
        manifest["extra"] = extra_metadata

    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    logger.info(
        f"Run manifest saved: {manifest_file.resolve()}",
        extra={"stage": "reproducibility"},
    )
    return manifest_file


# =============================================================================
# Step 5 — Model integrity
# =============================================================================

def compute_model_hash(
    model_path: str | Path,
    algorithm: str = "sha256",
) -> str:
    """
    Compute a cryptographic hash of a model file or SavedModel directory.

    Use this to verify that a TFLite file or SavedModel has not been modified
    since training. Store the hash in gesture_model_metadata.json alongside
    the model.

    Parameters
    ----------
    model_path : str | Path
        Path to a .tflite file or a SavedModel directory.
    algorithm : str
        Hash algorithm. Must be one of "md5" or "sha256". Default "sha256".

    Returns
    -------
    str
        Hex digest of the hash.

    Raises
    ------
    ValueError
        If algorithm is not "md5" or "sha256", or if the path does not exist.
    FileNotFoundError
        If model_path does not exist.
    """
    allowed_algorithms = {"md5", "sha256"}
    if algorithm not in allowed_algorithms:
        raise ValueError(
            f"algorithm must be one of {allowed_algorithms}, got '{algorithm}'. "
            "MD5 and SHA256 are the only supported algorithms."
        )

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model path does not exist: {path.resolve()}")

    hasher = hashlib.new(algorithm)

    if path.is_file():
        # Single file (e.g. .tflite)
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
    elif path.is_dir():
        # SavedModel directory — hash all files in deterministic (sorted) order
        all_files = sorted(p for p in path.rglob("*") if p.is_file())
        if not all_files:
            raise ValueError(f"SavedModel directory is empty: {path.resolve()}")
        for file_path in all_files:
            # Include relative path in hash so directory structure changes are detected
            hasher.update(str(file_path.relative_to(path)).encode("utf-8"))
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    hasher.update(chunk)
    else:
        raise ValueError(f"model_path is neither a file nor a directory: {path}")

    digest = hasher.hexdigest()
    logger.info(
        f"Model hash computed | algorithm={algorithm} | path={path} | hash={digest[:16]}...",
        extra={"stage": "export"},
    )
    return digest


# =============================================================================
# Convenience: setup_experiment — call once per run entry point
# =============================================================================

def setup_experiment(
    config: Any,
    run_name: str,
    output_dir: str,
    extra_mlflow_tags: Optional[dict[str, str]] = None,
    include_pip_freeze_in_manifest: bool = True,
) -> Path:
    """
    Full experiment setup in one call.

    Executes in order:
        1. set_seeds(config.seed)
        2. save_run_manifest(config, output_dir)
        3. log_environment(run_name, extra_mlflow_tags)  [if MLflow run is active]

    Parameters
    ----------
    config : ExperimentConfig
        The validated experiment config.
    run_name : str
        Human-readable name for this run.
    output_dir : str
        Directory for the run manifest and per-run artefacts.
    extra_mlflow_tags : dict[str, str], optional
        Extra tags to set in the active MLflow run.
    include_pip_freeze_in_manifest : bool
        Whether to capture a pip freeze snapshot in the manifest. Default True.

    Returns
    -------
    Path
        Path to the saved run manifest.

    Example
    -------
    with mlflow.start_run(run_name="bilstm_seq30_aug"):
        manifest = setup_experiment(
            config=cfg,
            run_name="bilstm_seq30_aug",
            output_dir="artifacts/experiments/bilstm_seq30_aug",
            extra_mlflow_tags={
                "experiment_group": "architecture_comparison",
                "model_type": "bilstm",
            },
        )
    """
    if isinstance(config, dict):
        seed = config.get("seed", 42)
    else:
        seed = getattr(config, "seed", 42)
    set_seeds(seed)

    manifest_path = save_run_manifest(
        config=config,
        output_dir=output_dir,
        seed=seed,
        include_pip_freeze=include_pip_freeze_in_manifest,
    )

    log_environment(
        run_name=run_name,
        extra_tags=extra_mlflow_tags,
        include_pip_freeze=False,  # Pip freeze already in manifest; skip MLflow upload
    )

    logger.info(
        f"Experiment setup complete | run={run_name} | seed={seed} | "
        f"manifest={manifest_path}",
        extra={"stage": "reproducibility"},
    )

    return manifest_path