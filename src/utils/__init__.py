"""
src/utils/__init__.py
======================
Public API for the wlasl-gesture-recognition utils package.

Import everything you need from a single location:

    from src.utils import get_logger, configure_logging
    from src.utils import set_seeds, setup_experiment, collect_environment_metadata
    from src.utils import load_config, load_config_from_manifest, ExperimentConfig
    from src.utils import LabelMap, get_label_map, invalidate_label_map_cache

Sub-module breakdown:
    logger.py           — Structured logging, configure_logging(), get_logger()
    reproducibility.py  — Seeds, env metadata, MLflow logging, run manifests
    config.py           — OmegaConf + Pydantic config loading and validation
    label_map.py        — Versioned bidirectional label map
"""

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
from src.utils.logger import (
    configure_logging,
    get_logger,
    get_log_file_path,
    get_preprocessing_logger,
    get_training_logger,
    get_inference_logger,
    get_demo_logger,
    StructuredAdapter,
)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
from src.utils.reproducibility import (
    set_seeds,
    collect_environment_metadata,
    log_environment,
    save_run_manifest,
    setup_experiment,
    compute_model_hash,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
from src.utils.config import (
    load_config,
    load_config_from_manifest,
    write_default_configs,
    ExperimentConfig,
    DataConfig,
    ModelConfig,
    AugmentationConfig,
    TrainingConfig,
    LoggingConfig,
    MLflowConfig,
    ExportConfig,
    ModelType,
    PaddingStrategy,
    NormalisationStrategy,
    MissingFrameStrategy,
    QuantisationMode,
    EarlyStoppingMode,
)

# ---------------------------------------------------------------------------
# Label Map
# ---------------------------------------------------------------------------
from src.utils.label_map import (
    LabelMap,
    get_label_map,
    invalidate_label_map_cache,
)

__all__ = [
    # Logger
    "configure_logging",
    "get_logger",
    "get_log_file_path",
    "get_preprocessing_logger",
    "get_training_logger",
    "get_inference_logger",
    "get_demo_logger",
    "StructuredAdapter",
    # Reproducibility
    "set_seeds",
    "collect_environment_metadata",
    "log_environment",
    "save_run_manifest",
    "setup_experiment",
    "compute_model_hash",
    # Config
    "load_config",
    "load_config_from_manifest",
    "write_default_configs",
    "ExperimentConfig",
    "DataConfig",
    "ModelConfig",
    "AugmentationConfig",
    "TrainingConfig",
    "LoggingConfig",
    "MLflowConfig",
    "ExportConfig",
    "ModelType",
    "PaddingStrategy",
    "NormalisationStrategy",
    "MissingFrameStrategy",
    "QuantisationMode",
    "EarlyStoppingMode",
    # Label Map
    "LabelMap",
    "get_label_map",
    "invalidate_label_map_cache",
]