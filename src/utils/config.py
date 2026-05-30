"""
src/utils/config.py
====================
Configuration management for the WLASL gesture recognition pipeline.

Architecture:
  - All hyperparameters and paths live in YAML files under configs/.
  - OmegaConf loads and merges hierarchical YAML configs with dot-access.
  - Pydantic v2 schemas validate the merged config, providing typed access
    and immediate errors on missing or mistyped fields.
  - Nothing is ever hardcoded in any source file.

Config composition:
    python pipelines/run_training.py \\
        --model lstm \\
        --data seq30 \\
        --augmentation spatial_temporal \\
        --run-name lstm_seq30_aug

    This loads:
        configs/base.yaml                    (global defaults)
        configs/model/lstm.yaml              (model architecture)
        configs/data/seq30.yaml              (sequence length)
        configs/augmentation/spatial_temporal.yaml

    Merges them left-to-right (later keys override earlier ones), validates
    the result against the Pydantic schema, and returns a typed config object.

Usage:
    from src.utils.config import load_config, ExperimentConfig

    cfg = load_config(model="bilstm", data="seq30", augmentation="spatial_temporal")
    print(cfg.training.learning_rate)   # 0.001
    print(cfg.model.hidden_units)       # 128
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from omegaconf import OmegaConf, DictConfig
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic import ValidationError as PydanticValidationError

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Root of the configs/ directory — resolved relative to this file's location
_CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"


# =============================================================================
# Pydantic v2 schema definitions
# All fields have defaults where sensible so that partial configs work.
# Required fields (no default) will raise ValidationError immediately if absent.
# =============================================================================

class DataConfig(BaseModel):
    """Paths and dataset parameters."""

    raw_dir: str = "data/raw"
    landmark_dir: str = "data/landmarks"
    splits_dir: str = "data/splits"
    num_classes: int = Field(35, ge=2, le=2000, description="Number of sign classes")
    sequence_length: int = Field(30, ge=5, le=120, description="Fixed sequence length in frames")
    padding: str = Field("post", pattern="^(pre|post)$")
    normalisation: str = Field(
        "wrist_relative",
        pattern="^(wrist_relative|none)$",
        description="Coordinate normalisation strategy",
    )
    missing_frame_strategy: str = Field(
        "zero_fill",
        pattern="^(zero_fill|interpolate|skip)$",
        description="How to handle frames where MediaPipe fails to detect landmarks",
    )
    max_missing_frame_pct: float = Field(
        0.30,
        ge=0.0,
        le=1.0,
        description="Skip videos where more than this fraction of frames are missing",
    )

    @field_validator("raw_dir", "landmark_dir", "splits_dir", mode="before")
    @classmethod
    def coerce_to_str(cls, v: Any) -> str:
        return str(v)


class ModelConfig(BaseModel):
    """Model architecture parameters."""

    name: str = Field(..., description="One of: dense, lstm, gru, bilstm")
    hidden_units: int = Field(128, ge=16, le=1024)
    num_layers: int = Field(2, ge=1, le=6)
    dropout: float = Field(0.3, ge=0.0, le=0.8)
    bidirectional: bool = False
    dense_units: int = Field(
        64,
        ge=16,
        le=512,
        description="Units in the intermediate Dense layer before the classifier",
    )
    activation: str = Field("relu", description="Activation for Dense layers")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        allowed = {"dense", "lstm", "gru", "bilstm"}
        if v not in allowed:
            raise ValueError(f"model.name must be one of {allowed}, got '{v}'")
        return v


class AugmentationConfig(BaseModel):
    """Data augmentation strategy."""

    enabled: bool = True
    temporal_jitter: bool = False
    frame_drop_prob: float = Field(0.0, ge=0.0, le=0.5)
    spatial_flip: bool = False
    gaussian_noise_std: float = Field(0.0, ge=0.0, le=0.1)
    rotation_deg: float = Field(0.0, ge=0.0, le=30.0)
    speed_jitter: bool = False

    @model_validator(mode="after")
    def check_augmentation_consistency(self) -> AugmentationConfig:
        """Warn if augmentation is enabled but all strategies are disabled."""
        if self.enabled:
            any_active = any([
                self.temporal_jitter,
                self.frame_drop_prob > 0,
                self.spatial_flip,
                self.gaussian_noise_std > 0,
                self.rotation_deg > 0,
                self.speed_jitter,
            ])
            if not any_active:
                # Not a hard error — the user may intend this for a baseline run
                pass  # Logger not accessible in Pydantic validators; handled in load_config
        return self


class TrainingConfig(BaseModel):
    """Training loop parameters."""

    batch_size: int = Field(32, ge=1, le=512)
    epochs: int = Field(50, ge=1, le=500)
    learning_rate: float = Field(0.001, gt=0.0, le=0.1)
    early_stopping_patience: int = Field(10, ge=1, le=100)
    early_stopping_monitor: str = "val_accuracy"
    early_stopping_mode: str = Field("max", pattern="^(min|max)$")
    reduce_lr_patience: int = Field(5, ge=1, le=50)
    reduce_lr_factor: float = Field(0.5, gt=0.0, lt=1.0)
    reduce_lr_min_lr: float = Field(1e-6, gt=0.0)
    shuffle: bool = True
    class_weight_balancing: bool = Field(
        False,
        description="Compute and apply class weights to handle imbalanced classes",
    )

    @field_validator("early_stopping_monitor")
    @classmethod
    def validate_monitor(cls, v: str) -> str:
        allowed_prefixes = ("val_", "train_", "loss", "accuracy")
        if not any(v.startswith(p) for p in allowed_prefixes):
            raise ValueError(
                f"early_stopping_monitor '{v}' does not look like a valid Keras metric name"
            )
        return v


class LoggingConfig(BaseModel):
    """Logging configuration."""

    log_dir: str = "logs"
    level: str = Field("INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    file_level: str = Field("DEBUG", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")


class MLflowConfig(BaseModel):
    """MLflow experiment tracking configuration."""

    experiment_name: str = "WLASL-35-class"
    tracking_uri: str = "mlruns"
    register_best_model: bool = True
    model_registry_name: str = "gesture-lstm-production"


class ExportConfig(BaseModel):
    """TFLite export and quantisation parameters."""

    output_dir: str = "models"
    quantise: bool = True
    quantisation_mode: str = Field(
        "dynamic_range",
        pattern="^(dynamic_range|full_integer|float16)$",
        description=(
            "dynamic_range: fastest, ~4x size reduction, minimal accuracy loss. "
            "full_integer: smallest file, requires representative dataset. "
            "float16: intermediate option, good for GPU inference."
        ),
    )
    representative_dataset_size: int = Field(
        100,
        ge=10,
        le=1000,
        description="Number of samples used for full-integer quantisation calibration",
    )
    max_accuracy_delta: float = Field(
        0.03,
        ge=0.0,
        le=0.20,
        description=(
            "Maximum acceptable accuracy drop after quantisation. "
            "Raise a warning if the TFLite model accuracy drops more than this."
        ),
    )


class ExperimentConfig(BaseModel):
    """
    Root configuration schema for a full experiment run.

    This is the single typed object passed throughout the pipeline.
    Every field has a clear purpose, default, and validation rule.
    """

    # --- Identity ---
    experiment_name: str = Field(..., description="Unique name for this experiment run")
    seed: int = Field(42, ge=0, description="Global random seed for reproducibility")
    num_classes: int = Field(35, ge=2, le=2000)

    # --- Sub-configs ---
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)

    @model_validator(mode="after")
    def sync_num_classes(self) -> ExperimentConfig:
        """Ensure data.num_classes is consistent with the top-level num_classes."""
        self.data.num_classes = self.num_classes
        return self

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict (JSON-serialisable) representation."""
        return self.model_dump()

    def to_json(self, indent: int = 2) -> str:
        """Return a JSON string representation."""
        return self.model_dump_json(indent=indent)


# =============================================================================
# Config loading
# =============================================================================

def _load_yaml(path: Path) -> DictConfig:
    """Load a single YAML file as an OmegaConf DictConfig."""
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            f"Expected location relative to project root: {path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path}"
        )
    cfg = OmegaConf.load(path)
    logger.debug(f"Loaded config: {path}")
    return cfg


def load_config(
    model: str,
    data: str = "seq30",
    augmentation: str = "none",
    experiment: Optional[str] = None,
    overrides: Optional[dict[str, Any]] = None,
) -> ExperimentConfig:
    """
    Load, merge, and validate the full experiment configuration.

    Merges configs in this order (later keys override earlier ones):
        1. configs/base.yaml                        (global defaults)
        2. configs/model/<model>.yaml               (architecture)
        3. configs/data/<data>.yaml                 (sequence/data params)
        4. configs/augmentation/<augmentation>.yaml (augmentation strategy)
        5. configs/experiment/<experiment>.yaml     (optional, run-level overrides)
        6. <overrides> dict                         (CLI-level overrides, highest priority)

    Parameters
    ----------
    model : str
        Model config name. One of: dense, lstm, gru, bilstm.
    data : str
        Data config name. One of: seq20, seq30, seq40.
    augmentation : str
        Augmentation config name. One of: none, temporal, spatial_temporal.
    experiment : str, optional
        Optional experiment-level config name (e.g. "best_model").
    overrides : dict, optional
        Key-value pairs that override any config field.
        Keys use dot-notation: {"training.learning_rate": 0.0005}.

    Returns
    -------
    ExperimentConfig
        A fully validated, typed config object.

    Raises
    ------
    FileNotFoundError
        If any required config YAML file does not exist.
    pydantic.ValidationError
        If the merged config fails schema validation (missing fields, wrong types).

    Examples
    --------
    # Standard experiment
    cfg = load_config(model="bilstm", data="seq30", augmentation="spatial_temporal")

    # With a CLI-level override
    cfg = load_config(
        model="lstm",
        data="seq30",
        augmentation="none",
        overrides={"training.learning_rate": 0.0005, "training.epochs": 30},
    )
    """
    # Step 1: Load individual YAML files
    configs_to_merge = [
        _load_yaml(_CONFIG_ROOT / "base.yaml"),
        _load_yaml(_CONFIG_ROOT / "model" / f"{model}.yaml"),
        _load_yaml(_CONFIG_ROOT / "data" / f"{data}.yaml"),
        _load_yaml(_CONFIG_ROOT / "augmentation" / f"{augmentation}.yaml"),
    ]

    if experiment is not None:
        configs_to_merge.append(
            _load_yaml(_CONFIG_ROOT / "experiment" / f"{experiment}.yaml")
        )

    # Step 2: Merge with OmegaConf (later configs override earlier ones)
    merged: DictConfig = OmegaConf.merge(*configs_to_merge)

    # Step 3: Apply CLI overrides using OmegaConf's dot-notation merge
    if overrides:
        override_cfg = OmegaConf.create(overrides)
        merged = OmegaConf.merge(merged, override_cfg)
        logger.debug(f"Applied {len(overrides)} CLI override(s): {list(overrides.keys())}")

    # Step 4: Resolve all interpolations (e.g. ${data.sequence_length})
    OmegaConf.resolve(merged)

    # Step 5: Convert to plain dict and validate with Pydantic
    raw_dict = OmegaConf.to_container(merged, resolve=True, throw_on_missing=True)

    try:
        config = ExperimentConfig(**raw_dict)
    except PydanticValidationError as e:
        logger.error(
            f"Configuration validation failed:\n{e}",
            extra={"stage": "config"},
        )
        raise

    # Step 6: Post-load warnings
    if config.augmentation.enabled and not any([
        config.augmentation.temporal_jitter,
        config.augmentation.frame_drop_prob > 0,
        config.augmentation.spatial_flip,
        config.augmentation.gaussian_noise_std > 0,
        config.augmentation.rotation_deg > 0,
    ]):
        logger.warning(
            "augmentation.enabled=True but all augmentation strategies are disabled. "
            "This is valid for a baseline run but may be unintentional.",
            extra={"stage": "config"},
        )

    logger.info(
        f"Config loaded and validated | "
        f"model={config.model.name} | "
        f"seq_len={config.data.sequence_length} | "
        f"augmentation={augmentation} | "
        f"classes={config.num_classes} | "
        f"seed={config.seed}",
        extra={"stage": "config"},
    )

    return config


def load_config_from_manifest(manifest_path: str) -> ExperimentConfig:
    """
    Reconstruct an ExperimentConfig from a saved run manifest JSON file.

    Useful for exactly reproducing a past experiment from its manifest,
    without needing to remember which CLI flags were used.

    Parameters
    ----------
    manifest_path : str
        Path to a run_manifest.json file written by save_run_manifest().

    Returns
    -------
    ExperimentConfig
        Validated config extracted from the manifest.
    """
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)

    config_dict = manifest.get("config", {})
    if not config_dict:
        raise ValueError(f"Manifest at {path} does not contain a 'config' key.")

    config = ExperimentConfig(**config_dict)
    logger.info(
        f"Config reconstructed from manifest: {path}",
        extra={"stage": "config"},
    )
    return config


# =============================================================================
# Config YAML files — written here to keep everything in one place
# =============================================================================

def write_default_configs(config_root: Optional[str] = None) -> None:
    """
    Write all default YAML config files to configs/.

    This is a bootstrap utility — run it once when setting up the project
    to create the full configs/ directory structure.

    Usage:
        python -c "from src.utils.config import write_default_configs; write_default_configs()"
    """
    root = Path(config_root) if config_root else _CONFIG_ROOT
    root.mkdir(parents=True, exist_ok=True)
    for sub in ("model", "data", "augmentation", "experiment"):
        (root / sub).mkdir(exist_ok=True)

    configs = {
        "base.yaml": """
# =============================================================================
# base.yaml — Global defaults for all experiments
# All values here can be overridden by model/, data/, augmentation/ configs.
# =============================================================================

experiment_name: "unnamed_experiment"
seed: 42
num_classes: 35

training:
  batch_size: 32
  epochs: 50
  learning_rate: 0.001
  early_stopping_patience: 10
  early_stopping_monitor: val_accuracy
  early_stopping_mode: max
  reduce_lr_patience: 5
  reduce_lr_factor: 0.5
  reduce_lr_min_lr: 1.0e-6
  shuffle: true
  class_weight_balancing: false

logging:
  log_dir: logs
  level: INFO
  file_level: DEBUG

mlflow:
  experiment_name: "WLASL-35-class"
  tracking_uri: mlruns
  register_best_model: true
  model_registry_name: gesture-lstm-production

export:
  output_dir: models
  quantise: true
  quantisation_mode: dynamic_range
  representative_dataset_size: 100
  max_accuracy_delta: 0.03
""",
        "model/dense.yaml": """
# Dense feedforward baseline — non-temporal, used to prove sequence modelling matters
model:
  name: dense
  hidden_units: 512   # Larger units compensate for lack of recurrence
  num_layers: 2
  dropout: 0.4
  bidirectional: false
  dense_units: 128
  activation: relu
""",
        "model/lstm.yaml": """
# LSTM — core sequence baseline
model:
  name: lstm
  hidden_units: 128
  num_layers: 2
  dropout: 0.3
  bidirectional: false
  dense_units: 64
  activation: relu
""",
        "model/gru.yaml": """
# GRU — fewer parameters than LSTM, often comparable accuracy, faster inference
model:
  name: gru
  hidden_units: 128
  num_layers: 2
  dropout: 0.3
  bidirectional: false
  dense_units: 64
  activation: relu
""",
        "model/bilstm.yaml": """
# Bidirectional LSTM — best accuracy candidate, higher latency
model:
  name: bilstm
  hidden_units: 128
  num_layers: 2
  dropout: 0.3
  bidirectional: true
  dense_units: 64
  activation: relu
""",
        "data/seq20.yaml": """
data:
  sequence_length: 20
  padding: post
  normalisation: wrist_relative
  missing_frame_strategy: zero_fill
  max_missing_frame_pct: 0.30
""",
        "data/seq30.yaml": """
data:
  sequence_length: 30
  padding: post
  normalisation: wrist_relative
  missing_frame_strategy: zero_fill
  max_missing_frame_pct: 0.30
""",
        "data/seq40.yaml": """
data:
  sequence_length: 40
  padding: post
  normalisation: wrist_relative
  missing_frame_strategy: zero_fill
  max_missing_frame_pct: 0.30
""",
        "augmentation/none.yaml": """
# No augmentation — used for all baseline comparison experiments
augmentation:
  enabled: false
  temporal_jitter: false
  frame_drop_prob: 0.0
  spatial_flip: false
  gaussian_noise_std: 0.0
  rotation_deg: 0.0
  speed_jitter: false
""",
        "augmentation/temporal.yaml": """
# Temporal augmentation only
augmentation:
  enabled: true
  temporal_jitter: true
  frame_drop_prob: 0.10
  spatial_flip: false
  gaussian_noise_std: 0.0
  rotation_deg: 0.0
  speed_jitter: true
""",
        "augmentation/spatial_temporal.yaml": """
# Full spatial + temporal augmentation — used for final model training
augmentation:
  enabled: true
  temporal_jitter: true
  frame_drop_prob: 0.10
  spatial_flip: true
  gaussian_noise_std: 0.01
  rotation_deg: 5.0
  speed_jitter: true
""",
        "experiment/baseline.yaml": """
# Group 1: Architecture comparison — isolates model type as the single variable
experiment_name: baseline_architecture_comparison
seed: 42
""",
        "experiment/ablation_augmentation.yaml": """
# Group 2: Augmentation ablation — fixed LSTM + seq30, varies augmentation
experiment_name: ablation_augmentation
seed: 42
""",
        "experiment/ablation_sequence.yaml": """
# Group 3: Sequence length ablation — fixed LSTM + best augmentation
experiment_name: ablation_sequence_length
seed: 42
""",
        "experiment/ablation_landmarks.yaml": """
# Group 4: Landmark configuration ablation
experiment_name: ablation_landmark_config
seed: 42
""",
        "experiment/best_model.yaml": """
# Champion model config — used for final training and TFLite export
experiment_name: best_model_bilstm_spatial_temporal_seq30
seed: 42

# Primary metric: validation accuracy
# Secondary metric (for deployment): accuracy / median_latency_ms
""",
    }

    for filename, content in configs.items():
        filepath = root / filename
        if filepath.exists():
            logger.debug(f"Skipping existing config: {filepath}")
            continue
        filepath.write_text(content.strip() + "\n", encoding="utf-8")
        logger.info(f"Wrote config: {filepath}")

    logger.info(
        f"Default config files written to {root.resolve()}. "
        f"Review and adjust before running experiments."
    )