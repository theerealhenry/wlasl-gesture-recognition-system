"""
src/utils/config.py
====================
Configuration management for the WLASL gesture recognition pipeline.

Architecture:
  - All hyperparameters and paths live in YAML files under configs/.
  - OmegaConf loads and merges hierarchical YAML configs with dot-access.
  - Pydantic v2 schemas validate the merged config, providing typed access
    and immediate errors on missing, mistyped, or unexpected fields.
  - Nothing is ever hardcoded in any source file.

Config composition:
    python pipelines/run_training.py \\
        --model lstm \\
        --data seq30 \\
        --augmentation spatial_temporal \\
        --run-name lstm_seq30_aug

    This loads:
        configs/base.yaml                         (global defaults)
        configs/model/lstm.yaml                   (model architecture)
        configs/data/seq30.yaml                   (sequence length)
        configs/augmentation/spatial_temporal.yaml

    Merges them left-to-right (later keys override earlier ones), validates
    the result against the Pydantic schema, and returns a typed, frozen config
    object. Attempting to mutate the config after construction raises an error.

CLI overrides use dot-notation and are resolved via OmegaConf.from_dotlist:
    overrides={"training.learning_rate": 0.0005}

Usage:
    from src.utils.config import load_config, ExperimentConfig

    cfg = load_config(model="bilstm", data="seq30", augmentation="spatial_temporal")
    print(cfg.training.learning_rate)   # 0.001
    print(cfg.model.hidden_units)       # 128
    print(cfg.config_hash)              # SHA256 fingerprint of the merged config
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from omegaconf import OmegaConf, DictConfig
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from pydantic import ValidationError as PydanticValidationError

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Root of the configs/ directory — resolved relative to this file's location
_CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"


# =============================================================================
# Enums — type-safe replacements for string constants
# =============================================================================

class ModelType(str, Enum):
    """Supported model architecture names."""
    DENSE  = "dense"
    LSTM   = "lstm"
    GRU    = "gru"
    BILSTM = "bilstm"


class PaddingStrategy(str, Enum):
    PRE  = "pre"
    POST = "post"


class NormalisationStrategy(str, Enum):
    WRIST_RELATIVE = "wrist_relative"
    NONE           = "none"


class MissingFrameStrategy(str, Enum):
    ZERO_FILL   = "zero_fill"
    INTERPOLATE = "interpolate"
    SKIP        = "skip"


class QuantisationMode(str, Enum):
    DYNAMIC_RANGE = "dynamic_range"
    FULL_INTEGER  = "full_integer"
    FLOAT16       = "float16"


class EarlyStoppingMode(str, Enum):
    MIN = "min"
    MAX = "max"


# =============================================================================
# Pydantic v2 schema definitions
# All models use extra="forbid" — unexpected YAML keys raise immediately,
# catching typos before they silently go unnoticed mid-training.
# All models are frozen after construction — mutation raises ValidationError.
# =============================================================================

class DataConfig(BaseModel):
    """Paths and dataset parameters."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    raw_dir: str = "data/raw"
    landmark_dir: str = "data/landmarks"
    splits_dir: str = "data/splits"
    num_classes: int = Field(35, ge=2, le=2000, description="Number of sign classes")
    sequence_length: int = Field(60, ge=5, le=150, description="Fixed sequence length in frames")
    padding: PaddingStrategy = PaddingStrategy.POST
    normalisation: NormalisationStrategy = NormalisationStrategy.WRIST_RELATIVE
    missing_frame_strategy: MissingFrameStrategy = MissingFrameStrategy.ZERO_FILL
    max_missing_frame_pct: float = Field(
        0.30,
        ge=0.0,
        le=1.0,
        description="Skip videos where more than this fraction of frames are missing",
    )

    z_coord_clip: float = Field(
        0.10,
        ge=0.0,
        le=1.0,
        description=(
            "Soft-clip z-coordinates to ±this value after wrist-relative normalisation. "
            "Removes physically implausible MediaPipe depth outliers. "
            "Set to 0.0 to disable. Recommended: 0.10 (Notebook 03 finding)."
        ),
    )
    flip_min_hand_presence: float = Field(
        0.30,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum fraction of frames where BOTH hands must be present "
            "for spatial flip augmentation to be applied to a clip. "
            "Enforced at clip level (not sign level) to protect one-handed signs. "
            "Recommended: 0.30 (Notebook 03 finding)."
        ),
    )
    normalise_pose: bool = Field(
        False,
        description=(
            "Whether to apply wrist-relative normalisation to pose landmarks. "
            "Should be False: pose body-position is discriminative signal, not noise. "
            "Confirmed by Notebook 03 analysis."
        ),
    )

    landmark_config: str = Field(
        "full",
        description=(
            "Which landmark bands to use as model input. "
            "One of: 'full' (225 dims, hands+pose), "
            "'hands_only' (126 dims, left+right hands), "
            "'pose_only' (99 dims, pose skeleton). "
            "Controlled by Group 4 ablation. "
            "Notebook 03: hands_only Fisher=0.752 vs full Fisher=0.432."
        ),
    )    

    @field_validator("landmark_config")
    @classmethod
    def validate_landmark_config(cls, v: str) -> str:
        allowed = {"full", "hands_only", "pose_only"}
        if v not in allowed:
            raise ValueError(
                f"landmark_config must be one of {allowed}, got '{v}'. "
            )
        return v

    @field_validator("raw_dir", "landmark_dir", "splits_dir", mode="before")
    @classmethod
    def coerce_to_str(cls, v: Any) -> str:
        return str(v)


class ModelConfig(BaseModel):
    """Model architecture parameters."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: ModelType = Field(..., description="Architecture: dense | lstm | gru | bilstm")
    hidden_units: int = Field(128, ge=16, le=1024)
    num_layers: int = Field(2, ge=1, le=6)
    dropout: float = Field(0.3, ge=0.0, le=0.8)
    bidirectional: bool = False
    dense_units: int = Field(
        64, ge=16, le=512,
        description="Units in the intermediate Dense layer before the classifier",
    )
    activation: str = Field("relu", description="Activation for Dense layers")

    @model_validator(mode="after")
    def validate_bidirectional_consistency(self) -> "ModelConfig":
        """BiLSTM must have bidirectional=True; others must have bidirectional=False."""
        if self.name == ModelType.BILSTM and not self.bidirectional:
            # Auto-correct rather than error — bilstm.yaml sets bidirectional: true,
            # but a user might pass model=bilstm with an override that forgets this.
            # We raise here to surface the inconsistency explicitly.
            raise ValueError(
                "model.name='bilstm' requires model.bidirectional=True. "
                "Set bidirectional: true in your model config or override."
            )
        if self.name != ModelType.BILSTM and self.bidirectional:
            raise ValueError(
                f"model.name='{self.name.value}' does not support bidirectional=True. "
                "Only 'bilstm' supports bidirectional processing."
            )
        return self


class AugmentationConfig(BaseModel):
    """Data augmentation strategy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    temporal_jitter: bool = False
    frame_drop_prob: float = Field(0.0, ge=0.0, le=0.5)
    spatial_flip: bool = False
    gaussian_noise_std: float = Field(0.0, ge=0.0, le=0.1)
    rotation_deg: float = Field(0.0, ge=0.0, le=30.0)
    speed_jitter: bool = False
    gaussian_noise_detected_only: bool = Field(
        True,
        description=(
            "If True, Gaussian noise is applied only to detected (non-zero) landmark "
            "frames. Zero-filled frames (one-handed signs, detection failures) are "
            "passed through unchanged. MUST be True to preserve semantic zero-fill signal. "
            "Notebook 03 finding: zero-fill is discriminative for one-handed signs."
        ),
    )


class TrainingConfig(BaseModel):
    """Training loop parameters."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    batch_size: int = Field(32, ge=1, le=512)
    epochs: int = Field(50, ge=1, le=500)
    learning_rate: float = Field(0.001, gt=0.0, le=0.1)
    early_stopping_patience: int = Field(10, ge=1, le=100)
    early_stopping_monitor: str = "val_accuracy"
    early_stopping_mode: EarlyStoppingMode = EarlyStoppingMode.MAX
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
                f"early_stopping_monitor '{v}' does not look like a valid Keras metric name. "
                f"Expected one of: val_accuracy, val_loss, train_accuracy, etc."
            )
        return v


class LoggingConfig(BaseModel):
    """Logging configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    log_dir: str = "logs"
    level: str = Field("INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    file_level: str = Field("DEBUG", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")


class MLflowConfig(BaseModel):
    """MLflow experiment tracking configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    experiment_name: str = "WLASL-35-class"
    tracking_uri: str = "mlruns"
    register_best_model: bool = True
    model_registry_name: str = "gesture-lstm-production"


class ExportConfig(BaseModel):
    """TFLite export and quantisation parameters."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    output_dir: str = "models"
    quantise: bool = True
    quantisation_mode: QuantisationMode = QuantisationMode.DYNAMIC_RANGE
    representative_dataset_size: int = Field(
        100, ge=10, le=1000,
        description="Samples used for full-integer quantisation calibration",
    )
    max_accuracy_delta: float = Field(
        0.03, ge=0.0, le=0.20,
        description=(
            "Maximum acceptable accuracy drop after quantisation. "
            "Logs a warning if the TFLite model accuracy drops more than this."
        ),
    )


class ExperimentConfig(BaseModel):
    """
    Root configuration schema for a full experiment run.

    This is the single typed, frozen object passed throughout the pipeline.
    Frozen means any attempt to mutate a field after construction raises
    a ValidationError — configs are immutable once loaded.

    A SHA256 fingerprint (config_hash) is computed over the JSON-serialised
    config at construction time. Log or store this alongside model artefacts
    for unambiguous experiment attribution.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

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

    # --- Computed fingerprint (set after construction via __init__) ---
    config_hash: str = Field(
        default="",
        description=(
            "SHA256 fingerprint of the serialised config. "
            "Computed automatically at construction time. "
            "Store alongside model artefacts for unambiguous experiment attribution."
        ),
    )

    @model_validator(mode="after")
    def validate_num_classes_consistency(self) -> "ExperimentConfig":
        """
        Validate that data.num_classes equals the top-level num_classes.
        Raises ValueError immediately if they disagree — no silent mutation.
        """
        if self.data.num_classes != self.num_classes:
            raise ValueError(
                f"num_classes mismatch: top-level num_classes={self.num_classes} "
                f"but data.num_classes={self.data.num_classes}. "
                f"These must be equal. Set one explicitly or remove one from your configs."
            )
        return self

    @model_validator(mode="after")
    def validate_augmentation_consistency(self) -> "ExperimentConfig":
        """Warn if augmentation is enabled but all strategies are disabled."""
        aug = self.augmentation
        if aug.enabled:
            any_active = any([
                aug.temporal_jitter,
                aug.frame_drop_prob > 0,
                aug.spatial_flip,
                aug.gaussian_noise_std > 0,
                aug.rotation_deg > 0,
                aug.speed_jitter,
            ])
            if not any_active:
                # Not a hard error — may be intentional for a controlled baseline.
                # Warning is emitted by load_config() post-construction.
                pass
        return self

    @model_validator(mode="after")
    def compute_config_fingerprint(self) -> "ExperimentConfig":
        """
        Compute a SHA256 hash over the canonical JSON representation.
        Excludes config_hash itself to avoid circular dependency.
        Uses model_copy to temporarily clear config_hash before hashing.
        """
        # Build a dict without config_hash to avoid circularity
        data = self.model_dump(exclude={"config_hash"})
        # Stable serialisation: sort keys, no whitespace variation
        canonical_json = json.dumps(data, sort_keys=True, default=str)
        fingerprint = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
        # Use object.__setattr__ because the model is frozen at this point
        object.__setattr__(self, "config_hash", fingerprint)
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
        try:
            rel = path.relative_to(Path.cwd())
        except ValueError:
            rel = path
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            f"Expected location: {rel}\n"
            f"Run 'python -c \"from src.utils.config import write_default_configs; "
            f"write_default_configs()\"' to create all default config files."
        )
    cfg = OmegaConf.load(path)
    logger.debug(f"Loaded config: {path}")
    return cfg


def _apply_dot_notation_overrides(
    merged: DictConfig,
    overrides: dict[str, Any],
) -> DictConfig:
    """
    Apply a dict of dot-notation overrides to a DictConfig.

    Converts {"training.learning_rate": 0.0005} into a proper nested
    OmegaConf structure before merging, so nested keys work correctly.

    Parameters
    ----------
    merged : DictConfig
        Base config to override.
    overrides : dict[str, Any]
        Dot-notation key-value pairs.

    Returns
    -------
    DictConfig
        Merged config with overrides applied.
    """
    # Convert {"a.b.c": v} → OmegaConf dotlist format ["a.b.c=v"]
    dotlist = []
    for key, value in overrides.items():
        # OmegaConf from_dotlist expects "key=value" strings
        # For non-string values, OmegaConf handles them via structured override
        if isinstance(value, bool):
            dotlist.append(f"{key}={'true' if value else 'false'}")
        elif isinstance(value, str):
            dotlist.append(f"{key}={value}")
        else:
            dotlist.append(f"{key}={value}")

    override_cfg = OmegaConf.from_dotlist(dotlist)
    return OmegaConf.merge(merged, override_cfg)


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
        1. configs/base.yaml                         (global defaults)
        2. configs/model/<model>.yaml                (architecture)
        3. configs/data/<data>.yaml                  (sequence/data params)
        4. configs/augmentation/<augmentation>.yaml  (augmentation strategy)
        5. configs/experiment/<experiment>.yaml      (optional, run-level overrides)
        6. <overrides> dict                          (CLI-level, highest priority)

    Parameters
    ----------
    model : str
        Model config name. One of: dense, lstm, gru, bilstm.
    data : str
        Data config name. One of: seq20, seq30, seq40, seq60, seq80, seq100.
    augmentation : str
        Augmentation config name. One of: none, temporal, spatial_temporal.
    experiment : str, optional
        Optional experiment-level config name (e.g. "best_model").
    overrides : dict, optional
        Dot-notation key-value pairs that override any config field.
        Example: {"training.learning_rate": 0.0005, "training.epochs": 30}

    Returns
    -------
    ExperimentConfig
        A fully validated, typed, frozen config object with a SHA256 fingerprint.

    Raises
    ------
    FileNotFoundError
        If any required config YAML file does not exist.
    pydantic.ValidationError
        If the merged config fails schema validation (missing fields, wrong
        types, unexpected extra fields, or num_classes mismatch).

    Examples
    --------
    Standard experiment:
        cfg = load_config(model="bilstm", data="seq30", augmentation="spatial_temporal")

    With CLI-level overrides:
        cfg = load_config(
            model="lstm",
            data="seq30",
            augmentation="none",
            overrides={"training.learning_rate": 0.0005, "training.epochs": 30},
        )

    Access config fingerprint for experiment tracking:
        print(cfg.config_hash)   # e.g. "a3f2c8b1..."
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

    # Step 3: Apply CLI overrides using dot-notation conversion
    if overrides:
        merged = _apply_dot_notation_overrides(merged, overrides)
        logger.debug(
            f"Applied {len(overrides)} CLI override(s): {list(overrides.keys())}"
        )

    # Step 4: Resolve all interpolations (e.g. ${data.sequence_length})
    OmegaConf.resolve(merged)

    # Step 5: Convert to plain dict and validate with Pydantic
    raw_dict = OmegaConf.to_container(merged, resolve=True, throw_on_missing=True)

    try:
        config = ExperimentConfig(**raw_dict)
    except PydanticValidationError as exc:
        logger.error(
            f"Configuration validation failed:\n{exc}",
            extra={"stage": "config"},
        )
        raise

    # Step 6: Post-load warnings (done here because validators can't use logger)
    aug = config.augmentation
    if aug.enabled and not any([
        aug.temporal_jitter,
        aug.frame_drop_prob > 0,
        aug.spatial_flip,
        aug.gaussian_noise_std > 0,
        aug.rotation_deg > 0,
        aug.speed_jitter,
    ]):
        logger.warning(
            "augmentation.enabled=True but all augmentation strategies are disabled. "
            "This is valid for a baseline run but may be unintentional. "
            "Set augmentation.enabled=False explicitly if this is intentional.",
            extra={"stage": "config"},
        )

    logger.info(
        f"Config loaded and validated | "
        f"model={config.model.name.value} | "
        f"seq_len={config.data.sequence_length} | "
        f"augmentation={augmentation} | "
        f"classes={config.num_classes} | "
        f"seed={config.seed} | "
        f"hash={config.config_hash[:12]}",
        extra={"stage": "config"},
    )

    return config


def load_config_from_manifest(manifest_path: str) -> ExperimentConfig:
    """
    Reconstruct an ExperimentConfig from a saved run manifest JSON file.

    Useful for exactly reproducing a past experiment from its manifest
    without needing to remember which CLI flags were used.

    Parameters
    ----------
    manifest_path : str
        Path to a run_manifest.json file written by save_run_manifest().

    Returns
    -------
    ExperimentConfig
        Validated config extracted from the manifest.

    Raises
    ------
    FileNotFoundError
        If the manifest file does not exist.
    ValueError
        If the manifest does not contain a 'config' key.
    """
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)

    config_dict = manifest.get("config", {})
    if not config_dict:
        raise ValueError(
            f"Manifest at {path} does not contain a 'config' key. "
            "Ensure the manifest was written by save_run_manifest()."
        )

    config = ExperimentConfig(**config_dict)
    logger.info(
        f"Config reconstructed from manifest: {path} | hash={config.config_hash[:12]}",
        extra={"stage": "config"},
    )
    return config


# =============================================================================
# Config YAML bootstrap
# =============================================================================

def write_default_configs(config_root: Optional[str] = None) -> None:
    """
    Write all default YAML config files to configs/.

    This is a ONE-TIME BOOTSTRAP utility. Run it immediately after cloning the
    repository and installing dependencies to create the configs/ directory
    structure. After this, the YAML files under configs/ become the source of
    truth — edit them there, not here.

    Usage:
        python -c "from src.utils.config import write_default_configs; write_default_configs()"

    Existing files are never overwritten — it is safe to re-run after making
    manual edits to your configs.

    Note: The YAML content below is the canonical default for a fresh project.
    Once configs/ exists on disk, those files take precedence. This function
    is intentionally not called during normal training — it is a setup tool only.
    """
    root = Path(config_root) if config_root else _CONFIG_ROOT
    root.mkdir(parents=True, exist_ok=True)
    for sub in ("model", "data", "augmentation", "experiment"):
        (root / sub).mkdir(exist_ok=True)

    configs: dict[str, str] = {
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
  class_weight_balancing: true

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
  hidden_units: 512
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
  max_missing_frame_pct: 0.95
  z_coord_clip: 0.10
  flip_min_hand_presence: 0.30
  normalise_pose: false
  num_classes: 35
""",
        "data/seq30.yaml": """
data:
  sequence_length: 30
  padding: post
  normalisation: wrist_relative
  missing_frame_strategy: zero_fill
  max_missing_frame_pct: 0.95
  z_coord_clip: 0.10
  flip_min_hand_presence: 0.30
  normalise_pose: false
  num_classes: 35
""",
        "data/seq40.yaml": """
data:
  sequence_length: 40
  padding: post
  normalisation: wrist_relative
  missing_frame_strategy: zero_fill
  max_missing_frame_pct: 0.95
  z_coord_clip: 0.10
  flip_min_hand_presence: 0.30
  normalise_pose: false
  num_classes: 35
""",
        "data/seq60.yaml": """
data:
  sequence_length: 60
  padding: post
  normalisation: wrist_relative
  missing_frame_strategy: zero_fill
  max_missing_frame_pct: 0.95
  z_coord_clip: 0.10
  flip_min_hand_presence: 0.30
  normalise_pose: false
  num_classes: 35
""",
        "data/seq80.yaml": """
data:
  sequence_length: 80
  padding: post
  normalisation: wrist_relative
  missing_frame_strategy: zero_fill
  max_missing_frame_pct: 0.95
  z_coord_clip: 0.10
  flip_min_hand_presence: 0.30
  normalise_pose: false
  num_classes: 35
""",
        "data/seq100.yaml": """
data:
  sequence_length: 100
  padding: post
  normalisation: wrist_relative
  missing_frame_strategy: zero_fill
  max_missing_frame_pct: 0.95
  z_coord_clip: 0.10
  flip_min_hand_presence: 0.30
  normalise_pose: false
  num_classes: 35
""",
        "augmentation/temporal.yaml": """
# Temporal augmentation only
augmentation:
  enabled: true
  temporal_jitter: true
  frame_drop_prob: 0.10
  spatial_flip: false
  gaussian_noise_std: 0.0
  gaussian_noise_detected_only: true
  rotation_deg: 0.0
  speed_jitter: true
""",
        "augmentation/spatial_temporal.yaml": """
# Full spatial + temporal augmentation — used for final model training.
# gaussian_noise_detected_only=true: preserves semantic zero-fill for one-handed signs.
# spatial_flip enforced at clip level via data.flip_min_hand_presence threshold.
augmentation:
  enabled: true
  temporal_jitter: true
  frame_drop_prob: 0.10
  spatial_flip: true
  gaussian_noise_std: 0.01
  gaussian_noise_detected_only: true
  rotation_deg: 5.0
  speed_jitter: true
""",
        "augmentation/none.yaml": """
# No augmentation — used for all baseline comparison experiments
augmentation:
  enabled: false
  temporal_jitter: false
  frame_drop_prob: 0.0
  spatial_flip: false
  gaussian_noise_std: 0.0
  gaussian_noise_detected_only: true
  rotation_deg: 0.0
  speed_jitter: false
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
# Extended to {20, 30, 40, 60, 80, 100} per Notebook 03 finding:
# median clip = 67 frames, P75 = 84, P90 = 95
# seq_len=30 captures only 50.5% of mean content — insufficient.
# seq_len=60 captures 85% — minimum defensible primary configuration.
experiment_name: ablation_sequence_length
seed: 42
""",
        "experiment/ablation_landmarks.yaml": """
# Group 4: Landmark configuration ablation
# Hands-only Fisher = 0.752 (1.74x full-225 baseline of 0.432)
# Run with best seq_len and best augmentation from Groups 2 and 3.
experiment_name: ablation_landmark_config
seed: 42
""",
        "experiment/best_model.yaml": """
# Champion model config — used for final training and TFLite export.
# Primary seq_len candidate: 60 (85% mean content coverage per Notebook 03).
# Final seq_len determined by Group 3 ablation results.
experiment_name: best_model_bilstm_spatial_temporal_seq60
seed: 42

training:
  class_weight_balancing: true
""",
    }

    written = 0
    skipped = 0
    for filename, content in configs.items():
        filepath = root / filename
        if filepath.exists():
            logger.debug(f"Skipping existing config (will not overwrite): {filepath}")
            skipped += 1
            continue
        filepath.write_text(content.strip() + "\n", encoding="utf-8")
        logger.info(f"Wrote config: {filepath}")
        written += 1

    logger.info(
        f"write_default_configs complete | "
        f"written={written} | skipped={skipped} | root={root.resolve()}"
    )
    if written > 0:
        logger.info(
            "Review and adjust the new config files under configs/ before running experiments. "
            "These files are now the source of truth — edit them there, not in config.py."
        )