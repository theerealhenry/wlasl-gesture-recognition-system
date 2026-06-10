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
        --data seq60 \\
        --augmentation spatial_temporal \\
        --run-name lstm_seq60_aug

    This loads:
        configs/base.yaml                         (global defaults)
        configs/model/lstm.yaml                   (model architecture)
        configs/data/seq60.yaml                   (sequence length)
        configs/augmentation/spatial_temporal.yaml

    Merges them left-to-right (later keys override earlier ones), validates
    the result against the Pydantic schema, and returns a typed, frozen config
    object. Attempting to mutate the config after construction raises an error.

CLI overrides use dot-notation and are resolved via OmegaConf.from_dotlist:
    overrides={"training.learning_rate": 0.0005}

Usage:
    from src.utils.config import load_config, ExperimentConfig

    cfg = load_config(model="bilstm", data="seq60", augmentation="spatial_temporal")
    print(cfg.training.learning_rate)   # 0.001
    print(cfg.model.hidden_units)       # 64  (ablation default)
    print(cfg.config_hash)              # SHA256 fingerprint of the merged config

Stage 5 changes (relative to Stage 4)
--------------------------------------
ModelConfig
    recurrent_dropout added (float, default 0.1).
    Required by architectures.py build_lstm/build_gru/build_bilstm and
    by train.py _log_mlflow_params. Absent in Stage 4 schema — would have
    raised AttributeError at first recurrent model build call.

TrainingConfig
    epochs default raised from 50 → 80.
    early_stopping_patience raised from 10 → 15.
    Rationale: signer-independent splits + class weighting on a 236-clip
    dataset converge slowly. 50 epochs / patience 10 terminates too early
    for underrepresented classes (clothes, think: 2–3 clips).

ExperimentConfig
    primary_metric, secondary_metric, deployment_metric added (Optional[str]).
    Required by configs/experiment/best_model.yaml (Step 8). With
    extra="forbid", adding these fields to the YAML without adding them to
    the Pydantic model raises ValidationError immediately.

write_default_configs()
    force parameter added (bool, default False). When True, existing files
    are overwritten. Required after fixing ModelConfig to propagate the
    recurrent_dropout field into all recurrent model YAMLs.
    YAML content updated throughout — see individual docstrings below.
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
    """
    Model architecture parameters.

    Stage 5 addition: recurrent_dropout (float, default 0.1).
    Required by architectures.py build_lstm/build_gru/build_bilstm which
    call cfg.model.recurrent_dropout directly. Also read by train.py
    _log_mlflow_params via _get_config_attr(cfg.model, 'recurrent_dropout', 0.0).

    dense.yaml does not set recurrent_dropout in practice (Dense has no
    recurrent connections), but the field defaults to 0.0 so Dense configs
    that omit it pass validation without error. The _get_config_attr helper
    in train.py provides an additional safety layer for this field.

    ablation default: hidden_units=64 for lstm/gru/bilstm YAMLs.
    champion default: hidden_units=128, set via run_all_experiments.py override
    or the best_model experiment config.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: ModelType = Field(..., description="Architecture: dense | lstm | gru | bilstm")
    hidden_units: int = Field(128, ge=16, le=1024)
    num_layers: int = Field(2, ge=1, le=6)
    dropout: float = Field(0.3, ge=0.0, le=0.8)
    recurrent_dropout: float = Field(
        0.1,
        ge=0.0,
        lt=1.0,
        description=(
            "Dropout applied to the recurrent connections (h_{t-1} → h_t gates). "
            "Default 0.1 — lighter than input dropout to avoid gradient instability "
            "in stacked recurrent networks on short sequences. "
            "Setting > 0.0 disables the CuDNN fast-path in TF 2.13; not a concern "
            "for CPU training on WLASL (236 clips). "
            "Set to 0.0 for Dense baseline configs — ignored by build_dense()."
        ),
    )
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
    """
    Training loop parameters.

    Stage 5 changes:
        epochs default raised 50 → 80.
            Rationale: signer-independent splits and class weighting on a
            236-clip dataset converge slowly. Early stopping (patience=15)
            handles termination; 80 is the ceiling, not the expected runtime.
        early_stopping_patience raised 10 → 15.
            Rationale: macro-F1 on 52 val clips has high variance epoch-to-epoch.
            Patience 10 prematurely stops runs that plateau before a LR reduction
            fires at patience 5. Patience 15 gives ReduceLROnPlateau room to
            fire twice before stopping.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    batch_size: int = Field(32, ge=1, le=512)
    epochs: int = Field(
        80,
        ge=1,
        le=500,
        description=(
            "Maximum training epochs. Raised from 50 to 80 for Stage 5. "
            "Early stopping with patience=15 will terminate earlier if appropriate. "
            "With 236 training clips and signer-independent splits, LSTM convergence "
            "typically requires 40–70 epochs."
        ),
    )
    learning_rate: float = Field(0.001, gt=0.0, le=0.1)
    early_stopping_patience: int = Field(
        15,
        ge=1,
        le=100,
        description=(
            "Epochs without val_macro_f1 improvement before stopping. "
            "Raised from 10 to 15 for Stage 5. Gives ReduceLROnPlateau "
            "(patience=5) room to fire twice before early stopping triggers. "
            "NOTE: early stopping in train.py is manual (Python loop on "
            "val_macro_f1), NOT a Keras EarlyStopping callback. "
            "This value feeds the manual patience counter in train_one_run()."
        ),
    )
    early_stopping_monitor: str = Field(
        "val_accuracy",
        description=(
            "Metric monitored by ReduceLROnPlateau. "
            "IMPORTANT: this field controls ONLY the Keras ReduceLROnPlateau "
            "callback — it does NOT control early stopping. "
            "Early stopping in train.py is implemented as a manual Python loop "
            "monitoring val_macro_f1 (sklearn), which is not a Keras metric and "
            "cannot be monitored by Keras callbacks. "
            "'val_accuracy' is correct here: it is computed natively by Keras "
            "during model.fit() and is available to ReduceLROnPlateau. "
            "Changing this to 'val_macro_f1' would raise a KeyError at runtime "
            "because val_macro_f1 is never added to hist.history."
        ),
    )
    early_stopping_mode: EarlyStoppingMode = EarlyStoppingMode.MAX
    reduce_lr_patience: int = Field(
        5,
        ge=1,
        le=50,
        description=(
            "Epochs without val_accuracy improvement before LR is halved. "
            "Feeds ReduceLROnPlateau(monitor='val_accuracy', patience=N). "
            "Must be < early_stopping_patience so LR reduction fires before "
            "the manual early stopping loop terminates the run."
        ),
    )
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

    @model_validator(mode="after")
    def validate_patience_ordering(self) -> "TrainingConfig":
        """
        Enforce that reduce_lr_patience < early_stopping_patience.

        If LR patience >= early stopping patience, ReduceLROnPlateau can never
        fire before training terminates — the LR schedule becomes a no-op.
        This is a silent correctness issue that would only be visible in MLflow
        logs (learning_rate never changes despite plateau).
        """
        if self.reduce_lr_patience >= self.early_stopping_patience:
            raise ValueError(
                f"reduce_lr_patience={self.reduce_lr_patience} must be strictly "
                f"less than early_stopping_patience={self.early_stopping_patience}. "
                "ReduceLROnPlateau must fire before early stopping terminates training. "
                "Recommended: reduce_lr_patience = early_stopping_patience // 3."
            )
        return self


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

    Stage 5 additions
    -----------------
    primary_metric : Optional[str]
        The metric used for all champion selection decisions across experiment
        groups. Set to "val_macro_f1" in best_model.yaml and any experiment
        config that participates in group-level selection. With 21 singleton
        validation classes, accuracy and macro-F1 can diverge significantly —
        only macro-F1 is trustworthy as a selection criterion.

    secondary_metric : Optional[str]
        Reported alongside primary_metric for completeness. Set to "val_acc".

    deployment_metric : Optional[str]
        The composite metric used for the final champion → deployment decision.
        Set to "val_macro_f1 / median_latency_ms" in best_model.yaml to
        reflect the accuracy/latency trade-off for mobile TFLite deployment.
        This field is informational only — no automated code reads it.

    These three fields are Optional[str] with None defaults so all existing
    experiment configs (baseline, ablation_*) that do not set them remain
    valid without change. extra="forbid" means they MUST be declared here
    to appear in any experiment YAML.
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

    # --- Stage 5: experiment-level metric declarations ---
    # Optional — absent from ablation/baseline configs, present in best_model.yaml.
    # Informational metadata logged to MLflow and stored in run_manifest.json.
    primary_metric: Optional[str] = Field(
        None,
        description=(
            "Primary selection metric for this experiment group. "
            "Set to 'val_macro_f1' in best_model.yaml. "
            "Overrides Stage 5 selection logic when set."
        ),
    )
    secondary_metric: Optional[str] = Field(
        None,
        description=(
            "Secondary metric reported alongside primary_metric. "
            "Set to 'val_acc' in best_model.yaml."
        ),
    )
    deployment_metric: Optional[str] = Field(
        None,
        description=(
            "Composite metric for the champion → deployment decision. "
            "Set to 'val_macro_f1 / median_latency_ms' in best_model.yaml. "
            "Informational only — not read by automated pipeline code."
        ),
    )

    # --- Computed fingerprint (set after construction via object.__setattr__) ---
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
        """
        data = self.model_dump(exclude={"config_hash"})
        canonical_json = json.dumps(data, sort_keys=True, default=str)
        fingerprint = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
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
            f"write_default_configs(force=True)\"' to (re)create all config files."
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
    dotlist = []
    for key, value in overrides.items():
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
    data: str = "seq60",
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
        cfg = load_config(model="bilstm", data="seq60", augmentation="spatial_temporal")

    With CLI-level overrides:
        cfg = load_config(
            model="lstm",
            data="seq60",
            augmentation="none",
            overrides={"training.learning_rate": 0.0005, "training.epochs": 30},
        )

    Champion run with higher hidden_units:
        cfg = load_config(
            model="bilstm",
            data="seq80",
            augmentation="spatial_temporal",
            experiment="best_model",
            overrides={"model.hidden_units": 128},
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

def write_default_configs(
    config_root: Optional[str] = None,
    force: bool = False,
) -> None:
    """
    Write all default YAML config files to configs/.

    This is a ONE-TIME BOOTSTRAP utility — and also the Stage 5 regeneration
    tool. Run it with force=True after any change to ModelConfig (e.g. adding
    recurrent_dropout) to propagate the new field into all YAML files.

    Usage (fresh install):
        python -c "from src.utils.config import write_default_configs; \\
                   write_default_configs()"

    Usage (Stage 5 update — regenerate all files):
        python -c "from src.utils.config import write_default_configs; \\
                   write_default_configs(force=True)"

    Parameters
    ----------
    config_root : str, optional
        Override directory. Defaults to configs/ relative to this file.
    force : bool, default False
        If False (default), existing files are never overwritten — safe
        for re-running after manual edits.
        If True, all files are overwritten with the canonical defaults
        defined in this function. Use this after a schema change (e.g.
        adding recurrent_dropout to ModelConfig) to ensure all YAML
        files are consistent with the Pydantic schema.

    Notes
    -----
    Stage 5 YAML changes relative to Stage 4:
        base.yaml
            training.epochs: 50 → 80
            training.early_stopping_patience: 10 → 15
        model/lstm.yaml
            model.hidden_units: 128 → 64  (ablation default)
            model.recurrent_dropout: 0.1  (NEW — required by ModelConfig)
        model/gru.yaml
            model.hidden_units: 128 → 64  (ablation default)
            model.recurrent_dropout: 0.1  (NEW — required by ModelConfig)
        model/bilstm.yaml
            model.hidden_units: 128 → 64  (ablation default; 128 used only
                                           for champion run via override)
            model.recurrent_dropout: 0.1  (NEW — required by ModelConfig)
        model/dense.yaml
            model.recurrent_dropout: 0.0  (NEW — required by ModelConfig;
                                           Dense has no recurrent connections)
        experiment/best_model.yaml
            primary_metric: val_macro_f1  (NEW — required by ExperimentConfig)
            secondary_metric: val_acc     (NEW)
            deployment_metric: val_macro_f1 / median_latency_ms  (NEW)
    """
    root = Path(config_root) if config_root else _CONFIG_ROOT
    root.mkdir(parents=True, exist_ok=True)
    for sub in ("model", "data", "augmentation", "experiment"):
        (root / sub).mkdir(exist_ok=True)

    configs: dict[str, str] = {

        # ──────────────────────────────────────────────────────────────────
        # base.yaml — global defaults
        # Stage 5: epochs 50→80, patience 10→15
        # ──────────────────────────────────────────────────────────────────
        "base.yaml": """
# =============================================================================
# base.yaml — Global defaults for all experiments
# All values here can be overridden by model/, data/, augmentation/ configs.
#
# Stage 5 changes:
#   training.epochs: 50 → 80
#       Signer-independent splits + class weighting on 236 clips converge
#       slowly. Early stopping (patience=15) handles termination.
#   training.early_stopping_patience: 10 → 15
#       Gives ReduceLROnPlateau (patience=5) room to fire twice before
#       early stopping terminates the run.
# =============================================================================

experiment_name: "unnamed_experiment"
seed: 42
num_classes: 35

training:
  batch_size: 32
  epochs: 80
  learning_rate: 0.001
  early_stopping_patience: 15
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

        # ──────────────────────────────────────────────────────────────────
        # model/dense.yaml — feedforward baseline
        # Flatten destroys temporal structure — proves LSTMs are necessary.
        # Dense(512) first layer; recurrent_dropout=0.0 (no recurrent connections).
        # Stage 5: recurrent_dropout field added (required by ModelConfig).
        # ──────────────────────────────────────────────────────────────────
        "model/dense.yaml": """
# Dense feedforward baseline — non-temporal, proves sequence modelling matters.
# Expected accuracy: 40–55% (significantly below LSTM/GRU at 60–70%).
# hidden_units=512 is the first Dense layer cap (build_dense() uses min(512, T*D//2)).
# recurrent_dropout=0.0: Dense has no recurrent connections; value is ignored
# by build_dense() but must be present for ModelConfig validation.
model:
  name: dense
  hidden_units: 512
  num_layers: 2
  dropout: 0.3
  recurrent_dropout: 0.0
  bidirectional: false
  dense_units: 128
  activation: relu
""",

        # ──────────────────────────────────────────────────────────────────
        # model/lstm.yaml — primary ablation architecture
        # Groups 2, 3, 4 all use LSTM so only one variable changes per group.
        # Stage 5: hidden_units 128→64 (ablation default), recurrent_dropout added.
        # Champion run uses hidden_units=128 via run_all_experiments.py override.
        # ──────────────────────────────────────────────────────────────────
        "model/lstm.yaml": """
# LSTM — primary ablation architecture (Groups 2, 3, 4 fix architecture to LSTM).
# hidden_units=64: ablation default. Champion run uses 128 via override.
# recurrent_dropout=0.1: lighter than dropout=0.3 to avoid gradient instability
# in stacked recurrent networks on short sequences. Sets >0 disables CuDNN
# fast-path in TF 2.13 — not a concern for CPU training on WLASL (236 clips).
model:
  name: lstm
  hidden_units: 64
  num_layers: 2
  dropout: 0.3
  recurrent_dropout: 0.1
  bidirectional: false
  dense_units: 64
  activation: relu
""",

        # ──────────────────────────────────────────────────────────────────
        # model/gru.yaml — streamlined alternative to LSTM
        # GRU: fewer parameters per unit (~1/3 fewer), often comparable accuracy.
        # Stage 5: hidden_units 128→64, recurrent_dropout added.
        # ──────────────────────────────────────────────────────────────────
        "model/gru.yaml": """
# GRU — streamlined alternative to LSTM.
# Fewer parameters per unit (2 gates vs LSTM's 3) → lower latency.
# Expected: within 1–3pp of LSTM accuracy; preferred deployment candidate
# if accuracy matches (smaller model, faster TFLite inference on Android).
# hidden_units=64: ablation default (same as LSTM for fair comparison).
model:
  name: gru
  hidden_units: 64
  num_layers: 2
  dropout: 0.3
  recurrent_dropout: 0.1
  bidirectional: false
  dense_units: 64
  activation: relu
""",

        # ──────────────────────────────────────────────────────────────────
        # model/bilstm.yaml — champion model candidate
        # BiLSTM reads forward + backward: sign resolution as discriminative
        # as onset. build_bilstm() uses hidden_units // 2 per direction:
        #   hidden_units=64  → 32 units/direction, concat output=64  (ablation)
        #   hidden_units=128 → 64 units/direction, concat output=128 (champion)
        # Stage 5: hidden_units 128→64 (ablation default), recurrent_dropout added.
        # Champion run uses 128 (→ 64/dir, output=128) via override.
        # ──────────────────────────────────────────────────────────────────
        "model/bilstm.yaml": """
# BiLSTM — champion model candidate.
# Bidirectional: each timestep sees past + future context.
# Sign resolution (how a sign ends) is as discriminative as onset.
# CRITICAL: hidden_units MUST be even — build_bilstm() uses hidden_units // 2
# per direction. With merge_mode="concat", output width = 2 × (hidden_units // 2).
#
# Ablation default (this file):
#   hidden_units=64 → 32 units/direction → concat output=64 (matches LSTM width)
# Champion run (override via run_all_experiments.py or best_model.yaml):
#   hidden_units=128 → 64 units/direction → concat output=128
#
# The handoff document comment "hidden_units: 64 # per direction; total=128"
# describes the CHAMPION config (hidden_units=128), not this ablation YAML.
model:
  name: bilstm
  hidden_units: 64
  num_layers: 2
  dropout: 0.3
  recurrent_dropout: 0.1
  bidirectional: true
  dense_units: 64
  activation: relu
""",

        # ──────────────────────────────────────────────────────────────────
        # data/seq20.yaml – seq20.yaml through seq100.yaml
        # All data configs are unchanged from Stage 4.
        # ──────────────────────────────────────────────────────────────────
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

        # ──────────────────────────────────────────────────────────────────
        # augmentation configs — unchanged from Stage 4
        # ──────────────────────────────────────────────────────────────────
        "augmentation/none.yaml": """
# No augmentation — used for all baseline comparison experiments (Group 1).
# augmentation.enabled=False: AugmentationPipeline.__call__() returns
# arr.astype(float32, copy=True) immediately without running any transform.
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
        "augmentation/temporal.yaml": """
# Temporal augmentation only — Group 2 ablation candidate.
# temporal_jitter: zero-fills randomly selected frames in place.
# speed_jitter: resamples clip at rate in [0.7, 1.3] with zero-aware interp.
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
# Full spatial + temporal augmentation — expected Group 2 winner.
# gaussian_noise_detected_only=true: preserves semantic zero-fill for
# one-handed signs (LH 70.18% missing rate is signal, not noise).
# spatial_flip enforced at clip level via data.flip_min_hand_presence=0.30.
# rotation_deg=5.0: ±5° wrist-relative rotation (post-normalisation only).
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

        # ──────────────────────────────────────────────────────────────────
        # experiment configs
        # baseline, ablation_* unchanged from Stage 4.
        # best_model updated: primary/secondary/deployment_metric added.
        # ──────────────────────────────────────────────────────────────────
        "experiment/baseline.yaml": """
# Group 1: Architecture comparison — isolates model type as the single variable.
# Fixed: seq60, augmentation=none, seed=42.
# Runs: dense_baseline, lstm_baseline, gru_baseline, bilstm_baseline.
experiment_name: baseline_architecture_comparison
seed: 42
""",
        "experiment/ablation_augmentation.yaml": """
# Group 2: Augmentation ablation — fixed LSTM + seq60, varies augmentation.
# Runs: lstm_no_aug, lstm_temporal_aug, lstm_spatial_temporal_aug.
experiment_name: ablation_augmentation
seed: 42
""",
        "experiment/ablation_sequence.yaml": """
# Group 3: Sequence length ablation — fixed LSTM + best augmentation from Group 2.
# Sequence lengths: {20, 30, 40, 60, 80, 100}.
# Run order: seq60 first (sanity check), seq80 second (highest expected gain —
# P75=84 frames, P90=95 frames from Notebook 03; 97% truncation at seq60).
experiment_name: ablation_sequence_length
seed: 42
""",
        "experiment/ablation_landmarks.yaml": """
# Group 4: Landmark configuration ablation.
# Fixed: LSTM, best augmentation (Group 2), best seq_len (Group 3).
# Configs: hands_only (Fisher=0.810), pose_only (Fisher=0.218), full (Fisher=0.549).
# Run hands_only first: highest Fisher ratio, most likely to outperform full.
experiment_name: ablation_landmark_config
seed: 42
""",
        "experiment/best_model.yaml": """
# Champion model config — used for final training and TFLite export.
# Architecture: bilstm (or Group 1 winner). Settings from Groups 2–4 results.
# Stage 5 addition: primary_metric, secondary_metric, deployment_metric.
# These are used by run_all_experiments.py for group-level champion selection
# and logged to MLflow for experiment documentation.
#
# hidden_units override:
#   Ablation runs use hidden_units=64 (from model/bilstm.yaml).
#   Champion run uses 128 (64/direction for BiLSTM) for higher capacity.
#   Apply via: overrides={"model.hidden_units": 128} in load_config()
#   OR add "model:\n  hidden_units: 128" directly here.
#
# primary_metric: val_macro_f1
#   With 21 singleton val classes, val_accuracy is unreliable as a selection
#   criterion. val_macro_f1 (sklearn, zero_division=0) is the primary metric
#   for all Stage 5 checkpoint and group-level selection decisions.

experiment_name: best_model_bilstm_champion
seed: 42

primary_metric: val_macro_f1
secondary_metric: val_acc
deployment_metric: "val_macro_f1 / median_latency_ms"

training:
  class_weight_balancing: true
  epochs: 100
  early_stopping_patience: 20
""",
    }

    written  = 0
    skipped  = 0
    replaced = 0

    for filename, content in configs.items():
        filepath = root / filename
        stripped = content.strip() + "\n"

        if filepath.exists():
            if force:
                filepath.write_text(stripped, encoding="utf-8")
                logger.info(f"Replaced (force=True): {filepath}")
                replaced += 1
            else:
                logger.debug(f"Skipping existing config (force=False): {filepath}")
                skipped += 1
        else:
            filepath.write_text(stripped, encoding="utf-8")
            logger.info(f"Wrote config: {filepath}")
            written += 1

    logger.info(
        f"write_default_configs complete | "
        f"written={written} | replaced={replaced} | skipped={skipped} | "
        f"force={force} | root={root.resolve()}"
    )

    if skipped > 0 and not force:
        logger.info(
            f"{skipped} existing config file(s) were NOT overwritten. "
            "To apply Stage 5 schema updates to all files, re-run with force=True:\n"
            "  python -c \"from src.utils.config import write_default_configs; "
            "write_default_configs(force=True)\""
        )