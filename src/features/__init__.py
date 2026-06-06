"""
src/features/__init__.py
=========================
Public API for the WLASL feature engineering package.

Import surface
--------------
External code (notebooks, pipeline entry points, tests) should import from
this package rather than reaching into submodules directly. This keeps the
internal module structure free to change without breaking callers.

Stage availability
------------------
    Stage 3  (extractor, constants)                        — available
    Stage 4  (augmentation, pipeline, dataset)             — available
"""

# ---------------------------------------------------------------------------
# Stage 3 — Landmark extraction
# ---------------------------------------------------------------------------

# Constants — always import first; they have no dependencies and are needed
# by both extractor.py and run_landmark_extraction.py.
from src.features.constants import (
    # Schema version
    EXTRACTOR_SCHEMA_VERSION,

    # MediaPipe landmark geometry
    N_HAND_LANDMARKS,
    N_POSE_LANDMARKS,
    N_COORDS_PER_LANDMARK,

    # Derived feature dimensions
    N_HAND_FEATURES,
    N_POSE_FEATURES,
    FEATURE_SIZE,

    # Feature vector slices
    LEFT_HAND_SLICE,
    RIGHT_HAND_SLICE,
    POSE_SLICE,

    # Wrist indices (used by FeaturePipeline for normalisation)
    LEFT_HAND_WRIST_LANDMARK_IDX,
    LEFT_WRIST_FEATURE_START,
    RIGHT_HAND_WRIST_LANDMARK_IDX,
    RIGHT_WRIST_FEATURE_START,

    # v1.2 skip policy defaults
    MIN_DETECTED_FRAMES_DEFAULT,
    MAX_MISSING_PCT_CATASTROPHE,

    # Sequence length defaults
    DEFAULT_SEQUENCE_LENGTH,
    ABLATION_SEQUENCE_LENGTHS,

    # Storage conventions
    LANDMARK_FILE_EXTENSION,
    SIDECAR_FILE_EXTENSION,
    LANDMARK_INVENTORY_FILENAME,
    LANDMARK_INVENTORY_COLUMNS,

    # Health check thresholds
    HEALTH_POLICY_SKIP_RATE_WARN,
    HEALTH_ERROR_RATE_WARN,
    HEALTH_GLOBAL_MISSING_RATE_WARN,

    # MediaPipe configuration defaults
    DEFAULT_MODEL_COMPLEXITY,
    DEFAULT_MIN_DETECTION_CONFIDENCE,
    DEFAULT_MIN_TRACKING_CONFIDENCE,

    # Stage 4 — Feature pipeline constants
    Z_COORD_CLIP_DEFAULT,
    FLIP_MIN_HAND_PRESENCE_DEFAULT,
    AUGMENTATION_NOISE_STD_DEFAULT,
    AUGMENTATION_ROTATION_DEG_DEFAULT,
    AUGMENTATION_FRAME_DROP_PROB_DEFAULT,
    AUGMENTATION_SPEED_RANGE,
    TRUNCATION_STRATEGY,
    TRUNCATION_WARN_FRACTION,
    LANDMARK_CONFIGS,
    MIN_USABLE_DETECTED_FRAMES,
)

# Extractor — MediaPipe Holistic landmark extraction.
# Heavy imports (cv2, mediapipe) are deferred inside extractor.py; importing
# this module does NOT trigger MediaPipe initialisation.
from src.features.extractor import (
    LandmarkExtractor,
    ExtractionResult,
    ExtractionStats,
    write_landmark_inventory,
)

# ---------------------------------------------------------------------------
# Stage 4 — Feature engineering pipeline
# ---------------------------------------------------------------------------

from src.features.augmentation import (
    TemporalAugmenter,
    SpatialAugmenter,
    AugmentationPipeline,
)

from src.features.pipeline import (
    FeaturePipeline,
)

from src.features.dataset import (
    GestureDataset,
)

# ---------------------------------------------------------------------------
# Package metadata
# ---------------------------------------------------------------------------

__all__ = [
    # ── Stage 3: Schema ───────────────────────────────────────────────────
    "EXTRACTOR_SCHEMA_VERSION",

    # ── Stage 3: Geometry constants ───────────────────────────────────────
    "N_HAND_LANDMARKS",
    "N_POSE_LANDMARKS",
    "N_COORDS_PER_LANDMARK",
    "N_HAND_FEATURES",
    "N_POSE_FEATURES",
    "FEATURE_SIZE",

    # ── Stage 3: Feature vector slices ────────────────────────────────────
    "LEFT_HAND_SLICE",
    "RIGHT_HAND_SLICE",
    "POSE_SLICE",

    # ── Stage 3: Wrist indices ────────────────────────────────────────────
    "LEFT_HAND_WRIST_LANDMARK_IDX",
    "LEFT_WRIST_FEATURE_START",
    "RIGHT_HAND_WRIST_LANDMARK_IDX",
    "RIGHT_WRIST_FEATURE_START",

    # ── Stage 3: Skip policy ──────────────────────────────────────────────
    "MIN_DETECTED_FRAMES_DEFAULT",
    "MAX_MISSING_PCT_CATASTROPHE",

    # ── Stage 3: Sequence lengths ─────────────────────────────────────────
    "DEFAULT_SEQUENCE_LENGTH",
    "ABLATION_SEQUENCE_LENGTHS",

    # ── Stage 3: Storage conventions ──────────────────────────────────────
    "LANDMARK_FILE_EXTENSION",
    "SIDECAR_FILE_EXTENSION",
    "LANDMARK_INVENTORY_FILENAME",
    "LANDMARK_INVENTORY_COLUMNS",

    # ── Stage 3: Health check thresholds ──────────────────────────────────
    "HEALTH_POLICY_SKIP_RATE_WARN",
    "HEALTH_ERROR_RATE_WARN",
    "HEALTH_GLOBAL_MISSING_RATE_WARN",

    # ── Stage 3: MediaPipe defaults ───────────────────────────────────────
    "DEFAULT_MODEL_COMPLEXITY",
    "DEFAULT_MIN_DETECTION_CONFIDENCE",
    "DEFAULT_MIN_TRACKING_CONFIDENCE",

    # ── Stage 3: Extractor classes ────────────────────────────────────────
    "LandmarkExtractor",
    "ExtractionResult",
    "ExtractionStats",
    "write_landmark_inventory",

    # ── Stage 4: Feature pipeline constants ───────────────────────────────
    "Z_COORD_CLIP_DEFAULT",
    "FLIP_MIN_HAND_PRESENCE_DEFAULT",
    "AUGMENTATION_NOISE_STD_DEFAULT",
    "AUGMENTATION_ROTATION_DEG_DEFAULT",
    "AUGMENTATION_FRAME_DROP_PROB_DEFAULT",
    "AUGMENTATION_SPEED_RANGE",
    "TRUNCATION_STRATEGY",
    "TRUNCATION_WARN_FRACTION",
    "LANDMARK_CONFIGS",
    "MIN_USABLE_DETECTED_FRAMES",

    # ── Stage 4: Augmentation classes ─────────────────────────────────────
    "TemporalAugmenter",
    "SpatialAugmenter",
    "AugmentationPipeline",

    # ── Stage 4: Pipeline class ───────────────────────────────────────────
    "FeaturePipeline",

    # ── Stage 4: Dataset class ────────────────────────────────────────────
    "GestureDataset",
]