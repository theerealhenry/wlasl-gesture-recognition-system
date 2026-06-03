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
Not all submodules exist yet. The import structure below is additive:
new exports are added as each stage is built and never removed, so
callers written against the current API remain valid after future stages
are completed.

    Stage 3  (extractor, constants)      — available now
    Stage 4  (augmentation, pipeline)    — added after Stage 4 is built
"""

# ---------------------------------------------------------------------------
# Stage 3 — Landmark extraction (available now)
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
)

# Extractor — MediaPipe Holistic landmark extraction.
# Heavy imports (cv2, mediapipe) are deferred inside extractor.py; importing
# this module does NOT trigger MediaPipe initialisation.
from src.features.extractor import (
    # Primary extraction class
    LandmarkExtractor,

    # Result data structures
    ExtractionResult,
    ExtractionStats,

    # Batch CSV output helper (used by run_landmark_extraction.py)
    write_landmark_inventory,
)

# ---------------------------------------------------------------------------
# Stage 4 — Feature engineering pipeline (not yet built)
# ---------------------------------------------------------------------------
# When Stage 4 is complete, add:
#
#   from src.features.augmentation import (
#       TemporalAugmentation,
#       SpatialAugmentation,
#       AugmentationPipeline,
#   )
#   from src.features.pipeline import FeaturePipeline
#
# The API surface below is the target; keep this comment as a placeholder
# so the Stage 4 author knows exactly where to add their exports.

# ---------------------------------------------------------------------------
# Package metadata
# ---------------------------------------------------------------------------

__all__ = [
    # Schema
    "EXTRACTOR_SCHEMA_VERSION",

    # Geometry constants
    "N_HAND_LANDMARKS",
    "N_POSE_LANDMARKS",
    "N_COORDS_PER_LANDMARK",
    "N_HAND_FEATURES",
    "N_POSE_FEATURES",
    "FEATURE_SIZE",

    # Slices
    "LEFT_HAND_SLICE",
    "RIGHT_HAND_SLICE",
    "POSE_SLICE",

    # Wrist indices
    "LEFT_HAND_WRIST_LANDMARK_IDX",
    "LEFT_WRIST_FEATURE_START",
    "RIGHT_HAND_WRIST_LANDMARK_IDX",
    "RIGHT_WRIST_FEATURE_START",

    # Skip policy
    "MIN_DETECTED_FRAMES_DEFAULT",
    "MAX_MISSING_PCT_CATASTROPHE",

    # Sequence lengths
    "DEFAULT_SEQUENCE_LENGTH",
    "ABLATION_SEQUENCE_LENGTHS",

    # Storage
    "LANDMARK_FILE_EXTENSION",
    "SIDECAR_FILE_EXTENSION",
    "LANDMARK_INVENTORY_FILENAME",
    "LANDMARK_INVENTORY_COLUMNS",

    # Health thresholds
    "HEALTH_POLICY_SKIP_RATE_WARN",
    "HEALTH_ERROR_RATE_WARN",
    "HEALTH_GLOBAL_MISSING_RATE_WARN",

    # MediaPipe defaults
    "DEFAULT_MODEL_COMPLEXITY",
    "DEFAULT_MIN_DETECTION_CONFIDENCE",
    "DEFAULT_MIN_TRACKING_CONFIDENCE",

    # Extractor
    "LandmarkExtractor",
    "ExtractionResult",
    "ExtractionStats",
    "write_landmark_inventory",
]