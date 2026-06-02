"""
src/features/__init__.py
=========================
Public API for the WLASL features package.

This package implements the three-stage feature representation pipeline:

    Stage 3 — Landmark Extraction (extractor.py)
        MediaPipe Holistic → raw (num_frames, 225) .npy arrays
        225 = 63 left_hand + 63 right_hand + 99 pose per frame

    Stage 4 — Feature Engineering (pipeline.py, augmentation.py)
        Normalisation → augmentation (train only) → pad/truncate → float32 tensor
        FeaturePipeline is the single class used at both training and inference time

Architecture rationale
----------------------
The deliberate split between extraction (Stage 3) and feature engineering
(Stage 4) is a key design decision:

- Extraction produces raw, unpadded arrays indexed to the clip's actual frame
  count. These are written once to disk and never recomputed.
- Padding/truncation is deferred to FeaturePipeline so the same .npy files
  can serve all sequence-length ablation experiments (seq_len ∈ {20, 30, 40})
  without any re-extraction.
- Normalisation is applied at read time by FeaturePipeline, not baked into
  the .npy files, so normalisation strategies can be changed without disk I/O.

Circular import prevention
--------------------------
Feature layout constants live in ``src.features.constants`` — a standalone,
dependency-free module. Both this ``__init__.py`` and ``extractor.py`` import
from ``constants`` directly, so there is no circular dependency:

    __init__.py  →  constants.py   (safe)
    extractor.py →  constants.py   (safe)

Import surface
--------------
Everything a downstream module needs is importable directly from src.features:

    from src.features import (
        # Layout constants
        FEATURE_SIZE, LEFT_HAND_SLICE, RIGHT_HAND_SLICE, POSE_SLICE,
        N_HAND_LANDMARKS, N_POSE_LANDMARKS, N_COORDS_PER_LANDMARK,
        N_HAND_FEATURES, N_POSE_FEATURES,
        WRIST_LANDMARK_INDEX, EXTRACTOR_SCHEMA_VERSION,
        # Stage 3
        LandmarkExtractor, ExtractionResult, ExtractionStats,
    )

FeaturePipeline and augmentation utilities are exported here once Stage 4
is implemented. Their symbols are pre-declared in __all__ so imports fail
fast with a clear ImportError rather than a silent AttributeError.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Feature layout constants — re-exported from the dependency-free constants
# module so that downstream code can do ``from src.features import FEATURE_SIZE``
# without worrying about which sub-module owns the definition.
# ---------------------------------------------------------------------------

from src.features.constants import (  # noqa: F401
    N_HAND_LANDMARKS,
    N_POSE_LANDMARKS,
    N_COORDS_PER_LANDMARK,
    N_HAND_FEATURES,
    N_POSE_FEATURES,
    FEATURE_SIZE,
    LEFT_HAND_SLICE,
    RIGHT_HAND_SLICE,
    POSE_SLICE,
    WRIST_LANDMARK_INDEX,
    NOSE_LANDMARK_INDEX,
    EXTRACTOR_SCHEMA_VERSION,
)

# ---------------------------------------------------------------------------
# Stage 3 exports — LandmarkExtractor
# (Imported after constants to guarantee no circular dependency)
# ---------------------------------------------------------------------------

from src.features.extractor import (  # noqa: F401
    LandmarkExtractor,
    ExtractionResult,
    ExtractionStats,
)

# ---------------------------------------------------------------------------
# Stage 4 exports — FeaturePipeline, augmentation
# Lazy import: raises ImportError with a clear, actionable message if the
# Stage 4 modules have not been built yet, rather than failing silently.
# ---------------------------------------------------------------------------

def __getattr__(name: str):
    """
    Lazy imports for Stage 4 symbols.

    Raises ImportError with a clear message if Stage 4 has not been built.
    """
    _stage4_symbols = {
        "FeaturePipeline":       "src.features.pipeline",
        "AugmentationPipeline":  "src.features.augmentation",
        "apply_temporal_jitter": "src.features.augmentation",
        "apply_speed_jitter":    "src.features.augmentation",
        "apply_spatial_flip":    "src.features.augmentation",
        "apply_gaussian_noise":  "src.features.augmentation",
        "apply_rotation":        "src.features.augmentation",
    }

    if name in _stage4_symbols:
        module_path = _stage4_symbols[name]
        raise ImportError(
            f"'{name}' is a Stage 4 symbol from '{module_path}'. "
            "Build src/features/pipeline.py and src/features/augmentation.py "
            "before importing this symbol. Stage 3 (landmark extraction) "
            "can be used independently without Stage 4."
        )

    raise AttributeError(f"module 'src.features' has no attribute '{name}'")


# ---------------------------------------------------------------------------
# Public API declaration
# ---------------------------------------------------------------------------

__all__ = [
    # Feature layout constants
    "FEATURE_SIZE",
    "N_HAND_LANDMARKS",
    "N_POSE_LANDMARKS",
    "N_COORDS_PER_LANDMARK",
    "N_HAND_FEATURES",
    "N_POSE_FEATURES",
    "LEFT_HAND_SLICE",
    "RIGHT_HAND_SLICE",
    "POSE_SLICE",
    "WRIST_LANDMARK_INDEX",
    "NOSE_LANDMARK_INDEX",
    "EXTRACTOR_SCHEMA_VERSION",

    # Stage 3 — Landmark Extraction
    "LandmarkExtractor",
    "ExtractionResult",
    "ExtractionStats",

    # Stage 4 — Feature Engineering (available once pipeline.py is built)
    "FeaturePipeline",
    "AugmentationPipeline",
    "apply_temporal_jitter",
    "apply_speed_jitter",
    "apply_spatial_flip",
    "apply_gaussian_noise",
    "apply_rotation",
]