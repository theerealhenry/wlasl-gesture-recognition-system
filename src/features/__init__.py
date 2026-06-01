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

Import surface
--------------
Everything a downstream module needs is importable directly from src.features:

    from src.features import (
        LandmarkExtractor,
        ExtractionResult,
        ExtractionStats,
        FEATURE_SIZE,
        LEFT_HAND_SLICE,
        RIGHT_HAND_SLICE,
        POSE_SLICE,
        N_HAND_LANDMARKS,
        N_POSE_LANDMARKS,
    )

FeaturePipeline and augmentation utilities are exported here once Stage 4
is implemented (augmentation.py, pipeline.py). Their symbols are pre-declared
as __all__ entries so imports fail fast with a clear error rather than
silently returning None.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Feature layout constants — single source of truth
#
# These constants are imported throughout the codebase wherever code needs to
# index into the 225-element feature vector. Using named slices prevents
# off-by-one errors and makes the code self-documenting.
# ---------------------------------------------------------------------------

#: Number of landmarks per hand (MediaPipe Hands model)
N_HAND_LANDMARKS: int = 21

#: Number of pose landmarks (MediaPipe Pose model)
N_POSE_LANDMARKS: int = 33

#: Values per landmark (x, y, z)
N_COORDS_PER_LANDMARK: int = 3

#: Flattened feature count for one hand (21 × 3 = 63)
N_HAND_FEATURES: int = N_HAND_LANDMARKS * N_COORDS_PER_LANDMARK   # 63

#: Flattened feature count for pose (33 × 3 = 99)
N_POSE_FEATURES: int = N_POSE_LANDMARKS * N_COORDS_PER_LANDMARK   # 99

#: Total feature vector length per frame (63 + 63 + 99 = 225)
FEATURE_SIZE: int = N_HAND_FEATURES + N_HAND_FEATURES + N_POSE_FEATURES  # 225

# Slice objects for indexing into the 225-element feature vector.
# Usage:  frame_vec[LEFT_HAND_SLICE]  →  shape (63,)
#         frame_vec[RIGHT_HAND_SLICE] →  shape (63,)
#         frame_vec[POSE_SLICE]       →  shape (99,)

LEFT_HAND_SLICE  = slice(0,                  N_HAND_FEATURES)                       # [0:63]
RIGHT_HAND_SLICE = slice(N_HAND_FEATURES,    N_HAND_FEATURES * 2)                   # [63:126]
POSE_SLICE       = slice(N_HAND_FEATURES * 2, N_HAND_FEATURES * 2 + N_POSE_FEATURES)  # [126:225]

# Wrist landmark index within the MediaPipe Hands landmark list.
# Used by FeaturePipeline for wrist-relative normalisation.
WRIST_LANDMARK_INDEX: int = 0

# Pose landmark index for the nose (used as a pose anchor if needed).
NOSE_LANDMARK_INDEX: int = 0


# ---------------------------------------------------------------------------
# Stage 3 exports — LandmarkExtractor
# ---------------------------------------------------------------------------

from src.features.extractor import (  # noqa: E402
    LandmarkExtractor,
    ExtractionResult,
    ExtractionStats,
)

# ---------------------------------------------------------------------------
# Stage 4 exports — FeaturePipeline, augmentation
# (imported lazily so Stage 3 can be used without Stage 4 being built yet)
# ---------------------------------------------------------------------------

def __getattr__(name: str):
    """
    Lazy imports for Stage 4 symbols.

    Raises ImportError with a clear message if Stage 4 has not been built.
    This prevents silent AttributeError failures and provides actionable
    guidance to the developer.
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