"""
src/features/constants.py
==========================
Single source of truth for all feature layout constants in the WLASL
gesture recognition pipeline.

Why a separate module
---------------------
Both ``src/features/__init__.py`` and ``src/features/extractor.py`` need
these constants. Defining them in ``__init__.py`` and importing from there
inside ``extractor.py`` creates a circular dependency:

    __init__.py  →  imports extractor.py
    extractor.py →  imports __init__.py   (circular)

Placing the constants in this standalone, dependency-free module breaks
the cycle cleanly. Both ``__init__.py`` and ``extractor.py`` import from
``src.features.constants`` — no circularity.

Layout specification
--------------------
The 225-element feature vector per frame is ordered as follows:

    Indices [0  : 63 ] — Left hand  (21 landmarks × x, y, z)
    Indices [63 : 126] — Right hand (21 landmarks × x, y, z)
    Indices [126: 225] — Pose       (33 landmarks × x, y, z)

This ordering is fixed for the lifetime of the v1 label map and must not
be changed without re-extracting all .npy files and bumping the extractor
version identifier in the sidecar metadata format.

Coordinate convention
---------------------
All coordinates are MediaPipe's normalised screen-space values:
  x ∈ [0, 1]          — horizontal, left-to-right
  y ∈ [0, 1]          — vertical, top-to-bottom
  z ∈ [-0.2,  0.2]    — estimated depth (relative to wrist for hands,
                         relative to hip midpoint for pose)

Wrist landmark for normalisation
---------------------------------
``WRIST_LANDMARK_INDEX = 0`` is the MediaPipe convention: the first
landmark in the Hands model is always the wrist. FeaturePipeline (Stage 4)
subtracts the wrist (x, y, z) from all 21 hand landmarks to produce
wrist-relative coordinates. The absolute position [0:3] of the hand slice
therefore encodes the wrist's screen position before normalisation and is
exactly zero after normalisation.
"""

# ---------------------------------------------------------------------------
# Landmark counts (MediaPipe model specification)
# ---------------------------------------------------------------------------

#: Number of landmarks per hand (MediaPipe Hands model)
N_HAND_LANDMARKS: int = 21

#: Number of pose landmarks (MediaPipe BlazePose model)
N_POSE_LANDMARKS: int = 33

#: Coordinate dimensions per landmark (x, y, z)
N_COORDS_PER_LANDMARK: int = 3

# ---------------------------------------------------------------------------
# Derived feature counts
# ---------------------------------------------------------------------------

#: Flattened feature count for one hand  (21 × 3 = 63)
N_HAND_FEATURES: int = N_HAND_LANDMARKS * N_COORDS_PER_LANDMARK   # 63

#: Flattened feature count for pose      (33 × 3 = 99)
N_POSE_FEATURES: int = N_POSE_LANDMARKS * N_COORDS_PER_LANDMARK   # 99

#: Total feature vector length per frame (63 + 63 + 99 = 225)
FEATURE_SIZE: int = N_HAND_FEATURES + N_HAND_FEATURES + N_POSE_FEATURES  # 225

# ---------------------------------------------------------------------------
# Slice objects — index into the 225-element feature vector
# ---------------------------------------------------------------------------
# Usage:
#   frame_vec[LEFT_HAND_SLICE]   → shape (63,)  left hand landmarks
#   frame_vec[RIGHT_HAND_SLICE]  → shape (63,)  right hand landmarks
#   frame_vec[POSE_SLICE]        → shape (99,)  pose landmarks

LEFT_HAND_SLICE  = slice(0,                     N_HAND_FEATURES)                        # [0:63]
RIGHT_HAND_SLICE = slice(N_HAND_FEATURES,        N_HAND_FEATURES * 2)                   # [63:126]
POSE_SLICE       = slice(N_HAND_FEATURES * 2,    N_HAND_FEATURES * 2 + N_POSE_FEATURES) # [126:225]

# ---------------------------------------------------------------------------
# Landmark index conventions
# ---------------------------------------------------------------------------

#: Index of the wrist landmark within the MediaPipe Hands landmark list.
#: Used by FeaturePipeline for wrist-relative normalisation. The wrist
#: (x, y, z) occupies positions [0:3] of each 63-element hand slice.
WRIST_LANDMARK_INDEX: int = 0

#: Index of the nose landmark within the MediaPipe Pose landmark list.
#: Retained for reference; not currently used in the feature pipeline.
NOSE_LANDMARK_INDEX: int = 0

# ---------------------------------------------------------------------------
# Extractor versioning
# ---------------------------------------------------------------------------

#: Semantic version of the extraction schema. Written into every sidecar
#: .meta.json file so that stale cached files can be detected when the
#: feature layout changes (e.g. adding visibility scores, swapping left/right
#: hand order, or switching from Holistic to a newer MediaPipe model).
EXTRACTOR_SCHEMA_VERSION: str = "1.1"