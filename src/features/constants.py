"""
src/features/constants.py
==========================
Dependency-free constants for the WLASL landmark extraction and feature
engineering pipeline.

This module has **no imports from src/** and no third-party imports. It exists
so that any module — including ``pipelines/run_landmark_extraction.py`` —
can import these constants without triggering MediaPipe, TensorFlow, or any
other heavy dependency. In particular, it breaks the circular-import risk
between the pipeline entry point and the extractor.

Schema versioning
-----------------
``EXTRACTOR_SCHEMA_VERSION`` is the single source of truth for the sidecar
.meta.json schema. Both ``extractor.py`` and ``run_landmark_extraction.py``
import this string and compare it against stored sidecar files to decide
whether a cached .npy can be trusted or must be reprocessed.

Version history
~~~~~~~~~~~~~~~
- "1.0"  : Original schema. Ratio-based skip policy (30% threshold).
- "1.1"  : Added ``decode_failure_frames`` to sidecar and ExtractionResult.
           ``missing_pct`` denominator changed to successfully-decoded frames only.
           Cache-hit status changed from "skipped" to "cached".
- "1.2"  : Replaced ratio-based skip policy with dual-criterion absolute policy.
           Added ``detected_frames`` field to ExtractionResult and sidecar.
           Added ``missing_one_hand_frames`` tracking.
           Skip reasons: "insufficient_detected_frames",
                         "catastrophic_missing_rate",
                         "no_frames_extracted".
           Primary skip criterion: detected_frames < min_detected_frames (15).
           Secondary criterion (catastrophe filter): missing_pct > 0.95.

Skip policy — v1.2 design rationale
-------------------------------------
The Notebook 02 sample-run analysis (208 clips queued, 76% skip rate with the
v1.0/v1.1 30%-ratio threshold) revealed that the ratio-based policy was
fundamentally wrong for WLASL data:

  * WLASL clips from YouTube contain large temporal dead zones — preparation
    movements, idle frames, lead-in/lead-out segments — that inflate the
    ``missing_both`` ratio without reducing training value.

  * Example: video 34824 ("many"), 69 frames, 21 both-hands-missing.
    Ratio = 31.3% → OLD policy: REJECTED.
    Detected frames: 46 → well above the 15-frame minimum → NEW policy: KEPT.

  * One-handed signs naturally have ~50% of the ``missing_left`` rate by design.
    The v1.0 policy conflated semantic absence with detection failure.

The correct question is: "does this clip contain enough usable frames for
the LSTM to learn from?" ``detected_frames`` directly answers this, while
the ratio answers a different (and less useful) question.

With the v1.2 policy and ``min_detected_frames=15``, the expected retention
rate for the full 350-clip WLASL dataset is ~94–97%, recovering the training
data needed to reach the ≥70% validation accuracy target.

Feature vector layout
---------------------
Every per-frame feature vector has exactly FEATURE_SIZE=225 elements:

    Index range   Content
    ──────────────────────────────────────────────────────────────────
    [0   : 63 ]   Left hand  — 21 landmarks × (x, y, z)
    [63  : 126]   Right hand — 21 landmarks × (x, y, z)
    [126 : 225]   Pose       — 33 landmarks × (x, y, z)

Landmark 0 of each hand is the wrist (MediaPipe convention), which is why
wrist-relative normalisation in FeaturePipeline can subtract index [0:3] and
[63:66] respectively.

Zero-fill semantics
-------------------
When MediaPipe fails to detect a component:
  - The corresponding slice in the feature vector is zero-filled.
  - Zero-filled frames are counted in ``missing_*`` statistics.
  - Frames where BOTH hands are absent are counted in ``missing_both_hands``.
  - Decode failures (OpenCV codec errors) are separately tracked in
    ``decode_failure_frames`` and are NOT included in any missing-rate
    denominator (v1.1+ behaviour).

Zero-fill for one-handed signs is semantically informative:
  - The consistently-zero left-hand slice distinguishes one-handed signs
    from two-handed signs.
  - FeaturePipeline MUST NOT apply wrist-relative normalisation to
    zero-filled frames (would overwrite the semantic zero with wrist origin).
  - The detection mask guard in FeaturePipeline enforces this invariant.

Inference alignment
-------------------
``model_complexity``, ``min_detection_confidence``, ``min_tracking_confidence``
must be identical between extraction (Stage 3) and inference (Stage 7).
These values are stored in the .meta.json sidecar and in the model card.

MediaPipe version
-----------------
This project is pinned to mediapipe==0.10.14. The landmark counts
(N_HAND_LANDMARKS=21, N_POSE_LANDMARKS=33) are specific to this version.
Upgrading MediaPipe without re-extracting all landmarks will corrupt the
feature vectors.
"""

# ---------------------------------------------------------------------------
# Extractor schema version
# ---------------------------------------------------------------------------

#: Single source of truth for the sidecar .meta.json schema version.
#: Both extractor.py and run_landmark_extraction.py import this value and
#: compare it against stored sidecar files to detect stale caches.
#:
#: Increment this when the sidecar schema or skip policy changes in a way
#: that makes old cached .npy/.meta.json pairs incompatible.
#: Current version: 1.2 (dual-criterion absolute skip policy, detected_frames field)
EXTRACTOR_SCHEMA_VERSION: str = "1.2"

# ---------------------------------------------------------------------------
# MediaPipe landmark counts (pinned to mediapipe==0.10.14)
# ---------------------------------------------------------------------------

#: MediaPipe hand landmark count (both left and right hands).
#: Includes: wrist (0), thumb (1–4), index (5–8), middle (9–12),
#:           ring (13–16), pinky (17–20). Total: 21.
N_HAND_LANDMARKS: int = 21

#: MediaPipe pose landmark count.
#: Includes nose, eyes, ears, shoulders, elbows, wrists, hips, knees,
#: ankles, heels, foot indices, and others. Total: 33.
N_POSE_LANDMARKS: int = 33

#: Coordinates per landmark: x (normalised), y (normalised), z (depth).
N_COORDS_PER_LANDMARK: int = 3

# ---------------------------------------------------------------------------
# Derived feature dimensions
# ---------------------------------------------------------------------------

#: Feature dimension for one hand: 21 landmarks × 3 coordinates = 63.
N_HAND_FEATURES: int = N_HAND_LANDMARKS * N_COORDS_PER_LANDMARK   # 63

#: Feature dimension for pose: 33 landmarks × 3 coordinates = 99.
N_POSE_FEATURES: int = N_POSE_LANDMARKS * N_COORDS_PER_LANDMARK   # 99

#: Total feature vector size per frame: left hand + right hand + pose = 225.
#: This is the canonical feature size for all stages of the pipeline.
#: Any deviation from 225 indicates a bug in landmark packing.
FEATURE_SIZE: int = N_HAND_FEATURES + N_HAND_FEATURES + N_POSE_FEATURES  # 225

# ---------------------------------------------------------------------------
# Feature vector slice indices
# ---------------------------------------------------------------------------

#: Slice for left hand features within the 225-element feature vector.
#: Contains 21 landmarks × 3 coords = 63 values.
#: Landmark 0 (wrist) is at indices [0:3].
LEFT_HAND_SLICE: slice = slice(0, N_HAND_FEATURES)                         # [0:63]

#: Slice for right hand features within the 225-element feature vector.
#: Contains 21 landmarks × 3 coords = 63 values.
#: Landmark 0 (wrist) is at indices [63:66].
RIGHT_HAND_SLICE: slice = slice(N_HAND_FEATURES, N_HAND_FEATURES * 2)     # [63:126]

#: Slice for pose features within the 225-element feature vector.
#: Contains 33 landmarks × 3 coords = 99 values.
POSE_SLICE: slice = slice(N_HAND_FEATURES * 2, FEATURE_SIZE)              # [126:225]

# Sanity check (evaluated at import time, zero runtime cost after first load)
assert FEATURE_SIZE == 225, (
    f"FEATURE_SIZE constant mismatch: expected 225, got {FEATURE_SIZE}. "
    "Check N_HAND_LANDMARKS, N_POSE_LANDMARKS, N_COORDS_PER_LANDMARK."
)
assert LEFT_HAND_SLICE.stop  == RIGHT_HAND_SLICE.start, (
    "LEFT_HAND_SLICE and RIGHT_HAND_SLICE are not contiguous."
)
assert RIGHT_HAND_SLICE.stop == POSE_SLICE.start, (
    "RIGHT_HAND_SLICE and POSE_SLICE are not contiguous."
)
assert POSE_SLICE.stop == FEATURE_SIZE, (
    "POSE_SLICE does not end at FEATURE_SIZE."
)

# ---------------------------------------------------------------------------
# Wrist landmark indices (for FeaturePipeline normalisation)
# ---------------------------------------------------------------------------

#: Index of the left-wrist landmark within the LEFT_HAND_SLICE.
#: MediaPipe convention: landmark 0 is the wrist for both hands.
#: Absolute index in the 225-vector: LEFT_HAND_WRIST_IDX * N_COORDS_PER_LANDMARK = 0.
LEFT_HAND_WRIST_LANDMARK_IDX: int = 0

#: Absolute start index of the left-wrist (x,y,z) triplet in the 225-vector.
LEFT_WRIST_FEATURE_START: int = (
    LEFT_HAND_SLICE.start
    + LEFT_HAND_WRIST_LANDMARK_IDX * N_COORDS_PER_LANDMARK
)  # = 0

#: Index of the right-wrist landmark within the RIGHT_HAND_SLICE.
RIGHT_HAND_WRIST_LANDMARK_IDX: int = 0

#: Absolute start index of the right-wrist (x,y,z) triplet in the 225-vector.
RIGHT_WRIST_FEATURE_START: int = (
    RIGHT_HAND_SLICE.start
    + RIGHT_HAND_WRIST_LANDMARK_IDX * N_COORDS_PER_LANDMARK
)  # = 63

# ---------------------------------------------------------------------------
# v1.2 Skip policy defaults
# ---------------------------------------------------------------------------

#: Primary skip criterion (v1.2): minimum number of frames where at least
#: one hand must be detected for the clip to be retained.
#:
#: Rationale: 15 is the floor below which there is insufficient temporal
#: context for an LSTM to learn meaningful motion patterns, even at the
#: shortest ablation sequence length (seq_len=20). Clips with fewer than
#: 15 detected frames are genuinely unusable regardless of their total
#: frame count.
#:
#: This value can be overridden per-run via the extractor constructor or
#: via the ``--min-detected-frames`` CLI argument in run_landmark_extraction.py.
MIN_DETECTED_FRAMES_DEFAULT: int = 15

#: Secondary skip criterion (v1.2 catastrophe filter): clips where this
#: fraction or more of successfully-decoded frames have BOTH hands absent
#: are considered genuinely unusable and are discarded.
#:
#: At 0.95, only clips where MediaPipe detected at least one hand in fewer
#: than 5% of decoded frames are rejected. This catches:
#:   - Corrupt video files where the signer is never visible
#:   - Wrong-codec videos that decode visually but show garbage frames
#:   - Clips accidentally included from a different scene / context
#:
#: One-handed signs (e.g. "think", "drink") naturally have ~50% of frames
#: with the non-dominant hand absent, but will never reach 95% both-absent
#: because the dominant hand is reliably detected.
#:
#: This value can be overridden via the ``--max-missing-frame-pct`` CLI
#: argument (which maps to this threshold in v1.2).
MAX_MISSING_PCT_CATASTROPHE: float = 0.95

# ---------------------------------------------------------------------------
# Sequence length defaults (for reference — authoritative value is in configs/)
# ---------------------------------------------------------------------------

#: Default sequence length used by FeaturePipeline when padding/truncating.
#: Stage 5 ablation covers {20, 30, 40, 60}. This default matches base.yaml.
DEFAULT_SEQUENCE_LENGTH: int = 30

#: Sequence lengths included in the Stage 5 ablation.
ABLATION_SEQUENCE_LENGTHS: tuple[int, ...] = (20, 30, 40, 60)

# ---------------------------------------------------------------------------
# Storage conventions
# ---------------------------------------------------------------------------

#: File extension for extracted landmark arrays.
LANDMARK_FILE_EXTENSION: str = ".npy"

#: File extension for per-clip sidecar metadata.
SIDECAR_FILE_EXTENSION: str = ".meta.json"

#: Filename for the per-run landmark inventory CSV.
LANDMARK_INVENTORY_FILENAME: str = "landmark_inventory.csv"

#: Required columns in the landmark inventory CSV (must match extractor.py).
LANDMARK_INVENTORY_COLUMNS: tuple[str, ...] = (
    "video_id",
    "sign_label",
    "split",
    "outcome",
    "num_frames",
    "decode_failure_frames",
    "detected_frames",          # Added in v1.2
    "missing_left_pct",
    "missing_right_pct",
    "missing_pose_pct",
    "missing_both_pct",
    "processing_time_sec",
    "output_path",
    "skip_reason",
    "error_message",
)

# ---------------------------------------------------------------------------
# Health check thresholds
# (calibrated for v1.2 policy and WLASL dataset characteristics)
# ---------------------------------------------------------------------------

#: Policy skip rate that triggers a WARNING in health checks.
#: With the v1.2 dual-criterion policy, expected skip rate for WLASL is
#: ~3–6% (clips with fewer than 15 detected frames or >95% both-absent).
#: A rate above 10% suggests the min_detected_frames threshold may be
#: too aggressive for the video quality in this dataset.
HEALTH_POLICY_SKIP_RATE_WARN: float = 0.10

#: Extraction error rate that triggers a WARNING in health checks.
#: Errors are unexpected exceptions (not skip-policy decisions).
#: Above 5% suggests video file integrity issues in data/raw/.
HEALTH_ERROR_RATE_WARN: float = 0.05

#: Global both-hands-absent rate (over decoded frames) that triggers a WARNING.
#: With the v1.2 catastrophe filter retaining clips up to 95%, the
#: aggregate missing rate may be higher than the v1.1 15% threshold.
#: Calibrated to 35% to avoid false alarms on one-handed sign-heavy splits.
HEALTH_GLOBAL_MISSING_RATE_WARN: float = 0.35

# ---------------------------------------------------------------------------
# MediaPipe configuration (must match between extraction and inference)
# ---------------------------------------------------------------------------

#: MediaPipe Holistic model complexity: 0=lite, 1=full, 2=heavy.
#: Full (1) is the project default — good balance of accuracy and speed.
#: MUST be identical between Stage 3 (extraction) and Stage 7 (inference).
DEFAULT_MODEL_COMPLEXITY: int = 1

#: Default minimum detection confidence for MediaPipe Holistic.
#: MUST be identical between Stage 3 (extraction) and Stage 7 (inference).
DEFAULT_MIN_DETECTION_CONFIDENCE: float = 0.5

#: Default minimum tracking confidence for MediaPipe Holistic.
#: MUST be identical between Stage 3 (extraction) and Stage 7 (inference).
DEFAULT_MIN_TRACKING_CONFIDENCE: float = 0.5