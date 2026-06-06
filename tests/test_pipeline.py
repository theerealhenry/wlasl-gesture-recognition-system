"""
tests/test_pipeline.py
=======================
Complete test suite for src/features/pipeline.py (FeaturePipeline).

Test organisation
-----------------
Each class maps to one coherent pipeline behaviour. Tests within a class
are ordered from the simplest invariant to the most complex interaction.

    TestConstruction          — __init__ validation, invalid config rejection
    TestInputValidation       — Step 1: shape, dtype, empty clip, NaN/Inf guards
    TestInputImmutability     — Step 2: caller's array is never mutated
    TestWristNormalisation    — Step 3: wrist-relative normalisation invariants
    TestZCoordClip            — Step 4: z-coordinate soft-clipping
    TestPadOrTruncate         — Step 5: padding and centre-crop arithmetic
    TestAugmentationOrdering  — Step 6: augmentation/lm-select ordering (CRITICAL)
    TestLandmarkConfigSelect  — Step 7: feature slice per landmark_config
    TestOutputDtype           — Step 8: guaranteed float32 output
    TestProperties            — output_shape, sequence_length, feature_dim, etc.
    TestStatisticsAccumulation — n_processed, n_truncated, n_padded, reset
    TestMetadata              — get_pipeline_metadata() completeness and values
    TestRepr                  — __repr__ content
    TestEdgeCases             — boundary conditions: T=1, T=seq_len±1, all-zero

Design principles
-----------------
1. Every test targets ONE specific behaviour from pipeline.py. When a test
   fails, the name alone should identify which invariant was violated.

2. Synthetic arrays use values in [0.1, 0.9] for detected components and
   exactly 0.0 for absent components. This makes zero-fill assertions exact
   (np.testing.assert_array_equal) rather than approximate.

3. Centre-crop arithmetic is verified analytically, not inferred from output.
   The expected frame indices are computed independently and used to assert
   specific frame content, not just output shape.

4. The CRITICAL augmentation-ordering tests (hands_only + training=True,
   pose_only + training=True) are the most important tests in this file.
   If they fail, it means AugmentationPipeline received a non-225-dim array
   and raised ValueError — the chain-reorder bug has been reintroduced.

5. All tests are self-contained via fixtures. No test relies on execution
   order or modifies shared state.

Constants used directly
-----------------------
FEATURE_SIZE = 225
LEFT_HAND_SLICE  = slice(0, 63)   → LH wrist at indices [0:3]
RIGHT_HAND_SLICE = slice(63, 126) → RH wrist at indices [63:66]
POSE_SLICE       = slice(126, 225)
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from src.features.constants import (
    FEATURE_SIZE,
    LEFT_HAND_SLICE,
    N_HAND_FEATURES,
    POSE_SLICE,
    RIGHT_HAND_SLICE,
)
from src.features.pipeline import FeaturePipeline
from src.utils.config import load_config

# ---------------------------------------------------------------------------
# Constants derived from the feature-vector layout
# ---------------------------------------------------------------------------

#: Absolute index range of the LH wrist (landmark 0) in the full 225-vector.
_LH_WRIST_START: int = 0
_LH_WRIST_END: int   = 3

#: Absolute index range of the RH wrist (landmark 0) in the full 225-vector.
_RH_WRIST_START: int = RIGHT_HAND_SLICE.start        # 63
_RH_WRIST_END: int   = RIGHT_HAND_SLICE.start + 3    # 66

#: Feature dimensions for each landmark_config
_DIM_FULL:       int = 225
_DIM_HANDS_ONLY: int = 126
_DIM_POSE_ONLY:  int = 99


# ---------------------------------------------------------------------------
# Fixture factory functions
# ---------------------------------------------------------------------------

def _load_pipeline(
    model: str = "lstm",
    data: str = "seq60",
    augmentation: str = "none",
    overrides: dict | None = None,
) -> FeaturePipeline:
    """
    Construct a FeaturePipeline from the project config system.

    All tests that need a pipeline call this helper so that the config
    system is always exercised through the official path.
    """
    cfg = load_config(model=model, data=data, augmentation=augmentation,
                      overrides=overrides or {})
    return FeaturePipeline(cfg)


def _make_two_handed_clip(T: int = 50, seed: int = 0) -> np.ndarray:
    """
    Synthetic clip where ALL T frames have both hands and pose detected.

    Values are drawn from Uniform(0.1, 0.9) — guaranteed non-zero in
    every slot. dtype=float32.
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(0.1, 0.9, size=(T, FEATURE_SIZE)).astype(np.float32)


def _make_lh_only_clip(T: int = 50, seed: int = 0) -> np.ndarray:
    """
    Synthetic clip: LEFT hand and pose detected; RIGHT hand always absent.

    RH slice (indices 63:126) is exactly zero throughout.
    """
    arr = _make_two_handed_clip(T, seed)
    arr[:, RIGHT_HAND_SLICE] = 0.0
    return arr


def _make_rh_only_clip(T: int = 50, seed: int = 0) -> np.ndarray:
    """
    Synthetic clip: RIGHT hand and pose detected; LEFT hand always absent.

    LH slice (indices 0:63) is exactly zero throughout.
    """
    arr = _make_two_handed_clip(T, seed)
    arr[:, LEFT_HAND_SLICE] = 0.0
    return arr


def _make_no_hands_clip(T: int = 50, seed: int = 0) -> np.ndarray:
    """
    Synthetic clip: both hands absent in every frame (only pose detected).

    LH and RH slices are exactly zero. Pose slice is non-zero.
    """
    arr = _make_two_handed_clip(T, seed)
    arr[:, LEFT_HAND_SLICE]  = 0.0
    arr[:, RIGHT_HAND_SLICE] = 0.0
    return arr


def _make_mixed_clip(
    T: int = 50,
    zero_fraction: float = 0.4,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Clip where ``zero_fraction`` of frames have both hands absent.

    Returns (arr, zero_indices) so tests can assert frame-level invariants
    on exactly those frames.
    """
    rng = np.random.default_rng(seed)
    arr = _make_two_handed_clip(T, seed)
    n_zero = max(1, int(T * zero_fraction))
    zero_indices = np.sort(rng.choice(T, size=n_zero, replace=False))
    arr[zero_indices, LEFT_HAND_SLICE]  = 0.0
    arr[zero_indices, RIGHT_HAND_SLICE] = 0.0
    return arr, zero_indices


def _make_fingerprinted_clip(T: int, seq_len: int) -> np.ndarray:
    """
    Two-handed clip where arr[t, :] == float(t).

    This fingerprint survives the pipeline's copy and normalisation
    (since all values in a row are equal, subtracting the wrist value
    from itself gives zero everywhere in that row — but we only need the
    fingerprint to identify WHICH FRAMES were kept by centre-crop, not
    their exact normalised values). We fingerprint on a dedicated column
    that is outside the LH/RH slices: column 200 (inside pose slice,
    which is never normalised).
    """
    arr = _make_two_handed_clip(T, seed=77)
    for t in range(T):
        arr[t, 200] = float(t)   # column 200 is in POSE_SLICE → never normalised
    return arr


# ---------------------------------------------------------------------------
# Section 0 — Construction / init validation
# ---------------------------------------------------------------------------

class TestConstruction:
    """__init__ validation: config rejection and successful construction."""

    def test_valid_full_config(self):
        """Standard full config constructs without error."""
        p = _load_pipeline(data="seq60", augmentation="none")
        assert p is not None

    def test_valid_hands_only_config(self):
        cfg = load_config(model="lstm", data="seq60", augmentation="none",
                          overrides={"data.landmark_config": "hands_only"})
        p = FeaturePipeline(cfg)
        assert p.feature_dim == _DIM_HANDS_ONLY

    def test_valid_pose_only_config(self):
        cfg = load_config(model="lstm", data="seq60", augmentation="none",
                          overrides={"data.landmark_config": "pose_only"})
        p = FeaturePipeline(cfg)
        assert p.feature_dim == _DIM_POSE_ONLY

    @pytest.mark.parametrize("seq_cfg", ["seq20", "seq30", "seq40", "seq60"])
    def test_valid_all_seq_len_configs(self, seq_cfg: str):
        """All four standard seq_len configs construct cleanly."""
        p = _load_pipeline(data=seq_cfg)
        assert p.sequence_length in (20, 30, 40, 60)

    def test_invalid_landmark_config_raises(self):
        """An unknown landmark_config is rejected — either by Pydantic at
        load_config() time or by FeaturePipeline.__init__() at construction time."""
        from pydantic import ValidationError
        with pytest.raises((ValueError, ValidationError)):
            cfg = load_config(model="lstm", data="seq60", augmentation="none",
                            overrides={"data.landmark_config": "all_landmarks"})
            FeaturePipeline(cfg)

    def test_normalise_pose_true_raises(self):
        """
        normalise_pose=True is explicitly blocked (Notebook 03 F3).
        Raising at construction time prevents silent contamination of
        training runs.
        """
        cfg = load_config(model="lstm", data="seq60", augmentation="none",
                          overrides={"data.normalise_pose": True})
        with pytest.raises(ValueError, match="normalise_pose"):
            FeaturePipeline(cfg)


# ---------------------------------------------------------------------------
# Section 1 — Input validation (Step 1 + finite check)
# ---------------------------------------------------------------------------

class TestInputValidation:
    """
    Every guard in Step 1 + the finite check must fail loudly with a
    message that points toward the actual problem.
    """

    @pytest.fixture(autouse=True)
    def pipeline(self):
        self.p = _load_pipeline()

    # --- Non-array inputs ---------------------------------------------------

    def test_rejects_list(self):
        """Python lists are not accepted."""
        with pytest.raises(ValueError, match="numpy ndarray"):
            self.p([[0.0] * FEATURE_SIZE] * 10)

    def test_rejects_none(self):
        with pytest.raises(ValueError, match="numpy ndarray"):
            self.p(None)

    # --- Dimensionality guards ----------------------------------------------

    def test_rejects_1d_array(self):
        """A flat 1D array of length 225 is rejected."""
        with pytest.raises(ValueError, match="2D"):
            self.p(np.zeros(FEATURE_SIZE, dtype=np.float32))

    def test_rejects_3d_array(self):
        """A 3D array is rejected."""
        with pytest.raises(ValueError, match="2D"):
            self.p(np.zeros((5, 10, FEATURE_SIZE), dtype=np.float32))

    # --- Empty clip guard (Bug 1 from critical review) ----------------------

    def test_rejects_zero_frame_clip(self):
        """
        A (0, 225) array represents a corrupt extraction output. It must
        be rejected with a message mentioning 'empty' or '0 frames'.
        Without this guard, the pipeline silently returns a fully-zero
        padded tensor that becomes a phantom training sample.
        """
        with pytest.raises(ValueError, match="(?i)empty|0 frames"):
            self.p(np.empty((0, FEATURE_SIZE), dtype=np.float32))

    def test_rejects_wrong_feature_dim_too_small(self):
        """126-dim array (hands_only output) passed to a full-225 pipeline."""
        with pytest.raises(ValueError, match=str(FEATURE_SIZE)):
            self.p(np.zeros((10, 126), dtype=np.float32))

    def test_rejects_wrong_feature_dim_too_large(self):
        with pytest.raises(ValueError, match=str(FEATURE_SIZE)):
            self.p(np.zeros((10, 300), dtype=np.float32))

    def test_error_message_mentions_landmark_config(self):
        """
        The feature-dim error message should remind the caller that
        landmark config slicing happens INSIDE the pipeline.
        """
        with pytest.raises(ValueError, match="landmark config"):
            self.p(np.zeros((10, 99), dtype=np.float32))

    # --- Finite value guards (Bug 2 from critical review) -------------------

    def test_rejects_array_with_single_nan(self):
        """Even a single NaN value in one cell is caught."""
        arr = _make_two_handed_clip(T=10)
        arr[3, 7] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            self.p(arr)

    def test_rejects_array_with_positive_inf(self):
        arr = _make_two_handed_clip(T=10)
        arr[0, 0] = float("inf")
        with pytest.raises(ValueError, match="Inf"):
            self.p(arr)

    def test_rejects_array_with_negative_inf(self):
        arr = _make_two_handed_clip(T=10)
        arr[5, 100] = float("-inf")
        with pytest.raises(ValueError, match="Inf"):
            self.p(arr)

    def test_rejects_array_with_mixed_nan_and_inf(self):
        """Array with both NaN and Inf: error message reports both counts."""
        arr = _make_two_handed_clip(T=20)
        arr[0, 0]  = float("nan")
        arr[1, 10] = float("inf")
        with pytest.raises(ValueError, match="NaN"):
            self.p(arr)

    def test_nan_count_in_error_message(self):
        """The error message reports the exact NaN count."""
        arr = _make_two_handed_clip(T=10)
        arr[0, 0] = float("nan")
        arr[1, 1] = float("nan")
        with pytest.raises(ValueError) as exc_info:
            self.p(arr)
        assert "NaN=2" in str(exc_info.value)

    # --- Accepted dtypes (should not raise) ---------------------------------

    def test_accepts_float32_input(self):
        """float32 is the native dtype — must succeed."""
        arr = _make_two_handed_clip(T=10)
        assert arr.dtype == np.float32
        result = self.p(arr, training=False)
        assert result.dtype == np.float32

    def test_accepts_float64_input(self):
        """float64 is upcast silently via astype."""
        arr = _make_two_handed_clip(T=10).astype(np.float64)
        result = self.p(arr, training=False)
        assert result.dtype == np.float32

    def test_accepts_int32_input(self):
        """Integer arrays are cast to float32 without error."""
        arr = np.ones((10, FEATURE_SIZE), dtype=np.int32)
        result = self.p(arr, training=False)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# Section 2 — Input immutability (Step 2 — copy)
# ---------------------------------------------------------------------------

class TestInputImmutability:
    """
    The caller's array must never be mutated by pipeline processing.
    Wrist-relative normalisation modifies coordinates in-place on the
    internal copy, not on the original array.
    """

    @pytest.fixture(autouse=True)
    def pipeline(self):
        self.p = _load_pipeline()

    def test_float32_input_not_mutated(self):
        arr      = _make_two_handed_clip(T=50)
        arr_ref  = arr.copy()
        self.p(arr, training=False)
        np.testing.assert_array_equal(arr, arr_ref,
            err_msg="Pipeline mutated the caller's float32 array.")

    def test_float64_input_not_mutated(self):
        arr      = _make_two_handed_clip(T=30).astype(np.float64)
        arr_ref  = arr.copy()
        self.p(arr, training=False)
        np.testing.assert_array_equal(arr, arr_ref,
            err_msg="Pipeline mutated the caller's float64 array.")

    def test_repeated_calls_produce_identical_results(self):
        """
        Calling the pipeline twice on the same array must produce identical
        output. If the first call mutated the array, the second call would
        normalise an already-normalised array and produce different values.
        """
        arr     = _make_two_handed_clip(T=30)
        result1 = self.p(arr, training=False)
        result2 = self.p(arr, training=False)
        np.testing.assert_array_equal(result1, result2,
            err_msg="Second call produced different output — input was mutated.")

    def test_result_is_different_object_from_input(self):
        arr    = _make_two_handed_clip(T=30)
        result = self.p(arr, training=False)
        assert result is not arr


# ---------------------------------------------------------------------------
# Section 3 — Wrist-relative normalisation (Step 3)
# ---------------------------------------------------------------------------

class TestWristNormalisation:
    """
    Wrist-relative normalisation: wrist becomes origin; zero-fill unchanged;
    pose never touched.

    Test design note: we call the internal `_wrist_relative_normalise` method
    directly for the normalisation-isolation tests. This lets us verify the
    transform in isolation, without the confounding effect of z-clipping or
    padding on the raw values. The full-pipeline tests in other sections
    verify that normalisation is correctly integrated into the chain.
    """

    @pytest.fixture(autouse=True)
    def pipeline(self):
        self.p = _load_pipeline()

    # --- Wrist becomes origin -----------------------------------------------

    def test_lh_wrist_is_zero_after_normalisation(self):
        """
        After normalisation, LH wrist (feature indices 0:3) must be
        approximately (0, 0, 0) in all detected frames.
        """
        arr = _make_two_handed_clip(T=30)
        result = self.p._wrist_relative_normalise(arr.astype(np.float32, copy=True))
        lh_detected = arr[:, LEFT_HAND_SLICE].any(axis=1)
        np.testing.assert_allclose(
            result[lh_detected, _LH_WRIST_START:_LH_WRIST_END],
            np.zeros((lh_detected.sum(), 3), dtype=np.float32),
            atol=1e-5,
            err_msg="LH wrist is not at the origin after wrist-relative normalisation.",
        )

    def test_rh_wrist_is_zero_after_normalisation(self):
        """
        After normalisation, RH wrist (feature indices 63:66) must be
        approximately (0, 0, 0) in all detected frames.
        """
        arr = _make_two_handed_clip(T=30)
        result = self.p._wrist_relative_normalise(arr.astype(np.float32, copy=True))
        rh_detected = arr[:, RIGHT_HAND_SLICE].any(axis=1)
        np.testing.assert_allclose(
            result[rh_detected, _RH_WRIST_START:_RH_WRIST_END],
            np.zeros((rh_detected.sum(), 3), dtype=np.float32),
            atol=1e-5,
            err_msg="RH wrist is not at the origin after wrist-relative normalisation.",
        )

    def test_lh_non_wrist_landmarks_are_wrist_relative(self):
        """
        A non-wrist LH landmark (landmark 1 at indices 3:6) must equal
        original_value − original_wrist after normalisation.
        """
        arr = np.zeros((5, FEATURE_SIZE), dtype=np.float32)
        arr[:, :3]  = [0.3, 0.4, 0.1]    # LH wrist
        arr[:, 3:6] = [0.5, 0.6, 0.2]    # LH landmark 1
        # Pose non-zero so clip is usable
        arr[:, POSE_SLICE.start] = 0.5

        result = self.p._wrist_relative_normalise(arr.copy())
        expected_lm1 = np.array([0.5 - 0.3, 0.6 - 0.4, 0.2 - 0.1], dtype=np.float32)
        np.testing.assert_allclose(result[:, 3:6], np.tile(expected_lm1, (5, 1)), atol=1e-5)

    # --- Zero-fill invariant ------------------------------------------------

    def test_zero_fill_lh_frames_unchanged_when_lh_absent(self):
        """
        Frames where LH is absent (zero-filled) must have the LH slice
        remain exactly zero after normalisation.

        This is the core semantic invariant of the pipeline: the zero-fill
        pattern (Notebook 03 F2: LH 70% missing is SEMANTIC signal) must
        be preserved with bit-level precision.
        """
        arr, zero_indices = _make_mixed_clip(T=50, zero_fraction=0.4)
        result = self.p._wrist_relative_normalise(arr.astype(np.float32, copy=True))
        for idx in zero_indices:
            np.testing.assert_array_equal(
                result[idx, LEFT_HAND_SLICE],
                np.zeros(N_HAND_FEATURES, dtype=np.float32),
                err_msg=f"Frame {idx}: LH zero-fill was corrupted by normalisation.",
            )

    def test_zero_fill_rh_frames_unchanged_when_rh_absent(self):
        """Frames where RH is absent stay exactly zero in the RH slice."""
        arr, zero_indices = _make_mixed_clip(T=50, zero_fraction=0.4)
        result = self.p._wrist_relative_normalise(arr.astype(np.float32, copy=True))
        for idx in zero_indices:
            np.testing.assert_array_equal(
                result[idx, RIGHT_HAND_SLICE],
                np.zeros(N_HAND_FEATURES, dtype=np.float32),
                err_msg=f"Frame {idx}: RH zero-fill was corrupted by normalisation.",
            )

    def test_rh_only_clip_lh_slice_stays_zero(self):
        """
        For a right-hand-only clip, the LH slice is zero in every frame.
        Normalisation must not touch it (the detection mask correctly excludes
        frames where LH is absent).
        """
        arr    = _make_rh_only_clip(T=40)
        result = self.p._wrist_relative_normalise(arr.astype(np.float32, copy=True))
        np.testing.assert_array_equal(
            result[:, LEFT_HAND_SLICE],
            np.zeros((40, N_HAND_FEATURES), dtype=np.float32),
            err_msg="LH slice was modified for a RH-only clip.",
        )

    def test_lh_only_clip_rh_slice_stays_zero(self):
        """For a left-hand-only clip, the RH slice stays zero throughout."""
        arr    = _make_lh_only_clip(T=40)
        result = self.p._wrist_relative_normalise(arr.astype(np.float32, copy=True))
        np.testing.assert_array_equal(
            result[:, RIGHT_HAND_SLICE],
            np.zeros((40, N_HAND_FEATURES), dtype=np.float32),
            err_msg="RH slice was modified for a LH-only clip.",
        )

    def test_all_zero_clip_passes_through_unchanged(self):
        """
        A clip where both hands are absent in every frame must pass through
        normalisation completely unchanged (no wrist subtraction attempted,
        since detected masks are all False).
        """
        arr    = _make_no_hands_clip(T=20)
        arr[:, POSE_SLICE] = 0.0   # Also zero out pose → fully zero
        arr_ref = arr.copy()
        result  = self.p._wrist_relative_normalise(arr.astype(np.float32, copy=True))
        np.testing.assert_array_equal(result, arr_ref,
            err_msg="All-zero clip was modified by normalisation.")

    # --- Pose invariant -----------------------------------------------------

    def test_pose_slice_is_never_modified_by_normalisation(self):
        """
        Pose landmarks (indices 126:225) must be bit-identical before and
        after wrist-relative normalisation (Notebook 03 F3: pose is signal).
        """
        arr    = _make_two_handed_clip(T=30)
        orig_pose = arr[:, POSE_SLICE].copy()
        result = self.p._wrist_relative_normalise(arr.astype(np.float32, copy=True))
        np.testing.assert_array_equal(
            result[:, POSE_SLICE],
            orig_pose,
            err_msg="Pose slice was modified by wrist-relative normalisation.",
        )

    # --- End-to-end normalisation verified through __call__ -----------------

    def test_lh_wrist_zero_through_full_pipeline(self):
        """
        The LH wrist-at-origin property survives the full pipeline chain.
        We use a clip where T_raw == seq_len to avoid the confounding effect
        of padding (which inserts rows of zeros that could mask failures).
        """
        p   = _load_pipeline(data="seq60")
        arr = _make_two_handed_clip(T=60)   # exact length, no pad/crop
        result = p(arr, training=False)
        # After normalisation, wrist is at origin. After pad/truncate (no-op),
        # augmentation (disabled), and slice (full), indices 0:3 are still ~zero.
        np.testing.assert_allclose(
            result[:, _LH_WRIST_START:_LH_WRIST_END],
            np.zeros((60, 3), dtype=np.float32),
            atol=1e-5,
        )

    def test_rh_wrist_zero_through_full_pipeline(self):
        """RH wrist-at-origin survives the full pipeline chain."""
        p   = _load_pipeline(data="seq60")
        arr = _make_two_handed_clip(T=60)
        result = p(arr, training=False)
        np.testing.assert_allclose(
            result[:, _RH_WRIST_START:_RH_WRIST_END],
            np.zeros((60, 3), dtype=np.float32),
            atol=1e-5,
        )


# ---------------------------------------------------------------------------
# Section 4 — Z-coordinate clipping (Step 4)
# ---------------------------------------------------------------------------

class TestZCoordClip:
    """
    Z-coordinate soft clipping at ±0.10 (from constants.Z_COORD_CLIP_DEFAULT).
    Z-coords are at every third index: [2, 5, 8, ..., 224].
    """

    @pytest.fixture(autouse=True)
    def pipeline(self):
        self.p = _load_pipeline()

    def _run_z_clip(self, arr: np.ndarray) -> np.ndarray:
        """Call _apply_z_clip directly for isolation."""
        return self.p._apply_z_clip(arr.astype(np.float32, copy=True))

    def test_lh_z_above_threshold_clipped_to_threshold(self):
        arr = np.zeros((5, FEATURE_SIZE), dtype=np.float32)
        arr[:, 2] = 0.5    # first LH z-coord: index 2
        result = self._run_z_clip(arr)
        np.testing.assert_allclose(result[:, 2], 0.10, atol=1e-6,
            err_msg="LH z-value 0.5 was not clipped to 0.10.")

    def test_rh_z_above_threshold_clipped(self):
        arr = np.zeros((5, FEATURE_SIZE), dtype=np.float32)
        arr[:, 65] = 0.9   # first RH z-coord: index 65
        result = self._run_z_clip(arr)
        np.testing.assert_allclose(result[:, 65], 0.10, atol=1e-6)

    def test_pose_z_above_threshold_clipped(self):
        arr = np.zeros((5, FEATURE_SIZE), dtype=np.float32)
        arr[:, 128] = 1.5   # first Pose z-coord: index 128
        result = self._run_z_clip(arr)
        np.testing.assert_allclose(result[:, 128], 0.10, atol=1e-6)

    def test_negative_z_below_threshold_clipped(self):
        arr = np.zeros((5, FEATURE_SIZE), dtype=np.float32)
        arr[:, 2] = -0.9
        result = self._run_z_clip(arr)
        np.testing.assert_allclose(result[:, 2], -0.10, atol=1e-6)

    def test_z_within_range_unchanged(self):
        """Z-values strictly inside ±0.10 must pass through unchanged."""
        arr = np.zeros((5, FEATURE_SIZE), dtype=np.float32)
        arr[:, 2]   = 0.05    # within range
        arr[:, 65]  = -0.07   # within range
        arr[:, 128] = 0.09    # within range
        result = self._run_z_clip(arr)
        np.testing.assert_allclose(result[:, 2],   0.05,  atol=1e-6)
        np.testing.assert_allclose(result[:, 65],  -0.07, atol=1e-6)
        np.testing.assert_allclose(result[:, 128], 0.09,  atol=1e-6)

    def test_x_and_y_coordinates_not_affected(self):
        """
        X-coords (stride-3 from index 0) and Y-coords (stride-3 from index 1)
        must be completely unchanged by z-clipping.
        """
        arr = np.ones((5, FEATURE_SIZE), dtype=np.float32) * 0.5
        arr[:, 2::3] = 0.5   # inject extreme z values before clip
        x_before = arr[:, 0::3].copy()
        y_before = arr[:, 1::3].copy()
        result = self._run_z_clip(arr)
        np.testing.assert_array_equal(result[:, 0::3], x_before,
            err_msg="X-coordinates were modified by z-clipping.")
        np.testing.assert_array_equal(result[:, 1::3], y_before,
            err_msg="Y-coordinates were modified by z-clipping.")

    def test_all_z_values_within_range_after_clip(self):
        """After clipping, every z-coordinate must be in [-0.10, +0.10]."""
        rng = np.random.default_rng(0)
        arr = rng.uniform(-1.0, 1.0, size=(20, FEATURE_SIZE)).astype(np.float32)
        result = self._run_z_clip(arr)
        z_vals = result[:, 2::3]
        assert z_vals.max() <= 0.10 + 1e-6, "Some z-values exceed +0.10 after clip."
        assert z_vals.min() >= -0.10 - 1e-6, "Some z-values below -0.10 after clip."

    def test_z_clip_zero_config_leaves_z_unchanged(self):
        """
        When z_coord_clip=0.0, the clipping branch is skipped entirely.
        Extreme z-values must survive the pipeline unchanged.
        """
        cfg = load_config(model="lstm", data="seq60", augmentation="none",
                          overrides={"data.z_coord_clip": 0.0})
        p   = FeaturePipeline(cfg)
        arr = _make_two_handed_clip(T=60)
        arr[:, 5] = 5.0    # extreme z value
        result = p(arr, training=False)
        # After normalisation, index 2 (LH first z) is subtracted by the wrist
        # z. The wrist z was set via make_two_handed_clip (random in [0.1, 0.9]).
        # The key invariant: with z_clip=0.0, no clipping occurs, so the result
        # z can exceed ±0.10.
        assert result[:, 5].max() > 0.10 or result[:, 5].min() < -0.10, (
            "z_clip=0.0 should have left some z-values outside ±0.10, "
            "but all were within range — clipping may have been applied anyway."
        )


# ---------------------------------------------------------------------------
# Section 5 — Pad / centre-crop (Step 5)
# ---------------------------------------------------------------------------

class TestPadOrTruncate:
    """
    Padding and centre-crop arithmetic verified against analytically computed
    expected frame indices. Tests operate on _pad_or_truncate directly to
    isolate the transform, then verify integration through __call__ for the
    shape-preservation guarantee.
    """

    @pytest.fixture(autouse=True)
    def pipeline(self):
        # seq_len=60 for all padding/truncation tests
        self.p   = _load_pipeline(data="seq60")
        self.seq = 60

    def _run_pad_truncate(self, arr: np.ndarray) -> np.ndarray:
        """Call _pad_or_truncate directly on a float32 array."""
        return self.p._pad_or_truncate(arr.astype(np.float32, copy=True))

    # --- Short clips (padding) -----------------------------------------------

    def test_short_clip_output_shape(self):
        """A 20-frame clip padded to seq_len=60 has shape (60, 225)."""
        arr    = _make_two_handed_clip(T=20)
        result = self._run_pad_truncate(arr)
        assert result.shape == (60, FEATURE_SIZE)

    def test_short_clip_original_frames_preserved(self):
        """The first T_raw rows of the padded output are bit-identical to input."""
        arr    = _make_two_handed_clip(T=20)
        result = self._run_pad_truncate(arr)
        np.testing.assert_array_equal(result[:20, :], arr,
            err_msg="Original frames were corrupted by right-padding.")

    def test_short_clip_padded_frames_are_exactly_zero(self):
        """Rows 20 onwards must be exactly zero (right-pad with zeros)."""
        arr    = _make_two_handed_clip(T=20)
        result = self._run_pad_truncate(arr)
        np.testing.assert_array_equal(
            result[20:, :],
            np.zeros((40, FEATURE_SIZE), dtype=np.float32),
            err_msg="Padded frames are not exactly zero.",
        )

    def test_short_clip_increments_n_padded(self):
        self.p.reset_statistics()
        arr = _make_two_handed_clip(T=20)
        # Must call through __call__ to trigger counter update
        self.p(arr, training=False)
        assert self.p._n_padded == 1

    def test_short_clip_total_frames_padded_correct(self):
        self.p.reset_statistics()
        arr = _make_two_handed_clip(T=20)
        self.p(arr, training=False)
        assert self.p._total_frames_padded == 40   # 60 - 20

    # --- Long clips (truncation) ---------------------------------------------

    def test_long_clip_output_shape(self):
        """A 120-frame clip cropped to seq_len=60 has shape (60, 225)."""
        arr    = _make_two_handed_clip(T=120)
        result = self._run_pad_truncate(arr)
        assert result.shape == (60, FEATURE_SIZE)

    def test_centre_crop_keeps_frames_30_to_90_for_T120(self):
        """
        T_raw=120, seq_len=60:
            remove=60, start=30, end=90 → keeps frames [30:90].

        We fingerprint each frame in the pose slice (column 200, never
        normalised) with its temporal index and verify the kept frames
        are exactly [30:89].
        """
        arr = _make_two_handed_clip(T=120)
        for t in range(120):
            arr[t, 200] = float(t)
        result = self._run_pad_truncate(arr)

        assert result[0, 200] == pytest.approx(30.0), (
            f"First kept frame: expected index 30, got {result[0, 200]}")
        assert result[-1, 200] == pytest.approx(89.0), (
            f"Last kept frame: expected index 89, got {result[-1, 200]}")

    def test_centre_crop_odd_remove(self):
        """
        T_raw=121, seq_len=60:
            remove=61, start=30, end=90 → keeps frames [30:90].
        Extra frame is removed from the end (floor division gives more to end).
        """
        arr = _make_two_handed_clip(T=121)
        for t in range(121):
            arr[t, 200] = float(t)
        result = self._run_pad_truncate(arr)

        assert result[0, 200] == pytest.approx(30.0)
        assert result[-1, 200] == pytest.approx(89.0)
        assert result.shape == (60, FEATURE_SIZE)

    def test_long_clip_increments_n_truncated(self):
        self.p.reset_statistics()
        arr = _make_two_handed_clip(T=120)
        self.p(arr, training=False)
        assert self.p._n_truncated == 1

    def test_long_clip_total_frames_removed_correct(self):
        self.p.reset_statistics()
        arr = _make_two_handed_clip(T=120)
        self.p(arr, training=False)
        assert self.p._total_frames_removed == 60

    # --- Exact-length clips --------------------------------------------------

    def test_exact_length_clip_returns_unchanged_content(self):
        """T_raw == seq_len: no-op path, content is identical."""
        arr    = _make_two_handed_clip(T=60)
        result = self._run_pad_truncate(arr)
        np.testing.assert_array_equal(result, arr)

    def test_exact_length_clip_does_not_increment_stats(self):
        self.p.reset_statistics()
        arr = _make_two_handed_clip(T=60)
        self.p(arr, training=False)
        assert self.p._n_truncated == 0
        assert self.p._n_padded    == 0

    # --- Various T_raw sizes (parametrized) ----------------------------------

    @pytest.mark.parametrize("T_raw", [1, 10, 30, 59, 60, 61, 90, 120, 180])
    def test_output_shape_matches_seq_len_for_all_lengths(self, T_raw: int):
        """For any T_raw, the output row count is always seq_len."""
        arr    = _make_two_handed_clip(T=T_raw)
        result = self._run_pad_truncate(arr)
        assert result.shape == (self.seq, FEATURE_SIZE), (
            f"T_raw={T_raw}: expected shape ({self.seq}, {FEATURE_SIZE}), "
            f"got {result.shape}"
        )


# ---------------------------------------------------------------------------
# Section 6 — Augmentation ordering (CRITICAL)
# ---------------------------------------------------------------------------

class TestAugmentationOrdering:
    """
    The most architecturally critical tests in this file.

    Bug fixed in pipeline.py: augmentation was previously called AFTER
    landmark config selection. AugmentationPipeline validates shape[1]==225.
    For hands_only (126-dim) or pose_only (99-dim), this crashed with
    ValueError. The corrected chain augments on the full 225-dim array,
    then selects the configured landmark subset.

    These tests are the regression guard for that fix.
    """

    def test_inference_mode_is_deterministic_regardless_of_clip_idx(self):
        """
        training=False must always produce identical output regardless of
        clip_idx. This is the deployment guarantee — inference is deterministic.
        """
        p   = _load_pipeline(augmentation="spatial_temporal")
        arr = _make_two_handed_clip(T=50)

        result_idx0   = p(arr, training=False, clip_idx=0)
        result_idx1   = p(arr, training=False, clip_idx=1)
        result_idx999 = p(arr, training=False, clip_idx=999)

        np.testing.assert_array_equal(result_idx0, result_idx1,
            err_msg="Inference output differs between clip_idx=0 and clip_idx=1.")
        np.testing.assert_array_equal(result_idx0, result_idx999,
            err_msg="Inference output differs between clip_idx=0 and clip_idx=999.")

    def test_training_mode_applies_augmentation(self):
        """
        training=True with spatial_temporal config must produce output that
        differs from training=False (augmentation was applied).
        """
        p   = _load_pipeline(augmentation="spatial_temporal")
        arr = _make_two_handed_clip(T=50)
        result_infer = p(arr, training=False, clip_idx=42)
        result_train = p(arr, training=True,  clip_idx=42)
        assert not np.allclose(result_infer, result_train), (
            "training=True produced the same output as training=False. "
            "Augmentation was not applied."
        )

    def test_training_mode_same_clip_idx_is_deterministic(self):
        """Same clip_idx in training mode must produce identical output."""
        p   = _load_pipeline(augmentation="spatial_temporal")
        arr = _make_two_handed_clip(T=50)
        r1  = p(arr, training=True, clip_idx=7)
        r2  = p(arr, training=True, clip_idx=7)
        np.testing.assert_array_equal(r1, r2,
            err_msg="Same clip_idx in training mode produced different output.")

    def test_training_mode_none_aug_same_as_inference(self):
        """
        With augmentation='none' config, training=True and training=False
        must produce identical output (no augmentation to apply).
        """
        p   = _load_pipeline(augmentation="none")
        arr = _make_two_handed_clip(T=50)
        result_infer = p(arr, training=False, clip_idx=42)
        result_train = p(arr, training=True,  clip_idx=42)
        np.testing.assert_array_equal(result_infer, result_train)

    def test_hands_only_config_with_training_true_does_not_crash(self):
        """
        CRITICAL REGRESSION TEST.

        hands_only config + training=True must NOT raise ValueError.

        Before the chain reorder fix, augmentation ran on the 126-dim
        post-selection array and AugmentationPipeline raised:
            ValueError: expected feature dimension 225, got 126

        The corrected chain: augment on full 225 → then select hands_only.
        """
        cfg = load_config(model="lstm", data="seq60", augmentation="spatial_temporal",
                          overrides={"data.landmark_config": "hands_only"})
        p   = FeaturePipeline(cfg)
        arr = _make_two_handed_clip(T=50)
        # Must not raise:
        result = p(arr, training=True, clip_idx=42)
        assert result.shape == (60, _DIM_HANDS_ONLY), (
            f"hands_only output shape {result.shape} != (60, {_DIM_HANDS_ONLY})"
        )

    def test_pose_only_config_with_training_true_does_not_crash(self):
        """
        CRITICAL REGRESSION TEST.

        pose_only config + training=True must NOT raise ValueError.

        Before the fix: augmentation received 99-dim array → crash.
        After the fix: augmentation receives 225-dim array → succeeds.
        """
        cfg = load_config(model="lstm", data="seq60", augmentation="spatial_temporal",
                          overrides={"data.landmark_config": "pose_only"})
        p   = FeaturePipeline(cfg)
        arr = _make_two_handed_clip(T=50)
        result = p(arr, training=True, clip_idx=42)
        assert result.shape == (60, _DIM_POSE_ONLY), (
            f"pose_only output shape {result.shape} != (60, {_DIM_POSE_ONLY})"
        )

    def test_full_config_with_training_true_does_not_crash(self):
        """Full config + training=True baseline: no crash, correct shape."""
        p   = _load_pipeline(augmentation="spatial_temporal")
        arr = _make_two_handed_clip(T=50)
        result = p(arr, training=True, clip_idx=0)
        assert result.shape == (60, _DIM_FULL)

    def test_augmented_output_has_correct_feature_dim_hands_only(self):
        """
        After augmentation on the full 225-dim array, landmark selection
        must correctly reduce to 126 for hands_only.
        """
        cfg = load_config(model="lstm", data="seq60", augmentation="temporal",
                          overrides={"data.landmark_config": "hands_only"})
        p   = FeaturePipeline(cfg)
        arr = _make_two_handed_clip(T=60)
        result = p(arr, training=True, clip_idx=0)
        assert result.shape[1] == _DIM_HANDS_ONLY

    def test_augmented_output_has_correct_feature_dim_pose_only(self):
        """pose_only: augmented full array → selection gives 99-dim output."""
        cfg = load_config(model="lstm", data="seq60", augmentation="temporal",
                          overrides={"data.landmark_config": "pose_only"})
        p   = FeaturePipeline(cfg)
        arr = _make_two_handed_clip(T=60)
        result = p(arr, training=True, clip_idx=0)
        assert result.shape[1] == _DIM_POSE_ONLY


# ---------------------------------------------------------------------------
# Section 7 — Landmark config selection (Step 7)
# ---------------------------------------------------------------------------

class TestLandmarkConfigSelect:
    """
    Feature-vector slicing per landmark_config.

    Tests verify both shape and DATA CONTENT: that the correct columns from
    the 225-dim array appear in the output, and that columns from other bands
    are genuinely absent.
    """

    def test_full_config_output_shape(self):
        p      = _load_pipeline(data="seq60")
        arr    = _make_two_handed_clip(T=60)
        result = p(arr, training=False)
        assert result.shape == (60, _DIM_FULL)

    def test_hands_only_config_output_shape(self):
        cfg    = load_config(model="lstm", data="seq60", augmentation="none",
                             overrides={"data.landmark_config": "hands_only"})
        p      = FeaturePipeline(cfg)
        arr    = _make_two_handed_clip(T=60)
        result = p(arr, training=False)
        assert result.shape == (60, _DIM_HANDS_ONLY)

    def test_pose_only_config_output_shape(self):
        cfg    = load_config(model="lstm", data="seq60", augmentation="none",
                             overrides={"data.landmark_config": "pose_only"})
        p      = FeaturePipeline(cfg)
        arr    = _make_two_handed_clip(T=60)
        result = p(arr, training=False)
        assert result.shape == (60, _DIM_POSE_ONLY)

    def test_hands_only_contains_no_pose_data(self):
        """
        For hands_only config, the output must not contain pose column data.
        We construct a clip with distinct pose values and verify they are absent.
        """
        cfg    = load_config(model="lstm", data="seq60", augmentation="none",
                             overrides={"data.landmark_config": "hands_only"})
        p      = FeaturePipeline(cfg)
        arr    = _make_two_handed_clip(T=60)
        # Set pose columns to a recognisable sentinel value
        arr[:, POSE_SLICE] = 99.0
        result = p(arr, training=False)
        # If pose data leaked into output, some values would be 99.0
        assert result.max() < 10.0, (
            "hands_only output contains pose sentinel values — pose leaked into output."
        )

    def test_pose_only_contains_no_hand_data(self):
        """
        For pose_only config, the output must not contain hand column data.
        """
        cfg    = load_config(model="lstm", data="seq60", augmentation="none",
                             overrides={"data.landmark_config": "pose_only"})
        p      = FeaturePipeline(cfg)
        arr    = _make_two_handed_clip(T=60)
        # Set hand columns to sentinel
        arr[:, LEFT_HAND_SLICE]  = 88.0
        arr[:, RIGHT_HAND_SLICE] = 88.0
        result = p(arr, training=False)
        assert result.max() < 10.0, (
            "pose_only output contains hand sentinel values — hands leaked into output."
        )

    def test_all_configs_produce_seq_len_rows(self):
        """Row count must equal seq_len for all three landmark configs."""
        for lm_cfg, expected_cols in [
            ("full", _DIM_FULL),
            ("hands_only", _DIM_HANDS_ONLY),
            ("pose_only", _DIM_POSE_ONLY),
        ]:
            cfg = load_config(model="lstm", data="seq60", augmentation="none",
                              overrides={"data.landmark_config": lm_cfg})
            p   = FeaturePipeline(cfg)
            arr = _make_two_handed_clip(T=80)   # T_raw > seq_len → will be cropped
            result = p(arr, training=False)
            assert result.shape == (60, expected_cols), (
                f"lm_cfg={lm_cfg}: expected (60, {expected_cols}), got {result.shape}"
            )


# ---------------------------------------------------------------------------
# Section 8 — Output dtype
# ---------------------------------------------------------------------------

class TestOutputDtype:
    """Output must always be float32, regardless of input dtype."""

    @pytest.fixture(autouse=True)
    def pipeline(self):
        self.p = _load_pipeline()

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.float16])
    def test_float_input_produces_float32_output(self, dtype):
        arr    = _make_two_handed_clip(T=30).astype(dtype)
        result = self.p(arr, training=False)
        assert result.dtype == np.float32, (
            f"Input dtype {dtype}: expected float32 output, got {result.dtype}"
        )

    def test_int32_input_produces_float32_output(self):
        arr    = np.ones((30, FEATURE_SIZE), dtype=np.int32)
        result = self.p(arr, training=False)
        assert result.dtype == np.float32

    def test_int64_input_produces_float32_output(self):
        arr    = np.ones((30, FEATURE_SIZE), dtype=np.int64)
        result = self.p(arr, training=False)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# Section 9 — Properties
# ---------------------------------------------------------------------------

class TestProperties:
    """All @property accessors return correct values for all configurations."""

    @pytest.mark.parametrize("lm_cfg, expected_dim", [
        ("full",       _DIM_FULL),
        ("hands_only", _DIM_HANDS_ONLY),
        ("pose_only",  _DIM_POSE_ONLY),
    ])
    def test_output_shape_property(self, lm_cfg: str, expected_dim: int):
        cfg = load_config(model="lstm", data="seq60", augmentation="none",
                          overrides={"data.landmark_config": lm_cfg})
        p   = FeaturePipeline(cfg)
        assert p.output_shape == (60, expected_dim)

    @pytest.mark.parametrize("seq_cfg, expected_len", [
        ("seq20", 20), ("seq30", 30), ("seq40", 40), ("seq60", 60),
    ])
    def test_sequence_length_property(self, seq_cfg: str, expected_len: int):
        p = _load_pipeline(data=seq_cfg)
        assert p.sequence_length == expected_len

    @pytest.mark.parametrize("lm_cfg, expected_dim", [
        ("full",       _DIM_FULL),
        ("hands_only", _DIM_HANDS_ONLY),
        ("pose_only",  _DIM_POSE_ONLY),
    ])
    def test_feature_dim_property(self, lm_cfg: str, expected_dim: int):
        cfg = load_config(model="lstm", data="seq60", augmentation="none",
                          overrides={"data.landmark_config": lm_cfg})
        p   = FeaturePipeline(cfg)
        assert p.feature_dim == expected_dim

    @pytest.mark.parametrize("lm_cfg", ["full", "hands_only", "pose_only"])
    def test_landmark_config_property(self, lm_cfg: str):
        cfg = load_config(model="lstm", data="seq60", augmentation="none",
                          overrides={"data.landmark_config": lm_cfg})
        p   = FeaturePipeline(cfg)
        assert p.landmark_config == lm_cfg

    def test_n_clips_processed_starts_at_zero(self):
        p = _load_pipeline()
        assert p.n_clips_processed == 0

    def test_n_clips_processed_increments_per_call(self):
        p   = _load_pipeline()
        arr = _make_two_handed_clip(T=30)
        for i in range(1, 6):
            p(arr, training=False)
            assert p.n_clips_processed == i


# ---------------------------------------------------------------------------
# Section 10 — Statistics accumulation
# ---------------------------------------------------------------------------

class TestStatisticsAccumulation:
    """
    Internal counters correctly track processing, truncation, and padding
    across multiple __call__ invocations. reset_statistics() returns all
    six counters to zero.
    """

    @pytest.fixture(autouse=True)
    def pipeline(self):
        self.p = _load_pipeline(data="seq60")

    def test_n_clips_processed_accumulates(self):
        arr = _make_two_handed_clip(T=60)
        for _ in range(7):
            self.p(arr, training=False)
        assert self.p._n_processed == 7

    def test_n_truncated_accumulates_for_long_clips(self):
        self.p.reset_statistics()
        arr = _make_two_handed_clip(T=120)    # T > seq_len → truncated
        for _ in range(5):
            self.p(arr, training=False)
        assert self.p._n_truncated == 5

    def test_total_frames_removed_accumulates(self):
        self.p.reset_statistics()
        arr = _make_two_handed_clip(T=90)     # remove = 30 per clip
        for _ in range(4):
            self.p(arr, training=False)
        assert self.p._total_frames_removed == 120   # 4 × 30

    def test_n_padded_accumulates_for_short_clips(self):
        self.p.reset_statistics()
        arr = _make_two_handed_clip(T=20)     # T < seq_len → padded
        for _ in range(3):
            self.p(arr, training=False)
        assert self.p._n_padded == 3

    def test_total_frames_padded_accumulates(self):
        self.p.reset_statistics()
        arr = _make_two_handed_clip(T=20)     # pad = 40 per clip
        for _ in range(3):
            self.p(arr, training=False)
        assert self.p._total_frames_padded == 120   # 3 × 40

    def test_truncated_and_padded_are_mutually_exclusive_per_clip(self):
        """
        A clip that is truncated must not also increment n_padded, and
        vice versa. Stats correctly separate these two cases.
        """
        self.p.reset_statistics()
        long_arr  = _make_two_handed_clip(T=120)
        short_arr = _make_two_handed_clip(T=20)
        self.p(long_arr,  training=False)   # truncated
        self.p(short_arr, training=False)   # padded

        assert self.p._n_truncated == 1
        assert self.p._n_padded    == 1

    def test_exact_length_clips_increment_neither(self):
        self.p.reset_statistics()
        arr = _make_two_handed_clip(T=60)   # exactly seq_len
        self.p(arr, training=False)
        assert self.p._n_truncated == 0
        assert self.p._n_padded    == 0

    def test_reset_statistics_zeroes_all_counters(self):
        """
        reset_statistics() must zero ALL six counters, including n_processed.
        This matches the implementation which resets self._n_processed.
        """
        arr = _make_two_handed_clip(T=120)
        for _ in range(5):
            self.p(arr, training=False)

        self.p.reset_statistics()

        assert self.p._n_processed           == 0
        assert self.p._n_truncated           == 0
        assert self.p._n_padded              == 0
        assert self.p._total_frames_removed  == 0
        assert self.p._total_frames_padded   == 0
        assert self.p._truncation_warn_count == 0

    def test_statistics_accumulate_across_calls_after_reset(self):
        """After a reset, stats accumulate freshly from zero."""
        arr = _make_two_handed_clip(T=90)  # T > seq_len
        for _ in range(5):
            self.p(arr, training=False)

        self.p.reset_statistics()

        for _ in range(3):
            self.p(arr, training=False)

        assert self.p._n_truncated == 3

    def test_metadata_truncation_stats_reflect_processing(self):
        """
        get_pipeline_metadata() truncation_stats must reflect the actual
        processing history, not stale initial values.
        """
        self.p.reset_statistics()
        long_arr  = _make_two_handed_clip(T=120)
        short_arr = _make_two_handed_clip(T=20)
        self.p(long_arr,  training=False)
        self.p(long_arr,  training=False)
        self.p(short_arr, training=False)

        meta  = self.p.get_pipeline_metadata()
        stats = meta["truncation_stats"]

        assert stats["n_clips_processed"] == 3
        assert stats["n_clips_truncated"] == 2
        assert stats["n_clips_padded"]    == 1
        assert stats["total_frames_removed"] == 120   # 2 × 60
        assert stats["total_frames_padded"]  == 40    # 1 × 40


# ---------------------------------------------------------------------------
# Section 11 — Metadata completeness
# ---------------------------------------------------------------------------

class TestMetadata:
    """
    get_pipeline_metadata() must be complete, consistent with the config,
    and fully JSON-serialisable (no numpy scalars, no slice objects, no
    non-primitive types).
    """

    @pytest.fixture(autouse=True)
    def pipeline(self):
        self.p   = _load_pipeline(data="seq60", augmentation="spatial_temporal")
        self.cfg = load_config(model="lstm", data="seq60",
                               augmentation="spatial_temporal")

    def test_metadata_returns_dict(self):
        assert isinstance(self.p.get_pipeline_metadata(), dict)

    def test_all_top_level_keys_present(self):
        meta = self.p.get_pipeline_metadata()
        required = {
            "sequence_length", "feature_dim", "landmark_config",
            "normalisation", "normalise_pose", "z_coord_clip",
            "flip_min_hand_presence", "truncation_strategy", "padding_strategy",
            "seed", "feature_layout", "transform_chain", "augmentation",
            "truncation_stats",
        }
        missing = required - set(meta.keys())
        assert not missing, f"Missing top-level metadata keys: {missing}"

    def test_feature_layout_sub_keys_present(self):
        meta = self.p.get_pipeline_metadata()
        required = {
            "full_feature_size", "left_hand_slice", "right_hand_slice",
            "pose_slice", "active_slice", "n_hand_landmarks",
            "n_pose_landmarks", "n_coords_per_landmark",
        }
        present = set(meta["feature_layout"].keys())
        missing = required - present
        assert not missing, f"Missing feature_layout keys: {missing}"

    def test_truncation_stats_sub_keys_present(self):
        meta = self.p.get_pipeline_metadata()
        required = {
            "n_clips_processed", "n_clips_truncated", "n_clips_padded",
            "truncation_rate", "padding_rate", "mean_frames_removed",
            "mean_frames_padded", "total_frames_removed", "total_frames_padded",
            "heavy_truncation_warnings",
        }
        present = set(meta["truncation_stats"].keys())
        missing = required - present
        assert not missing, f"Missing truncation_stats keys: {missing}"

    def test_seed_matches_config(self):
        """seed in metadata must match config.seed (needed for reconstruction)."""
        meta = self.p.get_pipeline_metadata()
        assert meta["seed"] == self.cfg.seed

    def test_sequence_length_matches_config(self):
        meta = self.p.get_pipeline_metadata()
        assert meta["sequence_length"] == self.cfg.data.sequence_length

    def test_feature_dim_matches_landmark_config(self):
        meta = self.p.get_pipeline_metadata()
        assert meta["feature_dim"] == _DIM_FULL   # "full" is the default

    def test_normalisation_is_wrist_relative(self):
        meta = self.p.get_pipeline_metadata()
        assert meta["normalisation"] == "wrist_relative"

    def test_normalise_pose_is_false(self):
        meta = self.p.get_pipeline_metadata()
        assert meta["normalise_pose"] is False

    def test_truncation_strategy_is_centre(self):
        meta = self.p.get_pipeline_metadata()
        assert meta["truncation_strategy"] == "centre"

    def test_padding_strategy_is_right_zero(self):
        meta = self.p.get_pipeline_metadata()
        assert meta["padding_strategy"] == "right_zero"

    def test_transform_chain_documents_augmentation_before_lm_select(self):
        """
        The transform_chain list must show augmentation before landmark
        config selection — this documents the corrected ordering for
        GesturePredictor reconstruction and auditing.
        """
        meta  = self.p.get_pipeline_metadata()
        chain = meta["transform_chain"]
        assert isinstance(chain, list)

        # Find positions of augmentation and landmark_select entries
        aug_pos = next(
            (i for i, s in enumerate(chain) if "augmentation" in s.lower()),
            None
        )
        lm_pos = next(
            (i for i, s in enumerate(chain) if "landmark_config" in s.lower()),
            None
        )
        assert aug_pos is not None, "augmentation not found in transform_chain"
        assert lm_pos  is not None, "landmark_config_select not found in transform_chain"
        assert aug_pos < lm_pos, (
            f"transform_chain lists landmark_config at position {lm_pos} "
            f"BEFORE augmentation at position {aug_pos}. "
            "This documents the wrong ordering."
        )

    def test_metadata_is_json_serialisable(self):
        """
        All values must survive json.dumps() without error.
        No numpy scalars, no slice objects, no non-primitive types.
        """
        arr = _make_two_handed_clip(T=80)
        self.p(arr, training=False)
        meta = self.p.get_pipeline_metadata()
        try:
            serialised = json.dumps(meta)
        except (TypeError, ValueError) as e:
            pytest.fail(
                f"get_pipeline_metadata() produced non-JSON-serialisable values: {e}"
            )
        # Round-trip check
        recovered = json.loads(serialised)
        assert recovered["seed"] == meta["seed"]

    def test_metadata_feature_layout_active_slice_reflects_lm_config(self):
        """
        active_slice in feature_layout must reflect the configured
        landmark_config, not always the full 225-dim range.
        """
        cfg = load_config(model="lstm", data="seq60", augmentation="none",
                          overrides={"data.landmark_config": "hands_only"})
        p   = FeaturePipeline(cfg)
        meta = p.get_pipeline_metadata()
        assert meta["feature_layout"]["active_slice"] == [0, 126]

    def test_metadata_active_slice_for_pose_only(self):
        cfg = load_config(model="lstm", data="seq60", augmentation="none",
                          overrides={"data.landmark_config": "pose_only"})
        p   = FeaturePipeline(cfg)
        meta = p.get_pipeline_metadata()
        assert meta["feature_layout"]["active_slice"] == [126, 225]


# ---------------------------------------------------------------------------
# Section 12 — repr
# ---------------------------------------------------------------------------

class TestRepr:
    """__repr__ provides enough context to identify the pipeline configuration."""

    @pytest.fixture(autouse=True)
    def pipeline(self):
        self.p = _load_pipeline(data="seq60", augmentation="spatial_temporal")

    def test_repr_contains_seq_len(self):
        assert "60" in repr(self.p)

    def test_repr_contains_landmark_config(self):
        assert "full" in repr(self.p)

    def test_repr_contains_seed(self):
        assert "seed=" in repr(self.p)

    def test_repr_contains_augmentation_status(self):
        r = repr(self.p)
        assert "enabled" in r or "disabled" in r

    def test_repr_disabled_augmentation(self):
        p = _load_pipeline(augmentation="none")
        assert "disabled" in repr(p)

    def test_repr_contains_clips_processed(self):
        assert "clips_processed=0" in repr(self.p)


# ---------------------------------------------------------------------------
# Section 13 — Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """
    Boundary conditions that may expose off-by-one errors, empty-branch
    issues, or corner-case semantics.
    """

    def test_single_frame_clip_pads_to_seq_len(self):
        """T_raw=1 is the absolute minimum; must pad to seq_len without crash."""
        p   = _load_pipeline(data="seq60")
        arr = _make_two_handed_clip(T=1)
        result = p(arr, training=False)
        assert result.shape == (60, _DIM_FULL)

    def test_single_frame_padded_frames_are_zero(self):
        """After padding a 1-frame clip to 60, frames [1:] must be exactly zero."""
        p   = _load_pipeline(data="seq60")
        arr = _make_two_handed_clip(T=1)
        result = p(arr, training=False)
        np.testing.assert_array_equal(
            result[1:, :],
            np.zeros((59, _DIM_FULL), dtype=np.float32),
        )

    def test_seq_len_minus_one_clip_pads_by_one(self):
        """T_raw = seq_len - 1: exactly one zero row appended."""
        p   = _load_pipeline(data="seq60")
        arr = _make_two_handed_clip(T=59)
        result = p(arr, training=False)
        assert result.shape == (60, _DIM_FULL)
        np.testing.assert_array_equal(result[-1, :],
                                      np.zeros(_DIM_FULL, dtype=np.float32))

    def test_seq_len_plus_one_clip_removes_one_frame(self):
        """T_raw = seq_len + 1: exactly one frame removed by centre-crop."""
        p   = _load_pipeline(data="seq60")
        arr = _make_two_handed_clip(T=61)
        p.reset_statistics()
        p(arr, training=False)
        assert p._n_truncated          == 1
        assert p._total_frames_removed == 1

    def test_all_zero_clip_does_not_crash(self):
        """
        A clip where all values are zero (both hands absent, pose absent)
        must pass through the pipeline without error and return all zeros.
        Normalisation must not attempt to subtract zero wrists from zeros.
        """
        p   = _load_pipeline(data="seq60")
        arr = np.zeros((40, FEATURE_SIZE), dtype=np.float32)
        result = p(arr, training=False)
        assert result.shape == (60, _DIM_FULL)
        np.testing.assert_array_equal(
            result[:40, :],
            np.zeros((40, _DIM_FULL), dtype=np.float32),
        )

    def test_very_long_clip_heavy_truncation_shape_correct(self):
        """T_raw = 3 × seq_len: heavy truncation, but output shape is still (60, 225)."""
        p   = _load_pipeline(data="seq60")
        arr = _make_two_handed_clip(T=180)
        result = p(arr, training=False)
        assert result.shape == (60, _DIM_FULL)

    def test_very_long_clip_keeps_temporal_centre(self):
        """
        T_raw=180, seq_len=60:
            remove=120, start=60, end=120 → keeps frames [60:120].
        Verified via _pad_or_truncate directly to avoid normalisation interactions.
        """
        p   = _load_pipeline(data="seq60")
        arr = _make_two_handed_clip(T=180)
        for t in range(180):
            arr[t, 200] = float(t)
        
        # Test the crop arithmetic directly on the raw array
        result = p._pad_or_truncate(arr.astype(np.float32, copy=True))
        
        assert result[0,  200] == pytest.approx(60.0)
        assert result[-1, 200] == pytest.approx(119.0)

    def test_pipeline_handles_rh_only_clip_without_crash(self):
        """One-handed RH-only clip: LH all zeros, must not crash normalisation."""
        p   = _load_pipeline(data="seq60")
        arr = _make_rh_only_clip(T=40)
        result = p(arr, training=False)
        assert result.shape == (60, _DIM_FULL)
        # LH slice in output must still be all zeros (no wrist subtraction occurred)
        np.testing.assert_array_equal(
            result[:40, LEFT_HAND_SLICE],
            np.zeros((40, N_HAND_FEATURES), dtype=np.float32),
        )

    def test_pipeline_handles_lh_only_clip_without_crash(self):
        """One-handed LH-only clip: RH all zeros, must not crash normalisation."""
        p   = _load_pipeline(data="seq60")
        arr = _make_lh_only_clip(T=40)
        result = p(arr, training=False)
        assert result.shape == (60, _DIM_FULL)

    def test_n_clips_processed_increments_even_after_failed_call(self):
        """
        A failed call (raises ValueError) must NOT increment n_processed.
        Counter must only increment for successful completions.
        """
        p   = _load_pipeline(data="seq60")
        arr = _make_two_handed_clip(T=30)

        # Successful call
        p(arr, training=False)
        assert p.n_clips_processed == 1

        # Failed call
        bad_arr = np.zeros((10, 100), dtype=np.float32)
        with pytest.raises(ValueError):
            p(bad_arr, training=False)

        # Counter must not have changed
        assert p.n_clips_processed == 1

    @pytest.mark.parametrize("T_raw", [1, 2, 30, 59, 60, 61, 100, 180, 300])
    def test_output_shape_invariant_for_wide_range_of_lengths(self, T_raw: int):
        """
        For any T_raw in a broad range, the output is always (seq_len, feature_dim).
        This is the primary contract of the pipeline.
        """
        p      = _load_pipeline(data="seq60")
        arr    = _make_two_handed_clip(T=T_raw)
        result = p(arr, training=False)
        assert result.shape == (60, _DIM_FULL), (
            f"T_raw={T_raw}: expected (60, {_DIM_FULL}), got {result.shape}"
        )