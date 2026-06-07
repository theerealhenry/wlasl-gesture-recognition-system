"""
tests/test_augmentation.py
===========================
Complete test suite for src/features/augmentation.py.

Coverage strategy
-----------------
Every public method is tested for:
  1. Shape invariant  — output shape always equals input shape (T, FEATURE_SIZE)
  2. dtype contract   — output is always float32
  3. Input immutability — the caller's array is never modified in place
  4. Zero-fill invariant — component slots that are zero in the original remain
     exactly zero after all transforms (invariant is per-slot, not per-row)
  5. Boundary conditions — zero probability, zero magnitude, edge T values

Additional tests specific to the implementation:
  6. Zero-in-place semantics for temporal_jitter (not compress-then-pad)
  7. speed_jitter produces T fully-interpolated frames for fast clips (no trailing zeros)
  8. gaussian_noise applies per-slot masking (absent hand slot stays zero within
     partially-detected frames — the one-handed sign invariant)
  9. Hybrid per-frame policy for spatial_flip (Cases 1-4)
  10. Rigid-transform property for rotation_2d (preserves pairwise distances)
  11. Double-flip involution: flip(flip(x)) == x for all detection patterns
  12. Pipeline copy-boundary ownership: input never mutated through full chain
  13. Parameter validation helpers: clamping and range checks

Key design decision documented:
  speed_jitter for fast clips (rate > 1.0) uses np.interp to resample back to
  T frames. The output has T non-zero frames — NO trailing zero padding. This is
  the correct behaviour: the transform is a resampler, not a crop-and-pad.
  Tests that assumed trailing zeros were incorrect and have been fixed.

Fixture design
--------------
All fixtures use synthetic arrays with values in [0.1, 0.9] for detected hands
and exactly 0.0 for absent hands. This makes zero-fill assertions exact (not
probabilistic) — np.testing.assert_array_equal can be used without tolerance.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.features.augmentation import (
    AugmentationPipeline,
    SpatialAugmenter,
    TemporalAugmenter,
    _clamp_drop_prob,
    _clamp_hand_presence,
    _validate_landmark_array,
    _validate_speed_range,
)
from src.features.constants import (
    FEATURE_SIZE,
    LEFT_HAND_SLICE,
    N_COORDS_PER_LANDMARK,
    N_HAND_FEATURES,
    N_HAND_LANDMARKS,
    POSE_SLICE,
    RIGHT_HAND_SLICE,
)

# ---------------------------------------------------------------------------
# Shared fixture factory functions
# ---------------------------------------------------------------------------


def make_two_handed_clip(T: int = 50, seed: int = 0) -> np.ndarray:
    """
    Synthetic clip where ALL T frames have both hands detected.

    Values drawn from Uniform(0.1, 0.9) — guaranteed non-zero in every
    slot (LH, RH, pose). Returns float32 array of shape (T, FEATURE_SIZE).
    """
    rng = np.random.default_rng(seed)
    arr = rng.uniform(0.1, 0.9, size=(T, FEATURE_SIZE)).astype(np.float32)
    return arr


def make_one_handed_clip(T: int = 50, seed: int = 0) -> np.ndarray:
    """
    Synthetic clip where the LEFT hand is absent in ALL T frames.

    Right hand and pose are detected (non-zero) in all frames.
    LH slice is exactly zero throughout.
    Returns float32 array of shape (T, FEATURE_SIZE).
    """
    arr = make_two_handed_clip(T, seed)
    arr[:, LEFT_HAND_SLICE] = 0.0   # Left hand always absent
    return arr


def make_rh_only_clip(T: int = 50, seed: int = 0) -> np.ndarray:
    """Alias for make_one_handed_clip — RH only, LH always absent."""
    return make_one_handed_clip(T, seed)


def make_lh_only_clip(T: int = 50, seed: int = 0) -> np.ndarray:
    """
    Synthetic clip where the RIGHT hand is absent in ALL T frames.

    Left hand and pose are detected (non-zero) in all frames.
    RH slice is exactly zero throughout.
    Returns float32 array of shape (T, FEATURE_SIZE).
    """
    arr = make_two_handed_clip(T, seed)
    arr[:, RIGHT_HAND_SLICE] = 0.0  # Right hand always absent
    return arr


def make_mixed_clip(
    T: int = 50, zero_fraction: float = 0.4, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Synthetic clip where a fraction of frames have BOTH hands absent.

    The zero_fraction frames have both LH and RH zeroed; the rest have both
    hands detected. Returns (arr, zero_indices).
    """
    rng    = np.random.default_rng(seed)
    arr    = make_two_handed_clip(T, seed)
    n_zero = max(1, int(T * zero_fraction))
    zero_indices = np.sort(rng.choice(T, size=n_zero, replace=False))
    arr[zero_indices, LEFT_HAND_SLICE]  = 0.0
    arr[zero_indices, RIGHT_HAND_SLICE] = 0.0
    return arr, zero_indices


def make_mixed_detection_clip(T: int = 40, seed: int = 99) -> np.ndarray:
    """
    Synthetic clip with ALL four detection states:

        frames [0 : T//4]    — both hands detected (Case 1)
        frames [T//4 : T//2] — LH only (Case 2)
        frames [T//2 : 3T//4]— RH only (Case 3)
        frames [3T//4 : T]   — neither (Case 4)
    """
    arr = make_two_handed_clip(T, seed)
    q   = T // 4
    arr[q     : 2 * q, RIGHT_HAND_SLICE] = 0.0  # LH only
    arr[2 * q : 3 * q, LEFT_HAND_SLICE]  = 0.0  # RH only
    arr[3 * q :,       LEFT_HAND_SLICE]  = 0.0  # neither
    arr[3 * q :,       RIGHT_HAND_SLICE] = 0.0
    return arr


def make_fingerprinted_clip(T: int = 50, seed: int = 0) -> np.ndarray:
    """
    Two-handed clip where arr[t, 0] = (t+1)*0.001 uniquely fingerprints each frame.

    Used to verify zero-in-place semantics for temporal_jitter: a non-zero
    result frame at position t must carry the fingerprint for frame t.
    """
    arr = make_two_handed_clip(T, seed)
    for t in range(T):
        arr[t, 0] = float(t + 1) * 0.001
    return arr


def _load_augmentation_config(augmentation: str = "spatial_temporal"):
    """Load a named augmentation config via load_config."""
    from src.utils.config import load_config
    cfg = load_config(model="lstm", data="seq60", augmentation=augmentation)
    return cfg.augmentation


# ---------------------------------------------------------------------------
# Section 0 — Validation helpers
# ---------------------------------------------------------------------------


class TestValidateLandmarkArray:
    """Tests for the _validate_landmark_array guard function."""

    def test_accepts_float32(self):
        arr = np.zeros((10, FEATURE_SIZE), dtype=np.float32)
        _validate_landmark_array(arr, "test")

    def test_accepts_float64(self):
        arr = np.zeros((10, FEATURE_SIZE), dtype=np.float64)
        _validate_landmark_array(arr, "test")

    def test_accepts_float16(self):
        arr = np.zeros((10, FEATURE_SIZE), dtype=np.float16)
        _validate_landmark_array(arr, "test")

    def test_rejects_non_ndarray(self):
        with pytest.raises(TypeError, match="np.ndarray"):
            _validate_landmark_array([[1.0, 2.0]], "test")

    def test_rejects_int_dtype(self):
        arr = np.zeros((10, FEATURE_SIZE), dtype=np.int32)
        with pytest.raises(TypeError, match="floating"):
            _validate_landmark_array(arr, "test")

    def test_rejects_bool_dtype(self):
        arr = np.zeros((10, FEATURE_SIZE), dtype=bool)
        with pytest.raises(TypeError, match="floating"):
            _validate_landmark_array(arr, "test")

    def test_rejects_1d(self):
        arr = np.zeros(FEATURE_SIZE, dtype=np.float32)
        with pytest.raises(ValueError, match="2D"):
            _validate_landmark_array(arr, "test")

    def test_rejects_3d(self):
        arr = np.zeros((5, 10, FEATURE_SIZE), dtype=np.float32)
        with pytest.raises(ValueError, match="2D"):
            _validate_landmark_array(arr, "test")

    def test_rejects_wrong_feature_dim(self):
        arr = np.zeros((10, 100), dtype=np.float32)
        with pytest.raises(ValueError, match=str(FEATURE_SIZE)):
            _validate_landmark_array(arr, "test")

    def test_error_message_contains_caller(self):
        arr = np.zeros((10, 100), dtype=np.float32)
        with pytest.raises(ValueError, match="my_transform"):
            _validate_landmark_array(arr, "my_transform")


class TestClampDropProb:
    """Tests for the _clamp_drop_prob helper."""

    def test_valid_prob_unchanged(self):
        assert _clamp_drop_prob(0.10, "t") == pytest.approx(0.10)

    def test_zero_unchanged(self):
        assert _clamp_drop_prob(0.0, "t") == pytest.approx(0.0)

    def test_negative_clamped_to_zero(self, caplog):
        result = _clamp_drop_prob(-0.5, "t")
        assert result == pytest.approx(0.0)
        assert "negative" in caplog.text.lower() or "clamping" in caplog.text.lower()

    def test_one_clamped_below_one(self, caplog):
        result = _clamp_drop_prob(1.0, "t")
        assert result < 1.0
        assert result > 0.95

    def test_above_one_clamped(self, caplog):
        result = _clamp_drop_prob(2.0, "t")
        assert result < 1.0

    def test_boundary_just_below_one(self):
        result = _clamp_drop_prob(0.99, "t")
        assert result == pytest.approx(0.99)


class TestValidateSpeedRange:
    """Tests for the _validate_speed_range helper."""

    def test_valid_range_unchanged(self):
        lo, hi = _validate_speed_range((0.7, 1.3), "t")
        assert lo == pytest.approx(0.7)
        assert hi == pytest.approx(1.3)

    def test_lo_equals_hi_valid(self):
        lo, hi = _validate_speed_range((1.0, 1.0), "t")
        assert lo == hi

    def test_inverted_range_raises(self):
        with pytest.raises(ValueError, match="≤"):
            _validate_speed_range((1.3, 0.7), "t")

    def test_zero_lo_raises(self):
        with pytest.raises(ValueError, match="> 0"):
            _validate_speed_range((0.0, 1.3), "t")

    def test_zero_hi_raises(self):
        with pytest.raises(ValueError, match="> 0"):
            _validate_speed_range((0.7, 0.0), "t")

    def test_negative_bound_raises(self):
        with pytest.raises(ValueError, match="> 0"):
            _validate_speed_range((-0.5, 1.0), "t")


class TestClampHandPresence:
    """Tests for the _clamp_hand_presence helper."""

    def test_valid_threshold_unchanged(self):
        assert _clamp_hand_presence(0.30, "t") == pytest.approx(0.30)

    def test_zero_unchanged(self):
        assert _clamp_hand_presence(0.0, "t") == pytest.approx(0.0)

    def test_one_unchanged(self):
        assert _clamp_hand_presence(1.0, "t") == pytest.approx(1.0)

    def test_negative_clamped_to_zero(self, caplog):
        result = _clamp_hand_presence(-0.1, "t")
        assert result == pytest.approx(0.0)

    def test_above_one_clamped_to_one(self, caplog):
        result = _clamp_hand_presence(1.5, "t")
        assert result == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Section 1 — TemporalAugmenter
# ---------------------------------------------------------------------------


class TestTemporalJitter:
    """Tests for TemporalAugmenter.temporal_jitter."""

    def test_shape_preserved(self):
        aug    = TemporalAugmenter()
        arr    = make_two_handed_clip(T=50)
        rng    = np.random.default_rng(42)
        result = aug.temporal_jitter(arr, rng, drop_prob=0.10)
        assert result.shape == arr.shape

    def test_dtype_is_float32(self):
        aug    = TemporalAugmenter()
        arr    = make_two_handed_clip(T=50)
        rng    = np.random.default_rng(42)
        result = aug.temporal_jitter(arr, rng, drop_prob=0.10)
        assert result.dtype == np.float32

    def test_accepts_float64_input(self):
        aug    = TemporalAugmenter()
        arr    = make_two_handed_clip(T=50).astype(np.float64)
        rng    = np.random.default_rng(42)
        result = aug.temporal_jitter(arr, rng, drop_prob=0.10)
        assert result.dtype == np.float32
        assert result.shape == (50, FEATURE_SIZE)

    def test_zero_prob_returns_content_equal_array(self):
        aug    = TemporalAugmenter()
        arr    = make_two_handed_clip(T=50)
        rng    = np.random.default_rng(42)
        result = aug.temporal_jitter(arr, rng, drop_prob=0.0)
        np.testing.assert_array_equal(result, arr)

    def test_input_not_mutated(self):
        aug     = TemporalAugmenter()
        arr     = make_two_handed_clip(T=50)
        arr_ref = arr.copy()
        rng     = np.random.default_rng(42)
        aug.temporal_jitter(arr, rng, drop_prob=0.10)
        np.testing.assert_array_equal(arr, arr_ref)

    def test_extreme_drop_prob_shape_preserved(self):
        aug    = TemporalAugmenter()
        arr    = make_two_handed_clip(T=50)
        rng    = np.random.default_rng(42)
        result = aug.temporal_jitter(arr, rng, drop_prob=0.99)
        assert result.shape == arr.shape

    def test_dropped_frames_are_exactly_zero(self):
        """
        Frames chosen for dropout must contain EXACT zeros.
        We use a two-handed clip (no original zeros) so any zero row
        in the result is definitively a dropped frame.
        """
        aug    = TemporalAugmenter()
        arr    = make_two_handed_clip(T=50)
        rng    = np.random.default_rng(7)
        result = aug.temporal_jitter(arr, rng, drop_prob=0.30)
        for t in range(50):
            row = result[t]
            if row.sum() == 0.0:
                assert (row == 0.0).all(), (
                    f"Frame {t} was dropped but contains non-zero values."
                )

    def test_zero_in_place_semantics(self):
        """
        Kept frames must preserve their ORIGINAL TEMPORAL POSITION.

        Uses a fingerprinted clip: arr[t, 0] = (t+1)*0.001.
        Any non-zero result frame at position t must carry the fingerprint
        for frame t — not t+1 or t-1.
        """
        aug    = TemporalAugmenter()
        arr    = make_fingerprinted_clip(T=50)
        rng    = np.random.default_rng(42)
        result = aug.temporal_jitter(arr, rng, drop_prob=0.30)
        for t in range(50):
            row = result[t]
            if not (row == 0.0).all():
                expected_fp = float(t + 1) * 0.001
                actual_fp   = float(row[0])
                assert abs(actual_fp - expected_fp) < 1e-5, (
                    f"Frame {t}: fingerprint {actual_fp:.4f} ≠ {expected_fp:.4f}. "
                    "Compress-then-pad behaviour detected."
                )

    def test_at_least_one_frame_always_kept(self):
        """Even at 99% drop probability, at least one frame must remain non-zero."""
        aug = TemporalAugmenter()
        arr = make_two_handed_clip(T=50)
        for trial_seed in range(20):
            rng    = np.random.default_rng(trial_seed)
            result = aug.temporal_jitter(arr, rng, drop_prob=0.99)
            n_nonzero_frames = (result.sum(axis=1) != 0).sum()
            assert n_nonzero_frames >= 1

    @pytest.mark.parametrize("T", [1, 5, 10, 30, 60, 100])
    def test_shape_preserved_various_lengths(self, T: int):
        aug    = TemporalAugmenter()
        arr    = make_two_handed_clip(T=T)
        rng    = np.random.default_rng(0)
        result = aug.temporal_jitter(arr, rng, drop_prob=0.20)
        assert result.shape == (T, FEATURE_SIZE)

    def test_negative_drop_prob_clamped(self):
        aug    = TemporalAugmenter()
        arr    = make_two_handed_clip(T=20)
        rng    = np.random.default_rng(0)
        result = aug.temporal_jitter(arr, rng, drop_prob=-0.5)
        assert result.shape == arr.shape

    def test_drop_prob_above_one_clamped(self):
        aug    = TemporalAugmenter()
        arr    = make_two_handed_clip(T=20)
        rng    = np.random.default_rng(0)
        result = aug.temporal_jitter(arr, rng, drop_prob=1.5)
        assert result.shape == arr.shape


class TestSpeedJitter:
    """Tests for TemporalAugmenter.speed_jitter."""

    def test_shape_preserved_slow_range(self):
        aug    = TemporalAugmenter()
        arr    = make_two_handed_clip(T=50)
        rng    = np.random.default_rng(0)
        result = aug.speed_jitter(arr, rng, speed_range=(0.7, 0.9))
        assert result.shape == arr.shape

    def test_shape_preserved_fast_range(self):
        aug    = TemporalAugmenter()
        arr    = make_two_handed_clip(T=50)
        rng    = np.random.default_rng(0)
        result = aug.speed_jitter(arr, rng, speed_range=(1.1, 1.3))
        assert result.shape == arr.shape

    def test_dtype_is_float32(self):
        aug    = TemporalAugmenter()
        arr    = make_two_handed_clip(T=50)
        rng    = np.random.default_rng(42)
        result = aug.speed_jitter(arr, rng)
        assert result.dtype == np.float32

    def test_input_not_mutated(self):
        aug     = TemporalAugmenter()
        arr     = make_two_handed_clip(T=50)
        arr_ref = arr.copy()
        rng     = np.random.default_rng(42)
        aug.speed_jitter(arr, rng)
        np.testing.assert_array_equal(arr, arr_ref)

    def test_accepts_float64_input(self):
        aug    = TemporalAugmenter()
        arr    = make_two_handed_clip(T=30).astype(np.float64)
        rng    = np.random.default_rng(0)
        result = aug.speed_jitter(arr, rng)
        assert result.dtype == np.float32
        assert result.shape == (30, FEATURE_SIZE)

    @pytest.mark.parametrize("T", [1, 3, 10, 30, 50, 60, 100, 150])
    def test_shape_preserved_many_lengths(self, T: int):
        aug    = TemporalAugmenter()
        arr    = make_two_handed_clip(T=T)
        rng    = np.random.default_rng(T)
        result = aug.speed_jitter(arr, rng)
        assert result.shape == (T, FEATURE_SIZE), (
            f"Shape mismatch at T={T}: got {result.shape}"
        )

    @pytest.mark.parametrize("seed", range(20))
    def test_shape_preserved_random_seeds(self, seed: int):
        aug    = TemporalAugmenter()
        arr    = make_two_handed_clip(T=60)
        rng    = np.random.default_rng(seed)
        result = aug.speed_jitter(arr, rng, speed_range=(0.7, 1.3))
        assert result.shape == (60, FEATURE_SIZE)

    def test_rate_one_approximately_unchanged(self):
        """When rate is forced to exactly 1.0, content should be identical."""
        aug    = TemporalAugmenter()
        arr    = make_two_handed_clip(T=30)
        rng    = np.random.default_rng(0)
        result = aug.speed_jitter(arr, rng, speed_range=(1.0, 1.0))
        np.testing.assert_allclose(result, arr, atol=1e-6)
    
    def test_faster_clip_fully_interpolated_no_trailing_zeros(self):
        """
        CORRECT BEHAVIOUR FOR FAST CLIPS (rate > 1.0):

        speed_jitter uses np.interp to resample back to T frames after
        extracting n_resampled < T frames. The output has T non-zero frames
        with interpolated landmark values — there are NO trailing zero pads.

        This is intentional design: the transform is a resampler, not a
        crop-and-pad. Trailing zeros would silently discard the sign's tail.

        This test verifies the correct behaviour: all T output frames are
        non-zero for a two-handed clip (all input frames non-zero).
        """
        aug = TemporalAugmenter()
        arr = make_two_handed_clip(T=50)    # all frames non-zero
        rng = np.random.default_rng(0)
        # Force fast rate (1.28–1.30): n_resampled ≈ round(50/1.29) ≈ 39
        # After interp back to T=50, all 50 frames should be non-zero
        result = aug.speed_jitter(arr, rng, speed_range=(1.28, 1.30))

        assert result.shape == (50, FEATURE_SIZE), "Shape must be preserved"

        # ALL frames must be non-zero (interpolation fills every frame)
        n_zero_frames = sum(1 for t in range(50) if (result[t] == 0.0).all())
        assert n_zero_frames == 0, (
            f"Found {n_zero_frames} all-zero frames in fast-clip output. "
            "speed_jitter should interpolate back to T frames (no trailing zeros). "
            "If this fails, the implementation is using crop-and-pad instead of interp."
        )

    def test_faster_clip_values_in_reasonable_range(self):
        """
        For a fast clip, interpolated output values should lie within the
        range of the input values (linear interpolation is bounded by input).
        """
        aug    = TemporalAugmenter()
        arr    = make_two_handed_clip(T=50)  # values in [0.1, 0.9]
        rng    = np.random.default_rng(0)
        result = aug.speed_jitter(arr, rng, speed_range=(1.28, 1.30))

        input_min  = float(arr.min())
        input_max  = float(arr.max())
        output_min = float(result.min())
        output_max = float(result.max())

        # Interpolated values must be within the input range (with small tolerance
        # for float32 rounding)
        assert output_min >= input_min - 1e-4, (
            f"Interpolated minimum {output_min:.6f} < input minimum {input_min:.6f}"
        )
        assert output_max <= input_max + 1e-4, (
            f"Interpolated maximum {output_max:.6f} > input maximum {input_max:.6f}"
        )

    def test_inverted_speed_range_raises(self):
        aug = TemporalAugmenter()
        arr = make_two_handed_clip(T=20)
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="≤"):
            aug.speed_jitter(arr, rng, speed_range=(1.3, 0.7))

    def test_zero_bound_raises(self):
        aug = TemporalAugmenter()
        arr = make_two_handed_clip(T=20)
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="> 0"):
            aug.speed_jitter(arr, rng, speed_range=(0.0, 1.3))


# ---------------------------------------------------------------------------
# Section 2 — SpatialAugmenter: Gaussian noise
# ---------------------------------------------------------------------------


class TestGaussianNoise:
    """
    Tests for SpatialAugmenter.gaussian_noise.

    Key correctness requirement: the zero-fill invariant operates at the
    COMPONENT SLOT level, not the frame level.

    For a one-handed frame (e.g., RH detected, LH absent), the frame as a
    whole is "partially detected". The per-row masking approach (old bug)
    applied noise to the entire 225-element row for any partially-detected
    frame, corrupting the absent-hand slot. The correct approach applies
    noise only to the detected component's slot columns.
    """

    def test_shape_preserved(self):
        aug    = SpatialAugmenter()
        arr    = make_two_handed_clip(T=50)
        rng    = np.random.default_rng(42)
        result = aug.gaussian_noise(arr, rng, std=0.01, detected_only=True)
        assert result.shape == arr.shape

    def test_dtype_is_float32(self):
        aug    = SpatialAugmenter()
        arr    = make_two_handed_clip(T=50)
        rng    = np.random.default_rng(42)
        result = aug.gaussian_noise(arr, rng, std=0.01, detected_only=True)
        assert result.dtype == np.float32

    def test_input_not_mutated(self):
        aug     = SpatialAugmenter()
        arr     = make_two_handed_clip(T=50)
        arr_ref = arr.copy()
        rng     = np.random.default_rng(42)
        aug.gaussian_noise(arr, rng, std=0.01, detected_only=True)
        np.testing.assert_array_equal(arr, arr_ref)

    def test_zero_std_returns_equal_array(self):
        aug    = SpatialAugmenter()
        arr    = make_two_handed_clip(T=50)
        rng    = np.random.default_rng(42)
        result = aug.gaussian_noise(arr, rng, std=0.0, detected_only=True)
        np.testing.assert_array_equal(result, arr)

    def test_detected_frames_are_modified(self):
        """For std > 0, detected frames must differ from the original."""
        aug    = SpatialAugmenter()
        arr    = make_two_handed_clip(T=50)
        rng    = np.random.default_rng(42)
        result = aug.gaussian_noise(arr, rng, std=0.01, detected_only=True)
        assert not np.allclose(result, arr)

    # -- Zero-fill invariant (per-slot) -----------------------------------

    def test_both_absent_frames_exactly_zero(self):
        """
        CRITICAL: Frames where BOTH hands are absent must remain exactly
        zero after noise, even at high std values.
        """
        aug, rng = SpatialAugmenter(), np.random.default_rng(42)
        arr, zero_indices = make_mixed_clip(T=50, zero_fraction=0.4)
        result = aug.gaussian_noise(arr, rng, std=0.10, detected_only=True)
        for idx in zero_indices:
            np.testing.assert_array_equal(
                result[idx, LEFT_HAND_SLICE],
                np.zeros(N_HAND_FEATURES, dtype=np.float32),
                err_msg=f"Frame {idx} LH corrupted despite both-absent state",
            )
            np.testing.assert_array_equal(
                result[idx, RIGHT_HAND_SLICE],
                np.zeros(N_HAND_FEATURES, dtype=np.float32),
                err_msg=f"Frame {idx} RH corrupted despite both-absent state",
            )

    def test_one_handed_lh_zero_preserved(self):
        """
        PER-SLOT INVARIANT: For a right-dominant one-handed sign (LH always
        absent), the entire LH slot must remain exactly zero after noise —
        even though the frame is "partially detected" (RH is present).

        This is the key test that distinguishes per-slot masking (correct)
        from per-row masking (old bug).
        """
        aug    = SpatialAugmenter()
        arr    = make_one_handed_clip(T=50)   # LH=0, RH≠0, pose≠0
        rng    = np.random.default_rng(42)
        result = aug.gaussian_noise(arr, rng, std=0.01, detected_only=True)
        np.testing.assert_array_equal(
            result[:, LEFT_HAND_SLICE],
            np.zeros((50, N_HAND_FEATURES), dtype=np.float32),
            err_msg=(
                "LH zero-fill corrupted for one-handed (RH-dominant) sign. "
                "Per-row masking bug: the whole 225-dim row received noise "
                "because RH was detected, overwriting the absent LH slot."
            ),
        )

    def test_lh_only_clip_rh_zero_preserved(self):
        """
        PER-SLOT INVARIANT: For a left-dominant one-handed sign (RH always
        absent), the entire RH slot must remain exactly zero after noise.
        """
        aug    = SpatialAugmenter()
        arr    = make_lh_only_clip(T=50)   # RH=0, LH≠0, pose≠0
        rng    = np.random.default_rng(42)
        result = aug.gaussian_noise(arr, rng, std=0.01, detected_only=True)
        np.testing.assert_array_equal(
            result[:, RIGHT_HAND_SLICE],
            np.zeros((50, N_HAND_FEATURES), dtype=np.float32),
            err_msg=(
                "RH zero-fill corrupted for one-handed (LH-dominant) sign. "
                "Per-row masking bug: the whole row received noise because "
                "LH was detected, overwriting the absent RH slot."
            ),
        )

    def test_rh_slot_is_modified_in_one_handed_rh_clip(self):
        """
        Complementary to test_one_handed_lh_zero_preserved: the DETECTED
        hand (RH) must actually receive noise. Confirms per-slot masking
        doesn't accidentally suppress the detected slot.
        """
        aug    = SpatialAugmenter()
        arr    = make_one_handed_clip(T=50)   # LH=0, RH≠0
        rng    = np.random.default_rng(42)
        result = aug.gaussian_noise(arr, rng, std=0.01, detected_only=True)
        # RH slot must differ from original (noise was applied)
        assert not np.allclose(
            result[:, RIGHT_HAND_SLICE],
            arr[:, RIGHT_HAND_SLICE],
        ), "RH slot was not modified — noise not applied to detected slot"

    def test_lh_slot_is_modified_in_lh_only_clip(self):
        """
        LH slot must receive noise in a LH-only clip (detected slot).
        """
        aug    = SpatialAugmenter()
        arr    = make_lh_only_clip(T=50)   # RH=0, LH≠0
        rng    = np.random.default_rng(42)
        result = aug.gaussian_noise(arr, rng, std=0.01, detected_only=True)
        assert not np.allclose(
            result[:, LEFT_HAND_SLICE],
            arr[:, LEFT_HAND_SLICE],
        ), "LH slot was not modified — noise not applied to detected slot"

    def test_detected_frames_never_become_all_zero(self):
        """
        After noise, a detected frame must never become all-zero.
        Values in [0.1, 0.9] with std=0.01 cannot become exactly 0.
        """
        aug    = SpatialAugmenter()
        arr    = make_two_handed_clip(T=60)
        rng    = np.random.default_rng(42)
        result = aug.gaussian_noise(arr, rng, std=0.01, detected_only=True)
        for t in range(60):
            assert not (result[t] == 0.0).all(), (
                f"Frame {t} became all-zero after noise."
            )

    def test_detected_only_false_applies_noise_everywhere(self):
        """
        With detected_only=False, noise is applied to all columns including
        absent slots. Absent LH slot should receive noise.
        """
        aug    = SpatialAugmenter()
        arr    = make_one_handed_clip(T=50)  # LH=0
        rng    = np.random.default_rng(42)
        result = aug.gaussian_noise(arr, rng, std=0.01, detected_only=False)
        # LH slot must be modified when detected_only=False
        assert not np.allclose(
            result[:, LEFT_HAND_SLICE],
            np.zeros((50, N_HAND_FEATURES), dtype=np.float32),
        ), "With detected_only=False, absent LH slot should receive noise"


# ---------------------------------------------------------------------------
# Section 3 — SpatialAugmenter: spatial_flip
# ---------------------------------------------------------------------------


class TestSpatialFlip:
    """Tests for SpatialAugmenter.spatial_flip."""

    def test_shape_preserved(self):
        aug    = SpatialAugmenter()
        arr    = make_two_handed_clip(T=50)
        rng    = np.random.default_rng(42)
        result = aug.spatial_flip(arr, rng, min_hand_presence=0.0)
        assert result.shape == arr.shape

    def test_dtype_is_float32(self):
        aug    = SpatialAugmenter()
        arr    = make_two_handed_clip(T=50)
        rng    = np.random.default_rng(42)
        result = aug.spatial_flip(arr, rng, min_hand_presence=0.0)
        assert result.dtype == np.float32

    def test_input_not_mutated(self):
        aug     = SpatialAugmenter()
        arr     = make_two_handed_clip(T=50)
        arr_ref = arr.copy()
        rng     = np.random.default_rng(42)
        aug.spatial_flip(arr, rng, min_hand_presence=0.0)
        np.testing.assert_array_equal(arr, arr_ref)

    def test_one_handed_clip_not_flipped(self):
        """A one-handed clip (0% both-hands) must be returned unchanged at threshold=0.30."""
        aug     = SpatialAugmenter()
        arr     = make_one_handed_clip(T=50)
        arr_ref = arr.copy()
        rng     = np.random.default_rng(42)
        result  = aug.spatial_flip(arr, rng, min_hand_presence=0.30)
        np.testing.assert_array_equal(result, arr_ref)

    def test_two_handed_clip_is_flipped(self):
        """Clip with 100% both-hands presence must be modified."""
        aug    = SpatialAugmenter()
        arr    = make_two_handed_clip(T=50)
        rng    = np.random.default_rng(42)
        result = aug.spatial_flip(arr, rng, min_hand_presence=0.30)
        assert not np.allclose(result, arr)

    def test_threshold_zero_forces_flip_on_one_handed(self):
        """With min_hand_presence=0.0, even a one-handed clip is flipped."""
        aug    = SpatialAugmenter()
        arr    = make_one_handed_clip(T=50)
        rng    = np.random.default_rng(42)
        result = aug.spatial_flip(arr, rng, min_hand_presence=0.0)
        assert not np.allclose(result, arr)

    # -- Case 1: Both hands detected ------------------------------------------

    def test_both_hands_lh_x_becomes_negated_rh_x(self):
        aug       = SpatialAugmenter()
        arr       = make_two_handed_clip(T=50, seed=42)
        orig_rh_x = arr[:, RIGHT_HAND_SLICE][:, 0::3].copy()
        rng       = np.random.default_rng(42)
        result    = aug.spatial_flip(arr, rng, min_hand_presence=0.0)
        np.testing.assert_allclose(
            result[:, LEFT_HAND_SLICE][:, 0::3], -orig_rh_x, atol=1e-6,
        )

    def test_both_hands_rh_x_becomes_negated_lh_x(self):
        aug       = SpatialAugmenter()
        arr       = make_two_handed_clip(T=50, seed=42)
        orig_lh_x = arr[:, LEFT_HAND_SLICE][:, 0::3].copy()
        rng       = np.random.default_rng(42)
        result    = aug.spatial_flip(arr, rng, min_hand_presence=0.0)
        np.testing.assert_allclose(
            result[:, RIGHT_HAND_SLICE][:, 0::3], -orig_lh_x, atol=1e-6,
        )

    def test_both_hands_yz_correctly_swapped(self):
        aug         = SpatialAugmenter()
        arr         = make_two_handed_clip(T=50, seed=42)
        orig_lh_yz  = arr[:, LEFT_HAND_SLICE][:, 1::3].copy()
        orig_lh_z   = arr[:, LEFT_HAND_SLICE][:, 2::3].copy()
        orig_rh_yz  = arr[:, RIGHT_HAND_SLICE][:, 1::3].copy()
        orig_rh_z   = arr[:, RIGHT_HAND_SLICE][:, 2::3].copy()
        rng         = np.random.default_rng(42)
        result      = aug.spatial_flip(arr, rng, min_hand_presence=0.0)
        np.testing.assert_allclose(result[:, LEFT_HAND_SLICE][:, 1::3], orig_rh_yz, atol=1e-6)
        np.testing.assert_allclose(result[:, LEFT_HAND_SLICE][:, 2::3], orig_rh_z,  atol=1e-6)
        np.testing.assert_allclose(result[:, RIGHT_HAND_SLICE][:, 1::3], orig_lh_yz, atol=1e-6)

    def test_pose_x_negated_two_handed(self):
        aug           = SpatialAugmenter()
        arr           = make_two_handed_clip(T=50, seed=42)
        orig_pose_x   = arr[:, POSE_SLICE][:, 0::3].copy()
        rng           = np.random.default_rng(42)
        result        = aug.spatial_flip(arr, rng, min_hand_presence=0.0)
        np.testing.assert_allclose(result[:, POSE_SLICE][:, 0::3], -orig_pose_x, atol=1e-6)

    # -- Case 2: LH only detected ---------------------------------------------

    def test_lh_only_data_moves_to_rh_slot(self):
        aug          = SpatialAugmenter()
        arr          = make_lh_only_clip(T=20, seed=7)
        orig_lh_yz   = arr[:, LEFT_HAND_SLICE][:, 1::3].copy()
        rng          = np.random.default_rng(42)
        result       = aug.spatial_flip(arr, rng, min_hand_presence=0.0)
        np.testing.assert_allclose(
            result[:, RIGHT_HAND_SLICE][:, 1::3], orig_lh_yz, atol=1e-6,
        )

    def test_lh_only_lh_slot_zeroed_after_flip(self):
        aug    = SpatialAugmenter()
        arr    = make_lh_only_clip(T=20, seed=7)
        rng    = np.random.default_rng(42)
        result = aug.spatial_flip(arr, rng, min_hand_presence=0.0)
        np.testing.assert_array_equal(
            result[:, LEFT_HAND_SLICE],
            np.zeros((20, N_HAND_FEATURES), dtype=np.float32),
        )

    def test_lh_only_x_coords_negated_in_rh_slot(self):
        aug       = SpatialAugmenter()
        arr       = make_lh_only_clip(T=20, seed=7)
        orig_lh_x = arr[:, LEFT_HAND_SLICE][:, 0::3].copy()
        rng       = np.random.default_rng(42)
        result    = aug.spatial_flip(arr, rng, min_hand_presence=0.0)
        np.testing.assert_allclose(
            result[:, RIGHT_HAND_SLICE][:, 0::3], -orig_lh_x, atol=1e-6,
        )

    def test_lh_only_pose_x_negated(self):
        aug           = SpatialAugmenter()
        arr           = make_lh_only_clip(T=20, seed=7)
        orig_pose_x   = arr[:, POSE_SLICE][:, 0::3].copy()
        rng           = np.random.default_rng(42)
        result        = aug.spatial_flip(arr, rng, min_hand_presence=0.0)
        np.testing.assert_allclose(
            result[:, POSE_SLICE][:, 0::3], -orig_pose_x, atol=1e-6,
        )

    # -- Case 3: RH only detected ---------------------------------------------

    def test_rh_only_data_moves_to_lh_slot(self):
        aug          = SpatialAugmenter()
        arr          = make_rh_only_clip(T=20, seed=7)
        orig_rh_yz   = arr[:, RIGHT_HAND_SLICE][:, 1::3].copy()
        rng          = np.random.default_rng(42)
        result       = aug.spatial_flip(arr, rng, min_hand_presence=0.0)
        np.testing.assert_allclose(
            result[:, LEFT_HAND_SLICE][:, 1::3], orig_rh_yz, atol=1e-6,
        )

    def test_rh_only_rh_slot_zeroed_after_flip(self):
        aug    = SpatialAugmenter()
        arr    = make_rh_only_clip(T=20, seed=7)
        rng    = np.random.default_rng(42)
        result = aug.spatial_flip(arr, rng, min_hand_presence=0.0)
        np.testing.assert_array_equal(
            result[:, RIGHT_HAND_SLICE],
            np.zeros((20, N_HAND_FEATURES), dtype=np.float32),
        )

    def test_rh_only_x_coords_negated_in_lh_slot(self):
        aug       = SpatialAugmenter()
        arr       = make_rh_only_clip(T=20, seed=7)
        orig_rh_x = arr[:, RIGHT_HAND_SLICE][:, 0::3].copy()
        rng       = np.random.default_rng(42)
        result    = aug.spatial_flip(arr, rng, min_hand_presence=0.0)
        np.testing.assert_allclose(
            result[:, LEFT_HAND_SLICE][:, 0::3], -orig_rh_x, atol=1e-6,
        )

    # -- Case 4: neither hand — zero-fill invariant ---------------------------

    def test_zero_fill_frames_unchanged_in_flip_safe_clip(self):
        aug  = SpatialAugmenter()
        arr, zero_indices = make_mixed_clip(T=50, zero_fraction=0.3, seed=0)
        rng  = np.random.default_rng(42)
        result = aug.spatial_flip(arr, rng, min_hand_presence=0.30)
        for idx in zero_indices:
            np.testing.assert_array_equal(
                result[idx, LEFT_HAND_SLICE],
                np.zeros(N_HAND_FEATURES, dtype=np.float32),
            )
            np.testing.assert_array_equal(
                result[idx, RIGHT_HAND_SLICE],
                np.zeros(N_HAND_FEATURES, dtype=np.float32),
            )

    # -- Hybrid policy: mixed-detection clip (all four cases) -----------------

    def test_mixed_detection_clip_cases_independent(self):
        """
        In a clip with all four detection states, verify each case is handled
        independently and correctly.
        """
        aug = SpatialAugmenter()
        arr = make_mixed_detection_clip(T=40, seed=5)
        rng = np.random.default_rng(42)
        result = aug.spatial_flip(arr, rng, min_hand_presence=0.0)
        q = 10

        # Case 1 — both hands [0:q]
        orig_rh_yz = arr[:q, RIGHT_HAND_SLICE][:, 1::3]
        np.testing.assert_allclose(result[:q, LEFT_HAND_SLICE][:, 1::3], orig_rh_yz, atol=1e-6)

        # Case 2 — LH only [q:2q]
        orig_lh_yz_c2 = arr[q:2*q, LEFT_HAND_SLICE][:, 1::3]
        np.testing.assert_allclose(result[q:2*q, RIGHT_HAND_SLICE][:, 1::3], orig_lh_yz_c2, atol=1e-6)
        np.testing.assert_array_equal(
            result[q:2*q, LEFT_HAND_SLICE],
            np.zeros((q, N_HAND_FEATURES), dtype=np.float32),
        )

        # Case 3 — RH only [2q:3q]
        orig_rh_yz_c3 = arr[2*q:3*q, RIGHT_HAND_SLICE][:, 1::3]
        np.testing.assert_allclose(result[2*q:3*q, LEFT_HAND_SLICE][:, 1::3], orig_rh_yz_c3, atol=1e-6)
        np.testing.assert_array_equal(
            result[2*q:3*q, RIGHT_HAND_SLICE],
            np.zeros((q, N_HAND_FEATURES), dtype=np.float32),
        )

        # Case 4 — neither [3q:]
        np.testing.assert_array_equal(result[3*q:, LEFT_HAND_SLICE], arr[3*q:, LEFT_HAND_SLICE])
        np.testing.assert_array_equal(result[3*q:, RIGHT_HAND_SLICE], arr[3*q:, RIGHT_HAND_SLICE])

    # -- Involution property (double-flip = identity) -------------------------

    def test_double_flip_identity_two_handed(self):
        aug  = SpatialAugmenter()
        arr  = make_two_handed_clip(T=50)
        rng1 = np.random.default_rng(0)
        rng2 = np.random.default_rng(0)
        once  = aug.spatial_flip(arr,  rng1, min_hand_presence=0.0)
        twice = aug.spatial_flip(once, rng2, min_hand_presence=0.0)
        np.testing.assert_allclose(twice, arr, atol=1e-6)

    def test_double_flip_identity_lh_only(self):
        aug  = SpatialAugmenter()
        arr  = make_lh_only_clip(T=30)
        rng1 = np.random.default_rng(0)
        rng2 = np.random.default_rng(0)
        once  = aug.spatial_flip(arr,  rng1, min_hand_presence=0.0)
        twice = aug.spatial_flip(once, rng2, min_hand_presence=0.0)
        np.testing.assert_allclose(twice, arr, atol=1e-6)

    def test_double_flip_identity_rh_only(self):
        aug  = SpatialAugmenter()
        arr  = make_rh_only_clip(T=30)
        rng1 = np.random.default_rng(0)
        rng2 = np.random.default_rng(0)
        once  = aug.spatial_flip(arr,  rng1, min_hand_presence=0.0)
        twice = aug.spatial_flip(once, rng2, min_hand_presence=0.0)
        np.testing.assert_allclose(twice, arr, atol=1e-6)

    def test_double_flip_identity_mixed_detection(self):
        aug  = SpatialAugmenter()
        arr  = make_mixed_detection_clip(T=40)
        rng1 = np.random.default_rng(0)
        rng2 = np.random.default_rng(0)
        once  = aug.spatial_flip(arr,  rng1, min_hand_presence=0.0)
        twice = aug.spatial_flip(once, rng2, min_hand_presence=0.0)
        np.testing.assert_allclose(twice, arr, atol=1e-6)

    def test_negative_threshold_clamped_to_zero(self):
        aug    = SpatialAugmenter()
        arr    = make_two_handed_clip(T=20)
        rng    = np.random.default_rng(0)
        result = aug.spatial_flip(arr, rng, min_hand_presence=-0.5)
        assert result.shape == arr.shape

    def test_threshold_above_one_clamped(self):
        aug    = SpatialAugmenter()
        arr    = make_two_handed_clip(T=20)
        rng    = np.random.default_rng(0)
        result = aug.spatial_flip(arr, rng, min_hand_presence=1.5)
        assert result.shape == arr.shape


# ---------------------------------------------------------------------------
# Section 4 — SpatialAugmenter: rotation_2d
# ---------------------------------------------------------------------------


class TestRotation2D:
    """Tests for SpatialAugmenter.rotation_2d."""

    def test_shape_preserved(self):
        aug    = SpatialAugmenter()
        arr    = make_two_handed_clip(T=50)
        rng    = np.random.default_rng(42)
        result = aug.rotation_2d(arr, rng, max_deg=5.0)
        assert result.shape == arr.shape

    def test_dtype_is_float32(self):
        aug    = SpatialAugmenter()
        arr    = make_two_handed_clip(T=50)
        rng    = np.random.default_rng(42)
        result = aug.rotation_2d(arr, rng, max_deg=5.0)
        assert result.dtype == np.float32

    def test_input_not_mutated(self):
        aug     = SpatialAugmenter()
        arr     = make_two_handed_clip(T=50)
        arr_ref = arr.copy()
        rng     = np.random.default_rng(42)
        aug.rotation_2d(arr, rng, max_deg=5.0)
        np.testing.assert_array_equal(arr, arr_ref)

    def test_zero_deg_returns_equal_array(self):
        aug    = SpatialAugmenter()
        arr    = make_two_handed_clip(T=50)
        rng    = np.random.default_rng(42)
        result = aug.rotation_2d(arr, rng, max_deg=0.0)
        np.testing.assert_allclose(result, arr, atol=1e-6)

    def test_zero_fill_frames_unchanged(self):
        aug  = SpatialAugmenter()
        arr, zero_indices = make_mixed_clip(T=50, zero_fraction=0.4)
        rng  = np.random.default_rng(42)
        result = aug.rotation_2d(arr, rng, max_deg=5.0)
        for idx in zero_indices:
            np.testing.assert_array_equal(
                result[idx, LEFT_HAND_SLICE],
                np.zeros(N_HAND_FEATURES, dtype=np.float32),
            )
            np.testing.assert_array_equal(
                result[idx, RIGHT_HAND_SLICE],
                np.zeros(N_HAND_FEATURES, dtype=np.float32),
            )

    def test_detected_frames_never_become_all_zero(self):
        aug    = SpatialAugmenter()
        arr    = make_two_handed_clip(T=50)
        rng    = np.random.default_rng(42)
        result = aug.rotation_2d(arr, rng, max_deg=45.0)
        for t in range(50):
            assert not (result[t] == 0.0).all()

    def test_pose_slice_unchanged(self):
        aug       = SpatialAugmenter()
        arr       = make_two_handed_clip(T=50)
        orig_pose = arr[:, POSE_SLICE].copy()
        rng       = np.random.default_rng(42)
        result    = aug.rotation_2d(arr, rng, max_deg=5.0)
        np.testing.assert_array_equal(result[:, POSE_SLICE], orig_pose)

    def test_z_coordinates_unchanged_lh(self):
        aug        = SpatialAugmenter()
        arr        = make_two_handed_clip(T=50)
        orig_lh_z  = arr[:, LEFT_HAND_SLICE][:, 2::3].copy()
        rng        = np.random.default_rng(42)
        result     = aug.rotation_2d(arr, rng, max_deg=5.0)
        np.testing.assert_allclose(result[:, LEFT_HAND_SLICE][:, 2::3], orig_lh_z, atol=1e-6)

    def test_z_coordinates_unchanged_rh(self):
        aug        = SpatialAugmenter()
        arr        = make_two_handed_clip(T=50)
        orig_rh_z  = arr[:, RIGHT_HAND_SLICE][:, 2::3].copy()
        rng        = np.random.default_rng(42)
        result     = aug.rotation_2d(arr, rng, max_deg=5.0)
        np.testing.assert_allclose(result[:, RIGHT_HAND_SLICE][:, 2::3], orig_rh_z, atol=1e-6)

    def test_rotation_preserves_wrist_to_landmark_distances_lh(self):
        """Rotation is rigid: wrist-to-landmark distances preserved in xy plane."""
        aug  = SpatialAugmenter()
        arr  = make_two_handed_clip(T=30, seed=0)
        rng  = np.random.default_rng(42)
        result = aug.rotation_2d(arr, rng, max_deg=30.0)
        for t in range(30):
            wrist_before = arr[t, LEFT_HAND_SLICE.start    : LEFT_HAND_SLICE.start + 2]
            wrist_after  = result[t, LEFT_HAND_SLICE.start : LEFT_HAND_SLICE.start + 2]
            for lm_idx in range(1, N_HAND_LANDMARKS):
                base = LEFT_HAND_SLICE.start + lm_idx * N_COORDS_PER_LANDMARK
                lm_before = arr[t, base    : base + 2]
                lm_after  = result[t, base : base + 2]
                dist_before = np.linalg.norm(lm_before.astype(np.float64) - wrist_before.astype(np.float64))
                dist_after  = np.linalg.norm(lm_after.astype(np.float64)  - wrist_after.astype(np.float64))
                assert abs(dist_before - dist_after) < 1e-4

    def test_rotation_preserves_pairwise_distances_rh(self):
        """Pairwise RH landmark distances in xy must be preserved by rotation."""
        aug  = SpatialAugmenter()
        arr  = make_two_handed_clip(T=30, seed=1)
        rng  = np.random.default_rng(42)
        result = aug.rotation_2d(arr, rng, max_deg=30.0)
        rng_pairs = np.random.default_rng(999)
        pairs = [
            (int(rng_pairs.integers(0, N_HAND_LANDMARKS)),
             int(rng_pairs.integers(0, N_HAND_LANDMARKS)))
            for _ in range(5)
        ]
        for t in range(30):
            for (i, j) in pairs:
                if i == j:
                    continue
                base_i = RIGHT_HAND_SLICE.start + i * N_COORDS_PER_LANDMARK
                base_j = RIGHT_HAND_SLICE.start + j * N_COORDS_PER_LANDMARK
                lm_i_before = arr[t, base_i    : base_i + 2].astype(np.float64)
                lm_j_before = arr[t, base_j    : base_j + 2].astype(np.float64)
                lm_i_after  = result[t, base_i : base_i + 2].astype(np.float64)
                lm_j_after  = result[t, base_j : base_j + 2].astype(np.float64)
                dist_before = np.linalg.norm(lm_i_before - lm_j_before)
                dist_after  = np.linalg.norm(lm_i_after  - lm_j_after)
                assert abs(dist_before - dist_after) < 1e-4


# ---------------------------------------------------------------------------
# Section 5 — AugmentationPipeline
# ---------------------------------------------------------------------------


class TestAugmentationPipeline:
    """Tests for the AugmentationPipeline orchestrator."""

    def test_shape_invariant_through_full_chain(self):
        cfg      = _load_augmentation_config("spatial_temporal")
        pipeline = AugmentationPipeline(cfg, seed=42, flip_min_hand_presence=0.30)
        arr      = make_two_handed_clip(T=50)
        result   = pipeline(arr, clip_idx=42)
        assert result.shape == arr.shape

    def test_dtype_is_float32_when_enabled(self):
        cfg      = _load_augmentation_config("spatial_temporal")
        pipeline = AugmentationPipeline(cfg, seed=42)
        arr      = make_two_handed_clip(T=50)
        result   = pipeline(arr, clip_idx=0)
        assert result.dtype == np.float32

    def test_dtype_is_float32_when_disabled(self):
        cfg      = _load_augmentation_config("none")
        pipeline = AugmentationPipeline(cfg, seed=42)
        arr      = make_two_handed_clip(T=50)
        result   = pipeline(arr, clip_idx=0)
        assert result.dtype == np.float32

    def test_accepts_float64_input(self):
        cfg      = _load_augmentation_config("spatial_temporal")
        pipeline = AugmentationPipeline(cfg, seed=42)
        arr      = make_two_handed_clip(T=30).astype(np.float64)
        result   = pipeline(arr, clip_idx=0)
        assert result.dtype == np.float32
        assert result.shape == (30, FEATURE_SIZE)

    def test_pipeline_owns_copy_boundary_enabled(self):
        """
        CRITICAL: The original input array must NOT be mutated by the pipeline.
        """
        cfg      = _load_augmentation_config("spatial_temporal")
        pipeline = AugmentationPipeline(cfg, seed=42, flip_min_hand_presence=0.30)
        arr      = make_two_handed_clip(T=50)
        arr_ref  = arr.copy()
        pipeline(arr, clip_idx=1)
        pipeline(arr, clip_idx=2)
        np.testing.assert_array_equal(arr, arr_ref)

    def test_pipeline_owns_copy_boundary_disabled(self):
        cfg      = _load_augmentation_config("none")
        pipeline = AugmentationPipeline(cfg, seed=42)
        arr      = make_two_handed_clip(T=50)
        arr_ref  = arr.copy()
        pipeline(arr, clip_idx=0)
        np.testing.assert_array_equal(arr, arr_ref)

    def test_return_is_new_object(self):
        cfg      = _load_augmentation_config("none")
        pipeline = AugmentationPipeline(cfg, seed=42)
        arr      = make_two_handed_clip(T=20)
        result   = pipeline(arr, clip_idx=0)
        assert result is not arr

    def test_disabled_returns_content_equal_to_input(self):
        cfg      = _load_augmentation_config("none")
        pipeline = AugmentationPipeline(cfg, seed=42)
        arr      = make_two_handed_clip(T=50)
        result   = pipeline(arr, clip_idx=42)
        np.testing.assert_array_equal(result, arr)

    def test_same_clip_idx_same_output(self):
        cfg      = _load_augmentation_config("spatial_temporal")
        pipeline = AugmentationPipeline(cfg, seed=42, flip_min_hand_presence=0.30)
        arr      = make_two_handed_clip(T=50)
        result1  = pipeline(arr, clip_idx=7)
        result2  = pipeline(arr, clip_idx=7)
        np.testing.assert_array_equal(result1, result2)

    def test_same_clip_idx_same_output_across_pipeline_instances(self):
        cfg = _load_augmentation_config("spatial_temporal")
        p1  = AugmentationPipeline(cfg, seed=42, flip_min_hand_presence=0.30)
        p2  = AugmentationPipeline(cfg, seed=42, flip_min_hand_presence=0.30)
        arr = make_two_handed_clip(T=50)
        r1  = p1(arr, clip_idx=99)
        r2  = p2(arr, clip_idx=99)
        np.testing.assert_array_equal(r1, r2)

    def test_different_clip_idx_different_output(self):
        cfg      = _load_augmentation_config("spatial_temporal")
        pipeline = AugmentationPipeline(cfg, seed=42, flip_min_hand_presence=0.30)
        arr      = make_two_handed_clip(T=50)
        n_trials = 10
        n_differ = sum(
            1 for idx in range(n_trials)
            if not np.array_equal(pipeline(arr, clip_idx=idx), pipeline(arr, clip_idx=idx + 100))
        )
        assert n_differ >= 9

    def test_different_seeds_different_output(self):
        cfg = _load_augmentation_config("spatial_temporal")
        p1  = AugmentationPipeline(cfg, seed=42)
        p2  = AugmentationPipeline(cfg, seed=99)
        arr = make_two_handed_clip(T=50)
        assert not np.array_equal(p1(arr, clip_idx=0), p2(arr, clip_idx=0))


    def test_one_handed_absent_slot_invariant_through_full_chain(self):
        """
        PER-SLOT INVARIANT for one-handed clips through the full chain.

        A one-handed clip (LH always absent) must have its LH slot remain
        exactly zero through the entire augmentation chain including
        gaussian_noise (per-slot fix) and spatial_flip (min_hand_presence=1.0
        forces no flip, preserving the one-handed structure).
        """
        cfg      = _load_augmentation_config("spatial_temporal")
        # Use min_hand_presence=1.0 to prevent flip from reassigning slots
        pipeline = AugmentationPipeline(cfg, seed=42, flip_min_hand_presence=1.0)
        arr      = make_one_handed_clip(T=50)   # LH=0, RH≠0
        result   = pipeline(arr, clip_idx=5)
        np.testing.assert_array_equal(
            result[:, LEFT_HAND_SLICE],
            np.zeros((50, N_HAND_FEATURES), dtype=np.float32),
            err_msg=(
                "LH slot corrupted through full augmentation chain for one-handed clip. "
                "Per-slot noise masking should prevent this."
            ),
        )

    # -- Metadata ---------------------------------------------------------

    def test_metadata_contains_all_required_keys(self):
        cfg      = _load_augmentation_config("spatial_temporal")
        pipeline = AugmentationPipeline(cfg, seed=42)
        meta     = pipeline.get_metadata()
        required_keys = [
            "enabled", "temporal_jitter", "frame_drop_prob", "speed_jitter",
            "gaussian_noise_std", "gaussian_noise_detected_only", "rotation_deg",
            "spatial_flip", "flip_min_hand_presence", "chain_order",
            "base_seed", "rng_seed_derivation",
        ]
        for key in required_keys:
            assert key in meta, f"Required metadata key '{key}' missing"

    def test_metadata_chain_order_correct(self):
        cfg      = _load_augmentation_config("spatial_temporal")
        pipeline = AugmentationPipeline(cfg, seed=42)
        meta     = pipeline.get_metadata()
        expected = ["temporal_jitter", "speed_jitter", "gaussian_noise", "rotation_2d", "spatial_flip"]
        assert meta["chain_order"] == expected

    def test_metadata_reflects_config(self):
        cfg      = _load_augmentation_config("spatial_temporal")
        pipeline = AugmentationPipeline(cfg, seed=77, flip_min_hand_presence=0.25)
        meta     = pipeline.get_metadata()
        assert meta["enabled"]            == cfg.enabled
        assert meta["temporal_jitter"]    == cfg.temporal_jitter
        assert meta["frame_drop_prob"]    == pytest.approx(cfg.frame_drop_prob)
        assert meta["gaussian_noise_std"] == pytest.approx(cfg.gaussian_noise_std)
        assert meta["rotation_deg"]       == pytest.approx(cfg.rotation_deg)
        assert meta["spatial_flip"]       == cfg.spatial_flip
        assert meta["base_seed"]          == 77
        assert meta["flip_min_hand_presence"] == pytest.approx(0.25)

    def test_metadata_documents_noise_mask_granularity(self):
        """Metadata must document the per-slot noise masking strategy."""
        cfg      = _load_augmentation_config("spatial_temporal")
        pipeline = AugmentationPipeline(cfg, seed=42)
        meta     = pipeline.get_metadata()
        assert "gaussian_noise_mask_granularity" in meta
        assert meta["gaussian_noise_mask_granularity"] == "per_component_slot"

    def test_metadata_documents_fast_clip_strategy(self):
        """Metadata must document that fast clips are interpolated (not zero-padded)."""
        cfg      = _load_augmentation_config("spatial_temporal")
        pipeline = AugmentationPipeline(cfg, seed=42)
        meta     = pipeline.get_metadata()
        assert "speed_jitter_fast_clip_strategy" in meta
        assert "interpolate" in meta["speed_jitter_fast_clip_strategy"]

    def test_repr_contains_status(self):
        cfg      = _load_augmentation_config("spatial_temporal")
        pipeline = AugmentationPipeline(cfg, seed=42)
        assert "ENABLED" in repr(pipeline)

    def test_repr_disabled(self):
        cfg      = _load_augmentation_config("none")
        pipeline = AugmentationPipeline(cfg, seed=42)
        assert "DISABLED" in repr(pipeline)

def test_speed_jitter_slot_zero_fill_invariant():
    """
    speed_jitter must not hallucinate non-zero values from zero source frames.
 
    Correct invariant
    -----------------
    speed_jitter is a temporal resampling transform. The TEMPORAL POSITION
    of zero-fill frames can shift (a sign performed faster has its zero-fill
    region at a different temporal position). The invariant is NOT about
    fixed frame indices staying zero.
 
    The correct invariant is: if a component SLOT is zero in ALL T source
    frames, it must remain zero in all T output frames. No non-zero value
    can be created from pure zero source data.
 
    Test design
    -----------
    Use a LH-always-absent clip (one-handed sign): LH slot is zero in every
    one of the 60 source frames. After speed_jitter at any rate:
    - Slow path (rate < 1.0): integer indexing → output[t] = source[i] → LH zero
    - Fast path (rate > 1.0): zero-aware interp → both neighbours zero → forced zero
    Both paths must produce LH=0 for all T output frames.
 
    Also test a clip-level invariant: an all-zero clip stays all-zero.
 
    Note on the previous (broken) assertion
    ----------------------------------------
    The old assertion `result[50:, RH].sum() == 0` after setting
    `arr[45:, RH] = 0` is WRONG for the slow path (rate < 1.0).
    When rate ≈ 0.85, output frame 50 maps to source frame 44 (non-zero RH).
    The old assertion rejected correct, valid behaviour.
    """
    aug = TemporalAugmenter()
 
    # --- Invariant 1: all-zero clip stays all-zero for all speeds ---
    arr_zeros = np.zeros((60, FEATURE_SIZE), dtype=np.float32)
    for seed in range(10):
        rng = np.random.default_rng(seed)
        result = aug.speed_jitter(arr_zeros, rng, speed_range=(0.7, 1.3))
        assert result.shape == arr_zeros.shape
        assert (result == 0.0).all(), (
            f"seed={seed}: all-zero clip produced non-zero output after speed_jitter."
        )
 
    # --- Invariant 2: LH always-absent slot stays zero (one-handed sign) ---
    # LH is zero in EVERY source frame → must be zero in EVERY output frame.
    rng_data = np.random.default_rng(42)
    arr_lh_absent = rng_data.uniform(0.1, 0.9, size=(60, FEATURE_SIZE)).astype(np.float32)
    arr_lh_absent[:, LEFT_HAND_SLICE] = 0.0   # LH absent in ALL 60 frames
 
    # Test across a range of seeds to cover both slow (rate<1) and fast (rate>1) paths
    for seed in range(20):
        rng = np.random.default_rng(seed)
        result = aug.speed_jitter(arr_lh_absent, rng, speed_range=(0.7, 1.3))
        assert result.shape == arr_lh_absent.shape, (
            f"seed={seed}: shape changed after speed_jitter."
        )
        assert (result[:, LEFT_HAND_SLICE] == 0.0).all(), (
            f"seed={seed}: LH always-absent slot was corrupted by speed_jitter. "
            "The zero-fill invariant failed for a one-handed sign's absent hand slot."
        )
 
    # --- Invariant 3: RH always-absent slot stays zero (LH-dominant sign) ---
    arr_rh_absent = rng_data.uniform(0.1, 0.9, size=(60, FEATURE_SIZE)).astype(np.float32)
    arr_rh_absent[:, RIGHT_HAND_SLICE] = 0.0  # RH absent in ALL 60 frames
 
    for seed in range(20):
        rng = np.random.default_rng(seed)
        result = aug.speed_jitter(arr_rh_absent, rng, speed_range=(0.7, 1.3))
        assert (result[:, RIGHT_HAND_SLICE] == 0.0).all(), (
            f"seed={seed}: RH always-absent slot was corrupted by speed_jitter."
        )
 
    # --- What about PARTIAL zero-fill (tail zeros)? ---
    # arr[45:, RH] = 0 does NOT imply result[45:, RH] = 0 after speed_jitter.
    # The zero region shifts temporally. This is correct behaviour, not a bug.
    # We verify that output values that DID come from non-zero source frames
    # are indeed non-zero (no information is destroyed from detected frames).
    arr_partial = rng_data.uniform(0.1, 0.9, size=(60, FEATURE_SIZE)).astype(np.float32)
    arr_partial[:, LEFT_HAND_SLICE] = 0.0   # LH always absent
    arr_partial[45:, RIGHT_HAND_SLICE] = 0.0 # RH absent in last 15 frames only
 
    rng = np.random.default_rng(42)
    result_partial = aug.speed_jitter(arr_partial, rng, speed_range=(1.3, 1.5))
    assert result_partial.shape == arr_partial.shape
 
    # The only invariant we can assert on partial zeros:
    # LH slot (always absent) must stay zero
    assert (result_partial[:, LEFT_HAND_SLICE] == 0.0).all(), (
        "LH always-absent slot corrupted when combined with partial RH zeros."
    )