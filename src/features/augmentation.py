"""
src/features/augmentation.py
==============================
Data augmentation transforms for the WLASL gesture recognition pipeline.

Overview
--------
This module implements all data augmentation logic for the landmark-based
gesture recognition pipeline. Augmentation is critical here because the
working dataset is severely constrained: ~236 training clips across 35 classes
(~7 clips/class mean). Without aggressive augmentation, the LSTM will overfit
the training signers' idiosyncratic styles and fail to generalise to the 7
held-out validation signers.

Architecture
------------
Three layers of abstraction:

    TemporalAugmenter     — pure, stateless frame-level temporal transforms
    SpatialAugmenter      — pure, stateless coordinate-level spatial transforms
    AugmentationPipeline  — stateful orchestrator: owns seed + config reference,
                            coordinates the chain order, exposes the public API

Design Principles
-----------------
1. **Stateless transforms** — all TemporalAugmenter and SpatialAugmenter methods
   are pure functions with no hidden state or globals. The same inputs always
   produce the same outputs. Each transform can be called, tested, and profiled
   independently.

2. **Deterministic per clip** — each clip_idx produces a unique but fixed RNG
   derived from (base_seed XOR clip_idx). The same (base_seed, clip_idx) pair
   always produces the same augmented output, enabling reproducible debugging
   and ablation studies. Different clip_idx values produce different augmentations
   from the same base array.

3. **Zero-fill invariant** — no spatial transform ever modifies a frame where
   both hands are absent. This preserves the semantic signal of zero-fill
   (one-handed signs, detection failures). The left-hand 70.18% missing rate
   discovered in Notebook 03 is semantic — not noise — and must not be corrupted.

4. **Copy before transform** — the caller's array is never mutated. The single
   copy is made once in AugmentationPipeline.__call__() before any transform runs.
   Individual transform methods receive and return arrays without additional copies.

5. **Shape invariant** — every transform returns an array of identical shape
   (T, D) to its input. Temporal transforms that drop frames pad back to T.
   Spatial transforms never change T or D.

6. **Detected-only noise** — Gaussian noise is applied exclusively to detected
   (non-zero) landmark frames. Applying noise to a zero-filled frame would
   corrupt its identity as "no detection" — the LSTM would receive a near-zero
   but non-zero signal and incorrectly treat it as a detected-but-barely-visible
   hand. This was confirmed as mandatory by the Notebook 03 analysis.

Critical Implementation Note: spatial_flip
------------------------------------------
spatial_flip() is the highest-risk transform in this file.

It must do BOTH of the following simultaneously:

    (a) Negate every x-coordinate in the LH slice, RH slice, AND pose slice
        x-coords are at indices 0, 3, 6, ... within each slice (i.e. [::3])

    (b) Swap the LH and RH slices

If (a) is done without (b): landmarks are mirrored but hand labels are wrong.
    The model receives RH landmark data labelled as LH and vice versa.
    This produces anatomically impossible geometry.

If (b) is done without (a): LH and RH are swapped but x-coords are not mirrored.
    The model receives the correct label but wrong spatial geometry.
    This also produces anatomically impossible geometry.

Both conditions must be verified in test_augmentation.py.

Why pose x-coords must be negated: MediaPipe pose landmarks use a consistent
x-axis regardless of handedness. Mirroring a signer without mirroring their
pose produces a body where the left shoulder is on the right and hands are on
the wrong side — internally inconsistent geometry that would mislead the LSTM's
implicit body model.

Clip-Level Flip Safety
-----------------------
The safety check (min_hand_presence) is applied BEFORE the transform. If the
clip has insufficient both-hands presence (< min_hand_presence fraction of
frames have both hands simultaneously detected), the function returns the input
unchanged. This protects one-handed signs from being anatomically corrupted.

Notebook 03 finding: all 35 signs are classified as flip-safe at the 30%
threshold, but clip-level enforcement (not sign-level) is the correct policy
because individual clips of even two-handed signs can be one-handed-dominant.

Transform Chain Order
---------------------
1. temporal_jitter  — drop frames (changes WHICH frames exist)
2. speed_jitter     — resample speed (changes frame timing)
3. gaussian_noise   — add noise to detected frames
4. rotation_2d      — rotate hand coords around wrist
5. spatial_flip     — mirror and swap hands (last: operates on final spatial state)

Rationale for this order:
- Temporal before spatial: temporal transforms change the set of frames. If
  rotation is applied before temporal_jitter, some rotated frames get dropped
  and re-padded with unrotated zeros — creating inconsistent spatial treatment
  across frames. Applying temporal first ensures all remaining frames receive
  identical spatial treatment.
- Noise before flip: noise is applied to detected frames. After flip, what was
  the RH becomes the LH. If noise were applied after flip, the RH-turned-LH
  frames would get noise based on their new label, not their original detection
  status. Applying noise before flip ensures noise is based on actual detection.
- Flip last: flip is the most semantically disruptive transform. Applying it
  last ensures all other transforms have produced a consistent spatial state
  before the final mirroring step.

Notebook 03 Findings Incorporated
----------------------------------
- Speed jitter is the highest-priority augmentation: 9/35 signs show high
  motion-energy coefficient of variation across clips. These signs (black, before,
  like, finish, give, go, thanksgiving, cousin, and others) are executed at
  significantly different speeds across signers. Speed jitter directly addresses
  this variance.
- gaussian_noise std=0.01 is confirmed optimal: ~6% of mean xy signal (~0.16),
  large enough to regularise but small enough to preserve wrist-relative shape.
- Rotation ±5°: at this angle, the distal fingertip moves by sin(5°) × 0.15 ≈
  0.013 units — comparable to noise std=0.01. Augmentations have consistent
  signal scales.
- Frame dropout at 10%: cosine similarity of 1.0000 (std=0.0001) confirmed
  that dropping 10% of frames from a 67-frame mean clip is essentially invisible
  to the mean feature vector. Safe and risk-free.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.random import Generator

from src.features.constants import (
    AUGMENTATION_FRAME_DROP_PROB_DEFAULT,
    AUGMENTATION_NOISE_STD_DEFAULT,
    AUGMENTATION_ROTATION_DEG_DEFAULT,
    AUGMENTATION_SPEED_RANGE,
    FEATURE_SIZE,
    FLIP_MIN_HAND_PRESENCE_DEFAULT,
    LEFT_HAND_SLICE,
    N_COORDS_PER_LANDMARK,
    N_HAND_FEATURES,
    N_HAND_LANDMARKS,
    POSE_SLICE,
    RIGHT_HAND_SLICE,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Internal validation helpers
# ---------------------------------------------------------------------------

def _validate_landmark_array(arr: np.ndarray, caller: str) -> None:
    """
    Assert that arr is a 2D float32 array of shape (T, FEATURE_SIZE).

    Parameters
    ----------
    arr : np.ndarray
        Array to validate.
    caller : str
        Name of the calling method, used in the error message.

    Raises
    ------
    ValueError
        If arr is not 2D or if arr.shape[1] != FEATURE_SIZE.
    TypeError
        If arr is not a numpy ndarray.
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError(
            f"{caller}: expected np.ndarray, got {type(arr).__name__}."
        )
    if arr.ndim != 2:
        raise ValueError(
            f"{caller}: expected 2D array (T, {FEATURE_SIZE}), "
            f"got ndim={arr.ndim}, shape={arr.shape}."
        )
    if arr.shape[1] != FEATURE_SIZE:
        raise ValueError(
            f"{caller}: expected feature dimension {FEATURE_SIZE}, "
            f"got {arr.shape[1]}. "
            "Always pass the full 225-dim array to augmentation transforms."
        )


# ---------------------------------------------------------------------------
# TemporalAugmenter
# ---------------------------------------------------------------------------

class TemporalAugmenter:
    """
    Frame-level temporal transforms operating on (T, D) arrays.

    All methods are pure functions with no state. The same inputs always
    produce the same outputs. All methods:

    - Accept a (T, D) float32 array and a numpy Generator
    - Return a (T, D) float32 array of IDENTICAL shape to the input
    - Never mutate the input array (they operate on views/slices, not copies)
    - Never reduce T below 1 frame (edge case handling)

    The shape-invariant guarantee means temporal transforms can be composed
    safely with spatial transforms without shape bookkeeping in the pipeline.
    """

    def temporal_jitter(
        self,
        arr: np.ndarray,
        rng: Generator,
        drop_prob: float = AUGMENTATION_FRAME_DROP_PROB_DEFAULT,
    ) -> np.ndarray:
        """
        Randomly drop a fraction of frames, then right-pad back to original length.

        Simulates the MediaPipe detection failure pattern: in real inference,
        some frames yield no detection and are zero-filled. By randomly dropping
        frames and padding with zeros, we train the LSTM to be robust to
        intermittent missing frames — a condition it will definitely encounter
        at inference time.

        Algorithm:
            1. Draw boolean keep-mask: rng.random(T) > drop_prob
            2. Guard: if all frames would be dropped, keep at least 1
               (prevents degenerate all-zero clips that waste gradient steps)
            3. arr_kept = arr[mask]   — integer indexing, C-contiguous result
            4. Right-pad with zeros to restore shape to (T, D)

        Why right-pad (not left-pad or random-pad):
            Right-padding is consistent with _pad_or_truncate in FeaturePipeline,
            which also right-pads. The LSTM already learns that trailing zeros
            are non-informative (they are semantically identical to genuine
            zero-fill frames). Padding at the end keeps the sign's temporal
            midpoint closer to the start of the sequence, which is where the
            LSTM's hidden state has the most capacity (it hasn't been diluted by
            many timesteps of zeros yet).

        Parameters
        ----------
        arr : np.ndarray
            Input landmark array, shape (T, D). Not mutated.
        rng : numpy.random.Generator
            Per-clip RNG from AugmentationPipeline.
        drop_prob : float
            Probability of dropping each frame. Must be in [0.0, 1.0).
            Default 0.10 (10% dropout confirmed safe in Notebook 03).

        Returns
        -------
        np.ndarray
            Shape (T, D), dtype float32. Right-padded with zeros where frames
            were dropped.
        """
        _validate_landmark_array(arr, "temporal_jitter")

        T, D = arr.shape

        if drop_prob <= 0.0:
            return arr

        # Build keep-mask: True where frame is retained
        keep_mask = rng.random(T) > drop_prob  # (T,) bool

        # Guard: keep at least 1 frame to avoid degenerate all-zero clips
        if not keep_mask.any():
            # Force-keep the frame with the most landmark signal (most non-zero values)
            signal_per_frame = (arr != 0.0).sum(axis=1)  # (T,)
            keep_idx = int(np.argmax(signal_per_frame))
            keep_mask[keep_idx] = True

        arr_kept = arr[keep_mask]         # shape (n_kept, D), guaranteed ≥ 1 row
        n_kept   = arr_kept.shape[0]
        n_pad    = T - n_kept

        if n_pad == 0:
            return arr_kept.astype(np.float32)

        padding = np.zeros((n_pad, D), dtype=np.float32)
        result  = np.concatenate([arr_kept, padding], axis=0)

        return result.astype(np.float32)

    def speed_jitter(
        self,
        arr: np.ndarray,
        rng: Generator,
        speed_range: tuple[float, float] = AUGMENTATION_SPEED_RANGE,
    ) -> np.ndarray:
        """
        Resample the clip at a random playback speed, preserving output shape.

        Simulates the same sign executed at different speeds — a pervasive source
        of within-class variance in WLASL (Notebook 03 F9: 9/35 signs have high
        motion-energy CV across clips). The same physical gesture takes ~0.7x to
        1.3x the time depending on the signer's cadence, fatigue, and style.

        Algorithm:
            1. Draw rate r from Uniform(speed_range[0], speed_range[1])
               r < 1.0 → signer is slower → clip appears longer
               r > 1.0 → signer is faster → clip appears shorter
            2. Compute resampled indices:
               n_resampled = max(1, round(T / r))
               new_indices = linspace(0, T-1, n_resampled), cast to int, clipped
            3. arr_resampled = arr[new_indices]
            4. If n_resampled > T (slower → more indices): centre-crop to T
            5. If n_resampled < T (faster → fewer indices): right-pad to T
            6. If n_resampled == T: no-op (unlikely but handled)

        Why integer indexing (not interpolation):
            Interpolated landmark positions are synthetic data MediaPipe never
            actually produces. At any given frame, MediaPipe outputs one specific
            set of landmark coordinates. Fractional positions between two frames
            create anatomically plausible but physically nonexistent poses that
            the LSTM has never been trained to interpret.
            Integer resampling selects real extracted frame positions, just at
            different temporal density — the LSTM has seen every selected frame
            during normal (non-augmented) training.

        Why centre-crop for slower clips:
            When r < 1.0, we have more indices than frames. We need to discard
            the surplus. Centre-cropping is preferred over end-cropping because
            ASL signs have preparatory and release movements at both ends, and
            the peak motion phase is concentrated in the temporal centre.

        Parameters
        ----------
        arr : np.ndarray
            Input landmark array, shape (T, D). Not mutated.
        rng : numpy.random.Generator
            Per-clip RNG from AugmentationPipeline.
        speed_range : tuple[float, float]
            (min_rate, max_rate). Default (0.7, 1.3): ±30% speed variation.
            Confirmed by Notebook 03 motion-energy CV analysis.

        Returns
        -------
        np.ndarray
            Shape (T, D), dtype float32.
        """
        _validate_landmark_array(arr, "speed_jitter")

        T, D = arr.shape
        lo, hi = float(speed_range[0]), float(speed_range[1])

        # Draw playback rate
        rate = rng.uniform(lo, hi)

        # Compute resampled frame count
        n_resampled = max(1, round(T / rate))

        # Build integer source indices, clamped to [0, T-1]
        float_indices = np.linspace(0, T - 1, n_resampled)
        int_indices   = np.clip(np.round(float_indices).astype(np.int64), 0, T - 1)

        arr_resampled = arr[int_indices]  # shape (n_resampled, D)

        if n_resampled == T:
            return arr_resampled.astype(np.float32)

        if n_resampled > T:
            # Slower clip: centre-crop excess frames
            excess = n_resampled - T
            start  = excess // 2
            result = arr_resampled[start : start + T]
            return result.astype(np.float32)

        # Faster clip: right-pad with zeros
        n_pad   = T - n_resampled
        padding = np.zeros((n_pad, D), dtype=np.float32)
        result  = np.concatenate([arr_resampled, padding], axis=0)
        return result.astype(np.float32)


# ---------------------------------------------------------------------------
# SpatialAugmenter
# ---------------------------------------------------------------------------

class SpatialAugmenter:
    """
    Landmark-coordinate spatial transforms operating on (T, D) arrays.

    All methods:
    - Accept a (T, D) float32 array and a numpy Generator
    - Return a (T, D) float32 array of IDENTICAL shape
    - Never mutate the input array
    - Respect the zero-fill invariant: frames where both hands are absent
      are never modified

    Detection masks (which frames have which components detected) are computed
    before any modification to the array. This ensures that the mask reflects
    the original detection state, not a modified state that might inadvertently
    create spurious non-zero values.
    """

    # ------------------------------------------------------------------
    # Detection mask helpers
    # ------------------------------------------------------------------

    def _get_lh_detected_mask(self, arr: np.ndarray) -> np.ndarray:
        """
        Boolean mask shape (T,) — True where left hand is detected.

        A frame has left hand detected if ANY value in LEFT_HAND_SLICE is
        non-zero. This is conservative: even a single non-zero landmark
        coordinate indicates MediaPipe detected the hand.

        The 'any' aggregation is correct here because after wrist-relative
        normalisation, the wrist landmark is subtracted to (0,0,0). A fully
        detected-and-normalised hand will have the wrist at (0,0,0) but all
        other landmarks at non-zero values. Using 'any' catches this.

        Note: FeaturePipeline normalises BEFORE augmentation is called, so
        in production this mask is computed on already-normalised arrays.
        In test contexts (raw arrays), the wrist is at its original position
        and all values are non-zero for detected hands — the mask still works.
        """
        return arr[:, LEFT_HAND_SLICE].any(axis=1)  # (T,) bool

    def _get_rh_detected_mask(self, arr: np.ndarray) -> np.ndarray:
        """
        Boolean mask shape (T,) — True where right hand is detected.

        See _get_lh_detected_mask for detection logic.
        """
        return arr[:, RIGHT_HAND_SLICE].any(axis=1)  # (T,) bool

    def _get_either_hand_detected_mask(self, arr: np.ndarray) -> np.ndarray:
        """
        Boolean mask shape (T,) — True where AT LEAST ONE hand is detected.

        Used by gaussian_noise and rotation_2d to identify frames eligible
        for spatial augmentation. The zero-fill invariant requires that frames
        where BOTH hands are absent are never modified. However, for spatial
        transforms we also want to augment frames where only one hand is present
        (one-handed signs, partial occlusion) — these are semantically valid
        and should be robustly trained.
        """
        lh = self._get_lh_detected_mask(arr)
        rh = self._get_rh_detected_mask(arr)
        return lh | rh  # (T,) bool — True if either hand detected

    def _get_both_hands_present_fraction(self, arr: np.ndarray) -> float:
        """
        Fraction of frames where BOTH hands are simultaneously detected.

        This is the clip-level safety check for spatial_flip. It answers
        the question: "is this clip geometrically safe to mirror?"

        One-handed signs naturally approach 0.0 (the non-dominant hand is
        absent in most frames). Two-handed signs typically range 0.3–0.9
        depending on clip quality and sign complexity.

        Clips below FLIP_MIN_HAND_PRESENCE_DEFAULT (0.30) are not flipped.
        This protects one-handed signs from anatomically impossible mirroring
        where the active hand changes side but the sign semantics remain
        handedness-specific.

        Returns 0.0 for empty arrays (edge case protection).
        """
        lh   = self._get_lh_detected_mask(arr)
        rh   = self._get_rh_detected_mask(arr)
        both = lh & rh   # (T,) bool — True where both present simultaneously
        return float(both.mean()) if len(both) > 0 else 0.0

    # ------------------------------------------------------------------
    # Gaussian noise
    # ------------------------------------------------------------------

    def gaussian_noise(
        self,
        arr: np.ndarray,
        rng: Generator,
        std: float = AUGMENTATION_NOISE_STD_DEFAULT,
        detected_only: bool = True,
    ) -> np.ndarray:
        """
        Add independent Gaussian noise N(0, std) to landmark coordinates.

        This is the primary regularisation augmentation. It prevents the model
        from memorising exact landmark coordinate values that are specific to
        individual signers or filming conditions. Coordinate-level noise is
        more semantically appropriate than feature-dropout for landmark data
        because landmark coordinates are inherently continuous and noisy in
        real MediaPipe output.

        Calibration (Notebook 03):
            std=0.01 is approximately 6% of mean xy signal (~0.16) and
            approximately 150% of mean z signal (~0.0066). Large enough to
            regularise against exact coordinate memorisation; small enough to
            preserve the wrist-relative hand shape that distinguishes signs.

        Zero-fill invariant (enforced when detected_only=True):
            1. Compute either-hand-detected mask (T,) BEFORE any modification.
            2. Generate noise of shape (T, D).
            3. Zero out noise rows where detected mask is False.
            4. arr += noise

            This precisely preserves the exact zero-fill pattern for undetected
            frames. If noise were applied to zero-filled frames, the LSTM would
            receive a near-zero but non-zero signal and incorrectly interpret it
            as a detected-but-barely-visible hand, corrupting the one-handed vs
            two-handed distinction.

        Algorithm detail — single noise matrix (not per-component):
            We generate one (T, D) noise matrix rather than separate matrices
            for LH, RH, and pose. This is equivalent in expectation (all draws
            are i.i.d. Gaussian) and requires only one RNG call, minimising
            the per-clip augmentation overhead.

        Parameters
        ----------
        arr : np.ndarray
            Input landmark array, shape (T, D). Not mutated.
        rng : numpy.random.Generator
            Per-clip RNG from AugmentationPipeline.
        std : float
            Noise standard deviation. Default 0.01 (Notebook 03 calibration).
        detected_only : bool
            If True (always True in production), noise is applied only to
            frames where at least one hand is detected. Must be True to
            preserve the semantic zero-fill invariant.

        Returns
        -------
        np.ndarray
            Shape (T, D), dtype float32. Identical to input on zero-fill frames.
        """
        _validate_landmark_array(arr, "gaussian_noise")

        if std <= 0.0:
            return arr

        T, D = arr.shape

        # Compute detection mask BEFORE modifying any values
        if detected_only:
            detected_mask = self._get_either_hand_detected_mask(arr)  # (T,) bool
        else:
            detected_mask = np.ones(T, dtype=bool)

        # Generate full noise matrix; zero out undetected rows
        noise = rng.standard_normal((T, D)).astype(np.float32) * float(std)

        # Zero out noise for undetected frames (broadcasting: mask is (T,))
        noise[~detected_mask, :] = 0.0

        result = arr + noise
        return result.astype(np.float32)

    # ------------------------------------------------------------------
    # Spatial flip
    # ------------------------------------------------------------------

    def spatial_flip(
        self,
        arr: np.ndarray,
        rng: Generator,
        min_hand_presence: float = FLIP_MIN_HAND_PRESENCE_DEFAULT,
    ) -> np.ndarray:
        """
        Mirror all x-coordinates and swap LH/RH slices to simulate a
        left-handed signer or a mirrored camera angle.

        This is the most semantically rich augmentation in the pipeline:
        it creates an entirely new signer perspective. For signs that are
        not handedness-specific (the majority of ASL signs), the mirrored
        version is a valid alternative execution of the same sign by a
        left-handed or right-handed signer.

        CLIP-LEVEL SAFETY CHECK (applied first)
        ----------------------------------------
        both_present_fraction = fraction of frames where both hands detected
        If both_present_fraction < min_hand_presence: return arr unchanged

        This is the most important guard in the augmentation system. One-handed
        signs naturally have both_present_fraction near 0.0. Flipping them
        produces a meaningless mirrored sign where the active hand changes side.
        The sign semantics may require a specific hand (e.g., signs where the
        dominant hand acts on the non-dominant hand are not symmetric).

        The 30% threshold (FLIP_MIN_HAND_PRESENCE_DEFAULT) was validated in
        Notebook 03: all 35 selected signs pass this threshold at the
        sign level, but enforcement at the clip level is the correct policy
        because individual clips can be one-handed-dominant even for two-handed
        signs.

        TRANSFORM SEQUENCE (only if safety check passes)
        --------------------------------------------------
        The four steps must be executed in exactly this order:

        Step 1 — Negate all x-coordinates in LH slice:
            The x-coordinates within LEFT_HAND_SLICE are at positions [0::3]
            relative to the start of the slice (indices 0, 3, 6, ..., 60
            within LEFT_HAND_SLICE, corresponding to absolute positions
            0, 3, 6, ..., 60 in the full 225-vector).
            arr_new[:, LEFT_HAND_SLICE][:, 0::3] *= -1

        Step 2 — Negate all x-coordinates in RH slice:
            arr_new[:, RIGHT_HAND_SLICE][:, 0::3] *= -1

        Step 3 — Negate all x-coordinates in POSE slice:
            MediaPipe pose landmarks use a consistent x-axis. Mirroring
            a signer without mirroring their pose produces an internally
            inconsistent body (left shoulder on the right, hands on the
            wrong side). POSE MUST be negated.
            arr_new[:, POSE_SLICE][:, 0::3] *= -1

        Step 4 — Swap LH and RH slices:
            After mirroring, what was the right hand is physically on the left.
            The model expects left-hand data in [0:63]. Without the swap,
            the left-hand slice contains the mirrored right hand's data —
            anatomically impossible (a left hand in the right side of the body).
            lh_copy = arr_new[:, LEFT_HAND_SLICE].copy()
            arr_new[:, LEFT_HAND_SLICE] = arr_new[:, RIGHT_HAND_SLICE]
            arr_new[:, RIGHT_HAND_SLICE] = lh_copy

        Why steps 1-3 before step 4:
            We negate x-coords while the data is still in its original slots
            (LH data in LH slice, RH data in RH slice). Swapping first and then
            negating would negate the already-swapped data — the result would
            be identical in this case (both approaches are mathematically
            equivalent), but negating-then-swapping is more readable: first we
            mirror the coordinates, then we relabel which hand is which.

        Mathematical verification (double-flip = identity):
            Applying this transform twice must return the original array.
            Negating x-coords twice: -(-x) = x ✓
            Swapping LH/RH twice: swap(swap(A, B)) = (A, B) ✓
            The double-flip identity is tested in test_augmentation.py.

        Parameters
        ----------
        arr : np.ndarray
            Input landmark array, shape (T, D). Not mutated.
        rng : numpy.random.Generator
            Per-clip RNG from AugmentationPipeline. Not used in this transform
            (the flip is deterministic given clip_idx). Accepted for API
            consistency with all other SpatialAugmenter methods.
        min_hand_presence : float
            Minimum fraction of frames where both hands must be simultaneously
            present for the clip to be considered flip-safe. Default 0.30.

        Returns
        -------
        np.ndarray
            Shape (T, D), dtype float32. Returns input unchanged if not flip-safe.
        """
        _validate_landmark_array(arr, "spatial_flip")

        # Clip-level safety check
        both_present_frac = self._get_both_hands_present_fraction(arr)
        if both_present_frac < min_hand_presence:
            return arr  # Not flip-safe; return unchanged

        # Make a copy — we will modify it in place
        arr_new = arr.copy()

        # -------------------------------------------------------------------
        # Steps 1–3: Negate x-coordinates in LH, RH, and POSE slices
        #
        # x-coordinates within any landmark slice are at positions 0::3
        # (landmark 0 x, landmark 1 x, ..., landmark N x).
        # We use slice notation on the second axis to select only x-coords.
        # -------------------------------------------------------------------

        # Step 1: Negate LH x-coords
        arr_new[:, LEFT_HAND_SLICE.start  : LEFT_HAND_SLICE.stop  : N_COORDS_PER_LANDMARK] *= -1

        # Step 2: Negate RH x-coords
        arr_new[:, RIGHT_HAND_SLICE.start : RIGHT_HAND_SLICE.stop : N_COORDS_PER_LANDMARK] *= -1

        # Step 3: Negate POSE x-coords
        arr_new[:, POSE_SLICE.start       : POSE_SLICE.stop       : N_COORDS_PER_LANDMARK] *= -1

        # -------------------------------------------------------------------
        # Step 4: Swap LH and RH slices
        #
        # After mirroring, the data that was in the RH slot (physical right hand)
        # is now on the left side of the body — it belongs in the LH slot.
        # We use a temporary copy to avoid clobbering data during the swap.
        # -------------------------------------------------------------------
        lh_copy = arr_new[:, LEFT_HAND_SLICE].copy()                           # preserve LH (now mirrored)
        arr_new[:, LEFT_HAND_SLICE]  = arr_new[:, RIGHT_HAND_SLICE]            # put mirrored RH into LH slot
        arr_new[:, RIGHT_HAND_SLICE] = lh_copy                                 # put mirrored LH into RH slot

        return arr_new.astype(np.float32)

    # ------------------------------------------------------------------
    # 2D rotation
    # ------------------------------------------------------------------

    def rotation_2d(
        self,
        arr: np.ndarray,
        rng: Generator,
        max_deg: float = AUGMENTATION_ROTATION_DEG_DEFAULT,
    ) -> np.ndarray:
        """
        Rotate wrist-relative hand landmarks around the wrist origin (0, 0).

        Simulates camera tilt: a camera that is slightly rotated clockwise or
        counter-clockwise produces the same physical gesture at a rotated angle
        in the image plane. After wrist-relative normalisation, the wrist is at
        (0, 0, z), so a 2D rotation around the origin is the correct operation.

        Applied to LH and RH independently, detected frames only.
        Pose is NOT rotated because pose represents body position in camera
        space — rotating pose coordinates would change body orientation semantics
        (a tilted camera doesn't change where the body is relative to itself).

        Calibration:
            ±5° rotation of the distal fingertip (landmark 12, ~0.15 units from
            wrist) produces a displacement of sin(5°) × 0.15 ≈ 0.013 units,
            comparable to gaussian_noise std=0.01. The augmentations therefore
            have consistent signal magnitudes.

        Algorithm:
            1. Draw theta from Uniform(-max_deg, +max_deg), convert to radians
            2. Build 2D rotation matrix:
                 R = [[cos(θ), -sin(θ)],
                      [sin(θ),  cos(θ)]]
               (standard counter-clockwise rotation in the xy-plane)
            3. For each detected frame t:
               a. Reshape LH slice to (N_HAND_LANDMARKS, 3) — (x, y, z) triples
               b. Apply R to the first two columns: lms[:, :2] = lms[:, :2] @ R.T
               c. Flatten back to (N_HAND_FEATURES,) and write into arr_new
               d. Repeat for RH slice if RH is detected in frame t

            The z-coordinates are NOT rotated. Depth is not affected by in-plane
            camera tilt. The rotation is purely in the (x, y) image plane.

        Why we use R.T (transpose) instead of R directly:
            NumPy stores data in row-major order. Each row of our (N, 3) matrix
            is one landmark [x, y, z]. We want to apply R to each [x, y] column
            vector. The operation lms[:, :2] @ R.T is equivalent to
            (R @ lms[:, :2].T).T — applying R to each column vector and
            transposing back. Using R.T avoids the intermediate transpose.

        Parameters
        ----------
        arr : np.ndarray
            Input landmark array, shape (T, D). Not mutated.
        rng : numpy.random.Generator
            Per-clip RNG from AugmentationPipeline.
        max_deg : float
            Maximum rotation angle in degrees (applied as ±max_deg range).
            Default 5.0 (Notebook 03 calibration).

        Returns
        -------
        np.ndarray
            Shape (T, D), dtype float32. Pose and z-coordinates unchanged.
        """
        _validate_landmark_array(arr, "rotation_2d")

        if max_deg <= 0.0:
            return arr

        T, D = arr.shape

        # Draw rotation angle and build rotation matrix
        theta_deg = rng.uniform(-max_deg, max_deg)
        theta_rad = math.radians(theta_deg)
        cos_t = math.cos(theta_rad)
        sin_t = math.sin(theta_rad)
        # Standard 2D counter-clockwise rotation matrix
        R = np.array([[cos_t, -sin_t],
                      [sin_t,  cos_t]], dtype=np.float32)

        # Compute detection masks BEFORE any modification
        lh_detected = self._get_lh_detected_mask(arr)  # (T,) bool
        rh_detected = self._get_rh_detected_mask(arr)  # (T,) bool

        arr_new = arr.copy()

        # Indices of LH and RH detected frames
        lh_frames = np.where(lh_detected)[0]
        rh_frames = np.where(rh_detected)[0]

        if lh_frames.size > 0:
            # Vectorised rotation for all LH-detected frames simultaneously
            # Extract LH data: shape (n_lh_frames, N_HAND_LANDMARKS, 3)
            lh_data = arr_new[np.ix_(lh_frames, range(LEFT_HAND_SLICE.start, LEFT_HAND_SLICE.stop))]
            lh_data = lh_data.reshape(len(lh_frames), N_HAND_LANDMARKS, 3)

            # Apply rotation to x,y coordinates; z is unchanged
            # lh_data[:, :, :2] has shape (n_frames, 21, 2)
            # R.T has shape (2, 2)
            # (n_frames, 21, 2) @ (2, 2) = (n_frames, 21, 2)
            lh_data[:, :, :2] = lh_data[:, :, :2] @ R.T

            # Write back
            lh_start = LEFT_HAND_SLICE.start
            lh_stop  = LEFT_HAND_SLICE.stop
            arr_new[np.ix_(lh_frames, range(lh_start, lh_stop))] = (
                lh_data.reshape(len(lh_frames), N_HAND_FEATURES)
            )

        if rh_frames.size > 0:
            # Vectorised rotation for all RH-detected frames simultaneously
            rh_data = arr_new[np.ix_(rh_frames, range(RIGHT_HAND_SLICE.start, RIGHT_HAND_SLICE.stop))]
            rh_data = rh_data.reshape(len(rh_frames), N_HAND_LANDMARKS, 3)
            rh_data[:, :, :2] = rh_data[:, :, :2] @ R.T

            rh_start = RIGHT_HAND_SLICE.start
            rh_stop  = RIGHT_HAND_SLICE.stop
            arr_new[np.ix_(rh_frames, range(rh_start, rh_stop))] = (
                rh_data.reshape(len(rh_frames), N_HAND_FEATURES)
            )

        # Pose is NOT rotated (see docstring rationale)
        return arr_new.astype(np.float32)


# ---------------------------------------------------------------------------
# AugmentationPipeline
# ---------------------------------------------------------------------------

class AugmentationPipeline:
    """
    Orchestrates the full augmentation chain for a single clip.

    This is the sole public interface for augmentation in the WLASL pipeline.
    FeaturePipeline and GestureDataset call this class; they never call
    TemporalAugmenter or SpatialAugmenter directly.

    State
    -----
    The only state this class holds is:
        - base_seed (int): determines the RNG family for this pipeline instance
        - config reference: read-only, never mutated
        - flip_min_hand_presence (float): clip-level safety threshold
        - Two immutable augmenter instances (TemporalAugmenter, SpatialAugmenter)

    All per-clip state (numpy Generator, intermediate arrays) lives only on
    the call stack of __call__() and is garbage-collected after it returns.

    Per-Clip RNG
    ------------
    RNG for clip_idx is: numpy.random.default_rng(base_seed XOR clip_idx)

    The XOR design gives every clip a unique seed while keeping seeds
    deterministic across runs. For any fixed base_seed, the same clip_idx
    always produces the same RNG and therefore the same augmented output.
    Different clip indices produce different (but reproducible) augmentations
    from the same input array.

    Design note: XOR is preferred over addition (base_seed + clip_idx) because
    XOR preserves bit diversity across the full 64-bit seed space, whereas
    addition can produce seeds that are very close together for small clip_idx
    values. np.random.default_rng() uses a SFC64 generator, which is excellent
    for divergent initialisation from XOR seeds.

    Copy Policy
    -----------
    AugmentationPipeline.__call__() receives an array that is ALREADY a copy.
    The copy is made by FeaturePipeline.__call__() before calling this method.
    Individual transform methods (TemporalAugmenter, SpatialAugmenter) receive
    and return arrays without making additional copies, except:
    - spatial_flip() makes one copy internally (required for safe swap)
    - rotation_2d() makes one copy internally (arr_new = arr.copy())

    This design minimises allocations: the expected overhead for a 60×225
    float32 array is ~108 KB per clip × 2–3 copies = ~216–324 KB per clip,
    all of which are short-lived stack allocations.

    Transform Chain Order
    ---------------------
    1. temporal_jitter  — changes which frames exist
    2. speed_jitter     — changes frame timing
    3. gaussian_noise   — adds coordinate noise to detected frames
    4. rotation_2d      — rotates hand coordinates around wrist
    5. spatial_flip     — mirrors and swaps hands (operates on final spatial state)

    See module docstring for the full rationale for this ordering.

    Parameters
    ----------
    config : AugmentationConfig
        Frozen Pydantic config from load_config(). Reads: enabled,
        temporal_jitter, frame_drop_prob, speed_jitter, gaussian_noise_std,
        gaussian_noise_detected_only, rotation_deg, spatial_flip.
    seed : int
        Base seed. XORed with clip_idx to produce per-clip RNG.
        Default 42 (project global seed from base.yaml).
    flip_min_hand_presence : float
        Passed through to SpatialAugmenter.spatial_flip(). Default 0.30
        (Notebook 03 validated threshold for clip-level flip safety).

    Examples
    --------
    Standard training usage (via FeaturePipeline — preferred):
        # Don't instantiate AugmentationPipeline directly in training code.
        # FeaturePipeline holds the instance and calls it with training=True.

    Direct usage (tests, notebooks):
        from src.utils.config import load_config
        from src.features.augmentation import AugmentationPipeline

        cfg = load_config(model="lstm", data="seq60", augmentation="spatial_temporal")
        pipeline = AugmentationPipeline(cfg.augmentation, seed=42)
        arr_augmented = pipeline(arr.copy(), clip_idx=7)
    """

    def __init__(
        self,
        config: Any,
        seed: int = 42,
        flip_min_hand_presence: float = FLIP_MIN_HAND_PRESENCE_DEFAULT,
    ) -> None:
        self._config          = config
        self._seed            = int(seed)
        self._flip_threshold  = float(flip_min_hand_presence)
        self._temporal        = TemporalAugmenter()
        self._spatial         = SpatialAugmenter()

        logger.debug(
            f"AugmentationPipeline initialised | "
            f"enabled={config.enabled} | "
            f"temporal_jitter={config.temporal_jitter} | "
            f"frame_drop_prob={config.frame_drop_prob} | "
            f"speed_jitter={config.speed_jitter} | "
            f"gaussian_noise_std={config.gaussian_noise_std} | "
            f"gaussian_noise_detected_only={config.gaussian_noise_detected_only} | "
            f"rotation_deg={config.rotation_deg} | "
            f"spatial_flip={config.spatial_flip} | "
            f"flip_min_hand_presence={flip_min_hand_presence} | "
            f"base_seed={seed}",
            extra={"stage": "augmentation"},
        )

    def __call__(
        self,
        arr: np.ndarray,
        clip_idx: int = 0,
    ) -> np.ndarray:
        """
        Apply the configured augmentation chain to a single clip.

        IMPORTANT: ``arr`` must already be a copy. This method does not copy
        the input — it modifies intermediate results in place where possible
        (each transform returns a new array, so the original ``arr`` reference
        is never modified, but the copies from previous transforms may be
        reused as inputs to the next transform). The copy responsibility is
        explicitly assigned to FeaturePipeline.__call__(), which makes one copy
        before calling this method.

        If augmentation is disabled (config.enabled=False), returns arr
        unchanged without any transform overhead.

        Parameters
        ----------
        arr : np.ndarray
            Landmark array of shape (T, D), dtype float32.
            Must be a copy — this method does not copy.
        clip_idx : int
            Per-clip index used to seed the RNG. Same clip_idx always
            produces the same augmented output for a given base_seed.
            Has no effect when config.enabled=False.

        Returns
        -------
        np.ndarray
            Shape (T, D), dtype float32. Augmented array.
        """
        if not self._config.enabled:
            return arr

        # Validate input shape — catch errors early before any transforms run
        _validate_landmark_array(arr, "AugmentationPipeline.__call__")

        # Per-clip RNG: base_seed XOR clip_idx ensures unique, deterministic seed
        # The XOR is applied on the full Python int (arbitrary precision) and
        # then passed to default_rng, which accepts any non-negative integer.
        seed = self._seed ^ int(clip_idx)
        rng  = np.random.default_rng(seed)

        # -------------------------------------------------------------------
        # Transform chain — applied in strict order (see module docstring)
        # -------------------------------------------------------------------

        # 1. Temporal jitter: drop random frames, re-pad to original length
        if self._config.temporal_jitter and self._config.frame_drop_prob > 0.0:
            arr = self._temporal.temporal_jitter(
                arr,
                rng,
                drop_prob=float(self._config.frame_drop_prob),
            )

        # 2. Speed jitter: resample at random rate, re-pad/crop to original length
        if self._config.speed_jitter:
            arr = self._temporal.speed_jitter(arr, rng)

        # 3. Gaussian noise: add coordinate noise to detected frames only
        if self._config.gaussian_noise_std > 0.0:
            arr = self._spatial.gaussian_noise(
                arr,
                rng,
                std=float(self._config.gaussian_noise_std),
                detected_only=bool(self._config.gaussian_noise_detected_only),
            )

        # 4. 2D rotation: rotate hand landmarks around wrist origin
        if self._config.rotation_deg > 0.0:
            arr = self._spatial.rotation_2d(
                arr,
                rng,
                max_deg=float(self._config.rotation_deg),
            )

        # 5. Spatial flip: mirror x-coords and swap LH/RH slices
        #    (applied last — operates on the final spatial state)
        if self._config.spatial_flip:
            arr = self._spatial.spatial_flip(
                arr,
                rng,
                min_hand_presence=self._flip_threshold,
            )

        return arr.astype(np.float32)

    def get_metadata(self) -> dict[str, Any]:
        """
        Return a complete, JSON-serialisable description of the augmentation config.

        This metadata is:
        - Stored in ``gesture_model_metadata.json`` alongside the TFLite model
        - Logged to MLflow as a run parameter via ``mlflow.log_dict()``
        - Included in the FeaturePipeline's ``get_pipeline_metadata()`` output
        - Used by GesturePredictor to instantiate an identical pipeline at inference

        The ``chain_order`` field documents the transform application order for
        any consumer of this metadata.

        Returns
        -------
        dict[str, Any]
            Flat dictionary with all augmentation parameters and chain order.
        """
        return {
            "enabled":                      self._config.enabled,
            "temporal_jitter":              self._config.temporal_jitter,
            "frame_drop_prob":              self._config.frame_drop_prob,
            "speed_jitter":                 self._config.speed_jitter,
            "gaussian_noise_std":           self._config.gaussian_noise_std,
            "gaussian_noise_detected_only": self._config.gaussian_noise_detected_only,
            "rotation_deg":                 self._config.rotation_deg,
            "spatial_flip":                 self._config.spatial_flip,
            "flip_min_hand_presence":       self._flip_threshold,
            "base_seed":                    self._seed,
            "rng_algorithm":                "SFC64 (numpy.random.default_rng)",
            "rng_seed_derivation":          "base_seed XOR clip_idx",
            "chain_order": [
                "temporal_jitter",
                "speed_jitter",
                "gaussian_noise",
                "rotation_2d",
                "spatial_flip",
            ],
            "chain_order_rationale": (
                "temporal before spatial: prevents inconsistent spatial treatment "
                "of dropped frames; noise before flip: preserves detection-based "
                "noise masking; flip last: operates on final spatial state."
            ),
        }

    def __repr__(self) -> str:
        cfg = self._config
        active = []
        if cfg.enabled:
            if cfg.temporal_jitter:
                active.append(f"jitter(p={cfg.frame_drop_prob})")
            if cfg.speed_jitter:
                active.append("speed")
            if cfg.gaussian_noise_std > 0:
                active.append(f"noise(std={cfg.gaussian_noise_std})")
            if cfg.rotation_deg > 0:
                active.append(f"rot(±{cfg.rotation_deg}°)")
            if cfg.spatial_flip:
                active.append(f"flip(min={self._flip_threshold})")
        transforms = "+".join(active) if active else "none"
        status = "ENABLED" if cfg.enabled else "DISABLED"
        return (
            f"AugmentationPipeline("
            f"status={status}, "
            f"transforms=[{transforms}], "
            f"seed={self._seed})"
        )