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

3. **Zero-fill invariant** — no spatial transform ever modifies a frame-component
   slot that is zero in the original. This preserves the semantic signal of
   zero-fill (one-handed signs, detection failures). The left-hand 70.18%
   missing rate discovered in Notebook 03 is semantic — not noise — and must
   not be corrupted.

   CRITICAL CLARIFICATION: the invariant is per-component (LH slot, RH slot,
   pose slot), not per-row. A frame is "detected" for a given slot only if
   that slot's values are non-zero in the original array. For one-handed signs,
   the absent hand's slot must stay exactly zero even though the frame as a
   whole is "partially detected".

4. **AugmentationPipeline owns the copy boundary** — AugmentationPipeline.__call__()
   makes one defensive copy of the input before the chain runs. Individual transform
   methods receive arrays they may safely operate on without additional copies (except
   where internal copies are required for correctness: spatial_flip's slot swap,
   rotation_2d's write-back). This eliminates silent mutation bugs from external
   callers passing shared arrays.

5. **Shape invariant** — every transform returns an array of identical shape (T, D)
   to its input. Temporal transforms that drop frames restore shape via zero-in-place
   (not compress+pad — see temporal_jitter docstring for the critical distinction).
   Spatial transforms never change T or D.

6. **Detected-only noise — per component slot** — Gaussian noise is applied
   exclusively to component slots (LH, RH, pose) that are non-zero in the original
   frame. Applying noise to a zero-filled slot would corrupt its identity as
   "no detection" — the LSTM would receive a near-zero but non-zero signal and
   incorrectly treat it as a detected-but-barely-visible hand. This invariant holds
   at the slot level, not just the frame level: a one-handed frame where LH is absent
   must have its LH slot remain exactly zero even though the frame itself is partially
   detected.

7. **dtype contract** — all transforms accept any numpy floating-point dtype and
   return float32. The .astype(np.float32, copy=False) idiom is used at all return
   boundaries: it is a zero-cost no-op when the array is already float32, avoiding
   redundant allocations in the hot training loop.

Critical Implementation Note: spatial_flip
------------------------------------------
spatial_flip() is the highest-risk transform in this file.

It must do BOTH of the following simultaneously:

    (a) Negate every x-coordinate in the LH slice, RH slice, AND pose slice
        x-coords are at indices 0, 3, 6, ... within each slice (i.e. [::3])

    (b) Swap the LH and RH data according to per-frame detection state

If (a) is done without (b): landmarks are mirrored but hand labels are wrong.
    The model receives RH landmark data labelled as LH and vice versa.
    This produces anatomically impossible geometry.

If (b) is done without (a): LH and RH are swapped but x-coords are not mirrored.
    The model receives the correct label but wrong spatial geometry.
    This also produces anatomically impossible geometry.

The swap in (b) uses a HYBRID PER-FRAME POLICY (not a global slot swap):
    - Frame with BOTH hands detected: negate all x-coords; swap LH and RH slots
    - Frame with ONLY LEFT hand detected: negate LH x + POSE x; move LH → RH slot,
      zero the LH slot
    - Frame with ONLY RIGHT hand detected: negate RH x + POSE x; move RH → LH slot,
      zero the RH slot
    - Frame with NO hands detected: unchanged (zero-fill invariant)

temporal_jitter: Zero-In-Place vs. Compress-Then-Pad
------------------------------------------------------
The stated goal of temporal_jitter is to simulate MediaPipe detection failures,
which produce zero-filled frames at the ORIGINAL temporal position. The correct
implementation zeros dropped frames IN PLACE:

    Original:  [A, B, C, D, E]
    Drop mask: [T, F, T, F, T]  (F = drop)
    Old result: [A, C, E, 0, 0]  — wrong: compressed sign + trailing silence
    New result: [A, 0, C, 0, E]  — correct: sign at right timing with dropout

speed_jitter: Interpolation back to T frames
--------------------------------------------
For faster clips (rate > 1.0), n_resampled < T frames are extracted. These are
then interpolated back to T frames using np.interp (per-feature linear
interpolation). This ensures the output always has T non-zero frames — there
are NO trailing zero pads. Trailing zeros would silently drop the sign's tail
content, which is worse than interpolation. The speed_jitter transform is a
resampling transform, not a crop-and-pad transform.

Transform Chain Order
---------------------
1. temporal_jitter  — zero-fill dropped frames in place (preserves timing)
2. speed_jitter     — resample speed (changes frame timing, may duplicate frames)
3. gaussian_noise   — add noise to detected component slots only (per-slot masking)
4. rotation_2d      — rotate hand coords around wrist (post-normalisation only)
5. spatial_flip     — per-frame-aware mirror and hand-slot reassignment
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
# Module-level constants
# ---------------------------------------------------------------------------

#: Maximum safe frame drop probability.
_MAX_SAFE_DROP_PROB: float = 0.99

#: Absolute minimum drop probability below which jitter is a no-op.
_MIN_EFFECTIVE_DROP_PROB: float = 1e-6

#: Absolute minimum speed jitter rate. Values ≤ 0 produce division-by-zero.
_MIN_SPEED_RATE: float = 0.05


# ---------------------------------------------------------------------------
# Internal validation helpers
# ---------------------------------------------------------------------------

def _validate_landmark_array(arr: np.ndarray, caller: str) -> None:
    """
    Validate that ``arr`` is a 2D numpy array of shape (T, FEATURE_SIZE)
    with a floating-point dtype.

    Contract
    --------
    Accepts any numpy floating-point dtype (float16, float32, float64).

    Raises
    ------
    TypeError
        If ``arr`` is not a numpy ndarray, or if its dtype is not floating-point.
    ValueError
        If ``arr`` is not 2-dimensional, or if ``arr.shape[1] != FEATURE_SIZE``.
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError(
            f"{caller}: expected np.ndarray, got {type(arr).__name__}."
        )
    if not np.issubdtype(arr.dtype, np.floating):
        raise TypeError(
            f"{caller}: expected a floating-point dtype (float16/32/64), "
            f"got dtype={arr.dtype}. "
            "Cast the array to float32 before passing to augmentation transforms."
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
            "Always pass the full 225-dim array to augmentation transforms. "
            "Landmark-config slicing (hands_only / pose_only) happens in "
            "FeaturePipeline._select_landmark_config(), AFTER augmentation."
        )


def _clamp_drop_prob(drop_prob: float, caller: str) -> float:
    """Clamp ``drop_prob`` to [0.0, _MAX_SAFE_DROP_PROB] with a warning on violation."""
    if drop_prob < 0.0:
        logger.warning(
            f"{caller}: drop_prob={drop_prob} is negative. Clamping to 0.0.",
            extra={"stage": "augmentation"},
        )
        return 0.0
    if drop_prob >= 1.0:
        logger.warning(
            f"{caller}: drop_prob={drop_prob} >= 1.0 would drop all frames on every call. "
            f"Clamping to {_MAX_SAFE_DROP_PROB}.",
            extra={"stage": "augmentation"},
        )
        return _MAX_SAFE_DROP_PROB
    return float(drop_prob)


def _validate_speed_range(
    speed_range: tuple[float, float], caller: str
) -> tuple[float, float]:
    """
    Validate and return a safe speed range (lo, hi) for speed_jitter.

    Raises
    ------
    ValueError
        If either bound is ≤ 0, or if lo > hi.
    """
    lo, hi = float(speed_range[0]), float(speed_range[1])
    if lo <= 0.0 or hi <= 0.0:
        raise ValueError(
            f"{caller}: speed_range bounds must be > 0, "
            f"got ({lo}, {hi}). "
            "A rate of 0 would produce a clip with infinite duration."
        )
    if lo > hi:
        raise ValueError(
            f"{caller}: speed_range lo ({lo}) must be ≤ hi ({hi}). "
            "Swap the values: speed_range=(min_rate, max_rate)."
        )
    return lo, hi


def _clamp_hand_presence(min_hand_presence: float, caller: str) -> float:
    """Clamp ``min_hand_presence`` to [0.0, 1.0] with a warning on violation."""
    if min_hand_presence < 0.0:
        logger.warning(
            f"{caller}: min_hand_presence={min_hand_presence} < 0. "
            "Clamping to 0.0 (flip always applied, safety check disabled).",
            extra={"stage": "augmentation"},
        )
        return 0.0
    if min_hand_presence > 1.0:
        logger.warning(
            f"{caller}: min_hand_presence={min_hand_presence} > 1. "
            "Clamping to 1.0 (flip never applied — no clip has 100% both-hands presence).",
            extra={"stage": "augmentation"},
        )
        return 1.0
    return float(min_hand_presence)


# ---------------------------------------------------------------------------
# TemporalAugmenter
# ---------------------------------------------------------------------------

class TemporalAugmenter:
    """
    Frame-level temporal transforms operating on (T, D) arrays.

    All methods are pure functions with no state. The same inputs always
    produce the same outputs. All methods:

    - Accept a (T, D) array with any floating-point dtype and a numpy Generator
    - Return a (T, D) float32 array of IDENTICAL shape to the input
    - Never mutate the input array
    - Never reduce T below 1 frame (edge-case handling)
    - Validate inputs and parameter ranges before any computation
    """

    def temporal_jitter(
        self,
        arr: np.ndarray,
        rng: Generator,
        drop_prob: float = AUGMENTATION_FRAME_DROP_PROB_DEFAULT,
    ) -> np.ndarray:
        """
        Zero-fill randomly selected frames in place to simulate MediaPipe dropout.

        Implementation — zero-in-place (NOT compress-then-pad)
        --------------------------------------------------------
        Dropped frames are zeroed AT THEIR ORIGINAL TEMPORAL POSITION:

            Original:   [A, B, C, D, E]
            Drop mask:  [T, F, T, F, T]  (F = zeroed out)
            Result:     [A, 0, C, 0, E]   ← CORRECT
            Old result: [A, C, E, 0, 0]   ← WRONG

        Edge case: if all frames selected for dropout, force-retain the frame
        with the highest landmark signal.

        Parameters
        ----------
        arr : np.ndarray
            Input landmark array, shape (T, D), any floating-point dtype.
        rng : numpy.random.Generator
        drop_prob : float
            Probability of zeroing each frame in [0.0, 0.99].

        Returns
        -------
        np.ndarray
            Shape (T, D), dtype float32.
        """
        _validate_landmark_array(arr, "temporal_jitter")

        drop_prob = _clamp_drop_prob(drop_prob, "temporal_jitter")

        if drop_prob < _MIN_EFFECTIVE_DROP_PROB:
            return arr.astype(np.float32, copy=False)

        T, D = arr.shape

        # Build keep-mask: True = keep original frame, False = zero out frame
        keep_mask = rng.random(T) > drop_prob  # (T,) bool

        # Edge case: if all frames would be zeroed, force-retain one
        if not keep_mask.any():
            signal_per_frame = (arr != 0.0).sum(axis=1)  # (T,) int
            keep_idx = int(np.argmax(signal_per_frame))
            keep_mask[keep_idx] = True

        # Zero-in-place: copy the array, then zero the dropped frames
        result = arr.astype(np.float32, copy=True)
        result[~keep_mask, :] = 0.0

        return result

    def speed_jitter(
        self,
        arr: np.ndarray,
        rng: Generator,
        speed_range: tuple[float, float] = AUGMENTATION_SPEED_RANGE,
    ) -> np.ndarray:
        """
        Resample the clip at a random speed, return array of same length.

        For slower clips (rate < 1.0): n_resampled > T frames are extracted,
        then centre-cropped back to T.

        For faster clips (rate > 1.0): n_resampled < T frames are extracted,
        then interpolated back to T frames using zero-aware linear interpolation.

        Zero-fill invariant for interpolation
        --------------------------------------
        Standard np.interp blends values across frame boundaries. At the boundary
        between a detected frame and a zero-fill frame, this produces non-zero
        values in what were originally zero-fill slots — violating the zero-fill
        invariant for one-handed signs.

        Fix: after interpolation, for each output frame, identify the floor and
        ceil source indices. If BOTH surrounding source frames have zero in a
        given component slot, the output frame is forced to zero for that slot.

        At a detected↔zero-fill boundary, the floor and ceil differ in zero
        status, so interpolation proceeds normally (a blended transition value
        is acceptable there). Only deep-interior zero-fill frames — where both
        surrounding source frames are zero — are forced back to zero.

        Parameters
        ----------
        arr : np.ndarray
            Input landmark array, shape (T, D), any floating-point dtype.
        rng : numpy.random.Generator
        speed_range : tuple[float, float]
            (lo, hi) rate range. Rate > 1.0 = faster sign, rate < 1.0 = slower.

        Returns
        -------
        np.ndarray
            Shape (T, D), dtype float32.
        """
        _validate_landmark_array(arr, "speed_jitter")

        lo, hi = _validate_speed_range(speed_range, "speed_jitter")
        T, D   = arr.shape

        rate        = rng.uniform(lo, hi)
        n_resampled = max(1, round(T / rate))

        float_indices = np.linspace(0, T - 1, n_resampled)
        int_indices   = np.clip(
            np.round(float_indices).astype(np.int64), 0, T - 1
        )

        arr_resampled = arr[int_indices]  # shape (n_resampled, D)

        if n_resampled == T:
            return arr_resampled.astype(np.float32, copy=True)

        if n_resampled > T:
            # Slower clip — centre-crop the excess
            excess = n_resampled - T
            start  = excess // 2
            return arr_resampled[start : start + T].astype(np.float32, copy=True)

        # ----------------------------------------------------------------
        # Faster clip — zero-aware interpolation back to T frames
        # ----------------------------------------------------------------
        target_idx = np.linspace(0, n_resampled - 1, T)   # (T,) float
        source_idx = np.arange(n_resampled, dtype=np.float64)

        # Step 1: standard per-feature linear interpolation
        result = np.empty((T, D), dtype=np.float32)
        for d in range(D):
            result[:, d] = np.interp(
                target_idx,
                source_idx,
                arr_resampled[:, d].astype(np.float64),
            )

        # Step 2: zero-fill invariant enforcement
        #
        # For each output frame, find the floor and ceil source indices.
        # If both surrounding source frames are zero for a component slot,
        # the output frame must also be zero for that slot.
        #
        # At a detected↔zero boundary (floor detected, ceil zero or vice
        # versa), the interpolated blend is kept — it represents a valid
        # transition. Only frames where BOTH neighbours are zero are forced
        # back to zero.
        floor_idx = np.clip(
            np.floor(target_idx).astype(np.int64), 0, n_resampled - 1
        )
        ceil_idx = np.clip(
            np.ceil(target_idx).astype(np.int64), 0, n_resampled - 1
        )

        # Per-component-slot zero masks on the RESAMPLED array
        # shape (n_resampled,) bool — True where the slot is all-zero
        lh_zero_rs   = ~arr_resampled[:, LEFT_HAND_SLICE].any(axis=1)
        rh_zero_rs   = ~arr_resampled[:, RIGHT_HAND_SLICE].any(axis=1)
        pose_zero_rs = ~arr_resampled[:, POSE_SLICE].any(axis=1)

        # shape (T,) bool — True where BOTH surrounding source frames are zero
        lh_both_zero   = lh_zero_rs[floor_idx]   & lh_zero_rs[ceil_idx]
        rh_both_zero   = rh_zero_rs[floor_idx]   & rh_zero_rs[ceil_idx]
        pose_both_zero = pose_zero_rs[floor_idx] & pose_zero_rs[ceil_idx]

        if lh_both_zero.any():
            result[lh_both_zero,
                LEFT_HAND_SLICE.start:LEFT_HAND_SLICE.stop] = 0.0

        if rh_both_zero.any():
            result[rh_both_zero,
                RIGHT_HAND_SLICE.start:RIGHT_HAND_SLICE.stop] = 0.0

        if pose_both_zero.any():
            result[pose_both_zero,
                POSE_SLICE.start:POSE_SLICE.stop] = 0.0

        return result


# ---------------------------------------------------------------------------
# SpatialAugmenter
# ---------------------------------------------------------------------------

class SpatialAugmenter:
    """
    Landmark-coordinate spatial transforms operating on (T, D) arrays.

    All methods:
    - Accept a (T, D) array with any floating-point dtype and a numpy Generator
    - Return a (T, D) float32 array of IDENTICAL shape
    - Never mutate the input array
    - Respect the zero-fill invariant at the component-slot level:
      a slot (LH, RH, pose) that is zero in the original is never modified
    - Validate inputs and parameter ranges before computation
    """

    # ------------------------------------------------------------------
    # Detection mask helpers
    # ------------------------------------------------------------------

    def _get_lh_detected_mask(self, arr: np.ndarray) -> np.ndarray:
        """
        Boolean mask shape (T,) — True where left hand is detected.

        A frame has left hand detected if ANY value in LEFT_HAND_SLICE is
        non-zero. After wrist-relative normalisation, the wrist (landmark 0)
        becomes (0,0,0) but all other landmarks remain non-zero — so .any()
        correctly identifies detection.
        """
        return arr[:, LEFT_HAND_SLICE].any(axis=1)   # (T,) bool

    def _get_rh_detected_mask(self, arr: np.ndarray) -> np.ndarray:
        """Boolean mask shape (T,) — True where right hand is detected."""
        return arr[:, RIGHT_HAND_SLICE].any(axis=1)  # (T,) bool

    def _get_pose_detected_mask(self, arr: np.ndarray) -> np.ndarray:
        """Boolean mask shape (T,) — True where pose is detected."""
        return arr[:, POSE_SLICE].any(axis=1)        # (T,) bool

    def _get_either_hand_detected_mask(self, arr: np.ndarray) -> np.ndarray:
        """
        Boolean mask shape (T,) — True where AT LEAST ONE hand is detected.

        NOTE: This is used only for transforms that need a frame-level "any
        detection" flag (e.g. rotation_2d). For gaussian_noise, the per-slot
        masks are used instead to avoid corrupting absent-hand slots within
        partially-detected frames.
        """
        lh = self._get_lh_detected_mask(arr)
        rh = self._get_rh_detected_mask(arr)
        return lh | rh   # (T,) bool

    def _get_both_hands_present_fraction(self, arr: np.ndarray) -> float:
        """
        Fraction of frames where BOTH hands are simultaneously detected.
        Used as the clip-level safety check for spatial_flip.
        """
        lh   = self._get_lh_detected_mask(arr)
        rh   = self._get_rh_detected_mask(arr)
        both = lh & rh
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

        Zero-fill invariant — per component slot (CRITICAL)
        -----------------------------------------------------
        The invariant operates at the component-slot level, not the frame level.

        For each frame t, noise is applied independently to each slot:
            - LH slot [0:63]:   only if left hand is detected in frame t
            - RH slot [63:126]: only if right hand is detected in frame t
            - Pose slot [126:225]: only if pose is detected in frame t

        This means a one-handed frame (e.g., RH detected, LH absent) receives
        noise only in its RH and pose slots. The LH slot remains EXACTLY zero.

        Previous implementation bug: using a per-ROW mask based on
        "either_hand_detected" applied noise to the ENTIRE 225-element row for
        any frame where at least one hand was detected — including the zero-filled
        absent-hand slot. This corrupted one-handed sign semantics.

        The correct implementation generates a (T, D) noise matrix and then
        zeroes out columns that correspond to absent component slots:

            noise_mask shape: (T, D) bool
            noise_mask[:, LH_cols] = lh_detected[:, None]
            noise_mask[:, RH_cols] = rh_detected[:, None]
            noise_mask[:, POSE_cols] = pose_detected[:, None]
            result = arr + noise * noise_mask

        Parameters
        ----------
        arr : np.ndarray
            Input landmark array, shape (T, D), any floating-point dtype.
        rng : numpy.random.Generator
        std : float
            Noise standard deviation. Must be ≥ 0. Default 0.01.
        detected_only : bool
            If True (always True in production), noise is applied only to
            detected component slots. Set False only for research ablations.

        Returns
        -------
        np.ndarray
            Shape (T, D), dtype float32. Zero-fill slots are bit-identical
            to the input when detected_only=True.
        """
        _validate_landmark_array(arr, "gaussian_noise")

        if std <= 0.0:
            return arr.astype(np.float32, copy=False)

        T, D = arr.shape

        # Generate full noise matrix
        noise = rng.standard_normal((T, D)).astype(np.float32) * float(std)

        if detected_only:
            # Build per-component-slot detection masks from the ORIGINAL array.
            # Each mask is (T,) bool; broadcast to (T, slot_width) to zero out
            # noise in absent-component columns.
            lh_detected   = self._get_lh_detected_mask(arr)    # (T,) bool
            rh_detected   = self._get_rh_detected_mask(arr)    # (T,) bool
            pose_detected = self._get_pose_detected_mask(arr)  # (T,) bool

            # Zero noise for LH columns in frames where LH is absent
            # ~lh_detected selects rows where LH slot must be kept at zero
            noise[~lh_detected, LEFT_HAND_SLICE]  = 0.0
            noise[~rh_detected, RIGHT_HAND_SLICE] = 0.0
            noise[~pose_detected, POSE_SLICE]     = 0.0

        result = arr.astype(np.float32, copy=True)
        result += noise

        return result

    # ------------------------------------------------------------------
    # Spatial flip — hybrid per-frame policy
    # ------------------------------------------------------------------

    def spatial_flip(
        self,
        arr: np.ndarray,
        _rng: Generator,
        min_hand_presence: float = FLIP_MIN_HAND_PRESENCE_DEFAULT,
    ) -> np.ndarray:
        """
        Mirror all x-coordinates and reassign hand slots per frame.

        CLIP-LEVEL SAFETY CHECK (applied first)
        ----------------------------------------
        both_present_fraction = fraction of frames where BOTH hands detected
        If both_present_fraction < min_hand_presence: return arr unchanged.

        HYBRID PER-FRAME POLICY
        -----------------------
        Case 1 — Both hands detected:
            (a) Negate all x-coords in LH, RH, POSE slices
            (b) Swap LH and RH data for this frame

        Case 2 — Only LEFT hand detected:
            (a) Negate LH x-coords
            (b) Negate POSE x-coords
            (c) Move LH data → RH slot; zero the LH slot

        Case 3 — Only RIGHT hand detected:
            (a) Negate RH x-coords
            (b) Negate POSE x-coords
            (c) Move RH data → LH slot; zero the RH slot

        Case 4 — Neither hand detected: unchanged (zero-fill invariant)

        Mathematical invariant: flip(flip(x)) == x (exact involution).

        Parameters
        ----------
        arr : np.ndarray
            Input landmark array, shape (T, D), any floating-point dtype.
        _rng : numpy.random.Generator
            Accepted for API consistency. Intentionally unused.
        min_hand_presence : float
            Minimum both-hands fraction for flip-safety. Default 0.30.

        Returns
        -------
        np.ndarray
            Shape (T, D), dtype float32.
        """
        _validate_landmark_array(arr, "spatial_flip")

        min_hand_presence = _clamp_hand_presence(min_hand_presence, "spatial_flip")

        # Clip-level safety check
        both_present_frac = self._get_both_hands_present_fraction(arr)
        if both_present_frac < min_hand_presence:
            return arr.astype(np.float32, copy=False)

        T, D = arr.shape

        # Compute per-frame detection masks from the ORIGINAL array
        lh_mask = self._get_lh_detected_mask(arr)   # (T,) bool
        rh_mask = self._get_rh_detected_mask(arr)   # (T,) bool

        # Start with a float32 copy
        result = arr.astype(np.float32, copy=True)

        # Frame index sets for each detection case
        both_frames    = np.where(lh_mask & rh_mask)[0]    # Case 1
        lh_only_frames = np.where(lh_mask & ~rh_mask)[0]   # Case 2
        rh_only_frames = np.where(~lh_mask & rh_mask)[0]   # Case 3
        # Case 4 (neither): no action needed

        # Pre-compute stride-3 x-coord column indices
        lh_x_cols   = np.arange(LEFT_HAND_SLICE.start,  LEFT_HAND_SLICE.stop,  N_COORDS_PER_LANDMARK)
        rh_x_cols   = np.arange(RIGHT_HAND_SLICE.start, RIGHT_HAND_SLICE.stop, N_COORDS_PER_LANDMARK)
        pose_x_cols = np.arange(POSE_SLICE.start,        POSE_SLICE.stop,       N_COORDS_PER_LANDMARK)

        # Full slot column ranges (for data swapping)
        lh_cols = np.arange(LEFT_HAND_SLICE.start,  LEFT_HAND_SLICE.stop)
        rh_cols = np.arange(RIGHT_HAND_SLICE.start, RIGHT_HAND_SLICE.stop)

        # ---------------------------------------------------------------
        # Case 1: Both hands — negate all x-coords; swap slots
        # ---------------------------------------------------------------
        if both_frames.size > 0:
            result[np.ix_(both_frames, lh_x_cols)]   *= -1
            result[np.ix_(both_frames, rh_x_cols)]   *= -1
            result[np.ix_(both_frames, pose_x_cols)] *= -1

            # Swap LH and RH slots
            lh_data_copy = result[np.ix_(both_frames, lh_cols)].copy()
            result[np.ix_(both_frames, lh_cols)] = result[np.ix_(both_frames, rh_cols)]
            result[np.ix_(both_frames, rh_cols)] = lh_data_copy

        # ---------------------------------------------------------------
        # Case 2: Only LH — negate LH x + POSE x; move LH → RH slot
        # ---------------------------------------------------------------
        if lh_only_frames.size > 0:
            result[np.ix_(lh_only_frames, lh_x_cols)]   *= -1
            result[np.ix_(lh_only_frames, pose_x_cols)] *= -1

            # Move mirrored LH data into RH slot, zero LH slot
            result[np.ix_(lh_only_frames, rh_cols)] = result[np.ix_(lh_only_frames, lh_cols)]
            result[np.ix_(lh_only_frames, lh_cols)] = 0.0

        # ---------------------------------------------------------------
        # Case 3: Only RH — negate RH x + POSE x; move RH → LH slot
        # ---------------------------------------------------------------
        if rh_only_frames.size > 0:
            result[np.ix_(rh_only_frames, rh_x_cols)]   *= -1
            result[np.ix_(rh_only_frames, pose_x_cols)] *= -1

            # Move mirrored RH data into LH slot, zero RH slot
            result[np.ix_(rh_only_frames, lh_cols)] = result[np.ix_(rh_only_frames, rh_cols)]
            result[np.ix_(rh_only_frames, rh_cols)] = 0.0

        return result

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

        Applied to: LH (detected frames) and RH (detected frames) independently.
        NOT applied to: POSE (rotating pose would change body orientation semantics).
        NOT applied to: z-coordinates (depth unaffected by in-plane tilt).

        IMPORTANT: Only geometrically correct on wrist-relative normalised arrays.
        FeaturePipeline guarantees normalisation before augmentation.

        Algorithm: vectorised over all LH-detected / RH-detected frames at once
        using 3D batch matrix multiply.

        Parameters
        ----------
        arr : np.ndarray
            Shape (T, D), any floating-point dtype.
        rng : numpy.random.Generator
        max_deg : float
            Maximum rotation angle in degrees, applied as Uniform(-max_deg, +max_deg).

        Returns
        -------
        np.ndarray
            Shape (T, D), dtype float32.
        """
        _validate_landmark_array(arr, "rotation_2d")

        if max_deg <= 0.0:
            return arr.astype(np.float32, copy=False)

        T, D = arr.shape

        theta_rad = math.radians(rng.uniform(-max_deg, max_deg))
        cos_t = math.cos(theta_rad)
        sin_t = math.sin(theta_rad)
        R = np.array([[cos_t, -sin_t],
                      [sin_t,  cos_t]], dtype=np.float32)

        # Detection masks from ORIGINAL array
        lh_frames = np.where(self._get_lh_detected_mask(arr))[0]
        rh_frames = np.where(self._get_rh_detected_mask(arr))[0]

        result = arr.astype(np.float32, copy=True)

        # Vectorised rotation for all LH-detected frames simultaneously
        if lh_frames.size > 0:
            lh_col_idx = list(range(LEFT_HAND_SLICE.start, LEFT_HAND_SLICE.stop))
            lh_data = result[np.ix_(lh_frames, lh_col_idx)].reshape(
                len(lh_frames), N_HAND_LANDMARKS, 3
            )
            lh_data[:, :, :2] = lh_data[:, :, :2] @ R.T
            result[np.ix_(lh_frames, lh_col_idx)] = lh_data.reshape(
                len(lh_frames), N_HAND_FEATURES
            )

        # Vectorised rotation for all RH-detected frames simultaneously
        if rh_frames.size > 0:
            rh_col_idx = list(range(RIGHT_HAND_SLICE.start, RIGHT_HAND_SLICE.stop))
            rh_data = result[np.ix_(rh_frames, rh_col_idx)].reshape(
                len(rh_frames), N_HAND_LANDMARKS, 3
            )
            rh_data[:, :, :2] = rh_data[:, :, :2] @ R.T
            result[np.ix_(rh_frames, rh_col_idx)] = rh_data.reshape(
                len(rh_frames), N_HAND_FEATURES
            )

        return result


# ---------------------------------------------------------------------------
# AugmentationPipeline
# ---------------------------------------------------------------------------

class AugmentationPipeline:
    """
    Orchestrates the full augmentation chain for a single clip.

    This is the sole public interface for augmentation in the WLASL pipeline.
    FeaturePipeline and GestureDataset call this class; they never call
    TemporalAugmenter or SpatialAugmenter directly.

    Copy Boundary
    -------------
    AugmentationPipeline.__call__() makes one defensive copy of the input
    before the chain begins. Callers do NOT need to pre-copy. The original
    array is guaranteed unmodified.

    Per-Clip RNG
    ------------
    RNG for clip_idx: numpy.random.default_rng(base_seed XOR clip_idx)

    Transform Chain Order
    ---------------------
    1. temporal_jitter  — zero-fill dropped frames in place
    2. speed_jitter     — resample at random rate
    3. gaussian_noise   — per-slot noise on detected components only
    4. rotation_2d      — rotate hand landmarks around wrist
    5. spatial_flip     — per-frame-aware mirror + hand-slot reassignment

    Parameters
    ----------
    config : AugmentationConfig
        Frozen Pydantic config from load_config().
    seed : int
        Base seed XORed with clip_idx for per-clip RNG. Default 42.
    flip_min_hand_presence : float
        Passed to spatial_flip. Default 0.30.
    """
    

    def __init__(
        self,
        config: Any,
        seed: int = 42,
        flip_min_hand_presence: float = FLIP_MIN_HAND_PRESENCE_DEFAULT,
    ) -> None:
        self._config         = config
        self._seed           = int(seed)
        self._flip_threshold = _clamp_hand_presence(
            flip_min_hand_presence, "AugmentationPipeline.__init__"
        )
        self._temporal = TemporalAugmenter()
        self._spatial  = SpatialAugmenter()

        logger.debug(
            "AugmentationPipeline initialised | "
            f"enabled={config.enabled} | "
            f"temporal_jitter={config.temporal_jitter} | "
            f"frame_drop_prob={config.frame_drop_prob} | "
            f"speed_jitter={config.speed_jitter} | "
            f"gaussian_noise_std={config.gaussian_noise_std} | "
            f"gaussian_noise_detected_only={config.gaussian_noise_detected_only} | "
            f"rotation_deg={config.rotation_deg} | "
            f"spatial_flip={config.spatial_flip} | "
            f"flip_min_hand_presence={self._flip_threshold} | "
            f"base_seed={self._seed}",
            extra={"stage": "augmentation"},
        )

        if config.rotation_deg > 0.0:
            logger.debug(
                "rotation_2d is enabled. This transform is only geometrically "
                "correct on wrist-relative normalised arrays (wrist at origin). "
                "FeaturePipeline guarantees this.",
                extra={"stage": "augmentation"},
            )

    def __call__(
        self,
        arr: np.ndarray,
        clip_idx: int = 0,
    ) -> np.ndarray:
        """
        Apply the configured augmentation chain to a single clip.

        Makes one defensive copy of the input. Original array is never mutated.
        If augmentation is disabled, returns a float32 copy without transform overhead.

        Parameters
        ----------
        arr : np.ndarray
            Landmark array of shape (T, FEATURE_SIZE), any floating-point dtype.
        clip_idx : int
            Per-clip index for RNG seeding via (base_seed XOR clip_idx).

        Returns
        -------
        np.ndarray
            Shape (T, FEATURE_SIZE), dtype float32. Always a new array object.
        """
        if not self._config.enabled:
            return arr.astype(np.float32, copy=True)

        _validate_landmark_array(arr, "AugmentationPipeline.__call__")

        # Defensive copy — single allocation point for the chain
        arr = arr.astype(np.float32, copy=True)

        # Per-clip RNG: XOR base seed with clip_idx
        rng = np.random.default_rng(self._seed ^ int(clip_idx))

        # -------------------------------------------------------------------
        # Transform chain
        # -------------------------------------------------------------------

        # 1. Zero-fill randomly selected frames in place
        if self._config.temporal_jitter and self._config.frame_drop_prob > 0.0:
            arr = self._temporal.temporal_jitter(
                arr,
                rng,
                drop_prob=float(self._config.frame_drop_prob),
            )

        # 2. Resample at random speed
        if self._config.speed_jitter:
            arr = self._temporal.speed_jitter(arr, rng)

        # 3. Add per-slot noise to detected components only
        if self._config.gaussian_noise_std > 0.0:
            arr = self._spatial.gaussian_noise(
                arr,
                rng,
                std=float(self._config.gaussian_noise_std),
                detected_only=bool(self._config.gaussian_noise_detected_only),
            )

        # 4. Rotate hand landmarks around wrist origin
        if self._config.rotation_deg > 0.0:
            arr = self._spatial.rotation_2d(
                arr,
                rng,
                max_deg=float(self._config.rotation_deg),
            )

        # 5. Per-frame-aware mirror + hand-slot reassignment
        if self._config.spatial_flip:
            arr = self._spatial.spatial_flip(
                arr,
                rng,
                min_hand_presence=self._flip_threshold,
            )

        return arr

    def get_metadata(self) -> dict[str, Any]:
        """
        Return a complete, JSON-serialisable description of the augmentation config.

        Stored in gesture_model_metadata.json, logged to MLflow, included in
        FeaturePipeline.get_pipeline_metadata(), used by GesturePredictor.
        """
        return {
            "enabled":                      self._config.enabled,
            "temporal_jitter":              self._config.temporal_jitter,
            "frame_drop_prob":              self._config.frame_drop_prob,
            "temporal_jitter_strategy":     "zero_in_place",
            "speed_jitter":                 self._config.speed_jitter,
            "speed_jitter_index_mode":      "integer_nearest_neighbour",
            "speed_jitter_fast_clip_strategy": "interpolate_to_T_frames",
            "gaussian_noise_std":           self._config.gaussian_noise_std,
            "gaussian_noise_detected_only": self._config.gaussian_noise_detected_only,
            "gaussian_noise_mask_granularity": "per_component_slot",
            "rotation_deg":                 self._config.rotation_deg,
            "rotation_requires_normalised": True,
            "spatial_flip":                 self._config.spatial_flip,
            "spatial_flip_policy":          "hybrid_per_frame",
            "flip_min_hand_presence":       self._flip_threshold,
            "base_seed":                    self._seed,
            "rng_algorithm":                "SFC64 (numpy.random.default_rng)",
            "rng_seed_derivation":          "base_seed XOR clip_idx",
            "copy_boundary":                "AugmentationPipeline.__call__",
            "chain_order": [
                "temporal_jitter",
                "speed_jitter",
                "gaussian_noise",
                "rotation_2d",
                "spatial_flip",
            ],
            "chain_rationale": (
                "temporal before spatial: preserves consistent spatial treatment "
                "of all remaining frames; noise before flip: masking based on "
                "original detection state not relabelled slots; flip last: operates "
                "on final settled spatial state."
            ),
        }

    def __repr__(self) -> str:
        cfg    = self._config
        active = []
        if cfg.enabled:
            if cfg.temporal_jitter:
                active.append(f"jitter_inplace(p={cfg.frame_drop_prob})")
            if cfg.speed_jitter:
                active.append("speed_jitter(interp_to_T)")
            if cfg.gaussian_noise_std > 0:
                active.append(f"noise(std={cfg.gaussian_noise_std},per_slot)")
            if cfg.rotation_deg > 0:
                active.append(f"rot2d(±{cfg.rotation_deg}°, post-norm only)")
            if cfg.spatial_flip:
                active.append(f"flip_per_frame(min={self._flip_threshold})")
        transforms = " → ".join(active) if active else "none"
        status     = "ENABLED" if cfg.enabled else "DISABLED"
        return (
            f"AugmentationPipeline("
            f"status={status}, "
            f"chain=[{transforms}], "
            f"seed={self._seed})"
        )