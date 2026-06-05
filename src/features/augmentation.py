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

6. **Detected-only noise** — Gaussian noise is applied exclusively to detected
   (non-zero) landmark frames. Applying noise to a zero-filled frame would
   corrupt its identity as "no detection" — the LSTM would receive a near-zero
   but non-zero signal and incorrectly treat it as a detected-but-barely-visible
   hand. This was confirmed as mandatory by the Notebook 03 analysis.

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

The hybrid policy ensures that within a flip-safe clip, single-hand frames are
also anatomically correct after mirroring — the active hand is placed in the slot
corresponding to its mirrored physical position. The previous global-slot-swap
approach left one-handed frames within two-handed clips in anatomically ambiguous
states.

Why pose x-coords must be negated on detected frames: MediaPipe pose landmarks use
a consistent x-axis regardless of handedness. Mirroring a signer without mirroring
their pose produces a body where the left shoulder is on the right and hands are on
the wrong side — internally inconsistent geometry that would mislead the LSTM's
implicit body model.

Both (a) and (b) must be verified in test_augmentation.py.

temporal_jitter: Zero-In-Place vs. Compress-Then-Pad
------------------------------------------------------
The stated goal of temporal_jitter is to simulate MediaPipe detection failures,
which produce zero-filled frames at the ORIGINAL temporal position. The previous
compress-then-pad implementation removed frames and right-padded with zeros,
producing a temporally compressed sign followed by trailing silence — a
fundamentally different signal that the LSTM should not learn as equivalent.

The correct implementation zeros dropped frames IN PLACE:

    Original:  [A, B, C, D, E]
    Drop mask: [T, F, T, F, T]  (F = drop)
    Old result: [A, C, E, 0, 0]  — wrong: compressed sign + trailing silence
    New result: [A, 0, C, 0, E]  — correct: sign at right timing with dropout

This exactly matches the MediaPipe failure semantics the augmentation models.

Transform Chain Order
---------------------
1. temporal_jitter  — zero-fill dropped frames in place (preserves timing)
2. speed_jitter     — resample speed (changes frame timing, may duplicate frames)
3. gaussian_noise   — add noise to detected frames only
4. rotation_2d      — rotate hand coords around wrist (post-normalisation only)
5. spatial_flip     — per-frame-aware mirror and hand-slot reassignment

Rationale for this order:
- Temporal before spatial: temporal transforms change frame content. Spatial
  transforms applied first would be applied to frames that will later be zeroed
  or resampled — creating inconsistent spatial treatment across frames.
- Noise before flip: noise is applied based on original detection state. After
  flip, hand slot labels change; applying noise after flip would base masking
  on the flipped labels, not the original detection status.
- Flip last: the most semantically disruptive transform. All other transforms
  have finalised the spatial state before the final mirroring step.

Notebook 03 Findings Incorporated
----------------------------------
- Speed jitter is the highest-priority augmentation: 9/35 signs show high
  motion-energy coefficient of variation across clips (black, before, like,
  finish, give, go, thanksgiving, cousin, and others). Speed jitter directly
  addresses this variance.
- gaussian_noise std=0.01 is confirmed optimal: ~6% of mean xy signal (~0.16),
  large enough to regularise but small enough to preserve wrist-relative shape.
- Rotation ±5°: at this angle, the distal fingertip moves by sin(5°) × 0.15 ≈
  0.013 units — comparable to noise std=0.01. Augmentations have consistent
  signal magnitudes. NOTE: rotation is only geometrically correct on
  wrist-relative normalised arrays (wrist at origin). FeaturePipeline guarantees
  this by normalising BEFORE calling AugmentationPipeline.
- Frame dropout at 10%: cosine similarity confirmed 1.0000 (std=0.0001).
  Zero-in-place implementation preserves temporal alignment.
- Clip-level flip safety: all 35 signs pass the 30% both-hands threshold at the
  sign level. Clip-level enforcement + per-frame hybrid policy provides stronger
  anatomical guarantees.
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

#: Maximum safe frame drop probability. Values at or above 1.0 would produce
#: degenerate all-zero clips (keep_mask all False). Clamped with a warning.
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
    Accepts any numpy floating-point dtype (float16, float32, float64). The
    caller is responsible for ensuring the returned array is cast to float32
    via ``.astype(np.float32, copy=False)`` at return boundaries. This relaxed
    contract allows test fixtures that use float64 (the numpy default for
    ``np.random.uniform``) to pass without requiring explicit dtype casting in
    every test.

    Parameters
    ----------
    arr : np.ndarray
        Array to validate.
    caller : str
        Name of the calling method, used in the error message.

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
    """
    Clamp ``drop_prob`` to [0.0, _MAX_SAFE_DROP_PROB] with a warning on violation.

    Values at or above 1.0 would guarantee all frames are dropped, triggering
    the keep-one fallback on every clip — a degenerate training signal. Values
    below 0.0 are meaningless. Both are clamped silently to valid bounds after
    a single warning log.

    Parameters
    ----------
    drop_prob : float
        Raw drop probability from the config or caller.
    caller : str
        Calling method name for log messages.

    Returns
    -------
    float
        Clamped probability in [0.0, _MAX_SAFE_DROP_PROB].
    """
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
    """
    Clamp ``min_hand_presence`` to [0.0, 1.0] with a warning on violation.

    A value < 0.0 would always flip (including clips with zero both-hand presence),
    defeating the safety check. A value > 1.0 would never flip (no clip can have
    more than 100% frame presence).

    Parameters
    ----------
    min_hand_presence : float
    caller : str

    Returns
    -------
    float
        Clamped value in [0.0, 1.0].
    """
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
        Zero-fill randomly selected frames in place to simulate MediaPipe dropout.

        Goal: simulate the intermittent hand-detection failures that MediaPipe
        produces in real inference. When MediaPipe fails to detect a hand on a
        frame, the extractor (Stage 3) writes a zero-filled feature vector for
        that frame. This augmentation replicates that pattern during training,
        teaching the LSTM to be robust to sparse dropout in the landmark stream.

        Implementation — zero-in-place (NOT compress-then-pad)
        --------------------------------------------------------
        A critical design choice: dropped frames are zeroed AT THEIR ORIGINAL
        TEMPORAL POSITION, not removed and replaced with trailing padding.

        Example:
            Original:   [A, B, C, D, E]
            Drop mask:  [T, F, T, F, T]  (F = zeroed out)
            Result:     [A, 0, C, 0, E]   ← CORRECT: sign at right timing
            Old result: [A, C, E, 0, 0]   ← WRONG: compressed + trailing silence

        The compress-then-pad approach produces a temporally compressed sign
        concatenated with trailing zeros — a signal the LSTM should NOT learn
        as equivalent to the original. The zero-in-place approach preserves the
        temporal structure of the sign while simulating detection failure, which
        is exactly what the LSTM will encounter at inference time.

        Frame-selection note: the keep-mask is independent of detection status.
        Frames already zero-filled (detection failures from MediaPipe) can be
        "kept" or "dropped" — if dropped, they remain zero-filled either way.
        This is correct: the augmentation adds dropout on top of whatever
        detection state the extractor produced.

        Edge case — all frames selected for dropout:
            Force-retain the frame with the highest landmark signal (most non-zero
            values). This prevents degenerate all-zero clips, which provide no
            gradient signal and waste a training step. The high-signal frame is
            chosen rather than a random frame because it contains the most
            geometrically informative landmarks.

        Parameters
        ----------
        arr : np.ndarray
            Input landmark array, shape (T, D), any floating-point dtype.
            Not mutated.
        rng : numpy.random.Generator
            Per-clip RNG from AugmentationPipeline.
        drop_prob : float
            Probability of zeroing each frame in [0.0, 0.99].
            Default 0.10 (10% dropout confirmed safe in Notebook 03).
            Values outside [0.0, 1.0) are clamped with a warning.

        Returns
        -------
        np.ndarray
            Shape (T, D), dtype float32. Dropped frames contain exact zeros.
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
        Resample the clip at a random playback speed, preserving output shape.

        Simulates the same sign executed at different speeds — a pervasive source
        of within-class variance in WLASL (Notebook 03 F9: 9/35 signs show high
        motion-energy CV across clips). The same physical gesture takes ~0.7x to
        1.3x the time depending on the signer's cadence, fatigue, and style.

        Algorithm
        ---------
        1. Draw rate r from Uniform(lo, hi):
               r < 1.0 → signer is slower → clip is stretched (more indices than T)
               r > 1.0 → signer is faster → clip is compressed (fewer indices than T)
        2. Compute the resampled index sequence:
               n_resampled = max(1, round(T / r))
               float_indices = linspace(0, T-1, n_resampled)
               int_indices = clip(round(float_indices), 0, T-1)
        3. arr_resampled = arr[int_indices]
        4. If n_resampled > T (slower): centre-crop to T
        5. If n_resampled < T (faster): right-pad with zeros to T
        6. If n_resampled == T: return as-is

        Frame duplication at slow rates (documented design decision)
        ------------------------------------------------------------
        When r < 1.0 and n_resampled > T, the linspace index sequence necessarily
        contains repeated values. For example, with T=5 and rate=0.7:

            float_indices ≈ [0.0, 0.86, 1.71, 2.57, 3.43, 4.29, 5.14, ...]
            int_indices   ≈ [0,   1,    2,    3,    3,    4,    5, ...]
                                                     ↑ duplicate

        This means frame 3 appears twice in the resampled sequence. The
        augmentation is no longer a pure speed change — it is a speed change
        with occasional frame duplication. For highly dynamic signs with fast
        motion between consecutive frames, duplicated frames create short temporal
        plateaus (the signer appears to pause momentarily).

        This is an accepted consequence of the integer-indexing design decision.
        The alternative — linear interpolation of landmark coordinates between
        frames — would produce anatomically plausible but physically nonexistent
        poses that MediaPipe never generates, violating the principle that the
        LSTM should only see real extracted landmark positions. Given that the
        speed range is moderate (0.7–1.3), the duplication artefact is mild and
        the training benefit outweighs the minor signal distortion.

        Why integer indexing (not interpolation):
            Interpolated landmark positions are synthetic data MediaPipe never
            actually produces. Integer resampling selects real extracted frame
            positions at different temporal density — the LSTM has seen every
            selected frame position during normal (non-augmented) training.

        Why centre-crop for slower clips (r < 1.0):
            When n_resampled > T, we must discard surplus frames. ASL signs have
            preparatory and release movements at both ends, so the peak motion
            phase is temporally centred. Centre-cropping retains more of the
            discriminative sign content than end-cropping.

        Parameters
        ----------
        arr : np.ndarray
            Input landmark array, shape (T, D), any floating-point dtype.
            Not mutated.
        rng : numpy.random.Generator
            Per-clip RNG from AugmentationPipeline.
        speed_range : tuple[float, float]
            (min_rate, max_rate). Both must be > 0 and min_rate ≤ max_rate.
            Default (0.7, 1.3): ±30% speed variation (Notebook 03 calibration).

        Returns
        -------
        np.ndarray
            Shape (T, D), dtype float32.

        Raises
        ------
        ValueError
            If speed_range bounds are ≤ 0 or lo > hi.
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
            return arr_resampled.astype(np.float32, copy=False)

        if n_resampled > T:
            # Slower clip — centre-crop the excess
            excess = n_resampled - T
            start  = excess // 2
            result = arr_resampled[start : start + T]
            return result.astype(np.float32, copy=False)

        # Faster clip — right-pad with zeros to restore shape
        n_pad   = T - n_resampled
        padding = np.zeros((n_pad, D), dtype=np.float32)
        result  = np.concatenate(
            [arr_resampled.astype(np.float32, copy=False), padding], axis=0
        )
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
    - Respect the zero-fill invariant: frames where both hands are absent
      are never modified by any spatial transform
    - Validate inputs and parameter ranges before computation

    Detection masks are computed from the original input array before any
    modification. This ensures masks reflect the true detection state, not an
    intermediate state that might contain spurious non-zero values from a
    previous transform step.
    """

    # ------------------------------------------------------------------
    # Detection mask helpers
    # ------------------------------------------------------------------

    def _get_lh_detected_mask(self, arr: np.ndarray) -> np.ndarray:
        """
        Boolean mask shape (T,) — True where left hand is detected.

        A frame has left hand detected if ANY value in LEFT_HAND_SLICE is
        non-zero. This correctly handles the post-normalisation case: after
        wrist-relative normalisation, the wrist (landmark 0) is subtracted
        to (0, 0, 0), but all other 20 landmarks remain non-zero for a detected
        hand. The 'any' aggregation catches detection even when wrist == 0.

        For raw (pre-normalisation) arrays used in tests, all landmarks of a
        detected hand are non-zero, so 'any' also works correctly.
        """
        return arr[:, LEFT_HAND_SLICE].any(axis=1)   # (T,) bool

    def _get_rh_detected_mask(self, arr: np.ndarray) -> np.ndarray:
        """
        Boolean mask shape (T,) — True where right hand is detected.

        See _get_lh_detected_mask for detection rationale.
        """
        return arr[:, RIGHT_HAND_SLICE].any(axis=1)  # (T,) bool

    def _get_either_hand_detected_mask(self, arr: np.ndarray) -> np.ndarray:
        """
        Boolean mask shape (T,) — True where AT LEAST ONE hand is detected.

        Used by gaussian_noise and rotation_2d to identify frames eligible
        for spatial augmentation. The zero-fill invariant only requires that
        frames where BOTH hands are absent are left unchanged. Frames where
        exactly one hand is present (one-handed signs, partial occlusion) are
        semantically valid and should be augmented.
        """
        lh = self._get_lh_detected_mask(arr)
        rh = self._get_rh_detected_mask(arr)
        return lh | rh   # (T,) bool

    def _get_both_hands_present_fraction(self, arr: np.ndarray) -> float:
        """
        Fraction of frames where BOTH hands are simultaneously detected.

        This is the clip-level safety check for spatial_flip. It answers:
        "Is this clip geometrically safe to mirror?"

        One-handed signs naturally approach 0.0 (the non-dominant hand is absent
        in most or all frames). Two-handed signs typically range 0.3–0.9 depending
        on clip quality and sign complexity.

        Clips below the threshold are not flipped at all. For clips above the
        threshold, a hybrid per-frame policy handles individual single-hand frames
        (see spatial_flip docstring).

        Returns 0.0 for empty arrays (edge case protection).
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

        This is the primary regularisation augmentation. It prevents the model
        from memorising exact landmark coordinate values specific to individual
        signers or filming conditions. Coordinate-level noise is more semantically
        appropriate than feature-dropout for landmark data because landmark
        coordinates are inherently continuous and noisy in real MediaPipe output.

        Calibration (Notebook 03):
            std=0.01 is approximately 6% of mean xy signal (~0.16) and
            approximately 150% of mean z signal (~0.0066). Large enough to
            regularise against exact coordinate memorisation; small enough to
            preserve the wrist-relative hand shape that distinguishes signs.

        Zero-fill invariant (enforced when detected_only=True):
            1. Compute either-hand-detected mask (T,) from the ORIGINAL arr
               before any modification.
            2. Generate noise of shape (T, D) in one RNG call.
            3. Zero out noise rows where detected_mask is False.
            4. result = arr + masked_noise

        This precisely preserves the exact zero-fill pattern. If noise were
        applied to zero-filled frames, the LSTM would receive near-zero but
        non-zero values, corrupting the semantic one-handed vs two-handed signal.

        Single noise matrix design:
            One (T, D) matrix is generated rather than separate matrices for LH,
            RH, and pose. This is statistically equivalent (all draws are i.i.d.
            Gaussian) and requires only one RNG call — minimal training loop
            overhead.

        Parameters
        ----------
        arr : np.ndarray
            Input landmark array, shape (T, D), any floating-point dtype.
            Not mutated.
        rng : numpy.random.Generator
            Per-clip RNG from AugmentationPipeline.
        std : float
            Noise standard deviation. Must be ≥ 0.
            Default 0.01 (Notebook 03 calibration).
        detected_only : bool
            If True (always True in production), noise is applied only to
            frames where at least one hand is detected. Set False only for
            research ablations on the zero-fill invariant.

        Returns
        -------
        np.ndarray
            Shape (T, D), dtype float32. Zero-fill frames are bit-identical
            to the input when detected_only=True.
        """
        _validate_landmark_array(arr, "gaussian_noise")

        if std <= 0.0:
            return arr.astype(np.float32, copy=False)

        T, D = arr.shape

        # Compute detection mask from the ORIGINAL array before any modification
        if detected_only:
            detected_mask = self._get_either_hand_detected_mask(arr)  # (T,) bool
        else:
            detected_mask = np.ones(T, dtype=bool)

        # Generate full noise matrix
        noise = rng.standard_normal((T, D)).astype(np.float32) * float(std)

        # Zero out rows corresponding to undetected frames
        # ~detected_mask selects rows to zero; shape broadcasts correctly
        noise[~detected_mask, :] = 0.0

        result = arr.astype(np.float32, copy=False) + noise
        return result

    # ------------------------------------------------------------------
    # Spatial flip — hybrid per-frame policy
    # ------------------------------------------------------------------

    def spatial_flip(
        self,
        arr: np.ndarray,
        _rng: Generator,  # unused; accepted for API consistency with other transforms
        min_hand_presence: float = FLIP_MIN_HAND_PRESENCE_DEFAULT,
    ) -> np.ndarray:
        """
        Mirror all x-coordinates and reassign hand slots per frame to simulate
        a left-handed signer or a mirrored camera angle.

        This is the most semantically rich augmentation in the pipeline: it
        creates an entirely new signer perspective. For signs that are not
        handedness-specific (the majority of ASL signs), the mirrored version
        is a valid alternative execution of the same sign.

        CLIP-LEVEL SAFETY CHECK (applied first)
        ----------------------------------------
        both_present_fraction = fraction of frames where BOTH hands detected
        If both_present_fraction < min_hand_presence: return arr unchanged.

        This guards one-handed signs (e.g., "think", "drink"), which naturally
        have both_present_fraction near 0.0. Flipping them without this guard
        would produce a sign where the active hand changes side, corrupting its
        handedness-specific semantics.

        HYBRID PER-FRAME POLICY (for clips that pass the safety check)
        ---------------------------------------------------------------
        Rather than a global LH/RH slot swap (which leaves single-hand frames
        in anatomically ambiguous states), the flip applies a per-frame policy
        based on each frame's individual detection state:

        Case 1 — Both hands detected in frame t:
            (a) Negate all x-coords in LH, RH, and POSE slices
            (b) Swap LH and RH data for frame t
            Result: valid mirrored two-handed frame with correct slot assignment.

        Case 2 — Only LEFT hand detected in frame t:
            (a) Negate LH x-coords for frame t
            (b) Negate POSE x-coords for frame t
            (c) Move LH data → RH slot; zero the LH slot
            Result: the mirrored signer's active hand (physically now on the right)
            is correctly placed in the RH slot.

        Case 3 — Only RIGHT hand detected in frame t:
            (a) Negate RH x-coords for frame t
            (b) Negate POSE x-coords for frame t
            (c) Move RH data → LH slot; zero the RH slot
            Result: the mirrored signer's active hand (physically now on the left)
            is correctly placed in the LH slot.

        Case 4 — Neither hand detected in frame t:
            No modification (zero-fill invariant).

        Why the hybrid policy matters:
            Consider a two-handed clip (passes the 30% threshold) where the signer
            briefly lowers their non-dominant hand in some frames. With the old
            global-swap approach, those single-hand frames would have their LH and
            RH slots globally swapped — leaving the detected hand in the wrong
            anatomical slot relative to the mirrored body. The hybrid policy
            ensures the active hand's data is always in the slot that corresponds
            to its physical position on the mirrored signer's body.

        Why POSE x-coords must always be negated on detected frames:
            MediaPipe pose landmarks use a global x-axis. The pose encodes the
            signer's body position in camera space (shoulder width, wrist vs hip
            position, etc.). Mirroring a signer without mirroring their pose
            produces an internally inconsistent body — left shoulder on the right,
            hands on the wrong side. This would misalign the spatial relationship
            between hand landmarks and body landmarks that the LSTM relies on.

        Mathematical invariant (double-flip = identity):
            Applying spatial_flip twice (with min_hand_presence=0.0 to force
            both flips) must return a bit-identical array.
            Proof: negating x twice: -(-x) = x ✓
                   swapping slots twice: swap(swap(A, B)) = (A, B) ✓
                   This is verified in test_augmentation.py.

        Note on the _rng parameter:
            The flip transform is deterministic given the clip content and
            min_hand_presence threshold — it does not draw any random numbers.
            The _rng parameter is accepted solely for API consistency with all
            other SpatialAugmenter methods (which do use rng). It is intentionally
            unused and named with a leading underscore to signal this.

        Parameters
        ----------
        arr : np.ndarray
            Input landmark array, shape (T, D), any floating-point dtype.
            Not mutated.
        _rng : numpy.random.Generator
            Accepted for API consistency. Intentionally unused.
        min_hand_presence : float
            Minimum fraction of frames where both hands must be simultaneously
            detected for the clip to be flip-safe. Clamped to [0.0, 1.0].
            Default 0.30 (Notebook 03 validated threshold).

        Returns
        -------
        np.ndarray
            Shape (T, D), dtype float32. Returns input unchanged if not flip-safe.
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

        # Start with a float32 copy — we will modify it in place per case
        result = arr.astype(np.float32, copy=True)

        # Pre-compute frame index sets for each detection case
        both_frames = np.where(lh_mask & rh_mask)[0]        # Case 1
        lh_only_frames = np.where(lh_mask & ~rh_mask)[0]    # Case 2
        rh_only_frames = np.where(~lh_mask & rh_mask)[0]    # Case 3
        # Case 4 (neither): no action needed — zero-fill invariant

        # Pre-compute stride-3 column slices for x-coord negation
        # x-coordinates within any landmark slice sit at columns [start::3]
        # relative to the full 225-dim vector.
        lh_x_cols   = np.arange(LEFT_HAND_SLICE.start,   LEFT_HAND_SLICE.stop,   N_COORDS_PER_LANDMARK)
        rh_x_cols   = np.arange(RIGHT_HAND_SLICE.start,  RIGHT_HAND_SLICE.stop,  N_COORDS_PER_LANDMARK)
        pose_x_cols = np.arange(POSE_SLICE.start,         POSE_SLICE.stop,        N_COORDS_PER_LANDMARK)

        # ---------------------------------------------------------------
        # Case 1: Both hands detected — negate all x-coords; swap slots
        # ---------------------------------------------------------------
        if both_frames.size > 0:
            # Negate x-coords for LH, RH, and POSE in all Case-1 frames
            result[np.ix_(both_frames, lh_x_cols)]   *= -1
            result[np.ix_(both_frames, rh_x_cols)]   *= -1
            result[np.ix_(both_frames, pose_x_cols)] *= -1

            # Swap LH and RH slots for Case-1 frames
            lh_cols = np.arange(LEFT_HAND_SLICE.start,  LEFT_HAND_SLICE.stop)
            rh_cols = np.arange(RIGHT_HAND_SLICE.start, RIGHT_HAND_SLICE.stop)

            lh_data_copy = result[np.ix_(both_frames, lh_cols)].copy()
            result[np.ix_(both_frames, lh_cols)] = result[np.ix_(both_frames, rh_cols)]
            result[np.ix_(both_frames, rh_cols)] = lh_data_copy

        # ---------------------------------------------------------------
        # Case 2: Only LH detected — negate LH + POSE x; move LH → RH slot
        # ---------------------------------------------------------------
        if lh_only_frames.size > 0:
            # Negate LH x-coords
            result[np.ix_(lh_only_frames, lh_x_cols)]   *= -1
            # Negate POSE x-coords
            result[np.ix_(lh_only_frames, pose_x_cols)] *= -1

            # Move mirrored LH data into the RH slot
            lh_cols = np.arange(LEFT_HAND_SLICE.start,  LEFT_HAND_SLICE.stop)
            rh_cols = np.arange(RIGHT_HAND_SLICE.start, RIGHT_HAND_SLICE.stop)

            result[np.ix_(lh_only_frames, rh_cols)] = result[np.ix_(lh_only_frames, lh_cols)]
            # Zero the LH slot (the mirrored signer's "left hand" is now absent)
            result[np.ix_(lh_only_frames, lh_cols)] = 0.0

        # ---------------------------------------------------------------
        # Case 3: Only RH detected — negate RH + POSE x; move RH → LH slot
        # ---------------------------------------------------------------
        if rh_only_frames.size > 0:
            # Negate RH x-coords
            result[np.ix_(rh_only_frames, rh_x_cols)]   *= -1
            # Negate POSE x-coords
            result[np.ix_(rh_only_frames, pose_x_cols)] *= -1

            # Move mirrored RH data into the LH slot
            lh_cols = np.arange(LEFT_HAND_SLICE.start,  LEFT_HAND_SLICE.stop)
            rh_cols = np.arange(RIGHT_HAND_SLICE.start, RIGHT_HAND_SLICE.stop)

            result[np.ix_(rh_only_frames, lh_cols)] = result[np.ix_(rh_only_frames, rh_cols)]
            # Zero the RH slot (the mirrored signer's "right hand" is now absent)
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

        Simulates camera tilt: a slightly rotated camera produces the same
        physical gesture at a rotated angle in the image plane. After
        wrist-relative normalisation (applied by FeaturePipeline BEFORE this
        augmentation is called), the wrist landmark sits at (0, 0, z), making
        the origin the geometrically correct pivot point for this rotation.

        IMPORTANT: This transform is only geometrically correct on arrays that
        have been wrist-relative normalised. On raw (un-normalised) arrays,
        the rotation occurs around the global image origin (0, 0), not around
        the wrist — producing a physically meaningless spatial transform.
        FeaturePipeline guarantees normalisation before augmentation. Direct
        callers (tests, notebooks) using raw arrays should be aware of this.

        Applied to: LH (detected frames) and RH (detected frames) independently.
        NOT applied to: POSE (rotating pose coordinates would change body
        orientation semantics — a tilted camera does not change the signer's
        body position relative to itself).

        Calibration:
            ±5° rotation of the distal fingertip (landmark 12, ~0.15 units from
            wrist after normalisation) produces a displacement of
            sin(5°) × 0.15 ≈ 0.013 units — comparable to gaussian_noise
            std=0.01. Both augmentations operate at similar signal scales.

        Algorithm
        ---------
        1. Draw theta from Uniform(-max_deg, +max_deg); convert to radians.
        2. Build the 2D counter-clockwise rotation matrix:
               R = [[cos(θ), -sin(θ)],
                    [sin(θ),  cos(θ)]]
        3. Compute LH-detected and RH-detected frame index arrays from the
           ORIGINAL input (before any modification).
        4. For LH-detected frames (vectorised over all at once):
               a. Extract LH slice → reshape to (n_frames, 21, 3)
               b. Apply: lm_data[:, :, :2] = lm_data[:, :, :2] @ R.T
               c. Flatten to (n_frames, 63); write back
        5. Repeat for RH-detected frames.
        6. z-coordinates are NOT rotated (depth unaffected by in-plane tilt).

        Vectorised implementation note:
            Rather than looping over each detected frame, we gather all
            LH-detected frames into a 3D batch (n_frames, 21, 3), apply the
            matrix multiplication once, and scatter back. This reduces Python
            loop overhead by O(n_frames) and is the dominant performance
            optimisation in this transform.

        Why R.T instead of R:
            Our data is row-major: each row of the (21, 3) matrix is one
            landmark [x, y, z]. To apply R to each [x, y] row vector:
                rotated_xy = xy_data @ R.T
            is equivalent to:
                rotated_xy = (R @ xy_data.T).T
            Using R.T avoids the intermediate transpose.

        Parameters
        ----------
        arr : np.ndarray
            Input landmark array, shape (T, D), any floating-point dtype.
            Must be wrist-relative normalised for geometrically correct results.
            Not mutated.
        rng : numpy.random.Generator
            Per-clip RNG from AugmentationPipeline.
        max_deg : float
            Maximum rotation angle magnitude in degrees. Applied as
            Uniform(-max_deg, +max_deg). Default 5.0.

        Returns
        -------
        np.ndarray
            Shape (T, D), dtype float32. Pose and z-coordinates unchanged.
            Zero-fill frames are bit-identical to the input.
        """
        _validate_landmark_array(arr, "rotation_2d")

        if max_deg <= 0.0:
            return arr.astype(np.float32, copy=False)

        T, D = arr.shape

        # Draw rotation angle and build matrix from the original array
        theta_rad = math.radians(rng.uniform(-max_deg, max_deg))
        cos_t = math.cos(theta_rad)
        sin_t = math.sin(theta_rad)
        R = np.array([[cos_t, -sin_t],
                      [sin_t,  cos_t]], dtype=np.float32)

        # Compute detection masks from ORIGINAL array before any modification
        lh_frames = np.where(self._get_lh_detected_mask(arr))[0]
        rh_frames = np.where(self._get_rh_detected_mask(arr))[0]

        # Work on a float32 copy
        result = arr.astype(np.float32, copy=True)

        # Vectorised rotation for all LH-detected frames simultaneously
        if lh_frames.size > 0:
            lh_start, lh_stop = LEFT_HAND_SLICE.start, LEFT_HAND_SLICE.stop
            lh_col_idx = list(range(lh_start, lh_stop))

            # Extract: (n_lh_frames, 63) → reshape to (n_lh_frames, 21, 3)
            lh_data = result[np.ix_(lh_frames, lh_col_idx)].reshape(
                len(lh_frames), N_HAND_LANDMARKS, 3
            )
            # Apply rotation to xy plane only
            # (n_frames, 21, 2) @ (2, 2) = (n_frames, 21, 2)
            lh_data[:, :, :2] = lh_data[:, :, :2] @ R.T

            # Write back: (n_lh_frames, 21, 3) → flatten to (n_lh_frames, 63)
            result[np.ix_(lh_frames, lh_col_idx)] = lh_data.reshape(
                len(lh_frames), N_HAND_FEATURES
            )

        # Vectorised rotation for all RH-detected frames simultaneously
        if rh_frames.size > 0:
            rh_start, rh_stop = RIGHT_HAND_SLICE.start, RIGHT_HAND_SLICE.stop
            rh_col_idx = list(range(rh_start, rh_stop))

            rh_data = result[np.ix_(rh_frames, rh_col_idx)].reshape(
                len(rh_frames), N_HAND_LANDMARKS, 3
            )
            rh_data[:, :, :2] = rh_data[:, :, :2] @ R.T

            result[np.ix_(rh_frames, rh_col_idx)] = rh_data.reshape(
                len(rh_frames), N_HAND_FEATURES
            )

        # Pose is NOT rotated (see docstring rationale)
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
    AugmentationPipeline.__call__() owns the copy boundary. It makes one
    defensive copy of the input array before passing it through the transform
    chain. This eliminates an entire class of silent mutation bugs: callers
    that pass a shared array (e.g., a cached array from GestureDataset) are
    guaranteed that their array is not modified, even if a future transform
    is accidentally implemented as in-place.

    The overhead is ~54 KB for a 60×225 float32 array — negligible compared
    with the cost of the LSTM forward pass.

    Per-Clip RNG
    ------------
    RNG for clip_idx: numpy.random.default_rng(base_seed XOR clip_idx)

    XOR is used instead of addition (base_seed + clip_idx) because XOR
    preserves bit diversity across the full 64-bit seed space. For small
    clip_idx values, addition would produce seeds very close together, which
    can lead to correlated initial states in some PRNGs. numpy's SFC64
    generator handles XOR seeds well due to its diffusion properties.

    The same (base_seed, clip_idx) pair always produces the same augmented
    output, enabling reproducible debugging and the "different clip_idx →
    different output" guarantee that training requires.

    Transform Chain
    ---------------
    1. temporal_jitter  — zero-fill dropped frames in place (timing preserved)
    2. speed_jitter     — resample at random rate (may duplicate frames at low rates)
    3. gaussian_noise   — coordinate noise on detected frames only
    4. rotation_2d      — rotate hand landmarks around wrist (post-normalisation only)
    5. spatial_flip     — per-frame-aware mirror + hand-slot reassignment

    Parameters
    ----------
    config : AugmentationConfig
        Frozen Pydantic config from load_config(). Reads: enabled,
        temporal_jitter, frame_drop_prob, speed_jitter, gaussian_noise_std,
        gaussian_noise_detected_only, rotation_deg, spatial_flip.
    seed : int
        Base seed XORed with clip_idx to produce per-clip RNG.
        Default 42 (project global seed from base.yaml).
    flip_min_hand_presence : float
        Passed to SpatialAugmenter.spatial_flip(). Default 0.30
        (Notebook 03 validated threshold). Clamped to [0.0, 1.0].

    Examples
    --------
    Via FeaturePipeline (preferred — normalisation is guaranteed):
        # FeaturePipeline holds the AugmentationPipeline instance.
        # Do not instantiate AugmentationPipeline directly in training code.

    Direct usage in tests and notebooks:
        from src.utils.config import load_config
        from src.features.augmentation import AugmentationPipeline
        import numpy as np

        cfg = load_config(model="lstm", data="seq60", augmentation="spatial_temporal")
        aug = AugmentationPipeline(cfg.augmentation, seed=42)

        arr = np.random.uniform(0.1, 0.9, (60, 225)).astype(np.float32)
        # NOTE: On raw arrays, rotation_2d rotates around global origin, not wrist.
        # For geometrically correct rotation, normalise first (as FeaturePipeline does).
        arr_aug = aug(arr, clip_idx=7)   # arr is NOT mutated (defensive copy inside)
        arr_aug2 = aug(arr, clip_idx=7)  # identical to arr_aug
        arr_aug3 = aug(arr, clip_idx=8)  # different (different clip_idx → different RNG)
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
                "FeaturePipeline guarantees this. Direct callers using raw "
                "arrays should normalise first.",
                extra={"stage": "augmentation"},
            )

    def __call__(
        self,
        arr: np.ndarray,
        clip_idx: int = 0,
    ) -> np.ndarray:
        """
        Apply the configured augmentation chain to a single clip.

        This method owns the copy boundary: it makes one defensive copy of
        the input before the chain begins. Callers do NOT need to pre-copy
        the array. The original ``arr`` is guaranteed to be unmodified on
        return.

        If augmentation is disabled (config.enabled=False), returns a float32
        view/copy of arr without any transform overhead. The return is always
        a new array (never the same object as the input).

        Parameters
        ----------
        arr : np.ndarray
            Landmark array of shape (T, FEATURE_SIZE), any floating-point dtype.
            This method does not mutate the input.
        clip_idx : int
            Per-clip index used to seed the RNG via (base_seed XOR clip_idx).
            The same clip_idx always produces the same augmented output.
            Has no effect when config.enabled=False.

        Returns
        -------
        np.ndarray
            Shape (T, FEATURE_SIZE), dtype float32. Always a new array object.
        """
        if not self._config.enabled:
            # Return a float32 copy — caller receives a safe, independent array
            return arr.astype(np.float32, copy=True)

        _validate_landmark_array(arr, "AugmentationPipeline.__call__")

        # Defensive copy — this is the single allocation point for the chain.
        # All transforms operate on this copy or on outputs from previous transforms.
        arr = arr.astype(np.float32, copy=True)

        # Per-clip RNG: XOR base seed with clip_idx for unique, deterministic seeding
        rng = np.random.default_rng(self._seed ^ int(clip_idx))

        # -------------------------------------------------------------------
        # Transform chain — applied in strict order (see module docstring)
        # -------------------------------------------------------------------

        # 1. Zero-fill randomly selected frames in place to simulate detection dropout
        if self._config.temporal_jitter and self._config.frame_drop_prob > 0.0:
            arr = self._temporal.temporal_jitter(
                arr,
                rng,
                drop_prob=float(self._config.frame_drop_prob),
            )

        # 2. Resample at random speed (may duplicate frames for slow rates)
        if self._config.speed_jitter:
            arr = self._temporal.speed_jitter(arr, rng)

        # 3. Add coordinate noise to detected frames only
        if self._config.gaussian_noise_std > 0.0:
            arr = self._spatial.gaussian_noise(
                arr,
                rng,
                std=float(self._config.gaussian_noise_std),
                detected_only=bool(self._config.gaussian_noise_detected_only),
            )

        # 4. Rotate hand landmarks around wrist origin (post-normalisation only)
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

        This dict is:
        - Stored in gesture_model_metadata.json alongside the TFLite model
        - Logged to MLflow via mlflow.log_dict()
        - Included in FeaturePipeline.get_pipeline_metadata()
        - Used by GesturePredictor to instantiate an identical pipeline at inference

        Returns
        -------
        dict[str, Any]
            All augmentation parameters, chain order, and design rationale notes.
        """
        return {
            "enabled":                      self._config.enabled,
            "temporal_jitter":              self._config.temporal_jitter,
            "frame_drop_prob":              self._config.frame_drop_prob,
            "temporal_jitter_strategy":     "zero_in_place",
            "speed_jitter":                 self._config.speed_jitter,
            "speed_jitter_index_mode":      "integer_nearest_neighbour",
            "gaussian_noise_std":           self._config.gaussian_noise_std,
            "gaussian_noise_detected_only": self._config.gaussian_noise_detected_only,
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
                active.append("speed_jitter(int_idx)")
            if cfg.gaussian_noise_std > 0:
                active.append(f"noise(std={cfg.gaussian_noise_std})")
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