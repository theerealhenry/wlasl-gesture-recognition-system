"""
src/features/pipeline.py
=========================
Central feature engineering pipeline for the WLASL gesture recognition system.

This is the most architecturally important file in the repository.

Data contract
-------------
    Input:  (T_raw, 225) float32  — raw landmark array loaded from .npy on disk
    Output: (seq_len, feature_dim) float32 — model-ready tensor

This contract must be IDENTICAL across every consumer in the pipeline:

    ┌─────────────────────────┬──────────────────────────────────────────────┐
    │ Consumer                │ How FeaturePipeline is used                  │
    ├─────────────────────────┼──────────────────────────────────────────────┤
    │ GestureDataset          │ training=True  (augmentation enabled)        │
    │   → val/test batches    │ training=False (deterministic)               │
    │ GesturePredictor (S7)   │ training=False on single frames/videos       │
    │ TFLite verify.py (S8)   │ training=False on full val set               │
    │ Webcam demo (S9)        │ training=False per-frame sliding window      │
    └─────────────────────────┴──────────────────────────────────────────────┘

Any deviation between training and inference preprocessing is a **silent
contamination**: the model learns slightly wrong patterns, degrading real-world
performance in ways that are invisible until deployment. The single
``FeaturePipeline`` instance shared across train/val/test is the enforcement
mechanism for this guarantee.

Transform chain (applied in this exact order)
---------------------------------------------
    1. Shape + dtype validation  — fail immediately on empty, corrupt, or
                                   non-finite input before any work is done
    2. Copy + float32 cast       — caller's array is NEVER mutated;
                                   single allocation for the entire chain
    3. Wrist-relative norm       — removes positional noise, preserves hand shape
    4. Z-coordinate soft clip    — removes physically implausible depth outliers
    5. Pad / centre-crop         — fixed sequence length on full 225-dim array
    6. Augmentation              — training mode only, NEVER at inference;
                                   MUST operate on full 225-dim array
    7. Landmark config select    — slice to hands_only / pose_only / full
                                   (applied AFTER augmentation)
    8. float32 cast + return     — guaranteed dtype; copy=False (zero-cost no-op
                                   when array is already float32)

Critical ordering constraint: Steps 5–7
----------------------------------------
Augmentation (step 6) MUST precede landmark config selection (step 7).

AugmentationPipeline hardcodes LEFT_HAND_SLICE, RIGHT_HAND_SLICE, and POSE_SLICE
constants (indices into a 225-element vector) and validates arr.shape[1] == 225.
If landmark selection ran first and produced a 126-dim (hands_only) or 99-dim
(pose_only) array, every augmentation call would raise a ValueError.

Pad/truncate (step 5) runs before augmentation (step 6) because AugmentationPipeline
is designed for fixed-shape (seq_len, 225) arrays. Temporal jitter and speed jitter
both preserve shape, and the spatial transforms operate frame-wise. Providing a
fixed-length array guarantees no shape mismatches inside the augmentation chain.

The semantic correctness of this order is preserved: we augment the full landmark
representation of the sign (all 225 dimensions at seq_len frames), then select the
configured dimensional subset. The augmented data is identical to what would be
produced if the signer had been filmed in a mirrored environment, at a different
speed, with minor hand tremor — the selection of which features to model is a
downstream architectural decision, not part of the physical data transformation.

Design decisions (all evidence-based from Notebook 03 executive report)
------------------------------------------------------------------------

    Wrist-relative normalisation
        Subtract LH/RH wrist (landmark 0) from all landmarks in that hand,
        per frame, for detected frames only. This removes the absolute screen
        position of the signer (where they stand in the frame) while preserving
        the hand SHAPE — the discriminative signal for sign identity.
        Pose is NEVER normalised. Notebook 03 F3 confirms: body position in
        camera space is signal (pose non-zero rate 96.67%, pose std is the most
        discriminative pose feature).

    Detection mask semantics after wrist-relative normalisation
        After normalisation, the LH wrist (landmark 0) becomes (0,0,0) by
        construction. The detection mask uses .any() over the FULL slice (all
        21 landmarks × 3 coords = 63 values). The other 20 landmarks remain
        non-zero for detected frames, so .any() reliably distinguishes detected
        frames from zero-fill frames. The mask is computed from the ORIGINAL
        array (before normalisation) to avoid any edge case.

    Z-coordinate soft clipping (Notebook 03 F8)
        Z carries ~4% of the xy signal magnitude but has outliers at ±0.08+.
        Clipping to ±0.10 removes physically implausible MediaPipe depth
        estimates without affecting the majority of z values. Applied to ALL
        three feature bands (LH, RH, pose) identically via ``arr[:, 2::3]``.

    Landmark configuration (Notebook 03 F3)
        hands_only (Fisher = 0.752) outperforms full (0.432) for mean feature
        separability. However, mean-feature Fisher is a summary test; the
        sequential LSTM may extract additional discriminative signal from pose
        trajectories. The ``landmark_config`` parameter drives Stage 5 Group 4
        ablation: full / hands_only / pose_only.
        CRITICAL: landmark slicing happens AFTER normalisation, z-clipping,
        padding, and augmentation. The full 225-vector passes through steps
        2–6 regardless of which config is selected.

    Centre-crop truncation strategy
        With mean clip length 67.6 frames (Notebook 03 F1) and typical seq_len
        values of 60–100, truncation is common. ASL signs have preparatory and
        release movements at both temporal ends — the peak discriminative motion
        is concentrated in the temporal centre. Centre-cropping (removing
        symmetrically from both ends) maximises retention of sign content.

    Right-pad with zeros
        Short clips are padded at the right (end) with zeros. Zero-padded
        frames are semantically identical to zero-fill detection-failure frames
        — the LSTM already learns these as non-informative. Padding at the end
        is consistent with temporal left-to-right processing.

    Truncation tracking
        Statistics are accumulated across all __call__ invocations and included
        in get_pipeline_metadata(). This directly informs the interpretation
        of the Stage 5 Group 3 sequence-length ablation.

    Vectorised wrist normalisation
        Boolean row masking + NumPy broadcasting replaces the np.where + np.ix_
        approach from the outline, reducing indexing overhead. The detection
        mask is computed on the original pre-normalisation values.

Notebook 03 findings incorporated
----------------------------------
    F1  seq_len ablation extended to {20,30,40,60,80,100}.
    F2  Left-hand 70% missing is SEMANTIC signal; zero-fill frames must pass
        through normalisation unchanged — enforced by detection mask guard.
    F3  Hands-only Fisher=0.752 motivates the landmark_config ablation.
    F4  Pose std > pose mean for discriminability — pose retained by default.
    F8  Z-clip at ±0.10 removes outliers, retains the ~4% z signal.
    F9  Speed jitter addresses 9/35 high-CV signs — handled in augmentation.
    F11 21 singleton val clips — per-class metrics unreliable; macro-F1 primary.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.features.augmentation import AugmentationPipeline
from src.features.constants import (
    FEATURE_SIZE,
    FLIP_MIN_HAND_PRESENCE_DEFAULT,
    LANDMARK_CONFIGS,
    LEFT_HAND_SLICE,
    MIN_USABLE_DETECTED_FRAMES,
    N_COORDS_PER_LANDMARK,
    N_HAND_FEATURES,
    N_HAND_LANDMARKS,
    N_POSE_FEATURES,
    N_POSE_LANDMARKS,
    POSE_SLICE,
    RIGHT_HAND_SLICE,
    TRUNCATION_WARN_FRACTION,
    Z_COORD_CLIP_DEFAULT,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Normalisation strategy name stored in pipeline metadata.
_NORMALISATION_NAME: str = "wrist_relative"

#: Truncation strategy name stored in pipeline metadata.
_TRUNCATION_STRATEGY: str = "centre"

#: Padding strategy name stored in pipeline metadata.
_PADDING_STRATEGY: str = "right_zero"

#: Number of times to log at INFO level about heavy truncation before
#: downgrading to DEBUG to avoid log spam on large datasets.
_TRUNCATION_WARN_LOG_LIMIT: int = 10

#: Minimum number of frames a clip must have after decoding to be processed.
#: Shape (0, 225) arrays pass the ndim/shape checks but contain no data.
_MIN_CLIP_FRAMES: int = 1


# ---------------------------------------------------------------------------
# FeaturePipeline
# ---------------------------------------------------------------------------


class FeaturePipeline:
    """
    Landmark-to-tensor preprocessing pipeline for WLASL gesture recognition.

    This class is the single source of truth for all feature engineering
    applied to raw landmark arrays. It is instantiated once per training run
    and shared across all dataset splits. The same instance (or a
    functionally identical one reconstructed from ``get_pipeline_metadata()``)
    is used at every stage where landmarks are consumed: training, evaluation,
    TFLite export verification, real-time inference, and the webcam demo.

    Parameters
    ----------
    config : ExperimentConfig
        Full frozen experiment config produced by ``load_config()``. The
        pipeline reads the following fields:

            config.data.sequence_length          (int)    e.g. 60
            config.data.normalise_pose           (bool)   must be False
            config.data.z_coord_clip             (float)  e.g. 0.10
            config.data.landmark_config          (str)    "full" | "hands_only" | "pose_only"
            config.data.flip_min_hand_presence   (float)  e.g. 0.30
            config.augmentation.*                         full augmentation config
            config.seed                          (int)    e.g. 42

    Raises
    ------
    ValueError
        If ``config.data.landmark_config`` is not a key in ``LANDMARK_CONFIGS``.
    ValueError
        If ``config.data.normalise_pose`` is True — contradicts Notebook 03 F3.

    Notes
    -----
    Thread safety: a single FeaturePipeline instance is NOT thread-safe because
    the truncation counters are mutable state. Create one instance per worker
    process if using multiprocessing data loaders.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, config: Any) -> None:
        self._config = config

        # --- Data config fields ---
        self._seq_len: int          = int(config.data.sequence_length)
        self._z_clip: float         = float(config.data.z_coord_clip)
        self._normalise_pose: bool  = bool(config.data.normalise_pose)
        self._lm_config: str        = str(config.data.landmark_config)
        self._flip_thresh: float    = float(config.data.flip_min_hand_presence)
        self._seed: int             = int(config.seed)

        # --- Validate landmark_config ---
        if self._lm_config not in LANDMARK_CONFIGS:
            raise ValueError(
                f"FeaturePipeline: unknown landmark_config '{self._lm_config}'. "
                f"Valid values: {sorted(LANDMARK_CONFIGS.keys())}. "
                "Check configs/data/*.yaml or your load_config() override."
            )

        # --- Validate normalise_pose: must be False per Notebook 03 F3 ---
        if self._normalise_pose:
            raise ValueError(
                "FeaturePipeline: config.data.normalise_pose=True is not "
                "supported. Pose landmarks represent body position in camera "
                "space, which is discriminative signal (Notebook 03 F3: pose "
                "non-zero rate 96.67%, Fisher analysis shows pose std > pose "
                "mean for class separability). Set normalise_pose=False in "
                "your data config YAML."
            )

        # --- Derived feature dimension from landmark config ---
        self._lm_slice: slice  = LANDMARK_CONFIGS[self._lm_config]
        self._feature_dim: int = self._lm_slice.stop - self._lm_slice.start

        # --- Augmentation pipeline ---
        # AugmentationPipeline always operates on the full 225-dim array.
        # Landmark config selection happens AFTER augmentation. See module
        # docstring for the detailed ordering rationale.
        self._aug_pipeline = AugmentationPipeline(
            config=config.augmentation,
            seed=self._seed,
            flip_min_hand_presence=self._flip_thresh,
        )

        # --- Truncation / padding tracking ---
        self._n_processed: int           = 0
        self._n_truncated: int           = 0
        self._n_padded: int              = 0
        self._total_frames_removed: int  = 0
        self._total_frames_padded: int   = 0
        self._truncation_warn_count: int = 0

        logger.info(
            "FeaturePipeline initialised | "
            f"seq_len={self._seq_len} | "
            f"landmark_config={self._lm_config} | "
            f"feature_dim={self._feature_dim} | "
            f"z_clip=±{self._z_clip} | "
            f"normalise_pose={self._normalise_pose} | "
            f"flip_min_hand_presence={self._flip_thresh} | "
            f"seed={self._seed} | "
            f"augmentation={'enabled' if config.augmentation.enabled else 'disabled'} | "
            f"transform_chain=validate→copy→norm→z_clip→pad_truncate→augment→lm_select→return",
            extra={"stage": "pipeline"},
        )

    # ------------------------------------------------------------------
    # Primary public interface
    # ------------------------------------------------------------------

    def __call__(
        self,
        arr: np.ndarray,
        training: bool = False,
        clip_idx: int = 0,
    ) -> np.ndarray:
        """
        Transform one raw landmark array into a model-ready tensor.

        This is the single method that ALL consumers of the pipeline call.
        It applies the full transform chain in the order specified in the
        module docstring.

        Parameters
        ----------
        arr : np.ndarray
            Raw landmark array of shape (T_raw, 225), dtype float32 or
            convertible to float32. Loaded directly from a .npy file.
            **This array is NEVER mutated.** A copy is made in step 2.
        training : bool, default False
            Controls whether augmentation (step 6) is applied.
            Must be True ONLY for training data.
            MUST be False for validation, test, and all inference contexts.
            Setting training=False guarantees deterministic, reproducible output
            regardless of clip_idx.
        clip_idx : int, default 0
            Passed to AugmentationPipeline to seed per-clip augmentation RNG
            via (base_seed XOR clip_idx). Has no effect when training=False.

        Returns
        -------
        np.ndarray
            Shape (seq_len, feature_dim), dtype float32.
            Always the same shape regardless of T_raw.

        Raises
        ------
        ValueError
            If arr is not a numpy ndarray.
        ValueError
            If arr.ndim != 2 — wrong array dimensionality.
        ValueError
            If arr.shape[0] == 0 — empty clip with no frames.
        ValueError
            If arr.shape[1] != FEATURE_SIZE (225) — wrong feature dimension.
            Note: always pass the full 225-dim array; landmark config slicing
            happens INSIDE the pipeline AFTER augmentation.
        ValueError
            If arr contains NaN or Inf values — corrupted extraction output.

        Examples
        --------
        Training usage (GestureDataset):
            result = pipeline(arr_raw, training=True, clip_idx=42)

        Inference usage (GesturePredictor, TFLite verify):
            result = pipeline(arr_raw, training=False)
            # clip_idx irrelevant; same output for any clip_idx value
        """
        # ------------------------------------------------------------------
        # Step 1: Comprehensive input validation
        #
        # All guards run before the copy (step 2) to fail cheaply on bad data.
        # Order: type → ndim → empty → feature_dim → finite
        # ------------------------------------------------------------------
        if not isinstance(arr, np.ndarray):
            raise ValueError(
                f"FeaturePipeline expects a numpy ndarray, "
                f"got {type(arr).__name__}. "
                "Load the .npy file with np.load() before passing to the pipeline."
            )
        if arr.ndim != 2:
            raise ValueError(
                f"FeaturePipeline expects a 2D landmark array (T_raw, {FEATURE_SIZE}), "
                f"got ndim={arr.ndim}, shape={arr.shape}. "
                "Ensure the .npy file is not corrupted and was produced by "
                "LandmarkExtractor (schema version 1.2)."
            )
        if arr.shape[0] == 0:
            raise ValueError(
                f"FeaturePipeline received an empty clip (0 frames, shape={arr.shape}). "
                "This indicates a corrupted extraction output. "
                "The extractor should produce at least 1 frame for any usable clip. "
                "Check the landmark_inventory.csv for this video_id and re-run extraction."
            )
        if arr.shape[1] != FEATURE_SIZE:
            raise ValueError(
                f"FeaturePipeline expects feature dimension {FEATURE_SIZE} "
                f"(63 LH + 63 RH + 99 pose), got {arr.shape[1]}. "
                "Always pass the full 225-dim array — landmark config slicing "
                "is applied INSIDE the pipeline after augmentation. "
                f"For landmark_config='{self._lm_config}', the output will "
                f"be sliced to {self._feature_dim} dims in the final step."
            )

        # ------------------------------------------------------------------
        # Step 2: Copy + float32 cast
        #
        # Single allocation that all subsequent steps work on.
        # The copy() call ensures the caller's array is never mutated
        # regardless of what transforms do downstream.
        # astype(..., copy=False) is a zero-cost no-op if already float32,
        # which is the common case (extractor writes float32).
        # We do copy=True here explicitly because we MUST have our own buffer.
        # ------------------------------------------------------------------
        arr = arr.astype(np.float32, copy=True)

        # ------------------------------------------------------------------
        # Finite check — AFTER copy, operating on our own buffer.
        #
        # Placed here (after copy) rather than in step 1 because np.isfinite
        # on the original array would scan it once, then astype copies it.
        # Scanning our own copy avoids two passes over the caller's data.
        # NaN/Inf will propagate through normalisation, augmentation, and into
        # TensorFlow loss computation, producing NaN gradients and silent model
        # collapse. Fail loudly here.
        # ------------------------------------------------------------------
        if not np.isfinite(arr).all():
            n_nan = int(np.isnan(arr).sum())
            n_inf = int(np.isinf(arr).sum())
            raise ValueError(
                f"FeaturePipeline received a landmark array containing non-finite values "
                f"(NaN={n_nan}, Inf={n_inf}). "
                "This indicates a corrupted .npy file or a bug in LandmarkExtractor. "
                "Re-run extraction with --force for this clip to regenerate the file."
            )

        # ------------------------------------------------------------------
        # Step 3: Wrist-relative normalisation
        # ------------------------------------------------------------------
        arr = self._wrist_relative_normalise(arr)

        # ------------------------------------------------------------------
        # Step 4: Z-coordinate soft clipping
        # ------------------------------------------------------------------
        if self._z_clip > 0.0:
            arr = self._apply_z_clip(arr)

        # ------------------------------------------------------------------
        # Step 5: Pad or centre-crop to seq_len
        #
        # Applied on the full 225-dim array BEFORE augmentation. This ensures
        # AugmentationPipeline receives fixed-shape (seq_len, 225) arrays,
        # which is required by its temporal transforms (temporal_jitter,
        # speed_jitter both preserve the shape they receive).
        # ------------------------------------------------------------------
        arr = self._pad_or_truncate(arr)

        # ------------------------------------------------------------------
        # Step 6: Augmentation — ONLY in training mode
        #
        # Applied to the fixed-length (seq_len, 225) full array.
        # AugmentationPipeline validates arr.shape[1] == FEATURE_SIZE (225).
        # Must never be applied at inference — enforced by the training flag,
        # which defaults to False.
        # ------------------------------------------------------------------
        if training and self._config.augmentation.enabled:
            arr = self._aug_pipeline(arr, clip_idx=clip_idx)

        # ------------------------------------------------------------------
        # Step 7: Landmark configuration selection
        #
        # Applied AFTER augmentation so that:
        # (a) AugmentationPipeline always receives the full 225-dim array
        # (b) hands_only and pose_only configs remain compatible with
        #     augmentation (no ValueError on shape mismatch)
        # (c) All three landmark configs receive identically augmented data
        #     — the only difference between runs is which features are modelled
        #
        # The slice produces a view (contiguous memory) — the subsequent
        # astype cast will materialise it as a new allocation if needed.
        # ------------------------------------------------------------------
        arr = self._select_landmark_config(arr)

        # ------------------------------------------------------------------
        # Step 8: Guarantee float32 output dtype
        #
        # copy=False: zero-cost no-op if arr is already float32, which is
        # the case for all non-augmented paths (augmentation also returns
        # float32). Avoids the redundant allocation present in the original
        # implementation's final `arr.astype(np.float32)` call.
        # ------------------------------------------------------------------
        self._n_processed += 1
        return arr.astype(np.float32, copy=False)

    # ------------------------------------------------------------------
    # Transform implementations
    # ------------------------------------------------------------------

    def _wrist_relative_normalise(self, arr: np.ndarray) -> np.ndarray:
        """
        Subtract each hand's wrist position from all landmarks in that hand.

        This is the primary spatial normalisation step. It removes the absolute
        screen position of the signer's hand (where they stand in the frame,
        how far from the camera, camera angle offset) while perfectly preserving
        the relative geometry of the hand — the finger configurations and hand
        shapes that define ASL signs.

        Zero-fill invariant (critical)
        -------------------------------
        Detection masks are computed on the ORIGINAL arr values BEFORE any
        modification. This is the correct approach:

        - A zero-filled frame (MediaPipe detection failure) has all-zero
          landmarks in the absent-hand slice.
        - If we naively applied wrist subtraction to every frame, we would
          subtract (0,0,0) from (0,0,0), yielding (0,0,0). This is
          mathematically equivalent, BUT it conflates two different semantic
          states after normalisation:
            * "hand detected; wrist is at the origin" → (0, 0, z) for wrist
            * "hand not detected; zero-fill" → all zeros
          Both produce (0, 0, 0) for the wrist. The distinction is preserved
          in the other 20 landmarks, but only for correctly detected frames.
        - To preserve the semantic identity of zero-fill frames exactly as
          they entered, we apply normalisation ONLY to detected frames.

        Vectorised implementation (boolean row masking)
        -----------------------------------------------
        Boolean row masking replaces the np.where + np.ix_ approach:

            detected = arr[:, SLICE].any(axis=1)     # (T,) bool
            wrists = arr[detected, wrist_start:wrist_start+3]  # (n, 3)
            arr[detected, SLICE] -= np.tile(wrists, 21)

        This is cleaner, equally fast, and avoids the intermediate np.ix_
        construction. NumPy's fancy indexing handles the boolean row selection
        + slice column selection efficiently.

        Detection mask note
        -------------------
        After this step, the wrist (landmark 0) of each detected hand becomes
        (0, 0, 0) by construction. The detection mask is computed from the
        ORIGINAL array (before any modification) to ensure the mask correctly
        identifies which frames had genuine detections vs zero-fill.

        Pose
        ----
        NOT normalised. Body position, arm angle, and torso orientation
        relative to the camera are discriminative features (Notebook 03 F3:
        pose std is more discriminative than pose mean). Normalising pose
        would remove inter-sign body positioning differences.

        Parameters
        ----------
        arr : np.ndarray
            Float32 copy of the input array, shape (T, 225).
            Modified in-place on this copy.

        Returns
        -------
        np.ndarray
            Shape (T, 225), float32. Same object as input (modified in-place).
        """
        # Compute detection masks from the ORIGINAL values before any modification.
        # .any(axis=1) is True for rows where at least one value is non-zero.
        lh_detected = arr[:, LEFT_HAND_SLICE].any(axis=1)    # (T,) bool
        rh_detected = arr[:, RIGHT_HAND_SLICE].any(axis=1)   # (T,) bool

        # --- Left hand ---
        if lh_detected.any():
            # LH wrist = landmark 0 = feature indices [0:3] in the full vector
            # shape: (n_lh, 3)
            lh_wrists = arr[lh_detected, :3]
            # Tile each (x,y,z) wrist triplet 21 times → (n_lh, 63)
            # This produces [wx, wy, wz, wx, wy, wz, ...] × 21 to subtract
            # from all 21 landmarks simultaneously.
            arr[lh_detected, LEFT_HAND_SLICE.start:LEFT_HAND_SLICE.stop] -= (
                np.tile(lh_wrists, N_HAND_LANDMARKS)
            )

        # --- Right hand ---
        if rh_detected.any():
            # RH wrist = landmark 0 = feature indices [63:66] in the full vector
            # shape: (n_rh, 3)
            rh_wrists = arr[rh_detected, RIGHT_HAND_SLICE.start:RIGHT_HAND_SLICE.start + 3]
            arr[rh_detected, RIGHT_HAND_SLICE.start:RIGHT_HAND_SLICE.stop] -= (
                np.tile(rh_wrists, N_HAND_LANDMARKS)
            )

        # Pose: NOT normalised (see docstring).

        return arr

    def _apply_z_clip(self, arr: np.ndarray) -> np.ndarray:
        """
        Soft-clip z-coordinates to the range [−z_clip, +z_clip].

        Z-coordinates in the 225-element feature vector are located at every
        third position starting from index 2: [2, 5, 8, 11, ...].
        The unified ``arr[:, 2::3]`` slice selects ALL z-coordinates across
        all three feature bands (LH, RH, pose) in one operation.

        This is applied to the full 225-dim array. For hands_only and
        pose_only landmark configs, the z-clipping of the unused band
        is discarded at step 7 (landmark config selection) — the small
        overhead is acceptable for the simplicity and correctness gained.

        Rationale (Notebook 03 F8)
        --------------------------
        The z-coordinate distribution is tightly clustered near −0.02 for
        both hands after wrist-relative normalisation, with a long tail.
        Values beyond ±0.08 represent physically implausible MediaPipe depth
        estimates. The ±0.10 threshold provides a safety margin above the
        observed outlier boundary without truncating meaningful z variation.

        Parameters
        ----------
        arr : np.ndarray
            Shape (T, 225), float32. Modified in-place.

        Returns
        -------
        np.ndarray
            Same object as input with z-coordinates clipped in-place.
        """
        arr[:, 2::3] = np.clip(arr[:, 2::3], -self._z_clip, self._z_clip)
        return arr

    def _pad_or_truncate(self, arr: np.ndarray) -> np.ndarray:
        """
        Bring the sequence to exactly ``self._seq_len`` frames.

        Applied on the FULL 225-dim array before augmentation. This is
        required because AugmentationPipeline expects fixed-shape inputs.

        Two cases:

        TRUNCATION (T_raw > seq_len) — centre-crop
        -------------------------------------------
        Algorithm:
            remove = T_raw - seq_len
            start  = remove // 2
            end    = T_raw - (remove - start)
            result = arr[start:end].copy()   ← materialise the slice

        Why .copy(): ``arr`` may be a view from a previous operation.
        ``_select_landmark_config`` returns a view. To ensure independent
        ownership of the result buffer, we always copy after slicing.

        Why centre-crop (not head-crop or tail-crop):
            ASL signs have preparatory movements at the start and release
            movements at the end. Peak discriminative motion is concentrated
            in the temporal centre of the clip. Removing symmetrically from
            both ends maximises retention of sign-informative content.

        Example (T_raw=120, seq_len=60):
            remove=60, start=30, end=90 → keeps frames [30:90].

        Truncation statistics:
            n_truncated, total_frames_removed accumulate across __call__s.
            Heavy truncation (>TRUNCATION_WARN_FRACTION of clip removed)
            is logged at WARNING level, rate-limited to avoid spam.

        PADDING (T_raw < seq_len) — right-pad with zeros
        -------------------------------------------------
        Algorithm:
            pad_count = seq_len - T_raw
            padding   = np.zeros((pad_count, D), dtype=float32)
            result    = concatenate([arr, padding], axis=0)

        Why right-pad:
            Zero-padded frames are semantically identical to zero-fill
            detection-failure frames. The LSTM treats them as non-informative
            regardless of cause. Padding at the right (temporal end) is
            consistent with left-to-right LSTM processing.

        NO-OP (T_raw == seq_len):
            Returns arr unchanged (O(1), no allocation).

        Parameters
        ----------
        arr : np.ndarray
            Shape (T_raw, 225), float32.

        Returns
        -------
        np.ndarray
            Shape (seq_len, 225), float32. New allocation except the no-op path.

        Raises
        ------
        RuntimeError
            If output shape is not (seq_len, 225) — indicates a bug in
            the crop/pad arithmetic.
        """
        T_raw, D = arr.shape

        if T_raw == self._seq_len:
            return arr   # No-op: exact match, no allocation

        if T_raw > self._seq_len:
            # Centre-crop
            remove = T_raw - self._seq_len
            start  = remove // 2
            end    = T_raw - (remove - start)
            result = arr[start:end].copy()   # .copy() for ownership

            self._n_truncated          += 1
            self._total_frames_removed += remove

            if remove / T_raw > TRUNCATION_WARN_FRACTION:
                self._truncation_warn_count += 1
                if self._truncation_warn_count <= _TRUNCATION_WARN_LOG_LIMIT:
                    logger.warning(
                        f"Heavy truncation: {remove}/{T_raw} frames removed "
                        f"({remove / T_raw:.0%}) to reach seq_len={self._seq_len}. "
                        f"Consider seq_len ≥ {T_raw} for this clip. "
                        f"(Centre-crop: kept frames [{start}:{end}])",
                        extra={"stage": "pipeline"},
                    )
                elif self._truncation_warn_count == _TRUNCATION_WARN_LOG_LIMIT + 1:
                    logger.warning(
                        f"Heavy-truncation warning suppressed after "
                        f"{_TRUNCATION_WARN_LOG_LIMIT} occurrences. "
                        "Further heavy-truncation events logged at DEBUG level.",
                        extra={"stage": "pipeline"},
                    )
                else:
                    logger.debug(
                        f"Heavy truncation (suppressed): {remove}/{T_raw} frames "
                        f"removed ({remove / T_raw:.0%}).",
                        extra={"stage": "pipeline"},
                    )

        else:
            # Right-pad with zeros
            pad_count = self._seq_len - T_raw
            padding   = np.zeros((pad_count, D), dtype=np.float32)
            result    = np.concatenate([arr, padding], axis=0)

            self._n_padded             += 1
            self._total_frames_padded  += pad_count

        # Shape guard — defence against arithmetic bugs
        if result.shape != (self._seq_len, D):
            raise RuntimeError(
                f"FeaturePipeline._pad_or_truncate: output shape {result.shape} "
                f"!= expected ({self._seq_len}, {D}). "
                f"Inputs: T_raw={T_raw}, seq_len={self._seq_len}, D={D}. "
                "This is a bug in the pad/truncate arithmetic — please report it."
            )

        return result

    def _select_landmark_config(self, arr: np.ndarray) -> np.ndarray:
        """
        Slice the feature vector to the configured landmark subset.

        Applied AFTER augmentation (step 7). See module docstring for the
        detailed rationale for this ordering.

        Landmark configuration mappings (from constants.LANDMARK_CONFIGS):
            "full"       → arr[:, 0:225]   → (seq_len, 225)
            "hands_only" → arr[:, 0:126]   → (seq_len, 126)
            "pose_only"  → arr[:, 126:225] → (seq_len, 99)

        Returns a NumPy view (not a copy) when the slice is contiguous.
        The subsequent ``astype(copy=False)`` in step 8 materialises it
        into a proper contiguous allocation only if needed.

        Parameters
        ----------
        arr : np.ndarray
            Shape (seq_len, 225), float32.

        Returns
        -------
        np.ndarray
            Shape (seq_len, feature_dim). May be a view of ``arr``.
        """
        return arr[:, self._lm_slice]
    
    def pre_augmentation(self, arr: np.ndarray) -> np.ndarray:
        """
        Apply the deterministic prefix of the transform chain WITHOUT augmentation
        or landmark config selection.

        Used by GestureDataset to build the per-epoch augmentation cache.
        The returned (seq_len, 225) array feeds AugmentationPipeline each epoch,
        after which landmark config slicing is applied inside _build_augmented_dataset().

        This is a PUBLIC method — GestureDataset calls it directly and NEVER
        calls any private pipeline method.

        Parameters
        ----------
        arr : np.ndarray
            Raw landmark array of shape (T_raw, 225), any floating-point dtype.

        Returns
        -------
        np.ndarray
            Shape (seq_len, 225), dtype float32.
            Produced by: validate → copy+cast → wrist_norm → z_clip → pad_truncate.
            No augmentation. No landmark config slicing.
        """
        if not isinstance(arr, np.ndarray):
            raise ValueError(
                f"pre_augmentation expects a numpy ndarray, got {type(arr).__name__}."
            )
        if arr.ndim != 2:
            raise ValueError(
                f"pre_augmentation expects a 2D array (T_raw, {FEATURE_SIZE}), "
                f"got ndim={arr.ndim}, shape={arr.shape}."
            )
        if arr.shape[0] == 0:
            raise ValueError(
                f"pre_augmentation received an empty clip (0 frames). "
                "Check the .npy file for this video_id."
            )
        if arr.shape[1] != FEATURE_SIZE:
            raise ValueError(
                f"pre_augmentation expects feature dimension {FEATURE_SIZE}, "
                f"got {arr.shape[1]}. Always pass the full 225-dim array."
            )

        arr = arr.astype(np.float32, copy=True)

        if not np.isfinite(arr).all():
            n_nan = int(np.isnan(arr).sum())
            n_inf = int(np.isinf(arr).sum())
            raise ValueError(
                f"pre_augmentation received non-finite values (NaN={n_nan}, Inf={n_inf})."
            )

        arr = self._wrist_relative_normalise(arr)

        if self._z_clip > 0.0:
            arr = self._apply_z_clip(arr)

        arr = self._pad_or_truncate(arr)

        # NOTE: no augmentation, no landmark config slicing.
        # Output is always (seq_len, 225) regardless of self._lm_config.
        return arr.astype(np.float32, copy=False)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def output_shape(self) -> tuple[int, int]:
        """
        The (seq_len, feature_dim) shape that every __call__ invocation returns.

        Used by model factory code to set input layer dimensions without
        needing to run a dummy forward pass.

        Returns
        -------
        tuple[int, int]
            (sequence_length, feature_dimension)

        Examples
        --------
        >>> pipeline.output_shape
        (60, 225)   # landmark_config="full",       seq_len=60
        (60, 126)   # landmark_config="hands_only", seq_len=60
        (80, 99)    # landmark_config="pose_only",  seq_len=80
        """
        return (self._seq_len, self._feature_dim)

    @property
    def sequence_length(self) -> int:
        """The configured output sequence length (number of frames)."""
        return self._seq_len

    @property
    def feature_dim(self) -> int:
        """The configured output feature dimension after landmark config slicing."""
        return self._feature_dim

    @property
    def landmark_config(self) -> str:
        """The active landmark configuration name."""
        return self._lm_config

    @property
    def n_clips_processed(self) -> int:
        """Total number of clips processed since instantiation."""
        return self._n_processed

    # ------------------------------------------------------------------
    # Metadata and serialisation
    # ------------------------------------------------------------------

    def get_pipeline_metadata(self) -> dict[str, Any]:
        """
        Return a complete, JSON-serialisable description of this pipeline.

        This metadata serves multiple critical purposes:

        1. **TFLite model card** (Stage 8): stored alongside the .tflite file
           in ``gesture_model_metadata.json`` so any consumer knows exactly
           what preprocessing to apply.

        2. **GesturePredictor reconstruction** (Stage 7): at inference time,
           GesturePredictor reads this metadata to instantiate an identical
           FeaturePipeline without needing the original training config.
           The ``seed`` field is included for complete reproducibility.

        3. **Ablation audit trail** (Stage 5/6): logged to MLflow via
           ``mlflow.log_dict(metadata, "pipeline_metadata.json")`` alongside
           each training run, creating a complete record of preprocessing.

        4. **Truncation analysis** (Stage 6): the ``truncation_stats`` field
           enables sequence-length ablation interpretation.

        Returns
        -------
        dict[str, Any]
            Flat-ish dictionary with all pipeline parameters and accumulated
            runtime statistics. All values are JSON-serialisable.
        """
        truncation_rate = (
            self._n_truncated / self._n_processed
            if self._n_processed > 0 else 0.0
        )
        padding_rate = (
            self._n_padded / self._n_processed
            if self._n_processed > 0 else 0.0
        )
        mean_frames_removed = (
            self._total_frames_removed / self._n_truncated
            if self._n_truncated > 0 else 0.0
        )
        mean_frames_padded = (
            self._total_frames_padded / self._n_padded
            if self._n_padded > 0 else 0.0
        )

        return {
            # ── Core transform parameters ──────────────────────────────────────
            "sequence_length":          self._seq_len,
            "feature_dim":              self._feature_dim,
            "landmark_config":          self._lm_config,
            "normalisation":            _NORMALISATION_NAME,
            "normalise_pose":           self._normalise_pose,
            "z_coord_clip":             self._z_clip,
            "flip_min_hand_presence":   self._flip_thresh,
            "truncation_strategy":      _TRUNCATION_STRATEGY,
            "padding_strategy":         _PADDING_STRATEGY,

            # ── Seed — required for GesturePredictor reconstruction ───────────
            # Without the seed, a reconstructed pipeline cannot initialise an
            # identical AugmentationPipeline (though augmentation is disabled
            # at inference, the seed is stored for completeness and auditability).
            "seed":                     self._seed,

            # ── Feature vector layout (for model card / GesturePredictor) ─────
            "feature_layout": {
                "full_feature_size":        FEATURE_SIZE,
                "left_hand_slice":          [LEFT_HAND_SLICE.start,  LEFT_HAND_SLICE.stop],
                "right_hand_slice":         [RIGHT_HAND_SLICE.start, RIGHT_HAND_SLICE.stop],
                "pose_slice":               [POSE_SLICE.start,       POSE_SLICE.stop],
                "active_slice":             [self._lm_slice.start,   self._lm_slice.stop],
                "n_hand_landmarks":         N_HAND_LANDMARKS,
                "n_pose_landmarks":         N_POSE_LANDMARKS,
                "n_coords_per_landmark":    N_COORDS_PER_LANDMARK,
            },

            # ── Transform chain (for documentation and audit) ─────────────────
            # Documents the corrected ordering: augmentation on full 225-dim
            # array, landmark config select as the final transform.
            "transform_chain": [
                "shape_and_finite_validation",
                "copy_and_float32_cast",
                "wrist_relative_normalisation",
                f"z_coord_clip(±{self._z_clip})",
                f"pad_or_truncate(seq_len={self._seq_len}, "
                f"truncation={_TRUNCATION_STRATEGY}, "
                f"padding={_PADDING_STRATEGY})",
                "augmentation(training_mode_only, on_full_225_dim_array)",
                f"landmark_config_select({self._lm_config}→{self._feature_dim}dim)",
                "float32_cast(copy=False)",
            ],

            # ── Augmentation configuration (for model card) ───────────────────
            "augmentation": self._aug_pipeline.get_metadata(),

            # ── Runtime statistics ────────────────────────────────────────────
            "truncation_stats": {
                "n_clips_processed":          self._n_processed,
                "n_clips_truncated":          self._n_truncated,
                "n_clips_padded":             self._n_padded,
                "truncation_rate":            round(truncation_rate, 4),
                "padding_rate":               round(padding_rate, 4),
                "mean_frames_removed":        round(mean_frames_removed, 1),
                "mean_frames_padded":         round(mean_frames_padded, 1),
                "total_frames_removed":       self._total_frames_removed,
                "total_frames_padded":        self._total_frames_padded,
                "heavy_truncation_warnings":  self._truncation_warn_count,
            },
        }

    def reset_statistics(self) -> None:
        """
        Reset all accumulated truncation/padding statistics to zero.

        Useful when the same pipeline instance is reused across multiple
        dataset passes and per-pass statistics are needed rather than
        aggregate statistics.
        """
        self._n_processed              = 0
        self._n_truncated              = 0
        self._n_padded                 = 0
        self._total_frames_removed     = 0
        self._total_frames_padded      = 0
        self._truncation_warn_count    = 0

        logger.debug(
            "FeaturePipeline statistics reset.",
            extra={"stage": "pipeline"},
        )

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        aug_status = "enabled" if self._config.augmentation.enabled else "disabled"
        return (
            f"FeaturePipeline("
            f"seq_len={self._seq_len}, "
            f"landmark_config={self._lm_config!r}, "
            f"feature_dim={self._feature_dim}, "
            f"z_clip=±{self._z_clip}, "
            f"normalise_pose={self._normalise_pose}, "
            f"augmentation={aug_status}, "
            f"seed={self._seed}, "
            f"clips_processed={self._n_processed})"
        )