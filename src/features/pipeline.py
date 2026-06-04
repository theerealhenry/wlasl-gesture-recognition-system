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
    1. Shape validation     — fail immediately on corrupt input before any work
    2. Copy                 — caller's array is NEVER mutated
    3. Wrist-relative norm  — removes positional noise, preserves hand shape
    4. Z-coordinate clip    — removes physically implausible depth outliers
    5. Landmark config      — slice to hands_only / pose_only / full
    6. Pad / centre-crop    — fixed output sequence length, tracks truncation stats
    7. Augmentation         — training mode only, NEVER at inference
    8. float32 cast + return — guaranteed dtype at every call site

Design decisions (all evidence-based from Notebook 03 executive report)
------------------------------------------------------------------------

    Wrist-relative normalisation
        Subtract LH/RH wrist (landmark 0) from all landmarks in that hand,
        per frame, for detected frames only. This removes the absolute screen
        position of the signer (where they stand in the frame) while preserving
        the hand SHAPE — the discriminative signal for sign identity.
        Pose is NEVER normalised. Notebook 03 F3 confirms: body position in
        camera space is signal (pose non-zero rate 96.67%, pose std is
        the most discriminative pose feature).

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
        CRITICAL: landmark slicing happens AFTER normalisation and z-clipping.
        The full 225-vector must be passed through steps 3 and 4 regardless
        of which config is selected.

    Centre-crop truncation strategy
        With mean clip length 67.6 frames (Notebook 03 F1) and typical seq_len
        values of 60–100, truncation is common. ASL signs have preparatory and
        release movements at both temporal ends — the peak discriminative motion
        is concentrated in the temporal centre. Centre-cropping (removing
        symmetrically from both ends) maximises retention of sign content.
        Compare: head-cropping (discard end) would remove the release phase;
        tail-cropping (discard start) would remove the preparation phase.
        Both are more likely to discard discriminative frames than centre-crop.

    Right-pad with zeros
        Short clips are padded at the right (end) with zeros. Zero-padded
        frames are semantically identical to zero-fill detection-failure frames
        — the LSTM already learns these as non-informative. Padding at the end
        is consistent with temporal left-to-right processing: the LSTM has full
        sign context before encountering the padding zone.

    Truncation tracking
        Statistics are accumulated across all __call__ invocations and included
        in get_pipeline_metadata(). This directly informs the interpretation
        of the Stage 5 Group 3 sequence-length ablation: a run where 90% of
        clips are truncated at seq_len=30 should show lower accuracy than a run
        at seq_len=80 where only ~20% are truncated. The statistics make this
        causal path transparent.

    Augmentation after pad/truncate
        Augmentation is applied to the (seq_len, feature_dim) fixed-length
        array, NOT to the raw (T_raw, 225) array. This is required because
        AugmentationPipeline expects a fixed-shape input (its temporal jitter
        and speed jitter transforms must preserve shape, and the shape they
        preserve is seq_len × feature_dim). Augmenting before pad/truncate
        would produce variable-length intermediate arrays.

    Vectorised wrist normalisation
        Rather than a Python loop over detected frames (O(T) function call
        overhead), normalisation uses NumPy fancy indexing + broadcasting:
        one array operation over all detected frames simultaneously. For a
        60-frame clip this is ~60× fewer Python-level operations than a loop.

Notebook 03 findings incorporated
----------------------------------
    F1  seq_len ablation extended to {20,30,40,60,80,100} — sequence_length
        is now a first-class config parameter, not a hardcoded constant.
    F2  Left-hand 70% missing is SEMANTIC signal; zero-fill frames must pass
        through normalisation unchanged — enforced by detection mask guard.
    F3  Hands-only Fisher=0.752 motivates the landmark_config ablation.
    F4  Pose std > pose mean for discriminability — pose is retained by default.
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
    TFLite export verification, real-time inference.

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
        If ``config.data.normalise_pose`` is True — this is an unsupported
        configuration that contradicts Notebook 03 F3 findings.

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
        self._seq_len: int           = int(config.data.sequence_length)
        self._z_clip: float          = float(config.data.z_coord_clip)
        self._normalise_pose: bool   = bool(config.data.normalise_pose)
        self._lm_config: str         = str(config.data.landmark_config)
        self._flip_thresh: float     = float(config.data.flip_min_hand_presence)

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
        self._lm_slice: slice = LANDMARK_CONFIGS[self._lm_config]
        self._feature_dim: int = self._lm_slice.stop - self._lm_slice.start

        # --- Augmentation pipeline ---
        self._aug_pipeline = AugmentationPipeline(
            config=config.augmentation,
            seed=int(config.seed),
            flip_min_hand_presence=self._flip_thresh,
        )

        # --- Truncation tracking (accumulated across all __call__ invocations) ---
        self._n_processed: int             = 0
        self._n_truncated: int             = 0
        self._n_padded: int                = 0
        self._total_frames_removed: int    = 0
        self._total_frames_padded: int     = 0
        self._truncation_warn_count: int   = 0  # rate-limits heavy-truncation warnings

        logger.info(
            "FeaturePipeline initialised | "
            f"seq_len={self._seq_len} | "
            f"landmark_config={self._lm_config} | "
            f"feature_dim={self._feature_dim} | "
            f"z_clip=±{self._z_clip} | "
            f"normalise_pose={self._normalise_pose} | "
            f"flip_min_hand_presence={self._flip_thresh} | "
            f"augmentation={'enabled' if config.augmentation.enabled else 'disabled'} | "
            f"truncation_strategy={_TRUNCATION_STRATEGY} | "
            f"padding_strategy={_PADDING_STRATEGY}",
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
            Controls whether augmentation (step 7) is applied.
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
            If arr.ndim != 2 — wrong array dimensionality.
        ValueError
            If arr.shape[1] != FEATURE_SIZE (225) — wrong feature dimension.
            Note: always pass the full 225-dim array; landmark config slicing
            happens INSIDE the pipeline after normalisation is applied to the
            full vector.

        Examples
        --------
        Training usage (GestureDataset):
            result = pipeline(arr_raw, training=True, clip_idx=42)

        Inference usage (GesturePredictor, TFLite verify):
            result = pipeline(arr_raw, training=False)
            # clip_idx irrelevant; same output for any clip_idx value
        """
        # ------------------------------------------------------------------
        # Step 1: Input validation — catch corrupt arrays before any work
        # ------------------------------------------------------------------
        if arr.ndim != 2:
            raise ValueError(
                f"FeaturePipeline expects a 2D landmark array (T_raw, {FEATURE_SIZE}), "
                f"got ndim={arr.ndim}, shape={arr.shape}. "
                "Ensure the .npy file is not corrupted and was produced by "
                "LandmarkExtractor (schema version 1.2)."
            )
        if arr.shape[1] != FEATURE_SIZE:
            raise ValueError(
                f"FeaturePipeline expects feature dimension {FEATURE_SIZE} "
                f"(63 LH + 63 RH + 99 pose), got {arr.shape[1]}. "
                "Always pass the full 225-dim array — landmark config slicing "
                "is applied INSIDE the pipeline after normalisation. "
                f"For landmark_config='{self._lm_config}', the output will "
                f"be sliced to {self._feature_dim} dims in step 5."
            )

        # ------------------------------------------------------------------
        # Step 2: Copy — the caller's array is NEVER mutated
        #
        # This single copy is the contract boundary. Every subsequent step
        # works on this copy. AugmentationPipeline receives this copy and
        # does not make an additional top-level copy (per the documented
        # AugmentationPipeline.__call__ contract).
        # ------------------------------------------------------------------
        arr = arr.copy().astype(np.float32)

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
        # Step 5: Landmark configuration selection
        #
        # Applied AFTER normalisation so that z-clipping and wrist subtraction
        # always operate on the full 225-dim vector. This guarantees that:
        # (a) a full run and a hands_only run receive identically normalised
        #     hand features — their wrist normalisation is identical;
        # (b) a pose_only run still benefits from the z-clip on pose z-coords.
        # ------------------------------------------------------------------
        arr = self._select_landmark_config(arr)

        # ------------------------------------------------------------------
        # Step 6: Pad or centre-crop to seq_len
        # ------------------------------------------------------------------
        arr = self._pad_or_truncate(arr)

        # ------------------------------------------------------------------
        # Step 7: Augmentation — ONLY in training mode
        #
        # Applied to the fixed-length (seq_len, feature_dim) array.
        # Must never be applied at inference — this is enforced by the
        # training flag, which is False by default.
        # ------------------------------------------------------------------
        if training and self._config.augmentation.enabled:
            arr = self._aug_pipeline(arr, clip_idx=clip_idx)

        # ------------------------------------------------------------------
        # Step 8: Guarantee float32 output dtype
        # ------------------------------------------------------------------
        self._n_processed += 1
        return arr.astype(np.float32)

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

        Zero-fill invariant (critical):
            Detection masks are computed on the ORIGINAL arr values BEFORE
            any modification. A zero-filled frame (MediaPipe detection failure
            or padding) has all-zero landmarks. If we subtracted (0,0,0) from
            (0,0,0) we would get (0,0,0) — mathematically harmless but wrong
            for one reason: after normalisation, a detected hand has its wrist
            at (0,0,0). We cannot distinguish a detected hand with zero wrist
            position from a zero-filled (not detected) hand. The mask guard
            ensures only genuinely detected frames are modified.

        Post-normalisation: LH wrist (feature indices 0:3) = (0, 0, 0) for
            all detected frames. RH wrist (feature indices 63:66) = (0, 0, 0)
            for all detected frames. Detection in downstream code (including
            AugmentationPipeline detection masks) still works because the
            OTHER 20 landmarks per hand remain non-zero.

        Vectorised implementation:
            Rather than a Python loop over T detected frames (O(T) Python
            overhead), we use NumPy fancy indexing to extract all detected
            frames simultaneously, then subtract the wrist vector in a single
            broadcast operation. For a 60-frame clip this is ~60× fewer Python-
            level function calls than a per-frame loop.

        Pose:
            NOT normalised. Body position, arm angle, and torso orientation
            relative to the camera are discriminative features (Notebook 03 F3:
            pose std is more discriminative than pose mean). Normalising pose
            would remove inter-sign body positioning differences.

        Parameters
        ----------
        arr : np.ndarray
            Copy of the input array, shape (T, 225), dtype float32.
            Modified in-place on the copy (not the caller's original).

        Returns
        -------
        np.ndarray
            Shape (T, 225), dtype float32. Same object as input (modified
            in-place for efficiency).
        """
        T = arr.shape[0]

        # --- Left hand ---
        # Detection mask: True for frames where LH slice has any non-zero value.
        # Shape: (T,) bool.
        lh_detected = arr[:, LEFT_HAND_SLICE].any(axis=1)

        if lh_detected.any():
            # Extract detected frame indices
            lh_frame_idx = np.where(lh_detected)[0]

            # Wrist for each detected frame: shape (n_lh, 3)
            # LH wrist = landmark 0 = feature indices [0:3] within LEFT_HAND_SLICE
            lh_wrist = arr[lh_frame_idx, :3].copy()  # shape (n_lh, 3)

            # Tile to full hand feature size: (n_lh, 3) → (n_lh, 63)
            # tile(wrist, 21): repeat the (x,y,z) triplet 21 times to match
            # the 21-landmark layout [x0,y0,z0, x1,y1,z1, ..., x20,y20,z20]
            lh_wrist_tiled = np.tile(lh_wrist, N_HAND_LANDMARKS)  # (n_lh, 63)

            # Subtract: all detected frames simultaneously, no Python loop
            arr[
                np.ix_(lh_frame_idx, range(LEFT_HAND_SLICE.start, LEFT_HAND_SLICE.stop))
            ] -= lh_wrist_tiled

        # --- Right hand ---
        rh_detected = arr[:, RIGHT_HAND_SLICE].any(axis=1)

        if rh_detected.any():
            rh_frame_idx = np.where(rh_detected)[0]

            # RH wrist = landmark 0 = feature indices [63:66] in the full vector
            rh_wrist = arr[rh_frame_idx, RIGHT_HAND_SLICE.start:RIGHT_HAND_SLICE.start + 3].copy()

            rh_wrist_tiled = np.tile(rh_wrist, N_HAND_LANDMARKS)  # (n_rh, 63)

            arr[
                np.ix_(rh_frame_idx, range(RIGHT_HAND_SLICE.start, RIGHT_HAND_SLICE.stop))
            ] -= rh_wrist_tiled

        # --- Pose: NOT normalised ---
        # Pose normalisation is explicitly disabled (config.data.normalise_pose
        # is validated to be False in __init__). No code needed here.

        return arr

    def _apply_z_clip(self, arr: np.ndarray) -> np.ndarray:
        """
        Soft-clip z-coordinates to the range [−z_clip, +z_clip].

        Z-coordinates in the 225-element feature vector are located at every
        third position starting from index 2: [2, 5, 8, 11, ...].
        The unified ``arr[:, 2::3]`` slice selects ALL z-coordinates across
        all three feature bands (LH, RH, pose) in one operation.

        Why unified rather than per-band:
            MediaPipe's depth estimation is relative to the detected component
            in all three cases. Outlier depth values (>±0.08) occur for the
            same physical reasons (fast motion, extrapolation at clip boundaries,
            partial occlusion) regardless of which component they belong to.
            Applying a single clip rule to all z-coordinates is simpler and
            correct.

        Why this magnitude (±0.10):
            Notebook 03 F8: the z-coordinate distribution is tightly clustered
            near −0.02 for both hands with a long left tail. Values beyond
            ±0.08 represent physically implausible depth positions (a hand
            significantly in front of or behind the camera plane). The ±0.10
            threshold gives a comfortable safety margin above the observed
            outlier boundary without truncating the meaningful z variation
            in the central distribution.

        Parameters
        ----------
        arr : np.ndarray
            Shape (T, 225), dtype float32. Modified in-place on the copy.

        Returns
        -------
        np.ndarray
            Same object as input with z-coordinates clipped in-place.
        """
        arr[:, 2::3] = np.clip(arr[:, 2::3], -self._z_clip, self._z_clip)
        return arr

    def _select_landmark_config(self, arr: np.ndarray) -> np.ndarray:
        """
        Slice the feature vector to the configured landmark subset.

        Landmark configuration mappings (from constants.LANDMARK_CONFIGS):
            "full"       → arr[:, 0:225]   → (T, 225)  — both hands + pose
            "hands_only" → arr[:, 0:126]   → (T, 126)  — LH + RH only
            "pose_only"  → arr[:, 126:225] → (T, 99)   — pose skeleton only

        This step is applied AFTER normalisation (step 3) and z-clipping (step 4).
        The rationale: all three configs must receive identically preprocessed
        hand features to ensure fair comparison in the Stage 5 Group 4 ablation.
        A ``hands_only`` run and a ``full`` run must differ ONLY in the presence
        or absence of pose columns — not in how their hand features were processed.

        Returns a NumPy view (not a copy) when the slice is contiguous, which
        is memory-efficient. The subsequent ``_pad_or_truncate`` step only reads
        from this view and returns a new allocation, so using a view is safe.

        Parameters
        ----------
        arr : np.ndarray
            Shape (T, 225), dtype float32.

        Returns
        -------
        np.ndarray
            Shape (T, feature_dim) where feature_dim is determined by
            ``self._lm_config``. May be a view of ``arr``.
        """
        return arr[:, self._lm_slice]

    def _pad_or_truncate(self, arr: np.ndarray) -> np.ndarray:
        """
        Bring the sequence to exactly ``self._seq_len`` frames.

        Two cases:

        TRUNCATION (T_raw > seq_len)  — centre-crop
        ────────────────────────────
        Algorithm:
            remove = T_raw - seq_len
            start  = remove // 2
            end    = T_raw - (remove - start)
            result = arr[start:end]          # shape (seq_len, D)

        Why centre-crop (not head-crop or tail-crop):
            ASL signs have preparatory movements at the start and release
            movements at the end. Peak discriminative motion is concentrated
            in the temporal centre of the clip. Removing symmetrically from
            both ends maximises retention of sign-informative content. This
            is the standard approach in sign language recognition literature
            (e.g., Koller et al. 2015, Li et al. 2020).

        Example (T_raw=120, seq_len=60):
            remove = 60, start = 30, end = 90
            Keeps frames [30:90] — the central 60 of 120 frames.

        Truncation statistics:
            - n_truncated incremented for each truncated clip
            - total_frames_removed accumulates the surplus frame count
            - Heavy truncation (>TRUNCATION_WARN_FRACTION of clip removed)
              is logged at WARNING level (rate-limited to avoid log spam)

        PADDING (T_raw < seq_len)  — right-pad with zeros
        ─────────────────────────
        Algorithm:
            pad_count = seq_len - T_raw
            padding   = np.zeros((pad_count, D), dtype=float32)
            result    = concatenate([arr, padding], axis=0)

        Why right-pad:
            Zero-padded frames are semantically identical to zero-fill frames
            from MediaPipe detection failures — both represent "no hand present
            in this frame." The LSTM has been trained to treat zero-fill frames
            as non-informative regardless of their cause. Padding at the right
            (temporal end) is consistent with LSTM left-to-right processing:
            the model sees the full sign trajectory before encountering the
            silent padding region.

        NO-OP (T_raw == seq_len):
            Returns arr unchanged (O(1), no allocation).

        Parameters
        ----------
        arr : np.ndarray
            Shape (T_raw, D) where D = self._feature_dim.

        Returns
        -------
        np.ndarray
            Shape (seq_len, D), dtype float32. New allocation in all cases
            except the no-op path.

        Raises
        ------
        RuntimeError
            If the output shape is not exactly (seq_len, D) — indicates a
            bug in the crop/pad arithmetic.
        """
        T_raw, D = arr.shape

        if T_raw == self._seq_len:
            # No-op: exact length match, no allocation needed
            return arr

        if T_raw > self._seq_len:
            # Centre-crop: remove surplus frames symmetrically from both ends
            remove = T_raw - self._seq_len
            start  = remove // 2
            end    = T_raw - (remove - start)
            result = arr[start:end].copy()  # copy: materialise the slice

            # --- Truncation tracking ---
            self._n_truncated        += 1
            self._total_frames_removed += remove

            # Warn on heavy truncation (rate-limited)
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
            # Right-pad: append zero frames to reach seq_len
            pad_count = self._seq_len - T_raw
            padding   = np.zeros((pad_count, D), dtype=np.float32)
            result    = np.concatenate([arr, padding], axis=0)

            # --- Padding tracking ---
            self._n_padded        += 1
            self._total_frames_padded += pad_count

        # --- Shape guard: defence against arithmetic bugs ---
        if result.shape != (self._seq_len, D):
            raise RuntimeError(
                f"FeaturePipeline._pad_or_truncate: output shape {result.shape} "
                f"!= expected ({self._seq_len}, {D}). "
                f"Inputs: T_raw={T_raw}, seq_len={self._seq_len}, D={D}. "
                "This is a bug in the pad/truncate arithmetic — please report it."
            )

        return result

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
        (60, 225)   # for landmark_config="full",  seq_len=60
        (60, 126)   # for landmark_config="hands_only", seq_len=60
        (80, 99)    # for landmark_config="pose_only",  seq_len=80
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
           in ``gesture_model_metadata.json`` so that any consumer of the
           model knows exactly what preprocessing to apply.

        2. **GesturePredictor reconstruction** (Stage 7): at inference time,
           GesturePredictor reads this metadata to instantiate an identical
           FeaturePipeline without needing access to the original training config.

        3. **Ablation audit trail** (Stage 5/6): logged to MLflow via
           ``mlflow.log_dict(metadata, "pipeline_metadata.json")`` alongside
           each training run, creating a complete record of what preprocessing
           was applied in each experiment.

        4. **Truncation analysis** (Stage 6): the ``truncation_stats`` field
           enables the sequence-length ablation interpretation. A run where
           80% of training clips are truncated at seq_len=30 is causally
           linked to lower accuracy than a seq_len=80 run where only 12% are
           truncated — the metadata makes this relationship explicit.

        Returns
        -------
        dict[str, Any]
            Flat-ish dictionary with all pipeline parameters and accumulated
            runtime statistics. All values are JSON-serialisable.
        """
        # --- Truncation rate ---
        truncation_rate = (
            self._n_truncated / self._n_processed
            if self._n_processed > 0 else 0.0
        )

        # --- Padding rate ---
        padding_rate = (
            self._n_padded / self._n_processed
            if self._n_processed > 0 else 0.0
        )

        # --- Mean frames removed per truncated clip ---
        mean_frames_removed = (
            self._total_frames_removed / self._n_truncated
            if self._n_truncated > 0 else 0.0
        )

        # --- Mean frames padded per padded clip ---
        mean_frames_padded = (
            self._total_frames_padded / self._n_padded
            if self._n_padded > 0 else 0.0
        )

        return {
            # ── Core transform parameters ───────────────────────────────────
            "sequence_length":          self._seq_len,
            "feature_dim":              self._feature_dim,
            "landmark_config":          self._lm_config,
            "normalisation":            _NORMALISATION_NAME,
            "normalise_pose":           self._normalise_pose,
            "z_coord_clip":             self._z_clip,
            "flip_min_hand_presence":   self._flip_thresh,
            "truncation_strategy":      _TRUNCATION_STRATEGY,
            "padding_strategy":         _PADDING_STRATEGY,

            # ── Feature vector layout (for model card / GesturePredictor) ───
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

            # ── Transform chain (for documentation / audit) ─────────────────
            "transform_chain": [
                "shape_validation",
                "copy",
                "wrist_relative_normalisation",
                f"z_coord_clip(±{self._z_clip})",
                f"landmark_config_select({self._lm_config})",
                f"pad_or_truncate(seq_len={self._seq_len}, "
                f"truncation={_TRUNCATION_STRATEGY}, "
                f"padding={_PADDING_STRATEGY})",
                "augmentation(training_mode_only)",
                "float32_cast",
            ],

            # ── Augmentation configuration (for model card) ──────────────────
            "augmentation": self._aug_pipeline.get_metadata(),

            # ── Runtime statistics (accumulated since instantiation) ─────────
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
        dataset passes (e.g., re-running validation after training is complete)
        and per-pass statistics are needed rather than aggregate statistics.
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
            f"clips_processed={self._n_processed})"
        )