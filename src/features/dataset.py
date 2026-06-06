"""
src/features/dataset.py
========================
GestureDataset — landmark loading, preprocessing, and tf.data pipeline
for the WLASL 35-class gesture recognition system.

Overview
--------
This module bridges raw .npy landmark files on disk and the TensorFlow
training loop. Three responsibilities:

    1. Inventory loading  — reads landmark_inventory.csv and the three split
                            CSVs, cross-references them, resolves every usable
                            clip's on-disk path.

    2. Two-phase preloading — at construction every .npy is loaded into RAM
                              and pushed through FeaturePipeline in two ways:

                              PHASE 1 (all clips):
                                (T_raw, 225) → norm → z_clip → pad
                                             → [no augmentation]
                                             → landmark_select
                                             → (seq_len, feature_dim)   [arr_final]

                              PHASE 2 (training clips, aug enabled only):
                                (T_raw, 225) → pipeline.pre_augmentation()
                                             → (seq_len, 225)            [arr_pre_select]

                              arr_pre_select feeds per-epoch augmentation
                              in the training Dataset.map(), exactly matching
                              FeaturePipeline.__call__(arr, training=True).

    3. tf.data.Dataset construction — load_split() assembles numpy-backed
                                      Datasets with per-epoch augmentation
                                      (training only), shuffle, batch, prefetch.

Critical design decisions
--------------------------

A. Public API contract with FeaturePipeline
   This file NEVER calls any private method of FeaturePipeline. All coupling
   is through the public API:
       pipeline(arr, training=False)     →  arr_final   (phase 1)
       pipeline.pre_augmentation(arr)    →  arr_pre_select  (phase 2)

   FeaturePipeline MUST expose:

       def pre_augmentation(self, arr: np.ndarray) -> np.ndarray:
           \"\"\"
           Apply norm → z_clip → pad on the full 225-dim array WITHOUT
           augmentation or landmark config selection.
           Returns shape (seq_len, 225) float32.
           \"\"\"

   If the pipeline's deterministic prefix changes (e.g. a temporal_smoothing
   step is added between norm and z_clip), only pre_augmentation() in
   pipeline.py needs updating. dataset.py is unchanged.

B. Epoch-varied augmentation seeding
   AugmentationPipeline seeds each clip's RNG as:
       rng = numpy.random.default_rng(base_seed XOR clip_idx)

   If clip_idx were a fixed position, every epoch would produce identical
   augmentations. To vary across epochs, _build_augmented_dataset() captures
   a call-time epoch_val (incremented once per load_split() call):
       effective_clip_idx = stable_clip_idx ^ epoch_val

   This gives:
     - Different augmentation every epoch (epoch_val changes each call)
     - Full reproducibility: (config, seed, epoch_number) always produces
       the same augmented tensors
     - No dependency on tf.random or TF graph-side state

    # IMPORTANT: To get varied augmentation across epochs, the training loop in
    # run_training.py MUST call load_split() once per epoch (not once total):
    #
    #     for epoch in range(cfg.training.epochs):
    #         train_ds = dataset.load_split("train", training=True)
    #         model.fit(train_ds, epochs=1, ...)
    #
    # Calling model.fit(train_ds, epochs=50) with a single dataset object will
    # NOT increment _epoch_counter between epochs — all epochs will receive
    # identical augmentation (effective_clip_idx = stable_clip_idx ^ 0 always).
    # The per-epoch load_split() call is the enforced contract for this class.

C. Augmentation fallback (arr_pre_select is None)
   If pipeline.pre_augmentation() fails for a clip, arr_pre_select is None.
   The training Dataset.map() then returns arr_final directly — the correct
   deterministic tensor, with no augmentation. No fake feature construction
   (zero-padding absent landmark bands) is ever used, as that would generate
   anatomically implausible inputs.

D. Cross-split duplicate detection
   A video_id appearing in two different splits is a data leakage event that
   the signer-aware splitter guarantees cannot happen. If detected, a
   RuntimeError is raised immediately — this is never silently skipped.
   Within-split duplicates (same clip listed twice in one CSV) are silently
   skipped; they are harmless CSV artefacts.

E. Pre-loading rationale
   Full extracted dataset ~20–30 MB in float32. Preloading into RAM eliminates
   all disk I/O from the training loop across all 12+ experiments × 50 epochs.

F. Class weight computation
   Computed from the TRAINING split only, using the sklearn-compatible formula:
       weight[c] = n_train_usable / (n_classes * count_in_train[c])

G. Thread safety
   GestureDataset is NOT thread-safe. Caches are built in __init__ and
   read-only thereafter (safe under CPython GIL). FeaturePipeline holds
   mutable truncation counters; do not share across worker processes.

H. Performance (tf.numpy_function)
   Augmentation runs via tf.numpy_function (pure NumPy). For WLASL (339 clips,
   CPU training) the Python boundary overhead is negligible. For large datasets
   with GPU training, consider replacing augmentation with native TF ops.

Notebook 03 findings
---------------------
    F2   LH 70% missing is semantic signal; zero-fill frames pass unchanged.
    F4   Pose non-zero 96.67%; all 225 dims pass to augmentation as-is.
    F6   Signer 11 dominates 10 training signs; signer metadata preserved.
    F11  21 singleton val clips; macro-F1 is the primary metric.
    F12  Confusable pairs available via LabelMap.get_confusable_signs().
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.features.augmentation import AugmentationPipeline
from src.features.constants import (
    FEATURE_SIZE,
    LANDMARK_CONFIGS,
    LANDMARK_INVENTORY_FILENAME,
    MIN_USABLE_DETECTED_FRAMES,
)
from src.features.pipeline import FeaturePipeline
from src.utils.label_map import LabelMap, get_label_map
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_USABLE_OUTCOMES: frozenset[str] = frozenset({"extracted", "cached"})

_REQUIRED_SPLIT_COLS: frozenset[str] = frozenset(
    {"video_id", "sign_label", "class_idx", "signer_id"}
)

_REQUIRED_INVENTORY_COLS: frozenset[str] = frozenset(
    {"video_id", "outcome", "output_path", "detected_frames"}
)

_UNKNOWN_SIGNER_ID: int = -1
_PRELOAD_LOG_INTERVAL: int = 50
_MIN_CLIPS_FOR_LEARNING: int = 2


# ---------------------------------------------------------------------------
# Numeric sort key for WLASL video IDs
# ---------------------------------------------------------------------------

def _video_id_sort_key(video_id: str) -> Any:
    """
    Return a sort key for a WLASL video_id string.

    WLASL IDs are numeric strings ("12345"). Sorting as raw strings gives
    lexicographic order ("10" < "9"), which is wrong. This returns the
    integer value when possible, falling back to the string for non-numeric IDs.
    Used everywhere entries need stable, numerically-correct ordering.
    """
    try:
        return (0, int(video_id))
    except (ValueError, TypeError):
        return (1, video_id)


# ---------------------------------------------------------------------------
# Internal cache entry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _CacheEntry:
    """
    Immutable record for one preloaded clip.

    Attributes
    ----------
    video_id : str
    sign_label : str
    class_idx : int
        Validated to be in [0, num_classes) during preload.
    signer_id : int
        _UNKNOWN_SIGNER_ID (-1) if absent.
    split : str
        "train", "val", or "test".
    npy_path : str
    arr_final : np.ndarray
        (seq_len, feature_dim) float32 — always present. Used for val/test
        and training without augmentation.
    arr_pre_select : Optional[np.ndarray]
        (seq_len, 225) float32 — present only for training clips when
        augmentation is enabled. None otherwise.
    detected_frames : int
    """
    video_id: str
    sign_label: str
    class_idx: int
    signer_id: int
    split: str
    npy_path: str
    arr_final: np.ndarray
    arr_pre_select: Optional[np.ndarray]
    detected_frames: int = 0

    def __hash__(self) -> int:
        # Hash on (video_id, split) for full identity.
        # Two entries with the same video_id but different splits would have
        # identical hashes if only video_id were used, masking cross-split
        # duplicates. (video_id, split) pairs are always unique in a valid
        # cache because cross-split duplicates are caught by the preload guard.
        return hash((self.video_id, self.split))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _CacheEntry):
            return NotImplemented
        return self.video_id == other.video_id and self.split == other.split


# ---------------------------------------------------------------------------
# GestureDataset
# ---------------------------------------------------------------------------

class GestureDataset:
    """
    Loads, preprocesses, and serves WLASL landmark data as tf.data.Dataset.

    Parameters
    ----------
    config : ExperimentConfig
        Full frozen experiment config from load_config(). Reads:
            config.data.sequence_length, landmark_config, landmark_dir,
            splits_dir, flip_min_hand_presence, augmentation.*, seed,
            training.batch_size, num_classes.
    pipeline : FeaturePipeline
        Pre-constructed pipeline shared across all splits. MUST expose
        pipeline.pre_augmentation(arr) as a public method (see module
        docstring section A for the contract).
    splits_dir : str | Path | None
        Override for split CSV directory. Defaults to config.data.splits_dir.
    landmarks_dir : str | Path | None
        Override for landmarks root. Defaults to config.data.landmark_dir.
    label_map_path : str | Path | None
        Path to label_map_v1.json. Defaults to
        <repo_root>/artifacts/label_map_v1.json.
    preload_val_test_aug_cache : bool
        If True, build arr_pre_select for val/test clips as well.
        Production runs: leave False.

    Raises
    ------
    AttributeError
        If pipeline does not expose pre_augmentation().
    ValueError
        If landmark_config is unknown, required CSV columns are missing,
        or no usable clips survive filtering.
    FileNotFoundError
        If any split CSV or landmark_inventory.csv is missing.
    RuntimeError
        If all training clips fail to load, or if a cross-split duplicate
        video_id is detected (data leakage event).
    """

    def __init__(
        self,
        config: Any,
        pipeline: FeaturePipeline,
        splits_dir: Optional[str | Path] = None,
        landmarks_dir: Optional[str | Path] = None,
        label_map_path: Optional[str | Path] = None,
        preload_val_test_aug_cache: bool = False,
    ) -> None:
        t_init = time.time()

        self._config   = config
        self._pipeline = pipeline

        # ── Validate public pipeline API ────────────────────────────────────
        if not callable(getattr(pipeline, "pre_augmentation", None)):
            raise AttributeError(
                "GestureDataset requires FeaturePipeline to expose a public "
                "method 'pre_augmentation(arr: np.ndarray) -> np.ndarray'. "
                "Add this method to FeaturePipeline. See module docstring A."
            )

        # ── Resolve directories ─────────────────────────────────────────────
        _repo_root = Path(__file__).resolve().parents[2]

        self._splits_dir = Path(
            splits_dir if splits_dir is not None else config.data.splits_dir
        ).resolve()
        self._landmarks_dir = Path(
            landmarks_dir if landmarks_dir is not None else config.data.landmark_dir
        ).resolve()
        self._label_map_path = Path(
            label_map_path if label_map_path is not None
            else _repo_root / "artifacts" / "label_map_v1.json"
        )

        # ── Core config values ──────────────────────────────────────────────
        self._num_classes: int   = int(config.num_classes)
        self._batch_size: int    = int(config.training.batch_size)
        self._seed: int          = int(config.seed)
        self._seq_len: int       = int(config.data.sequence_length)
        self._aug_enabled: bool  = bool(config.augmentation.enabled)
        self._flip_thresh: float = float(config.data.flip_min_hand_presence)
        self._preload_val_test_aug = preload_val_test_aug_cache

        # ── Validate landmark_config eagerly with a clear error ─────────────
        # LANDMARK_CONFIGS[key] raises KeyError; we convert to ValueError so
        # the user knows which values are valid.
        self._lm_config: str = str(config.data.landmark_config)
        if self._lm_config not in LANDMARK_CONFIGS:
            raise ValueError(
                f"GestureDataset: unknown landmark_config '{self._lm_config}'. "
                f"Valid values: {sorted(LANDMARK_CONFIGS.keys())}. "
                "Check configs/data/*.yaml or your load_config() override."
            )
        self._lm_slice: slice  = LANDMARK_CONFIGS[self._lm_config]
        self._feature_dim: int = self._lm_slice.stop - self._lm_slice.start

        # ── Label map ──────────────────────────────────────────────────────
        self._label_map: LabelMap = get_label_map(self._label_map_path)
        if self._label_map.num_classes != self._num_classes:
            raise ValueError(
                f"LabelMap has {self._label_map.num_classes} classes but "
                f"config.num_classes={self._num_classes}. "
                "Check label_map_path and your config."
            )

        # ── Augmentation pipeline ───────────────────────────────────────────
        # Always constructed (even if disabled) so get_metadata() is always
        # populated. AugmentationPipeline.enabled gates actual transforms.
        self._aug_pipeline = AugmentationPipeline(
            config=config.augmentation,
            seed=self._seed,
            flip_min_hand_presence=self._flip_thresh,
        )

        # ── Per-epoch counter for varied augmentation seeding ───────────────
        # Incremented once per load_split() call so successive calls
        # (one per model.fit epoch) produce different effective_clip_idx values.
        self._epoch_counter = None  # tf.Variable, initialised lazily in load_split()

        logger.info(
            "GestureDataset initialising | "
            f"landmark_config={self._lm_config} | "
            f"feature_dim={self._feature_dim} | "
            f"seq_len={self._seq_len} | "
            f"augmentation={'enabled' if self._aug_enabled else 'disabled'} | "
            f"splits_dir={self._splits_dir} | "
            f"landmarks_dir={self._landmarks_dir}",
            extra={"stage": "dataset"},
        )

        # ── Load CSVs ────────────────────────────────────────────────────────
        self._split_dfs: Dict[str, pd.DataFrame] = self._load_split_csvs()
        self._inventory_df: pd.DataFrame = self._load_inventory()

        self._inv_path_lookup: Dict[str, str] = (
            self._inventory_df.set_index("video_id")["output_path"].to_dict()
        )
        self._inv_detected_lookup: Dict[str, int] = (
            self._inventory_df
            .set_index("video_id")["detected_frames"]
            .astype(int)
            .to_dict()
        )

        # ── Preload ──────────────────────────────────────────────────────────
        self._cache: Dict[str, _CacheEntry] = {}
        self._preload_all_clips()

        if not self._cache:
            raise RuntimeError(
                "GestureDataset: no clips were successfully loaded. "
                "Verify that Stage 3 extraction completed and "
                "landmark_inventory.csv exists at the landmarks directory."
            )

        self._validate_split_coverage()

        # ── Summary ──────────────────────────────────────────────────────────
        elapsed   = time.time() - t_init
        n_train   = sum(1 for e in self._cache.values() if e.split == "train")
        n_val     = sum(1 for e in self._cache.values() if e.split == "val")
        n_test    = sum(1 for e in self._cache.values() if e.split == "test")
        cache_mb  = self._cache_ram_mb()

        logger.info(
            "GestureDataset ready | "
            f"clips={len(self._cache)} "
            f"(train={n_train}, val={n_val}, test={n_test}) | "
            f"output_shape=({self._seq_len},{self._feature_dim}) | "
            f"aug_cache={'yes' if self._aug_enabled else 'no'} | "
            f"ram_cache={cache_mb:.1f} MB | "
            f"init_time={elapsed:.1f}s",
            extra={"stage": "dataset"},
        )

    # ══════════════════════════════════════════════════════════════════════
    # CSV / inventory loading
    # ══════════════════════════════════════════════════════════════════════

    def _load_split_csvs(self) -> Dict[str, pd.DataFrame]:
        split_dfs: Dict[str, pd.DataFrame] = {}
        for split_name in ("train", "val", "test"):
            csv_path = self._splits_dir / f"{split_name}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(
                    f"Split CSV not found: {csv_path}. "
                    "Run Stage 1 (pipelines/run_preprocessing.py) first."
                )
            df = pd.read_csv(csv_path, dtype={"video_id": str})
            missing = _REQUIRED_SPLIT_COLS - set(df.columns)
            if missing:
                raise ValueError(
                    f"Split CSV '{csv_path}' missing required columns: "
                    f"{sorted(missing)}."
                )
            df["video_id"] = df["video_id"].astype(str)
            split_dfs[split_name] = df
            logger.debug(
                f"Split CSV: {split_name} | clips={len(df)} | "
                f"classes={df['class_idx'].nunique()} | "
                f"signers={df['signer_id'].nunique()}",
                extra={"stage": "dataset"},
            )
        return split_dfs

    def _load_inventory(self) -> pd.DataFrame:
        inv_path = self._landmarks_dir / LANDMARK_INVENTORY_FILENAME
        if not inv_path.exists():
            raise FileNotFoundError(
                f"Landmark inventory not found: {inv_path}. "
                "Run Stage 3 (pipelines/run_landmark_extraction.py) first."
            )
        df = pd.read_csv(inv_path, dtype={"video_id": str})
        missing = _REQUIRED_INVENTORY_COLS - set(df.columns)
        if missing:
            raise ValueError(
                f"Landmark inventory '{inv_path}' missing columns: "
                f"{sorted(missing)}. Re-run Stage 3 extraction."
            )
        df["video_id"] = df["video_id"].astype(str)
        n_total = len(df)

        df = df[df["outcome"].isin(_USABLE_OUTCOMES)].copy()
        n_after_outcome = len(df)

        df["detected_frames"] = df["detected_frames"].fillna(0).astype(int)
        df = df[df["detected_frames"] >= MIN_USABLE_DETECTED_FRAMES].copy()
        n_after_frames = len(df)

        df = df[
            df["output_path"].notna()
            & (df["output_path"].astype(str).str.strip() != "")
        ].copy()
        n_after_path = len(df)

        logger.info(
            "Landmark inventory | "
            f"total={n_total} | "
            f"after_outcome={n_after_outcome} "
            f"(-{n_total - n_after_outcome}) | "
            f"after_min_frames={n_after_frames} "
            f"(-{n_after_outcome - n_after_frames} <{MIN_USABLE_DETECTED_FRAMES} detected) | "
            f"after_path={n_after_path} "
            f"(-{n_after_frames - n_after_path} null/empty paths)",
            extra={"stage": "dataset"},
        )
        if n_after_path == 0:
            raise ValueError(
                "No usable clips survive landmark inventory filtering. "
                "Check that Stage 3 extraction completed successfully."
            )
        return df

    # ══════════════════════════════════════════════════════════════════════
    # Two-phase preloading
    # ══════════════════════════════════════════════════════════════════════

    def _preload_all_clips(self) -> None:
        """
        Load all usable clips into RAM via the two-phase cache strategy.

        Uses itertuples() (faster than iterrows for large DataFrames).

        Duplicate video_id policy:
          WITHIN a split  → silently skip (harmless CSV artefact).
          ACROSS splits   → raise RuntimeError (data leakage; fatal).

        class_idx validation:
          Every class_idx is checked against [0, num_classes). An out-of-range
          value raises ValueError before any I/O is performed, surfacing bad
          CSV data early rather than causing opaque errors in model.fit().
        """
        t0 = time.perf_counter()

        all_records: List[Dict[str, Any]] = []
        for split_name in ("train", "val", "test"):
            df = self._split_dfs[split_name]
            for row in df.itertuples(index=False):
                all_records.append({
                    "video_id":  str(row.video_id),
                    "sign_label": str(getattr(row, "sign_label", "")),
                    "class_idx": int(row.class_idx),
                    "signer_id": int(getattr(row, "signer_id", _UNKNOWN_SIGNER_ID)),
                    "split":     split_name,
                })

        n_total     = len(all_records)
        n_loaded    = 0
        n_skipped   = 0
        n_failed    = 0
        n_intra_dup = 0

        # Track first-seen split for each video_id to detect cross-split dups.
        video_id_to_split: Dict[str, str] = {}

        logger.info(
            f"Preloading {n_total} clips (train+val+test)...",
            extra={"stage": "dataset"},
        )

        for i, record in enumerate(all_records):
            video_id   = record["video_id"]
            sign_label = record["sign_label"]
            class_idx  = record["class_idx"]
            signer_id  = record["signer_id"]
            split       = record["split"]

            # ── class_idx range check ────────────────────────────────────
            if not (0 <= class_idx < self._num_classes):
                raise ValueError(
                    f"video_id={video_id} ({sign_label}): class_idx={class_idx} "
                    f"is outside valid range [0, {self._num_classes}). "
                    "The split CSV may have been generated with a different label "
                    "map. Re-run Stage 1 to regenerate splits."
                )

            # ── Duplicate video_id detection ─────────────────────────────
            if video_id in video_id_to_split:
                first_split = video_id_to_split[video_id]
                if first_split != split:
                    # Cross-split: data leakage event. Always fatal.
                    raise RuntimeError(
                        f"Data leakage detected: video_id={video_id} appears "
                        f"in split '{first_split}' AND split '{split}'. "
                        "The signer-aware splitter guarantees no clip appears "
                        "in more than one split. Re-run Stage 1 to regenerate "
                        "splits — do not use the current split CSVs for training."
                    )
                else:
                    # Within-split duplicate: harmless, skip silently.
                    logger.debug(
                        f"Within-split duplicate: video_id={video_id} "
                        f"appears more than once in '{split}'. Skipping.",
                        extra={"stage": "dataset", "video_id": video_id},
                    )
                    n_intra_dup += 1
                    continue

            video_id_to_split[video_id] = split

            # ── Inventory lookup ─────────────────────────────────────────
            npy_path_str = self._inv_path_lookup.get(video_id)
            if npy_path_str is None:
                logger.debug(
                    f"video_id={video_id} ({sign_label}) not in inventory "
                    "— clip was not extracted or was skipped by the extractor.",
                    extra={"stage": "dataset", "video_id": video_id},
                )
                n_skipped += 1
                continue

            npy_path = Path(str(npy_path_str))
            if not npy_path.exists():
                logger.warning(
                    f"video_id={video_id}: .npy listed in inventory but "
                    f"missing on disk: {npy_path}. Re-run Stage 3.",
                    extra={"stage": "dataset", "video_id": video_id},
                )
                n_skipped += 1
                continue

            detected_frames = self._inv_detected_lookup.get(video_id, 0)

            # ── Load raw array ───────────────────────────────────────────
            try:
                arr_raw = np.load(str(npy_path), allow_pickle=False)
            except Exception as exc:
                logger.warning(
                    f"Failed to load .npy for {video_id}: "
                    f"{type(exc).__name__}: {exc}",
                    extra={"stage": "dataset", "video_id": video_id},
                )
                n_failed += 1
                continue

            if arr_raw.ndim != 2 or arr_raw.shape[1] != FEATURE_SIZE:
                logger.warning(
                    f"video_id={video_id}: bad shape {arr_raw.shape} "
                    f"(expected (T, {FEATURE_SIZE})). Skipping.",
                    extra={"stage": "dataset", "video_id": video_id},
                )
                n_failed += 1
                continue

            arr_raw = arr_raw.astype(np.float32, copy=False)

            if not np.isfinite(arr_raw).all():
                n_nan = int(np.isnan(arr_raw).sum())
                n_inf = int(np.isinf(arr_raw).sum())
                logger.warning(
                    f"video_id={video_id}: non-finite values "
                    f"(NaN={n_nan}, Inf={n_inf}). Skipping. "
                    "Re-run extraction with --force.",
                    extra={"stage": "dataset", "video_id": video_id},
                )
                n_failed += 1
                continue

            # ── Phase 1: full deterministic pipeline (training=False) ────
            # norm → z_clip → pad_truncate → [no aug] → landmark_select
            try:
                arr_final = self._pipeline(arr_raw, training=False, clip_idx=0)
            except Exception as exc:
                logger.warning(
                    f"Pipeline failed for {video_id}: "
                    f"{type(exc).__name__}: {exc}",
                    extra={"stage": "dataset", "video_id": video_id},
                )
                n_failed += 1
                continue

            expected_final_shape = (self._seq_len, self._feature_dim)
            if arr_final.shape != expected_final_shape:
                logger.warning(
                    f"video_id={video_id}: pipeline returned shape {arr_final.shape}, "
                    f"expected {expected_final_shape}. Skipping. "
                    "This indicates a pipeline configuration mismatch.",
                    extra={"stage": "dataset", "video_id": video_id},
                )
                n_failed += 1
                continue

            # ── Phase 2: pre-augment intermediate via PUBLIC API ─────────
            # Calls pipeline.pre_augmentation() — a public method.
            # GestureDataset NEVER calls private pipeline methods.
            # Returns (seq_len, 225): norm → z_clip → pad, no lm_select.
            arr_pre_select: Optional[np.ndarray] = None
            build_aug_cache = self._aug_enabled and (
                split == "train" or self._preload_val_test_aug
            )

            if build_aug_cache:
                try:
                    arr_pre_select = self._compute_pre_select(arr_raw)
                except Exception as exc:
                    # Non-fatal: clip will not be augmented during training.
                    logger.warning(
                        f"pre_augmentation() failed for {video_id}: "
                        f"{type(exc).__name__}: {exc}. "
                        "This clip will use arr_final (no augmentation).",
                        extra={"stage": "dataset", "video_id": video_id},
                    )
                    arr_pre_select = None

                if arr_pre_select is not None:
                    expected_pre_shape = (self._seq_len, FEATURE_SIZE)
                    if arr_pre_select.shape != expected_pre_shape:
                        logger.warning(
                            f"video_id={video_id}: pre_augmentation returned shape "
                            f"{arr_pre_select.shape}, expected {expected_pre_shape}. "
                            "Falling back to no augmentation for this clip.",
                            extra={"stage": "dataset", "video_id": video_id},
                        )
                        arr_pre_select = None

            self._cache[video_id] = _CacheEntry(
                video_id=video_id,
                sign_label=sign_label,
                class_idx=class_idx,
                signer_id=signer_id,
                split=split,
                npy_path=str(npy_path),
                arr_final=arr_final,
                arr_pre_select=arr_pre_select,
                detected_frames=detected_frames,
            )
            n_loaded += 1

            if (i + 1) % _PRELOAD_LOG_INTERVAL == 0 or (i + 1) == n_total:
                elapsed = time.perf_counter() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0.0
                logger.info(
                    f"Preload | {i + 1}/{n_total} | "
                    f"loaded={n_loaded} skipped={n_skipped} "
                    f"failed={n_failed} intra_dup={n_intra_dup} | "
                    f"{rate:.1f}/s",
                    extra={"stage": "dataset"},
                )

        elapsed_total = time.perf_counter() - t0
        logger.info(
            "Preload complete | "
            f"loaded={n_loaded} | skipped={n_skipped} | "
            f"failed={n_failed} | intra_dup={n_intra_dup} | "
            f"elapsed={elapsed_total:.2f}s",
            extra={"stage": "dataset"},
        )

        if n_total > 0 and n_failed > 0.1 * n_total:
            logger.warning(
                f"{n_failed}/{n_total} clips ({n_failed / n_total:.0%}) "
                "failed — above 10% threshold. Re-run Stage 3.",
                extra={"stage": "dataset"},
            )

    def _compute_pre_select(self, arr_raw: np.ndarray) -> np.ndarray:
        """
        Compute the pre-landmark-select (seq_len, 225) intermediate.

        Delegates entirely to the PUBLIC pipeline.pre_augmentation() method.
        No private pipeline methods are called here.

        Parameters
        ----------
        arr_raw : np.ndarray
            Raw (T_raw, 225) float32 from .npy.

        Returns
        -------
        np.ndarray
            (seq_len, 225) float32. Produced by norm → z_clip → pad.
            No augmentation. No landmark_config slicing.
        """
        return self._pipeline.pre_augmentation(arr_raw)

    # ══════════════════════════════════════════════════════════════════════
    # Post-load validation
    # ══════════════════════════════════════════════════════════════════════

    def _validate_split_coverage(self) -> None:
        """
        Validate split contents after preloading.

        Fatal: training split is empty.
        Error: any class has zero training clips.
        Warning: any val class has exactly 1 clip (Notebook 03 F11).
        """
        for split_name in ("train", "val", "test"):
            entries = [e for e in self._cache.values() if e.split == split_name]
            if not entries:
                if split_name == "train":
                    raise RuntimeError(
                        "No training clips loaded. "
                        "Ensure Stage 3 extraction covered the training split."
                    )
                logger.warning(
                    f"No clips for split '{split_name}'. "
                    f"load_split('{split_name}') will raise RuntimeError.",
                    extra={"stage": "dataset"},
                )
                continue

            classes_present = {e.class_idx for e in entries}
            logger.info(
                f"Split '{split_name}': {len(entries)} clips | "
                f"{len(classes_present)}/{self._num_classes} classes | "
                f"{len({e.signer_id for e in entries})} signers",
                extra={"stage": "dataset"},
            )

        # Singleton detection — Notebook 03 F11
        val_entries = [e for e in self._cache.values() if e.split == "val"]
        if val_entries:
            val_counts: Dict[int, int] = {}
            for e in val_entries:
                val_counts[e.class_idx] = val_counts.get(e.class_idx, 0) + 1
            singletons = [
                self._label_map.get_name_safe(c, f"class_{c}")
                for c, cnt in val_counts.items() if cnt == 1
            ]
            if singletons:
                logger.warning(
                    f"Notebook 03 F11: {len(singletons)} val classes have "
                    f"exactly 1 clip — per-class val metrics unreliable for: "
                    f"{singletons}. Report macro-F1 as the primary metric.",
                    extra={"stage": "dataset"},
                )

        # Missing training classes
        train_classes = {e.class_idx for e in self._cache.values() if e.split == "train"}
        absent = sorted(set(range(self._num_classes)) - train_classes)
        if absent:
            absent_names = [
                self._label_map.get_name_safe(c, f"class_{c}") for c in absent
            ]
            logger.error(
                f"CRITICAL: {len(absent)} classes have ZERO training clips: "
                f"{absent_names}. These classes CANNOT be learned. "
                "Re-check Stage 3 extraction for these signs.",
                extra={"stage": "dataset"},
            )
            raise RuntimeError(
                f"{len(absent)} sign class(es) have zero training clips: "
                f"{absent_names}. "
                "Training cannot proceed — the model cannot learn these classes. "
                "Re-run Stage 3 extraction for the affected signs, or remove them "
                "from the label map if they are genuinely unrecoverable."
            )

    # ══════════════════════════════════════════════════════════════════════
    # tf.data.Dataset construction
    # ══════════════════════════════════════════════════════════════════════

    def load_split(
        self,
        split: str,
        training: bool,
        drop_remainder: bool = False,
    ):
        """
        Build a tf.data.Dataset for the specified split.

        Training (split="train", training=True, aug enabled):
            - Per-epoch augmentation via Dataset.map() with epoch-varied
              clip_idx seeding (each call to load_split produces different
              augmentations because self._epoch_counter increments by 1).
            - Shuffled with reshuffle_each_iteration=True.
            - NOT repeated — the training loop in run_training.py calls
            load_split() once per epoch (epochs=1 per fit call).
            See module docstring for the required training loop pattern.
            - Batched, shape-restored, prefetched.

        Validation/test (training=False):
            - No augmentation — fully deterministic.
            - No shuffle.
            - Identical output on every call.

        Parameters
        ----------
        split : str — "train", "val", or "test"
        training : bool — enables augmentation/shuffle
        drop_remainder : bool — passed to Dataset.batch()

        Returns
        -------
        tf.data.Dataset
            (features, labels): float32 (batch,seq_len,feature_dim),
                                 int32  (batch,)

        Raises
        ------
        ValueError  — bad split name
        RuntimeError — no cached clips for requested split
        """
        import tensorflow as tf

        if split not in ("train", "val", "test"):
            raise ValueError(
                f"split must be 'train', 'val', or 'test'; got '{split}'."
            )

        entries = [e for e in self._cache.values() if e.split == split]
        if not entries:
            raise RuntimeError(
                f"No cached clips for split '{split}'. "
                "Check that _preload_all_clips() completed successfully."
            )

        # Stable numeric sort before shuffle (reproducible across runs)
        entries.sort(key=lambda e: _video_id_sort_key(e.video_id))

        apply_aug = training and self._aug_enabled

        # Lazily create the tf.Variable on first call
        if self._epoch_counter is None:
            self._epoch_counter = tf.Variable(
                0, trainable=False, dtype=tf.int32,
                name="gesture_dataset_epoch_counter",
            )

        # Snapshot current epoch value for shuffle seed, then increment.
        # Each load_split() call gets a unique _epoch_val so shuffle order
        # and augmentation seeding differ across epochs.
        _epoch_val = int(self._epoch_counter.numpy())
        self._epoch_counter.assign_add(1)

        logger.info(
            f"Building tf.data.Dataset | split={split} | "
            f"clips={len(entries)} | training={training} | "
            f"augmentation={'yes' if apply_aug else 'no'} | "
            f"epoch={_epoch_val} | batch_size={self._batch_size}",
            extra={"stage": "dataset"},
        )

        if apply_aug:
            return self._build_augmented_dataset(
                entries, drop_remainder=drop_remainder, epoch_val=_epoch_val
            )
        return self._build_static_dataset(
            entries, training=training, drop_remainder=drop_remainder
        )

    def _build_static_dataset(
        self,
        entries: List[_CacheEntry],
        training: bool,
        drop_remainder: bool,
    ):
        """
        tf.data.Dataset from pre-computed arr_final tensors.

        Used for val/test (always) and training when augmentation is off.
        No tf.numpy_function overhead.
        """
        import tensorflow as tf

        features = np.stack([e.arr_final for e in entries], axis=0).astype(np.float32)
        labels   = np.array([e.class_idx for e in entries], dtype=np.int32)

        ds = tf.data.Dataset.from_tensor_slices((features, labels))
        if training:
            ds = ds.shuffle(
                buffer_size=len(entries),
                seed=self._seed,
                reshuffle_each_iteration=True,
            )
        ds = ds.batch(self._batch_size, drop_remainder=drop_remainder)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def _build_augmented_dataset(
        self,
        entries: List[_CacheEntry],
        drop_remainder: bool,
        epoch_val: int,
    ):
        """
        tf.data.Dataset with per-epoch varied augmentation.

        Augmentation seeding:
            effective_clip_idx = stable_clip_idx ^ epoch_val
        Since epoch_val increments each call, identical clips receive
        different augmentations every epoch. The same (config, seed,
        epoch_val) always produces the same output — fully reproducible.

        Shuffle seed also includes epoch_val so the ordering differs
        between epochs.

        Fallback for clips with arr_pre_select=None:
            arr_final is returned directly — no augmentation, no fake
            feature construction. arr_final is always correct.

        Performance note:
            Augmentation runs inside tf.numpy_function (Python/NumPy).
            For WLASL (339 clips, CPU training) this is negligible.
            See module docstring H for scaling guidance.
        """
        import tensorflow as tf

        # ── Warn about clips without pre_select ───────────────────────────
        n_no_ps = sum(1 for e in entries if e.arr_pre_select is None)
        if n_no_ps > 0:
            logger.warning(
                f"{n_no_ps}/{len(entries)} training clips have no pre_select "
                "cache — they will NOT be augmented. "
                "Re-run preload with augmentation enabled to fix.",
                extra={"stage": "dataset"},
            )

        # ── Build arrays ──────────────────────────────────────────────────
        # For clips without arr_pre_select: store a zero placeholder in the
        # pre_select stack; _augment_fn detects the flag and uses arr_final.
        pre_sel_list: List[np.ndarray] = []
        flag_list:    List[bool]       = []

        for e in entries:
            if e.arr_pre_select is not None:
                pre_sel_list.append(e.arr_pre_select)   # (seq_len, 225)
                flag_list.append(True)
            else:
                pre_sel_list.append(
                    np.zeros((self._seq_len, FEATURE_SIZE), dtype=np.float32)
                )
                flag_list.append(False)

        pre_select_stack = np.stack(pre_sel_list, axis=0).astype(np.float32)
        arr_final_stack  = np.stack(
            [e.arr_final for e in entries], axis=0
        ).astype(np.float32)
        labels_arr   = np.array([e.class_idx for e in entries], dtype=np.int32)
        flags_arr    = np.array(flag_list, dtype=bool)
        # Stable clip indices: position of each clip in the entries list.
        clip_indices = np.arange(len(entries), dtype=np.int32)

        # Capture for closure (avoids 'self' reference inside numpy_function)
        aug_pipeline = self._aug_pipeline
        lm_slice     = self._lm_slice
        feature_dim  = self._feature_dim
        seq_len      = self._seq_len
        _epoch_val   = int(epoch_val)  # immutable; each call gets its own copy

        # _epoch_val (int): used for shuffle seed — Python-side, fixed per load_split() call.
        # epoch_counter_var (tf.Variable): read inside _augment_fn at map() execution time,
        # giving each TF epoch a different augmentation seed when load_split() is called
        # once per epoch.
        epoch_counter_var = self._epoch_counter  # tf.Variable, captured in closure

        def _augment_fn(pre_sel, arr_fin, label, clip_idx_t, has_ps):
            ps_flag = bool(has_ps.item() if hasattr(has_ps, "item") else has_ps)
            idx_int = int(
                clip_idx_t.item() if hasattr(clip_idx_t, "item") else clip_idx_t
            )

            if ps_flag:
                # Read the epoch value from tf.Variable. This variable was incremented
                # once in load_split() before this dataset was built, so all clips in
                # this epoch read the same current_epoch value, producing clip-unique
                # but epoch-varied effective_idx values.
                current_epoch = int(epoch_counter_var.numpy())
                effective_idx = idx_int ^ current_epoch
                augmented = aug_pipeline(pre_sel, clip_idx=effective_idx)
                result = augmented[:, lm_slice].astype(np.float32)
            else:
                result = arr_fin.astype(np.float32)

            return result.astype(np.float32), label.astype(np.int32)

        # ── Assemble dataset ──────────────────────────────────────────────
        ds = tf.data.Dataset.from_tensor_slices(
            (pre_select_stack, arr_final_stack, labels_arr, clip_indices, flags_arr)
        )

        # Shuffle seed varies with epoch so ordering differs across epochs
        ds = ds.shuffle(
            buffer_size=len(entries),
            seed=self._seed ^ _epoch_val,
            reshuffle_each_iteration=True,
        )

        ds = ds.map(
            lambda ps, af, lbl, cidx, hps: tf.numpy_function(
                func=_augment_fn,
                inp=[ps, af, lbl, cidx, hps],
                Tout=[tf.float32, tf.int32],
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True,  # Preserve element order for reproducibility.
                                # At WLASL scale (~236 clips) there is no measurable
                                # throughput difference vs deterministic=False.
        )

        # Restore static shapes lost through tf.numpy_function
        ds = ds.map(
            lambda x, y: (
                tf.ensure_shape(x, [seq_len, feature_dim]),
                tf.ensure_shape(y, []),
            )
        )

        ds = ds.batch(self._batch_size, drop_remainder=drop_remainder)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    # ══════════════════════════════════════════════════════════════════════
    # Class weights
    # ══════════════════════════════════════════════════════════════════════

    def compute_class_weights(self) -> Dict[int, float]:
        """
        Compute sklearn-compatible inverse-frequency class weights from
        the TRAINING split only.

        Formula:
            weight[c] = n_train_usable / (n_classes * count[c])

        Returns
        -------
        Dict[int, float]  —  class_idx → weight, all num_classes entries.
        """
        train_entries = [e for e in self._cache.values() if e.split == "train"]
        n_usable = len(train_entries)

        if n_usable == 0:
            logger.error(
                "compute_class_weights(): no training clips. "
                "Returning uniform weights.",
                extra={"stage": "dataset"},
            )
            return {c: 1.0 for c in range(self._num_classes)}

        class_counts: Dict[int, int] = {}
        for e in train_entries:
            class_counts[e.class_idx] = class_counts.get(e.class_idx, 0) + 1

        class_weights: Dict[int, float] = {}
        for c in range(self._num_classes):
            count     = class_counts.get(c, 0)
            sign_name = self._label_map.get_name_safe(c, f"class_{c}")
            if count >= _MIN_CLIPS_FOR_LEARNING:
                class_weights[c] = n_usable / (self._num_classes * count)
            elif count == 1:
                class_weights[c] = n_usable / (self._num_classes * 1)
                logger.warning(
                    f"Class {c} ('{sign_name}'): only 1 training clip. "
                    f"Weight={class_weights[c]:.3f} — learning unreliable.",
                    extra={"stage": "dataset"},
                )
            else:
                class_weights[c] = 1.0
                logger.error(
                    f"Class {c} ('{sign_name}'): ZERO training clips. "
                    "Weight=1.0 — this class CANNOT be learned.",
                    extra={"stage": "dataset"},
                )

        weight_vals = list(class_weights.values())
        logger.info(
            "Class weights | "
            f"n_train={n_usable} | "
            f"classes_with_data={len(class_counts)} | "
            f"min={min(weight_vals):.4f} | max={max(weight_vals):.4f} | "
            f"ratio={max(weight_vals) / min(weight_vals):.2f}x",
            extra={"stage": "dataset"},
        )
        return class_weights

    # ══════════════════════════════════════════════════════════════════════
    # Analysis helpers for Stage 6
    # ══════════════════════════════════════════════════════════════════════

    def get_split_for_analysis(self, split: str) -> pd.DataFrame:
        """
        Per-clip metadata DataFrame for Stage 6 analysis.

        Columns: video_id, sign_label, class_idx, signer_id, split,
                 npy_path, detected_frames, sign_name.

        Sorted by (class_idx, numeric video_id).
        """
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test'; got '{split}'.")

        entries = [e for e in self._cache.values() if e.split == split]
        if not entries:
            logger.warning(
                f"get_split_for_analysis('{split}'): no entries.",
                extra={"stage": "dataset"},
            )
            return pd.DataFrame()

        records = [
            {
                "video_id":        e.video_id,
                "sign_label":      e.sign_label,
                "class_idx":       e.class_idx,
                "signer_id":       e.signer_id,
                "split":           e.split,
                "npy_path":        e.npy_path,
                "detected_frames": e.detected_frames,
                "sign_name":       self._label_map.get_name_safe(
                    e.class_idx, e.sign_label
                ),
            }
            for e in entries
        ]

        df = pd.DataFrame(records)
        df["_vid_sort"] = df["video_id"].apply(
            lambda v: int(v) if str(v).isdigit() else v
        )
        df = (
            df.sort_values(["class_idx", "_vid_sort"])
            .drop(columns=["_vid_sort"])
            .reset_index(drop=True)
        )
        return df

    def get_arrays_for_split(
        self,
        split: str,
        use_augmentation: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (X, y, signer_ids) numpy arrays for sklearn-style evaluation.

        Stable numeric sort order. NOT shuffled — use load_split() for training.

        Parameters
        ----------
        split : str — "train", "val", or "test"
        use_augmentation : bool — applies one aug round (clip_idx=0) if enabled

        Returns
        -------
        X          : (n_clips, seq_len, feature_dim) float32
        y          : (n_clips,) int32
        signer_ids : (n_clips,) int32
        """
        entries = sorted(
            [e for e in self._cache.values() if e.split == split],
            key=lambda e: _video_id_sort_key(e.video_id),
        )
        if not entries:
            raise RuntimeError(f"No cached clips for split '{split}'.")

        if use_augmentation and self._aug_enabled:
            arrays: List[np.ndarray] = []
            for e in entries:
                if e.arr_pre_select is not None:
                    aug = self._aug_pipeline(e.arr_pre_select, clip_idx=0)
                    arrays.append(aug[:, self._lm_slice].astype(np.float32))
                else:
                    arrays.append(e.arr_final.astype(np.float32))
            X = np.stack(arrays, axis=0)
        else:
            X = np.stack([e.arr_final for e in entries], axis=0).astype(np.float32)

        y          = np.array([e.class_idx for e in entries], dtype=np.int32)
        signer_ids = np.array([e.signer_id for e in entries], dtype=np.int32)
        return X, y, signer_ids

    # ══════════════════════════════════════════════════════════════════════
    # Metadata
    # ══════════════════════════════════════════════════════════════════════

    def get_split_statistics(self) -> Dict[str, Any]:
        """
        Complete JSON-serialisable statistics dictionary.

        Stored in run_manifest.json and logged to MLflow as
        dataset_statistics.json.
        """
        stats: Dict[str, Any] = {}

        for split_name in ("train", "val", "test"):
            entries = [e for e in self._cache.values() if e.split == split_name]
            if not entries:
                stats[split_name] = {"n_clips": 0, "n_classes": 0, "n_signers": 0}
                continue

            class_dist: Dict[int, int]  = {}
            signer_dist: Dict[int, int] = {}
            for e in entries:
                class_dist[e.class_idx]  = class_dist.get(e.class_idx, 0) + 1
                signer_dist[e.signer_id] = signer_dist.get(e.signer_id, 0) + 1

            singletons      = [c for c, cnt in class_dist.items() if cnt == 1]
            missing_classes = sorted(
                set(range(self._num_classes)) - set(class_dist.keys())
            )
            stats[split_name] = {
                "n_clips":             len(entries),
                "n_classes":           len(class_dist),
                "n_signers":           len(signer_dist),
                "n_singletons":        len(singletons),
                "singleton_classes":   singletons,
                "missing_classes":     missing_classes,
                "class_distribution":  {int(k): int(v)
                                        for k, v in sorted(class_dist.items())},
                "signer_distribution": {int(k): int(v)
                                        for k, v in sorted(signer_dist.items())},
            }

        train_class_counts: Dict[str, int] = {}
        for e in self._cache.values():
            if e.split == "train":
                name = self._label_map.get_name_safe(
                    e.class_idx, f"class_{e.class_idx}"
                )
                train_class_counts[name] = train_class_counts.get(name, 0) + 1

        stats["per_class_train_counts"] = {
            k: v for k, v in sorted(train_class_counts.items())
        }
        stats["pipeline_metadata"]     = self._pipeline.get_pipeline_metadata()
        stats["augmentation_metadata"] = self._aug_pipeline.get_metadata()
        stats["num_classes"]           = self._num_classes
        stats["label_map_version"]     = self._label_map.version
        stats["aug_cache_enabled"]     = self._aug_enabled
        stats["ram_cache_mb"]          = round(self._cache_ram_mb(), 2)
        return stats

    # ══════════════════════════════════════════════════════════════════════
    # Private helpers
    # ══════════════════════════════════════════════════════════════════════

    def _cache_ram_mb(self) -> float:
        """Return total RAM used by arr_final + arr_pre_select in MB."""
        total = sum(
            e.arr_final.nbytes
            + (0 if e.arr_pre_select is None else e.arr_pre_select.nbytes)
            for e in self._cache.values()
        )
        return total / (1024 ** 2)

    # ══════════════════════════════════════════════════════════════════════
    # Properties
    # ══════════════════════════════════════════════════════════════════════

    @property
    def n_train(self) -> int:
        return sum(1 for e in self._cache.values() if e.split == "train")

    @property
    def n_val(self) -> int:
        return sum(1 for e in self._cache.values() if e.split == "val")

    @property
    def n_test(self) -> int:
        return sum(1 for e in self._cache.values() if e.split == "test")

    @property
    def output_shape(self) -> Tuple[int, int]:
        """(seq_len, feature_dim) — shape of every Dataset sample."""
        return (self._seq_len, self._feature_dim)

    @property
    def label_map(self) -> LabelMap:
        return self._label_map

    @property
    def pipeline(self) -> FeaturePipeline:
        return self._pipeline

    # ══════════════════════════════════════════════════════════════════════
    # Dunder helpers
    # ══════════════════════════════════════════════════════════════════════

    def __len__(self) -> int:
        return len(self._cache)

    def __repr__(self) -> str:
        return (
            f"GestureDataset("
            f"clips={len(self._cache)}, "
            f"train={self.n_train}, val={self.n_val}, test={self.n_test}, "
            f"output_shape={self.output_shape}, "
            f"augmentation={'enabled' if self._aug_enabled else 'disabled'}, "
            f"landmark_config={self._lm_config!r}, "
            f"ram={self._cache_ram_mb():.1f} MB)"
        )