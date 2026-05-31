"""
src/data/downloader.py
=======================
WLASL dataset resolver and local inventory builder.

Design philosophy
-----------------
The WLASL dataset has already been downloaded locally. This module therefore
operates in "resolver" mode: it reads WLASL_v0.3.json, filters to the 35
selected signs, locates the corresponding video files on disk using a
multi-pattern search strategy, and produces a structured inventory manifest.

A ``download_missing()`` method is provided as a yt-dlp fallback for any clips
that cannot be located locally. In the standard workflow it is not exercised,
but it keeps the codebase complete and reusable for environments where the
dataset has not been pre-downloaded.

WLASL raw directory layouts handled
------------------------------------
The resolver tries all of the following patterns for each video:

    1. data/raw/<sign_name>/<video_id>.mp4          (sign-organised)
    2. data/raw/videos/<video_id>.mp4               (flat, videos subdir)
    3. data/raw/<video_id>.mp4                      (flat, root)
    4. data/raw/<sign_name>/<video_id>.*            (any extension)
    5. data/raw/**/<video_id>.mp4                   (recursive search — slow,
                                                     used only as last resort)

The first pattern that matches is used. Pattern 5 is only attempted if all
others fail and ``deep_search=True`` (default False — opt-in for large datasets).

Inventory schema
----------------
Produced at ``data/raw_inventory.json``. Every clip entry carries:

    video_id      : str   — WLASL identifier (zero-padded, e.g. "00648")
    sign_label    : str   — human-readable sign name ("book")
    class_idx     : int   — integer class index 0–34 from label_map_v1.json
    signer_id     : int   — WLASL signer identifier (required for splitting)
    frame_start   : int | null — bounding clip start frame from manifest
    frame_end     : int | null — bounding clip end frame from manifest
    fps           : float | null — frames per second from manifest (if present)
    url           : str   — original source URL (retained for download fallback)
    found         : bool  — True if video located on disk
    video_path    : str   — relative path from repo root (empty if not found)
    file_size_mb  : float — file size in MB (0.0 if not found)
    split_hint    : str   — WLASL's suggested split ("train"/"test"/"val")
                            recorded but NOT used — overridden by signer-aware split

Resumability
------------
If ``data/raw_inventory.json`` already exists and ``force=False`` (default),
``build_inventory()`` returns the cached inventory immediately without
re-scanning the filesystem. Pass ``force=True`` to force a full re-scan.

Usage
-----
    from src.data.downloader import WLASLResolver
    from src.utils import get_label_map, configure_logging

    configure_logging(log_dir="logs", run_name="stage1_ingestion")
    lm = get_label_map("artifacts/label_map_v1.json")

    resolver = WLASLResolver(
        manifest_path="data/raw/WLASL_v0.3.json",
        raw_dir="data/raw",
        label_map=lm,
    )
    inventory_df = resolver.build_inventory()
    resolver.save_inventory("data/raw_inventory.json")

    # Optional: attempt to download genuinely missing clips via yt-dlp
    resolver.download_missing(max_attempts=3)
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

try:
    from tqdm import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Repository root — resolved relative to this file
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Supported video file extensions (tried in order for pattern 4)
_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")

# Number of clips to process between progress log lines
_LOG_INTERVAL = 200


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ClipEntry:
    """
    Represents one video clip from the WLASL manifest.

    Attributes mirror the inventory JSON schema exactly. This dataclass is
    the canonical representation used throughout Stage 1 — do not add
    processing logic here; keep it as a pure data container.
    """
    video_id: str
    sign_label: str
    class_idx: int
    signer_id: int
    frame_start: Optional[int]
    frame_end: Optional[int]
    fps: Optional[float]
    url: str
    found: bool
    video_path: str          # relative to repo root; empty string if not found
    file_size_mb: float
    split_hint: str          # from WLASL manifest; NOT used for actual splitting

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InventoryMetadata:
    """Summary statistics written to the inventory JSON ``_metadata`` block."""
    created_utc: str
    manifest_version: str
    manifest_path: str
    raw_dir: str
    total_entries_in_manifest: int
    selected_signs: int
    total_clips_expected: int
    total_clips_found: int
    total_clips_missing: int
    unique_signers_found: int
    total_size_mb: float
    resolver_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# WLASLResolver
# ---------------------------------------------------------------------------

class WLASLResolver:
    """
    Resolves the WLASL dataset from local disk against WLASL_v0.3.json.

    Parameters
    ----------
    manifest_path : str | Path
        Path to ``WLASL_v0.3.json``. This file ships with the WLASL dataset.
    raw_dir : str | Path
        Root directory where WLASL videos are stored. Typically ``data/raw``.
    label_map : LabelMap
        The project's versioned label map. Used to filter to the 35 selected
        signs and to map sign names to class indices.
    deep_search : bool
        If True, fall back to recursive ``rglob("**/<video_id>.mp4")`` for
        clips not found by the four fast patterns. Slow on large directories.
        Default False.

    Raises
    ------
    FileNotFoundError
        If manifest_path does not exist.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        raw_dir: str | Path,
        label_map: Any,                    # src.utils.label_map.LabelMap
        deep_search: bool = False,
    ) -> None:
        self._manifest_path = Path(manifest_path).resolve()
        self._raw_dir = Path(raw_dir).resolve()
        self._label_map = label_map
        self._deep_search = deep_search

        # Internal state — populated by build_inventory()
        self._clips: list[ClipEntry] = []
        self._metadata: Optional[InventoryMetadata] = None
        self._raw_manifest: list[dict[str, Any]] = []

        if not self._manifest_path.exists():
            raise FileNotFoundError(
                f"WLASL manifest not found: {self._manifest_path}\n"
                "Download WLASL_v0.3.json from https://github.com/dxli94/WLASL "
                "and place it in data/raw/ before running this resolver."
            )

        if not self._raw_dir.exists():
            raise FileNotFoundError(
                f"Raw data directory not found: {self._raw_dir}\n"
                "Ensure the WLASL videos are stored under data/raw/."
            )

        logger.info(
            f"WLASLResolver initialised | "
            f"manifest={self._manifest_path} | "
            f"raw_dir={self._raw_dir} | "
            f"deep_search={deep_search}",
            extra={"stage": "ingestion"},
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build_inventory(
        self,
        force: bool = False,
        inventory_path: Optional[str | Path] = None,
    ) -> pd.DataFrame:
        """
        Scan the filesystem and build a clip inventory DataFrame.

        Reads WLASL_v0.3.json, filters to the 35 selected signs, and attempts
        to locate each clip on disk using a multi-pattern search strategy.

        Parameters
        ----------
        force : bool
            If False (default) and a cached inventory JSON already exists at
            ``inventory_path``, load and return it immediately without
            re-scanning. Pass True to force a full re-scan.
        inventory_path : str | Path | None
            Where to look for (and later save) the cached inventory JSON.
            Defaults to ``data/raw_inventory.json`` relative to the repo root.

        Returns
        -------
        pd.DataFrame
            One row per clip. Columns mirror the ClipEntry dataclass.

        Notes
        -----
        This method is idempotent and resumable. Re-running after a partial
        failure (e.g. interrupted filesystem scan) is safe.
        """
        inv_path = Path(inventory_path) if inventory_path else (
            _REPO_ROOT / "data" / "raw_inventory.json"
        )

        # Resumability: load cached inventory if it exists and force=False
        if not force and inv_path.exists():
            logger.info(
                f"Loading cached inventory from {inv_path} | "
                "pass force=True to re-scan",
                extra={"stage": "ingestion"},
            )
            return self._load_cached_inventory(inv_path)

        logger.info(
            "Starting WLASL inventory build | "
            f"manifest={self._manifest_path.name} | "
            f"selected_signs={self._label_map.num_classes}",
            extra={"stage": "ingestion"},
        )

        # Step 1: Load and parse the manifest
        self._raw_manifest = self._load_manifest()

        # Step 2: Filter to selected signs only
        filtered = self._filter_to_selected_signs(self._raw_manifest)
        total_expected = sum(len(v.get("instances", [])) for v in filtered)

        logger.info(
            f"Manifest filtered | "
            f"total_entries={len(self._raw_manifest)} | "
            f"selected_sign_entries={len(filtered)} | "
            f"total_clips_expected={total_expected}",
            extra={"stage": "ingestion"},
        )

        # Step 3: Resolve each clip against the filesystem
        self._clips = self._resolve_clips(filtered)

        # Step 4: Compute summary statistics
        found_clips = [c for c in self._clips if c.found]
        missing_clips = [c for c in self._clips if not c.found]
        unique_signers = len({c.signer_id for c in found_clips})
        total_size_mb = sum(c.file_size_mb for c in found_clips)

        self._metadata = InventoryMetadata(
            created_utc=datetime.now(timezone.utc).isoformat(),
            manifest_version="WLASL_v0.3",
            manifest_path=str(self._manifest_path),
            raw_dir=str(self._raw_dir),
            total_entries_in_manifest=len(self._raw_manifest),
            selected_signs=self._label_map.num_classes,
            total_clips_expected=len(self._clips),
            total_clips_found=len(found_clips),
            total_clips_missing=len(missing_clips),
            unique_signers_found=unique_signers,
            total_size_mb=round(total_size_mb, 2),
        )

        logger.info(
            f"Inventory built | "
            f"found={len(found_clips)} | "
            f"missing={len(missing_clips)} | "
            f"signers={unique_signers} | "
            f"size={total_size_mb:.1f} MB",
            extra={"stage": "ingestion"},
        )

        if missing_clips:
            logger.warning(
                f"{len(missing_clips)} clips could not be located on disk. "
                "Call download_missing() to attempt yt-dlp retrieval, or "
                "verify the raw_dir path is correct.",
                extra={"stage": "ingestion"},
            )
            # Log per-sign missing summary at DEBUG level
            missing_by_sign: dict[str, int] = {}
            for c in missing_clips:
                missing_by_sign[c.sign_label] = missing_by_sign.get(c.sign_label, 0) + 1
            for sign, count in sorted(missing_by_sign.items(), key=lambda x: -x[1]):
                logger.debug(
                    f"  Missing: {sign} — {count} clips",
                    extra={"stage": "ingestion", "sign": sign},
                )

        return self._to_dataframe()

    def save_inventory(
        self,
        output_path: str | Path,
        include_per_sign_summary: bool = True,
    ) -> Path:
        """
        Write the inventory to a JSON file.

        Parameters
        ----------
        output_path : str | Path
            Destination path. Parent directories are created if needed.
        include_per_sign_summary : bool
            If True, include a per-sign clip count summary in the JSON.

        Returns
        -------
        Path
            Resolved path to the written file.

        Raises
        ------
        RuntimeError
            If build_inventory() has not been called yet.
        """
        if not self._clips:
            raise RuntimeError(
                "No inventory to save. Call build_inventory() first."
            )

        path = Path(output_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        # Per-sign summary
        sign_summary: dict[str, dict[str, int]] = {}
        if include_per_sign_summary:
            for clip in self._clips:
                if clip.sign_label not in sign_summary:
                    sign_summary[clip.sign_label] = {
                        "expected": 0, "found": 0, "missing": 0,
                    }
                sign_summary[clip.sign_label]["expected"] += 1
                if clip.found:
                    sign_summary[clip.sign_label]["found"] += 1
                else:
                    sign_summary[clip.sign_label]["missing"] += 1

        missing_ids = [c.video_id for c in self._clips if not c.found]

        payload = {
            "_metadata": self._metadata.to_dict() if self._metadata else {},
            "sign_summary": sign_summary,
            "missing_clip_ids": missing_ids,
            "clips": [c.to_dict() for c in self._clips],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

        logger.info(
            f"Inventory saved: {path} | "
            f"clips={len(self._clips)} | "
            f"size={path.stat().st_size / 1024:.1f} KB",
            extra={"stage": "ingestion"},
        )
        return path

    def get_inventory_df(self) -> pd.DataFrame:
        """
        Return the inventory as a pandas DataFrame.

        Raises
        ------
        RuntimeError
            If build_inventory() has not been called yet.
        """
        if not self._clips:
            raise RuntimeError(
                "Inventory is empty. Call build_inventory() first."
            )
        return self._to_dataframe()

    def get_missing_clips(self) -> list[ClipEntry]:
        """Return clips that could not be located on disk."""
        return [c for c in self._clips if not c.found]

    def download_missing(
        self,
        max_attempts: int = 3,
        sleep_between_attempts: float = 2.0,
        dry_run: bool = False,
    ) -> dict[str, bool]:
        """
        Attempt to download missing clips via yt-dlp.

        This is a fallback for clips that could not be located locally.
        In the standard workflow (pre-downloaded dataset), this method
        will not be called. It is retained for completeness and reuse.

        yt-dlp must be installed: ``pip install yt-dlp``

        Parameters
        ----------
        max_attempts : int
            Number of retry attempts per clip before marking as permanently
            failed. Default 3.
        sleep_between_attempts : float
            Seconds to sleep between retry attempts. Default 2.0.
        dry_run : bool
            If True, log what would be downloaded without actually downloading.
            Useful for verifying URL availability. Default False.

        Returns
        -------
        dict[str, bool]
            Mapping of video_id → True (downloaded successfully) / False (failed).
        """
        missing = self.get_missing_clips()
        if not missing:
            logger.info(
                "No missing clips — download_missing() is a no-op.",
                extra={"stage": "ingestion"},
            )
            return {}

        logger.info(
            f"Attempting to download {len(missing)} missing clips | "
            f"dry_run={dry_run} | max_attempts={max_attempts}",
            extra={"stage": "ingestion"},
        )

        # Verify yt-dlp is available
        if not dry_run:
            try:
                result = subprocess.run(
                    ["yt-dlp", "--version"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode != 0:
                    raise FileNotFoundError
                logger.debug(f"yt-dlp version: {result.stdout.strip()}")
            except (FileNotFoundError, subprocess.TimeoutExpired):
                logger.error(
                    "yt-dlp is not installed or not on PATH. "
                    "Install with: pip install yt-dlp",
                    extra={"stage": "ingestion"},
                )
                return {c.video_id: False for c in missing}

        results: dict[str, bool] = {}
        iterator = tqdm(missing, desc="Downloading missing clips") if _TQDM_AVAILABLE else missing

        for clip in iterator:
            if not clip.url:
                logger.warning(
                    f"No URL available for video_id={clip.video_id} | "
                    f"sign={clip.sign_label} — skipping.",
                    extra={"stage": "ingestion", "video_id": clip.video_id},
                )
                results[clip.video_id] = False
                continue

            # Determine output path
            output_dir = self._raw_dir / clip.sign_label
            output_dir.mkdir(parents=True, exist_ok=True)
            output_template = str(output_dir / f"{clip.video_id}.%(ext)s")

            if dry_run:
                logger.info(
                    f"[DRY RUN] Would download: {clip.video_id} "
                    f"({clip.sign_label}) from {clip.url[:60]}...",
                    extra={"stage": "ingestion", "video_id": clip.video_id},
                )
                results[clip.video_id] = False
                continue

            success = False
            for attempt in range(1, max_attempts + 1):
                try:
                    cmd = [
                        "yt-dlp",
                        "--no-playlist",
                        "--format", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                        "--output", output_template,
                        "--quiet",
                        "--no-warnings",
                        clip.url,
                    ]
                    proc = subprocess.run(
                        cmd, capture_output=True, text=True, timeout=120,
                    )
                    if proc.returncode == 0:
                        # Verify the file was actually created
                        found_path = self._locate_video(clip.video_id, clip.sign_label)
                        if found_path:
                            # Update the clip entry in-place
                            clip.found = True
                            clip.video_path = str(
                                found_path.relative_to(_REPO_ROOT)
                            )
                            clip.file_size_mb = found_path.stat().st_size / 1_048_576
                            logger.info(
                                f"Downloaded: {clip.video_id} ({clip.sign_label}) "
                                f"→ {clip.video_path}",
                                extra={"stage": "ingestion", "video_id": clip.video_id},
                            )
                            success = True
                            break
                        else:
                            logger.warning(
                                f"yt-dlp reported success but file not found: "
                                f"{clip.video_id} (attempt {attempt}/{max_attempts})",
                                extra={"stage": "ingestion"},
                            )
                    else:
                        logger.warning(
                            f"yt-dlp failed for {clip.video_id} "
                            f"(attempt {attempt}/{max_attempts}): "
                            f"{proc.stderr[:200]}",
                            extra={"stage": "ingestion"},
                        )
                except subprocess.TimeoutExpired:
                    logger.warning(
                        f"yt-dlp timed out for {clip.video_id} "
                        f"(attempt {attempt}/{max_attempts})",
                        extra={"stage": "ingestion"},
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        f"Unexpected error downloading {clip.video_id}: {exc}",
                        extra={"stage": "ingestion"},
                    )

                if not success and attempt < max_attempts:
                    time.sleep(sleep_between_attempts)

            results[clip.video_id] = success
            if not success:
                logger.error(
                    f"Permanently failed to download: {clip.video_id} ({clip.sign_label})",
                    extra={"stage": "ingestion", "video_id": clip.video_id},
                )

        downloaded = sum(1 for v in results.values() if v)
        logger.info(
            f"Download complete | "
            f"succeeded={downloaded} | "
            f"failed={len(results) - downloaded} | "
            f"total_attempted={len(results)}",
            extra={"stage": "ingestion"},
        )
        return results

    # ------------------------------------------------------------------
    # Private helpers — manifest parsing
    # ------------------------------------------------------------------

    def _load_manifest(self) -> list[dict[str, Any]]:
        """
        Load and parse WLASL_v0.3.json.

        The WLASL manifest is a JSON array where each element represents one
        gloss (sign) and contains an ``instances`` array of video clips.

        Returns
        -------
        list[dict]
            The raw manifest as a list of gloss-level entries.

        Raises
        ------
        ValueError
            If the JSON does not have the expected top-level structure.
        """
        logger.info(
            f"Loading WLASL manifest: {self._manifest_path}",
            extra={"stage": "ingestion"},
        )

        with open(self._manifest_path, encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(
                f"Expected WLASL manifest to be a JSON array, "
                f"got {type(data).__name__}. "
                "Ensure you are using WLASL_v0.3.json."
            )

        logger.info(
            f"Manifest loaded | {len(data)} glosses | "
            f"size={self._manifest_path.stat().st_size / 1024:.1f} KB",
            extra={"stage": "ingestion"},
        )
        return data

    def _filter_to_selected_signs(
        self,
        manifest: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Keep only glosses whose name matches a sign in the label map.

        The matching is case-insensitive and strips leading/trailing whitespace.
        Logs any selected signs that have NO entry in the manifest (possible
        if a sign was renamed across WLASL versions).

        Parameters
        ----------
        manifest : list[dict]
            Full parsed WLASL manifest.

        Returns
        -------
        list[dict]
            Filtered list of gloss-level entries for the 35 selected signs.
        """
        selected_names = {name.lower().strip() for name in self._label_map.sign_names}
        filtered = []
        found_names: set[str] = set()

        for entry in manifest:
            gloss = entry.get("gloss", "").lower().strip()
            if gloss in selected_names:
                filtered.append(entry)
                found_names.add(gloss)

        # Warn about any selected signs missing from the manifest
        not_in_manifest = selected_names - found_names
        if not_in_manifest:
            logger.warning(
                f"{len(not_in_manifest)} selected sign(s) not found in manifest: "
                f"{sorted(not_in_manifest)}. "
                "Check for naming discrepancies between label_map_v1.json and "
                "the WLASL manifest gloss field.",
                extra={"stage": "ingestion"},
            )

        logger.info(
            f"Sign filter applied | "
            f"selected={len(selected_names)} | "
            f"found_in_manifest={len(found_names)} | "
            f"not_in_manifest={len(not_in_manifest)}",
            extra={"stage": "ingestion"},
        )
        return filtered

    # ------------------------------------------------------------------
    # Private helpers — clip resolution
    # ------------------------------------------------------------------

    def _resolve_clips(
        self,
        filtered_manifest: list[dict[str, Any]],
    ) -> list[ClipEntry]:
        """
        Iterate over all instances in the filtered manifest and attempt to
        locate each video file on disk.

        Parameters
        ----------
        filtered_manifest : list[dict]
            Filtered manifest entries (one per selected gloss).

        Returns
        -------
        list[ClipEntry]
            One entry per clip, with found/video_path populated.
        """
        clips: list[ClipEntry] = []
        processed = 0
        start_time = time.time()

        # Flatten: iterate gloss → instances
        all_instances: list[tuple[str, dict[str, Any]]] = []
        for entry in filtered_manifest:
            gloss = entry.get("gloss", "").lower().strip()
            for instance in entry.get("instances", []):
                all_instances.append((gloss, instance))

        total = len(all_instances)
        iterator = (
            tqdm(all_instances, desc="Resolving clips", unit="clip")
            if _TQDM_AVAILABLE
            else all_instances
        )

        for gloss, instance in iterator:
            clip = self._resolve_single_clip(gloss, instance)
            clips.append(clip)
            processed += 1

            if processed % _LOG_INTERVAL == 0:
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                eta_sec = (total - processed) / rate if rate > 0 else 0
                logger.info(
                    f"Resolving clips | "
                    f"{processed}/{total} | "
                    f"rate={rate:.0f} clips/s | "
                    f"ETA={eta_sec:.0f}s",
                    extra={"stage": "ingestion"},
                )

        found_count = sum(1 for c in clips if c.found)
        logger.info(
            f"Clip resolution complete | "
            f"total={len(clips)} | "
            f"found={found_count} | "
            f"missing={len(clips) - found_count} | "
            f"elapsed={time.time() - start_time:.1f}s",
            extra={"stage": "ingestion"},
        )
        return clips

    def _resolve_single_clip(
        self,
        gloss: str,
        instance: dict[str, Any],
    ) -> ClipEntry:
        """
        Resolve a single WLASL instance dict to a ClipEntry.

        Extracts all fields from the instance, attempts to locate the video
        file, and returns a populated ClipEntry.

        Parameters
        ----------
        gloss : str
            The lowercase sign name (e.g. "book").
        instance : dict
            A single instance dict from the WLASL manifest ``instances`` array.

        Returns
        -------
        ClipEntry
            Populated entry. If the video cannot be located, ``found=False``
            and ``video_path=""`` are set.
        """
        video_id = str(instance.get("video_id", "")).strip()
        signer_id = instance.get("signer_id", -1)
        frame_start = instance.get("frame_start")
        frame_end = instance.get("frame_end")
        fps = instance.get("fps")
        url = instance.get("url", "")
        split_hint = instance.get("split", "unknown")

        # Handle missing or invalid signer_id
        if signer_id is None or not isinstance(signer_id, (int, float)):
            logger.warning(
                f"Missing or invalid signer_id for video_id={video_id} "
                f"sign={gloss}. Assigning signer_id=-1 (unknown).",
                extra={"stage": "ingestion", "video_id": video_id, "sign": gloss},
            )
            signer_id = -1

        # Look up class index from label map
        class_idx = self._label_map.get_index_safe(gloss, default=-1)
        if class_idx == -1:
            logger.error(
                f"Sign '{gloss}' not found in label map. "
                f"This should not happen after filtering.",
                extra={"stage": "ingestion", "sign": gloss},
            )

        # Attempt to locate the video file
        found_path = self._locate_video(video_id, gloss)

        if found_path:
            rel_path = str(found_path.relative_to(_REPO_ROOT))
            file_size_mb = found_path.stat().st_size / 1_048_576
            found = True
        else:
            rel_path = ""
            file_size_mb = 0.0
            found = False

        return ClipEntry(
            video_id=video_id,
            sign_label=gloss,
            class_idx=class_idx,
            signer_id=int(signer_id),
            frame_start=int(frame_start) if frame_start is not None else None,
            frame_end=int(frame_end) if frame_end is not None else None,
            fps=float(fps) if fps is not None else None,
            url=url,
            found=found,
            video_path=rel_path,
            file_size_mb=round(file_size_mb, 4),
            split_hint=split_hint,
        )

    def _locate_video(self, video_id: str, sign_label: str) -> Optional[Path]:
        """
        Attempt to locate a video file using multiple search patterns.

        Tries patterns in order from fastest to slowest:
            1. <raw_dir>/<sign_label>/<video_id>.mp4
            2. <raw_dir>/videos/<video_id>.mp4
            3. <raw_dir>/<video_id>.mp4
            4. <raw_dir>/<sign_label>/<video_id>.<any supported extension>
            5. <raw_dir>/**/<video_id>.mp4  (recursive — only if deep_search=True)

        Parameters
        ----------
        video_id : str
            The WLASL video identifier (e.g. "00648").
        sign_label : str
            The lowercase sign name, used in pattern 1 and 4.

        Returns
        -------
        Path | None
            The first matching path, or None if no file is found.
        """
        # Pattern 1: sign-organised directory
        p1 = self._raw_dir / sign_label / f"{video_id}.mp4"
        if p1.exists():
            return p1

        # Pattern 2: flat videos/ subdirectory
        p2 = self._raw_dir / "videos" / f"{video_id}.mp4"
        if p2.exists():
            return p2

        # Pattern 3: flat root directory
        p3 = self._raw_dir / f"{video_id}.mp4"
        if p3.exists():
            return p3

        # Pattern 4: any supported extension in sign-organised directory
        sign_dir = self._raw_dir / sign_label
        if sign_dir.exists():
            for ext in _VIDEO_EXTENSIONS:
                p4 = sign_dir / f"{video_id}{ext}"
                if p4.exists():
                    return p4

        # Pattern 5: recursive search (expensive — opt-in only)
        if self._deep_search:
            matches = list(self._raw_dir.rglob(f"{video_id}.mp4"))
            if matches:
                return matches[0]
            for ext in _VIDEO_EXTENSIONS[1:]:
                matches = list(self._raw_dir.rglob(f"{video_id}{ext}"))
                if matches:
                    return matches[0]

        return None

    # ------------------------------------------------------------------
    # Private helpers — output
    # ------------------------------------------------------------------

    def _to_dataframe(self) -> pd.DataFrame:
        """Convert the internal clips list to a pandas DataFrame."""
        if not self._clips:
            return pd.DataFrame()

        df = pd.DataFrame([c.to_dict() for c in self._clips])

        # Enforce dtypes for downstream safety
        df["class_idx"] = df["class_idx"].astype(int)
        df["signer_id"] = df["signer_id"].astype(int)
        df["found"] = df["found"].astype(bool)
        df["file_size_mb"] = df["file_size_mb"].astype(float)

        return df

    def _load_cached_inventory(self, path: Path) -> pd.DataFrame:
        """
        Load a previously saved inventory JSON and reconstruct internal state.

        This enables resumability: if the inventory JSON exists, parsing
        and filesystem scanning are skipped entirely.

        Parameters
        ----------
        path : Path
            Path to the cached inventory JSON.

        Returns
        -------
        pd.DataFrame
            DataFrame reconstructed from the cached JSON.
        """
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        raw_clips = data.get("clips", [])
        self._clips = []
        for raw in raw_clips:
            self._clips.append(ClipEntry(
                video_id=raw.get("video_id", ""),
                sign_label=raw.get("sign_label", ""),
                class_idx=raw.get("class_idx", -1),
                signer_id=raw.get("signer_id", -1),
                frame_start=raw.get("frame_start"),
                frame_end=raw.get("frame_end"),
                fps=raw.get("fps"),
                url=raw.get("url", ""),
                found=raw.get("found", False),
                video_path=raw.get("video_path", ""),
                file_size_mb=raw.get("file_size_mb", 0.0),
                split_hint=raw.get("split_hint", "unknown"),
            ))

        meta = data.get("_metadata", {})
        self._metadata = InventoryMetadata(
            created_utc=meta.get("created_utc", ""),
            manifest_version=meta.get("manifest_version", ""),
            manifest_path=meta.get("manifest_path", ""),
            raw_dir=meta.get("raw_dir", ""),
            total_entries_in_manifest=meta.get("total_entries_in_manifest", 0),
            selected_signs=meta.get("selected_signs", 0),
            total_clips_expected=meta.get("total_clips_expected", 0),
            total_clips_found=meta.get("total_clips_found", 0),
            total_clips_missing=meta.get("total_clips_missing", 0),
            unique_signers_found=meta.get("unique_signers_found", 0),
            total_size_mb=meta.get("total_size_mb", 0.0),
        )

        found = sum(1 for c in self._clips if c.found)
        logger.info(
            f"Cached inventory loaded: {path} | "
            f"clips={len(self._clips)} | found={found}",
            extra={"stage": "ingestion"},
        )
        return self._to_dataframe()