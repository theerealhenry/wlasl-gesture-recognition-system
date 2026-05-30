"""
src/utils/label_map.py
=======================
Versioned, bidirectional label map for the WLASL gesture recognition pipeline.

Design principles:
  - The label map JSON stores ONLY the forward mapping (index → name).
    The inverse mapping (name → index) is generated at runtime to enforce a
    single source of truth and eliminate synchronisation errors.
  - Validation is strict: num_classes in metadata must equal the actual number
    of entries in the "classes" block. Any mismatch raises immediately.
  - All access is via the LabelMap class — never index dicts directly in pipeline code.
  - A module-level singleton (get_label_map) avoids repeated disk reads during
    multi-call inference loops.

Usage:
    from src.utils.label_map import LabelMap, get_label_map

    # Load once
    lm = get_label_map("artifacts/label_map_v1.json")

    # Forward lookups
    lm[0]                          # "before"
    lm.get_name(0)                 # "before"
    lm.get_name_safe(99, "UNK")    # "UNK" — no KeyError

    # Reverse lookups
    lm.get_index("book")           # 4
    lm.get_index_safe("xyz", -1)   # -1 — no KeyError

    # Sign metadata
    lm.get_sign_property(14, "handedness")    # "one"
    lm.get_sign_property(14, "confusable_with")  # ["eat", "candy"]

    # Bulk validation
    lm.validate_predictions([0, 14, 34])   # True — all valid indices
    lm.validate_predictions([0, 99])       # False — 99 is out of range
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class LabelMap:
    """
    Bidirectional label map loaded from a versioned JSON file.

    The JSON file stores the forward mapping only ("classes" block).
    The inverse mapping is built at construction time and kept in sync
    automatically — there is only one source of truth.

    Parameters
    ----------
    path : str | Path
        Path to the label map JSON file (e.g. artifacts/label_map_v1.json).

    Raises
    ------
    FileNotFoundError
        If the JSON file does not exist at the given path.
    ValueError
        If the JSON is structurally invalid, if indices are not consecutive
        starting from 0, or if metadata.num_classes disagrees with the
        actual number of entries in the "classes" block.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._classes: dict[int, str] = {}        # index → name (forward)
        self._inverse: dict[str, int] = {}        # name → index (derived)
        self._metadata: dict[str, Any] = {}
        self._sign_properties: dict[int, dict[str, Any]] = {}
        self._version: str = "unknown"

        self._load()

    # ------------------------------------------------------------------
    # Private loading and validation
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load, validate, and build the internal maps from disk."""
        if not self._path.exists():
            raise FileNotFoundError(
                f"Label map file not found: {self._path.resolve()}\n"
                "Run the pipeline setup step to create artifacts/label_map_v1.json."
            )

        with open(self._path, encoding="utf-8") as f:
            raw: dict[str, Any] = json.load(f)

        self._metadata = raw.get("_metadata", {})
        self._version = self._metadata.get("label_map_version", "unknown")

        # ------------------------------------------------------------------
        # Validate and build forward map
        # ------------------------------------------------------------------
        classes_raw: dict[str, str] = raw.get("classes", {})
        if not classes_raw:
            raise ValueError(
                f"Label map at {self._path} has an empty or missing 'classes' block."
            )

        # Convert string keys to int (JSON forces string keys)
        try:
            self._classes = {int(k): v for k, v in classes_raw.items()}
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Label map 'classes' block contains non-integer keys: {exc}"
            ) from exc

        # Validate consecutive indices starting at 0
        expected_indices = set(range(len(self._classes)))
        actual_indices = set(self._classes.keys())
        if expected_indices != actual_indices:
            missing = expected_indices - actual_indices
            extra = actual_indices - expected_indices
            raise ValueError(
                f"Label map indices are not consecutive starting from 0. "
                f"Missing: {sorted(missing)}, Extra: {sorted(extra)}"
            )

        # Validate num_classes metadata matches actual class count
        declared_num_classes = self._metadata.get("num_classes")
        if declared_num_classes is not None:
            if len(self._classes) != declared_num_classes:
                raise ValueError(
                    f"Label map integrity error: metadata.num_classes={declared_num_classes} "
                    f"but 'classes' block contains {len(self._classes)} entries. "
                    f"Update the JSON file to resolve this mismatch."
                )

        # ------------------------------------------------------------------
        # Warn about unexpected keys (future-proofing)
        # ------------------------------------------------------------------
        known_top_level_keys = {"_metadata", "classes", "sign_properties"}
        unknown_keys = set(raw.keys()) - known_top_level_keys
        if unknown_keys:
            logger.warning(
                f"Label map contains unexpected top-level keys: {unknown_keys}. "
                f"These will be ignored. (Note: 'inverse_classes' is no longer stored "
                f"in the JSON — it is generated at runtime.)"
            )

        # ------------------------------------------------------------------
        # Build inverse map at runtime — single source of truth
        # ------------------------------------------------------------------
        self._inverse = {name: idx for idx, name in self._classes.items()}

        if len(self._inverse) != len(self._classes):
            # Duplicate sign names detected
            name_counts: dict[str, int] = {}
            for name in self._classes.values():
                name_counts[name] = name_counts.get(name, 0) + 1
            duplicates = {n: c for n, c in name_counts.items() if c > 1}
            raise ValueError(
                f"Label map contains duplicate sign names: {duplicates}. "
                f"All sign names must be unique."
            )

        # ------------------------------------------------------------------
        # Load sign properties (optional block)
        # ------------------------------------------------------------------
        sign_props_raw: dict[str, Any] = raw.get("sign_properties", {})
        self._sign_properties = {
            int(k): v for k, v in sign_props_raw.items()
            if k.isdigit()
        }

        logger.info(
            f"Loaded label map v{self._version} | "
            f"path={self._path} | "
            f"classes={len(self._classes)} | "
            f"sign_properties={len(self._sign_properties)}"
        )

    # ------------------------------------------------------------------
    # Core access — forward lookups
    # ------------------------------------------------------------------

    def __getitem__(self, index: int) -> str:
        """
        Return the sign name for a class index.

        Raises KeyError if the index is not in the map.

        Example:
            lm[0]   # "before"
            lm[14]  # "drink"
        """
        try:
            return self._classes[index]
        except KeyError:
            raise KeyError(
                f"Class index {index} not found in label map. "
                f"Valid range: 0–{len(self._classes) - 1}."
            )

    def get_name(self, index: int) -> str:
        """Alias for __getitem__. Raises KeyError if not found."""
        return self[index]

    def get_name_safe(self, index: int, default: str = "UNKNOWN") -> str:
        """
        Return the sign name for a class index, or `default` if not found.
        Never raises. Safe for use in inference paths where out-of-range
        indices should be handled gracefully.

        Example:
            lm.get_name_safe(99, "UNK")  # "UNK"
        """
        return self._classes.get(index, default)

    # ------------------------------------------------------------------
    # Core access — reverse lookups
    # ------------------------------------------------------------------

    def get_index(self, name: str) -> int:
        """
        Return the class index for a sign name.

        Raises KeyError if the name is not in the map.

        Example:
            lm.get_index("book")  # 4
        """
        try:
            return self._inverse[name]
        except KeyError:
            raise KeyError(
                f"Sign name '{name}' not found in label map. "
                f"Known signs: {sorted(self._inverse.keys())}"
            )

    def get_index_safe(self, name: str, default: int = -1) -> int:
        """
        Return the class index for a sign name, or `default` if not found.
        Never raises.

        Example:
            lm.get_index_safe("xyz", -1)  # -1
        """
        return self._inverse.get(name, default)

    # ------------------------------------------------------------------
    # Sign property access
    # ------------------------------------------------------------------

    def get_sign_property(self, index: int, property_name: str) -> Any:
        """
        Return a named property for a sign by its class index.

        Available properties (defined in sign_properties block of the JSON):
            handedness      : "one" | "two"
            motion_type     : "static" | "dynamic"
            body_reference  : bool
            difficulty      : "easy" | "medium" | "hard"
            confusable_with : list[str]  — sign names this sign is often confused with

        Parameters
        ----------
        index : int
            Class index of the sign.
        property_name : str
            Name of the property to retrieve.

        Returns
        -------
        Any
            The property value, or None if the index or property is not found.

        Example:
            lm.get_sign_property(14, "handedness")       # "one"
            lm.get_sign_property(14, "confusable_with")  # ["eat", "candy"]
        """
        props = self._sign_properties.get(index, {})
        return props.get(property_name)

    def get_all_properties(self, index: int) -> dict[str, Any]:
        """Return all sign properties for a given class index as a dict."""
        return dict(self._sign_properties.get(index, {}))

    def get_confusable_signs(self, index: int) -> list[str]:
        """
        Return the list of sign names that are commonly confused with this sign.

        Example:
            lm.get_confusable_signs(14)  # ["eat", "candy"]
        """
        return self.get_sign_property(index, "confusable_with") or []

    # ------------------------------------------------------------------
    # Bulk validation
    # ------------------------------------------------------------------

    def validate_predictions(self, indices: list[int]) -> bool:
        """
        Return True if all indices in the list are valid class indices.
        Return False if any index is out of range.

        Example:
            lm.validate_predictions([0, 14, 34])  # True
            lm.validate_predictions([0, 99])       # False
        """
        valid_range = set(self._classes.keys())
        invalid = [i for i in indices if i not in valid_range]
        if invalid:
            logger.warning(
                f"validate_predictions: {len(invalid)} invalid index/indices found: {invalid}"
            )
            return False
        return True

    # ------------------------------------------------------------------
    # Introspection and serialisation
    # ------------------------------------------------------------------

    @property
    def num_classes(self) -> int:
        """Total number of sign classes."""
        return len(self._classes)

    @property
    def version(self) -> str:
        """Label map version string from metadata."""
        return self._version

    @property
    def sign_names(self) -> list[str]:
        """All sign names in class-index order."""
        return [self._classes[i] for i in sorted(self._classes)]

    @property
    def class_indices(self) -> list[int]:
        """All class indices in sorted order."""
        return sorted(self._classes.keys())

    def to_dict(self) -> dict[int, str]:
        """Return a copy of the forward (index → name) mapping."""
        return dict(self._classes)

    def to_inverse_dict(self) -> dict[str, int]:
        """Return a copy of the inverse (name → index) mapping."""
        return dict(self._inverse)

    def __len__(self) -> int:
        return len(self._classes)

    def __contains__(self, item: object) -> bool:
        """Support `0 in lm` (int index) and `"book" in lm` (sign name)."""
        if isinstance(item, int):
            return item in self._classes
        if isinstance(item, str):
            return item in self._inverse
        return False

    def __repr__(self) -> str:
        return (
            f"LabelMap(version={self._version!r}, "
            f"num_classes={self.num_classes}, "
            f"path={str(self._path)!r})"
        )

    # ------------------------------------------------------------------
    # Classmethods — alternative constructors
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> "LabelMap":
        """
        Load a LabelMap from a JSON file.

        Alias for the constructor — provided for explicitness at call sites.

        Example:
            lm = LabelMap.load("artifacts/label_map_v1.json")
        """
        return cls(path)

    @classmethod
    def from_names(cls, names: list[str]) -> "LabelMap":
        """
        Construct a LabelMap from a plain list of sign names.

        Indices are assigned in list order (0, 1, 2, ...).
        Useful for constructing temporary label maps in tests or notebooks
        without needing a JSON file.

        Example:
            lm = LabelMap.from_names(["book", "drink", "computer"])
            lm[0]  # "book"
            lm.get_index("drink")  # 1
        """
        import tempfile
        import json as _json

        classes_dict = {str(i): name for i, name in enumerate(names)}
        metadata = {
            "format_version": "1.1",
            "description": "In-memory label map constructed from name list",
            "num_classes": len(names),
            "label_map_version": "in-memory",
        }
        data = {"_metadata": metadata, "classes": classes_dict}

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            _json.dump(data, f)
            temp_path = f.name

        instance = cls(temp_path)
        Path(temp_path).unlink(missing_ok=True)
        return instance


# =============================================================================
# Module-level singleton with thread safety
# =============================================================================

_singleton_lock = threading.Lock()
_label_map_cache: dict[str, LabelMap] = {}


def get_label_map(path: str | Path = "artifacts/label_map_v1.json") -> LabelMap:
    """
    Return a cached LabelMap instance for the given path.

    Thread-safe. Loads from disk only on the first call for a given path;
    subsequent calls return the cached instance immediately. This is important
    for inference loops that call the predictor thousands of times per session.

    Parameters
    ----------
    path : str | Path
        Path to the label map JSON file. Defaults to the v1 label map.

    Returns
    -------
    LabelMap
        A validated, fully loaded LabelMap instance.

    Example:
        lm = get_label_map()                          # default v1 map
        lm = get_label_map("artifacts/label_map_v2.json")  # explicit path
    """
    key = str(Path(path).resolve())

    # Fast path — no lock needed for read if already cached
    if key in _label_map_cache:
        return _label_map_cache[key]

    with _singleton_lock:
        # Double-checked locking: re-check after acquiring the lock
        if key not in _label_map_cache:
            logger.debug(f"Cache miss — loading label map from disk: {path}")
            _label_map_cache[key] = LabelMap(path)

    return _label_map_cache[key]


def invalidate_label_map_cache(path: Optional[str | Path] = None) -> None:
    """
    Remove a cached LabelMap entry.

    Primarily used in tests to force a fresh load from disk.

    Parameters
    ----------
    path : str | Path | None
        Path whose cache entry to remove. If None, clears the entire cache.
    """
    with _singleton_lock:
        if path is None:
            _label_map_cache.clear()
            logger.debug("Label map cache fully cleared.")
        else:
            key = str(Path(path).resolve())
            removed = _label_map_cache.pop(key, None)
            if removed:
                logger.debug(f"Evicted label map cache entry: {key}")