"""
src/utils/logger.py
====================
Production-grade structured logging for the WLASL gesture recognition pipeline.

Design principles:
  - One logger per module, named after the module (__name__).
  - All loggers share a single root configuration set up by configure_logging().
  - Every run writes to both the console (coloured) and a timestamped rotating log file.
  - Structured extra fields (video_id, frame_idx, epoch, etc.) are supported via
    StructuredAdapter wrappers returned by get_logger().
  - Active MLflow run IDs are injected into every log line automatically.
  - Thread-safe: configure_logging() uses a lock; multiple threads calling it
    simultaneously will each get the correct log file path back.

Usage (in any module):
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Processing started", extra={"video_id": "v001", "sign": "book"})
    logger.warning("MediaPipe missed hand", extra={"video_id": "v001", "frame": 12})

The logger is idempotent: calling get_logger() multiple times with the same name
returns the same underlying logger without duplicating handlers.

configure_logging() is likewise idempotent: a second call returns the path of the
already-open log file without creating a new filename or duplicate handlers.
"""

import logging
import logging.handlers
import sys
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, MutableMapping, Optional

# ---------------------------------------------------------------------------
# ANSI colour codes — disabled automatically on Windows / non-TTY
# ---------------------------------------------------------------------------
_COLOURS = {
    "DEBUG":    "\033[36m",   # Cyan
    "INFO":     "\033[32m",   # Green
    "WARNING":  "\033[33m",   # Yellow
    "ERROR":    "\033[31m",   # Red
    "CRITICAL": "\033[35m",   # Magenta
    "RESET":    "\033[0m",
}

_USE_COLOUR = sys.stdout.isatty() and os.name != "nt"


# ---------------------------------------------------------------------------
# Custom formatters
# ---------------------------------------------------------------------------

class ColouredConsoleFormatter(logging.Formatter):
    """
    Console formatter with ANSI colour per log level.

    Format:
        2024-01-15 14:23:01 | INFO     | src.features.extractor | Processed 120/2140 videos
        2024-01-15 14:23:01 | WARNING  | src.features.extractor | video_id=v0234 frame=8 | MediaPipe missed left hand
    """

    _FMT = (
        "%(asctime)s | %(levelname)-8s | %(name)s"
        "%(extra_fields)s | %(message)s"
    )
    _DATE_FMT = "%Y-%m-%d %H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:
        extra_fields = _extract_extra_fields(record)
        record.extra_fields = (" | " + extra_fields) if extra_fields else ""
        formatted = logging.Formatter(self._FMT, datefmt=self._DATE_FMT).format(record)

        if _USE_COLOUR:
            colour = _COLOURS.get(record.levelname, _COLOURS["RESET"])
            reset = _COLOURS["RESET"]
            formatted = formatted.replace(
                record.levelname.ljust(8),
                f"{colour}{record.levelname.ljust(8)}{reset}",
                1,
            )
        return formatted


class FileFormatter(logging.Formatter):
    """
    Plain-text file formatter — no ANSI codes, machine-parseable.

    Format:
        2024-01-15 14:23:01 | INFO | src.features.extractor | video_id=v0234 | Processed 120/2140
    """

    _FMT = (
        "%(asctime)s | %(levelname)s | %(name)s"
        "%(extra_fields)s | %(message)s"
    )
    _DATE_FMT = "%Y-%m-%d %H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:
        extra_fields = _extract_extra_fields(record)
        record.extra_fields = (" | " + extra_fields) if extra_fields else ""
        return logging.Formatter(self._FMT, datefmt=self._DATE_FMT).format(record)


# ---------------------------------------------------------------------------
# Structured extra-fields helper
# ---------------------------------------------------------------------------

# Project-specific structured fields rendered as key=value in log lines.
# Only these fields are rendered — prevents leaking internal LogRecord attrs.
_KNOWN_EXTRA_FIELDS = (
    "video_id", "sign", "frame", "epoch", "run_id", "split",
    "model", "experiment", "signer_id", "confidence", "fps",
    "stage", "val_acc", "train_acc",
)


def _extract_extra_fields(record: logging.LogRecord) -> str:
    """
    Render known structured extra fields from a LogRecord as 'key=value' string.
    Preserves the field order defined in _KNOWN_EXTRA_FIELDS for readability.
    """
    parts = []
    for field in _KNOWN_EXTRA_FIELDS:
        value = getattr(record, field, None)
        if value is not None:
            parts.append(f"{field}={value}")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# MLflow run ID injection helper
# ---------------------------------------------------------------------------

def _try_get_active_mlflow_run_id() -> Optional[str]:
    """
    Attempt to retrieve the active MLflow run ID without hard-depending on MLflow.
    Returns None if MLflow is not installed, not imported, or no run is active.
    """
    try:
        import mlflow  # type: ignore[import]
        active = mlflow.active_run()
        if active is not None:
            return active.info.run_id
    except Exception:  # noqa: BLE001 — intentionally broad; must not crash the logger
        pass
    return None


# ---------------------------------------------------------------------------
# StructuredAdapter — wraps Logger with persistent context + MLflow injection
# ---------------------------------------------------------------------------

class StructuredAdapter(logging.LoggerAdapter):
    """
    A LoggerAdapter that:
      1. Merges persistent context (e.g. video_id, stage) into every log record.
      2. Injects the active MLflow run ID as run_id= on every line when a run
         is active — so log files and MLflow runs are always cross-referenceable.

    Usage:
        logger = get_logger(__name__, video_id="v001", stage="preprocessing")
        logger.info("Frame extracted")          # includes video_id=v001, stage=preprocessing
        logger.info("Epoch done", extra={"epoch": 5, "val_acc": 0.81})  # per-call override
    """

    def process(
        self,
        msg: str,
        kwargs: MutableMapping[str, Any],
    ) -> tuple[str, MutableMapping[str, Any]]:
        # Build extra: start with persistent context, then apply per-call overrides
        extra: dict[str, Any] = dict(self.extra)

        # Per-call extra — validate it is a Mapping before merging
        per_call_extra = kwargs.pop("extra", {})
        if isinstance(per_call_extra, dict):
            extra.update(per_call_extra)
        else:
            # Unexpected type — log a warning but don't crash
            logging.getLogger(__name__).warning(
                f"StructuredAdapter.process: 'extra' kwarg is not a dict "
                f"(got {type(per_call_extra).__name__}), ignoring."
            )

        # Inject active MLflow run ID if not already present and a run is active
        if "run_id" not in extra:
            run_id = _try_get_active_mlflow_run_id()
            if run_id is not None:
                extra["run_id"] = run_id[:8]  # Short prefix keeps log lines readable

        kwargs["extra"] = extra
        return msg, kwargs


# ---------------------------------------------------------------------------
# Global state — protected by a threading.Lock
# ---------------------------------------------------------------------------

_configure_lock = threading.Lock()
_LOG_FILE_PATH: Optional[Path] = None
_ROOT_CONFIGURED: bool = False

# Default maximum bytes before the log file rotates (10 MB)
_DEFAULT_MAX_BYTES: int = 10 * 1024 * 1024
# Default number of rotated backup files to keep
_DEFAULT_BACKUP_COUNT: int = 5


def configure_logging(
    log_dir: str = "logs",
    run_name: str = "run",
    level: str = "INFO",
    file_level: str = "DEBUG",
    max_bytes: int = _DEFAULT_MAX_BYTES,
    backup_count: int = _DEFAULT_BACKUP_COUNT,
) -> Path:
    """
    Configure the root logger for the entire pipeline.

    Call this ONCE at the entry point of each pipeline script before any
    other code that uses get_logger(). It is idempotent: if called a second
    time, it returns the path of the already-open log file without creating
    new handlers or a new file.

    Thread-safe: if two threads call configure_logging() simultaneously during
    startup, only one will create handlers; both will receive the same file path.

    Parameters
    ----------
    log_dir : str
        Directory where log files are written. Created if it does not exist.
    run_name : str
        Human-readable name embedded in the log filename.
        "bilstm_seq30_aug" → logs/20240115_143201_bilstm_seq30_aug.log
    level : str
        Minimum log level for console output. One of DEBUG/INFO/WARNING/ERROR.
    file_level : str
        Minimum log level for file output. Defaults to DEBUG (captures everything).
    max_bytes : int
        Maximum size in bytes before the log file rotates. Default 10 MB.
    backup_count : int
        Number of rotated backup log files to keep. Default 5.

    Returns
    -------
    Path
        Absolute path to the log file for this run.
    """
    global _LOG_FILE_PATH, _ROOT_CONFIGURED

    # Fast path — no lock needed for pure read if already configured
    if _ROOT_CONFIGURED and _LOG_FILE_PATH is not None:
        return _LOG_FILE_PATH

    with _configure_lock:
        # Double-checked locking: re-check after acquiring the lock
        if _ROOT_CONFIGURED and _LOG_FILE_PATH is not None:
            return _LOG_FILE_PATH

        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_run_name = "".join(
            c if c.isalnum() or c in "-_" else "_" for c in run_name
        )
        log_filename = f"{timestamp}_{safe_run_name}.log"
        _LOG_FILE_PATH = log_dir_path / log_filename

        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)  # Root captures all; handlers filter

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        console_handler.setFormatter(ColouredConsoleFormatter())
        root_logger.addHandler(console_handler)

        # Rotating file handler — prevents unbounded log growth on long training runs
        file_handler = logging.handlers.RotatingFileHandler(
            _LOG_FILE_PATH,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(getattr(logging, file_level.upper(), logging.DEBUG))
        file_handler.setFormatter(FileFormatter())
        root_logger.addHandler(file_handler)

        # Suppress noisy third-party loggers
        for noisy_lib in ("urllib3", "requests", "matplotlib", "PIL", "absl"):
            logging.getLogger(noisy_lib).setLevel(logging.WARNING)

        _ROOT_CONFIGURED = True

    # Log the confirmation outside the lock (handlers are now installed)
    root_logger = logging.getLogger()
    root_logger.info(
        "Logging configured",
        extra={"stage": "init"},
    )
    root_logger.info(f"Log file: {_LOG_FILE_PATH.resolve()}")

    return _LOG_FILE_PATH


def _apply_fallback_config() -> None:
    """
    Minimal fallback logging config for notebooks and tests.
    Does not write to a file. Sets _ROOT_CONFIGURED to prevent recursion.
    Thread-safe via _configure_lock.
    """
    global _ROOT_CONFIGURED

    with _configure_lock:
        if _ROOT_CONFIGURED:
            return

        root = logging.getLogger()
        if not root.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(ColouredConsoleFormatter())
            root.addHandler(handler)
            root.setLevel(logging.DEBUG)

        _ROOT_CONFIGURED = True


def get_logger(name: str, **context: Any) -> StructuredAdapter:
    """
    Return a StructuredAdapter wrapping the named logger.

    If configure_logging() has not been called yet (e.g. in unit tests or
    notebooks), a minimal fallback configuration is applied automatically.

    Parameters
    ----------
    name : str
        Logger name — always pass __name__ from the calling module.
    **context : Any
        Persistent structured fields attached to every log record from this adapter.
        Common fields: video_id, sign, stage, model, epoch.

    Returns
    -------
    StructuredAdapter
        Ready-to-use logger with .debug(), .info(), .warning(), .error().

    Examples
    --------
    Basic:
        logger = get_logger(__name__)
        logger.info("Extraction started")

    With persistent context:
        logger = get_logger(__name__, stage="preprocessing", split="train")
        logger.info("Processing video", extra={"video_id": "v0042", "sign": "book"})

    Per-call context override:
        logger = get_logger(__name__, model="bilstm")
        logger.info("Epoch complete", extra={"epoch": 5, "val_acc": 0.81})
    """
    if not _ROOT_CONFIGURED:
        _apply_fallback_config()

    underlying = logging.getLogger(name)
    return StructuredAdapter(underlying, extra=context)


def get_log_file_path() -> Optional[Path]:
    """Return the path to the current run's log file, or None if not yet configured."""
    return _LOG_FILE_PATH


# ---------------------------------------------------------------------------
# Pipeline-specific convenience loggers
# ---------------------------------------------------------------------------

def get_preprocessing_logger(video_id: Optional[str] = None) -> StructuredAdapter:
    """Logger pre-configured with stage=preprocessing context."""
    ctx: dict[str, Any] = {"stage": "preprocessing"}
    if video_id:
        ctx["video_id"] = video_id
    return get_logger("src.features.extractor", **ctx)


def get_training_logger(
    model: Optional[str] = None,
    run_id: Optional[str] = None,
) -> StructuredAdapter:
    """Logger pre-configured with stage=training context."""
    ctx: dict[str, Any] = {"stage": "training"}
    if model:
        ctx["model"] = model
    if run_id:
        ctx["run_id"] = run_id
    return get_logger("src.models.train", **ctx)


def get_inference_logger() -> StructuredAdapter:
    """Logger pre-configured with stage=inference context."""
    return get_logger("src.inference.predictor", stage="inference")


def get_demo_logger() -> StructuredAdapter:
    """Logger pre-configured with stage=demo context."""
    return get_logger("src.demo.webcam_demo", stage="demo")