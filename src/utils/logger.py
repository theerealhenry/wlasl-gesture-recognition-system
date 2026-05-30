"""
src/utils/logger.py
====================
Production-grade structured logging for the WLASL gesture recognition pipeline.

Design principles:
  - One logger per module, named after the module (__name__).
  - All loggers share a single root configuration set up by `configure_logging()`.
  - Every run writes to both the console (coloured) and a timestamped log file.
  - Log files are written to logs/<timestamp>_<run_name>.log and never overwritten.
  - Structured extra fields (video_id, frame_idx, epoch, etc.) are supported via
    LoggerAdapter wrappers returned by `get_logger()`.
  - MLflow run IDs are injected into every log line automatically when a run is active.

Usage (in any module):
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Processing started", extra={"video_id": "v001", "sign": "book"})
    logger.warning("MediaPipe missed hand", extra={"video_id": "v001", "frame": 12})

The logger is idempotent: calling get_logger() multiple times with the same name
returns the same logger without duplicating handlers.
"""

import logging
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, MutableMapping, Any

# ---------------------------------------------------------------------------
# ANSI colour codes for console output — degraded gracefully on Windows
# ---------------------------------------------------------------------------
_COLOURS = {
    "DEBUG":    "\033[36m",   # Cyan
    "INFO":     "\033[32m",   # Green
    "WARNING":  "\033[33m",   # Yellow
    "ERROR":    "\033[31m",   # Red
    "CRITICAL": "\033[35m",   # Magenta
    "RESET":    "\033[0m",
}

_USE_COLOUR = sys.stdout.isatty() and os.name != "nt"  # disable on Windows CI


# ---------------------------------------------------------------------------
# Custom formatters
# ---------------------------------------------------------------------------

class ColouredConsoleFormatter(logging.Formatter):
    """
    Console formatter with ANSI colour per log level.

    Format:
        2024-01-15 14:23:01 | INFO     | src.features.extractor | Processed 120/2140 videos
        2024-01-15 14:23:01 | WARNING  | src.features.extractor | video_id=v0234 | MediaPipe missed left hand at frame 8
    """

    _FMT = (
        "%(asctime)s | %(levelname)-8s | %(name)s"
        "%(extra_fields)s | %(message)s"
    )
    _DATE_FMT = "%Y-%m-%d %H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:
        # Inject any structured extra fields as key=value pairs
        extra_fields = _extract_extra_fields(record)
        record.extra_fields = (" | " + extra_fields) if extra_fields else ""

        formatted = logging.Formatter(self._FMT, datefmt=self._DATE_FMT).format(record)

        if _USE_COLOUR:
            colour = _COLOURS.get(record.levelname, _COLOURS["RESET"])
            reset = _COLOURS["RESET"]
            # Colour only the level name portion for readability
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
        2024-01-15 14:23:01 | INFO | src.features.extractor | video_id=v0234 | Processed 120/2140 videos
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

# Fields that are standard LogRecord attributes — never treated as "extras"
_STANDARD_LOG_RECORD_FIELDS = frozenset({
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "message", "asctime", "extra_fields",
    "taskName",  # Python 3.12+
})

# Project-specific structured fields that will be rendered as key=value
_KNOWN_EXTRA_FIELDS = {
    "video_id", "sign", "frame", "epoch", "run_id", "split",
    "model", "experiment", "signer_id", "confidence", "fps",
    "stage", "val_acc", "train_acc",
}


def _extract_extra_fields(record: logging.LogRecord) -> str:
    """
    Render known structured extra fields from a LogRecord as 'key=value' string.
    Only renders fields in _KNOWN_EXTRA_FIELDS to avoid leaking internal attrs.
    """
    parts = []
    for field in _KNOWN_EXTRA_FIELDS:
        value = getattr(record, field, None)
        if value is not None:
            parts.append(f"{field}={value}")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# StructuredAdapter — wraps a Logger to inject persistent context fields
# ---------------------------------------------------------------------------

class StructuredAdapter(logging.LoggerAdapter):
    """
    A LoggerAdapter that merges persistent context (e.g. video_id, stage)
    into every log record's extra dict.

    Usage:
        logger = get_logger(__name__, video_id="v001", stage="preprocessing")
        logger.info("Frame extracted")  # -> includes video_id=v001, stage=preprocessing
    """

    def process(
        self,
        msg: str,
        kwargs: MutableMapping[str, Any],
    ) -> tuple[str, MutableMapping[str, Any]]:
        extra = dict(self.extra)  # persistent context
        extra.update(kwargs.pop("extra", {}))  # per-call overrides
        kwargs["extra"] = extra
        return msg, kwargs


# ---------------------------------------------------------------------------
# Global state — module-level log file path set once by configure_logging()
# ---------------------------------------------------------------------------

_LOG_FILE_PATH: Optional[Path] = None
_ROOT_CONFIGURED: bool = False


def configure_logging(
    log_dir: str = "logs",
    run_name: str = "run",
    level: str = "INFO",
    file_level: str = "DEBUG",
) -> Path:
    """
    Configure the root logger for the entire pipeline. Call this ONCE at the
    entry point of each pipeline script (run_training.py, run_preprocessing.py,
    etc.) before any other import that uses get_logger().

    Parameters
    ----------
    log_dir : str
        Directory where log files are written. Created if it does not exist.
    run_name : str
        Human-readable name embedded in the log filename.
        Example: "bilstm_seq30_aug" → logs/20240115_143201_bilstm_seq30_aug.log
    level : str
        Minimum log level for console output. One of DEBUG/INFO/WARNING/ERROR.
    file_level : str
        Minimum log level for file output. Defaults to DEBUG (captures everything).

    Returns
    -------
    Path
        Absolute path to the log file created for this run.

    Example
    -------
    >>> from src.utils.logger import configure_logging
    >>> log_path = configure_logging(log_dir="logs", run_name="preprocess")
    >>> print(f"Logging to {log_path}")
    """
    global _LOG_FILE_PATH, _ROOT_CONFIGURED

    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Sanitise run_name for use in a filename
    safe_run_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in run_name)
    log_filename = f"{timestamp}_{safe_run_name}.log"
    _LOG_FILE_PATH = log_dir_path / log_filename

    root_logger = logging.getLogger()

    # Avoid adding duplicate handlers if configure_logging is called twice
    if _ROOT_CONFIGURED:
        return _LOG_FILE_PATH

    root_logger.setLevel(logging.DEBUG)  # Root captures everything; handlers filter

    # --- Console handler ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    console_handler.setFormatter(ColouredConsoleFormatter())
    root_logger.addHandler(console_handler)

    # --- File handler ---
    file_handler = logging.FileHandler(_LOG_FILE_PATH, encoding="utf-8")
    file_handler.setLevel(getattr(logging, file_level.upper(), logging.DEBUG))
    file_handler.setFormatter(FileFormatter())
    root_logger.addHandler(file_handler)

    # Suppress noisy third-party loggers
    for noisy_lib in ("urllib3", "requests", "matplotlib", "PIL", "absl"):
        logging.getLogger(noisy_lib).setLevel(logging.WARNING)

    _ROOT_CONFIGURED = True

    # Log first message to confirm setup
    root_logger.info(
        "Logging configured",
        extra={"stage": "init"},
    )
    root_logger.info(f"Log file: {_LOG_FILE_PATH.resolve()}")

    return _LOG_FILE_PATH


def get_logger(name: str, **context: Any) -> StructuredAdapter:
    """
    Return a StructuredAdapter wrapping the named logger.

    If configure_logging() has not been called yet (e.g. in unit tests or
    notebooks), a minimal fallback configuration is applied automatically
    so that log output is never silently dropped.

    Parameters
    ----------
    name : str
        Logger name — always pass __name__ from the calling module.
    **context : Any
        Persistent structured fields attached to every log record emitted
        by this adapter. Common fields: video_id, sign, stage, model, epoch.

    Returns
    -------
    StructuredAdapter
        A LoggerAdapter ready to use with .debug(), .info(), .warning(), .error().

    Examples
    --------
    Basic usage:
        logger = get_logger(__name__)
        logger.info("Extraction started")

    With persistent context:
        logger = get_logger(__name__, stage="preprocessing", split="train")
        logger.info("Processing video", extra={"video_id": "v0042", "sign": "book"})

    Override context per call:
        logger = get_logger(__name__, model="bilstm")
        logger.info("Epoch complete", extra={"epoch": 5, "val_acc": 0.81})
    """
    global _ROOT_CONFIGURED

    # Fallback: if configure_logging() was never called (e.g. in a notebook),
    # apply a minimal console-only config so nothing is silently dropped.
    if not _ROOT_CONFIGURED:
        _apply_fallback_config()

    underlying = logging.getLogger(name)
    return StructuredAdapter(underlying, extra=context)


def _apply_fallback_config() -> None:
    """
    Minimal fallback logging config for notebooks and tests.
    Does not write to a file. Marks root as configured to prevent recursion.
    """
    global _ROOT_CONFIGURED

    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(ColouredConsoleFormatter())
        root.addHandler(handler)
        root.setLevel(logging.DEBUG)

    _ROOT_CONFIGURED = True


def get_log_file_path() -> Optional[Path]:
    """Return the path to the current run's log file, or None if not configured."""
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


def get_training_logger(model: Optional[str] = None, run_id: Optional[str] = None) -> StructuredAdapter:
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