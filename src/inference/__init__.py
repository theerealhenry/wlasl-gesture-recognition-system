"""
src/inference
==============
Stage 7 — Unified Inference Engine package for the WLASL 35-class gesture
recognition system.

This package's sole purpose is to expose ``GesturePredictor`` as the single,
canonical entry point through which every downstream consumer — Stage 8's
TFLite verification (``src/export/verify.py``), Stage 9's real-time webcam
demo (``src/demo/webcam_demo.py``), a future Android wrapper, or an ad-hoc
notebook cell — runs the champion model. See ``src/inference/predictor.py``
for the full architectural rationale (module docstring) and the Stage 7
specification document for the design history.

Public API
----------
    GesturePredictor   — unified inference class (primary public API)
    PredictionSmoother  — sliding-window majority vote + exponential smoothing
    FrameBuffer         — rolling fixed-length raw-landmark accumulator
    PredictionResult    — TypedDict describing every predict_*() return value
    TopKEntry           — TypedDict for one entry in PredictionResult["top_k"]

Module-level defaults (re-exported for convenience — e.g. constructing a
PredictionSmoother or GesturePredictor with explicit, self-documenting
keyword arguments rather than bare numeric literals):
    DEFAULT_SMOOTHER_WINDOW
    DEFAULT_SMOOTHING_ALPHA
    DEFAULT_DISPLAY_THRESHOLD
    DEFAULT_TOP_K
    DEFAULT_AUTO_RESET_NO_DETECTION_FRAMES

Typical usage
-------------
Direct construction (explicit config + model path)::

    from src.inference import GesturePredictor
    from src.utils.config import load_config

    cfg = load_config(model="bilstm", data="seq100", augmentation="spatial_temporal")
    predictor = GesturePredictor(
        model_path="models/bilstm_hands_only_v4_aug_saved_model/",
        config=cfg,
    )
    result = predictor.predict_from_landmarks(raw_landmarks_225d)

Recommended construction (from a saved run's exact training config)::

    from src.inference import GesturePredictor

    predictor = GesturePredictor.from_config_snapshot(
        config_snapshot_path="artifacts/experiments/bilstm_hands_only_v4_aug/config_snapshot.yaml",
        model_path="models/bilstm_hands_only_v4_aug_saved_model/",
    )

Why this package intentionally stays thin
------------------------------------------
Every behavioural decision (calibration-aware thresholding, smoothing
parameters, model-format auto-detection, evaluation-framework compatibility
via ``__call__``) lives in ``predictor.py`` itself, where it sits next to the
Stage 6 findings that motivate it. This ``__init__.py`` performs no logic of
its own beyond re-exporting the public surface — duplicating any of that
reasoning here would create a second place it could drift out of sync with
the implementation, which is exactly the "documentation vs. reality"
divergence Stage 6 already caught once (``early_stopping_monitor`` narrated
as ``val_macro_f1`` but actually ``val_accuracy`` in the champion's
config snapshot). One source of truth: ``predictor.py``.

Import-time behaviour
----------------------
Importing this package does NOT import TensorFlow, OpenCV, or MediaPipe.
``predictor.py`` defers all three behind lazy, method-local imports (model
loading, ``predict_from_video()``, ``predict_from_webcam_frame()``
respectively) so that ``import src.inference`` remains cheap and side-effect
free — safe to do from a CLI tool, a test collection step, or a minimal
inference-only container (``Dockerfile.inference``) that may not even have
MediaPipe installed (hands_only inference never needs pose, and a
TFLite-only deployment never needs the Keras/TensorFlow training stack
beyond the interpreter).

If ``predictor.py`` itself fails to import (e.g. a missing sibling module
such as ``src.features.pipeline`` or ``src.utils.label_map``), the original
``ImportError`` is re-raised with the module's traceback intact — this
package does not swallow or re-wrap that error, since hiding the real cause
would only slow down debugging a broken environment.
"""

from __future__ import annotations

from src.inference.predictor import (
    DEFAULT_AUTO_RESET_NO_DETECTION_FRAMES,
    DEFAULT_DISPLAY_THRESHOLD,
    DEFAULT_SMOOTHER_WINDOW,
    DEFAULT_SMOOTHING_ALPHA,
    DEFAULT_TOP_K,
    FrameBuffer,
    GesturePredictor,
    PredictionResult,
    PredictionSmoother,
    TopKEntry,
)

#: Stage 7 package version. Bumped on any change to GesturePredictor's
#: public contract (constructor signature, PredictionResult schema, or
#: default threshold/smoothing values) — NOT on internal refactors that
#: preserve the existing contract. Consumers that pin against a specific
#: PredictionResult schema (e.g. a Stage 9 HUD renderer) can assert on this.
__version__: str = "1.0.0"

#: Stage 7 completion gate flag (informational only — never asserted at
#: import time). The accompanying test suite (tests/test_predictor.py) is
#: this package's actual completion gate; this flag exists purely so a
#: notebook or CLI banner can confirm at a glance which inference engine
#: revision is loaded without importing the test module.
__stage__: str = "Stage 7 — Unified Inference Engine"

__all__ = [
    "GesturePredictor",
    "PredictionSmoother",
    "FrameBuffer",
    "PredictionResult",
    "TopKEntry",
    "DEFAULT_SMOOTHER_WINDOW",
    "DEFAULT_SMOOTHING_ALPHA",
    "DEFAULT_DISPLAY_THRESHOLD",
    "DEFAULT_TOP_K",
    "DEFAULT_AUTO_RESET_NO_DETECTION_FRAMES",
    "__version__",
]