# =============================================================================
# Dockerfile.inference — WLASL Gesture Recognition: Lean Production Inference
# =============================================================================
#
# Author  : Henry Otsyula — Senior Data Scientist & ML Engineer
# Stage   : 10 (Infrastructure)
# Purpose : Minimal image for serving gesture predictions via GesturePredictor.
#           Contains ONLY what is needed at inference time — TFLite runtime,
#           MediaPipe Hands, OpenCV (headless), NumPy, OmegaConf, Pydantic.
#
# Build   : docker build -f Dockerfile.inference -t wlasl-inference:latest .
# Run     : docker run --rm \
#               -v $(pwd)/models:/app/models:ro \
#               -v $(pwd)/artifacts:/app/artifacts:ro \
#               wlasl-inference:latest
#
# WHAT IS EXCLUDED (deliberately)
# ─────────────────────────────────
#   ✗  TensorFlow 2.13.1 (~500 MB)       — replaced by tflite-runtime (~10 MB)
#   ✗  MLflow                             — no experiment tracking at inference
#   ✗  scikit-learn                       — no evaluation metrics at inference
#   ✗  SHAP                               — no interpretability at inference
#   ✗  pandas, scipy, matplotlib          — no data analysis at inference
#   ✗  src/models/                        — no Keras model building at inference
#   ✗  src/evaluation/                    — no benchmark/calibration at inference
#   ✗  src/export/                        — no TFLite conversion at inference
#   ✗  src/data/                          — no WLASL downloading at inference
#   ✗  pipelines/                         — no orchestration scripts at inference
#   ✗  tests/                             — no test suite at inference
#
# WHAT IS INCLUDED (minimum viable inference path)
# ──────────────────────────────────────────────────
#   ✓  tflite-runtime==2.13.0    — matches TF 2.13.1 ABI; includes SELECT_TF_OPS
#                                   flex delegate for BiLSTM TensorList ops
#   ✓  mediapipe==0.10.14        — landmark extraction (HandsExtractor)
#   ✓  opencv-python-headless    — frame capture and BGR→RGB conversion
#   ✓  numpy==1.24.3             — array operations (pinned for ABI compat)
#   ✓  omegaconf==2.3.0          — config snapshot loading
#   ✓  pydantic==2.8.2           — ExperimentConfig validation
#   ✓  protobuf==3.20.3          — MediaPipe dependency (pinned)
#   ✓  src/inference/            — GesturePredictor, FrameBuffer, PredictionSmoother
#   ✓  src/features/             — FeaturePipeline, constants, augmentation
#   ✓  src/utils/                — config, label_map, logger, reproducibility
#   ✓  src/demo/                 — GestureStreamSession, HandsExtractor, HUDRenderer
#
# SELECT_TF_OPS NOTE
# ───────────────────
# The champion BiLSTM (bilstm_hands_only_v4_aug) emits TensorListReserve/
# TensorListStack ops that the standard TFLite builtin-ops runtime cannot lower.
# tflite-runtime 2.13.0 includes the SELECT_TF_OPS flex delegate (~800 KB).
# This adds ~800 KB to the Android TFLite runtime binary — known and accepted.
# ABI compatibility: tflite-runtime 2.13.0 must exactly match TF 2.13.1.
# Using a mismatched version risks silent numerical differences in the flex
# delegate's dequantisation path.
#
# MODEL MOUNTING
# ───────────────
# The TFLite model file is NOT baked into this image. It is mounted at runtime:
#   docker run -v $(pwd)/models:/app/models:ro wlasl-inference:latest
# This decouples model versioning from image versioning — when the KSL model
# arrives, a new .tflite file is mounted without rebuilding the image.
# =============================================================================

FROM python:3.10-slim

# ─── Build arguments ─────────────────────────────────────────────────────────
ARG BUILD_DATE
ARG GIT_COMMIT=unknown
ARG VERSION=1.0.0

# ─── Labels ──────────────────────────────────────────────────────────────────
LABEL org.opencontainers.image.title="WLASL Gesture Recognition — Inference Image"
LABEL org.opencontainers.image.description="Lean production inference: GesturePredictor + TFLite runtime only (~300 MB)"
LABEL org.opencontainers.image.version="${VERSION}"
LABEL org.opencontainers.image.created="${BUILD_DATE}"
LABEL org.opencontainers.image.revision="${GIT_COMMIT}"
LABEL org.opencontainers.image.authors="Henry Otsyula"
LABEL maintainer="Henry Otsyula"

# ─── Environment ─────────────────────────────────────────────────────────────
# PYTHONHASHSEED is set here too (same reasoning as training image):
# must be present before Python starts for hash-seed reproducibility.
# Although inference is deterministic without it, consistent hashing
# avoids subtle dict-ordering differences between training and inference
# that could affect label map lookups in edge cases.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PYTHONHASHSEED=42 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ─── Minimal system dependencies ─────────────────────────────────────────────
# libgl1-mesa-glx   : Required by opencv-python-headless (libGL.so.1 is linked
#                     even in headless mode — omitting it causes ImportError).
# libglib2.0-0      : Required by MediaPipe's threading layer (GLib GMainLoop).
#                     Without it mediapipe crashes at the C++ runtime level.
#
# Deliberately excluded vs. training image:
#   git              — no git operations at inference time
#   ffmpeg           — no video encoding at inference time
#   libsm6/libxext6  — no X11/GUI dependencies in headless inference
#   libxrender1      — same
#
# Each excluded package saves image layers and reduces the attack surface.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# ─── Python inference dependencies ───────────────────────────────────────────
# requirements-inference.txt is the minimal dependency set for GesturePredictor.
# See that file for the full rationale for each pinned version.
#
# Install order matters:
#   1. protobuf first (MediaPipe's version constraint; must not be upgraded)
#   2. requirements-inference.txt (tflite-runtime, mediapipe, opencv-headless)
COPY requirements-inference.txt ./

RUN pip install --upgrade pip setuptools wheel \
    && pip install protobuf==3.20.3 \
    && pip install -r requirements-inference.txt

# ─── Inference-path source modules only ──────────────────────────────────────
# Copy ONLY the four src/ subdirectories that GesturePredictor's import chain
# requires. Anything not in this list is explicitly excluded — src/models/,
# src/evaluation/, src/export/, src/data/ are never imported at inference time.
#
# Import chain verification (from src/inference/predictor.py):
#   GesturePredictor
#     → src.features.pipeline (FeaturePipeline)
#     → src.features.constants (FEATURE_SIZE, LANDMARK_CONFIGS)
#     → src.features.augmentation (AugmentationPipeline — training=False bypass)
#     → src.utils.config (ExperimentConfig, load_config_from_manifest)
#     → src.utils.label_map (LabelMap, get_label_map)
#     → src.utils.logger (get_logger, StructuredAdapter)
#     → src.utils.reproducibility (set_seeds — used by config loading only)
#   GestureStreamSession (src/demo/webcam_demo.py)
#     → src.inference.predictor (all of the above)
#     → src.features.constants (FEATURE_SIZE)
COPY src/inference/  ./src/inference/
COPY src/features/   ./src/features/
COPY src/utils/      ./src/utils/
COPY src/demo/       ./src/demo/

# ─── Committed runtime artifacts ─────────────────────────────────────────────
# These small files are committed to the repository and are stable post-Stage 5.
# They are baked into the image (not mounted) because:
#   - label_map_v1.json: 35 class names — changes only if the label map changes
#   - config_snapshot.yaml: champion training config — locked at Stage 5
# Both are needed by GesturePredictor.from_config_snapshot() at startup.
COPY artifacts/label_map_v1.json                                             \
     ./artifacts/label_map_v1.json
COPY artifacts/experiments/bilstm_hands_only_v4_aug/config_snapshot.yaml    \
     ./artifacts/experiments/bilstm_hands_only_v4_aug/config_snapshot.yaml

# ─── Model mount point ────────────────────────────────────────────────────────
# The TFLite model file is NOT baked in. Create the target directory so the
# runtime mount point exists with correct ownership.
# Mount at runtime: docker run -v $(pwd)/models:/app/models:ro wlasl-inference
RUN mkdir -p models

# ─── Non-root user ────────────────────────────────────────────────────────────
RUN groupadd --gid 1000 mluser \
    && useradd --uid 1000 --gid mluser --shell /bin/bash --create-home mluser \
    && chown -R mluser:mluser /app

USER mluser

# ─── Health check ─────────────────────────────────────────────────────────────
# Verifies that the full GesturePredictor import chain resolves cleanly,
# including tflite-runtime, mediapipe, and the project's own modules.
# A failing health check indicates a broken image (e.g. a protobuf conflict
# that only manifests at import time) rather than a missing model file.
#
# --interval  : check every 30s after the container starts
# --timeout   : allow up to 10s for the Python import chain (MediaPipe is slow)
# --start-period: give the container 15s to initialise before failing
# --retries   : 3 failures → UNHEALTHY (not 1, to tolerate transient init lag)
HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=15s \
    --retries=3 \
    CMD python -c "\
from src.inference.predictor import GesturePredictor, FrameBuffer, PredictionSmoother; \
from src.features.pipeline import FeaturePipeline; \
from src.utils.label_map import get_label_map; \
print('HEALTHY: inference stack imports cleanly')"

# ─── Default command ──────────────────────────────────────────────────────────
# Show help by default so an accidental bare `docker run wlasl-inference`
# prints usage rather than failing with a cryptic error.
# For the actual webcam demo, mount models and run without --help:
#   docker run --rm \
#       -v $(pwd)/models:/app/models:ro \
#       -v $(pwd)/artifacts:/app/artifacts:ro \
#       --device /dev/video0:/dev/video0 \
#       -e DISPLAY=$DISPLAY \
#       --network host \
#       wlasl-inference:latest \
#       python src/demo/webcam_demo.py --model models/gesture_bilstm_v1.tflite
CMD ["python", "src/demo/webcam_demo.py", "--help"]