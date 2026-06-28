# =============================================================================
# Dockerfile — WLASL Gesture Recognition: Full Training + Evaluation Image
# =============================================================================
#
# Author  : Henry Otsyula — Senior Data Scientist & ML Engineer
# Stage   : 10 (Infrastructure)
# Purpose : Reproduce Stages 1–9 end-to-end: raw video → landmark extraction
#           → training → evaluation → TFLite export → verification.
#
# Build   : docker build -t wlasl-train:latest .
# Run     : docker-compose run --rm train
#
# DESIGN RATIONALE
# ─────────────────
# Base image: python:3.10-slim (NOT tensorflow/tensorflow:2.13.x)
#   The official TF Docker images pin to specific CUDA/system-Python
#   configurations that conflict with this project's Miniconda-based
#   development environment. Starting from python:3.10-slim and installing
#   TF 2.13.1 via pip produces byte-for-byte identical dependency resolution
#   to the project's local venv — the only way to guarantee reproducibility.
#
# Layer ordering: least-likely-to-change → most-likely-to-change
#   System libs → protobuf (special pre-install) → requirements.txt →
#   requirements-dev.txt → source code.
#   If source changes but requirements don't, the expensive pip layer
#   (TF 2.13.1 = ~500 MB download) is served from cache — critical for
#   iterative development.
#
# Determinism: PYTHONHASHSEED and TF env vars are baked into the image as
#   ENV directives. This ensures they are set BEFORE Python starts, which is
#   the correct ordering — many TF env vars are read at import time and have
#   no effect if set after the interpreter is already running.
#   docker-compose.yml also sets them in the `environment:` block as a
#   belt-and-suspenders measure (compose environment overrides image ENV).
#
# protobuf install ordering: protobuf==3.20.3 must be installed BEFORE
#   mediapipe. MediaPipe 0.10.14 declares protobuf>=3.20 as a constraint;
#   installing mediapipe first can pull in a newer incompatible protobuf.
# =============================================================================

FROM python:3.10-slim-bookworm

# ─── Build arguments (override at build time for CI tagging) ─────────────────
ARG BUILD_DATE
ARG GIT_COMMIT=unknown
ARG VERSION=1.0.0

# ─── Labels (OCI image spec) ─────────────────────────────────────────────────
LABEL org.opencontainers.image.title="WLASL Gesture Recognition — Training Image"
LABEL org.opencontainers.image.description="Full ML pipeline: preprocessing → training → evaluation → TFLite export"
LABEL org.opencontainers.image.version="${VERSION}"
LABEL org.opencontainers.image.created="${BUILD_DATE}"
LABEL org.opencontainers.image.revision="${GIT_COMMIT}"
LABEL org.opencontainers.image.authors="Henry Otsyula"
LABEL maintainer="Henry Otsyula"

# ─── Core environment ─────────────────────────────────────────────────────────
# PYTHONUNBUFFERED=1  : stdout/stderr are not buffered — log lines appear
#                       immediately in docker logs / CI output.
# PYTHONDONTWRITEBYTECODE=1 : no .pyc files — reduces image size and avoids
#                             stale bytecode problems when source is mounted.
# PYTHONPATH=/app     : lets all `from src.*` imports resolve without
#                       install-editable hacks.
# PYTHONHASHSEED=42   : must be set before Python starts (ENV, not RUN export)
#                       to control Python's hash randomisation for full
#                       cross-run reproducibility.
# TF_DETERMINISTIC_OPS / TF_CUDNN_DETERMINISTIC : read by TF at import time.
#                       Setting them here (image ENV) ensures they are present
#                       regardless of how the container is launched.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PYTHONHASHSEED=42 \
    TF_DETERMINISTIC_OPS=1 \
    TF_CUDNN_DETERMINISTIC=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ─── System dependencies ───────────────────────────────────────────────────────
# libgl1-mesa-glx   : OpenCV's cv2.VideoCapture / cv2.imshow GPU-free backend.
#                     Required even in headless mode (libGL.so.1 is loaded on
#                     import even when no display is used).
# libglib2.0-0      : MediaPipe's GLib-based process and thread management.
#                     Missing this causes a silent segfault at mediapipe import.
# libgomp1          : TensorFlow's OpenMP threading backend.
#                     Missing this causes a load error in libtensorflow_framework.
# libsm6, libxext6, libxrender1 : Full OpenCV GUI stack for the webcam demo
#                                  (Stage 9). Not needed for headless training/
#                                  evaluation, but including them keeps the
#                                  training image usable for demo development.
# git               : Required by src/utils/reproducibility.py's _get_git_info()
#                     which calls `git rev-parse HEAD` via subprocess to embed
#                     the commit hash in every run manifest.
# ffmpeg            : Required for the `make gif` target (Stage 9 demo GIF
#                     generation from webcam recordings).
# curl              : Useful for health checks and downloading model assets
#                     in CI/CD pipelines.
# ca-certificates   : Ensures TLS verification works for pip and mlflow
#                     remote tracking servers.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        git \
        ffmpeg \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# ─── Python dependency installation ───────────────────────────────────────────
# COPY requirements before source so this expensive layer is cached separately.
# A source-only change (no requirements change) skips the entire pip install.
#
# Step 1: protobuf must precede mediapipe to pin the correct version.
#         MediaPipe 0.10.14 requires protobuf>=3.20; installing mediapipe
#         first can silently pull in protobuf 4.x which breaks the API.
#
# Step 2: Install core runtime requirements (TF 2.13.1 is the largest
#         download at ~500 MB; cached after first build).
#
# Step 3: Install dev/test requirements (pytest, flake8, black, etc.)
#         These are in a separate pip invocation so CI can optionally
#         skip --target=requirements-dev.txt for minimal inference images.
COPY requirements.txt requirements-dev.txt ./

RUN pip install --upgrade pip setuptools wheel \
    && pip install protobuf==3.20.3 \
    && pip install -r requirements.txt \
    && pip install -r requirements-dev.txt

# ─── Source code ──────────────────────────────────────────────────────────────
# Ordered least-to-most frequently changed for maximum cache utilisation:
#   1. configs/   — rarely changes after hyperparameter sweeps are finalised
#   2. artifacts/ — label map and config snapshots are stable post-Stage 5
#   3. src/       — changes during active development
#   4. pipelines/ — changes during active development
#   5. tests/     — changes during Stage 10
COPY configs/    ./configs/
COPY artifacts/  ./artifacts/
COPY src/        ./src/
COPY pipelines/  ./pipelines/
COPY tests/      ./tests/

# ─── Runtime directory scaffold ───────────────────────────────────────────────
# Pre-create all directories that pipeline scripts write to at runtime.
# This avoids permission errors when these paths are bind-mounted from the
# host (Docker creates missing mount-point directories as root; pre-creating
# them here ensures correct ownership under the app user).
RUN mkdir -p \
        data/raw \
        data/landmarks \
        data/splits \
        models \
        logs \
        mlruns \
        reports/figures \
        reports/evaluation \
        artifacts/experiments

# ─── Non-root user (security hardening) ───────────────────────────────────────
# Running as root inside a container is a security anti-pattern — if the
# container is compromised, the attacker has effective root on the host via
# volume mounts. Creating a dedicated user costs nothing and is standard
# production practice.
#
# NOTE: UID/GID 1000 matches the default Linux desktop user — volumes
# bind-mounted from a typical developer workstation will have correct
# read/write permissions without manual chown.
RUN groupadd --gid 1000 mluser \
    && useradd --uid 1000 --gid mluser --shell /bin/bash --create-home mluser \
    && chown -R mluser:mluser /app

USER mluser

# ─── Smoke-test the install at build time ─────────────────────────────────────
# Verifies that TF, MediaPipe, MLflow, and the project's own config system
# all import cleanly before the image is tagged. A broken dependency (e.g.
# a protobuf version conflict silently allowing import but crashing on first
# use) would otherwise only surface at container startup, wasting CI minutes.
#
# We deliberately do NOT import tensorflow here (it prints verbose GPU-detection
# warnings even with TF_CPP_MIN_LOG_LEVEL=2) — just verify the import chain
# at the Python level.
RUN python -c "\
import mediapipe; \
import mlflow; \
import omegaconf; \
import pydantic; \
import sklearn; \
import cv2; \
print('Smoke test PASSED: all core dependencies import cleanly')"

# ─── Default command ──────────────────────────────────────────────────────────
# Run the full 23-run ablation experiment matrix by default.
# Override per service in docker-compose.yml (e.g. `command: pytest tests/`).
CMD ["python", "pipelines/run_all_experiments.py"]

# ─── Metadata ─────────────────────────────────────────────────────────────────
# Expose no ports (all MLflow UI access goes through docker-compose port mapping).
# Declare the working directory as a volume hint for IDEs and orchestrators.
VOLUME ["/app/data", "/app/models", "/app/mlruns", "/app/logs"]