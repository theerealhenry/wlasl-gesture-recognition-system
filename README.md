# WLASL Gesture Recognition
### A Production-Grade, End-to-End ML Pipeline for Real-Time Sign Language Recognition on Edge Devices

<p align="center">
  <!-- Replace with actual demo GIF after recording -->
  <img src="reports/figures/demo_placeholder.png" alt="Real-time gesture recognition demo" width="720"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10.20-blue?logo=python" alt="Python 3.10"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.13.1-orange?logo=tensorflow" alt="TensorFlow"/>
  <img src="https://img.shields.io/badge/MediaPipe-0.10.14-green" alt="MediaPipe"/>
  <img src="https://img.shields.io/badge/MLflow-2.14.3-blue?logo=mlflow" alt="MLflow"/>
  <img src="https://img.shields.io/badge/Docker-ready-2496ED?logo=docker" alt="Docker"/>
  <img src="https://github.com/<your-username>/wlasl-gesture-recognition/actions/workflows/ci.yml/badge.svg" alt="CI Status"/>
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License"/>
</p>

---

## Table of Contents

1. [Project Story and Framing](#1-project-story-and-framing)
2. [Problem Statement](#2-problem-statement)
3. [Objectives](#3-objectives)
4. [System Architecture](#4-system-architecture)
5. [Project Structure](#5-project-structure)
6. [Dataset](#6-dataset)
7. [Pipeline Overview](#7-pipeline-overview)
8. [Models and Experiments](#8-models-and-experiments)
9. [Results](#9-results)
10. [Real-Time Demo](#10-real-time-demo)
11. [Quickstart — Reproduce Everything](#11-quickstart--reproduce-everything)
12. [Docker](#12-docker)
13. [Experiment Tracking with MLflow](#13-experiment-tracking-with-mlflow)
14. [Project Roadmap and KSL Adaptation](#14-project-roadmap-and-ksl-adaptation)
15. [Limitations](#15-limitations)
16. [Contributing](#16-contributing)
17. [License](#17-license)

---

## 1. Project Story and Framing

Most sign language recognition research defaults to raw video pipelines fed into large convolutional neural networks — architectures that are accurate in controlled lab conditions, but computationally prohibitive on the edge devices that real signers actually carry. A modern smartphone running a 200MB vision model at 4 frames per second is not a tool for communication. It is a prototype.

This project investigates a different hypothesis: **how far can a lightweight, landmark-based sequence model go?**

The core insight is that gesture recognition does not require pixels. A signer's intent is fully encoded in the *structure* of their hands and the *motion* of their body over time — both of which can be represented as compact sequences of skeletal coordinates. By replacing raw video frames with structured landmark representations extracted by MediaPipe Holistic, and replacing heavy vision backbones with temporal sequence models (LSTM, GRU, BiLSTM), it becomes possible to build a gesture recognition system that:

- runs in real time on a laptop CPU with no GPU,
- fits in under 5 MB as a quantised TFLite file,
- achieves validation accuracy above 70% across 35 ASL signs,
- generalises across signers it has never seen during training.

This project is simultaneously a technical contribution, a portfolio artifact, and a direct precursor to Kenyan Sign Language (KSL) recognition work — a domain where landmark-based, low-data approaches are not just a design choice but a practical necessity.

---

## 2. Problem Statement

Sign language is the primary mode of communication for tens of millions of deaf and hard-of-hearing people worldwide, yet the vast majority of digital systems — smartphones, kiosks, assistive applications — cannot interpret it. Automated gesture recognition offers a path toward accessibility, but existing approaches either demand specialist hardware, rely on models too large for mobile deployment, or are trained on datasets so narrow that they fail to generalise across signers with different styles, backgrounds, or regional dialects.

This project addresses the following core question:

> **Can a lightweight, landmark-based temporal model trained on the publicly available WLASL dataset reliably classify 35 American Sign Language gestures in real time, at inference speeds and model sizes compatible with CPU-only mobile deployment?**

The project is further motivated by its direct relevance to Kenyan Sign Language (KSL) recognition — a domain with scarce labelled data, limited compute budgets, and a strong practical need for on-device, offline-capable inference across a 500-sign vocabulary.

---

## 3. Objectives

### Primary Objectives
- Build a complete, reproducible end-to-end ML pipeline from raw video to a deployed TFLite model
- Extract MediaPipe Holistic landmarks (hands + pose, 225 values per frame) from WLASL video clips across **35 selected signs**
- Train and rigorously compare multiple temporal architectures: Dense baseline, LSTM, GRU, BiLSTM
- Achieve **≥ 70% signer-independent validation accuracy**
- Export the best-performing model as a quantised TFLite file and verify accuracy retention post-quantisation
- Deploy a real-time webcam inference demo with live prediction overlay, confidence HUD, and temporal smoothing

### Secondary Objectives
- Conduct formal ablation studies across augmentation strategies, sequence lengths, and landmark configurations
- Perform SHAP-based interpretability analysis to identify which frames and landmarks drive model predictions
- Benchmark end-to-end inference latency and model size across all model variants
- Analyse signer-independent generalisation — the most honest evaluation of real-world deployability
- Document dataset bias, failure modes, and confidence calibration behaviour
- Provide a detailed KSL adaptation roadmap grounded in the differences between ASL and KSL as linguistic systems
- Produce a portfolio-quality repository demonstrating senior-level ML engineering practices: Docker, CI/CD, MLflow, OmegaConf config management, structured logging, and a unified inference engine

---

## 4. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      INPUT LAYER                            │
│              Video File  /  Webcam Stream                   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    FRAME SAMPLER                            │
│         OpenCV VideoCapture · configurable FPS              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│               MEDIAPIPE HOLISTIC                            │
│    Left Hand (21 kp) · Right Hand (21 kp) · Pose (33 kp)   │
│              225 (x, y, z) values per frame                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│             FEATURE ENGINEERING PIPELINE                    │
│   Wrist-relative normalisation · Padding / Truncation       │
│       Spatial + Temporal Augmentation (training only)       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              SEQUENCE BUFFER  [N frames × 225]              │
│                  Fixed-length input tensor                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│            TEMPORAL CLASSIFIER (trained model)              │
│       Dense  /  LSTM  /  GRU  /  BiLSTM  + Dropout         │
│              Softmax output  (35 classes)                   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│            PREDICTION SMOOTHER                              │
│    Majority voting (window=5) · Exponential smoothing       │
│           Confidence threshold gate                         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                     OUTPUT                                  │
│    Sign label  ·  Confidence score  ·  Top-3 predictions    │
│    TFLite file  /  Real-time HUD overlay                    │
└─────────────────────────────────────────────────────────────┘
```

> **Full system architecture diagram:** `reports/figures/system_architecture.png`

---

## 5. Project Structure

```
wlasl-gesture-recognition/
│
├── data/
│   ├── raw/                          # Downloaded WLASL videos — gitignored
│   ├── landmarks/                    # Cached .npy landmark arrays — gitignored
│   └── splits/
│       ├── train.csv                 # Signer-aware training split
│       ├── val.csv                   # Signer-aware validation split
│       └── test.csv                  # Held-out test split
│
├── notebooks/
│   ├── 01_data_exploration.ipynb         # Dataset EDA, class distribution, bias analysis
│   ├── 02_landmark_inspection.ipynb      # Landmark visualisation, missing rate analysis
│   ├── 03_feature_engineering.ipynb      # Augmentation experiments, normalisation validation
│   ├── 04_model_experiments.ipynb        # Architecture prototyping, training dynamics
│   ├── 05_evaluation_error_analysis.ipynb # Full evaluation suite, failure mode analysis
│   ├── 06_tflite_verification.ipynb      # Quantisation analysis, deployment benchmarks
│   └── 07_interpretability_shap.ipynb    # SHAP analysis, landmark importance, saliency
│
├── src/
│   ├── data/
│   │   ├── downloader.py             # WLASL JSON manifest → video fetch
│   │   ├── validator.py              # Data schema + integrity validation
│   │   └── splitter.py              # Signer-aware train/val/test split
│   │
│   ├── features/
│   │   ├── extractor.py              # MediaPipe Holistic landmark extraction
│   │   ├── augmentation.py           # Spatial + temporal augmentation strategies
│   │   └── pipeline.py              # FeaturePipeline: normalise → augment → pad
│   │
│   ├── models/
│   │   ├── architectures.py          # Dense, LSTM, GRU, BiLSTM definitions
│   │   ├── factory.py               # build_model(config) factory function
│   │   └── train.py                  # Training loop with MLflow integration
│   │
│   ├── evaluation/
│   │   ├── metrics.py               # Confusion matrix, per-class F1, accuracy
│   │   ├── benchmark.py             # Latency profiling, FPS, model size
│   │   ├── calibration.py           # Confidence calibration, reliability diagram
│   │   └── signer_analysis.py       # Per-signer generalisation analysis
│   │
│   ├── export/
│   │   ├── convert.py               # SavedModel → TFLite + post-training quantisation
│   │   └── verify.py                # TFLite interpreter accuracy verification + CLI
│   │
│   ├── inference/
│   │   └── predictor.py             # GesturePredictor: unified inference engine
│   │
│   ├── demo/
│   │   └── webcam_demo.py           # Real-time OpenCV webcam demo with HUD
│   │
│   └── utils/
│       ├── config.py                # OmegaConf loader + Pydantic schema validation
│       ├── logger.py                # Structured logging (file + console)
│       ├── reproducibility.py       # Seed setting + environment metadata logging
│       └── label_map.py             # Versioned label map loading
│
├── pipelines/
│   ├── run_preprocessing.py         # CLI: extract all landmarks to .npy cache
│   ├── run_training.py              # CLI: train one experiment configuration
│   ├── run_all_experiments.py       # CLI: execute full experiment matrix
│   └── run_evaluation.py           # CLI: generate all evaluation artefacts
│
├── configs/
│   ├── base.yaml                    # Global defaults (seed, batch size, epochs)
│   ├── model/
│   │   ├── dense.yaml               # Dense feedforward baseline config
│   │   ├── lstm.yaml                # LSTM config
│   │   ├── gru.yaml                 # GRU config
│   │   └── bilstm.yaml              # Bidirectional LSTM config
│   ├── data/
│   │   ├── seq20.yaml               # 20-frame sequence config
│   │   ├── seq30.yaml               # 30-frame sequence config
│   │   └── seq40.yaml               # 40-frame sequence config
│   ├── augmentation/
│   │   ├── none.yaml                # No augmentation
│   │   ├── temporal.yaml            # Temporal augmentation only
│   │   └── spatial_temporal.yaml    # Full spatial + temporal augmentation
│   └── experiment/
│       ├── baseline.yaml            # Group 1: architecture comparison
│       ├── ablation_augmentation.yaml
│       ├── ablation_sequence.yaml
│       ├── ablation_landmarks.yaml
│       └── best_model.yaml          # Champion model config
│
├── tests/
│   ├── test_downloader.py
│   ├── test_validator.py
│   ├── test_extractor.py
│   ├── test_augmentation.py
│   ├── test_pipeline.py
│   ├── test_model_factory.py
│   └── test_predictor.py
│
├── artifacts/
│   ├── label_map_v1.json            # Versioned class index → sign name mapping
│   └── experiments/                 # Per-run config snapshots and metrics JSON
│
├── models/
│   ├── gesture_dense_v1/            # Dense baseline SavedModel
│   ├── gesture_lstm_v1/             # LSTM SavedModel
│   ├── gesture_gru_v1/              # GRU SavedModel
│   ├── gesture_bilstm_v1/           # BiLSTM SavedModel (champion)
│   ├── gesture_bilstm_v1.tflite     # Quantised TFLite export
│   └── gesture_model_metadata.json  # Input shape, labels, normalisation params
│
├── logs/                            # Runtime logs — gitignored
├── mlruns/                          # MLflow experiment store — gitignored
│
├── reports/
│   ├── figures/                     # All generated plots and visualisations
│   ├── experiment_summary.md        # Full experiment registry table
│   └── report.pdf                   # One-page technical report
│
├── .github/
│   └── workflows/
│       └── ci.yml                   # GitHub Actions: lint, test, docker build
│
├── Dockerfile                       # Training image
├── Dockerfile.inference             # Lean inference-only image
├── docker-compose.yml
├── requirements.txt                 # Production dependencies (pinned)
├── requirements-dev.txt             # Development/testing dependencies
├── Makefile                         # One-word commands for all pipeline stages
├── MODEL_CARD.md                    # Model card: use, performance, limitations
├── LIMITATIONS.md                   # Honest limitations and future work
└── README.md
```

---

## 6. Dataset

### WLASL — Word-Level American Sign Language

The [WLASL dataset](https://dxli94.github.io/WLASL/) is the largest publicly available word-level ASL video dataset, comprising over 21,000 video clips spanning 2,000+ signs performed by 119 signers. For this project, **35 signs** are selected based on three criteria: minimum clip count per sign (≥ 20 clips), visual distinctiveness from other selected signs, and diversity of motion type (static handshapes, one-handed, two-handed, and body-referenced signs).

### Selected signs (35)

| Index | Sign | Index | Sign | Index | Sign | Index | Sign | Index | Sign |
|---|---|---|---|---|---|---|---|---|---|
| 0 | book | 7 | candy | 14 | drink | 21 | go | 28 | mother |
| 1 | before | 8 | chair | 15 | eat | 22 | help | 29 | name |
| 2 | birthday | 9 | change | 16 | family | 23 | house | 30 | now |
| 3 | black | 10 | clothes | 17 | finish | 24 | know | 31 | orange |
| 4 | blue | 11 | color | 18 | friend | 25 | later | 32 | thanksgiving |
| 5 | boy | 12 | computer | 19 | girl | 26 | like | 33 | think |
| 6 | can | 13 | cousin | 20 | give | 27 | many | 34 | who |

### Data split strategy

Splits are **signer-aware**: all clips from a given signer are assigned exclusively to one split (train, val, or test). A signer's style and motion characteristics never appear in both training and evaluation. This is the methodologically correct approach — random shuffling at the clip level inflates validation accuracy by leaking signer-specific motion patterns into the evaluation set.

| Split | Clips | Signers |
|---|---|---|
| Train | ~70% | ~80% of signers |
| Val | ~15% | held-out signers |
| Test | ~15% | held-out signers |

### Feature representation

Each video is processed by MediaPipe Holistic to extract:

| Landmark group | Keypoints | Values per frame |
|---|---|---|
| Left hand | 21 | 63 (x, y, z) |
| Right hand | 21 | 63 (x, y, z) |
| Pose (body skeleton) | 33 | 99 (x, y, z) |
| **Total** | **75** | **225** |

Each video is represented as a tensor of shape `(sequence_length, 225)`, where `sequence_length` is a configurable hyperparameter (20, 30, or 40 frames).

---

## 7. Pipeline Overview

The project is organised as a sequence of composable pipeline stages. Each stage is a standalone CLI script under `pipelines/` and can be run independently or via `make`.

### Stage 1 — Data ingestion and validation
Downloads WLASL videos via the official JSON manifest. Runs a 10-point validation checklist (class presence, minimum clip counts, file integrity, signer ID availability, frame count ranges). Produces `data/data_validation_report.json` and signer-aware split CSVs.

### Stage 2 — EDA and landmark inspection
Exploratory analysis at both the raw video level (class distribution, duration histograms, signer diversity) and the feature level (landmark visualisations, missing detection rate per class, inter-signer variability). Findings directly inform sequence length and normalisation decisions.

### Stage 3 — Preprocessing pipeline
Iterates over all videos, runs MediaPipe Holistic on every frame, and caches results as `.npy` arrays. Resumable (skips existing files), fully logged, and produces a per-video processing summary. Missing landmarks are zero-filled; videos with > 30% missing frames are excluded and logged.

### Stage 4 — Feature engineering
The `FeaturePipeline` class applies wrist-relative coordinate normalisation, configurable temporal and spatial augmentation, and fixed-length sequence padding. This class is the single source of truth for preprocessing — used identically during training and inference via `GesturePredictor`.

### Stage 5 — Multi-model training with MLflow
Trains 12+ experimental variants across four experiment groups (architecture comparison, augmentation ablation, sequence length ablation, landmark configuration ablation). Every run is logged to MLflow with full parameter tracking, per-epoch metrics, and model artefacts.

### Stage 6 — Evaluation and interpretability
Produces the complete evaluation suite: confusion matrices, per-class metrics, ablation study tables, latency benchmarks, pipeline timing profiles, confidence calibration analysis, signer-independent generalisation analysis, and SHAP-based interpretability visualisations.

### Stage 7 — Unified inference engine
The `GesturePredictor` class wraps TFLite inference, preprocessing, label decoding, and temporal smoothing in a single, reusable interface. Eliminates preprocessing mismatch between training and inference.

### Stage 8 — TFLite export and verification
Converts the champion model via post-training quantisation. Verifies TFLite accuracy against the original Keras model on the full validation set. Produces `models/gesture_model_metadata.json` for downstream integration.

### Stage 9 — Real-time webcam demo
OpenCV-based real-time inference application with a polished HUD: prediction label, confidence bar, top-3 sign distribution, FPS counter, and stability indicator. Uses `GesturePredictor` — no preprocessing duplication.

---

## 8. Models and Experiments

### Experiment groups

| Group | Fixed variables | Variable |
|---|---|---|
| 1 — Architecture comparison | seq_len=30, no augmentation | Model type (Dense, LSTM, GRU, BiLSTM) |
| 2 — Augmentation ablation | LSTM, seq_len=30 | Augmentation strategy |
| 3 — Sequence length ablation | LSTM, best augmentation | Sequence length (20, 30, 40) |
| 4 — Landmark config ablation | LSTM, best aug, best seq_len | Landmark set (hands, pose, hands+pose) |

### Architecture summary

| Model | Purpose | Parameters (approx.) |
|---|---|---|
| Dense baseline | Prove temporal modelling matters | ~3.5M |
| LSTM (single layer) | Sequence baseline | ~0.5M |
| LSTM (stacked, 2-layer) | Depth vs overfitting analysis | ~0.8M |
| GRU | Speed/accuracy trade-off | ~0.6M |
| BiLSTM | Best accuracy candidate | ~1.5M |

### Primary metric

Validation accuracy on the signer-independent held-out set. For deployment candidates, secondary ranking: `accuracy / median_inference_latency_ms` — the best performance-per-millisecond trade-off.

### Experiment results

> Results table to be populated after training. See `reports/experiment_summary.md` for the full registry.

| Experiment | Val Acc | Latency (ms) | Model Size | Notes |
|---|---|---|---|---|
| dense_baseline | — | — | — | Non-temporal reference |
| lstm_baseline | — | — | — | Sequence baseline |
| gru_baseline | — | — | — | Speed candidate |
| bilstm_baseline | — | — | — | Accuracy candidate |
| bilstm_spatial_temporal_seq30 | — | — | — | Champion (expected) |
| bilstm_v1 (TFLite quantised) | — | — | — | Deployment model |

---

## 9. Results

> This section will be updated with final numbers after all experiments complete.

### Key metrics (best model)

| Metric | Value |
|---|---|
| Validation accuracy (Keras) | — |
| Validation accuracy (TFLite) | — |
| Quantisation accuracy delta | — |
| Median inference latency | — |
| TFLite model file size | — |
| Training time | — |

### Ablation study summary

> See `reports/figures/ablation_*.png` and `reports/experiment_summary.md` for full tables and analysis.

### Key findings

> To be populated after evaluation. Expected findings:
> - Dense baseline confirms temporal modelling is essential (~20pp gap vs LSTM)
> - Spatial + temporal augmentation reduces overfitting gap by ~10–15pp
> - Sequence length 30 provides best accuracy/latency trade-off
> - Hands-only landmark config is competitive with full config at ~55% of the feature size
> - BiLSTM achieves highest accuracy; GRU achieves best accuracy/latency ratio

---

## 10. Real-Time Demo

```bash
make demo
# or
python src/demo/webcam_demo.py
```

The webcam demo opens a video window with the following HUD overlay:

- **Predicted sign** — large text, top centre
- **Confidence %** — displayed beneath sign name
- **Top-3 bar chart** — right side panel, colour-coded by confidence tier
- **FPS counter** — bottom left
- **Stability indicator** — green (stable prediction), yellow (fluctuating)
- **"No hands detected" warning** — when MediaPipe returns empty landmarks for 3+ consecutive frames

Temporal smoothing (majority voting, window = 5 frames) eliminates single-frame prediction noise.

> Demo GIF: `reports/demo.gif` — embedded at the top of this README after recording.

---

## 11. Quickstart — Reproduce Everything

### Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed
- Python 3.10.20 conda environment created

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/wlasl-gesture-recognition.git
cd wlasl-gesture-recognition

# 2. Activate your conda environment
conda activate <your-env-name>

# 3. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # optional, for testing and linting

# 4. Verify installation
python -c "import tensorflow as tf; import mediapipe; print('TF:', tf.__version__)"
```

### Run the full pipeline

```bash
# Download dataset and validate
make download

# Extract landmarks (one-time operation, ~30–90 min depending on hardware)
make preprocess

# Run all experiments (tracked in MLflow)
make train

# Generate all evaluation artefacts
make evaluate

# Launch real-time webcam demo
make demo
```

### Run a single experiment

```bash
python pipelines/run_training.py \
  --model bilstm \
  --data seq30 \
  --augmentation spatial_temporal \
  --run-name bilstm_spatial_temporal_seq30
```

### Run standalone inference on a video file

```bash
python src/export/verify.py --video path/to/video.mp4
# Output:
# Predicted sign : drink
# Confidence     : 0.91
# Top 3          : drink (0.91), eat (0.06), candy (0.03)
# Latency        : 24ms end-to-end
```

### View MLflow experiment dashboard

```bash
make mlflow
# Navigate to: http://localhost:5000
```

### Run tests

```bash
make test
# or
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## 12. Docker

Full pipeline is containerised for guaranteed reproducibility. Anyone who clones this repository can reproduce all results without installing Python, TensorFlow, or MediaPipe.

```bash
# Run preprocessing
docker-compose run preprocess

# Run all experiments
docker-compose run train

# Run evaluation
docker-compose run evaluate
```

Two images are provided:

| Image | Purpose | Size (approx.) |
|---|---|---|
| `Dockerfile` | Full training environment | ~4 GB |
| `Dockerfile.inference` | Lean inference + demo only | ~800 MB |

---

## 13. Experiment Tracking with MLflow

All experiments are logged to a local MLflow tracking server. Each run records:

- All hyperparameters from the config YAML
- Per-epoch train and validation accuracy and loss
- Final confusion matrix and training curves (as artefacts)
- The trained model (as an MLflow model artefact)
- Environment metadata: TF version, MediaPipe version, Python version, OS, CUDA availability
- A config snapshot YAML for exact reproducibility

```bash
# Launch the MLflow UI
mlflow ui --host 0.0.0.0 --port 5000
# Navigate to: http://localhost:5000
```

The best model is registered in the MLflow Model Registry under `gesture-lstm-production`.

> Screenshot of MLflow dashboard: `reports/figures/mlflow_dashboard.png`

---

## 14. Project Roadmap and KSL Adaptation

### Planned extensions

- [ ] Expand from 35 to 100+ signs
- [ ] Integrate attention mechanism on top of BiLSTM for improved SHAP analysis
- [ ] Add temperature scaling for confidence calibration
- [ ] Build Android inference wrapper using TFLite Android library
- [ ] Collect KSL data and validate transfer learning strategy

### KSL adaptation roadmap

ASL and KSL are structurally distinct languages with different phonologies (hand shapes, movements, locations), different spatial grammar conventions, and different roles for facial expressions as grammatical markers. A naive transfer from ASL to KSL risks actively biasing the model toward ASL-specific patterns that have no KSL equivalent.

**Proposed validation strategy:**

1. Train a KSL-only LSTM from scratch on the AI4KSL dataset (baseline)
2. Train an ASL-pretrained model, freeze the LSTM layers, fine-tune only the classifier on KSL
3. Train an ASL-pretrained model, fine-tune all layers on KSL
4. Compare all three on a held-out KSL test set — per-class accuracy, not just overall

**Critical warning:** overall accuracy numbers can mask catastrophic failure on rare KSL signs that have no ASL equivalent. Per-class recall on low-frequency signs is the metric that matters.

**Data requirements:** At 40 clips per sign, 500-sign KSL recognition at 85% accuracy is at the boundary of what this architecture can achieve. Data augmentation can partially compensate, but collecting diverse, multi-signer data across Kenyan regions is the most reliable path to robust generalisation.

---

## 15. Limitations

See [`LIMITATIONS.md`](LIMITATIONS.md) for a full discussion. Key limitations:

- **35-sign scope:** Not production-ready for a communication tool. A full ASL vocabulary is 2,000+ signs.
- **Signer-independent accuracy gap:** Model performs measurably worse on signers it has never seen. The gap is documented and analysed in the evaluation.
- **Lighting and background sensitivity:** Performance degrades under poor lighting or cluttered backgrounds. MediaPipe robustness is the limiting factor.
- **Confidence overconfidence:** Softmax outputs are not well-calibrated probabilities. Temperature scaling is recommended before production deployment.
- **MediaPipe dependency at inference:** Real-time inference requires MediaPipe to run on-device, adding ~18ms per frame to the pipeline latency.
- **KSL data scarcity:** Approximately 40 clips per sign is below the recommended minimum for reliable recognition at 500-sign scale.

---

## 16. Contributing

This project is under active development. Contributions, issues, and suggestions are welcome.

```bash
# Development setup
pip install -r requirements-dev.txt
pre-commit install  # installs git hooks for linting

# Run tests before submitting
make test
make lint
```

Please open an issue before submitting a pull request for significant changes.

---

## 17. License

This project is licensed under the MIT License — see [`LICENSE`](LICENSE) for details.

The WLASL dataset is subject to its own licence terms. See the [WLASL repository](https://github.com/dxli94/WLASL) for details. This project does not redistribute any WLASL video content.

---

<p align="center">
  Built as part of a sign language recognition research initiative. <br/>
  For questions about KSL adaptation, open an issue or reach out directly.
</p>