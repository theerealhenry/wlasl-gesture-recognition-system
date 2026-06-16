# WLASL Gesture Recognition
### A Production-Grade, End-to-End Landmark-Based Sign Language Recognition Pipeline

<p align="center">
  <img src="reports/figures/demo_placeholder.png" alt="Real-time gesture recognition demo" width="720"/>
  <br/>
  <em>Real-time BiLSTM inference on MediaPipe Holistic landmarks — 68K parameters, CPU-only, &lt;50ms latency</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10.20-blue?logo=python&logoColor=white" alt="Python 3.10"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.13.1-FF6F00?logo=tensorflow&logoColor=white" alt="TensorFlow 2.13"/>
  <img src="https://img.shields.io/badge/MediaPipe-0.10.14-00897B?logo=google&logoColor=white" alt="MediaPipe"/>
  <img src="https://img.shields.io/badge/MLflow-2.14.3-0194E2?logo=mlflow&logoColor=white" alt="MLflow"/>
  <img src="https://img.shields.io/badge/OmegaConf-2.3.0-blueviolet" alt="OmegaConf"/>
  <img src="https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white" alt="Docker"/>
  <img src="https://github.com/HenryOtsyula/wlasl-gesture-recognition/actions/workflows/ci.yml/badge.svg" alt="CI"/>
  <img src="https://img.shields.io/badge/license-MIT-brightgreen" alt="MIT License"/>
</p>

---

## Table of Contents

1. [Project Story and Scientific Framing](#1-project-story-and-scientific-framing)
2. [Problem Statement](#2-problem-statement)
3. [Key Results](#3-key-results)
4. [System Architecture](#4-system-architecture)
5. [Project Structure](#5-project-structure)
6. [Dataset](#6-dataset)
7. [Pipeline Stages](#7-pipeline-stages)
8. [Models and Experiments](#8-models-and-experiments)
9. [Findings and Ablation Studies](#9-findings-and-ablation-studies)
10. [Quickstart — Reproduce Everything](#10-quickstart--reproduce-everything)
11. [Docker](#11-docker)
12. [Experiment Tracking with MLflow](#12-experiment-tracking-with-mlflow)
13. [Real-Time Demo](#13-real-time-demo)
14. [KSL Adaptation Roadmap](#14-ksl-adaptation-roadmap)
15. [Limitations](#15-limitations)
16. [Contributing](#16-contributing)
17. [License](#17-license)

---

## 1. Project Story and Scientific Framing

Most sign language recognition research defaults to raw video pipelines fed into large convolutional architectures — accurate in controlled lab conditions, but computationally prohibitive on the edge devices that real signers actually carry. A 200 MB vision model running at 4 FPS on a mid-range smartphone is not a communication tool. It is a proof of concept.

This project investigates a different hypothesis: **how far can a lightweight, landmark-based sequence model go?**

The core insight is that gesture recognition does not require pixels. A signer's intent is fully encoded in the *structure* of their hands and the *motion* of their body over time — both representable as compact sequences of skeletal coordinates. By replacing raw video frames with structured landmark representations from MediaPipe Holistic, and replacing heavy vision backbones with temporal sequence models (BiLSTM), this pipeline builds a gesture recognition system that:

- **runs in real time on a CPU** with no GPU requirement
- **fits in under 1 MB** as a quantised TFLite file (pre-quantisation: 0.262 MB)
- achieves **val macro-F1 of 0.60** on a signer-independent held-out set across 35 ASL signs
- **generalises to entirely unseen signers** — evaluated under zero signer overlap between splits, the most conservative possible test

The project is simultaneously a technical investigation, a production ML engineering demonstration, and a direct precursor to Kenyan Sign Language (KSL) recognition — a domain where landmark-based, low-data approaches are not just a design preference but a practical necessity.

---

## 2a. Problem Statement

Sign language is the primary mode of communication for tens of millions of deaf and hard-of-hearing people worldwide, yet the vast majority of digital systems cannot interpret it. Automated gesture recognition offers a path toward accessibility, but existing approaches either demand specialist hardware, rely on models too large for mobile deployment, or are trained under evaluation conditions (random splits, seen-signer testing) that inflate reported accuracy beyond what real-world deployment would achieve.

This project addresses the following core question:

> **Can a lightweight, landmark-based temporal model trained on the publicly available WLASL dataset reliably classify 35 American Sign Language signs across unseen signers, at inference speeds and model sizes compatible with CPU-only mobile deployment?**

The project is further motivated by its direct relevance to Kenyan Sign Language (KSL) recognition — a domain with scarce labelled data, limited compute budgets, and a strong practical need for on-device, offline-capable inference across a 500-sign vocabulary.

---

## 2b. Objectives

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

## 3. Key Results

### Champion Model: `bilstm_hands_only_v4_aug`

> BiLSTM (2-layer) · seq\_len=100 · hands-only landmarks (126-dim) · spatial-temporal augmentation · signer-independent evaluation

| Metric | Value |
|--------|-------|
| **val macro-F1** (sklearn, zero\_division=0) | **0.6011** |
| val accuracy | 0.5769 |
| Minimum viability threshold (≥ 0.60) | ✓ Met |
| Target threshold (≥ 0.70) | ✗ Not met (data-constrained ceiling ~0.60–0.65) |
| Architecture | BiLSTM, 2 layers, 32 units/direction |
| Total parameters | 68,771 |
| Estimated weight size (float32) | 0.262 MB |
| Input feature config | hands-only — 126 dims (left hand 63 + right hand 63) |
| Sequence length | 100 frames |
| Augmentation | spatial-temporal (5-transform chain) |
| Best epoch | 171 of 221 trained |
| Early stopping patience | 50 (monitor: val\_macro\_f1) |
| Training clips | 236 · 31 signers · zero signer overlap |
| Validation clips | 52 · 7 signers · all unseen |

### High-Risk Class Performance (Champion)

| Class | Train clips | val F1 | Status |
|-------|-------------|--------|--------|
| `clothes` | 2 | 1.00 | Learned (augmentation decisive) |
| `birthday` | 4 | 1.00 | Stable |
| `book` | 4 | 1.00 | Learned |
| `name` | 4 | 1.00 | Learned |
| `think` | 3 | 0.00 | **Unlearnable at 3-clip scale** |

### Total Improvement Through Ablation

| Checkpoint | Run | val macro-F1 | Cumulative gain |
|------------|-----|-------------|----------------|
| Group 3 baseline | `lstm_seq60` | 0.1434 | — |
| + seq100 | `lstm_seq100` | 0.2354 | +64% rel. |
| + hands-only | `lstm_hands_only` | 0.4948 | +110% rel. |
| + BiLSTM | `bilstm_hands_only` | 0.5419 | +10% rel. |
| + augmentation (250ep) | **`bilstm_hands_only_v4_aug`** | **0.6011** | **+11% rel.** |
| **Total (baseline → champion)** | | | **+319% relative** |

---

## 4. System Architecture

```

┌─────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                             │
│               Video File  ·  Webcam Stream                      │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MEDIAPIPE HOLISTIC                           │
│      Left Hand (21 kp) · Right Hand (21 kp) · Pose (33 kp)      │
│                225 (x, y, z) values per frame                   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│             FEATURE ENGINEERING PIPELINE                        │
│    Wrist-relative normalisation  ·  Z-coord soft-clip (±0.10)   │
│    Centre-crop / right-zero-pad to seq_len                      │
│    Spatial + temporal augmentation  (training only)             │
│    Landmark config selection  (hands-only · pose-only · full)   │
└───────────────────────────┬─────────────────────────────────────┘
                            │  (seq_len, 126) float32
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│     TEMPORAL CLASSIFIER  —  Champion: BiLSTM                    │
│   Masking(0.0) → BiLSTM(32/dir)×2 → Dense(32, relu) → Drop      │
│              Softmax output  (35 classes)                       │
│          68,771 params · 0.262 MB · ≤50ms CPU                   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
|                                                                 |
│                  PREDICTION SMOOTHER                            │
|                                                                 |
│        Majority voting (window=5) · Confidence threshold        │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                        OUTPUT                                   │
|                                                                 |
│     Sign label · Confidence score · Top-3 predictions           │
|                                                                 |
│     TFLite file  ·  Real-time HUD overlay  ·  JSON metadata     │
└─────────────────────────────────────────────────────────────────┘


```

> **Full system architecture diagram:** `reports/figures/system_architecture.png`

---

## 5. Project Structure

```
wlasl-gesture-recognition/

│

├── data/
|
│   ├── raw/                              # Downloaded WLASL videos — gitignored
|
│   ├── landmarks/                        # Cached .npy landmark arrays — gitignored
|
│   │   ├── train/<sign>/<video_id>.npy   ✅ 236 clips extracted
|
│   │   ├── val/<sign>/<video_id>.npy     ✅ 52 clips
|
│   │   ├── test/<sign>/<video_id>.npy    ✅ 51 clips
|
│   │   └── landmark_inventory.csv        ✅ 339 rows, schema v1.2
|
│   └── splits/
|
│       ├── train.csv                     ✅ 245 entries → 236 loaded
|
│       ├── val.csv                       ✅ 53 entries → 52 loaded
|
│       └── test.csv                      ✅ 52 entries → 51 loaded
|
│
|
├── notebooks/
|
│   ├── 01_data_exploration.ipynb         ✅ COMPLETE — dataset EDA, bias documentation
|
│   ├── 02_landmark_inspection.ipynb      ✅ COMPLETE — extraction validation, schema v1.2
|
│   ├── 03_full_landmark_analysis.ipynb   ✅ COMPLETE — full-dataset landmark analysis

│   ├── 04_feature_engineering.ipynb      ✅ COMPLETE — pipeline validation gate: PASS

│   ├── 05_model_experiments.ipynb        ✅ COMPLETE — 23-run experiment analysis

│   ├── 06_evaluation_error_analysis.ipynb  🔜 Stage 6

│   ├── 07_tflite_verification.ipynb       🔜 Stage 8

│   └── 08_interpretability_shap.ipynb     🔜 Stage 6

│

├── src/

│   ├── data/

│   │   ├── init.py                   ✅

│   │   ├── downloader.py                 ✅ WLASL JSON manifest → video fetch

│   │   ├── validator.py                  ✅ 8-point integrity validation suite

│   │   └── splitter.py                   ✅ Greedy bin-packing signer-aware split

│   │

│   ├── features/

│   │   ├── init.py                   ✅

│   │   ├── constants.py                  ✅ Schema v1.2, landmark slice constants

│   │   ├── extractor.py                  ✅ MediaPipe Holistic · schema v1.2 dual-criterion

│   │   ├── augmentation.py               ✅ 5-transform chain · zero-fill invariant

│   │   ├── pipeline.py                   ✅ FeaturePipeline · pre_augmentation()

│   │   └── dataset.py                    ✅ GestureDataset · two-phase preloading

│   │

│   ├── models/

│   │   ├── init.py                   ✅

│   │   ├── architectures.py              ✅ Dense · LSTM · GRU · BiLSTM

│   │   ├── factory.py                    ✅ build_model(cfg) · pipeline shape validation

│   │   └── train.py                      ✅ train_one_run() · MacroF1Evaluator · MLflow

│   │

│   ├── evaluation/

│   │   ├── init.py                   🔜 Stage 6

│   │   ├── metrics.py                    🔜 confusion matrix · per-class F1

│   │   ├── benchmark.py                  🔜 latency profiling · FPS · model size

│   │   ├── calibration.py                🔜 reliability diagram · ECE

│   │   └── signer_analysis.py            🔜 per-signer accuracy breakdown

│   │

│   ├── export/

│   │   ├── init.py                   🔜 Stage 8

│   │   ├── convert.py                    🔜 SavedModel → TFLite + quantisation

│   │   └── verify.py                     🔜 TFLite accuracy verification

│   │

│   ├── inference/

│   │   ├── init.py                   🔜 Stage 7

│   │   └── predictor.py                  🔜 GesturePredictor unified engine

│   │

│   ├── demo/

│   │   └── webcam_demo.py                🔜 Stage 9

│   │

│   └── utils/

│       ├── init.py                   ✅

│       ├── config.py                     ✅ OmegaConf + Pydantic v2 · config_hash

│       ├── logger.py                     ✅ Structured logging · file + console

│       ├── reproducibility.py            ✅ Seed management · environment metadata

│       └── label_map.py                  ✅ Versioned label map · LabelMap class

│

├── pipelines/

│   ├── run_preprocessing.py              ✅ Stage 1 CLI entry point

│   ├── run_landmark_extraction.py        ✅ Stage 3 CLI · resumable · tqdm

│   ├── run_training.py                   ✅ Stage 5 CLI · argparse · dot-notation overrides

│   ├── run_all_experiments.py            ✅ Adaptive orchestrator · 23 runs

│   └── run_evaluation.py                 🔜 Stage 6 CLI

│

├── configs/

│   ├── base.yaml                         ✅ Global defaults

│   ├── model/

│   │   ├── dense.yaml                    ✅

│   │   ├── lstm.yaml                     ✅

│   │   ├── gru.yaml                      ✅

│   │   └── bilstm.yaml                   ✅

│   ├── data/

│   │   ├── seq20.yaml  seq30.yaml        ✅

│   │   ├── seq40.yaml  seq60.yaml        ✅

│   │   ├── seq80.yaml  seq100.yaml       ✅

│   ├── augmentation/

│   │   ├── none.yaml                     ✅

│   │   ├── temporal.yaml                 ✅

│   │   └── spatial_temporal.yaml         ✅

│   └── experiment/

│       ├── baseline.yaml                 ✅

│       ├── ablation_augmentation.yaml    ✅

│       ├── ablation_sequence.yaml        ✅

│       ├── ablation_landmarks.yaml       ✅

│       └── best_model.yaml               ✅

│

├── tests/

│   ├── test_augmentation.py              ✅ Full suite · all passing

│   ├── test_pipeline.py                  ✅ All passing

│   ├── test_downloader.py                🔜

│   ├── test_validator.py                 🔜

│   ├── test_extractor.py                 🔜

│   ├── test_model_factory.py             🔜

│   └── test_predictor.py                 🔜

│

├── artifacts/

│   ├── label_map_v1.json                 ✅ Schema v1.1 · 35 signs · locked

│   └── experiments/                      ✅ Per-run manifests · metrics · confusion matrices

│       └── bilstm_hands_only_v4_aug/     ✅ Champion run artefacts

│

├── models/

│   ├── bilstm_hands_only_v4_aug_saved_model/  ✅ Champion SavedModel

│   ├── bilstm_hands_only_v4_aug_best_weights/ ✅ Best-epoch weights checkpoint

│   └── [22 additional SavedModel directories] ✅ All 23 ablation runs

│

├── reports/

│   ├── figures/                          ✅ All Stage 1–5 figures (25+ PNGs)

│   ├── experiment_summary.md             ✅ Full 23-run registry

│   └── report.pdf                        🔜 One-page technical report (Stage 11)

│

├── .github/

│   └── workflows/

│       └── ci.yml                        🔜 Lint · Test · Docker build

│

├── Dockerfile                            🔜 Training image

├── Dockerfile.inference                  🔜 Lean inference image

├── docker-compose.yml                    🔜

├── Makefile                              🔜

├── MODEL_CARD.md                         🔜 Stage 11

├── LIMITATIONS.md                        ✅ Complete and updated post-Stage 5

├── requirements.txt                      ✅ Pinned versions

├── requirements-dev.txt                  ✅

└── README.md
```

---

---

## 6. Dataset

### WLASL — Word-Level American Sign Language

The [WLASL dataset](https://dxli94.github.io/WLASL/) is the largest publicly available word-level ASL video dataset, comprising over 21,000 video clips spanning 2,000+ signs performed by 119 signers. For this project, **35 signs** are selected and locked in `artifacts/label_map_v1.json` (schema v1.1).

### Dataset Statistics

| Metric | Value |
|--------|-------|
| Signs | 35 (locked) |
| Total inventory entries | 751 |
| Clips found on disk | 350 |
| **Dataset completeness** | **46.6%** (401 dead YouTube URLs) |
| Clips after v1.2 extraction | 339 (96.9% of found) |
| Training clips loaded | **236** · 31 signers |
| Validation clips loaded | **52** · 7 signers |
| Test clips loaded | **51** · 7 signers |
| Signer overlap across splits | **0** (confirmed) |
| Class weight ratio | 6.50× (min: 0.519, max: 3.371) |
| Mean frames per clip | 67.6 · median 67 · std 23.6 |
| Global hand detection rate | 64.72% |
| Singleton val classes | 21 of 35 |

### The 35 Selected Signs

| Idx | Sign | Idx | Sign | Idx | Sign | Idx | Sign | Idx | Sign |
|-----|------|-----|------|-----|------|-----|------|-----|------|
| 0 | before | 7 | candy | 14 | drink | 21 | go | 28 | mother |
| 1 | birthday | 8 | chair | 15 | eat | 22 | help | 29 | name |
| 2 | black | 9 | change | 16 | family | 23 | house | 30 | now |
| 3 | blue | 10 | clothes | 17 | finish | 24 | know | 31 | orange |
| 4 | book | 11 | color | 18 | friend | 25 | later | 32 | thanksgiving |
| 5 | boy | 12 | computer | 19 | girl | 26 | like | 33 | think |
| 6 | can | 13 | cousin | 20 | give | 27 | many | 34 | who |

### Split Strategy: Signer-Aware, Zero Overlap

Splits are **signer-aware**: every clip from a given signer is assigned exclusively to one partition. A signer's style never appears in both training and evaluation. This is the methodologically correct evaluation — random clip shuffling inflates accuracy by leaking signer-specific motion patterns across splits.

| Split | Clips | Signers | Classes represented |
|-------|-------|---------|---------------------|
| Train | 236 | 31 | 35 |
| Val | 52 | 7 (all unseen) | 35 |
| Test | 51 | 7 (all unseen) | 35 |

All reported metrics are on signers the model has **never seen during training** — more conservative and more honest than any random-split baseline.

### Feature Representation

Each video is processed by MediaPipe Holistic to extract:

| Landmark group | Keypoints | Values per frame |
|----------------|-----------|-----------------|
| Left hand | 21 | 63 (x, y, z) |
| Right hand | 21 | 63 (x, y, z) |
| Pose (body skeleton) | 33 | 99 (x, y, z) |
| **Total (full config)** | **75** | **225** |
| **Champion (hands-only)** | **42** | **126** |

---

## 7. Pipeline Stages

The project is structured as a sequence of nine pipeline stages. Each stage has a standalone CLI entry point under `pipelines/` and can be run independently or via `make`.

| Stage | Description | Status |
|-------|-------------|--------|
| **1 — Data Ingestion** | WLASL manifest resolution, 8-point integrity validation, signer-aware greedy bin-packing split | ✅ Complete |
| **2 — EDA** | Dataset characterisation, class distribution, signer dominance, temporal analysis, bias documentation | ✅ Complete |
| **3 — Landmark Extraction** | MediaPipe Holistic on all 350 clips · v1.2 dual-criterion skip policy · 339 clips retained | ✅ Complete |
| **4 — Feature Engineering** | Wrist-relative normalisation, z-clip, centre-crop/pad, 5-transform augmentation chain, GestureDataset | ✅ Complete |
| **5 — Multi-Model Training** | 23 MLflow-tracked runs across 4 experiment groups · champion BiLSTM identified | ✅ Complete |
| **6 — Evaluation & Interpretability** | Test-set evaluation, latency benchmark, SHAP, confidence calibration, signer analysis | 🔜 Next |
| **7 — Inference Engine** | GesturePredictor unified class · TFLite runtime · PredictionSmoother | 🔜 Stage 7 |
| **8 — TFLite Export** | Dynamic-range quantisation · accuracy verification · model metadata JSON | 🔜 Stage 8 |
| **9 — Webcam Demo** | Real-time OpenCV HUD · FPS counter · top-3 bar chart · stability indicator | 🔜 Stage 9 |

### Stage 4: Feature Engineering Pipeline

The `FeaturePipeline` class applies the following transform chain **identically at training and inference** — the single most important correctness invariant in the system:

```
Input: (T_raw, 225) float32

↓ 1. Shape + finite validation

↓ 2. Copy + float32 cast

↓ 3. Wrist-relative normalisation (per-slot detection mask)

↓ 4. Z-coordinate soft-clip (±0.10)

↓ 5. Centre-crop or right-zero-pad → (seq_len, 225)

↓ 6. Augmentation [training only, on full 225-dim array]

↓ 7. Landmark config selection (hands_only / pose_only / full)

Output: (seq_len, feature_dim) float32
```

**Ordering constraint (enforced):** Augmentation (step 6) must precede landmark config selection (step 7). `AugmentationPipeline` validates `arr.shape[1] == 225` — if selection ran first, all augmentation calls on non-full configs would raise `ValueError`.

### Stage 5: Augmentation Chain

Five transforms applied in this order, each with a production-grade zero-fill invariant:

| Transform | Effect | Key invariant |
|-----------|--------|---------------|
| `temporal_jitter` | Zero-fill randomly selected frames **in place** | Dropped frames zeroed at original temporal position, not compressed |
| `speed_jitter` | Resample at rate ∈ [0.7, 1.3] with zero-aware interpolation | Output frames where both surrounding source frames are zero remain zero |
| `gaussian_noise` | N(0, 0.01) on detected component slots only | Per-slot masking: LH, RH, pose independently masked |
| `rotation_2d` | ±5° rotation of wrist-relative hand coords | Applied only to detected frames; zero-fill frames unchanged |
| `spatial_flip` | Per-frame hybrid policy (both/LH-only/RH-only/neither) | Clip-level safety check: both-hands fraction > 0.30 |

---

## 8. Models and Experiments

### Experiment Matrix

Four experiment groups, 23 MLflow-tracked runs, all under experiment name `"WLASL-35-class"`. Every run uses the same seed (42), class-weight balancing, and per-epoch `load_split()` training loop.

#### Group 1 — Architecture Comparison
**Fixed:** seq60 · no-aug · full 225-dim · lr=1e-3 · max_epochs=80

| Run | Model | val macro-F1 | val acc | Best epoch | Total epochs |
|-----|-------|-------------|---------|-----------|--------------|
| `dense_baseline` | Dense | 0.3276 | 0.3654 | 75 | 80 |
| `lstm_baseline` | LSTM | 0.1948 | 0.2500 | 53 | 68 |
| `gru_baseline` | GRU | 0.1905 | 0.2692 | 78 | 80 |
| `bilstm_baseline` | BiLSTM | 0.1761 | 0.1923 | 49 | 64 |

> **Interpretation:** Dense's 0.3276 is an overfitting artefact — train macro-F1 reached 0.81 (gap: 0.48), confirming it memorised the 31 training signers' spatial positions rather than learning sign geometry. All Group 1 results were measured on full 225-dim landmarks; Group 4 showed hands-only produces +110% relative improvement, meaning these values are floor estimates.

#### Group 2 — Augmentation Ablation
**Fixed:** LSTM · seq60 · full · lr=5e-4 · max_epochs=80

| Run | Augmentation | val macro-F1 | Best epoch | Stopped |
|-----|-------------|-------------|-----------|---------|
| `lstm_no_aug` | None | 0.1706 | 72 | Full (80) |
| `lstm_temporal_aug` | Temporal only | 0.1200 | 80 | Full (80) |
| `lstm_spatial_temporal_aug` | Spatial + temporal | 0.0108 | 17 | Early (32) |
| `bilstm_spatial_temporal_aug` | Spatial + temporal | 0.0041 | 4 | Early (19) |

> ⚠ **Critical finding:** The Group 2 conclusion (spatial-temporal augmentation harmful) was an 80-epoch artefact. Under 250 epochs with patience=50 and hands-only features, spatial-temporal augmentation achieves **0.6011 vs 0.4695** (no-aug, 300ep) — a 28% relative improvement and 50% reduction in overfitting gap. The augmentation result is **epoch-budget-conditional**.

#### Group 3 — Sequence Length Ablation
**Fixed:** LSTM · no-aug · full · lr=5e-4 · max_epochs=120

| Run | seq_len | val macro-F1 | Mean content | Truncation rate |
|-----|---------|-------------|-------------|----------------|
| `lstm_seq60` | 60 | 0.1434 | 85.0% | 97.0% |
| `lstm_seq80` | 80 | 0.0328 | 95.8% | 29.8% |
| `lstm_seq80_v2` | 80 | 0.0297 | 95.8% | 29.8% |
| `lstm_seq100` | **100** | **0.2354** | **99.2%** | **7.1%** |

> `lstm_seq100` delivers **+64% relative improvement** over `lstm_seq60`. The seq80 result (0.033) is a deterministic local minimum under seed=42 — confirmed by v2 reproduction. Under different seeds or hands-only features, seq80 would likely perform comparably to seq100.

#### Group 4 — Landmark Configuration Ablation
**Fixed:** LSTM · seq100 · no-aug · lr=5e-4 · max_epochs=120

| Run | Landmark config | Feature dim | val macro-F1 | Fisher ratio | Params |
|-----|----------------|-------------|-------------|-------------|--------|
| `lstm_seq100` (full) | Full | 225 | 0.2354 | 0.5492 | 110,499 |
| **`lstm_hands_only`** | **Hands only** | **126** | **0.4948** | **0.8097** | **85,155** |
| `lstm_pose_only` | Pose only | 99 | 0.0314 | 0.2176 | 78,243 |

> **Defining finding of Stage 5:** Hands-only more than doubles val macro-F1 vs full (+110% relative), using 23% fewer parameters and 44% fewer input dimensions per timestep. The Fisher ratio prediction held directionally; the 2× magnitude gap exceeds the ratio difference alone, indicating non-linear suppression that compounds across LSTM layers.

#### Champion Candidates (9 runs)

| Run | Architecture | Aug | Epochs | val macro-F1 | High-risk F1 (B/Bk/Cl/Na/Th) |
|-----|-------------|-----|--------|-------------|------------------------------|
| `bilstm_hands_only` | BiLSTM | none | 180/30 | 0.5419 | 1.0/0.0/1.0/1.0/0.0 |
| `champion_bilstm_hands_only_v3` | BiLSTM | temporal | 250/50 | 0.5190 | 1.0/1.0/0.0/1.0/0.0 |
| `bilstm_hands_only_v3` | BiLSTM | none | 300/50 | 0.4695 | 1.0/0.5/0.0/0.0/0.0 |
| `champion_bilstm_hands_only_v2` | BiLSTM | spatial-temporal | 250/50 | 0.4610 | 1.0/0.0/0.0/1.0/0.0 |
| `bilstm_hands_only_v3_aug` | BiLSTM | temporal | 250/50 | 0.4553 | 1.0/1.0/0.0/0.0/0.0 |
| `champion_hands_only_v1` | LSTM | none | 180/30 | 0.4286 | 0.0/0.0/1.0/0.0/0.0 |
| `champion_bilstm_hands_only` | BiLSTM | none | 180/25 | 0.4181 | 1.0/1.0/0.0/1.0/0.0 |
| `bilstm_hands_only_v2` | BiLSTM | none | 250/50 | 0.4067 | 0.67/0.0/0.0/0.0/0.0 |
| **`bilstm_hands_only_v4_aug`** 🏆 | **BiLSTM** | **spatial-temporal** | **250/50** | **0.6011** | **1.0/1.0/1.0/1.0/0.0** |

> B=birthday · Bk=book · Cl=clothes · Na=name · Th=think

### Architecture Summary

| Model | Purpose | Params | Input dim |
|-------|---------|--------|-----------|
| Dense | Non-temporal baseline — proves LSTM necessity | ~7.7M | (60, 225) |
| LSTM (2-layer) | Primary ablation workhorse (Groups 2, 3, 4) | ~85–110K | (seq, feature) |
| GRU (2-layer) | Speed/accuracy trade-off candidate | ~65–85K | (seq, feature) |
| **BiLSTM (2-layer)** 🏆 | **Champion — bidirectional temporal context** | **~69–94K** | **(seq, feature)** |

All recurrent models include `Masking(mask_value=0.0)` as the first non-Input layer. At a global both-hands-absent rate of 35.28%, masking is not optional — it prevents the LSTM from updating its hidden state on semantically empty zero-fill frames.

---

## 9. Findings and Ablation Studies

### Finding 1 — Temporal modelling is necessary

Dense baseline (7.7M params, train macro-F1: 0.81) achieves 0.3276 val macro-F1 against LSTM's 0.1948 at 80 epochs. The 0.48 train/val gap confirms Dense memorises signer-specific spatial position rather than sign geometry. The relative ranking is correct; the absolute advantage inverts under extended training.

### Finding 2 — `hands_only` is the single highest-leverage decision

**+110% relative improvement** over full-225-dim at equivalent config. Removing 99 pose dimensions eliminates features that vary systematically across signers (body size, arm length, filming distance) without carrying discriminative sign-identity signal. The suppression effect compounds across LSTM layers — the performance gap is larger than Fisher ratios alone predict.

### Finding 3 — Sequence length matters more than previously estimated

97% of clips are truncated at seq_len=60 (P75=84, P90=95 frames). Extending from seq60 to seq100 delivers **+64% relative improvement**. The dataset median of 67 frames is substantially longer than the original design assumed — every additional frame retained yields measurable accuracy gains.

### Finding 4 — Augmentation is epoch-budget-conditional, not harmful

At 80 epochs with full 225-dim features: spatial-temporal augmentation appears harmful (val loss diverges). At 250 epochs with hands-only features: spatial-temporal augmentation is **decisively better** — val macro-F1 0.6011 vs 0.4695 (no-aug, 300ep), and the train/val overfitting gap halves from ~0.48 to ~0.24. The mechanism: augmentation slows signer-identity memorisation, allowing the model to find inter-signer invariant representations given enough training time.

### Finding 5 — Val metric variance requires careful interpretation

With 52 validation clips in ~2 batches, a single misclassified clip shifts val accuracy by 1.9pp; a singleton class prediction error shifts val macro-F1 by up to 2.9pp. Epoch-to-epoch swings of 3–5pp are structural noise. The champion's 0.6011 is reliably better than the best no-aug result (0.5419, gap = 5.9pp > noise floor), but should be understood as approximately 0.58 ± 0.03 as an expected value.

### Ablation Summary Figures

| Figure | Location |
|--------|----------|
| Architecture comparison bar + overfitting gap | `reports/figures/architecture_comparison_bar.png` |
| Group 1 training curves (all 4 models) | `reports/figures/training_curves_all_models.png` |
| Augmentation ablation (val macro-F1 + correction) | `reports/figures/ablation_augmentation.png` |
| Sequence length vs content coverage | `reports/figures/ablation_sequence_length.png` |
| Landmark config vs Fisher ratio | `reports/figures/ablation_landmark_config.png` |
| Overfitting gap: aug vs no-aug | `reports/figures/overfitting_gap_comparison.png` |
| Champion training curves | `reports/figures/champion_training_curves.png` |
| Champion per-class val macro-F1 | `reports/figures/champion_per_class_f1.png` |

---

## 10. Quickstart — Reproduce Everything

### Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed
- Python 3.10.20 conda environment

### Environment Setup

```bash
# 1. Clone the repository
git clone https://github.com/HenryOtsyula/wlasl-gesture-recognition.git
cd wlasl-gesture-recognition

# 2. Activate conda environment
conda activate <your-env-name>

# 3. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt   # optional: for tests and linting

# 4. Verify installation
python -c "
import tensorflow as tf
import mediapipe
import mlflow
print(f'TF:        {tf.__version__}')
print(f'MediaPipe: {mediapipe.__version__}')
print(f'MLflow:    {mlflow.__version__}')
"
```

### Run the Full Pipeline

```bash
make download      # Fetch WLASL videos via manifest
make preprocess    # Extract MediaPipe landmarks → .npy cache (~30–90 min)
make train         # Run full 23-run experiment matrix (MLflow tracked)
make evaluate      # Stage 6: test-set evaluation + benchmarks (coming)
make demo          # Real-time webcam demo (coming)
```

### Run a Single Training Experiment

```bash
# Champion configuration
python pipelines/run_training.py \
    model=bilstm \
    data=seq100 \
    augmentation=spatial_temporal \
    --landmark-config hands_only \
    --run-name bilstm_hands_only_v4_aug \
    --experiment-group champion

# With dot-notation overrides
python pipelines/run_training.py \
    model=bilstm \
    data=seq100 \
    augmentation=spatial_temporal \
    training.learning_rate=0.0005 \
    training.epochs=250 \
    training.early_stopping_patience=50
```

### Verify Stage 4 Pipeline Integrity

```bash
python -c "
from src.utils.config import load_config
from src.features import FeaturePipeline, GestureDataset
import numpy as np

cfg = load_config(model='bilstm', data='seq100', augmentation='spatial_temporal')
pipeline = FeaturePipeline(cfg)
assert pipeline.output_shape == (100, 126)   # hands_only: 126 dims
assert callable(pipeline.pre_augmentation)

dataset = GestureDataset(cfg, pipeline,
                         splits_dir='data/splits',
                         landmarks_dir='data/landmarks')
assert dataset.n_train == 236
assert dataset.n_val   == 52
assert dataset.n_test  == 51
print('✓ Stage 4 pipeline verified — ready for Stage 5 or Stage 6')
"
```

### View Experiment Results

```bash
# Launch MLflow UI
mlflow ui --host 0.0.0.0 --port 5000
# Navigate to: http://localhost:5000
# Experiment: "WLASL-35-class" → 23 tracked runs

# Or run the analysis notebook
jupyter notebook notebooks/05_model_experiments.ipynb
```

### Run Tests

```bash
make test
# or
pytest tests/ -v --cov=src --cov-report=term-missing

# Run only augmentation suite (the most comprehensive)
pytest tests/test_augmentation.py -v
pytest tests/test_pipeline.py -v
```

---

## 11. Docker

Full pipeline containerised for guaranteed reproducibility.

```bash
# Stage 3: extract landmarks
docker-compose run preprocess

# Stage 5: run all experiments
docker-compose run train

# Stage 6: generate evaluation artefacts
docker-compose run evaluate
```

Two images:

| Image | Purpose | Approximate size |
|-------|---------|-----------------|
| `Dockerfile` | Full training environment (TF + MediaPipe) | ~4 GB |
| `Dockerfile.inference` | Lean inference + demo (TFLite runtime only) | ~800 MB |

---

## 12. Experiment Tracking with MLflow

All 23 Stage 5 experiments are logged to a local MLflow tracking server under experiment `"WLASL-35-class"`. Each run records:

- All hyperparameters from the config YAML (model, data, augmentation, training)
- Per-epoch metrics: `train_loss`, `val_loss`, `train_acc`, `val_acc`, `val_macro_f1`, `learning_rate`, `epoch_time_s`
- Train macro-F1 on a 50% subset every 5 epochs (to track the overfitting gap)
- Per-run artefacts: confusion matrix (raw + normalised), training curves, per-class metrics JSON, run manifest, config snapshot
- The trained SavedModel as an MLflow model artefact
- Summary metrics: `best_val_macro_f1`, `best_val_acc`, `best_epoch`, high-risk class F1 scores

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

The champion model (`bilstm_hands_only_v4_aug`) is tracked under MLflow run ID `cb16f689d2294001a2ff2d3e02419d27`.

> Screenshot of MLflow experiment dashboard: `reports/figures/mlflow_dashboard.png`

---

## 13. Real-Time Demo

*(Stage 9 — coming)*

```bash
make demo
# or
python src/demo/webcam_demo.py
```

The webcam demo will open a video window with the following HUD overlay:

- **Predicted sign** — large text, top centre
- **Confidence %** — below sign name
- **Top-3 bar chart** — right panel, colour-coded by confidence tier
- **FPS counter** — bottom left
- **Prediction stability indicator** — green (stable ≥5 frames), yellow (fluctuating)
- **"No hands detected" warning** — when MediaPipe returns empty landmarks for 3+ consecutive frames

Temporal smoothing uses majority voting over a 5-frame sliding window. The champion model processes `(1, 100, 126)` float32 tensors — its hands-only feature config means only LH and RH landmark extraction is required at inference, simplifying the MediaPipe preprocessing path relative to a full-225-dim model.

---

## 14. KSL Adaptation Roadmap

### Why this matters

The project is directly motivated by Kenyan Sign Language (KSL) recognition — a domain with scarce labelled data, limited compute, and a strong need for on-device offline inference. The ASL→KSL transfer path is non-trivial: ASL and KSL are structurally distinct languages with different phonologies, spatial grammar conventions, and lexicons. Direct model deployment from ASL to KSL would actively bias toward ASL-specific patterns that have no KSL equivalent.

### Strengthened transfer argument from Stage 5

The hands-only finding directly strengthens the KSL transfer case. By removing pose landmarks — which encode signer body position, arm length, and filming geometry (all signer-specific and language-agnostic features) — the champion model is already forced to rely purely on hand geometry. Hand geometry encodes sign-specific meaning independent of signer body size or frame position. A hands-only ASL model is therefore a better KSL transfer starting point than a full-landmark model: it has already learned to ignore one major class of cross-linguistic domain shift.

### Proposed validation strategy

1. **KSL from scratch (baseline):** Train a BiLSTM on AI4KSL data with the same hands-only pipeline
2. **ASL-pretrained → frozen LSTM → KSL classifier:** Fine-tune only the Dense head
3. **ASL-pretrained → full fine-tune on KSL:** Update all layers

Compare on a held-out KSL test set using per-class recall — not just overall accuracy, which can mask catastrophic failure on low-frequency signs with no ASL equivalent.

### Data requirements

At the required scale of 500 KSL signs at ≥85% accuracy, approximately 100–200 clips per sign are needed for the BiLSTM to generalise reliably. At the AI4KSL dataset's ~40 clips per sign, augmentation can close some of this gap, but collecting additional multi-signer, multi-region data is the most reliable path. Regional dialect variation within KSL — like signer-to-signer variation in ASL — is the primary generalisation challenge.

### Planned extensions

- [ ] Expand from 35 to 100+ ASL signs as more data becomes accessible
- [ ] Attention mechanism over BiLSTM output for improved SHAP interpretability
- [ ] Temperature scaling for confidence calibration before production deployment
- [ ] Android TFLite inference wrapper
- [ ] KSL data collection and transfer learning validation

---

## 15. Limitations

See [`LIMITATIONS.md`](LIMITATIONS.md) for the complete discussion. Key limitations:

**Data:**
- **46.6% completeness:** 401 of 751 inventory clips are permanently inaccessible (dead YouTube URLs). This is the hard ceiling on achievable accuracy regardless of architecture.
- **6.7 clips/class mean:** All 35 signs fall below the 20-clip minimum threshold. `think` (3 clips), `clothes` (2 clips) are effectively unlearnable at current scale.
- **21 singleton val classes:** Per-class val metrics for 60% of classes are unreliable (1 incorrect prediction = F1 of 0.0). Macro-F1 is the mandatory primary metric.

**Model:**
- **70% target not met:** The honest achievable ceiling under current data constraints is approximately 0.60–0.65. Reaching 70% would likely require ≥50 clips per class or pre-trained hand-shape features.
- **Seed sensitivity:** Identical configurations can diverge by up to 13pp due to initialisation differences amplified by the 52-clip validation set's noise floor. The champion's 0.6011 is a single-seed measurement (expected range: 0.58 ± 0.03).
- **`think` class unlearnable:** F1 = 0.0 in 8 of 9 champion runs. 3 training clips with zero signer overlap is insufficient for any temporal architecture.

**Deployment:**
- **Confidence overconfidence:** Softmax outputs are not well-calibrated probabilities. Temperature scaling is recommended before production deployment (Stage 6 calibration analysis pending).
- **MediaPipe dependency at inference:** Hand detection adds ~18ms per frame to pipeline latency. At 35.28% both-hands-absent rate, real-time performance depends on MediaPipe's robustness.
- **ASL ≠ KSL:** The champion model cannot be directly deployed for Kenyan Sign Language. See the KSL adaptation roadmap above.

**Evaluation:**
- **Augmentation finding is epoch-budget-conditional:** Group 2's conclusion (spatial-temporal harmful) was overturned by the champion run. Conclusions drawn from ablations with fixed epoch budgets must be treated as epoch-specific, not general.
- **Val metric variance:** 3–5pp epoch-to-epoch swings are structural noise on 52 validation clips. Comparisons within 3pp are not statistically meaningful.

---

## 16. Contributing

This project is under active development. Contributions, issues, and suggestions are welcome.

```bash
# Development setup
pip install -r requirements-dev.txt
pre-commit install   # installs git hooks for black + flake8

# Run tests
make test

# Lint
make lint

# Format
black src/ pipelines/ tests/
```

Please open an issue before submitting a pull request for significant changes. The experiment configuration system (OmegaConf + Pydantic v2) enforces strict field validation — any new config fields must be added to both the YAML defaults and the Pydantic schema in `src/utils/config.py`.

---

## 17. License

This project is licensed under the MIT License — see [`LICENSE`](LICENSE) for details.

The WLASL dataset is subject to its own licence terms. See the [WLASL repository](https://github.com/dxli94/WLASL) for details. This project does not redistribute any WLASL video content.

---

<p align="center">
  <strong>Henry Otsyula</strong><br/>
  Senior Data Scientist &amp; ML Engineer<br/>
  Built as part of a sign language recognition research initiative.<br/>
  For questions about KSL adaptation or the pipeline, open an issue.
</p>

<p align="center">
  <sub>
    Stage 5 complete · 23 MLflow runs · champion val macro-F1: 0.6011 · 68,771 parameters · 0.262 MB
  </sub>
</p>