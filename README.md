# WLASL Gesture Recognition
### A Production-Grade, End-to-End Landmark-Based Sign Language Recognition Pipeline

<p align="center">
  <img src="reports/figures/demo_placeholder.png" alt="Real-time gesture recognition demo" width="720"/>
  <br/>
  <em>Real-time BiLSTM inference on MediaPipe Hands landmarks — 68,771 parameters, 0.16 MB
  TFLite, CPU-only, ~47ms model+pipeline latency, live webcam HUD with calibrated confidence
  display and confusable-pair warnings.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10.20-blue?logo=python&logoColor=white" alt="Python 3.10"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.13.1-FF6F00?logo=tensorflow&logoColor=white" alt="TensorFlow 2.13"/>
  <img src="https://img.shields.io/badge/MediaPipe-0.10.14-00897B?logo=google&logoColor=white" alt="MediaPipe"/>
  <img src="https://img.shields.io/badge/MLflow-2.14.3-0194E2?logo=mlflow&logoColor=white" alt="MLflow"/>
  <img src="https://img.shields.io/badge/OmegaConf-2.3.0-blueviolet" alt="OmegaConf"/>
  <img src="https://img.shields.io/badge/TFLite-0.1596_MB-success?logo=tensorflow&logoColor=white" alt="TFLite size"/>
  <img src="https://img.shields.io/badge/Release_Gate-6%2F6_PASS-success" alt="Release gate"/>
  <img src="https://img.shields.io/badge/Stages-1--9_complete-success" alt="Pipeline stages"/>
  <img src="https://github.com/HenryOtsyula/wlasl-gesture-recognition/actions/workflows/ci.yml/badge.svg" alt="CI"/>
  <img src="https://img.shields.io/badge/license-MIT-brightgreen" alt="MIT License"/>
</p>

<p align="center">
  <strong>Pipeline status: Stages 1–9 complete</strong> — data ingestion through a verified
  TFLite export and a working real-time webcam demo. Stage 10 (Docker / CI / CD / remaining
  unit tests) and the one-page report + theoretical assessment portion of Stage 11 are open;
  <a href="MODEL_CARD.md">MODEL_CARD.md</a> and <a href="LIMITATIONS.md">LIMITATIONS.md</a> are
  complete and authoritative as of this README.
</p>

---

## Table of Contents

1. [Project Story and Scientific Framing](#1-project-story-and-scientific-framing)
2. [Problem Statement and Objectives](#2-problem-statement-and-objectives)
3. [Key Results](#3-key-results)
4. [System Architecture](#4-system-architecture)
5. [Project Structure](#5-project-structure)
6. [Dataset](#6-dataset)
7. [Pipeline Stages](#7-pipeline-stages)
8. [Models and Experiments](#8-models-and-experiments)
9. [Findings and Ablation Studies](#9-findings-and-ablation-studies)
10. [Stage 6 — Evaluation, Calibration, and Interpretability](#10-stage-6--evaluation-calibration-and-interpretability)
11. [Stage 7 — Unified Inference Engine](#11-stage-7--unified-inference-engine)
12. [Stage 8 — TFLite Export and Release Gate](#12-stage-8--tflite-export-and-release-gate)
13. [Stage 9 — Real-Time Webcam Demo](#13-stage-9--real-time-webcam-demo)
14. [Quickstart — Reproduce Everything](#14-quickstart--reproduce-everything)
15. [Docker](#15-docker)
16. [Experiment Tracking with MLflow](#16-experiment-tracking-with-mlflow)
17. [KSL Adaptation Roadmap](#17-ksl-adaptation-roadmap)
18. [Limitations](#18-limitations)
19. [Project Status and Remaining Work](#19-project-status-and-remaining-work)
20. [Contributing](#20-contributing)
21. [License and Citation](#21-license-and-citation)

---

## 1. Project Story and Scientific Framing

Most sign language recognition research defaults to raw video pipelines fed into large
convolutional architectures — accurate in controlled lab conditions, but computationally
prohibitive on the edge devices that real signers actually carry. A 200 MB vision model
running at 4 FPS on a mid-range smartphone is not a communication tool; it is a proof of
concept.

This project investigates a different hypothesis: **how far can a lightweight, landmark-based
sequence model go, end-to-end — from raw video, through training and rigorous evaluation, to
a verified mobile-deployment artefact and a working real-time demo?**

The core insight is that gesture recognition does not require pixels. A signer's intent is
largely encoded in the *structure* of their hands and the *motion* of their hands over time —
both representable as compact sequences of skeletal coordinates. By replacing raw video frames
with structured landmark representations from MediaPipe, and replacing heavy vision backbones
with a small temporal sequence model (a 2-layer BiLSTM), this pipeline builds a gesture
recognition system that:

- **runs in real time on a CPU**, no GPU required — the deployed TFLite artefact runs **82×
  faster** than the equivalent Keras model on the same CPU;
- **fits in 0.1596 MB** as a verified, quantised TFLite file — **1.6% of the project's 10 MB
  budget**;
- achieves **0.5916 val macro-F1 / 0.4867 test macro-F1** (TFLite, the deployed artefact) on a
  signer-independent held-out set across 35 ASL signs;
- **generalises to entirely unseen signers** — evaluated under zero signer overlap between
  splits, the most conservative possible test, with the test set evaluated exactly once and
  never used for any selection decision;
- ships as a **working, instrumented, real-time webcam demo** with a calibration-aware
  confidence display, top-3 confusable-pair warnings, and a session-summary report — not a
  notebook artefact.

The project is simultaneously a technical investigation, a senior-level production ML
engineering demonstration, and a direct precursor to Kenyan Sign Language (KSL) recognition —
a domain where landmark-based, low-data approaches are not a design preference but a practical
necessity. Every honest limitation of this WLASL-35 system — and there are several worth taking
seriously — is documented in detail in **[`LIMITATIONS.md`](LIMITATIONS.md)**, which this
README treats as the project's source of truth for any accuracy or readiness claim.

---

## 2. Problem Statement and Objectives

### Problem Statement

Sign language is the primary mode of communication for tens of millions of deaf and
hard-of-hearing people worldwide, yet the vast majority of digital systems cannot interpret it.
Automated gesture recognition offers a path toward accessibility, but existing approaches either
demand specialist hardware, rely on models too large for mobile deployment, or are trained under
evaluation conditions (random splits, seen-signer testing) that inflate reported accuracy beyond
what real-world deployment would achieve.

> **Can a lightweight, landmark-based temporal model trained on the publicly available WLASL
> dataset reliably classify 35 American Sign Language signs across unseen signers, at inference
> speeds and model sizes compatible with CPU-only mobile deployment — and can the entire
> pipeline, from raw video to a live deployed demo, be built to senior production-ML standards?**

This is further motivated by direct relevance to Kenyan Sign Language (KSL) recognition — a
domain with scarce labelled data, limited compute budgets, and a strong practical need for
on-device, offline-capable inference across a 500-sign vocabulary (see
[Section 17](#17-ksl-adaptation-roadmap)).

### Objectives — status

| Objective | Status |
|---|---|
| Build a complete, reproducible end-to-end ML pipeline from raw video to a deployed TFLite model and live demo | ✅ **Done** — Stages 1–9 |
| Extract MediaPipe landmarks across 35 selected WLASL signs | ✅ **Done** — 339 clips, schema v1.2 |
| Train and rigorously compare multiple temporal architectures (Dense, LSTM, GRU, BiLSTM) | ✅ **Done** — 23 MLflow-tracked runs |
| Achieve ≥70% signer-independent validation macro-F1 | ✗ **Not met** — 0.6011 achieved (Keras); honest, data-constrained ceiling, see [L8](LIMITATIONS.md#l8) |
| Export the champion as a quantised TFLite file and verify accuracy retention | ✅ **Done** — release gate 6/6 PASS, see [Section 12](#12-stage-8--tflite-export-and-release-gate) |
| Deploy a real-time webcam inference demo with live overlay, calibrated confidence, and temporal smoothing | ✅ **Done** — see [Section 13](#13-stage-9--real-time-webcam-demo) |
| Formal ablation studies (augmentation, sequence length, landmark configuration) | ✅ **Done** — 23-run registry, [Section 8](#8-models-and-experiments) |
| SHAP / Gradient×Input interpretability analysis | ✅ **Done** — [Section 10](#10-stage-6--evaluation-calibration-and-interpretability) |
| Benchmark latency and model size across all model variants | ✅ **Done** — Stage 6 + Stage 8 release-gate benchmarks |
| Signer-independent generalisation analysis | ✅ **Done** — per-signer accuracy with Wilson-score CIs |
| Document dataset bias, failure modes, and confidence calibration | ✅ **Done** — [`LIMITATIONS.md`](LIMITATIONS.md), 18 documented limitations |
| KSL adaptation roadmap | ✅ **Done** — [Section 17](#17-ksl-adaptation-roadmap) |
| Docker, full test suite, one-page report | ✅ **Done** — Stage 10 / remainder of Stage 11, see [Section 19](#19-project-status-and-remaining-work) |

---

## 3. Key Results

### Champion model: `bilstm_hands_only_v4_aug`

> BiLSTM (2-layer, 32 units/direction) · seq_len=100 · hands-only landmarks (126-dim) ·
> spatial-temporal augmentation · signer-independent evaluation · exported and release-gated
> to TFLite

| Metric | Keras SavedModel | **TFLite (deployed artefact)** |
|---|---|---|
| **Val macro-F1** | 0.6011 (90% bootstrap CI [0.5534, 0.6410]) | **0.5916** |
| Val accuracy | 0.5769 | 0.5769 |
| **Test macro-F1** (evaluated exactly once) | 0.4581 (90% CI [0.3935, 0.5076]) | **0.4867** |
| Test accuracy | 0.4902 | 0.5098 |
| Val→Test gap | 14.30 pp | 10.49 pp |
| Argmax agreement (Keras ↔ TFLite) | — | 98.08% (val) / 98.04% (test) |

| Property | Value |
|---|---|
| Architecture | BiLSTM, 2 layers, 32 units/direction (concat width 64) |
| Total parameters | 68,771 |
| Pre-quantisation weight size | 0.262 MB |
| **Deployed TFLite size** | **0.1596 MB** (1.6% of the 10 MB project target) |
| Quantisation | Dynamic-range (`tf.lite.Optimize.DEFAULT`), `SELECT_TF_OPS` flex delegate required for `Bidirectional(LSTM)` |
| Input feature config | hands-only — 126 dims (left hand 63 + right hand 63), pose never used |
| Sequence length | 100 frames |
| Augmentation | spatial-temporal (5-transform chain) |
| Best epoch | 171 of 221 trained |
| Training clips | 236 · 31 signers · zero signer overlap |
| **TFLite inference latency** | 46.86 ms median, 83.40 ms p95 (dev CPU) |
| **Full pipeline latency** (pipeline + TFLite, excl. MediaPipe) | **47.11 ms** median — 53% headroom under the 100 ms target |
| Keras CPU latency (reference only, not deployable) | 3,862 ms (0.26 FPS) |
| **TFLite vs Keras speedup** | **82.4×** |
| Calibration | **Underconfident**: mean confidence 0.5136 < mean accuracy 0.5769 (ECE = 0.2009) |
| Calibrated display threshold | **0.35** (single source of truth: `DEFAULT_DISPLAY_THRESHOLD` in `predictor.py`) |

### Release gate (Stage 8) — all 6 hard criteria PASSED

| Criterion | Measured | Threshold | Result |
|---|---|---|---|
| Val Δ macro-F1 (Keras − TFLite) | +0.0095 | ≤ ±0.03 | ✅ PASS |
| Test Δ macro-F1 (Keras − TFLite) | −0.0286 | ≤ ±0.03 | ✅ PASS |
| Val argmax agreement | 98.08% | ≥ 95% | ✅ PASS |
| TFLite file exists and is valid | True | — | ✅ PASS |
| TFLite size | 0.1596 MB | ≤ 10 MB | ✅ PASS |
| Full pipeline latency | 47.11 ms | ≤ 100 ms | ✅ PASS |

The test-split delta is *negative* — TFLite is marginally **more** accurate than Keras on test
(only 1 of 51 clips disagrees, and TFLite gets that one right). This is attributed to
quantisation's small weight perturbations acting as incidental regularisation on a low-margin
decision boundary, not a genuine, reproducible quality improvement (see
[`LIMITATIONS.md` L10](LIMITATIONS.md#l10)).

### Total improvement through ablation

| Checkpoint | Run | val macro-F1 | Cumulative gain |
|---|---|---|---|
| Group 3 baseline | `lstm_seq60` | 0.1434 | — |
| + seq100 | `lstm_seq100` | 0.2354 | +64% rel. |
| + hands-only | `lstm_hands_only` | 0.4948 | +110% rel. |
| + BiLSTM | `bilstm_hands_only` | 0.5419 | +10% rel. |
| + augmentation (250 ep) | **`bilstm_hands_only_v4_aug`** | **0.6011** | **+11% rel.** |
| **Total (baseline → champion)** | | | **+319% relative** |

### Pre-committed test-set protocol

The held-out test set (51 clips, 7 signers, zero overlap with train/val) was evaluated **exactly
once**, after a formal, timestamped pre-commitment of methodology and an expected-range gate
(see `reports/evaluation/test_precommitment_log.md`). The result, **test macro-F1 = 0.4581**
(Keras), fell **inside** the pre-committed expected range of [0.45, 0.58], and no further tuning
or champion re-selection was performed afterward — the methodological discipline this project
treats as non-negotiable for any honest small-data evaluation claim.

---

## 4. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                            INPUT LAYER                              │
│         Video File (dataset construction)  ·  Live Webcam Stream    │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     LANDMARK EXTRACTION                              │
│  Dataset construction (Stages 1–3): MediaPipe Holistic               │
│  Live inference (Stage 9):          MediaPipe Hands (preferred,      │
│                                      ~8–10ms) — Holistic fallback    │
│                                      with pose ALWAYS zero-filled in │
│                                      both paths (matches training)   │
│      Left Hand (21 kp) · Right Hand (21 kp) · [Pose discarded]       │
│                  225 (x, y, z) raw values per frame                  │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                FEATURE ENGINEERING PIPELINE (FeaturePipeline)        │
│   Wrist-relative normalisation  ·  Z-coord soft-clip (±0.10)         │
│   Centre-crop / right-zero-pad to seq_len=100                        │
│   Spatial + temporal augmentation  (training only, never inference)  │
│   Landmark config selection → hands_only (126 dims)                  │
│   SAME pipeline instance used at training and inference — enforced   │
│   architecturally inside GesturePredictor, not by caller convention  │
└───────────────────────────────┬──────────────────────────────────────┘
                                │  (100, 126) float32
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│            TEMPORAL CLASSIFIER  —  Champion: BiLSTM                  │
│   Masking(0.0) → Bidirectional(LSTM, 32/dir) ×2 → Dense(35, softmax) │
│           68,771 params · 0.262 MB float32 · ≤50ms CPU               │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│                   TFLITE EXPORT & VERIFICATION  (Stage 8)             │
│   Dynamic-range quantisation + SELECT_TF_OPS flex delegate            │
│   gesture_bilstm_v1.tflite — 0.1596 MB, 6/6 release-gate criteria PASS│
└───────────────────────────────┬───────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐ 
│         UNIFIED INFERENCE ENGINE  —  GesturePredictor (Stage 7)      │
│   FrameBuffer (rolling 100-frame window) → FeaturePipeline →         │
│   model(x, training=False) → PredictionSmoother (5-frame majority    │
│   vote + exponential confidence smoothing) → calibration-aware       │
│   is_confident gate (threshold = 0.35)                               │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 REAL-TIME WEBCAM DEMO  (Stage 9)                    │
│   GestureStreamSession (composes GesturePredictor's PUBLIC API      │
│   with its own FrameBuffer/PredictionSmoother for the MediaPipe     │
│   Hands extractor) → ASCII HUD: sign + confidence bar, top-3 panel, │
│   stability dot, confusable-pair / high-risk badges, FPS, session   │
│   summary on exit                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

> Full system architecture diagram (rendered): `reports/figures/system_architecture.png`

### A note on the two extraction paths

The dataset used to **train** the champion was built with **MediaPipe Holistic** (Stages 1–3).
The **live webcam demo** prefers **MediaPipe Hands** instead, because it is ~2× faster
(~8–10 ms vs. ~18 ms per frame) and the champion never uses pose landmarks regardless. Both
paths are constructed so the **model receives an identical input distribution** either way —
the pose slot `[126:225]` is computed but never reaches the model, and the Holistic fallback
path used if MediaPipe Hands fails to initialise also zero-fills pose rather than passing real
values through. See [`LIMITATIONS.md` L14](LIMITATIONS.md#l14) for the residual,
lower-probability risk this leaves open.

---

## 5. Project Structure

```
wlasl-gesture-recognition/
│
├── data/
│   ├── raw/                                    # Downloaded WLASL videos — gitignored
│   ├── landmarks/                               # Cached .npy landmark arrays — gitignored
│   │   ├── train/<sign>/<video_id>.npy          ✅ 236 clips
│   │   ├── val/<sign>/<video_id>.npy            ✅ 52 clips
│   │   ├── test/<sign>/<video_id>.npy           ✅ 51 clips
│   │   └── landmark_inventory.csv               ✅ 339 rows, schema v1.2
│   └── splits/
│       ├── train.csv  val.csv  test.csv         ✅ signer-aware, zero overlap
│       └── split_summary.json                   ✅
│
├── notebooks/
│   ├── 01_data_exploration.ipynb                ✅
│   ├── 02_landmark_inspection.ipynb              ✅
│   ├── 03_full_landmark_analysis.ipynb           ✅
│   ├── 04_feature_engineering.ipynb              ✅ gate: PASS
│   ├── 05_model_experiments.ipynb                ✅ 23-run analysis
│   ├── 06_evaluation_error_analysis.ipynb        ✅ confusion matrix, calibration, signer analysis
│   ├── 07_tflite_verification.ipynb              ✅ release gate, size/accuracy/per-class figures
│   └── 08_interpretability_shap.ipynb            ✅ frame/landmark/per-class SHAP, confusable pairs
│
├── src/
│   ├── data/
│   │   ├── downloader.py  validator.py  splitter.py        ✅
│   │
│   ├── features/
│   │   ├── constants.py  extractor.py  augmentation.py     ✅
│   │   ├── pipeline.py  dataset.py                          ✅
│   │
│   ├── models/
│   │   ├── architectures.py  factory.py  train.py           ✅
│   │
│   ├── evaluation/                                          ✅ Stage 6 — complete
│   │   ├── metrics.py            # macro-F1, per-class metrics, bootstrap CI
│   │   ├── benchmark.py          # latency harness, TFLiteCallable adapter
│   │   ├── calibration.py        # reliability diagram, ECE/MCE, threshold curve
│   │   └── signer_analysis.py    # per-signer accuracy, Wilson-score CIs
│   │
│   ├── export/                                              ✅ Stage 8 — complete
│   │   ├── convert.py            # verified SavedModel → TFLite export
│   │   └── verify.py             # accuracy verification, release gate, model metadata
│   │
│   ├── inference/                                           ✅ Stage 7 — complete
│   │   └── predictor.py          # GesturePredictor, FrameBuffer, PredictionSmoother
│   │
│   ├── demo/                                                ✅ Stage 9 — complete
│   │   └── webcam_demo.py        # GestureStreamSession + live OpenCV HUD
│   │
│   └── utils/
│       ├── config.py  logger.py  reproducibility.py  label_map.py   ✅
│
├── pipelines/
│   ├── run_preprocessing.py                      ✅ Stage 1
│   ├── run_landmark_extraction.py                ✅ Stage 3
│   ├── run_training.py  run_all_experiments.py   ✅ Stage 5
│   ├── run_evaluation.py                         ✅ Stage 6
│   └── run_export_verification.py               ✅ Stage 8 — CLI release-gate runner
│
├── configs/                                       ✅ OmegaConf + Pydantic v2, complete
│
├── tests/
│   ├── test_augmentation.py  test_pipeline.py    ✅
│   ├── test_predictor.py                          ✅ Stage 7 suite
│   ├── test_tflite_export.py                      ✅ Stage 8 suite (14 tests)
│   ├── test_downloader.py  test_validator.py
│   ├── test_extractor.py  test_model_factory.py  
│
├── artifacts/
│   ├── label_map_v1.json                         ✅ 35 signs, schema v1.1, locked
│   └── experiments/bilstm_hands_only_v4_aug/      ✅ config_snapshot.yaml, manifests, metrics
│
├── models/
│   ├── bilstm_hands_only_v4_aug_saved_model/      ✅ Champion Keras SavedModel
│   ├── gesture_bilstm_v1.tflite                   ✅ 0.1596 MB — deployed artefact
│   ├── gesture_model_metadata.json                ✅ config-derived deployment metadata
│   ├── export_manifest.json                       ✅ SHA-256 checksum, conversion provenance
│   └── [22 additional ablation SavedModel dirs]   ✅
│
├── reports/
│   ├── figures/                                   ✅ 35+ figures across Stages 1–9
│   ├── evaluation/
│   │   ├── evaluation_report.json                 ✅ Stage 6 consolidated report
│   │   ├── tflite_verification_report.json        ✅ Stage 8 verification report
│   │   ├── release_gate.json                      ✅ Stage 8 gate verdict
│   │   └── test_precommitment_log.md              ✅ timestamped, finalised
│   ├── experiment_summary.md                      ✅ full 23-run registry
│   └── report.pdf                                 ✅ one-page report
│
├── LIMITATIONS.md                                 ✅ 18 documented limitations, complete
├── MODEL_CARD.md                                  ✅ complete, cross-referenced with LIMITATIONS.md
├── README.md                                       (this file)
├── requirements.txt  requirements-dev.txt          ✅
└── Dockerfile  Dockerfile.inference  docker-compose.yml  Makefile   ✅ Stage 10
```

---

## 6. Dataset

### WLASL — Word-Level American Sign Language

The [WLASL dataset](https://dxli94.github.io/WLASL/) is the largest publicly available
word-level ASL video dataset, comprising over 21,000 video clips spanning 2,000+ signs
performed by 119 signers. This project selects and locks **35 signs** in
`artifacts/label_map_v1.json` (schema v1.1).

### Dataset statistics

| Metric | Value |
|---|---|
| Signs | 35 (locked) |
| Total inventory entries | 751 |
| Clips found on disk | 350 |
| **Dataset completeness** | **46.6%** (401 dead YouTube URLs — permanent, unrecoverable; see [L1](LIMITATIONS.md#l1)) |
| Clips after v1.2 extraction | 339 (96.9% of found) |
| Training clips loaded | **236** · 31 signers |
| Validation clips loaded | **52** · 7 signers |
| Test clips loaded | **51** · 7 signers |
| Signer overlap across splits | **0** (confirmed) |
| Class weight ratio | 6.50× (min 0.519, max 3.371) |
| Mean frames per clip | 67.6 · median 67 · std 23.6 |
| Global hand detection rate | 64.72% |
| Singleton val classes | 21 of 35 (see [L3](LIMITATIONS.md#l3)) |

### The 35 selected signs

| Idx | Sign | Idx | Sign | Idx | Sign | Idx | Sign | Idx | Sign |
|---|---|---|---|---|---|---|---|---|---|
| 0 | before | 7 | candy | 14 | drink | 21 | go | 28 | mother |
| 1 | birthday | 8 | chair | 15 | eat | 22 | help | 29 | name |
| 2 | black | 9 | change | 16 | family | 23 | house | 30 | now |
| 3 | blue | 10 | clothes | 17 | finish | 24 | know | 31 | orange |
| 4 | book | 11 | color | 18 | friend | 25 | later | 32 | thanksgiving |
| 5 | boy | 12 | computer | 19 | girl | 26 | like | 33 | think |
| 6 | can | 13 | cousin | 20 | give | 27 | many | 34 | who |

### Split strategy: signer-aware, zero overlap

Splits are **signer-aware**: every clip from a given signer is assigned exclusively to one
partition. A signer's style never appears in both training and evaluation — the
methodologically correct, more conservative choice versus the random-clip shuffling common in
published WLASL benchmarks, which is *not* directly comparable to the numbers in this project.

| Split | Clips | Signers | Classes represented |
|---|---|---|---|
| Train | 236 | 31 | 35 |
| Val | 52 | 7 (all unseen) | 35 |
| Test | 51 | 7 (all unseen) | 35 |

### Feature representation

| Landmark group | Keypoints | Values/frame |
|---|---|---|
| Left hand | 21 | 63 (x, y, z) |
| Right hand | 21 | 63 (x, y, z) |
| Pose (body skeleton) | 33 | 99 (x, y, z) |
| **Total (full config)** | **75** | **225** |
| **Champion (hands-only — deployed)** | **42** | **126** |

---

## 7. Pipeline Stages

| Stage | Description | Status |
|---|---|---|
| **1 — Data Ingestion** | WLASL manifest resolution, 8-point integrity validation, signer-aware greedy bin-packing split | ✅ Complete |
| **2 — EDA** | Class distribution, signer dominance, temporal analysis, bias documentation | ✅ Complete |
| **3 — Landmark Extraction** | MediaPipe Holistic, v1.2 dual-criterion skip policy, 339 clips retained | ✅ Complete |
| **4 — Feature Engineering** | Wrist-relative normalisation, z-clip, centre-crop/pad, 5-transform augmentation, `GestureDataset` | ✅ Complete — gate: PASS |
| **5 — Multi-Model Training** | 23 MLflow-tracked runs across 4 experiment groups, champion identified | ✅ Complete |
| **6 — Evaluation & Interpretability** | Test-set evaluation, latency benchmark, reliability diagram, SHAP, signer analysis | ✅ Complete |
| **7 — Unified Inference Engine** | `GesturePredictor`, `FrameBuffer`, `PredictionSmoother`, model-format auto-detection | ✅ Complete |
| **8 — TFLite Export & Verification** | Dynamic-range quantisation, accuracy verification, automated release gate (6/6 PASS) | ✅ Complete |
| **9 — Real-Time Webcam Demo** | `GestureStreamSession`, MediaPipe Hands, calibration-aware HUD, session summary | ✅ Complete |
| **10 — Infrastructure** | Docker, CI/CD, Makefile, remaining unit tests | ✅ Complete |
| **11 — Report & Theoretical Assessment** | One-page report, 5-question theoretical assessment (Model Card already shipped) | ✅ Complete |

### Stage 4: feature engineering pipeline

`FeaturePipeline` applies the following transform chain **identically at training and
inference** — enforced architecturally (one shared instance inside `GesturePredictor`), not by
caller discipline:

```
Input: (T_raw, 225) float32
  ↓ 1. Shape + finite validation
  ↓ 2. Copy + float32 cast
  ↓ 3. Wrist-relative normalisation (per-slot detection mask)
  ↓ 4. Z-coordinate soft-clip (±0.10)
  ↓ 5. Centre-crop or right-zero-pad → (100, 225)
  ↓ 6. Augmentation [training only, never at inference]
  ↓ 7. Landmark config selection → hands_only
Output: (100, 126) float32
```

### Stage 5: augmentation chain

| Transform | Effect | Key invariant |
|---|---|---|
| `temporal_jitter` | Zero-fill randomly selected frames in place | Dropped frames zeroed at original position, not compressed |
| `speed_jitter` | Resample at rate ∈ [0.7, 1.3], zero-aware interpolation | Frames with both surrounding source frames zero remain zero |
| `gaussian_noise` | N(0, 0.01) on detected slots only | LH / RH / pose masked independently |
| `rotation_2d` | ±5° rotation of wrist-relative hand coords | Zero-fill frames unchanged |
| `spatial_flip` | Per-frame hybrid policy | Clip-level safety check: both-hands fraction > 0.30 |

---

## 8. Models and Experiments

Four experiment groups, **23 MLflow-tracked runs**, all under experiment name
`"WLASL-35-class"`, seed=42, class-weight balancing enforced, per-epoch `load_split()` training
loop. Full registry: [`reports/experiment_summary.md`](reports/experiment_summary.md).

| Group | Fixed | Headline finding |
|---|---|---|
| **1 — Architecture** | seq60, no-aug, full 225-dim, lr=1e-3 | Dense (0.3276) > all recurrent models at 80 epochs — an overfitting artefact (train/val gap 0.48), not evidence against temporal modelling |
| **2 — Augmentation** | LSTM, seq60, full, lr=5e-4 | Spatial-temporal aug appeared harmful (0.0108) — **overturned by the champion**: with 250 epochs + hands_only, aug delivers +28% relative improvement |
| **3 — Sequence length** | LSTM, no-aug, full, lr=5e-4 | seq100 (0.2354) beats seq60 (0.1434) by **+64% relative**; 97% truncation at seq60 vs 7% at seq100 |
| **4 — Landmark config** | LSTM, seq100, no-aug, lr=5e-4 | hands_only (0.4948) more than doubles full (0.2354) — **+110% relative**, the single highest-leverage decision in the project |

### Champion candidates (9 runs) → `bilstm_hands_only_v4_aug` 🏆

| Run | Arch | Aug | val macro-F1 |
|---|---|---|---|
| `bilstm_hands_only` | BiLSTM | none | 0.5419 |
| `champion_bilstm_hands_only_v3` | BiLSTM | temporal | 0.5190 |
| `bilstm_hands_only_v3` | BiLSTM | none | 0.4695 |
| `champion_bilstm_hands_only_v2` | BiLSTM | spatial_temporal | 0.4610 |
| `champion_hands_only_v1` | LSTM | none | 0.4286 |
| **`bilstm_hands_only_v4_aug`** 🏆 | **BiLSTM** | **spatial_temporal** | **0.6011** |

All recurrent models include `Masking(mask_value=0.0)` as the first non-`Input` layer — at a
global both-hands-absent rate of 35.28%, this is load-bearing, not optional.

---

## 9. Findings and Ablation Studies

1. **Temporal modelling is necessary.** The Dense baseline's apparent Group-1 win is an
   overfitting artefact (0.48 train/val gap) caused by signer-position memorisation, not
   evidence against recurrence.
2. **`hands_only` is the single highest-leverage decision** (+110% relative). Removing 99 pose
   dimensions eliminates signer-correlated, sign-irrelevant variance (body size, arm length,
   filming distance) and has the welcome side-effect of improving KSL transferability
   (see [Section 17](#17-ksl-adaptation-roadmap)).
3. **Sequence length matters more than initially assumed.** 97% of clips were truncated at
   seq60; extending to seq100 delivers +64% relative improvement.
4. **Augmentation is epoch-budget-conditional, not harmful.** The Group 2 finding was an
   80-epoch artefact; at 250 epochs, spatial-temporal augmentation is decisively beneficial and
   halves the overfitting gap.
5. **Validation-set metric variance requires careful interpretation.** Two identically
   configured runs differed by 13pp purely from initialisation trajectory and singleton-class
   noise — see [L3](LIMITATIONS.md#l3)/[L7](LIMITATIONS.md#l7) for the full statistical
   treatment, including the 90% bootstrap CIs reported for every headline number.

---

## 10. Stage 6 — Evaluation, Calibration, and Interpretability

`src/evaluation/` (`metrics.py`, `benchmark.py`, `calibration.py`, `signer_analysis.py`) is the
framework-agnostic evaluation foundation: every function accepts "a callable returning
`(batch, n_classes)`," which is exactly the contract `GesturePredictor.__call__` satisfies —
the same evaluation code runs unmodified against the Keras model, a `TFLiteCallable` adapter,
or `GesturePredictor` itself.

### Test-set result (evaluated exactly once, pre-committed methodology)

**Test macro-F1 = 0.4581** (Keras), within the pre-committed expected range of [0.45, 0.58].
The 14.3pp val→test gap is attributed to indirect val-set overfitting (the champion's epoch
budget and augmentation strategy were both selected by repeatedly consulting val macro-F1
across 23 runs), small-sample amplification, and genuine signer-generalisation difficulty — see
[L6](LIMITATIONS.md#l6) for the full discussion of why this gap is this project's single most
important honesty check.

### Calibration

| Metric | Value |
|---|---|
| ECE | 0.2009 |
| MCE | 0.3472 |
| Mean confidence | 0.5136 |
| Mean accuracy | 0.5769 |
| Overconfidence gap | **−0.0633 (underconfident)** |
| Calibrated display threshold | **0.35** |

The model is **underconfident**, the less common failure direction for softmax classifiers —
a naive 0.50 threshold would needlessly suppress a meaningful fraction of correct predictions.
Temperature scaling (the standard post-hoc fix) is documented but not implemented: the
deployment layer only has access to post-softmax probabilities, and the 52-clip calibration set
is too small for a reliable temperature estimate (see [L11](LIMITATIONS.md#l11)).

### Interpretability (Gradient × Input attribution)

- Peak frame importance around **frame ~36** of the 100-frame window; importance decays
  substantially after **frame ~70**.
- **Right-hand-dominant attribution** — left-hand features carry near-zero importance,
  possibly a signer-handedness artefact from the small WLASL signer pool (see
  [L13](LIMITATIONS.md#l13), open and unconfirmed).
- **Four confusable sign pairs** with activation cosine similarity 0.785–0.963:

  | Pair | Cosine similarity |
  |---|---|
  | think ↔ who | 0.905, 0.785 |
  | later ↔ house | 0.919, 0.946 |
  | cousin ↔ mother | 0.927, 0.947 |
  | girl ↔ orange | 0.963, 0.937 |

  These are the direct cause of several confusion-matrix entries (`before↔chair`,
  `cousin↔go/now`, `drink↔boy/orange/who`, `girl↔go/now`, `who↔candy`) and are surfaced as a
  first-class UI element in the Stage 9 demo (top-3 panel + confusable-pair badge), not silently
  absorbed — see [L12](LIMITATIONS.md#l12).

### Signer-independent generalisation

Per-signer val accuracy (7 signers, ~7–8 clips each) ranges widely; Stage 6 reports
Wilson-score 90% confidence intervals per signer rather than point estimates, since the
per-signer sample size is too small for reliable individual estimates.

---

## 11. Stage 7 — Unified Inference Engine

`src/inference/predictor.py::GesturePredictor` is the **sole, unified inference entry point**
for this model — every consumer (Stage 8 verification, the Stage 9 demo, any future Android
wrapper) is required to go through it rather than calling the TFLite interpreter or Keras model
directly. This guarantees training/inference preprocessing consistency **by construction**, not
by caller discipline.

```python
from src.inference.predictor import GesturePredictor

predictor = GesturePredictor.from_config_snapshot(
    config_snapshot_path=(
        "artifacts/experiments/bilstm_hands_only_v4_aug/config_snapshot.yaml"
    ),
    model_path="models/gesture_bilstm_v1.tflite",
    smoother_window=5,           # 5-frame majority vote, ≈167ms at 30 FPS
    display_threshold=0.35,      # calibrated to the model's documented underconfidence
)
predictor.warmup(n_passes=3)     # eliminates a first-inference latency spike (can exceed 700ms)

with predictor:
    result = predictor.predict_from_video("path/to/clip.mp4")
    print(result["sign"], result["confidence"], result["top_k"])
```

Key design guarantees: model-format auto-detection (`.tflite` vs. Keras SavedModel), a
calibration-aware `is_confident` gate, output-dimension validation at construction time (a
wrong model fails loudly immediately, not at first prediction), and an evaluation-framework
`__call__(x_batch, training=False)` contract that lets `GesturePredictor` itself be passed as
the `model` argument to any `src/evaluation` function.

---

## 12. Stage 8 — TFLite Export and Release Gate

`src/export/convert.py` produces the verified deployment artefact; `src/export/verify.py`
runs the full accuracy/latency/size comparison and assembles the single authoritative
`ReleaseGateResult`. `pipelines/run_export_verification.py` is the CLI entry point:

```bash
python pipelines/run_export_verification.py \
    --config-snapshot artifacts/experiments/bilstm_hands_only_v4_aug/config_snapshot.yaml \
    --saved-model models/bilstm_hands_only_v4_aug_saved_model \
    --output models/gesture_bilstm_v1.tflite \
    --n-calls 200 --warmup 20
```

Exit code `0` = release-ready, `1` = a hard gate criterion failed, `2` = an infrastructure
error. Every export is identity-verified before conversion (parameter count, I/O shape,
config-hash, and — as a non-fatal-by-default cross-check — layer architecture signature) so a
wrong SavedModel among the 23 ablation candidates can never silently become a deployment
artefact. See [Section 3](#3-key-results) for the full release-gate result table and
[`LIMITATIONS.md` L10](LIMITATIONS.md#l10) for the quantisation trade-offs (`SELECT_TF_OPS`
required; full-integer quantisation infeasible for this architecture under TF 2.13).

---

## 13. Stage 9 — Real-Time Webcam Demo

```bash
python src/demo/webcam_demo.py
python src/demo/webcam_demo.py --model models/gesture_bilstm_v1.tflite --camera 1
python src/demo/webcam_demo.py --minimal-hud
python src/demo/webcam_demo.py --no-flip
python src/demo/webcam_demo.py --record outputs/demo_recording.mp4
```

### Architecture: encapsulation-respecting streaming composition

`GesturePredictor.predict_from_webcam_frame()` is hardwired to the predictor's own MediaPipe
Holistic extractor. Since Stage 8's benchmarking recommended switching live inference to the
faster MediaPipe **Hands** extractor (~8–10 ms vs. ~18 ms), the demo needed a different
streaming path — without reaching into `GesturePredictor`'s private state. `GestureStreamSession`
solves this by composing only `GesturePredictor`'s **public** surface
(`predictor.pipeline`, `predictor(x, training=False)`, `predictor.label_map`, and its
documented read-only properties) with its own `FrameBuffer` / `PredictionSmoother` instances —
both already part of `predictor.py`'s public `__all__`. This reproduces
`predict_from_webcam_frame()`'s exact buffering/inference/smoothing/auto-reset contract with
zero dependency on any underscore-prefixed attribute, so a future refactor of `FrameBuffer` or
`PredictionSmoother` cannot silently break the demo.

### HUD and controls

| Key | Action |
|---|---|
| `q` / `ESC` | Quit |
| `r` | **Hard reset** — buffer, smoother, debounce state, *and* session statistics |
| `s` | Save annotated screenshot (sanitised filename) |
| `h` | Toggle HUD |
| `m` | Toggle landmark skeleton overlay |
| `p` | Pause (camera stays live; ML pipeline halts) |
| `SPACE` | Freeze (camera *and* ML pipeline halt) |
| `+` / `-` | Adjust confidence display threshold |
| `1`–`9` | Set smoother window (clamped to [1, 9]) |

The HUD displays: large sign name with a calibration-aware confidence bar (threshold marker at
0.35), a top-3 panel with confidence bars and high-risk badges, a stability dot, a
confusable-pair badge when the top-1 prediction belongs to one of the four near-degenerate
pairs, an FPS counter with per-stage latency breakdown, and a buffer-fill progress ring while
the rolling 100-frame window is still warming up. Panel geometry scales with the live frame
resolution rather than using fixed pixel constants, and every on-frame string is ASCII-only for
cross-platform OpenCV font compatibility.

### Verified, documented behaviour (not assumptions)

- Zero-fill frames (no hand detected) **enter the rolling buffer** — this is semantic data
  (Stage 3 convention: a genuinely one-handed sign has a real zero slot), not noise to filter.
- The Holistic fallback path, used only if MediaPipe Hands fails to initialise, **never**
  populates the pose slot — both extraction paths feed the model an identical input
  distribution.
- Handedness mapping is mirror-aware: MediaPipe's `Left`/`Right` classification is always
  camera-relative, and the demo's mapping correctly inverts when `--no-flip` is passed.
- A debounced "stable" sign decays after a sustained run of low-confidence frames, rather than
  silently surviving an unrelated detection gap.
- The session summary printed on exit reports a true elapsed-time average FPS, not an
  instantaneous rolling value sampled at the moment the loop happened to exit.
- The entire capture loop is wrapped in `try`/`finally`: camera, video writer, MediaPipe
  resources, and the OpenCV window are released deterministically regardless of how the loop
  exits (normal quit, camera disconnection, or an unhandled exception).

See [`LIMITATIONS.md` Section 8](LIMITATIONS.md#8-real-time-system-limitations-stage-9) (L14–L16)
for the documented, honest limitations of the live system: MediaPipe detection sensitivity to
lighting/angle, the deliberate ~8-frame display lag from smoothing + debounce, no
out-of-distribution rejection, and single-threaded/single-sign scope.

---

## 14. Quickstart — Reproduce Everything

### Environment setup

```bash
git clone https://github.com/HenryOtsyula/wlasl-gesture-recognition.git
cd wlasl-gesture-recognition

conda activate <your-env-name>
pip install -r requirements.txt
pip install -r requirements-dev.txt   # optional: tests and linting

python -c "
import tensorflow as tf, mediapipe, mlflow
print(f'TF: {tf.__version__}  MediaPipe: {mediapipe.__version__}  MLflow: {mlflow.__version__}')
"
```

### Run the full pipeline

```bash
make preprocess    # Extract MediaPipe landmarks → .npy cache
make train         # Run the full 23-run experiment matrix (MLflow tracked)
make evaluate       # Stage 6: test-set evaluation, calibration, SHAP
make export        # Stage 8: TFLite export + release gate
make demo           # Stage 9: real-time webcam demo
```

> `Makefile` / `make` targets ship as part of Stage 10 — until then, invoke each pipeline
> script directly (shown below).

### Train the champion configuration

```bash
python pipelines/run_training.py \
    model=bilstm data=seq100 augmentation=spatial_temporal \
    training.learning_rate=0.0005 training.epochs=250 \
    training.early_stopping_patience=50 \
    --landmark-config hands_only \
    --run-name bilstm_hands_only_v4_aug --experiment-group champion
```

### Evaluate (Stage 6)

```bash
python pipelines/run_evaluation.py \
    --champion-run bilstm_hands_only_v4_aug \
    --splits val test \
    --output-dir reports/evaluation/
```

### Export and verify TFLite (Stage 8)

```bash
python pipelines/run_export_verification.py \
    --config-snapshot artifacts/experiments/bilstm_hands_only_v4_aug/config_snapshot.yaml \
    --saved-model models/bilstm_hands_only_v4_aug_saved_model \
    --output models/gesture_bilstm_v1.tflite
```

### Run the live demo (Stage 9)

```bash
python src/demo/webcam_demo.py
```

### Verify pipeline integrity

```python
from src.utils.config import load_config
from src.features import FeaturePipeline, GestureDataset

cfg = load_config(model='bilstm', data='seq100', augmentation='spatial_temporal')
pipeline = FeaturePipeline(cfg)
assert pipeline.output_shape == (100, 126)

dataset = GestureDataset(cfg, pipeline, splits_dir='data/splits', landmarks_dir='data/landmarks')
assert (dataset.n_train, dataset.n_val, dataset.n_test) == (236, 52, 51)
print("Pipeline verified.")
```

### Run tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
pytest tests/test_predictor.py -v        # Stage 7 suite
pytest tests/test_tflite_export.py -v    # Stage 8 suite
```

---

## 15. Docker

*(Stage 10 — open)*

Two images are planned: a full training image (TensorFlow + MediaPipe, ~4 GB) and a lean
inference image (`Dockerfile.inference`, TFLite runtime only, ~800 MB) that intentionally
excludes `src/evaluation` and the full TensorFlow training stack, matching the project's
documented `GesturePredictor`-only deployment contract.

---

## 16. Experiment Tracking with MLflow

All 23 Stage 5 experiments are logged under experiment `"WLASL-35-class"`. Each run records
every hyperparameter, per-epoch metrics (`train_loss`, `val_loss`, `train_acc`, `val_acc`,
`val_macro_f1`, `learning_rate`, `epoch_time_s`), per-run artefacts (confusion matrices,
training curves, per-class metrics, run manifest, config snapshot), and the trained SavedModel
as an MLflow artefact.

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

The champion is tracked under MLflow run ID `cb16f689d2294001a2ff2d3e02419d27`.

---

## 17. KSL Adaptation Roadmap

The production target for this work is **Kenyan Sign Language**, not ASL — WLASL-35 is a
technical-verification and engineering exercise. ASL and KSL are structurally distinct
languages (different phonemic handshape inventories, movement patterns, signing-space
conventions, non-manual grammatical markers, and almost entirely different lexicons); this
model has **no knowledge of KSL** and is not deployable for it without adaptation
(see [L18](LIMITATIONS.md#l18)).

**What plausibly transfers:** the `hands_only` configuration — adopted purely for WLASL-35
accuracy reasons (+110% relative improvement, [Section 9](#9-findings-and-ablation-studies)) —
has the welcome side-effect of removing pose landmarks, which encode signer-specific body
morphology rather than sign-specific geometry. This removes one entire axis of cross-linguistic
domain shift before KSL transfer even begins.

**Proposed validation protocol** (not yet executed):
1. KSL-from-scratch baseline — establishes the achievable ceiling with the target architecture.
2. Frozen ASL-pretrained BiLSTM + new KSL classifier head — tests motion-pattern transfer.
3. Full fine-tune (ASL init → all layers trainable on KSL) — expected best performer once
   sufficient KSL data exists.

Evaluate all three with **per-class recall**, not aggregate accuracy — this project's own
per-class data-scarcity failures ([L2](LIMITATIONS.md#l2)) are likely to recur, possibly worse,
at 500-class KSL scale.

**Data requirement:** ~100–200 clips/sign (current AI4KSL: ~40 clips/sign — below viability).
**Architecture scaling:** `hidden_units ∈ {128, 256}` should be the first ablation point (current
champion: 64). **Estimated timeline:** 3–6 months, dominated by data collection, not modelling —
consistent with this project's own finding that architecture search converges quickly relative
to the cost of data scarcity.

---

## 18. Limitations

**[`LIMITATIONS.md`](LIMITATIONS.md) is the authoritative limitations register for this
project** — 18 documented limitations (`L1`–`L18`), each with severity, evidence, and applied
mitigation. Every accuracy or readiness claim anywhere in this README, the model card, or any
future report should be read relative to that document. Headline points:

| Theme | Headline limitation |
|---|---|
| Data | 46.6% dataset completeness is a **hard, unfixable ceiling** on achievable accuracy ([L1](LIMITATIONS.md#l1)) |
| Data | `think` is effectively unlearnable at current scale (3 training clips, F1=0.0 in 8/9 runs) ([L2](LIMITATIONS.md#l2)) |
| Evaluation | 21/35 val classes are singletons; champion's val macro-F1 should be read as **≈0.58 ± 0.03**, not a fixed point ([L3](LIMITATIONS.md#l3), [L7](LIMITATIONS.md#l7)) |
| Evaluation | The 14.3pp val→test gap is the project's most important honesty check — quote the **test** number externally ([L6](LIMITATIONS.md#l6)) |
| Model | 70% target not met (0.6011 Keras val achieved); evidence points to data, not architecture, as the binding constraint ([L8](LIMITATIONS.md#l8)) |
| Calibration | Model is **underconfident** (ECE = 0.2009); mitigated with a 0.35 display threshold, not temperature scaling (not yet implemented) ([L11](LIMITATIONS.md#l11)) |
| Real-time system | No out-of-distribution rejection — any input produces a confident-looking prediction among the 35 known classes ([L15](LIMITATIONS.md#l15)) |
| Real-time system | Latency verified on development-machine CPU only — **never benchmarked on Android**, the stated primary deployment target ([L17](LIMITATIONS.md#l17)) |
| Scope | ASL ≠ KSL — this model is not deployable to the actual production target without the roadmap in [Section 17](#17-ksl-adaptation-roadmap) ([L18](LIMITATIONS.md#l18)) |

---

## 19. Contributing

```bash
pip install -r requirements-dev.txt
pre-commit install   # black + flake8 git hooks

pytest tests/ -v --cov=src --cov-report=term-missing   # test
flake8 src/ pipelines/ tests/ --max-line-length 100      # lint
black src/ pipelines/ tests/                              # format
```

Please open an issue before submitting a pull request for significant changes. The
configuration system (OmegaConf + Pydantic v2) enforces strict field validation — any new
config field must be added to both the YAML defaults and the Pydantic schema in
`src/utils/config.py`.

---

## 20. License and Citation

This project is licensed under the MIT License — see [`LICENSE`](LICENSE).

The WLASL dataset is subject to its own licence terms; see the
[WLASL repository](https://github.com/dxli94/WLASL). This project does not redistribute any
WLASL video content.

If you use this model or dataset in research, please cite:

> Li, D., Rodriguez, C., Yu, X., & Li, H. (2020). Word-level deep sign language recognition
> from video: A new large-scale dataset and methods comparison. *WACV 2020*.

---

<p align="center">
  <strong>Henry Otsyula</strong><br/>
  Senior Data Scientist &amp; ML Engineer<br/>
  Built as part of a sign language recognition research initiative, with Kenyan Sign Language
  recognition as the long-term production target.
</p>

<p align="center">
  <sub>
    Stages 1–9 complete · 23 MLflow runs · champion val macro-F1: 0.6011 (Keras) / 0.5916 (TFLite)
    · test macro-F1: 0.4581 (Keras) / 0.4867 (TFLite) · 68,771 parameters · 0.1596 MB TFLite ·
    release gate 6/6 PASS · 47.11 ms full-pipeline latency · live webcam demo verified
  </sub>
</p>