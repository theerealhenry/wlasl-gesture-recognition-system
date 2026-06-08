# LIMITATIONS.md
## WLASL Gesture Recognition Pipeline — Known Limitations, Constraints, and Mitigations

**Project:** WLASL-35-Class Gesture Recognition System  
**Author:** Henry Otsyula  
**Last updated:** 2026-06-07 (post-Stage 4 validation)  
**Status:** Active — updated after each completed stage

---

## Overview

This document is the authoritative record of every known limitation, constraint, data quality issue, and design trade-off in the WLASL gesture recognition pipeline. It is intended to accompany the final submission and one-page report, providing the scientific context necessary to interpret all reported accuracy numbers honestly.

Every accuracy figure reported by this project must be understood relative to the constraints documented here. A model that achieves X% validation accuracy on a signer-independent held-out set using only 236 training clips across 35 classes is a materially different result from the same X% on a larger, cleaner dataset — and this document exists to make that context explicit.

---

## 1. Data Limitations

### 1.1 Dataset Completeness: 46.6% Recovery Rate

**Severity: CRITICAL**

The WLASL v0.3 dataset lists 751 clip entries across the 35 selected signs. Only **350 clips (46.6%)** were recoverable from disk. The remaining **401 clips (53.4%) are permanently inaccessible** — the original YouTube source URLs are dead and the videos have been removed from the platform.

This is not a pipeline failure. It is a structural property of the WLASL dataset and is documented as a known limitation in the original WLASL paper. No action can recover these clips without re-recording from signers.

**Consequences:**
- The working training set is 236 clips across 35 classes — a mean of **6.7 clips per class**, far below the 20-clip minimum threshold flagged by the Stage 1 validator.
- The 46.6% completeness rate is a hard ceiling on achievable accuracy. No model architecture improvement can compensate for insufficient training data.
- The accuracy targets (≥70% signer-independent validation accuracy) are ambitious relative to this dataset scale and should be interpreted as demonstrating the upper bound achievable under these constraints, not as a benchmark against resource-rich production systems.

**Mitigation applied:**
- Signer-aware splits ensure every clip in the working set is used efficiently.
- Aggressive spatial-temporal augmentation generates additional training variation from each clip.
- Class weight balancing ensures rare classes receive proportionally higher gradient contribution.
- The yt-dlp download fallback (`run_preprocessing.py --download-missing`) was implemented but not exercised — it should be re-attempted if any source URLs become newly accessible.

---

### 1.2 Severely Limited Per-Class Sample Counts

**Severity: HIGH**

| Sign | Train clips | Val clips | Test clips | Total |
|---|---|---|---|---|
| clothes | 2 | 1 | 2 | 5 |
| birthday | 4 | 1 | 1 | 6 |
| book | 4 | 1 | 1 | 6 |
| name | 4 | 1 | 1 | 6 |
| think | 3 | 2 | 2 | 7 |

Five signs have 6 or fewer total clips and 4 or fewer training clips. For `clothes` specifically, 2 training clips is insufficient to learn generalised features — the model is at high risk of either overfitting to those 2 signers' styles or failing the class entirely.

Additionally, **11 clips listed in the split CSVs were never extracted in Stage 3** (dead URLs discovered after Stage 1 splitting). These 11 absences are distributed unevenly across signs, amplifying the per-class imbalance from a 3.20× ratio (Stage 1 estimate) to **6.50× actual** (Stage 4 measured, based on 236 loaded training clips).

**Consequences:**
- Per-class F1 for `clothes`, `think`, `birthday`, `name`, and `book` may be zero or near-zero in Stage 5 results, even with class weight balancing.
- Overall validation accuracy masks this failure mode — a model reporting 65% overall accuracy may have 0% accuracy on 3–5 classes.
- The 6.50× class weight ratio is the largest the pipeline will encounter; the effective learning rate for `clothes` is 6.5× that of `before` during training.

**Mitigation applied:**
- `class_weight_balancing=True` in all training configs (mandatory, confirmed in all 12+ Stage 5 runs).
- **Primary metric is macro-F1, not overall accuracy.** Per-class F1 is monitored individually for the bottom quintile.
- Stage 6 evaluation will produce an explicit per-class metrics table flagging classes with insufficient validation data.

---

### 1.3 Validation Set Singleton Classes

**Severity: HIGH**

**21 of 35 sign classes have exactly 1 validation clip.** These classes are: birthday, black, blue, book, can, chair, change, clothes, color, computer, eat, family, finish, friend, give, house, many, mother, name, now, thanksgiving.

A single validation clip cannot produce a meaningful per-class accuracy or F1 estimate — it is either 0% (missed) or 100% (correct), with no statistical reliability. Macro-averaged F1 across all 35 classes will therefore have high variance driven by the binary outcomes of 21 singleton classes.

**Consequences:**
- Per-class validation metrics for the 21 singleton classes are unreliable as absolute performance estimates.
- Macro-F1 remains the primary metric because it weights all classes equally regardless of clip count, but its variance will be high.
- A model that correctly classifies 20 of 21 singleton classes but fails `clothes` will report the same macro-F1 as a model that misclassifies a different singleton — the metric cannot distinguish which error pattern occurred.

**Mitigation applied:**
- Stage 6 evaluation will explicitly flag all singleton val classes in the per-class metrics table.
- Stage 6 `signer_analysis.py` will report per-signer accuracy as a complementary signal.
- The test set (51 clips) provides a secondary evaluation that is not used for model selection, helping distinguish genuine generalisation from overfitting to the validation distribution.

---

### 1.4 Signer Dominance in Training Split

**Severity: MEDIUM**

Signer 11 is the dominant contributor in **10 of 35 training signs**, appearing in 30–58% of training clips for those signs. If Signer 11's signing style is systematically different from the 7 held-out validation signers, the model may learn signer-11-specific features rather than generalised sign representations.

Four signs have a single training signer contributing >50% of clips:
- `go`: Signer 10 at 58.3% (7/12 training clips)
- `clothes`: Signer 11 at 50% (1/2 training clips — critical)
- `black`: Signer 11 at 50% (6/12 training clips)
- `birthday`: Signer 11 at 50% (2/4 training clips)

**Consequences:**
- Signs with high signer dominance are at elevated risk of poor generalisation to the validation signers.
- The Stage 5 training/validation accuracy gap (expected 15–25 percentage points) will be partially attributable to signer style memorisation rather than class confusion.

**Mitigation applied:**
- Spatial augmentation (mirror flip, ±5° rotation, Gaussian noise σ=0.01) is prioritised for the high-dominance tier.
- The clip-level spatial flip safety check (both hands present in >30% of frames) prevents anatomically implausible augmentations.
- Stage 6 signer analysis will produce per-signer accuracy breakdowns specifically for the 10 Signer-11-dominant classes.

---

### 1.5 YouTube-Sourced Video Quality Heterogeneity

**Severity: MEDIUM**

WLASL clips were sourced from YouTube signing dictionaries, educational videos, and Deaf community content. This produces highly variable recording conditions:

- Variable camera distance (close-up vs full-body shots)
- Variable lighting (indoor, outdoor, artificial, natural)
- Variable background (plain wall vs cluttered room)
- Variable filming angle (frontal, slight angle, oblique)
- Variable compression quality (high-bitrate source vs heavily compressed re-upload)

MediaPipe Holistic's hand detection confidence varies significantly with these factors. The global hand detection rate of 64.72% (Notebook 03) reflects this heterogeneity — some signers achieve >90% detection rates while others fall to 30–35%.

**Consequences:**
- Zero-fill frames (detection failures) are not uniformly distributed — some clips have concentrated gaps that may bisect the sign's most discriminative motion phase.
- The train/val accuracy gap will partly reflect the fact that the 7 validation signers have different environmental conditions from the 31 training signers.

**Mitigation applied:**
- Zero-fill frames are preserved semantically (not imputed) — the LSTM learns that absent landmarks are an expected signal pattern.
- Temporal jitter augmentation (10% frame dropout) trains the model to be robust to random detection gaps.
- Speed jitter augmentation addresses the temporal rate heterogeneity across recording conditions.

---

## 2. Pipeline Limitations

### 2.1 Sequence Length Truncation Rate: 97%

**Severity: HIGH**

At the primary sequence length of 60 frames, **97% of clips (329/339) are truncated** by centre-cropping. The mean number of frames removed per truncated clip is 21.5, representing approximately 32% of mean clip content discarded.

The worst-case clips lose 70% of their content:
- One clip: 197 frames → 60 frames retained (137 frames, 70% removed)
- Multiple clips require seq_len ≥ 133 for full coverage

**The centre-crop strategy removes preparatory and release movements from both temporal ends of the sign.** While these phases are less discriminative than the peak motion phase, they may contain sign-onset cues that the LSTM could use for rapid classification.

**Consequences:**
- Stage 5 Group 3 ablation (seq_len ∈ {20, 30, 40, 60, 80, 100}) is essential — seq_len=80 covers 70% of clips fully and 95.8% of mean sign content, representing the most likely source of material accuracy improvement.
- At seq_len=60, the model sees only the temporal core of each sign. Signs where the discriminative motion occurs in the preparatory phase (frame positions 0–10 in a 67-frame clip) will be systematically harder to classify.

**Mitigation applied:**
- Centre-crop strategy preferentially retains the temporal midpoint, where peak discriminative motion is concentrated.
- Truncation statistics are logged per-clip and included in pipeline metadata for Stage 6 interpretation.
- The Stage 5 Group 3 ablation directly addresses this by comparing all 6 sequence lengths.

---

### 2.2 MediaPipe Holistic Detection Failures

**Severity: MEDIUM**

MediaPipe Holistic fails to detect hands in 35.28% of decoded frames on average (global both-hands-absent rate). Right-hand detection failures average 37.60% across the dataset. These failures occur when:

- Hand motion is too fast (motion blur exceeds MediaPipe's tracking tolerance)
- Hands are partially occluded (one hand behind the body during two-handed signs)
- Unusual camera angles exceed MediaPipe's frontal-pose assumption
- Poor lighting reduces contrast between skin and background

Detection failures are zero-filled at the frame level and are **semantically distinguishable from intentional one-handed signs** (which have LH absent in 100% of frames vs sporadic detection gaps). However, the LSTM cannot perfectly distinguish between "left hand genuinely absent" (one-handed sign) and "left hand present but undetected" (detection failure).

**Consequences:**
- Detection failures introduce noise into the temporal feature sequence that the model must learn to be robust to.
- Signs with high detection failure rates (drink: 28.6% both-absent even in passing clips) will have noisier temporal representations and lower per-class accuracy.

**Mitigation applied:**
- Temporal jitter augmentation trains the model to be robust to random detection gaps.
- Speed jitter produces clips with different temporal densities, reducing over-reliance on any specific frame's detection state.
- The v1.2 dual-criterion skip policy (minimum 15 detected frames; maximum 95% catastrophic missing rate) ensures all training clips have sufficient usable content.

---

### 2.3 Wrist-Relative Normalisation: Residual Inter-Signer Variability

**Severity: LOW**

Wrist-relative normalisation removes the absolute screen position of the signer's hand but cannot remove all inter-signer variability. The residual variability after normalisation includes:

- **Hand scale differences**: a signer with a large hand span produces fingertip landmarks at ±0.035 wrist-relative units; a compact signer produces ±0.010 units. Scale is not normalised.
- **Arc geometry**: different signers execute the same sign with different spatial arcs — the wrist path may be a tight circle for one signer and a wide sweep for another.
- **Sign speed**: execution rate varies 2–3× across signers for the same sign (Notebook 03 F9).

These residual variations are the core generalisation challenge. Wrist-relative normalisation is a necessary but not sufficient preprocessing step.

**Mitigation applied:**
- Rotation augmentation (±5°) addresses arc geometry variation at the finger level.
- Gaussian noise (σ=0.01) provides scale regularisation.
- Speed jitter (rate ∈ [0.7, 1.3]) addresses execution rate variation.

---

### 2.4 Z-Coordinate Reliability

**Severity: LOW**

MediaPipe's z-coordinates (depth estimates) carry only ~4% of the signal magnitude of xy coordinates and have a long outlier tail extending to ±0.22 in raw form. The z-distribution is non-Gaussian and includes physically implausible values produced by MediaPipe's depth estimation at frame boundaries and during fast motion.

Z-clipping at ±0.10 is applied (37.41% of all z-entries affected, including zero-fill frames), but the retained z-signal is weaker and less reliable than xy. The LSTM may down-weight z-coordinates implicitly through training, but this is not guaranteed.

**Mitigation applied:**
- Z-coordinates are retained (dropping z slightly reduces Fisher ratio from 0.5492 to 0.4116).
- Soft clipping at ±0.10 removes physically implausible outliers without affecting the core distribution.
- The landmark configuration ablation (Group 4) will empirically determine whether z contributes positively to classification accuracy.

---

### 2.5 Augmentation: speed_jitter Zero-Fill Invariant (Boundary Case)

**Severity: LOW**

The `speed_jitter` transform uses linear interpolation to restore fast-resampled clips to the original sequence length. At the boundary between a detected frame and a zero-fill frame, interpolation produces non-zero values in what were originally zero-fill slots — a partial violation of the zero-fill invariant.

**The fix applied:** Zero-aware interpolation forces output frames to zero when both surrounding source frames are zero for a given component slot. Frames at the detected↔zero boundary receive interpolated blended values, which is accepted as a valid transition representation.

**Residual limitation:** The zero-fill invariant for speed_jitter is guaranteed only for "always-absent" slots (zero in all source frames). Interior zero-fill regions (detection failures within a clip) may shift temporally after resampling — this is correct resampling behaviour, not a bug, but it means the LSTM receives slightly different zero-fill patterns in augmented vs non-augmented clips.

**Mitigation applied:**
- Zero-aware interpolation confirmed working across 20 random seeds on synthetic always-absent clips.
- The test suite (`test_augmentation.py::test_speed_jitter_slot_zero_fill_invariant`) formally verifies this behaviour.

---

## 3. Model and Architecture Limitations

### 3.1 Landmark-Based Representation: Fundamental Ceiling

**Severity: MEDIUM**

This pipeline replaces raw video pixels with MediaPipe Holistic skeletal landmarks. This architectural choice provides significant benefits (small model size, CPU-deployable, interpretable) but imposes a fundamental ceiling:

- **MediaPipe's detection failures are not recoverable.** When MediaPipe fails to detect a hand, the information is permanently lost. A CNN operating on raw pixels could potentially recover the hand location even when explicit keypoint detection fails.
- **Facial grammar markers are excluded.** ASL uses facial expressions (eyebrow raise, mouth morphemes, head tilt) as grammatical markers that modify sign meaning. MediaPipe Holistic provides face landmarks, but they are not included in the 225-element feature vector. This is appropriate for WLASL word-level recognition but would be a significant limitation for sentence-level or grammatically-aware recognition.
- **3D handshape is approximated.** MediaPipe provides x, y, z coordinates, but z is a weak depth estimate rather than true 3D reconstruction. Fine-grained handshapes that differ primarily in depth configuration may be confused.

**Mitigation applied:**
- The landmark-based approach is a deliberate design choice for edge deployment, not an oversight. Its limitations are documented here for transparency.
- For production KSL recognition, consider whether facial grammar markers are required for the target vocabulary.

---

### 3.2 LSTM vs Transformer: Architecture Trade-off

**Severity: LOW**

The project uses LSTM, GRU, and BiLSTM architectures. Transformer-based sequence models (e.g., TF-Pose, SignBERT) have demonstrated superior performance on sign language recognition benchmarks but are excluded here due to:

- **Model size constraint:** ≤10 MB post-quantisation TFLite target
- **Latency constraint:** ≤100ms end-to-end on CPU
- **Data constraint:** Transformer architectures require substantially more training data to outperform LSTMs; with ~236 training clips, a Transformer would likely underfit

The BiLSTM + spatial-temporal augmentation configuration is the highest-capacity architecture defensible under these constraints.

---

### 3.3 Sequence Length vs Latency Trade-off

**Severity: LOW**

Every 20-frame increase in sequence length adds approximately the same inference latency increment. The project targets ≤100ms end-to-end on CPU at seq_len=60. If Stage 5 Group 3 results show that seq_len=80 or seq_len=100 materially improves accuracy, a latency vs accuracy trade-off decision will be required before TFLite export.

At seq_len=100 with a 2-layer BiLSTM (hidden=128), the estimated inference time on CPU is ~60–80ms — still within the 100ms budget. This estimate should be verified empirically in Stage 8 benchmarking.

---

## 4. Evaluation Limitations

### 4.1 Signer-Independent Validation: Conservative but Honest

**Severity: LOW (by design)**

The signer-aware split ensures **zero signer overlap** between train, val, and test. This makes the validation accuracy a genuinely conservative, honest estimate of generalisation to unseen signers. It will systematically produce lower accuracy numbers than a naïve random split applied to the same clips.

For context: the original WLASL benchmark results used random splits that do not enforce signer independence. The signer-independent accuracy reported by this pipeline is not directly comparable to those benchmark numbers — ours is more rigorous and more conservative.

**Implication for the report:** The one-page report must explicitly state "X% signer-independent validation accuracy" and contrast this with the random-split baseline to contextualise the result for Abel Holla's evaluation.

---

### 4.2 Validation Set Size: 52 Clips, 7 Signers

**Severity: MEDIUM**

With only 52 validation clips across 35 classes (mean 1.49 clips/class) and 7 signers, the validation accuracy estimate has high variance. A single misclassification changes macro-F1 by approximately 2 percentage points. A lucky or unlucky run of predictions on the 21 singleton classes can shift overall accuracy by ±5%.

**Consequences:**
- Reported validation accuracy should be treated as a point estimate with an implicit confidence interval of approximately ±3–5 percentage points.
- Stage 5 MLflow runs should use the same random seed and be compared primarily by training trajectory shape, not by single-epoch val accuracy differences of <2%.
- The test set (51 clips, 7 signers) provides a final unbiased evaluation that is not used for model selection.

---

### 4.3 No External Benchmark Comparison

**Severity: LOW**

This pipeline cannot be directly compared to published WLASL-100 or WLASL-2000 benchmarks because:
- Those benchmarks use all 100 or 2000 classes, not the 35 selected here
- Published benchmarks used random splits without signer independence enforcement
- The 46.6% data recovery rate means the working dataset is a strict subset of even the WLASL-35 full inventory

The appropriate comparison baseline for this project is the Dense feedforward baseline (Group 1) and the no-augmentation LSTM baseline (Group 2), both trained on the same 236-clip training set.

---

## 5. Deployment Limitations

### 5.1 ASL to KSL Transfer Gap

**Severity: HIGH (for production deployment)**

This pipeline was developed on the WLASL dataset, which contains **American Sign Language (ASL)**. The target deployment language is **Kenyan Sign Language (KSL)**. ASL and KSL are structurally different languages with different:

- Handshapes (distinct phonemic inventories)
- Movement patterns and spatial conventions
- Non-manual markers (facial grammar)
- Lexical items (most signs are completely different)

**A model trained on WLASL ASL data will not generalise to KSL without significant adaptation.** Transfer from ASL provides low-level feature detection capabilities (hand segmentation, wrist-relative geometry extraction via MediaPipe) but the high-level sign recognition patterns learned from ASL are largely inapplicable to KSL.

**Recommended KSL adaptation strategy:**
1. Collect KSL training data (minimum ~100 clips/sign for the target vocabulary; AI4KSL dataset provides ~40 clips/sign which is below viability for a 500-sign system at 85% accuracy)
2. Train three models: KSL-only baseline, ASL-frozen feature extractor + KSL classifier head, ASL fully fine-tuned on KSL
3. Compare on held-out KSL test set with signer independence enforced
4. Estimated timeline for a 500-sign KSL system at 85% accuracy: 3–6 months, primarily driven by data collection

---

### 5.2 Real-Time Performance: CPU-Only Target

**Severity: LOW**

The ≤100ms end-to-end latency target is for **CPU-only inference** on the primary Android mobile deployment target. This constrains:

- Model architecture to lightweight LSTM/GRU (not Transformer)
- Sequence length to ≤100 frames at current throughput estimates
- TFLite dynamic-range quantisation (4× size reduction, minimal accuracy impact)

If GPU acceleration becomes available on the deployment device, the architecture constraints can be relaxed and a larger model can be evaluated.

---

### 5.3 Single-Sign Prediction Only

**Severity: MEDIUM (for production)**

This pipeline performs **word-level isolated sign recognition** — it classifies a single, pre-segmented sign clip. It does not:

- Perform continuous sign language recognition (connected signing without explicit boundaries)
- Handle co-articulation between consecutive signs
- Segment signs from continuous video streams (a separate detection problem)
- Recognise sentence-level meaning or grammatical structure

The webcam demo (Stage 9) uses a sliding window approach with majority voting over 5 frames, which approximates real-time recognition but is not a continuous recognition system. For production deployment, sign segmentation must be solved as a separate upstream module.

---

## 6. Known Data Pipeline Issues (Resolved)

The following issues were identified and resolved during Stages 1–4. They are documented here for completeness.

| Issue | Stage identified | Resolution |
|---|---|---|
| v1.1 30% skip threshold produced 76% skip rate | Notebook 02 | v1.2 dual-criterion policy: min 15 detected frames + 95% catastrophe filter |
| Leading-zero video_id mismatch between split CSVs and inventory | Notebook 04 | `fix_leading_zero_video_ids()` normalised all IDs to integer-string form |
| Individual hand missing rates zeroed in landmark_inventory.csv | Notebook 02 | Structural issue noted; `missing_both_pct` (used for skip policy) is correct |
| speed_jitter interpolation violated zero-fill invariant at boundaries | Notebook 04 | Zero-aware interpolation: output forced to zero when both surrounding frames are zero |
| stray debug `print()` in AugmentationPipeline.__call__() | Stage 4 review | Removed |
| GestureDataset epoch_counter not varied when model.fit() called once | Stage 4 review | tf.Variable counter + per-epoch load_split() training loop contract enforced |

---

## 7. Summary: Accuracy Interpretation Guide

Any accuracy number reported by this pipeline should be interpreted in the context of all of the above. The following table provides the key contextual facts for the one-page report and theoretical assessment answers:

| Context factor | Value | Implication |
|---|---|---|
| Training clips | 236 (loaded) | ~6.7 clips/class — severely data-limited |
| Validation clips | 52 | High-variance estimate; ±3–5pp implicit CI |
| Val singleton classes | 21/35 | Per-class val metrics unreliable for 60% of classes |
| Signer independence | Enforced (zero overlap) | Accuracy is conservative vs random-split baselines |
| Dataset completeness | 46.6% | Hard ceiling; no architecture change can compensate |
| Class weight ratio | 6.50× | Aggressive rebalancing required and applied |
| Primary metric | Macro-F1 | Not overall accuracy — masks per-class failures |
| Augmentation | Full spatial-temporal chain | Without this, overfitting gap expected 20+pp |
| Target language | ASL (not KSL) | Substantial adaptation required for production |

**What a result of X% signer-independent macro-F1 means:**  
A model achieving X% macro-F1 on 52 validation clips from 7 signers never seen during training, using only 236 training clips of American Sign Language, with full spatial-temporal augmentation and class weight balancing, validated against a 35-class vocabulary where 21 classes have exactly 1 validation clip. This is an honest, conservative, and methodologically rigorous estimate — not an inflated benchmark number.

---

*This document will be updated after Stage 5 (training results), Stage 6 (evaluation), and Stage 8 (TFLite export) to reflect any additional limitations discovered during those stages.*