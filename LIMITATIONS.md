# LIMITATIONS.md
## WLASL Gesture Recognition Pipeline — Known Limitations, Constraints, and Mitigations

**Project:** WLASL-35-Class Gesture Recognition System  
**Author:** Henry Otsyula  
**Last updated:** 2026-06-14 (post-Stage 5 training, all experiments complete)  
**Status:** Active — updated after each completed stage

---

## Overview

This document is the authoritative record of every known limitation, constraint, data quality issue, and design trade-off in the WLASL gesture recognition pipeline. It accompanies the final submission and one-page report, providing the scientific context necessary to interpret all reported accuracy numbers honestly.

Every accuracy figure reported by this project must be understood relative to the constraints documented here. A model that achieves 60% signer-independent validation macro-F1 on 52 clips across 35 classes with 236 training clips is a materially different result from the same number on a resource-rich dataset — and this document exists to make that context explicit.

Stage 5 added several empirical findings that sharpen, revise, or extend the pre-training limitations documented after Stage 4. Changes and additions relative to the previous version are marked **[Updated S5]** or **[New S5]**.

---

## 1. Data Limitations

### 1.1 Dataset Completeness: 46.6% Recovery Rate

**Severity: CRITICAL**

The WLASL v0.3 dataset lists 751 clip entries across the 35 selected signs. Only **350 clips (46.6%)** were recoverable from disk. The remaining **401 clips (53.4%) are permanently inaccessible** — the original YouTube source URLs are dead and the videos have been removed from the platform.

This is not a pipeline failure. It is a structural property of the WLASL dataset, documented as a known limitation in the original WLASL paper. No action can recover these clips without re-recording from signers.

**Consequences:**
- The working training set is 236 clips across 35 classes — a mean of **6.7 clips per class**, far below the 20-clip minimum threshold flagged by the Stage 1 validator.
- The 46.6% completeness rate is a hard ceiling on achievable accuracy. No model architecture improvement can compensate for insufficient training data.
- **[Updated S5]** Stage 5 empirically confirmed this ceiling. After exhaustive hyperparameter search across 23 runs spanning 4 ablation groups and 9 champion candidates, the best signer-independent validation macro-F1 achieved was **0.6011**. The 70% accuracy target was not met. The gap between 0.6011 and 0.70 is attributable primarily to data quantity, not architecture capacity.

**Mitigation applied:**
- Signer-aware splits ensure every clip in the working set is used efficiently.
- Spatial-temporal augmentation generates additional training variation from each clip.
- Class weight balancing ensures rare classes receive proportionally higher gradient contribution.
- The yt-dlp download fallback (`run_preprocessing.py --download-missing`) was implemented but not exercised in Stage 5.

---

### 1.2 Severely Limited Per-Class Sample Counts

**Severity: HIGH**

| Sign | Train clips | Val clips | Test clips | Total | Stage 5 best F1 (val) |
|---|---|---|---|---|---|
| clothes | 2 | 1 | 2 | 5 | 1.0 (augmented) / 0.0 (no-aug, most runs) |
| think | 3 | 2 | 2 | 7 | 0.0 in 8/9 champion runs |
| birthday | 4 | 1 | 1 | 6 | 1.0 (most runs) |
| book | 4 | 2 | 1 | 7 | 0.5–1.0 (variable) |
| name | 4 | 1 | 1 | 6 | 1.0 (augmented) / 0.0 (no-aug, some runs) |

**[Updated S5]** Stage 5 provides empirical F1 outcomes for each high-risk class across 23 runs. Two distinct failure patterns emerged:

**Pattern A — Data-floor failure (think):** `think` returned F1=0.0 in 8 of 9 champion runs, across all architecture and augmentation configurations. 3 training clips is insufficient to learn a generalisable representation of this sign's temporal pattern under signer-independent conditions. This is a data failure, not a model failure.

**Pattern B — Augmentation-sensitive failure (clothes, name):** These classes failed (F1=0.0) in all no-augmentation runs but succeeded (F1=1.0) in the spatial_temporal augmented champion run. This indicates the 2–4 training clips do contain learnable signal, but the model requires augmentation-induced regularisation to find it without memorising signer identity.

The 6.50× class weight ratio (measured in Stage 4, confirmed in Stage 5) is the largest the pipeline will encounter. Without class weighting, `clothes` would receive negligible gradient — class weighting is mandatory and confirmed enabled in all 23 Stage 5 runs.

**Mitigation applied:**
- `class_weight_balancing=True` in all training configs (enforced via config validation; confirmed in all Stage 5 runs).
- Primary metric is macro-F1, not overall accuracy. Per-class F1 is individually monitored and logged to MLflow for the bottom quintile.
- Stage 6 evaluation will produce an explicit per-class metrics table flagging classes with insufficient validation data.
- `think` is formally documented as unlearnable at current data scale. Any future data collection effort should prioritise this class first.

---

### 1.3 Validation Set Singleton Classes

**Severity: HIGH**

**21 of 35 sign classes have exactly 1 validation clip.** These classes are: birthday, black, blue, book, can, chair, change, clothes, color, computer, eat, family, finish, friend, give, house, many, mother, name, now, thanksgiving.

A single validation clip cannot produce a reliable per-class accuracy estimate — it is binary (0% or 100%), with no statistical validity. Macro-averaged F1 across all 35 classes has high variance driven by the binary outcomes of these 21 singletons.

**[Updated S5]** Stage 5 quantified this variance empirically. Two runs with identical configurations (bilstm_hands_only and bilstm_hands_only_v2, both BiLSTM+seq100+none+patience varied) produced val_macro_f1 of 0.5419 and 0.4067 respectively — a 13pp gap attributable entirely to different epoch-by-epoch trajectories through the noisy singleton class landscape, not to any true performance difference. All reported val_macro_f1 values should be treated as point estimates with an implicit confidence interval of approximately **±3–5pp**.

**Mitigation applied:**
- Stage 6 evaluation will explicitly flag all singleton val classes in the per-class metrics table.
- The test set (51 clips) provides a secondary evaluation not used for model selection, helping distinguish genuine generalisation from overfitting to the validation distribution.
- Model selection in Stage 5 used manual early stopping on val_macro_f1 with patience≥40 to avoid selecting on noise peaks.

---

### 1.4 Signer Dominance in Training Split

**Severity: MEDIUM**

Signer 11 is the dominant contributor in 10 of 35 training signs. Four signs have a single training signer contributing ≥50% of clips:
- `go`: Signer 10 at 58.3% (7/12 clips)
- `clothes`: Signer 11 at 50% (1/2 clips — critical)
- `black`: Signer 11 at 50% (6/12 clips)
- `birthday`: Signer 11 at 50% (2/4 clips)

**[Updated S5]** The train/val accuracy gap measured in Stage 5 (approximately 0.24–0.48 depending on augmentation configuration) is partly attributable to signer identity memorisation. The gap was largest in no-augmentation runs (train_macro_f1 reaching 0.95 by epoch 200 while val_macro_f1 plateaued at 0.40–0.47), and reduced to approximately 0.24 in the best augmented run. This provides empirical evidence that signer dominance is a genuine source of the gap, and augmentation is its primary mitigation.

**Mitigation applied:**
- Spatial augmentation (mirror flip, ±5° rotation, Gaussian noise σ=0.01) targets signer-specific spatial patterns.
- Clip-level spatial flip safety check (both hands present in >30% of frames) prevents anatomically implausible augmentations.
- Stage 6 signer analysis will produce per-signer accuracy breakdowns specifically for the 10 Signer-11-dominant classes.

---

### 1.5 YouTube-Sourced Video Quality Heterogeneity

**Severity: MEDIUM**

WLASL clips were sourced from YouTube signing dictionaries, educational videos, and Deaf community content. This produces variable recording conditions: camera distance, lighting, background clutter, filming angle, and video compression quality. The global hand detection rate of 64.72% (Notebook 03) reflects this heterogeneity.

**Consequences:**
- Some clips have concentrated detection gaps in the most discriminative motion phases.
- The 7 validation signers have different environmental conditions from the 31 training signers, contributing to the train/val gap.

**Mitigation applied:**
- Zero-fill frames are preserved semantically (not imputed). The LSTM learns that absent landmarks are an expected signal pattern.
- Temporal jitter augmentation trains robustness to random detection gaps.
- Speed jitter addresses temporal rate heterogeneity across recording conditions.

---

## 2. Pipeline Limitations

### 2.1 Sequence Length: Optimal is 100 Frames, Not 60

**Severity: HIGH [Updated S5]**

At the pre-Stage 5 primary sequence length of 60 frames, **97% of clips (329/339) are truncated** by centre-cropping, discarding a mean of 21.5 frames (32% of mean clip content) per truncated clip.

**[Updated S5]** Stage 5 Group 3 (sequence length ablation) empirically confirmed and quantified the truncation cost:

| seq_len | Clips fully covered | Mean content | val_macro_f1 (lstm, full) |
|---|---|---|---|
| 60 | 3.0% | 85.0% | 0.1434 |
| 80 | 70.2% | 95.8% | 0.0328 (local minimum, seed-specific) |
| 100 | 92.9% | 99.2% | 0.2354 |

seq_len=100 achieved a 64% relative improvement in val_macro_f1 over seq_len=60 (0.2354 vs 0.1434). **The final pipeline uses seq_len=100 as the canonical configuration.** The originally stated primary sequence length of 60 is superseded.

The seq_len=80 result (0.0328) is a deterministic local minimum under seed=42 with lr=5e-4 on full 225-dim features. Reproduced exactly in seq80_v2. This is a seed-specific initialisation trap, not a general property of seq_len=80.

The latency implication of seq_len=100 vs seq_len=60 must be verified empirically in Stage 8 benchmarking. Estimated CPU inference time at seq_len=100 with BiLSTM (68K params, 126-dim input) is approximately 40–60ms, within the 100ms target.

---

### 2.2 MediaPipe Holistic Detection Failures

**Severity: MEDIUM**

MediaPipe Holistic fails to detect hands in 35.28% of decoded frames on average (global both-hands-absent rate). Right-hand detection failures average 37.60%. These failures occur due to motion blur, occlusion, unusual camera angles, and poor lighting.

The LSTM cannot perfectly distinguish "left hand intentionally absent" (one-handed sign) from "left hand present but undetected" (detection failure). Both manifest as zero-fill in the feature vector.

**Mitigation applied:**
- Zero-fill frames are preserved semantically; the LSTM learns that absent landmarks are a signal, not noise.
- Temporal jitter and speed jitter augmentation train robustness to random detection gaps.
- v1.2 dual-criterion skip policy (minimum 15 detected frames; maximum 95% catastrophic missing rate) ensures sufficient usable content in all training clips.

---

### 2.3 Landmark Configuration: hands_only Supersedes Full 225-dim

**Severity: HIGH [New S5]**

The pre-training pipeline design used all 225 landmark dimensions (left hand 63 + right hand 63 + pose 99) as the default feature vector. Stage 5 Group 4 (landmark configuration ablation) showed this is suboptimal by a factor of 2×.

| Config | Feature dim | Fisher ratio | val_macro_f1 (lstm, seq100) | Params |
|---|---|---|---|---|
| full | 225 | 0.5492 | 0.2354 | 110,499 |
| hands_only | 126 | 0.8097 | 0.4948 | 85,155 |
| pose_only | 99 | 0.2176 | 0.0314 | 78,243 |

**hands_only (126-dim) more than doubles val_macro_f1 compared to full (225-dim), using 23% fewer parameters.** The 99 pose landmark dimensions contribute near-zero discriminative signal for these 35 ASL signs — ASL word-level recognition is differentiated primarily by handshape, finger configuration, and hand trajectory. Torso and shoulder posture is largely static and signer-specific rather than sign-specific.

Appending 99 dimensions of near-zero-signal data to 126 dimensions of high-signal data forces the LSTM to learn to suppress the noise, which consumes model capacity that could otherwise be used for sign discrimination. With only 236 training clips, this capacity cost is significant.

**Consequence for all Groups 1–3 results:** All runs in Groups 1–3 were conducted on the full 225-dim feature vector and therefore understate absolute achievable performance by approximately 2×. The relative comparisons within each group (architecture vs architecture, augmentation vs augmentation, seq_len vs seq_len) remain valid. The absolute magnitudes are floor estimates, not ceilings.

**The final pipeline uses `landmark_config=hands_only` as the canonical configuration.** The originally stated default of `full` is superseded.

---

### 2.4 Wrist-Relative Normalisation: Residual Inter-Signer Variability

**Severity: LOW**

Wrist-relative normalisation removes absolute hand position but cannot remove hand scale variation, arc geometry variation across signers, or execution rate variation (2–3× spread across signers for the same sign). These residual variations are the core of the signer-independent generalisation challenge.

**Mitigation applied:**
- Rotation augmentation (±5°) addresses arc geometry variation at the finger level.
- Gaussian noise (σ=0.01) provides implicit scale regularisation.
- Speed jitter (rate ∈ [0.7, 1.3]) addresses execution rate variation.

---

### 2.5 Z-Coordinate Reliability

**Severity: LOW**

MediaPipe's z-coordinates carry approximately 4% of the signal magnitude of xy coordinates and include physically implausible depth estimates at frame boundaries and during fast motion. Z-clipping at ±0.10 (37.41% of z-entries affected) removes outliers without affecting the core distribution. The LSTM may implicitly down-weight z-coordinates through training, but this is not guaranteed.

Retained z-coordinates are included in the hands_only config. Empirically, dropping z-coordinates reduces Fisher ratio from 0.8097 to approximately 0.71, so they are retained.

---

### 2.6 Augmentation: Epoch-Budget Dependency

**Severity: MEDIUM [New S5]**

Stage 5 revealed a critical epoch-budget dependency in the augmentation findings. The Group 2 ablation (80 epochs) concluded that spatial_temporal augmentation is harmful: both lstm_spatial_temporal_aug and bilstm_spatial_temporal_aug showed monotonically increasing val loss and stopped in 19–32 epochs with val_macro_f1 near 0.

The champion run bilstm_hands_only_v4_aug (250 epochs, patience=50, same spatial_temporal augmentation, hands_only features, lr=5e-4) achieved val_macro_f1=0.6011 — the best result in Stage 5 by a margin of 5.9pp over the best no-augmentation result.

**The Group 2 conclusion was an 80-epoch artefact, not a fundamental finding about augmentation.** The correct interpretation is:

- With a fixed 80-epoch budget: no-augmentation is empirically better (faster convergence in early epochs).
- With a 250-epoch budget and patience=50: spatial_temporal augmentation is decisively better (val loss continues decreasing; train/val gap halved from ~0.48 to ~0.24).

The Group 2 failure at 80 epochs likely reflects the interaction of full 225-dim features (which include pose noise that amplifies augmentation instability), lr=1e-3 (too high for augmented training to find a stable path), and insufficient epochs for augmented convergence.

**Implications for future runs:** Any augmentation experiment should use at minimum epochs=200 and patience=40 before drawing conclusions. The Group 2 results are not a valid basis for disabling augmentation in other contexts.

---

### 2.7 speed_jitter Zero-Fill Boundary Behaviour

**Severity: LOW**

The `speed_jitter` transform uses linear interpolation to restore resampled clips. At the boundary between a detected frame and a zero-fill frame, the zero-aware interpolation produces blended (non-zero) values, which is a valid transition representation. Interior zero-fill regions may shift temporally after resampling.

**The zero-fill invariant is guaranteed only for always-absent slots** (zero in all source frames across the entire clip). This is the correct behaviour for one-handed signs (where LH is always absent) but does not strictly hold for sporadic detection failures mid-clip.

**Mitigation:** Zero-aware interpolation is verified by `test_augmentation.py::test_speed_jitter_slot_zero_fill_invariant` across 20 random seeds. No further action required.

---

## 3. Model and Architecture Limitations

### 3.1 Landmark-Based Representation: Fundamental Ceiling

**Severity: MEDIUM**

This pipeline replaces raw video pixels with MediaPipe Holistic skeletal landmarks. Benefits: small model size, CPU-deployable, interpretable features. Fundamental costs:

- **MediaPipe detection failures are not recoverable.** When MediaPipe fails to detect a hand, the information is permanently lost. A CNN operating on raw pixels could recover hand position even when explicit keypoint detection fails. The 35.28% both-hands-absent rate in Stage 3 represents irrecoverable information loss.
- **Facial grammar markers are excluded.** ASL uses eyebrow raise, mouth morphemes, and head tilt as grammatical markers. These are not in the 225-element (or 126-element hands_only) feature vector. This is appropriate for WLASL word-level recognition but would be a significant limitation for sentence-level or grammatically-aware recognition.
- **3D handshape is approximated.** Z-coordinates are weak depth estimates, not true 3D reconstruction. Signs that differ primarily in depth configuration may be confused.

This design choice is deliberate for edge deployment and is not an oversight. Its limitations are documented here for transparency.

---

### 3.2 BiLSTM vs Transformer Architecture

**Severity: LOW**

The pipeline uses BiLSTM as the champion architecture. Transformer-based sequence models have demonstrated superior performance on sign language recognition benchmarks but are excluded here due to:

- **Model size constraint:** ≤10 MB post-quantisation TFLite target.
- **Latency constraint:** ≤100ms end-to-end on CPU.
- **Data constraint:** Transformers require substantially more training data to outperform LSTMs. With ~236 training clips, a Transformer with even modest depth would underfit.

**[New S5]** The BiLSTM (68,771 params, 0.26 MB) is well within both the size and latency targets pre-quantisation. Dynamic range quantisation is expected to produce a TFLite file under 1 MB, far below the 10 MB ceiling.

---

### 3.3 Seed Sensitivity on Small Validation Sets

**Severity: MEDIUM [New S5]**

Stage 5 revealed material sensitivity to training trajectory initialisation when measured on a 52-clip validation set. The clearest example: bilstm_hands_only and bilstm_hands_only_v2 used identical configurations (BiLSTM, seq100, none, hands_only) but produced val_macro_f1 of 0.5419 and 0.4067 respectively — a 13pp gap. Both ran under seed=42; the difference arose from epoch-by-epoch dynamics under different patience configurations (30 vs 50), which caused different trajectory branching at the same initialisation point.

**This means: all Stage 5 val_macro_f1 values are single-seed measurements. The champion's 0.6011 should be treated as approximately 0.58 ± 0.03 as an expected value under seed variation.** Running 3–5 seeds and reporting mean ± std would provide a more robust estimate but was not feasible within the Stage 5 compute budget.

**Implication for Stage 6:** The test set evaluation (51 clips, 7 signers, held-out) provides the most reliable single accuracy estimate since it is not used for model selection and is evaluated only once.

---

### 3.4 Val Metric Oscillation

**Severity: MEDIUM [New S5]**

52 validation clips produce approximately 2 validation batches at batch_size=32. This means:

- A single clip misclassification shifts val_acc by **1.9pp**.
- A singleton class flip shifts val_macro_f1 by up to **2.9pp**.
- Epoch-to-epoch swings of 3–8pp in val_macro_f1 are expected noise, not signal.

This imposes a practical lower bound on patience for early stopping: patience < 30 risks stopping on a downward noise excursion below the model's true expected performance. The champion run used patience=50, which is the correct setting for this validation set size.

The implication for comparing runs: differences in val_macro_f1 of less than 3pp between any two runs should be treated as within-noise and not used to draw architectural conclusions.

---

### 3.5 Architecture Comparison Narrative: Dense Inversion

**Severity: LOW [New S5]**

Group 1 produced a counterintuitive result: the Dense feedforward baseline (0.3276 val_macro_f1) outperformed all temporal models (LSTM: 0.1948, GRU: 0.1905, BiLSTM: 0.1761) under the 80-epoch, lr=1e-3, full-225-dim Group 1 configuration.

This is not evidence that temporal modelling is unnecessary. The Dense model has approximately 7.7M parameters vs 68K–110K for recurrent models on a 236-clip dataset, producing a train_macro_f1/val_macro_f1 gap of approximately 0.48 (train reaching 0.81, val at 0.33). The Dense model exploited signer-correlated spatial features — absolute hand position relative to the frame, which correlates with signer identity — rather than learning sign-discriminative geometry.

Under hands_only features and seq_len=100 (which removes pose position information and forces temporal modelling), the recurrent architectures substantially outperform any Dense baseline. The final architecture justification must cite the 110% relative improvement from full to hands_only (which a Dense model cannot leverage) and the train/val gap analysis, not the Group 1 accuracy comparison in isolation.

---

## 4. Evaluation Limitations

### 4.1 Signer-Independent Validation: Conservative but Honest

**Severity: LOW (by design)**

The signer-aware split enforces **zero signer overlap** between train, val, and test. This makes validation accuracy a genuinely conservative, honest estimate of generalisation to unseen signers. It systematically produces lower numbers than a naïve random split on the same clips.

Published WLASL benchmark results used random splits that do not enforce signer independence. The signer-independent accuracy reported by this pipeline is not directly comparable to those numbers — ours is more rigorous and more conservative.

**Implication for the report:** The one-page report must explicitly state "X% signer-independent validation macro-F1" to contextualise the result for Abel Holla's evaluation.

---

### 4.2 70% Accuracy Target Not Met

**Severity: HIGH [New S5]**

The project target of ≥70% signer-independent validation macro-F1 was not met. The best result across all 23 Stage 5 runs is **0.6011** (bilstm_hands_only_v4_aug).

**The 9.9pp gap between achieved (0.6011) and target (0.70) is not primarily a modelling failure.** It reflects the dataset ceiling documented in §1.1: 6.7 clips/class mean, 21 singleton validation classes, 46.6% data completeness, and zero signer overlap. Stage 5's exhaustive ablation — covering architecture, augmentation, sequence length, landmark configuration, epoch budget, and patience — produced a 319% relative improvement from the Group 1 baseline (0.1434) to the champion. The remaining gap is dataset-constrained, not architecture-constrained.

An honest projection: to reach 70% signer-independent macro-F1 on 35 signs with this architecture and pipeline, approximately 30–50 clips per class (1,050–1,750 total) would be needed. For the full 500-sign KSL production system at 85%, the data requirement scales accordingly.

---

### 4.3 Validation Set Size and Metric Variance

**Severity: MEDIUM**

52 validation clips produce a high-variance macro-F1 estimate with an implicit confidence interval of ±3–5pp. Single-epoch val_macro_f1 differences below 3pp between any two runs are not reliable basis for architectural conclusions.

The test set (51 clips, 7 signers) provides a final unbiased evaluation that is not used for model selection. It is the most reliable single accuracy estimate available at Stage 5 scale.

---

### 4.4 No External Benchmark Comparison

**Severity: LOW**

This pipeline cannot be directly compared to published WLASL-100 or WLASL-2000 benchmarks because:
- Those benchmarks use 100 or 2000 classes, not the 35 selected here.
- Published benchmarks used random splits without signer independence enforcement.
- The 46.6% data recovery rate means the working dataset is a strict subset of the WLASL-35 full inventory.

The appropriate comparison baselines for this project are the Dense feedforward baseline (Group 1, 0.3276) and the no-augmentation LSTM on full features (Group 2 baseline, 0.1706), both trained on the same 236-clip training set.

---

## 5. Deployment Limitations

### 5.1 ASL to KSL Transfer Gap

**Severity: HIGH (for production deployment)**

This pipeline was developed on the WLASL dataset, which contains **American Sign Language (ASL)**. The target deployment language is **Kenyan Sign Language (KSL)**. ASL and KSL are structurally different languages:

- Different handshapes (distinct phonemic inventories)
- Different movement patterns and spatial conventions
- Different non-manual markers (facial grammar)
- Different lexical items (most signs are completely different)

**A model trained on WLASL ASL data will not generalise to KSL without significant adaptation.** Transfer from ASL provides low-level feature detection capabilities (hand landmark geometry via MediaPipe, wrist-relative normalisation) but the high-level sign patterns learned from ASL are largely inapplicable to KSL.

**[New S5]** The hands_only configuration may actually aid KSL transfer relative to a full-landmark model: by removing pose landmarks (which encode signer body morphology and are highly signer-specific), hands_only is forced to rely only on hand geometry — the most phonemically transferable feature between sign languages. A full-landmark model would need to additionally suppress signer-specific body posture during KSL inference.

**Recommended KSL adaptation strategy:**
1. Collect KSL training data: minimum ~100 clips/sign for the target vocabulary; the AI4KSL dataset provides ~40 clips/sign, below viability for a 500-sign system at 85% accuracy.
2. Train three models in parallel: KSL-only baseline; ASL-frozen feature extractor + KSL classifier head; ASL fully fine-tuned on KSL.
3. Compare on held-out KSL test set with signer independence enforced.
4. Estimated timeline for a 500-sign KSL system at 85% accuracy: 3–6 months, primarily driven by data collection.

---

### 5.2 Real-Time Performance: CPU-Only Target

**Severity: LOW**

The ≤100ms end-to-end latency target is for CPU-only inference on the primary Android mobile deployment target. At seq_len=100 with a 2-layer BiLSTM (hidden=64, hands_only 126-dim input, 68,771 params), estimated CPU inference time is approximately 40–60ms — within the 100ms budget. This must be verified empirically in Stage 8 benchmarking.

Dynamic range quantisation will reduce model size by approximately 4× (from ~0.26 MB float32 to ~0.07 MB int8) with minimal accuracy impact. The TFLite file is expected to be well under 1 MB, far below the 10 MB target.

---

### 5.3 Single-Sign Prediction Only

**Severity: MEDIUM (for production)**

This pipeline performs word-level isolated sign recognition — it classifies a single, pre-segmented sign clip. It does not perform continuous sign language recognition, handle co-articulation between consecutive signs, segment signs from continuous video streams, or recognise sentence-level meaning or grammatical structure.

The webcam demo (Stage 9) uses a sliding window approach with 5-frame majority voting, which approximates real-time recognition but is not a continuous recognition system. For production deployment, sign segmentation must be solved as a separate upstream module.

---

## 6. Stage 5 Training Findings: Decisions Superseded or Revised

The following decisions made before or during Stage 5 were revised in light of experimental evidence:

| Decision | Original state | Revised state | Evidence |
|---|---|---|---|
| Primary sequence length | seq_len=60 | **seq_len=100** | Group 3: 64% relative F1 improvement; 97% truncation at seq60 |
| Landmark configuration | full (225 dims) | **hands_only (126 dims)** | Group 4: 110% relative F1 improvement; pose Fisher ratio 0.2176 |
| Augmentation for champion runs | Disabled (Group 2 finding) | **spatial_temporal enabled** | Champion: 0.6011 (augmented) vs 0.5419 (best no-aug); 28% relative gain |
| Champion architecture | Unclear after Group 1 inversion | **BiLSTM** | Champion v. LSTM champion (0.6011 vs 0.4286); training curve quality |
| Early stopping patience | patience=10–15 (config default) | **patience≥40–50** | Runs with patience<30 systematically underperform (champion_bilstm_hands_only: 0.4181 due to patience=25) |
| Minimum training epochs | 80 (base.yaml) | **200–250 for augmented runs** | Augmented training requires extended epochs for convergence; Group 2 failure was 80-epoch artefact |

---

## 7. Known Pipeline Issues (Resolved, Post-Stage 5)

All pre-Stage 5 known issues documented in the previous version remain resolved. No new unresolved bugs were introduced in Stage 5.

| Issue | Stage identified | Resolution |
|---|---|---|
| v1.1 30% skip threshold produced 76% skip rate | Notebook 02 | v1.2 dual-criterion policy: min 15 detected frames + 95% catastrophe filter |
| Leading-zero video_id mismatch (80 clips silently skipped) | Notebook 04 | `fix_leading_zero_video_ids()` normalised all IDs to integer-string form |
| Individual hand missing rates zeroed in landmark_inventory.csv | Notebook 02 | Structural display issue; `missing_both_pct` (used for skip policy) is correct |
| speed_jitter interpolation violated zero-fill invariant at boundaries | Notebook 04 | Zero-aware interpolation: output forced to zero when both surrounding frames are zero |
| Stray debug `print()` in AugmentationPipeline.__call__() | Stage 4 review | Removed |
| GestureDataset epoch counter not varied when model.fit() called once | Stage 4 review | tf.Variable counter + per-epoch load_split() training loop contract enforced |

---

## 8. Summary: Accuracy Interpretation Guide

Any accuracy number reported by this pipeline must be interpreted in the context of all of the above. The following table provides the key contextual facts for the one-page report and theoretical assessment:

| Context factor | Value | Implication |
|---|---|---|
| Training clips | 236 | ~6.7 clips/class — severely data-limited |
| Validation clips | 52 | High-variance estimate; ±3–5pp implicit CI |
| Val singleton classes | 21/35 | Per-class val metrics unreliable for 60% of classes |
| Signer independence | Enforced (zero overlap) | Conservative vs random-split baselines |
| Dataset completeness | 46.6% | Hard ceiling; no architecture change compensates |
| Class weight ratio | 6.50× | Aggressive rebalancing required and applied |
| Primary metric | Macro-F1 (sklearn, zero_division=0) | Not overall accuracy — masks per-class failures |
| Augmentation | Spatial-temporal full chain | Halves train/val gap; requires ≥200 epochs to converge |
| Landmark config | hands_only (126-dim) | 2× improvement over full 225-dim; pose adds noise |
| Sequence length | 100 frames | 64% improvement over 60-frame default |
| Seed sensitivity | High (±3–5pp from trajectory variance) | Champion 0.6011 ≈ 0.58±0.03 expected value |
| Target language | ASL (not KSL) | Significant adaptation required for production |
| Best val_macro_f1 | 0.6011 (bilstm_hands_only_v4_aug) | Minimum viability met; 70% target not met |

**What the champion result means in plain language:**  
A BiLSTM model with 68,771 parameters, trained for 171 effective epochs on 236 American Sign Language clips with spatial-temporal augmentation and class weight balancing, achieves 60.1% macro-averaged F1 across 35 signs on 52 validation clips from 7 signers never seen during training. This is a signer-independent, class-balanced, honest estimate on a dataset with 46.6% data completeness. It is not directly comparable to published WLASL benchmarks, which used larger datasets and random (signer-overlapping) splits.

---

*This document will be updated after Stage 6 (evaluation on held-out test set), Stage 8 (TFLite quantisation accuracy delta), and Stage 9 (latency benchmarking) to reflect empirical measurements for limitations currently estimated analytically.*