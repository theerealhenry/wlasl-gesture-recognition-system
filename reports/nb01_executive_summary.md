# Notebook 01 — Executive Summary: WLASL-35 Data Exploration & Inventory Analysis

**Project:** WLASL Gesture Recognition Pipeline — Stage 1: Data Ingestion & Validation  
**Author:** Henry Otsyula — Senior Data Scientist & ML Engineer  
**Executed:** 2026-06-01  
**Status:** ✓ Complete — all figures and reports produced

---

## 1. Purpose and Scope

This notebook is a post-pipeline reporting artifact. All data transformation — manifest resolution, video inventory, integrity validation, and signer-aware splitting — was performed upstream by `pipelines/run_preprocessing.py`. This notebook loads the produced artifacts, interrogates them analytically, and generates the figures and written conclusions that feed directly into the Stage 5 training configuration, the one-page report, and `LIMITATIONS.md`.

The analysis covers five interconnected concerns: **data completeness**, **class distribution**, **signer diversity and dominance**, **temporal characteristics**, and **split quality and integrity**.

---

## 2. Dataset Overview

| Metric | Value |
|---|---|
| Sign classes | 35 |
| Total clips in manifest (inventory) | 751 |
| Clips found on disk | 350 |
| Clips missing from disk | 401 |
| **Dataset completeness** | **46.6%** |
| Unique signers (found clips) | 45 |
| Mean clips per sign | 10.0 |
| Mean clip duration | 2.29 seconds |
| Mean frame count per clip | 65.7 frames |
| Total recorded content | 0.223 hours (13.4 minutes) |
| Total dataset size | 178.5 MB |
| Train / Val / Test split | 245 / 53 / 52 clips |

The most consequential finding at the dataset level is that **53.4% of inventory clips (401 of 751) could not be located on disk**. This is a known, documented limitation of the WLASL dataset: a large proportion of the original YouTube source URLs are no longer accessible. The effective working dataset is therefore 350 clips across 35 signs — well below the original inventory, but sufficient to demonstrate the pipeline at the required standard.

---

## 3. Class Distribution Analysis

*(Figure: `class_distribution.png`)*

**Finding:** Every sign in the dataset falls below the 20-clip minimum threshold specified in the project brief. With only 350 clips recovered and 35 classes to cover, the mean is exactly 10 clips per sign. The distribution is moderately skewed: the best-represented sign (`before`, 16 clips) has 3.2× as many clips as the least-represented (`clothes`, 5 clips).

| Statistic | Value |
|---|---|
| Min clips per sign | 5 (`clothes`) |
| Max clips per sign | 16 (`before`) |
| Mean / Median | 10.0 / 10.0 |
| Std deviation | 3.0 clips |
| Imbalance ratio (max/min) | 3.20× |
| Gini coefficient | 0.171 |
| Signs below 20-clip threshold | **35 of 35** (100%) |
| Signs with < 4 unique signers | 0 |

The **Gini coefficient of 0.171** places this dataset in the "low imbalance" category — the class distribution is fairly uniform in relative terms despite the small absolute counts. The Lorenz curve (bias documentation figure, panel C) visually confirms this: the orange curve hugs the diagonal closely, indicating that no sign massively dominates the dataset.

All 35 signs have at least 4 unique signers, satisfying the minimum selection criterion from the label map specification. The bottom tier — `clothes` (5 clips, 5 signers), `birthday` (6 clips, 5 signers), `book` (6 clips, 6 signers), `name` (6 clips, 6 signers) — will exhibit the highest per-class variance in Stage 5 metrics and should be monitored individually.

**Design decision:** `class_weight_balancing=True` should be activated in `TrainingConfig` since the imbalance ratio exceeds 2.5×. Overall accuracy in Stage 5 will mask poor performance on rare signs; **per-class F1 is the primary evaluation metric** for this dataset.

---

## 4. Signer Analysis

*(Figure: `signer_distribution.png`)*

**Finding:** 45 unique signers are represented in the 350 found clips. The signer-volume distribution is highly skewed: one signer (ID shown at far left of panel A) contributes approximately 60 clips — roughly 17% of the entire found dataset — while the median signer contributes only 1–2 clips. This is a structural property of WLASL, not an artifact of the resolver.

| Metric | Value |
|---|---|
| Unique signers in found clips | 45 |
| Max clips contributed (single signer) | 60 |
| Mean distinct signs per signer | 6.1 |
| Median distinct signs per signer | 4 |
| Max distinct signs per signer | 32 |

**Signer dominance analysis** (top 10 by dominance percentage):

| Sign | Top signer | Contribution | Dominance |
|---|---|---|---|
| `go` | Signer 10 | 7 / 15 | 46.7% |
| `computer` | Signer 4 | 5 / 14 | 35.7% |
| `birthday` | Signer 11 | 2 / 6 | 33.3% |
| `now` | Signer 11 | 3 / 9 | 33.3% |
| `black` | Signer 11 | 3 / 10 | 30.0% |
| `many` | Signer 11 | 3 / 10 | 30.0% |

The signer dominance CDF (bias documentation figure, panel B) shows that approximately 55% of signs have a top-signer dominance below 30% — the research target. However, the remaining 45% exceed this threshold, with `go` approaching 50%. **No sign reaches 50%+** dominance (F2 severity: LOW), meaning no single signer entirely defines any one class.

**Design decision:** Spatial augmentation (mirror flip, rotation ±5°, Gaussian noise) is particularly important for signs in the high-dominance tier (`go`, `computer`, `birthday`, `now`). The `signer_analysis.py` module in Stage 6 should produce per-signer accuracy breakdowns for these classes specifically, as they are most at risk of memorising an individual's motion style.

---

## 5. Video Duration and Frame Count Analysis

*(Figure: `sequence_length_distribution.png`)*

**Finding:** The dataset consists of short, focused gesture clips with a mean duration of 2.29 seconds and mean frame count of 65.7 frames. The distribution is right-skewed: most clips cluster below 100 frames, with a long tail extending to ~300 frames (the configured maximum). Clip file sizes are tightly concentrated below 1 MB (median 0.34 MB), consistent with short, compressed video clips.

| Metric | Value |
|---|---|
| Mean frame count | 65.7 frames |
| Mean duration | 2.29 seconds |
| Min frame count (validated) | within [10, 300] range |
| Max frame count (validated) | within [10, 300] range |
| All clips passed frame range check | ✓ |
| Median file size | 0.34 MB |

**Sequence length coverage analysis** (panel D of sequence length figure): at the default sequence length of 30 frames, approximately 30% of a clip's mean content is captured — the model sees roughly the first third of a typical sign. This is by design: with a mean of 65.7 frames, a 30-frame window captures the initial gesture phase, which research suggests is most discriminative for ASL signs. The ablation study in Stage 5 Group 3 over `seq_len ∈ {20, 30, 40}` will quantify the accuracy/latency trade-off empirically. Moving to 60 frames would capture ~90% of mean clip content but doubles inference latency and memory footprint.

**Design decision confirmed:** `sequence_length = 30` remains the default configuration for all Stage 5 baseline runs. The ablation study is essential given that 30 frames captures only ~30% of mean clip content — the stage-5 ablation over {20, 30, 40} should be extended to {30, 40, 60} if compute permits.

---

## 6. Validation Report

*(Figure: `validation_summary_table.png`)*

**Result: 6/8 checks passed | 2 warnings | PIPELINE CAN PROCEED**

| Check | Result | Notes |
|---|---|---|
| All Signs Present | ✓ PASS | All 35 signs present with ≥1 clip; class count matches expected 35 |
| Minimum Clips Per Sign | ⚠ WARNING | All 35 signs below the 20-clip threshold — expected given missing clips |
| Video Readability | ✓ PASS | All 350 found clips readable by OpenCV (13.29s check time) |
| Frame Count Range | ✓ PASS | All clips within [10, 300] frame bounds |
| No Duplicate Video IDs | ✓ PASS | All 751 inventory IDs unique |
| Signer IDs Complete | ✓ PASS | All 350 found clips have valid signer IDs; 45 unique signers |
| Dataset Size | ✓ PASS | 178.5 MB (0.17 GB) — within expected range |
| Class Imbalance | ⚠ WARNING | Ratio 3.20× exceeds 3.0× threshold (marginal) |

Both warnings are expected and benign given the known data completeness limitation:

- **Minimum clips warning** is a direct consequence of 53.4% of clips being inaccessible. This is not resolvable without additional data collection or the yt-dlp download fallback for any newly accessible URLs.
- **Class imbalance warning** is marginal (3.20× vs 3.0× threshold) and quantitatively mild (Gini = 0.171). It is addressed at training time via class weight balancing rather than at the data level.

The pipeline correctly proceeded to the split stage.

---

## 7. Split Quality Analysis

*(Figures: `class_balance_per_split.png`, signer distribution panel A)*

The greedy bin-packing signer-aware split algorithm (seed=42) produced the following allocation:

| Split | Clips | Share | Target | Drift | Signers | Classes |
|---|---|---|---|---|---|---|
| Train | 245 | 70.0% | 70% | 0.0 pp | 31 | 35 |
| Val | 53 | 15.1% | 15% | +0.1 pp | 7 | 35 |
| Test | 52 | 14.9% | 15% | −0.1 pp | 7 | 35 |

**Signer overlap (critical invariant):**

| Pair | Overlap |
|---|---|
| Train ∩ Val | **0 ✓** |
| Train ∩ Test | **0 ✓** |
| Val ∩ Test | **0 ✓** |

The split achieves the fundamental guarantee: **no signer appears in more than one partition**. All 35 classes are represented in all three splits. The drift from targets is negligible (≤0.1 percentage points), confirming the bin-packing algorithm performed correctly.

**Thin-class warnings** are expected at this dataset size. With a mean of only ~1.5 val clips per sign across 35 classes, the majority (31 of 35) have fewer than 3 clips in validation. This means per-class val metrics in Stage 5 will carry wide confidence intervals — particularly for the bottom quintile of classes. This is a fundamental data limitation, not a splitting artifact. Aggregate val accuracy remains a meaningful signal; per-class F1 should be interpreted with appropriate uncertainty.

The stacked clip-share heatmap (class balance figure, panel B) confirms that train/val/test proportions are broadly consistent across signs — no sign has an unusual split imbalance that would systematically advantage or disadvantage it in evaluation.

---

## 8. Dataset Bias Documentation

*(Figure: `dataset_bias_overview.png`)*

Three complementary bias analyses were conducted:

**Panel A — Within-Sign Recording Variance (proxy for environment diversity)**  
File size standard deviation is used as a proxy for recording diversity: high variance implies multiple recording environments, lighting conditions, and backgrounds; low variance implies a more homogeneous set of recordings. Signs such as `finish`, `later`, `thanksgiving`, `change`, and `birthday` show the highest variance, suggesting genuine environmental diversity. Signs like `drink` and `name` cluster at the low end, indicating they may have been recorded under more uniform conditions. This translates directly to expected within-class generalisation difficulty.

**Panel B — Signer Dominance CDF**  
Approximately 55% of signs have a top-signer clip share below the 30% dominance threshold (orange dashed line). The steepest portion of the CDF falls between 20–35% dominance, with the most extreme case (`go`, ~47%) still below the 50% critical threshold. This is a relatively healthy distribution for a dataset of this scale.

**Panel C — Class Imbalance Lorenz Curve**  
The Lorenz curve (Gini = 0.171) closely tracks the diagonal (perfect equality), confirming that class imbalance is mild despite the max/min ratio of 3.20×. The ratio is driven primarily by `before` (16 clips) vs `clothes` (5 clips) — a gap that is numerically significant but not structurally pathological.

**Overall bias assessment:** The dataset is broadly sound for a research pipeline at this scale. The dominant risks are (1) the very small per-class clip counts in the post-missing-clips working set, and (2) moderate signer dominance on a handful of classes. Both are mitigated by the signer-aware split and spatial augmentation rather than requiring dataset-level intervention.

---

## 9. Key Findings and Design Decisions

The following five findings are codified in `reports/nb01_findings.json` and directly govern downstream stage configurations.

### F1 — Class Imbalance | Severity: HIGH

**Finding:** Imbalance ratio is 3.20× (Gini = 0.171). All 35 signs fall below the 20-clip threshold.

**Decision:** Activate `class_weight_balancing=True` in `TrainingConfig` for all Stage 5 runs. Monitor per-class F1 — overall accuracy will systematically mask poor performance on the five lowest-frequency signs (`clothes`, `birthday`, `book`, `name`, `think`). These classes should be flagged in the Stage 6 per-class metrics table regardless of aggregate performance.

---

### F2 — Signer Dominance | Severity: LOW

**Finding:** Zero signs have a single signer contributing >50% of clips.

**Decision:** No critical signer dominance detected. Spatial augmentation (mirror flip, ±5° rotation, Gaussian noise σ=0.01) remains recommended for the high-dominance tier (`go` ~47%, `computer` ~36%) to reduce the risk of style memorisation. Stage 6 `signer_analysis.py` should produce per-signer accuracy breakdowns for these classes.

---

### F3 — Split Integrity | Severity: LOW

**Finding:** Signer-aware split with zero overlap across all pairs. Train 245 | Val 53 | Test 52 clips. Actual ratios: 70.0% / 15.1% / 14.9% vs targets 70% / 15% / 15%. Drift ≤ 0.1 pp.

**Decision:** Split is methodologically sound. Validation accuracy in Stage 5 is the honest estimate of generalisation to **entirely unseen signers** — a condition that a naïve random split would not satisfy. This should be reported prominently in the final submission: it is the primary methodological differentiator of this pipeline relative to baseline WLASL benchmarks.

---

### F4 — Sequence Length | Severity: LOW

**Finding:** Mean clip frame count is 65.7. The default sequence length of 30 frames covers approximately 30% of mean clip content.

**Decision:** `seq_len = 30` is confirmed as the primary baseline configuration. The Stage 5 Group 3 ablation over `seq_len ∈ {20, 30, 40}` will empirically quantify the accuracy/latency trade-off. Given that 30 frames captures only one-third of a typical clip, extending the ablation to include `seq_len = 60` is recommended if compute permits — this would cover ~90% of mean clip content.

---

### F5 — Data Completeness | Severity: HIGH

**Finding:** 401 clips (53.4% of inventory) are missing from disk. All 35 signs fall below the 20-clip minimum. The 350 recovered clips represent the complete working dataset.

**Decision:** Missing clips are excluded from all splits. Document prominently in `LIMITATIONS.md`. The download fallback (`run_preprocessing.py --download-missing`) should be re-attempted if any additional source URLs become accessible. The 46.6% completeness rate is a fundamental data limitation that caps achievable accuracy and generalisation regardless of model architecture. Achieving ≥70% validation accuracy with this dataset scale will require aggressive augmentation and careful regularisation.

---

## 10. Artifacts Produced

All outputs were confirmed present at notebook completion:

| Artifact | Size | Purpose |
|---|---|---|
| `reports/figures/class_distribution.png` | 149.8 KB | Clips per sign × split, signer diversity |
| `reports/figures/signer_distribution.png` | 193.1 KB | Signer volume, breadth, dominance |
| `reports/figures/sequence_length_distribution.png` | 262.8 KB | Frame count, duration, size, seq coverage |
| `reports/figures/validation_summary_table.png` | 161.7 KB | 8-check validation suite result table |
| `reports/figures/class_balance_per_split.png` | 318.0 KB | Per-class clip distribution × split |
| `reports/figures/dataset_bias_overview.png` | 281.6 KB | Recording variance, signer CDF, Lorenz curve |
| `reports/dataset_summary.json` | — | Machine-readable dataset statistics |
| `reports/nb01_findings.json` | — | Structured findings registry (F1–F5) |

---

## 11. Implications for Downstream Stages

| Stage | Implication |
|---|---|
| **Stage 3 — Preprocessing** | MediaPipe missing-landmark rate will be higher than typical WLASL benchmarks due to the concentration of short, dynamic clips. Target: zero-fill frames with <30% missing; skip clips with >30% missing. Expect ~5–8% clip loss. |
| **Stage 4 — Feature Engineering** | Wrist-relative normalisation is validated by the within-sign recording variance: multiple environments introduce positional noise that normalisation removes. Spatial augmentation is particularly important for the high-dominance tier. |
| **Stage 5 — Training** | `class_weight_balancing=True`. Primary metric: validation accuracy (aggregate, honest signer-independent). Secondary: per-class F1 for the bottom quintile. With 245 training clips across 35 classes (~7 clips/class mean), strong regularisation (dropout 0.3, early stopping, augmentation) is essential to prevent overfitting. Achieving ≥70% val accuracy is ambitious but plausible with a BiLSTM + spatial-temporal augmentation configuration. |
| **Stage 6 — Evaluation** | Report per-class F1 prominently. Signer generalisation analysis is the most scientifically important evaluation — Stage 6 `signer_analysis.py` should compare performance on the 7 val signers vs known performance on the 31 train signers. The gap between these two numbers is the honest measure of generalisation. |
| **Report** | Lead with dataset completeness (46.6%) as the primary context for all accuracy numbers. Any reported accuracy is relative to this constrained working set. The signer-aware split methodology must be explicitly contrasted with random splitting to justify why the reported numbers are conservative and honest. |

---

*This summary was generated from the executed outputs of `notebooks/01_data_exploration.ipynb`. All statistics are drawn directly from the pipeline-produced artifacts: `data/raw_inventory.json`, `data/data_validation_report.json`, `data/splits/{train,val,test}.csv`, and `data/splits/split_summary.json`.*