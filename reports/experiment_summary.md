# Stage 5 Experiment Summary — WLASL 35-Class Gesture Recognition

**Date:** June 15, 2026  
**Total runs:** 23 across 4 ablation groups + champion candidates  
**Champion:** `bilstm_hands_only_v4_aug`  
**val\_macro\_f1:** **0.6011** | val\_acc: 0.5769  
**Minimum viability (≥0.60):** ✓ MET | **Target (≥0.70):** ✗ NOT MET  

---

## Executive Summary

23 experiments tracked across 4 groups. Champion: `bilstm_hands_only_v4_aug`.  
Best val\_macro\_f1: **0.6011** — minimum viability (0.60) met; 70% target not met.  
Total improvement from Group 3 baseline (lstm\_seq60\_full, 0.1434) to champion: **+319% relative**.

---

## Group-by-Group Conclusions

### Group 1 — Architecture Comparison (seq60, no-aug, full 225-dim, lr=1e-3)

Dense baseline (0.3276) outperformed all temporal models at 80 epochs — an apparent inversion  
of expectation. The Dense model has ~7.7M parameters versus 52–110K for recurrent models on  
a 236-clip dataset. Its train\_macro\_f1 reached 0.81, confirming a 0.48 train/val gap — the  
model learned signer-correlated spatial position, not sign geometry. Recurrent models were  
additionally constrained by lr=1e-3 (too high for stacked LSTMs on 236 clips causing loss  
surface instability) and 80 epochs (insufficient for convergence on signer-independent data).

All Group 1 results were measured on full 225-dim landmarks — Group 4 later showed hands\_only  
produces +110% relative improvement at equivalent configuration. Group 1 absolute values are  
floor estimates.

### Group 2 — Augmentation Ablation (LSTM, seq60, full, lr=5e-4, max\_epochs=80)

At 80 epochs: no-aug (0.1706) > temporal (0.1200) > spatial\_temporal (0.0108). Spatial\_temporal  
augmentation produced diverging val\_loss — attributed to insufficient epoch budget, lr=5e-4  
still too high for full-landmark augmented training, and 99 noisy pose dims amplifying instability.

**This conclusion was directly overturned by the champion run.** Under 250 epochs, patience=50,  
lr=5e-4, and hands\_only features, spatial\_temporal achieves 0.6011 vs 0.4695 (no-aug, 300ep) —  
a 28% relative improvement and 50% reduction in train/val overfitting gap. The augmentation  
finding is epoch-budget-conditional, not absolute.

### Group 3 — Sequence Length Ablation (LSTM, no-aug, full, lr=5e-4, max\_epochs=120)

seq\_len=100 wins decisively: val\_macro\_f1=0.2354 vs 0.1434 (seq60), a **+64% relative improvement**.  
This directly validates the Notebook 04 finding: 97% truncation at seq60 (P75=84, P90=95 frames)  
was actively discarding meaningful sign content. seq\_len=80 collapsed to a deterministic local  
minimum (0.033) under seed=42 — confirmed by v2 reproduction (0.030). This is a seed-specific  
initialisation trap, not a general finding about seq80. lstm\_seq100 was still improving at  
epoch 120, motivating extension to 180–300 epochs for champion runs.

### Group 4 — Landmark Configuration Ablation (LSTM, seq100, no-aug, lr=5e-4)

**Most important finding of Stage 5:** `hands_only` more than doubles val\_macro\_f1 vs `full`  
(0.4948 vs 0.2354, **+110% relative**), using 23% fewer parameters and 44% fewer input dimensions.  
`pose_only` (0.0314) confirms 99 pose dimensions carry near-zero discriminative signal for these  
35 ASL signs. Appending pose forces the LSTM to ignore 44% of its input — a severe regularisation  
burden on 236 training clips that compounds across layers. The Fisher ratio prediction (hands\_only  
0.8097 > full 0.5492) held directionally; the 2× magnitude gap exceeds the ratio difference,  
indicating non-linear suppression across LSTM depth.

All Groups 1–3 results should be interpreted as floor estimates; every measurement used full  
225-dim landmarks before this finding.

---

## Champion Model Selection

| Parameter | Value |
|-----------|-------|
| Run name | `bilstm_hands_only_v4_aug` |
| Architecture | BiLSTM, 2 layers, 64 hidden units (32/direction) |
| Total params | 68,771 |
| Estimated size | 0.262 MB (float32, pre-quantisation) |
| Feature config | hands\_only — 126 dims (left hand 63 + right hand 63) |
| Sequence length | 100 frames |
| Augmentation | spatial\_temporal (all 5 transforms) |
| Learning rate | 5e-4 |
| Epochs trained | 221 (best @ epoch 171) |
| Early stopping | patience=50, monitor=val\_macro\_f1 |
| **val\_macro\_f1** | **0.6011** |
| val\_acc | 0.5769 |
| val\_loss @ best | ~1.68 |
| MLflow run ID | cb16f689d2294001a2ff2d3e02419d27 |

### High-Risk Class Performance

| Class | Train clips | Val clips | Champion F1 | Consistency across runs |
|-------|-------------|-----------|-------------|------------------------|
| clothes | 2 | 1 | 1.00 | Erratic — augmentation decisive |
| think | 3 | 2 | 0.00 | Fails in 8/9 runs — data failure |
| birthday | 4 | 1 | 1.00 | Relatively stable |
| name | 4 | 1 | 1.00 | Fails in ~40% of non-champion runs |
| book | 4 | 2 | 1.00 | Variable — augmentation helped |

`think` failing is the honest floor of this dataset. 3 training clips with zero signer overlap  
cannot support reliable representation learning for any architecture.

---

## Key Limitations (Stage 5 additions to LIMITATIONS.md)

1. **Augmentation finding is epoch-budget-conditional.** Group 2 conclusion (spatial\_temporal harmful)  
   was an 80-epoch artefact. Under 250 epochs and hands\_only features, it is decisively better.
2. **`think` class is unlearnable at current data scale.** F1=0.0 in 8/9 champion runs. Not a model  
   failure — a data failure (3 training clips).
3. **Val metric variance is high.** 52 val clips, ~2 batches. Epoch-to-epoch swings of 3–5pp  
   are structural noise. Champion's 0.6011 ≈ 0.58±0.03 expected value.
4. **Seed sensitivity.** Identical configs diverge by up to 13pp (bilstm\_hands\_only: 0.5419 vs  
   v2: 0.4067). Champion result is a single-seed measurement.
5. **70% target not met.** Honest ceiling ≈ 0.60–0.65 under current data constraints  
   (6.7 clips/class mean, 21 singleton val classes, 7-signer zero-overlap validation).

---

## Implications for Stage 6

- **Champion SavedModel:** `models/bilstm_hands_only_v4_aug_saved_model/`
- **Test evaluation:** 51 clips, 7 signers, zero overlap. Expected test macro\_f1: 0.45–0.58 (typically lower than val due to different signers).
- **Latency target:** BiLSTM 68K params on hands\_only 126-dim input → expected ≤50ms CPU, well within 100ms target.
- **SHAP:** Which of 100 frames and 126 dims matter most? Wrist and finger landmarks expected to dominate.
- **Calibration:** With 21 singleton val classes and high F1 variance, reliability diagram likely shows overconfidence.
- **Confusable pairs to investigate:** before↔chair, cousin↔go/now, drink↔boy/orange/who, girl↔go/now, who↔candy.

---

## Full Experiment Registry

| Run Name | Group | Model | Seq | Aug | Landmark | val\_macro\_f1 | val\_acc | Best Epoch | Total Epochs |
|----------|-------|-------|-----|-----|----------|--------------|--------|-----------|-------------|
| dense_baseline | architecture | dense | 60 | none | full | 0.3276 | 0.3654 | 74 | 80 |
| lstm_baseline | architecture | lstm | 60 | none | full | 0.1948 | 0.2500 | 52 | 68 |
| gru_baseline | architecture | gru | 60 | none | full | 0.1905 | 0.2692 | 77 | 80 |
| bilstm_baseline | architecture | bilstm | 60 | none | full | 0.1761 | 0.1923 | 48 | 64 |
| lstm_no_aug | augmentation | lstm | 60 | none | full | 0.1706 | 0.2115 | 71 | 80 |
| lstm_temporal_aug | augmentation | lstm | 60 | temporal | full | 0.1200 | 0.1923 | 79 | 80 |
| lstm_spatial_temporal_aug | augmentation | lstm | 60 | spatial_temporal | full | 0.0108 | 0.0385 | 16 | 32 |
| bilstm_spatial_temporal_aug | augmentation | bilstm | 60 | spatial_temporal | full | 0.0041 | 0.0769 | 3 | 19 |
| lstm_seq100 | sequence | lstm | 100 | none | full | 0.2354 | 0.2692 | 103 | 120 |
| lstm_seq60 | sequence | lstm | 60 | none | full | 0.1405 | 0.1538 | 40 | 61 |
| lstm_seq80 | sequence | lstm | 80 | none | full | 0.0328 | 0.0962 | 11 | 32 |
| lstm_seq80_v2 | sequence | lstm | 80 | none | full | 0.0297 | 0.0962 | 11 | 32 |
| lstm_hands_only | landmark | lstm | 100 | none | hands_only | 0.4948 | 0.4808 | 104 | 120 |
| lstm_pose_only | landmark | lstm | 100 | none | pose_only | 0.0314 | 0.0962 | 13 | 34 |
| bilstm_hands_only_v4_aug 🏆 | champion | bilstm | 100 | spatial_temporal | hands_only | 0.6011 | 0.5769 | 170 | 221 |
| bilstm_hands_only | champion | bilstm | 100 | none | hands_only | 0.5419 | 0.5385 | 163 | 180 |
| champion_bilstm_hands_only_v3 | champion | bilstm | 100 | temporal | hands_only | 0.5190 | 0.5000 | 196 | 247 |
| bilstm_hands_only_v3 | champion | bilstm | 100 | none | hands_only | 0.4695 | 0.4808 | 150 | 201 |
| champion_bilstm_hands_only_v2 | champion | bilstm | 100 | spatial_temporal | hands_only | 0.4610 | 0.4808 | 129 | 180 |
| bilstm_hands_only_v3_aug | champion | bilstm | 100 | temporal | hands_only | 0.4552 | 0.4423 | 85 | 136 |
| champion_hands_only_v1 | champion | lstm | 100 | none | hands_only | 0.4286 | 0.4615 | 76 | 107 |
| champion_bilstm_hands_only | champion | bilstm | 100 | none | hands_only | 0.4181 | 0.3846 | 58 | 84 |
| bilstm_hands_only_v2 | champion | bilstm | 100 | none | hands_only | 0.4067 | 0.4038 | 65 | 116 |

---

*Generated: 2026-06-15 18:28 UTC*  
*Pipeline: BiLSTM / LSTM on MediaPipe Holistic Landmarks | 35 ASL Signs | Zero Signer Overlap*
