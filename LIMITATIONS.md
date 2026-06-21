# LIMITATIONS.md
## WLASL Gesture Recognition System — Known Limitations, Constraints, and Mitigations

**Project:** WLASL-35-Class Gesture Recognition System
**Author:** Henry Otsyula — Senior Data Scientist & ML Engineer
**Last updated:** Stage 9 complete (post-webcam-demo verification)
**Champion model:** `bilstm_hands_only_v4_aug` — Keras val macro-F1 = 0.6011, TFLite val macro-F1 = 0.5916
**Status:** Active — authoritative as of the end of Stage 9; will receive a final pass after Stage 11 (report + theoretical assessment)

---

## How to read this document

This is a first-class project artefact, not an appendix. It is required reading before any
deployment decision, and every accuracy figure quoted anywhere else in this project — the
README, the one-page report, the model card, the theoretical assessment — must be understood
relative to the constraints documented here.

Each limitation has a stable ID (`L1`, `L2`, ...) for cross-referencing from other documents
(MODEL_CARD.md, the one-page report, code comments). IDs are assigned in rough order of
"how much this should worry a deployer," not chronological discovery order. Severity reflects
impact on a production KSL deployment decision, not impact on the WLASL-35 technical
verification result in isolation — a limitation can be `LOW` severity for "does this number
look credible" purposes while being `CRITICAL` for "should this ship to real signers" purposes,
and both readings are noted where they diverge.

**One disambiguation that matters throughout this document:** the project produced two
artefacts of the champion model — the original **Keras SavedModel** (Stage 5) and the
deployment **TFLite file** `gesture_bilstm_v1.tflite` (Stage 8, the one Stage 9's webcam
demo and any future Android wrapper actually load). Their accuracy differs by a small,
quantisation-driven amount (see L1 and L10). Unless explicitly marked "Keras," every figure
in this document refers to the TFLite deployment artefact, since that is what would actually
ship.

---

## Quick-reference summary

| ID | Limitation | Severity | Status |
|----|-----------|----------|--------|
| [L1](#l1) | Dataset completeness: 46.6% recovery rate — hard ceiling on accuracy | CRITICAL | Unfixable for WLASL |
| [L2](#l2) | All 35 classes below 20-clip minimum; 2 classes (`think`, `clothes`) near-unlearnable | HIGH | Partially mitigated |
| [L3](#l3) | 21/35 validation classes are singletons — per-class metrics statistically meaningless | HIGH | Mitigated via macro-F1 + bootstrap CIs |
| [L4](#l4) | Signer dominance in training data (4 classes ≥50% single-signer) | MEDIUM | Partially mitigated via augmentation |
| [L5](#l5) | High validation-set metric variance (52 clips, ±3–5pp noise floor) | MEDIUM | Mitigated via macro-F1, bootstrap CIs, held-out test |
| [L6](#l6) | Val→test generalisation gap (14.3pp, Keras) — indirect val-set overfitting | HIGH | Partially mitigated; inherent to small-data model selection |
| [L7](#l7) | Seed/trajectory sensitivity — champion ≈ 0.58 ± 0.03, not a fixed point estimate | MEDIUM | Documented, not fixed (no multi-seed budget) |
| [L8](#l8) | 70% accuracy target not met (0.60 achieved) — data-constrained, not architecture-constrained | HIGH | Unfixable without more data |
| [L9](#l9) | Landmark-based representation: irrecoverable MediaPipe detection loss, no facial grammar, weak depth | MEDIUM | Deliberate design trade-off |
| [L10](#l10) | TFLite quantisation: Δ vs Keras, SELECT_TF_OPS overhead, FULL_INTEGER infeasible | LOW–MEDIUM | Verified safe; documented trade-offs |
| [L11](#l11) | Confidence calibration: model is underconfident (ECE = 0.2009) | MEDIUM | Mitigated via calibrated 0.35 threshold |
| [L12](#l12) | Four near-degenerate confusable sign pairs (cosine similarity 0.79–0.96) | MEDIUM | Surfaced via top-3 UI, not resolved |
| [L13](#l13) | Right-hand-dominant attribution — possible signer-handedness bias | LOW–MEDIUM | Documented, not corrected |
| [L14](#l14) | MediaPipe dependency: real-time detection degrades in adverse conditions | MEDIUM | Partially mitigated (Hands > Holistic, auto-reset) |
| [L15](#l15) | No out-of-distribution / uncertainty detection in real-time inference | MEDIUM | Not implemented |
| [L16](#l16) | Real-time system is not thread-safe; single isolated-sign recognition only | LOW–MEDIUM | By design; documented |
| [L17](#l17) | Latency only measured on development-machine CPU, never on an Android device | MEDIUM | Open — Android benchmarking not yet performed |
| [L18](#l18) | ASL ≠ KSL: model cannot be directly deployed for the target language | CRITICAL (for project goal) | Adaptation protocol defined, not executed |

---

## 1. Data Limitations

### <a name="l1"></a>L1 — Dataset Completeness: 46.6% Recovery Rate

**Severity: CRITICAL. No architectural fix possible.**

The WLASL v0.3 dataset lists 751 clip entries across the 35 selected signs. Only **350 clips
(46.6%)** were recoverable from disk; the remaining **401 clips (53.4%) are permanently
inaccessible** because the original YouTube source URLs are dead and the videos have been
removed from the platform. This is a documented, structural property of the WLASL dataset
itself (noted in the original WLASL paper), not a pipeline failure — no action on this
project's part can recover these clips without re-recording from signers.

The 53.4% loss is **not uniform across classes**: some signs lost proportionally more clips
than others, which is the direct cause of the 6.50× class-weight imbalance ratio measured in
Stage 4 (compare to the much milder 3.20× imbalance ratio measured on the full WLASL
inventory in Stage 1, before data loss).

**Consequences:**
- The working training set is 236 clips across 35 classes — a mean of **6.7 clips per class**,
  far below the 20-clip minimum threshold flagged by the Stage 1 validator.
- This is a hard ceiling on achievable accuracy. Stage 5's exhaustive 23-run ablation —
  spanning architecture, augmentation, sequence length, landmark configuration, epoch budget,
  and patience — produced a +319% relative improvement from the Group 1 baseline (0.1434) to
  the champion (0.6011), demonstrating the pipeline was actively data-constrained rather than
  architecture-constrained: every dimension of model design was explored and the ceiling held.
- **We deliberately do not publish a specific projected accuracy under hypothetical full data
  completeness.** No data-scaling ablation (e.g. training on successive fractions of the
  available 236 clips to fit a learning curve) was run in this project, so any such number
  would be an unsupported guess dressed up as an estimate. What the evidence *does* support is
  the qualitative claim above — more data is the binding constraint — not a specific target
  value.

**Mitigation applied:** signer-aware splits use the working set efficiently; spatial-temporal
augmentation generates additional training variation per clip; class-weight balancing gives
rare classes proportionally higher gradient contribution. The `yt-dlp` download fallback
(`run_preprocessing.py --download-missing`) was implemented but not exercised, since dead
URLs are dead regardless of downloader.

---

### <a name="l2"></a>L2 — Severely Limited Per-Class Sample Counts

**Severity: HIGH. Affects two specific classes critically.**

| Sign | Train clips | Val clips | Test clips | Stage 5 best F1 (val) | Failure pattern |
|---|---|---|---|---|---|
| `think` | 3 | 2 | 2 | 0.0 in 8/9 champion runs | **Data-floor failure** |
| `clothes` | 2 | 1 | 2 | 1.0 (augmented) / 0.0 (no-aug) | **Augmentation-dependent** |
| `birthday` | 4 | 1 | 1 | 1.0 (most runs) | Relatively stable |
| `book` | 4 | 2 | 1 | 0.5–1.0 (variable) | Augmentation-sensitive |
| `name` | 4 | 1 | 1 | 1.0 (augmented) / 0.0 (no-aug) | Augmentation-dependent |

Two distinct, empirically-confirmed failure modes:

- **Pattern A — data-floor failure (`think`):** F1 = 0.0 in 8 of 9 champion candidate runs,
  across every architecture and augmentation configuration tested. 3 training clips is
  insufficient to learn a generalisable temporal representation of this sign under
  signer-independent evaluation. This is a data failure, not a model failure, and is not
  expected to improve with further hyperparameter search at this data scale.
- **Pattern B — augmentation-sensitive failure (`clothes`, `name`):** F1 = 0.0 in every
  no-augmentation run but F1 = 1.0 in the augmented champion run. The 2–4 training clips do
  contain learnable signal, but the model requires augmentation-induced regularisation to find
  it without simply memorising signer identity.

The resulting 6.50× class-weight ratio (Stage 4, confirmed unchanged through Stage 5) is the
largest the pipeline will encounter; `clothes` would receive negligible gradient without it.

**Mitigation applied:** `class_weight_balancing=True` enforced via config validation and
confirmed enabled in all 23 Stage 5 runs (Critical Rule #3 — no exceptions). Macro-F1, not
accuracy, is the primary metric specifically to surface these failures rather than averaging
them away. `think` is formally documented here as unlearnable at current data scale; any
future KSL data-collection effort should prioritise avoiding equivalently low per-class counts
for grammatically important signs.

---

### <a name="l3"></a>L3 — Validation Set Singleton Classes

**Severity: HIGH. Drives the macro-F1 noise floor for the whole project.**

**21 of the 35 sign classes have exactly one validation clip:** birthday, black, blue, book,
can, chair, change, clothes, color, computer, eat, family, finish, friend, give, house, many,
mother, name, now, thanksgiving. A single validation clip cannot produce a statistically
meaningful per-class accuracy estimate — its F1 is necessarily binary (0% or 100%).

This was empirically confirmed in Stage 5: two runs with identical configuration
(`bilstm_hands_only` and `bilstm_hands_only_v2`, both BiLSTM + seq100 + hands_only + no
augmentation, same seed=42, differing only in patience) produced val macro-F1 of 0.5419 and
0.4067 respectively — a 13pp gap from epoch-trajectory variance alone, not a genuine
performance difference. Every val macro-F1 figure in this project should be read with an
implicit confidence interval of approximately **±3–5pp**.

**Mitigation applied:** macro-F1 (not overall accuracy) as the primary metric; Stage 6's
per-class metrics table explicitly flags every singleton class so readers cannot mistake a
binary outcome for a reliable estimate; the held-out test set (51 clips, never used for model
selection) provides a secondary, less-biased evaluation; Stage 5 model selection used manual
early stopping on val macro-F1 with patience ≥ 40 specifically to avoid locking onto a noise
peak.

---

### <a name="l4"></a>L4 — Signer Dominance in Training Split

**Severity: MEDIUM. A measurable contributor to the train/val gap.**

Signer 11 is the dominant contributor in 10 of 35 training signs, with four signs reaching
≥50% single-signer contribution: `go` (Signer 10, 58.3%), `clothes` (Signer 11, 50% — only 2
total training clips), `black` (Signer 11, 50%), `birthday` (Signer 11, 50%).

The Stage 5 train/val macro-F1 gap (approximately 0.48 in no-augmentation runs, reduced to
approximately 0.24 in the best augmented run) is partly attributable to signer-identity
memorisation rather than genuine sign-geometry learning — the gap was largest precisely where
augmentation (which specifically targets signer-specific spatial patterns) was absent. This is
empirical, not inferred: train macro-F1 reached 0.95 by epoch 200 in no-augmentation runs while
val macro-F1 plateaued at 0.40–0.47, a classic signer-memorisation signature.

**Mitigation applied:** spatial augmentation (mirror flip, ±5° rotation, Gaussian noise
σ = 0.01) targets signer-specific spatial patterns directly; a clip-level spatial-flip safety
check (both hands present in > 30% of frames) prevents anatomically implausible augmented
samples; Stage 6's signer analysis produces per-signer accuracy breakdowns specifically for the
10 Signer-11-dominant classes.

---

### L4a — YouTube-Sourced Video Quality Heterogeneity

**Severity: MEDIUM. Folded under L4/L9 — noted here for completeness.**

WLASL clips were sourced from YouTube signing dictionaries, educational content, and Deaf
community video, producing variable camera distance, lighting, background clutter, angle, and
compression quality. The global hand-detection rate of 64.72% (Notebook 03) reflects this
heterogeneity. The 7 validation signers also have systematically different environmental
conditions from the 31 training signers, compounding L4's train/val gap.

**Mitigation applied:** zero-fill frames are preserved semantically rather than imputed
(L9 explains why this matters); temporal-jitter augmentation trains robustness to random
detection gaps; speed-jitter addresses cross-recording temporal-rate heterogeneity.

---

## 2. Pipeline and Feature-Engineering Limitations

### Sequence length and landmark configuration — superseded design decisions

The original pipeline design used `seq_len=60` and the full 225-dimensional landmark vector
as defaults. Both were superseded by Stage 5 ablation evidence:

| Decision | Original | Revised (final) | Evidence |
|---|---|---|---|
| Sequence length | 60 frames | **100 frames** | Group 3: +64% relative val macro-F1 (0.2354 vs 0.1434); 97% of clips truncated at seq60 vs 7% at seq100 |
| Landmark configuration | full (225-dim) | **hands_only (126-dim)** | Group 4: +110% relative val macro-F1 (0.4948 vs 0.2354); pose Fisher ratio 0.2176 vs hands-only 0.8097 |

These are not residual limitations of the *shipped* pipeline (the champion correctly uses
seq_len=100 and hands_only) but are documented here because **every Group 1–3 ablation result
in the Stage 5 registry was measured under the old, suboptimal full-225-dim configuration** and
therefore understates absolute achievable performance by roughly 2×. The *relative* comparisons
within each group (architecture vs architecture, augmentation vs augmentation) remain valid;
only the absolute magnitudes should be read as floor estimates, not ceilings, when cross-
referencing the experiment registry.

A full table of every decision superseded during the project, with evidence, is in
[Section 6](#6-decisions-superseded-during-the-project-full-history).

---

### MediaPipe Holistic Detection Failures (Dataset Construction)

**Severity: MEDIUM.**

MediaPipe Holistic fails to detect hands in 35.28% of decoded frames on average (global
both-hands-absent rate); right-hand detection failures alone average 37.60%. These occur from
motion blur, occlusion, unusual camera angles, and poor lighting in the source video. The model
cannot distinguish "left hand intentionally absent" (a genuine one-handed sign) from "left hand
present but undetected" (a detection failure) — both manifest identically as zero-fill in the
feature vector.

**Mitigation applied:** zero-fill is treated as semantic signal, never imputed (see L9); the
v1.2 dual-criterion skip policy (≥15 detected frames AND ≤95% missing rate) ensures every
retained training clip has sufficient usable content; temporal- and speed-jitter augmentation
train robustness to random detection gaps.

---

### Wrist-Relative Normalisation and Z-Coordinate Reliability

**Severity: LOW.**

Wrist-relative normalisation removes absolute hand position but cannot remove hand-scale
variation, arc-geometry variation, or execution-rate variation (a 2–3× spread across signers
for the same sign) — these residual variations are the core of the signer-independent
generalisation challenge and are only partially addressed by augmentation. MediaPipe
z-coordinates carry roughly 4% of the signal magnitude of xy coordinates and include physically
implausible depth estimates near frame boundaries and during fast motion; z-clipping at ±0.10
(affecting 37.41% of z-entries) removes outliers without distorting the core distribution.
Z-coordinates are retained in the hands_only config because dropping them empirically reduces
the Fisher ratio from 0.8097 to approximately 0.71.

---

### Augmentation: Epoch-Budget Dependency

**Severity: MEDIUM. A genuine methodological trap, now resolved but worth preserving as a
cautionary record.**

Stage 5's Group 2 ablation (80-epoch budget) concluded augmentation was *harmful*: both
augmented runs showed monotonically increasing validation loss and stopped after 19–32 epochs
with val macro-F1 near zero. The champion run (250 epochs, patience=50, identical augmentation
chain, hands_only features, lr=5e-4) achieved 0.6011 — the best result in the entire project,
5.9pp ahead of the best non-augmented run. **The Group 2 conclusion was an artefact of an
insufficient epoch budget interacting with full-225-dim noise and too-high a learning rate
(1e-3), not a real finding about augmentation.** With a 250-epoch budget, augmentation halves
the overfitting gap (≈0.48 → ≈0.24) rather than causing divergence.

**Implication for any future experimentation:** augmentation experiments must use ≥200 epochs
and patience ≥40 before drawing conclusions; the Group 2 results are not valid grounds for
disabling augmentation in any other context within this project.

---

## 3. Model and Architecture Limitations

### <a name="l9"></a>L9 — Landmark-Based Representation: Fundamental Ceiling

**Severity: MEDIUM. A deliberate, documented design trade-off, not an oversight.**

The pipeline replaces raw video pixels with MediaPipe Holistic/Hands skeletal landmarks,
trading some achievable accuracy for a CPU-deployable, sub-1MB, interpretable model. The costs
of that trade-off:

- **MediaPipe detection failures are not recoverable downstream.** When the upstream detector
  fails to find a hand, the information is permanently lost to every later pipeline stage. A
  CNN operating on raw pixels could in principle recover partial hand position even when
  explicit keypoint detection fails; this pipeline cannot. The 35.28% both-hands-absent rate in
  Stage 3 represents genuine, irrecoverable information loss, not a tunable parameter.
- **Facial grammar markers are entirely excluded.** ASL (and KSL) use eyebrow raise, mouth
  morphemes, and head tilt as grammatical markers. None of this is in the 126-dim hands_only
  feature vector. This is appropriate for isolated word-level recognition (this project's
  scope) but would be a first-order limitation for any future sentence-level or
  grammatically-aware extension.
- **3D handshape is only weakly approximated.** Z-coordinates are noisy depth estimates, not a
  true 3D reconstruction (see above). Signs that differ primarily in depth configuration may be
  confused more than the 2D-only Fisher-ratio numbers suggest.

---

### Architecture Choice: BiLSTM, not Transformer

**Severity: LOW.** Transformer-based sequence models have shown superior results on larger sign
language benchmarks but were excluded here on three grounds, all directly applicable to this
project: the ≤10MB post-quantisation size target, the ≤100ms CPU latency target, and — most
binding — the ~236-clip training set, which a Transformer of any useful depth would underfit
long before a BiLSTM does. The champion (68,771 params, 0.262 MB pre-quantisation) sits
comfortably inside both the size and latency budgets.

### <a name="l7"></a>L7 — Seed Sensitivity on a Small Validation Set

**Severity: MEDIUM.** Two identically-configured runs (`bilstm_hands_only`,
`bilstm_hands_only_v2`) produced val macro-F1 of 0.5419 and 0.4067 — a 13pp gap from
initialisation-trajectory variance alone, not a hyperparameter difference (see L3 for the
mechanism). **The champion's headline 0.6011 should therefore be read as a single favourable
draw from a distribution whose expected value is closer to 0.58 ± 0.03.** Running 3–5 seeds and
reporting mean ± std would give a materially more defensible estimate but was not run, owing
to the Stage 5 compute budget; this is an explicit, acknowledged gap rather than an oversight.
The held-out test set (evaluated exactly once, in Stage 6, never used for any selection
decision) is the most trustworthy single number this project produced for exactly this reason.

### Architecture Comparison Narrative: the Dense Inversion

**Severity: LOW.** Group 1 (full-225-dim, 80 epochs, lr=1e-3) showed the Dense feedforward
baseline (0.3276) *outperforming* every recurrent architecture (LSTM 0.1948, GRU 0.1905,
BiLSTM 0.1761). This is not evidence that temporal modelling is unnecessary — the 7.7M-parameter
Dense model produced a 0.48 train/val gap by exploiting signer-correlated absolute hand
position, a shortcut unavailable once hands_only + seq_len=100 removed that positional
information. Any architecture justification in the final report must cite the hands_only
ablation (L9 / Section 6) and the train/val gap analysis, not the Group 1 comparison in
isolation — citing Group 1 alone would be actively misleading about whether temporal modelling
matters.

---

## 4. Evaluation Limitations

### <a name="l5"></a>L5 — Validation Set Size and Metric Variance

**Severity: MEDIUM.** 52 validation clips at the project's batch size produce only ≈2 evaluation
batches. At this scale, a single misclassified clip shifts val accuracy by **1.9pp**, and a
single singleton-class flip shifts val macro-F1 by up to **2.9pp**. Stage 5 epoch-to-epoch
swings of 3–8pp are expected noise, not signal — this set a practical floor on early-stopping
patience (patience < 30 risks stopping on a downward noise excursion; the champion's patience=50
is the empirically correct choice for this dataset size). **Any two run results within 3pp of
each other should be treated as statistically indistinguishable.**

**Mitigation applied:** 90% bootstrap confidence intervals are reported for every headline
number (class-stratified clip-level resampling — see the caveat in L6); macro-F1 rather than
accuracy as the primary metric; the test set is evaluated exactly once and never used for any
selection decision.

### <a name="l6"></a>L6 — Signer-Independent Generalisation Gap (Val → Test)

**Severity: HIGH. The single most important number in this project for honestly assessing
deployment readiness.**

| Metric | Keras SavedModel | TFLite (`gesture_bilstm_v1.tflite`) |
|---|---|---|
| Val macro-F1 | 0.6011 (90% CI [0.5534, 0.6410]) | 0.5916 |
| Test macro-F1 | 0.4581 (90% CI [0.3935, 0.5076]) | **0.4867** |
| Val→Test gap | 14.30pp | 10.49pp |

The ~14pp Keras val→test gap (10.5pp on the deployed TFLite artefact — see L10 for why these
differ) is the most important honesty check in the project. Despite **zero signer overlap**
between every split, the champion's hyperparameters — best epoch, augmentation strategy,
learning rate, architecture choice — were all selected by repeatedly consulting val macro-F1
across 23 runs. This constitutes **indirect val-set overfitting**: no individual clip leaked,
but the val set's idiosyncrasies were implicitly fit through the model-selection process
itself. The test set, evaluated exactly once in Stage 6 and never consulted before that, is the
project's only genuinely unbiased estimate of generalisation to unseen signers, and it is
materially below the headline val number.

Per-signer val accuracy (7 signers, ~7–8 clips each) ranges widely — some signers are classified
near-perfectly, others show systematic failures — but the sample size per signer is too small
for individually reliable estimates; Stage 6 reports Wilson-score 90% CIs per signer rather than
point estimates for this reason.

**Mitigation applied:** signer-aware splits with zero overlap (the methodologically correct,
if conservative, choice — see Section 4 below); spatial-temporal augmentation reduces the
train/val gap from ≈0.48 to ≈0.24, which plausibly also reduces (though cannot eliminate) the
val/test gap; the test number, not the val number, should be the headline figure in any
external-facing claim about this system's accuracy.

### <a name="l8"></a>L8 — 70% Accuracy Target Not Met

**Severity: HIGH.** The project's stated target of ≥70% signer-independent validation macro-F1
was not reached; the best achieved result across all 23 Stage 5 runs is 0.6011 (Keras) /
0.5916 (TFLite). As established in L1, the available evidence (a 319% relative improvement
across systematic architecture/feature/augmentation ablation, with every dimension explored)
points to this gap being data-constrained rather than architecture-constrained, but — per the
discipline established in L1 — this project does not assert a specific clip-count that would
close the gap without a real data-scaling experiment to back that number.

### Signer-Independent Splits: Conservative but Honest

**Severity: LOW, by design.** The signer-aware split enforces zero signer overlap between
train/val/test, which systematically produces lower numbers than a naïve random split on the
same clips. Published WLASL benchmark results typically use random (signer-overlapping) splits
and are **not directly comparable** to the numbers in this project — this project's numbers are
more conservative and more representative of true deployment-time generalisation. Any external
comparison must state "signer-independent" explicitly to avoid an apples-to-oranges read.

### No External Benchmark Comparison

**Severity: LOW.** This pipeline cannot be directly compared to published WLASL-100 or
WLASL-2000 results: different class counts, random (non-signer-independent) splits, and a
46.6%-recovered subset of the intended WLASL-35 inventory. The appropriate in-project baselines
are the Dense feedforward run (Group 1, 0.3276) and the no-augmentation full-feature LSTM
(Group 2 baseline, 0.1706), both trained on the identical 236-clip set as the champion.

---

## 5. Interpretability Findings as Limitations

These findings come from Stage 6's Gradient × Input attribution analysis. They are reported
here, not only in the (not-yet-written) interpretability section of the final report, because
each one constitutes a genuine, actionable limitation that downstream consumers of this model
(including Stage 9's webcam demo, which encodes two of them directly into its UI) must be aware
of.

### <a name="l12"></a>L12 — Four Near-Degenerate Confusable Sign Pairs

**Severity: MEDIUM.** Four sign pairs show activation cosine similarity of 0.785–0.963 — close
enough that the model's internal decision boundary between them is fragile:

| Pair | Cosine similarity |
|---|---|
| think ↔ who | 0.905, 0.785 |
| later ↔ house | 0.919, 0.946 |
| cousin ↔ mother | 0.927, 0.947 |
| girl ↔ orange | 0.963, 0.937 |

This is not a hypothetical concern: it is the direct cause of several of the specific
misclassification patterns in Stage 6's confusion matrix (`before↔chair`, `cousin↔go/now`,
`drink↔boy/orange/who`, `girl↔go/now`, `who↔candy`), and it is treated as first-class
information in the shipped system — Stage 9's webcam demo deliberately surfaces a top-3
prediction list and a "~PARTNER" badge specifically so a user can see when the top prediction
might be the wrong member of one of these pairs. **This UI mitigation surfaces the ambiguity to
the user; it does not resolve it.** A genuinely improved decision boundary for these four pairs
would require either more discriminative per-clip data for the affected classes or an
architecture change, neither of which has been attempted.

### <a name="l13"></a>L13 — Right-Hand-Dominant Attribution (Possible Signer-Handedness Bias)

**Severity: LOW–MEDIUM.** Stage 6's per-landmark attribution found left-hand features carry
near-zero importance relative to right-hand features in the champion's learned representation.
This is explicitly flagged in the model's own metadata (`GesturePredictor.get_metadata()`'s
`attribution_notes`) as a **possible signer-handedness artefact**: if the training population
skewed right-hand-dominant (a plausible outcome given WLASL's small, non-random signer pool —
see L4), the model may have learned to rely on right-hand geometry more than is linguistically
justified for ASL generally. This has not been further investigated (e.g. by checking
signer-handedness metadata, which was not collected) and is reported here as an open,
unresolved bias risk rather than a confirmed one.

### Buffer Length vs. Discriminative Signal Window

**Severity: LOW, efficiency-only.** Frame-level attribution peaks around frame ~36 of the
100-frame input window and decays substantially after frame ~70. The model was trained on, and
the deployed pipeline still uses, the full 100-frame sequence — this is correct and required
for compatibility with the trained weights — but it does suggest the *system's* end-to-end
latency (buffer-fill time before the first prediction in Stage 9) could potentially be reduced
by a shorter effective window in a future retraining. This is noted in `predictor.py`'s own
docstring as a candidate future experiment, not something implemented or validated in this
project.

---

## 6. TFLite Export and Quantisation Limitations

### <a name="l10"></a>L10 — Quantisation Trade-offs

**Severity: LOW–MEDIUM overall; each sub-point individually low-risk.**

The deployment artefact uses **dynamic-range quantisation** (`tf.lite.Optimize.DEFAULT`):
int8 weights, float32 activations, no calibration data required. Stage 8's release gate passed
all six hard criteria (val/test Δ macro-F1 within ±0.03, argmax agreement ≥ 0.95 on both
splits, file under 10MB, full pipeline under 100ms) — see the Stage 8 Executive Summary for
the full gate report. Specific trade-offs documented for completeness:

- **TFLite vs Keras accuracy is not identical, and not uniformly worse.** Val macro-F1 drops
  by +0.0095 (TFLite slightly worse) but test macro-F1 *improves* by −0.0286 (TFLite
  slightly better; only 1 of 51 test clips disagrees between the two artefacts, and TFLite
  gets that one right where Keras doesn't). Both deltas are within the ±0.03 release-gate
  threshold and are attributed to quantisation's small weight perturbations acting as mild,
  essentially random regularisation on low-margin decision boundaries — not a directional
  quality regression. **Both numbers should be reported together, never just one**, since
  citing only the val delta (which looks like a regression) without the test delta (which
  doesn't) would misrepresent the quantisation impact.
- **The four confusable pairs (L12) were unaffected by quantisation** — zero non-singleton
  classes showed meaningful F1 degradation (|Δ| > 0.10) anywhere in the 35-class per-class
  delta table, and the confusable pairs specifically showed exactly zero delta. This is a
  positive finding worth stating precisely so it is not assumed that quantisation introduces
  fragility it demonstrably does not.
- **`SELECT_TF_OPS` (the TFLite flex delegate) is required, not optional, for this
  architecture.** `Bidirectional(LSTM(...))` under TF 2.13 emits `TensorListReserve` /
  `TensorListStack` ops outside the standard TFLite builtin op set; a builtins-only conversion
  attempt fails and the exporter falls back to the flex delegate automatically. This adds
  ~800KB to the Android TFLite runtime binary and ~80–100KB to the model file itself (the
  measured 0.1596 MB is roughly 2.5× the naively-expected ~0.065 MB weight-only estimate,
  entirely attributable to embedded flex-op metadata).
- **Full-integer (INT8 activation) quantisation is not currently possible for this
  architecture** — the same `TensorList` ops lack INT8 kernels in TF 2.13 without MLIR
  lowering tricks this project does not implement. This caps how far the model can be
  shrunk/accelerated on edge hardware that requires full-INT8 execution (some NPUs/DSPs).
  At 1.6% of the 10MB project target, this has no practical consequence for the current CPU
  deployment target, but would matter if a future deployment target required full-INT8.
- **Confidence-shift continuity:** TFLite's mean confidence shifts by only −0.0013 relative
  to Keras on val clips — comfortably inside the ±0.03 warning threshold — so the Stage
  6-calibrated display threshold (0.35; see L11) did not need re-calibration for the
  deployed TFLite artefact.

### <a name="l17"></a>L17 — Latency Measured on Development Hardware Only

**Severity: MEDIUM. An open gap, not a finding.**

All latency numbers in this project (TFLite median 46.86ms, p95 83.40ms, full pipeline 47.11ms
excluding MediaPipe; Stage 9's projected ~14.9 FPS end-to-end) were measured on a Windows
development machine CPU with no GPU. **The model has never been benchmarked on an actual
Android device**, which is the project's stated primary deployment target. ARM CPU
characteristics, thermal throttling under sustained use, and the specific TFLite Android
runtime's flex-delegate performance can all differ meaningfully from desktop x86 numbers. The
≤100ms project target was verified met on development hardware only; this should be treated as
a necessary-but-not-sufficient check before any real mobile deployment claim.

---

## 7. Confidence Calibration

### <a name="l11"></a>L11 — Model is Underconfident

**Severity: MEDIUM. Affects deployment UX, not raw accuracy.**

Stage 6 calibration analysis: **ECE = 0.2009, MCE = 0.3472, mean confidence = 0.5136 < mean
accuracy = 0.5769** (overconfidence gap = −0.0633, i.e. the model is *under*confident — the
less common failure direction for softmax classifiers, but still a real distortion). In plain
terms: when this model reports "55% confident," it is actually correct more often than 55% of
the time. A naive 0.50 display threshold would needlessly suppress a meaningful fraction of
correct predictions.

**Mitigation applied:** the deployed system uses a calibrated display threshold of **0.35**,
derived directly from this finding (`DEFAULT_DISPLAY_THRESHOLD` in `predictor.py`, consistently
threaded through Stage 8's verification and Stage 9's webcam demo so there is exactly one
source of truth for this value). **Not implemented:** temperature scaling (the standard
post-hoc remedy for this exact failure mode) — this project only has access to post-softmax
probabilities, and the 52-clip calibration set is too small for a reliable temperature estimate
in any case. This remains open future work, explicitly flagged rather than silently skipped.

---

## 8. Real-Time System Limitations (Stage 9)

These limitations are specific to the live webcam demo (`src/demo/webcam_demo.py`) and the
streaming inference layer it builds on (`GesturePredictor` / `GestureStreamSession`,
`src/inference/predictor.py`), as distinct from the static, clip-level evaluation numbers
discussed above.

### <a name="l14"></a>L14 — MediaPipe Dependency: Real-Time Detection Degrades in Adverse Conditions

**Severity: MEDIUM.** The same upstream-detector dependency documented for the *dataset* (see
the "MediaPipe Holistic Detection Failures" section above) applies, with additional real-time-
specific risk, to live inference. In practice, hand-landmark detection quality measurably drops
in low lighting (roughly < 200 lux), high-motion/fast signing, camera angles beyond ~45° from
frontal, and partial occlusion. The demo's auto-reset mechanism (clearing the rolling buffer
after 3 consecutive no-detection frames) prevents a stale buffer from silently corrupting the
next sign's prediction, but **cannot recover information from sustained poor detection** — if a
signer is in poor lighting for the whole clip, the system will simply fail to buffer a usable
sequence at all, not produce a degraded-but-useful prediction.

A secondary, lower-probability risk: the demo prefers **MediaPipe Hands** over **MediaPipe
Holistic** for live inference specifically (Stage 8's Executive Summary §11.3 recommendation,
implemented in Stage 9 — see the superseded-decisions table in Section 6) because it is faster
(~8–10ms vs ~18ms) and avoids unused pose computation. If MediaPipe Hands fails to initialise on
a given machine, the demo falls back to Holistic, with pose output always discarded to preserve
the hands_only input contract. The two detectors are not guaranteed to have identical hand-
detection accuracy in practice, so a small, environment-dependent (which extractor initialised
successfully) accuracy variance is possible, though the input *distribution* the model receives
is kept identical by design (pose always zero-filled in both paths).

### <a name="l15"></a>L15 — No Out-of-Distribution or Uncertainty Detection

**Severity: MEDIUM. A genuine, confirmed gap — not yet mitigated.**

The system has no mechanism to detect when its input does not correspond to any of the 35
trained signs. If a user performs a gesture outside the trained vocabulary, makes an incidental
hand movement, or simply rests their hands in frame, the model will still produce a full 35-way
softmax distribution and report *some* class as the top prediction — there is no "none of the
above" option, no entropy-based rejection, and no frame-quality (blur/occlusion) pre-filter.
The calibration-aware display threshold (L11) provides partial mitigation by suppressing
low-confidence predictions, but a confidently-wrong out-of-distribution prediction is possible
in principle and has not been specifically tested for. This was identified independently in two
rounds of code review during Stage 9 and is recorded here as confirmed-but-unaddressed, rather
than theoretical.

### <a name="l16"></a>L16 — Thread Safety and Single-Sign Scope

**Severity: LOW for the current single-process demo; MEDIUM for any future server deployment.**
`GesturePredictor`, and the `FrameBuffer` / `PredictionSmoother` instances `GestureStreamSession`
composes from its public API, all hold mutable streaming state and are **not thread-safe** — a
single `tf.lite.Interpreter` must not be invoked concurrently from multiple threads, and the
rolling buffer/smoother state has no locking. One instance per worker thread is required for any
concurrent deployment (e.g. a future multi-user server). Separately, by design, this system
performs **isolated, single-sign recognition only**: it does not segment continuous signing into
individual signs, does not handle co-articulation between consecutive signs, and does not
recognise sentence-level grammar. The Stage 9 sliding-window + majority-vote approach
approximates a usable real-time experience but is explicitly not a continuous recognition
system; production deployment would require a separate upstream sign-segmentation module.

A related, deliberate UX trade-off worth naming explicitly: the displayed sign in the HUD lags
the model's raw per-frame prediction by design — the 5-frame majority-vote smoother plus the
3-confident-frame debounce in `PredictionHistory` together mean a freshly-stable sign typically
takes ~8 frames (smoother window + debounce) of consistent, confident model output before it
is shown to the user. This is a deliberate stability/responsiveness trade-off, not a bug, but it
means the HUD is never a frame-accurate readout of the underlying model.

---

## 9. KSL Adaptation: The Actual Deployment Target

### <a name="l18"></a>L18 — ASL ≠ KSL: This Model Cannot Be Directly Deployed for the Target Language

**Severity: CRITICAL for the project's actual stated goal.** Everything above describes the
limitations of a WLASL-35 ASL classifier — which is, by design, a technical-verification
exercise, not the production target. The production target is **Kenyan Sign Language**, and
this model has no knowledge of KSL whatsoever. ASL and KSL differ structurally:

- Different phonemic handshape inventories.
- Different movement patterns and use of signing space.
- Different non-manual (facial) grammatical markers — which this pipeline excludes entirely
  regardless of language (see L9), compounding the gap.
- Almost entirely different lexical items — most individual signs share no resemblance.

A KSL sign that happens to resemble an ASL sign geometrically will not reliably trigger the
correct class, and a KSL sign with no ASL analogue will simply be forced into whichever of the
35 ASL classes the model finds nearest — with no mechanism (per L15) to flag that the input
doesn't belong to the trained vocabulary at all.

**What does plausibly transfer:** the hands_only feature configuration (L9, Section 6) excludes
pose landmarks, which encode signer-specific body morphology rather than sign-specific geometry
— this was an accuracy-driven decision for WLASL-35, but it has the secondary, welcome effect of
removing one entire axis of cross-linguistic, signer-specific domain shift before KSL transfer
even begins. A full-landmark model would need to *additionally* unlearn body-posture reliance
during KSL fine-tuning; this one does not carry that baggage.

**Recommended KSL adaptation strategy (not yet executed):**
1. Collect KSL training data — current AI4KSL provides ~40 clips/sign, below the empirically
   observed viability threshold (L1, L2) for a 500-sign system targeting 85% accuracy; ~100–200
   clips/sign is the working estimate, itself extrapolated from this project's own
   clips-per-class-vs-F1 relationship rather than asserted from nothing.
2. Train and compare three protocols on a genuinely held-out, signer-independent KSL test set:
   KSL-from-scratch (establishes the achievable ceiling with the target architecture); an
   ASL-pretrained frozen BiLSTM with a new KSL classifier head (tests whether ASL temporal
   motion patterns transfer at all); and a fully fine-tuned ASL→KSL model (the expected best
   performer once sufficient KSL data exists).
3. Evaluate with per-class recall, not aggregate accuracy, given the same per-class data
   scarcity risk (L2, L3) is likely to recur at KSL scale, possibly worse with 500 classes.
4. Architecture scaling: the champion's 64 hidden units (68,771 params) is almost certainly
   insufficient capacity for 500 classes; `hidden_units ∈ {128, 256}` should be the first
   ablation point for a KSL champion search, alongside a fresh sequence-length and
   landmark-config ablation specific to KSL phonology (L9's hands_only finding should not be
   assumed to transfer to KSL without re-verification).
5. Estimated timeline: 3–6 months, dominated by data collection rather than modelling — this
   matches the project's own Stage 5→8 finding that architecture search converges quickly
   relative to the cost of data scarcity.

---

## 10. Project Status Context

**Severity: not a model limitation — a documentation/process limitation, noted for completeness.**
As of this document's last update, Stages 1–9 are complete (data ingestion through the working
real-time webcam demo). **Stage 10 (Docker/CI/CD/Makefile/remaining unit tests) and Stage 11
(the one-page report, the five-question theoretical assessment, and `MODEL_CARD.md`) are not
yet complete.** Any claim about this system's accuracy or readiness made before those documents
exist should be sourced to this file and the Stage 6/8 evaluation/verification reports directly,
not to a not-yet-written report. This document will receive one further update pass once Stage
11's report is drafted, to ensure the two documents agree on every quoted number.

---

## 6. Decisions Superseded During the Project (Full History)

The following design decisions were made early and later revised in light of empirical
evidence accumulated across stages. Preserved here as an audit trail — anyone reading an older
note, commit message, or the Part-2 "locked constants" table in the original project handoff
document should cross-check against this table before trusting a stale default.

| Decision | Original | Revised (final) | Evidence | Stage |
|---|---|---|---|---|
| Primary sequence length | seq_len = 60 | **seq_len = 100** | +64% relative val macro-F1; 97% truncation at seq60 vs 7% at seq100 | 5 |
| Landmark configuration | full (225-dim) | **hands_only (126-dim)** | +110% relative val macro-F1; pose Fisher ratio 0.2176 vs 0.8097 | 5 |
| Augmentation for champion runs | Disabled (Group 2 finding) | **spatial_temporal enabled** | Champion 0.6011 (aug) vs 0.5419 (best no-aug); Group 2 was an 80-epoch artefact | 5 |
| Champion architecture | Ambiguous after Group 1 inversion | **BiLSTM** | Champion vs LSTM champion (0.6011 vs 0.4286); training-curve quality, not Group 1 in isolation | 5 |
| Early-stopping patience | 10–15 (config default) | **≥40–50** | patience=25 run underperformed (0.4181) vs patience≥30 (0.5419+) | 5 |
| Minimum training epochs | 80 (base.yaml) | **200–250 for augmented runs** | Augmented convergence requires extended budget; see Section 2's epoch-budget finding | 5 |
| Live-inference landmark extractor | MediaPipe Holistic | **MediaPipe Hands** (Holistic retained as a fallback only) | ~8–10ms vs ~18ms per frame, no unused pose computation; Stage 8 §11.3 recommendation | 8→9 |
| Quantisation strategy | Unspecified / assumed builtins-only | **Builtins-first, SELECT_TF_OPS fallback** | Builtins-only conversion fails for `Bidirectional(LSTM)` under TF 2.13; confirmed in Stage 6 benchmark integration testing | 6→8 |

*Note: live-inference extractor and dataset-construction extractor are intentionally different
and should not be conflated — Stage 1–3's landmark dataset was, and remains, built with
MediaPipe Holistic; only the real-time Stage 9 demo's extraction path uses Hands.*

---

## 11. Known Issues Resolved (Full History)

### Pre-Stage-6 (caught and fixed during Stages 1–5)

| Issue | Identified | Resolution |
|---|---|---|
| v1.1 30% ratio-based skip threshold produced a 76% clip skip rate | Notebook 02 | v1.2 dual-criterion policy: min 15 detected frames + 95% catastrophic-missing filter |
| Leading-zero `video_id` mismatch silently skipped 80 clips | Notebook 04 | All video IDs normalised to integer-string form before split/inventory join |
| Individual hand missing-rate columns zeroed in `landmark_inventory.csv` | Notebook 02 | Display-only bug; `missing_both_pct` (the field the skip policy actually uses) was always correct |
| `speed_jitter` interpolation violated the zero-fill invariant at detected↔zero boundaries | Notebook 04 | Zero-aware interpolation: output forced to zero whenever both surrounding source frames are zero for that slot |
| Stray debug `print()` in `AugmentationPipeline.__call__()` | Stage 4 review | Removed; all modules use `get_logger(__name__)` |
| `GestureDataset` epoch counter not incrementing under `model.fit(epochs=N)` | Stage 4 review | Per-epoch `load_split()` training-loop contract enforced (Critical Rule #1) |

### Stage 6–8 (caught in design review before shipping — never reached production)

These did not ship as bugs in any released artefact; they are recorded because catching them
is itself evidence of, and a model for, the review discipline this project tries to maintain.

| Issue | Where caught | Would-have-been impact if shipped |
|---|---|---|
| Double-inference in the original Stage 8 verification orchestrator | `verify.py` design review (bug B1) | Every accuracy-comparison run would have silently done 2× the necessary inference work |
| Inverted variable naming (`X` array bound to a `y_true_val` name) | `verify.py` design review (bug B2) | Would have produced confusing, possibly mismatched metrics if ever refactored carelessly |
| Singleton-class fallback logic always evaluating False (`0 == 1`) | `verify.py` design review (bug B3) | All classes with missing `support` metadata would have been silently mismarked as non-singleton in the per-class delta table |
| `model.input_shape` compared as a raw tuple without normalisation | `convert.py` design review (critical-review #2) | Would silently and confusingly fail shape verification for any future multi-input Functional-API model |

### Stage 9 (caught and fixed in an intermediate draft of the shipped webcam demo)

| Issue | Identified | Resolution |
|---|---|---|
| FPS/latency tracker recorded synthetic `0.0` placeholders alongside genuine measurements for the "stage that didn't run" each frame | Critical review of `webcam_demo.py` | `pipeline_ms`/`inference_ms` are now genuinely measured on every frame they apply to and simply not recorded (not zero-recorded) otherwise — the HUD's displayed latency numbers are now trustworthy |
| `GesturePredictor.display_threshold` (read-only) and the demo session's live, hotkey-adjustable threshold were two independent sources of truth | Critical review of `webcam_demo.py` | The predictor's resolved threshold is read exactly once at startup to seed the session; the session-local value is the single source of truth thereafter |

---

## 12. Summary: Accuracy Interpretation Guide

| Context factor | Value | Implication |
|---|---|---|
| Training clips | 236 | ~6.7 clips/class — severely data-limited (L1, L2) |
| Validation clips | 52 | High-variance estimate; ±3–5pp implicit CI (L5) |
| Val singleton classes | 21/35 | Per-class val metrics unreliable for 60% of classes (L3) |
| Signer independence | Enforced, zero overlap | Conservative vs. random-split published benchmarks |
| Dataset completeness | 46.6% | Hard ceiling; no architecture change compensates (L1) |
| Class-weight ratio | 6.50× | Aggressive rebalancing required and applied throughout |
| Primary metric | Macro-F1 (sklearn, zero_division=0) | Not overall accuracy — chosen specifically to surface per-class failure |
| Val macro-F1 (Keras / TFLite) | 0.6011 / 0.5916 | Minimum-viability gate met; 70% target not met (L8) |
| Test macro-F1 (Keras / TFLite) | 0.4581 / 0.4867 | **The honest, deployment-relevant number** (L6) |
| Seed sensitivity | ±3–5pp from trajectory variance | Champion ≈ 0.58 ± 0.03 expected value, not a fixed point (L7) |
| TFLite deployment artefact | `gesture_bilstm_v1.tflite`, 0.1596 MB | Release-gate verified; SELECT_TF_OPS required (L10) |
| Calibration | ECE = 0.2009, underconfident | 0.35 display threshold calibrated accordingly (L11) |
| Confusable pairs | 4 pairs, cosine sim 0.79–0.96 | Surfaced via top-3 UI, not resolved (L12) |
| Real-time latency | 47.11ms full pipeline (dev CPU only) | Never benchmarked on target Android hardware (L17) |
| Target language | ASL (not KSL) | Model not deployable to production target without adaptation (L18) |

**What the champion result means in plain language:** a BiLSTM model with 68,771 parameters,
trained for 171 effective epochs on 236 American Sign Language clips with spatial-temporal
augmentation and class-weight balancing, achieves **59.2% macro-averaged F1 on its held-out
validation set and 48.7% on its only-ever-evaluated-once held-out test set** (both TFLite,
deployment-artefact numbers), across 35 signs, from signers entirely unseen during training,
on a dataset with only 46.6% completeness. This is a signer-independent, class-balanced, honest
estimate, conservative relative to published benchmarks that use random splits, and it is the
correct, intentionally-conservative baseline for evaluating whether this engineering approach —
not this specific ASL vocabulary — is sound enough to invest in for the actual KSL production
target.

---

*This document was last revised at the conclusion of Stage 9. It supersedes both the
post-Stage-5 original and the post-Stage-9 draft that preceded this version. It will receive a
final update once Stage 11 (the one-page report and theoretical assessment) is drafted, to
confirm both documents quote identical figures throughout.*