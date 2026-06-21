# Model Card — gesture_bilstm_v1 (WLASL-35 Champion)

**Companion document:** [`LIMITATIONS.md`](./LIMITATIONS.md) is the authoritative, detailed
limitations register for this project. This card summarises the deployment-relevant subset
and cross-references specific limitation IDs (`L1`–`L18`) where useful — the two documents
are written to agree on every quoted number, and `LIMITATIONS.md` should be treated as the
source of truth if either ever appears to diverge.

---

## 1. Model Overview

| Field | Value |
|-------|-------|
| Name | gesture_bilstm_v1 |
| Version | 1.0.0 |
| Task | 35-class American Sign Language word-level gesture recognition |
| Architecture | BiLSTM, 2 layers, 32 units/direction (concatenated output width = 64) |
| Input | `(1, 100, 126)` float32 — 100 frames × 126 hands-only landmarks |
| Output | `(1, 35)` float32 — softmax probabilities over 35 ASL signs |
| Parameters | 68,771 |
| Deployment artefact | `models/gesture_bilstm_v1.tflite`, 0.1596 MB, dynamic-range quantised |
| Created | 2026-06-20 |
| MLflow Run ID | `cb16f689d2294001a2ff2d3e02419d27` |
| Config hash | `5809193d37e0d480e409b8e3112e70c8de9008497a29727b411a7128e73287a6` |
| Release gate | **PASS** — all 6 Stage 8 hard criteria met (see Section 5) |
| Pipeline stage status | Stages 1–9 complete (data → training → evaluation → TFLite export → live demo). Stages 10–11 (CI/CD, theoretical assessment, one-page report) not yet complete. |

This model is the output of a systematic, 23-run ablation across architecture, augmentation,
sequence length, and landmark configuration (Stage 5), evaluated on a held-out test set exactly
once (Stage 6), exported to a verified TFLite artefact through an automated release gate
(Stage 8), and deployed in a working real-time webcam demo (Stage 9). Every number in this
card traces to one of those stages' logged outputs — none are estimated or interpolated unless
explicitly marked "estimated."

---

## 2. Intended Use

### Primary use case
Real-time, isolated ASL word-level sign classification from live video, deployed via the
**TFLite artefact** (`gesture_bilstm_v1.tflite`) through `GesturePredictor`
(`src/inference/predictor.py`) — the project's single, unified inference entry point (see
Section 10). The **stated primary deployment target is Android mobile**, with the desktop/
laptop webcam demo (`src/demo/webcam_demo.py`, Stage 9) serving as both a development tool and
a CPU-only reference implementation. **Latency has only been measured on development-machine
CPU (Windows, no GPU) — never on an actual Android device** (see `LIMITATIONS.md` L17). Treat
the Android deployment as unverified until that benchmarking is performed.

### Secondary use case
A transfer-learning baseline for Kenyan Sign Language (KSL) recognition. The model is
deliberately *not* a finished KSL system — see Section 9.

### Out-of-scope uses
- Production accessibility tooling without additional human review and a fallback
  communication channel.
- Recognition of any sign outside the 35 trained classes. **The model has no
  out-of-distribution or uncertainty-rejection mechanism** (`LIMITATIONS.md` L15): an
  unrecognised gesture, incidental hand motion, or a KSL sign will still produce a confident-
  looking top-1 prediction among the 35 known classes, not a "this isn't a trained sign"
  signal. The calibrated 0.35 display threshold (Section 5) reduces but does not eliminate
  this risk.
- Signer identification, authentication, or any biometric use.
- Continuous or sentence-level sign language recognition. This model performs **isolated,
  single-sign classification only** — it has no sign-segmentation, co-articulation handling,
  or grammatical structure recognition (`LIMITATIONS.md` L16).
- Concurrent/multi-threaded serving of a single `GesturePredictor` instance — it is not
  thread-safe (Section 7; `LIMITATIONS.md` L16).
- Any use where a misclassification could harm a deaf or hard-of-hearing user without a human
  fallback mechanism.

---

## 3. Training Data

| Field | Value |
|-------|-------|
| Dataset | WLASL (Word-Level American Sign Language) v0.3 |
| Total clips listed in inventory | 751 |
| Clips recoverable on disk | 350 / 751 (**46.6% completeness** — see Section 7) |
| Usable after landmark extraction | 339 clips (schema v1.2 dual-criterion skip policy) |
| Training split | 236 clips, 31 signers |
| Validation split | 52 clips, 7 signers |
| Test split | 51 clips, 7 signers |
| Signer overlap across splits | **Zero** — splits are fully signer-disjoint by construction |
| Mean clips per class | 6.7 |
| Class imbalance ratio (working set) | 6.50× (up from 3.20× on the full intended inventory — the missing 53.4% of clips is not lost uniformly across classes) |
| Permanently inaccessible clips | 401 / 751 — dead YouTube source URLs, not recoverable by this project |

**Data completeness is a hard ceiling on achievable accuracy** (`LIMITATIONS.md` L1), not a
tunable pipeline parameter. Stage 5's systematic ablation across every available design
dimension — architecture, augmentation, sequence length, landmark configuration, training
budget — produced a +319% relative val macro-F1 improvement over the initial baseline while
holding the data fixed, evidence that the pipeline was actively data-constrained rather than
architecture-constrained throughout. **This project deliberately does not publish a specific
projected accuracy under hypothetical full data completeness** — no data-scaling ablation was
run to support such a number, and asserting one would be unsupported precision dressed up as
an estimate.

---

## 4. Preprocessing

All preprocessing is implemented once, in `src/features/pipeline.py::FeaturePipeline`, and is
applied identically at training and inference time via a single shared instance inside
`GesturePredictor` — this is an architectural guarantee, not a convention callers must
remember to follow (see Section 10). Callers must never apply any preprocessing of their own
before passing landmarks to the model.

| Step | Description |
|------|-------------|
| Feature vector layout | 225 raw values/frame: `[0:63]` left hand, `[63:126]` right hand, `[126:225]` pose (pose is always present in the raw vector but discarded by landmark selection below) |
| Wrist normalisation | Subtract wrist position (landmark 0) from all hand landmarks, per hand slot, guarded by that slot's own detection state |
| Z-coordinate clipping | Soft-clip z to ±0.10 (removes implausible MediaPipe depth outliers; affects ~37% of z-entries) |
| Sequence length | Pad with zeros or centre-crop to **100 frames** |
| Landmark selection | **hands_only**: only `[0:126]` (both hands) is retained as the actual model input — pose `[126:225]` is computed but never reaches the model |
| Zero-fill semantics | Zero-filled frames/slots are **semantic** (e.g. a genuinely one-handed sign), never imputed or treated as missing data — this is a deliberate, load-bearing modelling choice, not a placeholder |
| Augmentation | Spatial-temporal augmentation chain used **during training only**; never applied at inference (`training=False` is enforced unconditionally inside `GesturePredictor`) |

**Landmark extraction backend — training data vs. live inference differ, by design:**

| Context | Extractor | Why |
|---|---|---|
| Dataset construction (Stages 1–3, the 339 training/val/test clips) | MediaPipe **Holistic** | Holistic was the extractor available and used when the WLASL clips were processed offline; its pose output was simply never fed to the model even then. |
| Live inference (`src/demo/webcam_demo.py`, Stage 9) | MediaPipe **Hands** (Holistic as an automatic fallback only) | ~8–10ms vs Holistic's ~18ms per frame, with no pose computation wasted on a slot the model discards. Recommended explicitly in the Stage 8 Executive Summary (§11.3) and implemented in Stage 9. |

Both code paths are constructed so the **model receives an identical input distribution**
either way — the Holistic fallback in the live demo explicitly zero-fills its pose output to
match the Hands path, rather than passing through real pose values (which would silently shift
the input distribution depending on which extractor happened to initialise — see
`LIMITATIONS.md` L14 for the residual, lower-probability accuracy-variance risk this leaves
open). Any future integration that calls `GesturePredictor.predict_from_video()` or
`predict_from_landmarks()` directly with a raw 225-dim array (e.g. a pre-recorded clip) should
also use the same extraction convention as the training data (Holistic) for landmarks not
already hands_only-sliced.

---

## 5. Performance

### Overall metrics (val split, 52 clips, 7 unseen signers)

| Model artefact | val macro-F1 | val accuracy |
|-------|-------------|--------------|
| Keras SavedModel | **0.6011** (90% bootstrap CI [0.5534, 0.6410]) | 0.5769 |
| **TFLite, deployed** | 0.5916 | 0.5769 |

### Test split (51 clips, 7 unseen signers — evaluated exactly once, never used for model selection)

| Model artefact | test macro-F1 | test accuracy |
|-------|--------------|---------------|
| Keras SavedModel | 0.4581 (90% bootstrap CI [0.3935, 0.5076]) | 0.4902 |
| **TFLite, deployed** | **0.4867** | 0.5098 |

**Read the val and test numbers together, not in isolation.** The ~14pp Keras val→test drop
is this project's most important honesty check (`LIMITATIONS.md` L6): despite zero signer
overlap anywhere, the champion's hyperparameters were selected by repeatedly consulting val
macro-F1 across 23 runs, which constitutes indirect val-set overfitting even with no literal
data leakage. **The test number is the more trustworthy estimate of generalisation to unseen
signers**, and TFLite's test macro-F1 (0.4867) is the actual deployed artefact's number — it
should be the headline figure in any external claim about this system's accuracy, not the
higher val number.

**Seed sensitivity caveat:** the champion's headline 0.6011 (Keras val) is a single favourable
training run, not a stable point estimate — two identically-configured runs differing only in
random initialisation trajectory produced 0.5419 and 0.4067 on the same val set
(`LIMITATIONS.md` L7). **Treat the champion's val macro-F1 as ≈0.58 ± 0.03**, not as a fixed
constant; the 0.6011 figure quoted throughout this card is the actual selected checkpoint's
measured value, reported honestly as such, not adjusted toward the expected value.

### Release gate (Stage 8) — all 6 hard criteria PASSED

| Criterion | Measured | Threshold | Result |
|-----------|----------|-----------|--------|
| Val Δ macro-F1 (Keras − TFLite) | +0.0095 | ≤ ±0.03 | PASS |
| Test Δ macro-F1 (Keras − TFLite) | −0.0286 | ≤ ±0.03 | PASS |
| Val argmax agreement (Keras ↔ TFLite) | 98.08% | ≥ 95% | PASS |
| TFLite file exists and is valid | True | — | PASS |
| TFLite size | 0.1596 MB | ≤ 10 MB | PASS |
| Full pipeline latency (excl. MediaPipe) | 47.11 ms | ≤ 100 ms | PASS |

The test-split delta is *negative* — TFLite is slightly **more** accurate than Keras on test
(only 1 of 51 clips disagrees, and TFLite gets that one right where Keras doesn't). This is
attributed to quantisation's small weight perturbations acting as incidental, near-random
regularisation on a low-margin decision boundary, not a genuine quality improvement to rely on
(`LIMITATIONS.md` L10). Both deltas are well inside the ±0.03 gate regardless of sign.

### Deployment characteristics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| TFLite file size | 0.1596 MB | ≤ 10 MB | Verified — 1.6% of budget |
| Quantisation mode | Dynamic-range (`tf.lite.Optimize.DEFAULT`) | — | int8 weights, float32 activations |
| Flex delegate requirement | **`SELECT_TF_OPS` required** | — | `Bidirectional(LSTM)` under TF 2.13 emits `TensorListReserve`/`TensorListStack` ops outside the standard builtin set; adds ~800KB to the Android runtime binary |
| Pipeline latency (FeaturePipeline + TFLite, excl. MediaPipe) | 47.11 ms median, 83.40 ms p95 | ≤ 100 ms | Verified on dev-machine CPU only — **not yet verified on Android** |
| TFLite vs Keras CPU speedup | 82.4× | — | Keras SavedModel is not viable for real-time use (≈3.9s/inference) |
| Estimated end-to-end FPS (incl. MediaPipe) | ~14.9 FPS (dev machine, **estimated**, not a measured live-session average) | ≥ 10 FPS | Stage 9's `webcam_demo.py` instruments and prints a true measured session-average FPS at exit; no specific session's measured value is recorded in project documentation as of this card |
| Argmax agreement (Keras ↔ TFLite) | 98.08% (val) / 98.04% (test) | ≥ 95% | PASS |

**Full-integer (INT8 activation) quantisation is not available for this architecture** — the
same `TensorList` ops lack INT8 kernels in TF 2.13 without MLIR lowering tricks this project
does not implement. This is a hard constraint on how much further this specific architecture
could be compressed for edge hardware that requires full-INT8 execution; at 1.6% of the
project's size budget it has no practical consequence for the current target, but would matter
for a different future deployment target.

### Calibration

The model is **underconfident**, not overconfident: mean confidence = 0.5136 < mean accuracy =
0.5769 on the val set (ECE = 0.2009, MCE = 0.3472). The recommended and deployed display
threshold is **0.35**, calibrated directly from this finding and threaded through the codebase
from a single constant (`DEFAULT_DISPLAY_THRESHOLD` in `predictor.py`) so the value cannot
silently drift between modules. **A naive 0.50 threshold would suppress a meaningful fraction
of genuinely correct predictions.** Temperature scaling, the standard post-hoc fix for this
calibration direction, is not implemented: this module receives post-softmax probabilities
(no access to pre-softmax logits at the deployment layer), and the 52-clip calibration set is
too small for a reliable temperature estimate in any case (`LIMITATIONS.md` L11). The TFLite
artefact's confidence shift relative to Keras (−0.0013, well inside the ±0.03 warning
threshold) confirmed no re-calibration was needed after quantisation.

---

## 6. Per-Class Performance

### High-risk classes (small training data — interpret with active caution)

| Sign | Train clips | Val clips | Champion F1 | Failure pattern |
|------|------------|-----------|-------------|-------|
| `think` | 3 | 2 | 0.00 | **Data-floor failure** — F1=0.0 in 8/9 champion candidate runs across every config tested. Treated as effectively unlearnable at this data scale, not as a tunable weakness. |
| `clothes` | 2 | 1 | 1.00 (erratic) | **Augmentation-dependent** — F1=0.0 in every no-augmentation run, F1=1.0 only with spatial-temporal augmentation; the model needs augmentation-induced regularisation to find the learnable signal in just 2 training clips. |
| `birthday` | 4 | 1 | 1.00 | Relatively stable across configurations. |
| `name` | 4 | 1 | 1.00 (augmented) | Augmentation-dependent, same pattern as `clothes`. |
| `book` | 4 | 2 | 0.50–1.00 (variable) | Augmentation-sensitive. |

21 of 35 validation classes are singletons (support = 1). F1 on these classes is necessarily
binary (0.0 or 1.0) and carries no statistical reliability — see `LIMITATIONS.md` L3 for the
empirical demonstration that two identically-configured runs differed by up to 13pp in val
macro-F1 purely from singleton-class noise.

### Known confusable pairs (Stage 6 attribution analysis, activation cosine similarity > 0.78)

| Pair | Cosine similarity | Quantisation impact |
|------|------------------|---|
| think ↔ who | 0.905, 0.785 | Zero F1 delta after TFLite quantisation |
| later ↔ house | 0.919, 0.946 | Zero F1 delta after TFLite quantisation |
| cousin ↔ mother | 0.927, 0.947 | Zero F1 delta after TFLite quantisation |
| girl ↔ orange | 0.963, 0.937 | Zero F1 delta after TFLite quantisation |

These four pairs sit close enough in the model's learned activation space that the decision
boundary between them is fragile (`LIMITATIONS.md` L12) — this is the direct cause of several
specific confusion-matrix entries (`before↔chair`, `cousin↔go/now`, `drink↔boy/orange/who`,
`girl↔go/now`, `who↔candy`). The deployed webcam demo treats this as first-class information:
it displays a top-3 prediction list and an inline "~PARTNER" badge whenever the top prediction
belongs to one of these pairs, specifically so a user can see when the second-place prediction
might be the correct one. **This UI surfaces the ambiguity; it does not resolve it.** Reassuringly,
Stage 8 confirmed none of these four pairs were further destabilised by quantisation — all four
show exactly zero F1 delta between the Keras and TFLite artefacts.

A related, lower-confidence finding: left-hand landmark features carry near-zero attribution
importance relative to right-hand features in the champion's learned representation, possibly
a signer-handedness artefact from the small, non-random WLASL signer pool. This has not been
further investigated and is reported as an open, unconfirmed bias risk (`LIMITATIONS.md` L13).

---

## 7. Limitations

*Full detail, evidence, and severity ratings for every item below are in
[`LIMITATIONS.md`](./LIMITATIONS.md); IDs in brackets cross-reference that document.*

### Data limitations
- **46.6% data completeness is a hard ceiling on achievable accuracy** — not a tunable
  parameter (`L1`).
- **6.7 clips per class on average** — all 35 classes are below the 20-clip minimum considered
  adequate for reliable signer-independent LSTM training (`L2`).
- **`think` is effectively unlearnable** at current data scale — 3 training clips with zero
  signer overlap is insufficient; F1=0.0 in 8 of 9 champion candidate runs (`L2`).
- **7 validation signers, ~7–8 clips each** — per-signer accuracy estimates are highly noisy;
  expect a wide spread (`L4`, `L5`).

### Model limitations
- **ASL-only.** Cannot recognise KSL, BSL, or any other sign language (`L18`).
- **35 signs only** — not a complete vocabulary for any real-world communication task.
- **Signer-independent generalisation gap.** Test macro-F1 (0.4581 Keras / 0.4867 TFLite) is
  meaningfully lower than val macro-F1 (0.6011 Keras / 0.5916 TFLite), indicating indirect
  val-set overfitting from the model-selection process itself, not a methodological flaw or
  data leakage (`L6`).
- **Seed-sensitive headline number.** The champion's val macro-F1 should be read as ≈0.58 ±
  0.03, not as a fixed point estimate (`L7`).
- **70% target not met.** Best achieved: 0.6011 (Keras val). Evidence points to this being
  data-constrained rather than architecture-constrained, but no specific data quantity is
  asserted as sufficient to close the gap without a real scaling experiment (`L1`, `L8`).
- **No out-of-distribution detection.** Any input — including a non-trained gesture or a KSL
  sign — produces a confident-looking prediction among the 35 known classes, never a "not
  recognised" signal (`L15`).
- **One-handed sign / handedness ambiguity.** Signs performed with the non-dominant hand
  produce zero-filled vectors for the expected dominant-hand slot; robustness to handedness
  variation has not been specifically tested, and the model may carry a right-hand-dominant
  bias from the training population (`L13`).

### Real-time system limitations (Stage 9)
- **Lighting and detection sensitivity.** MediaPipe hand detection measurably degrades below
  ~200 lux, during fast/high-motion signing, at camera angles beyond ~45° off-frontal, and
  under partial occlusion. The system's auto-reset (after 3 consecutive no-detection frames)
  prevents a stale buffer from corrupting the next prediction but cannot recover information
  lost to sustained poor detection (`L14`).
- **Displayed prediction lags the raw model output by design.** The 5-frame majority-vote
  smoother plus 3-confident-frame debounce together mean a freshly-stable sign typically takes
  ~8 frames of consistent model output before the HUD updates — a deliberate stability trade-off,
  not a bug, but the HUD is never a frame-accurate readout of the underlying model (`L16`).
- **Latency verified on development hardware only.** All latency numbers in this card were
  measured on a Windows CPU development machine with no GPU. The model has never been
  benchmarked on the stated primary deployment target (an Android device) (`L17`).

### Calibration limitations
- Softmax outputs are underconfident (ECE = 0.2009). Temperature scaling is the standard fix
  but is not implemented — only post-softmax probabilities are available at the deployment
  layer, and the 52-clip val set is too small for a reliable temperature estimate (`L11`).

### Deployment limitations
- **`GesturePredictor` is not thread-safe** — `FrameBuffer` and `PredictionSmoother` hold
  mutable streaming state, and a single `tf.lite.Interpreter` must not be invoked concurrently.
  Construct one instance per worker thread for any concurrent deployment (`L16`).
- **`SELECT_TF_OPS` required.** The Android TFLite runtime binary must include the flex
  delegate (~800 KB overhead). Not compatible with a minimal/builtins-only TFLite runtime
  build (`L10`).
- **Full-integer quantisation unavailable** for this architecture — caps further size/speed
  optimisation on INT8-only edge accelerators (`L10`).
- **`Dockerfile.inference`** (the minimal inference-only image) cannot train, evaluate, or
  export models — it intentionally excludes `src/evaluation` and the full TensorFlow training
  stack.

---

## 8. Ethical Considerations

### Dataset provenance
WLASL was collected from publicly available YouTube videos. Signers did not provide explicit
consent for use in this specific machine learning research project. The dataset is used here
in accordance with its published terms, for academic and research purposes only.

### Demographic representation
The WLASL dataset was collected primarily in North American settings with English-speaking
signer communities. It does not represent regional ASL dialect variation, non-North-American
signing communities, or a controlled/diverse distribution of skin tones, hand sizes, or signing
environments. The 50%+ single-signer dominance observed in four training classes (`L4`)
illustrates how thin this representation actually is even within the recovered 46.6% of the
dataset. Models trained on WLASL may perform meaningfully worse for signers from
underrepresented communities; this has not been measured, because the dataset carries no
demographic metadata to measure it against.

### Accessibility risk
A gesture recognition system that fails silently — producing a confident, wrong label rather
than a "not recognised" signal (`L15`) — poses a real accessibility risk if deployed as a
communication aid without a human fallback. **This model is a research and engineering
baseline, not a production accessibility tool**, and should not be the sole channel for any
communication that matters.

### Misuse risk
This model outputs a sign-class label, not a person identity, and performs no biometric
inference. It poses minimal misuse risk beyond general computer-vision misuse patterns common
to any landmark-based pose/gesture system.

---

## 9. KSL Adaptation

ASL and KSL are structurally distinct languages — different phonemic handshape inventories,
different movement patterns and use of signing space, different non-manual grammatical
markers, and almost entirely different lexical items. **Direct deployment of this model for
KSL recognition is not appropriate** (`LIMITATIONS.md` L18), and — per L15 above — it will not
even fail visibly: a KSL sign will simply be forced into whichever of the 35 ASL classes the
model finds nearest, with no signal that the input doesn't belong to the trained vocabulary.

### Why hands-only is a reasonable transfer starting point
The champion's hands-only configuration already discards signer-specific body-position
information (arm length, filming distance, posture) that a full-landmark model would have
additionally learned to rely on as ASL-specific spatial grammar. This removes one entire axis
of cross-linguistic, signer-specific domain shift before KSL transfer even begins — a
secondary, welcome side-effect of a decision that was originally made purely for WLASL-35
accuracy reasons (`LIMITATIONS.md` Section 2, "Decisions Superseded").

### Recommended protocol (not yet executed)
1. **KSL from scratch (baseline)** — establishes the achievable ceiling with the target
   architecture on KSL data alone.
2. **Frozen ASL-pretrained BiLSTM + new `Dense(n_ksl_classes)` head** — tests whether ASL
   temporal motion patterns transfer at all.
3. **Full fine-tune: ASL initialisation → all layers trainable on KSL** — expected best
   performer once sufficient KSL data exists.

Evaluate all three on a genuinely held-out, signer-independent KSL test set using per-class
recall, not aggregate accuracy — the same per-class data-scarcity risk that produced `think`'s
failure (`L2`) is likely to recur at KSL scale, plausibly worse across 500 target classes.

### Data requirements
At ~40 clips/sign (current AI4KSL dataset), the target of 85% accuracy on 500 KSL signs is not
achievable, by the same clips-per-class-vs-F1 relationship this project's own ablation
established. Minimum recommended: **100–200 clips/sign**, with multi-signer, multi-region
diversity. Data collection, not modelling, is the highest-priority investment for KSL
deployment — directly consistent with this project's own finding that architecture search
converges quickly relative to the cost of data scarcity.

### Architecture scaling
500 signs requires materially more capacity than 35. Recommended starting points for a KSL
ablation:
- `hidden_units ∈ {128, 256}` (current champion: 64)
- A **fresh** `seq_len` ablation specific to KSL sign-duration distribution — the WLASL-derived
  finding that `seq_len=100` and `hands_only` are optimal **should not be assumed to transfer**
  to KSL phonology without independent re-verification.
- Retain `hands_only` as the default starting configuration given the transfer rationale above,
  but confirm it empirically on KSL data rather than inheriting it unverified.

**Estimated timeline:** 3–6 months for a 500-sign KSL system at the 85% target, dominated by
data collection rather than model development.

---

## 10. How to Use

### Architectural guarantee
`GesturePredictor` (`src/inference/predictor.py`) is the **sole, unified inference entry
point** for this model. Every consumer — the Stage 9 webcam demo, Stage 8's TFLite
verification, any future Android wrapper — is required to go through this class rather than
calling the TFLite interpreter or Keras model directly. This guarantees training/inference
preprocessing consistency by construction (one `FeaturePipeline` instance, one config source)
and eliminates an entire class of deployment bug (preprocessing drift between training and
serving) that this project treats as unacceptable, not merely undesirable.

### Recommended construction
Always construct via `from_config_snapshot()`, which loads the champion's *actual* trained
configuration (including `data.landmark_config: hands_only`, applied as a Stage 5 runtime
override and recoverable only from this snapshot file) rather than reconstructing it from
CLI-style defaults, which would silently produce the wrong preprocessing:

```python
from src.inference.predictor import GesturePredictor

predictor = GesturePredictor.from_config_snapshot(
    config_snapshot_path=(
        "artifacts/experiments/bilstm_hands_only_v4_aug/config_snapshot.yaml"
    ),
    model_path="models/gesture_bilstm_v1.tflite",
    smoother_window=5,          # 5-frame majority vote, ≈167ms at 30 FPS
    display_threshold=0.35,     # calibrated to the model's documented underconfidence
)
predictor.warmup(n_passes=3)    # eliminates first-inference latency spike (can exceed 700ms uncached)
```

### Streaming inference (webcam / live video)
```python
with predictor:  # context manager releases MediaPipe resources on exit
    while True:
        ret, frame = cap.read()
        result = predictor.predict_from_webcam_frame(frame)
        if result is None:
            continue  # rolling buffer still filling, or just auto-reset
        if result["is_confident"]:
            print(result["sign"], result["confidence"], result["top_k"])
```

### Offline / single-clip inference
```python
prediction = predictor.predict_from_video("path/to/clip.mp4")
# or, with landmarks already extracted as a (T, 225) array:
prediction = predictor.predict_from_landmarks(landmarks_225, update_smoother=False)
```

For a complete, production-grade reference implementation — including HUD rendering,
calibration-aware confidence display, confusable-pair badges, camera-failure recovery, and
session-summary logging — see `src/demo/webcam_demo.py` (Stage 9).

---

## 11. Provenance and Versioning

This is the only released version of `gesture_bilstm_v1`. It corresponds to MLflow run
`cb16f689d2294001a2ff2d3e02419d27` under the `"WLASL-35-class"` experiment, config hash
`5809193d37e0d480e409b8e3112e70c8de9008497a29727b411a7128e73287a6`, and was exported and
release-gated by `src/export/convert.py` / `src/export/verify.py` (Stage 8). Any future retrain
or re-export that changes the config hash should be issued as `gesture_bilstm_v2` with its own
model card and `LIMITATIONS.md` delta, rather than overwriting this one — the config-hash-based
identity check in `convert.py`'s `_verify_champion_model()` exists precisely to make this kind
of silent substitution structurally difficult.

### Citation
If you use this model or dataset in research, please cite the WLASL dataset:
> Li, D., Rodriguez, C., Yu, X., & Li, H. (2020). Word-level deep sign language
> recognition from video: A new large-scale dataset and methods comparison.
> *WACV 2020*.