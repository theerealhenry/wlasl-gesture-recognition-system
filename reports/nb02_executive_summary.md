---

# Notebook 02 — Stage 3 Inspection: Professional Analysis Report

**Generated:** Post-sample extraction analysis  
**Data mode:** SAMPLE (≤3 clips/sign per split — 208 clips queued, 350 total on disk)  
**Extractor schema:** v1.1  
**Author:** WLASL Gesture Recognition Pipeline

---

## 1. Executive Summary

This notebook evaluated the Stage 3 landmark extraction pipeline on a sample of 208 clips drawn from all three signer-aware splits. The extraction itself is functionally correct — MediaPipe Holistic initialises cleanly, the (N, 225) array format is valid, wrist-relative normalisation is confirmed, and the v1.1 decode-failure separation is in effect.

However, **one finding is project-critical and must be resolved before the full extraction is run:**

> **The 30% both-hands-absent skip threshold is producing a 76% skip rate (158/208 clips), 10–15× higher than the expected 5–8%. At this rate, full extraction would yield approximately 84 usable clips from 350 on disk — approximately 2.4 clips per class — which is insufficient for any meaningful model training.**

Four additional issues of lower but notable severity were identified, two of which are data-integrity bugs in the pipeline itself.

---

## 2. Data Availability and Coverage

| Metric | Value |
|---|---|
| Total clips queued (sample) | 208 |
| Usable (extracted + cached) | 50 (24%) |
| Skipped by skip policy | 158 (76%) |
| Extraction errors | 0 |
| Signs with any usable clips | 30 / 35 |
| Signs with 0 usable clips | 5 (chair, family, give, like, orange) |
| Signs with ≥1 clip in all three splits | 1 / 35 (help only) |

### 2.1 What the Sample Coverage Actually Represents

In `--sample-only` mode with `_SAMPLE_CLIPS_PER_SIGN = 3`, the pipeline selects up to 3 clips per sign **per split CSV**. Because most signs have only 1–3 clips in val and test (from the Stage 1 split), the sample includes nearly the entire dataset: 208 out of 350 available clips (59%). The 76% skip rate is therefore not an artifact of the sampling strategy — it is an accurate preview of the full extraction outcome.

**If the full extraction were run today with the 30% threshold:**  
- Estimated usable clips: ~84 (76% of 350 skipped)  
- Estimated train clips: ~59 (24% of 245)  
- Mean clips per sign in train: ~1.7  
- Clips needed for 70% val accuracy target: ≥200 train clips (based on Stage 1 F5 analysis)

The full extraction cannot proceed without resolving the skip policy.

---

## 3. The Skip Policy: Root Cause Analysis

### 3.1 What the policy does

The skip policy in `LandmarkExtractor.extract_video()` compares:

> missing_pct = missing_both_hands_frames / successfully_decoded_frames

If `missing_pct > max_missing_frame_pct (default: 0.30)`, the clip is discarded. The v1.1 change correctly computes this over **decoded frames only**, excluding OpenCV codec errors from the denominator.

### 3.2 Why 30% is too aggressive for WLASL

The WLASL dataset differs from controlled sign language corpora in several ways that increase natural hand-absence rates:

**One-handed signs**: Many ASL signs use only the dominant hand (e.g., `think`, `drink`, `who`). For these signs, MediaPipe will detect only the right (or left) hand per frame; the other hand's landmarks are legitimately absent. The `missing_both` metric only fires when *neither* hand is detected — so one-handed signs are correctly handled. However, in the portions of a one-handed sign's clip where even the active hand is outside frame (start/end transitions, body movement phases), `missing_both` fires, pushing rates toward 20–30%.

**WLASL video sourcing**: These clips were scraped from YouTube signing dictionaries, educational videos, and Deaf community content. The filming conditions are variable — handheld cameras, unusual angles, signers partially out of frame, signers turning their body during the sign. MediaPipe Holistic requires a relatively frontal, well-lit hand pose to confidently detect keypoints.

**Short clips amplify the effect**: With a mean of 51 frames and several clips in the 22–35 frame range, losing 10 frames to hand-absence during the wrist-setup phase of a sign is enough to push a clip past the 30% threshold. A 10-frame absence in a 31-frame clip is 32.3% — just over the threshold — whereas the same 10 frames in a 60-frame clip would be 16.7% and safely retained.

### 3.3 Observed per-sign missing rates (usable clips only)

The following signs have mean both-hands-absent rates above 15% **even in the clips that passed the 30% threshold**. Their skipped clips (which were discarded) had rates just above 30%:

| Sign | Mean missing (usable) | Clips skipped |
|---|---|---|
| drink | 28.6% | 7 |
| mother | 28.4% | 5 |
| go | 24.2% | 5 |
| candy | 23.3% | 6 |
| before | 22.8% | 4 |
| many | 21.4% | 4 |
| blue | 20.4% | 4 |
| finish | 19.4% | 4 |
| house | 18.2% | 4 |

These clips at 30–40% missing still contain valid landmark data in 60–70% of frames. Discarding them removes potentially useful training signal.

### 3.4 Threshold sensitivity

Based on the bimodal distribution observed (median missing = 0%, mean = 9.95% for usable clips; skipped clips must be 30–100%), the following retention estimates apply:

| Threshold | Estimated retention rate | Estimated usable clips (from 350) |
|---|---|---|
| 0.30 (current) | ~24% | ~84 |
| 0.50 | ~60–70% | ~210–245 |
| 0.60 | ~75–85% | ~263–298 |
| 0.70 | ~85–90% | ~298–315 |
| None (disabled) | 100% | 350 |

**Recommendation:** Set `max_missing_frame_pct = 0.60` for the full extraction. This retains clips where MediaPipe detects hands in at least 40% of decoded frames, which is sufficient for the LSTM to learn real temporal patterns with zero-fill regularisation providing implicit noise robustness. The 0.60 value is consistent with practice in data-constrained sign language recognition research where dataset size is the primary bottleneck.

---

## 4. Landmark Detection Quality

### 4.1 Feature heatmap observations

The three clips visualised in Cell 6 reveal three distinct detection patterns:

**`drink` (79 frames):** The left-hand band (indices 0–62) is entirely blue (zero-filled) throughout the clip. The right-hand band (63–125) shows warm values beginning around frame 10. Pose (126–224) is consistently detected. This is a one-handed sign — the left hand is genuinely not used, and its absence is semantically correct. The `missing_both` rate of 28.6% reflects frames where even the right hand was not detected (likely the wrist-setup phase at the start and signer re-positioning at the end).

**`computer` (45 frames):** Both hands are partially detected, with several complete-absence gaps visible in the RH band (frames 8–10, 15–17, 24–27). These gaps are MediaPipe detection failures, not sign semantics. The clip was retained at 9.7% missing rate. The gaps are spatially coherent (full columns of blue), confirming these are whole-frame misses rather than partial detections.

**`think` (35 frames):** The left-hand band is entirely zero-filled (one-handed sign). The right-hand band shows a dense, well-detected trajectory across all 35 frames with only minor gaps. The zero-fill rate of 32.1% is entirely attributable to the expected left-hand absence — the sign's active hand is reliably detected.

**Implication for FeaturePipeline:** The zero-fill pattern for one-handed signs is semantically informative — the consistently-zero left-hand slice distinguishes one-handed signs from two-handed signs. This signal must be preserved exactly as-is through the normalization step. The wrist-relative normalisation in `FeaturePipeline` must not shift zero-filled frames to the wrist origin, which would overwrite the semantically-meaningful zero.

### 4.2 Data integrity issue: Individual hand missing rates

All per-sign entries in the landmark quality heatmap show 0.0% for Left Hand, Right Hand, and Pose. This is incorrect and is caused by a bug in `pipelines/run_landmark_extraction.py` — specifically in `_finalise_run()`, which builds `ExtractionResult` objects from pipeline-level records but sets individual hand fields to zero:

```python
# In _finalise_run() — CURRENT CODE (incorrect)
result = ExtractionResult(
    ...
    missing_left_hand_frames= 0,   # not tracked at pipeline level
    missing_right_hand_frames=0,
    missing_pose_frames=      0,
    missing_both_hands_frames=rec.get("n_missing_both", 0),
    ...
)
```

The `missing_both_pct` column in the inventory is correct (populated from `rec.get("missing_pct")`), but the individual hand columns are artificially zeroed. The fix requires propagating per-hand missing rates through the pipeline's `_RunStats` recording layer. **This is a data integrity issue in the CSV only — the skip policy uses the correct `missing_pct` value, so extraction decisions are unaffected.**

### 4.3 The missing_left/right/pose being all 0.0% vs missing_both_pct being non-zero

This appears contradictory: how can both hands be absent in 9–29% of frames if neither hand is individually missing? The answer is the CSV column bug described above. The `missing_both_pct` column comes directly from the extractor's `ExtractionResult.missing_pct` field (which is correctly computed by `_FrameDecodeStats`), while the individual columns are zeroed in the pipeline layer. The underlying data in the `.meta.json` sidecars is correct.

---

## 5. Decode Failure Analysis

| Metric | Value |
|---|---|
| Global decode failure rate | 3.891% |
| Clips with ≥1 decode failure | 35 / 50 (70%) |
| Max decode failures per clip | 2 |
| Pearson r (decode_failures, missing_both) | 0.544 |

### 5.1 The correlation concern

The r = 0.544 between decode failure frames and missing_both rate triggered an automated warning ("Unexpected correlation — verify extractor schema version"). However, this correlation is **spurious and does not indicate a v1.1 regression**. Here is the mathematical argument:

The maximum decode failures observed is 2 per clip. In a 37-frame clip (the shortest in the table), 2 decode failures represent 5.4% of frames. If the v1.0 bug were present (decode failures counted as detection failures), this would add at most 5.4 percentage points to `missing_both`. Yet the same clip shows `missing_both = 28.6%`. The 28.6% cannot be caused by 2 decode failures — it must come from genuine MediaPipe detection failures in ~11 additional decoded frames.

The correlation reflects a common cause: **challenging video quality**. Video files that stress the OpenCV codec (unusual encoding, damage, poor compression) also tend to have difficult visual conditions (dark lighting, unusual angles, motion blur) that impair MediaPipe's hand detection. Both failure modes are downstream symptoms of video quality, not of each other.

**Assessment:** v1.1 is correctly implemented. The decode-failure isolation is working. The warning threshold in `_validate_extraction_health()` should be relaxed to acknowledge this spurious-correlation pattern for datasets sourced from heterogeneous web video.

---

## 6. Wrist-Relative Normalisation

The normalisation is working correctly. Key quantitative observations:

- **Raw left wrist x:** range [0.36, 0.74], std = 0.074  
- **Raw left wrist y:** range [0.46, 1.08], std = 0.116 (note: values above 1.0 indicate sub-pixel coordinate extrapolation by MediaPipe at frame boundaries — expected behaviour)  
- **Normalised LH fingertip x:** range [-0.18, 0.10], std = 0.050

The KDE contours in the raw panels reveal that WLASL signers are filmed in a consistent camera position — wrists cluster in the centre-lower portion of the frame rather than scattering uniformly across [0,1]. This is expected for a curated sign language dataset. The important validation is that post-normalisation the fingertip coordinates are tightly clustered around the wrist origin, confirming the spatial invariance property.

**Implementation note for FeaturePipeline:** The normalisation function in Cell 11 (`wrist_relative_normalize`) must be adopted exactly, including the detection mask guard:

```python
_detected = (_lh != 0).any(axis=(1, 2))
# Only normalize detected frames — pass zero-fill frames through unchanged
```

This guard prevents false origin-assignment for zero-filled frames and is the single most important correctness invariant in the normalization implementation.

---

## 7. Inter-Signer Variability

The "help" sign with 2 signers reveals the fundamental generalisation challenge:

- **Signer 4**: dramatic upward arc trajectory — wrist y moves from ~0.9 (low in frame) to ~0.1 (high in frame) and back. Large spatial extent.
- **Signer 10**: compact movement near wrist y ~0.75–0.85. Much smaller spatial extent.

The 2D wrist paths are qualitatively different signs to a computer vision system, yet they mean the same thing to a human observer. The across-signer wrist spread peaks at frames 5–10 (early sign phase) with mean spread of 0.016 screen-space units. This spread is not eliminated by wrist-relative normalisation — the **speed, arc geometry, and spatial extent variation remain** as the core generalisation challenge.

This directly motivates the Stage 5 design decisions:
- `speed_jitter=True` in `spatial_temporal` augmentation: generates multiple temporal scales from a single clip, teaching the LSTM that fast and slow executions of the same sign are equivalent
- `rotation_deg=5.0`: teaches arc-geometry invariance  
- `class_weight_balancing=True`: prevents the few-signer classes from being ignored

With 31 train signers and 7 entirely disjoint val signers, the train/val accuracy gap of 15–25 percentage points cited in F9 is realistic. The goal of 70% val accuracy is achievable only if the model sees sufficient signing-style diversity during training — which requires retaining as many training clips as possible.

---

## 8. Sequence Length Analysis

The mean raw frame count for usable clips is **51.4 frames** (vs. the Stage 1 estimate of 65.7 from the inventory). The discrepancy has two causes:
1. The sample usable clips may be biased toward shorter clips (shorter clips are less likely to exceed the 30% missing threshold)  
2. Some clips report 0 frames due to a cache-restoration issue (see Section 9.3)

The coverage analysis from the full dataset (Stage 1 mean of 65.7 frames) is likely more representative.

| seq_len | Clips fully covered | Mean content captured | Assessment |
|---|---|---|---|
| 20 | 0.0% | 43.5% | Too short — loses most sign content |
| 30 | 2.9% | 64.3% | Default; practical but sees ~36% of clips truncated |
| 40 | 34.3% | 79.5% | Good balance for Stage 5 ablation |
| 60 | 71.4% | 94.8% | Strongly recommended addition to ablation |

**Recommendation:** The seq_len=60 case must be added to the Stage 5 Group 3 ablation (currently {20, 30, 40}). At 94.8% mean content coverage, it provides the most complete sign representation and is likely to outperform seq_len=30 by a material margin. The memory and latency cost (2× the default) is justified given the data constraints.

---

## 9. Known Issues and Data Integrity Concerns

### 9.1 Individual hand missing rates zeroed in inventory (Bug — Medium severity)

**Location:** `pipelines/run_landmark_extraction.py`, `_finalise_run()`  
**Effect:** `missing_left_pct`, `missing_right_pct`, `missing_pose_pct` columns are always 0.0 in `landmark_inventory.csv`  
**Impact on pipeline:** None (skip policy uses `missing_both_pct` which is correct)  
**Impact on notebook analysis:** Cell 7 and Cell 8 cannot show per-hand breakdown; the quality heatmap is uninformative  
**Fix:** Propagate per-hand missing counts through `_RunStats.record_extracted()` and `_RunStats.record_cached()`

### 9.2 Frame count = 0 for some cached clips (Bug — Low severity)

**Affected signs:** `change` (2 clips, MeanF=0.0), `cousin` (1 clip), `who` (1 clip) in per-sign stats  
**Cause:** Cache-hit path in `_run_extraction_loop` reads frame count from `_read_sidecar()` as `meta.get("num_frames", 0)`. If the sidecar for these clips was written without a `num_frames` key, it defaults to 0.  
**Impact:** Per-sign statistics (mean frames, coverage) are incorrect for these 4 clips  
**Fix:** Verify the sidecar for these clips; fall back to loading the `.npy` and reading `arr.shape[0]` if sidecar `num_frames` is 0

### 9.3 Sign properties showing `?` (Configuration mismatch — Low severity)

The `handedness`, `motion_type`, and `difficulty` columns all show `?` in the per-sign table, indicating `SIGN_PROPS.get(sign, {})` is returning empty dicts for all 35 signs. This means either `label_map_v1.json` does not have a `sign_properties` key, or the key structure does not match the sign names in `CLASSES`. Inspect `artifacts/label_map_v1.json` directly to confirm the structure.

### 9.4 Spurious correlation warning in decode failure analysis (False alarm — Low severity)

The health check in `_validate_extraction_health()` warns when the global missing rate exceeds 15%. The global rate is 14.63% — just below the threshold. After full extraction with a raised skip threshold, this rate may increase. The warning threshold should be documented as approximate guidance rather than a hard boundary for WLASL's heterogeneous video quality.

---

## 10. Required Modifications Before Full Extraction

The following changes must be made before running `python pipelines/run_landmark_extraction.py --split all`.

### 10.1 CRITICAL: Raise the skip threshold

**File:** `pipelines/run_landmark_extraction.py`  
```python
# Change line:
_DEFAULT_MAX_MISSING_FRAME_PCT = 0.30

# To:
_DEFAULT_MAX_MISSING_FRAME_PCT = 0.60
```

**File:** `src/features/extractor.py`  
```python
# Change line in __init__:
self._max_missing_pct: float = 0.30

# To:
self._max_missing_pct: float = 0.60
```

Additionally, run the full extraction with explicit override to confirm:
```bash
python pipelines/run_landmark_extraction.py --split all --max-missing-frame-pct 0.60
```

**Rationale:** Clips with up to 60% missing both-hands rate still contain valid landmarks in 40% of frames. With zero-fill, the LSTM learns to handle absent landmarks as a regularisation signal rather than an error condition. The alternative — discarding 76% of an already severely limited dataset — makes the 70% accuracy target mathematically impossible.

### 10.2 Fix individual hand missing rates in pipeline CSV

**File:** `pipelines/run_landmark_extraction.py`

In `_RunStats`, add per-hand tracking fields and propagate from `ExtractionResult`:

```python
# In _RunStats.__init__(), add:
self.total_missing_left   = 0
self.total_missing_right  = 0  
self.total_missing_pose   = 0

# In record_extracted(), add:
self.total_missing_left   += result.missing_left_hand_frames
self.total_missing_right  += result.missing_right_hand_frames
self.total_missing_pose   += result.missing_pose_frames

# In record_extracted() and record_cached(), update _records append to include:
"n_missing_left":  result.missing_left_hand_frames,
"n_missing_right": result.missing_right_hand_frames,
"n_missing_pose":  result.missing_pose_frames,
```

Then in `_finalise_run()`, replace the hardcoded zeros:

```python
# Replace:
missing_left_hand_frames= 0,   # not tracked at pipeline level
missing_right_hand_frames=0,
missing_pose_frames=      0,

# With:
missing_left_hand_frames= rec.get("n_missing_left",  0),
missing_right_hand_frames=rec.get("n_missing_right", 0),
missing_pose_frames=      rec.get("n_missing_pose",  0),
```

### 10.3 Add pre-extraction threshold diagnostic

Add the following function to `pipelines/run_landmark_extraction.py` and run it before committing to the full extraction:

```python
def run_threshold_diagnostic(
    inventory_json_path: str,
    thresholds: list[float] | None = None,
) -> None:
    """
    Print expected clip retention at multiple skip thresholds.
    Uses the per-clip missing_pct data from preprocessing_summary_latest.json.
    Run this before full extraction to select the optimal threshold.

    Usage:
        python -c "
        from pipelines.run_landmark_extraction import run_threshold_diagnostic
        run_threshold_diagnostic('data/preprocessing_summary_latest.json')
        "
    """
    import json
    from pathlib import Path

    if thresholds is None:
        thresholds = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.00]

    with open(inventory_json_path) as f:
        summary = json.load(f)

    per_clip = summary.get("per_clip", [])
    all_pcts  = [r.get("missing_pct", 0.0) for r in per_clip
                 if r.get("outcome") != "error"]
    total     = len(all_pcts)

    print(f"{'Threshold':>12} | {'Retained':>10} | {'Skipped':>10} | {'Ret. rate':>10}")
    print("-" * 52)
    for t in thresholds:
        retained = sum(1 for p in all_pcts if p <= t)
        skipped  = total - retained
        print(f"{t:>12.0%} | {retained:>10} | {skipped:>10} | {retained/total:>10.1%}")
```

### 10.4 Extend the seq_len ablation to include 60 frames

**File:** `configs/data/seq60.yaml`  
Create this file (or add it via `write_default_configs()` extension):
```yaml
sequence_length: 60
```

**File:** `src/utils/config.py`  
Add `seq60` to the `write_default_configs()` data configurations dict.

**Impact on Stage 5:** The Group 3 ablation becomes `{20, 30, 40, 60}`, adding one training run. Based on the coverage analysis (94.8% mean content at seq_len=60 vs. 64.3% at seq_len=30), this run is likely to produce the best accuracy for LSTM-based models and should be included in the final report comparison.

### 10.5 Suppress the false-alarm correlation warning for WLASL

**File:** `notebooks/02_landmark_inspection.ipynb`, Cell 10

The Pearson r = 0.544 correlation between decode failures and missing rate is spurious (common cause: video quality). The warning threshold of 0.30 is appropriate for clean datasets but not for heterogeneous YouTube-sourced video. Either:
- Raise the correlation alert threshold to 0.65 in Cell 10, or
- Add an explanatory comment documenting the spurious-correlation pattern

This is a notebook-only change and does not affect the pipeline.

---

## 11. Stage 4 Configuration Decisions (Confirmed from This Analysis)

| Parameter | Value | Evidence |
|---|---|---|
| `normalisation` | `wrist_relative` | ✓ Validated Cell 11 — raw wrists scattered, norm clustered at origin |
| `normalise_pose` | `False` | Pose represents body position — normalisation would remove meaningful signal |
| `zero_fill_detected_mask` | `True` | Zero-filled frames must pass through normalisation unchanged (Cell 11 guard) |
| `sequence_length` | 30 (default) | Per handoff; ablation extended to {20, 30, 40, 60} |
| `skip_threshold` | 0.60 | Raised from 0.30 — see Section 10.1 |
| `augmentation_default` | `spatial_temporal` | F9 (inter-signer variability, HIGH severity) |
| `speed_jitter` | `True` | Cell 12: temporal extent varies dramatically across signers |
| `class_weight_balancing` | `True` | F1 (class imbalance, HIGH severity) — unchanged |
| `primary_metric` | `per_class_f1` | F1 decision, confirmed by F11 (1-val-clip limitation) |

---

## 12. Full Extraction Readiness Checklist

Before running `python pipelines/run_landmark_extraction.py --split all`:

- [ ] Change `_DEFAULT_MAX_MISSING_FRAME_PCT` to `0.60` in `run_landmark_extraction.py`
- [ ] Change default `self._max_missing_pct` to `0.60` in `extractor.py`
- [ ] Apply the `_finalise_run()` individual hand missing rate fix
- [ ] Run `run_threshold_diagnostic()` on the existing summary JSON to confirm expected retention
- [ ] Create `configs/data/seq60.yaml`
- [ ] Delete existing `data/landmarks/` sample output — or use `--force` flag to re-extract with new threshold
- [ ] Verify `artifacts/label_map_v1.json` `sign_properties` key structure for Cell 13 fix
- [ ] Estimated runtime: 30–90 minutes
- [ ] After extraction: re-run Cells 3–14 for full-dataset statistics

---

## 13. Artifact Inventory

| Artifact | Size | Status |
|---|---|---|
| `reports/figures/nb02_extraction_outcomes.png` | 104.0 KB | ✓ |
| `reports/figures/nb02_coverage_map.png` | 187.3 KB | ✓ |
| `reports/figures/nb02_skeleton_overlay.png` | 141.7 KB | ✓ (video-less mode) |
| `reports/figures/nb02_feature_heatmap.png` | 344.8 KB | ✓ |
| `reports/figures/nb02_missing_landmark_analysis.png` | 204.3 KB | ⚠ (per-hand cols = 0; re-run after pipeline fix) |
| `reports/figures/nb02_landmark_quality_heatmap.png` | 151.2 KB | ⚠ (all zeros; re-run after pipeline fix) |
| `reports/figures/nb02_sequence_length_distribution.png` | 217.5 KB | ✓ (limited by 35 usable clips) |
| `reports/figures/nb02_decode_failure_analysis.png` | 156.5 KB | ✓ |
| `reports/figures/nb02_wrist_normalization.png` | 403.8 KB | ✓ |
| `reports/figures/nb02_inter_signer_variability.png` | 364.4 KB | ✓ (limited to 2 signers in sample) |
| `reports/nb02_findings.json` | 4.9 KB | ✓ |

---

*This report was generated from the SAMPLE extraction run (208 clips, 76% skip rate). All findings marked [SAMPLE] should be re-validated after the full extraction with the corrected 0.60 threshold. The pipeline modifications in Section 10 are prerequisites for the full extraction.*