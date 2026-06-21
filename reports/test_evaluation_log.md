# Test Set Evaluation Pre-Commitment Log

## Timestamp
2026-06-18T23:20:58.709546+00:00

## Statement of Finality
All methodology, champion model selection, and hyperparameters are final
as of this timestamp. No further training, hyperparameter tuning, or
champion re-selection will occur based on the test results that follow.

## Champion Model
- Run name: bilstm_hands_only_v4_aug
- MLflow run ID: cb16f689d2294001a2ff2d3e02419d27
- SavedModel path: models/bilstm_hands_only_v4_aug_saved_model/
- val_macro_f1: 0.6011 (52 val clips, 7 unseen signers)
- Parameters: 68,771
- Config hash: 5809193d37e0d480e409b8e3112e70c8de9008497a29727b411a7128e73287a6

## Methodology
- Inference pipeline: FeaturePipeline(training=False), landmark_config=hands_only,
  sequence_length=100. Identical preprocessing to the val inference in Phase B1.
- Evaluation: compute_evaluation_summary() from src/evaluation/metrics.py with
  n_classes=35, labels=list(range(35)), zero_division=0.
- Bootstrap CI: clip-level resampling, 1000 iterations, seed=42, 90% CI level.
  Note: resampling unit is clip, not signer — CI likely understates true
  uncertainty. See metrics.py bootstrap caveat.

## Known Config Discrepancy
The champion model's config_snapshot.yaml records
`early_stopping_monitor: val_accuracy`, not `val_macro_f1` as narrated in
the Stage 5 handoff. Manual early stopping in train.py monitored val_macro_f1
via sklearn (the Python patience counter). The Keras ReduceLROnPlateau callback
monitored val_accuracy. This discrepancy is informational; the champion model
weights were selected on val_macro_f1 by the manual patience loop.
This discrepancy is also recorded in evaluation_report.json.

## Expected Test macro-F1 Range
Per the Stage 5 handoff (Part 7, Stage 6 spec): expected 0.45–0.58.
Lower than val due to indirect val-set overfitting (champion epoch and
spatial_temporal augmentation decision were informed by val numbers).

## Test Results (filled in after inference)
<!-- DO NOT EDIT ABOVE THIS LINE AFTER TIMESTAMP -->

### Inference timestamp
2026-06-19 (Stage 6 Phase C execution)

### Test set composition
- n_samples: 51 clips
- n_signers: 7 unseen signers
- Split path: data/splits/test.csv

### Primary metric
- test_macro_f1: 0.4581
- 90% CI: [0.3935, 0.5076]
- CI width: 0.1141

### Secondary metric
- test_accuracy: 0.4902

### Gate check
- Expected range: [0.45, 0.58]
- Within expected range: YES (0.4581 is within [0.45, 0.58])
- val→test gap: 0.1430 (14.3pp)

### Bootstrap metadata
- n_bootstrap: 1000
- seed: 42
- ci_level: 0.90
- resampling_unit: clip (not signer — CI understates true uncertainty)

### Interpretation
The test macro-F1 of 0.4581 falls within the pre-committed expected range
of [0.45, 0.58], confirming the analytical integrity of Stage 5 predictions.
The val→test gap of 14.3pp is attributed to indirect val-set overfitting
(champion epoch budget and augmentation strategy were selected on val numbers),
small-N amplification (51 clips, ~2pp per prediction flip), and genuine
signer generalisation difficulty. No data leakage was detected — the
signer-aware split guarantees zero overlap across all three partitions.
No further tuning will be performed based on these results.