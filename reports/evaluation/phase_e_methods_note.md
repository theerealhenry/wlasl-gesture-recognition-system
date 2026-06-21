## Stage 6, Phase E — Interpretability Methods Note

**Attempted method:** `shap.DeepExplainer` (5-clip, 1-class smoke test).
**Result:** FAILED — fell back to Gradient × Input.
**Failure reason:** shap.DeepExplainer is incompatible with this architecture (Masking + stacked Bidirectional(LSTM) under TF 2.13.1). Additionally, shap.DeepExplainer monkey-patches TensorFlow's gradient registry (e.g. renaming 'Transpose' -> 'shap_Transpose') and does not restore it, which breaks all subsequent tf.GradientTape() calls in the same process. Confirmed via an isolated smoke-test kernel run. Per the Phase E hard-stop policy, DeepExplainer is not re-attempted in a session that also needs to run Gradient x Input.

**Method actually used for all four Phase E outputs:** Gradient × Input.

Gradient × Input computes, for input x and a target class's PRE-SOFTMAX logit
y_c, the elementwise product x * d(y_c)/dx via `tf.GradientTape()`. This is
explicitly NOT full Integrated Gradients (no baseline-interpolation path was
computed) and NOT SHAP (no Shapley-value game-theoretic attribution was
computed). It is the simplest member of the gradient-attribution family and
is reported as such throughout — no figure or table in this notebook implies
a more sophisticated method was used.

**Why SHAP failed (expected):** the champion architecture
(`bilstm_hands_only_v4_aug`) requires a `Masking(mask_value=0.0)` layer ahead
of a stacked `Bidirectional(LSTM)` block (Stage 5 Critical Rule: 35.28% of
frames are zero-filled and must be masked, not treated as signal).
`shap.DeepExplainer` under TF 2.13.1 does not reliably register
gradient ops through this Masking + Bidirectional combination. This is a
known, documented limitation class for DeepExplainer, not a project-specific
bug to be debugged further at this stage.

**Background set:** 100 non-augmented training clips, sampled with
as-even-as-possible per-class stratification (quota=2/class,
covering 35/35 classes).

**Explained set:** all 52 validation clips (test set untouched beyond
Phase C's single metric pass, per the test pre-commitment policy).

**Outputs produced:**
1. `shap_frame_importance.png` — mean |attribution| per timestep, ±1 SD band.
2. `shap_landmark_heatmap.png` — mean |attribution| per LH/RH MediaPipe
   hand landmark (no pose dimension; champion is hands_only).
3. `shap_per_class_summary.png` — curated 15-class subset
   (['birthday', 'book', 'clothes', 'name', 'think'] [high-risk] + ['cousin', 'girl', 'house', 'later', 'mother', 'orange', 'think', 'who'] [Phase D
   confusable pairs] + ['before', 'boy', 'candy'] [easy contrast]).
   `shap_per_class_summary_appendix_full35.png` — full 35-class grid.
4. `shap_misclassification_analysis.png` — attribution vs. true vs. predicted
   class for clips in Phase D's actual top-4 confusable pairs
   (think↔who, later↔house, cousin↔mother, girl↔orange) — NOT the handoff's
   stale predicted pairs, which Phase D confirmed did not materialise.
