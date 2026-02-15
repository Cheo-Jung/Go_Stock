# Predictor Accuracy Re-Examination (Training -> Prediction)

## Verdict
The predictor is **not reliably accurate yet** in this constrained runtime environment.

- Directional accuracy in prior repeated end-to-end checks has been observed around **0.34-0.36 on test windows** (below coin-toss).
- Training-directional accuracy has been materially higher (around **0.55-0.56**), indicating likely overfitting / weak generalization.
- Network/API constraints force fallback synthetic data frequently, so measured "accuracy" often does not reflect true live-market forecasting quality.

## Why this happens
1. Frequent fallback to synthetic price data when market/news APIs fail.
2. Sparse/empty news and factor feeds in restricted network environments.
3. Confidence is agreement-based and can stay high even for wrong direction.
4. Heavy feature expansion increases overfitting risk when true signal quality is low.

## Improvement plan
1. Run walk-forward evaluation only on confirmed live data windows (no fallback).
2. Add strict model acceptance gates (e.g., directional accuracy floor over multiple non-overlapping windows).
3. Reduce feature set using stability selection / ablation to remove noisy factors.
4. Penalize overconfidence with calibration (isotonic/Platt or temperature scaling).
5. Add regime-aware model selection (trend/chop/high-vol buckets).
6. Introduce transaction-cost-aware objectives and direction-specific loss balancing.
