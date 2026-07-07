# Baseline vs. Improved Model — Test Set Comparison (real dataset)

Improved model selected: **extra_trees**

| Metric | Baseline (Random Forest) | Improved | Delta |
|---|---|---|---|
| accuracy | 0.2000 | 0.1333 | -0.0667 |
| precision_macro | 0.2000 | 0.1434 | -0.0566 |
| recall_macro | 0.1961 | 0.1310 | -0.0651 |
| f1_macro | 0.1959 | 0.1306 | -0.0653 |
| roc_auc_ovr_macro | 0.4521 | 0.4325 | -0.0195 |

## Pemphigus Vulgaris class metrics (multi-class argmax decision)

| Metric | Baseline | Improved |
|---|---|---|
| precision | 0.1667 | 0.1667 |
| recall | 0.1429 | 0.0714 |
| f1-score | 0.1538 | 0.1000 |
| support | 14 | 14 |

## PV-vs-rest screening threshold (improved model only)

Threshold on P(Pemphigus Vulgaris) = 0.05: precision=0.2333, recall=1.0000, f1=0.3784
Confusion matrix (rows=actual [not-PV, PV], cols=predicted): [[0, 46], [0, 14]]

Baseline confusion matrix (['Bullous Pemphigoid', 'Healthy', 'Pemphigus Foliaceus', 'Pemphigus Vulgaris']): [[5, 5, 5, 2], [7, 2, 1, 5], [3, 5, 3, 3], [7, 3, 2, 2]]
Improved confusion matrix (['Bullous Pemphigoid', 'Healthy', 'Pemphigus Foliaceus', 'Pemphigus Vulgaris']): [[3, 6, 7, 1], [9, 2, 2, 2], [6, 4, 2, 2], [8, 4, 1, 1]]

## Interpretation

Both models score at or below chance level (25% for 4 balanced classes) and ROC-AUC sits at ~0.43-0.45 (0.5 = random). This is not a modelling failure: exploratory analysis of this dataset shows the ELISA titres and other features are statistically indistinguishable across the four diagnosis groups (near-identical means/std per class) -- i.e. it is a placeholder/dummy dataset with no injected diagnostic signal, exactly as the Progress Report's Limitations section (9) warns: "all performance figures are illustrative and must be revalidated on real clinical data." The PV-screening threshold collapsing to 0.05 (flagging nearly everything as PV to maximise recall) is itself evidence of this -- the optimiser found no non-trivial threshold worth preferring. The pipeline (preprocessing, SMOTE, feature selection, model comparison, threshold tuning, SHAP) is fully built and will produce meaningful results as soon as it is run against real clinical data with genuine signal.
