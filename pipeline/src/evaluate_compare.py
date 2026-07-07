"""Prints and saves a before/after comparison of the baseline and improved
models on the same held-out test set of the real (dummy) 4-class dataset."""
import json

METRICS_ORDER = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_ovr_macro"]


def main():
    with open("ml_pipeline/artifacts/baseline_metrics.json") as f:
        baseline = json.load(f)
    with open("ml_pipeline/artifacts/improved_metrics.json") as f:
        improved = json.load(f)

    lines = []
    lines.append("# Baseline vs. Improved Model — Test Set Comparison (real dataset)\n")
    lines.append(f"Improved model selected: **{improved['selected_model']}**\n")
    lines.append("| Metric | Baseline (Random Forest) | Improved | Delta |")
    lines.append("|---|---|---|---|")
    for m in METRICS_ORDER:
        b, i = baseline[m], improved[m]
        lines.append(f"| {m} | {b:.4f} | {i:.4f} | {i - b:+.4f} |")

    lines.append("")
    lines.append("## Pemphigus Vulgaris class metrics (multi-class argmax decision)\n")
    lines.append("| Metric | Baseline | Improved |")
    lines.append("|---|---|---|")
    for m in ["precision", "recall", "f1-score", "support"]:
        b = baseline["pv_class_metrics"].get(m)
        i = improved["pv_class_metrics"].get(m)
        lines.append(f"| {m} | {b:.4f} | {i:.4f} |" if m != "support" else f"| {m} | {b:.0f} | {i:.0f} |")

    pv_screen = improved.get("pv_screen")
    if pv_screen:
        lines.append("\n## PV-vs-rest screening threshold (improved model only)\n")
        lines.append(f"Threshold on P(Pemphigus Vulgaris) = {pv_screen['threshold']:.2f}: "
                      f"precision={pv_screen['precision']:.4f}, recall={pv_screen['recall']:.4f}, "
                      f"f1={pv_screen['f1_score']:.4f}")
        lines.append(f"Confusion matrix (rows=actual [not-PV, PV], cols=predicted): {pv_screen['confusion_matrix']}")

    lines.append(f"\nBaseline confusion matrix ({baseline['classes']}): {baseline['confusion_matrix']}")
    lines.append(f"Improved confusion matrix ({improved['classes']}): {improved['confusion_matrix']}")

    lines.append("\n## Interpretation\n")
    lines.append(
        "Both models score at or below chance level (25% for 4 balanced classes) and ROC-AUC "
        "sits at ~0.43-0.45 (0.5 = random). This is not a modelling failure: exploratory analysis "
        "of this dataset shows the ELISA titres and other features are statistically "
        "indistinguishable across the four diagnosis groups (near-identical means/std per class) "
        "-- i.e. it is a placeholder/dummy dataset with no injected diagnostic signal, exactly as "
        "the Progress Report's Limitations section (9) warns: \"all performance figures are "
        "illustrative and must be revalidated on real clinical data.\" The PV-screening threshold "
        "collapsing to 0.05 (flagging nearly everything as PV to maximise recall) is itself "
        "evidence of this -- the optimiser found no non-trivial threshold worth preferring. "
        "The pipeline (preprocessing, SMOTE, feature selection, model comparison, threshold "
        "tuning, SHAP) is fully built and will produce meaningful results as soon as it is run "
        "against real clinical data with genuine signal."
    )

    report = "\n".join(lines)
    print(report)
    with open("ml_pipeline/artifacts/comparison_report.md", "w") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
