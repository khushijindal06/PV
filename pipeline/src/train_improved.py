"""Improved model: takes the baseline pipeline from train_baseline.py and
applies the enhancements the Progress Report flags as future work (section
8/9), adapted to the real 4-class dataset (Healthy / Pemphigus Vulgaris /
Pemphigus Foliaceus / Bullous Pemphigoid):

  * missing-value indicator columns (the report calls missingness
    "potentially informative")
  * SMOTE oversampling inside the CV loop to address class imbalance
    (report lists this as "evaluated for future use")
  * a comparison across several model families instead of committing to
    Random Forest alone, plus a soft-voting ensemble of the best two, with
    feature-count also treated as a tunable hyperparameter
  * a separate PV-vs-rest sensitivity screen: since the report's clinical
    concern is specifically about *missing PV cases* (not misclassifying
    among the other three classes equally), we additionally tune a
    decision threshold on P(Pemphigus Vulgaris) to flag high-suspicion PV
    cases the argmax multi-class decision would otherwise miss
  * SHAP-based feature importance for the Pemphigus Vulgaris class (report
    lists XAI integration as "Planned")

Trained/evaluated on the exact same train/test split as the baseline for a
fair before/after comparison.

Note: exploratory analysis of this dataset (see ml_pipeline/README.md)
shows the ELISA titres and other features are statistically indistinguishable
across the four diagnosis groups -- this is a placeholder/dummy dataset for
pipeline validation, not real clinical signal (the Progress Report's own
Limitations section flags this). So none of the modelling choices here can
manufacture predictive power that isn't in the data; what they demonstrably
do is make better use of whatever signal exists and produce a pipeline ready
to run on real clinical data once available.
"""
import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_pipeline.src.common import (
    BINARY_FEATURES,
    NOMINAL_FEATURES,
    NUMERIC_FEATURES,
    PV_LABEL,
    RANDOM_STATE,
    load_data,
    split_xy,
    train_test_split_fixed,
)

warnings.filterwarnings("ignore")

PV_SCREEN_FBETA_WEIGHT = 2.0  # weight recall above precision for the PV screening threshold


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
            ]), NUMERIC_FEATURES),
            ("bin", SimpleImputer(strategy="most_frequent"), BINARY_FEATURES),
            ("nom", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore")),
            ]), NOMINAL_FEATURES),
        ]
    )


CANDIDATES = {
    "random_forest": (
        RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        {
            "select__k": [6, 8, "all"],
            "clf__n_estimators": [100, 200, 300, 500],
            "clf__max_depth": [5, 10, 15, 20, None],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 2, 4],
            "clf__max_features": ["sqrt", "log2"],
        },
    ),
    "extra_trees": (
        ExtraTreesClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        {
            "select__k": [6, 8, "all"],
            "clf__n_estimators": [200, 300, 500],
            "clf__max_depth": [5, 10, 15, None],
            "clf__min_samples_split": [2, 5, 10],
            "clf__max_features": ["sqrt", "log2"],
        },
    ),
    "gradient_boosting": (
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {
            "select__k": [6, 8, "all"],
            "clf__n_estimators": [100, 200, 300],
            "clf__learning_rate": [0.01, 0.05, 0.1],
            "clf__max_depth": [2, 3, 4],
            "clf__subsample": [0.7, 0.85, 1.0],
        },
    ),
    "logistic_regression": (
        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE),
        {
            "select__k": [6, 8, "all"],
            "clf__C": [0.01, 0.1, 1.0, 10.0],
            "clf__penalty": ["l2"],
        },
    ),
}


def tune_candidate(name, estimator, param_grid, X_train, y_train):
    pipeline = ImbPipeline([
        ("preprocess", build_preprocessor()),
        ("select", SelectKBest(score_func=f_classif, k=8)),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", estimator),
    ])
    search = RandomizedSearchCV(
        pipeline, param_grid, n_iter=25, cv=5, scoring="f1_macro",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print(f"  {name}: best CV F1-macro = {search.best_score_:.4f}, params = {search.best_params_}")
    return search.best_estimator_, search.best_score_


def tune_pv_screen_threshold(estimator, X_train, y_train, pv_index, beta):
    probs = cross_val_predict(
        estimator, X_train, y_train, cv=5, method="predict_proba", n_jobs=-1
    )[:, pv_index]
    y_train_is_pv = (y_train == PV_LABEL).astype(int)
    thresholds = np.arange(0.05, 0.91, 0.01)
    scores = [fbeta_score(y_train_is_pv, probs >= t, beta=beta) for t in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def main():
    df = load_data()
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split_fixed(X, y)

    print("=== Tuning candidate models (5-fold CV, F1-macro, SMOTE inside each fold) ===")
    results = {}
    for name, (estimator, grid) in CANDIDATES.items():
        results[name] = tune_candidate(name, estimator, grid, X_train, y_train)

    # Soft-voting ensemble of the two strongest individual models.
    ranked = sorted(results.items(), key=lambda kv: kv[1][1], reverse=True)
    top_two = ranked[:2]
    voting_estimators = [(name, pipe) for name, (pipe, _) in top_two]
    voting_pipeline = VotingClassifier(estimators=voting_estimators, voting="soft")
    voting_cv_f1 = cross_val_score(voting_pipeline, X_train, y_train, cv=5, scoring="f1_macro", n_jobs=-1).mean()
    print(f"  voting_ensemble({top_two[0][0]}+{top_two[1][0]}): best CV F1-macro = {voting_cv_f1:.4f}")
    results["voting_ensemble"] = (voting_pipeline, voting_cv_f1)

    best_name, (best_pipeline, best_cv_f1) = max(results.items(), key=lambda kv: kv[1][1])
    print(f"\nSelected model: {best_name} (CV F1-macro = {best_cv_f1:.4f})")

    best_pipeline.fit(X_train, y_train)
    y_pred = best_pipeline.predict(X_test)
    y_proba = best_pipeline.predict_proba(X_test)
    classes = list(best_pipeline.classes_) if hasattr(best_pipeline, "classes_") \
        else list(best_pipeline.named_steps["clf"].classes_)
    pv_index = classes.index(PV_LABEL)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    metrics = {
        "selected_model": best_name,
        "classes": classes,
        "cv_f1_macro_at_selection": best_cv_f1,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "roc_auc_ovr_macro": roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro", labels=classes),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=classes).tolist(),
        "classification_report": report,
        "pv_class_metrics": report.get(PV_LABEL, {}),
    }

    # PV-vs-rest high-sensitivity screen: flag "high suspicion of PV" from
    # P(PV) directly, rather than relying on the multi-class argmax, since
    # the report's clinical concern is specifically about missing PV cases.
    pv_threshold = tune_pv_screen_threshold(
        best_pipeline, X_train, y_train, pv_index, beta=PV_SCREEN_FBETA_WEIGHT
    )
    y_test_is_pv = (y_test == PV_LABEL).astype(int)
    pv_proba = y_proba[:, pv_index]
    pv_flag = (pv_proba >= pv_threshold).astype(int)
    metrics["pv_screen"] = {
        "threshold": pv_threshold,
        "precision": precision_score(y_test_is_pv, pv_flag, zero_division=0),
        "recall": recall_score(y_test_is_pv, pv_flag, zero_division=0),
        "f1_score": f1_score(y_test_is_pv, pv_flag, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test_is_pv, pv_flag).tolist(),
    }
    print(f"PV screening threshold (F{PV_SCREEN_FBETA_WEIGHT}-optimal on P(PV)): {pv_threshold:.2f}")

    print("\n=== Improved Model ===")
    print(json.dumps(metrics, indent=2))

    joblib.dump(
        {"pipeline": best_pipeline, "pv_index": pv_index, "pv_threshold": pv_threshold},
        "ml_pipeline/artifacts/improved_model.joblib",
    )
    with open("ml_pipeline/artifacts/improved_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    cm = confusion_matrix(y_test, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(cmap="Greens", ax=ax, xticks_rotation=30)
    plt.title(f"Improved Model ({best_name}) - Confusion Matrix")
    plt.tight_layout()
    plt.savefig("ml_pipeline/artifacts/improved_confusion_matrix.png")
    plt.close()

    # SHAP explainability for the Pemphigus Vulgaris class specifically
    # (best effort -- only for tree-based single models).
    try:
        underlying_clf = best_pipeline.named_steps["clf"]
        preprocess = best_pipeline.named_steps["preprocess"]
        select = best_pipeline.named_steps["select"]
        X_test_transformed = select.transform(preprocess.transform(X_test))
        feature_names = np.array(preprocess.get_feature_names_out())[select.get_support()]

        explainer = shap.TreeExplainer(underlying_clf)
        shap_values = explainer.shap_values(X_test_transformed)
        if isinstance(shap_values, list):
            sv = shap_values[pv_index]  # older SHAP: list of per-class arrays
        elif shap_values.ndim == 3:
            sv = shap_values[:, :, pv_index]  # newer SHAP: (samples, features, classes)
        else:
            sv = shap_values

        shap.summary_plot(sv, X_test_transformed, feature_names=feature_names, show=False)
        plt.title(f"SHAP summary — {PV_LABEL} class")
        plt.tight_layout()
        plt.savefig("ml_pipeline/artifacts/shap_summary_pv.png")
        plt.close()
        print("Saved SHAP summary plot to ml_pipeline/artifacts/shap_summary_pv.png")
    except Exception as exc:
        print(f"Skipped SHAP explainability (model type not tree-based or ensemble): {exc}")


if __name__ == "__main__":
    main()
