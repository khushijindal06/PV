"""Baseline model: reproduces the pipeline described in the PV Progress
Report (sections 4 and 6) — median/mode imputation, one-hot encoding,
StandardScaler, SelectKBest feature selection, and a Random Forest tuned
with GridSearchCV — applied to the real 4-class dataset (Healthy /
Pemphigus Vulgaris / Pemphigus Foliaceus / Bullous Pemphigoid).

This is the "before" model used as the comparison point for
train_improved.py.
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_pipeline.src.common import (
    BINARY_FEATURES,
    NOMINAL_FEATURES,
    NUMERIC_FEATURES,
    PV_LABEL,
    load_data,
    split_xy,
    train_test_split_fixed,
)

PARAM_GRID = {
    "clf__n_estimators": [100, 300, 500],
    "clf__max_depth": [5, 10, 15, 20],
    "clf__min_samples_split": [2, 5, 10],
    "clf__min_samples_leaf": [1, 2, 4],
    "clf__max_features": ["sqrt", "log2"],
}


def build_pipeline(k_features) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), NUMERIC_FEATURES),
            ("bin", SimpleImputer(strategy="most_frequent"), BINARY_FEATURES),
            ("nom", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore")),
            ]), NOMINAL_FEATURES),
        ]
    )
    return Pipeline([
        ("preprocess", preprocessor),
        ("select", SelectKBest(score_func=f_classif, k=k_features)),
        ("clf", RandomForestClassifier(class_weight="balanced", random_state=42)),
    ])


def main():
    df = load_data()
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split_fixed(X, y)

    pipeline = build_pipeline(k_features=8)
    search = GridSearchCV(pipeline, PARAM_GRID, cv=5, scoring="f1_macro", n_jobs=-1)
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)
    classes = best_model.named_steps["clf"].classes_.tolist()

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    metrics = {
        "classes": classes,
        "best_params": search.best_params_,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "roc_auc_ovr_macro": roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro", labels=classes),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=classes).tolist(),
        "classification_report": report,
        "pv_class_metrics": report.get(PV_LABEL, {}),
    }

    print("=== Baseline Random Forest (4-class) ===")
    print(json.dumps(metrics, indent=2))

    joblib.dump(best_model, "ml_pipeline/artifacts/baseline_model.joblib")
    with open("ml_pipeline/artifacts/baseline_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    cm = confusion_matrix(y_test, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(cmap="Blues", ax=ax, xticks_rotation=30)
    plt.title("Baseline Random Forest - Confusion Matrix")
    plt.tight_layout()
    plt.savefig("ml_pipeline/artifacts/baseline_confusion_matrix.png")
    plt.close()


if __name__ == "__main__":
    main()
