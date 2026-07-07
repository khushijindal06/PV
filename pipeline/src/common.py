"""Shared constants/helpers used by both the baseline and improved training
scripts, so both are trained and evaluated on an identical train/test split
of the real project dataset (multi-class: Healthy vs. Pemphigus Vulgaris vs.
Pemphigus Foliaceus vs. Bullous Pemphigoid).
"""
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "ml_pipeline/data/pemphigus_dataset_dummy_300.csv"
TARGET = "diagnosis"
ID_COL = "patient_id"

NUMERIC_FEATURES = ["age", "anti_dsg3_elisa_value", "anti_dsg1_elisa_value"]
BINARY_FEATURES = ["nikolsky_sign_present"]
NOMINAL_FEATURES = ["gender", "ethnicity", "initial_symptom_location", "direct_immunofluorescence_result"]

PV_LABEL = "Pemphigus Vulgaris"
RANDOM_STATE = 42


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["nikolsky_sign_present"] = df["nikolsky_sign_present"].astype(int)
    return df


def split_xy(df: pd.DataFrame):
    X = df.drop(columns=[TARGET, ID_COL])
    y = df[TARGET]
    return X, y


def train_test_split_fixed(X, y):
    return train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
