
# ==========================================================
# EMAIL PHISHING MODEL — VERSION 2
# Dataset: Zenodo Phishing-Email-Detection-Dataset
# Labels: 0 = Safe/Legitimate, 1 = Phishing
# ==========================================================

from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# ----------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATASET = BASE_DIR / "Balanced_Dataset.csv"

# Keep the existing model safe.
MODEL_OUTPUT = BASE_DIR / "email_phishing_model_v2.pkl"
REPORT_OUTPUT = BASE_DIR / "email_phishing_model_v2_evaluation.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20

# ----------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------

print("=" * 60)
print("EMAIL PHISHING MODEL — VERSION 2")
print("=" * 60)

if not DATASET.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATASET}\n"
        "Place Balanced_Dataset.csv beside this script."
    )

print("\nLoading dataset...")

df = pd.read_csv(
    DATASET,
    usecols=["body", "label"],
    dtype={"body": "string", "label": "string"},
    on_bad_lines="error",
)

df = df.dropna(subset=["body", "label"]).copy()
df["body"] = df["body"].str.strip()
df["label"] = df["label"].str.strip()

df = df[df["body"] != ""]

# Verify that labels are exactly the expected classes.
valid_labels = {"0", "0.0", "1", "1.0"}
unexpected = set(df["label"].unique()) - valid_labels

if unexpected:
    raise ValueError(
        f"Unexpected labels found: {unexpected}"
    )

# Convert verified labels to integers.
df["label"] = df["label"].map({
    "0": 0,
    "0.0": 0,
    "1": 1,
    "1.0": 1,
})

df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

# Remove exact duplicate bodies that have conflicting labels.
label_counts_by_body = df.groupby("body")["label"].nunique()
conflicting_bodies = label_counts_by_body[
    label_counts_by_body > 1
].index

if len(conflicting_bodies):
    print(
        "Removing conflicting duplicate bodies:",
        len(conflicting_bodies)
    )
    df = df[~df["body"].isin(conflicting_bodies)].copy()

# Remove exact duplicate body/label pairs.
before = len(df)
df = df.drop_duplicates(subset=["body", "label"])
print("Duplicate rows removed:", before - len(df))

if df["label"].nunique() != 2:
    raise ValueError(
        "The dataset must contain both labels 0 and 1."
    )

print("\nUsable emails:", len(df))
print("\nLabel counts:")
print(df["label"].value_counts().sort_index())
print("\nLabel meanings: 0 = legitimate; 1 = phishing")

# ----------------------------------------------------------
# SPLIT DATA
# ----------------------------------------------------------

X = df["body"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("\nTraining samples:", len(X_train))
print("Test samples:", len(X_test))

# ----------------------------------------------------------
# BUILD AND TRAIN MODEL
# ----------------------------------------------------------

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            max_features=20000,
            min_df=2,
            sublinear_tf=True,
            dtype=float,
        ),
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    ),
])

print("\nTraining model...")
model.fit(X_train, y_train)

# ----------------------------------------------------------
# EVALUATE ON HELD-OUT TEST DATA
# ----------------------------------------------------------

print("\nEvaluating held-out test data...")

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

classes = list(model.classes_)
phishing_column = classes.index(1)
phishing_scores = probabilities[:, phishing_column]

accuracy = accuracy_score(y_test, predictions)
balanced_accuracy = balanced_accuracy_score(y_test, predictions)

precision = precision_score(
    y_test, predictions, pos_label=1, zero_division=0
)
recall = recall_score(
    y_test, predictions, pos_label=1, zero_division=0
)
f1 = f1_score(
    y_test, predictions, pos_label=1, zero_division=0
)

# Positive = phishing; negative = legitimate.
tn, fp, fn, tp = confusion_matrix(
    y_test, predictions, labels=[0, 1]
).ravel()

false_positive_rate = (
    fp / (fp + tn) if (fp + tn) > 0 else 0.0
)

try:
    roc_auc = roc_auc_score(y_test, phishing_scores)
except ValueError:
    roc_auc = None

print("\n" + "=" * 60)
print("HELD-OUT TEST RESULTS")
print("=" * 60)

print(f"Accuracy:            {accuracy:.4f}")
print(f"Balanced accuracy:   {balanced_accuracy:.4f}")
print(f"Phishing precision:  {precision:.4f}")
print(f"Phishing recall:     {recall:.4f}")
print(f"Phishing F1-score:   {f1:.4f}")
print(f"False-positive rate: {false_positive_rate:.4f}")

if roc_auc is not None:
    print(f"ROC-AUC:             {roc_auc:.4f}")

print("\nConfusion matrix")
print("Rows = actual [legitimate, phishing]")
print("Columns = predicted [legitimate, phishing]")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Legitimate", "Phishing"],
        zero_division=0,
    )
)

# ----------------------------------------------------------
# SAVE MODEL AND EVALUATION REPORT
# ----------------------------------------------------------

# Save to a NEW filename. Do not overwrite the existing model.
joblib.dump(model, MODEL_OUTPUT)

report = {
    "dataset": DATASET.name,
    "label_mapping": {
        "0": "Legitimate",
        "1": "Phishing",
    },
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE,
    "total_samples": int(len(df)),
    "training_samples": int(len(X_train)),
    "test_samples": int(len(X_test)),
    "accuracy": float(accuracy),
    "balanced_accuracy": float(balanced_accuracy),
    "phishing_precision": float(precision),
    "phishing_recall": float(recall),
    "phishing_f1": float(f1),
    "false_positive_rate": float(false_positive_rate),
    "roc_auc": float(roc_auc) if roc_auc is not None else None,
    "confusion_matrix": [
        [int(tn), int(fp)],
        [int(fn), int(tp)],
    ],
    "warning": (
        "Metrics are from a random held-out split of this dataset. "
        "They do not establish performance on independent modern emails."
    ),
}

with open(REPORT_OUTPUT, "w", encoding="utf-8") as file:
    json.dump(report, file, indent=4)

print("\nModel saved to:")
print(MODEL_OUTPUT)

print("\nEvaluation report saved to:")
print(REPORT_OUTPUT)

print("\nTraining and evaluation completed.")
print("The original email_phishing_model.pkl was not overwritten.")
