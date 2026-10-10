
# ============================================================
# PHISHING EMAIL MODEL TRAINER
# Dataset: phishing_legit_dataset_KD_10000.csv
# Output: phishing_model.pkl
# ============================================================

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
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# ------------------------------------------------------------
# 1. FILE PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "phishing_legit_dataset_KD_10000.csv"
MODEL_PATH = BASE_DIR / "phishing_model.pkl"
RESULTS_PATH = BASE_DIR / "phishing_model_evaluation.json"

TEST_SIZE = 0.20
RANDOM_STATE = 42


# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

print("=" * 60)
print("PHISHING EMAIL MODEL TRAINER")
print("=" * 60)

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_PATH.name}\n"
        "Make sure the CSV is in the same folder as this script."
    )

df = pd.read_csv(DATASET_PATH)

print("\nDataset columns:", df.columns.tolist())
print("Original number of rows:", len(df))

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"

if TEXT_COLUMN not in df.columns or LABEL_COLUMN not in df.columns:
    raise ValueError(
        "Expected columns 'text' and 'label'. "
        f"Found: {df.columns.tolist()}"
    )


# ------------------------------------------------------------
# 3. CLEAN DATA
# ------------------------------------------------------------

df = df[[TEXT_COLUMN, LABEL_COLUMN]].copy()

df[TEXT_COLUMN] = (
    df[TEXT_COLUMN]
    .fillna("")
    .astype(str)
    .str.strip()
)

df[LABEL_COLUMN] = pd.to_numeric(
    df[LABEL_COLUMN],
    errors="coerce"
)

df = df.dropna(subset=[LABEL_COLUMN])
df = df[df[TEXT_COLUMN] != ""]
df = df[df[LABEL_COLUMN].isin([0, 1])]

df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)

# Remove identical email text before splitting.
# If identical text has conflicting labels, exclude it.
label_counts = df.groupby(TEXT_COLUMN)[LABEL_COLUMN].nunique()
conflicting_texts = label_counts[label_counts > 1].index

if len(conflicting_texts) > 0:
    print(
        "\nRemoving identical texts with conflicting labels:",
        len(conflicting_texts)
    )
    df = df[~df[TEXT_COLUMN].isin(conflicting_texts)]

df = df.drop_duplicates(subset=[TEXT_COLUMN]).reset_index(drop=True)

if df[LABEL_COLUMN].nunique() != 2:
    raise ValueError(
        "Both labels 0 and 1 must be present after cleaning."
    )

print("\nCleaned rows:", len(df))
print("\nLabel counts:")
print(df[LABEL_COLUMN].value_counts().sort_index())

print("\nLabel mapping:")
print("0 = Legitimate")
print("1 = Phishing")


# ------------------------------------------------------------
# 4. SPLIT INTO TRAINING AND TEST SETS
# ------------------------------------------------------------

X = df[TEXT_COLUMN]
y = df[LABEL_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))


# ------------------------------------------------------------
# 5. CREATE THE MODEL PIPELINE
# ------------------------------------------------------------

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=20000,
            sublinear_tf=True,
            strip_accents="unicode"
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        )
    )
])


# ------------------------------------------------------------
# 6. TRAIN
# ------------------------------------------------------------

print("\nTraining model...")
model.fit(X_train, y_train)
print("Training complete.")


# ------------------------------------------------------------
# 7. EVALUATE ON THE HELD-OUT TEST SET
# ------------------------------------------------------------

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, predictions)
balanced_acc = balanced_accuracy_score(y_test, predictions)

precision = precision_score(
    y_test, predictions, pos_label=1, zero_division=0
)

recall = recall_score(
    y_test, predictions, pos_label=1, zero_division=0
)

f1 = f1_score(
    y_test, predictions, pos_label=1, zero_division=0
)

roc_auc = roc_auc_score(y_test, probabilities)

matrix = confusion_matrix(
    y_test, predictions, labels=[0, 1]
)

tn, fp, fn, tp = matrix.ravel()

false_positive_rate = (
    fp / (fp + tn) if (fp + tn) else 0.0
)

print("\n" + "=" * 60)
print("HELD-OUT TEST RESULTS")
print("=" * 60)

print(f"Accuracy:            {accuracy:.4f}")
print(f"Balanced accuracy:   {balanced_acc:.4f}")
print(f"Phishing precision:  {precision:.4f}")
print(f"Phishing recall:     {recall:.4f}")
print(f"Phishing F1-score:   {f1:.4f}")
print(f"ROC-AUC:             {roc_auc:.4f}")
print(f"False-positive rate: {false_positive_rate:.4f}")

print("\nConfusion matrix:")
print("Rows = actual; columns = predicted")
print("Class order: 0 = Legitimate, 1 = Phishing")
print(matrix)

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Legitimate", "Phishing"],
        zero_division=0
    )
)

print("\nConfusion matrix details:")
print("True negatives:", tn)
print("False positives:", fp)
print("False negatives:", fn)
print("True positives:", tp)


# ------------------------------------------------------------
# 8. SAVE EVALUATION RESULTS
# ------------------------------------------------------------

results = {
    "dataset": DATASET_PATH.name,
    "cleaned_rows": int(len(df)),
    "training_rows": int(len(X_train)),
    "testing_rows": int(len(X_test)),
    "test_size": TEST_SIZE,
    "random_state": RANDOM_STATE,
    "label_mapping": {
        "0": "Legitimate",
        "1": "Phishing"
    },
    "accuracy": float(accuracy),
    "balanced_accuracy": float(balanced_acc),
    "phishing_precision": float(precision),
    "phishing_recall": float(recall),
    "phishing_f1_score": float(f1),
    "roc_auc": float(roc_auc),
    "false_positive_rate": float(false_positive_rate),
    "confusion_matrix": {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp)
    }
}

with open(RESULTS_PATH, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4)

print("\nEvaluation saved to:", RESULTS_PATH.name)


# ------------------------------------------------------------
# 9. SAVE MODEL
# ------------------------------------------------------------

joblib.dump(model, MODEL_PATH)

print("\nModel saved to:", MODEL_PATH.name)
print("Existing pei_model.pkl was not modified.")

print("\nIMPORTANT:")
print("- Results are from a held-out test split.")
print("- Good test results do not guarantee real-world accuracy.")
print("- This model predicts email text, not screenshot images.")
print("- Your app must extract screenshot text with OCR first.")

print("\nTRAINING PROCESS COMPLETED")
print("=" * 60)
