
# train_phishing_model.py
# Train a separate phishing email text classifier.
# Does not modify the existing PEI model.

from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

DATA_PATH = Path("email_dataset.csv")
MODEL_PATH = Path("phishing_model.pkl")
RANDOM_STATE = 42

# Add or adjust aliases to match the dataset's documented labels.
LABEL_MAP = {
    "legitimate": 0,
    "legit": 0,
    "safe": 0,
    "ham": 0,
    "0": 0,
    "phishing": 1,
    "phish": 1,
    "scam": 1,
    "1": 1,
}


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "email_dataset.csv was not found. "
            "Obtain a labeled dataset and prepare its text and label columns first."
        )

    df = pd.read_csv(DATA_PATH)

    # Expected format: text,label
    required = {"text", "label"}
    if not required.issubset(df.columns):
        raise ValueError(
            "CSV must contain columns named 'text' and 'label'. "
            f"Found: {list(df.columns)}"
        )

    df = df[["text", "label"]].dropna().copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = (
        df["label"].astype(str).str.strip().str.lower()
    )

    df = df[df["text"].str.len() > 0]
    df["label"] = df["label"].map(LABEL_MAP)

    if df["label"].isna().any():
        unknown = sorted(
            set(
                df.loc[df["label"].isna(), "label"]
                .astype(str)
                .tolist()
            )
        )
        raise ValueError(
            "Unrecognized labels detected. Review the dataset's label "
            "definitions and update LABEL_MAP before training."
        )

    df["label"] = df["label"].astype(int)
    df = df.drop_duplicates(subset=["text", "label"])

    if df["label"].nunique() != 2:
        raise ValueError(
            "Training requires both legitimate and phishing examples."
        )

    if df["label"].value_counts().min() < 2:
        raise ValueError(
            "Each class needs at least two records for a stratified split."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                max_features=30000,
                sublinear_tf=True,
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

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\nTEST SET RESULTS")
    print("----------------")
    print("Total usable records:", len(df))
    print("Training records:", len(X_train))
    print("Test records:", len(X_test))
    print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
    print(
        "Phishing precision:",
        round(precision_score(y_test, predictions, zero_division=0), 4),
    )
    print(
        "Phishing recall:",
        round(recall_score(y_test, predictions, zero_division=0), 4),
    )
    print(
        "Phishing F1-score:",
        round(f1_score(y_test, predictions, zero_division=0), 4),
    )

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

    print("Confusion matrix (rows=actual, columns=predicted):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    # Train the final pipeline on all available records only after
    # evaluating the held-out split.
    model.fit(df["text"], df["label"])
    joblib.dump(model, MODEL_PATH)

    print(f"\nSaved model to: {MODEL_PATH}")
    print("The held-out metrics above are from the initial test split.")


if __name__ == "__main__":
    main()
