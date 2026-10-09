import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ==========================================================
# LOAD DATASET
# ==========================================================

data = pd.read_csv("dataset.csv")

print("Dataset loaded successfully.")
print("Total samples:", len(data))


# ==========================================================
# FEATURES
# ==========================================================

X = data[
    [
        "email_count",
        "phone_count",
        "url_count",
        "qr_count",
        "raw_exposure",
        "pei"
    ]
]


# ==========================================================
# TARGET
# ==========================================================

y = data["class"]


# ==========================================================
# USE THE SAME SPLIT AS TRAINING
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


# ==========================================================
# LOAD TRAINED MODEL
# ==========================================================

model = joblib.load(
    "pei_model.pkl"
)

print("ML model loaded successfully.")


# ==========================================================
# PREDICT VALIDATION DATA
# ==========================================================

predictions = model.predict(
    X_test
)


# ==========================================================
# ACCURACY
# ==========================================================

accuracy = accuracy_score(
    y_test,
    predictions
)


print()
print("==========================================")
print("       PEI ML MODEL VALIDATION")
print("==========================================")

print()
print(
    "Validation samples:",
    len(X_test)
)

print(
    "Accuracy:",
    round(accuracy * 100, 2),
    "%"
)


# ==========================================================
# CLASSIFICATION REPORT
# ==========================================================

print()
print("Classification Report")
print("------------------------------------------")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# ==========================================================
# CONFUSION MATRIX
# ==========================================================

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=[
        "LOW",
        "MODERATE",
        "HIGH"
    ]
)


print()
print("Confusion Matrix")
print("------------------------------------------")

print(
    "             LOW  MODERATE  HIGH"
)

print(
    "LOW       ",
    matrix[0]
)

print(
    "MODERATE  ",
    matrix[1]
)

print(
    "HIGH      ",
    matrix[2]
)


# ==========================================================
# INDIVIDUAL PREDICTIONS
# ==========================================================

print()
print("Individual Validation Results")
print("------------------------------------------")

for actual, predicted in zip(
    y_test,
    predictions
):

    if actual == predicted:

        result = "CORRECT"

    else:

        result = "INCORRECT"

    print(
        "Actual:",
        actual,
        "| Predicted:",
        predicted,
        "|",
        result
    )


print()
print("==========================================")
print("Validation complete.")
print("==========================================")