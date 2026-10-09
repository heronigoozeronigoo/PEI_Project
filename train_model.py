import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

import joblib


# ==========================================================
# LOAD DATASET
# ==========================================================

data = pd.read_csv("dataset.csv")

print("Dataset loaded successfully!")
print()
print(data)


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
# SPLIT DATA
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


# ==========================================================
# CREATE MODEL
# ==========================================================

model = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)


# ==========================================================
# TRAIN MODEL
# ==========================================================

model.fit(
    X_train,
    y_train
)


# ==========================================================
# TEST MODEL
# ==========================================================

predictions = model.predict(
    X_test
)


accuracy = accuracy_score(
    y_test,
    predictions
)


print()
print("===================================")
print("ML MODEL RESULTS")
print("===================================")

print(
    "Accuracy:",
    round(accuracy * 100, 2),
    "%"
)


print()
print("Classification Report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# ==========================================================
# SAVE MODEL
# ==========================================================

joblib.dump(
    model,
    "pei_model.pkl"
)


print()
print("===================================")
print("MODEL SAVED")
print("===================================")

print(
    "File created: pei_model.pkl"
)