
# phishing_checker.py
# Loads the trained phishing classifier and predicts email risk.

from pathlib import Path

import joblib


MODEL_PATH = Path(__file__).resolve().parent / "phishing_model.pkl"


def load_phishing_model():
    """Load the trained phishing model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "phishing_model.pkl was not found. "
            "Upload the trained model to the repository."
        )

    return joblib.load(MODEL_PATH)


def check_email(email_text, model=None):
    """
    Classify extracted email text.

    Label mapping:
        0 = Legitimate
        1 = Phishing
    """
    if not isinstance(email_text, str) or not email_text.strip():
        raise ValueError("No readable email text was provided.")

    if model is None:
        model = load_phishing_model()

    prediction = int(model.predict([email_text])[0])

    if prediction == 1:
        classification = "Potentially Phishing"
    elif prediction == 0:
        classification = "Likely Legitimate"
    else:
        classification = "Unknown"

    result = {
        "prediction": prediction,
        "classification": classification,
    }

    # Include a model score if the model supports probabilities.
    # This is not necessarily a calibrated probability of real-world risk.
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([email_text])[0]
        classes = list(model.classes_)

        if 1 in classes:
            phishing_index = classes.index(1)
            result["phishing_model_score"] = float(
                probabilities[phishing_index]
            )

    return result
