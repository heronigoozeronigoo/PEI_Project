
# phishing_checker.py
# Loads the trained email classifier and evaluates email text.

from pathlib import Path

import joblib

MODEL_PATH = Path(__file__).resolve().parent / "phishing_model.pkl"

# Starting threshold for testing, not a validated security threshold.
PHISHING_THRESHOLD = 0.70


def load_phishing_model():
    """Load the trained phishing model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH.name} was not found in {MODEL_PATH.parent}. "
            "Place the trained model in this folder."
        )

    return joblib.load(MODEL_PATH)


def check_email(email_text, model=None):
    """
    Analyze email text.

    Assumed label mapping:
        0 = Legitimate
        1 = Phishing

    Verify this mapping against the model's training code.
    """
    if not isinstance(email_text, str) or not email_text.strip():
        raise ValueError("No readable email text was provided.")

    if model is None:
        model = load_phishing_model()

    result = {
        "classification": "Unable to determine",
        "prediction": None,
        "phishing_model_score": None,
        "legitimate_model_score": None,
        "threshold": PHISHING_THRESHOLD,
        "warning": (
            "This text-based model cannot verify the sender's identity "
            "or guarantee that an email is safe."
        ),
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([email_text])[0]
        classes = [int(label) for label in model.classes_]

        if 0 not in classes or 1 not in classes:
            raise ValueError(
                "Expected model classes 0 (Legitimate) and "
                "1 (Phishing). Check the training label mapping."
            )

        legitimate_score = float(probabilities[classes.index(0)])
        phishing_score = float(probabilities[classes.index(1)])

        result["legitimate_model_score"] = legitimate_score
        result["phishing_model_score"] = phishing_score

        if phishing_score >= PHISHING_THRESHOLD:
            result["prediction"] = 1
            result["classification"] = "Potentially Phishing"
        else:
            result["prediction"] = 0
            result["classification"] = "Needs Further Review"

    else:
        prediction = int(model.predict([email_text])[0])
        result["prediction"] = prediction

        if prediction == 1:
            result["classification"] = "Potentially Phishing"
        elif prediction == 0:
            result["classification"] = "Needs Further Review"
        else:
            result["classification"] = "Unable to determine"

    return result
