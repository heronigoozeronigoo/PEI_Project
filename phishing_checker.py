
# phishing_checker.py
# Loads the trained email phishing classifier and predicts email risk.

from pathlib import Path
import joblib


MODEL_PATH = (
    Path(__file__).resolve().parent / "email_phishing_model_v2.pkl"
)



def load_phishing_model():
    """Load the trained email phishing model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH.name} was not found. "
            f"Expected location: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def check_email(email_text, model=None):
    """
    Classify email text.

    Expected labels:
        0 = Legitimate
        1 = Phishing

    Confirm these labels match the model's training code.
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
        "phishing_model_score": None,
        "legitimate_model_score": None,
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([email_text])[0]
        classes = [int(label) for label in model.classes_]

        if 0 in classes:
            result["legitimate_model_score"] = float(
                probabilities[classes.index(0)]
            )

        if 1 in classes:
            result["phishing_model_score"] = float(
                probabilities[classes.index(1)]
            )

    return result
