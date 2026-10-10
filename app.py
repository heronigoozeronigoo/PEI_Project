"""Sentinel — screenshot privacy exposure and phishing screening research app.

Run with: streamlit run app.py
Optional local files: detector.py, pei_model.pkl, phishing_model.pkl.
"""
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from PIL import Image, UnidentifiedImageError

try:
    import joblib
except ImportError:
    joblib = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

try:
    from detector import detect_sensitive_information
except Exception:
    detect_sensitive_information = None

BASE_DIR = Path(__file__).resolve().parent
PEI_MODEL_PATH = BASE_DIR / "pei_model.pkl"
PHISHING_MODEL_PATH = BASE_DIR / "phishing_model.pkl"
ALLOWED_TYPES = ["png", "jpg", "jpeg"]

# PEI weights are the research prototype's chosen sensitivity weights.
WEIGHTS = {"Email": 0.15, "Phone": 0.15, "URL": 0.10, "QR Code": 0.20}
R_MAX = sum(WEIGHTS.values())  # 0.60: maximum if each category saturates at its weight
DEFAULT_AREA = 1.00
DEFAULT_VISIBILITY = 1.00
DEFAULT_CONFIDENCE = 0.90

# Provisional phishing indicator weights. These are points, not probabilities.
PHISHING_WEIGHTS = {
    "Urgency / pressure": 20,
    "Credential request": 25,
    "Visible link": 15,
    "Threat / consequence": 20,
    "Money / reward bait": 10,
    "Multiple URLs": 10,
}

st.set_page_config(page_title="Sentinel | Digital Safety Lab", page_icon="🛡️", layout="wide")


# ------------------------------ Model loading ------------------------------
@st.cache_resource(show_spinner=False)
def load_model(path_string: str):
    """Load and cache an optional joblib model; raise useful errors on failure."""
    if joblib is None:
        raise ImportError("Install joblib to load .pkl models: pip install joblib")
    path = Path(path_string)
    if not path.exists():
        raise FileNotFoundError(f"{path.name} was not found beside app.py.")
    return joblib.load(path)


def optional_model(path: Path) -> Tuple[Optional[Any], str]:
    """Return a model or an explanation without preventing the app from starting."""
    try:
        return load_model(str(path)), ""
    except Exception as exc:
        return None, str(exc)


pei_model, pei_model_error = optional_model(PEI_MODEL_PATH)
phishing_model, phishing_model_error = optional_model(PHISHING_MODEL_PATH)


# ------------------------------ Shared helpers -----------------------------
def open_uploaded_image(uploaded_file) -> Image.Image:
    """Validate and fully load an uploaded image, returning an RGB copy."""
    if uploaded_file is None:
        raise ValueError("Please upload a screenshot first.")
    if not uploaded_file.getvalue():
        raise ValueError("The uploaded file is empty. Choose a valid screenshot.")
    try:
        uploaded_file.seek(0)
        with Image.open(uploaded_file) as source:
            source.verify()
        uploaded_file.seek(0)
        with Image.open(uploaded_file) as source:
            source.load()
            return source.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("This file could not be read as an image. Try a clear PNG or JPG screenshot.") from exc


def clip_unit(value: float, label: str) -> float:
