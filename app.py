from pathlib import Path
import re
import socket

import joblib
import streamlit as st
import pytesseract
from PIL import Image

from detector import detect_sensitive_information


# ==========================================================
# PAGE SETTINGS AND MODEL PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent
PEI_MODEL_PATH = BASE_DIR / "pei_model.pkl"
PHISHING_MODEL_PATH = BASE_DIR / "phishing_model.pkl"

st.set_page_config(
    page_title="Sentinel | Privacy & Phishing Lab",
    page_icon="🛡️",
    layout="wide",
)

# ==========================================================
# LOAD MODELS INDEPENDENTLY
# ==========================================================

@st.cache_resource
def load_model(model_path: str):
    """Load a joblib model from a path and cache it for the session."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"{path.name} was not found in the app folder.")
    return joblib.load(path)


try:
    pei_model = load_model(str(PEI_MODEL_PATH))
    pei_model_loaded = True
    pei_model_error = ""
except Exception as exc:
    pei_model = None
    pei_model_loaded = False
    pei_model_error = str(exc)

try:
    phishing_model = load_model(str(PHISHING_MODEL_PATH))
    phishing_model_loaded = True
    phishing_model_error = ""
except Exception as exc:
    phishing_model = None
    phishing_model_loaded = False
    phishing_model_error = str(exc)


# ==========================================================
# PEI SETTINGS
# ==========================================================

WEIGHTS = {
    "Email": 0.15,
    "Phone": 0.15,
    "URL": 0.10,
    "QR Code": 0.20,
}
R_MAX = 0.60

DEFAULT_AREA = 1.00
DEFAULT_VISIBILITY = 1.00
DEFAULT_CONFIDENCE = 0.90


def calculate_item_score(information_type, area, visibility, confidence):
    """Calculate one sensitive item's weighted exposure score."""
    weight = WEIGHTS[information_type]
    return weight * area * visibility * confidence


def classify_pei(pei):
    """Convert a PEI score to the prototype's exposure category."""
    if pei <= 33.33:
        return "LOW"
    if pei <= 66.67:
        return "MODERATE"
    return "HIGH"



def extract_screenshot_text(image):
    """Extract readable text from an uploaded screenshot using Tesseract OCR."""
    return pytesseract.image_to_string(image).strip()

