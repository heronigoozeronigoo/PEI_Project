from pathlib import Path
import re

import streamlit as st
from PIL import Image

# Optional dependencies: the app can still open if these are unavailable.
try:
    import joblib
except Exception:
    joblib = None

try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    import cv2
    import numpy as np
except Exception:
    cv2 = None
    np = None

BASE_DIR = Path(__file__).resolve().parent
PEI_MODEL_PATH = BASE_DIR / "pei_model.pkl"
PHISHING_MODEL_PATH = BASE_DIR / "phishing_model.pkl"

st.set_page_config(page_title="Sentinel | Digital Safety Lab", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
.stApp {background: linear-gradient(135deg,#07111f 0%,#101b2e 55%,#071a20 100%); color:#edf4ff;}
.block-container {max-width:1150px;padding-top:2rem;padding-bottom:3rem;}
.hero {padding:30px;border:1px solid #29445f;border-radius:22px;background:linear-gradient(110deg,#142a49,#103b3b);margin-bottom:22px;box-shadow:0 12px 34px rgba(0,0,0,.18);}
.eyebrow {color:#7dd3fc;font-size:.78rem;letter-spacing:.15em;font-weight:800;}
.hero h1 {color:#fff;font-size:2.5rem;margin:.45rem 0;}
.hero p {color:#c4d4e8;max-width:760px;line-height:1.6;}
[data-testid="stMetric"] {background:#132238;border:1px solid #2b405a;border-radius:14px;padding:14px;}
section[data-testid="stSidebar"] {background:#0b1627;}
div[data-testid="stExpander"] {border:1px solid #2b405a;border-radius:14px;}
.result-card {padding:18px;border:1px solid #2b405a;border-radius:16px;background:#101e31;margin:10px 0;}
.small-label {color:#7dd3fc;text-transform:uppercase;letter-spacing:.12em;font-size:.75rem;font-weight:800;}
[data-testid="stFileUploader"] {background:#101e31;border:1px dashed #426485;border-radius:14px;padding:12px;}
.stButton>button {border-radius:10px;background:#167d89;color:white;border:0;font-weight:700;min-height:42px;}
.stButton>button:hover {background:#2099a5;color:white;}
</style>
<div class="hero">
  <div class="eyebrow">SENTINEL · DIGITAL SAFETY LAB</div>
  <h1>See the risk before you share.</h1>
  <p>Scan screenshots for personal information, calculate a Mathematical Privacy Exposure Index (PEI), and inspect email screenshots for possible phishing signals.</p>
</div>
""", unsafe_allow_html=True)

WEIGHTS = {"Email": 0.15, "Phone": 0.15, "URL": 0.10, "QR Code": 0.20}
R_MAX = 0.60

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
URL_RE = re.compile(r"\b(?:https?://|www\.)[^\s<>()\[\]{}\"']+", re.I)
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d ().-]{7,}\d)(?!\w)")


def load_model(path):
    if joblib is None or not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


pei_model = load_model(PEI_MODEL_PATH)
phishing_model = load_model(PHISHING_MODEL_PATH)


def ocr_image(image):
    if pytesseract is None:
        raise RuntimeError("OCR package pytesseract is not installed. Add pytesseract to requirements.txt and tesseract-ocr to packages.txt.")
    try:
        return pytesseract.image_to_string(image).strip()
    except Exception as exc:
        raise RuntimeError("OCR could not read this image. Check that Tesseract OCR is installed in the deployment environment.") from exc


def read_qr(image):
    if cv2 is None or np is None:
        return None
    try:
        rgb = np.array(image.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        detector = cv2.QRCodeDetector()
        data, points, _ = detector.detectAndDecode(bgr)
        return data.strip() if data else None
    except Exception:
        return None
