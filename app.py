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
.hero {padding:28px;border:1px solid #29445f;border-radius:22px;background:linear-gradient(110deg,#142a49,#103b3b);margin-bottom:22px;}
.eyebrow {color:#7dd3fc;font-size:.78rem;letter-spacing:.15em;font-weight:800;}
.hero h1 {color:#fff;font-size:2.5rem;margin:.45rem 0;}
.hero p {color:#c4d4e8;max-width:760px;line-height:1.6;}
[data-testid="stMetric"] {background:#132238;border:1px solid #2b405a;border-radius:14px;padding:14px;}
[data-testid="stFileUploader"] {background:#101e31;border:1px dashed #426485;border-radius:14px;padding:12px;}
.stButton>button {border-radius:10px;background:#167d89;color:white;border:0;font-weight:700;min-height:42px;}
.stButton>button:hover {background:#2099a5;color:white;}
</style>
<div class="hero">
  <div class="eyebrow">SENTINEL · DIGITAL SAFETY LAB</div>
  <h1>See the risk before you share.</h1>
  <p>Scan screenshots for personal information and inspect email screenshots for possible phishing signals. This is a research prototype, not a guarantee of safety.</p>
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


def find_sensitive(text, qr_data=None):
    emails = sorted(set(EMAIL_RE.findall(text)))
    urls = sorted(set(match.rstrip(".,;:!?") for match in URL_RE.findall(text)))
    phones = sorted(set(match.strip() for match in PHONE_RE.findall(text) if sum(ch.isdigit() for ch in match) >= 8))
    found = []
    if emails:
        found.append(("Email", emails))
    if phones:
        found.append(("Phone", phones))
    if urls:
        found.append(("URL", urls))
    if qr_data:
        found.append(("QR Code", [qr_data]))
    return found


def calculate_pei(found, area, visibility, confidence):
    raw = sum(WEIGHTS[kind] * area * visibility * confidence for kind, _ in found)
    score = min(100.0, max(0.0, 100.0 * raw / R_MAX))
    if score <= 33.33:
        level = "LOW"
    elif score <= 66.67:
        level = "MODERATE"
    else:
        level = "HIGH"
    return raw, score, level


def phishing_prediction(text):
    if phishing_model is not None:
        try:
            prediction = phishing_model.predict([text])[0]
            label_text = str(prediction).strip().lower()
            if label_text in {"1", "phishing", "phish", "malicious", "spam"}:
                label = "Potentially phishing"
            elif label_text in {"0", "legitimate", "legit", "safe", "ham"}:
                label = "Likely legitimate"
            else:
                label = f"Model output: {prediction}"
            probability = None
            if hasattr(phishing_model, "predict_proba") and hasattr(phishing_model, "classes_"):
                probs = phishing_model.predict_proba([text])[0]
                classes = [str(c).strip().lower() for c in phishing_model.classes_]
                for candidate in ("1", "phishing", "phish", "malicious", "spam"):
                    if candidate in classes:
                        probability = float(probs[classes.index(candidate)])
                        break
            return label, probability, "trained model"
        except Exception:
            pass

    # Transparent fallback heuristic if the trained model is absent/unreadable.
    lower = text.lower()
    indicators = {
        "urgent or threatening language": any(x in lower for x in ["urgent", "immediately", "suspended", "will be closed", "act now"]),
        "credential or payment request": any(x in lower for x in ["password", "verify your account", "credit card", "bank details", "payment information"]),
        "link with a shortened or unusual pattern": any(x in lower for x in ["bit.ly/", "tinyurl.com/", "login-", "secure-"]),
        "unexpected prize or reward": any(x in lower for x in ["you won", "claim your prize", "free gift", "winner"]),
    }
    hits = [name for name, present in indicators.items() if present]
    if len(hits) >= 2:
        label = "Suspicious signs detected"
    elif len(hits) == 1:
        label = "Use caution"
    else:
        label = "No common warning signs found"
    return label, None, "basic text heuristic (no trained model loaded)"


st.caption("Prototype notice: PEI is a research score based on selected weights and image ratings. It is not a certified risk standard. Avoid uploading screenshots containing real passwords, private messages, or other sensitive data.")

privacy_tab, phishing_tab = st.tabs(["🔎 Privacy Exposure Scanner", "✉️ Phishing Email Screenshot"])

with privacy_tab:
    st.header("Screenshot Privacy Scanner")
    st.write("Upload a screenshot to extract readable text, detect common personal-information patterns, and calculate a PEI score.")
    privacy_file = st.file_uploader("Upload screenshot", type=["png", "jpg", "jpeg", "webp"], key="privacy_upload")
    if privacy_file:
        try:
            image = Image.open(privacy_file).convert("RGB")
            left, right = st.columns([1, 1])
            with left:
                st.image(image, caption="Uploaded screenshot", use_container_width=True)
            with right:
                st.subheader("Exposure assumptions")
                area = st.slider("Relative area of exposed information", 0.0, 1.0, 1.0, 0.05, key="area")
                visibility = st.slider("Visibility / readability", 0.0, 1.0, 1.0, 0.05, key="visibility")
                confidence = st.slider("Detection confidence", 0.0, 1.0, 0.90, 0.05, key="confidence")
                run_privacy = st.button("Scan screenshot", key="scan_privacy", use_container_width=True)
            if run_privacy:
                with st.spinner("Reading screenshot..."):
                    try:
                        extracted = ocr_image(image)
                    except RuntimeError as exc:
                        st.error(str(exc))
                        extracted = None
                    qr_data = read_qr(image)
                if extracted is not None:
                    found = find_sensitive(extracted, qr_data)
                    raw, pei, level = calculate_pei(found, area, visibility, confidence)
                    m1, m2, m3 = st.columns(3)
                    m1.metric("PEI score", f"{pei:.2f} / 100")
                    m2.metric("Exposure level", level)
                    m3.metric("Detected categories", str(len(found)))
                    if found:
                        st.subheader("Detected information")
                        for kind, values in found:
                            st.markdown(f"**{kind}** · {len(values)} item(s)")
                            for value in values:
                                st.code(value, language=None)
                    else:
                        st.info("No email, phone-number, URL, or QR-code content was detected. OCR may miss information in blurry or stylized screenshots.")
                    if pei_model is not None:
                        try:
                            features = [[sum(k == "Email" for k, _ in found), sum(k == "Phone" for k, _ in found), sum(k == "URL" for k, _ in found), sum(k == "QR Code" for k, _ in found), raw, pei]]
                            model_result = pei_model.predict(features)[0]
                            st.write(f"**PEI model output:** `{model_result}`")
                        except Exception as exc:
                            st.warning(f"PEI model could not score this input. The formula-based PEI above is still available. Details: {exc}")
                    st.caption("Interpretation: higher PEI means more detected exposure under the current weights and selected assumptions; it does not prove that an account has been compromised.")
        except Exception as exc:
            st.error(f"Could not open this image: {exc}")

with phishing_tab:
    st.header("Email Screenshot Threat Checker")
    st.write("Upload an email screenshot. The app uses OCR, then checks the extracted text with your trained model if available.")
    email_file = st.file_uploader("Upload email screenshot", type=["png", "jpg", "jpeg", "webp"], key="email_upload")
    if email_file:
        try:
            email_image = Image.open(email_file).convert("RGB")
            st.image(email_image, caption="Email screenshot", use_container_width=True)
            if st.button("Analyze email screenshot", key="analyze_email", use_container_width=True):
                try:
                    email_text = ocr_image(email_image)
                except RuntimeError as exc:
                    st.error(str(exc))
                    email_text = ""
                if email_text:
                    with st.expander("Text extracted from screenshot", expanded=False):
                        st.text(email_text)
                    label, probability, method = phishing_prediction(email_text)
                    st.subheader("Analysis result")
                    if "phishing" in label.lower() or "suspicious" in label.lower():
                        st.error(label)
                    elif "caution" in label.lower():
                        st.warning(label)
                    else:
                        st.info(label)
                    st.caption(f"Analysis method: {method}.")
                    if probability is not None:
                        st.metric("Model phishing probability", f"{probability * 100:.1f}%")
                    st.warning("This result is only a screening aid. Do not click links or provide credentials based only on this prediction; verify suspicious messages through the organization’s official channel.")
                elif not email_text:
                    st.info("No readable text was extracted. Try a clearer image or a closer crop of the email body.")
        except Exception as exc:
            st.error(f"Could not open this image: {exc}")

with st.expander("System status / troubleshooting"):
    st.write(f"OCR library available: **{'Yes' if pytesseract is not None else 'No'}**")
    st.write(f"PEI model loaded: **{'Yes' if pei_model is not None else 'No'}**")
    st.write(f"Phishing model loaded: **{'Yes' if phishing_model is not None else 'No'}**")
    st.write(f"QR scanner available: **{'Yes' if cv2 is not None else 'No'}**")
    st.caption("If deploying to Streamlit Community Cloud, requirements.txt should include streamlit, pillow, pytesseract, joblib, scikit-learn, opencv-python-headless, and numpy. Add a packages.txt file containing tesseract-ocr for the system OCR engine. Keep model files in the same repository folder as app.py.")
