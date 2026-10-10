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


def predict_phishing(email_text, model):
    """Predict email text using the trained phishing model (0=legitimate, 1=phishing)."""
    if not isinstance(email_text, str) or not email_text.strip():
        raise ValueError("No readable email text was found. Try a clearer screenshot.")

    prediction = int(model.predict([email_text])[0])
    if prediction == 1:
        label = "Potentially Phishing"
    elif prediction == 0:
        label = "Likely Legitimate"
    else:
        label = "Unknown"

    score = None
    if hasattr(model, "predict_proba") and hasattr(model, "classes_"):
        probabilities = model.predict_proba([email_text])[0]
        classes = list(model.classes_)
        if 1 in classes:
            score = float(probabilities[classes.index(1)])

    return prediction, label, score


# ==========================================================
# VISUAL DESIGN
# ==========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --ink:#e8eefc; --muted:#9eacc7; --line:rgba(148,163,184,.20); }
.stApp { background: radial-gradient(ellipse at 15% 0%, rgba(37,99,235,.18), transparent 34%), radial-gradient(ellipse at 95% 12%, rgba(13,148,136,.13), transparent 30%), #080d18; color:var(--ink); font-family:'DM Sans',sans-serif; }
[data-testid="stHeader"] { background:rgba(8,13,24,.78); }
.block-container { max-width: 1220px; padding-top: 2rem; padding-bottom: 3rem; }
h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.035em; }
h1 { color:#f8fbff; }
p, label, .stCaption { color:#b8c5dc; }
[data-testid="stMetric"] { background:linear-gradient(145deg,rgba(24,36,58,.88),rgba(13,21,36,.92)); border:1px solid var(--line); padding:17px 18px; border-radius:16px; box-shadow:0 10px 30px rgba(0,0,0,.14); }
[data-testid="stMetricLabel"] { color:#9eacc7 !important; }
[data-testid="stMetricValue"] { color:#f5f8ff !important; }
[data-testid="stFileUploader"] { background:rgba(20,31,51,.65); border:1px dashed rgba(96,165,250,.55); border-radius:18px; padding:12px; }
.stButton > button { border-radius:12px; min-height:46px; font-weight:700; border:1px solid rgba(96,165,250,.45); background:linear-gradient(100deg,#2563eb,#0f9f9a); color:white; transition:transform .15s ease, filter .15s ease; }
.stButton > button:hover { filter:brightness(1.12); transform:translateY(-1px); border-color:#7dd3fc; color:white; }
[data-testid="stExpander"] { border:1px solid var(--line); border-radius:14px; background:rgba(15,23,42,.48); }
[data-baseweb="tab-list"] { gap:8px; }
[data-baseweb="tab"] { border-radius:10px; padding:10px 18px; }
hr { border-color:var(--line); }
.hero { padding:28px 30px; border-radius:24px; background:linear-gradient(120deg,rgba(30,64,175,.35),rgba(15,118,110,.22)),rgba(15,23,42,.82); border:1px solid rgba(125,211,252,.22); margin-bottom:20px; box-shadow:0 20px 60px rgba(0,0,0,.18); }
.eyebrow { color:#7dd3fc; font-size:.76rem; font-weight:700; letter-spacing:.16em; text-transform:uppercase; }
.hero-title { font-family:'Space Grotesk',sans-serif; font-size:clamp(2rem,4vw,3.2rem); line-height:1.06; font-weight:700; color:#f8fbff; margin:.5rem 0 .8rem; }
.hero-copy { max-width:760px; color:#b9c8df; font-size:1rem; line-height:1.7; }
.pill { display:inline-block; padding:6px 10px; margin:4px 5px 0 0; border:1px solid rgba(125,211,252,.25); border-radius:999px; color:#dbeafe; background:rgba(15,23,42,.45); font-size:.78rem; }
.section-note { color:#9eacc7; margin-top:-.4rem; margin-bottom:1rem; }
[data-testid="stAlert"] { border-radius:14px; }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# PAGE HEADER
# ==========================================================

st.markdown("""
<div class="hero">
  <div class="eyebrow">SENTINEL • DIGITAL SAFETY LAB</div>
  <div class="hero-title">See the risk before you share.</div>
  <div class="hero-copy">A research prototype that helps you inspect screenshots for exposed personal information and screen email screenshots for possible phishing signals.</div>
  <div style="margin-top:14px"><span class="pill">◈ Mathematical PEI</span><span class="pill">◈ AI-assisted detection</span><span class="pill">◈ Email threat screening</span></div>
</div>
""", unsafe_allow_html=True)

with st.expander("⚙️ System Status · model availability", expanded=False):
    status1, status2, status3 = st.columns(3)
    with status1:
        st.success("Sensitive-information detector imported")
    with status2:
        if pei_model_loaded:
            st.success("PEI model loaded")
        else:
            st.error("PEI model not loaded")
            st.caption(pei_model_error)
    with status3:
        if phishing_model_loaded:
            st.success("Phishing model loaded")
        else:
            st.error("Phishing model not loaded")
            st.caption(phishing_model_error)


# ==========================================================
# SECTION 1 — PRIVACY EXPOSURE INDEX
# ==========================================================

st.divider()
st.header("Screenshot Privacy Scanner")
st.caption("Upload an image to find visible emails, phone numbers, URLs, and QR codes, then calculate a privacy exposure score.")

uploaded_file = st.file_uploader(
    "Upload a screenshot for privacy analysis",
    type=["png", "jpg", "jpeg"],
    key="privacy_screenshot",
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as exc:
        st.error("The uploaded file could not be opened as an image.")
        st.caption(str(exc))
        image = None

    if image is not None:
        st.image(image, caption="Uploaded screenshot", use_container_width=True)

        if st.button("🔍 Detect Sensitive Information", type="primary", key="run_privacy_detection"):
            with st.spinner("Analyzing screenshot..."):
                try:
                    results = detect_sensitive_information(image)
                except Exception as exc:
                    st.error("Sensitive-information detection failed.")
                    st.exception(exc)
                    results = None

            if results is not None:
                # Normalize detector output so missing optional keys do not crash the app.
                emails = results.get("emails", []) or []
                phones = results.get("phones", []) or []
                urls = results.get("urls", []) or []
                qr_detected = bool(results.get("qr_detected", False))
                qr_data = results.get("qr_data", "")
                ocr_text = results.get("text", "")

                email_count = len(emails)
                phone_count = len(phones)
                url_count = len(urls)
                qr_count = int(qr_detected)
                total_items = email_count + phone_count + url_count + qr_count

                st.success("Detection completed.")
                st.subheader("📊 Detection Summary")
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Emails", email_count)
                c2.metric("Phone numbers", phone_count)
                c3.metric("URLs", url_count)
                c4.metric("QR codes", qr_count)
                c5.metric("Total items", total_items)

                with st.expander("📋 View detected information", expanded=True):
                    if emails:
                        for item in emails:
                            st.write("📧 Email:", item)
                    if phones:
                        for item in phones:
                            st.write("📱 Phone:", item)
                    if urls:
                        for item in urls:
                            st.write("🌐 URL:", item)
                    if qr_detected:
                        st.write("🔳 QR code detected")
                        if qr_data:
                            st.write("QR content:", qr_data)
                    if not total_items:
                        st.info("No supported sensitive-information types were detected.")

                st.subheader("2. PEI Parameters")
                st.info(
                    "Area, visibility, and confidence are prototype inputs. "
                    "For research, define and validate how each value is measured."
                )
                p1, p2, p3 = st.columns(3)
                with p1:
                    area = st.slider(
                        "Relative area (A)", 0.0, 1.0, DEFAULT_AREA, 0.05,
                        key="pei_area",
                    )
                with p2:
                    visibility = st.slider(
                        "Visibility (V)", 0.0, 1.0, DEFAULT_VISIBILITY, 0.05,
                        key="pei_visibility",
                    )
                with p3:
                    confidence = st.slider(
                        "Detection confidence (C)", 0.0, 1.0, DEFAULT_CONFIDENCE, 0.05,
                        key="pei_confidence",
                    )

                st.subheader("3. Mathematical PEI Calculation")
                item_scores = []
                total_exposure = 0.0

                for item_type, values in (
                    ("Email", emails),
                    ("Phone", phones),
                    ("URL", urls),
                ):
                    for value in values:
                        score = calculate_item_score(item_type, area, visibility, confidence)
                        total_exposure += score
                        item_scores.append({
                            "type": item_type,
                            "value": str(value),
                            "weight": WEIGHTS[item_type],
                            "score": score,
                        })

                if qr_detected:
                    score = calculate_item_score("QR Code", area, visibility, confidence)
                    total_exposure += score
                    item_scores.append({
                        "type": "QR Code",
                        "value": str(qr_data or "Detected"),
                        "weight": WEIGHTS["QR Code"],
                        "score": score,
                    })

                detected_types = sum(
                    count > 0 for count in (email_count, phone_count, url_count, qr_count)
                )
                diversity_factor = 1.0 + max(detected_types - 1, 0) * 0.15
                adjusted_exposure = total_exposure * diversity_factor
                pei = max(0.0, min(100.0, 100.0 * adjusted_exposure / R_MAX))
                mathematical_class = classify_pei(pei)

                if item_scores:
                    st.markdown("**Individual item scores**")
                    for item in item_scores:
                        st.write(
                            f"**{item['type']}** — weight {item['weight']:.2f} × "
                            f"area {area:.2f} × visibility {visibility:.2f} × "
                            f"confidence {confidence:.2f} = **{item['score']:.4f}**"
                        )

                st.write(f"**Detected information types:** {detected_types}")
                st.write(f"**Diversity factor:** {diversity_factor:.2f}")
                r1, r2, r3 = st.columns(3)
                r1.metric("Raw exposure", f"{total_exposure:.4f}")
                r2.metric("PEI score", f"{pei:.2f} / 100")
                r3.metric("Sensitive items", total_items)

                if mathematical_class == "LOW":
                    st.success(f"🟢 Mathematical PEI: LOW — {pei:.2f}")
                elif mathematical_class == "MODERATE":
                    st.warning(f"🟡 Mathematical PEI: MODERATE — {pei:.2f}")
                else:
                    st.error(f"🔴 Mathematical PEI: HIGH — {pei:.2f}")

                st.subheader("4. Existing PEI Machine-Learning Prediction")
                if pei_model_loaded:
                    try:
                        ml_input = [[
                            email_count,
                            phone_count,
                            url_count,
                            qr_count,
                            total_exposure,
                            pei,
                        ]]
                        ml_prediction = str(pei_model.predict(ml_input)[0]).upper()
                        st.write("**ML predicted exposure level:**", ml_prediction)

                        left, right = st.columns(2)
                        left.metric("Mathematical PEI", mathematical_class)
                        right.metric("PEI model prediction", ml_prediction)

                        if mathematical_class == ml_prediction:
                            st.success("The mathematical classification and PEI model prediction agree.")
                        else:
                            st.warning("The mathematical classification and PEI model prediction differ.")
                    except Exception as exc:
                        st.error("The PEI model prediction failed. Check the model's expected input features.")
                        st.exception(exc)
                else:
                    st.warning("PEI model unavailable. The mathematical PEI result is still shown.")

                with st.expander("📘 View mathematical formulas"):
                    st.markdown("**Individual exposure score**")
                    st.latex(r"s_i = w_i \times A_i \times V_i \times C_i")
                    st.markdown("**Adjusted total exposure**")
                    st.latex(r"R = \left(\sum_i s_i\right) \times D")
                    st.markdown("**Privacy Exposure Index**")
                    st.latex(r"PEI = 100 \times \frac{R}{R_{\max}}")
                    st.write("D is the diversity factor; Rmax is the prototype reference value.")

                with st.expander("📝 View OCR text"):
                    st.code(ocr_text or "The detector did not return OCR text.")



# ==========================================================
# SECTION 2 — SCREENSHOT PHISHING EMAIL CHECKER
# ==========================================================

st.divider()
st.header("Email Threat Scanner")
st.caption("Upload a screenshot of a suspicious email. OCR extracts its text, and the phishing model evaluates that text for warning patterns.")
st.write(
    "Upload a screenshot of an email. The app extracts visible text with OCR, then passes "
    "that text to `phishing_model.pkl`. The model was trained on email text, not image pixels."
)
st.warning(
    "Use caution with real personal information. OCR can misread text, and a model result "
    "is not a guarantee. Verify suspicious messages through the official website or app "
    "using a known address—not links in the message."
)

phishing_upload = st.file_uploader(
    "Upload an email screenshot",
    type=["png", "jpg", "jpeg"],
    key="phishing_email_screenshot",
)

if phishing_upload is not None:
    try:
        phishing_image = Image.open(phishing_upload).convert("RGB")
        st.image(phishing_image, caption="Email screenshot", use_container_width=True)
    except Exception as exc:
        phishing_image = None
        st.error("The screenshot could not be opened.")
        st.caption(str(exc))

    if phishing_image is not None and st.button(
        "🛡️ Analyze Email for Phishing",
        type="primary",
        key="run_phishing_check",
    ):
        if not phishing_model_loaded:
            st.error("The phishing model could not be loaded.")
            st.caption(phishing_model_error)
        else:
            try:
                with st.spinner("Extracting text and checking the email..."):
                    extracted_text = extract_screenshot_text(phishing_image)
                    prediction, classification, phishing_score = predict_phishing(
                        extracted_text, phishing_model
                    )

                st.subheader("Extracted Email Text")
                st.text_area(
                    "OCR result (review for recognition errors)",
                    extracted_text,
                    height=180,
                    key="phishing_ocr_result",
                )

                st.subheader("Phishing Model Result")
                if prediction == 1:
                    st.error("⚠️ Potentially Phishing")
                    st.write(
                        "The model classified the extracted email text as phishing. "
                        "Do not click links or provide passwords or verification codes."
                    )
                elif prediction == 0:
                    st.success("Model result: Likely Legitimate")
                    st.write(
                        "This is only a model prediction. It does not prove the email is safe."
                    )
                else:
                    st.info("The model returned an unknown label.")

                st.write("**Classification:**", classification)
                if phishing_score is not None:
                    st.metric("Model phishing score", f"{phishing_score:.1%}")
                    st.caption(
                        "This score is the model's output and may not be a calibrated "
                        "real-world probability."
                    )
            except Exception as exc:
                st.error("The phishing analysis could not be completed.")
                st.caption(str(exc))


# ==========================================================
# FOOTER
# ==========================================================

st.divider()
st.caption(
    "Research prototype only. Validate the detector, PEI thresholds, model performance, "
    "and OCR pipeline on appropriate independent test data before making research claims."
)
