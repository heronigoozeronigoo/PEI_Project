
# ==========================================================
# PRIVACY & PHISHING SCREENSHOT ANALYZER
#
# Project 1: Mathematical Privacy Exposure Index (PEI)
# Project 2: Phishing Email Detection
# ==========================================================
 
import json
from pathlib import Path
 
import joblib
import pandas as pd
import pytesseract
import streamlit as st
from PIL import Image, ImageOps, ImageFilter
 
BASE_DIR = Path(__file__).resolve().parent
 
 
# ==========================================================
# PAGE SETTINGS  (must be the first Streamlit call)
# ==========================================================
 
st.set_page_config(
    page_title="PEI & Phishing Detector",
    page_icon="🔐",
    layout="wide"
)
 
 
# ==========================================================
# IMPORT PROJECT MODULES
# ==========================================================
 
try:
    from detector import (
        detect_sensitive_information,
        detect_emails,
        detect_urls,
    )
except Exception as error:
    st.error("The detector module could not be loaded.")
    st.code(str(error))
    st.info(
        "On Streamlit Cloud, add a file named packages.txt containing "
        "the line `tesseract-ocr`. On Windows, install Tesseract OCR."
    )
    st.stop()
 
from email_checker import check_email as check_email_address
from phishing_checker import check_email as classify_email_text
 
 
# ==========================================================
# LOAD MODELS (cached so they load only once)
# ==========================================================
 
@st.cache_resource
def load_pei_model():
    return joblib.load(BASE_DIR / "pei_model.pkl")
 
 
@st.cache_resource
def load_phishing_model():
    path = BASE_DIR / "phishing_model.pkl"
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(
            "phishing_model.pkl is missing or empty (0 bytes). "
            "Run train_phishing_model.py and upload the new file."
        )
    return joblib.load(path)
 
 
try:
    pei_model = load_pei_model()
    pei_model_error = None
except Exception as error:
    pei_model = None
    pei_model_error = str(error)
 
try:
    phishing_model = load_phishing_model()
    phishing_model_error = None
except Exception as error:
    phishing_model = None
    phishing_model_error = str(error)
 
 
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
DIVERSITY_STEP = 0.15
 
 
def classify_pei(pei):
    if pei <= 33.33:
        return "LOW"
    if pei <= 66.67:
        return "MODERATE"
    return "HIGH"
 
 
def show_level(level, text):
    if level == "LOW":
        st.success(f"🟢 {text}")
    elif level == "MODERATE":
        st.warning(f"🟡 {text}")
    else:
        st.error(f"🔴 {text}")
 
 
def calculate_pei(results, area, visibility, confidence):
    """Return the PEI numbers for one detection result."""
 
    items = []
 
    for email in results["emails"]:
        items.append(("Email", email))
    for phone in results["phones"]:
        items.append(("Phone", phone))
    for url in results["urls"]:
        items.append(("URL", url))
    if results["qr_detected"]:
        items.append(("QR Code", "Detected"))
 
    item_scores = []
    total_exposure = 0.0
 
    for kind, value in items:
        weight = WEIGHTS[kind]
        score = weight * area * visibility * confidence
        total_exposure += score
        item_scores.append(
            {"type": kind, "value": value, "weight": weight, "score": score}
        )
 
    counts = {
        "email_count": len(results["emails"]),
        "phone_count": len(results["phones"]),
        "url_count": len(results["urls"]),
        "qr_count": 1 if results["qr_detected"] else 0,
    }
 
    detected_types = sum(1 for c in counts.values() if c > 0)
    diversity = 1.0 + max(detected_types - 1, 0) * DIVERSITY_STEP
 
    adjusted = total_exposure * diversity
    pei = max(0.0, min(100.0, 100 * adjusted / R_MAX))
 
    return {
        "counts": counts,
        "item_scores": item_scores,
        "total_items": sum(counts.values()),
        "total_exposure": total_exposure,
        "detected_types": detected_types,
        "diversity": diversity,
        "pei": pei,
        "level": classify_pei(pei),
    }
 
 
# ==========================================================
# PHISHING HELPERS
# ==========================================================
 
def ocr_for_classifier(image):
    """
    One clean OCR pass for the phishing model.
    (detector.run_ocr joins 12 passes, which repeats the same
    text many times and does not look like a real email.)
    """
    image = image.convert("RGB")
    enlarged = image.resize(
        (image.width * 2, image.height * 2),
        Image.Resampling.LANCZOS
    )
    gray = ImageOps.grayscale(enlarged).filter(ImageFilter.SHARPEN)
    return pytesseract.image_to_string(gray, config="--psm 6").strip()
 
 
def show_phishing_result(email_text):
    try:
        result = classify_email_text(email_text, model=phishing_model)
    except Exception as error:
        st.error("The phishing check could not be completed.")
        st.caption(str(error))
        return
 
    st.subheader("🤖 Phishing Model Result")
 
    if result["prediction"] == 1:
        st.error(f"🚨 {result['classification']}")
    else:
        st.success(f"✅ {result['classification']}")
 
    if "phishing_model_score" in result:
        score = result["phishing_model_score"]
        st.metric("Model phishing score", f"{score * 100:.1f}%")
        st.progress(min(max(score, 0.0), 1.0))
        st.caption(
            "This is the model's score, not a calibrated probability "
            "of real-world risk."
        )
 
    # ---- Extra signals found inside the text ----
    emails = detect_emails(email_text)
    urls = detect_urls(email_text)
 
    st.subheader("🔎 Extra Signals in the Email")
 
    col1, col2 = st.columns(2)
 
    with col1:
        st.write("**Email addresses found**")
        if emails:
            for address in emails[:5]:
                check = check_email_address(address)
                if check["status"] == "Invalid format":
                    st.write(f"❌ `{address}` — invalid format")
                elif check["status"] == "Likely legitimate":
                    st.write(f"✅ `{address}` — domain can receive email")
                elif check["status"] == "Potentially suspicious":
                    st.write(f"⚠️ `{address}` — no MX record confirmed")
                else:
                    st.write(f"ℹ️ `{address}` — unable to verify")
        else:
            st.write("None")
 
    with col2:
        st.write("**Links found**")
        if urls:
            for url in urls[:10]:
                st.write(f"🌐 `{url}`")
            st.caption("Do not open links from suspicious emails.")
        else:
            st.write("None")
 
    st.info(
        "A valid sender domain does not prove an email is safe, and a "
        "model result is not a guarantee. When unsure, contact the "
        "sender through an official channel."
    )
 
 
# ==========================================================
# TITLE
# ==========================================================
 
st.title("🔐 Privacy & Phishing Screenshot Analyzer")
 
st.write(
    "Project 1: Mathematical Privacy Exposure Index (PEI). "
    "Project 2: Phishing email detection."
)
 
col1, col2, col3 = st.columns(3)
with col1:
    st.success("✅ Detector Ready")
with col2:
    if pei_model is not None:
        st.success("✅ PEI ML Model Loaded")
    else:
        st.error("❌ PEI ML Model Not Loaded")
with col3:
    if phishing_model is not None:
        st.success("✅ Phishing Model Loaded")
    else:
        st.error("❌ Phishing Model Not Loaded")
 
if pei_model_error:
    st.caption(f"PEI model error: {pei_model_error}")
if phishing_model_error:
    st.caption(f"Phishing model error: {phishing_model_error}")
 
st.divider()
 
tab_pei, tab_phish, tab_email = st.tabs(
    [
        "🔐 Privacy Exposure (PEI)",
        "🎣 Phishing Detector",
        "📧 Email Address Checker",
    ]
)
 
 
# ==========================================================
# TAB 1: PRIVACY EXPOSURE INDEX
# ==========================================================
 
with tab_pei:
 
    st.header("Step 1: Upload Screenshot")
 
    pei_file = st.file_uploader(
        "Choose a screenshot",
        type=["png", "jpg", "jpeg"],
        key="pei_upload"
    )
 
    if pei_file is not None:
 
        pei_image = Image.open(pei_file).convert("RGB")
        st.image(pei_image, caption="Uploaded Screenshot", width="stretch")
 
        st.header("Step 2: Detect Sensitive Information")
 
        if st.button("🔍 Detect Sensitive Information", key="pei_detect"):
            with st.spinner("Analyzing screenshot..."):
                try:
                    st.session_state["pei_results"] = (
                        detect_sensitive_information(pei_image)
                    )
                    st.session_state["pei_file_id"] = pei_file.file_id
                except Exception as error:
                    st.error("Detection failed.")
                    st.code(str(error))
 
        # Results are kept in session_state so moving the sliders
        # does not make them disappear.
        results = st.session_state.get("pei_results")
        same_file = st.session_state.get("pei_file_id") == pei_file.file_id
 
        if results is not None and same_file:
 
            counts_preview = {
                "📧 Email": len(results["emails"]),
                "📱 Phone": len(results["phones"]),
                "🌐 URL": len(results["urls"]),
                "🔳 QR Code": 1 if results["qr_detected"] else 0,
            }
 
            st.subheader("📊 Detection Summary")
            cols = st.columns(5)
            for col, (label, value) in zip(cols, counts_preview.items()):
                col.metric(label, value)
            cols[4].metric("Total", sum(counts_preview.values()))
 
            st.subheader("📋 Detected Information")
            for email in results["emails"]:
                st.write("📧 Email:", email)
            for phone in results["phones"]:
                st.write("📱 Phone:", phone)
            for url in results["urls"]:
                st.write("🌐 URL:", url)
            if results["qr_detected"]:
                st.write("🔳 QR Code: Detected")
                if results["qr_data"]:
                    st.write("QR Content:", results["qr_data"])
            if sum(counts_preview.values()) == 0:
                st.write("No sensitive information detected.")
 
            # ---- Step 3 ----
            st.divider()
            st.header("Step 3: PEI Parameters")
            st.info(
                "These are prototype values. They can be replaced with "
                "validated measurements during your research."
            )
 
            area = st.slider("Relative Area (A)", 0.0, 1.0,
                             DEFAULT_AREA, 0.05)
            visibility = st.slider("Visibility (V)", 0.0, 1.0,
                                   DEFAULT_VISIBILITY, 0.05)
            confidence = st.slider("Detection Confidence (C)", 0.0, 1.0,
                                   DEFAULT_CONFIDENCE, 0.05)
 
            # ---- Step 4 ----
            calc = calculate_pei(results, area, visibility, confidence)
 
            st.divider()
            st.header("Step 4: Mathematical PEI Calculation")
 
            if calc["item_scores"]:
                st.subheader("📐 Individual Exposure Scores")
                for item in calc["item_scores"]:
                    st.write(
                        f"**{item['type']}** — "
                        f"Weight: {item['weight']:.2f} × "
                        f"Area: {area:.2f} × "
                        f"Visibility: {visibility:.2f} × "
                        f"Confidence: {confidence:.2f} "
                        f"= **{item['score']:.4f}**"
                    )
 
            st.write(f"**Detected Information Types:** {calc['detected_types']}")
            st.write(f"**Diversity Factor:** {calc['diversity']:.2f}")
 
            c1, c2, c3 = st.columns(3)
            c1.metric("Raw Exposure", f"{calc['total_exposure']:.4f}")
            c2.metric("PEI Score", f"{calc['pei']:.2f} / 100")
            c3.metric("Sensitive Items", calc["total_items"])
 
            st.subheader("📐 Mathematical PEI Classification")
            show_level(calc["level"], f"{calc['level']} — PEI = {calc['pei']:.2f}")
 
            # ---- Step 5 ----
            st.divider()
            st.header("Step 5: Machine Learning Prediction")
 
            if pei_model is None:
                st.warning("ML model is not available.")
            else:
                try:
                    ml_input = pd.DataFrame([{
                        **calc["counts"],
                        "raw_exposure": calc["total_exposure"],
                        "pei": calc["pei"],
                    }])
                    ml_prediction = pei_model.predict(ml_input)[0]
 
                    st.subheader("🤖 ML Predicted Exposure Level")
                    show_level(ml_prediction, ml_prediction)
 
                    st.subheader("🔎 PEI vs ML Result")
                    a, b = st.columns(2)
                    a.write("**Mathematical PEI:**")
                    a.write(calc["level"])
                    b.write("**ML Prediction:**")
                    b.write(ml_prediction)
 
                    if calc["level"] == ml_prediction:
                        st.success("✅ The mathematical PEI and ML prediction agree.")
                    else:
                        st.warning("⚠️ The mathematical PEI and ML prediction differ.")
 
                except Exception as error:
                    st.error("ML prediction failed.")
                    st.code(str(error))
 
            with st.expander("📘 View Mathematical Formulas"):
                st.write("Individual Exposure Score:")
                st.latex(r"s_i = w_i \times A_i \times V_i \times C_i")
                st.write("Adjusted Total Exposure:")
                st.latex(r"R = \left(\sum s_i\right) \times D")
                st.write("Privacy Exposure Index:")
                st.latex(r"PEI = 100 \times \frac{R}{R_{max}}")
                st.write("D = Diversity Factor, Rmax = Maximum reference exposure")
 
            with st.expander("📝 View OCR Text"):
                st.code(results["text"])
 
 
# ==========================================================
# TAB 2: PHISHING DETECTOR
# ==========================================================
 
with tab_phish:
 
    st.header("Phishing Email Detector")
    st.write(
        "Upload a screenshot of an email, or paste the email text. "
        "The app reads the text and checks it with the trained model."
    )
 
    if phishing_model is None:
        st.warning(
            "The phishing model is not loaded. Run "
            "`train_phishing_model.py`, then upload the new "
            "`phishing_model.pkl` next to this app."
        )
 
    mode = st.radio(
        "Input type",
        ["📷 Screenshot", "✍️ Paste text"],
        horizontal=True,
        key="phish_mode"
    )
 
    email_text = ""
 
    if mode == "📷 Screenshot":
        phish_file = st.file_uploader(
            "Upload an email screenshot",
            type=["png", "jpg", "jpeg"],
            key="phish_upload"
        )
        if phish_file is not None:
            phish_image = Image.open(phish_file).convert("RGB")
            st.image(phish_image, caption="Email Screenshot", width="stretch")
 
            if st.button("🔍 Analyze Email", key="phish_analyze_img"):
                with st.spinner("Reading text from screenshot..."):
                    email_text = ocr_for_classifier(phish_image)
                if not email_text:
                    st.warning("No readable text was found in the screenshot.")
    else:
        pasted = st.text_area(
            "Paste the email (subject and body)",
            height=250,
            key="phish_text"
        )
        if st.button("🔍 Analyze Email", key="phish_analyze_txt"):
            email_text = pasted.strip()
            if not email_text:
                st.warning("Please paste some email text first.")
 
    if email_text and phishing_model is not None:
        with st.expander("📝 Text used for analysis"):
            st.code(email_text)
        show_phishing_result(email_text)
 
    # ---- Model evaluation info ----
    eval_path = BASE_DIR / "phishing_model_evaluation.json"
    if eval_path.exists():
        with st.expander("📈 Model evaluation (held-out test set)"):
            try:
                ev = json.loads(eval_path.read_text(encoding="utf-8"))
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Accuracy", f"{ev['accuracy'] * 100:.1f}%")
                m2.metric("Phishing precision", f"{ev['phishing_precision'] * 100:.1f}%")
                m3.metric("Phishing recall", f"{ev['phishing_recall'] * 100:.1f}%")
                m4.metric("False-positive rate", f"{ev['false_positive_rate'] * 100:.1f}%")
                st.caption(
                    f"{ev['testing_rows']} test emails from "
                    f"{ev['dataset']}. Scores this close to perfect usually "
                    "mean the dataset is very uniform (for example, "
                    "template-generated). Real-world emails will be harder."
                )
            except Exception as error:
                st.caption(f"Could not read evaluation file: {error}")
 
 
# ==========================================================
# TAB 3: EMAIL ADDRESS CHECKER
# ==========================================================
 
with tab_email:
 
    st.header("📧 Email Address Checker")
    st.write(
        "Check an email address's format and domain mail-server "
        "configuration. This does not guarantee that the mailbox exists "
        "or that the sender is trustworthy."
    )
 
    email_input = st.text_input(
        "Enter an email address",
        placeholder="example@domain.com",
        key="email_authenticity_input"
    )
 
    if st.button("🔎 Check Email", key="check_email_button"):
 
        if not email_input.strip():
            st.warning("Please enter an email address.")
        else:
            with st.spinner("Checking email format and domain..."):
                try:
                    r = check_email_address(email_input)
 
                    st.subheader("Email Check Results")
                    st.write("**Email:**", r["email"])
 
                    if r["status"] == "Invalid format":
                        st.error("❌ Invalid email format")
                    elif r["status"] == "Likely legitimate":
                        st.success("✅ Likely legitimate domain configuration")
                    elif r["status"] == "Potentially suspicious":
                        st.warning("⚠️ Potentially suspicious — further verification needed")
                    else:
                        st.info("ℹ️ Unable to verify")
 
                    st.write("**Format valid:**", "Yes" if r["format_valid"] else "No")
                    if r["domain"]:
                        st.write("**Domain:**", r["domain"])
                    if r["format_valid"]:
                        st.write("**MX record confirmed:**",
                                 "Yes" if r["has_mx_record"] else "No")
                    for detail in r["details"]:
                        st.write("•", detail)
 
                except Exception as error:
                    st.error("The email check could not be completed.")
                    st.caption(str(error))
