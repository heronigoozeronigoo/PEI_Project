import streamlit as st
from PIL import Image
import joblib

from detector import detect_sensitive_information


# ==========================================================
# PAGE SETTINGS
# ==========================================================

st.set_page_config(
    page_title="PEI Research Prototype",
    page_icon="🔐",
    layout="wide"
)


# ==========================================================
# LOAD TRAINED ML MODEL
# ==========================================================

try:

    ml_model = joblib.load(
        "pei_model.pkl"
    )

    model_loaded = True

except Exception as error:

    ml_model = None
    model_loaded = False

    st.error(
        "ML model could not be loaded."
    )

    st.code(
        str(error)
    )


# ==========================================================
# SENSITIVITY WEIGHTS
# ==========================================================

WEIGHTS = {
    "Email": 0.15,
    "Phone": 0.15,
    "URL": 0.10,
    "QR Code": 0.20
}


# ==========================================================
# PEI SETTINGS
# ==========================================================

R_MAX = 0.60

DEFAULT_AREA = 1.00
DEFAULT_VISIBILITY = 1.00
DEFAULT_CONFIDENCE = 0.90


# ==========================================================
# ITEM SCORE
# ==========================================================

def calculate_item_score(
    information_type,
    area,
    visibility,
    confidence
):

    weight = WEIGHTS[
        information_type
    ]

    score = (
        weight
        * area
        * visibility
        * confidence
    )

    return score


# ==========================================================
# PEI CLASSIFICATION
# ==========================================================

def classify_pei(pei):

    if pei <= 33.33:

        return "LOW"

    elif pei <= 66.67:

        return "MODERATE"

    else:

        return "HIGH"


# ==========================================================
# TITLE
# ==========================================================

st.title(
    "🔐 Mathematical Privacy Exposure Index"
)

st.write(
    "Machine Learning-Assisted Detection, Mathematical PEI, "
    "and Privacy Classification"
)

st.divider()


# ==========================================================
# SYSTEM STATUS
# ==========================================================

st.subheader(
    "⚙️ System Status"
)

col1, col2 = st.columns(2)


with col1:

    st.success(
        "✅ Sensitive Information Detector Ready"
    )


with col2:

    if model_loaded:

        st.success(
            "✅ ML Model Loaded"
        )

    else:

        st.error(
            "❌ ML Model Not Loaded"
        )


st.divider()


# ==========================================================
# STEP 1 — UPLOAD SCREENSHOT
# ==========================================================

st.header(
    "Step 1: Upload Screenshot"
)

uploaded_file = st.file_uploader(
    "Choose a screenshot",
    type=[
        "png",
        "jpg",
        "jpeg"
    ]
)


# ==========================================================
# PROCESS SCREENSHOT
# ==========================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    st.success(
        "Screenshot uploaded successfully!"
    )


    st.image(
        image,
        caption="Uploaded Screenshot",
        use_container_width=True
    )


    st.divider()


    # ======================================================
    # STEP 2 — DETECTION
    # ======================================================

    st.header(
        "Step 2: Sensitive Information Detection"
    )


    if st.button(
        "🔍 Detect Sensitive Information"
    ):

        with st.spinner(
            "Analyzing screenshot..."
        ):

            try:

                results = (
                    detect_sensitive_information(
                        image
                    )
                )

            except Exception as error:

                st.error(
                    "Detection failed."
                )

                st.code(
                    str(error)
                )

                st.stop()


        st.success(
            "Detection completed!"
        )


        # ==================================================
        # COUNT DETECTED INFORMATION
        # ==================================================

        email_count = len(
            results["emails"]
        )

        phone_count = len(
            results["phones"]
        )

        url_count = len(
            results["urls"]
        )

        qr_count = (
            1
            if results["qr_detected"]
            else 0
        )


        total_items = (
            email_count
            + phone_count
            + url_count
            + qr_count
        )


        # ==================================================
        # DETECTION SUMMARY
        # ==================================================

        st.subheader(
            "📊 Detection Summary"
        )


        col1, col2, col3, col4, col5 = (
            st.columns(5)
        )


        with col1:

            st.metric(
                "📧 Email",
                email_count
            )


        with col2:

            st.metric(
                "📱 Phone",
                phone_count
            )


        with col3:

            st.metric(
                "🌐 URL",
                url_count
            )


        with col4:

            st.metric(
                "🔳 QR Code",
                qr_count
            )


        with col5:

            st.metric(
                "Total",
                total_items
            )


        # ==================================================
        # DISPLAY DETECTED INFORMATION
        # ==================================================

        st.subheader(
            "📋 Detected Information"
        )


        if results["emails"]:

            for email in results["emails"]:

                st.write(
                    "📧 Email:",
                    email
                )


        if results["phones"]:

            for phone in results["phones"]:

                st.write(
                    "📱 Phone:",
                    phone
                )


        if results["urls"]:

            for url in results["urls"]:

                st.write(
                    "🌐 URL:",
                    url
                )


        if results["qr_detected"]:

            st.write(
                "🔳 QR Code: Detected"
            )

            if results["qr_data"]:

                st.write(
                    "QR Content:",
                    results["qr_data"]
                )


        # ==================================================
        # STEP 3 — PEI PARAMETERS
        # ==================================================

        st.divider()

        st.header(
            "Step 3: PEI Parameters"
        )


        st.info(
            "These are prototype values. "
            "They can be replaced with validated "
            "measurements during your research."
        )


        area = st.slider(
            "Relative Area (A)",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_AREA,
            step=0.05
        )


        visibility = st.slider(
            "Visibility (V)",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_VISIBILITY,
            step=0.05
        )


        confidence = st.slider(
            "Detection Confidence (C)",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_CONFIDENCE,
            step=0.05
        )


        # ==================================================
        # STEP 4 — MATHEMATICAL PEI
        # ==================================================

        st.divider()

        st.header(
            "Step 4: Mathematical PEI Calculation"
        )


        total_exposure = 0.0

        item_scores = []


        # --------------------------------------------------
        # EMAIL
        # --------------------------------------------------

        for email in results["emails"]:

            score = calculate_item_score(
                "Email",
                area,
                visibility,
                confidence
            )

            total_exposure += score

            item_scores.append(
                {
                    "type": "Email",
                    "value": email,
                    "weight": WEIGHTS["Email"],
                    "score": score
                }
            )


        # --------------------------------------------------
        # PHONE
        # --------------------------------------------------

        for phone in results["phones"]:

            score = calculate_item_score(
                "Phone",
                area,
                visibility,
                confidence
            )

            total_exposure += score

            item_scores.append(
                {
                    "type": "Phone",
                    "value": phone,
                    "weight": WEIGHTS["Phone"],
                    "score": score
                }
            )


        # --------------------------------------------------
        # URL
        # --------------------------------------------------

        for url in results["urls"]:

            score = calculate_item_score(
                "URL",
                area,
                visibility,
                confidence
            )

            total_exposure += score

            item_scores.append(
                {
                    "type": "URL",
                    "value": url,
                    "weight": WEIGHTS["URL"],
                    "score": score
                }
            )


        # --------------------------------------------------
        # QR CODE
        # --------------------------------------------------

        if results["qr_detected"]:

            score = calculate_item_score(
                "QR Code",
                area,
                visibility,
                confidence
            )

            total_exposure += score

            item_scores.append(
                {
                    "type": "QR Code",
                    "value": "Detected",
                    "weight": WEIGHTS["QR Code"],
                    "score": score
                }
            )


        # ==================================================
        # DIVERSITY FACTOR
        # ==================================================

        detected_types = 0


        if email_count > 0:

            detected_types += 1


        if phone_count > 0:

            detected_types += 1


        if url_count > 0:

            detected_types += 1


        if qr_count > 0:

            detected_types += 1


        diversity_factor = (
            1.0
            + (
                max(
                    detected_types - 1,
                    0
                )
                * 0.15
            )
        )


        adjusted_exposure = (
            total_exposure
            * diversity_factor
        )


        # ==================================================
        # CALCULATE PEI
        # ==================================================

        pei = (
            100
            * adjusted_exposure
            / R_MAX
        )


        # Keep between 0 and 100

        pei = max(
            0,
            min(
                pei,
                100
            )
        )


        mathematical_class = (
            classify_pei(
                pei
            )
        )


        # ==================================================
        # DISPLAY ITEM SCORES
        # ==================================================

        if item_scores:

            st.subheader(
                "📐 Individual Exposure Scores"
            )


            for item in item_scores:

                st.write(
                    f"**{item['type']}** — "
                    f"Weight: {item['weight']:.2f} × "
                    f"Area: {area:.2f} × "
                    f"Visibility: {visibility:.2f} × "
                    f"Confidence: {confidence:.2f} "
                    f"= **{item['score']:.4f}**"
                )


        # ==================================================
        # DIVERSITY INFORMATION
        # ==================================================

        st.write(
            f"**Detected Information Types:** "
            f"{detected_types}"
        )


        st.write(
            f"**Diversity Factor:** "
            f"{diversity_factor:.2f}"
        )


        # ==================================================
        # FINAL MATHEMATICAL PEI
        # ==================================================

        st.divider()

        st.header(
            "📊 Mathematical PEI Results"
        )


        col1, col2, col3 = (
            st.columns(3)
        )


        with col1:

            st.metric(
                "Raw Exposure",
                f"{total_exposure:.4f}"
            )


        with col2:

            st.metric(
                "PEI Score",
                f"{pei:.2f} / 100"
            )


        with col3:

            st.metric(
                "Sensitive Items",
                total_items
            )


        # ==================================================
        # MATHEMATICAL CLASSIFICATION
        # ==================================================

        st.subheader(
            "📐 Mathematical PEI Classification"
        )


        if mathematical_class == "LOW":

            st.success(
                f"🟢 LOW — PEI = {pei:.2f}"
            )


        elif mathematical_class == "MODERATE":

            st.warning(
                f"🟡 MODERATE — PEI = {pei:.2f}"
            )


        else:

            st.error(
                f"🔴 HIGH — PEI = {pei:.2f}"
            )


        # ==================================================
        # STEP 5 — MACHINE LEARNING
        # ==================================================

        st.divider()

        st.header(
            "Step 5: Machine Learning Prediction"
        )


        if model_loaded:

            # ------------------------------------------------
            # Prepare ML input
            # ------------------------------------------------

            ml_input = [[
                email_count,
                phone_count,
                url_count,
                qr_count,
                total_exposure,
                pei
            ]]


            # ------------------------------------------------
            # Make prediction
            # ------------------------------------------------

            try:

                ml_prediction = (
                    ml_model.predict(
                        ml_input
                    )[0]
                )


                st.success(
                    "🤖 ML prediction completed!"
                )


                # ------------------------------------------------
                # Display prediction
                # ------------------------------------------------

                st.subheader(
                    "🤖 ML Predicted Exposure Level"
                )


                if ml_prediction == "LOW":

                    st.success(
                        "🟢 LOW"
                    )


                elif ml_prediction == "MODERATE":

                    st.warning(
                        "🟡 MODERATE"
                    )


                else:

                    st.error(
                        "🔴 HIGH"
                    )


                # ------------------------------------------------
                # Compare ML and mathematical PEI
                # ------------------------------------------------

                st.subheader(
                    "🔎 PEI vs ML Result"
                )


                comparison_col1, comparison_col2 = (
                    st.columns(2)
                )


                with comparison_col1:

                    st.write(
                        "**Mathematical PEI:**"
                    )

                    st.write(
                        mathematical_class
                    )


                with comparison_col2:

                    st.write(
                        "**ML Prediction:**"
                    )

                    st.write(
                        ml_prediction
                    )


                # ------------------------------------------------
                # Agreement
                # ------------------------------------------------

                if (
                    mathematical_class
                    == ml_prediction
                ):

                    st.success(
                        "✅ The mathematical PEI "
                        "and ML prediction agree."
                    )

                else:

                    st.warning(
                        "⚠️ The mathematical PEI "
                        "and ML prediction differ."
                    )


            except Exception as error:

                st.error(
                    "ML prediction failed."
                )

                st.code(
                    str(error)
                )


        else:

            st.warning(
                "ML model is not available."
            )


        # ==================================================
        # FORMULAS
        # ==================================================

        st.divider()

        with st.expander(
            "📘 View Mathematical Formulas"
        ):

            st.write(
                "Individual Exposure Score:"
            )

            st.latex(
                r"s_i = w_i \times A_i \times V_i \times C_i"
            )


            st.write(
                "Adjusted Total Exposure:"
            )

            st.latex(
                r"R = \left(\sum s_i\right) \times D"
            )


            st.write(
                "Privacy Exposure Index:"
            )

            st.latex(
                r"PEI = 100 \times \frac{R}{R_{max}}"
            )


            st.write(
                "Where:"
            )

            st.write(
                "D = Diversity Factor"
            )


            st.write(
                "Rmax = Maximum reference exposure"
            )


        # ==================================================
        # OCR TEXT
        # ==================================================

        with st.expander(
            "📝 View OCR Text"
        ):

            st.code(
                results["text"]
            )


# ==========================================================
# STEP 7 — SCREENSHOT-BASED PHISHING EMAIL CHECKER
# ==========================================================

import re
from pathlib import Path

import pytesseract
from PIL import Image
import streamlit as st

# Separate model for phishing detection.
# Do NOT reuse pei_model.pkl for this task.
PHISHING_MODEL_PATH = Path("phishing_model.pkl")


def extract_email_text(image):
    """Extract visible email text from a screenshot using OCR."""
    return pytesseract.image_to_string(image).strip()


def analyze_phishing_features(text):
    """
    Extract simple, explainable phishing indicators.
    These indicators are heuristics, not proof of fraud.
    """
    text_lower = text.lower()

    urgency_terms = [
        "urgent", "immediately", "act now", "expires today",
        "within 24 hours", "suspended", "final warning",
        "account will be closed", "verify now"
    ]

    credential_terms = [
        "password", "login", "log in", "sign in",
        "verify your account", "confirm your identity",
        "one-time password", "otp", "security code"
    ]

    threat_terms = [
        "account suspended", "account blocked",
        "permanently disabled", "legal action",
        "unauthorized activity", "unusual activity",
        "account will be closed"
    ]

    money_terms = [
        "prize", "winner", "claim your reward",
        "free money", "payment required", "transfer money",
        "bank details", "claim your gift"
    ]

    # Look for visible URLs in the OCR text.
    urls = re.findall(
        r"(?:https?://|www\.)[^\s<>\"']+",
        text,
        flags=re.IGNORECASE
    )

    urgency = int(any(term in text_lower for term in urgency_terms))
    credentials = int(
        any(term in text_lower for term in credential_terms)
    )
    threat = int(any(term in text_lower for term in threat_terms))
    money = int(any(term in text_lower for term in money_terms))
    link_present = int(len(urls) > 0)

    # Normalize the number of visible URLs to a value between 0 and 1.
    url_count_score = min(len(urls) / 3, 1.0)

    features = {
        "urgency": urgency,
        "credentials": credentials,
        "link_present": link_present,
        "threat": threat,
        "money": money,
        "url_count_score": url_count_score,
        "urls": urls
    }

    return features


def calculate_pri(features):
    """
    Phishing Risk Index (PRI), ranging from 0 to 100.

    PRI = 100 * (
        0.20U + 0.25C + 0.15L
        + 0.20T + 0.10M + 0.10N
    )

    U = urgency indicator
    C = credential-request indicator
    L = visible-link indicator
    T = threat indicator
    M = money/reward indicator
    N = normalized URL count

    Weights are provisional and require empirical validation.
    """
    pri = 100 * (
        0.20 * features["urgency"]
        + 0.25 * features["credentials"]
        + 0.15 * features["link_present"]
        + 0.20 * features["threat"]
        + 0.10 * features["money"]
        + 0.10 * features["url_count_score"]
    )

    return round(pri, 2)


def classify_pri(pri):
    """Provisional screening categories, not validated probabilities."""
    if pri < 30:
        return "LOWER RISK"
    elif pri < 60:
        return "NEEDS REVIEW"
    return "HIGH PHISHING RISK"


def load_phishing_model():
    """Load a separately trained phishing model when available."""
    if not PHISHING_MODEL_PATH.exists():
        return None

    try:
        import joblib
        return joblib.load(PHISHING_MODEL_PATH)
    except Exception:
        return None


# ----------------------------------------------------------
# USER INTERFACE
# ----------------------------------------------------------

st.divider()
st.header("📧 Screenshot-Based Phishing Email Checker")

st.write(
    "Upload a screenshot of an email to extract its visible text, "
    "identify potential warning signs, and calculate a mathematical "
    "Phishing Risk Index (PRI)."
)

st.info(
    "A low score does not guarantee that an email is safe. "
    "The checker cannot confirm sender identity from a screenshot alone."
)

email_screenshot = st.file_uploader(
    "Upload an email screenshot",
    type=["png", "jpg", "jpeg"],
    key="phishing_email_screenshot"
)

if email_screenshot is not None:
    try:
        screenshot = Image.open(email_screenshot).convert("RGB")

        st.subheader("Uploaded Screenshot")
        st.image(
            screenshot,
            caption="Email screenshot for analysis",
            use_container_width=True
        )

        if st.button(
            "🔎 Analyze Phishing Risk",
            key="analyze_phishing_screenshot"
        ):
            with st.spinner("Extracting text and analyzing warning signs..."):
                email_text = extract_email_text(screenshot)

            if not email_text:
                st.error(
                    "No readable text was detected. Try a clearer screenshot."
                )
            else:
                features = analyze_phishing_features(email_text)
                pri = calculate_pri(features)
                category = classify_pri(pri)

                st.subheader("📊 Phishing Risk Results")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Phishing Risk Index", f"{pri}/100")

                with col2:
                    st.metric("Screening Category", category)

                st.progress(int(pri))

                st.caption(
                    "The score is a heuristic screening index, "
                    "not the probability that the email is a scam."
                )

                if category == "LOWER RISK":
                    st.success(
                        "Fewer of the selected warning signs were detected. "
                        "This does not establish that the email is legitimate."
                    )
                elif category == "NEEDS REVIEW":
                    st.warning(
                        "Some warning signs were detected. Verify the sender "
                        "through an official channel before taking action."
                    )
                else:
                    st.error(
                        "Multiple or strongly weighted warning indicators "
                        "were detected. Do not click links or provide "
                        "passwords, OTPs, or financial information until verified."
                    )

                # ------------------------------------------
                # FEATURE BREAKDOWN
                # ------------------------------------------

                st.subheader("🧮 Mathematical Feature Breakdown")

                breakdown = [
                    ("Urgency wording", features["urgency"], 0.20),
                    ("Credential-related wording", features["credentials"], 0.25),
                    ("Visible URL detected", features["link_present"], 0.15),
                    ("Threat-related wording", features["threat"], 0.20),
                    ("Money or reward wording", features["money"], 0.10),
                    ("URL count score", features["url_count_score"], 0.10),
                ]

                for name, value, weight in breakdown:
                    contribution = 100 * weight * value

                    st.write(
                        f"**{name}:** {value:.2f} "
                        f"× {weight:.2f} × 100 "
                        f"= {contribution:.2f} points"
                    )

                st.code(
                    "PRI = 100 × ("
                    "0.20U + 0.25C + 0.15L + "
                    "0.20T + 0.10M + 0.10N)"
                )

                # ------------------------------------------
                # DETECTED URLS
                # ------------------------------------------

                st.subheader("🔗 Visible Links")

                if features["urls"]:
                    st.warning(
                        "Review these links carefully. They were extracted "
                        "from the screenshot and have not been opened."
                    )

                    for url in features["urls"]:
                        st.code(url)
                else:
                    st.write(
                        "No URLs matching the supported patterns were detected. "
                        "Links in images or unusual formats may be missed."
                    )

    ```python
# ==========================================================
# STEP 7 — SCREENSHOT-BASED PHISHING EMAIL CHECKER
# ==========================================================

import re
from pathlib import Path

import pytesseract
from PIL import Image
import streamlit as st
import joblib


# ----------------------------------------------------------
# LOAD THE PHISHING MODEL
# ----------------------------------------------------------

PHISHING_MODEL_PATH = Path(__file__).parent / "phishing_model.pkl"


@st.cache_resource
def load_phishing_model():
    """Load the separately trained phishing text-classification model."""
    if not PHISHING_MODEL_PATH.exists():
        return None

    try:
        return joblib.load(PHISHING_MODEL_PATH)
    except Exception as error:
        st.error("Unable to load the phishing model.")
        st.code(str(error))
        return None


# ----------------------------------------------------------
# OCR TEXT EXTRACTION
# ----------------------------------------------------------

def extract_email_text(image):
    """Extract readable email text from an uploaded screenshot."""
    return pytesseract.image_to_string(image).strip()


# ----------------------------------------------------------
# PHISHING INDICATORS
# ----------------------------------------------------------

def analyze_phishing_features(text):
    """Identify selected warning signs in the extracted email text."""

    text_lower = text.lower()

    urgency_terms = [
        "urgent", "immediately", "act now", "expires today",
        "within 24 hours", "suspended", "final warning",
        "account will be closed", "verify now"
    ]

    credential_terms = [
        "password", "login", "log in", "sign in",
        "verify your account", "confirm your identity",
        "one-time password", "otp", "security code"
    ]

    threat_terms = [
        "account suspended", "account blocked",
        "permanently disabled", "legal action",
        "unauthorized activity", "unusual activity",
        "account will be closed"
    ]

    money_terms = [
        "prize", "winner", "claim your reward",
        "free money", "payment required", "transfer money",
        "bank details", "claim your gift"
    ]

    urls = re.findall(
        r"(?:https?://|www\.)[^\s<>\"']+",
        text,
        flags=re.IGNORECASE
    )

    features = {
        "urgency": int(
            any(term in text_lower for term in urgency_terms)
        ),
        "credentials": int(
            any(term in text_lower for term in credential_terms)
        ),
        "link_present": int(len(urls) > 0),
        "threat": int(
            any(term in text_lower for term in threat_terms)
        ),
        "money": int(
            any(term in text_lower for term in money_terms)
        ),
        "url_count_score": min(len(urls) / 3, 1.0),
        "urls": urls
    }

    return features


# ----------------------------------------------------------
# MATHEMATICAL PHISHING RISK INDEX
# ----------------------------------------------------------

def calculate_pri(features):
    """
    Calculate a provisional 0–100 screening index.
    This is not a validated probability of phishing.
    """

    pri = 100 * (
        0.20 * features["urgency"]
        + 0.25 * features["credentials"]
        + 0.15 * features["link_present"]
        + 0.20 * features["threat"]
        + 0.10 * features["money"]
        + 0.10 * features["url_count_score"]
    )

    return round(pri, 2)


def classify_pri(pri):
    """Assign provisional screening categories."""

    if pri < 30:
        return "LOWER RISK"
    elif pri < 60:
        return "NEEDS REVIEW"
    else:
        return "HIGH PHISHING RISK"


# ----------------------------------------------------------
# USER INTERFACE
# ----------------------------------------------------------

st.divider()
st.header("📧 Screenshot-Based Phishing Email Checker")

st.write(
    "Upload an email screenshot to extract its visible text, "
    "identify potential warning signs, calculate the mathematical "
    "Phishing Risk Index (PRI), and obtain a separate machine-learning "
    "classification."
)

st.info(
    "Neither a low PRI nor a legitimate model classification guarantees "
    "that an email is safe. Always verify suspicious messages."
)

email_screenshot = st.file_uploader(
    "Upload an email screenshot",
    type=["png", "jpg", "jpeg"],
    key="phishing_email_screenshot"
)


# ----------------------------------------------------------
# PROCESS SCREENSHOT
# ----------------------------------------------------------

if email_screenshot is not None:

    try:
        screenshot = Image.open(email_screenshot).convert("RGB")

        st.subheader("Uploaded Screenshot")
        st.image(
            screenshot,
            caption="Email screenshot for analysis",
            use_container_width=True
        )

        if st.button(
            "🔎 Analyze Phishing Risk",
            key="analyze_phishing_screenshot"
        ):

            with st.spinner("Extracting and analyzing email text..."):
                email_text = extract_email_text(screenshot)

            if not email_text:
                st.error(
                    "No readable text was detected. "
                    "Try uploading a clearer screenshot."
                )

            else:
                # ------------------------------------------
                # DISPLAY EXTRACTED TEXT
                # ------------------------------------------

                with st.expander("📝 View Extracted Email Text"):
                    st.text(email_text)

                # ------------------------------------------
                # MATHEMATICAL ANALYSIS
                # ------------------------------------------

                features = analyze_phishing_features(email_text)
                pri = calculate_pri(features)
                category = classify_pri(pri)

                st.subheader("📊 Mathematical Risk Results")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Phishing Risk Index",
                        f"{pri:.2f}/100"
                    )

                with col2:
                    st.metric(
                        "Screening Category",
                        category
                    )

                st.progress(int(pri))

                st.caption(
                    "The PRI is a provisional, rule-based screening index, "
                    "not the probability that an email is phishing."
                )

                if category == "LOWER RISK":
                    st.success(
                        "Fewer of the selected warning signs were detected. "
                        "This does not prove the email is legitimate."
                    )

                elif category == "NEEDS REVIEW":
                    st.warning(
                        "Some warning signs were detected. Verify the sender "
                        "through an official channel before taking action."
                    )

                else:
                    st.error(
                        "Several weighted warning indicators were detected. "
                        "Avoid clicking links or sharing sensitive information "
                        "until you verify the message."
                    )

                # ------------------------------------------
                # FEATURE BREAKDOWN
                # ------------------------------------------

                st.subheader("🧮 Mathematical Feature Breakdown")

                breakdown = [
                    ("Urgency wording", features["urgency"], 0.20),
                    ("Credential-related wording", features["credentials"], 0.25),
                    ("Visible URL detected", features["link_present"], 0.15),
                    ("Threat-related wording", features["threat"], 0.20),
                    ("Money or reward wording", features["money"], 0.10),
                    ("URL count score", features["url_count_score"], 0.10)
                ]

                for name, value, weight in breakdown:
                    contribution = 100 * weight * value

                    st.write(
                        f"**{name}:** {value:.2f} × "
                        f"{weight:.2f} × 100 = "
                        f"**{contribution:.2f} points**"
                    )

                # ------------------------------------------
                # DISPLAY VISIBLE URLS
                # ------------------------------------------

                st.subheader("🔗 Visible Links")

                if features["urls"]:
                    st.warning(
                        "These links were extracted from the screenshot. "
                        "They have not been opened or verified."
                    )

                    for url in features["urls"]:
                        st.code(url)

                else:
                    st.write(
                        "No URLs matching the supported patterns were found. "
                        "Some links may not be recognized by OCR."
                    )

                # ------------------------------------------
                # MACHINE-LEARNING PREDICTION
                # ------------------------------------------

                st.subheader("🤖 Machine-Learning Assessment")

                phishing_model = load_phishing_model()

                if phishing_model is None:
                    st.warning(
                        "The phishing model could not be loaded. "
                        "Check that phishing_model.pkl is in your project."
                    )

                else:
                    try:
                        # This model expects raw email text.
                        prediction = int(
                            phishing_model.predict([email_text])[0]
                        )

                        if prediction == 1:
                            st.error(
                                "Model classification: Potentially phishing"
                            )

                        elif prediction == 0:
                            st.success(
                                "Model classification: Likely legitimate"
                            )

                        else:
                            st.warning(
                                f"Unexpected model label: {prediction}"
                            )

                        # ----------------------------------
                        # MODEL SCORE
                        # ----------------------------------

                        if hasattr(phishing_model, "predict_proba"):
                            probabilities = (
                                phishing_model.predict_proba([email_text])[0]
                            )

                            classes = list(phishing_model.classes_)

                            if 1 in classes:
                                score = float(
                                    probabilities[classes.index(1)]
                                )

                                st.metric(
                                    "Phishing Model Score",
                                    f"{score * 100:.1f}%"
                                )

                                st.caption(
                                    "This is the model's estimated score for "
                                    "the phishing class, not a guarantee of "
                                    "real-world probability."
                                )

                    except Exception as error:
                        st.error(
                            "The phishing model could not analyze the "
                            "extracted email text."
                        )
                        st.code(str(error))

    except Exception as error:
        st.error(
            "An error occurred while processing the email screenshot."
        )
        st.code(str(error))
```

                        st.error(
                            "The phishing model could not analyze the email text."
                        )
                        st.code(str(error))

