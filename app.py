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
    "and Machine Privacy Classification"
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
