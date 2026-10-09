# ==========================================================
# PEI SENSITIVE INFORMATION DETECTOR
# Multi-Pass OCR Version
# Detects:
#   - Email
#   - Philippine Phone Numbers
#   - URLs
#   - QR Codes
# ==========================================================

import os
import re
import cv2
import numpy as np
import pytesseract
import streamlit as st

from PIL import Image, ImageOps, ImageFilter


# ==========================================================
# TESSERACT SETUP
# ==========================================================

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if not os.path.exists(TESSERACT_PATH):

    st.error(
        "Tesseract was not found."
    )

    st.code(
        TESSERACT_PATH
    )

    st.stop()


pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# ==========================================================
# PAGE SETTINGS
# ==========================================================

st.set_page_config(
    page_title="PEI Sensitive Information Detector",
    page_icon="🔐",
    layout="wide"
)


# ==========================================================
# TITLE
# ==========================================================

st.title(
    "🔐 Sensitive Information Detection"
)

st.write(
    "Upload a screenshot to detect sensitive information."
)


# ==========================================================
# UPLOAD
# ==========================================================

uploaded_file = st.file_uploader(
    "Upload Screenshot",
    type=[
        "png",
        "jpg",
        "jpeg"
    ]
)


# ==========================================================
# PHONE DETECTION
# ==========================================================

def detect_phone_numbers(text):

    # ------------------------------------------------------
    # Normalize OCR characters
    # ------------------------------------------------------

    text = text.replace(
        "—",
        "-"
    )

    text = text.replace(
        "–",
        "-"
    )

    text = text.replace(
        "O",
        "0"
    )

    text = text.replace(
        "o",
        "0"
    )

    text = text.replace(
        "I",
        "1"
    )

    text = text.replace(
        "l",
        "1"
    )

    text = text.replace(
        "|",
        "1"
    )


    # ------------------------------------------------------
    # Patterns
    # ------------------------------------------------------

    patterns = [

        # 09123456789
        r"(?<!\d)09\d{9}(?!\d)",

        # 0912 345 6789
        r"(?<!\d)09\d{2}\s+\d{3}\s+\d{4}(?!\d)",

        # 0912-345-6789
        r"(?<!\d)09\d{2}-\d{3}-\d{4}(?!\d)",

        # +639123456789
        r"(?<!\d)\+639\d{9}(?!\d)",

        # +63 912 345 6789
        r"(?<!\d)\+63\s*9\d{2}\s*\d{3}\s*\d{4}(?!\d)",

        # 639123456789
        r"(?<!\d)639\d{9}(?!\d)"
    ]


    found = []


    for pattern in patterns:

        matches = re.findall(
            pattern,
            text
        )

        for match in matches:

            cleaned = re.sub(
                r"[\s\-]",
                "",
                match
            )

            if cleaned not in found:

                found.append(
                    cleaned
                )


    return found


# ==========================================================
# EMAIL DETECTION
# ==========================================================

def detect_emails(text):

    pattern = (
        r"\b[A-Za-z0-9._%+-]+"
        r"@"
        r"[A-Za-z0-9.-]+"
        r"\.[A-Za-z]{2,}\b"
    )

    matches = re.findall(
        pattern,
        text
    )

    return list(
        dict.fromkeys(
            matches
        )
    )


# ==========================================================
# URL DETECTION
# ==========================================================

def detect_urls(text):

    pattern = (
        r"\b(?:https?://|www\.)"
        r"[^\s<>\"']+"
    )

    matches = re.findall(
        pattern,
        text,
        re.IGNORECASE
    )

    cleaned = []

    for url in matches:

        url = url.rstrip(
            ".,;:!?)]}"
        )

        if url not in cleaned:

            cleaned.append(
                url
            )

    return cleaned


# ==========================================================
# QR CODE DETECTION
# ==========================================================

def detect_qr(image):

    image_cv = np.array(
        image.convert("RGB")
    )

    image_cv = cv2.cvtColor(
        image_cv,
        cv2.COLOR_RGB2BGR
    )

    detector = cv2.QRCodeDetector()

    try:

        data, points, _ = (
            detector.detectAndDecode(
                image_cv
            )
        )

        if points is not None:

            return True, data

    except Exception:

        pass

    return False, ""


# ==========================================================
# MULTI-PASS OCR
# ==========================================================

def run_ocr(image):

    results = []


    # ------------------------------------------------------
    # PASS 1
    # Normal enlarged OCR
    # ------------------------------------------------------

    enlarged = image.resize(
        (
            image.width * 4,
            image.height * 4
        )
    )

    text1 = pytesseract.image_to_string(
        enlarged,
        config="--psm 6"
    )

    results.append(
        text1
    )


    # ------------------------------------------------------
    # PASS 2
    # Grayscale
    # ------------------------------------------------------

    gray = ImageOps.grayscale(
        enlarged
    )

    gray = gray.filter(
        ImageFilter.SHARPEN
    )

    text2 = pytesseract.image_to_string(
        gray,
        config="--psm 6"
    )

    results.append(
        text2
    )


    # ------------------------------------------------------
    # PASS 3
    # Adaptive threshold
    # ------------------------------------------------------

    gray_cv = np.array(
        gray
    )

    threshold = cv2.adaptiveThreshold(
        gray_cv,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )


    text3 = pytesseract.image_to_string(
        threshold,
        config="--psm 6"
    )

    results.append(
        text3
    )


    # ------------------------------------------------------
    # PASS 4
    # Sparse text mode
    # ------------------------------------------------------

    text4 = pytesseract.image_to_string(
        enlarged,
        config="--psm 11"
    )

    results.append(
        text4
    )


    return results


# ==========================================================
# MAIN PROGRAM
# ==========================================================

if uploaded_file is not None:

    # ------------------------------------------------------
    # LOAD IMAGE
    # ------------------------------------------------------

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    # ------------------------------------------------------
    # SHOW IMAGE
    # ------------------------------------------------------

    st.subheader(
        "📷 Uploaded Screenshot"
    )

    st.image(
        image,
        width="stretch"
    )


    # ======================================================
    # OCR
    # ======================================================

    with st.spinner(
        "Analyzing screenshot..."
    ):

        ocr_results = run_ocr(
            image
        )


    # ------------------------------------------------------
    # Combine OCR results
    # ------------------------------------------------------

    combined_text = "\n".join(
        ocr_results
    )


    # ======================================================
    # DISPLAY OCR
    # ======================================================

    with st.expander(
        "📝 View OCR Results"
    ):

        for i, text in enumerate(
            ocr_results,
            start=1
        ):

            st.write(
                f"OCR Pass {i}"
            )

            st.code(
                text
            )


    # ======================================================
    # DETECT INFORMATION
    # ======================================================

    emails = detect_emails(
        combined_text
    )

    phones = detect_phone_numbers(
        combined_text
    )

    urls = detect_urls(
        combined_text
    )

    qr_detected, qr_data = detect_qr(
        image
    )


    # ======================================================
    # EXTRA PHONE RECOVERY
    # ======================================================
    #
    # If OCR separates the phone number into individual
    # pieces, this attempts to reconstruct it.
    #
    # Example:
    #
    # 0912 345 6789
    #
    # ------------------------------------------------------

    if not phones:

        # Find groups of digits
        digit_groups = re.findall(
            r"\d+",
            combined_text
        )


        for group in digit_groups:

            cleaned = group.strip()


            # Exact 11-digit Philippine mobile
            if re.fullmatch(
                r"09\d{9}",
                cleaned
            ):

                if cleaned not in phones:

                    phones.append(
                        cleaned
                    )


            # International format
            elif re.fullmatch(
                r"639\d{9}",
                cleaned
            ):

                if cleaned not in phones:

                    phones.append(
                        cleaned
                    )


    # ======================================================
    # RESULTS
    # ======================================================

    st.divider()

    st.header(
        "📊 Detection Results"
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Emails",
            len(emails)
        )


    with col2:

        st.metric(
            "Phone Numbers",
            len(phones)
        )


    with col3:

        st.metric(
            "URLs",
            len(urls)
        )


    with col4:

        st.metric(
            "QR Codes",
            1 if qr_detected else 0
        )


    # ======================================================
    # PHONE RESULTS
    # ======================================================

    st.subheader(
        "📱 Phone Numbers"
    )


    if phones:

        for phone in phones:

            st.success(
                f"✅ Phone detected: {phone}"
            )

    else:

        st.warning(
            "⚠️ No phone number detected."
        )


    # ======================================================
    # EMAIL RESULTS
    # ======================================================

    st.subheader(
        "📧 Emails"
    )


    if emails:

        for email in emails:

            st.success(
                f"✅ Email detected: {email}"
            )

    else:

        st.write(
            "No email detected."
        )


    # ======================================================
    # URL RESULTS
    # ======================================================

    st.subheader(
        "🌐 URLs"
    )


    if urls:

        for url in urls:

            st.success(
                f"✅ URL detected: {url}"
            )

    else:

        st.write(
            "No URL detected."
        )


    # ======================================================
    # QR RESULTS
    # ======================================================

    st.subheader(
        "🔳 QR Code"
    )


    if qr_detected:

        st.success(
            "✅ QR Code detected."
        )

        if qr_data:

            st.write(
                "QR content:"
            )

            st.code(
                qr_data
            )

    else:

        st.write(
            "No QR Code detected."
        )


    # ======================================================
    # TOTAL
    # ======================================================

    total_items = (
        len(emails)
        + len(phones)
        + len(urls)
        + (1 if qr_detected else 0)
    )


    st.divider()


    st.header(
        "📈 Summary"
    )


    st.metric(
        "Total Sensitive Items",
        total_items
    )


    # ======================================================
    # STATUS
    # ======================================================

    if total_items == 0:

        st.error(
            "❌ No sensitive information detected."
        )

    elif total_items >= 3:

        st.warning(
            "⚠️ Multiple sensitive information types detected."
        )

    else:

        st.info(
            "ℹ️ Sensitive information detected."
        )


    # ======================================================
    # DEBUG
    # ======================================================

    with st.expander(
        "🔧 Debug Information"
    ):

        st.write(
            "Combined OCR text:"
        )

        st.code(
            combined_text
        )

        st.write(
            "Detected phone numbers:"
        )

        st.write(
            phones
        )