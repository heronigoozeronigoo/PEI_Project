# ==========================================================
# PEI MACHINE LEARNING DATASET COLLECTOR
# Direct OCR + Sensitive Information Detection
# ==========================================================

import os
import re

import cv2
import pandas as pd
import pytesseract
import streamlit as st

from PIL import Image


# ==========================================================
# TESSERACT CONFIGURATION
# ==========================================================

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# ==========================================================
# DATASET CONFIGURATION
# ==========================================================

DATASET_FOLDER = "dataset"
DATASET_PATH = os.path.join(
    DATASET_FOLDER,
    "pei_dataset.csv"
)

os.makedirs(
    DATASET_FOLDER,
    exist_ok=True
)


DATASET_COLUMNS = [
    "email_count",
    "phone_count",
    "url_count",
    "qr_count",
    "total_items",
    "avg_confidence",
    "sensitive_area",
    "pei_class"
]


# ==========================================================
# CREATE DATASET
# ==========================================================

if not os.path.exists(DATASET_PATH):

    empty_df = pd.DataFrame(
        columns=DATASET_COLUMNS
    )

    empty_df.to_csv(
        DATASET_PATH,
        index=False
    )


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="PEI ML Dataset Collector",
    page_icon="🔐",
    layout="wide"
)


# ==========================================================
# TITLE
# ==========================================================

st.title(
    "🔐 PEI Machine Learning Dataset Collector"
)

st.write(
    "Upload a screenshot and the system will detect "
    "potentially sensitive information."
)

st.info(
    "Use fictional or test information when creating "
    "your research dataset."
)


# ==========================================================
# OCR STATUS
# ==========================================================

st.subheader(
    "System Status"
)


if os.path.exists(TESSERACT_PATH):

    try:

        version = pytesseract.get_tesseract_version()

        st.success(
            f"✅ Tesseract OCR is ready — Version {version}"
        )

    except Exception as e:

        st.error(
            "Tesseract was found but could not start."
        )

        st.code(
            str(e)
        )

else:

    st.error(
        "❌ Tesseract was not found."
    )

    st.write(
        "Expected location:"
    )

    st.code(
        TESSERACT_PATH
    )

    st.stop()


# ==========================================================
# UPLOAD IMAGE
# ==========================================================

uploaded_file = st.file_uploader(
    "Upload a screenshot",
    type=[
        "png",
        "jpg",
        "jpeg"
    ]
)


# ==========================================================
# DETECTION FUNCTION
# ==========================================================

def detect_sensitive_information(
    image
):

    # ------------------------------------------------------
    # Convert PIL image to OpenCV
    # ------------------------------------------------------

    rgb_image = image.convert(
        "RGB"
    )

    image_array = cv2.cvtColor(
        __import__("numpy").array(
            rgb_image
        ),
        cv2.COLOR_RGB2BGR
    )


    height, width = image_array.shape[:2]

    image_area = width * height


    # ------------------------------------------------------
    # OCR
    # ------------------------------------------------------

    ocr_data = pytesseract.image_to_data(
        image_array,
        output_type=pytesseract.Output.DICT
    )


    results = []


    # ------------------------------------------------------
    # REGULAR EXPRESSIONS
    # ------------------------------------------------------

    email_pattern = re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    )


    url_pattern = re.compile(
        r"\b(?:https?://|www\.)[^\s<>\"]+",
        re.IGNORECASE
    )


    phone_pattern = re.compile(
        r"(?<!\d)"
        r"(?:\+63|0)"
        r"(?:\s|-|\(|\))*"
        r"\d{3}"
        r"(?:\s|-|\))*"
        r"\d{3}"
        r"(?:\s|-|\))*"
        r"\d{4}"
        r"(?!\d)"
    )


    # ------------------------------------------------------
    # OCR WORDS
    # ------------------------------------------------------

    words = []

    number_of_words = len(
        ocr_data["text"]
    )


    for i in range(
        number_of_words
    ):

        text = ocr_data["text"][i].strip()

        if not text:
            continue


        try:

            confidence = float(
                ocr_data["conf"][i]
            )

        except:

            confidence = 0


        if confidence < 0:

            continue


        x = int(
            ocr_data["left"][i]
        )

        y = int(
            ocr_data["top"][i]
        )

        w = int(
            ocr_data["width"][i]
        )

        h = int(
            ocr_data["height"][i]
        )


        words.append(
            {
                "text": text,
                "confidence": confidence / 100,
                "x": x,
                "y": y,
                "width": w,
                "height": h
            }
        )


    # ------------------------------------------------------
    # CHECK INDIVIDUAL OCR WORDS
    # ------------------------------------------------------

    for word in words:

        text = word["text"]


        # --------------------------------------------------
        # EMAIL
        # --------------------------------------------------

        if email_pattern.search(text):

            results.append(
                {
                    "type": "Email",
                    "text": text,
                    "confidence": word["confidence"],
                    "x": word["x"],
                    "y": word["y"],
                    "width": word["width"],
                    "height": word["height"]
                }
            )


        # --------------------------------------------------
        # URL
        # --------------------------------------------------

        elif url_pattern.search(text):

            results.append(
                {
                    "type": "URL",
                    "text": text,
                    "confidence": word["confidence"],
                    "x": word["x"],
                    "y": word["y"],
                    "width": word["width"],
                    "height": word["height"]
                }
            )


        # --------------------------------------------------
        # PHONE
        # --------------------------------------------------

        elif phone_pattern.search(text):

            results.append(
                {
                    "type": "Phone",
                    "text": text,
                    "confidence": word["confidence"],
                    "x": word["x"],
                    "y": word["y"],
                    "width": word["width"],
                    "height": word["height"]
                }
            )


    # ======================================================
    # CHECK COMBINED OCR TEXT
    # ======================================================

    full_text = " ".join(
        word["text"]
        for word in words
    )


    # ------------------------------------------------------
    # COMBINED EMAIL DETECTION
    # ------------------------------------------------------

    existing_emails = {
        item["text"]
        for item in results
        if item["type"] == "Email"
    }


    for match in email_pattern.finditer(
        full_text
    ):

        email = match.group(0)

        if email not in existing_emails:

            results.append(
                {
                    "type": "Email",
                    "text": email,
                    "confidence": 0.85,
                    "x": 0,
                    "y": 0,
                    "width": max(
                        1,
                        len(email) * 10
                    ),
                    "height": 20
                }
            )


    # ------------------------------------------------------
    # COMBINED URL DETECTION
    # ------------------------------------------------------

    existing_urls = {
        item["text"]
        for item in results
        if item["type"] == "URL"
    }


    for match in url_pattern.finditer(
        full_text
    ):

        url = match.group(0)

        if url not in existing_urls:

            results.append(
                {
                    "type": "URL",
                    "text": url,
                    "confidence": 0.85,
                    "x": 0,
                    "y": 0,
                    "width": max(
                        1,
                        len(url) * 8
                    ),
                    "height": 20
                }
            )


    # ------------------------------------------------------
    # COMBINED PHONE DETECTION
    # ------------------------------------------------------

    existing_phones = {
        item["text"]
        for item in results
        if item["type"] == "Phone"
    }


    for match in phone_pattern.finditer(
        full_text
    ):

        phone = match.group(0)

        if phone not in existing_phones:

            results.append(
                {
                    "type": "Phone",
                    "text": phone,
                    "confidence": 0.85,
                    "x": 0,
                    "y": 0,
                    "width": max(
                        1,
                        len(phone) * 10
                    ),
                    "height": 20
                }
            )


    # ======================================================
    # QR CODE DETECTION
    # ======================================================

    qr_detector = cv2.QRCodeDetector()


    try:

        qr_data, points, _ = (
            qr_detector.detectAndDecode(
                image_array
            )
        )


        if points is not None:

            points = points.astype(
                int
            )


            x_coordinates = points[
                0
            ][:, 0]

            y_coordinates = points[
                0
            ][:, 1]


            qr_x = int(
                min(x_coordinates)
            )

            qr_y = int(
                min(y_coordinates)
            )

            qr_w = int(
                max(x_coordinates)
                - qr_x
            )

            qr_h = int(
                max(y_coordinates)
                - qr_y
            )


            results.append(
                {
                    "type": "QR Code",
                    "text": (
                        qr_data
                        if qr_data
                        else "QR Code detected"
                    ),
                    "confidence": 0.95,
                    "x": qr_x,
                    "y": qr_y,
                    "width": qr_w,
                    "height": qr_h
                }
            )

    except Exception:

        pass


    # ======================================================
    # REMOVE DUPLICATES
    # ======================================================

    unique_results = []

    seen = set()


    for item in results:

        key = (
            item["type"],
            item["text"]
        )


        if key not in seen:

            seen.add(
                key
            )

            unique_results.append(
                item
            )


    return unique_results


# ==========================================================
# PROCESS UPLOADED IMAGE
# ==========================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    # ------------------------------------------------------
    # DISPLAY IMAGE
    # ------------------------------------------------------

    st.subheader(
        "Uploaded Screenshot"
    )


    st.image(
        image,
        caption="Screenshot being analyzed",
        width="stretch"
    )


    image_width, image_height = (
        image.size
    )


    image_area = (
        image_width *
        image_height
    )


    st.write(
        f"Image size: "
        f"**{image_width} × {image_height} pixels**"
    )


    # ------------------------------------------------------
    # RUN DETECTION
    # ------------------------------------------------------

    with st.spinner(
        "Analyzing screenshot..."
    ):

        results = detect_sensitive_information(
            image
        )


    # ======================================================
    # FEATURE COUNTS
    # ======================================================

    email_count = 0
    phone_count = 0
    url_count = 0
    qr_count = 0


    confidences = []

    sensitive_pixels = 0


    for item in results:

        item_type = item["type"]

        confidence = float(
            item["confidence"]
        )


        confidences.append(
            confidence
        )


        if item_type == "Email":

            email_count += 1


        elif item_type == "Phone":

            phone_count += 1


        elif item_type == "URL":

            url_count += 1


        elif item_type == "QR Code":

            qr_count += 1


        item_width = max(
            0,
            item["width"]
        )

        item_height = max(
            0,
            item["height"]
        )


        sensitive_pixels += (
            item_width *
            item_height
        )


    # ======================================================
    # CALCULATE ML FEATURES
    # ======================================================

    total_items = len(
        results
    )


    if confidences:

        avg_confidence = (
            sum(confidences)
            /
            len(confidences)
        )

    else:

        avg_confidence = 0.0


    if image_area > 0:

        sensitive_area = (
            sensitive_pixels
            /
            image_area
        )

    else:

        sensitive_area = 0.0


    sensitive_area = max(
        0.0,
        min(
            sensitive_area,
            1.0
        )
    )


    # ======================================================
    # DISPLAY FEATURES
    # ======================================================

    st.subheader(
        "📊 Extracted ML Features"
    )


    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.metric(
            "Emails",
            email_count
        )


    with c2:

        st.metric(
            "Phone Numbers",
            phone_count
        )


    with c3:

        st.metric(
            "URLs",
            url_count
        )


    with c4:

        st.metric(
            "QR Codes",
            qr_count
        )


    c5, c6, c7 = st.columns(3)


    with c5:

        st.metric(
            "Total Items",
            total_items
        )


    with c6:

        st.metric(
            "Avg. Confidence",
            f"{avg_confidence:.3f}"
        )


    with c7:

        st.metric(
            "Sensitive Area",
            f"{sensitive_area:.4f}"
        )


    # ======================================================
    # DETECTION DETAILS
    # ======================================================

    st.subheader(
        "🔎 Detection Details"
    )


    if results:

        for i, item in enumerate(
            results,
            start=1
        ):

            st.write(
                f"**{i}. {item['type']}**"
            )

            st.write(
                f"Detected text: "
                f"`{item['text']}`"
            )

            st.write(
                f"Confidence: "
                f"`{item['confidence']:.2f}`"
            )

            st.write(
                f"Position: "
                f"x={item['x']}, "
                f"y={item['y']}"
            )

            st.write(
                f"Bounding box: "
                f"{item['width']} × "
                f"{item['height']} pixels"
            )

            st.divider()


    else:

        st.warning(
            "⚠️ No sensitive information was detected."
        )


    # ======================================================
    # LABEL
    # ======================================================

    st.subheader(
        "🏷️ Training Label"
    )


    st.write(
        "Select the correct privacy exposure "
        "category for this screenshot."
    )


    pei_class = st.selectbox(
        "Privacy Exposure Level",
        [
            "LOW",
            "MODERATE",
            "HIGH"
        ]
    )


    # ======================================================
    # SAVE
    # ======================================================

    if st.button(
        "💾 Save Example to ML Dataset",
        type="primary"
    ):

        new_row = {

            "email_count":
                email_count,

            "phone_count":
                phone_count,

            "url_count":
                url_count,

            "qr_count":
                qr_count,

            "total_items":
                total_items,

            "avg_confidence":
                round(
                    avg_confidence,
                    4
                ),

            "sensitive_area":
                round(
                    sensitive_area,
                    6
                ),

            "pei_class":
                pei_class
        }


        df = pd.read_csv(
            DATASET_PATH
        )


        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [new_row]
                )
            ],
            ignore_index=True
        )


        df.to_csv(
            DATASET_PATH,
            index=False
        )


        st.success(
            "✅ Training example saved!"
        )


        st.write(
            f"Dataset size: "
            f"**{len(df)} examples**"
        )


# ==========================================================
# DATASET PREVIEW
# ==========================================================

st.divider()

st.subheader(
    "📁 Current ML Dataset"
)


try:

    df = pd.read_csv(
        DATASET_PATH
    )


    if len(df) == 0:

        st.info(
            "The dataset is currently empty."
        )

    else:

        st.dataframe(
            df,
            width="stretch"
        )

        st.write(
            f"Total examples: "
            f"**{len(df)}**"
        )


except Exception as e:

    st.error(
        "Unable to load the dataset."
    )

    st.code(
        str(e)
    )