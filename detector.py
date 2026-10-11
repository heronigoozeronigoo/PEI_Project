
import os
import re
import shutil
import cv2
import numpy as np
import pytesseract

from PIL import Image, ImageOps, ImageEnhance, ImageFilter



# ==========================================================
# TESSERACT SETUP
# ==========================================================

TESSERACT_PATH = shutil.which("tesseract")

if TESSERACT_PATH is None:
    windows_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(windows_path):
        TESSERACT_PATH = windows_path

if TESSERACT_PATH is None:
    raise FileNotFoundError(
        "Tesseract OCR is not installed or could not be found."
    )

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# ==========================================================
# CLEAN OCR TEXT
# ==========================================================

def clean_text(text):

    replacements = {
        "\u00a0": " ",
        "—": "-",
        "–": "-",
        "−": "-",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


# ==========================================================
# EMAIL DETECTION
# ==========================================================

def detect_emails(text):

    text = clean_text(text)

    # Normal email
    pattern = r"""
    (?<![\w.-])
    [A-Za-z0-9._%+-]+
    @
    [A-Za-z0-9.-]+
    \.
    [A-Za-z]{2,}
    (?![\w.-])
    """

    emails = re.findall(
        pattern,
        text,
        re.VERBOSE
    )

    # Handle OCR spaces:
    # example @ gmail .com
    spaced_pattern = r"""
    ([A-Za-z0-9._%+-]+)
    \s*@\s*
    ([A-Za-z0-9.-]+)
    \s*\.\s*
    ([A-Za-z]{2,})
    """

    spaced_emails = re.findall(
        spaced_pattern,
        text,
        re.VERBOSE
    )

    for username, domain, extension in spaced_emails:

        email = (
            username
            + "@"
            + domain
            + "."
            + extension
        )

        emails.append(email)

    # Remove duplicates
    final = []

    for email in emails:

        email = email.lower().strip()

        if email not in final:
            final.append(email)

    return final


# ==========================================================
# URL DETECTION
# ==========================================================

def detect_urls(text):
    """Detect URLs with or without a scheme or www prefix."""
    text = clean_text(text)

    pattern = re.compile(
        r"""
        (?<![@A-Za-z0-9_-])
        (?:
            https?://
            |
            www\.
        )?
        (?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+
        [A-Za-z]{2,}
        (?::\d{1,5})?
        (?:/[^\s<>"']*)?
        """,
        re.IGNORECASE | re.VERBOSE
    )

    urls = []
    for match in pattern.finditer(text):
        url = match.group(0).strip().rstrip(".,;:!?)]}>\"'")

        if url and url.lower() not in [existing.lower() for existing in urls]:
            urls.append(url)

    return urls


# ==========================================================
# PHONE NUMBER DETECTION
# ==========================================================

def normalize_phone(phone):

    # Remove spaces, dashes, dots and parentheses
    phone = re.sub(
        r"[^\d+]",
        "",
        phone
    )

    # +639171234567
    if phone.startswith("+63"):

        phone = "0" + phone[3:]

    # 639171234567
    elif phone.startswith("63") and len(phone) == 12:

        phone = "0" + phone[2:]

    return phone


def detect_phones(text):

    text = clean_text(text)

    patterns = [

        # 09171234567
        r"(?<!\d)09\d{9}(?!\d)",

        # 0917 123 4567
        r"(?<!\d)09\d{2}\s+\d{3}\s+\d{4}(?!\d)",

        # 0917-123-4567
        r"(?<!\d)09\d{2}-\d{3}-\d{4}(?!\d)",

        # 0917.123.4567
        r"(?<!\d)09\d{2}\.\d{3}\.\d{4}(?!\d)",

        # +639171234567
        r"(?<!\d)\+639\d{9}(?!\d)",

        # +63 917 123 4567
        r"(?<!\d)\+63\s*9\d{2}\s*\d{3}\s*\d{4}(?!\d)",

        # +63-917-123-4567
        r"(?<!\d)\+63[-\s]*9\d{2}[-\s]*\d{3}[-\s]*\d{4}(?!\d)",

        # 639171234567
        r"(?<!\d)639\d{9}(?!\d)",
    ]

    phones = []

    for pattern in patterns:

        matches = re.findall(
            pattern,
            text
        )

        for phone in matches:

            phone = normalize_phone(phone)

            if re.fullmatch(
                r"09\d{9}",
                phone
            ):

                if phone not in phones:
                    phones.append(phone)

    return phones


# ==========================================================
# QR CODE DETECTION
# ==========================================================

def detect_qr(image):
    """Detect QR codes in several image variants; decoding text is optional."""
    image_rgb = np.array(image.convert("RGB"))
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    enlarged = cv2.resize(
        image_bgr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC
    )
    enlarged_gray = cv2.resize(
        gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC
    )
    threshold = cv2.threshold(
        enlarged_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    detector = cv2.QRCodeDetector()

    for candidate in [image_bgr, gray, enlarged, enlarged_gray, threshold]:
        try:
            data, points, _ = detector.detectAndDecode(candidate)
            # A visible QR code counts as detected even if its payload cannot be decoded.
            if points is not None:
                return True, data or ""

            if hasattr(detector, "detectAndDecodeMulti"):
                result = detector.detectAndDecodeMulti(candidate)
                if result and result[0]:
                    decoded_info = result[1]
                    nonempty = [value for value in decoded_info if value]
                    return True, "\n".join(nonempty)
        except cv2.error:
            continue
        except Exception:
            continue

    return False, ""


# ==========================================================
# OCR PREPROCESSING
# ==========================================================

def prepare_images(image):

    image = image.convert("RGB")

    # Resize
    enlarged = image.resize(
        (
            image.width * 3,
            image.height * 3
        ),
        Image.Resampling.LANCZOS
    )

    # Grayscale
    gray = ImageOps.grayscale(
        enlarged
    )

    # Improve contrast
    contrast = ImageEnhance.Contrast(
        gray
    ).enhance(2.0)

    # Sharpen
    sharpened = contrast.filter(
        ImageFilter.SHARPEN
    )

    # Convert to numpy
    array = np.array(
        sharpened
    )

    # OTSU threshold
    otsu = cv2.threshold(
        array,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    # Adaptive threshold
    adaptive = cv2.adaptiveThreshold(
        array,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    return [
        enlarged,
        sharpened,
        otsu,
        adaptive
    ]


# ==========================================================
# OCR
# ==========================================================

def run_ocr(image):

    prepared_images = prepare_images(
        image
    )

    results = []

    configurations = [
        "--psm 6",
        "--psm 11",
        "--psm 12"
    ]

    for img in prepared_images:

        for config in configurations:

            try:

                text = pytesseract.image_to_string(
                    img,
                    config=config
                )

                if text.strip():

                    results.append(text)

            except Exception:
                pass

    combined_text = "\n".join(
        results
    )

    return clean_text(
        combined_text
    )


# ==========================================================
# MAIN DETECTOR
# ==========================================================

def detect_sensitive_information(image):

    # OCR
    text = run_ocr(
        image
    )

    # Detect email
    emails = detect_emails(
        text
    )

    # Detect phone
    phones = detect_phones(
        text
    )

    # Detect URL
    urls = detect_urls(
        text
    )

    # Detect QR
    qr_detected, qr_data = detect_qr(
        image
    )

    return {

        "emails": emails,

        "phones": phones,

        "urls": urls,

        "qr_detected": qr_detected,

        "qr_data": qr_data,

        "text": text
    }
    
