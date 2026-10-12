"""Screenshot detector for email addresses, Philippine mobile numbers, URLs, and QR codes.

OCR and pattern matching are heuristic. Results should be validated against labeled
screenshots; this module does not guarantee that every visible item is detected.
"""
import os
import re
import shutil

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageOps, ImageEnhance, ImageFilter


# ==========================================================
# TESSERACT SETUP (works on Windows and common hosted systems)
# ==========================================================
def configure_tesseract():
    configured = os.environ.get("TESSERACT_CMD")
    candidates = [
        configured,
        shutil.which("tesseract"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract",
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            pytesseract.pytesseract.tesseract_cmd = candidate
            return candidate
        if candidate and candidate == shutil.which(candidate):
            pytesseract.pytesseract.tesseract_cmd = candidate
            return candidate
    # Don't crash the whole Streamlit app at import time. run_ocr() will return
    # a clear error if OCR is unavailable; QR and direct text helpers still work.
    return None


TESSERACT_PATH = configure_tesseract()


def clean_text(text):
    replacements = {
        "\u00a0": " ", "—": "-", "–": "-", "−": "-",
        "“": '"', "”": '"', "‘": "'", "’": "'",
        "＠": "@", "。": ".", "．": ".", "：": ":",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _unique(values, case_insensitive=True):
    output, seen = [], set()
    for value in values:
        value = value.strip()
        key = value.lower() if case_insensitive else value
        if value and key not in seen:
            seen.add(key)
            output.append(value)
    return output


# ==========================================================
# EMAIL DETECTION
# ==========================================================
def detect_emails(text):
    text = clean_text(text)
    pattern = re.compile(
        r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![\w.-])",
        re.IGNORECASE,
    )
    emails = pattern.findall(text)

    # Recover common OCR spacing, e.g. "name @ gmail . com".
    spaced = re.compile(
        r"([A-Za-z0-9_%+-][A-Za-z0-9._%+-]*)\s*@\s*"
        r"([A-Za-z0-9-]+(?:\s*\.\s*[A-Za-z0-9-]+)*)\s*\.\s*([A-Za-z]{2,})",
        re.IGNORECASE,
    )
    for user, domain, suffix in spaced.findall(text):
        domain = re.sub(r"\s*\.\s*", ".", domain)
        emails.append(f"{user}@{domain}.{suffix}")
    return _unique([item.lower().strip(".,;:!?()[]{}<>") for item in emails])


# ==========================================================
# URL / DOMAIN DETECTION
# ==========================================================
def detect_urls(text):
    text = clean_text(text)
    # Recognizes explicit links and common bare domains (example.com).
    pattern = re.compile(
        r"(?<![@A-Za-z0-9_-])"
        r"(?:https?://|www\.)?"
        r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+"
        r"[A-Za-z]{2,}"
        r"(?::\d{1,5})?"
        r"(?:/[^\s<>\"']*)?",
        re.IGNORECASE,
    )
    found = []
    for match in pattern.finditer(text):
        item = match.group(0).strip().rstrip(".,;:!?)]}>\"'")
        # Avoid returning a bare domain fragment from an email address.
        if item and "@" not in item:
            found.append(item)
    return _unique(found)


# ==========================================================
# PHILIPPINE MOBILE NUMBER DETECTION
# ==========================================================
def normalize_phone(phone):
    digits = re.sub(r"[^\d+]", "", phone)
    if digits.startswith("+63"):
        digits = "0" + digits[3:]
    elif digits.startswith("63") and len(digits) == 12:
        digits = "0" + digits[2:]
    return digits


def detect_phones(text):
    text = clean_text(text)
    patterns = [
        r"(?<!\d)09\d{9}(?!\d)",
        r"(?<!\d)09\d{2}[\s.-]+\d{3}[\s.-]+\d{4}(?!\d)",
        r"(?<!\d)\+?63[\s.-]*9\d{2}[\s.-]*\d{3}[\s.-]*\d{4}(?!\d)",
    ]
    phones = []
    for pattern in patterns:
        for match in re.findall(pattern, text):
            normalized = normalize_phone(match)
            if re.fullmatch(r"09\d{9}", normalized):
                phones.append(normalized)
    return _unique(phones, case_insensitive=False)


# ==========================================================
# QR CODE DETECTION
# ==========================================================
def detect_qr(image):
    rgb = np.array(image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    enlarged = cv2.resize(bgr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    enlarged_gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    threshold = cv2.threshold(
        enlarged_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]
    detector = cv2.QRCodeDetector()

    for candidate in (bgr, gray, enlarged, enlarged_gray, threshold):
        try:
            data, points, _ = detector.detectAndDecode(candidate)
            if points is not None:
                return True, data or ""
        except cv2.error:
            pass
        try:
            multi = detector.detectAndDecodeMulti(candidate)
            if multi and multi[0]:
                decoded = [value for value in multi[1] if value]
                return True, "\n".join(decoded)
        except (cv2.error, Exception):
            pass
    return False, ""


# ==========================================================
# OCR PREPROCESSING
# ==========================================================
def prepare_images(image):
    image = image.convert("RGB")
    # Avoid extreme memory use for very large screenshots.
    max_dimension = 2200
    scale = min(1.0, max_dimension / max(image.size))
    if scale < 1:
        image = image.resize(
            (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
            Image.Resampling.LANCZOS,
        )

    enlarged = image.resize(
        (image.width * 2, image.height * 2), Image.Resampling.LANCZOS
    )
    gray = ImageOps.grayscale(enlarged)
    contrast = ImageEnhance.Contrast(gray).enhance(2.0)
    sharpened = contrast.filter(ImageFilter.SHARPEN)
    array = np.array(sharpened)
    otsu = cv2.threshold(
        array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]
    adaptive = cv2.adaptiveThreshold(
        array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )
    return [enlarged, sharpened, otsu, adaptive]


def run_ocr(image):
    """Run OCR on several image variants and combine distinct outputs."""
    try:
        pytesseract.get_tesseract_version()
    except Exception as error:
        raise RuntimeError(
            "Tesseract OCR is unavailable. Install Tesseract on the host or set "
            "the TESSERACT_CMD environment variable to its executable path."
        ) from error

    results = []
    for candidate in prepare_images(image):
        for config in ("--oem 3 --psm 6", "--oem 3 --psm 11", "--oem 3 --psm 12"):
            try:
                text = pytesseract.image_to_string(candidate, config=config)
                text = clean_text(text).strip()
                if text and text not in results:
                    results.append(text)
            except (pytesseract.TesseractError, OSError):
                continue
    return "\n".join(results)


# ==========================================================
# MAIN DETECTOR
# ==========================================================
def detect_sensitive_information(image):
    """Return the same result keys expected by the website's PEI calculator."""
    try:
        text = run_ocr(image)
        ocr_error = ""
    except RuntimeError as error:
        text = ""
        ocr_error = str(error)

    emails = detect_emails(text)
    phones = detect_phones(text)
    urls = detect_urls(text)
    qr_detected, qr_data = detect_qr(image)

    return {
        "emails": emails,
        "phones": phones,
        "urls": urls,
        "qr_detected": qr_detected,
        "qr_data": qr_data,
        "text": text,
        "ocr_error": ocr_error,
    }
