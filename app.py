"""Sentinel — screenshot privacy exposure and phishing screening research app.

Run with: streamlit run app.py
Required/optional local files: detector.py, pei_model.pkl, email_phishing_model_v2.pkl.
"""
from pathlib import Path
import re
from urllib.parse import urlparse
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from PIL import Image, UnidentifiedImageError

try:
    import joblib
except ImportError:
    joblib = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

try:
    from detector import detect_sensitive_information
except Exception:
    detect_sensitive_information = None

BASE_DIR = Path(__file__).resolve().parent
PEI_MODEL_PATH = BASE_DIR / "pei_model.pkl"
PHISHING_MODEL_PATH = BASE_DIR / "email_phishing_model_v2.pkl"
ALLOWED_TYPES = ["png", "jpg", "jpeg"]

# PEI weights are the research prototype's chosen sensitivity weights.
WEIGHTS = {"Email": 0.15, "Phone": 0.15, "URL": 0.10, "QR Code": 0.20}
R_MAX = sum(WEIGHTS.values())  # 0.60: maximum if each category saturates at its weight
DEFAULT_AREA = 1.00
DEFAULT_VISIBILITY = 1.00
DEFAULT_CONFIDENCE = 0.90

# Provisional phishing indicator weights. These are points, not probabilities.
PHISHING_WEIGHTS = {
    "Credential / sensitive-data request": 25,
    "Suspicious URL pattern": 25,
    "Urgency / pressure": 15,
    "Threat / consequence": 15,
    "Money / reward bait": 15,
    "Multiple URLs": 5,
}

st.set_page_config(page_title="Sentinel | Digital Safety Lab", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")


# ------------------------------ Model loading ------------------------------
@st.cache_resource(show_spinner=False)
def load_model(path_string: str):
    """Load and cache an optional joblib model; raise useful errors on failure."""
    if joblib is None:
        raise ImportError("Install joblib to load .pkl models: pip install joblib")
    path = Path(path_string)
    if not path.exists():
        raise FileNotFoundError(f"{path.name} was not found beside app.py.")
    return joblib.load(path)


def optional_model(path: Path) -> Tuple[Optional[Any], str]:
    """Return a model or an explanation without preventing the app from starting."""
    try:
        return load_model(str(path)), ""
    except Exception as exc:
        return None, str(exc)


pei_model, pei_model_error = optional_model(PEI_MODEL_PATH)
phishing_model, phishing_model_error = optional_model(PHISHING_MODEL_PATH)


# ------------------------------ Shared helpers -----------------------------
def open_uploaded_image(uploaded_file) -> Image.Image:
    """Validate and fully load an uploaded image, returning an RGB copy."""
    if uploaded_file is None:
        raise ValueError("Please upload a screenshot first.")
    if not uploaded_file.getvalue():
        raise ValueError("The uploaded file is empty. Choose a valid screenshot.")
    try:
        uploaded_file.seek(0)
        with Image.open(uploaded_file) as source:
            source.verify()
        uploaded_file.seek(0)
        with Image.open(uploaded_file) as source:
            source.load()
            return source.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("This file could not be read as an image. Try a clear PNG or JPG screenshot.") from exc


def clip_unit(value: float, label: str) -> float:
    """Validate a finite numeric parameter in the inclusive range 0–1."""
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number from 0 to 1.") from exc
    if not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{label} must be between 0 and 1.")
    return numeric


def count_items(found: Dict[str, Any]) -> Dict[str, int]:
    """Count supported detector categories safely, even if keys are missing."""
    return {
        "Email": len(found.get("emails") or []),
        "Phone": len(found.get("phones") or []),
        "URL": len(found.get("urls") or []),
        "QR Code": int(bool(found.get("qr_detected", False))),
    }


def detect_privacy_items(image: Image.Image) -> Dict[str, Any]:
    """Call the project's detector and normalize its result structure."""
    if detect_sensitive_information is None:
        raise RuntimeError(
            "The privacy detector could not be imported. Make sure detector.py is in the same folder as app.py "
            "and that its dependencies are installed."
        )
    result = detect_sensitive_information(image)
    if not isinstance(result, dict):
        raise RuntimeError("The detector returned an unexpected result. It must return a dictionary.")
    return {
        "emails": list(result.get("emails") or []),
        "phones": list(result.get("phones") or []),
        "urls": list(result.get("urls") or []),
        "qr_detected": bool(result.get("qr_detected", False)),
        "qr_data": str(result.get("qr_data") or ""),
        "text": str(result.get("text") or ""),
    }


# ------------------------------ PEI calculation ----------------------------
def calculate_pei(
    counts: Dict[str, int], area: float, visibility: float, confidence: float
) -> Dict[str, Any]:
    """Calculate bounded category exposure and a 0–100 PEI.

    For category k with n_k detections:
      E_k = w_k * (1 - 2**(-n_k)) * A * V * C
      PEI = 100 * sum(E_k) / sum(w_k)

    The diminishing-return term ensures repeated items in one category do not
    make the score exceed 100. Each input is validated before arithmetic.
    """
    area = clip_unit(area, "Relative area")
    visibility = clip_unit(visibility, "Visibility")
    confidence = clip_unit(confidence, "Detection confidence")
    factor = area * visibility * confidence
    category_rows = []
    raw_total = 0.0
    for category, weight in WEIGHTS.items():
        try:
            count = int(counts.get(category, 0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Count for {category} must be a non-negative integer.") from exc
        if count < 0:
            raise ValueError(f"Count for {category} cannot be negative.")
        saturation = 1.0 - (0.5 ** count) if count else 0.0
        contribution = weight * saturation * factor
        raw_total += contribution
        category_rows.append({
            "category": category, "count": count, "weight": weight,
            "saturation": saturation, "factor": factor, "contribution": contribution,
        })
    score = 0.0 if R_MAX == 0 else 100.0 * raw_total / R_MAX
    score = max(0.0, min(100.0, score))
    level = "LOW" if score <= 33.33 else "MODERATE" if score <= 66.67 else "HIGH"
    return {
        "counts": counts, "area": area, "visibility": visibility,
        "confidence": confidence, "factor": factor, "rows": category_rows,
        "raw_total": raw_total, "r_max": R_MAX, "score": score, "level": level,
    }


def show_pei_calculation(result: Dict[str, Any]) -> None:
    """Render the PEI formula, substituted values, and transparent arithmetic."""
    st.subheader("Mathematical PEI calculation")
    st.latex(r"E_k = w_k(1 - 2^{-n_k})AVC")
    st.latex(r"PEI = 100\times\frac{\sum_k E_k}{R_{\max}},\quad R_{\max}=\sum_k w_k=0.60")
    st.caption("A = relative area, V = visibility, C = detection confidence, nₖ = number of detected items in category k.")
    st.markdown("**Step 1 — Substitute values for each category**")
    for row in result["rows"]:
        st.write(
            f"**{row['category']}**: E = {row['weight']:.2f} × (1 − 2⁻{row['count']}) × "
            f"{result['area']:.2f} × {result['visibility']:.2f} × {result['confidence']:.2f} "
            f"= **{row['contribution']:.4f}**"
        )
    pieces = " + ".join(f"{row['contribution']:.4f}" for row in result["rows"])
    st.markdown("**Step 2 — Add category contributions**")
    st.code(f"R = {pieces} = {result['raw_total']:.4f}", language="text")
    st.markdown("**Step 3 — Normalize to a 0–100 index**")
    st.code(
        f"PEI = 100 × ({result['raw_total']:.4f} / {result['r_max']:.2f})\n"
        f"PEI = {result['score']:.2f} / 100",
        language="text",
    )


# ---------------------------- Phishing analysis ----------------------------
def extract_text(image: Image.Image) -> str:
    """Extract screenshot text with Tesseract OCR and validate the output."""
    if pytesseract is None:
        raise RuntimeError("OCR library missing. Install pytesseract and the Tesseract OCR engine.")
    try:
        text = pytesseract.image_to_string(image).strip()
    except Exception as exc:
        raise RuntimeError(
            "OCR could not process this screenshot. Confirm Tesseract OCR is installed and try a clearer image."
        ) from exc
    if not text:
        raise ValueError("No readable text was found. Upload a clearer email screenshot with visible text.")
    return text


def get_phishing_indicators(email_text: str) -> Dict[str, Any]:
    """Calculate a transparent heuristic risk score from specific warning signals.

    This score is a screening aid, not a probability or a validated substitute for
    the trained classifier. Merely containing a URL no longer adds risk by itself.
    """
    text = email_text.lower()
    urgency_terms = [
        "urgent", "immediately", "act now", "within 24 hours", "final warning",
        "verify now", "respond immediately", "limited time",
    ]
    credential_phrases = [
        "enter your password", "confirm your password", "send your password",
        "provide your password", "verify your account", "verify your identity",
        "share your otp", "send the otp", "verification code", "one-time code",
        "login credentials", "sign in to restore", "update your payment details",
    ]
    threat_terms = [
        "account will be closed", "account suspended", "account is suspended",
        "account locked", "will be terminated", "legal action", "access will be blocked",
        "unauthorized activity", "service will be discontinued",
    ]
    money_terms = [
        "claim your reward", "free money", "payment required", "transfer money",
        "claim your gift", "you have won", "you are the winner", "prize money",
        "refund available", "urgent payment", "bank details",
    ]
    urls = re.findall(r"(?:https?://|www\.)[^\s<>\"']+", email_text, flags=re.IGNORECASE)
    shortener_hosts = {
        "bit.ly", "tinyurl.com", "t.co", "is.gd", "cutt.ly", "rb.gy", "shorturl.at",
    }
    suspicious_url_reasons = []
    for raw_url in urls:
        candidate = raw_url.rstrip(".,;:!?) ]}")
        parsed = urlparse(candidate if candidate.lower().startswith(("http://", "https://")) else "http://" + candidate)
        host = (parsed.hostname or "").lower().strip(".")
        reasons = []
        if host in shortener_hosts or any(host.endswith("." + domain) for domain in shortener_hosts):
            reasons.append("URL shortener hides the destination")
        if host and re.fullmatch(r"[0-9.]+", host):
            reasons.append("IP-address host instead of a normal domain")
        if "@" in parsed.netloc:
            reasons.append("URL contains @, which can obscure the apparent destination")
        if "xn--" in host:
            reasons.append("punycode domain may use look-alike characters")
        if host.count("-") >= 3:
            reasons.append("domain contains several hyphens")
        if reasons:
            suspicious_url_reasons.append({"url": candidate, "reasons": reasons})

    flags = {
        "Credential / sensitive-data request": int(any(term in text for term in credential_phrases)),
        "Suspicious URL pattern": int(bool(suspicious_url_reasons)),
        "Urgency / pressure": int(any(term in text for term in urgency_terms)),
        "Threat / consequence": int(any(term in text for term in threat_terms)),
        "Money / reward bait": int(any(term in text for term in money_terms)),
        "Multiple URLs": int(len(urls) >= 2),
    }
    points = {name: PHISHING_WEIGHTS[name] * flag for name, flag in flags.items()}
    total = min(100, sum(points.values()))
    verdict = "Lower indicator score" if total < 25 else "Elevated indicators" if total < 50 else "High indicator score"

    sentence_patterns = {
        "Urgency / pressure": urgency_terms,
        "Credential / sensitive-data request": credential_phrases,
        "Threat / consequence": threat_terms,
        "Money / reward bait": money_terms,
    }
    suspicious_sentences = []
    for sentence in re.split(r"(?<=[.!?])\s+|[\r\n]+", email_text):
        cleaned = sentence.strip(" \t\r\n•-–")
        if not cleaned:
            continue
        reasons = [
            indicator for indicator, terms in sentence_patterns.items()
            if any(term in cleaned.lower() for term in terms)
        ]
        if any(cleaned_url in cleaned for item in suspicious_url_reasons for cleaned_url in [item["url"]]):
            reasons.append("Contains a URL with a suspicious pattern")
        if reasons:
            suspicious_sentences.append({"sentence": cleaned, "reasons": list(dict.fromkeys(reasons))})

    return {
        "urls": urls, "flags": flags, "points": points, "score": total,
        "verdict": verdict, "suspicious_sentences": suspicious_sentences,
        "suspicious_urls": suspicious_url_reasons,
    }


def predict_with_model(email_text: str):
    """Use existing text-trained model when available; tolerate model API differences."""
    if phishing_model is None:
        return None, None, ""
    try:
        prediction = phishing_model.predict([email_text])[0]
        # The user's existing model uses 0 = legitimate and 1 = phishing.
        label_text = str(prediction).strip().lower()
        if prediction == 1 or label_text in {"phishing", "spam", "malicious"}:
            label = "Phishing"
        elif prediction == 0 or label_text in {"legitimate", "safe", "ham", "not phishing"}:
            label = "Safe"
        else:
            label = "Suspicious"
        probability = None
        if hasattr(phishing_model, "predict_proba"):
            try:
                probabilities = phishing_model.predict_proba([email_text])[0]
                classes = list(getattr(phishing_model, "classes_", []))
                if 1 in classes:
                    probability = float(probabilities[classes.index(1)])
            except Exception:
                probability = None
        return label, probability, ""
    except Exception as exc:
        return None, None, str(exc)


def render_phishing_result(email_text: str) -> None:
    """Show the ML prediction as primary, plus an explainable rule score."""
    analysis = get_phishing_indicators(email_text)
    model_label, model_probability, model_error = predict_with_model(email_text)

    st.subheader("1. Primary classification — trained machine-learning model")
    st.caption(
        "Your trained TF-IDF + Logistic Regression pipeline is the primary classifier. "
        "The rule score below is a separate heuristic and does not override the model."
    )

    if model_label == "Phishing":
        st.markdown(
            "<div style='display:flex;align-items:center;gap:12px;padding:16px 18px;"
            "border:1px solid #ef4444;border-radius:14px;background:rgba(239,68,68,.13);'>"
            "<span style='font-size:2rem;line-height:1'>🔴</span>"
            "<div><strong style='color:#f87171;font-size:1.2rem'>RED LIGHT · MODEL PREDICTS PHISHING</strong>"
            "<div>Do not click links or share passwords, codes, or payment details.</div></div></div>",
            unsafe_allow_html=True,
        )
    elif model_label == "Safe" and analysis["score"] >= 30:
        st.markdown(
            "<div style='display:flex;align-items:center;gap:12px;padding:16px 18px;"
            "border:1px solid #eab308;border-radius:14px;background:rgba(234,179,8,.13);'>"
            "<span style='font-size:2rem;line-height:1'>🟡</span>"
            "<div><strong style='color:#facc15;font-size:1.2rem'>YELLOW LIGHT · REVIEW RECOMMENDED</strong>"
            "<div>The model predicts legitimate, but the rule-based indicators raise concerns. Verify independently.</div></div></div>",
            unsafe_allow_html=True,
        )
    elif model_label == "Safe":
        st.markdown(
            "<div style='display:flex;align-items:center;gap:12px;padding:16px 18px;"
            "border:1px solid #22c55e;border-radius:14px;background:rgba(34,197,94,.12);'>"
            "<span style='font-size:2rem;line-height:1'>🟢</span>"
            "<div><strong style='color:#4ade80;font-size:1.2rem'>GREEN LIGHT · MODEL PREDICTS LEGITIMATE</strong>"
            "<div>No strong conflict with the rule indicators was detected. Still check sender and links.</div></div></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='display:flex;align-items:center;gap:12px;padding:16px 18px;"
            "border:1px solid #eab308;border-radius:14px;background:rgba(234,179,8,.13);'>"
            "<span style='font-size:2rem;line-height:1'>🟡</span>"
            "<div><strong style='color:#facc15;font-size:1.2rem'>YELLOW LIGHT · MODEL UNAVAILABLE</strong>"
            "<div>Do not treat the rule score alone as a definitive classifier. Review the message manually.</div></div></div>",
            unsafe_allow_html=True,
        )
        if model_error:
            st.caption(f"Model error: {model_error}")

    if model_label is not None:
        st.metric("ML classification", model_label)
        if model_probability is not None:
            st.metric("ML phishing score", f"{model_probability * 100:.1f} / 100")
            st.caption(
                "This score is the model's predict_proba output for class 1 (phishing). "
                "It is not necessarily calibrated to real-world probability."
            )
    elif phishing_model is None:
        st.warning("phishing_model.pkl was not loaded. The app cannot produce its primary ML classification.")

    st.divider()
    st.subheader("2. Explainable rule-based indicator score")
    st.latex(r"S_{rules}=\min\left(100,\sum_{i=1}^{n} w_i I_i\right)")
    st.caption(
        "Each indicator is binary (Iᵢ = 1 when detected, otherwise 0). The weights sum to 100, "
        "so the score is the sum of triggered points. Weights are provisional and not empirically validated; "
        "this score is not a probability and does not replace the ML classification. A URL alone is not treated as suspicious."
    )
    st.markdown("**Indicators and point calculation**")
    for name, flag in analysis["flags"].items():
        points = analysis["points"][name]
        status = "Found" if flag else "Not found"
        st.write(f"- **{name}** — {status}: {flag} × {PHISHING_WEIGHTS[name]} = **{points} points**")
    st.code(
        "Rule score = " + " + ".join(str(analysis["points"][name]) for name in analysis["points"])
        + f" = {analysis['score']}/100",
        language="text",
    )
    st.metric("Rule-based indicator score", f"{analysis['score']}/100")
    st.caption(
        f"Heuristic result: {analysis['verdict']}. This band is provisional; do not interpret it as a verified prediction."
    )
    if analysis["suspicious_urls"]:
        with st.expander("Why a URL was flagged"):
            for item in analysis["suspicious_urls"]:
                st.write(f"**{item['url']}** — " + "; ".join(item["reasons"]))

    st.subheader("3. Suspicious sentence review")
    if analysis["suspicious_sentences"]:
        st.write("These exact OCR-extracted sentence(s) matched common warning terms or contained a visible link:")
        for item in analysis["suspicious_sentences"]:
            st.warning(f"“{item['sentence']}”\n\n**Signals:** {', '.join(item['reasons'])}")
    else:
        st.info(
            "No sentence matched the current keyword rules. The model may still detect phishing patterns, "
            "and absence of keyword matches does not prove legitimacy."
        )

    if analysis["urls"]:
        with st.expander("Visible URLs found"):
            for url in analysis["urls"]:
                st.write(url)


# ---------------------------------- UI -------------------------------------
def render_header() -> None:
    """Render consistent application header and global styling."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    .stApp { background: radial-gradient(ellipse at 12% 0%, rgba(37,99,235,.14), transparent 35%), #0b1120; }
    .block-container { max-width: 1220px; padding-top: 2rem; padding-bottom: 3rem; }
    h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.025em; }
    [data-testid="stMetric"] { background:rgba(24,36,58,.72); border:1px solid rgba(148,163,184,.2); padding:15px; border-radius:15px; }
    [data-testid="stFileUploader"] { background:rgba(20,31,51,.45); border:1px dashed rgba(96,165,250,.5); border-radius:15px; padding:10px; }
    .hero { padding:26px 28px; border-radius:22px; background:linear-gradient(120deg,rgba(30,64,175,.35),rgba(15,118,110,.2)),rgba(15,23,42,.8); border:1px solid rgba(125,211,252,.22); margin-bottom:20px; }
    .eyebrow { color:#7dd3fc; font-size:.75rem; font-weight:700; letter-spacing:.15em; text-transform:uppercase; }
    .hero-title { font-family:'Space Grotesk',sans-serif; font-size:clamp(2rem,4vw,3rem); line-height:1.08; font-weight:700; color:#f8fbff; margin:.55rem 0 .75rem; }
    .hero-copy { max-width:780px; color:#bdcbe0; line-height:1.65; }
    .pill { display:inline-block; padding:5px 10px; margin:8px 6px 0 0; border:1px solid rgba(125,211,252,.25); border-radius:999px; color:#dbeafe; background:rgba(15,23,42,.4); font-size:.78rem; }
    .stButton>button { min-height:44px; border-radius:11px; font-weight:700; }
    [data-testid="stAlert"] { border-radius:12px; }
    </style>
    <div class="hero">
      <div class="eyebrow">Sentinel • Digital Safety Lab</div>
      <div class="hero-title">See the risk before you share.</div>
      <div class="hero-copy">Analyze screenshot privacy exposure and inspect email screenshots for common phishing indicators using transparent, explainable calculations.</div>
      <span class="pill">Mathematical PEI</span><span class="pill">OCR + detector</span><span class="pill">Phishing indicators</span>
    </div>
    """, unsafe_allow_html=True)


def render_status() -> None:
    """Display optional dependency and model status without blocking the app."""
    with st.expander("System status · dependencies and models", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.write("**Privacy detector**")
        c1.success("Available") if detect_sensitive_information else c1.error("detector.py import failed")
        c2.write("**PEI model**")
        c2.success("Loaded") if pei_model is not None else c2.info(f"Optional model unavailable: {pei_model_error}")
        c3.write("**Phishing model**")
        c3.success("Loaded") if phishing_model is not None else c3.info(f"Optional model unavailable: {phishing_model_error}")
        if pytesseract is None:
            st.warning("pytesseract is not installed. Install requirements and the Tesseract OCR system engine for phishing screenshot OCR.")


def render_pei_tool() -> None:
    """Upload, analyze, and calculate the screenshot PEI."""
    st.header("Screenshot Privacy Scanner")
    st.write("Find supported visible personal-information types, then calculate the PEI with an itemized formula.")
    uploaded = st.file_uploader("Upload a privacy screenshot", type=ALLOWED_TYPES, key="privacy_upload", help="Supported formats: PNG, JPG, JPEG. Use a screenshot you have permission to analyze.")
    if uploaded is None:
        st.info("Upload a PNG or JPG screenshot to begin.")
        return
    try:
        image = open_uploaded_image(uploaded)
    except ValueError as exc:
        st.error(str(exc))
        return
    st.image(image, caption="Uploaded screenshot", use_container_width=True)
    with st.expander("PEI formula used for this scan", expanded=True):
        st.latex(r"E_k = w_k(1 - 2^{-n_k})AVC")
        st.latex(r"PEI = 100\times\frac{\sum_k E_k}{R_{\max}},\quad R_{\max}=0.60")
        st.caption("The numerical substitution and final result appear after you run the scan.")
    with st.form("pei_form"):
        st.markdown("**Measurement inputs**")
        col1, col2, col3 = st.columns(3)
        with col1:
            area = st.slider("Relative area (A)", 0.0, 1.0, DEFAULT_AREA, 0.05)
        with col2:
            visibility = st.slider("Visibility (V)", 0.0, 1.0, DEFAULT_VISIBILITY, 0.05)
        with col3:
            confidence = st.slider("Detection confidence (C)", 0.0, 1.0, DEFAULT_CONFIDENCE, 0.05)
        submitted = st.form_submit_button("Detect and calculate PEI", type="primary", use_container_width=True)
    if not submitted:
        return
    with st.spinner("Detecting supported sensitive information..."):
        try:
            found = detect_privacy_items(image)
            counts = count_items(found)
            result = calculate_pei(counts, area, visibility, confidence)
        except Exception as exc:
            st.error("The PEI analysis could not be completed.")
            st.caption(str(exc))
            return
    st.success("Screenshot processed successfully.")
    metrics = st.columns(5)
    for col, label, key in zip(metrics, ["Emails", "Phone numbers", "URLs", "QR codes", "Total detections"], ["Email", "Phone", "URL", "QR Code", "total"]):
        value = sum(counts.values()) if key == "total" else counts[key]
        col.metric(label, value)
    with st.expander("Detected items", expanded=True):
        for label, key in [("Email", "emails"), ("Phone", "phones"), ("URL", "urls")]:
            values = found[key]
            st.markdown(f"**{label}**")
            if values:
                for value in values:
                    st.write(f"- {value}")
            else:
                st.caption("None detected")
        st.markdown("**QR Code**")
        if found["qr_detected"]:
            st.write("QR code detected")
            if found["qr_data"]:
                st.code(found["qr_data"])
        else:
            st.caption("None detected")
    show_pei_calculation(result)
    out1, out2, out3 = st.columns(3)
    out1.metric("Raw exposure R", f"{result['raw_total']:.4f}")
    out2.metric("Privacy Exposure Index", f"{result['score']:.2f} / 100")
    out3.metric("Exposure category", result["level"])
    if result["level"] == "LOW":
        st.success(f"PEI result: LOW — {result['score']:.2f}/100")
        missing_categories = [
            category for category, count in counts.items() if count == 0
        ]
        if missing_categories:
            st.warning(
                "Debug note: the LOW score may be caused by undetected categories: "
                + ", ".join(missing_categories)
                + ". Check the detected-items list and OCR text below. "
                "The PEI formula only scores categories that the detector finds."
            )
        elif area == 1.0 and visibility == 1.0 and confidence >= 0.9:
            st.error(
                "Calculation diagnostic: all four categories were detected, "
                "but the score is LOW. With the current weights and formula, "
                "this combination should not produce a LOW score. Please review "
                "the calculation values and deployed version."
            )
    elif result["level"] == "MODERATE":
        st.warning(f"PEI result: MODERATE — {result['score']:.2f}/100")
    else:
        st.error(f"PEI result: HIGH — {result['score']:.2f}/100")
    if pei_model is not None:
        st.subheader("Optional PEI machine-learning comparison")
        try:
            model_input = [[counts["Email"], counts["Phone"], counts["URL"], counts["QR Code"], result["raw_total"], result["score"]]]
            prediction = str(pei_model.predict(model_input)[0]).strip().upper()
            st.write(f"PEI model prediction: **{prediction}**")
            if prediction == result["level"]:
                st.success("Model prediction matches the formula-based category.")
            else:
                st.warning("Model prediction differs from the formula-based category.")
        except Exception as exc:
            st.warning("The optional PEI model could not process its six expected numeric features; the mathematical PEI remains available.")
            st.caption(str(exc))
    with st.expander("OCR text returned by detector"):
        st.text(found["text"] or "No OCR text was returned.")


def render_phishing_tool() -> None:
    """Analyze an email screenshot with OCR or directly pasted email text."""
    st.header("Email Phishing Detector")
    st.write(
        "Analyze an email screenshot using OCR or paste email text directly. "
        "The trained text model provides the primary classification, while "
        "the rule-based indicators explain additional warning signs."
    )

    input_method = st.radio(
        "Choose your input method:",
        ["Upload screenshot", "Paste email text"],
        horizontal=True,
        key="email_input_method",
    )

    email_text = ""

    if input_method == "Upload screenshot":
        uploaded = st.file_uploader(
            "Upload an email screenshot",
            type=ALLOWED_TYPES,
            key="phishing_upload",
            help="Supported formats: PNG, JPG, JPEG. OCR accuracy depends on image clarity.",
        )
        if uploaded is None:
            st.info("Upload a PNG or JPG screenshot of an email to begin.")
            return

        try:
            image = open_uploaded_image(uploaded)
        except ValueError as exc:
            st.error(str(exc))
            return

        st.image(image, caption="Uploaded email screenshot", use_container_width=True)
        if not st.button(
            "Analyze email screenshot",
            type="primary",
            key="analyze_phishing",
            use_container_width=True,
        ):
            return

        with st.spinner("Extracting screenshot text with OCR..."):
            try:
                email_text = extract_text(image)
            except Exception as exc:
                st.error("The phishing analysis could not be completed.")
                st.caption(str(exc))
                return

        st.subheader("Extracted text (check for OCR errors)")
        st.text_area(
            "Recognized email content",
            value=email_text,
            height=190,
            key="ocr_email_text",
        )

    else:
        pasted_text = st.text_area(
            "Paste the email or message here",
            height=220,
            placeholder="Paste the sender, message body, and any visible links here...",
            key="pasted_email_text",
        )
        if not st.button(
            "Analyze pasted email",
            type="primary",
            key="analyze_pasted_email",
            use_container_width=True,
        ):
            return
        email_text = pasted_text.strip()
        if not email_text:
            st.warning("Please paste some email text before analyzing.")
            return

    st.divider()
    render_phishing_result(email_text)


def main() -> None:
    """Render app navigation and selected tool."""
    render_header()
    render_status()
    privacy_tab, phishing_tab, formula_tab = st.tabs(["🔐 PEI Calculator", "📧 Phishing Detector", "∑ Formula & Validation"])
    with privacy_tab:
        render_pei_tool()
    with phishing_tab:
        render_phishing_tool()
    with formula_tab:
        st.header("Formulas and interpretation")
        st.info("The equations below are the exact formulas used by this app. PEI weights and phishing thresholds are configurable research assumptions, not universally accepted standards; validate them before reporting scientific conclusions.")
        st.subheader("Privacy Exposure Index")
        st.latex(r"E_k = w_k(1 - 2^{-n_k})AVC")
        st.latex(r"PEI = 100\times\frac{\sum_k E_k}{R_{\max}},\quad R_{\max}=0.60")
        st.write("Category weights: Email 0.15, Phone 0.15, URL 0.10, QR Code 0.20. A, V, and C are values from 0 to 1. Repeated detections contribute with diminishing returns so the PEI remains bounded by 100.")
        st.subheader("Phishing indicator score")
        st.latex(r"S = \min\left(100,\sum_i w_i I_i\right)")
        st.write("The rule-based verdict is Safe for scores below 30, Suspicious for scores from 30 to below 60, and Phishing for scores of 60 or above. These are provisional research thresholds, not empirically validated probabilities.")
        st.subheader("Validation examples")
        st.markdown("**Example 1 — one email, A = 1.00, V = 1.00, C = 0.90**")
        sample1 = calculate_pei({"Email": 1, "Phone": 0, "URL": 0, "QR Code": 0}, 1, 1, 0.9)
        st.code(f"R = 0.15 × (1 − 2⁻¹) × 1.00 × 1.00 × 0.90 = 0.0675\nPEI = 100 × 0.0675 / 0.60 = {sample1['score']:.2f}")
        st.markdown("**Example 2 — one email, one phone, A = 1.00, V = 1.00, C = 0.90**")
        sample2 = calculate_pei({"Email": 1, "Phone": 1, "URL": 0, "QR Code": 0}, 1, 1, 0.9)
        st.code(f"R = 0.0675 + 0.0675 = 0.1350\nPEI = 100 × 0.1350 / 0.60 = {sample2['score']:.2f}")
        st.caption("For a defensible research result, document how area, visibility, confidence, weights, and thresholds were selected, then validate them against independently labelled screenshots.")
    st.divider()
    st.caption("Sentinel is a research prototype. OCR and rule/model predictions can be wrong; validate the detector, weights, thresholds, and model on independent test data before making research claims.")


if __name__ == "__main__":
    main()
