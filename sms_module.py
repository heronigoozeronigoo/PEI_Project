"""SMS smishing classifier, dataset audit, and Streamlit analytics."""
from pathlib import Path
import json, re
import pandas as pd
import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight

BASE = Path(__file__).resolve().parent
# Accept the documented data/ location and the filename currently committed
# at the repository root. This prevents Streamlit Cloud from failing when the
# dataset was uploaded to the root with GitHub's duplicate-safe "_2" suffix.
DATA_CANDIDATES = (
    BASE / "data" / "SMS_Phishing_Dataset_Cleaned_5971.csv",
    BASE / "SMS_Phishing_Dataset_Cleaned_5971.csv",
    BASE / "SMS_Phishing_Dataset_Cleaned_5971_2.csv",
)
DATA_PATH = next((path for path in DATA_CANDIDATES if path.is_file()), DATA_CANDIDATES[0])
MODEL_PATH = BASE / "models" / "sms_classifier.joblib"
METRICS_PATH = BASE / "models" / "sms_evaluation.json"
LABELS = ["ham", "spam", "smishing"]
URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d ()-]{7,}\d)(?!\w)")

def _normalize_text(s):
    return re.sub(r"\s+", " ", str(s).lower().strip())

@st.cache_data(show_spinner=False)
def load_sms_dataset():
    if not DATA_PATH.is_file():
        expected = ", ".join(str(path.relative_to(BASE)) for path in DATA_CANDIDATES)
        raise FileNotFoundError(
            "SMS dataset not found. Add the supplied CSV to one of these locations: " + expected
        )
    # utf-8-sig safely handles the optional BOM often present in exported CSVs.
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    required = {"LABEL", "TEXT", "URL", "EMAIL", "PHONE"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")
    df = df.dropna(subset=["LABEL", "TEXT"]).copy()
    df["LABEL"] = df["LABEL"].astype(str).str.strip().str.lower()
    df["TEXT"] = df["TEXT"].astype(str)
    unknown = sorted(set(df["LABEL"]) - set(LABELS))
    if unknown:
        raise ValueError(f"Unexpected dataset labels: {unknown}")
    if (df["TEXT"].str.strip() == "").any():
        raise ValueError("Dataset contains blank SMS text values.")
    return df

def train_and_evaluate(df):
    grouped = df.groupby("TEXT")["LABEL"].nunique()
    conflict_texts = set(grouped[grouped > 1].index)
    conflict_rows = int(df["TEXT"].isin(conflict_texts).sum())
    duplicate_text_rows = int(df.duplicated("TEXT", keep=False).sum())
    clean = df[~df["TEXT"].isin(conflict_texts)].copy()
    before = len(clean)
    clean = clean.drop_duplicates(subset=["TEXT", "LABEL"], keep="first")
    same_label_duplicates_removed = before - len(clean)
    if set(clean["LABEL"].unique()) != set(LABELS):
        raise ValueError("One or more classes are absent after duplicate/conflict review.")
    X_train, X_test, y_train, y_test = train_test_split(
        clean["TEXT"], clean["LABEL"], test_size=0.20, random_state=42, stratify=clean["LABEL"])
    candidates = {
        "Multinomial Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=100000, sublinear_tf=True)),
            ("classifier", MultinomialNB(alpha=0.5))]),
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=100000, sublinear_tf=True)),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))]),
        "Linear SVM": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=100000, sublinear_tf=True)),
            ("classifier", LinearSVC(class_weight="balanced", random_state=42))])
    }
    scores, fitted = {}, {}
    for name, pipe in candidates.items():
        if name == "Multinomial Naive Bayes":
            pipe.fit(X_train, y_train, classifier__sample_weight=compute_sample_weight("balanced", y_train))
        else:
            pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        p, r, f, _ = precision_recall_fscore_support(y_test, pred, labels=LABELS, average="macro", zero_division=0)
        scores[name] = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision_macro": float(p), "recall_macro": float(r), "f1_macro": float(f),
            "classification_report": classification_report(y_test, pred, labels=LABELS, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, pred, labels=LABELS).tolist()
        }
        fitted[name] = pipe
    best = max(scores, key=lambda n: (scores[n]["f1_macro"], scores[n]["accuracy"]))
    summary = {
        "dataset_records_loaded": int(len(df)),
        "dataset_label_counts": {k:int(v) for k,v in df["LABEL"].value_counts().reindex(LABELS, fill_value=0).items()},
        "duplicate_text_rows_including_conflicts": duplicate_text_rows,
        "conflicting_unique_messages": int(len(conflict_texts)),
        "rows_excluded_due_to_conflicting_labels": conflict_rows,
        "same_label_duplicate_rows_removed": int(same_label_duplicates_removed),
        "modeling_records_after_review": int(len(clean)),
        "train_records": int(len(X_train)), "test_records": int(len(X_test)),
        "test_label_counts": {k:int(v) for k,v in y_test.value_counts().reindex(LABELS, fill_value=0).items()},
        "class_order": LABELS, "selection_metric": "macro F1", "selected_model": best,
        "models": scores,
        "preprocessing": "Lowercase and trim; collapse whitespace; TF-IDF word unigrams and bigrams; min_df=2; sublinear_tf=True. Vectorizer fit only within training Pipeline.",
        "imbalance_handling": "Balanced class weights for Logistic Regression and Linear SVM; balanced per-row sample weights for Multinomial Naive Bayes.",
        "split": "Stratified 80/20 train/test split; random_state=42; exact conflicting message texts excluded; exact same-label duplicates deduplicated before splitting.",
        "source_note": "User-provided SMS_Phishing_Dataset_Cleaned_5971.csv. Original publisher/source provenance was not specified in the supplied file."
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": fitted[best], "model_name": best, "labels": LABELS}, MODEL_PATH, compress=3)
    METRICS_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

@st.cache_resource(show_spinner=False)
def load_sms_model():
    if not MODEL_PATH.exists():
        df = load_sms_dataset()
        train_and_evaluate(df)
    return joblib.load(MODEL_PATH)

@st.cache_data(show_spinner=False)
def load_sms_metrics():
    if not METRICS_PATH.exists():
        df = load_sms_dataset()
        train_and_evaluate(df)
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))

def analyze_indicators(message):
    return {
        "URLs": URL_RE.findall(message),
        "Email addresses": EMAIL_RE.findall(message),
        "Phone numbers": PHONE_RE.findall(message)
    }

def explain_prediction(label, indicators):
    signals = [f"{k.lower()}: {len(v)} detected" for k,v in indicators.items() if v]
    if label == "smishing":
        base = "The trained model identifies patterns associated with SMS-based phishing. Do not open links, share credentials, or call numbers in the message until independently verified."
    elif label == "spam":
        base = "The trained model classifies this as unwanted or promotional spam. It may be unsolicited; avoid replying or following unexpected instructions."
    else:
        base = "The trained model classifies this as a likely ordinary message based on patterns learned from the dataset. HAM does not guarantee that a message is safe."
    return base + ((" Detected indicators: " + "; ".join(signals) + ".") if signals else " No URL, email address, or phone number pattern was detected by the indicator scanner.")

def render_sms_module():
    st.header("SMS Smishing Detection")
    st.write("Paste an SMS to classify it as HAM, SPAM, or SMISHING using the trained TF-IDF text pipeline.")
    try:
        df = load_sms_dataset()
        bundle = load_sms_model()
        metrics = load_sms_metrics()
    except Exception as exc:
        st.error("SMS detection resources are unavailable.")
        st.warning(str(exc))
        st.info("Place SMS_Phishing_Dataset_Cleaned_5971.csv inside the project's data/ folder, then rerun the app. The model and evaluation report are generated from that file.")
        return

    with st.form("sms_analysis_form"):
        message = st.text_area("SMS message", height=150, placeholder="Paste or type the full SMS message here...")
        submitted = st.form_submit_button("Analyze SMS", type="primary", use_container_width=True)
    if submitted:
        if not message.strip():
            st.warning("Enter an SMS message before analyzing.")
        else:
            prediction = str(bundle["pipeline"].predict([message])[0]).upper()
            indicators = analyze_indicators(message)
            st.session_state["sms_last_result"] = {"message": message, "prediction": prediction, "indicators": indicators}
    result = st.session_state.get("sms_last_result")
    if result:
        label = result["prediction"]
        st.subheader("Classification result")
        if label == "SMISHING": st.error("SMISHING — SMS-based phishing message")
        elif label == "SPAM": st.warning("SPAM — Unwanted or suspicious message")
        else: st.success("HAM — Likely legitimate message")
        st.write(explain_prediction(label.lower(), result["indicators"]))
        cols = st.columns(3)
        for col, (name, values) in zip(cols, result["indicators"].items()):
            col.metric(name, len(values))
            if values:
                with col.expander(f"View {name.lower()}"):
                    for value in values: st.code(value)
        st.caption("This is a machine-learning classification, not a guarantee. Verify unexpected requests through an official channel.")

    st.divider()
    st.subheader("SMS Dataset Analytics")
    counts = df["LABEL"].value_counts().reindex(LABELS, fill_value=0)
    a,b,c,d = st.columns(4)
    a.metric("Total SMS records", f"{len(df):,}")
    b.metric("HAM", f"{counts['ham']:,}")
    c.metric("SPAM", f"{counts['spam']:,}")
    d.metric("SMISHING", f"{counts['smishing']:,}")
    st.caption("Dataset statistics describe the supplied labeled CSV; they are not live user-message predictions.")
    st.bar_chart(counts.rename(index={"ham":"HAM","spam":"SPAM","smishing":"SMISHING"}))
    indicator_counts = {}
    for col, label in [("URL","URL"),("EMAIL","Email"),("PHONE","Phone")]:
        indicator_counts[label] = int(df[col].astype(str).str.strip().str.lower().isin(["yes","true","1"]).sum()) if col in df else 0
    st.write("**Dataset indicator counts**")
    x,y,z = st.columns(3)
    for col, (name, value) in zip([x,y,z], indicator_counts.items()): col.metric(name, f"{value:,}")

    st.divider()
    st.subheader("Model comparison and held-out evaluation")
    st.caption(f"Selected model: {metrics['selected_model']} · selection criterion: highest macro F1 on the stratified test set.")
    table = pd.DataFrame([{ "Model": n, "Accuracy": v["accuracy"], "Macro precision": v["precision_macro"], "Macro recall": v["recall_macro"], "Macro F1": v["f1_macro"] } for n,v in metrics["models"].items()])
    st.dataframe(table.style.format({k:"{:.3f}" for k in ["Accuracy","Macro precision","Macro recall","Macro F1"]}), use_container_width=True, hide_index=True)
    st.subheader("Confusion matrix — selected model")
    cm = metrics["models"][metrics["selected_model"]]["confusion_matrix"]
    cm_df = pd.DataFrame(cm, index=["Actual HAM","Actual SPAM","Actual SMISHING"], columns=["Predicted HAM","Predicted SPAM","Predicted SMISHING"])
    st.dataframe(cm_df, use_container_width=True)
    st.caption(f"Audit: {metrics['conflicting_unique_messages']} unique message texts had conflicting labels ({metrics['rows_excluded_due_to_conflicting_labels']} rows excluded); {metrics['same_label_duplicate_rows_removed']} same-label duplicate rows removed. Modeling set: {metrics['modeling_records_after_review']:,}; test set: {metrics['test_records']:,}.")
    with st.expander("Research methodology and data provenance"):
        st.write(metrics["preprocessing"])
        st.write(metrics["imbalance_handling"])
        st.write(metrics["split"])
        st.write(metrics["source_note"])
        st.write("No dataset publisher or original source URL was included in the supplied CSV metadata. This app identifies it as the user-provided cleaned dataset and does not invent a publisher.")
        st.json({"Original class counts": metrics["dataset_label_counts"], "Test class counts": metrics["test_label_counts"]})
