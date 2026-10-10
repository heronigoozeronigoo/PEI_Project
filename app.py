                            st.warning(f"PEI model could not score this input. The formula-based PEI above is still available. Details: {exc}")
                    st.caption("Interpretation: higher PEI means more detected exposure under the current weights and selected assumptions; it does not prove that an account has been compromised.")
        except Exception as exc:
            st.error(f"Could not open this image: {exc}")

with phishing_tab:
    st.header("Email Screenshot Threat Checker")
    st.write("Turn an email screenshot into readable text, evaluate phishing indicators, and review the mathematical logic behind the result.")
    st.markdown("""
    <div class="result-card">
      <div class="small-label">Analysis workflow</div>
      <div style="font-size:1.03rem;color:#edf4ff;margin-top:8px;">Screenshot → OCR text extraction → trained classifier or transparent fallback rules → risk interpretation</div>
    </div>
    """, unsafe_allow_html=True)
    email_file = st.file_uploader("Upload email screenshot", type=["png", "jpg", "jpeg", "webp"], key="email_upload")
    if email_file:
        try:
            email_image = Image.open(email_file).convert("RGB")
            st.image(email_image, caption="Email screenshot", use_container_width=True)
            if st.button("Analyze email screenshot", key="analyze_email", use_container_width=True):
                try:
                    email_text = ocr_image(email_image)
                except RuntimeError as exc:
                    st.error(str(exc))
                    email_text = ""
                if email_text:
                    with st.expander("Text extracted from screenshot", expanded=False):
                        st.text(email_text)
                    label, probability, method = phishing_prediction(email_text)
                    st.subheader("Analysis result")
                    if "phishing" in label.lower() or "suspicious" in label.lower():
                        st.error(label)
                    elif "caution" in label.lower():
                        st.warning(label)
                    else:
                        st.info(label)
                    st.caption(f"Analysis method: {method}.")
                    if probability is not None:
                        st.metric("Model phishing probability", f"{probability * 100:.1f}%")
                        st.progress(max(0.0, min(1.0, probability)))
                    elif "heuristic" in method:
                        lower_text = email_text.lower()
                        signal_tests = {
                            "Urgency or threat": any(x in lower_text for x in ["urgent", "immediately", "suspended", "will be closed", "act now"]),
                            "Credential or payment request": any(x in lower_text for x in ["password", "verify your account", "credit card", "bank details", "payment information"]),
                            "Shortened or unusual link pattern": any(x in lower_text for x in ["bit.ly/", "tinyurl.com/", "login-", "secure-"]),
                            "Unexpected prize or reward": any(x in lower_text for x in ["you won", "claim your prize", "free gift", "winner"]),
                        }
                        hits = sum(signal_tests.values())
                        risk_score = hits / len(signal_tests) * 100
                        c1, c2 = st.columns(2)
                        c1.metric("Rule-based signal score", f"{risk_score:.0f}%")
                        c2.metric("Signals matched", f"{hits} / {len(signal_tests)}")
                        st.progress(risk_score / 100)
                        st.write("**Matched warning signals**")
                        for signal, matched in signal_tests.items():
                            st.write(("✓" if matched else "—") + " " + signal)
                    with st.expander("Phishing formula and decision logic", expanded=True):
                        st.markdown("**A. When the trained model is available**")
                        st.latex(r"x = \operatorname{TFIDF}(T)")
                        st.latex(r"\hat{y} = f_{\theta}(x)")
                        st.markdown("Here, $T$ is the OCR-extracted email text, TF-IDF converts text into numeric features, and $f_\theta$ is the classifier learned during training. If the model supports calibrated class probabilities, the displayed probability is the model's estimated phishing-class probability—not a guarantee that the email is malicious.")
                        st.markdown("**B. If no trained model is loaded: transparent rule-based score**")
                        st.latex(r"S = \frac{H}{N} \times 100")
                        st.markdown("$H$ = number of matched warning-signal groups; $N$ = total warning-signal groups checked (currently 4). The fallback labels 0 matched signals as ‘No common warning signs found,’ 1 as ‘Use caution,’ and 2 or more as ‘Suspicious signs detected.’ This is a simple heuristic score, not a trained-model probability.")
                        st.markdown("**C. Validation plan**")
                        st.write("Evaluate on a separate, labeled email dataset that was not used for training. Report a confusion matrix, precision, recall, F1-score, accuracy, and false-positive/false-negative counts. For probability outputs, also assess calibration. Review errors and avoid tuning the model on the final test set.")
                elif not email_text:
                    st.info("No readable text was extracted. Try a clearer image or a closer crop of the email body.")
        except Exception as exc:
            st.error(f"Could not open this image: {exc}")

with st.expander("Mathematical Formula & Validation", expanded=True):
    st.markdown("### 1. PEI formula")
    st.latex(r"E_k = w_k(1-2^{-n_k})A V C")
    st.markdown("Each category's contribution grows as more items are detected, with diminishing returns. A category is capped at its assigned weight, so repeated detections cannot alone force the PEI to 100.")
    st.markdown("Where:")
    st.markdown("- **$w_k$** = sensitivity weight for category $k$")
    st.markdown("- **$n_k$** = number of distinct detected items in category $k$")
    st.markdown("- **$A,V,C$** = area, visibility, and detection-confidence ratings (each 0–1)")
    st.latex(r"E_{total} = \sum_k E_k")
    st.latex(r"PEI = 100 \times \frac{E_{total}}{\sum_k w_k}")
    st.markdown(f"Normalization is **$R_{{max}} = \sum w_k = {R_MAX:.2f}$**. Weights: " + ", ".join(f"{k} = {v:.2f}" for k, v in WEIGHTS.items()) + ".")
    st.caption("This is a proposed scoring formula. The weights and score bands still need empirical validation against independently labeled screenshots.")
    st.markdown("**Current score bands:** LOW = 0–33.33, MODERATE = above 33.33–66.67, HIGH = above 66.67–100.")

    st.markdown("### 2. How to validate the PEI model")
    st.write("The formula produces a score from the selected weights and ratings. To validate whether the score reflects real exposure, compare it against independent human ratings for a labeled set of screenshots.")
    st.markdown("1. Prepare a test set of screenshots with consent and remove real private information.")
    st.markdown("2. Have at least two reviewers independently rate each screenshot's exposure level using the same LOW/MODERATE/HIGH rubric.")
    st.markdown("3. Calculate PEI for every screenshot without changing the weights after seeing the test labels.")
    st.markdown("4. Compare PEI bands with reviewer labels using **confusion matrix, accuracy, precision, recall, and F1-score**. If reviewers provide numeric scores, also report **MAE** and **Spearman correlation**.")
    st.markdown("5. Report the sample size, class distribution, disagreements between reviewers, and all metrics. Do not claim validation results until these tests have actually been run.")

    st.markdown("### 3. How to validate phishing detection")
    st.write("Test the phishing model on a separate labeled dataset that was not used for training. Report confusion matrix, precision, recall, F1-score, and false-positive/false-negative counts. Keep the test set separate from training data to avoid data leakage.")

with st.expander("System status / troubleshooting"):
    st.write(f"OCR library available: **{'Yes' if pytesseract is not None else 'No'}**")
    st.write(f"PEI model loaded: **{'Yes' if pei_model is not None else 'No'}**")
    st.write(f"Phishing model loaded: **{'Yes' if phishing_model is not None else 'No'}**")
    st.write(f"QR scanner available: **{'Yes' if cv2 is not None else 'No'}**")
    st.caption("If deploying to Streamlit Community Cloud, requirements.txt should include streamlit, pillow, pytesseract, joblib, scikit-learn, opencv-python-headless, and numpy. Add a packages.txt file containing tesseract-ocr for the system OCR engine. Keep model files in the same repository folder as app.py.")
