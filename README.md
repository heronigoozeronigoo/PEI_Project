# Sentinel — Screenshot Privacy, Email Phishing & SMS Smishing

Sentinel is a Streamlit research prototype with three analysis areas:
1. Screenshot Privacy Assessment (existing PEI scanner)
2. Email Phishing Detection (existing OCR/text scanner)
3. SMS Smishing Detection (three-class TF-IDF classifier)

## SMS dataset setup

Place the supplied file at:

```
data/SMS_Phishing_Dataset_Cleaned_5971.csv
```

The application expects the columns `LABEL`, `TEXT`, `URL`, `EMAIL`, and `PHONE`. The supplied file includes 5,971 records and labels ham, spam, and smishing. Keep the original dataset file unchanged and retain its source/provenance documentation with your research materials. The CSV supplied to this project did not contain a publisher or source URL, so this repository does not attribute it to an unverified source.

## Train the SMS model

Install dependencies, then run:

```bash
python -m pip install -r requirements.txt
python train_sms_model.py
```

This audits exact duplicate messages and conflicting labels, excludes all rows belonging to messages with conflicting labels, removes repeated same-label texts, performs a stratified 80/20 split (random_state=42), compares Multinomial Naive Bayes, Logistic Regression, and Linear SVM, and selects by macro F1. TF-IDF is fitted only inside each model pipeline on the training split. Class balancing is applied during fitting. Outputs:
- `models/sms_classifier.joblib`
- `models/sms_evaluation.json`

If the model files are absent, the Streamlit SMS module attempts to train them from the CSV on first use. The app reports an error if the CSV is missing or invalid.

## Run locally

```bash
python -m streamlit run app.py
```

For screenshot OCR, install the Tesseract OCR system executable in addition to the Python `pytesseract` package. See `packages.txt` for the Streamlit Cloud system package configuration.

## Streamlit Community Cloud

Deploy this repository with `app.py` as the entry point. Ensure the dataset exists at `data/SMS_Phishing_Dataset_Cleaned_5971.csv` in the deployed repository before expecting SMS classification to work. Include the generated model files if you want to avoid training on the first app visit; otherwise the app trains from the committed dataset.

## Research and interpretation notes

- Dataset class counts describe the CSV; they are not predictions.
- Evaluation metrics describe the held-out test partition only.
- Macro F1 is used to select a model because the classes are imbalanced.
- HAM is not proof of safety. The classifier is a research aid and can make mistakes.
- The provided CSV metadata does not identify the original publisher. Verify and cite the actual source in the thesis/report.
