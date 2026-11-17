"""Explicit command-line entry point for SMS model training and evaluation."""
import json
from sms_module import load_sms_dataset, train_and_evaluate

if __name__ == "__main__":
    dataset = load_sms_dataset()
    report = train_and_evaluate(dataset)
    print(json.dumps({
        "selected_model": report["selected_model"],
        "dataset_records_loaded": report["dataset_records_loaded"],
        "modeling_records_after_review": report["modeling_records_after_review"],
        "train_records": report["train_records"],
        "test_records": report["test_records"],
        "models": {
            name: {metric: round(result[metric], 4) for metric in
                   ("accuracy", "precision_macro", "recall_macro", "f1_macro")}
            for name, result in report["models"].items()
        }
    }, indent=2))
