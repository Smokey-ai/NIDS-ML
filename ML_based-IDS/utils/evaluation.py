import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score
)

class Evaluator:
    def __init__(self, results_path="results/"):
        self.results_path = results_path
        os.makedirs(results_path, exist_ok=True)

        # 41-class → 5-class mapping
        self.class_mapping = {
            # Normal
            "normal": "Normal",

            # DoS
            "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
            "smurf": "DoS", "teardrop": "DoS",

            # Probe
            "satan": "Probe", "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",

            # R2L
            "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L", "multihop": "R2L",
            "phf": "R2L", "spy": "R2L", "warezclient": "R2L", "warezmaster": "R2L",

            # U2R
            "buffer_overflow": "U2R", "loadmodule": "U2R",
            "perl": "U2R", "rootkit": "U2R"
        }

    def _map_to_category(self, labels):
        """Convert fine-grained labels to 5 broad categories."""
        return labels.map(self.class_mapping).fillna("Other")

    def _plot_confusion_matrix(self, cm, labels, title, save_path):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.ylabel("True")
        plt.xlabel("Predicted")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def evaluate(self, model, X_test, y_test, model_name):
        """Evaluate any classifier (DT, XGB, etc.) on both 41-class and 5-class levels."""
        # Predict
        y_pred = model.predict(X_test)

        # -----------------
        # Fine-grained (41-class)
        # -----------------
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        print(f"\n=== {model_name} (41-class) ===")
        print(classification_report(y_test, y_pred, zero_division=0))
        print(f"Accuracy: {acc:.4f}, Macro-F1: {macro_f1:.4f}, Weighted-F1: {weighted_f1:.4f}")

        # Save reports
        pd.DataFrame(report).transpose().to_csv(f"{self.results_path}/{model_name}_41class.csv")
        with open(f"{self.results_path}/{model_name}_41class.txt", "w") as f:
            f.write(classification_report(y_test, y_pred, zero_division=0))
        self._plot_confusion_matrix(cm, sorted(y_test.unique()),
                                    f"{model_name} Confusion Matrix (41 classes)",
                                    f"{self.results_path}/{model_name}_cm_41class.png")

        # -----------------
        # Category-level (5-class)
        # -----------------
        y_test_cat = self._map_to_category(y_test)
        y_pred_cat = self._map_to_category(pd.Series(y_pred))

        acc_cat = accuracy_score(y_test_cat, y_pred_cat)
        report_cat = classification_report(y_test_cat, y_pred_cat, output_dict=True, zero_division=0)
        cm_cat = confusion_matrix(y_test_cat, y_pred_cat)

        print(f"\n=== {model_name} (5-class) ===")
        print(classification_report(y_test_cat, y_pred_cat, zero_division=0))
        print(f"Accuracy: {acc_cat:.4f}")

        pd.DataFrame(report_cat).transpose().to_csv(f"{self.results_path}/{model_name}_5class.csv")
        with open(f"{self.results_path}/{model_name}_5class.txt", "w") as f:
            f.write(classification_report(y_test_cat, y_pred_cat, zero_division=0))
        self._plot_confusion_matrix(cm_cat, sorted(y_test_cat.unique()),
                                    f"{model_name} Confusion Matrix (5 classes)",
                                    f"{self.results_path}/{model_name}_cm_5class.png")

        return {
            "41class": {"acc": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1, "report": report, "cm": cm},
            "5class": {"acc": acc_cat, "report": report_cat, "cm": cm_cat}
        }
