import os
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from utils.preprocessing import load_data
from utils.evaluation import Evaluator
from utils.feature_selection import drop_correlated_features, select_features_mi 

TEST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/nsl-kdd/clean-test.csv"))
TRAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/nsl-kdd/clean-train.csv"))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results/"))

def train_xgboost(
    train_path=TRAIN_DIR,
    test_path=TEST_DIR,
    results_path=RESULTS_DIR,
    use_gpu=False
):
    # 1. Load data (labels are strings)
    X_train, X_test, y_train, y_test, encoders = load_data(train_path, test_path)

    # 🔹 2. Apply feature selection
    # Step 1: drop correlated
    X_train, dropped_corr = drop_correlated_features(X_train, threshold=0.9)
    X_test = X_test[X_train.columns]  # align test set

    # Step 2: mutual information (keep top 20 for example)
    X_train, mi_scores = select_features_mi(X_train, y_train, top_k=20)
    X_test = X_test[X_train.columns]  # align test set again

    # ---- Minimal, robust label mapping: make labels 0..N-1 ----
    labels_union = pd.concat([y_train.astype(str), y_test.astype(str)])
    unique_labels = pd.unique(labels_union)  # numpy array of unique labels in order

    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    y_train_enc = y_train.astype(str).map(label2id).to_numpy()
    y_test_enc  = y_test.astype(str).map(label2id).to_numpy()

    if np.any(pd.isna(y_train_enc)) or np.any(pd.isna(y_test_enc)):
        missing = set(labels_union.unique()) - set(label2id.keys())
        raise ValueError(f"Unmapped labels found: {missing}")

    # 3. Initialize baseline XGB
    params = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "eval_metric": "mlogloss"
    }
    if use_gpu:
        params.update({
            "tree_method": "gpu_hist",
            "predictor": "gpu_predictor"
        })

    xgb = XGBClassifier(**params)

    # 4. Train
    xgb.fit(X_train, y_train_enc)

    # 5. Predict + map back to string labels
    y_pred_enc = xgb.predict(X_test)
    y_pred = [id2label[int(i)] for i in y_pred_enc]

    # 6. Evaluate
    evaluator = Evaluator(results_path=results_path)
    class _PredWrapper:
        def __init__(self, preds): self._preds = preds
        def predict(self, X): return self._preds

    wrapper = _PredWrapper(y_pred)
    results = evaluator.evaluate(wrapper, X_test, y_test, model_name="xgb")

    acc = results["41class"]["acc"]
    acc_cat = results["5class"]["acc"]

    print(f"[INFO] Baseline XGB Accuracy (41-class): {acc:.4f}")
    print(f"[INFO] Baseline XGB Accuracy (5-class): {acc_cat:.4f}")

    return xgb, label2id, id2label, results

if __name__ == "__main__":
    train_xgboost()
