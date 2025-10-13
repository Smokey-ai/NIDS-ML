import os
from sklearn.tree import DecisionTreeClassifier
from utils.preprocessing import load_data
from utils.evaluation import Evaluator
from imblearn.over_sampling import SMOTE
from utils.feature_selection import drop_correlated_features, select_features_mi

# Dataset configurations
DATASETS = {
    "nsl-kdd": {
        "train_path": os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/nsl-kdd/clean-train.csv")),
        "test_path": os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/nsl-kdd/clean-test.csv")),
        "type": "nsl-kdd"
    },
    "cicids2017": {
        "train_path": os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/cicids2017/Cicid2017_clean.csv")),
        "test_path": None,  # We'll split this file
        "type": "cicids2017"
    }
}

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results/"))

def train_decision_tree(
    dataset="nsl-kdd",  # 🔹 CHANGED: Use dataset name instead of paths
    results_path=RESULTS_DIR
):
    # 🔹 NEW: Get dataset configuration
    if dataset not in DATASETS:
        raise ValueError(f"Dataset '{dataset}' not supported. Choose from: {list(DATASETS.keys())}")
    
    config = DATASETS[dataset]
    train_path = config["train_path"]
    test_path = config["test_path"]
    dataset_type = config["type"]

    # 1. Load data
    X_train, X_test, y_train, y_test, encoders = load_data(train_path, test_path, dataset_type=dataset_type)   
    # 🔹 2. Feature selection
    # Step 1: drop correlated
    X_train, dropped_corr = drop_correlated_features(X_train, threshold=0.9)
    X_test = X_test[X_train.columns]  # align

    # Step 2: mutual information (keep top 20 for example)
    X_train, mi_scores = select_features_mi(X_train, y_train, top_k=20)
    X_test = X_test[X_train.columns]  # align again

    # 3. Fix imbalance with SMOTE
    sm = SMOTE(random_state=42, k_neighbors=1)
    X_train, y_train = sm.fit_resample(X_train, y_train)

    # 4. Initialize baseline DT
    dt = DecisionTreeClassifier(random_state=42, class_weight="balanced")

    # 5. Train
    dt.fit(X_train, y_train)

    # 6. Evaluate with new Evaluator
    evaluator = Evaluator(results_path=results_path)
    results = evaluator.evaluate(dt, X_test, y_test, model_name="dt")

    acc = results["41class"]["acc"]
    report = results["41class"]["report"]
    cm = results["41class"]["cm"]

    acc_cat = results["5class"]["acc"]

    print(f"[INFO] Baseline DT Accuracy (41-class): {acc:.4f}")
    print(f"[INFO] Baseline DT Accuracy (5-class): {acc_cat:.4f}")

    return dt, acc, report, cm

if __name__ == "__main__":
    # 🔹 CHANGED: Can now run with different datasets
    # train_decision_tree(dataset="nsl-kdd")  # Default behavior
    train_decision_tree(dataset="cicids2017")  # For CICIDS2017
