import os
from sklearn.tree import DecisionTreeClassifier
from utils.preprocessing import load_data
from utils.evaluation import evaluate_model

TEST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/nsl-kdd/clean-test.csv"))
TRAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/nsl-kdd/clean-train.csv"))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results/"))

def train_decision_tree(
    train_path=TEST_DIR,
    test_path=TRAIN_DIR,
    results_path=RESULTS_DIR
):
    # 1. Load data
    X_train, X_test, y_train, y_test, encoders = load_data(train_path, test_path)

    # 2. Initialize baseline DT
    dt = DecisionTreeClassifier(random_state=42)

    # 3. Train
    dt.fit(X_train, y_train)

    # 4. Evaluate
    acc, report, cm = evaluate_model(dt, X_test, y_test, model_name="dt", results_path=results_path)

    print(f"[INFO] Baseline DT Accuracy: {acc:.4f}")
    return dt, acc, report, cm

if __name__ == "__main__":
    train_decision_tree()
