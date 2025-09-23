import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

def load_data(train_path, test_path, target_col=None, apply_smote=False):
    # Load CSVs
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # Auto-detect target column if not provided
    if target_col is None:
        possible_targets = ["label", "class", "attack", "target"]
        for col in train.columns:
            if col.lower() in possible_targets:
                target_col = col
                break
        if target_col is None:
            raise ValueError(f"❌ No target column found. Available columns: {train.columns.tolist()}")

    # Separate features & labels
    X_train = train.drop(columns=[target_col])
    y_train = train[target_col]
    X_test = test.drop(columns=[target_col])
    y_test = test[target_col]

    # Encode categorical cols
    cat_cols = ['protocol_type', 'service', 'flag']
    encoders = {}
    for col in cat_cols:
        if col in X_train.columns:
            le = LabelEncoder()
            combined = pd.concat([X_train[col], X_test[col]], axis=0)
            le.fit(combined)
            X_train[col] = le.transform(X_train[col])
            X_test[col] = le.transform(X_test[col])
            encoders[col] = le

    # Apply SMOTE oversampling (train only)
    if apply_smote:
        smote = SMOTE(random_state=42, sampling_strategy='auto', k_neighbors=1)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test, encoders
