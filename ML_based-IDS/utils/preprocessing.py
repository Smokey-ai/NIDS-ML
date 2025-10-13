import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def load_data(train_path, test_path=None, target_col=None, apply_smote=False, 
              categorical_cols=None, dataset_type="auto"):
    """
    Load dataset with support for both NSL-KDD and CICIDS2017
    
    Parameters:
    - train_path: Path to training data or full dataset
    - test_path: Path to test data (None for CICIDS2017)
    - target_col: Target column name (auto-detect if None)
    - apply_smote: Whether to apply SMOTE oversampling
    - categorical_cols: List of categorical columns to encode (None for auto-detection)
    - dataset_type: "nsl-kdd", "cicids2017", or "auto-detect"
    
    Returns:
    - X_train, X_test, y_train, y_test, encoders
    """
    
    # 🔹 IMPROVED: Better dataset type detection
    if dataset_type == "auto":
        if test_path is None:
            dataset_type = "cicids2017"
            print(f"[INFO] Auto-detected dataset: CICIDS2017 (single file)")
        else:
            dataset_type = "nsl-kdd"
            print(f"[INFO] Auto-detected dataset: NSL-KDD (separate train/test)")
    
    # 🔹 IMPROVED: Default categorical columns per dataset
    if categorical_cols is None:
        if dataset_type == "nsl-kdd":
            categorical_cols = ['protocol_type', 'service', 'flag']
            print(f"[INFO] Using NSL-KDD categorical columns: {categorical_cols}")
        else:  # cicids2017
            categorical_cols = []  # No categorical columns in CICIDS2017
            print(f"[INFO] CICIDS2017: No categorical columns (all numerical)")
    
    # 🔹 Handle CICIDS2017 (single file - need to split)
    if dataset_type == "cicids2017" and test_path is None:
        print(f"[INFO] Loading CICIDS2017 from: {train_path}")
        full_data = pd.read_csv(train_path)
        
        # Auto-detect target column for CICIDS2017
        if target_col is None:
            if "Label" in full_data.columns:
                target_col = "Label"
            else:
                # Try common target column names
                possible_targets = ["label", "class", "attack", "target", "outcome"]
                for col in full_data.columns:
                    if col.lower() in possible_targets:
                        target_col = col
                        break
                if target_col is None:
                    raise ValueError(f"❌ No target column found. Available columns: {full_data.columns.tolist()}")
        
        print(f"[INFO] Using target column: '{target_col}'")
        print(f"[INFO] Dataset shape: {full_data.shape}, Target distribution:\n{full_data[target_col].value_counts()}")
        
        # Split into train/test (80/20)
        train, test = train_test_split(full_data, test_size=0.2, random_state=42, stratify=full_data[target_col])
        
        X_train = train.drop(columns=[target_col])
        y_train = train[target_col]
        X_test = test.drop(columns=[target_col])
        y_test = test[target_col]
        
        print(f"[INFO] Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        
    else:
        # 🔹 NSL-KDD behavior (separate train/test files)
        print(f"[INFO] Loading NSL-KDD - Train: {train_path}, Test: {test_path}")
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

        print(f"[INFO] Using target column: '{target_col}'")
        print(f"[INFO] Train shape: {train.shape}, Test shape: {test.shape}")

        # Separate features & labels
        X_train = train.drop(columns=[target_col])
        y_train = train[target_col]
        X_test = test.drop(columns=[target_col])
        y_test = test[target_col]

    # 🔹 IMPROVED: Apply encoding ONLY to specified categorical columns that exist
    encoders = {}
    actual_categorical_cols = []
    
    for col in categorical_cols:
        if col in X_train.columns:
            le = LabelEncoder()
            # Combine train + test for consistent encoding
            combined = pd.concat([X_train[col], X_test[col]], axis=0)
            le.fit(combined)
            X_train[col] = le.transform(X_train[col])
            X_test[col] = le.transform(X_test[col])
            encoders[col] = le
            actual_categorical_cols.append(col)
            print(f"[INFO] Encoded categorical column: '{col}'")
        else:
            print(f"⚠️  Warning: Categorical column '{col}' not found in dataset")
    
    if actual_categorical_cols:
        print(f"[INFO] Successfully encoded {len(actual_categorical_cols)} categorical columns: {actual_categorical_cols}")
    else:
        print(f"[INFO] No categorical columns to encode")

    # 🔹 EXISTING: Apply SMOTE oversampling (train only)
    if apply_smote:
        print(f"[INFO] Applying SMOTE oversampling...")
        smote = SMOTE(random_state=42, sampling_strategy='auto', k_neighbors=1)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"[INFO] After SMOTE - Train shape: {X_train.shape}")

    print(f"[INFO] Final shapes - X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"[INFO] Label encoder(s) created: {list(encoders.keys())}")
    
    return X_train, X_test, y_train, y_test, encoders
