#!/usr/bin/env python3
"""
Correlation-based feature selection + Random Forest pipeline for NSL-KDD.

Steps:
1. Load train/test CSVs
2. Identify categorical and numeric features
3. Encode categorical features (LabelEncoder on combined train+test)
4. Train a baseline RandomForest on all features (get baseline accuracy & importances)
5. Compute correlation matrix on numeric features
6. For highly correlated numeric pairs (abs(corr) > threshold), drop the feature with lower RF importance
7. Retrain RandomForest on reduced feature set and report metrics & plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.utils.multiclass import unique_labels


warnings.filterwarnings('ignore')
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# -------------------------
# User-configurable params
# -------------------------
TRAIN_FILE = "kdd_train_clean.csv"
TEST_FILE = "kdd_test_clean_v2.csv"
TARGET_COL = "class"                # name of target column
CORR_THRESHOLD = 0.90               # absolute correlation threshold to consider "highly correlated"
BASELINE_N_ESTIMATORS = 300
FINAL_N_ESTIMATORS = 300
CV_FOLDS_FOR_BASELINE = 5
SAVE_SELECTED_FEATURES_CSV = "selected_features.csv"  # set None to skip saving
# -------------------------

def load_data(train_file, test_file):
    print("Loading files...")
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    # Drop unwanted index columns if they exist
    for df in [df_train, df_test]:
        df.drop(columns=[col for col in df.columns if "Unnamed" in col], inplace=True, errors="ignore")

    print(f"Train shape: {df_train.shape}, Test shape: {df_test.shape}")
    return df_train, df_test

def identify_feature_types(df, target_col):
    all_cols = df.columns.tolist()
    feature_cols = [c for c in all_cols if c != target_col]
    categorical = df[feature_cols].select_dtypes(include=['object']).columns.tolist()
    numeric = [c for c in feature_cols if c not in categorical]
    return numeric, categorical

def encode_categoricals(X_train, X_test, cat_cols):
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([X_train[col].astype(str), X_test[col].astype(str)], axis=0)
        le.fit(combined)
        X_train[col] = le.transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        encoders[col] = le
    return X_train, X_test, encoders

def baseline_random_forest(X_train, y_train, n_estimators=300):
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight='balanced',
        criterion='entropy',
        oob_score=True
    )
    print("Training baseline Random Forest (may take a moment)...")
    start = time.time()
    rf.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"Baseline RF trained in {elapsed:.1f}s, OOB score: {getattr(rf, 'oob_score_', None)}")
    # cross-validated accuracy (optional)
    try:
        cv_scores = cross_val_score(rf, X_train, y_train, cv=CV_FOLDS_FOR_BASELINE, scoring='accuracy', n_jobs=-1)
        print(f"Baseline {CV_FOLDS_FOR_BASELINE}-fold CV accuracy: {cv_scores.mean():.4f} (std {cv_scores.std():.4f})")
    except Exception:
        cv_scores = None
    importances = rf.feature_importances_
    feat_imp = pd.Series(importances, index=X_train.columns).sort_values(ascending=False)
    return rf, feat_imp

def drop_correlated_features_by_importance(df_numeric, feature_importances, threshold=0.90):
    print(f"Computing correlation matrix for {df_numeric.shape[1]} numeric features...")
    corr = df_numeric.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = set()
    for col in upper.columns:
        high_corr_cols = upper.index[upper[col] > threshold].tolist()
        for other in high_corr_cols:
            imp_col = feature_importances.get(col, 0.0)
            imp_other = feature_importances.get(other, 0.0)
            if imp_col >= imp_other:
                to_drop.add(other)
            else:
                to_drop.add(col)
    print(f"Marked {len(to_drop)} numeric features for dropping due to correlation threshold {threshold:.2f}")
    return list(to_drop), corr

def plot_results(history=None, corr_matrix=None, top_importances=None, cm=None, labels=None):
    plt.figure(figsize=(14, 10))

    if corr_matrix is not None:
        plt.subplot(2, 2, 1)
        dim = corr_matrix.shape[0]
        if dim > 30:
            var_rank = corr_matrix.var().sort_values(ascending=False).index[:30]
            sns.heatmap(corr_matrix.loc[var_rank, var_rank], cmap='coolwarm', vmin=-1, vmax=1)
            plt.title('Correlation (top 30 numeric features by variance)')
        else:
            sns.heatmap(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            plt.title('Correlation (numeric features)')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)

    if top_importances is not None:
        plt.subplot(2, 2, 2)
        top_importances.head(20).sort_values().plot(kind='barh')
        plt.title('Top 20 Feature Importances (Baseline RF)')
        plt.xlabel('Importance')

    if cm is not None:
        plt.subplot(2, 2, 3)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

    plt.tight_layout()
    plt.show()

def main():
    df_train, df_test = load_data(TRAIN_FILE, TEST_FILE)

    numeric_cols, categorical_cols = identify_feature_types(df_train, TARGET_COL)
    print(f"Numeric features: {len(numeric_cols)}; Categorical features: {len(categorical_cols)}")
    print("Categorical columns:", categorical_cols)

    X_train = df_train.drop(columns=[TARGET_COL]).copy()
    y_train = df_train[TARGET_COL].copy()
    X_test = df_test.drop(columns=[TARGET_COL]).copy()
    y_test = df_test[TARGET_COL].copy()

    target_encoder = None
    if y_train.dtype == 'object':
        target_encoder = LabelEncoder()
        target_encoder.fit(pd.concat([y_train, y_test], axis=0).astype(str))
        y_train = target_encoder.transform(y_train.astype(str))
        y_test = target_encoder.transform(y_test.astype(str))
        print(f"Encoded target classes: {len(target_encoder.classes_)}")

    if len(categorical_cols) > 0:
        X_train, X_test, encs = encode_categoricals(X_train, X_test, categorical_cols)
        print("Categorical columns encoded with LabelEncoder.")

    X_train_all = X_train.copy()
    X_test_all = X_test.copy()
    baseline_rf, feat_imp = baseline_random_forest(X_train_all, y_train, n_estimators=BASELINE_N_ESTIMATORS)
    print("\nTop 15 features by baseline RF importance:")
    print(feat_imp.head(15))

    print("\nEvaluating baseline on test set...")
    y_pred_base = baseline_rf.predict(X_test_all)
    base_acc = accuracy_score(y_test, y_pred_base)
    print(f"Baseline test accuracy (all features): {base_acc:.4f}")

    df_train_numeric = df_train[numeric_cols].copy()
    feat_imp_map = feat_imp.to_dict()
    to_drop_numeric, corr_matrix = drop_correlated_features_by_importance(df_train_numeric, feat_imp_map, threshold=CORR_THRESHOLD)

    final_features = [f for f in X_train_all.columns if f not in to_drop_numeric]
    print(f"Final selected features count: {len(final_features)} from {X_train_all.shape[1]} original features")
    print("Some selected features (first 30):", final_features[:30])

    if SAVE_SELECTED_FEATURES_CSV:
        pd.Series(final_features, name='selected_features').to_csv(SAVE_SELECTED_FEATURES_CSV, index=False)
        print(f"Saved selected features to {SAVE_SELECTED_FEATURES_CSV}")

    X_train_sel = X_train_all[final_features].copy()
    X_test_sel = X_test_all[final_features].copy()
    print("\nTraining final Random Forest on selected features...")
    final_rf = RandomForestClassifier(
        n_estimators=FINAL_N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight='balanced',
        criterion='entropy',
        oob_score=True
    )
    start = time.time()
    final_rf.fit(X_train_sel, y_train)
    elapsed = time.time() - start
    print(f"Final RF trained in {elapsed:.1f}s, OOB score: {getattr(final_rf, 'oob_score_', None)}")

    print("\nEvaluating final model on test set...")
    y_pred_final = final_rf.predict(X_test_sel)
    acc_final = accuracy_score(y_test, y_pred_final)
    print(f"Final test accuracy (selected features): {acc_final:.4f}")

    print("\nClassification report (final model):")
    labels_used = unique_labels(y_test, y_pred_final)
    print(classification_report(
        y_test,
        y_pred_final,
        labels=labels_used,
        target_names=[str(c) for c in labels_used]
    ))

    cm = confusion_matrix(y_test, y_pred_final)
    final_feat_imp = pd.Series(final_rf.feature_importances_, index=final_features).sort_values(ascending=False)

    reduction_pct = (1 - len(final_features) / X_train_all.shape[1]) * 100.0
    print("\n" + "="*40)
    print("SUMMARY")
    print("="*40)
    print(f"Original features: {X_train_all.shape[1]}")
    print(f"Selected features: {len(final_features)}")
    print(f"Feature reduction: {reduction_pct:.1f}%")
    print(f"Baseline test accuracy (all features): {base_acc:.4f}")
    print(f"Final test accuracy (selected features): {acc_final:.4f}")
    print("="*40)

    plot_results(corr_matrix=corr_matrix, top_importances=feat_imp, cm=cm, labels=target_encoder.classes_ if target_encoder is not None else None)

    print("\nTop 25 final features by importance:")
    print(final_feat_imp.head(25))

    print("\nPipeline complete.")

if __name__ == "__main__":
    main()

