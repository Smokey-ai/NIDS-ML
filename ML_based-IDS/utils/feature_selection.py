import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif

def drop_correlated_features(X, threshold=0.9):
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    X_reduced = X.drop(columns=to_drop)
    return X_reduced, to_drop

def select_features_mi(X, y, top_k=None, threshold=None):
    """
    Select features based on Mutual Information with target variable.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target labels.
        top_k (int): Keep top-k features (ranked by MI). If None, use threshold.
        threshold (float): Keep features with MI >= threshold. If None, use top_k.

    Returns:
        X_selected (pd.DataFrame): Reduced feature set.
        feature_scores (pd.DataFrame): Feature names + MI scores.
    """
    mi_scores = mutual_info_classif(X, y, discrete_features='auto', random_state=42)
    feature_scores = pd.DataFrame({
        "feature": X.columns,
        "mi_score": mi_scores
    }).sort_values(by="mi_score", ascending=False)

    if top_k:
        selected_features = feature_scores.head(top_k)["feature"].tolist()
    elif threshold:
        selected_features = feature_scores[feature_scores["mi_score"] >= threshold]["feature"].tolist()
    else:
        selected_features = feature_scores[feature_scores["mi_score"] > 0]["feature"].tolist()

    X_selected = X[selected_features]
    return X_selected, feature_scores
