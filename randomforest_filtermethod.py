import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif, VarianceThreshold
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

import warnings
warnings.filterwarnings('ignore')

# Load the datasets
print("Loading datasets...")
train_data = pd.read_csv('kdd_train_clean.csv')
test_data = pd.read_csv('kdd_test_clean.csv')

# Display basic information about the datasets
print(f"Training data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")
print(f"Training data columns: {train_data.columns.tolist()}")

# Assuming the last column contains the labels
target_column = train_data.columns[-1]
print(f"Target column: {target_column}")

# Separate features and target
X_train = train_data.drop(columns=[target_column])
y_train = train_data[target_column]
X_test = test_data.drop(columns=[target_column])
y_test = test_data[target_column]

print(f"Original number of features: {X_train.shape[1]}")
print(f"Class distribution in training set:")
print(Counter(y_train))

# Handle categorical variables
categorical_columns = X_train.select_dtypes(include=['object']).columns
numerical_columns = X_train.select_dtypes(exclude=['object']).columns
print(f"Categorical columns ({len(categorical_columns)}): {categorical_columns.tolist()}")
print(f"Numerical columns: {len(numerical_columns)}")

# Create label encoders for categorical features
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    combined_data = pd.concat([X_train[col], X_test[col]], axis=0)
    le.fit(combined_data.astype(str))
    X_train[col] = le.transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    label_encoders[col] = le

# Encode target labels if they are strings
if y_train.dtype == 'object':
    target_encoder = LabelEncoder()
    combined_labels = pd.concat([y_train, y_test], axis=0)
    target_encoder.fit(combined_labels)
    y_train = target_encoder.transform(y_train)
    y_test = target_encoder.transform(y_test)
    print(f"Target classes: {target_encoder.classes_}")
    print(f"Number of unique classes: {len(target_encoder.classes_)}")

# === FILTER METHODS FOR FEATURE SELECTION ===
print("\n" + "="*50)
print("FILTER METHODS FOR FEATURE SELECTION")
print("="*50)

# Store all filter method results
filter_results = {}
all_feature_scores = pd.DataFrame()
all_feature_scores['feature'] = X_train.columns

# FILTER METHOD 1: Variance Threshold (Remove low variance features)
print("\n1. VARIANCE THRESHOLD FILTER")
print("-" * 30)
variance_selector = VarianceThreshold(threshold=0.01)  # Remove features with very low variance
X_train_var = variance_selector.fit_transform(X_train)
X_test_var = variance_selector.transform(X_test)
high_variance_features = X_train.columns[variance_selector.get_support()]

print(f"Features before variance filtering: {X_train.shape[1]}")
print(f"Features after variance filtering: {len(high_variance_features)}")
print(f"Removed {X_train.shape[1] - len(high_variance_features)} low-variance features")

# Get variance scores
feature_variances = []
for col in X_train.columns:
    if col in high_variance_features:
        feature_variances.append(X_train[col].var())
    else:
        feature_variances.append(0)  # Low variance features get 0

all_feature_scores['variance'] = feature_variances

# FILTER METHOD 2: ANOVA F-test (f_classif)
print("\n2. ANOVA F-TEST FILTER")
print("-" * 25)
f_selector = SelectKBest(score_func=f_classif, k='all')
f_selector.fit(X_train, y_train)
f_scores = f_selector.scores_
f_pvalues = f_selector.pvalues_

all_feature_scores['f_score'] = f_scores
all_feature_scores['f_pvalue'] = f_pvalues

# Show top features by F-score
f_ranking = all_feature_scores.sort_values('f_score', ascending=False)
print("Top 15 features by ANOVA F-test:")
print(f_ranking[['feature', 'f_score', 'f_pvalue']].head(15))

# FILTER METHOD 3: Chi-Square Test (for categorical features)
print("\n3. CHI-SQUARE TEST FILTER")
print("-" * 26)
try:
    # Ensure all values are non-negative for chi2
    X_train_nonneg = X_train.copy()
    X_train_nonneg = X_train_nonneg - X_train_nonneg.min() + 1e-6
    
    chi2_selector = SelectKBest(score_func=chi2, k='all')
    chi2_selector.fit(X_train_nonneg, y_train)
    chi2_scores = chi2_selector.scores_
    chi2_pvalues = chi2_selector.pvalues_
    
    all_feature_scores['chi2_score'] = chi2_scores
    all_feature_scores['chi2_pvalue'] = chi2_pvalues
    
    chi2_ranking = all_feature_scores.sort_values('chi2_score', ascending=False)
    print("Top 15 features by Chi-Square test:")
    print(chi2_ranking[['feature', 'chi2_score', 'chi2_pvalue']].head(15))
    
except Exception as e:
    print(f"Chi-square test failed: {e}")
    all_feature_scores['chi2_score'] = 0
    all_feature_scores['chi2_pvalue'] = 1

# FILTER METHOD 4: Mutual Information
print("\n4. MUTUAL INFORMATION FILTER")
print("-" * 30)
try:
    mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
    all_feature_scores['mutual_info'] = mi_scores
    
    mi_ranking = all_feature_scores.sort_values('mutual_info', ascending=False)
    print("Top 15 features by Mutual Information:")
    print(mi_ranking[['feature', 'mutual_info']].head(15))
    
except Exception as e:
    print(f"Mutual information failed: {e}")
    all_feature_scores['mutual_info'] = 0

# FILTER METHOD 5: Correlation with Target
print("\n5. CORRELATION WITH TARGET FILTER")
print("-" * 35)
correlations = []
for col in X_train.columns:
    try:
        corr, _ = pearsonr(X_train[col], y_train)
        correlations.append(abs(corr))  # Use absolute correlation
    except:
        correlations.append(0)

all_feature_scores['correlation'] = correlations
corr_ranking = all_feature_scores.sort_values('correlation', ascending=False)
print("Top 15 features by Correlation with Target:")
print(corr_ranking[['feature', 'correlation']].head(15))

# === COMBINED FILTER RANKING ===
print("\n" + "="*50)
print("COMBINED FILTER RANKING")
print("="*50)

# Normalize scores to [0,1] for fair combination
def normalize_scores(scores):
    min_score = scores.min()
    max_score = scores.max()
    if max_score - min_score == 0:
        return scores * 0
    return (scores - min_score) / (max_score - min_score)

# Normalize all scores
all_feature_scores['f_score_norm'] = normalize_scores(all_feature_scores['f_score'])
all_feature_scores['chi2_score_norm'] = normalize_scores(all_feature_scores['chi2_score'])
all_feature_scores['mutual_info_norm'] = normalize_scores(all_feature_scores['mutual_info'])
all_feature_scores['correlation_norm'] = normalize_scores(all_feature_scores['correlation'])
all_feature_scores['variance_norm'] = normalize_scores(all_feature_scores['variance'])

# Combined score (weighted average of all methods)
weights = {
    'f_score_norm': 0.3,
    'mutual_info_norm': 0.25,
    'correlation_norm': 0.2,
    'chi2_score_norm': 0.15,
    'variance_norm': 0.1
}

all_feature_scores['combined_score'] = (
    all_feature_scores['f_score_norm'] * weights['f_score_norm'] +
    all_feature_scores['mutual_info_norm'] * weights['mutual_info_norm'] +
    all_feature_scores['correlation_norm'] * weights['correlation_norm'] +
    all_feature_scores['chi2_score_norm'] * weights['chi2_score_norm'] +
    all_feature_scores['variance_norm'] * weights['variance_norm']
)

# Final ranking
final_ranking = all_feature_scores.sort_values('combined_score', ascending=False)
print("\nTOP 20 FEATURES BY COMBINED FILTER METHODS:")
print(final_ranking[['feature', 'combined_score', 'f_score', 'mutual_info', 'correlation']].head(20))

# === SELECT OPTIMAL NUMBER OF FEATURES ===
print("\n" + "="*50)
print("SELECTING OPTIMAL NUMBER OF FEATURES")
print("="*50)

# Test different numbers of top features
feature_counts = [10, 15, 20, 25, 30, 35, len(X_train.columns)]
print("Testing different numbers of top-ranked features...")

for k in feature_counts:
    if k >= len(X_train.columns):
        print(f"   {len(X_train.columns)} features (all): Using all features")
    else:
        top_features = final_ranking.head(k)['feature'].tolist()
        print(f"   {k} features: {', '.join(top_features[:5])}{'...' if k > 5 else ''}")

# Let user choose or auto-select
print(f"\nChoose number of features to use:")
for i, k in enumerate(feature_counts):
    display_k = len(X_train.columns) if k >= len(X_train.columns) else k
    print(f"  {i+1}. {display_k} features")

try:
    choice = int(input("Enter your choice (1-7) or press Enter for auto-selection: ") or "4")  # Default to 25 features
    selected_k = feature_counts[choice - 1]
except:
    selected_k = 25  # Default
    print("Using default: 25 features")

if selected_k >= len(X_train.columns):
    print(f"\nUsing all {len(X_train.columns)} features")
    X_train_selected = X_train
    X_test_selected = X_test
    selected_features = X_train.columns
else:
    print(f"\nUsing top {selected_k} features by combined filter ranking")
    selected_features = final_ranking.head(selected_k)['feature']
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

print(f"Selected features: {list(selected_features)}")

# === TRAIN RANDOM FOREST WITH SELECTED FEATURES ===
print("\n" + "="*50)
print("TRAINING RANDOM FOREST WITH SELECTED FEATURES")
print("="*50)

print("Training Random Forest classifier...")
rf_classifier = RandomForestClassifier(
    n_estimators=100,
    max_depth=30,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    class_weight='balanced',
    criterion='entropy',
    random_state=42,
    n_jobs=-1,
    verbose=1,
    oob_score=True
)

# Train the model
rf_classifier.fit(X_train_selected, y_train)
print(f"Out-of-bag score: {rf_classifier.oob_score_:.4f}")

# Make predictions
print("Making predictions...")
y_pred = rf_classifier.predict(X_test_selected)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 FILTER-BASED ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Compare with typical baseline
print(f"Baseline (all features): ~91%")
if accuracy > 0.91:
    improvement = (accuracy - 0.91) * 100
    print(f"🎉 IMPROVEMENT: +{improvement:.2f} percentage points!")

# Detailed classification report
print(f"\nDetailed Classification Report:")
try:
    if 'target_encoder' in locals():
        unique_test = np.unique(y_test)
        unique_pred = np.unique(y_pred)
        all_unique = np.unique(np.concatenate([unique_test, unique_pred]))
        target_names_filtered = [target_encoder.classes_[i] for i in all_unique]
        print(classification_report(y_test, y_pred, labels=all_unique, target_names=target_names_filtered))
    else:
        print(classification_report(y_test, y_pred))
except Exception as e:
    print(f"Using basic classification report: {e}")
    print(classification_report(y_test, y_pred))

# Feature importance from Random Forest
print(f"\nRandom Forest Feature Importance (Top 15):")
rf_importance = pd.DataFrame({
    'feature': selected_features,
    'rf_importance': rf_classifier.feature_importances_
}).sort_values('rf_importance', ascending=False)
print(rf_importance.head(15))

# === VISUALIZATIONS ===
print("\nGenerating visualizations...")

# 1. Filter Methods Comparison
plt.figure(figsize=(15, 10))

# Plot 1: Top features by different methods
plt.subplot(2, 2, 1)
top_15 = final_ranking.head(15)
plt.barh(range(len(top_15)), top_15['combined_score'])
plt.yticks(range(len(top_15)), top_15['feature'])
plt.xlabel('Combined Filter Score')
plt.title('Top 15 Features by Combined Filter Methods')
plt.gca().invert_yaxis()

# Plot 2: Comparison of filter methods
plt.subplot(2, 2, 2)
methods = ['F-Score', 'Mutual Info', 'Correlation', 'Chi2', 'Variance']
method_cols = ['f_score_norm', 'mutual_info_norm', 'correlation_norm', 'chi2_score_norm', 'variance_norm']
top_5_features = final_ranking.head(5)['feature']

x = np.arange(len(top_5_features))
width = 0.15

for i, (method, col) in enumerate(zip(methods, method_cols)):
    scores = [final_ranking[final_ranking['feature'] == feat][col].iloc[0] for feat in top_5_features]
    plt.bar(x + i*width, scores, width, label=method)

plt.xlabel('Top 5 Features')
plt.ylabel('Normalized Score')
plt.title('Filter Methods Comparison (Top 5 Features)')
plt.xticks(x + width*2, [f[:8] + '...' if len(f) > 8 else f for f in top_5_features], rotation=45)
plt.legend()

# Plot 3: Random Forest Feature Importance
plt.subplot(2, 2, 3)
top_rf_features = rf_importance.head(10)
plt.barh(range(len(top_rf_features)), top_rf_features['rf_importance'])
plt.yticks(range(len(top_rf_features)), top_rf_features['feature'])
plt.xlabel('Random Forest Importance')
plt.title('Top 10 RF Feature Importance')
plt.gca().invert_yaxis()

# Plot 4: Filter vs RF Importance Comparison
plt.subplot(2, 2, 4)
common_features = set(final_ranking.head(15)['feature']).intersection(set(rf_importance.head(15)['feature']))
if len(common_features) >= 5:
    common_list = list(common_features)[:10]
    filter_scores = [final_ranking[final_ranking['feature'] == f]['combined_score'].iloc[0] for f in common_list]
    rf_scores = [rf_importance[rf_importance['feature'] == f]['rf_importance'].iloc[0] for f in common_list]
    
    plt.scatter(filter_scores, rf_scores)
    for i, feature in enumerate(common_list):
        plt.annotate(feature[:8], (filter_scores[i], rf_scores[i]), fontsize=8)
    plt.xlabel('Filter Score')
    plt.ylabel('RF Importance')
    plt.title('Filter vs RF Importance')

plt.tight_layout()
plt.show()

# Confusion Matrix
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test, y_pred)
if 'target_encoder' in locals() and len(target_encoder.classes_) <= 15:
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_encoder.classes_, 
                yticklabels=target_encoder.classes_)
    plt.xticks(rotation=45)
else:
    sns.heatmap(cm, annot=False, cmap='Blues')
    plt.title('Confusion Matrix (too many classes for labels)')

plt.title(f'Confusion Matrix - Filter-based Selection\nAccuracy: {accuracy:.4f}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# === SUMMARY ===
print("\n" + "="*60)
print("FILTER-BASED FEATURE SELECTION SUMMARY")
print("="*60)
print(f"✅ Original features: {len(X_train.columns)}")
print(f"✅ Selected features: {len(selected_features)}")
print(f"✅ Feature reduction: {((len(X_train.columns) - len(selected_features)) / len(X_train.columns) * 100):.1f}%")
print(f"✅ Filter methods used: 5 (Variance, ANOVA F-test, Chi-square, Mutual Info, Correlation)")
print(f"✅ Final accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"✅ Out-of-bag score: {rf_classifier.oob_score_:.4f}")

print(f"\nSelected features by combined filter ranking:")
for i, feature in enumerate(selected_features, 1):
    score = final_ranking[final_ranking['feature'] == feature]['combined_score'].iloc[0]
    print(f"  {i:2d}. {feature} (score: {score:.3f})")

print("\nAnalysis complete with filter-based feature selection!")