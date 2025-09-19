import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

print("=== Optimized Random Forest for NSL-KDD Dataset ===")

# Load the datasets
print("Loading datasets...")
train_data = pd.read_csv('kdd_train_clean.csv')
test_data = pd.read_csv('kdd_test_clean.csv')

print(f"Training data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")

# Assuming the last column contains the labels
target_column = train_data.columns[-1]
print(f"Target column: {target_column}")

# Separate features and target
X_train = train_data.drop(columns=[target_column])
y_train = train_data[target_column]
X_test = test_data.drop(columns=[target_column])
y_test = test_data[target_column]

print(f"Original class distribution in training set:")
class_counts = Counter(y_train)
for class_name, count in class_counts.most_common():
    print(f"  {class_name}: {count}")

# Check for class imbalance
total_samples = len(y_train)
imbalance_ratio = min(class_counts.values()) / max(class_counts.values())
print(f"Class imbalance ratio: {imbalance_ratio:.4f}")

# Handle categorical variables
categorical_columns = X_train.select_dtypes(include=['object']).columns
numerical_columns = X_train.select_dtypes(exclude=['object']).columns
print(f"Categorical columns ({len(categorical_columns)}): {categorical_columns.tolist()}")
print(f"Numerical columns: {len(numerical_columns)}")

# Encode categorical features
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    combined_data = pd.concat([X_train[col], X_test[col]], axis=0)
    le.fit(combined_data.astype(str))
    X_train[col] = le.transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    label_encoders[col] = le

# Encode target labels
if y_train.dtype == 'object':
    target_encoder = LabelEncoder()
    combined_labels = pd.concat([y_train, y_test], axis=0)
    target_encoder.fit(combined_labels)
    y_train = target_encoder.transform(y_train)
    y_test = target_encoder.transform(y_test)
    print(f"Target classes ({len(target_encoder.classes_)}): {target_encoder.classes_}")

# === OPTIMIZATION 1: DATA CLEANING ===
print("\n=== OPTIMIZATION 1: Data Cleaning ===")

# Remove duplicate rows
initial_size = len(X_train)
combined_train = pd.concat([X_train, pd.Series(y_train, name='target')], axis=1)
combined_train = combined_train.drop_duplicates()
X_train = combined_train.drop('target', axis=1)
y_train = combined_train['target'].values
duplicates_removed = initial_size - len(X_train)
print(f"Removed {duplicates_removed} duplicate rows ({duplicates_removed/initial_size*100:.1f}%)")

# === OPTIMIZATION 2: FEATURE SCALING ===
print("\n=== OPTIMIZATION 2: Feature Scaling ===")

# Use RobustScaler (better for outliers than StandardScaler)
scaler = RobustScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

if len(numerical_columns) > 0:
    X_train_scaled[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
    X_test_scaled[numerical_columns] = scaler.transform(X_test[numerical_columns])
    print(f"Scaled {len(numerical_columns)} numerical features")

# === OPTIMIZATION 3: FEATURE SELECTION ===
print("\n=== OPTIMIZATION 3: Feature Selection ===")

# Statistical feature selection
selector = SelectKBest(score_func=f_classif, k='all')
selector.fit(X_train_scaled, y_train)

# Get feature importance scores
feature_scores = pd.DataFrame({
    'feature': X_train.columns,
    'score': selector.scores_
}).sort_values('score', ascending=False)

print("Top 20 features by statistical importance:")
print(feature_scores.head(20))

# Select optimal number of features (try different values)
feature_counts = [10, 15, 20, 25, 30, 'all']
best_features = None
best_score = 0
best_k = None

print("\nTesting different numbers of features...")
for k in feature_counts:
    if k == 'all':
        k_val = len(X_train.columns)
    else:
        k_val = min(k, len(X_train.columns))
    
    selector_k = SelectKBest(score_func=f_classif, k=k_val)
    X_train_selected = selector_k.fit_transform(X_train_scaled, y_train)
    
    # Quick RF test
    rf_test = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    scores = cross_val_score(rf_test, X_train_selected, y_train, cv=3, scoring='accuracy')
    avg_score = scores.mean()
    
    print(f"  {k} features: {avg_score:.4f} (+/- {scores.std() * 2:.4f})")
    
    if avg_score > best_score:
        best_score = avg_score
        best_k = k_val
        best_features = X_train.columns[selector_k.get_support()]

print(f"Best number of features: {best_k} (CV accuracy: {best_score:.4f})")

# Apply best feature selection
selector_best = SelectKBest(score_func=f_classif, k=best_k)
X_train_final = selector_best.fit_transform(X_train_scaled, y_train)
X_test_final = selector_best.transform(X_test_scaled)

print(f"Selected features: {list(best_features)}")

# === OPTIMIZATION 4: HANDLE CLASS IMBALANCE ===
print("\n=== OPTIMIZATION 4: Class Imbalance Handling ===")

# Apply SMOTE if there's significant imbalance
apply_smote = imbalance_ratio < 0.3  # Apply if minority class < 30% of majority

if apply_smote:
    print("Applying SMOTE to balance classes...")
    try:
        # Use conservative SMOTE parameters to avoid overfitting
        smote = SMOTE(random_state=42, k_neighbors=3, sampling_strategy='auto')
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_final, y_train)
        
        print(f"Original shape: {X_train_final.shape}")
        print(f"Balanced shape: {X_train_balanced.shape}")
        print("New class distribution:")
        for class_name, count in Counter(y_train_balanced).most_common():
            if 'target_encoder' in locals():
                original_name = target_encoder.classes_[class_name]
                print(f"  {original_name}: {count}")
            else:
                print(f"  {class_name}: {count}")
        
        X_train_final = X_train_balanced
        y_train_final = y_train_balanced
    except Exception as e:
        print(f"SMOTE failed: {e}. Using original data.")
        y_train_final = y_train
else:
    print("Class distribution acceptable, skipping SMOTE")
    y_train_final = y_train

# === OPTIMIZATION 5: COMPREHENSIVE HYPERPARAMETER TUNING ===
print("\n=== OPTIMIZATION 5: Hyperparameter Tuning ===")

# Extensive parameter grid for Random Forest
param_distributions = {
    'n_estimators': [100, 200, 300, 500, 800],
    'max_depth': [None, 10, 20, 30, 50, 70],
    'min_samples_split': [2, 5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 4, 6, 8],
    'max_features': ['sqrt', 'log2', None, 0.3, 0.5, 0.7, 0.9],
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy'],
    'class_weight': [None, 'balanced', 'balanced_subsample'],
    'max_samples': [None, 0.7, 0.8, 0.9]  # For bootstrap sampling
}

print("Performing comprehensive hyperparameter search...")
print("This may take several minutes...")

# Use RandomizedSearchCV for efficiency
rf_base = RandomForestClassifier(random_state=42, n_jobs=-1, warm_start=False)

random_search = RandomizedSearchCV(
    estimator=rf_base,
    param_distributions=param_distributions,
    n_iter=100,  # Try 100 different combinations
    cv=5,  # 5-fold cross-validation
    scoring='f1_weighted',  # Use weighted F1 for imbalanced data
    n_jobs=-1,
    verbose=1,
    random_state=42,
    return_train_score=True
)

# Fit the random search
random_search.fit(X_train_final, y_train_final)

print(f"\nBest parameters found:")
for param, value in random_search.best_params_.items():
    print(f"  {param}: {value}")
print(f"Best cross-validation F1-weighted score: {random_search.best_score_:.4f}")

# === OPTIMIZATION 6: TRAIN FINAL OPTIMIZED MODEL ===
print("\n=== OPTIMIZATION 6: Final Model Training ===")

# Get the best model
best_rf = random_search.best_estimator_

# Train on full dataset
print("Training final optimized Random Forest...")
best_rf.fit(X_train_final, y_train_final)

# Make predictions
print("Making predictions...")
y_pred = best_rf.predict(X_test_final)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_weighted = f1_score(y_test, y_pred, average='weighted')

print(f"\n=== FINAL RESULTS ===")
print(f"Optimized Random Forest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"F1-Score (Macro): {f1_macro:.4f}")
print(f"F1-Score (Weighted): {f1_weighted:.4f}")

if accuracy > 0.91:
    improvement = (accuracy - 0.91) * 100
    print(f"🎉 IMPROVEMENT ACHIEVED: +{improvement:.2f} percentage points!")
else:
    print(f"Current result: {accuracy*100:.2f}% (vs your original 91%)")

# Detailed classification report
print(f"\nDetailed Classification Report:")
if 'target_encoder' in locals():
    print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))
else:
    print(classification_report(y_test, y_pred))

# Feature importance analysis
print(f"\nTop 15 Most Important Features:")
if hasattr(best_rf, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': best_features,
        'importance': best_rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(15))
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(15)
    sns.barplot(data=top_features, x='importance', y='feature')
    plt.title('Top 15 Feature Importances - Optimized Random Forest')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()

# Confusion Matrix
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test, y_pred)
if 'target_encoder' in locals():
    class_names = target_encoder.classes_
    # If too many classes, show without labels
    if len(class_names) > 10:
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix - Optimized Random Forest')
    else:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix - Optimized Random Forest')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
else:
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Optimized Random Forest')

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# Model complexity analysis
print(f"\nModel Complexity:")
print(f"Number of trees: {best_rf.n_estimators}")
print(f"Max depth: {best_rf.max_depth}")
print(f"Min samples split: {best_rf.min_samples_split}")
print(f"Min samples leaf: {best_rf.min_samples_leaf}")
print(f"Max features: {best_rf.max_features}")

# === ADDITIONAL ANALYSIS ===
print(f"\n=== OPTIMIZATION SUMMARY ===")
print(f"1. Data cleaning: Removed {duplicates_removed} duplicates")
print(f"2. Feature scaling: Applied RobustScaler to {len(numerical_columns)} numerical features")
print(f"3. Feature selection: Selected {best_k} out of {len(X_train.columns)} features")
print(f"4. Class balancing: {'Applied SMOTE' if apply_smote else 'No balancing needed'}")
print(f"5. Hyperparameter tuning: Tested 100 parameter combinations")
print(f"6. Final accuracy: {accuracy*100:.2f}%")

print(f"\n=== TIPS FOR FURTHER IMPROVEMENT ===")
print("If you want to squeeze out more performance:")
print("1. Increase n_iter in RandomizedSearchCV (try 200-500)")
print("2. Try different SMOTE strategies if class imbalance exists")
print("3. Experiment with feature engineering (create new features)")
print("4. Use more sophisticated feature selection methods")
print("5. Try ensemble of multiple Random Forest models with different random seeds")