import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load the preprocessed dataset
print("Loading CICIDS2017 dataset...")
df = pd.read_csv('cicid2017\\Cicid2017_clean.csv')  # Replace with your file path

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)[:10]}...")  # Show first 10 columns

# Identify label column
label_col = [col for col in df.columns if 'label' in col.lower()][0]
print(f"\nLabel column: '{label_col}'")

# Check class distribution
print(f"\nClass distribution:")
print(df[label_col].value_counts())

# Separate features and labels
X = df.drop(columns=[label_col])
y = df[label_col]

print(f"\nNumber of features: {X.shape[1]}")
print(f"Number of samples: {X.shape[0]}")
print(f"Number of classes: {y.nunique()}")

# Check for and handle infinite/NaN values
print("\nChecking for data quality issues...")
print(f"NaN values: {X.isnull().sum().sum()}")
print(f"Infinite values: {np.isinf(X.select_dtypes(include=[np.number])).sum().sum()}")

# Replace infinite values with NaN, then fill with median
X = X.replace([np.inf, -np.inf], np.nan)
print(f"After replacing inf with NaN: {X.isnull().sum().sum()} NaN values")

# Fill NaN with median of each column
for col in X.columns:
    if X[col].isnull().any():
        X[col].fillna(X[col].median(), inplace=True)

print(f"After filling NaN: {X.isnull().sum().sum()} NaN values")
print(f"Final check - Infinite values: {np.isinf(X.select_dtypes(include=[np.number])).sum().sum()}")

# Encode labels if they're strings
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"\nEncoded classes: {le.classes_}")

# Train-test split (80-20)
print("\n" + "="*60)
print("TRAIN-TEST SPLIT")
print("="*60)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Feature scaling
print("\nScaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

# Train initial Random Forest for feature importance
print("\n" + "="*60)
print("STEP 1: INITIAL RANDOM FOREST (Feature Importance Analysis)")
print("="*60)

rf_initial = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("Training initial model...")
rf_initial.fit(X_train_scaled, y_train)

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_initial.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 25 Most Important Features:")
print(feature_importance.head(25).to_string(index=False))

# Initial model evaluation
y_pred_initial = rf_initial.predict(X_test_scaled)
accuracy_initial = accuracy_score(y_test, y_pred_initial)
f1_initial = f1_score(y_test, y_pred_initial, average='weighted')

print(f"\nInitial Model Performance (All {X.shape[1]} features):")
print(f"   Accuracy: {accuracy_initial:.4f} ({accuracy_initial*100:.2f}%)")
print(f"   F1-Score: {f1_initial:.4f}")

# Recursive Feature Elimination (RFE)
print("\n" + "="*60)
print("STEP 2: RECURSIVE FEATURE ELIMINATION (RFE)")
print("="*60)

# You can adjust this number based on your needs
n_features_to_select = min(30, len(X.columns))
print(f"Target: Select top {n_features_to_select} features")

rf_rfe = RandomForestClassifier(
    n_estimators=50,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

rfe = RFE(
    estimator=rf_rfe,
    n_features_to_select=n_features_to_select,
    step=5,
    verbose=2
)

print("\nRunning RFE (this may take a few minutes)...")
rfe.fit(X_train_scaled, y_train)

# Get selected features
selected_features = X.columns[rfe.support_].tolist()
print(f"\nRFE Complete! Selected {len(selected_features)} features")

# Show selected features with their importance
selected_importance = feature_importance[feature_importance['feature'].isin(selected_features)]
print("\nSelected Features (sorted by importance):")
print(selected_importance.to_string(index=False))

# Train final model with selected features
print("\n" + "="*60)
print("STEP 3: FINAL RANDOM FOREST MODEL (With RFE Features)")
print("="*60)

X_train_rfe = X_train_scaled[selected_features]
X_test_rfe = X_test_scaled[selected_features]

rf_final = RandomForestClassifier(
    n_estimators=200,
    max_depth=25,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print(f"Training final model with {len(selected_features)} features...")
rf_final.fit(X_train_rfe, y_train)

# Final model evaluation
print("\n" + "="*60)
print("STEP 4: MODEL EVALUATION")
print("="*60)

y_pred_final = rf_final.predict(X_test_rfe)
y_pred_proba = rf_final.predict_proba(X_test_rfe)

accuracy_final = accuracy_score(y_test, y_pred_final)
f1_final = f1_score(y_test, y_pred_final, average='weighted')

print(f"\nFinal Model Performance ({len(selected_features)} features):")
print(f"   Accuracy: {accuracy_final:.4f} ({accuracy_final*100:.2f}%)")
print(f"   F1-Score: {f1_final:.4f}")

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred_final, target_names=le.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_final)
print("\nConfusion Matrix:")
print(cm)

# Model Comparison
print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)
print(f"Initial Model: {X.shape[1]} features -> Accuracy: {accuracy_initial:.4f}, F1: {f1_initial:.4f}")
print(f"Final Model:   {len(selected_features)} features -> Accuracy: {accuracy_final:.4f}, F1: {f1_final:.4f}")
print(f"\nFeatures reduced by: {X.shape[1] - len(selected_features)} ({((X.shape[1] - len(selected_features))/X.shape[1]*100):.1f}%)")
print(f"Accuracy change: {(accuracy_final - accuracy_initial)*100:+.2f}%")
print(f"F1-Score change: {(f1_final - f1_initial)*100:+.2f}%")

# Visualizations
print("\n" + "="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)

# 1. Confusion Matrix Heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - Random Forest with RFE', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix_rf_rfe.png', dpi=300, bbox_inches='tight')
print("Saved: confusion_matrix_rf_rfe.png")
plt.close()

# 2. Feature Importance (Top 20)
plt.figure(figsize=(12, 8))
top_features = selected_importance.head(20)
plt.barh(range(len(top_features)), top_features['importance'].values)
plt.yticks(range(len(top_features)), top_features['feature'].values)
plt.xlabel('Feature Importance', fontsize=12)
plt.title('Top 20 Most Important Features (RFE Selected)', fontsize=16, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance_top20.png', dpi=300, bbox_inches='tight')
print("Saved: feature_importance_top20.png")
plt.close()

# 3. Model Comparison Chart
plt.figure(figsize=(10, 6))
models = ['Initial Model\n(All Features)', 'Final Model\n(RFE Features)']
accuracies = [accuracy_initial * 100, accuracy_final * 100]
f1_scores = [f1_initial * 100, f1_final * 100]

x = np.arange(len(models))
width = 0.35

plt.bar(x - width/2, accuracies, width, label='Accuracy', color='skyblue')
plt.bar(x + width/2, f1_scores, width, label='F1-Score', color='lightcoral')

plt.ylabel('Score (%)', fontsize=12)
plt.title('Model Performance Comparison', fontsize=16, fontweight='bold')
plt.xticks(x, models)
plt.legend()
plt.ylim(0, 105)

for i, v in enumerate(accuracies):
    plt.text(i - width/2, v + 1, f'{v:.2f}%', ha='center', fontweight='bold')
for i, v in enumerate(f1_scores):
    plt.text(i + width/2, v + 1, f'{v:.2f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: model_comparison.png")
plt.close()

# Save results to files
print("\n" + "="*60)
print("SAVING RESULTS TO FILES")
print("="*60)

# Save selected features
with open('selected_features.txt', 'w') as f:
    f.write("="*60 + "\n")
    f.write("SELECTED FEATURES BY RFE\n")
    f.write("="*60 + "\n\n")
    f.write(f"Total selected: {len(selected_features)} out of {X.shape[1]} features\n\n")
    for i, feat in enumerate(selected_importance['feature'].values, 1):
        imp = selected_importance[selected_importance['feature']==feat]['importance'].values[0]
        f.write(f"{i:2d}. {feat:50s} (importance: {imp:.6f})\n")
print("Saved: selected_features.txt")

# Save model performance report
with open('model_performance_report.txt', 'w') as f:
    f.write("="*60 + "\n")
    f.write("RANDOM FOREST WITH RFE - MODEL PERFORMANCE REPORT\n")
    f.write("="*60 + "\n\n")
    f.write(f"Dataset: CICIDS2017\n")
    f.write(f"Total Samples: {X.shape[0]}\n")
    f.write(f"Training Samples: {X_train.shape[0]}\n")
    f.write(f"Test Samples: {X_test.shape[0]}\n")
    f.write(f"Number of Classes: {y.nunique()}\n\n")
    
    f.write("INITIAL MODEL (All Features)\n")
    f.write("-" * 40 + "\n")
    f.write(f"Features: {X.shape[1]}\n")
    f.write(f"Accuracy: {accuracy_initial:.4f} ({accuracy_initial*100:.2f}%)\n")
    f.write(f"F1-Score: {f1_initial:.4f}\n\n")
    
    f.write("FINAL MODEL (RFE Selected Features)\n")
    f.write("-" * 40 + "\n")
    f.write(f"Features: {len(selected_features)}\n")
    f.write(f"Accuracy: {accuracy_final:.4f} ({accuracy_final*100:.2f}%)\n")
    f.write(f"F1-Score: {f1_final:.4f}\n\n")
    
    f.write("IMPROVEMENT\n")
    f.write("-" * 40 + "\n")
    f.write(f"Features Reduced: {X.shape[1] - len(selected_features)} ({((X.shape[1] - len(selected_features))/X.shape[1]*100):.1f}%)\n")
    f.write(f"Accuracy Change: {(accuracy_final - accuracy_initial)*100:+.2f}%\n")
    f.write(f"F1-Score Change: {(f1_final - f1_initial)*100:+.2f}%\n\n")
    
    f.write("DETAILED CLASSIFICATION REPORT\n")
    f.write("="*60 + "\n")
    f.write(classification_report(y_test, y_pred_final, target_names=le.classes_))
print("Saved: model_performance_report.txt")

print("\n" + "="*60)
print("PIPELINE COMPLETED SUCCESSFULLY!")
print("="*60)
print("\nGenerated Files:")
print("   1. confusion_matrix_rf_rfe.png")
print("   2. feature_importance_top20.png")
print("   3. model_comparison.png")
print("   4. selected_features.txt")
print("   5. model_performance_report.txt")