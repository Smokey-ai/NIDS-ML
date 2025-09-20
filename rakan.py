import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# === Step 1: Load training and test datasets ===
train_df = pd.read_csv("kdd_train_clean.csv")
test_df = pd.read_csv("kdd_test_clean.csv")

# === Step 2: Prepare features and target ===
X_train = train_df.drop("class", axis=1)
y_train = LabelEncoder().fit_transform(train_df["class"])

X_test = test_df.drop("class", axis=1)
y_test = LabelEncoder().fit_transform(test_df["class"])

# === Step 3: Apply RFE to select top 20 features ===
rfe_model = RandomForestClassifier(
    n_estimators=100,               # Use fewer trees for faster RFE
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rfe = RFE(estimator=rfe_model, n_features_to_select=10, step=2)
rfe.fit(X_train, y_train)

# === Step 4: Extract top 20 features ===
selected_features = X_train.columns[rfe.support_]
print("Top 20 selected features:", selected_features.tolist())

# === Step 5: Filter training and test data ===
X_train_rfe = X_train[selected_features]
X_test_rfe = X_test[selected_features]

# === Step 6: Retrain high-power RandomForest on selected features ===
final_rf = RandomForestClassifier(
    n_estimators=500,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
final_rf.fit(X_train_rfe, y_train)

# === Step 7: Evaluate model ===
y_pred = final_rf.predict(X_test_rfe)

print("\n Accuracy:", accuracy_score(y_test, y_pred))
print("\n Classification Report:\n", classification_report(y_test, y_pred))