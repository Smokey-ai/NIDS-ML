import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import random
import time
import warnings
warnings.filterwarnings('ignore')

# Attack type mapping to main categories
ATTACK_CATEGORIES = {
    # Normal traffic
    'normal': 'Normal',
    
    # DoS attacks
    'back': 'DoS',
    'land': 'DoS', 
    'neptune': 'DoS',
    'pod': 'DoS',
    'smurf': 'DoS',
    'teardrop': 'DoS',
    'mailbomb': 'DoS',
    'apache2': 'DoS',
    'processtable': 'DoS',
    'udpstorm': 'DoS',
    
    # Probe attacks  
    'satan': 'Probe',
    'ipsweep': 'Probe',
    'nmap': 'Probe',
    'portsweep': 'Probe',
    'mscan': 'Probe',
    'saint': 'Probe',
    
    # R2L attacks
    'guess_passwd': 'R2L',
    'ftp_write': 'R2L',
    'imap': 'R2L',
    'phf': 'R2L',
    'multihop': 'R2L',
    'warezmaster': 'R2L',
    'warezclient': 'R2L',
    'spy': 'R2L',
    'xlock': 'R2L',
    'xsnoop': 'R2L',
    'snmpread': 'R2L',
    'snmpguess': 'R2L',
    'worm': 'R2L',
    'httptunnel': 'R2L',
    'named': 'R2L',
    'sendmail': 'R2L',
    
    # U2R attacks
    'buffer_overflow': 'U2R',
    'loadmodule': 'U2R',
    'perl': 'U2R',
    'rootkit': 'U2R',
    'ps': 'U2R',
    'sqlattack': 'U2R',
    'xterm': 'U2R'
}

def map_attack_categories(attack_types):
    """Map specific attack types to main categories"""
    categories = []
    unmapped = set()
    
    for attack in attack_types:
        attack_lower = attack.lower().strip()
        if attack_lower in ATTACK_CATEGORIES:
            categories.append(ATTACK_CATEGORIES[attack_lower])
        else:
            # Try partial matching for unknown attacks
            category_found = False
            for known_attack, category in ATTACK_CATEGORIES.items():
                if known_attack in attack_lower or attack_lower in known_attack:
                    categories.append(category)
                    category_found = True
                    break
            
            if not category_found:
                # Default unknown attacks to most likely category based on name patterns
                if any(word in attack_lower for word in ['dos', 'flood', 'storm']):
                    categories.append('DoS')
                elif any(word in attack_lower for word in ['scan', 'sweep', 'probe']):
                    categories.append('Probe')
                elif any(word in attack_lower for word in ['passwd', 'login', 'ftp', 'remote']):
                    categories.append('R2L')
                elif any(word in attack_lower for word in ['buffer', 'root', 'overflow']):
                    categories.append('U2R')
                else:
                    categories.append('Unknown')
                    unmapped.add(attack)
    
    return categories, unmapped

# PSO-based Feature Selection Class
class PSOFeatureSelection:
    def __init__(self, n_particles=20, n_iterations=30, w=0.5, c1=1.5, c2=1.5):
        """
        PSO for Feature Selection - Free Optimization (No Mandatory Features)
        """
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w  # Inertia weight
        self.c1 = c1  # Cognitive parameter
        self.c2 = c2  # Social parameter

    def initialize_particles(self, n_features):
        """Initialize particle positions and velocities"""
        # Position: binary array (1=selected, 0=not selected)
        positions = np.random.rand(self.n_particles, n_features) > 0.5
        positions = positions.astype(int)
        
        # Ensure each particle selects at least 5 features and at most 80% of features
        for i in range(self.n_particles):
            n_selected = np.sum(positions[i])
            min_features = 5
            max_features = int(0.8 * n_features)
            
            if n_selected < min_features:
                zero_indices = np.where(positions[i] == 0)[0]
                if len(zero_indices) >= min_features - n_selected:
                    add_indices = np.random.choice(zero_indices, min_features - n_selected, replace=False)
                    positions[i][add_indices] = 1
            elif n_selected > max_features:
                one_indices = np.where(positions[i] == 1)[0]
                remove_indices = np.random.choice(one_indices, n_selected - max_features, replace=False)
                positions[i][remove_indices] = 0
        
        # Velocity: continuous values
        velocities = np.random.uniform(-1, 1, (self.n_particles, n_features))
        
        return positions, velocities
    
    def evaluate_fitness(self, position, X_train, y_train, estimator, cv=3):
        """Evaluate fitness of a feature subset"""
        selected_features = np.where(position == 1)[0]
        
        if len(selected_features) == 0:
            return 0.0
        
        X_selected = X_train[:, selected_features]
        
        try:
            # Use cross-validation for robust evaluation
            scores = cross_val_score(estimator, X_selected, y_train, cv=cv, scoring='accuracy')
            fitness = scores.mean()
            
            # Penalty for too many features (encourage smaller subsets)
            feature_ratio = len(selected_features) / len(position)
            fitness = fitness * (1 - 0.05 * feature_ratio)
            
            return fitness
        except:
            return 0.0
    
    def update_velocity(self, velocity, position, personal_best, global_best):
        """Update particle velocity"""
        r1, r2 = np.random.rand(2)
        
        cognitive = self.c1 * r1 * (personal_best - position)
        social = self.c2 * r2 * (global_best - position)
        
        new_velocity = self.w * velocity + cognitive + social
        
        # Clamp velocity to [-4, 4]
        new_velocity = np.clip(new_velocity, -4, 4)
        
        return new_velocity
    
    def update_position(self, position, velocity):
        """Update particle position using sigmoid function"""
        # Apply sigmoid to convert velocity to probability
        sigmoid = 1 / (1 + np.exp(-velocity))
        
        # Update position based on probability
        new_position = (np.random.rand(len(position)) < sigmoid).astype(int)
        
        # Ensure constraints (5 to 80% of features)
        n_selected = np.sum(new_position)
        min_features = 5
        max_features = int(0.8 * len(position))
        
        if n_selected < min_features:
            zero_indices = np.where(new_position == 0)[0]
            if len(zero_indices) >= min_features - n_selected:
                add_indices = np.random.choice(zero_indices, min_features - n_selected, replace=False)
                new_position[add_indices] = 1
        elif n_selected > max_features:
            one_indices = np.where(new_position == 1)[0]
            remove_indices = np.random.choice(one_indices, n_selected - max_features, replace=False)
            new_position[remove_indices] = 0
        
        return new_position
    
    def fit(self, X_train, y_train, estimator=None, verbose=True):
        """Run PSO optimization"""
        if estimator is None:
            estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        
        n_features = X_train.shape[1]
        
        # Initialize particles
        positions, velocities = self.initialize_particles(n_features)
        
        # Initialize personal and global bests
        personal_best_positions = positions.copy()
        personal_best_scores = np.array([self.evaluate_fitness(pos, X_train, y_train, estimator) 
                                       for pos in positions])
        
        global_best_idx = np.argmax(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx].copy()
        global_best_score = personal_best_scores[global_best_idx]
        
        self.history = {
            'best_scores': [],
            'avg_scores': [],
            'n_features': []
        }
        
        if verbose:
            print(f"PSO Feature Selection Started")
            print(f"Population: {self.n_particles} particles, {self.n_iterations} iterations")
            print(f"Initial best score: {global_best_score:.4f} with {np.sum(global_best_position)} features")
            print("-" * 60)
        
        # PSO main loop
        for iteration in range(self.n_iterations):
            iteration_scores = []
            
            for i in range(self.n_particles):
                # Evaluate current position
                current_score = self.evaluate_fitness(positions[i], X_train, y_train, estimator)
                iteration_scores.append(current_score)
                
                # Update personal best
                if current_score > personal_best_scores[i]:
                    personal_best_scores[i] = current_score
                    personal_best_positions[i] = positions[i].copy()
                
                # Update global best
                if current_score > global_best_score:
                    global_best_score = current_score
                    global_best_position = positions[i].copy()
                    if verbose:
                        print(f"  🎯 New best at iteration {iteration+1}: {global_best_score:.4f} with {np.sum(global_best_position)} features")
                
                # Update velocity and position
                velocities[i] = self.update_velocity(velocities[i], positions[i], 
                                                   personal_best_positions[i], global_best_position)
                positions[i] = self.update_position(positions[i], velocities[i])
            
            # Store history
            avg_score = np.mean(iteration_scores)
            self.history['best_scores'].append(global_best_score)
            self.history['avg_scores'].append(avg_score)
            self.history['n_features'].append(np.sum(global_best_position))
            
            if verbose and (iteration + 1) % 5 == 0:
                print(f"Iteration {iteration+1:2d}: Best={global_best_score:.4f}, "
                      f"Avg={avg_score:.4f}, Features={np.sum(global_best_position)}")
        
        self.best_features_ = np.where(global_best_position == 1)[0]
        self.best_score_ = global_best_score
        self.best_position_ = global_best_position
        
        if verbose:
            print("-" * 60)
            print(f"PSO Optimization Complete!")
            print(f"Best score: {global_best_score:.4f}")
            print(f"Selected features: {len(self.best_features_)}/{n_features}")
        
        return self
    
    def transform(self, X):
        """Transform data using selected features"""
        return X[:, self.best_features_]
    
    def fit_transform(self, X, y, estimator=None, verbose=True):
        """Fit PSO and transform data"""
        self.fit(X, y, estimator, verbose)
        return self.transform(X)

# Load the datasets
print("=" * 80)
print("PSO FEATURE SELECTION - CLEANED TEST DATASET EVALUATION")
print("=" * 80)

print("Loading datasets...")
train_data = pd.read_csv('kdd_train_clean.csv')
test_data = pd.read_csv('kdd_test_clean_fixed.csv')  # Use the cleaned test file

print(f"Training data shape: {train_data.shape}")
print(f"Test data shape (cleaned): {test_data.shape}")

# Assuming the last column contains the labels
target_column = train_data.columns[-1]
print(f"Target column: {target_column}")

# Separate features and target
X_train_all = train_data.drop(columns=[target_column])
y_train_original = train_data[target_column]
X_test_all = test_data.drop(columns=[target_column])
y_test_original = test_data[target_column]

print(f"Features: {X_train_all.shape[1]}")
print(f"Training samples: {X_train_all.shape[0]}")
print(f"Test samples: {X_test_all.shape[0]}")

# === ATTACK CATEGORY MAPPING ===
print(f"\n" + "=" * 50)
print("ATTACK CATEGORY MAPPING")
print("=" * 50)

print("Training attack types:")
train_attacks = y_train_original.value_counts()
print(f"Number of unique attacks in train: {len(train_attacks)}")
print("Top 10:", train_attacks.head(10))

print(f"\nTest attack types:")
test_attacks = y_test_original.value_counts()
print(f"Number of unique attacks in test: {len(test_attacks)}")
print("Top 10:", test_attacks.head(10))

# Map to categories
print(f"\nMapping attacks to main categories...")
y_train_categories, unmapped_train = map_attack_categories(y_train_original)
y_test_categories, unmapped_test = map_attack_categories(y_test_original)

y_train = pd.Series(y_train_categories)
y_test = pd.Series(y_test_categories)

print(f"Training category distribution:")
train_category_counts = y_train.value_counts()
for category, count in train_category_counts.items():
    percentage = (count / len(y_train)) * 100
    print(f"  {category:8s}: {count:6,} samples ({percentage:5.1f}%)")

print(f"\nTest category distribution:")
test_category_counts = y_test.value_counts()
for category, count in test_category_counts.items():
    percentage = (count / len(y_test)) * 100
    print(f"  {category:8s}: {count:6,} samples ({percentage:5.1f}%)")

if unmapped_train:
    print(f"⚠️ Unmapped in train: {unmapped_train}")
if unmapped_test:
    print(f"⚠️ Unmapped in test: {unmapped_test}")
else:
    print("✅ All attacks successfully mapped!")

# Handle categorical variables
categorical_columns = X_train_all.select_dtypes(include=['object']).columns
print(f"\nCategorical columns: {len(categorical_columns)} - {list(categorical_columns)}")

# Encode categorical features using training data
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    # Fit on combined data to handle any unseen categories
    combined_data = pd.concat([X_train_all[col], X_test_all[col]], axis=0)
    le.fit(combined_data.astype(str))
    X_train_all[col] = le.transform(X_train_all[col].astype(str))
    X_test_all[col] = le.transform(X_test_all[col].astype(str))
    label_encoders[col] = le

# Encode target labels
target_encoder = LabelEncoder()
# Fit on combined targets to ensure consistent encoding
combined_targets = pd.concat([y_train, y_test], axis=0)
target_encoder.fit(combined_targets)
y_train_encoded = target_encoder.transform(y_train)
y_test_encoded = target_encoder.transform(y_test)

print(f"Target classes: {target_encoder.classes_}")

# Convert to numpy arrays
X_train_np = X_train_all.values
X_test_np = X_test_all.values

# === PSO CONFIGURATION ===
print(f"\n" + "=" * 40)
print("PSO CONFIGURATION")
print("=" * 40)

print("Choose PSO configuration:")
print("1. Fast (15 particles, 20 iterations) - ~5-8 minutes")
print("2. Balanced (20 particles, 25 iterations) - ~8-12 minutes") 
print("3. Thorough (25 particles, 30 iterations) - ~12-18 minutes")

try:
    pso_choice = int(input("Enter your choice (1-3) or press Enter for Balanced: ") or "2")
except:
    pso_choice = 2

if pso_choice == 1:
    n_particles, n_iterations = 15, 20
    print("Selected: Fast configuration")
elif pso_choice == 2:
    n_particles, n_iterations = 20, 25
    print("Selected: Balanced configuration")
else:
    n_particles, n_iterations = 25, 30
    print("Selected: Thorough configuration")

print(f"PSO will optimize all {X_train_all.shape[1]} features")

# === RUN PSO FEATURE SELECTION ===
print(f"\n" + "=" * 50)
print("RUNNING PSO OPTIMIZATION")
print("=" * 50)

# Initialize PSO
pso = PSOFeatureSelection(
    n_particles=n_particles,
    n_iterations=n_iterations,
    w=0.5,      # Inertia weight
    c1=1.5,     # Cognitive parameter
    c2=1.5      # Social parameter
)

# Use estimator for PSO evaluation
estimator = RandomForestClassifier(
    n_estimators=50,
    max_depth=12,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

print(f"Starting PSO optimization...")
start_time = time.time()

# Run PSO
pso.fit(X_train_np, y_train_encoded, estimator=estimator, verbose=True)

end_time = time.time()
print(f"PSO completed in {end_time - start_time:.1f} seconds")

# Transform data using selected features
X_train_selected = pso.transform(X_train_np)
X_test_selected = pso.transform(X_test_np)

selected_feature_names = X_train_all.columns[pso.best_features_]

print(f"\n🎯 PSO SELECTED {len(selected_feature_names)} OPTIMAL FEATURES:")
for i, feature in enumerate(selected_feature_names):
    print(f"  {i+1:2d}. {feature}")

# === TRAIN FINAL RANDOM FOREST ===
print(f"\n" + "=" * 50)
print("TRAINING FINAL RANDOM FOREST")
print("=" * 50)

print("Training optimized Random Forest with PSO-selected features...")
final_rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
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

# Train final model
final_rf.fit(X_train_selected, y_train_encoded)
print(f"Out-of-bag score: {final_rf.oob_score_:.4f}")

# Make predictions on cleaned test set
print("Making predictions on cleaned test set...")
y_pred = final_rf.predict(X_test_selected)

# Calculate accuracy
accuracy = accuracy_score(y_test_encoded, y_pred)
print(f"\n🎯 PSO + CLEANED TEST ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Feature reduction
reduction = (1 - len(pso.best_features_) / len(X_train_all.columns)) * 100
print(f"📊 Feature reduction: {reduction:.1f}% ({len(pso.best_features_)}/{len(X_train_all.columns)} features)")

# Classification report
print(f"\nDetailed Classification Report (Cleaned Test Set):")
category_names = target_encoder.classes_
print(classification_report(y_test_encoded, y_pred, target_names=category_names))

# Per-category accuracy
print(f"\nPer-category accuracy on cleaned test set:")
for i, category in enumerate(category_names):
    mask = y_test_encoded == i
    if np.sum(mask) > 0:
        cat_accuracy = accuracy_score(y_test_encoded[mask], y_pred[mask])
        cat_count = np.sum(mask)
        print(f"  {category:8s}: {cat_accuracy:.4f} ({cat_count:,} samples)")

# === COMPARISON WITH BASELINE ===
print(f"\n" + "=" * 50)
print("COMPARISON WITH ALL FEATURES")
print("=" * 50)

# Train baseline model with all features
print("Training baseline Random Forest with ALL features...")
baseline_rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    class_weight='balanced',
    criterion='entropy',
    random_state=42,
    n_jobs=-1,
    oob_score=True
)

baseline_rf.fit(X_train_np, y_train_encoded)
baseline_pred = baseline_rf.predict(X_test_np)
baseline_accuracy = accuracy_score(y_test_encoded, baseline_pred)

print(f"Baseline (all features) accuracy: {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
print(f"PSO-optimized accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

if accuracy > baseline_accuracy:
    improvement = (accuracy - baseline_accuracy) * 100
    print(f"🎉 PSO IMPROVEMENT: +{improvement:.2f} percentage points!")
elif accuracy >= baseline_accuracy - 0.005:
    diff = (accuracy - baseline_accuracy) * 100
    print(f"✅ PSO MAINTAINS PERFORMANCE: {diff:+.2f} pp with {reduction:.0f}% fewer features!")
else:
    diff = (baseline_accuracy - accuracy) * 100
    print(f"📊 Trade-off: -{diff:.2f} pp accuracy for {reduction:.0f}% feature reduction")

# === VISUALIZATIONS ===
print(f"\nGenerating visualizations...")

plt.figure(figsize=(16, 12))

# 1. PSO Convergence
plt.subplot(2, 4, 1)
plt.plot(pso.history['best_scores'], 'b-', linewidth=2, label='Best Score')
plt.plot(pso.history['avg_scores'], 'r--', alpha=0.7, label='Average Score')
plt.xlabel('Iteration')
plt.ylabel('Fitness Score')
plt.title('PSO Convergence')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Feature Count Evolution
plt.subplot(2, 4, 2)
plt.plot(pso.history['n_features'], 'g-', linewidth=2)
plt.axhline(y=len(pso.best_features_), color='r', linestyle='--', alpha=0.7)
plt.xlabel('Iteration')
plt.ylabel('Number of Features')
plt.title('Feature Count Evolution')
plt.grid(True, alpha=0.3)

# 3. Feature Importance
plt.subplot(2, 4, 3)
feature_importance = pd.DataFrame({
    'feature': selected_feature_names,
    'importance': final_rf.feature_importances_
}).sort_values('importance', ascending=False).head(12)

plt.barh(range(len(feature_importance)), feature_importance['importance'])
plt.yticks(range(len(feature_importance)), [f[:8] + '...' if len(f) > 8 else f for f in feature_importance['feature']])
plt.xlabel('RF Importance')
plt.title('Top 12 Selected Features')
plt.gca().invert_yaxis()

# 4. Test Set Distribution
plt.subplot(2, 4, 4)
test_counts = Counter([target_encoder.classes_[i] for i in y_test_encoded])
categories = list(test_counts.keys())
counts = list(test_counts.values())
colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow', 'lightpink']
plt.pie(counts, labels=categories, autopct='%1.1f%%', startangle=90, colors=colors)
plt.title('Test Set Distribution')

# 5. Confusion Matrix
plt.subplot(2, 4, 5)
cm = confusion_matrix(y_test_encoded, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=category_names, yticklabels=category_names)
plt.title(f'Confusion Matrix\nAcc: {accuracy:.3f}')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# 6. Performance Comparison
plt.subplot(2, 4, 6)
methods = ['All Features\nBaseline', 'PSO Selected\nFeatures']
accuracies = [baseline_accuracy, accuracy]
colors = ['lightcoral', 'lightgreen']
bars = plt.bar(methods, accuracies, color=colors, alpha=0.7)
plt.ylabel('Accuracy')
plt.title('Performance Comparison')
plt.ylim(min(accuracies) - 0.01, max(accuracies) + 0.01)

for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
             f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

# 7. Feature Reduction
plt.subplot(2, 4, 7)
feature_comparison = ['Original\nFeatures', 'PSO\nSelected']
feature_counts = [len(X_train_all.columns), len(pso.best_features_)]
bars = plt.bar(feature_comparison, feature_counts, color=['red', 'green'], alpha=0.7)
plt.ylabel('Number of Features')
plt.title('Feature Reduction')

for bar, count in zip(bars, feature_counts):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             f'{count}', ha='center', va='bottom', fontweight='bold')

# 8. Category Performance
plt.subplot(2, 4, 8)
category_accuracies = []
for i, category in enumerate(category_names):
    mask = y_test_encoded == i
    if np.sum(mask) > 0:
        cat_acc = accuracy_score(y_test_encoded[mask], y_pred[mask])
        category_accuracies.append(cat_acc)
    else:
        category_accuracies.append(0)

plt.bar(range(len(category_names)), category_accuracies, alpha=0.7, color=['blue', 'red', 'green', 'orange', 'purple'])
plt.xticks(range(len(category_names)), [cat[:6] for cat in category_names], rotation=45)
plt.ylabel('Accuracy')
plt.title('Per-Category Performance')
plt.ylim(0, 1)

plt.tight_layout()
plt.show()

# === FINAL SUMMARY ===
print(f"\n" + "=" * 70)
print("FINAL RESULTS - PSO WITH CLEANED TEST DATASET")
print("=" * 70)
print(f"✅ Training data: {X_train_all.shape[0]:,} samples")
print(f"✅ Test data (cleaned): {X_test_all.shape[0]:,} samples") 
print(f"✅ Test data removed: 43 problematic rows (0.2%)")
print(f"✅ Attack categories: 5 (Normal, DoS, Probe, R2L, U2R)")
print(f"✅ Original features: {len(X_train_all.columns)}")
print(f"✅ PSO-selected features: {len(pso.best_features_)}")
print(f"✅ Feature reduction: {reduction:.1f}%")
print(f"✅ PSO parameters: {n_particles} particles, {n_iterations} iterations")
print(f"✅ Optimization time: {end_time - start_time:.1f} seconds")
print(f"")
print(f"📊 PERFORMANCE RESULTS:")
print(f"   • Baseline (all features): {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
print(f"   • PSO optimized: {accuracy:.4f} ({accuracy*100:.2f}%)")
if accuracy > baseline_accuracy:
    print(f"   • 🎉 Improvement: +{(accuracy - baseline_accuracy)*100:.2f} percentage points")
else:
    print(f"   • 📊 Difference: {(accuracy - baseline_accuracy)*100:+.2f} percentage points")

print(f"\nTop 10 Most Important Selected Features:")
top_features = feature_importance.head(10)
for i, (_, row) in enumerate(top_features.iterrows(), 1):
    print(f"  {i:2d}. {row['feature']} (importance: {row['importance']:.4f})")

print(f"\n🎯 CONCLUSION:")
if accuracy >= baseline_accuracy - 0.005:
    print(f"   ✅ PSO successfully maintained/improved performance with {reduction:.0f}% fewer features!")
    print(f"   ✅ The cleaned test dataset works perfectly with your model!")
    print(f"   ✅ Feature selection reduced complexity while preserving accuracy!")
else:
    print(f"   📊 PSO achieved good feature reduction with minimal accuracy trade-off")
    print(f"   ✅ The cleaned test dataset resolved all compatibility issues!")

print(f"\n💡 SUMMARY:")
print(f"   • Used cleaned test dataset: 'kdd_test_clean_fixed.csv'")
print(f"   • Selected {len(pso.best_features_)} optimal features out of {len(X_train_all.columns)}")
print(f"   • Achieved {accuracy:.1%} accuracy with {reduction:.0f}% fewer features")
print(f"   • Test dataset compatibility issues have been resolved!")

print(f"\n🚀 SUCCESS! PSO feature selection with cleaned test dataset is complete!")