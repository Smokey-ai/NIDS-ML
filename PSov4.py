import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import time
import warnings
warnings.filterwarnings('ignore')

# Attack type mapping (same as before)
ATTACK_CATEGORIES = {
    'normal': 'Normal',
    'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS', 'smurf': 'DoS',
    'teardrop': 'DoS', 'mailbomb': 'DoS', 'apache2': 'DoS', 'processtable': 'DoS', 'udpstorm': 'DoS',
    'satan': 'Probe', 'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe', 'mscan': 'Probe', 'saint': 'Probe',
    'guess_passwd': 'R2L', 'ftp_write': 'R2L', 'imap': 'R2L', 'phf': 'R2L', 'multihop': 'R2L',
    'warezmaster': 'R2L', 'warezclient': 'R2L', 'spy': 'R2L', 'xlock': 'R2L', 'xsnoop': 'R2L',
    'snmpread': 'R2L', 'snmpguess': 'R2L', 'worm': 'R2L', 'httptunnel': 'R2L', 'named': 'R2L', 'sendmail': 'R2L',
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R', 'rootkit': 'U2R', 'ps': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R'
}

def map_attack_categories(attack_types):
    """Map specific attack types to main categories"""
    categories = []
    for attack in attack_types:
        attack_lower = attack.lower().strip()
        if attack_lower in ATTACK_CATEGORIES:
            categories.append(ATTACK_CATEGORIES[attack_lower])
        elif any(word in attack_lower for word in ['dos', 'flood', 'storm']):
            categories.append('DoS')
        elif any(word in attack_lower for word in ['scan', 'sweep', 'probe']):
            categories.append('Probe')
        elif any(word in attack_lower for word in ['passwd', 'login', 'ftp', 'remote']):
            categories.append('R2L')
        elif any(word in attack_lower for word in ['buffer', 'root', 'overflow']):
            categories.append('U2R')
        else:
            categories.append('Unknown')
    return categories, set()

# Enhanced PSO with Advanced Features
class AdvancedPSOFeatureSelection:
    def __init__(self, n_particles=25, n_iterations=35, w=0.5, c1=1.5, c2=1.5, 
                 adaptive_inertia=True, elite_particles=3):
        """Enhanced PSO with advanced features"""
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.adaptive_inertia = adaptive_inertia
        self.elite_particles = elite_particles
        self.w_min = 0.1
        self.w_max = 0.9

    def initialize_particles(self, n_features):
        """Initialize particles with better diversity"""
        positions = np.zeros((self.n_particles, n_features), dtype=int)
        
        # Create diverse initial population
        for i in range(self.n_particles):
            # Vary the selection probability for diversity
            selection_prob = 0.3 + 0.4 * np.random.rand()  # 30-70% features
            n_select = int(selection_prob * n_features)
            n_select = max(8, min(n_select, int(0.8 * n_features)))  # 8 to 80% features
            
            # Random selection
            selected_indices = np.random.choice(n_features, n_select, replace=False)
            positions[i][selected_indices] = 1
        
        # Ensure some particles start with known good features
        good_features = [0, 1, 2, 3, 4, 8, 9]  # Common important features
        for i in range(min(3, self.n_particles)):
            for feat in good_features:
                if feat < n_features:
                    positions[i][feat] = 1
        
        velocities = np.random.uniform(-2, 2, (self.n_particles, n_features))
        return positions, velocities
    
    def evaluate_fitness(self, position, X_train, y_train, estimator, cv=5):
        """Enhanced fitness evaluation"""
        selected_features = np.where(position == 1)[0]
        
        if len(selected_features) == 0:
            return 0.0
            
        X_selected = X_train[:, selected_features]
        
        try:
            # Use 5-fold CV for more robust evaluation
            scores = cross_val_score(estimator, X_selected, y_train, cv=cv, scoring='accuracy')
            fitness = scores.mean()
            
            # Enhanced penalty system
            n_features = len(selected_features)
            feature_ratio = n_features / len(position)
            
            # Penalty for too many features
            if feature_ratio > 0.7:
                fitness *= (1 - 0.1 * (feature_ratio - 0.7))
            
            # Bonus for optimal range (15-25 features)
            if 15 <= n_features <= 25:
                fitness *= 1.02
            
            return fitness
        except:
            return 0.0
    
    def update_velocity_advanced(self, velocity, position, personal_best, global_best, iteration):
        """Advanced velocity update with adaptive parameters"""
        # Adaptive inertia weight
        if self.adaptive_inertia:
            w = self.w_max - (self.w_max - self.w_min) * iteration / self.n_iterations
        else:
            w = self.w
        
        # Random coefficients
        r1, r2 = np.random.rand(2)
        
        # Enhanced cognitive and social components
        cognitive = self.c1 * r1 * (personal_best - position)
        social = self.c2 * r2 * (global_best - position)
        
        # Add exploration component for diversity
        exploration = 0.1 * np.random.uniform(-1, 1, len(position))
        
        new_velocity = w * velocity + cognitive + social + exploration
        new_velocity = np.clip(new_velocity, -6, 6)  # Wider velocity range
        
        return new_velocity
    
    def update_position_advanced(self, position, velocity):
        """Advanced position update with constraints"""
        sigmoid = 1 / (1 + np.exp(-velocity))
        new_position = (np.random.rand(len(position)) < sigmoid).astype(int)
        
        # Ensure reasonable feature count
        n_selected = np.sum(new_position)
        min_features = 8
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
        """Enhanced PSO with elite preservation"""
        if estimator is None:
            estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        
        n_features = X_train.shape[1]
        positions, velocities = self.initialize_particles(n_features)
        
        # Initialize personal and global bests
        personal_best_positions = positions.copy()
        personal_best_scores = np.array([self.evaluate_fitness(pos, X_train, y_train, estimator) 
                                       for pos in positions])
        
        # Elite tracking
        elite_indices = np.argsort(personal_best_scores)[-self.elite_particles:]
        
        global_best_idx = np.argmax(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx].copy()
        global_best_score = personal_best_scores[global_best_idx]
        
        self.history = {'best_scores': [], 'avg_scores': [], 'n_features': []}
        
        if verbose:
            print(f"Advanced PSO Feature Selection Started")
            print(f"Population: {self.n_particles} particles, {self.n_iterations} iterations")
            print(f"Initial best score: {global_best_score:.4f} with {np.sum(global_best_position)} features")
            print("-" * 60)
        
        stagnation_count = 0
        
        # PSO main loop
        for iteration in range(self.n_iterations):
            iteration_scores = []
            
            for i in range(self.n_particles):
                # Skip elite particles for some iterations to preserve good solutions
                if i in elite_indices and iteration < self.n_iterations * 0.7 and np.random.rand() < 0.3:
                    current_score = personal_best_scores[i]
                else:
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
                    stagnation_count = 0
                    if verbose:
                        print(f"  🎯 New best at iteration {iteration+1}: {global_best_score:.4f} with {np.sum(global_best_position)} features")
                else:
                    stagnation_count += 1
                
                # Update velocity and position with advanced methods
                velocities[i] = self.update_velocity_advanced(
                    velocities[i], positions[i], personal_best_positions[i], 
                    global_best_position, iteration
                )
                positions[i] = self.update_position_advanced(positions[i], velocities[i])
            
            # Diversity injection if stagnating
            if stagnation_count > 10 and iteration < self.n_iterations * 0.8:
                # Reinitialize worst particles
                worst_indices = np.argsort(personal_best_scores)[:self.n_particles//4]
                for idx in worst_indices:
                    positions[idx] = (np.random.rand(n_features) > 0.6).astype(int)
                    velocities[idx] = np.random.uniform(-2, 2, n_features)
                stagnation_count = 0
                if verbose:
                    print(f"  🔄 Diversity injection at iteration {iteration+1}")
            
            # Update elite indices
            elite_indices = np.argsort(personal_best_scores)[-self.elite_particles:]
            
            avg_score = np.mean(iteration_scores)
            self.history['best_scores'].append(global_best_score)
            self.history['avg_scores'].append(avg_score)
            self.history['n_features'].append(np.sum(global_best_position))
            
            if verbose and (iteration + 1) % 5 == 0:
                diversity = np.mean([np.sum(pos) for pos in positions])
                print(f"Iteration {iteration+1:2d}: Best={global_best_score:.4f}, "
                      f"Avg={avg_score:.4f}, Features={np.sum(global_best_position)}, "
                      f"Diversity={diversity:.1f}")
        
        self.best_features_ = np.where(global_best_position == 1)[0]
        self.best_score_ = global_best_score
        self.best_position_ = global_best_position
        
        if verbose:
            print("-" * 60)
            print(f"Advanced PSO Complete! Best score: {global_best_score:.4f}")
            print(f"Selected features: {len(self.best_features_)}/{n_features}")
        
        return self
    
    def transform(self, X):
        return X[:, self.best_features_]

print("=" * 80)
print("ADVANCED PSO FOR HIGHER ACCURACY (Current: 92.19%)")
print("=" * 80)

# Load datasets
train_data = pd.read_csv('kdd_train_clean.csv')
test_data = pd.read_csv('kdd_test_clean_fixed.csv')

target_column = train_data.columns[-1]
X_train_all = train_data.drop(columns=[target_column])
y_train_original = train_data[target_column]
X_test_all = test_data.drop(columns=[target_column])
y_test_original = test_data[target_column]

# Map to categories and encode
y_train_categories, _ = map_attack_categories(y_train_original)
y_test_categories, _ = map_attack_categories(y_test_original)
y_train = pd.Series(y_train_categories)
y_test = pd.Series(y_test_categories)

# Encode categorical features
categorical_columns = X_train_all.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    combined_data = pd.concat([X_train_all[col], X_test_all[col]], axis=0)
    le.fit(combined_data.astype(str))
    X_train_all[col] = le.transform(X_train_all[col].astype(str))
    X_test_all[col] = le.transform(X_test_all[col].astype(str))
    label_encoders[col] = le

# Encode targets
target_encoder = LabelEncoder()
combined_targets = pd.concat([y_train, y_test], axis=0)
target_encoder.fit(combined_targets)
y_train_encoded = target_encoder.transform(y_train)
y_test_encoded = target_encoder.transform(y_test)

X_train_np = X_train_all.values
X_test_np = X_test_all.values

print(f"Training samples: {X_train_np.shape[0]:,}")
print(f"Test samples: {X_test_np.shape[0]:,}")
print(f"Features: {X_train_np.shape[1]}")

# === ENHANCEMENT 1: ADVANCED DATA PREPROCESSING ===
print(f"\n" + "=" * 50)
print("ENHANCEMENT 1: ADVANCED DATA PREPROCESSING")
print("=" * 50)

print("Choose preprocessing enhancement:")
print("1. No preprocessing (current approach)")
print("2. Feature scaling + class balancing") 
print("3. Feature scaling only")
print("4. Class balancing only")

try:
    prep_choice = int(input("Enter choice (1-4) or press Enter for option 2: ") or "2")
except:
    prep_choice = 2

X_train_processed = X_train_np.copy()
X_test_processed = X_test_np.copy()
y_train_processed = y_train_encoded.copy()

if prep_choice in [2, 3]:  # Feature scaling
    print("Applying RobustScaler to numerical features...")
    numerical_mask = ~np.isin(range(X_train_np.shape[1]), 
                            [X_train_all.columns.get_loc(col) for col in categorical_columns])
    
    scaler = RobustScaler()
    X_train_processed[:, numerical_mask] = scaler.fit_transform(X_train_processed[:, numerical_mask])
    X_test_processed[:, numerical_mask] = scaler.transform(X_test_processed[:, numerical_mask])
    print("✅ Feature scaling applied")

if prep_choice in [2, 4]:  # Class balancing
    print("Applying SMOTE for class balancing...")
    try:
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_train_processed, y_train_processed = smote.fit_resample(X_train_processed, y_train_processed)
        
        print(f"Original training size: {len(y_train_encoded):,}")
        print(f"Balanced training size: {len(y_train_processed):,}")
        
        # Show new distribution
        balanced_counts = Counter(y_train_processed)
        for i, category in enumerate(target_encoder.classes_):
            count = balanced_counts.get(i, 0)
            print(f"  {category}: {count:,}")
        
        print("✅ Class balancing applied")
    except Exception as e:
        print(f"⚠️ SMOTE failed: {e}. Using original data.")
        y_train_processed = y_train_encoded.copy()

# === ENHANCEMENT 2: ADVANCED PSO PARAMETERS ===
print(f"\n" + "=" * 50)
print("ENHANCEMENT 2: ADVANCED PSO OPTIMIZATION")
print("=" * 50)

print("Choose PSO enhancement level:")
print("1. Standard PSO (your current)")
print("2. Enhanced PSO (25 particles, 35 iterations)")
print("3. Intensive PSO (30 particles, 45 iterations)")
print("4. Maximum PSO (35 particles, 60 iterations)")

try:
    pso_choice = int(input("Enter choice (1-4) or press Enter for option 3: ") or "3")
except:
    pso_choice = 3

if pso_choice == 1:
    n_particles, n_iterations = 20, 25
    print("Using standard PSO")
elif pso_choice == 2:
    n_particles, n_iterations = 25, 35
    print("Using enhanced PSO")
elif pso_choice == 3:
    n_particles, n_iterations = 30, 45
    print("Using intensive PSO")
else:
    n_particles, n_iterations = 35, 60
    print("Using maximum PSO")

# === ENHANCEMENT 3: ADVANCED ESTIMATOR ===
print(f"\n" + "=" * 50)
print("ENHANCEMENT 3: ADVANCED ESTIMATOR")
print("=" * 50)

print("Choose base estimator for PSO:")
print("1. Standard Random Forest")
print("2. Optimized Random Forest")
print("3. Extra Trees Classifier")

try:
    est_choice = int(input("Enter choice (1-3) or press Enter for option 2: ") or "2")
except:
    est_choice = 2

if est_choice == 1:
    pso_estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1, class_weight='balanced')
elif est_choice == 2:
    pso_estimator = RandomForestClassifier(
        n_estimators=75, max_depth=15, min_samples_split=3, min_samples_leaf=1,
        max_features='sqrt', bootstrap=True, class_weight='balanced_subsample',
        criterion='entropy', random_state=42, n_jobs=-1
    )
else:
    pso_estimator = ExtraTreesClassifier(
        n_estimators=75, max_depth=15, min_samples_split=3, min_samples_leaf=1,
        max_features='sqrt', bootstrap=False, class_weight='balanced',
        criterion='entropy', random_state=42, n_jobs=-1
    )

print(f"Selected estimator: {pso_estimator.__class__.__name__}")

# === RUN ADVANCED PSO ===
print(f"\n" + "=" * 50)
print("RUNNING ADVANCED PSO OPTIMIZATION")
print("=" * 50)

print(f"Configuration:")
print(f"  • Particles: {n_particles}")
print(f"  • Iterations: {n_iterations}")
print(f"  • Preprocessing: {'Scaling + Balancing' if prep_choice == 2 else 'Scaling Only' if prep_choice == 3 else 'Balancing Only' if prep_choice == 4 else 'None'}")
print(f"  • Estimator: {pso_estimator.__class__.__name__}")
print(f"  • Expected time: {n_particles * n_iterations // 50}-{n_particles * n_iterations // 25} minutes")

start_time = time.time()

# Initialize advanced PSO
advanced_pso = AdvancedPSOFeatureSelection(
    n_particles=n_particles,
    n_iterations=n_iterations,
    w=0.5,
    c1=1.5,
    c2=1.5,
    adaptive_inertia=True,
    elite_particles=max(3, n_particles // 8)
)

# Run optimization
advanced_pso.fit(X_train_processed, y_train_processed, estimator=pso_estimator, verbose=True)

end_time = time.time()
print(f"Advanced PSO completed in {end_time - start_time:.1f} seconds")

# Transform data
X_train_selected = advanced_pso.transform(X_train_processed)
X_test_selected = advanced_pso.transform(X_test_processed)

selected_feature_names = X_train_all.columns[advanced_pso.best_features_]
print(f"\n🎯 SELECTED {len(selected_feature_names)} FEATURES:")
for i, feature in enumerate(selected_feature_names):
    print(f"  {i+1:2d}. {feature}")

# === TRAIN FINAL ENHANCED MODEL ===
print(f"\n" + "=" * 50)
print("TRAINING ENHANCED FINAL MODEL")
print("=" * 50)

# Enhanced final model
final_enhanced_rf = RandomForestClassifier(
    n_estimators=500,       # More trees
    max_depth=None,         # Full depth
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    class_weight='balanced_subsample',  # Better for imbalanced data
    criterion='entropy',    # Often better for classification
    random_state=42,
    n_jobs=-1,
    oob_score=True
)

# Train and evaluate
final_enhanced_rf.fit(X_train_selected, y_train_processed)
print(f"Out-of-bag score: {final_enhanced_rf.oob_score_:.4f}")

y_pred_enhanced = final_enhanced_rf.predict(X_test_selected)
accuracy_enhanced = accuracy_score(y_test_encoded, y_pred_enhanced)

print(f"\n🎯 ENHANCED ACCURACY: {accuracy_enhanced:.4f} ({accuracy_enhanced*100:.2f}%)")
print(f"📊 Previous accuracy: 92.19%")

if accuracy_enhanced > 0.9219:
    improvement = (accuracy_enhanced - 0.9219) * 100
    print(f"🎉 IMPROVEMENT: +{improvement:.2f} percentage points!")
else:
    diff = (0.9219 - accuracy_enhanced) * 100
    print(f"📊 Difference: -{diff:.2f} percentage points")

# Feature reduction
reduction = (1 - len(advanced_pso.best_features_) / len(X_train_all.columns)) * 100
print(f"📊 Feature reduction: {reduction:.1f}% ({len(advanced_pso.best_features_)}/{len(X_train_all.columns)} features)")

# Detailed results
print(f"\nDetailed Classification Report:")
print(classification_report(y_test_encoded, y_pred_enhanced, target_names=target_encoder.classes_))

# Visualizations
plt.figure(figsize=(15, 10))

# PSO Convergence
plt.subplot(2, 3, 1)
plt.plot(advanced_pso.history['best_scores'], 'b-', linewidth=2, label='Best Score')
plt.plot(advanced_pso.history['avg_scores'], 'r--', alpha=0.7, label='Average Score')
plt.xlabel('Iteration')
plt.ylabel('Fitness Score')
plt.title('Advanced PSO Convergence')
plt.legend()
plt.grid(True, alpha=0.3)

# Feature Count Evolution
plt.subplot(2, 3, 2)
plt.plot(advanced_pso.history['n_features'], 'g-', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Number of Features')
plt.title('Feature Count Evolution')
plt.grid(True, alpha=0.3)

# Feature Importance
plt.subplot(2, 3, 3)
feature_importance = pd.DataFrame({
    'feature': selected_feature_names,
    'importance': final_enhanced_rf.feature_importances_
}).sort_values('importance', ascending=False).head(12)

plt.barh(range(len(feature_importance)), feature_importance['importance'])
plt.yticks(range(len(feature_importance)), [f[:10] + '...' if len(f) > 10 else f for f in feature_importance['feature']])
plt.xlabel('Importance')
plt.title('Top 12 Features')
plt.gca().invert_yaxis()

# Confusion Matrix
plt.subplot(2, 3, 4)
cm = confusion_matrix(y_test_encoded, y_pred_enhanced)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_encoder.classes_, yticklabels=target_encoder.classes_)
plt.title(f'Confusion Matrix\nAcc: {accuracy_enhanced:.3f}')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Accuracy Comparison
plt.subplot(2, 3, 5)
methods = ['Previous\n(92.19%)', 'Enhanced\nPSO']
accuracies = [0.9219, accuracy_enhanced]
colors = ['lightcoral', 'lightgreen' if accuracy_enhanced > 0.9219 else 'lightyellow']
bars = plt.bar(methods, accuracies, color=colors, alpha=0.7)
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison')
plt.ylim(min(accuracies) - 0.01, max(accuracies) + 0.01)

for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
             f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

# Per-category Performance
plt.subplot(2, 3, 6)
category_accs = []
for i, category in enumerate(target_encoder.classes_):
    mask = y_test_encoded == i
    if np.sum(mask) > 0:
        cat_acc = accuracy_score(y_test_encoded[mask], y_pred_enhanced[mask])
        category_accs.append(cat_acc)
    else:
        category_accs.append(0)

colors = ['blue', 'red', 'green', 'orange', 'purple']
plt.bar(range(len(target_encoder.classes_)), category_accs, alpha=0.7, color=colors)
plt.xticks(range(len(target_encoder.classes_)), [cat[:6] for cat in target_encoder.classes_], rotation=45)
plt.ylabel('Accuracy')
plt.title('Per-Category Performance')
plt.ylim(0, 1)

plt.tight_layout()
plt.show()

print(f"\n" + "=" * 70)
print("ENHANCEMENT RESULTS SUMMARY")
print("=" * 70)
print(f"🎯 Previous accuracy: 92.19%")
print(f"🎯 Enhanced accuracy: {accuracy_enhanced*100:.2f}%")
print(f"🎯 Change: {(accuracy_enhanced - 0.9219)*100:+.2f} percentage points")
print(f"📊 Features used: {len(advanced_pso.best_features_)} out of {len(X_train_all.columns)}")
print(f"📊 Feature reduction: {reduction:.1f}%")
print(f"⏱️  Optimization time: {end_time - start_time:.1f} seconds")

if accuracy_enhanced > 0.9219:
    print(f"\n🎉 SUCCESS! Enhanced PSO achieved better accuracy!")
elif accuracy_enhanced > 0.921:
    print(f"\n✅ GOOD! Very close to previous best with significant feature reduction!")
else:
    print(f"\n💡 TIP: Try different enhancement combinations or increase PSO parameters!")

print(f"\n💡 NEXT STEPS TO TRY:")
print(f"   • Ensemble multiple PSO runs with different seeds")
print(f"   • Combine with other algorithms (XGBoost, CatBoost)")
print(f"   • Feature engineering (create new features)")
print(f"   • Advanced sampling techniques (ADASYN, BorderlineSMOTE)")
print(f"   • Hyperparameter tuning of the final Random Forest")