import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
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

# PSO-based Feature Selection Class (Simplified - No Mandatory Features)
class PSOFeatureSelection:
    def __init__(self, n_particles=20, n_iterations=30, w=0.5, c1=1.5, c2=1.5):
        """
        PSO for Feature Selection - Free Optimization (No Mandatory Features)
        
        Parameters:
        - n_particles: Number of particles in swarm
        - n_iterations: Number of PSO iterations
        - w: Inertia weight
        - c1: Cognitive parameter
        - c2: Social parameter
        """
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w  # Inertia weight
        self.c1 = c1  # Cognitive parameter
        self.c2 = c2  # Social parameter

    def initialize_particles(self, n_features):
        """Initialize particle positions and velocities"""
        # Position: binary array (1=selected, 0=not selected)
        # Start with 30-70% of features selected
        positions = np.random.rand(self.n_particles, n_features) > 0.5
        positions = positions.astype(int)
        
        # Ensure each particle selects at least 5 features and at most 80% of features
        for i in range(self.n_particles):
            n_selected = np.sum(positions[i])
            min_features = 5
            max_features = int(0.8 * n_features)
            
            if n_selected < min_features:
                # Add random features to reach minimum
                zero_indices = np.where(positions[i] == 0)[0]
                if len(zero_indices) >= min_features - n_selected:
                    add_indices = np.random.choice(zero_indices, min_features - n_selected, replace=False)
                    positions[i][add_indices] = 1
            elif n_selected > max_features:
                # Remove random features to reach maximum
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
            fitness = fitness * (1 - 0.05 * feature_ratio)  # Small penalty for many features
            
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
            print(f"PSO Feature Selection Started (Free Optimization)")
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

# Load the dataset
print("=" * 70)
print("PSO FEATURE SELECTION - ATTACK CATEGORIES (NO MANDATORY FEATURES)")
print("=" * 70)

print("Loading training dataset...")
train_data = pd.read_csv('kdd_train_clean.csv')
print(f"Training data shape: {train_data.shape}")

# Assuming the last column contains the labels
target_column = train_data.columns[-1]
print(f"Target column: {target_column}")

# Separate features and target
X_all = train_data.drop(columns=[target_column])
y_all_original = train_data[target_column]

print(f"Total features: {X_all.shape[1]}")
print(f"Total samples: {X_all.shape[0]}")

# === ATTACK CATEGORY MAPPING ===
print(f"\n" + "=" * 50)
print("ATTACK CATEGORY MAPPING")
print("=" * 50)

print("Original attack types found:")
original_attacks = y_all_original.value_counts()
print(f"Number of unique attack types: {len(original_attacks)}")
print("Top 10 most common:")
print(original_attacks.head(10))

# Map to categories
y_all_categories, unmapped = map_attack_categories(y_all_original)
y_all = pd.Series(y_all_categories)

print(f"\nMapped to 5 main categories:")
category_counts = y_all.value_counts()
print(category_counts)

if unmapped:
    print(f"\nUnmapped attack types: {unmapped}")
    print("These were classified based on name patterns")
else:
    print(f"\n✅ All attack types successfully mapped!")

# Show mapping summary
print(f"\nFinal Category Distribution:")
total_samples = len(y_all)
for category, count in category_counts.items():
    percentage = (count / total_samples) * 100
    print(f"  {category:8s}: {count:6,} samples ({percentage:5.1f}%)")

# Handle categorical variables
categorical_columns = X_all.select_dtypes(include=['object']).columns
numerical_columns = X_all.select_dtypes(exclude=['object']).columns
print(f"\nFeature types:")
print(f"  Categorical: {len(categorical_columns)}")
print(f"  Numerical: {len(numerical_columns)}")

# Encode categorical features
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    le.fit(X_all[col].astype(str))
    X_all[col] = le.transform(X_all[col].astype(str))
    label_encoders[col] = le

# Encode target labels
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y_all)
print(f"\nEncoded target classes: {target_encoder.classes_}")

# === TRAIN-TEST SPLIT FROM TRAINING DATA ===
print(f"\n" + "=" * 50)
print("INTERNAL TRAIN-TEST SPLIT")
print("=" * 50)

# Split the training data
test_size = 0.25  # Use 25% for internal testing
random_state = 42

print(f"Splitting data: {100-test_size*100:.0f}% train, {test_size*100:.0f}% test")

X_train, X_test, y_train, y_test = train_test_split(
    X_all.values, y_encoded, 
    test_size=test_size, 
    random_state=random_state,
    stratify=y_encoded  # Maintain class distribution
)

print(f"Train set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Show class distribution in splits
print(f"\nClass distribution comparison:")
print(f"{'Category':<10} {'Train Count':<12} {'Test Count':<11} {'Train %':<9} {'Test %'}")
print("-" * 60)

train_categories = [target_encoder.classes_[i] for i in y_train]
test_categories = [target_encoder.classes_[i] for i in y_test]
train_counts = Counter(train_categories)
test_counts = Counter(test_categories)

for category in target_encoder.classes_:
    train_count = train_counts.get(category, 0)
    test_count = test_counts.get(category, 0)
    train_pct = (train_count / len(y_train)) * 100
    test_pct = (test_count / len(y_test)) * 100
    print(f"{category:<10} {train_count:<12,} {test_count:<11,} {train_pct:<8.1f}% {test_pct:<8.1f}%")

# === PSO CONFIGURATION ===
print(f"\n" + "=" * 40)
print("PSO CONFIGURATION")
print("=" * 40)

print("Choose PSO configuration for 5-category classification:")
print("1. Fast (15 particles, 20 iterations) - ~3-5 minutes")
print("2. Balanced (20 particles, 25 iterations) - ~6-10 minutes") 
print("3. Thorough (25 particles, 35 iterations) - ~10-15 minutes")
print("4. Intensive (30 particles, 40 iterations) - ~15-20 minutes")

try:
    pso_choice = int(input("Enter your choice (1-4) or press Enter for Balanced: ") or "2")
except:
    pso_choice = 2

if pso_choice == 1:
    n_particles, n_iterations = 15, 20
    print("Selected: Fast configuration")
elif pso_choice == 2:
    n_particles, n_iterations = 20, 25
    print("Selected: Balanced configuration")
elif pso_choice == 3:
    n_particles, n_iterations = 25, 35
    print("Selected: Thorough configuration")
else:
    n_particles, n_iterations = 30, 40
    print("Selected: Intensive configuration")

print(f"PSO will optimize all {X_all.shape[1]} features freely (no mandatory constraints)")

# === RUN PSO FEATURE SELECTION ===
print(f"\n" + "=" * 50)
print("RUNNING PSO OPTIMIZATION")
print("=" * 50)

# Initialize PSO (no mandatory features)
pso = PSOFeatureSelection(
    n_particles=n_particles,
    n_iterations=n_iterations,
    w=0.5,      # Inertia weight
    c1=1.5,     # Cognitive parameter
    c2=1.5      # Social parameter
)

# Use a balanced estimator for PSO evaluation
estimator = RandomForestClassifier(
    n_estimators=50,  # Smaller for speed during PSO
    max_depth=12,     # Moderate depth
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'  # Handle category imbalance
)

print(f"Starting PSO optimization for 5 attack categories...")
print(f"Search space: 2^{X_all.shape[1]} possible feature combinations")
start_time = time.time()

# Run PSO
pso.fit(X_train, y_train, estimator=estimator, verbose=True)

end_time = time.time()
print(f"PSO completed in {end_time - start_time:.1f} seconds")

# Transform data using selected features
X_train_selected = pso.transform(X_train)
X_test_selected = pso.transform(X_test)

selected_feature_names = X_all.columns[pso.best_features_]

print(f"\n🎯 PSO SELECTED {len(selected_feature_names)} OPTIMAL FEATURES:")
for i, feature in enumerate(selected_feature_names):
    print(f"  {i+1:2d}. {feature}")

# === TRAIN FINAL RANDOM FOREST ===
print(f"\n" + "=" * 50)
print("TRAINING FINAL RANDOM FOREST")
print("=" * 50)

print("Training optimized Random Forest with PSO-selected features...")
final_rf = RandomForestClassifier(
    n_estimators=300,       # More trees for final model
    max_depth=None,         # Full depth
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    class_weight='balanced',  # Handle category imbalance
    criterion='entropy',
    random_state=42,
    n_jobs=-1,
    verbose=1,
    oob_score=True
)

# Train final model
final_rf.fit(X_train_selected, y_train)
print(f"Out-of-bag score: {final_rf.oob_score_:.4f}")

# Make predictions
print("Making predictions on test set...")
y_pred = final_rf.predict(X_test_selected)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 PSO-OPTIMIZED ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Feature reduction
reduction = (1 - len(pso.best_features_) / len(X_all.columns)) * 100
print(f"📊 Feature reduction: {reduction:.1f}% ({len(pso.best_features_)}/{len(X_all.columns)} features)")

# Classification report with category names
print(f"\nDetailed Classification Report (5 Attack Categories):")
category_names = target_encoder.classes_
print(classification_report(y_test, y_pred, target_names=category_names))

# Per-category accuracy
print(f"\nPer-category accuracy:")
for i, category in enumerate(category_names):
    mask = y_test == i
    if np.sum(mask) > 0:
        cat_accuracy = accuracy_score(y_test[mask], y_pred[mask])
        cat_count = np.sum(mask)
        print(f"  {category:8s}: {cat_accuracy:.4f} ({cat_count:,} samples)")

# === VISUALIZATIONS ===
print(f"\nGenerating comprehensive visualizations...")

plt.figure(figsize=(16, 12))

# 1. PSO Convergence Plot
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
plt.axhline(y=len(pso.best_features_), color='r', linestyle='--', alpha=0.7, label=f'Final: {len(pso.best_features_)}')
plt.xlabel('Iteration')
plt.ylabel('Number of Selected Features')
plt.title('Feature Count Over Iterations')
plt.legend()
plt.grid(True, alpha=0.3)

# 3. Top Feature Importance
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

# 4. Attack Category Distribution (Test Set)
plt.subplot(2, 4, 4)
test_category_counts = Counter([target_encoder.classes_[i] for i in y_test])
categories = list(test_category_counts.keys())
counts = list(test_category_counts.values())

colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow', 'lightpink']
plt.pie(counts, labels=categories, autopct='%1.1f%%', startangle=90, colors=colors)
plt.title('Test Set Distribution')

# 5. Confusion Matrix
plt.subplot(2, 4, 5)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=category_names, yticklabels=category_names)
plt.title(f'Confusion Matrix\nAcc: {accuracy:.3f}')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# 6. Feature Selection Comparison
plt.subplot(2, 4, 6)
all_features = len(X_all.columns)
selected_features = len(pso.best_features_)

categories_fs = ['Original\nFeatures', 'PSO\nSelected']
feature_counts = [all_features, selected_features]
colors_fs = ['lightcoral', 'lightgreen']

bars = plt.bar(categories_fs, feature_counts, color=colors_fs, alpha=0.7)
plt.ylabel('Number of Features')
plt.title('Feature Reduction')

for bar, count in zip(bars, feature_counts):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{count}', ha='center', va='bottom', fontweight='bold')

# 7. PSO Performance Evolution
plt.subplot(2, 4, 7)
plt.plot(range(1, len(pso.history['best_scores'])+1), pso.history['best_scores'], 'bo-', markersize=4)
plt.xlabel('Iteration')
plt.ylabel('Best Fitness')
plt.title('PSO Best Performance')
plt.grid(True, alpha=0.3)

# 8. Category-wise Performance
plt.subplot(2, 4, 8)
category_accuracies = []
for i, category in enumerate(category_names):
    mask = y_test == i
    if np.sum(mask) > 0:
        cat_acc = accuracy_score(y_test[mask], y_pred[mask])
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
print(f"\n" + "=" * 60)
print("PSO FEATURE SELECTION SUMMARY")
print("=" * 60)
print(f"✅ Dataset: Training data only (internal split)")
print(f"✅ Attack categories: 5 (Normal, DoS, Probe, R2L, U2R)")
print(f"✅ Original features: {len(X_all.columns)}")
print(f"✅ PSO-selected features: {len(pso.best_features_)}")
print(f"✅ Feature reduction: {reduction:.1f}%")
print(f"✅ PSO parameters: {n_particles} particles, {n_iterations} iterations")
print(f"✅ No mandatory features: Full optimization freedom")
print(f"✅ Final accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"✅ Out-of-bag score: {final_rf.oob_score_:.4f}")
print(f"✅ Optimization time: {end_time - start_time:.1f} seconds")

print(f"\nPSO Optimization Details:")
print(f"  • Best fitness achieved: {pso.best_score_:.4f}")
print(f"  • Search space explored: {n_particles * n_iterations:,} evaluations")
print(f"  • Feature space reduction: {reduction:.1f}%")
print(f"  • Algorithm: Particle Swarm Optimization")
print(f"  • Selection method: Wrapper (CV-based fitness)")

print(f"\nTop 10 Most Important Selected Features:")
top_features = feature_importance.head(10)
for i, (_, row) in enumerate(top_features.iterrows(), 1):
    print(f"  {i:2d}. {row['feature']} (importance: {row['importance']:.4f})")

print(f"\n🎯 PSO successfully optimized feature selection for 5-category attack classification!")
print(f"   Achieved {accuracy:.1%} accuracy with {reduction:.0f}% fewer features")