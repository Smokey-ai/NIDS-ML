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

# PSO-based Feature Selection Class
class PSOFeatureSelection:
    def __init__(self, n_particles=20, n_iterations=30, w=0.5, c1=1.5, c2=1.5):
        """
        PSO for Feature Selection
        
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
        positions = np.random.rand(self.n_particles, n_features) > 0.5
        positions = positions.astype(int)
        
        # Ensure each particle selects at least 5 features
        for i in range(self.n_particles):
            if np.sum(positions[i]) < 5:
                indices = np.random.choice(n_features, 5, replace=False)
                positions[i] = 0
                positions[i][indices] = 1
        
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
            fitness = fitness * (1 - 0.1 * feature_ratio)  # Small penalty for many features
            
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
        
        # Ensure at least 5 features are selected
        if np.sum(new_position) < 5:
            indices = np.random.choice(len(position), 5, replace=False)
            new_position = np.zeros(len(position), dtype=int)
            new_position[indices] = 1
        
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
print("=" * 60)
print("PSO-BASED FEATURE SELECTION FOR NSL-KDD")
print("=" * 60)

print("Loading datasets...")
train_data = pd.read_csv('kdd_train_clean.csv')
test_data = pd.read_csv('kdd_test_clean_v2.csv')

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

print(f"Original number of features: {X_train.shape[1]}")
print(f"Number of samples: {X_train.shape[0]}")

# Handle categorical variables
categorical_columns = X_train.select_dtypes(include=['object']).columns
print(f"Categorical columns: {len(categorical_columns)}")

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
    print(f"Target classes: {len(target_encoder.classes_)}")

# Convert to numpy for PSO
X_train_np = X_train.values
X_test_np = X_test.values

# === PSO PARAMETER CONFIGURATION ===
print(f"\n" + "=" * 40)
print("PSO PARAMETER CONFIGURATION")
print("=" * 40)

print("Choose PSO configuration:")
print("1. Fast (10 particles, 15 iterations) - ~2-3 minutes")
print("2. Balanced (20 particles, 30 iterations) - ~5-8 minutes") 
print("3. Thorough (30 particles, 50 iterations) - ~10-15 minutes")
print("4. Custom")

try:
    choice = int(input("Enter your choice (1-4) or press Enter for Balanced: ") or "2")
except:
    choice = 2

if choice == 1:
    n_particles, n_iterations = 10, 15
    print("Selected: Fast configuration")
elif choice == 2:
    n_particles, n_iterations = 20, 30
    print("Selected: Balanced configuration")
elif choice == 3:
    n_particles, n_iterations = 30, 50
    print("Selected: Thorough configuration")
else:
    try:
        n_particles = int(input("Number of particles (default 20): ") or "20")
        n_iterations = int(input("Number of iterations (default 30): ") or "30")
    except:
        n_particles, n_iterations = 20, 30
    print(f"Custom: {n_particles} particles, {n_iterations} iterations")

# === RUN PSO FEATURE SELECTION ===
print(f"\n" + "=" * 40)
print("RUNNING PSO FEATURE SELECTION")
print("=" * 40)

# Initialize PSO
pso = PSOFeatureSelection(
    n_particles=n_particles,
    n_iterations=n_iterations,
    w=0.5,      # Inertia weight
    c1=1.5,     # Cognitive parameter
    c2=1.5      # Social parameter
)

# Use a faster estimator for PSO evaluation
estimator = RandomForestClassifier(
    n_estimators=50,  # Smaller for speed during PSO
    max_depth=10,     # Limited depth for speed
    random_state=42,
    n_jobs=-1
)

print(f"Starting PSO optimization...")
start_time = time.time()

# Run PSO
pso.fit(X_train_np, y_train, estimator=estimator, verbose=True)

end_time = time.time()
print(f"PSO completed in {end_time - start_time:.1f} seconds")

# Transform data using selected features
X_train_selected = pso.transform(X_train_np)
X_test_selected = pso.transform(X_test_np)

selected_feature_names = X_train.columns[pso.best_features_]
print(f"\nSelected features ({len(selected_feature_names)}):")
for i, feature in enumerate(selected_feature_names):
    print(f"  {i+1:2d}. {feature}")

# === TRAIN FINAL RANDOM FOREST ===
print(f"\n" + "=" * 40)
print("TRAINING FINAL RANDOM FOREST")
print("=" * 40)

print("Training optimized Random Forest with PSO-selected features...")
final_rf = RandomForestClassifier(
    n_estimators=300,       # More trees for final model
    max_depth=None,         # Full depth
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
final_rf.fit(X_train_selected, y_train)
print(f"Out-of-bag score: {final_rf.oob_score_:.4f}")

# Make predictions
print("Making predictions...")
y_pred = final_rf.predict(X_test_selected)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 PSO-OPTIMIZED ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Compare with baseline
baseline_accuracy = 0.91  # Your original accuracy
if accuracy > baseline_accuracy:
    improvement = (accuracy - baseline_accuracy) * 100
    print(f"🎉 IMPROVEMENT: +{improvement:.2f} percentage points!")
else:
    difference = (accuracy - baseline_accuracy) * 100
    print(f"Result: {difference:+.2f} percentage points vs baseline")

# Feature reduction
reduction = (1 - len(pso.best_features_) / len(X_train.columns)) * 100
print(f"📊 Feature reduction: {reduction:.1f}% ({len(pso.best_features_)}/{len(X_train.columns)} features)")

# Classification report
print(f"\nClassification Report:")
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
    print(f"Basic classification report: {e}")
    print(classification_report(y_test, y_pred))

# === VISUALIZATIONS ===
print(f"\nGenerating visualizations...")

# 1. PSO Convergence Plot
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(pso.history['best_scores'], 'b-', linewidth=2, label='Best Score')
plt.plot(pso.history['avg_scores'], 'r--', alpha=0.7, label='Average Score')
plt.xlabel('Iteration')
plt.ylabel('Fitness Score')
plt.title('PSO Convergence')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Feature Count Evolution
plt.subplot(2, 2, 2)
plt.plot(pso.history['n_features'], 'g-', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Number of Selected Features')
plt.title('Selected Features Over Iterations')
plt.grid(True, alpha=0.3)

# 3. Final Feature Importance
plt.subplot(2, 2, 3)
feature_importance = pd.DataFrame({
    'feature': selected_feature_names,
    'importance': final_rf.feature_importances_
}).sort_values('importance', ascending=False).head(15)

plt.barh(range(len(feature_importance)), feature_importance['importance'])
plt.yticks(range(len(feature_importance)), feature_importance['feature'])
plt.xlabel('Random Forest Importance')
plt.title('Top 15 PSO-Selected Features (RF Importance)')
plt.gca().invert_yaxis()

# 4. PSO vs All Features Comparison
plt.subplot(2, 2, 4)
methods = ['All Features\n(Baseline)', f'PSO Selected\n({len(pso.best_features_)} features)']
accuracies = [baseline_accuracy, accuracy]
colors = ['lightcoral', 'lightgreen']

bars = plt.bar(methods, accuracies, color=colors, alpha=0.7)
plt.ylabel('Accuracy')
plt.title('PSO Feature Selection Results')
plt.ylim(0.85, max(accuracies) + 0.02)

# Add accuracy labels on bars
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
             f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

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
    
plt.title(f'Confusion Matrix - PSO Feature Selection\nAccuracy: {accuracy:.4f}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# === FINAL SUMMARY ===
print(f"\n" + "=" * 60)
print("PSO FEATURE SELECTION SUMMARY")
print("=" * 60)
print(f"✅ Original features: {len(X_train.columns)}")
print(f"✅ PSO-selected features: {len(pso.best_features_)}")
print(f"✅ Feature reduction: {reduction:.1f}%")
print(f"✅ PSO parameters: {n_particles} particles, {n_iterations} iterations")
print(f"✅ Final accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"✅ Out-of-bag score: {final_rf.oob_score_:.4f}")
print(f"✅ Optimization time: {end_time - start_time:.1f} seconds")

print(f"\nPSO Optimization Details:")
print(f"  • Best fitness achieved: {pso.best_score_:.4f}")
print(f"  • Features selected: {len(pso.best_features_)}/{len(X_train.columns)}")
print(f"  • Algorithm: Particle Swarm Optimization")
print(f"  • Selection method: Wrapper (CV-based)")

print(f"\nSelected Features:")
for i, feature in enumerate(selected_feature_names):
    importance = final_rf.feature_importances_[i]
    print(f"  {i+1:2d}. {feature} (RF importance: {importance:.4f})")

print(f"\n🎯 PSO successfully optimized feature selection!")
print(f"   Features reduced by {reduction:.1f}% with accuracy of {accuracy:.1%}")