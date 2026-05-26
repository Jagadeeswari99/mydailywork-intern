"""
IRIS FLOWER CLASSIFICATION
Machine Learning Model for Iris Species Classification
Author: Data Science Student
Date: May 2026

This script loads the Iris dataset, performs exploratory data analysis,
trains multiple classification models, and evaluates their performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report, 
                             ConfusionMatrixDisplay)
import joblib

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================================
# 1. LOAD AND EXPLORE THE IRIS DATASET
# ============================================================================

print("="*80)
print("IRIS FLOWER CLASSIFICATION - MACHINE LEARNING PROJECT")
print("="*80)
print("\n[1] LOADING DATASET...")

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Create a DataFrame for easier manipulation
df = pd.DataFrame(X, columns=feature_names)
df['Species'] = y
df['Species_Name'] = df['Species'].map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})

print(f"Dataset shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())

# ============================================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

print("\n[2] EXPLORATORY DATA ANALYSIS...")

print(f"\nDataset Statistics:")
print(df.describe())

print(f"\nMissing Values:")
print(df.isnull().sum())

print(f"\nClass Distribution:")
print(df['Species_Name'].value_counts())

# Correlation analysis
print(f"\nFeature Correlations:")
correlation_matrix = df.iloc[:, :-2].corr()
print(correlation_matrix)

# ============================================================================
# 3. VISUALIZATION
# ============================================================================

print("\n[3] GENERATING VISUALIZATIONS...")

# Create a comprehensive visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Iris Dataset - Exploratory Data Analysis', fontsize=16, fontweight='bold')

# Feature distributions
for idx, feature in enumerate(feature_names):
    ax = axes[idx // 3, idx % 3]
    for species_id, species_name in enumerate(target_names):
        data = df[df['Species'] == species_id][feature]
        ax.hist(data, alpha=0.5, label=species_name, bins=15)
    ax.set_xlabel(feature)
    ax.set_ylabel('Frequency')
    ax.set_title(f'Distribution of {feature}')
    ax.legend()

# Class distribution
ax = axes[1, 2]
class_counts = df['Species_Name'].value_counts()
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
ax.bar(class_counts.index, class_counts.values, color=colors)
ax.set_ylabel('Count')
ax.set_title('Class Distribution')
ax.set_xlabel('Iris Species')

plt.tight_layout()
plt.savefig('iris_distributions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: iris_distributions.png")
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=1, fmt='.2f')
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('iris_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: iris_correlation_heatmap.png")
plt.show()

# Pairplot
from pandas.plotting import scatter_matrix
scatter_matrix(df.iloc[:, :-2], c=df['Species'], figsize=(10, 8), 
               marker='o', s=100, alpha=0.6, cmap='viridis')
plt.suptitle('Iris Features - Pairplot', fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('iris_pairplot.png', dpi=300, bbox_inches='tight')
print("✓ Saved: iris_pairplot.png")
plt.show()

# ============================================================================
# 4. DATA PREPROCESSING
# ============================================================================

print("\n[4] DATA PREPROCESSING...")

# Feature standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")
print(f"Number of features: {X_train.shape[1]}")

# ============================================================================
# 5. TRAIN MULTIPLE MODELS
# ============================================================================

print("\n[5] TRAINING CLASSIFICATION MODELS...")

models = {
    'Logistic Regression': LogisticRegression(max_iter=200, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Support Vector Machine': SVC(kernel='rbf', random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
}

trained_models = {}
results = {}

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)
    trained_models[model_name] = model
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Cross-validation score
    cv_score = cross_val_score(model, X_train, y_train, cv=5).mean()
    
    results[model_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'CV-Score': cv_score,
        'Predictions': y_pred
    }
    
    print(f"  ✓ Accuracy: {accuracy:.4f}")
    print(f"  ✓ Precision: {precision:.4f}")
    print(f"  ✓ Recall: {recall:.4f}")
    print(f"  ✓ F1-Score: {f1:.4f}")
    print(f"  ✓ Cross-Validation Score: {cv_score:.4f}")

# ============================================================================
# 6. MODEL EVALUATION
# ============================================================================

print("\n[6] MODEL EVALUATION...")

# Create results DataFrame
results_df = pd.DataFrame(results).T
print("\nModel Comparison Table:")
print(results_df.to_string())

# Find best model
best_model_name = results_df['Accuracy'].idxmax()
best_accuracy = results_df['Accuracy'].max()
print(f"\n🏆 BEST MODEL: {best_model_name} with {best_accuracy:.4f} accuracy")

# ============================================================================
# 7. DETAILED EVALUATION FOR BEST MODEL
# ============================================================================

print(f"\n[7] DETAILED EVALUATION FOR {best_model_name.upper()}...")

best_model = trained_models[best_model_name]
y_pred_best = results[best_model_name]['Predictions']

print("\nClassification Report:")
print(classification_report(y_test, y_pred_best, target_names=target_names))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_best)
print(cm)

# ============================================================================
# 8. VISUALIZE RESULTS
# ============================================================================

print("\n[8] VISUALIZING RESULTS...")

# Model comparison bar chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy comparison
ax1 = axes[0]
results_df['Accuracy'].plot(kind='barh', ax=ax1, color='skyblue')
ax1.set_xlabel('Accuracy')
ax1.set_title('Model Accuracy Comparison')
ax1.set_xlim([0.9, 1.0])
for i, v in enumerate(results_df['Accuracy']):
    ax1.text(v - 0.01, i, f'{v:.4f}', ha='right', va='center', fontweight='bold')

# Metrics comparison for best model
ax2 = axes[1]
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [results[best_model_name][m] for m in metrics]
ax2.bar(metrics, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#95E1D3'])
ax2.set_ylabel('Score')
ax2.set_title(f'{best_model_name} - Performance Metrics')
ax2.set_ylim([0.9, 1.0])
for i, v in enumerate(values):
    ax2.text(i, v - 0.01, f'{v:.4f}', ha='center', va='top', fontweight='bold')

plt.tight_layout()
plt.savefig('iris_model_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: iris_model_comparison.png")
plt.show()

# Confusion matrix heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names,
            cbar_kws={'label': 'Count'})
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig(f'iris_confusion_matrix_{best_model_name.replace(" ", "_")}.png', 
            dpi=300, bbox_inches='tight')
print(f"✓ Saved: iris_confusion_matrix_{best_model_name.replace(' ', '_')}.png")
plt.show()

# ============================================================================
# 9. FEATURE IMPORTANCE (for Random Forest)
# ============================================================================

if best_model_name == 'Random Forest' or isinstance(best_model, RandomForestClassifier):
    print("\n[9] FEATURE IMPORTANCE ANALYSIS...")
    
    importances = best_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=True)
    
    print("\nFeature Importance:")
    print(feature_importance_df.to_string())
    
    plt.figure(figsize=(10, 5))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], 
             color='coral')
    plt.xlabel('Importance Score')
    plt.title(f'{best_model_name} - Feature Importance')
    plt.tight_layout()
    plt.savefig('iris_feature_importance.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: iris_feature_importance.png")
    plt.show()

# ============================================================================
# 10. SAVE THE MODEL
# ============================================================================

print("\n[10] SAVING THE TRAINED MODEL...")

model_filename = 'iris_model.pkl'
joblib.dump(best_model, model_filename)
print(f"✓ Model saved as '{model_filename}'")

# ============================================================================
# 11. MAKE PREDICTIONS ON NEW DATA
# ============================================================================

print("\n[11] MAKING PREDICTIONS ON NEW DATA...")

# Example predictions
new_data = np.array([
    [5.1, 3.5, 1.4, 0.2],  # Expected: Setosa
    [6.2, 2.9, 4.3, 1.3],  # Expected: Versicolor
    [6.9, 3.1, 5.4, 2.1]   # Expected: Virginica
])

new_data_scaled = scaler.transform(new_data)
predictions = best_model.predict(new_data_scaled)

print("\nSample Predictions:")
for i, (data, pred) in enumerate(zip(new_data, predictions)):
    print(f"  Sample {i+1}: {data} → {target_names[pred]}")

# ============================================================================
# 12. SUMMARY
# ============================================================================

print("\n" + "="*80)
print("PROJECT SUMMARY")
print("="*80)
print(f"✓ Dataset: Iris Flower Dataset (150 samples, 4 features, 3 classes)")
print(f"✓ Best Model: {best_model_name}")
print(f"✓ Test Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
print(f"✓ Precision: {results[best_model_name]['Precision']:.4f}")
print(f"✓ Recall: {results[best_model_name]['Recall']:.4f}")
print(f"✓ F1-Score: {results[best_model_name]['F1-Score']:.4f}")
print(f"✓ Cross-Validation Score: {results[best_model_name]['CV-Score']:.4f}")
print(f"✓ Model saved as: {model_filename}")
print(f"✓ Visualizations saved: iris_*.png")
print("="*80)
print("\n✅ PROJECT COMPLETED SUCCESSFULLY!\n")
