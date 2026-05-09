# =============================================================
#  TASK 1 — TITANIC SURVIVAL PREDICTION
#  MyDailyWork Data Science Internship
#  Author: Jagadeeswari J M
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)

# ─── Style ────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor':   '#161b22',
    'axes.edgecolor':   '#30363d',
    'axes.labelcolor':  '#c9d1d9',
    'xtick.color':      '#8b949e',
    'ytick.color':      '#8b949e',
    'text.color':       '#c9d1d9',
    'grid.color':       '#21262d',
    'grid.linewidth':   0.5,
    'font.family':      'monospace',
})
COLORS = ['#58a6ff', '#3fb950', '#f78166', '#d2a8ff', '#ffa657']

# ═══════════════════════════════════════════════════════════════
# 1. LOAD / CREATE DATASET
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("  TITANIC SURVIVAL PREDICTION — MyDailyWork Internship")
print("=" * 60)

# Built-in sample dataset (891 passengers, same structure as Kaggle)
np.random.seed(42)
n = 891

pclass   = np.random.choice([1, 2, 3], n, p=[0.24, 0.21, 0.55])
sex      = np.random.choice(['male', 'female'], n, p=[0.65, 0.35])
age_raw  = np.random.normal(29, 14, n)
age_raw  = np.clip(age_raw, 1, 80)
sibsp    = np.random.choice([0,1,2,3,4], n, p=[0.68,0.23,0.05,0.02,0.02])
parch    = np.random.choice([0,1,2,3],   n, p=[0.76,0.13,0.08,0.03])
fare_raw = np.where(pclass==1,
                    np.random.normal(84, 78, n),
                    np.where(pclass==2,
                             np.random.normal(21, 14, n),
                             np.random.normal(13, 12, n)))
fare_raw = np.clip(fare_raw, 5, 512)

# Survival probabilities that mirror real Titanic patterns
p_survive = (
    0.10
    + 0.40 * (sex == 'female')
    + 0.15 * (pclass == 1)
    + 0.05 * (pclass == 2)
    - 0.05 * (age_raw > 60)
    + 0.03 * (parch > 0)
)
p_survive = np.clip(p_survive, 0.05, 0.95)
survived  = (np.random.rand(n) < p_survive).astype(int)

# Add some missing values (like real data)
age_missing = age_raw.copy().astype(float)
missing_idx = np.random.choice(n, size=177, replace=False)
age_missing[missing_idx] = np.nan

embarked = np.random.choice(['S','C','Q'], n, p=[0.72,0.19,0.09])

df = pd.DataFrame({
    'PassengerId': range(1, n+1),
    'Survived':    survived,
    'Pclass':      pclass,
    'Name':        [f'Passenger_{i}' for i in range(1, n+1)],
    'Sex':         sex,
    'Age':         age_missing,
    'SibSp':       sibsp,
    'Parch':       parch,
    'Fare':        fare_raw,
    'Embarked':    embarked
})

print(f"\n✅ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"   Survivors : {survived.sum()} ({survived.mean()*100:.1f}%)")
print(f"   Non-Survivors: {n - survived.sum()} ({(1-survived.mean())*100:.1f}%)")

# ═══════════════════════════════════════════════════════════════
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ═══════════════════════════════════════════════════════════════
print("\n📊 Starting EDA...")

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('TITANIC — Exploratory Data Analysis', fontsize=15,
             fontweight='bold', color='#c9d1d9', y=1.01)

# 2a. Survival count
ax = axes[0, 0]
vals = df['Survived'].value_counts()
bars = ax.bar(['Did Not Survive', 'Survived'], vals.values,
              color=[COLORS[2], COLORS[1]], width=0.5, edgecolor='none')
for bar, val in zip(bars, vals.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            str(val), ha='center', va='bottom', fontsize=11, color='#c9d1d9')
ax.set_title('Survival Count', color='#58a6ff', fontweight='bold')
ax.set_ylabel('Count')
ax.grid(axis='y')

# 2b. Survival by Gender
ax = axes[0, 1]
gender_surv = df.groupby(['Sex', 'Survived']).size().unstack()
gender_surv.plot(kind='bar', ax=ax, color=[COLORS[2], COLORS[1]],
                 edgecolor='none', width=0.6)
ax.set_title('Survival by Gender', color='#58a6ff', fontweight='bold')
ax.set_xlabel('')
ax.set_xticklabels(['Female', 'Male'], rotation=0)
ax.legend(['Did Not Survive', 'Survived'], fontsize=8)
ax.grid(axis='y')

# 2c. Survival by Class
ax = axes[0, 2]
class_surv = df.groupby(['Pclass', 'Survived']).size().unstack()
class_surv.plot(kind='bar', ax=ax, color=[COLORS[2], COLORS[1]],
                edgecolor='none', width=0.6)
ax.set_title('Survival by Passenger Class', color='#58a6ff', fontweight='bold')
ax.set_xlabel('Class')
ax.set_xticklabels(['1st', '2nd', '3rd'], rotation=0)
ax.legend(['Did Not Survive', 'Survived'], fontsize=8)
ax.grid(axis='y')

# 2d. Age distribution
ax = axes[1, 0]
survivors     = df[df['Survived'] == 1]['Age'].dropna()
non_survivors = df[df['Survived'] == 0]['Age'].dropna()
ax.hist(non_survivors, bins=30, alpha=0.7, color=COLORS[2], label='Did Not Survive')
ax.hist(survivors,     bins=30, alpha=0.7, color=COLORS[1], label='Survived')
ax.set_title('Age Distribution by Survival', color='#58a6ff', fontweight='bold')
ax.set_xlabel('Age')
ax.set_ylabel('Count')
ax.legend(fontsize=8)
ax.grid(axis='y')

# 2e. Fare distribution
ax = axes[1, 1]
ax.hist(df[df['Survived']==0]['Fare'], bins=30, alpha=0.7,
        color=COLORS[2], label='Did Not Survive')
ax.hist(df[df['Survived']==1]['Fare'], bins=30, alpha=0.7,
        color=COLORS[1], label='Survived')
ax.set_title('Fare Distribution by Survival', color='#58a6ff', fontweight='bold')
ax.set_xlabel('Fare (£)')
ax.set_ylabel('Count')
ax.legend(fontsize=8)
ax.grid(axis='y')

# 2f. Survival rate by class + gender
ax = axes[1, 2]
pivot = df.groupby(['Pclass', 'Sex'])['Survived'].mean().unstack()
x = np.arange(3)
width = 0.35
ax.bar(x - width/2, pivot['female'], width, color=COLORS[0], label='Female', alpha=0.9)
ax.bar(x + width/2, pivot['male'],   width, color=COLORS[4], label='Male',   alpha=0.9)
ax.set_title('Survival Rate: Class × Gender', color='#58a6ff', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['1st Class', '2nd Class', '3rd Class'])
ax.set_ylabel('Survival Rate')
ax.legend(fontsize=8)
ax.grid(axis='y')

plt.tight_layout()
plt.savefig('/home/claude/titanic_eda.png', dpi=150, bbox_inches='tight',
            facecolor='#0d1117')
plt.close()
print("   ✅ EDA chart saved → titanic_eda.png")

# ═══════════════════════════════════════════════════════════════
# 3. DATA PREPROCESSING
# ═══════════════════════════════════════════════════════════════
print("\n🔧 Preprocessing data...")

data = df.copy()

# Fill missing Age with median
data['Age'].fillna(data['Age'].median(), inplace=True)
print(f"   Missing Age values filled with median: {data['Age'].median():.1f}")

# Encode Sex
le = LabelEncoder()
data['Sex_encoded'] = le.fit_transform(data['Sex'])   # female=0, male=1

# Encode Embarked
data['Embarked_encoded'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Feature Engineering
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
data['IsAlone']    = (data['FamilySize'] == 1).astype(int)
data['FareBand']   = pd.qcut(data['Fare'], 4, labels=[0, 1, 2, 3]).astype(int)
data['AgeBand']    = pd.cut(data['Age'], bins=[0,12,18,35,60,100],
                            labels=[0,1,2,3,4]).cat.add_categories(-1).fillna(-1).astype(int)

FEATURES = ['Pclass', 'Sex_encoded', 'Age', 'SibSp', 'Parch',
            'Fare', 'Embarked_encoded', 'FamilySize', 'IsAlone',
            'FareBand', 'AgeBand']

X = data[FEATURES].fillna(0)
y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"   Train set: {X_train.shape[0]} samples")
print(f"   Test  set: {X_test.shape[0]}  samples")
print(f"   Features : {len(FEATURES)}")

# ═══════════════════════════════════════════════════════════════
# 4. TRAIN MODELS
# ═══════════════════════════════════════════════════════════════
print("\n🤖 Training models...")

models = {
    'Logistic Regression': LogisticRegression(max_iter=500, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    acc   = accuracy_score(y_test, preds)
    auc   = roc_auc_score(y_test, proba)
    results[name] = {'model': model, 'preds': preds, 'proba': proba,
                     'acc': acc, 'auc': auc}
    print(f"   {name:25s} | Accuracy: {acc*100:.2f}%  |  AUC: {auc:.4f}")

best_name = max(results, key=lambda k: results[k]['acc'])
best      = results[best_name]
print(f"\n🏆 Best Model: {best_name}  (Accuracy: {best['acc']*100:.2f}%)")

# ═══════════════════════════════════════════════════════════════
# 5. MODEL EVALUATION CHARTS
# ═══════════════════════════════════════════════════════════════
print("\n📈 Generating evaluation charts...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('TITANIC — Model Evaluation Results', fontsize=15,
             fontweight='bold', color='#c9d1d9')

# 5a. Model Comparison
ax = axes[0, 0]
names = list(results.keys())
accs  = [results[n]['acc']*100 for n in names]
aucs  = [results[n]['auc']*100 for n in names]
x = np.arange(len(names))
bars1 = ax.bar(x - 0.2, accs, 0.35, color=COLORS[0], label='Accuracy %', alpha=0.9)
bars2 = ax.bar(x + 0.2, aucs, 0.35, color=COLORS[3], label='AUC %',      alpha=0.9)
for bar in bars1:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
            f'{bar.get_height():.1f}', ha='center', fontsize=8, color='#c9d1d9')
for bar in bars2:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
            f'{bar.get_height():.1f}', ha='center', fontsize=8, color='#c9d1d9')
ax.set_title('Model Comparison', color='#58a6ff', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['Logistic\nRegression', 'Random\nForest', 'Gradient\nBoosting'],
                   fontsize=8)
ax.set_ylim(60, 105)
ax.legend(fontsize=8)
ax.grid(axis='y')

# 5b. Confusion Matrix (Best Model)
ax = axes[0, 1]
cm = confusion_matrix(y_test, best['preds'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Predicted\nNot Survived', 'Predicted\nSurvived'],
            yticklabels=['Actual\nNot Survived', 'Actual\nSurvived'],
            linewidths=1, linecolor='#21262d',
            annot_kws={'size': 14, 'weight': 'bold'})
ax.set_title(f'Confusion Matrix — {best_name}', color='#58a6ff', fontweight='bold')

# 5c. ROC Curves
ax = axes[1, 0]
for i, (name, res) in enumerate(results.items()):
    fpr, tpr, _ = roc_curve(y_test, res['proba'])
    ax.plot(fpr, tpr, color=COLORS[i], linewidth=2,
            label=f"{name} (AUC={res['auc']:.3f})")
ax.plot([0,1], [0,1], 'k--', alpha=0.4, linewidth=1)
ax.fill_between([0,1], [0,1], alpha=0.05, color='white')
ax.set_title('ROC Curves — All Models', color='#58a6ff', fontweight='bold')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(fontsize=8)
ax.grid()

# 5d. Feature Importance (Best Model if RF/GB)
ax = axes[1, 1]
if hasattr(best['model'], 'feature_importances_'):
    importances = best['model'].feature_importances_
    feat_imp = pd.Series(importances, index=FEATURES).sort_values(ascending=True)
    colors_bar = [COLORS[0] if v > feat_imp.median() else COLORS[2]
                  for v in feat_imp.values]
    feat_imp.plot(kind='barh', ax=ax, color=colors_bar, edgecolor='none')
    ax.set_title(f'Feature Importance — {best_name}', color='#58a6ff', fontweight='bold')
    ax.set_xlabel('Importance Score')
    ax.grid(axis='x')
else:
    # Logistic regression coefficients
    coef = np.abs(best['model'].coef_[0])
    feat_imp = pd.Series(coef, index=FEATURES).sort_values(ascending=True)
    feat_imp.plot(kind='barh', ax=ax, color=COLORS[0], edgecolor='none')
    ax.set_title(f'Feature Coefficients — {best_name}', color='#58a6ff', fontweight='bold')
    ax.set_xlabel('|Coefficient|')
    ax.grid(axis='x')

plt.tight_layout()
plt.savefig('/home/claude/titanic_model_results.png', dpi=150,
            bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("   ✅ Model evaluation chart saved → titanic_model_results.png")

# ═══════════════════════════════════════════════════════════════
# 6. CLASSIFICATION REPORT
# ═══════════════════════════════════════════════════════════════
print(f"\n📋 Classification Report — {best_name}")
print("-" * 45)
print(classification_report(y_test, best['preds'],
                             target_names=['Not Survived', 'Survived']))

# ═══════════════════════════════════════════════════════════════
# 7. PREDICT ON NEW PASSENGER
# ═══════════════════════════════════════════════════════════════
print("🔮 Prediction on a New Sample Passenger:")
print("-" * 45)
sample = pd.DataFrame([{
    'Pclass': 2, 'Sex_encoded': 0,  # female
    'Age': 25, 'SibSp': 1, 'Parch': 0,
    'Fare': 26, 'Embarked_encoded': 0,
    'FamilySize': 2, 'IsAlone': 0,
    'FareBand': 1, 'AgeBand': 2
}])
pred  = best['model'].predict(sample)[0]
prob  = best['model'].predict_proba(sample)[0]
print(f"   Passenger: Female, Age 25, 2nd Class, Fare £26")
print(f"   Prediction : {'✅ SURVIVED' if pred==1 else '❌ DID NOT SURVIVE'}")
print(f"   Confidence : Survived={prob[1]*100:.1f}%  |  Not Survived={prob[0]*100:.1f}%")

print("\n" + "=" * 60)
print("  ✅ Task 1 Complete! All charts saved.")
print("  📁 Files: titanic_eda.png | titanic_model_results.png")
print("=" * 60)
