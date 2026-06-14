"""
E-COMMERCE 360° — ML Churn Prediction
Trains 3 models, compares them, picks the best,
explains it with SHAP, and saves churn_model.pkl.

Outputs:
  models/churn_model.pkl
  plots/model_comparison.png
  plots/shap_summary.png
  plots/roc_curves.png
  data/model_metrics.csv
  data/predictions.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib, os, warnings

from sklearn.model_selection   import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing     import StandardScaler, LabelEncoder
from sklearn.linear_model      import LogisticRegression
from sklearn.ensemble          import RandomForestClassifier
from sklearn.metrics           import (accuracy_score, precision_score, recall_score,
                                       f1_score, roc_auc_score, roc_curve,
                                       classification_report, confusion_matrix)
from sklearn.pipeline          import Pipeline
from xgboost                   import XGBClassifier
from imblearn.over_sampling    import SMOTE
import shap

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

#  Paths
BASE   = os.path.dirname(os.path.abspath(__file__))
DATA   = os.path.join(BASE, "data")
PLOTS  = os.path.join(BASE, "plots")
MODELS = os.path.join(BASE, "models")
for p in [PLOTS, MODELS]:
    os.makedirs(p, exist_ok=True)

# STEP 1 — LOAD & PREPARE FEATURES
print("\n" + "="*60)
print("  STEP 1 — Load RFM features")
print("="*60)

rfm = pd.read_csv(os.path.join(DATA, "rfm_features.csv"))
print(f"Shape: {rfm.shape}")
print(rfm.head(3))

# Encode state as a label-encoded feature
le_state = LabelEncoder()
rfm["state_enc"] = le_state.fit_transform(rfm["state"].fillna("UNKNOWN"))

# Feature columns
FEATURES = [
    "recency",          # days since last order
    "frequency",        # number of distinct orders
    "monetary",         # total spend
    "avg_order_value",  # avg spend per order
    "avg_qty",          # avg items per order
    "total_qty",        # total items bought
    "num_categories",   # category breadth
    "r_score",          # RFM recency score
    "f_score",          # RFM frequency score
    "m_score",          # RFM monetary score
    "state_enc",        # encoded state
]

TARGET = "churned"

X = rfm[FEATURES].copy()
y = rfm[TARGET].copy()

print(f"\nClass distribution:\n{y.value_counts()}")
print(f"Churn rate: {y.mean()*100:.1f}%")

# STEP 2 — SPLIT + SMOTE

print("\n" + "="*60)
print("  STEP 2 — Train/Test Split + SMOTE")
print("="*60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"After SMOTE → {y_train_sm.value_counts().to_dict()}")

# Scale features (for LR)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_sm)
X_test_sc  = scaler.transform(X_test)

# STEP 3 — TRAIN 3 MODELS

print("\n" + "="*60)
print("  STEP 3 — Training Models")
print("="*60)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=8,
                                                  random_state=42, n_jobs=-1),
    "XGBoost":             XGBClassifier(n_estimators=200, max_depth=6,
                                         learning_rate=0.05, subsample=0.8,
                                         colsample_bytree=0.8, eval_metric="logloss",
                                         random_state=42, n_jobs=-1),
}

results = {}
fitted  = {}

for name, model in models.items():
    print(f"\n  Training {name}...")

    if name == "Logistic Regression":
        model.fit(X_train_sc, y_train_sm)
        y_pred  = model.predict(X_test_sc)
        y_proba = model.predict_proba(X_test_sc)[:, 1]
    else:
        model.fit(X_train_sm, y_train_sm)
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy" : round(accuracy_score(y_test, y_pred),  4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall"   : round(recall_score(y_test, y_pred,    zero_division=0), 4),
        "f1"       : round(f1_score(y_test, y_pred,        zero_division=0), 4),
        "roc_auc"  : round(roc_auc_score(y_test, y_proba),  4),
    }
    results[name] = metrics
    fitted[name]  = (model, y_proba)

    print(f"    Accuracy={metrics['accuracy']}  "
          f"F1={metrics['f1']}  "
          f"ROC-AUC={metrics['roc_auc']}")
    print(classification_report(y_test, y_pred, target_names=["Active","Churned"]))

# STEP 4 — MODEL COMPARISON CHART

print("\n  Generating model comparison chart...")

metrics_df = pd.DataFrame(results).T
metrics_df.to_csv(os.path.join(DATA, "model_metrics.csv"))

fig, axes = plt.subplots(1, 5, figsize=(16, 4))
metric_names = ["accuracy","precision","recall","f1","roc_auc"]
colors = ["#4C72B0","#DD8452","#55A868"]

for i, metric in enumerate(metric_names):
    vals = [results[m][metric] for m in models.keys()]
    bars = axes[i].bar(list(models.keys()), vals, color=colors, edgecolor="white")
    axes[i].set_title(metric.upper(), fontsize=10, fontweight="bold")
    axes[i].set_ylim(0, 1.1)
    axes[i].set_xticklabels(list(models.keys()), rotation=25, ha="right", fontsize=7)
    for bar, v in zip(bars, vals):
        axes[i].text(bar.get_x()+bar.get_width()/2, v+0.02,
                     f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")

fig.suptitle("Model Comparison — Churn Prediction", fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(PLOTS, "model_comparison.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  ✓  model_comparison.png")

# STEP 5 — ROC CURVES

fig, ax = plt.subplots(figsize=(7, 5))
for (name, (model, y_proba)), color in zip(fitted.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = results[name]["roc_auc"]
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, linewidth=2)

ax.plot([0,1],[0,1], "k--", linewidth=1)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — Churn Prediction", fontsize=13, fontweight="bold")
ax.legend()
fig.savefig(os.path.join(PLOTS, "roc_curves.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  ✓  roc_curves.png")

# STEP 6 — PICK BEST MODEL & SHAP

best_name = max(results, key=lambda m: results[m]["roc_auc"])
print(f"\n  🏆  Best model: {best_name}  (ROC-AUC={results[best_name]['roc_auc']})")

best_model = fitted[best_name][0]

# SHAP (use tree explainer for RF/XGB, linear for LR)
print("  Computing SHAP values (may take ~30s)...")
if best_name == "Logistic Regression":
    explainer  = shap.LinearExplainer(best_model, scaler.transform(X_train_sm))
    shap_values= explainer.shap_values(X_test_sc)
else:
    explainer  = shap.TreeExplainer(best_model)
    shap_values= explainer.shap_values(X_test)
    # XGBoost returns 3D array for binary; take positive class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

# SHAP Summary plot
fig, ax = plt.subplots(figsize=(9, 5))
shap.summary_plot(
    shap_values,
    X_test if best_name != "Logistic Regression" else X_test_sc,
    feature_names=FEATURES,
    plot_type="bar",
    show=False,
    color="#4C72B0"
)
plt.title(f"SHAP Feature Importance ({best_name})", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "shap_summary.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  ✓  shap_summary.png")

# SHAP Beeswarm
fig, ax = plt.subplots(figsize=(9, 5))
shap.summary_plot(
    shap_values,
    X_test if best_name != "Logistic Regression" else X_test_sc,
    feature_names=FEATURES,
    show=False,
)
plt.title(f"SHAP Beeswarm — Why model predicts churn ({best_name})", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "shap_beeswarm.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  ✓  shap_beeswarm.png")

# STEP 7 — SAVE MODEL + PREDICTIONS

artifact = {
    "model"     : best_model,
    "scaler"    : scaler,
    "features"  : FEATURES,
    "le_state"  : le_state,
    "model_name": best_name,
    "metrics"   : results[best_name],
    "explainer" : explainer,
}
joblib.dump(artifact, os.path.join(MODELS, "churn_model.pkl"))
print(f"\n  ✓  churn_model.pkl  saved to models/")

# Predictions on test set
pred_df = X_test.copy()
pred_df["actual_churn"]       = y_test.values
pred_df["predicted_churn"]    = fitted[best_name][0].predict(
    X_test_sc if best_name=="Logistic Regression" else X_test
)
pred_df["churn_probability"]  = fitted[best_name][1]
pred_df.to_csv(os.path.join(DATA, "predictions.csv"), index=False)

print("\n✅  All models, plots, and artifacts saved.")

