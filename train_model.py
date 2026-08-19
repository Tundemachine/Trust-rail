"""
Train the Fraud Detection Model
---------------------------------
Trains a gradient-boosted tree classifier on the engineered feature
table and evaluates it with metrics that actually matter for fraud
(precision-recall, precision@top-K, false positive rate) rather than
plain accuracy, which is meaningless on a ~1.5%-positive dataset.

Uses XGBoost if available; falls back to sklearn's
HistGradientBoostingClassifier (same family: histogram-based gradient
boosted trees) if XGBoost isn't installed in this environment. On your
own machine, `pip install xgboost` and this will use it automatically.

Split strategy: TIME-BASED, not random. We train on the earlier ~80%
of transactions and test on the most recent ~20% — this mirrors how
the model will actually be used (predict the future from the past)
and avoids leakage from shuffling fraud/legit pairs across the split.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, roc_auc_score,
    confusion_matrix, precision_score, recall_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IN_PATH = "model_ready_features.csv"

# ------------------------------------------------------------------
# Load + prep
# ------------------------------------------------------------------
df = pd.read_csv(IN_PATH, parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

drop_cols = ["transaction_id", "user_id", "timestamp", "fraud_type", "is_fraud"]
feature_cols = [c for c in df.columns if c not in drop_cols]

X = df[feature_cols]
y = df["is_fraud"]

# ------------------------------------------------------------------
# Time-based train/test split (80/20)
# ------------------------------------------------------------------
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Train: {len(X_train):,} rows ({y_train.sum()} fraud, {y_train.mean()*100:.2f}%)")
print(f"Test:  {len(X_test):,} rows ({y_test.sum()} fraud, {y_test.mean()*100:.2f}%)\n")

# ------------------------------------------------------------------
# Model: XGBoost if available, else sklearn HistGradientBoosting
# ------------------------------------------------------------------
model_name = None
try:
    from xgboost import XGBClassifier
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
    )
    model_name = "XGBoost"
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    model = HistGradientBoostingClassifier(
        max_iter=300,
        max_depth=5,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=42,
    )
    model_name = "HistGradientBoostingClassifier (sklearn fallback — xgboost not installed in this sandbox)"

print(f"Model: {model_name}\n")
model.fit(X_train, y_train)

# ------------------------------------------------------------------
# Predictions
# ------------------------------------------------------------------
y_proba = model.predict_proba(X_test)[:, 1]

# ------------------------------------------------------------------
# Metric 1: Average Precision (area under PR curve) — the honest
# summary metric for imbalanced fraud data. ROC-AUC is included too
# but shouldn't be the headline number.
# ------------------------------------------------------------------
ap_score = average_precision_score(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)
print(f"Average Precision (PR-AUC): {ap_score:.3f}")
print(f"ROC-AUC (reference only):   {roc_auc:.3f}\n")

# ------------------------------------------------------------------
# Metric 2: Precision @ top-K% — "if analysts can only review the
# top K% highest-risk transactions, how many frauds do we actually
# catch?" This is the metric that maps to real operational capacity.
# ------------------------------------------------------------------
print("Precision & recall @ top-K% flagged (operational capacity view):")
test_df = X_test.copy()
test_df["y_true"] = y_test.values
test_df["y_proba"] = y_proba
test_df = test_df.sort_values("y_proba", ascending=False).reset_index(drop=True)

total_fraud = y_test.sum()
for k_pct in [0.5, 1, 2, 5, 10]:
    k = max(1, int(len(test_df) * k_pct / 100))
    top_k = test_df.iloc[:k]
    caught = top_k["y_true"].sum()
    precision_at_k = caught / k
    recall_at_k = caught / total_fraud if total_fraud > 0 else 0
    print(f"  Top {k_pct:>4}% ({k:>4} txns): precision={precision_at_k:.2%}, "
          f"recall={recall_at_k:.2%}  ({caught}/{total_fraud} frauds caught)")

# ------------------------------------------------------------------
# Metric 3: Confusion matrix at a fixed operating threshold
# ------------------------------------------------------------------
threshold = 0.5
y_pred = (y_proba >= threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

print(f"\nAt threshold={threshold}:")
print(f"  Precision: {precision_score(y_test, y_pred, zero_division=0):.2%}")
print(f"  Recall:    {recall_score(y_test, y_pred, zero_division=0):.2%}")
print(f"  False Positive Rate: {fpr:.4%}  ({fp} legit txns wrongly flagged out of {fp+tn})")
print(f"  Confusion matrix -> TN:{tn} FP:{fp} FN:{fn} TP:{tp}")

# ------------------------------------------------------------------
# Feature importance
# ------------------------------------------------------------------
# HistGradientBoostingClassifier (our XGBoost fallback) doesn't expose
# feature_importances_ the way tree-ensemble models like XGBoost/
# RandomForest do, so we use permutation importance instead: shuffle
# each feature column and measure how much the model's score drops.
# This works for ANY model type and is arguably more honest anyway
# (it measures actual predictive contribution, not just tree splits).
if hasattr(model, "feature_importances_"):
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=False)
else:
    from sklearn.inspection import permutation_importance
    perm_result = permutation_importance(
        model, X_test, y_test,
        scoring="average_precision",
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )
    importances = pd.Series(perm_result.importances_mean, index=feature_cols)
    importances = importances.sort_values(ascending=False)

print("\nTop 10 features by importance:")
print(importances.head(10).to_string())

# ------------------------------------------------------------------
# Plots: PR curve + feature importance
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

precision, recall, _ = precision_recall_curve(y_test, y_proba)
axes[0].plot(recall, precision, color="#c0392b", linewidth=2)
axes[0].set_xlabel("Recall")
axes[0].set_ylabel("Precision")
axes[0].set_title(f"Precision-Recall Curve (AP = {ap_score:.3f})")
axes[0].grid(alpha=0.3)

top_features = importances.head(12).iloc[::-1]
axes[1].barh(top_features.index, top_features.values, color="#2c3e50")
importance_kind = "Split-based" if hasattr(model, "feature_importances_") else "Permutation (AP drop)"
axes[1].set_title(f"Top 12 Feature Importances ({importance_kind})")
axes[1].set_xlabel("Importance")

plt.tight_layout()
plt.savefig("model_evaluation.png", dpi=150)
print("\nSaved evaluation chart -> model_evaluation.png")

# ------------------------------------------------------------------
# Persist predictions for downstream dashboard/demo use
# ------------------------------------------------------------------
output_df = df.iloc[split_idx:].copy()
output_df["fraud_probability"] = y_proba
output_df["predicted_fraud"] = y_pred
output_df.to_csv("test_predictions.csv", index=False)
print("Saved test-set predictions -> test_predictions.csv")
