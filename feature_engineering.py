"""
Feature Engineering: Raw Transactions -> Model-Ready Feature Table
--------------------------------------------------------------------
Takes fraud_transactions.csv (raw, timestamp-ordered transaction log)
and derives the features an XGBoost fraud model would actually use.

Key principle: velocity/behavioral features are RECOMPUTED from the
transaction sequence itself (rolling windows per user, per recipient)
rather than trusting the raw injected counters — this mirrors how a
real feature pipeline works: it only knows what happened *before* the
current transaction, never anything derived from the label.

Input:  fraud_transactions.csv
Output: model_ready_features.csv (one row per transaction, engineered
        features + is_fraud target, ready for train/test split)
"""

import pandas as pd
import numpy as np
from bisect import bisect_left, bisect_right

IN_PATH = "fraud_transactions.csv"
OUT_PATH = "model_ready_features.csv"

# ------------------------------------------------------------------
# Load
# ------------------------------------------------------------------
df = pd.read_csv(IN_PATH, parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)
df["ts_epoch"] = df["timestamp"].astype("int64") // 10**9  # seconds since epoch

# ------------------------------------------------------------------
# Helper: rolling count of PRIOR events within a window, using a
# per-group sorted epoch array + bisect (fast, no leakage from future
# or from the current row itself).
# ------------------------------------------------------------------
def rolling_count_prior(epochs, window_seconds):
    """epochs: sorted list of epoch-seconds for one group (user or recipient).
    Returns array: for each i, count of epochs[j] with epochs[i]-window <= epochs[j] < epochs[i]."""
    counts = np.zeros(len(epochs), dtype=int)
    for i, t in enumerate(epochs):
        lo = bisect_left(epochs, t - window_seconds, 0, i)
        counts[i] = i - lo
    return counts

def time_since_prev(epochs):
    out = np.full(len(epochs), np.nan)
    for i in range(1, len(epochs)):
        out[i] = (epochs[i] - epochs[i - 1]) / 60.0  # minutes
    return out

def expanding_mode_mismatch(values):
    """For each i, is values[i] different from the most common prior value?
    First occurrence -> 0 (no history yet, can't flag a change)."""
    from collections import Counter
    counter = Counter()
    out = np.zeros(len(values), dtype=int)
    for i, v in enumerate(values):
        if counter:
            most_common_val, _ = counter.most_common(1)[0]
            out[i] = int(v != most_common_val)
        counter[v] += 1
    return out

def rolling_amount_stats(epochs, amounts, window_seconds):
    """Mean/std of amounts in the prior window (excluding current txn)."""
    means = np.full(len(amounts), np.nan)
    stds = np.full(len(amounts), np.nan)
    for i, t in enumerate(epochs):
        lo = bisect_left(epochs, t - window_seconds, 0, i)
        window_vals = amounts[lo:i]
        if len(window_vals) >= 2:
            means[i] = np.mean(window_vals)
            stds[i] = np.std(window_vals)
        elif len(window_vals) == 1:
            means[i] = window_vals[0]
            stds[i] = np.nan
    return means, stds

# ------------------------------------------------------------------
# Step 1: Per-USER velocity + behavioral features
# ------------------------------------------------------------------
user_feature_frames = []
for user_id, g in df.groupby("user_id", sort=False):
    g = g.sort_values("ts_epoch")
    epochs = g["ts_epoch"].tolist()
    amounts = g["amount"].to_numpy()

    txn_count_1h = rolling_count_prior(epochs, 3600)
    txn_count_24h = rolling_count_prior(epochs, 86400)
    txn_count_7d = rolling_count_prior(epochs, 7 * 86400)
    time_since_last = time_since_prev(epochs)
    state_change_flag = expanding_mode_mismatch(g["sender_state"].tolist())
    amt_mean_30d, amt_std_30d = rolling_amount_stats(epochs, amounts, 30 * 86400)

    feat = pd.DataFrame({
        "transaction_id": g["transaction_id"].values,
        "txn_count_last_1h": txn_count_1h,
        "txn_count_last_24h": txn_count_24h,
        "txn_count_last_7d": txn_count_7d,
        "time_since_last_txn_min": time_since_last,
        "sender_state_change_flag": state_change_flag,
        "amount_mean_30d_prior": amt_mean_30d,
        "amount_std_30d_prior": amt_std_30d,
    })
    user_feature_frames.append(feat)

user_features = pd.concat(user_feature_frames, ignore_index=True)

# ------------------------------------------------------------------
# Step 2: Per-RECIPIENT features (mule / collusion signals)
# ------------------------------------------------------------------
recipient_feature_frames = []
for recipient_id, g in df.groupby("recipient_id", sort=False):
    g = g.sort_values("ts_epoch")
    epochs = g["ts_epoch"].tolist()
    senders = g["user_id"].tolist()

    recv_count_24h = rolling_count_prior(epochs, 86400)

    # Unique senders to this recipient in the prior 24h window
    unique_senders_24h = np.zeros(len(epochs), dtype=int)
    for i, t in enumerate(epochs):
        lo = bisect_left(epochs, t - 86400, 0, i)
        unique_senders_24h[i] = len(set(senders[lo:i]))

    feat = pd.DataFrame({
        "transaction_id": g["transaction_id"].values,
        "recipient_txn_count_last_24h": recv_count_24h,
        "recipient_unique_senders_last_24h": unique_senders_24h,
    })
    recipient_feature_frames.append(feat)

recipient_features = pd.concat(recipient_feature_frames, ignore_index=True)

# ------------------------------------------------------------------
# Step 3: Merge everything back together
# ------------------------------------------------------------------
model_df = df.merge(user_features, on="transaction_id", suffixes=("", "_recomputed"))
model_df = model_df.merge(recipient_features, on="transaction_id")

# Drop the raw injected velocity columns now that we have real rolling
# versions of them (avoid duplicate/confusing columns)
model_df = model_df.drop(columns=[
    "txn_count_last_1h_recomputed", "txn_count_last_24h_recomputed",
    "time_since_last_txn_min_recomputed",
], errors="ignore")

# ------------------------------------------------------------------
# Step 4: Derived risk features
# ------------------------------------------------------------------
# Amount z-score vs the user's own prior 30-day behavior. Falls back to
# a neutral 0 when there isn't enough history yet (new users).
model_df["amount_zscore_user"] = np.where(
    model_df["amount_std_30d_prior"].notna() & (model_df["amount_std_30d_prior"] > 0),
    (model_df["amount"] - model_df["amount_mean_30d_prior"]) / model_df["amount_std_30d_prior"],
    0.0,
)
model_df["amount_zscore_user"] = model_df["amount_zscore_user"].clip(-10, 10)

model_df["amount_vs_user_avg_ratio"] = (
    model_df["amount"] / model_df["sender_avg_txn_30d"].replace(0, np.nan)
).fillna(1.0)

model_df["is_new_recipient"] = (model_df["recipient_account_age_days"] <= 3).astype(int)
model_df["is_very_new_recipient_and_large_amount"] = (
    (model_df["recipient_account_age_days"] <= 5) & (model_df["amount_zscore_user"] > 3)
).astype(int)

model_df["hour_of_day"] = model_df["timestamp"].dt.hour
model_df["day_of_week"] = model_df["timestamp"].dt.dayofweek
model_df["is_odd_hour"] = model_df["hour_of_day"].isin(range(0, 5)).astype(int)

# Fill remaining NaNs from cold-start (first transaction for a user/recipient)
model_df["time_since_last_txn_min"] = model_df["time_since_last_txn_min"].fillna(999999)
model_df["amount_mean_30d_prior"] = model_df["amount_mean_30d_prior"].fillna(model_df["amount"])
model_df["amount_std_30d_prior"] = model_df["amount_std_30d_prior"].fillna(0)

# ------------------------------------------------------------------
# Step 5: Encode categoricals for XGBoost
# ------------------------------------------------------------------
model_df = pd.get_dummies(model_df, columns=["transaction_type"], prefix="txn_type")

# ------------------------------------------------------------------
# Step 6: Final column selection
# ------------------------------------------------------------------
feature_cols = [
    "transaction_id", "user_id", "timestamp",
    "amount", "amount_zscore_user", "amount_vs_user_avg_ratio",
    "kyc_tier", "sender_account_age_days",
    "recipient_account_age_days", "is_new_recipient",
    "is_very_new_recipient_and_large_amount",
    "recipient_txn_count_last_24h", "recipient_unique_senders_last_24h",
    "is_new_device", "sim_change_flag", "sender_state_change_flag",
    "txn_count_last_1h", "txn_count_last_24h", "txn_count_last_7d",
    "time_since_last_txn_min", "hour_of_day", "day_of_week", "is_odd_hour",
] + [c for c in model_df.columns if c.startswith("txn_type_")] + [
    "is_fraud", "fraud_type",
]

final_df = model_df[feature_cols].sort_values("timestamp").reset_index(drop=True)
final_df.to_csv(OUT_PATH, index=False)

# ------------------------------------------------------------------
# Sanity check summary
# ------------------------------------------------------------------
print(f"Model-ready feature table saved -> {OUT_PATH}")
print(f"Rows: {len(final_df):,} | Columns: {final_df.shape[1]}")
print(f"Fraud rate: {final_df['is_fraud'].mean()*100:.2f}%\n")

print("Mean feature values: fraud vs. legit (signal check)")
compare_cols = [
    "amount_zscore_user", "amount_vs_user_avg_ratio", "is_new_device",
    "sim_change_flag", "recipient_account_age_days",
    "recipient_unique_senders_last_24h", "txn_count_last_1h",
    "txn_count_last_24h", "time_since_last_txn_min",
]
summary = final_df.groupby("is_fraud")[compare_cols].mean().T
summary.columns = ["legit", "fraud"]
print(summary.round(2))
