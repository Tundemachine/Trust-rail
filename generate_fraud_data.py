"""
Synthetic Mobile Money Transaction Data Generator
--------------------------------------------------
Generates a realistic, labeled dataset of mobile money transactions
(Nigerian-market flavored: amounts, states, KYC tiers) with injected
fraud patterns for training/demoing a fraud detection model.

No external dependencies beyond numpy + pandas (no faker, offline-safe).

Output: fraud_transactions.csv with columns:
    transaction_id, user_id, timestamp, amount, transaction_type,
    recipient_id, recipient_account_age_days, sender_state, device_id,
    is_new_device, sim_change_flag, kyc_tier, sender_account_age_days,
    sender_avg_txn_30d, txn_count_last_1h, txn_count_last_24h,
    time_since_last_txn_min, is_fraud, fraud_type
"""

import numpy as np
import pandas as pd
import random
import uuid
from datetime import datetime, timedelta

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
SEED = 42
N_USERS = 2000
N_TRANSACTIONS = 20000
FRAUD_RATE = 0.015          # ~1.5% of transactions are fraudulent (realistic imbalance)
START_DATE = datetime(2026, 1, 1)
END_DATE = datetime(2026, 6, 30)

NIGERIAN_STATES = [
    "Lagos", "Ogun", "Oyo", "Kano", "Rivers", "Kaduna", "Enugu",
    "Delta", "Anambra", "Ondo", "Edo", "Abuja (FCT)", "Cross River"
]

TXN_TYPES = ["P2P_TRANSFER", "AIRTIME_TOPUP", "BILL_PAYMENT", "MERCHANT_PAYMENT", "CASH_OUT"]
TXN_TYPE_WEIGHTS = [0.35, 0.20, 0.15, 0.20, 0.10]

KYC_TIERS = [1, 2, 3]
KYC_WEIGHTS = [0.45, 0.40, 0.15]   # Tier 1 = lowest verification, most common & riskiest

rng = np.random.default_rng(SEED)
random.seed(SEED)

# ------------------------------------------------------------------
# Step 1: Build a user base with baseline behavioral profiles
# ------------------------------------------------------------------
def build_users(n_users):
    users = []
    for i in range(n_users):
        kyc_tier = rng.choice(KYC_TIERS, p=KYC_WEIGHTS)
        account_age_days = int(rng.integers(1, 1500))
        # Baseline "typical" transaction size varies a lot by user
        avg_txn_size = float(rng.lognormal(mean=8.5, sigma=0.9))  # centered a few thousand naira
        avg_txn_size = min(max(avg_txn_size, 200), 200000)
        home_state = random.choice(NIGERIAN_STATES)
        device_id = f"DEV-{uuid.uuid4().hex[:8]}"
        users.append({
            "user_id": f"U{i:05d}",
            "kyc_tier": int(kyc_tier),
            "account_age_days": account_age_days,
            "avg_txn_size": round(avg_txn_size, 2),
            "home_state": home_state,
            "primary_device_id": device_id,
        })
    return pd.DataFrame(users)

users_df = build_users(N_USERS)

# ------------------------------------------------------------------
# Step 2: Helpers
# ------------------------------------------------------------------
def random_timestamp():
    delta = END_DATE - START_DATE
    random_seconds = rng.integers(0, int(delta.total_seconds()))
    return START_DATE + timedelta(seconds=int(random_seconds))

def new_device_id():
    return f"DEV-{uuid.uuid4().hex[:8]}"

def clip_amount(x):
    return float(np.clip(x, 100, 2_000_000))

# ------------------------------------------------------------------
# Step 3: Generate LEGITIMATE transactions
# ------------------------------------------------------------------
def generate_legit_transaction(user_row, txn_history_count):
    txn_type = rng.choice(TXN_TYPES, p=TXN_TYPE_WEIGHTS)

    # Amount fluctuates around the user's normal baseline...
    amount = clip_amount(rng.lognormal(
        mean=np.log(user_row["avg_txn_size"]), sigma=0.5
    ))

    # ...but ~6% of the time it's a legitimate big-ticket payment (school
    # fees, rent, business restock, family emergency) that looks similar
    # in scale to a fraud transaction. This is what creates real overlap
    # instead of a perfectly separable dataset.
    is_big_legit_payment = rng.random() < 0.06
    if is_big_legit_payment:
        amount = clip_amount(user_row["avg_txn_size"] * rng.uniform(4, 15))

    ts = random_timestamp()

    # Real new-phone/new-SIM rate — people genuinely upgrade devices and
    # switch SIMs; this overlaps with the fraud device-change signal.
    is_new_device = rng.random() < 0.05
    sim_change_flag = rng.random() < 0.015

    # Recipients are usually familiar, but ~8% of the time it's a
    # brand-new contact (new vendor, new landlord, first-time transfer
    # to a friend) — same recipient-age range fraud uses.
    if rng.random() < 0.08:
        recipient_account_age_days = int(rng.integers(0, 10))
    else:
        recipient_account_age_days = int(rng.integers(1, 2000))

    sender_state = user_row["home_state"] if rng.random() > 0.03 else random.choice(NIGERIAN_STATES)

    # Occasional legitimate velocity burst (small business owner paying
    # several staff/suppliers back-to-back) — overlaps with fraud's
    # rapid-fire signal.
    is_velocity_burst = rng.random() < 0.03
    txn_count_1h = int(rng.integers(2, 6)) if is_velocity_burst else int(rng.poisson(0.3))
    txn_count_24h = int(rng.integers(5, 12)) if is_velocity_burst else int(rng.poisson(2))
    time_since_last = float(rng.uniform(1, 30)) if is_velocity_burst else float(rng.exponential(600))

    return {
        "transaction_id": f"T{uuid.uuid4().hex[:10]}",
        "user_id": user_row["user_id"],
        "timestamp": ts,
        "amount": round(amount, 2),
        "transaction_type": txn_type,
        "recipient_id": f"U{rng.integers(0, N_USERS):05d}",
        "recipient_account_age_days": recipient_account_age_days,
        "sender_state": sender_state,
        "device_id": new_device_id() if is_new_device else user_row["primary_device_id"],
        "is_new_device": int(is_new_device),
        "sim_change_flag": int(sim_change_flag),
        "kyc_tier": user_row["kyc_tier"],
        "sender_account_age_days": user_row["account_age_days"],
        "sender_avg_txn_30d": user_row["avg_txn_size"],
        "txn_count_last_1h": txn_count_1h,
        "txn_count_last_24h": txn_count_24h,
        "time_since_last_txn_min": time_since_last,
        "is_fraud": 0,
        "fraud_type": "NONE",
    }

# ------------------------------------------------------------------
# Step 4: Generate FRAUDULENT transactions (4 distinct patterns)
# ------------------------------------------------------------------
def generate_fraud_transaction(user_row):
    fraud_type = random.choice([
        "SIM_SWAP", "ACCOUNT_TAKEOVER", "SMURFING", "MULE_VELOCITY"
    ])
    ts = random_timestamp()
    base = generate_legit_transaction(user_row, 0)  # start from a legit template

    # ~30% of fraud is "stealthy": the fraudster deliberately keeps one or
    # two signals in the normal range to reduce detection risk (e.g. waits
    # before cashing out, moves a smaller amount, uses a recipient that's
    # a few weeks old instead of brand new). This is what real adversarial
    # fraud looks like and is what creates genuine overlap with legit
    # big-ticket/new-device/new-recipient cases above.
    is_stealthy = rng.random() < 0.30

    if fraud_type == "SIM_SWAP":
        # New device + SIM change, immediately followed by a large transfer
        amount_mult = rng.uniform(3, 8) if is_stealthy else rng.uniform(8, 25)
        wait_min = rng.uniform(15, 120) if is_stealthy else rng.uniform(1, 15)
        recipient_age = int(rng.integers(3, 21)) if is_stealthy else int(rng.integers(0, 5))
        base.update({
            "transaction_type": "P2P_TRANSFER",
            "amount": clip_amount(user_row["avg_txn_size"] * amount_mult),
            "device_id": new_device_id(),
            "is_new_device": 1,
            "sim_change_flag": 0 if is_stealthy else 1,  # stealthy: skip flaggable SIM re-registration
            "time_since_last_txn_min": float(wait_min),
            "recipient_account_age_days": recipient_age,
            "txn_count_last_1h": int(rng.integers(0, 2)) if is_stealthy else int(rng.integers(1, 3)),
        })

    elif fraud_type == "ACCOUNT_TAKEOVER":
        # New device login, a burst of small "testing" txns, then a big drain
        amount_mult = rng.uniform(2, 5) if is_stealthy else rng.uniform(5, 15)
        burst_1h = int(rng.integers(1, 4)) if is_stealthy else int(rng.integers(4, 12))
        burst_24h = int(rng.integers(3, 8)) if is_stealthy else int(rng.integers(8, 25))
        wait_min = rng.uniform(10, 60) if is_stealthy else rng.uniform(0.5, 5)
        base.update({
            "transaction_type": random.choice(["P2P_TRANSFER", "CASH_OUT"]),
            "amount": clip_amount(user_row["avg_txn_size"] * amount_mult),
            "device_id": new_device_id(),
            "is_new_device": 1,
            "sim_change_flag": 0,
            "txn_count_last_1h": burst_1h,
            "txn_count_last_24h": burst_24h,
            "time_since_last_txn_min": float(wait_min),
            "recipient_account_age_days": int(rng.integers(5, 45)) if is_stealthy else int(rng.integers(0, 30)),
        })

    elif fraud_type == "SMURFING":
        # Many small transactions just under a reporting threshold (e.g. ~100,000 NGN)
        threshold = 100000
        # Stealthy smurfing spaces transactions further apart and stays
        # further under the threshold to look like normal bill payments
        amount_frac = rng.uniform(0.55, 0.80) if is_stealthy else rng.uniform(0.85, 0.98)
        burst_24h = int(rng.integers(3, 7)) if is_stealthy else int(rng.integers(6, 15))
        gap_min = rng.uniform(60, 240) if is_stealthy else rng.uniform(10, 90)
        base.update({
            "transaction_type": "P2P_TRANSFER",
            "amount": round(threshold * amount_frac, 2),
            "txn_count_last_24h": burst_24h,
            "recipient_account_age_days": int(rng.integers(10, 120)) if is_stealthy else int(rng.integers(0, 60)),
            "time_since_last_txn_min": float(gap_min),
        })

    else:  # MULE_VELOCITY
        # Money-mule pattern: many *different* senders funnel to one new recipient fast
        amount_mult = rng.uniform(1.2, 3) if is_stealthy else rng.uniform(2, 6)
        recipient_age = int(rng.integers(3, 15)) if is_stealthy else int(rng.integers(0, 3))
        burst_1h = int(rng.integers(0, 3)) if is_stealthy else int(rng.integers(2, 6))
        burst_24h = int(rng.integers(2, 8)) if is_stealthy else int(rng.integers(5, 20))
        gap_min = rng.uniform(20, 90) if is_stealthy else rng.uniform(1, 20)
        base.update({
            "transaction_type": "P2P_TRANSFER",
            "amount": clip_amount(user_row["avg_txn_size"] * amount_mult),
            "recipient_account_age_days": recipient_age,
            "txn_count_last_1h": burst_1h,
            "txn_count_last_24h": burst_24h,
            "time_since_last_txn_min": float(gap_min),
        })

    base["timestamp"] = ts
    base["is_fraud"] = 1
    base["fraud_type"] = fraud_type
    base["transaction_id"] = f"T{uuid.uuid4().hex[:10]}"
    return base

# ------------------------------------------------------------------
# Step 5: Assemble full dataset
# ------------------------------------------------------------------
def generate_dataset(n_transactions, fraud_rate):
    n_fraud = int(n_transactions * fraud_rate)
    n_legit = n_transactions - n_fraud

    records = []
    for _ in range(n_legit):
        user_row = users_df.iloc[rng.integers(0, N_USERS)]
        records.append(generate_legit_transaction(user_row, 0))

    # Fraud is concentrated on a subset of "compromised" users (more realistic
    # than spreading it evenly — real fraud clusters on specific accounts)
    n_compromised_users = max(1, n_fraud // 3)
    compromised_users = users_df.sample(n=min(n_compromised_users, N_USERS), random_state=SEED)

    for _ in range(n_fraud):
        user_row = compromised_users.iloc[rng.integers(0, len(compromised_users))]
        records.append(generate_fraud_transaction(user_row))

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)  # shuffle
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

if __name__ == "__main__":
    df = generate_dataset(N_TRANSACTIONS, FRAUD_RATE)

    out_path = "fraud_transactions.csv"
    df.to_csv(out_path, index=False)

    print(f"Generated {len(df):,} transactions -> {out_path}")
    print(f"Fraud transactions: {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.2f}%)")
    print("\nFraud type breakdown:")
    print(df[df['is_fraud'] == 1]['fraud_type'].value_counts())
    print("\nSample rows:")
    print(df.head(5).to_string())
