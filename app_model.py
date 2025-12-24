"""Application Classification Model (APP-1 Challenge)

Goal: small + easy to modify (similar style to att_model.py)

Rules:
- Train ONLY on: data/APP-1/radcom_app_train.csv
- Test  ONLY on: data/APP-1/radcom_app_test.csv
- Predict on     data/APP-1/radcom_app_val_without_labels.csv (submission)
"""

import os
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

import logging
from log_setup import configure_logging

warnings.filterwarnings("ignore")
configure_logging()
logger = logging.getLogger(__name__)


# =============================================================================
# GLOBALS (easy knobs)
# =============================================================================

TRAIN_PATH = "data/APP-1/radcom_app_train.csv"
TEST_PATH = "data/APP-1/radcom_app_test.csv"
VAL_PATH = "data/APP-1/radcom_app_val_without_labels.csv"
TARGET_COL = "label"

MODEL_DIR = "models"
SUBMISSION_PATH = "submission_app.csv"

RANDOM_STATE = 42

# Model hyperparameters (selected because RF performed best here)
RF_PARAMS = {
    "n_estimators": 800,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

# Feature knobs
HANDSHAKE_PREFIX = "first_packet_sizes"
BANDWIDTH_PREFIX = "bandwidth_"
BEACON_PREFIX = "beaconning_"
BPP_PREFIX = "bpp_"

DROP_COLS = {
    "Source_IP",
    "Source_port",
    "Destination_IP",
    "Destination_port",
    "Timestamp",
    "Protocol",
    TARGET_COL,
}


# =============================================================================
# FEATURES
# =============================================================================

def extract_app_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering focused on being robust + readable."""
    df = df.copy()

    # Basic totals
    if {"fwd_packets_amount", "bwd_packets_amount"}.issubset(df.columns):
        total_packets = df["fwd_packets_amount"] + df["bwd_packets_amount"] + 1e-6
    else:
        total_packets = 1.0

    if {"fwd_packets_length", "bwd_packets_length"}.issubset(df.columns):
        total_bytes = df["fwd_packets_length"] + df["bwd_packets_length"]
        df["download_upload_ratio"] = df["bwd_packets_length"] / (df["fwd_packets_length"] + 1e-6)
        df["avg_bytes_per_packet"] = total_bytes / total_packets
        df["log_asymmetry"] = np.abs(
            np.log((df["bwd_packets_length"] + 1.0) / (df["fwd_packets_length"] + 1.0))
        )

    # Direction ratio
    if {"fwd_packets_amount", "bwd_packets_amount"}.issubset(df.columns):
        df["packet_direction_ratio"] = df["bwd_packets_amount"] / (df["fwd_packets_amount"] + 1e-6)

    # Handshake DNA (first packet sizes)
    hs_cols = [c for c in df.columns if HANDSHAKE_PREFIX in c]
    if hs_cols:
        hs = df[hs_cols].fillna(0.0)
        hs_abs = hs.abs()

        df["handshake_mean"] = hs_abs.mean(axis=1)
        df["handshake_std"] = hs_abs.std(axis=1)
        df["handshake_max"] = hs_abs.max(axis=1)
        df["handshake_net_flow"] = hs.sum(axis=1)
        df["handshake_nonzero"] = (hs_abs > 0).sum(axis=1)

        def count_flips(row: pd.Series) -> int:
            signs = np.sign(row.values)
            signs = signs[signs != 0]
            return int(np.sum(signs[1:] != signs[:-1])) if signs.size > 1 else 0

        df["handshake_flips"] = hs.apply(count_flips, axis=1)

    # Timing / burstiness
    if {"max_fwd_inter_arrival_time", "mean_fwd_inter_arrival_time"}.issubset(df.columns):
        df["fwd_burst_factor"] = df["max_fwd_inter_arrival_time"] / (df["mean_fwd_inter_arrival_time"] + 1e-6)
    if {"max_bwd_inter_arrival_time", "mean_bwd_inter_arrival_time"}.issubset(df.columns):
        df["bwd_burst_factor"] = df["max_bwd_inter_arrival_time"] / (df["mean_bwd_inter_arrival_time"] + 1e-6)
    if "silence_windows" in df.columns:
        df["silence_per_packet"] = df["silence_windows"] / total_packets

    # Bandwidth fingerprint
    bw_cols = [c for c in df.columns if c.startswith(BANDWIDTH_PREFIX)]
    if bw_cols:
        bw = df[bw_cols].fillna(0.0)
        df["bw_mean"] = bw.mean(axis=1)
        df["bw_std"] = bw.std(axis=1)
        df["bw_max"] = bw.max(axis=1)
        df["bw_nonzero_count"] = (bw > 0).sum(axis=1)

    # Beaconning / regularity
    beacon_cols = [c for c in df.columns if c.startswith(BEACON_PREFIX)]
    if beacon_cols:
        b = df[beacon_cols].fillna(0.0)
        df["beacon_sum"] = b.sum(axis=1)
        df["beacon_nonzero"] = (b > 0).sum(axis=1)

    # BPP
    bpp_cols = [c for c in df.columns if c.startswith(BPP_PREFIX)]
    if bpp_cols:
        bpp = df[bpp_cols].fillna(0.0)
        df["bpp_mean"] = bpp.mean(axis=1)
        df["bpp_std"] = bpp.std(axis=1)
        df["bpp_max"] = bpp.max(axis=1)

    # TCP flags ratios
    for flag_col, out_col in [
        ("PSH_count", "psh_per_packet"),
        ("ACK_count", "ack_per_packet"),
        ("SYN_count", "syn_per_packet"),
    ]:
        if flag_col in df.columns:
            df[out_col] = df[flag_col] / total_packets

    # Protocol encoding
    if "Protocol" in df.columns:
        p = df["Protocol"].astype(str).str.lower()
        df["is_udp"] = p.eq("udp").astype(int)
        df["is_tcp"] = p.eq("tcp").astype(int)

    return df


def build_feature_matrix(df: pd.DataFrame, target_col: str | None = None):
    """Return X and (optional) y. Keeps column handling in one place."""
    df_eng = extract_app_features(df)

    drop = set(DROP_COLS)
    if target_col is None:
        drop.discard(TARGET_COL)
    else:
        drop.add(target_col)

    features = [c for c in df_eng.columns if c not in drop]
    X = df_eng[features].fillna(0.0)

    if target_col is None:
        return X, None, features

    y = df_eng[target_col].astype(str)
    return X, y, features


# =============================================================================
# TRAIN / PREDICT
# =============================================================================

def train_app_model():
    logger.info("=" * 50)
    logger.info("[RUN] TRAINING APPLICATION MODEL (APP-1)")
    logger.info("=" * 50)

    logger.info("-> Loading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    logger.info(f"   Train samples: {len(train_df)}, Test samples: {len(test_df)}")
    logger.info(f"   Unique labels in train: {train_df[TARGET_COL].nunique()}")

    X_train, y_train, features = build_feature_matrix(train_df, target_col=TARGET_COL)
    X_test, y_test, _ = build_feature_matrix(test_df, target_col=TARGET_COL)

    # Align test columns to train
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    logger.info(f"-> Training RandomForest... (features={X_train.shape[1]})")
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train_enc)

    logger.info("-> Evaluating on test set...")
    preds = model.predict(X_test)
    acc = accuracy_score(y_test_enc, preds)
    logger.info(f"[RESULT] APPLICATION ACCURACY: {acc:.2%}")
    logger.info("\n" + classification_report(y_test_enc, preds, target_names=le.classes_, zero_division=0))

    logger.info("-> Saving model artifacts...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, "app_model.pkl"))
    joblib.dump(le, os.path.join(MODEL_DIR, "app_label_encoder.pkl"))
    joblib.dump(features, os.path.join(MODEL_DIR, "app_columns.pkl"))
    joblib.dump({"rf_params": RF_PARAMS, "test_accuracy": float(acc)}, os.path.join(MODEL_DIR, "app_meta.pkl"))
    logger.info("[SUCCESS] Application model saved to models/")

    return model, le, features


def predict_app(df: pd.DataFrame, model=None, le=None, features=None):
    if model is None:
        model = joblib.load(os.path.join(MODEL_DIR, "app_model.pkl"))
        le = joblib.load(os.path.join(MODEL_DIR, "app_label_encoder.pkl"))
        features = joblib.load(os.path.join(MODEL_DIR, "app_columns.pkl"))

    X, _, _ = build_feature_matrix(df, target_col=None)
    X = X.reindex(columns=features, fill_value=0.0)
    preds = model.predict(X)
    return le.inverse_transform(preds)


def generate_submission(output_path: str = SUBMISSION_PATH):
    logger.info("=" * 50)
    logger.info("[RUN] GENERATING SUBMISSION FOR VALIDATION SET")
    logger.info("=" * 50)

    logger.info(f"-> Loading validation data from {VAL_PATH}...")
    val_df = pd.read_csv(VAL_PATH)
    logger.info(f"   Validation samples: {len(val_df)}")

    logger.info("-> Running inference...")
    predictions = predict_app(val_df)

    submission = pd.DataFrame({"prediction": predictions})
    submission.to_csv(output_path, index=False)
    logger.info(f"[SUCCESS] Submission saved to {output_path}")
    logger.info(f"   Total predictions: {len(submission)}")
    return submission


if __name__ == "__main__":
    train_app_model()
    generate_submission(SUBMISSION_PATH)
    