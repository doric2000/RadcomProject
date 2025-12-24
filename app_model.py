"""APP-1 application classifier with richer feature set and stronger ensemble.

Rules:
- Train ONLY on: data/APP-1/radcom_app_train.csv
- Test  ONLY on: data/APP-1/radcom_app_test.csv
- Predict on     data/APP-1/radcom_app_val_without_labels.csv (submission)
"""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

from log_setup import configure_logging

warnings.filterwarnings("ignore")
configure_logging()
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class Paths:
    train: str = "data/APP-1/radcom_app_train.csv"
    test: str = "data/APP-1/radcom_app_test.csv"
    val: str = "data/APP-1/radcom_app_val_without_labels.csv"
    model_dir: str = "models"
    submission: str = "submission_app.csv"


PATHS = Paths()
TARGET_COL = "label"
RANDOM_STATE = 42


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
# FEATURE ENGINEERING UTILITIES
# =============================================================================

def _safe_ratio(num: pd.Series, den: pd.Series, eps: float = 1e-6) -> pd.Series:
    return num / (den + eps)


def _spectral_summary(mat: np.ndarray, top_k: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (peak, entropy, energy_ratio) per row using rFFT; safe for short vectors."""
    if mat.size == 0:
        return (
            np.zeros(mat.shape[0]),
            np.zeros(mat.shape[0]),
            np.zeros(mat.shape[0]),
        )

    mag = np.abs(np.fft.rfft(mat, axis=1))
    mag[:, 0] = 0.0  # drop DC component

    peak = np.max(mag, axis=1)

    ps = mag / (mag.sum(axis=1, keepdims=True) + 1e-9)
    entropy = -np.sum(ps * np.log(ps + 1e-9), axis=1)

    k = min(top_k, mag.shape[1])
    if k == 0:
        energy_ratio = np.zeros(mag.shape[0])
    else:
        top = np.partition(mag, -k, axis=1)[:, -k:]
        energy_ratio = top.sum(axis=1) / (mag.sum(axis=1) + 1e-9)

    return peak, entropy, energy_ratio


def _add_sequence_features(df: pd.DataFrame, cols: List[str], prefix: str) -> pd.DataFrame:
    seq = df[cols].fillna(0.0)
    seq_abs = seq.abs()

    df[f"{prefix}_mean"] = seq_abs.mean(axis=1)
    df[f"{prefix}_std"] = seq_abs.std(axis=1)
    df[f"{prefix}_max"] = seq_abs.max(axis=1)
    df[f"{prefix}_nonzero"] = (seq_abs > 0).sum(axis=1)
    df[f"{prefix}_net_flow"] = seq.sum(axis=1)

    peak, entropy, energy = _spectral_summary(seq.values)
    df[f"{prefix}_fft_peak"] = peak
    df[f"{prefix}_fft_entropy"] = entropy
    df[f"{prefix}_fft_energy_top3"] = energy

    return df


def _add_interarrival_features(df: pd.DataFrame) -> pd.DataFrame:
    if {"mean_fwd_inter_arrival_time", "min_fwd_inter_arrival_time", "max_fwd_inter_arrival_time"}.issubset(df.columns):
        df["fwd_burst_factor"] = _safe_ratio(df["max_fwd_inter_arrival_time"], df["mean_fwd_inter_arrival_time"])
        df["fwd_iat_range"] = df["max_fwd_inter_arrival_time"] - df["min_fwd_inter_arrival_time"]
        df["fwd_iat_cv"] = _safe_ratio(df["fwd_iat_range"], df["mean_fwd_inter_arrival_time"])

    if {"mean_bwd_inter_arrival_time", "min_bwd_inter_arrival_time", "max_bwd_inter_arrival_time"}.issubset(df.columns):
        df["bwd_burst_factor"] = _safe_ratio(df["max_bwd_inter_arrival_time"], df["mean_bwd_inter_arrival_time"])
        df["bwd_iat_range"] = df["max_bwd_inter_arrival_time"] - df["min_bwd_inter_arrival_time"]
        df["bwd_iat_cv"] = _safe_ratio(df["bwd_iat_range"], df["mean_bwd_inter_arrival_time"])

    if "silence_windows" in df.columns:
        total_packets = df.get("fwd_packets_amount", 0) + df.get("bwd_packets_amount", 0) + 1e-6
        df["silence_per_packet"] = _safe_ratio(df["silence_windows"], total_packets)

    return df


def _add_flag_densities(df: pd.DataFrame) -> pd.DataFrame:
    total_packets = df.get("fwd_packets_amount", 0) + df.get("bwd_packets_amount", 0) + 1e-6
    for flag_col, out_col in [
        ("PSH_count", "psh_per_packet"),
        ("ACK_count", "ack_per_packet"),
        ("SYN_count", "syn_per_packet"),
        ("FIN_count", "fin_per_packet"),
        ("RST_count", "rst_per_packet"),
    ]:
        if flag_col in df.columns:
            df[out_col] = _safe_ratio(df[flag_col], total_packets)
    return df


def _add_balance_features(df: pd.DataFrame) -> pd.DataFrame:
    df["total_packets"] = df.get("fwd_packets_amount", 0) + df.get("bwd_packets_amount", 0)
    df["total_bytes"] = df.get("fwd_packets_length", 0) + df.get("bwd_packets_length", 0)

    df["download_upload_ratio"] = _safe_ratio(df.get("bwd_packets_length", 0), df.get("fwd_packets_length", 0))
    df["packet_direction_ratio"] = _safe_ratio(df.get("bwd_packets_amount", 0), df.get("fwd_packets_amount", 0))
    df["avg_bytes_per_packet"] = _safe_ratio(df["total_bytes"], df["total_packets"])
    df["log_asymmetry"] = np.abs(np.log(_safe_ratio(df.get("bwd_packets_length", 0) + 1.0, df.get("fwd_packets_length", 0) + 1.0)))

    # Stability of packet sizes
    if {"min_packet_size", "max_packet_size", "mean_packet_size"}.issubset(df.columns):
        df["packet_size_range"] = df["max_packet_size"] - df["min_packet_size"]
        df["packet_size_range_ratio"] = _safe_ratio(df["packet_size_range"], df["mean_packet_size"] + 1e-6)

    # PPS ratio if available
    if {"pps_fwd", "pps_bwd"}.issubset(df.columns):
        df["pps_ratio"] = _safe_ratio(df["pps_bwd"], df["pps_fwd"])

    return df


def _add_protocol_flags(df: pd.DataFrame) -> pd.DataFrame:
    if "Protocol" not in df.columns:
        return df
    p = df["Protocol"].astype(str).str.lower()
    df["is_udp"] = p.eq("udp").astype(int)
    df["is_tcp"] = p.eq("tcp").astype(int)
    return df


def extract_app_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering that mixes statistical, spectral, and balance signals."""
    df = df.copy()

    df = _add_balance_features(df)
    df = _add_interarrival_features(df)
    df = _add_flag_densities(df)
    df = _add_protocol_flags(df)

    # Handshake shape + spectral regularity
    hs_cols = [c for c in df.columns if HANDSHAKE_PREFIX in c]
    if hs_cols:
        df = _add_sequence_features(df, hs_cols, "handshake")

    # Bandwidth (time series) spectral fingerprint
    bw_cols = [c for c in df.columns if c.startswith(BANDWIDTH_PREFIX)]
    if bw_cols:
        df = _add_sequence_features(df, bw_cols, "bw")

    # Beaconing / periodicity signals
    beacon_cols = [c for c in df.columns if c.startswith(BEACON_PREFIX)]
    if beacon_cols:
        df = _add_sequence_features(df, beacon_cols, "beacon")

    # Bytes-per-packet curve
    bpp_cols = [c for c in df.columns if c.startswith(BPP_PREFIX)]
    if bpp_cols:
        df = _add_sequence_features(df, bpp_cols, "bpp")

    return df


def build_feature_matrix(df: pd.DataFrame, target_col: str | None = None):
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
# MODELING
# =============================================================================

def _build_model() -> VotingClassifier:
    rf = RandomForestClassifier(
        n_estimators=600,
        max_depth=None,
        min_samples_leaf=1,
        min_samples_split=2,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
    )

    et = ExtraTreesClassifier(
        n_estimators=500,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    hgb = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_depth=10,
        max_leaf_nodes=64,
        min_samples_leaf=20,
        random_state=RANDOM_STATE,
    )

    return VotingClassifier(
        estimators=[("rf", rf), ("et", et), ("hgb", hgb)],
        voting="soft",
        weights=[2, 2, 1],
        n_jobs=-1,
    )


# =============================================================================
# TRAIN / PREDICT
# =============================================================================

def train_app_model():
    logger.info("=" * 50)
    logger.info("[RUN] TRAINING APPLICATION MODEL (APP-1)")
    logger.info("=" * 50)

    logger.info("-> Loading data...")
    train_df = pd.read_csv(PATHS.train)
    test_df = pd.read_csv(PATHS.test)

    logger.info(f"   Train samples: {len(train_df)}, Test samples: {len(test_df)}")
    logger.info(f"   Unique labels in train: {train_df[TARGET_COL].nunique()}")

    X_train, y_train, features = build_feature_matrix(train_df, target_col=TARGET_COL)
    X_test, y_test, _ = build_feature_matrix(test_df, target_col=TARGET_COL)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    model = _build_model()
    logger.info(f"-> Training ensemble... (features={X_train.shape[1]})")
    model.fit(X_train, y_train_enc)

    logger.info("-> Evaluating on test set...")
    preds = model.predict(X_test)
    acc = accuracy_score(y_test_enc, preds)
    logger.info(f"[RESULT] APPLICATION ACCURACY: {acc:.2%}")
    logger.info("\n" + classification_report(y_test_enc, preds, target_names=le.classes_, zero_division=0))

    logger.info("-> Saving model artifacts...")
    os.makedirs(PATHS.model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(PATHS.model_dir, "app_model.pkl"))
    joblib.dump(le, os.path.join(PATHS.model_dir, "app_label_encoder.pkl"))
    joblib.dump(features, os.path.join(PATHS.model_dir, "app_columns.pkl"))
    meta = {
        "ensemble": "RF+ET+HGB (soft voting)",
        "random_state": RANDOM_STATE,
        "test_accuracy": float(acc),
        "feature_count": int(X_train.shape[1]),
    }
    joblib.dump(meta, os.path.join(PATHS.model_dir, "app_meta.pkl"))
    logger.info("[SUCCESS] Application model saved to models/")

    return model, le, features


def predict_app(df: pd.DataFrame, model=None, le=None, features=None):
    if model is None:
        model = joblib.load(os.path.join(PATHS.model_dir, "app_model.pkl"))
        le = joblib.load(os.path.join(PATHS.model_dir, "app_label_encoder.pkl"))
        features = joblib.load(os.path.join(PATHS.model_dir, "app_columns.pkl"))

    X, _, _ = build_feature_matrix(df, target_col=None)
    X = X.reindex(columns=features, fill_value=0.0)
    preds = model.predict(X)
    return le.inverse_transform(preds)


def generate_submission():
    logger.info("=" * 50)
    logger.info("[RUN] GENERATING SUBMISSION FOR VALIDATION SET (NO FILE OUTPUT)")
    logger.info("=" * 50)

    logger.info(f"-> Loading validation data from {PATHS.val}...")
    val_df = pd.read_csv(PATHS.val)
    logger.info(f"   Validation samples: {len(val_df)}")

    logger.info("-> Running inference...")
    predictions = predict_app(val_df)

    submission = pd.DataFrame({"prediction": predictions})
    logger.info(f"-> Generated predictions. Total: {len(submission)}")
    return submission


if __name__ == "__main__":
    train_app_model()
    generate_submission()