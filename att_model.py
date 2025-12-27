import pandas as pd
import numpy as np
import joblib
import os
import warnings

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings("ignore")

import logging
from log_setup import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

# Paths configuration
TRAIN_PATH = "data/attribution/radcom_att_train.csv"
TEST_PATH = "data/attribution/radcom_att_test.csv"
VAL_PATH = "data/attribution/radcom__att_val_without_labels.csv"
MODEL_DIR = "models"
RESULT_DIR = "result"
SUBMISSION_FILE = "submission_att.csv"


def extract_isolation_features(df, q_large=0.90, n_packets=20):
    """
    Your winning logic + minimal robust behavior features:
    - Relative "large" packets (per-flow quantile) -> no hardcoded threshold
    - Isolation: neighbors around large packets
    - Directionality: ping-pong via direction flips
    - Simple burst signal: max_large_run + large_density
    """
    df = df.copy()

    # 1) Protocol
    if "Protocol" in df.columns:
        df["is_udp"] = df["Protocol"].astype(str).str.lower().eq("udp").astype(int)

    # 2) First N packets
    handshake_cols = [c for c in df.columns if "first_packet_sizes" in c][:n_packets]
    raw_packets = df[handshake_cols].fillna(0.0)
    abs_packets = raw_packets.abs()

    # --------------------------
    # A) RELATIVE "large" packets
    # --------------------------
    def row_quantile_nonzero(row):
        v = row.values
        v = v[v > 0]
        return float(np.quantile(v, q_large)) if v.size else 0.0

    large_thr = abs_packets.apply(row_quantile_nonzero, axis=1).values
    is_large = abs_packets.values >= large_thr[:, None]

    # --------------------------
    # B) ISOLATION (neighbor analysis)
    # --------------------------
    left_neighbor = np.roll(abs_packets.values, 1, axis=1)
    right_neighbor = np.roll(abs_packets.values, -1, axis=1)
    left_neighbor[:, 0] = 0.0
    right_neighbor[:, -1] = 0.0
    neighbor_avg = (left_neighbor + right_neighbor) / 2.0

    large_neighbors = np.where(is_large, neighbor_avg, np.nan)
    df["max_neighbor_size"] = np.nan_to_num(np.nanmax(large_neighbors, axis=1), nan=0.0)

    flow_max = abs_packets.max(axis=1).values
    df["max_neighbor_rel"] = df["max_neighbor_size"] / (flow_max + 1e-6)

    large_pkt = np.where(is_large, abs_packets.values, np.nan)
    ratio = large_neighbors / (large_pkt + 1e-6)
    max_ratio = np.nan_to_num(np.nanmax(ratio, axis=1), nan=0.0)
    df["isolation_score"] = 1.0 - np.clip(max_ratio, 0.0, 1.0)

    # --------------------------
    # C) BURSTINESS (minimal + safe)
    # --------------------------
    def max_run(mask_1d):
        best = 0
        cur = 0
        for v in mask_1d:
            if v:
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        return best

    df["max_large_run"] = [max_run(is_large[i]) for i in range(is_large.shape[0])]
    df["large_density"] = is_large.mean(axis=1)

    # --------------------------
    # D) DIRECTIONALITY (ping-pong)
    # --------------------------
    def sign_flips(x):
        s = np.sign(x)
        s = s[s != 0]  # ignore zeros
        return int(np.sum(s[1:] != s[:-1])) if s.size > 1 else 0

    raw_np = raw_packets.values
    df["direction_flips"] = [sign_flips(raw_np[i]) for i in range(raw_np.shape[0])]
    nonzero_cnt = (raw_np != 0).sum(axis=1)
    df["flip_rate"] = df["direction_flips"] / np.maximum(nonzero_cnt - 1, 1)

    # --------------------------
    # E) Your existing robust features
    # --------------------------
    if "bwd_packets_length" in df.columns and "fwd_packets_length" in df.columns:
        down = df["bwd_packets_length"] + 1
        up = df["fwd_packets_length"] + 1
        df["log_symmetry"] = np.abs(np.log(down / up))

    if "max_fwd_inter_arrival_time" in df.columns and "mean_fwd_inter_arrival_time" in df.columns:
        df["silence_ratio"] = df["max_fwd_inter_arrival_time"] / (df["mean_fwd_inter_arrival_time"] + 1e-6)

    def p90(row):
        v = row.values
        v = np.abs(v[v != 0])
        return float(np.percentile(v, 90)) if v.size else 0.0

    df["p90_size"] = raw_packets.apply(p90, axis=1)

    return df


def predict_att(df, model=None, scaler=None, le=None, features=None):
    """Make predictions on new data using trained attribution model."""
    if model is None:
        model = joblib.load(os.path.join(MODEL_DIR, "att_model.pkl"))
        scaler = joblib.load(os.path.join(MODEL_DIR, "att_scaler.pkl"))
        le = joblib.load(os.path.join(MODEL_DIR, "att_label_encoder.pkl"))
        features = joblib.load(os.path.join(MODEL_DIR, "att_columns.pkl"))
    
    # Apply same feature engineering
    df_eng = extract_isolation_features(df, q_large=0.90, n_packets=20)
    
    # Select same features and fill missing
    X = df_eng[features].fillna(0.0)
    X = X.reindex(columns=features, fill_value=0.0)
    
    # Scale and predict
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    
    # Convert back to original labels
    return le.inverse_transform(preds)


def generate_att_submission(output_dir=RESULT_DIR, output_file=SUBMISSION_FILE):
    """Generate predictions for validation set and save to result directory."""
    logger.info("=" * 50)
    logger.info("[RUN] GENERATING ATTRIBUTION SUBMISSION FOR VALIDATION SET")
    logger.info("=" * 50)
    
    logger.info(f"-> Loading validation data from {VAL_PATH}...")
    val_df = pd.read_csv(VAL_PATH)
    logger.info(f"   Validation samples: {len(val_df)}")
    
    logger.info("-> Running inference...")
    predictions = predict_att(val_df)
    
    # Create result directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Add predictions to original dataframe
    submission = val_df.copy()
    submission["prediction"] = predictions
    
    output_path = os.path.join(output_dir, output_file)
    submission.to_csv(output_path, index=False)
    
    logger.info(f"[SUCCESS] Attribution submission saved to {output_path}")
    logger.info(f"   Total predictions: {len(submission)}")
    logger.info(f"   Original columns + prediction column: {len(submission.columns)}")
    logger.info(f"   Unique predictions: {submission['prediction'].nunique()}")
    
    return submission


def train_att_isolation():
    logger.info("-> Loading data...")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    logger.info("-> Extracting features (your winning logic)...")
    train_eng = extract_isolation_features(train, q_large=0.90, n_packets=20)
    test_eng = extract_isolation_features(test, q_large=0.90, n_packets=20)

    drop_cols = [
        "Source_IP", "Source_port", "Destination_IP", "Destination_port",
        "Timestamp", "Protocol", "attribution"
    ]
    features = [c for c in train_eng.columns if c not in drop_cols]

    X = train_eng[features].fillna(0.0)
    y = train_eng["attribution"].astype(str)

    X_test = test_eng[features].fillna(0.0)
    y_test = test_eng["attribution"].astype(str)
    X_test = X_test.reindex(columns=X.columns, fill_value=0.0)

    # Scaling (needed for KNN)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    y_test_enc = le.transform(y_test)

    logger.info("-> Training ensemble...")

    # Keep KNN, but reduce its influence (it overfits hard on tiny train sets)
    clf_knn = KNeighborsClassifier(n_neighbors=1, metric="manhattan")

    # RF: main learner, mild regularization
    clf_rf = RandomForestClassifier(
        n_estimators=400,          # stable sweet spot
        class_weight="balanced",
        random_state=42,
        min_samples_leaf=2,
        min_samples_split=4,
        n_jobs=1                   # stable + avoids overhead
    )

    # IMPORTANT CHANGE: RF dominates the vote -> better generalization
    ensemble = VotingClassifier(
        estimators=[("knn", clf_knn), ("rf", clf_rf)],
        voting="soft",
        weights=[1, 6]
    )

    ensemble.fit(X_scaled, y_enc)

    preds = ensemble.predict(X_test_scaled)
    acc = accuracy_score(y_test_enc, preds)

    logger.info(f"[RESULT] RESULT: {acc:.2%}")
    logger.info('\n' + classification_report(y_test_enc, preds, target_names=le.classes_))

    # Save assets
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(ensemble, os.path.join(MODEL_DIR, "att_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "att_scaler.pkl"))
    joblib.dump(le, os.path.join(MODEL_DIR, "att_label_encoder.pkl"))
    joblib.dump(features, os.path.join(MODEL_DIR, "att_columns.pkl"))
    logger.info("[SUCCESS] Assets saved.")
    
    return ensemble, scaler, le, features


if __name__ == "__main__":
    train_att_isolation()
    generate_att_submission()
