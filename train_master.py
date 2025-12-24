import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import warnings
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings('ignore')

import logging
from log_setup import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

# Configuration for paths
PATHS = {
    'att': {'train': 'data/attribution/radcom_att_train.csv', 'test': 'data/attribution/radcom_att_test.csv'},
    'app': {'train': 'data/APP-1/radcom_app_train.csv', 'test': 'data/APP-1/radcom_app_test.csv'}
}

def extract_traffic_dna(df):
    """
    Advanced Feature Engineering (The 'DNA' of traffic).
    Extracts ratios, handshakes, and burst patterns.
    """
    df = df.copy()
    
    # 1. Ratios (Critical for encrypted traffic)
    total_packets = df['fwd_packets_amount'] + df['bwd_packets_amount'] + 1e-6
    df['download_upload_ratio'] = df['bwd_packets_length'] / (df['fwd_packets_length'] + 1e-6)
    df['packet_direction_ratio'] = df['bwd_packets_amount'] / (df['fwd_packets_amount'] + 1e-6)
    
    # 2. Averages
    total_bytes = df['fwd_packets_length'] + df['bwd_packets_length']
    df['avg_bytes_per_packet'] = total_bytes / total_packets

    # 3. Handshake DNA (First packets signature)
    handshake_cols = [c for c in df.columns if 'first_packet_sizes' in c]
    if len(handshake_cols) > 0:
        df['handshake_std'] = df[handshake_cols].std(axis=1)
        df['handshake_net_flow'] = df[handshake_cols].sum(axis=1)
        df['handshake_complexity'] = df[handshake_cols].apply(lambda x: len(np.unique(x)), axis=1)

    # 4. Timing / Burstiness
    if 'max_fwd_inter_arrival_time' in df.columns:
        df['fwd_burst_factor'] = df['max_fwd_inter_arrival_time'] / (df['mean_fwd_inter_arrival_time'] + 1e-6)

    return df

def prepare_data(df_train, df_test, target_col):
    """
    Common data preparation pipeline.
    """
    # Feature Engineering
    train_eng = extract_traffic_dna(df_train)
    test_eng = extract_traffic_dna(df_test)
    
    # Aggressive Cleanup: Drop IDs but KEEP Protocol (One-Hot Encoded later)
    drop_cols = ['Source_IP', 'Source_port', 'Destination_IP', 'Destination_port', 'Timestamp', target_col]
    
    X = train_eng.drop(columns=drop_cols, errors='ignore')
    y = train_eng[target_col]
    
    X_test = test_eng.drop(columns=drop_cols, errors='ignore')
    y_test = test_eng[target_col]
    
    # One-Hot Encoding for Protocol (Crucial for Attribution!)
    X = pd.get_dummies(X, columns=['Protocol'], drop_first=True)
    X_test = pd.get_dummies(X_test, columns=['Protocol'], drop_first=True)
    
    # Align Columns (Ensure test set has exactly same columns as train)
    X_test = X_test.reindex(columns=X.columns, fill_value=0)
    
    # Fill NAs
    X = X.fillna(0)
    X_test = X_test.fillna(0)
    
    return X, y, X_test, y_test

def train_attribution_task():
    logger.info(f"\n{'='*40}\n[RUN] TRAINING ATTRIBUTION (The Specialist)\n{'='*40}")
    
    # Load Data
    try:
        train = pd.read_csv(PATHS['att']['train'])
        test = pd.read_csv(PATHS['att']['test'])
    except FileNotFoundError:
        logger.error("[ERROR] Files not found. Check PATHS config.")
        return
        return

    X, y, X_test, y_test = prepare_data(train, test, 'attribution')
    
    # Scaling is mandatory for KNN/SVM in the ensemble
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)
    
    # Label Encoding
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    y_test_enc = le.transform(y_test)
    
    # --- THE CHAMPION ENSEMBLE ---
    # XGBoost: The Gradient Boosting Powerhouse
    clf_xgb = xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=4, n_jobs=-1, random_state=42)
    # Random Forest: The Robust Average
    clf_rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    # Extra Trees: More randomness, good for small data
    clf_et = ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    # KNN: Distance based (Good for few-shot attribution)
    clf_knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
    
    ensemble = VotingClassifier(
        estimators=[('xgb', clf_xgb), ('rf', clf_rf), ('et', clf_et), ('knn', clf_knn)],
        voting='soft',
        weights=[2, 1, 1, 2] # Tuned weights
    )
    
    ensemble.fit(X_scaled, y_enc)
    
    # Metrics
    preds = ensemble.predict(X_test_scaled)
    acc = accuracy_score(y_test_enc, preds)
    logger.info(f"[RESULT] ATTRIBUTION ACCURACY: {acc:.2%}")
    
    # Save Artifacts
    os.makedirs('models', exist_ok=True)
    joblib.dump(ensemble, 'models/att_model.joblib')
    joblib.dump(scaler, 'models/att_scaler.joblib') # Don't forget scaler!
    joblib.dump(X.columns, 'models/att_cols.joblib')
    joblib.dump(le, 'models/att_le.joblib')

def train_application_task():
    logger.info(f"\n{'='*40}\n[RUN] TRAINING APPLICATION (The Beast)\n{'='*40}")
    
    try:
        train = pd.read_csv(PATHS['app']['train'])
        test = pd.read_csv(PATHS['app']['test'])
    except FileNotFoundError:
        logger.error("[ERROR] Files not found. Check PATHS config.")
        return
        return

    X, y, X_test, y_test = prepare_data(train, test, 'label')
    
    # No Scaling needed for Trees, but good practice
    # No KNN here! It fails on high dimensionality with sparse data.
    
    # Label Encoding
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    y_test_enc = le.transform(y_test)
    
    # --- THE ROBUST ENSEMBLE (Tree-Only) ---
    # Using only Trees because they handle high-dim/low-sample data better than KNN
    clf_rf = RandomForestClassifier(n_estimators=500, class_weight='balanced', random_state=42, n_jobs=-1)
    clf_xgb = xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, n_jobs=-1, random_state=42)
    
    ensemble = VotingClassifier(
        estimators=[('rf', clf_rf), ('xgb', clf_xgb)],
        voting='soft'
    )
    
    ensemble.fit(X, y_enc)
    
    preds = ensemble.predict(X_test)
    acc = accuracy_score(y_test_enc, preds)
    logger.info(f"[RESULT] APPLICATION ACCURACY: {acc:.2%}")
    
    # Save Artifacts
    joblib.dump(ensemble, 'models/app_model.joblib')
    joblib.dump(X.columns, 'models/app_cols.joblib')
    joblib.dump(le, 'models/app_le.joblib')

if __name__ == "__main__":
    train_attribution_task()
    train_application_task()