import pandas as pd
import joblib
from celery import Celery
import os
import sys

# Add project root to sys.path so we can import app_model and att_model
sys.path.append(os.getcwd())

from app_model import extract_app_features, build_feature_matrix
from att_model import extract_isolation_features

# Configure Celery to talk to Redis
celery_app = Celery('worker', broker='redis://redis:6379/0', backend='redis://redis:6379/0')

# Global variables for model assets
models = {}

def load_model_assets(task_type):
    """
    Smart loading: load the model, scaler, label encoder, and columns.
    task_type: 'app' or 'att'
    """
    if task_type in models:
        return models[task_type]
    
    base_path = "/models"  # the path mounted in docker-compose
    print(f"Loading {task_type} model assets from {base_path}...")
    
    try:
        clf = joblib.load(os.path.join(base_path, f'{task_type}_model.pkl'))
        
        # Helper to safely load optional files
        def load_safe(filename):
            p = os.path.join(base_path, filename)
            return joblib.load(p) if os.path.exists(p) else None

        scaler = load_safe(f'{task_type}_scaler.pkl')
        le = load_safe(f'{task_type}_label_encoder.pkl')
        cols = load_safe(f'{task_type}_columns.pkl')
        
        models[task_type] = (clf, scaler, le, cols)
        return clf, scaler, le, cols
    except Exception as e:
        print(f"ERROR loading models: {e}")
        raise e

@celery_app.task(name="predict_process")
def predict_process(data_json, task_type):
    """
    Worker task:
    1. Reconstruct DataFrame
    2. Apply Feature Engineering
    3. Predict
    4. Decode Labels
    """
    # 1. Convert JSON back to DataFrame
    print(f"--- [WORKER] RECEIVED PREDICTION REQUEST. MODEL TYPE: {task_type.upper()} ---")
    df = pd.DataFrame(data_json)
    
    # 2. Load assets
    clf, scaler, le, model_columns = load_model_assets(task_type)
    
    # 3. Feature Engineering & Preprocessing
    if task_type == 'app':
        # Re-use logic from app_model.py
        # build_feature_matrix calls extract_app_features internally
        X, _, _ = build_feature_matrix(df, target_col=None)
        
    elif task_type == 'att':
        # Re-use logic from att_model.py
        df_eng = extract_isolation_features(df, q_large=0.90, n_packets=20)
        X = df_eng
    else:
        return ["ERROR: Unknown task type"]

    # 4. Alignment
    # Ensure columns match exactly what the model expects
    if model_columns is not None:
        # Reindex ensures we have the same columns in the same order
        # (fills missing with 0, drops extras)
        X = X.reindex(columns=model_columns, fill_value=0.0)
    
    # 5. Scaling (if applicable)
    if scaler:
        X = scaler.transform(X)
    
    # 6. Predict
    predictions_indices = clf.predict(X)
    
    # 7. Decode Labels (indices -> strings)
    if le:
        predictions_labels = le.inverse_transform(predictions_indices)
        return predictions_labels.tolist()
    
    return predictions_indices.tolist()