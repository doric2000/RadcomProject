import pandas as pd
import joblib
from celery import Celery
import os

# Configure Celery to talk to Redis
# Note: the URL 'redis://redis:6379/0' matches the service name in docker-compose
celery_app = Celery('worker', broker='redis://redis:6379/0', backend='redis://redis:6379/0')

# Global variables for model assets (to avoid reloading per request)
models = {}

def load_model_assets(task_type):
    """
    Smart loading: load the model, scaler (if exists), columns, and label encoder only if not already in memory.
    task_type: 'app' or 'att'
    """
    if task_type in models:
        return models[task_type]

    base_path = "/models"  # the path mounted in docker-compose
    print(f"Loading {task_type} model assets from {base_path}...")

    try:
        clf = joblib.load(os.path.join(base_path, f'{task_type}_model.pkl'))
        scaler_path = os.path.join(base_path, f'{task_type}_scaler.pkl')
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
        else:
            scaler = None
        cols = joblib.load(os.path.join(base_path, f'{task_type}_columns.pkl'))
        # Try to load label encoder if it exists
        le_path = os.path.join(base_path, f'{task_type}_label_encoder.pkl')
        if os.path.exists(le_path):
            label_encoder = joblib.load(le_path)
        else:
            label_encoder = None
        models[task_type] = (clf, scaler, cols, label_encoder)
        return clf, scaler, cols, label_encoder
    except Exception as e:
        print(f"ERROR loading models: {e}")
        raise e

@celery_app.task(name="predict_process")
def predict_process(data_json, task_type):
    """
    This function is called by the Worker.
    It receives the data (as JSON), processes it and returns predictions.
    """
    # 1. Convert JSON back to a DataFrame
    df = pd.DataFrame(data_json)

    # 2. Load relevant models
    clf, scaler, model_columns, label_encoder = load_model_assets(task_type)

    # 3. Cleaning (same process as training)
    drop_cols = ['Source_IP', 'Source_port', 'Destination_IP', 'Destination_port', 'Timestamp']
    # drop only columns that exist
    existing_drop_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=existing_drop_cols)

    # 4. Handle categorical features (One-Hot Encoding)
    X = pd.get_dummies(X)

    # --- Critical step: align columns ---
    # ensure we have exactly the same columns as in training
    X = X.reindex(columns=model_columns, fill_value=0)

    # 5. Scaling (only if scaler exists)
    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X

    # 6. Prediction
    predictions = clf.predict(X_scaled)

    # If label encoder exists, convert predictions to original labels
    if label_encoder is not None:
        predictions = label_encoder.inverse_transform(predictions)

    # return result as a list (so it can be JSON serializable)
    return predictions.tolist()