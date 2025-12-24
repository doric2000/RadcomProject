import pandas as pd
import joblib
from celery import Celery
import os

# הגדרת ה-Celery שידבר עם Redis
# שים לב: הכתובת 'redis://redis:6379/0' מתאימה לשם השירות ב-docker-compose
celery_app = Celery('worker', broker='redis://redis:6379/0', backend='redis://redis:6379/0')

# משתנים גלובליים לטעינת המודלים (כדי שלא נטען אותם בכל בקשה מחדש)
models = {}

def load_model_assets(task_type):
    """
    טעינה חכמה: טוען את המודל, הסקיילר והעמודות רק אם הם לא בזיכרון.
    task_type: 'app' או 'att'
    """
    if task_type in models:
        return models[task_type]
    
    base_path = "/models"  # הנתיב שמופה ב-docker-compose
    print(f"Loading {task_type} model assets from {base_path}...")
    
    try:
        clf = joblib.load(os.path.join(base_path, f'{task_type}_model.pkl'))
        scaler = joblib.load(os.path.join(base_path, f'{task_type}_scaler.pkl'))
        cols = joblib.load(os.path.join(base_path, f'{task_type}_columns.pkl'))
        
        models[task_type] = (clf, scaler, cols)
        return clf, scaler, cols
    except Exception as e:
        print(f"ERROR loading models: {e}")
        raise e

@celery_app.task(name="predict_process")
def predict_process(data_json, task_type):
    """
    זו הפונקציה שנקראת ע"י ה-Worker.
    היא מקבלת את הדאטה (כ-JSON), מעבדת אותו ומחזירה תחזית.
    """
    # 1. המרת ה-JSON חזרה ל-DataFrame
    df = pd.DataFrame(data_json)
    
    # 2. טעינת המודלים הרלוונטיים
    clf, scaler, model_columns = load_model_assets(task_type)
    
    # 3. ניקוי (אותו תהליך כמו באימון)
    drop_cols = ['Source_IP', 'Source_port', 'Destination_IP', 'Destination_port', 'Timestamp']
    # מורידים רק מה שקיים
    existing_drop_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=existing_drop_cols)
    
    # 4. טיפול בקטגוריות (One Hot Encoding)
    X = pd.get_dummies(X)
    
    # --- הצעד הקריטי: יישור עמודות ---
    # מוודאים שיש לנו בדיוק את אותן עמודות כמו באימון
    X = X.reindex(columns=model_columns, fill_value=0)
    
    # 5. נרמול
    X_scaled = scaler.transform(X)
    
    # 6. חיזוי
    predictions = clf.predict(X_scaled)
    
    # מחזירים את התוצאה כרשימה (כדי שיהיה אפשר להעביר ב-JSON)
    return predictions.tolist()