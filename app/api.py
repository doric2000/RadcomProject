from fastapi import FastAPI, UploadFile, File
from celery.result import AsyncResult
from worker import celery_app  # <--- שים לב: אנחנו מייבאים רק את האפליקציה, לא את המשימה!
import pandas as pd
import io

app = FastAPI()

@app.post("/predict/{task_type}")
async def predict_endpoint(task_type: str, file: UploadFile = File(...)):
    # 1. קריאת הקובץ
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    
    # 2. המרה ל-JSON
    data_json = df.to_dict(orient='records')
    
    # 3. שליחה לתור (לפי שם המשימה במרכאות)
    # זה התיקון הגדול! במקום לקרוא לפונקציה, אנחנו שולחים הודעה לתור
    task = celery_app.send_task('predict_process', args=[data_json, task_type])
    
    return {"task_id": task.id}

@app.get("/result/{task_id}")
async def get_result(task_id: str):
    task_result = AsyncResult(task_id, app=celery_app)
    
    if task_result.state == 'PENDING':
        return {"status": "PENDING"}
    elif task_result.state == 'SUCCESS':
        return {"status": "SUCCESS", "result": task_result.result}
    elif task_result.state == 'FAILURE':
        return {"status": "FAILURE", "error": str(task_result.info)}
    else:
        return {"status": task_result.state}