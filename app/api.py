from fastapi import FastAPI, UploadFile, File
from celery.result import AsyncResult
from app.worker import celery_app
import pandas as pd
import io

app = FastAPI()

@app.post("/predict/{task_type}")
async def predict_endpoint(task_type: str, file: UploadFile = File(...)):
    # 1. Read the file
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    
    # 2. Convert to JSON
    data_json = df.to_dict(orient='records')
    
    # 3. Send to queue (task name as string)
    # This is the big fix: instead of calling the function, we send a message to the queue
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