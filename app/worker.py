from celery import Celery
import os

# הגדרת ה-Celery
celery_app = Celery(
    'worker',
    broker='redis://redis:6379/0',
    backend='redis://redis:6379/0',
    include=['tasks']  # <--- הוספנו את זה! אומר ל-Celery: "תחפש משימות גם בקובץ tasks.py"
)

celery_app.conf.update(
    result_expires=3600,
)