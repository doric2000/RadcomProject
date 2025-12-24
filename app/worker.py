from celery import Celery
import os

# Celery configuration
celery_app = Celery(
    'worker',
    broker='redis://redis:6379/0',
    backend='redis://redis:6379/0',
    include=['tasks']  # <-- added: tells Celery to also look for tasks in tasks.py
)

celery_app.conf.update(
    result_expires=3600,
)