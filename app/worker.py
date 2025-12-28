from celery import Celery
import os

# Celery configuration
celery_app = Celery(
    'worker',
    broker='redis://redis:6379/0',
    backend='redis://redis:6379/0',
    include=['app.tasks']  # tells Celery to also look for tasks in app/tasks.py
)

celery_app.conf.update(
    result_expires=3600,
)