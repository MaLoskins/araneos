from celery import Celery
from app.config import get_settings

settings = get_settings()

celery_app = Celery(
    "araneos",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    worker_concurrency=1,  # GPU is the bottleneck
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)

celery_app.autodiscover_tasks(["app.tasks"])
