from datetime import datetime
from airflow.decorators import dag, task
from src.yolo_dataset import validate_yolo_dataset
from src.model import train_model

# start mlflow
import config.mlflow


@dag(
    dag_id="neurovision_pipeline",
    start_date=datetime(2026, 11, 2),
    schedule=None,
    catchup=False,
    tags=["neurovision", "yolo", "mlflow"]
)
def neurovision_pipeline():
    
    # Task 1: Verify dataset
    @task
    def prepare_dataset():
        validate_yolo_dataset()

    # Task 2: Train et  Evaluate model YOLO 
    @task
    def training_and_evaluate():
        train_model()

    # Execute tasks in order
    dataset = prepare_dataset()

    model = training_and_evaluate()

    dataset >> model


neurovision_pipeline()