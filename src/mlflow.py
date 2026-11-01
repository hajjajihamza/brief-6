import mlflow
from ultralytics import YOLO
from const import REGISTER_MODEL_NAME

# set model alias
def set_model_alias(name, version:int):   
    client = mlflow.MlflowClient()

    client.set_registered_model_alias(
        name=name,
        alias="champion",
        version=version
    )


# Charger le modèle YOLO depuis MLflow Model Registry en utilisant un alias.
def load_model():
    client = mlflow.MlflowClient()
    alias = 'champion'
    
    # Récupérer le modèle associé à l'alias
    model_version = client.get_model_version_by_alias(
        name=REGISTER_MODEL_NAME,
        alias=alias
    )

    # Récupérer le chemin du modèle dans MLflow
    model_path = model_version.source

    # Charger le modèle YOLO
    model = YOLO(model_path)

    return model
