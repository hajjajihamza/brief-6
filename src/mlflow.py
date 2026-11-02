import mlflow
from .const import REGISTER_MODEL_NAME

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
    return mlflow.pyfunc.load_model(
        f"models:/{REGISTER_MODEL_NAME}@champion"
    )
