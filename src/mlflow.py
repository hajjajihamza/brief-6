import mlflow
from ultralytics import YOLO
from mlflow import MlflowClient
from .const import REGISTER_MODEL_NAME
from pathlib import Path

# set model alias
def set_model_alias(name, version:int):   
    client = mlflow.MlflowClient()

    client.set_registered_model_alias(
        name=name,
        alias="champion",
        version=version
    )


# Charger le modèle YOLO depuis MLflow Model Registry en utilisant un alias.
def load_champion_model():
    client = MlflowClient()

    # Récupérer le modèle champion
    champion = client.get_model_version_by_alias(
        name=REGISTER_MODEL_NAME,
        alias="champion"
    )

    # Télécharger les fichiers du modèle enregistré
    model_dir = mlflow.artifacts.download_artifacts(
        artifact_uri=champion.source
    )

    # Chercher le fichier .pt
    weights_files = list(Path(model_dir).rglob("*.pt"))

    if not weights_files:
        raise FileNotFoundError(
            f"Aucun fichier .pt trouvé dans : {model_dir}"
        )

    # Charger le modèle avec Ultralytics YOLO
    return YOLO(str(weights_files[0]))