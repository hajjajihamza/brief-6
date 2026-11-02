from pathlib import Path
import mlflow
from ultralytics import YOLO
from .const import DATA_YAML, RANDOM_KEY, REGISTER_MODEL_NAME, SIZE_IMAGE
import cv2
import math
import matplotlib.pyplot as plt
from .visualization import show_image
from mlflow.pyfunc import PythonModel


class YOLOModel(PythonModel):

    def load_context(self, context):
        self.model = YOLO(context.artifacts["weights"])

    def predict(self, context, model_input, params=None):
        return self.model.predict(
            source=model_input,
            conf=0.25,
            save=False
        )

# run train of model and return result
def train_model(
        base_model: str,
        epochs:int,
        image_size:int,
        batch:int,
        patience:int,
        run_dir:Path
    ):
    
    # Paramètres du modèle
    params = {
        "base_model": base_model,
        "epochs": epochs,
        "image_size": image_size,
        "batch": batch,
        "patience": patience,
    }
    
    # Start MLflow Run
    with mlflow.start_run(run_name=run_dir.name) as run:
        run_id = run.info.run_id
        
        # Enregistrer les paramètres
        mlflow.log_params(params)
        
        # Log tags
        mlflow.set_tags({
            "project": "NeuroVision AI",
            "framework": "Ultralytics YOLOv8",
            "model_type": "Object Detection",
        })
        
        # Load pretrained model
        model = YOLO(base_model)
        
        # Entraînement
        model.train(
            data=str(DATA_YAML), 
            epochs=epochs,
            imgsz=image_size,
            batch=batch,
            seed=RANDOM_KEY,
            patience=patience,

            # Data Augmentation
            fliplr=0.5,
            flipud=0.0,
            degrees=5.0,
            translate=0.05,
            scale=0.10,
            mosaic=0.0,
            mixup=0.0,
            
            # Output directory
            project=str(run_dir.parent),
            name=run_dir.name,
            exist_ok=True
        )
        
        # Best model path
        best_model_path = run_dir / "weights" / "best.pt"

        if not best_model_path.is_file():
            raise FileNotFoundError(
                f"Best model not found: {best_model_path}"
            )
            
        # Load best model
        best_model = YOLO(str(best_model_path))
        
        # Evaluate on test dataset
        evaluate_model(
            model=best_model,
            image_size=image_size,
            batch=batch
        )
        
        # Log YOLO as MLflow PyFunc Model
        model_info = mlflow.pyfunc.log_model(
            name=run_dir.name,
            python_model=YOLOModel(),
            artifacts={
                "weights": str(best_model_path)
            }
        )
        
        # Register model
        registered_model = mlflow.register_model(
            model_uri=model_info.model_uri,
            name=REGISTER_MODEL_NAME
        )
        
        print("Run MLflow terminé avec succès !")
        print("Run ID :", run_id)
        print("Model Name:", registered_model.name)
        print("Model Version:", registered_model.version)
    
    

# Évaluer le modèle sur le test dataset
def evaluate_model(
    model: YOLO,
    image_size: int,
    batch: int
):
    # Évaluation
    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=image_size,
        batch=batch,
        plots=True
    )

    # Extraire les métriques globales
    evaluation_metrics = {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
    }

    # Enregistrer les métriques globales dans MLflow
    mlflow.log_metrics(evaluation_metrics)

    # Enregistrer le mAP50-95 par classe
    for class_id, class_name in model.names.items():
        mlflow.log_metric(
            f"mAP50_95_{class_name}",
            float(metrics.box.maps[class_id])
        )

    # Enregistrer la matrice de confusion
    confusion_matrix = Path(metrics.save_dir) / "confusion_matrix.png"

    if confusion_matrix.exists():
        mlflow.log_artifact(
            str(confusion_matrix),
            artifact_path="evaluation"
        )
        
    print('Évaluation sur le test dataset terminé avec succès !')


    
# Predict et visualiser les prédictions    
def predict(model, sources:list, run_dir:Path):
    # Faire les prédictions sur les source
    results = model.predict(
        source=sources,
        imgsz=SIZE_IMAGE,
        conf=0.25,
        save=True,
        project=str(run_dir.parent),
        name=run_dir.name,
        exist_ok=True
    )

    n_rows = math.ceil(len(sources) / 4)
    # Afficher les images avec les prédictions
    _, axes = plt.subplots(n_rows, 4, figsize=(16, 10))
    axes = axes.flatten()

    for ax, result in zip(axes, results):
        # plot() retourne une image en BGR
        annotated_image = result.plot()

        # Convertir BGR vers RGB pour Matplotlib
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        show_image(ax, annotated_image, Path(result.path).name)

    plt.tight_layout()
    plt.show()

    print(f"{len(results)} images affichées.")
    print("Résultats sauvegardés dans :")
    print(str(run_dir))