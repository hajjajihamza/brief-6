from pathlib import Path
import mlflow
from ultralytics import YOLO
from const import DATA_YAML, RANDOM_KEY, REGISTER_MODEL_NAME, SIZE_IMAGE
import cv2
import math
import matplotlib.pyplot as plt
from visualization import show_image

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
    with mlflow.start_run(run_name=run_dir.name):
        # Enregistrer les paramètres
        mlflow.log_params(params)
        
        # Charger le modèle
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
            
            # Run dir
            project=str(run_dir.parent),
            name=run_dir.name,
            exist_ok=True
        )
        
        # Charger le meilleur checkpoint
        best_model_path = str(run_dir / "weights" / "best.pt")
        best_model = YOLO(best_model_path)
        
         # Évaluation sur test data
        metrics = best_model.val(
            data=str(DATA_YAML),
            split="test",
            imgsz=image_size,
            batch=batch,
            plots=True
        )
        
        # Enregistrer les métriques
        mlflow.log_metrics({
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "mAP50": float(metrics.box.map50),
            "mAP50_95": float(metrics.box.map),
        })
        
        # Enregistrer le métrique mAP50-95 par classe
        for class_id, class_name in best_model.names.items():
             mlflow.log_metric(
                f"mAP50_95_{class_name}",
                float(metrics.box.maps[class_id])
            )
        
        # Enregistrer le modèle comme Artifact
        mlflow.log_artifact(
            best_model_path,
            artifact_path="model"
        )
        
        # Enregistrer les résultats d'entraînement
        results_csv = run_dir / "results.csv"
        if results_csv.exists():
            mlflow.log_artifact(
                str(results_csv),
                artifact_path="training_results"
            )
            
        # Enregistrer les graphiques de validation
        confusion_matrix = Path(metrics.save_dir) / "confusion_matrix.png"
        if confusion_matrix.exists():
            mlflow.log_artifact(
                str(confusion_matrix),
                artifact_path="evaluation"
            )
            
        # Register model in MLflow Model Registry
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model/best.pt"

        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=REGISTER_MODEL_NAME
        )    
        
        # Enregistrer les Tags
        mlflow.set_tags({
            "project": "NeuroVision AI",
            "framework": "Ultralytics YOLOv8",
            "model_type": "Object Detection",
        })

        print("Run MLflow terminé avec succès !")
        print("Run ID :", mlflow.active_run().info.run_id)
        print("Model Name:", registered_model.name)
        print("Model Version:", registered_model.version)
    
    
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