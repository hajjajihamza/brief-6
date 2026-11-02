import mlflow
from src.const import PROJECT_ROOT
import os

# MLflow Directories
MLFLOW_DIR = PROJECT_ROOT / "mlflow"

# Artifacts path to absolute path
ARTIFACTS_DIR = MLFLOW_DIR / "artifacts"

# Database path to absolute path
MLFLOW_DB = MLFLOW_DIR / "mlflow.db"

# Create Directories
MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Convert paths to absolute
MLFLOW_DB = MLFLOW_DB.resolve()
ARTIFACTS_DIR = ARTIFACTS_DIR.resolve()

# Create Tracking URI
MLFLOW_TRACKING_URI = "https://nat-exclusive-conditional-cardiff.trycloudflare.com"

# Experiment name
EXPERIMENT_NAME = "neuro_vision_experiment"

# Configure MLflow
os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
os.environ["MLFLOW_EXPERIMENT_NAME"] = EXPERIMENT_NAME

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Create or get the same Experiment
experiment = mlflow.set_experiment(EXPERIMENT_NAME)

print("Experiment:", experiment.name)
print("Experiment ID:", experiment.experiment_id)