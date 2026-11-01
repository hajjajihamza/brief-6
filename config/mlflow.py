from pathlib import Path
import mlflow
from src.const import PROJECT_ROOT

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
TRACKING_URI = "http://127.0.0.1:5000"

# Experiment name
EXPERIMENT_NAME = "neuro_vision_experiment"

# Configure MLflow Client
mlflow.set_tracking_uri(TRACKING_URI)

# Create or get Experiment
mlflow.set_experiment(EXPERIMENT_NAME)