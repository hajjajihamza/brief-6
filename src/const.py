from pathlib import Path
import kagglehub

# ==================== Project Root ====================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ==================== Configuration du dataset ====================

# Noms des classes de détection (ID de chaque classe → Nom de la classe)
DATASET_PATH = Path(kagglehub.dataset_download("ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes"))

CLASS_NAMES = {
    0: "Glioma",
    1: "Meningioma",
    2: "No Tumor",
    3: "Pituitary",
}

# Extensions des fichiers images acceptées par le dataset
EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]

# Graine aléatoire pour garantir la reproductibilité des résultats
RANDOM_KEY = 26

# Dimensions des images utilisées pour l'entraînement (640 × 640 pixels)
SIZE_IMAGE = 640

# Nombre d'images traitées simultanément pendant l'entraînement
BATCH_SIZE = 16

# Nombre maximal de passages sur l'ensemble du dataset
EPOCHS = 50

# Nombre d'époques sans amélioration avant l'arrêt anticipé (Early Stopping)
PATIENCE = 10


# ==================== Chemins du dataset ====================

# Chemin vers le dossier principal contenant le dataset YOLO
YOLO_DATASET_PATH = PROJECT_ROOT / 'dataset_yolo'

# Chemin vers le fichier YAML contenant la configuration du dataset
DATA_YAML = YOLO_DATASET_PATH / "data.yaml"

# ==================== register model ====================

REGISTER_MODEL_NAME = 'neurovision_yolo'