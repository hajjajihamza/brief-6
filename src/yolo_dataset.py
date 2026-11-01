import shutil
from pathlib import Path
import yaml
from ultralytics.data.utils import check_det_dataset
from collections import Counter
from const import YOLO_DATASET_PATH, EXTENSIONS, CLASS_NAMES, DATA_YAML

            
# Préparer la structure du dataset pour YOLOv8
def prepare_yolo_dataset(splits):
    for split_name, samples in splits.items():
        # Create folder of splite exampel train , val, test
        images_dir = YOLO_DATASET_PATH / "images" / split_name
        labels_dir = YOLO_DATASET_PATH / "labels" / split_name

        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        for sample in samples:
            image_path = Path(sample["image_path"])
            label_path = Path(sample["label_path"])

            # Nom unique pour éviter les collisions
            unique_name = f"{sample['class']}_{image_path.stem}"

            # Conserver l'extension originale
            new_image_path = images_dir / f"{unique_name}{image_path.suffix.lower()}"
            new_label_path = labels_dir / f"{unique_name}.txt"

            # Move files to folders of splits
            shutil.copy2(image_path, new_image_path)
            shutil.copy2(label_path, new_label_path)

        print(
            f"{split_name}: {len(samples)} samples copiés "
            f"vers {images_dir}"
        )


# Vérifier l’intégrité du dataset YOLO
def validate_yolo_dataset():
    for split in ["train", "val", "test"]:
        images_dir = YOLO_DATASET_PATH / "images" / split
        labels_dir = YOLO_DATASET_PATH / "labels" / split

        if not images_dir.is_dir() or not labels_dir.is_dir():
            raise FileNotFoundError(f"Dossiers introuvables pour : {split}")

        image_count = sum(
            p.is_file() and p.suffix.lower() in EXTENSIONS
            for p in images_dir.iterdir()
        )

        label_count = len(list(labels_dir.glob("*.txt")))

        print(f"{split.upper()} : Images = {image_count} | Labels = {label_count}")

    data = check_det_dataset(str(DATA_YAML))

    print("\nDataset YAML validé par Ultralytics.")
    print("Classes :", data["names"])
    print("Nombre de classes :", data["nc"])


# Créer le fichier data.yaml
def create_data_yaml():
    # stractur of file yaml
    data_yaml = {
        "path": str(DATA_YAML.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": CLASS_NAMES,
        "nc": len(CLASS_NAMES),
    }

    # create and write in file yaml
    with open(DATA_YAML, "w", encoding="utf-8") as file:
        yaml.safe_dump(
            data_yaml,
            file,
            sort_keys=False,
            allow_unicode=True,
        )

    print(f"Fichier créé : {DATA_YAML.resolve()}")
    
    
# Analyser la distribution des classes dans les Bounding Boxes
def analyze_yolo_annotations():
    for split in ("train", "val", "test"):
        labels_dir = YOLO_DATASET_PATH / "labels" / split

        box_counts = Counter()
        images_per_class = Counter()

        for label_path in labels_dir.glob("*.txt"):
            seen_classes = set()

            for line in label_path.read_text(encoding="utf-8").splitlines():
                values = line.split()

                class_id = int(values[0])

                box_counts[class_id] += 1
                seen_classes.add(class_id)

            images_per_class.update(seen_classes)

        # rapport
        print(f"\n--- {split.upper()} ---")

        for class_id, name in CLASS_NAMES.items():
            print(
                f"{name:<12} | "
                f"Bounding Boxes: {box_counts[class_id]:5} | "
                f"Images avec cette classe: {images_per_class[class_id]:5}"
            )

