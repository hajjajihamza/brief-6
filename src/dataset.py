from const import EXTENSIONS, CLASS_NAMES
import math
from collections import Counter, defaultdict
from hashlib import sha256
import cv2

# Récupérer les images et labels
def get_image_and_label_files(class_path):
    images_path = class_path / "images"
    labels_path = class_path / "labels"

    if not images_path.is_dir() or not labels_path.is_dir():
        return [], []

    # GEt all paths of images validas in class folder
    image_files = [
        p for p in images_path.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    ]

    # Get all files text in class folder
    label_files = list(labels_path.glob("*.txt"))

    return image_files, label_files


# Vérifier les annotations YOLO
def validate_yolo_annotation(line):
    # Splite values
    values = line.split()

    # check if the vlaues is 5
    if len(values) != 5:
        raise ValueError("Il faut exactement 5 valeurs")

    # get and validate class id is int and valid
    class_id = int(values[0])

    if str(class_id) != values[0]:
        raise ValueError("class_id doit être un entier")
    
    if not 0 <= class_id < len(CLASS_NAMES):
            raise ValueError(f"class_id invalide : {class_id}")

    # get x , y center and width and hieght
    xc, yc, w, h = map(float, values[1:])

    # validate values of annotation
    if not all(map(math.isfinite, (xc, yc, w, h))):
        raise ValueError("Valeur NaN ou infinie")

    if not (0 <= xc <= 1 and 0 <= yc <= 1):
        raise ValueError("Centre hors de l'image")

    if not (0 < w <= 1 and 0 < h <= 1):
        raise ValueError("Width ou height invalide")

    if (
        xc - w / 2 < -1e-6
        or yc - h / 2 < -1e-6
        or xc + w / 2 > 1 + 1e-6
        or yc + h / 2 > 1 + 1e-6
    ):
        raise ValueError("Bounding Box dépasse l'image")


# Valider un fichier label
def validate_label_file(label_path):
    errors = []
    annotations = []

    # Read lines from file text
    lines = label_path.read_text(encoding="utf-8").splitlines()

    # handle annotations in each line
    for line_number, line in enumerate(lines, start=1):
        # chcek if the line is empty
        if not line.strip():
            continue

        try:
            # Validate annotation of line is compatible with yolo
            validate_yolo_annotation(line)
            # add text of line in list 
            annotations.append(line)

        except (ValueError, OverflowError) as error:
            errors.append(
                f"{label_path} | ligne {line_number} | {error}"
            )

    return annotations, errors, not annotations # for check is empy file


# Vérifier la correspondance entre images et labels
def validate_class_dataset(class_path, report):
    classe = class_path.name

    image_files, label_files = get_image_and_label_files(class_path)

    # Create Object with name of files without extention as a key and path of file as value
    images_by_stem = {p.stem: p for p in image_files}
    labels_by_stem = {p.stem: p for p in label_files}

    report["total_images"] += len(image_files)
    report["total_labels"] += len(label_files)

    report["missing_labels"].extend(
        str(images_by_stem[stem]) for stem in images_by_stem.keys() - labels_by_stem.keys()
    )

    report["missing_images"].extend(
        str(labels_by_stem[stem]) for stem in labels_by_stem.keys() - images_by_stem.keys()
    )

    # samples is list 
    samples = []

    # Set Intersection & pour get just images and labels has some stem
    for stem in images_by_stem.keys() & labels_by_stem.keys():
        image_path = images_by_stem[stem]
        label_path = labels_by_stem[stem]

        annotations, errors, is_empty = validate_label_file(label_path)

        report["invalid_annotations"].extend(errors)

        # is empty file pass
        if is_empty:
            report["empty_labels"].append(str(label_path))
            continue
        
        # if any proble in file label pass
        if errors:
            continue

        samples.append({
            "image_path": image_path,
            "label_path": label_path,
            "class": classe,
            "annotations": annotations,
        })

    return samples


# Affichr le rapport
def print_dataset_report(report):
    print("\n" + "=" * 45)
    print(" RAPPORT D'INTÉGRITÉ DU DATASET")
    print("=" * 45)

    print(f"Total images          : {report['total_images']}")
    print(f"Total fichiers labels : {report['total_labels']}")
    print(f"Images sans label     : {len(report['missing_labels'])}")
    print(f"Labels sans image     : {len(report['missing_images'])}")
    print(f"Labels vides          : {len(report['empty_labels'])}")
    print(f"Annotations invalides : {len(report['invalid_annotations'])}")

    sections = [
        ("Images sans label", report["missing_labels"]),
        ("Labels sans image", report["missing_images"]),
        ("Labels vides", report["empty_labels"]),
        ("Annotations invalides", report["invalid_annotations"]),
    ]

    for title, errors in sections:
        if errors:
            print(f"\n--- {title} (10 premiers) ---")
            for error in errors[:10]:
                print(error)

    print("\nNettoyage effectué en mémoire uniquement.")


# La fonction principale
def load_and_validate_yolo_dataset(dataset_path):
    # dataset complate
    dataset = []

    # variable of stock rapport of dataset
    report = {
        "total_images": 0,
        "total_labels": 0,
        "missing_labels": [],
        "missing_images": [],
        "empty_labels": [],
        "invalid_annotations": [],
    }

    # Loop les dosser dans dataset Train and Val
    for split_path in sorted(dataset_path.iterdir()):
        if not split_path.is_dir():
            continue

        # Handle classes folders in parent folder
        for class_path in sorted(split_path.iterdir()):
            if not class_path.is_dir():
                continue

            # Extract images and annotation from folder of class
            samples = validate_class_dataset(class_path, report)

            dataset.append({
                "split": split_path.name,
                "class": class_path.name,
                "samples": samples,
            })

    print_dataset_report(report)

    return dataset


# Vérification des dimensions et de la qualité des images
def analyze_image_quality(dataset):
    dimensions = Counter()
    extensions = Counter()
    unreadable_images = []

    total_images = 0

    for group in dataset:
        for sample in group["samples"]:
            image_path = sample["image_path"]
            total_images += 1

            # Compter les extensions
            extensions[image_path.suffix.lower()] += 1

            # Lire l'image
            image = cv2.imread(str(image_path))

            if image is None:
                unreadable_images.append(str(image_path))
                continue

            # Récupérer les dimensions
            height, width = image.shape[:2]
            dimensions[(width, height)] += 1

    print("=" * 45)
    print("ANALYSE DE LA QUALITÉ DES IMAGES")
    print("=" * 45)

    print(f"Total images       : {total_images}")
    print(f"Images illisibles  : {len(unreadable_images)}")
    print(f"Dimensions uniques : {len(dimensions)}")

    print("\nFormats des images :")
    for extension, count in extensions.most_common():
        print(f"{extension}: {count}")

    print("\nDimensions les plus fréquentes :")
    for (width, height), count in dimensions.most_common(10):
        print(f"{width}x{height}: {count} images")

    if unreadable_images:
        print("\nExemples d'images illisibles :")
        for path in unreadable_images[:10]:
            print(path)
            
            
# Vérifier les doublons d’images
def find_duplicate_images(dataset):
    # dict of ket is hash and values is paths
    image_hashes = defaultdict(list)

    # create hashing of eamch image
    for group in dataset:
        for sample in group["samples"]:
            image_path = sample["image_path"]

            try:
                with image_path.open("rb") as file:
                    file_hash = sha256(file.read()).hexdigest()

                image_hashes[file_hash].append(str(image_path))

            except OSError as error:
                print(f"Erreur de lecture : {image_path}")
                print(error)

    # create object with just hasheges has multi images
    duplicates = {
        file_hash: paths
        for file_hash, paths in image_hashes.items()
        if len(paths) > 1
    }

    # Display rapport
    print("=" * 45)
    print("RAPPORT DES DOUBLONS")
    print("=" * 45)
    print(f"Groupes d'images identiques : {len(duplicates)}")

    # afficher sulment les 10 prémier identiques
    for paths in list(duplicates.values())[:10]:
        print("\nImages identiques :")
        for path in paths:
            print(f" - {path}")