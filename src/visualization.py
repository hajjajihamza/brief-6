from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import cv2
import math
from const import CLASS_NAMES


# Lire les annotations YOLO
def parse_annotations(annotations):
    boxes = []

    for line in annotations:
        class_id, xc, yc, w, h = line.split()

        boxes.append((
            int(class_id),
            float(xc),
            float(yc),
            float(w),
            float(h)
        ))

    return boxes


# Dessiner les Bounding Boxes
def draw_boxes(image, boxes):
    result = image.copy()

    height, width = result.shape[:2]

    for class_id, xc, yc, w, h in boxes:
        # Normalized coordinates -> Pixels
        x1 = int((xc - w / 2) * width)
        y1 = int((yc - h / 2) * height)
        x2 = int((xc + w / 2) * width)
        y2 = int((yc + h / 2) * height)

        # Dessiner la Bounding Box
        cv2.rectangle(
            result,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Nom de la classe
        label = CLASS_NAMES[class_id]

        # Display class name
        cv2.putText(
            result,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return result

# Afficher image
def show_image(ax, image, title):
    ax.imshow(image)
    ax.set_title(title)
    ax.axis("off")

# drow bounding box for sample
def draw_single_yolo_image(ax, sample):
    image_path = sample["image_path"]

    # Load image with OpenCV
    image = cv2.imread(str(image_path))

    if image is None:
        ax.set_title(f"Image illisible : {image_path.name}")
        ax.axis("off")
        return

    # BGR -> RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Lire les annotations
    boxes = parse_annotations(sample["annotations"])

    if not boxes:
        raise ValueError("Cette image ne contient aucune Bounding Box.")


    # Image avec Bounding Boxes
    image_display = draw_boxes(image, boxes)

    # Afficher image
    show_image(ax, image_display, f"{sample['class']} | {image_path.name}")

# Organize layout of samples
def draw_yolo_samples(samples, n_cols=3):
    if not samples:
        print("Aucun sample valide à afficher.")
        return

    n_rows = math.ceil(len(samples) / n_cols)

    _, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(15, 5 * n_rows),
        squeeze=False
    )

    axes = axes.flatten()

    for ax, sample in zip(axes, samples):
        draw_single_yolo_image(ax, sample)

    plt.tight_layout()
    plt.show()


# Analyse de la distribution des classes
def analyze_class_distribution(dataset):
    class_counts = Counter(
        sample["class"]
        for group in dataset
        for sample in group["samples"]
    )

    if not class_counts:
        print("Aucune image valide dans le dataset.")
        return

    total = sum(class_counts.values())

    # Afficher la distribution
    print(f"{'Classe':<15} {'Images':>10} {'Pourcentage':>15}")
    print("-" * 42)

    for classe, count in class_counts.most_common():
        print(f"{classe:<15} {count:>10} {count / total * 100:>14.2f}%")

    print("-" * 42)
    print(f"{'Total':<15} {total:>10} {'100.00%':>15}")

    # Visualisation
    classes, counts = zip(*class_counts.items())

    bars = plt.bar(classes, counts)

    plt.figure(figsize=(9, 5))
    bars = plt.bar(classes, counts)
    plt.title("Distribution des images par classe")
    plt.xlabel("Classes")
    plt.ylabel("Nombre d'images")
    plt.xticks(rotation=20)

    for bar in bars:
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            int(bar.get_height()),
            ha="center",
            va="bottom"
        )

    plt.tight_layout()
    plt.show()


# Analisy result.csv file 
def visualization_results_csv(results_path: Path, name:str):
    if not results_path.exists():
        raise FileNotFoundError(
            f"Fichier results.csv introuvable : {results_path}\n"
            "Vérifie que l'entraînement est terminé et le chemin."
        )

    # Lire le fichier CSV
    df = pd.read_csv(str(results_path))

    # Nettoyer les espaces dans les noms des colonnes
    df.columns = df.columns.str.strip()

    print("Colonnes disponibles :")
    print(df.columns.tolist())

    print("\nDernières lignes :")
    print(df.tail())

    # Afficher les courbes de Loss
    loss_columns = [
        col for col in [
            "train/box_loss",
            "train/cls_loss",
            "train/dfl_loss",
            "val/box_loss",
            "val/cls_loss",
            "val/dfl_loss"
        ]
        if col in df.columns
    ]

    if loss_columns:
        df.plot(
            x="epoch",
            y=loss_columns,
            figsize=(12, 6),
            title=f"{name} - Évolution des Loss"
        )
        plt.ylabel("Loss")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print("Aucune colonne Loss reconnue.")

    # Afficher les métriques de validation
    metric_columns = [
        col for col in [
            "metrics/precision(B)",
            "metrics/recall(B)",
            "metrics/mAP50(B)",
            "metrics/mAP50-95(B)"
        ]
        if col in df.columns
    ]

    if metric_columns:
        df.plot(
            x="epoch",
            y=metric_columns,
            figsize=(12, 6),
            title=f"{name} - Métriques de validation"
        )
        plt.ylabel("Score")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print("Aucune colonne de métriques reconnue.")

    # Identifier l'epoch avec le meilleur mAP50-95
    map_column = "metrics/mAP50-95(B)"

    if map_column in df.columns:
        best_index = df[map_column].idxmax()
        best_row = df.loc[best_index]

        print("\nMeilleur résultat mAP50-95 :")
        print("Epoch :", int(best_row["epoch"]) + 1)
        print("mAP50-95 :", round(best_row[map_column], 4))

        if "metrics/mAP50(B)" in df.columns:
            print(
                "mAP50 :",
                round(best_row["metrics/mAP50(B)"], 4)
            )

        if "metrics/precision(B)" in df.columns:
            print(
                "Precision :",
                round(best_row["metrics/precision(B)"], 4)
            )

        if "metrics/recall(B)" in df.columns:
            print(
                "Recall :",
                round(best_row["metrics/recall(B)"], 4)
            )
