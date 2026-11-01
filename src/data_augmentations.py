from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from visualization import parse_annotations, draw_boxes, show_image


# Appliquer Horizontal Flip sur l'image et les Bounding Boxes
def horizontal_flip(image, boxes):

    # Retourner l'image horizontalement
    flipped_image = cv2.flip(image, 1)

    flipped_boxes = []

    for class_id, xc, yc, w, h in boxes:

        # Mettre à jour le centre horizontal
        new_xc = 1 - xc

        flipped_boxes.append((
            class_id,
            new_xc,
            yc,
            w,
            h
        ))

    return flipped_image, flipped_boxes


# Charger une image depuis son chemin
def load_image(image_path):

    image_bgr = cv2.imread(str(image_path))

    if image_bgr is None:
        raise ValueError(f"Impossible de lire l'image : {image_path}")

    # Convertir BGR vers RGB
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


# Préparer les images originale et augmentée
def prepare_flip_samples(sample):

    # Lire l'image
    image_path = Path(sample["image_path"])
    image = load_image(image_path)

    # Lire les annotations
    boxes = parse_annotations(sample["annotations"])

    if not boxes:
        raise ValueError("Cette image ne contient aucune Bounding Box.")

    # Image originale
    original_display = draw_boxes(image, boxes)

    # Image après Horizontal Flip
    flipped_image, flipped_boxes = horizontal_flip(image, boxes)
    flipped_display = draw_boxes(flipped_image, flipped_boxes)

    return original_display, flipped_display


# Afficher les images originale et augmentée
def display_flip_samples(original_display, flipped_display):

    _, axes = plt.subplots(1, 2, figsize=(14, 6))

    show_image(axes[0], original_display, "Image originale")

    show_image(axes[1], flipped_display, "Image après Horizontal Flip")

    plt.tight_layout()
    plt.show()


# Exécuter la visualisation
def visualize_horizontal_flip(sample):

    # Préparer les images
    original_display, flipped_display = prepare_flip_samples(sample)

    # Afficher les résultats
    display_flip_samples(original_display, flipped_display)