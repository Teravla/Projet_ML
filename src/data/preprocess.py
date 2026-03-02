"""Prétraitement des images (resize, normalisation, etc.)."""

from __future__ import annotations

import cv2
import numpy as np


def resize_images(
    images: np.ndarray, target_size: tuple[int, int] = (224, 224)
) -> np.ndarray:
    """Redimensionne un batch d'images vers la même taille."""

    width, height = target_size
    resized = [
        cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        for image in images
    ]
    return np.asarray(resized, dtype=np.float32)


def normalize_images(images: np.ndarray) -> np.ndarray:
    """Normalise les pixels dans [0, 1]."""

    return images.astype(np.float32) / 255.0


def one_hot_encode(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Transforme des labels entiers en encodage one-hot."""

    return np.eye(num_classes, dtype=np.float32)[labels]


def preprocess_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    target_size: tuple[int, int] = (224, 224),
    normalize: bool = True,
    one_hot: bool = False,
    num_classes: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pipeline complet de prétraitement pour un split."""

    processed_images = resize_images(images, target_size=target_size)
    if normalize:
        processed_images = normalize_images(processed_images)

    processed_labels = labels
    if one_hot:
        effective_num_classes = int(
            num_classes if num_classes is not None else np.max(labels) + 1
        )
        processed_labels = one_hot_encode(labels, effective_num_classes)

    return processed_images, processed_labels
