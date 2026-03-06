"""Prétraitement des images (resize, normalisation, etc.)."""

from __future__ import annotations

import numpy as np

from src.config.config import CV_INTER_AREA, CV_RESIZE
from src.enums.dataclass import LabelEncodingConfig


def resize_images(
    images: np.ndarray, target_size: tuple[int, int] = (224, 224)
) -> np.ndarray:
    """Redimensionne un batch d'images vers la même taille."""

    width, height = target_size
    resized = [
        CV_RESIZE(image, (width, height), interpolation=CV_INTER_AREA)
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
    label_config: LabelEncodingConfig | None = None,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Pipeline complet de prétraitement pour un split."""

    # Backward compatibility: support legacy kwargs one_hot / num_classes.
    if label_config is None:
        label_config = LabelEncodingConfig(
            one_hot=bool(kwargs.pop("one_hot", False)),
            num_classes=kwargs.pop("num_classes", None),
        )
    if kwargs:
        unknown = ", ".join(sorted(kwargs.keys()))
        raise TypeError(f"Unexpected keyword arguments: {unknown}")

    processed_images = resize_images(images, target_size=target_size)
    if normalize:
        processed_images = normalize_images(processed_images)

    processed_labels = labels
    if label_config.one_hot:
        effective_num_classes = int(
            label_config.num_classes
            if label_config.num_classes is not None
            else np.max(labels) + 1
        )
        processed_labels = one_hot_encode(labels, effective_num_classes)

    return processed_images, processed_labels
