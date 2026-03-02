"""Augmentation de données pour l'entraînement."""

from __future__ import annotations

import numpy as np
import tensorflow as tf


def create_training_augmenter(seed: int = 42) -> tf.keras.Sequential:
    """Crée un pipeline d'augmentation modéré pour les IRM."""

    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip(mode="horizontal", seed=seed),
            tf.keras.layers.RandomRotation(factor=0.08, seed=seed),
            tf.keras.layers.RandomZoom(height_factor=0.1, width_factor=0.1, seed=seed),
            tf.keras.layers.RandomContrast(factor=0.1, seed=seed),
        ],
        name="mri_data_augmentation",
    )


def augment_batch(
    images: np.ndarray,
    augmenter: tf.keras.Sequential,
    training: bool = True,
) -> np.ndarray:
    """Applique l'augmentation à un batch d'images."""

    tensor = tf.convert_to_tensor(images, dtype=tf.float32)
    augmented = augmenter(tensor, training=training)
    return augmented.numpy().astype(np.float32)
