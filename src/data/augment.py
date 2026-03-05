"""Augmentation de données pour l'entraînement."""

from __future__ import annotations

import keras
import numpy as np
import tensorflow as tf


def create_training_augmenter(seed: int = 42) -> keras.Sequential:
    """Crée un pipeline d'augmentation modéré pour les IRM."""

    return keras.Sequential(
        [
            keras.layers.RandomFlip(mode="horizontal", seed=seed),
            keras.layers.RandomRotation(factor=0.08, seed=seed),
            keras.layers.RandomZoom(height_factor=0.1, width_factor=0.1, seed=seed),
            keras.layers.RandomContrast(factor=0.1, seed=seed),
        ],
        name="mri_data_augmentation",
    )


def augment_batch(
    images: np.ndarray,
    augmenter: keras.Sequential,
    training: bool = True,
) -> np.ndarray:
    """Applique l'augmentation à un batch d'images."""

    tensor = tf.convert_to_tensor(images, dtype=tf.float32)
    augmented = augmenter(tensor, training=training)
    return augmented.numpy().astype(np.float32)
