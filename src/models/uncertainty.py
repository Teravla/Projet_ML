"""Estimation d'incertitude (MC Dropout, etc.)."""

from __future__ import annotations

from dataclasses import dataclass

import keras
import numpy as np
import tensorflow as tf
from src.models.utils import summarize_uncertainty_ratio


@dataclass(frozen=True)
class MCUncertaintySummary:
    """Résumé de l'inférence MC Dropout."""

    mean_probabilities: np.ndarray
    variance_probabilities: np.ndarray
    predicted_labels: np.ndarray
    max_probabilities: np.ndarray
    predictive_entropy: np.ndarray


def mc_dropout_predict(
    model: keras.Model,
    x_data: np.ndarray,
    n_iter: int = 20,
) -> MCUncertaintySummary:
    """Estime l'incertitude par Monte Carlo Dropout."""

    if n_iter <= 0:
        raise ValueError("n_iter doit être > 0")

    predictions = []
    inputs = tf.convert_to_tensor(x_data, dtype=tf.float32)
    for _ in range(n_iter):
        probs = model(inputs, training=True).numpy()
        predictions.append(probs)

    stacked = np.stack(predictions, axis=0)
    mean_probs = np.mean(stacked, axis=0)
    variance_probs = np.var(stacked, axis=0)

    predicted_labels = np.argmax(mean_probs, axis=1)
    max_probabilities = np.max(mean_probs, axis=1)
    predictive_entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-12), axis=1)

    return MCUncertaintySummary(
        mean_probabilities=mean_probs,
        variance_probabilities=variance_probs,
        predicted_labels=predicted_labels,
        max_probabilities=max_probabilities,
        predictive_entropy=predictive_entropy,
    )


def summarize_uncertainty(
    max_probabilities: np.ndarray, threshold: float = 0.7
) -> dict[str, float]:
    """Résume le taux de prédictions considérées incertaines."""

    return summarize_uncertainty_ratio(max_probabilities, threshold)
