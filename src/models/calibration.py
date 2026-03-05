"""Calibration des probabilités (Platt, Isotonic, Temperature Scaling)."""

from __future__ import annotations

import keras
import numpy as np
import tensorflow as tf
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression


def calibrate_classifier(
    base_model: LogisticRegression,
    method: str = "sigmoid",
    cv: int = 3,
) -> CalibratedClassifierCV:
    """Crée un calibrateur sklearn (sigmoid=Platt, isotonic=Isotonic)."""

    if method not in {"sigmoid", "isotonic"}:
        raise ValueError("method doit être 'sigmoid' ou 'isotonic'")
    return CalibratedClassifierCV(estimator=base_model, method=method, cv=cv)


def summarize_confidence_distribution(
    max_probabilities: np.ndarray,
) -> dict[str, float]:
    """Produit une synthèse des niveaux de confiance."""

    return {
        "p25": float(np.percentile(max_probabilities, 25)),
        "p50": float(np.percentile(max_probabilities, 50)),
        "p75": float(np.percentile(max_probabilities, 75)),
        "mean": float(np.mean(max_probabilities)),
        "std": float(np.std(max_probabilities)),
    }


def analyze_uncertain_predictions(
    max_probabilities: np.ndarray, threshold: float = 0.7
) -> dict[str, float]:
    """Calcule la part de prédictions incertaines pour un seuil donné."""

    uncertain_mask = max_probabilities < threshold
    uncertain_count = int(np.sum(uncertain_mask))
    total = int(max_probabilities.shape[0])
    uncertain_ratio = float(uncertain_count / total) if total else 0.0
    return {
        "threshold": float(threshold),
        "uncertain_count": uncertain_count,
        "total": total,
        "uncertain_ratio": uncertain_ratio,
    }


class TemperatureScaler:
    """Calibre les logits via un facteur de température scalaire."""

    def __init__(self, initial_temperature: float = 1.0) -> None:
        if initial_temperature <= 0:
            raise ValueError("initial_temperature doit être > 0")
        self._log_temperature = tf.Variable(
            float(np.log(initial_temperature)),
            dtype=tf.float32,
            trainable=True,
        )

    @property
    def temperature(self) -> float:
        return float(tf.exp(self._log_temperature).numpy())

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        learning_rate: float = 1e-2,
        epochs: int = 300,
        verbose: int = 0,
    ) -> None:
        """Ajuste la température en minimisant la NLL sur un set de calibration."""

        if epochs <= 0:
            raise ValueError("epochs doit être > 0")

        logits_tf = tf.convert_to_tensor(logits, dtype=tf.float32)
        labels_tf = tf.convert_to_tensor(labels, dtype=tf.int32)
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        for step in range(epochs):
            with tf.GradientTape() as tape:
                temperature = tf.exp(self._log_temperature)
                scaled_logits = logits_tf / temperature
                loss = loss_fn(labels_tf, scaled_logits)

            grads = tape.gradient(loss, [self._log_temperature])
            optimizer.apply_gradients(zip(grads, [self._log_temperature]))

            if verbose and (step + 1) % 50 == 0:
                print(
                    f"Temperature scaling step={step + 1}/{epochs} "
                    f"loss={float(loss.numpy()):.4f} T={self.temperature:.4f}"
                )

    def apply(self, logits: np.ndarray) -> np.ndarray:
        """Applique la température aux logits."""

        temperature = self.temperature
        return logits / temperature

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        """Retourne les probabilités calibrées après temperature scaling."""

        scaled_logits = self.apply(logits)
        return tf.nn.softmax(scaled_logits, axis=1).numpy().astype(np.float32)
