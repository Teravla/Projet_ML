"""Utilities modèles partagées (entraînement, compilation, incertitude)."""

from __future__ import annotations

from dataclasses import dataclass

import keras
import numpy as np


def compile_logits_classifier(
    model: keras.Model,
    learning_rate: float,
    clipnorm: float | None = None,
) -> None:
    """Compile un classifieur logits avec SparseCategoricalCrossentropy."""

    optimizer_kwargs: dict[str, float] = {"learning_rate": learning_rate}
    if clipnorm is not None:
        optimizer_kwargs["clipnorm"] = clipnorm

    model.compile(
        optimizer=keras.optimizers.Adam(**optimizer_kwargs),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )


@dataclass(frozen=True)
class EarlyStoppingFitConfig:
    """Configuration standard d'entraînement avec early stopping."""

    epochs: int
    batch_size: int
    verbose: int = 1
    monitor: str = "val_loss"
    patience: int = 3


def train_with_early_stopping(
    model: keras.Model,
    train_data: tuple[np.ndarray, np.ndarray],
    validation_data: tuple[np.ndarray, np.ndarray],
    config: EarlyStoppingFitConfig,
) -> keras.callbacks.History:
    """Entraîne un modèle avec early stopping standard."""

    x_train, y_train = train_data
    x_val, y_val = validation_data

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor=config.monitor,
            patience=config.patience,
            restore_best_weights=True,
        )
    ]
    return model.fit(
        x=x_train,
        y=y_train,
        validation_data=(x_val, y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        verbose=config.verbose,
    )


def summarize_uncertainty_ratio(
    max_probabilities: np.ndarray,
    threshold: float = 0.7,
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
