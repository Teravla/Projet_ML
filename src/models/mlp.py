"""Modèle MLP probabiliste."""

from __future__ import annotations

from dataclasses import dataclass

import keras
import numpy as np


@dataclass(frozen=True)
class MLPTrainingConfig:
    """Paramètres d'entraînement du MLP."""

    epochs: int = 15
    batch_size: int = 64
    learning_rate: float = 1e-3
    dropout_rate: float = 0.3
    hidden_units: tuple[int, int] = (256, 128)


def build_mlp_classifier(
    input_dim: int,
    num_classes: int,
    hidden_units: tuple[int, int] = (256, 128),
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """Construit un classifieur MLP à sortie probabiliste (softmax)."""

    inputs = keras.Input(shape=(input_dim,), name="features")
    x = keras.layers.Dense(hidden_units[0], activation="relu")(inputs)
    x = keras.layers.Dropout(dropout_rate, name="dropout_1")(x)
    x = keras.layers.Dense(hidden_units[1], activation="relu")(x)
    x = keras.layers.Dropout(dropout_rate, name="dropout_2")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="probs")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="mlp_probabilistic")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_mlp_classifier(
    model: keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 15,
    batch_size: int = 64,
    verbose: int = 1,
) -> keras.callbacks.History:
    """Entraîne le MLP avec early stopping sur la validation."""

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        )
    ]
    return model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose,
    )


def predict_probabilities(
    model: keras.Model, x_data: np.ndarray, batch_size: int = 256
) -> np.ndarray:
    """Retourne les probabilités de classes pour un batch de données."""

    probabilities = model.predict(x_data, batch_size=batch_size, verbose=0)
    return np.asarray(probabilities, dtype=np.float32)
