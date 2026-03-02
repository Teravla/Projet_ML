"""Architecture CNN pour classification IRM."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf


@dataclass(frozen=True)
class CNNTrainingConfig:
    """Paramètres d'entraînement CNN."""

    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3
    dropout_rate: float = 0.3


def build_cnn_classifier(
    input_shape: tuple[int, int, int],
    num_classes: int,
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """Construit un CNN classique avec sortie logits."""

    inputs = tf.keras.Input(shape=input_shape, name="image")

    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu", name="conv1")(
        inputs
    )
    x = tf.keras.layers.MaxPooling2D(pool_size=2, name="pool1")(x)

    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu", name="conv2")(
        x
    )
    x = tf.keras.layers.MaxPooling2D(pool_size=2, name="pool2")(x)

    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu", name="conv3")(
        x
    )
    x = tf.keras.layers.MaxPooling2D(pool_size=2, name="pool3")(x)

    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="dense1")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout")(x)
    logits = tf.keras.layers.Dense(num_classes, name="logits")(x)

    model = tf.keras.Model(inputs=inputs, outputs=logits, name="cnn_logits")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def train_cnn_classifier(
    model: tf.keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 10,
    batch_size: int = 64,
    verbose: int = 1,
) -> tf.keras.callbacks.History:
    """Entraîne le CNN avec early stopping."""

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
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


def predict_logits(
    model: tf.keras.Model, x_data: np.ndarray, batch_size: int = 256
) -> np.ndarray:
    """Retourne les logits prédits."""

    logits = model.predict(x_data, batch_size=batch_size, verbose=0)
    return np.asarray(logits, dtype=np.float32)


def predict_probabilities(
    model: tf.keras.Model, x_data: np.ndarray, batch_size: int = 256
) -> np.ndarray:
    """Retourne les probabilités softmax."""

    logits = predict_logits(model, x_data, batch_size=batch_size)
    return tf.nn.softmax(logits, axis=1).numpy().astype(np.float32)


def extract_intermediate_activations(
    model: tf.keras.Model,
    x_data: np.ndarray,
    layer_names: list[str],
    batch_size: int = 256,
) -> dict[str, np.ndarray]:
    """Extrait les activations intermédiaires de couches nommées."""

    activations: dict[str, np.ndarray] = {}
    for layer_name in layer_names:
        layer = model.get_layer(layer_name)
        activation_model = tf.keras.Model(inputs=model.input, outputs=layer.output)
        values = activation_model.predict(x_data, batch_size=batch_size, verbose=0)
        activations[layer_name] = np.asarray(values)
    return activations
