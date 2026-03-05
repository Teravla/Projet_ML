"""Architecture CNN pour classification IRM."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import keras
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
) -> keras.Model:
    """Construit un CNN classique avec sortie logits."""

    inputs = keras.Input(shape=input_shape, name="image")

    x = keras.layers.Conv2D(32, 3, padding="same", activation="relu", name="conv1")(
        inputs
    )
    x = keras.layers.MaxPooling2D(pool_size=2, name="pool1")(x)

    x = keras.layers.Conv2D(64, 3, padding="same", activation="relu", name="conv2")(x)
    x = keras.layers.MaxPooling2D(pool_size=2, name="pool2")(x)

    x = keras.layers.Conv2D(128, 3, padding="same", activation="relu", name="conv3")(x)
    x = keras.layers.MaxPooling2D(pool_size=2, name="pool3")(x)

    x = keras.layers.Flatten(name="flatten")(x)
    x = keras.layers.Dense(256, activation="relu", name="dense1")(x)
    x = keras.layers.Dropout(dropout_rate, name="dropout")(x)
    logits = keras.layers.Dense(num_classes, name="logits")(x)

    model = keras.Model(inputs=inputs, outputs=logits, name="cnn_logits")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def train_cnn_classifier(
    model: keras.Model,
    train_data: tuple[np.ndarray, np.ndarray],
    validation_data: tuple[np.ndarray, np.ndarray],
    training_config: CNNTrainingConfig | None = None,
    verbose: int = 1,
) -> keras.callbacks.History:
    """Entraîne le CNN avec early stopping."""

    x_train, y_train = train_data
    x_val, y_val = validation_data
    config = training_config or CNNTrainingConfig()

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
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        verbose=verbose,
    )


def predict_logits(
    model: keras.Model, x_data: np.ndarray, batch_size: int = 256
) -> np.ndarray:
    """Retourne les logits prédits."""

    logits = model.predict(x_data, batch_size=batch_size, verbose=0)
    return np.asarray(logits, dtype=np.float32)


def predict_probabilities(
    model: keras.Model, x_data: np.ndarray, batch_size: int = 256
) -> np.ndarray:
    """Retourne les probabilités prédites via softmax."""

    logits = predict_logits(model, x_data, batch_size)
    return tf.nn.softmax(logits, axis=1).numpy()


def build_cnn_optimized(
    input_shape: tuple[int, int, int],
    num_classes: int,
    dropout_rate: float = 0.3,
    learning_rate: float = 3e-4,
    l2_reg: float = 3e-5,
) -> keras.Model:
    """CNN optimisé stable pour limiter l'overfit et les écarts train/val."""

    regularizer = keras.regularizers.l2(l2_reg)
    inputs = keras.Input(shape=input_shape, name="image")

    # Bloc 1
    x = keras.layers.Conv2D(
        32, 3, padding="same", activation="relu", kernel_regularizer=regularizer
    )(inputs)
    x = keras.layers.Conv2D(
        32, 3, padding="same", activation="relu", kernel_regularizer=regularizer
    )(x)
    x = keras.layers.MaxPooling2D(pool_size=2)(x)
    x = keras.layers.Dropout(dropout_rate * 0.5)(x)

    # Bloc 2
    x = keras.layers.Conv2D(
        64, 3, padding="same", activation="relu", kernel_regularizer=regularizer
    )(x)
    x = keras.layers.Conv2D(
        64, 3, padding="same", activation="relu", kernel_regularizer=regularizer
    )(x)
    x = keras.layers.MaxPooling2D(pool_size=2)(x)
    x = keras.layers.Dropout(dropout_rate * 0.8)(x)

    # Bloc 3
    x = keras.layers.Conv2D(
        128, 3, padding="same", activation="relu", kernel_regularizer=regularizer
    )(x)
    x = keras.layers.Conv2D(
        128, 3, padding="same", activation="relu", kernel_regularizer=regularizer
    )(x)
    x = keras.layers.MaxPooling2D(pool_size=2)(x)
    x = keras.layers.Dropout(dropout_rate)(x)

    # Global Average Pooling (meilleur que Flatten pour réduire overfitting)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(dropout_rate * 1.2)(x)

    # Dense finale
    x = keras.layers.Dense(256, activation="relu", kernel_regularizer=regularizer)(x)
    x = keras.layers.Dropout(dropout_rate * 1.5)(x)

    logits = keras.layers.Dense(num_classes, name="logits")(x)

    model = keras.Model(inputs=inputs, outputs=logits, name="cnn_optimized")

    # Optimizer avec gradient clipping pour stabilité
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def extract_intermediate_activations(
    model: keras.Model,
    x_data: np.ndarray,
    layer_names: list[str],
    batch_size: int = 256,
) -> dict[str, np.ndarray]:
    """Extrait les activations intermédiaires de couches nommées."""

    activations: dict[str, np.ndarray] = {}
    for layer_name in layer_names:
        layer = model.get_layer(layer_name)
        activation_model = keras.Model(inputs=model.input, outputs=layer.output)
        values = activation_model.predict(x_data, batch_size=batch_size, verbose=0)
        activations[layer_name] = np.asarray(values)
    return activations
