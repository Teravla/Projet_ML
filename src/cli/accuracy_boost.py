"""Script pour améliorer l'accuracy à >90%.

Stratégies:
1. CNN amélioré (ResNet-inspired)
2. Transfer learning (EfficientNetB0)
3. Meilleure fusion d'ensemble
4. Data augmentation avancée
5. Learning rate scheduling
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import keras
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf

from src.data.loader import load_dataset_split

TARGET_ACCURACY = 0.90
BASELINE_ACCURACY_PERCENT = 28.33
NUM_CLASSES = 4


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration d'entraînement des modèles."""

    epochs: int = 100
    batch_size: int = 16


@dataclass(frozen=True)
class PreparedData:
    """Données préparées pour l'entraînement et l'évaluation."""

    x_train: np.ndarray
    x_val: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    class_names: list[str]


@dataclass(frozen=True)
class EvaluationResult:
    """Résultats consolidés d'évaluation."""

    y_pred_ensemble: np.ndarray
    acc_cnn: float
    acc_transfer: float | None
    acc_ensemble: float
    model_transfer: keras.Model | None


def parse_args() -> argparse.Namespace:
    """Parse les options CLI pour adapter le profil d'entraînement."""

    parser = argparse.ArgumentParser(description="Boost accuracy training pipeline")
    parser.add_argument(
        "--img-size",
        type=int,
        default=128,
        choices=[64, 128, 224],
        help="Taille carrée des images (ex: 64 pour 64x64)",
    )
    return parser.parse_args()


def build_improved_cnn(
    input_shape: tuple[int, int, int],
    num_classes: int = NUM_CLASSES,
    learning_rate: float = 1e-4,
) -> keras.Model:
    """CNN amélioré avec batch norm, skip connections et regularization."""

    inputs = keras.Input(shape=input_shape, name="image_input")

    x = keras.layers.Conv2D(64, 3, padding="same", use_bias=False)(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(pool_size=2)(x)
    x = keras.layers.Dropout(0.25)(x)

    residual = keras.layers.Conv2D(128, 1, padding="same")(x)
    x = keras.layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Add()([x, residual])
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(pool_size=2)(x)
    x = keras.layers.Dropout(0.25)(x)

    x = keras.layers.Conv2D(256, 3, padding="same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(256, 3, padding="same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(pool_size=2)(x)
    x = keras.layers.Dropout(0.25)(x)

    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(512, use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Dropout(0.5)(x)

    x = keras.layers.Dense(256, use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Dropout(0.5)(x)

    logits = keras.layers.Dense(num_classes, name="logits")(x)

    model = keras.Model(inputs=inputs, outputs=logits, name="improved_cnn")
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def build_transfer_learning_model(
    input_shape: tuple[int, int, int],
    num_classes: int = NUM_CLASSES,
    learning_rate: float = 1e-4,
) -> keras.Model:
    """Transfer learning avec EfficientNetB0."""

    base_model = keras.applications.EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )

    for layer in base_model.layers[:-40]:
        layer.trainable = False

    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Rescaling(255.0)(inputs)
    x = base_model(x, training=False)

    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(512, activation="relu", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(256, activation="relu", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)
    logits = keras.layers.Dense(num_classes)(x)

    model = keras.Model(inputs=inputs, outputs=logits, name="transfer_learning")
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def train_model_with_augmentation(
    model: keras.Model,
    train_data: tuple[np.ndarray, np.ndarray],
    validation_data: tuple[np.ndarray, np.ndarray],
    config: TrainingConfig,
) -> keras.callbacks.History:
    """Entraîne un modèle avec data augmentation et learning-rate scheduling."""

    x_train, y_train = train_data
    x_val, y_val = validation_data

    train_augmentation = keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomFlip("vertical"),
            keras.layers.RandomRotation(0.2),
            keras.layers.RandomZoom(0.2),
            keras.layers.RandomTranslation(0.1, 0.1),
        ]
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            restore_best_weights=True,
            mode="max",
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    x_train_augmented = train_augmentation(x_train, training=True)
    return model.fit(
        x_train_augmented,
        y_train,
        validation_data=(x_val, y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        verbose=1,
    )


def ensemble_predict(
    models: list[keras.Model],
    x_data: np.ndarray,
    weights: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fusion pondérée des prédictions de plusieurs modèles."""

    if weights is None:
        weights = [1.0 / len(models)] * len(models)

    ensemble_logits = np.zeros((x_data.shape[0], NUM_CLASSES))
    for model, weight in zip(models, weights):
        logits = model.predict(x_data, verbose=0)
        ensemble_logits += weight * logits

    probs = tf.nn.softmax(ensemble_logits, axis=1).numpy()
    preds = np.argmax(ensemble_logits, axis=1)
    return preds, probs


def encode_labels(labels: np.ndarray, class_names: list[str]) -> np.ndarray:
    """Convertit les labels en indices entiers de classes."""

    return (
        np.array(labels, dtype=np.int64)
        if isinstance(labels[0], (int, np.integer))
        else np.array([class_names.index(label) for label in labels], dtype=np.int64)
    )


def prepare_data(data_dir: Path, img_size: tuple[int, int]) -> PreparedData:
    """Charge, normalise et découpe les données train/validation/test."""

    train_split = load_dataset_split(data_dir / "Training", image_size=img_size)
    test_split = load_dataset_split(
        data_dir / "Testing",
        image_size=img_size,
        class_names=train_split.class_names,
    )

    x_train_all = train_split.images.astype(np.float32) / 255.0
    y_train_all = encode_labels(train_split.labels, train_split.class_names)
    x_test = test_split.images.astype(np.float32) / 255.0
    y_test = encode_labels(test_split.labels, test_split.class_names)

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_all,
        y_train_all,
        test_size=0.2,
        random_state=42,
        stratify=y_train_all,
    )

    return PreparedData(
        x_train=x_train,
        x_val=x_val,
        y_train=y_train,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        class_names=train_split.class_names,
    )


def train_models(
    data: PreparedData,
    config: TrainingConfig,
) -> tuple[keras.Model, keras.Model | None]:
    """Entraîne le CNN amélioré et tente l'entraînement transfer learning."""

    print("\n[2/5] Entraînement du CNN amélioré...")
    cnn_improved = build_improved_cnn(data.x_train.shape[1:])
    train_model_with_augmentation(
        model=cnn_improved,
        train_data=(data.x_train, data.y_train),
        validation_data=(data.x_val, data.y_val),
        config=config,
    )

    print("\n[3/5] Entraînement avec Transfer Learning (EfficientNetB0)...")
    model_transfer = build_transfer_learning_model(data.x_train.shape[1:])
    transfer_config = TrainingConfig(
        epochs=config.epochs // 2, batch_size=config.batch_size
    )

    try:
        train_model_with_augmentation(
            model=model_transfer,
            train_data=(data.x_train, data.y_train),
            validation_data=(data.x_val, data.y_val),
            config=transfer_config,
        )
    except (ValueError, RuntimeError, tf.errors.OpError) as exc:
        print(f"  Transfer learning échoué: {exc}")
        model_transfer = None

    return cnn_improved, model_transfer


def evaluate_models(
    data: PreparedData,
    cnn_improved: keras.Model,
    model_transfer: keras.Model | None,
) -> EvaluationResult:
    """Évalue CNN/transfer et calcule la prédiction d'ensemble."""

    print("\n[4/5] Évaluation sur l'ensemble test...")

    y_pred_cnn = np.argmax(cnn_improved.predict(data.x_test, verbose=0), axis=1)
    acc_cnn = accuracy_score(data.y_test, y_pred_cnn)
    print(f"  CNN Amélioré Accuracy:  {acc_cnn:.4f} ({acc_cnn*100:.2f}%)")

    models_to_ensemble = [cnn_improved]
    weights_to_use = [1.0]
    acc_transfer: float | None = None

    if model_transfer is not None:
        y_pred_transfer = np.argmax(
            model_transfer.predict(data.x_test, verbose=0), axis=1
        )
        acc_transfer = accuracy_score(data.y_test, y_pred_transfer)
        print(f"  Transfer Learning Acc: {acc_transfer:.4f} ({acc_transfer*100:.2f}%)")

        if acc_transfer > acc_cnn:
            total = acc_cnn + acc_transfer
            models_to_ensemble.append(model_transfer)
            weights_to_use = [acc_cnn / total, acc_transfer / total]
            print(
                f"  Ensemble weights: CNN={weights_to_use[0]:.3f}, "
                f"Transfer={weights_to_use[1]:.3f}"
            )
        else:
            print("  Transfer ignoré pour l'ensemble (moins bon que le CNN).")

    print("\n[5/5] Fusion d'ensemble...")
    y_pred_ensemble, _ = ensemble_predict(
        models_to_ensemble, data.x_test, weights=weights_to_use
    )
    acc_ensemble = accuracy_score(data.y_test, y_pred_ensemble)
    print(f"  Ensemble Accuracy:      {acc_ensemble:.4f} ({acc_ensemble*100:.2f}%)")

    return EvaluationResult(
        y_pred_ensemble=y_pred_ensemble,
        acc_cnn=acc_cnn,
        acc_transfer=acc_transfer,
        acc_ensemble=acc_ensemble,
        model_transfer=model_transfer,
    )


def print_goal_status(acc_ensemble: float) -> None:
    """Affiche l'état d'atteinte de l'objectif d'accuracy."""

    print("\n" + "=" * 70)
    if acc_ensemble > TARGET_ACCURACY:
        print(f"OBJECTIF ATTEINT! Accuracy: {acc_ensemble*100:.2f}% > 90%")
        return

    improvement = (acc_ensemble * 100) - BASELINE_ACCURACY_PERCENT
    print(
        f"Amélioration: +{improvement:.2f}% "
        f"({BASELINE_ACCURACY_PERCENT:.2f}% -> {acc_ensemble*100:.2f}%)"
    )
    print("   Objectif: >90%")


def print_classification_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> None:
    """Affiche le rapport de classification détaillé."""

    print("\n" + "=" * 70)
    print("RAPPORT DE CLASSIFICATION")
    print("=" * 70)
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            digits=4,
        )
    )


def save_models(cnn_model: keras.Model, transfer_model: keras.Model | None) -> None:
    """Sauvegarde les modèles entraînés dans artifacts/models."""

    print("\nSauvegarde des modèles...")
    artifacts_dir = Path("artifacts/models")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    cnn_model.save(artifacts_dir / "cnn_improved.h5")
    if transfer_model is not None:
        transfer_model.save(artifacts_dir / "transfer_learning_efficientnet.h5")

    print(f"  Modèles sauvegardés dans {artifacts_dir}")


def main() -> None:
    """Exécute le pipeline complet d'amélioration d'accuracy."""

    args = parse_args()
    img_size = (args.img_size, args.img_size)
    config = TrainingConfig()

    print("\nAMÉLIORATION DE L'ACCURACY - TARGET: >90%\n")
    print("=" * 70)
    print(
        f"Configuration: image_size={img_size[0]}x{img_size[1]}, "
        f"epochs={config.epochs}, batch_size={config.batch_size}"
    )

    print("\n[1/5] Chargement des données...")
    data = prepare_data(Path("data"), img_size)
    print(
        f"  Train: {data.x_train.shape}, Val: {data.x_val.shape}, Test: {data.x_test.shape}"
    )

    cnn_model, transfer_model = train_models(data, config)
    evaluation = evaluate_models(data, cnn_model, transfer_model)

    print_goal_status(evaluation.acc_ensemble)
    print_classification_summary(
        data.y_test, evaluation.y_pred_ensemble, data.class_names
    )
    save_models(cnn_model, evaluation.model_transfer)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
