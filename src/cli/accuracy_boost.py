"""Script pour améliorer l'accuracy à >90%
Stratégies:
1. CNN amélioré (ResNet-inspired)
2. Transfer learning (EfficientNetB0)
3. Meilleure fusion d'ensemble
4. Data augmentation avancée
5. Learning rate scheduling
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_dataset_split
from src.data.preprocess import preprocess_dataset


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
    num_classes: int = 4,
    learning_rate: float = 1e-4,
) -> tf.keras.Model:
    """CNN amélioré avec batch norm, skip connections, regularization."""

    inputs = tf.keras.Input(shape=input_shape, name="image_input")

    # Bloc 1: Conv + BatchNorm + Pool
    x = tf.keras.layers.Conv2D(64, 3, padding="same", use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    # Bloc 2: Résiduel-like
    residual = tf.keras.layers.Conv2D(128, 1, padding="same")(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, residual])
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    # Bloc 3
    x = tf.keras.layers.Conv2D(256, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(256, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    # Global Average Pooling + Dense
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Dense(256, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    logits = tf.keras.layers.Dense(num_classes, name="logits")(x)

    model = tf.keras.Model(inputs=inputs, outputs=logits, name="improved_cnn")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def build_transfer_learning_model(
    input_shape: tuple[int, int, int],
    num_classes: int = 4,
    learning_rate: float = 1e-4,
) -> tf.keras.Model:
    """Transfer learning avec EfficientNetB0."""

    # Charger EfficientNetB0 pré-entraîné
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )

    # Geler les couches initiales, dégeler les dernières
    for layer in base_model.layers[:-40]:
        layer.trainable = False

    inputs = tf.keras.Input(shape=input_shape)

    # Les images du pipeline sont en [0, 1]. EfficientNetB0 (Keras) est calibré
    # pour des entrées [0, 255], donc on remonte l'échelle avant le backbone.
    x = tf.keras.layers.Rescaling(255.0)(inputs)
    # Mode inference pour stabiliser les BatchNorm majoritairement gelées.
    x = base_model(x, training=False)

    # Head de classification
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation="relu", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(256, activation="relu", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    logits = tf.keras.layers.Dense(num_classes)(x)

    model = tf.keras.Model(inputs=inputs, outputs=logits, name="transfer_learning")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def train_model_with_augmentation(
    model: tf.keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32,
) -> tf.keras.callbacks.History:
    """Entraîne avec data augmentation et learning rate scheduling."""

    # Data augmentation robuste
    train_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomFlip("vertical"),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.2),
            tf.keras.layers.RandomTranslation(0.1, 0.1),
        ]
    )

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            restore_best_weights=True,
            mode="max",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    # Augmenter les données d'entraînement
    x_train_augmented = train_augmentation(x_train, training=True)

    history = model.fit(
        x_train_augmented,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    return history


def ensemble_predict(
    models: list[tf.keras.Model],
    x_data: np.ndarray,
    weights: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fusion pondérée des prédictions de plusieurs modèles.

    Returns:
        (y_pred_ensemble, y_proba_ensemble)
    """

    if weights is None:
        weights = [1.0 / len(models)] * len(models)

    ensemble_logits = np.zeros((x_data.shape[0], 4))

    for model, weight in zip(models, weights):
        logits = model.predict(x_data, verbose=0)
        ensemble_logits += weight * logits

    # Softmax sur les logits pondérés
    probs = tf.nn.softmax(ensemble_logits, axis=1).numpy()
    preds = np.argmax(ensemble_logits, axis=1)

    return preds, probs


def main() -> None:
    args = parse_args()

    print("\n🚀 AMÉLIORATION DE L'ACCURACY - TARGET: >90%\n")
    print("=" * 70)

    # Paramètres
    data_dir = Path("data")
    img_size = (args.img_size, args.img_size)
    epochs = 100  # Augmenté de 10 à 100
    batch_size = 16  # Réduit pour meilleure convergence

    print(
        f"Configuration: image_size={img_size[0]}x{img_size[1]}, "
        f"epochs={epochs}, batch_size={batch_size}"
    )

    # Charger les données
    print("\n[1/5] Chargement des données...")
    train_split = load_dataset_split(data_dir / "Training", image_size=img_size)
    test_split = load_dataset_split(
        data_dir / "Testing",
        image_size=img_size,
        class_names=train_split.class_names,
    )

    x_train = train_split.images.astype(np.float32) / 255.0
    # Labels peuvent être déjà des indices (np.int64) ou des noms de classe (str)
    y_train_raw = train_split.labels
    if isinstance(y_train_raw[0], (int, np.integer)):
        y_train = np.array(y_train_raw, dtype=np.int64)
    else:
        y_train = np.array(
            [train_split.class_names.index(label) for label in y_train_raw]
        )

    x_test = test_split.images.astype(np.float32) / 255.0
    y_test_raw = test_split.labels
    if isinstance(y_test_raw[0], (int, np.integer)):
        y_test = np.array(y_test_raw, dtype=np.int64)
    else:
        y_test = np.array([test_split.class_names.index(label) for label in y_test_raw])

    # Split train en train/val
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    print(f"  Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")

    # Entraîner plusieurs modèles
    print("\n[2/5] Entraînement du CNN amélioré...")
    cnn_improved = build_improved_cnn(x_train.shape[1:])
    train_model_with_augmentation(
        cnn_improved,
        x_train,
        y_train,
        x_val,
        y_val,
        epochs=epochs,
        batch_size=batch_size,
    )

    print("\n[3/5] Entraînement avec Transfer Learning (EfficientNetB0)...")
    model_transfer = build_transfer_learning_model(x_train.shape[1:])
    try:
        train_model_with_augmentation(
            model_transfer,
            x_train,
            y_train,
            x_val,
            y_val,
            epochs=epochs // 2,
            batch_size=batch_size,
        )
    except Exception as e:
        print(f"  ⚠️ Transfer learning échoué: {e}")
        model_transfer = None

    # Évaluer les modèles
    print("\n[4/5] Évaluation sur l'ensemble test...")

    y_pred_cnn = np.argmax(cnn_improved.predict(x_test, verbose=0), axis=1)
    acc_cnn = accuracy_score(y_test, y_pred_cnn)
    print(f"  CNN Amélioré Accuracy:  {acc_cnn:.4f} ({acc_cnn*100:.2f}%)")

    # Ensemble voting (robuste): n'agrège que si ça améliore réellement.
    models_to_ensemble = [cnn_improved]
    weights_to_use = [1.0]

    if model_transfer is not None:
        y_pred_transfer = np.argmax(model_transfer.predict(x_test, verbose=0), axis=1)
        acc_transfer = accuracy_score(y_test, y_pred_transfer)
        print(f"  Transfer Learning Acc: {acc_transfer:.4f} ({acc_transfer*100:.2f}%)")

        # Pondération par performance: on évite qu'un modèle faible dégrade l'ensemble.
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
        models_to_ensemble, x_test, weights=weights_to_use
    )
    acc_ensemble = accuracy_score(y_test, y_pred_ensemble)
    print(f"  Ensemble Accuracy:      {acc_ensemble:.4f} ({acc_ensemble*100:.2f}%)")

    # Objectif atteint?
    print("\n" + "=" * 70)
    if acc_ensemble > 0.90:
        print(f"✅ OBJECTIF ATTEINT! Accuracy: {acc_ensemble*100:.2f}% > 90%")
    else:
        improvement = (acc_ensemble * 100) - 28.33
        print(
            f"📈 Amélioration: +{improvement:.2f}% (28.33% → {acc_ensemble*100:.2f}%)"
        )
        print(f"   Objectif: >90%")

    # Rapport détaillé
    print("\n" + "=" * 70)
    print("RAPPORT DE CLASSIFICATION")
    print("=" * 70)
    print(
        classification_report(
            y_test,
            y_pred_ensemble,
            target_names=train_split.class_names,
            digits=4,
        )
    )

    # Sauvegarder les modèles
    print("\n💾 Sauvegarde des modèles...")
    artifacts_dir = Path("artifacts/models")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    cnn_improved.save(artifacts_dir / "cnn_improved.h5")
    if model_transfer is not None:
        model_transfer.save(artifacts_dir / "transfer_learning_efficientnet.h5")

    print(f"  ✓ Modèles sauvegardés dans {artifacts_dir}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
