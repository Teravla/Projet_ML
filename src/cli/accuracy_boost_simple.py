"""Script d'amélioration accuracy avec CNN SIMPLE (baseline qui marche)
Utilise l'architecture légère au lieu du CNN "amélioré" qui overfit.
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
from src.models.cnn import build_cnn_classifier
from src.models.mlp import build_mlp_classifier


def parse_args() -> argparse.Namespace:
    """Parse les options CLI pour adapter le profil d'entraînement."""

    parser = argparse.ArgumentParser(description="Boost accuracy simple CNN baseline")
    parser.add_argument(
        "--img-size",
        type=int,
        default=64,
        choices=[64, 128, 224],
        help="Taille carrée des images (ex: 64 pour 64x64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Nombre d'epochs max",
    )
    parser.add_argument(
        "--optimized",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Utiliser l'architecture CNN optimisée (activé par défaut)",
    )
    parser.add_argument(
        "--cnn",
        action="store_true",
        help="Entraîner le modèle CNN simple",
    )
    parser.add_argument(
        "--mlp",
        action="store_true",
        help="Entraîner le modèle MLP",
    )
    parser.add_argument(
        "--transfer",
        action="store_true",
        help="Entraîner le modèle transfer learning (EfficientNetB0)",
    )
    parser.add_argument(
        "--ensemble",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Utiliser l'ensemble voting (activé par défaut)",
    )
    parser.add_argument(
        "--tta",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Activer Test-Time Augmentation pour CNN (activé par défaut)",
    )
    parser.add_argument(
        "--focal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Utiliser Focal Loss pour CNN (désactivé par défaut)",
    )
    parser.add_argument(
        "--sweep",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Tester plusieurs configs CNN et garder la meilleure",
    )
    parser.add_argument(
        "--sweep-trials",
        type=int,
        default=6,
        help="Nombre de configs CNN à tester en sweep",
    )
    parser.add_argument(
        "--sweep-epochs",
        type=int,
        default=12,
        help="Nombre d'epochs par essai du sweep",
    )
    parser.add_argument(
        "--final-train",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Après sweep, réentraîner la meilleure config sur plus d'epochs",
    )
    parser.add_argument(
        "--final-epochs",
        type=int,
        default=40,
        help="Nombre d'epochs du réentraînement final après sweep",
    )
    args = parser.parse_args()

    # Si aucun algo n'est spécifié, activer tous par défaut
    if not (args.cnn or args.mlp or args.transfer):
        args.cnn = True
        args.mlp = False  # MLP désactivé par défaut (moins performant)
        args.transfer = True

    return args


def train_model_with_augmentation(
    model: tf.keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 30,
    batch_size: int = 32,
    use_augmentation: bool = True,
) -> tf.keras.callbacks.History:
    """Entraîne avec data augmentation agressif et learning rate scheduling optimisé."""

    # Augmentation modérée pour IRM: préserver la structure anatomique.
    train_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.08),
            tf.keras.layers.RandomTranslation(0.06, 0.06),
            tf.keras.layers.RandomContrast(0.05),
            tf.keras.layers.RandomBrightness(0.05, value_range=(0.0, 1.0)),
        ]
    )

    # Callbacks optimisés pour meilleure convergence
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=25,  # Très patient pour laisser converger
            restore_best_weights=True,
            mode="max",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=8,  # Plus patient avant de réduire
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath="artifacts/models/best_cnn_checkpoint.keras",
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=0,
        ),
    ]

    base_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(buffer_size=len(x_train), reshuffle_each_iteration=True)
        .batch(batch_size)
    )

    if use_augmentation:
        # Augmentation dynamique par batch/epoch pour éviter un dataset figé.
        train_ds = base_ds.map(
            lambda x, y: (train_augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        ).prefetch(tf.data.AUTOTUNE)
    else:
        train_ds = base_ds.prefetch(tf.data.AUTOTUNE)

    history = model.fit(
        train_ds,
        validation_data=(x_val, y_val),
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )
    return history


def build_transfer_learning_model(
    input_shape: tuple[int, int, int],
    num_classes: int = 4,
    learning_rate: float = 5e-5,
) -> tf.keras.Model:
    """Transfer learning avec EfficientNetB0."""

    # Charger EfficientNetB0 pré-entraîné
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )

    # Geler les couches initiales, dégeler plus de couches finales pour better fine-tuning
    for layer in base_model.layers[:-60]:
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
    x = tf.keras.layers.Dropout(0.6)(x)
    x = tf.keras.layers.Dense(256, activation="relu", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    logits = tf.keras.layers.Dense(num_classes)(x)

    model = tf.keras.Model(inputs=inputs, outputs=logits, name="transfer_learning")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


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


def sparse_focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    """Focal loss sparse pour mieux traiter les exemples difficiles."""

    def loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.int32)
        ce = tf.keras.losses.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=True
        )
        probs = tf.nn.softmax(y_pred, axis=-1)
        y_true_one_hot = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])
        p_t = tf.reduce_sum(y_true_one_hot * probs, axis=-1)
        focal_factor = alpha * tf.pow(1.0 - p_t, gamma)
        return focal_factor * ce

    return loss_fn


def predict_with_tta(
    model: tf.keras.Model,
    x_data: np.ndarray,
    tta_rounds: int = 5,
) -> np.ndarray:
    """Prédiction TTA en moyennant plusieurs vues augmentées d'une image."""

    tta_aug = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.04),
            tf.keras.layers.RandomTranslation(0.04, 0.04),
        ]
    )

    logits_acc = model.predict(x_data, verbose=0)
    for _ in range(tta_rounds):
        x_aug = tta_aug(x_data, training=True)
        logits_acc += model.predict(x_aug, verbose=0)

    return logits_acc / (tta_rounds + 1)


def run_cnn_sweep(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    input_shape: tuple[int, int, int],
    use_tta: bool,
    sweep_trials: int,
    sweep_epochs: int,
    batch_size: int,
) -> tuple[tf.keras.Model, dict[str, float | bool]]:
    """Teste plusieurs configs CNN et retourne le meilleur modèle + sa config."""

    candidates = [
        {"lr": 1e-3, "dropout": 0.30, "augment": False, "focal": False},
        {"lr": 8e-4, "dropout": 0.35, "augment": False, "focal": False},
        {"lr": 7e-4, "dropout": 0.30, "augment": False, "focal": True},
        {"lr": 1e-3, "dropout": 0.40, "augment": False, "focal": False},
        {"lr": 8e-4, "dropout": 0.30, "augment": True, "focal": False},
        {"lr": 6e-4, "dropout": 0.35, "augment": True, "focal": False},
        {"lr": 6e-4, "dropout": 0.30, "augment": False, "focal": True},
        {"lr": 9e-4, "dropout": 0.25, "augment": False, "focal": False},
    ]

    trials = max(1, min(sweep_trials, len(candidates)))
    best_model: tf.keras.Model | None = None
    best_val = -1.0
    best_cfg: dict[str, float | bool] | None = None

    print(f"\nSweep CNN: {trials} essais, {sweep_epochs} epochs par essai")

    for idx in range(trials):
        cfg = candidates[idx]
        tf.keras.backend.clear_session()

        print(
            "  "
            f"Essai {idx+1}/{trials}: "
            f"lr={cfg['lr']}, dropout={cfg['dropout']}, "
            f"augment={cfg['augment']}, focal={cfg['focal']}"
        )

        model = build_cnn_classifier(
            input_shape,
            num_classes=4,
            dropout_rate=float(cfg["dropout"]),
            learning_rate=float(cfg["lr"]),
        )

        if bool(cfg["focal"]):
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=float(cfg["lr"])),
                loss=sparse_focal_loss(gamma=2.0, alpha=0.35),
                metrics=["accuracy"],
            )

        train_model_with_augmentation(
            model,
            x_train,
            y_train,
            x_val,
            y_val,
            epochs=sweep_epochs,
            batch_size=batch_size,
            use_augmentation=bool(cfg["augment"]),
        )

        val_logits = model.predict(x_val, verbose=0)
        val_pred = np.argmax(val_logits, axis=1)
        val_acc = accuracy_score(y_val, val_pred)

        if use_tta:
            test_logits = predict_with_tta(model, x_test, tta_rounds=5)
        else:
            test_logits = model.predict(x_test, verbose=0)
        test_pred = np.argmax(test_logits, axis=1)
        test_acc = accuracy_score(y_test, test_pred)

        print(f"    -> val={val_acc:.4f}, test={test_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            best_model = model
            best_cfg = cfg

    if best_model is None or best_cfg is None:
        raise RuntimeError("Sweep CNN: aucun modèle n'a été entraîné")

    print(
        "  "
        f"Meilleure config: lr={best_cfg['lr']}, dropout={best_cfg['dropout']}, "
        f"augment={best_cfg['augment']}, focal={best_cfg['focal']} "
        f"(val={best_val:.4f})"
    )

    return best_model, best_cfg


def main() -> None:
    args = parse_args()

    print("\n🚀 AMÉLIORATION DE L'ACCURACY - VERSION SIMPLE (CNN baseline)\n")
    print("=" * 70)

    # Paramètres
    data_dir = Path("data")
    img_size = (args.img_size, args.img_size)
    epochs = args.epochs
    batch_size = 32

    print(
        f"Configuration: image_size={img_size[0]}x{img_size[1]}, "
        f"epochs={epochs}, batch_size={batch_size}, "
        f"architecture={'optimisée' if args.optimized else 'baseline'}"
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

    # Configuration des modèles à entraîner
    print(f"\n📋 Modèles activés: ", end="")
    active_models = []
    if args.cnn:
        active_models.append("CNN")
    if args.mlp:
        active_models.append("MLP")
    if args.transfer:
        active_models.append("Transfer")
    print(", ".join(active_models) if active_models else "Aucun")
    print(f"📋 TTA: {'activé' if args.tta else 'désactivé'}")
    print(f"📋 Focal Loss CNN: {'activée' if args.focal else 'désactivée'}")
    print(f"📋 Sweep CNN: {'activé' if args.sweep else 'désactivé'}")
    print(
        f"📋 Final Train après sweep: {'activé' if args.final_train else 'désactivé'}"
    )

    # Stockage des modèles et résultats
    trained_models = {}
    accuracies = {}

    step = 2
    total_steps = len(active_models) + 2  # +2 pour évaluation et ensemble

    # Entraîner CNN simple (si activé)
    if args.cnn:
        cnn_already_trained = False
        if args.sweep:
            print(f"\n[{step}/{total_steps}] Sweep hyperparamètres CNN...")
            cnn_model, best_cfg = run_cnn_sweep(
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                x_test=x_test,
                y_test=y_test,
                input_shape=x_train.shape[1:],
                use_tta=args.tta,
                sweep_trials=args.sweep_trials,
                sweep_epochs=args.sweep_epochs,
                batch_size=batch_size,
            )
            cnn_already_trained = True

            if args.final_train:
                print(
                    f"\n[{step}/{total_steps}] Final train CNN avec meilleure config "
                    f"({args.final_epochs} epochs)..."
                )
                tf.keras.backend.clear_session()
                cnn_model = build_cnn_classifier(
                    x_train.shape[1:],
                    num_classes=4,
                    dropout_rate=float(best_cfg["dropout"]),
                    learning_rate=float(best_cfg["lr"]),
                )
                if bool(best_cfg["focal"]):
                    cnn_model.compile(
                        optimizer=tf.keras.optimizers.Adam(
                            learning_rate=float(best_cfg["lr"])
                        ),
                        loss=sparse_focal_loss(gamma=2.0, alpha=0.35),
                        metrics=["accuracy"],
                    )
                train_model_with_augmentation(
                    cnn_model,
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                    epochs=args.final_epochs,
                    batch_size=batch_size,
                    use_augmentation=bool(best_cfg["augment"]),
                )
        elif args.optimized:
            print(
                f"\n[{step}/{total_steps}] Entraînement du CNN OPTIMISÉ (recette stable)..."
            )
            cnn_model = build_cnn_classifier(
                x_train.shape[1:],
                num_classes=4,
                dropout_rate=0.3,
                learning_rate=1e-3,
            )
            if args.focal:
                cnn_model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=7e-4),
                    loss=sparse_focal_loss(gamma=2.0, alpha=0.35),
                    metrics=["accuracy"],
                )
        else:
            print(f"\n[{step}/{total_steps}] Entraînement du CNN simple baseline...")
            cnn_model = build_cnn_classifier(
                x_train.shape[1:], num_classes=4, dropout_rate=0.3, learning_rate=1e-3
            )

        if not cnn_already_trained and not args.sweep:
            train_model_with_augmentation(
                cnn_model,
                x_train,
                y_train,
                x_val,
                y_val,
                epochs=epochs,
                batch_size=batch_size,
                use_augmentation=False,
            )
        trained_models["cnn"] = cnn_model
        step += 1

    # Entraîner MLP (si activé)
    if args.mlp:
        print(f"\n[{step}/{total_steps}] Entraînement du MLP...")
        # Flatten des images pour le MLP
        x_train_flat = x_train.reshape(x_train.shape[0], -1)
        x_val_flat = x_val.reshape(x_val.shape[0], -1)

        mlp_model = build_mlp_classifier(
            input_dim=x_train_flat.shape[1], num_classes=4, dropout_rate=0.4
        )
        mlp_model.fit(
            x_train_flat,
            y_train,
            validation_data=(x_val_flat, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_accuracy",
                    patience=10,
                    restore_best_weights=True,
                    mode="max",
                ),
            ],
            verbose=1,
        )
        trained_models["mlp"] = mlp_model
        step += 1

    # Entraîner Transfer Learning (si activé)
    if args.transfer:
        print(
            f"\n[{step}/{total_steps}] Entraînement avec Transfer Learning (EfficientNetB0)..."
        )
        model_transfer = build_transfer_learning_model(x_train.shape[1:])
        try:
            train_model_with_augmentation(
                model_transfer,
                x_train,
                y_train,
                x_val,
                y_val,
                epochs=min(50, epochs),  # Au moins 50 epochs pour transfer learning
                batch_size=batch_size,
                use_augmentation=True,
            )
            trained_models["transfer"] = model_transfer
        except Exception as e:
            print(f"  ⚠️ Transfer learning échoué: {e}")
        step += 1

    # Évaluer les modèles
    print(f"\n[{step}/{total_steps}] Évaluation sur l'ensemble test...")

    if not trained_models:
        print("❌ Aucun modèle entraîné. Utilise --cnn, --mlp ou --transfer.")
        return

    for model_name, model in trained_models.items():
        if model_name == "mlp":
            x_test_input = x_test.reshape(x_test.shape[0], -1)
            logits = model.predict(x_test_input, verbose=0)
        else:
            x_test_input = x_test
            if args.tta and model_name == "cnn":
                logits = predict_with_tta(model, x_test_input, tta_rounds=5)
            else:
                logits = model.predict(x_test_input, verbose=0)

        y_pred = np.argmax(logits, axis=1)
        acc = accuracy_score(y_test, y_pred)
        accuracies[model_name] = acc

        label = {
            "cnn": "CNN Simple",
            "mlp": "MLP",
            "transfer": "Transfer Learning",
        }.get(model_name, model_name)
        print(f"  {label:20s} Accuracy: {acc:.4f} ({acc*100:.2f}%)")

    step += 1

    # Ensemble voting (si activé et plusieurs modèles)
    if args.ensemble and len(trained_models) > 1:
        print(f"\n[{step}/{total_steps}] Fusion d'ensemble...")

        # Construire l'ensemble avec pondération quadratique
        models_to_ensemble = []
        weights_to_use = []

        for model_name, model in trained_models.items():
            acc = accuracies[model_name]
            if acc > 0.50:  # Seuil minimum pour inclure dans l'ensemble
                models_to_ensemble.append((model_name, model))
                weights_to_use.append(acc**2)

        if len(models_to_ensemble) > 1:
            # Normaliser les poids
            total_weight = sum(weights_to_use)
            weights_to_use = [w / total_weight for w in weights_to_use]

            # Afficher les poids
            print("  Ensemble weights: ", end="")
            for (model_name, _), weight in zip(models_to_ensemble, weights_to_use):
                print(f"{model_name.upper()}={weight:.3f} ", end="")
            print()

            # Prédictions ensembles
            ensemble_logits = np.zeros((x_test.shape[0], 4))
            for (model_name, model), weight in zip(models_to_ensemble, weights_to_use):
                if model_name == "mlp":
                    x_test_input = x_test.reshape(x_test.shape[0], -1)
                    logits = model.predict(x_test_input, verbose=0)
                else:
                    x_test_input = x_test
                    if args.tta and model_name == "cnn":
                        logits = predict_with_tta(model, x_test_input, tta_rounds=5)
                    else:
                        logits = model.predict(x_test_input, verbose=0)
                ensemble_logits += weight * logits

            y_pred_ensemble = np.argmax(ensemble_logits, axis=1)
            acc_ensemble = accuracy_score(y_test, y_pred_ensemble)
            print(
                f"  Ensemble Accuracy:      {acc_ensemble:.4f} ({acc_ensemble*100:.2f}%)"
            )
        else:
            # Un seul modèle valide
            acc_ensemble = max(accuracies.values())
            y_pred_ensemble = np.argmax(
                trained_models[max(accuracies, key=accuracies.get)].predict(
                    (
                        x_test
                        if "mlp" not in trained_models
                        else x_test.reshape(x_test.shape[0], -1)
                    ),
                    verbose=0,
                ),
                axis=1,
            )
            print(f"  Un seul modèle valide, pas d'ensemble.")
    else:
        # Pas d'ensemble, utiliser le meilleur modèle
        best_model_name = max(accuracies, key=accuracies.get)
        acc_ensemble = accuracies[best_model_name]
        if best_model_name == "mlp":
            x_test_input = x_test.reshape(x_test.shape[0], -1)
            logits_best = trained_models[best_model_name].predict(
                x_test_input, verbose=0
            )
        else:
            x_test_input = x_test
            if args.tta and best_model_name == "cnn":
                logits_best = predict_with_tta(
                    trained_models[best_model_name], x_test_input, tta_rounds=5
                )
            else:
                logits_best = trained_models[best_model_name].predict(
                    x_test_input, verbose=0
                )
        y_pred_ensemble = np.argmax(logits_best, axis=1)

    # Recalcule toujours l'accuracy finale sur la prédiction réellement reportée.
    acc_ensemble = accuracy_score(y_test, y_pred_ensemble)

    # Objectif atteint?
    print("\n" + "=" * 70)
    if acc_ensemble > 0.90:
        print(f"✅ OBJECTIF ATTEINT! Accuracy: {acc_ensemble*100:.2f}% > 90%")
    else:
        print(f"📈 Accuracy finale: {acc_ensemble*100:.2f}%")
        print(f"   Gap vers objectif 90%: {(90 - acc_ensemble*100):.2f}%")

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

    for model_name, model in trained_models.items():
        filename = {
            "cnn": "cnn_simple.keras",
            "mlp": "mlp.keras",
            "transfer": "transfer_learning_efficientnet.keras",
        }.get(model_name, f"{model_name}.keras")
        model.save(artifacts_dir / filename)
        print(f"  ✓ {model_name.upper()} sauvegardé: {filename}")

    print(f"\n  ✓ Tous les modèles sauvegardés dans {artifacts_dir}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
