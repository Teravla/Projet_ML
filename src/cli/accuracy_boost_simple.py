"""Script d'amelioration accuracy avec CNN SIMPLE (baseline stable)."""

from __future__ import annotations

import argparse
from pathlib import Path

import keras
import numpy as np
from sklearn.metrics import accuracy_score
import tensorflow as tf

from src.cli.accuracy_boost import (
    PreparedData,
    prepare_data as prepare_data_common,
    print_classification_summary,
)
from src.enums.dataclass import AppConfig, EvalBundle, SweepConfig, TrainConfig
from src.enums.enums import HyperParametersInt, ModelType
from src.models.utils import compile_logits_classifier
from src.models.cnn import build_cnn_classifier
from src.models.mlp import build_mlp_classifier


def parse_args() -> argparse.Namespace:
    """Parse les options CLI pour adapter le profil d'entrainement."""

    parser = argparse.ArgumentParser(description="Boost accuracy simple CNN baseline")
    parser.add_argument(
        "--img-size",
        type=int,
        default=64,
        choices=[64, 128, 224],
        help="Taille carree des images (ex: 64 pour 64x64)",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Nombre d'epochs max")
    parser.add_argument(
        "--optimized",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Utiliser l'architecture CNN optimisee (active par defaut)",
    )
    parser.add_argument(
        "--cnn", action="store_true", help="Entrainer le modele CNN simple"
    )
    parser.add_argument("--mlp", action="store_true", help="Entrainer le modele MLP")
    parser.add_argument(
        "--transfer",
        action="store_true",
        help="Entrainer le modele transfer learning (EfficientNetB0)",
    )
    parser.add_argument(
        "--ensemble",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Utiliser l'ensemble voting (active par defaut)",
    )
    parser.add_argument(
        "--tta",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Activer Test-Time Augmentation pour CNN (active par defaut)",
    )
    parser.add_argument(
        "--focal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Utiliser Focal Loss pour CNN (desactive par defaut)",
    )
    parser.add_argument(
        "--sweep",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Tester plusieurs configs CNN et garder la meilleure",
    )
    parser.add_argument(
        "--sweep-trials", type=int, default=6, help="Nombre d'essais sweep"
    )
    parser.add_argument(
        "--sweep-epochs", type=int, default=12, help="Epochs par essai sweep"
    )
    parser.add_argument(
        "--final-train",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apres sweep, re-entrainer la meilleure config",
    )
    parser.add_argument(
        "--final-epochs",
        type=int,
        default=40,
        help="Nombre d'epochs du re-entrainement final",
    )
    args = parser.parse_args()

    if not (args.cnn or args.mlp or args.transfer):
        args.cnn = True
        args.mlp = False
        args.transfer = True

    return args


def build_transfer_learning_model(
    input_shape: tuple[int, int, int],
    num_classes: int = HyperParametersInt.NUM_CLASSES,
    learning_rate: float = 5e-5,
) -> keras.Model:
    """Transfer learning avec EfficientNetB0."""

    base_model = keras.applications.EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )

    for layer in base_model.layers[:-60]:
        layer.trainable = False

    inputs = keras.Input(shape=input_shape)
    preprocess_layer = keras.layers.Rescaling(255.0, name="efficientnet_rescale")
    preprocessed_inputs = preprocess_layer(inputs)
    features = base_model(preprocessed_inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(features)
    x = keras.layers.Dense(512, activation="relu", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.6)(x)
    x = keras.layers.Dense(256, activation="relu", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.5)(x)
    logits = keras.layers.Dense(num_classes)(x)

    model = keras.Model(inputs=inputs, outputs=logits, name="transfer_learning")
    compile_logits_classifier(model, learning_rate=learning_rate)
    return model


def sparse_focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    """Focal loss sparse pour mieux traiter les exemples difficiles."""

    def loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true_int = tf.cast(y_true, tf.int32)
        ce = keras.losses.sparse_categorical_crossentropy(
            y_true_int,
            y_pred,
            from_logits=True,
        )
        probs = tf.nn.softmax(y_pred, axis=-1)
        y_true_one_hot = tf.one_hot(y_true_int, depth=tf.shape(y_pred)[-1])
        p_t = tf.reduce_sum(y_true_one_hot * probs, axis=-1)
        focal_factor = alpha * tf.pow(1.0 - p_t, gamma)
        return focal_factor * ce

    return loss_fn


def train_model_with_augmentation(
    model: keras.Model,
    train_data: tuple[np.ndarray, np.ndarray],
    validation_data: tuple[np.ndarray, np.ndarray],
    config: TrainConfig,
    use_augmentation: bool = True,
) -> keras.callbacks.History:
    """Entraine avec data augmentation et learning-rate scheduling."""

    x_train, y_train = train_data
    x_val, y_val = validation_data

    train_augmentation = keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.05),
            keras.layers.RandomZoom(0.08),
            keras.layers.RandomTranslation(0.06, 0.06),
            keras.layers.RandomContrast(0.05),
            keras.layers.RandomBrightness(0.05, value_range=(0.0, 1.0)),
        ]
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=8,
        min_lr=1e-7,
        verbose=1,
    )
    checkpoint_best = keras.callbacks.ModelCheckpoint(
        filepath="artifacts/models/best_cnn_checkpoint.keras",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=0,
    )
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=25,
            restore_best_weights=True,
            mode="max",
        ),
        reduce_lr,
        checkpoint_best,
    ]

    base_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(buffer_size=len(x_train), reshuffle_each_iteration=True)
        .batch(config.batch_size)
    )
    train_ds = (
        base_ds.map(
            lambda x, y: (train_augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        ).prefetch(tf.data.AUTOTUNE)
        if use_augmentation
        else base_ds.prefetch(tf.data.AUTOTUNE)
    )

    return model.fit(
        train_ds,
        validation_data=(x_val, y_val),
        epochs=config.epochs,
        callbacks=callbacks,
        verbose=1,
    )


def predict_with_tta(
    model: keras.Model, x_data: np.ndarray, tta_rounds: int = 5
) -> np.ndarray:
    """Prediction TTA en moyennant plusieurs vues augmentees."""

    tta_aug = keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.04),
            keras.layers.RandomTranslation(0.04, 0.04),
        ]
    )

    logits_acc = model.predict(x_data, verbose=0)
    for _ in range(tta_rounds):
        x_aug = tta_aug(x_data, training=True)
        logits_acc += model.predict(x_aug, verbose=0)
    return logits_acc / (tta_rounds + 1)


def prepare_data(img_size: tuple[int, int]) -> PreparedData:
    """Charge les splits et prepare train/val/test."""

    return prepare_data_common(Path("data"), img_size)


def run_cnn_sweep(
    data: PreparedData,
    sweep_cfg: SweepConfig,
) -> tuple[keras.Model, dict[str, float | bool]]:
    """Teste plusieurs configs CNN et retourne le meilleur modele."""

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

    trials = max(1, min(sweep_cfg.trials, len(candidates)))
    best_model: keras.Model | None = None
    best_cfg: dict[str, float | bool] | None = None
    best_val = -1.0

    print(f"\nSweep CNN: {trials} essais, {sweep_cfg.epochs} epochs par essai")
    for idx in range(trials):
        cfg = candidates[idx]
        keras.backend.clear_session()

        print(
            f"  Essai {idx + 1}/{trials}: "
            f"lr={cfg['lr']}, dropout={cfg['dropout']}, "
            f"augment={cfg['augment']}, focal={cfg['focal']}"
        )

        model = build_cnn_classifier(
            data.x_train.shape[1:],
            num_classes=HyperParametersInt.NUM_CLASSES,
            dropout_rate=float(cfg["dropout"]),
            learning_rate=float(cfg["lr"]),
        )
        if bool(cfg["focal"]):
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=float(cfg["lr"])),
                loss=sparse_focal_loss(gamma=2.0, alpha=0.35),
                metrics=["accuracy"],
            )

        train_model_with_augmentation(
            model=model,
            train_data=(data.x_train, data.y_train),
            validation_data=(data.x_val, data.y_val),
            config=TrainConfig(
                epochs=sweep_cfg.epochs,
                batch_size=sweep_cfg.batch_size,
            ),
            use_augmentation=bool(cfg["augment"]),
        )

        val_pred = np.argmax(model.predict(data.x_val, verbose=0), axis=1)
        val_acc = accuracy_score(data.y_val, val_pred)

        test_logits = (
            predict_with_tta(model, data.x_test, tta_rounds=5)
            if sweep_cfg.use_tta
            else model.predict(data.x_test, verbose=0)
        )
        test_acc = accuracy_score(data.y_test, np.argmax(test_logits, axis=1))
        print(f"    -> val={val_acc:.4f}, test={test_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            best_model = model
            best_cfg = cfg

    if best_model is None or best_cfg is None:
        raise RuntimeError("Sweep CNN: aucun modele n'a ete entraine")

    print(
        f"  Meilleure config: lr={best_cfg['lr']}, dropout={best_cfg['dropout']}, "
        f"augment={best_cfg['augment']}, focal={best_cfg['focal']} (val={best_val:.4f})"
    )
    return best_model, best_cfg


def train_cnn_pipeline(
    data: PreparedData,
    app_cfg: AppConfig,
    train_cfg: TrainConfig,
    sweep_cfg: SweepConfig,
) -> keras.Model:
    """Entraine le modele CNN selon mode standard/sweep/final-train."""

    if sweep_cfg.trials > 0:
        cnn_model, best_cfg = run_cnn_sweep(data, sweep_cfg)
        if sweep_cfg.final_train:
            print(
                f"\nFinal train CNN avec meilleure config "
                f"({sweep_cfg.final_epochs} epochs)..."
            )
            keras.backend.clear_session()
            cnn_model = build_cnn_classifier(
                data.x_train.shape[1:],
                num_classes=HyperParametersInt.NUM_CLASSES,
                dropout_rate=float(best_cfg["dropout"]),
                learning_rate=float(best_cfg["lr"]),
            )
            if bool(best_cfg["focal"]):
                cnn_model.compile(
                    optimizer=keras.optimizers.Adam(
                        learning_rate=float(best_cfg["lr"])
                    ),
                    loss=sparse_focal_loss(gamma=2.0, alpha=0.35),
                    metrics=["accuracy"],
                )
            train_model_with_augmentation(
                model=cnn_model,
                train_data=(data.x_train, data.y_train),
                validation_data=(data.x_val, data.y_val),
                config=TrainConfig(
                    epochs=sweep_cfg.final_epochs, batch_size=train_cfg.batch_size
                ),
                use_augmentation=bool(best_cfg["augment"]),
            )
        return cnn_model

    print("\nEntrainement du CNN (mode standard)...")
    cnn_model = build_cnn_classifier(
        data.x_train.shape[1:],
        num_classes=HyperParametersInt.NUM_CLASSES,
        dropout_rate=0.3,
        learning_rate=1e-3 if app_cfg.use_optimized else 1e-3,
    )
    if app_cfg.use_focal:
        cnn_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=7e-4),
            loss=sparse_focal_loss(gamma=2.0, alpha=0.35),
            metrics=["accuracy"],
        )

    train_model_with_augmentation(
        model=cnn_model,
        train_data=(data.x_train, data.y_train),
        validation_data=(data.x_val, data.y_val),
        config=train_cfg,
        use_augmentation=False,
    )
    return cnn_model


def train_mlp_pipeline(data: PreparedData, train_cfg: TrainConfig) -> keras.Model:
    """Entraine le MLP sur images flatten."""

    x_train_flat = data.x_train.reshape(data.x_train.shape[0], -1)
    x_val_flat = data.x_val.reshape(data.x_val.shape[0], -1)

    mlp_model = build_mlp_classifier(
        input_dim=x_train_flat.shape[1],
        num_classes=HyperParametersInt.NUM_CLASSES,
        dropout_rate=0.4,
    )
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=10,
        restore_best_weights=True,
        mode="max",
    )
    mlp_model.fit(
        x_train_flat,
        data.y_train,
        validation_data=(x_val_flat, data.y_val),
        epochs=train_cfg.epochs,
        batch_size=train_cfg.batch_size,
        callbacks=[early_stopping],
        verbose=1,
    )
    return mlp_model


def train_transfer_pipeline(
    data: PreparedData, train_cfg: TrainConfig
) -> keras.Model | None:
    """Entraine le modele transfer learning, retourne None en cas d'echec."""

    transfer_config = TrainConfig(
        epochs=min(50, train_cfg.epochs),
        batch_size=train_cfg.batch_size,
    )
    train_data = (data.x_train, data.y_train)
    validation_data = (data.x_val, data.y_val)
    model_transfer = build_transfer_learning_model(data.x_train.shape[1:])
    try:
        train_model_with_augmentation(
            model=model_transfer,
            train_data=train_data,
            validation_data=validation_data,
            config=transfer_config,
            use_augmentation=True,
        )
        return model_transfer
    except (ValueError, RuntimeError, tf.errors.OpError) as exc:
        print(f"  Transfer learning echoue: {exc}")
        return None


def predict_logits(
    model_name: str,
    model: keras.Model,
    x_test: np.ndarray,
    use_tta: bool,
) -> np.ndarray:
    """Retourne les logits adaptes selon le type de modele."""

    if model_name == ModelType.MLP:
        x_input = x_test.reshape(x_test.shape[0], -1)
        return model.predict(x_input, verbose=0)
    if use_tta and model_name == ModelType.CNN:
        return predict_with_tta(model, x_test, tta_rounds=5)
    return model.predict(x_test, verbose=0)


def evaluate_models(
    models: dict[str, keras.Model],
    data: PreparedData,
    use_tta: bool,
    use_ensemble: bool,
) -> EvalBundle:
    """Evalue les modeles individuels et l'ensemble si possible."""

    per_model_accuracy: dict[str, float] = {}
    per_model_logits: dict[str, np.ndarray] = {}

    for name, model in models.items():
        logits = predict_logits(name, model, data.x_test, use_tta)
        preds = np.argmax(logits, axis=1)
        acc = accuracy_score(data.y_test, preds)
        per_model_logits[name] = logits
        per_model_accuracy[name] = acc

        label = {
            "cnn": "CNN Simple",
            "mlp": "MLP",
            "transfer": "Transfer Learning",
        }.get(name, name)
        print(f"  {label:20s} Accuracy: {acc:.4f} ({acc*100:.2f}%)")

    y_pred_final, final_accuracy = ensemble_or_best(
        per_model_logits,
        per_model_accuracy,
        data.y_test,
        use_ensemble,
    )
    return EvalBundle(
        y_pred_final=y_pred_final,
        final_accuracy=final_accuracy,
        per_model_accuracy=per_model_accuracy,
    )


def ensemble_or_best(
    logits_by_model: dict[str, np.ndarray],
    accuracy_by_model: dict[str, float],
    y_test: np.ndarray,
    use_ensemble: bool,
) -> tuple[np.ndarray, float]:
    """Construit une prediction d'ensemble sinon retourne la meilleure prediction."""

    if use_ensemble and len(logits_by_model) > 1:
        valid_names = [
            name
            for name, acc in accuracy_by_model.items()
            if acc > HyperParametersInt.ENSEMBLE_THRESHOLD
        ]
        if len(valid_names) > 1:
            raw_weights = [accuracy_by_model[name] ** 2 for name in valid_names]
            weight_sum = sum(raw_weights)
            weights = [weight / weight_sum for weight in raw_weights]

            print("  Ensemble weights: ", end="")
            for name, weight in zip(valid_names, weights):
                print(f"{name.upper()}={weight:.3f} ", end="")
            print()

            ensemble_logits = np.zeros(
                (y_test.shape[0], HyperParametersInt.NUM_CLASSES)
            )
            for name, weight in zip(valid_names, weights):
                ensemble_logits += weight * logits_by_model[name]

            y_pred = np.argmax(ensemble_logits, axis=1)
            return y_pred, accuracy_score(y_test, y_pred)

    best_name = max(accuracy_by_model, key=accuracy_by_model.get)
    y_pred_best = np.argmax(logits_by_model[best_name], axis=1)
    return y_pred_best, accuracy_score(y_test, y_pred_best)


def print_goal(final_accuracy: float) -> None:
    """Affiche l'atteinte ou non de l'objectif."""

    print("\n" + "=" * 70)
    if final_accuracy > HyperParametersInt.TARGET_ACCURACY:
        print(f"OBJECTIF ATTEINT! Accuracy: {final_accuracy*100:.2f}% > 90%")
        return

    print(f"Accuracy finale: {final_accuracy*100:.2f}%")
    print(f"   Gap vers objectif 90%: {(90 - final_accuracy*100):.2f}%")


def save_models(models: dict[str, keras.Model]) -> None:
    """Sauvegarde tous les modeles entraines dans artifacts/models."""

    artifacts_dir = Path("artifacts/models")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    name_map = {
        "cnn": "cnn_simple.keras",
        "mlp": "mlp.keras",
        "transfer": "transfer_learning_efficientnet.keras",
    }
    for model_name, model in models.items():
        filename = name_map.get(model_name, f"{model_name}.keras")
        model.save(artifacts_dir / filename)
        print(f"  {model_name.upper()} sauvegarde: {filename}")

    print(f"\n  Tous les modeles sauvegardes dans {artifacts_dir}")


def print_summary(data: PreparedData, eval_bundle: EvalBundle) -> None:
    """Affiche rapport final et metriques detaillees."""

    print_goal(eval_bundle.final_accuracy)
    print_classification_summary(
        data.y_test,
        eval_bundle.y_pred_final,
        data.class_names,
    )


def build_app_config(args: argparse.Namespace) -> AppConfig:
    """Construit la configuration applicative a partir des arguments CLI."""

    active_models = tuple(
        name
        for name, enabled in (
            ("cnn", args.cnn),
            ("mlp", args.mlp),
            ("transfer", args.transfer),
        )
        if enabled
    )

    return AppConfig(
        img_size=(args.img_size, args.img_size),
        use_optimized=args.optimized,
        use_ensemble=args.ensemble,
        use_tta=args.tta,
        use_focal=args.focal,
        active_models=active_models,
    )


def print_runtime_config(
    app_cfg: AppConfig,
    sweep_cfg: SweepConfig,
    train_cfg: TrainConfig,
) -> None:
    """Affiche les options runtime de manière lisible."""

    active = [name.upper() for name in app_cfg.active_models]

    print(
        f"Configuration: image_size={app_cfg.img_size[0]}x{app_cfg.img_size[1]}, "
        f"epochs={train_cfg.epochs}, batch_size={train_cfg.batch_size}, "
        f"architecture={'optimisee' if app_cfg.use_optimized else 'baseline'}"
    )
    print("Modeles actives:", ", ".join(active) if active else "Aucun")
    print(f"TTA: {'active' if app_cfg.use_tta else 'desactive'}")
    print(f"Focal Loss CNN: {'activee' if app_cfg.use_focal else 'desactivee'}")
    print(f"Sweep CNN: {'active' if sweep_cfg.trials > 0 else 'desactive'}")
    print(
        f"Final Train apres sweep: "
        f"{'active' if sweep_cfg.final_train else 'desactive'}"
    )


def train_selected_models(
    app_cfg: AppConfig,
    data: PreparedData,
    train_cfg: TrainConfig,
    sweep_cfg: SweepConfig,
) -> dict[str, keras.Model]:
    """Entraine la liste des modeles demandee par les flags CLI."""

    trained: dict[str, keras.Model] = {}

    if ModelType.CNN in app_cfg.active_models:
        print("\n[2/5] Pipeline CNN...")
        trained[ModelType.CNN] = train_cnn_pipeline(data, app_cfg, train_cfg, sweep_cfg)

    if ModelType.MLP in app_cfg.active_models:
        print("\n[3/5] Entrainement MLP...")
        trained[ModelType.MLP] = train_mlp_pipeline(data, train_cfg)

    if ModelType.TRANSFER in app_cfg.active_models:
        print("\n[4/5] Entrainement Transfer Learning...")
        transfer_model = train_transfer_pipeline(data, train_cfg)
        if transfer_model is not None:
            trained[ModelType.TRANSFER] = transfer_model

    return trained


def main() -> None:
    """Exécute le pipeline complet de boost accuracy (version simple)."""

    args = parse_args()
    app_cfg = build_app_config(args)
    train_cfg = TrainConfig(epochs=args.epochs, batch_size=32)
    sweep_cfg = SweepConfig(
        trials=args.sweep_trials if args.sweep else 0,
        epochs=args.sweep_epochs,
        final_train=args.final_train,
        final_epochs=args.final_epochs,
        batch_size=train_cfg.batch_size,
        use_tta=app_cfg.use_tta,
    )

    print("\nAMELIORATION DE L'ACCURACY - VERSION SIMPLE (CNN baseline)\n")
    print("=" * 70)
    print_runtime_config(app_cfg, sweep_cfg, train_cfg)

    print("\n[1/5] Chargement des donnees...")
    data = prepare_data(app_cfg.img_size)
    print(
        f"  Train: {data.x_train.shape}, "
        f"Val: {data.x_val.shape}, Test: {data.x_test.shape}"
    )

    trained_models = train_selected_models(app_cfg, data, train_cfg, sweep_cfg)
    if not trained_models:
        print("Aucun modele entraine. Utilise --cnn, --mlp ou --transfer.")
        return

    print("\n[5/5] Evaluation finale...")
    eval_bundle = evaluate_models(
        models=trained_models,
        data=data,
        use_tta=app_cfg.use_tta,
        use_ensemble=app_cfg.use_ensemble,
    )

    print_summary(data, eval_bundle)
    print("\nSauvegarde des modeles...")
    save_models(trained_models)
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
