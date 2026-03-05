"""Script d'exécution de la Tâche 4: CNN + Temperature Scaling + activations."""

# pylint: disable=duplicate-code

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from src.data.loader import load_dataset_split
from src.data.preprocess import preprocess_dataset
from src.models.calibration import (
    TemperatureScaler,
    analyze_uncertain_predictions,
    summarize_confidence_distribution,
)
from src.models.cnn import (
    CNNTrainingConfig,
    build_cnn_classifier,
    extract_intermediate_activations,
    predict_logits,
    predict_probabilities,
    train_cnn_classifier,
)


@dataclass(frozen=True)
class RuntimeConfig:
    """Paramètres effectifs d'exécution de la tâche 4."""

    img_size: int
    epochs: int
    batch_size: int
    ts_epochs: int
    mode: str


@dataclass(frozen=True)
class DataBundle:
    """Données préparées pour entraînement/évaluation CNN."""

    x_train: np.ndarray
    x_val: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    class_names: list[str]


@dataclass(frozen=True)
class EvalBundle:
    """Résultats d'évaluation du CNN calibré."""

    acc_base: float
    acc_calibrated: float
    temperature: float
    confidence_stats: dict[str, float]
    uncertain_stats: dict[str, float]


def parse_args() -> argparse.Namespace:
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Tâche 4 - CNN optimisé pour la décision avec Temperature Scaling"
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout-rate", type=float, default=0.3)
    parser.add_argument("--ts-epochs", type=int, default=300)
    parser.add_argument("--uncertainty-threshold", type=float, default=0.7)
    parser.add_argument("--activation-samples", type=int, default=128)
    parser.add_argument(
        "--activations-path",
        type=str,
        default="artifacts/models/cnn_task4_activations.npz",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Mode rapide: img-size<=32, epochs<=3, batch-size>=128, ts-epochs<=100",
    )
    return parser.parse_args()


def resolve_runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    """Construit la configuration effective (FAST/STD)."""
    img_size = args.img_size
    epochs = args.epochs
    batch_size = args.batch_size
    ts_epochs = args.ts_epochs
    mode = "FAST" if args.fast else "STANDARD"

    if args.fast:
        print("Mode FAST activé: réduction du coût d'entraînement CNN")
        img_size = min(img_size, 32)
        epochs = min(epochs, 3)
        batch_size = max(batch_size, 128)
        ts_epochs = min(ts_epochs, 100)

    return RuntimeConfig(
        img_size=img_size,
        epochs=epochs,
        batch_size=batch_size,
        ts_epochs=ts_epochs,
        mode=mode,
    )


def load_data_bundle(data_dir: str, runtime: RuntimeConfig) -> DataBundle:
    """Charge et prépare les données train/val/test."""
    train_dir = Path(data_dir) / "Training"
    test_dir = Path(data_dir) / "Testing"
    image_size = (runtime.img_size, runtime.img_size)

    train_split = load_dataset_split(train_dir, image_size=image_size)
    test_split = load_dataset_split(
        test_dir, image_size=image_size, class_names=train_split.class_names
    )

    x_train_all, y_train_all = preprocess_dataset(
        train_split.images,
        train_split.labels,
        target_size=image_size,
        normalize=True,
        one_hot=False,
    )
    x_test, y_test = preprocess_dataset(
        test_split.images,
        test_split.labels,
        target_size=image_size,
        normalize=True,
        one_hot=False,
    )

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_all,
        y_train_all,
        test_size=0.2,
        random_state=42,
        stratify=y_train_all,
    )

    return DataBundle(
        x_train=x_train,
        x_val=x_val,
        y_train=y_train,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        class_names=train_split.class_names,
    )


def train_cnn_model(
    data: DataBundle,
    args: argparse.Namespace,
    runtime: RuntimeConfig,
) -> object:
    """Construit et entraîne le CNN."""
    model = build_cnn_classifier(
        input_shape=(runtime.img_size, runtime.img_size, 3),
        num_classes=len(data.class_names),
        dropout_rate=args.dropout_rate,
        learning_rate=args.learning_rate,
    )

    train_cnn_classifier(
        model=model,
        train_data=(data.x_train, data.y_train),
        validation_data=(data.x_val, data.y_val),
        training_config=CNNTrainingConfig(
            epochs=runtime.epochs,
            batch_size=runtime.batch_size,
            learning_rate=args.learning_rate,
            dropout_rate=args.dropout_rate,
        ),
        verbose=1,
    )
    return model


def evaluate_calibrated_cnn(
    model: object,
    data: DataBundle,
    args: argparse.Namespace,
    runtime: RuntimeConfig,
) -> tuple[EvalBundle, TemperatureScaler, np.ndarray]:
    """Évalue le CNN avant/après calibration par température."""
    probs_base = predict_probabilities(model, data.x_test)
    acc_base = accuracy_score(data.y_test, np.argmax(probs_base, axis=1))

    val_logits = predict_logits(model, data.x_val)
    test_logits = predict_logits(model, data.x_test)

    temp_scaler = TemperatureScaler(initial_temperature=1.0)
    temp_scaler.fit(
        logits=val_logits,
        labels=data.y_val,
        learning_rate=1e-2,
        epochs=runtime.ts_epochs,
        verbose=0,
    )

    probs_calibrated = temp_scaler.predict_proba(test_logits)
    acc_calibrated = accuracy_score(data.y_test, np.argmax(probs_calibrated, axis=1))
    max_prob = np.max(probs_calibrated, axis=1)

    eval_bundle = EvalBundle(
        acc_base=acc_base,
        acc_calibrated=acc_calibrated,
        temperature=temp_scaler.temperature,
        confidence_stats=summarize_confidence_distribution(max_prob),
        uncertain_stats=analyze_uncertain_predictions(
            max_probabilities=max_prob,
            threshold=args.uncertainty_threshold,
        ),
    )
    return eval_bundle, temp_scaler, probs_calibrated


def save_activations(
    model: object,
    data: DataBundle,
    args: argparse.Namespace,
    runtime: RuntimeConfig,
) -> tuple[Path, list[str], dict[str, np.ndarray], int]:
    """Extrait et sauvegarde les activations intermédiaires."""
    activation_count = min(args.activation_samples, data.x_test.shape[0])
    activation_layers = ["conv1", "conv2", "conv3", "dense1"]
    activations = extract_intermediate_activations(
        model=model,
        x_data=data.x_test[:activation_count],
        layer_names=activation_layers,
        batch_size=runtime.batch_size,
    )

    output_path = Path(args.activations_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        sample_images=data.x_test[:activation_count],
        sample_labels=data.y_test[:activation_count],
        class_names=np.array(data.class_names),
        **{f"activation_{layer}": value for layer, value in activations.items()},
    )
    return output_path, activation_layers, activations, activation_count


def main() -> None:
    """Point d'entrée principal pour la tâche 4."""
    args = parse_args()
    runtime = resolve_runtime_config(args)
    data = load_data_bundle(args.data_dir, runtime)

    print("--- Tâche 4: CNN + Temperature Scaling ---")
    print(
        f"Mode={runtime.mode} | img_size={runtime.img_size} | "
        f"epochs={runtime.epochs} | batch_size={runtime.batch_size} | "
        f"ts_epochs={runtime.ts_epochs}"
    )
    print(
        f"Train={data.x_train.shape}, Val={data.x_val.shape}, Test={data.x_test.shape}"
    )

    model = train_cnn_model(data, args, runtime)
    eval_bundle, _temp_scaler, _probs = evaluate_calibrated_cnn(
        model=model,
        data=data,
        args=args,
        runtime=runtime,
    )
    output_path, activation_layers, activations, _count = save_activations(
        model=model,
        data=data,
        args=args,
        runtime=runtime,
    )

    print("\n--- Accuracy ---")
    print(f"CNN base accuracy                : {eval_bundle.acc_base:.4f}")
    print(f"CNN + Temperature Scaling acc    : {eval_bundle.acc_calibrated:.4f}")
    print(f"Température apprise              : {eval_bundle.temperature:.4f}")

    print("\n--- Confiance calibrée ---")
    stats = eval_bundle.confidence_stats
    print(
        f"p25={stats['p25']:.3f} | p50={stats['p50']:.3f} | "
        f"p75={stats['p75']:.3f} | mean={stats['mean']:.3f} | std={stats['std']:.3f}"
    )
    print(
        f"Seuil={eval_bundle.uncertain_stats['threshold']:.2f} | "
        f"Incertaines={eval_bundle.uncertain_stats['uncertain_count']}/"
        f"{eval_bundle.uncertain_stats['total']} "
        f"({100 * eval_bundle.uncertain_stats['uncertain_ratio']:.2f}%)"
    )

    print("\n--- Activations sauvegardées ---")
    print(f"Fichier: {output_path}")
    for layer in activation_layers:
        print(f"- {layer:<8}: {activations[layer].shape}")


if __name__ == "__main__":
    main()
