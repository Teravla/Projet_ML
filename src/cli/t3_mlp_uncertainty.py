"""Script d'exécution de la Tâche 3: MLP probabiliste + incertitude."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import accuracy_score
from src.data.pipeline import (
    TrainValTestData,
    load_preprocessed_train_test,
    split_train_validation,
)
from src.enums.dataclass import RuntimeConfigT3
from src.models.log_reg import flatten_images
from src.models.mlp import (
    MLPTrainingConfig,
    MLPTrainingData,
    build_mlp_classifier,
    predict_probabilities,
    train_mlp_classifier,
)
from src.models.uncertainty import mc_dropout_predict, summarize_uncertainty


DataBundle = TrainValTestData


@dataclass(frozen=True)
class EvalBundle:
    """Résumé des métriques de performance/incertitude."""

    deterministic_acc: float
    mc_acc: float
    uncertainty_stats: dict[str, float]
    entropy_mean: float
    entropy_std: float


def parse_args() -> argparse.Namespace:
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Task 3 - Probabilistic MLP with Uncertainty Management"
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout-rate", type=float, default=0.3)
    parser.add_argument("--n-iter", type=int, default=20, help="itérations MC Dropout")
    parser.add_argument("--uncertainty-threshold", type=float, default=0.7)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Mode rapide: img-size<=32, epochs<=5, batch-size>=128, n-iter<=10",
    )
    return parser.parse_args()


def resolve_runtime_config(args: argparse.Namespace) -> RuntimeConfigT3:
    """Construit la configuration effective (FAST/STD)."""
    img_size = args.img_size
    epochs = args.epochs
    batch_size = args.batch_size
    n_iter = args.n_iter
    hidden_units = (256, 128)
    mode = "FAST" if args.fast else "STANDARD"

    if args.fast:
        print("Mode FAST activé: réduction de la charge d'entraînement MLP")
        img_size = min(img_size, 32)
        epochs = min(epochs, 5)
        batch_size = max(batch_size, 128)
        n_iter = min(n_iter, 10)
        hidden_units = (128, 64)

    return RuntimeConfigT3(
        img_size=img_size,
        epochs=epochs,
        batch_size=batch_size,
        n_iter=n_iter,
        hidden_units=hidden_units,
        mode=mode,
    )


def load_data_bundle(data_dir: str, runtime: RuntimeConfigT3) -> DataBundle:
    """Charge et prépare les données pour la tâche 3."""
    image_size = (runtime.img_size, runtime.img_size)

    _, x_train_all, y_train_all, x_test, y_test = load_preprocessed_train_test(
        data_dir=data_dir,
        image_size=image_size,
        normalize=True,
        one_hot=False,
    )

    x_train_all = flatten_images(x_train_all)
    x_test = flatten_images(x_test)

    x_train, x_val, y_train, y_val = split_train_validation(x_train_all, y_train_all)

    return DataBundle(x_train, x_val, y_train, y_val, x_test, y_test)


def train_and_evaluate(
    data: DataBundle,
    args: argparse.Namespace,
    runtime: RuntimeConfigT3,
) -> EvalBundle:
    """Entraîne le MLP puis évalue performance et incertitude."""
    model = build_mlp_classifier(
        input_dim=data.x_train.shape[1],
        num_classes=len(np.unique(data.y_train)),
        hidden_units=runtime.hidden_units,
        dropout_rate=args.dropout_rate,
        learning_rate=args.learning_rate,
    )

    training_data = MLPTrainingData(
        x_train=data.x_train,
        y_train=data.y_train,
        x_val=data.x_val,
        y_val=data.y_val,
    )
    training_config = MLPTrainingConfig(
        epochs=runtime.epochs,
        batch_size=runtime.batch_size,
        learning_rate=args.learning_rate,
        dropout_rate=args.dropout_rate,
        hidden_units=runtime.hidden_units,
        verbose=1,
    )
    train_mlp_classifier(
        model=model,
        training_data=training_data,
        training_config=training_config,
    )

    deterministic_probs = predict_probabilities(model, data.x_test)
    deterministic_preds = np.argmax(deterministic_probs, axis=1)
    deterministic_acc = accuracy_score(data.y_test, deterministic_preds)

    mc_summary = mc_dropout_predict(
        model=model, x_data=data.x_test, n_iter=runtime.n_iter
    )
    mc_acc = accuracy_score(data.y_test, mc_summary.predicted_labels)
    uncertainty_stats = summarize_uncertainty(
        mc_summary.max_probabilities,
        threshold=args.uncertainty_threshold,
    )

    return EvalBundle(
        deterministic_acc=deterministic_acc,
        mc_acc=mc_acc,
        uncertainty_stats=uncertainty_stats,
        entropy_mean=float(np.mean(mc_summary.predictive_entropy)),
        entropy_std=float(np.std(mc_summary.predictive_entropy)),
    )


def main() -> None:
    """Point d'entrée principal pour la tâche 3."""
    args = parse_args()
    runtime = resolve_runtime_config(args)
    data = load_data_bundle(args.data_dir, runtime)

    print("--- Tâche 3: MLP + Gestion de l'incertitude ---")
    print(
        f"Mode={runtime.mode} | "
        f"img_size={runtime.img_size} | epochs={runtime.epochs} | "
        f"batch_size={runtime.batch_size} | n_iter={runtime.n_iter}"
    )
    print(
        f"Train={data.x_train.shape}, Val={data.x_val.shape}, Test={data.x_test.shape}"
    )

    eval_bundle = train_and_evaluate(data, args, runtime)

    print("\n--- Performance ---")
    print(f"Accuracy (inférence standard) : {eval_bundle.deterministic_acc:.4f}")
    print(f"Accuracy (MC Dropout)         : {eval_bundle.mc_acc:.4f}")

    print("\n--- Incertitude ---")
    print(
        f"Seuil={eval_bundle.uncertainty_stats['threshold']:.2f} | "
        f"Incertaines={eval_bundle.uncertainty_stats['uncertain_count']}/"
        f"{eval_bundle.uncertainty_stats['total']} "
        f"({100 * eval_bundle.uncertainty_stats['uncertain_ratio']:.2f}%)"
    )
    print(
        f"Entropie prédictive moyenne={eval_bundle.entropy_mean:.4f} | "
        f"std={eval_bundle.entropy_std:.4f}"
    )


if __name__ == "__main__":
    main()
