"""Script d'exécution de la Tâche 3: MLP probabiliste + incertitude."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from src.data.loader import load_dataset_split
from src.data.preprocess import preprocess_dataset
from src.models.log_reg import flatten_images
from src.models.mlp import (
    build_mlp_classifier,
    predict_probabilities,
    train_mlp_classifier,
)
from src.models.uncertainty import mc_dropout_predict, summarize_uncertainty


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


def main() -> None:
    """Point d'entrée principal pour la tâche 3."""
    args = parse_args()

    effective_img_size = args.img_size
    effective_epochs = args.epochs
    effective_batch_size = args.batch_size
    effective_n_iter = args.n_iter

    hidden_units = (256, 128)

    if args.fast:
        print("Mode FAST activé: réduction de la charge d'entraînement MLP")
        effective_img_size = min(effective_img_size, 32)
        effective_epochs = min(effective_epochs, 5)
        effective_batch_size = max(effective_batch_size, 128)
        effective_n_iter = min(effective_n_iter, 10)
        hidden_units = (128, 64)

    train_dir = Path(args.data_dir) / "Training"
    test_dir = Path(args.data_dir) / "Testing"
    image_size = (effective_img_size, effective_img_size)

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

    x_train_all = flatten_images(x_train_all)
    x_test = flatten_images(x_test)

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_all,
        y_train_all,
        test_size=0.2,
        random_state=42,
        stratify=y_train_all,
    )

    num_classes = len(train_split.class_names)
    model = build_mlp_classifier(
        input_dim=x_train.shape[1],
        num_classes=num_classes,
        hidden_units=hidden_units,
        dropout_rate=args.dropout_rate,
        learning_rate=args.learning_rate,
    )

    print("--- Tâche 3: MLP + Gestion de l'incertitude ---")
    print(
        f"Mode={'FAST' if args.fast else 'STANDARD'} | "
        f"img_size={effective_img_size} | epochs={effective_epochs} | "
        f"batch_size={effective_batch_size} | n_iter={effective_n_iter}"
    )
    print(f"Train={x_train.shape}, Val={x_val.shape}, Test={x_test.shape}")

    train_mlp_classifier(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        epochs=effective_epochs,
        batch_size=effective_batch_size,
        verbose=1,
    )

    deterministic_probs = predict_probabilities(model, x_test)
    deterministic_preds = np.argmax(deterministic_probs, axis=1)
    deterministic_acc = accuracy_score(y_test, deterministic_preds)

    mc_summary = mc_dropout_predict(model=model, x_data=x_test, n_iter=effective_n_iter)
    mc_acc = accuracy_score(y_test, mc_summary.predicted_labels)
    uncertainty_stats = summarize_uncertainty(
        mc_summary.max_probabilities,
        threshold=args.uncertainty_threshold,
    )

    entropy_mean = float(np.mean(mc_summary.predictive_entropy))
    entropy_std = float(np.std(mc_summary.predictive_entropy))

    print("\n--- Performance ---")
    print(f"Accuracy (inférence standard) : {deterministic_acc:.4f}")
    print(f"Accuracy (MC Dropout)         : {mc_acc:.4f}")

    print("\n--- Incertitude ---")
    print(
        f"Seuil={uncertainty_stats['threshold']:.2f} | "
        f"Incertaines={uncertainty_stats['uncertain_count']}/{uncertainty_stats['total']} "
        f"({100 * uncertainty_stats['uncertain_ratio']:.2f}%)"
    )
    print(f"Entropie prédictive moyenne={entropy_mean:.4f} | std={entropy_std:.4f}")


if __name__ == "__main__":
    main()
