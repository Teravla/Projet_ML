"""Script d'exécution de la Tâche 4: CNN + Temperature Scaling + activations."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_dataset_split
from src.data.preprocess import preprocess_dataset
from src.models.calibration import (
    TemperatureScaler,
    analyze_uncertain_predictions,
    summarize_confidence_distribution,
)
from src.models.cnn import (
    build_cnn_classifier,
    extract_intermediate_activations,
    predict_logits,
    predict_probabilities,
    train_cnn_classifier,
)


def parse_args() -> argparse.Namespace:
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


def main() -> None:
    args = parse_args()

    effective_img_size = args.img_size
    effective_epochs = args.epochs
    effective_batch_size = args.batch_size
    effective_ts_epochs = args.ts_epochs

    if args.fast:
        print("Mode FAST activé: réduction du coût d'entraînement CNN")
        effective_img_size = min(effective_img_size, 32)
        effective_epochs = min(effective_epochs, 3)
        effective_batch_size = max(effective_batch_size, 128)
        effective_ts_epochs = min(effective_ts_epochs, 100)

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

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_all,
        y_train_all,
        test_size=0.2,
        random_state=42,
        stratify=y_train_all,
    )

    model = build_cnn_classifier(
        input_shape=(image_size[0], image_size[1], 3),
        num_classes=len(train_split.class_names),
        dropout_rate=args.dropout_rate,
        learning_rate=args.learning_rate,
    )

    print("--- Tâche 4: CNN + Temperature Scaling ---")
    print(
        f"Mode={'FAST' if args.fast else 'STANDARD'} | img_size={effective_img_size} | "
        f"epochs={effective_epochs} | batch_size={effective_batch_size} | ts_epochs={effective_ts_epochs}"
    )
    print(f"Train={x_train.shape}, Val={x_val.shape}, Test={x_test.shape}")

    train_cnn_classifier(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        epochs=effective_epochs,
        batch_size=effective_batch_size,
        verbose=1,
    )

    probs_base = predict_probabilities(model, x_test)
    preds_base = np.argmax(probs_base, axis=1)
    acc_base = accuracy_score(y_test, preds_base)

    val_logits = predict_logits(model, x_val)
    test_logits = predict_logits(model, x_test)

    temp_scaler = TemperatureScaler(initial_temperature=1.0)
    temp_scaler.fit(
        logits=val_logits,
        labels=y_val,
        learning_rate=1e-2,
        epochs=effective_ts_epochs,
        verbose=0,
    )

    probs_calibrated = temp_scaler.predict_proba(test_logits)
    preds_calibrated = np.argmax(probs_calibrated, axis=1)
    acc_calibrated = accuracy_score(y_test, preds_calibrated)

    max_prob = np.max(probs_calibrated, axis=1)
    confidence_stats = summarize_confidence_distribution(max_prob)
    uncertain_stats = analyze_uncertain_predictions(
        max_probabilities=max_prob,
        threshold=args.uncertainty_threshold,
    )

    activation_count = min(args.activation_samples, x_test.shape[0])
    activation_layers = ["conv1", "conv2", "conv3", "dense1"]
    activations = extract_intermediate_activations(
        model=model,
        x_data=x_test[:activation_count],
        layer_names=activation_layers,
        batch_size=effective_batch_size,
    )

    output_path = Path(args.activations_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        sample_images=x_test[:activation_count],
        sample_labels=y_test[:activation_count],
        class_names=np.array(train_split.class_names),
        **{f"activation_{layer}": value for layer, value in activations.items()},
    )

    print("\n--- Accuracy ---")
    print(f"CNN base accuracy                : {acc_base:.4f}")
    print(f"CNN + Temperature Scaling acc    : {acc_calibrated:.4f}")
    print(f"Température apprise              : {temp_scaler.temperature:.4f}")

    print("\n--- Confiance calibrée ---")
    print(
        "p25={p25:.3f} | p50={p50:.3f} | p75={p75:.3f} | mean={mean:.3f} | std={std:.3f}".format(
            **confidence_stats
        )
    )
    print(
        f"Seuil={uncertain_stats['threshold']:.2f} | "
        f"Incertaines={uncertain_stats['uncertain_count']}/{uncertain_stats['total']} "
        f"({100 * uncertain_stats['uncertain_ratio']:.2f}%)"
    )

    print("\n--- Activations sauvegardées ---")
    print(f"Fichier: {output_path}")
    for layer in activation_layers:
        print(f"- {layer:<8}: {activations[layer].shape}")


if __name__ == "__main__":
    main()
