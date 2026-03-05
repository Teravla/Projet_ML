"""Script d'exécution de la Tâche 2: RegLog multinomiale + calibration."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_dataset_split
from src.data.preprocess import preprocess_dataset
from src.models.calibration import (
    analyze_uncertain_predictions,
    calibrate_classifier,
    summarize_confidence_distribution,
)
from src.models.log_reg import (
    flatten_images,
    predict_with_confidence,
    train_logistic_regression,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Task 2 - Calibrated Multinomial Logistic Regression"
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--max-iter", type=int, default=600)
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument(
        "--calibration",
        type=str,
        default="sigmoid",
        choices=["sigmoid", "isotonic"],
        help="sigmoid=Platt scaling, isotonic=Isotonic regression",
    )
    parser.add_argument("--uncertainty-threshold", type=float, default=0.7)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Mode rapide: img-size<=32, max-iter<=200, cv<=2",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    effective_img_size = args.img_size
    effective_max_iter = args.max_iter
    effective_cv = args.cv

    if args.fast:
        print("Mode FAST activé: réduction de img-size, max-iter et cv pour accélérer")
        effective_img_size = min(effective_img_size, 32)
        effective_max_iter = min(effective_max_iter, 200)
        effective_cv = min(effective_cv, 2)

    train_dir = Path(args.data_dir) / "Training"
    test_dir = Path(args.data_dir) / "Testing"
    image_size = (effective_img_size, effective_img_size)

    train_split = load_dataset_split(train_dir, image_size=image_size)
    test_split = load_dataset_split(
        test_dir, image_size=image_size, class_names=train_split.class_names
    )

    x_train, y_train = preprocess_dataset(
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

    x_train_flat = flatten_images(x_train)
    x_test_flat = flatten_images(x_test)

    print("--- Tâche 2: Régression Logistique + Calibration ---")
    print(
        f"Mode={'FAST' if args.fast else 'STANDARD'} | "
        f"img_size={effective_img_size} | max_iter={effective_max_iter} | cv={effective_cv}"
    )
    print(f"Train shape: {x_train_flat.shape} | Test shape: {x_test_flat.shape}")
    print(f"Classes: {train_split.class_names}")

    base_model = train_logistic_regression(
        x_train_flat,
        y_train,
        max_iter=effective_max_iter,
    )
    base_probs = base_model.predict_proba(x_test_flat)
    base_preds = np.argmax(base_probs, axis=1)
    base_acc = accuracy_score(y_test, base_preds)

    calibrated_model = calibrate_classifier(
        base_model=base_model,
        method=args.calibration,
        cv=effective_cv,
    )
    calibrated_model.fit(x_train_flat, y_train)

    calibrated_probs = calibrated_model.predict_proba(x_test_flat)
    calibrated_preds = np.argmax(calibrated_probs, axis=1)
    calibrated_acc = accuracy_score(y_test, calibrated_preds)

    summary = predict_with_confidence(
        model=calibrated_model,
        x_data=x_test_flat,
        uncertainty_threshold=args.uncertainty_threshold,
    )
    confidence_stats = summarize_confidence_distribution(summary.max_prob)
    uncertain_stats = analyze_uncertain_predictions(
        summary.max_prob, threshold=args.uncertainty_threshold
    )

    print("\n--- Accuracy ---")
    print(f"Base model accuracy       : {base_acc:.4f}")
    print(f"Calibrated model accuracy : {calibrated_acc:.4f}")

    print("\n--- Distribution des scores de confiance (max_prob) ---")
    print(
        "p25={p25:.3f} | p50={p50:.3f} | p75={p75:.3f} | mean={mean:.3f} | std={std:.3f}".format(
            **confidence_stats
        )
    )

    print("\n--- Prédictions incertaines ---")
    print(
        f"Seuil={uncertain_stats['threshold']:.2f} | "
        f"Incertaines={uncertain_stats['uncertain_count']}/{uncertain_stats['total']} "
        f"({100 * uncertain_stats['uncertain_ratio']:.2f}%)"
    )


if __name__ == "__main__":
    main()
