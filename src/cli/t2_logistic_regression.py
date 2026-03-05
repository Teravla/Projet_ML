"""Script d'exécution de la Tâche 2: RegLog multinomiale + calibration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import accuracy_score

from src.data.pipeline import load_preprocessed_train_test
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


@dataclass(frozen=True)
class RuntimeConfig:
    """Paramètres effectifs d'exécution."""

    img_size: int
    max_iter: int
    cv: int
    mode: str


@dataclass(frozen=True)
class DatasetBundle:
    """Données preprocessées pour la tâche 2."""

    x_train_flat: np.ndarray
    y_train: np.ndarray
    x_test_flat: np.ndarray
    y_test: np.ndarray
    class_names: list[str]


def parse_args() -> argparse.Namespace:
    """Parse les arguments de la ligne de commande."""
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


def resolve_runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    """Construit la configuration effective (FAST/STD)."""
    img_size = args.img_size
    max_iter = args.max_iter
    cv = args.cv
    mode = "FAST" if args.fast else "STANDARD"

    if args.fast:
        print("Mode FAST activé: réduction de img-size, max-iter et cv pour accélérer")
        img_size = min(img_size, 32)
        max_iter = min(max_iter, 200)
        cv = min(cv, 2)

    return RuntimeConfig(img_size=img_size, max_iter=max_iter, cv=cv, mode=mode)


def load_dataset_bundle(data_dir: str, runtime: RuntimeConfig) -> DatasetBundle:
    """Charge et preprocess les données train/test."""
    image_size = (runtime.img_size, runtime.img_size)

    train_split, x_train, y_train, x_test, y_test = load_preprocessed_train_test(
        data_dir=data_dir,
        image_size=image_size,
        normalize=True,
        one_hot=False,
    )

    return DatasetBundle(
        x_train_flat=flatten_images(x_train),
        y_train=y_train,
        x_test_flat=flatten_images(x_test),
        y_test=y_test,
        class_names=train_split.class_names,
    )


def run_logreg_pipeline(
    bundle: DatasetBundle,
    runtime: RuntimeConfig,
    calibration_method: str,
    uncertainty_threshold: float,
) -> tuple[float, float, dict[str, float], dict[str, float]]:
    """Entraîne, calibre et évalue le modèle de régression logistique."""
    base_model = train_logistic_regression(
        bundle.x_train_flat,
        bundle.y_train,
        max_iter=runtime.max_iter,
    )
    base_probs = base_model.predict_proba(bundle.x_test_flat)
    base_acc = accuracy_score(bundle.y_test, np.argmax(base_probs, axis=1))

    calibrated_model = calibrate_classifier(
        base_model=base_model,
        method=calibration_method,
        cv=runtime.cv,
    )
    calibrated_model.fit(bundle.x_train_flat, bundle.y_train)

    calibrated_probs = calibrated_model.predict_proba(bundle.x_test_flat)
    calibrated_acc = accuracy_score(bundle.y_test, np.argmax(calibrated_probs, axis=1))

    summary = predict_with_confidence(
        model=calibrated_model,
        x_data=bundle.x_test_flat,
        uncertainty_threshold=uncertainty_threshold,
    )
    confidence_stats = summarize_confidence_distribution(summary.max_prob)
    uncertain_stats = analyze_uncertain_predictions(
        summary.max_prob,
        threshold=uncertainty_threshold,
    )

    return base_acc, calibrated_acc, confidence_stats, uncertain_stats


def main() -> None:
    """Point d'entrée principal pour la tâche 2."""
    args = parse_args()
    runtime = resolve_runtime_config(args)
    bundle = load_dataset_bundle(args.data_dir, runtime)

    print("--- Tâche 2: Régression Logistique + Calibration ---")
    print(
        f"Mode={runtime.mode} | "
        f"img_size={runtime.img_size} | max_iter={runtime.max_iter} | cv={runtime.cv}"
    )
    print(
        f"Train shape: {bundle.x_train_flat.shape} | Test shape: {bundle.x_test_flat.shape}"
    )
    print(f"Classes: {bundle.class_names}")

    base_acc, calibrated_acc, confidence_stats, uncertain_stats = run_logreg_pipeline(
        bundle=bundle,
        runtime=runtime,
        calibration_method=args.calibration,
        uncertainty_threshold=args.uncertainty_threshold,
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
