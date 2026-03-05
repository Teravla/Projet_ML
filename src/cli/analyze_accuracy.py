"""Analyseur des problèmes d'accuracy basse (28.33% → >90%)

Identifie les goulots d'étranglement:
- Données déséquilibrées
- Classes confuses
- Mauvaise architecture
- Problèmes d'entraînement
"""

from __future__ import annotations

from pathlib import Path
import keras
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from src.data.loader import load_dataset_split
from src.models.cnn import build_cnn_classifier

# Thresholds and constants
IMBALANCE_THRESHOLD = 2.0
NORMALIZATION_MAX = 1.5
VARIANCE_THRESHOLD = 0.001
IMAGE_NDIM = 4


def analyze_class_distribution(
    y_train: np.ndarray, y_test: np.ndarray, class_names: list[str]
) -> None:
    """Analyse le déséquilibre des classes."""
    print("\n📊 ANALYSE DE LA DISTRIBUTION DES CLASSES")
    print("=" * 70)

    unique, counts = np.unique(y_train, return_counts=True)
    print("\nEnsemble d'entraînement:")
    for idx, count in zip(unique, counts):
        pct = (count / len(y_train)) * 100
        print(f"  {class_names[idx]:15s}: {count:4d} ({pct:5.2f}%)")

    unique, counts = np.unique(y_test, return_counts=True)
    print("\nEnsemble de test:")
    for idx, count in zip(unique, counts):
        pct = (count / len(y_test)) * 100
        print(f"  {class_names[idx]:15s}: {count:4d} ({pct:5.2f}%)")

    # Ratio de déséquilibre
    train_counts = np.bincount(y_train)
    imbalance_ratio = train_counts.max() / train_counts.min()
    print(f"\n⚠️  Ratio de déséquilibre: {imbalance_ratio:.2f}x")
    if imbalance_ratio > IMBALANCE_THRESHOLD:
        print("    Solution: Utiliser class_weight dans le training")


def print_confusion_matrix(
    y_test: np.ndarray, y_pred: np.ndarray, class_names: list[str]
) -> None:
    """Affiche la matrice de confusion."""
    print("\nMatrice de confusion:")
    cm = confusion_matrix(y_test, y_pred)

    header = "      " + "  ".join(f"{c[:4]:>4s}" for c in class_names)
    print(header)
    for idx, class_name in enumerate(class_names):
        row_str = f"{class_name[:4]:>4s}  "
        for val in cm[idx]:
            row_str += f"{val:>4d} "
        print(row_str)


def analyze_model_predictions(
    model: keras.Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
) -> np.ndarray:
    """Analyse les prédictions du modèle."""
    print("\n🔍 ANALYSE DES PRÉDICTIONS DU MODÈLE")
    print("=" * 70)

    # Prédictions
    y_pred_raw = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_raw, axis=1)
    y_proba = np.max(y_pred_raw, axis=1)

    # Accuracy
    accuracy = np.mean(y_test == y_pred)
    print(f"\n📈 Accuracy globale: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Accuracy par classe
    print("\nAccuracy par classe:")
    for idx, class_name in enumerate(class_names):
        mask = y_test == idx
        if mask.sum() > 0:
            class_acc = np.mean(y_pred[mask] == y_test[mask])
            print(f"  {class_name:15s}: {class_acc:.4f} (n={mask.sum()})")

    # Confiance moyenne
    print(f"\nConfiance moyenne: {y_proba.mean():.4f}")
    print(f"Confiance min-max: {y_proba.min():.4f} - {y_proba.max():.4f}")

    # Cas mal classifiés
    wrong_mask = y_test != y_pred
    wrong_count = wrong_mask.sum()
    print(
        f"\nCas mal classifies: {wrong_count} / {len(y_test)} "
        f"({wrong_count / len(y_test) * 100:.2f}%)"
    )

    if wrong_mask.sum() > 0:
        wrong_confidences = y_proba[wrong_mask]
        print(f"  Confiance moyenne (erreurs): {wrong_confidences.mean():.4f}")

    # Matrice de confusion
    print_confusion_matrix(y_test, y_pred, class_names)

    return y_pred


def diagnose_training_issues(x_train: np.ndarray) -> None:
    """Diagnostique les problèmes potentiels de données."""
    print("\n⚙️  DIAGNOSTIC DES DONNÉES D'ENTRAÎNEMENT")
    print("=" * 70)

    # Vérifier les valeurs
    print(f"\nRage de pixels: [{x_train.min():.4f}, {x_train.max():.4f}]")
    if x_train.max() > NORMALIZATION_MAX or x_train.min() < -0.5:
        print("  ⚠️  PROBLÈME: Les données ne sont pas normalisées [0,1] ou [-1,1]")
        print("     Solution: Diviser par 255 si données brutes")
    else:
        print("  ✓ Données normalisées correctement")

    # Vérifier les NaN
    nan_count = np.isnan(x_train).sum()
    if nan_count > 0:
        print(f"  ⚠️  PROBLÈME: {nan_count} valeurs NaN trouvées")
    else:
        print("  ✓ Pas de valeurs NaN")

    # Variance
    variance = np.var(x_train)
    print(f"\nVariance globale: {variance:.6f}")
    if variance < VARIANCE_THRESHOLD:
        print("  ⚠️  PROBLÈME: Variance très faible (données constantes?)")
    else:
        print("  ✓ Variance acceptable")

    # Distribution par canal
    if len(x_train.shape) == IMAGE_NDIM:  # Images (batch, H, W, C)
        print("\nMoyenne par canal:")
        for c in range(x_train.shape[-1]):
            mean_c = x_train[..., c].mean()
            print(f"  Canal {c}: {mean_c:.4f}")


def load_and_prepare_data() -> (
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]
):
    """Charge et prepare les donnees d'entrainement et de test."""
    data_dir = Path("data")
    img_size = (64, 64)

    train_split = load_dataset_split(data_dir / "Training", image_size=img_size)
    test_split = load_dataset_split(
        data_dir / "Testing",
        image_size=img_size,
        class_names=train_split.class_names,
    )

    x_train = train_split.images.astype(np.float32) / 255.0
    y_train_raw = train_split.labels
    y_train = (
        np.array(y_train_raw, dtype=np.int64)
        if isinstance(y_train_raw[0], (int, np.integer))
        else np.array([train_split.class_names.index(label) for label in y_train_raw])
    )

    x_test = test_split.images.astype(np.float32) / 255.0
    y_test_raw = test_split.labels
    y_test = (
        np.array(y_test_raw, dtype=np.int64)
        if isinstance(y_test_raw[0], (int, np.integer))
        else np.array([test_split.class_names.index(label) for label in y_test_raw])
    )

    print(f"✓ Train: {x_train.shape}, Test: {x_test.shape}")
    return x_train, y_train, x_test, y_test, train_split.class_names


def train_test_model(x_train: np.ndarray, y_train: np.ndarray) -> keras.Model:
    """Entraine un modele CNN simple pour le diagnostic."""
    x_train_tmp, x_val_tmp, y_train_tmp, y_val_tmp = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42
    )

    model = build_cnn_classifier(x_train.shape[1:], num_classes=4)

    print("  ▶ Entrainement sur 5 epochs (rapide)...")
    model.fit(
        x_train_tmp,
        y_train_tmp,
        validation_data=(x_val_tmp, y_val_tmp),
        epochs=5,
        batch_size=32,
        verbose=0,
    )

    return model


def print_recommendations() -> None:
    """Affiche les recommandations pour ameliorer l'accuracy."""
    print("\n" + "=" * 70)
    print("💡 RECOMMANDATIONS POUR AMÉLIORER L'ACCURACY")
    print("=" * 70)

    recommendations = [
        (
            "Architecture CNN",
            [
                "✓ Ajouter BatchNormalization après chaque Conv2D",
                "✓ Utiliser résiduel blocks (skip connections)",
                "✓ Augmenter le nombre de filtres (64→256)",
                "✓ Ajouter Global Average Pooling au lieu de Flatten",
            ],
        ),
        (
            "Hyperparamétres",
            [
                "✓ Augmenter epochs: 10 → 50-100",
                "✓ Réduire batch_size: 64 → 16-32",
                "✓ Utiliser learning rate schedule (ReduceLROnPlateau)",
                "✓ Ajouter regularization (L2, Dropout 0.5)",
            ],
        ),
        (
            "Données",
            [
                "✓ Augmenter la résolution: 64×64 → 128×128 ou 224×224",
                "✓ Data augmentation: Flips, Rotations, Zoom, Translate",
                "✓ Appliquer class_weight pour déséquilibre",
                "✓ Utiliser image normalization (ImageNet mean/std)",
            ],
        ),
        (
            "Modèles",
            [
                "✓ Transfer Learning: EfficientNetB0, ResNet50",
                "✓ Ensemble voting de plusieurs modèles",
                "✓ K-Fold Cross Validation",
                "✓ Test Time Augmentation (TTA)",
            ],
        ),
    ]

    for category, items in recommendations:
        print(f"\n{category}:")
        for item in items:
            print(f"  {item}")

    print("\n" + "=" * 70)
    print(f"\n🎯 OBJECTIF: Passer de {28.33:.2f}% à >90%")
    print("\n📌 Commande pour amélioration automatique:")
    print("   poetry run task boost")
    print("\n" + "=" * 70 + "\n")


def main() -> None:
    """Point d'entree principal pour l'analyse d'accuracy."""
    print("\n" + "=" * 70)
    print("🔧 ANALYSEUR D'ACCURACY - Diagnostic detaillé")
    print("=" * 70)

    # Charger les donnees
    print("\n[1/4] Chargement des données...")
    x_train, y_train, x_test, y_test, class_names = load_and_prepare_data()

    # Analyze class distribution
    print("\n[2/4] Analyse de la distribution...")
    analyze_class_distribution(y_train, y_test, class_names)

    # Diagnose training issues
    print("\n[3/4] Diagnostic des données...")
    diagnose_training_issues(x_train)

    # Train simple model and analyze
    print("\n[4/4] Entraînement d'un modèle de test...")
    model = train_test_model(x_train, y_train)

    # Analyze predictions
    analyze_model_predictions(model, x_test, y_test, class_names)

    # Recommendations
    print_recommendations()


if __name__ == "__main__":
    main()
