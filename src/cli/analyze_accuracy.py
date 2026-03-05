"""Analyseur des problèmes d'accuracy basse (28.33% → >90%)

Identifie les goulots d'étranglement:
- Données déséquilibrées
- Classes confuses
- Mauvaise architecture
- Problèmes d'entraînement
"""

from __future__ import annotations

from pathlib import Path
import sys
import keras
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_dataset_split
from src.models.cnn import build_cnn_classifier, predict_probabilities


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
    if imbalance_ratio > 2.0:
        print("    Solution: Utiliser class_weight dans le training")


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
    print(
        f"\nCas mal classifiés: {wrong_mask.sum()} / {len(y_test)} ({wrong_mask.sum()/len(y_test)*100:.2f}%)"
    )

    if wrong_mask.sum() > 0:
        wrong_confidences = y_proba[wrong_mask]
        print(f"  Confiance moyenne (erreurs): {wrong_confidences.mean():.4f}")

    # Matrice de confusion
    print("\nMatrice de confusion:")
    cm = confusion_matrix(y_test, y_pred)

    header = "      " + "  ".join(f"{c[:4]:>4s}" for c in class_names)
    print(header)
    for idx, class_name in enumerate(class_names):
        row_str = f"{class_name[:4]:>4s}  "
        for val in cm[idx]:
            pct = (val / cm[idx].sum() * 100) if cm[idx].sum() > 0 else 0
            color = "🔴" if val > 0 and idx != np.argmax(cm[idx]) else "🟢"
            row_str += f"{val:>4d} "
        print(row_str)

    return y_pred


def diagnose_training_issues(x_train: np.ndarray, y_train: np.ndarray) -> None:
    """Diagnostique les problèmes potentiels de données."""
    print("\n⚙️  DIAGNOSTIC DES DONNÉES D'ENTRAÎNEMENT")
    print("=" * 70)

    # Vérifier les valeurs
    print(f"\nRage de pixels: [{x_train.min():.4f}, {x_train.max():.4f}]")
    if x_train.max() > 1.5 or x_train.min() < -0.5:
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
    if variance < 0.001:
        print("  ⚠️  PROBLÈME: Variance très faible (données constantes?)")
    else:
        print("  ✓ Variance acceptable")

    # Distribution par canal
    if len(x_train.shape) == 4:  # Images (batch, H, W, C)
        print(f"\nMoyenne par canal:")
        for c in range(x_train.shape[-1]):
            mean_c = x_train[..., c].mean()
            print(f"  Canal {c}: {mean_c:.4f}")


def main() -> None:
    print("\n" + "=" * 70)
    print("🔧 ANALYSEUR D'ACCURACY - Diagnostic detaillé")
    print("=" * 70)

    # Charger les données
    print("\n[1/4] Chargement des données...")
    data_dir = Path("data")
    img_size = (64, 64)  # Taille actuelle

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

    print(f"✓ Train: {x_train.shape}, Test: {x_test.shape}")

    # Analyze class distribution
    print("\n[2/4] Analyse de la distribution...")
    analyze_class_distribution(y_train, y_test, train_split.class_names)

    # Diagnose training issues
    print("\n[3/4] Diagnostic des données...")
    diagnose_training_issues(x_train, y_train)

    # Train simple model and analyze
    print("\n[4/4] Entraînement d'un modèle de test...")

    x_train_tmp, x_val_tmp, y_train_tmp, y_val_tmp = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42
    )

    # Modèle simple
    model = build_cnn_classifier(x_train.shape[1:], num_classes=4)

    print("  ▶ Entraînement sur 5 epochs (rapide)...")
    history = model.fit(
        x_train_tmp,
        y_train_tmp,
        validation_data=(x_val_tmp, y_val_tmp),
        epochs=5,
        batch_size=32,
        verbose=0,
    )

    # Analyze predictions
    y_pred = analyze_model_predictions(model, x_test, y_test, train_split.class_names)

    # Recommendations
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
            "Hyperparámétres",
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


if __name__ == "__main__":
    main()
