"""Script d'exécution de la Tâche 1: exploration et prétraitement."""

from __future__ import annotations

import argparse
from pathlib import Path
from src.data.augment import augment_batch, create_training_augmenter
from src.data.loader import load_dataset_split, summarize_split
from src.data.preprocess import preprocess_dataset
from src.enums.dataclass import PreprocessResult


def parse_args() -> argparse.Namespace:
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Task 1 - Data Exploration and Preprocessing"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Répertoire racine contenant Training/Testing",
    )
    parser.add_argument(
        "--img-size", type=int, default=224, help="Taille cible des images carrées"
    )
    parser.add_argument(
        "--no-normalize", action="store_true", help="Désactive la normalisation [0,1]"
    )
    return parser.parse_args()


def print_split_summary(split_name: str, summary: dict[str, int]) -> None:
    """Affiche un résumé clair de la distribution des classes pour un split."""
    total = sum(summary.values())
    print(f"\n[{split_name}] distribution des classes (total={total})")
    for class_name, count in summary.items():
        print(f"- {class_name:<12}: {count}")


def preprocess_and_augment(
    data_dir: Path, img_size: int, normalize: bool
) -> PreprocessResult:
    """Charge, preprocess et augmente un batch d'images."""
    train_dir = data_dir / "Training"
    test_dir = data_dir / "Testing"
    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            f"Directories not found. Expected: {train_dir} et {test_dir}"
        )

    print_split_summary("Training", summarize_split(train_dir))
    print_split_summary("Testing", summarize_split(test_dir))

    image_size = (img_size, img_size)
    train_split = load_dataset_split(train_dir, image_size=image_size)
    test_split = load_dataset_split(
        test_dir, image_size=image_size, class_names=train_split.class_names
    )

    train_images, train_labels = preprocess_dataset(
        train_split.images,
        train_split.labels,
        target_size=image_size,
        normalize=normalize,
        one_hot=False,
    )
    test_images, test_labels = preprocess_dataset(
        test_split.images,
        test_split.labels,
        target_size=image_size,
        normalize=normalize,
        one_hot=False,
    )

    augmenter = create_training_augmenter(seed=42)
    augmented_sample = augment_batch(
        train_images[: min(8, len(train_images))], augmenter
    )

    return PreprocessResult(
        train_images_shape=train_images.shape,
        train_labels_shape=train_labels.shape,
        test_images_shape=test_images.shape,
        test_labels_shape=test_labels.shape,
        augmented_shape=augmented_sample.shape,
        class_names=train_split.class_names,
    )


def main() -> None:
    """Point d'entrée principal pour la tâche 1."""
    args = parse_args()
    result = preprocess_and_augment(
        data_dir=Path(args.data_dir),
        img_size=args.img_size,
        normalize=not args.no_normalize,
    )

    print("\nPrétraitement terminé")
    print(f"- Train images shape: {result.train_images_shape}")
    print(f"- Train labels shape: {result.train_labels_shape}")
    print(f"- Test images shape : {result.test_images_shape}")
    print(f"- Test labels shape : {result.test_labels_shape}")
    print(f"- Batch augmenté   : {result.augmented_shape}")
    print(f"- Classes          : {result.class_names}")


if __name__ == "__main__":
    main()
