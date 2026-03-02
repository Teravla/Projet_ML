"""Script d'exécution de la Tâche 1: exploration et prétraitement."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.augment import augment_batch, create_training_augmenter
from src.data.loader import load_dataset_split, summarize_split
from src.data.preprocess import preprocess_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tâche 1 - Exploration et Prétraitement"
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
    total = sum(summary.values())
    print(f"\n[{split_name}] distribution des classes (total={total})")
    for class_name, count in summary.items():
        print(f"- {class_name:<12}: {count}")


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    train_dir = data_dir / "Training"
    test_dir = data_dir / "Testing"

    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            f"Dossiers introuvables. Attendu: {train_dir} et {test_dir}"
        )

    train_summary = summarize_split(train_dir)
    test_summary = summarize_split(test_dir)
    print_split_summary("Training", train_summary)
    print_split_summary("Testing", test_summary)

    image_size = (args.img_size, args.img_size)
    train_split = load_dataset_split(train_dir, image_size=image_size)
    test_split = load_dataset_split(
        test_dir, image_size=image_size, class_names=train_split.class_names
    )

    train_images, train_labels = preprocess_dataset(
        train_split.images,
        train_split.labels,
        target_size=image_size,
        normalize=not args.no_normalize,
        one_hot=False,
    )
    test_images, test_labels = preprocess_dataset(
        test_split.images,
        test_split.labels,
        target_size=image_size,
        normalize=not args.no_normalize,
        one_hot=False,
    )

    augmenter = create_training_augmenter(seed=42)
    sample_size = min(8, len(train_images))
    augmented_sample = augment_batch(train_images[:sample_size], augmenter)

    print("\nPrétraitement terminé")
    print(f"- Train images shape: {train_images.shape}")
    print(f"- Train labels shape: {train_labels.shape}")
    print(f"- Test images shape : {test_images.shape}")
    print(f"- Test labels shape : {test_labels.shape}")
    print(f"- Batch augmenté   : {augmented_sample.shape}")
    print(f"- Classes          : {train_split.class_names}")


if __name__ == "__main__":
    main()
