"""Utilities de pipeline data pour limiter la duplication dans les CLI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from src.data.loader import DatasetSplit, load_dataset_split
from src.data.preprocess import preprocess_dataset


@dataclass(frozen=True)
class TrainValTestData:
    """Conteneur standard pour les tenseurs train/val/test."""

    x_train: np.ndarray
    x_val: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def load_train_test_splits(
    data_dir: str | Path,
    image_size: tuple[int, int],
) -> tuple[DatasetSplit, DatasetSplit]:
    """Charge les splits Training/Testing avec classes alignées."""

    root = Path(data_dir)
    train_split = load_dataset_split(root / "Training", image_size=image_size)
    test_split = load_dataset_split(
        root / "Testing",
        image_size=image_size,
        class_names=train_split.class_names,
    )
    return train_split, test_split


def preprocess_train_test_splits(
    train_split: DatasetSplit,
    test_split: DatasetSplit,
    target_size: tuple[int, int],
    normalize: bool = True,
    one_hot: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prétraite train/test avec les mêmes paramètres."""

    x_train, y_train = preprocess_dataset(
        train_split.images,
        train_split.labels,
        target_size=target_size,
        normalize=normalize,
        one_hot=one_hot,
    )
    x_test, y_test = preprocess_dataset(
        test_split.images,
        test_split.labels,
        target_size=target_size,
        normalize=normalize,
        one_hot=one_hot,
    )
    return x_train, y_train, x_test, y_test


def load_preprocessed_train_test(
    data_dir: str | Path,
    image_size: tuple[int, int],
    normalize: bool = True,
    one_hot: bool = False,
) -> tuple[DatasetSplit, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Charge puis prétraite train/test avec une seule fonction utilitaire."""

    train_split, test_split = load_train_test_splits(data_dir, image_size)
    x_train, y_train, x_test, y_test = preprocess_train_test_splits(
        train_split=train_split,
        test_split=test_split,
        target_size=image_size,
        normalize=normalize,
        one_hot=one_hot,
    )
    return train_split, x_train, y_train, x_test, y_test


def split_train_validation(
    x_train_all: np.ndarray,
    y_train_all: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Découpe un train en train/validation avec stratification."""

    return train_test_split(
        x_train_all,
        y_train_all,
        test_size=test_size,
        random_state=random_state,
        stratify=y_train_all,
    )
