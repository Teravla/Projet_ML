"""Chargement des données IRM."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


DEFAULT_CLASS_NAMES = ["glioma", "meningioma", "pituitary", "notumor"]
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

# OpenCV est une extension C, pylint peut ne pas résoudre ses membres statiquement.
_CV_IMREAD = getattr(cv2, "imread")
_CV_IMREAD_COLOR = getattr(cv2, "IMREAD_COLOR")
_CV_CVT_COLOR = getattr(cv2, "cvtColor")
_CV_COLOR_BGR2RGB = getattr(cv2, "COLOR_BGR2RGB")
_CV_RESIZE = getattr(cv2, "resize")
_CV_INTER_AREA = getattr(cv2, "INTER_AREA")


@dataclass(frozen=True)
class DatasetSplit:
    """Contient les tenseurs images + labels d'un split."""

    images: np.ndarray
    labels: np.ndarray
    class_names: list[str]
    image_paths: list[Path]


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS


def discover_classes(
    split_dir: str | Path, class_names: Iterable[str] | None = None
) -> list[str]:
    """Retourne les classes présentes dans le répertoire du split."""

    split_path = Path(split_dir)
    expected = list(class_names) if class_names is not None else DEFAULT_CLASS_NAMES
    discovered = [
        class_name for class_name in expected if (split_path / class_name).is_dir()
    ]
    if not discovered:
        discovered = sorted([p.name for p in split_path.iterdir() if p.is_dir()])
    return discovered


def summarize_split(
    split_dir: str | Path, class_names: Iterable[str] | None = None
) -> dict[str, int]:
    """Compte le nombre d'images par classe dans un split."""

    split_path = Path(split_dir)
    classes = discover_classes(split_path, class_names)
    summary: dict[str, int] = {}
    for class_name in classes:
        class_dir = split_path / class_name
        summary[class_name] = sum(
            1 for file_path in class_dir.rglob("*") if _is_image_file(file_path)
        )
    return summary


def load_image(
    image_path: str | Path, image_size: tuple[int, int] = (224, 224), rgb: bool = True
) -> np.ndarray:
    """Charge une image depuis le disque et la redimensionne."""

    path = Path(image_path)
    image = _CV_IMREAD(str(path), _CV_IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Image illisible: {path}")
    if rgb:
        image = _CV_CVT_COLOR(image, _CV_COLOR_BGR2RGB)
    width, height = image_size
    image = _CV_RESIZE(image, (width, height), interpolation=_CV_INTER_AREA)
    return image.astype(np.float32)


def load_dataset_split(
    split_dir: str | Path,
    image_size: tuple[int, int] = (224, 224),
    class_names: Iterable[str] | None = None,
) -> DatasetSplit:
    """Charge l'ensemble des images d'un split en mémoire."""

    split_path = Path(split_dir)
    classes = discover_classes(split_path, class_names)
    images: list[np.ndarray] = []
    labels: list[int] = []
    image_paths: list[Path] = []

    for label_idx, class_name in enumerate(classes):
        class_dir = split_path / class_name
        for file_path in sorted(class_dir.rglob("*")):
            if not _is_image_file(file_path):
                continue
            image = load_image(file_path, image_size=image_size, rgb=True)
            images.append(image)
            labels.append(label_idx)
            image_paths.append(file_path)

    if not images:
        raise ValueError(f"Aucune image trouvée dans {split_path}")

    return DatasetSplit(
        images=np.stack(images).astype(np.float32),
        labels=np.array(labels, dtype=np.int64),
        class_names=classes,
        image_paths=image_paths,
    )
