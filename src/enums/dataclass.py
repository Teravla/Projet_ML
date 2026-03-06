from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import keras
import numpy as np

from src.config.thresholds import DecisionThresholds
from src.data.pipeline import TrainValTestData
from src.enums.enums import TumorType


@dataclass(frozen=True)
class TrainConfig:
    """Configuration d'entrainement generique."""

    epochs: int
    batch_size: int


@dataclass(frozen=True)
class SweepConfig:
    """Configuration du sweep CNN."""

    trials: int
    epochs: int
    final_train: bool
    final_epochs: int
    batch_size: int
    use_tta: bool


@dataclass(frozen=True)
class AppConfig:
    """Configuration globale de l'application."""

    img_size: tuple[int, int]
    use_optimized: bool
    use_ensemble: bool
    use_tta: bool
    use_focal: bool
    active_models: tuple[str, ...]


@dataclass(frozen=True)
class EvalBundle:
    """Resultat d'evaluation final."""

    y_pred_final: np.ndarray
    final_accuracy: float
    per_model_accuracy: dict[str, float]


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration d'entraînement des modèles."""

    epochs: int = 100
    batch_size: int = 16


@dataclass(frozen=True)
class PreparedData(TrainValTestData):
    """Données préparées pour l'entraînement et l'évaluation."""

    class_names: list[str]


@dataclass(frozen=True)
class EvaluationResult:
    """Résultats consolidés d'évaluation."""

    y_pred_ensemble: np.ndarray
    acc_cnn: float
    acc_transfer: float | None
    acc_ensemble: float
    model_transfer: keras.Model | None


@dataclass(frozen=True)
class RuntimeConfigT2:
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


@dataclass(frozen=True)
class RuntimeConfigT3:
    """Paramètres effectifs d'exécution de la tâche 3."""

    img_size: int
    epochs: int
    batch_size: int
    n_iter: int
    hidden_units: tuple[int, int]
    mode: str


@dataclass(frozen=True)
class RuntimeConfigT4:
    """Paramètres effectifs d'exécution de la tâche 4."""

    img_size: int
    epochs: int
    batch_size: int
    ts_epochs: int
    mode: str


@dataclass(frozen=True)
class DataBundleT4:
    """Données préparées pour entraînement/évaluation CNN."""

    x_train: np.ndarray
    x_val: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    class_names: list[str]


@dataclass(frozen=True)
class EvalBundleT4:
    """Résultats d'évaluation du CNN calibré."""

    acc_base: float
    acc_calibrated: float
    temperature: float
    confidence_stats: dict[str, float]
    uncertain_stats: dict[str, float]


@dataclass(frozen=True)
class RuntimeConfigT5:
    """Configuration d'exécution de la tâche 5."""

    img_size: int
    n_samples_max: int | None
    data_path: Path
    seuils: DecisionThresholds


@dataclass(frozen=True)
class PreprocessResult:
    """Contient les sorties utiles du preprocessing."""

    train_images_shape: tuple[int, ...]
    train_labels_shape: tuple[int, ...]
    test_images_shape: tuple[int, ...]
    test_labels_shape: tuple[int, ...]
    augmented_shape: tuple[int, ...]
    class_names: list[str]


@dataclass
class ModelState:
    """Global model state for the dashboard."""

    model: Optional[keras.Model] = None
    test_images: Optional[np.ndarray] = None
    test_labels: Optional[np.ndarray] = None
    class_names: list[str] = field(
        default_factory=lambda: list(TumorType.__members__.keys())
    )
    model_loaded: bool = False
    error_message: Optional[str] = None
    model_path: Optional[str] = None
    last_decisions: Optional[list] = None
    last_true_labels: Optional[list] = None
