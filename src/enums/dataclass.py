from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import keras
import numpy as np

from src.config.config import DecisionThresholds
from src.data.pipeline import TrainValTestData
from src.enums.enums import ConfidenceLevel, CostReview, TumorType


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


@dataclass
class DecisionThresholds:
    """Ensemble des seuils pour le moteur de décision."""

    haute: float = ConfidenceLevel.CONFIDENCE_HAUTE
    moyenne: float = ConfidenceLevel.CONFIDENCE_MOYENNE
    faible: float = ConfidenceLevel.CONFIDENCE_FAIBLE
    securite_negatif: float = ConfidenceLevel.CONFIDENCE_TRES_FAIBLE


@dataclass
class CostParameters:
    """Paramètres de coût pour l'analyse métier."""

    faux_negatif: int = CostReview.COUT_FAUX_NEGATIF
    faux_positif: int = CostReview.COUT_FAUX_POSITIF
    revision: int = CostReview.COUT_REVISION_HUMAINE


@dataclass(frozen=True)
class TrainValTestData:
    """Conteneur standard pour les tenseurs train/val/test."""

    x_train: np.ndarray
    x_val: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


@dataclass(frozen=True)
class LabelEncodingConfig:
    """Configuration d'encodage des labels."""

    one_hot: bool = False
    num_classes: int | None = None


@dataclass
class DecisionWorkflow:
    """Informations workflow et sécurité associées à une décision."""

    decision: str
    action_recommandee: str
    priorite: str
    revision_requise: bool
    alerte_securite: bool = False


@dataclass
class ClinicalDecision:
    """Résultat d'une décision clinique automatisée.

    Attributes:
        patient_id: Identifiant du patient (ou index image)
        classe_predite: Classe avec la probabilité maximale
        confiance: Probabilité maximale (0-1)
        probabilites: Dictionnaire {classe: probabilité} pour toutes les classes
        niveau_confiance: Catégorie (HAUTE, MOYENNE, FAIBLE, TRES_FAIBLE)
        decision: Texte descriptif de la décision
        action_recommandee: Action clinique à entreprendre
        priorite: Niveau d'urgence/priorité
        revision_requise: Booléen indiquant si révision humaine obligatoire
        alerte_securite: Indicateur d'alerte de sécurité (faux négatifs)
    """

    patient_id: str
    classe_predite: str
    confiance: float
    probabilites: dict[str, float]
    niveau_confiance: str
    workflow: DecisionWorkflow

    @property
    def decision(self) -> str:
        """Texte descriptif de la décision clinique."""
        return self.workflow.decision

    @decision.setter
    def decision(self, value: str) -> None:
        self.workflow.decision = value

    @property
    def action_recommandee(self) -> str:
        """Action clinique recommandée."""
        return self.workflow.action_recommandee

    @action_recommandee.setter
    def action_recommandee(self, value: str) -> None:
        self.workflow.action_recommandee = value

    @property
    def priorite(self) -> str:
        """Priorité clinique du cas."""
        return self.workflow.priorite

    @priorite.setter
    def priorite(self, value: str) -> None:
        self.workflow.priorite = value

    @property
    def revision_requise(self) -> bool:
        """Indique si une révision humaine est requise."""
        return self.workflow.revision_requise

    @revision_requise.setter
    def revision_requise(self, value: bool) -> None:
        self.workflow.revision_requise = value

    @property
    def alerte_securite(self) -> bool:
        """Indique si une alerte de sécurité est active."""
        return self.workflow.alerte_securite

    @alerte_securite.setter
    def alerte_securite(self, value: bool) -> None:
        self.workflow.alerte_securite = value
