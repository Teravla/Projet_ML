from enum import IntEnum, StrEnum


class ModelType(StrEnum):
    """Model types for training and evaluation."""

    CNN = "cnn"
    MLP = "mlp"
    TRANSFER = "transfer"
    LOGISTIC_REGRESSION = "logreg"


class HyperParametersInt(IntEnum):
    """Integer Hyperparameters for training and evaluation."""

    BASELINE_ACCURACY_PERCENT = 28.33
    ENSEMBLE_THRESHOLD = 0.5
    IMAGE_NDIM = 4
    IMBALANCE_THRESHOLD = 2.0
    NORMALIZATION_MAX = 1.5
    NUM_CLASSES = 4
    TARGET_ACCURACY = 0.90
    VARIANCE_THRESHOLD = 0.001
    RANGE = 42
    MIN_PROBABILITIES_FOR_AMBIGUITY = 2


class HyperParametersStr(StrEnum):
    """String hyperparameters for training and evaluation."""

    URGENT_QUEUE_KEY = "URGENTE (12h)"


class PriorityLevel(StrEnum):
    """Priority levels for triage and decision making."""

    PRIORITY_URGENTE = "URGENTE (12h)"
    PRIORITY_ELEVEE = "Élevée (24h)"
    PRIORITY_NORMALE = "Normale (48h)"
    PRIORITY_ROUTINE = "Routine (1 semaine)"


class ConfidenceLevel(StrEnum):
    """Confidence levels for decision making."""

    CONFIDENCE_HAUTE = "HAUTE"
    CONFIDENCE_MOYENNE = "MOYENNE"
    CONFIDENCE_FAIBLE = "FAIBLE"
    CONFIDENCE_TRES_FAIBLE = "TRES_FAIBLE"


class ConfidenceLevelThresholds(IntEnum):
    # High confidence threshold: automatic diagnosis possible
    SEUIL_CONFIANCE_HAUTE = 0.85

    # Medium confidence threshold: validation recommended
    SEUIL_CONFIANCE_MOYENNE = 0.65

    # Low confidence threshold: senior expertise required
    SEUIL_CONFIANCE_FAIBLE = 0.50

    # Very high confidence threshold for "No tumor" (false negative safety)
    SEUIL_SECURITE_NEGATIF = 0.95


class TumorType(StrEnum):
    """Tumors types for classification."""

    GLIOMA = "glioma"
    MENINGIOMA = "meningioma"
    NOTUMOR = "notumor"
    PITUITARY = "pituitary"


class FileExtension(StrEnum):
    """Allowed file extensions for image loading."""

    PNG = ".png"
    JPG = ".jpg"
    JPEG = ".jpeg"
    BMP = ".bmp"


class Colors(StrEnum):
    """ANSI color codes for terminal output (see halp command)."""

    HEADER = "\033[95m"  # Magenta
    BLUE = "\033[94m"  # Blue
    CYAN = "\033[96m"  # Cyan
    GREEN = "\033[92m"  # Green
    YELLOW = "\033[93m"  # Yellow
    RED = "\033[91m"  # Red
    ENDC = "\033[0m"  # Reset
    BOLD = "\033[1m"  # Bold
    UNDERLINE = "\033[4m"  # Underline


class CostReview(IntEnum):
    """Cost values for business analysis."""

    COUT_FAUX_NEGATIF = 1000  # Euros - High cost for missing a tumor (false negative)
    COUT_FAUX_POSITIF = 100  # Euros - High cost for a false positive
    COUT_REVISION_HUMAINE = 50  # Euros - Cost of human review

class NoTumorAlias(StrEnum):
    """Aliases for the "notumor" class to handle variations in naming."""

    NOTUMOR = "notumor"
    PAS_DE_TUMEUR = "pasdetumeur"
    SANS_TUMEUR = "sanstumeur"
    SAIN_TUMEUR = "saintumeur"