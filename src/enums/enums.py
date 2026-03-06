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


class HyperParametersStr(StrEnum):
    """String hyperparameters for training and evaluation."""
    URGENT_QUEUE_KEY = "URGENTE (12h)"


class PriorityLevel(StrEnum):
    """Priority levels for triage and decision making."""

    PRIORITY_URGENTE = "URGENTE"
    PRIORITY_ELEVEE = "Elevee"
    PRIORITY_ELEVEE_ACCENT = "Élevée"
    PRIORITY_NORMALE = "Normale"
    PRIORITY_ROUTINE = "Routine"


class ConfidenceLevel(StrEnum):
    """Confidence levels for decision making."""

    CONFIDENCE_HAUTE = "HAUTE"
    CONFIDENCE_MOYENNE = "MOYENNE"
    CONFIDENCE_FAIBLE = "FAIBLE"
    CONFIDENCE_TRES_FAIBLE = "TRES_FAIBLE"


class TumorType(StrEnum):
    """Tumors types for classification."""

    GLIOMA = "glioma"
    MENINGIOMA = "meningioma"
    NOTUMOR = "notumor"
    PITUITARY = "pituitary"


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
