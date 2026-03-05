"""Templates et helpers de sortie des rapports patients."""

from datetime import datetime


CLASS_LABELS: dict[str, str] = {
    "glioma": "Gliome",
    "meningioma": "Meningiome",
    "pituitary": "Tumeur pituitaire",
    "notumor": "Pas de tumeur",
}

CONFIDENCE_HAUTE = "HAUTE"
PRIORITY_URGENTE = "URGENTE"
PRIORITY_ELEVEE = "ELEVEE"
ACCENTED_E = "É"
UNACCENTED_E = "E"


def format_date_fr() -> str:
    """Retourne la date courante au format JJ/MM/AAAA."""
    return datetime.now().strftime("%d/%m/%Y")


def to_clinical_label(class_name: str) -> str:
    """Convertit un nom de classe technique en libelle clinique."""
    key = class_name.lower().replace(" ", "").replace("_", "")
    if key in CLASS_LABELS:
        return CLASS_LABELS[key]
    return class_name


def certitude_flag(niveau_confiance: str) -> str:
    """Retourne le drapeau de certitude du rapport."""
    return "OK" if niveau_confiance == CONFIDENCE_HAUTE else "A_VERIFIER"


def priorite_badge(priorite: str) -> str:
    """Ajoute un prefixe visuel a la priorite dans le rapport."""
    normalized = priorite.upper()
    if PRIORITY_URGENTE in normalized:
        return f"[!] {priorite}"
    if PRIORITY_ELEVEE in normalized or PRIORITY_ELEVEE in normalized.replace(
        ACCENTED_E, UNACCENTED_E
    ):
        return f"[~] {priorite}"
    return f"[-] {priorite}"
