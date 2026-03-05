"""Templates et helpers de sortie des rapports patients."""

from datetime import datetime


CLASS_LABELS: dict[str, str] = {
    "glioma": "Gliome",
    "meningioma": "Meningiome",
    "pituitary": "Tumeur pituitaire",
    "notumor": "Pas de tumeur",
}


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
    return "OK" if niveau_confiance == "HAUTE" else "A_VERIFIER"


def priorite_badge(priorite: str) -> str:
    """Ajoute un prefixe visuel a la priorite dans le rapport."""
    if "URGENTE" in priorite.upper():
        return f"[!] {priorite}"
    if "ELEVEE" in priorite.upper() or "ELEVEE" in priorite.upper().replace("É", "E"):
        return f"[~] {priorite}"
    return f"[-] {priorite}"
