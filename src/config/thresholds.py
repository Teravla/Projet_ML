"""Seuils de confiance et paramètres de triage clinique.

Ce module définit les seuils et paramètres utilisés par le moteur
de décision clinique pour le système d'aide à la décision (SAD).
"""

from dataclasses import dataclass


# ===== Seuils de Confiance =====

# Seuil haute confiance : diagnostic automatique possible
SEUIL_CONFIANCE_HAUTE = 0.85

# Seuil confiance moyenne : validation recommandée
SEUIL_CONFIANCE_MOYENNE = 0.65

# Seuil confiance faible : expertise senior requise
SEUIL_CONFIANCE_FAIBLE = 0.50

# Seuil très haute confiance pour "Pas de tumeur" (sécurité faux négatifs)
SEUIL_SECURITE_NEGATIF = 0.95


# ===== Niveaux de Priorité =====

PRIORITE_URGENTE = "URGENTE (12h)"
PRIORITE_ELEVEE = "Elevée (24h)"
PRIORITE_NORMALE = "Normale (48h)"
PRIORITE_ROUTINE = "Routine (1 semaine)"


# ===== Coûts pour Analyse Métier =====

COUT_FAUX_NEGATIF = 1000  # Euros - cas grave non détecté
COUT_FAUX_POSITIF = 100  # Euros - alarme injustifiée
COUT_REVISION_HUMAINE = 50  # Euros - temps radiologue


@dataclass
class DecisionThresholds:
    """Ensemble des seuils pour le moteur de décision."""

    haute: float = SEUIL_CONFIANCE_HAUTE
    moyenne: float = SEUIL_CONFIANCE_MOYENNE
    faible: float = SEUIL_CONFIANCE_FAIBLE
    securite_negatif: float = SEUIL_SECURITE_NEGATIF


@dataclass
class CostParameters:
    """Paramètres de coût pour l'analyse métier."""

    faux_negatif: int = COUT_FAUX_NEGATIF
    faux_positif: int = COUT_FAUX_POSITIF
    revision: int = COUT_REVISION_HUMAINE


# Mapping des classes vers la gravité clinique (1=routine, 5=urgent)
GRAVITE_CLINIQUE: dict[str, int] = {
    "glioma": 5,  # Tumeur maligne - urgent
    "meningioma": 3,  # Tumeur bénigne mais intervention possible
    "pituitary": 3,  # Tumeur bénigne mais impact hormonal
    "notumor": 1,  # Pas de tumeur - routine si haute confiance
}
