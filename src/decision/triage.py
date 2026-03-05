"""Priorisation des cas selon urgence clinique.

Ce module implémente le système de triage qui détermine la priorité
de traitement pour chaque cas en fonction de la gravité clinique,
du niveau de confiance, et des alertes de sécurité.
"""

from collections import Counter

from src.decision.engine import (
    ClinicalDecision,
    CONFIDENCE_FAIBLE,
    CONFIDENCE_HAUTE,
    CONFIDENCE_MOYENNE,
    CONFIDENCE_TRES_FAIBLE,
)
from src.config.thresholds import (
    GRAVITE_CLINIQUE,
    PRIORITE_URGENTE,
    PRIORITE_ELEVEE,
    PRIORITE_NORMALE,
    PRIORITE_ROUTINE,
)


DEFAULT_GRAVITE = 3
GRAVITE_SEUIL_URGENTE = 4
GRAVITE_MALIGNE = 5
GRAVITE_BENIGNE = 3
MALIGNANCY_SUSPECT_LEVELS = {CONFIDENCE_MOYENNE, CONFIDENCE_FAIBLE}


def _priorite_incertitude_tres_faible(gravite: int) -> str:
    """Détermine la priorité en cas de confiance très faible."""
    if gravite >= GRAVITE_SEUIL_URGENTE:
        return PRIORITE_URGENTE
    return PRIORITE_ELEVEE


def _priorite_cas_malin(niveau_confiance: str) -> str:
    """Détermine la priorité pour un cas de gravité maximale."""
    if niveau_confiance == CONFIDENCE_HAUTE:
        return PRIORITE_URGENTE
    if niveau_confiance in MALIGNANCY_SUSPECT_LEVELS:
        return PRIORITE_ELEVEE
    return PRIORITE_URGENTE


def _priorite_cas_benin(niveau_confiance: str) -> str:
    """Détermine la priorité pour un cas bénin."""
    if niveau_confiance == CONFIDENCE_HAUTE:
        return PRIORITE_NORMALE
    return PRIORITE_ELEVEE


def _priorite_pas_tumeur(niveau_confiance: str) -> str:
    """Détermine la priorité pour un cas non tumoral."""
    if niveau_confiance == CONFIDENCE_HAUTE:
        return PRIORITE_ROUTINE
    if niveau_confiance == CONFIDENCE_MOYENNE:
        return PRIORITE_NORMALE
    return PRIORITE_ELEVEE


def _priorite_par_gravite(gravite: int, niveau_confiance: str) -> str:
    """Détermine la priorité à partir de la gravité et de la confiance."""
    if gravite >= GRAVITE_MALIGNE:
        return _priorite_cas_malin(niveau_confiance)
    if gravite >= GRAVITE_BENIGNE:
        return _priorite_cas_benin(niveau_confiance)
    return _priorite_pas_tumeur(niveau_confiance)


def determiner_priorite(
    decision: ClinicalDecision, gravites: dict[str, int] = None
) -> str:
    """Détermine la priorité clinique d'un cas.

    La priorité est déterminée par:
    1. Alertes de sécurité → URGENTE (tumeur possiblement manquée)
    2. Gravité de la classe prédite + niveau de confiance
       - Gliome (malin) haute confiance → URGENTE
       - Gliome incertain → ELEVEE
       - Méningiome/Pituitaire haute confiance → NORMALE
       - Méningiome/Pituitaire incertain → ELEVEE
       - Pas de tumeur haute confiance → ROUTINE
       - Incertitude élevée → URGENTE ou ELEVEE selon contexte

    Args:
        decision: Décision clinique à prioriser
        gravites: Mapping {classe: gravité_1_à_5} (optionnel, utilise défaut)

    Returns:
        Priorité: URGENTE, ELEVEE, NORMALE, ou ROUTINE
    """
    if gravites is None:
        gravites = GRAVITE_CLINIQUE

    # Règle 1: Alerte de sécurité = priorité urgente
    if decision.alerte_securite:
        return PRIORITE_URGENTE

    classe_norm = decision.classe_predite.lower()
    gravite = gravites.get(classe_norm, DEFAULT_GRAVITE)

    # Règle 2: Incertitude très élevée = priorité élevée ou urgente
    if decision.niveau_confiance == CONFIDENCE_TRES_FAIBLE:
        return _priorite_incertitude_tres_faible(gravite)

    # Règle 3: Combiner gravité de la classe et confiance
    return _priorite_par_gravite(gravite, decision.niveau_confiance)


def appliquer_triage(
    decision: ClinicalDecision, gravites: dict[str, int] = None
) -> ClinicalDecision:
    """Applique le triage et met à jour la priorité de la décision.

    Args:
        decision: Décision clinique à mettre à jour
        gravites: Mapping {classe: gravité} (optionnel)

    Returns:
        Décision mise à jour avec priorité déterminée
    """
    decision.priorite = determiner_priorite(decision, gravites)
    return decision


def appliquer_triage_batch(
    decisions: list[ClinicalDecision], gravites: dict[str, int] = None
) -> list[ClinicalDecision]:
    """Applique le triage sur un lot de décisions.

    Args:
        decisions: Liste de décisions cliniques
        gravites: Mapping {classe: gravité} (optionnel)

    Returns:
        Liste de décisions avec priorités mises à jour
    """
    return [appliquer_triage(d, gravites) for d in decisions]


def trier_par_priorite(decisions: list[ClinicalDecision]) -> list[ClinicalDecision]:
    """Trie les décisions par ordre de priorité décroissante.

    Ordre: URGENTE > ELEVEE > NORMALE > ROUTINE

    Args:
        decisions: Liste de décisions cliniques

    Returns:
        Liste triée par priorité décroissante
    """
    ordre_priorite = {
        PRIORITE_URGENTE: 4,
        PRIORITE_ELEVEE: 3,
        PRIORITE_NORMALE: 2,
        PRIORITE_ROUTINE: 1,
        "A_DETERMINER": 0,  # Cas non traité
    }

    return sorted(
        decisions, key=lambda d: ordre_priorite.get(d.priorite, 0), reverse=True
    )


def filtrer_par_priorite(
    decisions: list[ClinicalDecision], priorites: list[str]
) -> list[ClinicalDecision]:
    """Filtre les décisions pour ne garder que certaines priorités.

    Args:
        decisions: Liste de décisions cliniques
        priorites: Liste des priorités à conserver (ex: ["URGENTE", "ELEVEE"])

    Returns:
        Liste filtrée
    """
    return [d for d in decisions if d.priorite in priorites]


def statistiques_triage(decisions: list[ClinicalDecision]) -> dict:
    """Calcule des statistiques sur la distribution des priorités.

    Args:
        decisions: Liste de décisions cliniques

    Returns:
        Dictionnaire avec répartition des priorités
    """
    n_total = len(decisions)
    if n_total == 0:
        return {}

    # Compter les priorités
    compteur = Counter(d.priorite for d in decisions)

    n_urgente = compteur.get(PRIORITE_URGENTE, 0)
    n_elevee = compteur.get(PRIORITE_ELEVEE, 0)
    n_normale = compteur.get(PRIORITE_NORMALE, 0)
    n_routine = compteur.get(PRIORITE_ROUTINE, 0)

    return {
        "n_total": n_total,
        "n_urgente": n_urgente,
        "n_elevee": n_elevee,
        "n_normale": n_normale,
        "n_routine": n_routine,
        "taux_urgente": n_urgente / n_total,
        "taux_elevee": n_elevee / n_total,
        "taux_normale": n_normale / n_total,
        "taux_routine": n_routine / n_total,
        "taux_critique_total": (n_urgente + n_elevee) / n_total,
    }


def generer_file_attente(
    decisions: list[ClinicalDecision],
) -> dict[str, list[ClinicalDecision]]:
    """Génère des files d'attente par priorité pour le workflow clinique.

    Les décisions sont regroupées par niveau de priorité pour faciliter
    la gestion du workflow (ex: traiter d'abord tous les cas urgents).

    Args:
        decisions: Liste de décisions cliniques

    Returns:
        Dictionnaire {priorité: [décisions]}
    """
    files = {
        PRIORITE_URGENTE: [],
        PRIORITE_ELEVEE: [],
        PRIORITE_NORMALE: [],
        PRIORITE_ROUTINE: [],
    }

    for decision in decisions:
        if decision.priorite in files:
            files[decision.priorite].append(decision)

    return files
