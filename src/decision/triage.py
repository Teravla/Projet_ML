"""Priorisation des cas selon urgence clinique.

Ce module implémente le système de triage qui détermine la priorité
de traitement pour chaque cas en fonction de la gravité clinique,
du niveau de confiance, et des alertes de sécurité.
"""

from collections import Counter

from src.decision.engine import ClinicalDecision
from src.config.thresholds import (
    GRAVITE_CLINIQUE,
    PRIORITE_URGENTE,
    PRIORITE_ELEVEE,
    PRIORITE_NORMALE,
    PRIORITE_ROUTINE,
)


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

    # Règle 2: Incertitude très élevée = priorité élevée ou urgente
    if decision.niveau_confiance == "TRES_FAIBLE":
        # Si la classe suspectée est grave, urgence maximale
        classe_norm = decision.classe_predite.lower()
        if gravites.get(classe_norm, 3) >= 4:
            return PRIORITE_URGENTE
        else:
            return PRIORITE_ELEVEE

    # Règle 3: Combiner gravité de la classe et confiance
    classe_norm = decision.classe_predite.lower()
    gravite = gravites.get(classe_norm, 3)  # Défaut gravité moyenne

    if gravite >= 5:  # Gliome (malin)
        if decision.niveau_confiance == "HAUTE":
            return PRIORITE_URGENTE  # Tumeur maligne confirmée
        elif decision.niveau_confiance in ["MOYENNE", "FAIBLE"]:
            return PRIORITE_ELEVEE  # Tumeur maligne suspectée
        else:
            return PRIORITE_URGENTE  # Incertitude déjà gérée plus haut

    elif gravite >= 3:  # Méningiome, Pituitaire (bénignes)
        if decision.niveau_confiance == "HAUTE":
            return PRIORITE_NORMALE  # Tumeur bénigne confirmée
        else:
            return PRIORITE_ELEVEE  # Tumeur bénigne suspectée

    else:  # Pas de tumeur
        if decision.niveau_confiance == "HAUTE":
            return PRIORITE_ROUTINE  # Pas de tumeur, haute confiance
        elif decision.niveau_confiance == "MOYENNE":
            return PRIORITE_NORMALE  # Pas de tumeur, à vérifier
        else:
            return PRIORITE_ELEVEE  # Pas de tumeur mais doute


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
