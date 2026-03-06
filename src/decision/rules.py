"""Règles métier de triage et sécurité faux négatifs.

Ce module implémente les règles de sécurité pour minimiser les faux négatifs,
c'est-à-dire les cas où le système prédit "Pas de tumeur" alors qu'une tumeur
est présente. Dans un contexte médical, ces erreurs sont particulièrement graves.
"""

from src.decision.engine import ClinicalDecision
from src.enums.enums import (
    ConfidenceLevel,
    ConfidenceLevelThresholds,
    HyperParametersInt,
    NoTumorAlias,
)


NOTUMOR_ALIASES = tuple(NoTumorAlias.__members__.values())


def appliquer_regle_securite_negatif(
    decision: ClinicalDecision,
    seuil_securite: float = ConfidenceLevelThresholds.SEUIL_SECURITE_NEGATIF,
) -> ClinicalDecision:
    """Applique la règle de sécurité pour les prédictions "Pas de tumeur".

    Règle : Si la prédiction est "notumor" (pas de tumeur) mais que la confiance
    est inférieure à 0.95, une vérification obligatoire est requise pour minimiser
    le risque de faux négatif (tumeur non détectée).

    Cette tolérance asymétrique reflète le fait qu'un faux négatif (tumeur manquée)
    est beaucoup plus grave qu'un faux positif (alarme injustifiée).

    Args:
        decision: Décision clinique à vérifier
        seuil_securite: Seuil de confiance minimal pour "notumor" (défaut: 0.95)

    Returns:
        ClinicalDecision mise à jour avec alerte si nécessaire
    """
    # Normaliser le nom de classe (support multiples formats)
    classe_lower = decision.classe_predite.lower().replace(" ", "").replace("_", "")

    if classe_lower in NOTUMOR_ALIASES:
        if decision.confiance < seuil_securite:
            # Déclencher l'alerte de sécurité
            decision.alerte_securite = True
            decision.revision_requise = True

            # Modifier la recommandation
            decision.action_recommandee = (
                "Verification obligatoire (risque faux negatif) - "
                "Double lecture + IRM de controle recommandee"
            )

            # Ajuster la décision textuelle
            decision.decision = (
                f"Prédiction 'Pas de tumeur' à {decision.confiance:.1%} confiance - "
                "Seuil de sécurité non atteint"
            )

    return decision


def appliquer_regles_securite_batch(
    decisions: list[ClinicalDecision],
    seuil_securite: float = ConfidenceLevelThresholds.SEUIL_SECURITE_NEGATIF,
) -> list[ClinicalDecision]:
    """Applique les règles de sécurité sur un lot de décisions.

    Args:
        decisions: Liste de décisions cliniques
        seuil_securite: Seuil de confiance minimal pour "notumor"

    Returns:
        Liste de décisions mises à jour
    """
    return [appliquer_regle_securite_negatif(d, seuil_securite) for d in decisions]


def detecter_cas_ambigus(decision: ClinicalDecision, seuil_ecart: float = 0.15) -> bool:
    """Détecte les cas où plusieurs classes ont des probabilités proches.

    Un cas ambigu est défini comme un cas où les deux classes les plus
    probables ont des probabilités séparées par moins de `seuil_ecart`.

    Args:
        decision: Décision clinique à analyser
        seuil_ecart: Écart minimal entre top-1 et top-2 (défaut: 0.15)

    Returns:
        True si le cas est ambigu, False sinon
    """
    # Trier les probabilités par ordre décroissant
    probs_sorted = sorted(decision.probabilites.values(), reverse=True)

    if len(probs_sorted) < HyperParametersInt.MIN_PROBABILITIES_FOR_AMBIGUITY:
        return False

    ecart = probs_sorted[0] - probs_sorted[1]
    return ecart < seuil_ecart


def identifier_cas_limites(
    decisions: list[ClinicalDecision], seuil_ecart: float = 0.15
) -> list[ClinicalDecision]:
    """Identifie les cas limites nécessitant une attention particulière.

    Les cas limites incluent:
    - Prédictions "notumor" sous le seuil de sécurité (alertes activées)
    - Cas ambigus avec probabilités proches entre classes
    - Cas à confiance très faible

    Args:
        decisions: Liste de décisions cliniques
        seuil_ecart: Seuil pour détecter les ambiguïtés

    Returns:
        Liste des cas limites
    """
    cas_limites = []

    for decision in decisions:
        # Cas 1: Alerte de sécurité activée
        if decision.alerte_securite:
            cas_limites.append(decision)
            continue

        # Cas 2: Confiance très faible
        if decision.niveau_confiance == ConfidenceLevel.CONFIDENCE_TRES_FAIBLE:
            cas_limites.append(decision)
            continue

        # Cas 3: Cas ambigu (probabilités proches)
        if detecter_cas_ambigus(decision, seuil_ecart):
            cas_limites.append(decision)
            continue

    return cas_limites


def statistiques_securite(decisions: list[ClinicalDecision]) -> dict:
    """Calcule des statistiques de sécurité sur les décisions.

    Args:
        decisions: Liste de décisions cliniques

    Returns:
        Dictionnaire avec statistiques (nombre alertes, cas ambigus, etc.)
    """
    n_total = len(decisions)
    if n_total == 0:
        return {}

    n_alertes = sum(1 for d in decisions if d.alerte_securite)
    n_ambigus = sum(1 for d in decisions if detecter_cas_ambigus(d))

    # Compter les prédictions "notumor"
    n_notumor = sum(
        1
        for d in decisions
        if d.classe_predite.lower().replace(" ", "").replace("_", "") in NOTUMOR_ALIASES
    )

    n_notumor_alertes = sum(
        1
        for d in decisions
        if (
            d.classe_predite.lower().replace(" ", "").replace("_", "")
            in NOTUMOR_ALIASES
        )
        and d.alerte_securite
    )

    return {
        "n_total": n_total,
        "n_alertes_securite": n_alertes,
        "taux_alertes": n_alertes / n_total,
        "n_cas_ambigus": n_ambigus,
        "taux_ambigus": n_ambigus / n_total,
        "n_predictions_notumor": n_notumor,
        "n_notumor_avec_alerte": n_notumor_alertes,
        "taux_alertes_parmi_notumor": (
            n_notumor_alertes / n_notumor if n_notumor > 0 else 0.0
        ),
    }
