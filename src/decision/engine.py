"""Moteur de décision clinique basé sur des seuils.

Ce module implémente le moteur de décision du SAD qui traduit
des probabilités de modèle en recommandations cliniques actionnables.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from src.config.thresholds import (
    DecisionThresholds,
)


CONFIDENCE_HAUTE = "HAUTE"
CONFIDENCE_MOYENNE = "MOYENNE"
CONFIDENCE_FAIBLE = "FAIBLE"
CONFIDENCE_TRES_FAIBLE = "TRES_FAIBLE"


def count_confidence_levels(decisions: list["ClinicalDecision"]) -> dict[str, int]:
    """Compte le nombre de décisions par niveau de confiance."""

    return {
        CONFIDENCE_HAUTE: sum(
            1 for d in decisions if d.niveau_confiance == CONFIDENCE_HAUTE
        ),
        CONFIDENCE_MOYENNE: sum(
            1 for d in decisions if d.niveau_confiance == CONFIDENCE_MOYENNE
        ),
        CONFIDENCE_FAIBLE: sum(
            1 for d in decisions if d.niveau_confiance == CONFIDENCE_FAIBLE
        ),
        CONFIDENCE_TRES_FAIBLE: sum(
            1 for d in decisions if d.niveau_confiance == CONFIDENCE_TRES_FAIBLE
        ),
    }


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


def categoriser_confiance(
    confiance: float, seuils: Optional[DecisionThresholds] = None
) -> str:
    """Catégorise le niveau de confiance selon les seuils définis.

    Args:
        confiance: Probabilité maximale (0-1)
        seuils: Seuils personnalisés (si None, utilise les valeurs par défaut)

    Returns:
        Catégorie: "HAUTE", "MOYENNE", "FAIBLE", ou "TRES_FAIBLE"
    """
    if seuils is None:
        seuils = DecisionThresholds()

    if confiance >= seuils.haute:
        return CONFIDENCE_HAUTE
    if confiance >= seuils.moyenne:
        return CONFIDENCE_MOYENNE
    if confiance >= seuils.faible:
        return CONFIDENCE_FAIBLE
    return CONFIDENCE_TRES_FAIBLE


def generer_decision_clinique(
    patient_id: str,
    probabilites: np.ndarray,
    classes: list[str],
    seuils: Optional[DecisionThresholds] = None,
) -> ClinicalDecision:
    """Génère une décision clinique à partir des probabilités du modèle.

    Cette fonction implémente les règles métier du SAD:
    - Haute confiance (≥0.85): diagnostic automatique valide
    - Confiance moyenne (0.65-0.85): révision recommandée par radiologue junior
    - Confiance faible (0.50-0.65): révision par radiologue senior
    - Confiance très faible (<0.50): double lecture obligatoire + examens complémentaires

    Args:
        patient_id: Identifiant du patient
        probabilites: Array de probabilités pour chaque classe (shape: [n_classes])
        classes: Liste des noms de classes correspondant aux probabilités
        seuils: Seuils personnalisés (optionnel)

    Returns:
        ClinicalDecision avec toutes les recommandations
    """
    if seuils is None:
        seuils = DecisionThresholds()

    # Identifier la classe prédite et sa confiance
    idx_max = np.argmax(probabilites)
    classe_predite = classes[idx_max]
    confiance = float(probabilites[idx_max])

    # Créer dictionnaire probabilités
    prob_dict = {classe: float(probabilites[i]) for i, classe in enumerate(classes)}

    # Catégoriser la confiance
    niveau_confiance = categoriser_confiance(confiance, seuils)

    # Générer les recommandations selon le niveau de confiance
    if confiance >= seuils.haute:
        decision = "Diagnostic automatique valide"
        action = "Rapport envoye au medecin traitant"
        revision_requise = False

    elif confiance >= seuils.moyenne:
        decision = "Diagnostic probable - Revision recommandee"
        action = "Validation par radiologue junior"
        revision_requise = True

    elif confiance >= seuils.faible:
        decision = "Cas incertain"
        action = "Revision par radiologue senior"
        revision_requise = True

    else:
        decision = "Incertitude elevee"
        action = "Double lecture obligatoire + IRM complementaire"
        revision_requise = True

    # La priorité sera déterminée par le module triage
    # Pour l'instant, placeholder
    priorite = "A_DETERMINER"

    return ClinicalDecision(
        patient_id=patient_id,
        classe_predite=classe_predite,
        confiance=confiance,
        probabilites=prob_dict,
        niveau_confiance=niveau_confiance,
        workflow=DecisionWorkflow(
            decision=decision,
            action_recommandee=action,
            priorite=priorite,
            revision_requise=revision_requise,
            alerte_securite=False,
        ),
    )


def generer_recommandation(
    probabilites: np.ndarray,
    classes: list[str],
    seuils: Optional[DecisionThresholds] = None,
    patient_id: str = "P_00000",
) -> ClinicalDecision:
    """Applique les regles de decision et retourne une recommandation clinique.

    Cette fonction fournit une API simple conforme au cahier des charges
    de la Tache 5 et repose sur ``generer_decision_clinique``.

    Args:
        probabilites: Probabilites par classe (shape: [n_classes])
        classes: Liste des classes dans le meme ordre que ``probabilites``
        seuils: Seuils personnalises (optionnel)
        patient_id: Identifiant patient (defaut: P_00000)

    Returns:
        ClinicalDecision
    """
    return generer_decision_clinique(
        patient_id=patient_id,
        probabilites=probabilites,
        classes=classes,
        seuils=seuils,
    )


def traiter_batch_decisions(
    probabilites_batch: np.ndarray,
    classes: list[str],
    patient_ids: Optional[list[str]] = None,
    seuils: Optional[DecisionThresholds] = None,
) -> list[ClinicalDecision]:
    """Traite un lot de prédictions pour générer des décisions cliniques.

    Args:
        probabilites_batch: Array de probabilités (shape: [n_samples, n_classes])
        classes: Liste des noms de classes
        patient_ids: Liste d'identifiants patients (si None, génère des IDs automatiques)
        seuils: Seuils personnalisés (optionnel)

    Returns:
        Liste de ClinicalDecision pour chaque échantillon
    """
    n_samples = probabilites_batch.shape[0]

    if patient_ids is None:
        patient_ids = [f"P_{i:05d}" for i in range(n_samples)]

    decisions = []
    for i in range(n_samples):
        decision = generer_decision_clinique(
            patient_id=patient_ids[i],
            probabilites=probabilites_batch[i],
            classes=classes,
            seuils=seuils,
        )
        decisions.append(decision)

    return decisions


def statistiques_decisions(decisions: list[ClinicalDecision]) -> dict[str, float]:
    """Calcule des statistiques sur un ensemble de décisions.

    Args:
        decisions: Liste de décisions cliniques

    Returns:
        Dictionnaire de statistiques (taux de haute confiance, révisions requises, etc.)
    """
    n_total = len(decisions)
    if n_total == 0:
        return {}

    confidence_counts = count_confidence_levels(decisions)

    n_revisions = sum(1 for d in decisions if d.revision_requise)
    n_alertes = sum(1 for d in decisions if d.alerte_securite)

    confiances = [d.confiance for d in decisions]

    return {
        "n_total": n_total,
        "taux_haute_confiance": confidence_counts[CONFIDENCE_HAUTE] / n_total,
        "taux_moyenne_confiance": confidence_counts[CONFIDENCE_MOYENNE] / n_total,
        "taux_faible_confiance": confidence_counts[CONFIDENCE_FAIBLE] / n_total,
        "taux_tres_faible_confiance": confidence_counts[CONFIDENCE_TRES_FAIBLE]
        / n_total,
        "taux_revision_requise": n_revisions / n_total,
        "taux_alertes_securite": n_alertes / n_total,
        "confiance_moyenne": np.mean(confiances),
        "confiance_mediane": np.median(confiances),
        "confiance_min": np.min(confiances),
        "confiance_max": np.max(confiances),
    }
