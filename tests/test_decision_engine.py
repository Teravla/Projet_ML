"""Tests unitaires du moteur de decision clinique (Tache 5)."""

import numpy as np

from src.decision.engine import (
    categoriser_confiance,
    generer_decision_clinique,
    traiter_batch_decisions,
    generer_recommandation,
)
from src.decision.rules import appliquer_regle_securite_negatif
from src.decision.triage import determiner_priorite, appliquer_triage
from src.config.thresholds import (
    PRIORITE_URGENTE,
    PRIORITE_NORMALE,
    PRIORITE_ELEVEE,
)


CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]


def test_categoriser_confiance_par_seuils():
    """Verifie les bandes de confiance du cahier des charges."""
    assert categoriser_confiance(0.90) == "HAUTE"
    assert categoriser_confiance(0.70) == "MOYENNE"
    assert categoriser_confiance(0.55) == "FAIBLE"
    assert categoriser_confiance(0.20) == "TRES_FAIBLE"


def test_regle_haute_confiance():
    probs = np.array([0.90, 0.05, 0.03, 0.02])
    decision = generer_decision_clinique("P_001", probs, CLASSES)

    assert decision.classe_predite == "glioma"
    assert decision.niveau_confiance == "HAUTE"
    assert decision.decision == "Diagnostic automatique valide"
    assert decision.action_recommandee == "Rapport envoye au medecin traitant"
    assert decision.revision_requise is False


def test_regle_confiance_moyenne():
    probs = np.array([0.10, 0.67, 0.18, 0.05])
    decision = generer_decision_clinique("P_002", probs, CLASSES)

    assert decision.niveau_confiance == "MOYENNE"
    assert decision.decision == "Diagnostic probable - Revision recommandee"
    assert decision.action_recommandee == "Validation par radiologue junior"
    assert decision.revision_requise is True


def test_regle_confiance_faible():
    probs = np.array([0.12, 0.22, 0.09, 0.57])
    decision = generer_decision_clinique("P_003", probs, CLASSES)

    assert decision.niveau_confiance == "FAIBLE"
    assert decision.decision == "Cas incertain"
    assert decision.action_recommandee == "Revision par radiologue senior"
    assert decision.revision_requise is True


def test_regle_tres_faible_confiance():
    probs = np.array([0.40, 0.20, 0.25, 0.15])
    decision = generer_decision_clinique("P_004", probs, CLASSES)

    assert decision.niveau_confiance == "TRES_FAIBLE"
    assert decision.decision == "Incertitude elevee"
    assert (
        decision.action_recommandee == "Double lecture obligatoire + IRM complementaire"
    )
    assert decision.revision_requise is True


def test_regle_securite_pas_de_tumeur_declenchee():
    """Si 'notumor' < 0.95, la verification obligatoire doit etre activee."""
    probs = np.array([0.02, 0.03, 0.92, 0.03])
    decision = generer_decision_clinique("P_005", probs, CLASSES)
    decision = appliquer_regle_securite_negatif(decision, seuil_securite=0.95)

    assert decision.classe_predite == "notumor"
    assert decision.alerte_securite is True
    assert decision.revision_requise is True
    assert "Verification obligatoire" in decision.action_recommandee


def test_regle_securite_pas_de_tumeur_non_declenchee():
    probs = np.array([0.01, 0.01, 0.97, 0.01])
    decision = generer_decision_clinique("P_006", probs, CLASSES)
    decision = appliquer_regle_securite_negatif(decision, seuil_securite=0.95)

    assert decision.classe_predite == "notumor"
    assert decision.alerte_securite is False


def test_triage_alerte_securite_priorite_urgente():
    probs = np.array([0.02, 0.03, 0.92, 0.03])
    decision = generer_decision_clinique("P_007", probs, CLASSES)
    decision = appliquer_regle_securite_negatif(decision, seuil_securite=0.95)

    priorite = determiner_priorite(decision)
    assert priorite == PRIORITE_URGENTE


def test_triage_glioma_haute_confiance_priorite_urgente():
    probs = np.array([0.90, 0.03, 0.03, 0.04])
    decision = generer_decision_clinique("P_008", probs, CLASSES)

    assert determiner_priorite(decision) == PRIORITE_URGENTE


def test_triage_meningioma_haute_confiance_priorite_normale():
    probs = np.array([0.06, 0.89, 0.03, 0.02])
    decision = generer_decision_clinique("P_009", probs, CLASSES)

    assert determiner_priorite(decision) == PRIORITE_NORMALE


def test_triage_notumor_confiance_moyenne_priorite_normale():
    probs = np.array([0.10, 0.10, 0.70, 0.10])
    decision = generer_decision_clinique("P_010", probs, CLASSES)

    assert determiner_priorite(decision) == PRIORITE_NORMALE


def test_generer_recommandation_api_simple():
    probs = np.array([0.08, 0.70, 0.12, 0.10])
    decision = generer_recommandation(probs, CLASSES, patient_id="P_011")

    assert decision.patient_id == "P_011"
    assert decision.classe_predite == "meningioma"
    assert decision.niveau_confiance == "MOYENNE"


def test_traiter_batch_decisions_retourne_bonne_taille_et_ids():
    batch = np.array(
        [
            [0.90, 0.05, 0.03, 0.02],
            [0.10, 0.70, 0.15, 0.05],
            [0.02, 0.03, 0.92, 0.03],
        ]
    )
    ids = ["P_A", "P_B", "P_C"]
    decisions = traiter_batch_decisions(batch, CLASSES, patient_ids=ids)

    assert len(decisions) == 3
    assert [d.patient_id for d in decisions] == ids


def test_appliquer_triage_met_a_jour_priorite():
    probs = np.array([0.90, 0.05, 0.03, 0.02])
    decision = generer_decision_clinique("P_012", probs, CLASSES)
    decision = appliquer_triage(decision)

    assert decision.priorite == PRIORITE_URGENTE


def test_tres_faible_glioma_reste_urgent():
    probs = np.array([0.45, 0.25, 0.20, 0.10])
    decision = generer_decision_clinique("P_013", probs, CLASSES)

    assert decision.niveau_confiance == "TRES_FAIBLE"
    assert determiner_priorite(decision) == PRIORITE_URGENTE


def test_tres_faible_notumor_est_elevee():
    probs = np.array([0.20, 0.15, 0.40, 0.25])
    decision = generer_decision_clinique("P_014", probs, CLASSES)

    assert decision.niveau_confiance == "TRES_FAIBLE"
    assert determiner_priorite(decision) == PRIORITE_ELEVEE
