"""Tests de generation des rapports patients (Tache 6)."""

import numpy as np

from src.decision.engine import generer_decision_clinique
from src.decision.rules import appliquer_regle_securite_negatif
from src.decision.triage import appliquer_triage
from src.reporting.report_generator import (
    creer_rapport_decision,
    generer_rapports_batch,
    creer_tableau_bord,
)


CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]


def _build_decision(prob_vector, patient_id="P_00001"):
    decision = generer_decision_clinique(patient_id, np.array(prob_vector), CLASSES)
    decision = appliquer_regle_securite_negatif(decision)
    decision = appliquer_triage(decision)
    return decision


def test_creer_rapport_decision_sections_principales():
    decision = _build_decision([0.90, 0.05, 0.03, 0.02], "P_10000")
    rapport = creer_rapport_decision("P_10000", decision, decision.confiance)

    assert "RAPPORT D'AIDE A LA DECISION" in rapport
    assert "PREDICTION PRINCIPALE" in rapport
    assert "SCORES PAR CLASSE" in rapport
    assert "RECOMMANDATIONS CLINIQUES" in rapport
    assert "ELEMENTS D'ATTENTION" in rapport
    assert "Patient ID: P_10000" in rapport


def test_rapport_contient_alerte_securite_notumor():
    decision = _build_decision([0.01, 0.01, 0.97, 0.01], "P_20000")
    # 0.97 >= 0.95 => pas d'alerte
    rapport = creer_rapport_decision("P_20000", decision, decision.confiance)
    assert "risque faux negatif" not in rapport

    decision_alert = _build_decision([0.02, 0.03, 0.92, 0.03], "P_20001")
    rapport_alert = creer_rapport_decision(
        "P_20001", decision_alert, decision_alert.confiance
    )
    assert "risque faux negatif" in rapport_alert


def test_generer_rapports_batch_retourne_un_rapport_par_decision():
    decisions = [
        _build_decision([0.90, 0.05, 0.03, 0.02], "P_A"),
        _build_decision([0.10, 0.70, 0.15, 0.05], "P_B"),
        _build_decision([0.02, 0.03, 0.92, 0.03], "P_C"),
    ]

    rapports = generer_rapports_batch(decisions)
    assert len(rapports) == 3
    assert "Patient ID: P_A" in rapports[0]
    assert "Patient ID: P_B" in rapports[1]
    assert "Patient ID: P_C" in rapports[2]


def test_creer_tableau_bord_colonnes_attendues():
    decisions = [
        _build_decision([0.90, 0.05, 0.03, 0.02], "P_1"),
        _build_decision([0.10, 0.70, 0.15, 0.05], "P_2"),
    ]
    df = creer_tableau_bord(decisions)

    expected_cols = {
        "patient_id",
        "classe_predite",
        "confiance",
        "niveau_confiance",
        "priorite",
        "revision_requise",
        "alerte_securite",
    }
    assert expected_cols.issubset(set(df.columns))
    assert len(df) == 2
