"""Tests phase 7: metriques metier et cout-benefice."""

from src.evaluation.analysis import analyser_performance_sad
from src.evaluation.costs import calculer_cout_total, compter_fn_fp_tumeur
from src.evaluation.metrics import (
    accuracy_par_tranche_confiance,
    taux_couverture_automatique,
    verifier_objectif_haute_confiance,
)


def test_couverture_automatique():
    revision_flags = [False, True, False, False, True]
    assert taux_couverture_automatique(revision_flags) == 3 / 5


def test_accuracy_par_tranche_retourne_dataframe():
    y_true = ["glioma", "notumor", "meningioma", "notumor"]
    y_pred = ["glioma", "notumor", "pituitary", "glioma"]
    conf = [0.92, 0.88, 0.60, 0.40]

    df = accuracy_par_tranche_confiance(y_true, y_pred, conf)
    assert {"tranche", "n_cas", "accuracy"}.issubset(df.columns)
    assert df["n_cas"].sum() == 4


def test_objectif_haute_confiance():
    y_true = ["glioma", "notumor", "meningioma", "pituitary"]
    y_pred = ["glioma", "notumor", "meningioma", "glioma"]
    conf = [0.95, 0.90, 0.86, 0.70]

    result = verifier_objectif_haute_confiance(y_true, y_pred, conf)
    assert result["n_cas"] == 3
    assert result["accuracy"] == 1.0
    assert result["objectif_atteint"] is True


def test_compter_fn_fp_tumeur():
    y_true = ["glioma", "notumor", "pituitary", "notumor"]
    y_pred = ["notumor", "glioma", "pituitary", "notumor"]

    counts = compter_fn_fp_tumeur(y_true, y_pred)
    assert counts["FN"] == 1
    assert counts["FP"] == 1


def test_calcul_cout_total():
    assert calculer_cout_total(2, 3, 4) == (2 * 1000) + (3 * 100) + (4 * 50)


def test_analyser_performance_sad_structure():
    y_true = ["glioma", "notumor", "meningioma", "notumor", "pituitary"]
    y_pred = ["glioma", "notumor", "pituitary", "glioma", "pituitary"]
    conf = [0.93, 0.88, 0.61, 0.55, 0.72]
    revision = [False, False, True, True, True]

    result = analyser_performance_sad(y_true, y_pred, conf, revision)

    assert "accuracy_globale" in result
    assert "taux_couverture_automatique" in result
    assert "accuracy_par_tranche" in result
    assert "objectif_haute_confiance" in result
    assert "couts" in result
    assert "Cost_total" in result["couts"]
