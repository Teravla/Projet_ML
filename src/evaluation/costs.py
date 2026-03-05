"""Calcul du cout clinique (FN/FP/Revision)."""

from typing import Iterable, Dict

from src.config.thresholds import CostParameters


def compter_fn_fp_tumeur(
    y_true: Iterable[str], y_pred: Iterable[str], classe_saine: str = "notumor"
) -> Dict[str, int]:
    """Compte FN/FP pour un scenario de depistage tumeur vs non tumeur.

    - Faux negatif (FN): vraie classe tumeur, prediction classe saine.
    - Faux positif (FP): vraie classe saine, prediction tumeur.
    """
    fn = 0
    fp = 0

    for true_label, pred_label in zip(y_true, y_pred):
        true_is_saine = str(true_label).lower() == classe_saine.lower()
        pred_is_saine = str(pred_label).lower() == classe_saine.lower()

        if (not true_is_saine) and pred_is_saine:
            fn += 1
        elif true_is_saine and (not pred_is_saine):
            fp += 1

    return {"FN": fn, "FP": fp}


def calculer_cout_total(
    faux_negatifs: int,
    faux_positifs: int,
    revisions: int,
    cost_params: CostParameters | None = None,
) -> int:
    """Calcule le cout total metier.

    Formule du cahier des charges:
    Cost_total = (FN * 1000) + (FP * 100) + (Revision * 50)
    """
    if cost_params is None:
        cost_params = CostParameters()

    return (
        faux_negatifs * cost_params.faux_negatif
        + faux_positifs * cost_params.faux_positif
        + revisions * cost_params.revision
    )


def analyser_couts(
    y_true: Iterable[str],
    y_pred: Iterable[str],
    revisions: int,
    classe_saine: str = "notumor",
    cost_params: CostParameters | None = None,
) -> Dict[str, int]:
    """Retourne un resume complet FN/FP/Revision/Cout total."""
    counts = compter_fn_fp_tumeur(y_true, y_pred, classe_saine=classe_saine)
    cout_total = calculer_cout_total(
        counts["FN"], counts["FP"], revisions, cost_params=cost_params
    )
    return {
        "FN": counts["FN"],
        "FP": counts["FP"],
        "Revision": revisions,
        "Cost_total": cout_total,
    }
