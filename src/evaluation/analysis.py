"""Analyse metier SAD: couverture, tranches de confiance, cout total."""

from typing import Sequence, Iterable

import pandas as pd

from src.evaluation.metrics import (
    accuracy_globale,
    accuracy_par_tranche_confiance,
    taux_couverture_automatique,
    verifier_objectif_haute_confiance,
)
from src.evaluation.costs import analyser_couts


def analyser_performance_sad(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    confiances: Sequence[float],
    revision_requise: Iterable[bool],
    classe_saine: str = "notumor",
) -> dict:
    """Produit un resume metier complet pour la phase 7."""
    revisions = int(sum(bool(v) for v in revision_requise))

    accuracy_global = accuracy_globale(y_true, y_pred)
    couverture_auto = taux_couverture_automatique(revision_requise)
    by_conf = accuracy_par_tranche_confiance(y_true, y_pred, confiances)
    objectif = verifier_objectif_haute_confiance(y_true, y_pred, confiances)
    costs = analyser_couts(y_true, y_pred, revisions, classe_saine=classe_saine)

    return {
        "accuracy_globale": accuracy_global,
        "taux_couverture_automatique": couverture_auto,
        "accuracy_par_tranche": by_conf,
        "objectif_haute_confiance": objectif,
        "couts": costs,
    }


def resume_business_markdown(resultats: dict) -> str:
    """Construit un resume markdown pret a afficher en notebook."""
    obj = resultats["objectif_haute_confiance"]
    costs = resultats["couts"]

    lines = [
        "### Resume metier SAD",
        f"- Accuracy globale: {resultats['accuracy_globale']:.2%}",
        f"- Couverture automatique: {resultats['taux_couverture_automatique']:.2%}",
        (
            f"- Accuracy confiance > 0.85: {obj['accuracy']:.2%} "
            f"(objectif > {obj['objectif_accuracy']:.0%})"
            if obj["n_cas"] > 0
            else "- Accuracy confiance > 0.85: N/A (aucun cas)"
        ),
        f"- FN: {costs['FN']}",
        f"- FP: {costs['FP']}",
        f"- Revisions: {costs['Revision']}",
        f"- Cost_total: {costs['Cost_total']}",
    ]
    return "\n".join(lines)
