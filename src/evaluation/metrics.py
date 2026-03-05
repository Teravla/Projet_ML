"""Metriques de performance classiques et metier."""

from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def accuracy_globale(y_true: Iterable[str], y_pred: Iterable[str]) -> float:
    """Calcule l'accuracy multiclasses."""
    y_true_arr = np.array(list(y_true))
    y_pred_arr = np.array(list(y_pred))
    if y_true_arr.size == 0:
        return 0.0
    return float((y_true_arr == y_pred_arr).mean())


def taux_couverture_automatique(revision_requise: Iterable[bool]) -> float:
    """Pourcentage de cas traites sans intervention humaine."""
    flags = np.array(list(revision_requise), dtype=bool)
    if flags.size == 0:
        return 0.0
    return float((~flags).mean())


def accuracy_par_tranche_confiance(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    confiances: Sequence[float],
    bins: Sequence[float] = (0.0, 0.50, 0.65, 0.85, 1.000001),
) -> pd.DataFrame:
    """Calcule l'accuracy par tranche de confiance."""
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    conf_arr = np.array(confiances, dtype=float)

    rows = []
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (conf_arr >= left) & (conf_arr < right)
        n = int(mask.sum())
        if n == 0:
            acc = np.nan
        else:
            acc = float((y_true_arr[mask] == y_pred_arr[mask]).mean())

        rows.append(
            {
                "tranche": f"[{left:.2f}, {right:.2f}[",
                "n_cas": n,
                "accuracy": acc,
            }
        )

    return pd.DataFrame(rows)


def verifier_objectif_haute_confiance(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    confiances: Sequence[float],
    seuil_confiance: float = 0.85,
    objectif_accuracy: float = 0.95,
) -> dict:
    """Verifie l'objectif: accuracy > 95% pour les cas confiance > 0.85."""
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    conf_arr = np.array(confiances, dtype=float)

    mask = conf_arr > seuil_confiance
    n = int(mask.sum())

    if n == 0:
        return {
            "n_cas": 0,
            "accuracy": np.nan,
            "objectif_accuracy": objectif_accuracy,
            "objectif_atteint": False,
        }

    acc = float((y_true_arr[mask] == y_pred_arr[mask]).mean())
    return {
        "n_cas": n,
        "accuracy": acc,
        "objectif_accuracy": objectif_accuracy,
        "objectif_atteint": acc > objectif_accuracy,
    }
