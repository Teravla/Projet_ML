"""Generation de rapports textuels d'aide a la decision."""

import pandas as pd

from src.decision.engine import ClinicalDecision
from src.reporting.templates import (
    certitude_flag,
    format_date_fr,
    priorite_badge,
    to_clinical_label,
)


def creer_rapport_decision(
    patient_id: str, prediction: ClinicalDecision, confiance: float
) -> str:
    """Genere un rapport textuel formate pour un patient.

    Args:
            patient_id: Identifiant patient
            prediction: Objet ClinicalDecision enrichi par les regles SAD
            confiance: Confiance principale (souvent prediction.confiance)

    Returns:
            Rapport texte multi-lignes
    """
    scores = sorted(prediction.probabilites.items(), key=lambda x: x[1], reverse=True)

    lines = [
        "========================================",
        "RAPPORT D'AIDE A LA DECISION",
        "========================================",
        f"Patient ID: {patient_id} Date: {format_date_fr()}",
        "",
        "PREDICTION PRINCIPALE",
        "---",
        f"Classe: {to_clinical_label(prediction.classe_predite)}",
        f"Confiance: {confiance:.1%}",
        f"Niveau de certitude: {prediction.niveau_confiance} [{certitude_flag(prediction.niveau_confiance)}]",
        "",
        "SCORES PAR CLASSE",
        "---",
    ]

    for classe, proba in scores:
        lines.append(f"- {to_clinical_label(classe)}: {proba:.1%}")

    lines.extend(
        [
            "",
            "RECOMMANDATIONS CLINIQUES",
            "---",
            f"Diagnostic: {prediction.decision}",
            f"Action: {prediction.action_recommandee}",
            f"Priorite: {priorite_badge(prediction.priorite)}",
            (
                "Revision humaine: Optionnelle (validation finale)"
                if not prediction.revision_requise
                else "Revision humaine: Obligatoire"
            ),
            "",
            "ELEMENTS D'ATTENTION",
            "---",
        ]
    )

    if prediction.alerte_securite:
        lines.append("- Verification obligatoire (risque faux negatif)")

    if prediction.classe_predite.lower() == "glioma":
        lines.append("- Tumeur maligne suspectee")
        lines.append("- IRM de controle recommandee")

    if prediction.niveau_confiance in {"FAIBLE", "TRES_FAIBLE"}:
        lines.append("- Cas incertain: relecture senior prioritaire")

    if lines[-1] == "---":
        lines.append("- Aucun element critique additionnel")

    lines.append("========================================")
    return "\n".join(lines)


def generer_rapports_batch(decisions: list[ClinicalDecision]) -> list[str]:
    """Genere un rapport textuel pour chaque decision clinique."""
    return [creer_rapport_decision(d.patient_id, d, d.confiance) for d in decisions]


def creer_tableau_bord(decisions: list[ClinicalDecision]) -> pd.DataFrame:
    """Construit un tableau de bord compact pour pilotage clinique."""
    rows = []
    for d in decisions:
        rows.append(
            {
                "patient_id": d.patient_id,
                "classe_predite": to_clinical_label(d.classe_predite),
                "confiance": d.confiance,
                "niveau_confiance": d.niveau_confiance,
                "priorite": d.priorite,
                "revision_requise": d.revision_requise,
                "alerte_securite": d.alerte_securite,
            }
        )

    return pd.DataFrame(rows).sort_values(
        by=["priorite", "confiance"], ascending=[True, False]
    )
