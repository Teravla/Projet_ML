"""Public API for reporting utilities (Task 6)."""

from src.reporting.report_generator import (
    creer_rapport_decision,
    creer_tableau_bord,
    generer_rapports_batch,
)

__all__ = [
    "creer_rapport_decision",
    "generer_rapports_batch",
    "creer_tableau_bord",
]
