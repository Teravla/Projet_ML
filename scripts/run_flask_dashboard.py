#!/usr/bin/env python3
"""API Flask pour le dashboard SAD.

Lance un serveur web qui expose les donnees du pipeline SAD en JSON.

Usage:
    python scripts/run_flask_dashboard.py

    Puis ouvrir: http://localhost:5000/dashboard
"""

import sys
from pathlib import Path
import numpy as np
from datetime import datetime

try:
    from flask import Flask, jsonify
except ImportError:
    print("Flask not installed. Install with: pip install flask")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.decision.engine import generer_recommandation
from src.decision.rules import appliquer_regle_securite_negatif
from src.decision.triage import appliquer_triage
from src.evaluation.analysis import analyser_performance_sad
from src.config.thresholds import GRAVITE_CLINIQUE

app = Flask(__name__, static_folder=str(PROJECT_ROOT / "web"), static_url_path="/web")

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
RNG = np.random.default_rng(42)


def generate_decisions(n_cases: int = 120):
    """Genere un lot de decisions SAD simulees."""
    probas = RNG.dirichlet(alpha=[1.4, 1.2, 1.8, 1.1], size=n_cases)

    for idx in [7, 33, 89]:
        if idx < n_cases:
            probas[idx] = np.array([0.02, 0.03, 0.92, 0.03])

    decisions = []
    for i in range(n_cases):
        pid = f"P_{i+1:05d}"
        d = generer_recommandation(probas[i], CLASSES, patient_id=pid)
        d = appliquer_regle_securite_negatif(d)
        d = appliquer_triage(d)
        decisions.append(d)

    return decisions


@app.route("/")
def index():
    """Redirige vers le dashboard."""
    return app.send_static_file("dashboard.html")


@app.route("/dashboard")
def dashboard():
    """Serve le fichier dashboard.html depuis web/."""
    return app.send_static_file("dashboard.html")


@app.route("/api/decisions", methods=["GET"])
def api_decisions():
    """Retourne les decisions SAD."""
    decisions = generate_decisions(n_cases=120)

    data = []
    for d in decisions[:10]:  # Top 10 pour affichage
        data.append(
            {
                "patient_id": d.patient_id,
                "classe_predite": d.classe_predite,
                "confiance": round(d.confiance, 3),
                "niveau_confiance": d.niveau_confiance,
                "decision": d.decision,
                "action_recommandee": d.action_recommandee,
                "priorite": d.priorite,
                "revision_requise": d.revision_requise,
                "alerte_securite": d.alerte_securite,
                "probabilites": {k: round(v, 3) for k, v in d.probabilites.items()},
            }
        )

    return jsonify({"decisions": data, "total": len(decisions)})


@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    """Retourne les metriques phase 7."""
    decisions = generate_decisions(n_cases=120)

    y_true = RNG.choice(CLASSES, size=len(decisions), p=[0.28, 0.22, 0.30, 0.20])
    y_pred = [d.classe_predite for d in decisions]
    confiances = [d.confiance for d in decisions]
    revision_requise = [d.revision_requise for d in decisions]

    resultats = analyser_performance_sad(y_true, y_pred, confiances, revision_requise)

    return jsonify(
        {
            "accuracy_globale": round(resultats["accuracy_globale"], 4),
            "taux_couverture_automatique": round(
                resultats["taux_couverture_automatique"], 4
            ),
            "objectif_haute_confiance": {
                "n_cas": resultats["objectif_haute_confiance"]["n_cas"],
                "accuracy": (
                    round(resultats["objectif_haute_confiance"]["accuracy"], 4)
                    if not np.isnan(resultats["objectif_haute_confiance"]["accuracy"])
                    else None
                ),
                "objectif_atteint": resultats["objectif_haute_confiance"][
                    "objectif_atteint"
                ],
            },
            "couts": {
                "FN": resultats["couts"]["FN"],
                "FP": resultats["couts"]["FP"],
                "Revision": resultats["couts"]["Revision"],
                "Cost_total": resultats["couts"]["Cost_total"],
            },
        }
    )


@app.route("/api/stats", methods=["GET"])
def api_stats():
    """Retourne les statistiques globales."""
    decisions = generate_decisions(n_cases=120)

    n_total = len(decisions)
    n_haute = sum(1 for d in decisions if d.niveau_confiance == "HAUTE")
    n_moyenne = sum(1 for d in decisions if d.niveau_confiance == "MOYENNE")
    n_faible = sum(1 for d in decisions if d.niveau_confiance == "FAIBLE")
    n_tres_faible = sum(1 for d in decisions if d.niveau_confiance == "TRES_FAIBLE")

    n_urgente = sum(1 for d in decisions if "URGENTE" in d.priorite)
    n_elevee = sum(
        1 for d in decisions if "Elevee" in d.priorite or "Élevée" in d.priorite
    )
    n_normale = sum(1 for d in decisions if "Normale" in d.priorite)
    n_routine = sum(1 for d in decisions if "Routine" in d.priorite)

    n_alertes = sum(1 for d in decisions if d.alerte_securite)
    n_revisions = sum(1 for d in decisions if d.revision_requise)

    return jsonify(
        {
            "n_total": n_total,
            "confiance_distribution": {
                "HAUTE": {
                    "count": n_haute,
                    "percent": round(n_haute / n_total * 100, 1),
                },
                "MOYENNE": {
                    "count": n_moyenne,
                    "percent": round(n_moyenne / n_total * 100, 1),
                },
                "FAIBLE": {
                    "count": n_faible,
                    "percent": round(n_faible / n_total * 100, 1),
                },
                "TRES_FAIBLE": {
                    "count": n_tres_faible,
                    "percent": round(n_tres_faible / n_total * 100, 1),
                },
            },
            "priorite_distribution": {
                "URGENTE": {
                    "count": n_urgente,
                    "percent": round(n_urgente / n_total * 100, 1),
                },
                "ELEVEE": {
                    "count": n_elevee,
                    "percent": round(n_elevee / n_total * 100, 1),
                },
                "NORMALE": {
                    "count": n_normale,
                    "percent": round(n_normale / n_total * 100, 1),
                },
                "ROUTINE": {
                    "count": n_routine,
                    "percent": round(n_routine / n_total * 100, 1),
                },
            },
            "alertes": n_alertes,
            "revisions": n_revisions,
            "couverture_automatique": round((n_total - n_revisions) / n_total * 100, 1),
        }
    )


@app.route("/api/rapport/<patient_id>", methods=["GET"])
def api_rapport(patient_id: str):
    """Retourne le rapport textuel pour un patient."""
    from src.reporting.report_generator import creer_rapport_decision

    decisions = generate_decisions(n_cases=120)
    decision = next((d for d in decisions if d.patient_id == patient_id), None)

    if not decision:
        return jsonify({"error": f"Patient {patient_id} not found"}), 404

    rapport = creer_rapport_decision(patient_id, decision, decision.confiance)
    return jsonify({"patient_id": patient_id, "rapport": rapport})


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🚀 Lancement du serveur SAD Dashboard")
    print("=" * 70)
    print("\n📊 Dashboard disponible à:")
    print("   http://localhost:5000/dashboard")
    print("\n📡 Endpoints API:")
    print("   GET  http://localhost:5000/api/decisions")
    print("   GET  http://localhost:5000/api/stats")
    print("   GET  http://localhost:5000/api/metrics")
    print("   GET  http://localhost:5000/api/rapport/<patient_id>")
    print("   GET  http://localhost:5000/api/health")
    print("\n🛑 Appuyer sur CTRL+C pour arrêter le serveur")
    print("=" * 70 + "\n")

    app.run(debug=True, host="localhost", port=5000)
