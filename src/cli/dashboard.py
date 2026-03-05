#!/usr/bin/env python3
"""API Flask pour le dashboard SAD.

Lance un serveur web qui expose les donnees du pipeline SAD en JSON.

Usage:
    python src/cli/dashboard.py

    Puis ouvrir: http://localhost:5000/dashboard
"""

import sys
from pathlib import Path
import keras
import numpy as np
from datetime import datetime
from typing import Optional
from io import BytesIO
from reportlab.pdfgen import canvas

try:
    from flask import Flask, jsonify, request, make_response
except ImportError:
    print("Flask not installed. Install with: pip install flask")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.decision.engine import generer_recommandation
from src.decision.rules import appliquer_regle_securite_negatif
from src.decision.triage import appliquer_triage
from src.evaluation.analysis import analyser_performance_sad
from src.data.loader import load_dataset_split

app = Flask(__name__, static_folder=str(PROJECT_ROOT / "web"), static_url_path="")

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
RNG = np.random.default_rng(42)

# Global model state
MODEL_STATE = {
    "model": None,
    "test_images": None,
    "test_labels": None,
    "class_names": CLASSES,
    "model_loaded": False,
    "error_message": None,
    "model_path": None,
    "last_decisions": None,  # Cache des dernières décisions générées
    "last_true_labels": None,  # Cache des vrais labels correspondants
}


def find_latest_model() -> Optional[Path]:
    """Cherche le dernier modèle sauvegardé dans artifacts/models/."""
    artifacts_dir = PROJECT_ROOT / "artifacts" / "models"
    if not artifacts_dir.exists():
        return None

    # Chercher les modèles disponibles
    model_files = list(artifacts_dir.glob("*.keras"))
    if not model_files:
        return None

    # Retourner le plus récent
    return max(model_files, key=lambda p: p.stat().st_mtime)


def load_model_and_data() -> bool:
    """Charge le modèle et les données de test."""
    try:
        model_path = find_latest_model()
        if not model_path:
            MODEL_STATE["error_message"] = "Aucun modèle trouvé. Lancez l'entraînement."
            MODEL_STATE["model_loaded"] = False
            return False

        # Charger le modèle
        print(f"📦 Chargement du modèle: {model_path.name}")
        MODEL_STATE["model"] = keras.models.load_model(model_path)
        MODEL_STATE["model_path"] = str(model_path)

        # Charger les données de test
        print("📊 Chargement des données de test...")
        img_size = (224, 224)  # Default size, can be adjusted

        # Chercher la taille d'image à partir du modèle input shape
        try:
            input_shape = MODEL_STATE["model"].input_shape
            if len(input_shape) > 1:
                img_size = (input_shape[1], input_shape[2])
        except:
            pass

        test_dir = PROJECT_ROOT / "data" / "Testing"
        if not test_dir.exists():
            MODEL_STATE["error_message"] = "Répertoire 'data/Testing' non trouvé"
            MODEL_STATE["model_loaded"] = False
            return False

        test_split = load_dataset_split(test_dir, image_size=img_size)
        MODEL_STATE["test_images"] = test_split.images
        MODEL_STATE["test_labels"] = test_split.labels
        MODEL_STATE["class_names"] = test_split.class_names
        MODEL_STATE["model_loaded"] = True
        MODEL_STATE["error_message"] = None

        print(f"✓ Modèle chargé: {model_path.name}")
        print(f"✓ Données de test chargées: {len(test_split.images)} images")
        return True

    except Exception as e:
        MODEL_STATE["error_message"] = f"Erreur lors du chargement: {str(e)}"
        MODEL_STATE["model_loaded"] = False
        print(f"❌ Erreur: {str(e)}")
        return False


def generate_decisions_from_model(n_cases: int = 120) -> list:
    """Génère les décisions basées sur les prédictions du modèle réel."""
    if not MODEL_STATE["model_loaded"] or MODEL_STATE["model"] is None:
        # Fallback sur les données simulées si pas de modèle
        return generate_decisions_simulated(n_cases)

    try:
        # Shuffler les indices pour avoir un mix de classes (éviter de prendre que glioma)
        total_samples = len(MODEL_STATE["test_images"])
        indices = np.random.choice(
            total_samples, size=min(n_cases, total_samples), replace=False
        )

        # Prendre les cas correspondants aux indices shufflés
        test_images = MODEL_STATE["test_images"][indices]
        test_labels = MODEL_STATE["test_labels"][indices]

        # Normaliser les images [0-1] si nécessaire (le modèle a été entraîné avec normalisation)
        if test_images.max() > 1.0:
            test_images_normalized = test_images / 255.0
        else:
            test_images_normalized = test_images

        # Faire les prédictions
        predictions = MODEL_STATE["model"].predict(test_images_normalized, verbose=0)

        decisions = []
        true_labels = []
        for i, (image, label, pred) in enumerate(
            zip(test_images, test_labels, predictions)
        ):
            pid = f"P_{i+1:05d}"

            # Appliquer softmax pour obtenir des probabilités normalisées [0,1]
            # Le modèle peut ne pas avoir de softmax en dernière couche
            exp_pred = np.exp(pred - np.max(pred))  # Stabilité numérique
            probas = exp_pred / exp_pred.sum()

            # Générer la recommandation basée sur les vraies probabilités
            d = generer_recommandation(
                probas, MODEL_STATE["class_names"], patient_id=pid
            )
            d = appliquer_regle_securite_negatif(d)
            d = appliquer_triage(d)
            decisions.append(d)

            # Stocker le vrai label
            true_labels.append(MODEL_STATE["class_names"][int(label)])

        # Mettre en cache les décisions et vrais labels pour api_metrics
        MODEL_STATE["last_decisions"] = decisions
        MODEL_STATE["last_true_labels"] = true_labels

        return decisions
    except Exception as e:
        print(f"❌ Erreur lors de la génération des décisions: {str(e)}")
        import traceback

        traceback.print_exc()
        return generate_decisions_simulated(n_cases)


def generate_decisions_simulated(n_cases: int = 120):
    """Génère des décisions simulées (fallback)."""
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


def generate_decisions(n_cases: int = 120):
    """Wrapper pour générer les décisions (réelles ou simulées)."""
    return generate_decisions_from_model(n_cases)


def get_or_generate_decisions(n_cases: int = 120) -> tuple:
    """Récupère les décisions du cache ou les génère si nécessaire.

    Returns:
        tuple: (decisions, true_labels)
    """
    # Si le cache existe et contient le bon nombre de cas, le retourner
    if (
        MODEL_STATE["last_decisions"] is not None
        and MODEL_STATE["last_true_labels"] is not None
        and len(MODEL_STATE["last_decisions"]) == n_cases
    ):
        return MODEL_STATE["last_decisions"], MODEL_STATE["last_true_labels"]

    # Sinon, générer de nouvelles décisions (qui seront automatiquement mises en cache)
    decisions = generate_decisions(n_cases)
    true_labels = MODEL_STATE["last_true_labels"]

    return decisions, true_labels


@app.route("/")
def index():
    """Redirige vers le dashboard."""
    return app.send_static_file("dashboard.html")


@app.route("/dashboard")
def dashboard():
    """Serve le fichier dashboard.html depuis web/."""
    return app.send_static_file("dashboard.html")


@app.route("/api/status", methods=["GET"])
def api_status():
    """Retourne le statut du modèle et du système."""
    model_name = (
        Path(MODEL_STATE["model_path"]).name if MODEL_STATE["model_path"] else None
    )
    return jsonify(
        {
            "model_loaded": MODEL_STATE["model_loaded"],
            "model_filename": model_name,
            "error": MODEL_STATE["error_message"],
            "data_available": MODEL_STATE["model_loaded"]
            and MODEL_STATE["test_images"] is not None
            and len(MODEL_STATE["test_images"]) > 0,
            "num_test_samples": (
                len(MODEL_STATE["test_images"])
                if MODEL_STATE["test_images"] is not None
                else 0
            ),
        }
    )


@app.route("/api/decisions", methods=["GET"])
def api_decisions():
    """Retourne les decisions SAD avec pagination."""
    # Récupérer les paramètres de pagination
    limit = request.args.get("limit", 10, type=int)
    offset = request.args.get("offset", 0, type=int)

    decisions, _ = get_or_generate_decisions(n_cases=120)

    # Appliquer la pagination
    paginated_decisions = decisions[offset : offset + limit]

    data = []
    for d in paginated_decisions:
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
    decisions, y_true = get_or_generate_decisions(n_cases=120)

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
    decisions, _ = get_or_generate_decisions(n_cases=120)

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


@app.route("/api/rapport/<patient_id>/pdf", methods=["GET"])
def api_rapport_pdf(patient_id: str):
    """Génère un rapport PDF pour un patient."""

    decisions, _ = get_or_generate_decisions(n_cases=120)
    decision = next((d for d in decisions if d.patient_id == patient_id), None)

    if not decision:
        return jsonify({"error": f"Patient {patient_id} not found"}), 404

    # Créer le PDF en mémoire
    buffer = BytesIO()
    c = canvas.Canvas(buffer)

    # Configuration
    c.setFont("Helvetica", 11)
    y = 750
    line_height = 15

    # En-tête
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "=" * 60)
    y -= line_height
    c.drawString(50, y, "RAPPORT D'AIDE A LA DECISION")
    y -= line_height
    c.drawString(50, y, "=" * 60)
    y -= line_height * 1.5

    # Patient info
    c.setFont("Helvetica", 10)
    c.drawString(
        50, y, f"Patient ID: {patient_id}   Date: {datetime.now().strftime('%d/%m/%Y')}"
    )
    y -= line_height * 2

    # Prédiction principale
    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, y, "PREDICTION PRINCIPALE")
    y -= line_height * 0.8
    c.setFont("Helvetica", 10)
    c.drawString(50, y, "---")
    y -= line_height
    c.drawString(50, y, f"Classe: {decision.classe_predite}")
    y -= line_height
    c.drawString(50, y, f"Confiance: {decision.confiance*100:.1f}%")
    y -= line_height
    c.drawString(50, y, f"Niveau de certitude: {decision.niveau_confiance}")
    y -= line_height * 1.5

    # Scores par classe
    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, y, "SCORES PAR CLASSE")
    y -= line_height * 0.8
    c.setFont("Helvetica", 10)
    c.drawString(50, y, "---")
    y -= line_height

    for classe, prob in sorted(
        decision.probabilites.items(), key=lambda x: x[1], reverse=True
    ):
        c.drawString(50, y, f"• {classe}: {prob*100:.1f}%")
        y -= line_height

    y -= line_height * 0.5

    # Recommandations cliniques
    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, y, "RECOMMANDATIONS CLINIQUES")
    y -= line_height * 0.8
    c.setFont("Helvetica", 10)
    c.drawString(50, y, "---")
    y -= line_height
    c.drawString(50, y, f"Diagnostic: {decision.decision}")
    y -= line_height
    c.drawString(50, y, f"Action: {decision.action_recommandee}")
    y -= line_height
    c.drawString(50, y, f"Priorite: {decision.priorite}")
    y -= line_height
    revision_text = (
        "Requise" if decision.revision_requise else "Optionnelle (validation finale)"
    )
    c.drawString(50, y, f"Revision humaine: {revision_text}")
    y -= line_height * 1.5

    # Éléments d'attention
    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, y, "ELEMENTS D'ATTENTION")
    y -= line_height * 0.8
    c.setFont("Helvetica", 10)
    c.drawString(50, y, "---")
    y -= line_height

    if decision.alerte_securite:
        c.drawString(50, y, "• [ALERTE] Revision obligatoire detectee")
        y -= line_height

    if decision.classe_predite != "notumor":
        c.drawString(50, y, "• Tumeur suspecte detectee")
        y -= line_height
        c.drawString(50, y, "• IRM de controle recommandee")
        y -= line_height
    else:
        c.drawString(50, y, "• Pas de tumeur detectee")
        y -= line_height

    y -= line_height
    c.drawString(50, y, "=" * 60)

    # Sauvegarder le PDF
    c.save()
    buffer.seek(0)

    response = make_response(buffer.read())
    response.headers["Content-Type"] = "application/pdf"
    response.headers["Content-Disposition"] = (
        f"inline; filename=rapport_{patient_id}.pdf"
    )
    return response


@app.route("/api/rapport/<patient_id>", methods=["GET"])
def api_rapport(patient_id: str):
    """Retourne le rapport textuel pour un patient."""
    from src.reporting.report_generator import creer_rapport_decision

    decisions, _ = get_or_generate_decisions(n_cases=120)
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
    print("Lancement du serveur SAD Dashboard")
    print("=" * 70)

    # Charger le modèle et les données
    print("\nChargement du modèle et des données...")
    if load_model_and_data():
        print("OK: Modèle et données chargés avec succès")
    else:
        print(f"AVERTISSEMENT: {MODEL_STATE['error_message']}")
        print("   Les prédictions seront simulées pour le moment")

    print("\nDashboard disponible a:")
    print("   http://localhost:5000/dashboard")
    print("\nEndpoints API:")
    print("   GET  http://localhost:5000/api/status")
    print("   GET  http://localhost:5000/api/decisions?limit=10&offset=0  (pagination)")
    print("   GET  http://localhost:5000/api/stats")
    print("   GET  http://localhost:5000/api/metrics")
    print("   GET  http://localhost:5000/api/rapport/<patient_id>")
    print("   GET  http://localhost:5000/api/rapport/<patient_id>/pdf  (NOUVEAU!)")
    print("   GET  http://localhost:5000/api/health")
    print("\n🛑 Appuyer sur CTRL+C pour arrêter le serveur")
    print("=" * 70 + "\n")

    # Désactiver le reloader en debug pour éviter les recharges multiples du modèle
    app.run(debug=True, use_reloader=False, host="localhost", port=5000)
