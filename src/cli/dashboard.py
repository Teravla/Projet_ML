#!/usr/bin/env python3
"""API Flask pour le dashboard SAD.

Lance un serveur web qui expose les donnees du pipeline SAD en JSON.

Usage:
    python src/cli/dashboard.py

    Puis ouvrir: http://localhost:5000/dashboard
"""

import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional
from io import BytesIO
import keras
import numpy as np
from reportlab.pdfgen import canvas
from flask import Flask, jsonify, request, make_response
from src.decision.engine import generer_recommandation
from src.decision.rules import appliquer_regle_securite_negatif
from src.decision.triage import appliquer_triage
from src.evaluation.analysis import analyser_performance_sad
from src.data.loader import load_dataset_split
from src.reporting.report_generator import creer_rapport_decision

PROJECT_ROOT = Path(__file__).resolve().parents[2]


app = Flask(__name__, static_folder=str(PROJECT_ROOT / "web"), static_url_path="")

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
RNG = np.random.default_rng(42)

# Confidence level constants
CONFIDENCE_HAUTE = "HAUTE"
CONFIDENCE_MOYENNE = "MOYENNE"
CONFIDENCE_FAIBLE = "FAIBLE"
CONFIDENCE_TRES_FAIBLE = "TRES_FAIBLE"

# Priority constants
PRIORITY_URGENTE = "URGENTE"
PRIORITY_ELEVEE = "Elevee"
PRIORITY_ELEVEE_ACCENT = "Élevée"
PRIORITY_NORMALE = "Normale"
PRIORITY_ROUTINE = "Routine"

# Class name constant
CLASS_NO_TUMOR = "notumor"

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


def get_image_size_from_model(model: keras.Model) -> tuple[int, int]:
    """Extrait la taille d'image du modèle ou retourne la taille par défaut."""
    default_size = (224, 224)
    input_shape = model.input_shape
    if len(input_shape) > 1:
        return (input_shape[1], input_shape[2])
    return default_size


def load_test_data(img_size: tuple[int, int]) -> tuple:
    """Charge les données de test."""
    test_dir = PROJECT_ROOT / "data" / "Testing"
    if not test_dir.exists():
        raise FileNotFoundError("Répertoire 'data/Testing' non trouvé")

    test_split = load_dataset_split(test_dir, image_size=img_size)
    return test_split.images, test_split.labels, test_split.class_names


def load_model_and_data() -> bool:
    """Charge le modèle et les données de test."""
    model_path = find_latest_model()
    if not model_path:
        MODEL_STATE["error_message"] = "Aucun modèle trouvé. Lancez l'entraînement."
        MODEL_STATE["model_loaded"] = False
        return False

    # Charger le modèle
    print(f"[*] Chargement du modèle: {model_path.name}")
    MODEL_STATE["model"] = keras.models.load_model(model_path)
    MODEL_STATE["model_path"] = str(model_path)

    # Charger les données de test
    print("[*] Chargement des données de test...")
    img_size = get_image_size_from_model(MODEL_STATE["model"])
    images, labels, class_names = load_test_data(img_size)

    MODEL_STATE["test_images"] = images
    MODEL_STATE["test_labels"] = labels
    MODEL_STATE["class_names"] = class_names
    MODEL_STATE["model_loaded"] = True
    MODEL_STATE["error_message"] = None

    print(f"[OK] Modèle chargé: {model_path.name}")
    print(f"[OK] Données de test chargées: {len(images)} images")
    return True


def normalize_images(images: np.ndarray) -> np.ndarray:
    """Normalise les images si nécessaire."""
    return images / 255.0 if images.max() > 1.0 else images


def apply_softmax(predictions: np.ndarray) -> np.ndarray:
    """Applique softmax pour obtenir des probabilités normalisées."""
    exp_pred = np.exp(predictions - np.max(predictions))
    return exp_pred / exp_pred.sum()


def create_decision_for_case(
    label: int, pred: np.ndarray, case_index: int, class_names: list[str]
) -> tuple:
    """Crée une décision pour un cas donné."""
    pid = f"P_{case_index+1:05d}"
    probas = apply_softmax(pred)

    # Générer la recommandation basée sur les vraies probabilités
    decision = generer_recommandation(probas, class_names, patient_id=pid)
    decision = appliquer_regle_securite_negatif(decision)
    decision = appliquer_triage(decision)

    true_label = class_names[int(label)]
    return decision, true_label


def get_test_data_samples(n_cases: int) -> tuple[np.ndarray, np.ndarray]:
    """Récupère des échantillons aléatoires des données de test."""
    test_images = MODEL_STATE["test_images"]
    test_labels = MODEL_STATE["test_labels"]

    # Type guard for unsubscriptable check
    if test_images is None or test_labels is None:
        raise ValueError("Test data not available")

    # Assert type for pylint
    assert isinstance(test_images, np.ndarray)
    assert isinstance(test_labels, np.ndarray)

    # Shuffler les indices pour avoir un mix de classes
    total_samples = len(test_images)
    indices = np.random.choice(
        total_samples, size=min(n_cases, total_samples), replace=False
    )

    # pylint: disable=unsubscriptable-object
    return test_images[indices], test_labels[indices]


def process_predictions_to_decisions(
    labels: np.ndarray, predictions: np.ndarray, class_names: list[str]
) -> tuple[list, list]:
    """Transforme les prédictions en décisions SAD."""
    decisions = []
    true_labels = []
    for i, (label, pred) in enumerate(zip(labels, predictions)):
        decision, true_label = create_decision_for_case(label, pred, i, class_names)
        decisions.append(decision)
        true_labels.append(true_label)
    return decisions, true_labels


def generate_model_predictions(n_cases: int) -> tuple[list, list]:
    """Génère les prédictions et les décisions depuis le modèle."""
    selected_images, selected_labels = get_test_data_samples(n_cases)
    normalized_images = normalize_images(selected_images)
    predictions = MODEL_STATE["model"].predict(normalized_images, verbose=0)
    return process_predictions_to_decisions(
        selected_labels, predictions, MODEL_STATE["class_names"]
    )


def generate_decisions_from_model(n_cases: int = 120) -> list:
    """Génère les décisions basées sur les prédictions du modèle réel."""
    if not MODEL_STATE["model_loaded"] or MODEL_STATE["model"] is None:
        return generate_decisions_simulated(n_cases)

    try:
        decisions, true_labels = generate_model_predictions(n_cases)
    except (ValueError, TypeError, KeyError) as e:
        print(f"[ERROR] Erreur génération décisions: {str(e)}")
        traceback.print_exc()
        return generate_decisions_simulated(n_cases)

    # Mettre en cache
    MODEL_STATE["last_decisions"] = decisions
    MODEL_STATE["last_true_labels"] = true_labels

    return decisions


def generate_decisions_simulated(n_cases: int = 120):
    """Génère des décisions simulées (fallback)."""
    probas = RNG.dirichlet(alpha=[1.4, 1.2, 1.8, 1.1], size=n_cases)

    for idx in (7, 33, 89):
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
    n_haute = sum(1 for d in decisions if d.niveau_confiance == CONFIDENCE_HAUTE)
    n_moyenne = sum(1 for d in decisions if d.niveau_confiance == CONFIDENCE_MOYENNE)
    n_faible = sum(1 for d in decisions if d.niveau_confiance == CONFIDENCE_FAIBLE)
    n_tres_faible = sum(
        1 for d in decisions if d.niveau_confiance == CONFIDENCE_TRES_FAIBLE
    )

    n_urgente = sum(1 for d in decisions if PRIORITY_URGENTE in d.priorite)
    n_elevee = sum(
        1
        for d in decisions
        if PRIORITY_ELEVEE in d.priorite or PRIORITY_ELEVEE_ACCENT in d.priorite
    )
    n_normale = sum(1 for d in decisions if PRIORITY_NORMALE in d.priorite)
    n_routine = sum(1 for d in decisions if PRIORITY_ROUTINE in d.priorite)

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


def draw_pdf_header(
    c: canvas.Canvas, patient_id: str, y_pos: float, line_height: float
) -> float:
    """Dessine l'en-tête du PDF."""
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y_pos, "=" * 60)
    y_pos -= line_height
    c.drawString(50, y_pos, "RAPPORT D'AIDE A LA DECISION")
    y_pos -= line_height
    c.drawString(50, y_pos, "=" * 60)
    y_pos -= line_height * 1.5

    c.setFont("Helvetica", 10)
    c.drawString(
        50,
        y_pos,
        f"Patient ID: {patient_id}   Date: {datetime.now().strftime('%d/%m/%Y')}",
    )
    return y_pos - line_height * 2


def draw_pdf_prediction(
    c: canvas.Canvas, decision, y_pos: float, line_height: float
) -> float:
    """Dessine la section prédiction principale."""
    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, y_pos, "PREDICTION PRINCIPALE")
    y_pos -= line_height * 0.8
    c.setFont("Helvetica", 10)
    c.drawString(50, y_pos, "---")
    y_pos -= line_height
    c.drawString(50, y_pos, f"Classe: {decision.classe_predite}")
    y_pos -= line_height
    c.drawString(50, y_pos, f"Confiance: {decision.confiance*100:.1f}%")
    y_pos -= line_height
    c.drawString(50, y_pos, f"Niveau de certitude: {decision.niveau_confiance}")
    return y_pos - line_height * 1.5


def draw_pdf_scores(
    c: canvas.Canvas, decision, y_pos: float, line_height: float
) -> float:
    """Dessine la section scores par classe."""
    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, y_pos, "SCORES PAR CLASSE")
    y_pos -= line_height * 0.8
    c.setFont("Helvetica", 10)
    c.drawString(50, y_pos, "---")
    y_pos -= line_height

    for classe, prob in sorted(
        decision.probabilites.items(), key=lambda x: x[1], reverse=True
    ):
        c.drawString(50, y_pos, f"• {classe}: {prob*100:.1f}%")
        y_pos -= line_height

    return y_pos - line_height * 0.5


def draw_pdf_recommendations(
    c: canvas.Canvas, decision, y_pos: float, line_height: float
) -> float:
    """Dessine la section recommandations cliniques."""
    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, y_pos, "RECOMMANDATIONS CLINIQUES")
    y_pos -= line_height * 0.8
    c.setFont("Helvetica", 10)
    c.drawString(50, y_pos, "---")
    y_pos -= line_height
    c.drawString(50, y_pos, f"Diagnostic: {decision.decision}")
    y_pos -= line_height
    c.drawString(50, y_pos, f"Action: {decision.action_recommandee}")
    y_pos -= line_height
    c.drawString(50, y_pos, f"Priorite: {decision.priorite}")
    y_pos -= line_height
    revision_text = (
        "Requise" if decision.revision_requise else "Optionnelle (validation finale)"
    )
    c.drawString(50, y_pos, f"Revision humaine: {revision_text}")
    return y_pos - line_height * 1.5


def draw_pdf_attention(
    c: canvas.Canvas, decision, y_pos: float, line_height: float
) -> float:
    """Dessine la section éléments d'attention."""
    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, y_pos, "ELEMENTS D'ATTENTION")
    y_pos -= line_height * 0.8
    c.setFont("Helvetica", 10)
    c.drawString(50, y_pos, "---")
    y_pos -= line_height

    if decision.alerte_securite:
        c.drawString(50, y_pos, "• [ALERTE] Revision obligatoire detectee")
        y_pos -= line_height

    if decision.classe_predite != CLASS_NO_TUMOR:
        c.drawString(50, y_pos, "• Tumeur suspecte detectee")
        y_pos -= line_height
        c.drawString(50, y_pos, "• IRM de controle recommandee")
        y_pos -= line_height
    else:
        c.drawString(50, y_pos, "• Pas de tumeur detectee")
        y_pos -= line_height

    y_pos -= line_height
    c.drawString(50, y_pos, "=" * 60)
    return y_pos


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
    y = 750.0
    line_height = 15

    # Dessiner les sections
    y = draw_pdf_header(c, patient_id, y, line_height)
    y = draw_pdf_prediction(c, decision, y, line_height)
    y = draw_pdf_scores(c, decision, y, line_height)
    y = draw_pdf_recommendations(c, decision, y, line_height)
    draw_pdf_attention(c, decision, y, line_height)

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
    print("\n[STOP] Appuyer sur CTRL+C pour arrêter le serveur")
    print("=" * 70 + "\n")

    # Désactiver le reloader en debug pour éviter les recharges multiples du modèle
    app.run(debug=True, use_reloader=False, host="localhost", port=5000)
