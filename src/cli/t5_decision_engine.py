"""Script runner pour la Tâche 5 : Moteur de Décision Clinique.

Ce script démontre le fonctionnement du SAD (Système d'Aide à la Décision)
en appliquant les règles métier, la gestion de sécurité (faux négatifs),
et le système de triage sur des prédictions de modèle.

Usage:
    python scripts/run_task5.py [--fast] [--model MODEL_TYPE] [--n-samples N]

    poetry run task t5 --fast
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import keras
import numpy as np


from src.data.loader import load_dataset_split, discover_classes, DatasetSplit
from src.data.preprocess import preprocess_dataset
from src.decision.engine import (
    traiter_batch_decisions,
    statistiques_decisions,
)
from src.decision.rules import (
    appliquer_regles_securite_batch,
    identifier_cas_limites,
    statistiques_securite,
)
from src.decision.triage import (
    appliquer_triage_batch,
    trier_par_priorite,
    generer_file_attente,
    statistiques_triage,
)
from src.config.config import DecisionThresholds
from src.enums.dataclass import RuntimeConfigT5
from src.enums.enums import ConfidenceLevel, HyperParametersStr, ModelType
from src.models.cnn import build_cnn_classifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Tâche 5 : Moteur de Décision Clinique"
    )

    parser.add_argument(
        "--fast",
        action="store_true",
        help="Mode rapide : réduit la taille des images et le nombre d'échantillons",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="cnn",
        choices=["logreg", "mlp", "cnn"],
        help="Type de modèle à utiliser pour les prédictions (défaut: cnn)",
    )

    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Nombre d'échantillons de test à traiter (défaut: tous)",
    )

    parser.add_argument(
        "--img-size", type=int, default=64, help="Taille des images (défaut: 64)"
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Chemin vers les données (défaut: data/Brain Tumor MRI Dataset)",
    )

    parser.add_argument(
        "--seuil-haute",
        type=float,
        default=0.85,
        help="Seuil haute confiance (défaut: 0.85)",
    )

    parser.add_argument(
        "--seuil-moyenne",
        type=float,
        default=0.65,
        help="Seuil confiance moyenne (défaut: 0.65)",
    )

    parser.add_argument(
        "--seuil-faible",
        type=float,
        default=0.50,
        help="Seuil confiance faible (défaut: 0.50)",
    )

    return parser.parse_args()


def charger_modele(
    model_type: str, input_shape: tuple[int, int, int], num_classes: int
) -> keras.Model:
    """Charge ou crée un modèle pour les prédictions.

    Note: Cette version simplifiée crée un nouveau modèle.
    Dans un contexte réel, on chargerait un modèle pré-entraîné depuis artifacts/.
    """

    if model_type == ModelType.LOGISTIC_REGRESSION:
        # Simuler un modèle logistique simple
        model = keras.Sequential(
            [
                keras.layers.Flatten(input_shape=input_shape),
                keras.layers.Dense(num_classes, activation="softmax"),
            ]
        )
        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )
    elif model_type == ModelType.MLP:
        model = keras.Sequential(
            [
                keras.layers.Flatten(input_shape=input_shape),
                keras.layers.Dense(256, activation="relu"),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(128, activation="relu"),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(num_classes, activation="softmax"),
            ]
        )
        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )
    else:  # cnn
        # Note: build_cnn_classifier retourne un modèle avec logits, pas softmax
        # On va le modifier pour avoir des probabilités
        model_logits = build_cnn_classifier(
            input_shape=input_shape, num_classes=num_classes
        )

        # Créer un modèle avec softmax sur les logits
        inputs = model_logits.input
        logits = model_logits.output
        probs = keras.layers.Softmax(name="probs")(logits)
        model = keras.Model(inputs=inputs, outputs=probs, name="cnn_with_softmax")

        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )

    return model


def afficher_exemple_decision(decision, idx: int = None):
    """Affiche un exemple de décision clinique formaté."""
    print("\n" + "=" * 70)
    if idx is not None:
        print(f"EXEMPLE DE DECISION #{idx}")
    print("=" * 70)
    print(f"Patient ID          : {decision.patient_id}")
    print(f"Classe prédite      : {decision.classe_predite}")
    print(f"Confiance           : {decision.confiance:.1%}")
    print(f"Niveau confiance    : {decision.niveau_confiance}")
    print(f"\nDécision            : {decision.decision}")
    print(f"Action recommandée  : {decision.action_recommandee}")
    print(f"Priorité            : {decision.priorite}")
    print(f"Révision requise    : {'OUI' if decision.revision_requise else 'NON'}")

    if decision.alerte_securite:
        print("\n  ALERTE SECURITE : Risque de faux négatif détecté!")

    print("\nScores par classe:")
    for classe, prob in sorted(
        decision.probabilites.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  • {classe:20s} : {prob:6.1%}")
    print("=" * 70)


def print_execution_header(args: argparse.Namespace) -> None:
    """Affiche l'en-tête de lancement."""
    print("\n" + "=" * 70)
    print("TACHE 5 : MOTEUR DE DECISION CLINIQUE (SAD)")
    print("=" * 70)
    print(f"Date d'exécution : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode rapide      : {args.fast}")
    print(f"Type de modèle   : {args.model.upper()}")


def build_runtime_config(args: argparse.Namespace) -> RuntimeConfigT5:
    """Construit la configuration runtime et affiche le mode."""
    img_size = min(args.img_size, 32) if args.fast else args.img_size
    n_samples_max = (args.n_samples or 200) if args.fast else args.n_samples
    if args.fast:
        print(f"⚡ Mode fast : img_size={img_size}, max_samples={n_samples_max}")

    seuils = DecisionThresholds(
        haute=args.seuil_haute,
        moyenne=args.seuil_moyenne,
        faible=args.seuil_faible,
    )
    data_path = Path(args.data_path) if args.data_path else PROJECT_ROOT / "data"
    return RuntimeConfigT5(
        img_size=img_size,
        n_samples_max=n_samples_max,
        data_path=data_path,
        seuils=seuils,
    )


def print_thresholds(seuils: DecisionThresholds) -> None:
    """Affiche les seuils de confiance utilisés."""
    print("\nSeuils de confiance :")
    print(f"  Haute   : ≥ {seuils.haute:.2f}")
    print(f"  Moyenne : ≥ {seuils.moyenne:.2f}")
    print(f"  Faible  : ≥ {seuils.faible:.2f}")
    print(f"  Sécurité 'notumor' : ≥ {seuils.securite_negatif:.2f}")


def load_and_preprocess_test_data(
    config: RuntimeConfigT5,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Charge et preprocess les données de test."""
    print("\n" + "-" * 70)
    print("ETAPE 1 : Chargement des données de test")
    print("-" * 70)

    test_path = config.data_path / "Testing"
    if not test_path.exists():
        raise FileNotFoundError(f"répertoire de test introuvable : {test_path}")

    classes = discover_classes(test_path)
    print(f"Classes découvertes : {classes}")

    test_dataset = load_dataset_split(
        test_path,
        image_size=(config.img_size, config.img_size),
        class_names=classes,
    )
    print(f"Images de test chargées : {test_dataset.images.shape[0]}")

    if config.n_samples_max and test_dataset.images.shape[0] > config.n_samples_max:
        print(f"Limitation à {config.n_samples_max} échantillons")
        test_dataset = DatasetSplit(
            images=test_dataset.images[: config.n_samples_max],
            labels=test_dataset.labels[: config.n_samples_max],
            class_names=test_dataset.class_names,
            image_paths=test_dataset.image_paths[: config.n_samples_max],
        )

    x_test, y_test_labels = preprocess_dataset(
        test_dataset.images,
        test_dataset.labels,
        target_size=(config.img_size, config.img_size),
        normalize=True,
        one_hot=True,
        num_classes=len(classes),
    )
    print(f"Shape après prétraitement : X={x_test.shape}, y={y_test_labels.shape}")
    return x_test, y_test_labels, classes


def train_and_predict(
    args: argparse.Namespace,
    x_test: np.ndarray,
    y_test_labels: np.ndarray,
    classes: list[str],
) -> tuple[np.ndarray, float]:
    """Construit le modèle, entraîne brièvement et prédit."""
    print("\n" + "-" * 70)
    print("ETAPE 2 : Génération des prédictions")
    print("-" * 70)

    print(f"Création du modèle {args.model.upper()}...")
    model = charger_modele(args.model, x_test.shape[1:], len(classes))
    print(f"Architecture : {model.count_params()} paramètres")

    if args.fast:
        print("Entraînement rapide (2 epochs pour simulation)...")
        model.fit(
            x_test,
            y_test_labels,
            epochs=2,
            batch_size=128,
            verbose=0,
            validation_split=0.2,
        )
    else:
        print(" Mode complet : entraînement de 10 epochs...")
        model.fit(
            x_test,
            y_test_labels,
            epochs=10,
            batch_size=64,
            verbose=1,
            validation_split=0.2,
        )

    print("\nGénération des prédictions...")
    probabilites = model.predict(x_test, verbose=0)
    accuracy = float(
        np.mean(np.argmax(probabilites, axis=1) == np.argmax(y_test_labels, axis=1))
    )
    print(f"Accuracy du modèle : {accuracy:.1%}")
    return probabilites, accuracy


def apply_decision_pipeline(
    probabilites: np.ndarray,
    classes: list[str],
    seuils: DecisionThresholds,
) -> tuple[list[Any], dict, dict, dict, dict[str, list[Any]]]:
    """Applique moteur de décision, sécurité et triage."""
    print("\n" + "-" * 70)
    print("ETAPE 3 : Application du moteur de décision")
    print("-" * 70)

    decisions = traiter_batch_decisions(
        probabilites_batch=probabilites,
        classes=classes,
        patient_ids=[f"P_{idx:05d}" for idx in range(len(probabilites))],
        seuils=seuils,
    )
    print(f"Décisions générées : {len(decisions)}")

    stats_base = statistiques_decisions(decisions)
    print("\nDistribution des niveaux de confiance :")
    print(f"  HAUTE       : {stats_base['taux_haute_confiance']:.1%}")
    print(f"  MOYENNE     : {stats_base['taux_moyenne_confiance']:.1%}")
    print(f"  FAIBLE      : {stats_base['taux_faible_confiance']:.1%}")
    print(f"  TRES_FAIBLE : {stats_base['taux_tres_faible_confiance']:.1%}")
    print(f"\nConfiance moyenne : {stats_base['confiance_moyenne']:.1%}")
    print(f"Révisions requises : {stats_base['taux_revision_requise']:.1%}")

    print("\n" + "-" * 70)
    print("ETAPE 4 : Application des règles de sécurité")
    print("-" * 70)
    decisions = appliquer_regles_securite_batch(decisions, seuils.securite_negatif)
    stats_secu = statistiques_securite(decisions)
    print(f"Alertes de sécurité : {stats_secu['n_alertes_securite']}")
    print(f"Taux d'alertes : {stats_secu['taux_alertes']:.1%}")
    print(f"Prédictions 'notumor' : {stats_secu['n_predictions_notumor']}")
    print(f"Alertes parmi 'notumor' : {stats_secu['taux_alertes_parmi_notumor']:.1%}")
    print(
        f"Cas ambigus : {stats_secu['n_cas_ambigus']} ({stats_secu['taux_ambigus']:.1%})"
    )
    print(f"\nCas limites identifiés : {len(identifier_cas_limites(decisions))}")

    print("\n" + "-" * 70)
    print("ETAPE 5 : Application du système de triage")
    print("-" * 70)
    decisions = appliquer_triage_batch(decisions)
    stats_triage = statistiques_triage(decisions)
    print("Distribution des priorités :")
    print(
        f"  URGENTE : {stats_triage['n_urgente']} ({stats_triage['taux_urgente']:.1%})"
    )
    print(f"  ELEVEE  : {stats_triage['n_elevee']} ({stats_triage['taux_elevee']:.1%})")
    print(
        f"  NORMALE : {stats_triage['n_normale']} ({stats_triage['taux_normale']:.1%})"
    )
    print(
        f"  ROUTINE : {stats_triage['n_routine']} ({stats_triage['taux_routine']:.1%})"
    )
    print(
        f"\nTaux de cas critiques (URGENTE + ELEVEE) : {stats_triage['taux_critique_total']:.1%}"
    )

    files_attente = generer_file_attente(trier_par_priorite(decisions))
    print("\nFiles d'attente générées :")
    for priorite, file_items in files_attente.items():
        print(f"  {priorite:20s} : {len(file_items)} cas")

    return decisions, stats_base, stats_secu, stats_triage, files_attente


def print_examples(decisions: list[Any], files_attente: dict[str, list[Any]]) -> None:
    """Affiche des exemples de décisions cliniques."""
    print("\n" + "-" * 70)
    print("ETAPE 6 : Exemples de décisions cliniques")
    print("-" * 70)

    if files_attente.get(HyperParametersStr.URGENT_QUEUE_KEY):
        print("\n>>> EXEMPLE 1 : Cas URGENT")
        afficher_exemple_decision(
            files_attente[HyperParametersStr.URGENT_QUEUE_KEY][0], idx=1
        )

    cas_alertes = [decision for decision in decisions if decision.alerte_securite]
    if cas_alertes:
        print("\n>>> EXEMPLE 2 : Cas avec ALERTE SECURITE")
        afficher_exemple_decision(cas_alertes[0], idx=2)

    cas_haute_confiance = [
        decision
        for decision in decisions
        if decision.niveau_confiance == ConfidenceLevel.CONFIDENCE_HAUTE
    ]
    if cas_haute_confiance:
        print("\n>>> EXEMPLE 3 : Cas HAUTE CONFIANCE")
        afficher_exemple_decision(cas_haute_confiance[0], idx=3)

    cas_incertains = [
        decision
        for decision in decisions
        if decision.niveau_confiance == ConfidenceLevel.CONFIDENCE_TRES_FAIBLE
    ]
    if cas_incertains:
        print("\n>>> EXEMPLE 4 : Cas TRES INCERTAIN")
        afficher_exemple_decision(cas_incertains[0], idx=4)


def print_final_summary(
    decisions: list[Any],
    accuracy: float,
    stats_base: dict,
    stats_secu: dict,
    stats_triage: dict,
) -> None:
    """Affiche le résumé global de la tâche 5."""
    print("\n" + "=" * 70)
    print("RESUME GLOBAL")
    print("=" * 70)
    print(f"Total de cas traités : {len(decisions)}")
    print(f"Accuracy du modèle : {accuracy:.1%}")
    print("\nDistribution finale :")
    print(f"  Haute confiance : {stats_base['taux_haute_confiance']:.1%}")
    print(f"  Révisions requises : {stats_base['taux_revision_requise']:.1%}")
    print(f"  Alertes de sécurité : {stats_secu['taux_alertes']:.1%}")
    print(f"  Cas critiques (URG+ELEV) : {stats_triage['taux_critique_total']:.1%}")

    print("\n✅ Tâche 5 terminée avec succès!")
    print("=" * 70 + "\n")


def main() -> None:
    """Point d'entrée principal."""
    args = parse_args()
    print_execution_header(args)

    config = build_runtime_config(args)
    print_thresholds(config.seuils)

    try:
        x_test, y_test_labels, classes = load_and_preprocess_test_data(config)
    except FileNotFoundError as err:
        print(f"❌ Erreur : {err}")
        raise SystemExit(1) from err

    probabilites, accuracy = train_and_predict(args, x_test, y_test_labels, classes)
    decisions, stats_base, stats_secu, stats_triage, files_attente = (
        apply_decision_pipeline(
            probabilites=probabilites,
            classes=classes,
            seuils=config.seuils,
        )
    )
    print_examples(decisions, files_attente)
    print_final_summary(decisions, accuracy, stats_base, stats_secu, stats_triage)


if __name__ == "__main__":
    main()
