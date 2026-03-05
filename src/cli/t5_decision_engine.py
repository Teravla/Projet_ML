#!/usr/bin/env python3
"""Script runner pour la Tâche 5 : Moteur de Décision Clinique.

Ce script démontre le fonctionnement du SAD (Système d'Aide à la Décision)
en appliquant les règles métier, la gestion de sécurité (faux négatifs),
et le système de triage sur des prédictions de modèle.

Usage:
    python scripts/run_task5.py [--fast] [--model MODEL_TYPE] [--n-samples N]

    poetry run task t5 --fast
"""

import argparse
import sys
from pathlib import Path
import keras
import numpy as np
from datetime import datetime

# Ajouter le répertoire racine au path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

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
from src.config.thresholds import DecisionThresholds


def parse_args():
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


def charger_modele(model_type: str, input_shape, num_classes: int):
    """Charge ou crée un modèle pour les prédictions.

    Note: Cette version simplifiée crée un nouveau modèle.
    Dans un contexte réel, on chargerait un modèle pré-entraîné depuis artifacts/.
    """
    import tensorflow as tf

    if model_type == "logreg":
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
    elif model_type == "mlp":
        # MLP prend input_dim (flattened), pas input_shape
        input_dim = np.prod(input_shape)
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
        from src.models.cnn import build_cnn_classifier

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
        print(f"\n⚠️  ALERTE SECURITE : Risque de faux négatif détecté!")

    print(f"\nScores par classe:")
    for classe, prob in sorted(
        decision.probabilites.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  • {classe:20s} : {prob:6.1%}")
    print("=" * 70)


def main():
    """Point d'entrée principal."""
    args = parse_args()

    print("\n" + "=" * 70)
    print("TACHE 5 : MOTEUR DE DECISION CLINIQUE (SAD)")
    print("=" * 70)
    print(f"Date d'exécution : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode rapide      : {args.fast}")
    print(f"Type de modèle   : {args.model.upper()}")

    # Configuration
    if args.fast:
        img_size = min(args.img_size, 32)
        n_samples_max = args.n_samples or 200
        print(f"⚡ Mode fast : img_size={img_size}, max_samples={n_samples_max}")
    else:
        img_size = args.img_size
        n_samples_max = args.n_samples

    # Seuils personnalisés
    seuils = DecisionThresholds(
        haute=args.seuil_haute, moyenne=args.seuil_moyenne, faible=args.seuil_faible
    )

    print(f"\nSeuils de confiance :")
    print(f"  Haute   : ≥ {seuils.haute:.2f}")
    print(f"  Moyenne : ≥ {seuils.moyenne:.2f}")
    print(f"  Faible  : ≥ {seuils.faible:.2f}")
    print(f"  Sécurité 'notumor' : ≥ {seuils.securite_negatif:.2f}")

    # 1. Charger les données de test
    print("\n" + "-" * 70)
    print("ETAPE 1 : Chargement des données de test")
    print("-" * 70)

    data_path = Path(args.data_path) if args.data_path else PROJECT_ROOT / "data"
    test_path = data_path / "Testing"

    if not test_path.exists():
        print(f"❌ Erreur : répertoire de test introuvable : {test_path}")
        sys.exit(1)

    classes = discover_classes(test_path)
    print(f"Classes découvertes : {classes}")

    test_dataset = load_dataset_split(
        test_path, image_size=(img_size, img_size), class_names=classes
    )
    n_samples = test_dataset.images.shape[0]
    print(f"Images de test chargées : {n_samples}")

    # Limiter nombre d'échantillons si nécessaire
    if n_samples_max and n_samples > n_samples_max:
        print(f"Limitation à {n_samples_max} échantillons")
        images_limited = test_dataset.images[:n_samples_max]
        labels_limited = test_dataset.labels[:n_samples_max]
        test_dataset = DatasetSplit(
            images=images_limited,
            labels=labels_limited,
            class_names=test_dataset.class_names,
            image_paths=test_dataset.image_paths[:n_samples_max],
        )
        n_samples = n_samples_max

    # Prétraitement
    X_test, y_test_labels = preprocess_dataset(
        test_dataset.images,
        test_dataset.labels,
        target_size=(img_size, img_size),
        normalize=True,
        one_hot=True,
        num_classes=len(classes),
    )

    print(f"Shape après prétraitement : X={X_test.shape}, y={y_test_labels.shape}")

    # 2. Charger/créer le modèle et générer des prédictions
    print("\n" + "-" * 70)
    print("ETAPE 2 : Génération des prédictions")
    print("-" * 70)

    input_shape = X_test.shape[1:]  # (height, width, channels)
    num_classes = len(classes)

    print(f"Création du modèle {args.model.upper()}...")
    model = charger_modele(args.model, input_shape, num_classes)

    print(f"Architecture : {model.count_params()} paramètres")

    # Entraînement rapide (simulé en mode fast)
    # Dans un contexte réel, on chargerait un modèle pré-entraîné
    if args.fast:
        print("Entraînement rapide (2 epochs pour simulation)...")
        model.fit(
            X_test,
            y_test_labels,
            epochs=2,
            batch_size=128,
            verbose=0,
            validation_split=0.2,
        )
    else:
        print("⚠️ Mode complet : entraînement de 10 epochs...")
        model.fit(
            X_test,
            y_test_labels,
            epochs=10,
            batch_size=64,
            verbose=1,
            validation_split=0.2,
        )

    # Prédictions
    print("\nGénération des prédictions...")
    probabilites = model.predict(X_test, verbose=0)
    predictions_idx = np.argmax(probabilites, axis=1)
    y_true_idx = np.argmax(y_test_labels, axis=1)

    accuracy = np.mean(predictions_idx == y_true_idx)
    print(f"Accuracy du modèle : {accuracy:.1%}")

    # 3. Appliquer le moteur de décision
    print("\n" + "-" * 70)
    print("ETAPE 3 : Application du moteur de décision")
    print("-" * 70)

    patient_ids = [f"P_{i:05d}" for i in range(len(probabilites))]

    decisions = traiter_batch_decisions(
        probabilites_batch=probabilites,
        classes=classes,
        patient_ids=patient_ids,
        seuils=seuils,
    )

    print(f"Décisions générées : {len(decisions)}")

    # Statistiques initiales
    stats_base = statistiques_decisions(decisions)
    print(f"\nDistribution des niveaux de confiance :")
    print(f"  HAUTE       : {stats_base['taux_haute_confiance']:.1%}")
    print(f"  MOYENNE     : {stats_base['taux_moyenne_confiance']:.1%}")
    print(f"  FAIBLE      : {stats_base['taux_faible_confiance']:.1%}")
    print(f"  TRES_FAIBLE : {stats_base['taux_tres_faible_confiance']:.1%}")
    print(f"\nConfiance moyenne : {stats_base['confiance_moyenne']:.1%}")
    print(f"Révisions requises : {stats_base['taux_revision_requise']:.1%}")

    # 4. Appliquer les règles de sécurité
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

    # Identifier les cas limites
    cas_limites = identifier_cas_limites(decisions)
    print(f"\nCas limites identifiés : {len(cas_limites)}")

    # 5. Appliquer le triage
    print("\n" + "-" * 70)
    print("ETAPE 5 : Application du système de triage")
    print("-" * 70)

    decisions = appliquer_triage_batch(decisions)

    stats_triage = statistiques_triage(decisions)
    print(f"Distribution des priorités :")
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

    # 6. Trier par priorité
    decisions_triees = trier_par_priorite(decisions)

    # 7. Générer les files d'attente
    files_attente = generer_file_attente(decisions_triees)
    print(f"\nFiles d'attente générées :")
    for priorite, file in files_attente.items():
        print(f"  {priorite:20s} : {len(file)} cas")

    # 8. Afficher des exemples
    print("\n" + "-" * 70)
    print("ETAPE 6 : Exemples de décisions cliniques")
    print("-" * 70)

    # Exemple 1: Cas urgent (si disponible)
    if files_attente["URGENTE (12h)"]:
        print("\n>>> EXEMPLE 1 : Cas URGENT")
        afficher_exemple_decision(files_attente["URGENTE (12h)"][0], idx=1)

    # Exemple 2: Cas avec alerte sécurité (si disponible)
    cas_alertes = [d for d in decisions if d.alerte_securite]
    if cas_alertes:
        print("\n>>> EXEMPLE 2 : Cas avec ALERTE SECURITE")
        afficher_exemple_decision(cas_alertes[0], idx=2)

    # Exemple 3: Cas haute confiance
    cas_haute_confiance = [d for d in decisions if d.niveau_confiance == "HAUTE"]
    if cas_haute_confiance:
        print("\n>>> EXEMPLE 3 : Cas HAUTE CONFIANCE")
        afficher_exemple_decision(cas_haute_confiance[0], idx=3)

    # Exemple 4: Cas incertain
    cas_incertains = [d for d in decisions if d.niveau_confiance == "TRES_FAIBLE"]
    if cas_incertains:
        print("\n>>> EXEMPLE 4 : Cas TRES INCERTAIN")
        afficher_exemple_decision(cas_incertains[0], idx=4)

    # 9. Résumé final
    print("\n" + "=" * 70)
    print("RESUME GLOBAL")
    print("=" * 70)
    print(f"Total de cas traités : {len(decisions)}")
    print(f"Accuracy du modèle : {accuracy:.1%}")
    print(f"\nDistribution finale :")
    print(f"  Haute confiance : {stats_base['taux_haute_confiance']:.1%}")
    print(f"  Révisions requises : {stats_base['taux_revision_requise']:.1%}")
    print(f"  Alertes de sécurité : {stats_secu['taux_alertes']:.1%}")
    print(f"  Cas critiques (URG+ELEV) : {stats_triage['taux_critique_total']:.1%}")

    print("\n✅ Tâche 5 terminée avec succès!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
