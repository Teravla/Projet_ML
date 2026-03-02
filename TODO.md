# Cahier des Charges

## Exploration et Modèles de Base

### Tâche 1 : Exploration et Prétraitement

Identique à la version standard (chargement, redimensionnement, augmentation de données).

### Tâche 2 : Régression Logistique avec Calibration

- Entraîner une régression logistique multinomiale.
- Calibration : Utiliser Platt Scaling ou Isotonic Regression pour obtenir des probabilités fiables.
- Analyser la distribution des scores : identifier les prédictions incertaines (max_prob < 0.7).

### Tâche 3 : MLP avec Gestion de l'Incertitude

- Architecture MLP avec sortie probabiliste.
- Objectif : Détecter les limitations du modèle.

## CNN et Système de Décision

### Tâche 4 : CNN Optimisé pour la Décision

Architecture CNN classique, mais avec ajout de Temperature Scaling en sortie pour calibrer les probabilités finales. Sauvegarder les activations pour analyse future.

### Tâche 5 : Moteur de Décision Clinique

Implémenter les règles métiers suivantes:

```python
# Seuils de confiance
SEUIL_HAUTE_CONFIANCE = 0.85
SEUIL_CONFIANCE_MOYENNE = 0.65
SEUIL_CONFIANCE_FAIBLE = 0.50

# Logique de decision
if max_probabilite >= SEUIL_HAUTE_CONFIANCE:
    decision = "Diagnostic automatique valide"
    action = "Rapport envoye au medecin traitant"
    priorite = determiner_urgence(classe_predite)
elif max_probabilite >= SEUIL_CONFIANCE_MOYENNE:
    decision = "Diagnostic probable - Revision recommandee"
    action = "Validation par radiologue junior"
    priorite = "Normale (48h)"
elif max_probabilite >= SEUIL_CONFIANCE_FAIBLE:
    decision = "Cas incertain"
    action = "Revision par radiologue senior"
    priorite = "Elevee (24h)"
else:
    decision = "Incertitude elevee"
    action = "Double lecture obligatoire + IRM complementaire"
    priorite = "Urgente (12h)"
```

Gestion des Faux Négatifs (Sécurité):

```python
# Tolerance asymetrique pour minimiser faux negatifs
if classe_predite == "Pas de tumeur":
    if max_probabilite < 0.95:  # Seuil tres eleve exige
        action = "Verification obligatoire (risque faux negatif)"
```

### Tâche 6 : Tableau de Bord de Décision

Générer un rapport textuel automatisé pour chaque patient:

```log
========================================
RAPPORT D'AIDE A LA DECISION
========================================
Patient ID: P_12345 Date: 01/02/2026

PREDICTION PRINCIPALE
---
Classe: Gliome
Confiance: 91.3%
Niveau de certitude: ELEVE [OK]

SCORES PAR CLASSE
---
• Gliome: 91.3%
• Meningiome: 5.2%
• Tumeur pituitaire: 2.1%
• Pas de tumeur: 1.4%

RECOMMANDATIONS CLINIQUES
---
Diagnostic: Gliome detecte (haute confiance)
Action: Referer immediatement en oncologie
Priorite: [!] URGENT - Prise en charge sous 12h
Revision humaine: Optionnelle (validation finale)

ELEMENTS D'ATTENTION
---
• Tumeur maligne suspectee
• IRM de controle recommandee
========================================
```

### Tâche 7 : Analyse de Performance du SAD

Calculer les métriques orientées "métier":

- Taux de couverture automatique : % de cas gérés sans intervention humaine.
- Accuracy par tranche de confiance : L'accuracy doit être > 95% quand la confiance est > 0.85.
- Analyse Coût-Bénéfice: `Cost_total = (FN × 1000) + (FP × 100) + (Revision × 50)`

## Livrables Attendus

### 6.1 Notebook Jupyter

Le notebook doit contenir:

1. Introduction au SAD (Classification vs Décision).
2. Modèles calibrés (RegLog, MLP, CNN).
3. Moteur de décision : Implémentation des règles.
4. Simulation : 20 exemples de rapports générés.
5. Analyse critique et éthique.

### 6.2 Fonctions Python

Implémenter les fonctions suivantes:

```python
def predire_avec_confiance(image, model):
    """Retourne prediction + scores de confiance"""
    pass

def generer_recommandation(probabilites, seuils):
    """Applique les regles de decision"""
    pass

def calculer_incertitude_mc_dropout(image, model, n_iter=20):
    """Estime l'incertitude via Monte Carlo Dropout"""
    pass

def creer_rapport_decision(patient_id, prediction, confiance):
    """Genere le rapport formate"""
    pass
```
