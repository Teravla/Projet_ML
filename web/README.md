# SAD Dashboard Web avec API Flask

Dashboard interactif connecté à une API Flask pour le système d'aide à la décision (SAD) - Diagnostic de tumeurs cérébrales.

## 🚀 Démarrage Rapide

### 1. Lancer le serveur Flask

#### Windows (PowerShell)

```powershell
cd c:\Users\valen\Documents\EFREI\I2\Machine_Learning\Projet
python scripts/run_flask_dashboard.py
```

#### macOS/Linux (Bash)

```bash
cd ~/path/to/projet
python scripts/run_flask_dashboard.py
```

Le serveur démarrera sur `http://localhost:5000`

### 2. Accéder au Dashboard

Ouvrir dans le navigateur:

```
http://localhost:5000/dashboard
```

ou directement:

```
http://localhost:5000
```

## 📋 Fichiers

- `dashboard.html` - Dashboard web qui charge les données depuis l'API Flask
- `README.md` - Cette documentation
- `../scripts/run_flask_dashboard.py` - Serveur Flask avec endpoints API

## 🔌 Endpoints API

Le serveur Flask expose les endpoints suivants:

### 1. Health Check

```
GET /api/health
Response: { "status": "ok", "timestamp": "2026-03-05T..." }
```

### 2. Statistiques Globales

```
GET /api/stats
Response: {
  "n_total": 120,
  "couverture_automatique": 87.5,
  "alertes": 3,
  "revisions": 15,
  "confiance_distribution": {
    "HAUTE": { "count": 62, "percent": 51.7 },
    "MOYENNE": { "count": 34, "percent": 28.3 },
    "FAIBLE": { "count": 17, "percent": 14.2 },
    "TRES_FAIBLE": { "count": 7, "percent": 5.8 }
  },
  "priorite_distribution": { ... }
}
```

### 3. Décisions Patients

```
GET /api/decisions
Response: {
  "decisions": [
    {
      "patient_id": "P_00001",
      "classe_predite": "glioma",
      "confiance": 0.913,
      "niveau_confiance": "HAUTE",
      "priorite": "URGENTE (12h)",
      "alerte_securite": false,
      "decision": "Diagnostic automatique valide",
      "probabilites": { "glioma": 0.913, ... }
    },
    ...
  ],
  "total": 120
}
```

### 4. Métriques Performance (Phase 7)

```
GET /api/metrics
Response: {
  "accuracy_globale": 0.7917,
  "taux_couverture_automatique": 0.8667,
  "objectif_haute_confiance": {
    "n_cas": 62,
    "accuracy": 0.973,
    "objectif_atteint": true
  },
  "couts": {
    "FN": 0,
    "FP": 5,
    "Revision": 23,
    "Cost_total": 2150
  }
}
```

### 5. Rapport Patient

```
GET /api/rapport/<patient_id>
Response: {
  "patient_id": "P_00001",
  "rapport": "========================================\nRAPPORT D'AIDE A LA DECISION\n..."
}
```

## 📊 Onglets du Dashboard

1. **Statistiques** - Vue globale avec distributions
2. **Décisions Patients** - Table des 10 décisions récentes
3. **Métriques Performance** - Analyse phase 7 (coûts, accuracy)
4. **Pipeline SAD** - Exécution du système complet

## 🎨 Fonctionnalités

✅ Chargement automatique des données au démarrage
✅ Bouton "Rafraîchir" pour actualiser les données
✅ Bouton "Exporter JSON" pour télécharger les données
✅ Indicateur de statut API (🟢 Connecté / 🔴 Déconnecté)
✅ Design responsive (Desktop/Mobile)
✅ Pas de dépendances externes côté client (JavaScript vanilla)

## 🛠️ Dépendances Python

- `flask` ≥ 2.0
- `numpy` (déjà présent pour les modules SAD)
- Modules `src/` du projet (decision, reporting, evaluation)

La dépendance Flask est déjà installée dans votre environnement.

## 📝 Données Simulées

Le serveur génère automatiquement:

- **120 patients** avec distributions probabilistes réalistes
- **Seed aléatoire contrôlé** pour reproductibilité
- **Probabilités bruitées** pour 3 cas avec notumor haute confiance (test de la règle de sécurité)

## 🔒 Règles de Sécurité du SAD

1. **Notumor faible confiance**: Si class="notumor" ET confiance < 0.95 → Alerte + révision
2. **Objectif haute confiance**: Accuracy > 95% pour confiance > 0.85 (HAUTE)
3. **Coûts asymétriques**: FN=1000€ >> FP=100€ >> Révision=50€

## 🐛 Dépannage

### "🔴 API Indisponible"

```
✗ Le serveur Flask n'est pas lancé
✓ Relancer avec: python scripts/run_flask_dashboard.py
```

### CORS Error (en développement)

```
✓ Le dashboard assume que l'API est sur localhost:5000
✓ Si besoin de CORS: installer flask-cors et l'ajouter au serveur
```

### Données ne se mettent pas à jour

```
✓ Cliquer le bouton "Rafraîchir"
✓ Vérifier que le serveur Flask affiche "🟢 API Connectée"
```

## 📱 Intégration Production

Pour déployer en production:

1. **Remplacer les données simulées** par des vraies predictions
2. **Ajouter une base de données** (SQLite, PostgreSQL) pour l'historique
3. **Implémenter l'authentification** pour sécuriser les accès
4. **Utiliser un serveur WSGI** (Gunicorn, uWSGI) au lieu de Flask debug
5. **Activer HTTPS** et les en-têtes de sécurité

## 📖 Technologies

- **Backend**: Flask 3.1.2 (Python)
- **Frontend**: HTML5 + CSS3 + JavaScript Vanilla
- **API**: RESTful JSON
- **Aucun framework frontend** - Code autonome et léger

## 📄 Notes

- Les données sont **simulées à chaque requête** (pas de cache)
- Le serveur affiche les endpoints disponibles au démarrage
- Appuyer `CTRL+C` pour arrêter le serveur
- Le dashboard respecte le cahier des charges TODO.md
- Phases 5-7 du projet entièrement intégrées
