# 🏦 BNP Paribas — Détection de churn bancaire

> **Prédiction du désabonnement client** via CatBoost + API FastAPI

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![CatBoost](https://img.shields.io/badge/Model-CatBoost-orange)](https://catboost.ai/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![F1-Score](https://img.shields.io/badge/F1--Score-0.6425-success)](.)
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.8688-blue)](.)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🎯 Objectif

Identifier **à l'avance** les clients à risque de désabonnement (churn) pour
leur proposer une **offre personnalisée** et réduire l'attrition bancaire.

**Contexte** : Projet Data Science M2 — Banque BNP Paribas  
**Dataset** : [Kaggle Churn Modelling](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction) — 10 000 clients  
**Cible** : `Exited` (0 = reste, 1 = part) — Taux de churn : **20.37 %**

---

## 📊 Résultats

### Performances du modèle final

| Métrique | Valeur | Benchmark Kaggle |
|---|---|---|
| **F1-Score** | **0.6425** | 0.55 - 0.60 ✅ |
| **ROC-AUC** | **0.8688** | 0.80 - 0.85 ✅ |
| **Recall** | ~0.75 | — |
| **Precision** | ~0.55 | — |

🏆 **Modèle retenu** : **CatBoost** avec **feature engineering** + **seuil optimal (0.58)**

### Comparaison des modèles

| Modèle | F1 | ROC-AUC |
|---|---|---|
| 🥇 **CatBoost + FE** | **0.6425** | **0.8688** |
| 🥈 Stacking (XGB+LGB+Cat) | 0.6304 | 0.8615 |
| 🥉 Voting | 0.6220 | — |
| XGBoost + FE | 0.6032 | 0.8439 |
| LightGBM + FE | 0.5922 | 0.8486 |

**Gain vs baseline** : **+29 % F1**, **+13 % ROC-AUC**

---

## 🏗️ Structure du projet

```
bnp-churn-prediction/
│
├── 📓 EDA.ipynb                    # Analyse exploratoire
├── 📓 Extended_Churn.ipynb         # Features + 5 modèles + stacking
├── 📄 RAPPORT.md                   # Rapport de synthèse complet
├── 📄 README.md                    # Ce fichier
├── 📄 requirements.txt             # Dépendances
├── 📄 .gitignore
│
├── 📁 api/
│   └── main.py                     # API FastAPI
│
├── 📁 models/  (à la racine)
│   ├── churn_model_catboost.pkl    # Modèle final
│   ├── churn_preprocessor.pkl      # Preprocessor
│   ├── churn_threshold.pkl         # Seuil optimal (0.58)
│   └── churn_metadata.pkl          # Métadonnées
│
├── 📄 clients_a_risque_v2.xlsx     # Liste clients à contacter
├── 📄 churn_analysis_final.png     # Graphiques de synthèse
├── 📄 Churn_Modelling.csv          # Dataset source
└── 📄 3.1 Pipeline global.txt      # Notes
```

---

## 🚀 Installation & lancement

### 1. Cloner le repo

```bash
git clone https://github.com/kabangesylvain-ui/bnp-churn-prediction.git
cd bnp-churn-prediction
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Lancer l'API FastAPI

```bash
python api/main.py
```

**Sortie attendue** :

```
OK Modele : CatBoost + FE
OK Seuil  : 0.58
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 4. Tester dans le navigateur

- **Health check** : http://localhost:8000/
- **Swagger UI** : http://localhost:8000/docs

---

## 🔌 API Endpoints

### `GET /`

Health check — vérifie que l'API est opérationnelle.

```json
{
  "status": "ok",
  "model": "CatBoost + FE",
  "f1_score": 0.6425,
  "threshold": 0.58
}
```

### `POST /predict`

Prédit la probabilité de churn d'un client.

**Requête** :

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "CreditScore": 600,
    "Geography": "Germany",
    "Gender": "Female",
    "Age": 55,
    "Tenure": 3,
    "Balance": 120000,
    "NumOfProducts": 3,
    "HasCrCard": 1,
    "IsActiveMember": 0,
    "EstimatedSalary": 80000
  }'
```

**Réponse** :

```json
{
  "churn_probability": 0.9754,
  "prediction": 1,
  "risk_level": "CRITIQUE",
  "action_recommandee": "Contacter immediatement (offre premium)",
  "threshold_used": 0.58
}
```

---

## 🧠 Facteurs de churn identifiés

| Rang | Feature | Impact |
|---|---|---|
| 🥇 | **Age** | Les clients âgés partent plus |
| 🥈 | **NumOfProducts** | 3-4 produits = churn massif |
| 🥉 | **IsActiveMember** | Inactifs = 2× plus de churn |
| 4 | **Balance** | Solde élevé → churn (contre-intuitif) |
| 5 | **Geography_Germany** | Allemagne sur-représentée |

### 💡 Règle métier validée

> **Contacter en priorité :**
> - Clients avec **3+ produits**, OU
> - Clients **inactifs** + **âge > 45 ans** + **solde élevé**

---

## 📈 Exemples de prédictions

| Profil client | Proba churn | Niveau |
|---|---|---|
| 55 ans, Allemagne, 3 produits, inactif, 120k€ | **97.5 %** | 🔴 CRITIQUE |
| 45 ans, Espagne, 1 produit, inactif, 0€ | **88.6 %** | 🔴 ÉLEVÉ |
| 35 ans, France, 2 produits, actif, 50k€ | **21.0 %** | 🟢 FAIBLE |

---

## 🛠️ Stack technique

| Catégorie | Technologies |
|---|---|
| **Data** | pandas, numpy, matplotlib, seaborn |
| **ML** | scikit-learn, XGBoost, LightGBM, CatBoost |
| **Interprétabilité** | feature importance, arbre de décision |
| **API** | FastAPI, Pydantic, Uvicorn |
| **Sérialisation** | joblib |
| **Versioning** | Git, GitHub |

---

## 📊 Livrables

| Livrable | Description |
|---|---|
| **Notebooks** | EDA complet + 5 modèles comparés |
| **Modèle final** | CatBoost sérialisé (F1 = 0.6425) |
| **API REST** | Service FastAPI temps réel |
| **Liste clients** | 1 270 clients à contacter (Excel) |
| **Rapport** | Document de synthèse complet |

---

## 🎯 Impact business

| Indicateur | Valeur |
|---|---|
| Clients actifs | 7 963 |
| **Clients à contacter** | **1 270** (16 %) |
| **Lift vs campagne aléatoire** | **×3** |
| **ROI estimé** | 25 - 50 k€ / campagne |

---

## 📚 Documentation

- 📄 [`RAPPORT.md`](RAPPORT.md) — Rapport de synthèse complet
- 📓 [`EDA.ipynb`](EDA.ipynb) — Analyse exploratoire
- 📓 [`Extended_Churn.ipynb`](Extended_Churn.ipynb) — Modélisation + stacking
- 🔌 [`api/main.py`](api/main.py) — Code de l'API

---

## 👤 Auteur

**KABANGE SYLVAIN**

🎓 M2 Data Science — Horizon University  
📧 [kabange.sylvain@horizon-university.tn](mailto:kabange.sylvain@horizon-university.tn)  
🔗 [GitHub @kabangesylvain-ui](https://github.com/kabangesylvain-ui)

---

## 📄 License

Ce projet est sous licence MIT. Voir [LICENSE](LICENSE) pour plus de détails.

---

⭐ **Si ce projet vous a plu, n'hésitez pas à lui donner une étoile !**

---

**Dernière mise à jour** : Octobre 2026