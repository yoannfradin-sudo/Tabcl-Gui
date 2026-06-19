# TabICL GUI

Application desktop (PyQt6) pour exploiter facilement [TabICL](https://github.com/soda-inria/tabicl), un modèle de fondation tabulaire état-de-l'art basé sur l'apprentissage en contexte.

## Fonctionnalités

| Onglet | Description |
|---|---|
| **Données** | Chargement CSV / Excel / JSON / Parquet (local ou URL), aperçu, sélection des colonnes |
| **Entraînement** | Classification ou régression, split train/test, paramètres TabICL |
| **Résultats** | Métriques, matrice de confusion / scatter plot, export CSV des prédictions, SHAP |
| **Prévision** | Séries temporelles univariées via `TabICLForecaster` |
| **Fine-tuning** | Adaptation des poids via `FinetunedTabICLClassifier/Regressor` |

## Installation

```bash
pip install -r requirements.txt
```

> `tabicl[all]` installe les dépendances pour forecast, SHAP et fine-tuning.

## Utilisation

```bash
python app.py
```

## Prérequis

- Python 3.10+
- PyQt6
- GPU recommandé pour le fine-tuning
