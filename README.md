# TabICL GUI

Application desktop (PyQt6) pour exploiter facilement [TabICL](https://github.com/soda-inria/tabicl), un modèle de fondation tabulaire état-de-l'art basé sur l'apprentissage en contexte.

## Fonctionnalités

| Onglet | Description |
|---|---|
| **Données** | Chargement CSV / Excel / JSON / Parquet (local ou URL), aperçu, sélection des colonnes |
| **Connaissances (LLM)** | Un LLM local (LM Studio) infère des contraintes métier par colonne (bornes physiques, unités) pour affiner les prédictions |
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

## Couche LLM : injection de connaissances métier

L'onglet **Connaissances (LLM)** utilise votre serveur **LM Studio** local
(API compatible OpenAI) comme expert métier afin d'affiner les prédictions
statistiques pures de TabICL.

**Principe :**
1. Le LLM reçoit la description des colonnes (nom, type, statistiques, exemples).
2. Il propose des contraintes métier réalistes : bornes physiques (ex. `Âge ∈ [0, 120]`),
   unités, valeurs impossibles.
3. Vous **vérifiez et corrigez** ces propositions dans une table éditable.
4. Les contraintes sont injectées dans le pipeline :
   - **Nettoyage des données d'entraînement** : bornage (clip) ou suppression (drop)
     des valeurs aberrantes avant `fit`.
   - **Bornage des prédictions** de régression à l'intervalle plausible de la cible.

**Configuration LM Studio :**
- Lancez LM Studio et démarrez le serveur local (onglet *Developer* → *Start Server*).
- Base URL par défaut : `http://localhost:1234/v1`, clé API : `lm-studio` (factice).
- Cliquez **« Lister les modèles »** pour récupérer le modèle chargé.

> La couche LLM est entièrement optionnelle : sans contraintes validées,
> l'entraînement utilise les données brutes.

## Installation sur une machine hors ligne

L'application peut être empaquetée avec **toutes ses dépendances** pour une
machine sans accès Internet.

### 1. Sur une machine connectée — fabriquer le bundle

```bash
./scripts/bundle_offline.sh
```

Cela produit `tabicl-gui-offline.tar.gz` contenant le code, toutes les wheels
Python, et (si possible) le checkpoint TabICL pré-téléchargé.

> **Plateforme cible différente ?** Si la machine hors ligne tourne sous un
> autre OS/architecture, précisez-le :
> ```bash
> TARGET_PLATFORM=manylinux2014_x86_64 TARGET_PYTHON=311 ./scripts/bundle_offline.sh
> # Windows : TARGET_PLATFORM=win_amd64
> # macOS ARM : TARGET_PLATFORM=macosx_11_0_arm64
> ```

### 2. Sur la machine hors ligne — installer

```bash
tar -xzf tabicl-gui-offline.tar.gz
cd tabicl-gui-offline
./install_offline.sh
```

Puis lancer :

```bash
source .venv/bin/activate
python app.py
```

### Note sur le checkpoint du modèle

TabICL télécharge normalement son checkpoint au premier usage. Hors ligne,
ce n'est pas possible. Le script de bundle tente de l'embarquer dans
`checkpoints/`. Au lancement de l'app, indiquez ce fichier `.ckpt` via le
champ **« Checkpoint local »** de l'onglet *Entraînement* (cela désactive
le téléchargement automatique).

## Prérequis

- Python 3.10+
- PyQt6
- GPU recommandé pour le fine-tuning
