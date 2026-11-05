# Challenge_copilot_users_tracking

Identification des utilisateurs du logiciel **Copilote d’Infologic** à partir de leurs traces d’activité. Projet de classification supervisée multiclasse réalisé dans le cadre du module MOD 7.2 — Introduction à la science des données.

## État du projet

Le notebook `solution.ipynb` suit la progression du notebook d’indications du professeur :

1. Compréhension du problème et chargement des sessions.
2. Exploration des données, graphiques et contrôle de qualité.
3. Construction des caractéristiques par session et préparation de la cible.
4. Statistiques descriptives, comparaisons entre utilisateurs et corrélations.

Les sections consacrées aux modèles, à leur évaluation et à la soumission sont préparées et restent à compléter. Le préprocesseur est défini ; il sera ajusté uniquement sur l’entraînement après la séparation entraînement/validation.

## Contenu

- `solution.ipynb` : notebook principal, avec explications et résultats enregistrés.
- `BE_CS1_indications.ipynb` : trame pédagogique fournie par le professeur.
- `BE_CS1_instructions.ipynb` : énoncé et consignes du projet.
- `requirements.txt` : dépendances Python et versions utilisées.
- `figures/` : graphiques exportés par le notebook.

## Données

Télécharger les données depuis la [compétition Kaggle](https://www.kaggle.com/competitions/qui-utilise-mon-appli-challenge/), puis les placer dans un dossier `Data` à la racine :

```text
Data/
├── train.csv
├── test.csv
└── sample_submission.csv
```

Les CSV et l’environnement virtuel sont exclus du versionnement. Les données comprennent 3 279 sessions d’entraînement, 247 utilisateurs et 324 sessions de test. Une ligne correspond à une session ; les sessions ont des longueurs variables.

## Installation sous Windows (PowerShell)

L’environnement utilisé pour le développement est **Python 3.13**. L’énoncé mentionne Python 3.14 ; la compatibilité avec cette version reste à vérifier avant le rendu.

Depuis le dossier du projet :

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name tp_ml_s9 --display-name "Python (TP_ML_S9)"
```

## Exécution du notebook

1. Ouvrir le dossier du projet dans VS Code avec les extensions Python et Jupyter.
2. Ouvrir `solution.ipynb` et sélectionner l’interpréteur `.venv` ou le noyau `Python (TP_ML_S9)`.
3. Exécuter les cellules dans l’ordre, depuis le début. Les chemins sont relatifs à la racine du projet.

Le chargement construit les DataFrames complets comme dans les indications. L’aperçu des 20 premières sessions affiche toutes les colonnes : sa sortie volumineuse n’est pas enregistrée dans cette version, mais peut être régénérée en exécutant la cellule. Les graphiques sont enregistrés dans `figures/`.

## Représentation des sessions

La construction produit 48 colonnes : volumes d’activité, proportions par famille d’actions, diversité, enchaînements, contextes dominants et indicateurs temporels, dont des transformations logarithmiques. La cible utilisateur est séparée des caractéristiques.

Le plus grand marqueur `tXX` représente un horizon temporel enregistré, pas nécessairement la durée réelle de connexion. Les répétitions de marqueurs sont prises en compte explicitement. Les statistiques et graphiques ne constituent pas encore une évaluation prédictive.

## Parties 1 et 2 : exploration complète, modélisation et soumission

Les dossiers `travail_partie_1/` et `travail_partie_2/` contiennent une chaîne complète, de la lecture des données au fichier de soumission. Ce travail est indépendant de `solution.ipynb`.

**Meilleur score Kaggle : 0,991** (F1 moyen), avec sept SVM linéaires sur une représentation TF-IDF des sessions et un ensemble de forêts (Random Forest + Extra Trees), combinés par une moyenne géométrique (moyenne des log-probabilités), avec la règle du navigateur et une légère correction des fréquences des utilisateurs.

### Contenu

| Fichier | Rôle |
|---|---|
| `travail_partie_1/partie_1_decouverte_donnees.ipynb` | Chargement, exploration (utilisateurs, temps, navigateur, types d'actions, écrans, configurations, chaînes), valeurs manquantes et aberrantes, construction d'un tableau de 616 variables par session |
| `travail_partie_2/partie_2_modelisation.ipynb` | Protocole d'évaluation, comparaison de 15 modèles, réglage des hyperparamètres, représentation « texte » (TF-IDF + SVM), combinaison, analyse des erreurs, interprétation, soumission |
| `travail_partie_2/representation_texte.py` | Représentation des sessions en documents (actions brutes, paires d'actions, composants), SVM TF-IDF, augmentation par extraits, combinaisons arithmétique et géométrique |
| `travail_partie_2/recherche_hyperparametres.py` | Recherche d'hyperparamètres des forêts et de XGBoost (≈ 1 h) |
| `travail_partie_2/recherche_tfidf_combinaison.py` | Variantes TF-IDF et réglage de la combinaison sur 3 découpages (≈ 25 min) |
| `travail_partie_2/recherche_ameliorations.py` | Améliorations v2 et v3 (temps, augmentation, moyenne de SVM, règle du navigateur) sur 3 découpages (≈ 45 min) |
| `travail_partie_2/resultats/` | Résultats des deux recherches, relus par le notebook de la partie 2 |
| `notes_rapport.md` | Notes détaillées pour le rapport : constats, chiffres, choix et justifications |
| `soumissions/` | Fichiers soumis sur Kaggle (voir le tableau ci-dessous) |

### Données

Les notebooks cherchent `train`, `test` et `sample_submission` au format `.csv` ou `.csv.GZ` (tel que téléchargé depuis Kaggle), dans `Data/`, `data/` ou à la racine du projet. Les deux organisations fonctionnent.

### Exécution

1. `travail_partie_1/partie_1_decouverte_donnees.ipynb` : exécuter toutes les cellules. La section 7 crée `donnees_preparees/` (tableaux de variables), utilisé par la partie 2.
2. `travail_partie_2/partie_2_modelisation.ipynb` : exécuter toutes les cellules (≈ 1 h 10, dont 15 min pour le boosting, 25 min pour les prédictions hors échantillon des 7 SVM du modèle final si `resultats/cache_oof/` est absent, et 7 min pour la soumission). Il relit les résultats de `resultats/` et ne relance pas les recherches ; il écrit `submission.csv` à la racine.
3. Facultatif : relancer les recherches depuis `travail_partie_2/` avec `python recherche_hyperparametres.py`, `python recherche_tfidf_combinaison.py` et `python recherche_ameliorations.py`.

### Résultats

Validation croisée stratifiée à 4 plis (moyenne sur 3 découpages pour les combinaisons).

| Modèle | Accuracy | F1 macro | Kaggle |
|---|---|---|---|
| Random Forest pondérée (616 variables) | 0,946 | 0,942 | 0,920 |
| Extra Trees pondérés | 0,949 | 0,945 | 0,930 |
| Ensemble Random Forest + Extra Trees | 0,951 | 0,947 | 0,938 |
| SVM TF-IDF seul (actions + paires + composants) | 0,957 | 0,947 | — |
| SVM TF-IDF + forêts (T = 0,2, w = 0,6) | 0,968 | 0,965 | 0,969 |
| v2 : SVM avec repères de temps (C = 10) + forêts + règle du navigateur | 0,974 | 0,971 | 0,980 |
| v3 : moyenne de 3 SVM (avec temps ; avec temps + augmentation ; sans temps) + forêts + règle du navigateur | 0,978 | 0,975 | 0,985 |
| v4 : moyenne de 7 SVM (+ 3 SVM augmentés, + SVM sans IDF) + forêts + navigateur + correction des fréquences | 0,981 | 0,979 | 0,985 |
| **v5 : les mêmes modèles, combinaison géométrique (moyenne des log-probabilités, forêts 20 %)** | **0,981** | **0,980** | **0,991** |

Le notebook de la partie 2 reproduit à l'identique la soumission v5 (`soumissions/submission_v5_geometrique.csv`). Les autres fichiers de `soumissions/` correspondent aux lignes précédentes du tableau.
