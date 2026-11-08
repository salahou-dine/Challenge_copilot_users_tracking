# Challenge_copilot_users_tracking

Identification des utilisateurs du logiciel **Copilote d’Infologic** à partir de leurs traces d’activité. Projet de classification supervisée multiclasse réalisé dans le cadre du module MOD 7.2 — Introduction à la science des données.

## État du projet

Le notebook `solution.ipynb` suit la progression du notebook d’indications du professeur :

1. Compréhension du problème et chargement des sessions.
2. Exploration des données, graphiques et contrôle de qualité.
3. Construction des caractéristiques par session, représentation TF-IDF des écrans et des actions, préparation de la cible.
4. Statistiques descriptives, comparaisons entre utilisateurs et corrélations.
5. Comparaison de neuf entraînements (six méthodes sur les 48 caractéristiques, deux sur la TF-IDF, un sur la combinaison des deux), chacun dans sa propre cellule, avec recherche d’hyperparamètres par validation croisée. Meilleur résultat : SVM linéaire sur la représentation combinée, 94,8 % d’accuracy et 93,4 % de F1 macro en validation croisée.

La sélection des modèles est effectuée par validation croisée stratifiée sur 80 % des sessions étiquetées. Les 20 % restants sont réservés à l’évaluation de la partie 7, qui reste à compléter avec la discussion et la soumission. Chaque préprocesseur est ajusté au sein du pipeline, uniquement sur les données d’entraînement du pli concerné.

## Rendu final : `rendu_final.ipynb`

`rendu_final.ipynb` est le notebook de rendu. Il suit la structure des indications (parties 1 à 10) et reprend la méthode qui a obtenu **0,991** sur Kaggle :

- un tableau de 616 variables par session (proportions des types d'actions, écrans, configurations et chaînes, rythme, navigateur), exploité par un Random Forest et des Extra Trees ;
- une représentation TF-IDF des sessions (actions complètes, paires d'actions consécutives, composants), exploitée par sept SVM linéaires ;
- une combinaison des deux familles par moyenne géométrique des probabilités, avec la règle du navigateur et une correction des fréquences des utilisateurs.

Le notebook réserve 20 % des sessions étiquetées pour l'évaluation. Il compare les modèles par validation croisée à 3 plis sur les 80 % restants, une cellule par entraînement. Il évalue ensuite le modèle final sur la validation réservée, puis le réajuste sur toutes les sessions pour écrire `submission.csv` à la racine.

La section 7.5 mesure à nouveau le modèle final avec le protocole de nos essais préliminaires : validation croisée à 4 plis sur toutes les sessions étiquetées, répétée sur 3 découpages.

| Évaluation du modèle final | Sessions d'entraînement | Accuracy | F1 macro |
|---|---|---|---|
| Validation croisée à 3 plis sur 80 % des sessions (partie 6) | ≈ 1 747 par pli | 97,5 % | 97,2 % |
| Validation réservée, 656 sessions (partie 7) | 2 620 | 96,8 % | 95,7 % |
| Validation croisée répétée, 4 plis × 3 découpages sur toutes les sessions (section 7.5) | ≈ 2 457 par pli | 98,1 % | 97,8 % |
| Classement public Kaggle (partie 10) | 3 276 | — | 0,991 |

Le modèle est le même dans toutes ces mesures. Les écarts viennent du protocole, en particulier du nombre de sessions d'entraînement : avec 247 utilisateurs, chaque session compte. La section 7.5 retrouve les scores de nos essais préliminaires (98,1 % d'accuracy, 0,980 de F1 macro).

**Exécution complète : environ deux heures et demie, dont une heure et demie pour la section 7.5, pour un pic de mémoire d'environ 3 Go.** Fermer les applications gourmandes avant de lancer `Run All`. La constante `N_PROCESSUS` de la première cellule de code peut être portée à 3 sur une machine de 16 Go ou plus, ce qui accélère la partie 6. Les graphiques sont écrits dans `figures/rendu_final/` et les résultats dans `resultats/rendu_final/`. Le fichier `submission.csv` produit est identique à celui soumis sur Kaggle (0,991), `soumissions/submission_v5_geometrique.csv`.

## Contenu

- `rendu_final.ipynb` : notebook de rendu (voir ci-dessus).
- `solution.ipynb` : notebook de travail précédent (48 caractéristiques résumées, TF-IDF des composants).
- `BE_CS1_indications.ipynb` : trame pédagogique fournie par le professeur.
- `BE_CS1_instructions.ipynb` : énoncé et consignes du projet.
- `requirements.txt` : dépendances Python et versions utilisées.
- `figures/` : graphiques exportés par le notebook.
- `resultats/` : comparaison des modèles, détail des hyperparamètres et configuration du protocole ; `resultats/rendu_final/` contient aussi l'évaluation sur la validation réservée et la validation croisée répétée.
- `submission.csv` : prédictions du test produites par `rendu_final.ipynb` (0,991 sur Kaggle).

## Données

Télécharger les données depuis la [compétition Kaggle](https://www.kaggle.com/competitions/qui-utilise-mon-appli-challenge/), puis les placer dans un dossier `Data` à la racine :

```text
Data/
├── train.csv
├── test.csv
└── sample_submission.csv
```

Les CSV du dossier `Data/` et l’environnement virtuel sont exclus du versionnement. Les données comprennent 3 279 sessions d’entraînement, 247 utilisateurs et 324 sessions de test. Une ligne correspond à une session ; les sessions ont des longueurs variables.

## Installation sous Windows (PowerShell)

Le notebook de rendu s’exécute avec **Python 3.14**, comme demandé dans l’énoncé (version utilisée : 3.14.10). Depuis le dossier du projet :

```powershell
py -3.14 -m venv .venv314
.\.venv314\Scripts\python.exe -m pip install -r requirements.txt
.\.venv314\Scripts\python.exe -m ipykernel install --user --name tp_ml_s9_py314 --display-name "Python 3.14 (TP_ML_S9)"
```

Le notebook de travail `solution.ipynb` a été développé sous Python 3.13, avec l’environnement `.venv` et le noyau `Python (TP_ML_S9)`.

## Exécution du notebook

1. Ouvrir le dossier du projet dans VS Code avec les extensions Python et Jupyter.
2. Ouvrir `rendu_final.ipynb` et sélectionner l’interpréteur `.venv314` ou le noyau `Python 3.14 (TP_ML_S9)`.
3. Exécuter les cellules dans l’ordre, depuis le début. Les chemins sont relatifs à la racine du projet.

Le détail ci-dessous concerne le notebook de travail `solution.ipynb` (pour `rendu_final.ipynb`, voir la section « Rendu final »). Le chargement construit les DataFrames complets comme dans les indications. L’aperçu des 20 premières sessions affiche toutes les colonnes : sa sortie volumineuse n’est pas enregistrée dans cette version, mais peut être régénérée en exécutant la cellule. Les graphiques sont enregistrés dans `figures/`. La cellule avant la partie 5 exporte les caractéristiques dans `Data/caracteristiques_train.csv` et `Data/caracteristiques_test.csv`. La partie 6 teste 32 configurations sur trois plis, réparties en neuf entraînements, puis réentraîne la meilleure configuration de chacun sur les sessions d’apprentissage. Ces recherches prennent de 20 à 30 minutes sur l’environnement de développement, hors préparation des données ; la durée dépend de la machine. Les scores et paramètres sont exportés dans `resultats/`.

## Représentation des sessions

La construction produit 48 colonnes : volumes d’activité, proportions par famille d’actions, diversité, enchaînements, contextes dominants et indicateurs temporels, dont des transformations logarithmiques. La cible utilisateur est séparée des caractéristiques.

Chaque session est aussi transformée en document : navigateur, types d’actions, écrans, configurations, chaînes et paires d’éléments consécutifs. Ces documents sont pondérés par TF-IDF dans le pipeline (vocabulaire appris sur les seules sessions d’apprentissage), seuls ou combinés aux 48 caractéristiques pondérées.

Le plus grand marqueur `tXX` représente un horizon temporel enregistré, pas nécessairement la durée réelle de connexion. Les répétitions de marqueurs sont prises en compte explicitement. Les statistiques des parties 3 à 5 sont descriptives. Les scores de la partie 6 servent à sélectionner un modèle ; ils doivent être complétés par l’évaluation sur les sessions réservées en partie 7.

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
