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

**Exécution complète : environ deux heures et demie, dont une heure et demie pour la section 7.5, pour un pic de mémoire d'environ 3 Go.** Fermer les applications gourmandes avant de lancer `Run All`. La constante `N_PROCESSUS` de la première cellule de code peut être portée à 3 sur une machine de 16 Go ou plus, ce qui accélère la partie 6. Les graphiques sont écrits dans `figures/rendu_final/` et les résultats dans `resultats/rendu_final/`. Le fichier `submission.csv` produit est identique à celui soumis sur Kaggle (0,991).

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
