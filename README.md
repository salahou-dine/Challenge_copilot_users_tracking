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
