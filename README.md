# Challenge_copilot_users_tracking

Identification des utilisateurs du logiciel **Copilote d’Infologic** à partir de leurs traces d’activité, sur la compétition Kaggle [« Qui utilise mon appli ? »](https://www.kaggle.com/competitions/qui-utilise-mon-appli-challenge/). Projet de classification supervisée multiclasse (247 utilisateurs) réalisé dans le cadre du module MOD 7.2 — Introduction à la science des données.

Le rendu est le notebook **`rendu_final.ipynb`**. Il suit la structure des indications du professeur (parties 1 à 10) : exploration des données, construction des caractéristiques, comparaison des modèles, évaluation, interprétation, puis prédiction du test et fichier de soumission.

## Méthode

- Un **tableau de 616 variables** par session (proportions des types d’actions, écrans, configurations et chaînes, rythme, navigateur), exploité par un Random Forest et des Extra Trees.
- Une **représentation TF-IDF** des sessions (actions complètes, paires d’actions consécutives, composants), exploitée par sept SVM linéaires.
- Une **combinaison** des deux familles par moyenne géométrique des probabilités, avec la règle du navigateur et une correction des fréquences des utilisateurs.

## Résultats

Le notebook réserve 20 % des sessions étiquetées pour l’évaluation finale. Il compare les modèles par validation croisée à 3 plis sur les 80 % restants, une cellule par entraînement. Il évalue ensuite le modèle final sur la validation réservée, puis le réajuste sur toutes les sessions pour prédire le test.

| Évaluation du modèle final | Sessions d’entraînement | Accuracy | F1 macro |
|---|---|---|---|
| Validation croisée à 3 plis sur 80 % des sessions (partie 6) | ≈ 1 747 par pli | 97,5 % | 97,2 % |
| Validation réservée, 656 sessions (partie 7) | 2 620 | 96,8 % | 95,7 % |
| Validation croisée répétée, 4 plis × 3 découpages sur toutes les sessions (section 7.5) | ≈ 2 457 par pli | 98,1 % | 97,8 % |
| Classement public Kaggle (partie 10) | 3 276 | — | **0,991** |

Le modèle est le même dans toutes ces mesures. Les écarts viennent du protocole, en particulier du nombre de sessions d’entraînement : avec 247 utilisateurs, chaque session compte. Le meilleur modèle seul (un SVM linéaire sur la TF-IDF) plafonne à 93,6 % de F1 macro en validation croisée.

## Contenu

- `rendu_final.ipynb` : notebook de rendu, exécuté de bout en bout.
- `submission.csv` : prédictions du test produites par le notebook (0,991 sur Kaggle).
- `figures/rendu_final/` : graphiques exportés par le notebook.
- `resultats/rendu_final/` : comparaison des modèles, réglages testés, évaluation sur la validation réservée et validation croisée répétée.
- `BE_CS1_instructions.ipynb` et `BE_CS1_indications.ipynb` : énoncé et trame fournis par le professeur.
- `requirements.txt` : dépendances Python et versions utilisées.

## Données

Télécharger les données depuis la [compétition Kaggle](https://www.kaggle.com/competitions/qui-utilise-mon-appli-challenge/data), puis les placer dans un dossier `Data` à la racine :

```text
Data/
├── train.csv
├── test.csv
└── sample_submission.csv
```

Les données ne sont pas versionnées. Elles comprennent 3 279 sessions d’entraînement (3 276 après le retrait de 3 sessions de 5 actions ou moins), 247 utilisateurs et 324 sessions de test. Une ligne correspond à une session ; les sessions ont des longueurs variables.

## Installation sous Windows (PowerShell)

Le notebook s’exécute avec **Python 3.14**, comme demandé dans l’énoncé (version utilisée : 3.14.10). Depuis le dossier du projet :

```powershell
py -3.14 -m venv .venv314
.\.venv314\Scripts\python.exe -m pip install -r requirements.txt
.\.venv314\Scripts\python.exe -m ipykernel install --user --name tp_ml_s9_py314 --display-name "Python 3.14 (TP_ML_S9)"
```

## Exécution

1. Ouvrir le dossier du projet dans VS Code avec les extensions Python et Jupyter.
2. Ouvrir `rendu_final.ipynb` et sélectionner l’interpréteur `.venv314` ou le noyau `Python 3.14 (TP_ML_S9)`.
3. Exécuter toutes les cellules dans l’ordre (`Run All`). Les chemins sont relatifs à la racine du projet.

**Durée : environ deux heures et demie, dont une heure et demie pour la section 7.5, pour un pic de mémoire d’environ 3 Go.** Fermer les applications gourmandes avant de lancer l’exécution. La constante `N_PROCESSUS` de la première cellule de code peut être portée à 3 sur une machine de 16 Go ou plus, ce qui accélère la partie 6.

Les graphiques sont écrits dans `figures/rendu_final/`, les résultats dans `resultats/rendu_final/` et la soumission dans `submission.csv`, à la racine.
