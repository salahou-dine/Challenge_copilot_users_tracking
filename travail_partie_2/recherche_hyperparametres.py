"""Recherche d'hyperparamètres pour Random Forest, Extra Trees et XGBoost.

Lancement (depuis le dossier travail_partie_2/) :
    python recherche_hyperparametres.py

Durée : environ 1 heure. Les résultats sont enregistrés dans resultats/ et relus par
partie_2_modelisation.ipynb, qui n'a donc pas besoin de refaire ces calculs.

- Random Forest et Extra Trees : grille complète, validation croisée stratifiée à 4 plis.
- XGBoost : trop coûteux avec 247 classes (un arbre par classe et par itération) pour une
  validation croisée complète ; chaque configuration est évaluée sur le premier pli.
"""
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

DOSSIER = Path(__file__).resolve().parent
DOSSIER_PREPARE = DOSSIER.parent / 'donnees_preparees'
DOSSIER_RESULTATS = DOSSIER / 'resultats'
DOSSIER_RESULTATS.mkdir(exist_ok=True)

train = pd.read_csv(DOSSIER_PREPARE / 'train_variables.csv.gz', index_col='session')
y = train.pop('util')
X = train
cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)  # même découpage que le notebook


def recherche_grille(nom, modele, grille):
    """Validation croisée de toutes les combinaisons de la grille ; résultats enregistrés en CSV."""
    debut = time.time()
    recherche = GridSearchCV(modele, grille, cv=cv, scoring=['accuracy', 'f1_macro'], refit=False, n_jobs=1)
    recherche.fit(X, y)
    res = pd.DataFrame(recherche.cv_results_)
    colonnes = [c for c in res.columns if c.startswith('param_')] + [
        'mean_test_accuracy', 'std_test_accuracy', 'mean_test_f1_macro', 'std_test_f1_macro', 'mean_fit_time']
    res = res[colonnes].sort_values('mean_test_f1_macro', ascending=False)
    res.to_csv(DOSSIER_RESULTATS / f'recherche_{nom}.csv', index=False)
    print(f'{nom} : {len(res)} combinaisons en {(time.time() - debut) / 60:.0f} min')
    print(res.head(5).round(3).to_string(index=False), '\n', flush=True)


# ---------------------------------------------------------------------------
# 1. Random Forest
# ---------------------------------------------------------------------------
recherche_grille('random_forest', RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=0), {
    'class_weight': [None, 'balanced', 'balanced_subsample'],
    'max_features': ['sqrt', 0.05, 0.1],
    'min_samples_leaf': [1, 2],
})

# ---------------------------------------------------------------------------
# 2. Extra Trees
# ---------------------------------------------------------------------------
recherche_grille('extra_trees', ExtraTreesClassifier(n_estimators=500, n_jobs=-1, random_state=0), {
    'class_weight': [None, 'balanced'],
    'max_features': ['sqrt', 0.05, 0.15, 0.3],
    'min_samples_leaf': [1, 2],
})

# ---------------------------------------------------------------------------
# 3. Nombre d'arbres (avec les réglages par défaut pondérés de chaque forêt)
# ---------------------------------------------------------------------------
recherche_grille('nombre_arbres', RandomForestClassifier(class_weight='balanced', n_jobs=-1, random_state=0), {
    'n_estimators': [100, 300, 500, 1000],
})

# ---------------------------------------------------------------------------
# 4. XGBoost (premier pli seulement)
# ---------------------------------------------------------------------------
y_code = LabelEncoder().fit_transform(y)
idx_train, idx_eval = next(cv.split(X, y))
poids_equilibres = compute_sample_weight('balanced', y_code[idx_train])

configurations = [
    # (max_depth, learning_rate, n_estimators, colsample_bytree, pondération équilibrée)
    (3, 0.1, 300, 0.2, False),
    (6, 0.1, 300, 0.2, False),
    (3, 0.3, 300, 0.2, False),
    (6, 0.3, 300, 0.2, False),
    (3, 0.1, 300, 0.2, True),
]
lignes = []
for profondeur, taux, n_iter, colsample, pondere in configurations:
    debut = time.time()
    modele = XGBClassifier(n_estimators=n_iter, max_depth=profondeur, learning_rate=taux, subsample=0.8,
                           colsample_bytree=colsample, tree_method='hist', n_jobs=-1, random_state=0)
    modele.fit(X.iloc[idx_train], y_code[idx_train], sample_weight=poids_equilibres if pondere else None)
    pred = modele.predict(X.iloc[idx_eval])
    lignes.append({
        'max_depth': profondeur, 'learning_rate': taux, 'n_estimators': n_iter,
        'colsample_bytree': colsample, 'ponderation_equilibree': pondere,
        'accuracy': accuracy_score(y_code[idx_eval], pred),
        'f1_macro': f1_score(y_code[idx_eval], pred, average='macro'),
        'temps_s': time.time() - debut,
    })
    print(lignes[-1], flush=True)
    pd.DataFrame(lignes).sort_values('f1_macro', ascending=False).to_csv(
        DOSSIER_RESULTATS / 'recherche_xgboost_pli1.csv', index=False)

print('Terminé.')
