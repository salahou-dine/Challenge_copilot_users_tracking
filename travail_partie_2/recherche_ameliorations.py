"""Améliorations du modèle combiné (v2, v3) : évaluation sur 3 découpages.

Lancement (depuis le dossier travail_partie_2/) :
    python recherche_ameliorations.py

Durée : environ 45 minutes. Étapes :
1. Prédictions hors échantillon de chaque modèle de base, pour 3 découpages (graines 0, 1, 2), mises en cache
   dans resultats/cache_oof/ (non versionné) :
   - forêts (Random Forest + Extra Trees, 616 variables) ;
   - SVM TF-IDF sans repères de temps, C = 1 (modèle de la section 6) ;
   - SVM TF-IDF sans repères de temps, C = 3 ;
   - SVM TF-IDF avec repères de temps, C = 10 ;
   - SVM TF-IDF avec repères de temps, C = 10, entraîné avec augmentation (8 extraits de 30 à 200 éléments par
     session). L'augmentation n'est appliquée qu'aux plis d'entraînement.
2. Évaluation des combinaisons (température T, poids w du SVM, règle du navigateur) ; résultats dans
   resultats/ameliorations.csv, relus par le notebook.
"""
from pathlib import Path
import itertools
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from representation_texte import (augmenter, charger_documents, combiner, masque_navigateur, svm_tfidf,
                                  trouver_fichier)

warnings.filterwarnings('ignore')
DOSSIER = Path(__file__).resolve().parent
RACINE = DOSSIER.parent
CACHE = DOSSIER / 'resultats' / 'cache_oof'
CACHE.mkdir(parents=True, exist_ok=True)
GRAINES = [0, 1, 2]

train = pd.read_csv(RACINE / 'donnees_preparees' / 'train_variables.csv.gz', index_col='session')
y = train.pop('util')
X = train
y_liste = list(y)
classes = np.sort(y.unique())
docs_sans_temps = charger_documents(trouver_fichier(RACINE, 'train'), X.index, avec_util=True)
docs_avec_temps = charger_documents(trouver_fichier(RACINE, 'train'), X.index, avec_util=True, avec_temps=True)
nav_cols = [c for c in X if c.startswith('nav_')]
navigateur = X[nav_cols].idxmax(axis=1)
masque = masque_navigateur(navigateur, navigateur.groupby(y).first(), classes)


def forets():
    return VotingClassifier([
        ('rf', RandomForestClassifier(n_estimators=500, class_weight='balanced', n_jobs=-1, random_state=0)),
        ('et', ExtraTreesClassifier(n_estimators=500, class_weight='balanced', max_features=0.15,
                                    min_samples_leaf=2, n_jobs=-1, random_state=0)),
    ], voting='soft')


def svm_augmente_oof(cv, graine):
    """Validation croisée manuelle : augmentation des seuls plis d'entraînement."""
    rng = np.random.default_rng(graine)
    scores = np.zeros((len(docs_avec_temps), len(classes)))
    for idx_train, idx_eval in cv.split(docs_avec_temps, y):
        docs, ys = augmenter([docs_avec_temps[i] for i in idx_train], [y_liste[i] for i in idx_train], rng,
                             n=8, long_min=30, long_max=200)
        modele = svm_tfidf(C=10).fit(docs, ys)
        scores[idx_eval] = modele.decision_function([docs_avec_temps[i] for i in idx_eval])
    return scores


MODELES = {
    'forets': lambda cv, g: cross_val_predict(forets(), X, y, cv=cv, method='predict_proba'),
    'svm': lambda cv, g: cross_val_predict(svm_tfidf(C=1), docs_sans_temps, y, cv=cv, method='decision_function'),
    'svm_C3': lambda cv, g: cross_val_predict(svm_tfidf(C=3), docs_sans_temps, y, cv=cv, method='decision_function'),
    'svm_temps_C10': lambda cv, g: cross_val_predict(svm_tfidf(C=10), docs_avec_temps, y, cv=cv,
                                                     method='decision_function'),
    'svm_temps_C10_augmente': svm_augmente_oof,
}

# ---------------------------------------------------------------------------
# 1. Prédictions hors échantillon (cache)
# ---------------------------------------------------------------------------
for nom, calcul in MODELES.items():
    for g in GRAINES:
        fichier = CACHE / f'{nom}_g{g}.npy'
        if fichier.exists():
            continue
        debut = time.time()
        np.save(fichier, calcul(StratifiedKFold(n_splits=4, shuffle=True, random_state=g), g))
        print(f'{nom} graine {g} : {time.time() - debut:.0f} s', flush=True)


def charger(nom, g):
    return np.load(CACHE / f'{nom}_g{g}.npy')


def mesures(proba):
    pred = classes[np.argmax(proba, axis=1)]
    return accuracy_score(y, pred), f1_score(y, pred, average='macro')


# ---------------------------------------------------------------------------
# 2. Évaluation des versions
# ---------------------------------------------------------------------------
VERSIONS = {
    'forêts seules': lambda g: charger('forets', g),
    'SVM sans temps C=1 seul': lambda g: charger('svm', g),
    'SVM avec temps C=10 seul': lambda g: charger('svm_temps_C10', g),
    'SVM avec temps C=10 augmenté seul': lambda g: charger('svm_temps_C10_augmente', g),
    'SVM C=1 + forêts (T=0,2, w=0,6) [modèle de la section 7]':
        lambda g: combiner(charger('svm', g), charger('forets', g), 0.2, 0.6),
    '+ règle du navigateur':
        lambda g: combiner(charger('svm', g), charger('forets', g), 0.2, 0.6) * masque,
    'v2 : SVM avec temps C=10 + forêts (T=0,2, w=0,7) + navigateur':
        lambda g: combiner(charger('svm_temps_C10', g), charger('forets', g), 0.2, 0.7) * masque,
    'v3 : 3 SVM + forêts (T=0,5, w=0,9) + navigateur':
        lambda g: combiner([charger(n, g) for n in ['svm_temps_C10', 'svm_temps_C10_augmente', 'svm_C3']],
                           charger('forets', g), 0.5, 0.9) * masque,
}
for T, w in itertools.product([0.2, 0.3, 0.5], [0.6, 0.7, 0.8, 0.9]):
    VERSIONS[f'grille v3 : T={T}, w={w}'] = (
        lambda g, T=T, w=w: combiner([charger(n, g) for n in ['svm_temps_C10', 'svm_temps_C10_augmente', 'svm_C3']],
                                     charger('forets', g), T, w) * masque)

lignes = []
for version, calcul in VERSIONS.items():
    for g in GRAINES:
        acc, f1 = mesures(calcul(g))
        lignes.append({'version': version, 'graine': g, 'accuracy': acc, 'f1_macro': f1})
resultats = pd.DataFrame(lignes)
resultats.to_csv(DOSSIER / 'resultats' / 'ameliorations.csv', index=False)
print(resultats.groupby('version', sort=False)[['accuracy', 'f1_macro']].mean().round(4).to_string())
