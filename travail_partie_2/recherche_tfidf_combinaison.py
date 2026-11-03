"""Recherche pour la représentation « texte » (TF-IDF + SVM) et sa combinaison avec les forêts.

Lancement (depuis le dossier travail_partie_2/) :
    python recherche_tfidf_combinaison.py

Durée : environ 25 minutes. Résultats enregistrés dans resultats/ et relus par partie_2_modelisation.ipynb.

1. Variantes du TF-IDF + SVM linéaire : validation croisée à 4 plis (même découpage que le notebook).
2. Combinaison SVM + forêts (température T du softmax, poids du SVM) : validation croisée répétée sur
   3 découpages différents, pour distinguer un vrai gain du bruit de la mesure.
"""
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from representation_texte import charger_documents, combiner, svm_tfidf, trouver_fichier

warnings.filterwarnings('ignore')

DOSSIER = Path(__file__).resolve().parent
DATA_DIR = DOSSIER.parent
DOSSIER_RESULTATS = DOSSIER / 'resultats'
DOSSIER_RESULTATS.mkdir(exist_ok=True)

train = pd.read_csv(DATA_DIR / 'donnees_preparees' / 'train_variables.csv.gz', index_col='session')
y = train.pop('util')
X = train
documents = charger_documents(trouver_fichier(DATA_DIR, 'train'), y.index, avec_util=True)
classes = np.sort(y.unique())


def scores(pred):
    return accuracy_score(y, pred), f1_score(y, pred, average='macro')


# ---------------------------------------------------------------------------
# 1. Variantes TF-IDF + SVM linéaire
# ---------------------------------------------------------------------------
cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
variantes = [
    ('actions seules', dict(paires=False, composants=False)),
    ('actions + paires', dict(paires=True, composants=False)),
    ('actions + paires, C = 0,3', dict(paires=True, composants=False, C=0.3)),
    ('actions + paires + triplets', dict(paires=True, triplets=True, composants=False)),
    ('actions + paires + composants', dict(paires=True, composants=True)),
    ('actions + paires + triplets + composants', dict(paires=True, triplets=True, composants=True)),
    ('actions + paires + composants, min_df = 2', dict(paires=True, composants=True, min_df=2)),
]
lignes = []
for nom, options in variantes:
    debut = time.time()
    pred = cross_val_predict(svm_tfidf(**options), documents, y, cv=cv)
    acc, f1 = scores(pred)
    lignes.append({'variante': nom, 'accuracy': acc, 'f1_macro': f1, 'erreurs': int((pred != y.values).sum()),
                   'temps_s': time.time() - debut})
    print(lignes[-1], flush=True)
pd.DataFrame(lignes).to_csv(DOSSIER_RESULTATS / 'tfidf_variantes.csv', index=False)

# ---------------------------------------------------------------------------
# 2. Combinaison SVM (meilleure variante) + forêts, sur 3 découpages
# ---------------------------------------------------------------------------
TEMPERATURES = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
POIDS_SVM = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
lignes = []
for graine in [0, 1, 2]:
    debut = time.time()
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=graine)
    scores_svm = cross_val_predict(svm_tfidf(paires=True, composants=True), documents, y, cv=cv,
                                   method='decision_function')
    forets = VotingClassifier([
        ('rf', RandomForestClassifier(n_estimators=500, class_weight='balanced', n_jobs=-1, random_state=0)),
        ('et', ExtraTreesClassifier(n_estimators=500, class_weight='balanced', max_features=0.15,
                                    min_samples_leaf=2, n_jobs=-1, random_state=0)),
    ], voting='soft')
    proba_forets = cross_val_predict(forets, X, y, cv=cv, method='predict_proba')

    for nom, proba in [('forêts seules', proba_forets), ('SVM TF-IDF seul', scores_svm)]:
        acc, f1 = scores(classes[np.argmax(proba, axis=1)])
        lignes.append({'graine': graine, 'version': nom, 'temperature': None, 'poids_svm': None,
                       'accuracy': acc, 'f1_macro': f1})
    for temperature in TEMPERATURES:
        for poids in POIDS_SVM:
            acc, f1 = scores(classes[np.argmax(combiner(scores_svm, proba_forets, temperature, poids), axis=1)])
            lignes.append({'graine': graine, 'version': 'combinaison', 'temperature': temperature,
                           'poids_svm': poids, 'accuracy': acc, 'f1_macro': f1})
    print(f'graine {graine} terminée en {time.time() - debut:.0f} s', flush=True)
    pd.DataFrame(lignes).to_csv(DOSSIER_RESULTATS / 'combinaison.csv', index=False)

print('Terminé.')
