"""Représentation « texte » des sessions, partagée par le notebook et les scripts de recherche.

Chaque session devient un document : une liste de « mots ». Les mots sont le navigateur, chaque action
brute (type + écran + configuration + chaîne + suffixe), chaque paire d'actions consécutives, et les
composants séparés de chaque action. Ces documents sont ensuite vectorisés par TF-IDF.
"""
import gzip
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

est_repere_temps = re.compile(r't\d+')
motif_type = re.compile(r'[(<$]')
motif_ecran = re.compile(r'\((.*?)\)')
motif_conf = re.compile(r'<(.*?)>')
motif_chaine = re.compile(r'\$(.*?)\$')


def trouver_fichier(racine, nom):
    """Chemin de `nom`.csv ou `nom`.csv.GZ, cherché dans Data/, data/ puis à la racine du projet."""
    for dossier in [racine / 'Data', racine / 'data', racine]:
        for suffixe in ['.csv', '.csv.GZ', '.csv.gz']:
            chemin = dossier / f'{nom}{suffixe}'
            if chemin.exists():
                return chemin
    raise FileNotFoundError(f'{nom}.csv introuvable dans {racine}/Data, {racine}/data ou {racine}')


def lire_lignes(chemin):
    """Lignes d'un fichier texte, compressé (.gz) ou non."""
    ouvrir = gzip.open if str(chemin).lower().endswith('.gz') else open
    with ouvrir(chemin, 'rt', encoding='utf-8') as f:
        return f.read().splitlines()


def charger_documents(chemin, index, avec_util):
    """Documents des sessions `index` (numéros de ligne du fichier) : [navigateur, action_1, action_2, ...].

    Les repères de temps tXX sont retirés.
    """
    lignes = lire_lignes(chemin)
    decalage = 2 if avec_util else 1  # train : util, navigateur, actions... ; test : navigateur, actions...
    documents = []
    for i in index:
        champs = lignes[i].split(',')
        actions = [a for a in champs[decalage:] if not est_repere_temps.fullmatch(a)]
        documents.append([f'NAV={champs[decalage - 1]}'] + actions)
    return documents


def mots_session(document, paires=True, triplets=False, composants=True):
    """Liste des « mots » d'une session, utilisée comme analyseur par le TF-IDF."""
    navigateur, actions = document[0], document[1:]
    mots = [navigateur] + actions
    if paires:
        mots += [f'{a} >> {b}' for a, b in zip(actions, actions[1:])]
    if triplets:
        mots += [f'{a} >> {b} >> {c}' for a, b, c in zip(actions, actions[1:], actions[2:])]
    if composants:
        for a in actions:
            mots.append('TYPE=' + motif_type.split(a)[0])
            mots += ['ECRAN=' + e for e in motif_ecran.findall(a)]
            mots += ['CONF=' + c for c in motif_conf.findall(a)]
            mots += ['CHAINE=' + c for c in motif_chaine.findall(a)]
    return mots


class Analyseur:
    """Analyseur paramétrable (objet plutôt que lambda, pour pouvoir être copié par scikit-learn)."""

    def __init__(self, paires=True, triplets=False, composants=True):
        self.paires, self.triplets, self.composants = paires, triplets, composants

    def __call__(self, document):
        return mots_session(document, self.paires, self.triplets, self.composants)


def svm_tfidf(C=1.0, min_df=1, **options_mots):
    """TF-IDF des mots de session + SVM linéaire pondéré."""
    return make_pipeline(
        TfidfVectorizer(analyzer=Analyseur(**options_mots), sublinear_tf=True, lowercase=False, min_df=min_df),
        LinearSVC(C=C, class_weight='balanced'),
    )


def softmax(scores, temperature):
    """Transforme les scores de décision du SVM en pseudo-probabilités (plus T est petit, plus elles sont tranchées)."""
    z = np.asarray(scores) / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def combiner(scores_svm, proba_forets, temperature, poids_svm):
    """Moyenne pondérée des pseudo-probabilités du SVM et des probabilités des forêts."""
    return poids_svm * softmax(scores_svm, temperature) + (1 - poids_svm) * np.asarray(proba_forets)
