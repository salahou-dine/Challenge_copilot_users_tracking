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


def charger_documents(chemin, index, avec_util, avec_temps=False):
    """Documents des sessions `index` (numéros de ligne du fichier) : [navigateur, action_1, action_2, ...].

    - avec_temps=False : les repères de temps tXX sont retirés ;
    - avec_temps=True : chaque repère tXX devient le mot TEMPS, gardé à sa place dans la séquence (repères
      consécutifs fusionnés). Les paires « action >> TEMPS » indiquent alors les actions suivies d'un passage
      de 5 secondes, c'est-à-dire le rythme de l'utilisateur.
    """
    lignes = lire_lignes(chemin)
    decalage = 2 if avec_util else 1  # train : util, navigateur, actions... ; test : navigateur, actions...
    documents = []
    for i in index:
        champs = lignes[i].split(',')
        document = [f'NAV={champs[decalage - 1]}']
        for a in champs[decalage:]:
            if not est_repere_temps.fullmatch(a):
                document.append(a)
            elif avec_temps and document[-1] != 'TEMPS':
                document.append('TEMPS')
        documents.append(document)
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
            if a == 'TEMPS':
                continue
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
    """Moyenne pondérée des pseudo-probabilités du SVM et des probabilités des forêts.

    `scores_svm` peut être une liste de matrices de scores (plusieurs SVM) : on fait alors la moyenne de leurs
    pseudo-probabilités avant de les combiner aux forêts.
    """
    liste = scores_svm if isinstance(scores_svm, (list, tuple)) else [scores_svm]
    proba_svm = np.mean([softmax(s, temperature) for s in liste], axis=0)
    return poids_svm * proba_svm + (1 - poids_svm) * np.asarray(proba_forets)


def fenetres(document, rng, n=8, long_min=30, long_max=200):
    """`n` extraits contigus de `long_min` à `long_max` éléments d'une session (navigateur gardé en tête).

    Sert à l'augmentation de données : chaque extrait est une « session courte » supplémentaire du même
    utilisateur. Les sessions trop courtes (moins de 2 × long_min éléments) ne sont pas découpées.
    """
    navigateur, sequence = document[0], document[1:]
    if len(sequence) < 2 * long_min:
        return []
    extraits = []
    for _ in range(n):
        longueur = rng.integers(long_min, min(long_max, len(sequence) - 1) + 1)
        debut = rng.integers(0, len(sequence) - longueur + 1)
        extraits.append([navigateur] + sequence[debut:debut + longueur])
    return extraits


def augmenter(documents, etiquettes, rng, **options):
    """Documents d'origine + leurs extraits (même étiquette). À n'appliquer qu'aux données d'entraînement."""
    docs, ys = [], []
    for document, etiquette in zip(documents, etiquettes):
        docs.append(document)
        ys.append(etiquette)
        for extrait in fenetres(document, rng, **options):
            docs.append(extrait)
            ys.append(etiquette)
    return docs, ys


def masque_navigateur(navigateurs_sessions, navigateur_par_utilisateur, classes):
    """Matrice (sessions × utilisateurs) : 1 si l'utilisateur a déjà utilisé le navigateur de la session, 0 sinon.

    Dans le train, chaque utilisateur utilise toujours le même navigateur : un utilisateur d'un autre navigateur
    ne peut pas être le bon.
    """
    nav_sessions = np.asarray(list(navigateurs_sessions))
    nav_classes = np.asarray([navigateur_par_utilisateur[c] for c in classes])
    return (nav_sessions[:, None] == nav_classes[None, :]).astype(float)
