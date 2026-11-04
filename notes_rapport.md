# Notes pour le rapport — Identification des utilisateurs de Copilote

Document de travail : constats, chiffres et choix faits au fil de l'exploration, à reprendre dans le rapport final (PDF, 5 à 8 pages).
Les chiffres viennent du notebook `travail_partie_1/partie_1_decouverte_donnees.ipynb`.

---

## 1. Le problème

- **Objectif :** retrouver quel utilisateur est derrière une session du logiciel Copilote (ERP agroalimentaire édité par Infologic), à partir des traces d'utilisation.
- **Type de tâche :** classification supervisée multi-classe.
- **Entrée :** une session = navigateur + suite ordonnée d'actions.
- **Sortie :** l'identifiant de l'utilisateur (code de 3 lettres, ex. `nuh`).
- **Rendu :** `submission.csv` (`RowId`, `prediction`) pour les 324 sessions du test, sur Kaggle et Moodle. Date limite : 9/11/26.
- **Métriques demandées :** accuracy et F1-score (au minimum).
- **Métrique du classement Kaggle :** « F1 score moyen », c'est-à-dire le F1 (moyenne harmonique de la précision et du rappel) calculé par classe puis moyenné, soit le **F1 macro** : chaque utilisateur compte autant, quel que soit son nombre de sessions. Nos choix ont été faits sur cette métrique (tri des résultats, `class_weight='balanced'`, ensemble). Réserve : certaines compétitions Kaggle calculent un « Mean F1 » session par session, qui équivaut alors à l'accuracy pour une prédiction unique ; sur toutes nos comparaisons, accuracy et F1 macro désignent le même modèle gagnant.

**Contexte métier (page Kaggle) :**
- Les utilisateurs sont des **testeurs internes d'Infologic** : ils utilisent divers profils pour tester des fonctionnalités, puis doivent reporter chaque problème depuis leur vrai profil, ce qui est long et rébarbatif. Le modèle permettrait d'identifier automatiquement le testeur.
- Second usage : **détection d'intrusion**, quand quelqu'un se connecte avec le compte d'un autre utilisateur.
- Ce contexte explique nos observations : les testeurs travaillent par rôle et par module, d'où les configurations métier (`INSTALLATEUR`, `dispatch`, `MT_AppelN1`), les chaînes de spécialité et des habitudes très marquées.

## 2. Les données

### 2.1 Fichiers

| Fichier | Contenu | Taille |
|---|---|---|
| `train.csv.GZ` | 3 279 sessions étiquetées | 70 Mo décompressé |
| `test.csv.GZ` | 324 sessions sans étiquette | 6,6 Mo décompressé |
| `sample_submission.csv` | format de soumission attendu | — |

Format d'une ligne :
- train : `utilisateur, navigateur, action_1, action_2, …`
- test : `navigateur, action_1, action_2, …`

### 2.2 Incident sur les données (à mentionner brièvement)

Un premier `train.csv` local était corrompu : il ne contenait que les 4,5 premiers Mo du vrai fichier (244 sessions au lieu de 3 279), avec 61 octets remplacés par des retours à la ligne au milieu des mots. La comparaison avec le notebook des enseignants (3 279 × 14 470) a permis de le détecter. Les fichiers ont été retéléchargés depuis Kaggle et vérifiés (UTF-8 valide, aucune ligne anormale, `test` identique octet pour octet à l'ancien).

**Choix :** les fichiers sont lus directement compressés avec `gzip`, sans décompression ni ouverture dans un éditeur ou Excel (limite de 32 767 caractères par cellule, incompatible avec nos sessions).

### 2.3 Choix de chargement

- Lecture ligne par ligne puis découpage sur la virgule, comme recommandé dans les indications.
- Stockage des actions sous forme de **liste** dans une seule colonne, plutôt qu'au format large de `pd.read_csv` (14 470 colonnes presque toutes vides, qui ne sont pas des variables comparables d'une session à l'autre).
- Il n'y a pas de vraies valeurs manquantes dans les données brutes : les NaN du format large ne sont que du remplissage dû aux longueurs variables.
- Aucun doublon de session dans le train ni dans le test.

### 2.4 Structure d'une action

```
Exécution d'un bouton(infologic.orga.FicheForm)<DEF_03/24>$JCP$1
└──── type d'action ──┘└──────── écran ────────┘└─ conf ─┘└chaîne┘└ suffixe
```

| Élément | Délimiteur | Valeurs distinctes | % des actions qui le portent |
|---|---|---|---|
| type d'action | (début) | 35 | 100 % |
| écran | `( … )` | 418 | 12,7 % |
| configuration d'écran | `< … >` | 219 | 4,6 % |
| chaîne (catégorie de fiche) | `$ … $` | 76 | 15,9 % |
| suffixe `1` | fin | — | 15,6 % |

- Les pourcentages portent sur les 2,27 millions d'actions, pas sur les sessions : 68 % des sessions contiennent au moins une chaîne.
- Le sens du suffixe `1` n'est pas documenté dans nos fichiers (**à vérifier sur la page Kaggle**).

### 2.5 Les repères de temps `tXX`

- `t5`, `t10`, `t15`… marquent le temps écoulé dans la session, **en secondes**, par pas de 5 s. Ce ne sont pas des actions : ils sont exclus des comptages d'actions.
- Constat : deux repères consécutifs sont presque toujours espacés d'exactement 5 s (jamais de saut du type `t100` → `t400`), et il y a presque toujours au moins une action entre deux repères. Les pauses n'apparaissent donc pas.
- **Interprétation probable (non confirmée) :** le compteur mesure le **temps actif**, pas le temps réel de connexion.
- Aucune date ni heure n'est disponible : impossible d'étudier la période (heure de la journée, jour de la semaine).

## 3. La variable cible

- **247 utilisateurs**, 3 279 sessions.
- Sessions par utilisateur : min 4, médiane 12, moyenne 13, max 75.
- Les plus représentés : `skm` (75), `slq` (71), `cjr` (46), `flj` (42), `hjs` (37).
- **Classes déséquilibrées** (rapport de 1 à 19) : un modèle naïf favorisera les gros utilisateurs. D'où l'intérêt du **F1 macro** en plus de l'accuracy.
- Chaque utilisateur a au moins 4 sessions : une **validation croisée stratifiée** (ex. `StratifiedKFold` à 4 plis) est possible.

## 4. Constats de l'exploration

### 4.1 Volume d'activité

| | min | médiane | max |
|---|---|---|---|
| Actions par session | 0 | 340 | 11 977 |
| Actions moyennes par session (par utilisateur) | 91 | 595 | 6 729 |
| Actions totales (par utilisateur) | 435 | 7 607 | 38 815 |

- Le nombre de sessions et le volume d'actions ne désignent pas les mêmes utilisateurs :
  - `skm` et `slq` ont beaucoup de sessions courtes (≈ 210-265 actions) ;
  - `fxg` n'a que 5 sessions, mais énormes (6 729 actions en moyenne).
- Moyenne nettement au-dessus de la médiane pour les actions par session (790 contre 595) : quelques utilisateurs aux sessions très longues tirent la moyenne vers le haut.
- **Conséquence :** utiliser des **proportions** plutôt que des comptages bruts pour comparer des sessions de longueurs très différentes.

### 4.2 Dimension temporelle

| | min | médiane | max |
|---|---|---|---|
| Durée (temps actif) | 5 s | ≈ 6,5 min | 3 h 17 |
| Actions par minute | 0 | ≈ 53 | 556 |
| Actions par fenêtre de 5 s (moyenne) | 1 | 4,3 | 35 |
| Pic d'actions dans une fenêtre de 5 s | 1 | 25 | 1 859 |

- Rythme typique : environ une action par seconde.
- Les utilisateurs ont des vitesses différentes : `sii`, `qof` ≈ 87 actions/min contre `zll` ≈ 28 (3 fois plus lent).
- `fxg` : sessions de 72 min en médiane, contre ≈ 6,5 min pour l'ensemble des sessions.
- **Part des écarts entre sessions expliquée par l'utilisateur (η²) :** actions/min 32 %, nombre d'actions 29 %, durée 27 %, pic par fenêtre 26 %.
  → signal réel mais insuffisant seul : les profils des utilisateurs se chevauchent (voir les boxplots de la section 4.2 du notebook).

### 4.3 Le navigateur

- **Chacun des 247 utilisateurs utilise toujours le même navigateur**, sans aucune exception dans le train.

| Navigateur | Utilisateurs | Sessions train | Sessions test |
|---|---|---|---|
| Firefox | 114 | 1 466 | 147 |
| Google Chrome | 93 | 1 339 | 133 |
| Microsoft Edge | 39 | 451 | 42 |
| Opera | **1** | 23 | 2 |

- Le navigateur sépare les utilisateurs en 4 groupes disjoints. Les 2 sessions Opera du test appartiennent très probablement à l'unique utilisateur Opera.
- **Prudence :** c'est une observation sur le train, pas une règle garantie pour le test.
- **Choix :** encodage one-hot (4 colonnes) donné au modèle, plutôt qu'une règle codée à la main.

### 4.4 Types d'actions

- 35 types. Les plus fréquents : Exécution d'un bouton (≈ 510 000), Saisie dans un champ (≈ 266 000), Sélection d'un écran (≈ 207 000).
- On distingue deux catégories d'actions :
  - **utilisées par (presque) tous** : bouton, saisie, dialogues, affichage d'une erreur (≈ 246)… Elles distinguent par leur **proportion** (ex. `rmf` : 38 % de clics sur un bouton, contre 23 % pour l'utilisateur typique) ;
  - **utilisées par une minorité** : leur **simple présence** est un indice fort.

| Type | Utilisateurs (sur 247) |
|---|---|
| Entrée en saisie dans un formulaire | 48 |
| Lancement d'un tableau de bord | 57 |
| Sélection d'un flag | 76 |
| Sélection / Désélection d'un élément | ≈ 100 |
| Clic sur une grille d'historique de recherche | ≈ 205 |
| Raccourci | ≈ 226 |

- Le graphique compte les utilisateurs ayant fait l'action **au moins une fois sur toutes leurs sessions**, pas à chaque session.
- **Part des écarts entre sessions expliquée par l'utilisateur (η²) pour la proportion du type :** > 95 % pour Lancement d'un tableau de bord, Sélection d'un flag, Entrée en saisie dans un formulaire, Filtrage / Tri ; 92 % pour Clic sur une grille d'historique.
  → **signal beaucoup plus fort que le temps.**
- Suffixe `1` : sa proportion explique ≈ 40 % des écarts entre sessions.

**Carte de chaleur des profils (15 utilisateurs ayant le plus de sessions) :**
- Socle commun : Exécution d'un bouton (21 à 35 %) et Saisie dans un champ (7 à 32 %) dominent chez tous.
- Signatures individuelles sur des types moins fréquents : Chainage 0,14 et grille d'historique 0,10 pour `hjs` ; Raccourci 0,11 pour `vfo` ; Affichage d'un toast 0,12 pour `nqg` et 0,16 pour `bgn` ; Retour sur un écran 0,13 pour `muz` ; Entrée en saisie dans un formulaire 0,10 pour `fvc` (contre ≈ 0 à 0,05 chez les autres).
- Profils presque identiques pour certains utilisateurs : `skm` (bouton 0,32, saisie 0,29) et `fwf` (0,31, 0,27). Les types d'actions seuls ne suffiront pas : il faut les combiner avec le navigateur, les écrans, les chaînes et le rythme.

### 4.5 Écrans, configurations, chaînes

| | Valeurs | Utilisées par un seul utilisateur |
|---|---|---|
| Écrans | 418 | **147** |
| Configurations | 219 | 41 |
| Chaînes | 76 | 9 |

- Les écrans les plus fréquents (BlankController, FichiersController, MAINT…) sont utilisés par plus de 200 utilisateurs : peu discriminants. Ce sont les **écrans rares** qui portent le signal.
**Écrans les plus fréquents :** InputFormNestedWindow (240 utilisateurs), BlankController (236), FichiersController (232), InputFormHomeView (225), MAINT (212), PlanningController (208), AccueilController (198), DEV (178), StatFBController (150).
- Ce sont les écrans de base du logiciel (accueil, écran vide, saisie, fichiers) : presque tout le monde y passe, ils ne distinguent pas les utilisateurs. Le signal est dans les écrans rares (147 propres à un seul utilisateur).
- `MAINT` et `DEV` ne sont pas des noms de classes `infologic.…` : peut-être un mode ou un environnement (maintenance, développement). **Hypothèse à vérifier.**

**Configurations les plus fréquentes :**

| Configuration | Actions | Utilisateurs | Lecture |
|---|---|---|---|
| STD | 10 544 | 241 | standard, tout le monde |
| DEF_03/24 / DEF_07/23 | 18 186 / 16 313 | 202 / 203 | par défaut, apparemment datées |
| TELEM_MINIMAL | 7 595 | 196 | très répandue |
| 001 / 005 | 4 116 / 2 630 | 162 / 111 | moyennement répandues |
| INSTALLATEUR | 9 567 | 108 | réservée à un groupe |
| ACCUEIL_INST | 2 648 | 90 | réservée à un groupe |
| MT_AppelN1 | 3 092 | 26 | très restreinte (≈ 120 actions/utilisateur) |
| dispatch | 2 279 | 12 | très restreinte (≈ 190 actions/utilisateur) |

- Les configurations restreintes semblent correspondre à des **rôles métier** (installateur, appel niveau 1, dispatch). **Hypothèse** cohérente : le poste détermine les écrans utilisés.
- `dispatch` (12 utilisateurs) et `MT_AppelN1` (26) sont très discriminantes ; STD et DEF_… ne le sont pas.

**Bilan des trois composantes :**

| | Les plus fréquentes sont utilisées par | Pouvoir discriminant |
|---|---|---|
| Écrans | 150 à 240 utilisateurs | faible pour les fréquents, fort pour les rares |
| Configurations | 12 à 241 utilisateurs | certaines très restreintes, liées au rôle |
| Chaînes | 77 à 162 utilisateurs | moyen |

→ Ne pas se limiter aux valeurs les plus fréquentes. **Piste : TF-IDF**, qui donne plus de poids aux valeurs rares et moins à celles partagées par tous.

- **21 utilisateurs n'utilisent jamais de chaîne** (dont `skm`, `slq`, `shh`, `fwf`) : cette absence est elle-même une information.
- 32 % des sessions ne contiennent aucune chaîne.
- Pour les autres, **66 %** des sessions ont pour chaîne principale la chaîne habituelle de l'utilisateur.
- Chaînes les plus fréquentes : VT, GP, EDI, TEC, OT.

**Top des chaînes (nombre d'actions et d'utilisateurs) :**

| Chaîne | Actions | Utilisateurs (sur 247) | Actions par utilisateur |
|---|---|---|---|
| VT | 53 480 | 123 | ≈ 435 |
| GP | 39 424 | 108 | ≈ 365 |
| EDI | 33 359 | 124 | ≈ 269 |
| TEC | 26 315 | 112 | ≈ 235 |
| OT | 24 509 | 162 | ≈ 151 |
| SV | 23 681 | 111 | ≈ 213 |
| GESCO | 22 173 | 111 | ≈ 200 |
| JCP | 21 653 | 77 | ≈ 281 |
| ST | 17 160 | 115 | ≈ 149 |
| WEB | 11 410 | 113 | ≈ 101 |

- Une chaîne courante ne concerne qu'**environ la moitié des utilisateurs** (77 à 162 sur 247), contrairement à beaucoup de types d'actions utilisés par tous. Connaître la chaîne d'une session élimine environ la moitié des candidats : c'est utile, sans être décisif seul.
- **OT** : la plus partagée (162 utilisateurs) mais peu utilisée par chacun (≈ 151 actions), donc peu discriminante.
- **VT, GP** : utilisées intensivement par un groupe de 110 à 120 personnes (≈ 365 à 435 actions chacune), probablement leur cœur de métier.
- **JCP** : la plus restreinte du top 10 (77 utilisateurs), donc plus discriminante.
- Le tableau est trié par fréquence : les chaînes **les plus discriminantes sont les rares** (9 propres à un seul utilisateur), qui n'y figurent pas.
- L'**absence totale de chaîne** (21 utilisateurs) est aussi une information.

**Carte « chaîne principale des sessions » (15 utilisateurs ayant le plus de sessions)** — trois profils :
- **Sans chaîne** : `skm`, `slq`, `shh`, `fwf` (aucune session avec chaîne) ; `flj` et `kwi` presque (3 à 6 sessions avec chaîne sur 31 à 42). `skm` et `fwf`, déjà proches par leurs types d'actions, ne se distinguent pas non plus par les chaînes : il faudra d'autres variables (navigateur, rythme, écrans).
- **Spécialistes** : `vfo` → VT (21 sessions sur 35), `nqg` → GP (19 sur 34), `nzq` → IADEV (17 sur 26), `muz` → DB (14 sur 28). Pour eux, la chaîne est un indice très fort.
- **Polyvalents** : `cjr` (ORGA 7, WEB 7, ACTI 3… une dizaine de chaînes), `bgn` (GESCO 8, EDI 5, GP 4, SV 3), `fvc` (GESCO 9, EDI 8, TEC 3), `kks`. Une chaîne seule est moins décisive, mais la combinaison reste caractéristique.
- → **Choix de variables :** la proportion de chaque chaîne dans la session, plus un indicateur « aucune chaîne », plutôt que la seule chaîne principale.
- 8 des 219 écrans présents dans le test n'apparaissent jamais dans le train.

## 5. Valeurs manquantes, valeurs aberrantes, cohérence train / test

### 5.1 Valeurs manquantes
- Aucune valeur manquante dans les données brutes.
- Variables calculées : seulement 2 sessions concernées, pour des calculs impossibles et non des données perdues.
  - `fvc` : 0 action (`t5,t10,t15`), donc moyenne et pic par fenêtre de 5 s impossibles à calculer.
  - `fou` : 1 action dans une seule fenêtre, donc écart-type impossible (il faut au moins 2 valeurs).

### 5.2 Valeurs aberrantes : l'échelle compte
Règle de Tukey (boxplot : en dehors de Q1 − 1,5 × IQR et Q3 + 1,5 × IQR) :

| Variable | Échelle normale | Échelle log |
|---|---|---|
| nb_actions | 297 | 6 |
| duree_s | 256 | 5 |
| actions_par_min | 135 | 102 |
| pic d'actions sur 5 s | 157 | 59 |

→ Les variables de volume et de durée sont très asymétriques. Les sessions longues ne sont pas des erreurs, c'est la forme normale des données. **Choix : passage au log, pas de suppression.**

### 5.3 Sessions vides ou très courtes
- 3 sessions avec 5 actions ou moins : `fvc` (0), `fou` (1), `zus` (2). Les deux dernières ouvrent seulement l'écran d'accueil.
- Elles n'apportent aucune information. Leurs utilisateurs gardent 13 à 29 autres sessions, et le test n'en contient aucune de ce genre.
- **Choix : retirées du train.**

### 5.4 Rafales : accident ou habitude ?

| Utilisateur | Plus gros pic (actions / 5 s) | Pic médian de ses sessions | Interprétation |
|---|---|---|---|
| opp | 1 859 | 28 | accident isolé |
| vzc | 1 226 | 38 | accident isolé |
| flj | 1 008 | 16 | accident isolé |
| oec | 403, 331, 249 | 26 | rafales fréquentes |
| fxg | 294, 264, 261 | 261 | habitude (signature) |

- Les accidents sont dominés par « Lancement d'une action générique » et « Raccourci » en cascade : ce n'est pas un rythme humain, probablement un traitement automatique.
- Les pics récurrents (`fxg`) sont caractéristiques de l'utilisateur et doivent être conservés.
- **Choix : aucune suppression** (chaque session compte quand certains utilisateurs n'en ont que 4). **Plafonnement** au 99e percentile (≈ 140) et passage au log.

### 5.5 Cohérence train / test

| | Médiane train | Médiane test | Max train | Max test |
|---|---|---|---|---|
| nb_actions | 340 | 365 | 11 977 | 6 203 |
| duree_s | 395 | 408 | 11 790 | 7 110 |
| actions_par_min | 53 | 52 | 556 | 171 |
| pic sur 5 s | 25 | 25 | 1 859 | 252 |

- Médianes presque identiques : le test est représentatif du train.
- Les rafales extrêmes n'existent que dans le train, ce qui confirme qu'il s'agit d'accidents.
- Le test ne contient ni session vide ni session de 5 actions ou moins.
- Valeurs du test jamais vues dans le train : 8 écrans (7 sessions), 2 configurations (2 sessions), 0 chaîne, 0 type d'action. Impact faible : elles seront ignorées en construisant les variables du test avec **la liste des colonnes du train**.

### 5.6 Récapitulatif des traitements

| Problème | Sessions | Traitement |
|---|---|---|
| Sessions vides ou presque (≤ 5 actions) | 3 | retirées du train |
| Valeurs manquantes (variables calculées) | 2 | disparaissent avec le retrait ci-dessus |
| Asymétrie des variables de volume et de temps | toutes | passage au log |
| Rafales accidentelles | une dizaine | plafonnement au 99e percentile |
| Écrans / configurations inconnus dans le test | 9 | ignorés (colonnes du train) |

## 6. Construction des variables (fait)

Tableau final : **une ligne par session, 616 variables numériques**.

| Famille | Variables | Colonnes | Justification |
|---|---|---|---|
| Navigateur | one-hot | 4 | groupes d'utilisateurs disjoints |
| Volume et temps | log(nb actions), log(durée), actions/min (plafonné), moyenne / log du max (plafonné) / écart-type des actions par fenêtre de 5 s | 6 | ≈ 30 % des écarts expliqués par l'utilisateur |
| Types d'actions | proportion de chacun des 35 types | 35 | signatures très fortes (η² > 0,9 pour certains) |
| Suffixe `1` | proportion | 1 | ≈ 40 % des écarts expliqués |
| Chaînes | proportion de chaque chaîne + indicateur « aucune chaîne » | 68 + 1 | profils spécialistes / polyvalents / sans chaîne |
| Écrans | proportion de chaque écran | 303 | écrans rares = signatures |
| Configurations | proportion de chaque configuration | 198 | configurations liées au rôle métier |

**Choix méthodologiques :**
- **Proportions** (comptage ÷ nombre d'actions de la session) plutôt que comptages bruts, car les sessions ont des longueurs très différentes.
- **Log et plafonnement au 99e percentile du train** : pic sur 5 s plafonné à 140 actions, vitesse plafonnée à 134 actions/min.
- **Vocabulaire appris sur le train uniquement** : écrans, configurations et chaînes présents dans au moins 2 sessions (418 → 303 écrans, 219 → 198 configurations, 76 → 68 chaînes). Il est ensuite appliqué au test, qui a donc les mêmes colonnes dans le même ordre ; les valeurs inconnues sont ignorées. Cela évite toute **fuite d'information** du test vers l'apprentissage.
- Écart-type non calculable (une seule fenêtre de 5 s) remplacé par 0.
- Cible encodée avec `pd.Categorical` : 247 classes, codes de 0 à 246.
- Résultat : X_train 3 276 × 616, X_test 324 × 616, aucune valeur manquante, aucune colonne constante.
- Sauvegarde : `donnees_preparees/train_variables.csv.gz` et `test_variables.csv.gz`.

**Pistes d'amélioration à tester en modélisation :** TF-IDF au lieu des proportions ; séquences d'actions (paires d'actions consécutives) ; seuil `min_sessions` du vocabulaire.

## 7. Modélisation (partie 2)

Notebook : `travail_partie_2/partie_2_modelisation.ipynb`.

### 7.1 Protocole
- Validation croisée **stratifiée à 4 plis** (`StratifiedKFold`, `shuffle=True`, `random_state=0`), possible car chaque utilisateur a au moins 4 sessions.
- Métriques : **accuracy** et **F1 macro** (chaque utilisateur compte autant, adapté au déséquilibre des classes).
- Standardisation (`StandardScaler`) dans un pipeline pour les modèles sensibles à l'échelle (régression logistique, SVM, kNN, MLP), apprise sur les plis d'entraînement uniquement.

### 7.2 Comparaison des modèles (réglages par défaut)

| Modèle | Accuracy | F1 macro |
|---|---|---|
| Extra Trees (500 arbres) | 0,933 | 0,912 |
| Random Forest (500 arbres) | 0,925 | 0,907 |
| Naive Bayes (gaussien) | 0,836 | 0,800 |
| Régression logistique | 0,804 | 0,754 |
| MLP (1 couche de 256) | 0,799 | 0,746 |
| SVM noyau linéaire | 0,793 | 0,742 |
| Arbre de décision | 0,728 | 0,687 |
| Analyse discriminante linéaire (LDA) | 0,719 | 0,687 |
| SVM noyau RBF | 0,695 | 0,673 |
| kNN (k = 1) | 0,603 | 0,546 |
| SVM noyau polynomial (deg. 2) | 0,436 | 0,382 |
| Référence (classe majoritaire) | 0,023 | 0,000 |

- Écarts-types entre plis < 1,5 point : le classement est fiable.
- **Pourquoi les forêts gagnent :** nos variables fonctionnent comme des seuils (« utilise cet écran ? », « plus de 10 % de raccourcis ? ») et des combinaisons (navigateur + chaîne + type), ce que les arbres découpent naturellement. La moyenne de centaines d'arbres corrige l'instabilité d'un arbre seul.
- Les méthodes à distance (kNN, SVM RBF / polynomial) souffrent des 616 dimensions, dont beaucoup de proportions presque toujours nulles.
- Naive Bayes étonnamment bon (83,6 %) malgré l'hypothèse d'indépendance : il modélise le profil moyen de chaque utilisateur. La LDA est pénalisée par son hypothèse de covariance commune à tous les utilisateurs.

**Boosting (évalué sur un seul pli pour des raisons de coût : 247 classes = un arbre par classe et par itération) :**

| Modèle (1er pli) | Accuracy | F1 macro | Temps |
|---|---|---|---|
| Random Forest balanced (référence, même pli) | 0,947 | 0,940 | ≈ 10 s |
| XGBoost (300 itérations, profondeur 4, lr 0,1) | 0,888 | 0,852 | ≈ 280 s |
| AdaBoost (200 arbres de profondeur 10) | 0,676 | 0,659 | ≈ 190 s |
| HistGradientBoosting (200 itérations, lr 0,1) | 0,210 | 0,181 | ≈ 270 s |

- XGBoost reste 6 points sous la Random Forest, pour un coût bien plus élevé.
- AdaBoost est sensible aux sessions atypiques et mal adapté à 247 classes.
- HistGradientBoosting diverge : trop peu d'exemples positifs par classe et par arbre ; il faudrait un taux d'apprentissage plus faible et de l'arrêt précoce.
- **Conclusion :** avec beaucoup de classes et peu d'exemples par classe, le **bagging** (forêts d'arbres indépendants) est plus efficace et plus robuste que le **boosting** séquentiel.

### 7.3 Recherche d'hyperparamètres

Script `travail_partie_2/recherche_hyperparametres.py` (≈ 1 h), résultats dans `travail_partie_2/resultats/`, relus par le notebook.

| Famille | Hyperparamètres testés | Combinaisons | Évaluation |
|---|---|---|---|
| Random Forest (500 arbres) | `class_weight` (aucune / balanced / balanced_subsample) × `max_features` (sqrt / 0,05 / 0,1) × `min_samples_leaf` (1 / 2) | 18 | CV 4 plis |
| Extra Trees (500 arbres) | `class_weight` (aucune / balanced) × `max_features` (sqrt / 0,05 / 0,15 / 0,3) × `min_samples_leaf` (1 / 2) | 16 | CV 4 plis |
| Random Forest balanced | `n_estimators` (100 / 300 / 500 / 1 000) | 4 | CV 4 plis |
| XGBoost | `max_depth` (3 / 6) × `learning_rate` (0,1 / 0,3), + variante pondérée | 5 | 1er pli (coût) |

**Effet moyen de chaque hyperparamètre (F1 macro moyen sur les grilles) :**

| Hyperparamètre | Random Forest | Extra Trees |
|---|---|---|
| `class_weight` | aucune 0,909 / **balanced 0,940** / balanced_subsample 0,926 | aucune 0,914 / **balanced 0,934** |
| `max_features` | sqrt 0,923 / 0,05 0,925 / 0,1 0,927 | sqrt 0,910 / 0,05 0,918 / **0,15 0,934** / 0,3 0,933 |
| `min_samples_leaf` | 1 : 0,925 / 2 : 0,925 | 1 : 0,926 / 2 : 0,922 |
| `n_estimators` (RF) | 100 : 0,930 / 300 : 0,936 / **500 : 0,942** / 1 000 : 0,941 | — |

- **`class_weight='balanced'` est le réglage décisif** (+2 à +3 points). Sans pondération, les forêts favorisent les utilisateurs aux nombreuses sessions.
- Les **Extra Trees** ont besoin de plus de variables par nœud (`max_features` 0,15) : leurs seuils étant tirés au hasard, ils compensent en examinant plus de variables. Effet faible pour la RF.
- `min_samples_leaf` : sans effet notable. Plateau à partir de 500 arbres.
- **XGBoost** (1er pli) : meilleur réglage profondeur 3, `learning_rate` 0,1 → **0,880 / 0,847**, contre 0,947 / 0,940 pour la RF sur le même pli, et ≈ 30 fois plus lent. `learning_rate` 0,3 dégrade (0,844), profondeur 6 n'apporte rien, la pondération non plus (0,868). **Écarté.**

### 7.4 Choix du modèle final

| Candidat (meilleur réglage) | Accuracy | F1 macro |
|---|---|---|
| Random Forest : balanced, `max_features=sqrt`, `min_samples_leaf=1` | 0,946 ± 0,005 | 0,942 ± 0,003 |
| Extra Trees : balanced, `max_features=0.15`, `min_samples_leaf=2` | 0,949 ± 0,008 | 0,945 ± 0,004 |
| **Ensemble RF + ET** (`VotingClassifier`, `voting='soft'`) | **0,951 ± 0,005** | **0,947 ± 0,005** |

- Écarts faibles mais ordre identique sur les deux métriques, et confirmé par Kaggle (RF 0,920 < ET 0,930 < ensemble 0,938).
- **Modèle retenu : ensemble RF + ET**, moyenne des probabilités des deux forêts. La RF (seuils optimisés) et les ET (seuils aléatoires) ne font pas les mêmes erreurs : la moyenne en corrige une partie.

### 7.5 Analyse des erreurs du modèle final (prédictions hors échantillon, `cross_val_predict`)
- **95,1 %** bien classées, **161 erreurs** sur 3 276 (178 pour la RF seule), F1 macro 0,950.
- Confusions principales : `skm` → `slq` (6), `slq` → `skm` (6), `flj` → `skm` (4). Les trois utilisent **Chrome**, **aucune chaîne**, et ont des profils d'actions proches. L'ensemble réduit ces confusions (`slq` → `skm` : 11 → 6).
- Les moins bien reconnus ont peu de sessions : `azx` (10 sessions, F1 0,14), `jew` (4), `hll` (9), `urs` (5), `lra` (4), `azn` (4).
- Piste : des variables de **séquence** (paires d'actions consécutives) pour séparer `skm`, `slq` et `flj`.

### 7.6 Importance des variables (impureté, moyenne des deux forêts)

| Famille | % de l'importance | Colonnes |
|---|---|---|
| Types d'actions | 38,9 % | 35 |
| Écrans | 24,9 % | 303 |
| Configurations | 16,1 % | 198 |
| Chaînes | 10,7 % | 68 |
| Volume / temps / autres | 6,8 % | 8 |
| Navigateur | 2,6 % | 4 |

- Top des variables individuelles : proportions de Raccourci, Retour sur un écran, Affichage d'un toast, Clic sur une grille d'historique, Double-clic, Filtrage / Tri (≈ 2 % chacune), soit les signatures vues dans la carte de chaleur. Puis des écrans (BlankController, AccueilController, PlanningController) et `conf_INSTALLATEUR`.
- **Le navigateur ne pèse que 2,6 %** alors qu'il sépare parfaitement les utilisateurs : l'importance par impureté favorise les variables continues et compte peu une variable binaire utilisée tôt et rarement. Limite de la mesure à signaler ; l'importance par permutation serait complémentaire.

### 7.7 Prédiction sur le test (modèle final)
- `submission.csv` : 324 lignes, `RowId` de 1 à 324, 218 utilisateurs différents prédits.
- **100 %** des prédictions respectent la règle « même navigateur » : le modèle l'a apprise seul.
- Confiance médiane **0,54** (0,44 pour la RF seule) ; 22 sessions sous 0,2 (37 pour la RF seule).
- 4 prédictions diffèrent de la soumission « RF + ET » déjà envoyée (0,938) : effet du réglage `min_samples_leaf=2` des Extra Trees.

### 7.8 Soumissions Kaggle (Qui utilise mon appli ? v2026/2027 - Groupe 3)

| Date | Modèle | Score public | Validation croisée (acc. / F1 macro) |
|---|---|---|---|
| 02/11/2026 | Random Forest 500 arbres, `class_weight='balanced'`, 616 variables | 0,920 | 0,946 / 0,944 |
| 02/11/2026 | Extra Trees 500 arbres, `class_weight='balanced'`, `max_features=0.15` | 0,930 | 0,950 / 0,946 |
| 02/11/2026 | **Moyenne des probabilités RF + ET** (ensemble, ET avec `min_samples_leaf=1`) | **0,938** | **0,952 / 0,949** |
| à soumettre | Ensemble RF + ET réglés (ET avec `min_samples_leaf=2`) | ? | 0,951 / 0,950 |
| 03/11/2026 | SVM TF-IDF (actions + paires + composants) + forêts RF + ET, T = 0,2, poids SVM 0,6 | 0,969 | 0,968 / 0,965 (3 découpages) |
| 04/11/2026 | v2 : SVM avec temps (C = 10) + forêts, T = 0,2, w = 0,7, règle du navigateur | 0,980 | 0,974 / 0,971 (3 découpages) |
| 04/11/2026 | **v3 : moyenne de 3 SVM (temps C = 10 ; temps C = 10 augmenté ; sans temps C = 3) + forêts, T = 0,5, w = 0,9, règle du navigateur** | **0,985** | **0,978 / 0,975** (3 découpages) |

- v2 et v3 ne diffèrent de la soumission à 0,969 que de 2 et 5 prédictions, mais gagnent 1,1 et 1,6 point : la partie publique du test est petite, chaque session y pèse lourd. L'ordre des scores Kaggle reste celui de la validation croisée.

(Les valeurs de validation croisée de ce tableau sont calculées sur les prédictions hors échantillon de `cross_val_predict`, d'où 0,944 au lieu de 0,942 pour la moyenne des plis.)

- **Les scores Kaggle suivent exactement l'ordre de la validation croisée** : le protocole d'évaluation est fiable et sert de guide pour choisir le modèle.
- L'**ensemble** RF + ET est le meilleur : moyenner deux familles d'arbres différentes (seuils optimisés pour la RF, tirés au hasard pour les ET) corrige une partie des erreurs propres à chacune.
- Les trois modèles ne diffèrent que sur 9 à 12 des 324 sessions du test : les gains restent modestes.
- Écart de ≈ 1,5 à 2,5 points avec la validation croisée. Explication probable : le score public ne porte souvent que sur une partie des 324 sessions du test (une erreur ≈ 0,3 à 0,6 point selon la taille de cette partie), donc il est bruité. Les sessions du test peuvent aussi être un peu plus difficiles (voir les 37 sessions à confiance < 0,2).
- Métrique du classement : **F1 score moyen** (F1 macro), voir partie 1.
- **Limite de 3 soumissions par jour et par équipe.** On ne peut donc pas choisir les modèles « au score public » : les décisions sont prises sur la validation croisée (3 276 sessions, plus fiable que le score public), et seuls les candidats qui la battent sont soumis.

### 7.9 Travail sur les variables (après le choix du modèle)

Toutes les expériences utilisent le modèle final (ensemble RF + ET) et la même validation croisée à 4 plis. Référence : 616 variables, accuracy 0,951, F1 macro 0,950, 161 erreurs.

**Diagnostic des erreurs : les sessions courtes**

| Sessions de… | Nombre | Accuracy (RF) |
|---|---|---|
| ≤ 50 actions | 47 | 0,723 |
| ≤ 100 actions | 429 | 0,860 |
| ≤ 200 actions | 1 060 | 0,888 |
| ≤ 500 actions | 2 002 | 0,922 |
| toutes | 3 276 | 0,946 |

- Sessions mal classées : médiane de 142 actions et 6 écrans distincts, contre 362 actions et 10 écrans pour les bien classées. Une session courte contient trop peu d'écrans et de configurations « signatures » : c'est une **limite de l'information disponible**.
- `skm`, `slq` et `flj` ont pourtant des écrans très différents (`skm` : facturation des achats, `FactureAchForm`, `DocumentController` ; `slq` : interventions et chantiers, `InterventionController`, `ChantierForm`). Isolés, ils sont distingués à 88 % ; les confusions viennent de leurs sessions courtes sans ces écrans.

**Expériences**

| Version | Variables | Accuracy | F1 macro | Erreurs | Acc. sessions ≤ 200 actions |
|---|---|---|---|---|---|
| Référence | 616 | 0,951 | 0,950 | 161 | 0,895 |
| + bigrammes de types (paires d'actions consécutives, ≥ 20 sessions) | 1 320 | 0,949 | 0,946 | 166 | — |
| + bigrammes de types (≥ 5 sessions) | 1 463 | 0,949 | 0,946 | 168 | — |
| + actions complètes (type @ écran, ≥ 5 sessions) | 1 416 | 0,955 | 0,951 | 147 | 0,908 |
| + début de session (premier écran, types des 20 premières actions) | 680 | 0,953 | 0,954 | 153 | 0,900 |
| + actions complètes + début de session | 1 480 | 0,952 | 0,949 | 157 | 0,900 |

- **Bigrammes : rejetés.** Ils doublent le nombre de variables et diluent le signal ; les confusions `skm` / `slq` / `flj` ne baissent pas (18 → 16). Ces utilisateurs ne se distinguent pas par l'ordre de leurs actions.
- Actions complètes et début de session : petits gains au premier essai, ciblés sur les sessions courtes. Mais leur combinaison fait moins bien que chacune seule : **les écarts (0,2 à 0,4 point) sont du même ordre que le bruit de la validation croisée** (écart-type entre plis ≈ 0,5 point).

**Vérification : validation croisée répétée sur 3 découpages différents (graines 1, 2, 3)**

| Version | Accuracy (moyenne ± écart-type) | F1 macro |
|---|---|---|
| **Référence (616 variables)** | 0,953 ± 0,002 | **0,951 ± 0,003** |
| + actions complètes | 0,953 ± 0,001 | 0,950 ± 0,000 |
| + début de session | 0,950 ± 0,002 | 0,950 ± 0,005 |

- Gain de F1 selon le découpage : actions complètes −0,36 / +0,15 / −0,13 point ; début de session −0,03 / −0,33 / +0,12 point. Aucune variante ne bat la référence de façon systématique : **les gains du premier essai étaient du bruit**.
- **Décision : on garde les 616 variables.** Les nouvelles familles ajoutent de la complexité sans gain démontré.
- **Leçon méthodologique (à mettre dans le rapport) :** un écart inférieur à l'écart-type entre plis ne prouve rien ; il faut répéter la validation croisée avant de conclure. Les variables actuelles semblent proches de ce que les données permettent : les erreurs restantes viennent surtout de sessions courtes, trop pauvres en écrans et configurations caractéristiques.

### 7.10 Rebondissement : représentation « texte » des sessions (TF-IDF) et combinaison

Le classement Kaggle montre des scores de 0,98 : notre conclusion « limite de l'information » était fausse, il restait du signal. Nos 616 variables **résument** les sessions (proportions par famille) et jettent des informations : les valeurs rares, les combinaisons exactes type + écran + configuration + chaîne, et l'ordre des actions.

**Approche :** chaque session devient un **document**, chaque action brute (`Exécution d'un bouton<DEF_03/24>$JCP$`) un **mot**. On vectorise avec **TF-IDF** (`TfidfVectorizer`, `sublinear_tf=True`) : poids fort pour un mot fréquent dans la session mais rare dans l'ensemble des sessions. Modèle : **SVM linéaire** (`LinearSVC`, `C=1`, `class_weight='balanced'`), adapté aux représentations très larges et creuses.

| TF-IDF + SVM linéaire (CV 4 plis, graine 0) | Accuracy | F1 macro | Erreurs |
|---|---|---|---|
| actions brutes seules | 0,921 | 0,905 | 258 |
| actions + **paires d'actions brutes consécutives** | 0,956 | 0,946 | 145 |
| actions + paires, C = 0,3 | 0,947 | 0,936 | 175 |
| actions + paires + triplets | 0,951 | 0,940 | 161 |
| **actions + paires + composants séparés** (`TYPE=`, `ECRAN=`, `CONF=`, `CHAINE=`) | **0,957** | **0,949** | **140** |
| actions + paires + triplets + composants | 0,955 | 0,946 | 146 |
| idem, `min_df=2` | 0,957 | 0,949 | 141 |

- Les **paires d'actions brutes** font gagner 3,5 points, alors que les paires de *types* n'apportaient rien : l'ordre compte, **à condition de garder tout le détail de l'action**.
- Les composants séparés aident (deux actions sur le même écran se « ressemblent ») ; les triplets non.

**Combinaison SVM + forêts :** scores du SVM transformés en pseudo-probabilités par un *softmax* de température T, puis moyenne pondérée avec les probabilités de l'ensemble RF + ET. Validation sur **3 découpages** (graines 0, 1, 2).

| Version (moyenne ± écart-type sur 3 découpages) | Accuracy | F1 macro |
|---|---|---|
| Forêts seules (RF + ET, 616 variables) | 0,952 ± 0,002 | 0,951 ± 0,003 |
| SVM TF-IDF seul | 0,957 ± 0,002 | 0,947 ± 0,003 |
| Combinaison T = 0,3, poids SVM 0,8 | 0,969 ± 0,002 | 0,966 ± 0,003 |
| **Combinaison T = 0,2, poids SVM 0,6 (retenue)** | **0,968 ± 0,002** | **0,965 ± 0,002** |
| Combinaison T = 0,2, poids SVM 0,5 | 0,967 ± 0,002 | 0,964 ± 0,002 |
| Combinaison T = 1,0, poids SVM 0,5 | 0,953 ± 0,001 | 0,952 ± 0,003 |

- **Gain réel : +1,4 point de F1 macro** (≈ 5 écarts-types), sur les 3 découpages. L'erreur passe de 4,9 % à 3,2 %.
- Les deux modèles sont **complémentaires** : les forêts exploitent les proportions et le rythme, le SVM les combinaisons exactes et l'ordre des actions.
- Plateau stable autour de T = 0,2 à 0,3 et poids SVM 0,6 à 0,8. Réglage retenu **au centre du plateau** (T = 0,2, poids 0,6), le plus stable, plutôt que le maximum isolé.
- Soumission `submission_svm_forets.csv` (223 utilisateurs prédits, confiance médiane 0,66), qui diffère de 9 prédictions de la soumission à 0,938 : **score Kaggle 0,969** (+3,1 points). Le score public rejoint la validation croisée (0,968), qui reste un guide fiable.

**Analyse du modèle final (notebook partie 2, sections 8 à 10, découpage graine 0) :**
- **96,8 % d'accuracy, F1 macro 0,965, 105 erreurs** sur 3 276 (161 pour les forêts seules, soit −35 %).
- Gain surtout sur les **sessions courtes** :

| Nombre d'actions | Sessions | Forêts seules | Modèle final |
|---|---|---|---|
| ≤ 50 | 47 | 0,766 | **0,851** |
| 51-100 | 392 | 0,880 | **0,934** |
| 101-200 | 621 | 0,915 | **0,942** |
| 201-500 | 942 | 0,969 | 0,976 |
| 501-1000 | 623 | 0,976 | 0,984 |
| > 1000 | 651 | 0,991 | 0,995 |

- Confusions `skm` / `slq` / `flj` : 16 → 12. Les moins bien reconnus sont presque tous des utilisateurs à 4 ou 5 sessions (`lra`, `jew`, `azn`, `itt`, `lfq`).
- **Mots du SVM qui votent le plus** (vocabulaire de ≈ 99 000 mots) : pour `skm`, la configuration `ADM_055`, l'écran de saisie comptable `SaisiEcritForm`, l'enchaînement « Action de table >> Lancement d'une action générique » ; pour `slq`, les configurations `ChantierInstallation` et `Feuille_Presence`, le visualiseur PDF, l'écran des interventions. Ce sont des paires précises et des configurations rares, diluées ou absentes dans nos proportions : les deux modèles regardent des choses différentes, d'où le gain de la combinaison.
- Test : 223 utilisateurs prédits, confiance médiane 0,66, 27 sessions sous 0,2. **99,7 %** de cohérence du navigateur (1 session sur 324 fait exception : le SVM peut l'emporter sur la règle). Piste : imposer la règle du navigateur.
- Le notebook reproduit **exactement** la soumission notée 0,969 (0 prédiction différente).
- Code : `travail_partie_2/representation_texte.py` (représentation texte, SVM, combinaison) et `travail_partie_2/recherche_tfidf_combinaison.py` (recherche, ≈ 25 min, résultats dans `resultats/tfidf_variantes.csv` et `resultats/combinaison.csv`).

**Légitimité de la combinaison :** combiner des modèles (*ensemble learning*) est une technique standard, citée dans les indications (« modèle ensembliste »). Elle est valide ici parce que : (1) la combinaison et ses réglages (T, poids) sont choisis **uniquement sur le train**, par validation croisée ; (2) aucune information du test (étiquettes ou distribution) n'est utilisée ; (3) le tout est un modèle unique, reproductible et expliqué.

### 7.11 Vers 0,98 : améliorations du modèle combiné (03-04/11)

Méthode : prédictions hors échantillon de chaque modèle de base calculées une fois par découpage (graines 0, 1, 2) et mises en cache, puis combinaisons évaluées sur les 3 découpages. **Une version n'est retenue que si elle bat la référence sur les 3 découpages.**

**Étape 1 : v2** (référence de départ : SVM + forêts, 0,968 / 0,965)

| Piste | Gain de F1 | Verdict |
|---|---|---|
| Règle du navigateur (probabilité nulle pour les utilisateurs d'un autre navigateur) | +0,2 | retenue |
| Repères de temps gardés dans la séquence (mot `TEMPS`, consécutifs fusionnés) : les paires « action >> TEMPS » encodent le rythme | +0,2 à +0,3 | retenue |
| SVM moins régularisé (C = 3 ou 10) | +0,2 à +0,3 | retenue (C = 10) |
| TF-IDF binaire | +0,2, sans gain cumulé | écartée |
| SVM sans pondération des classes | −1 | écartée |
| Régression logistique sur TF-IDF (seule 0,942 ; en 3e modèle) | 0 | écartée |
| Plus proches voisins cosinus sur TF-IDF (seuls 0,85) | — | écartée |
| Recherche de copies ou de préfixes communs entre train et test | aucun | pas de raccourci dans les données |

**v2 = SVM avec temps (C = 10) + forêts, T = 0,2, w = 0,7, règle du navigateur : 0,974 / 0,971** (gain +0,62 / +0,63 / +0,53 point selon le découpage).

**Étape 2 : analyse des erreurs de la v2** (découpage 0) : 86 erreurs ; le bon utilisateur est 2e dans 58 % des cas, dans le top 5 dans 86 % ; confiance médiane 0,12 sur les erreurs contre 0,78 sur les bonnes réponses ; erreurs dispersées entre de nombreuses paires ; 52 sessions mal classées sur les 3 découpages (cas vraiment ambigus).

**Étape 3 : pistes testées après la v2**

| Piste | Résultat | Verdict |
|---|---|---|
| Méta-modèle (*stacking* : régression logistique sur les scores, rangs, interactions avec la longueur) | +0,06 / +0,16 / −0,12 | écarté : la façon de combiner n'est pas le facteur limitant |
| Mots de navigation entre écrans (paires et triplets d'écrans visités) | −0,04 à −0,08 | écartés |
| Mots de rythme (nombre d'actions par fenêtre de 5 s, en catégories) | +0,1 (1 découpage) | écartés |
| Naive Bayes multinomial / complément sur TF-IDF (seuls 0,72 à 0,80) | — | écartés |
| **Augmentation de données** : fenêtres extraites des sessions d'entraînement (uniquement dans les plis d'entraînement) | 8 fenêtres de 30 à 200 : +0,59 / +0,13 / −0,14 | instable seule ; utile en moyenne de SVM |
| **Moyenne de 3 SVM** (temps C = 10 ; temps C = 10 augmenté ; sans temps C = 3) + forêts, T = 0,5, w = 0,9, navigateur | **+0,47 / +0,49 / +0,12** | **retenue (v3)** |

Tailles de fenêtres testées (découpage 0, gain de F1 en combinaison) : 2 × 200-2 000 : +0,08 ; 3 × 50-400 : +0,34 ; 5 × 100-1 000 : +0,28 ; 8 × 30-200 : +0,59 ; 15 × 20-150 : +0,25. Les fenêtres courtes visent les sessions courtes, où se concentrent les erreurs, mais le gain varie beaucoup selon le découpage.

**v3 : 0,978 / 0,975** en validation croisée (3 découpages). Soumission `submission_v3_3svm.csv` (226 utilisateurs prédits ; 3 prédictions différentes de la v2, 5 de la soumission à 0,969).

**Leçon :** au-delà de 0,97, chaque piste rapporte quelques dixièmes de point et le premier découpage est souvent trompeur (aug8 : +0,59 puis −0,14). La validation répétée est indispensable.

**Résultat Kaggle : v2 = 0,980, v3 = 0,985.** Le notebook de la partie 2 (section 8 et suivantes) reproduit la v3 à l'identique ; script `recherche_ameliorations.py`, résultats `resultats/ameliorations.csv`. Analyse des erreurs de la v3 (découpage 0) : **71 erreurs** (105 pour la combinaison de la section 7, 161 pour les forêts) ; accuracy sous 50 actions 0,766 (forêts) → 0,851 → **0,894** ; entre 51 et 100 actions 0,880 → 0,934 → **0,957** ; confusions `skm` / `slq` / `flj` 16 → 12 → 9. Dans le SVM avec temps, 170 des 1 000 mots les plus influents contiennent `TEMPS` (ex. « TEMPS >> Raccourci(SaisiEcritForm) » pour `skm`) : le rythme est une signature.

### 7.12 Au-delà de la v3 (04/11) : pistes sans gain significatif

| Piste | Résultat (gain de F1 par rapport à la v3) | Verdict |
|---|---|---|
| 4e et 5e SVM augmentés, fenêtres tirées différemment (*bagging*) | +0,18 / −0,02 (2 découpages) | instable |
| SVM augmenté avec 15 fenêtres de 20 à 150 | +0,21 (1 découpage) | non confirmé |
| SVM avec temps C = 3 en 4e modèle | +0,06 / +0,04 / +0,10 | stable mais négligeable |
| Mots « hiérarchie des écrans » (`MODULE=`, `SOUSMODULE=`) | +0,10 (1 découpage, meilleur réglage choisi a posteriori) | négligeable |
| Mots « début / fin de session » (30 premières / dernières actions) | +0,16 (idem) | négligeable |
| Auto-apprentissage (*pseudo-labeling* : sessions d'évaluation les plus sûres ajoutées à l'entraînement, 50 % ou 80 %) | +0,01 à +0,06 (1 découpage) | aucun gain |
| v3 avec T = 0,7, w = 0,95 | +0,10 / +0,08 / +0,06 | stable mais négligeable |

### 7.13 Deuxième session d'améliorations (04/11) : autres modèles, réglages et sélection de variables

| Levier | Piste | Résultat (F1, gain par rapport à la v3) | Verdict |
|---|---|---|---|
| A | **Correction des fréquences de classes** : probabilités divisées par (fréquence de l'utilisateur)^α, pour favoriser les petits utilisateurs (cohérent avec le F1 macro) | α = 0,1 : +0,42 / +0,03 / +0,05 ; α ≥ 0,2 ou < 0 : moins bien | **retenue** (α = 0,1) |
| B | Autres formulations du SVM linéaire (seuls, découpage 0) : Crammer-Singer 0,951 ; perte *hinge* 0,952 ; **sans IDF** 0,952 ; Ridge 0,937 ; SGD *modified huber* 0,894 ; TF non logarithmique 0,905 | en 4e modèle : +0,1 sauf **sans IDF +0,30** | **SVM sans IDF retenu** (+0,31 / +0,24 / +0,12 avec α) |
| C | Forêts enrichies de 200 composantes SVD du TF-IDF (SVD apprise dans chaque pli) | seules 0,9504 (vs 0,9498) ; combinaison −0,05 | écartée |
| D | Sélection non supervisée des variables des forêts (non calculée jusqu'au bout, disque plein) | — | non conclue |
| — | **Bagging de SVM augmentés** : 3 SVM augmentés supplémentaires (8 × 30-200 avec deux autres tirages, 15 × 20-150) | +0,10 / +0,21 / +0,39 | **retenu** |

**v4 = 7 SVM** (avec temps C = 10 ; 4 augmentés ; sans temps C = 3 ; avec temps sans IDF) **+ forêts, T = 0,7, w = 0,95, règle du navigateur, correction des fréquences α = 0,1** : **0,981 d'accuracy, F1 macro 0,979** (gain sur la v3 : +0,23 / +0,38 / +0,58 point ; pire découpage 0,975). Soumission `submission_v4_7svm.csv`.

**Kaggle : v4 = 0,985**, comme la v3 : les 2 prédictions qui diffèrent ne sont visiblement pas dans la partie publique du test.

**Pistes testées après la v4 :**

| Piste | Résultat | Verdict |
|---|---|---|
| **Repères de temps répétés** (tXX identique au précédent, ≈ 25 000 cas) : part de repères répétés très discriminante (η² = 0,58 ; 0 % pour `azn`, `azx`, `bez`, ≈ 13 % pour `syt`, `sdx`, `wwk` ; surtout après Raccourci, Clic sur une grille d'historique, Entrée en saisie dans un formulaire) | SVM avec mot `TREP` : identique au SVM avec temps ; forêts + part et nombre de repères répétés : +0,04 / 0 / +0,12 | information déjà captée indirectement par les paires « action >> TEMPS » |
| Autres artefacts : apostrophe typographique, préfixe `fr.` des écrans, parenthèses en majuscules (MAINT, DEV) | η² de 0,34 à 0,55 mais déjà dans les types d'actions et les écrans | rien de nouveau |
| SVM augmenté sans IDF (meilleur SVM seul : F1 0,9585) ; SVM augmenté sans temps (0,9516) | ajoutés à la v4 : −0,29 à +0,12 | saturation de l'ensemble |

**Autres familles de modèles et sélection de variables (après la v4) :**

| Piste | Seul (F1) | Ajouté à la v4 | Verdict |
|---|---|---|---|
| Profil moyen par utilisateur (Rocchio, cosinus sur TF-IDF) | 0,895 | rattrape 12 des 64 erreurs mais en crée davantage : −0,07 | écarté |
| Session la plus semblable par utilisateur (max ou moyenne des 2 meilleurs cosinus) | 0,864 / 0,875 | −0,1 | écarté |
| Ridge sur TF-IDF | 0,938 | +0,01 / +0,02 / 0 | écarté |
| SVM perte *hinge* ; SVM Crammer-Singer | 0,950 / 0,950 | ≤ +0,05 | écartés |
| Sélection d'ensemble gloutonne (Caruana) parmi 13 modèles, apprise sur 2 découpages, testée sur le 3e | — | 0,976 / 0,978 / 0,969 contre 0,978 / 0,983 / 0,975 pour la v4 | sur-ajustement, écartée |
| Sélection de variables par χ² pour le SVM (10 000 ou 30 000 mots sur ≈ 82 000) | 0,939 / 0,948 (vs 0,952) | — | dégrade : les mots rares sont des signatures |

**Potentiel non exploité :** 45 des 64 erreurs de la v4 (découpage 0) sont bien classées par au moins un des modèles calculés (Ridge en rattrape 21), et le bon utilisateur est 2e dans 32 cas. Mais chaque modèle rattrape des sessions différentes, et aucune règle de combinaison testée (moyenne, méta-modèle, sélection gloutonne) ne sait reconnaître, session par session, quel modèle croire.

**Méta-modèle riche (choisir le bon modèle session par session) :**
- Candidats d'une session : top-5 de la v4 + premier choix de chacun des 17 modèles de base (≈ 5 candidats par session ; le bon utilisateur en fait partie dans 99,6 à 99,7 % des cas).
- Variables par candidat : probabilité et rang selon chaque modèle (34), nombre de modèles qui le placent en tête, probabilité et rang dans la v4, log-fréquence de l'utilisateur ; variables de session : longueur, confiance et marge de la v4, part des modèles d'accord avec la v4.
- Apprentissage imbriqué : méta-modèle entraîné sur 3/4 des sessions, évalué sur le quart restant, à partir de prédictions de base déjà hors échantillon ; 3 découpages.

| Méta-modèle | Gain de F1 par rapport à la v4 (découpages 0 / 1 / 2) | Moyenne |
|---|---|---|
| Gradient boosting (300 itérations, 15 feuilles) | +0,11 / −0,14 / −0,18 | −0,07 |
| Gradient boosting fortement régularisé | −0,04 / 0,00 / −0,27 | −0,10 |
| Gradient boosting, variables agrégées seulement | +0,04 / −0,03 / −0,31 | −0,10 |
| Régression logistique, toutes variables | +0,08 / +0,02 / −0,20 | −0,03 |
| Régression logistique, agrégées + probabilités | +0,08 / −0,02 / −0,12 | −0,02 |

→ **Écarté.** Avec ≈ 3 000 sessions, le méta-modèle n'a pas assez d'exemples d'erreurs (≈ 60 par découpage) pour apprendre de façon fiable quand faire confiance à quel modèle : il sur-ajuste les quelques cas vus.

**Plateau atteint :** v4 = 0,981 / 0,979 en validation croisée (≈ 62 erreurs sur 3 276). Les nouveaux modèles, même meilleurs seuls, n'améliorent plus l'ensemble : ils se trompent sur les mêmes sessions.

**Conclusion (avant la deuxième session, 7.13) :** avec cette famille d'approches, la validation croisée plafonnait vers **0,976 à 0,978** de F1 macro ; la v4 la porte à 0,979. Les erreurs restantes (≈ 70 sur 3 276) sont surtout des sessions courtes et des utilisateurs à 4 ou 5 sessions ; 52 sessions sont mal classées sur les 3 découpages, des cas probablement ambigus. Le score public (0,985) dépasse la validation croisée parce qu'il porte sur une petite partie du test, où chaque session pèse lourd. Atteindre 0,99 de façon fiable demanderait une information nouvelle, que nous n'avons pas trouvée.

## 8. Questions ouvertes

- Sens du suffixe `1` (page Kaggle).
- Le compteur `tXX` mesure-t-il le temps actif ou le temps réel ?
- Sens de `MAINT` et `DEV` entre parenthèses (mode, environnement ?) et des configurations métier (INSTALLATEUR, MT_AppelN1, dispatch).
- Seuil de plafonnement des rafales : 99e percentile à confirmer pendant la modélisation.
