# Insula MNI152 Plotter — V13 (filled markers)

Projet Python de visualisation de contacts iEEG sur une surface insulaire standard **Julich-Brain / fsLR 32k**. Il prend en entrée des coordonnées supposées en MNI152 (mm, convention RAS+) et produit des figures 3D, bilatérales, des vues séparées, des flatmaps expérimentales, des figures avec `t_value` et des fichiers de contrôle qualité.

> **Important — interprétation anatomique :** il s'agit d'une projection sur un atlas/surface standard, pas d'une reconstruction corticale individuelle du patient. Les coordonnées MNI d'origine sont conservées. Lorsqu'un groupe dorsal/ventral manuel est fourni, le snap vers une région Julich de ce groupe sert à rendre la visualisation cohérente avec cette sélection manuelle ; il ne constitue pas une correction anatomique indépendante de la coordonnée.

## 1. Chaîne de traitement, de l'entrée aux figures

1. **Lecture des contacts** — `insula_plotter/io.py` lit le CSV, vérifie les colonnes et les coordonnées numériques, déduit l'hémisphère du signe de `x` (`x < 0` gauche, `x > 0` droite) et conserve les champs facultatifs comme `t_value`, le statut et l'annotation manuelle dorsal/ventral.
2. **Téléchargement des cartes Julich-Brain** — `insula_plotter/atlas.py` utilise `siibra` pour interroger la parcellation `julich 3.1` dans l'espace `mni152`, récupérer les cartes statistiques/probabilistes des régions insulaires demandées et les charger comme images NIfTI avec `nibabel`.
3. **Construction d'un volume de labels** — les cartes probabilistes des régions sont comparées voxel par voxel. Chaque voxel reçoit le label de la région dont la probabilité est la plus élevée, à condition qu'elle dépasse `atlas.probability_threshold`. Ce volume sert notamment à évaluer l'atlas au niveau des coordonnées MNI originales.
4. **Projection des probabilités sur la surface** — `insula_plotter/projection.py` projette chaque carte probabiliste MNI152 séparément vers fsLR avec `neuromaps.transforms.mni152_to_fslr(..., method="linear")`. Les valeurs interpolées sont ensuite comparées sommet par sommet (winner-takes-all) et seuillées avec `surface_probability_threshold`. Cette approche évite de projeter directement une image catégorielle en nearest-neighbour pour construire les frontières des parcelles.
5. **Projection des contacts** — une petite sphère autour de chaque coordonnée MNI est créée dans la grille NIfTI de référence, puis projetée vers fsLR en nearest-neighbour. Plusieurs rayons de secours sont essayés. Si le CSV contient `insula_dv_group`, le code peut ensuite rechercher le sommet de surface Julich le plus proche appartenant au groupe dorsal/ventral et à l'hémisphère attendus. Les coordonnées MNI et les distances/méthodes de projection sont conservées dans les tableaux de sortie.
6. **Préparation des maillages et rendu** — `plotting.py` charge les surfaces fsLR gonflées, extrait les triangles, dessine le cortex de contexte, les parcelles, leurs frontières et les contacts. Les contacts partageant une position sont décalés uniquement pour l'affichage, afin de les distinguer.
7. **Sorties de contrôle** — les coordonnées de surface, la méthode de projection, le groupe dorsal/ventral, les contacts superposés et la visibilité sont exportés en CSV pour permettre une vérification site par site.

## 2. D'où viennent les surfaces cérébrales ?

Le projet ne reconstruit pas lui-même une surface à partir d'une IRM individuelle et ne nécessite pas de fichiers FreeSurfer locaux pour son rendu standard. Il récupère le maillage standard **fsLR** via :

```python
from neuromaps.datasets import fetch_atlas
atlas = fetch_atlas("fsLR", "32k")
inflated = atlas["inflated"]
```

Cette fonction `neuromaps` télécharge/récupère dans son cache local les fichiers de l'atlas fsLR. Le code charge ensuite, pour chaque hémisphère, le fichier GIFTI de surface gonflée (`inflated`) avec `nibabel` et extrait :

- les coordonnées 3D des sommets (`NIFTI_INTENT_POINTSET`) ;
- les indices des triangles (`NIFTI_INTENT_TRIANGLE`).

La surface gonflée facilite la visualisation du cortex enfoui dans les sillons, mais reste un maillage standard fsLR : ce n'est pas la surface anatomique propre à chaque patient.

### D'où viennent les régions Julich ?

`insula_plotter/atlas.py` appelle l'API Python `siibra` :

```python
maps = siibra.get_map(
    space="mni152",
    parcellation="julich 3.1",
    maptype="statistical",
)
```

Le programme identifie les régions insulaires (Ig, Ia, Id) et les deux hémisphères, puis télécharge chaque carte statistique/probabiliste avec `maps.fetch(region=region)`. Les cartes sont des volumes NIfTI dans l'espace MNI152. Elles ne sont pas des labels binaires parfaitement délimités : elles représentent des probabilités/valeurs statistiques de localisation cytoarchitectonique.

Le programme transforme ces cartes en labels de surface fsLR. Les couleurs et noms des régions proviennent des métadonnées/palettes internes du projet ; les positions des sommets et la topologie des triangles proviennent des fichiers de surface fsLR.

### Cache

Au premier lancement, les ressources doivent être accessibles en ligne. `neuromaps` et `siibra` gèrent leurs caches de téléchargement. En plus, le projet enregistre sa propre projection des labels Julich fsLR dans :

```text
outputs/julich_probability_wta_fslr_32k.npz
```

Le cache contient les labels gauche/droite et des métadonnées (parcellation, espace, densité, seuil). Il est réutilisé si ces paramètres correspondent à la configuration actuelle. Supprimer ce fichier force le recalcul de la projection des cartes probabilistes.

## 3. Librairies principales et rôle de chacune

| Librairie | Rôle dans le projet |
|---|---|
| **NumPy** | Tableaux numériques, calculs sur les sommets/triangles, traitement des cartes et géométrie. |
| **Pandas** | Lecture/écriture CSV et manipulation des tableaux de contacts et rapports. |
| **NiBabel** | Lecture des images NIfTI (cartes Julich, volumes de marqueurs) et des surfaces GIFTI fsLR. |
| **NiLearn** | Pas utilisé directement par le code actuel. |
| **neuromaps** | Récupération de l'atlas de surfaces fsLR et transformation des données MNI152 vers fsLR. |
| **siibra** | Accès à l'atlas Julich-Brain et téléchargement des cartes statistiques/probabilistes des régions. |
| **SciPy** | Structures spatiales `cKDTree`, recherche du sommet le plus proche et opérations numériques/géométriques. |
| **Matplotlib** | Rendu des figures 3D/2D, couleurs, légendes et colorbars. |
| **PyYAML** | Lecture de `config.yaml`. |
| **pytest** | Exécution des tests automatisés (développement/validation, pas requis pour générer les figures). |

Dépendances Python déclarées dans `requirements.txt` et `pyproject.toml` :

```text
numpy, pandas, nibabel, matplotlib, scipy, neuromaps, siibra, PyYAML
```

`pytest` est nécessaire uniquement pour lancer les tests. `Connectome Workbench` n'est pas nécessaire au calcul ou aux PNG ; il est utile si l'on souhaite ouvrir les exports GIFTI dans `wb_view`. FreeSurfer n'est pas appelé par le pipeline Python actuel.

## 4. Organisation du projet

```text
insula_mni152_plotter_v13/
├── config.yaml                  # paramètres d'entrée, atlas, projection et figures
├── requirements.txt             # dépendances Python
├── pyproject.toml               # installation du package et commande insula-plot
├── data/
│   ├── sites_user.csv            # CSV actif des contacts
│   ├── sites_example.csv         # exemple minimal
│   └── sites_surface_coordinates_v12_reference.csv
├── insula_plotter/
│   ├── io.py                     # validation et lecture du CSV
│   ├── atlas.py                  # téléchargement/cartes Julich et volume de labels
│   ├── assignment.py             # attribution de région à la coordonnée MNI initiale
│   ├── projection.py             # transformations MNI152 -> fsLR et projection des contacts
│   ├── plotting.py               # surfaces, contacts, légendes, figures et exports Workbench
│   ├── flatmap.py                # dépliage 2D expérimental du patch insulaire
│   ├── config.py                 # utilitaires de configuration
│   └── cli.py                    # orchestration du pipeline
├── scripts/
│   ├── run_plot.py               # lanceur
│   ├── list_atlas_regions.py     # inspection des régions exposées par siibra
│   └── make_coordinate_qc.py     # contrôle visuel des coordonnées/volumes
├── tests/                        # tests unitaires et de non-régression
└── outputs/                      # créé à l'exécution
```

## 5. Format du CSV d'entrée

Le fichier actif est `data/sites_user.csv`. Les noms de colonnes sont configurables dans la section `input` de `config.yaml`.

Colonnes principales :

| Colonne | Description |
|---|---|
| `site_id` | Identifiant unique du contact (ex. S01). |
| `label` | Nom descriptif du contact/patient. |
| `x`, `y`, `z` | Coordonnées MNI152 en millimètres, supposées RAS+. |
| `color` | Couleur du contact au format Matplotlib/hexadécimal. |
| `t_value` | Valeur continue utilisée pour les figures avec colorbar, si présente. |
| `insula_area_for_dv` | Région annotée/choisie pour guider la classification D/V. |
| `insula_dv_group` | Annotation manuelle `Dorsale` ou `Ventrale`, utilisée comme cible de snap. |

Le signe de `x` détermine l'hémisphère : négatif = gauche, positif = droite. Cette hypothèse doit être vérifiée si les coordonnées sont dans une convention différente (par exemple LPS) ou si elles ne sont pas réellement en MNI152 millimètres.

## 6. Installation et exécution sous Windows / Conda

Dans PowerShell, depuis le dossier du projet :

```powershell
conda activate insula-plot
python -m pip install -e . --no-deps --upgrade
python scripts\run_plot.py --config config.yaml
```

L'option `--no-deps` évite de remplacer les versions de NumPy/SciPy déjà présentes dans l'environnement. Si l'environnement ne possède pas encore les dépendances listées, installer `requirements.txt` dans un environnement dédié est préférable :

```powershell
python -m pip install -r requirements.txt
```

Le premier lancement nécessite une connexion Internet pour récupérer les ressources Julich et fsLR. Les exécutions suivantes réutilisent les caches quand ils sont disponibles.

La commande installée par `pyproject.toml` peut aussi être utilisée :

```powershell
insula-plot --config config.yaml
```

## 7. Paramètres importants dans `config.yaml`

- `input.csv` : chemin du CSV d'entrée.
- `atlas.parcellation`, `atlas.space` : parcellation et espace demandés à siibra.
- `atlas.region_codes` : régions insulaires demandées.
- `atlas.probability_threshold` : seuil pour le volume Julich winner-takes-all.
- `atlas.surface_probability_threshold` : seuil après projection linéaire des cartes probabilistes sur fsLR.
- `surface.density` : densité fsLR (`32k` dans la configuration fournie).
- `surface.marker_radius_mm` et les listes de rayons de secours : taille des sphères MNI servant à faire apparaître les contacts sur la surface.
- `surface.snap_to_insula` : active le snap vers les sommets étiquetés de l'insula.
- `plot.marker_size`, `plot.context_alpha`, `plot.parcel_alpha` : apparence des contacts et du cortex de contexte.
- `plot.overlap_offset_mm` : séparation graphique des contacts superposés.
- `plot.dorsal_regions`, `plot.ventral_regions` : définition des groupes D/V pour les figures.
- `plot.save_flatmaps` (si présent) : génération des vues aplaties.

Les paramètres exacts disponibles et leurs valeurs par défaut sont documentés directement dans `config.yaml` et dans les fonctions qui les lisent.

## 8. Fichiers de sortie et contrôle qualité

Les principaux résultats sont écrits dans `outputs/` :

- `insula_sites_fslr32k.png` et vues associées : rendu 3D sur fsLR.
- figures bilatérales, dorsal/ventral, vues de lissage local et figures `t_value` : selon les options activées.
- `sites_surface_coordinates.csv` : une ligne par contact avec coordonnées d'origine, hémisphère, sommet fsLR, coordonnées de surface, région/groupe attribué, méthode et distance de snap.
- `julich_insula_labels_fslr_32k.npz` : labels Julich projetés sur les sommets fsLR.
- `julich_probability_wta_fslr_32k.npz` : cache de la projection des cartes probabilistes.
- `site_display_positions.csv` : positions d'affichage après séparation des points superposés, si généré.
- `overlapping_sites_report.csv`, `site_visibility_report.csv`, `site_counts_dorsal_ventral.csv` : rapports de contrôle.
- `workbench/` : exports GIFTI si `plot.export_workbench` est activé.

**Toujours inspecter les rapports de projection.** Un contact déplacé de plusieurs millimètres vers une parcelle Julich par snap manuel peut être visible sur la figure, mais sa position affichée ne doit pas être interprétée comme une localisation anatomique mesurée à cette nouvelle position. Le déplacement graphique des contacts superposés est lui aussi uniquement cosmétique.

## 9. Flatmap et déformation locale

`flatmap.py` calcule une représentation 2D du patch insulaire à partir de sa connectivité triangulaire (paramétrisation harmonique). Elle est conservée comme vue complémentaire, mais elle introduit nécessairement des distorsions géométriques.

La vue localement aplatie de la surface 3D, lorsqu'elle est activée, est également une déformation de visualisation. Elle ne modifie ni les coordonnées MNI originales, ni les identifiants des sommets fsLR enregistrés dans les fichiers de projection.

## 10. Tests

Pour exécuter les tests :

```powershell
python -m pip install pytest
python -m pytest -q
```

Les tests vérifient des fonctions et cas synthétiques (projection, aplatissement, affichage, collisions, etc.). Ils ne remplacent pas une validation anatomique des ressources téléchargées, des coordonnées MNI ni des choix de snap pour chaque contact.

## 11. Limites méthodologiques à garder en tête

- L'espace d'entrée est **supposé** MNI152 millimètres RAS+ ; le code ne peut pas garantir que les coordonnées sources utilisent exactement cette variante/template.
- La surface est un fsLR standard 32k, non une surface reconstruite à partir des IRM individuelles.
- Les cartes Julich sont probabilistes/statistiques ; les frontières affichées dépendent du seuil, de l'interpolation et de l'assignation winner-takes-all.
- Le snap dorsal/ventral guidé par une annotation manuelle privilégie la cohérence avec la sélection fournie, mais peut éloigner un contact de sa coordonnée MNI. Examiner `surface_snap_distance_mm`.
- Le déplacement des marqueurs superposés ne représente pas une distance anatomique réelle.
- Les surfaces gonflées et les aplatissements facilitent la lecture des régions enfouies mais déforment les distances et les formes.
- Cette chaîne de visualisation n'est pas identique à une chaîne individuelle CT/MRI + FreeSurfer + recalage de surface telle qu'on peut la trouver dans des études d'électrophysiologie intracrânienne.

## Références logicielles

- [siibra — atlas et espaces cérébraux](https://siibra-python.readthedocs.io/)
- [Julich-Brain Atlas](https://julich-brain-atlas.de/)
- [neuromaps — transformations et ressources de surface](https://neuromaps.readthedocs.io/)
- [fsLR / Connectome Workbench](https://www.humanconnectome.org/software/connectome-workbench)
- [NiBabel](https://nipy.org/nibabel/)
