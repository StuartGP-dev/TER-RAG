# TER-RAG — Évaluation de pipelines RAG francophones

## Présentation

Ce dépôt contient le prototype expérimental du projet **TER-RAG**.

L’objectif du projet est de construire et d’évaluer plusieurs approches de **Retrieval-Augmented Generation** en français. Le cas d’usage visé est un assistant conversationnel capable de répondre à des questions à partir d’un corpus documentaire lié à TALN.

Le projet ne sert donc pas seulement à tester un chatbot. Il sert surtout à comparer plusieurs stratégies de recherche d’information et de génération de réponses sur un même corpus et un même jeu de questions.

## Objectif du projet

Le pipeline suit les grandes étapes suivantes.

1. Charger un corpus documentaire TALN
2. Découper les documents en passages exploitables par les modèles
3. Indexer les passages avec différentes méthodes de retrieval
4. Récupérer les contextes les plus pertinents pour chaque question
5. Générer une réponse avec un modèle de langue
6. Évaluer les méthodes avec des métriques de retrieval, de génération, de temps et d’impact carbone

## Données utilisées

Le corpus principal est stocké dans le fichier suivant.

```text
data/documents_all.jsonl
```

Ce fichier correspond au corpus documentaire utilisé par le projet. Il a été généré à partir du dépôt suivant.

```text
https://github.com/StuartGP-dev/TER-RAG
```

Le jeu de questions utilisé pour l’évaluation se trouve dans le dossier `questions/`.

```text
questions/questions_taln2025.json
```

Chaque question contient notamment la question posée, la réponse attendue et le document attendu. Le document attendu est utilisé pour calculer les scores de retrieval comme le `Recall@1`, le `Recall@3` et le `Recall@5`.

## Méthodes comparées

Le projet compare plusieurs approches de retrieval.

- **TF-IDF**
  Approche lexicale simple utilisée comme baseline

- **BM25**
  Méthode lexicale plus robuste, souvent utilisée comme référence en recherche d’information

- **DPR**
  Méthode dense basée sur des encodeurs CamemBERT pour représenter les questions et les documents

- **HyDE**
  Méthode qui génère d’abord une réponse hypothétique, puis utilise cette réponse pour améliorer la recherche sémantique

Ces méthodes sont ensuite utilisées dans un pipeline RAG complet afin de comparer la qualité des contextes récupérés et des réponses générées.

## Fonctionnement général

Le projet commence par charger le corpus depuis `data/documents_all.jsonl`. Les documents sont ensuite découpés en chunks afin d’obtenir des passages plus courts et plus adaptés aux modèles utilisés.

Les différents retrievers indexent ensuite ces chunks. Pour chaque question, ils retournent les passages jugés les plus pertinents. Ces passages sont transmis au modèle génératif, qui doit produire une réponse courte en français en se basant uniquement sur le contexte fourni.

L’évaluation permet ensuite de comparer les méthodes selon plusieurs critères.

- La capacité à retrouver le bon document
- La qualité de la réponse générée
- La fidélité de la réponse au contexte
- Le temps moyen par question
- L’impact carbone mesuré avec CodeCarbon

## Arborescence du dépôt

```text
.
├── Benchmarks.ipynb
├── chat.ipynb
├── statistiques_corpus.ipynb
├── data/
│   └── documents_all.jsonl
├── questions/
│   └── questions_taln2025.json
└── src/
    ├── __init__.py
    ├── bm25.py
    ├── chunker.py
    ├── config.py
    ├── dpr.py
    ├── hyde.py
    └── tf_idf.py
```

## Rôle des principaux fichiers

### `src/config.py`

Ce fichier centralise les chemins, les modèles et les paramètres principaux du projet. Il définit notamment le chemin du corpus, le modèle génératif, les modèles d’embeddings, les modèles DPR, le nombre de documents retournés et le périphérique utilisé.

### `src/chunker.py`

Ce fichier s’occupe du découpage du corpus en chunks. Le but est de transformer des documents parfois longs en passages plus courts, mieux adaptés au retrieval et à la génération.

### `src/tf_idf.py`

Ce fichier implémente le retriever TF-IDF.

### `src/bm25.py`

Ce fichier implémente le retriever BM25 avec LangChain.

### `src/dpr.py`

Ce fichier implémente le retriever dense DPR avec deux encodeurs, un pour les questions et un pour les documents.

### `src/hyde.py`

Ce fichier implémente l’approche HyDE. Elle repose sur une génération intermédiaire, puis sur une recherche sémantique à partir de cette génération.

## Notebooks disponibles

### `Benchmarks.ipynb`

Notebook principal du projet. Il permet de lancer les expériences, de comparer les méthodes et de produire les résultats d’évaluation.

### `chat.ipynb`

Notebook utilisé pour tester le pipeline sous forme de chatbot interactif.

### `statistiques_corpus.ipynb`

Notebook utilisé pour analyser le corpus et produire des statistiques descriptives.

## Installation

Créer un environnement virtuel.

```bash
python -m venv .venv
source .venv/bin/activate
```

Sous Windows PowerShell.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Installer les dépendances avec le fichier `requirements.txt`.

```bash
pip install -r requirements.txt
```

Selon la machine utilisée, il peut être nécessaire d’installer une version de PyTorch compatible avec la version CUDA disponible.

## Utilisation

L’utilisation principale du projet passe par les notebooks.

1. Vérifier que le corpus est présent dans `data/documents_all.jsonl`
2. Vérifier que le jeu de questions est présent dans `questions/questions_taln2025.json`
3. Lancer `Benchmarks.ipynb` pour exécuter les expériences
4. Utiliser `chat.ipynb` pour tester le chatbot
5. Utiliser `statistiques_corpus.ipynb` pour analyser le corpus

Les résultats expérimentaux peuvent générer des fichiers dans les dossiers suivants.

```text
debug/
emissions/
outputs/
```

## Métriques utilisées

| Type | Métrique | Rôle |
|---|---|---|
| Retrieval | `Recall@1` | Vérifie si le bon document est le premier résultat récupéré |
| Retrieval | `Recall@3` | Vérifie si le bon document est présent dans les trois premiers résultats |
| Retrieval | `Recall@5` | Vérifie si le bon document est présent dans les cinq premiers résultats |
| Génération | `faithfulness` | Vérifie si la réponse est fidèle au contexte |
| Génération | `relevancy` | Vérifie si la réponse répond correctement à la question |
| Génération | `context_usage` | Vérifie si le contexte récupéré est bien utilisé |
| Robustesse | `parse_rate` | Vérifie si les sorties du juge sont correctement parsées |
| Performance | temps par question | Mesure le coût temporel moyen |
| Impact carbone | CO2 par question | Mesure les émissions moyennes avec CodeCarbon |

## Points d’attention

Le projet peut demander beaucoup de ressources, surtout lors de l’utilisation de DPR et HyDE. Ces méthodes chargent des modèles plus lourds et peuvent consommer beaucoup de VRAM.

Le modèle génératif est configuré pour être utilisé sur GPU. Une exécution sur CPU est possible en théorie, mais elle serait beaucoup plus lente et demanderait probablement des adaptations.

Certains chemins sont liés à l’environnement de travail utilisé pendant le projet. En particulier, le chemin du modèle génératif peut devoir être modifié si le dépôt est lancé sur une autre machine.

Si une erreur de mémoire GPU apparaît, il faut généralement réduire la taille des batchs, libérer la mémoire GPU ou lancer les méthodes une par une.

## Auteurs

Khalil MHEDHBI \\
Yanis DABIN \\
Marius MABULU \\
Helio BARTHES
