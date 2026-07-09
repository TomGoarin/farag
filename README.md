# farag — RAG minimal, sourcé, évalué

Un pipeline RAG (Retrieval-Augmented Generation) de bout en bout : ingestion de
PDF/DOCX/images, chunking, retrieval (dense + reranking), et génération de
réponses **sourcées uniquement sur le corpus**. Le projet est agnostique au
domaine.

Le corpus de développement est composé de polycopiés de télécoms (UMTS,
sécurité réseau, cryptographie) parce que **je maîtrise ce domaine** : je peux
juger la justesse des réponses sans m'auto-tromper. C'est une condition d'une
évaluation honnête, pas un choix marketing.

## Architecture

```
       data/raw/                          data/chunks/
    (PDF, DOCX, JPG, PNG)   ── ingestion ──▶ <doc_id>_chunks.jsonl
                                                    │
                                                    ▼
                                         load_chunks (corpus)
                                                    │
                       eval/gold/gold.jsonl         │
                       (paires Q/passage)           ▼
                                ┌─── harnais recall@k  (mesure)
                                │           │
                                │    ┌──────┴─ oracle | random | dense | rerank
                                │
                                └─── rag/generate.py   (usage)
                                             │
                             question ──▶ rerank ──▶ contexte + LLM ──▶ réponse + sources
```

**Flux de données**

1. Documents source déposés dans `data/raw/`.
2. `ingest_corpus.py` parse chacun (Docling), reconstruit un Markdown
   plein-document, découpe en chunks, et écrit `data/chunks/<doc_id>_chunks.jsonl`.
3. À la question : `RerankRetriever` présélectionne le top-20 via embeddings
   e5-small, puis un cross-encoder BGE réordonne, on garde les k premiers.
4. Le LLM reçoit ces k chunks numérotés + un prompt système strict, produit une
   réponse ou refuse. Le code (jamais le LLM) construit la liste des sources
   à partir des chunks retrievés.

**Layout**

```
farag/
├── ingest.py            # 1 fichier → JSONL
├── ingest_corpus.py     # batch data/raw → data/chunks
├── data/
│   ├── raw/             # sources
│   └── chunks/          # <doc_id>_chunks.jsonl
├── eval/
│   ├── harness.py       # recall@k + CLI
│   ├── retrievers.py    # Oracle / Random / Dense / Rerank
│   └── gold/
│       ├── gold.jsonl   # paires question/passage
│       └── README.md
└── rag/
    └── generate.py      # RAG génération + CLI
```

## Décisions de conception notables

### Markdown plein-document + offsets absolus

Docling produit une structure d'items ; on ne réutilise pas son Markdown natif,
on **le reconstruit nous-mêmes** en accumulant les items, ce qui garantit
l'invariant :

```
sourcedoc.markdown[b.char_start:b.char_end] == b.text  (pour chaque bloc / chunk)
```

Les offsets sont dans les coordonnées du **document entier**, jamais relatifs à
la page ou au bloc. Conséquence : un chunk sait précisément où il vit dans le
document, et cet ancrage survit à toute logique de chunking ultérieure. Un
assert le vérifie à chaque ingestion.

### Chunking block-native

Le découpage opère sur `doc.blocks` (accumulation jusqu'à un budget en
caractères, recouvrement en nombre de blocs), pas sur la chaîne brute du
Markdown. Objectif : **on ne coupe jamais au milieu d'une phrase, d'un titre ou
d'une puce**. Un jour on voudra un chunking structure-aware (par section, par
groupe sémantique) : ça se fera en changeant **le corps d'une seule fonction**,
`chunk()`. Rien en aval (retrieval, éval, génération) n'a besoin d'être touché.

### `doc_id` = hash du contenu

L'identité d'un document est le SHA-256 tronqué de son contenu, **pas** de son
chemin. Deux copies identiques → même id. Un fichier renommé/déplacé → même
id. Conséquences concrètes :

- Idempotence du batch : `ingest_corpus.py` skip proprement les documents déjà
  ingérés (`<doc_id>_chunks.jsonl` déjà présent).
- Les références dans le gold pointent vers un contenu, pas vers une URL de
  fichier fragile.
- Une modif du contenu → nouveau `doc_id` → réingestion automatique.

### Gold ancré sur le texte du passage

Un item gold ne stocke ni chunk_id ni offset : uniquement le **texte du
passage** attendu. À l'évaluation, on retrouve les chunks qui contiennent ce
passage. Résultat : le gold survit à un re-chunking, un re-parsing avec un
autre extractor, un changement de version de Docling. Il est ancré sur ce qui
est stable — le contenu — et pas sur un artéfact de pipeline.

### Matching d'éval : containment normalisé + égalité `doc_id`

Un chunk matche un item gold si :

```
normalize(gold.answer_passage_text) in normalize(chunk.text)  ET  chunk.doc_id == gold.doc_id
```

Le `normalize()` gère les cosmétiques (NFKC, casefold, apostrophes/guillemets
unifiés, emphase markdown, espaces autour de la ponctuation) — **pas de
fuzzy**, car masquer une vraie dégradation d'extraction dans l'éval reviendrait
à mentir sur la qualité du retrieval. La contrainte de `doc_id` évite les faux
positifs dans un corpus au vocabulaire commun (ex. le mot "clé" apparaît dans
les deux polys de sécurité).

### Swappabilité derrière une interface

Tout composant lourd est derrière une petite interface :

- **Parser** : `parse(path) -> SourceDoc`. Docling par défaut ; brancher
  PyMuPDF4LLM en fallback ne demande que de réimplémenter cette fonction.
- **Retriever** : `Protocol Retriever` avec `search(query, k) -> list[Chunk]`.
  Le harnais et la génération ne connaissent que ce contrat — d'où la
  cohabitation d'Oracle, Random, Dense, Rerank sans branchements conditionnels
  disséminés.
- **Chunking** : une seule fonction pure `chunk(doc, ...) -> list[Chunk]`,
  déjà mentionné.
- **Matching** : `is_match(gold_item, chunk) -> bool` isolé, remplaçable par un
  recouvrement de spans plus tard sans toucher au scoring.

## Évaluation

### Instrument avant mesure

Avant de mesurer un vrai retriever, il faut savoir si l'**instrument** de mesure
fonctionne. Deux retrievers factices bornent l'espace :

- `OracleRetriever` triche (lit le gold, place le bon chunk en tête). Sur des
  items valides, il doit donner recall@1 = 1.0. Sinon, c'est le scorer qui est
  cassé, pas le retriever.
- `RandomRetriever` tire k chunks au hasard. Il doit donner un recall proche de
  `k / |corpus|`. Il sert de plancher : n'importe quel vrai retriever doit
  faire mieux.

C'est un pattern d'ingénierie : ne pas ajuster un système tant qu'on n'est pas
sûr de l'instrument qui le mesure.

### Reranking : gain net sur recall@1

Le pipeline mesuré est `dense (e5-small) → rerank (bge-reranker-v2-m3)`. Le
reranking apporte **un gain substantiel sur recall@1** : le bon chunk est déjà
souvent dans le top-20 du dense, mais pas toujours en tête ; le cross-encoder
le remonte. Le recall@10 est mécaniquement plafonné par la présélection dense
(le reranker ne trouve rien de nouveau, il ne fait que réordonner).

### Discipline du gold

Le gold est enrichi de **questions relationnelles paraphrasées** ("dans quel
état RRC…", "quel principe garantit que…") plutôt que définitionnelles
recopiant le vocabulaire du passage. Objectif : éviter un banc trop facile, où
un simple appariement lexical suffirait — auquel cas un recall parfait ne
discriminerait plus rien. Le gold vaut la peine d'être écrit à la main pour ça.

Le harnais valide chaque item gold au chargement (`answer_passage_text`
trouvable dans son `doc_id`) et exclut proprement ceux qui ne le sont pas, avec
un rapport d'erreur — pas de plantage silencieux, pas de score gonflé par des
items faux.

## Utilisation

**Setup**

```bash
# installer les dépendances (cf. section Dépendances)
pip3 install docling==2.9.0 docling-core==2.19.1 \
             sentence-transformers==3.4.1 tf-keras \
             openai python-dotenv

# renseigner la clé dans .env à la racine
echo "OPENAI_API_KEY=sk-..." > .env
```

**Ingestion**

```bash
# 1 fichier
python3 ingest.py chemin/vers/doc.pdf

# Batch : traite tout data/raw, skip les documents déjà ingérés
python3 ingest_corpus.py
python3 ingest_corpus.py --force        # réingère même si présent
```

Sortie : `data/chunks/<doc_id>_chunks.jsonl`, un fichier par document.

**Évaluation**

```bash
python3 -m eval.harness --retriever oracle    # borne haute (validation instrument)
python3 -m eval.harness --retriever random    # borne basse
python3 -m eval.harness --retriever dense     # e5-small seul
python3 -m eval.harness --retriever rerank    # e5-small → bge-reranker
python3 -m eval.harness --retriever rerank --k 1,3,5,10
```

Rapport : validation gold, tableau `k | recall@k`, liste des items manqués au
plus grand `k`.
Premier jet : 

Result eval (dense) ==> 

  k  | recall@k
-----+---------
  1  | 0.478
  3  | 0.652
  5  | 0.761
  10 | 0.826
[missed @ k=10] 8 item(s) :
  - umts-sgsn-role: Dans le domaine paquet, quel équipement prend en charge les terminaux d'une même zone de routage, ainsi que leur taxation et leur authentification ?
  - umts-urnti-compose: Quel identifiant désigne un mobile de façon unique sur l'ensemble du réseau d'accès UMTS, et de quels deux éléments se compose-t-il ?
  - sec-bufferoverflow-def: Quelle faille permet d'écrire en dehors de la zone mémoire réservée, faute de contrôle sur la taille des entrées ?
  - prf-buffer-agrandir-limite: Quand les paquets arrivent plus vite qu'ils ne sont traités, pourquoi agrandir la file finit-il par ne plus rien apporter ?
  - prf-ns2-delai-transmission: Pourquoi le délai réel constaté entre deux nœuds ne se limite-t-il pas au temps de propagation qu'on a configuré ?
  - prf-mminfini-attente-nulle: Pourquoi n'y a-t-il aucune attente lorsqu'un système dispose d'un nombre illimité de serveurs ?
  - prf-mm2-vs-deux-files: Vaut-il mieux mutualiser une seule file pour deux serveurs ou dédier une file à chacun ?
  - td1-saut-frequence-fading: Comment le fait de combiner temps, fréquence et saut de fréquence aide-t-il à contrer le fast fading ?


Result eval (rerank) ==> 

  k  | recall@k
-----+---------
  1  | 0.739
  3  | 0.891
  5  | 0.935
  10 | 0.935

[missed @ k=10] 3 item(s) :
  - umts-sgsn-role: Dans le domaine paquet, quel équipement prend en charge les terminaux d'une même zone de routage, ainsi que leur taxation et leur authentification ?
  - prf-ns2-delai-transmission: Pourquoi le délai réel constaté entre deux nœuds ne se limite-t-il pas au temps de propagation qu'on a configuré ?
  - prf-mminfini-attente-nulle: Pourquoi n'y a-t-il aucune attente lorsqu'un système dispose d'un nombre illimité de serveurs ?

**Génération sourcée**

```bash
python3 -m rag.generate "Quel équipement en UMTS joue le rôle du BSC en GSM ?"
python3 -m rag.generate "Quelle est la capitale de l'Australie ?"   # → refus, pas de sources
```

Sortie : réponse rédigée + `Sources :` (doc_id, section, pages). Sur une
question hors corpus, la phrase de refus exacte est renvoyée sans sources.

## Limites & backlog assumés

- **Reranking lent sur CPU (~5,5 s/requête)**. Contrainte matérielle : macOS 13,
  pas de MPS (Metal), CPU seul. Ce n'est pas un défaut d'architecture — un
  cross-encoder est intrinsèquement quadratique en pairs (query, passage) et
  ne scale pas avec la taille du corpus (top-N fixe présélectionné par le
  dense). Un upgrade macOS 14+ activerait MPS et effacerait le problème.
- **Fidélité de la génération jugée à la main**. Le prompt est strict
  (interdiction d'utiliser des connaissances hors contexte, phrase de refus
  exacte), et le CLI ne recopie jamais les sources depuis le texte du LLM
  (elles viennent des chunks retrievés côté code). Mais il n'y a **pas d'éval
  automatique de hallucination**. Acceptable pour un MVP, à outiller ensuite
  (un juge LLM sur `(question, réponse, contexte)` par exemple).
- **Chunks parfois trop agrégés**. Le chunking block-native produit parfois des
  blocs mixant plusieurs concepts (un bloc UTRAN mêle RNC, NodeB et interfaces),
  ce qui dilue le signal du retriever sur une question ciblée. Un chunking
  structure-aware (par section ou par entité) le corrigerait. C'est
  exactement le cas de refactor prévu : la fonction `chunk()` est isolée, tout
  l'aval reste stable.
- **Stockage plat en JSONL**. Suffisant à l'échelle actuelle (dizaines de
  documents, centaines de chunks). Un vrai déploiement demanderait une base
  vectorielle : index ANN pour ne pas rescanner tout le corpus, persistance des
  embeddings, ingestion incrémentale, concurrence. Cette bascule ne touche que
  l'implémentation de `DenseRetriever`.
- **OCR = texte seulement**. EasyOCR récupère le texte des scans et images,
  mais rien de la sémantique des figures (schémas, diagrammes). Qualité
  variable selon la qualité du document source.

## Dépendances

Ingestion : `docling==2.9.0`, `docling-parse==2.1.2`, `docling-core==2.19.1`
(versions avec wheels pour macOS arm64 + Python 3.12).
Retrieval : `sentence-transformers==3.4.1`, `torch`, `tf-keras` (shim de
compatibilité pour `transformers`).
Génération : `openai>=2`, `python-dotenv`.

Modèles téléchargés au premier run (mis en cache dans `~/.cache/huggingface/`) :
`intfloat/multilingual-e5-small` (~470 MB), `BAAI/bge-reranker-v2-m3` (~2 GB),
plus les modèles EasyOCR si un scan ou une image est ingéré.
