# Gold set — questions / passages de référence

Le gold est un fichier JSONL (une ligne par paire) qui sert d'oracle pour mesurer
le recall@k du retrieval. Chaque ligne :

```json
{"id": "slug-court", "question": "...", "doc_id": "hash_doc", "answer_passage_text": "passage exact tiré du document"}
```

## Règles

- **Passages courts et atomiques.** Idéalement une phrase ou une puce. Si le
  passage déborde sur plusieurs idées, scinder en plusieurs items gold.
- **Aucun span n'est stocké.** Le matching se fait par *containment* sur le
  texte normalisé (NFKC, casefold, ponctuation/markdown nettoyés). Donc copier
  le passage tel qu'il apparaît dans le doc suffit, sans se soucier des offsets.
- **Questions paraphrasées.** Formuler comme un étudiant qui n'a pas le cours
  sous les yeux : ne pas recopier le vocabulaire exact du passage. Si la
  question contient déjà les mots-clés rares du passage, le recall sera flatteur
  et ne mesurera plus la qualité du retrieval.
- **`doc_id`** = celui produit par `ingest.py` (hash du contenu, stable au
  déplacement du fichier). À récupérer dans n'importe quelle ligne du
  `*_chunks.jsonl` correspondant.
- **Volume cible** : ~20-30 paires pour un signal exploitable. Démarrer avec
  l'exemple seed et étoffer.

## Validation automatique

`python -m eval.harness …` vérifie au chargement que chaque
`answer_passage_text` est trouvable dans au moins un chunk de son `doc_id`. Les
items invalides sont listés et exclus du score (typo, mauvais doc_id, passage
mal recopié) — pas de plantage du run.

## Exemple seed

Voir [gold.jsonl](gold.jsonl). Le garder comme modèle de format.
