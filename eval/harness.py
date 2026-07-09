"""Harnais d'évaluation recall@k pour le retrieval d'un RAG.

Le retrieval réel n'existe pas encore : le harnais s'interface via
`retrievers.Retriever` et peut tourner contre des stubs (Oracle, Random).

Usage :
    python -m eval.harness --chunks data/chunks --gold eval/gold/gold.jsonl \
        --retriever oracle --k 1,3,5,10
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

# On réutilise Chunk + read_jsonl depuis ingest.py sans le modifier.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ingest import Chunk, read_jsonl  # noqa: E402


# ---------------------------------------------------------------------------
# Modèle gold
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GoldItem:
    id: str
    question: str
    doc_id: str
    answer_passage_text: str


# ---------------------------------------------------------------------------
# Normalisation + matching
# ---------------------------------------------------------------------------

_QUOTES = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "–": "-", "—": "-", "−": "-",  # en-dash, em-dash, minus
    " ": " ",  # NBSP
    " ": " ",  # narrow NBSP
})

# Apostrophes → ' (U+0027). Backtick U+0060 y est inclus : on le traite comme
# une apostrophe, il n'est donc PLUS supprimé par le strip d'emphase.
_APO_RE = re.compile(r"[‘’‚‛´`]")
# Guillemets doubles → "
_DQ_RE = re.compile(r"[“”„‟]")
# Guillemets français : on collapse l'espace collé côté contenu en un seul geste.
_FR_OPEN_RE = re.compile(r"«\s*")
_FR_CLOSE_RE = re.compile(r"\s*»")

# Emphase markdown *_ (backtick désormais géré par _APO_RE, donc retiré ici).
_EMPHASIS_RE = re.compile(r"[*_]+")

# Resserrage ponctuation.
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.,;:!?)\]}])")
_SPACE_AFTER_OPEN_RE = re.compile(r"([([{])\s+")

_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """NFKC + unification quotes/apostrophes + casefold + strip emphase markdown
    + resserrage des espaces autour de la ponctuation + collapse espaces.
    À appliquer des DEUX côtés avant comparaison."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    # Quotes/apostrophes (avant collapse : l'espace collé aux « » doit sauter ici).
    text = _FR_OPEN_RE.sub('"', text)
    text = _FR_CLOSE_RE.sub('"', text)
    text = _APO_RE.sub("'", text)
    text = _DQ_RE.sub('"', text)
    text = text.translate(_QUOTES)  # tirets, NBSP (existant)
    text = text.casefold()
    text = _EMPHASIS_RE.sub("", text)
    # "mot ." → "mot." ; "( mot" → "(mot"
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    text = _SPACE_AFTER_OPEN_RE.sub(r"\1", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def is_match(gold_item: GoldItem, chunk: Chunk) -> bool:
    """Containment normalisé. Logique isolée : remplacer ici pour passer à un
    recouvrement de spans plus tard, sans toucher au harnais."""
    return normalize(gold_item.answer_passage_text) in normalize(chunk.text)


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------

def load_chunks(directory: str | Path) -> list[Chunk]:
    directory = Path(directory)
    files = sorted(directory.glob("*_chunks.jsonl")) + sorted(
        directory.glob("*.chunks.jsonl")
    )
    files = sorted(set(files))
    out: list[Chunk] = []
    for f in files:
        out.extend(read_jsonl(f))
    return out


def load_gold(path: str | Path) -> list[GoldItem]:
    items: list[GoldItem] = []
    with Path(path).open(encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            d = json.loads(line)
            try:
                items.append(
                    GoldItem(
                        id=d["id"],
                        question=d["question"],
                        doc_id=d["doc_id"],
                        answer_passage_text=d["answer_passage_text"],
                    )
                )
            except KeyError as e:
                raise ValueError(f"Gold ligne {ln} : champ manquant {e}") from None
    return items


# ---------------------------------------------------------------------------
# Validation du gold
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GoldError:
    item: GoldItem
    reason: str


def validate_gold(
    gold: list[GoldItem], chunks: list[Chunk]
) -> tuple[list[GoldItem], list[GoldError]]:
    """Vérifie que chaque passage est trouvable dans au moins un chunk de son
    doc_id. Renvoie (items_valides, erreurs)."""
    by_doc: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_doc.setdefault(c.doc_id, []).append(c)

    valid: list[GoldItem] = []
    errors: list[GoldError] = []
    for g in gold:
        doc_chunks = by_doc.get(g.doc_id, [])
        if not doc_chunks:
            errors.append(GoldError(g, f"doc_id introuvable dans le corpus: {g.doc_id}"))
            continue
        if not any(is_match(g, c) for c in doc_chunks):
            errors.append(GoldError(g, "passage introuvable dans le doc (typo ? markdown ?)"))
            continue
        valid.append(g)
    return valid, errors


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def evaluate(
    retriever, gold: list[GoldItem], k_values: list[int]
) -> dict:
    if not gold:
        return {
            "recall": {k: 0.0 for k in k_values},
            "n_valid": 0,
            "missed_at_max": [],
        }
    K = max(k_values)
    hits = {k: 0 for k in k_values}
    missed: list[GoldItem] = []
    for g in gold:
        results = retriever.search(g.question, k=K)
        first_hit: int | None = None
        for i, c in enumerate(results):
            if is_match(g, c):
                first_hit = i
                break
        for k in k_values:
            if first_hit is not None and first_hit < k:
                hits[k] += 1
        if first_hit is None:
            missed.append(g)
    n = len(gold)
    return {
        "recall": {k: hits[k] / n for k in k_values},
        "n_valid": n,
        "missed_at_max": missed,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_retriever(name: str, chunks: list[Chunk], gold: list[GoldItem], seed: int):
    from eval.retrievers import DenseRetriever, OracleRetriever, RandomRetriever, RerankRetriever

    name = name.lower()
    if name == "oracle":
        return OracleRetriever(gold=gold, chunks=chunks, is_match=is_match)
    if name == "random":
        return RandomRetriever(chunks=chunks, seed=seed)
    if name == "dense":
        return DenseRetriever(chunks=chunks)
    if name == "rerank":
        return RerankRetriever(chunks=chunks)
    raise ValueError(f"Retriever inconnu: {name!r} (oracle|random|dense|rerank)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Harnais recall@k pour retrieval RAG.")
    ap.add_argument("--chunks", default="data/chunks", help="Dossier contenant les *_chunks.jsonl")
    ap.add_argument("--gold", default="eval/gold/gold.jsonl", help="Fichier gold JSONL")
    ap.add_argument("--retriever", default="oracle", help="oracle|random")
    ap.add_argument("--k", default="1,3,5,10", help="Liste de k séparés par virgule")
    ap.add_argument("--seed", type=int, default=0, help="Seed RNG pour RandomRetriever")
    args = ap.parse_args(argv)

    k_values = sorted({int(x) for x in args.k.split(",") if x.strip()})
    if not k_values:
        print("[k] aucune valeur fournie", file=sys.stderr)
        return 2

    chunks = load_chunks(args.chunks)
    gold = load_gold(args.gold)
    print(f"[corpus] {len(chunks)} chunks chargés depuis {args.chunks}")
    print(f"[gold]   {len(gold)} items chargés depuis {args.gold}")

    valid, errors = validate_gold(gold, chunks)
    if errors:
        print(f"[validation] {len(errors)} item(s) invalides (exclus du score) :")
        for e in errors:
            print(f"  - {e.item.id}: {e.reason}")
    print(f"[validation] {len(valid)} item(s) valides")

    if not valid:
        print("Rien à évaluer.", file=sys.stderr)
        return 1

    retriever = _build_retriever(args.retriever, chunks, valid, args.seed)
    print(f"[retriever] {args.retriever}")

    res = evaluate(retriever, valid, k_values)

    print("\n  k  | recall@k")
    print("-----+---------")
    for k in k_values:
        print(f"  {k:<3}| {res['recall'][k]:.3f}")

    missed = res["missed_at_max"]
    if missed:
        print(f"\n[missed @ k={max(k_values)}] {len(missed)} item(s) :")
        for g in missed:
            print(f"  - {g.id}: {g.question}")
    else:
        print(f"\n[missed @ k={max(k_values)}] aucun")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
