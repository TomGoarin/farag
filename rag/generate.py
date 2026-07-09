"""RAG génération : question → retrieval (rerank) → réponse sourcée.

Réutilise le pipeline existant :
    - retrieval : `eval.retrievers.RerankRetriever` par défaut ;
    - corpus   : `data/chunks/*_chunks.jsonl` via `eval.harness.load_chunks`.

CLI :
    python -m rag.generate "ma question"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Charge .env à la racine du projet AVANT toute lecture d'env var.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ingest import Chunk  # noqa: E402


REFUSAL_SENTENCE = (
    "Le corpus ne contient pas d'information permettant de répondre à cette question."
)

SYSTEM_PROMPT = (
    "Tu réponds à une question en te basant EXCLUSIVEMENT sur les passages "
    "numérotés fournis dans le contexte. Tu ne dois utiliser AUCUNE connaissance "
    "extérieure à ce contexte.\n\n"
    "Si les passages ne contiennent pas de quoi répondre, réponds exactement "
    "cette phrase et rien d'autre :\n"
    f"\"{REFUSAL_SENTENCE}\"\n\n"
    "Sinon, rédige une réponse claire et concise en français, tirée uniquement "
    "des passages. N'invente aucune référence : la liste des sources est "
    "construite séparément par le code, pas par toi."
)


def _is_refusal(text: str) -> bool:
    """Comparaison robuste : strip, casefold, point final optionnel."""
    a = text.strip().rstrip(".").casefold()
    b = REFUSAL_SENTENCE.rstrip(".").casefold()
    return a == b


def _build_context(chunks: list[Chunk]) -> str:
    parts: list[str] = []
    for i, c in enumerate(chunks, start=1):
        section = " > ".join(c.section_path) if c.section_path else "(racine)"
        parts.append(
            f"[{i}] (doc_id={c.doc_id}, section={section})\n{c.text}"
        )
    return "\n\n".join(parts)


def _build_sources(chunks: list[Chunk]) -> list[dict]:
    """Dédup par (doc_id, section_path), ordre du retriever préservé."""
    seen: set[tuple[str, tuple[str, ...]]] = set()
    sources: list[dict] = []
    for c in chunks:
        key = (c.doc_id, c.section_path)
        if key in seen:
            continue
        seen.add(key)
        sources.append({
            "doc_id": c.doc_id,
            "source_file": Path(c.source_file).name,  # basename, jamais le chemin
            "section_path": list(c.section_path),
            "pages": list(c.pages),
        })
    return sources


def answer(
    question: str,
    retriever,
    k: int = 3,
    model: str = "gpt-4o-mini",
) -> dict:
    """Retrieval → prompt → génération. Retourne
    {answer, sources, chunks_used}. En cas d'erreur API, `answer` contient
    un message lisible et la fonction retourne quand même le dict."""
    chunks = retriever.search(question, k=k)
    context = _build_context(chunks)

    user_msg = (
        f"CONTEXTE\n--------\n{context}\n\n"
        f"QUESTION\n--------\n{question}"
    )

    try:
        from openai import OpenAI
        client = OpenAI()  # lit OPENAI_API_KEY
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
        )
        text = (resp.choices[0].message.content or "").strip()
    except Exception as e:  # noqa: BLE001 — on veut n'importe quelle erreur API
        text = f"[erreur API OpenAI] {type(e).__name__}: {e}"

    sources = [] if _is_refusal(text) else _build_sources(chunks)
    return {
        "answer": text,
        "sources": sources,
        "chunks_used": chunks,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _fmt_source(idx: int, src: dict) -> str:
    parts = [src.get("source_file") or src["doc_id"]]
    if src.get("section_path"):
        parts.append(" > ".join(src["section_path"]))
    if src.get("pages"):
        parts.append(f"p. {', '.join(str(p) for p in src['pages'])}")
    return f"  [{idx}] " + " · ".join(parts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="RAG: question → réponse sourcée.")
    ap.add_argument("question", help="Question à poser")
    ap.add_argument("--chunks", default="data/chunks", help="Dossier des chunks")
    ap.add_argument("--k", type=int, default=3, help="Nb de chunks à retrouver")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument(
        "--retriever",
        default="rerank",
        help="rerank|dense (défaut: rerank)",
    )
    args = ap.parse_args(argv)

    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY absente. Renseigne-la dans .env à la racine :\n"
            "    OPENAI_API_KEY=sk-...\n"
            "ou exporte-la dans le shell :\n"
            "    export OPENAI_API_KEY=sk-...",
            file=sys.stderr,
        )
        return 2

    from eval.harness import load_chunks
    from eval.retrievers import DenseRetriever, RerankRetriever

    chunks = load_chunks(args.chunks)
    if not chunks:
        print(f"Aucun chunk dans {args.chunks}", file=sys.stderr)
        return 2

    if args.retriever == "rerank":
        retriever = RerankRetriever(chunks=chunks)
    elif args.retriever == "dense":
        retriever = DenseRetriever(chunks=chunks)
    else:
        print(f"Retriever inconnu: {args.retriever}", file=sys.stderr)
        return 2

    result = answer(args.question, retriever, k=args.k, model=args.model)

    print(result["answer"])
    if _is_refusal(result["answer"]):
        return 0
    print("\n" + "-" * 60)
    if result["sources"]:
        print("Sources :")
        for i, src in enumerate(result["sources"], start=1):
            print(_fmt_source(i, src))
    else:
        print("Aucune source retrievée.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
