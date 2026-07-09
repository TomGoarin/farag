"""Backend FastAPI exposant le pipeline RAG.

Cette couche APPELLE l'existant (retrievers, generate), elle ne le réécrit pas.
Chunks chargés + retrievers instanciés une seule fois au démarrage.

Démarrage :
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Racine du projet sur sys.path pour importer ingest/eval/rag.
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

# Charge .env (OPENAI_API_KEY) avant tout.
try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

from eval.harness import load_chunks  # noqa: E402
from eval.retrievers import DenseRetriever, RerankRetriever  # noqa: E402
from rag.generate import answer  # noqa: E402


logger = logging.getLogger("api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")


# ---------------------------------------------------------------------------
# État global chargé au démarrage
# ---------------------------------------------------------------------------

class _State:
    chunks: list = []
    dense: DenseRetriever | None = None
    rerank: RerankRetriever | None = None


state = _State()


@asynccontextmanager
async def lifespan(app: FastAPI):
    chunks_dir = os.environ.get("FARAG_CHUNKS_DIR", "data/chunks")
    logger.info("Chargement chunks depuis %s ...", chunks_dir)
    state.chunks = load_chunks(chunks_dir)
    logger.info("%d chunks chargés", len(state.chunks))

    if not state.chunks:
        logger.warning("Corpus vide — /ask retournera 503 ou 422 selon la validation.")

    logger.info("Initialisation DenseRetriever (e5-small)...")
    state.dense = DenseRetriever(chunks=state.chunks)
    logger.info("Initialisation RerankRetriever (bge-reranker-v2-m3)...")
    state.rerank = RerankRetriever(chunks=state.chunks, dense=state.dense)
    logger.info("Retrievers prêts.")

    if not os.environ.get("OPENAI_API_KEY"):
        logger.warning(
            "OPENAI_API_KEY absente : le serveur démarre mais /ask renverra 503."
        )

    yield


app = FastAPI(title="farag RAG API", version="0.1.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# CORS (dev)
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    # Regex pour couvrir n'importe quel port localhost / 127.0.0.1 en dev.
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schémas
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    mode: Literal["dense", "rerank"]


class SourceOut(BaseModel):
    doc_id: str
    section_path: str
    pages: list[int]

    class Config:
        extra = "allow"  # accueille un futur chunk_text sans casser


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceOut]
    mode_used: Literal["dense", "rerank"]
    elapsed_ms: int


class HealthResponse(BaseModel):
    status: str
    chunks_loaded: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_API_ERROR_PREFIX = "[erreur API OpenAI]"


def _format_section(sp) -> str:
    if not sp:
        return "(racine)"
    return " > ".join(sp)


def _to_source_out(src: dict) -> SourceOut:
    return SourceOut(
        doc_id=src["doc_id"],
        section_path=_format_section(src.get("section_path", [])),
        pages=list(src.get("pages", [])),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", chunks_loaded=len(state.chunks))


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not state.chunks:
        raise HTTPException(status_code=503, detail={"error": "corpus vide côté serveur"})

    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail={"error": "OPENAI_API_KEY absente côté serveur"},
        )

    retriever = state.dense if req.mode == "dense" else state.rerank
    if retriever is None:
        raise HTTPException(status_code=503, detail={"error": "retriever non initialisé"})

    t0 = time.perf_counter()
    # k = nombre de chunks fournis à la génération ET listés comme sources
    # (invariant : contexte du LLM == sources affichées). Distinct du k du
    # harnais recall@k, qui reste plus large côté eval.
    result = answer(req.question, retriever, k=3)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    # `answer()` avale les erreurs OpenAI et injecte un message préfixé.
    # On les propage en 503 pour que le front puisse afficher proprement.
    if result["answer"].startswith(_API_ERROR_PREFIX):
        raise HTTPException(
            status_code=503,
            detail={"error": result["answer"][len(_API_ERROR_PREFIX):].strip()},
        )

    return AskResponse(
        answer=result["answer"],
        sources=[_to_source_out(s) for s in result["sources"]],
        mode_used=req.mode,
        elapsed_ms=elapsed_ms,
    )
