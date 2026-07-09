"""Interface Retriever + deux stubs pour tester le harnais.

Le vrai retriever (embeddings + index) viendra plus tard. Il devra exposer
`search(query, k) -> list[Chunk]` et rien d'autre — le harnais n'a pas à savoir
comment il fonctionne.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Callable, Protocol

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ingest import Chunk  # noqa: E402


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[Chunk]: ...


# ---------------------------------------------------------------------------
# OracleRetriever — triche en lisant le gold. Sert UNIQUEMENT à vérifier que
# le scorer compte bien un hit. recall@1 doit valoir 1.0 sur des items valides.
# ---------------------------------------------------------------------------

class OracleRetriever:
    def __init__(
        self,
        gold,
        chunks: list[Chunk],
        is_match: Callable,
    ) -> None:
        # Pour chaque question, on précalcule les chunks qui contiennent le passage.
        self._matches_by_question: dict[str, list[Chunk]] = {}
        for g in gold:
            self._matches_by_question[g.question] = [c for c in chunks if is_match(g, c)]
        self._chunks = chunks

    def search(self, query: str, k: int) -> list[Chunk]:
        head = self._matches_by_question.get(query, [])
        # On complète avec des chunks "neutres" pour respecter le contrat de taille k.
        head_set = {id(c) for c in head}
        tail = [c for c in self._chunks if id(c) not in head_set]
        return (head + tail)[:k]


# ---------------------------------------------------------------------------
# RandomRetriever — baseline. Doit donner un recall bas (≈ k / |corpus|).
# ---------------------------------------------------------------------------

class RandomRetriever:
    def __init__(self, chunks: list[Chunk], seed: int = 0) -> None:
        self._chunks = chunks
        self._rng = random.Random(seed)

    def search(self, query: str, k: int) -> list[Chunk]:
        n = len(self._chunks)
        if n == 0:
            return []
        k = min(k, n)
        return self._rng.sample(self._chunks, k)


# ---------------------------------------------------------------------------
# DenseRetriever — embeddings e5 + cosinus in-memory. Premier vrai retriever.
# ---------------------------------------------------------------------------

class DenseRetriever:
    """Embeddings multilingual-e5-small + produit scalaire sur matrice normalisée.

    Convention e5 : préfixer `passage: ` à l'indexation et `query: ` à la
    recherche. La qualité chute nettement si on oublie.
    """

    def __init__(
        self,
        chunks: list[Chunk],
        model_name: str = "intfloat/multilingual-e5-small",
    ) -> None:
        # Imports locaux pour ne pas plomber le démarrage des autres retrievers.
        import numpy as np
        from sentence_transformers import SentenceTransformer

        self._chunks = list(chunks)
        self._model = SentenceTransformer(model_name)
        self._np = np

        if not self._chunks:
            self._matrix = np.zeros((0, 1), dtype=np.float32)
            return

        passages = [f"passage: {c.text}" for c in self._chunks]
        embs = self._model.encode(
            passages,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 → produit scalaire = cosinus
            show_progress_bar=False,
        )
        self._matrix = embs.astype(np.float32, copy=False)

    def search(self, query: str, k: int) -> list[Chunk]:
        if not self._chunks:
            return []
        q = self._model.encode(
            [f"query: {query}"],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        scores = self._matrix @ q  # (N,)
        k = min(k, len(self._chunks))
        # argpartition puis tri exact du top-k — évite un sort O(N log N).
        top = self._np.argpartition(-scores, k - 1)[:k]
        top = top[self._np.argsort(-scores[top])]
        return [self._chunks[i] for i in top]


# ---------------------------------------------------------------------------
# RerankRetriever — dense pour la présélection, cross-encoder pour le tri final.
# ---------------------------------------------------------------------------

class RerankRetriever:
    """2 étages : DenseRetriever renvoie top-N candidats, puis un cross-encoder
    (question, chunk) les réordonne. Le rerank remonte typiquement le bon chunk
    en tête ; il ne peut rien trouver hors du top-N dense."""

    def __init__(
        self,
        chunks: list[Chunk],
        dense_top_n: int = 20,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        dense: "DenseRetriever | None" = None,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self._dense = dense if dense is not None else DenseRetriever(chunks)
        self._dense_top_n = max(1, dense_top_n)
        self._reranker = CrossEncoder(model_name)

    def search(self, query: str, k: int) -> list[Chunk]:
        # Présélection dense.
        candidates = self._dense.search(query, k=self._dense_top_n)
        if not candidates:
            return []
        # Cross-encoder : paires BRUTES, pas de préfixes e5.
        pairs = [(query, c.text) for c in candidates]
        scores = self._reranker.predict(pairs, show_progress_bar=False)
        order = sorted(range(len(candidates)), key=lambda i: -float(scores[i]))
        top = order[: min(k, len(candidates))]
        return [candidates[i] for i in top]
