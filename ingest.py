"""RAG ingestion: parse -> markdown plein-document -> chunks block-native -> JSONL.

Dépendances: `pip install docling`
Usage: `python ingest.py <fichier> [--out out.jsonl] [--target 1200] [--overlap 1]`
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".gif"}


def _ensure_ssl_certs() -> None:
    """Python.org sur macOS n'a pas de CA root → tous les téléchargements HTTPS
    (modèles EasyOCR, etc.) échouent. On force le bundle certifi."""
    if os.environ.get("SSL_CERT_FILE"):
        return
    try:
        import certifi
        os.environ["SSL_CERT_FILE"] = certifi.where()
        os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Block:
    page: int
    section_path: tuple[str, ...]
    char_start: int
    char_end: int
    text: str


@dataclass
class SourceDoc:
    doc_id: str
    source_file: str
    extractor: str
    markdown: str
    blocks: list[Block]
    page_spans: list[tuple[int, int, int]]


@dataclass
class Chunk:
    doc_id: str
    source_file: str
    pages: list[int]
    section_path: tuple[str, ...]
    char_start: int
    char_end: int
    text: str
    extractor: str


# ---------------------------------------------------------------------------
# Parser (Docling) — derrière une fonction simple à substituer
# ---------------------------------------------------------------------------

SEP = "\n\n"


def compute_doc_id(path: str | Path) -> str:
    """Hash stable du contenu — indépendant du chemin (un fichier déplacé garde
    le même id, deux copies identiques partagent leur id)."""
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()[:16]


def parse(path: str | Path, do_ocr: bool | None = None) -> SourceDoc:
    """Parse un document via Docling et retourne un SourceDoc.

    - PDF texte natif    : do_ocr=False (rapide).
    - Image / PDF scanné : do_ocr=True (téléchargement EasyOCR la première fois).
    - do_ocr=None (défaut) : auto. Images → OCR on. PDF/DOCX → essai sans OCR,
      puis re-essai avec OCR si rien n'a été extrait.

    Invariant garanti : pour chaque Block b,
        sourcedoc.markdown[b.char_start:b.char_end] == b.text
    """
    path = Path(path)
    if do_ocr is None:
        is_image = path.suffix.lower() in IMAGE_EXTS
        attempts = [True] if is_image else [False, True]
    else:
        attempts = [bool(do_ocr)]

    sd: SourceDoc | None = None
    for ocr_on in attempts:
        if ocr_on:
            _ensure_ssl_certs()
        sd = _parse_once(path, do_ocr=ocr_on)
        if sd.blocks:
            return sd
    assert sd is not None
    return sd


def _parse_once(path: Path, do_ocr: bool) -> SourceDoc:
    from importlib.metadata import version as _pkg_version, PackageNotFoundError
    from docling.document_converter import DocumentConverter, PdfFormatOption

    def _pkg_ver(name: str) -> str:
        try:
            return _pkg_version(name)
        except PackageNotFoundError:
            return "unknown"

    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling_core.types.doc.labels import DocItemLabel

    pdf_opts = PdfPipelineOptions(do_ocr=do_ocr)
    fmt_opts = {InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)}
    # Docling 2.9 traite les images via le même pipeline que les PDF ;
    # on lui passe les mêmes options OCR.
    img_fmt = getattr(InputFormat, "IMAGE", None)
    if img_fmt is not None:
        fmt_opts[img_fmt] = PdfFormatOption(pipeline_options=pdf_opts)
    converter = DocumentConverter(format_options=fmt_opts)
    result = converter.convert(str(path))
    ddoc = result.document

    blocks: list[Block] = []
    md_parts: list[str] = []
    cursor = 0
    section_stack: list[tuple[int, str]] = []  # (heading_level, text)

    def _page_of(item) -> int:
        prov = getattr(item, "prov", None)
        if prov:
            return getattr(prov[0], "page_no", 1) or 1
        return 1

    for item, _tree_level in ddoc.iterate_items():
        label = getattr(item, "label", None)
        md_text: str
        emit_as_block = True

        if label in (DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE):
            heading_level = getattr(item, "level", None) or 1
            text = (getattr(item, "text", "") or "").strip()
            if not text:
                continue
            md_text = "#" * max(1, min(heading_level, 6)) + " " + text
            # met à jour la pile de sections (pop jusqu'au niveau courant)
            section_stack = [(lvl, t) for lvl, t in section_stack if lvl < heading_level]
            section_stack.append((heading_level, text))
        elif label == DocItemLabel.LIST_ITEM:
            text = (getattr(item, "text", "") or "").strip()
            if not text:
                continue
            md_text = "- " + text
        elif label == DocItemLabel.TABLE:
            # Docling sait exporter le tableau en markdown
            try:
                md_text = item.export_to_markdown(ddoc).strip()
            except Exception:
                md_text = ""
            if not md_text:
                continue
        elif label == DocItemLabel.CODE:
            text = getattr(item, "text", "") or ""
            md_text = "```\n" + text + "\n```"
        else:
            text = (getattr(item, "text", "") or "").strip()
            if not text:
                continue
            md_text = text

        if not emit_as_block:
            continue

        if md_parts:
            md_parts.append(SEP)
            cursor += len(SEP)

        char_start = cursor
        md_parts.append(md_text)
        cursor += len(md_text)
        char_end = cursor

        blocks.append(
            Block(
                page=_page_of(item),
                section_path=tuple(t for _, t in section_stack),
                char_start=char_start,
                char_end=char_end,
                text=md_text,
            )
        )

    markdown = "".join(md_parts)

    # page_spans : (page, span_start, span_end) sur le markdown global
    by_page: dict[int, list[int]] = {}
    for b in blocks:
        if b.page not in by_page:
            by_page[b.page] = [b.char_start, b.char_end]
        else:
            by_page[b.page][1] = max(by_page[b.page][1], b.char_end)
    page_spans = [(p, s, e) for p, (s, e) in sorted(by_page.items())]

    return SourceDoc(
        doc_id=compute_doc_id(path),
        source_file=str(path),
        extractor=f"docling=={_pkg_ver('docling')}",
        markdown=markdown,
        blocks=blocks,
        page_spans=page_spans,
    )


# ---------------------------------------------------------------------------
# Chunking — fenêtré simple, BLOCK-NATIVE
# ---------------------------------------------------------------------------

def chunk(doc: SourceDoc, target_chars: int = 1200, overlap_blocks: int = 1) -> list[Chunk]:
    """Accumule des blocs jusqu'à `target_chars`, recouvrement en blocs.

    Tout l'aval ne voit que list[Chunk]. Pour passer en structure-aware plus
    tard : ne change que cette fonction.
    """
    chunks: list[Chunk] = []
    blocks = doc.blocks
    n = len(blocks)
    if n == 0:
        return chunks

    overlap_blocks = max(0, overlap_blocks)

    i = 0
    while i < n:
        j = i
        size = 0
        while j < n:
            b_size = blocks[j].char_end - blocks[j].char_start
            if j > i and size + b_size > target_chars:
                break
            size += b_size
            j += 1
        if j == i:  # garde-fou : au moins un bloc
            j = i + 1

        first = blocks[i]
        last = blocks[j - 1]
        char_start = first.char_start
        char_end = last.char_end
        text = doc.markdown[char_start:char_end]
        pages = sorted({b.page for b in blocks[i:j]})

        chunks.append(
            Chunk(
                doc_id=doc.doc_id,
                source_file=doc.source_file,
                pages=pages,
                section_path=first.section_path,
                char_start=char_start,
                char_end=char_end,
                text=text,
                extractor=doc.extractor,
            )
        )

        if j >= n:
            break
        step = max(1, (j - i) - overlap_blocks)
        i = i + step

    return chunks


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _chunk_to_jsonable(c: Chunk) -> dict:
    d = asdict(c)
    # tuple -> list pour JSON ; on rétablit au read
    d["section_path"] = list(c.section_path)
    return d


def _jsonable_to_chunk(d: dict) -> Chunk:
    return Chunk(
        doc_id=d["doc_id"],
        source_file=d["source_file"],
        pages=list(d["pages"]),
        section_path=tuple(d["section_path"]),
        char_start=d["char_start"],
        char_end=d["char_end"],
        text=d["text"],
        extractor=d["extractor"],
    )


def write_jsonl(chunks: list[Chunk], path: str | Path) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(_chunk_to_jsonable(c), ensure_ascii=False))
            f.write("\n")


def read_jsonl(path: str | Path) -> list[Chunk]:
    path = Path(path)
    out: list[Chunk] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(_jsonable_to_chunk(json.loads(line)))
    return out


# ---------------------------------------------------------------------------
# Invariants — asserts utilisés en CLI ET disponibles pour tests
# ---------------------------------------------------------------------------

def assert_span_invariants(doc: SourceDoc, chunks: list[Chunk]) -> None:
    md = doc.markdown
    for k, b in enumerate(doc.blocks):
        assert md[b.char_start:b.char_end] == b.text, f"Block {k} viole l'invariant offset"
    for k, c in enumerate(chunks):
        assert md[c.char_start:c.char_end] == c.text, f"Chunk {k} viole l'invariant offset"


# ---------------------------------------------------------------------------
# Pipeline 1-fichier — réutilisable par ingest_corpus.py
# ---------------------------------------------------------------------------

def ingest_file(
    path: str | Path,
    out_dir: str | Path,
    target_chars: int = 1200,
    overlap_blocks: int = 1,
) -> tuple[SourceDoc, list[Chunk], Path]:
    """parse → chunk → asserte les invariants → écrit `<out_dir>/<doc_id>_chunks.jsonl`.
    Retourne (SourceDoc, chunks, chemin écrit). À appeler depuis le CLI mono-fichier
    et depuis le batch."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = parse(path)
    chunks = chunk(doc, target_chars=target_chars, overlap_blocks=overlap_blocks)
    assert_span_invariants(doc, chunks)
    out_path = out_dir / f"{doc.doc_id}_chunks.jsonl"
    write_jsonl(chunks, out_path)
    return doc, chunks, out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _fmt_preview(text: str, n: int = 80) -> str:
    s = text.replace("\n", " ⏎ ")
    return s[:n] + ("…" if len(s) > n else "")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="RAG ingestion (parse -> chunks -> JSONL).")
    ap.add_argument("file", help="Chemin du document à ingérer (pdf/docx/image...)")
    ap.add_argument("--out", default="data/chunks", help="Dossier de sortie (défaut: data/chunks)")
    ap.add_argument("--target", type=int, default=1200, help="Budget chars par chunk")
    ap.add_argument("--overlap", type=int, default=1, help="Recouvrement en nb de blocs")
    ap.add_argument("--preview", type=int, default=5, help="Nb de chunks à prévisualiser")
    args = ap.parse_args(argv)

    src_path = Path(args.file)
    if not src_path.exists():
        print(f"Fichier introuvable: {src_path}", file=sys.stderr)
        return 2

    print(f"[ingest] {src_path} -> {args.out}/")
    doc, chunks, out_path = ingest_file(
        src_path, args.out, target_chars=args.target, overlap_blocks=args.overlap
    )
    print(f"  extractor : {doc.extractor}")
    print(f"  doc_id    : {doc.doc_id}")
    print(f"  markdown  : {len(doc.markdown)} chars")
    print(f"  blocks    : {len(doc.blocks)}")
    print(f"  pages     : {len(doc.page_spans)}")

    sizes = [len(c.text) for c in chunks]
    if sizes:
        print(f"  chunks    : {len(chunks)}")
        print(f"  size chars: min={min(sizes)} avg={statistics.mean(sizes):.0f} max={max(sizes)}")
    else:
        print("  chunks    : 0")

    print("[invariants] markdown[start:end] == text  OK (asserts internes)")

    reread = read_jsonl(out_path)
    assert reread == chunks, "Round-trip JSONL cassé"
    print(f"[write] {out_path}  round-trip OK ({len(reread)} chunks)")

    print(f"[preview] {min(args.preview, len(chunks))} premiers chunks")
    for k, c in enumerate(chunks[: args.preview]):
        sp = " > ".join(c.section_path) if c.section_path else "(racine)"
        print(
            f"  #{k:03d} p={c.pages} span=[{c.char_start}:{c.char_end}] "
            f"section={sp!r}\n        {_fmt_preview(c.text)}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
