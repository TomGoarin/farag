"""Ingestion batch : data/raw/ → data/chunks/<doc_id>_chunks.jsonl

Skip au niveau fichier si le JSONL cible existe déjà (sauf --force).
Un fichier qui plante n'arrête pas le batch — il est marqué ÉCHEC.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from ingest import compute_doc_id, ingest_file


SUPPORTED_EXTS = {".pdf", ".docx", ".png", ".jpg", ".jpeg"}


def _count_lines(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for _ in f)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Batch d'ingestion sur data/raw.")
    ap.add_argument("--raw", default="data/raw", help="Dossier des sources")
    ap.add_argument("--out", default="data/chunks", help="Dossier des JSONL")
    ap.add_argument("--force", action="store_true", help="Ré-ingère même si la sortie existe")
    ap.add_argument("--target", type=int, default=1200)
    ap.add_argument("--overlap", type=int, default=1)
    args = ap.parse_args(argv)

    raw = Path(args.raw)
    out = Path(args.out)
    if not raw.is_dir():
        print(f"Dossier source introuvable: {raw}", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)

    recap: list[tuple[str, str, int, str]] = []  # (doc_id, source, n_chunks, status)

    files = sorted(p for p in raw.iterdir() if p.is_file() and not p.name.startswith("."))
    for f in files:
        ext = f.suffix.lower()
        if ext not in SUPPORTED_EXTS:
            print(f"  [IGNORÉ] {f.name} (extension {ext or '∅'} non supportée)")
            continue

        try:
            doc_id = compute_doc_id(f)
        except OSError as e:
            print(f"  [ÉCHEC]  {f.name}: lecture impossible ({e})", file=sys.stderr)
            recap.append(("?" * 16, str(f), 0, "ÉCHEC"))
            continue

        target = out / f"{doc_id}_chunks.jsonl"

        if target.exists() and not args.force:
            n = _count_lines(target)
            print(f"  [SAUTÉ]  {f.name} → {target.name} ({n} chunks déjà présents)")
            recap.append((doc_id, str(f), n, "SAUTÉ"))
            continue

        print(f"  [.....]  {f.name} → ingestion...")
        try:
            doc, chunks, written = ingest_file(
                f, out, target_chars=args.target, overlap_blocks=args.overlap
            )
            print(f"  [INGÉRÉ] {f.name} → {written.name} ({len(chunks)} chunks)")
            recap.append((doc.doc_id, str(f), len(chunks), "INGÉRÉ"))
        except Exception as e:  # noqa: BLE001 — on isole l'échec par fichier
            print(f"  [ÉCHEC]  {f.name}: {type(e).__name__}: {e}", file=sys.stderr)
            traceback.print_exc(limit=2)
            recap.append((doc_id, str(f), 0, "ÉCHEC"))

    # ------------------------------------------------------------------ récap
    print()
    print(f"{'doc_id':<18} {'source_file':<40} {'n_chunks':>8}  status")
    print("-" * 82)
    for did, src, n, status in recap:
        print(f"{did:<18} {Path(src).name:<40} {n:>8}  {status}")
    print()
    n_ok = sum(1 for r in recap if r[3] in {"INGÉRÉ", "SAUTÉ"})
    n_ko = sum(1 for r in recap if r[3] == "ÉCHEC")
    print(f"Total : {n_ok} OK, {n_ko} échec(s), {len(recap)} fichier(s) supportés.")

    return 1 if n_ko else 0


if __name__ == "__main__":
    raise SystemExit(main())
