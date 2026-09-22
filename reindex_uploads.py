"""Rebuild the local Qdrant index from files in data/uploads.

Run after changing extraction/chunking logic or when the local index contains
stale/duplicate vectors:

    python reindex_uploads.py
"""

import hashlib
from pathlib import Path

from app.config import settings
from app.core.extractors import DocumentExtractor, DocumentExtractionError
from app.core.chunker import TextChunker
from app.core.embeddings import embedding_manager
from app.core.vector_db import vector_db

SUPPORTED = {".pdf", ".docx", ".doc", ".txt", ".xlsx", ".xls", ".csv"}


def content_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    files = sorted(
        p for p in settings.UPLOADS_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED
    )

    if not files:
        print("No supported files found in data/uploads.")
        return

    print(f"Found {len(files)} uploaded file(s). Clearing old Qdrant vectors...")
    vector_db.clear_collection()

    seen_hashes = set()
    total_chunks = 0
    indexed = 0

    for path in files:
        file_hash = content_hash(path)
        if file_hash in seen_hashes:
            print(f"SKIP duplicate: {path.name}")
            continue
        seen_hashes.add(file_hash)

        try:
            blocks = DocumentExtractor.extract(path)
            chunker = TextChunker(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
            )
            doc_id = file_hash[:32]
            chunks = chunker.chunk_document_blocks(blocks, doc_id=doc_id)
            embeddings = embedding_manager.embed_texts([c["text"] for c in chunks])
            vector_db.upsert_chunks(chunks, embeddings)
            indexed += 1
            total_chunks += len(chunks)
            print(f"OK   {path.name}: {len(chunks)} chunks")
        except DocumentExtractionError as exc:
            print(f"FAIL {path.name}: {exc}")

    print(f"\nIndexed {indexed} unique file(s), {total_chunks} chunks.")


if __name__ == "__main__":
    main()
