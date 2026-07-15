import hashlib
from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from config import CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_STRATEGY


def load_pdf(file_path: str | Path) -> list[dict]:
    reader = PdfReader(str(file_path))
    file_hash = hashlib.md5(Path(file_path).read_bytes()).hexdigest()[:8]
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append({
                "content": text.strip(),
                "metadata": {
                    "source": Path(file_path).name,
                    "file_hash": file_hash,
                    "page": i + 1,
                    "total_pages": len(reader.pages),
                },
            })
    return pages


def _recursive_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n\n", "\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
    )


def _character_splitter() -> CharacterTextSplitter:
    return CharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separator="\n",
    )


def _get_splitter():
    strategy = CHUNK_STRATEGY.lower()
    if strategy == "character":
        return _character_splitter()
    return _recursive_splitter()


def chunk_text(pages: list[dict]) -> list[dict]:
    splitter = _get_splitter()
    chunks = []
    for page in pages:
        splits = splitter.split_text(page["content"])
        for j, split in enumerate(splits):
            tokens_approx = len(split.split())
            chunks.append({
                "content": split,
                "metadata": {
                    **page["metadata"],
                    "chunk_index": j,
                    "chunk_count": len(splits),
                    "word_count": tokens_approx,
                },
            })

    _link_chunks(chunks)
    return chunks


def _link_chunks(chunks: list[dict]) -> None:
    for i, chunk in enumerate(chunks):
        if i > 0:
            chunk["metadata"]["prev_chunk"] = chunks[i - 1]["content"][:150]
        if i < len(chunks) - 1:
            chunk["metadata"]["next_chunk"] = chunks[i + 1]["content"][:150]


def process_pdf(file_path: str | Path) -> dict:
    pages = load_pdf(file_path)
    chunks = chunk_text(pages)
    total_words = sum(c["metadata"]["word_count"] for c in chunks)
    page_numbers = sorted(set(c["metadata"]["page"] for c in chunks))
    return {
        "chunks": chunks,
        "stats": {
            "total_pages": len(pages),
            "total_chunks": len(chunks),
            "total_words": total_words,
            "pages": page_numbers,
            "strategy": CHUNK_STRATEGY,
        },
    }
