import chromadb
from sentence_transformers import SentenceTransformer
from config import (
    CHROMA_DIR, EMBEDDING_MODEL, TOP_K, SIMILARITY_THRESHOLD,
)

_client = None
_collection = None
_encoder = None


def get_encoder() -> SentenceTransformer:
    global _encoder
    if _encoder is None:
        _encoder = SentenceTransformer(EMBEDDING_MODEL)
    return _encoder


def get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = _client.get_or_create_collection(
            name="pdf_chunks",
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def add_documents(chunks: list[dict]) -> int:
    collection = get_collection()
    encoder = get_encoder()

    ids = []
    for c in chunks:
        meta = c["metadata"]
        chunk_id = f"{meta['file_hash']}_p{meta['page']}_c{meta['chunk_index']}"
        ids.append(chunk_id)

    texts = [c["content"] for c in chunks]
    embeddings = encoder.encode(texts, show_progress_bar=False).tolist()
    metadatas = []
    for c in chunks:
        clean_meta = {}
        for k, v in c["metadata"].items():
            if isinstance(v, (str, int, float, bool)):
                clean_meta[k] = v
        metadatas.append(clean_meta)

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return len(ids)


def search(query: str, top_k: int = TOP_K, threshold: float = SIMILARITY_THRESHOLD) -> list[dict]:
    collection = get_collection()
    encoder = get_encoder()

    if collection.count() == 0:
        return []

    query_embedding = encoder.encode([query], show_progress_bar=False).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        score = 1 - dist
        if score >= threshold:
            hits.append({
                "content": doc,
                "metadata": meta,
                "score": round(score, 4),
            })
    return hits


def delete_by_source(source_name: str) -> int:
    collection = get_collection()
    results = collection.get(
        include=["metadatas"],
    )
    ids_to_delete = [
        doc_id
        for doc_id, meta in zip(results["ids"], results["metadatas"])
        if meta.get("source") == source_name
    ]
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
    return len(ids_to_delete)


def get_all_sources() -> list[dict]:
    collection = get_collection()
    if collection.count() == 0:
        return []

    results = collection.get(include=["metadatas"])
    source_map: dict[str, dict] = {}
    for meta in results["metadatas"]:
        source = meta.get("source", "unknown")
        if source not in source_map:
            source_map[source] = {
                "source": source,
                "pages": set(),
                "chunk_count": 0,
            }
        source_map[source]["pages"].add(meta.get("page", 0))
        source_map[source]["chunk_count"] += 1

    return [
        {
            "source": s["source"],
            "pages": sorted(s["pages"]),
            "chunk_count": s["chunk_count"],
        }
        for s in source_map.values()
    ]


def get_stats() -> dict:
    collection = get_collection()
    sources = get_all_sources()
    return {
        "total_chunks": collection.count(),
        "total_documents": len(sources),
        "sources": sources,
    }


def clear_all() -> int:
    collection = get_collection()
    count = collection.count()
    if count > 0:
        all_data = collection.get()
        collection.delete(ids=all_data["ids"])
    return count
