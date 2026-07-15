from typing import Generator
from rag.vectorstore import search, add_documents, get_stats
from rag.chunking import process_pdf
from rag.memory import memory
from rag.llm import llm_stream, llm_generate
from config import TOP_K

SYSTEM_PROMPT = """Sen bir PDF RAG (Retrieval-Augmented Generation) asistanisin.

Gorevlerin:
1. Kullanicinin sorularini, saglanan PDF baglamlarina dayanarak cevapla
2. Cevaplari Turkce ver, net ve kapsamli ol
3. Her cevabin sonunda hangi PDF ve sayfadan alindigini belirt
4. Eger cevap belgelerde yoksa, "Bu bilgi saglanan belgelerde bulunamadi" de ve genel bilgini kullanarak yardimci olmaya calis
5. Sorularin anlamini kavramaya calis, dogrudan cevap verme - aciklamali ol
6. Kod ornekleri veya liste gerektiren sorulari duzgun formatla

Cevap formati:
-Once cevabini ver
-Sonra [Kaynak: dosya_adi.pdf, Sayfa X] formatinda referanslari belirt"""


def _build_context(hits: list[dict]) -> str:
    if not hits:
        return "Henuz hicbir PDF yuklenmedi."

    parts = []
    for h in hits:
        meta = h["metadata"]
        source = meta.get("source", "bilinmiyor")
        page = meta.get("page", "?")
        score = h.get("score", 0)
        parts.append(
            f"[Kaynak: {source}, Sayfa {page}, Benzerlik: {score}]\n{h['content']}"
        )
    return "\n\n---\n\n".join(parts)


def _rewrite_query(question: str, chat_history: list[dict]) -> str:
    if not chat_history:
        return question

    recent = chat_history[-4:]
    history_text = "\n".join(
        f"{'Kullanici' if m['role'] == 'user' else 'Asistan'}: {m['content'][:200]}"
        for m in recent
    )

    rewrite_prompt = f"""Onceki konusma:
{history_text}

Simdi kullanicinin yeni sorusu: "{question}"

Eger soru onceki konuya referans veriyorsa (ornegin "peki bu", "diger sayfa", "ayni belgede") 
onu baglamli bir soruya cevir. Eger bagimsiz bir soruysa ayni birak.

Sadece duzeltilmis soruyu dondur, baska bir sey yazma."""

    try:
        messages = [{"role": "user", "content": rewrite_prompt}]
        rewritten = llm_generate(messages, temperature=0.1)
        if rewritten and len(rewritten) > 5:
            return rewritten.strip().strip('"').strip("'")
    except Exception:
        pass
    return question


def ingest_pdf(file_path: str) -> dict:
    result = process_pdf(file_path)
    count = add_documents(result["chunks"])
    return {
        "chunks_added": count,
        **result["stats"],
    }


def ask_stream(
    question: str,
    session_id: str = "default",
    top_k: int = TOP_K,
) -> Generator[dict, None, None]:
    chat_history = memory.get_history(session_id)
    rewritten = _rewrite_query(question, chat_history)

    hits = search(rewritten, top_k=top_k)
    context = _build_context(hits)

    history_messages = memory.get_context_messages(session_id, n=6)
    system_msg = {"role": "system", "content": SYSTEM_PROMPT}
    context_msg = {
        "role": "system",
        "content": f"Gelen PDF Baglamlari:\n\n{context}",
    }

    messages = [system_msg, context_msg] + history_messages + [
        {"role": "user", "content": question}
    ]

    full_answer = ""
    for chunk in llm_stream(messages, temperature=0.3):
        full_answer += chunk
        yield {"type": "chunk", "content": chunk}

    sources = [
        {
            "content": h["content"][:300] + "..." if len(h["content"]) > 300 else h["content"],
            "source": h["metadata"].get("source", "?"),
            "page": h["metadata"].get("page", "?"),
            "score": h.get("score", 0),
        }
        for h in hits
    ]

    memory.add_message(session_id, "user", question)
    memory.add_message(session_id, "assistant", full_answer, sources)

    yield {
        "type": "done",
        "answer": full_answer,
        "sources": sources,
        "rewritten_query": rewritten if rewritten != question else None,
    }


def ask(
    question: str,
    session_id: str = "default",
    top_k: int = TOP_K,
) -> dict:
    chat_history = memory.get_history(session_id)
    rewritten = _rewrite_query(question, chat_history)

    hits = search(rewritten, top_k=top_k)
    context = _build_context(hits)

    history_messages = memory.get_context_messages(session_id, n=6)
    system_msg = {"role": "system", "content": SYSTEM_PROMPT}
    context_msg = {
        "role": "system",
        "content": f"Gelen PDF Baglamlari:\n\n{context}",
    }

    messages = [system_msg, context_msg] + history_messages + [
        {"role": "user", "content": question}
    ]

    answer = llm_generate(messages, temperature=0.3)

    sources = [
        {
            "content": h["content"][:300] + "..." if len(h["content"]) > 300 else h["content"],
            "source": h["metadata"].get("source", "?"),
            "page": h["metadata"].get("page", "?"),
            "score": h.get("score", 0),
        }
        for h in hits
    ]

    memory.add_message(session_id, "user", question)
    memory.add_message(session_id, "assistant", answer, sources)

    return {
        "answer": answer,
        "sources": sources,
        "rewritten_query": rewritten if rewritten != question else None,
    }
