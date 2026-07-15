import json
import uuid
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag.engine import ingest_pdf, ask, ask_stream
from rag.vectorstore import get_stats, delete_by_source, get_all_sources, clear_all
from rag.memory import memory
from rag.llm import check_connection
from config import UPLOAD_DIR

app = FastAPI(title="PDF RAG Asistani", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=5000)
    session_id: str = Field(default="default")
    top_k: int = Field(default=6, ge=1, le=20)
    stream: bool = Field(default=True)


class ClearHistoryRequest(BaseModel):
    session_id: str = "default"


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Sadece PDF dosyalari yuklenebilir.")

    dest = UPLOAD_DIR / file.filename
    content = await file.read()

    if len(content) == 0:
        raise HTTPException(400, "Dosya bos.")

    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(400, "Dosya cok buyuk (maks 50MB).")

    dest.write_bytes(content)

    try:
        result = ingest_pdf(str(dest))
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(500, f"PDF islenemedi: {str(e)}")

    return {"filename": file.filename, "status": "ok", **result}


@app.post("/ask")
async def ask_question(req: AskRequest):
    if req.stream:
        return StreamingResponse(
            _stream_response(req.question, req.session_id, req.top_k),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    result = ask(req.question, session_id=req.session_id, top_k=req.top_k)
    return result


async def _stream_response(question: str, session_id: str, top_k: int):
    for event in ask_stream(question, session_id=session_id, top_k=top_k):
        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


@app.get("/stats")
async def stats():
    return get_stats()


@app.get("/sources")
async def sources():
    return get_all_sources()


@app.delete("/sources/{source_name}")
async def remove_source(source_name: str):
    count = delete_by_source(source_name)
    if count == 0:
        raise HTTPException(404, "Kaynak bulunamadi.")
    upload_file = UPLOAD_DIR / source_name
    upload_file.unlink(missing_ok=True)
    return {"deleted_chunks": count, "source": source_name}


@app.delete("/sources")
async def remove_all_sources():
    count = clear_all()
    for f in UPLOAD_DIR.glob("*.pdf"):
        f.unlink(missing_ok=True)
    return {"deleted_chunks": count}


@app.get("/history/{session_id}")
async def get_history(session_id: str):
    return memory.get_history(session_id)


@app.delete("/history/{session_id}")
async def clear_history(session_id: str):
    memory.clear_session(session_id)
    return {"status": "ok"}


@app.get("/sessions")
async def list_sessions():
    return memory.get_all_sessions()


@app.get("/health")
async def health():
    llm_status = check_connection()
    stats = get_stats()
    return {
        "status": "ok",
        "llm": llm_status,
        "vectorstore": stats,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
