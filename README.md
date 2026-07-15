# PDF RAG Asistanı

PDF dosyalarınızı yükleyin, içeriklerini vektörel olarak indeksleyin ve sorularınızı doğrudan belgelerinize dayanarak cevaplandıran yerel RAG asistanı.

## Özellikler

- **PDF Yükleme & Chunking** - Recursive ve character tabanlı iki chunking stratejisi, chunk birleştirme ve sayfa bazlı indeksleme
- **Vektörel Arama** - ChromaDB + SentenceTransformers (all-MiniLM-L6-v2) ile cosine similarity tabanlı semantic search
- **Yerel LLM** - Ollama üzerinden Qwen 3.5 4B modeli (OpenAI seçeneği mevcut)
- **Oturum Hafızası** - Session bazlı conversation memory, query rewrite ile bağlamlı soru anlama
- **Streaming Cevaplama** - SSE (Server-Sent Events) ile gerçek zamanlı cevaplama
- **Yönetim Paneli** - PDF yükleme/silme, istatistikler, oturum geçmişi
- **Kaynak Atıfları** - Her cevapta kullanılan PDF ve sayfa bilgisini gösterir

## Mimari

```
rag-pdf/
├── main.py                 # FastAPI sunucu
├── config.py               # Ortam değişkenleri ve yapılandırma
├── rag/
│   ├── engine.py           # RAG pipeline (ingest, ask, ask_stream)
│   ├── llm.py              # LLM entegrasyonu (Ollama / OpenAI)
│   ├── chunking.py         # PDF okuma ve metin bölme
│   ├── vectorstore.py      # ChromaDB vektörel veritabanı
│   └── memory.py           # Oturum hafızası (JSON dosyası)
├── static/
│   └── index.html          # Web arayüzü
├── data/                   # ChromaDB ve chat geçmişi burada saklanır
├── uploads/                # Yüklenen PDF dosyaları
├── .env                    # Ortam değişkenleri (git'e commit edilmez)
└── .env.example            # Örnek ortam değişkenleri
```

### Pipeline Akışı

```
PDF Yükleme:  PDF -> PyPDF (sayfalara böl) -> TextSplitter (chunk'lara böl) -> Embedding -> ChromaDB

Soru Cevaplama:
  Soru -> Query Rewrite (geçmiş bağlamıyla) -> Embedding -> ChromaDB'de arama
  -> Bağlam oluşturma (kaynak + sayfa + benzerlik skoru)
  -> System Prompt + Bağlam + Geçmiş + Soru -> LLM -> Streaming cevap
```

## Kurulum

### Ön Gereksinimler

- Python 3.14+
- [Ollama](https://ollama.com) (yerel LLM için)
- 4 GB+ RAM (embedding model + LLM için)

### 1. Depoyu Klonla

```bash
git clone https://github.com/erenpinyall/rag-pdf.git
cd rag-pdf
```

### 2. Sanal Ortam Oluştur

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

### 3. Bağımlılıkları Yükle

```bash
pip install -e .
# veya uv kullanılıyorsa
uv sync
```

### 4. Ollama Kur ve Model İndir

```bash
# Ollama'yı kurduktan sonra
ollama pull qwen3.5:4b
```

### 5. Ortam Değişkenlerini Yapılandır

```bash
cp .env.example .env
```

`.env` dosyasını düzenle:

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3.5:4b
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=600
CHUNK_OVERLAP=100
CHUNK_STRATEGY=recursive
TOP_K=6
SIMILARITY_THRESHOLD=0.3
MAX_HISTORY=10
```

### 6. Sunucuyu Başlat

```bash
python main.py
```

Tarayıcıda `http://localhost:8000` adresine git.

## API Endpoints

| Method | Endpoint | Açıklama |
|--------|----------|----------|
| `GET` | `/` | Web arayüzü |
| `POST` | `/upload` | PDF yükle (multipart/form-data) |
| `POST` | `/ask` | Soru sor (JSON body: `question`, `session_id`, `top_k`, `stream`) |
| `GET` | `/stats` | Vektörel veritabanı istatistikleri |
| `GET` | `/sources` | Yüklenen PDF'lerin listesi |
| `DELETE` | `/sources/{name}` | Belirli bir PDF'i sil |
| `DELETE` | `/sources` | Tüm PDF'leri sil |
| `GET` | `/history/{session_id}` | Oturum geçmişi |
| `DELETE` | `/history/{session_id}` | Oturum geçmişini temizle |
| `GET` | `/sessions` | Tüm oturumların listesi |
| `GET` | `/health` | Sağlık kontrolü (LLM + Vectorstore durumu) |

### Örnek Kullanım

```bash
# PDF yükle
curl -X POST http://localhost:8000/upload \
  -F "file=@belge.pdf"

# Soru sor (streaming)
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Bu belgede ne anlatılıyor?", "stream": true}'

# İstatistikleri görüntüle
curl http://localhost:8000/stats
```

## Yapılandırma Parametreleri

| Parametre | Varsayılan | Açıklama |
|-----------|------------|----------|
| `LLM_PROVIDER` | `ollama` | LLM sağlayıcı: `ollama` veya `openai` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API adresi |
| `OLLAMA_MODEL` | `qwen3.5:4b` | Kullanılacak Ollama modeli |
| `OPENAI_API_KEY` | - | OpenAI API anahtarı (openai provider için) |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI modeli |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformers embedding modeli |
| `CHUNK_SIZE` | `600` | Metin chunk boyutu (karakter) |
| `CHUNK_OVERLAP` | `100` | Chunk'lar arası üstüme boyutu |
| `CHUNK_STRATEGY` | `recursive` | Chunking stratejisi: `recursive` veya `character` |
| `TOP_K` | `6` | Aramada döndürülen max chunk sayısı |
| `SIMILARITY_THRESHOLD` | `0.3` | Minimum benzerlik eşiği (0-1) |
| `MAX_HISTORY` | `10` | Oturum başına saklanan mesaj çifti sayısı |

## Teknolojiler

- **Backend:** FastAPI, Uvicorn
- **LLM:** Ollama (Qwen 3.5 4B) / OpenAI
- **Embedding:** SentenceTransformers (all-MiniLM-L6-v2)
- **Vectorstore:** ChromaDB (cosine similarity)
- **PDF:** PyPDF
- **Chunking:** LangChain Text Splitters
