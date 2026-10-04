
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ================= IMPORTS =================
import os
import uuid
import faiss
import whisper
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
import json
#from transformers import pipeline
from PIL import Image
import fitz  # PyMuPDF
import docx2txt
import pytesseract
from rank_bm25 import BM25Okapi
# Tesseract path (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from fastapi.middleware.cors import CORSMiddleware
import hashlib
from transformers import AutoTokenizer
from fastapi.responses import FileResponse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from transformers import CLIPProcessor, CLIPModel
import torch
import requests
from typing import Any
from collections import Counter

from fastapi.responses import StreamingResponse
import asyncio

from novarag_memory import (
    append_turn, get_turns, clear_session, MAX_MEMORY_TURNS,
    lookup_semantic_cache, store_semantic_cache, clear_semantic_cache, get_semantic_cache_stats
)

# ================= STRICT THRESHOLDS & CACHE CONFIG =================
SEMANTIC_CACHE_THRESHOLD = 0.90   # Strict cosine similarity for semantic cache hits (0.90)
MIN_RETRIEVAL_SCORE = 0.35        # Strict minimum hybrid score to retain a chunk (drops random out-of-domain queries <0.35)
MIN_CONFIDENCE_THRESHOLD = 35.0   # Strict minimum confidence percentage required to answer

# ================= MEMORY =================
conversation_memory = []
MAX_MEMORY_TURNS = 5
device = "cuda" if torch.cuda.is_available() else "cpu"
from sentence_transformers import SentenceTransformer
#clip_model = SentenceTransformer("clip-ViT-B-32",device=device)
from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

from PIL import Image

clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32",
    use_safetensors=True
)

clip_model = clip_model.to(torch.device(device)) # type: ignore
clip_model.eval()   # IMPORTANT


clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def embed_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")  # type: ignore
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        # Use vision_model then apply visual_projection to get 512-dim
        vision_outputs = clip_model.vision_model(**inputs)
        pooled = vision_outputs.pooler_output  # (1, 768)
        features = clip_model.visual_projection(pooled).float()  # (1, 512)

    features = features / (features.norm(dim=-1, keepdim=True) + 1e-10)
    return features.detach().cpu().numpy().flatten()


def embed_text_clip(text):
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True)  # type: ignore
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        # Use text_model then apply text_projection to get 512-dim
        text_outputs = clip_model.text_model(**inputs)
        pooled = text_outputs.pooler_output  # (1, 512)
        features = clip_model.text_projection(pooled).float()  # (1, 512)

    features = features / (features.norm(dim=-1, keepdim=True) + 1e-10)
    return features.detach().cpu().numpy().flatten()
#-- system prompt for LLM-----
SYSTEM_PROMPT = """
You are NovaRAG, a document-grounded assistant.

STRICT RULES:
1) Answer ONLY using information explicitly stated in the CONTEXT below
2) Do NOT infer, guess, or use your general knowledge
3) Do NOT say "it can be inferred" or "it can be assumed"
4) If the exact term is in the context, quote and explain it directly
5) If the answer is genuinely absent, say exactly:
   "The answer is not present in the provided sources."
6) Never add information beyond what the context contains
"""
# ================= INIT =================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


#---- LLM + EMBEDDING CONFIG FIRST
OLLAMA_URL = "http://localhost:11434/api/generate"

embed_model = SentenceTransformer("all-MiniLM-L6-v2",device=device)




dimension=384

#-----MODELS-----
from transformers import pipeline
'''
llm = pipeline(
    task="text2text-generation", # type: ignore
    model="google/flan-t5-base",
    max_length=512
)
'''
whisper_model = whisper.load_model("base")
#tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
#----- DATABASE (FAISS + METADATA)FILES -----
INDEX_FILE = "faiss.index"
META_FILE = "meta.pkl"

import pickle

# Load index safely
# ================= FAISS + METADATA INIT =================

import pickle

INDEX_FILE = "faiss.index"
IMAGE_INDEX_FILE = "image.index"
META_FILE = "meta.pkl"

TEXT_DIM = 384
IMAGE_DIM = 512

# -------- TEXT INDEX --------
if os.path.exists(INDEX_FILE):
    print("🔁 Loading existing TEXT FAISS index...")
    text_index = faiss.read_index(INDEX_FILE) # type: ignore
else:
    print("🆕 Creating new TEXT FAISS index...")
    text_index = faiss.IndexFlatIP(TEXT_DIM)# type: ignore

# -------- IMAGE INDEX --------
if os.path.exists(IMAGE_INDEX_FILE):
    loaded = faiss.read_index(IMAGE_INDEX_FILE)  # type: ignore
    if loaded.d == IMAGE_DIM:
        print("🔁 Loading existing IMAGE FAISS index...")
        image_index = loaded
    else:
        print(f"⚠️ IMAGE index dimension mismatch ({loaded.d} vs {IMAGE_DIM}) → rebuilding...")
        image_index = faiss.IndexFlatIP(IMAGE_DIM)  # type: ignore
        image_metadata = []  # reset metadata too since index is stale
else:
    print("🆕 Creating new IMAGE FAISS index...")
    image_index = faiss.IndexFlatIP(IMAGE_DIM)  # type: ignore
# -------- METADATA --------
if os.path.exists(META_FILE):
    print("🔁 Loading metadata store...")
    with open(META_FILE, "rb") as f:
        data = pickle.load(f)

        if isinstance(data, dict):
            metadata_store = data.get("text", [])
            image_metadata = data.get("image", [])
        else:
            metadata_store = data
            image_metadata = []
else:
    metadata_store = []
    image_metadata = []


# -------- REBUILD BM25 FROM DISK -------- ← ADD FROM HERE
bm25_corpus = []
bm25_chunks = []

for i, item in enumerate(metadata_store):
    tokens = item["text"].lower().split()
    bm25_corpus.append(tokens)
    bm25_chunks.append(i)

bm25 = BM25Okapi(bm25_corpus) if bm25_corpus else None
print(f"✅ BM25 rebuilt: {len(bm25_corpus)} chunks")
# -------- END BM25 REBUILD --------



# -------- DEDUP HASH --------
chunk_hashes = set(
    hashlib.sha256(str(m.get("text", "")).strip().encode()).hexdigest()
    for m in metadata_store
)

def call_llm(prompt: str, model_choice: str) -> str:
    # Fix: always assign model_name
    if model_choice == "fast":
        model_name = "mistral:latest"   # was redirecting but never setting model_name
    elif model_choice == "mistral":
        model_name = "mistral:latest"
    elif model_choice == "llama3":
        model_name = "llama3:latest"
    else:
        model_name = "mistral:latest"
    
    

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 512
                }
            },
            stream=True,
            timeout=180
        )

        full_response = ""

        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode("utf-8"))
                    full_response += data.get("response", "")
                except:
                    pass

        return full_response.strip()

    except Exception as e:
        print("Ollama error:", e)
        return "Local LLM not running."



#-- detector to identify if question is asking for a list (to preserve formatting)
def is_list_question(q: str):
    q = q.lower()
    keywords = [
        "phases", "steps", "stages", "types", "categories",
        "lifecycle", "life cycle", "components", "modules"
    ]
    return any(k in q for k in keywords)

#-- extractor to pull out clean list items from messy LLM output
import re

def extract_list_items(text: str):
    """
    Extract short noun-like lines (phases/steps) from messy LLM output
    """
    if not isinstance(text, str):
        text = str(text)

    lines = re.split(r"\n|,|•|;", text)
    candidates = []

    for line in lines:
        line = line.strip("•-–—: ").strip()

        # ignore long sentences
        if len(line.split()) > 20:
            continue

        # must contain letters
        if not re.search(r"[a-zA-Z]", line):
            continue

        # remove explanations in brackets
        line = re.sub(r"\(.*?\)", "", line).strip()

        if len(line) > 3:
            candidates.append(line)

    # deduplicate + preserve order
    seen = set()
    final = []
    for item in candidates:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            final.append(item)

    return final


# ================= HELPERS =================





def embed(text: str):
    if not isinstance(text, str):
        text = str(text)
    vec = embed_model.encode(text, normalize_embeddings=True)
    return np.array(vec, dtype="float32")

#--- cosine similarity for topic shift detection
def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return float(np.dot(vec1, vec2))
#-- simple topic shift detector based on embedding similarity
def detect_topic_shift_by_file(current_docs, session="default"):
    turns = get_turns(session)
    if not turns or not current_docs: return False
    last_file = turns[-1].get("file")
    if not last_file:
        return False
    current_file = current_docs[0]["path"]
    if current_file == last_file:
        return False
    # Only call it a topic shift if majority of retrieved docs
    # are from a different file — not just the top one
    current_files = [d.get("path") for d in current_docs[:3]]
    same_count = sum(1 for f in current_files if f == last_file)
    if same_count >= 2:
        return False  # still mostly same file, not a real shift
    print(f"🔄 Topic shift: {last_file} → {current_file}")
    return True
# ================= SMART CHUNKING =================
import re


def create_chunks(text, chunk_size=80, overlap=20):
    # Split on double newlines (paragraph boundaries)
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    
    chunks = []
    current_words = []
    current_paras = []
    
    for para in paragraphs:
        para_words = para.split()
        
        # If adding this paragraph exceeds chunk_size, flush current
        if current_words and len(current_words) + len(para_words) > chunk_size:
            chunks.append(" ".join(current_words))
            # Keep last paragraph for overlap
            if current_paras:
                last_para = current_paras[-1]
                current_words = last_para.split()
                current_paras = [last_para]
            else:
                current_words = []
                current_paras = []
        
        current_words.extend(para_words)
        current_paras.append(para)
    
    # Add remaining
    if current_words:
        chunks.append(" ".join(current_words))
    
    return chunks
'''
bm25_corpus = []
bm25_chunks = []
bm25 = None
'''
def add_to_index(text: str, filename: str, page=None, chunk_id=None):


    # ----- FAISS -----
    # --- Deduplication ---
    # --- SAFETY CHECK ---
    if not isinstance(text, str):
        text = str(text)

    text = text.strip()

    chunk_hash = hashlib.sha256(text.encode()).hexdigest()
    print("Adding chunk:", text[:50])
    
    if chunk_hash in chunk_hashes:
        return False # Skip duplicate chunk
    
    # --- FAISS ---
    vec = embed(text)
    if vec is None or len(vec) != dimension:
        return False
    

    chunk_hashes.add(chunk_hash)

    

    text_index.add(np.array([vec], dtype="float32")) #type: ignore
    
    metadata_store.append({
        "id": str(uuid.uuid4()),
        "text": text,
        "path": filename,
        "page": page,
        "chunk_id": chunk_id,
        
    })
    


    # ----- BM25 -----
    tokens = text.lower().split()
    bm25_corpus.append(tokens)
    bm25_chunks.append(len(metadata_store) - 1)  # store metadata index
    return True

def add_chunks_batch(chunks: list, filename: str, page=None, start_chunk_id=0, batch_size=32) -> int:
    """
    Batched ingestion for large files. Computes embeddings in batches of 32
    using SentenceTransformer, speeding up indexing by 5x-8x on large documents.
    """
    if not chunks:
        return 0

    valid_chunks = []
    chunk_meta = []
    current_id = start_chunk_id

    for c in chunks:
        c_text = str(c).strip() if c is not None else ""
        if not c_text:
            continue
        c_hash = hashlib.sha256(c_text.encode()).hexdigest()
        if c_hash in chunk_hashes:
            continue
        chunk_hashes.add(c_hash)
        valid_chunks.append(c_text)
        chunk_meta.append((page, current_id))
        current_id += 1

    if not valid_chunks:
        return 0

    # Batch encode vectors
    all_vecs = embed_model.encode(
        valid_chunks,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    all_vecs = np.array(all_vecs, dtype="float32")

    # Add all vectors to FAISS at once
    text_index.add(all_vecs)

    for c_text, (pg, c_id) in zip(valid_chunks, chunk_meta):
        metadata_store.append({
            "id": str(uuid.uuid4()),
            "text": c_text,
            "path": filename,
            "page": pg,
            "chunk_id": c_id,
        })
        tokens = c_text.lower().split()
        bm25_corpus.append(tokens)
        bm25_chunks.append(len(metadata_store) - 1)

    return len(valid_chunks)


#----- HYBRID SEARCH (FAISS + BM25) ----

def hybrid_search(query, k=5, image_path=None, file_filter=None):
    
    if text_index.ntotal == 0:
        return []

    # -------- VECTOR --------
    if image_path:
        qvec = embed_image(image_path)
    else:
        qvec = embed(query)

    query_vector = np.array([qvec]).astype("float32")
    distances, labels = text_index.search(query_vector, k) # type: ignore
    # -------- IMAGE SEARCH (NEW) --------
    image_results = []

    if image_index.ntotal > 0:
        if image_path:
            img_query_vec = embed_image(image_path)
        else:
            img_query_vec = embed_text_clip(query)   # 🔥 TEXT → IMAGE SEARCH

        img_query_vec = np.array([img_query_vec]).astype("float32")

        img_distances, img_indices = image_index.search(img_query_vec, k) # type: ignore

        for score, idx in zip(img_distances[0], img_indices[0]):
            if idx < len(image_metadata):
                image_results.append({
                    "snippet": f"[IMAGE: {image_metadata[idx]['path']}]",  # ← add this line
                    "path": image_metadata[idx]["path"],
                    "preview": image_metadata[idx]["preview"],
                    "score": float(score),
                    "type": "image"
                })
    faiss_scores = {}
    for score, idx in zip(distances[0], labels[0]):
        if 0 <= idx < len(metadata_store):
            # Inner product of normalized vectors is true cosine similarity [-1.0, 1.0]
            # Clip negative similarities to 0.0 to represent non-negative relevance
            faiss_scores[idx] = max(0.0, float(score))

    # ---------------- BM25 SEARCH ----------------
    bm25_scores = {}

    if bm25:
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)

        # Get top-k BM25 results
        top_bm25_idx = np.argsort(scores)[::-1][:k]

        for pos in top_bm25_idx:
            meta_idx = bm25_chunks[pos]  # translate BM25 position → metadata index
            raw_b = float(scores[pos])
            # Calibrate BM25 score against reference saturation (score of 12.0 ≈ 1.0)
            bm25_scores[meta_idx] = min(max(0.0, raw_b / 12.0), 1.0)

    # ---------------- HYBRID FUSION ----------------
    combined = {}

    all_indices = set(faiss_scores.keys()).union(bm25_scores.keys())

    for idx in all_indices:
        f_score = faiss_scores.get(idx, 0)
        b_score = bm25_scores.get(idx, 0)
        if len(query.split()) <= 3:
            alpha = 0.5   # short query → keyword important
        else:
            alpha = 0.75  # long query → semantic important
        hybrid_score = alpha * f_score + (1 - alpha) * b_score

        combined[idx] = hybrid_score

    # Sort by hybrid score
    sorted_indices = sorted(combined.items(), key=lambda x: x[1], reverse=True)

    # Build final results
    results = []
    for idx, final_score in sorted_indices[:k]:
        item = metadata_store[idx]
        results.append({
            "idx": idx,
            "snippet": item["text"],
            "path": item["path"],
            "page": item["page"],
            "chunk_id": item.get("chunk_id"),
            "score": final_score,
            "type": item.get("type") if item.get("type") in {"text", "table", "image", "audio"} else "text"
        })
    if file_filter:
        results = [r for r in results if r["path"] == file_filter]
    print(f"Retrieved {len(results)} results")
    return results + image_results
    


def build_prompt(context: str, question: str, sources: list):
    """
    sources: list of dicts with keys 'num', 'path', 'page'
    Instructs the LLM to embed [1], [2] inline markers.
    """
    source_legend = "\n".join(
        f"[{s['num']}] {s['path']}" + (f" (page {s['page']})" if s.get('page') else "")
        for s in sources
    )

    return f"""
{SYSTEM_PROMPT}

SOURCES (use ONLY these citation numbers):
{source_legend}

STRICT CITATION RULES:
- You MUST use ONLY the numbers listed above e.g. [1] or [2]
- NEVER invent formats like [N3] or [N1] or [N]
- Place citation immediately after the fact: "Pizza takes 20 minutes [1]."
- If a fact appears in multiple sources: "It is true [1][2]."
- NEVER cite a source not listed above
- If answer is not in sources say exactly: "The answer is not present in the provided sources."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER (with inline citations using only the numbered sources above):
"""


def rerank_with_cross_encoder(question, docs):
    if not docs:
        return docs
    # Filter out image-only results before reranking
    text_docs = [d for d in docs if d.get("type") != "image"]
    image_docs = [d for d in docs if d.get("type") == "image"]
    
    if not text_docs:
        return image_docs
    
    pairs = [(question, d["snippet"]) for d in text_docs]
    scores = cross_encoder.predict(pairs)
    for i, d in enumerate(text_docs):
        d["ce_score"] = float(scores[i])
    text_docs = sorted(text_docs, key=lambda x: x["ce_score"], reverse=True)
    return text_docs[:5] + image_docs


#retrieved_docs = docs.copy()
#-- SEMANTIC WINDOWING (RETRIEVE CONTEXT AROUND TOP CHUNKS) --------

def build_semantic_section(docs, window=1):
    selected = []

    for d in docs:
        # Image results: pass through directly, no windowing needed
        if d.get("type") == "image" or d.get("chunk_id") is None:
            selected.append(d)
            continue

        base_id = d["chunk_id"]
        file = d["path"]

        for meta in metadata_store:
            if meta["path"] == file and meta.get("chunk_id") is not None:
                if abs(meta["chunk_id"] - base_id) <= window:
                    selected.append({
                        "snippet": meta["text"],
                        "path": meta["path"],
                        "page": meta.get("page"),        # safe .get()
                        "chunk_id": meta.get("chunk_id"),
                        "score": d.get("score", 0),
                        "type": meta.get("type", "text")
                    })

    # Remove duplicates safely — use .get() for page
    unique = {(x.get("snippet"), x.get("path"), x.get("page")): x for x in selected}

    ordered = sorted(
        unique.values(),
        key=lambda x: (x.get("path", ""), x.get("chunk_id") or 0)
    )

    return ordered

def clean_answer(ans: str):
    if not isinstance(ans, str):
        ans = str(ans)
    return ans.strip()

def convert_numpy(obj):
    """Recursively convert numpy types to native Python types."""
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_database():
    global image_index   # ✅ IMPORTANT

    import pickle

    faiss.write_index(text_index, INDEX_FILE)
    faiss.write_index(image_index, "image.index")

    with open(META_FILE, "wb") as f:
        pickle.dump({
            "text": metadata_store,
            "image": image_metadata
        }, f)

    print("💾 Database saved to disk")

def build_list_prompt(context: str, question: str):
    return f"""
You extract lists from documents.

STRICT RULES:
- Extract list items EXACTLY as written in the context
- Do NOT add items from your own knowledge
- Do NOT use generic knowledge about BPM, SDLC, or any framework
- If the context does not contain a numbered list, describe what phases ARE mentioned
- Return ONLY what is in the context below
- If nothing relevant exists say: "The answer is not present in the provided sources."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER (only from context above):
"""

def build_memory_context(session: str = "default"):
    turns = get_turns(session)
    if not turns:
        return ""
    memory_text = "Previous Conversation:\n"
    for turn in turns:
        memory_text += f"User: {turn['question']}\n"
        memory_text += f"Assistant: {turn['answer']}\n"
    return memory_text + "\n"


def build_context(docs):
    parts = []
    filtered_docs = []

    for d in docs:
        score = d.get("score", 0)
        try:
            score = float(score)
        except:
            score = 0
        threshold = 0.15 if d.get("type") == "image" else 0.25
        if score > threshold:
            filtered_docs.append(d)

    # Build stable source → citation number mapping
    source_map: dict = {}
    num = 1
    for d in filtered_docs:
        key = (d.get("path"), d.get("page"))
        if key not in source_map:
            source_map[key] = num
            num += 1

    for d in filtered_docs:
        key = (d.get("path"), d.get("page"))
        citation_num = source_map[key]
        src = f"[{citation_num}] [{d.get('path', 'Unknown')}]"
        if d.get("page"):
            src += f" (page {d['page']})"
        parts.append(f"{src}\n{d.get('snippet', '')}")

    return "\n\n".join(parts), source_map  # ← NOW returns tuple
def is_image_query(q):
    keywords = ["image", "screenshot", "photo", "picture"]
    return any(k in q.lower() for k in keywords)

# ================= INGEST =================

@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):

    if not file.filename:
        return {"status": "Invalid file"}

    save_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(save_path, "wb") as f:
        f.write(await file.read())

    ingested_chunks = 0
    ext = file.filename.lower()
    

    # -------- PDF --------
    if ext.endswith(".pdf"):
        doc = fitz.open(save_path)
        global_chunk_id = 0

        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            text = page.get_text("text") or ""
            if not isinstance(text, str):
                text = str(text)

            # OCR fallback for scanned pages with safeguard for massive PDFs
            if not text.strip():
                if len(doc) > 30 and page_index >= 10:
                    print(f"⚠️ Page {page_index+1}: Scanned PDF has {len(doc)} pages. Skipping OCR beyond page 10 to avoid timeout.")
                    continue
                print(f"⚠️ Page {page_index+1}: No text found, running OCR...")
                try:
                    pix = page.get_pixmap()
                    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                    text = pytesseract.image_to_string(img, config='--oem 3 --psm 6')
                except Exception as e:
                    print(f"OCR error on page {page_index+1}:", e)

            # Table extraction
            try:
                tables = []
                if hasattr(page, "find_tables"):
                    tables = page.find_tables() or []  # type: ignore

                for table in tables:
                    table_data = table.extract()
                    for row in table_data:
                        row_text = " ".join(str(cell) for cell in row if cell)
                        if len(row_text.strip()) > 10:
                            text += "\n" + row_text
            except Exception as e:
                print("Table extraction error:", e)

            chunks = create_chunks(text)
            added = add_chunks_batch(chunks, filename=file.filename, page=page_index + 1, start_chunk_id=global_chunk_id)
            ingested_chunks += added
            global_chunk_id += added

    # -------- DOCX --------
    elif ext.endswith(".docx"):
        text = docx2txt.process(save_path) or ""

        # Remove only exact duplicate paragraphs (not all repeated lines)
        paragraphs = text.split("\n\n")
        seen_paragraphs = set()
        clean_paragraphs = []
        for para in paragraphs:
            stripped = para.strip()
            if not stripped:
                continue
            if len(stripped) > 50:
                if stripped in seen_paragraphs:
                    continue
                seen_paragraphs.add(stripped)
            clean_paragraphs.append(para)

        clean_text = "\n\n".join(clean_paragraphs)
        print(f"Original: {len(text)} chars → Cleaned: {len(clean_text)} chars")
        chunks = create_chunks(clean_text)
        print(f"Total chunks created: {len(chunks)}")
        ingested_chunks = add_chunks_batch(chunks, filename=file.filename, page=None, start_chunk_id=0)

    # -------- IMAGE --------
    elif ext.endswith((".png", ".jpg", ".jpeg")):
        img = Image.open(save_path)

        # 1. OCR text (existing)
        text = pytesseract.image_to_string(img) or ""

        # 2. Image embedding (NEW)
        img_vec = embed_image(save_path)
        # Add to IMAGE index
        image_index.add(np.array([img_vec], dtype="float32")) #type: ignore

        image_metadata.append({
            "path": file.filename,
            "preview": save_path,
            "type": "image"
        })

        caption = "image showing " + text[:150]

        add_to_index(
            text=caption,
            filename=file.filename,
            chunk_id=None
        )

        metadata_store[-1]["type"] = "image"
        metadata_store[-1]["preview"] = save_path
        ingested_chunks = 1

    # -------- AUDIO --------
    elif ext.endswith((".mp3", ".wav", ".m4a")):
        result = whisper_model.transcribe(save_path, word_timestamps=True)
        text = str(result.get("text", ""))

        # ----- CLEAN YOUTUBE / VIDEO NOISE -----
        noise_patterns = [
            r"like (and )?share this video.*",
            r"don't forget to subscribe.*",
            r"subscribe.*",
            r"thanks for watching.*",
            r"see you in the next video.*",
            r"if you find it useful.*",
            r"click the bell.*",
            r"hit the like button.*",
            r"thank you[.!]?\s*$"
        ]

        for pattern in noise_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)

        chunks = create_chunks(text)
        ingested_chunks = add_chunks_batch(chunks, filename=file.filename, start_chunk_id=0)

        chunk_id_counter = ingested_chunks
        for segment in result.get("segments", []):
            if not isinstance(segment, dict):  # type: ignore
                continue
            seg_text = segment.get("text", "")
            if not isinstance(seg_text, str):
                seg_text = str(seg_text)
            seg_text = seg_text.strip()
            start = segment.get("start", 0)
            if not seg_text:
                continue

            if add_to_index(text=seg_text, filename=file.filename, chunk_id=chunk_id_counter):
                metadata_store[-1]["timestamp"] = start
                chunk_id_counter += 1
                ingested_chunks += 1

    else:
        return {"status": "Unsupported file"}

    # Save AFTER indexing
    save_database()
    global bm25
    if bm25_corpus:
        bm25 = BM25Okapi(bm25_corpus)

    # Invalidate semantic cache when knowledge base is updated
    clear_semantic_cache()

    return {"status": "File indexed successfully", "chunks": ingested_chunks}

# ================= ROUTES =================

@app.get("/stats")        # ← already here
def stats():
    files = len(set(m["path"] for m in metadata_store))
    return {"total": text_index.ntotal, "files": files}

#-- new route to list ingested files and their chunk counts----
@app.get("/files")
def list_files():
    file_map = {}

    for m in metadata_store:
        fname = m["path"]
        file_map.setdefault(fname, 0)
        file_map[fname] += 1

    return [
        {"file": f, "chunks": c}
        for f, c in file_map.items()
    ]






@app.get("/document/{filename}")
def get_document(filename: str):
    if not filename:
        return {"error": "Filename is missing"}
    safe_name = Path(filename).name   # removes ../ attacks
    file_path = os.path.join(UPLOAD_DIR, str(safe_name))

    if os.path.exists(file_path):
        return FileResponse(file_path)

    return {"error": "File not found"}
def rewrite_query_with_memory(question: str, model_choice: str, session: str = "default"):
    pronouns = [" it ", " its ", " they ", " them ", " their ", " this ", " that "]
    q_lower = " " + question.lower() + " "
    has_pronoun = any(p in q_lower for p in pronouns)

    turns = get_turns(session)   # ← SQLite instead of conversation_memory

    if model_choice == "fast":
        if has_pronoun and turns:
            last_q = turns[-1]["question"]
            return f"{question} (regarding: {last_q})"
        return question

    if not has_pronoun:
        return question

    if not turns:
        return question

    current_vec = embed(question)
    best_q = None
    best_sim = 0

    for turn in turns:
        past_vec = turn.get("embedding")
        if past_vec is None:
            continue
        sim = cosine_similarity(current_vec, past_vec)
        if sim > best_sim:
            best_sim = sim
            best_q = turn["question"]

    if best_sim < 0.3:
        clear_session(session)
        return question

    rewrite_prompt = f"""
You rewrite follow-up questions into fully standalone questions.
Previous Question: {best_q}
Current Question: {question}
Return ONLY the rewritten question.
"""
    rewritten = call_llm(rewrite_prompt, model_choice)
    if not isinstance(rewritten, str):
        rewritten = str(rewritten)
    if rewritten and len(rewritten.strip()) > 5:
        return rewritten.strip()
    return question



def build_linked_sources(retrieved_docs):
    # Separate by modality using unique file paths only
    text_files  = list(dict.fromkeys(
        d.get("path") for d in retrieved_docs
        if d.get("type") not in {"image"} and not str(d.get("path","")).endswith((".mp3",".wav",".m4a",".png",".jpg",".jpeg"))
    ))
    audio_files = list(dict.fromkeys(
        d.get("path") for d in retrieved_docs
        if str(d.get("path","")).endswith((".mp3",".wav",".m4a"))
    ))
    image_files = list(dict.fromkeys(
        d.get("path") for d in retrieved_docs
        if d.get("type") == "image" or str(d.get("path","")).endswith((".png",".jpg",".jpeg"))
    ))

    # Helper: get best timestamp for an audio file from retrieved docs
    def get_timestamp(audio_path):
        for d in retrieved_docs:
            if d.get("path") == audio_path and d.get("timestamp") is not None:
                return d["timestamp"]
        return None

    # Helper: get best page for a text file
    def get_page(text_path):
        for d in retrieved_docs:
            if d.get("path") == text_path and d.get("page") is not None:
                return d["page"]
        return None

    links = []

    for t in text_files:
        for a in audio_files:
            links.append({
                "type": "text↔audio",
                "text_source": t,
                "text_page": get_page(t),
                "audio_source": a,
                "audio_timestamp": get_timestamp(a),
            })

    for t in text_files:
        for img in image_files:
            links.append({
                "type": "text↔image",
                "text_source": t,
                "text_page": get_page(t),
                "image_source": img,
            })

    for a in audio_files:
        for img in image_files:
            links.append({
                "type": "audio↔image",
                "audio_source": a,
                "audio_timestamp": get_timestamp(a),
                "image_source": img,
            })

    return links

def expand_query(question: str) -> str:
    action_words = [
        "explain ", "describe ", "tell me about ",
        "elaborate on ", "give details on ",
        "give me details about ", "summarize ",
        "summarise ", "walk me through ",
        "what is ", "what are ", "what was ",
        "how is ", "how does ", "how do ",
        "define ", "what do you mean by ",
        "can you explain ", "can you describe ",
    ]
    q_lower = question.lower().strip()
    for word in action_words:
        if q_lower.startswith(word):
            stripped = question[len(word):].strip()
            if len(stripped) > 2:
                return stripped
    return question
# ================= QUERY =================


@app.post("/query")
async def query(
    q: str = Form(...),
    model: str = Form("fast"),
    k: int = Form(20),
    session: str = Form("default"),
    stream: bool = Form(False),   # ← frontend sets True for streaming
):
    model = model.strip().lower()
    q_vec = embed(q)

    # ── 1. SEMANTIC CACHE LOOKUP (Strict threshold: 0.90) ────────
    cached = lookup_semantic_cache(q_vec, model=model, threshold=SEMANTIC_CACHE_THRESHOLD)
    if cached:
        print(f"⚡ [CACHE HIT] Similarity: {cached['similarity']} >= {SEMANTIC_CACHE_THRESHOLD} | Query: '{q}' matched '{cached['matched_query']}'")
        cached_ans = cached["answer"]
        cached_cits = cached["citations"]
        cached_conf = cached["confidence"]

        # Append to conversational memory so multi-turn context remains continuous
        append_turn(
            question=q, answer=cached_ans, embedding=q_vec,
            file=cached_cits[0]["path"] if cached_cits else None,
            session=session
        )

        if stream:
            async def cache_stream():
                yield json.dumps({
                    "type": "meta",
                    "confidence": cached_conf,
                    "chunks_used": len(cached_cits),
                    "citations": convert_numpy(cached_cits),
                    "trace": [],
                    "linked_sources": [],
                    "cached": True,
                    "cache_similarity": cached["similarity"],
                }) + "\n"
                yield json.dumps({"type": "token", "text": cached_ans}) + "\n"
                yield json.dumps({"type": "done"}) + "\n"
            return StreamingResponse(cache_stream(), media_type="text/event-stream")

        return convert_numpy({
            "answer": cached_ans,
            "citations": cached_cits,
            "confidence": cached_conf,
            "chunks_used": len(cached_cits),
            "trace": [],
            "linked_sources": [],
            "cached": True,
            "cache_similarity": cached["similarity"]
        })

    # ── 2. RETRIEVAL & QUERY EXPANSION ─────────────────────────
    rewritten_q = rewrite_query_with_memory(q, model, session)
    print(f"\n🧠 Original: {q}\n🔎 Rewritten: {rewritten_q}")

    search_q = expand_query(rewritten_q)
    print(f"🔍 Search query: {search_q}")
    docs = hybrid_search(search_q, k=k)

    # ── 3. STRICT RETRIEVAL THRESHOLD (>= 0.50) ────────────────
    docs = [d for d in docs if float(d.get("score", 0)) >= MIN_RETRIEVAL_SCORE]
    print(f"After strict threshold filter ({MIN_RETRIEVAL_SCORE}): {len(docs)} docs remaining")
    for d in docs:
        print(f"  → {d.get('path')} | page {d.get('page')} | score {d.get('score', 0):.3f} | text: {d.get('snippet','')[:60]}")
    docs = sorted(docs, key=lambda x: x.get("score", 0), reverse=True)

    if detect_topic_shift_by_file(docs, session):
        clear_session(session)

    if not docs:
        if stream:
            async def empty_stream():
                yield json.dumps({"type": "meta", "confidence": 0, "chunks_used": 0, "citations": [], "trace": [], "linked_sources": [], "cached": False}) + "\n"
                yield json.dumps({"type": "token", "text": "The answer is not present in the provided sources."}) + "\n"
                yield json.dumps({"type": "done"}) + "\n"
            return StreamingResponse(empty_stream(), media_type="text/event-stream")
        return {"answer": "The answer is not present in the provided sources.", "citations": [], "confidence": 0, "chunks_used": 0, "cached": False}

    if model in ["mistral", "llama3"]:
        docs = rerank_with_cross_encoder(q, docs)
    docs = docs[:6]
    retrieved_docs = docs.copy()
    docs = build_semantic_section(docs, window=2)
    context, source_map = build_context(docs)
    memory_context = build_memory_context(session)
    sources = [
        {"num": num, "path": path, "page": page}
        for (path, page), num in sorted(source_map.items(), key=lambda x: x[1])
    ]

    if model in ["mistral", "llama3"]:
        scores = [d.get("ce_score", d.get("score", 0)) for d in docs]
    else:
        scores = [d.get("score", 0) for d in docs]

    avg = sum(scores) / len(scores) if scores else 0
    if len(docs) < 2: avg *= 0.85
    if np.var(scores) > 0.15: avg *= 0.9
    confidence = round(min(max((avg + 10) / 20 * 100, 0), 100), 1) if model in ["mistral", "llama3"] else round(avg * 100, 1)

    trace = [
        {"file": d.get("path"), "page": d.get("page"), "chunk_id": d.get("chunk_id"),
         "score": round(d.get("score", d.get("llm_score", 0)), 3)}
        for d in docs
    ]

    # ── 4. STRICT CONFIDENCE GATE (< 50.0% REFUSAL) ────────────
    if confidence < MIN_CONFIDENCE_THRESHOLD:
        print(f"⚠️ Confidence {confidence}% is below strict threshold {MIN_CONFIDENCE_THRESHOLD}% → refusing to answer.")
        refusal_ans = "The answer is not present in the provided sources."
        if stream:
            async def low_conf_stream():
                yield json.dumps({"type": "meta", "confidence": confidence, "chunks_used": len(docs), "citations": convert_numpy(retrieved_docs), "trace": convert_numpy(trace), "linked_sources": [], "cached": False}) + "\n"
                yield json.dumps({"type": "token", "text": refusal_ans}) + "\n"
                yield json.dumps({"type": "done"}) + "\n"
            return StreamingResponse(low_conf_stream(), media_type="text/event-stream")
        return convert_numpy({
            "answer": refusal_ans,
            "citations": retrieved_docs,
            "confidence": confidence,
            "chunks_used": len(docs),
            "trace": trace,
            "linked_sources": [],
            "cached": False,
        })

    if model == "fast":
        context = " ".join((memory_context + context).split()[:600])
        memory_context = ""
    else:
        context = " ".join(context.split()[:1200])

    prompt = build_list_prompt(memory_context + context, q) if is_list_question(q) else build_prompt(memory_context + context, q, sources)

    model_name = {"fast": "mistral:latest", "mistral": "mistral:latest", "llama3": "llama3:latest"}.get(model, "mistral:latest")

    # ── STREAMING MODE ──────────────────────────────────────────
    if stream:
        async def token_stream():
            # 1. Send metadata first so frontend can show citations immediately
            meta = {
                "type": "meta",
                "confidence": confidence,
                "chunks_used": len(docs),
                "citations": convert_numpy(retrieved_docs),
                "trace": convert_numpy(trace),
                "linked_sources": convert_numpy(build_linked_sources(retrieved_docs)),
                "cached": False,
            }
            yield json.dumps(meta) + "\n"

            # 2. Stream tokens from Ollama
            full_answer = ""
            try:
                resp = requests.post(
                    OLLAMA_URL,
                    json={
                        "model": model_name,
                        "prompt": prompt,
                        "stream": True,
                        "options": {"temperature": 0.1, "num_predict": 512},
                    },
                    stream=True,
                    timeout=180,
                )
                for line in resp.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line.decode("utf-8"))
                            token = chunk.get("response", "")
                            if token:
                                full_answer += token
                                yield json.dumps({"type": "token", "text": token}) + "\n"
                        except:
                            pass
            except Exception as e:
                yield json.dumps({"type": "token", "text": "LLM error: " + str(e)}) + "\n"

            # 3. Save to memory and semantic cache after full answer collected
            final_ans = full_answer.strip()
            append_turn(
                question=q, answer=final_ans,
                embedding=q_vec,
                file=docs[0]["path"] if docs else None,
                session=session,
            )
            store_semantic_cache(
                query=q, query_embedding=q_vec, answer=final_ans,
                citations=retrieved_docs, confidence=confidence, model=model
            )
            yield json.dumps({"type": "done"}) + "\n"

        return StreamingResponse(token_stream(), media_type="text/event-stream")

    # ── NON-STREAMING MODE ──────────────────────────────────────
    raw_answer = call_llm(prompt, model)
    if is_list_question(q):
        items = extract_list_items(raw_answer)
        answer = "\n\n".join("• " + i for i in items) if items else "The answer is not present in the provided sources."
    else:
        answer = raw_answer.strip() if isinstance(raw_answer, str) else str(raw_answer)

    # ── Save turn to SQLite and Semantic Cache ──
    append_turn(question=q, answer=answer, embedding=q_vec,
                file=docs[0]["path"] if docs else None, session=session)
    store_semantic_cache(
        query=q, query_embedding=q_vec, answer=answer,
        citations=retrieved_docs, confidence=confidence, model=model
    )

    trace = [{"file":d.get("path"),"page":d.get("page"),"chunk_id":d.get("chunk_id"),
              "score":round(d.get("score",d.get("llm_score",0)),3)} for d in docs]

    return convert_numpy({
        "answer": answer, "citations": retrieved_docs,
        "confidence": confidence, "chunks_used": len(docs),
        "trace": trace, "linked_sources": build_linked_sources(retrieved_docs),
        "cached": False,
    })


@app.post("/reset")
def reset_memory(session: str = Form("default")):
    clear_session(session)
    return {"status": "Conversation memory cleared"}

@app.post("/image_query")
async def image_query(file: UploadFile = File(...)):
    if not file.filename:
        return {"error": "Invalid file"}
    path = os.path.join(UPLOAD_DIR, file.filename)

    with open(path, "wb") as f:
        f.write(await file.read())

    results = []

    # -------- 1. IMAGE → IMAGE SEARCH --------
    if image_index.ntotal > 0:
        qvec = embed_image(path)
        qvec_arr = np.array([qvec]).astype("float32")
        distances, indices = image_index.search(qvec_arr, 5) # type: ignore

        for score, idx in zip(distances[0], indices[0]):
            if idx < len(image_metadata):
                results.append({
                    "score": float(score),
                    "path": image_metadata[idx]["path"],
                    "preview": image_metadata[idx]["preview"],
                    "type": "image"
                })
    # -------- 2. IMAGE → TEXT/AUDIO SEARCH (NEW) --------
    # Uses the image's CLIP embedding to query the text index
    text_results = hybrid_search(query="", k=5, image_path=path)

    for r in text_results:
        if r.get("type") != "image":          # exclude image results (already above)
            results.append({
                "score": float(r.get("score", 0)),
                "path": r.get("path"),
                "snippet": r.get("snippet", ""),
                "page": r.get("page"),
                "chunk_id": r.get("chunk_id"),
                "timestamp": r.get("timestamp"),
                "type": r.get("type", "text")
            })

    # Sort all results by score descending
    results = sorted(results, key=lambda x: x["score"], reverse=True)

    return {"results": results}

# ================= TRANSCRIBE (Voice Input) =================
@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    import tempfile

    suffix = ".webm"
    if file.content_type:
        if "ogg" in file.content_type:
            suffix = ".ogg"
        elif "wav" in file.content_type:
            suffix = ".wav"
        elif "mp4" in file.content_type:
            suffix = ".mp4"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = whisper_model.transcribe(tmp_path, language="en")
        text = str(result.get("text", "")).strip()
        print(f"[Transcribe] '{text}'")
        return {"text": text}
    except Exception as e:
        print(f"[Transcribe] Error: {e}")
        return {"text": "", "error": str(e)}
    finally:
        try:
            os.remove(tmp_path)
        except:
            pass


@app.get("/query")
def query_get():
    return {"error": "Use POST /query with form data: q, model, k"}


# ================= FILE DELETE =================
# Add this new endpoint to main.py:

@app.delete("/file/{filename}")
def delete_file(filename: str):
    global bm25, text_index, metadata_store, chunk_hashes

    safe = Path(filename).name

    # 1. Remove from disk
    fp = os.path.join(UPLOAD_DIR, safe)
    if os.path.exists(fp):
        os.remove(fp)

    # 2. Find indices to remove
    indices_to_remove = [
        i for i, m in enumerate(metadata_store)
        if m.get("path") == safe
    ]

    if not indices_to_remove:
        return {"status": "not_found", "file": safe}

    # 3. Rebuild metadata without deleted entries
    keep_indices  = [i for i in range(len(metadata_store)) if i not in set(indices_to_remove)]
    metadata_store[:] = [metadata_store[i] for i in keep_indices]

    # 4. Rebuild FAISS index from scratch (simplest correct approach)
    text_index = faiss.IndexFlatIP(TEXT_DIM)  # type: ignore
    for m in metadata_store:
        vec = embed(m["text"])
        text_index.add(np.array([vec], dtype="float32"))  # type: ignore

    # 5. Rebuild BM25
    bm25_corpus.clear()
    bm25_chunks.clear()
    for i, m in enumerate(metadata_store):
        bm25_corpus.append(m["text"].lower().split())
        bm25_chunks.append(i)
    bm25 = BM25Okapi(bm25_corpus) if bm25_corpus else None

    # 6. Rebuild chunk hashes
    chunk_hashes.clear()
    for m in metadata_store:
        chunk_hashes.add(hashlib.sha256(str(m.get("text", "")).strip().encode()).hexdigest())

    # 7. Handle image index
    img_keep = [im for im in image_metadata if im.get("path") != safe]
    image_metadata[:] = img_keep

    # 8. Persist
    save_database()
    clear_semantic_cache()

    print(f"🗑 Deleted '{safe}': removed {len(indices_to_remove)} chunks")
    return {
        "status": "deleted",
        "file": safe,
        "chunks_removed": len(indices_to_remove),
    }

# ================= SEMANTIC SEARCH =================
@app.post("/search")
async def semantic_search(
    q: str = Form(...),
    k: int = Form(10),
    file_filter: str = Form(""),
):
    """
    Returns raw matching chunks with scores — no LLM involved.
    Fast keyword + semantic search across all indexed documents.
    """
    if text_index.ntotal == 0:
        return {"results": [], "query": q, "total": 0}
 
    # Run hybrid search
    results = hybrid_search(q, k=k, file_filter=file_filter or None)
 
    # Sort by score
    results = sorted(results, key=lambda x: x.get("score", 0), reverse=True)
 
    # Format response
    output = []
    for r in results:
        output.append({
            "text":     r.get("snippet", ""),
            "path":     r.get("path", ""),
            "page":     r.get("page"),
            "chunk_id": r.get("chunk_id"),
            "score":    round(float(r.get("score", 0)), 4),
            "type":     r.get("type", "text"),
        })
 
    return {"results": output, "query": q, "total": len(output)}

# ================= SEMANTIC CACHE ROUTES =================
@app.get("/cache/stats")
async def cache_stats():
    """Returns metrics about semantic cache (total entries, hit counts, top queries)."""
    return get_semantic_cache_stats()

@app.post("/cache/clear")
async def cache_clear():
    """Manually flushes all entries in the semantic cache."""
    clear_semantic_cache()
    return {"status": "success", "message": "Semantic cache cleared"}
 