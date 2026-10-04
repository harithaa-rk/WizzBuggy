
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

RULES:
1) Extract exact information from the context
2) Do NOT hallucinate
3) If missing, say:
"The answer is not present in the provided sources."
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
def detect_topic_shift_by_file(current_docs):
    if not conversation_memory:
        return False

    if not current_docs:
        return False

    current_file = current_docs[0]["path"]

    last_file = conversation_memory[-1].get("file")

    if last_file and current_file != last_file:
        print("🔄 File changed → Topic shift detected")
        return True

    return False

# ================= SMART CHUNKING =================
import re

def create_chunks(text, chunk_size=120, overlap=40):
    sentences = re.split(r'(?<=[.!?]) +', text)

    chunks = []
    current = []

    for sentence in sentences:
        current.append(sentence)

        if len(" ".join(current).split()) >= chunk_size:
            chunks.append(" ".join(current))
            current = current[-overlap:]

    if current:
        chunks.append(" ".join(current))

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
            faiss_scores[idx] = float(score)

    # Normalize FAISS scores (cosine similarity already -1 to 1)
    if faiss_scores:
        max_f = max(faiss_scores.values())
        min_f = min(faiss_scores.values())

        for idx in faiss_scores:
            if max_f - min_f > 0:
                faiss_scores[idx] = (faiss_scores[idx] - min_f) / (max_f - min_f)
            else:
                faiss_scores[idx] = 0.0

    # ---------------- BM25 SEARCH ----------------
    bm25_scores = {}

    if bm25:
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)

        # Get top-k BM25 results
        top_bm25_idx = np.argsort(scores)[::-1][:k]

        for pos in top_bm25_idx:
            meta_idx = bm25_chunks[pos]  # translate BM25 position → metadata index
            bm25_scores[meta_idx] = float(scores[pos])

        # Normalize BM25 scores
        if bm25_scores:
            max_b = max(bm25_scores.values())
            min_b = min(bm25_scores.values())

            for idx in bm25_scores:
                if max_b - min_b > 0:
                    bm25_scores[idx] = (bm25_scores[idx] - min_b) / (max_b - min_b)
                else:
                    bm25_scores[idx] = 0.0

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
The context below is labelled with source numbers like [1], [2], etc.
You MUST cite sources inline in your answer using these numbers, e.g. "The project started in 2022 [1]."

CITATION RULES:
- Place the citation number immediately after the fact it supports, e.g. "Sales rose 20% [2]."
- Use multiple citations if a fact comes from more than one source, e.g. [1][3].
- Every factual sentence must have at least one citation.
- If information is missing from all sources, say exactly:
  "The answer is not present in the provided sources."

SOURCES:
{source_legend}

CONTEXT:
{context}

QUESTION:
{question}

FINAL ANSWER (with inline [N] citations):
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

Rules:
- Extract list items EXACTLY as written
- Do NOT summarize
- Return each item on a new line
- Do NOT add explanations
- Only return the list

CONTEXT:
{context}

QUESTION:
{question}

LIST:
"""


def build_memory_context():
    if not conversation_memory:
        return ""
    
    memory_text = "Previous Conversation:\n"
    for turn in conversation_memory[-MAX_MEMORY_TURNS:]:
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
        threshold = 0.15 if d.get("type") == "image" else 0.50
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
            #-------- EXTRACT TEXT --------
            page = doc.load_page(page_index)
            text = page.get_text("text") or ""

            if not isinstance(text, str):
                text = str(text)
            print("Extracted text length:", len(text))
            if not text.strip():
                print("⚠️ No text found, using OCR...")
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                text = pytesseract.image_to_string(img, config='--oem 3 --psm 6')

                # ---- TABLE EXTRACTION ----
            try:
                tables = []
                if hasattr(page, "find_tables"):
                    tables = page.find_tables() or [] # type: ignore

                for table in tables:
                    table_data = table.extract()

                    for row in table_data:
                        row_text = " ".join(str(cell) for cell in row if cell)

                        if len(row_text.strip()) > 10:
                            text += "\n" + row_text

            except Exception as e:
                print("Table extraction error:", e)


            chunks = create_chunks(text)

            for chunk in chunks:
                if add_to_index(
                    text=chunk,
                    filename=file.filename,
                    page=page_index + 1,
                    chunk_id=global_chunk_id
                ):
                    ingested_chunks += 1
                    global_chunk_id += 1
                
            

    # -------- DOCX --------
    elif ext.endswith(".docx"):
        text = docx2txt.process(save_path) or ""

        # ── Remove repeated paragraphs before chunking ──
        lines = text.split("\n")
        seen_lines = set()
        clean_lines = []
        for line in lines:
            stripped = line.strip()
            # Keep empty lines for paragraph spacing
            if not stripped:
                clean_lines.append(line)
                continue
            # Only keep a line if we haven't seen it before
            if stripped not in seen_lines:
                seen_lines.add(stripped)
                clean_lines.append(line)

        clean_text = "\n".join(clean_lines)
        print(f"Original: {len(text)} chars → Cleaned: {len(clean_text)} chars")
        print(f"Last 300 chars of clean text: {clean_text[-300:]}")

        chunks = create_chunks(clean_text)
        chunk_id_counter = 0
        for chunk in chunks:
            if add_to_index(chunk, file.filename, chunk_id=chunk_id_counter):
                ingested_chunks += 1
                chunk_id_counter += 1

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


    # -------- AUDIO --------
    elif ext.endswith((".mp3", ".wav", ".m4a")):
        result = whisper_model.transcribe(save_path,word_timestamps=True)
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

        chunk_id_counter = 0

        for chunk in chunks:
            if add_to_index(chunk, file.filename, chunk_id=chunk_id_counter):
                ingested_chunks += 1
                chunk_id_counter += 1
        
        for segment in result.get("segments", []):
            if not isinstance(segment, dict):  # type: ignore
                continue
            text = segment.get("text", "")

            if not isinstance(text, str):
                text = str(text)

            text = text.strip()
            start = segment.get("start", 0)

            if not text:
                continue

            if add_to_index(text=text, filename=file.filename, chunk_id=chunk_id_counter):
                metadata_store[-1]["timestamp"] = start
                chunk_id_counter += 1

    else:
        return {"status": "Unsupported file"}

    # Save AFTER indexing
    save_database()
    global bm25
    if bm25_corpus:
        bm25 = BM25Okapi(bm25_corpus)

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

def rewrite_query_with_memory(question: str, model_choice: str):
    pronouns = [" it ", " its ", " they ", " them ", " their ", " this ", " that "]
    q_lower = " " + question.lower() + " "
    has_pronoun = any(p in q_lower for p in pronouns)

    # Fast mode: no LLM rewrite, but still do simple pronoun resolution
    if model_choice == "fast":
        if has_pronoun and conversation_memory:
            last_q = conversation_memory[-1]["question"]
            return f"{question} (regarding: {last_q})"
        return question

    # Smart/Llama modes: no pronoun → no rewrite needed
    if not has_pronoun:
        return question

    if not conversation_memory:
        return question

    # Find most similar past question via embedding
    current_vec = embed(question)
    best_q = None
    best_sim = 0

    for turn in conversation_memory:
        past_vec = turn.get("embedding")
        if past_vec is None:
            continue
        sim = cosine_similarity(current_vec, past_vec)
        if sim > best_sim:
            best_sim = sim
            best_q = turn["question"]

    if best_sim < 0.3:
        conversation_memory.clear()
        return question

    rewrite_prompt = f"""
You rewrite follow-up questions into fully standalone questions.

ONLY rewrite if the question depends on the previous topic.

Previous Question:
{best_q}

Current Question:
{question}

Rewrite the current question so it is fully self-contained.
Return ONLY the rewritten question.
If not dependent, return the original question.
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
# ================= QUERY =================

@app.post("/query")
async def query(q: str = Form(...), model: str = Form("fast"), k: int = Form(8)):

    model = model.strip().lower()

    # -------- MEMORY-AWARE QUERY REWRITE --------
    rewritten_q = rewrite_query_with_memory(q, model)
    print(f"\n🧠 Original: {q}")
    print(f"🔎 Rewritten: {rewritten_q}\n")

    # -------- RETRIEVAL --------
    docs = hybrid_search(rewritten_q, k=k)

    # -------- REMOVE WEAK MATCHES --------
    docs = [d for d in docs if float(d.get("score", 0)) > 0.50]

    # -------- SORT --------
    docs = sorted(docs, key=lambda x: x.get("score", 0), reverse=True)
    print("Retrieved docs:", len(docs))

    # -------- TOPIC SHIFT DETECTION --------
    if detect_topic_shift_by_file(docs):
        print("🔄 Topic shift detected → clearing memory")
        conversation_memory.clear()

    # -------- DOMINANT FILE FILTER --------
    if docs:
        # Find the highest score across all docs
        top_score = docs[0].get("score", 0)  # already sorted descending

        # Find which file owns the top score
        dominant_file = docs[0].get("path")

        # Keep dominant file always; keep others only if their BEST chunk
        # is within 10% of the top score (tight gap)
        file_best_score = {}
        for d in docs:
            p = d.get("path")
            file_best_score[p] = max(file_best_score.get(p, 0), d.get("score", 0))

        file_chunk_counts = Counter(d.get("path") for d in docs)
        
        docs = [
            d for d in docs
            if d.get("path") == dominant_file
            or (
                file_best_score.get(d.get("path"), 0) >= top_score * 0.92
                and file_chunk_counts.get(d.get("path"), 0) >= file_chunk_counts.get(dominant_file, 1)
            )
        ]
    if not docs:
        return {
            "answer": "The answer is not present in the provided sources.",
            "citations": [],
            "confidence": 0,
            "chunks_used": 0
        }

    # -------- RERANK (SMART MODE ONLY) --------
    if model in ["mistral", "llama3"]:
        docs = rerank_with_cross_encoder(q, docs)

    docs = docs[:6]
    retrieved_docs = docs.copy()

    # -------- SEMANTIC WINDOW --------
    docs = build_semantic_section(docs, window=1)

    # -------- BUILD CONTEXT + SOURCE MAP --------
    context, source_map = build_context(docs)   # ← tuple unpack
    memory_context = build_memory_context()

    # Build sources list for prompt legend
    sources = [
        {"num": num, "path": path, "page": page}
        for (path, page), num in sorted(source_map.items(), key=lambda x: x[1])  # type: ignore
    ]

    # -------- CONFIDENCE CALC --------
    if model in ["mistral", "llama3"]:
        scores = [d.get("ce_score", d.get("score", 0)) for d in docs]
    else:
        scores = [d.get("score", 0) for d in docs]

    if not scores:
        confidence = 0
    else:
        avg_score = sum(scores) / len(scores)
        if len(docs) < 2:
            avg_score *= 0.85
        if np.var(scores) > 0.15:
            avg_score *= 0.9
        # Normalize ALL models to 0–100
        if model in ["mistral", "llama3"]:
            confidence = round(min(max((avg_score + 10) / 20 * 100, 0), 100), 1)
        else:
            confidence = round(avg_score * 100, 1)

    # -------- TOKEN SAFE TRUNCATION --------
    if model == "fast":
        combined = memory_context + context
        context = " ".join(combined.split()[:600])
        memory_context = ""
    else:
        context = " ".join(context.split()[:1200])

    # -------- PROMPT BUILD --------
    if is_list_question(q):
        prompt = build_list_prompt(memory_context + context, q)
    else:
        prompt = build_prompt(memory_context + context, q, sources)

    # -------- LLM CALL --------
    raw_answer = call_llm(prompt, model)

    # -------- LIST CLEANUP --------
    if is_list_question(q):
        items = extract_list_items(raw_answer)
        answer = "\n\n".join(f"• {i}" for i in items) if items else "The answer is not present in the provided sources."
    else:
        answer = clean_answer(raw_answer)

    # -------- MEMORY UPDATE --------
    conversation_memory.append({
        "question": q,
        "answer": answer,
        "embedding": embed(q),
        "file": docs[0]["path"] if docs else None
    })
    if len(conversation_memory) > MAX_MEMORY_TURNS:
        conversation_memory.pop(0)

    # -------- TRACE --------
    trace = [
        {
            "file": d.get("path"),
            "page": d.get("page"),
            "chunk_id": d.get("chunk_id"),
            "score": round(d.get("score", d.get("llm_score", 0)), 3)
        }
        for d in docs
    ]

    return convert_numpy({
        "answer": answer,
        "citations": retrieved_docs,
        "confidence": confidence,
        "chunks_used": len(docs),
        "trace": trace,
        "linked_sources": build_linked_sources(retrieved_docs) 
    })


@app.post("/reset")
def reset_memory():
    conversation_memory.clear()
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
    
@app.get("/query")
def query_get():
    return {"error": "Use POST /query with form data: q, model, k"}