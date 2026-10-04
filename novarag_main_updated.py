# ================= IMPORTS =================
import os
import uuid
import faiss
import whisper
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
import json
from PIL import Image
import fitz  # PyMuPDF
import docx2txt
import pytesseract
from rank_bm25 import BM25Okapi
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from fastapi.middleware.cors import CORSMiddleware
import hashlib
from fastapi.responses import FileResponse
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel
import torch
import requests
from collections import Counter
import re
import pickle

# ── SQLite memory (replaces in-memory list) ──────────────────────
from novarag_memory import append_turn, get_turns, clear_session, MAX_MEMORY_TURNS

# ================= DEVICE =================
device = "cuda" if torch.cuda.is_available() else "cpu"

# ================= MODELS =================
from sentence_transformers import SentenceTransformer, CrossEncoder

embed_model    = SentenceTransformer("all-MiniLM-L6-v2", device=device)
cross_encoder  = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
whisper_model  = whisper.load_model("base")

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to(device) #type: ignore
clip_model.eval()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ================= CLIP HELPERS =================
def embed_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = {k: v.to(device) for k, v in clip_processor(images=image, return_tensors="pt").items()} #type: ignore
    with torch.no_grad():
        out = clip_model.vision_model(**inputs)
        feat = clip_model.visual_projection(out.pooler_output).float()
    feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-10)
    return feat.detach().cpu().numpy().flatten()

def embed_text_clip(text):
    inputs = {k: v.to(device) for k, v in clip_processor(text=[text], return_tensors="pt", padding=True).items()} #type: ignore
    with torch.no_grad():
        out = clip_model.text_model(**inputs)
        feat = clip_model.text_projection(out.pooler_output).float()
    feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-10)
    return feat.detach().cpu().numpy().flatten()

# ================= SYSTEM PROMPT =================
SYSTEM_PROMPT = """
You are NovaRAG, a document-grounded assistant.
RULES:
1) Extract exact information from the context
2) Do NOT hallucinate
3) If missing, say: "The answer is not present in the provided sources."
"""

# ================= FASTAPI =================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

OLLAMA_URL = "http://localhost:11434/api/generate"
dimension  = 384

# ================= FAISS + METADATA =================
INDEX_FILE       = "faiss.index"
IMAGE_INDEX_FILE = "image.index"
META_FILE        = "meta.pkl"
TEXT_DIM, IMAGE_DIM = 384, 512

text_index  = faiss.read_index(INDEX_FILE)       if os.path.exists(INDEX_FILE)       else faiss.IndexFlatIP(TEXT_DIM)
_img_loaded = faiss.read_index(IMAGE_INDEX_FILE) if os.path.exists(IMAGE_INDEX_FILE) else None
image_index = (_img_loaded if (_img_loaded and _img_loaded.d == IMAGE_DIM)
               else faiss.IndexFlatIP(IMAGE_DIM))

if os.path.exists(META_FILE):
    with open(META_FILE, "rb") as f:
        _data = pickle.load(f)
    if isinstance(_data, dict):
        metadata_store = _data.get("text", [])
        image_metadata  = _data.get("image", [])
    else:
        metadata_store, image_metadata = _data, []
else:
    metadata_store, image_metadata = [], []

# ── BM25 ──
bm25_corpus, bm25_chunks = [], []
for i, item in enumerate(metadata_store):
    bm25_corpus.append(item["text"].lower().split())
    bm25_chunks.append(i)
bm25 = BM25Okapi(bm25_corpus) if bm25_corpus else None
print(f"✅ BM25 rebuilt: {len(bm25_corpus)} chunks")

chunk_hashes = set(
    hashlib.sha256(str(m.get("text","")).strip().encode()).hexdigest()
    for m in metadata_store
)

# ================= LLM =================
def call_llm(prompt: str, model_choice: str) -> str:
    model_name = {"fast": "mistral:latest", "mistral": "mistral:latest", "llama3": "llama3:latest"}.get(model_choice, "mistral:latest")
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": model_name, "prompt": prompt, "stream": True,
            "options": {"temperature": 0.1, "num_predict": 512}
        }, stream=True, timeout=180)
        full = ""
        for line in resp.iter_lines():
            if line:
                try: full += json.loads(line.decode()).get("response","")
                except: pass
        return full.strip()
    except Exception as e:
        print("Ollama error:", e)
        return "Local LLM not running."

# ================= HELPERS =================
def embed(text: str):
    if not isinstance(text, str): text = str(text)
    return np.array(embed_model.encode(text, normalize_embeddings=True), dtype="float32")

def cosine_similarity(v1, v2): return float(np.dot(v1, v2))

def is_list_question(q):
    return any(k in q.lower() for k in ["phases","steps","stages","types","categories","lifecycle","life cycle","components","modules"])

def extract_list_items(text):
    if not isinstance(text, str): text = str(text)
    lines = re.split(r"\n|,|•|;", text)
    seen, final = set(), []
    for line in lines:
        line = re.sub(r"\(.*?\)","",line.strip("•-–—: ").strip()).strip()
        if 3 < len(line) and len(line.split()) <= 20 and re.search(r"[a-zA-Z]",line):
            key = line.lower()
            if key not in seen:
                seen.add(key); final.append(line)
    return final

def create_chunks(text, chunk_size=120, overlap=40):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks, current = [], []
    for s in sentences:
        current.append(s)
        if len(" ".join(current).split()) >= chunk_size:
            chunks.append(" ".join(current))
            current = current[-overlap:]
    if current: chunks.append(" ".join(current))
    return chunks

def add_to_index(text, filename, page=None, chunk_id=None):
    global bm25
    if not isinstance(text, str): text = str(text)
    text = text.strip()
    h = hashlib.sha256(text.encode()).hexdigest()
    if h in chunk_hashes: return False
    vec = embed(text)
    if vec is None or len(vec) != dimension: return False
    chunk_hashes.add(h)
    text_index.add(np.array([vec], dtype="float32")) #type: ignore
    metadata_store.append({"id": str(uuid.uuid4()), "text": text, "path": filename, "page": page, "chunk_id": chunk_id})
    bm25_corpus.append(text.lower().split())
    bm25_chunks.append(len(metadata_store)-1)
    return True

def hybrid_search(query, k=5, image_path=None, file_filter=None):
    if text_index.ntotal == 0: return []
    qvec = embed_image(image_path) if image_path else embed(query)
    distances, labels = text_index.search(np.array([qvec], dtype="float32"), k) #type: ignore

    image_results = []
    if image_index.ntotal > 0:
        ivec = embed_image(image_path) if image_path else embed_text_clip(query)
        id2, il2 = image_index.search(np.array([ivec], dtype="float32"), k) #type: ignore
        for score, idx in zip(id2[0], il2[0]):
            if idx < len(image_metadata):
                image_results.append({"snippet": f"[IMAGE: {image_metadata[idx]['path']}]",
                    "path": image_metadata[idx]["path"], "preview": image_metadata[idx]["preview"],
                    "score": float(score), "type": "image"})

    faiss_scores = {idx: float(sc) for sc, idx in zip(distances[0], labels[0]) if 0 <= idx < len(metadata_store)}
    if faiss_scores:
        mx, mn = max(faiss_scores.values()), min(faiss_scores.values())
        faiss_scores = {i: (v-mn)/(mx-mn) if mx-mn>0 else 0.0 for i,v in faiss_scores.items()}

    bm25_scores = {}
    if bm25:
        scores = bm25.get_scores(query.lower().split())
        for pos in np.argsort(scores)[::-1][:k]:
            bm25_scores[bm25_chunks[pos]] = float(scores[pos])
        if bm25_scores:
            mx, mn = max(bm25_scores.values()), min(bm25_scores.values())
            bm25_scores = {i: (v-mn)/(mx-mn) if mx-mn>0 else 0.0 for i,v in bm25_scores.items()}

    alpha = 0.5 if len(query.split())<=3 else 0.75
    combined = {idx: alpha*faiss_scores.get(idx,0)+(1-alpha)*bm25_scores.get(idx,0)
                for idx in set(faiss_scores)|set(bm25_scores)}

    results = []
    for idx, sc in sorted(combined.items(), key=lambda x:x[1], reverse=True)[:k]:
        item = metadata_store[idx]
        results.append({"idx":idx, "snippet":item["text"], "path":item["path"], "page":item["page"],
            "chunk_id":item.get("chunk_id"), "score":sc,
            "type": item.get("type") if item.get("type") in {"text","table","image","audio"} else "text"})
    if file_filter: results = [r for r in results if r["path"]==file_filter]
    return results + image_results

def rerank_with_cross_encoder(question, docs):
    text_docs = [d for d in docs if d.get("type")!="image"]
    image_docs = [d for d in docs if d.get("type")=="image"]
    if not text_docs: return image_docs
    scores = cross_encoder.predict([(question, d["snippet"]) for d in text_docs])
    for i,d in enumerate(text_docs): d["ce_score"] = float(scores[i])
    return sorted(text_docs, key=lambda x:x["ce_score"], reverse=True)[:5] + image_docs

def build_semantic_section(docs, window=1):
    selected = []
    for d in docs:
        if d.get("type")=="image" or d.get("chunk_id") is None:
            selected.append(d); continue
        base_id, file = d["chunk_id"], d["path"]
        for meta in metadata_store:
            if meta["path"]==file and meta.get("chunk_id") is not None and abs(meta["chunk_id"]-base_id)<=window:
                selected.append({"snippet":meta["text"],"path":meta["path"],"page":meta.get("page"),
                    "chunk_id":meta.get("chunk_id"),"score":d.get("score",0),"type":meta.get("type","text")})
    unique = {(x.get("snippet"),x.get("path"),x.get("page")):x for x in selected}
    return sorted(unique.values(), key=lambda x:(x.get("path",""),x.get("chunk_id") or 0))

def build_context(docs):
    parts, filtered_docs = [], []
    for d in docs:
        score = float(d.get("score",0)) if d.get("score") else 0
        if score > (0.15 if d.get("type")=="image" else 0.50):
            filtered_docs.append(d)
    source_map, num = {}, 1
    for d in filtered_docs:
        key = (d.get("path"), d.get("page"))
        if key not in source_map:
            source_map[key] = num; num += 1
    for d in filtered_docs:
        key = (d.get("path"),d.get("page"))
        src = f"[{source_map[key]}] [{d.get('path','Unknown')}]"
        if d.get("page"): src += f" (page {d['page']})"
        parts.append(f"{src}\n{d.get('snippet','')}")
    return "\n\n".join(parts), source_map

def build_prompt(context, question, sources):
    legend = "\n".join(f"[{s['num']}] {s['path']}" + (f" (page {s['page']})" if s.get('page') else "") for s in sources)
    return f"""
{SYSTEM_PROMPT}
Context is labelled [1],[2],… Cite inline: "fact [1]."
SOURCES:\n{legend}\nCONTEXT:\n{context}\nQUESTION:\n{question}\nFINAL ANSWER (with inline [N] citations):
"""

def build_list_prompt(context, question):
    return f"Extract list items EXACTLY as written. One per line.\nCONTEXT:\n{context}\nQUESTION:\n{question}\nLIST:"

def build_memory_context(session="default"):
    turns = get_turns(session)
    if not turns: return ""
    mem = "Previous Conversation:\n"
    for t in turns:
        mem += f"User: {t['question']}\nAssistant: {t['answer']}\n"
    return mem + "\n"

def detect_topic_shift_by_file(current_docs, session="default"):
    turns = get_turns(session)
    if not turns or not current_docs: return False
    last_file = turns[-1].get("file")
    return bool(last_file and current_docs[0]["path"] != last_file)

def rewrite_query_with_memory(question, model_choice, session="default"):
    pronouns = [" it "," its "," they "," them "," their "," this "," that "]
    q_lower = " " + question.lower() + " "
    has_pronoun = any(p in q_lower for p in pronouns)
    if model_choice == "fast":
        turns = get_turns(session)
        if has_pronoun and turns:
            return f"{question} (regarding: {turns[-1]['question']})"
        return question
    if not has_pronoun: return question
    turns = get_turns(session)
    if not turns: return question
    current_vec = embed(question)
    best_q, best_sim = None, 0
    for t in turns:
        if t.get("embedding") is not None:
            sim = cosine_similarity(current_vec, t["embedding"])
            if sim > best_sim:
                best_sim, best_q = sim, t["question"]
    if best_sim < 0.3:
        clear_session(session); return question
    rewritten = call_llm(f"Rewrite as standalone question.\nPrev: {best_q}\nCurrent: {question}\nReturn ONLY rewritten question.", model_choice)
    return rewritten.strip() if rewritten and len(rewritten.strip())>5 else question

def build_linked_sources(retrieved_docs):
    def ends(p,exts): return str(p or "").endswith(tuple(exts))
    text_files  = list(dict.fromkeys(d.get("path") for d in retrieved_docs
        if d.get("type")!="image" and not ends(d.get("path"),(".mp3",".wav",".m4a",".png",".jpg",".jpeg"))))
    audio_files = list(dict.fromkeys(d.get("path") for d in retrieved_docs if ends(d.get("path"),(".mp3",".wav",".m4a"))))
    image_files = list(dict.fromkeys(d.get("path") for d in retrieved_docs
        if d.get("type")=="image" or ends(d.get("path"),(".png",".jpg",".jpeg"))))
    def get_ts(p):
        for d in retrieved_docs:
            if d.get("path")==p and d.get("timestamp") is not None: return d["timestamp"]
    def get_pg(p):
        for d in retrieved_docs:
            if d.get("path")==p and d.get("page") is not None: return d["page"]
    links = []
    for t in text_files:
        for a in audio_files: links.append({"type":"text↔audio","text_source":t,"text_page":get_pg(t),"audio_source":a,"audio_timestamp":get_ts(a)})
        for img in image_files: links.append({"type":"text↔image","text_source":t,"text_page":get_pg(t),"image_source":img})
    for a in audio_files:
        for img in image_files: links.append({"type":"audio↔image","audio_source":a,"audio_timestamp":get_ts(a),"image_source":img})
    return links

def convert_numpy(obj):
    if isinstance(obj, dict): return {k:convert_numpy(v) for k,v in obj.items()}
    if isinstance(obj, list): return [convert_numpy(i) for i in obj]
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return obj

def save_database():
    faiss.write_index(text_index, INDEX_FILE)
    faiss.write_index(image_index, IMAGE_INDEX_FILE)
    with open(META_FILE,"wb") as f:
        pickle.dump({"text":metadata_store,"image":image_metadata}, f)
    print("💾 Database saved")

# ================= INGEST =================
@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    if not file.filename: return {"status":"Invalid file"}
    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(save_path,"wb") as f: f.write(await file.read())
    ingested_chunks, ext = 0, file.filename.lower()

    if ext.endswith(".pdf"):
        doc = fitz.open(save_path)
        global_chunk_id = 0
        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            text = str(page.get_text("text") or "") #type: ignore
            if not text.strip():
                pix = page.get_pixmap()
                img = Image.frombytes("RGB",(pix.width,pix.height),pix.samples)
                text = pytesseract.image_to_string(img, config='--oem 3 --psm 6')
            try:
                for table in (page.find_tables() or []): #type: ignore
                    for row in table.extract():
                        row_text = " ".join(str(c) for c in row if c)
                        if len(row_text.strip())>10: text += "\n" + row_text
            except: 
                pass
            for chunk in create_chunks(text):
                if add_to_index(chunk, file.filename, page=page_index+1, chunk_id=global_chunk_id):
                    ingested_chunks += 1; global_chunk_id += 1

    elif ext.endswith(".docx"):
        text = docx2txt.process(save_path) or ""
        for i, chunk in enumerate(create_chunks(text)):
            if add_to_index(chunk, file.filename, chunk_id=i): ingested_chunks += 1

    elif ext.endswith((".png",".jpg",".jpeg")):
        img = Image.open(save_path)
        text = pytesseract.image_to_string(img) or ""
        img_vec = embed_image(save_path)
        image_index.add(np.array([img_vec], dtype="float32")) #type: ignore
        image_metadata.append({"path":file.filename,"preview":save_path,"type":"image"})
        add_to_index("image showing "+text[:150], file.filename, chunk_id=None)
        metadata_store[-1]["type"] = "image"
        metadata_store[-1]["preview"] = save_path

    elif ext.endswith((".mp3",".wav",".m4a")):
        result = whisper_model.transcribe(save_path, word_timestamps=True)
        text = str(result.get("text",""))
        for pat in [r"like (and )?share this video.*",r"don't forget to subscribe.*",r"subscribe.*",
                    r"thanks for watching.*",r"see you in the next video.*",r"thank you[.!]?\s*$"]:
            text = re.sub(pat,"",text,flags=re.IGNORECASE|re.DOTALL)
        cid = 0
        for chunk in create_chunks(text):
            if add_to_index(chunk, file.filename, chunk_id=cid): ingested_chunks += 1; cid += 1
        for seg in result.get("segments",[]):
            if not isinstance(seg,dict): continue
            t = seg.get("text","").strip()
            if t and add_to_index(t, file.filename, chunk_id=cid):
                metadata_store[-1]["timestamp"] = seg.get("start",0); cid += 1
    else:
        return {"status":"Unsupported file"}

    save_database()
    global bm25
    if bm25_corpus: bm25 = BM25Okapi(bm25_corpus)
    return {"status":"File indexed successfully","chunks":ingested_chunks}

# ================= ROUTES =================
@app.get("/stats")
def stats():
    return {"total":text_index.ntotal,"files":len(set(m["path"] for m in metadata_store))}

@app.get("/files")
def list_files():
    file_map = {}
    for m in metadata_store:
        file_map.setdefault(m["path"],0); file_map[m["path"]] += 1
    return [{"file":f,"chunks":c} for f,c in file_map.items()]

@app.get("/document/{filename}")
def get_document(filename: str):
    safe = Path(filename).name
    fp = os.path.join(UPLOAD_DIR, safe)
    return FileResponse(fp) if os.path.exists(fp) else {"error":"File not found"}

# ================= QUERY =================
@app.post("/query")
async def query(
    q: str = Form(...),
    model: str = Form("fast"),
    k: int = Form(8),
    session: str = Form("default"),   # ← NEW: frontend can pass session id
):
    model = model.strip().lower()
    rewritten_q = rewrite_query_with_memory(q, model, session)
    print(f"\n🧠 Original: {q}\n🔎 Rewritten: {rewritten_q}")

    docs = hybrid_search(rewritten_q, k=k)
    docs = [d for d in docs if float(d.get("score",0)) > 0.50]
    docs = sorted(docs, key=lambda x:x.get("score",0), reverse=True)

    if detect_topic_shift_by_file(docs, session):
        clear_session(session)

    if docs:
        top_score = docs[0].get("score",0)
        dominant_file = docs[0].get("path")
        file_best = {}
        for d in docs: file_best[d.get("path")] = max(file_best.get(d.get("path"),0), d.get("score",0))
        file_counts = Counter(d.get("path") for d in docs)
        docs = [d for d in docs if d.get("path")==dominant_file
                or (file_best.get(d.get("path"),0)>=top_score*0.92
                    and file_counts.get(d.get("path"),0)>=file_counts.get(dominant_file,1))]

    if not docs:
        return {"answer":"The answer is not present in the provided sources.","citations":[],"confidence":0,"chunks_used":0}

    if model in ["mistral","llama3"]:
        docs = rerank_with_cross_encoder(q, docs)
    docs = docs[:6]
    retrieved_docs = docs.copy()
    docs = build_semantic_section(docs, window=1)
    context, source_map = build_context(docs)
    memory_context = build_memory_context(session)
    sources = [{"num":num,"path":path,"page":page} for (path,page),num in sorted(source_map.items(),key=lambda x:x[1])]

    if model in ["mistral","llama3"]:
        scores = [d.get("ce_score", d.get("score",0)) for d in docs]
    else:
        scores = [d.get("score",0) for d in docs]

    avg = sum(scores)/len(scores) if scores else 0
    if len(docs)<2: avg *= 0.85
    if np.var(scores)>0.15: avg *= 0.9
    confidence = round(min(max((avg+10)/20*100,0),100),1) if model in ["mistral","llama3"] else round(avg*100,1)

    if model=="fast":
        context = " ".join((memory_context+context).split()[:600]); memory_context=""
    else:
        context = " ".join(context.split()[:1200])

    prompt = build_list_prompt(memory_context+context, q) if is_list_question(q) else build_prompt(memory_context+context, q, sources)
    raw_answer = call_llm(prompt, model)

    if is_list_question(q):
        items = extract_list_items(raw_answer)
        answer = "\n\n".join(f"• {i}" for i in items) if items else "The answer is not present in the provided sources."
    else:
        answer = raw_answer.strip() if isinstance(raw_answer, str) else str(raw_answer)

    # ── Save turn to SQLite ──
    append_turn(question=q, answer=answer, embedding=embed(q),
                file=docs[0]["path"] if docs else None, session=session)

    trace = [{"file":d.get("path"),"page":d.get("page"),"chunk_id":d.get("chunk_id"),
              "score":round(d.get("score",d.get("llm_score",0)),3)} for d in docs]

    return convert_numpy({
        "answer": answer, "citations": retrieved_docs,
        "confidence": confidence, "chunks_used": len(docs),
        "trace": trace, "linked_sources": build_linked_sources(retrieved_docs)
    })

@app.post("/reset")
def reset_memory(session: str = Form("default")):
    clear_session(session)
    return {"status":"Conversation memory cleared"}

@app.post("/image_query")
async def image_query(file: UploadFile = File(...)):
    if not file.filename: return {"error":"Invalid file"}
    path = os.path.join(UPLOAD_DIR, file.filename)
    with open(path,"wb") as f: f.write(await file.read())
    results = []
    if image_index.ntotal > 0:
        qvec = np.array([embed_image(path)], dtype="float32")
        d2, i2 = image_index.search(qvec, 5) #type: ignore
        for score, idx in zip(d2[0],i2[0]):
            if idx < len(image_metadata):
                results.append({"score":float(score),"path":image_metadata[idx]["path"],
                    "preview":image_metadata[idx]["preview"],"type":"image"})
    for r in hybrid_search("", k=5, image_path=path):
        if r.get("type")!="image":
            results.append({"score":float(r.get("score",0)),"path":r.get("path"),
                "snippet":r.get("snippet",""),"page":r.get("page"),
                "chunk_id":r.get("chunk_id"),"timestamp":r.get("timestamp"),"type":r.get("type","text")})
    return {"results": sorted(results, key=lambda x:x["score"], reverse=True)}

@app.get("/query")
def query_get():
    return {"error":"Use POST /query with form data: q, model, k, session"}


# ================= TRANSCRIBE (Voice Input) =================
@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """
    Receives a short audio blob from the browser (webm/ogg),
    transcribes it with the already-loaded Whisper model,
    and returns { text: "..." }
    """
    import tempfile

    # Save to a temp file with the right extension
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