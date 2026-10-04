                                # novaRAG_app.py
import streamlit as st
import requests
from pathlib import Path
import re
API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="NovaRAG Offline Multimodal RAG", layout="wide")
st.title("NovaRAG Offline Multimodal RAG")


# --------------------------
# 📁 Indexed Files Sidebar
# --------------------------
with st.sidebar:
    st.header("📁 Indexed Files")
    try:
        r = requests.get(f"{API_URL}/files", timeout=5)
        r.raise_for_status()
        files = r.json()

        if files:
            for f in files:
                st.write(f"📄 {f['file']} — {f['chunks']} chunks")
        else:
            st.write("No files indexed yet.")
    except:
        st.write("Backend not connected")

    st.divider()

    if st.button("🧹 Clear Conversation"):
        try:
            requests.post(f"{API_URL}/reset")
            st.session_state.history = []
            st.success("Conversation cleared")
        except:
            st.error("Could not reset memory")

    

        
# session memory
if "history" not in st.session_state:
    st.session_state.history = []

# --------------------------
# 1️⃣ File Ingestion
# --------------------------
st.header("1️⃣ Ingest File")

uploaded_file = st.file_uploader(
    "Upload PDF, DOCX, IMAGE, or AUDIO",
    type=["pdf", "docx", "png", "jpg", "jpeg", "wav", "mp3", "m4a"],
    help="Limit 200MB per file"
)
image_query = st.file_uploader("Upload image to search", type=["png","jpg"])
if image_query:
        if st.button("Search Image"):
            with st.spinner("Searching image..."):
                try:
                    files = {"file": ("query.png", image_query.getvalue())}
                    resp = requests.post(f"{API_URL}/image_query", files=files)
                    data = resp.json()
                    results = data.get("results", [])

                    if results:
                        img_results  = [r for r in results if r.get("type") == "image"]
                        text_results = [r for r in results if r.get("type") == "text"]
                        audio_results= [r for r in results if r.get("type") == "audio"]

                        if img_results:
                            st.markdown("**Matching images**")
                            for r in img_results:
                                st.image(f"{API_URL}/document/{Path(r['path']).name}")
                                st.caption(f"Score: {r['score']:.2f}")

                        if text_results:
                            st.markdown("**Matching text passages**")
                            for r in text_results:
                                fname = Path(r['path']).name
                                page  = r.get('page', 'N/A')
                                with st.expander(f"{fname} — page {page} (score {r['score']:.2f})"):
                                    st.write(r.get("snippet", ""))
                                    st.markdown(f"[⬇ Open document]({API_URL}/document/{fname})")
                        
                        if audio_results:
                            st.markdown("**Matching audio segments**")
                            for r in audio_results:
                                fname = Path(r['path']).name
                                ts    = int(r.get("timestamp", 0))
                                st.audio(f"{API_URL}/document/{fname}", start_time=ts)
                                st.caption(f"{fname} — starts at {ts}s (score {r['score']:.2f})")
                    else:
                        st.info("No matching results found")
                except Exception as e:
                    st.error(f"Error: {e}")
audio_file = st.file_uploader("Upload audio to play", type=["mp3","wav"])

if audio_file:
    st.audio(audio_file)
if uploaded_file:
    st.info(f"Selected file: {uploaded_file.name} ({uploaded_file.size/1024/1024:.2f} MB)")

    if st.button("Ingest File"):
        with st.spinner("Uploading and ingesting..."):
            try:
                files = {
                    "file": (uploaded_file.name, uploaded_file.getvalue())
                }

                resp = requests.post(f"{API_URL}/ingest", files=files, timeout=300)
                resp.raise_for_status()

                data = resp.json()
                st.success("✅ File ingested successfully!")

                if "chunks" in data:
                    st.caption(f"Indexed chunks: {data['chunks']}")

            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to backend. Start FastAPI first.")
            except Exception as e:
                st.error(f"❌ Error: {e}")
# --------------------------
# 2️⃣ Model Selection
# --------------------------
st.subheader("Model Selection")

model_map = {
    "Fast (Small Offline Model)": "fast",
    "Smart (Mistral Local LLM)": "mistral",
    "Creative (Llama3 Local LLM)": "llama3"
}

selected_label = st.selectbox(
    "Choose reasoning engine",
    list(model_map.keys())
)

model_choice = model_map[selected_label]

# --------------------------
# 2️⃣ Chat Interface
# --------------------------
st.header("2️⃣ Ask Questions")
k_value = st.slider("🔍 Retrieval Depth", 3, 15, 8)
question = st.chat_input("Ask anything about your documents...")

if question:
    st.session_state.history.append({
        "role": "user",
        "message": question
    })

    with st.spinner("Thinking..."):
        data = {}

        try:
            resp = requests.post(
                f"{API_URL}/query",
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data={"q": question, "model": model_choice, "k": k_value},
                timeout=180
            )

            resp.raise_for_status()
            data = resp.json()

            answer = data.get("answer", "No answer returned.")
            citations = data.get("citations", [])
            confidence = data.get("confidence", None)
            chunks_used = data.get("chunks_used", None)
            trace = data.get("trace", [])

            st.session_state.history.append({
                "role": "assistant",
                "message": answer,
                "user_question": question,
                "citations": citations,
                "confidence": confidence,
                "chunks_used": chunks_used,
                "trace": trace,
                "linked_sources": data.get("linked_sources", [])
            })

        except Exception as e:
            st.session_state.history.append({
                "role": "assistant",
                "message": f"Error: {e}",
                "citations": [],
                "confidence": None,
                "chunks_used": None,
                "trace": []
            })

# --------------------------
# Conversation Display
# --------------------------
for item in st.session_state.history:
    if item["role"] == "user":
        with st.chat_message("user"):
            st.markdown(item["message"])

    else:
        with st.chat_message("assistant"):
            st.markdown(item["message"])

            if item.get("confidence") is not None:
                st.caption(f"Confidence: {item['confidence']}%")
            if item.get("chunks_used") is not None:
                st.caption(f"Chunks Used: {item['chunks_used']}")

            citations = item.get("citations", [])
            if citations:
                st.markdown("**Sources:**")
                seen = set()

                for c in citations:
                    score = c.get("score", 0)
                    try:
                        st.progress(float(min(max(score, 0), 1)))
                    except:
                        pass
                    key = (c.get("path"), c.get("page"))

                    if key in seen:
                        continue

                    seen.add(key)
                    file_name = Path(c.get("path", "Unknown")).name
                    if c.get("type") == "image":
                        st.image(f"{API_URL}/document/{file_name}")

                    doc_url = f"{API_URL}/document/{file_name}"

                    if "timestamp" in c:
                        st.audio(doc_url, start_time=int(c.get("timestamp", 0)))
                        st.caption(f"Starts at {c['timestamp']} sec")
                    page_no = c.get("page", "N/A")
                    snippet = c.get("snippet", "")

                    user_question = item.get("user_question", "")
                    if user_question:
                        safe_q = re.escape(user_question)
                        snippet = re.sub(f"({safe_q})", r"**\1**", snippet, flags=re.IGNORECASE)

                    with st.expander(f"{file_name} (page {page_no})"):
                        st.text(snippet)   # ← st.text instead of st.write — no markdown rendering
                        doc_url = f"{API_URL}/document/{file_name}"
                        st.markdown(f"[⬇ Download Source Document]({doc_url})")

            trace = item.get("trace", [])
            if trace:
                st.markdown("**Traceability Map:**")
                for t in trace:
                    st.write(
                        f"📄 {t['file']} | Page: {t['page']} | Chunk: {t['chunk_id']} | Score: {t['score']}"
                    )  

            linked = item.get("linked_sources", [])
            if linked:
                st.markdown("**Cross-modal Links:**")
                for lnk in linked:
                    icon = "🔗"
                    if lnk["type"] == "text↔audio":
                        ts = f" @ {lnk['audio_timestamp']}s" if lnk.get("audio_timestamp") else ""
                        st.info(f"{icon} **Text↔Audio** — `{Path(lnk['text_source']).name}` (page {lnk.get('text_page','?')}) linked to `{Path(lnk['audio_source']).name}`{ts}")
                    elif lnk["type"] == "text↔image":
                        st.info(f"{icon} **Text↔Image** — `{Path(lnk['text_source']).name}` (page {lnk.get('text_page','?')}) linked to `{Path(lnk['image_source']).name}`")
                    elif lnk["type"] == "audio↔image":
                        ts = f" @ {lnk['audio_timestamp']}s" if lnk.get("audio_timestamp") else ""
                        st.info(f"{icon} **Audio↔Image** — `{Path(lnk['audio_source']).name}`{ts} linked to `{Path(lnk['image_source']).name}`")