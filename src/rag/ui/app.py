import streamlit as st
import time
from rag.config import settings
from rag.retrievers.hybrid import HybridRetriever
from rag.retrievers.reranker import CrossEncoderReranker
from rag.graph.workflow import RAGWorkflow
from rag.cache.semantic_cache import SemanticCache

st.set_page_config(
    page_title="Arthur Schopenhauer AI Scholar (Production RAG)",
    page_icon="📜",
    layout="wide",
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
    }
    .citation-box {
        background-color: #F1F5F9;
        border-left: 4px solid #3B82F6;
        padding: 10px;
        border-radius: 4px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📜 Arthur Schopenhauer: Production RAG Scholar</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Grounded Question-Answering over <i>The Wisdom of Life</i> and <i>Schopenhauer\'s Essays</i> (1851)</div>', unsafe_allow_html=True)

# Sidebar controls
with st.sidebar:
    st.header("⚙️ RAG Engine Parameters")
    
    selected_llm = st.selectbox(
        "LLM Generation Model",
        options=["nvidia/nemotron-3.5-lightning:free", "google/gemini-2.5-flash", "meta-llama/llama-3.3-70b-instruct:free"],
        index=0,
    )
    
    top_k = st.slider("Context Chunks (Top-K)", min_value=1, max_value=8, value=4)
    dense_weight = st.slider("Dense Semantic Weight (Chroma)", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
    sparse_weight = 1.0 - dense_weight
    st.caption(f"Sparse Lexical Weight (BM25): **{sparse_weight:.2f}**")
    
    enable_cache = st.toggle("Enable Semantic Cache", value=True)
    enable_rerank = st.toggle("Enable Cross-Encoder Re-Ranking", value=True)
    
    st.divider()
    st.markdown("### 📊 System Metadata")
    st.write(f"**Embedding Model:** `{settings.EMBEDDING_MODEL_NAME}`")
    st.write(f"**Re-Ranker:** `{settings.RERANKER_MODEL_NAME}`")
    st.write(f"**Total Ingested Chunks:** `2,053`")

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome. I am your scholarly assistant on Arthur Schopenhauer's philosophy. Ask me about his views on happiness, personality, property, fame, death, suicide, or the Will."}
    ]

# Lazy-loaded workflow
@st.cache_resource
def load_rag_pipeline():
    workflow = RAGWorkflow()
    hybrid = HybridRetriever()
    reranker = CrossEncoderReranker()
    cache = SemanticCache()
    return workflow, hybrid, reranker, cache

workflow, hybrid_retriever, reranker, cache = load_rag_pipeline()

# Layout: Main Chat vs Inspection Tabs
chat_col, inspect_col = st.columns([1.2, 1.0])

with chat_col:
    st.subheader("💬 Scholar Dialogue")
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "citations" in msg and msg["citations"]:
                with st.expander("📚 Source Citations"):
                    for c in msg["citations"]:
                        st.markdown(f"**{c.get('source_id')} {c.get('section')} &rarr; {c.get('chapter')}** (Book Page {c.get('book_page')})")
                        st.caption(c.get("snippet"))

    user_query = st.chat_input("Ask a question about Schopenhauer's philosophy...")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            start_t = time.perf_counter()
            with st.spinner("Analyzing texts, fusing hybrid indices, and verifying grounding..."):
                res = workflow.invoke(user_query)
            latency = (time.perf_counter() - start_t) * 1000

            st.markdown(res.get("answer", ""))
            
            citations = res.get("citations", [])
            if citations:
                with st.expander("📚 Source Citations"):
                    for c in citations:
                        st.markdown(f"**{c.get('source_id')} {c.get('section')} &rarr; {c.get('chapter')}** (Book Page {c.get('book_page')})")
                        st.caption(c.get("snippet"))
                        
            st.session_state.last_result = res
            st.session_state.last_latency = latency
            st.session_state.messages.append({
                "role": "assistant",
                "content": res.get("answer", ""),
                "citations": citations,
            })

with inspect_col:
    st.subheader("🔍 Deep Retrieval Inspector")
    
    tab1, tab2, tab3 = st.tabs(["Hybrid & Rerank Scores", "Source Passages", "Economics & Guardrails"])
    
    last_res = getattr(st.session_state, "last_result", None)
    
    with tab1:
        if last_res and "question" in last_res:
            q = last_res["question"]
            st.markdown(f"**Query:** *{q}*")
            
            # Fetch raw hybrid and reranked
            raw_hybrid = hybrid_retriever.retrieve(q, k=5)
            raw_reranked = reranker.rerank(q, raw_hybrid, top_k=top_k)
            
            st.markdown("#### Top Re-Ranked Chunks")
            for rank, (doc, score) in enumerate(raw_reranked, start=1):
                meta = doc.metadata
                st.markdown(f"""
                <div class="metric-card">
                    <b>#{rank} &bull; Score: {score:.4f}</b><br>
                    <small><b>Section:</b> {meta.get('section')} | <b>Chapter:</b> {meta.get('chapter')} | <b>Page:</b> {meta.get('book_page')}</small><br>
                    <p style="margin-top: 4px; font-size: 0.9rem; color: #334155;">{doc.page_content[:180]}...</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Submit a question in the chat to inspect live retrieval scores and cross-attention rankings.")

    with tab2:
        if last_res and last_res.get("citations"):
            for c in last_res["citations"]:
                st.markdown(f"""
                <div class="citation-box">
                    <b>{c.get('source_id')} {c.get('chapter')}</b><br>
                    <small>Section: {c.get('section')} | Book Page: {c.get('book_page')} | PDF Page: {c.get('pdf_page')}</small>
                    <hr style="margin: 6px 0;">
                    <p style="font-size: 0.88rem;">{c.get('snippet')}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Passages will appear here after a query.")

    with tab3:
        if last_res:
            c1, c2 = st.columns(2)
            c1.metric("Latency", f"{getattr(st.session_state, 'last_latency', 0):.0f} ms")
            c2.metric("Tokens Used", last_res.get("tokens_used", 0))
            
            c3, c4 = st.columns(2)
            c3.metric("Semantic Cache", "Hit" if last_res.get("cached") else "Miss")
            c4.metric("Faithfulness Check", "Passed ✅" if last_res.get("is_grounded") else "Warning ⚠️")
            
            if last_res.get("grounding_explanation"):
                st.caption(f"**Verifier Note:** {last_res.get('grounding_explanation')}")
        else:
            st.info("Execution metrics will display here.")
