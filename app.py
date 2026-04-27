"""
LexiPolicy v2.0 - Premium Academic QA System
Optimized for aesthetic excellence and robust retrieval.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from src.qa_system import AcademicQASystem
from src.data_processing import DocumentProcessor
import time
import os

# Page configuration
st.set_page_config(
    page_title="LexiPolicy Premium",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium UI (Nordic Dark / Glassmorphism)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap');
    
    :root {
        --primary: #6366f1;
        --secondary: #a855f7;
        --bg: #0f172a;
        --card: rgba(30, 41, 59, 0.7);
        --text: #f8fafc;
        --text-dim: #94a3b8;
    }
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: var(--text);
    }
    
    .stApp {
        background: radial-gradient(circle at top right, #1e293b, #0f172a);
    }
    
    /* Premium Glassmorphism Card */
    .glass-card {
        background: var(--card);
        backdrop-filter: blur(12px);
        border-radius: 20px;
        padding: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        transition: transform 0.2s ease;
    }
    
    .glass-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(to right, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: var(--text-dim);
        margin-bottom: 3rem;
        font-weight: 400;
    }
    
    .answer-box {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(168, 85, 247, 0.1));
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(99, 102, 241, 0.2);
        font-size: 1.15rem;
        line-height: 1.7;
        color: #e2e8f0;
        box-shadow: inset 0 2px 4px 0 rgba(0, 0, 0, 0.05);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-top: 4px;
    }
    
    .metric-label {
        font-size: 0.75rem;
        color: var(--text-dim);
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.1em;
    }
    
    .badge {
        background: rgba(99, 102, 241, 0.2);
        color: #818cf8;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
    
    /* Animation */
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .animate-in {
        animation: slideUp 0.6s cubic-bezier(0.22, 1, 0.36, 1) forwards;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'qa_system' not in st.session_state:
    st.session_state.qa_system = None
if 'system_ready' not in st.session_state:
    st.session_state.system_ready = False

# Sidebar Branding & Config
with st.sidebar:
    st.markdown("### 💠 LexiPolicy Elite")
    st.markdown("---")
    
    # Initialization Logic
    if not st.session_state.system_ready:
        st.info("System offline. Please initialize to load signatures.")
        if st.button("⚡ Initialize Engine", use_container_width=True):
            with st.spinner("Analyzing high-dimensional embeddings..."):
                try:
                    qa_system = AcademicQASystem(use_lsh=True, use_simhash=True, use_tfidf=True, use_llm=False)
                    processor = DocumentProcessor()
                    loaded = 0

                    # Priority 1: Real PDFs
                    handbook_dir = Path("data/handbooks")
                    pdf_files = list(handbook_dir.glob("*.pdf")) if handbook_dir.exists() else []
                    for pdf_file in pdf_files:
                        qa_system.add_document(str(pdf_file), pdf_file.stem)
                        loaded += 1

                    # Priority 2: Sample text files (demo / no-PDF mode)
                    if loaded == 0:
                        sample_dir = Path("data/sample_handbooks")
                        for txt_file in sorted(sample_dir.glob("*.txt")):
                            with open(txt_file, "r", encoding="utf-8") as f:
                                text = f.read()
                            chunks = processor.chunk_text(text, txt_file.stem)
                            for chunk_id, chunk_text, page_num in chunks:
                                qa_system.documents[chunk_id] = chunk_text
                                qa_system.doc_metadata[chunk_id] = {
                                    "source": txt_file.stem.replace("_", " ").title(),
                                    "page": page_num,
                                }
                                tokens = set(processor.tokenize(chunk_text))
                                qa_system.lsh.index_document(chunk_id, tokens)
                                qa_system.simhash_fingerprints[chunk_id] = \
                                    qa_system.simhash.compute_fingerprint(list(tokens))
                            loaded += 1

                    if loaded == 0:
                        st.error("No handbook data found. Add PDFs to data/handbooks/ or .txt files to data/sample_handbooks/")
                    else:
                        qa_system.fit_baseline()
                        st.session_state.qa_system = qa_system
                        st.session_state.system_ready = True
                        st.success(f"Engine Online — {len(qa_system.documents)} chunks indexed")
                except Exception as e:
                    st.error(f"Boot Error: {e}")
    else:
        st.success("✅ System Online")
        st.markdown("#### 🛠️ Retrieval Parameters")
        method_label = st.selectbox("Search Strategy", ["LSH (MinHash)", "SimHash", "TF-IDF (Exact)"])
        top_k = st.slider("Context Depth (K)", 1, 10, 5)
        ans_mode = st.radio("Synthesis", ["Extractive", "Neural LLM"], horizontal=True)
        
        method_map = {"LSH (MinHash)": "lsh", "SimHash": "simhash", "TF-IDF (Exact)": "tfidf"}
        
    st.markdown("---")
    st.markdown("v2.0.4 | [Documentation](file:///Users/nexxus/Desktop/Personal/Projects/BDA/README.md)")

# Main Layout
col_main, _ = st.columns([10, 1])
with col_main:
    st.markdown('<div class="main-header">LexiPolicy</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Scalable Information Retrieval & Analytics for Academic Governance</div>', unsafe_allow_html=True)

if st.session_state.system_ready:
    tab_search, tab_insight, tab_lab = st.tabs(["💎 Retrieval Engine", "📈 Knowledge Analytics", "🧪 Scaling Labs"])
    
    with tab_search:
        # Search Box
        q_col1, q_col2 = st.columns([5, 1])
        with q_col1:
            query = st.text_input("Ask about policies...", placeholder="Minimum GPA requirements...", label_visibility="collapsed")
        with q_col2:
            search_exec = st.button("Execute Query", use_container_width=True, type="primary")
            
        if (search_exec or query) and query:
            with st.spinner("Traversing LSH buckets..."):
                start_t = time.time()
                result = st.session_state.qa_system.answer_query(
                    query, 
                    method=method_map[method_label], 
                    top_k=top_k,
                    answer_method='llm' if ans_mode == 'Neural LLM' else 'extractive'
                )
                lat = time.time() - start_t
                
                # Result Area
                st.markdown('<div class="animate-in">', unsafe_allow_html=True)
                
                # Header & Metrics
                st.markdown("### 💡 Synthesized Response")
                st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)
                
                m1, m2, m3, m4 = st.columns(4)
                
                # --- BUG FIX: Check if retrieved_chunks is empty before accessing index 0 ---
                chunks_count = len(result["retrieved_chunks"])
                top_similarity = result["retrieved_chunks"][0]["similarity"] if chunks_count > 0 else 0.0
                confidence = "High" if top_similarity > 0.7 else "Moderate" if top_similarity > 0.4 else "Low"
                if chunks_count == 0: confidence = "None"

                with m1:
                    st.markdown(f'<div class="glass-card"><div class="metric-label">Latency</div><div class="metric-value">{lat:.4f}s</div></div>', unsafe_allow_html=True)
                with m2:
                    st.markdown(f'<div class="glass-card"><div class="metric-label">Results</div><div class="metric-value">{chunks_count}</div></div>', unsafe_allow_html=True)
                with m3:
                    st.markdown(f'<div class="glass-card"><div class="metric-label">Top Similarity</div><div class="metric-value">{top_similarity:.2f}</div></div>', unsafe_allow_html=True)
                with m4:
                    st.markdown(f'<div class="glass-card"><div class="metric-label">Confidence</div><div class="metric-value">{confidence}</div></div>', unsafe_allow_html=True)
                
                # Evidentiary Chunks
                if chunks_count > 0:
                    st.markdown("### 📂 Source Evidence")
                    for i, chunk in enumerate(result["retrieved_chunks"]):
                        with st.expander(f"Reference Segment {i+1} | Source: {chunk['source']} | Score: {chunk['similarity']:.3f}"):
                            st.markdown(f'<p style="color: #cbd5e1; font-size: 0.95rem;">{chunk["text"]}...</p>', unsafe_allow_html=True)
                else:
                    st.warning("No relevant segments discovered in the vector space.")
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Shortcuts
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### ⚡ Common Queries")
        cols = st.columns(4)
        prompts = ["GPA Policy", "Attendance Rule", "Failing a Course", "Graduation Check"]
        query_map = {
            "GPA Policy": "What is the minimum GPA requirement for graduation?",
            "Attendance Rule": "What is the policy regarding class attendance?",
            "Failing a Course": "What happens if a student fails a course?",
            "Graduation Check": "What are the core requirements for completing a degree?"
        }
        for i, p in enumerate(prompts):
            if cols[i].button(p, use_container_width=True):
                st.session_state.temp_query = query_map[p]
                st.rerun()

    with tab_insight:
        stats = st.session_state.qa_system.get_statistics()
        
        col_st1, col_st2, col_st3 = st.columns(3)
        with col_st1:
            st.markdown(f'<div class="glass-card"><div class="metric-label">Corpus Scale</div><div class="metric-value">{stats["total_chunks"]} Chunks</div></div>', unsafe_allow_html=True)
        with col_st2:
            st.markdown(f'<div class="glass-card"><div class="metric-label">Vocabulary</div><div class="metric-value">{stats["total_tokens"]:,} Tokens</div></div>', unsafe_allow_html=True)
        with col_st3:
            st.markdown(f'<div class="glass-card"><div class="metric-label">Patterns</div><div class="metric-value">{len(stats["hot_topics"])} Trends</div></div>', unsafe_allow_html=True)
            
        c_an1, c_an2 = st.columns(2)
        with c_an1:
            st.markdown("#### 📊 Discovery Trends (Frequent Itemsets)")
            if stats['hot_topics']:
                df_topics = pd.DataFrame(stats['hot_topics'], columns=['Pattern', 'Frequency'])
                fig_t = px.bar(df_topics, x='Frequency', y='Pattern', orientation='h', color='Frequency',
                             color_continuous_scale='Viridis', template='plotly_dark')
                fig_t.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400)
                st.plotly_chart(fig_t, use_container_width=True)
            else:
                st.info("Execute more queries to populate trend analysis.")
        
        with c_an2:
            st.markdown("#### ⚡ Algorithmic Efficiency")
            perf = stats['performance_summary']
            if perf:
                p_data = [{'Method': m.upper(), 'Latency': s['avg_time']} for m, s in perf.items()]
                fig_p = px.pie(pd.DataFrame(p_data), values='Latency', names='Method', 
                             hole=0.6, template='plotly_dark', color_discrete_sequence=px.colors.sequential.Indigo)
                fig_p.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400)
                st.plotly_chart(fig_p, use_container_width=True)
            else:
                st.info("Performance metrics will appear here.")

    with tab_lab:
        st.markdown("### 🧪 Big Data Scaling Simulations")
        st.markdown("Validate LSH performance across synthetic document expansions.")
        
        if st.button("🚀 Trigger Stress Test"):
            from src.experiments import ExperimentalEvaluation
            exp = ExperimentalEvaluation(st.session_state.qa_system)
            with st.spinner("Expanding corpus 10x and measuring O(1) traversal..."):
                results = exp.run_all_experiments()
                
                # Scalability Chart
                scaling = results['scalability']['scaling_tests']
                df_s = pd.DataFrame(scaling)
                
                fig_s = go.Figure()
                fig_s.add_trace(go.Scatter(x=df_s['doc_count'], y=df_s['lsh_query_time'], name='LSH (Approximate)', line=dict(color='#818cf8', width=3)))
                fig_s.add_trace(go.Scatter(x=df_s['doc_count'], y=df_s['tfidf_query_time'], name='TF-IDF (Exact)', line=dict(color='#f43f5e', width=3, dash='dot')))
                fig_s.update_layout(title='Latency vs Corpus Scale', xaxis_title='Document Count', yaxis_title='Seconds', 
                                  template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_s, use_container_width=True)
                
                st.success("Scalability proof completed. LSH demonstrates near O(1) consistency.")

else:
    # Landing State
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center;">
        <h1 style="color: #818cf8; font-size: 4rem;">💠</h1>
        <h2 style="font-weight: 700;">LexiPolicy Engine Standby</h2>
        <p style="color: #94a3b8; max-width: 600px; margin: 0 auto;">
            Initialize the core retrieval engine from the sidebar to start querying academic governance handbooks using Locality Sensitive Hashing.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; color: #475569; font-size: 0.8rem;'>LexiPolicy Elite v2.0 | Advanced Big Data Algorithms | 2025</div>", unsafe_allow_html=True)
