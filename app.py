"""
Finrate - Academic Policy QA System
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

st.set_page_config(
    page_title="Finrate",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap');

    :root {
        --bg:       #000000;
        --surface:  #111111;
        --border:   #2a2a2a;
        --text:     #ffffff;
        --text-dim: #888888;
        --accent:   #ffffff;
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: var(--text);
        background-color: var(--bg);
    }

    .stApp {
        background-color: var(--bg);
    }

    /* Cards */
    .glass-card {
        background: var(--surface);
        border-radius: 14px;
        padding: 22px 26px;
        border: 1px solid var(--border);
        margin-bottom: 18px;
        transition: border-color 0.2s ease;
    }
    .glass-card:hover {
        border-color: #555555;
    }

    /* Header */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.03em;
        margin-bottom: 0.4rem;
    }
    .sub-header {
        font-size: 1rem;
        color: var(--text-dim);
        margin-bottom: 2.5rem;
        font-weight: 400;
    }

    /* Answer box */
    .answer-box {
        background: #111111;
        border-radius: 12px;
        padding: 22px 26px;
        border: 1px solid #2a2a2a;
        font-size: 1.05rem;
        line-height: 1.75;
        color: #e0e0e0;
    }

    /* Metrics */
    .metric-value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #ffffff;
        margin-top: 4px;
    }
    .metric-label {
        font-size: 0.7rem;
        color: var(--text-dim);
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.12em;
    }

    /* Badge */
    .badge {
        background: #1a1a1a;
        color: #cccccc;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 600;
        border: 1px solid #333333;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #080808 !important;
        border-right: 1px solid #1e1e1e;
    }

    /* Slide-up animation */
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(16px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .animate-in { animation: slideUp 0.5s ease forwards; }

    /* Buttons */
    div.stButton > button {
        background: #ffffff;
        color: #000000;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: background 0.15s;
    }
    div.stButton > button:hover {
        background: #dddddd;
        color: #000000;
    }

    /* Primary button override */
    div.stButton > button[kind="primary"] {
        background: #ffffff;
        color: #000000;
    }
</style>
""", unsafe_allow_html=True)

# Session state
if 'qa_system' not in st.session_state:
    st.session_state.qa_system = None
if 'system_ready' not in st.session_state:
    st.session_state.system_ready = False

# Sidebar
with st.sidebar:
    st.markdown("### ◈ Finrate")
    st.markdown("---")

    if not st.session_state.system_ready:
        st.info("System offline. Initialize to load signatures.")
        if st.button("Initialize Engine", use_container_width=True):
            with st.spinner("Loading document index..."):
                try:
                    qa_system = AcademicQASystem(use_lsh=True, use_simhash=True, use_tfidf=True, use_llm=False)
                    processor = DocumentProcessor()
                    loaded = 0

                    handbook_dir = Path("data/handbooks")
                    pdf_files = list(handbook_dir.glob("*.pdf")) if handbook_dir.exists() else []
                    for pdf_file in pdf_files:
                        qa_system.add_document(str(pdf_file), pdf_file.stem)
                        loaded += 1

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
                        st.error("No handbook data found.")
                    else:
                        qa_system.fit_baseline()
                        st.session_state.qa_system = qa_system
                        st.session_state.system_ready = True
                        st.success(f"Online — {len(qa_system.documents)} chunks indexed")
                except Exception as e:
                    st.error(f"Boot Error: {e}")
    else:
        st.success("System Online")
        st.markdown("#### Retrieval Parameters")
        method_label = st.selectbox("Search Strategy", ["LSH (MinHash)", "SimHash", "TF-IDF (Exact)"])
        top_k = st.slider("Context Depth (K)", 1, 10, 5)
        ans_mode = st.radio("Synthesis", ["Extractive", "Neural LLM"], horizontal=True)
        method_map = {"LSH (MinHash)": "lsh", "SimHash": "simhash", "TF-IDF (Exact)": "tfidf"}

    st.markdown("---")
    st.markdown('<span style="color:#555;font-size:0.75rem;">v2.0.4 | Finrate</span>', unsafe_allow_html=True)

# Main header
col_main, _ = st.columns([10, 1])
with col_main:
    st.markdown('<div class="main-header">Finrate</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Scalable Information Retrieval & Analytics for Academic Governance</div>', unsafe_allow_html=True)

BW_SEQ = ['#ffffff', '#cccccc', '#999999', '#666666', '#333333']
BW_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#111111',
        font=dict(color='#cccccc'),
        xaxis=dict(gridcolor='#222222', zerolinecolor='#333333'),
        yaxis=dict(gridcolor='#222222', zerolinecolor='#333333'),
    )
)

if st.session_state.system_ready:
    tab_search, tab_insight, tab_lab = st.tabs(["Retrieval Engine", "Knowledge Analytics", "Scaling Labs"])

    with tab_search:
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

                st.markdown('<div class="animate-in">', unsafe_allow_html=True)
                st.markdown("### Synthesized Response")
                st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)

                chunks_count = len(result["retrieved_chunks"])
                top_similarity = result["retrieved_chunks"][0]["similarity"] if chunks_count > 0 else 0.0
                confidence = "High" if top_similarity > 0.7 else "Moderate" if top_similarity > 0.4 else "Low"
                if chunks_count == 0:
                    confidence = "None"

                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.markdown(f'<div class="glass-card"><div class="metric-label">Latency</div><div class="metric-value">{lat:.4f}s</div></div>', unsafe_allow_html=True)
                with m2:
                    st.markdown(f'<div class="glass-card"><div class="metric-label">Results</div><div class="metric-value">{chunks_count}</div></div>', unsafe_allow_html=True)
                with m3:
                    st.markdown(f'<div class="glass-card"><div class="metric-label">Top Similarity</div><div class="metric-value">{top_similarity:.2f}</div></div>', unsafe_allow_html=True)
                with m4:
                    st.markdown(f'<div class="glass-card"><div class="metric-label">Confidence</div><div class="metric-value">{confidence}</div></div>', unsafe_allow_html=True)

                if chunks_count > 0:
                    st.markdown("### Source Evidence")
                    for i, chunk in enumerate(result["retrieved_chunks"]):
                        with st.expander(f"Segment {i+1} | {chunk['source']} | Score: {chunk['similarity']:.3f}"):
                            st.markdown(f'<p style="color:#aaaaaa;font-size:0.95rem;">{chunk["text"]}...</p>', unsafe_allow_html=True)
                else:
                    st.warning("No relevant segments found.")

                st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Quick Queries")
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
            st.markdown("#### Discovery Trends (Frequent Itemsets)")
            if stats['hot_topics']:
                df_topics = pd.DataFrame(stats['hot_topics'], columns=['Pattern', 'Frequency'])
                fig_t = px.bar(df_topics, x='Frequency', y='Pattern', orientation='h',
                               color='Frequency', color_continuous_scale=['#333333', '#ffffff'])
                fig_t.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#111111',
                                    font=dict(color='#cccccc'), height=400,
                                    xaxis=dict(gridcolor='#222222'), yaxis=dict(gridcolor='#222222'))
                st.plotly_chart(fig_t, use_container_width=True)
            else:
                st.info("Execute queries to populate trend analysis.")

        with c_an2:
            st.markdown("#### Algorithmic Efficiency")
            perf = stats['performance_summary']
            if perf:
                p_data = [{'Method': m.upper(), 'Latency': s['avg_time']} for m, s in perf.items()]
                fig_p = px.pie(pd.DataFrame(p_data), values='Latency', names='Method',
                               hole=0.6, color_discrete_sequence=BW_SEQ)
                fig_p.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                    font=dict(color='#cccccc'), height=400)
                st.plotly_chart(fig_p, use_container_width=True)
            else:
                st.info("Performance metrics will appear here.")

    with tab_lab:
        st.markdown("### Big Data Scaling Simulations")
        st.markdown("Validate LSH performance across synthetic document expansions.")

        if st.button("Trigger Stress Test"):
            from src.experiments import ExperimentalEvaluation
            exp = ExperimentalEvaluation(st.session_state.qa_system)
            with st.spinner("Expanding corpus 10x and measuring traversal..."):
                results = exp.run_all_experiments()
                scaling = results['scalability']['scaling_tests']
                df_s = pd.DataFrame(scaling)

                fig_s = go.Figure()
                fig_s.add_trace(go.Scatter(x=df_s['doc_count'], y=df_s['lsh_query_time'],
                                           name='LSH (Approximate)', line=dict(color='#ffffff', width=3)))
                fig_s.add_trace(go.Scatter(x=df_s['doc_count'], y=df_s['tfidf_query_time'],
                                           name='TF-IDF (Exact)', line=dict(color='#888888', width=2, dash='dot')))
                fig_s.update_layout(title='Latency vs Corpus Scale',
                                    xaxis_title='Document Count', yaxis_title='Seconds',
                                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#111111',
                                    font=dict(color='#cccccc'),
                                    xaxis=dict(gridcolor='#222222'), yaxis=dict(gridcolor='#222222'))
                st.plotly_chart(fig_s, use_container_width=True)
                st.success("Scalability proof complete. LSH demonstrates near O(1) consistency.")

else:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;">
        <h1 style="color:#ffffff;font-size:4rem;">◈</h1>
        <h2 style="font-weight:700;color:#ffffff;">Finrate — System Standby</h2>
        <p style="color:#666666;max-width:560px;margin:0 auto;">
            Initialize the retrieval engine from the sidebar to start querying academic
            governance handbooks using Locality Sensitive Hashing.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;color:#333333;font-size:0.78rem;'>Finrate v2.0 | Advanced Big Data Algorithms | 2025</div>", unsafe_allow_html=True)
