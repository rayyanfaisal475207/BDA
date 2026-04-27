"""
Finrate v3.3 - Monochrome Academic Governance System
Pure Black & White Aesthetic | Gemini 3 Flash Powered
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
    page_title="Finrate | Academic Governance",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# High-Precision Monochrome Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=Outfit:wght@400;700&display=swap');
    
    /* Global Overrides */
    html, body, [class*="css"], [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        color: #000000 !important;
        background-color: #FFFFFF !important;
    }
    
    .stApp {
        background-color: #FFFFFF !important;
    }

    /* Sidebar - Heavy Precision */
    section[data-testid="stSidebar"] {
        background-color: #000000 !important;
        border-right: 1px solid #222222;
    }
    
    section[data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }
    
    /* Sidebar Buttons - Specificity Fix */
    div[data-testid="stSidebar"] .stButton > button {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border-radius: 0 !important;
        border: 2px solid #FFFFFF !important;
        width: 100% !important;
        font-weight: 800 !important;
        transition: all 0.3s ease-in-out !important;
    }
    
    div[data-testid="stSidebar"] .stButton > button:hover {
        background-color: #000000 !important;
        color: #FFFFFF !important;
        border: 2px solid #FFFFFF !important;
    }
    
    div[data-testid="stSidebar"] .stButton > button:hover * {
        color: #FFFFFF !important;
    }

    /* Branding */
    .brand-title {
        font-family: 'Outfit', sans-serif;
        font-size: 5.5rem;
        font-weight: 800;
        letter-spacing: -4px;
        color: #000000;
        margin-bottom: 0;
        line-height: 0.8;
    }
    
    .brand-subtitle {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 7px;
        color: #000000;
        margin-bottom: 4rem;
        font-weight: 800;
        opacity: 0.8;
    }

    /* Main Page Cards */
    .mono-card {
        background: #FFFFFF;
        border: 2px solid #000000;
        border-radius: 0;
        padding: 25px;
        margin-bottom: 20px;
        color: #000000 !important;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .mono-card:hover {
        box-shadow: 6px 6px 0px #000000;
        transform: translate(-3px, -3px);
    }

    .metric-val {
        font-family: 'Outfit', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        color: #000000 !important;
        line-height: 1.1;
    }
    
    .metric-lab {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #000000 !important;
        font-weight: 700;
        opacity: 0.5;
        margin-bottom: 10px;
    }

    /* Main Area Buttons */
    div[data-testid="stAppViewContainer"] button {
        background-color: #000000 !important;
        color: #FFFFFF !important;
        border-radius: 0 !important;
        border: 2px solid #000000 !important;
        font-weight: 800 !important;
        padding: 1rem 2rem !important;
        width: 100%;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    div[data-testid="stAppViewContainer"] button:hover {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        box-shadow: 8px 8px 0px #000000 !important;
        transform: translate(-4px, -4px);
    }
    
    div[data-testid="stAppViewContainer"] button:hover * {
        color: #000000 !important;
    }

    /* Input Field */
    .stTextInput>div>div>input {
        border-radius: 0 !important;
        border: 3px solid #000000 !important;
        font-size: 1.2rem !important;
        height: 70px !important;
        background-color: #FFFFFF !important;
        color: #000000 !important;
        padding: 0 20px !important;
    }

    .answer-panel {
        background: #000000;
        color: #FFFFFF !important;
        padding: 45px;
        font-size: 1.25rem;
        line-height: 1.7;
        border: 5px solid #000000;
        margin-top: 30px;
    }

    /* Expander / Evidence Fix */
    .stExpander {
        border: 2px solid #000000 !important;
        border-radius: 0 !important;
        background-color: #FFFFFF !important;
        margin-bottom: 15px !important;
    }
    
    .stExpander summary {
        background-color: #000000 !important;
        color: #FFFFFF !important;
        padding: 10px 15px !important;
    }
    
    .stExpander summary:hover {
        background-color: #333333 !important;
    }
    
    .stExpander summary * {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    .stExpander [data-testid="stExpanderDetails"] {
        padding: 20px !important;
        border-top: 2px solid #000000 !important;
    }
    
    .stExpander [data-testid="stExpanderDetails"] * {
        color: #000000 !important;
    }

    [data-testid="stHeader"] {
        background-color: rgba(255, 255, 255, 0.9) !important;
    }

    /* Adjust Deploy Button Position */
    [data-testid="stAppDeploy"] {
        margin-right: 25px !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'qa_system' not in st.session_state:
    st.session_state.qa_system = None
if 'system_ready' not in st.session_state:
    st.session_state.system_ready = False

# Sidebar Branding
with st.sidebar:
    st.markdown("<h1 style='color: white; font-family: Outfit; font-size: 2.5rem; letter-spacing: -2px;'>FINRATE</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #666; letter-spacing: 2px; font-weight: 600;'>BETA v3.3</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    if not st.session_state.system_ready:
        if st.button("BOOT ENGINE"):
            with st.spinner("QUANTIZING..."):
                try:
                    qa_system = AcademicQASystem(use_lsh=True, use_simhash=True, use_tfidf=True, use_llm=True)
                    handbook_dir = Path("data/handbooks")
                    pdf_files = list(handbook_dir.glob("*.pdf")) if handbook_dir.exists() else []
                    
                    if pdf_files:
                        for pdf_file in pdf_files:
                            qa_system.add_document(str(pdf_file), pdf_file.stem)
                        qa_system.fit_baseline()
                        st.session_state.qa_system = qa_system
                        st.session_state.system_ready = True
                        st.success("CORE ONLINE")
                    else:
                        st.error("DATA ERROR")
                except Exception as e:
                    st.error(f"FATAL: {e}")
    else:
        st.success("STATUS: OPTIMAL")
        st.markdown("### CONFIG")
        method_label = st.selectbox("ALGORITHM", ["LSH / MINHASH", "SIMHASH", "TF-IDF / EXACT"])
        top_k = st.slider("CANDIDATES", 1, 15, 5)
        ans_mode = st.radio("SYNTHESIS", ["EXTRACTIVE", "GEMINI FLASH"], horizontal=True)
        
        method_map = {"LSH / MINHASH": "lsh", "SIMHASH": "simhash", "TF-IDF / EXACT": "tfidf"}

# Main Content
st.markdown('<div class="brand-title">Finrate.</div>', unsafe_allow_html=True)
st.markdown('<div class="brand-subtitle">Automated Policy Governance & Retrieval</div>', unsafe_allow_html=True)

if st.session_state.system_ready:
    tab_qa, tab_stats, tab_bench = st.tabs(["[ QUERY ]", "[ ANALYTICS ]", "[ BENCHMARK ]"])
    
    with tab_qa:
        query = st.text_input("ENTER QUESTION", placeholder="Search academic statutes...", label_visibility="collapsed")
        search_btn = st.button("SEARCH")
        
        if (search_btn or query) and query:
            with st.spinner("RETRIEVING..."):
                start_t = time.time()
                result = st.session_state.qa_system.answer_query(
                    query, 
                    method=method_map[method_label], 
                    top_k=top_k,
                    answer_method='llm' if "GEMINI" in ans_mode else 'extractive'
                )
                duration = time.time() - start_t
                
                # Result Display
                st.markdown(f'<div class="answer-panel">{result["answer"]}</div>', unsafe_allow_html=True)
                
                # Metrics Containers
                st.markdown("<br>", unsafe_allow_html=True)
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                
                chunks_count = len(result["retrieved_chunks"])
                top_sim = result["retrieved_chunks"][0]["similarity"] if chunks_count > 0 else 0.0
                
                with m_col1:
                    st.markdown(f'<div class="mono-card"><div class="metric-lab">LATENCY</div><div class="metric-val">{duration:.4f}s</div></div>', unsafe_allow_html=True)
                with m_col2:
                    st.markdown(f'<div class="mono-card"><div class="metric-lab">SIMILARITY</div><div class="metric-val">{top_sim:.2f}</div></div>', unsafe_allow_html=True)
                with m_col3:
                    st.markdown(f'<div class="mono-card"><div class="metric-lab">CHUNKS</div><div class="metric-val">{chunks_count}</div></div>', unsafe_allow_html=True)
                with m_col4:
                    st.markdown(f'<div class="mono-card"><div class="metric-lab">ENGINE</div><div class="metric-val">3.0F</div></div>', unsafe_allow_html=True)
                
                # Evidence
                st.markdown("<br>### EVIDENCE", unsafe_allow_html=True)
                if chunks_count > 0:
                    for i, chunk in enumerate(result["retrieved_chunks"]):
                        with st.expander(f"CHUNK_{i+1} | SCORE_{chunk['similarity']:.3f} | SOURCE_{chunk['source']}"):
                            st.markdown(f"<div style='font-family: monospace; font-size: 1rem; color: #000;'>{chunk['text']}</div>", unsafe_allow_html=True)
                else:
                    st.warning("NO RELEVANT DATA LOCATED")

    with tab_stats:
        stats = st.session_state.qa_system.get_statistics()
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="mono-card"><div class="metric-lab">CORPUS SCALE</div><div class="metric-val">{stats["total_chunks"]}</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="mono-card"><div class="metric-lab">TOKENS</div><div class="metric-val">{stats["total_tokens"]:,}</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="mono-card"><div class="metric-lab">HOT TOPICS</div><div class="metric-val">{len(stats["hot_topics"])}</div></div>', unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        ch_col1, ch_col2 = st.columns(2)
        with ch_col1:
            st.markdown("### TREND ANALYSIS")
            if stats['hot_topics']:
                df = pd.DataFrame(stats['hot_topics'], columns=['Pattern', 'Count'])
                fig = px.bar(df, x='Count', y='Pattern', orientation='h', color_discrete_sequence=['#000000'])
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', xaxis_title="", yaxis_title="", font_family="Inter")
                st.plotly_chart(fig, use_container_width=True)
        with ch_col2:
            st.markdown("### LATENCY DISTRIBUTION")
            perf = stats['performance_summary']
            if perf:
                p_df = pd.DataFrame([{'Method': m.upper(), 'Avg Time': s['avg_time']} for m, s in perf.items()])
                fig_p = px.pie(p_df, values='Avg Time', names='Method', color_discrete_sequence=['#000000', '#666666', '#CCCCCC'])
                fig_p.update_traces(hole=0.4)
                fig_p.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_p, use_container_width=True)

    with tab_bench:
        st.markdown("### STRESS TEST")
        if st.button("RUN SCALABILITY SUITE"):
            from src.experiments import ExperimentalEvaluation
            exp = ExperimentalEvaluation(st.session_state.qa_system)
            with st.spinner("SIMULATING SCALE..."):
                res = exp.run_all_experiments()
                df_scale = pd.DataFrame(res['scalability']['scaling_tests'])
                fig_sc = go.Figure()
                fig_sc.add_trace(go.Scatter(x=df_scale['doc_count'], y=df_scale['lsh_query_time'], name='LSH', line=dict(color='#000000', width=4)))
                fig_sc.add_trace(go.Scatter(x=df_scale['doc_count'], y=df_scale['tfidf_query_time'], name='EXACT', line=dict(color='#999999', width=2, dash='dot')))
                fig_sc.update_layout(title="", xaxis_title="DOC_COUNT", yaxis_title="SECONDS", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_sc, use_container_width=True)

else:
    st.markdown("""
    <div style="text-align: center; margin-top: 100px;">
        <h2 style="font-family: Outfit; font-size: 3rem; font-weight: 800; color: #000;">Engine Standby</h2>
        <p style="color: #666; letter-spacing: 4px; font-weight: 600;">READY FOR STATUTE QUANTIZATION</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #000; font-size: 0.75rem; letter-spacing: 3px; font-weight: 700;'>FINRATE | BIG DATA ARCHITECTURE | 2025</p>", unsafe_allow_html=True)
