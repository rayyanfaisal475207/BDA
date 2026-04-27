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
    
    /* Sidebar Buttons - Ultra Specific Fix */
    div[data-testid="stSidebar"] button {
        background-color: #FFFFFF !important;
        border: 2px solid #FFFFFF !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
        border-radius: 0 !important;
    }
    
    div[data-testid="stSidebar"] button p {
        color: #000000 !important;
        font-weight: 800 !important;
    }
    
    div[data-testid="stSidebar"] button:hover {
        background-color: #000000 !important;
        border: 2px solid #FFFFFF !important;
    }
    
    div[data-testid="stSidebar"] button:hover p {
        color: #FFFFFF !important;
    }

    /* Sidebar Selectbox & Inputs Fix - MAXIMUM SPECIFICITY */
    section[data-testid="stSidebar"] div[data-baseweb="select"] {
        background-color: #FFFFFF !important;
        border-radius: 0 !important;
    }

    section[data-testid="stSidebar"] div[data-baseweb="select"] * {
        color: #000000 !important;
        border-radius: 0 !important;
    }
    
    /* Target the actual text span that Streamlit uses */
    section[data-testid="stSidebar"] [data-baseweb="select"] [role="button"] {
        color: #000000 !important;
    }
    
    /* Ensure any input text is also black */
    section[data-testid="stSidebar"] [data-baseweb="select"] input {
        color: #000000 !important;
    }

    /* Target the dropdown menu (popover) */
    div[data-baseweb="popover"] * {
        color: #000000 !important;
        border-radius: 0 !important;
    }

    /* Section Headers */
    .section-header {
        font-family: 'Outfit', sans-serif;
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: 2px;
        color: #000000;
        margin-top: 40px;
        margin-bottom: 20px;
        border-left: 10px solid #000000;
        padding-left: 20px;
        text-transform: uppercase;
    }

    /* Sidebar Labels & Radio Buttons */
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #FFFFFF !important;
        font-weight: 800 !important;
        letter-spacing: 1px !important;
    }
    
    section[data-testid="stSidebar"] [data-testid="stRadio"] * {
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

    /* --- SEARCH INPUT: ARCHITECTURAL REWRITE --- */
    .stTextInput {
        margin-top: 2rem !important;
    }

    /* Target the container wrapper */
    div[data-testid="stTextInput"] > div {
        background-color: transparent !important;
    }

    /* Target the actual input element with maximum authority */
    div[data-testid="stTextInput"] input {
        border-radius: 0 !important;
        border: 4px solid #000000 !important;
        background-color: #FFFFFF !important;
        color: #000000 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 1.6rem !important;
        font-weight: 600 !important;
        height: 120px !important;
        line-height: 120px !important; /* Perfect Vertical Centering */
        padding: 0 40px !important;
        transition: all 0.4s cubic-bezier(0.165, 0.84, 0.44, 1) !important;
        box-shadow: 0px 0px 0px rgba(0,0,0,0) !important;
        cursor: text !important;
    }

    /* Hover & Focus State: 3D Interaction */
    div[data-testid="stTextInput"] input:hover,
    div[data-testid="stTextInput"] input:focus {
        box-shadow: 15px 15px 0px #000000 !important;
        transform: translate(-8px, -8px) !important;
        outline: none !important;
        border: 4px solid #000000 !important;
    }

    /* Placeholder Text Styling */
    div[data-testid="stTextInput"] input::placeholder {
        color: #BBBBBB !important;
        opacity: 1 !important;
        font-weight: 300 !important;
        letter-spacing: 1px !important;
    }

    /* Main Area Answer Panel */
    .answer-panel {
        background: #000000;
        color: #FFFFFF !important;
        padding: 60px;
        font-size: 1.4rem;
        line-height: 1.85;
        border: 5px solid #000000;
        margin-top: 40px;
        font-family: 'Inter', sans-serif;
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
        search_btn = st.button("SEARCH", width="stretch")
        
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
                
                # Metrics row
                st.markdown("<br>", unsafe_allow_html=True)
                m_col1, m_col2, m_col3, m_col4 = st.columns(4, gap="medium")

                chunks_count = len(result["retrieved_chunks"])
                top_conf = result["retrieved_chunks"][0].get("confidence", 0.0) if chunks_count > 0 else 0.0
                top_sim  = result["retrieved_chunks"][0]["similarity"] if chunks_count > 0 else 0.0

                with m_col1:
                    st.markdown(f'<div class="mono-card"><div class="metric-lab">LATENCY</div><div class="metric-val">{duration:.4f}s</div></div>', unsafe_allow_html=True)
                with m_col2:
                    st.markdown(f'<div class="mono-card"><div class="metric-lab">CONFIDENCE</div><div class="metric-val">{top_conf:.0%}</div></div>', unsafe_allow_html=True)
                with m_col3:
                    st.markdown(f'<div class="mono-card"><div class="metric-lab">CHUNKS</div><div class="metric-val">{chunks_count}</div></div>', unsafe_allow_html=True)
                with m_col4:
                    st.markdown(f'<div class="mono-card"><div class="metric-lab">SIMILARITY</div><div class="metric-val">{top_sim:.4f}</div></div>', unsafe_allow_html=True)

                # Evidence
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### EVIDENCE")
                if chunks_count > 0:
                    for i, chunk in enumerate(result["retrieved_chunks"]):
                        conf_pct = chunk.get('confidence', 0.0)
                        bar_w = max(2, int(conf_pct * 100))
                        with st.expander(f"CHUNK_{i+1:02d}  ·  CONF {conf_pct:.0%}  ·  SIM {chunk['similarity']:.4f}  ·  {chunk['source']}"):
                            st.markdown(
                                f'<div style="background:#EEE;height:4px;width:100%;margin-bottom:12px;border-radius:0;">'
                                f'<div style="background:#000;height:4px;width:{bar_w}%;"></div></div>'
                                f'<div style="font-family:monospace;font-size:0.95rem;color:#111;line-height:1.75;">{chunk["text"]}</div>',
                                unsafe_allow_html=True
                            )
                else:
                    st.warning("NO RELEVANT DATA LOCATED")

    with tab_stats:
        stats = st.session_state.qa_system.get_statistics()
        total_q = stats.get('total_queries', 0)

        _EMPTY = '<div style="border:2px dashed #CCC;padding:50px;text-align:center;color:#AAA;font-size:0.75rem;letter-spacing:2px;font-family:Inter;">RUN QUERIES TO POPULATE</div>'

        # ── Row 1: 4 metric cards ───────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4, gap="medium")
        with c1:
            st.markdown(f'<div class="mono-card"><div class="metric-lab">CORPUS SCALE</div><div class="metric-val">{stats["total_chunks"]}</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="mono-card"><div class="metric-lab">TOKENS</div><div class="metric-val">{stats["total_tokens"]:,}</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="mono-card"><div class="metric-lab">QUERIES RUN</div><div class="metric-val">{total_q}</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="mono-card"><div class="metric-lab">HOT TOPICS</div><div class="metric-val">{len(stats["hot_topics"])}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Row 2: TREEMAP (Hot Topics) + BUBBLE SCATTER (Method Performance) ─
        ch_col1, ch_col2 = st.columns(2, gap="large")

        with ch_col1:
            st.markdown("### QUERY PATTERN MINING")
            st.markdown("<small style='color:#888;letter-spacing:1px;'>Frequent Itemset Analysis (Apriori) · area ∝ frequency</small>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            if stats['hot_topics']:
                df_ht = pd.DataFrame(stats['hot_topics'], columns=['Pattern', 'Count'])
                fig_ht = px.treemap(
                    df_ht, path=['Pattern'], values='Count',
                    color='Count',
                    color_continuous_scale=['#c7d2fe', '#6366f1', '#3730a3'],
                )
                fig_ht.update_traces(
                    textinfo='label+value',
                    textfont=dict(size=13, family='Inter'),
                    marker_line_width=2,
                    marker_line_color='white'
                )
                fig_ht.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=370,
                    margin=dict(l=0, r=0, t=10, b=0),
                    coloraxis_showscale=False,
                    font_family="Inter"
                )
                st.plotly_chart(fig_ht, use_container_width=True, config={'displayModeBar': False})
            else:
                st.markdown(_EMPTY, unsafe_allow_html=True)

        with ch_col2:
            st.markdown("### METHOD PERFORMANCE")
            st.markdown("<small style='color:#888;letter-spacing:1px;'>Speed–Recall tradeoff · bubble size = queries run</small>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            perf = stats['performance_summary']
            if perf:
                _MLABELS = {'lsh': 'LSH / MinHash', 'simhash': 'SimHash', 'tfidf': 'TF-IDF'}
                _MCOLORS = {'LSH / MinHash': '#6366f1', 'SimHash': '#10b981', 'TF-IDF': '#f59e0b'}
                p_df = pd.DataFrame([
                    {
                        'Method': _MLABELS.get(m, m.upper()),
                        'Latency (ms)': round(s['avg_time'] * 1000, 3),
                        'Avg Results': round(s['avg_results'], 1),
                        'Queries': max(s['total_queries'], 1),
                    }
                    for m, s in perf.items()
                ])
                fig_p = px.scatter(
                    p_df,
                    x='Latency (ms)', y='Avg Results',
                    size='Queries', color='Method', text='Method',
                    color_discrete_map=_MCOLORS,
                    size_max=65,
                )
                fig_p.update_traces(
                    textposition='top center',
                    textfont=dict(size=11, family='Inter', color='#222'),
                    marker_line_width=2, marker_line_color='#1a1a1a'
                )
                fig_p.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    font_family="Inter", height=370,
                    margin=dict(l=40, r=20, t=10, b=50),
                    xaxis_title="Avg Latency (ms) — lower is faster",
                    yaxis_title="Avg Chunks Retrieved",
                    showlegend=False
                )
                fig_p.update_xaxes(showgrid=True, gridcolor='#EEE', zeroline=False)
                fig_p.update_yaxes(showgrid=True, gridcolor='#EEE', zeroline=False)
                st.plotly_chart(fig_p, use_container_width=True, config={'displayModeBar': False})
            else:
                st.markdown(_EMPTY, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Row 3: SCATTER BUBBLE (Section Importance / PageRank-style) ────────
        sec_imp = stats.get('section_importance', [])
        st.markdown(
            "### SECTION IMPORTANCE"
            "<small style='color:#888;font-size:0.65rem;letter-spacing:2px;font-weight:400;margin-left:12px;'>"
            "PageRank-style · bubble size & colour = query access count</small>",
            unsafe_allow_html=True
        )
        if sec_imp:
            si_df = pd.DataFrame(sec_imp)
            si_df['page_num'] = pd.to_numeric(
                si_df['label'].str.extract(r'p(\d+)', expand=False), errors='coerce'
            ).fillna(0)
            fig_si = px.scatter(
                si_df,
                x='page_num', y='hits',
                size='hits', color='hits',
                hover_name='label',
                text='hits',
                color_continuous_scale=['#fde68a', '#f59e0b', '#ef4444'],
                size_max=55,
                labels={'page_num': 'Page Number', 'hits': 'Access Count'},
            )
            fig_si.update_traces(
                textposition='top center',
                textfont=dict(size=10, family='Inter'),
                marker_line_width=2, marker_line_color='#333'
            )
            fig_si.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font_family="Inter", height=300,
                margin=dict(l=20, r=20, t=10, b=40),
                coloraxis_showscale=False,
            )
            fig_si.update_xaxes(showgrid=True, gridcolor='#EEE', title='Page Number')
            fig_si.update_yaxes(showgrid=True, gridcolor='#EEE', title='Query Access Count')
            st.plotly_chart(fig_si, use_container_width=True, config={'displayModeBar': False})
        else:
            st.markdown(_EMPTY, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Row 4: LINE CHART (Latency Trend over Queries) + history table ─────
        q_hist = stats.get('query_history', [])
        st.markdown(
            "### QUERY LATENCY TREND"
            "<small style='color:#888;font-size:0.65rem;letter-spacing:2px;font-weight:400;margin-left:12px;'>"
            "per-query latency coloured by retrieval method</small>",
            unsafe_allow_html=True
        )
        if q_hist:
            chron = list(reversed(q_hist))  # oldest → newest
            hist_df = pd.DataFrame(chron)
            hist_df.insert(0, 'Query #', range(1, len(hist_df) + 1))
            _HIST_COLORS = {'LSH': '#6366f1', 'SIMHASH': '#10b981', 'TFIDF': '#f59e0b'}
            fig_hist = px.line(
                hist_df, x='Query #', y='time_ms', color='method',
                markers=True,
                color_discrete_map=_HIST_COLORS,
                labels={'time_ms': 'Latency (ms)', 'method': 'Method'},
            )
            fig_hist.update_traces(line_width=2.5, marker_size=8)
            fig_hist.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font_family="Inter", height=260,
                margin=dict(l=20, r=20, t=10, b=40),
                legend=dict(orientation='h', y=1.1, bgcolor='rgba(0,0,0,0)', font_size=12),
                xaxis=dict(dtick=1),
            )
            fig_hist.update_xaxes(showgrid=True, gridcolor='#EEE')
            fig_hist.update_yaxes(showgrid=True, gridcolor='#EEE')
            st.plotly_chart(fig_hist, use_container_width=True, config={'displayModeBar': False})

            st.markdown("<br>", unsafe_allow_html=True)
            disp = pd.DataFrame(chron[-10:])
            disp.columns = ['QUERY', 'METHOD', 'TIME (ms)', 'RESULTS']
            st.dataframe(disp, use_container_width=True, hide_index=True)
        else:
            st.markdown(_EMPTY, unsafe_allow_html=True)

    with tab_bench:
        st.markdown("### SCALABILITY & PARAMETER ANALYSIS")
        b_col1, b_col2 = st.columns([1, 1], gap="large")

        with b_col1:
            st.markdown("#### STRESS TEST — Corpus Scaling")
            st.markdown("<small style='color:#666;'>Compares LSH vs TF-IDF query latency as corpus grows 1x → 10x</small>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("RUN SCALABILITY SUITE"):
                from src.experiments import ExperimentalEvaluation
                exp = ExperimentalEvaluation(st.session_state.qa_system)
                with st.spinner("SIMULATING SCALE (1x – 10x)..."):
                    res = exp.run_all_experiments()
                    st.session_state['bench_results'] = res

        with b_col2:
            st.markdown("#### PARAMETER SENSITIVITY — LSH Bands")
            st.markdown("<small style='color:#666;'>How many candidates are returned at different band counts</small>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("RUN PARAMETER SWEEP"):
                from src.experiments import ExperimentalEvaluation
                exp2 = ExperimentalEvaluation(st.session_state.qa_system)
                with st.spinner("SWEEPING PARAMETERS..."):
                    param_res = exp2.analyze_parameter_sensitivity()
                    st.session_state['param_results'] = param_res

        # Show cached results if available
        if 'bench_results' in st.session_state:
            res = st.session_state['bench_results']
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### SCALABILITY RESULTS")
            df_scale = pd.DataFrame(res['scalability']['scaling_tests'])
            fig_sc = go.Figure()
            fig_sc.add_trace(go.Scatter(
                x=df_scale['doc_count'], y=df_scale['lsh_query_time'],
                name='LSH (Approximate)', line=dict(color='#000000', width=3)
            ))
            fig_sc.add_trace(go.Scatter(
                x=df_scale['doc_count'], y=df_scale['tfidf_query_time'],
                name='TF-IDF (Exact)', line=dict(color='#999999', width=2, dash='dash')
            ))
            fig_sc.update_layout(
                xaxis_title="Document Count", yaxis_title="Query Time (s)",
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font_family="Inter", height=380,
                legend=dict(bgcolor='rgba(0,0,0,0)'),
                margin=dict(l=20, r=20, t=20, b=40)
            )
            fig_sc.update_xaxes(showgrid=True, gridcolor='#EEE')
            fig_sc.update_yaxes(showgrid=True, gridcolor='#EEE')
            st.plotly_chart(fig_sc, use_container_width=True, config={'displayModeBar': False})

            # Speedup table
            if not df_scale.empty:
                df_scale['speedup_x'] = (df_scale['tfidf_query_time'] / df_scale['lsh_query_time'].replace(0, 1e-9)).round(1)
                st.dataframe(
                    df_scale[['doc_count', 'lsh_query_time', 'tfidf_query_time', 'speedup_x']].rename(
                        columns={'doc_count': 'Docs', 'lsh_query_time': 'LSH (s)',
                                 'tfidf_query_time': 'TF-IDF (s)', 'speedup_x': 'Speedup'}
                    ),
                    use_container_width=True, hide_index=True
                )

        if 'param_results' in st.session_state:
            param_res = st.session_state['param_results']
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### PARAMETER SENSITIVITY RESULTS")
            p2_col1, p2_col2 = st.columns(2, gap="large")

            with p2_col1:
                st.markdown("**Band Count vs Candidates Retrieved**")
                bands_data = param_res.get('bands', {})
                if bands_data:
                    b_df = pd.DataFrame([
                        {'Bands': k, 'Candidates': v['results'], 'Time (s)': round(v['time'], 6)}
                        for k, v in sorted(bands_data.items())
                    ])
                    fig_b = px.line(b_df, x='Bands', y='Candidates', markers=True,
                                   color_discrete_sequence=['#000000'])
                    fig_b.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                        font_family="Inter", height=280,
                        margin=dict(l=20, r=20, t=10, b=40)
                    )
                    fig_b.update_xaxes(showgrid=True, gridcolor='#EEE')
                    fig_b.update_yaxes(showgrid=True, gridcolor='#EEE')
                    st.plotly_chart(fig_b, use_container_width=True, config={'displayModeBar': False})

            with p2_col2:
                st.markdown("**Hash Functions vs Candidates**")
                hf_data = param_res.get('hash_functions', {})
                if hf_data:
                    hf_df = pd.DataFrame([
                        {'Hashes': k, 'Candidates': v['results'], 'Time (s)': round(v['time'], 6)}
                        for k, v in sorted(hf_data.items())
                    ])
                    fig_hf = px.line(hf_df, x='Hashes', y='Candidates', markers=True,
                                    color_discrete_sequence=['#333333'])
                    fig_hf.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                        font_family="Inter", height=280,
                        margin=dict(l=20, r=20, t=10, b=40)
                    )
                    fig_hf.update_xaxes(showgrid=True, gridcolor='#EEE')
                    fig_hf.update_yaxes(showgrid=True, gridcolor='#EEE')
                    st.plotly_chart(fig_hf, use_container_width=True, config={'displayModeBar': False})

else:
    st.markdown("""
    <div style="text-align: center; margin-top: 100px;">
        <h2 style="font-family: Outfit; font-size: 3rem; font-weight: 800; color: #000;">Engine Standby</h2>
        <p style="color: #666; letter-spacing: 4px; font-weight: 600;">READY FOR STATUTE QUANTIZATION</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #000; font-size: 0.75rem; letter-spacing: 3px; font-weight: 700;'>FINRATE | BIG DATA ANALYSIS | 2026</p>", unsafe_allow_html=True)
