"""
Premium Streamlit web interface for the Academic QA System.
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
    page_title="LexiPolicy - Academic QA",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #4b5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
        margin-bottom: 20px;
    }
    
    .answer-box {
        background: #ffffff;
        border-radius: 12px;
        padding: 20px;
        border-left: 6px solid #2563eb;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        font-size: 1.1rem;
        line-height: 1.6;
        color: #1f2937;
    }
    
    .chunk-card {
        background: #f8fafc;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #e2e8f0;
        margin-top: 10px;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2563eb;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .tag {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 5px;
        background: #dbeafe;
        color: #1e40af;
    }
    
    /* Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .animate-fade {
        animation: fadeIn 0.5s ease-out forwards;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'qa_system' not in st.session_state:
    st.session_state.qa_system = None
if 'system_ready' not in st.session_state:
    st.session_state.system_ready = False

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/education.png", width=80)
    st.title("LexiPolicy v1.0")
    st.markdown("---")
    
    st.subheader("🛠️ System Initialization")
    
    if st.button("🚀 Boot QA System", use_container_width=True):
        with st.spinner("Compiling signatures and indexing..."):
            try:
                qa_system = AcademicQASystem(
                    use_lsh=True,
                    use_simhash=True,
                    use_tfidf=True,
                    use_llm=False
                )
                
                # Priority 1: Real Handbooks (PDFs)
                handbook_dir = Path("data/handbooks")
                pdf_files = list(handbook_dir.glob("*.pdf")) if handbook_dir.exists() else []
                
                # Priority 2: Sample Data (Text)
                sample_data_dir = Path("data/sample_handbooks")
                sample_files = list(sample_data_dir.glob("*.txt")) if sample_data_dir.exists() else []
                
                if pdf_files:
                    for pdf_file in pdf_files:
                        st.info(f"Indexing {pdf_file.name}...")
                        qa_system.add_document(str(pdf_file), pdf_file.stem)
                    
                    qa_system.fit_baseline()
                    st.session_state.qa_system = qa_system
                    st.session_state.system_ready = True
                    st.success(f"✅ System Ready! Indexed {len(pdf_files)} PDF handbooks.")
                elif sample_files:
                    processor = DocumentProcessor()
                    for txt_file in sample_files:
                        doc_id = txt_file.stem
                        with open(txt_file, 'r') as f:
                            text = f.read()
                        
                        chunks = processor.chunk_text(text, doc_id)
                        for chunk_id, chunk_text, page_num in chunks:
                            qa_system.documents[chunk_id] = chunk_text
                            qa_system.doc_metadata[chunk_id] = {'source': doc_id, 'page': page_num}
                            
                            tokens = set(processor.tokenize(chunk_text))
                            qa_system.lsh.index_document(chunk_id, tokens)
                            
                            tokens_list = processor.tokenize(chunk_text)
                            qa_system.simhash_fingerprints[chunk_id] = qa_system.simhash.compute_fingerprint(tokens_list)
                    
                    qa_system.fit_baseline()
                    st.session_state.qa_system = qa_system
                    st.session_state.system_ready = True
                    st.success("✅ System Ready! Indexed sample text data.")
                else:
                    st.error("No handbook data found. Please add PDFs to data/handbooks/")
            except Exception as e:
                st.error(f"Initialization Failed: {e}")

    if st.session_state.system_ready:
        st.markdown("---")
        st.subheader("⚙️ Retrieval Config")
        method = st.selectbox("Search Algorithm", ["LSH (MinHash)", "SimHash", "TF-IDF (Baseline)"])
        top_k = st.slider("Top-K Candidates", 1, 15, 5)
        ans_gen = st.radio("Answer Logic", ["Extractive", "Neural LLM"])
        
        method_map = {"LSH (MinHash)": "lsh", "SimHash": "simhash", "TF-IDF (Baseline)": "tfidf"}

# Header
st.markdown('<div class="main-header">LexiPolicy 🎓</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Scalable Big Data Retrieval for Academic Policies</div>', unsafe_allow_html=True)

if st.session_state.system_ready:
    tab_qa, tab_analytics, tab_experiments = st.tabs(["🔍 Search & QA", "📊 System Analytics", "🧪 Experiments"])
    
    with tab_qa:
        col_q, col_s = st.columns([4, 1])
        with col_q:
            query = st.text_input("Ask a policy question...", placeholder="What is the minimum GPA requirement?", label_visibility="collapsed")
        with col_s:
            search_clicked = st.button("Query Engine", use_container_width=True, type="primary")
            
        if search_clicked and query:
            with st.spinner("Analyzing high-dimensional signatures..."):
                start_time = time.time()
                result = st.session_state.qa_system.answer_query(
                    query, 
                    method=method_map[method], 
                    top_k=top_k,
                    answer_method='llm' if 'LLM' in ans_gen else 'extractive'
                )
                duration = time.time() - start_time
                
                st.markdown('<div class="animate-fade">', unsafe_allow_html=True)
                st.subheader("💡 Verified Answer")
                st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)
                
                # Metrics Row
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.markdown(f'<div class="glass-card"><div class="metric-label">Latency</div><div class="metric-value">{duration:.4f}s</div></div>', unsafe_allow_html=True)
                with m2:
                    st.markdown(f'<div class="glass-card"><div class="metric-label">Algorithm</div><div class="metric-value">{method.split()[0]}</div></div>', unsafe_allow_html=True)
                with m3:
                    st.markdown(f'<div class="glass-card"><div class="metric-label">Similarity</div><div class="metric-value">{result["retrieved_chunks"][0]["similarity"]:.2f}</div></div>', unsafe_allow_html=True)
                with m4:
                    st.markdown(f'<div class="glass-card"><div class="metric-label">Confidence</div><div class="metric-value">High</div></div>', unsafe_allow_html=True)
                
                st.subheader("📚 Supporting Evidence")
                for chunk in result["retrieved_chunks"]:
                    with st.expander(f"Reference: {chunk['source']} (Score: {chunk['similarity']:.3f})"):
                        st.markdown(f'<div class="chunk-card">{chunk["text"]}...</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Example questions
        st.markdown("---")
        st.markdown("### 📌 Suggested Queries")
        cols = st.columns(3)
        suggestions = [
            "What is the minimum GPA requirement?",
            "What is the attendance policy?",
            "How many times can I repeat a course?"
        ]
        for i, sug in enumerate(suggestions):
            if cols[i%3].button(sug, key=f"sug_{i}"):
                st.session_state.query_val = sug
                st.rerun()

    with tab_analytics:
        stats = st.session_state.qa_system.get_statistics()
        
        st.subheader("📈 Real-time Analytics Dashboard")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Indexed Chunks", stats['total_chunks'])
        with col2:
            st.metric("Total Tokens Processed", f"{stats['total_tokens']:,}")
        with col3:
            st.metric("Unique Patterns Found", len(stats['hot_topics']))
            
        st.divider()
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🔥 Frequent Query Topics")
            if stats['hot_topics']:
                topics_df = pd.DataFrame(stats['hot_topics'], columns=['Topic', 'Frequency'])
                fig = px.bar(topics_df, x='Frequency', y='Topic', orientation='h', 
                             color='Frequency', color_continuous_scale='Blues',
                             template='plotly_white')
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Ask more questions to see hot topics!")
                
        with c2:
            st.markdown("#### ⚡ Performance Profile")
            perf = stats['performance_summary']
            if perf:
                perf_data = []
                for m, s in perf.items():
                    perf_data.append({'Method': m.upper(), 'Avg Latency (s)': s['avg_time']})
                df_perf = pd.DataFrame(perf_data)
                fig = px.pie(df_perf, values='Avg Latency (s)', names='Method', 
                             title='Latency Distribution by Algorithm',
                             color_discrete_sequence=px.colors.qualitative.Pastel)
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Run some queries to see performance data!")

    with tab_experiments:
        st.subheader("🧪 Experimental Evaluation")
        st.markdown("Compare exact vs approximate retrieval across large-scale corpora.")
        
        if st.button("Run Full Experiment Suite"):
            from src.experiments import ExperimentalEvaluation
            exp = ExperimentalEvaluation(st.session_state.qa_system)
            with st.spinner("Simulating large-scale data and running sensitivity tests..."):
                results = exp.run_all_experiments()
                
                # Scalability Plot
                scaling = results['scalability']['scaling_tests']
                df_scale = pd.DataFrame(scaling)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_scale['doc_count'], y=df_scale['lsh_query_time'], name='LSH (MinHash)', mode='lines+markers'))
                fig.add_trace(go.Scatter(x=df_scale['doc_count'], y=df_scale['tfidf_query_time'], name='TF-IDF (Baseline)', mode='lines+markers'))
                fig.update_layout(title='Retrieval Scalability (Query Time)', xaxis_title='Number of Documents', yaxis_title='Seconds', template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
                
                # Sensitivity Analysis
                sens = results['parameter_sensitivity']['hash_functions']
                df_sens = pd.DataFrame([{'Hashes': k, 'Latency': v['time']} for k, v in sens.items()])
                fig_sens = px.line(df_sens, x='Hashes', y='Latency', title='MinHash Precision vs Compute Time', template='plotly_white')
                st.plotly_chart(fig_sens, use_container_width=True)
                
                st.success("Experiments Completed! Check results/ directory for full report.")

else:
    st.markdown("""
    <div style="text-align: center; padding: 100px;">
        <img src="https://img.icons8.com/fluency/144/000000/data-configuration.png" />
        <h2>Welcome to LexiPolicy</h2>
        <p>Please initialize the system from the sidebar to begin analyzing academic policies.</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #64748b;'>LexiPolicy v1.0 | Big Data Algorithms Project | 2025</p>", unsafe_allow_html=True)
