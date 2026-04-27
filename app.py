"""
Streamlit web interface for the Academic QA System.
"""

import streamlit as st
import json
from pathlib import Path
from src.qa_system import AcademicQASystem
import os

# Page configuration
st.set_page_config(
    page_title="Academic Policy QA System",
    page_icon="📚",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .answer-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .chunk-box {
        background-color: #e8f4f8;
        padding: 10px;
        border-left: 4px solid #1f77b4;
        margin: 10px 0;
    }
    .metric-box {
        background-color: #fff7ed;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("📚 Academic Policy QA System")
st.markdown("""
A scalable question-answering system for university handbooks using **Locality Sensitive Hashing (LSH)** and Big Data techniques.

### Features:
- **MinHash + LSH**: Efficient approximate similarity search
- **SimHash**: Fingerprint-based document similarity
- **TF-IDF Baseline**: Exact similarity matching
- **Multiple Retrieval Methods**: Compare exact vs approximate approaches
""")

# Sidebar configuration
st.sidebar.title("⚙️ System Configuration")

# Initialize session state
if 'qa_system' not in st.session_state:
    st.session_state.qa_system = None
if 'system_ready' not in st.session_state:
    st.session_state.system_ready = False

# System initialization
with st.sidebar:
    st.subheader("Initialize System")

    # Check for sample data
    sample_data_dir = Path("data/sample_handbooks")
    sample_files = list(sample_data_dir.glob("*.txt")) if sample_data_dir.exists() else []

    if st.button("🚀 Initialize QA System"):
        with st.spinner("Initializing system..."):
            try:
                # Create QA system
                qa_system = AcademicQASystem(
                    use_lsh=True,
                    use_simhash=True,
                    use_tfidf=True,
                    use_llm=False
                )

                # Load sample data
                if sample_files:
                    for txt_file in sample_files[:5]:  # Load first 5 files
                        doc_id = txt_file.stem
                        with open(txt_file, 'r') as f:
                            text = f.read()

                        # Convert to chunks manually for demo
                        from src.data_processing import DocumentProcessor
                        processor = DocumentProcessor()
                        words = text.split()
                        chunk_num = 0

                        for i in range(0, len(words), 300):
                            chunk_words = words[i:i+300]
                            if len(chunk_words) > 10:
                                chunk_text = " ".join(chunk_words)
                                chunk_id = f"{doc_id}_chunk_{chunk_num}"

                                qa_system.documents[chunk_id] = chunk_text
                                qa_system.doc_metadata[chunk_id] = {
                                    'source': doc_id,
                                    'page': chunk_num
                                }

                                # Index with LSH
                                tokens = set(processor.tokenize(chunk_text))
                                qa_system.lsh.index_document(chunk_id, tokens)

                                # SimHash fingerprints
                                tokens_list = processor.tokenize(chunk_text)
                                qa_system.simhash_fingerprints[chunk_id] = \
                                    qa_system.simhash.compute_fingerprint(tokens_list)

                                chunk_num += 1

                    qa_system.fit_baseline()
                    st.session_state.qa_system = qa_system
                    st.session_state.system_ready = True
                    st.success("✅ System initialized successfully!")
                else:
                    st.warning("⚠️ No sample data found. Using empty system.")
                    st.session_state.qa_system = qa_system
                    st.session_state.system_ready = True

            except Exception as e:
                st.error(f"❌ Error initializing system: {e}")

    # Configuration options
    st.divider()
    st.subheader("Retrieval Settings")

    retrieval_method = st.radio(
        "Select retrieval method:",
        ["LSH (MinHash)", "SimHash", "TF-IDF (Baseline)"],
        help="LSH and SimHash are approximate; TF-IDF is exact"
    )

    method_map = {
        "LSH (MinHash)": "lsh",
        "SimHash": "simhash",
        "TF-IDF (Baseline)": "tfidf"
    }

    top_k = st.slider("Number of results (top-k):", 1, 20, 5)

    answer_method = st.radio(
        "Answer generation:",
        ["Extractive", "LLM (requires API key)"]
    )

# Main interface
if st.session_state.system_ready and st.session_state.qa_system:
    qa_system = st.session_state.qa_system

    st.divider()
    st.subheader("📝 Query the System")

    # Sample queries
    with st.expander("📌 Example Queries"):
        st.markdown("""
        - What is the minimum GPA requirement?
        - What happens if a student fails a course?
        - What is the attendance policy?
        - How many times can a course be repeated?
        - What are the requirements for graduation?
        """)

    # Query input
    query = st.text_area(
        "Enter your question:",
        placeholder="What is the minimum GPA requirement?",
        height=80
    )

    # Query button
    col1, col2, col3 = st.columns(3)
    with col1:
        search_button = st.button("🔍 Search", use_container_width=True)
    with col2:
        clear_button = st.button("🗑️ Clear", use_container_width=True)

    if clear_button:
        st.rerun()

    # Process query
    if search_button and query.strip():
        with st.spinner("Processing query..."):
            try:
                result = qa_system.answer_query(
                    query,
                    method=method_map[retrieval_method],
                    top_k=top_k,
                    answer_method='llm' if 'LLM' in answer_method else 'extractive'
                )

                # Display answer
                st.divider()
                st.subheader("💡 Answer")
                st.markdown(f'<div class="answer-box">{result["answer"]}</div>',
                           unsafe_allow_html=True)

                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f'<div class="metric-box"><b>Method:</b> {result["method"]}</div>',
                               unsafe_allow_html=True)
                with col2:
                    retrieval_time = result.get('timing', {}).get('retrieval_time', 0)
                    st.markdown(f'<div class="metric-box"><b>Retrieval Time:</b> {retrieval_time:.4f}s</div>',
                               unsafe_allow_html=True)
                with col3:
                    st.markdown(f'<div class="metric-box"><b>Results:</b> {len(result["retrieved_chunks"])}</div>',
                               unsafe_allow_html=True)

                # Display retrieved chunks
                st.divider()
                st.subheader("📖 Retrieved Chunks")

                for idx, chunk in enumerate(result["retrieved_chunks"], 1):
                    with st.expander(f"Chunk {idx} (Similarity: {chunk['similarity']:.4f}) - Source: {chunk['source']}"):
                        st.markdown(f'<div class="chunk-box">{chunk["text"]}</div>',
                                   unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Error processing query: {e}")
    elif search_button:
        st.warning("⚠️ Please enter a query")

else:
    st.info("👈 Click 'Initialize System' in the sidebar to start")

# Footer
st.divider()
st.markdown("""
### About This System
This system implements **Locality Sensitive Hashing (LSH)** for efficient document retrieval.
- **MinHash + LSH**: Approximate Jaccard similarity with O(1) query time
- **SimHash**: Hamming distance-based fingerprints
- **TF-IDF**: Baseline exact similarity matching

For more information, see the project documentation.
""")
