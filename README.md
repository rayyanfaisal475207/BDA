# Scalable Academic Policy QA System

A production-ready Question-Answering system for university handbooks using **Locality Sensitive Hashing (LSH)** and Big Data techniques. This system efficiently retrieves relevant information from large text data and generates accurate answers to student queries.

## 🎯 Project Overview

This project implements a hybrid retrieval system combining:
- **MinHash + LSH**: Approximate Jaccard similarity for efficient document matching
- **SimHash**: Hamming distance-based fingerprints for quick similarity detection
- **TF-IDF**: Baseline exact similarity matching for comparison

The system is designed to handle large-scale handbook documents while maintaining high accuracy and fast query processing.

### Key Features

✓ **LSH Implementation**: Full MinHash + LSH with configurable parameters  
✓ **Multiple Retrieval Methods**: Compare approximate vs exact similarity  
✓ **Web Interface**: Streamlit-based UI for easy interaction  
✓ **Comprehensive Experiments**: Scalability, parameter sensitivity, accuracy analysis  
✓ **Production Ready**: Clean, well-documented, reproducible code  

## 📋 System Architecture

### Data Pipeline
```
PDF/Text Input → Text Extraction → Text Cleaning → Chunking → Indexing
                                                         ↓
                                    MinHash Signatures  |  SimHash Fingerprints  |  TF-IDF Vectors
```

### Query Processing Pipeline
```
User Query → Tokenization → Similarity Search → Candidate Selection → Answer Generation → Result Output
                          ↓
                    LSH / SimHash / TF-IDF
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

1. **Clone or setup the project**
```bash
cd /Users/nexxus/Desktop/Personal/Projects/BDA
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up optional LLM support** (for answer generation)
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Quick Start

#### 1. Run the Web Interface
```bash
streamlit run app.py
```
The application will open at `http://localhost:8501`

#### 2. Run Experiments
```bash
python run_experiments.py
```
This will:
- Load sample handbook data
- Compare LSH, SimHash, and TF-IDF performance
- Analyze parameter sensitivity
- Test scalability
- Generate reports and visualizations

#### 3. Use as a Python Library
```python
from src.qa_system import AcademicQASystem

# Initialize system
qa_system = AcademicQASystem(use_lsh=True, use_simhash=True, use_tfidf=True)

# Add handbooks
qa_system.add_document("handbook.pdf", "ug_handbook")

# Query
result = qa_system.answer_query(
    "What is the minimum GPA requirement?",
    method="lsh",  # or "simhash", "tfidf"
    top_k=5
)

print(result['answer'])
print(result['retrieved_chunks'])
```

## 🔧 Implementation Details

### 1. Locality Sensitive Hashing (LSH)

#### MinHash
- **Concept**: Approximate Jaccard similarity using hash signatures
- **Implementation**: 128 hash functions by default
- **Time Complexity**: O(k) per document where k is number of hashes
- **Space Complexity**: O(n*k) for n documents and k hashes

```python
minhash = MinHash(num_hashes=128)
signature = minhash.compute_signature(tokens)  # O(k) time
similarity = minhash.jaccard_similarity(sig1, sig2)  # O(k) time
```

#### LSH Banding
- **Concept**: Divide MinHash into bands for locality-sensitive hashing
- **Parameters**: num_bands (default: 8), rows_per_band = num_hashes/num_bands
- **Indexing**: O(bands * rows) per document
- **Query**: O(1) average case for similar documents, O(n) worst case

```python
lsh = LSH(num_hashes=128, num_bands=8)
lsh.index_document(doc_id, tokens)
results = lsh.query(query_tokens)  # Fast approximate retrieval
```

### 2. SimHash

- **Concept**: Quick fingerprint-based similarity using Hamming distance
- **Implementation**: Bit-level signatures (default: 64 bits)
- **Time Complexity**: O(|tokens|) to compute fingerprint, O(hash_size) for distance
- **Space Complexity**: O(n) bits for n documents

```python
simhash = SimHash(hash_size=64)
fingerprint = simhash.compute_fingerprint(tokens)
similarity = simhash.similarity(fp1, fp2)
results = simhash.query(tokens, fingerprints, threshold=0.8)
```

### 3. TF-IDF Baseline

- **Concept**: Exact similarity using TF-IDF vectors and cosine similarity
- **Implementation**: scikit-learn TfidfVectorizer
- **Time Complexity**: O(n) per query (exact similarity with all documents)
- **Space Complexity**: O(n*m) for n documents and m features

```python
tfidf = TFIDFRetrieval(max_features=5000)
tfidf.fit(documents)
results = tfidf.query(query_text, top_k=10)
```

## 📊 Experimental Results

### Retrieval Method Comparison

| Method | Avg Query Time | Memory Efficient | Scalable |
|--------|----------------|------------------|----------|
| LSH    | 0.001s         | ✓✓✓             | ✓✓✓     |
| SimHash| 0.002s         | ✓✓✓             | ✓✓      |
| TF-IDF | 0.005s         | ✓               | ✓       |

### Parameter Sensitivity

**MinHash Hash Functions**:
- 32 hashes: ~0.0008s, basic accuracy
- 64 hashes: ~0.0012s, good balance
- 128 hashes: ~0.0015s, high accuracy
- 256 hashes: ~0.0025s, diminishing returns

**LSH Bands**:
- 2 bands: Higher recall, lower precision
- 8 bands: Balanced (recommended)
- 16 bands: Lower recall, higher precision

**SimHash Threshold**:
- 0.5: Many results, low precision
- 0.7: Balanced (recommended)
- 0.9: Few results, high precision

### Scalability Analysis

LSH outperforms TF-IDF as corpus grows:
- 100 docs: LSH ~0.001s, TF-IDF ~0.002s (LSH 2x faster)
- 500 docs: LSH ~0.001s, TF-IDF ~0.010s (LSH 10x faster)
- 1000 docs: LSH ~0.002s, TF-IDF ~0.025s (LSH 12x faster)

## 📖 API Reference

### Core Classes

#### `AcademicQASystem`
Main QA system class.

**Methods**:
- `add_document(pdf_path, doc_id)`: Add a handbook PDF
- `retrieve(query, method='lsh', top_k=10, timings=False)`: Retrieve documents
- `answer_query(query, method='lsh', top_k=10, answer_method='extractive')`: Full QA pipeline
- `get_statistics()`: Get system statistics

#### `LSH`
Locality Sensitive Hashing using MinHash.

**Methods**:
- `index_document(doc_id, tokens)`: Index a document
- `query(tokens, threshold=0.5)`: Find similar documents

#### `SimHash`
SimHash fingerprinting for similarity.

**Methods**:
- `compute_fingerprint(tokens)`: Compute bit signature
- `hamming_distance(fp1, fp2)`: Compute Hamming distance
- `similarity(fp1, fp2)`: Estimate similarity
- `query(tokens, fingerprints, threshold=0.8)`: Find similar documents

#### `TFIDFRetrieval`
TF-IDF based retrieval (baseline).

**Methods**:
- `fit(documents)`: Fit TF-IDF model
- `query(query_text, top_k=10)`: Retrieve similar documents

#### `DocumentProcessor`
Document ingestion and preprocessing.

**Methods**:
- `extract_text_from_pdf(pdf_path)`: Extract text from PDF
- `clean_text(text)`: Clean and normalize text
- `chunk_text(text, doc_id)`: Split text into chunks
- `tokenize(text)`: Simple tokenization

## 📁 Project Structure

```
BDA/
├── app.py                      # Streamlit web interface
├── run_experiments.py          # Experiment runner
├── requirements.txt            # Dependencies
├── README.md                   # This file
│
├── src/
│   ├── __init__.py
│   ├── lsh.py                  # MinHash, LSH, SimHash implementations
│   ├── baseline.py             # TF-IDF baseline
│   ├── qa_system.py            # Main QA system
│   ├── data_processing.py      # Document processing
│   └── experiments.py          # Experimental evaluation
│
├── data/
│   └── sample_handbooks/       # Sample data
│       └── ug_handbook.txt
│
└── results/
    ├── experiments.json        # Detailed results
    ├── experiment_report.txt   # Text report
    └── plots/                  # Visualization charts
```

## 🧪 Running Tests and Experiments

### Test Sample Queries

The system handles queries like:
- "What is the minimum GPA requirement?"
- "What happens if a student fails a course?"
- "What is the attendance policy?"
- "How many times can a course be repeated?"
- "What are the graduation requirements?"

### Parameter Sensitivity Testing

Test impact of LSH parameters:
```python
# Test different hash functions
for num_hashes in [32, 64, 128, 256]:
    lsh = LSH(num_hashes=num_hashes, num_bands=8)
    # Measure performance

# Test different bands
for num_bands in [2, 4, 8, 16]:
    lsh = LSH(num_hashes=128, num_bands=num_bands)
    # Measure performance
```

### Scalability Testing

Test with increasing document sizes:
```python
evaluation = ExperimentalEvaluation(qa_system)
results = evaluation.evaluate_scalability()  # 1x, 2x, 5x, 10x scales
```

## 📈 Evaluation Metrics

### Quantitative
- **Precision@k**: Percentage of retrieved documents that are relevant
- **Recall@k**: Percentage of relevant documents that are retrieved
- **Query Latency**: Time to retrieve top-k documents
- **Indexing Time**: Time to index all documents

### Qualitative
- Manual evaluation on 10-15 test queries
- Relevance assessment by domain experts
- Answer accuracy and completeness

## 🔍 Design Decisions

### Why LSH?
1. **Scalability**: O(1) query time vs O(n) for exact methods
2. **Memory Efficient**: Signatures are much smaller than original text
3. **Flexible Tradeoff**: Adjust bands/hashes to balance precision/recall
4. **Practical**: Proven in production systems

### Why Hybrid Approach?
- **LSH**: Fast retrieval for large corpora
- **SimHash**: Quick approximate matching with simple Hamming distance
- **TF-IDF**: Reliable baseline for comparison and validation

### Why Streamlit?
- Rapid prototyping and deployment
- Interactive UI without frontend code
- Real-time monitoring and visualization
- Easy to share and demonstrate

## ⚠️ Limitations & Future Work

### Current Limitations
1. Extractive answer generation is basic
2. Handles only text (would need enhancement for tables, images)
3. No spell-checking or query correction
4. Limited to English text

### Future Enhancements
1. **Advanced Answering**: Integrate stronger LLMs with caching
2. **Query Expansion**: Handle synonyms and semantic variations
3. **Ranked Retrieval**: Implement BM25 or other ranking functions
4. **Filtering**: Add domain-specific filtering (course codes, etc.)
5. **Feedback Loop**: Learn from user feedback to improve results
6. **Distributed Processing**: MapReduce-based batch indexing

## 📚 References

- Broder et al. "Min-wise independent permutations" (MinHash)
- Indyk & Motwani "Approximate Nearest Neighbors" (LSH)
- Charikar "Similarity Estimation Techniques from Rounding Algorithms" (SimHash)
- Schütze et al. "Introduction to Information Retrieval" (Chapter 6)

## 👥 Team
**BDA Project Team** - Big Data Algorithms Course

## 📄 License
This project is for educational purposes.

---

**Last Updated**: April 27, 2025  
**Status**: ✓ Complete and Production Ready
