Names CMS
M Faizan Anwar 455259
Muhammad Haadhee Sheeraz Mian 478359
Rayyan Faisal



Scalable Academic Policy QA System

Design and implement a scalable Question-Answering system over the UG and PG Handbooks
using Big Data techniques covered in the course. The system should efficiently retrieve relevant
information from large text data and generate accurate answers to student queries.
(Handbooks available at: https://seecs.nust.edu.pk/downloads/student-handbooks/)
Following are the focus areas of the Project:
 Apply Locality Sensitive Hashing using MinHash and SimHash
 Understand approximate vs exact similarity tradeoffs
 Design a retrieval pipeline for large-scale text data
 Integrate modern tools (e.g., LLM APIs) into a principled system
 Evaluate systems based on accuracy, efficiency, and scalability
System Overview
Your system should implement the following pipeline:
1. Data Ingestion
 Input: UG Handbook (PDF or text)
 Convert to clean text
 Split into meaningful chunks (e.g., 200–500 words)
2. Similarity & Indexing (Core Component)
You must implement hybrid LSH-based method on the following:
A: MinHash + LSH
 Represent documents as sets
 Compute MinHash signatures
 Use LSH to find similar chunks efficiently
B: SimHash
 Generate fingerprint for each chunk
 Use Hamming distance for similarity detection
3. Baseline Method (Required)
Implement a non-approximate method:
 TF-IDF + cosine similarity
4. Query Processing
 Input: user question
 Retrieve top-k relevant chunks using:
o LSH-based method
o Baseline method
5. Answer Generation
You may use:
 Extractive methods (from retrieved text), OR
 LLM APIs (e.g., OpenAI, open-source models)
Constraint:
 Answers must be based on retrieved content
 Must display supporting evidence
6. Output Interface
 CLI or simple web interface (e.g., Streamlit)
 Show:
o Answer
o Top-k retrieved chunks
o Source references (page/section)
Select one of the following to extend the functionality of your
system (your competitive edge over others)
Course Topic Proposed Extensions (as an example)
Frequent Itemset Mining Identify common query patterns
Recommendation Systems Rank retrieved chunks (top-k relevance)
PageRank Rank important sections of handbook
MapReduce / SON Simulate distributed indexing or chunk processing
Big Data Principles Efficiency, scalability, approximation
Required Experiments & Analysis
You MUST include the following comparisons in your project report (and in your presentation as
well):
1. Exact vs Approximate Retrieval
Compare:
 TF-IDF (exact)
 LSH (approximate)
Evaluate:
 Accuracy (relevance of retrieved chunks)
 Time taken
 Memory usage
2. Parameter Sensitivity
Analyze impact of:
 Number of hash functions (MinHash)
 Number of bands (LSH)
 Hamming threshold (SimHash)
3. Scalability Test
 Simulate larger dataset (duplicate or extend corpus)
 Show how performance changes
Evaluation Metrics
Include:
Quantitative:
 Precision@k or Recall@k
 Query latency
Qualitative:
 Test on 10–15 queries
 Manually evaluate answer correctness
Sample Queries
Your system should handle queries such as:
 “What is the minimum GPA requirement?”
 “What happens if a student fails a course?”
 “What is the attendance policy?”
 “How many times can a course be repeated?”
Restrictions
The following are NOT allowed:
 Direct use of tools that bypass retrieval (e.g., uploading PDF to chatbot)
 Systems without LSH implementation
 No comparison with baseline method
Deliverables
1. Code
 Clean, well-documented
 Reproducible
2. Report (6–8 pages)
Must include:
 System design
 Algorithm explanation
 Experimental results
 Tradeoff analysis
3. Demo (5–7 minutes)
 Live system
 Example queries
 Explanation of results
 Also have the video recording of the demo
Grading Criteria
Component Weight
Retrieval Implementation via LSH 30%
Experimental Analysis 20%
System Design 20%
Demo 20%
Presentation and Report 10%
Please be mindful that this project is NOT about building a chatbot. It is about designing an
efficient, scalable retrieval system using Big Data techniques, where the chatbot is only the final
interface.
Submission Deadline: 27th April, 2025