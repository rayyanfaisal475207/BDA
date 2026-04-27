#!/usr/bin/env python3
"""
Run test queries across all available algorithms and generate a detailed report.
"""

import sys
import time
from pathlib import Path
from src.qa_system import AcademicQASystem
from src.data_processing import DocumentProcessor

def generate_report():
    print("=" * 80)
    print("FINRATE: MULTI-ALGORITHM TEST RUN")
    print("=" * 80)

    # Initialize system
    print("\n[1] Initializing QA System with all algorithms...")
    qa_system = AcademicQASystem(
        use_lsh=True,
        use_simhash=True,
        use_tfidf=True,
        use_llm=False  # Keep LLM off for pure speed/retrieval testing
    )

    # Load data
    handbook_dir = Path("data/handbooks")
    pdf_files = list(handbook_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("ERROR: No PDF files found in data/handbooks")
        return

    print(f"[2] Indexing {len(pdf_files)} PDF handbooks...")
    for pdf_file in pdf_files:
        qa_system.add_document(str(pdf_file), pdf_file.stem)
    qa_system.fit_baseline()

    # Test Queries
    queries = [
        "What is the minimum GPA requirement?",
        "What happens if a student fails a course?",
        "What is the attendance policy?",
        "How many times can a course be repeated?",
        "What are the graduation requirements?",
        "How do I appeal a grade?",
        "What is the credit hour system?",
        "When can a student be dismissed?",
        "What is academic probation?",
        "How do I qualify for honors?"
    ]

    methods = [
        ('lsh', 'LSH (MinHash)'),
        ('simhash', 'SimHash'),
        ('tfidf', 'TF-IDF (Exact)'),
        ('hybrid', 'Hybrid (RRF Ensemble)')
    ]

    report_path = Path("results/algorithm_comparison_report.txt")
    report_path.parent.mkdir(exist_ok=True)

    print(f"[3] Running {len(queries)} queries across {len(methods)} algorithms...")

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("FINRATE ALGORITHM COMPARISON REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total Documents: {len(qa_system.documents)} chunks\n")
        f.write(f"Algorithms Tested: {', '.join([m[1] for m in methods])}\n\n")

        for query in queries:
            f.write("-" * 80 + "\n")
            f.write(f"QUERY: {query}\n")
            f.write("-" * 80 + "\n")
            
            for method_id, method_name in methods:
                start_time = time.time()
                # Use retrieve directly to avoid generating extractive answers for the report
                results, _ = qa_system.retrieve(query, method=method_id, top_k=3)
                duration = (time.time() - start_time) * 1000
                
                f.write(f"\nMETHOD: {method_name}\n")
                f.write(f"Latency: {duration:.2f} ms\n")
                f.write(f"Top 3 Results:\n")
                
                if not results:
                    f.write("  (No results found)\n")
                else:
                    for i, (chunk_id, score) in enumerate(results):
                        source = qa_system.doc_metadata[chunk_id]['source']
                        page = qa_system.doc_metadata[chunk_id]['page']
                        text_snippet = qa_system.documents[chunk_id][:150].replace('\n', ' ')
                        f.write(f"  {i+1}. [{source} p.{page}] (Score: {score:.4f}): {text_snippet}...\n")
            f.write("\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("SUMMARY OF AVERAGE LATENCY\n")
        f.write("-" * 80 + "\n")
        
        stats = qa_system.get_statistics()
        perf = stats['performance_summary']
        for method_id, method_name in methods:
            if method_id in perf:
                avg = perf[method_id]['avg_time'] * 1000
                f.write(f"{method_name:25} : {avg:.2f} ms\n")

        f.write("=" * 80 + "\n")

    print(f"\n[4] Report generated successfully at: {report_path}")

if __name__ == "__main__":
    generate_report()
