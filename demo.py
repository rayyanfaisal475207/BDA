#!/usr/bin/env python3
"""
Quick demo of the Academic QA System.
Run this for a fast test of all retrieval methods.
"""

from pathlib import Path
from src.qa_system import AcademicQASystem
from src.data_processing import DocumentProcessor

def load_sample_data():
    """Load sample handbook data."""
    qa_system = AcademicQASystem(
        use_lsh=True,
        use_simhash=True,
        use_tfidf=True,
        use_llm=False
    )

    processor = DocumentProcessor()
    sample_dir = Path("data/sample_handbooks")

    if not sample_dir.exists():
        print("❌ Sample data directory not found. Creating...")
        return None

    # Load UG handbook
    ug_file = sample_dir / "ug_handbook.txt"
    if ug_file.exists():
        with open(ug_file, 'r') as f:
            text = f.read()

        words = text.split()
        chunk_num = 0

        for i in range(0, len(words), 300):
            chunk_words = words[i:i+300]
            if len(chunk_words) > 10:
                chunk_text = " ".join(chunk_words)
                chunk_id = f"ug_handbook_chunk_{chunk_num}"

                qa_system.documents[chunk_id] = chunk_text
                qa_system.doc_metadata[chunk_id] = {
                    'source': 'UG Handbook',
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

        # Fit TF-IDF
        qa_system.fit_baseline()

        print(f"✓ Loaded {chunk_num} chunks from UG Handbook")
        return qa_system

    return None


def demo():
    """Run interactive demo."""
    print("=" * 80)
    print("ACADEMIC POLICY QA SYSTEM - DEMO")
    print("=" * 80)
    print()

    # Load system
    print("[1/3] Loading system...")
    qa_system = load_sample_data()

    if not qa_system:
        print("❌ Failed to load sample data")
        return

    print(f"✓ System ready with {len(qa_system.documents)} chunks")
    print()

    # Sample queries
    queries = [
        "What is the minimum GPA requirement?",
        "What happens if a student fails a course?",
        "What is the attendance policy?",
        "How many times can a course be repeated?",
        "What are the graduation requirements?"
    ]

    print("[2/3] Testing retrieval methods on 5 sample queries...\n")

    methods = ['lsh', 'simhash', 'tfidf']

    for query_num, query in enumerate(queries, 1):
        print(f"Query {query_num}: {query}")
        print("-" * 80)

        for method in methods:
            retrieved, timing = qa_system.retrieve(
                query, method=method, top_k=3, timings=True
            )

            time_ms = timing.get('retrieval_time', 0) * 1000 if timing else 0

            print(f"\n  {method.upper()} ({time_ms:.2f}ms):")
            if retrieved:
                for i, (chunk_id, similarity) in enumerate(retrieved, 1):
                    print(f"    {i}. {chunk_id}")
                    print(f"       Similarity: {similarity:.4f}")
                    text_preview = qa_system.documents[chunk_id][:100]
                    print(f"       Preview: {text_preview}...")
            else:
                print("    No results found")

        print()

    # Performance summary
    print("[3/3] Performance Summary")
    print("-" * 80)

    print("\nRetrieval Time Comparison (across all queries):")
    for method in methods:
        times = []
        for query in queries:
            _, timing = qa_system.retrieve(
                query, method=method, top_k=3, timings=True
            )
            times.append(timing.get('retrieval_time', 0) if timing else 0)

        avg_time = sum(times) / len(times)
        print(f"  {method.upper():10s}: {avg_time*1000:6.2f}ms (avg)")

    print("\nSystem Statistics:")
    stats = qa_system.get_statistics()
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Total tokens: {stats['total_tokens']}")
    print(f"  Methods enabled: {', '.join(k for k, v in stats['methods_enabled'].items() if v)}")

    print("\n" + "=" * 80)
    print("✓ DEMO COMPLETE")
    print("=" * 80)
    print("\nTo use the web interface:")
    print("  streamlit run app.py")
    print("\nTo run full experiments:")
    print("  python run_experiments.py")
    print()


if __name__ == "__main__":
    demo()
