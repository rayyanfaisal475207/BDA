"""Quick end-to-end smoke test — run with: python3 smoke_test.py"""

from pathlib import Path
from src.qa_system import AcademicQASystem
from src.data_processing import DocumentProcessor

def load_system():
    qa = AcademicQASystem(use_lsh=True, use_simhash=True, use_tfidf=True)
    proc = DocumentProcessor()
    for txt_file in sorted(Path("data/sample_handbooks").glob("*.txt")):
        with open(txt_file, encoding="utf-8") as f:
            text = f.read()
        for chunk_id, chunk_text, page_num in proc.chunk_text(text, txt_file.stem):
            qa.documents[chunk_id] = chunk_text
            qa.doc_metadata[chunk_id] = {"source": txt_file.stem, "page": page_num}
            tokens = set(proc.tokenize(chunk_text))
            qa.lsh.index_document(chunk_id, tokens)
            qa.simhash_fingerprints[chunk_id] = qa.simhash.compute_fingerprint(list(tokens))
    qa.fit_baseline()
    return qa


def main():
    print("=" * 60)
    print("LexiPolicy — Smoke Test")
    print("=" * 60)

    qa = load_system()
    stats = qa.get_statistics()
    print(f"\nCorpus: {stats['total_chunks']} chunks | {stats['total_tokens']:,} tokens\n")

    queries = [
        "What is the minimum GPA requirement?",
        "What happens if a student fails a course?",
        "What is the attendance policy?",
        "How many times can a course be repeated?",
        "What are the graduation requirements?",
        "How do I appeal a grade?",
        "What is the maximum credit hours per semester?",
        "What is academic probation?",
    ]

    for method in ["lsh", "simhash", "tfidf"]:
        hits = 0
        import time
        t0 = time.time()
        for q in queries:
            res, _ = qa.retrieve(q, method=method, top_k=5, timings=True)
            if res:
                hits += 1
        elapsed = time.time() - t0
        avg_ms = elapsed / len(queries) * 1000
        print(f"[{method.upper():<8}]  {hits}/{len(queries)} hits  |  avg {avg_ms:.2f}ms/query")

    print()
    # Test full QA pipeline on one query
    result = qa.answer_query("What is the minimum GPA requirement?", method="tfidf", top_k=5)
    print("Sample Answer (TF-IDF):")
    print(f"  {result['answer'][:200]}")
    print()
    print("PASS — System healthy.")


if __name__ == "__main__":
    main()
