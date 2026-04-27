"""
Experimental analysis and evaluation of the QA system.
Compares exact vs approximate retrieval methods.
"""

import time
import json
from pathlib import Path
from typing import List, Dict, Tuple
from src.qa_system import AcademicQASystem
from src.data_processing import DocumentProcessor
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class ExperimentalEvaluation:
    """Run experiments to evaluate system performance."""

    def __init__(self, qa_system: AcademicQASystem):
        """
        Args:
            qa_system: The QA system to evaluate
        """
        self.qa_system = qa_system
        self.results = {}

    @staticmethod
    def load_sample_data(sample_dir: Path = Path("data/sample_handbooks")) -> Dict[str, str]:
        """Load sample handbook data."""
        documents = {}
        processor = DocumentProcessor()

        for txt_file in sample_dir.glob("*.txt"):
            doc_id = txt_file.stem
            with open(txt_file, 'r') as f:
                text = f.read()

            # Split into chunks
            words = text.split()
            chunk_num = 0

            for i in range(0, len(words), 300):
                chunk_words = words[i:i+300]
                if len(chunk_words) > 10:
                    chunk_text = " ".join(chunk_words)
                    chunk_id = f"{doc_id}_chunk_{chunk_num}"
                    documents[chunk_id] = chunk_text
                    chunk_num += 1

        return documents

    def add_documents_to_system(self, documents: Dict[str, str]):
        """Add documents to the QA system."""
        processor = DocumentProcessor()

        for chunk_id, chunk_text in documents.items():
            self.qa_system.documents[chunk_id] = chunk_text

            # Extract source and page from chunk_id
            parts = chunk_id.rsplit('_', 1)
            source = parts[0] if len(parts) > 1 else 'Unknown'
            page = int(parts[1]) if len(parts) > 1 else 0

            self.qa_system.doc_metadata[chunk_id] = {
                'source': source,
                'page': page
            }

            # Index with LSH
            if self.qa_system.lsh:
                tokens = set(processor.tokenize(chunk_text))
                self.qa_system.lsh.index_document(chunk_id, tokens)

            # SimHash fingerprints
            if self.qa_system.simhash:
                tokens_list = processor.tokenize(chunk_text)
                self.qa_system.simhash_fingerprints[chunk_id] = \
                    self.qa_system.simhash.compute_fingerprint(tokens_list)

        # Fit TF-IDF baseline
        if self.qa_system.tfidf:
            self.qa_system.fit_baseline()

    def test_queries(self) -> List[str]:
        """Sample queries for testing."""
        return [
            "What is the minimum GPA requirement?",
            "What happens if a student fails a course?",
            "What is the attendance policy?",
            "How many times can a course be repeated?",
            "What are the graduation requirements?",
            "How do I appeal a grade?",
            "What is the credit hour system?",
            "When can a student be dismissed?",
            "What is academic probation?",
            "How do I qualify for honors?",
            "What is the registration process?",
            "How many credit hours should I take?",
            "What is the grading scale?",
            "How do I get reinstated after dismissal?",
            "What is the deadline for graduation application?"
        ]

    def evaluate_retrieval_methods(self, top_k: int = 5) -> Dict:
        """
        Compare retrieval methods: LSH, SimHash, and TF-IDF.

        Returns:
            Dictionary with comparison results
        """
        queries = self.test_queries()
        methods = [('lsh', 'LSH (MinHash)'), ('simhash', 'SimHash'),
                  ('tfidf', 'TF-IDF (Baseline)')]

        results = {
            'queries': len(queries),
            'top_k': top_k,
            'methods': {}
        }

        for method_name, method_label in methods:
            method_results = {
                'label': method_label,
                'queries': [],
                'avg_time': 0,
                'total_time': 0,
                'retrieval_times': []
            }

            for query in queries:
                retrieved, timing = self.qa_system.retrieve(
                    query, method=method_name, top_k=top_k, timings=True
                )

                retrieval_time = timing.get('retrieval_time', 0) if timing else 0
                method_results['retrieval_times'].append(retrieval_time)

                method_results['queries'].append({
                    'query': query,
                    'num_results': len(retrieved),
                    'time': retrieval_time,
                    'top_result': retrieved[0] if retrieved else None
                })

                method_results['total_time'] += retrieval_time

            method_results['avg_time'] = method_results['total_time'] / len(queries)
            results['methods'][method_name] = method_results

        return results

    def analyze_parameter_sensitivity(self) -> Dict:
        """
        Analyze sensitivity to parameters.
        Test impact of hash functions and bands.
        """
        query = "What is the minimum GPA requirement?"
        test_query_tokens = set(self.qa_system.processor.tokenize(query))

        results = {
            'hash_functions': {},
            'bands': {},
            'simhash_threshold': {}
        }

        # Test different numbers of hash functions
        for num_hashes in [32, 64, 128, 256]:
            from src.lsh import LSH
            lsh = LSH(num_hashes=num_hashes, num_bands=num_hashes)

            # Re-index documents
            for chunk_id, chunk_text in self.qa_system.documents.items():
                tokens = set(self.qa_system.processor.tokenize(chunk_text))
                lsh.index_document(chunk_id, tokens)

            start = time.time()
            retrieved = lsh.query(test_query_tokens, threshold=0.0)
            elapsed = time.time() - start

            results['hash_functions'][num_hashes] = {
                'time': elapsed,
                'results': len(retrieved)
            }

        # Test different numbers of bands (sensitivity: fewer rows/band = more sensitive)
        for num_bands in [8, 16, 32, 64, 128]:
            from src.lsh import LSH
            lsh = LSH(num_hashes=128, num_bands=num_bands)

            for chunk_id, chunk_text in self.qa_system.documents.items():
                tokens = set(self.qa_system.processor.tokenize(chunk_text))
                lsh.index_document(chunk_id, tokens)

            start = time.time()
            retrieved = lsh.query(test_query_tokens, threshold=0.0)
            elapsed = time.time() - start

            results['bands'][num_bands] = {
                'time': elapsed,
                'results': len(retrieved)
            }

        # Test SimHash thresholds
        simhash_tokens = self.qa_system.processor.tokenize(query)
        for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
            retrieved = self.qa_system.simhash.query(
                simhash_tokens,
                self.qa_system.simhash_fingerprints,
                threshold=threshold
            )
            results['simhash_threshold'][threshold] = len(retrieved)

        return results

    def evaluate_scalability(self) -> Dict:
        """
        Test scalability by duplicating documents.

        Returns:
            Scalability test results
        """
        query = "What is the minimum GPA requirement?"
        original_doc_count = len(self.qa_system.documents)

        results = {
            'original_count': original_doc_count,
            'scaling_tests': []
        }

        # Test with increasing document counts
        for scale_factor in [1, 2, 5, 10]:
            from src.lsh import LSH
            from src.baseline import TFIDFRetrieval

            # Create scaled version
            scaled_docs = {}
            doc_counter = 0

            for orig_id, text in list(self.qa_system.documents.items())[:
                                                                        original_doc_count]:
                for i in range(scale_factor):
                    new_id = f"{orig_id}_scale{i}"
                    scaled_docs[new_id] = text
                    doc_counter += 1

            # Test LSH performance
            if self.qa_system.lsh:
                lsh = LSH(num_hashes=128, num_bands=128)
                start = time.time()

                for doc_id, text in scaled_docs.items():
                    tokens = set(self.qa_system.processor.tokenize(text))
                    lsh.index_document(doc_id, tokens)

                index_time = time.time() - start

                query_tokens = set(self.qa_system.processor.tokenize(query))
                start = time.time()
                retrieved = lsh.query(query_tokens, threshold=0.0)
                query_time = time.time() - start
            else:
                index_time = query_time = 0

            # Test TF-IDF
            if self.qa_system.tfidf:
                tfidf = TFIDFRetrieval()
                start = time.time()
                tfidf.fit(scaled_docs)
                tfidf_index_time = time.time() - start

                start = time.time()
                tfidf_results = tfidf.query(query, top_k=5)
                tfidf_query_time = time.time() - start
            else:
                tfidf_index_time = tfidf_query_time = 0

            results['scaling_tests'].append({
                'scale_factor': scale_factor,
                'doc_count': doc_counter,
                'lsh_index_time': index_time,
                'lsh_query_time': query_time,
                'tfidf_index_time': tfidf_index_time,
                'tfidf_query_time': tfidf_query_time
            })

        return results

    def run_all_experiments(self) -> Dict:
        """Run all experiments and return results."""
        experiments = {
            'retrieval_comparison': self.evaluate_retrieval_methods(),
            'parameter_sensitivity': self.analyze_parameter_sensitivity(),
            'scalability': self.evaluate_scalability()
        }

        self.results = experiments
        return experiments

    def save_results(self, output_file: str = "results/experiments.json"):
        """Save experiment results to file."""
        Path(output_file).parent.mkdir(exist_ok=True)

        with open(output_file, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json.dump(self.results, f, indent=2, default=str)

        print(f"Results saved to {output_file}")

    def plot_results(self, output_dir: str = "results/plots"):
        """Generate visualization plots."""
        Path(output_dir).mkdir(exist_ok=True)

        if not self.results:
            print("No results to plot. Run experiments first.")
            return

        # Plot 1: Retrieval time comparison
        methods_data = self.results.get('retrieval_comparison', {}).get('methods', {})
        if methods_data:
            fig, ax = plt.subplots(figsize=(10, 6))
            method_names = []
            avg_times = []

            for method_name, method_data in methods_data.items():
                method_names.append(method_data['label'])
                avg_times.append(method_data['avg_time'])

            ax.bar(method_names, avg_times, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            ax.set_ylabel('Average Query Time (seconds)')
            ax.set_title('Retrieval Time Comparison')
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/retrieval_time_comparison.png", dpi=300)
            plt.close()

        # Plot 2: Scalability test
        scalability = self.results.get('scalability', {}).get('scaling_tests', [])
        if scalability:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            doc_counts = [t['doc_count'] for t in scalability]
            lsh_times = [t['lsh_query_time'] for t in scalability]
            tfidf_times = [t['tfidf_query_time'] for t in scalability]

            ax1.plot(doc_counts, lsh_times, marker='o', label='LSH', linewidth=2)
            ax1.plot(doc_counts, tfidf_times, marker='s', label='TF-IDF', linewidth=2)
            ax1.set_xlabel('Number of Documents')
            ax1.set_ylabel('Query Time (seconds)')
            ax1.set_title('Query Time Scalability')
            ax1.legend()
            ax1.grid(alpha=0.3)

            lsh_index_times = [t['lsh_index_time'] for t in scalability]
            tfidf_index_times = [t['tfidf_index_time'] for t in scalability]

            ax2.plot(doc_counts, lsh_index_times, marker='o', label='LSH', linewidth=2)
            ax2.plot(doc_counts, tfidf_index_times, marker='s', label='TF-IDF', linewidth=2)
            ax2.set_xlabel('Number of Documents')
            ax2.set_ylabel('Indexing Time (seconds)')
            ax2.set_title('Indexing Time Scalability')
            ax2.legend()
            ax2.grid(alpha=0.3)

            plt.tight_layout()
            plt.savefig(f"{output_dir}/scalability_comparison.png", dpi=300)
            plt.close()

        print(f"Plots saved to {output_dir}")

    def generate_report(self, output_file: str = "results/experiment_report.txt"):
        """Generate a text report of the experiments."""
        Path(output_file).parent.mkdir(exist_ok=True)

        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("EXPERIMENTAL EVALUATION REPORT\n")
            f.write("Scalable Academic Policy QA System\n")
            f.write("=" * 80 + "\n\n")

            # Retrieval comparison
            f.write("1. RETRIEVAL METHOD COMPARISON\n")
            f.write("-" * 80 + "\n")
            methods_data = self.results.get('retrieval_comparison', {})
            if methods_data:
                f.write(f"Total Queries Tested: {methods_data.get('queries', 0)}\n")
                f.write(f"Top-K Results: {methods_data.get('top_k', 0)}\n\n")

                for method_name, data in methods_data.get('methods', {}).items():
                    f.write(f"\n{data['label']}:\n")
                    f.write(f"  Average Query Time: {data['avg_time']:.6f} seconds\n")
                    f.write(f"  Total Time: {data['total_time']:.6f} seconds\n")
                    f.write(f"  Min Time: {min(data['retrieval_times']):.6f} seconds\n")
                    f.write(f"  Max Time: {max(data['retrieval_times']):.6f} seconds\n")

            # Parameter sensitivity
            f.write("\n\n2. PARAMETER SENSITIVITY ANALYSIS\n")
            f.write("-" * 80 + "\n")
            param_data = self.results.get('parameter_sensitivity', {})

            if 'hash_functions' in param_data:
                f.write("\nHash Functions Impact:\n")
                for num_hashes, data in sorted(param_data['hash_functions'].items()):
                    f.write(f"  {num_hashes} hashes: {data['time']:.6f}s, {data['results']} results\n")

            if 'bands' in param_data:
                f.write("\nBands Impact (LSH Bands):\n")
                for num_bands, data in sorted(param_data['bands'].items()):
                    f.write(f"  {num_bands} bands: {data['time']:.6f}s, {data['results']} results\n")

            if 'simhash_threshold' in param_data:
                f.write("\nSimHash Threshold Impact:\n")
                for threshold, count in sorted(param_data['simhash_threshold'].items()):
                    f.write(f"  Threshold {threshold}: {count} results\n")

            # Scalability
            f.write("\n\n3. SCALABILITY ANALYSIS\n")
            f.write("-" * 80 + "\n")
            scalability = self.results.get('scalability', {})
            if scalability:
                f.write(f"Original Document Count: {scalability.get('original_count', 0)}\n\n")

                for test in scalability.get('scaling_tests', []):
                    f.write(f"Scale Factor: {test['scale_factor']}x ({test['doc_count']} documents)\n")
                    f.write(f"  LSH Index Time: {test['lsh_index_time']:.6f}s\n")
                    f.write(f"  LSH Query Time: {test['lsh_query_time']:.6f}s\n")
                    f.write(f"  TF-IDF Index Time: {test['tfidf_index_time']:.6f}s\n")
                    f.write(f"  TF-IDF Query Time: {test['tfidf_query_time']:.6f}s\n\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")

        print(f"Report saved to {output_file}")
