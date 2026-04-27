#!/usr/bin/env python3
"""
Run experiments and generate evaluation report.
"""

import sys
from pathlib import Path
from src.qa_system import AcademicQASystem
from src.experiments import ExperimentalEvaluation

def main():
    """Main experiment runner."""
    print("=" * 80)
    print("ACADEMIC POLICY QA SYSTEM - EXPERIMENTAL EVALUATION")
    print("=" * 80)

    # Initialize system
    print("\n[1/4] Initializing QA System...")
    qa_system = AcademicQASystem(
        use_lsh=True,
        use_simhash=True,
        use_tfidf=True,
        use_llm=False
    )

    # Load sample data
    print("[2/4] Loading sample handbook data...")
    evaluation = ExperimentalEvaluation(qa_system)
    documents = evaluation.load_sample_data()
    evaluation.add_documents_to_system(documents)
    print(f"    Loaded {len(documents)} document chunks")

    # Run experiments
    print("[3/4] Running experiments...")
    print("    - Retrieval method comparison...")
    print("    - Parameter sensitivity analysis...")
    print("    - Scalability testing...")
    results = evaluation.run_all_experiments()
    print("    ✓ Experiments completed")

    # Generate output
    print("[4/4] Generating reports and visualizations...")
    evaluation.save_results("results/experiments.json")
    evaluation.generate_report("results/experiment_report.txt")
    evaluation.plot_results("results/plots")
    print("    ✓ Reports saved")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    methods_data = results.get('retrieval_comparison', {}).get('methods', {})
    if methods_data:
        print("\nRetrieval Time Comparison (Average):")
        for method_name, data in methods_data.items():
            print(f"  {data['label']}: {data['avg_time']*1000:.2f} ms")

    print("\n✓ Full results saved to results/ directory")
    print("  - experiments.json (detailed data)")
    print("  - experiment_report.txt (text report)")
    print("  - plots/ (visualization charts)")

if __name__ == "__main__":
    main()
