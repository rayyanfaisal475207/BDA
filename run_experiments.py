#!/usr/bin/env python3
"""
Run experiments on FULL corpus and generate evaluation report with plots.
"""

import sys
from pathlib import Path
from src.qa_system import AcademicQASystem
from src.experiments import ExperimentalEvaluation

def main():
    """Main experiment runner using full corpus."""
    print("=" * 80)
    print("FINRATE: FULL CORPUS EXPERIMENTAL EVALUATION")
    print("=" * 80)

    # Initialize system
    print("\n[1/4] Initializing QA System...")
    qa_system = AcademicQASystem(
        use_lsh=True,
        use_simhash=True,
        use_tfidf=True,
        use_llm=False
    )

    # Load FULL data
    print("[2/4] Loading ALL handbook data...")
    handbook_dir = Path("data/handbooks")
    pdf_files = list(handbook_dir.glob("*.pdf"))
    for pdf_file in pdf_files:
        qa_system.add_document(str(pdf_file), pdf_file.stem)
    qa_system.fit_baseline()
    
    evaluation = ExperimentalEvaluation(qa_system)
    print(f"    Total chunks in system: {len(qa_system.documents)}")

    # Run experiments
    print("[3/4] Running experiments...")
    results = evaluation.run_all_experiments()
    print("    ✓ Experiments completed")

    # Generate output
    print("[4/4] Generating reports and visualizations...")
    evaluation.save_results("results/experiments.json")
    evaluation.generate_report("results/experiment_report.txt")
    evaluation.plot_results("results/plots")
    print("    ✓ Reports saved")

if __name__ == "__main__":
    main()
