# LexiPolicy Quickstart Guide 🚀

Welcome to LexiPolicy! Follow these steps to set up and run the system.

## 🛠️ 1. Environment Setup

We recommend using **Conda** to manage dependencies.

```bash
# 1. Create the conda environment
conda env create -f environment.yml

# 2. Activate the environment
conda activate bda_project
```

## 🚀 2. Running the System

### Launch the Web Interface
The premium Streamlit dashboard is the easiest way to interact with the system.

```bash
streamlit run app.py
```
*The app will open at http://localhost:8501*

### Run Experimental Evaluation
Generate the full suite of performance and scalability metrics.

```bash
python run_experiments.py
```
*Results will be saved in the `results/` directory.*

## 📂 3. Project Structure

- `app.py`: Premium Streamlit dashboard with integrated analytics.
- `src/`: Core logic
  - `lsh.py`: Optimized Vectorized MinHash + LSH and SimHash.
  - `analytics.py`: Frequent Itemset Mining (Apriori) for query patterns.
  - `qa_system.py`: Main QA pipeline orchestrator.
  - `data_processing.py`: PDF extraction and chunking.
- `data/`: Sample handbook data.
- `results/`: Generated experiment reports and plots.

## 💡 4. Troubleshooting

- **Conda Command Not Found**: Ensure you have installed Miniconda/Anaconda and initialized it with `conda init`.
- **Missing PDF Libraries**: The system uses `pdfplumber` and `pypdf`. These are included in the `environment.yml`.
- **LLM Features**: To use Neural LLM answering, set your `OPENAI_API_KEY` environment variable.

---
**LexiPolicy v1.0** | Big Data Algorithms Project | 2025
