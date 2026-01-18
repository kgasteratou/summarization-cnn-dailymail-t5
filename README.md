# Summarization (CNN/DailyMail) – Baselines + T5 Fine-Tuning

This repository contains a compact, reproducible pipeline for English news summarization experiments on the CNN/DailyMail dataset.

## What is included
- **Lead-2 baseline**
- **Classical ML baseline** (TF-IDF + Logistic Regression)
- **Transformer fine-tuning**: **T5-small** for abstractive summarization
- **Evaluation** using **ROUGE-2**

## What is NOT included (for safety/licensing)
- The original dataset files are not uploaded.
- Model checkpoints are not uploaded.
- No API keys or private credentials are used.

## Method overview
### Baseline 1: Lead-2
A strong extractive heuristic baseline that selects the first two sentences of the article.

### Baseline 2: TF-IDF + Logistic Regression
A simple supervised approach using TF-IDF features to predict sentence relevance for extractive summaries.

### Model: T5-small fine-tuning
- Framework: Hugging Face Transformers + Datasets
- Task: Abstractive summarization
- Decoding: beam search
- Metric: ROUGE-2

## How to run (Jupyter/Python)
- Minimal demos live in `notebooks/`.
- Core helper functions live in `src/`.

## Dependencies
Python 3.10+ and common NLP libraries (Transformers, Datasets, PyTorch, ROUGE evaluation).

## Results (summary)
The fine-tuned T5-small model improves ROUGE-2 over the simpler baselines, as expected for an abstractive setup. Outputs and evaluation are produced by the notebook.

## Author note
This repository is intended as a **portfolio artifact** demonstrating:
- dataset handling and preprocessing
- model fine-tuning workflow
- evaluation and reporting practices in NLP

