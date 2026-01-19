# Experiment Summary (public-safe)

This project explored abstractive summarization with a Transformer model (**T5-small**) alongside extractive baselines.

## Dataset (not included)
CNN/DailyMail (via Hugging Face Datasets). Dataset files are not uploaded to this public repository.

## Models / Baselines
- Lead-2 baseline (extractive heuristic)
- TF-IDF + Logistic Regression (extractive sentence relevance)
- T5-small fine-tuning (abstractive summarization)

## Fine-tuning setup (high level)
- Framework: Hugging Face Transformers
- Objective: sequence-to-sequence summarization
- Decoding: beam search
- Evaluation: ROUGE-2

## Results (high level)
The fine-tuned T5-small model improved ROUGE-2 over the simpler extractive baselines in this project setup.

## Notes
No trained checkpoints are uploaded. This summary is provided as a portfolio/evidence artifact.
