# Vexoo Labs Assignment Summary

## Part 1: Ingestion + Knowledge Pyramid
I implemented a lightweight document ingestion pipeline using a sliding window strategy based on character limits to simulate a 2-page chunking approach. Each chunk is converted into a 4-layer knowledge pyramid: raw text, placeholder summary, rule-based category label, and distilled knowledge (keywords). Retrieval is performed across all pyramid levels using TF-IDF and cosine similarity.

## Part 2: GSM8K Training Setup
I implemented a GSM8K fine-tuning pipeline using Hugging Face datasets and transformers. The pipeline loads 3000 training samples and 1000 evaluation samples, formats question-answer pairs, tokenizes them, applies LoRA-based fine-tuning, and runs a simple evaluation based on final numeric answer extraction. A lightweight model is used for demonstration, while the pipeline is compatible with larger causal language models.

## Key Design Decisions
I prioritized modularity, simplicity, and reproducibility. Placeholder summarization and rule-based labels were intentionally used as lightweight, explainable baselines. The training pipeline is designed to be resource-aware and easily upgradeable to larger models or stronger evaluation strategies.