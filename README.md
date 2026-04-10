# Vexoo Labs – AI Engineer Assignment

## Overview
This submission contains two parts:

1. **Part 1: Document Ingestion + Knowledge Pyramid**
   - Reads a document
   - Splits it using a sliding window strategy
   - Builds a 4-layer knowledge pyramid:
     - Raw Text
     - Chunk Summary
     - Category Label
     - Distilled Knowledge (keywords)
   - Retrieves the most relevant answer using TF-IDF + cosine similarity

2. **Part 2: GSM8K Fine-Tuning Pipeline**
   - Loads GSM8K from Hugging Face
   - Uses 3000 training samples and 1000 evaluation samples
   - Includes tokenization
   - Includes LoRA-based fine-tuning structure
   - Performs simple numeric-answer evaluation

## Setup
```bash
pip install -r requirements.txt# Vexoo Labs – AI Engineer Assignment

## Overview
This submission contains two parts:

1. **Part 1: Document Ingestion + Knowledge Pyramid**
   - Reads a document
   - Splits it using a sliding window strategy
   - Builds a 4-layer knowledge pyramid:
     - Raw Text
     - Chunk Summary
     - Category Label
     - Distilled Knowledge (keywords)
   - Retrieves the most relevant answer using TF-IDF + cosine similarity

2. **Part 2: GSM8K Fine-Tuning Pipeline**
   - Loads GSM8K from Hugging Face
   - Uses 3000 training samples and 1000 evaluation samples
   - Includes tokenization
   - Includes LoRA-based fine-tuning structure
   - Performs simple numeric-answer evaluation

## Setup
```bash
pip install -r requirements.txt