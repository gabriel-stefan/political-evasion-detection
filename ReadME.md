# QEvasion Dataset: Model Performance Results

## Overview
Three classification tasks are evaluated:
1. **Direct Clarity Classification** (3 classes: Clear Reply, Ambivalent, Clear Non-Reply)
2. **Evasion-based Clarity Classification** (9→3 hierarchical mapping)
3. **Evasion Classification** (9 classes with multi-annotator evaluation)

## Results Table

| Model | Direct Clarity F1 | Evasion→Clarity F1 | Evasion F1 (9-class) |
|-------|------------------:|-------------------:|---------------------:|
| **Traditional ML (TF-IDF only)** |
| Logistic Regression | 0.4997 | 0.5078 | 0.3010 |
| XGBoost | 0.4943 | 0.4691 | 0.2676 |
| **Traditional ML + Features** |
| Logistic Regression + Features | 0.5916 | 0.5717 | 0.3321 |
| XGBoost + Features | 0.6360 | 0.6236 | 0.3852 |
| **Transformer Models (Fine-tuned)** |
| ModernBERT-base | 0.5693 | 0.5035 | 0.3041  |
| ModernBERT-large | 0.6494 | 0.6064 | 0.3636 |
| Llama 3.1-8B (QLoRA) | 0.6277 | — | — |
| **Large Language Models (Prompting)** |
| GPT-5 (Few-Shot CoT) | **0.7171** | **0.7109** | **0.4564** |

**Features include question length, answer length, a gpt3.5 summary of the question and answer pair and a gpt3.5 predcition, these are only available in the train dataset.



