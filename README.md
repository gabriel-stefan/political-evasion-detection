# Clarity: Political Evasion Detection

This repository contains my submission for **SemEval 2026 Task 6: Clarity**.
- [Task Website](https://clarity-semeval2026.github.io/)
- [Reference Paper](reference_paper.pdf)


## Model Architecture
The best-performing model achieves a **Macro F1 Score of 0.70**.

### Hierarchical Multi-Head RoBERTa
I designed a custom architecture to address two key challenges in political interviews: **input length** and **task correlation**.

#### 1. Hierarchical Processing (Chunking & Pooling)
Political answers often exceed the standard 512-token limit of BERT-based models. Instead of truncating valuable context, I implemented a Hierarchical Max-Pooling strategy:
- **Chunking**: The input (Question + Answer) is split into overlapping chunks of 512 tokens (stride: 256).
- **Shared Encoder**: Each chunk is processed independently by a shared `roberta-large` backbone.
- **Max Pooling**: I aggregate the [CLS] embeddings from all chunks using a Max-Pooling operation.

Max pooling captures the strongest activation of evasion features across the entire text, unlike average pooling which might dilute the signal.

#### 2. Multi-Head Joint Learning
Evasion detection (9 classes) and Clarity detection (3 classes) are correlated tasks.
- I added two classification heads on top of the pooled representation.
- The model is trained to minimize a joint loss: $L_{total} = L_{clarity} + L_{evasion}$.
- **Result**: While performance on the primary Clarity task (Task 1) remained consistent, the multi-head approach drove a significant improvement on the Evasion task (Task 2), boosting the F1 score from **0.45 to 0.49** (an ~8.9% increase).

#### 3. K-Fold Evaluation
To ensure our results are not artifacts of a lucky split, I employ 7-Fold Stratified Cross-Validation.
- The final prediction is an ensemble (average probability) of the 7 fold models.


## Experimental Journey
I conducted extensive experiments to reach this solution. Below is a summary of the research path.

### Model Exploration
- **ModernBERT**: I tested `ModernBERT-base` and `ModernBERT-large` for their efficiency with long contexts (8k tokens). They are behind RoBERTa in detecting subtle evasion.
- **DeBERTa V3**: I experimented with `deberta-v3-large` and attention-based pooling layers.
- **Llama 3.1**: I fine-tuned `Llama-3.1-8B` using LoRA/PEFT. While it showed high reasoning capability, the compute cost too high for my deployment constraints compared to the encoder-only models.

### Advanced Techniques
#### Data Augmentation
To combat the limited size of the QEvasion dataset, I implemented:
- **Backtranslation**: Round-trip translation (English → Chinese/German → English) to generate paraphrased training examples.
- **Synthetic Generation**: Using GPT-5.1 to generate semantically diverse evasion examples, targeting the Clear Reply and Clear Non-Reply classes.

#### Loss Functions & Training
- **Focal Loss**: I implemented Focal Loss to penalize hard-to-classify examples and address the class imbalance in the 9-way evasion task.
- **NLI Formulation**: I experimented with casting the problem as a Natural Language Inference (NLI) task (e.g., *Premise: Question, Hypothesis: The speaker answers directly*), but standard classification heads proved more effective.

#### Ensemble Methods
- **Stacking**: I built a stacking ensemble combining predictions from RoBERTa, DeBERTa, and ModernBERT.
- **Feature Engineering**: I extracted linguistic features (sentiment, perplexity, lexical density) to feed into a meta-classifier.

## Experiments Results


Three classification tasks are evaluated:
1. **Direct Clarity Classification** (3 classes: Clear Reply, Ambivalent, Clear Non-Reply)
2. **Evasion-based Clarity Classification** (9→3 hierarchical mapping)
3. **Evasion Classification** (9 classes with multi-annotator evaluation)

| Model | Direct Clarity F1 | Evasion→Clarity F1 | Evasion F1 (9-class) |
| :--- | :---: | :---: | :---: |
| **Traditional ML (TF-IDF only)** | | | |
| Logistic Regression | 0.4997 | 0.5078 | 0.3010 |
| XGBoost | 0.4943 | 0.4691 | 0.2676 |
| **Traditional ML + Features** | | | |
| Logistic Regression + Features | 0.5916 | 0.5717 | 0.3321 |
| XGBoost + Features | 0.6360 | 0.6236 | 0.3852 |
| **Transformer Models (Fine-tuned)** | | | |
| ModernBERT-base | 0.5693 | 0.5035 | 0.3041 |
| ModernBERT-large | 0.6494 | 0.6064 | 0.3636 |
| Llama 3.1-8B (QLoRA) | 0.6277 | — | — |
| **Large Language Models (Prompting)** | | | |
| GPT-5 (Few-Shot CoT) | 0.7171 | 0.7109 | 0.4564 |

*> **Note**: Features include question length, answer length, a gpt3.5 summary of the question and answer pair and a gpt3.5 prediction (only available in the train dataset).*

### Augmentation + NLI Results

| Model | Dataset | Macro F1 |
| :--- | :--- | :---: |
| ModernBERT-Large | Original | 0.6479 |
| **ModernBERT-Large** | GPT-5.1 Augmented | 0.6789 |
| ModernBERT-Large | Backtranslation Augmented | 0.4926 |
| ModernBERT-Large + Focal Loss (γ=2.0) | GPT-5.1 Augmented | 0.6629 |
| ModernBERT-Large + Focal Loss (γ=3.0) | GPT-5.1 Augmented | 0.5960 |
| ModernBERT-Large + Focal Loss (γ=1.0) | GPT-5.1 Augmented | 0.6245 |
| ModernBERT-Large + Weighted CE | GPT-5.1 Augmented | 0.6755 |
| DeBERTa-v3-Large | Original | 0.5807 |
| DeBERTa-v3-Large | Backtranslation Augmented | 0.5823 |
| DeBERTa-v3-Large | GPT-5.1 Augmented | 0.5669 |
| Political_DEBATE (NLI) | Original | 0.5969 |
| Political_DEBATE (NLI) | GPT-5.1 Augmented | 0.5915 |
| ModernBERT-Large-NLI | Original | 0.6442 |

---


## Repository Structure
```
Clarity/
├── notebooks/
│   ├── final/                
│   │   ├── train_roberta_maxpool_multihead_kfold.ipynb
│   │   └── train_roberta_multihead.ipynb
│   └── experiments/          
│       ├── modernbert_*.ipynb
│       ├── deberta_*.ipynb
│       ├── llama3.1_*.ipynb
│       └── data_augmentation_*.ipynb
├── src/                       
│   ├── feature_extractor.py
│   └── process_dataset.py
└── data/                       
```
