# Mental health meme classification

## Dataset

### Depressive dataset

The dataset has 7 categories

```
Lack of Interest
Self-Harm
Feeling Down
Concentration Problem
Lack of Energy
Sleeping Disorder
Low Self-Esteem
Eating Disorder
```

### Anxiety dataset

The dataset has 8 categories

```
Nervousness
Lack of Worry Control
Excessive Worry
Restlessness
Difficulty Relaxing
Impending Doom
Irritatbily
Unknown
```

Ignore the below **Unknown** images in the anxiety dataset: `TR-2027, TR-2094, TR-212`

## Task

1. Model for Depression (Multi-Label Classification)

    - Objective: Predict one or more of the 7 depression categories for a given meme.

2. Model for Anxiety (Single-Label Classification)

    - Objective: Predict one of the 8 anxiety categories for a given meme.

## URLS

Overleaf document link: [Overleaf](https://www.overleaf.com/project/67df3968f379053f3ccaf55e)

Common document (docs) link: [Docs](https://docs.google.com/document/d/1gIpCO9sxX8YXN0CIDuMw5P5mokUMzSWILebsZNbfFf0/edit?usp=sharing)

Paper: [Figurative-cum-Commonsense Knowledge Infusion for Multimodal Mental Health Meme Classification](https://arxiv.org/abs/2501.15321)

## Baseline Implementation

### **Overview**

The baseline implementation focuses on leveraging **OCR text** extracted from memes and using **BERT embeddings** for classification. The models were trained and evaluated using the following setup:

### **Depression Task**

- **Type**: Multi-Label Classification
- **Model**: BERT (`bert-base-uncased`) with a classification head.
- **Input**: OCR text extracted from memes.
- **Loss Function**: Binary Cross-Entropy Loss (`BCEWithLogitsLoss`).
- **Metrics**:
  - Macro F1 Score
  - Weighted F1 Score
  - Hamming Loss
- **Training Details**:
  - **Optimizer**: AdamW
    - Learning Rate: `5e-5`
    - Beta1: `0.9`
    - Beta2: `0.999`
    - Epsilon: `1e-8`
    - Weight Decay: `1e-2`
  - **Scheduler**: Constant learning rate.
  - **Dropout**: `0.2`
  - **Batch Size**: `16`
  - **Max Sequence Length**: `512`
  - **Gradient Clipping**: `1.0`
  - **Epochs**: `10`
- **Output**:
  - Best model (`best_model.pt`) saved based on validation Macro F1 score.
  - Last model (`last_model.pt`) saved after the final epoch.
  - Training logs (`training_logs.csv`) with epoch-wise metrics.
  - Test evaluation results (`test_report.txt`).

### **Anxiety Task**

- **Type**: Single-Label Classification
- **Model**: BERT (`bert-base-uncased`) with a classification head.
- **Input**: OCR text extracted from memes.
- **Loss Function**: Cross-Entropy Loss.
- **Metrics**:
  - Accuracy
  - Macro F1 Score
  - Weighted F1 Score
  - Hamming Loss
- **Training Details**:
  - **Optimizer**: AdamW
    - Learning Rate: `5e-5`
    - Beta1: `0.9`
    - Beta2: `0.999`
    - Epsilon: `1e-8`
    - Weight Decay: `1e-2`
  - **Scheduler**: Constant learning rate.
  - **Dropout**: `0.2`
  - **Batch Size**: `16`
  - **Max Sequence Length**: `512`
  - **Gradient Clipping**: `1.0`
  - **Epochs**: `10`
- **Output**:
  - Best model (`best_model.pt`) saved based on validation Macro F1 score.
  - Last model (`last_model.pt`) saved after the final epoch.
  - Training logs (`training_logs.csv`) with epoch-wise metrics.
  - Test evaluation results (`test_report.txt`).

---

### **Implementation Details**

1. **OCR Text**:
    - The OCR text extracted from memes is used as the primary input for both tasks.
    - The text is tokenized using the BERT tokenizer (`bert-base-uncased`) with a maximum sequence length of `512`.

2. **BERT Embeddings**:
    - Pre-trained BERT embeddings (`bert-base-uncased`) are fine-tuned for both tasks.
    - A classification head is added on top of the BERT model:
      - For the depression task, the output layer has 7 neurons (one for each category) with a sigmoid activation function.
      - For the anxiety task, the output layer has 8 neurons (one for each category) with a softmax activation function.

3. **Training and Evaluation**:
    - Both tasks are trained for `10 epochs` using the AdamW optimizer and a constant learning rate scheduler.
    - Metrics such as Macro F1, Weighted F1, and Hamming Loss are calculated for evaluation.

4. **Outputs**:
    - Models (`best_model.pt` and `last_model.pt`) are saved in the respective output directories.
    - Training logs (`training_logs.csv`) and test evaluation results (`test_report.txt`) are saved for analysis.
