# s26projcais
# Email Phishing Detection using NLP

## Overview
Phishing emails pose a significant cybersecurity and societal threat, targeting individuals and organizations through deceptive messaging. This project uses Natural Language Processing (NLP) and deep learning to classify emails as phishing (spam) or legitimate.

## Dataset
**Source:** Kaggle phishing email datasets  
**Dataset Used:** `phishing_email.csv`  
**Size:** ~82,500 emails  
**Class Balance:** ~52% phishing, ~48% legitimate  

### Preprocessing
- Lowercasing text
- Removing URLs and email addresses
- Removing punctuation and special characters
- Stratified train/validation/test split (80/10/10)


## Model Development
We fine-tuned **DistilBERT**, a lightweight Transformer-based model pre-trained on large-scale English corpora.

### Why DistilBERT?
- Strong performance on text classification tasks, computationally efficient, and captures contextual meaning better than traditional models

### Training Details
- Max sequence length: 256
- Learning rate: 2e-5
- Batch size: 16
- Epochs: 3
- Optimizer: AdamW


## Evaluation Metrics
Given the cost of false negatives (missing phishing emails), we prioritize:
- Precision
- Recall
- F1-score


## Discussion & Social Implications
This model can be extended to:
- Email clients for real-time phishing detection, used in corporate security tools, or as an educational tool to raise phishing awareness

### Limitations
- Model performance depends on dataset quality, simple phishing styles may reduce accuracy, and ethical concerns around automated email monitoring

## Future Work
- Incorporate multiple datasets for improved generalization, explore multilingual phishing detection, deploy as a simple browser extension